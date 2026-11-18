#include <vc_reader/native/thread_pool.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace std::chrono_literals;
int main(){try{
    const auto root=std::filesystem::temp_directory_path()/("vcr-scheduler-profile-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    struct Cleanup{std::filesystem::path path;~Cleanup(){std::error_code ec;std::filesystem::remove_all(path,ec);}} cleanup{root};
    const auto path=(root/"trace.jsonl").string();
    if(setenv("VC_READER_COORDINATION_TRACE",path.c_str(),1)||setenv("VC_READER_SCHEDULER_PROFILE","1",1))throw std::runtime_error("setenv failed");
    {
        vcr::diagnostics::SourceElapsed elapsed(vcr::diagnostics::SchedulerMetric::FileRead,42);
        std::this_thread::sleep_for(5ms);
    }
    const auto* timing=vcr::diagnostics::schedulerCounter(vcr::diagnostics::SchedulerMetric::FileRead);
    if(timing->count!=1 || timing->samples!=1 || timing->units!=42 || !timing->wait_ns)
        throw std::runtime_error("source syscall timing/count/bytes not recorded");
    vcr::ThreadPool pool(0);
    {
        vcr::diagnostics::SchedulerStageScope stage(vcr::diagnostics::SchedulerStage::Planning);
        pool.parallel_for(4,[](size_t){});
    }
    pool.parallel_for(0,[](size_t){});pool.parallel_for(1,[](size_t){});
    // A condition wait must not be counted as lock hold time.
    std::mutex mutex;std::condition_variable cv;bool ready=false;
    std::thread notifier([&]{std::this_thread::sleep_for(100ms);{std::lock_guard lock(mutex);ready=true;}cv.notify_one();});
    {vcr::diagnostics::SchedulerLock lock(mutex,vcr::diagnostics::SchedulerMetric::BatchLock);
        lock.wait(cv,[&]{return ready;},vcr::diagnostics::SchedulerMetric::BatchWait);}
    notifier.join();
    bool observed=false;
    for(int i=0;i<100&&!observed;++i){
        std::this_thread::sleep_for(20ms);std::ifstream input(path);std::string line;
        while(std::getline(input,line)){
            const auto row=nlohmann::json::parse(line,nullptr,false);if(row.is_discarded()||!row.contains("scheduler"))continue;
            const auto& p=row["scheduler"]["planning"];const auto& other=row["scheduler"]["other"];
            observed=p["calls"]["count"]==1&&p["calls"]["units"]==4&&p["caller_items"]["count"]==4&&
                p["size_2_4"]["count"]==1&&other["size_0"]["count"]==1&&other["size_1"]["count"]==1&&
                other["batch_wait"]["wait_ns"].get<uint64_t>()>50000000;
            if(observed&&other["batch_lock"]["hold_ns"].get<uint64_t>()>50000000)
                throw std::runtime_error("condition wait counted as lock hold");
        }
    }
    if(!observed)throw std::runtime_error("scheduler counter snapshot missing or incorrect");
    std::cout<<"Scheduler counts, stage propagation, histograms and condition-wait exclusion passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
