#include <vc_reader/native/thread_pool.hpp>
#include <vc_reader/native/single_flight.hpp>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
using namespace std::chrono_literals;
int main(){try{
    const auto root=std::filesystem::temp_directory_path()/("vcr-coordination-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    struct Cleanup{std::filesystem::path path;~Cleanup(){std::error_code e;std::filesystem::remove_all(path,e);}} cleanup{root};
    const auto path=(root/"trace.jsonl").string();
    if(setenv("VC_READER_COORDINATION_TRACE",path.c_str(),1))throw std::runtime_error("setenv failed");
    vcr::ThreadPool pool(2);vcr::SingleFlight<int> flights;
    auto owner=flights.acquire("held");std::atomic<int> errors=0;
    auto caller=std::async(std::launch::async,[&]{vcr::diagnostics::Scope batch(vcr::diagnostics::ChunkBatch);
        pool.parallel_for(4,[&](auto){if(flights.acquire("held").resolve([]{return 9;})!=9)++errors;});});
    bool observed=false;
    const auto deadline=std::chrono::steady_clock::now()+2s;
    while(std::chrono::steady_clock::now()<deadline&&!observed){
        std::this_thread::sleep_for(20ms);std::ifstream input(path);std::string line;
        while(std::getline(input,line)){
            const auto row=nlohmann::json::parse(line,nullptr,false);if(row.is_discarded())continue;
            if(row.at("chunk_waiters")>=3&&row.at("helper_chunk_waiters")==2&&row.at("chunk_ready")>=1)observed=true;
        }
    }
    owner.resolve([]{return 9;});caller.get();
    if(!observed||errors)throw std::runtime_error("did not observe queued work with blocked caller and helpers");
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
