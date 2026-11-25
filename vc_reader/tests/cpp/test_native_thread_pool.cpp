#include <vc_reader/native/thread_pool.hpp>
#include <future>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <set>

using namespace std::chrono_literals;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
    for(size_t helpers:{0,1,4,16}) {
        vcr::ThreadPool pool(helpers);
        for(size_t n:{0,1,2,3,4,17,257,4099}){
            auto visits=std::make_unique<std::atomic<int>[]>(n);
            pool.parallel_for(n,[&](size_t i){++visits[i];});
            for(size_t i=0;i<n;++i)require(visits[i]==1,"visit each item exactly once");
        }
        std::vector<std::future<void>> callers;
        for(size_t caller=0;caller<32;++caller)callers.push_back(std::async(std::launch::async,[&]{
            for(size_t repeat=0;repeat<15;++repeat){
                std::vector<size_t> values(257);
                pool.parallel_for(values.size(),[&](size_t i){
                    // Nested same-pool jobs cannot rely on idle helpers to progress.
                    size_t nested[3]{};pool.parallel_for(3,[&](size_t j){nested[j]=i+j+1;});
                    values[i]=nested[0]+nested[1]+nested[2];
                });
                for(size_t i=0;i<values.size();++i)require(values[i]==3*i+6,"nested result visible on return");
            }
        }));
        for(auto& caller:callers){require(caller.wait_for(15s)==std::future_status::ready,"concurrent nested progress");caller.get();}
        for(size_t repeat=0;repeat<100;++repeat){
            bool caught=false;try{pool.parallel_for(257,[](size_t i){if(i==0)throw std::runtime_error("expected");});}
            catch(const std::runtime_error&){caught=true;}
            require(caught,"propagate exception");std::atomic<int> count=0;
            pool.parallel_for(257,[&](size_t){++count;});require(count==257,"reuse after exception");
        }
        // Sequential reuse exercises helper sleep/wakeup and final completion races.
        for(size_t repeat=0;repeat<1000;++repeat){
            size_t values[16]{};pool.parallel_for(16,[&](size_t i){values[i]=i+1;});
            for(size_t i=0;i<16;++i)require(values[i]==i+1,"completed before returning");
        }
    }
    vcr::ThreadPool pool(2);std::atomic<bool> started=false;
    auto longJob=std::async(std::launch::async,[&]{pool.parallel_for(2000,[&](size_t){started=true;std::this_thread::sleep_for(100us);});});
    while(!started)std::this_thread::yield();
    auto caller=std::this_thread::get_id();std::atomic<int> helped=0;
    pool.parallel_for(128,[&](size_t){helped+=std::this_thread::get_id()!=caller;std::this_thread::sleep_for(100us);});
    longJob.get();require(helped>0,"helpers share concurrent jobs");
    {
        vcr::ThreadPool parallel(4);std::mutex mutex;std::set<std::thread::id> participants;
        parallel.parallel_for(128,[&](size_t){
            {std::lock_guard lock(mutex);participants.insert(std::this_thread::get_id());}
            std::this_thread::sleep_for(100us);
        });
        require(participants.size()>1,"one large request recruits helpers");
    }
    // Minimal preallocation stresses failed publications, including nested calls.
    // The queue rounds storage to blocks rather than enforcing a ticket count.
    // Completed descriptors may outlive the temporary callback; helpers must
    // never dereference it after parallel_for returns.
    for(int repetition=0;repetition<8;++repetition){
        vcr::ThreadPool small(4,1);std::vector<std::future<void>> callers;
        for(int c=0;c<32;++c)callers.push_back(std::async(std::launch::async,[&]{
            for(int i=0;i<100;++i){
                size_t output[4]{};
                small.parallel_for(4,[&](size_t j){size_t nested[2]{};
                    small.parallel_for(2,[&](size_t k){nested[k]=j+k;});output[j]=nested[0]+nested[1];});
                for(size_t j=0;j<4;++j)require(output[j]==2*j+1,"bounded queue nesting/lifetime");
            }
        }));
        for(auto& caller:callers){require(caller.wait_for(15s)==std::future_status::ready,"queue saturation cannot block callers");caller.get();}
    }
    {
        vcr::ThreadPool single(1,2);std::promise<void> entered,release;
        std::atomic<bool> announced=false;
        auto resume=release.get_future().share();
        auto occupied=std::async(std::launch::async,[&]{const auto owner=std::this_thread::get_id();
            single.parallel_for(128,[&](size_t){
                if(std::this_thread::get_id()!=owner){if(!announced.exchange(true)){entered.set_value();resume.wait();}}
                else std::this_thread::sleep_for(1ms);
            });
        });
        const bool helping=entered.get_future().wait_for(2s)==std::future_status::ready;
        bool valid=true;
        for(int i=0;i<2000;++i){size_t output[2]{};
            single.parallel_for(2,[&](size_t j){output[j]=j+1;});valid&=output[0]==1&&output[1]==2;}
        release.set_value();occupied.get();
        require(helping&&valid,"stale queue tickets cannot access completed temporary callbacks");
    }
    std::cout<<"Thread-pool exact completion, concurrency, nesting, fairness, wakeup and exception tests passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
