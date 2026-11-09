#include "TestSupport.hpp"
#include "../../apps/support/ShardQueue.hpp"
#include <future>
#include <iostream>

using namespace vc::objects::imports;
using namespace std::chrono_literals;
namespace {
void backpressureAndClose(){
    StreamFailure failure;ShardQueues<int> queue(1,1,failure);
    queue.push(0,{1},20); // One oversize batch still makes progress.
    std::promise<void> started;
    auto writer=std::async(std::launch::async,[&]{started.set_value();queue.push(0,{2},1);});
    started.get_future().wait();
    require(writer.wait_for(60ms)==std::future_status::timeout,"queue ignored backpressure");
    std::vector<int> rows;
    require(queue.pop(0,rows,[]{})&&rows==std::vector<int>{1},"queue did not preserve first batch");
    writer.get();queue.close();
    require(queue.pop(0,rows,[]{})&&rows==std::vector<int>{2},"close dropped a queued batch");
    require(!queue.pop(0,rows,[]{}),"closed drained queue did not finish");
}
void cancellation(){
    for(const bool producer:{false,true}){
        StreamFailure failure;ShardQueues<int> queue(1,1,failure);
        if(producer)queue.push(0,{1},1);
        std::promise<void> started;
        auto blocked=std::async(std::launch::async,[&]{
            started.set_value();
            if(producer)queue.push(0,{2},1);
            else {std::vector<int> rows;queue.pop(0,rows,[]{});}
        });
        started.get_future().wait();
        require(blocked.wait_for(60ms)==std::future_status::timeout,"queue did not wait before cancellation");
        failure.fail(std::make_exception_ptr(std::runtime_error("peer failed")));
        require(blocked.wait_for(2s)==std::future_status::ready,"peer failure left queue blocked");
        mustFail([&]{blocked.get();});
    }
    StreamFailure failure;ShardQueues<int> queue(1,1,failure);std::vector<int> rows;
    mustFail([&]{queue.pop(0,rows,[]{throw std::runtime_error("builder cancelled");});});
}
}
int main()try{backpressureAndClose();cancellation();std::cout<<"shard queue passed\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
