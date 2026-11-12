#include "vc_object_store/objects/FiberConstraintGraph.hpp"
#include <chrono>
#include <iostream>
int main(int argc,char** argv)try{
    if(argc!=3)throw std::invalid_argument("usage: vc_bench_fiber_graph_read STORE WORKERS");
    const auto workers=std::stoull(argv[2]);if(workers<1||workers>1024)throw std::invalid_argument("workers must be 1..1024");
    const auto start=std::chrono::steady_clock::now();
    const auto seconds=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
    auto store=vc::objects::ObjectCollection::open(argv[1],vc::objects::OpenOptions{.readOnly=true});
    auto txn=store.read();std::cout<<"opened seconds="<<seconds()<<std::endl;
    const auto rows=vc::objects::readCompactFiberConstraintGraph(store,txn,workers,[&](const char* phase,size_t count,size_t shard){
        std::cout<<phase<<" rows="<<count<<" shard="<<shard<<" seconds="<<seconds()<<std::endl;
    });
    std::cout<<"complete rows="<<rows.size()<<" record_bytes="<<sizeof(vc::objects::CompactFiberConstraint)
             <<" payload_bytes="<<rows.size()*sizeof(vc::objects::CompactFiberConstraint)<<" seconds="<<seconds()<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
