// Queue-only and scheduler-ticket contention. No volume I/O, scheduler or database.
#include <boost/lockfree/queue.hpp>
#include <nlohmann/json.hpp>
#ifdef VCR_BENCH_RIGTORP
#include <rigtorp/MPMCQueue.h>
#endif
#ifdef VCR_BENCH_MOODYCAMEL
#include <concurrentqueue.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <ctime>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock=std::chrono::steady_clock;
using Json=nlohmann::json;
using Value=std::uintptr_t;
template<class Queue>struct EmptyToken {explicit EmptyToken(Queue&) {}};
struct BoostQueue {
    using Producer=EmptyToken<BoostQueue>;using Consumer=Producer;
    boost::lockfree::queue<Value,boost::lockfree::fixed_sized<true>> queue;
    explicit BoostQueue(size_t capacity,size_t):queue(capacity){}
    bool push(Producer&,Value v){return queue.bounded_push(v);}
    bool pop(Consumer&,Value& v){return queue.pop(v);}
};
#ifdef VCR_BENCH_RIGTORP
struct RigtorpQueue {
    using Producer=EmptyToken<RigtorpQueue>;using Consumer=Producer;
    rigtorp::MPMCQueue<Value> queue;
    explicit RigtorpQueue(size_t capacity,size_t):queue(capacity){}
    bool push(Producer&,Value v){return queue.try_push(v);}
    bool pop(Consumer&,Value& v){return queue.try_pop(v);}
};
#endif
#ifdef VCR_BENCH_MOODYCAMEL
struct MoodyQueue {
    moodycamel::ConcurrentQueue<Value> queue;
    std::mutex preparation;
    struct Producer {
        moodycamel::ProducerToken token;
        explicit Producer(MoodyQueue& q):token(q.queue){
            // Explicit producers retain their blocks. Seed a reusable two-block
            // ring for every producer before the timed barrier so a greedy first
            // producer cannot monopolize the initial pool and starve the others.
            std::lock_guard lock(q.preparation);
            for(size_t i=0;i<64;++i)if(!q.queue.enqueue(token,Value{}))throw std::bad_alloc();
            Value value;
            for(size_t i=0;i<64;++i)if(!q.queue.try_dequeue(value))throw std::runtime_error("producer preparation drain failed");
        }
    };
    struct Consumer {moodycamel::ConsumerToken token;explicit Consumer(MoodyQueue& q):token(q.queue){}};
    static size_t slots(size_t capacity,size_t producers){return (capacity+64*producers+31)/32*32;}
    explicit MoodyQueue(size_t capacity,size_t producers):queue(slots(capacity,producers)){}
    bool push(Producer& p,Value v){return queue.try_enqueue(p.token,v);}
    bool pop(Consumer& c,Value& v){return queue.try_dequeue(c.token,v);}
};
#endif

struct Histogram {
    std::array<uint64_t,64> bins{};uint64_t count=0,total=0;
    void add(uint64_t ns){++bins[std::min<unsigned>(63,std::bit_width(ns))];++count;total+=ns;}
    void merge(const Histogram& h){for(size_t i=0;i<64;++i)bins[i]+=h.bins[i];count+=h.count;total+=h.total;}
    Json json()const {
        const auto percentile=[&](double p)->Json {if(!count)return nullptr;uint64_t sum=0;
            for(size_t i=0;i<64;++i){sum+=bins[i];if(double(sum)>=double(count)*p)return double(uint64_t(1)<<i)/1000.;}return nullptr;};
        return {{"samples",count},{"mean_us",count?Json(double(total)/count/1000.):Json(nullptr)},
            {"p50_upper_us",percentile(.5)},{"p95_upper_us",percentile(.95)}};
    }
};
struct alignas(64) Stats {
    uint64_t attempts=0,failures=0,success=0,sum=0,xorValue=0,errors=0,stale=0,slotBusy=0;
    Histogram operation,handoff;
};
struct alignas(64) Ticket {
    std::atomic<unsigned> refs{0};std::atomic<bool> done{false};uint64_t id=0,stamp=0;
};
static uint64_t nanoseconds(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
static void backoff(uint64_t misses){if((misses&63)==0)std::this_thread::yield();}
static Json rates(std::vector<double> values){std::sort(values.begin(),values.end());double sum=0;for(double v:values)sum+=v;
    return {{"mean",sum/values.size()},{"median",values[values.size()/2]},{"min",values.front()},{"max",values.back()}};}

// Exact-once regression is separate from throughput: the hot benchmark uses
// worker-local counts/sum/xor rather than a globally contended validation bitmap.
template<class Queue>static void check() {
    constexpr size_t producers=8,consumers=8,messages=4096;
    Queue queue(64,producers);auto seen=std::make_unique<std::atomic<unsigned>[]>(producers*messages);
    std::atomic<unsigned> complete=0,errors=0;std::vector<std::thread> threads;
    std::barrier prepared(ptrdiff_t(producers+consumers+1));
    for(size_t p=0;p<producers;++p)threads.emplace_back([&,p]{typename Queue::Producer token(queue);
        prepared.arrive_and_wait();
        for(size_t i=0;i<messages;++i){Value v=p*messages+i;while(!queue.push(token,v))std::this_thread::yield();}
        complete.fetch_add(1,std::memory_order_acq_rel);
    });
    for(size_t c=0;c<consumers;++c)threads.emplace_back([&]{typename Queue::Consumer token(queue);Value v;bool finalDrain=false;
        prepared.arrive_and_wait();
        for(;;){if(queue.pop(token,v)){if(v>=producers*messages||seen[v].fetch_add(1)!=0)++errors;}
            else if(finalDrain)break;
            else if(complete.load(std::memory_order_acquire)==producers)finalDrain=true;
            else std::this_thread::yield();}
    });
    prepared.arrive_and_wait();for(auto& t:threads)t.join();for(size_t i=0;i<producers*messages;++i)if(seen[i]!=1)++errors;
    if(errors)throw std::runtime_error("exact-once queue regression failed");
    std::cout<<Json{{"exact_once",true},{"messages",producers*messages}}.dump()<<'\n';
}

template<class Queue>static void benchmark(const std::string& backend,const std::string& mode,
    size_t producers,size_t consumers,size_t capacity,double seconds) {
    constexpr size_t perProducer=64;
    const bool tickets=mode=="tickets";
    Queue queue(capacity,producers);
    auto descriptors=std::make_unique<Ticket[]>(tickets?producers*perProducer:0);
    std::vector<Stats> stats(producers+consumers);
    std::vector<std::exception_ptr> exceptions(producers+consumers);
    std::vector<std::thread> threads;std::atomic<bool> stop=false,abort=false;
    std::atomic<size_t> complete{0};std::promise<void> gate;auto open=gate.get_future().share();
    std::barrier ready(ptrdiff_t(producers+consumers+1)),finished(ptrdiff_t(producers+consumers+1));
    try {for(size_t t=0;t<producers+consumers;++t)threads.emplace_back([&,t]{
        std::optional<typename Queue::Producer> producer;
        std::optional<typename Queue::Consumer> consumer;
        try {if(t<producers)producer.emplace(queue);else consumer.emplace(queue);}catch(...){exceptions[t]=std::current_exception();}
        open.wait();if(abort)return;
        uint64_t sequence=0;size_t cursor=0;
        for(size_t window=0;window<4;++window){
            ready.arrive_and_wait();auto& s=stats[t];s={};
            if(t<producers){
                while(!exceptions[t]&&!stop.load(std::memory_order_relaxed)){
                    Ticket* ticket=nullptr;
                    if(tickets){ticket=&descriptors[t*perProducer+cursor];
                        if(ticket->refs.load(std::memory_order_acquire)!=0){backoff(++s.slotBusy);continue;}}
                    const uint64_t id=(uint64_t(t)<<48)|(++sequence);
                    Value value=Value(id);
                    if(tickets){ticket->id=id;ticket->done=false;ticket->stamp=(sequence&1023)==0?nanoseconds():0;
                        ticket->refs.store(2,std::memory_order_relaxed);value=reinterpret_cast<Value>(ticket);}
                    const bool sample=(++s.attempts&1023)==0;const auto before=sample?nanoseconds():0;
                    const bool published=queue.push(*producer,value);
                    if(sample)s.operation.add(nanoseconds()-before);
                    if(tickets){ticket->done.store(true,std::memory_order_release);
                        if(published){ticket->refs.fetch_sub(1,std::memory_order_acq_rel);cursor=(cursor+1)%perProducer;}
                        else ticket->refs.store(0,std::memory_order_release);}
                    if(published){++s.success;s.sum+=id;s.xorValue^=id;}
                    else backoff(++s.failures);
                }
                complete.fetch_add(1,std::memory_order_acq_rel);
            } else if(!exceptions[t]) {
                bool finalDrain=false;
                for(;;){Value value;const bool sample=(++s.attempts&1023)==0;const auto before=sample?nanoseconds():0;
                    const bool found=queue.pop(*consumer,value);
                    if(sample)s.operation.add(nanoseconds()-before);
                    if(!found){++s.failures;
                        // The final enqueue can occur between an empty poll and
                        // observing producer completion. Recheck after acquiring
                        // that completion before deciding the queue is drained.
                        if(finalDrain)break;
                        if(complete.load(std::memory_order_acquire)==producers)finalDrain=true;
                        backoff(s.failures);continue;}
                    uint64_t id=value;
                    if(tickets){const auto base=reinterpret_cast<Value>(descriptors.get());
                        if(value<base||value-base>=producers*perProducer*sizeof(Ticket)||(value-base)%sizeof(Ticket)) {++s.errors;continue;}
                        auto* ticket=reinterpret_cast<Ticket*>(value);id=ticket->id;
                        s.stale+=ticket->done.load(std::memory_order_acquire);
                        if(ticket->stamp)s.handoff.add(nanoseconds()-ticket->stamp);
                        const auto refs=ticket->refs.fetch_sub(1,std::memory_order_acq_rel);if(refs<1||refs>2)++s.errors;
                    }
                    if((id>>48)>=producers)++s.errors;
                    ++s.success;s.sum+=id;s.xorValue^=id;
                }
            }
            finished.arrive_and_wait();
        }
    });}catch(...){abort=true;gate.set_value();for(auto& t:threads)t.join();throw;}
    gate.set_value();Json windows=Json::array();std::vector<double> throughput;uint64_t allErrors=0;
    for(size_t window=0;window<4;++window){
        stop=false;complete=0;ready.arrive_and_wait();const auto started=Clock::now();const auto cpu=std::clock();
        std::this_thread::sleep_for(std::chrono::duration<double>(window?seconds:.2));
        stop=true;const auto stopped=Clock::now();finished.arrive_and_wait();
        const auto ended=Clock::now();const double elapsed=std::chrono::duration<double>(ended-started).count();
        const double cpuSeconds=double(std::clock()-cpu)/CLOCKS_PER_SEC;
        uint64_t enqueued=0,dequeued=0,psum=0,csum=0,pxor=0,cxor=0,full=0,empty=0,stale=0,busy=0,errors=0;
        Histogram enqueueLatency,dequeueLatency,handoff;
        for(size_t t=0;t<stats.size();++t){const auto& s=stats[t];errors+=s.errors;
            if(t<producers){enqueued+=s.success;psum+=s.sum;pxor^=s.xorValue;full+=s.failures;busy+=s.slotBusy;enqueueLatency.merge(s.operation);}
            else{dequeued+=s.success;csum+=s.sum;cxor^=s.xorValue;empty+=s.failures;stale+=s.stale;dequeueLatency.merge(s.operation);handoff.merge(s.handoff);}}
        if(enqueued!=dequeued||psum!=csum||pxor!=cxor)++errors;
        if(tickets)for(size_t i=0;i<producers*perProducer;++i)if(descriptors[i].refs!=0)++errors;
        allErrors+=errors;
        if(window){throughput.push_back(dequeued/elapsed);windows.push_back({{"seconds",elapsed},
            {"post_stop_drain_seconds",std::chrono::duration<double>(ended-stopped).count()},
            {"transfers",dequeued},{"transfers_per_second",dequeued/elapsed},{"queue_operations_per_second",2.*dequeued/elapsed},
            {"enqueue_failures",full},{"empty_polls",empty},{"descriptor_wait_polls",busy},
            {"completed_tickets",stale},{"completed_fraction",dequeued?double(stale)/dequeued:0.},
            {"cpu_seconds",cpuSeconds},{"cpu_cores",cpuSeconds/elapsed},{"errors",errors},
            {"enqueue_attempt_latency",enqueueLatency.json()},{"dequeue_attempt_latency",dequeueLatency.json()},{"ticket_handoff_latency",handoff.json()}});}
    }
    for(auto& t:threads)t.join();for(auto& error:exceptions)if(error)std::rethrow_exception(error);
    std::cout<<Json{{"backend",backend},{"mode",mode},{"producers",producers},{"consumers",consumers},
        {"target_capacity",capacity},{"strict_capacity",backend!="moodycamel"},
        {"preallocated_message_slots",backend=="moodycamel"?(capacity+64*producers+31)/32*32:capacity},
        {"descriptor_slots",tickets?producers*perProducer:0},{"allocation_in_transfer_path",false},
        {"seconds_per_window",seconds},{"repetitions",3},{"hardware_threads",std::thread::hardware_concurrency()},
        {"transfers_per_second",rates(throughput)},{"errors",allErrors},{"windows",windows}}.dump()<<'\n';
    if(allErrors)throw std::runtime_error("transfer/checksum/descriptor validation failed");
}
int main(int argc,char** argv){try{
    const std::string backend=argc>1?argv[1]:"boost",mode=argc>2?argv[2]:"transfer";
    const size_t producers=argc>3?std::stoul(argv[3]):128,consumers=argc>4?std::stoul(argv[4]):16,capacity=argc>5?std::stoul(argv[5]):4096;
    const double seconds=argc>6?std::stod(argv[6]):.5;
    if(!producers||producers>1920||!consumers||consumers>192||!capacity||capacity>65533||seconds<=0||seconds>3||
       (mode!="check"&&mode!="transfer"&&mode!="tickets"))throw std::invalid_argument("invalid queue benchmark arguments");
    const auto run=[&]<class Queue>(){if(mode=="check")check<Queue>();else benchmark<Queue>(backend,mode,producers,consumers,capacity,seconds);};
    if(backend=="boost")run.template operator()<BoostQueue>();
#ifdef VCR_BENCH_RIGTORP
    else if(backend=="rigtorp")run.template operator()<RigtorpQueue>();
#endif
#ifdef VCR_BENCH_MOODYCAMEL
    else if(backend=="moodycamel")run.template operator()<MoodyQueue>();
#endif
    else throw std::invalid_argument("backend not built: "+backend);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
