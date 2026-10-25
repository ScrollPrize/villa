#include "vc/core/render/ChunkLoadQueue.hpp"
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace vc::render {
struct ChunkLoadQueue::Impl {
    struct Signal {
        struct Notice {bool all=false;std::vector<ChunkKey> keys;};
        std::mutex mutex;
        std::condition_variable cv;
        std::uint64_t generation=0;
        bool stopping=false;
        std::map<std::uint64_t,Notice> notices;
        void wake(){std::lock_guard lock(mutex);++generation;cv.notify_one();}
        void changed(std::uint64_t source,std::optional<ChunkKey> key){
            std::lock_guard lock(mutex);if(stopping)return;
            auto& notice=notices[source];
            if(!key){notice.all=true;notice.keys.clear();}
            else if(!notice.all){key->sourceId={};notice.keys.push_back(*key);}
            ++generation;cv.notify_one();
        }
    };
    struct Request {
        Source source;
        std::vector<ChunkKey> keys;
        bool foreground=false, retain=true;
        int priority=0;
        std::promise<std::vector<ChunkResult>> promise;
        std::vector<ChunkResult> results;
        std::size_t remaining=0, bytes=0;
    };
    struct Key {
        std::uint64_t source;
        ChunkKey chunk;
        bool operator==(const Key&) const = default;
    };
    struct Hash {std::size_t operator()(const Key& k) const {return ChunkKeyHash{}(k.chunk) ^ std::hash<std::uint64_t>{}(k.source);}};
    struct Entry {
        Key key;
        Query query;
        bool active=false, reserved=false, done=false;
        std::uint64_t revision=0;
        std::vector<std::pair<std::shared_ptr<Request>,std::size_t>> consumers;
    };
    struct Pending {
        bool foreground;
        int priority;
        std::size_t ordinal;
        std::uint64_t sequence, revision;
        std::shared_ptr<Entry> entry;
        bool operator<(const Pending& b) const {
            return std::tuple(!foreground,priority,ordinal,sequence)>std::tuple(!b.foreground,b.priority,b.ordinal,b.sequence);
        }
    };
    struct Registered {Source source;std::function<void()> unsubscribe;};
    const std::size_t limit;
    std::shared_ptr<Signal> signal=std::make_shared<Signal>();
    std::deque<std::shared_ptr<Request>> inbox; // protected by signal mutex
    struct Cancellation {std::uint64_t source;std::promise<void> done;};
    std::deque<std::shared_ptr<Cancellation>> cancellations; // signal mutex
    mutable std::mutex statsMutex;
    Stats counters;
    std::map<std::uint64_t,Registered> sources;
    std::unordered_map<Key,std::shared_ptr<Entry>,Hash> entries;
    std::unordered_set<Key,Hash> dirty;
    std::priority_queue<Pending> pending;
    std::size_t active=0;
    std::uint64_t nextSequence=0;
    std::thread worker;

    explicit Impl(std::size_t n):limit(n) {
        if(!n)throw std::invalid_argument("queued loader window must be positive");
        worker=std::thread([this]{run();});
    }
    void publishStats(){
        std::lock_guard lock(statsMutex);
        counters.inFlight=active;counters.pending=entries.size()-active;
        counters.peakInFlight=std::max(counters.peakInFlight,active);
        counters.peakPending=std::max(counters.peakPending,counters.pending);
    }
    void accept(std::shared_ptr<Request> request){
        if(!sources.contains(request->source.id)){
            std::weak_ptr<Signal> weak=signal;
            const auto id=request->source.id;
            auto unsubscribe=request->source.subscribeKeys
                ?request->source.subscribeKeys([weak,id](auto key){if(auto s=weak.lock())s->changed(id,key);})
                :request->source.subscribe([weak,id]{if(auto s=weak.lock())s->changed(id,std::nullopt);});
            sources.emplace(request->source.id,Registered{request->source,std::move(unsubscribe)});
        }
        request->remaining=request->keys.size();
        if(request->retain)request->results.resize(request->keys.size());
        if(request->keys.empty()){request->promise.set_value({});return;}
        const auto sequence=nextSequence++;
        for(std::size_t i=0;i<request->keys.size();++i){
            auto key=request->keys[i];key.sourceId={};
            Key combined{request->source.id,key};
            auto [it,inserted]=entries.try_emplace(combined);
            if(inserted){
                it->second=std::make_shared<Entry>();it->second->key=combined;
                it->second->query={key,request->foreground,request->priority};
                pending.push({request->foreground,request->priority,i,sequence,0,it->second});
            }else{
                std::lock_guard lock(statsMutex);++counters.coalesced;
                auto& e=*it->second;
                const bool promote=request->priority<e.query.priority||(request->foreground&&!e.query.foreground);
                e.query.foreground |= request->foreground;
                e.query.priority=std::min(e.query.priority,request->priority);
                if(promote){
                    if(!e.active)pending.push({e.query.foreground,e.query.priority,i,sequence,++e.revision,it->second});
                    else dirty.insert(combined);
                }
            }
            it->second->consumers.emplace_back(request,i);
        }
        request->keys.clear(); // consumers retain request; no second key copy
    }
    void finish(const std::shared_ptr<Entry>& entry,ChunkResult result){
        entry->done=true;--active;entries.erase(entry->key);
        for(auto& [request,index]:entry->consumers){
            if(request->retain){
                const auto bytes=result.bytes?result.bytes->size():0;
                request->results[index]=result;request->bytes+=bytes;
                std::lock_guard lock(statsMutex);counters.replyBytes+=bytes;
                counters.peakReplyBytes=std::max(counters.peakReplyBytes,counters.replyBytes);
            }
            if(!--request->remaining){
                {std::lock_guard lock(statsMutex);counters.replyBytes-=request->bytes;}
                request->promise.set_value(std::move(request->results));
            }
        }
        entry->consumers.clear();
        std::lock_guard lock(statsMutex);++counters.completed;
    }
    void run(){
        try {
            while(true){
                std::deque<std::shared_ptr<Request>> incoming;
                std::deque<std::shared_ptr<Cancellation>> cancelled;
                std::map<std::uint64_t,Signal::Notice> notices;
                std::uint64_t observed;
                {
                    std::lock_guard lock(signal->mutex);
                    if(signal->stopping)break;
                    observed=signal->generation;incoming.swap(inbox);cancelled.swap(cancellations);
                    notices.swap(signal->notices);
                }
                for(auto& request:incoming)accept(std::move(request));
                for(auto& cancel:cancelled){cancelSource(cancel->source);cancel->done.set_value();}
                // Inspect only completions/promotions/new admissions. A source
                // invalidation (or legacy notification) deliberately rescans
                // that source; ordinary keyed completions do not rescan the
                // full window or the potentially much larger pending backlog.
                for(const auto& [id,notice]:notices){
                    if(notice.all){for(const auto& [key,e]:entries)if(key.source==id&&e->active)dirty.insert(key);}
                    else for(const auto& key:notice.keys)dirty.insert({id,key});
                }
                std::map<std::uint64_t,std::vector<std::shared_ptr<Entry>>> batches;
                while(active<limit && !pending.empty()){
                    auto item=pending.top();pending.pop();auto& e=*item.entry;
                    if(e.done||e.active||e.revision!=item.revision)continue;
                    e.active=true;++active;
                    dirty.insert(e.key);
                }
                for(const auto& key:dirty){auto it=entries.find(key);if(it!=entries.end()&&it->second->active)batches[key.source].push_back(it->second);}
                dirty.clear();
                publishStats();bool completedAny=false;
                for(auto& [id,batch]:batches){
                    std::vector<Query> queries;queries.reserve(batch.size());
                    for(const auto& e:batch)queries.push_back(e->query);
                    auto& source=sources.at(id).source;
                    std::vector<ChunkKey> newlyReserved;
                    for(const auto& e:batch)if(!e->reserved)newlyReserved.push_back(e->key.chunk);
                    if(!newlyReserved.empty()){
                        source.acquire(newlyReserved);
                        for(const auto& e:batch)e->reserved=true;
                    }
                    auto results=source.poll(queries);
                    if(results.size()!=batch.size())throw std::logic_error("queued source result count differs");
                    std::vector<ChunkKey> released;
                    for(std::size_t i=0;i<batch.size();++i){
                        if(results[i].status==ChunkStatus::MissQueued)continue;
                        released.push_back(batch[i]->key.chunk);
                        finish(batch[i],std::move(results[i]));completedAny=true;
                    }
                    if(!released.empty())source.release(released);
                    std::lock_guard lock(statsMutex);++counters.polls;counters.queriedKeys+=queries.size();
                }
                publishStats();
                if(completedAny && !pending.empty())continue;
                std::unique_lock lock(signal->mutex);
                signal->cv.wait(lock,[&]{return signal->stopping||signal->generation!=observed||!inbox.empty()||!cancellations.empty();});
                {std::lock_guard statsLock(statsMutex);++counters.wakes;}
            }
            cleanup(std::make_exception_ptr(std::runtime_error("chunk load queue stopped")));
        }catch(...){
            {std::lock_guard lock(signal->mutex);signal->stopping=true;}
            cleanup(std::current_exception());
        }
    }
    void cancelSource(std::uint64_t id){
        std::vector<ChunkKey> release;
        std::unordered_map<Request*,std::shared_ptr<Request>> requests;
        for(auto it=entries.begin();it!=entries.end();){
            auto& entry=*it->second;
            if(entry.key.source!=id){++it;continue;}
            if(entry.reserved)release.push_back(entry.key.chunk);
            if(entry.active)--active;
            for(auto& [request,index]:entry.consumers)requests.emplace(request.get(),request);
            entry.done=true;entry.consumers.clear();it=entries.erase(it);
        }
        if(!release.empty())sources.at(id).source.release(release);
        for(auto& [ptr,request]:requests){
            {std::lock_guard lock(statsMutex);counters.replyBytes-=request->bytes;}
            request->promise.set_exception(std::make_exception_ptr(std::runtime_error("chunk source load cancelled")));
        }
        // Keep the cheap subscription for future requests; no service handles.
        publishStats();
    }
    void cleanup(std::exception_ptr error){
        std::unordered_map<Request*,std::shared_ptr<Request>> requests;
        std::map<std::uint64_t,std::vector<ChunkKey>> release;
        for(auto& [key,e]:entries){
            if(e->reserved)release[key.source].push_back(key.chunk);
            for(auto& [r,index]:e->consumers)requests.emplace(r.get(),r);
        }
        {std::lock_guard lock(signal->mutex);for(auto& r:inbox)requests.emplace(r.get(),r);inbox.clear();
            for(auto& c:cancellations)c->done.set_exception(error);cancellations.clear();}
        // A faulty source callback must not strand other request futures or
        // throw out of this worker's exception handler.
        for(auto& [id,keys]:release)try{sources.at(id).source.release(keys);}catch(...){}
        for(auto& [id,registered]:sources)try{registered.unsubscribe();}catch(...){}
        for(auto& [ptr,r]:requests)try{r->promise.set_exception(error);}catch(...){}
        entries.clear();sources.clear();pending={};dirty.clear();active=0;
        {std::lock_guard lock(statsMutex);counters.replyBytes=0;}
        publishStats();
    }
};

ChunkLoadQueue::ChunkLoadQueue(std::size_t n):impl_(std::make_unique<Impl>(n)){}
ChunkLoadQueue::~ChunkLoadQueue(){stop();}
void ChunkLoadQueue::stop(){
    {std::lock_guard lock(impl_->signal->mutex);impl_->signal->stopping=true;++impl_->signal->generation;}
    impl_->signal->cv.notify_one();
    if(impl_->worker.joinable())impl_->worker.join();
}
std::future<std::vector<ChunkResult>> ChunkLoadQueue::submit(Source source,
    std::vector<ChunkKey> keys,bool foreground,int priority,bool retain){
    if(!source.id||!source.acquire||!source.poll||!source.release||(!source.subscribe&&!source.subscribeKeys))
        throw std::invalid_argument("queued loader requires a complete source adapter");
    auto r=std::make_shared<Impl::Request>();r->source=std::move(source);r->keys=std::move(keys);
    r->foreground=foreground;r->priority=priority;r->retain=retain;
    auto future=r->promise.get_future();
    {std::lock_guard lock(impl_->signal->mutex);
        if(impl_->signal->stopping)throw std::runtime_error("chunk load queue is stopped");
        {std::lock_guard statsLock(impl_->statsMutex);++impl_->counters.requests;impl_->counters.keys+=r->keys.size();}
        impl_->inbox.push_back(std::move(r));++impl_->signal->generation;
    }
    impl_->signal->cv.notify_one();return future;
}
ChunkLoadQueue::Stats ChunkLoadQueue::stats() const {std::lock_guard lock(impl_->statsMutex);return impl_->counters;}
void ChunkLoadQueue::cancelSource(std::uint64_t id){
    auto c=std::make_shared<Impl::Cancellation>();c->source=id;auto done=c->done.get_future();
    {std::lock_guard lock(impl_->signal->mutex);
        if(impl_->signal->stopping)return;
        impl_->cancellations.push_back(std::move(c));++impl_->signal->generation;
    }
    impl_->signal->cv.notify_one();done.get();
}
}
