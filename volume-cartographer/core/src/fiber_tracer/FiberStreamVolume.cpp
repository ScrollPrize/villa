#include "vc/fiber_tracer/FiberStreamVolume.hpp"
#include "vc/fiber_tracer/FiberVolumeCoverage.hpp"
#include "vc/fiber_tracer/detail/FiberChunkTensor.hpp"
#include "vc/fiber_tracer/detail/FiberPackedStorage.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <chrono>
#include <ctime>

namespace vc::fiber_tracer {
namespace {
using Json=nlohmann::json;
using Vec=cv::Vec3d;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point begin){return std::chrono::duration<double>(Clock::now()-begin).count();}
Vec unit(Vec v){return v/std::max(cv::norm(v),1e-9);}
Json vectorJson(const Vec& p){return {p[0],p[1],p[2]};}
Vec vectorOf(const Json& j){return {j[0].get<double>(),j[1].get<double>(),j[2].get<double>()};}
std::string tileName(const std::array<int,3>& k){return std::to_string(k[0])+"_"+std::to_string(k[1])+"_"+std::to_string(k[2]);}

// One OS pool for all computational jobs. No task waits for another task.
// A cache miss registers a continuation and releases its worker immediately.
class WorkQueue {
    std::mutex mutex;
    std::condition_variable ready;
    std::array<std::deque<std::function<void()>>,4> queues;
    std::vector<std::thread> threads;
    std::exception_ptr error;
    bool stopping=false;
    std::atomic_bool& cancel;
public:
    std::array<std::atomic<double>,4> workSeconds{};
    std::array<std::atomic<uint64_t>,4> jobs{};
    std::atomic<int> busy{0};
    explicit WorkQueue(size_t count,std::atomic_bool& stop):cancel(stop) {
        for(size_t i=0;i<count;++i)threads.emplace_back([this]{
            while(true) {
                std::function<void()> job;size_t tier=0;
                {
                    std::unique_lock lock(mutex);
                    ready.wait(lock,[&]{return stopping||std::any_of(queues.begin(),queues.end(),[](auto&q){return !q.empty();});});
                    if(stopping)return;
                    for(;tier<queues.size()&&queues[tier].empty();++tier){}
                    job=std::move(queues[tier].front());queues[tier].pop_front();
                }
                if(cancel.load())continue;
                ++busy;const auto begin=Clock::now();
                try{job();}catch(...){std::lock_guard lock(mutex);if(!error)error=std::current_exception();cancel=true;}
                workSeconds[tier].fetch_add(seconds(begin));++jobs[tier];--busy;
            }
        });
    }
    ~WorkQueue(){stop();}
    void push(size_t tier,std::function<void()> job) {
        {std::lock_guard lock(mutex);if(stopping||cancel.load())return;queues.at(tier).push_back(std::move(job));}
        ready.notify_one();
    }
    void rethrow(){std::lock_guard lock(mutex);if(error)std::rethrow_exception(error);}
    void stop() {
        {std::lock_guard lock(mutex);stopping=true;for(auto&q:queues)q.clear();}
        ready.notify_all();for(auto&t:threads)if(t.joinable())t.join();
    }
};

class CTChecks {
    // Cache notifications can already have been copied by a publisher when
    // removeChunkReadyListener runs. Keep their lifetime separate from ours.
    struct ListenerGate {std::mutex mutex; CTChecks* owner=nullptr;};
    std::shared_ptr<ListenerGate> listenerGate=std::make_shared<ListenerGate>();
    std::shared_ptr<Volume> volume;
    vc::render::IChunkedArray* cache;
    int level;double baseScale;
    WorkQueue& pool;
    std::mutex mutex;
    std::unordered_map<vc::render::ChunkKey,std::vector<std::function<void()>>,vc::render::ChunkKeyHash> waiting;
    std::atomic_bool wakeQueued{false};
    vc::render::IChunkedArray::ChunkReadyCallbackId listener;
    void wake() {
        if(wakeQueued.exchange(true))return;
        pool.push(0,[this]{
            std::vector<std::function<void()>> ready;
            {
                std::lock_guard lock(mutex);wakeQueued=false;
                for(auto i=waiting.begin();i!=waiting.end();) {
                    // A notification is not a lease: pressure can evict the
                    // chunk before this queued wake runs. Re-request it rather
                    // than waiting forever on a fetch that already completed.
                    const auto& k=i->first;auto r=cache->tryGetChunk(k.level,k.iz,k.iy,k.ix);
                    if(r.status==vc::render::ChunkStatus::MissQueued){++i;continue;}
                    for(auto& f:i->second)ready.push_back(std::move(f));i=waiting.erase(i);
                }
            }
            for(auto& f:ready)pool.push(0,std::move(f));
        });
    }
public:
    std::atomic<uint64_t> suspensions{0};
    CTChecks(const std::string& path,int requestedLevel,WorkQueue& q):
        volume(path.find("://")==std::string::npos ? Volume::New(path) : Volume::NewFromUrl(path)),
        cache(volume->chunkedCache()),level(requestedLevel-volume->baseScaleLevel()),
        baseScale(std::pow(2.,volume->baseScaleLevel())),pool(q) {
        if(level<0||level>=cache->numLevels())throw std::invalid_argument("CT level unavailable");
        listenerGate->owner=this;
        listener=cache->addChunkReadyListener([gate=listenerGate]{
            std::lock_guard lock(gate->mutex);
            if(gate->owner)gate->owner->wake();
        });
    }
    ~CTChecks(){
        {std::lock_guard lock(listenerGate->mutex);listenerGate->owner=nullptr;}
        cache->removeChunkReadyListener(listener);
    }
    bool sample(const std::vector<Vec>& points,std::vector<uint8_t>& outside,std::function<void()> retry) {
        using Sampler=vc::render::ChunkedPlaneSampler;
        cv::Mat_<cv::Vec3f> coords(1,int(points.size()));
        for(size_t i=0;i<points.size();++i)coords(0,int(i))=cv::Vec3f(points[i]/baseScale);
        cv::Mat_<uint8_t> coverage(1,int(points.size()),uint8_t{0}),values(1,int(points.size()),uint8_t{0});
        Sampler::Options opts(vc::Sampling::Nearest,64);opts.parallelThreads=1;
        const auto keys=Sampler::collectCoordsDependencies(*cache,level,coords,coverage,opts);
        std::vector<vc::render::ChunkResult> pinned;bool pending=false;vc::render::ChunkKey missing;
        for(const auto& k:keys) {
            auto r=cache->tryGetChunk(k.level,k.iz,k.iy,k.ix);
            if(r.status==vc::render::ChunkStatus::Error)throw std::runtime_error(r.error);
            if(r.status==vc::render::ChunkStatus::MissQueued){if(!pending)missing=k;pending=true;}
            pinned.push_back(std::move(r));
        }
        if(pending) {
            {std::lock_guard lock(mutex);waiting[missing].push_back(std::move(retry));}
            ++suspensions;wake(); // close the ready-before-registration race
            return false;
        }
        Sampler::sampleCoordsLevelBlockingRequestedLevel(*cache,level,coords,values,coverage,opts);
        outside.resize(points.size());
        for(size_t i=0;i<points.size();++i)outside[i]=values(0,int(i))==0;
        return true;
    }
};

struct Tile;
struct Pair;
struct Direction {
    Pair* pair=nullptr;
    int half=0,streak=0,outsideStreak=0;
    Vec direction,previous;
    std::unique_ptr<FiberTraceCursor> cursor;
    std::deque<FiberTraceCommit> pending;
    std::vector<Vec> points;
    std::vector<double> widths,losses;
    std::vector<uint8_t> outside;
    bool traceDone=false;
    std::string reason;
    double widthMean=0;uint64_t widthCount=1;
};
struct Tile {
    std::array<int,3> key;
    std::vector<FiberChunkSeedFit> seeds;
    std::deque<size_t> pending;
    size_t remaining=0;
    bool scheduled=false;
    Json pieces=Json::array();
    std::mutex piecesMutex;
};
struct Pair {
    int64_t id;
    std::shared_ptr<Tile> tile;
    Vec seed,dir;double width;
    Direction half[2];
    std::atomic<int> unfinished{2};
    bool prepared=false,resampled=false;
    std::vector<uint8_t> combinedOutside;
    std::vector<std::pair<std::shared_ptr<Tile>,size_t>> waiters;
};

class Stream {
    const FiberPredictionSource& predictions;
    const vc::lasagna::NormalSampler& normals;
    FiberTraceConfig trace;
    FiberChunkTraceConfig chunks;
    Json options;
    const std::vector<std::array<int,3>>& keys;
    std::filesystem::path output;
    std::atomic_bool& cancel;
    double scale,grid,radius,spread,maximum,tolerance;
    int decisionSteps;
    Vec lo,hi;
    size_t workers,capacity,seedWorkers,maxOpen,cursor=0,loading=0;
    WorkQueue pool;
    CTChecks ct;
    FiberVolumeCoverage coverage;
    std::mutex admission,publish;
    std::deque<std::shared_ptr<Pair>> publishQueue;
    bool publishing=false;
    std::vector<std::shared_ptr<Tile>> open;
    std::map<int64_t,std::shared_ptr<Pair>> active;
    std::set<std::string> done;
    int64_t nextPair=0;
    std::atomic<int64_t> nextFiber{0};
    std::atomic<uint64_t> epoch{1},finishedTiles{0},pieces{0},startedPairs{0},finishedPairs{0},candidates{0},covered{0},checks{0};
    std::atomic<size_t> peakActive{0};
    std::atomic<double> kept{0};
    std::mutex reasonMutex;
    std::map<std::string,uint64_t> reasons;
    Clock::time_point began=Clock::now();
    std::clock_t cpuBegin=std::clock();

    void loadMore() { // admission held
        while(loading<seedWorkers && open.size()<maxOpen && cursor<keys.size()) {
            auto k=keys[cursor++];if(done.count(tileName(k)))continue;
            auto tile=std::make_shared<Tile>();tile->key=k;open.push_back(tile);++loading;
            pool.push(2,[this,tile]{
                FiberChunkPaths bounds;bounds.key=tile->key;
                for(int a=0;a<3;++a){bounds.lo[a]=tile->key[a]*chunks.coreBase-chunks.haloBase;
                    bounds.hi[a]=(tile->key[a]+1)*chunks.coreBase+chunks.haloBase;}
                const auto* field=dynamic_cast<const FiberPredictionField*>(&predictions);
                const auto* normalField=dynamic_cast<const vc::lasagna::LasagnaNormalSampler*>(&normals);
                if(!field||!normalField)throw std::invalid_argument("native seeds require channel fields");
                detail::FiberChunkTensor tensor(*field,*normalField,bounds.lo/scale,bounds.hi/scale,true,
                    chunks.channelHints,chunks.workerCacheBytes);
                tile->seeds=findFiberChunkSeeds(tensor,tensor,predictions,bounds,chunks,scale,true);
                std::erase_if(tile->seeds,[&](const auto& seed){
                    for(int a=0;a<3;++a)if(seed.point[a]<tile->key[a]*chunks.coreBase||seed.point[a]>=(tile->key[a]+1)*chunks.coreBase)return true;
                    return false;
                });
                std::lock_guard lock(admission);--loading;tile->remaining=tile->seeds.size();candidates+=tile->remaining;
                for(size_t i=0;i<tile->seeds.size();++i)tile->pending.push_back(i);
                if(tile->remaining==0)save(tile);else scheduleAdmission(tile);
                loadMore();
            });
        }
    }
    void scheduleAdmission(const std::shared_ptr<Tile>& tile) { // admission held
        if(tile->scheduled||tile->pending.empty())return;
        tile->scheduled=true;pool.push(1,[this,tile]{admit(tile);});
    }
    void save(const std::shared_ptr<Tile>& tile) { // admission held; no remaining users
        std::erase(open,tile);
        pool.push(1,[this,tile]{
            detail::writeFiberPacked(output/"tiles"/(tileName(tile->key)+".msgpack.zst"),
                Json{{"version",1},{"coordinate_units","base_voxels"},{"pieces",std::move(tile->pieces)}});
            ++finishedTiles;
        });
    }
    void admit(const std::shared_ptr<Tile>& tile) {
        std::unique_lock lock(admission);tile->scheduled=false;
        for(size_t n=0;n<128&&!tile->pending.empty()&&active.size()<capacity;++n) {
            const auto index=tile->pending.front();tile->pending.pop_front();const auto& s=tile->seeds[index];
            const Vec p=s.point,dir=s.direction;const double width=s.width*scale;
            bool outside=false;for(int a=0;a<3;++a)outside|=p[a]<lo[a]||p[a]>=hi[a];
            lock.unlock();
            const bool isCovered=outside||coverage.owner(p,dir,radius,width,tolerance)>=0;
            lock.lock();
            if(isCovered) {
                ++covered;if(--tile->remaining==0)save(tile);continue;
            }
            if(active.size()>=capacity){tile->pending.push_front(index);break;}
            std::shared_ptr<Pair> blocker;
            for(const auto& [id,other]:active)if(cv::norm(other->seed-p)<spread &&
                FiberVolumeCoverage::compatibleWidth(other->width,width,tolerance)){blocker=other;break;}
            if(blocker){blocker->waiters.emplace_back(tile,index);continue;}
            auto pair=std::make_shared<Pair>();pair->id=nextPair++;pair->tile=tile;pair->seed=p;pair->dir=dir;pair->width=width;
            active.emplace(pair->id,pair);++startedPairs;peakActive=std::max(peakActive.load(),active.size());
            for(int half=0;half<2;++half) {
                auto& d=pair->half[half];d.pair=pair.get();d.half=half;d.direction=half?-dir:dir;d.previous=p;
                d.points.push_back(p);d.widths.push_back(width);d.losses.push_back(0);d.widthMean=width;
                pool.push(0,[this,pair,half]{check(pair,half);});
            }
        }
        if(active.size()<capacity)scheduleAdmission(tile);
        loadMore();
    }
    void advance(const std::shared_ptr<Pair>& pair,int half) {
        auto& d=pair->half[half];
        if(d.pending.empty()&&!d.traceDone) {
            if(!d.cursor) {auto config=trace;config.initialWidthMean=pair->width/scale;config.preserveInitialDirection=true;
                d.cursor=std::make_unique<FiberTraceCursor>(predictions,pair->seed/scale,d.direction,maximum/scale,config,&normals);}
            d.traceDone=d.cursor->advance();
            for(auto& c:d.cursor->takeCommitted())d.pending.push_back(std::move(c));
        }
        while(!d.pending.empty()) {
            const auto c=d.pending.front();d.pending.pop_front();
            d.points.push_back(c.point*scale);d.widths.push_back(c.widthMean*scale);d.losses.push_back(c.loss);
            d.widthMean=c.widthMean*scale;d.widthCount=uint64_t(c.step)+1;
            d.direction=unit(d.points.back()-d.points[d.points.size()-2]);
            if(c.step%decisionSteps==0){pool.push(0,[this,pair,half]{check(pair,half);});return;}
        }
        if(d.traceDone) {
            const auto& r=d.cursor->result();d.reason=r.reason;
            if(r.points.size()>d.points.size())throw std::logic_error("cursor missed committed geometry");
            d.points.clear();for(const auto&p:r.points)d.points.push_back(p*scale);
            d.widths.resize(d.points.size());d.losses.resize(d.points.size());
            d.widthMean=r.widthSamples?r.widthMean*scale:pair->width;d.widthCount=std::max<uint64_t>(1,r.widthSamples);
            end(pair,half);return;
        }
        pool.push(2,[this,pair,half]{advance(pair,half);});
    }
    void check(const std::shared_ptr<Pair>& pair,int half) {
        auto& d=pair->half[half];const auto p=d.points.back();std::vector<uint8_t> outside;
        if(!ct.sample({p},outside,[this,pair,half]{check(pair,half);} ))return;
        const auto generation=epoch.fetch_add(1);
        const bool near=coverage.owner(p,d.direction,radius,d.widthMean,tolerance)>=0 ||
            coverage.liveHit(p,d.direction,radius,pair->id,generation,d.widthMean,tolerance);
        coverage.addLive(d.previous,p,pair->id,generation,d.widthMean);d.previous=p;
        d.streak=near?d.streak+1:0;d.outsideStreak=outside[0]?d.outsideStreak+1:0;++checks;
        for(int a=0;a<3;++a)if(p[a]<lo[a]-4*grid||p[a]>hi[a]+4*grid)d.reason="bounds";
        if(d.reason.empty()&&d.outsideStreak>=2)d.reason="outside_ct";
        if(d.reason.empty()&&d.streak>=3)d.reason="owned_or_live";
        if(!d.reason.empty()){end(pair,half);return;}
        pool.push(2,[this,pair,half]{advance(pair,half);});
    }
    void end(const std::shared_ptr<Pair>& pair,int half) {
        auto& d=pair->half[half];d.cursor.reset();d.pending.clear();
        {std::lock_guard lock(reasonMutex);++reasons[d.reason];}
        if(pair->unfinished.fetch_sub(1)==1)pool.push(1,[this,pair]{finish(pair);});
    }
    // Same 8-base policy samples and linear interpolation used by Python.
    void resample(Direction& d) {
        if(d.points.size()<2)return;
        std::vector<double> arc(d.points.size(),0);
        for(size_t i=1;i<arc.size();++i)arc[i]=arc[i-1]+cv::norm(d.points[i]-d.points[i-1]);
        if(arc.back()<=0)return;
        std::vector<Vec> p;std::vector<double>w,c;size_t j=0;
        auto at=[&](double a){while(j+2<arc.size()&&arc[j+1]<a)++j;
            double t=arc[j+1]>arc[j]?(a-arc[j])/(arc[j+1]-arc[j]):0;
            p.push_back(d.points[j]*(1-t)+d.points[j+1]*t);w.push_back(d.widths[j]*(1-t)+d.widths[j+1]*t);c.push_back(d.losses[j]*(1-t)+d.losses[j+1]*t);};
        for(double a=0;a<arc.back();a+=grid)at(a);at(arc.back());
        d.points=std::move(p);d.widths=std::move(w);d.losses=std::move(c);
    }
    void publishNext() {
        std::shared_ptr<Pair> pair;
        {std::lock_guard lock(publish);pair=std::move(publishQueue.front());publishQueue.pop_front();}
        finish(pair);
        std::lock_guard lock(publish);
        if(publishQueue.empty())publishing=false;
        else pool.push(1,[this]{publishNext();});
    }
    void finish(const std::shared_ptr<Pair>& pair) {
        auto& fw=pair->half[0];auto& bw=pair->half[1];
        if(!pair->resampled){resample(fw);resample(bw);pair->resampled=true;}
        std::vector<Vec> p(bw.points.rbegin(),bw.points.rend());p.insert(p.end(),fw.points.begin()+1,fw.points.end());
        if(!pair->prepared) {
            if(p.size()>=48 && !ct.sample(p,pair->combinedOutside,[this,pair]{finish(pair);} )){fw.outside={1};return;}
            pair->prepared=true;
            std::lock_guard lock(publish);publishQueue.push_back(pair);
            if(!publishing){publishing=true;pool.push(1,[this]{publishNext();});}
            return;
        }
        if(p.size()>=48) {
            const auto& outside=pair->combinedOutside;
            std::vector<double>w(bw.widths.rbegin(),bw.widths.rend()),c(bw.losses.rbegin(),bw.losses.rend());
            w.insert(w.end(),fw.widths.begin()+1,fw.widths.end());c.insert(c.end(),fw.losses.begin()+1,fw.losses.end());
            std::vector<uint8_t> inside(p.size(),1);
            for(size_t i=0;i<p.size();++i)for(int a=0;a<3;++a)if(p[i][a]<lo[a]||p[i][a]>hi[a])inside[i]=0;
            for(size_t i=0;i<outside.size();) {if(!outside[i]){++i;continue;}size_t j=i+1;while(j<outside.size()&&outside[j])++j;
                if(i==0||j==outside.size()||j-i>16)std::fill(inside.begin()+i,inside.begin()+j,0);i=j;}
            // Serial ownership publication only; beam/CT/live queries continue
            // independently, and geometry preparation above is parallel.
            std::vector<std::pair<size_t,size_t>> parts;
            for(size_t a=0;a<p.size();) {
                if(!inside[a]){++a;continue;}size_t b=a+1;while(b<p.size()&&inside[b])++b;
                if(b-a>=48) {
                    std::vector<uint8_t> dup(b-a),keep(b-a,1);
                    for(size_t i=a;i<b;++i) {
                        const Vec t=unit(p[std::min(i+1,b-1)]-p[i>a?i-1:a]);
                        dup[i-a]=coverage.owner(p[i],t,radius,w[i],tolerance)>=0;
                    }
                    for(size_t i=0;i<dup.size();) {if(!dup[i]){++i;continue;}size_t j=i+1;while(j<dup.size()&&dup[j])++j;
                        if(j-i>24)std::fill(keep.begin()+i+16,keep.begin()+std::max(i+16,j-16),0);i=j;}
                    for(size_t i=0;i<keep.size();) {if(!keep[i]){++i;continue;}size_t j=i+1;while(j<keep.size()&&keep[j])++j;
                        if(j-i>=48&&std::count(dup.begin()+i,dup.begin()+j,1)*2<=int(j-i))parts.emplace_back(a+i,a+j);i=j;}
                }a=b;
            }
            for(const auto&[a,b]:parts) {
                std::vector<Vec> piece(p.begin()+a,p.begin()+b);const auto id=nextFiber++;
                const double mean=(fw.widthMean*fw.widthCount+bw.widthMean*bw.widthCount-pair->width)/(fw.widthCount+bw.widthCount-1);
                Json links=Json::array();links.push_back(coverage.owner(piece.front(),unit(piece[1]-piece[0]),radius+grid,w[a],tolerance));
                links.push_back(coverage.owner(piece.back(),unit(piece.back()-piece[piece.size()-2]),radius+grid,w[b-1],tolerance));
                coverage.addOwned(piece,id,mean);
                Json points=Json::array();double length=0;
                for(size_t i=0;i<piece.size();++i){points.push_back(vectorJson(piece[i]));if(i)length+=cv::norm(piece[i]-piece[i-1]);}
                Json record{{"id",id},{"points_base",std::move(points)},{"width_mean_base",mean},
                    {"width_history_base",std::vector<double>(w.begin()+a,w.begin()+b)},
                    {"cumulative_cost",std::vector<double>(c.begin()+a,c.begin()+b)},
                    {"seed_base",vectorJson(pair->seed)},{"seed_direction",vectorJson(pair->dir)},
                    {"reasons",{bw.reason,fw.reason}},{"links",std::move(links)}};
                {std::lock_guard lock(pair->tile->piecesMutex);pair->tile->pieces.push_back(std::move(record));}
                ++pieces;kept.fetch_add(length);
            }
        }
        coverage.dropLive(pair->id);
        std::lock_guard lock(admission);active.erase(pair->id);++finishedPairs;
        for(auto& [tile,index]:pair->waiters)tile->pending.push_back(index);
        if(--pair->tile->remaining==0)save(pair->tile);
        for(auto& tile:open)scheduleAdmission(tile);
        loadMore();
    }
public:
    Stream(const FiberPredictionSource& p,const vc::lasagna::NormalSampler& n,FiberTraceConfig t,FiberChunkTraceConfig c,
        Json cfg,const std::vector<std::array<int,3>>& k,std::filesystem::path out,std::atomic_bool& stop):
        predictions(p),normals(n),trace(t),chunks(c),options(cfg),keys(k),output(std::move(out)),cancel(stop),
        scale(t.traceToBaseScale),grid(cfg.value("policy_grid_base",8.)),radius(2*grid),spread(16*grid),
        maximum(cfg.value("maximum_base",64000.)),tolerance(t.widthHardRelativeTolerance),
        decisionSteps(int(std::lround(cfg.value("decision_base",64.)/(t.stepVoxels*scale)))),
        lo(vectorOf(cfg.at("box_base_xyz")[0])),hi(vectorOf(cfg.at("box_base_xyz")[1])),
        workers(cfg.value("workers",size_t{128})),capacity(cfg.value("active_fibers",10*workers)),
        seedWorkers(cfg.value("seed_workers",size_t{8})),maxOpen(cfg.value("max_open",size_t{32})),
        pool(workers,cancel),ct(cfg.at("ct_locator").get<std::string>(),cfg.value("ct_level",2),pool),coverage(c.coreBase) {
        if(workers==0||capacity==0||decisionSteps<1||seedWorkers==0||maxOpen==0||tolerance<=0)
            throw std::invalid_argument("invalid native stream settings");
        std::filesystem::create_directories(output/"tiles");
        for(const auto& file:std::filesystem::directory_iterator(output/"tiles")) {
            const auto name=file.path().filename().string();if(!name.ends_with(".msgpack.zst"))continue;
            const auto data=detail::readFiberPacked(file.path());
            for(const auto& piece:data.at("pieces")) {
                std::vector<Vec> line;for(const auto& q:piece.at("points_base"))line.push_back(vectorOf(q));
                const auto id=piece.at("id").get<int64_t>();coverage.addOwned(line,id,piece.at("width_mean_base"));
                nextFiber=std::max(nextFiber.load(),id+1);++pieces;
                for(size_t i=1;i<line.size();++i)kept.fetch_add(cv::norm(line[i]-line[i-1]));
            }
            done.insert(name.substr(0,name.size()-12));++finishedTiles;
        }
        const auto progress=output/"progress.json";
        if(std::filesystem::exists(progress)){std::ifstream in(progress);Json old;in>>old;nextFiber=std::max(nextFiber.load(),old.value("next_fiber_id",int64_t{0}));}
        coverage.flushOwned();
    }
    ~Stream(){pool.stop();}
    Json progress(bool complete=false) {
        Json r{{"finished_tiles",finishedTiles.load()},{"total_tiles",keys.size()},{"pieces",pieces.load()},
            {"kept_base",kept.load()},{"seed_candidates",candidates.load()},{"covered_seeds",covered.load()},
            {"started_pairs",startedPairs.load()},{"finished_pairs",finishedPairs.load()},{"collision_checks",checks.load()},
            {"next_fiber_id",nextFiber.load()},{"wall_seconds",seconds(began)},
            {"cpu_seconds",double(std::clock()-cpuBegin)/CLOCKS_PER_SEC},
            {"workers",workers},{"active_fiber_limit",capacity},{"peak_active_fibers",peakActive.load()},
            {"busy_workers",pool.busy.load()},{"ct_suspensions",ct.suspensions.load()},{"complete",complete}};
        {std::lock_guard lock(admission);r["active_fibers"]=active.size();r["open_tiles"]=open.size();}
        {std::lock_guard lock(reasonMutex);for(const auto&[reason,count]:reasons)r["stop_"+reason]=count;}
        for(size_t i=0;i<4;++i){r["worker_tier_seconds"].push_back(pool.workSeconds[i].load());r["worker_tier_jobs"].push_back(pool.jobs[i].load());}
        const auto budget=vc::render::processChunkCacheService()->decodedByteBudget()->stats();
        r["decoded_cache"]={{"bytes",budget.decodedBytes},{"capacity",budget.maximumBytes},
            {"recycled_bytes",budget.recycledBytes},{"reserve_bytes",budget.reserveBytes},
            {"pending_requests",budget.pendingRequests},{"background_passes",budget.backgroundPasses},
            {"background_evicted_bytes",budget.backgroundEvictedBytes},{"hard_limit_passes",budget.hardLimitPasses}};
        auto path=output/(complete?"summary.json":"progress.json"),tmp=path;tmp+=".partial";
        {std::ofstream f(tmp);f<<r.dump(2)<<'\n';}std::filesystem::rename(tmp,path);std::cout<<r.dump()<<std::endl;return r;
    }
    Json run() {
        {std::lock_guard lock(admission);loadMore();}
        auto last=Clock::now();
        while(finishedTiles.load()<keys.size()&&!cancel.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));pool.rethrow();
            if(seconds(last)>=10){progress();last=Clock::now();}
        }
        pool.stop();pool.rethrow();return progress(!cancel.load());
    }
};
}
Json runFiberStreamVolume(const FiberPredictionSource& p,const vc::lasagna::NormalSampler& n,
    const FiberTraceConfig& t,const FiberChunkTraceConfig& c,const Json& options,
    const std::vector<std::array<int,3>>& keys,const std::filesystem::path& output,std::atomic_bool& cancel) {
    Stream stream(p,n,t,c,options,keys,output,cancel);return stream.run();
}
} // namespace vc::fiber_tracer
