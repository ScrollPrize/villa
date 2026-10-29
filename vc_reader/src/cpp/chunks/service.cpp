// The process-wide chunk service: one decoded-chunk LRU over every source,
// prioritized fetch queues, local and remote worker pools.
#include "vc_reader/chunks.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace vcr::chunks {

std::size_t KeyHash::operator()(const Key& k) const noexcept
{
    std::uint64_t h = 0x9e3779b97f4a7c15ull ^ k.source;
    for (const std::uint64_t v : {std::uint64_t(std::uint32_t(k.level)), std::uint64_t(k.z), std::uint64_t(k.y),
                                  std::uint64_t(k.x)}) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 0xff51afd7ed558ccdull;
    }
    return std::size_t(h ^ (h >> 33));
}

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t kFillCost = 64;                      // bookkeeping of a chunk without bytes
constexpr auto kErrorRetry = std::chrono::seconds(5);      // failed chunks are fetched again after this
constexpr auto kAdaptiveWindow = std::chrono::seconds(3);

// Ordering keys; smaller sorts first.
struct ViewOrder {
    int coarser;  // negated relative level: coarse fallbacks first
    float distance;
    std::uint64_t seq;
    friend auto operator<=>(const ViewOrder&, const ViewOrder&) = default;
};
struct BackgroundOrder {
    int priority;  // negated
    std::uint64_t seq;
    friend auto operator<=>(const BackgroundOrder&, const BackgroundOrder&) = default;
};
}  // namespace

struct Service::Impl : std::enable_shared_from_this<Service::Impl> {
    struct Source {
        SourceId id;
        std::vector<std::shared_ptr<Array>> levels;
        Loader loader;
        std::int32_t levelCount = 0;
        SourceOptions options;
        bool remote = false;
        std::vector<std::uint8_t> fill;
        std::vector<std::size_t> unresolved;
        std::size_t running = 0;
        std::map<ListenerId, std::function<void(const Key&)>> ready;
        std::map<ListenerId, std::function<void(const Key&, bool)>> fetching;
    };
    struct Completion {
        bool done = false;
        Chunk result;
    };
    struct Task {
        Key key;
        std::uint64_t seq = 0;
        bool remote = false, running = false, foreground = false, invalidated = false;
        std::optional<int> background;
        std::unordered_map<std::uint64_t, std::pair<int, float>> views;  // view -> (relative level, distance^2)
        std::size_t waiters = 0;
        std::shared_ptr<Completion> completion = std::make_shared<Completion>();
        // Where the task sits in its pool's queues, for removal.
        std::optional<ViewOrder> global;
        std::optional<BackgroundOrder> back;
    };
    struct Entry {
        Status status;
        ChunkPtr bytes;
        std::string error;
        std::size_t cost;
        bool preferSelf;
        Clock::time_point when;
        std::list<Key>::iterator lru;
    };
    struct Pool {
        std::vector<std::thread> threads;
        std::condition_variable cv;
        std::unordered_map<std::uint64_t, std::set<ViewOrder>> views;  // per view, its queued chunks
        std::set<ViewOrder> global;                                    // every view-demanded chunk, best demand
        std::set<std::uint64_t> foreground;                            // FIFO
        std::set<BackgroundOrder> background;
        std::size_t active = 0;
        bool preferViews = true;  // alternates views and foreground work
        bool empty() const { return global.empty() && foreground.empty() && background.empty(); }
    };

    Options options;
    mutable std::mutex mutex;
    std::condition_variable resolved;
    bool stopping = false;
    std::map<SourceId, Source> sources;
    SourceId nextSource = 1;
    ListenerId nextListener = 1;
    std::unordered_map<Key, Task, KeyHash> tasks;
    std::unordered_map<std::uint64_t, Key> bySeq;
    std::uint64_t nextSeq = 1;
    std::unordered_map<Key, Entry, KeyHash> entries;
    std::list<Key> lru, preferSelfLru;  // most recent first
    std::size_t decodedBytes = 0;
    std::uint64_t activeView = 0;
    std::unordered_map<std::uint64_t, std::uint64_t> viewVersions;
    std::unordered_map<std::uint64_t, std::unordered_set<Key, KeyHash>> viewKeys;
    Pool local, remote;
    std::size_t admission;
    std::uint64_t fetched = 0, failed = 0, evicted = 0;
    // Adaptive admission.
    std::thread monitor;
    std::condition_variable monitorCv;
    double bytesPerSecond = 0.0, lastWindowRate = 0.0;
    std::uint64_t lastRemoteBytes = 0;
    Clock::time_point lastWindow = Clock::now();
    int direction = 1;
    AdaptiveState settled;

    explicit Impl(Options o) : options(std::move(o))
    {
        admission = std::max<std::size_t>(1, std::min(options.remoteAdmission, std::max<std::size_t>(options.remoteWorkers, 1)));
        if (options.adaptive && options.adaptiveState && options.adaptiveState->limit)
            admission = std::min(admission, options.adaptiveState->limit);
        settled = {admission, options.adaptiveState ? options.adaptiveState->bytesPerSecond : 0.0};
    }

    void start()
    {
        auto self = shared_from_this();
        for (std::size_t i = 0; i < std::max<std::size_t>(options.localWorkers, 1); ++i)
            local.threads.emplace_back([self] { self->work(self->local); });
        for (std::size_t i = 0; i < std::max<std::size_t>(options.remoteWorkers, 1); ++i)
            remote.threads.emplace_back([self] { self->work(self->remote); });
        monitor = std::thread([self] { self->watch(); });
    }

    void stop()
    {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        local.cv.notify_all();
        remote.cv.notify_all();
        monitorCv.notify_all();
        resolved.notify_all();
        for (auto* pool : {&local, &remote})
            for (auto& t : pool->threads)
                t.join();
        monitor.join();
    }

    Pool& poolOf(const Task& t) { return t.remote ? remote : local; }

    // ---- queues (mutex held) ------------------------------------------------------------

    std::optional<ViewOrder> bestView(const Task& t) const
    {
        std::optional<ViewOrder> best;
        for (const auto& [view, d] : t.views) {
            const ViewOrder o{-d.first, d.second, t.seq};
            if (!best || o < *best)
                best = o;
        }
        return best;
    }

    void unqueue(Task& t)
    {
        Pool& p = poolOf(t);
        for (const auto& [view, d] : t.views)
            if (auto it = p.views.find(view); it != p.views.end()) {
                it->second.erase(ViewOrder{-d.first, d.second, t.seq});
                if (it->second.empty())
                    p.views.erase(it);
            }
        if (t.global)
            p.global.erase(*t.global);
        if (t.back)
            p.background.erase(*t.back);
        p.foreground.erase(t.seq);
        t.global.reset();
        t.back.reset();
    }

    void enqueue(Task& t)
    {
        if (t.running)
            return;
        Pool& p = poolOf(t);
        for (const auto& [view, d] : t.views)
            p.views[view].insert(ViewOrder{-d.first, d.second, t.seq});
        t.global = bestView(t);
        if (t.global)
            p.global.insert(*t.global);
        if (t.foreground)
            p.foreground.insert(t.seq);
        if (t.background) {
            t.back = BackgroundOrder{-*t.background, t.seq};
            p.background.insert(*t.back);
        }
        p.cv.notify_one();
    }

    bool demanded(const Task& t) const { return !t.views.empty() || t.foreground || t.background || t.waiters; }

    void cancelIfIdle(Task& t)
    {
        if (t.running || demanded(t))
            return;
        unqueue(t);
        if (auto s = sources.find(t.key.source); s != sources.end())
            --s->second.unresolved[std::size_t(t.key.level)];
        bySeq.erase(t.seq);
        tasks.erase(t.key);
    }

    Task& taskFor(const Key& key, Source& src)
    {
        auto [it, fresh] = tasks.try_emplace(key);
        Task& t = it->second;
        if (fresh) {
            t.key = key;
            t.seq = nextSeq++;
            t.remote = src.remote;
            bySeq[t.seq] = key;
            ++src.unresolved[std::size_t(key.level)];
        }
        return t;
    }

    // Applies a demand to a chunk that is not resident.
    Task& demand(const Key& key, Source& src, const Demand& d)
    {
        Task& t = taskFor(key, src);
        unqueue(t);
        if (d.foreground)
            t.foreground = true;
        else if (!d.view)
            t.background = std::max(t.background.value_or(d.priority), d.priority);
        enqueue(t);
        return t;
    }

    std::optional<Key> pick(Pool& p)
    {
        if (p.empty())
            return std::nullopt;
        std::uint64_t seq = 0;
        if (auto it = p.views.find(activeView); activeView && it != p.views.end() && !it->second.empty() && (p.preferViews || p.foreground.empty())) {
            seq = it->second.begin()->seq;
            p.preferViews = false;
        } else if (!p.global.empty() && (p.preferViews || p.foreground.empty())) {
            seq = p.global.begin()->seq;
            p.preferViews = false;
        } else if (!p.foreground.empty()) {
            seq = *p.foreground.begin();
            p.preferViews = true;
        } else {
            seq = p.background.begin()->seq;
        }
        const Key key = bySeq.at(seq);
        Task& t = tasks.at(key);
        unqueue(t);
        t.running = true;
        return key;
    }

    // ---- decoded chunks (mutex held) ------------------------------------------------------

    Chunk result(const Entry& e) const { return Chunk{e.status, e.bytes, e.error}; }

    void touch(Entry& e, const Key& key)
    {
        auto& list = e.preferSelf ? preferSelfLru : lru;
        list.splice(list.begin(), list, e.lru);
        (void)key;
    }

    void insert(const Key& key, Chunk chunk, bool preferSelf)
    {
        auto it = entries.find(key);
        if (it != entries.end())
            drop(it);
        const std::size_t cost = chunk.bytes ? chunk.bytes->size() : kFillCost;
        auto& list = preferSelf ? preferSelfLru : lru;
        list.push_front(key);
        entries.emplace(key, Entry{chunk.status, std::move(chunk.bytes), std::move(chunk.error), cost, preferSelf,
                                   Clock::now(), list.begin()});
        decodedBytes += cost;
        while (decodedBytes > options.decodedBytes && (!preferSelfLru.empty() || !lru.empty())) {
            auto& victims = !preferSelfLru.empty() ? preferSelfLru : lru;
            drop(entries.find(victims.back()));
            ++evicted;
        }
    }

    void drop(std::unordered_map<Key, Entry, KeyHash>::iterator it)
    {
        decodedBytes -= it->second.cost;
        (it->second.preferSelf ? preferSelfLru : lru).erase(it->second.lru);
        entries.erase(it);
    }

    // A resident chunk, if any. Failed chunks expire so they are fetched again.
    Entry* resident(const Key& key)
    {
        auto it = entries.find(key);
        if (it == entries.end())
            return nullptr;
        if (it->second.status == Status::Error && Clock::now() - it->second.when > kErrorRetry) {
            drop(it);
            return nullptr;
        }
        return &it->second;
    }

    Source& source(const Key& key)
    {
        auto it = sources.find(key.source);
        if (it == sources.end())
            throw std::invalid_argument("unknown chunk source " + std::to_string(key.source));
        Source& s = it->second;
        if (key.level < 0 || key.level >= s.levelCount)
            throw std::out_of_range("chunk level outside the source");
        return s;
    }

    // ---- workers -----------------------------------------------------------------------------

    bool admitted(const Pool& p) const { return &p == &local || p.active < admission; }

    // The service whose worker runs on this thread, if any. Loaders (e.g. computed
    // arrays) may read other chunks of the service; such nested waits run the load
    // themselves instead of queueing behind workers that may all be waiting.
    static inline thread_local const Impl* worker = nullptr;

    void work(Pool& p)
    {
        worker = this;
        std::unique_lock lock(mutex);
        while (true) {
            p.cv.wait(lock, [&] { return stopping || (!p.empty() && admitted(p)); });
            if (stopping)
                return;
            const auto key = pick(p);
            if (!key)
                continue;
            ++p.active;
            execute(*key, lock, &p);
            p.cv.notify_one();
        }
    }

    // Loads a task marked running and resolves it. Unlocks while loading and
    // notifying listeners; returns locked. `p`: the pool whose admission it holds.
    void execute(const Key key, std::unique_lock<std::mutex>& lock, Pool* p)
    {
        Source& src = sources.at(key.source);
        ++src.running;
        auto arrays = src.levels;
        auto loader = src.loader;
        const auto fill = arrays.empty() ? src.fill : arrays[std::size_t(key.level)]->spec().fill;
        const bool detect = src.options.detectFill;
        std::vector<std::function<void(const Key&, bool)>> fetchListeners;
        if (src.remote)
            for (const auto& [_, f] : src.fetching)
                fetchListeners.push_back(f);
        lock.unlock();

        for (const auto& f : fetchListeners)
            try { f(key, true); } catch (...) {}
        Chunk chunk;
        try {
            ChunkPtr bytes = loader ? loader(key.level, key.z, key.y, key.x)
                                    : arrays[std::size_t(key.level)]->chunk({key.z, key.y, key.x});
            if (!bytes) {
                chunk.status = Status::Fill;
            } else if (detect && !fill.empty() && allFill(*bytes, fill)) {
                chunk.status = Status::Fill;
            } else {
                chunk.status = Status::Data;
                chunk.bytes = std::move(bytes);
            }
        } catch (const std::exception& e) {
            chunk.status = Status::Error;
            chunk.error = e.what();
        } catch (...) {
            chunk.status = Status::Error;
            chunk.error = "chunk loader threw a non-standard exception";
        }
        for (const auto& f : fetchListeners)
            try { f(key, false); } catch (...) {}

        lock.lock();
        if (p)
            --p->active;
        std::vector<std::function<void(const Key&)>> readyListeners;
        auto s = sources.find(key.source);
        auto t = tasks.find(key);
        if (s != sources.end()) {
            --s->second.running;
            if (t != tasks.end() && !t->second.invalidated)
                insert(key, chunk, s->second.options.preferSelfEviction);
            for (const auto& [_, f] : s->second.ready)
                readyListeners.push_back(f);
            chunk.status == Status::Error ? ++failed : ++fetched;
        }
        if (t != tasks.end()) {
            t->second.completion->done = true;
            t->second.completion->result = chunk;
            if (s != sources.end())
                --s->second.unresolved[std::size_t(key.level)];
            bySeq.erase(t->second.seq);
            for (const auto& [view, _] : t->second.views)
                if (auto v = viewKeys.find(view); v != viewKeys.end())
                    v->second.erase(key);
            tasks.erase(t);
        }
        resolved.notify_all();
        lock.unlock();
        for (const auto& f : readyListeners)
            try { f(key); } catch (...) {}
        lock.lock();
    }

    // Waits for a demanded task; on one of this service's workers, a task nobody
    // runs yet is loaded on this thread.
    Chunk await(Task& t, std::unique_lock<std::mutex>& lock)
    {
        const Key key = t.key;
        ++t.waiters;
        auto completion = t.completion;
        if (worker == this && !t.running) {
            unqueue(t);
            t.running = true;
            execute(key, lock, nullptr);
        }
        resolved.wait(lock, [&] { return completion->done || stopping; });
        if (auto it = tasks.find(key); it != tasks.end() && it->second.completion == completion)
            --it->second.waiters;
        if (!completion->done)
            return Chunk{Status::Error, nullptr, "chunk service stopped"};
        return completion->result;
    }

    static bool allFill(const std::vector<std::uint8_t>& bytes, const std::vector<std::uint8_t>& fill)
    {
        const std::size_t n = fill.size();
        if (bytes.size() % n)
            return false;
        if (n == 1)
            return std::all_of(bytes.begin(), bytes.end(), [f = fill[0]](std::uint8_t b) { return b == f; });
        for (std::size_t i = 0; i < bytes.size(); i += n)
            if (std::memcmp(bytes.data() + i, fill.data(), n) != 0)
                return false;
        return true;
    }

    // Tunes remote admission: keep stepping while throughput rises, step back when it falls.
    void watch()
    {
        std::unique_lock lock(mutex);
        while (!monitorCv.wait_for(lock, kAdaptiveWindow, [&] { return stopping; })) {
            std::uint64_t total = 0;
            for (const auto& [_, s] : sources)
                if (s.remote)
                    for (const auto& a : s.levels)
                        total += a->stats().remote_bytes.load();
            const auto now = Clock::now();
            const double seconds = std::chrono::duration<double>(now - lastWindow).count();
            const double rate = seconds > 0 ? double(total - std::min(total, lastRemoteBytes)) / seconds : 0.0;
            lastRemoteBytes = total;
            lastWindow = now;
            bytesPerSecond = rate;
            const bool saturated = !remote.empty() && remote.active >= admission;
            if (!options.adaptive || !saturated) {
                lastWindowRate = 0.0;
                continue;
            }
            const std::size_t upper = std::max<std::size_t>(1, std::min(options.remoteAdmission, options.remoteWorkers));
            if (lastWindowRate > 0.0 && rate < lastWindowRate * 0.95)
                direction = -direction;  // the last step hurt: go back
            const std::size_t next = direction > 0 ? std::min(upper, admission + 2)
                                                   : std::max<std::size_t>(1, admission > 2 ? admission - 2 : 1);
            if (rate >= lastWindowRate)
                settled = {admission, rate};
            admission = next;
            lastWindowRate = rate;
            remote.cv.notify_all();
        }
    }
};

Service::Service(Options options) : impl_(std::make_shared<Impl>(std::move(options))) { impl_->start(); }

Service::~Service() { impl_->stop(); }

SourceId Service::addSource(std::vector<std::shared_ptr<Array>> levels, SourceOptions options)
{
    if (levels.empty())
        throw std::invalid_argument("a chunk source needs at least one level");
    for (const auto& a : levels)
        if (!a || a->spec().shape.size() != 3)
            throw std::invalid_argument("chunk source levels must be rank-three arrays");
    std::lock_guard lock(impl_->mutex);
    const SourceId id = impl_->nextSource++;
    auto& s = impl_->sources[id];
    s.id = id;
    s.levelCount = std::int32_t(levels.size());
    s.options = options;
    s.unresolved.assign(levels.size(), 0);
    for (const auto& a : levels) {
        if (!a)
            throw std::invalid_argument("a chunk source level has no array");
        s.remote |= a->remote();
    }
    s.fill = levels.front()->spec().fill;
    s.levels = std::move(levels);
    return id;
}

SourceId Service::addSource(Loader loader, std::int32_t levels, SourceOptions options)
{
    if (!loader || levels < 1)
        throw std::invalid_argument("a keyed source needs a loader and at least one level");
    std::lock_guard lock(impl_->mutex);
    const SourceId id = impl_->nextSource++;
    auto& s = impl_->sources[id];
    s.id = id;
    s.loader = std::move(loader);
    s.levelCount = levels;
    s.options = options;
    s.remote = options.remote;
    s.unresolved.assign(std::size_t(levels), 0);
    return id;
}

void Service::removeSource(SourceId source)
{
    std::lock_guard lock(impl_->mutex);
    auto& I = *impl_;
    for (auto it = I.tasks.begin(); it != I.tasks.end();) {
        if (it->first.source != source) {
            ++it;
            continue;
        }
        Impl::Task& t = it->second;
        if (!t.running) {
            I.unqueue(t);
            t.completion->done = true;
            t.completion->result = Chunk{Status::Error, nullptr, "chunk source removed"};
            I.bySeq.erase(t.seq);
            it = I.tasks.erase(it);
        } else {
            ++it;  // finishes, and its result is discarded
        }
    }
    for (auto it = I.entries.begin(); it != I.entries.end();)
        it = it->first.source == source ? (I.decodedBytes -= it->second.cost,
                                           (it->second.preferSelf ? I.preferSelfLru : I.lru).erase(it->second.lru),
                                           I.entries.erase(it))
                                        : std::next(it);
    for (auto& [_, keys] : I.viewKeys)
        std::erase_if(keys, [&](const Key& k) { return k.source == source; });
    I.sources.erase(source);
    I.resolved.notify_all();
}

void Service::invalidate(SourceId source)
{
    std::lock_guard lock(impl_->mutex);
    auto& I = *impl_;
    for (auto& [key, task] : I.tasks)
        if (key.source == source && task.running)
            task.invalidated = true;
    for (auto it = I.entries.begin(); it != I.entries.end();)
        it = it->first.source == source ? (I.decodedBytes -= it->second.cost,
                                           (it->second.preferSelf ? I.preferSelfLru : I.lru).erase(it->second.lru),
                                           I.entries.erase(it))
                                        : std::next(it);
}

Chunk Service::probe(const Key& key)
{
    std::lock_guard lock(impl_->mutex);
    auto it = impl_->entries.find(key);
    return it == impl_->entries.end()
        ? Chunk{impl_->tasks.contains(key) ? Status::Queued : Status::Unavailable, nullptr, {}}
        : impl_->result(it->second);
}

Chunk Service::request(const Key& key, const Demand& d)
{
    std::lock_guard lock(impl_->mutex);
    auto& I = *impl_;
    auto& src = I.source(key);
    if (auto* e = I.resident(key)) {
        I.touch(*e, key);
        return I.result(*e);
    }
    if (d.view) {  // a single view request: treat like a one-sample demand without replacing others
        Impl::Task& t = I.taskFor(key, src);
        I.unqueue(t);
        auto& slot = t.views[d.view];
        slot = {std::max(slot.first, 0), 0.0f};
        I.viewKeys[d.view].insert(key);
        I.enqueue(t);
    } else {
        I.demand(key, src, d);
    }
    return {Status::Queued, nullptr, {}};
}

Chunk Service::get(const Key& key, const Demand& d)
{
    std::unique_lock lock(impl_->mutex);
    auto& I = *impl_;
    auto& src = I.source(key);
    if (auto* e = I.resident(key)) {
        I.touch(*e, key);
        return I.result(*e);
    }
    return I.await(I.demand(key, src, d.view ? Demand{0, true, d.priority} : d), lock);
}

void Service::read(SourceId source, std::int32_t level, const Index& start, const Index& size, std::uint8_t* out,
                   const Demand& demand)
{
    std::shared_ptr<Array> array;
    {
        std::lock_guard lock(impl_->mutex);
        auto& src = impl_->source(Key{source, level});
        if (src.levels.empty())
            throw std::invalid_argument("region reads need a source of arrays");
        array = src.levels[std::size_t(level)];
    }
    const auto& spec = array->spec();
    if (start.size() != 3 || size.size() != 3)
        throw std::invalid_argument("region reads are three-dimensional");
    const std::size_t isz = spec.itemsize;
    std::int64_t lo[3], hi[3];
    bool empty = false, clipped = false;
    for (int a = 0; a < 3; ++a) {
        if (size[a] < 0)
            throw std::invalid_argument("negative region size");
        lo[a] = std::max<std::int64_t>(start[a], 0);
        hi[a] = std::min(start[a] + size[a], spec.shape[a]);
        empty |= hi[a] <= lo[a];
        clipped |= lo[a] != start[a] || hi[a] != start[a] + size[a];
    }
    const std::int64_t total = size[0] * size[1] * size[2];
    if (empty || clipped)
        for (std::int64_t i = 0; i < total; ++i)
            std::memcpy(out + std::size_t(i) * isz, spec.fill.data(), isz);
    if (empty)
        return;
    const auto& c = spec.chunks;
    std::vector<Key> keys;
    for (auto z = lo[0] / c[0]; z <= (hi[0] - 1) / c[0]; ++z)
        for (auto y = lo[1] / c[1]; y <= (hi[1] - 1) / c[1]; ++y)
            for (auto x = lo[2] / c[2]; x <= (hi[2] - 1) / c[2]; ++x)
                keys.push_back(Key{source, level, z, y, x});
    if (keys.size() > 1)
        prefetch(keys, demand.view ? Demand{0, false, demand.priority} : demand);
    for (const auto& key : keys) {
        const Chunk chunk = get(key, demand);
        if (chunk.status == Status::Error)
            throw std::runtime_error("chunk read failed: " + chunk.error);
        if (chunk.status == Status::Data && (!chunk.bytes || chunk.bytes->size() != array->chunk_bytes()))
            throw std::runtime_error("chunk read returned the wrong number of bytes");
        const std::int64_t base[3] = {key.z * c[0], key.y * c[1], key.x * c[2]};
        const std::int64_t a[3] = {std::max(lo[0], base[0]), std::max(lo[1], base[1]), std::max(lo[2], base[2])};
        const std::int64_t b[3] = {std::min(hi[0], base[0] + c[0]), std::min(hi[1], base[1] + c[1]),
                                   std::min(hi[2], base[2] + c[2])};
        const std::size_t row = std::size_t(b[2] - a[2]) * isz;
        for (auto z = a[0]; z < b[0]; ++z)
            for (auto y = a[1]; y < b[1]; ++y) {
                std::uint8_t* dst = out + std::size_t(((z - start[0]) * size[1] + (y - start[1])) * size[2] +
                                                      (a[2] - start[2])) * isz;
                if (chunk.status == Status::Data) {
                    const auto offset = ((z - base[0]) * c[1] + (y - base[1])) * c[2] + (a[2] - base[2]);
                    std::memcpy(dst, chunk.bytes->data() + std::size_t(offset) * isz, row);
                } else {
                    for (std::size_t i = 0; i < row; i += isz)
                        std::memcpy(dst + i, spec.fill.data(), isz);
                }
            }
    }
}

void Service::prefetch(std::span<const Key> keys, const Demand& d, bool wait)
{
    std::vector<std::shared_ptr<Impl::Completion>> pending;
    std::unique_lock lock(impl_->mutex);
    auto& I = *impl_;
    for (const auto& key : keys) {
        auto& src = I.source(key);
        if (I.resident(key))
            continue;
        Impl::Task& t = I.demand(key, src, d.view ? Demand{0, false, d.priority} : d);
        if (wait)
            pending.push_back(t.completion);
    }
    if (wait)
        I.resolved.wait(lock, [&] {
            return I.stopping || std::all_of(pending.begin(), pending.end(), [](const auto& c) { return c->done; });
        });
}

void Service::replaceViewDemand(std::uint64_t view, std::uint64_t version, std::array<float, 2> focus,
                                std::vector<ViewSample> samples)
{
    if (!view)
        throw std::invalid_argument("view demand needs a nonzero view id");
    std::lock_guard lock(impl_->mutex);
    auto& I = *impl_;
    auto& current = I.viewVersions[view];
    if (version < current)
        return;  // a newer render already replaced this demand
    current = version;
    auto& keys = I.viewKeys[view];
    std::unordered_map<Key, std::pair<int, float>, KeyHash> wanted;
    for (const auto& s : samples) {
        const float dx = s.position[0] - focus[0], dy = s.position[1] - focus[1];
        const float d2 = dx * dx + dy * dy;
        auto [it, fresh] = wanted.try_emplace(s.key, s.relativeLevel, d2);
        if (!fresh)
            it->second = {std::max(it->second.first, s.relativeLevel), std::min(it->second.second, d2)};
    }
    // Withdraw what the view no longer needs.
    for (auto it = keys.begin(); it != keys.end();) {
        if (wanted.contains(*it)) {
            ++it;
            continue;
        }
        if (auto t = I.tasks.find(*it); t != I.tasks.end()) {
            I.unqueue(t->second);
            t->second.views.erase(view);
            I.enqueue(t->second);
            I.cancelIfIdle(t->second);
        }
        it = keys.erase(it);
    }
    for (const auto& [key, d] : wanted) {
        auto s = I.sources.find(key.source);
        if (s == I.sources.end() || key.level < 0 || key.level >= s->second.levelCount || I.resident(key))
            continue;
        Impl::Task& t = I.taskFor(key, s->second);
        I.unqueue(t);
        t.views[view] = d;
        I.enqueue(t);
        keys.insert(key);
    }
}

void Service::clearViewDemand(std::uint64_t view, std::uint64_t version, std::optional<SourceId> source)
{
    std::lock_guard lock(impl_->mutex);
    auto& I = *impl_;
    auto& current = I.viewVersions[view];
    if (version < current)
        return;
    current = version;
    auto it = I.viewKeys.find(view);
    if (it == I.viewKeys.end())
        return;
    for (auto k = it->second.begin(); k != it->second.end();) {
        if (source && k->source != *source) {
            ++k;
            continue;
        }
        if (auto t = I.tasks.find(*k); t != I.tasks.end()) {
            I.unqueue(t->second);
            t->second.views.erase(view);
            I.enqueue(t->second);
            I.cancelIfIdle(t->second);
        }
        k = it->second.erase(k);
    }
}

void Service::markViewActive(std::uint64_t view)
{
    std::lock_guard lock(impl_->mutex);
    impl_->activeView = view;
}

Service::ListenerId Service::addReadyListener(SourceId source, std::function<void(const Key&)> listener)
{
    std::lock_guard lock(impl_->mutex);
    const auto id = impl_->nextListener++;
    impl_->sources.at(source).ready.emplace(id, std::move(listener));
    return id;
}

Service::ListenerId Service::addFetchListener(SourceId source, std::function<void(const Key&, bool)> listener)
{
    std::lock_guard lock(impl_->mutex);
    const auto id = impl_->nextListener++;
    impl_->sources.at(source).fetching.emplace(id, std::move(listener));
    return id;
}

void Service::removeListener(ListenerId id)
{
    std::lock_guard lock(impl_->mutex);
    for (auto& [_, s] : impl_->sources) {
        s.ready.erase(id);
        s.fetching.erase(id);
    }
}

void Service::configure(std::size_t decodedBytes, std::size_t remoteAdmission, bool adaptive)
{
    std::lock_guard lock(impl_->mutex);
    auto& I = *impl_;
    I.options.decodedBytes = decodedBytes;
    I.options.remoteAdmission = std::max<std::size_t>(1, remoteAdmission);
    I.options.adaptive = adaptive;
    I.admission = std::max<std::size_t>(1, std::min(I.options.remoteAdmission, I.options.remoteWorkers));
    while (I.decodedBytes > I.options.decodedBytes && (!I.preferSelfLru.empty() || !I.lru.empty())) {
        auto& victims = !I.preferSelfLru.empty() ? I.preferSelfLru : I.lru;
        I.drop(I.entries.find(victims.back()));
        ++I.evicted;
    }
    I.remote.cv.notify_all();
}

AdaptiveState Service::adaptiveState() const
{
    std::lock_guard lock(impl_->mutex);
    return impl_->settled;
}

Stats Service::stats(std::optional<SourceId> source) const
{
    std::lock_guard lock(impl_->mutex);
    const auto& I = *impl_;
    Stats s;
    s.decodedBytes = I.decodedBytes;
    s.decodedCapacity = I.options.decodedBytes;
    s.resident = I.entries.size();
    s.queued = I.tasks.size();
    s.remoteInFlight = I.remote.active;
    s.localInFlight = I.local.active;
    s.remoteAdmission = I.admission;
    s.remoteBytesPerSecond = I.bytesPerSecond;
    s.fetched = I.fetched;
    s.failed = I.failed;
    s.evicted = I.evicted;
    if (source)
        if (auto it = I.sources.find(*source); it != I.sources.end())
            s.unresolvedByLevel = it->second.unresolved;
    return s;
}

void Service::drain(SourceId source)
{
    std::unique_lock lock(impl_->mutex);
    auto& I = *impl_;
    I.resolved.wait(lock, [&] {
        if (I.stopping)
            return true;
        auto it = I.sources.find(source);
        if (it == I.sources.end())
            return true;
        const auto& u = it->second.unresolved;
        return it->second.running == 0 && std::all_of(u.begin(), u.end(), [](std::size_t n) { return n == 0; });
    });
}

namespace {
std::mutex processMutex;
std::shared_ptr<Service> processInstance;
}  // namespace

std::shared_ptr<Service> processService()
{
    std::lock_guard lock(processMutex);
    if (!processInstance)
        processInstance = std::make_shared<Service>();
    return processInstance;
}

std::shared_ptr<Service> configureProcessService(Service::Options options)
{
    std::lock_guard lock(processMutex);
    if (!processInstance)
        processInstance = std::make_shared<Service>(std::move(options));
    else
        processInstance->configure(options.decodedBytes, options.remoteAdmission, options.adaptive);
    return processInstance;
}

}  // namespace vcr::chunks
