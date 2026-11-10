// Byte-bounded LRU of decoded chunks, shared between arrays.
#pragma once

#include <cstddef>
#include <cstdint>
#include "lru.hpp"
#include "available_memory.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <thread>
#include <condition_variable>
#include <algorithm>

namespace vcr {

using Chunk = std::vector<std::uint8_t>;
using ChunkPtr = std::shared_ptr<const Chunk>;  // nullptr: the chunk is not stored (reads as fill)

class MemoryCache {
public:
    struct LockTiming { std::uint64_t calls=0, wait_ns=0, hold_ns=0; };
    struct Profile {
        LockTiming get, put, erase, clear, refresh;
        std::uint64_t refresh_calls=0, refresh_probe_ns=0;
    };
    // Opt-in diagnostics. Counters are updated under the existing cache lock;
    // no extra contended counter lock/atomic is added to measured operations.
    void enable_profiling(bool enabled=true) { profiling_.store(enabled, std::memory_order_relaxed); }
    Profile profile() const {
        Profile result;
        { std::lock_guard lock(refresh_mutex_); result = profile_; }
        for (const auto& shard : shards_) {
            std::lock_guard lock(shard->mutex);
            for (auto member : {&Profile::get, &Profile::put, &Profile::erase, &Profile::clear}) {
                auto& dst = result.*member;
                const auto& src = shard->profile.*member;
                dst.calls += src.calls; dst.wait_ns += src.wait_ns; dst.hold_ns += src.hold_ns;
            }
        }
        return result;
    }
    static std::size_t default_shards() { return std::max(1u, std::thread::hardware_concurrency()); }
    struct MaintenanceStats {
        std::uint64_t bypasses, admitted_bytes, evicted_bytes, batches;
        std::size_t headroom_bytes, reclaiming_bytes;
        std::uint64_t assist_puts, assist_bytes, assist_batches;
        std::uint64_t selection_scans, reclaim_claims, evicted_entries, inline_batches;
    };
    // Configure an explicitly synchronous cache before concurrent use.
    void enable_maintenance(std::size_t workers=4, std::size_t candidate_scan=8) {
        if (!workers || workers>64)
            throw std::invalid_argument("maintenance requires 1..64 workers");
        candidate_scan=std::min(candidate_scan,shards_.size());
        if (maintenance_.load()) {
            if (workers==vacuum_threads_.size() && candidate_scan==candidate_scan_) return;
            throw std::invalid_argument("maintenance policy cannot change after startup");
        }
        stopping_.store(false);
        candidate_scan_=candidate_scan;
        headroom_.store(std::max<std::size_t>(1,max_bytes()/100));
        maintenance_.store(true);
        try {
            for(std::size_t i=0;i<workers;++i)
                vacuum_threads_.emplace_back([this,i]{ vacuum_loop(i); });
        } catch (...) { stop_maintenance(); maintenance_.store(false); throw; }
    }
    ~MemoryCache() { stop_maintenance(); }
    MaintenanceStats maintenance_stats() const {
        return {bypasses_.load(),admitted_.load(),evicted_.load(),batches_.load(),effective_headroom(),reclaiming_.load(),
                assist_puts_.load(),assist_bytes_.load(),assist_batches_.load(),
                selection_scans_.load(),reclaim_claims_.load(),evicted_entries_.load(),inline_batches_.load()};
    }
    // Explicit testing/shutdown barrier; ordinary reads/writes never wait here.
    bool wait_for_maintenance(std::chrono::milliseconds timeout=std::chrono::seconds(2)) {
        const auto end=std::chrono::steady_clock::now()+timeout;
        if(maintenance_.load())request_eviction();
        while(maintenance_.load() && (pressure_.load() || reclaiming_.load())) {
            if(std::chrono::steady_clock::now()>=end)return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return true;
    }
    struct AutoOptions {
        double fraction = 0.5;
        double smoothing_seconds = 10;
        std::chrono::milliseconds refresh_interval{1000};
        // Injectable for deterministic budget/pressure tests. Production callers
        // use the platform probe, including the container's available memory.
        std::function<std::uint64_t()> available = memory::available_bytes;
        std::function<std::chrono::steady_clock::time_point()> now = std::chrono::steady_clock::now;
    };
    explicit MemoryCache(std::size_t max_bytes, std::size_t shards = default_shards(), bool background = true)
        : budget_(max_bytes)
    {
        init_shards(shards);
        if (background && max_bytes) enable_maintenance();
    }
    explicit MemoryCache(AutoOptions options, std::size_t shards = default_shards(), bool background = true)
        : auto_(std::move(options))
    {
        init_shards(shards);
        if (!std::isfinite(auto_->fraction) || auto_->fraction <= 0 || auto_->fraction >= 1)
            throw std::invalid_argument("auto cache fraction must be between zero and one");
        if (!std::isfinite(auto_->smoothing_seconds) || auto_->smoothing_seconds < 0)
            throw std::invalid_argument("auto cache smoothing_seconds must be finite and nonnegative");
        if (auto_->refresh_interval.count() < 0 || !auto_->available || !auto_->now)
            throw std::invalid_argument("auto cache requires a nonnegative refresh interval, memory probe and clock");
        refresh_budget();
        if (background) enable_maintenance();
    }
    static std::shared_ptr<MemoryCache> automatic(double fraction = 0.5, double smoothing_seconds = 10)
    {
        AutoOptions options;
        options.fraction = fraction;
        options.smoothing_seconds = smoothing_seconds;
        return std::make_shared<MemoryCache>(std::move(options));
    }

    // Found: (true, chunk or nullptr for "missing"). Not cached: (false, nullptr).
    std::pair<bool, ChunkPtr> get(const std::string& key)
    {
        if (!maintenance_.load(std::memory_order_relaxed)) refresh_if_due();
        auto& shard = shard_for(key);
        TimedLock lock(*this, shard.mutex, shard.profile.get);
        auto* value = shard.entries.get(key);
        if (!value) { ++shard.misses; return {false, nullptr}; }
        ++shard.hits;
        value->epoch=epoch_.load(std::memory_order_relaxed);
        update_oldest(shard);
        return {true, value->chunk};
    }

    void put(const std::string& key, ChunkPtr value)
    {
        if (!maintenance_.load(std::memory_order_relaxed)) refresh_if_due();
        const std::size_t cost = value ? value->size() : kMissingCost;
        if (maintenance_.load(std::memory_order_relaxed)) {
            put_async(key,std::move(value),cost);
            return;
        }
        if (cost > budget_.load(std::memory_order_relaxed)) return;
        auto& shard = shard_for(key);
        {
            TimedLock lock(*this, shard.mutex, shard.profile.put);
            const auto before = shard.entries.bytes();
            try { shard.entries.put(key, Record{std::move(value),epoch_.load(std::memory_order_relaxed)}, cost); }
            catch (...) { account(before, shard.entries.bytes()); throw; }
            account(before, shard.entries.bytes());
            // Do not immediately reject a new key just because its shard was
            // empty when the shared budget filled. Evict elsewhere first.
            trim_locked(shard, cost);
            update_oldest(shard);
        }
        trim(&shard);
    }

    void clear()
    {
        for (auto& shard : shards_) {
            TimedLock lock(*this, shard->mutex, shard->profile.clear);
            const auto before = shard->entries.bytes();
            shard->entries.clear();
            update_oldest(*shard);
            account(before, 0);
        }
    }

    bool erase(const std::string& key)
    {
        auto& shard = shard_for(key);
        TimedLock lock(*this, shard.mutex, shard.profile.erase);
        const auto before = shard.entries.bytes();
        const auto removed = shard.entries.erase(key);
        update_oldest(shard);
        account(before, shard.entries.bytes());
        return removed;
    }

    std::size_t max_bytes() const { return budget_.load(std::memory_order_relaxed); }
    bool is_auto() const noexcept { return auto_.has_value(); }
    double fraction() const noexcept { return auto_ ? auto_->fraction : 0; }
    double smoothing_seconds() const noexcept { return auto_ ? auto_->smoothing_seconds : 0; }
    // Normally refreshed at most once a second during reads/writes. An explicit
    // refresh is useful before a new workload or after releasing other memory.
    void refresh_budget()
    {
        if (!auto_) return;
        std::lock_guard refresh_lock(refresh_mutex_);
        const bool profiling = profiling_.load(std::memory_order_relaxed);
        const auto probe_start = profiling ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto available = auto_->available(); // no I/O under the cache mutex
        const auto probe_ns = profiling ? elapsed_ns(probe_start, std::chrono::steady_clock::now()) : 0;
        const auto now = auto_->now();
        if (profiling) { ++profile_.refresh_calls; profile_.refresh_probe_ns += probe_ns; }
        const auto resident = nbytes();
        const long double target = auto_->fraction * (static_cast<long double>(available) + resident);
        if (!last_refresh_ || auto_->smoothing_seconds == 0) smoothed_budget_ = target;
        else {
            const auto elapsed = std::max(0.L,std::chrono::duration<long double>(now-*last_refresh_).count());
            const auto alpha = -std::expm1(-elapsed/auto_->smoothing_seconds);
            smoothed_budget_ += alpha*(target-smoothed_budget_);
        }
        // Smoothing must not leave a stale high growth allowance when another
        // workload takes memory. Keep the configured fraction of free headroom;
        // at zero available RAM this permits no further cache growth.
        const long double ceiling = resident + auto_->fraction*static_cast<long double>(available);
        smoothed_budget_ = std::min(smoothed_budget_,ceiling);
        budget_.store(static_cast<std::size_t>(std::min(smoothed_budget_,
            static_cast<long double>(std::numeric_limits<std::size_t>::max()))), std::memory_order_relaxed);
        if (maintenance_.load(std::memory_order_relaxed)) request_eviction();
        else trim();
        last_refresh_ = now;
        next_refresh_.store(milliseconds(now) + auto_->refresh_interval.count(), std::memory_order_relaxed);
    }
    std::size_t nbytes() const { return bytes_.load(std::memory_order_relaxed); }
    std::size_t size() { std::size_t n=0; for(auto& s:shards_) { std::lock_guard lock(s->mutex); n+=s->entries.size(); } return n; }
    std::uint64_t hits() { std::uint64_t n=0; for(auto& s:shards_) { std::lock_guard lock(s->mutex); n+=s->hits; } return n; }
    std::uint64_t misses() { std::uint64_t n=0; for(auto& s:shards_) { std::lock_guard lock(s->mutex); n+=s->misses; } return n; }

private:
    static std::uint64_t elapsed_ns(std::chrono::steady_clock::time_point a,
                                    std::chrono::steady_clock::time_point b)
    { return std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count(); }
    class TimedLock {
    public:
        TimedLock(MemoryCache& cache, std::mutex& mutex, LockTiming& timing)
            : lock_(mutex, std::defer_lock), timing_(timing),
              enabled_(cache.profiling_.load(std::memory_order_relaxed))
        {
            const auto start = enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            lock_.lock();
            if (enabled_) {
                acquired_=std::chrono::steady_clock::now();
                ++timing_.calls; timing_.wait_ns+=elapsed_ns(start,acquired_);
            }
        }
        ~TimedLock() {
            if(enabled_) timing_.hold_ns+=elapsed_ns(acquired_,std::chrono::steady_clock::now());
        }
    private:
        std::unique_lock<std::mutex> lock_;
        LockTiming& timing_;
        bool enabled_;
        std::chrono::steady_clock::time_point acquired_;
    };
    static std::int64_t milliseconds(std::chrono::steady_clock::time_point now)
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
    }
    void refresh_if_due()
    {
        if (!auto_) return;
        const auto now = milliseconds(auto_->now());
        auto next = next_refresh_.load(std::memory_order_relaxed);
        if (now < next || !next_refresh_.compare_exchange_strong(next,
                now + std::max<std::int64_t>(auto_->refresh_interval.count(), 1), std::memory_order_relaxed)) return;
        refresh_budget();
    }
    static constexpr std::size_t kMissingCost = 64;
    static constexpr std::uint64_t empty_epoch=std::numeric_limits<std::uint64_t>::max();
    struct Record { ChunkPtr chunk; std::uint64_t epoch; };
    struct alignas(64) Shard {
        mutable std::mutex mutex;
        Lru<Record> entries{std::numeric_limits<std::size_t>::max()};
        std::atomic<std::uint64_t>* published_epoch=nullptr;
        std::uint64_t last_published=empty_epoch;
        std::atomic<bool> vacuum_busy{false};
        std::uint64_t hits=0, misses=0;
        Profile profile;
    };
    static void update_oldest(Shard& shard) {
        const auto* oldest=shard.entries.oldest();
        const auto age=oldest?oldest->epoch:empty_epoch;
        if(age!=shard.last_published) {
            shard.published_epoch->store(age,std::memory_order_relaxed);
            shard.last_published=age;
        }
    }
    void init_shards(std::size_t count) {
        if (!count) throw std::invalid_argument("cache requires at least one shard");
        oldest_epochs_=std::make_unique<std::atomic<std::uint64_t>[]>(count);
        for (std::size_t i=0; i<count; ++i) {
            oldest_epochs_[i].store(empty_epoch,std::memory_order_relaxed);
            auto shard=std::make_unique<Shard>();
            shard->published_epoch=&oldest_epochs_[i];
            shards_.push_back(std::move(shard));
        }
    }
    struct EvictionCursor {
        struct Candidate { std::uint64_t epoch; std::size_t shard; };
        explicit EvictionCursor(std::size_t count): candidates(count) {}
        std::vector<Candidate> candidates;
        std::size_t next=0,end=0;
        std::uint64_t cutoff=0,epoch=0;
        std::uint64_t random=0;
    };
    void refill_candidates(EvictionCursor& cursor) {
        if(candidate_scan_) {
            // Independent bounded scans, without a shared RNG on each batch.
            if(!cursor.random)cursor.random=eviction_cursor_.fetch_add(1,std::memory_order_relaxed)+1;
            cursor.random^=cursor.random<<13;cursor.random^=cursor.random>>7;cursor.random^=cursor.random<<17;
            const auto count=shards_.size(),start=cursor.random%count;
            EvictionCursor::Candidate best{empty_epoch,0};
            for(std::size_t i=0;i<candidate_scan_;++i) {
                const auto shard=(start+i)%count;
                const auto age=oldest_epochs_[shard].load(std::memory_order_relaxed);
                if(age<best.epoch && !shards_[shard]->vacuum_busy.load(std::memory_order_relaxed))best={age,shard};
            }
            cursor.candidates[0]=best;cursor.next=0;cursor.end=best.epoch==empty_epoch?0:1;
            cursor.cutoff=best.epoch;cursor.epoch=epoch_.load(std::memory_order_relaxed);
            // Old neighbors need not share precisely the same millisecond.
            // Retain at least 7/8 of the selected tail's age for the whole batch.
            if(cursor.epoch>cursor.cutoff)cursor.cutoff+=(cursor.epoch-cursor.cutoff)/8;
            selection_scans_.fetch_add(1,std::memory_order_relaxed);
            return;
        }
        const auto count=shards_.size(), start=eviction_cursor_.fetch_add(16,std::memory_order_relaxed);
        for(std::size_t i=0;i<count;++i) {
            const auto shard=(start+i)%count;
            cursor.candidates[i]={oldest_epochs_[shard].load(std::memory_order_relaxed),shard};
        }
        cursor.next=0;cursor.end=std::min<std::size_t>(16,count);
        std::partial_sort(cursor.candidates.begin(),cursor.candidates.begin()+cursor.end,cursor.candidates.end(),
                          [](const auto& a,const auto& b){return a.epoch<b.epoch;});
        while(cursor.end && cursor.candidates[cursor.end-1].epoch==empty_epoch)--cursor.end;
        cursor.cutoff=cursor.end?cursor.candidates[cursor.end-1].epoch:0;
        cursor.epoch=epoch_.load(std::memory_order_relaxed);
        selection_scans_.fetch_add(1,std::memory_order_relaxed);
    }
    Shard& shard_for(const std::string& key) { return *shards_[std::hash<std::string>{}(key)%shards_.size()]; }
    void account(std::size_t before, std::size_t after) {
        if (after>before) bytes_.fetch_add(after-before, std::memory_order_relaxed);
        else if (before>after) bytes_.fetch_sub(before-after, std::memory_order_relaxed);
    }
    void trim_locked(Shard& shard, std::size_t keep = 0) {
        auto used=nbytes();
        while (shard.entries.size() && used>max_bytes()) {
            const auto cost=shard.entries.oldest_cost();
            if (shard.entries.bytes()-cost<keep) break;
            // Claim exactly one eviction against the shared byte count before
            // removing it. Independent shards must not all evict the same
            // observed excess, which would unnecessarily drain the cache.
            if (bytes_.compare_exchange_weak(used,used-cost,std::memory_order_relaxed)) {
                shard.entries.pop_oldest();
                update_oldest(shard);
                used=nbytes();
            }
        }
    }
    bool reserve_bytes(std::size_t cost) {
        auto used=nbytes();
        do {
            const auto limit=max_bytes();
            if(used>limit || cost>limit-used)return false;
        } while(!bytes_.compare_exchange_weak(used,used+cost,std::memory_order_relaxed));
        return true;
    }
    bool above_watermark() const {
        const auto budget=max_bytes(), head=std::min(effective_headroom(),budget);
        return nbytes()>budget-(head-head/4);
    }
    void request_eviction(bool inline_batch=false) {
        if(!above_watermark() || pressure_.load(std::memory_order_relaxed))return;
        if(inline_batch) {
            std::unique_lock inline_lock(inline_mutex_,std::try_to_lock);
            if(inline_lock.owns_lock() && !pressure_.load(std::memory_order_relaxed)) {
                // One nonblocking batch. Contention or insufficient progress
                // hands off to the background workers.
                thread_local EvictionCursor cursor(1);
                thread_local std::vector<Record> retired;
                retired.reserve(64);
                cursor.next=cursor.end;
                if(vacuum_batch(retired,cursor)) {
                    inline_batches_.fetch_add(1,std::memory_order_relaxed);
                    if(!above_watermark())return;
                }
            }
        }
        if(!pressure_.exchange(true))vacuum_cv_.notify_all();
    }
    std::size_t effective_headroom() const { return std::min(headroom_.load(std::memory_order_relaxed),max_bytes()/20); }
    std::size_t eviction_target() const {
        const auto budget=max_bytes();
        return budget-std::min(effective_headroom(),budget);
    }
    void put_async(const std::string& key, ChunkPtr value, std::size_t cost) {
        ChunkPtr previous; // Keep replaced payload destruction outside the lock.
        auto& shard=shard_for(key);
        std::size_t release=0;
        bool offered=false,assisted=false;
        std::vector<Record> retired;
        std::optional<EvictionCursor> cursor;
        for (;;) {
          std::size_t needed=0;
          {
            TimedLock lock(*this,shard.mutex,shard.profile.put);
            if(cost>max_bytes()) { bypasses_.fetch_add(1,std::memory_order_relaxed);return; }
            const auto epoch=epoch_.load(std::memory_order_relaxed);
            const auto [existing,old_cost,replaced]=shard.entries.replace_same_cost(key,cost,[&](Record& old) {
                previous=std::move(old.chunk);
                old={std::move(value),epoch};
            });
            if(replaced) {
                update_oldest(shard);
                return; // The lock unwinds before the previous payload is released.
            }
            const auto before=shard.entries.bytes();
            const auto extra=cost>old_cost?cost-old_cost:0;
            if(extra && !offered) { offered_.fetch_add(extra,std::memory_order_relaxed);offered=true; }
            if(extra && !reserve_bytes(extra)) {
                needed=extra;
            } else {
            if(existing)previous=existing->chunk;
            try { shard.entries.put(key,Record{std::move(value),epoch_.load(std::memory_order_relaxed)},cost); }
            catch (...) {
                account(before+extra,shard.entries.bytes());
                update_oldest(shard);
                throw;
            }
            release=old_cost>cost?old_cost-cost:0;
            update_oldest(shard);
            if(extra)admitted_.fetch_add(extra,std::memory_order_relaxed);
            }
          }
          if(!needed)break;
          // Help only after releasing the insertion shard. Recheck the key and
          // reservation on retry: another writer may replace it in the meantime.
          if(!assisted) {
              assisted=true;retired.reserve(64);cursor.emplace(candidate_scan_?1:shards_.size());
              assist_puts_.fetch_add(1,std::memory_order_relaxed);
              pressure_events_.fetch_add(1,std::memory_order_relaxed);
          }
          request_eviction();
          const auto freed=vacuum_batch(retired,*cursor,needed);
          if(freed) {
              assist_bytes_.fetch_add(freed,std::memory_order_relaxed);
              assist_batches_.fetch_add(1,std::memory_order_relaxed);
          } else std::this_thread::yield();
        }
        previous.reset();
        if(release)bytes_.fetch_sub(release,std::memory_order_relaxed);
        request_eviction(candidate_scan_!=0);
    }
    std::size_t vacuum_batch(std::vector<Record>& retired, EvictionCursor& cursor, std::size_t required_free=0) {
        if(cursor.next==cursor.end || cursor.epoch!=epoch_.load(std::memory_order_relaxed))refill_candidates(cursor);
        std::size_t released=0;
        while(cursor.next<cursor.end && retired.size()<64 && released<16*1024*1024) {
            auto& selected=*shards_[cursor.candidates[cursor.next++].shard];
            if(selected.vacuum_busy.exchange(true))continue;
            std::unique_lock lock(selected.mutex,std::try_to_lock);
            if(lock.owns_lock()) {
                auto pending=reclaiming_.load(std::memory_order_relaxed);
                for (;;) {
                    const auto used=nbytes(),budget=max_bytes();
                    const auto target=std::min(eviction_target(),budget-std::min(required_free,budget));
                    if(used<=target || pending>=used-target)break;
                    // Revalidate the entire prefix under its shard lock. One
                    // claim covers this prefix, not one atomic claim per entry.
                    const auto age=cursor.epoch>cursor.cutoff?cursor.epoch-cursor.cutoff:0;
                    const auto entry_limit=candidate_scan_?std::min<std::size_t>(64,1+4*std::min<std::uint64_t>(age,16)):64;
                    const auto plan=selected.entries.oldest_batch(std::min(entry_limit,64-retired.size()),
                        std::min(used-target-pending,16*1024*1024-released),
                        [&](const Record& entry){return entry.epoch<=cursor.cutoff;});
                    if(!plan.first)break;
                    if(!reclaiming_.compare_exchange_weak(pending,pending+plan.second,std::memory_order_relaxed))continue;
                    for(std::size_t i=0;i<plan.first;++i)retired.push_back(selected.entries.take_oldest());
                    released+=plan.second;
                    reclaim_claims_.fetch_add(1,std::memory_order_relaxed);
                    break;
                }
                update_oldest(selected);
            }
            if(lock.owns_lock())lock.unlock();
            selected.vacuum_busy.store(false);
        }
        // Payloads are released without holding a shard lock. Their reservation
        // remains charged until destruction completes (external refs may live on).
        const auto entries=retired.size();
        retired.clear();
        if(released) {
            bytes_.fetch_sub(released,std::memory_order_relaxed);
            reclaiming_.fetch_sub(released,std::memory_order_relaxed);
            evicted_.fetch_add(released,std::memory_order_relaxed);
            batches_.fetch_add(1,std::memory_order_relaxed);
            evicted_entries_.fetch_add(entries,std::memory_order_relaxed);
        }
        return released;
    }
    void vacuum_loop(std::size_t worker) {
        using Clock=std::chrono::steady_clock;
        std::vector<Record> retired;retired.reserve(64);
        EvictionCursor cursor(candidate_scan_?1:shards_.size());
        auto last_control=Clock::now();
        std::uint64_t previous_offered=0,previous_bypasses=0;
        double rate=0,recovery_seconds=.002;
        std::optional<Clock::time_point> recovering;
        while(!stopping_.load()) {
            const auto now=Clock::now();
            if(worker==0) {
                // One periodic writer; hits read this epoch rather than causing
                // a global atomic increment for every access.
                const auto epoch=std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()+1;
                if(epoch_.load(std::memory_order_relaxed)!=static_cast<std::uint64_t>(epoch))
                    epoch_.store(epoch,std::memory_order_relaxed);
                refresh_if_due();
                if(pressure_.load() && !recovering)recovering=now;
                if(pressure_.load() && nbytes()<=eviction_target() && !reclaiming_.load()) {
                    pressure_.store(false);
                    if(recovering) {
                        recovery_seconds=.8*recovery_seconds+.2*std::chrono::duration<double>(now-*recovering).count();
                        recovering.reset();
                    }
                    // An insertion may have crossed the high watermark just
                    // before pressure was cleared; do not lose that wakeup.
                    request_eviction();
                }
                const double elapsed=std::chrono::duration<double>(now-last_control).count();
                if(elapsed>=.01) {
                    const auto offered=offered_.load(),bypassed=pressure_events_.load(),budget=max_bytes();
                    const auto alpha=-std::expm1(-elapsed/.1);
                    rate+=alpha*((offered-previous_offered)/elapsed-rate);
                    const double cap=budget*.05, floor=std::min(cap,std::max(64.,budget*.001));
                    double desired=std::clamp(rate*(.002+recovery_seconds)*2,floor,cap);
                    const auto current=headroom_.load();
                    if(bypassed>previous_bypasses)desired=std::min(cap,std::max(desired,current*1.5+64));
                    const double blend=-std::expm1(-elapsed/(desired>current?.05:2.));
                    headroom_.store(static_cast<std::size_t>(std::clamp(current+blend*(desired-current),floor,cap)));
                    previous_offered=offered;previous_bypasses=bypassed;last_control=now;
                    request_eviction();
                }
            }
            if(pressure_.load()) {
                if(!vacuum_batch(retired,cursor))std::this_thread::yield();
                continue;
            }
            std::unique_lock lock(vacuum_mutex_);
            vacuum_cv_.wait_for(lock,std::chrono::milliseconds(1),[&]{return stopping_.load()||pressure_.load();});
        }
    }
    void stop_maintenance() {
        stopping_.store(true);vacuum_cv_.notify_all();
        for(auto& worker:vacuum_threads_)if(worker.joinable())worker.join();
    }
    void trim(Shard* preferred = nullptr) {
        if (nbytes()<=max_bytes()) return;
        // Never hold two shard locks. No per-shard quota strands unused budget.
        const auto start=eviction_cursor_.fetch_add(1,std::memory_order_relaxed);
        for(std::size_t i=0;i<shards_.size() && nbytes()>max_bytes();++i) {
            auto& shard=*shards_[(start+i)%shards_.size()];
            if (&shard==preferred) continue;
            std::lock_guard lock(shard.mutex);
            trim_locked(shard);
        }
        // A simultaneous auto-budget shrink can make even the incoming chunk
        // too large. In that case it must also be eligible for eviction.
        if (preferred && nbytes()>max_bytes()) {
            std::lock_guard lock(preferred->mutex);
            trim_locked(*preferred);
        }
    }
    std::unique_ptr<std::atomic<std::uint64_t>[]> oldest_epochs_;
    std::vector<std::unique_ptr<Shard>> shards_;
    std::size_t candidate_scan_=0; // Zero: full-directory cohort; nonzero: sampled tails.
    std::mutex inline_mutex_;
    std::atomic<std::uint64_t> inline_batches_{0};
    alignas(64) std::atomic<std::size_t> bytes_{0};
    alignas(64) std::atomic<std::size_t> budget_{0};
    std::atomic<std::size_t> eviction_cursor_{0};
    std::atomic<bool> profiling_{false};
    Profile profile_;
    std::optional<AutoOptions> auto_;
    mutable std::mutex refresh_mutex_;
    std::optional<std::chrono::steady_clock::time_point> last_refresh_;
    long double smoothed_budget_ = 0;
    std::atomic<std::int64_t> next_refresh_{0};
    std::atomic<bool> maintenance_{false},stopping_{false},pressure_{false};
    std::atomic<std::uint64_t> epoch_{1},bypasses_{0},admitted_{0},evicted_{0},batches_{0};
    std::atomic<std::uint64_t> offered_{0},pressure_events_{0};
    std::atomic<std::uint64_t> assist_puts_{0},assist_bytes_{0},assist_batches_{0};
    std::atomic<std::uint64_t> selection_scans_{0},reclaim_claims_{0},evicted_entries_{0};
    std::atomic<std::size_t> headroom_{0},reclaiming_{0};
    std::mutex vacuum_mutex_;
    std::condition_variable vacuum_cv_;
    std::vector<std::thread> vacuum_threads_;
};

}  // namespace vcr
