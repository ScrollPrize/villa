#include <vc_reader/native/memory_cache.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <limits>

void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
int main()
{
    try {
        require(vcr::memory::cgroup_available(1000, 700, 100) == 400,
                "cgroup headroom includes reclaimable file cache");
        require(vcr::memory::cgroup_available(1000, 1500, 100) == 0,
                "over-limit cgroup cannot underflow");
        require(vcr::memory::cgroup_available(1000, 100, 200) == 1000,
                "racing reclaimable counter cannot underflow usage");
        vcr::MemoryCache fixed(100,1,false); // Exact synchronous LRU control.
        require(!fixed.is_auto() && fixed.max_bytes() == 100, "fixed mode unchanged");
        fixed.put("a", std::make_shared<vcr::Chunk>(60));
        fixed.put("b", std::make_shared<vcr::Chunk>(60));
        require(!fixed.get("a").first && fixed.get("b").first, "fixed LRU eviction");
        require(fixed.erase("b") && !fixed.erase("b") && fixed.nbytes() == 0 && fixed.size() == 0,
                "erase removes entry and byte accounting exactly once");
        require(fixed.profile().get.calls == 0 && fixed.profile().put.calls == 0,
                "profiling disabled by default");
        std::uint64_t available = 1000;
        vcr::MemoryCache::AutoOptions options;
        options.available = [&] { return available; };
        options.refresh_interval = std::chrono::hours(1);
        options.smoothing_seconds = 0; // exact unsmoothed arithmetic/eviction checks
        vcr::MemoryCache cache(options,1,false);
        require(cache.is_auto() && cache.fraction() == .5 && cache.max_bytes() == 500, "default 50 percent");
        cache.put("a", std::make_shared<vcr::Chunk>(200));
        available = 800;
        cache.refresh_budget();
        require(cache.max_bytes() == 500, "own resident cache added back, no shrinking oscillation");
        cache.put("b", std::make_shared<vcr::Chunk>(200));
        available = 600;
        cache.refresh_budget();
        require(cache.max_bytes() == 500, "stable after further cache growth");
        available = 0;
        cache.refresh_budget();
        require(cache.max_bytes() == 200 && cache.nbytes() == 200 && !cache.get("a").first,
                "memory pressure evicts LRU to reduced budget");
        available = 800;
        cache.refresh_budget();
        require(cache.max_bytes() == 500, "budget recovers after pressure");
        cache.clear();
        available = 1000;
        cache.refresh_budget();
        require(cache.nbytes() == 0 && cache.max_bytes() == 500, "clear accounts resident bytes");
        for (double fraction : {0., 1., -1., std::numeric_limits<double>::quiet_NaN()}) {
            auto invalid = options;
            invalid.fraction = fraction;
            bool rejected = false;
            try { vcr::MemoryCache other(invalid); } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "reject invalid fractions");
        }
        // At 50%, two full caches share 2/3 of initially available RAM: each sees
        // the other's occupancy as unavailable, but adds its own occupancy back.
        std::size_t first = 0, second = 0;
        for (int round = 0; round < 30; ++round) {
            first = (900-second)/2;
            second = (900-first)/2;
        }
        require(first == 300 && second == 300, "two-cache equilibrium");
        {
            using Clock = std::chrono::steady_clock;
            auto now = Clock::time_point{};
            std::uint64_t free = 1000;
            vcr::MemoryCache::AutoOptions ema;
            ema.available = [&] { return free; };
            ema.now = [&] { return now; };
            ema.refresh_interval = std::chrono::hours(1);
            vcr::MemoryCache smooth(ema,1,false);
            require(smooth.max_bytes() == 500 && smooth.smoothing_seconds() == 10,
                "EMA initial budget is immediate, default time constant ten seconds");
            for (int i=0;i<4;++i) smooth.put(std::to_string(i),std::make_shared<vcr::Chunk>(100));
            free = 200; // another workload consumes memory; target becomes 300
            smooth.refresh_budget();
            require(smooth.max_bytes() == 500,"zero elapsed time does not advance EMA");
            now += std::chrono::seconds(10);
            smooth.refresh_budget();
            const auto expected = std::size_t(300 + 200*std::exp(-1.));
            require(smooth.max_bytes() == expected && smooth.nbytes() == 300,
                "time-based EMA softens decrease and evicts to resulting watermark");
            free = 0;
            smooth.refresh_budget();
            require(smooth.max_bytes() == 300,"headroom cap prevents growth despite stale EMA");
            now += std::chrono::seconds(10);
            smooth.refresh_budget();
            require(smooth.max_bytes() == std::size_t(150 + 150*std::exp(-1.)) && smooth.nbytes() == 200,
                "zero available memory continues smooth release of resident cache");
            free = 2000;
            now += std::chrono::seconds(1);
            smooth.refresh_budget();
            require(smooth.max_bytes() > 205 && smooth.max_bytes() < 300,
                "memory recovery grows budget gradually rather than jumping to target");

            // Refresh frequency does not alter the EMA for a constant target.
            now = Clock::time_point{};
            free = 1000;
            vcr::MemoryCache one_step(ema,1,false), ten_steps(ema,1,false);
            free = 2000;
            for (int i=0;i<10;++i) {
                now += std::chrono::seconds(1);
                ten_steps.refresh_budget();
            }
            one_step.refresh_budget();
            require(one_step.max_bytes() == ten_steps.max_bytes() && one_step.max_bytes() == 816,
                "EMA is elapsed-time based, not refresh-count based");

            // Two active caches compete for one available-memory pool. Fill
            // caches to their current budgets, then let observed free RAM react.
            now = Clock::time_point{};
            std::unique_ptr<vcr::MemoryCache> a,b;
            constexpr std::size_t total = 90000;
            ema.available = [&] { return total-(a ? a->nbytes() : 0)-(b ? b->nbytes() : 0); };
            a = std::make_unique<vcr::MemoryCache>(ema,1,false);
            b = std::make_unique<vcr::MemoryCache>(ema,1,false);
            std::uint64_t serial = 0;
            const auto fill = [&](vcr::MemoryCache& cache) {
                while (cache.nbytes()+10 <= cache.max_bytes())
                    cache.put(std::to_string(serial++),std::make_shared<vcr::Chunk>(10));
            };
            fill(*a); fill(*b);
            require(a->nbytes() == 45000 && b->nbytes() == 45000,"initial competing cache fill");
            now += std::chrono::seconds(1);
            a->refresh_budget(); b->refresh_budget();
            require(a->max_bytes() > 42000 && b->max_bytes() > 42000,
                "competing caches do not abruptly halve their watermarks");
            for (int i=0;i<300;++i) {
                now += std::chrono::seconds(1);
                a->refresh_budget(); b->refresh_budget();
                fill(*a); fill(*b);
            }
            require(std::abs(std::int64_t(a->nbytes())-30000) <= 30 &&
                    std::abs(std::int64_t(b->nbytes())-30000) <= 30,
                "competing EMA caches converge to roughly one third each");
            for (double tau : {-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
                auto invalid = options;
                invalid.smoothing_seconds = tau;
                bool rejected = false;
                try { vcr::MemoryCache bad(invalid); } catch (const std::invalid_argument&) { rejected = true; }
                require(rejected,"reject invalid smoothing time constants");
            }
        }
        // Concurrent reads and forced refreshes exercise the mutable limit lock.
        std::atomic<int> calls{0};
        options.available = [&] { ++calls; return 1000; };
        options.refresh_interval = std::chrono::milliseconds(0);
        vcr::MemoryCache concurrent(options);
        concurrent.enable_profiling();
        auto work = [&] {
            for (int i = 0; i < 1000; ++i) {
                concurrent.put("x", std::make_shared<vcr::Chunk>(20));
                (void)concurrent.get("x");
                (void)concurrent.max_bytes();
                if (i % 100 == 0) concurrent.refresh_budget();
            }
        };
        std::thread one(work), two(work);
        one.join(); two.join();
        const auto profile=concurrent.profile();
        require(profile.get.calls==2000 && profile.put.calls==2000 && profile.refresh_calls>=20,
                "profile counts concurrent operations without losing updates");
        require(profile.get.hold_ns>0 && profile.put.wait_ns>0 && profile.refresh_probe_ns>0,
                "profile separates lock wait, hold and memory probing");
        require(calls > 2 && concurrent.nbytes() <= concurrent.max_bytes(), "concurrent refresh");
        // Shared budget, not a fixed quota per shard: even one hot key may use
        // the whole budget. Concurrent replacement/erase must conserve bytes.
        for (const auto shards : {1u, 8u, 64u}) {
            vcr::MemoryCache sharded(4096,shards,false);
            sharded.put("large",std::make_shared<vcr::Chunk>(4096));
            require(sharded.get("large").first && sharded.nbytes()==4096,"no per-shard quota");
            for(int i=0;i<32;++i) {
                const auto key="large"+std::to_string(i);
                sharded.put(key,std::make_shared<vcr::Chunk>(4096));
                require(sharded.get(key).first && sharded.nbytes()==4096,
                        "new keys in empty shards can replace a full cache");
            }
            sharded.clear();
            std::vector<std::thread> threads;
            for (int t=0;t<8;++t) threads.emplace_back([&,t] {
                for(int i=0;i<2000;++i) {
                    const auto key=std::to_string((i*13+t)%128);
                    sharded.put(key,std::make_shared<vcr::Chunk>(32));
                    sharded.get(key);
                    if(i%3==0)sharded.erase(key);
                }
            });
            for(auto& thread:threads)thread.join();
            require(sharded.nbytes()==sharded.size()*32 && sharded.nbytes()<=4096,"sharded concurrent accounting");
            sharded.clear();
            require(sharded.nbytes()==0 && sharded.size()==0,"sharded clear accounting");
            sharded.put("fill",nullptr);
            require(sharded.get("fill").first && !sharded.get("fill").second && sharded.nbytes()==64,
                    "sharded negative entry has bounded accounting");
            sharded.put("fill",std::make_shared<vcr::Chunk>(19));
            require(sharded.nbytes()==19,"replacement subtracts previous cost");
            sharded.erase("fill");
            require(sharded.nbytes()==0,"variable-size erase accounting");
            threads.clear();
            for(int t=0;t<8;++t) threads.emplace_back([&,t] {
                for(int i=0;i<2000;++i)
                    sharded.put(std::to_string(t*2000+i),std::make_shared<vcr::Chunk>(32));
            });
            for(auto& thread:threads)thread.join();
            require(sharded.nbytes()==4096 && sharded.size()==128,
                    "concurrent eviction claims do not multiply the global excess");
        }
        {
            std::atomic<std::uint64_t> free{8192};
            auto pressure=options;
            pressure.available=[&]{return free.load();};
            pressure.refresh_interval=std::chrono::hours(1);
            vcr::MemoryCache cache(pressure,32,false);
            std::vector<std::thread> writers;
            for(int t=0;t<8;++t) writers.emplace_back([&,t] {
                for(int i=0;i<4000;++i) {
                    cache.put(std::to_string((i+t*19)%1024),std::make_shared<vcr::Chunk>(32));
                    if(i%7==0) cache.erase(std::to_string((i*7+t)%1024));
                }
            });
            for(int i=0;i<100;++i) { free=i%2?8192:0; cache.refresh_budget(); }
            for(auto& writer:writers) writer.join();
            require(cache.nbytes()==cache.size()*32 && cache.nbytes()<=cache.max_bytes(),
                    "concurrent multi-shard eviction and budget shrink accounting");
        }
        auto system = vcr::MemoryCache::automatic();
        require(system->max_bytes() > 0, "platform available memory probe");
        std::cout << "Auto-cache stability, pressure, fixed mode, validation, concurrency and system probe passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
