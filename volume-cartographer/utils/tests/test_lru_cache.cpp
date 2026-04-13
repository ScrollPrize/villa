#include <utils/test.hpp>
#include <utils/lru_cache.hpp>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <cstddef>

UTILS_TEST_MAIN()

TEST_CASE("LRUCache: basic put and get") {
    utils::LRUCache<int, std::string> cache;
    cache.put(1, "one");
    cache.put(2, "two");

    auto v1 = cache.get(1);
    REQUIRE(v1.has_value());
    REQUIRE_EQ(v1.value(), std::string("one"));

    auto v2 = cache.get(2);
    REQUIRE(v2.has_value());
    REQUIRE_EQ(v2.value(), std::string("two"));
}

TEST_CASE("LRUCache: miss returns nullopt") {
    utils::LRUCache<int, int> cache;
    cache.put(1, 100);
    auto v = cache.get(42);
    REQUIRE(!v.has_value());
}

TEST_CASE("LRUCache: contains and remove") {
    utils::LRUCache<int, int> cache;
    cache.put(1, 10);
    REQUIRE(cache.contains(1));
    REQUIRE(!cache.contains(2));

    bool removed = cache.remove(1);
    REQUIRE(removed);
    REQUIRE(!cache.contains(1));
    REQUIRE(!cache.remove(1));
}

TEST_CASE("LRUCache: eviction when over budget") {
    utils::LRUCache<int, int>::Config cfg;
    cfg.max_bytes = sizeof(int) * 3;  // room for ~3 entries
    cfg.size_fn = [](const int&) -> std::size_t { return sizeof(int); };
    utils::LRUCache<int, int> cache(cfg);

    cache.put(1, 10);
    cache.put(2, 20);
    cache.put(3, 30);
    REQUIRE_EQ(cache.size(), 3u);

    // Adding a 4th should trigger eviction of the oldest
    cache.put(4, 40);
    // After eviction, some entries should have been removed
    CHECK_LE(cache.byte_size(), cfg.max_bytes);
}

TEST_CASE("LRUCache: pinned entries survive eviction") {
    utils::LRUCache<int, int>::Config cfg;
    cfg.max_bytes = sizeof(int) * 2;
    cfg.size_fn = [](const int&) -> std::size_t { return sizeof(int); };
    utils::LRUCache<int, int> cache(cfg);

    cache.put_pinned(1, 100);
    cache.put(2, 200);

    // Force eviction by adding more
    cache.put(3, 300);
    cache.put(4, 400);

    // Pinned entry should still be present
    auto v = cache.get(1);
    REQUIRE(v.has_value());
    REQUIRE_EQ(v.value(), 100);
}

TEST_CASE("LRUCache: generation-based LRU ordering") {
    utils::LRUCache<int, int>::Config cfg;
    cfg.max_bytes = sizeof(int) * 3;
    cfg.size_fn = [](const int&) -> std::size_t { return sizeof(int); };
    utils::LRUCache<int, int> cache(cfg);

    cache.put(1, 10);
    cache.put(2, 20);
    cache.put(3, 30);

    // Touch key 1 so it becomes most recently used
    (void)cache.get(1);

    // Insert key 4 -- should evict key 2 (oldest generation, not 1)
    cache.put(4, 40);

    CHECK(cache.contains(1));   // was promoted by get()
    CHECK(!cache.contains(2));  // oldest, should be evicted
}

TEST_CASE("LRUCache: stats tracking") {
    utils::LRUCache<int, int>::Config cfg;
    cfg.max_bytes = sizeof(int) * 2;
    cfg.size_fn = [](const int&) -> std::size_t { return sizeof(int); };
    utils::LRUCache<int, int> cache(cfg);

    cache.put(1, 10);
    cache.put(2, 20);

    (void)cache.get(1);  // hit
    (void)cache.get(1);  // hit
    (void)cache.get(99); // miss

    REQUIRE_EQ(cache.hits(), 2u);
    REQUIRE_EQ(cache.misses(), 1u);

    // Trigger eviction
    cache.put(3, 30);
    CHECK_GT(cache.evictions(), 0u);
}

TEST_CASE("LRUCache: missing_keys batch operation") {
    utils::LRUCache<int, int> cache;
    cache.put(1, 10);
    cache.put(3, 30);
    cache.put(5, 50);

    std::vector<int> query = {1, 2, 3, 4, 5, 6};
    auto missing = cache.missing_keys(query.begin(), query.end());

    REQUIRE_EQ(missing.size(), 3u);
    // Should contain 2, 4, 6
    CHECK(std::find(missing.begin(), missing.end(), 2) != missing.end());
    CHECK(std::find(missing.begin(), missing.end(), 4) != missing.end());
    CHECK(std::find(missing.begin(), missing.end(), 6) != missing.end());
}

TEST_CASE("LRUCache: for_each iteration") {
    utils::LRUCache<int, int> cache;
    cache.put(1, 10);
    cache.put(2, 20);
    cache.put(3, 30);

    int sum = 0;
    int count = 0;
    cache.for_each([&](const int& /*k*/, const int& v) {
        sum += v;
        ++count;
    });

    REQUIRE_EQ(count, 3);
    REQUIRE_EQ(sum, 60);
}

TEST_CASE("LRUCache: clear") {
    utils::LRUCache<int, int> cache;
    cache.put(1, 10);
    cache.put(2, 20);
    cache.clear();
    REQUIRE_EQ(cache.size(), 0u);
    REQUIRE_EQ(cache.byte_size(), 0u);
    REQUIRE(!cache.get(1).has_value());
}

TEST_CASE("LRUCache: concurrent reads") {
    utils::LRUCache<int, int> cache;
    for (int i = 0; i < 100; ++i) {
        cache.put(i, i * 10);
    }

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < 100; ++i) {
                auto v = cache.get(i);
                CHECK(v.has_value());
            }
        });
    }
    for (auto& th : threads) th.join();

    REQUIRE_EQ(cache.hits(), 400u);
}

TEST_CASE("LRUCache: concurrent writes trigger eviction") {
    utils::LRUCache<int, int>::Config cfg;
    cfg.max_bytes = sizeof(int) * 50;
    cfg.size_fn = [](const int&) -> std::size_t { return sizeof(int); };
    utils::LRUCache<int, int> cache(cfg);

    // Multiple threads inserting concurrently to trigger eviction paths.
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 50; ++i) {
                cache.put(t * 1000 + i, i);
            }
        });
    }
    for (auto& th : threads) th.join();

    // 200 total inserts into a cache that holds ~50 entries; evictions must have occurred.
    CHECK_GT(cache.evictions(), 0u);
    CHECK_LE(cache.byte_size(), cfg.max_bytes);
}

TEST_CASE("LRUCache: concurrent mixed reads and writes") {
    utils::LRUCache<int, int>::Config cfg;
    cfg.max_bytes = sizeof(int) * 100;
    cfg.size_fn = [](const int&) -> std::size_t { return sizeof(int); };
    utils::LRUCache<int, int> cache(cfg);

    // Pre-populate
    for (int i = 0; i < 50; ++i) {
        cache.put(i, i);
    }

    std::atomic<bool> go{false};
    std::vector<std::thread> threads;

    // Readers
    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&] {
            while (!go.load(std::memory_order_acquire)) {}
            for (int i = 0; i < 100; ++i) {
                (void)cache.get(i % 50);
            }
        });
    }

    // Writers
    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&, t] {
            while (!go.load(std::memory_order_acquire)) {}
            for (int i = 0; i < 100; ++i) {
                cache.put(100 + t * 1000 + i, i);
            }
        });
    }

    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    // Cache should still be in a valid state.
    CHECK_LE(cache.byte_size(), cfg.max_bytes);
    CHECK_GT(cache.size(), 0u);
}

TEST_CASE("LRUCache: eviction respects evict_ratio hysteresis") {
    utils::LRUCache<int, int>::Config cfg;
    cfg.max_bytes = sizeof(int) * 4;
    cfg.evict_ratio = 0.5;  // target = 50% of max = 2 entries worth
    cfg.size_fn = [](const int&) -> std::size_t { return sizeof(int); };
    utils::LRUCache<int, int> cache(cfg);

    cache.put(1, 10);
    cache.put(2, 20);
    cache.put(3, 30);
    cache.put(4, 40);
    // At 4 entries (16 bytes) == max, no eviction yet.
    // Adding 5th triggers eviction down to target (8 bytes = 2 entries).
    cache.put(5, 50);

    // After eviction, byte_size should be at or below target.
    auto target = static_cast<std::size_t>(cfg.max_bytes * cfg.evict_ratio);
    CHECK_LE(cache.byte_size(), target + sizeof(int)); // +1 entry for the newly inserted
    CHECK_GT(cache.evictions(), 0u);
}

TEST_CASE("LRUCache: custom size_fn is used") {
    utils::LRUCache<int, std::string>::Config cfg;
    cfg.max_bytes = 20;
    cfg.size_fn = [](const std::string& s) -> std::size_t { return s.size(); };
    utils::LRUCache<int, std::string> cache(cfg);

    cache.put(1, "aaaaaaaaaa"); // 10 bytes
    REQUIRE_EQ(cache.byte_size(), 10u);

    cache.put(2, "bbbbb"); // 5 bytes
    REQUIRE_EQ(cache.byte_size(), 15u);
}

TEST_CASE("LRUCache: miss increments miss counter") {
    utils::LRUCache<int, int> cache;
    (void)cache.get(1);
    (void)cache.get(2);
    (void)cache.get(3);
    REQUIRE_EQ(cache.misses(), 3u);
    REQUIRE_EQ(cache.hits(), 0u);
}

TEST_CASE("LRUCache: update existing key updates bytes") {
    utils::LRUCache<int, std::string>::Config cfg;
    cfg.max_bytes = 1024;
    cfg.size_fn = [](const std::string& s) -> std::size_t { return s.size(); };
    utils::LRUCache<int, std::string> cache(cfg);

    cache.put(1, "short");    // 5 bytes
    REQUIRE_EQ(cache.byte_size(), 5u);

    cache.put(1, "a longer string"); // 15 bytes -- update existing
    REQUIRE_EQ(cache.byte_size(), 15u);
    REQUIRE_EQ(cache.size(), 1u);

    auto v = cache.get(1);
    REQUIRE(v.has_value());
    REQUIRE_EQ(*v, std::string("a longer string"));
}

TEST_CASE("LRUCache: pinned entries not evicted even under heavy pressure") {
    utils::LRUCache<int, int>::Config cfg;
    cfg.max_bytes = sizeof(int) * 3;
    cfg.evict_ratio = 0.5;
    cfg.size_fn = [](const int&) -> std::size_t { return sizeof(int); };
    utils::LRUCache<int, int> cache(cfg);

    // Pin two entries
    cache.put_pinned(1, 100);
    cache.put_pinned(2, 200);
    cache.put(3, 300);

    // Force many evictions
    for (int i = 10; i < 20; ++i) {
        cache.put(i, i * 10);
    }

    // Pinned entries must survive
    REQUIRE(cache.get(1).has_value());
    REQUIRE_EQ(cache.get(1).value(), 100);
    REQUIRE(cache.get(2).has_value());
    REQUIRE_EQ(cache.get(2).value(), 200);
}

TEST_CASE("LRUCache: for_each with concurrent modification safety") {
    utils::LRUCache<int, int> cache;
    for (int i = 0; i < 10; ++i) {
        cache.put(i, i * 10);
    }

    // for_each holds shared lock -- verify it works.
    std::vector<std::pair<int, int>> entries;
    cache.for_each([&](const int& k, const int& v) {
        entries.emplace_back(k, v);
    });
    REQUIRE_EQ(entries.size(), 10u);
}

// -- Value type with byte_size() method for HasByteSize concept path --
struct SizedValue {
    std::vector<char> data;
    [[nodiscard]] std::size_t byte_size() const { return data.size(); }
};

TEST_CASE("LRUCache: HasByteSize concept path used when no size_fn") {
    // No size_fn provided -- should use v.byte_size() for SizedValue
    utils::LRUCache<int, SizedValue>::Config cfg;
    cfg.max_bytes = 100;
    // Intentionally NOT setting cfg.size_fn
    utils::LRUCache<int, SizedValue> cache(cfg);

    SizedValue v1;
    v1.data.resize(30, 'a');
    cache.put(1, v1);
    REQUIRE_EQ(cache.byte_size(), 30u);

    SizedValue v2;
    v2.data.resize(50, 'b');
    cache.put(2, v2);
    REQUIRE_EQ(cache.byte_size(), 80u);
}

TEST_CASE("LRUCache: HasByteSize eviction path") {
    utils::LRUCache<int, SizedValue>::Config cfg;
    cfg.max_bytes = 60;
    cfg.evict_ratio = 0.5; // target = 30 bytes
    utils::LRUCache<int, SizedValue> cache(cfg);

    SizedValue v1;
    v1.data.resize(20, 'a');
    cache.put(1, v1);

    SizedValue v2;
    v2.data.resize(20, 'b');
    cache.put(2, v2);

    SizedValue v3;
    v3.data.resize(20, 'c');
    cache.put(3, v3);
    // Now at 60 bytes == max, no eviction yet

    SizedValue v4;
    v4.data.resize(20, 'd');
    cache.put(4, v4);
    // Now over budget, should trigger eviction

    CHECK_GT(cache.evictions(), 0u);
    // After eviction, should be at or below target + one entry
    CHECK_LE(cache.byte_size(), 60u);
}

TEST_CASE("LRUCache: update existing key with HasByteSize type") {
    utils::LRUCache<int, SizedValue>::Config cfg;
    cfg.max_bytes = 1024;
    utils::LRUCache<int, SizedValue> cache(cfg);

    SizedValue v1;
    v1.data.resize(10, 'a');
    cache.put(1, v1);
    REQUIRE_EQ(cache.byte_size(), 10u);

    // Update with larger value
    SizedValue v2;
    v2.data.resize(50, 'b');
    cache.put(1, v2);
    REQUIRE_EQ(cache.byte_size(), 50u);
    REQUIRE_EQ(cache.size(), 1u);
}

TEST_CASE("LRUCache: sizeof(V) fallback when no size_fn and no byte_size") {
    // int has no byte_size() and no size_fn => defaults to sizeof(int)
    utils::LRUCache<int, int>::Config cfg;
    cfg.max_bytes = 1024;
    // No size_fn set
    utils::LRUCache<int, int> cache(cfg);

    cache.put(1, 42);
    REQUIRE_EQ(cache.byte_size(), sizeof(int));

    cache.put(2, 43);
    REQUIRE_EQ(cache.byte_size(), sizeof(int) * 2);
}

TEST_CASE("LRUCache: max_bytes accessor") {
    utils::LRUCache<int, int>::Config cfg;
    cfg.max_bytes = 12345;
    utils::LRUCache<int, int> cache(cfg);
    REQUIRE_EQ(cache.max_bytes(), 12345u);
}

TEST_CASE("LRUCache: put_pinned then update to non-pinned") {
    utils::LRUCache<int, int>::Config cfg;
    cfg.max_bytes = sizeof(int) * 3;
    cfg.size_fn = [](const int&) -> std::size_t { return sizeof(int); };
    utils::LRUCache<int, int> cache(cfg);

    cache.put_pinned(1, 100);
    cache.put(2, 200);
    cache.put(3, 300);

    // Update key 1 to non-pinned
    cache.put(1, 101);

    // Now add more entries to trigger eviction
    cache.put(4, 400);
    cache.put(5, 500);

    // Key 1 is no longer pinned, so it may be evicted
    // We just check the cache is in a valid state
    CHECK_LE(cache.byte_size(), cfg.max_bytes + sizeof(int));
}

TEST_CASE("LRUCache: eviction with all entries pinned") {
    utils::LRUCache<int, int>::Config cfg;
    cfg.max_bytes = sizeof(int) * 2;
    cfg.size_fn = [](const int&) -> std::size_t { return sizeof(int); };
    utils::LRUCache<int, int> cache(cfg);

    cache.put_pinned(1, 100);
    cache.put_pinned(2, 200);

    // Adding a 3rd triggers eviction, but all entries are pinned
    // so nothing gets evicted -- cache will be over budget
    cache.put_pinned(3, 300);

    REQUIRE(cache.contains(1));
    REQUIRE(cache.contains(2));
    REQUIRE(cache.contains(3));
    REQUIRE_EQ(cache.size(), 3u);
}

TEST_CASE("LRUCache: remove non-existent key returns false") {
    utils::LRUCache<int, int> cache;
    REQUIRE(!cache.remove(999));
    REQUIRE_EQ(cache.size(), 0u);
}

TEST_CASE("LRUCache: missing_keys with all present") {
    utils::LRUCache<int, int> cache;
    cache.put(1, 10);
    cache.put(2, 20);

    std::vector<int> query = {1, 2};
    auto missing = cache.missing_keys(query.begin(), query.end());
    REQUIRE_EQ(missing.size(), 0u);
}

TEST_CASE("LRUCache: missing_keys with all absent") {
    utils::LRUCache<int, int> cache;

    std::vector<int> query = {1, 2, 3};
    auto missing = cache.missing_keys(query.begin(), query.end());
    REQUIRE_EQ(missing.size(), 3u);
}

TEST_CASE("LRUCache: missing_keys with empty range") {
    utils::LRUCache<int, int> cache;

    std::vector<int> query;
    auto missing = cache.missing_keys(query.begin(), query.end());
    REQUIRE_EQ(missing.size(), 0u);
}

TEST_CASE("LRUCache: string cache eviction with size_fn") {
    utils::LRUCache<int, std::string>::Config cfg;
    cfg.max_bytes = 30;
    cfg.evict_ratio = 15.0 / 16.0; // target = ~28 bytes
    cfg.size_fn = [](const std::string& s) -> std::size_t { return s.size(); };
    utils::LRUCache<int, std::string> cache(cfg);

    cache.put(1, "aaaaaaaaaa"); // 10 bytes
    cache.put(2, "bbbbbbbbbb"); // 10 bytes
    cache.put(3, "cccccccccc"); // 10 bytes
    // At 30 bytes == max, no eviction yet

    // Promote key 3 so it has a newer generation
    (void)cache.get(3);

    // Adding another triggers eviction -- key 1 is oldest, should be evicted first
    cache.put(4, "dddddddddd"); // 10 bytes, pushes to 40 > 30
    CHECK_GT(cache.evictions(), 0u);

    // Key 1 is oldest (never touched after initial put), should be evicted
    CHECK(!cache.contains(1));
    // Key 3 was promoted by get(), should survive
    CHECK(cache.contains(3));
}

TEST_CASE("LRUCache: string cache miss tracking") {
    utils::LRUCache<int, std::string>::Config cfg;
    cfg.max_bytes = 1024;
    cfg.size_fn = [](const std::string& s) -> std::size_t { return s.size(); };
    utils::LRUCache<int, std::string> cache(cfg);

    cache.put(1, "hello");
    auto miss = cache.get(999);
    REQUIRE(!miss.has_value());
    REQUIRE_EQ(cache.misses(), 1u);
}

TEST_CASE("LRUCache: string cache pinned eviction") {
    utils::LRUCache<int, std::string>::Config cfg;
    cfg.max_bytes = 20;
    cfg.evict_ratio = 0.5;
    cfg.size_fn = [](const std::string& s) -> std::size_t { return s.size(); };
    utils::LRUCache<int, std::string> cache(cfg);

    cache.put_pinned(1, "aaaaaaaaaa"); // 10 bytes, pinned
    cache.put(2, "bbbbbbbbbb"); // 10 bytes

    // Adding another triggers eviction -- key 2 should be evicted, key 1 pinned
    cache.put(3, "cccccccccc"); // 10 bytes
    REQUIRE(cache.contains(1)); // pinned, survives
}

TEST_CASE("LRUCache: SizedValue miss and for_each") {
    utils::LRUCache<int, SizedValue>::Config cfg;
    cfg.max_bytes = 1024;
    utils::LRUCache<int, SizedValue> cache(cfg);

    SizedValue v1;
    v1.data.resize(10, 'a');
    cache.put(1, v1);

    // Miss
    auto miss = cache.get(999);
    REQUIRE(!miss.has_value());
    REQUIRE_EQ(cache.misses(), 1u);

    // for_each
    int count = 0;
    cache.for_each([&](const int&, const SizedValue&) { ++count; });
    REQUIRE_EQ(count, 1);

    // contains
    REQUIRE(cache.contains(1));
    REQUIRE(!cache.contains(999));

    // remove
    REQUIRE(cache.remove(1));
    REQUIRE(!cache.remove(1));
    REQUIRE_EQ(cache.size(), 0u);

    // clear
    SizedValue v2;
    v2.data.resize(20, 'b');
    cache.put(2, v2);
    cache.clear();
    REQUIRE_EQ(cache.size(), 0u);
    REQUIRE_EQ(cache.byte_size(), 0u);

    // missing_keys
    cache.put(1, v1);
    std::vector<int> query = {1, 2, 3};
    auto missing = cache.missing_keys(query.begin(), query.end());
    REQUIRE_EQ(missing.size(), 2u);
}

TEST_CASE("LRUCache: int cache no size_fn eviction") {
    // Test eviction with int type and no size_fn (sizeof(V) fallback)
    utils::LRUCache<int, int>::Config cfg;
    cfg.max_bytes = sizeof(int) * 3;
    cfg.evict_ratio = 0.5;
    // No size_fn set -- uses sizeof(int) fallback
    utils::LRUCache<int, int> cache(cfg);

    cache.put(1, 10);
    cache.put(2, 20);
    cache.put(3, 30);
    // At 3 * sizeof(int) == max

    cache.put(4, 40);
    // Over budget, should evict
    CHECK_GT(cache.evictions(), 0u);
}
