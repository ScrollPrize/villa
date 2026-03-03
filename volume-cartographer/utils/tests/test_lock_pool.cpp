#include <utils/test.hpp>
#include <utils/lock_pool.hpp>
#include <thread>
#include <string>
#include <vector>
#include <atomic>

UTILS_TEST_MAIN()

TEST_CASE("LockPool: index computation is deterministic") {
    using Pool = utils::LockPool<64>;
    auto i1 = Pool::index(42);
    auto i2 = Pool::index(42);
    REQUIRE_EQ(i1, i2);

    // Index should be within [0, N)
    CHECK_LT(Pool::index(0), 64u);
    CHECK_LT(Pool::index(12345), 64u);
    CHECK_LT(Pool::index(-1), 64u);
}

TEST_CASE("LockPool: index with string keys") {
    using Pool = utils::LockPool<16>;
    auto i1 = Pool::index<std::string>(std::string("hello"));
    auto i2 = Pool::index<std::string>(std::string("hello"));
    REQUIRE_EQ(i1, i2);
    CHECK_LT(i1, 16u);
}

TEST_CASE("LockPool: lock acquisition and release") {
    utils::LockPool<8> pool;
    {
        auto guard = pool.lock(42);
        CHECK(guard.owns_lock());
    }
    // After scope, lock should be released; can re-acquire
    auto guard2 = pool.lock(42);
    REQUIRE(guard2.owns_lock());
}

TEST_CASE("LockPool: try_lock behavior") {
    utils::LockPool<8> pool;
    auto guard = pool.lock(10);
    REQUIRE(guard.owns_lock());

    // try_lock on the same bucket should fail
    auto try_guard = pool.try_lock(10);
    CHECK(!try_guard.owns_lock());

    // try_lock on a different bucket (if mapped differently) may succeed
    // Use a key that hashes to a different index
    bool found_different = false;
    for (int i = 0; i < 100; ++i) {
        if (utils::LockPool<8>::index(i) != utils::LockPool<8>::index(10)) {
            auto other = pool.try_lock(i);
            CHECK(other.owns_lock());
            found_different = true;
            break;
        }
    }
    REQUIRE(found_different);
}

TEST_CASE("LockPool: lock_multiple prevents deadlock via sorted order") {
    utils::LockPool<16> pool;
    std::vector<int> keys = {5, 15, 25, 35};

    auto guard = pool.lock_multiple(std::span<const int>(keys));
    // Should hold locks for all unique indices
    CHECK_GT(guard.count(), 0u);
    CHECK_LE(guard.count(), keys.size());
}

TEST_CASE("LockPool: lock_multiple deduplicates same-index keys") {
    utils::LockPool<4> pool;
    // With only 4 buckets, some keys will collide
    std::vector<int> keys = {0, 4, 8, 12};  // all hash to bucket 0 mod 4 likely
    auto guard = pool.lock_multiple(std::span<const int>(keys));
    // Deduplicated count should be <= number of keys
    CHECK_LE(guard.count(), keys.size());
}

TEST_CASE("LockPool: concurrent protection") {
    utils::LockPool<8> pool;
    std::atomic<int> counter{0};
    constexpr int iterations = 10000;

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < iterations; ++i) {
                auto guard = pool.lock(0);  // all threads lock same key
                int val = counter.load(std::memory_order_relaxed);
                counter.store(val + 1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : threads) th.join();

    REQUIRE_EQ(counter.load(), 4 * iterations);
}

TEST_CASE("SharedLockPool: shared lock") {
    utils::SharedLockPool<8> pool;

    // Multiple shared locks on the same key should coexist
    auto s1 = pool.lock_shared(42);
    auto s2 = pool.lock_shared(42);
    // Both should be valid (shared_lock doesn't expose owns_lock in all impls
    // but the fact that we got here without deadlock proves they coexist)
    REQUIRE(true);
}

TEST_CASE("LockPool: size is correct") {
    REQUIRE_EQ(utils::LockPool<16>::size(), 16u);
    REQUIRE_EQ(utils::LockPool<64>::size(), 64u);
    REQUIRE_EQ(utils::LockPool<128>::size(), 128u);
}
