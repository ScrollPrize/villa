#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <utils/thread_pool.hpp>

#include <atomic>
#include <future>

TEST_CASE("PriorityThreadPool cancels one producer group without draining it")
{
    utils::PriorityThreadPool pool(1);
    constexpr utils::PriorityThreadPool::TaskGroup group = 42;

    std::promise<void> blockerStarted;
    std::promise<void> releaseBlocker;
    auto release = releaseBlocker.get_future().share();
    pool.submit(-100, [&] {
        blockerStarted.set_value();
        release.wait();
    });
    blockerStarted.get_future().wait();

    std::atomic<int> staleRan{0};
    for (int i = 0; i < 100; ++i) {
        pool.submit(0, group, 1, [&] {
            staleRan.fetch_add(1, std::memory_order_relaxed);
        });
    }
    CHECK(pool.pending(group) == 100);

    pool.cancel_group_before(group, 2);
    CHECK(pool.pending(group) == 0);
    CHECK(pool.pending() == 0);

    // A first-stage task racing with cancellation cannot reintroduce work from
    // the rejected epoch.
    pool.submit(0, group, 1, [&] {
        staleRan.fetch_add(1, std::memory_order_relaxed);
    });
    CHECK(pool.pending(group) == 0);

    std::atomic<int> currentRan{0};
    pool.submit(0, group, 2, [&] {
        currentRan.fetch_add(1, std::memory_order_relaxed);
    });
    CHECK(pool.pending(group) == 1);

    releaseBlocker.set_value();
    pool.wait_idle();
    CHECK(staleRan.load(std::memory_order_relaxed) == 0);
    CHECK(currentRan.load(std::memory_order_relaxed) == 1);
}
