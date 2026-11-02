// Repeated construction and destruction of utils::ThreadPool and
// utils::PriorityThreadPool must terminate (#1686).
//
// The destructors used to call request_stop() and notify_all() without holding
// the pool mutex. A worker that had just evaluated its wait predicate (no stop,
// empty queue) but not yet parked in cv_.wait() then missed the notification,
// and the jthread join in the destructor never returned. The window is widest
// when a pool is destroyed right after construction, before its workers reach
// their first wait, so most rounds here do exactly that. On Windows (MSYS2
// UCRT64) the unfixed pool hung within 32 rounds in every run; the fixed one
// completed 225 of 225 runs. On Linux the unfixed pool hung in about 1 of 150
// runs of the plane-sampler test that owns a pool.
//
// Rounds can be raised for a soak run: VC_THREAD_POOL_LIFECYCLE_ROUNDS=4096.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "utils/thread_pool.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <thread>

namespace {

std::size_t rounds()
{
    if (const char* env = std::getenv("VC_THREAD_POOL_LIFECYCLE_ROUNDS")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0)
            return static_cast<std::size_t>(v);
    }
    return 256;
}

std::size_t workersFor(std::size_t round)
{
    // 1, 2, 4, 8 workers in turn, so that both the single-worker case and
    // several workers racing the destructor are covered.
    return std::size_t{1} << (round % 4);
}

} // namespace

TEST_CASE("ThreadPool destroyed right after construction terminates")
{
    const std::size_t n = rounds();
    for (std::size_t round = 0; round < n; ++round) {
        utils::ThreadPool pool(workersFor(round));
        CHECK(pool.worker_count() == workersFor(round));
    }
}

TEST_CASE("ThreadPool destroyed after a batch terminates and ran every task")
{
    const std::size_t n = rounds();
    for (std::size_t round = 0; round < n; ++round) {
        std::atomic<std::size_t> ran{0};
        {
            utils::ThreadPool pool(workersFor(round));
            pool.run_indexed_batch(17, [&](std::size_t) {
                ran.fetch_add(1, std::memory_order_relaxed);
            });
            pool.enqueue([&] { ran.fetch_add(1, std::memory_order_relaxed); });
            pool.wait_idle();
        }
        CHECK(ran.load() == 18);
    }
}

TEST_CASE("ThreadPool destroyed with work still queued terminates")
{
    // Workers are parked on the condition variable or busy; the destructor
    // must wake every one of them. Tasks left in the queue are not run.
    const std::size_t n = rounds() / 4 + 1;
    for (std::size_t round = 0; round < n; ++round) {
        std::atomic<std::size_t> ran{0};
        {
            utils::ThreadPool pool(workersFor(round));
            for (int i = 0; i < 64; ++i)
                pool.enqueue([&] {
                    ran.fetch_add(1, std::memory_order_relaxed);
                    std::this_thread::yield();
                });
        }
        CHECK(ran.load() <= 64);
    }
}

TEST_CASE("PriorityThreadPool destroyed right after construction terminates")
{
    const std::size_t n = rounds();
    for (std::size_t round = 0; round < n; ++round) {
        utils::PriorityThreadPool pool(workersFor(round));
        CHECK(pool.worker_count() == workersFor(round));
    }
}

TEST_CASE("PriorityThreadPool destroyed after work terminates and ran every live task")
{
    const std::size_t n = rounds();
    for (std::size_t round = 0; round < n; ++round) {
        std::atomic<std::size_t> ran{0};
        {
            utils::PriorityThreadPool pool(workersFor(round));
            for (int i = 0; i < 9; ++i)
                pool.submit(i % 3, [&] { ran.fetch_add(1, std::memory_order_relaxed); });
            pool.wait_idle();
        }
        CHECK(ran.load() == 9);
    }
}
