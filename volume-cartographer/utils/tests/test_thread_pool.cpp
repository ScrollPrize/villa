#include <utils/test.hpp>
#include <utils/thread_pool.hpp>
#include <atomic>
#include <vector>
#include <numeric>
#include <thread>
#include <chrono>

TEST_CASE("ThreadPool basic submit and future") {
    utils::ThreadPool pool(2);
    auto fut = pool.submit([] { return 42; });
    REQUIRE_EQ(fut.get(), 42);
}

TEST_CASE("ThreadPool multiple concurrent tasks") {
    utils::ThreadPool pool(4);
    std::atomic<int> counter{0};
    constexpr int N = 100;

    std::vector<std::future<void>> futs;
    futs.reserve(N);
    for (int i = 0; i < N; ++i) {
        futs.push_back(pool.submit([&counter] { counter.fetch_add(1); }));
    }
    for (auto& f : futs) f.get();

    REQUIRE_EQ(counter.load(), N);
}

TEST_CASE("ThreadPool wait_idle") {
    utils::ThreadPool pool(2);
    std::atomic<int> counter{0};

    for (int i = 0; i < 50; ++i) {
        pool.enqueue([&counter] {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            counter.fetch_add(1);
        });
    }

    pool.wait_idle();
    REQUIRE_EQ(counter.load(), 50);
    REQUIRE_EQ(pool.pending(), std::size_t(0));
    REQUIRE_EQ(pool.active(), std::size_t(0));
}

TEST_CASE("ThreadPool worker_count") {
    utils::ThreadPool pool(3);
    REQUIRE_EQ(pool.worker_count(), std::size_t(3));
}

TEST_CASE("PriorityThreadPool ordering") {
    utils::PriorityThreadPool pool(1);
    // Use a single worker to guarantee serial execution once tasks start.
    // Submit high and low priority tasks while the pool is busy.
    std::atomic<bool> gate{false};
    std::vector<int> order;
    std::mutex mu;

    // Block the single worker.
    pool.submit(0, [&gate] {
        while (!gate.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
    });

    // Queue tasks with different priorities (lower = higher priority).
    pool.submit(10, [&] { std::lock_guard lk(mu); order.push_back(10); });
    pool.submit(1,  [&] { std::lock_guard lk(mu); order.push_back(1); });
    pool.submit(5,  [&] { std::lock_guard lk(mu); order.push_back(5); });

    // Release the gate.
    gate.store(true);
    pool.wait_idle();

    REQUIRE_GE(order.size(), std::size_t(3));
    // Priority 1 should come before 5, and 5 before 10.
    CHECK_EQ(order[0], 1);
    CHECK_EQ(order[1], 5);
    CHECK_EQ(order[2], 10);
}

TEST_CASE("PriorityThreadPool epoch-based staleness") {
    utils::PriorityThreadPool pool(1);
    std::atomic<bool> gate{false};
    std::atomic<int> executed{0};

    // Block the worker.
    pool.submit(0, [&gate] {
        while (!gate.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
    });

    // Submit tasks at epoch 1.
    pool.submit(0, std::uint64_t(1), [&executed] { executed.fetch_add(1); });
    pool.submit(0, std::uint64_t(1), [&executed] { executed.fetch_add(1); });

    // Advance the pool epoch to 2 so the above become stale.
    pool.set_epoch(2);
    REQUIRE_EQ(pool.epoch(), std::uint64_t(2));

    gate.store(true);
    pool.wait_idle();

    // Stale tasks should have been discarded.
    REQUIRE_EQ(executed.load(), 0);
}

TEST_CASE("PriorityThreadPool cancel_pending") {
    utils::PriorityThreadPool pool(1);
    std::atomic<bool> gate{false};

    pool.submit(0, [&gate] {
        while (!gate.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
    });

    for (int i = 0; i < 20; ++i)
        pool.submit(0, [] {});

    REQUIRE_GT(pool.pending(), std::size_t(0));
    pool.cancel_pending();
    REQUIRE_EQ(pool.pending(), std::size_t(0));

    gate.store(true);
    pool.wait_idle();
}

TEST_CASE("parallel_for") {
    utils::ThreadPool pool(4);
    std::vector<std::atomic<int>> results(100);
    for (auto& a : results) a.store(0);

    utils::parallel_for(pool, std::size_t(0), std::size_t(100),
        [&results](std::size_t i) {
            results[i].store(static_cast<int>(i * i));
        }, 10);

    for (std::size_t i = 0; i < 100; ++i)
        REQUIRE_EQ(results[i].load(), static_cast<int>(i * i));
}

TEST_CASE("parallel_for_each") {
    utils::ThreadPool pool(4);
    std::vector<int> data(50);
    std::iota(data.begin(), data.end(), 0);

    std::atomic<int> sum{0};
    utils::parallel_for_each(pool, data,
        [&sum](int v) { sum.fetch_add(v); }, 10);

    REQUIRE_EQ(sum.load(), 50 * 49 / 2);
}

TEST_CASE("parallel_reduce") {
    utils::ThreadPool pool(4);
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 1);

    int result = utils::parallel_reduce(pool, data, 0,
        [](int a, int b) { return a + b; },
        [](int x) { return x; },
        10);

    REQUIRE_EQ(result, 100 * 101 / 2);
}

TEST_CASE("ThreadPool submit with arguments") {
    utils::ThreadPool pool(2);
    auto fut = pool.submit([](int a, int b) { return a + b; }, 17, 25);
    REQUIRE_EQ(fut.get(), 42);
}

TEST_CASE("ThreadPool default worker count") {
    utils::ThreadPool pool; // 0 = auto
    REQUIRE_GE(pool.worker_count(), std::size_t(1));
}

TEST_CASE("ThreadPool exception propagation") {
    utils::ThreadPool pool(2);
    auto fut = pool.submit([]() -> int { throw std::runtime_error("test error"); });
    bool caught = false;
    try {
        fut.get();
    } catch (const std::runtime_error& e) {
        caught = true;
        REQUIRE_EQ(std::string(e.what()), std::string("test error"));
    }
    REQUIRE(caught);
}

TEST_CASE("ThreadPool submit returns different types") {
    utils::ThreadPool pool(2);

    auto f_string = pool.submit([] { return std::string("hello"); });
    REQUIRE_EQ(f_string.get(), std::string("hello"));

    auto f_double = pool.submit([] { return 3.14; });
    REQUIRE_NEAR(f_double.get(), 3.14, 1e-12);

    auto f_void = pool.submit([] { /* no return */ });
    f_void.get(); // should not throw
}

TEST_CASE("ThreadPool active count during work") {
    utils::ThreadPool pool(2);
    std::atomic<bool> gate{false};
    std::atomic<int> started{0};

    // Block both workers
    for (int i = 0; i < 2; ++i) {
        pool.enqueue([&gate, &started] {
            started.fetch_add(1);
            while (!gate.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
        });
    }

    // Wait until both workers are active
    while (started.load() < 2)
        std::this_thread::sleep_for(std::chrono::microseconds(50));

    REQUIRE_EQ(pool.active(), std::size_t(2));

    gate.store(true);
    pool.wait_idle();
    REQUIRE_EQ(pool.active(), std::size_t(0));
}

TEST_CASE("parallel_for empty range") {
    utils::ThreadPool pool(2);
    int counter = 0;
    utils::parallel_for(pool, std::size_t(5), std::size_t(5),
        [&counter](std::size_t) { counter++; });
    REQUIRE_EQ(counter, 0);

    // begin > end
    utils::parallel_for(pool, std::size_t(10), std::size_t(5),
        [&counter](std::size_t) { counter++; });
    REQUIRE_EQ(counter, 0);
}

TEST_CASE("parallel_for_each empty range") {
    utils::ThreadPool pool(2);
    std::vector<int> data;
    std::atomic<int> sum{0};
    utils::parallel_for_each(pool, data,
        [&sum](int v) { sum.fetch_add(v); });
    REQUIRE_EQ(sum.load(), 0);
}

TEST_CASE("parallel_reduce empty range") {
    utils::ThreadPool pool(2);
    std::vector<int> data;
    int result = utils::parallel_reduce(pool, data, 42,
        [](int a, int b) { return a + b; },
        [](int x) { return x; });
    REQUIRE_EQ(result, 42); // should return init
}

TEST_CASE("PriorityThreadPool worker_count") {
    utils::PriorityThreadPool pool(5);
    REQUIRE_EQ(pool.worker_count(), std::size_t(5));
}

TEST_CASE("PriorityThreadPool default worker count") {
    utils::PriorityThreadPool pool;
    REQUIRE_GE(pool.worker_count(), std::size_t(1));
}

TEST_CASE("PriorityThreadPool FIFO within same priority") {
    utils::PriorityThreadPool pool(1);
    std::atomic<bool> gate{false};
    std::vector<int> order;
    std::mutex mu;

    // Block the single worker.
    pool.submit(0, [&gate] {
        while (!gate.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
    });

    // All same priority -- should be FIFO
    for (int i = 0; i < 5; ++i) {
        pool.submit(0, [&, i] { std::lock_guard lk(mu); order.push_back(i); });
    }

    gate.store(true);
    pool.wait_idle();

    REQUIRE_EQ(order.size(), std::size_t(5));
    for (int i = 0; i < 5; ++i) {
        CHECK_EQ(order[i], i);
    }
}

TEST_CASE("PriorityThreadPool epoch staleness with priority reordering") {
    utils::PriorityThreadPool pool(1);
    std::atomic<bool> gate{false};
    std::vector<int> order;
    std::mutex mu;

    // Block the single worker.
    pool.submit(0, [&gate] {
        while (!gate.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
    });

    // Submit tasks at epoch 1 with various priorities.
    pool.submit(10, std::uint64_t(1), [&] { std::lock_guard lk(mu); order.push_back(10); });
    pool.submit(1,  std::uint64_t(1), [&] { std::lock_guard lk(mu); order.push_back(1); });

    // Submit a task with no epoch (always valid).
    pool.submit(5, [&] { std::lock_guard lk(mu); order.push_back(5); });

    // Advance epoch to 2 -- epoch-1 tasks become stale.
    pool.set_epoch(2);

    // Submit a task at current epoch 2.
    pool.submit(3, std::uint64_t(2), [&] { std::lock_guard lk(mu); order.push_back(3); });

    gate.store(true);
    pool.wait_idle();

    // Epoch-1 tasks (priority 10 and 1) should be discarded.
    // Remaining: priority 3 (epoch 2) and priority 5 (always valid).
    // Priority ordering: 3 before 5.
    REQUIRE_EQ(order.size(), std::size_t(2));
    CHECK_EQ(order[0], 3);
    CHECK_EQ(order[1], 5);
}

TEST_CASE("PriorityThreadPool cancel_pending with no pending tasks") {
    utils::PriorityThreadPool pool(2);
    pool.wait_idle();

    // cancel_pending on an empty queue should be safe.
    pool.cancel_pending();
    REQUIRE_EQ(pool.pending(), std::size_t(0));
}

TEST_CASE("PriorityThreadPool cancel_pending does not affect in-flight") {
    utils::PriorityThreadPool pool(1);
    std::atomic<bool> gate{false};
    std::atomic<int> in_flight_done{0};
    std::atomic<int> queued_done{0};

    // Submit a task that blocks until gate is opened.
    pool.submit(0, [&gate, &in_flight_done] {
        while (!gate.load()) std::this_thread::sleep_for(std::chrono::microseconds(100));
        in_flight_done.fetch_add(1);
    });
    // Give the worker time to pick up the blocking task.
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // Submit more tasks while first is in-flight.
    for (int i = 0; i < 10; ++i) {
        pool.submit(0, [&queued_done] { queued_done.fetch_add(1); });
    }

    // Cancel pending -- should drop queued tasks but not the in-flight one.
    pool.cancel_pending();

    gate.store(true);
    pool.wait_idle();

    // The in-flight task must have completed.
    REQUIRE_EQ(in_flight_done.load(), 1);
    // Most or all queued tasks should have been cancelled.
    CHECK_LT(queued_done.load(), 10);
}

TEST_CASE("PriorityThreadPool epoch always-valid tasks never discarded") {
    utils::PriorityThreadPool pool(1);
    std::atomic<bool> gate{false};
    std::atomic<int> executed{0};

    pool.submit(0, [&gate] {
        while (!gate.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
    });

    // Submit without epoch (always valid).
    pool.submit(0, [&executed] { executed.fetch_add(1); });
    pool.submit(0, [&executed] { executed.fetch_add(1); });

    // Even after advancing epoch, always-valid tasks should execute.
    pool.set_epoch(100);

    gate.store(true);
    pool.wait_idle();

    REQUIRE_EQ(executed.load(), 2);
}

TEST_CASE("PriorityThreadPool destructor drains or discards tasks") {
    std::atomic<int> counter{0};
    {
        utils::PriorityThreadPool pool(2);
        for (int i = 0; i < 20; ++i) {
            pool.submit(0, [&counter] { counter.fetch_add(1); });
        }
        // Pool destructor runs here -- should not hang.
    }
    // Some tasks may or may not have executed; the key is no deadlock.
    CHECK_GE(counter.load(), 0);
}

TEST_CASE("ThreadPool destructor with pending tasks") {
    std::atomic<int> counter{0};
    {
        utils::ThreadPool pool(2);
        for (int i = 0; i < 50; ++i) {
            pool.enqueue([&counter] { counter.fetch_add(1); });
        }
        // Destructor should complete without hanging.
    }
    CHECK_GE(counter.load(), 0);
}

TEST_CASE("ThreadPool enqueue fire-and-forget") {
    utils::ThreadPool pool(2);
    std::atomic<int> counter{0};

    for (int i = 0; i < 100; ++i) {
        pool.enqueue([&counter] { counter.fetch_add(1); });
    }
    pool.wait_idle();

    REQUIRE_EQ(counter.load(), 100);
}

TEST_CASE("PriorityThreadPool mixed epoch and no-epoch ordering") {
    utils::PriorityThreadPool pool(1);
    std::atomic<bool> gate{false};
    std::vector<int> order;
    std::mutex mu;

    pool.submit(0, [&gate] {
        while (!gate.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
    });

    // No-epoch tasks at various priorities.
    pool.submit(20, [&] { std::lock_guard lk(mu); order.push_back(20); });
    pool.submit(2,  [&] { std::lock_guard lk(mu); order.push_back(2); });

    // Epoch tasks at current epoch (0).
    pool.submit(15, std::uint64_t(0), [&] { std::lock_guard lk(mu); order.push_back(15); });
    pool.submit(1,  std::uint64_t(0), [&] { std::lock_guard lk(mu); order.push_back(1); });

    // Advance epoch to 1 -- epoch-0 tasks become stale.
    pool.set_epoch(1);

    gate.store(true);
    pool.wait_idle();

    // Only no-epoch tasks should execute, in priority order.
    REQUIRE_EQ(order.size(), std::size_t(2));
    CHECK_EQ(order[0], 2);
    CHECK_EQ(order[1], 20);
}

TEST_CASE("parallel_for default chunk_size") {
    utils::ThreadPool pool(4);
    std::atomic<int> sum{0};

    // Use default chunk_size (0 = auto).
    utils::parallel_for(pool, std::size_t(0), std::size_t(100),
        [&sum](std::size_t i) {
            sum.fetch_add(static_cast<int>(i));
        });

    REQUIRE_EQ(sum.load(), 4950); // sum of 0..99
}

TEST_CASE("parallel_reduce with transform") {
    utils::ThreadPool pool(4);
    std::vector<int> data = {1, 2, 3, 4, 5};

    // Square each element then sum.
    int result = utils::parallel_reduce(pool, data, 0,
        [](int a, int b) { return a + b; },
        [](int x) { return x * x; },
        1);

    REQUIRE_EQ(result, 1 + 4 + 9 + 16 + 25);
}

TEST_CASE("parallel_for_each default chunk_size") {
    utils::ThreadPool pool(4);
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 0);

    std::atomic<int> sum{0};
    // Use default chunk_size (0 = auto)
    utils::parallel_for_each(pool, data,
        [&sum](int v) { sum.fetch_add(v); });

    REQUIRE_EQ(sum.load(), 100 * 99 / 2);
}

TEST_CASE("parallel_reduce default chunk_size") {
    utils::ThreadPool pool(4);
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 1);

    // Use default chunk_size (0 = auto)
    int result = utils::parallel_reduce(pool, data, 0,
        [](int a, int b) { return a + b; },
        [](int x) { return x; });

    REQUIRE_EQ(result, 100 * 101 / 2);
}

TEST_CASE("parallel_for_each with single element") {
    utils::ThreadPool pool(2);
    std::vector<int> data = {42};

    std::atomic<int> sum{0};
    utils::parallel_for_each(pool, data,
        [&sum](int v) { sum.fetch_add(v); }, 1);

    REQUIRE_EQ(sum.load(), 42);
}

TEST_CASE("parallel_reduce with single element") {
    utils::ThreadPool pool(2);
    std::vector<int> data = {7};

    int result = utils::parallel_reduce(pool, data, 0,
        [](int a, int b) { return a + b; },
        [](int x) { return x * 10; },
        1);

    REQUIRE_EQ(result, 70);
}

TEST_CASE("parallel_reduce max operation") {
    utils::ThreadPool pool(4);
    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};

    int result = utils::parallel_reduce(pool, data, std::numeric_limits<int>::min(),
        [](int a, int b) { return std::max(a, b); },
        [](int x) { return x; },
        3);

    REQUIRE_EQ(result, 9);
}

TEST_CASE("parallel_for with chunk_size 1") {
    utils::ThreadPool pool(4);
    std::atomic<int> sum{0};

    utils::parallel_for(pool, std::size_t(0), std::size_t(10),
        [&sum](std::size_t i) {
            sum.fetch_add(static_cast<int>(i));
        }, 1);

    REQUIRE_EQ(sum.load(), 45); // sum of 0..9
}

TEST_CASE("parallel_for_each with string range") {
    utils::ThreadPool pool(2);
    std::vector<std::string> data = {"hello", " ", "world"};

    std::mutex mu;
    std::size_t total_len = 0;
    utils::parallel_for_each(pool, data,
        [&](const std::string& s) {
            std::lock_guard lk(mu);
            total_len += s.size();
        }, 1);

    REQUIRE_EQ(total_len, 11u); // 5 + 1 + 5
}

TEST_CASE("default_chunk_size returns at least 1") {
    // When count is small relative to workers
    auto cs = utils::detail::default_chunk_size(1, 100);
    REQUIRE_GE(cs, 1u);

    // When workers is 0 (edge case protection)
    cs = utils::detail::default_chunk_size(100, 0);
    REQUIRE_GE(cs, 1u);
}

UTILS_TEST_MAIN()
