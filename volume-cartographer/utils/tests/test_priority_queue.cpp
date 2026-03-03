#include <utils/test.hpp>
#include <utils/priority_queue.hpp>
#include <thread>
#include <vector>
#include <algorithm>

UTILS_TEST_MAIN()

TEST_CASE("PriorityQueue: submit and pop ordering (min-queue)") {
    utils::DeduplicatingPriorityQueue<int> pq;
    pq.submit(30);
    pq.submit(10);
    pq.submit(20);

    // Default is std::greater<int>, yielding a min-queue
    auto v1 = pq.try_pop();
    auto v2 = pq.try_pop();
    auto v3 = pq.try_pop();
    REQUIRE(v1.has_value());
    REQUIRE(v2.has_value());
    REQUIRE(v3.has_value());
    REQUIRE_EQ(v1.value(), 10);
    REQUIRE_EQ(v2.value(), 20);
    REQUIRE_EQ(v3.value(), 30);

    // Complete them to clear in-flight
    pq.complete(10);
    pq.complete(20);
    pq.complete(30);
}

TEST_CASE("PriorityQueue: deduplication rejects same item") {
    utils::DeduplicatingPriorityQueue<int> pq;
    REQUIRE(pq.submit(1));
    REQUIRE(!pq.submit(1));  // already queued

    auto v = pq.try_pop();
    REQUIRE(v.has_value());

    // Now it's in-flight, still rejected
    REQUIRE(!pq.submit(1));

    pq.complete(1);
    // After completion, can submit again
    REQUIRE(pq.submit(1));
    pq.complete(pq.try_pop().value());
}

TEST_CASE("PriorityQueue: in-flight tracking") {
    utils::DeduplicatingPriorityQueue<int> pq;
    pq.submit(5);
    REQUIRE(pq.is_queued(5));
    REQUIRE(!pq.is_in_flight(5));

    auto v = pq.try_pop();
    REQUIRE(v.has_value());
    REQUIRE(!pq.is_queued(5));
    REQUIRE(pq.is_in_flight(5));
    REQUIRE(pq.is_known(5));

    pq.complete(5);
    REQUIRE(!pq.is_in_flight(5));
    REQUIRE(!pq.is_known(5));
}

TEST_CASE("PriorityQueue: batch submit") {
    utils::DeduplicatingPriorityQueue<int> pq;
    std::vector<int> items = {1, 2, 3, 2, 1};  // duplicates within batch
    auto added = pq.submit_batch(items.begin(), items.end());
    REQUIRE_EQ(added, 3u);
    REQUIRE_EQ(pq.queued_count(), 3u);

    // Adding a second batch with overlap
    std::vector<int> more = {3, 4, 5};
    auto added2 = pq.submit_batch(more.begin(), more.end());
    REQUIRE_EQ(added2, 2u);  // only 4 and 5 are new
}

TEST_CASE("PriorityQueue: cancel_pending") {
    utils::DeduplicatingPriorityQueue<int> pq;
    pq.submit(1);
    pq.submit(2);
    pq.submit(3);

    // Pop one to make it in-flight
    auto v = pq.try_pop();
    REQUIRE(v.has_value());

    pq.cancel_pending();
    REQUIRE_EQ(pq.queued_count(), 0u);
    // In-flight item is unaffected
    REQUIRE_EQ(pq.in_flight_count(), 1u);

    // Nothing left to pop
    auto empty = pq.try_pop();
    REQUIRE(!empty.has_value());

    pq.complete(v.value());
}

TEST_CASE("PriorityQueue: empty queue try_pop returns nullopt") {
    utils::DeduplicatingPriorityQueue<int> pq;
    auto v = pq.try_pop();
    REQUIRE(!v.has_value());
}

TEST_CASE("PriorityQueue: pop with timeout") {
    utils::DeduplicatingPriorityQueue<int> pq;

    // Timeout on empty queue
    auto v = pq.pop_for(std::chrono::milliseconds(10));
    REQUIRE(!v.has_value());

    // Submit then pop_for should succeed
    pq.submit(42);
    auto v2 = pq.pop_for(std::chrono::milliseconds(100));
    REQUIRE(v2.has_value());
    REQUIRE_EQ(v2.value(), 42);
    pq.complete(42);
}

TEST_CASE("PriorityQueue: shutdown unblocks waiters") {
    utils::DeduplicatingPriorityQueue<int> pq;
    bool threw = false;

    std::thread waiter([&] {
        try {
            (void)pq.pop();  // blocks
        } catch (const std::runtime_error&) {
            threw = true;
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pq.shutdown();
    waiter.join();

    REQUIRE(threw);
}

TEST_CASE("PriorityQueue: submit after shutdown returns false") {
    utils::DeduplicatingPriorityQueue<int> pq;
    pq.shutdown();
    REQUIRE(!pq.submit(1));

    std::vector<int> items = {1, 2, 3};
    REQUIRE_EQ(pq.submit_batch(items.begin(), items.end()), 0u);
}

TEST_CASE("PriorityQueue: membership queries") {
    utils::DeduplicatingPriorityQueue<int> pq;
    pq.submit(10);
    REQUIRE(pq.is_queued(10));
    REQUIRE(!pq.is_queued(20));
    REQUIRE(pq.is_known(10));
    REQUIRE(!pq.empty());
}
