#include "FiberCaseQueue.hpp"
#include "FiberTrialPruning.hpp"
#include <cassert>
#include <set>

int main() {
    std::atomic<size_t> entered{0}, completed{0};
    std::atomic<bool> ready{false};
    std::mutex mutex;
    std::set<std::thread::id> ids;
    FiberCaseQueue queue(4, 600, [&] {
        const auto owner = std::this_thread::get_id();
        { std::lock_guard lock(mutex); ids.insert(owner); }
        const size_t value = entered.fetch_add(1);
        volatile size_t preserved = value;
        FiberCaseQueue::await([&] { return ready.load(); });
        assert(owner == std::this_thread::get_id());
        assert(preserved == value);
        ++completed;
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (entered < 32 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    assert(entered == 32); // Bounded suspended contexts; later jobs remain lightweight.
    ready = true;
    FiberCaseQueue::notify();
    queue.join();
    assert(completed == 600 && ids.size() <= 4);
    const auto start = std::chrono::steady_clock::now();
    std::atomic<size_t> jobs{0};
    FiberCaseQueue large(8, 301134, [&] { ++jobs; });
    large.join();
    assert(jobs == 301134);
    std::printf("bounded queue 301134 jobs: %.3fs\n",
        std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    FiberTrialPruning pruning;
    pruning.record("a:forward", 300, true);
    assert(!pruning.skip("a:forward", 500));
    pruning.record("a:forward", 200, true);
    assert(pruning.skip("a:forward", 400));
    assert(!pruning.skip("a:forward", 300));
    assert(!pruning.skip("a:forward", 100));
    assert(!pruning.skip("a:reverse", 400));
    pruning.record("b:forward", 100, false);
    pruning.record("b:forward", 200, true);
    assert(!pruning.skip("b:forward", 300));
    bool threw = false;
    try {
        FiberCaseQueue failing(1, 1, [] { throw std::runtime_error("test"); });
        failing.join();
    } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    std::atomic<size_t> waiting{0};
    threw = false;
    try {
        FiberCaseQueue failing(2, 20, [&] {
            const auto id = waiting.fetch_add(1);
            if (id == 4) throw std::runtime_error("sibling failure");
            FiberCaseQueue::await([] { return false; });
        });
        failing.join();
    } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
}
