#include "LineModelWarmupQueue.hpp"

#include <atomic>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using vc3d::line_annotation::LineModelWarmupQueue;

void require(bool value)
{
    if (!value)
        throw std::runtime_error("warmup lifecycle assertion failed");
}

int main()
{
    // Replacements before dispatch leave only the newest geometry job.
    {
        LineModelWarmupQueue queue;
        std::vector<int> observed;
        require(queue.replace([&](const auto&) { observed.push_back(1); }));
        require(!queue.replace([&](const auto&) { observed.push_back(2); }));
        queue.run();
        require(observed == std::vector<int>{2});
        require(queue.replace([&](const auto&) { observed.push_back(3); }));
        queue.cancel();
        queue.run();
        require(observed == std::vector<int>{2});
    }

    // An active metadata read may finish; edits cancel its future submissions
    // and coalesce to one replacement without creating another active worker.
    {
        auto queue = std::make_shared<LineModelWarmupQueue>();
        std::promise<LineModelWarmupQueue::Cancel> started;
        std::promise<void> release;
        auto releaseFuture = release.get_future();
        std::vector<int> observed;
        require(queue->replace([&](const auto& cancel) {
            started.set_value(cancel);
            releaseFuture.wait();
            if (!cancel->load())
                observed.push_back(1);
        }));
        std::thread worker([queue] { queue->run(); });
        auto activeCancel = started.get_future().get();
        require(!queue->replace([&](const auto&) { observed.push_back(2); }));
        require(activeCancel->load());
        require(!queue->replace([&](const auto&) { observed.push_back(3); }));
        release.set_value();
        worker.join();
        require(observed == std::vector<int>{3});
    }

    // Closing never waits for the active resource read. A worker-owned
    // snapshot survives its GUI owner and is released once that read returns.
    {
        auto queue = std::make_shared<LineModelWarmupQueue>();
        auto resource = std::make_shared<int>(42);
        std::weak_ptr<int> resourceLifetime = resource;
        std::promise<LineModelWarmupQueue::Cancel> started;
        std::promise<void> release;
        auto releaseFuture = release.get_future();
        std::atomic<bool> pendingRan{false};
        require(queue->replace([&, resource](const auto& cancel) {
            started.set_value(cancel);
            releaseFuture.wait();
            require(*resource == 42);
        }));
        resource.reset();
        std::thread worker([queue] { queue->run(); });
        auto activeCancel = started.get_future().get();
        require(!queue->replace([&](const auto&) { pendingRan.store(true); }));
        queue->cancel();
        queue.reset();
        require(activeCancel->load());
        require(!resourceLifetime.expired());
        release.set_value();
        worker.join();
        require(resourceLifetime.expired());
        require(!pendingRan.load());
    }

    // Failed speculative work still releases the worker and runs its latest
    // replacement; failures never become terminal state in the GUI lifecycle.
    {
        LineModelWarmupQueue queue;
        bool recovered = false;
        require(queue.replace([&](const auto&) {
            require(!queue.replace([&](const auto&) { recovered = true; }));
            throw std::runtime_error("metadata unavailable");
        }));
        queue.run();
        require(recovered);
    }
}
