#include "vc/core/render/ChunkRequestScheduler.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <set>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vc::render {

namespace {

float finiteDistance(float value) noexcept
{
    return std::isfinite(value) ? value : std::numeric_limits<float>::infinity();
}

} // namespace

struct ChunkRequestSelectionGate::Impl {
    std::mutex mutex;
};

ChunkRequestSelectionGate::ChunkRequestSelectionGate()
    : impl_(std::make_unique<Impl>())
{
}

ChunkRequestSelectionGate::~ChunkRequestSelectionGate() = default;

void ChunkRequestSelectionGate::publish(const std::function<void()>& update)
{
    if (!update)
        return;
    std::lock_guard lock(impl_->mutex);
    update();
}

struct ChunkRequestScheduler::Impl {
    struct Item {
        TaskId id = 0;
        ChunkWorkPriority priority;
        TaskGroup group = 0;
        std::uint64_t groupEpoch = 0;
        std::uint64_t sequence = 0;
        std::function<void()> task;
    };

    struct GuiLess {
        bool operator()(const std::shared_ptr<Item>& lhs,
                        const std::shared_ptr<Item>& rhs) const noexcept
        {
            if (lhs->priority.levelPriority != rhs->priority.levelPriority)
                return lhs->priority.levelPriority > rhs->priority.levelPriority;
            if (lhs->priority.activeView != rhs->priority.activeView)
                return lhs->priority.activeView > rhs->priority.activeView;
            const float ld = finiteDistance(lhs->priority.distanceSquared);
            const float rd = finiteDistance(rhs->priority.distanceSquared);
            if (ld != rd)
                return ld < rd;
            if (lhs->sequence != rhs->sequence)
                return lhs->sequence < rhs->sequence;
            return lhs->id < rhs->id;
        }
    };

    struct BackgroundLess {
        bool operator()(const std::shared_ptr<Item>& lhs,
                        const std::shared_ptr<Item>& rhs) const noexcept
        {
            if (lhs->priority.backgroundPriority != rhs->priority.backgroundPriority)
                return lhs->priority.backgroundPriority < rhs->priority.backgroundPriority;
            if (lhs->sequence != rhs->sequence)
                return lhs->sequence < rhs->sequence;
            return lhs->id < rhs->id;
        }
    };

    using GuiQueue = std::set<std::shared_ptr<Item>, GuiLess>;
    using BackgroundQueue = std::set<std::shared_ptr<Item>, BackgroundLess>;

    struct Location {
        bool interactive = false;
        GuiQueue::iterator gui;
        BackgroundQueue::iterator background;
    };

    explicit Impl(std::size_t workerCount,
                  std::size_t burst,
                  std::shared_ptr<ChunkRequestSelectionGate> gate)
        : selectionGate(std::move(gate))
        , interactiveBurst(std::max<std::size_t>(1, burst))
    {
        if (!selectionGate)
            selectionGate = std::make_shared<ChunkRequestSelectionGate>();
        workerCount = std::max<std::size_t>(1, workerCount);
        workers.reserve(workerCount);
        for (std::size_t i = 0; i < workerCount; ++i) {
            workers.emplace_back([this](std::stop_token stop) { workerLoop(stop); });
        }
    }

    ~Impl()
    {
        for (auto& worker : workers)
            worker.request_stop();
        cv.notify_all();
    }

    bool staleLocked(const Item& item) const
    {
        const auto found = minimumGroupEpoch.find(item.group);
        return item.group != 0 && found != minimumGroupEpoch.end() &&
               item.groupEpoch < found->second;
    }

    void eraseLocked(TaskId id)
    {
        const auto found = locations.find(id);
        if (found == locations.end())
            return;
        if (found->second.interactive)
            gui.erase(found->second.gui);
        else
            background.erase(found->second.background);
        locations.erase(found);
    }

    void insertLocked(const std::shared_ptr<Item>& item)
    {
        Location location;
        location.interactive = item->priority.interactive;
        if (location.interactive)
            location.gui = gui.insert(item).first;
        else
            location.background = background.insert(item).first;
        locations[item->id] = location;
    }

    std::shared_ptr<Item> popLocked()
    {
        const bool chooseGui = !gui.empty() &&
            (background.empty() || consecutiveInteractive < interactiveBurst);
        std::shared_ptr<Item> item;
        if (chooseGui) {
            auto it = gui.begin();
            item = *it;
            gui.erase(it);
            ++consecutiveInteractive;
        } else if (!background.empty()) {
            auto it = background.begin();
            item = *it;
            background.erase(it);
            consecutiveInteractive = 0;
        }
        if (item)
            locations.erase(item->id);
        return item;
    }

    void workerLoop(std::stop_token stop)
    {
        while (!stop.stop_requested()) {
            std::shared_ptr<Item> item;
            {
                std::unique_lock lock(mutex);
                cv.wait(lock, [&] {
                    return stop.stop_requested() || !gui.empty() || !background.empty();
                });
                if (stop.stop_requested() && gui.empty() && background.empty())
                    return;
            }
            {
                // Do not hold the queue mutex while waiting for publication:
                // the publisher updates queued items through reprioritize().
                std::lock_guard selectionLock(selectionGate->impl_->mutex);
                std::unique_lock lock(mutex);
                if (stop.stop_requested() && gui.empty() && background.empty())
                    return;
                item = popLocked();
                if (!item)
                    continue;
                if (staleLocked(*item)) {
                    if (gui.empty() && background.empty() &&
                        activeCount.load(std::memory_order_acquire) == 0) {
                        idleCv.notify_all();
                    }
                    continue;
                }
                activeCount.fetch_add(1, std::memory_order_acq_rel);
            }
            item->task();
            activeCount.fetch_sub(1, std::memory_order_release);
            std::lock_guard lock(mutex);
            if (gui.empty() && background.empty() &&
                activeCount.load(std::memory_order_acquire) == 0) {
                idleCv.notify_all();
            }
        }
    }

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable idleCv;
    GuiQueue gui;
    BackgroundQueue background;
    std::unordered_map<TaskId, Location> locations;
    std::unordered_map<TaskGroup, std::uint64_t> minimumGroupEpoch;
    std::shared_ptr<ChunkRequestSelectionGate> selectionGate;
    // Declared after the gate so jthread destruction joins every worker before
    // the gate they may be waiting on is released.
    std::vector<std::jthread> workers;
    std::atomic_size_t activeCount{0};
    std::uint64_t nextSequence = 0;
    std::size_t consecutiveInteractive = 0;
    const std::size_t interactiveBurst;
};

ChunkRequestScheduler::ChunkRequestScheduler(std::size_t workers,
                                             std::size_t interactiveBurst,
                                             std::shared_ptr<ChunkRequestSelectionGate> selectionGate)
    : impl_(std::make_unique<Impl>(workers, interactiveBurst,
                                  std::move(selectionGate)))
{
}

ChunkRequestScheduler::~ChunkRequestScheduler() = default;

void ChunkRequestScheduler::submit(TaskId id,
                                   ChunkWorkPriority priority,
                                   TaskGroup group,
                                   std::uint64_t groupEpoch,
                                   std::function<void()> task)
{
    if (id == 0 || !task)
        return;
    std::lock_guard lock(impl_->mutex);
    const auto minimum = impl_->minimumGroupEpoch.find(group);
    if (group != 0 && minimum != impl_->minimumGroupEpoch.end() &&
        groupEpoch < minimum->second) {
        return;
    }
    impl_->eraseLocked(id);
    auto item = std::make_shared<Impl::Item>();
    item->id = id;
    item->priority = priority;
    item->group = group;
    item->groupEpoch = groupEpoch;
    item->sequence = impl_->nextSequence++;
    item->task = std::move(task);
    impl_->insertLocked(item);
    impl_->cv.notify_one();
}

bool ChunkRequestScheduler::reprioritize(TaskId id, ChunkWorkPriority priority)
{
    std::lock_guard lock(impl_->mutex);
    const auto found = impl_->locations.find(id);
    if (found == impl_->locations.end())
        return false;
    std::shared_ptr<Impl::Item> item = found->second.interactive
        ? *found->second.gui
        : *found->second.background;
    impl_->eraseLocked(id);
    item->priority = priority;
    impl_->insertLocked(item);
    impl_->cv.notify_one();
    return true;
}

void ChunkRequestScheduler::cancelGroupBefore(TaskGroup group,
                                              std::uint64_t minimumEpoch)
{
    std::lock_guard lock(impl_->mutex);
    auto& accepted = impl_->minimumGroupEpoch[group];
    accepted = std::max(accepted, minimumEpoch);
    for (auto it = impl_->locations.begin(); it != impl_->locations.end();) {
        const auto& location = it->second;
        const auto item = location.interactive ? *location.gui : *location.background;
        if (item->group == group && item->groupEpoch < accepted) {
            const auto id = it->first;
            ++it;
            impl_->eraseLocked(id);
        } else {
            ++it;
        }
    }
    if (impl_->gui.empty() && impl_->background.empty() &&
        impl_->activeCount.load(std::memory_order_acquire) == 0) {
        impl_->idleCv.notify_all();
    }
}

std::size_t ChunkRequestScheduler::pending() const
{
    std::lock_guard lock(impl_->mutex);
    return impl_->locations.size();
}

std::size_t ChunkRequestScheduler::active() const noexcept
{
    return impl_->activeCount.load(std::memory_order_relaxed);
}

void ChunkRequestScheduler::waitIdle()
{
    std::unique_lock lock(impl_->mutex);
    impl_->idleCv.wait(lock, [&] {
        return impl_->gui.empty() && impl_->background.empty() &&
               impl_->activeCount.load(std::memory_order_acquire) == 0;
    });
}

} // namespace vc::render
