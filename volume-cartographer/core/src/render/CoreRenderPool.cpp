#include "vc/core/render/CoreRenderPool.hpp"

#include "vc/core/render/TileRenderer.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"

#include <utils/thread_pool.hpp>

namespace vc::render {

CoreRenderPool::CoreRenderPool(int numThreads)
    : pool_(std::make_unique<utils::PriorityThreadPool>(numThreads))
{
}

CoreRenderPool::~CoreRenderPool() noexcept
{
    clearAll();
}

void CoreRenderPool::submit(const TileRenderParams& params,
                            const std::shared_ptr<Surface>& surface,
                            const std::shared_ptr<Volume>& volume,
                            int controllerId)
{
    int priority = params.submitPriority;
    pool_->submit(priority,
        [this, params, surface, volume, controllerId]() {
            TileRenderResult result = TileRenderer::renderTile(params, surface, volume.get());
            result.controllerId = controllerId;
            pushResult(std::move(result));
        });
}

std::vector<TileRenderResult> CoreRenderPool::takeResults(int controllerId)
{
    std::vector<TileRenderResult> all;
    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        std::swap(all, completedResults_);
    }

    std::vector<TileRenderResult> mine;
    std::vector<TileRenderResult> others;
    for (auto& item : all) {
        if (item.controllerId == controllerId)
            mine.push_back(std::move(item));
        else
            others.push_back(std::move(item));
    }

    if (!others.empty()) {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        completedResults_.insert(completedResults_.end(),
            std::make_move_iterator(others.begin()),
            std::make_move_iterator(others.end()));
    }

    return mine;
}

void CoreRenderPool::clearQueue()
{
    pool_->cancel_pending();
}

void CoreRenderPool::clearAll()
{
    pool_->cancel_pending();
    std::lock_guard<std::mutex> lock(resultsMutex_);
    completedResults_.clear();
}

bool CoreRenderPool::busy() const noexcept
{
    return pool_->pending() > 0 || pool_->active() > 0;
}

void CoreRenderPool::setReadyCallback(std::function<void()> cb)
{
    readyCallback_ = std::move(cb);
}

void CoreRenderPool::pushResult(TileRenderResult result)
{
    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        completedResults_.push_back(std::move(result));
    }
    if (readyCallback_)
        readyCallback_();
}

} // namespace vc::render
