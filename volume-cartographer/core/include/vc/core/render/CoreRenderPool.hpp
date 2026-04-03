#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "vc/core/render/TileTypes.hpp"

namespace utils { class PriorityThreadPool; }
class Surface;
class Volume;

namespace vc::render {

// Background tile rendering pool.
// Submit tiles, take results. That's it.
class CoreRenderPool final {
public:
    explicit CoreRenderPool(int numThreads = 2);
    ~CoreRenderPool() noexcept;

    void submit(const TileRenderParams& params,
                const std::shared_ptr<Surface>& surface,
                const std::shared_ptr<Volume>& volume,
                int controllerId);

    // Take all completed results for controllerId.
    std::vector<TileRenderResult> takeResults(int controllerId);

    // Drop queued tasks. In-flight tasks finish normally.
    void clearQueue();

    // Drop queued tasks + completed results.
    void clearAll();

    // True if pool has queued or in-flight tasks.
    [[nodiscard]] bool busy() const noexcept;

    void setReadyCallback(std::function<void()> cb);

private:
    void pushResult(TileRenderResult result);

    std::unique_ptr<utils::PriorityThreadPool> pool_;
    std::mutex resultsMutex_;
    std::vector<TileRenderResult> completedResults_;
    std::function<void()> readyCallback_;
};

} // namespace vc::render
