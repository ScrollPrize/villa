#pragma once

#include <QObject>

#include <memory>
#include <vector>

#include "TileRenderer.hpp"
#include "vc/core/render/CoreRenderPool.hpp"

class Surface;
class Volume;

// Qt wrapper around CoreRenderPool.
class RenderPool : public QObject
{
    Q_OBJECT

public:
    explicit RenderPool(int numThreads = 2, QObject* parent = nullptr);
    ~RenderPool() override;

    void submit(const TileRenderParams& params,
                const std::shared_ptr<Surface>& surface,
                const std::shared_ptr<Volume>& volume,
                int controllerId);

    std::vector<QtTileRenderResult> takeResults(int controllerId);

    void cancelAll();

    vc::render::CoreRenderPool& corePool() { return *corePool_; }
    const vc::render::CoreRenderPool& corePool() const { return *corePool_; }

signals:
    void tileReady();

private:
    std::unique_ptr<vc::render::CoreRenderPool> corePool_;
};
