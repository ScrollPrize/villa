#pragma once

#include <QPixmap>

#include "vc/core/render/TileRenderer.hpp"

// Backward compatibility: expose vc::TileRenderer at global scope.
using TileRenderer = vc::TileRenderer;

// Qt-layer extension: adds QPixmap to TileRenderResult for display.
// RenderPool converts raw pixels -> QPixmap on worker threads;
// TileRenderController applies the pixmap to the scene.
struct QtTileRenderResult : TileRenderResult {
    QPixmap pixmap;
};
