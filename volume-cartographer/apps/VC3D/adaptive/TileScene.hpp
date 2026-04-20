#pragma once

// Stub header: TileScene no longer exists in the adaptive viewer.
// This satisfies compilation of code that references TileScene methods
// but guards with null checks (e.g. SegmentationOverlayController).
// At runtime, CAdaptiveVolumeViewer::tileScene() returns nullptr,
// so none of these methods are ever called.

#include <QPointF>
#include <opencv2/core.hpp>

class TileScene {
public:
    QPointF surfaceToScene(float, float) const { return {}; }
    cv::Vec2f sceneToSurface(const QPointF&) const { return {}; }
};
