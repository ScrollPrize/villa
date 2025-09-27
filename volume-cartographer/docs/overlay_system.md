# VC3D Overlay System: Usage Guide

The VC3D overlay framework renders auxiliary graphics (points, paths, text, arrows, etc.) on top of a `CVolumeViewer` without directly manipulating `QGraphicsItem`s. Overlay controllers inherit from `ViewerOverlayControllerBase`, compute primitives, and hand them back to the base which manages lifetimes and viewer attachment.

**Key takeaways up front**

- Work in **volume coordinates**. Use `filterPoints()` or `volumeToScene()` to project onto whatever viewer surface is active (segmentation quad, seg xz/yz slice, etc.).
- To pick a viewer, either branch on `viewer->surfName()` inside a controller or loop across viewers via `ViewerManager::forEachViewer` when using the inline helpers.
- For one-off or in-line overlays, use  `ViewerOverlayControllerBase::applyPrimitives(viewer, key, primitives)` , which lets lets you slap a primitive on any viewer by name without writing a full controller.
  - key is the overlay group identifier (a std::string) you pass to applyPrimitives. It tells the viewer which logical overlay slot to populate (so later calls with the same key replace existing primitives).


## Core Concepts

### ViewerOverlayControllerBase

Each controller derives from `ViewerOverlayControllerBase` and implements:

```cpp
class MyOverlayController : public ViewerOverlayControllerBase {
    void collectPrimitives(CVolumeViewer* viewer, OverlayBuilder& builder) override;
};
```

Key responsibilities handled by the base:

- Attaching/detaching to viewers.
- Reacting to `viewer->overlaysUpdated()`.
- Translating primitives to `QGraphicsItem`s and managing lifetimes.
- Utility methods (`volumeToScene`, `filterPoints`, `visibleSceneRect`, etc.).

### OverlayBuilder Primitives

Controllers add any combination of primitives when collecting:

- `addPoint(position, radius, style)`
- `addCircle(center, radius, filled, style)`
- `addLineStrip(points, closed, style)`
- `addRect(rect, filled, style)`
- `addText(position, text, font, style)`
- `addPath(pathPrimitive)` for complex paths (like seeding widget or drawing widget).
- `addArrow(start, end, headLength, headWidth, style)`

`OverlayStyle` supports pen/brush colors, width, dash pattern, cap/join, and z-order.

### Filtering Helpers

`filterPoints()` clips volume-space points against plane/quad surfaces, visibility rectangles, and predicates:

```cpp
PointFilterOptions opts;
opts.clipToSurface = true;
opts.planeDistanceTolerance = 4.0f;
opts.requireSceneVisibility = true;
opts.volumePredicate = [](const cv::Vec3f& wp, size_t idx) { /* custom logic */ };
opts.scenePredicate  = [](const QPointF& sp, size_t idx) { return sp.y() > 0; };

auto filtered = filterPoints(viewer, rawPoints, opts);
for (size_t i = 0; i < filtered.volumePoints.size(); ++i) {
    const auto& wp = filtered.volumePoints[i];
    const auto& sp = filtered.scenePoints[i];
    builder.addPoint(sp, 4.0, style);
}
```

### `applyPrimitives`

For one-off overlays (e.g., inline tooling), call `ViewerOverlayControllerBase::applyPrimitives(viewer, key, primitives)` to render primitives without a dedicated controller.

```cpp
std::vector<ViewerOverlayControllerBase::OverlayPrimitive> prims;
ViewerOverlayControllerBase::OverlayStyle style;
style.penColor = Qt::yellow;
style.brushColor = QColor(255, 255, 0, 128);
style.penWidth = 2.0;
style.z = 50.0;
prims.emplace_back(ViewerOverlayControllerBase::CirclePrimitive{scenePt, 5.0, true, style});
ViewerOverlayControllerBase::applyPrimitives(viewer, "inline_marker", std::move(prims));
```

This replaces older manual `QGraphicsItem` creation.

## Stock Controllers

### PointsOverlayController

- Renders VCCollection points with highlight/selection styling and optional winding labels.
- Uses `filterPoints` to fade points based on plane/quad distance and current viewport.
- Relies on `CVolumeViewer` accessors (`highlightedPointId`, `selectedPointId`, etc.).

### PathsOverlayController

- Receives `PathPrimitive`s from drawing and seeding widgets.
- Supports line-strip and point render modes plus eraser styling (dashed red path).

### BBoxOverlayController

- Renders persistent selection rectangles and active drag preview.
- Viewer exposes `selections()` (stored surf-space rect + color) and `activeBBoxSceneRect()`.

### VectorOverlayController

- Handles direction hints/step markers using arrow and circle primitives.
- Controllers can call `addProvider` to register additional vector producers (e.g., surface normals, rays) without modifying existing logic.
- Example provider registration:

```cpp
auto vectorOverlay = std::make_unique<VectorOverlayController>(surfaces, this);
vectorOverlay->addProvider([](CVolumeViewer* viewer, ViewerOverlayControllerBase::OverlayBuilder& builder) {
    // compute normals, arrows, etc.
});
```

## Inline Usage Examples

### Highlight a Single Scene Point

```cpp
ViewerOverlayControllerBase::OverlayStyle style;
style.penColor = Qt::cyan;
style.brushColor = QColor(0, 255, 255, 120);
style.penWidth = 1.5;
style.z = 90.0;

ViewerOverlayControllerBase::applyPrimitives(viewer, "debug_marker", {
    ViewerOverlayControllerBase::CirclePrimitive{scenePt, 6.0, true, style}
});
```

### Show a Volume-Space Arrow

```cpp
cv::Vec3f start = ...;
cv::Vec3f end = ...;
QPointF startScene = viewer->volumePointToScene(start);
QPointF endScene = viewer->volumePointToScene(end);
ViewerOverlayControllerBase::OverlayStyle style;
style.penColor = Qt::magenta;
style.penWidth = 2.0;
style.z = 95.0;
ViewerOverlayControllerBase::applyPrimitives(viewer, "vector_preview", {
    ViewerOverlayControllerBase::ArrowPrimitive{startScene, endScene, 12.0, 7.0, style}
});
```

### Quick Text Annotation

```cpp
ViewerOverlayControllerBase::OverlayStyle style;
style.penColor = Qt::white;
style.z = 96.0;
QFont font;
font.setPointSize(10);
ViewerOverlayControllerBase::applyPrimitives(viewer, "note", {
    ViewerOverlayControllerBase::TextPrimitive{scenePos, QStringLiteral("Peak"), font, style}
});
```

## Integrating a New Controller

1. Derive from `ViewerOverlayControllerBase` and implement `collectPrimitives`.
2. Register with `ViewerManager` (e.g., via `CWindow` constructor).
3. Expose necessary state via `CVolumeViewer` accessors (avoid direct scene manipulation).
4. Trigger refresh with `viewer->overlaysUpdated()` when underlying state changes.

Example skeleton:

```cpp
class NormalOverlayController : public ViewerOverlayControllerBase {
public:
    explicit NormalOverlayController(QObject* parent = nullptr)
        : ViewerOverlayControllerBase("normals", parent) {}

protected:
    void collectPrimitives(CVolumeViewer* viewer, OverlayBuilder& builder) override {
        auto opts = PointFilterOptions{};
        opts.clipToSurface = true;
        opts.planeDistanceTolerance = 5.0f;
        auto points = ...; // compute volume-space positions
        auto filtered = filterPoints(viewer, points, opts);
        for (size_t i = 0; i < filtered.volumePoints.size(); ++i) {
            const auto& wp = filtered.volumePoints[i];
            cv::Vec3f normal = ...;
            QPointF base = filtered.scenePoints[i];
            QPointF tip = base + QPointF(normal[0], normal[1]) * 40.0;
            OverlayStyle style;
            style.penColor = Qt::yellow;
            style.penWidth = 1.5;
            style.z = 70.0;
            builder.addArrow(base, tip, 8.0, 4.0, style);
        }
    }
};
```

## Viewer State Hooks

`CVolumeViewer` exposes overlay-relevant state:

- `highlightedPointId()`, `selectedPointId()`, `selectedCollectionId()`
- `isShowDirectionHints()`, `activeBBoxSceneRect()`
- `selections()` returns stored bbox rects with colors
- `onPathsChanged(...)` stores path primitives and emits `overlaysUpdated()`

Emit `overlaysUpdated()` whenever the state driving a controller changes.

## Inline Debugging

Use `applyPrimitives` for quick diagnostics in tooling or tests without formal controllers. Remember to reuse overlay keys (`debug_*`) to avoid leaking previous primitives.

## Summary

- Controllers focus solely on computing primitives; the base handles rendering and viewer lifecycle.
- Shared helpers (`filterPoints`, `applyPrimitives`) reduce duplication.
- Existing controllers (points, paths, bbox, vector) provide ready-to-use patterns for new overlays.
