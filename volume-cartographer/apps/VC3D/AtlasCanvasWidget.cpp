#include "AtlasCanvasWidget.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <QGraphicsEllipseItem>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QResizeEvent>

#include <algorithm>
#include <cmath>

AtlasCanvasWidget::AtlasCanvasWidget(QWidget* parent)
    : QGraphicsView(parent),
      _scene(new QGraphicsScene(this))
{
    setObjectName(QStringLiteral("atlasCanvas"));
    setScene(_scene);
    setRenderHint(QPainter::Antialiasing, true);
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
}

void AtlasCanvasWidget::setAtlas(const std::filesystem::path& atlasDir,
                                 const vc::atlas::Atlas& atlas)
{
    _atlasDir = atlasDir;
    _atlas = atlas;
    _hasAtlas = true;
    rebuildScene();
}

void AtlasCanvasWidget::clearAtlas()
{
    _hasAtlas = false;
    _atlasDir.clear();
    _atlas = {};
    _scene->clear();
}

void AtlasCanvasWidget::resizeEvent(QResizeEvent* event)
{
    QGraphicsView::resizeEvent(event);
    if (_scene && !_scene->items().empty()) {
        fitInView(_scene->itemsBoundingRect().adjusted(-10, -10, 10, 10), Qt::KeepAspectRatio);
    }
}

void AtlasCanvasWidget::rebuildScene()
{
    _scene->clear();
    if (!_hasAtlas) {
        return;
    }

    const std::filesystem::path basePath = _atlasDir / _atlas.metadata.baseMeshPath;
    QuadSurface base(basePath);
    const auto* points = base.rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }

    const int rows = points->rows;
    const int cols = points->cols;
    const int seedWinding = cols > 0
        ? static_cast<int>(std::floor(_atlas.metadata.seedAtlasU / cols))
        : 0;
    const int stride = std::max(1, std::max(rows, cols) / 180);

    QPen gridPen(QColor(120, 130, 140, 90));
    gridPen.setWidthF(0.0);

    auto valid = [points](int row, int col) {
        const cv::Vec3f p = (*points)(row, col);
        return p[0] != -1.0f &&
               std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
    };

    for (int winding = seedWinding - 1; winding <= seedWinding + 1; ++winding) {
        const double offset = static_cast<double>(winding * cols);
        for (int row = 0; row < rows; row += stride) {
            QPainterPath path;
            bool open = false;
            for (int col = 0; col < cols; col += stride) {
                if (!valid(row, col)) {
                    open = false;
                    continue;
                }
                const QPointF p(offset + col, row);
                if (!open) {
                    path.moveTo(p);
                    open = true;
                } else {
                    path.lineTo(p);
                }
            }
            _scene->addPath(path, gridPen);
        }
        for (int col = 0; col < cols; col += stride) {
            QPainterPath path;
            bool open = false;
            for (int row = 0; row < rows; row += stride) {
                if (!valid(row, col)) {
                    open = false;
                    continue;
                }
                const QPointF p(offset + col, row);
                if (!open) {
                    path.moveTo(p);
                    open = true;
                } else {
                    path.lineTo(p);
                }
            }
            _scene->addPath(path, gridPen);
        }
    }

    QPen fiberPen(QColor(220, 60, 50));
    fiberPen.setWidthF(std::max(1.0, static_cast<double>(stride) * 0.35));
    QPen controlPen(QColor(30, 110, 210));
    controlPen.setWidthF(0.0);
    QBrush controlBrush(QColor(30, 110, 210));

    for (const auto& fiber : _atlas.fibers) {
        QPainterPath fiberPath;
        bool open = false;
        for (const auto& anchor : fiber.lineAnchors) {
            const QPointF p(anchor.atlasU, anchor.atlasV);
            if (!open) {
                fiberPath.moveTo(p);
                open = true;
            } else {
                fiberPath.lineTo(p);
            }
        }
        _scene->addPath(fiberPath, fiberPen);

        const double radius = std::max(2.0, static_cast<double>(stride));
        for (const auto& anchor : fiber.controlAnchors) {
            _scene->addEllipse(anchor.atlasU - radius,
                               anchor.atlasV - radius,
                               radius * 2.0,
                               radius * 2.0,
                               controlPen,
                               controlBrush);
        }
    }

    _scene->setSceneRect(_scene->itemsBoundingRect().adjusted(-10, -10, 10, 10));
    fitInView(_scene->sceneRect(), Qt::KeepAspectRatio);
}
