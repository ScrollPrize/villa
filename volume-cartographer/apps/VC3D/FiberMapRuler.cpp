#include "FiberMapRuler.hpp"

#include "FiberMapRulerMath.hpp"

#include <QCoreApplication>
#include <QFontMetrics>
#include <QGraphicsView>
#include <QPainter>
#include <QPen>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <utility>

namespace
{

using namespace vc3d::fiber_map::ruler;

constexpr int kHorizontalBandPx = 22;
constexpr int kVerticalBandPx = 48;
constexpr int kMajorTickPx = 8;
constexpr int kMinorTickPx = 4;
// The band's backing: the map's surface colour at this alpha, so labels read
// over fibers when the band is clamped onto the map and all but vanish over
// the empty ground when it floats beside it.
constexpr int kBackingAlpha = 200;
// Winding labels are two or three digits: this keeps neighbours apart.
constexpr double kMinWindingLabelSpacingPx = 44.0;
// Minor (unlabelled) winding ticks disappear once windings pack tighter than
// this, or they merge into a solid bar.
constexpr double kMinWindingTickSpacingPx = 6.0;
// Distance ticks carry longer labels, and a ladder step is at least this far
// apart on screen at the tightest point of the visible range.
constexpr double kMinDistanceTickSpacingPx = 72.0;
// Cap on the ticks one paint may draw; the ladder keeps the count far below
// this, it exists so a pathological transform can never spin.
constexpr int kMaxTicksPerPaint = 2000;
constexpr double kTwoPi = 2.0 * M_PI;

QString tr(const char* text)
{
    return QCoreApplication::translate("FiberMapRuler", text);
}

// The 1-2-5 step for a ruler that labels distances, in voxels, along with the
// caption and a label formatter for that step's unit. minStepVx is the
// smallest step that keeps ticks readable at the current zoom.
struct DistanceTicks {
    double stepVx = 1.0;
    QString caption;
    std::function<QString(double)> label;
};

DistanceTicks chooseDistanceTicks(double minStepVx, const std::optional<double>& voxelSizeUm)
{
    DistanceTicks ticks;
    if (voxelSizeUm && *voxelSizeUm > 0.0) {
        const double voxelUm = *voxelSizeUm;
        const double stepUm = niceStepAtLeast(minStepVx * voxelUm);
        const LengthUnit unit = lengthUnitForStepUm(stepUm);
        ticks.stepVx = stepUm / voxelUm;
        ticks.caption = lengthUnitSuffix(unit);
        ticks.label = [voxelUm, unit](double valueVx) {
            return formatLength(valueVx * voxelUm, unit);
        };
        return ticks;
    }
    ticks.stepVx = niceStepAtLeast(minStepVx);
    ticks.caption = QStringLiteral("vx");
    ticks.label = [](double valueVx) { return formatVoxels(valueVx); };
    return ticks;
}

} // namespace

int FiberMapRuler::thicknessFor(Edge edge)
{
    return edge == Edge::Left ? kVerticalBandPx : kHorizontalBandPx;
}

FiberMapRuler::FiberMapRuler(QGraphicsView* view, Edge edge, Mode mode)
    : _view(view)
    , _edge(edge)
    , _mode(mode)
{
    _font.setPointSizeF(8.0);
}

void FiberMapRuler::setModel(FiberMapRulerModel model)
{
    _model = std::move(model);
}

void FiberMapRuler::setStyle(const FiberMapRulerStyle& style)
{
    _style = style;
}

void FiberMapRuler::setFont(const QFont& font)
{
    _font = font;
    _font.setPointSizeF(8.0);
}

QString FiberMapRuler::toolTipText() const
{
    switch (_mode) {
    case Mode::Windings:
        return tr("Winding number; the innermost anchored winding is 0.");
    case Mode::Height:
        return _model.voxelSizeUm
            ? tr("Height above the volume floor.")
            : tr("Height above the volume floor, in voxels (the package has no "
                 "voxel size).");
    case Mode::SheetDistance: {
        QString text = tr("Distance along the sheet from winding 0.");
        if (_model.hasLayout && _model.sheet.pitchVx > 0.0) {
            const QString pitch = _model.voxelSizeUm
                ? tr("%1 mm").arg(_model.sheet.pitchVx * *_model.voxelSizeUm / 1000.0, 0,
                                  'f', 3)
                : tr("%1 vx").arg(_model.sheet.pitchVx, 0, 'f', 0);
            text += QLatin1Char('\n') +
                    tr("The radius is modelled as growing linearly with the winding "
                       "(fitted pitch %1 per winding), so outer windings measure "
                       "longer than inner ones.")
                        .arg(pitch);
        } else if (_model.hasLayout) {
            text += QLatin1Char('\n') +
                    tr("Measured at the reference radius: the placed fibers span too "
                       "little winding to fit a pitch.");
        }
        if (!_model.voxelSizeUm) {
            text += QLatin1Char('\n') + tr("In voxels: the package has no voxel size.");
        }
        return text;
    }
    }
    return QString();
}

QRect FiberMapRuler::bandRect(const QRect& viewport) const
{
    if (!_view || !_model.hasLayout || viewport.isEmpty()) {
        return QRect();
    }
    const int thickness = thicknessFor(_edge);
    // The extent's four edges in viewport pixels; the band runs along the
    // extent, cut to the viewport, and rests against its edge, clamped so
    // the band never leaves the viewport.
    const int ceiling = _view->mapFromScene(QPointF(0.0, _model.extentTopSceneY)).y();
    const int floor = _view->mapFromScene(QPointF(0.0, _model.extentBottomSceneY)).y();
    const int leftEdge = _view->mapFromScene(QPointF(_model.extentLeftSceneX, 0.0)).x();
    const int rightEdge = _view->mapFromScene(QPointF(_model.extentRightSceneX, 0.0)).x();
    const int runLeft = std::max(std::min(leftEdge, rightEdge), viewport.left());
    const int runRight = std::min(std::max(leftEdge, rightEdge), viewport.right() + 1);
    const int runTop = std::max(std::min(ceiling, floor), viewport.top());
    const int runBottom = std::min(std::max(ceiling, floor), viewport.bottom() + 1);
    switch (_edge) {
    case Edge::Top: {
        if (runRight <= runLeft) {
            return QRect();
        }
        const int bottom = std::clamp(ceiling, viewport.top() + thickness, viewport.bottom() + 1);
        return QRect(runLeft, bottom - thickness, runRight - runLeft, thickness);
    }
    case Edge::Bottom: {
        if (runRight <= runLeft) {
            return QRect();
        }
        const int top = std::clamp(floor, viewport.top(), viewport.bottom() + 1 - thickness);
        return QRect(runLeft, top, runRight - runLeft, thickness);
    }
    case Edge::Left: {
        if (runBottom <= runTop) {
            return QRect();
        }
        const int right = std::clamp(leftEdge, viewport.left() + thickness, viewport.right() + 1);
        return QRect(right - thickness, runTop, thickness, runBottom - runTop);
    }
    }
    return QRect();
}

void FiberMapRuler::paint(QPainter& painter, const QRect& viewport)
{
    const QRect band = bandRect(viewport);
    if (band.isEmpty()) {
        return;
    }
    painter.save();
    painter.setRenderHint(QPainter::TextAntialiasing, true);
    painter.setRenderHint(QPainter::Antialiasing, false);
    painter.setFont(_font);

    QColor backing = _style.background;
    backing.setAlpha(kBackingAlpha);
    painter.fillRect(band, backing);

    // The edge line along the side that faces the map.
    QPen edgePen(_style.tick);
    edgePen.setWidth(1);
    painter.setPen(edgePen);
    switch (_edge) {
    case Edge::Top:
        painter.drawLine(band.left(), band.bottom(), band.right(), band.bottom());
        paintWindings(painter, band);
        break;
    case Edge::Bottom:
        painter.drawLine(band.left(), band.top(), band.right(), band.top());
        paintSheetDistance(painter, band);
        break;
    case Edge::Left:
        painter.drawLine(band.right(), band.top(), band.right(), band.bottom());
        paintHeight(painter, band);
        break;
    }
    painter.restore();
}

QRect FiberMapRuler::paintCaption(QPainter& painter, const QRect& band, const QString& caption)
{
    if (caption.isEmpty()) {
        return QRect();
    }
    const QFontMetrics metrics(_font);
    const int textWidth = metrics.horizontalAdvance(caption) + 6;
    QRect rect;
    // The caption takes the band's full height so descenders are not cut by
    // the tick zone; it sits at the far end, and labels keep clear of it.
    switch (_edge) {
    case Edge::Top:
        rect = QRect(band.right() - textWidth - 3, band.top(), textWidth, band.height() - 2);
        break;
    case Edge::Bottom:
        rect = QRect(band.right() - textWidth - 3, band.top() + 2, textWidth, band.height() - 2);
        break;
    case Edge::Left:
        rect = QRect(band.left(), band.top() + 1, band.width() - kMajorTickPx - 2,
                     metrics.height());
        break;
    }
    painter.setPen(_style.ink);
    painter.drawText(rect, Qt::AlignRight | Qt::AlignVCenter, caption);
    return rect;
}

void FiberMapRuler::paintWindings(QPainter& painter, const QRect& band)
{
    const double scale = std::abs(_view->transform().m11());
    if (!(scale > 0.0) || _model.windings.empty() || !(_model.sheet.rRefVx > 0.0)) {
        return;
    }
    const QRect caption = paintCaption(painter, band, tr("winding"));
    const double pxPerWinding = scale * kTwoPi * _model.sheet.rRefVx;
    const int labelStep = niceIntegerStepAtLeast(kMinWindingLabelSpacingPx / pxPerWinding);
    const bool minorTicks = pxPerWinding >= kMinWindingTickSpacingPx;
    const QFontMetrics metrics(_font);
    const int baseline = band.bottom();
    const int textHeight = band.height() - kMajorTickPx - 1;

    for (const vc3d::fiber_map::WindingMark& mark : _model.windings) {
        const int x = _view->mapFromScene(QPointF(mark.xVx, 0.0)).x();
        if (x < band.left() - 1 || x > band.right() + 1) {
            continue;
        }
        const bool labelled = mark.number % labelStep == 0;
        if (!labelled && !minorTicks) {
            continue;
        }
        const int tickLength = labelled ? kMajorTickPx : kMinorTickPx;
        painter.setPen(_style.tick);
        painter.drawLine(x, baseline - tickLength, x, baseline);
        if (!labelled) {
            continue;
        }
        const QString text = QString::number(mark.number);
        const int textWidth = metrics.horizontalAdvance(text) + 4;
        const QRect textRect(x - textWidth / 2, band.top(), textWidth, textHeight);
        if (caption.isValid() && textRect.intersects(caption)) {
            continue;
        }
        painter.setPen(_style.ink);
        painter.drawText(textRect, Qt::AlignHCenter | Qt::AlignVCenter, text);
    }
}

void FiberMapRuler::paintSheetDistance(QPainter& painter, const QRect& band)
{
    const double scale = std::abs(_view->transform().m11());
    const vc3d::fiber_map::SheetModel& sheet = _model.sheet;
    if (!(scale > 0.0) || !(sheet.rRefVx > 0.0) || !(sheet.radius0Vx > 0.0)) {
        return;
    }
    // The visible scene x range, cut to where the modelled radius is positive;
    // the distance function is monotonic only there.
    double sceneLeft = _view->mapToScene(QPoint(band.left(), 0)).x();
    const double sceneRight = _view->mapToScene(QPoint(band.right() + 1, 0)).x();
    if (sheet.pitchVx > 0.0) {
        const double xFloor = -(sheet.radius0Vx / sheet.pitchVx) * kTwoPi * sheet.rRefVx;
        sceneLeft = std::max(sceneLeft, xFloor);
    }
    if (!(sceneRight > sceneLeft)) {
        return;
    }
    // Ticks a fixed distance apart are closest on screen where the radius is
    // largest (the right end, the pitch being non-negative), so the step is
    // chosen against that worst case.
    const double windingRight = sceneRight / (kTwoPi * sheet.rRefVx);
    const double radiusRight = sheet.radius0Vx + sheet.pitchVx * windingRight;
    const double distanceVxPerPx = std::max(radiusRight, sheet.radius0Vx) /
                                   (sheet.rRefVx * scale);
    const DistanceTicks ticks =
        chooseDistanceTicks(kMinDistanceTickSpacingPx * distanceVxPerPx, _model.voxelSizeUm);
    if (!(ticks.stepVx > 0.0)) {
        return;
    }
    const QRect caption = paintCaption(painter, band, ticks.caption);

    const double distanceLeft = vc3d::fiber_map::sheetDistanceVx(sheet, sceneLeft);
    const double distanceRight = vc3d::fiber_map::sheetDistanceVx(sheet, sceneRight);
    const long long first = static_cast<long long>(std::ceil(distanceLeft / ticks.stepVx)) - 1;
    const long long last = static_cast<long long>(std::floor(distanceRight / ticks.stepVx)) + 1;
    if (last - first > kMaxTicksPerPaint) {
        return;
    }
    const QFontMetrics metrics(_font);
    const int textTop = band.top() + kMajorTickPx + 1;
    for (long long k = first; k <= last; ++k) {
        for (int half = 0; half < 2; ++half) {
            const double distance = (static_cast<double>(k) + 0.5 * half) * ticks.stepVx;
            const double sceneX = vc3d::fiber_map::sheetXForDistanceVx(sheet, distance);
            if (!std::isfinite(sceneX)) {
                continue;
            }
            const int x = _view->mapFromScene(QPointF(sceneX, 0.0)).x();
            if (x < band.left() - 1 || x > band.right() + 1) {
                continue;
            }
            const bool major = half == 0;
            painter.setPen(_style.tick);
            painter.drawLine(x, band.top() + 1, x,
                             band.top() + 1 + (major ? kMajorTickPx : kMinorTickPx));
            if (!major) {
                continue;
            }
            const QString text = ticks.label(distance);
            const int textWidth = metrics.horizontalAdvance(text) + 4;
            const QRect textRect(x - textWidth / 2, textTop, textWidth,
                                 band.bottom() + 1 - textTop);
            if (caption.isValid() && textRect.intersects(caption)) {
                continue;
            }
            painter.setPen(_style.ink);
            painter.drawText(textRect, Qt::AlignHCenter | Qt::AlignVCenter, text);
        }
    }
}

void FiberMapRuler::paintHeight(QPainter& painter, const QRect& band)
{
    const double scale = std::abs(_view->transform().m22());
    if (!(scale > 0.0)) {
        return;
    }
    // Scene y is -z: the top of the band is the greater height.
    const double zHigh = -_view->mapToScene(QPoint(0, band.top())).y();
    const double zLow = -_view->mapToScene(QPoint(0, band.bottom() + 1)).y();
    if (!(zHigh > zLow)) {
        return;
    }
    const DistanceTicks ticks =
        chooseDistanceTicks(kMinDistanceTickSpacingPx / scale, _model.voxelSizeUm);
    if (!(ticks.stepVx > 0.0)) {
        return;
    }
    const QRect caption = paintCaption(painter, band, ticks.caption);

    const long long first = static_cast<long long>(std::ceil(zLow / ticks.stepVx)) - 1;
    const long long last = static_cast<long long>(std::floor(zHigh / ticks.stepVx)) + 1;
    if (last - first > kMaxTicksPerPaint) {
        return;
    }
    const QFontMetrics metrics(_font);
    const int tickEnd = band.right() - 1;
    const int textRight = band.right() - kMajorTickPx - 3;
    for (long long k = first; k <= last; ++k) {
        for (int half = 0; half < 2; ++half) {
            const double z = (static_cast<double>(k) + 0.5 * half) * ticks.stepVx;
            const int y = _view->mapFromScene(QPointF(0.0, -z)).y();
            if (y < band.top() - 1 || y > band.bottom() + 1) {
                continue;
            }
            const bool major = half == 0;
            painter.setPen(_style.tick);
            painter.drawLine(tickEnd - (major ? kMajorTickPx : kMinorTickPx), y, tickEnd, y);
            if (!major) {
                continue;
            }
            const QString text = ticks.label(z);
            const int textHeight = metrics.height();
            const QRect textRect(band.left(), y - textHeight / 2, textRight - band.left(),
                                 textHeight);
            if (caption.isValid() && textRect.intersects(caption)) {
                continue;
            }
            painter.setPen(_style.ink);
            painter.drawText(textRect, Qt::AlignRight | Qt::AlignVCenter, text);
        }
    }
}
