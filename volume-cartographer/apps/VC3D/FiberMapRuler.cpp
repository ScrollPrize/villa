#include "FiberMapRuler.hpp"

#include "FiberMapRulerMath.hpp"

#include <QFontMetrics>
#include <QGraphicsView>
#include <QPaintEvent>
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

FiberMapRuler::FiberMapRuler(QGraphicsView* view, Edge edge, Mode mode, QWidget* parent)
    : QWidget(parent)
    , _view(view)
    , _edge(edge)
    , _mode(mode)
{
    // The rulers repaint from the view's transform; they never take input.
    setAttribute(Qt::WA_TransparentForMouseEvents, true);
    setFocusPolicy(Qt::NoFocus);
    QFont small = font();
    small.setPointSizeF(8.0);
    setFont(small);
    _style.background = palette().color(QPalette::Window);
    _style.ink = palette().color(QPalette::WindowText);
    _style.tick = palette().color(QPalette::Mid);
    updateToolTip();
}

void FiberMapRuler::setModel(FiberMapRulerModel model)
{
    _model = std::move(model);
    updateToolTip();
    update();
}

void FiberMapRuler::setRulerStyle(const FiberMapRulerStyle& style)
{
    _style = style;
    update();
}

void FiberMapRuler::updateToolTip()
{
    switch (_mode) {
    case Mode::Windings:
        setToolTip(tr("Winding number; the innermost anchored winding is 0."));
        break;
    case Mode::Height:
        setToolTip(_model.voxelSizeUm
                       ? tr("Height above the volume floor.")
                       : tr("Height above the volume floor, in voxels (the package "
                            "has no voxel size)."));
        break;
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
            text += QLatin1Char('\n') +
                    tr("In voxels: the package has no voxel size.");
        }
        setToolTip(text);
        break;
    }
    }
}

void FiberMapRuler::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.fillRect(event->rect(), _style.background);
    painter.setRenderHint(QPainter::TextAntialiasing, true);
    painter.setFont(font());

    // The edge line along the side that faces the viewport.
    QPen edgePen(_style.tick);
    edgePen.setWidth(1);
    painter.setPen(edgePen);
    switch (_edge) {
    case Edge::Top:
        painter.drawLine(0, height() - 1, width(), height() - 1);
        break;
    case Edge::Bottom:
        painter.drawLine(0, 0, width(), 0);
        break;
    case Edge::Left:
        painter.drawLine(width() - 1, 0, width() - 1, height());
        break;
    }

    if (!_view || !_model.hasLayout) {
        return;
    }
    switch (_mode) {
    case Mode::Windings:
        paintWindings(painter);
        break;
    case Mode::SheetDistance:
        paintSheetDistance(painter);
        break;
    case Mode::Height:
        paintHeight(painter);
        break;
    }
}

QRect FiberMapRuler::paintCaption(QPainter& painter, const QString& caption)
{
    if (caption.isEmpty()) {
        return QRect();
    }
    const QFontMetrics metrics(font());
    const int textWidth = metrics.horizontalAdvance(caption) + 6;
    QRect rect;
    // The caption takes the band's full height so descenders are not cut by
    // the tick zone; it sits at the far end, and labels keep clear of it.
    switch (_edge) {
    case Edge::Top:
        rect = QRect(width() - textWidth - 4, 0, textWidth, height() - 2);
        break;
    case Edge::Bottom:
        rect = QRect(width() - textWidth - 4, 2, textWidth, height() - 2);
        break;
    case Edge::Left:
        rect = QRect(0, 1, width() - kMajorTickPx - 2, metrics.height());
        break;
    }
    painter.setPen(_style.ink);
    painter.drawText(rect, Qt::AlignRight | Qt::AlignVCenter, caption);
    return rect;
}

void FiberMapRuler::paintWindings(QPainter& painter)
{
    const double scale = std::abs(_view->transform().m11());
    if (!(scale > 0.0) || _model.windings.empty() || !(_model.sheet.rRefVx > 0.0)) {
        return;
    }
    const QRect caption = paintCaption(painter, tr("winding"));
    const double pxPerWinding = scale * kTwoPi * _model.sheet.rRefVx;
    const int labelStep = niceIntegerStepAtLeast(kMinWindingLabelSpacingPx / pxPerWinding);
    const bool minorTicks = pxPerWinding >= kMinWindingTickSpacingPx;
    const QFontMetrics metrics(font());
    const int textBottom = height() - kMajorTickPx - 1;

    for (const vc3d::fiber_map::WindingMark& mark : _model.windings) {
        const int x = _view->mapFromScene(QPointF(mark.xVx, 0.0)).x();
        if (x < -1 || x > width() + 1) {
            continue;
        }
        const bool labelled = mark.number % labelStep == 0;
        if (!labelled && !minorTicks) {
            continue;
        }
        const int tickLength = labelled ? kMajorTickPx : kMinorTickPx;
        painter.setPen(_style.tick);
        painter.drawLine(x, height() - 1 - tickLength, x, height() - 1);
        if (!labelled) {
            continue;
        }
        const QString text = QString::number(mark.number);
        const int textWidth = metrics.horizontalAdvance(text) + 4;
        const QRect textRect(x - textWidth / 2, 0, textWidth, textBottom);
        if (caption.isValid() && textRect.intersects(caption)) {
            continue;
        }
        painter.setPen(_style.ink);
        painter.drawText(textRect, Qt::AlignHCenter | Qt::AlignVCenter, text);
    }
}

void FiberMapRuler::paintSheetDistance(QPainter& painter)
{
    const double scale = std::abs(_view->transform().m11());
    const vc3d::fiber_map::SheetModel& sheet = _model.sheet;
    if (!(scale > 0.0) || !(sheet.rRefVx > 0.0) || !(sheet.radius0Vx > 0.0)) {
        return;
    }
    // The visible scene x range, cut to where the modelled radius is positive;
    // the distance function is monotonic only there.
    double sceneLeft = _view->mapToScene(QPoint(0, 0)).x();
    const double sceneRight = _view->mapToScene(QPoint(width(), 0)).x();
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
    const QRect caption = paintCaption(painter, ticks.caption);

    const double distanceLeft = vc3d::fiber_map::sheetDistanceVx(sheet, sceneLeft);
    const double distanceRight = vc3d::fiber_map::sheetDistanceVx(sheet, sceneRight);
    const long long first = static_cast<long long>(std::ceil(distanceLeft / ticks.stepVx)) - 1;
    const long long last = static_cast<long long>(std::floor(distanceRight / ticks.stepVx)) + 1;
    if (last - first > kMaxTicksPerPaint) {
        return;
    }
    const QFontMetrics metrics(font());
    const int textTop = kMajorTickPx + 1;
    for (long long k = first; k <= last; ++k) {
        for (int half = 0; half < 2; ++half) {
            const double distance = (static_cast<double>(k) + 0.5 * half) * ticks.stepVx;
            const double sceneX = vc3d::fiber_map::sheetXForDistanceVx(sheet, distance);
            if (!std::isfinite(sceneX)) {
                continue;
            }
            const int x = _view->mapFromScene(QPointF(sceneX, 0.0)).x();
            if (x < -1 || x > width() + 1) {
                continue;
            }
            const bool major = half == 0;
            painter.setPen(_style.tick);
            painter.drawLine(x, 1, x, 1 + (major ? kMajorTickPx : kMinorTickPx));
            if (!major) {
                continue;
            }
            const QString text = ticks.label(distance);
            const int textWidth = metrics.horizontalAdvance(text) + 4;
            const QRect textRect(x - textWidth / 2, textTop, textWidth, height() - textTop);
            if (caption.isValid() && textRect.intersects(caption)) {
                continue;
            }
            painter.setPen(_style.ink);
            painter.drawText(textRect, Qt::AlignHCenter | Qt::AlignVCenter, text);
        }
    }
}

void FiberMapRuler::paintHeight(QPainter& painter)
{
    const double scale = std::abs(_view->transform().m22());
    if (!(scale > 0.0)) {
        return;
    }
    // Scene y is -z: the top of the band is the greater height.
    const double zHigh = -_view->mapToScene(QPoint(0, 0)).y();
    const double zLow = -_view->mapToScene(QPoint(0, height())).y();
    if (!(zHigh > zLow)) {
        return;
    }
    const DistanceTicks ticks =
        chooseDistanceTicks(kMinDistanceTickSpacingPx / scale, _model.voxelSizeUm);
    if (!(ticks.stepVx > 0.0)) {
        return;
    }
    const QRect caption = paintCaption(painter, ticks.caption);

    const long long first = static_cast<long long>(std::ceil(zLow / ticks.stepVx)) - 1;
    const long long last = static_cast<long long>(std::floor(zHigh / ticks.stepVx)) + 1;
    if (last - first > kMaxTicksPerPaint) {
        return;
    }
    const QFontMetrics metrics(font());
    const int textRight = width() - kMajorTickPx - 3;
    for (long long k = first; k <= last; ++k) {
        for (int half = 0; half < 2; ++half) {
            const double z = (static_cast<double>(k) + 0.5 * half) * ticks.stepVx;
            const int y = _view->mapFromScene(QPointF(0.0, -z)).y();
            if (y < -1 || y > height() + 1) {
                continue;
            }
            const bool major = half == 0;
            painter.setPen(_style.tick);
            painter.drawLine(width() - 2 - (major ? kMajorTickPx : kMinorTickPx), y,
                             width() - 2, y);
            if (!major) {
                continue;
            }
            const QString text = ticks.label(z);
            const int textHeight = metrics.height();
            const QRect textRect(0, y - textHeight / 2, textRight, textHeight);
            if (caption.isValid() && textRect.intersects(caption)) {
                continue;
            }
            painter.setPen(_style.ink);
            painter.drawText(textRect, Qt::AlignRight | Qt::AlignVCenter, text);
        }
    }
}
