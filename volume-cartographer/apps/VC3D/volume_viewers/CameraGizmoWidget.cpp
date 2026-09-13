#include "CameraGizmoWidget.hpp"

#include <QEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>

#include <algorithm>
#include <cmath>

namespace
{
constexpr int kPaneSize = 92;
constexpr int kPaneGap = 8;
constexpr int kMargin = 12;
constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
} // namespace

CameraGizmoWidget::CameraGizmoWidget(QWidget* parent)
    : QWidget(parent)
{
    setFixedSize(kPaneSize * 3 + kPaneGap * 2, kPaneSize);
    setCursor(Qt::CrossCursor);
    setToolTip(tr("Volumetric camera. Left dial: azimuth — spins the patch in "
                  "the view plane. Middle gauge: tilt — tips the camera from "
                  "straight down (vertical needle) toward flat; on screen the "
                  "tilt is always toward the top edge. Right gauge: "
                  "perspective strength.\nDouble-click a pane to reset it.\n"
                  "The slab rotates rigidly in the flattened (slab) space, so "
                  "the render follows the page as it bends."));
    if (parent) {
        parent->installEventFilter(this);
        repositionInParent();
    }
}

void CameraGizmoWidget::setCamera(float azimuthDeg, float tiltDeg, float perspective)
{
    _azimuthDeg = azimuthDeg;
    _tiltDeg = std::clamp(tiltDeg, 0.0f, kMaxTiltDeg);
    _perspective = std::clamp(perspective, 0.0f, 1.0f);
    update();
}

QPointF CameraGizmoWidget::azimuthCenter() const
{
    return QPointF(kPaneSize * 0.5, height() * 0.5);
}

QPointF CameraGizmoWidget::elevationCenter() const
{
    // Needle pivot: bottom-center of the middle pane.
    return QPointF(kPaneSize + kPaneGap + kPaneSize * 0.5, height() - 8.0);
}

QRectF CameraGizmoWidget::perspectiveTrackRect() const
{
    // Bounding box of the vertical trapezoid track centered in the right
    // pane; 0 at the (narrow) bottom, 1 at the (wide) top.
    const double paneLeft = (kPaneSize + kPaneGap) * 2.0;
    const double trackWidth = 36.0;
    return QRectF(paneLeft + kPaneSize * 0.5 - trackWidth * 0.5, 8.0,
                  trackWidth, height() - 16.0);
}

double CameraGizmoWidget::dialRadius() const
{
    return kPaneSize * 0.5 - 5.0;
}

CameraGizmoWidget::Pane CameraGizmoWidget::paneAt(const QPointF& pos) const
{
    if (pos.x() < kPaneSize + kPaneGap * 0.5)
        return Pane::Azimuth;
    if (pos.x() < (kPaneSize + kPaneGap) * 2.0 - kPaneGap * 0.5)
        return Pane::Elevation;
    return Pane::Perspective;
}

void CameraGizmoWidget::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    const QColor accent(0, 220, 255);
    const QColor faint(200, 200, 200, 70);

    // ---- Azimuth dial (left) ----
    {
        const QPointF c = azimuthCenter();
        const double r = dialRadius();

        p.setPen(Qt::NoPen);
        p.setBrush(QColor(20, 20, 20, 160));
        p.drawEllipse(c, r, r);

        QPen rim(QColor(accent.red(), accent.green(), accent.blue(), 200), 1.4);
        rim.setCosmetic(true);
        p.setPen(rim);
        p.setBrush(Qt::NoBrush);
        p.drawEllipse(c, r, r);

        // Cardinal ticks (+U right, +V down, matching slab coords).
        p.setPen(QPen(faint, 1.0));
        for (int i = 0; i < 4; ++i) {
            const double a = i * 90.0 * kDegToRad;
            const QPointF dir(std::cos(a), std::sin(a));
            p.drawLine(c + dir * (r * 0.82), c + dir * r);
        }

        // Needle + rim dot at the current azimuth.
        const double a = double(_azimuthDeg) * kDegToRad;
        const QPointF dir(std::cos(a), std::sin(a));
        QPen needle(accent, 1.6);
        needle.setCosmetic(true);
        p.setPen(needle);
        p.drawLine(c, c + dir * (r * 0.86));
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 100));
        p.drawEllipse(c + dir * (r * 0.86), 5.0, 5.0);
        p.setBrush(accent);
        p.drawEllipse(c + dir * (r * 0.86), 2.5, 2.5);
        p.drawEllipse(c, 2.0, 2.0);
    }

    // ---- Elevation gauge (middle) ----
    {
        const QPointF pivot = elevationCenter();
        const double r = kPaneSize - 18.0;

        p.setPen(Qt::NoPen);
        p.setBrush(QColor(20, 20, 20, 160));
        QPainterPath bg;
        bg.moveTo(pivot);
        // True-angle sector: vertical (0 deg tilt) to 45 deg.
        bg.arcTo(QRectF(pivot.x() - r, pivot.y() - r, r * 2.0, r * 2.0), 45.0, 45.0);
        bg.closeSubpath();
        p.drawPath(bg);

        // Gauge arc; the needle angle equals the actual tilt angle.
        QPen rim(QColor(0, 220, 255, 200), 1.4);
        rim.setCosmetic(true);
        p.setPen(rim);
        p.setBrush(Qt::NoBrush);
        p.drawArc(QRectF(pivot.x() - r, pivot.y() - r, r * 2.0, r * 2.0),
                  45 * 16, 45 * 16);

        // Reference ticks at 0, 22.5 and 45 degrees of tilt.
        p.setPen(QPen(faint, 1.0));
        for (const double tickTilt : {0.0, 22.5, 45.0}) {
            const double a = tickTilt * kDegToRad;
            const QPointF dir(std::sin(a), -std::cos(a));
            p.drawLine(pivot + dir * (r * 0.86), pivot + dir * r);
        }

        // Needle at the current tilt (vertical = straight down the normal).
        const double a = double(_tiltDeg) * kDegToRad;
        const QPointF dir(std::sin(a), -std::cos(a));
        QPen needle(accent, 1.6);
        needle.setCosmetic(true);
        p.setPen(needle);
        p.drawLine(pivot, pivot + dir * (r * 0.9));
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 100));
        p.drawEllipse(pivot + dir * (r * 0.9), 5.0, 5.0);
        p.setBrush(accent);
        p.drawEllipse(pivot + dir * (r * 0.9), 2.5, 2.5);
        p.drawEllipse(pivot, 2.0, 2.0);
    }

    // ---- Perspective gauge (right) ----
    {
        const QRectF track = perspectiveTrackRect();

        // Trapezoid: narrow at the bottom (0), spreading toward the top (1) —
        // a diverging perspective frustum.
        const double bottomHalf = 4.0;
        const double topHalf = track.width() * 0.5;
        const double cx = track.center().x();
        auto halfWidthAt = [&](double y) {
            const double t = (track.bottom() - y) / track.height();
            return bottomHalf + (topHalf - bottomHalf) * t;
        };
        auto trapezoid = [&](double topY) {
            QPainterPath path;
            path.moveTo(cx - bottomHalf, track.bottom());
            path.lineTo(cx - halfWidthAt(topY), topY);
            path.lineTo(cx + halfWidthAt(topY), topY);
            path.lineTo(cx + bottomHalf, track.bottom());
            path.closeSubpath();
            return path;
        };

        p.setPen(Qt::NoPen);
        p.setBrush(QColor(20, 20, 20, 160));
        p.drawPath(trapezoid(track.top()));

        // Fill from the bottom, proportional to the perspective strength.
        if (_perspective > 0.0f) {
            p.setBrush(QColor(accent.red(), accent.green(),
                              accent.blue(), 170));
            p.drawPath(trapezoid(track.bottom() - track.height() * double(_perspective)));
        }

        QPen rim(QColor(accent.red(), accent.green(),
                        accent.blue(), 200), 1.4);
        rim.setCosmetic(true);
        p.setPen(rim);
        p.setBrush(Qt::NoBrush);
        p.drawPath(trapezoid(track.top()));

        // Reference ticks at 0, 0.5 and 1.
        p.setPen(QPen(faint, 1.0));
        for (const double t : {0.0, 0.5, 1.0}) {
            const double y = track.bottom() - track.height() * t;
            p.drawLine(QPointF(track.right() + 3.0, y),
                       QPointF(track.right() + 9.0, y));
        }

        // Handle dot at the current value.
        const double y = track.bottom() - track.height() * double(_perspective);
        const QPointF handle(track.center().x(), y);
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(accent.red(), accent.green(),
                          accent.blue(), 100));
        p.drawEllipse(handle, 5.0, 5.0);
        p.setBrush(accent);
        p.drawEllipse(handle, 2.5, 2.5);
    }
}

void CameraGizmoWidget::updateFromDrag(const QPointF& pos)
{
    if (_dragPane == Pane::Azimuth) {
        const QPointF c = azimuthCenter();
        const double dx = pos.x() - c.x();
        const double dy = pos.y() - c.y();
        if (std::hypot(dx, dy) < 3.0)
            return;  // too close to the center to define a direction
        _azimuthDeg = float(std::atan2(dy, dx) / kDegToRad);
    } else if (_dragPane == Pane::Elevation) {
        const QPointF pivot = elevationCenter();
        const double dx = pos.x() - pivot.x();
        const double dy = pivot.y() - pos.y();  // up is positive
        const double angle = std::atan2(std::max(dx, 0.0), std::max(dy, 0.0));
        _tiltDeg = std::clamp(float(angle / kDegToRad), 0.0f, kMaxTiltDeg);
    } else if (_dragPane == Pane::Perspective) {
        const QRectF track = perspectiveTrackRect();
        const double t = (track.bottom() - pos.y()) / track.height();
        _perspective = std::clamp(float(t), 0.0f, 1.0f);
    } else {
        return;
    }
    update();
    emit cameraChanged(_azimuthDeg, _tiltDeg, _perspective);
}

void CameraGizmoWidget::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        _dragPane = paneAt(event->position());
        updateFromDrag(event->position());
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

void CameraGizmoWidget::mouseMoveEvent(QMouseEvent* event)
{
    if ((event->buttons() & Qt::LeftButton) && _dragPane != Pane::None) {
        updateFromDrag(event->position());
        event->accept();
        return;
    }
    QWidget::mouseMoveEvent(event);
}

void CameraGizmoWidget::mouseDoubleClickEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        switch (paneAt(event->position())) {
            case Pane::Azimuth:
                _azimuthDeg = 0.0f;
                break;
            case Pane::Elevation:
                _tiltDeg = 0.0f;
                break;
            default:
                _perspective = 0.0f;
                break;
        }
        update();
        emit cameraChanged(_azimuthDeg, _tiltDeg, _perspective);
        event->accept();
        return;
    }
    QWidget::mouseDoubleClickEvent(event);
}

bool CameraGizmoWidget::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == parent() &&
        (event->type() == QEvent::Resize || event->type() == QEvent::Show)) {
        repositionInParent();
    }
    return QWidget::eventFilter(watched, event);
}

void CameraGizmoWidget::repositionInParent()
{
    if (auto* p = parentWidget()) {
        move((p->width() - width()) / 2, p->height() - height() - kMargin);
    }
}
