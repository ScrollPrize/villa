#pragma once
// vc4d::Viewer — volume slice rendering widget using only libvc.
// Supports XY, XZ, YZ planes. No OpenCV, no vc_core.

#include "volume.hpp"
#include "types.hpp"

#include <QWidget>
#include <QImage>
#include <QTimer>
#include <QPainter>
#include <QMouseEvent>
#include <QWheelEvent>

#include <algorithm>
#include <cstdint>

namespace vc4d {

// Which orthogonal plane this viewer shows.
enum class Plane { XY, XZ, YZ };

class Viewer : public QWidget {
    Q_OBJECT

public:
    explicit Viewer(Plane plane = Plane::XY, QWidget* parent = nullptr)
        : QWidget(parent), _plane(plane) {
        setFocusPolicy(Qt::StrongFocus);
        setMouseTracking(true);
        _timer.setInterval(16);
        connect(&_timer, &QTimer::timeout, this, [this] {
            if (_dirty) { _dirty = false; render(); update(); }
        });
        _timer.start();
        vc::build_lut(_lut, 0, 255);
    }

    Plane plane() const { return _plane; }

    void set_volume(vc::Volume* vol) {
        _vol = vol;
        if (_vol) {
            auto s = _vol->shape();
            _focus = {float(s.x) / 2, float(s.y) / 2, float(s.z) / 2};
        }
        _dirty = true;
    }

    void set_window(float lo, float hi) {
        vc::build_lut(_lut, lo, hi);
        _dirty = true;
    }

    // Focus point — the 3D location this viewer is centered on.
    // Shared across all viewers for linked navigation.
    vc::Vec3f focus() const { return _focus; }
    void set_focus(vc::Vec3f f) { _focus = f; _dirty = true; }
    float zoom() const { return _zoom; }

    void mark_dirty() { _dirty = true; }

signals:
    // Emitted when scroll changes the "depth" axis for this plane.
    void focus_changed(vc::Vec3f new_focus);
    void cursor_world(vc::Vec3f world_pos);
    void clicked(vc::Vec3f world_pos, Qt::MouseButton button);

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        if (!_fb.isNull()) p.drawImage(0, 0, _fb);
        // Draw crosshair
        p.setPen(QPen(QColor(255, 255, 0, 80), 1));
        p.drawLine(width() / 2, 0, width() / 2, height());
        p.drawLine(0, height() / 2, width(), height() / 2);
        // Label
        const char* labels[] = {"XY", "XZ", "YZ"};
        p.setPen(Qt::white);
        p.drawText(8, 18, labels[int(_plane)]);
    }

    void resizeEvent(QResizeEvent*) override { _dirty = true; }

    void wheelEvent(QWheelEvent* e) override {
        int steps = e->angleDelta().y() / 120;
        if (e->modifiers() & Qt::ControlModifier) {
            _zoom *= (steps > 0) ? 1.1f : 0.9f;
            _zoom = std::clamp(_zoom, 0.01f, 100.0f);
        } else {
            // Scroll the depth axis for this plane
            float delta = float(steps) * std::max(1.0f, _zoom);
            switch (_plane) {
            case Plane::XY: _focus.z += delta; break;
            case Plane::XZ: _focus.y += delta; break;
            case Plane::YZ: _focus.x += delta; break;
            }
            emit focus_changed(_focus);
        }
        _dirty = true;
    }

    void mousePressEvent(QMouseEvent* e) override {
        if (e->button() == Qt::LeftButton) { _panning = true; _last_mouse = e->pos(); }
        emit clicked(screen_to_world(e->pos()), e->button());
    }

    void mouseMoveEvent(QMouseEvent* e) override {
        if (_panning) {
            auto d = e->pos() - _last_mouse;
            float dx = float(d.x()) * _zoom, dy = float(d.y()) * _zoom;
            switch (_plane) {
            case Plane::XY: _focus.x -= dx; _focus.y -= dy; break;
            case Plane::XZ: _focus.x -= dx; _focus.z -= dy; break;
            case Plane::YZ: _focus.y -= dx; _focus.z -= dy; break;
            }
            _last_mouse = e->pos();
            _dirty = true;
            emit focus_changed(_focus);
        }
        emit cursor_world(screen_to_world(e->pos()));
    }

    void mouseReleaseEvent(QMouseEvent* e) override {
        if (e->button() == Qt::LeftButton) _panning = false;
    }

private:
    void render() {
        if (!_vol) return;
        int w = width(), h = height();
        if (w <= 0 || h <= 0) return;
        if (_fb.width() != w || _fb.height() != h)
            _fb = QImage(w, h, QImage::Format_ARGB32);

        vc::Vec3f org{}, vx{}, vy{};
        float hw = float(w / 2) * _zoom, hh = float(h / 2) * _zoom;

        switch (_plane) {
        case Plane::XY:
            org = {_focus.x - hw, _focus.y - hh, _focus.z};
            vx = {_zoom, 0, 0}; vy = {0, _zoom, 0};
            break;
        case Plane::XZ:
            org = {_focus.x - hw, _focus.y, _focus.z - hh};
            vx = {_zoom, 0, 0}; vy = {0, 0, _zoom};
            break;
        case Plane::YZ:
            org = {_focus.x, _focus.y - hw, _focus.z - hh};
            vx = {0, _zoom, 0}; vy = {0, 0, _zoom};
            break;
        }

        auto* bits = reinterpret_cast<uint32_t*>(_fb.bits());
        _vol->sample_plane_argb32(bits, _fb.bytesPerLine() / 4, org, vx, vy, w, h, _lut);
    }

    vc::Vec3f screen_to_world(QPoint s) const {
        float sx = (float(s.x()) - float(width()) / 2) * _zoom;
        float sy = (float(s.y()) - float(height()) / 2) * _zoom;
        switch (_plane) {
        case Plane::XY: return {_focus.x + sx, _focus.y + sy, _focus.z};
        case Plane::XZ: return {_focus.x + sx, _focus.y, _focus.z + sy};
        case Plane::YZ: return {_focus.x, _focus.y + sx, _focus.z + sy};
        }
        __builtin_unreachable();
    }

    vc::Volume* _vol = nullptr;
    QImage _fb;
    QTimer _timer;
    Plane _plane;
    bool _dirty = true;

    vc::Vec3f _focus{0, 0, 0};
    float _zoom = 1.0f;
    uint32_t _lut[256]{};

    bool _panning = false;
    QPoint _last_mouse;
};

} // namespace vc4d
