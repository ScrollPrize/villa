#pragma once

// Camera pad for the volumetric composite mode. Lives as a child widget of
// the viewer's CVolumeViewerView (centered in the viewport, does not pan or
// zoom with the scene). Three independent panes:
//  - left: azimuth dial (compass) — drag rotates the in-plane tilt direction
//  - middle: elevation gauge — drag sets the tilt angle away from the surface
//    normal (needle vertical = straight down, 0..45 degrees)
//  - right: perspective gauge — drag sets the perspective strength (0..1)
// Double-click resets the pane under the cursor.

#include <QWidget>

class CameraGizmoWidget : public QWidget
{
    Q_OBJECT

public:
    static constexpr float kMaxTiltDeg = 45.0f;

    explicit CameraGizmoWidget(QWidget* parent = nullptr);

    // Update the displayed state without emitting cameraChanged.
    void setCamera(float azimuthDeg, float tiltDeg, float perspective);
    float azimuthDeg() const { return _azimuthDeg; }
    float tiltDeg() const { return _tiltDeg; }
    float perspective() const { return _perspective; }

signals:
    void cameraChanged(float azimuthDeg, float tiltDeg, float perspective);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    enum class Pane { None, Azimuth, Elevation, Perspective };

    Pane paneAt(const QPointF& pos) const;
    void updateFromDrag(const QPointF& pos);
    void repositionInParent();
    QPointF azimuthCenter() const;
    QPointF elevationCenter() const;
    QRectF perspectiveTrackRect() const;
    double dialRadius() const;

    float _azimuthDeg = 0.0f;
    float _tiltDeg = 0.0f;
    float _perspective = 0.0f;
    Pane _dragPane = Pane::None;
};
