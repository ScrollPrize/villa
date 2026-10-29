#pragma once

#include <array>
#include <functional>
#include <optional>

#include <QGraphicsView>
#include <QPointF>
#include <QString>
#include <QVector3D>

#include <opencv2/core.hpp>
#include "vc/core/util/PlaneSurface.hpp"
#include "ViewerHud.hpp"

class CVolumeViewerView : public QGraphicsView
{
    Q_OBJECT
    
public:
    CVolumeViewerView(QWidget* parent = 0);
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void leaveEvent(QEvent *event) override;
    bool viewportEvent(QEvent* event) override;
    void wheelEvent(QWheelEvent *event) override;
    void paintEvent(QPaintEvent* event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    // physical=false: sx/sy are voxels (not µm) per scene px; the scalebar
    // labels in "vx" instead of physical units.
    void setVoxelSize(double sx, double sy, bool physical = true)
    {
        m_vx = sx;
        m_vy = sy;
        m_physicalUnits = physical;
        update();
    }

    bool physicalUnits() const { return m_physicalUnits; }

    void setMiddleButtonPanEnabled(bool enabled) { _middleButtonPanEnabled = enabled; }
    bool middleButtonPanEnabled() const { return _middleButtonPanEnabled; }
    void setScrollPanDisabled(bool disabled) { _scrollPanDisabled = disabled; }
    
    void showCoordinateFrame(bool visible);
    void setCoordinateFrame(PlaneSurface* plane);

    // Measurement overlays drawn from the rendered frame (see ViewerHud.hpp).
    void setHudOptions(const vc3d::hud::Options& options);
    const vc3d::hud::Options& hudOptions() const { return _hud; }
    // Line for the intensity profile, in viewport pixels; nullopt clears it.
    void setProfileLine(std::optional<QLineF> line);
    std::optional<QLineF> profileLine() const { return _profileLine; }
    // How the overlays find volume positions: `sceneToVolume` gives NaN when the viewer
    // shows nothing; `planar` says whether that map is affine (a plane view), which the
    // grid needs; `extent` is the volume size in voxels (x, y, z), so the histogram and
    // profile can leave out what lies outside the volume.
    struct VolumeMapping {
        std::function<cv::Vec3f(const QPointF&)> sceneToVolume;
        std::function<bool()> planar;
        std::function<std::optional<cv::Vec3d>()> extent;
    };
    void setVolumeMapping(VolumeMapping mapping);
    // Numbers behind the line profile as last drawn, for scripts and tests.
    struct ProfileReadout {
        std::vector<float> values;
        std::vector<int> peaks;
        double lengthUnits = 0.0;  // in the scalebar's units (µm, or voxels)
    };
    ProfileReadout profileReadout() const;

    struct ScaleBarLabel {
        double displayLength = 0.0;
        QString unit;
        QString text;
    };
    static ScaleBarLabel formatScaleBarLength(double barUm);
    
    enum class TiltHandleMode {
        Hidden,
        Square,
        SemiCircleX,
        SemiCircleY,
    };
    void setTiltHandle(TiltHandleMode mode, QPointF tilt);

signals:
    void sendResized();
    void sendZoom(int steps, QPointF scene_point, Qt::KeyboardModifiers);
    void sendVolumeClicked(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void sendPanRelease(Qt::MouseButton, Qt::KeyboardModifiers);
    void sendPanStart(Qt::MouseButton, Qt::KeyboardModifiers);
    void sendCursorMove(QPointF);
    void sendMousePress(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void sendMouseDoubleClick(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void sendMouseMove(QPointF, Qt::MouseButtons, Qt::KeyboardModifiers);
    void sendMouseRelease(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void sendMouseLeftView();
    void sendAnnotationContextMenuRequested(QPointF scenePoint, QPoint globalPos, Qt::KeyboardModifiers modifiers);
    void sendKeyPress(int key, Qt::KeyboardModifiers modifiers);
    void sendKeyRelease(int key, Qt::KeyboardModifiers modifiers);
    void sendTiltHandleChanged(QPointF tilt);
    void sendTiltHandleReset();
    void paintCompleted();
    
protected:
    bool _regular_pan = false;
    QPoint _last_pan_position;
    bool _left_button_pressed = false;
    bool _right_button_mouse_forwarded = false;
    bool _tiltHandleDragging = false;
    /// Draw our scalebar on every repaint
    void drawForeground(QPainter* painter, const QRectF& sceneRect) override;
    /// Paint framebuffer directly, bypassing QGraphicsPixmapItem
    void drawBackground(QPainter* painter, const QRectF& rect) override;

public:
    void setDirectFramebuffer(const QImage* fb) { _directFb = fb; }
    void setDirectFramebufferMapping(qreal scale, QPointF offset)
    {
        _directFbScale = scale;
        _directFbOffset = offset;
    }
private:
    const QImage* _directFb = nullptr;
    qreal _directFbScale = 1.0;
    QPointF _directFbOffset{0.0, 0.0};

 private:
    /// Round “ideal” length to 1,2 or 5 × 10^n
    double chooseNiceLength(double nominal) const;
    QRectF tiltHandleRect() const;
    bool pointInTiltHandle(const QPointF& viewportPos) const;
    QPointF tiltFromHandlePos(const QPointF& viewportPos) const;
    void drawCoordinateFrame(QPainter* painter) const;
    void drawTiltHandle(QPainter* painter) const;
    void drawScaleBar(QPainter* painter) const;
    void drawHistogram(QPainter* painter) const;
    void drawLineProfile(QPainter* painter) const;
    void drawGrid(QPainter* painter) const;
    void drawLoupe(QPainter* painter) const;
    QPointF viewportToFramebuffer(const QPointF& vp) const;
    std::optional<vc3d::hud::AffineMap> viewportToVolumeMap() const;
    bool showsVolume() const;
    // True for viewport points inside the volume; empty when that cannot be told.
    std::function<bool(QPointF)> insideVolume() const;
    bool pointInSceneWidget(const QPointF& viewportPos) const;

    // µm (or voxels, when !m_physicalUnits) per scene-unit (pixel)
    double m_vx = 32.0, m_vy = 32.0;
    bool m_physicalUnits = true;
    bool _middleButtonPanEnabled = true;
    bool _scrollPanDisabled = false;
    bool _sceneWidgetMouseCapture = false;
    int _wheelAccum = 0;  // fractional wheel delta accumulator
    Qt::KeyboardModifiers _wheelAccumModifiers = Qt::NoModifier;  // modifiers _wheelAccum was gathered under
    mutable QFont _cachedFont;
    mutable bool _scalebarCacheDirty = true;
    mutable double _cachedBarPx = 0;
    mutable QString _cachedBarLabel;
    mutable double _cachedM11 = 0;
    mutable int _cachedVpW = 0;
    mutable int _cachedVpH = 0;
    mutable bool _cachedPhysicalUnits = true;
    mutable double _cachedVx = 0;

    TiltHandleMode _tiltHandleMode = TiltHandleMode::Hidden;
    QPointF _tiltHandleValue{0.0, 0.0};
    
    bool _showFrame = false;
    cv::Vec3f _frame[3];

    vc3d::hud::Options _hud;
    std::optional<QLineF> _profileLine;
    bool _profileDragging = false;
    QPointF _lastCursorVp{-1.0, -1.0};
    VolumeMapping _mapping;
    mutable qint64 _histogramKey = 0;
    mutable vc3d::hud::Histogram _histogram;
};
