#pragma once

#include <array>
#include <cstdint>
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
    // physical=false: sx is voxels (not µm) per scene px; the scalebar
    // labels in "vx" instead of physical units.
    void setVoxelSize(double sx, bool physical = true)
    {
        m_vx = sx;
        m_physicalUnits = physical;
        update();
    }

    bool physicalUnits() const { return m_physicalUnits; }

    void setMiddleButtonPanEnabled(bool enabled) { _middleButtonPanEnabled = enabled; }
    bool middleButtonPanEnabled() const { return _middleButtonPanEnabled; }
    
    void showCoordinateFrame(bool visible);
    void setCoordinateFrame(PlaneSurface* plane);

    // Measurement overlays drawn from the rendered frame (see ViewerHud.hpp).
    void setHudOptions(const vc3d::hud::Options& options);
    const vc3d::hud::Options& hudOptions() const { return _hud; }
    // Line for the intensity profile, in viewport pixels; nullopt clears it.
    void setProfileLine(std::optional<QLineF> line);
    std::optional<QLineF> profileLine() const { return _profileLine; }
    // Drawing a profile line takes one left drag after it is armed (View > Measurement
    // Overlays > Draw Profile Line), so the overlay never takes clicks from the tools
    // and embedded controls that own them. The drag, or Escape, disarms it.
    void armProfileLine(bool armed);
    bool profileLineArmed() const { return _profileArmed; }

    // The rendered frame the overlays measure. Only a frame that shows the live view is
    // measured: one still drawn for an earlier pan, zoom, offset, camera or surface is not.
    struct MeasurementFrame {
        std::uint64_t serial = 0;
        cv::Mat_<std::uint8_t> values;    // base-volume samples, 8-bit, before the display window
        // Nonzero where the frame measured the volume: sampled, inside the volume (not
        // the reader's fill beyond its edges) and from this frame's own geometry (not a
        // pixel carried over from an earlier surface while chunks load).
        cv::Mat_<std::uint8_t> coverage;
        double windowLow = 0.0;
        double windowHigh = 255.0;
        // The gray the renderer's window gives each value; the histogram counts black
        // and white from it.
        vc3d::hud::DisplayMap displayed = vc3d::hud::identityDisplay();
        bool overlayBlended = false;      // a colored overlay volume is drawn over these samples
        // Set when each value reduces several slices along the normal (a Max, Mean or
        // other slab composite), e.g. "max of 9 slices". Such a value has no single
        // position, so the profile and loupe do not report one.
        QString composite;
        // False while some of the frame's chunks are still loading.
        bool complete = true;
        // False when the frame has positions but no measurement mask yet (it was
        // rendered before an overlay that reads samples was turned on): only the grid
        // may use it.
        bool samples = true;
        // Frame pixel to volume position (level-0 voxels) from the geometry this frame
        // was sampled with, including the normal offset (Ctrl+. and Ctrl+,); NaN where
        // the pixel is off the surface.
        std::function<cv::Vec3f(const QPointF&)> toVolume;
        // Set on plane views, where toVolume is affine; the grid needs it.
        std::optional<vc3d::hud::AffineMap> planeMap;
        double voxelSizeUm = 0.0;  // > 0 only when the volume states its voxel size
    };
    struct FrameResult {
        std::optional<MeasurementFrame> frame;
        QString unavailable;  // why there is no frame to measure, shown in place of a readout
    };
    void setMeasurementFrameSource(std::function<FrameResult()> source);

    // Numbers behind the line profile, for scripts and tests.
    struct ProfileReadout {
        bool available = false;
        QString unavailable;          // why not, when !available
        std::vector<float> values;    // 8-bit samples before the window; NaN where not measured
        int missing = 0;              // samples that are NaN
        bool complete = true;         // false while some of the frame's chunks are still loading
        std::vector<int> peaks;
        std::vector<double> arc;      // length from the start to each sample; NaN once a position is missing
        std::optional<double> length; // arc.back() when every position is known
        bool physical = false;        // lengths in µm (true) or level-0 voxels
        bool alongSurface = false;    // along a curved surface rather than a plane
        std::optional<std::array<double, 3>> start;
        std::optional<std::array<double, 3>> end;
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
    // A drag began drawing the profile line, which disarms the other views.
    void profileLineDrawStarted();
    // Escape cancelled arming or drawing the profile line, which disarms every view.
    void profileLineCancelled();
    
protected:
    bool _regular_pan = false;
    QPoint _last_pan_position;
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
    // The live frame; with needSamples, only one whose samples may be measured.
    FrameResult measurementFrame(bool needSamples = true) const;
    vc3d::hud::PanelLayout panelLayout() const;
    QRectF scaleBarRect() const;
    void cancelProfileGesture();
    bool pointInSceneWidget(const QPointF& viewportPos) const;

    // µm (or voxels, when !m_physicalUnits) per scene-unit (pixel)
    double m_vx = 32.0;
    bool m_physicalUnits = true;
    bool _middleButtonPanEnabled = true;
    bool _sceneWidgetMouseCapture = false;
    int _wheelAccum = 0;  // fractional wheel delta accumulator
    Qt::KeyboardModifiers _wheelAccumModifiers = Qt::NoModifier;  // modifiers _wheelAccum was gathered under
    mutable QFont _cachedFont;
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
    bool _profileArmed = false;
    bool _profileDragging = false;
    // Escape cancelled a profile drag: the rest of that drag (moves and the release)
    // stays with the profile instead of reaching the tools.
    bool _profileDragCancelled = false;
    // Escape was taken by the profile: its auto-repeats and release are taken too.
    bool _profileOwnsEscape = false;
    QPointF _lastCursorVp{-1.0, -1.0};
    std::function<FrameResult()> _frameSource;
    mutable std::uint64_t _histogramSerial = 0;
    mutable vc3d::hud::Histogram _histogram;
};
