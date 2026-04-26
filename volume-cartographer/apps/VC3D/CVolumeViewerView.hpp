#pragma once

#include <QGraphicsView>

class CVolumeViewerView : public QGraphicsView
{
    Q_OBJECT
    
public:
    CVolumeViewerView(QWidget* parent = 0);
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void scrollContentsBy(int dx, int dy) override;
    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    /// Set physical voxel size (units per scene-unit, e.g. µm/pixel).
    /// Call this after you load your Zarr spacing metadata.
    void setVoxelSize(double sx, double sy) { m_vx = sx; m_vy = sy; update(); }
    void setMiddleButtonPanEnabled(bool enabled) { _middleButtonPanEnabled = enabled; }
    bool middleButtonPanEnabled() const { return _middleButtonPanEnabled; }
    void setScrollPanDisabled(bool disabled) { _scrollPanDisabled = disabled; }

signals:
    void sendResized();
    void sendScrolled();
    void sendZoom(int steps, QPointF scene_point, Qt::KeyboardModifiers);
    void sendVolumeClicked(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void sendPanRelease(Qt::MouseButton, Qt::KeyboardModifiers);
    void sendPanStart(Qt::MouseButton, Qt::KeyboardModifiers);
    void sendMousePress(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void sendMouseMove(QPointF, Qt::MouseButtons, Qt::KeyboardModifiers);
    void sendMouseRelease(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void sendKeyPress(int key, Qt::KeyboardModifiers modifiers);
    void sendKeyRelease(int key, Qt::KeyboardModifiers modifiers);
    
protected:
    bool _regular_pan = false;
    QPoint _last_pan_position;
    bool _left_button_pressed = false;
    /// Draw our scalebar on every repaint
    void drawForeground(QPainter* painter, const QRectF& sceneRect) override;
    /// Paint framebuffer directly, bypassing QGraphicsPixmapItem
    void drawBackground(QPainter* painter, const QRectF& rect) override;

public:
    void setDirectFramebuffer(const QImage* fb) { _directFb = fb; }
private:
    const QImage* _directFb = nullptr;

 private:
    /// Round “ideal” length to 1,2 or 5 × 10^n
    double chooseNiceLength(double nominal) const;

    // µm per scene-unit (pixel)
    double m_vx = 32.0, m_vy = 32.0;
    bool _middleButtonPanEnabled = true;
    bool _scrollPanDisabled = false;
    int _wheelAccum = 0;  // fractional wheel delta accumulator

    // Scalebar cache — avoid recomputing font/text every frame
    mutable QFont _cachedFont;
    mutable bool _scalebarCacheDirty = true;
    mutable double _cachedBarPx = 0;
    mutable QString _cachedBarLabel;
    mutable double _cachedM11 = 0;
    mutable double _cachedDpr = 0;
    mutable int _cachedVpW = 0;
    mutable int _cachedVpH = 0;
    mutable double _cachedVx = 0;
};
