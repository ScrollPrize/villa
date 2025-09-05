#pragma once

#include <QGraphicsView>

class CVolumeViewerView : public QGraphicsView
{
    Q_OBJECT
    
public:
    CVolumeViewerView(QWidget* parent = 0);
    void mouseReleaseEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void scrollContentsBy(int dx, int dy);
    void keyPressEvent(QKeyEvent *event);
    void keyReleaseEvent(QKeyEvent *event);
    void resizeEvent(QResizeEvent *event) override;
    /// Set physical voxel size (units per scene-unit, e.g. µm/pixel).
    /// Call this after you load your Zarr spacing metadata.
    void setVoxelSize(double sx, double sy) { m_vx = sx; m_vy = sy; update(); }

signals:
    void sendResized();
    void sendScrolled();
    void sendZoom(int steps, QPointF scene_point, Qt::KeyboardModifiers);
    void sendVolumeClicked(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void sendPanRelease(Qt::MouseButton, Qt::KeyboardModifiers);
    void sendPanStart(Qt::MouseButton, Qt::KeyboardModifiers);
    void sendCursorMove(QPointF);
    void sendMousePress(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void sendMouseMove(QPointF, Qt::MouseButtons, Qt::KeyboardModifiers);
    void sendMouseRelease(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void sendKeyRelease(int key, Qt::KeyboardModifiers modifiers);
    
protected:
    bool _regular_pan = false;
    QPoint _last_pan_position;
    bool _left_button_pressed = false;
    /// Draw our scalebar on every repaint
    void drawForeground(QPainter* painter, const QRectF& sceneRect) override;

 private:
    /// Round “ideal” length to 1,2 or 5 × 10^n
    double chooseNiceLength(double nominal) const;

    // µm per scene-unit (pixel)  
    double m_vx = 32.0, m_vy = 32.0;
};

