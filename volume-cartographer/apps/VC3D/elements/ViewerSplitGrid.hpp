#pragma once

#include <array>
#include <functional>

#include <QPoint>
#include <QWidget>

class QEvent;
class QFrame;
class QResizeEvent;

// Reusable four-pane grid used by the Main and Spiral workspaces. Panes are
// ordered top-left, top-right, bottom-left, bottom-right.
class ViewerSplitGrid : public QWidget
{
public:
    explicit ViewerSplitGrid(QWidget* parent = nullptr);

    void setViewer(int index, QWidget* widget);
    QWidget* viewer(int index) const;
    int indexOf(QWidget* widget) const;
    void swapViewers(int first, int second);
    void setPaneHidden(int index, bool hidden);
    bool paneHidden(int index) const;
    bool fullSizeActive() const;
    bool fullSizeActiveForPane(int index) const;
    void setFullSizePane(int index);
    void exitFullSize();
    void resetSplits();
    void setSplits(double splitX, double splitY);
    double splitX() const { return _splitX; }
    double splitY() const { return _splitY; }

    std::function<void()> onSplitChanged;

protected:
    void resizeEvent(QResizeEvent*) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    enum class HandleKind { None, Column, Row, Both };
    QFrame* makeHandle(Qt::CursorShape cursor);
    HandleKind handleKind(QObject* object) const;
    int splitXPx() const;
    int splitYPx() const;
    int clampSplitPx(int value, int extent) const;
    int handleWidth() const { return 5; }
    void layoutChildren();
    void layoutColumn(int topIndex, int bottomIndex, const QRect& columnRect, int splitY);
    void applyVisibility();
    void setViewerGeometry(int index, const QRect& rect);
    bool paneVisible(int index) const;
    int visiblePaneCount() const;
    void notifySplitChanged();

    QWidget* _viewers[4] = {};
    std::array<bool, 4> _hidden{};
    std::array<bool, 4> _savedFullSizeHidden{};
    QFrame* _topColumnHandle = nullptr;
    QFrame* _bottomColumnHandle = nullptr;
    QFrame* _leftRowHandle = nullptr;
    QFrame* _rightRowHandle = nullptr;
    QFrame* _centerHandle = nullptr;
    double _splitX = 0.5;
    double _splitY = 0.5;
    double _savedFullSizeSplitX = 0.5;
    double _savedFullSizeSplitY = 0.5;
    int _fullSizePane = -1;
    static constexpr int _minPanePx = 80;
    HandleKind _dragging = HandleKind::None;
    QPoint _dragStartGlobal;
    int _dragStartSplitX = 0;
    int _dragStartSplitY = 0;
};
