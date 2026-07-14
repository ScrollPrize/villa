#include "elements/ViewerSplitGrid.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <QEvent>
#include <QFrame>
#include <QMouseEvent>
#include <QResizeEvent>

ViewerSplitGrid::ViewerSplitGrid(QWidget* parent) : QWidget(parent)
{
    setContentsMargins(0, 0, 0, 0);
    _topColumnHandle = makeHandle(Qt::SplitHCursor);
    _bottomColumnHandle = makeHandle(Qt::SplitHCursor);
    _leftRowHandle = makeHandle(Qt::SplitVCursor);
    _rightRowHandle = makeHandle(Qt::SplitVCursor);
    _centerHandle = makeHandle(Qt::SizeAllCursor);
}

void ViewerSplitGrid::setViewer(int index, QWidget* widget)
{
    if (index < 0 || index >= 4) return;
    if (_viewers[index] && _viewers[index] != widget) _viewers[index]->hide();
    if (widget) {
        for (int i = 0; i < 4; ++i) if (i != index && _viewers[i] == widget) _viewers[i] = nullptr;
        widget->setParent(this);
        widget->setVisible(!_hidden[index]);
    }
    _viewers[index] = widget;
    layoutChildren();
}

QWidget* ViewerSplitGrid::viewer(int index) const
{
    return index >= 0 && index < 4 ? _viewers[index] : nullptr;
}

int ViewerSplitGrid::indexOf(QWidget* widget) const
{
    if (!widget) return -1;
    for (int i = 0; i < 4; ++i) if (_viewers[i] == widget) return i;
    return -1;
}

void ViewerSplitGrid::swapViewers(int first, int second)
{
    if (first < 0 || first >= 4 || second < 0 || second >= 4 || first == second) return;
    std::swap(_viewers[first], _viewers[second]);
    layoutChildren();
}

void ViewerSplitGrid::setPaneHidden(int index, bool hidden)
{
    if (index < 0 || index >= 4) return;
    if (hidden && visiblePaneCount() <= 1 && !_hidden[index]) return;
    _hidden[index] = hidden;
    if (_viewers[index]) _viewers[index]->setVisible(!hidden);
    layoutChildren();
    notifySplitChanged();
}

bool ViewerSplitGrid::paneHidden(int index) const
{
    return index >= 0 && index < 4 ? _hidden[index] : false;
}

bool ViewerSplitGrid::fullSizeActive() const { return _fullSizePane >= 0; }
bool ViewerSplitGrid::fullSizeActiveForPane(int index) const { return _fullSizePane == index; }

void ViewerSplitGrid::setFullSizePane(int index)
{
    if (index < 0 || index >= 4 || !_viewers[index]) return;
    if (_fullSizePane < 0) {
        _savedFullSizeHidden = _hidden;
        _savedFullSizeSplitX = _splitX;
        _savedFullSizeSplitY = _splitY;
    }
    _fullSizePane = index;
    for (int pane = 0; pane < 4; ++pane) _hidden[pane] = pane != index;
    applyVisibility();
    layoutChildren();
}

void ViewerSplitGrid::exitFullSize()
{
    if (_fullSizePane < 0) return;
    _hidden = _savedFullSizeHidden;
    _splitX = _savedFullSizeSplitX;
    _splitY = _savedFullSizeSplitY;
    _fullSizePane = -1;
    applyVisibility();
    layoutChildren();
}

void ViewerSplitGrid::resetSplits()
{
    _fullSizePane = -1;
    _splitX = _splitY = 0.5;
    layoutChildren();
    notifySplitChanged();
}

void ViewerSplitGrid::setSplits(double splitX, double splitY)
{
    _splitX = std::clamp(splitX, 0.1, 0.9);
    _splitY = std::clamp(splitY, 0.1, 0.9);
    layoutChildren();
}

void ViewerSplitGrid::resizeEvent(QResizeEvent*) { layoutChildren(); }

bool ViewerSplitGrid::eventFilter(QObject* watched, QEvent* event)
{
    const auto handle = handleKind(watched);
    if (handle == HandleKind::None) return QWidget::eventFilter(watched, event);
    if (event->type() == QEvent::MouseButtonPress) {
        auto* mouse = static_cast<QMouseEvent*>(event);
        if (mouse->button() != Qt::LeftButton) return false;
        _dragging = handle;
        _dragStartGlobal = mouse->globalPosition().toPoint();
        _dragStartSplitX = splitXPx();
        _dragStartSplitY = splitYPx();
        event->accept();
        return true;
    }
    if (event->type() == QEvent::MouseMove && _dragging != HandleKind::None) {
        auto* mouse = static_cast<QMouseEvent*>(event);
        const QPoint delta = mouse->globalPosition().toPoint() - _dragStartGlobal;
        if (_dragging == HandleKind::Column || _dragging == HandleKind::Both)
            _splitX = static_cast<double>(clampSplitPx(_dragStartSplitX + delta.x(), std::max(1, width()))) / std::max(1, width());
        if (_dragging == HandleKind::Row || _dragging == HandleKind::Both)
            _splitY = static_cast<double>(clampSplitPx(_dragStartSplitY + delta.y(), std::max(1, height()))) / std::max(1, height());
        layoutChildren();
        notifySplitChanged();
        event->accept();
        return true;
    }
    if (event->type() == QEvent::MouseButtonRelease && _dragging != HandleKind::None) {
        _dragging = HandleKind::None;
        notifySplitChanged();
        event->accept();
        return true;
    }
    return QWidget::eventFilter(watched, event);
}

QFrame* ViewerSplitGrid::makeHandle(Qt::CursorShape cursor)
{
    auto* handle = new QFrame(this);
    handle->setFrameShape(QFrame::NoFrame);
    handle->setCursor(cursor);
    handle->setAutoFillBackground(true);
    handle->setStyleSheet(QStringLiteral("background: rgba(80, 80, 80, 96);"));
    handle->installEventFilter(this);
    handle->show();
    return handle;
}

ViewerSplitGrid::HandleKind ViewerSplitGrid::handleKind(QObject* object) const
{
    if (object == _topColumnHandle || object == _bottomColumnHandle) return HandleKind::Column;
    if (object == _leftRowHandle || object == _rightRowHandle) return HandleKind::Row;
    if (object == _centerHandle) return HandleKind::Both;
    return HandleKind::None;
}

int ViewerSplitGrid::splitXPx() const { return clampSplitPx(static_cast<int>(std::lround(_splitX * width())), width()); }
int ViewerSplitGrid::splitYPx() const { return clampSplitPx(static_cast<int>(std::lround(_splitY * height())), height()); }

int ViewerSplitGrid::clampSplitPx(int value, int extent) const
{
    const int half = handleWidth() / 2;
    const int minValue = std::min(std::max(_minPanePx + half, half), extent / 2);
    return std::clamp(value, minValue, std::max(minValue, extent - minValue));
}

void ViewerSplitGrid::layoutChildren()
{
    const int w = width(), h = height();
    if (w <= 0 || h <= 0) return;
    const int handle = handleWidth(), half = handle / 2;
    const int sx = splitXPx(), sy = splitYPx();
    const bool left = paneVisible(0) || paneVisible(2);
    const bool right = paneVisible(1) || paneVisible(3);
    layoutColumn(0, 2, QRect(0, 0, right ? sx : w, h), sy);
    layoutColumn(1, 3, QRect(left ? sx : 0, 0, left ? w - sx : w, h), sy);
    const bool tl = paneVisible(0), bl = paneVisible(2), tr = paneVisible(1), br = paneVisible(3);
    _topColumnHandle->setVisible(left && right && (tl || tr));
    _bottomColumnHandle->setVisible(left && right && (bl || br));
    _leftRowHandle->setVisible(tl && bl);
    _rightRowHandle->setVisible(tr && br);
    _centerHandle->setVisible(left && right && (tl || tr) && (bl || br));
    _topColumnHandle->setGeometry(sx - half, 0, handle, sy);
    _bottomColumnHandle->setGeometry(sx - half, sy, handle, h - sy);
    _leftRowHandle->setGeometry(0, sy - half, sx, handle);
    _rightRowHandle->setGeometry(sx, sy - half, w - sx, handle);
    _centerHandle->setGeometry(sx - half, sy - half, handle, handle);
    for (auto* widget : {_topColumnHandle, _bottomColumnHandle, _leftRowHandle, _rightRowHandle, _centerHandle}) widget->raise();
}

void ViewerSplitGrid::layoutColumn(int top, int bottom, const QRect& rect, int splitY)
{
    const bool topVisible = paneVisible(top), bottomVisible = paneVisible(bottom);
    if (topVisible && bottomVisible) {
        setViewerGeometry(top, QRect(rect.x(), 0, rect.width(), splitY));
        setViewerGeometry(bottom, QRect(rect.x(), splitY, rect.width(), height() - splitY));
    } else if (topVisible) setViewerGeometry(top, rect);
    else if (bottomVisible) setViewerGeometry(bottom, rect);
    if (_viewers[top]) _viewers[top]->setVisible(topVisible);
    if (_viewers[bottom]) _viewers[bottom]->setVisible(bottomVisible);
}

void ViewerSplitGrid::applyVisibility() { for (int i = 0; i < 4; ++i) if (_viewers[i]) _viewers[i]->setVisible(!_hidden[i]); }
void ViewerSplitGrid::setViewerGeometry(int i, const QRect& rect) { if (i >= 0 && i < 4 && _viewers[i]) _viewers[i]->setGeometry(rect.normalized()); }
bool ViewerSplitGrid::paneVisible(int i) const { return i >= 0 && i < 4 && _viewers[i] && !_hidden[i]; }
int ViewerSplitGrid::visiblePaneCount() const { int n = 0; for (int i = 0; i < 4; ++i) n += paneVisible(i); return n; }
void ViewerSplitGrid::notifySplitChanged() { if (onSplitChanged) onSplitChanged(); }
