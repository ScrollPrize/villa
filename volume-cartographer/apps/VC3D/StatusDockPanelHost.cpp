#include "StatusDockPanelHost.hpp"

#include <QApplication>
#include <QDockWidget>
#include <QEvent>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QMenu>
#include <QMouseEvent>
#include <QPushButton>
#include <QResizeEvent>
#include <QStackedWidget>
#include <QStyle>
#include <QToolButton>
#include <QVBoxLayout>

#include <algorithm>
#include <cstddef>

namespace {
constexpr int kResizeHitWidth = 10;
}

StatusDockPanelHost::StatusDockPanelHost(QWidget* centralWidget, QWidget* parent)
    : QWidget(parent)
{
    setObjectName(QStringLiteral("statusDockPanelHost"));

    _layout = new QVBoxLayout(this);
    _layout->setContentsMargins(0, 0, 0, 0);
    _layout->setSpacing(0);

    if (centralWidget) {
        centralWidget->setParent(this);
        _layout->addWidget(centralWidget, 1);
    }

    _panelFrame = new QFrame(this);
    _panelFrame->setObjectName(QStringLiteral("statusDockPanelFrame"));
    _panelFrame->setFrameShape(QFrame::StyledPanel);
    _panelFrame->setWindowFlags(Qt::Widget);
    _panelFrame->setMouseTracking(true);
    _panelFrame->setAttribute(Qt::WA_StyledBackground, true);
    _panelFrame->hide();

    auto* panelLayout = new QVBoxLayout(_panelFrame);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(0);

    _stack = new QStackedWidget(_panelFrame);
    panelLayout->addWidget(_stack, 1);

    _barFrame = new QFrame(this);
    _barFrame->setObjectName(QStringLiteral("statusDockBar"));
    _barFrame->setFrameShape(QFrame::NoFrame);
    _barLayout = new QHBoxLayout(_barFrame);
    _barLayout->setContentsMargins(4, 3, 4, 3);
    _barLayout->setSpacing(4);
    _barLayout->addStretch(1);
    _layout->addWidget(_barFrame, 0);

    setStyleSheet(QStringLiteral(
        "QFrame#statusDockBar { background: palette(window); border-top: 1px solid palette(mid); }"
        "QFrame#statusDockPanelFrame { background: palette(window); border-left: 1px solid palette(mid);"
        " border-bottom: 1px solid palette(mid); border-top: 5px solid palette(mid);"
        " border-right: 5px solid palette(mid); }"
        "QWidget[statusDockPanelPage=\"true\"] { background: palette(window); }"
        "QFrame[statusDockPanelHeader=\"true\"] { background: palette(window);"
        " border-bottom: 1px solid palette(mid); }"
        "QPushButton[statusDockPanelButton=\"true\"] { padding: 3px 8px; text-align: left; }"
        "QPushButton[statusDockPanelButton=\"true\"][active=\"true\"] { font-weight: 600; }"));

    if (qApp) {
        qApp->installEventFilter(this);
    }
    _panelFrame->installEventFilter(this);
}

StatusDockPanelHost::~StatusDockPanelHost()
{
    if (qApp) {
        qApp->removeEventFilter(this);
    }
}

QWidget* StatusDockPanelHost::takeBarWidget()
{
    if (_barFrame && _layout) {
        _layout->removeWidget(_barFrame);
    }
    return _barFrame;
}

void StatusDockPanelHost::addDock(QDockWidget* dock)
{
    if (!dock || itemForDock(dock)) {
        return;
    }

    QWidget* content = dock->widget();
    if (!content) {
        return;
    }

    if (auto* mainWindow = qobject_cast<QMainWindow*>(dock->parentWidget())) {
        mainWindow->removeDockWidget(dock);
    }
    dock->setWidget(nullptr);
    dock->hide();

    Item item;
    item.dock = dock;
    item.content = content;

    auto* page = new QWidget(_stack);
    page->setObjectName(QStringLiteral("statusDockPanelPage_%1").arg(dock->objectName()));
    page->setProperty("statusDockPanelPage", true);
    page->setAutoFillBackground(true);
    page->setAttribute(Qt::WA_StyledBackground, true);
    auto* pageLayout = new QVBoxLayout(page);
    pageLayout->setContentsMargins(0, 0, 0, 0);
    pageLayout->setSpacing(0);

    auto* header = new QFrame(page);
    header->setProperty("statusDockPanelHeader", true);
    header->setAutoFillBackground(true);
    header->setAttribute(Qt::WA_StyledBackground, true);
    header->setFrameShape(QFrame::NoFrame);
    auto* headerLayout = new QHBoxLayout(header);
    headerLayout->setContentsMargins(8, 4, 6, 4);
    headerLayout->setSpacing(6);

    auto* title = new QLabel(dock->windowTitle(), header);
    title->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    headerLayout->addWidget(title);

    auto* collapse = new QToolButton(header);
    collapse->setText(QStringLiteral("▼"));
    collapse->setToolTip(tr("Collapse"));
    headerLayout->addWidget(collapse);
    pageLayout->addWidget(header, 0);

    content->setParent(page);
    pageLayout->addWidget(content, 1);
    content->show();
    page->show();

    auto* button = new QPushButton(this);
    button->setProperty("statusDockPanelButton", true);
    button->setContextMenuPolicy(Qt::CustomContextMenu);
    button->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    button->setToolTip(tr("Left-click to expand. Right-click for pin and detach options."));

    item.page = page;
    item.button = button;
    _items.push_back(item);
    const int index = static_cast<int>(_items.size() - 1);
    _stack->addWidget(page);
    _barLayout->insertWidget(std::max(0, _barLayout->count() - 1), button);

    connect(button, &QPushButton::clicked, this, [this, index]() {
        if (index >= 0 && index < static_cast<int>(_items.size())) {
            toggleItem(_items[static_cast<std::size_t>(index)]);
        }
    });
    connect(button, &QPushButton::customContextMenuRequested, this, [this, index, button](const QPoint& pos) {
        if (index >= 0 && index < static_cast<int>(_items.size())) {
            showItemMenu(_items[static_cast<std::size_t>(index)], button->mapToGlobal(pos));
        }
    });
    connect(collapse, &QToolButton::clicked, this, [this]() {
        collapseCurrent();
    });
    connect(dock, &QDockWidget::visibilityChanged, this, [this, dock](bool visible) {
        Item* item = itemForDock(dock);
        if (!item || !item->detached || visible) {
            return;
        }
        attachItem(*item, false);
    });

    page->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(page, &QWidget::customContextMenuRequested, this, [this, index, page](const QPoint& pos) {
        if (index >= 0 && index < static_cast<int>(_items.size())) {
            showItemMenu(_items[static_cast<std::size_t>(index)], page->mapToGlobal(pos));
        }
    });

    updateButton(_items.back());
}

bool StatusDockPanelHost::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == _panelFrame && event->type() == QEvent::Resize && _currentIndex >= 0 &&
        _currentIndex < static_cast<int>(_items.size()) && !_positioningPanel) {
        Item& item = _items[static_cast<std::size_t>(_currentIndex)];
        item.panelSize = static_cast<QResizeEvent*>(event)->size();
        item.userSized = true;
        updateButton(item);
        positionPanelForItem(item);
    }

    if ((event->type() == QEvent::MouseButtonPress || event->type() == QEvent::MouseMove ||
         event->type() == QEvent::MouseButtonRelease) &&
        _panelFrame && _panelFrame->isVisible() && _currentIndex >= 0 &&
        _currentIndex < static_cast<int>(_items.size())) {
        auto* mouse = static_cast<QMouseEvent*>(event);
        const QPoint globalPos = mouse->globalPosition().toPoint();

        if (event->type() == QEvent::MouseButtonPress && mouse->button() == Qt::LeftButton) {
            const ResizeMode mode = resizeModeAtGlobalPoint(globalPos);
            if (mode != ResizeMode::None) {
                _resizingPanel = true;
                _resizeMode = mode;
                _resizeStartGlobal = globalPos;
                _resizeStartSize = _panelFrame->size();
                updateResizeCursor(globalPos);
                return true;
            }
        }

        if (event->type() == QEvent::MouseMove) {
            if (_resizingPanel) {
                Item& item = _items[static_cast<std::size_t>(_currentIndex)];
                const QPoint delta = globalPos - _resizeStartGlobal;
                QSize nextSize = _resizeStartSize;
                if (_resizeMode == ResizeMode::Width || _resizeMode == ResizeMode::Both) {
                    nextSize.setWidth(_resizeStartSize.width() + delta.x());
                }
                if (_resizeMode == ResizeMode::Height || _resizeMode == ResizeMode::Both) {
                    nextSize.setHeight(_resizeStartSize.height() - delta.y());
                }
                item.panelSize = nextSize;
                item.userSized = true;
                positionPanelForItem(item);
                updateButton(item);
                return true;
            }

            updateResizeCursor(globalPos);
        }

        if (event->type() == QEvent::MouseButtonRelease && _resizingPanel) {
            _resizingPanel = false;
            _resizeMode = ResizeMode::None;
            updateResizeCursor(globalPos);
            return true;
        }
    }

    if (event->type() == QEvent::MouseButtonPress && _currentIndex >= 0) {
        auto* mouse = static_cast<QMouseEvent*>(event);
        if (mouse->button() == Qt::LeftButton &&
            _currentIndex < static_cast<int>(_items.size()) &&
            !_items[static_cast<std::size_t>(_currentIndex)].pinned &&
            !globalPointInsidePanelOrBar(mouse->globalPosition().toPoint())) {
            collapseCurrent();
        }
    }

    return QWidget::eventFilter(watched, event);
}

StatusDockPanelHost::Item* StatusDockPanelHost::itemForDock(QDockWidget* dock)
{
    auto it = std::find_if(_items.begin(), _items.end(), [dock](const Item& item) {
        return item.dock == dock;
    });
    return it == _items.end() ? nullptr : &(*it);
}

int StatusDockPanelHost::itemIndex(const Item& item) const
{
    const auto* base = _items.data();
    const auto* ptr = &item;
    if (!base || ptr < base || ptr >= base + static_cast<std::ptrdiff_t>(_items.size())) {
        return -1;
    }
    return static_cast<int>(ptr - base);
}

void StatusDockPanelHost::toggleItem(Item& item)
{
    if (item.detached) {
        attachItem(item, true);
        return;
    }

    const int index = itemIndex(item);
    if (_currentIndex == index) {
        collapseCurrent();
    } else {
        expandItem(item);
    }
}

void StatusDockPanelHost::expandItem(Item& item)
{
    if (!item.page) {
        return;
    }

    const int index = _stack->indexOf(item.page);
    const int itemVectorIndex = itemIndex(item);
    if (index < 0 || itemVectorIndex < 0) {
        return;
    }

    _currentIndex = itemVectorIndex;
    _stack->setCurrentIndex(index);
    showPanelForItem(item);

    for (Item& candidate : _items) {
        updateButton(candidate);
    }
}

void StatusDockPanelHost::collapseCurrent()
{
    if (_currentIndex < 0) {
        return;
    }

    _currentIndex = -1;
    hidePanel();
    for (Item& item : _items) {
        updateButton(item);
    }
}

void StatusDockPanelHost::detachItem(Item& item)
{
    if (!item.dock || !item.content || !item.page || item.detached) {
        return;
    }

    const bool wasCurrent = item.page && _stack->currentWidget() == item.page;
    if (wasCurrent) {
        collapseCurrent();
    }

    item.page->layout()->removeWidget(item.content);
    item.dock->setWidget(item.content);
    item.dock->setFloating(true);
    item.dock->show();
    item.detached = true;
    item.pinned = false;
    updateButton(item);
}

void StatusDockPanelHost::attachItem(Item& item, bool expand)
{
    if (!item.dock || !item.content || !item.page || !item.detached) {
        return;
    }

    item.detached = false;
    item.dock->hide();
    item.dock->setWidget(nullptr);
    item.content->setParent(item.page);
    item.page->layout()->addWidget(item.content);
    item.content->show();
    item.page->show();
    updateButton(item);

    if (expand) {
        expandItem(item);
    }
}

void StatusDockPanelHost::showItemMenu(Item& item, const QPoint& globalPos)
{
    QMenu menu(this);
    QAction* pin = menu.addAction(item.pinned ? tr("Unpin") : tr("Pin open"));
    pin->setEnabled(!item.detached);
    QAction* detach = menu.addAction(item.detached ? tr("Attach to status bar") : tr("Detach"));

    QAction* selected = menu.exec(globalPos);
    if (selected == pin) {
        item.pinned = !item.pinned;
        if (item.pinned && _currentIndex != itemIndex(item)) {
            expandItem(item);
        }
        updateButton(item);
    } else if (selected == detach) {
        if (item.detached) {
            attachItem(item, true);
        } else {
            detachItem(item);
        }
    }
}

void StatusDockPanelHost::updateButton(Item& item)
{
    if (!item.button || !item.dock) {
        return;
    }

    const bool active = !item.detached && item.page && _stack->currentWidget() == item.page && _currentIndex >= 0;
    const QString prefix = item.detached ? QStringLiteral("↙") : (active ? QStringLiteral("▼") : QStringLiteral("▲"));
    const QString pinSuffix = item.pinned ? QStringLiteral(" •") : QString();
    item.button->setText(QStringLiteral("%1 %2%3").arg(prefix, item.dock->windowTitle(), pinSuffix));
    item.button->setProperty("active", active);
    item.button->setMinimumWidth(0);
    item.button->style()->unpolish(item.button);
    item.button->style()->polish(item.button);
}

void StatusDockPanelHost::showPanelForItem(Item& item)
{
    if (!_panelFrame) {
        return;
    }

    QWidget* top = window();
    if (top && _panelFrame->parentWidget() != top) {
        _panelFrame->setParent(top);
    }
    if (item.content) {
        item.content->show();
    }
    if (item.page) {
        item.page->show();
    }
    positionPanelForItem(item);
    _panelFrame->show();
    _panelFrame->raise();
    _panelFrame->activateWindow();
}

void StatusDockPanelHost::positionPanelForItem(Item& item)
{
    if (!_panelFrame || !item.button) {
        return;
    }

    QWidget* top = window();
    if (!top) {
        top = this;
    }

    const QRect buttonGlobalRect(item.button->mapToGlobal(QPoint(0, 0)), item.button->size());
    QSize size = item.panelSize;
    if (!size.isValid() || size.isEmpty()) {
        QSize contentSize;
        if (item.page) {
            contentSize = item.page->sizeHint();
        }
        if (item.content) {
            contentSize = contentSize.expandedTo(item.content->sizeHint());
        }
        size = contentSize.expandedTo(
            QSize(std::max(buttonGlobalRect.width(), item.button->sizeHint().width()),
                  expandedPanelHeight()));
        item.panelSize = size;
    }

    const QSize minSize(std::max(80, item.button->minimumSizeHint().width()), 120);
    const QSize maxSize(std::max(minSize.width(), top->width() - 16),
                        std::max(minSize.height(), top->height() - 48));
    size = size.expandedTo(minSize).boundedTo(maxSize);
    item.panelSize = size;

    QPoint topLeft = top->mapFromGlobal(QPoint(buttonGlobalRect.left(), buttonGlobalRect.top() - size.height()));
    topLeft.setX(std::clamp(topLeft.x(), 4, std::max(4, top->width() - size.width() - 4)));
    topLeft.setY(std::clamp(topLeft.y(), 4, std::max(4, top->height() - size.height() - 4)));

    _positioningPanel = true;
    _panelFrame->setGeometry(QRect(topLeft, size));
    _positioningPanel = false;
}

void StatusDockPanelHost::hidePanel()
{
    if (_panelFrame) {
        _panelFrame->hide();
    }
}

int StatusDockPanelHost::expandedPanelHeight() const
{
    const QWidget* top = window();
    const int available = top ? top->height() : height();
    return std::clamp(available / 2, 260, 520);
}

bool StatusDockPanelHost::globalPointInsidePanelOrBar(const QPoint& globalPos) const
{
    const QWidget* widget = QApplication::widgetAt(globalPos);
    if (!widget) {
        return false;
    }
    if (widget == this || isAncestorOf(widget)) {
        return true;
    }
    if (_panelFrame && (widget == _panelFrame || _panelFrame->isAncestorOf(widget))) {
        return true;
    }
    if (_barFrame && (widget == _barFrame || _barFrame->isAncestorOf(widget))) {
        return true;
    }
    return std::any_of(_items.begin(), _items.end(), [widget](const Item& item) {
        return item.button && (widget == item.button || item.button->isAncestorOf(widget));
    });
}

StatusDockPanelHost::ResizeMode StatusDockPanelHost::resizeModeAtGlobalPoint(const QPoint& globalPos) const
{
    if (!_panelFrame || !_panelFrame->isVisible()) {
        return ResizeMode::None;
    }

    const QPoint panelPos = _panelFrame->mapFromGlobal(globalPos);
    const QRect panelRect = _panelFrame->rect();
    if (!panelRect.contains(panelPos)) {
        return ResizeMode::None;
    }

    const bool onTop = panelPos.y() <= kResizeHitWidth;
    const bool onRight = panelPos.x() >= panelRect.width() - kResizeHitWidth;
    if (onTop && onRight) {
        return ResizeMode::Both;
    }
    if (onTop) {
        return ResizeMode::Height;
    }
    if (onRight) {
        return ResizeMode::Width;
    }
    return ResizeMode::None;
}

void StatusDockPanelHost::updateResizeCursor(const QPoint& globalPos)
{
    if (!_panelFrame) {
        return;
    }

    const ResizeMode mode = _resizingPanel ? _resizeMode : resizeModeAtGlobalPoint(globalPos);
    switch (mode) {
    case ResizeMode::Both:
        _panelFrame->setCursor(Qt::SizeBDiagCursor);
        break;
    case ResizeMode::Width:
        _panelFrame->setCursor(Qt::SizeHorCursor);
        break;
    case ResizeMode::Height:
        _panelFrame->setCursor(Qt::SizeVerCursor);
        break;
    case ResizeMode::None:
        _panelFrame->unsetCursor();
        break;
    }
}
