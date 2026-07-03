#include "StatusDockPanelHost.hpp"

#include "VCSettings.hpp"

#include <QApplication>
#include <QAction>
#include <QCursor>
#include <QDockWidget>
#include <QEvent>
#include <QFrame>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QMainWindow>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QResizeEvent>
#include <QSettings>
#include <QSignalBlocker>
#include <QStackedWidget>
#include <QStyle>
#include <QStyleHints>
#include <QToolButton>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cstddef>

namespace {
constexpr int kResizeHitWidth = 10;
constexpr int kDefaultPanelWidth = 360;
constexpr int kMinimumPanelWidth = 320;
constexpr int kButtonDragHoldMs = 400;
constexpr auto kPanelSizeSettingsPrefix = "statusDockPanels/sizes";
constexpr auto kPanelVisibleSettingsPrefix = "statusDockPanels/visible";
constexpr auto kPanelOrderSettingsKey = "statusDockPanels/order";

QIcon makePinIcon(const QPalette& palette, bool pinned)
{
    QPixmap pixmap(16, 16);
    pixmap.fill(Qt::transparent);

    const QColor color = pinned ? palette.highlight().color() : palette.mid().color();
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(color, 1.6, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.setBrush(pinned ? QBrush(color) : Qt::NoBrush);
    painter.translate(8.0, 8.0);
    painter.rotate(35.0);
    painter.drawRoundedRect(QRectF(-4.0, -6.0, 8.0, 5.0), 1.0, 1.0);
    painter.drawLine(QPointF(0.0, -1.0), QPointF(0.0, 5.0));
    painter.drawLine(QPointF(-3.0, 5.0), QPointF(3.0, 5.0));
    painter.drawLine(QPointF(0.0, 5.0), QPointF(0.0, 7.0));
    return QIcon(pixmap);
}
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
        "QFrame#statusDockPanelFrame[resizeHandleActive=\"true\"] {"
        " border-top: 5px solid palette(highlight); border-right: 5px solid palette(highlight); }"
        "QWidget[statusDockPanelPage=\"true\"] { background: palette(window); }"
        "QFrame[statusDockPanelHeader=\"true\"] { background: palette(window);"
        " border-bottom: 1px solid palette(mid); }"
        "QPushButton[statusDockPanelButton=\"true\"] { padding: 3px 8px; text-align: left; }"
        "QPushButton[statusDockPanelButton=\"true\"][active=\"true\"] { font-weight: 600; }"
        "QPushButton[statusDockPanelButton=\"true\"][dragArmed=\"true\"] {"
        " border: 1px solid palette(highlight); background: palette(alternate-base); }"));

    _dragArmTimer = new QTimer(this);
    _dragArmTimer->setSingleShot(true);
    _dragArmTimer->setInterval(kButtonDragHoldMs);
    connect(_dragArmTimer, &QTimer::timeout, this, [this]() {
        if (!_dragButton) {
            return;
        }
        _buttonDragArmed = true;
        setButtonDragArmed(_dragButton, true);
        _dragButton->setCursor(Qt::OpenHandCursor);
    });

    if (qApp) {
        qApp->installEventFilter(this);
    }
    _panelFrame->installEventFilter(this);
}

StatusDockPanelHost::~StatusDockPanelHost()
{
    setResizeFeedback(ResizeMode::None);
    resetButtonDragState(true);

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
    loadPanelSize(item);
    item.visibleInBar = loadItemVisibleInBar(item);

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

    auto* pin = new QToolButton(header);
    pin->setCheckable(true);
    pin->setAutoRaise(true);
    pin->setIconSize(QSize(16, 16));
    pin->setToolTip(tr("Pin open"));
    headerLayout->addWidget(pin);

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
    button->installEventFilter(this);

    item.page = page;
    item.button = button;
    item.pinButton = pin;
    _items.push_back(item);
    _stack->addWidget(page);
    _barLayout->insertWidget(std::max(0, _barLayout->count() - 1), button);
    button->setVisible(item.visibleInBar);

    if (QAction* action = dock->toggleViewAction()) {
        action->disconnect(dock);
        dock->disconnect(action);
        action->setCheckable(true);
        action->setChecked(item.visibleInBar);
        connect(action, &QAction::toggled, this, [this, dock](bool checked) {
            if (Item* actionItem = itemForDock(dock)) {
                setItemVisibleInBar(*actionItem, checked);
            }
        });
    }

    connect(button, &QPushButton::clicked, this, [this, dock]() {
        if (_draggingButton) {
            return;
        }
        if (Item* clickedItem = itemForDock(dock)) {
            toggleItem(*clickedItem);
        }
    });
    connect(button, &QPushButton::customContextMenuRequested, this, [this, dock, button](const QPoint& pos) {
        if (Item* menuItem = itemForDock(dock)) {
            showItemMenu(*menuItem, button->mapToGlobal(pos));
        }
    });
    connect(collapse, &QToolButton::clicked, this, [this]() {
        collapseCurrent();
    });
    connect(pin, &QToolButton::clicked, this, [this, dock](bool checked) {
        if (Item* pinItem = itemForDock(dock)) {
            pinItem->pinned = checked;
            updateButton(*pinItem);
        }
    });
    connect(dock, &QDockWidget::visibilityChanged, this, [this, dock](bool visible) {
        Item* item = itemForDock(dock);
        if (!item || !item->detached || visible) {
            return;
        }
        attachItem(*item, false);
    });

    page->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(page, &QWidget::customContextMenuRequested, this, [this, dock, page](const QPoint& pos) {
        if (Item* menuItem = itemForDock(dock)) {
            showItemMenu(*menuItem, page->mapToGlobal(pos));
        }
    });

    loadItemOrder();
    if (Item* addedItem = itemForDock(dock)) {
        updateButton(*addedItem);
        syncViewAction(*addedItem);
    }
}

bool StatusDockPanelHost::eventFilter(QObject* watched, QEvent* event)
{
    if (auto* button = qobject_cast<QPushButton*>(watched)) {
        if (itemIndexForButton(button) >= 0) {
            if (event->type() == QEvent::MouseButtonPress) {
                auto* mouse = static_cast<QMouseEvent*>(event);
                if (mouse->button() == Qt::LeftButton) {
                    _dragButton = button;
                    _dragStartGlobal = mouse->globalPosition().toPoint();
                    _buttonDragArmed = false;
                    _draggingButton = false;
                    setButtonDragArmed(button, false);
                    if (_dragArmTimer) {
                        _dragArmTimer->start();
                    }
                    const int buttonIndex = itemIndexForButton(button);
                    if (buttonIndex >= 0 && buttonIndex < static_cast<int>(_items.size())) {
                        toggleItem(_items[static_cast<std::size_t>(buttonIndex)]);
                    }
                    button->setDown(true);
                    return true;
                }
            } else if (event->type() == QEvent::MouseMove && _dragButton == button) {
                auto* mouse = static_cast<QMouseEvent*>(event);
                const QPoint globalPos = mouse->globalPosition().toPoint();
                const int dragDistance = qApp && qApp->styleHints()
                    ? qApp->styleHints()->startDragDistance()
                    : QApplication::startDragDistance();
                if (_buttonDragArmed && !_draggingButton &&
                    (globalPos - _dragStartGlobal).manhattanLength() >= dragDistance) {
                    _draggingButton = true;
                    if (_currentIndex >= 0 &&
                        _currentIndex < static_cast<int>(_items.size())) {
                        Item& current = _items[static_cast<std::size_t>(_currentIndex)];
                        if (!current.pinned) {
                            collapseCurrent();
                        }
                    }
                    button->setDown(true);
                    button->setCursor(Qt::ClosedHandCursor);
                }
                if (_draggingButton) {
                    const int from = itemIndexForButton(button);
                    const int to = reorderDropIndexAtGlobalPoint(globalPos);
                    moveItem(from, to);
                    return true;
                }
            } else if (event->type() == QEvent::MouseButtonRelease && _dragButton == button) {
                const bool wasDragInteraction = _draggingButton;
                if (_draggingButton) {
                    saveItemOrder();
                }
                if (wasDragInteraction) {
                    resetButtonDragState(true);
                    return true;
                }
                resetButtonDragState(false);
                button->setDown(false);
                return true;
            }
        }
    }

    if (_dragButton && watched != _dragButton &&
        (event->type() == QEvent::MouseButtonRelease || event->type() == QEvent::MouseButtonPress)) {
        auto* mouse = static_cast<QMouseEvent*>(event);
        if (mouse->button() == Qt::LeftButton) {
            if (_draggingButton) {
                saveItemOrder();
            }
            resetButtonDragState(_buttonDragArmed || _draggingButton);
            if (event->type() == QEvent::MouseButtonRelease) {
                return true;
            }
        }
    }

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
            if (_currentIndex >= 0 && _currentIndex < static_cast<int>(_items.size())) {
                savePanelSize(_items[static_cast<std::size_t>(_currentIndex)]);
            }
            _resizingPanel = false;
            _resizeMode = ResizeMode::None;
            updateResizeCursor(globalPos);
            return true;
        }
    }

    if (event->type() == QEvent::MouseButtonPress && _currentIndex >= 0) {
        auto* mouse = static_cast<QMouseEvent*>(event);
        if ((mouse->button() == Qt::LeftButton || mouse->button() == Qt::RightButton) &&
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
    if (!item.visibleInBar) {
        return;
    }

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
    if (!item.visibleInBar || !item.page) {
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
    if (!item.visibleInBar || !item.dock || !item.content || !item.page || item.detached) {
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

void StatusDockPanelHost::setItemVisibleInBar(Item& item, bool visible)
{
    if (item.visibleInBar == visible) {
        syncViewAction(item);
        return;
    }

    if (!visible) {
        if (item.detached) {
            attachItem(item, false);
        }
        if (_currentIndex == itemIndex(item)) {
            collapseCurrent();
        }
        item.pinned = false;
    }

    item.visibleInBar = visible;
    saveItemVisibleInBar(item);
    if (item.button) {
        item.button->setVisible(visible);
    }
    if (item.dock && !item.detached) {
        item.dock->hide();
    }
    updateButton(item);
    syncViewAction(item);
}

void StatusDockPanelHost::syncViewAction(Item& item)
{
    if (!item.dock) {
        return;
    }

    if (QAction* action = item.dock->toggleViewAction()) {
        const QSignalBlocker blocker(action);
        action->setCheckable(true);
        action->setChecked(item.visibleInBar);
    }
}

void StatusDockPanelHost::showItemMenu(Item& item, const QPoint& globalPos)
{
    QMenu menu(this);
    QAction* pin = menu.addAction(item.pinned ? tr("Unpin") : tr("Pin open"));
    pin->setEnabled(!item.detached);
    QAction* detach = menu.addAction(item.detached ? tr("Attach to status bar") : tr("Detach"));
    QAction* close = menu.addAction(tr("Close"));

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
    } else if (selected == close) {
        setItemVisibleInBar(item, false);
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

    if (item.pinButton) {
        item.pinButton->setChecked(item.pinned);
        item.pinButton->setIcon(makePinIcon(item.pinButton->palette(), item.pinned));
        item.pinButton->setToolTip(item.pinned ? tr("Unpin") : tr("Pin open"));
    }
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
            QSize(std::max({kDefaultPanelWidth, buttonGlobalRect.width(), item.button->sizeHint().width()}),
                  expandedPanelHeight()));
        item.panelSize = size;
    }

    const QSize minSize(std::max(kMinimumPanelWidth, item.button->minimumSizeHint().width()), 120);
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
    setResizeFeedback(ResizeMode::None);
}

int StatusDockPanelHost::expandedPanelHeight() const
{
    const QWidget* top = window();
    const int available = top ? top->height() : height();
    return std::clamp(available / 2, 260, 520);
}

bool StatusDockPanelHost::globalPointInsidePanelOrBar(const QPoint& globalPos) const
{
    if (QWidget* popup = QApplication::activePopupWidget()) {
        const QRect popupRect(popup->mapToGlobal(QPoint(0, 0)), popup->size());
        if (popup->isVisible() && popupRect.contains(globalPos)) {
            return true;
        }
    }

    const QWidget* widget = QApplication::widgetAt(globalPos);
    if (!widget) {
        return false;
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

QString StatusDockPanelHost::settingsKeyForItem(const Item& item) const
{
    return QStringLiteral("%1/%2").arg(QString::fromLatin1(kPanelSizeSettingsPrefix), itemSettingsId(item));
}

QString StatusDockPanelHost::visibilitySettingsKeyForItem(const Item& item) const
{
    return QStringLiteral("%1/%2").arg(QString::fromLatin1(kPanelVisibleSettingsPrefix), itemSettingsId(item));
}

void StatusDockPanelHost::loadPanelSize(Item& item)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QSize size = settings.value(settingsKeyForItem(item)).toSize();
    if (size.isValid() && !size.isEmpty()) {
        item.panelSize = size;
        item.userSized = true;
    }
}

void StatusDockPanelHost::savePanelSize(const Item& item) const
{
    if (!item.userSized || !item.panelSize.isValid() || item.panelSize.isEmpty()) {
        return;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(settingsKeyForItem(item), item.panelSize);
}

bool StatusDockPanelHost::loadItemVisibleInBar(const Item& item) const
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    return settings.value(visibilitySettingsKeyForItem(item), true).toBool();
}

void StatusDockPanelHost::saveItemVisibleInBar(const Item& item) const
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(visibilitySettingsKeyForItem(item), item.visibleInBar);
}

QString StatusDockPanelHost::itemSettingsId(const Item& item) const
{
    QString id;
    if (item.dock) {
        id = item.dock->objectName();
        if (id.isEmpty()) {
            id = item.dock->windowTitle();
        }
    }
    if (id.isEmpty()) {
        const int index = itemIndex(item);
        id = index >= 0 ? QStringLiteral("panel_%1").arg(index) : QStringLiteral("unknown");
    }
    id.replace(QLatin1Char('/'), QLatin1Char('_'));
    id.replace(QLatin1Char('\\'), QLatin1Char('_'));
    return id;
}

void StatusDockPanelHost::loadItemOrder()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QStringList order = settings.value(QString::fromLatin1(kPanelOrderSettingsKey)).toStringList();
    if (order.isEmpty() || _items.size() < 2) {
        rebuildBarLayout();
        return;
    }

    QWidget* currentPage = nullptr;
    if (_currentIndex >= 0 && _currentIndex < static_cast<int>(_items.size())) {
        currentPage = _items[static_cast<std::size_t>(_currentIndex)].page;
    }

    std::stable_sort(_items.begin(), _items.end(), [this, &order](const Item& lhs, const Item& rhs) {
        const int lhsIndex = order.indexOf(itemSettingsId(lhs));
        const int rhsIndex = order.indexOf(itemSettingsId(rhs));
        const bool lhsKnown = lhsIndex >= 0;
        const bool rhsKnown = rhsIndex >= 0;
        if (lhsKnown != rhsKnown) {
            return lhsKnown;
        }
        if (!lhsKnown) {
            return false;
        }
        return lhsIndex < rhsIndex;
    });

    _currentIndex = -1;
    if (currentPage) {
        for (int i = 0; i < static_cast<int>(_items.size()); ++i) {
            if (_items[static_cast<std::size_t>(i)].page == currentPage) {
                _currentIndex = i;
                break;
            }
        }
    }
    rebuildBarLayout();
}

void StatusDockPanelHost::saveItemOrder() const
{
    QStringList order;
    order.reserve(static_cast<int>(_items.size()));
    for (const Item& item : _items) {
        order.push_back(itemSettingsId(item));
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(QString::fromLatin1(kPanelOrderSettingsKey), order);
}

void StatusDockPanelHost::rebuildBarLayout()
{
    if (!_barLayout) {
        return;
    }

    for (const Item& item : _items) {
        if (item.button) {
            _barLayout->removeWidget(item.button);
        }
    }
    for (const Item& item : _items) {
        if (item.button) {
            _barLayout->insertWidget(std::max(0, _barLayout->count() - 1), item.button);
        }
    }
}

int StatusDockPanelHost::itemIndexForButton(const QPushButton* button) const
{
    if (!button) {
        return -1;
    }
    for (int i = 0; i < static_cast<int>(_items.size()); ++i) {
        if (_items[static_cast<std::size_t>(i)].button == button) {
            return i;
        }
    }
    return -1;
}

int StatusDockPanelHost::reorderDropIndexAtGlobalPoint(const QPoint& globalPos) const
{
    if (_items.empty()) {
        return -1;
    }

    for (int i = 0; i < static_cast<int>(_items.size()); ++i) {
        const Item& item = _items[static_cast<std::size_t>(i)];
        if (!item.button || !item.button->isVisible()) {
            continue;
        }
        const QRect rect(item.button->mapToGlobal(QPoint(0, 0)), item.button->size());
        if (globalPos.x() < rect.center().x()) {
            return i;
        }
    }

    return static_cast<int>(_items.size()) - 1;
}

void StatusDockPanelHost::moveItem(int from, int to)
{
    if (from < 0 || to < 0 ||
        from >= static_cast<int>(_items.size()) ||
        to >= static_cast<int>(_items.size()) ||
        from == to) {
        return;
    }

    QWidget* currentPage = nullptr;
    if (_currentIndex >= 0 && _currentIndex < static_cast<int>(_items.size())) {
        currentPage = _items[static_cast<std::size_t>(_currentIndex)].page;
    }

    Item item = std::move(_items[static_cast<std::size_t>(from)]);
    _items.erase(_items.begin() + from);
    _items.insert(_items.begin() + to, std::move(item));

    _currentIndex = -1;
    if (currentPage) {
        for (int i = 0; i < static_cast<int>(_items.size()); ++i) {
            if (_items[static_cast<std::size_t>(i)].page == currentPage) {
                _currentIndex = i;
                break;
            }
        }
    }

    rebuildBarLayout();
    if (_currentIndex >= 0 && _currentIndex < static_cast<int>(_items.size())) {
        positionPanelForItem(_items[static_cast<std::size_t>(_currentIndex)]);
    }
}

void StatusDockPanelHost::setButtonDragArmed(QPushButton* button, bool armed)
{
    if (!button) {
        return;
    }
    button->setProperty("dragArmed", armed);
    button->style()->unpolish(button);
    button->style()->polish(button);
    button->update();
}

void StatusDockPanelHost::resetButtonDragState(bool resetPressedVisual)
{
    if (_dragArmTimer) {
        _dragArmTimer->stop();
    }
    if (_dragButton) {
        setButtonDragArmed(_dragButton, false);
        _dragButton->unsetCursor();
        if (resetPressedVisual) {
            _dragButton->setDown(false);
        }
    }
    _dragButton.clear();
    _buttonDragArmed = false;
    _draggingButton = false;
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
    setResizeFeedback(mode);
}

void StatusDockPanelHost::setResizeFeedback(ResizeMode mode)
{
    if (!_panelFrame) {
        return;
    }
    if (_resizeFeedbackMode == mode &&
        _resizeCursorOverridden == (mode != ResizeMode::None)) {
        return;
    }

    const bool shouldOverrideCursor = mode != ResizeMode::None;
    if (_resizeCursorOverridden && (!shouldOverrideCursor || _resizeFeedbackMode != mode)) {
        QApplication::restoreOverrideCursor();
        _resizeCursorOverridden = false;
    }

    switch (mode) {
    case ResizeMode::Both:
        QApplication::setOverrideCursor(QCursor(Qt::SizeBDiagCursor));
        _resizeCursorOverridden = true;
        break;
    case ResizeMode::Width:
        QApplication::setOverrideCursor(QCursor(Qt::SizeHorCursor));
        _resizeCursorOverridden = true;
        break;
    case ResizeMode::Height:
        QApplication::setOverrideCursor(QCursor(Qt::SizeVerCursor));
        _resizeCursorOverridden = true;
        break;
    case ResizeMode::None:
        break;
    }

    _resizeFeedbackMode = mode;
    _panelFrame->setProperty("resizeHandleActive", shouldOverrideCursor);
    _panelFrame->style()->unpolish(_panelFrame);
    _panelFrame->style()->polish(_panelFrame);
    _panelFrame->update();
}
