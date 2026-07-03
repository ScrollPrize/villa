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
#include <QPropertyAnimation>
#include <QPushButton>
#include <QStackedWidget>
#include <QStyle>
#include <QToolButton>
#include <QVBoxLayout>

#include <algorithm>

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
    _panelFrame->setMaximumHeight(0);
    _panelFrame->hide();

    auto* panelLayout = new QVBoxLayout(_panelFrame);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(0);

    _stack = new QStackedWidget(_panelFrame);
    panelLayout->addWidget(_stack);
    _layout->addWidget(_panelFrame, 0);

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
        "QFrame#statusDockPanelFrame { background: palette(base); border-top: 1px solid palette(mid); }"
        "QPushButton[statusDockPanelButton=\"true\"] { padding: 3px 8px; text-align: left; }"
        "QPushButton[statusDockPanelButton=\"true\"][active=\"true\"] { font-weight: 600; }"));

    if (qApp) {
        qApp->installEventFilter(this);
    }
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
    auto* pageLayout = new QVBoxLayout(page);
    pageLayout->setContentsMargins(0, 0, 0, 0);
    pageLayout->setSpacing(0);

    auto* header = new QFrame(page);
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
    if (event->type() == QEvent::MouseButtonPress && _currentIndex >= 0) {
        auto* mouse = static_cast<QMouseEvent*>(event);
        if (mouse->button() == Qt::LeftButton &&
            _currentIndex < static_cast<int>(_items.size()) &&
            !_items[static_cast<std::size_t>(_currentIndex)].pinned &&
            !globalPointInsideHost(mouse->globalPosition().toPoint())) {
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

void StatusDockPanelHost::toggleItem(Item& item)
{
    if (item.detached) {
        attachItem(item, true);
        return;
    }

    const int index = static_cast<int>(&item - _items.data());
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
    if (index < 0) {
        return;
    }

    _currentIndex = index;
    _stack->setCurrentIndex(index);
    animatePanel(true);

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
    animatePanel(false);
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
        if (item.pinned && _currentIndex != _stack->indexOf(item.page)) {
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
    item.button->style()->unpolish(item.button);
    item.button->style()->polish(item.button);
}

void StatusDockPanelHost::animatePanel(bool expanded)
{
    if (!_panelFrame) {
        return;
    }

    _panelFrame->show();
    auto* animation = new QPropertyAnimation(_panelFrame, "maximumHeight", _panelFrame);
    animation->setDuration(140);
    animation->setStartValue(_panelFrame->maximumHeight());
    animation->setEndValue(expanded ? expandedPanelHeight() : 0);
    animation->setEasingCurve(QEasingCurve::OutCubic);
    connect(animation, &QPropertyAnimation::finished, _panelFrame, [this, expanded]() {
        if (!expanded && _panelFrame) {
            _panelFrame->hide();
        }
    });
    animation->start(QAbstractAnimation::DeleteWhenStopped);
}

int StatusDockPanelHost::expandedPanelHeight() const
{
    const QWidget* top = window();
    const int available = top ? top->height() : height();
    return std::clamp(available / 2, 260, 520);
}

bool StatusDockPanelHost::globalPointInsideHost(const QPoint& globalPos) const
{
    const QWidget* widget = QApplication::widgetAt(globalPos);
    return widget && (widget == this || isAncestorOf(widget));
}
