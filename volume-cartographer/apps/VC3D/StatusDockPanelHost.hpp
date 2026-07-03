#pragma once

#include <QPoint>
#include <QPointer>
#include <QSize>
#include <QWidget>

#include <vector>

class QDockWidget;
class QFrame;
class QHBoxLayout;
class QPushButton;
class QStackedWidget;
class QVBoxLayout;

class StatusDockPanelHost final : public QWidget
{
    Q_OBJECT

public:
    explicit StatusDockPanelHost(QWidget* centralWidget, QWidget* parent = nullptr);
    ~StatusDockPanelHost() override;

    QWidget* takeBarWidget();
    void addDock(QDockWidget* dock);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    struct Item {
        QPointer<QDockWidget> dock;
        QPointer<QWidget> content;
        QPointer<QWidget> page;
        QPointer<QPushButton> button;
        QSize panelSize;
        bool pinned = false;
        bool detached = false;
        bool userSized = false;
    };

    Item* itemForDock(QDockWidget* dock);
    int itemIndex(const Item& item) const;
    void toggleItem(Item& item);
    void expandItem(Item& item);
    void collapseCurrent();
    void detachItem(Item& item);
    void attachItem(Item& item, bool expand);
    void showItemMenu(Item& item, const QPoint& globalPos);
    void updateButton(Item& item);
    void showPanelForItem(Item& item);
    void positionPanelForItem(Item& item);
    void hidePanel();
    int expandedPanelHeight() const;
    bool globalPointInsidePanelOrBar(const QPoint& globalPos) const;
    enum class ResizeMode {
        None,
        Width,
        Height,
        Both
    };
    ResizeMode resizeModeAtGlobalPoint(const QPoint& globalPos) const;
    void updateResizeCursor(const QPoint& globalPos);

    QVBoxLayout* _layout{nullptr};
    QFrame* _panelFrame{nullptr};
    QStackedWidget* _stack{nullptr};
    QFrame* _barFrame{nullptr};
    QHBoxLayout* _barLayout{nullptr};
    std::vector<Item> _items;
    int _currentIndex{-1};
    bool _positioningPanel{false};
    bool _resizingPanel{false};
    ResizeMode _resizeMode{ResizeMode::None};
    QPoint _resizeStartGlobal;
    QSize _resizeStartSize;
};
