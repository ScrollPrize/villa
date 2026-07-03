#pragma once

#include <QPointer>
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
        bool pinned = false;
        bool detached = false;
    };

    Item* itemForDock(QDockWidget* dock);
    void toggleItem(Item& item);
    void expandItem(Item& item);
    void collapseCurrent();
    void detachItem(Item& item);
    void attachItem(Item& item, bool expand);
    void showItemMenu(Item& item, const QPoint& globalPos);
    void updateButton(Item& item);
    void animatePanel(bool expanded);
    int expandedPanelHeight() const;
    bool globalPointInsideHost(const QPoint& globalPos) const;

    QVBoxLayout* _layout{nullptr};
    QFrame* _panelFrame{nullptr};
    QStackedWidget* _stack{nullptr};
    QFrame* _barFrame{nullptr};
    QHBoxLayout* _barLayout{nullptr};
    std::vector<Item> _items;
    int _currentIndex{-1};
};
