#pragma once

#include <QFrame>
#include <QWidget>

#include <functional>
#include <vector>

class QDoubleSpinBox;
class QGridLayout;
class QHBoxLayout;
class QLabel;
class QToolButton;
class QVBoxLayout;

class CollapsibleSettingsGroup : public QWidget
{
    Q_OBJECT

public:
    explicit CollapsibleSettingsGroup(const QString& title, QWidget* parent = nullptr);

    void setExpanded(bool expanded);
    bool isExpanded() const { return _expanded; }

    QWidget* addLabeledWidget(const QString& label,
                              QWidget* widget,
                              const QString& tooltip = {});
    QWidget* addRow(const QString& label,
                    const std::function<void(QHBoxLayout*)>& builder,
                    const QString& tooltip = {});
    QDoubleSpinBox* addDoubleSpinBox(const QString& label,
                                     double minimum,
                                     double maximum,
                                     double step = 0.1,
                                     int decimals = 2,
                                     const QString& tooltip = {});

    void addFullWidthWidget(QWidget* widget, const QString& tooltip = {});

    QWidget* contentWidget() const { return _contentWidget; }
    QVBoxLayout* contentLayout() const { return _contentLayout; }

signals:
    void toggled(bool expanded);

private:
    void updateIndicator();
    void ensureGridLayout();
    void rebuildGrid();

    QToolButton* _toggleButton;
    QFrame* _contentWidget;
    QVBoxLayout* _contentLayout;
    QGridLayout* _gridLayout;
    struct Entry {
        QLabel* label;
        QWidget* widget;
    };
    std::vector<Entry> _entries;
    bool _expanded;
};
