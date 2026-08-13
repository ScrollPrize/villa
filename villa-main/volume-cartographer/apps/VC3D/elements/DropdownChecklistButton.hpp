#pragma once

#include <QList>
#include <QString>
#include <QToolButton>

class QCheckBox;

class DropdownChecklistButton : public QToolButton
{
    Q_OBJECT

public:
    explicit DropdownChecklistButton(QWidget* parent = nullptr);

    QCheckBox* addOption(const QString& text,
                         const QString& objectName = QString(),
                         bool checked = false);
    void addSeparator();
    void clearOptions();
    QList<QCheckBox*> options() const { return _options; }
    int checkedCount() const;

signals:
    void optionToggled(QCheckBox* option, bool checked);

private:
    QList<QCheckBox*> _options;
};
