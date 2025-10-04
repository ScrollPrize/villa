#include "DropdownChecklistButton.hpp"

#include <QCheckBox>
#include <QMenu>
#include <QWidgetAction>

DropdownChecklistButton::DropdownChecklistButton(QWidget* parent)
    : QToolButton(parent)
{
    setToolButtonStyle(Qt::ToolButtonTextOnly);
    setPopupMode(QToolButton::InstantPopup);
    auto* dropdownMenu = new QMenu(this);
    setMenu(dropdownMenu);
}

QCheckBox* DropdownChecklistButton::addOption(const QString& text,
                                              const QString& objectName,
                                              bool checked)
{
    auto* dropdownMenu = menu();
    if (!dropdownMenu) {
        dropdownMenu = new QMenu(this);
        setMenu(dropdownMenu);
    }

    auto* checkBox = new QCheckBox(text, dropdownMenu);
    if (!objectName.isEmpty()) {
        checkBox->setObjectName(objectName);
    }
    checkBox->setChecked(checked);

    auto* action = new QWidgetAction(dropdownMenu);
    action->setDefaultWidget(checkBox);
    dropdownMenu->addAction(action);

    connect(checkBox, &QCheckBox::toggled, this, [this, checkBox](bool state) {
        emit optionToggled(checkBox, state);
    });

    _options.append(checkBox);
    return checkBox;
}

void DropdownChecklistButton::addSeparator()
{
    if (auto* dropdownMenu = menu()) {
        dropdownMenu->addSeparator();
    }
}

void DropdownChecklistButton::clearOptions()
{
    if (auto* dropdownMenu = menu()) {
        dropdownMenu->clear();
    }
    _options.clear();
}

int DropdownChecklistButton::checkedCount() const
{
    int count = 0;
    for (auto* option : _options) {
        if (option && option->isChecked()) {
            ++count;
        }
    }
    return count;
}
