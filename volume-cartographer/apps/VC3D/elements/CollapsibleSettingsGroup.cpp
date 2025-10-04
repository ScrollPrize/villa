#include "CollapsibleSettingsGroup.hpp"

#include <algorithm>

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>

CollapsibleSettingsGroup::CollapsibleSettingsGroup(const QString& title, QWidget* parent)
    : QWidget(parent)
    , _toggleButton(new QToolButton(this))
    , _contentWidget(new QFrame(this))
    , _contentLayout(nullptr)
    , _gridLayout(nullptr)
    , _preferredColumns(1)
    , _preferredRows(0)
    , _expanded(true)
{
    auto* outerLayout = new QVBoxLayout(this);
    outerLayout->setContentsMargins(0, 0, 0, 0);
    outerLayout->setSpacing(0);

    _toggleButton->setText(title);
    _toggleButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    _toggleButton->setArrowType(Qt::DownArrow);
    _toggleButton->setCheckable(true);
    _toggleButton->setChecked(true);
    _toggleButton->setAutoRaise(true);
    _toggleButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    outerLayout->addWidget(_toggleButton);

    _contentLayout = new QVBoxLayout(_contentWidget);
    _contentLayout->setContentsMargins(12, 8, 12, 12);
    _contentLayout->setSpacing(8);

    _contentWidget->setFrameShape(QFrame::StyledPanel);
    _contentWidget->setFrameShadow(QFrame::Raised);
    _contentWidget->setVisible(true);

    outerLayout->addWidget(_contentWidget);

    ensureGridLayout();

    connect(_toggleButton, &QToolButton::toggled, this, [this](bool checked) {
        setExpanded(checked);
    });

    updateIndicator();
}

void CollapsibleSettingsGroup::setExpanded(bool expanded)
{
    if (_expanded == expanded) {
        return;
    }

    _expanded = expanded;
    _contentWidget->setVisible(expanded);
    _toggleButton->setChecked(expanded);
    updateIndicator();
    emit toggled(expanded);
}

void CollapsibleSettingsGroup::setColumns(int columns)
{
    const int normalized = std::max(1, columns);
    if (_preferredColumns == normalized && _preferredRows == 0) {
        return;
    }
    _preferredColumns = normalized;
    _preferredRows = 0;
    rebuildGrid();
}

void CollapsibleSettingsGroup::setRows(int rows)
{
    const int normalized = std::max(1, rows);
    if (_preferredRows == normalized && _preferredColumns == 0) {
        return;
    }
    _preferredRows = normalized;
    _preferredColumns = 0;
    rebuildGrid();
}

void CollapsibleSettingsGroup::setGrid(int rows, int columns)
{
    const int normalizedRows = std::max(0, rows);
    const int normalizedColumns = std::max(0, columns);

    int newRows = normalizedRows;
    int newColumns = normalizedColumns;
    if (newRows == 0 && newColumns == 0) {
        newColumns = 1;
    }

    if (_preferredRows == newRows && _preferredColumns == newColumns) {
        return;
    }

    _preferredRows = newRows;
    _preferredColumns = newColumns;
    rebuildGrid();
}

QWidget* CollapsibleSettingsGroup::addLabeledWidget(const QString& labelText,
                                                    QWidget* widget,
                                                    const QString& tooltip)
{
    if (!widget) {
        return nullptr;
    }

    QLabel* label = nullptr;
    const QString trimmedLabel = labelText.trimmed();
    if (!trimmedLabel.isEmpty()) {
        label = new QLabel(trimmedLabel, _contentWidget);
        label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    }

    if (!tooltip.isEmpty()) {
        if (label) {
            label->setToolTip(tooltip);
        }
        widget->setToolTip(tooltip);
    }

    _entries.push_back({label, widget});
    rebuildGrid();
    return widget;
}

QWidget* CollapsibleSettingsGroup::addRow(const QString& label,
                                          const std::function<void(QHBoxLayout*)>& builder,
                                          const QString& tooltip)
{
    if (!builder) {
        return nullptr;
    }

    auto* container = new QWidget(_contentWidget);
    auto* rowLayout = new QHBoxLayout(container);
    rowLayout->setContentsMargins(0, 0, 0, 0);
    rowLayout->setSpacing(8);
    builder(rowLayout);
    return addLabeledWidget(label, container, tooltip);
}

QSpinBox* CollapsibleSettingsGroup::addSpinBox(const QString& label,
                                               int minimum,
                                               int maximum,
                                               int step,
                                               const QString& tooltip)
{
    auto* spin = new QSpinBox(_contentWidget);
    spin->setRange(minimum, maximum);
    spin->setSingleStep(step);
    addLabeledWidget(label, spin, tooltip);
    return spin;
}

QDoubleSpinBox* CollapsibleSettingsGroup::addDoubleSpinBox(const QString& label,
                                                           double minimum,
                                                           double maximum,
                                                           double step,
                                                           int decimals,
                                                           const QString& tooltip)
{
    auto* spin = new QDoubleSpinBox(_contentWidget);
    spin->setDecimals(decimals);
    spin->setRange(minimum, maximum);
    spin->setSingleStep(step);
    addLabeledWidget(label, spin, tooltip);
    return spin;
}

QCheckBox* CollapsibleSettingsGroup::addCheckBox(const QString& text, const QString& tooltip)
{
    auto* checkbox = new QCheckBox(text, _contentWidget);
    addLabeledWidget(QString(), checkbox, tooltip);
    return checkbox;
}

void CollapsibleSettingsGroup::addFullWidthWidget(QWidget* widget, const QString& tooltip)
{
    addLabeledWidget(QString(), widget, tooltip);
}

void CollapsibleSettingsGroup::updateIndicator()
{
    _toggleButton->setArrowType(_expanded ? Qt::DownArrow : Qt::RightArrow);
}

void CollapsibleSettingsGroup::ensureGridLayout()
{
    if (_gridLayout) {
        return;
    }

    _gridLayout = new QGridLayout();
    _gridLayout->setContentsMargins(0, 0, 0, 0);
    _gridLayout->setHorizontalSpacing(12);
    _gridLayout->setVerticalSpacing(8);
    _contentLayout->addLayout(_gridLayout);
}

void CollapsibleSettingsGroup::rebuildGrid()
{
    ensureGridLayout();

    while (_gridLayout->count() > 0) {
        delete _gridLayout->takeAt(0);
    }

    const int totalEntries = static_cast<int>(_entries.size());
    if (totalEntries == 0) {
        return;
    }

    const int entriesPerRow = std::max(1, slotsPerRow());

    for (int slot = 0; slot < entriesPerRow; ++slot) {
        _gridLayout->setColumnStretch(slot * 2, 0);
        _gridLayout->setColumnStretch(slot * 2 + 1, 1);
    }

    int row = 0;
    int columnSlot = 0;
    for (const auto& entry : _entries) {
        if (columnSlot >= entriesPerRow) {
            columnSlot = 0;
            ++row;
        }

        const int baseColumn = columnSlot * 2;
        if (entry.label) {
            _gridLayout->addWidget(entry.label, row, baseColumn);
        }

        if (entry.widget) {
            const int widgetColumn = entry.label ? baseColumn + 1 : baseColumn;
            const int span = entry.label ? 1 : 2;
            _gridLayout->addWidget(entry.widget, row, widgetColumn, 1, span);
        }

        ++columnSlot;
    }
}

int CollapsibleSettingsGroup::slotsPerRow() const
{
    if (_preferredColumns > 0) {
        return _preferredColumns;
    }

    if (_preferredRows > 0 && !_entries.empty()) {
        const int totalEntries = static_cast<int>(_entries.size());
        return std::max(1, (totalEntries + _preferredRows - 1) / _preferredRows);
    }

    return 1;
}
