#include "SettingsFilter.hpp"

#include <QAbstractButton>
#include <QAbstractSpinBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QSpinBox>
#include <QString>
#include <QStringList>
#include <QTextDocumentFragment>

#include <map>
#include <utility>
#include <vector>

namespace vc3d::settings_filter {

namespace {

// Marks widgets the filter hid, so clearing it never shows anything else.
constexpr const char* kHiddenByFilter = "vc3dHiddenBySettingsFilter";

void collectWidgets(QLayoutItem* item, QList<QWidget*>& widgets)
{
    if (!item) {
        return;
    }
    if (QWidget* widget = item->widget()) {
        widgets.append(widget);
        return;
    }
    if (QLayout* layout = item->layout()) {
        for (int i = 0; i < layout->count(); ++i) {
            collectWidgets(layout->itemAt(i), widgets);
        }
    }
}

std::vector<QList<QWidget*>> layoutRows(QLayout* layout)
{
    std::vector<QList<QWidget*>> rows;
    if (auto* grid = qobject_cast<QGridLayout*>(layout)) {
        std::map<int, QList<QWidget*>> byRow;
        for (int i = 0; i < grid->count(); ++i) {
            int row = 0;
            int column = 0;
            int rowSpan = 0;
            int columnSpan = 0;
            grid->getItemPosition(i, &row, &column, &rowSpan, &columnSpan);
            collectWidgets(grid->itemAt(i), byRow[row]);
        }
        for (const auto& [row, widgets] : byRow) {
            if (!widgets.isEmpty()) {
                rows.push_back(widgets);
            }
        }
    } else if (auto* form = qobject_cast<QFormLayout*>(layout)) {
        for (int row = 0; row < form->rowCount(); ++row) {
            QList<QWidget*> widgets;
            collectWidgets(form->itemAt(row, QFormLayout::LabelRole), widgets);
            collectWidgets(form->itemAt(row, QFormLayout::FieldRole), widgets);
            collectWidgets(form->itemAt(row, QFormLayout::SpanningRole), widgets);
            if (!widgets.isEmpty()) {
                rows.push_back(widgets);
            }
        }
    } else if (layout) {
        for (int i = 0; i < layout->count(); ++i) {
            QList<QWidget*> widgets;
            collectWidgets(layout->itemAt(i), widgets);
            if (!widgets.isEmpty()) {
                rows.push_back(widgets);
            }
        }
    }
    return rows;
}

QString plainText(const QString& text)
{
    return Qt::mightBeRichText(text) ? QTextDocumentFragment::fromHtml(text).toPlainText() : text;
}

void appendSearchText(QWidget* widget, QStringList& haystack)
{
    QList<QWidget*> widgets{widget};
    widgets += widget->findChildren<QWidget*>();
    for (QWidget* current : widgets) {
        if (auto* label = qobject_cast<QLabel*>(current)) {
            haystack.append(plainText(label->text()));
        } else if (auto* button = qobject_cast<QAbstractButton*>(current)) {
            haystack.append(button->text());
        } else if (auto* edit = qobject_cast<QLineEdit*>(current)) {
            haystack.append(edit->placeholderText());
        } else if (auto* combo = qobject_cast<QComboBox*>(current)) {
            for (int i = 0; i < combo->count(); ++i) {
                haystack.append(combo->itemText(i));
            }
        } else if (auto* group = qobject_cast<QGroupBox*>(current)) {
            haystack.append(group->title());
        }
        if (auto* spin = qobject_cast<QSpinBox*>(current)) {
            haystack.append(spin->suffix());
        } else if (auto* doubleSpin = qobject_cast<QDoubleSpinBox*>(current)) {
            haystack.append(doubleSpin->suffix());
        }
        if (auto* spin = qobject_cast<QAbstractSpinBox*>(current)) {
            haystack.append(spin->specialValueText());
        }
        haystack.append(plainText(current->toolTip()));
    }
}

bool matchesAll(const QStringList& words, const QString& text)
{
    for (const QString& word : words) {
        if (!text.contains(word, Qt::CaseInsensitive)) {
            return false;
        }
    }
    return true;
}

void setFilteredVisible(QWidget* widget, bool visible)
{
    if (!visible) {
        if (!widget->isHidden()) {
            widget->setProperty(kHiddenByFilter, true);
            widget->hide();
        }
    } else if (widget->property(kHiddenByFilter).toBool()) {
        widget->setProperty(kHiddenByFilter, QVariant());
        widget->show();
    }
}

bool isOwnedByInnerGroup(QGroupBox* group, QWidget* root)
{
    for (QWidget* parent = group->parentWidget(); parent && parent != root; parent = parent->parentWidget()) {
        if (qobject_cast<QGroupBox*>(parent)) {
            return true;
        }
    }
    return false;
}

void focusFirstMatch(QWidget* root, QWidget* field)
{
    for (QGroupBox* group : root->findChildren<QGroupBox*>()) {
        if (!group->isVisibleTo(root)) {
            continue;
        }
        for (const auto& row : layoutRows(group->layout())) {
            for (QWidget* widget : row) {
                if (widget != field && widget->isVisibleTo(root) && widget->isEnabled() &&
                    (widget->focusPolicy() & Qt::TabFocus)) {
                    widget->setFocus(Qt::TabFocusReason);
                    return;
                }
            }
        }
    }
}

class FilterKeys : public QObject
{
public:
    FilterKeys(QLineEdit* field, std::function<void()> onEnter)
        : QObject(field)
        , _field(field)
        , _onEnter(std::move(onEnter))
    {
    }

protected:
    bool eventFilter(QObject* watched, QEvent* event) override
    {
        if (watched != _field || event->type() != QEvent::KeyPress || _field->text().isEmpty()) {
            return false;
        }
        const int key = static_cast<QKeyEvent*>(event)->key();
        if (key == Qt::Key_Return || key == Qt::Key_Enter) {
            if (_onEnter) {
                _onEnter();
            }
            return true;
        }
        if (key == Qt::Key_Escape) {
            _field->clear();
            return true;
        }
        return false;
    }

private:
    QLineEdit* _field;
    std::function<void()> _onEnter;
};

} // namespace

int apply(QWidget* root, const QString& text)
{
    if (!root) {
        return 0;
    }
    const QStringList words = text.split(QLatin1Char(' '), Qt::SkipEmptyParts);
    int matches = 0;
    for (QGroupBox* group : root->findChildren<QGroupBox*>()) {
        if (isOwnedByInnerGroup(group, root)) {
            continue;
        }
        if (group->isHidden() && !group->property(kHiddenByFilter).toBool()) {
            continue;
        }
        int groupMatches = 0;
        for (const auto& row : layoutRows(group->layout())) {
            QStringList haystack{group->title()};
            for (QWidget* widget : row) {
                appendSearchText(widget, haystack);
            }
            const bool match = matchesAll(words, haystack.join(QLatin1Char('\n')));
            bool shown = false;
            for (QWidget* widget : row) {
                setFilteredVisible(widget, match);
                shown = shown || !widget->isHidden();
            }
            if (match && shown) {
                ++groupMatches;
            }
        }
        setFilteredVisible(group, words.isEmpty() || groupMatches > 0);
        matches += groupMatches;
    }
    return matches;
}

QLineEdit* createFilterField(QWidget* root, QLabel* noMatchLabel)
{
    auto* field = new QLineEdit(root);
    field->setObjectName(QStringLiteral("edtSettingsFilter"));
    field->setPlaceholderText(QObject::tr("Filter settings"));
    field->setClearButtonEnabled(true);
    installFilterKeys(field, [root, field]() { focusFirstMatch(root, field); });
    if (noMatchLabel) {
        noMatchLabel->hide();
    }
    QObject::connect(field, &QLineEdit::textChanged, root, [root, noMatchLabel](const QString& text) {
        const int matches = apply(root, text);
        if (noMatchLabel) {
            noMatchLabel->setVisible(matches == 0 && !text.trimmed().isEmpty());
        }
    });
    return field;
}

void installFilterKeys(QLineEdit* field, std::function<void()> onEnter)
{
    if (field) {
        field->installEventFilter(new FilterKeys(field, std::move(onEnter)));
    }
}

} // namespace vc3d::settings_filter
