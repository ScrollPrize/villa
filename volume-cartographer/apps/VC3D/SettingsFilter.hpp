#pragma once

#include <functional>

class QLabel;
class QLineEdit;
class QString;
class QWidget;

namespace vc3d::settings_filter {

// Shows only the settings rows under root that match every whitespace-separated
// word of text, ignoring case. A row is one line of a group box's grid or form
// layout. It matches through its labels, check box and button texts, tooltips,
// placeholders, spin box suffixes and special values, combo box items, and the
// group's title, so a group whose title matches keeps all its rows. Group boxes
// without a matching row are hidden. Widgets hidden before filtering stay
// hidden. An empty text shows everything again. Returns the number of matching
// rows that have a visible widget.
int apply(QWidget* root, const QString& text);

// A "Filter settings" field that applies the filter to root as the text
// changes, with the keys of installFilterKeys: while it holds text, Enter
// moves focus to the first matching control. noMatchLabel, when given, is
// shown while nothing matches.
QLineEdit* createFilterField(QWidget* root, QLabel* noMatchLabel = nullptr);

// Keys for a filter field in a dialog. While the field holds text, Enter calls
// onEnter instead of pressing the dialog's default button, and Escape clears
// the field instead of closing the dialog. With an empty field both keys reach
// the dialog as usual.
void installFilterKeys(QLineEdit* field, std::function<void()> onEnter);

} // namespace vc3d::settings_filter
