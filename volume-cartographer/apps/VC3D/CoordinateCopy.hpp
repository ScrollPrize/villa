#pragma once

#include <QAbstractItemView>
#include <QGraphicsProxyWidget>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QString>
#include <QTextEdit>

#include <opencv2/core/matx.hpp>

namespace vc3d {

// What the Focus field accepts: three non-negative whole numbers.
inline constexpr const char* kFocusFieldPattern = "^\\s*\\d+\\s*,\\s*\\d+\\s*,\\s*\\d+\\s*$";

// One coordinate in whole voxels, rounded the way the viewers' status bars
// round their [X=, Y=, Z=] readout. A value that rounds to zero is written
// "0", never "-0".
inline QString wholeVoxelText(float value)
{
    const QString text = QString::number(value, 'f', 0);
    return text == QLatin1String("-0") ? QStringLiteral("0") : text;
}

// A volume position as "x, y, z" in whole voxels: the form the Focus field
// shows and accepts, so a copied position can be pasted back into it.
inline QString coordinateTripleText(const cv::Vec3f& position)
{
    return QStringLiteral("%1, %2, %3")
        .arg(wholeVoxelText(position[0]),
             wholeVoxelText(position[1]),
             wholeVoxelText(position[2]));
}

// True when each coordinate rounds to a voxel inside a volume of the given
// width, height and slice count. Only such positions paste back into the
// Focus field as they are: it takes no negative numbers and clamps to the
// volume. Rounding is half away from zero, so -0.5 would be written "-1".
inline bool roundsInsideVolume(const cv::Vec3f& position, int width, int height, int slices)
{
    const float sizes[3] = {float(width), float(height), float(slices)};
    for (int i = 0; i < 3; ++i) {
        if (!(position[i] > -0.5f && position[i] < sizes[i] - 0.5f)) {
            return false;
        }
    }
    return true;
}

// True when Ctrl+C, having reached the main window, already did something
// for the focused widget, so the cursor coordinates must not replace it:
// item views copy their current cell and still pass the key on, and a viewer
// passes on every key, including Ctrl+C that a field embedded in its scene
// (such as the Rotate Surface angle box) used to copy its selected text.
inline bool focusedWidgetHandlesCopy(const QWidget* focus)
{
    if (qobject_cast<const QAbstractItemView*>(focus)) {
        return true;
    }
    const auto* view = qobject_cast<const QGraphicsView*>(focus);
    const auto* proxy = view && view->scene()
        ? dynamic_cast<const QGraphicsProxyWidget*>(view->scene()->focusItem())
        : nullptr;
    const QWidget* embedded = proxy ? proxy->widget() : nullptr;
    if (!embedded) {
        return false;
    }
    auto hasSelectedText = [](const QWidget* widget) {
        if (const auto* edit = qobject_cast<const QLineEdit*>(widget)) {
            return edit->hasSelectedText();
        }
        if (const auto* edit = qobject_cast<const QTextEdit*>(widget)) {
            return edit->textCursor().hasSelection();
        }
        if (const auto* edit = qobject_cast<const QPlainTextEdit*>(widget)) {
            return edit->textCursor().hasSelection();
        }
        return false;
    };
    if (hasSelectedText(embedded)) {
        return true;
    }
    for (const QWidget* child : embedded->findChildren<QWidget*>()) {
        if (hasSelectedText(child)) {
            return true;
        }
    }
    return false;
}

} // namespace vc3d
