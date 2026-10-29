#pragma once

#include <QKeySequence>
#include <QString>

#include <vector>

namespace vc3d::keybinds {

enum class ShortcutKind {
    Text,
    Standard
};

struct ShortcutDef {
    const char* id;
    const char* section;
    const char* description;
    ShortcutKind kind;
    const char* sequenceText;
    QKeySequence::StandardKey standardKey;
};

struct KeyPressDef {
    const char* id;
    const char* section;
    const char* description;
    Qt::Key key;
    Qt::KeyboardModifiers modifiers;
    bool requireNoAutoRepeat;
};

namespace shortcuts {
extern const ShortcutDef CommandPalette;
extern const ShortcutDef OpenLasagnaWorkspace;
extern const ShortcutDef RepeatLasagnaAction;
extern const ShortcutDef OpenVolpkg;
extern const ShortcutDef AxisAlignedSlices;
extern const ShortcutDef SurfaceNormals;
extern const ShortcutDef CompositeView;
extern const ShortcutDef RawPointsOverlay;
extern const ShortcutDef ZoomIn;
extern const ShortcutDef ZoomOut;
extern const ShortcutDef ResetView;
extern const ShortcutDef ZoomToFit;
extern const ShortcutDef WorldOffsetZPos;
extern const ShortcutDef WorldOffsetZNeg;
extern const ShortcutDef CycleNextSegment;
extern const ShortcutDef CyclePrevSegment;
extern const ShortcutDef ApplyApprovedTag;
extern const ShortcutDef ApplyDefectiveTag;
extern const ShortcutDef FocusedView;
extern const ShortcutDef CycleViewers;
} // namespace shortcuts

namespace keypress {
extern const KeyPressDef ToggleVolumeOverlay;
extern const KeyPressDef CenterFocusOnCursor;
extern const KeyPressDef RecenterFocus;
extern const KeyPressDef SliceStepDecrease;
extern const KeyPressDef SliceStepIncrease;

extern const KeyPressDef SegmentationUndo;
extern const KeyPressDef ManualAddToggle;
extern const KeyPressDef GrowSegmentation;
extern const KeyPressDef CancelOperation;
extern const KeyPressDef PushPullIn;
extern const KeyPressDef PushPullOut;
extern const KeyPressDef PushPullInAlpha;
extern const KeyPressDef PushPullOutAlpha;
extern const KeyPressDef PushPullRadiusDown;
extern const KeyPressDef PushPullRadiusUp;
extern const KeyPressDef EnableEditing;
extern const KeyPressDef ToggleAnnotation;
extern const KeyPressDef GrowthLeft;
extern const KeyPressDef GrowthUp;
extern const KeyPressDef GrowthDown;
extern const KeyPressDef GrowthRight;
extern const KeyPressDef GrowthAll;
extern const KeyPressDef GrowthStepAll;
extern const KeyPressDef GrowthFill;

extern const KeyPressDef DeletePoint;
} // namespace keypress

namespace range_slider {
extern const KeyPressDef StepDownLeft;
extern const KeyPressDef StepDownDown;
extern const KeyPressDef StepUpRight;
extern const KeyPressDef StepUpUp;
extern const KeyPressDef PageDown;
extern const KeyPressDef PageUp;
} // namespace range_slider

namespace standard {
constexpr QKeySequence::StandardKey Undo = QKeySequence::Undo;
constexpr QKeySequence::StandardKey Open = QKeySequence::Open;
} // namespace standard

QKeySequence sequenceFor(const ShortcutDef& def);
QKeySequence sequenceFor(const KeyPressDef& def);

// One row of the keybinds help: a key or mouse gesture and what it does.
struct HelpRow {
    QString section;
    QString description;
    QString keys;           // display text, native format for registry keys
    QString id;             // ShortcutDef or KeyPressDef id; empty for literal rows
    QKeySequence sequence;  // the registry key; empty for literal rows
};

// Every user-visible key binding, in display order. An id may appear in more
// than one section.
std::vector<HelpRow> helpRows();

} // namespace vc3d::keybinds
