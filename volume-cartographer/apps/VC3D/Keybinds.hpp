#pragma once

#include <QKeySequence>
#include <QString>

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
extern const ShortcutDef OpenVolpkg;
extern const ShortcutDef TeleaInpaint;
extern const ShortcutDef AxisAlignedSlices;
extern const ShortcutDef SurfaceNormals;
extern const ShortcutDef DirectionHints;
extern const ShortcutDef CompositeView;
extern const ShortcutDef DrawingMode;
extern const ShortcutDef RawPointsOverlay;
extern const ShortcutDef ZoomIn;
extern const ShortcutDef ZoomOut;
extern const ShortcutDef ResetView;
extern const ShortcutDef WorldOffsetZPos;
extern const ShortcutDef WorldOffsetZNeg;
extern const ShortcutDef CycleNextSegment;
extern const ShortcutDef CyclePrevSegment;
extern const ShortcutDef FocusedView;
extern const ShortcutDef CycleViewers;
} // namespace shortcuts

namespace keypress {
extern const KeyPressDef ToggleVolumeOverlay;
extern const KeyPressDef CenterFocusOnCursor;
extern const KeyPressDef FocusHistoryBack;
extern const KeyPressDef FocusHistoryForward;
extern const KeyPressDef SliceStepDecrease;
extern const KeyPressDef SliceStepIncrease;

extern const KeyPressDef ApprovalPaintToggle;
extern const KeyPressDef UnapprovalPaintToggle;
extern const KeyPressDef ApprovalUndo;
extern const KeyPressDef SegmentationUndo;
extern const KeyPressDef LineDrawHold;
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

QString buildKeybindsHelpText();

} // namespace vc3d::keybinds
