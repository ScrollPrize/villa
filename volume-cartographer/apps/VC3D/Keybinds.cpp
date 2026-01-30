#include "Keybinds.hpp"

#include <QString>

namespace vc3d::keybinds {

namespace {
constexpr const char* kSectionFileMenu = "File Menu";
constexpr const char* kSectionViewMenu = "View Menu";
constexpr const char* kSectionViewerControls = "Viewer Controls";
constexpr const char* kSectionNavigation = "Navigation";
constexpr const char* kSectionSegments = "Segments";
constexpr const char* kSectionSegEditing = "Segmentation Editing";
constexpr const char* kSectionApprovalMask = "Approval Mask";
constexpr const char* kSectionSegGrowth = "Segmentation Growth";
constexpr const char* kSectionPushPull = "Push/Pull";
constexpr const char* kSectionPointCollection = "Point Collection";
constexpr const char* kSectionMouseControls = "Mouse Controls";
constexpr const char* kSectionSliceStep = "Slice Step Size";

enum class HelpKeyType {
    Shortcut,
    KeyPress,
    Literal
};

struct HelpEntry {
    const char* section;
    const char* description;
    HelpKeyType keyType;
    const ShortcutDef* shortcut;
    const KeyPressDef* keypress;
    const char* literal;
};

QString formatSequence(const QKeySequence& sequence)
{
    return sequence.toString(QKeySequence::NativeText);
}
} // namespace

namespace shortcuts {
// Add new keybind definitions here, then list them in buildKeybindsHelpText() if they should be user-visible.
const ShortcutDef OpenVolpkg{
    "open_volpkg",
    kSectionFileMenu,
    "Open Volume Package",
    ShortcutKind::Standard,
    nullptr,
    QKeySequence::Open
};
const ShortcutDef TeleaInpaint{
    "telea_inpaint",
    kSectionFileMenu,
    "Inpaint (Telea) & Rebuild Segment",
    ShortcutKind::Text,
    "Ctrl+I",
    QKeySequence::Open
};
const ShortcutDef AxisAlignedSlices{
    "axis_aligned_slices",
    kSectionViewMenu,
    "Toggle axis-aligned slice planes",
    ShortcutKind::Text,
    "Ctrl+J",
    QKeySequence::Open
};
const ShortcutDef SurfaceNormals{
    "surface_normals",
    kSectionViewMenu,
    "Toggle surface normals visualization",
    ShortcutKind::Text,
    "Ctrl+N",
    QKeySequence::Open
};
const ShortcutDef DirectionHints{
    "direction_hints",
    kSectionViewMenu,
    "Toggle direction hints (flip_x arrows)",
    ShortcutKind::Text,
    "Ctrl+T",
    QKeySequence::Open
};
const ShortcutDef CompositeView{
    "composite_view",
    kSectionViewMenu,
    "Toggle composite view",
    ShortcutKind::Text,
    "C",
    QKeySequence::Open
};
const ShortcutDef DrawingMode{
    "drawing_mode",
    kSectionViewMenu,
    "Toggle drawing mode",
    ShortcutKind::Text,
    "Ctrl+Shift+D",
    QKeySequence::Open
};
const ShortcutDef RawPointsOverlay{
    "raw_points_overlay",
    kSectionViewMenu,
    "Toggle raw points overlay",
    ShortcutKind::Text,
    "P",
    QKeySequence::Open
};
const ShortcutDef ZoomIn{
    "zoom_in",
    kSectionViewerControls,
    "Zoom in (active viewer)",
    ShortcutKind::Text,
    "Shift+=",
    QKeySequence::Open
};
const ShortcutDef ZoomOut{
    "zoom_out",
    kSectionViewerControls,
    "Zoom out (active viewer)",
    ShortcutKind::Text,
    "Shift+-",
    QKeySequence::Open
};
const ShortcutDef ResetView{
    "reset_view",
    kSectionViewerControls,
    "Reset view (fit surface and reset Z offset)",
    ShortcutKind::Text,
    "m",
    QKeySequence::Open
};
const ShortcutDef WorldOffsetZPos{
    "world_offset_z_pos",
    kSectionViewerControls,
    "Z offset deeper (surface normal direction)",
    ShortcutKind::Text,
    "Ctrl+.",
    QKeySequence::Open
};
const ShortcutDef WorldOffsetZNeg{
    "world_offset_z_neg",
    kSectionViewerControls,
    "Z offset closer (surface normal direction)",
    ShortcutKind::Text,
    "Ctrl+,",
    QKeySequence::Open
};
const ShortcutDef CycleNextSegment{
    "cycle_next_segment",
    kSectionSegments,
    "Cycle to next visible segment",
    ShortcutKind::Text,
    "]",
    QKeySequence::Open
};
const ShortcutDef CyclePrevSegment{
    "cycle_prev_segment",
    kSectionSegments,
    "Cycle to previous visible segment",
    ShortcutKind::Text,
    "[",
    QKeySequence::Open
};
const ShortcutDef FocusedView{
    "focused_view",
    kSectionViewMenu,
    "Toggle focused view",
    ShortcutKind::Text,
    "Shift+Ctrl+F",
    QKeySequence::Open
};
const ShortcutDef CycleViewers{
    "cycle_viewers",
    kSectionViewerControls,
    "Cycle between viewers",
    ShortcutKind::Text,
    "Ctrl+Tab",
    QKeySequence::Open
};
} // namespace shortcuts

namespace keypress {
const KeyPressDef ToggleVolumeOverlay{
    "toggle_volume_overlay",
    kSectionViewMenu,
    "Toggle volume overlay visibility",
    Qt::Key_Space,
    Qt::NoModifier,
    false
};
const KeyPressDef CenterFocusOnCursor{
    "center_focus_on_cursor",
    kSectionNavigation,
    "Center focus on cursor",
    Qt::Key_R,
    Qt::NoModifier,
    false
};
const KeyPressDef FocusHistoryBack{
    "focus_history_back",
    kSectionNavigation,
    "Step backward in focus history",
    Qt::Key_F,
    Qt::NoModifier,
    false
};
const KeyPressDef FocusHistoryForward{
    "focus_history_forward",
    kSectionNavigation,
    "Step forward in focus history",
    Qt::Key_F,
    Qt::ControlModifier,
    false
};
const KeyPressDef SliceStepDecrease{
    "slice_step_decrease",
    kSectionSliceStep,
    "Decrease slice step size",
    Qt::Key_G,
    Qt::ShiftModifier,
    false
};
const KeyPressDef SliceStepIncrease{
    "slice_step_increase",
    kSectionSliceStep,
    "Increase slice step size",
    Qt::Key_H,
    Qt::ShiftModifier,
    false
};

const KeyPressDef ApprovalPaintToggle{
    "approval_paint_toggle",
    kSectionApprovalMask,
    "Toggle approval painting (when mask is shown)",
    Qt::Key_B,
    Qt::NoModifier,
    true
};
const KeyPressDef UnapprovalPaintToggle{
    "unapproval_paint_toggle",
    kSectionApprovalMask,
    "Toggle unapproval painting (when mask is shown)",
    Qt::Key_N,
    Qt::NoModifier,
    true
};
const KeyPressDef ApprovalUndo{
    "approval_undo",
    kSectionApprovalMask,
    "Undo last approval mask stroke",
    Qt::Key_B,
    Qt::ControlModifier,
    true
};
const KeyPressDef SegmentationUndo{
    "segmentation_undo",
    kSectionSegEditing,
    "Undo last change (segmentation or approval mask)",
    Qt::Key_Z,
    Qt::ControlModifier,
    true
};
const KeyPressDef LineDrawHold{
    "line_draw_hold",
    kSectionSegEditing,
    "Line draw mode (hold)",
    Qt::Key_S,
    Qt::NoModifier,
    true
};
const KeyPressDef GrowSegmentation{
    "grow_segmentation",
    kSectionSegGrowth,
    "Grow segmentation (all directions)",
    Qt::Key_G,
    Qt::ControlModifier,
    true
};
const KeyPressDef CancelOperation{
    "cancel_operation",
    kSectionSegEditing,
    "Cancel current drag/operation",
    Qt::Key_Escape,
    Qt::NoModifier,
    false
};
const KeyPressDef PushPullIn{
    "push_pull_in",
    kSectionPushPull,
    "Push (move inward) (hold)",
    Qt::Key_A,
    Qt::NoModifier,
    false
};
const KeyPressDef PushPullOut{
    "push_pull_out",
    kSectionPushPull,
    "Pull (move outward) (hold)",
    Qt::Key_D,
    Qt::NoModifier,
    false
};
const KeyPressDef PushPullInAlpha{
    "push_pull_in_alpha",
    kSectionPushPull,
    "Push with alpha override",
    Qt::Key_A,
    Qt::ControlModifier,
    false
};
const KeyPressDef PushPullOutAlpha{
    "push_pull_out_alpha",
    kSectionPushPull,
    "Pull with alpha override",
    Qt::Key_D,
    Qt::ControlModifier,
    false
};
const KeyPressDef PushPullRadiusDown{
    "push_pull_radius_down",
    kSectionPushPull,
    "Decrease push/pull radius",
    Qt::Key_Q,
    Qt::NoModifier,
    false
};
const KeyPressDef PushPullRadiusUp{
    "push_pull_radius_up",
    kSectionPushPull,
    "Increase push/pull radius",
    Qt::Key_E,
    Qt::NoModifier,
    false
};
const KeyPressDef EnableEditing{
    "enable_editing",
    kSectionSegEditing,
    "Enable editing",
    Qt::Key_T,
    Qt::ShiftModifier,
    false
};
const KeyPressDef ToggleAnnotation{
    "toggle_annotation",
    kSectionSegEditing,
    "Toggle correction annotation mode",
    Qt::Key_T,
    Qt::NoModifier,
    false
};
const KeyPressDef GrowthLeft{
    "growth_left",
    kSectionSegGrowth,
    "Grow left",
    Qt::Key_1,
    Qt::NoModifier,
    true
};
const KeyPressDef GrowthUp{
    "growth_up",
    kSectionSegGrowth,
    "Grow up",
    Qt::Key_2,
    Qt::NoModifier,
    true
};
const KeyPressDef GrowthDown{
    "growth_down",
    kSectionSegGrowth,
    "Grow down",
    Qt::Key_3,
    Qt::NoModifier,
    true
};
const KeyPressDef GrowthRight{
    "growth_right",
    kSectionSegGrowth,
    "Grow right",
    Qt::Key_4,
    Qt::NoModifier,
    true
};
const KeyPressDef GrowthAll{
    "growth_all",
    kSectionSegGrowth,
    "Grow all directions",
    Qt::Key_5,
    Qt::NoModifier,
    true
};
const KeyPressDef GrowthStepAll{
    "growth_step_all",
    kSectionSegGrowth,
    "Grow one step (all directions)",
    Qt::Key_6,
    Qt::NoModifier,
    true
};

const KeyPressDef DeletePoint{
    "delete_point",
    kSectionPointCollection,
    "Remove selected point",
    Qt::Key_Delete,
    Qt::NoModifier,
    false
};
} // namespace keypress

namespace range_slider {
const KeyPressDef StepDownLeft{
    "range_slider_left",
    "Range Slider (Focused)",
    "Step range down",
    Qt::Key_Left,
    Qt::NoModifier,
    false
};
const KeyPressDef StepDownDown{
    "range_slider_down",
    "Range Slider (Focused)",
    "Step range down",
    Qt::Key_Down,
    Qt::NoModifier,
    false
};
const KeyPressDef StepUpRight{
    "range_slider_right",
    "Range Slider (Focused)",
    "Step range up",
    Qt::Key_Right,
    Qt::NoModifier,
    false
};
const KeyPressDef StepUpUp{
    "range_slider_up",
    "Range Slider (Focused)",
    "Step range up",
    Qt::Key_Up,
    Qt::NoModifier,
    false
};
const KeyPressDef PageDown{
    "range_slider_page_down",
    "Range Slider (Focused)",
    "Step range down (page)",
    Qt::Key_PageDown,
    Qt::NoModifier,
    false
};
const KeyPressDef PageUp{
    "range_slider_page_up",
    "Range Slider (Focused)",
    "Step range up (page)",
    Qt::Key_PageUp,
    Qt::NoModifier,
    false
};
} // namespace range_slider

QKeySequence sequenceFor(const ShortcutDef& def)
{
    if (def.kind == ShortcutKind::Standard) {
        return QKeySequence(def.standardKey);
    }
    return QKeySequence(QString::fromUtf8(def.sequenceText));
}

QKeySequence sequenceFor(const KeyPressDef& def)
{
    return QKeySequence(def.key | def.modifiers);
}

QString buildKeybindsHelpText()
{
    const HelpEntry kHelpEntries[] = {
        { kSectionFileMenu, shortcuts::OpenVolpkg.description, HelpKeyType::Shortcut, &shortcuts::OpenVolpkg, nullptr, nullptr },
        { kSectionFileMenu, shortcuts::TeleaInpaint.description, HelpKeyType::Shortcut, &shortcuts::TeleaInpaint, nullptr, nullptr },

        { kSectionViewMenu, shortcuts::AxisAlignedSlices.description, HelpKeyType::Shortcut, &shortcuts::AxisAlignedSlices, nullptr, nullptr },
        { kSectionViewMenu, shortcuts::SurfaceNormals.description, HelpKeyType::Shortcut, &shortcuts::SurfaceNormals, nullptr, nullptr },
        { kSectionViewMenu, shortcuts::DirectionHints.description, HelpKeyType::Shortcut, &shortcuts::DirectionHints, nullptr, nullptr },
        { kSectionViewMenu, shortcuts::CompositeView.description, HelpKeyType::Shortcut, &shortcuts::CompositeView, nullptr, nullptr },
        { kSectionViewMenu, shortcuts::DrawingMode.description, HelpKeyType::Shortcut, &shortcuts::DrawingMode, nullptr, nullptr },
        { kSectionViewMenu, shortcuts::RawPointsOverlay.description, HelpKeyType::Shortcut, &shortcuts::RawPointsOverlay, nullptr, nullptr },
        { kSectionViewMenu, keypress::ToggleVolumeOverlay.description, HelpKeyType::KeyPress, nullptr, &keypress::ToggleVolumeOverlay, nullptr },
        { kSectionViewMenu, shortcuts::FocusedView.description, HelpKeyType::Shortcut, &shortcuts::FocusedView, nullptr, nullptr },

        { kSectionViewerControls, shortcuts::CycleViewers.description, HelpKeyType::Shortcut, &shortcuts::CycleViewers, nullptr, nullptr },
        { kSectionViewerControls, shortcuts::ZoomIn.description, HelpKeyType::Shortcut, &shortcuts::ZoomIn, nullptr, nullptr },
        { kSectionViewerControls, shortcuts::ZoomOut.description, HelpKeyType::Shortcut, &shortcuts::ZoomOut, nullptr, nullptr },
        { kSectionViewerControls, shortcuts::ResetView.description, HelpKeyType::Shortcut, &shortcuts::ResetView, nullptr, nullptr },
        { kSectionViewerControls, shortcuts::WorldOffsetZPos.description, HelpKeyType::Shortcut, &shortcuts::WorldOffsetZPos, nullptr, nullptr },
        { kSectionViewerControls, shortcuts::WorldOffsetZNeg.description, HelpKeyType::Shortcut, &shortcuts::WorldOffsetZNeg, nullptr, nullptr },
        { kSectionViewerControls, "Pan view", HelpKeyType::Literal, nullptr, nullptr, "Arrow Keys" },

        { kSectionNavigation, keypress::CenterFocusOnCursor.description, HelpKeyType::KeyPress, nullptr, &keypress::CenterFocusOnCursor, nullptr },
        { kSectionNavigation, keypress::FocusHistoryBack.description, HelpKeyType::KeyPress, nullptr, &keypress::FocusHistoryBack, nullptr },
        { kSectionNavigation, keypress::FocusHistoryForward.description, HelpKeyType::KeyPress, nullptr, &keypress::FocusHistoryForward, nullptr },

        { kSectionSegments, shortcuts::CycleNextSegment.description, HelpKeyType::Shortcut, &shortcuts::CycleNextSegment, nullptr, nullptr },
        { kSectionSegments, shortcuts::CyclePrevSegment.description, HelpKeyType::Shortcut, &shortcuts::CyclePrevSegment, nullptr, nullptr },

        { kSectionSegEditing, keypress::SegmentationUndo.description, HelpKeyType::KeyPress, nullptr, &keypress::SegmentationUndo, nullptr },
        { kSectionSegEditing, keypress::LineDrawHold.description, HelpKeyType::KeyPress, nullptr, &keypress::LineDrawHold, nullptr },
        { kSectionSegEditing, keypress::ToggleAnnotation.description, HelpKeyType::KeyPress, nullptr, &keypress::ToggleAnnotation, nullptr },
        { kSectionSegEditing, "Add correction point", HelpKeyType::Literal, nullptr, nullptr, "  Click" },
        { kSectionSegEditing, "Move existing correction point", HelpKeyType::Literal, nullptr, nullptr, "  Shift+Drag" },
        { kSectionSegEditing, "Remove correction point", HelpKeyType::Literal, nullptr, nullptr, "  Right-click" },
        { kSectionSegEditing, keypress::EnableEditing.description, HelpKeyType::KeyPress, nullptr, &keypress::EnableEditing, nullptr },
        { kSectionSegEditing, keypress::CancelOperation.description, HelpKeyType::KeyPress, nullptr, &keypress::CancelOperation, nullptr },

        { kSectionApprovalMask, keypress::ApprovalPaintToggle.description, HelpKeyType::KeyPress, nullptr, &keypress::ApprovalPaintToggle, nullptr },
        { kSectionApprovalMask, keypress::UnapprovalPaintToggle.description, HelpKeyType::KeyPress, nullptr, &keypress::UnapprovalPaintToggle, nullptr },
        { kSectionApprovalMask, keypress::ApprovalUndo.description, HelpKeyType::KeyPress, nullptr, &keypress::ApprovalUndo, nullptr },

        { kSectionSegGrowth, keypress::GrowSegmentation.description, HelpKeyType::KeyPress, nullptr, &keypress::GrowSegmentation, nullptr },
        { kSectionSegGrowth, keypress::GrowthLeft.description, HelpKeyType::KeyPress, nullptr, &keypress::GrowthLeft, nullptr },
        { kSectionSegGrowth, keypress::GrowthUp.description, HelpKeyType::KeyPress, nullptr, &keypress::GrowthUp, nullptr },
        { kSectionSegGrowth, keypress::GrowthDown.description, HelpKeyType::KeyPress, nullptr, &keypress::GrowthDown, nullptr },
        { kSectionSegGrowth, keypress::GrowthRight.description, HelpKeyType::KeyPress, nullptr, &keypress::GrowthRight, nullptr },
        { kSectionSegGrowth, keypress::GrowthAll.description, HelpKeyType::KeyPress, nullptr, &keypress::GrowthAll, nullptr },
        { kSectionSegGrowth, keypress::GrowthStepAll.description, HelpKeyType::KeyPress, nullptr, &keypress::GrowthStepAll, nullptr },

        { kSectionPushPull, keypress::PushPullIn.description, HelpKeyType::KeyPress, nullptr, &keypress::PushPullIn, nullptr },
        { kSectionPushPull, keypress::PushPullOut.description, HelpKeyType::KeyPress, nullptr, &keypress::PushPullOut, nullptr },
        { kSectionPushPull, keypress::PushPullInAlpha.description, HelpKeyType::KeyPress, nullptr, &keypress::PushPullInAlpha, nullptr },
        { kSectionPushPull, keypress::PushPullOutAlpha.description, HelpKeyType::KeyPress, nullptr, &keypress::PushPullOutAlpha, nullptr },
        { kSectionPushPull, keypress::PushPullRadiusDown.description, HelpKeyType::KeyPress, nullptr, &keypress::PushPullRadiusDown, nullptr },
        { kSectionPushPull, keypress::PushPullRadiusUp.description, HelpKeyType::KeyPress, nullptr, &keypress::PushPullRadiusUp, nullptr },

        { kSectionPointCollection, keypress::DeletePoint.description, HelpKeyType::KeyPress, nullptr, &keypress::DeletePoint, nullptr },

        { kSectionMouseControls, "Select/Add point or drag surface", HelpKeyType::Literal, nullptr, nullptr, "Left Click" },
        { kSectionMouseControls, "Drag surface point or draw with active tool", HelpKeyType::Literal, nullptr, nullptr, "Left Drag" },
        { kSectionMouseControls, "Pan slice image", HelpKeyType::Literal, nullptr, nullptr, "Right Drag" },
        { kSectionMouseControls, "Pan (if enabled)", HelpKeyType::Literal, nullptr, nullptr, "Middle Drag" },
        { kSectionMouseControls, "Zoom in/out (when editing: adjust tool radius)", HelpKeyType::Literal, nullptr, nullptr, "Mouse Wheel" },
        { kSectionMouseControls, "Pan through slices", HelpKeyType::Literal, nullptr, nullptr, "Shift+Scroll Wheel" },

        { kSectionSliceStep, keypress::SliceStepDecrease.description, HelpKeyType::KeyPress, nullptr, &keypress::SliceStepDecrease, nullptr },
        { kSectionSliceStep, keypress::SliceStepIncrease.description, HelpKeyType::KeyPress, nullptr, &keypress::SliceStepIncrease, nullptr },
    };

    QString result;
    QString currentSection;
    for (const auto& entry : kHelpEntries) {
        const QString section = QString::fromUtf8(entry.section);
        if (section != currentSection) {
            if (!result.isEmpty()) {
                result += "\n";
            }
            result += "=== " + section + " ===\n";
            currentSection = section;
        }
        QString keyText;
        if (entry.keyType == HelpKeyType::Shortcut && entry.shortcut) {
            keyText = formatSequence(sequenceFor(*entry.shortcut));
        } else if (entry.keyType == HelpKeyType::KeyPress && entry.keypress) {
            keyText = formatSequence(sequenceFor(*entry.keypress));
        } else if (entry.keyType == HelpKeyType::Literal && entry.literal) {
            keyText = QString::fromUtf8(entry.literal);
        }
        if (keyText.isEmpty()) {
            continue;
        }
        result += keyText + ": " + QString::fromUtf8(entry.description) + "\n";
    }

    return result.trimmed();
}

} // namespace vc3d::keybinds
