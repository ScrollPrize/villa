#pragma once

#include <QDir>
#include <QString>

namespace vc3d {

inline QString settingsFilePath()
{
    const QString homeDir = QDir::homePath();
    const QString configDir = homeDir + "/.VC3D";
    QDir dir;
    if (!dir.exists(configDir)) {
        dir.mkpath(configDir);
    }
    return configDir + "/VC3D.ini";
}

// =============================================================================
// Setting Keys & Defaults
// =============================================================================
// Usage example:
//   using namespace vc3d::settings;
//   settings.value(viewer::INTERSECTION_OPACITY, viewer::INTERSECTION_OPACITY_DEFAULT)

namespace settings {

// -----------------------------------------------------------------------------
// Volume Package Settings
// -----------------------------------------------------------------------------
namespace volpkg {
    constexpr auto DEFAULT_PATH = "volpkg/default_path";
    constexpr auto AUTO_OPEN = "volpkg/auto_open";
    constexpr auto RECENT = "volpkg/recent";

    constexpr bool AUTO_OPEN_DEFAULT = true;
}

// -----------------------------------------------------------------------------
// Viewer Settings
// -----------------------------------------------------------------------------
namespace viewer {
    // Navigation & Interaction
    constexpr auto FWD_BACK_STEP_MS = "viewer/fwd_back_step_ms";
    constexpr auto CENTER_ON_ZOOM = "viewer/center_on_zoom";
    constexpr auto SCROLL_SPEED = "viewer/scroll_speed";
    constexpr auto IMPACT_RANGE_STEPS = "viewer/impact_range_steps";
    constexpr auto SCAN_RANGE_STEPS = "viewer/scan_range_steps";

    constexpr int FWD_BACK_STEP_MS_DEFAULT = 25;
    constexpr bool CENTER_ON_ZOOM_DEFAULT = false;
    constexpr int SCROLL_SPEED_DEFAULT = -1;
    constexpr auto IMPACT_RANGE_STEPS_DEFAULT = "1-3, 5, 8, 11, 15, 20, 28, 40, 60, 100, 200";
    constexpr auto SCAN_RANGE_STEPS_DEFAULT = "1, 2, 5, 10, 20, 50, 100, 200, 500, 1000";

    // Display & Appearance
    constexpr auto DISPLAY_SEGMENT_OPACITY = "viewer/display_segment_opacity";
    constexpr auto SHOW_DIRECTION_HINTS = "viewer/show_direction_hints";
    constexpr auto DIRECTION_STEP = "viewer/direction_step";
    constexpr auto USE_SEG_STEP_FOR_HINTS = "viewer/use_seg_step_for_hints";
    constexpr auto DIRECTION_STEP_POINTS = "viewer/direction_step_points";
    constexpr auto RESET_VIEW_ON_SURFACE_CHANGE = "viewer/reset_view_on_surface_change";
    constexpr auto MIRROR_CURSOR_TO_SEGMENTATION = "viewer/mirror_cursor_to_segmentation";

    constexpr int DISPLAY_SEGMENT_OPACITY_DEFAULT = 70;
    constexpr bool SHOW_DIRECTION_HINTS_DEFAULT = true;
    constexpr double DIRECTION_STEP_DEFAULT = 10.0;
    constexpr bool USE_SEG_STEP_FOR_HINTS_DEFAULT = true;
    constexpr int DIRECTION_STEP_POINTS_DEFAULT = 5;
    constexpr bool RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT = true;
    constexpr bool MIRROR_CURSOR_TO_SEGMENTATION_DEFAULT = false;

    // Volume Window (Base Grayscale Window)
    constexpr auto BASE_WINDOW_LOW = "viewer/base_window_low";
    constexpr auto BASE_WINDOW_HIGH = "viewer/base_window_high";

    constexpr float BASE_WINDOW_LOW_DEFAULT = 0.0f;
    constexpr float BASE_WINDOW_HIGH_DEFAULT = 255.0f;

    // Intersection Rendering
    constexpr auto INTERSECTION_OPACITY = "viewer/intersection_opacity";
    constexpr auto INTERSECTION_THICKNESS = "viewer/intersection_thickness";
    constexpr auto INTERSECTION_SAMPLING_STRIDE = "viewer/intersection_sampling_stride";

    constexpr int INTERSECTION_OPACITY_DEFAULT = 100;
    constexpr float INTERSECTION_THICKNESS_DEFAULT = 0.0f;
    constexpr int INTERSECTION_SAMPLING_STRIDE_DEFAULT = 1;

    // Axis Overlays
    constexpr auto SHOW_AXIS_OVERLAYS = "viewer/show_axis_overlays";
    constexpr auto AXIS_OVERLAY_OPACITY = "viewer/axis_overlay_opacity";
    constexpr auto USE_AXIS_ALIGNED_SLICES = "viewer/use_axis_aligned_slices";
    constexpr auto SLICE_STEP_SIZE = "viewer/slice_step_size";

    constexpr bool SHOW_AXIS_OVERLAYS_DEFAULT = true;
    constexpr int AXIS_OVERLAY_OPACITY_DEFAULT = 100;
    constexpr bool USE_AXIS_ALIGNED_SLICES_DEFAULT = true;
    constexpr int SLICE_STEP_SIZE_DEFAULT = 1;

    // Audio/UX
    constexpr auto PLAY_SOUND_AFTER_SEG_RUN = "viewer/play_sound_after_seg_run";
    constexpr auto USERNAME = "viewer/username";

    constexpr bool PLAY_SOUND_AFTER_SEG_RUN_DEFAULT = true;
    constexpr auto USERNAME_DEFAULT = "";
}

// -----------------------------------------------------------------------------
// Performance Settings
// -----------------------------------------------------------------------------
namespace perf {
    constexpr auto PRELOADED_SLICES = "perf/preloaded_slices";
    constexpr auto SKIP_IMAGE_FORMAT_CONV = "perf/chkSkipImageFormatConvExp";
    constexpr auto PARALLEL_PROCESSES = "perf/parallel_processes";
    constexpr auto ITERATION_COUNT = "perf/iteration_count";
    constexpr auto DOWNSCALE_OVERRIDE = "perf/downscale_override";
    constexpr auto FAST_INTERPOLATION = "perf/fast_interpolation";
    constexpr auto ENABLE_FILE_WATCHING = "perf/enable_file_watching";

    constexpr int PRELOADED_SLICES_DEFAULT = 200;
    constexpr bool SKIP_IMAGE_FORMAT_CONV_DEFAULT = false;
    constexpr int PARALLEL_PROCESSES_DEFAULT = 8;
    constexpr int ITERATION_COUNT_DEFAULT = 1000;
    constexpr int DOWNSCALE_OVERRIDE_DEFAULT = 0;
    constexpr bool FAST_INTERPOLATION_DEFAULT = false;
    constexpr bool ENABLE_FILE_WATCHING_DEFAULT = true;
}

// -----------------------------------------------------------------------------
// Main Window Settings
// -----------------------------------------------------------------------------
namespace window {
    constexpr auto GEOMETRY = "mainWin/geometry";
    constexpr auto STATE = "mainWin/state";
}

// -----------------------------------------------------------------------------
// Export Settings
// -----------------------------------------------------------------------------
namespace export_ {  // underscore because 'export' is reserved keyword
    constexpr auto CHUNK_WIDTH_PX = "export/chunk_width_px";
    constexpr auto CHUNK_OVERLAP_PX = "export/chunk_overlap_px";
    constexpr auto OVERWRITE = "export/overwrite";
    constexpr auto DIR = "export/dir";

    constexpr int CHUNK_WIDTH_PX_DEFAULT = 40000;
    constexpr int CHUNK_OVERLAP_PX_DEFAULT = 0;
    constexpr bool OVERWRITE_DEFAULT = true;
    constexpr auto DIR_DEFAULT = "";
}

// -----------------------------------------------------------------------------
// AWS Settings
// -----------------------------------------------------------------------------
namespace aws {
    constexpr auto DEFAULT_PROFILE = "aws/default_profile";

    constexpr auto DEFAULT_PROFILE_DEFAULT = "";
}

// -----------------------------------------------------------------------------
// Tools Settings
// -----------------------------------------------------------------------------
namespace tools {
    constexpr auto FLATBOI_PATH = "tools/flatboi_path";
    constexpr auto FLATBOI = "tools/flatboi";  // Legacy key
}

// -----------------------------------------------------------------------------
// Segmentation Tool Settings
// -----------------------------------------------------------------------------
namespace segmentation {
    // Tool group expansion states
    constexpr auto GROUP_EDITING_EXPANDED = "group_editing_expanded";
    constexpr auto GROUP_DRAG_EXPANDED = "group_drag_expanded";
    constexpr auto GROUP_LINE_EXPANDED = "group_line_expanded";
    constexpr auto GROUP_PUSH_PULL_EXPANDED = "group_push_pull_expanded";
    constexpr auto GROUP_DIRECTION_FIELD_EXPANDED = "group_direction_field_expanded";

    constexpr bool GROUP_EDITING_EXPANDED_DEFAULT = true;
    constexpr bool GROUP_DRAG_EXPANDED_DEFAULT = true;
    constexpr bool GROUP_LINE_EXPANDED_DEFAULT = true;
    constexpr bool GROUP_PUSH_PULL_EXPANDED_DEFAULT = true;
    constexpr bool GROUP_DIRECTION_FIELD_EXPANDED_DEFAULT = true;

    // Drag tool (note: these are stored in a QSettings group)
    constexpr auto DRAG_RADIUS_STEPS = "drag_radius_steps";
    constexpr auto DRAG_SIGMA_STEPS = "drag_sigma_steps";
    constexpr auto RADIUS_STEPS = "radius_steps";  // Legacy key
    constexpr auto SIGMA_STEPS = "sigma_steps";    // Legacy key

    // Line tool
    constexpr auto LINE_RADIUS_STEPS = "line_radius_steps";
    constexpr auto LINE_SIGMA_STEPS = "line_sigma_steps";

    // Push/Pull tool
    constexpr auto PUSH_PULL_RADIUS_STEPS = "push_pull_radius_steps";
    constexpr auto PUSH_PULL_SIGMA_STEPS = "push_pull_sigma_steps";
    constexpr auto PUSH_PULL_STEP = "push_pull_step";
    constexpr auto PUSH_PULL_ALPHA_START = "push_pull_alpha_start";
    constexpr auto PUSH_PULL_ALPHA_STOP = "push_pull_alpha_stop";
    constexpr auto PUSH_PULL_ALPHA_STEP = "push_pull_alpha_step";
    constexpr auto PUSH_PULL_ALPHA_LOW = "push_pull_alpha_low";
    constexpr auto PUSH_PULL_ALPHA_HIGH = "push_pull_alpha_high";
    constexpr auto PUSH_PULL_ALPHA_BORDER = "push_pull_alpha_border";
    constexpr auto PUSH_PULL_ALPHA_RADIUS = "push_pull_alpha_radius";
    constexpr auto PUSH_PULL_ALPHA_LIMIT = "push_pull_alpha_limit";
    constexpr auto PUSH_PULL_ALPHA_PER_VERTEX = "push_pull_alpha_per_vertex";

    // Smoothing
    constexpr auto SMOOTH_STRENGTH = "smooth_strength";
    constexpr auto SMOOTH_ITERATIONS = "smooth_iterations";

    // Growth
    constexpr auto GROWTH_METHOD = "growth_method";
    constexpr auto GROWTH_STEPS = "growth_steps";
    constexpr auto GROWTH_DIRECTION_MASK = "growth_direction_mask";
    constexpr auto DIRECTION_FIELDS = "direction_fields";
    constexpr auto GROWTH_KEYBINDS_ENABLED = "growth_keybinds_enabled";

    constexpr bool GROWTH_KEYBINDS_ENABLED_DEFAULT = true;

    // Corrections
    constexpr auto CORRECTIONS_ENABLED = "corrections_enabled";
    constexpr auto CORRECTIONS_Z_RANGE_ENABLED = "corrections_z_range_enabled";
    constexpr auto CORRECTIONS_Z_MIN = "corrections_z_min";
    constexpr auto CORRECTIONS_Z_MAX = "corrections_z_max";

    // Custom parameters
    constexpr auto CUSTOM_PARAMS_TEXT = "custom_params_text";

    // Hover marker
    constexpr auto SHOW_HOVER_MARKER = "show_hover_marker";

    // Approval brush
    constexpr auto APPROVAL_BRUSH_RADIUS = "approval_brush_radius";
    constexpr auto APPROVAL_BRUSH_DEPTH = "approval_brush_depth";
    constexpr auto APPROVAL_MASK_OPACITY = "approval_mask_opacity";
    constexpr auto APPROVAL_BRUSH_COLOR = "approval_brush_color";
    constexpr auto SHOW_APPROVAL_MASK = "show_approval_mask";

    constexpr bool CORRECTIONS_ENABLED_DEFAULT = false;
    constexpr bool CORRECTIONS_Z_RANGE_ENABLED_DEFAULT = false;
    constexpr int CORRECTIONS_Z_MIN_DEFAULT = 0;
}

// -----------------------------------------------------------------------------
// Volume Overlay Settings (stored in a QSettings group per overlay)
// -----------------------------------------------------------------------------
namespace volume_overlay {
    constexpr auto PATH = "path";
    constexpr auto VOLUME_ID = "volume_id";
    constexpr auto OPACITY = "opacity";
    constexpr auto WINDOW_LOW = "window_low";
    constexpr auto WINDOW_HIGH = "window_high";
    constexpr auto THRESHOLD = "threshold";  // Legacy key
    constexpr auto COLORMAP = "colormap";
}

} // namespace settings
} // namespace vc3d
