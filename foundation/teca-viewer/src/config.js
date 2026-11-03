// Sizes are bytes, times are milliseconds, lengths are full-resolution voxels unless the name says otherwise.

export const SETTLE_DELAY_MS = 320;
export const INTERACTION_SETTLE_DELAY_MS = 260;
export const VIEW_RESIZE_RENDER_DELAY_MS = 100;
export const BASE_VIEW_WIDTH_VOXELS = 768;
export const SURFACE_BASE_VIEW_CELLS = 1024;
export const SCALE_PICK_MAX_PIXELS = 640;
export const SCALE_PICK_MIN_PIXELS = 32;
export const SCALE_PICK_TOLERANCE = 1.35;

export const HELD_SLICE_INTERVAL_MS = 8;
export const HELD_SLICE_STEP = 2;
export const HELD_SLICE_STEP_INCREMENT = 1;
export const MIN_HELD_SLICE_STEP = 1;
export const MAX_HELD_SLICE_STEP = 20;
// A held slice or arrow key with no repeat for this long counts as released, so a keyup lost to a focus change stops
// it; macOS's slowest wait for its first repeat is about 1.8 s.
export const HELD_WATCHDOG_MS = 2000;
export const HELD_ZOOM_OCTAVE_MS = 297;
export const KEY_ZOOM_FACTOR = 1.2;
export const HELD_ROTATION_MS = 33;

// An [x, y, z, w] quaternion in the camera's XYZ frame.
export const DEFAULT_CAMERA_ORIENTATION = [
  -0.3470984282541847, 0.2971192589835064, 0.11695403655523892, 0.8817962238397273,
];
export const DEFAULT_ZOOM = 1.3;
export const MIN_ZOOM = 0.05;
export const MAX_ZOOM = 8;

export const SLICE_AHEAD_MAX_LEAD_CHUNKS = 8;
export const SLICE_AHEAD_MAX_QUEUED_CHUNKS = 48;
export const SHOWN_CHUNK_PRIORITY_LIMIT = 400;
export const SLICE_AHEAD_PREFETCH_PRIORITY = 500;
export const PREFETCH_CHUNK_PRIORITY = 8000;
export const SURFACE_GEOMETRY_PRIORITY = 1000;

export const ORTHOGONAL_PLANES = ["xy", "zy", "xz"];

export const CHUNK_CACHE_BYTE_LIMIT = 256 * 1024 * 1024;
export const CHUNK_CACHE_MIB_RANGE = Object.freeze([64, 16384]);
export const DOWNLOAD_CACHE_BYTE_LIMIT = 2 * 1024 ** 3;
export const DOWNLOAD_CACHE_GIB_RANGE = Object.freeze([0.25, 1024]);
// An IndexedDB database not open by then is given up, download storage until settings retry it: a page left while it
// created or upgraded the database can hold every later open for as long as the browser keeps it.
export const STORAGE_OPEN_LIMIT_MS = 3000;
// A stored download whose host gives it no lifetime is served unasked for this share of the time from its
// Last-Modified to its storing, as Chromium's HTTP cache serves such a body (RFC 9111 §4.2.2).
export const HEURISTIC_FRESHNESS_SHARE = 0.1;
export const CACHE_TRAILING_NAVIGATION_GENERATIONS = 80;
export const CACHE_COARSE_LEVEL_OFFSET = 2;
export const CACHE_COARSE_RESIDENCY_FRACTION = 0.25;

export const CT_BRICKS_PER_FRAME = 16;
export const UPLOAD_BYTES_PER_FRAME = 512 * 1024;
export const IDLE_UPLOAD_BYTES_PER_FRAME = 4 * 1024 * 1024;
export const INTERACTIVE_UPLOAD_BYTES_LIMIT = 2 * 1024 * 1024;
export const UPLOAD_TIME_LIMIT_MS = 4;
export const CHUNK_READS_IN_FLIGHT = 32;
export const PREFETCH_READS_IN_FLIGHT = 8;
export const PLANE_PREFETCH_CHUNKS_IN_FLIGHT = 4;
export const BRICK_CUT_BYTES = 128 * 1024 * 1024;
export const WORKER_QUEUE_LIMIT = 64;
export const WORKER_CACHE_BYTE_LIMIT = 64 * 1024 * 1024;
export const BRICK_EDGE = 64;
export const MIN_BRICK_EDGE = 16;
export const ATLAS_BYTE_LIMIT = 1024 * 1024 * 1024;
export const PRECOMPILE_IDLE_TIMEOUT_MS = 250;
export const GPU_BUDGET_BYTES = 384 * 1024 * 1024;
export const GPU_BUDGET_MIN_MIB = 256;
export const FILTER_PICTURE_BYTES = 256 * 1024 * 1024;
export const CT_ATLAS_BYTES = 288 * 1024 * 1024;
export const CT_LEVEL_LIMIT = 4;
export const CT_REPLAN_INTERVAL_MS = 100;

export const SURFACE_WORK_LIMIT = 256 * 1024 * 1024;

export const OVERLAY_RENDER_VERTEX_LIMIT = 1_000_000;
export const OVERLAY_LINE_WIDTH_RANGE = Object.freeze([0.5, 30]);
export const DISPLAY_GAMMA_RANGE = Object.freeze([0.25, 3]);

// Connections a browser opens to one HTTP/1.1 hostname; a request beyond them waits inside the browser,
// out of reach of priority and cancellation. A reader's reads for one navigation take at most all but one of them.
export const HOST_REQUEST_CONCURRENCY = 6;
// A read unanswered this long after a hostname admitted it is also sent to another hostname of the object.
// S3 chunk reads answer within about 3 s at p95; a stalled one takes 7-16 s.
export const HEDGE_AFTER_MS = 3000;
// A Zarr metadata read that none of its hostnames has answered this long after one admitted it fails, and with it the
// volume's open: a stalled S3 read answers in 7-16 s, and some S3 never answers.
export const ZARR_METADATA_TIME_LIMIT_MS = 20000;
// A HEAD checking a stale stored chunk that is unanswered this long is given up, and the chunk read over HTTP. It
// counts the wait inside the browser behind reads, as a reload checks every stale chunk at once: 8 of the checks' 74
// passed 3 s with six browsers running. A stalled S3 read answers in 7-16 s.
export const DOWNLOAD_VALIDATION_TIME_LIMIT_MS = 20000;
export const FETCH_ATTEMPTS = 3;
// A failed request is tried again after this long, and each later time after twice the wait before: 0.5 s then 1 s,
// the mean waits of AWS's SDKs after S3's 503 SlowDown, which S3 answers while it scales to a prefix's request rate.
export const FETCH_RETRY_DELAY_MS = 500;
export const LATENCY_WEIGHT = 0.2;
export const CHUNK_JOBS_IN_FLIGHT = 32;
export const CHUNK_STAGING_BYTES = 256 * 1024 * 1024;
export const DECODER_REQUEST_LIMIT = 32;
export const ZARR_METADATA_BYTE_LIMIT = 4 * 1024 * 1024;
export const ZARR_METADATA_TOTAL_BYTE_LIMIT = 16 * 1024 * 1024;
export const ZARR_OBJECT_BYTE_LIMIT = 64 * 1024 * 1024;
export const WIRE_CHUNK_BYTE_LIMIT = 16 * 1024 * 1024;
export const ZARR_SHARD_INDEX_BYTE_LIMIT = 256 * 1024;
export const ZARR_SCALE_LIMIT = 32;
export const VIEWER_CHANNEL_LIMIT = 64;
export const DISPLAY_WINDOW_SAMPLE_CHUNKS = 16;
export const DISPLAY_WINDOW_SAMPLE_VALUES = 65536;
export const DISPLAY_WINDOW_SAMPLE_PRIORITY = 1000;
export const DISPLAY_WINDOW_TAIL = 0.005;

export const LOADED_COVERAGE = 0.99;
// Most device pixels across a CT pane the layout hides is drawn at for the 3D pane: at a pixel ratio of 2 the
// maximised 3D pane spans a plane 800 across, and level 0 for its three planes reads about three times the chunks of
// 512's level 1, over 2 GiB in six moves of the focus, which a move back reads again.
export const HIDDEN_PANE_MAX_PIXELS = 512;
export const PREDICTED_CHUNK_LIMIT = 32;
export const STATUS_HOLD_MS = 4000;
export const NETWORK_STATUS_REFRESH_MS = 1000;
export const NETWORK_SPEED_SAMPLES = 4;
// Session changes closer than this are one burst, whose first and last reach the address: Safari refuses more than
// 100 address writes in 30 s, Chromium more than 200 in 10 s.
export const SESSION_PUBLISH_DELAY_MS = 160;
export const SESSION_JSON_BYTE_LIMIT = 32 * 1024;
// Bytes of entries a zip's Blob takes at once: making a Blob waits on the browser to register it, the longer the more
// Blobs it holds, and a Blob for each of the 3,748 entries of a stroke of radius 256 across the Surface took 0.57 to
// 0.82 s.
export const ZIP_BLOB_BYTES = 4 * 2 ** 20;

export const SLICE_AHEAD_VELOCITY_WEIGHT = 0.125;
export const SLICE_AHEAD_MIN_HORIZON_MS = 50;
export const SLICE_AHEAD_MAX_HORIZON_MS = 1000;
export const SLICE_AHEAD_FIRST_LATENCY_MS = 120;
export const SLICE_AHEAD_HORIZON_MARGIN = 1.5;
export const ROTATION_PREDICTION_LIFETIME_MS = 250;
export const ROTATION_PREDICTION_LATENCY_FACTOR = 1.5;
export const ROTATION_PREDICTION_MIN_HORIZON_MS = 50;
export const ROTATION_PREDICTION_FIRST_LATENCY_MS = 120;
export const ROTATION_PREDICTION_TURN_LIMIT_RADIANS = Math.PI / 18;

export const CLICK_SLOP = 4;
export const WHEEL_ZOOM_FACTOR = 1.12;
export const ROTATION_STEP_DEGREES = 1;
export const RELIEF_MASK_KEY_STEP = 0.01;
export const SURFACE_DIVIDER_KEY_STEP = 0.025;
export const SURFACE_ROTATION_DEGREES_PER_PIXEL = 0.4;
export const PLANE_TILT_DEGREES_PER_PIXEL = 0.4;
export const CAMERA_DEGREES_PER_PIXEL = 0.6;
export const SURFACE_TILT_LIMIT = 85;
export const SURFACE_ZOOM_LIMITS = Object.freeze([0.05, 512]);
export const SURFACE_NORMAL_OFFSET_LIMIT = 256;
export const SURFACE_NORMAL_STEP_LIMITS = Object.freeze([0.01, 32]);

export const SURFACE_REPLAN_DELAY_MS = 100;
export const SURFACE_ZOOM_SEARCH_LIMIT = 64;
// A mesh window of at most this many cells gets a bound per cell; bounding and packing 4,096 cells takes about 1.5 ms.
export const SURFACE_CELL_BOUNDS_LIMIT = 4096;
export const SURFACE_STRIP_ROWS = 64;
export const SURFACE_STRIP_BYTE_LIMIT = 64 * 1024 * 1024;
export const SURFACE_READ_BYTES = 4 * 1024 * 1024;
export const SURFACE_READS_IN_FLIGHT = 6;
export const SURFACE_STRIP_CACHE_BYTES = 64 * 1024 * 1024;
export const SURFACE_PYRAMID_CACHE_BYTES = 256 * 1024 * 1024;
export const GEOMETRY_CACHE_BYTES = 512 * 1024 * 1024;
export const TIFXYZ_TILE_CACHE_BYTES = 64 * 1024 * 1024;
export const TIFXYZ_TILE_READ_BYTES = 64 * 1024 * 1024;
export const TIFXYZ_TILES_IN_FLIGHT = 4;
export const TIFXYZ_FILE_BYTE_LIMIT = 512 * 1024 * 1024;
export const TIFXYZ_PIXEL_LIMIT = 64 * 1024 * 1024;
export const TIFXYZ_META_BYTE_LIMIT = 1024 * 1024;
export const TIFF_TILE_BYTE_LIMIT = 16 * 1024 * 1024;
export const SURFACE_OVERVIEW_POINT_LIMIT = 512 * 512;
export const FOCUS_STRIP_BYTES = 64 * 1024 * 1024;
export const SURFACE_FOCUS_RADIUS = 256;
export const SURFACE_FOCUS_CANDIDATE_LIMIT = 64;
export const SURFACE_SEARCH_SLICE_MS = 4;
export const SURFACE_MARKER_OFF_NORMAL_LIMIT = 1;
export const SURFACE_INTERSECTION_CELL_LIMIT = 65536;
export const SURFACE_INTERSECTION_CACHE_BYTES = 16 * 1024 * 1024;
export const SURFACE_CUT_DELAY_MS = 100;
export const SURFACE_CUT_DEPTH = 0.2;
export const SURFACE_CUT_REACH = 1.2;
export const SURFACE_FALLBACK_BUDGET_SHARE = 0.45;
export const SURFACE_PAGE_TABLE_RESERVE_BYTES = 8 * 1024 * 1024;
