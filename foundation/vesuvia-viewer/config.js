export const CONCURRENCY = 6               // requests in flight per origin (shared by CT and layer): what a browser opens to one HTTP/1.1 host
export const WORKERS = 3
export const PART_BYTES = 2 ** 20
export const REPLAN_MS = 100
export const ZOOM_TOLERANCE = 1.35
export const FIT_FRACTION = 0.9             // wanted chunks may fill this fraction of the atlas. Tilted panes in a 1400×900 window want up to 225 of the CT's 252 slots, and a plan that does not fit gives up its intermediate levels
export const WINDOW_CHUNKS = 64
export const PIN_COARSEST_CHUNKS = 64
export const UPLOAD_BYTES_PER_FRAME = 8 * 2 ** 20   // chunk and page-table bytes sent to the GPU in one frame, four 128³ chunks of uint8: a cached reload of the scroll shows in 2.5 s, 3.6 at 4 MiB, every frame kept at 16.8 ms
export const UPLOAD_CALLS_PER_FRAME = 64       // an image brick of 128² is two calls, its 16 KiB and its page entry: at 8 the surface pane took 13.1 s to fill at 4 image pixels a CSS pixel, at 64 4.7 s, about when its last tile arrived, every frame kept at 16.8 ms; bytes bound the large ones
export const CT_ATLAS_BYTES = 512 * 2 ** 20   // WebGL does not say what a device holds: at 1024 MiB a software renderer loses the context, with no error to catch. The hash's atlas= key, in MiB, replaces it
export const LAYER_ATLAS_BYTES = 384 * 2 ** 20
export const MEMORY_ATLAS_BYTES = 128 * 2 ** 20
export const IMAGE_ATLAS_BYTES = 64 * 2 ** 20 // GPU budget of the surface image: 4096 slots of IMAGE_BRICK_SIZE² uint8; the surface pane alone in a 2560×1440 window at the finest zoom wants 1800 with its prefetch ring
export const LAYER_BRICK_SIZE = 128
export const IMAGE_BRICK_SIZE = 128
export const CT_CACHE_BYTES = 256 * 2 ** 20
export const LAYER_CACHE_BYTES = 128 * 2 ** 20
export const IMAGE_CACHE_BYTES = 256 * 2 ** 20
export const AHEAD_MS = 2000                // the coarsest level is fetched where the focus will be after this long, at most one of its chunks away: a far bucket answers in 0.2 to 1.8 s, and a pane whose coarsest chunk is late draws nothing there
export const PREFETCH_SLICES = 16          // display-level voxels ahead of the last slice-step direction: a wheel's 12 slices a second then stay on the display level, where without them 4% of the pane is a level coarser
export const SURFACE_MAX_POINTS = 4 * 2 ** 20
export const SLAB_MAX_SAMPLES = 64
export const SURFACE_DEMAND_TILES = 8
export const MARKER_REACH_CELLS = 2
export const OUTLINE_BLOCK = 8
export const OUTLINE_GROUP = 8
export const HASH_DEBOUNCE_MS = 300
export const STATUS_MS = 200
export const ZOOM_STEP = 2 ** (1 / 4)
export const ZOOM_RANGE = [1 / 32, 1024]
export const WHEEL_NOTCH_PX = 100          // wheel deltaY in CSS px that zooms by one ZOOM_STEP: a notch in Chromium on Windows. A high-resolution or smooth-scrolling wheel sends many smaller events
export const WHEEL_LINE_PX = 40            // CSS px per line of a deltaMode 1 wheel event, Chromium's own, so Firefox's three lines a notch zoom one ZOOM_STEP
export const wheelZoom = (deltaY, deltaMode) =>
  ZOOM_STEP ** Math.max(-1, Math.min(1, deltaY * [1, WHEEL_LINE_PX, WHEEL_NOTCH_PX][deltaMode] / WHEEL_NOTCH_PX))
export const HELD_ZOOM_OCTAVE_MS = 528
export const HELD_STEP_MS = 8
export const HELD_WATCHDOG_MS = 2000       // held navigation stops this long after the last key event: macOS's slowest wait for its first repeat is about 1.8 s, and at 1 s a hold there dropped out once
export const STEP_RANGE = [1, 20]
export const PAGE_STEPS = 10
export const CLICK_SLOP_PX = 3
export const DTYPE_MAX = { uint8: 255, uint16: 65535, float32: 1 }
export const DTYPE_ARRAY = { uint8: Uint8Array, uint16: Uint16Array, float32: Float32Array }
export const AUTO_WINDOW_PERCENTILES = [0.02, 0.98]
export const AUTO_WINDOW_SAMPLES = 2 ** 20
export const COLORS = {
  background: '#0e0f11',
  pane: '#14161a',
  border: '#2a2d33',
  text: '#d6d9de',
  muted: '#8a9099',
  accent: '#7fdbff',
  outline: '#ffd166',
  error: '#ff6b6b',
}
export const hexRgb = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255)
export const RETRY_MS = 10_000
export const REGION_PRIORITY = 999         // volume.read() chunks load after every visible chunk (< 1000, views/plan.js), before prefetch
export const REGION_MAX_BYTES = 512 * 2 ** 20

// Origins besides the page's own whose modules a link's ext= loads without asking. Every path of an origin
// listed here is trusted, so not a CDN or a host that serves anyone's files.
export const EXTENSION_ORIGINS = []
// The most layers the viewer opens. Every one is sampled for every pixel of every pane, so this is a choice of
// speed; the device may allow fewer (layerCapacity() in gpu/gl.js), and viewer.maxLayers is what holds.
export const MAX_LAYERS = 8
export const LAYER_COLORS = [[1, 0, 1], [0, 0.8, 1], [1, 0.7, 0], [0.3, 1, 0.3], [1, 0.3, 0.3], [0.5, 0.5, 1], [1, 1, 0.3], [0.3, 1, 0.9]]
export const LAYER_OPACITY = 0.6
export const LAYER_THRESHOLD = 0.3
export const ROTATE_RADIANS_PER_PX = 0.005
export const ROTATE_STEP_RADIANS = Math.PI / 180
export const HELD_ROTATE_MS = 33
export const TILT_DEMAND_TILES = 4
export const OUTLINE_CACHE_MARGIN_PX = 128
export const OUTLINE_CELL_PX = 1
export const OUTLINE_PLANE_EPSILON = 1e-7
