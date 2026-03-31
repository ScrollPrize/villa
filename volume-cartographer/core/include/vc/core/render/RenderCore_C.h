#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Platform-agnostic volume rendering — C API for WASM/FFI/SDL/etc.
// All functions are thread-safe with respect to different contexts.
// A single context must not be used from multiple threads simultaneously.

void*           vc3d_create(int width, int height);
void            vc3d_destroy(void* ctx);
void            vc3d_resize(void* ctx, int width, int height);

// Volume (caller retains ownership of the Volume*)
void            vc3d_set_volume(void* ctx, void* volume);

// Camera: set the cutting plane (origin + normal)
void            vc3d_set_plane(void* ctx,
                               float ox, float oy, float oz,
                               float nx, float ny, float nz);
void            vc3d_pan(void* ctx, float dx, float dy);
void            vc3d_zoom(void* ctx, float factor);
void            vc3d_scroll(void* ctx, float dz);

// Window/level
void            vc3d_set_window_level(void* ctx, float window, float level);

// Render: returns number of tiles rendered
int             vc3d_render(void* ctx);

// Framebuffer: ARGB32, row-major, width*height uint32_t values
const uint32_t* vc3d_pixels(void* ctx);
int             vc3d_width(void* ctx);
int             vc3d_height(void* ctx);

#ifdef __cplusplus
}
#endif
