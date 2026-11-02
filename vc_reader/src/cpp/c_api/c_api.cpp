// The C API (vc_reader/vc_reader_c.h): handles around the C++ reader, no logic of its
// own beyond converting arguments, mapping exceptions to statuses and the volume's
// GPU-failure policy.
#define VCR_C_BUILDING 1
#include "vc_reader/vc_reader_c.h"

#include "vc_reader/cache.hpp"
#include "vc_reader/composite.hpp"
#include "vc_reader/cuda_available.hpp"
#include "vc_reader/display.hpp"
#include "vc_reader/grid.hpp"
#include "vc_reader/slab.hpp"
#include "vc_reader/tifxyz.hpp"
#include "vc_reader/zarr.hpp"

#include "../array.hpp"
#ifdef VC_READER_CUDA
#include "../reader_cuda/reader.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

thread_local std::string lastError;

vcr_status fail(vcr_status status, const char* message)
{
    lastError = message;
    return status;
}

template <class F>
vcr_status guarded(F&& f) noexcept
{
    try {
        return f();
    } catch (const std::length_error& e) {
        return fail(VCR_ERROR_BUFFER_TOO_SMALL, e.what());
    } catch (const std::bad_alloc&) {
        return fail(VCR_ERROR_OUT_OF_MEMORY, "out of memory");
    } catch (const std::invalid_argument& e) {
        return fail(VCR_ERROR_INVALID_ARGUMENT, e.what());
    } catch (const std::out_of_range& e) {
        return fail(VCR_ERROR_INVALID_ARGUMENT, e.what());
    } catch (const std::exception& e) {
        return fail(VCR_ERROR_RUNTIME, e.what());
    } catch (...) {
        return fail(VCR_ERROR_RUNTIME, "unknown error");
    }
}

void need(bool condition, const char* message)
{
    if (!condition)
        throw std::invalid_argument(message);
}

template <class T>
void needStruct(const T* s, const char* name)
{
    need(s != nullptr, name);
    need(s->struct_size >= sizeof(T), name);
}

// One opened pyramid: its levels by number (absent ones empty) in level-0 voxels.
struct Volume {
    std::vector<std::unique_ptr<vcr::zarr::OpenedArray>> levels;
#ifdef VC_READER_CUDA
    std::vector<std::unique_ptr<vcr::reader_cuda::Reader>> cuda;
#endif
    std::atomic<bool> gpuFailed{false};

    ~Volume()
    {
#ifdef VC_READER_CUDA
        cuda.clear();  // readers refer to the arrays
#endif
        for (auto& level : levels)
            if (level)
                level->close();
    }

    bool has(int level) const
    {
        return level >= 0 && std::size_t(level) < levels.size() && levels[std::size_t(level)];
    }

    vcr::reader_cuda::Reader* cudaReader(int level) const
    {
#ifdef VC_READER_CUDA
        if (!gpuFailed.load() && std::size_t(level) < cuda.size())
            return cuda[std::size_t(level)].get();
#else
        (void)level;
#endif
        return nullptr;
    }

    vcr::grid::Level gridLevel(int level) const
    {
        return vcr::zarr::gridLevel(*levels[std::size_t(level)], {}, cudaReader(level));
    }

    std::vector<vcr::grid::Level> gridLevels() const
    {
        std::vector<vcr::grid::Level> out(levels.size());
        for (std::size_t level = 0; level < levels.size(); ++level)
            if (levels[level])
                out[level] = gridLevel(int(level));
        return out;
    }

    void renderOnCpu() { gpuFailed.store(true); }
};

}  // namespace

struct vcr_volume {
    std::shared_ptr<Volume> volume;
};
struct vcr_surface {
    std::shared_ptr<const vcr::grid::Surface> surface;
    int rows = 0, cols = 0;
};
struct vcr_pending {
    vcr::grid::Pending pending;
    std::vector<std::shared_ptr<Volume>> volumes;
};

namespace {

vcr::grid::Raster toRaster(const vcr_raster& r)
{
    return {r.x0, r.y0, r.sx, r.sy, r.width, r.height, r.kx, r.ky};
}

vcr::grid::Frame toFrame(const vcr_frame* frame)
{
    need(frame != nullptr, "frame is NULL");
    need((frame->plane != nullptr) != (frame->surface != nullptr), "a frame has exactly one of a plane and a surface");
    vcr::grid::Frame out;
    if (frame->plane) {
        const auto& p = *frame->plane;
        vcr::grid::Plane plane;
        std::copy_n(p.origin, 3, plane.origin.begin());
        std::copy_n(p.dx, 3, plane.dx.begin());
        std::copy_n(p.dy, 3, plane.dy.begin());
        std::copy_n(p.normal, 3, plane.normal.begin());
        plane.width = p.width;
        plane.height = p.height;
        need(plane.width >= 0 && plane.height >= 0, "negative plane size");
        out.plane = plane;
    } else {
        out.surface = frame->surface->surface;
        out.raster = toRaster(frame->raster);
        need(out.raster.width >= 0 && out.raster.height >= 0, "negative raster size");
    }
    return out;
}

vcr::composite::Options toComposite(const vcr_composite& c)
{
    need(c.method >= VCR_COMPOSITE_MEAN && c.method <= VCR_COMPOSITE_BEER_LAMBERT, "unknown composite method");
    vcr::composite::Options o;
    o.method = static_cast<vcr::composite::Method>(c.method);
    o.isoCutoff = c.iso_cutoff;
    o.alphaMin = c.alpha_min;
    o.alphaMax = c.alpha_max;
    o.alphaOpacity = c.alpha_opacity;
    o.alphaCutoff = c.alpha_cutoff;
    o.extinction = c.extinction;
    o.emission = c.emission;
    o.ambient = c.ambient;
    return o;
}

// The request's levels (start, then present coarser ones) and their level numbers.
vcr::grid::Request toRequest(const vcr_request* r, std::vector<int>& numbers)
{
    needStruct(r, "request is NULL or not initialised");
    need(r->volume != nullptr, "request has no volume");
    const Volume& volume = *r->volume->volume;
    need(volume.has(r->start_level), "start_level is not a present level");
    need(r->order == 0 || r->order == 1, "order must be 0 (nearest) or 1 (linear)");
    need(r->offsets != nullptr || r->offset_count == 0, "offset_count without offsets");
    vcr::grid::Request request;
    const int last = std::min(int(volume.levels.size()) - 1, r->start_level + std::max(0, r->fallback_levels));
    for (int level = r->start_level; level <= last; ++level)
        if (volume.has(level)) {
            request.levels.push_back(volume.gridLevel(level));
            numbers.push_back(level);
        }
    request.order = r->order;
    if (r->offset_count)
        request.offsets.assign(r->offsets, r->offsets + r->offset_count);
    request.pixels = r->pixels;
    if (r->composite)
        request.composite = toComposite(*r->composite);
    request.cachedOnly = r->cached_only != 0;
    return request;
}

std::size_t valueCount(const vcr::grid::Frame& frame, const vcr_request* r)
{
    const std::size_t layers = r->composite ? 1 : std::max<std::size_t>(1, r->offset_count);
    return std::size_t(frame.width()) * std::size_t(frame.height()) * layers;
}

void checkResult(const vcr_result* result, std::size_t values, std::size_t pixels)
{
    needStruct(result, "result is NULL or not initialised");
    if (result->values_capacity < values || !result->values)
        throw std::length_error("values buffer too small");
    if (result->levels && result->levels_capacity < pixels)
        throw std::length_error("levels buffer too small");
}

void writeResult(const vcr::grid::Result& r, const std::vector<int>& numbers, Volume& volume, vcr_result* out)
{
    out->width = r.width;
    out->height = r.height;
    out->layers = r.width && r.height ? int(r.values.size() / (std::size_t(r.width) * std::size_t(r.height))) : 0;
    std::copy(r.values.begin(), r.values.end(), out->values);
    if (out->levels)
        for (std::size_t q = 0; q < r.level.size(); ++q) {
            const auto l = r.level[q];
            out->levels[q] = l >= 0 ? std::int8_t(numbers[std::size_t(l)]) : l;
        }
    out->gpu_failed = !r.cudaFailure.empty();
    if (out->gpu_failed) {
        volume.renderOnCpu();
        lastError = r.cudaFailure;
    }
}

template <class F>
vcr_status sampleWith(const vcr_frame* frame, const vcr_request* request, vcr_result* result, F&& sample)
{
    return guarded([&] {
        const auto f = toFrame(frame);
        std::vector<int> numbers;
        auto req = toRequest(request, numbers);
        checkResult(result, valueCount(f, request), std::size_t(f.width()) * std::size_t(f.height()));
        const auto r = sample(f, std::move(req));
        writeResult(r, numbers, *request->volume->volume, result);
        return VCR_OK;
    });
}

}  // namespace

extern "C" {

int vcr_abi_version(void) { return VCR_C_ABI_VERSION; }

int vcr_has_cuda(void)
{
#ifdef VC_READER_CUDA
    return vcr::cuda::available() ? 1 : 0;
#else
    return 0;
#endif
}

const char* vcr_last_error(void) { return lastError.c_str(); }

void vcr_open_options_init(vcr_open_options* o)
{
    if (!o)
        return;
    *o = {};
    o->struct_size = sizeof(*o);
    o->cache_writes = 1;
    o->memory_cache_bytes = std::int64_t(2) << 30;
}

vcr_status vcr_volume_open(const char* location, const vcr_open_options* options, vcr_volume** out)
{
    return guarded([&] {
        need(location && out, "location and out are required");
        *out = nullptr;
        vcr_open_options defaults;
        vcr_open_options_init(&defaults);
        if (!options)
            options = &defaults;
        needStruct(options, "options are not initialised");
        need(options->cache_dirs || !options->cache_dir_count, "cache_dir_count without cache_dirs");
        need(options->memory_cache_bytes >= 0 && options->gpu_cache_bytes >= 0, "negative cache size");
        vcr::zarr::OpenOptions open;
        open.offline = options->offline != 0;
        open.access = vcr::zarr::resolveAccess(location, vcr::zarr::AuthMode::Auto, open.offline);
        for (std::size_t i = 0; i < options->cache_dir_count; ++i)
            open.cacheDirs.emplace_back(options->cache_dirs[i]);
        open.cacheWrites = options->cache_writes != 0;
        open.threads = options->threads;
        if (options->memory_cache_bytes)
            open.memoryCache = std::make_shared<vcr::MemoryCache>(std::size_t(options->memory_cache_bytes));

        auto volume = std::make_shared<Volume>();
        const auto infos = vcr::zarr::pyramidLevels(location, open);
        need(!infos.empty(), "the location holds no pyramid levels");
        volume->levels.resize(std::size_t(infos.back().level) + 1);
        std::string firstError;
        for (const auto& info : infos) {
            try {
                volume->levels[std::size_t(info.level)] = std::make_unique<vcr::zarr::OpenedArray>(
                    vcr::zarr::openPyramidLevel(location, info.level, open));
                need(volume->levels[std::size_t(info.level)]->meta.shape.size() == 3, "volume levels must be 3-D");
            } catch (const std::exception& e) {
                volume->levels[std::size_t(info.level)].reset();
                if (firstError.empty())
                    firstError = e.what();
            }
        }
        if (std::none_of(volume->levels.begin(), volume->levels.end(), [](const auto& l) { return bool(l); }))
            throw std::runtime_error("no level of the volume opened: " + firstError);
#ifdef VC_READER_CUDA
        volume->cuda.resize(volume->levels.size());
        if (options->gpu_cache_bytes > 0 && vcr::cuda::available())
            for (std::size_t level = 0; level < volume->levels.size(); ++level) {
                const auto& l = volume->levels[level];
                if (l && l->meta.type.kind == 'u' && l->meta.type.itemsize == 1)
                    volume->cuda[level] = std::make_unique<vcr::reader_cuda::Reader>(
                        *l->array, std::size_t(options->gpu_cache_bytes), 0);
            }
#endif
        *out = new vcr_volume{std::move(volume)};
        return VCR_OK;
    });
}

void vcr_volume_release(vcr_volume* volume) { delete volume; }

void vcr_level_info_init(vcr_level_info* info)
{
    if (!info)
        return;
    *info = {};
    info->struct_size = sizeof(*info);
}

int vcr_volume_level_count(const vcr_volume* volume) { return volume ? int(volume->volume->levels.size()) : 0; }

vcr_status vcr_volume_level_info(const vcr_volume* volume, int level, vcr_level_info* out)
{
    return guarded([&] {
        need(volume != nullptr, "volume is NULL");
        needStruct(out, "info is NULL or not initialised");
        const Volume& v = *volume->volume;
        need(level >= 0 && std::size_t(level) < v.levels.size(), "no such level");
        const std::size_t size = out->struct_size;
        *out = {};
        out->struct_size = size;
        if (!v.has(level))
            return VCR_OK;
        const auto& l = *v.levels[std::size_t(level)];
        out->present = 1;
        for (std::size_t a = 0; a < 3; ++a) {
            out->shape[a] = l.meta.shape[a];
            out->chunks[a] = l.meta.chunks[a];
        }
        std::snprintf(out->dtype, sizeof(out->dtype), "%s", l.meta.dtype.c_str());
        const auto extent = vcr::grid::voxelExtent(v.gridLevel(level));
        std::copy(extent.begin(), extent.end(), out->voxel_extent);
        out->gpu = v.cudaReader(level) != nullptr;
        return VCR_OK;
    });
}

int vcr_level_for_view(const vcr_volume* volume, float pixelsPerVoxel, float qualityBias)
{
    if (!volume)
        return 0;
    return vcr::grid::levelForView(volume->volume->gridLevels(), pixelsPerVoxel, qualityBias);
}

int vcr_fallback_levels(const vcr_volume* volume, int start, int width, int height, float pixelsPerVoxel, int maximum)
{
    if (!volume)
        return 0;
    try {
        return vcr::grid::fallbackLevels(volume->volume->gridLevels(), start, width, height, pixelsPerVoxel, maximum);
    } catch (const std::exception& e) {
        lastError = e.what();
        return 0;
    }
}

vcr_status vcr_volume_prefill(vcr_volume* volume, int level, vcr_progress_fn progress, void* user)
{
    return guarded([&] {
        need(volume != nullptr && volume->volume->has(level), "no such level");
        std::atomic<bool> cancel{false};
        const auto r = vcr::cache::prefill(*volume->volume->levels[std::size_t(level)]->array, &cancel,
                                           [&](std::size_t resolved, std::size_t total) {
                                               if (progress && progress(user, resolved, total))
                                                   cancel.store(true);
                                           });
        using Status = vcr::cache::PrefillResult::Status;
        if (r.status == Status::Cancelled)
            return fail(VCR_ERROR_CANCELLED, "prefill cancelled");
        if (r.status == Status::Failed)
            return fail(VCR_ERROR_RUNTIME, r.message.c_str());
        return VCR_OK;
    });
}

void vcr_surface_options_init(vcr_surface_options* o)
{
    if (!o)
        return;
    *o = {};
    o->struct_size = sizeof(*o);
    o->interpolation = VCR_INTERPOLATION_BICUBIC;
    o->spacing[0] = o->spacing[1] = o->spacing[2] = 1.0f;
}

namespace {

vcr::grid::SurfaceOptions toSurfaceOptions(const vcr_surface_options* o)
{
    vcr::grid::SurfaceOptions out;
    if (!o)
        return out;
    needStruct(o, "surface options are not initialised");
    need(o->interpolation == VCR_INTERPOLATION_BICUBIC || o->interpolation == VCR_INTERPOLATION_LINEAR,
         "unknown interpolation");
    out.interpolation = o->interpolation == VCR_INTERPOLATION_LINEAR ? vcr::grid::Interpolation::Linear
                                                                      : vcr::grid::Interpolation::Bicubic;
    std::copy_n(o->spacing, 3, out.spacing.begin());
    out.flipNormals = o->flip_normals != 0;
    return out;
}

}  // namespace

vcr_status vcr_surface_create(const float* xyz, int rows, int cols, std::size_t rowStride, const std::uint8_t* valid,
                              const vcr_surface_options* options, vcr_surface** out)
{
    return guarded([&] {
        need(xyz && out && rows > 0 && cols > 0, "xyz, out and a positive shape are required");
        *out = nullptr;
        if (!rowStride)
            rowStride = std::size_t(cols) * 3;
        need(rowStride >= std::size_t(cols) * 3, "row_stride is shorter than a row");
        // The snapshot owns its vertices; validity is taken here when given.
        std::vector<std::uint8_t> flags;
        if (valid)
            flags.assign(valid, valid + std::size_t(rows) * std::size_t(cols));
        auto surface = vcr::grid::Surface::snapshot(
            {xyz, rows, cols, rowStride, nullptr, valid ? flags.data() : nullptr}, toSurfaceOptions(options));
        *out = new vcr_surface{std::move(surface), rows, cols};
        return VCR_OK;
    });
}

vcr_status vcr_surface_open_tifxyz(const char* location, const vcr_surface_options* options, vcr_surface** out,
                                   double scale[2])
{
    return guarded([&] {
        need(location && out, "location and out are required");
        *out = nullptr;
        const auto stored = vcr::tifxyz::read(location);
        const int rows = int(stored.rows()), cols = int(stored.cols());
        auto surface = vcr::grid::Surface::snapshot(
            {stored.xyz().data(), rows, cols, std::size_t(cols) * 3, nullptr, stored.valid().data()},
            toSurfaceOptions(options));
        if (scale) {
            scale[0] = stored.metadata().scale[0];
            scale[1] = stored.metadata().scale[1];
        }
        *out = new vcr_surface{std::move(surface), rows, cols};
        return VCR_OK;
    });
}

void vcr_surface_shape(const vcr_surface* surface, int* rows, int* cols)
{
    if (rows)
        *rows = surface ? surface->rows : 0;
    if (cols)
        *cols = surface ? surface->cols : 0;
}

void vcr_surface_release(vcr_surface* surface) { delete surface; }

vcr_status vcr_surface_points_at(const vcr_surface* surface, const float* xy, std::size_t n, float* xyz,
                                 float* normals, float* columnTangents, float* rowTangents)
{
    return guarded([&] {
        need(surface && (xy || !n), "surface and xy are required");
        surface->surface->pointsAt(xy, n, xyz, normals, columnTangents, rowTangents);
        return VCR_OK;
    });
}

vcr_status vcr_surface_resample(const vcr_surface* surface, const vcr_raster* raster, float* xyz, float* normals)
{
    return guarded([&] {
        need(surface && raster, "surface and raster are required");
        need(raster->width >= 0 && raster->height >= 0, "negative raster size");
        surface->surface->resample(toRaster(*raster), xyz, normals);
        return VCR_OK;
    });
}

float vcr_surface_pixels_per_voxel(const vcr_surface* surface, const vcr_raster* raster)
{
    if (!surface || !raster)
        return 0.0f;
    try {
        return vcr::grid::pixelsPerVoxel(*surface->surface, toRaster(*raster));
    } catch (const std::exception& e) {
        lastError = e.what();
        return 0.0f;
    }
}

void vcr_composite_init(vcr_composite* c, int method)
{
    if (!c)
        return;
    const vcr::composite::Options o;
    c->method = method;
    c->iso_cutoff = o.isoCutoff;
    c->alpha_min = o.alphaMin;
    c->alpha_max = o.alphaMax;
    c->alpha_opacity = o.alphaOpacity;
    c->alpha_cutoff = o.alphaCutoff;
    c->extinction = o.extinction;
    c->emission = o.emission;
    c->ambient = o.ambient;
}

void vcr_request_init(vcr_request* r)
{
    if (!r)
        return;
    *r = {};
    r->struct_size = sizeof(*r);
    r->order = 1;
}

void vcr_result_init(vcr_result* r)
{
    if (!r)
        return;
    *r = {};
    r->struct_size = sizeof(*r);
}

std::size_t vcr_request_value_count(const vcr_frame* frame, const vcr_request* request)
{
    try {
        return request ? valueCount(toFrame(frame), request) : 0;
    } catch (const std::exception& e) {
        lastError = e.what();
        return 0;
    }
}

vcr_status vcr_sample(const vcr_frame* frame, const vcr_request* request, vcr_result* result)
{
    return sampleWith(frame, request, result,
                      [](const vcr::grid::Frame& f, const vcr::grid::Request& r) { return vcr::grid::sample(f, r); });
}

vcr_status vcr_sample_cached(const vcr_frame* frame, const vcr_request* request, vcr_pending* pending,
                             vcr_result* result)
{
    return sampleWith(frame, request, result, [&](const vcr::grid::Frame& f, vcr::grid::Request r) {
        if (!pending)
            return vcr::grid::sampleCached(f, std::move(r), nullptr);
        const std::size_t reads = pending->pending.coarse.size() + pending->pending.target.size();
        auto out = vcr::grid::sampleCached(f, std::move(r), &pending->pending);
        const auto& volume = request->volume->volume;
        if (pending->pending.coarse.size() + pending->pending.target.size() > reads &&
            std::find(pending->volumes.begin(), pending->volumes.end(), volume) == pending->volumes.end())
            pending->volumes.push_back(volume);
        return out;
    });
}

vcr_status vcr_pending_create(vcr_pending** out)
{
    return guarded([&] {
        need(out != nullptr, "out is NULL");
        *out = new vcr_pending{};
        return VCR_OK;
    });
}

int vcr_pending_empty(const vcr_pending* pending) { return !pending || pending->pending.empty(); }

void vcr_pending_clear(vcr_pending* pending)
{
    if (pending) {
        pending->pending = {};
        pending->volumes.clear();
    }
}

vcr_status vcr_pending_load(const vcr_pending* pending, vcr_part_fn afterPart, void* user)
{
    return guarded([&] {
        need(pending != nullptr, "pending is NULL");
        const auto failure = vcr::grid::load(pending->pending, [&] { return !afterPart || afterPart(user) != 0; });
        if (!failure.empty()) {
            for (const auto& volume : pending->volumes)
                volume->renderOnCpu();
            lastError = failure;
        }
        return VCR_OK;
    });
}

void vcr_pending_release(vcr_pending* pending) { delete pending; }

void vcr_to_uint8(const float* input, std::size_t n, std::uint8_t* output, std::uint8_t* coverage)
{
    vcr::display::toUint8(input, n, output, coverage);
}

namespace {

std::array<std::uint32_t, 256> table(const std::uint32_t* values)
{
    std::array<std::uint32_t, 256> out{};
    std::copy_n(values, 256, out.begin());
    return out;
}

}  // namespace

void vcr_window_level_lut(std::uint32_t lut[256], float low, float high)
{
    std::array<std::uint32_t, 256> out;
    vcr::display::windowLevelLut(out, low, high);
    std::copy(out.begin(), out.end(), lut);
}

void vcr_palette_lut(std::uint32_t lut[256], const std::uint32_t palette[256], float low, float high, int blackZero)
{
    std::array<std::uint32_t, 256> out;
    vcr::display::paletteLut(out, table(palette), low, high, blackZero != 0);
    std::copy(out.begin(), out.end(), lut);
}

void vcr_overlay_lut(std::uint32_t lut[256], const std::uint32_t color[256], float low, float high,
                     int intensityAsAlpha, int premultiplied)
{
    std::array<std::uint32_t, 256> out;
    vcr::display::overlayLut(out, table(color), low, high, intensityAsAlpha != 0, premultiplied != 0);
    std::copy(out.begin(), out.end(), lut);
}

void vcr_apply_lut(const std::uint8_t* input, std::size_t rows, std::size_t cols, std::size_t inputStride,
                   const std::uint32_t lut[256], std::uint32_t* output, std::size_t outputStride)
{
    vcr::display::applyLut(input, rows, cols, inputStride, lut, output, outputStride);
}

void vcr_blend(const std::uint32_t* base, const std::uint32_t* overlay, std::size_t n, float opacity,
               std::uint32_t* output)
{
    vcr::display::blend(base, overlay, n, opacity, output);
}

vcr_status vcr_composite_reduce(const float* values, std::size_t layers, std::size_t n, const vcr_composite* composite,
                                float* output)
{
    return guarded([&] {
        need(values && composite && output, "values, composite and output are required");
        vcr::composite::reduce(values, layers, n, toComposite(*composite), output);
        return VCR_OK;
    });
}

void vcr_opacity_lut(float lut[256], float low, float high, float opacity, float gamma, std::uint8_t cutoff,
                     float segmentLength)
{
    const auto out = vcr::display::opacityLut(low, high, opacity, gamma, cutoff, segmentLength);
    std::copy(out.begin(), out.end(), lut);
}

vcr_status vcr_composite_slab(const vcr_slab* slab, std::uint8_t* rgb, std::uint8_t* coverage)
{
    return guarded([&] {
        need(slab && rgb && coverage && slab->colors && slab->opacity, "slab, colors, opacity and outputs are required");
        need(slab->count >= 0 && (slab->layers || !slab->count), "slab layers are required");
        std::vector<vcr::display::SlabLayer> layers(std::size_t(slab->count));
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& l = slab->layers[i];
            layers[i] = {l.values, l.coverage, l.value_stride, l.coverage_stride};
        }
        vcr::display::Slab s;
        s.layers = layers.data();
        s.count = slab->count;
        s.rows = slab->rows;
        s.cols = slab->cols;
        s.left = slab->left;
        s.top = slab->top;
        s.width = slab->width;
        s.height = slab->height;
        const auto& p = slab->projection;
        auto& q = s.projection;
        q.perspective = p.perspective != 0;
        q.m00 = p.m00;
        q.m01 = p.m01;
        q.m10 = p.m10;
        q.m11 = p.m11;
        q.layerOffsets = p.layer_offsets;
        std::copy_n(p.position, 3, q.pos);
        std::copy_n(p.ray_base, 3, q.rayBase);
        std::copy_n(p.e1_over_f, 3, q.e1OverF);
        std::copy_n(p.e2_over_f, 3, q.e2OverF);
        q.centerU = p.center_u;
        q.centerV = p.center_v;
        q.layerNum = p.layer_num;
        s.colors = slab->colors;
        s.opacity = slab->opacity;
        s.lighting = slab->lighting;
        vcr::display::compositeSlab(s, rgb, coverage);
        return VCR_OK;
    });
}

}  // extern "C"
