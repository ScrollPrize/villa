// Registered coordinate grids and bounded surface operations.
#include "aligned_internal.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "vc_reader/grid.hpp"
#include "vc_reader/tifxyz.hpp"
#include "../element_type.hpp"
#include "../reader_types.hpp"
#include "../rounding.hpp"

namespace vcr::sources {
using nlohmann::json;
namespace {
using Clock = std::chrono::steady_clock;
using Pair = std::array<std::int64_t, 2>;
[[noreturn]] void bad(const std::string& message) { throw std::invalid_argument(message); }
double since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
std::string matrixKey(const Affine& a)
{
    std::string out;
    for (const double v : a.values) out += std::to_string(std::bit_cast<std::uint64_t>(v)) + ",";
    return out;
}
std::size_t itemOf(const std::string& dtype) { return std::size_t(dtype[2] - '0'); }
std::array<std::string, 2> surfaceFrame(const Surface& surface) { return {surface.sampleId, surface.volumeId}; }
const SurfaceRegistration* surfaceOf(const std::optional<SourceRegistration>& registration)
{
    return registration ? std::get_if<SurfaceRegistration>(&*registration) : nullptr;
}
Pair checkPatch(const Pair& start, const Pair& size)
{
    for (const auto n : size)
        if (n <= 0 || n > (1 << 24)) bad("surface start/size must be YX pairs with positive sizes <= 2**24");
    for (const auto n : start)
        if (std::abs(n) > (1LL << 40)) bad("surface start is outside supported coordinate range");
    return start;
}
double checkDistance(double value, const std::string& name)
{
    if (!std::isfinite(value) || value < 0) bad(name + " must be finite and nonnegative");
    return value;
}
Affine shift(double x, double y, double z)
{
    Affine out;
    out(0, 3) = x, out(1, 3) = y, out(2, 3) = z;
    return out;
}
Affine scaled(const Vec3& values)
{
    Affine out;
    for (int i = 0; i < 3; ++i) out(i, i) = values[std::size_t(i)];
    return out;
}
// Splits the small transformed origin before bulk float32 arithmetic.
struct Local {
    std::vector<float> points;
    Triple base{};
};
Local transformPoints(const float* points, std::size_t n, const Affine& mapping, const std::array<double, 3>& origin = {0, 0, 0})
{
    const auto at = mapping.apply(origin);
    Local out;
    float local[3];
    for (int a = 0; a < 3; ++a) {
        out.base[std::size_t(a)] = std::int64_t(std::floor(at[std::size_t(a)]));
        local[a] = float(at[std::size_t(a)] - double(out.base[std::size_t(a)]));
    }
    float m[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) m[r * 3 + c] = float(mapping(r, c));
    out.points.resize(n * 3);
    for (std::size_t i = 0; i < n; ++i) {
        const float* p = points + i * 3;
        for (int r = 0; r < 3; ++r) out.points[i * 3 + std::size_t(r)] = p[0] * m[r * 3] + p[1] * m[r * 3 + 1] + p[2] * m[r * 3 + 2] + local[r];
    }
    return out;
}
// ZYX local float32 pixel points of XY uv pairs.
Local pixelPoints(const float* uv, std::size_t n, const Affine& mapping, const std::array<double, 3>& origin = {0, 0, 0})
{
    std::vector<float> xyz(n * 3, 0.f);
    for (std::size_t i = 0; i < n; ++i) {
        xyz[i * 3] = uv[i * 2];
        xyz[i * 3 + 1] = uv[i * 2 + 1];
    }
    auto local = transformPoints(xyz.data(), n, mapping, origin);
    for (std::size_t i = 0; i < n; ++i) std::swap(local.points[i * 3], local.points[i * 3 + 2]);
    std::swap(local.base[0], local.base[2]);
    return local;
}
std::vector<float> patchUv(const Pair& size)
{
    std::vector<float> uv(std::size_t(size[0] * size[1]) * 2);
    std::size_t at = 0;
    for (std::int64_t y = 0; y < size[0]; ++y)
        for (std::int64_t x = 0; x < size[1]; ++x) {
            uv[at++] = float(x);
            uv[at++] = float(y);
        }
    return uv;
}
Block hostBlock(const SampledArray& array)
{
    Block block;
    block.shape = array.shape;
    block.dtype = array.dtype;
    block.data = array.host;
    block.nbytes = std::size_t(array.count()) * itemOf(array.dtype);
    return block;
}
} // namespace

// ---- geometry ----
std::optional<Target> Geometry::auxiliary(const std::optional<Shared<Source>>& source)
{
    if (!source) return std::nullopt;
    auto r = owner.open(**source);
    const auto* reg = surfaceOf(r->registration());
    if (!reg || reg->surface.id != surface.id || !reg->pixelToUv) bad("geometry masks/normals require explicit registration to this surface");
    auto target = owner.samplingTarget(**source, r, VolumeTransform{pixelAffine(*reg->pixelToUv).forward.inverse(), {}}, true, true);
    if ((*source)->role == "normal") target.order = 1;
    return target;
}

Geometry::Geometry(AlignedEngine::Impl& o, const Surface& s) : owner(o), surface(s), key(encode(s).dump(-1, ' ', false, json::error_handler_t::replace))
{
    if (!surface.xyzToVolume || !surface.componentOrder) bad(surface.id + ": unknown geometry coordinate frame/components");
    auto grid = surface.pixelToUv;
    if (surface.geometry.format == "tifxyz") {
        if (!surface.geometry.location) bad("surface geometry location is unresolved");
        // The first of the location and its alternatives that holds the directory.
        std::vector<std::string> candidates{*surface.geometry.location};
        candidates.insert(candidates.end(), surface.geometry.alternatives.begin(), surface.geometry.alternatives.end());
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            try {
                tifxyz = std::make_unique<tifxyz::Reader>(candidates[i], surface.tifxyzMask, owner.options.reader);
                break;
            } catch (const zarr::NotFound&) {
                if (i + 1 == candidates.size()) throw;
            } catch (const CacheMiss&) {
                if (i + 1 == candidates.size()) throw;
            }
        }
        const auto& scale = tifxyz->metadata().scale;  // rows, columns
        const PixelAffine observed{{{1 / scale[1], 0, 0}, {0, 1 / scale[0], 0}}};
        if (grid && *grid != observed) bad("surface pixel_to_uv contradicts tifxyz scale");
        shape = tifxyz->shape();
        // Full-resolution pixels sit at stored cells scale * pixel, exactly as tifxyz::fullRaster places them.
        Affine from;
        from(0, 0) = scale[1];
        from(1, 1) = scale[0];
        fromUv = VolumeTransform{from, {}};
        toUv = VolumeTransform{from.inverse(), {}};
    } else {
        if (!grid) bad("coordinate grid requires explicit pixel_to_uv");
        Source source;
        source.id = surface.id + "/coordinates";
        source.storage = surface.geometry;
        source.role = "scalar";
        source.axes = surface.axes;
        coordinates = owner.open(finished(std::move(source)));
        const auto& axes = coordinates->axes();
        const auto& sizes = coordinates->shape();
        if (coordinates->spatialAxes() != std::vector<std::string>{"y", "x"}) bad("surface coordinates require YX grids");
        for (std::size_t i = 0; i < axes.size(); ++i) {
            if (axes[i] == "y") shape[0] = sizes[i];
            if (axes[i] == "x") shape[1] = sizes[i];
        }
        const auto at = std::find(axes.begin(), axes.end(), "c");
        if (at == axes.end() || sizes[std::size_t(at - axes.begin())] != 3) bad("coordinate grid must contain three components");
        toUv = pixelAffine(*grid);
        fromUv = VolumeTransform{toUv.forward.inverse(), {}};
    }
    // A mirrored UV grid turns the patch normals around.
    const auto& m = toUv.forward;
    flipNormals = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0) < 0;
    explicitMask = auxiliary(surface.mask);
    normals = auxiliary(surface.normals);
    if (normals) {
        const auto& source = **surface.normals;
        if (source.role != "normal" || !source.valueFrame || source.valueFrame->kind != "volume" || !source.componentOrder)
            bad("surface normals require a declared volume value frame and XYZ components");
        const auto& value = *source.valueFrame;
        auto mapping = owner.valuePlacement(value).forward;
        if (!surface.spacingUm) bad("normal values require surface physical spacing");
        if (*value.sampleId != surface.sampleId || *value.volumeId != surface.volumeId) {
            if (owner.referenceFrame[0].empty()) bad("cross-volume normals need a volume-aligned reader with a known frame relationship");
            const auto from = owner.between(source, surfaceFrame(surface)).forward;
            const auto values = owner.between(source, {*value.sampleId, *value.volumeId}).forward;
            mapping = from.inverse() * values * mapping;
        }
        const auto inverse = (scaled(*surface.spacingUm) * surface.xyzToVolume->forward.inverse() * mapping).inverse();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) normalTransform[std::size_t(r * 3 + c)] = float(inverse(c, r));
    }
}

std::shared_ptr<const GeometryTile> Geometry::tile(const Pair& start, const Pair& size)
{
    const auto cacheKey = "geometry|" + key + "|" + std::to_string(start[0]) + "," + std::to_string(start[1]) + "|" + std::to_string(size[0]) + "," + std::to_string(size[1]);
    if (const auto* cached = owner.tiles.get(cacheKey)) return std::static_pointer_cast<const GeometryTile>(*cached);
    const auto cells = std::size_t(size[0] * size[1]);
    if (cells * 80 > owner.workBytes) throw DecodeBudgetError("geometry tile exceeds sampling_work_bytes");
    owner.largestGeometryCells = std::max(owner.largestGeometryCells, std::size_t(std::max(size[0], size[1])));
    auto out = std::make_shared<GeometryTile>();
    out->rows = size[0];
    out->cols = size[1];
    if (tifxyz) {
        auto stored = tifxyz->window({start[0], start[1], size[0], size[1]});
        owner.geometryReadBytes += stored.storedBytes;
        out->xyz = std::move(stored.xyz);
        out->valid = std::move(stored.valid);
    } else {
        // Channels in XYZ order; a vertex with a non-finite coordinate is invalid.
        const auto block = convertBlock(owner.canonical(coordinates->read({start[0], start[1]}, {size[0], size[1]}), coordinates->axes()), "<f4");
        owner.geometryReadBytes += block.nbytes;
        const auto* values = reinterpret_cast<const float*>(block.data.get());
        const std::string letters = "xyz";
        out->xyz.resize(cells * 3);
        for (std::size_t component = 0; component < 3; ++component) {
            const auto channel = std::size_t(std::find(surface.componentOrder->begin(), surface.componentOrder->end(), std::string(1, letters[component])) - surface.componentOrder->begin());
            for (std::size_t i = 0; i < cells; ++i) out->xyz[i * 3 + component] = values[channel * cells + i];
        }
        out->valid.assign(cells, 1);
        for (std::size_t i = 0; i < cells; ++i)
            for (int a = 0; a < 3; ++a)
                if (!std::isfinite(out->xyz[i * 3 + std::size_t(a)])) out->valid[i] = 0;
    }
    if (explicitMask || normals) {
        std::vector<float> coords(cells * 3, 0.f);
        std::size_t at = 0;
        for (std::int64_t y = 0; y < size[0]; ++y)
            for (std::int64_t x = 0; x < size[1]; ++x, at += 3) {
                coords[at] = float(x);
                coords[at + 1] = float(y);
            }
        const auto uv = transformPoints(coords.data(), cells, toUv.forward, {double(start[1]), double(start[0]), 0});
        const auto sampleOn = [&](const Target& target) {
            auto local = transformPoints(uv.points.data(), cells, target.mapping.forward, {double(uv.base[0]), double(uv.base[1]), double(uv.base[2])});
            for (std::size_t i = 0; i < cells; ++i) std::swap(local.points[i * 3], local.points[i * 3 + 2]);
            std::swap(local.base[0], local.base[2]);
            auto [values, support] = owner.samplePoints(target, local.points, local.base);
#ifdef VC_READER_CUDA
            if (owner.backend == "cuda") {
                const auto download = [&](SampledArray& array) {
                    auto storage = std::static_pointer_cast<device::Buffer>(array.device);
                    Block host = Block::allocate(array.shape, array.dtype);
                    device::Guard guard(storage->device);
                    if (host.nbytes) device::check(cudaMemcpyAsync(host.data.get(), storage->ptr, host.nbytes, cudaMemcpyDeviceToHost, device::stream()));
                    device::complete();
                    array.host = host.data;
                    array.device.reset();
                };
                download(values);
                download(support);
            }
#endif
            return std::pair{std::move(values), std::move(support)};
        };
        if (explicitMask) {
            auto [values, support] = sampleOn(*explicitMask);
            if (values.shape[0] != 1) bad("geometry mask must have one channel");
            const auto item = itemOf(values.dtype);
            for (std::size_t i = 0; i < cells; ++i) {
                const auto* p = values.host.get() + i * item;
                if (!support.host[i] || std::all_of(p, p + item, [](std::uint8_t b) { return b == 0; })) out->valid[i] = 0;
            }
        }
        if (normals) {
            auto [values, support] = sampleOn(*normals);
            for (std::size_t i = 0; i < cells; ++i)
                if (!support.host[i]) out->valid[i] = 0;
            const auto& components = *(*surface.normals)->componentOrder;
            const std::string letters = "xyz";
            int order3[3];
            for (int i = 0; i < 3; ++i) order3[i] = int(std::find(components.begin(), components.end(), std::string(1, letters[std::size_t(i)])) - components.begin());
            const auto* data = reinterpret_cast<const float*>(values.host.get());
            out->normals.resize(cells * 3);
            for (std::size_t i = 0; i < cells; ++i) {
                const float v[3] = {data[std::size_t(order3[0]) * cells + i], data[std::size_t(order3[1]) * cells + i], data[std::size_t(order3[2]) * cells + i]};
                for (int a = 0; a < 3; ++a) out->normals[i * 3 + std::size_t(a)] = v[0] * normalTransform[std::size_t(a * 3)] + v[1] * normalTransform[std::size_t(a * 3 + 1)] + v[2] * normalTransform[std::size_t(a * 3 + 2)];
            }
        }
    }
    owner.tiles.put(cacheKey, out, out->xyz.size() * 4 + out->valid.size() + out->normals.size() * 4);
    return out;
}

std::shared_ptr<Geometry> geometryOf(AlignedEngine::Impl& impl, const Surface& surface)
{
    const auto key = encode(surface).dump(-1, ' ', false, json::error_handler_t::replace);
    auto& slot = impl.geometries[key];
    if (!slot) slot = std::make_shared<Geometry>(impl, surface);
    return slot;
}

Target prepareSurfaceTarget(AlignedEngine::Impl& impl, const Source& source, std::shared_ptr<SourceReader> opened)
{
    const auto* reg = surfaceOf(opened->registration());
    if (opened->spatialAxes() != std::vector<std::string>{"y", "x"} || !reg->pixelToUv)
        bad(source.id + ": surface targets require YX axes and explicit pixel_to_uv registration");
    if (source.role != "label") bad("surface targets must be categorical labels");
    auto target = impl.samplingTarget(source, opened, VolumeTransform{pixelAffine(*reg->pixelToUv).forward.inverse(), {}}, true, true);
    (void)geometryOf(impl, reg->surface);  // Fail unknown frames while opening aligned data.
    (void)impl.between(source, surfaceFrame(reg->surface));
    for (const auto& companion : source.companions) {
        auto child = opened->companion(companion.name);
        const auto* childReg = surfaceOf(child->registration());
        if (!childReg || childReg->surface.id != reg->surface.id || !childReg->pixelToUv) bad("surface companions require an explicit shared surface registration");
        target.companions.emplace_back(companion.name, impl.samplingTarget(*companion.source, child, VolumeTransform{pixelAffine(*childReg->pixelToUv).forward.inverse(), {}}, true, true));
    }
    return target;
}

// ---- shared ----
namespace {
const Surface& registeredSurface(const Target& target) { return surfaceOf(target.reader->registration())->surface; }
void requireMatching(AlignedEngine::Impl& s, const Surface& surface)
{
    for (const auto& [name, target] : s.surfaceTargets)
        if (!(registeredSurface(target) == surface)) bad("all selected surface targets must register to the requested surface descriptor");
}
void checkBudget(AlignedEngine::Impl& s, const Triple& shape, const std::optional<Pair>& surfaceShape = std::nullopt)
{
    const auto size = std::size_t(shape[0] * shape[1] * shape[2]);
    std::size_t estimate = 0;
    const auto floatItem = [](const Target& t) { return std::max<std::size_t>(4, itemOf(t.dtype)); };
    for (const auto& [name, target] : s.targets) {
        estimate += size * (std::size_t(target.channels) * floatItem(target) + 1);
        for (const auto& [key, child] : target.companions) estimate += size * std::size_t(child.channels) * floatItem(child);
    }
    for (const auto& [name, target] : s.surfaceTargets) {
        const auto count = surfaceShape ? std::size_t((*surfaceShape)[0] * (*surfaceShape)[1]) : size;
        estimate += count * (std::size_t(target.channels) * itemOf(target.dtype) + 3);
        for (const auto& [key, child] : target.companions) estimate += count * std::size_t(child.channels) * floatItem(child);
    }
    if (estimate > s.options.surfaceOutputBytes)
        throw DecodeBudgetError("derived output shape (" + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2]) + ") requires approximately " +
                                std::to_string(estimate) + " bytes; increase surface_output_bytes or use iter_project_surface");
}
// A UV patch as a raster over the stored grid, with the vertex window it reads. None when
// the patch misses the grid. A grid position's patch is the cell at its floor, so the window
// holds every cell the patch can reach.
struct Patch {
    std::shared_ptr<const GeometryTile> tile;
    grid::Raster raster;
};
std::optional<Patch> patchOf(Geometry& geom, const Pair& start, const Pair& size)
{
    const auto& m = geom.fromUv.forward;
    const double u = double(start[1]), v = double(start[0]);
    const double sx = m(0, 0), kx = m(0, 1), ky = m(1, 0), sy = m(1, 1);
    const double x0 = sx * u + kx * v + m(0, 3), y0 = ky * u + sy * v + m(1, 3);
    double minX = INFINITY, maxX = -INFINITY, minY = INFINITY, maxY = -INFINITY;
    for (const std::int64_t i : {std::int64_t(0), size[1] - 1})
        for (const std::int64_t j : {std::int64_t(0), size[0] - 1}) {
            const double x = x0 + double(i) * sx + double(j) * kx, y = y0 + double(i) * ky + double(j) * sy;
            minX = std::min(minX, x), maxX = std::max(maxX, x);
            minY = std::min(minY, y), maxY = std::max(maxY, y);
        }
    const std::int64_t rows = geom.shape[0], cols = geom.shape[1];
    const auto c0 = std::max<std::int64_t>(0, std::int64_t(std::floor(minX))), c1 = std::min<std::int64_t>(cols, std::int64_t(std::floor(maxX)) + 2);
    const auto r0 = std::max<std::int64_t>(0, std::int64_t(std::floor(minY))), r1 = std::min<std::int64_t>(rows, std::int64_t(std::floor(maxY)) + 2);
    if (c1 - c0 < 2 || r1 - r0 < 2) return std::nullopt;
    Patch patch;
    patch.tile = geom.tile({r0, c0}, {r1 - r0, c1 - c0});
    patch.raster = grid::Raster{x0 - double(c0), y0 - double(r0), sx, sy, int(size[1]), int(size[0]), kx, ky};
    return patch;
}
// The vcr::grid surface over a tile's vertices: normals in the surface's physical space.
std::unique_ptr<grid::Surface> gridSurface(const Geometry& geom, const GeometryTile& tile, grid::Interpolation interpolation)
{
    grid::SurfaceOptions options;
    options.interpolation = interpolation;
    options.flipNormals = geom.flipNormals;
    if (geom.surface.spacingUm) {
        const auto& spacing = *geom.surface.spacingUm;
        options.spacing = {float(spacing[0]), float(spacing[1]), float(spacing[2])};
    }
    return std::make_unique<grid::Surface>(grid::Vertices{tile.xyz.data(), int(tile.rows), int(tile.cols), std::size_t(tile.cols) * 3, tile.normals.empty() ? nullptr : tile.normals.data(), tile.valid.data()}, options);
}
TargetResult renderVolume(AlignedEngine::Impl& s, Target& target, const grid::Surface* surface, const grid::Raster& raster, const Pair& size, const std::vector<float>& layers, const Affine& toReference)
{
    const std::vector<std::string> zyx{"z", "y", "x"};
    Array* array = target.reader->directArray();
    if (!array || target.reader->spatialAxes() != zyx || target.field) bad(target.source.id + ": render_surface samples single-channel 3-D Zarr volumes");
    const auto& dtype = target.reader->zarrArray()->meta.dtype;
    if (!target.order && itemOf(dtype) > 2)
        throw DeclarationTypeError(target.source.id + ": render_surface samples labels through float32, exact for integers up to 16 bits");
    if (s.backend == "cuda" && dtype != "|u1") throw DeclarationTypeError(target.source.id + ": CUDA render_surface samples uint8 volumes");
    const auto n = std::size_t(size[0] * size[1]);
    std::vector<float> values(layers.size() * n, std::numeric_limits<float>::quiet_NaN());
    std::vector<std::int8_t> level(n, std::int8_t(-2));
    if (surface) {
        const auto mapping = target.mapping.forward * toReference;
        std::array<double, 12> map{};
        for (int a = 0; a < 3; ++a)
            for (int c = 0; c < 4; ++c) map[std::size_t(a * 4 + c)] = mapping(2 - a, c);
        grid::Request request;
        request.levels = {grid::levelFromMap(*array, elementOf(dtype), map)};
        request.order = target.order;
        request.offsets = layers;
        grid::Result result;
#ifdef VC_READER_CUDA
        if (s.backend == "cuda") {
            if (!target.cuda) target.cuda = std::make_shared<reader_cuda::Reader>(*array, s.tiles.limit, s.device);
            result = grid::cuda::sample(*surface, raster, request, {static_cast<reader_cuda::Reader*>(target.cuda.get())});
        } else
#endif
            result = grid::sample(*surface, raster, request);
        std::copy(result.values.begin(), result.values.end(), values.begin());
        std::copy(result.level.begin(), result.level.end(), level.begin());
    }
    const std::vector<std::int64_t> shape{1, std::int64_t(layers.size()), size[0], size[1]};
    Block valid = Block::allocate(shape, "|b1");
    for (std::size_t k = 0; k < layers.size(); ++k)
        for (std::size_t i = 0; i < n; ++i) valid.data[k * n + i] = level[i] >= 0;
    Block data = Block::allocate(shape, "<f4");
    std::memcpy(data.data.get(), values.data(), values.size() * 4);
    if (!target.order) {
        const float fill = float(target.reader->fillValue().get<double>());
        auto* floats = reinterpret_cast<float*>(data.data.get());
        for (std::size_t i = 0; i < values.size(); ++i)
            if (!valid.data[i]) floats[i] = fill;
        data = convertBlock(data, s.backend == "cpu" ? target.dtype : target.nativeDtype);
    }
    TargetResult out;
    out.data = s.fromHost(data);
    out.valid = s.fromHost(valid);
    return out;
}
} // namespace

SampleResult SampleReader::renderSurface(const Surface& surface, const Pair& start, const Pair& size, const std::vector<double>& offsetsUm, grid::Interpolation interpolation)
{
    auto& s = *impl_;
    std::lock_guard lock(s.mutex);
    s.checkOpen();
    checkPatch(start, size);
    if (offsetsUm.empty() || std::any_of(offsetsUm.begin(), offsetsUm.end(), [](double v) { return !std::isfinite(v); })) bad("offsets_um must be nonempty and finite");
    auto geom = geometryOf(s, surface);
    s.usedSurface = true;
    requireMatching(s, surface);
    checkBudget(s, {std::int64_t(offsetsUm.size()), size[0], size[1]}, size);
    if (!surface.spacingUm) bad("surface physical spacing is unknown");
    const Source& relating = s.surfaceTargets.empty() ? s.reference : s.surfaceTargets.front().second.source;
    const auto between = s.between(relating, surfaceFrame(surface));
    // Vertex coordinates to reference array coordinates; normals and layer offsets are
    // taken in the surface's physical space.
    const auto toReference = s.referenceMap.forward.inverse() * between.forward * surface.xyzToVolume->forward;
    SampleResult result;
    result.metadata = s.metadata.is_null() ? json::object() : s.metadata;
    result.metadata["surface"] = {{"id", surface.id}, {"uv_start", start}, {"uv_size", size}, {"offsets_um", offsetsUm},
                                  {"coordinate_frame", surfaceFrame(surface)}, {"reference_frame", s.referenceFrame}, {"array_to_level0", s.referenceMap.forward.json()}};
    const auto patch = patchOf(*geom, start, size);
    const auto sheet = patch ? gridSurface(*geom, *patch->tile, interpolation) : nullptr;
    const grid::Raster raster = patch ? patch->raster : grid::Raster{};
    const std::vector<float> layers(offsetsUm.begin(), offsetsUm.end());
    for (auto& [name, target] : s.targets) {
        auto out = renderVolume(s, target, sheet.get(), raster, size, layers, toReference);
        for (auto& [key, child] : target.companions) out.masks.emplace_back(key, renderVolume(s, child, sheet.get(), raster, size, layers, toReference).data);
        result.targets.emplace_back(name, std::move(out));
    }
    // Surface-registered targets are read at the patch's own UV pixels.
    const auto uv = patchUv(size);
    const auto count = uv.size() / 2;
    const std::array<double, 3> uvOrigin{double(start[1]), double(start[0]), 0};
    for (const auto& [name, target] : s.surfaceTargets) {
        const auto local = pixelPoints(uv.data(), count, target.mapping.forward, uvOrigin);
        auto [values, valid] = s.samplePoints(target, local.points, local.base);
        TargetResult out;
        out.data = values.reshaped({target.channels, size[0], size[1]});
        out.valid = valid.reshaped({1, size[0], size[1]});
        for (const auto& [key, child] : target.companions) {
            const auto p = pixelPoints(uv.data(), count, child.mapping.forward, uvOrigin);
            auto [maskValues, unused] = s.samplePoints(child, p.points, p.base);
            (void)unused;
            out.masks.emplace_back(key, maskValues.reshaped({child.channels, size[0], size[1]}));
        }
        result.targets.emplace_back(name, std::move(out));
    }
    return result;
}

Surface SampleReader::surfaceNamed(const std::string& id) const
{
    std::optional<Surface> found;
    for (const auto& [name, target] : impl_->surfaceTargets) {
        const auto& surface = registeredSurface(target);
        if (surface.id != id) continue;
        if (found && !(*found == surface)) bad("select an explicit registered Surface; geometry relationship is missing or ambiguous");
        found = surface;
    }
    if (!found) bad("select an explicit registered Surface; geometry relationship is missing or ambiguous");
    return *found;
}

// ---- projection ----
namespace {
// Bounded BVH of tile metadata; decoded geometry stays in the tile cache.
using Bounds = std::array<std::array<double, 3>, 2>;
struct TileRef {
    Pair start, size;
};
struct SurfaceIndex {
    Bounds bounds{};
    std::vector<std::pair<TileRef, Bounds>> entries;
    std::vector<std::unique_ptr<SurfaceIndex>> children;
    explicit SurfaceIndex(std::vector<std::pair<TileRef, Bounds>> all)
    {
        if (all.empty()) return;
        bounds = all[0].second;
        for (const auto& [tile, b] : all)
            for (int a = 0; a < 3; ++a) {
                bounds[0][std::size_t(a)] = std::min(bounds[0][std::size_t(a)], b[0][std::size_t(a)]);
                bounds[1][std::size_t(a)] = std::max(bounds[1][std::size_t(a)], b[1][std::size_t(a)]);
            }
        if (all.size() <= 8) {
            entries = std::move(all);
            return;
        }
        std::size_t axis = 0;
        for (std::size_t a = 1; a < 3; ++a)
            if (bounds[1][a] - bounds[0][a] > bounds[1][axis] - bounds[0][axis]) axis = a;
        std::stable_sort(all.begin(), all.end(), [axis](const auto& x, const auto& y) { return x.second[0][axis] + x.second[1][axis] < y.second[0][axis] + y.second[1][axis]; });
        const auto middle = all.size() / 2;
        children.push_back(std::make_unique<SurfaceIndex>(std::vector(all.begin(), all.begin() + std::ptrdiff_t(middle))));
        children.push_back(std::make_unique<SurfaceIndex>(std::vector(all.begin() + std::ptrdiff_t(middle), all.end())));
    }
    template <class F>
    void query(const std::array<double, 3>& lower, const std::array<double, 3>& upper, const std::array<double, 3>& expansion, F&& f) const
    {
        const auto overlaps = [&](const Bounds& b) {
            for (std::size_t a = 0; a < 3; ++a)
                if (!(b[1][a] + expansion[a] + .5 >= lower[a] && b[0][a] - expansion[a] - .5 <= upper[a])) return false;
            return true;
        };
        if (!overlaps(bounds)) return;
        for (const auto& [tile, b] : entries)
            if (overlaps(b)) f(tile);
        for (const auto& child : children) child->query(lower, upper, expansion, f);
    }
};
// The surface points of one leaf of the grid: positions in reference array coordinates
// relative to `base`, per point the move per micrometre along the unit normal, the UV
// relative to the selecting patch's origin, and an id ordering the points by grid position.
struct Samples {
    std::vector<float> xyz, step;  // n x 3
    std::vector<float> uv;         // n x 2
    std::vector<std::int64_t> ids;
    std::optional<Bounds> bounds;  // of the points, absolute
    Triple base{};
    [[nodiscard]] std::size_t count() const { return ids.size(); }
    [[nodiscard]] std::size_t bytes() const { return (xyz.size() + step.size() + uv.size()) * 4 + ids.size() * 8 + 128; }
};
// Points per grid cell along one direction whose spacing in reference voxels is `extent`: a
// power of two, at most kSubdivisions, that puts them half a voxel apart or closer.
constexpr int kSubdivisions = 64;
int subdivisions(double extent)
{
    int n = 1;
    while (n < kSubdivisions && double(n) * .5 < extent) n *= 2;
    return n;
}
std::vector<std::pair<Triple, Triple>> tileRegions(const Triple& origin, const Triple& shape, const Triple& tile)
{
    std::vector<std::pair<Triple, Triple>> out;
    for (std::int64_t z = 0; z < shape[0]; z += tile[0])
        for (std::int64_t y = 0; y < shape[1]; y += tile[1])
            for (std::int64_t x = 0; x < shape[2]; x += tile[2])
                out.push_back({Triple{origin[0] + z, origin[1] + y, origin[2] + x},
                               Triple{std::min(tile[0], shape[0] - z), std::min(tile[1], shape[1] - y), std::min(tile[2], shape[2] - x)}});
    return out;
}
// The vertices of a geometry tile in reference array coordinates about `anchor` (floor of
// the first valid vertex); none when no vertex is valid.
std::optional<std::pair<Local, std::array<double, 3>>> movedVertices(const GeometryTile& tile, const Affine& destination)
{
    const auto firstValid = std::find(tile.valid.begin(), tile.valid.end(), std::uint8_t(1));
    if (firstValid == tile.valid.end()) return std::nullopt;
    const auto first = std::size_t(firstValid - tile.valid.begin());
    const std::array<double, 3> anchor{std::floor(double(tile.xyz[first * 3])), std::floor(double(tile.xyz[first * 3 + 1])), std::floor(double(tile.xyz[first * 3 + 2]))};
    std::vector<float> shifted(tile.xyz.size(), 0.f);
    for (std::size_t i = 0; i < tile.valid.size(); ++i)
        if (tile.valid[i])
            for (std::size_t a = 0; a < 3; ++a) shifted[i * 3 + a] = tile.xyz[i * 3 + a] - float(anchor[a]);
    return std::pair{transformPoints(shifted.data(), tile.valid.size(), destination, anchor), anchor};
}
} // namespace

struct SurfaceProjection::Impl {
    AlignedEngine::Impl& reader;
    std::shared_ptr<Geometry> geom;
    Surface surface;
    std::vector<const std::pair<std::string, Target>*> targets;
    std::string labelTarget;
    const Target* selecting = nullptr;
    Pair start{}, size{};
    double thickness = 0, padding = 0, radius = 0;
    Affine metric, destination, labelToUv;
    std::array<double, 3> uvOrigin{}, expansion{};
    std::array<std::int64_t, 3> margin{};
    Pair gridStart{}, gridEnd{};
    std::vector<float> offsets;  // the band's layers, micrometres along the normal
    std::string key;
    Triple origin{}, shape{};
    double preparationSeconds = 0;
    std::shared_ptr<SurfaceIndex> index;

    Impl(AlignedEngine::Impl& r, const Surface& s, const Pair& patchStart, const Pair& patchSize, const std::string& label, double thicknessUm, double paddingUm);
    std::vector<std::pair<Pair, Pair>> tiles() const;
    std::shared_ptr<SurfaceIndex> spatialIndex();
    std::optional<Bounds> footprintBounds(const TileRef& tile);
    std::shared_ptr<const Samples> samples(const TileRef& tile);
    json metadataFor(const Triple& at, const Triple& extent) const;
    using State = std::array<SampledArray, 7>;
    State newState(const Target& target, const Triple& extent);
    void rasterize(const std::vector<projection::Sample>& batch, const Triple& extent, std::vector<std::pair<std::string, State>>& states,
                   const std::vector<const std::pair<std::string, Target>*>& active);
    SampleResult read(const Triple& at, const Triple& extent);
    SampleResult readCrop(const Triple& at, const Triple& extent);
};

SurfaceProjection::Impl::Impl(AlignedEngine::Impl& r, const Surface& s, const Pair& patchStart, const Pair& patchSize, const std::string& label, double thicknessUm, double paddingUm)
    : reader(r), surface(s), labelTarget(label), start(patchStart), size(patchSize)
{
    geom = geometryOf(r, s);
    r.usedSurface = true;
    requireMatching(r, surface);
    for (const auto& item : r.surfaceTargets) {
        targets.push_back(&item);
        if (item.first == label) selecting = &item.second;
    }
    if (!selecting) bad("label_target must name a configured surface label");
    checkPatch(start, size);
    for (int a = 0; a < 2; ++a)
        if (start[std::size_t(a)] < 0 || start[std::size_t(a)] + size[std::size_t(a)] > selecting->shape[std::size_t(a + 1)]) bad("selected label patch must lie inside its opened pixel grid");
    thickness = checkDistance(thicknessUm, "thickness_um");
    padding = checkDistance(paddingUm, "padding_um");
    if (!surface.spacingUm) bad("surface physical spacing is unknown");
    const auto* registration = std::get_if<VolumeRegistration>(&*r.referenceReader->registration());
    if (!registration || !registration->volumeSpacingUm) bad("projection requires known reference volume physical spacing");
    metric = scaled(*registration->volumeSpacingUm) * r.referenceMap.forward;
    const auto inverse = metric.inverse();
    radius = thickness / 2;
    double voxelUm = INFINITY;
    for (int i = 0; i < 3; ++i) {
        const double norm = std::sqrt(inverse(i, 0) * inverse(i, 0) + inverse(i, 1) * inverse(i, 1) + inverse(i, 2) * inverse(i, 2));
        expansion[std::size_t(i)] = radius * norm;
        margin[std::size_t(i)] = std::int64_t(std::ceil(padding * norm));
        voxelUm = std::min(voxelUm, std::sqrt(metric(0, i) * metric(0, i) + metric(1, i) * metric(1, i) + metric(2, i) * metric(2, i)));
    }
    // The band's layers: at most half the smallest reference voxel apart, its two faces included.
    const auto half = radius > 0 ? std::max<std::int64_t>(1, std::int64_t(std::ceil(radius / (.5 * voxelUm) - 1e-9))) : 0;
    for (std::int64_t k = -half; k <= half; ++k) offsets.push_back(half ? float(radius * double(k) / double(half)) : 0.f);
    const auto between = r.between(selecting->source, surfaceFrame(surface));
    destination = r.referenceMap.forward.inverse() * between.forward * surface.xyzToVolume->forward;
    labelToUv = pixelAffine(*surfaceOf(selecting->reader->registration())->pixelToUv).forward;
    uvOrigin = labelToUv.apply({double(start[1]), double(start[0]), 0});
    const double x0 = double(start[1]) - .5, y0 = double(start[0]) - .5, x1 = double(start[1] + size[1]) - .5, y1 = double(start[0] + size[0]) - .5;
    const auto toGrid = geom->fromUv.forward * labelToUv;
    std::array<std::array<double, 3>, 4> corners{toGrid.apply({x0, y0, 0}), toGrid.apply({x1, y0, 0}), toGrid.apply({x1, y1, 0}), toGrid.apply({x0, y1, 0})};
    const auto extreme = [&](int a, bool high) {
        double v = corners[0][std::size_t(a)];
        for (const auto& c : corners) v = high ? std::max(v, c[std::size_t(a)]) : std::min(v, c[std::size_t(a)]);
        return v;
    };
    gridStart = {std::max<std::int64_t>(0, std::int64_t(std::floor(extreme(1, false)))), std::max<std::int64_t>(0, std::int64_t(std::floor(extreme(0, false))))};
    gridEnd = {std::min<std::int64_t>(geom->shape[0] - 1, std::int64_t(std::ceil(extreme(1, true)))), std::min<std::int64_t>(geom->shape[1] - 1, std::int64_t(std::ceil(extreme(0, true))))};
    key = "projection|" + geom->key + "|" + std::to_string(start[0]) + "," + std::to_string(start[1]) + "," + std::to_string(size[0]) + "," + std::to_string(size[1]) + "|" + matrixKey(labelToUv) + "|" + matrixKey(destination);
    // The crop: the sheet under the selecting patch, the band and the padding around it.
    const auto before = Clock::now();
    std::array<std::optional<double>, 3> low, high;
    for (const auto& [tileStart, tileSize] : tiles()) {
        const auto bounds = footprintBounds({tileStart, tileSize});
        if (!bounds) continue;
        for (std::size_t a = 0; a < 3; ++a) {
            low[a] = low[a] ? std::min(*low[a], (*bounds)[0][a]) : (*bounds)[0][a];
            high[a] = high[a] ? std::max(*high[a], (*bounds)[1][a]) : (*bounds)[1][a];
        }
    }
    if (!low[0] || !low[1] || !low[2]) bad("selected label patch has no valid nondegenerate surface geometry");
    Triple xyzOrigin{}, xyzEnd{};
    for (std::size_t a = 0; a < 3; ++a) {
        xyzOrigin[a] = std::int64_t(std::ceil(*low[a] - .5 - expansion[a])) - margin[a];
        xyzEnd[a] = std::int64_t(std::floor(*high[a] + .5 + expansion[a])) + margin[a] + 1;
    }
    origin = {xyzOrigin[2], xyzOrigin[1], xyzOrigin[0]};
    shape = {xyzEnd[2] - xyzOrigin[2], xyzEnd[1] - xyzOrigin[1], xyzEnd[0] - xyzOrigin[0]};
    for (const auto n : shape)
        if (n > (1 << 24)) bad("derived crop extent exceeds supported coordinate range");
    preparationSeconds = since(before);
    r.preparationSeconds += preparationSeconds;
}

std::vector<std::pair<Pair, Pair>> SurfaceProjection::Impl::tiles() const
{
    std::vector<std::pair<Pair, Pair>> out;
    for (std::int64_t y = gridStart[0]; y < gridEnd[0]; y += 32)
        for (std::int64_t x = gridStart[1]; x < gridEnd[1]; x += 32)
            out.push_back({Pair{y, x}, Pair{std::min<std::int64_t>(32, gridEnd[0] - y) + 1, std::min<std::int64_t>(32, gridEnd[1] - x) + 1}});
    return out;
}

// The extent in reference array coordinates of the surface under the selecting patch, on
// one tile: the patch's pixel corners and edges (every half pixel) through the patches.
std::optional<Bounds> SurfaceProjection::Impl::footprintBounds(const TileRef& tile)
{
    const auto boundsKey = "surface-bounds|" + key + "|" + std::to_string(tile.start[0]) + "," + std::to_string(tile.start[1]);
    if (const auto* cached = reader.plans.get(boundsKey)) return *std::static_pointer_cast<const std::optional<Bounds>>(*cached);
    std::optional<Bounds> out;
    const auto toGrid = geom->fromUv.forward * labelToUv;  // label pixel to grid position
    const auto toPixel = toGrid.inverse();
    // Footprint pixel (i, j) is label pixel (start - .5) + (i, j) / 2; those over this tile.
    double lo[2]{INFINITY, INFINITY}, hi[2]{-INFINITY, -INFINITY};
    for (const double c : {double(tile.start[1]), double(tile.start[1] + tile.size[1] - 1)})
        for (const double r : {double(tile.start[0]), double(tile.start[0] + tile.size[0] - 1)}) {
            const auto p = toPixel.apply({c, r, 0});
            const double i = (p[0] - (double(start[1]) - .5)) * 2, j = (p[1] - (double(start[0]) - .5)) * 2;
            lo[0] = std::min(lo[0], i), hi[0] = std::max(hi[0], i);
            lo[1] = std::min(lo[1], j), hi[1] = std::max(hi[1], j);
        }
    const auto i0 = std::max<std::int64_t>(0, std::int64_t(std::floor(lo[0]))), i1 = std::min<std::int64_t>(2 * size[1], std::int64_t(std::ceil(hi[0])));
    const auto j0 = std::max<std::int64_t>(0, std::int64_t(std::floor(lo[1]))), j1 = std::min<std::int64_t>(2 * size[0], std::int64_t(std::ceil(hi[1])));
    if (i1 >= i0 && j1 >= j0 && std::isfinite(lo[0]) && std::isfinite(lo[1])) {
        const auto grid = geom->tile(tile.start, tile.size);
        const auto g0 = toGrid.apply({double(start[1]) - .5 + double(i0) * .5, double(start[0]) - .5 + double(j0) * .5, 0});
        const grid::Raster raster{g0[0] - double(tile.start[1]), g0[1] - double(tile.start[0]), .5 * toGrid(0, 0), .5 * toGrid(1, 1), int(i1 - i0 + 1), int(j1 - j0 + 1), .5 * toGrid(0, 1), .5 * toGrid(1, 0)};
        const auto n = std::size_t(raster.width) * std::size_t(raster.height);
        std::vector<float> points(n * 3);
        gridSurface(*geom, *grid, grid::Interpolation::Linear)->resample(raster, points.data(), nullptr);
        std::vector<float> kept;
        std::optional<std::array<double, 3>> anchor;
        for (std::size_t i = 0; i < n; ++i) {
            const float* p = &points[i * 3];
            if (!std::isfinite(p[0])) continue;
            if (!anchor) anchor = {std::floor(double(p[0])), std::floor(double(p[1])), std::floor(double(p[2]))};
            for (std::size_t a = 0; a < 3; ++a) kept.push_back(p[a] - float((*anchor)[a]));
        }
        if (anchor) {
            const auto moved = transformPoints(kept.data(), kept.size() / 3, destination, *anchor);
            Bounds bounds{{{INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}}};
            for (std::size_t i = 0; i < moved.points.size(); i += 3)
                for (std::size_t a = 0; a < 3; ++a) {
                    bounds[0][a] = std::min(bounds[0][a], double(moved.points[i + a]) + double(moved.base[a]));
                    bounds[1][a] = std::max(bounds[1][a], double(moved.points[i + a]) + double(moved.base[a]));
                }
            out = bounds;
        }
    }
    reader.plans.put(boundsKey, std::make_shared<std::optional<Bounds>>(out), 512);
    return out;
}

std::shared_ptr<SurfaceIndex> SurfaceProjection::Impl::spatialIndex()
{
    // Index the whole registered sheet once, including folds returning to a crop. Bulk
    // coordinate reads are large; index leaves are smaller, and only intersecting leaves
    // are sampled and read as labels.
    std::int64_t block = 256;
    while (std::size_t((block + 1) * (block + 1)) * 80 > reader.workBytes / 2 && block > 1) block /= 2;
    std::int64_t step = std::min<std::int64_t>(64, block);
    const auto leaves = [&](std::int64_t s) { return std::size_t((geom->shape[0] - 1 + s - 1) / s) * std::size_t((geom->shape[1] - 1 + s - 1) / s); };
    while (leaves(step) * 2048 > reader.workBytes / 2 && step < block) step *= 2;
    const auto count = leaves(step);
    if (count * 2048 > reader.workBytes / 2) throw DecodeBudgetError("surface spatial index exceeds sampling_work_bytes");
    const auto cacheKey = "surface-spatial-index|" + geom->key + "|" + matrixKey(destination) + "|" + std::to_string(step);
    if (const auto* cached = reader.plans.get(cacheKey)) return std::static_pointer_cast<SurfaceIndex>(*cached);
    std::vector<std::pair<TileRef, Bounds>> entries;
    for (std::int64_t y = 0; y < geom->shape[0] - 1; y += block)
        for (std::int64_t x = 0; x < geom->shape[1] - 1; x += block) {
            const Pair tileSize{std::min(block, geom->shape[0] - 1 - y) + 1, std::min(block, geom->shape[1] - 1 - x) + 1};
            const auto tile = geom->tile({y, x}, tileSize);
            const auto moved = movedVertices(*tile, destination);
            if (!moved) continue;
            for (std::int64_t iy = 0; iy < tileSize[0] - 1; iy += step)
                for (std::int64_t ix = 0; ix < tileSize[1] - 1; ix += step) {
                    const Pair leaf{std::min(step, tileSize[0] - 1 - iy) + 1, std::min(step, tileSize[1] - 1 - ix) + 1};
                    std::array<float, 3> lo{INFINITY, INFINITY, INFINITY}, hi{-INFINITY, -INFINITY, -INFINITY};
                    bool any = false;
                    for (std::int64_t j = iy; j < iy + leaf[0]; ++j)
                        for (std::int64_t i = ix; i < ix + leaf[1]; ++i) {
                            const auto at = std::size_t(j * tileSize[1] + i);
                            if (!tile->valid[at]) continue;
                            any = true;
                            for (std::size_t a = 0; a < 3; ++a) {
                                lo[a] = std::min(lo[a], moved->first.points[at * 3 + a]);
                                hi[a] = std::max(hi[a], moved->first.points[at * 3 + a]);
                            }
                        }
                    if (!any) continue;
                    Bounds bounds;
                    for (std::size_t a = 0; a < 3; ++a) {
                        bounds[0][a] = double(lo[a]) + double(moved->first.base[a]);
                        bounds[1][a] = double(hi[a]) + double(moved->first.base[a]);
                    }
                    entries.push_back({TileRef{Pair{y + iy, x + ix}, leaf}, bounds});
                }
        }
    auto built = std::make_shared<SurfaceIndex>(std::move(entries));
    reader.plans.put(cacheKey, built, count * 2048);
    reader.peakWork = std::max(reader.peakWork, count * 2048 + std::size_t((block + 1) * (block + 1)) * 80);
    return built;
}

std::shared_ptr<const Samples> SurfaceProjection::Impl::samples(const TileRef& tile)
{
    const auto tileKey = key + "|samples|" + std::to_string(tile.start[0]) + "," + std::to_string(tile.start[1]) + "," + std::to_string(tile.size[0]) + "," + std::to_string(tile.size[1]);
    if (const auto* cached = reader.tiles.get(tileKey)) return std::static_pointer_cast<const Samples>(*cached);
    auto out = std::make_shared<Samples>();
    const auto grid = geom->tile(tile.start, tile.size);
    const auto rows = grid->rows, cols = grid->cols;
    const auto moved = movedVertices(*grid, destination);
    if (!moved) return out;
    // Points per cell: the longest step between neighbouring valid vertices, in reference
    // voxels, is covered half a voxel at a time.
    const auto validAt = [&](std::int64_t y, std::int64_t x) { return grid->valid[std::size_t(y * cols + x)] != 0; };
    const auto distance = [&](std::int64_t a, std::int64_t b) {
        const float* p = &moved->first.points[std::size_t(a) * 3];
        const float* q = &moved->first.points[std::size_t(b) * 3];
        return std::sqrt(double(p[0] - q[0]) * (p[0] - q[0]) + double(p[1] - q[1]) * (p[1] - q[1]) + double(p[2] - q[2]) * (p[2] - q[2]));
    };
    double stepU = 0, stepV = 0;
    for (std::int64_t y = 0; y < rows; ++y)
        for (std::int64_t x = 0; x < cols; ++x) {
            if (!validAt(y, x)) continue;
            if (x + 1 < cols && validAt(y, x + 1)) stepU = std::max(stepU, distance(y * cols + x, y * cols + x + 1));
            if (y + 1 < rows && validAt(y + 1, x)) stepV = std::max(stepV, distance(y * cols + x, (y + 1) * cols + x));
        }
    const int nu = subdivisions(stepU), nv = subdivisions(stepV);
    // The last row and column of cells belong to the next leaf, except at the grid's end.
    const bool lastCol = tile.start[1] + tile.size[1] == geom->shape[1], lastRow = tile.start[0] + tile.size[0] == geom->shape[0];
    const std::int64_t width = (cols - 1) * nu + (lastCol ? 1 : 0), height = (rows - 1) * nv + (lastRow ? 1 : 0);
    if (width <= 0 || height <= 0) return out;
    const auto n = std::size_t(width) * std::size_t(height);
    std::vector<float> points(n * 3), steps(n * 3);
    gridSurface(*geom, *grid, grid::Interpolation::Linear)->resample(grid::Raster{0, 0, 1.0 / nu, 1.0 / nv, int(width), int(height)}, points.data(), steps.data());
    // The points that exist: about the vertices' anchor, then into reference coordinates.
    const auto& anchor = moved->second;
    float linear[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) linear[r * 3 + c] = float(destination(r, c));
    std::vector<float> kept;
    const std::int64_t layers = std::int64_t(offsets.size()), ids = geom->shape[1] * kSubdivisions;
    for (std::int64_t j = 0; j < height; ++j)
        for (std::int64_t i = 0; i < width; ++i) {
            const auto at = std::size_t(j * width + i);
            const float* p = &points[at * 3];
            if (!std::isfinite(p[0])) continue;
            for (std::size_t a = 0; a < 3; ++a) kept.push_back(p[a] - float(anchor[a]));
            const float* s = &steps[at * 3];
            for (int r = 0; r < 3; ++r) out->step.push_back(linear[r * 3] * s[0] + linear[r * 3 + 1] * s[1] + linear[r * 3 + 2] * s[2]);
            const auto uv = geom->toUv.forward.apply({double(tile.start[1]) + double(i) / nu, double(tile.start[0]) + double(j) / nv, 0});
            out->uv.push_back(float(uv[0] - uvOrigin[0]));
            out->uv.push_back(float(uv[1] - uvOrigin[1]));
            const std::int64_t row = tile.start[0] * kSubdivisions + j * (kSubdivisions / nv), col = tile.start[1] * kSubdivisions + i * (kSubdivisions / nu);
            out->ids.push_back((row * ids + col) * layers);
        }
    if (!kept.empty()) {
        auto local = transformPoints(kept.data(), kept.size() / 3, destination, anchor);
        Bounds bounds{{{INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}}};
        for (std::size_t i = 0; i < local.points.size(); i += 3)
            for (std::size_t a = 0; a < 3; ++a) {
                bounds[0][a] = std::min(bounds[0][a], double(local.points[i + a]) + double(local.base[a]));
                bounds[1][a] = std::max(bounds[1][a], double(local.points[i + a]) + double(local.base[a]));
            }
        out->xyz = std::move(local.points);
        out->base = local.base;
        out->bounds = bounds;
    }
    reader.tiles.put(tileKey, out, out->bytes());
    return out;
}

json SurfaceProjection::Impl::metadataFor(const Triple& at, const Triple& extent) const
{
    const auto* registration = surfaceOf(selecting->reader->registration());
    return {{"label_target", labelTarget}, {"label_id", selecting->source.id}, {"label_start_yx", start}, {"label_size_yx", size}, {"surface_id", surface.id},
            {"pixel_to_uv", *registration->pixelToUv}, {"reference_id", reader.reference.id}, {"reference_frame", reader.referenceFrame},
            {"selection", plainFields(encode(reader.referenceReader->resolved().selection))}, {"origin_zyx", at}, {"shape_zyx", extent},
            {"crop_to_reference", shift(double(at[2]), double(at[1]), double(at[0])).json()}, {"array_to_level0", reader.referenceMap.forward.json()},
            {"full_origin_zyx", origin}, {"full_shape_zyx", shape}, {"thickness_um", thickness}, {"padding_um", padding}, {"annotation_extent", "crop"},
            {"preparation_seconds", preparationSeconds}};
}

SurfaceProjection::Impl::State SurfaceProjection::Impl::newState(const Target& target, const Triple& extent)
{
    const auto before = Clock::now();
    const std::int64_t n = extent[0] * extent[1] * extent[2];
    const auto count = std::size_t(n);
    Block values = Block::allocate({target.channels, n}, target.nativeDtype);
    const auto fill = elementBytes(target.reader->fillValue(), target.nativeDtype);
    for (std::size_t i = 0; i < values.nbytes; i += fill.size()) std::memcpy(values.data.get() + i, fill.data(), fill.size());
    Block covered = Block::allocate({1, n}, "|b1"), conflict = Block::allocate({1, n}, "|b1"), coverage = Block::allocate({1, n}, "|b1");
    std::memset(covered.data.get(), 0, count);
    std::memset(conflict.data.get(), 0, count);
    std::memset(coverage.data.get(), 0, count);
    Block distance = Block::allocate({n}, "<f4");
    std::fill_n(reinterpret_cast<float*>(distance.data.get()), count, std::numeric_limits<float>::infinity());
    Block winner = Block::allocate({n}, "<i8");
    std::fill_n(reinterpret_cast<std::int64_t*>(winner.data.get()), count, std::numeric_limits<std::int64_t>::max());
    Block uv = Block::allocate({n, 2}, "<f4");
    std::fill_n(reinterpret_cast<float*>(uv.data.get()), count * 2, std::numeric_limits<float>::quiet_NaN());
    State state{reader.fromHost(values), reader.fromHost(covered), reader.fromHost(conflict), reader.fromHost(distance), reader.fromHost(winner), reader.fromHost(uv), reader.fromHost(coverage)};
    reader.outputAllocationSeconds += since(before);
    return state;
}

void SurfaceProjection::Impl::rasterize(const std::vector<projection::Sample>& batch, const Triple& extent, std::vector<std::pair<std::string, State>>& states,
                                        const std::vector<const std::pair<std::string, Target>*>& active)
{
    auto& r = reader;
    const auto before = Clock::now();
    const auto splitAll = [&](const std::vector<const std::pair<std::string, Target>*>& which) {
        const auto middle = batch.size() / 2;
        rasterize(std::vector(batch.begin(), batch.begin() + std::ptrdiff_t(middle)), extent, states, which);
        rasterize(std::vector(batch.begin() + std::ptrdiff_t(middle), batch.end()), extent, states, which);
    };
    std::optional<projection::Plan> plan;
    try {
        plan.emplace(batch.data(), batch.size(), extent, r.workBytes / 3);
    } catch (const std::length_error&) {
        if (batch.size() <= 1) throw DecodeBudgetError("projection sample batch exceeds sampling_work_bytes");
        splitAll(active);
        return;
    }
    r.candidateIndexSeconds += since(before);
    r.peakWork = std::max(r.peakWork, plan->bytes() * 3);
    // Samples carry UV relative to the selecting label patch origin.
    const auto patchShift = shift(uvOrigin[0], uvOrigin[1], uvOrigin[2]);
    for (const auto* item : active) {
        const auto& [name, target] = *item;
        const auto mapping = target.mapping.forward * patchShift;
        const auto p = pixelPoints(plan->uv.data(), plan->uv.size() / 2, mapping);
        float minY = INFINITY, maxY = -INFINITY, minX = INFINITY, maxX = -INFINITY;
        for (std::size_t i = 0; i < p.points.size() / 3; ++i) {
            minY = std::min(minY, p.points[i * 3 + 1]), maxY = std::max(maxY, p.points[i * 3 + 1]);
            minX = std::min(minX, p.points[i * 3 + 2]), maxX = std::max(maxX, p.points[i * 3 + 2]);
        }
        Pair lo{std::max<std::int64_t>(0, std::int64_t(std::floor(double(minY))) + p.base[1] - 1), std::max<std::int64_t>(0, std::int64_t(std::floor(double(minX))) + p.base[2] - 1)};
        const Pair hi{std::min<std::int64_t>(target.shape[1], std::int64_t(std::ceil(double(maxY))) + p.base[1] + 2), std::min<std::int64_t>(target.shape[2], std::int64_t(std::ceil(double(maxX))) + p.base[2] + 2)};
        Pair extentYX{std::max<std::int64_t>(0, hi[0] - lo[0]), std::max<std::int64_t>(0, hi[1] - lo[1])};
        if (!extentYX[0] || !extentYX[1]) {
            lo = {0, 0};
            extentYX = {0, 0};
        }
        if (std::size_t(extentYX[0] * extentYX[1] * target.channels) * itemOf(target.dtype) * 3 > r.workBytes) {
            if (batch.size() <= 1) throw DecodeBudgetError("one sample's label footprint exceeds sampling_work_bytes");
            // This target only, so that earlier targets are not processed twice.
            splitAll({item});
            continue;
        }
        const auto labelKey = "surface-label|" + target.storageKey + "|" + std::to_string(lo[0]) + "," + std::to_string(lo[1]) + "|" + std::to_string(extentYX[0]) + "," + std::to_string(extentYX[1]) + "|" + r.backend;
        std::shared_ptr<Tile> label;
        if (const auto* cached = r.tiles.get(labelKey)) {
            label = std::static_pointer_cast<Tile>(*cached);
        } else {
            label = std::make_shared<Tile>();
            Block values;
            if (extentYX[0] && extentYX[1]) {
                values = r.canonical(target.reader->read({lo[0], lo[1]}, {extentYX[0], extentYX[1]}), target.reader->axes());
            } else {
                values = Block::allocate({target.channels, 1, 0, 0}, target.nativeDtype);
            }
            ++r.tileReads;
#ifdef VC_READER_CUDA
            if (r.backend == "cuda") {
                r.uploads += values.nbytes;
                const auto uploaded = r.fromHost(values);
                label->device = std::static_pointer_cast<device::Buffer>(uploaded.device);
            } else
#endif
                label->host = values;
            r.tiles.put(labelKey, label, values.nbytes);
        }
        const auto matrix = shift(double(-lo[1]), double(-lo[0]), 0) * mapping;
        std::array<float, 6> rows{float(matrix(0, 0)), float(matrix(0, 1)), float(matrix(0, 3)), float(matrix(1, 0)), float(matrix(1, 1)), float(matrix(1, 3))};
        for (const float v : rows)
            if (!std::isfinite(v)) bad("nonfinite UV mapping");
        auto config = plan->grid;
        config.channels = int(target.channels);
        config.label_h = int(extentYX[0]);
        config.label_w = int(extentYX[1]);
        std::copy(rows.begin(), rows.end(), config.uv_to_label);
        config.label_origin[0] = lo[0];
        config.label_origin[1] = lo[1];
        auto* entry = &std::find_if(states.begin(), states.end(), [&](const auto& s) { return s.first == name; })->second;
        const auto type = elementOf(target.nativeDtype);
        if (type.kind == 'f') bad("projection requires categorical integers");
        const auto rasterBefore = Clock::now();
#ifdef VC_READER_CUDA
        if (r.backend == "cuda") {
            const auto ptr = [](SampledArray& a) { return std::static_pointer_cast<device::Buffer>(a.device)->ptr; };
            device::Guard guard(r.device);
            projection::launch(*plan, config, label->device->ptr, type, ptr((*entry)[0]), static_cast<bool*>(ptr((*entry)[1])), static_cast<bool*>(ptr((*entry)[2])),
                               static_cast<float*>(ptr((*entry)[3])), static_cast<std::int64_t*>(ptr((*entry)[4])), static_cast<float*>(ptr((*entry)[5])), static_cast<bool*>(ptr((*entry)[6])), r.device);
            r.candidateUploadBytes += plan->bytes();
        } else
#endif
        {
            const auto* labelData = label->host.data.get();
            type.visit_load([&](auto load) {
                using T = decltype(load(static_cast<const std::uint8_t*>(nullptr)));
                for (std::size_t i = 0; i < config.count; ++i)
                    projection::raster(config, plan->uv.data(), plan->distance.data(), plan->ids.data(), plan->offsets.data(), reinterpret_cast<const T*>(labelData),
                                       reinterpret_cast<T*>((*entry)[0].host.get()), reinterpret_cast<bool*>((*entry)[1].host.get()), reinterpret_cast<bool*>((*entry)[2].host.get()),
                                       reinterpret_cast<float*>((*entry)[3].host.get()), reinterpret_cast<std::int64_t*>((*entry)[4].host.get()), reinterpret_cast<float*>((*entry)[5].host.get()),
                                       reinterpret_cast<bool*>((*entry)[6].host.get()), i);
            });
        }
        r.rasterCallSeconds += since(rasterBefore);
    }
}

SampleResult SurfaceProjection::Impl::read(const Triple& at, const Triple& extent)
{
    auto& r = reader;
    checkBudget(r, extent);
    if (!index) {
        const auto before = Clock::now();
        index = spatialIndex();
        const auto elapsed = since(before);
        preparationSeconds += elapsed;
        r.preparationSeconds += elapsed;
    }
    // Ordinary volume reads; surface targets follow.
    SampleResult result;
    result.metadata = r.metadata.is_null() ? json::object() : r.metadata;
    result.metadata["projection"] = metadataFor(at, extent);
    for (const auto& [name, target] : r.targets) {
        TargetResult out;
        auto [data, valid] = std::move(r.sampleMany(target, {{at, extent}}).at(0));
        out.data = std::move(data);
        out.valid = std::move(valid);
        for (const auto& [key_, child] : target.companions) out.masks.emplace_back(key_, r.sampleMany(child, {{at, extent}}).at(0).first);
        result.targets.emplace_back(name, std::move(out));
    }
    std::vector<std::pair<std::string, State>> states;
    for (const auto* item : targets) states.emplace_back(item->first, newState(item->second, extent));
    const std::array<double, 3> requestLo{double(at[2]), double(at[1]), double(at[0])};
    const std::array<double, 3> requestHi{double(at[2] + extent[2] - 1), double(at[1] + extent[1] - 1), double(at[0] + extent[0] - 1)};
    const auto outside = [&](const Bounds& b) {
        for (std::size_t a = 0; a < 3; ++a)
            if (b[1][a] + expansion[a] + .5 < requestLo[a] || b[0][a] - expansion[a] - .5 > requestHi[a]) return true;
        return false;
    };
    float metricLinear[9];
    for (int row = 0; row < 3; ++row)
        for (int c = 0; c < 3; ++c) metricLinear[row * 3 + c] = float(metric(row, c));
    std::vector<TileRef> hits;
    index->query(requestLo, requestHi, expansion, [&](const TileRef& tile) { hits.push_back(tile); });
    // Every layer of every sample rounds to a voxel; inside the crop it is a candidate,
    // at the physical distance from that voxel's centre to the sample's surface point.
    std::vector<projection::Sample> batch;
    for (const auto& tile : hits) {
        const auto s = samples(tile);
        if (!s->bounds || outside(*s->bounds)) continue;
        float origin[3];
        for (std::size_t a = 0; a < 3; ++a) origin[a] = float(requestLo[a] - double(s->base[a]));
        for (std::size_t i = 0; i < s->count(); ++i) {
            const float* p = &s->xyz[i * 3];
            const float* step = &s->step[i * 3];
            for (std::size_t k = 0; k < offsets.size(); ++k) {
                std::int64_t index3[3];
                bool inside = true;
                for (std::size_t a = 0; a < 3 && inside; ++a) {
                    index3[a] = nearest_index(p[a] + step[a] * offsets[k] - origin[a]);
                    inside = index3[a] >= 0 && index3[a] < extent[2 - a];
                }
                if (!inside) continue;
                float c[3];
                for (std::size_t a = 0; a < 3; ++a) c[a] = float(index3[a]) + origin[a] - p[a];
                float d2 = 0;
                for (int row = 0; row < 3; ++row) {
                    const float v = metricLinear[row * 3] * c[0] + metricLinear[row * 3 + 1] * c[1] + metricLinear[row * 3 + 2] * c[2];
                    d2 += v * v;
                }
                batch.push_back({(index3[2] * extent[1] + index3[1]) * extent[2] + index3[0], {s->uv[i * 2], s->uv[i * 2 + 1]}, std::sqrt(d2), s->ids[i] + std::int64_t(k)});
            }
        }
    }
    if (!batch.empty()) rasterize(batch, extent, states, targets);
    const auto patchShift = shift(uvOrigin[0], uvOrigin[1], uvOrigin[2]);
    for (std::size_t s = 0; s < states.size(); ++s) {
        const auto& [name, target] = *targets[s];
        auto& state = states[s].second;
        TargetResult out;
        out.data = state[0].reshaped({target.channels, extent[0], extent[1], extent[2]});
        if (r.backend == "cpu" && target.dtype != target.nativeDtype) {
            const auto converted = convertBlock(hostBlock(out.data), target.dtype);
            out.data.host = converted.data;
            out.data.dtype = target.dtype;
        }
        out.valid = state[1].reshaped({1, extent[0], extent[1], extent[2]});
        out.masks.emplace_back("coverage", state[6].reshaped({1, extent[0], extent[1], extent[2]}));
        out.masks.emplace_back("conflict", state[2].reshaped({1, extent[0], extent[1], extent[2]}));
        if (!target.companions.empty()) {
            auto uvState = state[5];
#ifdef VC_READER_CUDA
            if (r.backend == "cuda") {
                auto storage = std::static_pointer_cast<device::Buffer>(uvState.device);
                Block host = Block::allocate(uvState.shape, uvState.dtype);
                device::Guard guard(storage->device);
                if (host.nbytes) device::check(cudaMemcpyAsync(host.data.get(), storage->ptr, host.nbytes, cudaMemcpyDeviceToHost, device::stream()));
                device::complete();
                uvState.host = host.data;
                uvState.device.reset();
            }
#endif
            const auto count = std::size_t(uvState.shape[0]);
            for (const auto& [key_, child] : target.companions) {
                const auto p = pixelPoints(reinterpret_cast<const float*>(uvState.host.get()), count, child.mapping.forward * patchShift);
                auto [mask, unused] = r.samplePoints(child, p.points, p.base);
                (void)unused;
                out.masks.emplace_back(key_, mask.reshaped({child.channels, extent[0], extent[1], extent[2]}));
            }
        }
        result.targets.emplace_back(name, std::move(out));
    }
    return result;
}

SampleResult SurfaceProjection::Impl::readCrop(const Triple& at, const Triple& extent)
{
    auto& r = reader;
    checkBudget(r, extent);
    std::optional<SampleResult> result;
    const auto copy = [&](const SampledArray& source, SampledArray& destination, const Triple& offset) {
        const auto channels = source.shape[0];
        const auto item = itemOf(source.dtype);
#ifdef VC_READER_CUDA
        if (r.backend == "cuda") {
            device::Guard guard(r.device);
            aligned::copy_region_cuda(std::static_pointer_cast<device::Buffer>(source.device)->ptr, std::static_pointer_cast<device::Buffer>(destination.device)->ptr,
                                      Triple{source.shape[1], source.shape[2], source.shape[3]}, Triple{destination.shape[1], destination.shape[2], destination.shape[3]}, offset, int(channels), item);
            return;
        }
#endif
        const auto& s = source.shape;
        const auto& d = destination.shape;
        for (std::int64_t c = 0; c < channels; ++c)
            for (std::int64_t z = 0; z < s[1]; ++z)
                for (std::int64_t y = 0; y < s[2]; ++y)
                    std::memcpy(destination.host.get() + std::size_t(((c * d[1] + z + offset[0]) * d[2] + y + offset[1]) * d[3] + offset[2]) * item,
                                source.host.get() + std::size_t(((c * s[1] + z) * s[2] + y) * s[3]) * item, std::size_t(s[3]) * item);
    };
    for (const auto& [start_, size_] : tileRegions(at, extent, Triple{16, 32, 64})) {
        auto block = read(start_, size_);
        if (size_ == extent) return block;
        if (!result) {
            result.emplace();
            result->metadata = block.metadata;
            result->metadata["projection"] = metadataFor(at, extent);
            for (const auto& [name, target] : block.targets) {
                TargetResult out;
                out.data = r.empty({target.data.shape[0], extent[0], extent[1], extent[2]}, target.data.dtype);
                out.valid = r.empty({target.valid.shape[0], extent[0], extent[1], extent[2]}, target.valid.dtype);
                for (const auto& [key_, array] : target.masks) out.masks.emplace_back(key_, r.empty({array.shape[0], extent[0], extent[1], extent[2]}, array.dtype));
                result->targets.emplace_back(name, std::move(out));
            }
        }
        const Triple offset{start_[0] - at[0], start_[1] - at[1], start_[2] - at[2]};
        for (std::size_t t = 0; t < block.targets.size(); ++t) {
            const auto& from = block.targets[t].second;
            auto& to = result->targets[t].second;
            copy(from.data, to.data, offset);
            copy(from.valid, to.valid, offset);
            for (std::size_t m = 0; m < from.masks.size(); ++m) copy(from.masks[m].second, to.masks[m].second, offset);
        }
    }
    if (!result) bad("derived crop is empty");
    return *result;
}

SurfaceProjection::SurfaceProjection(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SurfaceProjection::~SurfaceProjection() = default;
std::array<std::int64_t, 3> SurfaceProjection::origin() const { return impl_->origin; }
std::array<std::int64_t, 3> SurfaceProjection::shape() const { return impl_->shape; }
std::vector<std::pair<Triple, Triple>> SurfaceProjection::regions(const Triple& tile) const
{
    for (const auto n : tile)
        if (n <= 0 || n > (1 << 24)) bad("tile_size must be three positive ZYX dimensions");
    return tileRegions(impl_->origin, impl_->shape, tile);
}
SampleResult SurfaceProjection::readCrop(const Triple& at, const Triple& extent)
{
    std::lock_guard lock(impl_->reader.mutex);
    impl_->reader.checkOpen();
    return impl_->readCrop(at, extent);
}

std::shared_ptr<SurfaceProjection> SampleReader::projection(const Surface& surface, const Pair& start, const Pair& size, const std::string& labelTarget, double thicknessUm, double paddingUm)
{
    auto& s = *impl_;
    std::lock_guard lock(s.mutex);
    s.checkOpen();
    return std::make_shared<SurfaceProjection>(std::make_unique<SurfaceProjection::Impl>(s, surface, start, size, labelTarget, thicknessUm, paddingUm));
}
SampleResult SampleReader::projectSurface(const Surface& surface, const Pair& start, const Pair& size, const std::string& labelTarget, double thicknessUm, double paddingUm)
{
    auto cursor = projection(surface, start, size, labelTarget, thicknessUm, paddingUm);
    return cursor->readCrop(cursor->origin(), cursor->shape());
}

} // namespace vcr::sources
