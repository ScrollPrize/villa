// Registered coordinate grids and bounded surface operations.
#include "aligned_internal.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "vc_reader/grid.hpp"
#include "vc_reader/tifxyz.hpp"
#include "../reader_types.hpp"

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
    const auto trimmed = [](std::string text) {
        while (!text.empty() && text.back() == '/') text.pop_back();
        return text;
    };
    if (surface.geometry.format == "tifxyz") {
        if (!surface.geometry.location) bad("surface geometry location is unresolved");
        std::string root = *surface.geometry.location;
        json meta;
        EncodedFiles files(owner.options.reader.options);
        std::vector<std::string> candidates{root};
        candidates.insert(candidates.end(), surface.geometry.alternatives.begin(), surface.geometry.alternatives.end());
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            try {
                const auto opened = files.open(trimmed(candidates[i]) + "/meta.json");
                std::string text;
                char block[1 << 16];
                ssize_t n;
                off_t at = 0;
                while ((n = ::pread(opened.fd, block, sizeof block, at)) > 0) {
                    text.append(block, std::size_t(n));
                    at += n;
                }
                meta = json::parse(text);
                root = candidates[i];
                break;
            } catch (const zarr::NotFound&) {
                if (i + 1 == candidates.size()) throw;
            } catch (const CacheMiss&) {
                if (i + 1 == candidates.size()) throw;
            }
        }
        files.close();
        const auto scale = tifxyz::parseMetadata(meta, root).scale;  // rows, columns
        const PixelAffine observed{{{1 / scale[1], 0, 0}, {0, 1 / scale[0], 0}}};
        if (grid && *grid != observed) bad("surface pixel_to_uv contradicts tifxyz scale");
        grid = observed;
        for (const char* axis : {"x", "y", "z"}) {
            Source source;
            source.id = surface.id + "/" + axis;
            source.storage.location = trimmed(root) + "/" + axis + ".tif";
            source.storage.format = "tiff";
            source.role = "scalar";
            source.axes = std::vector<std::string>{"y", "x"};
            coordinateReaders.push_back(owner.open(finished(std::move(source))));
        }
        if (surface.tifxyzMask) {
            Source source;
            source.id = surface.id + "/geometry-mask";
            source.storage.location = trimmed(root) + "/mask.tif";
            source.storage.format = "tiff";
            source.role = "label";
            source.axes = std::vector<std::string>{"y", "x"};
            try {
                mask = owner.open(finished(std::move(source)));
            } catch (const zarr::NotFound&) {
            }
        }
    } else {
        if (!grid) bad("coordinate grid requires explicit pixel_to_uv");
        Source source;
        source.id = surface.id + "/coordinates";
        source.storage = surface.geometry;
        source.role = "scalar";
        source.axes = surface.axes;
        coordinateReaders.push_back(owner.open(finished(std::move(source))));
    }
    const auto yxOf = [](SourceReader& r) -> Pair {
        const auto& axes = r.axes();
        const auto& sizes = r.shape();
        Pair out{};
        for (std::size_t i = 0; i < axes.size(); ++i) {
            if (axes[i] == "y") out[0] = sizes[i];
            if (axes[i] == "x") out[1] = sizes[i];
        }
        return out;
    };
    auto& first = *coordinateReaders.front();
    shape = yxOf(first);
    for (auto& r : coordinateReaders)
        if (r->spatialAxes() != std::vector<std::string>{"y", "x"} || yxOf(*r) != shape) bad("surface coordinates require matching YX grids");
    if (coordinateReaders.size() == 1) {
        const auto& axes = first.axes();
        const auto at = std::find(axes.begin(), axes.end(), "c");
        if (at == axes.end() || first.shape()[std::size_t(at - axes.begin())] != 3) bad("coordinate grid must contain three components");
    }
    if (mask && (mask->shape() != std::vector<std::int64_t>{shape[0], shape[1]})) bad("tifxyz mask shape does not match coordinate grid");
    toUv = pixelAffine(*grid);
    fromUv = VolumeTransform{toUv.forward.inverse(), {}};
    orientation = (*grid)[0][0] * (*grid)[1][1] - (*grid)[0][1] * (*grid)[1][0] > 0 ? 1 : -1;
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
    out->xyz.resize(cells * 3);
    // Channels of each component, in XYZ order.
    const std::vector<std::int64_t> startList{start[0], start[1]}, sizeList{size[0], size[1]};
    std::vector<int> order{0, 1, 2};
    std::vector<Block> arrays;
    for (auto& reader : coordinateReaders) {
        arrays.push_back(owner.canonical(reader->read(startList, sizeList), reader->axes()));
        owner.geometryReadBytes += arrays.back().nbytes;
    }
    if (coordinateReaders.size() == 1) {
        const std::string letters = "xyz";
        for (int i = 0; i < 3; ++i)
            order[std::size_t(i)] = int(std::find(surface.componentOrder->begin(), surface.componentOrder->end(), std::string(1, letters[std::size_t(i)])) - surface.componentOrder->begin());
    }
    const auto load = [&](const Block& block, std::size_t index) {
        ElementType type;
        type.kind = block.dtype[1];
        type.itemsize = block.itemsize();
        return type.kind == 'f' && type.itemsize == 4 ? *reinterpret_cast<const float*>(block.data.get() + index * 4) : float(type.load(block.data.get() + index * type.itemsize));
    };
    for (int component = 0; component < 3; ++component) {
        const Block& block = arrays[coordinateReaders.size() == 1 ? 0 : std::size_t(component)];
        const std::size_t channel = coordinateReaders.size() == 1 ? std::size_t(order[std::size_t(component)]) : 0;
        for (std::size_t i = 0; i < cells; ++i) out->xyz[i * 3 + std::size_t(component)] = load(block, channel * cells + i);
    }
    out->valid.assign(cells, 1);
    const bool tifxyz = surface.geometry.format == "tifxyz";
    for (std::size_t i = 0; i < cells; ++i) {
        for (int a = 0; a < 3; ++a) {
            const float v = out->xyz[i * 3 + std::size_t(a)];
            if (!std::isfinite(v) || (tifxyz && !(v >= 0))) out->valid[i] = 0;
        }
    }
    if (mask) {
        const auto values = mask->read(startList, sizeList);
        const auto item = values.itemsize();
        for (std::size_t i = 0; i < cells; ++i) {
            const auto* p = values.data.get() + i * item;
            if (std::all_of(p, p + item, [](std::uint8_t b) { return b == 0; })) out->valid[i] = 0;
        }
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
// A UV patch as a raster over the stored grid, with the vertex window it reads. Vertices
// missing are NaN. None when the patch misses the grid. A grid position's patch is the cell
// at its floor, so the window holds every cell the patch can reach.
struct Patch {
    std::shared_ptr<const GeometryTile> tile;
    std::vector<float> xyz;
    double x0, y0, sx, sy, kx, ky;
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
    patch.xyz = patch.tile->xyz;
    for (std::size_t i = 0; i < patch.tile->valid.size(); ++i)
        if (!patch.tile->valid[i])
            for (int a = 0; a < 3; ++a) patch.xyz[i * 3 + std::size_t(a)] = std::numeric_limits<float>::quiet_NaN();
    patch.x0 = x0 - double(c0);
    patch.y0 = y0 - double(r0);
    patch.sx = sx, patch.sy = sy, patch.kx = kx, patch.ky = ky;
    return patch;
}
std::vector<float> layerOffsets(const Geometry& geom, bool explicitNormals, const std::vector<double>& offsets)
{
    // A mirrored UV grid flips the patch normals; explicit vertex normals keep their sign.
    std::vector<float> out;
    for (const double v : offsets) out.push_back(float(v * (explicitNormals ? 1 : geom.orientation)));
    return out;
}
grid::Surface gridSurface(const Patch& patch, const Vec3& spacing)
{
    grid::SurfaceOptions options;
    options.spacing = {float(spacing[0]), float(spacing[1]), float(spacing[2])};
    grid::Vertices vertices{patch.xyz.data(), int(patch.tile->rows), int(patch.tile->cols), std::size_t(patch.tile->cols) * 3, patch.tile->normals.empty() ? nullptr : patch.tile->normals.data()};
    return grid::Surface(vertices, options);
}
TargetResult renderVolume(AlignedEngine::Impl& s, Target& target, const std::optional<Patch>& patch, const Pair& size, const std::vector<float>& layers, const Affine& toReference, const Vec3& spacing)
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
    if (patch) {
        const auto mapping = target.mapping.forward * toReference;
        std::array<double, 12> map{};
        for (int a = 0; a < 3; ++a)
            for (int c = 0; c < 4; ++c) map[std::size_t(a * 4 + c)] = mapping(2 - a, c);
        ElementType type;
        type.kind = dtype[1];
        type.itemsize = itemOf(dtype);
        type.swap = dtype[0] == '>';
        grid::Request request;
        request.levels = {grid::levelFromMap(*array, type, map)};
        request.order = target.order;
        request.offsets = layers;
        const auto surface = gridSurface(*patch, spacing);
        const grid::Raster raster{patch->x0, patch->y0, patch->sx, patch->sy, int(size[1]), int(size[0]), patch->kx, patch->ky};
        grid::Result result;
#ifdef VC_READER_CUDA
        if (s.backend == "cuda") {
            if (!target.cuda) target.cuda = std::make_shared<reader_cuda::Reader>(*array, s.tiles.limit, s.device);
            result = grid::cuda::sample(surface, raster, request, {static_cast<reader_cuda::Reader*>(target.cuda.get())});
        } else
#endif
            result = grid::sample(surface, raster, request);
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

SampleResult SampleReader::renderSurface(const Surface& surface, const Pair& start, const Pair& size, const std::vector<double>& offsetsUm)
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
    const auto layers = layerOffsets(*geom, patch && !patch->tile->normals.empty(), offsetsUm);
    for (auto& [name, target] : s.targets) {
        auto out = renderVolume(s, target, patch, size, layers, toReference, *surface.spacingUm);
        for (auto& [key, child] : target.companions) out.masks.emplace_back(key, renderVolume(s, child, patch, size, layers, toReference, *surface.spacingUm).data);
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
struct Triangles {
    std::vector<float> data;  // N x 3 x 5: XYZ, then UV
    std::vector<std::int64_t> ids;
    std::optional<Bounds> bounds;
    Triple base{};
};
// One vertex: position and label-pixel attributes.
using Vertex = std::array<float, 5>;
// Clips XYZ and pixel attributes together to a convex label-pixel rectangle.
std::vector<std::array<Vertex, 3>> clipped(const std::array<Vertex, 3>& triangle, const std::array<float, 2>& lower, const std::array<float, 2>& upper)
{
    std::vector<Vertex> polygon(triangle.begin(), triangle.end());
    for (int axis : {3, 4}) {
        for (const auto& [limit, sign] : {std::pair{lower[std::size_t(axis - 3)], 1.f}, std::pair{upper[std::size_t(axis - 3)], -1.f}}) {
            if (polygon.empty()) return {};
            std::vector<Vertex> result;
            Vertex previous = polygon.back();
            float distance = (previous[std::size_t(axis)] - limit) * sign;
            for (const auto& point : polygon) {
                const float d = (point[std::size_t(axis)] - limit) * sign;
                if ((d >= 0) != (distance >= 0)) {
                    const float t = distance / (distance - d);
                    Vertex crossing;
                    for (std::size_t k = 0; k < 5; ++k) crossing[k] = previous[k] + (point[k] - previous[k]) * t;
                    crossing[std::size_t(axis)] = limit;
                    result.push_back(crossing);
                }
                if (d >= 0) result.push_back(point);
                previous = point;
                distance = d;
            }
            polygon = std::move(result);
        }
    }
    std::vector<std::array<Vertex, 3>> out;
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i) out.push_back({polygon[0], polygon[i], polygon[i + 1]});
    return out;
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
    Affine metric, destination, labelToUv, gridToLabel;
    std::array<double, 3> uvOrigin{}, expansion{};
    std::array<std::int64_t, 3> margin{};
    Pair gridStart{}, gridEnd{};
    std::string key;
    Triple origin{}, shape{};
    double preparationSeconds = 0;
    std::shared_ptr<SurfaceIndex> index;

    Impl(AlignedEngine::Impl& r, const Surface& s, const Pair& patchStart, const Pair& patchSize, const std::string& label, double thicknessUm, double paddingUm);
    std::vector<std::pair<Pair, Pair>> tiles() const;
    std::shared_ptr<SurfaceIndex> spatialIndex();
    std::shared_ptr<const Triangles> triangles(const TileRef& tile, bool clip);
    json metadataFor(const Triple& at, const Triple& extent) const;
    using State = std::array<SampledArray, 7>;
    State newState(const Target& target, const Triple& extent);
    void rasterize(const std::vector<float>& triangleData, const std::vector<std::int64_t>& ids, const std::array<float, 3>& rasterOrigin, const Triple& extent,
                   std::vector<std::pair<std::string, State>>& states, const std::vector<const std::pair<std::string, Target>*>& active);
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
    const auto* registration = std::get_if<VolumeRegistration>(&*r.referenceReader->registration());
    if (!registration || !registration->volumeSpacingUm) bad("projection requires known reference volume physical spacing");
    metric = scaled(*registration->volumeSpacingUm) * r.referenceMap.forward;
    const auto inverse = metric.inverse();
    radius = thickness / 2;
    for (int i = 0; i < 3; ++i) {
        const double norm = std::sqrt(inverse(i, 0) * inverse(i, 0) + inverse(i, 1) * inverse(i, 1) + inverse(i, 2) * inverse(i, 2));
        expansion[std::size_t(i)] = radius * norm;
        margin[std::size_t(i)] = std::int64_t(std::ceil(padding * norm));
    }
    const auto between = r.between(selecting->source, surfaceFrame(surface));
    destination = r.referenceMap.forward.inverse() * between.forward * surface.xyzToVolume->forward;
    labelToUv = pixelAffine(*surfaceOf(selecting->reader->registration())->pixelToUv).forward;
    uvOrigin = labelToUv.apply({double(start[1]), double(start[0]), 0});
    gridToLabel = labelToUv.inverse() * geom->toUv.forward;
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
    const auto before = Clock::now();
    std::array<std::optional<double>, 3> low, high;
    for (const auto& [tileStart, tileSize] : tiles()) {
        const auto boundsKey = "surface-bounds|" + key + "|" + std::to_string(tileStart[0]) + "," + std::to_string(tileStart[1]) + "|1";
        std::optional<Bounds> bounds;
        if (const auto* cached = r.plans.get(boundsKey)) bounds = *std::static_pointer_cast<const Bounds>(*cached);
        else bounds = triangles({tileStart, tileSize}, true)->bounds;
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

std::shared_ptr<SurfaceIndex> SurfaceProjection::Impl::spatialIndex()
{
    // Index the whole registered sheet once, including folds returning to a crop. Bulk
    // coordinate reads are large; index leaves are smaller, and only intersecting leaves
    // are triangulated and read as labels.
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
            const auto firstValid = std::find(tile->valid.begin(), tile->valid.end(), std::uint8_t(1));
            if (firstValid == tile->valid.end()) continue;
            const auto first = std::size_t(firstValid - tile->valid.begin());
            const std::array<double, 3> anchor{std::floor(double(tile->xyz[first * 3])), std::floor(double(tile->xyz[first * 3 + 1])), std::floor(double(tile->xyz[first * 3 + 2]))};
            std::vector<float> shifted(tile->xyz.size());
            for (std::size_t i = 0; i < tile->valid.size(); ++i)
                for (std::size_t a = 0; a < 3; ++a) shifted[i * 3 + a] = tile->xyz[i * 3 + a] - float(anchor[a]);
            const auto moved = transformPoints(shifted.data(), tile->valid.size(), destination, anchor);
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
                                lo[a] = std::min(lo[a], moved.points[at * 3 + a]);
                                hi[a] = std::max(hi[a], moved.points[at * 3 + a]);
                            }
                        }
                    if (!any) continue;
                    Bounds bounds;
                    for (std::size_t a = 0; a < 3; ++a) {
                        bounds[0][a] = double(lo[a]) + double(moved.base[a]);
                        bounds[1][a] = double(hi[a]) + double(moved.base[a]);
                    }
                    entries.push_back({TileRef{Pair{y + iy, x + ix}, leaf}, bounds});
                }
        }
    auto built = std::make_shared<SurfaceIndex>(std::move(entries));
    reader.plans.put(cacheKey, built, count * 2048);
    reader.peakWork = std::max(reader.peakWork, count * 2048 + std::size_t((block + 1) * (block + 1)) * 80);
    return built;
}

std::shared_ptr<const Triangles> SurfaceProjection::Impl::triangles(const TileRef& tile, bool clip)
{
    const auto tileKey = key + "|" + std::to_string(tile.start[0]) + "," + std::to_string(tile.start[1]) + "," + std::to_string(tile.size[0]) + "," + std::to_string(tile.size[1]) + "|" + (clip ? "1" : "0");
    if (const auto* cached = reader.tiles.get(tileKey)) return std::static_pointer_cast<const Triangles>(*cached);
    const auto grid = geom->tile(tile.start, tile.size);
    const auto rows = grid->rows, cols = grid->cols;
    const auto validAt = [&](std::int64_t y, std::int64_t x) { return grid->valid[std::size_t(y * cols + x)] != 0; };
    std::vector<std::pair<std::int64_t, std::int64_t>> cells;
    for (std::int64_t y = 0; y + 1 < rows; ++y)
        for (std::int64_t x = 0; x + 1 < cols; ++x)
            if (validAt(y, x) && validAt(y + 1, x) && validAt(y, x + 1) && validAt(y + 1, x + 1)) cells.emplace_back(y, x);
    auto out = std::make_shared<Triangles>();
    if (cells.empty()) return out;
    const auto xyzAt = [&](std::int64_t y, std::int64_t x) { return &grid->xyz[std::size_t(y * cols + x) * 3]; };
    const float* firstPoint = xyzAt(cells[0].first, cells[0].second);
    const std::array<double, 3> anchor{std::floor(double(firstPoint[0])), std::floor(double(firstPoint[1])), std::floor(double(firstPoint[2]))};
    const auto triangleCount = cells.size() * 2;
    std::vector<float> points(triangleCount * 9), gridPixels(triangleCount * 9, 0.f);
    std::vector<std::int64_t> ids(triangleCount);
    for (std::size_t c = 0; c < cells.size(); ++c) {
        const auto [y, x] = cells[c];
        const std::array<std::array<std::int64_t, 2>, 4> corners{{{y, x}, {y, x + 1}, {y + 1, x + 1}, {y + 1, x}}};
        const std::array<std::array<int, 3>, 2> order{{{0, 1, 2}, {0, 2, 3}}};
        for (std::size_t t = 0; t < 2; ++t) {
            for (std::size_t v = 0; v < 3; ++v) {
                const auto& corner = corners[std::size_t(order[t][v])];
                const float* p = xyzAt(corner[0], corner[1]);
                for (std::size_t a = 0; a < 3; ++a) points[(c * 2 + t) * 9 + v * 3 + a] = p[a] - float(anchor[a]);
                gridPixels[(c * 2 + t) * 9 + v * 3] = float(corner[1]);
                gridPixels[(c * 2 + t) * 9 + v * 3 + 1] = float(corner[0]);
            }
        }
        const auto id = ((y + tile.start[0]) * (geom->shape[1] - 1) + x + tile.start[1]) * 2;
        ids[c * 2] = id;
        ids[c * 2 + 1] = id + 1;
    }
    const auto local = transformPoints(points.data(), triangleCount * 3, destination, anchor);
    const auto pixels = transformPoints(gridPixels.data(), triangleCount * 3, gridToLabel, {double(tile.start[1]), double(tile.start[0]), 0});
    // Clip in patch-local label pixels to preserve boundary ties at large origins.
    const float pixelShiftX = float(pixels.base[0] - start[1]), pixelShiftY = float(pixels.base[1] - start[0]);
    std::vector<std::array<Vertex, 3>> all(triangleCount);
    for (std::size_t t = 0; t < triangleCount; ++t)
        for (std::size_t v = 0; v < 3; ++v) {
            const auto at = (t * 3 + v) * 3;
            all[t][v] = {local.points[at], local.points[at + 1], local.points[at + 2], pixels.points[at] + pixelShiftX, pixels.points[at + 1] + pixelShiftY};
        }
    std::vector<std::array<Vertex, 3>> kept;
    std::vector<std::int64_t> keptIds;
    if (clip) {
        const std::array<float, 2> lower{-.5f, -.5f}, upper{float(size[1]) - .5f, float(size[0]) - .5f};
        for (std::size_t t = 0; t < all.size(); ++t) {
            bool inside = true;
            for (const auto& v : all[t]) inside = inside && v[3] >= lower[0] && v[4] >= lower[1] && v[3] <= upper[0] && v[4] <= upper[1];
            if (inside) {
                kept.push_back(all[t]);
                keptIds.push_back(ids[t]);
            }
        }
        for (std::size_t t = 0; t < all.size(); ++t) {
            bool inside = true;
            for (const auto& v : all[t]) inside = inside && v[3] >= lower[0] && v[4] >= lower[1] && v[3] <= upper[0] && v[4] <= upper[1];
            if (inside) continue;
            for (const auto& piece : clipped(all[t], lower, upper)) {
                kept.push_back(piece);
                keptIds.push_back(ids[t]);
            }
        }
    } else {
        kept = std::move(all);
        keptIds = std::move(ids);
    }
    // Drop degenerate triangles.
    std::vector<std::array<Vertex, 3>> good;
    std::vector<std::int64_t> goodIds;
    for (std::size_t t = 0; t < kept.size(); ++t) {
        const auto& p = kept[t];
        const float a[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]};
        const float b[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
        const float n[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
        if ((n[0] != 0 || n[1] != 0 || n[2] != 0) && std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])) {
            good.push_back(p);
            goodIds.push_back(keptIds[t]);
        }
    }
    if (good.empty()) return out;
    // UV values are local to the selecting patch's mapped origin. All target mappings
    // include this same integer label-pixel origin explicitly.
    Affine uvLinear = labelToUv;
    for (int r = 0; r < 3; ++r) uvLinear(r, 3) = 0;
    std::vector<float> uvIn(good.size() * 9, 0.f);
    for (std::size_t t = 0; t < good.size(); ++t)
        for (std::size_t v = 0; v < 3; ++v) {
            uvIn[(t * 3 + v) * 3] = good[t][v][3];
            uvIn[(t * 3 + v) * 3 + 1] = good[t][v][4];
        }
    const auto uv = transformPoints(uvIn.data(), good.size() * 3, uvLinear);
    out->data.resize(good.size() * 15);
    std::array<float, 3> lo{INFINITY, INFINITY, INFINITY}, hi{-INFINITY, -INFINITY, -INFINITY};
    for (std::size_t t = 0; t < good.size(); ++t)
        for (std::size_t v = 0; v < 3; ++v) {
            auto* slot = &out->data[(t * 3 + v) * 5];
            for (std::size_t a = 0; a < 3; ++a) {
                slot[a] = good[t][v][a];
                lo[a] = std::min(lo[a], slot[a]);
                hi[a] = std::max(hi[a], slot[a]);
            }
            slot[3] = uv.points[(t * 3 + v) * 3] + float(uv.base[0]);
            slot[4] = uv.points[(t * 3 + v) * 3 + 1] + float(uv.base[1]);
        }
    out->ids = std::move(goodIds);
    Bounds bounds;
    for (std::size_t a = 0; a < 3; ++a) {
        bounds[0][a] = double(lo[a]) + double(local.base[a]);
        bounds[1][a] = double(hi[a]) + double(local.base[a]);
    }
    out->bounds = bounds;
    out->base = local.base;
    if (!clip) reader.tiles.put(tileKey, out, out->data.size() * 4 + out->ids.size() * 8 + 128);
    reader.plans.put("surface-bounds|" + key + "|" + std::to_string(tile.start[0]) + "," + std::to_string(tile.start[1]) + "|" + (clip ? "1" : "0"), std::make_shared<Bounds>(bounds), 512);
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

void SurfaceProjection::Impl::rasterize(const std::vector<float>& data, const std::vector<std::int64_t>& ids, const std::array<float, 3>& rasterOrigin, const Triple& extent,
                                        std::vector<std::pair<std::string, State>>& states, const std::vector<const std::pair<std::string, Target>*>& active)
{
    auto& r = reader;
    const auto before = Clock::now();
    const auto splitAll = [&](const std::vector<const std::pair<std::string, Target>*>& which) {
        const auto middle = ids.size() / 2;
        rasterize(std::vector<float>(data.begin(), data.begin() + std::ptrdiff_t(middle * 15)), std::vector<std::int64_t>(ids.begin(), ids.begin() + std::ptrdiff_t(middle)), rasterOrigin, extent, states, which);
        rasterize(std::vector<float>(data.begin() + std::ptrdiff_t(middle * 15), data.end()), std::vector<std::int64_t>(ids.begin() + std::ptrdiff_t(middle), ids.end()), rasterOrigin, extent, states, which);
    };
    std::array<float, 12> metricFloat{};
    for (std::size_t i = 0; i < 12; ++i) metricFloat[i] = float(metric.values[i]);
    std::optional<surface::Plan> plan;
    try {
        plan.emplace(data.data(), ids.data(), ids.size(), extent, metricFloat, float(radius),
                     std::array<float, 3>{float(expansion[0]), float(expansion[1]), float(expansion[2])}, rasterOrigin, r.workBytes / 3);
    } catch (const std::length_error&) {
        if (ids.size() <= 1) throw DecodeBudgetError("raster candidate batch exceeds sampling_work_bytes");
        splitAll(active);
        return;
    }
    r.candidateIndexSeconds += since(before);
    const auto planBytes = sizeof(*plan) + plan->triangles.size() * 4 + plan->ids.size() * 8 + plan->offsets.size() * 8 + plan->candidates.size() * 4;
    r.peakWork = std::max(r.peakWork, planBytes * 3);
    std::vector<float> uv(ids.size() * 6);
    for (std::size_t t = 0; t < ids.size(); ++t)
        for (std::size_t v = 0; v < 3; ++v) {
            uv[(t * 3 + v) * 2] = data[(t * 3 + v) * 5 + 3];
            uv[(t * 3 + v) * 2 + 1] = data[(t * 3 + v) * 5 + 4];
        }
    // Triangles carry UV relative to the selecting label patch origin.
    const auto patchShift = shift(uvOrigin[0], uvOrigin[1], uvOrigin[2]);
    for (const auto* item : active) {
        const auto& [name, target] = *item;
        const auto mapping = target.mapping.forward * patchShift;
        const auto p = pixelPoints(uv.data(), uv.size() / 2, mapping);
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
            if (ids.size() <= 1) throw DecodeBudgetError("one triangle label footprint exceeds sampling_work_bytes");
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
        ElementType type;
        type.kind = target.nativeDtype[1];
        type.itemsize = itemOf(target.nativeDtype);
        if (type.kind == 'f') bad("projection requires categorical integers");
        const auto rasterBefore = Clock::now();
#ifdef VC_READER_CUDA
        if (r.backend == "cuda") {
            const auto ptr = [](SampledArray& a) { return std::static_pointer_cast<device::Buffer>(a.device)->ptr; };
            device::Guard guard(r.device);
            surface::launch(*plan, config, label->device->ptr, type, ptr((*entry)[0]), static_cast<bool*>(ptr((*entry)[1])), static_cast<bool*>(ptr((*entry)[2])),
                            static_cast<float*>(ptr((*entry)[3])), static_cast<std::int64_t*>(ptr((*entry)[4])), static_cast<float*>(ptr((*entry)[5])), static_cast<bool*>(ptr((*entry)[6])), r.device);
            r.candidateUploadBytes += plan->triangles.size() * 4 + plan->ids.size() * 8 + plan->offsets.size() * 8 + plan->candidates.size() * 4;
        } else
#endif
        {
            const auto* labelData = label->host.data.get();
            type.visit_load([&](auto load) {
                using T = decltype(load(static_cast<const std::uint8_t*>(nullptr)));
                for (std::size_t i = 0; i < config.count; ++i)
                    surface::raster(config, plan->triangles.data(), plan->ids.data(), plan->offsets.data(), plan->candidates.data(), reinterpret_cast<const T*>(labelData),
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
    std::vector<TileRef> hits;
    index->query(requestLo, requestHi, expansion, [&](const TileRef& tile) { hits.push_back(tile); });
    for (const auto& tile : hits) {
        const auto boundsKey = "surface-bounds|" + key + "|" + std::to_string(tile.start[0]) + "," + std::to_string(tile.start[1]) + "|0";
        if (const auto* cached = r.plans.get(boundsKey))
            if (outside(*std::static_pointer_cast<const Bounds>(*cached))) continue;
        const auto tris = triangles(tile, false);
        if (!tris->bounds || outside(*tris->bounds)) continue;
        const std::array<float, 3> rasterOrigin{float(requestLo[0] - double(tris->base[0])), float(requestLo[1] - double(tris->base[1])), float(requestLo[2] - double(tris->base[2]))};
        // Reject triangles before label reads and candidate construction: a spatial-index
        // leaf can be much larger than a small training crop.
        std::array<float, 3> lower{}, upper{};
        for (std::size_t a = 0; a < 3; ++a) {
            lower[a] = rasterOrigin[a] - .5f - float(expansion[a]);
            upper[a] = lower[a] + float(extent[2 - a]) + 2 * float(expansion[a]);
            lower[a] = std::nextafter(lower[a], -INFINITY);
            upper[a] = std::nextafter(upper[a], INFINITY);
        }
        std::vector<float> keptData;
        std::vector<std::int64_t> keptIds;
        for (std::size_t t = 0; t < tris->ids.size(); ++t) {
            bool keep = true;
            const float* p = &tris->data[t * 15];
            for (std::size_t a = 0; a < 3; ++a) {
                const float high = std::max({p[a], p[5 + a], p[10 + a]}), low = std::min({p[a], p[5 + a], p[10 + a]});
                keep = keep && high >= lower[a] && low <= upper[a];
            }
            if (keep) {
                keptData.insert(keptData.end(), p, p + 15);
                keptIds.push_back(tris->ids[t]);
            }
        }
        if (keptIds.empty()) continue;
        const Triple rasterExtent = {extent[0], extent[1], extent[2]};
        rasterize(keptData, keptIds, rasterOrigin, rasterExtent, states, targets);
    }
    const auto patchShift = shift(uvOrigin[0], uvOrigin[1], uvOrigin[2]);
    const std::vector<std::int64_t> spatial{extent[0], extent[1], extent[2]};
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
            surface::copy_region_cuda(std::static_pointer_cast<device::Buffer>(source.device)->ptr, std::static_pointer_cast<device::Buffer>(destination.device)->ptr,
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

// ---- geometry alone ----
SurfaceGeometry::SurfaceGeometry(const Surface& surface, AlignedOptions options) : AlignedEngine(std::move(options))
{
    try {
        impl_->geometries["surface-geometry"] = std::make_shared<Geometry>(*impl_, surface);
    } catch (...) {
        close();
        throw;
    }
}
SurfaceGeometry::~SurfaceGeometry() = default;
Pair SurfaceGeometry::shape() const { return impl_->geometries.at("surface-geometry")->shape; }
std::pair<Block, Block> SurfaceGeometry::readPoints(const Pair& start, const Pair& size, const std::vector<double>& offsetsUm)
{
    auto& s = *impl_;
    std::lock_guard lock(s.mutex);
    s.checkOpen();
    checkPatch(start, size);
    if (offsetsUm.empty() || std::any_of(offsetsUm.begin(), offsetsUm.end(), [](double v) { return !std::isfinite(v); })) bad("offsets_um must be nonempty and finite");
    if (std::size_t(size[0] * size[1]) * offsetsUm.size() * 32 > s.workBytes) throw DecodeBudgetError("geometry point request exceeds sampling_work_bytes; request smaller UV tiles");
    auto geom = s.geometries.at("surface-geometry");
    if (!geom->surface.spacingUm) bad("surface physical spacing is unknown");
    const auto layers = std::int64_t(offsetsUm.size());
    Block points = Block::allocate({layers, size[0], size[1], 3}, "<f4");
    auto* out = reinterpret_cast<float*>(points.data.get());
    const auto n = std::size_t(size[0] * size[1]);
    std::fill_n(out, n * 3 * offsetsUm.size(), std::numeric_limits<float>::quiet_NaN());
    if (const auto patch = patchOf(*geom, start, size)) {
        std::vector<float> at(n * 3), step(n * 3);
        const auto surface = gridSurface(*patch, *geom->surface.spacingUm);
        const grid::Raster raster{patch->x0, patch->y0, patch->sx, patch->sy, int(size[1]), int(size[0]), patch->kx, patch->ky};
        surface.resample(raster, at.data(), step.data());
        const auto offsets = layerOffsets(*geom, !patch->tile->normals.empty(), offsetsUm);
        for (std::size_t k = 0; k < offsets.size(); ++k)
            for (std::size_t i = 0; i < n * 3; ++i) out[k * n * 3 + i] = offsets[k] != 0 ? at[i] + step[i] * offsets[k] : at[i];
    }
    Block valid = Block::allocate({layers, size[0], size[1]}, "|b1");
    for (std::size_t i = 0; i < n * offsetsUm.size(); ++i) valid.data[i] = std::isfinite(out[i * 3]) && std::isfinite(out[i * 3 + 1]) && std::isfinite(out[i * 3 + 2]);
    return {std::move(points), std::move(valid)};
}
} // namespace vcr::sources
