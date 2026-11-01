#include "aligned_internal.hpp"

#include <algorithm>
#include <cstring>

#include "../sigv4.hpp"

namespace vcr::sources {
using nlohmann::json;
namespace {
[[noreturn]] void bad(const std::string& message) { throw std::invalid_argument(message); }

std::array<std::string, 2> frameOf(const std::optional<SourceRegistration>& registration)
{
    return std::visit([](const auto& r) -> std::array<std::string, 2> {
        using T = std::decay_t<decltype(r)>;
        if constexpr (std::is_same_v<T, VolumeRegistration>) return {r.sampleId, r.volumeId};
        else return {r.surface.sampleId, r.surface.volumeId};
    }, *registration);
}
bool hasAxis(const std::vector<std::string>& axes, std::string_view axis) { return std::find(axes.begin(), axes.end(), axis) != axes.end(); }
std::int64_t axisSize(const SourceReader& unused, const std::vector<std::string>& axes, const std::vector<std::int64_t>& shape, std::string_view axis, std::int64_t missing)
{
    (void)unused;
    const auto at = std::find(axes.begin(), axes.end(), axis);
    return at == axes.end() ? missing : shape[std::size_t(at - axes.begin())];
}
std::int64_t product(const std::vector<std::int64_t>& values)
{
    std::int64_t out = 1;
    for (const auto v : values) out *= v;
    return out;
}
std::string nativeOf(const std::string& typestr)
{
    auto out = typestr;
    if (out[0] == '>') out[0] = '<';
    return out;
}
std::string floatDtype() { return "<f4"; }
void requireSamplable(const Source& source, const std::string& dtype)
{
    const auto kind = dtype[1];
    const auto item = dtype[2] - '0';
    if (kind == 'f' && item != 4 && item != 8) throw DeclarationTypeError("sampling requires bool, integer, float32 or float64 values");
    if (source.role == "label" && kind == 'f') throw DeclarationTypeError("categorical sampling requires an integer/bool dtype or explicit encoding");
}
template <class F> auto prefixed(const std::string& id, F&& f) -> decltype(f())
{
    try {
        return f();
    } catch (const DeclarationTypeError& e) {
        throw DeclarationTypeError(id + ": " + e.what());
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(id + ": " + e.what());
    }
}
std::size_t itemOf(const std::string& dtype) { return std::size_t(dtype[2] - '0'); }
#ifdef VC_READER_CUDA
device::Storage storageOf(const SampledArray& array) { return std::static_pointer_cast<device::Buffer>(array.device); }
device::Storage upload(const void* data, std::size_t bytes, int id)
{
    device::Guard guard(id);
    auto storage = device::allocate(bytes, id);
    if (bytes) device::check(cudaMemcpyAsync(storage->ptr, data, bytes, cudaMemcpyHostToDevice, device::stream()));
    device::complete();
    return storage;
}
#endif
} // namespace

std::int64_t SampledArray::count() const { return product(shape); }
SampledArray SampledArray::reshaped(std::vector<std::int64_t> newShape) const
{
    if (product(newShape) != count()) bad("reshape changes array size");
    auto out = *this;
    out.shape = std::move(newShape);
    return out;
}
const SampledArray& TargetResult::mask(const std::string& name) const
{
    for (const auto& [key, value] : masks)
        if (key == name) return value;
    throw std::out_of_range("no mask '" + name + "'");
}
const TargetResult& SampleResult::operator[](const std::string& name) const
{
    for (const auto& [key, value] : targets)
        if (key == name) return value;
    throw std::out_of_range("no target '" + name + "'");
}

std::string keyOf(const Triple& value) { return std::to_string(value[0]) + "," + std::to_string(value[1]) + "," + std::to_string(value[2]); }

Block convertBlock(const Block& block, const std::string& dtype)
{
    if (block.dtype == dtype) return block;
    const auto from = elementOf(block.dtype), to = elementOf(dtype);
    if (from.kind == to.kind && from.itemsize == to.itemsize) {
        // Byte order only.
        Block out = Block::allocate(block.shape, dtype);
        for (std::size_t i = 0; i < block.nbytes; i += from.itemsize)
            std::reverse_copy(block.data.get() + i, block.data.get() + i + from.itemsize, out.data.get() + i);
        return out;
    }
    Block out = Block::allocate(block.shape, dtype);
    const auto count = std::size_t(product(block.shape));
    for (std::size_t i = 0; i < count; ++i) {
        const double v = from.load(block.data.get() + i * from.itemsize);
        auto* p = out.data.get() + i * to.itemsize;
        const auto put = [&](auto x) { std::memcpy(p, &x, sizeof x); };
        switch (to.kind) {
        case 'f':
            if (to.itemsize == 4) put(float(v));
            else put(v);
            break;
        case 'b': put(std::uint8_t(v != 0)); break;
        case 'i':
            if (to.itemsize == 1) put(std::int8_t(v));
            else if (to.itemsize == 2) put(std::int16_t(v));
            else if (to.itemsize == 4) put(std::int32_t(v));
            else put(std::int64_t(v));
            break;
        default:
            if (to.itemsize == 1) put(std::uint8_t(v));
            else if (to.itemsize == 2) put(std::uint16_t(v));
            else if (to.itemsize == 4) put(std::uint32_t(v));
            else put(std::uint64_t(v));
        }
    }
    return out;
}

// ---- engine ----
AlignedEngine::Impl::Impl(AlignedOptions o) : options(std::move(o)), tiles(options.samplingCacheBytes), plans(options.planCacheBytes), workBytes(options.samplingWorkBytes)
{
    backend = options.backend;
    if (backend != "cpu" && backend != "cuda") bad("backend must be 'cpu' or 'cuda'");
    device = int(options.device);
    if (options.samplingWorkBytes == 0 || options.device < 0) bad("cache/device settings must be nonnegative and sampling_work_bytes positive");
    if (options.surfaceOutputBytes == 0) bad("surface_output_bytes must be positive");
#ifndef VC_READER_CUDA
    if (backend == "cuda") throw std::runtime_error("CUDA sampling requires a configured CUDA build and device");
#endif
}

void AlignedEngine::Impl::checkOpen() const
{
    if (closed) bad("sample reader is closed");
}

std::shared_ptr<SourceReader> AlignedEngine::Impl::open(const Source& source)
{
    // IDs alone do not identify selected arrays, encodings, or placement.
    const auto key = source.toDict().dump(-1, ' ', false, json::error_handler_t::replace);
    auto& slot = readers[key];
    if (!slot) {
        auto settings = options.reader;
        slot = std::make_shared<SourceReader>(source, std::move(settings));
        opened.push_back(slot);
    }
    return slot;
}

Target AlignedEngine::Impl::samplingTarget(const Source& source, std::shared_ptr<SourceReader> reader, const VolumeTransform& mapping, bool plane, bool planarGrid, std::optional<FieldSpec> field, json metadata)
{
    Target target;
    target.dtype = reader->dtype();
    requireSamplable(source, target.dtype);
    target.nativeDtype = nativeOf(target.dtype);
    target.source = source;
    target.mapping = mapping;
    target.reader = reader;
    const auto& axes = reader->axes();
    const auto& shape = reader->shape();
    target.shape = {axisSize(*reader, axes, shape, "z", 1), axisSize(*reader, axes, shape, "y", 1), axisSize(*reader, axes, shape, "x", 1)};
    target.channels = axisSize(*reader, axes, shape, "c", 1);
    target.plane = plane;
    std::string location = reader->resolved().location;
    if (!source.storage.locations.empty()) {
        // Every plane in order, hashed once: tile lookups stay constant-cost for long stacks.
        Sha256 hash;
        for (const auto& plane_ : source.storage.locations) hash.update(plane_ + "\n");
        location = "stack:";
        for (const auto byte : hash.finish()) location += std::to_string(int(byte)) + ".";
    }
    target.storageKey = location + "|" + encode(reader->resolved().selection).dump() + "|" + json(axes).dump() + "|" + target.dtype + "|" +
                        (source.encoding ? encode(*source.encoding).dump() : std::string("-")) + "|" + reader->fillValue().dump();
    bool integerShift = true;
    bool identityLinear = true;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) identityLinear = identityLinear && mapping.forward(r, c) == (r == c ? 1. : 0.);
    for (int r = 0; r < 3; ++r) integerShift = integerShift && mapping.forward(r, 3) == std::floor(mapping.forward(r, 3));
    target.direct = (!plane || planarGrid) && identityLinear && integerShift;
    target.order = source.role != "label" ? 1 : 0;
    target.field = field;
    if (field) target.fieldState = aligned::make_field(field->matrix, field->translation, field->components, field->normalize, field->normal);
    target.metadata = std::move(metadata);
    return target;
}

Target* AlignedEngine::Impl::find(const std::string& name)
{
    for (auto& [key, target] : targets)
        if (key == name) return &target;
    return nullptr;
}

VolumeTransform AlignedEngine::Impl::valuePlacement(const ValueFrame& value)
{
    if (value.affine) return *value.affine;
    if (value.level == 0) return {};
    if (value.level && catalog) {
        const auto scroll = catalog->scroll(*value.sampleId);
        const auto volume = scroll.volumes().at(*value.volumeId);
        return arrayPlacement(open(catalog::VolumeLevel(volume, *value.level).source())->registration());
    }
    bad("value frame needs an affine or a resolvable volume level");
}

VolumeTransform AlignedEngine::Impl::between(const Source& source, const std::array<std::string, 2>& frame)
{
    const auto& destination = referenceFrame;
    std::optional<VolumeTransform> declared;
    for (const Source* descriptor : std::initializer_list<const Source*>{&source, &reference}) {
        if (!descriptor->query) continue;
        const auto& q = *descriptor->query;
        const auto reg = frameOf(descriptor->registration);
        const std::array<std::string, 2> queryFrame{q.reference.sampleId, q.reference.volumeId};
        if (catalog) {
            std::optional<VolumeTransform> expected;
            if (reg[0] == q.reference.sampleId) expected = catalog->scroll(reg[0]).transform(reg[1], q.reference.volumeId);
            if (!expected || expected->forward != q.transform.forward) bad(descriptor->id + ": query context conflicts with catalog registration");
        }
        std::optional<VolumeTransform> candidate;
        if (reg == frame && queryFrame == destination) candidate = q.transform;
        else if (reg == destination && queryFrame == frame) candidate = VolumeTransform{q.transform.forward.inverse(), {}};
        if (candidate) {
            if (declared && candidate->forward != declared->forward) bad(source.id + ": conflicting descriptor frame relationships");
            declared = candidate;
        }
    }
    std::optional<VolumeTransform> known;
    if (frame == destination) known = VolumeTransform{};
    else if (catalog && frame[0] == destination[0]) known = catalog->scroll(frame[0]).transform(frame[1], destination[1]);
    if (known && declared && known->forward != declared->forward) bad(source.id + ": conflicting frame relationships");
    const auto result = known ? known : declared;
    if (!result)
        bad(source.id + ": unresolved relationship ('" + frame[0] + "', '" + frame[1] + "') -> ('" + destination[0] + "', '" + destination[1] + "'); supply catalog=");
    (void)result->forward.inverse();
    return *result;
}

Target AlignedEngine::Impl::prepare(const Source& source, std::shared_ptr<SourceReader> reader)
{
    return prefixed(source.id, [&]() -> Target {
        const auto placement = arrayPlacement(reader->registration());
        const auto& spatial = reader->spatialAxes();
        const std::vector<std::string> zyx{"z", "y", "x"}, yx{"y", "x"};
        if (spatial != zyx && spatial != yx) bad("sampling requires ZYX or YX spatial axes");
        const auto& registration = std::get<VolumeRegistration>(*reader->registration());
        const bool plane = registration.plane.has_value();
        if ((spatial == yx) != plane) bad("2D sources require an explicit Plane, 3D sources require a volume affine");
        if (!source.role) bad("an explicit sampling role is required");
        requireSamplable(source, reader->dtype());
        const auto betweenFrames = between(source, frameOf(reader->registration()));
        VolumeTransform mapping;
        if (!(placement.forward == referenceMap.forward && betweenFrames.forward == Affine{}))
            mapping = VolumeTransform{placement.forward.inverse() * betweenFrames.forward.inverse() * referenceMap.forward, {}};
        std::optional<FieldSpec> field;
        json metadata(nullptr);
        const auto& axes = reader->axes();
        const auto channels = axisSize(*reader, axes, reader->shape(), "c", 1);
        if (source.role == "vector" || source.role == "normal" || source.role == "coordinate") {
            if (channels != 3 || !source.componentOrder || !source.valueFrame) bad("geometric fields require three channels, component_order and value_frame");
            if (source.role == "vector" && !source.normalize) bad("vectors require an explicit normalize boolean");
            const auto& value = *source.valueFrame;
            Affine transform;
            if (value.kind == "external") {
                if (source.role != "coordinate") bad("external value frames are supported only for coordinates");
                metadata = {{"kind", "external"}, {"name", *value.name}, {"units", *value.units}};
            } else {
                const auto valueMap = valuePlacement(value);
                (void)valueMap.forward.inverse();
                transform = referenceMap.forward.inverse() * between(source, {*value.sampleId, *value.volumeId}).forward * valueMap.forward;
                metadata = {{"kind", "volume"}, {"sample_id", referenceFrame[0]}, {"volume_id", referenceFrame[1]}, {"array_to_level0", referenceMap.forward.json()},
                            {"reference_id", reference.id}, {"selection", plainFields(encode(referenceReader->resolved().selection))}};
            }
            FieldSpec spec;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) spec.matrix[std::size_t(r * 3 + c)] = float(transform(r, c));
            if (source.role == "normal") {
                const auto inverse = transform.inverse();
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c) spec.matrix[std::size_t(r * 3 + c)] = float(inverse(c, r));
            }
            if (source.role == "coordinate")
                for (int r = 0; r < 3; ++r) spec.translation[std::size_t(r)] = float(transform(r, 3));
            const std::string letters = "xyz";
            for (int i = 0; i < 3; ++i) spec.components[std::size_t(i)] = int(letters.find((*source.componentOrder)[std::size_t(i)]));
            spec.normalize = source.role == "normal" || source.normalize.value_or(false);
            spec.normal = source.role == "normal";
            metadata["component_order"] = *source.componentOrder;
            if (value.kind != "external") field = spec;
        }
        return samplingTarget(source, reader, mapping, plane, false, field, metadata);
    });
}

Block AlignedEngine::Impl::canonical(const Block& block, const std::vector<std::string>& axes)
{
    // To channel, depth, rows, columns in native byte order.
    std::vector<int> from;
    std::vector<std::int64_t> shape;
    for (const char* axis : {"c", "z", "y", "x"}) {
        const auto at = std::find(axes.begin(), axes.end(), axis);
        if (at == axes.end()) {
            from.push_back(-1);
            shape.push_back(1);
        } else {
            from.push_back(int(at - axes.begin()));
            shape.push_back(block.shape[std::size_t(at - axes.begin())]);
        }
    }
    bool ordered = true;
    int last = -1;
    for (const int i : from)
        if (i >= 0) {
            ordered = ordered && i > last;
            last = i;
        }
    Block out;
    const auto item = block.itemsize();
    if (ordered) {
        out = block;
        out.shape = shape;
    } else {
        out = Block::allocate(shape, block.dtype);
        std::vector<std::int64_t> strides(block.shape.size());
        std::int64_t stride = std::int64_t(item);
        for (std::size_t i = block.shape.size(); i-- > 0;) {
            strides[i] = stride;
            stride *= block.shape[i];
        }
        const std::array<std::int64_t, 4> step{from[0] >= 0 ? strides[std::size_t(from[0])] : 0, from[1] >= 0 ? strides[std::size_t(from[1])] : 0,
                                               from[2] >= 0 ? strides[std::size_t(from[2])] : 0, from[3] >= 0 ? strides[std::size_t(from[3])] : 0};
        auto* to = out.data.get();
        for (std::int64_t c = 0; c < shape[0]; ++c)
            for (std::int64_t z = 0; z < shape[1]; ++z)
                for (std::int64_t y = 0; y < shape[2]; ++y)
                    for (std::int64_t x = 0; x < shape[3]; ++x, to += item)
                        std::memcpy(to, block.data.get() + c * step[0] + z * step[1] + y * step[2] + x * step[3], item);
    }
    if (out.dtype[0] == '>') return convertBlock(out, nativeOf(out.dtype));
    return out;
}

std::pair<std::shared_ptr<aligned::Plan>, Triple> AlignedEngine::Impl::plan(const Target& target, const Triple& start, const Triple& size, const Triple& offset, const Triple& block, bool withTiles)
{
    const auto& m = target.mapping.forward;
    // Only this small matrix/origin calculation uses double. Splitting before float32
    // preserves e.g. the subvoxel part of 2**26 + .25.
    const auto xyz = m.apply({double(start[2]), double(start[1]), double(start[0])});
    const std::array<double, 3> origin{xyz[2], xyz[1], xyz[0]};
    Triple base{};
    std::array<float, 3> local{};
    for (int a = 0; a < 3; ++a) {
        base[a] = target.plane && a == 0 ? 0 : std::int64_t(std::floor(origin[a]));
        local[a] = float(origin[a] - double(base[a]));
    }
    std::array<float, 9> axes{};
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) axes[std::size_t(a * 3 + b)] = float(m(2 - a, 2 - b));
    const Triple limits = target.plane ? Triple{1, 32, 32} : Triple{16, 16, 16};
    Triple tile{};
    for (int a = 0; a < 3; ++a) tile[a] = std::max<std::int64_t>(1, std::min(target.shape[a], limits[a]));
    std::string key = keyOf(target.shape) + "|" + keyOf(block) + "|" + keyOf(size) + "|" + keyOf(offset) + "|" + keyOf(base) + "|";
    for (const float v : local) key += std::to_string(std::bit_cast<std::uint32_t>(v)) + ",";
    for (const float v : axes) key += std::to_string(std::bit_cast<std::uint32_t>(v)) + ",";
    key += keyOf(tile) + "|" + std::to_string(target.channels) + std::to_string(target.order) + (target.plane ? "p" : "v") + (withTiles ? "t" : "n");
    if (const auto* cached = plans.get(key)) return {std::static_pointer_cast<aligned::Plan>(*cached), tile};
    auto made = std::make_shared<aligned::Plan>(target.shape, block, size, offset, base, local, axes, tile, int(target.channels), target.order, target.plane, withTiles);
    plans.put(key, made, sizeof(aligned::Plan) + made->keys.size() * sizeof(std::int64_t));
    return {std::move(made), tile};
}

std::optional<std::vector<std::shared_ptr<Tile>>> AlignedEngine::Impl::tileInputs(const Target& target, const aligned::Plan& plan, const Triple& tile)
{
    const auto starts = plan.starts();
    const auto tileBytes = std::size_t(tile[0] * tile[1] * tile[2] * target.channels) * itemOf(target.dtype);
    // Source results, canonical copies, and CUDA tiles can overlap.
    const auto working = starts.size() * tileBytes * 3;
    if (working > workBytes) return std::nullopt;
    peakWork = std::max(peakWork, working);
    std::vector<std::shared_ptr<Tile>> inputs(starts.size());
    std::vector<std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>> missing;
    std::vector<std::size_t> positions;
    const auto keyAt = [&](const Triple& start) { return target.storageKey + "|" + keyOf(start) + "|" + keyOf(tile); };
    for (std::size_t i = 0; i < starts.size(); ++i) {
        if (const auto* cached = tiles.get(keyAt(starts[i]))) {
            inputs[i] = std::static_pointer_cast<Tile>(*cached);
            continue;
        }
        if (target.plane) missing.push_back({{starts[i][1], starts[i][2]}, {tile[1], tile[2]}});
        else missing.push_back({{starts[i][0], starts[i][1], starts[i][2]}, {tile[0], tile[1], tile[2]}});
        positions.push_back(i);
    }
    if (!missing.empty()) {
        const auto arrays = target.reader->readMany(missing);
        tileReads += arrays.size();
        for (std::size_t i = 0; i < arrays.size(); ++i) {
            auto value = std::make_shared<Tile>();
            value->host = canonical(arrays[i], target.reader->axes());
#ifdef VC_READER_CUDA
            if (backend == "cuda") {
                value->device = upload(value->host.data.get(), value->host.nbytes, device);
                uploads += value->host.nbytes;
                value->host = Block{};
            }
#endif
            inputs[positions[i]] = value;
            tiles.put(keyAt(starts[positions[i]]), value, tileBytes);
        }
    }
    return inputs;
}

SampledArray AlignedEngine::Impl::empty(std::vector<std::int64_t> shape, const std::string& dtype)
{
    SampledArray out;
    out.dtype = dtype;
    out.shape = std::move(shape);
#ifdef VC_READER_CUDA
    if (backend == "cuda") {
        const auto bytes = std::size_t(out.count()) * itemOf(dtype);
        device::Guard guard(device);
        out.device = device::allocate(bytes, device);
        out.deviceId = device;
        return out;
    }
#endif
    out.host = Block::allocate(out.shape, dtype).data;
    return out;
}

SampledArray AlignedEngine::Impl::fromHost(const Block& block)
{
    SampledArray out;
    out.shape = block.shape;
    out.dtype = block.dtype;
#ifdef VC_READER_CUDA
    if (backend == "cuda") {
        uploads += block.nbytes;
        out.device = upload(block.data.get(), block.nbytes, device);
        out.deviceId = device;
        return out;
    }
#endif
    out.host = block.data;
    return out;
}

void AlignedEngine::Impl::execute(const Target& target, const Triple& start, const Triple& size, const Triple& offset, const Triple& block, SampledArray& output, SampledArray& valid)
{
    auto [plan_, tile] = plan(target, start, size, offset, block);
    const auto inputs = tileInputs(target, *plan_, tile);
    if (!inputs) {
        int axis = 0;
        for (int a = 1; a < 3; ++a)
            if (block[a] > block[axis]) axis = a;
        if (block[axis] <= 1) throw DecodeBudgetError(target.source.id + ": one sampling stencil exceeds sampling_work_bytes");
        auto left = block, right = block, after = offset;
        left[axis] /= 2;
        right[axis] -= left[axis];
        after[axis] += left[axis];
        execute(target, start, size, offset, left, output, valid);
        execute(target, start, size, after, right, output, valid);
        return;
    }
    const auto fill = elementBytes(target.reader->fillValue(), target.nativeDtype);
    const auto type = elementOf(target.nativeDtype);
#ifdef VC_READER_CUDA
    if (backend == "cuda") {
        std::vector<const void*> pointers;
        for (const auto& t : *inputs) pointers.push_back(t->device->ptr);
        device::Guard guard(device);
        auto grid = plan_->grid;
        grid.field = target.fieldState;
        aligned::launch(grid, plan_->keys, pointers, fill.data(), type, storageOf(output)->ptr, static_cast<bool*>(storageOf(valid)->ptr), device);
        device::complete();
        return;
    }
#endif
    std::vector<const void*> pointers;
    for (const auto& t : *inputs) pointers.push_back(t->host.data.get());
    plan_->cpu(pointers, fill.data(), type, output.host.get(), reinterpret_cast<bool*>(valid.host.get()), target.fieldState);
}

namespace {
// Bounded affine work even when a request contains millions of voxels.
template <class F> void forBlocks(const Triple& size, F&& f)
{
    for (std::int64_t z = 0; z < size[0]; z += 16)
        for (std::int64_t y = 0; y < size[1]; y += 32)
            for (std::int64_t x = 0; x < size[2]; x += 64)
                f(Triple{z, y, x}, Triple{std::min<std::int64_t>(size[0] - z, 16), std::min<std::int64_t>(size[1] - y, 32), std::min<std::int64_t>(size[2] - x, 64)});
}
} // namespace

#ifdef VC_READER_CUDA
void AlignedEngine::Impl::cudaBlocks(const Target& target, const Triple& start, const Triple& size, SampledArray& output, SampledArray& valid)
{
    const auto fill = elementBytes(target.reader->fillValue(), target.nativeDtype);
    const auto type = elementOf(target.nativeDtype);
    std::vector<std::shared_ptr<aligned::Plan>> batch;
    std::vector<std::vector<std::shared_ptr<Tile>>> inputs;
    std::size_t working = 0;
    const auto flush = [&] {
        if (batch.empty()) return;
        std::vector<aligned::Grid> grids;
        std::vector<std::vector<std::int64_t>> keys;
        std::vector<std::vector<const void*>> pointers;
        for (std::size_t i = 0; i < batch.size(); ++i) {
            auto grid = batch[i]->grid;
            grid.field = target.fieldState;
            grids.push_back(grid);
            keys.push_back(batch[i]->keys);
            pointers.emplace_back();
            for (const auto& t : inputs[i]) pointers.back().push_back(t->device->ptr);
        }
        device::Guard guard(device);
        aligned::launch_many(grids, keys, pointers, fill.data(), type, storageOf(output)->ptr, static_cast<bool*>(storageOf(valid)->ptr), device);
        batch.clear();
        inputs.clear();
    };
    forBlocks(size, [&](const Triple& offset, const Triple& block) {
        auto [plan_, tile] = plan(target, start, size, offset, block);
        const auto needed = plan_->starts().size() * std::size_t(tile[0] * tile[1] * tile[2] * target.channels) * itemOf(target.nativeDtype) * 3;
        if (needed + working > workBytes || batch.size() == 64) {
            flush();
            working = 0;
        }
        auto tilesFor = tileInputs(target, *plan_, tile);
        if (!tilesFor) {
            execute(target, start, size, offset, block, output, valid);
        } else {
            batch.push_back(plan_);
            inputs.push_back(std::move(*tilesFor));
            working += needed;
            peakWork = std::max(peakWork, working);
        }
    });
    flush();
}
#endif

std::vector<std::pair<SampledArray, SampledArray>> AlignedEngine::Impl::sampleMany(const Target& target, const std::vector<std::pair<Triple, Triple>>& regions)
{
    std::optional<std::vector<Block>> arrays;
    Triple shift{};
    if (target.direct) {
        for (int a = 0; a < 3; ++a) shift[a] = std::int64_t(target.mapping.forward(2 - a, 3));
        std::vector<std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>> shifted;
        for (const auto& [start, size] : regions) {
            const Triple at{start[0] + shift[0], start[1] + shift[1], start[2] + shift[2]};
            if (target.plane) shifted.push_back({{at[1], at[2]}, {size[1], size[2]}});
            else shifted.push_back({{at[0], at[1], at[2]}, {size[0], size[1], size[2]}});
        }
        arrays = target.reader->readMany(shifted);
    }
    // A uint8 volume read directly samples through the array's own float32 path.
    std::vector<Block> affine;
    bool haveAffine = false;
    Array* directArray = nullptr;
    const std::vector<std::string> zyx{"z", "y", "x"};
    if (backend == "cpu" && !target.direct && !target.plane && target.dtype == "|u1" && target.reader->spatialAxes() == zyx)
        directArray = target.reader->directArray();
    if (directArray) {
        struct Prepared {
            FloatAffine grid;
            std::size_t count;
            std::vector<std::int64_t> shape;
        };
        std::vector<Prepared> prepared;
        bool inRange = true;
        const auto& m = target.mapping.forward;
        for (const auto& [start, size] : regions) {
            const auto xyz = m.apply({double(start[2]), double(start[1]), double(start[0])});
            const std::array<double, 3> origin{xyz[2], xyz[1], xyz[0]};
            for (const double v : origin) inRange = inRange && v >= 0 && v <= double(1LL << 30);
            if (!inRange) break;
            const auto local = localOrigin(origin.data());
            Prepared p;
            for (int a = 0; a < 3; ++a) {
                p.grid.shape[std::size_t(a)] = std::size_t(size[a]);
                p.grid.origin[std::size_t(a)] = local.origin[std::size_t(a)];
                p.grid.offset[std::size_t(a)] = local.base[std::size_t(a)];
                for (int b = 0; b < 3; ++b) p.grid.axes[std::size_t(a * 3 + b)] = float(m(2 - a, 2 - b));
            }
            p.count = std::size_t(size[0] * size[1] * size[2]);
            p.shape = {size[0], size[1], size[2]};
            prepared.push_back(std::move(p));
        }
        if (inRange) {
            haveAffine = true;
            for (const auto& p : prepared) {
                auto block = Block::allocate(p.shape, target.order ? floatDtype() : std::string("|u1"));
                if (p.count) directArray->sample_u8_float32(p.grid, p.count, target.order, block.data.get());
                affine.push_back(std::move(block));
            }
        }
    }
    std::vector<std::pair<SampledArray, SampledArray>> results;
    for (std::size_t index = 0; index < regions.size(); ++index) {
        const auto& [start, size] = regions[index];
        std::string dtype = target.direct || !target.order ? target.nativeDtype : floatDtype();
        if (!target.metadata.is_null()) dtype = floatDtype();
        const std::vector<std::int64_t> shape{target.channels, size[0], size[1], size[2]};
        SampledArray valid = empty({1, size[0], size[1], size[2]}, "|b1");
        SampledArray output;
        if (arrays) {
            const auto canonicalBlock = convertBlock(canonical((*arrays)[index], target.reader->axes()), dtype);
            output = fromHost(canonicalBlock);
            const Triple at{start[0] + shift[0], start[1] + shift[1], start[2] + shift[2]};
#ifdef VC_READER_CUDA
            if (backend == "cuda") {
                device::Guard guard(device);
                aligned::launch_box(target.shape, at, size, static_cast<bool*>(storageOf(valid)->ptr));
                device::complete();
            } else
#endif
            {
                auto* flags = valid.host.get();
                std::memset(flags, 0, std::size_t(valid.count()));
                std::array<std::int64_t, 3> low{}, high{};
                for (int a = 0; a < 3; ++a) {
                    low[a] = std::max<std::int64_t>(0, -at[a]);
                    high[a] = std::max<std::int64_t>(0, std::min(size[a], target.shape[a] - at[a]));
                }
                for (std::int64_t z = low[0]; z < high[0]; ++z)
                    for (std::int64_t y = low[1]; y < high[1]; ++y)
                        if (high[2] > low[2]) std::memset(flags + ((z * size[1] + y) * size[2] + low[2]), 1, std::size_t(high[2] - low[2]));
            }
        } else if (haveAffine) {
            output.shape = shape;
            output.dtype = affine[index].dtype;
            output.host = affine[index].data;
            forBlocks(size, [&](const Triple& offset, const Triple& block) {
                auto [plan_, tile] = plan(target, start, size, offset, block, false);
                (void)tile;
                const auto& g = plan_->grid;
                auto* flags = reinterpret_cast<bool*>(valid.host.get());
                for (std::size_t i = 0; i < g.count; ++i) flags[aligned::output_index(g, i)] = aligned::supported(g, i);
            });
        } else {
            output = empty(shape, dtype);
#ifdef VC_READER_CUDA
            if (backend == "cuda") cudaBlocks(target, start, size, output, valid);
            else
#endif
                forBlocks(size, [&](const Triple& offset, const Triple& block) { execute(target, start, size, offset, block, output, valid); });
        }
        if (target.field && (arrays || haveAffine)) {
            const auto& f = target.fieldState;
#ifdef VC_READER_CUDA
            if (backend == "cuda") {
                device::Guard guard(device);
                aligned::launch_field(static_cast<float*>(storageOf(output)->ptr), static_cast<bool*>(storageOf(valid)->ptr), std::size_t(valid.count()), f);
                device::complete();
            } else
#endif
            {
                auto* data = reinterpret_cast<float*>(output.host.get());
                auto* flags = reinterpret_cast<bool*>(valid.host.get());
                for (std::size_t i = 0; i < std::size_t(valid.count()); ++i)
                    if (flags[i]) aligned::field(data, flags, std::size_t(valid.count()), i, f);
            }
        }
        if (backend == "cpu" && !target.order && output.dtype != target.dtype && target.dtype[0] == '>') {
            Block block;
            block.shape = output.shape;
            block.dtype = output.dtype;
            block.data = output.host;
            block.nbytes = std::size_t(output.count()) * itemOf(output.dtype);
            const auto converted = convertBlock(block, target.dtype);
            output.host = converted.data;
            output.dtype = target.dtype;
        }
        results.emplace_back(std::move(output), std::move(valid));
    }
    return results;
}

void AlignedEngine::Impl::pointBlock(const Target& target, const float* points, std::size_t n, const Triple& base, SampledArray& out, SampledArray& valid, std::size_t& cursor)
{
    const Triple limits = target.plane ? Triple{1, 32, 32} : Triple{16, 16, 16};
    Triple tile{};
    for (int a = 0; a < 3; ++a) tile[a] = std::max<std::int64_t>(1, std::min(target.shape[a], limits[a]));
    aligned::PointPlan plan_(target.shape, base, points, n, tile, int(target.channels), target.order, target.plane);
    const auto inputs = tileInputs(target, plan_, tile);
    if (!inputs) {
        if (n <= 1) throw DecodeBudgetError("one point stencil exceeds sampling_work_bytes");
        pointBlock(target, points, n / 2, base, out, valid, cursor);
        pointBlock(target, points + 3 * (n / 2), n - n / 2, base, out, valid, cursor);
        return;
    }
    const std::string dtype = target.order ? floatDtype() : target.nativeDtype;
    const auto fill = elementBytes(target.reader->fillValue(), target.nativeDtype);
    const auto type = elementOf(target.nativeDtype);
    const auto item = itemOf(dtype);
    const auto total = std::size_t(out.shape[1]);
    const auto channels = std::size_t(target.channels);
    SampledArray block = empty({target.channels, std::int64_t(n)}, dtype);
    SampledArray support = empty({1, std::int64_t(n)}, "|b1");
#ifdef VC_READER_CUDA
    if (backend == "cuda") {
        std::vector<const void*> pointers;
        for (const auto& t : *inputs) pointers.push_back(t->device->ptr);
        device::Guard guard(device);
        auto grid = plan_.grid;
        grid.field = target.fieldState;
        aligned::launch(grid, plan_.keys, pointers, fill.data(), type, storageOf(block)->ptr, static_cast<bool*>(storageOf(support)->ptr), device);
        device::complete();
        aligned::copy_cuda(storageOf(block)->ptr, storageOf(out)->ptr, channels, n, total, cursor, item, device);
        aligned::copy_cuda(storageOf(support)->ptr, storageOf(valid)->ptr, 1, n, total, cursor, 1, device);
        cursor += n;
        return;
    }
#endif
    std::vector<const void*> pointers;
    for (const auto& t : *inputs) pointers.push_back(t->host.data.get());
    plan_.cpu(pointers, fill.data(), type, block.host.get(), reinterpret_cast<bool*>(support.host.get()), target.fieldState);
    for (std::size_t c = 0; c < channels; ++c)
        std::memcpy(out.host.get() + (c * total + cursor) * item, block.host.get() + c * n * item, n * item);
    std::memcpy(valid.host.get() + cursor, support.host.get(), n);
    cursor += n;
}

std::pair<SampledArray, SampledArray> AlignedEngine::Impl::samplePoints(const Target& target, const std::vector<float>& points, const Triple& base)
{
    const std::size_t n = points.size() / 3;
    const std::string dtype = target.order ? floatDtype() : target.nativeDtype;
    SampledArray out = empty({target.channels, std::int64_t(n)}, dtype), valid = empty({1, std::int64_t(n)}, "|b1");
    std::size_t cursor = 0;
    for (std::size_t start = 0; start < n; start += 32768) {
        const auto count = std::min<std::size_t>(32768, n - start);
        pointBlock(target, points.data() + start * 3, count, base, out, valid, cursor);
    }
    if (backend == "cpu" && !target.order && target.dtype != out.dtype) {
        Block block;
        block.shape = out.shape;
        block.dtype = out.dtype;
        block.data = out.host;
        block.nbytes = std::size_t(out.count()) * itemOf(out.dtype);
        const auto converted = convertBlock(block, target.dtype);
        out.host = converted.data;
        out.dtype = target.dtype;
    }
    return {std::move(out), std::move(valid)};
}

AlignedEngine::AlignedEngine(AlignedOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
AlignedEngine::~AlignedEngine()
{
    try {
        close();
    } catch (...) {
    }
}
const std::string& AlignedEngine::backend() const { return impl_->backend; }
const nlohmann::json& AlignedEngine::metadata() const { return impl_->metadata; }
std::int64_t AlignedEngine::device() const { return impl_->device; }
bool AlignedEngine::closed() const { return impl_->closed; }
std::vector<std::string> AlignedEngine::targetNames() const
{
    std::vector<std::string> out;
    for (const auto& [name, target] : impl_->targets) out.push_back(name);
    return out;
}
std::int64_t AlignedEngine::targetChannels(const std::string& name) const
{
    for (const auto& [key, target] : impl_->targets)
        if (key == name) return target.channels;
    throw std::out_of_range("no target '" + name + "'");
}
AlignedStats AlignedEngine::stats()
{
    auto& s = *impl_;
    s.checkOpen();
    std::lock_guard lock(s.mutex);
    AlignedStats out;
    out.tileCacheBytes = s.tiles.bytes;
    out.tileCacheHits = s.tiles.hits;
    out.planCacheBytes = s.plans.bytes;
    out.planCacheHits = s.plans.hits;
    out.tileReads = s.tileReads;
    out.uploadedBytes = s.uploads;
    out.peakTileWorkBytes = s.peakWork;
    for (const auto& [name, target] : s.targets) out.sources.emplace_back(name, target.reader->stats());
    out.hasSurface = s.usedSurface || !s.geometries.empty();
    if (out.hasSurface)
        for (const auto& [name, target] : s.surfaceTargets) out.sources.emplace_back(name, target.reader->stats());
    out.geometryReadBytes = s.geometryReadBytes;
    out.candidateUploadBytes = s.candidateUploadBytes;
    out.largestGeometryCells = s.largestGeometryCells;
    out.preparationSeconds = s.preparationSeconds;
    out.candidateIndexSeconds = s.candidateIndexSeconds;
    out.rasterCallSeconds = s.rasterCallSeconds;
    out.outputAllocationSeconds = s.outputAllocationSeconds;
    return out;
}
void AlignedEngine::close()
{
    auto& s = *impl_;
    std::lock_guard lock(s.mutex);
    if (s.closed) return;
    // Device readers of renders hold their arrays: drop them first.
    for (auto& [name, target] : s.targets) {
        target.cuda.reset();
        for (auto& [key, child] : target.companions) child.cuda.reset();
    }
    s.geometries.clear();
    for (auto& reader : s.opened) reader->close();
    s.opened.clear();
    s.readers.clear();
    s.tiles.clear();
    s.plans.clear();
    s.closed = true;
}

// ---- reference grid ----
namespace {
void prepareSampleReader(AlignedEngine::Impl& s, const Source& reference, const std::vector<std::pair<std::string, Source>>& targets)
{
    if (reference.role != "image") bad("reference must be a VolumeLevel or registered volumetric image Source");
    s.reference = reference;
    auto ref = s.open(reference);
    s.referenceMap = arrayPlacement(ref->registration());
    const std::vector<std::string> zyx{"z", "y", "x"};
    const auto* volume = std::get_if<VolumeRegistration>(&*ref->registration());
    if (ref->spatialAxes() != zyx || (volume && volume->plane)) bad("reference must have three volume spatial axes");
    s.referenceReader = ref;
    s.referenceFrame = frameOf(ref->registration());
    std::vector<std::pair<std::string, Source>> all{{"image", reference}};
    for (const auto& [name, source] : targets) {
        if (name == "image" || name.empty()) bad("target names must be nonempty strings; image is reserved");
        all.emplace_back(name, source);
    }
    for (const auto& [name, source] : all) {
        auto reader = s.open(source);
        if (reader->registration() && std::holds_alternative<SurfaceRegistration>(*reader->registration())) {
            s.surfaceTargets.emplace_back(name, prepareSurfaceTarget(s, source, reader));
            continue;
        }
        auto target = s.prepare(source, reader);
        for (const auto& companion : source.companions) target.companions.emplace_back(companion.name, s.prepare(*companion.source, reader->companion(companion.name)));
        s.targets.emplace_back(name, std::move(target));
    }
    json frames = json::object();
    for (const auto& [name, target] : s.targets)
        if (!target.metadata.is_null()) frames[name] = target.metadata;
    s.metadata = frames.empty() ? json(nullptr) : json{{"value_frames", frames}};
}
std::pair<Triple, Triple> checkRegion(const Triple& start, const Triple& size)
{
    for (const auto n : size)
        if (n < 0 || n > (1 << 24)) bad("start/size must be ZYX triples, with sizes in [0, 2**24]");
    for (const auto n : start)
        if (std::abs(n) > (1LL << 40)) bad("crop starts must be within +/- 2**40");
    return {start, size};
}
} // namespace

SampleReader::SampleReader(Source reference, std::vector<std::pair<std::string, Source>> targets, std::optional<catalog::Catalog> catalog, AlignedOptions options) : AlignedEngine(std::move(options))
{
    impl_->catalog = std::move(catalog);
    try {
        prepareSampleReader(*impl_, reference, targets);
    } catch (...) {
        close();
        throw;
    }
}
SampleReader::SampleReader(const catalog::VolumeLevel& reference, std::vector<std::pair<std::string, Source>> targets, std::optional<catalog::Catalog> catalog, AlignedOptions options) : AlignedEngine(std::move(options))
{
    const auto& index = reference.volume.index();
    impl_->catalog = catalog::Catalog(index->published() ? index->published() : index);
    if (catalog && catalog->manifest().raw != impl_->catalog->manifest().raw)
        throw std::invalid_argument("VolumeLevel and explicit catalog must belong to the same catalog");
    try {
        prepareSampleReader(*impl_, reference.source(), targets);
    } catch (...) {
        close();
        throw;
    }
}
SampleReader::~SampleReader() = default;

catalog::OpenedGrid openedGrid(SourceReader& reader)
{
    const auto& resolved = reader.resolved();
    catalog::OpenedGrid grid;
    grid.storageFromArray = resolved.storageFromArray;
    grid.storageUnits = resolved.storageUnits;
    grid.registration = resolved.registration;
    grid.selection = resolved.selection;
    const auto& axes = reader.axes();
    const auto& shape = reader.shape();
    const char* names[] = {"z", "y", "x"};
    for (int i = 0; i < 3; ++i) {
        const auto axis = std::find(axes.begin(), axes.end(), names[i]);
        if (axis == axes.end()) throw std::invalid_argument(std::string("source has no ") + names[i] + " axis");
        grid.shapeZYX[i] = shape[std::size_t(axis - axes.begin())];
    }
    return grid;
}
std::array<std::int64_t, 3> SampleReader::shape() const { return impl_->find("image")->shape; }
const VolumeTransform& SampleReader::referenceMap() const { return impl_->referenceMap; }
bool SampleReader::hasTarget(const std::string& name) const { return impl_->find(name) != nullptr; }
SampleResult SampleReader::read(const std::array<std::int64_t, 3>& start, const std::array<std::int64_t, 3>& size) { return readMany({{start, size}}).at(0); }
std::vector<SampleResult> SampleReader::readMany(const std::vector<std::pair<std::array<std::int64_t, 3>, std::array<std::int64_t, 3>>>& regions)
{
    auto& s = *impl_;
    std::lock_guard lock(s.mutex);
    s.checkOpen();
    if (!s.surfaceTargets.empty()) bad("surface targets require explicit render_surface or project_surface");
    std::vector<std::pair<Triple, Triple>> checked;
    for (const auto& [start, size] : regions) checked.push_back(checkRegion(start, size));
    std::vector<SampleResult> results(checked.size());
    for (auto& result : results) result.metadata = s.metadata;
    for (const auto& [name, target] : s.targets) {
        auto values = s.sampleMany(target, checked);
        for (std::size_t i = 0; i < results.size(); ++i) {
            TargetResult out;
            out.data = std::move(values[i].first);
            out.valid = std::move(values[i].second);
            results[i].targets.emplace_back(name, std::move(out));
        }
        for (const auto& [mask, child] : target.companions) {
            auto masks = s.sampleMany(child, checked);
            for (std::size_t i = 0; i < results.size(); ++i) results[i].targets.back().second.masks.emplace_back(mask, std::move(masks[i].first));
        }
    }
    return results;
}
std::pair<SampledArray, SampledArray> SampleReader::sampleTarget(const std::string& name, const std::array<std::int64_t, 3>& start, const std::array<std::int64_t, 3>& size)
{
    auto& s = *impl_;
    std::lock_guard lock(s.mutex);
    s.checkOpen();
    auto* target = s.find(name);
    if (!target) throw std::out_of_range("no target '" + name + "'");
    return s.sampleMany(*target, {{start, size}}).at(0);
}

// ---- flattened pairs ----
namespace {
Affine xyOf(const Affine& matrix)
{
    Affine out;
    out(0, 0) = matrix(0, 0), out(0, 1) = matrix(0, 1), out(0, 3) = matrix(0, 3);
    out(1, 0) = matrix(1, 0), out(1, 1) = matrix(1, 1), out(1, 3) = matrix(1, 3);
    return out;
}
std::array<std::int64_t, 2> shapeYX(SourceReader& reader)
{
    const auto& axes = reader.axes();
    const auto& shape = reader.shape();
    return {axisSize(reader, axes, shape, "y", 1), axisSize(reader, axes, shape, "x", 1)};
}
Affine relativeGrid(SourceReader& base, SourceReader& selected)
{
    const auto& a = base.resolved();
    const auto& b = selected.resolved();
    if (a.location == b.location && a.selection == b.selection) return {};
    if (!a.storageFromArray || !b.storageFromArray || a.storageUnits != b.storageUnits) bad("selected pixel grid has no known transform to its base grid");
    const auto original = xyOf(a.storageFromArray->forward), current = xyOf(b.storageFromArray->forward);
    return original == current ? Affine{} : original.inverse() * current;
}
Affine companionToLabel(const std::optional<SourceRegistration>& label, const std::optional<SourceRegistration>& mask)
{
    const auto* labelSurface = label ? std::get_if<SurfaceRegistration>(&*label) : nullptr;
    const auto* maskSurface = mask ? std::get_if<SurfaceRegistration>(&*mask) : nullptr;
    if (labelSurface && maskSurface) {
        if (labelSurface->surface.id != maskSurface->surface.id || !labelSurface->pixelToUv || !maskSurface->pixelToUv) bad("companion surface registration has no shared pixel frame");
        return pixelAffine(*labelSurface->pixelToUv).forward.inverse() * pixelAffine(*maskSurface->pixelToUv).forward;
    }
    const auto* labelVolume = label ? std::get_if<VolumeRegistration>(&*label) : nullptr;
    const auto* maskVolume = mask ? std::get_if<VolumeRegistration>(&*mask) : nullptr;
    if (labelVolume && maskVolume) {
        if (labelVolume->sampleId != maskVolume->sampleId || labelVolume->volumeId != maskVolume->volumeId) bad("flattened companion registration requires a shared volume frame");
        const auto mapping = arrayPlacement(label).forward.inverse() * arrayPlacement(mask).forward;
        if (std::abs(mapping(2, 0)) > 1e-7 || std::abs(mapping(2, 1)) > 1e-7 || std::abs(mapping(2, 3)) > 1e-7) bad("flattened companion registration must lie in the label plane");
        return xyOf(mapping);
    }
    bad("flattened companion needs same_grid or complete pixel registration");
}
Source withoutPlacement(Source source)
{
    source.registration.reset();
    source.companions.clear();
    return finished(std::move(source));
}
Source withoutCompanions(Source source)
{
    source.companions.clear();
    return finished(std::move(source));
}
Target planarTarget(AlignedEngine::Impl& s, const Source& source, const std::shared_ptr<SourceReader>& base, const Affine& baseToImage, std::int64_t preferredLevel)
{
    auto selected = base;
    Affine selectedToImage = baseToImage;
    if (!(selectedToImage == Affine{}) && source.storage.pyramid) {
        // Header-declared levels are candidates, not an assumed shared level count.
        std::vector<std::int64_t> levels;
        for (std::size_t i = 0; i < base->zarrDatasets(); ++i) levels.push_back(std::int64_t(i));
        if (const auto at = std::find(levels.begin(), levels.end(), preferredLevel); at != levels.end()) {
            levels.erase(at);
            levels.insert(levels.begin(), preferredLevel);
        }
        for (const auto level : levels) {
            auto candidate = s.open(selectLevel(source, level));
            Affine mapping;
            try {
                mapping = baseToImage * relativeGrid(*base, *candidate);
            } catch (const std::invalid_argument&) {
                continue;
            }
            if (mapping == Affine{}) {
                selected = candidate;
                selectedToImage = Affine{};
                break;
            }
        }
    }
    return s.samplingTarget(source, selected, VolumeTransform{selectedToImage.inverse(), {}}, true, true);
}
} // namespace

PairedReader::PairedReader(const Source& label, const Source& image, const catalog::Pairing& pairing, std::int64_t level, const LayerChoice& layerChoice, AlignedOptions options) : AlignedEngine(std::move(options))
{
    auto& s = *impl_;
    try {
        if (label.role != "label" || image.role != "image") bad("paired reading requires label and image source roles");
        // The original registration remains on public descriptors. A flattened read uses
        // its explicit pixel relationship, without projecting to 3D.
        const auto rawImage = withoutPlacement(image), rawLabel = withoutPlacement(label);
        auto imageBase = s.open(selectLevel(rawImage, 0));
        auto selectedImage = s.open(selectLevel(rawImage, level));
        if (selectedImage->spatialAxes() != std::vector<std::string>{"z", "y", "x"}) bad("paired image must have sampled layers and YX spatial axes");
        shape_ = shapeYX(*selectedImage);
        layerDepth_ = axisSize(*selectedImage, selectedImage->axes(), selectedImage->shape(), "z", 1);
        const auto layers = layerChoice ? layerChoice(layerDepth_) : Layers{};
        layerStart_ = layers.start;
        layerCount_ = layers.count.value_or(layerDepth_ - layers.start);
        dropLayer_ = layers.drop;
        s.targets.emplace_back("image", s.samplingTarget(rawImage, selectedImage, VolumeTransform{}));
        auto labelBase = s.open(selectLevel(rawLabel, 0));
        if (labelBase->spatialAxes() != std::vector<std::string>{"y", "x"}) bad("paired labels must have YX spatial axes");
        if (!pairing.pixelAffine) bad("label-pixel-to-image-pixel mapping is unresolved");
        auto mapping = pixelAffine(*pairing.pixelAffine).forward;
        // Published target-canvas relationships are expressed in the shared storage frame
        // when both arrays publish dimensional transforms.
        const auto& a = imageBase->resolved();
        const auto& b = labelBase->resolved();
        const bool shared = a.storageFromArray && b.storageFromArray;
        const auto published = [&] { return pairing.evidence.starts_with("published") || pairing.evidence.starts_with("segment/"); };
        if (shared && a.storageUnits != b.storageUnits && pairing.evidence != "explicit user pixel mapping") bad("paired pixel grids have incompatible storage units");
        if (shared && a.storageUnits == b.storageUnits) {
            const auto original = xyOf(a.storageFromArray->forward), current = xyOf(b.storageFromArray->forward);
            const Affine observed = original == current ? Affine{} : original.inverse() * current;
            if (published()) mapping = observed;
            else if (pairing.evidence == "explicit same grid" && !(observed == mapping)) bad("storage transforms contradict same_grid pairing");
        } else if (published() && shapeYX(*imageBase) != shapeYX(*labelBase)) {
            bad("published pixel grids differ without storage transforms; pairing is unresolved");
        }
        if (pairing.evidence == "explicit same grid" && shapeYX(*imageBase) != shapeYX(*labelBase)) bad("same_grid pairing has different base shapes");
        const auto imageFromSelected = relativeGrid(*imageBase, *selectedImage);
        const auto labelToSelected = imageFromSelected.inverse() * mapping;
        s.targets.emplace_back("label", planarTarget(s, rawLabel, labelBase, labelToSelected, level));
        for (const auto& companion : label.companions) {
            const auto maskSource = withoutPlacement(*companion.source);
            auto base = s.open(selectLevel(maskSource, 0));
            if (base->spatialAxes() != std::vector<std::string>{"y", "x"}) bad("flattened masks require YX spatial axes");
            if (hasAxis(base->axes(), "c") && axisSize(*base, base->axes(), base->shape(), "c", 1) != 1) bad("companion masks require one channel");
            Affine maskToLabel;
            if (companion.sameGrid) {
                if (shapeYX(*base) != shapeYX(*labelBase)) bad(companion.name + ": companion base shape contradicts same_grid");
                const auto& x = base->resolved();
                const auto& y = labelBase->resolved();
                if (x.storageFromArray && y.storageFromArray && (x.storageUnits != y.storageUnits || !(xyOf(x.storageFromArray->forward) == xyOf(y.storageFromArray->forward))))
                    bad(companion.name + ": companion storage transforms contradict same_grid");
            } else {
                // Resolve each declared grid before comparing placements.
                auto fullLabel = s.open(selectLevel(withoutCompanions(label), 0));
                auto fullMask = s.open(selectLevel(withoutCompanions(*companion.source), 0));
                maskToLabel = companionToLabel(fullLabel->registration(), fullMask->registration());
            }
            s.masks.emplace_back(companion.name, planarTarget(s, maskSource, base, labelToSelected * maskToLabel, level));
        }
        s.metadata = {{"image_id", image.id}, {"label_id", label.id}, {"image_selection", plainFields(encode(selectedImage->resolved().selection))},
                      {"label_selection", plainFields(encode(s.find("label")->reader->resolved().selection))},
                      {"depth_semantics", "sampled_layers"}, {"pairing_evidence", pairing.evidence}};
    } catch (...) {
        close();
        throw;
    }
}
PairedReader::~PairedReader() = default;
std::array<std::int64_t, 2> PairedReader::shape() const { return shape_; }
std::int64_t PairedReader::layerDepth() const { return layerDepth_; }
std::int64_t PairedReader::layerCount() const { return layerCount_; }
std::int64_t PairedReader::layerStart() const { return layerStart_; }
bool PairedReader::dropLayer() const { return dropLayer_; }
bool PairedReader::hasTarget(const std::string& name) const { return impl_->find(name) != nullptr; }
std::pair<SampledArray, SampledArray> PairedReader::sampleTarget(const std::string& name, const std::array<std::int64_t, 3>& start, const std::array<std::int64_t, 3>& size)
{
    auto& s = *impl_;
    std::lock_guard lock(s.mutex);
    s.checkOpen();
    auto* target = s.find(name);
    if (!target) throw std::out_of_range("no target '" + name + "'");
    return s.sampleMany(*target, {{start, size}}).at(0);
}
SampleResult PairedReader::read(const std::array<std::int64_t, 2>& start, const std::array<std::int64_t, 2>& size) { return readMany({{start, size}}).at(0); }
std::vector<SampleResult> PairedReader::readMany(const std::vector<std::pair<std::array<std::int64_t, 2>, std::array<std::int64_t, 2>>>& regions)
{
    auto& s = *impl_;
    std::lock_guard lock(s.mutex);
    s.checkOpen();
    for (const auto& [start, size] : regions)
        for (int a = 0; a < 2; ++a)
            if (size[a] < 0 || size[a] > (1 << 24) || std::abs(start[a]) > (1LL << 40)) bad("paired regions require YX start/size, nonnegative bounded sizes");
    std::vector<SampleResult> results(regions.size());
    for (auto& result : results) result.metadata = s.metadata;
    for (const auto& [name, target] : s.targets) {
        const std::int64_t z = name == "image" ? layerStart_ : 0, depth = name == "image" ? layerCount_ : 1;
        std::vector<std::pair<Triple, Triple>> requests;
        for (const auto& [start, size] : regions) requests.push_back({Triple{z, start[0], start[1]}, Triple{depth, size[0], size[1]}});
        auto values = s.sampleMany(target, requests);
        for (std::size_t i = 0; i < results.size(); ++i) {
            TargetResult out;
            auto data = std::move(values[i].first);
            auto valid = std::move(values[i].second);
            if (name != "image" || dropLayer_) {
                const auto n = data.shape;
                data = data.reshaped({n[0], n[2], n[3]});
                valid = valid.reshaped({1, n[2], n[3]});
            }
            out.data = std::move(data);
            out.valid = std::move(valid);
            results[i].targets.emplace_back(name, std::move(out));
        }
    }
    std::vector<std::pair<Triple, Triple>> maskRequests;
    for (const auto& [start, size] : regions) maskRequests.push_back({Triple{0, start[0], start[1]}, Triple{1, size[0], size[1]}});
    for (const auto& [name, target] : s.masks) {
        auto values = s.sampleMany(target, maskRequests);
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& n = values[i].first.shape;
            for (auto& [key, result] : results[i].targets)
                if (key == "label") result.masks.emplace_back(name, values[i].first.reshaped({n[0], n[2], n[3]}));
        }
    }
    return results;
}
AlignedStats PairedReader::stats()
{
    auto out = AlignedEngine::stats();
    for (const auto& [name, target] : impl_->masks) out.masks.emplace_back(name, target.reader->stats());
    return out;
}
} // namespace vcr::sources
