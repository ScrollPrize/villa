#include "vc_reader/source_reader.hpp"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <set>

#include "../array.hpp"
#include "../reader_types.hpp"

namespace vcr::sources {
using nlohmann::json;
namespace {
[[noreturn]] void bad(const std::string& message) { throw std::invalid_argument(message); }

std::int64_t product(const std::vector<std::int64_t>& values)
{
    std::int64_t out = 1;
    for (const auto v : values) out *= v;
    return out;
}
ElementType elementType(const std::string& typestr)
{
    ElementType type;
    type.kind = typestr[1];
    type.itemsize = std::size_t(typestr[2] - '0');
    type.swap = typestr[0] == '>';
    return type;
}
// Selectors by index or by the names storage gives.
std::int64_t selectIndex(const Selector& value, std::int64_t length, const std::string& name, const std::vector<std::string>& names)
{
    std::int64_t index = 0;
    if (const auto* text = std::get_if<std::string>(&value)) {
        std::vector<std::int64_t> matches;
        for (std::size_t i = 0; i < names.size(); ++i)
            if (names[i] == *text) matches.push_back(std::int64_t(i));
        if (matches.size() != 1) bad(name + " name '" + *text + "' is absent or ambiguous; select by integer index");
        index = matches[0];
    } else {
        index = std::get<std::int64_t>(value);
    }
    if (index < 0 || index >= length) bad(name + " index " + std::to_string(index) + " is outside [0, " + std::to_string(length) + ")");
    return index;
}
std::size_t axisIndex(const std::vector<std::string>& axes, std::string_view axis)
{
    return std::size_t(std::find(axes.begin(), axes.end(), axis) - axes.begin());
}
bool hasAxis(const std::vector<std::string>& axes, std::string_view axis) { return std::find(axes.begin(), axes.end(), axis) != axes.end(); }

std::string list(const std::vector<std::int64_t>& values)
{
    std::string text = "(";
    for (std::size_t i = 0; i < values.size(); ++i) text += (i ? ", " : "") + std::to_string(values[i]);
    if (values.size() == 1) text += ",";
    return text + ")";
}

// Number <-> stored element bytes, exactly.
json elementValue(const std::uint8_t* p, const ElementType& type)
{
    std::uint8_t b[8]{};
    std::memcpy(b, p, type.itemsize);
    if (type.swap) std::reverse(b, b + type.itemsize);
    switch (type.kind) {
    case 'b': return b[0] != 0 ? 1 : 0;
    case 'f':
        if (type.itemsize == 4) {
            float v;
            std::memcpy(&v, b, 4);
            return v;
        } else {
            double v;
            std::memcpy(&v, b, 8);
            return v;
        }
    case 'i': {
        std::int64_t v = 0;
        std::memcpy(&v, b, type.itemsize);
        const int shift = 64 - int(type.itemsize) * 8;
        return (v << shift) >> shift;
    }
    default: {
        std::uint64_t v = 0;
        std::memcpy(&v, b, type.itemsize);
        return v;
    }
    }
}

void fillBlock(Block& block, const std::vector<std::uint8_t>& element)
{
    const auto size = element.size();
    auto* p = block.data.get();
    if (size == 1 || std::all_of(element.begin(), element.end(), [&](auto byte) { return byte == element[0]; })) {
        std::memset(p, element[0], block.nbytes);
        return;
    }
    for (std::size_t i = 0; i < block.nbytes; i += size) std::memcpy(p + i, element.data(), size);
}

std::vector<std::int64_t> cStrides(const std::vector<std::int64_t>& shape, std::size_t itemsize)
{
    std::vector<std::int64_t> strides(shape.size());
    std::int64_t stride = std::int64_t(itemsize);
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}
// Calls f(offset) for each index of a shape, in C order, over all but its last `inner` dimensions.
template <class F>
void forEach(const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& strides, std::size_t outerRank, F&& f)
{
    for (std::size_t i = 0; i < outerRank; ++i)
        if (shape[i] <= 0) return;
    std::vector<std::int64_t> at(outerRank, 0);
    std::int64_t offset = 0;
    for (;;) {
        f(offset);
        std::size_t d = outerRank;
        for (;;) {
            if (d == 0) return;
            --d;
            offset += strides[d];
            if (++at[d] < shape[d]) break;
            offset -= strides[d] * shape[d];
            at[d] = 0;
        }
    }
}
// target[origin + ...] = source, same rank; `sourceAt` picks a box inside the source (default: its whole).
void assign(Block& target, const std::vector<std::int64_t>& targetOrigin, const Block& source)
{
    if (!source.nbytes) return;
    const auto itemsize = target.itemsize();
    const auto targetStrides = cStrides(target.shape, itemsize);
    const auto sourceStrides = cStrides(source.shape, itemsize);
    std::size_t rank = source.shape.size();
    std::size_t tail = rank;
    std::size_t run = itemsize;
    // Whole trailing dimensions copy as one run.
    while (tail > 0) {
        run *= std::size_t(source.shape[tail - 1]);
        --tail;
        if (tail == 0 || source.shape[tail] != target.shape[tail]) break;
    }
    std::int64_t base = 0;
    for (std::size_t i = 0; i < rank; ++i) base += targetOrigin[i] * targetStrides[i];
    std::vector<std::int64_t> at(tail, 0);
    std::int64_t from = 0, to = base;
    for (;;) {
        std::memcpy(target.data.get() + to, source.data.get() + from, run);
        std::size_t d = tail;
        for (;;) {
            if (d == 0) return;
            --d;
            from += sourceStrides[d];
            to += targetStrides[d];
            if (++at[d] < source.shape[d]) break;
            from -= sourceStrides[d] * source.shape[d];
            to -= targetStrides[d] * source.shape[d];
            at[d] = 0;
        }
    }
}
// One channel subset along an axis.
Block takeAlong(const Block& block, std::size_t axis, const std::vector<std::int64_t>& indices)
{
    auto shape = block.shape;
    shape[axis] = std::int64_t(indices.size());
    Block out = Block::allocate(shape, block.dtype);
    const auto itemsize = block.itemsize();
    const auto sourceStrides = cStrides(block.shape, itemsize);
    const auto outStrides = cStrides(shape, itemsize);
    auto outer = shape;
    for (std::size_t c = 0; c < indices.size(); ++c) {
        // Iterate every other dimension.
        std::vector<std::int64_t> dims = block.shape, strideIn = sourceStrides, strideOut = outStrides;
        dims.erase(dims.begin() + std::ptrdiff_t(axis));
        strideIn.erase(strideIn.begin() + std::ptrdiff_t(axis));
        strideOut.erase(strideOut.begin() + std::ptrdiff_t(axis));
        std::vector<std::int64_t> at(dims.size(), 0);
        std::int64_t from = indices[c] * sourceStrides[axis], to = std::int64_t(c) * outStrides[axis];
        const auto advance = [&] {
            std::size_t d = dims.size();
            for (;;) {
                if (d == 0) return false;
                --d;
                from += strideIn[d];
                to += strideOut[d];
                if (++at[d] < dims[d]) return true;
                from -= strideIn[d] * dims[d];
                to -= strideOut[d] * dims[d];
                at[d] = 0;
            }
        };
        bool more = block.nbytes != 0;
        while (more) {
            std::memcpy(out.data.get() + to, block.data.get() + from, itemsize);
            more = advance();
        }
    }
    return out;
}
} // namespace

std::vector<std::uint8_t> elementBytes(const json& value, const std::string& typestr)
{
    const auto type = elementType(typestr);
    const auto unrepresentable = [&] { throw std::invalid_argument("fill value " + value.dump() + " is not representable in " + dtypeName(typestr)); };
    std::uint8_t b[8]{};
    if (type.kind == 'f') {
        const double v = value.get<double>();
        if (type.itemsize == 4) {
            const auto f = float(v);
            if (std::isfinite(v) && !std::isfinite(f)) unrepresentable();
            std::memcpy(b, &f, 4);
        } else {
            std::memcpy(b, &v, 8);
        }
    } else {
        const bool isSigned = type.kind == 'i';
        const int bits = type.kind == 'b' ? 1 : int(type.itemsize) * 8;
        if (value.is_number_float()) {
            const double v = value.get<double>();
            if (!std::isfinite(v) || v != std::floor(v) || std::abs(v) >= 1.8446744073709552e19) unrepresentable();
        }
        std::uint64_t magnitude = 0;
        bool negative = false;
        if (value.is_number_float()) {
            const double v = value.get<double>();
            negative = v < 0;
            magnitude = std::uint64_t(std::abs(v));
        } else if (value.is_number_unsigned()) {
            magnitude = value.get<std::uint64_t>();
        } else {
            const auto v = value.get<std::int64_t>();
            negative = v < 0;
            magnitude = negative ? ~std::uint64_t(v) + 1 : std::uint64_t(v);
        }
        const std::uint64_t maximum = type.kind == 'b' ? 1 : isSigned ? (std::uint64_t(1) << (bits - 1)) - 1 : (bits == 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << bits) - 1);
        if (negative && (!isSigned || magnitude > maximum + 1)) unrepresentable();
        if (!negative && magnitude > maximum) unrepresentable();
        const std::uint64_t raw = negative ? ~magnitude + 1 : magnitude;
        std::memcpy(b, &raw, type.itemsize);
    }
    if (type.swap) std::reverse(b, b + type.itemsize);
    return std::vector<std::uint8_t>(b, b + type.itemsize);
}

// ---- exact encodings ----
// Sorted lookup; no label passes through a float.
struct SourceReader::Lookup {
    std::string outputDtype;  // typestr
    bool color = false;
    std::size_t channels = 1;
    ElementType input;
    std::vector<std::uint64_t> keys;  // flattened, comparable as unsigned
    std::vector<std::uint64_t> values;  // output bits
    std::vector<std::int32_t> direct;   // index by raw value, for one small channel
    std::optional<std::uint64_t> fallback;
    std::size_t outSize = 1;

    static std::uint64_t outputBits(const WideInt& v) { return v.negative ? ~v.magnitude + 1 : v.magnitude; }
    [[nodiscard]] std::uint64_t keyOf(const std::uint8_t* p) const
    {
        std::uint8_t b[8]{};
        std::memcpy(b, p, input.itemsize);
        if (input.swap) std::reverse(b, b + input.itemsize);
        std::uint64_t v = 0;
        std::memcpy(&v, b, input.itemsize);
        if (input.kind == 'i') {
            const int shift = 64 - int(input.itemsize) * 8;
            v = std::uint64_t(std::int64_t(v << shift) >> shift) ^ (std::uint64_t(1) << 63);
        }
        return v;
    }
    Lookup(const Encoding& encoding, const std::string& inputDtype, std::size_t selectedChannels)
    {
        input = elementType(inputDtype);
        if (input.kind == 'f') bad("integer/color encoding requires integer decoded samples");
        outputDtype = canonicalDtype(encoding.dtype);
        outSize = std::size_t(outputDtype[2] - '0');
        color = encoding.kind == "color";
        channels = color ? selectedChannels : 1;
        const int bits = input.kind == 'b' ? 1 : int(input.itemsize) * 8;
        const auto fits = [&](const WideInt& v) {
            if (input.kind == 'i') {
                const std::uint64_t maximum = (std::uint64_t(1) << (bits - 1)) - 1;
                return v.negative ? v.magnitude <= maximum + 1 : v.magnitude <= maximum;
            }
            return !v.negative && (bits == 64 || v.magnitude <= (std::uint64_t(1) << bits) - 1);
        };
        std::vector<std::pair<std::vector<std::uint64_t>, std::uint64_t>> pairs;
        for (const auto& pair : encoding.pairs) {
            if (color && pair.key.size() != channels) bad("color lookup tuple length must match selected channels");
            if (!std::all_of(pair.key.begin(), pair.key.end(), fits)) continue;
            std::vector<std::uint64_t> key;
            for (const auto& item : pair.key) {
                std::uint64_t k = item.negative ? ~item.magnitude + 1 : item.magnitude;
                if (input.kind == 'i') k ^= std::uint64_t(1) << 63;
                key.push_back(k);
            }
            pairs.emplace_back(std::move(key), outputBits(pair.value));
        }
        std::sort(pairs.begin(), pairs.end());
        for (const auto& [key, value] : pairs) {
            keys.insert(keys.end(), key.begin(), key.end());
            values.push_back(value);
        }
        if (encoding.fallback) fallback = outputBits(*encoding.fallback);
        if (!color && input.itemsize <= 2) {
            direct.assign(std::size_t(1) << (input.itemsize * 8), -1);
            for (std::size_t i = 0; i < values.size(); ++i) {
                std::uint64_t raw = keys[i];
                if (input.kind == 'i') raw = (raw ^ (std::uint64_t(1) << 63)) & ((std::uint64_t(1) << (input.itemsize * 8)) - 1);
                direct[raw] = std::int32_t(i);
            }
        }
    }
    [[nodiscard]] std::size_t tableBytes() const { return (keys.size() * 8 + values.size() * 8 + direct.size() * 4); }
    [[noreturn]] void unknown(const std::uint8_t* p) const
    {
        std::string text = "(";
        for (std::size_t c = 0; c < channels; ++c) text += (c ? ", " : "") + elementValue(p + c * input.itemsize, input).dump();
        if (channels == 1) text = elementValue(p, input).dump();
        else text += ")";
        bad("unknown encoding value " + text + "; declare a fallback to accept it");
    }
    // Encoded values of a block; a color channel axis is consumed.
    [[nodiscard]] Block apply(const Block& block, std::optional<std::size_t> channelAxis) const
    {
        std::vector<std::int64_t> shape;
        for (std::size_t i = 0; i < block.shape.size(); ++i)
            if (!(color && channelAxis && i == *channelAxis)) shape.push_back(block.shape[i]);
        Block out = Block::allocate(shape, outputDtype);
        const auto itemsize = input.itemsize;
        const std::int64_t count = product(shape);
        const auto put = [&](std::int64_t i, std::uint64_t value) {
            std::memcpy(out.data.get() + std::size_t(i) * outSize, &value, outSize);
        };
        const auto search = [&](const std::uint64_t* query) -> std::int64_t {
            std::size_t low = 0, high = values.size();
            while (low < high) {
                const auto middle = (low + high) / 2;
                if (std::lexicographical_compare(keys.data() + middle * channels, keys.data() + (middle + 1) * channels, query, query + channels)) low = middle + 1;
                else high = middle;
            }
            if (low < values.size() && std::equal(query, query + channels, keys.data() + low * channels)) return std::int64_t(low);
            return -1;
        };
        const auto emit = [&](std::int64_t i, std::int64_t found, const std::uint8_t* element) {
            if (found >= 0) put(i, values[std::size_t(found)]);
            else if (fallback) put(i, *fallback);
            else unknown(element);
        };
        if (!color) {
            const auto* p = block.data.get();
            if (!direct.empty()) {
                for (std::int64_t i = 0; i < count; ++i, p += itemsize) {
                    std::uint32_t raw = p[0];
                    if (itemsize == 2) raw = input.swap ? (std::uint32_t(p[0]) << 8) | p[1] : (std::uint32_t(p[1]) << 8) | p[0];
                    emit(i, direct[raw], p);
                }
            } else {
                for (std::int64_t i = 0; i < count; ++i, p += itemsize) {
                    const auto key = keyOf(p);
                    emit(i, search(&key), p);
                }
            }
            return out;
        }
        const auto axis = *channelAxis;
        const auto strides = cStrides(block.shape, itemsize);
        std::vector<std::int64_t> dims, stepsIn;
        for (std::size_t i = 0; i < block.shape.size(); ++i)
            if (i != axis) {
                dims.push_back(block.shape[i]);
                stepsIn.push_back(strides[i]);
            }
        std::vector<std::uint64_t> query(channels);
        std::vector<std::uint8_t> element(channels * itemsize);
        std::int64_t i = 0;
        forEach(dims, stepsIn, dims.size(), [&](std::int64_t offset) {
            for (std::size_t c = 0; c < channels; ++c) {
                const auto* p = block.data.get() + offset + std::int64_t(c) * strides[axis];
                query[c] = keyOf(p);
                std::memcpy(element.data() + c * itemsize, p, itemsize);
            }
            emit(i++, search(query.data()), element.data());
        });
        return out;
    }
};

// ---- Zarr ----
namespace {
std::pair<std::optional<std::vector<std::optional<std::string>>>, std::vector<std::optional<std::string>>> axesMetadata(const json& value)
{
    if (!value.is_array()) return {std::nullopt, {}};
    std::vector<std::optional<std::string>> names, units;
    for (const auto& axis : value) {
        if (axis.is_object()) {
            names.emplace_back(axis.value("name", ""));
            if (axis.contains("unit") && axis["unit"].is_string()) units.emplace_back(axis["unit"].get<std::string>());
            else units.emplace_back(std::nullopt);
        } else if (axis.is_string()) {
            names.emplace_back(axis.get<std::string>());
            units.emplace_back(std::nullopt);
        } else {
            names.emplace_back(std::nullopt);
            units.emplace_back(std::nullopt);
        }
    }
    return {names, units};
}
const json& orNullJson(const json& object, const char* key)
{
    static const json null;
    if (!object.is_object()) return null;
    const auto it = object.find(key);
    return it == object.end() ? null : *it;
}
bool truthyJson(const json& value) { return !(value.is_null() || (value.is_array() && value.empty()) || (value.is_object() && value.empty()) || (value.is_string() && value.get_ref<const std::string&>().empty())); }
} // namespace

struct SourceReader::Backend {
    Source source;
    zarr::OpenOptions options;
    std::optional<zarr::Group> group;
    std::shared_ptr<zarr::OpenedArray> reader;
    json series = json::object();
    json datasets = json::array();
    json multiscales = json::array();
    std::optional<std::int64_t> seriesIndex;
    std::optional<SourceRegistration> registration;
    std::vector<std::string> axes;
    json fill;
    std::vector<std::string> channelNames, timeNames;
    ResolvedSource resolved;
    std::string path;
    struct Grid {
        std::vector<std::string> axes;
        std::vector<std::optional<std::string>> units;
        StorageGrid storage;
    };
    std::map<std::string, Grid> grids;
    std::map<std::string, std::shared_ptr<zarr::OpenedArray>> opened;

    Backend(Source s, zarr::OpenOptions o) : source(std::move(s)), options(std::move(o)) { open(); }

    std::shared_ptr<zarr::OpenedArray> arrayAt(const std::string& p)
    {
        auto& slot = opened[p];
        if (!slot) slot = std::make_shared<zarr::OpenedArray>(group->array(p));
        return slot;
    }
    std::string pathOf(const Selection& selection)
    {
        if (selection.arrayPath) {
            auto p = *selection.arrayPath;
            while (!p.empty() && p.front() == '/') p.erase(0, 1);
            while (!p.empty() && p.back() == '/') p.pop_back();
            return p;
        }
        if (selection.level) {
            if (!datasets.empty()) {
                if (*selection.level >= std::int64_t(datasets.size())) bad("requested source level " + std::to_string(*selection.level) + " does not exist");
                return datasets[std::size_t(*selection.level)].value("path", "");
            }
            return std::to_string(*selection.level);
        }
        bad("group child is ambiguous; select array_path or a pyramid level explicitly");
    }
    [[nodiscard]] Selection canonical(const std::string& p, const Selection& selection) const
    {
        Selection out;
        if (!p.empty() || group) out.arrayPath = p;
        if (seriesIndex) out.series = *seriesIndex;
        out.channels = selection.channels;
        out.time = selection.time;
        return out;
    }
    const Grid& grid(const std::string& p)
    {
        const auto found = grids.find(p);
        if (found != grids.end()) return found->second;
        const auto opened = group ? arrayAt(p) : reader;
        const auto& attrs = zarr::arrayAttributes(*opened, options);
        const auto [groupAxes, groupUnits] = axesMetadata(orNullJson(series, "axes"));
        const auto& arrayDeclared = truthyJson(orNullJson(attrs, "axes")) ? orNullJson(attrs, "axes") : orNullJson(attrs, "_ARRAY_DIMENSIONS");
        const auto [arrayAxes, arrayUnits] = axesMetadata(arrayDeclared);
        if (groupAxes && arrayAxes && !groupAxes->empty() && !arrayAxes->empty() && *groupAxes != *arrayAxes) bad("array axes contradict group axes");
        std::vector<std::optional<std::string>> observed;
        if (arrayAxes && !arrayAxes->empty()) observed = *arrayAxes;
        else if (groupAxes && !groupAxes->empty()) observed = *groupAxes;
        else observed.assign(opened->meta.shape.size(), std::nullopt);
        Grid out;
        out.axes = resolveAxes(source.axes, observed, std::vector<std::int64_t>(opened->meta.shape.begin(), opened->meta.shape.end()));
        const bool arrayHasUnits = std::any_of(arrayUnits.begin(), arrayUnits.end(), [](const auto& u) { return u.has_value(); });
        if (!groupUnits.empty() && arrayHasUnits) {
            for (std::size_t i = 0; i < out.axes.size() && i < groupUnits.size() && i < arrayUnits.size(); ++i) {
                const auto& a = groupUnits[i];
                const auto& b = arrayUnits[i];
                if (a && b && *a != *b && (!micrometersPer(*a) || micrometersPer(*a) != micrometersPer(*b)))
                    bad("array and group unit declarations conflict for " + out.axes[i] + ": " + *a + ", " + *b);
            }
        }
        out.units = arrayHasUnits ? arrayUnits : groupUnits;
        const json* dataset = nullptr;
        for (const auto& d : datasets)
            if (d.value("path", "") == p) {
                dataset = &d;
                break;
            }
        const json datasetTransforms = dataset ? orNullJson(*dataset, "coordinateTransformations") : json(nullptr);
        const json arrayTransforms = orNullJson(attrs, "coordinateTransformations");
        if (truthyJson(datasetTransforms) && truthyJson(arrayTransforms) && datasetTransforms != arrayTransforms)
            bad("array and multiscale dataset declare conflicting transformations for the same grid");
        json levelTransforms = truthyJson(arrayTransforms) ? arrayTransforms : datasetTransforms;
        if (dataset) levelTransforms = blockReducedTransforms(levelTransforms, orNullJson(datasets[0], "coordinateTransformations"));
        out.storage = storageTransform(out.axes, out.units, {levelTransforms, orNullJson(series, "coordinateTransformations")});
        return grids.emplace(p, std::move(out)).first->second;
    }
    std::optional<SourceRegistration> registrationAt(const std::string& p, const Selection& selection)
    {
        if (!source.registration) return std::nullopt;
        const auto* volume = std::get_if<VolumeRegistration>(&*source.registration);
        const auto& declaredGrid = volume ? volume->declaredGrid : std::get<SurfaceRegistration>(*source.registration).declaredGrid;
        const Selection original = declaredGrid.value_or(source.storage.selection);
        if (group && !multiscales.empty()) {
            std::optional<std::int64_t> oldSeries;
            if (original.series) {
                if (const auto* text = std::get_if<std::string>(&*original.series)) {
                    std::vector<std::int64_t> candidates;
                    for (std::size_t i = 0; i < multiscales.size(); ++i)
                        if (multiscales[i].value("name", "") == *text) candidates.push_back(std::int64_t(i));
                    oldSeries = candidates.size() == 1 ? candidates[0] : -1;
                } else {
                    oldSeries = std::get<std::int64_t>(*original.series);
                }
            } else if (multiscales.size() == 1) {
                oldSeries = 0;
            }
            if (oldSeries != seriesIndex) bad("selected series has no declared connection to the original series; supply placement");
        }
        std::optional<std::string> oldPath;
        if (original.spatial() == source.storage.selection.spatial() && !original.corresponding) oldPath = p;
        else if (group) oldPath = pathOf(original);
        const auto canonicalSelection = canonical(p, selection);
        const auto declared = canonical(oldPath.value_or(""), original);
        const auto current = grid(p);
        const Grid old = oldPath ? grid(*oldPath) : Grid{};
        auto resolvedSource = source;
        std::visit([&](auto& r) { r.declaredGrid = declared; }, *resolvedSource.registration);
        const auto* placed = std::get_if<VolumeRegistration>(&*resolvedSource.registration);
        if (placed && placed->mode == "storage") {
            const auto opened = group ? arrayAt(p) : reader;
            std::vector<json> frames{orNullJson(zarr::arrayAttributes(*opened, options), "coordinateSystem")};
            if (group) frames.push_back(orNullJson(group->attributes(), "coordinateSystem"));
            for (const auto& frame : frames)
                if (!frame.is_null() && frame != json(placed->storageFrame.value_or("")))
                    bad("storage frame " + frame.dump() + " contradicts declared frame '" + placed->storageFrame.value_or("") + "'");
        }
        std::optional<VolumeTransform> relative;
        if (placed && placed->origin == "catalog-fallback" && !current.storage.transform && !old.storage.transform) {
            std::optional<std::int64_t> oldLevel, level;
            const auto digits = [](const std::string& text) -> std::optional<std::int64_t> {
                if (text.empty() || !std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c); })) return std::nullopt;
                return std::stoll(text);
            };
            if (!datasets.empty()) {
                for (std::size_t i = 0; i < datasets.size(); ++i) {
                    if (oldPath && datasets[i].value("path", "") == *oldPath && !oldLevel) oldLevel = std::int64_t(i);
                    if (datasets[i].value("path", "") == p && !level) level = std::int64_t(i);
                }
            } else {
                if (oldPath) oldLevel = digits(*oldPath);
                level = digits(p);
            }
            if (oldLevel && level) relative = VolumeTransform{Affine::blockReduced(std::ldexp(1.0, int(*level - *oldLevel))), {}};
        }
        return resolveRegistration(resolvedSource, canonicalSelection, current.storage.transform, current.storage.units, old.storage.transform, old.storage.units, relative);
    }
    std::string corresponding(const Selection& selection)
    {
        const auto& request = *selection.corresponding;
        const auto& reference = request.reference;
        const auto* volume = source.registration ? std::get_if<VolumeRegistration>(&*source.registration) : nullptr;
        if (!volume) bad("corresponding level requires volume registration; select a source level explicitly");
        const auto level = std::to_string(reference.level);
        std::vector<std::string> paths;
        for (const auto& d : datasets) paths.push_back(d.value("path", ""));
        const bool haveTransforms = std::any_of(paths.begin(), paths.end(), [&](const auto& p) { return grid(p).storage.transform.has_value(); });
        if (!haveTransforms) {
            if (!request.sublevel || *request.sublevel < 0) bad("no exact correspondence for requested reference level " + level + "; select a source level explicitly");
            Selection chosen = selection;
            chosen.corresponding.reset();
            chosen.level = *request.sublevel;
            const auto p = pathOf(chosen);
            (void)arrayAt(p);  // No fallback when the array does not exist.
            if (!matchesReference(source, registrationAt(p, selection), reference, grid(p).axes))
                bad("no exact correspondence for requested reference level " + level + "; select a source level explicitly");
            return p;
        }
        std::vector<std::string> matches;
        for (const auto& p : paths)
            if (matchesReference(source, registrationAt(p, selection), reference, grid(p).axes)) matches.push_back(p);
        if (matches.size() != 1) bad("requested reference level " + level + " has " + std::to_string(matches.size()) + " exact source levels; select a source level explicitly");
        return matches[0];
    }
    void open()
    {
        const auto& selection = source.storage.selection;
        const auto& location = *source.storage.location;
        auto trimmed = location;
        while (trimmed.size() > 1 && trimmed.back() == '/') trimmed.pop_back();
        if (source.storage.pyramid) {
            group.emplace(zarr::Group::open(trimmed, options));
        } else {
            try {
                reader = std::make_shared<zarr::OpenedArray>(zarr::openArray(trimmed, options));
            } catch (const zarr::NotFound&) {
                group.emplace(zarr::Group::open(trimmed, options));
            } catch (const CacheMiss&) {
                group.emplace(zarr::Group::open(trimmed, options));
            }
        }
        std::string p;
        if (!group) {
            if (selection.arrayPath || selection.level || selection.series || selection.corresponding)
                bad("an array location cannot also select a group child, series or pyramid level");
        } else {
            multiscales = group->multiscales();
            if (!multiscales.is_array()) multiscales = json::array();
            if (!multiscales.empty()) {
                std::int64_t index = 0;
                if (!selection.series) {
                    if (multiscales.size() != 1) bad("multiple multiscale series; select series explicitly");
                } else if (const auto* text = std::get_if<std::string>(&*selection.series)) {
                    std::vector<std::int64_t> matches;
                    for (std::size_t i = 0; i < multiscales.size(); ++i)
                        if (multiscales[i].value("name", "") == *text) matches.push_back(std::int64_t(i));
                    if (matches.size() != 1) bad("absent or ambiguous multiscale series '" + *text + "'");
                    index = matches[0];
                } else {
                    index = std::get<std::int64_t>(*selection.series);
                }
                if (index < 0 || index >= std::int64_t(multiscales.size())) bad("multiscale series " + std::to_string(index) + " does not exist");
                series = multiscales[std::size_t(index)];
                datasets = series.value("datasets", json::array());
                seriesIndex = index;
            } else if (selection.series) {
                bad("series selection requires multiscale series metadata");
            }
            p = selection.corresponding ? corresponding(selection) : pathOf(selection);
            reader = arrayAt(p);
        }
        axes = grid(p).axes;
        registration = registrationAt(p, selection);
        const auto type = reader->meta.type;
        fill = elementValue(reader->meta.fill.data(), type);
        const auto& attrs = zarr::arrayAttributes(*reader, options);
        const auto strings = [](const json& value) {
            std::vector<std::string> out;
            if (value.is_array())
                for (const auto& item : value) out.push_back(item.is_string() ? item.get<std::string>() : std::string());
            return out;
        };
        channelNames = strings(orNullJson(attrs, "channel_names"));
        if (channelNames.empty()) {
            const auto& root = group ? group->attributes() : attrs;
            const auto& channels = orNullJson(orNullJson(root, "omero"), "channels");
            if (channels.is_array())
                for (const auto& c : channels) channelNames.push_back(c.value("label", ""));
        }
        timeNames = strings(orNullJson(attrs, "time_names"));
        const auto& g = grid(p);
        resolved.location = reader->location;
        resolved.selection = canonical(p, selection);
        resolved.axes = axes;
        resolved.shape.assign(reader->meta.shape.begin(), reader->meta.shape.end());
        resolved.dtype = dtypeName(reader->meta.dtype);
        resolved.registration = registration;
        resolved.storageFromArray = g.storage.transform;
        resolved.storageUnits = g.storage.units;
        path = p;
    }
};

// ---- the reader ----
namespace {
// Keeps engines a forked process inherited: their threads and locks are not usable here.
void keepInherited(std::shared_ptr<void> engine)
{
    static auto* kept = new std::vector<std::shared_ptr<void>>();
    kept->push_back(std::move(engine));
}

template <class F>
auto annotated(const std::string& id, F&& f) -> decltype(f())
{
    const auto prefix = id + ": ";
    const auto named = [&](const char* what) { return std::string_view(what).starts_with(prefix) ? std::string(what) : prefix + what; };
    try {
        return f();
    } catch (const zarr::NotFound& e) {
        throw zarr::NotFound(named(e.what()));
    } catch (const CacheMiss& e) {
        throw CacheMiss(named(e.what()));
    } catch (const HttpDenied& e) {
        throw HttpDenied(named(e.what()));
    } catch (const DecodeBudgetError& e) {
        throw DecodeBudgetError(named(e.what()));
    } catch (const ImageFormatError& e) {
        throw ImageFormatError(named(e.what()));
    } catch (const DeclarationTypeError&) {
        throw;
    } catch (const std::out_of_range& e) {
        throw std::out_of_range(named(e.what()));
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(named(e.what()));
    } catch (const std::filesystem::filesystem_error&) {
        throw;
    } catch (const std::system_error&) {
        throw;
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(named(e.what()));
    }
}
} // namespace

SourceReader::SourceReader(Source source, Settings settings)
    : source_(std::move(source)), options_(settings.options ? std::move(settings.options) : defaultOptions()), pid_(::getpid())
{
    if (settings.decodedCache) {
        if (settings.decodedCacheBytes || settings.maxDecodeBytes) bad("configure byte budgets on the supplied decoded_cache object");
        cache_ = std::move(settings.decodedCache);
        ownsCache_ = false;
    } else {
        cache_ = std::make_shared<DecodedCache>(settings.decodedCacheBytes.value_or(kDefaultDecodeBytes), settings.maxDecodeBytes.value_or(kDefaultDecodeBytes));
        ownsCache_ = true;
    }
    encoded_ = std::make_shared<EncodedFiles>(options_);
    try {
        annotated(source_.id, [&] { open(); });
    } catch (...) {
        close();
        throw;
    }
}

SourceReader::~SourceReader()
{
    try {
        if (pid_ == ::getpid()) close();
        else reset();
    } catch (...) {
    }
}

void SourceReader::reset()
{
    if (backend_) {
        if (pid_ != ::getpid()) keepInherited(std::shared_ptr<void>(std::make_shared<std::unique_ptr<Backend>>(std::move(backend_))));
        backend_.reset();
    }
    companions_.clear();
    companionOrder_.clear();
    files_.clear();
    lookup_.reset();
    opened_ = false;
}

void SourceReader::ready()
{
    if (closed_) bad("reader is closed");
    if (pid_ != ::getpid()) {
        reset();
        pid_ = ::getpid();
        annotated(source_.id, [&] { open(); });
    } else if (!opened_) {
        annotated(source_.id, [&] { open(); });
    }
}

Source imageSource(const std::string& location, const OptionsFor& options)
{
    Source source;
    source.id = location;
    source.storage.location = location;
    source.role = "scalar";
    auto files = std::make_shared<EncodedFiles>(options ? options : defaultOptions());
    Selection selection;
    selection.series = std::int64_t(0);
    selection.level = std::int64_t(0);
    try {
        ImageFile image(location, std::nullopt, selection, std::nullopt, files, true);
        source.axes = image.info().axes;
        source.storage.format = image.format();
        source.storage.selection = std::move(selection);
    } catch (const ImageFormatError&) {
        throw;
    } catch (const std::invalid_argument&) {
        // PNG and JPEG select nothing.
        ImageFile image(location, std::nullopt, {}, std::nullopt, files);
        source.axes = image.info().axes;
        source.storage.format = image.format();
    }
    files->close();
    return finished(std::move(source));
}

Block readImage(const std::string& location, const OptionsFor& options, const std::optional<std::vector<std::string>>& axes)
{
    SourceReader::Settings settings;
    settings.options = options;
    settings.decodedCacheBytes = 0;
    settings.maxDecodeBytes = std::numeric_limits<std::size_t>::max() / 2;
    auto source = [&] {
        if (!axes) return imageSource(location, options);
        Source declared;
        declared.id = location;
        declared.storage.location = location;
        declared.role = "scalar";
        declared.axes = axes;
        return finished(std::move(declared));
    }();
    SourceReader reader(std::move(source), std::move(settings));
    std::vector<std::int64_t> size;
    for (const auto& axis : reader.spatialAxes())
        size.push_back(reader.shape()[std::size_t(std::find(reader.axes().begin(), reader.axes().end(), axis) - reader.axes().begin())]);
    return reader.read(std::vector<std::int64_t>(size.size(), 0), size);
}

Block readImagePage(const std::string& location, std::int64_t page, const OptionsFor& options)
{
    SourceReader::Settings settings;
    settings.options = options;
    settings.decodedCacheBytes = 0;
    settings.maxDecodeBytes = std::numeric_limits<std::size_t>::max() / 2;
    SourceReader reader(imageSource(location, options), std::move(settings));
    const auto& axes = reader.axes();
    std::vector<std::int64_t> start, size;
    for (const auto& axis : reader.spatialAxes()) {
        const auto n = reader.shape()[std::size_t(std::find(axes.begin(), axes.end(), axis) - axes.begin())];
        start.push_back(axis == "z" ? page : 0);
        size.push_back(axis == "z" ? 1 : n);
    }
    auto block = reader.read(start, size);
    const auto z = std::find(axes.begin(), axes.end(), "z");
    if (z != axes.end()) block.shape.erase(block.shape.begin() + (z - axes.begin()));
    return block;
}

ImageFile& SourceReader::file(std::int64_t index)
{
    const auto found = files_.find(index);
    if (found != files_.end()) return *found->second;
    auto image = std::make_unique<ImageFile>(locations_.at(std::size_t(index)), stack_ ? std::nullopt : source_.storage.format, imageSelection_, fileAxes_, encoded_);
    if (!files_.empty()) {
        const auto& first = files_.begin()->second->info();
        const auto& info = image->info();
        if (info.shape != first.shape || info.axes != first.axes || info.dtype != first.dtype)
            bad("stack plane " + std::to_string(index) + " shape/dtype/axes differs from the first plane");
    }
    return *(files_[index] = std::move(image));
}

Selection SourceReader::correspondingImage()
{
    const auto& request = *source_.storage.selection.corresponding;
    const auto* volume = source_.registration ? std::get_if<VolumeRegistration>(&*source_.registration) : nullptr;
    if (stack_ || !volume) bad("corresponding image levels require a registered TIFF pyramid");
    Selection selection = source_.storage.selection;
    selection.corresponding.reset();
    selection.level = 0;
    ImageFile base(locations_[0], source_.storage.format, selection, fileAxes_, encoded_);
    if (base.format() != "tiff") bad("corresponding levels require a TIFF pyramid or Zarr group");
    const auto& original = volume->declaredGrid;
    if (!original || original->corresponding) bad("corresponding levels need an explicitly placed base grid");
    ImageFile old(locations_[0], base.format(), *original, fileAxes_, encoded_);
    std::vector<Selection> matches;
    for (std::int64_t level = 0; level < base.levelCount(); ++level) {
        Selection selected = selection;
        selected.level = level;
        ImageFile image(locations_[0], base.format(), selected, fileAxes_, encoded_);
        std::optional<VolumeTransform> relative;
        if (!image.storageFromArray() && !old.storageFromArray()) {
            if (volume->origin != "catalog-fallback" || !original->level) continue;
            relative = VolumeTransform{Affine::blockReduced(std::ldexp(1.0, int(level - *original->level))), {}};
        }
        const auto placement = resolveRegistration(source_, selected, image.storageFromArray(), image.storageUnits(), old.storageFromArray(), old.storageUnits(), relative);
        if (matchesReference(source_, placement, request.reference, image.info().axes)) matches.push_back(selected);
    }
    if (matches.size() != 1) bad("requested reference level " + std::to_string(request.reference.level) + " has " + std::to_string(matches.size()) + " exact TIFF levels; select a level explicitly");
    return matches[0];
}

void SourceReader::open()
{
    const auto& storage = source_.storage;
    stack_ = storage.format == "stack";
    const auto endsWithZarr = [](std::string location) {
        while (!location.empty() && location.back() == '/') location.pop_back();
        return location.ends_with(".zarr");
    };
    zarr_ = storage.format == "zarr" || (!storage.format && storage.location && endsWithZarr(*storage.location));
    if (!storage.location && !stack_) bad("storage.location is required");
    locations_ = stack_ ? storage.locations : std::vector<std::string>{*storage.location};
    fileAxes_ = source_.axes;
    imageSelection_ = storage.selection;
    if (stack_ && source_.axes) {
        if (source_.axes->front() != "z") bad("an ordered file stack requires a leading z axis");
        fileAxes_ = std::vector<std::string>(source_.axes->begin() + 1, source_.axes->end());
    }
    std::vector<std::optional<std::string>> attempts;
    attempts.push_back(storage.location);
    for (const auto& alternative : storage.alternatives) attempts.emplace_back(alternative);
    std::exception_ptr failure;
    bool opened = false;
    std::vector<std::int64_t> infoShape;
    std::vector<std::string> infoAxes;
    std::string infoDtype;
    std::vector<std::string> timeNames, channelNames;
    ImageFile* first = nullptr;
    for (const auto& location : attempts) {
        try {
            if (zarr_) {
                Source selected = source_;
                selected.storage.location = location;
                selected.storage.alternatives.clear();
                auto options = options_(location.value_or(""));
                backend_ = std::make_unique<Backend>(std::move(selected), std::move(options));
                infoShape.assign(backend_->reader->meta.shape.begin(), backend_->reader->meta.shape.end());
                infoAxes = backend_->axes;
                infoDtype = backend_->reader->meta.dtype;
                timeNames = backend_->timeNames;
                channelNames = backend_->channelNames;
            } else {
                if (!stack_) locations_ = {location.value_or("")};
                files_.clear();
                if (storage.selection.corresponding) imageSelection_ = correspondingImage();
                first = &file(0);
                const auto& info = first->info();
                infoShape = info.shape;
                infoAxes = info.axes;
                infoDtype = info.dtype;
                channelNames = first->channelNames();
                if (stack_ && (hasAxis(info.axes, "z") || hasAxis(info.axes, "t")))
                    bad("stack entries must be individual 2D images, optionally with channels");
            }
            opened = true;
            break;
        } catch (const zarr::NotFound&) {
            failure = std::current_exception();
        } catch (const CacheMiss&) {
            failure = std::current_exception();
        }
    }
    if (!opened) std::rethrow_exception(failure);
    rawAxes_.clear();
    rawShape_.clear();
    if (stack_) {
        rawAxes_.emplace_back("z");
        rawShape_.push_back(std::int64_t(locations_.size()));
    }
    rawAxes_.insert(rawAxes_.end(), infoAxes.begin(), infoAxes.end());
    rawShape_.insert(rawShape_.end(), infoShape.begin(), infoShape.end());
    if (source_.shape && *source_.shape != rawShape_) bad("declared shape " + list(*source_.shape) + " contradicts storage shape " + list(rawShape_));
    if (source_.dtype && canonicalDtype(*source_.dtype) != canonicalDtype(infoDtype))
        bad("declared dtype " + *source_.dtype + " contradicts storage dtype " + dtypeName(infoDtype));
    rawDtype_ = infoDtype;
    const auto& selection = storage.selection;
    time_.reset();
    if (hasAxis(rawAxes_, "t")) {
        const auto ntime = rawShape_[axisIndex(rawAxes_, "t")];
        if (!selection.time && ntime != 1) bad("time selection is required for a multi-time source");
        time_ = selectIndex(selection.time.value_or(Selector(std::int64_t(0))), ntime, "time", timeNames);
    } else if (selection.time) {
        bad("time was selected but storage has no time axis");
    }
    channels_.reset();
    if (hasAxis(rawAxes_, "c")) {
        const auto nchannel = rawShape_[axisIndex(rawAxes_, "c")];
        channels_.emplace();
        if (selection.channels) {
            for (const auto& value : *selection.channels) channels_->push_back(selectIndex(value, nchannel, "channel", channelNames));
        } else {
            for (std::int64_t i = 0; i < nchannel; ++i) channels_->push_back(i);
        }
    } else if (selection.channels) {
        bad("channels were selected but storage has no channel axis");
    }
    selectedAxes_.clear();
    shape_.clear();
    for (std::size_t i = 0; i < rawAxes_.size(); ++i) {
        if (rawAxes_[i] == "t") continue;
        selectedAxes_.push_back(rawAxes_[i]);
        shape_.push_back(rawAxes_[i] == "c" ? std::int64_t(channels_->size()) : rawShape_[i]);
    }
    axes_ = selectedAxes_;
    dtype_ = canonicalDtype(infoDtype);
    const std::size_t channelCount = channels_ ? channels_->size() : 0;
    if (source_.encoding) {
        if (source_.encoding->kind == "color" && !channels_) bad("color encoding requires a decoded color channel axis; palette PNG uses indices");
        const auto tableBytes = source_.encoding->pairs.size() * (elementType(infoDtype).itemsize * std::max<std::size_t>(channelCount, 1) + 8) * 2;
        auto lease = cache_->working(tableBytes);
        lookup_ = std::make_unique<Lookup>(*source_.encoding, infoDtype, channelCount);
        dtype_ = lookup_->outputDtype;
        if (lookup_->color) {
            std::vector<std::int64_t> shape;
            std::vector<std::string> axes;
            for (std::size_t i = 0; i < axes_.size(); ++i)
                if (axes_[i] != "c") {
                    axes.push_back(axes_[i]);
                    shape.push_back(shape_[i]);
                }
            axes_ = std::move(axes);
            shape_ = std::move(shape);
        }
    }
    spatialAxes_.clear();
    for (const char* axis : {"z", "y", "x"})
        if (hasAxis(axes_, axis)) spatialAxes_.emplace_back(axis);
    json fill = 0;
    if (!source_.fillValue.is_null()) {
        fill = source_.fillValue;
    } else if (zarr_) {
        fill = backend_->fill;
        if (lookup_) {
            const auto bytes = elementBytes(fill, rawDtype_);
            Block values = Block::allocate(lookup_->color ? std::vector<std::int64_t>{std::int64_t(channelCount)} : std::vector<std::int64_t>{1}, rawDtype_);
            for (std::size_t i = 0; i < values.nbytes; i += bytes.size()) std::memcpy(values.data.get() + i, bytes.data(), bytes.size());
            const Block encoded = lookup_->apply(values, lookup_->color ? std::optional<std::size_t>(0) : std::nullopt);
            fill = elementValue(encoded.data.get(), elementType(dtype_));
        }
    }
    fillBytes_ = elementBytes(fill, dtype_);
    fill_ = elementValue(fillBytes_.data(), elementType(dtype_));
    palette_ = zarr_ ? json(nullptr) : first->palette();
    if (zarr_) {
        registration_ = backend_->registration;
        resolved_ = backend_->resolved;
        resolved_.axes = axes_;
        resolved_.shape = shape_;
        resolved_.dtype = dtypeName(dtype_);
    } else {
        const auto current = imageSelection_;
        std::optional<Selection> original;
        if (source_.registration) original = std::visit([](const auto& r) { return r.declaredGrid; }, *source_.registration);
        Source placement = source_;
        std::optional<VolumeTransform> oldTransform, relative;
        std::optional<std::string> oldUnits;
        if (original && original->spatial() != current.spatial()) {
            ImageFile old(locations_[0], first->format(), *original, fileAxes_, encoded_);
            if (old.info().series != first->info().series) bad("selected TIFF series has no declared connection to the original series; supply placement");
            if (old.info().level == first->info().level)
                std::visit([&](auto& r) { r.declaredGrid = current; }, *placement.registration);
            oldTransform = old.storageFromArray();
            oldUnits = old.storageUnits();
        }
        const auto* volume = source_.registration ? std::get_if<VolumeRegistration>(&*source_.registration) : nullptr;
        if (volume && volume->origin == "catalog-fallback" && !first->storageFromArray() && !oldTransform && original && original->level && current.level)
            relative = VolumeTransform{Affine::blockReduced(std::ldexp(1.0, int(*current.level - *original->level))), {}};
        registration_ = resolveRegistration(placement, current, first->storageFromArray(), first->storageUnits(), oldTransform, oldUnits, relative);
        resolved_ = {locations_[0], current, axes_, shape_, dtypeName(dtype_), registration_, first->storageFromArray(), first->storageUnits()};
    }
    opened_ = true;
    openCompanions();
}

void SourceReader::openCompanions()
{
    companionOrder_.clear();
    for (const auto& companion : source_.companions) {
        Settings settings;
        settings.options = options_;
        settings.decodedCache = cache_;
        auto reader = std::make_shared<SourceReader>(*companion.source, std::move(settings));
        const auto& axes = reader->axes();
        if (hasAxis(axes, "c") && reader->shape()[axisIndex(axes, "c")] != 1) bad("companion " + companion.name + " must have one channel");
        if (companion.sameGrid) {
            const auto spatial = [](const std::vector<std::string>& names, const std::vector<std::int64_t>& sizes) {
                std::map<std::string, std::int64_t> out;
                for (std::size_t i = 0; i < names.size(); ++i)
                    if (names[i] == "z" || names[i] == "y" || names[i] == "x") out[names[i]] = sizes[i];
                return out;
            };
            if (spatial(axes_, shape_) != spatial(axes, reader->shape())) bad("companion " + companion.name + " same_grid shape differs from its parent");
            const auto& child = reader->resolved_;
            if (resolved_.storageFromArray && child.storageFromArray && (resolved_.storageUnits != child.storageUnits || resolved_.storageFromArray->forward != child.storageFromArray->forward))
                bad("companion " + companion.name + " storage metadata contradicts same_grid binding");
            reader->registration_ = registration_;
            reader->resolved_.registration = registration_;
        }
        companionOrder_.push_back(companion.name);
        companions_[companion.name] = std::move(reader);
    }
}

const std::vector<std::string>& SourceReader::axes() { ready(); return axes_; }
const std::vector<std::int64_t>& SourceReader::shape() { ready(); return shape_; }
const std::string& SourceReader::dtype() { ready(); return dtype_; }
const std::vector<std::string>& SourceReader::spatialAxes() { ready(); return spatialAxes_; }
const json& SourceReader::fillValue() { ready(); return fill_; }
const json& SourceReader::palette() { ready(); return palette_; }
const std::optional<SourceRegistration>& SourceReader::registration() { ready(); return registration_; }
const ResolvedSource& SourceReader::resolved() { ready(); return resolved_; }
std::shared_ptr<SourceReader> SourceReader::companion(const std::string& name)
{
    ready();
    const auto it = companions_.find(name);
    if (it == companions_.end()) throw std::out_of_range("no companion '" + name + "'");
    return it->second;
}
std::vector<std::string> SourceReader::companionNames() { ready(); return companionOrder_; }
bool SourceReader::isZarr() { ready(); return zarr_; }
const std::vector<std::string>& SourceReader::rawAxes() { ready(); return rawAxes_; }
std::shared_ptr<zarr::OpenedArray> SourceReader::zarrArray() { ready(); return backend_ ? backend_->reader : nullptr; }
std::size_t SourceReader::zarrDatasets() { ready(); return backend_ ? backend_->datasets.size() : 0; }
Array* SourceReader::directArray()
{
    ready();
    if (!zarr_ || rawAxes_ != spatialAxes_ || lookup_ || fill_ != backend_->fill) return nullptr;
    return backend_->reader->array.get();
}

Block SourceReader::read(const std::vector<std::int64_t>& start, const std::vector<std::int64_t>& size)
{
    return annotated(source_.id, [&]() -> Block {
        ready();
        if (start.size() != spatialAxes_.size() || size.size() != start.size() || std::any_of(size.begin(), size.end(), [](auto n) { return n < 0; }))
            bad("start/size must have " + std::to_string(spatialAxes_.size()) + " spatial entries and nonnegative sizes");
        if (Array* direct = directArray()) {
            Block out = Block::allocate(size, dtype_);
            if (out.nbytes) direct->read_many({start}, {size}, {out.data.get()});
            return out;
        }
        std::map<std::string, std::int64_t> origin, requested;
        for (std::size_t i = 0; i < spatialAxes_.size(); ++i) {
            origin[spatialAxes_[i]] = start[i];
            requested[spatialAxes_[i]] = size[i];
        }
        std::map<std::string, std::int64_t> dims;
        for (std::size_t i = 0; i < rawAxes_.size(); ++i) dims[rawAxes_[i]] = rawShape_[i];
        std::map<std::string, std::pair<std::int64_t, std::int64_t>> bounds;
        for (const auto& axis : spatialAxes_) bounds[axis] = {std::max<std::int64_t>(0, origin[axis]), std::min(dims[axis], origin[axis] + requested[axis])};
        std::vector<std::int64_t> shape;
        for (std::size_t i = 0; i < axes_.size(); ++i) {
            const auto it = requested.find(axes_[i]);
            shape.push_back(it != requested.end() ? it->second : shape_[i]);
        }
        Block output = Block::allocate(shape, dtype_);
        fillBlock(output, fillBytes_);
        if (std::any_of(bounds.begin(), bounds.end(), [](const auto& b) { return b.second.second <= b.second.first; })) return output;
        const std::size_t channelCount = channels_ ? channels_->size() : 0;
        if (zarr_) {
            std::size_t count = 1;
            for (const auto& [axis, range] : bounds) count *= std::size_t(range.second - range.first);
            const std::size_t scratch = count * (std::max<std::size_t>(channelCount, 1) * elementType(rawDtype_).itemsize * 3 + (lookup_ ? 64 + std::size_t(dtype_[2] - '0') : 0));
            auto lease = cache_->working(scratch);
            std::vector<std::pair<std::int64_t, std::int64_t>> ordered;
            for (const auto& axis : spatialAxes_) ordered.push_back(bounds[axis]);
            std::vector<std::int64_t> originList(start);
            readZarr(ordered, originList, output);
            return output;
        }
        // Files: the axes before a page's are chosen one at a time.
        ImageFile& head = file(0);
        const auto prefixShape = head.prefixShape();
        std::vector<std::string> prefixAxes;
        if (stack_) prefixAxes.emplace_back("z");
        prefixAxes.insert(prefixAxes.end(), head.info().axes.begin(), head.info().axes.begin() + std::ptrdiff_t(prefixShape.size()));
        const bool wholeColor = lookup_ && lookup_->color;
        std::vector<std::vector<std::optional<std::int64_t>>> choices;
        for (const auto& axis : prefixAxes) {
            std::vector<std::optional<std::int64_t>> values;
            if (axis == "t") {
                values.emplace_back(*time_);
            } else if (axis == "c") {
                if (wholeColor) values.emplace_back(std::nullopt);
                else for (const auto c : *channels_) values.emplace_back(c);
            } else {
                for (auto v = bounds[axis].first; v < bounds[axis].second; ++v) values.emplace_back(v);
            }
            choices.push_back(std::move(values));
        }
        std::vector<std::size_t> position(choices.size(), 0);
        for (;;) {
            std::map<std::string, std::optional<std::int64_t>> fixed;
            for (std::size_t i = 0; i < choices.size(); ++i) fixed[prefixAxes[i]] = choices[i][position[i]];
            ImageFile& image = file(stack_ ? *fixed["z"] : 0);
            const auto ordinalOf = [&](const std::optional<std::int64_t>& channel) {
                std::int64_t ordinal = 0;
                std::size_t extent = 0;
                for (std::size_t i = stack_ ? 1 : 0; i < prefixAxes.size(); ++i, ++extent) {
                    std::int64_t value = prefixAxes[i] == "c" && channel ? *channel : fixed[prefixAxes[i]].value_or(0);
                    ordinal = ordinal * prefixShape[extent] + value;
                }
                return ordinal;
            };
            const auto& info = image.info();
            std::vector<PageSlice> slices;
            std::vector<std::string> pageAxes;
            for (std::size_t i = 0; i < info.pageAxes.size(); ++i) {
                const auto& axis = info.pageAxes[i];
                if (axis == "t") {
                    slices.push_back({*time_, *time_ + 1, true});
                } else {
                    const auto it = bounds.find(axis);
                    slices.push_back(it != bounds.end() ? PageSlice{it->second.first, it->second.second, false} : PageSlice{0, info.pageShape[i], false});
                    pageAxes.push_back(axis);
                }
            }
            const bool channelInPage = hasAxis(pageAxes, "c");
            std::size_t count = 1;
            for (const auto& [axis, range] : bounds)
                if (hasAxis(pageAxes, axis)) count *= std::size_t(range.second - range.first);
            const auto pageItem = elementType(info.dtype).itemsize;
            const std::size_t scratch = lookup_ || channelInPage ? count * (64 + std::max<std::size_t>(channelCount, 1) * pageItem * 3) : 0;
            std::vector<std::int64_t> target;
            for (const auto& axis : axes_) {
                const auto f = fixed.find(axis);
                if (f != fixed.end()) {
                    target.push_back(axis == "c" ? std::int64_t(std::find(channels_->begin(), channels_->end(), *f->second) - channels_->begin()) : *f->second - origin[axis]);
                } else if (bounds.count(axis)) {
                    target.push_back(bounds[axis].first - origin[axis]);
                } else {
                    target.push_back(0);
                }
            }
            // Rank of what lands in the output, with fixed axes as single entries.
            const auto place = [&](const Block& region) {
                std::vector<std::int64_t> shape;
                for (const auto& axis : axes_) {
                    if (fixed.count(axis)) {
                        shape.push_back(1);
                    } else {
                        const auto at = std::find(pageAxes.begin(), pageAxes.end(), axis);
                        shape.push_back(at == pageAxes.end() ? 1 : -1);
                    }
                }
                std::size_t next = 0;
                for (auto& n : shape)
                    if (n == -1) n = region.shape[next++];
                Block view = region;
                view.shape = shape;
                assign(output, target, view);
            };
            const auto fixedColor = fixed.find("c");
            if (fixedColor != fixed.end() && !fixedColor->second) {
                // One decoded page per channel, combined before the color lookup.
                std::vector<std::int64_t> pageShape;
                std::size_t pageCount = 1;
                for (const auto& axis : pageAxes) {
                    pageShape.push_back(bounds[axis].second - bounds[axis].first);
                    pageCount *= std::size_t(pageShape.back());
                }
                const std::size_t combinedBytes = pageCount * channelCount * pageItem;
                auto lease = cache_->working(combinedBytes + scratch);
                auto combinedShape = pageShape;
                combinedShape.insert(combinedShape.begin(), std::int64_t(channelCount));
                Block combined = Block::allocate(combinedShape, info.dtype);
                for (std::size_t c = 0; c < channelCount; ++c) {
                    const Block region = image.region(ordinalOf((*channels_)[c]), slices, *cache_, 0);
                    std::memcpy(combined.data.get() + c * pageCount * pageItem, region.data.get(), region.nbytes);
                }
                place(lookup_->apply(combined, 0));
            } else {
                auto lease = cache_->working(scratch);
                Block region = image.region(ordinalOf(std::nullopt), slices, *cache_, 0);
                if (channelInPage) region = takeAlong(region, axisIndex(pageAxes, "c"), *channels_);
                if (lookup_) {
                    const auto axis = channelInPage ? std::optional<std::size_t>(axisIndex(pageAxes, "c")) : std::nullopt;
                    region = lookup_->apply(region, axis);
                    if (lookup_->color) {
                        std::erase(pageAxes, "c");
                    }
                }
                place(region);
            }
            // Next prefix combination.
            std::size_t d = choices.size();
            bool more = false;
            while (d-- > 0) {
                if (++position[d] < choices[d].size()) {
                    more = true;
                    break;
                }
                position[d] = 0;
            }
            if (!more) break;
        }
        return output;
    });
}

void SourceReader::readZarr(const std::vector<std::pair<std::int64_t, std::int64_t>>& ordered, const std::vector<std::int64_t>& origin, Block& output)
{
    // Only selected channels and time. The array owns chunk planning, authentication,
    // missing-chunk fill and its caches.
    std::map<std::string, std::pair<std::int64_t, std::int64_t>> bounds;
    std::map<std::string, std::int64_t> originBy;
    for (std::size_t i = 0; i < spatialAxes_.size(); ++i) {
        bounds[spatialAxes_[i]] = ordered[i];
        originBy[spatialAxes_[i]] = origin[i];
    }
    Index starts, sizes;
    for (std::size_t i = 0; i < rawAxes_.size(); ++i) {
        const auto& axis = rawAxes_[i];
        std::int64_t low = 0, high = 1;
        if (bounds.count(axis)) std::tie(low, high) = bounds[axis];
        else if (axis == "t") low = *time_, high = *time_ + 1;
        starts.push_back(low);
        sizes.push_back(high - low);
    }
    const auto channelAxis = axisIndex(rawAxes_, "c");
    const std::vector<std::int64_t> channels = channels_ ? *channels_ : std::vector<std::int64_t>{-1};
    std::vector<Index> regionStarts;
    std::vector<Index> regionSizes;
    std::vector<Block> pieces;
    std::vector<std::uint8_t*> outs;
    for (const auto channel : channels) {
        auto regionStart = starts;
        if (channel >= 0) regionStart[channelAxis] = channel;
        regionStarts.push_back(regionStart);
        regionSizes.push_back(sizes);
        pieces.push_back(Block::allocate(std::vector<std::int64_t>(sizes.begin(), sizes.end()), rawDtype_));
        outs.push_back(pieces.back().data.get());
    }
    backend_->reader->array->read_many(regionStarts, regionSizes, outs);
    Block raw;
    if (pieces.size() > 1) {
        auto shape = pieces[0].shape;
        shape[channelAxis] = std::int64_t(pieces.size());
        raw = Block::allocate(shape, rawDtype_);
        std::vector<std::int64_t> at(shape.size(), 0);
        for (std::size_t c = 0; c < pieces.size(); ++c) {
            at[channelAxis] = std::int64_t(c);
            assign(raw, at, pieces[c]);
        }
    } else {
        raw = pieces[0];
    }
    auto squeezed = raw.shape;
    auto squeezedAxes = rawAxes_;
    if (hasAxis(rawAxes_, "t")) {
        const auto at = axisIndex(rawAxes_, "t");
        squeezed.erase(squeezed.begin() + std::ptrdiff_t(at));
        squeezedAxes.erase(squeezedAxes.begin() + std::ptrdiff_t(at));
    }
    raw.shape = squeezed;
    if (lookup_) {
        const auto axis = hasAxis(selectedAxes_, "c") ? std::optional<std::size_t>(axisIndex(selectedAxes_, "c")) : std::nullopt;
        raw = lookup_->apply(raw, lookup_->color ? axis : std::nullopt);
    }
    std::vector<std::int64_t> target;
    for (const auto& axis : axes_) target.push_back(bounds.count(axis) ? bounds[axis].first - originBy[axis] : 0);
    assign(output, target, raw);
}

std::vector<Block> SourceReader::readMany(const std::vector<std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>>& regions)
{
    return annotated(source_.id, [&] {
        ready();
        std::vector<Block> out;
        if (Array* direct = directArray()) {
            std::vector<Index> starts, sizes;
            std::vector<std::uint8_t*> outs;
            for (const auto& [start, size] : regions) {
                if (start.size() != spatialAxes_.size() || size.size() != start.size() || std::any_of(size.begin(), size.end(), [](auto n) { return n < 0; }))
                    bad("start/size must have " + std::to_string(spatialAxes_.size()) + " spatial entries and nonnegative sizes");
                out.push_back(Block::allocate(size, dtype_));
                starts.push_back(start);
                sizes.push_back(size);
                outs.push_back(out.back().data.get());
            }
            direct->read_many(starts, sizes, outs);
            return out;
        }
        for (const auto& [start, size] : regions) out.push_back(read(start, size));
        return out;
    });
}

SourceReader::Stats SourceReader::stats()
{
    ready();
    Stats out;
    out.decodedCache = cache_->stats();
    out.remoteBytes = encoded_->bytesRead();
    out.encodedCacheHits = encoded_->cacheHits();
    for (const auto& [index, file] : files_) out.imagePayloadBytes += file->bytesRead();
    if (backend_) {
        const auto& options = backend_->options;
        out.zarrCacheBytes = options.memoryCache ? options.memoryCache->nbytes() : 0;
        out.zarrChunkBytes = backend_->reader->meta.chunkBytes();
        auto& array = *backend_->reader->array;
        const auto& counters = array.stats();
        out.zarr = {{"memory_hits", counters.memory_hits.load()}, {"pending_hits", counters.pending_hits.load()}, {"pending_writes", array.pending_writes()},
                    {"disk_hits", counters.disk_hits.load()}, {"remote_fetches", counters.remote_fetches.load()}, {"remote_missing", counters.remote_missing.load()},
                    {"remote_bytes", counters.remote_bytes.load()}, {"write_errors", counters.write_errors.load()}, {"last_write_error", array.last_write_error()}};
    }
    return out;
}

std::string SourceReader::temporaryDirectory() const { return encoded_->temporaryDirectory(); }
bool SourceReader::loaded(std::int64_t plane) const { return files_.count(plane) != 0; }

void SourceReader::close()
{
    if (closed_) return;
    if (backend_) {
        for (auto& [path, opened] : backend_->opened)
            if (opened) opened->close();
        if (backend_->reader) backend_->reader->close();
        backend_.reset();
    }
    for (auto& [name, child] : companions_) child->close();
    companions_.clear();
    files_.clear();
    lookup_.reset();
    encoded_->close();
    if (ownsCache_) cache_->clear();
    closed_ = true;
}
} // namespace vcr::sources
