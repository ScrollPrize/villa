#include "vc_reader/sources.hpp"

#include <charconv>
#include <cmath>
#include <set>

#include "sigv4.hpp"

namespace vcr::sources {
namespace {
using nlohmann::json;

[[noreturn]] void bad(const std::string& message) { throw std::invalid_argument(message); }
[[noreturn]] void wrongType(const std::string& message) { throw DeclarationTypeError(message); }

void name(const std::string& value, const std::string& field)
{
    if (value.empty()) bad(field + " must be a nonempty string");
}
void name(const std::optional<std::string>& value, const std::string& field)
{
    if (!value || value->empty()) bad(field + " must be a nonempty string");
}
void index(std::int64_t value, const std::string& field)
{
    if (value < 0) bad(field + " must be a nonnegative integer");
}
void axesOf(const std::optional<std::vector<std::string>>& axes)
{
    if (!axes) return;
    if (axes->empty() || std::set(axes->begin(), axes->end()).size() != axes->size())
        bad("axes must be nonempty and unique");
    for (const auto& axis : *axes) name(axis, "axis");
}
void components(const std::optional<std::vector<std::string>>& value)
{
    if (value && (value->size() != 3 || std::set(value->begin(), value->end()) != std::set<std::string>{"x", "y", "z"}))
        bad("component_order must be a permutation of x, y, z");
}
void vector3(const Vec3& value, const std::string& field)
{
    for (const double v : value)
        if (!std::isfinite(v)) bad(field + " must contain three finite numbers");
}
void affineOf(const std::optional<VolumeTransform>& value)
{
    if (value) (void)value->forward.inverse();  // rejects singular declarations
}
void selector(const Selector& value, const std::string& field)
{
    if (const auto* text = std::get_if<std::string>(&value)) name(*text, field);
    else index(std::get<std::int64_t>(value), field);
}
} // namespace

Selection Selection::spatial() const
{
    Selection out = *this;
    out.channels.reset();
    out.time.reset();
    return out;
}

void validate(const Frame& value)
{
    name(value.sampleId, "sample_id");
    name(value.volumeId, "volume_id");
    index(value.level, "level");
}

void validate(const LevelRequest& value) { validate(value.reference); }

void validate(const Selection& value)
{
    if (int(value.arrayPath.has_value()) + int(value.level.has_value()) + int(value.corresponding.has_value()) > 1)
        bad("select only one of array_path, level, or corresponding");
    if (value.arrayPath) name(*value.arrayPath, "array_path");
    if (value.level) index(*value.level, "level");
    if (value.corresponding) validate(*value.corresponding);
    if (value.series) selector(*value.series, "series");
    if (value.time) selector(*value.time, "time");
    if (value.channels) {
        for (const auto& channel : *value.channels) selector(channel, "channels");
        if (value.channels->empty() || std::set(value.channels->begin(), value.channels->end()).size() != value.channels->size())
            bad("channels must be nonempty and unique");
    }
}

void validate(const Storage& value)
{
    if (value.location && !value.locations.empty()) bad("use location or ordered stack locations, not both");
    for (const auto& location : value.locations) name(location, "location");
    for (const auto& location : value.alternatives) name(location, "location");
    if (value.location) name(*value.location, "location");
    if (value.format) name(*value.format, "format");
    if (!value.locations.empty() && value.format != "stack") bad("ordered locations require format='stack'");
    if (value.format == "stack" && value.locations.empty()) bad("a stack needs explicit ordered locations");
    validate(value.selection);
    if (value.gridLevel) index(*value.gridLevel, "grid_level");
}

void validate(const Plane& value)
{
    vector3(value.origin, "origin");
    vector3(value.basisX, "basis_x");
    vector3(value.basisY, "basis_y");
    vector3(value.thickness, "thickness");
    const auto& x = value.basisX;
    const auto& y = value.basisY;
    const auto& n = value.thickness;
    const Vec3 cross{x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]};
    const auto zero = [](const Vec3& v) { return v[0] == 0 && v[1] == 0 && v[2] == 0; };
    if (zero(cross) || zero(n)) bad("plane needs independent pixel bases and nonzero thickness");
    // Allows ordinary rounding of an explicitly computed normal, not a change to
    // sample support or interpolation tolerances.
    for (const Vec3* basis : {&x, &y}) {
        const auto& b = *basis;
        const double dot = b[0] * n[0] + b[1] * n[1] + b[2] * n[2];
        const double scale = std::sqrt((b[0] * b[0] + b[1] * b[1] + b[2] * b[2]) * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]));
        if (std::abs(dot) > 1e-12 * scale) bad("plane thickness must be perpendicular to its pixel bases");
    }
}

void validate(const VolumeRegistration& value)
{
    name(value.sampleId, "sample_id");
    name(value.volumeId, "volume_id");
    affineOf(value.affine);
    if (value.mode != "array" && value.mode != "storage") bad("registration mode must be 'array' or 'storage'");
    if (value.mode == "storage") name(value.storageFrame, "storage_frame");
    else if (value.storageFrame) bad("storage_frame requires mode='storage'");
    if (value.plane) {
        if (value.affine || value.mode != "array") bad("plane placement requires a Plane and no affine/storage mode");
        validate(*value.plane);
    }
    if (value.declaredGrid) validate(*value.declaredGrid);
    if (value.storageUnits) {
        name(*value.storageUnits, "storage_units");
        if (value.mode != "storage") bad("storage_units requires storage placement");
    }
    if (value.volumeSpacingUm) {
        vector3(*value.volumeSpacingUm, "volume_spacing_um");
        for (const double v : *value.volumeSpacingUm)
            if (v <= 0) bad("volume spacing must be positive");
    }
    name(value.origin, "registration origin");
}

void validate(const ValueFrame& value)
{
    if (value.kind == "volume") {
        name(value.sampleId, "sample_id");
        name(value.volumeId, "volume_id");
        if (value.affine && value.level) bad("value frame selects an affine or a level");
        affineOf(value.affine);
        if (value.level) index(*value.level, "level");
    } else if (value.kind == "external") {
        name(value.name, "external frame name");
        name(value.units, "external frame units");
        if (value.sampleId || value.volumeId || value.affine || value.level)
            bad("external frame cannot declare a volume transform");
    } else {
        bad("value frame kind must be 'volume' or 'external'");
    }
}

void validate(const Encoding& value)
{
    if (value.kind != "integer" && value.kind != "color") bad("encoding kind must be 'integer' or 'color'");
    const auto signedType = value.dtype.starts_with("int");
    const auto digits = value.dtype.substr(signedType ? 3 : value.dtype.starts_with("uint") ? 4 : 0);
    if ((!signedType && !value.dtype.starts_with("uint")) || (digits != "8" && digits != "16" && digits != "32" && digits != "64"))
        bad("encoding needs an integer output dtype");
    const int bits = std::stoi(digits);
    const WideInt low{signedType ? std::uint64_t(1) << (bits - 1) : 0, signedType};
    const WideInt high{signedType ? (std::uint64_t(1) << (bits - 1)) - 1 : (bits == 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << bits) - 1), false};
    const auto inRange = [&](const WideInt& v) {
        if (v < low || high < v) bad("encoding output is outside its integer dtype");
    };
    std::set<std::vector<WideInt>> keys;
    std::size_t colorSize = 0;
    for (const auto& pair : value.pairs) {
        if (value.kind == "integer") {
            if (pair.key.size() != 1) bad("integer lookup keys must be integers");
        } else {
            if (pair.key.empty()) bad("color keys must be tuples of nonnegative integers");
            for (const auto& component : pair.key)
                if (component.negative) bad("color keys must be tuples of nonnegative integers");
            if (!colorSize) colorSize = pair.key.size();
            if (pair.key.size() != colorSize) bad("color keys must have equal component counts");
        }
        if (!keys.insert(pair.key).second) bad("duplicate encoding key");
        inRange(pair.value);
    }
    if (value.fallback) inRange(*value.fallback);
}

void validatePixelAffine(const PixelAffine& value)
{
    for (const auto& row : value)
        for (const double v : row)
            if (!std::isfinite(v)) bad("pixel-to-UV transform must be finite");
    if (value[0][0] * value[1][1] == value[0][1] * value[1][0]) bad("pixel-to-UV transform is singular");
}

void validate(const Surface& value)
{
    name(value.id, "id");
    name(value.sampleId, "sample_id");
    name(value.volumeId, "volume_id");
    validate(value.geometry);
    axesOf(value.axes);
    components(value.componentOrder);
    affineOf(value.xyzToVolume);
    if (value.pixelToUv) validatePixelAffine(*value.pixelToUv);
    if (value.spacingUm) {
        vector3(*value.spacingUm, "spacing_um");
        for (const double v : *value.spacingUm)
            if (v <= 0) bad("physical spacing must be positive");
    }
}

void validate(const SurfaceRegistration& value)
{
    validate(value.surface);
    if (value.pixelToUv) validatePixelAffine(*value.pixelToUv);
    name(value.origin, "registration origin");
    if (value.declaredGrid) validate(*value.declaredGrid);
}

void validate(const Companion& value)
{
    name(value.name, "companion name");
    if (value.name == "coverage" || value.name == "conflict") bad("reserved companion name: " + value.name);
    if (value.source->role != "label" && value.source->role != "scalar") bad("companion must be a label or scalar Source");
    if (value.sameGrid && value.source->registration) bad("same_grid companion cannot also declare registration");
    if (!value.sameGrid && !value.source->registration) bad("companion needs registration or explicit same_grid=True");
}

void validate(const QueryContext& value)
{
    validate(value.reference);
    if (value.relation != "direct" && value.relation != "transform") bad("unknown query relation");
    (void)value.transform.forward.inverse();
}

Source finished(Source source)
{
    name(source.id, "source ID");
    if (!source.id.starts_with("catalog:") && !source.id.starts_with("user:")) source.id = "user:" + source.id;
    const auto where = source.id + ": ";
    try {
        validate(source.storage);
        if (source.role && *source.role != "image" && *source.role != "label" && *source.role != "scalar" &&
            *source.role != "vector" && *source.role != "normal" && *source.role != "coordinate")
            bad("unknown role '" + *source.role + "'");
        axesOf(source.axes);
        components(source.componentOrder);
        if (source.registration) {
            if (auto* volume = std::get_if<VolumeRegistration>(&*source.registration)) {
                validate(*volume);
                if (!volume->declaredGrid && volume->mode == "array") volume->declaredGrid = source.storage.selection;
            } else {
                auto& surface = std::get<SurfaceRegistration>(*source.registration);
                validate(surface);
                if (!surface.declaredGrid) surface.declaredGrid = source.storage.selection;
            }
        }
        if (source.valueFrame) validate(*source.valueFrame);
        if (source.encoding) validate(*source.encoding);
        if (source.shape) {
            if (source.shape->empty()) bad("shape must be a nonempty sequence");
            for (const auto size : *source.shape) index(size, "shape");
            if (source.axes && source.shape->size() != source.axes->size()) bad("shape and axes have different ranks");
        }
        if (!source.provenance->is_object()) wrongType("provenance must be a mapping");
        std::set<std::string> names;
        for (const auto& companion : source.companions) {
            validate(companion);
            if (!names.insert(companion.name).second) bad("duplicate companion name");
        }
        for (const auto& identifier : source.supersedes) name(identifier, "superseded source ID");
        if (source.query) validate(*source.query);
    } catch (const DeclarationTypeError& error) {
        throw DeclarationTypeError(where + error.what());
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(where + error.what());
    }
    return source;
}

std::vector<std::string> Source::unresolvedFields() const
{
    std::vector<std::string> missing;
    if (!storage.location && storage.locations.empty()) missing.push_back("storage.location");
    if (!storage.format) missing.push_back("storage.format");
    if (!role) missing.push_back("role");
    if (!axes) missing.push_back("axes");
    if (!registration) {
        missing.push_back("registration");
    } else if (const auto* volume = std::get_if<VolumeRegistration>(&*registration)) {
        if (!volume->affine && !volume->plane) missing.push_back("registration.placement");
    } else {
        const auto& surface = std::get<SurfaceRegistration>(*registration);
        if (!surface.pixelToUv) missing.push_back("registration.pixel_to_uv");
        if (!surface.surface.xyzToVolume) missing.push_back("registration.surface.xyz_to_volume");
        if (!surface.surface.pixelToUv) missing.push_back("registration.surface.pixel_to_uv");
        if (!surface.surface.geometry.location) missing.push_back("registration.surface.geometry.location");
        if (!surface.surface.geometry.format) missing.push_back("registration.surface.geometry.format");
        if (!surface.surface.axes) missing.push_back("registration.surface.axes");
        if (!surface.surface.componentOrder) missing.push_back("registration.surface.component_order");
    }
    if (storage.pyramid && storage.selection == Selection{}) missing.push_back("storage.selection");
    if (storage.selection.corresponding) missing.push_back("storage.selection.corresponding");
    if (role == "vector" || role == "normal" || role == "coordinate") {
        if (!componentOrder) missing.push_back("component_order");
        if (!valueFrame) missing.push_back("value_frame");
        else if (valueFrame->kind == "volume" && !valueFrame->affine && !valueFrame->level)
            missing.push_back("value_frame.placement");
    }
    if (role == "vector" && !normalize) missing.push_back("normalize");
    for (const auto& companion : companions)
        for (const auto& field : companion.source->unresolvedFields())
            if (!(companion.sameGrid && field == "registration"))
                missing.push_back("companions." + companion.name + "." + field);
    return missing;
}

Source Source::withSelection(Selection selection) const
{
    Source out = *this;
    out.storage.selection = std::move(selection);
    return finished(std::move(out));
}

Source selectLevel(const Source& source, std::int64_t level)
{
    if (level < 0) bad("level must be nonnegative");
    if (source.storage.pyramid) {
        Source out = source;
        out.storage.selection.level = level;
        out.storage.selection.arrayPath.reset();
        out.storage.selection.corresponding.reset();
        return finished(std::move(out));
    }
    if (level) bad("a single-resolution array has only level 0");
    return source;
}

// ---- schema 1 ----
namespace {
template <class T> json orNull(const std::optional<T>& value) { return value ? json(*value) : json(nullptr); }
json tagged(const char* type, json fields) { return json{{"type", type}, {"fields", std::move(fields)}}; }
template <class T> json encodeOptional(const std::optional<T>& value) { return value ? encode(*value) : json(nullptr); }
json toJson(const Selector& value)
{
    if (const auto* text = std::get_if<std::string>(&value)) return *text;
    return std::get<std::int64_t>(value);
}
json toJson(const std::optional<Selector>& value) { return value ? toJson(*value) : json(nullptr); }
json toJson(const WideInt& value)
{
    if (!value.negative) return value.magnitude;
    return -static_cast<std::int64_t>(value.magnitude - 1) - 1;
}
template <class T> json encodeAll(const std::vector<T>& values)
{
    json out = json::array();
    for (const auto& value : values) out.push_back(encode(value));
    return out;
}

const json* field(const json& fields, const char* key)
{
    if (!fields.is_object()) bad("descriptor fields must be a mapping");
    const auto it = fields.find(key);
    return it == fields.end() || it->is_null() ? nullptr : &*it;
}
void onlyFields(const json& fields, std::initializer_list<const char*> names, const char* type)
{
    if (!fields.is_object()) bad("descriptor fields must be a mapping");
    for (const auto& item : fields.items()) {
        if (std::none_of(names.begin(), names.end(), [&](const char* n) { return item.key() == n; }))
            wrongType(std::string(type) + " has no field '" + item.key() + "'");
    }
}
std::optional<std::string> text(const json& fields, const char* key)
{
    const auto* v = field(fields, key);
    if (!v) return std::nullopt;
    if (!v->is_string()) bad(std::string(key) + " must be a nonempty string");
    return v->get<std::string>();
}
std::string required(const json& fields, const char* key)
{
    auto value = text(fields, key);
    if (!value) wrongType(std::string("missing field ") + key);
    return *value;
}
std::int64_t integerValue(const json& value, const std::string& key)
{
    if (value.is_number_unsigned() && value.get<std::uint64_t>() > std::uint64_t(INT64_MAX)) bad(key + " is out of range");
    if (!value.is_number_integer()) bad(key + " must be an integer");
    return value.get<std::int64_t>();
}
std::optional<std::int64_t> integer(const json& fields, const char* key)
{
    const auto* v = field(fields, key);
    if (!v) return std::nullopt;
    return integerValue(*v, key);
}
std::optional<bool> boolean(const json& fields, const char* key)
{
    const auto* v = field(fields, key);
    if (!v) return std::nullopt;
    if (!v->is_boolean()) wrongType(std::string(key) + " must be boolean");
    return v->get<bool>();
}
std::vector<std::string> strings(const json& value, const char* key)
{
    if (!value.is_array()) wrongType(std::string(key) + " must be a sequence");
    std::vector<std::string> out;
    for (const auto& item : value) {
        if (!item.is_string()) bad(std::string(key) + " must contain nonempty strings");
        out.push_back(item.get<std::string>());
    }
    return out;
}
std::vector<std::string> stringList(const json& fields, const char* key)
{
    const auto* v = field(fields, key);
    return v ? strings(*v, key) : std::vector<std::string>{};
}
std::optional<std::vector<std::string>> optionalStrings(const json& fields, const char* key)
{
    const auto* v = field(fields, key);
    if (!v) return std::nullopt;
    return strings(*v, key);
}
Selector selectorValue(const json& value, const char* key)
{
    if (value.is_string()) return value.get<std::string>();
    if (value.is_boolean() || !value.is_number_integer()) bad(std::string(key) + " must be a nonnegative integer");
    return integerValue(value, key);
}
std::optional<Selector> optionalSelector(const json& fields, const char* key)
{
    const auto* v = field(fields, key);
    if (!v) return std::nullopt;
    return selectorValue(*v, key);
}
double number(const json& value, const std::string& key)
{
    if (!value.is_number()) bad(key + " must contain three finite numbers");
    const double v = value.get<double>();
    if (!std::isfinite(v)) bad("nonfinite declaration");
    return v;
}
Vec3 vec3Value(const json& value, const std::string& key)
{
    if (!value.is_array() || value.size() != 3) bad(key + " must contain three finite numbers");
    return {number(value[0], key), number(value[1], key), number(value[2], key)};
}
std::optional<Vec3> optionalVec3(const json& fields, const char* key)
{
    const auto* v = field(fields, key);
    if (!v) return std::nullopt;
    return vec3Value(*v, key);
}
Vec3 requiredVec3(const json& fields, const char* key)
{
    auto value = optionalVec3(fields, key);
    if (!value) wrongType(std::string("missing field ") + key);
    return *value;
}
std::optional<PixelAffine> pixelAffine(const json& fields, const char* key)
{
    const auto* v = field(fields, key);
    if (!v) return std::nullopt;
    if (!v->is_array() || v->size() != 2 || !(*v)[0].is_array() || !(*v)[1].is_array() || (*v)[0].size() != 3 || (*v)[1].size() != 3)
        bad("pixel-to-UV transform must have shape 2x3");
    PixelAffine out;
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c) out[r][c] = number((*v)[r][c], key);
    return out;
}
WideInt wideInt(const json& value, const char* what)
{
    if (value.is_number_unsigned()) return {value.get<std::uint64_t>(), false};
    if (value.is_number_integer()) {
        const auto v = value.get<std::int64_t>();
        return v < 0 ? WideInt{~static_cast<std::uint64_t>(v) + 1, true} : WideInt{static_cast<std::uint64_t>(v), false};
    }
    bad(std::string(what) + " must be integers");
}
template <class T> T decodeTagged(const json& value, const char* type)
{
    if (!value.is_object() || value.value("type", "") != type || !value.contains("fields") || value.size() != 2)
        wrongType(std::string("expected a ") + type);
    return decodeFields<T>(value.at("fields"));
}
template <class T> std::optional<T> optionalTagged(const json& fields, const char* key, const char* type)
{
    const auto* v = field(fields, key);
    if (!v) return std::nullopt;
    return decodeTagged<T>(*v, type);
}
} // namespace

template <> json encode(const VolumeTransform& v)
{
    return tagged("Affine", {{"matrix", v.forward.json()},
                             {"derivation_path", v.derivationPath.empty() ? json(nullptr) : json(v.derivationPath)}});
}
template <> VolumeTransform decodeFields(const json& f)
{
    onlyFields(f, {"matrix", "derivation_path"}, "Affine");
    const auto* matrix = field(f, "matrix");
    if (!matrix) wrongType("missing field matrix");
    return {Affine::fromJson(*matrix), text(f, "derivation_path").value_or("")};
}
template <> json encode(const OpenDataVolumeCoverage& v)
{
    return tagged("VolumeCoverage", {{"volume_id", v.volumeId}, {"bbox_min", v.bbox.low}, {"bbox_max", v.bbox.high},
                                     {"overlap_ratio", v.overlapRatio},
                                     {"computed_at", v.computedAt.empty() ? json(nullptr) : json(v.computedAt)}});
}
template <> OpenDataVolumeCoverage decodeFields(const json& f)
{
    onlyFields(f, {"volume_id", "bbox_min", "bbox_max", "overlap_ratio", "computed_at"}, "VolumeCoverage");
    OpenDataVolumeCoverage v;
    v.volumeId = required(f, "volume_id");
    v.bbox.low = requiredVec3(f, "bbox_min");
    v.bbox.high = requiredVec3(f, "bbox_max");
    const auto* ratio = field(f, "overlap_ratio");
    if (!ratio) wrongType("missing field overlap_ratio");
    v.overlapRatio = number(*ratio, "overlap_ratio");
    v.computedAt = text(f, "computed_at").value_or("");
    return v;
}
template <> json encode(const Frame& v)
{
    return tagged("Frame", {{"sample_id", v.sampleId}, {"volume_id", v.volumeId}, {"level", v.level}});
}
template <> Frame decodeFields(const json& f)
{
    onlyFields(f, {"sample_id", "volume_id", "level"}, "Frame");
    Frame v{required(f, "sample_id"), required(f, "volume_id"), integer(f, "level").value_or(0)};
    validate(v);
    return v;
}
template <> json encode(const LevelRequest& v)
{
    return tagged("LevelRequest", {{"reference", encode(v.reference)}, {"sublevel", orNull(v.sublevel)}});
}
template <> LevelRequest decodeFields(const json& f)
{
    onlyFields(f, {"reference", "sublevel"}, "LevelRequest");
    auto reference = optionalTagged<Frame>(f, "reference", "Frame");
    if (!reference) wrongType("reference must be a Frame");
    return {std::move(*reference), integer(f, "sublevel")};
}
template <> json encode(const Selection& v)
{
    json channels(nullptr);
    if (v.channels) {
        channels = json::array();
        for (const auto& channel : *v.channels) channels.push_back(toJson(channel));
    }
    return tagged("Selection", {{"array_path", orNull(v.arrayPath)}, {"level", orNull(v.level)}, {"series", toJson(v.series)},
                                {"channels", channels}, {"time", toJson(v.time)}, {"corresponding", encodeOptional(v.corresponding)}});
}
template <> Selection decodeFields(const json& f)
{
    onlyFields(f, {"array_path", "level", "series", "channels", "time", "corresponding"}, "Selection");
    Selection v;
    v.arrayPath = text(f, "array_path");
    v.level = integer(f, "level");
    v.series = optionalSelector(f, "series");
    v.time = optionalSelector(f, "time");
    if (const auto* channels = field(f, "channels")) {
        if (!channels->is_array()) wrongType("channels must be a sequence of indices or names");
        v.channels.emplace();
        for (const auto& channel : *channels) v.channels->push_back(selectorValue(channel, "channels"));
    }
    v.corresponding = optionalTagged<LevelRequest>(f, "corresponding", "LevelRequest");
    validate(v);
    return v;
}
template <> json encode(const Storage& v)
{
    return tagged("Storage", {{"location", orNull(v.location)}, {"format", orNull(v.format)}, {"locations", v.locations},
                              {"alternatives", v.alternatives}, {"selection", encode(v.selection)},
                              {"pyramid", v.pyramid}, {"grid_level", orNull(v.gridLevel)}});
}
template <> Storage decodeFields(const json& f)
{
    onlyFields(f, {"location", "format", "locations", "alternatives", "selection", "pyramid", "grid_level"}, "Storage");
    Storage v;
    v.location = text(f, "location");
    v.format = text(f, "format");
    v.locations = stringList(f, "locations");
    v.alternatives = stringList(f, "alternatives");
    v.selection = optionalTagged<Selection>(f, "selection", "Selection").value_or(Selection{});
    v.pyramid = boolean(f, "pyramid").value_or(false);
    v.gridLevel = integer(f, "grid_level");
    validate(v);
    return v;
}
template <> json encode(const Plane& v)
{
    return tagged("Plane", {{"origin", v.origin}, {"basis_x", v.basisX}, {"basis_y", v.basisY}, {"thickness", v.thickness}});
}
template <> Plane decodeFields(const json& f)
{
    onlyFields(f, {"origin", "basis_x", "basis_y", "thickness"}, "Plane");
    Plane v{requiredVec3(f, "origin"), requiredVec3(f, "basis_x"), requiredVec3(f, "basis_y"), requiredVec3(f, "thickness")};
    validate(v);
    return v;
}
template <> json encode(const VolumeRegistration& v)
{
    return tagged("VolumeRegistration", {{"sample_id", v.sampleId}, {"volume_id", v.volumeId}, {"affine", encodeOptional(v.affine)},
        {"plane", encodeOptional(v.plane)}, {"mode", v.mode}, {"storage_frame", orNull(v.storageFrame)},
        {"declared_grid", encodeOptional(v.declaredGrid)}, {"origin", v.origin}, {"storage_units", orNull(v.storageUnits)},
        {"volume_spacing_um", orNull(v.volumeSpacingUm)}});
}
template <> VolumeRegistration decodeFields(const json& f)
{
    onlyFields(f, {"sample_id", "volume_id", "affine", "plane", "mode", "storage_frame", "declared_grid", "origin", "storage_units", "volume_spacing_um"}, "VolumeRegistration");
    VolumeRegistration v;
    v.sampleId = required(f, "sample_id");
    v.volumeId = required(f, "volume_id");
    v.affine = optionalTagged<VolumeTransform>(f, "affine", "Affine");
    v.plane = optionalTagged<Plane>(f, "plane", "Plane");
    v.mode = text(f, "mode").value_or("array");
    v.storageFrame = text(f, "storage_frame");
    v.declaredGrid = optionalTagged<Selection>(f, "declared_grid", "Selection");
    v.origin = text(f, "origin").value_or("explicit");
    v.storageUnits = text(f, "storage_units");
    v.volumeSpacingUm = optionalVec3(f, "volume_spacing_um");
    validate(v);
    return v;
}
template <> json encode(const ValueFrame& v)
{
    return tagged("ValueFrame", {{"kind", v.kind}, {"sample_id", orNull(v.sampleId)}, {"volume_id", orNull(v.volumeId)},
        {"affine", encodeOptional(v.affine)}, {"level", orNull(v.level)}, {"name", orNull(v.name)}, {"units", orNull(v.units)}});
}
template <> ValueFrame decodeFields(const json& f)
{
    onlyFields(f, {"kind", "sample_id", "volume_id", "affine", "level", "name", "units"}, "ValueFrame");
    ValueFrame v;
    v.kind = text(f, "kind").value_or("volume");
    v.sampleId = text(f, "sample_id");
    v.volumeId = text(f, "volume_id");
    v.affine = optionalTagged<VolumeTransform>(f, "affine", "Affine");
    v.level = integer(f, "level");
    v.name = text(f, "name");
    v.units = text(f, "units");
    validate(v);
    return v;
}
template <> json encode(const Encoding& v)
{
    json pairs = json::array();
    for (const auto& pair : v.pairs) {
        json key;
        if (v.kind == "integer") {
            key = toJson(pair.key.at(0));
        } else {
            key = json::array();
            for (const auto& component : pair.key) key.push_back(toJson(component));
        }
        pairs.push_back(json::array({key, toJson(pair.value)}));
    }
    return tagged("Encoding", {{"kind", v.kind}, {"pairs", pairs}, {"dtype", v.dtype},
                               {"fallback", v.fallback ? toJson(*v.fallback) : json(nullptr)}});
}
template <> Encoding decodeFields(const json& f)
{
    onlyFields(f, {"kind", "pairs", "dtype", "fallback"}, "Encoding");
    Encoding v;
    v.kind = required(f, "kind");
    if (v.kind != "integer" && v.kind != "color") bad("encoding kind must be 'integer' or 'color'");
    const auto* pairs = field(f, "pairs");
    if (!pairs || !pairs->is_array()) wrongType("encoding pairs must be a sequence");
    for (const auto& pair : *pairs) {
        if (!pair.is_array() || pair.size() != 2) bad("encoding pairs contain a key and output value");
        EncodingPair item;
        if (v.kind == "integer") {
            item.key = {wideInt(pair[0], "integer lookup keys")};
        } else {
            if (!pair[0].is_array()) bad("color keys must be tuples of nonnegative integers");
            for (const auto& component : pair[0]) item.key.push_back(wideInt(component, "color keys"));
        }
        if (!pair[1].is_number_integer()) bad("encoding output is outside its integer dtype");
        item.value = wideInt(pair[1], "encoding outputs");
        v.pairs.push_back(std::move(item));
    }
    v.dtype = required(f, "dtype");
    if (const auto* fallback = field(f, "fallback")) {
        if (!fallback->is_number_integer()) bad("encoding output is outside its integer dtype");
        v.fallback = wideInt(*fallback, "encoding outputs");
    }
    validate(v);
    return v;
}
namespace {
json encodeShared(const std::optional<Shared<Source>>& v) { return v ? encode(**v) : json(nullptr); }
std::optional<Shared<Source>> sharedSource(const json& fields, const char* key)
{
    auto v = optionalTagged<Source>(fields, key, "Source");
    if (!v) return std::nullopt;
    return Shared<Source>(std::move(*v));
}
} // namespace
template <> json encode(const Surface& v)
{
    return tagged("Surface", {{"id", v.id}, {"geometry", encode(v.geometry)}, {"sample_id", v.sampleId}, {"volume_id", v.volumeId},
        {"axes", orNull(v.axes)}, {"component_order", orNull(v.componentOrder)}, {"xyz_to_volume", encodeOptional(v.xyzToVolume)},
        {"pixel_to_uv", orNull(v.pixelToUv)}, {"spacing_um", orNull(v.spacingUm)}, {"mask", encodeShared(v.mask)},
        {"normals", encodeShared(v.normals)}, {"tifxyz_mask", v.tifxyzMask}});
}
template <> Surface decodeFields(const json& f)
{
    onlyFields(f, {"id", "geometry", "sample_id", "volume_id", "axes", "component_order", "xyz_to_volume", "pixel_to_uv", "spacing_um", "mask", "normals", "tifxyz_mask"}, "Surface");
    Surface v;
    v.id = required(f, "id");
    auto geometry = optionalTagged<Storage>(f, "geometry", "Storage");
    if (!geometry) wrongType("geometry must be Storage");
    v.geometry = std::move(*geometry);
    v.sampleId = required(f, "sample_id");
    v.volumeId = required(f, "volume_id");
    v.axes = optionalStrings(f, "axes");
    v.componentOrder = optionalStrings(f, "component_order");
    v.xyzToVolume = optionalTagged<VolumeTransform>(f, "xyz_to_volume", "Affine");
    v.pixelToUv = pixelAffine(f, "pixel_to_uv");
    v.spacingUm = optionalVec3(f, "spacing_um");
    v.mask = sharedSource(f, "mask");
    v.normals = sharedSource(f, "normals");
    v.tifxyzMask = boolean(f, "tifxyz_mask").value_or(true);
    validate(v);
    return v;
}
template <> json encode(const SurfaceRegistration& v)
{
    return tagged("SurfaceRegistration", {{"surface", encode(v.surface)}, {"pixel_to_uv", orNull(v.pixelToUv)},
                                          {"origin", v.origin}, {"declared_grid", encodeOptional(v.declaredGrid)}});
}
template <> SurfaceRegistration decodeFields(const json& f)
{
    onlyFields(f, {"surface", "pixel_to_uv", "origin", "declared_grid"}, "SurfaceRegistration");
    auto surface = optionalTagged<Surface>(f, "surface", "Surface");
    if (!surface) wrongType("surface must be a Surface");
    SurfaceRegistration v{std::move(*surface), pixelAffine(f, "pixel_to_uv"), text(f, "origin").value_or("explicit"),
                          optionalTagged<Selection>(f, "declared_grid", "Selection")};
    validate(v);
    return v;
}
template <> json encode(const Companion& v)
{
    return tagged("Companion", {{"name", v.name}, {"source", encode(*v.source)}, {"same_grid", v.sameGrid}});
}
template <> Companion decodeFields(const json& f)
{
    onlyFields(f, {"name", "source", "same_grid"}, "Companion");
    auto source = sharedSource(f, "source");
    if (!source) wrongType("companion must be a label or scalar Source");
    Companion v{required(f, "name"), std::move(*source), boolean(f, "same_grid").value_or(false)};
    validate(v);
    return v;
}
template <> json encode(const QueryContext& v)
{
    return tagged("QueryContext", {{"reference", encode(v.reference)}, {"relation", v.relation},
                                   {"transform", encode(v.transform)}, {"coverage", encodeOptional(v.coverage)}});
}
template <> QueryContext decodeFields(const json& f)
{
    onlyFields(f, {"reference", "relation", "transform", "coverage"}, "QueryContext");
    auto reference = optionalTagged<Frame>(f, "reference", "Frame");
    if (!reference) wrongType("reference must be a Frame");
    auto transform = optionalTagged<VolumeTransform>(f, "transform", "Affine");
    if (!transform) wrongType("query transform must be an Affine");
    QueryContext v{std::move(*reference), required(f, "relation"), std::move(*transform),
                   optionalTagged<OpenDataVolumeCoverage>(f, "coverage", "VolumeCoverage")};
    validate(v);
    return v;
}
template <> json encode(const Source& v)
{
    json registration(nullptr);
    if (v.registration)
        registration = std::visit([](const auto& r) { return encode(r); }, *v.registration);
    return tagged("Source", {{"id", v.id}, {"storage", encode(v.storage)}, {"role", orNull(v.role)}, {"axes", orNull(v.axes)},
        {"registration", registration}, {"component_order", orNull(v.componentOrder)}, {"value_frame", encodeOptional(v.valueFrame)},
        {"normalize", orNull(v.normalize)}, {"encoding", encodeOptional(v.encoding)}, {"companions", encodeAll(v.companions)},
        {"provenance", encodeMapping(*v.provenance)}, {"artifact_id", orNull(v.artifactId)}, {"artifact_kind", orNull(v.artifactKind)},
        {"sample_id", orNull(v.sampleId)}, {"owner_id", orNull(v.ownerId)}, {"model_id", orNull(v.modelId)}, {"label_set", orNull(v.labelSet)},
        {"supersedes", v.supersedes}, {"coverage", encodeAll(v.coverage)}, {"derived", v.derived}, {"shape", orNull(v.shape)},
        {"dtype", orNull(v.dtype)}, {"fill_value", v.fillValue}, {"query", encodeOptional(v.query)}});
}
template <> Source decodeFields(const json& f)
{
    onlyFields(f, {"id", "storage", "role", "axes", "registration", "component_order", "value_frame", "normalize", "encoding", "companions",
                   "provenance", "artifact_id", "artifact_kind", "sample_id", "owner_id", "model_id", "label_set", "supersedes", "coverage",
                   "derived", "shape", "dtype", "fill_value", "query"}, "Source");
    Source v;
    v.id = required(f, "id");
    v.storage = optionalTagged<Storage>(f, "storage", "Storage").value_or(Storage{});
    v.role = text(f, "role");
    v.axes = optionalStrings(f, "axes");
    if (const auto* registration = field(f, "registration")) {
        if (registration->is_object() && registration->value("type", "") == "SurfaceRegistration")
            v.registration = decodeTagged<SurfaceRegistration>(*registration, "SurfaceRegistration");
        else if (registration->is_object() && registration->value("type", "") == "VolumeRegistration")
            v.registration = decodeTagged<VolumeRegistration>(*registration, "VolumeRegistration");
        else
            wrongType(v.id + ": invalid registration");
    }
    v.componentOrder = optionalStrings(f, "component_order");
    v.valueFrame = optionalTagged<ValueFrame>(f, "value_frame", "ValueFrame");
    v.normalize = boolean(f, "normalize");
    v.encoding = optionalTagged<Encoding>(f, "encoding", "Encoding");
    if (const auto* companions = field(f, "companions")) {
        if (!companions->is_array()) wrongType(v.id + ": companions must be a sequence");
        for (const auto& companion : *companions) v.companions.push_back(decodeTagged<Companion>(companion, "Companion"));
    }
    if (const auto* provenance = field(f, "provenance")) v.provenance = decodeMapping(*provenance);
    v.artifactId = text(f, "artifact_id");
    v.artifactKind = text(f, "artifact_kind");
    v.sampleId = text(f, "sample_id");
    v.ownerId = text(f, "owner_id");
    v.modelId = text(f, "model_id");
    v.labelSet = text(f, "label_set");
    v.supersedes = stringList(f, "supersedes");
    if (const auto* coverage = field(f, "coverage")) {
        if (!coverage->is_array()) wrongType(v.id + ": coverage must be a sequence");
        for (const auto& item : *coverage) v.coverage.push_back(decodeTagged<OpenDataVolumeCoverage>(item, "VolumeCoverage"));
    }
    v.derived = boolean(f, "derived").value_or(false);
    if (const auto* shape = field(f, "shape")) {
        if (!shape->is_array()) bad(v.id + ": shape must be a nonempty sequence");
        v.shape.emplace();
        for (const auto& size : *shape) v.shape->push_back(integerValue(size, "shape"));
    }
    v.dtype = text(f, "dtype");
    if (const auto* fill = field(f, "fill_value")) {
        if (!fill->is_number() || (fill->is_number_float() && !std::isfinite(fill->get<double>()))) bad("nonfinite declaration");
        v.fillValue = *fill;
    }
    v.query = optionalTagged<QueryContext>(f, "query", "QueryContext");
    return finished(std::move(v));
}

json Source::toDict() const { return {{"schema_version", 1}, {"source", encode(*this)}}; }

Source Source::fromDict(const json& value)
{
    if (!value.is_object() || value.size() != 2 || !value.contains("schema_version") || !value.contains("source") ||
        !value["schema_version"].is_number_integer() || value["schema_version"].get<std::int64_t>() != 1)
        bad("unsupported source schema version or envelope");
    const auto& root = value["source"];
    if (!root.is_object() || root.value("type", "") != "Source") bad("serialized root must be a Source");
    return decodeTagged<Source>(root, "Source");
}

json encodeMapping(const json& plain)
{
    if (plain.is_object()) {
        json items = json::array();
        for (const auto& item : plain.items()) items.push_back(json::array({item.key(), encodeMapping(item.value())}));
        return {{"type", "mapping"}, {"items", std::move(items)}};
    }
    if (plain.is_array()) {
        json out = json::array();
        for (const auto& item : plain) out.push_back(encodeMapping(item));
        return out;
    }
    return plain;
}

json plainFields(const json& tagged)
{
    if (tagged.is_array()) {
        json out = json::array();
        for (const auto& item : tagged) out.push_back(plainFields(item));
        return out;
    }
    if (!tagged.is_object()) return tagged;
    if (tagged.value("type", "") == "mapping") return decodeMapping(tagged);
    json out = json::object();
    for (const auto& item : tagged.at("fields").items()) out[item.key()] = plainFields(item.value());
    return out;
}

json decodeMapping(const json& value)
{
    if (value.is_object()) {
        if (value.value("type", "") != "mapping" || value.size() != 2 || !value.contains("items") || !value["items"].is_array())
            bad("unknown descriptor tag: '" + value.value("type", "") + "'");
        json out = json::object();
        for (const auto& pair : value["items"]) {
            if (!pair.is_array() || pair.size() != 2 || !pair[0].is_string()) bad("invalid mapping entries");
            if (out.contains(pair[0].get<std::string>())) bad("duplicate mapping key");
            out[pair[0].get<std::string>()] = decodeMapping(pair[1]);
        }
        return out;
    }
    if (value.is_array()) {
        json out = json::array();
        for (const auto& item : value) out.push_back(decodeMapping(item));
        return out;
    }
    if (value.is_number_float() && !std::isfinite(value.get<double>())) bad("nonfinite declaration");
    return value;
}

// ---- snapshot ----
namespace {
// repr() of a float: the shortest digits, positional for exponents in [-4, 16).
void pythonFloat(double value, std::string& out)
{
    if (!std::isfinite(value)) bad("catalog snapshot holds a nonfinite number");
    char buffer[64];
    const auto end = std::to_chars(buffer, buffer + sizeof buffer, value, std::chars_format::scientific).ptr;
    std::string_view text(buffer, end - buffer);
    const auto e = text.find('e');
    const int exponent = std::stoi(std::string(text.substr(e + 1)));
    std::string digits;
    bool negative = false;
    for (const char c : text.substr(0, e)) {
        if (c == '-') negative = true;
        else if (c != '.') digits += c;
    }
    if (negative) out += '-';
    if (exponent >= -4 && exponent < 16) {
        if (exponent < 0) {
            out += "0.";
            out.append(std::size_t(-exponent - 1), '0');
            out += digits;
        } else if (digits.size() <= std::size_t(exponent) + 1) {
            out += digits;
            out.append(std::size_t(exponent) + 1 - digits.size(), '0');
            out += ".0";
        } else {
            out.append(digits, 0, std::size_t(exponent) + 1);
            out += '.';
            out.append(digits, std::size_t(exponent) + 1);
        }
        return;
    }
    out += digits[0];
    if (digits.size() > 1) {
        out += '.';
        out.append(digits, 1);
    }
    out += exponent < 0 ? "e-" : "e+";
    const int magnitude = std::abs(exponent);
    if (magnitude < 10) out += '0';
    out += std::to_string(magnitude);
}

void pythonJson(const json& value, std::string& out, Sha256& hash)
{
    if (out.size() > (1u << 16)) {
        hash.update(out);
        out.clear();
    }
    switch (value.type()) {
    case json::value_t::object: {
        out += '{';
        bool first = true;
        for (const auto& item : value.items()) {  // std::map: sorted by UTF-8 bytes, which is code point order
            if (!first) out += ',';
            first = false;
            out += json(item.key()).dump(-1, ' ', true);
            out += ':';
            pythonJson(item.value(), out, hash);
        }
        out += '}';
        break;
    }
    case json::value_t::array: {
        out += '[';
        bool first = true;
        for (const auto& item : value) {
            if (!first) out += ',';
            first = false;
            pythonJson(item, out, hash);
        }
        out += ']';
        break;
    }
    case json::value_t::number_float:
        pythonFloat(value.get<double>(), out);
        break;
    default:
        out += value.dump(-1, ' ', true);
    }
}
} // namespace

std::string snapshotSha256(const json& value)
{
    Sha256 hash;
    std::string buffer;
    pythonJson(value, buffer, hash);
    hash.update(buffer);
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    for (const auto byte : hash.finish()) {
        out += hex[byte >> 4];
        out += hex[byte & 15];
    }
    return out;
}
} // namespace vcr::sources
