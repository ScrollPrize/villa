#pragma once
// Immutable descriptions of storage, values and registration. Building or decoding
// one never opens storage; fields nobody declared stay empty, and unresolvedFields()
// reports them without any I/O.
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>
#include "vc_reader/affine.hpp"
#include "vc_reader/catalog.hpp"

namespace vcr::sources {

using catalog::Affine;
using catalog::OpenDataVolumeCoverage;
using catalog::VolumeTransform;

// A declaration of the wrong kind of thing, as opposed to a bad value.
struct DeclarationTypeError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

// A value behind a shared pointer: copies are cheap, == compares values.
template <class T>
class Shared {
public:
    Shared() = default;
    Shared(T value) : value_(std::make_shared<const T>(std::move(value))) {}
    const T& operator*() const { return *value_; }
    const T* operator->() const { return value_.get(); }
    bool operator==(const Shared& other) const
    {
        return value_ == other.value_ || (value_ && other.value_ && *value_ == *other.value_);
    }
private:
    std::shared_ptr<const T> value_;
};

using Vec3 = std::array<double, 3>;
// An explicit 2x3 pixel transform.
using PixelAffine = std::array<std::array<double, 3>, 2>;
// A channel, series or time: by name or by index.
using Selector = std::variant<std::string, std::int64_t>;

// Any value of a 64-bit integer dtype.
struct WideInt {
    std::uint64_t magnitude = 0;
    bool negative = false;
    bool operator==(const WideInt&) const = default;
    auto operator<=>(const WideInt& other) const
    {
        if (negative != other.negative) return negative ? std::strong_ordering::less : std::strong_ordering::greater;
        return negative ? other.magnitude <=> magnitude : magnitude <=> other.magnitude;
    }
};

struct Frame {
    std::string sampleId, volumeId;
    std::int64_t level = 0;
    bool operator==(const Frame&) const = default;
};

// A requested reference resolution; a negative sublevel means unavailable.
struct LevelRequest {
    Frame reference;
    std::optional<std::int64_t> sublevel;
    bool operator==(const LevelRequest&) const = default;
};

struct Selection {
    std::optional<std::string> arrayPath;
    std::optional<std::int64_t> level;
    std::optional<Selector> series;
    std::optional<std::vector<Selector>> channels;
    std::optional<Selector> time;
    std::optional<LevelRequest> corresponding;
    bool operator==(const Selection&) const = default;
    // The same grid whatever channels and time it shows.
    [[nodiscard]] Selection spatial() const;
};

struct Storage {
    std::optional<std::string> location, format;
    std::vector<std::string> locations, alternatives;
    Selection selection;
    bool pyramid = false;
    std::optional<std::int64_t> gridLevel;
    bool operator==(const Storage&) const = default;
};

// XY pixel centers and a one-voxel normal footprint in volume level-0 XYZ.
struct Plane {
    Vec3 origin{}, basisX{}, basisY{}, thickness{};
    bool operator==(const Plane&) const = default;
};

struct VolumeRegistration {
    std::string sampleId, volumeId;
    std::optional<VolumeTransform> affine;
    std::optional<Plane> plane;
    std::string mode = "array";
    std::optional<std::string> storageFrame;
    std::optional<Selection> declaredGrid;
    std::string origin = "explicit";
    std::optional<std::string> storageUnits;
    std::optional<Vec3> volumeSpacingUm;
    bool operator==(const VolumeRegistration&) const = default;
};

struct ValueFrame {
    std::string kind = "volume";
    std::optional<std::string> sampleId, volumeId;
    std::optional<VolumeTransform> affine;
    std::optional<std::int64_t> level;
    std::optional<std::string> name, units;
    bool operator==(const ValueFrame&) const = default;
};

struct EncodingPair {
    std::vector<WideInt> key;  // one integer, or one per color component
    WideInt value;
    bool operator==(const EncodingPair&) const = default;
};

struct Encoding {
    std::string kind;  // integer or color
    std::vector<EncodingPair> pairs;
    std::string dtype;
    std::optional<WideInt> fallback;
    bool operator==(const Encoding&) const = default;
};

struct Source;

struct Surface {
    std::string id;
    Storage geometry;
    std::string sampleId, volumeId;
    std::optional<std::vector<std::string>> axes, componentOrder;
    std::optional<VolumeTransform> xyzToVolume;
    std::optional<PixelAffine> pixelToUv;
    std::optional<Vec3> spacingUm;
    std::optional<Shared<Source>> mask, normals;
    bool tifxyzMask = true;
    bool operator==(const Surface&) const = default;
};

struct SurfaceRegistration {
    Surface surface;
    std::optional<PixelAffine> pixelToUv;
    std::string origin = "explicit";
    std::optional<Selection> declaredGrid;
    bool operator==(const SurfaceRegistration&) const = default;
};

using SourceRegistration = std::variant<VolumeRegistration, SurfaceRegistration>;

struct Companion {
    std::string name;
    Shared<Source> source;
    bool sameGrid = false;
    bool operator==(const Companion&) const = default;
};

struct QueryContext {
    Frame reference;
    std::string relation;
    VolumeTransform transform;
    std::optional<OpenDataVolumeCoverage> coverage;
    bool operator==(const QueryContext&) const = default;
};

struct Source {
    std::string id;
    Storage storage;
    std::optional<std::string> role;
    std::optional<std::vector<std::string>> axes;
    std::optional<SourceRegistration> registration;
    std::optional<std::vector<std::string>> componentOrder;
    std::optional<ValueFrame> valueFrame;
    std::optional<bool> normalize;
    std::optional<Encoding> encoding;
    std::vector<Companion> companions;
    Shared<nlohmann::json> provenance = nlohmann::json::object();
    std::optional<std::string> artifactId, artifactKind, sampleId, ownerId, modelId, labelSet;
    std::vector<std::string> supersedes;
    std::vector<OpenDataVolumeCoverage> coverage;
    bool derived = false;
    std::optional<std::vector<std::int64_t>> shape;
    std::optional<std::string> dtype;
    nlohmann::json fillValue;  // null when undeclared
    std::optional<QueryContext> query;
    bool operator==(const Source&) const = default;

    // What a reader still needs and nobody declared; no I/O.
    [[nodiscard]] std::vector<std::string> unresolvedFields() const;
    // The source with another storage selection; the declaration's original grid stays fixed.
    [[nodiscard]] Source withSelection(Selection selection) const;
    // Schema version 1: {"schema_version": 1, "source": ...}.
    [[nodiscard]] nlohmann::json toDict() const;
    [[nodiscard]] static Source fromDict(const nlohmann::json& value);
};

// The source at a pyramid level; a single-resolution source has only level 0.
[[nodiscard]] Source selectLevel(const Source& source, std::int64_t level);

// Applies a source's defaults (its "user:" namespace and declared grid) and checks the
// whole declaration; throws std::invalid_argument. Decoding does this already.
[[nodiscard]] Source finished(Source source);
void validate(const Frame&);
void validate(const LevelRequest&);
void validate(const Selection&);
void validate(const Storage&);
void validate(const Plane&);
void validate(const VolumeRegistration&);
void validate(const ValueFrame&);
void validate(const Encoding&);
void validate(const Surface&);
void validate(const SurfaceRegistration&);
void validate(const Companion&);
void validate(const QueryContext&);
void validatePixelAffine(const PixelAffine&);

// The tagged schema-1 form: {"type": <name>, "fields": {...}}. A mapping (provenance) is
// {"type": "mapping", "items": [[key, value], ...]}.
template <class T> [[nodiscard]] nlohmann::json encode(const T& value);
// The fields of a tagged value, by name; absent and null mean undeclared. Validated.
template <class T> [[nodiscard]] T decodeFields(const nlohmann::json& fields);
#define VC_READER_DESCRIPTOR(T) \
    template <> nlohmann::json encode(const T&); \
    template <> T decodeFields(const nlohmann::json&);
VC_READER_DESCRIPTOR(VolumeTransform)
VC_READER_DESCRIPTOR(OpenDataVolumeCoverage)
VC_READER_DESCRIPTOR(Frame)
VC_READER_DESCRIPTOR(LevelRequest)
VC_READER_DESCRIPTOR(Selection)
VC_READER_DESCRIPTOR(Storage)
VC_READER_DESCRIPTOR(Plane)
VC_READER_DESCRIPTOR(VolumeRegistration)
VC_READER_DESCRIPTOR(ValueFrame)
VC_READER_DESCRIPTOR(Encoding)
VC_READER_DESCRIPTOR(Surface)
VC_READER_DESCRIPTOR(SurfaceRegistration)
VC_READER_DESCRIPTOR(Companion)
VC_READER_DESCRIPTOR(QueryContext)
VC_READER_DESCRIPTOR(Source)
#undef VC_READER_DESCRIPTOR
// A tagged mapping back to plain JSON; rejects what a declaration cannot hold.
[[nodiscard]] nlohmann::json decodeMapping(const nlohmann::json& value);
[[nodiscard]] nlohmann::json encodeMapping(const nlohmann::json& plain);
// A tagged value as plain nested fields, without its tags.
[[nodiscard]] nlohmann::json plainFields(const nlohmann::json& tagged);

// SHA-256 hex digest of JSON encoded as Python's json module does with sorted keys,
// compact separators and ensure_ascii, which persisted source ids depend on.
[[nodiscard]] std::string snapshotSha256(const nlohmann::json& value);

} // namespace vcr::sources
