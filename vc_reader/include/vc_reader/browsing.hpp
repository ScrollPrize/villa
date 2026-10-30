#pragma once
// Scroll-centered views over one metadata-only index: what a sample has, how its
// assets relate, and which sources describe them. Nothing here touches storage.
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include "vc_reader/catalog.hpp"
#include "vc_reader/registrations.hpp"
#include "vc_reader/sources.hpp"

namespace vcr::catalog {

inline constexpr double kDefaultResolutionRelTol = 0.005;
inline constexpr int kPyramidLevels = 6;

// A query matched nothing.
struct NotFoundError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
// A query matched more than one item.
struct AmbiguousMatchError : std::runtime_error {
    AmbiguousMatchError(const std::string& message, std::vector<std::string> matches)
        : std::runtime_error(message), candidates(std::move(matches)) {}
    std::vector<std::string> candidates;
};

// How one label's pixels lie on one image: label base XY -> image base XY.
// No pixelAffine means inspect metadata.
struct Pairing {
    std::string labelId, imageId;
    std::optional<sources::PixelAffine> pixelAffine;
    std::string evidence;
};

// One catalog's sources with their ownership and relationships, plus any user overlays.
class RelationshipIndex {
public:
    struct Entry {
        sources::Source source;
        std::string kind;  // volume, image, label, representation, geometry or artifact
        std::string sample;
        std::optional<std::string> volume, segment;
    };
    // The published sources.
    explicit RelationshipIndex(std::shared_ptr<const OpenDataManifest> manifest);
    // The published sources of `published` with overlays added: a JSON array of
    // {source, kind, sample, volume, segment, pairs: [{label, image, affine, same_grid}]}.
    RelationshipIndex(std::shared_ptr<const RelationshipIndex> published, nlohmann::json overlays);

    [[nodiscard]] const OpenDataManifest& manifest() const { return *manifest_; }
    [[nodiscard]] const std::shared_ptr<const OpenDataManifest>& sharedManifest() const { return manifest_; }
    [[nodiscard]] const nlohmann::json& overlays() const { return overlays_; }
    // The index without overlays this one extends; null if it is that index.
    [[nodiscard]] const std::shared_ptr<const RelationshipIndex>& published() const { return published_; }
    [[nodiscard]] const Entry* find(const std::string& id) const;
    [[nodiscard]] bool hidden(const std::string& id) const { return hidden_.count(id) != 0; }
    [[nodiscard]] const std::vector<std::string>& bySample(const std::string& sample) const;
    [[nodiscard]] const std::vector<std::string>& byVolume(const std::string& sample, const std::string& volume) const;
    [[nodiscard]] const std::vector<std::string>& bySegment(const std::string& sample, const std::string& segment) const;
    [[nodiscard]] const std::vector<Pairing>& pairsOfLabel(const std::string& label) const;
    [[nodiscard]] const std::vector<Pairing>& pairsOfImage(const std::string& image) const;

private:
    void add(sources::Source source, std::string kind, std::string sample, std::optional<std::string> volume,
             std::optional<std::string> segment);
    void addOverlays();
    void finish();
    void linkPairs();
    void pair(Pairing relationship);
    void sortTables();

    std::shared_ptr<const OpenDataManifest> manifest_;
    std::shared_ptr<const RelationshipIndex> published_;
    nlohmann::json overlays_ = nlohmann::json::array();
    std::map<std::string, Entry> entries_;
    std::map<std::string, std::vector<std::string>> bySample_;
    std::map<std::pair<std::string, std::string>, std::vector<std::string>> byVolume_, bySegment_;
    std::map<std::string, std::vector<Pairing>> pairs_, reversePairs_;
    std::set<std::string> hidden_;
};
using IndexPtr = std::shared_ptr<const RelationshipIndex>;

// Deterministic views with explicit filtering and unambiguous selection. Iteration hides
// only explicitly superseded entities; id lookup also finds historical ones.
template <class T>
class Collection {
public:
    using Predicate = std::function<bool(const T&)>;
    Collection() = default;
    // Sorted by id then owning sample, unless `ordered`.
    explicit Collection(std::vector<T> values, bool includeSuperseded = false, bool ordered = false);

    [[nodiscard]] const std::vector<T>& all() const& { return all_; }
    [[nodiscard]] std::vector<T> all() && { return std::move(all_); }
    [[nodiscard]] bool includeSuperseded() const { return includeSuperseded_; }
    // What iteration shows.
    [[nodiscard]] std::vector<T> visible() const;
    [[nodiscard]] std::size_t size() const { return visible().size(); }
    [[nodiscard]] bool empty() const { return visible().empty(); }
    [[nodiscard]] const T& at(std::string_view id) const;
    // The matches, showing superseded ones if asked (default: as this collection does).
    [[nodiscard]] Collection filter(const Predicate& predicate, std::optional<bool> includeSuperseded = std::nullopt) const;
    // The one item a predicate matches; NotFoundError or AmbiguousMatchError otherwise.
    [[nodiscard]] T get(const Predicate& predicate, std::optional<bool> includeSuperseded = std::nullopt,
                        const std::string& description = "criteria") const;

private:
    std::vector<T> all_;
    bool includeSuperseded_ = false;
};

class ScrollSource;
class Volume;
class Segment;

struct VolumeMapping;

// One source of a scroll, or a mask of one.
class Asset {
public:
    Asset(IndexPtr index, std::string id) : index_(std::move(index)), id_(std::move(id)) {}
    Asset(IndexPtr index, sources::Source mask, std::string parent);
    [[nodiscard]] const std::string& id() const { return id_; }
    [[nodiscard]] const IndexPtr& index() const { return index_; }
    [[nodiscard]] const sources::Source& source() const;
    // volume, image, label, representation, geometry, artifact or mask.
    [[nodiscard]] std::string kind() const;
    [[nodiscard]] bool isSuperseded() const;
    [[nodiscard]] std::optional<Volume> volume() const;
    [[nodiscard]] std::optional<Segment> segment() const;
    [[nodiscard]] ScrollSource scroll() const;
    [[nodiscard]] Collection<Asset> images() const;
    [[nodiscard]] Collection<Asset> labels() const;
    // Surface layer stacks paired with this label; volume is the source CT.
    [[nodiscard]] Collection<Asset> surfaceVolumes() const;
    [[nodiscard]] Collection<Asset> masks() const;
    // "sampled_layers" for images: their depth follows surface layers, not CT Z.
    [[nodiscard]] std::optional<std::string> depthSemantics() const;
    // Volume affines reachable from this asset's source volume. Surface labels keep
    // their source volume even when pixel-to-UV fields are unresolved.
    [[nodiscard]] Collection<VolumeMapping> mappableVolumes() const;
    // Source ownership, volume mappings and unresolved fields, as JSON. Coverage
    // describes published segment bounds, not a crop guarantee.
    [[nodiscard]] nlohmann::json summary() const;
    // The relationship between this label and an image of the same view; std::invalid_argument
    // if it is absent or ambiguous. Without an image, the label's only one.
    [[nodiscard]] Pairing pairingWith(const std::optional<Asset>& image) const;
private:
    IndexPtr index_;
    std::string id_;
    std::optional<sources::Source> mask_;
    std::string parent_;
    [[nodiscard]] const RelationshipIndex::Entry& owner() const;
};

// Assets in a context: a sample, a volume or a segment.
class AssetContext {
public:
    virtual ~AssetContext() = default;
    [[nodiscard]] Collection<Asset> artifacts() const { return assets(); }
    [[nodiscard]] Collection<Asset> labels() const { return assets("label"); }
    [[nodiscard]] Collection<Asset> images() const { return assets("image"); }
    // Rendered layer stacks; depth follows sampled surface layers, not the CT Z axis.
    [[nodiscard]] Collection<Asset> surfaceVolumes() const;
    [[nodiscard]] Collection<Asset> representations() const { return assets("representation"); }
    [[nodiscard]] Collection<Asset> geometry() const { return assets("geometry"); }
protected:
    [[nodiscard]] virtual std::vector<std::string> assetIds() const = 0;
    [[nodiscard]] virtual const IndexPtr& assetIndex() const = 0;
    [[nodiscard]] Collection<Asset> assets(std::string_view kind = {}) const;
};

class Volume : public AssetContext {
public:
    Volume(IndexPtr index, std::string sample, std::string id)
        : index_(std::move(index)), sampleId_(std::move(sample)), id_(std::move(id)) {}
    [[nodiscard]] const std::string& id() const { return id_; }
    [[nodiscard]] const std::string& sampleId() const { return sampleId_; }
    [[nodiscard]] const IndexPtr& index() const { return index_; }
    [[nodiscard]] ScrollSource scroll() const;
    // Empty for a volume only an overlay declares.
    [[nodiscard]] const OpenDataVolume* record() const;
    [[nodiscard]] bool isSuperseded() const;
    // Published, else the one uniform spacing a user volume's single image declares.
    [[nodiscard]] std::optional<double> pixelSizeUm() const;
    // Its volume assets, not rendered images.
    [[nodiscard]] Collection<Asset> volumeImages() const { return assets("volume"); }
    [[nodiscard]] sources::Source source() const;
    [[nodiscard]] Collection<Segment> segments() const;
protected:
    [[nodiscard]] std::vector<std::string> assetIds() const override;
    [[nodiscard]] const IndexPtr& assetIndex() const override { return index_; }
private:
    IndexPtr index_;
    std::string sampleId_, id_;
};

class Segment : public AssetContext {
public:
    Segment(IndexPtr index, std::string sample, std::string id, std::optional<std::string> volume = std::nullopt)
        : index_(std::move(index)), sampleId_(std::move(sample)), id_(std::move(id)), volume_(std::move(volume)) {}
    [[nodiscard]] const std::string& id() const { return id_; }
    [[nodiscard]] const std::string& sampleId() const { return sampleId_; }
    [[nodiscard]] const std::optional<std::string>& volumeContext() const { return volume_; }
    [[nodiscard]] const OpenDataSegment* record() const;
    [[nodiscard]] ScrollSource scroll() const;
    [[nodiscard]] Collection<Volume> volumes() const;
    [[nodiscard]] Segment onVolume(const std::string& volume) const;
protected:
    [[nodiscard]] std::vector<std::string> assetIds() const override;
    [[nodiscard]] const IndexPtr& assetIndex() const override { return index_; }
private:
    IndexPtr index_;
    std::string sampleId_, id_;
    std::optional<std::string> volume_;
};

// How a source joins a scroll view (ScrollSource::withSource).
struct SourceAttachment {
    std::string kind;
    std::optional<std::string> volume, segment;
    // Pair a label with `image`, or an image with `label`, declaring exactly one of
    // sameGrid or a label-XY-to-image-XY pixelAffine.
    std::optional<std::string> image, label;
    bool sameGrid = false;
    std::optional<sources::PixelAffine> pixelAffine;
};

// A sample, with an optional volume context.
class ScrollSource : public AssetContext {
public:
    ScrollSource(IndexPtr index, std::string id, std::optional<std::string> volume = std::nullopt)
        : index_(std::move(index)), id_(std::move(id)), volume_(std::move(volume)) {}
    [[nodiscard]] const std::string& id() const { return id_; }
    [[nodiscard]] const std::optional<std::string>& volumeContext() const { return volume_; }
    [[nodiscard]] const IndexPtr& index() const { return index_; }
    [[nodiscard]] const OpenDataSample& record() const;
    [[nodiscard]] std::optional<Volume> volume() const;
    [[nodiscard]] Collection<Volume> volumes() const;
    [[nodiscard]] Collection<Segment> segments() const;
    [[nodiscard]] ScrollSource onVolume(const std::string& volume) const;
    // Level-0 XYZ of one volume to level-0 XYZ of another; none when they are not
    // registered to each other. A volume and the one superseding it share a frame.
    [[nodiscard]] std::optional<VolumeTransform> transform(const std::string& source, const std::string& target) const;
    // A new view with a descriptor attached; published metadata is not changed.
    [[nodiscard]] ScrollSource withSource(sources::Source source, const SourceAttachment& attachment) const;
    // Schema 1: the whole catalog, the overlays and the context.
    [[nodiscard]] nlohmann::json toDict() const;
    [[nodiscard]] static ScrollSource fromDict(const nlohmann::json& value);
protected:
    [[nodiscard]] std::vector<std::string> assetIds() const override;
    [[nodiscard]] const IndexPtr& assetIndex() const override { return index_; }
private:
    IndexPtr index_;
    std::string id_;
    std::optional<std::string> volume_;
};

struct VolumeMapping {
    Volume volume;
    VolumeTransform transform;
    std::optional<OpenDataVolumeCoverage> coverage;
    std::vector<std::string> requiredOperations;  // surface_projection, volume_transform
    [[nodiscard]] const std::string& id() const { return volume.id(); }
};

struct VolumeLevel {
    Volume volume;
    std::int64_t level = 0;
    VolumeLevel(Volume v, std::int64_t l);
    [[nodiscard]] sources::Source source() const;
};

// What a reader found at a candidate's level.
struct OpenedGrid {
    std::optional<VolumeTransform> storageFromArray;
    std::optional<std::string> storageUnits;
    std::optional<sources::SourceRegistration> registration;
    sources::Selection selection;
    std::array<std::int64_t, 3> shapeZYX{};
};

// A volume's image at a level, whose resolution the catalog estimates.
struct ResolutionCandidate {
    Volume volume;
    sources::Source source;
    std::int64_t level = 0;
    double estimatedResolutionUm = 0;
    bool verified = false;
    std::optional<sources::Selection> selection;
    std::optional<sources::Vec3> spacingUm;
    std::optional<std::array<std::int64_t, 3>> shape;
    [[nodiscard]] std::string id() const;
    // This candidate checked against storage; std::invalid_argument on a contradiction.
    [[nodiscard]] ResolutionCandidate verifiedAgainst(const OpenedGrid& grid) const;
};

// The images of these volumes at a voxel spacing, full-resolution or at a pyramid
// level, closest first. Levels are estimated until a candidate is verified.
[[nodiscard]] Collection<ResolutionCandidate> atResolution(const Collection<Volume>& volumes, double um,
                                                           double relTol = kDefaultResolutionRelTol);
// The volumes of these scrolls.
[[nodiscard]] Collection<Volume> volumesOf(const Collection<ScrollSource>& scrolls);
// The assets a Zarr or ink filter would pick, by format and kind.
[[nodiscard]] bool isZarrAsset(const Asset& asset);
[[nodiscard]] bool isInkLabel(const Asset& asset);

// A parsed catalog and its index, built once; copies share both.
class Catalog {
public:
    Catalog() = default;
    explicit Catalog(OpenDataManifest manifest, std::string source = {});
    // The catalog a published index was built from.
    explicit Catalog(IndexPtr published);
    [[nodiscard]] static Catalog load(const std::string& source = std::string(kDefaultManifestUrl),
                                      const CatalogLoadOptions& options = {}, std::string* warning = nullptr);
    [[nodiscard]] const OpenDataManifest& manifest() const;
    [[nodiscard]] const std::string& source() const;
    [[nodiscard]] const IndexPtr& index() const;
    [[nodiscard]] Collection<ScrollSource> samples() const;
    [[nodiscard]] Collection<ScrollSource> scrolls() const;
    // A sample by id or any name for it; NotFoundError or AmbiguousMatchError otherwise.
    [[nodiscard]] ScrollSource scroll(const std::string& name, const std::optional<std::string>& volume = std::nullopt) const;
private:
    struct State;
    std::shared_ptr<State> state_;
};

extern template class Collection<Asset>;
extern template class Collection<Volume>;
extern template class Collection<Segment>;
extern template class Collection<ScrollSource>;
extern template class Collection<VolumeMapping>;
extern template class Collection<ResolutionCandidate>;

} // namespace vcr::catalog
