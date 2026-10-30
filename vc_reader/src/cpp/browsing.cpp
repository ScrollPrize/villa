#include "vc_reader/browsing.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <tuple>

#include "sigv4.hpp"

namespace vcr::catalog {
using namespace sources;
using nlohmann::json;
namespace {

[[noreturn]] void bad(const std::string& message) { throw std::invalid_argument(message); }

std::string trimmed(std::string value)
{
    while (!value.empty() && value.back() == '/') value.pop_back();
    return value;
}
std::string quote(const std::string& value) { return aws_uri_encode(value, false); }
bool startsWithUserOrCatalog(const std::string& value) { return value.starts_with("catalog:") || value.starts_with("user:"); }

const json* jsonObject(const json& value, const char* key)
{
    if (!value.is_object()) return nullptr;
    const auto it = value.find(key);
    return it == value.end() ? nullptr : &*it;
}
bool truthy(const json* value)
{
    if (!value || value->is_null()) return false;
    if (value->is_boolean()) return value->get<bool>();
    if (value->is_number()) return value->get<double>() != 0;
    if (value->is_string()) return !value->get_ref<const std::string&>().empty();
    return !value->empty();
}
const json empty = json::object();
const json& propertiesOfProvenance(const Source& source)
{
    const auto* artifact = jsonObject(*source.provenance, "artifact");
    const auto* properties = artifact ? jsonObject(*artifact, "properties") : nullptr;
    return properties && properties->is_object() ? *properties : empty;
}

Storage storageOf(const OpenDataArtifact& artifact)
{
    Storage storage;
    const auto& location = artifact.resolvedUrl.empty() ? artifact.sourcePath : artifact.resolvedUrl;
    if (!location.empty()) storage.location = location;
    storage.alternatives = artifact.urls();
    if (const auto grid = artifact.gridLevel()) storage.gridLevel = *grid;
    return storage;
}

Storage labelStorage(const OpenDataArtifact& artifact, const std::optional<std::string>& root, const std::string& name)
{
    Storage storage;
    if (root) storage.location = trimmed(*root) + "/" + name + ".zarr/";
    for (const auto& url : artifact.urls()) storage.alternatives.push_back(trimmed(url) + "/" + name + ".zarr/");
    storage.format = "zarr";
    storage.pyramid = true;
    storage.selection.level = 0;
    return storage;
}

std::string artifactIdOf(const OpenDataArtifact& artifact, const std::string& owner, std::size_t position, const std::string& snapshot)
{
    return "catalog:" + owner + "/" + (artifact.id ? "id/" + quote(*artifact.id) : "snapshot/" + snapshot + "/data/" + std::to_string(position));
}

Surface surfaceDescriptor(const OpenDataSample& sample, const OpenDataArtifact* mesh, const std::string& meshId,
                          const std::string& target, const std::string& identifier)
{
    Surface surface;
    surface.id = mesh ? meshId : identifier + "/surface";
    if (mesh) {
        surface.geometry = storageOf(*mesh);
        surface.geometry.format = "tifxyz";
        surface.xyzToVolume = VolumeTransform{};
    }
    surface.sampleId = sample.id;
    surface.volumeId = target;
    surface.axes = std::vector<std::string>{"y", "x"};
    surface.componentOrder = std::vector<std::string>{"x", "y", "z"};
    const auto* volume = sample.findVolume(target);
    if (volume && volume->pixelSizeUm) surface.spacingUm = Vec3{*volume->pixelSizeUm, *volume->pixelSizeUm, *volume->pixelSizeUm};
    validate(surface);
    return surface;
}

struct Owner {
    std::string kind, id;
    const std::vector<OpenDataArtifact>* artifacts;
    const OpenDataVolume* volume = nullptr;
    const OpenDataSegment* segment = nullptr;
};

Source adapt(const OpenDataSample& sample, const Owner& owner, const OpenDataArtifact& artifact, const std::string& identifier,
             const std::string& ownerKey, const std::string& snapshot)
{
    Source source;
    source.id = identifier;
    source.storage = storageOf(artifact);
    source.derived = owner.kind == "volume" && artifact.type != "ome-zarr";
    const auto volumeRegistration = [&](const std::string& volume, const std::string& origin) {
        VolumeRegistration registration;
        registration.sampleId = sample.id;
        registration.volumeId = volume;
        registration.origin = origin;
        return registration;
    };
    if (owner.kind == "volume") {
        source.registration = volumeRegistration(owner.id, "catalog");
        if (artifact.type == "ome-zarr" || artifact.type == "surface-prediction-zarr" || artifact.type == "ink-detection-3d-zarr") {
            source.storage.format = "zarr";
            source.storage.pyramid = true;
            source.role = artifact.type == "ome-zarr" ? "image" : "scalar";
            source.axes = std::vector<std::string>{"z", "y", "x"};
            auto registration = volumeRegistration(owner.id, "catalog-fallback");
            if (const auto grid = artifact.gridLevel())
                registration.affine = VolumeTransform{Affine::blockReduced(std::ldexp(1.0, *grid)), {}};
            registration.declaredGrid = Selection{};
            registration.declaredGrid->level = 0;
            if (owner.volume->pixelSizeUm) {
                const double spacing = *owner.volume->pixelSizeUm;
                registration.volumeSpacingUm = Vec3{spacing, spacing, spacing};
            }
            source.registration = std::move(registration);
        } else if (artifact.type == "normal-grids") {
            source.role = "normal";  // Bundle layout and component frame stay unknown.
        }
    } else if (owner.kind == "segment" && artifact.targetVolumeId) {
        const auto& target = *artifact.targetVolumeId;
        if (artifact.type == "ink-labels") {
            const OpenDataArtifact* mesh = nullptr;
            std::size_t meshes = 0, meshPosition = 0;
            for (std::size_t i = 0; i < owner.artifacts->size(); ++i) {
                const auto& candidate = (*owner.artifacts)[i];
                if (candidate.type == "tifxyz-transformed" && candidate.targetVolumeId == target) {
                    ++meshes;
                    mesh = &candidate;
                    meshPosition = i;
                }
            }
            if (meshes != 1) mesh = nullptr;
            SurfaceRegistration registration;
            registration.surface = surfaceDescriptor(sample, mesh, mesh ? artifactIdOf(*mesh, ownerKey, meshPosition, snapshot) : "", target, identifier);
            registration.origin = "catalog";
            source.registration = std::move(registration);
            const auto root = source.storage.location;
            source.storage = labelStorage(artifact, root, "inklabels");
            source.role = "label";
            source.axes = std::vector<std::string>{"y", "x"};
            for (const char* name : {"supervision", "validation", "mapping_validity"}) {
                if (std::find(artifact.arrays.begin(), artifact.arrays.end(), name) == artifact.arrays.end()) continue;
                Source companion;
                companion.id = identifier + "/" + name;
                companion.storage = labelStorage(artifact, root, name);
                companion.role = "label";
                companion.axes = source.axes;
                source.companions.push_back({name, Shared<Source>(finished(std::move(companion))), true});
            }
        } else if (artifact.type == "tifxyz-transformed") {
            source.storage.format = "tifxyz";
            SurfaceRegistration registration;
            registration.surface = surfaceDescriptor(sample, &artifact, identifier, target, identifier);
            registration.origin = "catalog";
            source.registration = std::move(registration);
        } else if (artifact.type == "layers-zarr") {
            source.storage.format = "zarr";
            source.storage.pyramid = true;
            source.storage.selection.level = 0;
            source.role = "image";
            source.axes = std::vector<std::string>{"z", "y", "x"};
            // Stored z enumerates sampled surface layers, not volume voxels.
        } else {
            // Target identity is useful even when array/geometry semantics are not declared.
            source.registration = volumeRegistration(target, "catalog");
        }
    }
    if (artifact.type != "ink-labels") {
        for (const auto& declared : artifact.supersedes)
            source.supersedes.push_back(startsWithUserOrCatalog(declared) ? declared : "catalog:" + ownerKey + "/id/" + quote(declared));
    }
    source.artifactId = artifact.id;
    source.artifactKind = artifact.type;
    source.sampleId = sample.id;
    source.ownerId = owner.id;
    source.modelId = artifact.modelId;
    if (!artifact.labelSet.empty()) source.labelSet = artifact.labelSet;
    if (owner.segment) source.coverage = owner.segment->volumeCoverage;
    source.provenance = json{{"owner_kind", owner.kind}, {"artifact", artifact.raw}};
    return finished(std::move(source));
}

std::string assetKind(const Source& source, const std::string& ownerKind, const std::string& type)
{
    if (ownerKind == "volume" && type == "ome-zarr") return "volume";
    if (type == "ink-labels") return "label";
    if (type == "layers-zarr" || type == "layers-tif" || type == "alpha-render" || type == "alpha-render-downsampled") return "image";
    if (type.starts_with("tifxyz") || type.starts_with("obj") || type == "umbilicus" || type == "fiberlets") return "geometry";
    if ((source.derived && source.role) || type == "lasagna" || type == "ink-detection" || type == "ink-detection-downsampled" || type == "3d-ink-prediction")
        return "representation";
    return "artifact";
}

PixelAffine pixelAffineValue(const json& rows)
{
    if (!rows.is_array() || rows.size() != 2 || !rows[0].is_array() || !rows[1].is_array() || rows[0].size() != 3 || rows[1].size() != 3)
        bad("pixel-to-UV transform must have shape 2x3");
    PixelAffine out;
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c) {
            if (!rows[r][c].is_number()) bad("pixel-to-UV transform must have shape 2x3");
            out[r][c] = rows[r][c].get<double>();
        }
    validatePixelAffine(out);
    return out;
}

std::optional<std::string> optionalText(const json& value, const char* key)
{
    const auto* v = jsonObject(value, key);
    if (!v || v->is_null()) return std::nullopt;
    if (!v->is_string()) bad(std::string(key) + " must be a string");
    return v->get<std::string>();
}

bool isclose(double a, double b, double relTol) { return std::abs(a - b) <= relTol * std::max(std::abs(a), std::abs(b)); }

const std::string& idOf(const ScrollSource& v) { return v.id(); }
const std::string& idOf(const Volume& v) { return v.id(); }
const std::string& idOf(const Segment& v) { return v.id(); }
const std::string& idOf(const Asset& v) { return v.id(); }
const std::string& idOf(const VolumeMapping& v) { return v.id(); }
std::string idOf(const ResolutionCandidate& v) { return v.id(); }
std::string sampleOf(const ScrollSource&) { return {}; }
std::string sampleOf(const Volume& v) { return v.sampleId(); }
std::string sampleOf(const Segment& v) { return v.sampleId(); }
std::string sampleOf(const Asset& v) { return v.source().sampleId.value_or(""); }
std::string sampleOf(const VolumeMapping&) { return {}; }
std::string sampleOf(const ResolutionCandidate&) { return {}; }
bool supersededOf(const ScrollSource&) { return false; }
bool supersededOf(const Volume& v) { return v.isSuperseded(); }
bool supersededOf(const Segment&) { return false; }
bool supersededOf(const Asset& v) { return v.isSuperseded(); }
bool supersededOf(const VolumeMapping& v) { return v.volume.isSuperseded(); }
bool supersededOf(const ResolutionCandidate&) { return false; }

Collection<Asset> surfaceVolumesAmong(const Collection<Asset>& images)
{
    std::vector<Asset> out;
    for (const auto& asset : images.all())
        if (asset.source().artifactKind == "layers-zarr" || asset.source().artifactKind == "layers-tif") out.push_back(asset);
    return Collection<Asset>(std::move(out), images.includeSuperseded(), true);
}
} // namespace

// ---- Collection ----
template <class T>
Collection<T>::Collection(std::vector<T> values, bool includeSuperseded, bool ordered)
    : all_(std::move(values)), includeSuperseded_(includeSuperseded)
{
    if (!ordered)
        std::stable_sort(all_.begin(), all_.end(), [](const T& a, const T& b) {
            const auto& x = idOf(a);
            const auto& y = idOf(b);
            return x != y ? x < y : sampleOf(a) < sampleOf(b);
        });
}
template <class T>
std::vector<T> Collection<T>::visible() const
{
    if (includeSuperseded_) return all_;
    std::vector<T> out;
    for (const auto& value : all_)
        if (!supersededOf(value)) out.push_back(value);
    return out;
}
namespace {
template <class T>
[[noreturn]] void noOne(const std::vector<std::string>& matches, const std::vector<T>& all, const std::string& description)
{
    auto list = [](const std::vector<std::string>& ids) {
        std::string text = "[";
        for (std::size_t i = 0; i < ids.size(); ++i) text += (i ? ", '" : "'") + ids[i] + "'";
        return text + "]";
    };
    std::vector<std::string> candidates = matches;
    if (matches.empty())
        for (const auto& value : all) candidates.push_back(idOf(value));
    const auto message = description + ": " + std::to_string(matches.size()) + " matches; candidates: " + list(candidates);
    if (matches.empty()) throw NotFoundError(message);
    throw AmbiguousMatchError(message, matches);
}
} // namespace
template <class T>
const T& Collection<T>::at(std::string_view id) const
{
    const T* found = nullptr;
    std::vector<std::string> matches;
    for (const auto& value : all_)
        if (idOf(value) == id) {
            found = &value;
            matches.emplace_back(idOf(value));
        }
    if (matches.size() != 1) noOne(matches, all_, "ID '" + std::string(id) + "'");
    return *found;
}
template <class T>
Collection<T> Collection<T>::filter(const Predicate& predicate, std::optional<bool> includeSuperseded) const
{
    std::vector<T> out;
    for (const auto& value : all_)
        if (predicate(value)) out.push_back(value);
    return Collection(std::move(out), includeSuperseded.value_or(includeSuperseded_), true);
}
template <class T>
T Collection<T>::get(const Predicate& predicate, std::optional<bool> includeSuperseded, const std::string& description) const
{
    const auto matches = filter(predicate, includeSuperseded).visible();
    if (matches.size() == 1) return matches[0];
    std::vector<std::string> ids;
    for (const auto& value : matches) ids.emplace_back(idOf(value));
    noOne(ids, all_, description);
}
template class Collection<Asset>;
template class Collection<Volume>;
template class Collection<Segment>;
template class Collection<ScrollSource>;
template class Collection<VolumeMapping>;
template class Collection<ResolutionCandidate>;

// ---- index ----
RelationshipIndex::RelationshipIndex(std::shared_ptr<const OpenDataManifest> manifest) : manifest_(std::move(manifest))
{
    std::vector<const OpenDataSample*> samples;
    for (const auto& sample : manifest_->samples) samples.push_back(&sample);
    std::sort(samples.begin(), samples.end(), [](const auto* a, const auto* b) { return a->id < b->id; });
    // Only artifacts without an id of their own need the snapshot in theirs.
    std::string snapshot;
    const auto lacksId = [](const std::vector<OpenDataArtifact>& artifacts) {
        return std::any_of(artifacts.begin(), artifacts.end(), [](const auto& a) { return !a.id; });
    };
    const bool needsSnapshot = std::any_of(samples.begin(), samples.end(), [&](const auto* sample) {
        return lacksId(sample->artifacts) || std::any_of(sample->volumes.begin(), sample->volumes.end(), [&](const auto& v) { return lacksId(v.artifacts); }) ||
               std::any_of(sample->segments.begin(), sample->segments.end(), [&](const auto& s) { return lacksId(s.artifacts); });
    });
    if (needsSnapshot) snapshot = snapshotSha256(manifest_->raw);
    for (const auto* sample : samples) {
        std::vector<Owner> owners{{"sample", sample->id, &sample->artifacts}};
        std::vector<const OpenDataVolume*> volumes;
        for (const auto& volume : sample->volumes) volumes.push_back(&volume);
        std::sort(volumes.begin(), volumes.end(), [](const auto* a, const auto* b) { return a->id < b->id; });
        for (const auto* volume : volumes) owners.push_back({"volume", volume->id, &volume->artifacts, volume});
        std::vector<const OpenDataSegment*> segments;
        for (const auto& segment : sample->segments) segments.push_back(&segment);
        std::sort(segments.begin(), segments.end(), [](const auto* a, const auto* b) { return a->id < b->id; });
        for (const auto* segment : segments) owners.push_back({"segment", segment->id, &segment->artifacts, nullptr, segment});
        for (const auto& owner : owners) {
            const auto ownerKey = quote(sample->id) + "/" + owner.kind + "/" + quote(owner.id);
            for (std::size_t position = 0; position < owner.artifacts->size(); ++position) {
                const auto& artifact = (*owner.artifacts)[position];
                auto source = adapt(*sample, owner, artifact, artifactIdOf(artifact, ownerKey, position, snapshot), ownerKey, snapshot);
                auto kind = assetKind(source, owner.kind, artifact.type);
                std::optional<std::string> volume = owner.kind == "volume" ? std::optional(owner.id) : artifact.targetVolumeId;
                std::optional<std::string> segment = owner.kind == "segment" ? std::optional(owner.id) : std::nullopt;
                add(std::move(source), std::move(kind), sample->id, std::move(volume), std::move(segment));
            }
        }
    }
    finish();
}

RelationshipIndex::RelationshipIndex(std::shared_ptr<const RelationshipIndex> published, json overlays)
    : manifest_(published->manifest_), published_(std::move(published)), overlays_(std::move(overlays))
{
    if (!overlays_.is_array()) bad("overlays must be a sequence");
    entries_ = published_->entries_;
    bySample_ = published_->bySample_;
    byVolume_ = published_->byVolume_;
    bySegment_ = published_->bySegment_;
    addOverlays();
    finish();
}

void RelationshipIndex::add(Source source, std::string kind, std::string sample, std::optional<std::string> volume, std::optional<std::string> segment)
{
    const auto id = source.id;
    if (entries_.count(id)) bad("duplicate source ID " + id);
    if (volume) byVolume_[{sample, *volume}].push_back(id);
    if (segment) bySegment_[{sample, *segment}].push_back(id);
    bySample_[sample].push_back(id);
    entries_.emplace(id, Entry{std::move(source), std::move(kind), std::move(sample), std::move(volume), std::move(segment)});
}

void RelationshipIndex::addOverlays()
{
    for (const auto& entry : overlays_) {
        const auto* encoded = jsonObject(entry, "source");
        if (!encoded) bad("overlay entries require a source");
        auto source = Source::fromDict(*encoded);
        const auto sample = optionalText(entry, "sample").value_or("");
        const auto kind = optionalText(entry, "kind").value_or("");
        const auto volume = optionalText(entry, "volume");
        const auto segment = optionalText(entry, "segment");
        if (!source.id.starts_with("user:") || !manifest_->findSample(sample) || source.sampleId != sample)
            bad("overlay sources require user: IDs and a consistent owning sample");
        if (kind != "volume" && kind != "image" && kind != "label" && kind != "representation" && kind != "geometry" && kind != "artifact")
            bad("unknown overlay asset kind");
        if (kind == "volume" && (!volume || !volume->starts_with("user:"))) bad("new volume IDs must use the user: namespace");
        if (source.registration) {
            const bool consistent = std::visit([&](const auto& registration) {
                using T = std::decay_t<decltype(registration)>;
                if constexpr (std::is_same_v<T, VolumeRegistration>)
                    return registration.sampleId == sample && std::optional(registration.volumeId) == volume;
                else
                    return registration.surface.sampleId == sample && std::optional(registration.surface.volumeId) == volume;
            }, *source.registration);
            if (!consistent) bad("overlay ownership contradicts source registration");
        }
        add(std::move(source), kind, sample, volume, segment);
    }
}

void RelationshipIndex::finish()
{
    // Explicit supersession is resolved globally before filtering any view.
    for (auto& [id, entry] : entries_) {
        const auto& source = entry.source;
        hidden_.insert(source.supersedes.begin(), source.supersedes.end());
        const auto& properties = propertiesOfProvenance(source);
        if (truthy(jsonObject(properties, "superseded_by"))) hidden_.insert(id);
        if (source.artifactKind != "ink-labels" || !entry.segment) continue;
        const auto* older = jsonObject(properties, "supersedes");
        if (!truthy(older)) continue;
        const auto candidates = bySegment_.find({entry.sample, *entry.segment});
        if (candidates == bySegment_.end()) continue;
        std::vector<std::string> added;
        for (const auto& candidateId : candidates->second) {
            const auto& candidate = entries_.at(candidateId);
            if (older->is_string() && candidate.source.labelSet == older->get<std::string>() && candidate.volume == entry.volume) {
                hidden_.insert(candidateId);
                if (std::find(source.supersedes.begin(), source.supersedes.end(), candidateId) == source.supersedes.end())
                    added.push_back(candidateId);
            }
        }
        if (!added.empty()) {
            entry.source.supersedes.insert(entry.source.supersedes.end(), added.begin(), added.end());
        }
    }
    // No source may transitively supersede itself.
    std::set<std::string> visited, active;
    for (const auto& [root, unused] : entries_) {
        std::vector<std::pair<std::string, bool>> stack{{root, false}};
        while (!stack.empty()) {
            auto [id, done] = std::move(stack.back());
            stack.pop_back();
            if (done) {
                active.erase(id);
                visited.insert(id);
            } else if (active.count(id)) {
                bad("supersession cycle at " + id);
            } else if (!visited.count(id) && entries_.count(id)) {
                active.insert(id);
                stack.emplace_back(id, true);
                for (const auto& older : entries_.at(id).source.supersedes) stack.emplace_back(older, false);
            }
        }
    }
    linkPairs();
    sortTables();
}

void RelationshipIndex::linkPairs()
{
    const PixelAffine identity{{{1, 0, 0}, {0, 1, 0}}};
    for (const auto& [id, entry] : entries_) {
        if (entry.kind != "label" || !id.starts_with("catalog:") || !entry.segment) continue;
        const auto* artifact = jsonObject(*entry.source.provenance, "artifact");
        const auto* creation = artifact ? jsonObject(*artifact, "creation_info") : nullptr;
        const auto* target = creation ? jsonObject(*creation, "target") : nullptr;
        const auto* render = target ? jsonObject(*target, "render") : nullptr;
        std::optional<std::string> required;
        if (truthy(render) && render->is_string()) required = trimmed(render->get<std::string>());
        const auto others = bySegment_.find({entry.sample, *entry.segment});
        if (others == bySegment_.end()) continue;
        for (const auto& other : others->second) {
            const auto& image = entries_.at(other);
            if (image.kind != "image" || image.volume != entry.volume) continue;
            // Published labels are on the target segment canvas. An explicit image
            // reference narrows this convention, never shape matching.
            if (required) {
                bool named = image.source.storage.location && trimmed(*image.source.storage.location) == *required;
                for (const auto& alternative : image.source.storage.alternatives) named = named || trimmed(alternative) == *required;
                if (!named) continue;
            }
            pair({id, other, identity, required ? "published image reference" : "segment/target-volume canvas convention"});
        }
    }
    for (const auto& entry : overlays_) {
        const auto* pairs = jsonObject(entry, "pairs");
        if (!pairs || !pairs->is_array()) continue;
        for (const auto& relationship : *pairs) {
            const auto label = optionalText(relationship, "label").value_or("");
            const auto image = optionalText(relationship, "image").value_or("");
            const auto* labelEntry = find(label);
            const auto* imageEntry = find(image);
            if (!labelEntry || !imageEntry || labelEntry->kind != "label" || imageEntry->kind != "image")
                bad("pair relationships require known label and image IDs");
            if (labelEntry->sample != imageEntry->sample) bad("paired assets must belong to the same sample");
            const auto* affine = jsonObject(relationship, "affine");
            const auto* sameGrid = jsonObject(relationship, "same_grid");
            pair({label, image, pixelAffineValue(affine ? *affine : json(nullptr)),
                  truthy(sameGrid) ? "explicit same grid" : "explicit user pixel mapping"});
        }
    }
}

void RelationshipIndex::pair(Pairing relationship)
{
    auto& existing = pairs_[relationship.labelId];
    if (std::any_of(existing.begin(), existing.end(), [&](const auto& p) { return p.imageId == relationship.imageId; }))
        bad("duplicate label/image relationship");
    existing.push_back(relationship);
    reversePairs_[relationship.imageId].push_back(std::move(relationship));
}

void RelationshipIndex::sortTables()
{
    for (auto& [key, ids] : bySample_) std::sort(ids.begin(), ids.end());
    for (auto& [key, ids] : byVolume_) std::sort(ids.begin(), ids.end());
    for (auto& [key, ids] : bySegment_) std::sort(ids.begin(), ids.end());
}

const RelationshipIndex::Entry* RelationshipIndex::find(const std::string& id) const
{
    const auto it = entries_.find(id);
    return it == entries_.end() ? nullptr : &it->second;
}
namespace {
const std::vector<std::string> noIds;
const std::vector<Pairing> noPairs;
template <class Map, class Key>
const typename Map::mapped_type& lookup(const Map& map, const Key& key, const typename Map::mapped_type& fallback)
{
    const auto it = map.find(key);
    return it == map.end() ? fallback : it->second;
}
} // namespace
const std::vector<std::string>& RelationshipIndex::bySample(const std::string& sample) const { return lookup(bySample_, sample, noIds); }
const std::vector<std::string>& RelationshipIndex::byVolume(const std::string& sample, const std::string& volume) const
{
    return lookup(byVolume_, std::pair(sample, volume), noIds);
}
const std::vector<std::string>& RelationshipIndex::bySegment(const std::string& sample, const std::string& segment) const
{
    return lookup(bySegment_, std::pair(sample, segment), noIds);
}
const std::vector<Pairing>& RelationshipIndex::pairsOfLabel(const std::string& label) const { return lookup(pairs_, label, noPairs); }
const std::vector<Pairing>& RelationshipIndex::pairsOfImage(const std::string& image) const { return lookup(reversePairs_, image, noPairs); }

// ---- contexts ----
Collection<Asset> AssetContext::assets(std::string_view kind) const
{
    const auto& index = assetIndex();
    std::vector<Asset> out;
    for (auto& id : assetIds())
        if (kind.empty() || index->find(id)->kind == kind) out.emplace_back(index, std::move(id));
    return Collection<Asset>(std::move(out));
}
Collection<Asset> AssetContext::surfaceVolumes() const { return surfaceVolumesAmong(assets("image")); }

// ---- Asset ----
Asset::Asset(IndexPtr index, Source mask, std::string parent)
    : index_(std::move(index)), id_(mask.id), mask_(std::move(mask)), parent_(std::move(parent)) {}
const RelationshipIndex::Entry& Asset::owner() const { return *index_->find(parent_.empty() ? id_ : parent_); }
const Source& Asset::source() const { return mask_ ? *mask_ : index_->find(id_)->source; }
std::string Asset::kind() const
{
    const auto* entry = index_->find(id_);
    return entry ? entry->kind : "mask";
}
bool Asset::isSuperseded() const { return index_->hidden(id_); }
std::optional<Volume> Asset::volume() const
{
    const auto& entry = owner();
    if (!entry.volume) return std::nullopt;
    return Volume(index_, entry.sample, *entry.volume);
}
std::optional<Segment> Asset::segment() const
{
    const auto& entry = owner();
    if (!entry.segment) return std::nullopt;
    return Segment(index_, entry.sample, *entry.segment, entry.volume);
}
ScrollSource Asset::scroll() const { return ScrollSource(index_, owner().sample); }
Collection<Asset> Asset::images() const
{
    std::vector<Asset> out;
    for (const auto& pairing : index_->pairsOfLabel(id_)) out.emplace_back(index_, pairing.imageId);
    return Collection<Asset>(std::move(out));
}
Collection<Asset> Asset::labels() const
{
    std::vector<Asset> out;
    for (const auto& pairing : index_->pairsOfImage(id_)) out.emplace_back(index_, pairing.labelId);
    return Collection<Asset>(std::move(out));
}
Collection<Asset> Asset::surfaceVolumes() const { return surfaceVolumesAmong(images()); }
Collection<Asset> Asset::masks() const
{
    std::vector<Asset> out;
    for (const auto& companion : source().companions) out.emplace_back(index_, *companion.source, id_);
    return Collection<Asset>(std::move(out));
}
std::optional<std::string> Asset::depthSemantics() const
{
    if (kind() == "image") return "sampled_layers";
    return std::nullopt;
}
Collection<VolumeMapping> Asset::mappableVolumes() const
{
    const auto sourceVolume = volume();
    if (!sourceVolume || !source().registration) return {};
    const auto view = scroll();
    std::vector<VolumeMapping> out;
    for (const auto& candidate : view.volumes().all()) {
        auto transform = view.transform(sourceVolume->id(), candidate.id());
        if (!transform) continue;
        std::optional<OpenDataVolumeCoverage> coverage;
        for (const auto& item : source().coverage)
            if (item.volumeId == candidate.id()) {
                coverage = item;
                break;
            }
        std::vector<std::string> operations;
        if (std::holds_alternative<SurfaceRegistration>(*source().registration)) operations.emplace_back("surface_projection");
        if (candidate.id() != sourceVolume->id()) operations.emplace_back("volume_transform");
        out.push_back({candidate, std::move(*transform), std::move(coverage), std::move(operations)});
    }
    return Collection<VolumeMapping>(std::move(out));
}
json Asset::summary() const
{
    const auto sourceVolume = volume();
    const auto owningSegment = segment();
    json mappable = json::array();
    for (const auto& mapping : mappableVolumes().visible()) {
        mappable.push_back({{"volume", mapping.id()}, {"required_operations", mapping.requiredOperations},
                            {"affine", mapping.transform.forward.json()},
                            {"derivation_path", mapping.transform.derivationPath.empty() ? json(nullptr) : json(mapping.transform.derivationPath)},
                            {"coverage", mapping.coverage ? encode(*mapping.coverage).at("fields") : json(nullptr)}});
    }
    return {{"id", id_}, {"kind", kind()}, {"sample", scroll().id()},
            {"segment", owningSegment ? json(owningSegment->id()) : json(nullptr)},
            {"source_volume", sourceVolume ? json(sourceVolume->id()) : json(nullptr)},
            {"unresolved_fields", source().unresolvedFields()}, {"mappable_volumes", std::move(mappable)}};
}
Pairing Asset::pairingWith(const std::optional<Asset>& image) const
{
    if (kind() != "label") throw DeclarationTypeError("open_paired requires a label");
    const auto chosen = image ? *image : images().get([](const Asset&) { return true; });
    const auto* entry = index_->find(chosen.id());
    if (!entry || entry->source != chosen.source()) bad("image must belong to the same scroll view");
    std::vector<Pairing> matches;
    for (const auto& pairing : index_->pairsOfLabel(id_))
        if (pairing.imageId == chosen.id()) matches.push_back(pairing);
    if (matches.size() != 1) bad("label/image relationship is absent or ambiguous");
    return matches[0];
}

// ---- Volume ----
ScrollSource Volume::scroll() const { return ScrollSource(index_, sampleId_, id_); }
const OpenDataVolume* Volume::record() const { return index_->manifest().findSample(sampleId_)->findVolume(id_); }
bool Volume::isSuperseded() const
{
    const auto* volume = record();
    return volume && volume->supersededBy.has_value();
}
std::optional<double> Volume::pixelSizeUm() const
{
    if (const auto* volume = record()) return volume->pixelSizeUm;
    const auto images = volumeImages().visible();
    if (images.size() != 1 || !images[0].source().registration) return std::nullopt;
    const auto* registration = std::get_if<VolumeRegistration>(&*images[0].source().registration);
    if (!registration || !registration->volumeSpacingUm) return std::nullopt;
    const auto& spacing = *registration->volumeSpacingUm;
    return spacing[0] == spacing[1] && spacing[1] == spacing[2] ? std::optional(spacing[0]) : std::nullopt;
}
Source Volume::source() const { return volumeImages().get([](const Asset&) { return true; }).source(); }
Collection<Segment> Volume::segments() const { return scroll().segments(); }
std::vector<std::string> Volume::assetIds() const { return index_->byVolume(sampleId_, id_); }

// ---- Segment ----
const OpenDataSegment* Segment::record() const
{
    const auto& segments = index_->manifest().findSample(sampleId_)->segments;
    const auto it = std::find_if(segments.begin(), segments.end(), [&](const auto& s) { return s.id == id_; });
    return it == segments.end() ? nullptr : &*it;
}
ScrollSource Segment::scroll() const { return ScrollSource(index_, sampleId_, volume_); }
Collection<Volume> Segment::volumes() const
{
    std::set<std::string> ids;
    for (const auto& id : index_->bySegment(sampleId_, id_))
        if (const auto& volume = index_->find(id)->volume) ids.insert(*volume);
    std::vector<Volume> out;
    for (const auto& id : ids) out.emplace_back(index_, sampleId_, id);
    return Collection<Volume>(std::move(out));
}
Segment Segment::onVolume(const std::string& volume) const
{
    return Segment(index_, sampleId_, id_, volumes().at(volume).id());
}
std::vector<std::string> Segment::assetIds() const
{
    std::vector<std::string> out;
    for (const auto& id : index_->bySegment(sampleId_, id_))
        if (!volume_ || index_->find(id)->volume == volume_) out.push_back(id);
    return out;
}

// ---- ScrollSource ----
const OpenDataSample& ScrollSource::record() const { return *index_->manifest().findSample(id_); }
std::optional<Volume> ScrollSource::volume() const
{
    if (!volume_) return std::nullopt;
    return volumes().at(*volume_);
}
Collection<Volume> ScrollSource::volumes() const
{
    std::set<std::string> ids;
    for (const auto& volume : record().volumes) ids.insert(volume.id);
    for (const auto& entry : index_->overlays())
        if (optionalText(entry, "sample") == id_ && optionalText(entry, "kind") == "volume")
            if (auto volume = optionalText(entry, "volume")) ids.insert(std::move(*volume));
    std::vector<Volume> out;
    for (const auto& id : ids) out.emplace_back(index_, id_, id);
    return Collection<Volume>(std::move(out));
}
Collection<Segment> ScrollSource::segments() const
{
    std::set<std::string> ids;
    for (const auto& segment : record().segments) ids.insert(segment.id);
    for (const auto& entry : index_->overlays())
        if (optionalText(entry, "sample") == id_)
            if (auto segment = optionalText(entry, "segment"); segment && !segment->empty()) ids.insert(std::move(*segment));
    if (volume_) {
        std::set<std::string> onVolume;
        for (const auto& id : index_->byVolume(id_, *volume_))
            if (const auto& segment = index_->find(id)->segment) onVolume.insert(*segment);
        std::erase_if(ids, [&](const auto& id) { return !onVolume.count(id); });
    }
    std::vector<Segment> out;
    for (const auto& id : ids) out.emplace_back(index_, id_, id, volume_);
    return Collection<Segment>(std::move(out));
}
ScrollSource ScrollSource::onVolume(const std::string& volume) const { return ScrollSource(index_, id_, volumes().at(volume).id()); }
std::optional<VolumeTransform> ScrollSource::transform(const std::string& source, const std::string& target) const
{
    return resolveVolumeTransform(record(), source, target);
}
std::vector<std::string> ScrollSource::assetIds() const { return volume_ ? index_->byVolume(id_, *volume_) : index_->bySample(id_); }

ScrollSource ScrollSource::withSource(Source source, const SourceAttachment& a) const
{
    if (!source.id.starts_with("user:")) bad("custom sources require the user: namespace");
    const auto& kind = a.kind;
    if (kind != "volume" && kind != "image" && kind != "label" && kind != "representation" && kind != "geometry" && kind != "artifact")
        bad("unknown asset kind '" + kind + "'");
    if (source.sampleId && *source.sampleId != id_) bad("source sample differs from owning scroll");
    auto volume = a.volume;
    if (kind == "volume") {
        if (!volume) volume = source.id;
        if (record().findVolume(*volume) || !volume->starts_with("user:")) bad("new volume IDs must use the user: namespace");
    } else {
        if (!volume) volume = volume_;
        if (volume) (void)volumes().at(*volume);
    }
    const auto& segment = a.segment;
    if (segment && !segment->starts_with("user:")) {
        const auto& published = record().segments;
        if (std::none_of(published.begin(), published.end(), [&](const auto& s) { return s.id == *segment; }))
            bad("new segment IDs must use the user: namespace");
    }
    if (!source.role && (kind == "volume" || kind == "image" || kind == "label")) source.role = kind == "label" ? "label" : "image";
    if (source.registration) {
        const auto [sample, registered] = std::visit([](const auto& registration) {
            using T = std::decay_t<decltype(registration)>;
            if constexpr (std::is_same_v<T, VolumeRegistration>)
                return std::pair(registration.sampleId, registration.volumeId);
            else
                return std::pair(registration.surface.sampleId, registration.surface.volumeId);
        }, *source.registration);
        if (sample != id_ || (volume && registered != *volume)) bad("source registration contradicts owning sample/volume");
        if (!volume) volume = registered;
    }
    if (kind == "volume" && !source.registration) {
        VolumeRegistration registration;
        registration.sampleId = id_;
        registration.volumeId = *volume;
        registration.affine = VolumeTransform{};
        source.registration = std::move(registration);
    }
    source.sampleId = id_;
    source.ownerId = segment ? *segment : volume ? *volume : id_;
    source = finished(std::move(source));
    json pairs = json::array();
    if (a.image || a.label) {
        if ((a.image && (kind != "label" || a.label)) || (a.label && kind != "image"))
            bad("pair a label with image=, or an image with label=");
        if (a.sameGrid == a.pixelAffine.has_value()) bad("declare exactly one of same_grid=True or pixel_affine");
        const PixelAffine matrix = a.sameGrid ? PixelAffine{{{1, 0, 0}, {0, 1, 0}}} : *a.pixelAffine;
        validatePixelAffine(matrix);
        pairs.push_back({{"label", kind == "label" ? source.id : *a.label}, {"image", kind == "label" ? *a.image : source.id},
                         {"affine", matrix}, {"same_grid", a.sameGrid}});
    } else if (a.sameGrid || a.pixelAffine) {
        bad("pixel mapping requires an explicit paired asset");
    }
    auto overlays = index_->overlays();
    overlays.push_back({{"source", source.toDict()}, {"kind", kind}, {"sample", id_},
                        {"volume", volume ? json(*volume) : json(nullptr)}, {"segment", segment ? json(*segment) : json(nullptr)},
                        {"pairs", std::move(pairs)}});
    auto published = index_->published() ? index_->published() : index_;
    return ScrollSource(std::make_shared<RelationshipIndex>(std::move(published), std::move(overlays)), id_, volume_);
}

json ScrollSource::toDict() const
{
    return {{"schema_version", 1}, {"catalog", index_->manifest().raw}, {"overlays", index_->overlays()}, {"scroll", id_},
            {"volume", volume_ ? json(*volume_) : json(nullptr)}};
}
ScrollSource ScrollSource::fromDict(const json& value)
{
    if (!value.is_object() || value.size() != 5 || !value.contains("catalog") || !value.contains("overlays") || !value.contains("scroll") ||
        !value.contains("volume") || !value.contains("schema_version") || !value["schema_version"].is_number_integer() ||
        value["schema_version"].get<std::int64_t>() != 1)
        bad("unsupported scroll view schema version");
    IndexPtr index = Catalog(parseOpenDataManifest(value["catalog"].dump())).index();
    if (!value["overlays"].empty()) index = std::make_shared<RelationshipIndex>(index, value["overlays"]);
    const auto scroll = optionalText(value, "scroll").value_or("");
    if (!index->manifest().findSample(scroll)) bad("overlay view names an unknown sample");
    ScrollSource view(std::move(index), scroll, optionalText(value, "volume"));
    if (view.volumeContext()) (void)view.volumes().at(*view.volumeContext());
    return view;
}

// ---- levels and resolutions ----
VolumeLevel::VolumeLevel(Volume v, std::int64_t l) : volume(std::move(v)), level(l)
{
    if (level < 0) bad("level must be nonnegative");
}
Source VolumeLevel::source() const { return selectLevel(volume.source(), level); }

std::string ResolutionCandidate::id() const
{
    return volume.sampleId() + "/" + volume.id() + "/" + std::to_string(level) + "/" + source.id;
}

ResolutionCandidate ResolutionCandidate::verifiedAgainst(const OpenedGrid& grid) const
{
    const auto columns = [](const Affine& matrix, double factor) {
        Vec3 spacing;
        for (int i = 0; i < 3; ++i) {
            const int column = 2 - i;
            spacing[i] = std::sqrt(matrix(0, column) * matrix(0, column) + matrix(1, column) * matrix(1, column) + matrix(2, column) * matrix(2, column)) * factor;
        }
        return spacing;
    };
    Vec3 spacing;
    if (grid.storageFromArray && grid.storageUnits == "micrometer") {
        spacing = columns(grid.storageFromArray->forward, 1);
    } else {
        const auto* registration = grid.registration ? std::get_if<VolumeRegistration>(&*grid.registration) : nullptr;
        if (!registration || !registration->affine) bad("storage has no verifiable volume grid");
        spacing = columns(registration->affine->forward, volume.pixelSizeUm().value_or(0));
    }
    for (const double v : spacing) {
        if (!isclose(v, estimatedResolutionUm, 1e-6))
            bad("storage spacing (" + std::to_string(spacing[0]) + ", " + std::to_string(spacing[1]) + ", " + std::to_string(spacing[2]) +
                ") contradicts catalog estimate " + std::to_string(estimatedResolutionUm));
    }
    if (const auto* record = volume.record(); record && record->shapeZYX) {
        for (int i = 0; i < 3; ++i) {
            const auto size = static_cast<std::int64_t>((*record->shapeZYX)[i]);
            const std::int64_t factor = std::int64_t(1) << level;
            if (grid.shapeZYX[i] != (size + factor - 1) / factor)
                bad("storage shape contradicts catalog shape at level " + std::to_string(level));
        }
    }
    auto out = *this;
    out.verified = true;
    out.selection = grid.selection;
    out.spacingUm = spacing;
    out.shape = grid.shapeZYX;
    return out;
}

Collection<ResolutionCandidate> atResolution(const Collection<Volume>& volumes, double um, double relTol)
{
    if (!std::isfinite(um) || um <= 0 || !std::isfinite(relTol) || relTol < 0)
        bad("resolution must be positive and tolerance nonnegative, both finite");
    std::vector<ResolutionCandidate> matches;
    for (const auto& volume : volumes.visible()) {
        const auto pixelSize = volume.pixelSizeUm();
        if (!pixelSize) continue;
        for (const auto& asset : volume.volumeImages().visible()) {
            const int count = asset.source().storage.pyramid ? kPyramidLevels : 1;
            for (int level = 0; level < count; ++level) {
                const double estimate = *pixelSize * std::ldexp(1.0, level);
                if (isclose(estimate, um, relTol)) matches.push_back({volume, asset.source(), level, estimate});
            }
        }
    }
    std::sort(matches.begin(), matches.end(), [um](const auto& a, const auto& b) {
        return std::tuple(std::abs(a.estimatedResolutionUm - um), a.volume.sampleId(), a.volume.id(), a.level, a.source.id) <
               std::tuple(std::abs(b.estimatedResolutionUm - um), b.volume.sampleId(), b.volume.id(), b.level, b.source.id);
    });
    return Collection<ResolutionCandidate>(std::move(matches), false, true);
}

Collection<Volume> volumesOf(const Collection<ScrollSource>& scrolls)
{
    std::vector<Volume> out;
    for (const auto& scroll : scrolls.visible())
        for (const auto& volume : scroll.volumes().all()) out.push_back(volume);
    return Collection<Volume>(std::move(out), scrolls.includeSuperseded());
}

bool isZarrAsset(const Asset& asset) { return asset.source().storage.format == "zarr"; }
bool isInkLabel(const Asset& asset) { return asset.source().artifactKind == "ink-labels"; }

// ---- Catalog ----
struct Catalog::State {
    std::shared_ptr<const OpenDataManifest> manifest;
    std::string source;
    std::once_flag indexed;
    IndexPtr index;
};
Catalog::Catalog(OpenDataManifest manifest, std::string source) : state_(std::make_shared<State>())
{
    state_->manifest = std::make_shared<const OpenDataManifest>(std::move(manifest));
    state_->source = std::move(source);
}
Catalog Catalog::load(const std::string& source, const CatalogLoadOptions& options, std::string* warning)
{
    auto result = loadCatalog(source, options);
    if (warning) *warning = std::move(result.warning);
    return Catalog(std::move(result.manifest), source);
}
Catalog::Catalog(IndexPtr published) : state_(std::make_shared<State>())
{
    state_->manifest = published->sharedManifest();
    state_->index = std::move(published);
}
const OpenDataManifest& Catalog::manifest() const { return *state_->manifest; }
const std::string& Catalog::source() const { return state_->source; }
const IndexPtr& Catalog::index() const
{
    std::call_once(state_->indexed, [&] {
        if (!state_->index) state_->index = std::make_shared<RelationshipIndex>(state_->manifest);
    });
    return state_->index;
}
Collection<ScrollSource> Catalog::samples() const
{
    std::vector<ScrollSource> out;
    for (const auto& sample : manifest().samples) out.emplace_back(index(), sample.id);
    return Collection<ScrollSource>(std::move(out));
}
Collection<ScrollSource> Catalog::scrolls() const
{
    return samples().filter([](const ScrollSource& s) { return s.record().type == "scroll"; });
}
ScrollSource Catalog::scroll(const std::string& name, const std::optional<std::string>& volume) const
{
    auto ids = manifest().samplesNamed(name);
    if (manifest().findSample(name)) ids = {name};
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (ids.empty()) throw NotFoundError("no sample '" + name + "'");
    if (ids.size() != 1) throw AmbiguousMatchError("ambiguous sample '" + name + "'", ids);
    ScrollSource view(index(), ids[0]);
    return volume ? view.onVolume(*volume) : view;
}
} // namespace vcr::catalog
