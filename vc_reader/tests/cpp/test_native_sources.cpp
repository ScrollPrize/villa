// Source descriptors, the relationship index and the views over it, from C++.
#include "vc_reader/browsing.hpp"
#include "vc_reader/sources.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace vcr::catalog;
using namespace vcr::sources;
using nlohmann::json;

namespace {
void require(bool v, const std::string& message)
{
    if (!v) throw std::runtime_error(message);
}
template <class F> void rejects(F&& make, const std::string& message)
{
    try {
        (void)make();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("accepted: " + message);
}

json artifact(const std::string& kind, const std::string& id, const std::string& target = "", json parameters = json::object())
{
    if (!target.empty()) parameters["target_volume"] = target;
    return {{"type", kind}, {"id", id}, {"parameters", parameters}};
}

Catalog fixture()
{
    auto newer = artifact("ink-labels", "new", "a", {{"label_set", "2021"}});
    newer["properties"] = {{"supersedes", "2020"}, {"arrays", {"inklabels", "supervision"}}};
    json doc = {{"samples", {{"s", {
        {"sample", {{"id", "s"}, {"properties", {{"type", "scroll"}, {"volume_transforms", {{{"from_volume_id", "a"}, {"transforms", {{{"to_volume_id", "b"},
            {"matrix", {{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}}}}}}}}}}}}},
        {"volumes", {
            {"a", {{"id", "a"}, {"properties", {{"pixel_size_um", 4}}}, {"data", {artifact("ome-zarr", "ct"), artifact("surface-prediction-zarr", "rep", "", {{"level", 1}})}}}},
            {"b", {{"id", "b"}, {"properties", {{"pixel_size_um", 8}}}, {"data", {artifact("ome-zarr", "ct")}}}},
            {"old", {{"id", "old"}, {"properties", {{"pixel_size_um", 8}, {"superseded_by", {{"volume_id", "b"}}}}}, {"data", {artifact("ome-zarr", "ct")}}}}}},
        {"segments", {{"seg", {{"id", "seg"}, {"original_volume_id", "b"}, {"data", {
            artifact("ink-labels", "old", "a", {{"label_set", "2020"}}), newer,
            artifact("layers-zarr", "render", "a"), artifact("layers-tif", "tif", "a"), artifact("alpha-render", "alpha", "a")}}}}}}}}}}};
    return Catalog(parseOpenDataManifest(doc.dump()));
}

void testDescriptors()
{
    Selection selection;
    selection.level = 2;
    Source source;
    source.id = "crop";
    source.storage.location = "/tmp/crop.zarr";
    source.storage.format = "zarr";
    source.storage.selection = selection;
    source.axes = std::vector<std::string>{"z", "y", "x"};
    VolumeRegistration registration;
    registration.sampleId = "s";
    registration.volumeId = "a";
    registration.affine = VolumeTransform{Affine::scale(4), "derived"};
    source.registration = registration;
    source.provenance = json{{"origin", {{"notes", {"original"}}, {"type", "mapping"}}}};
    Encoding encoding;
    encoding.kind = "color";
    encoding.dtype = "uint64";
    encoding.pairs.push_back({{{255, false}, {0, false}, {0, false}}, {std::uint64_t(1) << 32, false}});
    source.encoding = encoding;
    source = finished(std::move(source));
    require(source.id == "user:crop", "namespace");
    require(std::get<VolumeRegistration>(*source.registration).declaredGrid == selection, "declared grid defaults to the selection");
    const auto restored = Source::fromDict(json::parse(source.toDict().dump()));
    require(restored == source, "schema 1 round trip");
    require((*restored.provenance)["origin"]["type"] == "mapping", "a mapping named like a tag stays data");
    Selection other;
    other.level = 0;
    require(std::get<VolumeRegistration>(*source.withSelection(other).registration).declaredGrid == selection, "the original grid stays fixed");
    require(selectLevel(source.withSelection({}), 0) == source.withSelection({}), "a single-resolution source has level 0");
    rejects([&] { return selectLevel(source, 1); }, "level of a single resolution");

    rejects([] { Selection s; s.level = 1; s.arrayPath = "0"; validate(s); return 0; }, "two selectors");
    rejects([] { Source s; s.id = "x"; s.axes = std::vector<std::string>{"x", "x"}; return finished(s); }, "repeated axes");
    rejects([] { return Affine::scale(0); }, "zero scale");
    rejects([] { validate(Plane{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 0, 1}}); return 0; }, "dependent bases");
    rejects([] { validate(Plane{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 0, 1}}); return 0; }, "skew thickness");
    rejects([] { Encoding e; e.kind = "integer"; e.dtype = "uint8"; e.pairs.push_back({{{1, false}}, {256, false}}); validate(e); return 0; }, "output out of range");
    rejects([] { Encoding e; e.kind = "integer"; e.dtype = "uint8"; e.pairs.push_back({{{1, false}}, {1, false}}); e.pairs.push_back({{{1, false}}, {2, false}}); validate(e); return 0; }, "duplicate key");
    rejects([] { return Source::fromDict({{"schema_version", 2}, {"source", json::object()}}); }, "schema version");
    rejects([] { return decodeFields<Source>({{"id", "x"}, {"fill_value", std::nan("")}}); }, "nonfinite");
    Source label;
    label.id = "mask";
    label.role = "label";
    require(label.id == "mask" && !finished(label).unresolvedFields().empty(), "unresolved fields");
    const auto missing = finished(label).unresolvedFields();
    require(std::find(missing.begin(), missing.end(), "storage.location") != missing.end(), "location reported");
}

void testSnapshot()
{
    // Pinned from Python: json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).
    const json value = json::parse(R"({"é":"ü☃\n","values":[1.0,1e15,1e16,1.5e-05,0.0001,123456.789,-2,9223372036854775808,true,null],"a":0.1})");
    require(snapshotSha256(value) == "c016e7ecb8f2f326071704f6b9752633facb81f3ba73275db68b4ccd8a3170f0", "digest");
}

void testIndex()
{
    const auto catalog = fixture();
    const auto scroll = catalog.scroll("s");
    require(catalog.scrolls().size() == 1, "scrolls");
    const auto onA = scroll.onVolume("a");
    const auto labels = onA.labels();
    require(labels.visible().size() == 1, "older label set hidden");
    require(labels.filter([](const Asset&) { return true; }, true).visible().size() == 2, "history on request");
    const auto label = labels.get([](const Asset& a) { return a.source().labelSet == "2021"; });
    require(label.isSuperseded() == false && label.volume()->id() == "a" && label.segment()->id() == "seg", "ownership");
    require(labels.at(scroll.artifacts().get([](const Asset& a) { return a.source().artifactId == "old"; }, true).id()).isSuperseded(), "id lookup finds history");
    require(label.source().companions.size() == 1 && label.masks().at(label.id() + "/supervision").kind() == "mask", "companions");
    require(label.images().visible().size() == 3, "pairs on the segment canvas");
    require(label.surfaceVolumes().visible().size() == 2, "layer stacks only");
    require(isZarrAsset(label) && isInkLabel(label), "filters");
    try {
        (void)scroll.artifacts().get([](const Asset&) { return true; });
        throw std::runtime_error("ambiguous accepted");
    } catch (const AmbiguousMatchError& e) {
        require(e.candidates.size() > 1, "candidates");
    }
    try {
        (void)scroll.volumes().at("absent");
        throw std::runtime_error("absent accepted");
    } catch (const NotFoundError&) {
    }
    require(scroll.volumes().visible().size() == 2 && scroll.volumes().all().size() == 3 && scroll.volumes().at("old").isSuperseded(), "superseded volumes");

    const auto a = scroll.volumes().at("a");
    require(a.source().artifactKind == "ome-zarr" && a.source().axes == std::vector<std::string>{"z", "y", "x"}, "volume image");
    require(a.source().storage.pyramid && a.pixelSizeUm() == 4.0, "pyramid and spacing");
    const auto image = a.source();
    const auto& fallback = std::get<VolumeRegistration>(*image.registration);
    require(fallback.origin == "catalog-fallback" && fallback.affine->forward == Affine::blockReduced(1), "placed on its base");

    const auto mappings = label.mappableVolumes();
    require(mappings.all().size() >= 2, "mappable volumes");
    const auto toB = mappings.at("b");
    require(toB.transform.forward == Affine::scale(2) && toB.requiredOperations == std::vector<std::string>{"surface_projection", "volume_transform"}, "mapping");
    const auto summary = label.summary();
    require(summary["source_volume"] == "a" && summary["segment"] == "seg", "summary");

    const auto matches = atResolution(volumesOf(catalog.scrolls()), 8);
    require(matches.visible().size() == 2, "b level 0 and a level 1");
    require(matches.visible()[0].volume.id() == "a" && matches.visible()[0].level == 1, "ordered");
    rejects([&] { return atResolution(scroll.volumes(), -1); }, "negative resolution");
}

void testOverlays()
{
    const auto catalog = fixture();
    const auto scroll = catalog.scroll("s");
    Source render;
    render.id = "user:render";
    render.storage.location = "/tmp/render.zarr";
    render.storage.format = "zarr";
    render = finished(render);
    Source label;
    label.id = "user:label";
    label.storage.location = "/tmp/label.zarr";
    label = finished(label);
    SourceAttachment attachRender;
    attachRender.kind = "image";
    attachRender.volume = "a";
    attachRender.segment = "user:seg";
    auto view = scroll.withSource(render, attachRender);
    SourceAttachment attachLabel = attachRender;
    attachLabel.kind = "label";
    attachLabel.image = "user:render";
    attachLabel.sameGrid = true;
    view = view.withSource(label, attachLabel);
    require(view.labels().at("user:label").images().at("user:render").kind() == "image", "explicit pair");
    require(view.images().at("user:render").labels().at("user:label").segment()->id() == "user:seg", "reverse pair");
    require(!scroll.index()->find("user:label"), "published metadata unchanged");
    const auto restored = ScrollSource::fromDict(json::parse(view.toDict().dump()));
    require(restored.labels().at("user:label").images().at("user:render").kind() == "image", "view round trip");
    rejects([&] { return view.withSource(label, attachLabel); }, "duplicate");
    SourceAttachment both = attachLabel;
    both.pixelAffine = PixelAffine{{{1, 0, 0}, {0, 1, 0}}};
    Source other = label;
    other.id = "user:other";
    rejects([&] { return view.withSource(finished(other), both); }, "same grid and an affine");
    Source cycleA;
    cycleA.id = "user:first";
    cycleA.supersedes = {"user:second"};
    SourceAttachment plain;
    plain.kind = "label";
    const auto first = scroll.withSource(finished(cycleA), plain);
    Source cycleB;
    cycleB.id = "user:second";
    cycleB.supersedes = {"user:first"};
    rejects([&] { return first.withSource(finished(cycleB), plain); }, "cycle");
    Source volume;
    volume.id = "user:volume";
    volume.storage.location = "/tmp/volume.zarr";
    SourceAttachment attachVolume;
    attachVolume.kind = "volume";
    const auto withVolume = scroll.withSource(finished(volume), attachVolume);
    require(withVolume.volumes().at("user:volume").source().role == "image", "user volume image");
    require(!withVolume.volumes().at("user:volume").record(), "no published record");
}
} // namespace

int main()
{
    try {
        testDescriptors();
        testSnapshot();
        testIndex();
        testOverlays();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
