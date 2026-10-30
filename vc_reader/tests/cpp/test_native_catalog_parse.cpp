// Parsing of published catalog records into typed fields, and the catalog's
// own rules over them: surfaces of a segment, ink images, timestamps, origins.
#include "vc_reader/catalog.hpp"
#include "vc_reader/registrations.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace vcr::catalog;

namespace {
void require(bool v, const std::string& message)
{
    if (!v) throw std::runtime_error(message);
}

const char* kSample = R"({
  "samples": {
    "PHerc0139": {
      "sample": {
        "id": "PHerc0139",
        "description": "Example sample",
        "canonical_volume_id": "vol1",
        "properties": {"type": "scroll", "unknown_sample_field": 7, "volume_transforms": [
          {"from_volume_id": "vol1", "transforms": [{"to_volume_id": "vol2",
            "matrix": [[0.5, 0, 0, 10], [0, 0.5, 0, 20], [0, 0, 0.5, 30]], "derivation_path": "vol1-vol2"}]}]},
        "data": [{"type": "photo", "origins": [{"path": "PHerc0139/photos/photo.jpg",
          "access_roots": [{"type": "s3", "url": "s3://vesuvius-challenge-open-data/", "usage": "public-read"}]}]}]
      },
      "scans": {
        "scan-a": {"id": "scan-a", "long_id": "scan-a-long",
          "creation": {"date": "2024-01-01T00:00:00Z", "process": "xray tomography",
            "metadata": {"location": "ESRF Grenoble", "beamline": "bm18",
              "metadata": {"tomo": {"acquisition": {"scanRadix": "SCROLL_RADIX"}}}}},
          "properties": {"pixel_size_um": 2.403, "energy_keV": 111.0, "detector_distance_mm": 220.0}}
      },
      "volumes": {
        "vol1": {"id": "vol1", "long_id": "vol1-long", "scan_id": "scan-a", "suffix": "masked.zarr",
          "creation": {"date": "2025-05-26T15:17:18Z", "process": "export",
            "metadata": {"window_u16_min": -8171, "window_u16_max": 62026},
            "derived_from": {"type": "scan", "id": "scan-a"}},
          "properties": {"pixel_size_um": 7.91, "energy_keV": 88.0, "detector_distance_mm": 12.5,
            "data_format": "uint8", "shape": [100, 200, 300], "z_direction_is_top_to_bottom": true,
            "left_handed_coordinates": false, "text_side": "recto",
            "superseded_by": {"volume_id": "vol3", "date": "2026-01-01", "reason": "rescan"},
            "license": {"name": "CC BY-NC 4.0", "url": "https://creativecommons.org/licenses/by-nc/4.0/"},
            "render_level": 2, "render_segments": true},
          "data": [
            {"type": "zarr", "origins": [{"path": "PHerc0139/volumes/vol1.zarr",
              "copy_info": {"source_path": "s3://scans/PHerc0139/vol1_masked.zarr/"}, "access_roots": [
              {"type": "s3", "url": "s3://private-root/", "usage": "internal"},
              {"type": "s3", "url": "s3://vesuvius-challenge-open-data/", "usage": "public-read"}]}]},
            {"type": "surface-prediction-zarr", "parameters": {"level": 2}},
            {"type": "normal-grids", "parameters": {"level": 2.5}},
            {"type": "lasagna", "origins": [{"path": "PHerc0139/representations/predictions/fibers/f1/", "access_roots": [
              {"type": "s3", "url": "s3://vesuvius-challenge-open-data/", "usage": "public-read"}]}]},
            {"type": "ink_detection_3d_zarr"},
            {"type": "future-pred-zarr"}
          ]},
        "vol2": {"id": "vol2", "scan_id": "scan-a", "properties": {}}
      },
      "segments": {
        "seg-a": {"id": "seg-a", "long_id": "PHerc0139-seg-a", "suffix": "seg-a", "original_volume_id": "vol1",
          "creation": {"date": "2026-03-11T00:00:00Z", "process": "surface_mesh_generation",
            "derived_from": {"type": "volume", "id": "vol1"},
            "metadata": {"bbox": [[1, 2, 3], [4, 5, 6]], "format": "tifxyz", "scale": [0.05, 0.05], "type": "seg",
              "uuid": "z_dbg_gen_00440_inp_hr", "tiff_dimensions": [360, 468]}},
          "properties": {"width": 123, "height": 456, "original_volume_downscale": 2,
            "render": null, "ink_models_allow": ["model-a"], "ink_models_deny": null,
            "spiral_dataset_revision": "r7",
            "volume_coverage": {
              "vol1": {"bbox_transformed": [[10, 20, 30], [40, 50, 60]], "overlap_ratio": 1.0,
                       "computed_at": "2026-10-05T10:15:13Z"},
              "vol2": {"bbox_transformed": [[0, 0, 0], [1, 1, 1]], "overlap_ratio": 0.0, "computed_at": "x"}}},
          "data": [
            {"type": "tifxyz", "origins": [{"path": "PHerc0139/segments/seg-a/tifxyz/", "access_roots": [
              {"type": "s3", "url": "s3://vesuvius-challenge-open-data/", "usage": "public-read"}]}]},
            {"type": "tifxyz-transformed", "parameters": {"target_volume": "vol2"},
             "creation_info": {"process": "transform", "date": "2026-04-01T00:00:00Z",
               "derived_from": {"type": "segment", "id": "seg-a"}, "publication": {"tier": "testing"}},
             "origins": [{"path": "PHerc0139/segments/seg-a/on-vol2.tifxyz/", "access_roots": [
              {"type": "s3", "url": "s3://vesuvius-challenge-open-data/", "usage": "public-read"}]}]},
            {"type": "ink-detection", "origins": [{"path": "PHerc0139/segments/seg-a/ink.png", "access_roots": [
              {"type": "https", "url": "https://example.test/data/", "usage": "public-read"}]}]},
            {"type": "ink-detection-downsampled", "origins": [{"path": "PHerc0139/segments/seg-a/ink_small.jpg",
              "access_roots": [{"type": "https", "url": "https://example.test/data/", "usage": "public-read"}]}]},
            {"type": "ink-labels", "parameters": {"target_volume": "vol1", "label_set": "20260918"},
             "properties": {"label_set": "20260918", "supersedes": "older", "arrays": ["inklabels", "supervision"],
               "registration": "transferred", "annotated_on": "unknown", "mapping_coverage": 0.5}},
            {"type": "layers-zarr", "parameters": {"render_level": 1, "downscale": 4, "format": "zarr",
               "composite_mode": "max"}, "properties": {"width": 99, "height": 77}, "origins": []}
          ]}
      }
    }
  },
  "models": {
    "model-a": {"id": "model-a", "long_id": "model-a-long", "suffix": "ink", "creation": {"date": "2025-07-01T15:42:04Z"},
      "properties": {"task": "ink-detection", "architecture": "timesformer", "compatible_samples": ["PHerc0139"],
        "excluded_samples": null}}
  }
})";
} // namespace

int main()
{
    try {
        const auto manifest = parseOpenDataManifest(kSample, "fixture://metadata.json");
        require(manifest.manifestUrl == "fixture://metadata.json", "manifest url");
        const auto* sample = manifest.findSample("PHerc0139");
        require(sample, "sample");
        require(sample->type == "scroll" && sample->description == "Example sample", "sample fields");
        require(sample->properties.at("unknown_sample_field") == 7, "sample properties kept");
        require(preferredPhotoArtifact(*sample) &&
                    preferredPhotoArtifact(*sample)->resolvedUrl ==
                        "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/photos/photo.jpg",
                "photo");

        // Scans and volumes.
        const auto& scan = sample->scans.front();
        require(scan.longId == "scan-a-long" && scan.createdAt == "2024-01-01T00:00:00Z", "scan identity");
        require(scan.energyKeV == 111.0 && scan.detectorDistanceMm == 220.0 && scan.pixelSizeUm == 2.403, "scan physics");
        require(scan.location == "ESRF Grenoble" && scan.name == "SCROLL_RADIX", "acquisition");
        const auto* vol1 = sample->findVolume("vol1");
        require(vol1 && vol1->longId == "vol1-long" && vol1->scanId == "scan-a", "volume identity");
        require(vol1->pixelSizeUm == 7.91 && vol1->energyKeV == 88.0 && vol1->detectorDistanceMm == 12.5, "volume physics");
        require(vol1->shapeZYX == std::array<std::size_t, 3>{100, 200, 300} && vol1->dataFormat == "uint8", "volume shape");
        require(vol1->orientation.zTopToBottom == true && vol1->orientation.leftHandedCoordinates == false &&
                    vol1->textSide == "recto",
                "orientation");
        require(vol1->supersededBy && vol1->supersededBy->volumeId == "vol3" && vol1->supersededBy->reason == "rescan",
                "superseded_by");
        require(vol1->license && vol1->license->name == "CC BY-NC 4.0", "license");
        require(vol1->renderLevel == 2 && vol1->renderSegments == true, "render level");
        require(vol1->windowU16Min == -8171 && vol1->windowU16Max == 62026, "windowing");
        require(vol1->creation.derivedFrom && vol1->creation.derivedFrom->id == "scan-a", "volume derived_from");
        require(sample->findVolume("vol2")->pixelSizeUm == 2.403, "pixel size from the referenced scan");
        const auto* zarr = preferredVolumeArtifact(*vol1);
        require(zarr && zarr->sourcePath == "s3://vesuvius-challenge-open-data/PHerc0139/volumes/vol1.zarr" &&
                    zarr->resolvedUrl ==
                        "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/vol1.zarr",
                "public origin preferred over internal");

        // Coordinate-bearing parameters are exact.
        const auto& artifacts = vol1->artifacts;
        require(artifacts[1].levelParameterPresent && artifacts[1].sourceCoordinateLevel == 2, "integer level");
        require(artifacts[2].levelParameterPresent && !artifacts[2].sourceCoordinateLevel, "fractional level rejected");
        require(!artifacts[3].levelParameterPresent, "absent level");
        const auto representations = derivedRepresentations(*sample);
        require(representations.size() == 5 && representations[0].kind == OpenDataRepresentationKind::Prediction &&
                    representations[1].kind == OpenDataRepresentationKind::NormalGrids &&
                    representations[2].kind == OpenDataRepresentationKind::Lasagna &&
                    representationKindName(representations[2].kind) == "Lasagna",
                "derived representations");

        // Registrations: the sample graph, and supersession as an identity edge.
        RegistrationGraph registrations;
        registrations.addSample(*sample);
        const auto forward = registrations.resolve("vol1", "vol2");
        require(forward.status == TransformStatus::Resolved && forward.resolved().forward(0, 0) == 0.5 &&
                    forward.resolved().forward(2, 3) == 30, "volume transform");
        require(sample->volumeTransforms.front().transforms.front().derivationPath == "vol1-vol2", "derivation path");
        require(registrations.resolve("vol2", "vol1").status == TransformStatus::Resolved, "inverse transform");
        require(registrations.resolve("vol1", "missing").status == TransformStatus::Unavailable, "unrelated volume");

        // Opened volumes: the source zarr at a base level, a derived dataset at its level.
        const auto zarrUrl =
            std::string("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/vol1.zarr");
        const auto atL2 = manifest.volumeAt(zarrUrl + "/", 2);
        require(atL2 && atL2->sampleId == "PHerc0139" && atL2->volumeId == "vol1" && atL2->level == 2 && atL2->source,
                "source volume at a base level");
        const auto atL0 = manifest.volumeAt("s3://vesuvius-challenge-open-data/PHerc0139/volumes/vol1.zarr");
        require(atL0 && atL0->level == 0, "published source path");
        require(!atL0->sameGrid(*atL2) && atL2->sameGrid(*manifest.volumeAt(zarrUrl, 2)), "a level is its own grid");
        require(!manifest.volumeAt("https://example.test/other.zarr"), "unknown volume");

        require(sampleNameKey("PHerc. Paris 4") == "phercparis4" && sampleNameKey("PHerc 0332") == "pherc332" &&
                    sampleNameKey("x100") == "x100" && sampleNameKey("000") == "0" && sampleNameKey("Scroll 1") == "scroll1",
                "name keys");
        require(communityAliases("PHercParis4") == std::vector<std::string>{"Scroll 1"} && communityAliases("x").empty(), "aliases");
        require(manifest.samplesNamed("pherc 139") == std::vector<std::string>{"PHerc0139"}, "a padded id");
        require(manifest.samplesNamed("0139") == std::vector<std::string>{"PHerc0139"}, "without the prefix");
        require(manifest.samplesNamed("nothing").empty(), "no such sample");
        require(Affine::blockReduced(4)(0, 3) == 1.5 && Affine::blockReduced(4)(1, 1) == 4, "block reduced");
        OpenDataArtifact gridded;
        gridded.type = "surface-prediction-zarr";
        gridded.sourceCoordinateLevel = 2;
        require(gridded.gridLevel() == 2, "prediction grid level");
        gridded.type = "ome-zarr";
        require(gridded.gridLevel() == 0, "volume grid level");
        gridded.type = "lasagna";
        require(!gridded.gridLevel(), "no grid level");

        const auto metadata = manifest.volumeMetadataAt("s3://vesuvius-challenge-open-data/PHerc0139/volumes/vol1.zarr");
        require(metadata && metadata->sampleId == "PHerc0139" && metadata->volumeId == "vol1" &&
                    metadata->scanId == "scan-a" && metadata->voxelSizeUm == 7.91 && metadata->energyKeV == 88.0 &&
                    metadata->detectorDistanceMm == 12.5 && metadata->dataFormat == "uint8" &&
                    metadata->raw.value("id", "") == "vol1",
                "volume metadata at a location");
        require(!manifest.volumeMetadataAt("https://example.test/other.zarr"), "no metadata for an unknown volume");
        const auto& sample0 = *manifest.findSample("PHerc0139");
        const auto direct = resolveVolumeTransform(sample0, "vol1", "vol2");
        require(direct && direct->derivationPath == "vol1-vol2" && direct->forward(0, 3) == 10, "direct volume transform");
        const auto inverse = resolveVolumeTransform(sample0, "vol2", "vol1");
        require(inverse && inverse->derivationPath.empty() && inverse->forward(0, 0) == 2, "inverse volume transform");
        require(!resolveVolumeTransform(sample0, "vol1", "nowhere"), "no route");
        // The transform and the supersession only: a pyramid level is a scale of its volume.
        require(registrations.registrations().size() == 2, "no level registrations");

        // Segments.
        const auto& segment = sample->segments.front();
        require(segment.longId == "PHerc0139-seg-a" && segment.createdAt == "2026-03-11T00:00:00Z", "segment identity");
        require(segment.width == 123 && segment.height == 456, "segment size");
        require(segment.sourceVolumeId() == "vol1" && segment.stableId() == "PHerc0139-seg-a", "source and lineage");
        require(segment.originalVolumeDownscale == 2, "downscale");
        require(segment.render.is_null(), "segment render");
        require(segment.volumeCoverage.size() == 2, "volume coverage");
        const auto* coverage = segment.coverage("vol1");
        require(coverage && coverage->bbox.low == std::array<double, 3>{10, 20, 30} &&
                    coverage->bbox.high == std::array<double, 3>{40, 50, 60} && coverage->overlapRatio == 1.0,
                "coverage bbox");
        require(segment.coverage("vol2")->overlapRatio == 0 && !segment.coverage("vol3"), "coverage lookup");
        require(segment.originalTifxyz && segment.originalTifxyz->scale == std::array<double, 2>{0.05, 0.05} &&
                    segment.originalTifxyz->tiffDimensions == std::array<int64_t, 2>{360, 468},
                "original tifxyz");
        require(segment.hasInkDetection() && segment.hasLayersZarr(), "artifact availability");
        const auto& published = segment.artifacts[1];
        require(published.creation.process == "transform" && published.creation.derivedFrom, "artifact creation info");
        const auto& labels = segment.artifacts[4];
        require(labels.labelSet == "20260918" && labels.supersedes == std::vector<std::string>{"older"} &&
                    labels.arrays.size() == 2 && labels.registration == "transferred" && labels.mappingCoverage == 0.5,
                "ink labels");
        const auto& layers = segment.artifacts[5];
        require(layers.renderLevel == 1 && layers.downscale == 4 && layers.format == "zarr" &&
                    layers.compositeMode == "max" && layers.width == 99 && layers.height == 77,
                "render parameters");
        require(segment.artifacts[0].url() ==
                    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/seg-a/tifxyz",
                "artifact url without trailing slash");

        // Surfaces: the authored tifxyz is canonical without a published one onto
        // the source; a published one onto the source wins over it.
        auto surfaces = sample->surfaces(segment);
        require(surfaces.size() == 2, "authored and published surfaces");
        require(surfaces[0].authored && surfaces[0].canonical && surfaces[0].volumeId == "vol1" &&
                    surfaces[0].unitsPerVoxel == 0.5 && surfaces[0].artifact == &segment.artifacts[0],
                "authored canonical surface");
        require(surfaces[0].id == "PHerc0139-seg-a-authored", "authored id");
        require(!surfaces[1].authored && !surfaces[1].canonical && surfaces[1].volumeId == "vol2" &&
                    surfaces[1].unitsPerVoxel == 1 && surfaces[1].id == "PHerc0139-seg-a-published-vol2",
                "published surface");
        auto withSource = *sample;
        auto& edited = withSource.segments.front();
        auto onSource = edited.artifacts[1];
        onSource.targetVolumeId = "vol1";
        onSource.resolvedUrl = "https://example.test/on-source";
        edited.artifacts.push_back(onSource);
        auto elsewhere = edited.artifacts[1];
        elsewhere.targetVolumeId = "missing";
        edited.artifacts.push_back(elsewhere);
        surfaces = withSource.surfaces(edited);
        require(surfaces.size() == 2 && !surfaces[0].authored && surfaces[0].canonical &&
                    surfaces[0].artifact == &edited.artifacts[6],
                "published onto the source is canonical; other volumes' artifacts ignored");
        auto noSource = *sample;
        noSource.segments.front().originalVolumeId = "missing";
        noSource.segments.front().creation.derivedFrom.reset();
        require(noSource.surfaces(noSource.segments.front()).empty() && noSource.surfaceSegmentCount() == 0,
                "no surfaces without the source volume");
        require(sample->surfaceSegmentCount() == 1 && sample->inkDetectionSegmentCount() == 1, "counts");

        // Ink images prefer JPEG renders.
        const auto images = inkDetectionImages(segment);
        require(images.size() == 1 && images[0].fileName == "ink-detection-downsampled.jpg" &&
                    images[0].url == "https://example.test/data/PHerc0139/segments/seg-a/ink_small.jpg" &&
                    images[0].label == "seg-a - ink-detection-downsampled",
                "ink images");

        // Where an artifact was copied from, and its prediction type.
        const auto& vol1Artifacts = sample->findVolume("vol1")->artifacts;
        require(vol1Artifacts[0].origins.front().copySourcePath == "s3://scans/PHerc0139/vol1_masked.zarr/" &&
                    vol1Artifacts[0].predictionType.empty(),
                "copy source");
        require(vol1Artifacts[3].type == "lasagna" && vol1Artifacts[3].predictionType == "fibers", "prediction type");
        require(isZarrArtifact(vol1Artifacts[0]) && isZarrArtifact(vol1Artifacts[1]) && !isZarrArtifact(vol1Artifacts[3]),
                "zarr by type");
        OpenDataArtifact byUrl;
        byUrl.resolvedUrl = "https://example.test/data/v.ZARR/?x=1";
        require(isZarrArtifact(byUrl), "zarr by URL");

        // URLs.
        require(resolveOpenDataUrl("s3://vesuvius-challenge/full-scrolls/x") ==
                    "https://data.aws.ash2txt.org/samples/full-scrolls/x",
                "website rewrite");
        require(safePathComponent("..a/b c") == "a_b_c" && safePathComponent("._") == "unnamed", "path component");
        require(joinOpenDataUrl("s3://vesuvius-challenge-open-data/", "/a/b") == "s3://vesuvius-challenge-open-data/a/b",
                "join");

        // Models.
        require(manifest.models.size() == 1, "models");
        const auto& model = manifest.models.front();
        require(model.id == "model-a" && model.longId == "model-a-long" && model.suffix == "ink" &&
                    model.task == "ink-detection" && model.architecture == "timesformer",
                "model");

        // Only published keys are read.
        const auto aliases = parseOpenDataManifest(R"({"metadata": {"samples": {"s": {}}}})");
        require(aliases.samples.empty(), "no metadata wrapper");
        const auto camel = parseOpenDataManifest(
            R"({"samples": {"s": {"volumes": {"v": {"scanId": "x", "pixelSizeUm": 1}}}}})");
        require(camel.samples.front().volumes.front().scanId.empty() &&
                    !camel.samples.front().volumes.front().pixelSizeUm,
                "camelCase keys are not read");
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
