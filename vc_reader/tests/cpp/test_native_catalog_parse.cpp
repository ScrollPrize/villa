// Parsing of published catalog records into typed fields, and the catalog's
// own rules over them: surfaces of a segment, ink images, timestamps, origins.
#include "vc_reader/catalog.hpp"
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
            {"type": "zarr", "origins": [{"path": "PHerc0139/volumes/vol1.zarr", "access_roots": [
              {"type": "s3", "url": "s3://private-root/", "usage": "internal"},
              {"type": "s3", "url": "s3://vesuvius-challenge-open-data/", "usage": "public-read"}]}]},
            {"type": "surface-prediction-zarr", "parameters": {"level": 2}},
            {"type": "normal-grids", "parameters": {"level": 2.5}},
            {"type": "lasagna"},
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
        require(sample->canonicalVolumeId == "vol1", "canonical volume");
        require(sample->properties.at("unknown_sample_field") == 7, "sample properties kept");
        require(preferredPhotoArtifact(*sample) &&
                    preferredPhotoArtifact(*sample)->resolvedUrl ==
                        "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/photos/photo.jpg",
                "photo");

        // Scans and volumes.
        const auto& scan = sample->scans.front();
        require(scan.longId == "scan-a-long" && scan.createdAt == "2024-01-01T00:00:00Z", "scan identity");
        require(scan.energyKeV == 111.0 && scan.detectorDistanceMm == 220.0 && scan.pixelSizeUm == 2.403, "scan physics");
        require(scan.location == "ESRF Grenoble" && scan.beamline == "bm18" && scan.name == "SCROLL_RADIX", "acquisition");
        const auto* vol1 = sample->findVolume("vol1");
        require(vol1 && vol1->longId == "vol1-long" && vol1->scanId == "scan-a", "volume identity");
        require(vol1->pixelSizeUm == 7.91 && vol1->energyKeV == 88.0 && vol1->detectorDistanceMm == 12.5, "volume physics");
        require(vol1->shapeZYX == std::array<std::size_t, 3>{100, 200, 300} && vol1->dataFormat == "uint8", "volume shape");
        require(vol1->zDirectionIsTopToBottom == true && vol1->leftHandedCoordinates == false &&
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
        const auto forward = findSampleVolumeTransform(*sample, "vol1", "vol2");
        require(forward && (*forward)(0, 0) == 0.5 && (*forward)(2, 3) == 30, "volume transform");
        require(sample->volumeTransforms.front().transforms.front().derivationPath == "vol1-vol2", "derivation path");
        require(findSampleVolumeTransform(*sample, "vol2", "vol1").has_value(), "inverse transform");
        require(!findSampleVolumeTransform(*sample, "vol1", "missing"), "unrelated volume");

        // Segments.
        const auto& segment = sample->segments.front();
        require(segment.longId == "PHerc0139-seg-a" && segment.createdAt == "2026-03-11T00:00:00Z", "segment identity");
        require(segment.width == 123 && segment.height == 456, "segment size");
        require(segment.sourceVolumeId() == "vol1" && segment.stableId() == "PHerc0139-seg-a", "source and lineage");
        require(segment.originalVolumeDownscale == 2, "downscale");
        require(segment.inkModelsAllow == std::vector<std::string>{"model-a"} && !segment.inkModelsDeny, "ink models");
        require(segment.spiralDatasetRevision == "r7" && segment.render.is_null(), "segment properties");
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
        require(published.creation.process == "transform" && published.creation.derivedFrom &&
                    published.publicationTier == "testing",
                "artifact creation info");
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
        require(surfaces[0].id.starts_with("PHerc0139-seg-a-derived-native-L0-"), "authored id");
        require(!surfaces[1].authored && !surfaces[1].canonical && surfaces[1].volumeId == "vol2" &&
                    surfaces[1].unitsPerVoxel == 1 && surfaces[1].id.starts_with("PHerc0139-seg-a-published-vol2-L0-"),
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

        // Timestamps and URLs.
        require(catalogTimestamp("2026-06-02T20:44:01.955138Z") == "20260602204401", "fractional timestamp");
        require(catalogTimestamp("2026-06-02T20:44:01Z") == "20260602204401", "timestamp");
        require(catalogTimestamp("20260602204401") == "20260602204401", "digits");
        require(catalogTimestamp("yesterday") == "yesterday", "other dates unchanged");
        require(resolveOpenDataUrl("s3://vesuvius-challenge/full-scrolls/x") ==
                    "https://data.aws.ash2txt.org/samples/full-scrolls/x",
                "website rewrite");
        require(joinOpenDataUrl("s3://vesuvius-challenge-open-data/", "/a/b") == "s3://vesuvius-challenge-open-data/a/b",
                "join");

        // Models.
        const auto* model = manifest.findModel("model-a");
        require(model && model->longId == "model-a-long" && model->suffix == "ink" && model->task == "ink-detection" &&
                    model->architecture == "timesformer" &&
                    model->compatibleSamples == std::vector<std::string>{"PHerc0139"} && !model->excludedSamples,
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
