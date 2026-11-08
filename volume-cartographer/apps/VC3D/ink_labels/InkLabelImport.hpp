#pragma once

// Importing existing label images into an ink label session, and the
// dataset entry that hands saved labels to the ink detection trainer. GUI-free.
//
// An import sets one label bit (ink, supervision or validation) from a file:
// pixels above 0 get the bit, pixels at 0 lose it, matching how the trainer
// and create_label_zarrs read label images. Files must be at the canvas size;
// a half or quarter resolution render can be resampled when asked to.

#include "ink_labels/InkLabelAssist.hpp"
#include "ink_labels/InkLabelRaster.hpp"
#include "ink_labels/InkLabelStore.hpp"

#include "vc/core/util/RenderCanvas.hpp"

#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace vc3d::ink_labels {

enum class ImportFormat : uint8_t {
    Unknown,
    Tiff,
    Png,
    LabelZarr65,    // create_label_zarrs output: <x>.zarr/0 of shape (65, H, W), label in plane 32
    Zarr2D,         // a 2D uint8 or uint16 array at <x>.zarr/0 (zarr v2 or v3)
    LegacyPngPair,  // <seg>_inklabels.png with <seg>_mask.png beside it
};
enum class LegacyMaskAs : uint8_t { Ask, Supervision, Ignore };

// PNG imports read the whole image; anything larger is refused.
inline constexpr std::uint64_t kMaxPngBytes = std::uint64_t(1) << 30;

struct ImportOptions {
    LabelKind kind = LabelKind::Ink;
    // Allow nearest-neighbor import of a render at scale 0.5 or 0.25.
    bool allowResample = false;
    // What <seg>_mask.png (the valid surface) becomes for a legacy pair.
    LegacyMaskAs legacyMaskAs = LegacyMaskAs::Ask;
};

struct ImportOutcome {
    bool ok = false;
    QString error;
    // The caller must decide: a legacy mask (legacyMaskAs is Ask) or a
    // resample (allowResample is false). Nothing was imported.
    bool needsDecision = false;
    ImportFormat format = ImportFormat::Unknown;
    cv::Size sourceShape;
    // 1, or the render scale the source was resampled from.
    double resampledFrom = 1.0;
    // Pixels whose value is neither 0 nor full scale.
    std::uint64_t nonBinary = 0;
    // Pixels that got the bit.
    std::uint64_t setPixels = 0;
    QStringList warnings;
};

[[nodiscard]] ImportFormat detectFormat(const std::filesystem::path& path);
[[nodiscard]] const char* formatName(ImportFormat format);
// The label kind a trainer file name implies: *_supervision_mask*, *_validation_mask*, else ink.
[[nodiscard]] LabelKind kindForFileName(const std::filesystem::path& path);

// Reads `path` and applies it to `into` (whose raster is the canvas's size).
// Safe on a worker when `into` edits a raster the worker owns.
[[nodiscard]] ImportOutcome importLabel(const std::filesystem::path& path,
                                        const vc::RenderCanvas& canvas,
                                        const ImportOptions& options,
                                        EditTxn& into);

// Label versions present in a store folder: 1 for the unversioned files,
// N >= 2 for *_vN (VC3D never writes _v1 or _v0, and a number too large for
// an int is skipped). Sorted, without duplicates.
[[nodiscard]] std::vector<int> listVersions(const std::filesystem::path& dir, std::string_view prefix);

// The ink detection trainer's dataset entry for this store's segment:
// segments_path, segment_names, surface_volume_paths (the surface volume
// rendered next to the labels) and, when the store has more than one label
// version, label_version, so the trainer reads the version that was saved
// instead of the newest one. volume_scale is left for the caller's config.
// The commands that make the label zarrs and the surface volume are in
// docs/ink_labels.md.
[[nodiscard]] QJsonObject datasetEntry(const StorePaths& paths, const std::filesystem::path& projectDir,
                                       const std::vector<int>& versions);

}  // namespace vc3d::ink_labels
