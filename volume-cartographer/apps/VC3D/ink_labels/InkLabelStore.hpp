#pragma once

// Ink label files: where they live, the JSON sidecar, and the worker-safe save
// and load protocols. GUI-free.
//
// A segment's labels live in <project dir>/ink_labels/<seg>/ and are written
// as the ink detection trainer reads them: <seg>_inklabels.tif,
// <seg>_supervision_mask.tif and optionally <seg>_validation_mask.tif, uint8
// {0,255}, one sample, tiled 1024, LZW, at the vc_render_tifxyz canvas size.
// Auxiliary files (.json, .prev, .autosave, .lock, .tmp-*) use extensions the
// trainer and create_label_zarrs ignore.

#include "ink_labels/InkLabelRaster.hpp"

#include "vc/core/util/RenderCanvas.hpp"

#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class QuadSurface;

namespace vc3d::ink_labels {

enum class LabelKind : uint8_t { Ink, Supervision, Validation };
enum class Target : uint8_t { Trainer, Autosave };
enum class RecoverPolicy : uint8_t { Ask, Restore, Discard, Fail };

inline constexpr LabelKind kAllKinds[] = {LabelKind::Ink, LabelKind::Supervision, LabelKind::Validation};

// "inklabels", "supervision_mask" or "validation_mask": the trainer's names.
[[nodiscard]] const char* kindName(LabelKind kind);
[[nodiscard]] std::optional<LabelKind> kindFromName(std::string_view name);
[[nodiscard]] uint8_t bitFor(LabelKind kind);

// Identity of a label canvas. The scale is compared by its float32 bits, so a
// JSON round trip cannot change it.
struct CanvasKey {
    int width = 0, height = 0, gridCols = 0, gridRows = 0;
    uint32_t scaleXBits = 0, scaleYBits = 0;
    double renderScale = 1.0;
    bool operator==(const CanvasKey&) const = default;
    [[nodiscard]] float scaleX() const;
    [[nodiscard]] float scaleY() const;
    // "grid 152 x 76 at scale 0.05 (3040 x 1520 px)"
    [[nodiscard]] QString describe() const;
};
[[nodiscard]] CanvasKey canvasKeyFor(const vc::RenderCanvas& canvas);

struct StorePaths {
    std::filesystem::path dir;
    std::string prefix;
    int version = 1;

    // <prefix>_inklabels[_vN].tif, plus ".autosave" for the autosave target.
    [[nodiscard]] std::filesystem::path file(LabelKind kind, Target target = Target::Trainer) const;
    // <prefix>_inklabels[_vN].json, plus ".autosave".
    [[nodiscard]] std::filesystem::path sidecar(Target target = Target::Trainer) const;
    // The trainer file's previous generation: <name>.tif.prev.
    [[nodiscard]] std::filesystem::path previous(LabelKind kind) const;
    [[nodiscard]] std::filesystem::path lock() const;
};
[[nodiscard]] StorePaths storePathsFor(const std::filesystem::path& projectDir, const std::string& segmentName);
// The folder that holds ink_labels/ for a project file: the project's folder,
// unless the project file lies inside `cacheRoot` (an open-data catalog
// project, kept in the remote cache that can be wiped and that every sample
// shares). Then it is <dataRoot>/<project file name without .volpkg.json>.
[[nodiscard]] std::filesystem::path labelRootFor(const std::filesystem::path& projectFile,
                                                 const std::filesystem::path& cacheRoot,
                                                 const std::filesystem::path& dataRoot);

struct Sidecar {
    CanvasKey canvas;
    std::string segmentName, segmentUuid, segmentPath, geometrySha256, savedAtUtc, savedBy;
    std::map<LabelKind, std::string> fileNames;
    std::map<LabelKind, std::string> sha256;
    uint64_t stateSerial = 0;
    // Counts every write of the store, trainer save or checkpoint, and only
    // goes up. An autosave is newer than the trainer save when its write_seq
    // is larger; the state serial cannot tell, because undo moves it back.
    uint64_t writeSeq = 0;
    bool autosave = false;
    QJsonObject edits;
    QJsonArray provenance;
    std::optional<std::string> predictionShown;
};
inline constexpr const char* kSidecarSchema = "vc3d.ink_labels/1";
inline constexpr int kMaxProvenanceEntries = 1000;

[[nodiscard]] std::optional<Sidecar> readSidecar(const std::filesystem::path& path, QString* error);
// Written with QSaveFile, so a reader sees the old or the new sidecar.
[[nodiscard]] bool writeSidecar(const std::filesystem::path& path, const Sidecar& sidecar, QString* error);
// SHA-256 over the grid size, the scale bits and the raw x/y/z float32 rows.
[[nodiscard]] std::string geometrySha256(const QuadSurface& surface);
// Lowercase hex SHA-256 of a file's bytes, or empty if it cannot be read.
[[nodiscard]] std::string fileSha256(const std::filesystem::path& path);

struct SaveRequest {
    TileRaster snapshot;
    StorePaths paths;
    Sidecar sidecar;
    Target target = Target::Trainer;
    bool writeValidation = false;
    bool overwriteExternalEdits = false;
    // SHA-256 of each trainer file when it was loaded or last saved. A
    // trainer file that no longer matches was changed outside VC3D.
    std::map<LabelKind, std::string> expectedSha256;
};

struct SaveOutcome {
    bool ok = false;
    QString error;
    // Trainer files that changed outside VC3D; the save wrote nothing.
    QStringList externallyEdited;
    std::vector<std::pair<LabelKind, std::filesystem::path>> files;
    std::map<LabelKind, std::string> sha256;
    uint64_t stateSerial = 0;
    Target target = Target::Trainer;
    double ms = 0;
};

// Writes temp files from the snapshot (three writers in parallel), checks the
// trainer files for outside edits, renames validation, supervision then ink
// into place (keeping one .prev generation for the trainer target), and
// writes the sidecar last. A trainer save then deletes the store's
// checkpoint: the controller waits for a running checkpoint before it starts
// a save, so any checkpoint on disk is older than the state just written.
// Safe to run on a worker: it reads only the immutable snapshot and touches
// only the store's files.
[[nodiscard]] SaveOutcome saveLabels(const SaveRequest& request);

struct LoadOutcome {
    bool ok = false;
    QString error;
    // The sidecar's canvas differs from the expected one; nothing was read.
    bool canvasMismatch = false;
    TileRaster raster;
    std::optional<Sidecar> sidecar;
    std::map<LabelKind, std::string> sha256;
    bool validationPresent = false;
    bool anyFile = false;
    QStringList warnings;
};

// Reads the target's sidecar (if any) and each present label file, streaming
// block by block. A file whose shape differs from the canvas is refused.
[[nodiscard]] LoadOutcome loadLabels(const StorePaths& paths, const CanvasKey& expected, Target target);

// True if any of the target's label files or its sidecar exist.
[[nodiscard]] bool targetExists(const StorePaths& paths, Target target);
// Removes the target's label files and sidecar. Returns false if one stayed.
bool removeTarget(const StorePaths& paths, Target target);
// Renames the target's label files and sidecar to <name>.unusable (replacing
// an older one), keeping a checkpoint that cannot be restored out of the way
// without deleting it. Returns false if one could not be moved.
bool moveTargetAside(const StorePaths& paths, Target target);

}  // namespace vc3d::ink_labels
