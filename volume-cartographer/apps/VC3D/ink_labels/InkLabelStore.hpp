#pragma once

// Ink label files: where they live, the JSON sidecar, and the worker-safe save
// and load protocols. GUI-free.
//
// A segment's labels live in <project dir>/ink_labels/<seg>/ and are written
// as the ink detection trainer reads them: <seg>_inklabels.tif,
// <seg>_supervision_mask.tif and optionally <seg>_validation_mask.tif, uint8
// {0,255}, one sample, tiled 1024, LZW, at the vc_render_tifxyz canvas size.
// Auxiliary files (.json, .prev, .autosave, .saving, .lock, .tmp-*) use
// extensions the trainer and create_label_zarrs ignore.

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
    // The sidecar's name plus ".saving": present only while a save replaces
    // the target's files (see saveLabels).
    [[nodiscard]] std::filesystem::path saving(Target target = Target::Trainer) const;
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

// A few grid vertices and where they were when the labels were saved. A
// labeled pixel belongs to a grid position, so a later flip or rotation of
// the grid (same size, vertices reordered) must move the labels with it.
struct GridAnchor {
    int row = 0, col = 0;
    cv::Vec3f point;
};

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
    // Written as "grid_anchors": [[row, col, x, y, z], ...]. Empty for
    // sidecars written before VC3D recorded them.
    std::vector<GridAnchor> gridAnchors;
};
inline constexpr const char* kSidecarSchema = "vc3d.ink_labels/1";
// The .saving record: {"schema", "sidecar": the sidecar being written,
// "replaces": {kind: {"sha256": hash of the file replaced, or null when the
// save creates it}}}. A kind the save does not touch is not listed; a listed
// kind whose sidecar entry is null is removed by the save.
inline constexpr const char* kSavingSchema = "vc3d.ink_labels.saving/1";
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
    // Tests: stop as a killed process would after this many files were
    // renamed into place, leaving the .saving record behind. -1 never stops.
    int crashAfterRenamesForTesting = -1;
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
// trainer files for outside edits, records the generation it is about to put
// in place (the new sidecar and the hash of each file it replaces) in the
// .saving file, renames validation, supervision then ink into place (keeping
// one .prev generation for the trainer target), writes the sidecar, and only
// then removes the .saving file. A failure on the way puts every replaced
// file back, so a failed save leaves the previous generation. A trainer save
// then deletes the store's checkpoint: the controller waits for a running
// checkpoint before it starts a save, so any checkpoint on disk is older than
// the state just written. Safe to run on a worker: it reads only the
// immutable snapshot and touches only the store's files.
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
    // A save stopped part way (the process died) and left files of two
    // generations; nothing was read.
    bool interruptedSave = false;
    QStringList warnings;
};

// What an interrupted save (a .saving file) left of a target's files.
enum class SaveGeneration : uint8_t {
    // No .saving file: no save was interrupted.
    Settled,
    // Every file is the previous generation: the save stopped before
    // replacing any.
    Previous,
    // Every file is the new generation; only the sidecar was not written.
    New,
    // Files of both generations, files changed since, or an unreadable
    // .saving file. Readers refuse these labels.
    Mixed,
};
[[nodiscard]] SaveGeneration saveGeneration(const StorePaths& paths, Target target);

enum class RepairResult : uint8_t { NothingToRepair, Finished, RolledBack, Failed };
// Settles an interrupted save of the target: a complete new generation gets
// its sidecar, a complete previous one just loses the .saving file, and the
// trainer target's mixed files are put back from .prev when each .prev is the
// previous generation the save recorded. Failed leaves everything in place.
[[nodiscard]] RepairResult repairInterruptedSave(const StorePaths& paths, Target target, QString* message);

// Reads the target's sidecar (if any) and each present label file, streaming
// block by block. A file whose shape differs from the canvas is refused, and
// so are files a save left of two generations (SaveGeneration::Mixed); files
// a save replaced completely load with the sidecar it was writing.
[[nodiscard]] LoadOutcome loadLabels(const StorePaths& paths, const CanvasKey& expected, Target target);

// True if any of the target's label files, its sidecar or a .saving file
// exist.
[[nodiscard]] bool targetExists(const StorePaths& paths, Target target);
// Removes the target's label files, sidecar and .saving file. Returns false if
// one stayed.
bool removeTarget(const StorePaths& paths, Target target);
// Renames the target's label files, sidecar and .saving file to
// <name>.unusable (replacing
// an older one), keeping a checkpoint that cannot be restored out of the way
// without deleting it. Returns false if one could not be moved.
bool moveTargetAside(const StorePaths& paths, Target target);

// ---------------------------------------------------------------------------
// Grid reorders
// ---------------------------------------------------------------------------

// Up to 7 x 7 valid vertices spread over the grid.
[[nodiscard]] std::vector<GridAnchor> gridAnchors(const cv::Mat_<cv::Vec3f>& points);

// How the vertices of the saved grid were reordered into the current one, as
// the old (row, col) of a current vertex: swap exchanges row and column (a
// quarter turn or transpose of a square grid), then flipRows and flipCols
// mirror them. All false is the same order.
struct GridReorder {
    bool swap = false, flipRows = false, flipCols = false;
    [[nodiscard]] bool identity() const { return !swap && !flipRows && !flipCols; }
    bool operator==(const GridReorder&) const = default;
};

enum class ReorderMatch : uint8_t {
    // No anchors to compare, or too few vertices still valid.
    Unknown,
    // The anchors are where they were, or most of them are: an edit in place.
    Same,
    // The grid was reordered; `reorder` says how.
    Reordered,
};
struct ReorderResult {
    ReorderMatch match = ReorderMatch::Unknown;
    GridReorder reorder;
};
// Which reorder of the saved grid the current points are. A reorder is
// reported only when it explains at least half of the comparable anchors and
// more of them than leaving the order alone does.
[[nodiscard]] ReorderResult detectReorder(const std::vector<GridAnchor>& anchors, const cv::Mat_<cv::Vec3f>& points);

struct RemapOutcome {
    bool ok = false;
    // Why the labels cannot be moved exactly (ok is false).
    QString error;
    TileRaster raster;
    // Labeled pixels whose grid position lands off the canvas.
    std::uint64_t droppedLabeled = 0;
};
// The labels of `raster` (on `canvas`) moved to where `reorder` put their
// grid positions. Exact only when every label pixel lands on a label pixel:
// the label pixel pitch must divide the grid step, and a swap needs a square
// canvas with equal scales. Otherwise ok is false and nothing is moved.
[[nodiscard]] RemapOutcome remapLabels(const TileRaster& raster, const vc::RenderCanvas& canvas, const GridReorder& reorder);

}  // namespace vc3d::ink_labels
