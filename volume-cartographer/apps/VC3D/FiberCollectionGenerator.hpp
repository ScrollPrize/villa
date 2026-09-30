#pragma once

#include <QByteArray>
#include <QProcessEnvironment>
#include <QString>
#include <QStringList>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

// Runs of `python -m vesuvius.afv_spline_generator`, which predicts fibers in a
// zone of CT block by block, stitches them into long fibers and writes them as
// an Automated Fiber Volume.
namespace vc3d::fibergen
{

inline constexpr auto kModule = "vesuvius.afv_spline_generator";
inline constexpr auto kDefaultModel = "Qualzz20/afv_fiber_9um";
inline constexpr double kDefaultThresholdPercent = 60.0;
inline constexpr int kDefaultBlockSize = 512;
inline constexpr double kDefaultMaxJoinAngle = 45.0;
inline constexpr double kDefaultMinLength = 32.0;
inline constexpr double kDefaultBlackDistance = 16.0;

struct Request {
    // Local OME-Zarr directory or http(s) URL, and the array read as the volume.
    QString volume;
    int level = 0;
    // XYZ voxels of the volume.
    std::array<int, 3> origin{};
    std::array<int, 3> size{};
    QString output;
    QString coordinateSpace;
    std::uint64_t nativeScale = 1;
    // Native voxel size in micrometres; 0 when unknown.
    double voxelSizeUm = 0.0;
    QString sourcePath;
    QString model = kDefaultModel;
    bool mirror = false;
    double thresholdPercent = kDefaultThresholdPercent;
    int blockSize = kDefaultBlockSize;
    // Join fibers across gaps with the gap model (slow).
    bool extend = false;
    // Inferred joins turning more than this many degrees are cut.
    double maxJoinAngle = kDefaultMaxJoinAngle;
    // Fibers shorter than this, or passing this close to the black outside the
    // papyrus, are removed; 0 keeps them. Voxels of the volume.
    double minLength = kDefaultMinLength;
    double blackDistance = kDefaultBlackDistance;
    // Receives a .afv of the fibers stitched so far after each block; empty for none.
    QString previewDirectory;
};

// A block of the zone, in XYZ voxels of the volume.
struct Block {
    std::array<int, 3> origin{};
    std::array<int, 3> size{};
    bool operator==(const Block&) const = default;
};

// The zone cut into cubes of `blockSize`, the last ones shorter, x varying
// fastest: the blocks the generator processes, in its order.
std::vector<Block> zoneBlocks(const std::array<int, 3>& origin, const std::array<int, 3>& size, int blockSize);

// Arguments after the Python executable, with progress as JSON lines.
QStringList arguments(const Request& request);

struct Event {
    enum class Kind { Progress, Plan, Block, Preview, Warning, Done, Error };
    Kind kind{};
    QString message;
    // Completed part of the run, from 0 to 1.
    double fraction = 0.0;
    // Plan: every block of the run.
    std::vector<Block> blocks;
    // Block: the block and its new state (reading, predicting, splines,
    // stitching, stitched, extending or done).
    int index = -1;
    QString state;
    // Preview: a .afv of the fibers stitched so far.
    QString path;
    // Done only.
    QString output;
    qint64 fibers = 0;
};

// One line of the generator's standard output; nullopt for anything that is not an event.
std::optional<Event> parseEvent(const QByteArray& line);

// The zone of `size` (clamped to the volume) centered as close to `center` as
// the volume allows. All values are XYZ voxels.
std::array<int, 3> zoneOrigin(const std::array<int, 3>& center, const std::array<int, 3>& size, const std::array<int, 3>& shape);

// vesuvius/src of the source tree VC3D was built from, or empty when vesuvius
// must be installed in the Python environment.
QString vesuviusSourceDirectory(const QString& applicationDirectory);

// The Python of the environment next to `vesuviusSource` (vesuvius/.venv, as
// `uv sync` creates it), or empty.
QString checkoutPython(const QString& vesuviusSource);

// `environment` with `vesuviusSource`, if any, first on PYTHONPATH.
QProcessEnvironment environment(QProcessEnvironment environment, const QString& vesuviusSource);

}  // namespace vc3d::fibergen
