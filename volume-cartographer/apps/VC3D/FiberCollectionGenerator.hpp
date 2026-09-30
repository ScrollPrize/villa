#pragma once

#include <QByteArray>
#include <QProcessEnvironment>
#include <QString>
#include <QStringList>
#include <array>
#include <cstdint>
#include <optional>

// Runs of `python -m vesuvius.afv_spline_generator`, which predicts fibers in a
// zone of CT and writes them as an Automated Fiber Volume.
namespace vc3d::fibergen
{

inline constexpr auto kModule = "vesuvius.afv_spline_generator";
inline constexpr auto kDefaultModel = "Qualzz20/afv_fiber_9um";
inline constexpr double kDefaultThresholdPercent = 60.0;

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
    bool mirror = true;
    double thresholdPercent = kDefaultThresholdPercent;
};

// Arguments after the Python executable, with progress as JSON lines.
QStringList arguments(const Request& request);

struct Event {
    enum class Kind { Progress, Warning, Done, Error };
    Kind kind{};
    QString message;
    // Completed part of the run, from 0 to 1.
    double fraction = 0.0;
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

// `environment` with `vesuviusSource`, if any, first on PYTHONPATH.
QProcessEnvironment environment(QProcessEnvironment environment, const QString& vesuviusSource);

}  // namespace vc3d::fibergen
