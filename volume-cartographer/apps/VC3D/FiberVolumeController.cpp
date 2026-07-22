#include "FiberVolumeController.hpp"

#include "CState.hpp"
#include "LineAnnotationController.hpp"
#include "OpenDataCoordinateIdentity.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/FiberAnnotationVolume.hpp"
#include "vc/core/util/SparseAnnotationVolume.hpp"

#include <QCryptographicHash>
#include <QFileSystemWatcher>
#include <QtConcurrent/QtConcurrentRun>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <string_view>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

struct FiberVolumeController::BuildInput {
    std::uint64_t serial{};
    fs::path root;
    std::string hash;
    vc::SparseAnnotationVolumeSpec spec;
    vc::FiberAnnotationBatches adapter;
};

namespace {

void hashBytes(QCryptographicHash& hash, const void* data, std::size_t size)
{
    hash.addData(QByteArrayView(static_cast<const char*>(data),
                                static_cast<qsizetype>(size)));
}

template <typename T>
void hashValue(QCryptographicHash& hash, const T& value)
{
    hashBytes(hash, &value, sizeof(value));
}

std::optional<std::string> coordinateSpaceFromFiber(const fs::path& path)
{
    try {
        std::ifstream input(path);
        if (!input)
            return std::nullopt;
        nlohmann::json root;
        input >> root;
        if (root.contains("vc_open_data_coordinate_space") &&
            root["vc_open_data_coordinate_space"].is_string()) {
            return root["vc_open_data_coordinate_space"].get<std::string>();
        }
    } catch (...) {
    }
    return std::nullopt;
}

std::string coordinateSpaceForVolume(const VolumePkg& package,
                                     const std::string& volumeId)
{
    constexpr std::string_view prefix = "vc-open-data-coordinate-space:";
    for (const auto& tag : package.volumeTags(volumeId)) {
        if (tag.rfind(prefix, 0) == 0)
            return tag.substr(prefix.size());
    }
    return {};
}

FiberVolumeController::BuildResult buildVolume(FiberVolumeController::BuildInput input)
{
    FiberVolumeController::BuildResult result;
    result.serial = input.serial;
    result.hash = input.hash;
    result.path = input.root / input.hash;
    try {
        if (fs::exists(result.path / ".zattrs"))
            return result;

        const fs::path temporary = input.root /
            (input.hash + ".tmp-" + std::to_string(input.serial));
        std::error_code ec;
        fs::remove_all(temporary, ec);
        fs::create_directories(input.root);
        vc::writeSparseAnnotationVolume(input.spec, input.adapter.batches, temporary);
        fs::rename(temporary, result.path, ec);
        if (ec) {
            if (!fs::exists(result.path / ".zattrs"))
                throw std::runtime_error("failed to publish fiber volume: " + ec.message());
            fs::remove_all(temporary, ec);
        }
    } catch (const std::exception& ex) {
        result.error = ex.what();
        std::error_code ec;
        fs::remove_all(input.root /
            (input.hash + ".tmp-" + std::to_string(input.serial)), ec);
    }
    return result;
}

} // namespace

FiberVolumeController::FiberVolumeController(CState* state,
                                             LineAnnotationController* fibers,
                                             QObject* parent)
    : QObject(parent), _state(state), _fibers(fibers),
      _watcher(std::make_unique<QFileSystemWatcher>())
{
    _debounce.setSingleShot(true);
    _debounce.setInterval(250);
    connect(&_debounce, &QTimer::timeout, this, &FiberVolumeController::startBuild);
    connect(&_buildWatcher, &QFutureWatcher<BuildResult>::finished,
            this, &FiberVolumeController::finishBuild);
    connect(_watcher.get(), &QFileSystemWatcher::directoryChanged,
            this, [this] { requestRebuild(); });
    connect(_watcher.get(), &QFileSystemWatcher::fileChanged,
            this, [this] { requestRebuild(); });

    if (_fibers) {
        connect(_fibers, &LineAnnotationController::fibersChanged,
                this, [this] { requestRebuild(); });
        connect(_fibers, &LineAnnotationController::fiberSaved,
                this, [this] { requestRebuild(); });
        connect(_fibers, &LineAnnotationController::fibersDeleted,
                this, [this] { requestRebuild(); });
    }
    if (_state) {
        connect(_state, &CState::volumeChanged,
                this, [this] { requestRebuild(); });
        connect(_state, &CState::vpkgChanged,
                this, [this] { requestRebuild(); });
    }
    requestRebuild();
}

FiberVolumeController::~FiberVolumeController()
{
    if (_buildWatcher.isRunning())
        _buildWatcher.waitForFinished();
}

void FiberVolumeController::requestRebuild()
{
    ++_requestSerial;
    _rebuildPending = true;
    refreshWatchPaths();
    _debounce.start();
}

void FiberVolumeController::refreshWatchPaths()
{
    if (!_watcher || !_fibers)
        return;
    const auto oldDirectories = _watcher->directories();
    const auto oldFiles = _watcher->files();
    if (!oldDirectories.empty())
        _watcher->removePaths(oldDirectories);
    if (!oldFiles.empty())
        _watcher->removePaths(oldFiles);

    const fs::path directory = _fibers->fiberDirectory();
    if (fs::is_directory(directory))
        _watcher->addPath(QString::fromStdString(directory.string()));
    for (const auto& snapshot : _fibers->fiberSnapshotsFromStorageWithPaths()) {
        const fs::path absolute = directory / snapshot.fiberPath.filename();
        if (fs::is_regular_file(absolute))
            _watcher->addPath(QString::fromStdString(absolute.string()));
    }
}

FiberVolumeController::BuildInput FiberVolumeController::captureInput() const
{
    BuildInput input;
    input.serial = _requestSerial;
    if (!_state || !_state->currentVolume() || !_state->vpkg() || !_fibers)
        return input;

    const auto volume = _state->currentVolume();
    const auto directory = _fibers->fiberDirectory();
    if (directory.empty())
        return input;
    input.root = directory / ".fiber-volumes";
    const auto shape = volume->shape();
    input.spec.shapeZYX = {
        static_cast<std::size_t>(shape[0]),
        static_cast<std::size_t>(shape[1]),
        static_cast<std::size_t>(shape[2]),
    };
    for (int level : volume->presentScaleLevels()) {
        if (level <= 0)
            continue;
        const auto levelShape = volume->shape(level);
        vc::SparseAnnotationPyramidLevel outputLevel;
        outputLevel.shapeZYX = {
            static_cast<std::size_t>(levelShape[0]),
            static_cast<std::size_t>(levelShape[1]),
            static_cast<std::size_t>(levelShape[2]),
        };
        for (std::size_t axis = 0; axis < 3; ++axis) {
            outputLevel.scaleZYX[axis] =
                static_cast<double>(input.spec.shapeZYX[axis]) /
                static_cast<double>(outputLevel.shapeZYX[axis]);
        }
        input.spec.pyramidLevels.push_back(outputLevel);
    }

    const auto identity = vc3d::opendata::coordinateIdentityForVolume(
        *_state->vpkg(), _state->currentVolumeId());
    const std::string coordinateSpace = identity
        ? identity->coordinateSpace
        : coordinateSpaceForVolume(*_state->vpkg(), _state->currentVolumeId());
    std::vector<vc::FiberAnnotationInput> fibers;
    for (const auto& snapshot : _fibers->fiberSnapshotsFromStorageWithPaths()) {
        vc::FiberAnnotationInput fiber;
        fiber.identity = snapshot.fiberPath.generic_string();
        fiber.sourcePath = snapshot.fiberPath;
        const fs::path absolute = directory / snapshot.fiberPath.filename();
        fiber.coordinateSpace = coordinateSpaceFromFiber(absolute);
        fiber.linePointsXYZ.reserve(snapshot.fiber.points.size());
        for (const auto& point : snapshot.fiber.points)
            fiber.linePointsXYZ.push_back({point.position[0], point.position[1], point.position[2]});
        fiber.controlPointsXYZ.reserve(snapshot.fiber.controlPoints.size());
        for (const auto& point : snapshot.fiber.controlPoints)
            fiber.controlPointsXYZ.push_back({point[0], point[1], point[2]});
        fibers.push_back(std::move(fiber));
    }
    input.adapter = vc::makeFiberAnnotationBatches(fibers, coordinateSpace);
    input.spec.rootAttributes = vc::fiberAnnotationAttributes(input.adapter, coordinateSpace);
    input.spec.rootAttributes.update(vc3d::opendata::coordinateIdentityJson(identity));

    QCryptographicHash hash(QCryptographicHash::Sha256);
    for (const auto value : input.spec.shapeZYX)
        hashValue(hash, value);
    for (const auto value : input.spec.chunkShapeZYX)
        hashValue(hash, value);
    hashValue(hash, input.spec.fillValue);
    hashValue(hash, input.spec.compressionLevel);
    hash.addData(QByteArrayView(input.spec.compressor.data(),
                                static_cast<qsizetype>(input.spec.compressor.size())));
    for (const auto& level : input.spec.pyramidLevels) {
        for (const auto value : level.shapeZYX)
            hashValue(hash, value);
        for (const auto value : level.scaleZYX)
            hashValue(hash, value);
        for (const auto value : level.translationZYX)
            hashValue(hash, value);
    }
    const std::string attributes = input.spec.rootAttributes.dump();
    hash.addData(QByteArrayView(attributes.data(),
                                static_cast<qsizetype>(attributes.size())));
    for (const auto& mapping : input.adapter.labels) {
        hash.addData(QByteArrayView(mapping.identity.data(),
                                    static_cast<qsizetype>(mapping.identity.size())));
        hashValue(hash, mapping.label);
    }
    for (const auto& batch : input.adapter.batches) {
        hashValue(hash, batch.label);
        hashValue(hash, batch.radius);
        hashValue(hash, batch.geometryMode);
        for (const auto& point : batch.coordinates)
            hashBytes(hash, point.data(), point.size() * sizeof(double));
    }
    input.hash = hash.result().toHex().left(24).toStdString();
    return input;
}

void FiberVolumeController::startBuild()
{
    if (_buildWatcher.isRunning())
        return;
    _rebuildPending = false;
    BuildInput input;
    try {
        input = captureInput();
    } catch (const std::exception& ex) {
        emit buildFailed(QString::fromUtf8(ex.what()));
        emit availabilityChanged(false);
        return;
    }
    if (input.root.empty() || input.hash.empty() || input.adapter.labels.empty()) {
        _publishedHash.clear();
        _publishedVolume.reset();
        emit availabilityChanged(false);
        return;
    }
    if (input.hash == _publishedHash) {
        if (_publishedVolume)
            emit volumeReady(_publishedVolume, "fiber-volume:" + _publishedHash);
        emit availabilityChanged(true);
        return;
    }
    _runningSerial = input.serial;
    _buildWatcher.setFuture(QtConcurrent::run(buildVolume, std::move(input)));
}

void FiberVolumeController::finishBuild()
{
    const BuildResult result = _buildWatcher.result();
    if (!result.error.empty())
        emit buildFailed(QString::fromStdString(result.error));
    if (result.error.empty() && result.serial == _requestSerial) {
        try {
            auto volume = Volume::New(result.path);
            _publishedHash = result.hash;
            _publishedVolume = volume;
            emit volumeReady(std::move(volume), "fiber-volume:" + result.hash);
            emit availabilityChanged(true);
        } catch (const std::exception& ex) {
            emit buildFailed(QString::fromUtf8(ex.what()));
        }
    }
    if (_rebuildPending || result.serial != _requestSerial)
        _debounce.start();
}
