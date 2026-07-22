#pragma once

#include <QFutureWatcher>
#include <QObject>
#include <QTimer>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

class CState;
class LineAnnotationController;
class QFileSystemWatcher;
class Volume;

class FiberVolumeController final : public QObject
{
    Q_OBJECT

public:
    struct BuildInput;
    struct BuildResult {
        std::uint64_t serial{};
        std::string hash;
        std::filesystem::path path;
        std::string error;
    };

    FiberVolumeController(CState* state,
                          LineAnnotationController* fibers,
                          QObject* parent = nullptr);
    ~FiberVolumeController() override;

    void requestRebuild();

signals:
    void volumeReady(std::shared_ptr<Volume> volume, std::string volumeId);
    void availabilityChanged(bool available);
    void buildFailed(QString message);

private:
    void refreshWatchPaths();
    void startBuild();
    BuildInput captureInput() const;
    void finishBuild();

    CState* _state{};
    LineAnnotationController* _fibers{};
    QTimer _debounce;
    std::unique_ptr<QFileSystemWatcher> _watcher;
    QFutureWatcher<BuildResult> _buildWatcher;
    std::uint64_t _requestSerial{};
    std::uint64_t _runningSerial{};
    bool _rebuildPending{};
    std::string _publishedHash;
    std::shared_ptr<Volume> _publishedVolume;
};
