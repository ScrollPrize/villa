#pragma once

#include <QObject>
#include <QProcess>
#include <QString>
#include <QTemporaryFile>
#include <memory>

/**
 * Manages the lifecycle of the Python neural trace service.
 *
 * The service is started on first use and kept alive for the duration
 * of the application to avoid repeated startup overhead (torch model
 * compilation takes significant time).
 */
class NeuralTraceServiceManager : public QObject
{
    Q_OBJECT

public:
    static NeuralTraceServiceManager& instance();

    /**
     * Ensure the service is running with the given configuration.
     * If the service is already running with different parameters, it will be
     * restarted. If running with the same parameters, this is a no-op.
     *
     * @param checkpointPath Path to the neural network checkpoint file
     * @param volumeZarr Path to the OME-Zarr volume folder
     * @param volumeScale Scale level to use (0 = full resolution)
     * @param pythonPath Optional path to Python executable (empty = auto-detect)
     * @return true if service is running (or was successfully started)
     */
    bool ensureServiceRunning(const QString& checkpointPath,
                              const QString& volumeZarr,
                              int volumeScale,
                              const QString& pythonPath = QString());

    /**
     * Stop the service if running.
     */
    void stopService();

    /**
     * Check if the service is currently running.
     */
    [[nodiscard]] bool isRunning() const;

    /**
     * Get the socket path for connecting to the service.
     * Returns empty string if service is not running.
     */
    [[nodiscard]] QString socketPath() const;

    /**
     * Get the last error message, if any.
     */
    [[nodiscard]] QString lastError() const { return _lastError; }

    /**
     * Get current service configuration.
     */
    [[nodiscard]] QString currentCheckpointPath() const { return _currentCheckpointPath; }
    [[nodiscard]] QString currentVolumeZarr() const { return _currentVolumeZarr; }
    [[nodiscard]] int currentVolumeScale() const { return _currentVolumeScale; }

signals:
    /**
     * Emitted when the service starts successfully.
     */
    void serviceStarted();

    /**
     * Emitted when the service stops (either intentionally or due to error).
     */
    void serviceStopped();

    /**
     * Emitted when a service error occurs.
     */
    void serviceError(const QString& message);

    /**
     * Emitted with status messages during startup.
     */
    void statusMessage(const QString& message);

private:
    explicit NeuralTraceServiceManager(QObject* parent = nullptr);
    ~NeuralTraceServiceManager() override;

    // Non-copyable
    NeuralTraceServiceManager(const NeuralTraceServiceManager&) = delete;
    NeuralTraceServiceManager& operator=(const NeuralTraceServiceManager&) = delete;

    bool startService(const QString& checkpointPath,
                      const QString& volumeZarr,
                      int volumeScale,
                      const QString& pythonPath);
    void handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void handleProcessError(QProcess::ProcessError error);
    void handleReadyReadStandardOutput();
    void handleReadyReadStandardError();

    std::unique_ptr<QProcess> _process;
    QString _socketPath;
    QString _lastError;

    // Current configuration
    QString _currentCheckpointPath;
    QString _currentVolumeZarr;
    int _currentVolumeScale{0};

    bool _serviceReady{false};
};
