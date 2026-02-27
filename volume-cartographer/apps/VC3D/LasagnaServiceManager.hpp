#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QProcess>
#include <QString>
#include <QTimer>
#include <memory>

class QNetworkAccessManager;
class QNetworkReply;

/**
 * Manages the lifecycle of the Python lasagna HTTP service.
 *
 * Supports two modes:
 *   - Internal: launches lasagna_service.py as a subprocess
 *   - External: connects to a pre-started service on a known host:port
 *
 * The service is kept alive so repeated optimizations avoid
 * Python/torch startup overhead.
 */
class LasagnaServiceManager : public QObject
{
    Q_OBJECT

public:
    static LasagnaServiceManager& instance();

    /**
     * Ensure the service process is running (internal mode).
     * @param pythonPath  Path to python executable (empty = auto-detect).
     * @return true if service is running (or was successfully started).
     */
    bool ensureServiceRunning(const QString& pythonPath = QString());

    /**
     * Connect to an externally started service.
     * Pings GET /health; on success marks the service as ready.
     */
    void connectToExternal(const QString& host, int port);

    /** Stop the service process (internal) or disconnect (external). */
    void stopService();

    [[nodiscard]] bool isRunning() const;
    [[nodiscard]] QString lastError() const { return _lastError; }
    [[nodiscard]] int port() const { return _port; }
    [[nodiscard]] QString host() const { return _host; }
    [[nodiscard]] bool isExternal() const { return _isExternal; }

    /**
     * Submit an optimization job.
     * @param config  JSON body for POST /optimize.
     * @param localOutputDir  Where to unpack results after completion.
     */
    void startOptimization(const QJsonObject& config,
                           const QString& localOutputDir = QString());

    /** Request cancellation of the running optimization. */
    void stopOptimization();

    /**
     * Scan ~/.fit_services/*.json for running services.
     * Stale entries (dead PIDs) are removed.
     */
    static QJsonArray discoverServices();

    /** Fetch available datasets from the connected service (GET /datasets). */
    void fetchDatasets();

signals:
    void serviceStarted();
    void serviceStopped();
    void serviceError(const QString& message);
    void statusMessage(const QString& message);

    void optimizationStarted();
    void optimizationProgress(const QString& stage, int step, int totalSteps, double loss,
                              double stageProgress, double overallProgress,
                              const QString& stageName);
    void optimizationFinished(const QString& outputDir);
    void optimizationError(const QString& message);

    /** Emitted after GET /datasets reply with the list of datasets. */
    void datasetsReceived(const QJsonArray& datasets);

private:
    explicit LasagnaServiceManager(QObject* parent = nullptr);
    ~LasagnaServiceManager() override;

    LasagnaServiceManager(const LasagnaServiceManager&) = delete;
    LasagnaServiceManager& operator=(const LasagnaServiceManager&) = delete;

    /** Construct base URL from current _host and _port. */
    QString baseUrl() const;

    bool startService(const QString& pythonPath);
    void handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void handleProcessError(QProcess::ProcessError error);
    void handleReadyReadStdout();
    void handleReadyReadStderr();

    void pollStatus();
    void handleStatusReply(QNetworkReply* reply);
    void handleOptimizeReply(QNetworkReply* reply);

    /** Download results archive from service and unpack locally. */
    void downloadResults();

    std::unique_ptr<QProcess> _process;
    QNetworkAccessManager* _nam{nullptr};
    QTimer* _pollTimer{nullptr};

    QString _host{"127.0.0.1"};
    int _port{0};
    bool _isExternal{false};
    QString _lastError;
    bool _serviceReady{false};
    bool _optimizationRunning{false};
    QString _localOutputDir;  // where to unpack results
};
