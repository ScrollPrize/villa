#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QList>
#include <QObject>
#include <QProcess>
#include <QHash>
#include <QSet>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <functional>
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
     * Create an isolated manager for a short-lived, locally owned service.
     *
     * Unlike instance(), this manager is parent-owned and contains the local
     * Python process tree so destroying or stopping it cannot leave workers
     * behind. External services are still only disconnected, never killed.
     */
    static LasagnaServiceManager* createTransient(QObject* parent);

    /**
     * Ensure the service process is running (internal mode).
     * @param pythonPath  Path to python executable (empty = auto-detect).
     * @param dataDirectory Directory containing one or more .lasagna.json
     *        dataset descriptors. Required when starting an internal service.
     * @return true if service is running (or was successfully started).
     */
    bool ensureServiceRunning(const QString& pythonPath = QString(),
                              const QString& dataDirectory = QString());

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
    void submitOptimization(const QJsonObject& config,
                            const QString& localOutputDir = QString())
    {
        startOptimization(config, localOutputDir);
    }

    /** Request cancellation of the running optimization. */
    void stopOptimization();
    void cancelJob(const QString& jobId);
    void moveJobBefore(const QString& jobId, const QString& beforeJobId);
    void moveJobToEnd(const QString& jobId);
    void fetchJobs();

    /**
     * Export multi-layer OBJ visualization.
     * Synchronous POST to /export_vis; emits visExportFinished or visExportError.
     */
    void exportLasagnaVis(const QJsonObject& config);

    /**
     * Scan ~/.fit_services for running service .json files.
     * Stale entries (dead PIDs) are removed.
     */
    static QJsonArray discoverServices();

    /** Locate a config beside the Lasagna service checkout. */
    static QString findConfigFile(const QString& fileName);

    /** Describe a local tifxyz directory for the existing artifact-upload path. */
    static QJsonObject makeTifxyzArtifactUpload(const QString& tifxyzDirectory);

    /** Fetch available datasets from the connected service (GET /datasets). */
    void fetchDatasets();

    /**
     * Queue Laplace snap-ranking jobs through the fit service.
     * Completion callbacks are invoked on the Qt event thread.
     */
    void rankLaplaceSnapPairs(
        const QJsonObject& request,
        std::function<void(const QJsonObject&)> onSuccess,
        std::function<void(const QString&)> onError,
        std::function<void(int, const QJsonObject&)> onPartialResult = {});

signals:
    void serviceStarted();
    void serviceStopped();
    void serviceError(const QString& message);
    void statusMessage(const QString& message);

    void optimizationStarted();
    void optimizationProgress(const QString& stage, int step, int totalSteps, double loss,
                              double stageProgress, double overallProgress,
                              const QString& stageName);
    void artifactUploadProgress(const QString& jobId, int current, int total, double progress,
                                const QString& label);
    void optimizationFinished(const QString& outputDir);
    void resultsPlaced(const QString& outputDir, const QStringList& segmentNames);
    void optimizationError(const QString& message);
    void jobsUpdated(const QJsonArray& jobs);
    void jobStarted(const QString& jobId);
    void jobFinished(const QString& jobId, const QString& outputDir);
    void jobError(const QString& jobId, const QString& message);

    void visExportFinished(const QString& outputDir);
    void visExportError(const QString& message);

    /** Emitted after GET /datasets reply with the list of datasets. */
    void datasetsReceived(const QJsonArray& datasets);

private:
    explicit LasagnaServiceManager(QObject* parent = nullptr,
                                   bool containProcessTree = false);
    ~LasagnaServiceManager() override;

    LasagnaServiceManager(const LasagnaServiceManager&) = delete;
    LasagnaServiceManager& operator=(const LasagnaServiceManager&) = delete;

    /** Construct base URL from current _host and _port. */
    QString baseUrl() const;

    bool startService(const QString& pythonPath, const QString& dataDirectory);
    void handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void handleProcessError(QProcess::ProcessError error);
    void handleReadyReadStdout();
    void handleReadyReadStderr();

    void pollStatus();
    void handleStatusReply(QNetworkReply* reply);
    void handleJobsReply(QNetworkReply* reply);
    void handleOptimizeReply(QNetworkReply* reply, const QString& localUploadJobId = QString());
    bool validateApiVersion(QNetworkReply* reply, const QString& context);
    void postOptimizationRequest(const QJsonObject& requestConfig,
                                 const QString& localOutputDir,
                                 const QString& localUploadJobId = QString());
    void enqueueArtifactUpload(const QJsonObject& requestConfig,
                               const QJsonArray& objects,
                               const QString& localOutputDir,
                               const QString& source);
    void processNextArtifactUpload();
    void finishActiveArtifactUpload();
    bool cancelLocalUploadJob(const QString& jobId);
    void clearLocalUploadJobs();
    void upsertLocalUploadJob(const QJsonObject& job);
    void updateLocalUploadJob(const QString& jobId, const QJsonObject& updates);
    void removeLocalUploadJob(const QString& jobId);
    void updateCachedJobFromStatus(const QJsonObject& status);
    QJsonArray jobsWithLocalUploads(const QJsonArray& serviceJobs) const;
    void emitJobsUpdatedOverlay();

    /** Download results archive from service and unpack locally. */
    void downloadResults(const QString& jobId = QString(),
                         const QString& outputDir = QString());
    QString localSourceName() const;

    std::unique_ptr<QProcess> _process;
    QNetworkAccessManager* _nam{nullptr};
    QTimer* _pollTimer{nullptr};
    bool _containProcessTree{false};
#ifdef Q_OS_WIN
    void* _processJob{nullptr};
#endif

    struct ArtifactUploadJob
    {
        QString jobId;
        QJsonObject requestConfig;
        QJsonArray objects;
        QString localOutputDir;
        QString source;
    };

    QString _host{"127.0.0.1"};
    int _port{0};
    bool _isExternal{false};
    QString _lastError;
    QString _dataDirectory;
    bool _serviceReady{false};
    bool _optimizationRunning{false};
    QString _localOutputDir;  // where to unpack optimization results
    QString _visOutputDir;    // where to unpack vis export results
    QString _activeJobId;
    QSet<QString> _submittedJobIds;
    QSet<QString> _startedJobIds;
    QSet<QString> _completedJobIds;
    QHash<QString, QString> _jobOutputDirs;
    QHash<QString, QString> _jobOutputNames;
    QJsonArray _lastJobs;
    QList<ArtifactUploadJob> _artifactUploadQueue;
    ArtifactUploadJob _activeArtifactUpload;
    bool _hasActiveArtifactUpload{false};
    QNetworkReply* _activeArtifactReply{nullptr};
    QString _activeArtifactReplyJobId;
    QSet<QString> _cancelledLocalUploadJobs;
    QHash<QString, QJsonObject> _localUploadJobs;
    QStringList _localUploadOrder;
    quint64 _nextLocalUploadId{1};
    qint64 _lastQueueGeneration{-1};
    qint64 _fetchedQueueGeneration{-1};
    quint64 _requestGeneration{0};
    bool _statusRequestInFlight{false};
    bool _jobsRequestInFlight{false};
    bool _jobsRequestPending{false};
};
