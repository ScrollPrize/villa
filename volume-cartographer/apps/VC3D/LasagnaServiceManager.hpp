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
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>

class QNetworkAccessManager;
class QNetworkReply;

/**
 * Manages the lifecycle of the Python lasagna HTTP service.
 *
 * Supports two modes:
 *   - Internal: launches fit_service.py as a subprocess on the attached
 *     object store's file (--store --shared-store; this process's session
 *     syncs it)
 *   - External: connects to a pre-started service on a known host:port; it
 *     must work on the same store (a replica of the same hub)
 *
 * Jobs read stored surfaces by UUID and create their results as new surfaces
 * in the store (results attached to them, the model checkpoint in their
 * artifact directory); nothing travels as files.
 *
 * The service is kept alive so repeated optimizations avoid
 * Python/torch startup overhead.
 */
class LasagnaServiceManager : public QObject
{
    Q_OBJECT

public:
    // Roles of the result documents a fit attaches to its surfaces.
    static constexpr const char* kCorrResultsRole = "lasagna_corr_points_results";

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

    /** The store jobs read and write (the surface catalog's attachment). */
    void setObjectStore(std::shared_ptr<vc::objects::ObjectStoreSession> session,
                        std::optional<vc::objects::AttachmentIdentity> attachment);
    /** The model checkpoint a fit left with a stored surface, if any. */
    std::optional<std::filesystem::path> surfaceModel(const std::string& surfaceUuid) const;

    [[nodiscard]] bool isRunning() const;
    [[nodiscard]] QString lastError() const { return _lastError; }
    [[nodiscard]] int port() const { return _port; }
    [[nodiscard]] QString host() const { return _host; }
    [[nodiscard]] bool isExternal() const { return _isExternal; }

    /**
     * Submit an optimization job on the attached store.
     * @param config  JSON body for POST /jobs (surface refs, volume_uuid);
     *                the store's UUID is added here.
     */
    void startOptimization(const QJsonObject& config);

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
    // The surfaces a job created (UUIDs in the attached store).
    void optimizationFinished(const QStringList& surfaceUuids);
    void optimizationError(const QString& message);
    void jobsUpdated(const QJsonArray& jobs);
    void jobStarted(const QString& jobId);
    void jobFinished(const QString& jobId, const QStringList& surfaceUuids);
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
    void handleOptimizeReply(QNetworkReply* reply);
    bool validateApiVersion(QNetworkReply* reply, const QString& context);
    void postOptimizationRequest(const QJsonObject& requestConfig);
    /** Run `then` once the store's hub has this replica's edits (an external
     *  service reads the hub); right away for the internal service. */
    void afterStoreSync(std::function<void()> then);
    QString storeError() const;
    void updateCachedJobFromStatus(const QJsonObject& status);

    QString localSourceName() const;

    std::unique_ptr<QProcess> _process;
    QNetworkAccessManager* _nam{nullptr};
    QTimer* _pollTimer{nullptr};
    bool _containProcessTree{false};
#ifdef Q_OS_WIN
    void* _processJob{nullptr};
#endif

    QString _host{"127.0.0.1"};
    int _port{0};
    bool _isExternal{false};
    QString _lastError;
    QString _dataDirectory;
    // Not owned: the session belongs to the window's state and must not outlive
    // it (this manager is a process-lifetime singleton).
    std::weak_ptr<vc::objects::ObjectStoreSession> _storeSession;
    std::optional<vc::objects::AttachmentIdentity> _storeAttachment;
    QString _storePath;         // the attached store's file
    QString _serviceStorePath;  // the file the internal service was started on
    QString _serviceStoreUuid;  // the store the connected service writes to
    bool _serviceReady{false};
    bool _optimizationRunning{false};
    QString _visOutputDir;    // where to unpack vis export results
    QString _activeJobId;
    QSet<QString> _submittedJobIds;
    QSet<QString> _startedJobIds;
    QSet<QString> _completedJobIds;
    QJsonArray _lastJobs;
    qint64 _lastQueueGeneration{-1};
    qint64 _fetchedQueueGeneration{-1};
    quint64 _requestGeneration{0};
    bool _statusRequestInFlight{false};
    bool _jobsRequestInFlight{false};
    bool _jobsRequestPending{false};
};
