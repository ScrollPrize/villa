#pragma once

#include <QJsonObject>
#include <QObject>
#include <QProcess>
#include <QString>
#include <QTimer>
#include <memory>

class QNetworkAccessManager;
class QNetworkReply;

/**
 * Manages the lifecycle of the Python fit optimizer HTTP service.
 *
 * Launches fit_service.py as a subprocess, communicates via HTTP on
 * localhost.  The service is kept alive so repeated optimizations avoid
 * Python/torch startup overhead.
 */
class FitServiceManager : public QObject
{
    Q_OBJECT

public:
    static FitServiceManager& instance();

    /**
     * Ensure the service process is running.
     * @param pythonPath  Path to python executable (empty = auto-detect).
     * @return true if service is running (or was successfully started).
     */
    bool ensureServiceRunning(const QString& pythonPath = QString());

    /** Stop the service process. */
    void stopService();

    [[nodiscard]] bool isRunning() const;
    [[nodiscard]] QString lastError() const { return _lastError; }
    [[nodiscard]] int port() const { return _port; }

    /**
     * Submit an optimization job.
     * @param config  JSON body for POST /optimize  (model_input, output_dir, config).
     */
    void startOptimization(const QJsonObject& config);

    /** Request cancellation of the running optimization. */
    void stopOptimization();

signals:
    void serviceStarted();
    void serviceStopped();
    void serviceError(const QString& message);
    void statusMessage(const QString& message);

    void optimizationStarted();
    void optimizationProgress(const QString& stage, int step, int totalSteps, double loss);
    void optimizationFinished(const QString& outputDir);
    void optimizationError(const QString& message);

private:
    explicit FitServiceManager(QObject* parent = nullptr);
    ~FitServiceManager() override;

    FitServiceManager(const FitServiceManager&) = delete;
    FitServiceManager& operator=(const FitServiceManager&) = delete;

    bool startService(const QString& pythonPath);
    void handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void handleProcessError(QProcess::ProcessError error);
    void handleReadyReadStdout();
    void handleReadyReadStderr();

    void pollStatus();
    void handleStatusReply(QNetworkReply* reply);
    void handleOptimizeReply(QNetworkReply* reply);

    std::unique_ptr<QProcess> _process;
    QNetworkAccessManager* _nam{nullptr};
    QTimer* _pollTimer{nullptr};

    int _port{0};
    QString _lastError;
    bool _serviceReady{false};
    bool _optimizationRunning{false};
};
