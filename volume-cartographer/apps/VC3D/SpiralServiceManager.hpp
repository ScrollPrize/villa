#pragma once

#include <QJsonObject>
#include <QObject>
#include <QPointer>
#include <QProcess>
#include <QTimer>

class QNetworkAccessManager;
class QNetworkReply;

class SpiralServiceManager : public QObject
{
    Q_OBJECT
public:
    explicit SpiralServiceManager(QObject* parent = nullptr);
    ~SpiralServiceManager() override;

    void ensureStarted();
    void stopService();
    void resolveDataset(const QString& root);
    void loadSession(QJsonObject request);
    void runIterations(int iterations);
    void stopAfterIteration();
    void saveCheckpoint(const QString& path);
    void deleteSession();
    bool isReady() const { return _ready; }

signals:
    void serviceStateChanged(const QString& state);
    void datasetResolved(const QJsonObject& resolution);
    void sessionAccepted(const QJsonObject& inputPaths, qint64 sessionGeneration);
    void sessionStatusChanged(const QJsonObject& status);
    void previewAvailable(const QString& manifestPath, qint64 generation);
    void logMessage(const QString& message);
    void errorOccurred(const QString& message);

private:
    QString findPython() const;
    QString findService() const;
    void post(const QString& path, QJsonObject body,
              std::function<void(const QJsonObject&)> success = {});
    void get(const QString& path, std::function<void(const QJsonObject&)> success);
    void handleReply(QNetworkReply* reply, std::function<void(const QJsonObject&)> success);
    void pollStatus();
    QString commandId();

    QProcess* _process = nullptr;
    QNetworkAccessManager* _network = nullptr;
    QTimer* _poll = nullptr;
    QString _nonce;
    int _port = 0;
    bool _ready = false;
    quint64 _commandCounter = 0;
    qint64 _lastStatusGeneration = -1;
    qint64 _lastPreviewGeneration = -1;
};
