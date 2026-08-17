#pragma once

#include <QObject>
#include <QProcess>
#include <QString>

#include <functional>

class QTimer;

// VC3D-managed SSH local port forward to a persistent loopback Spiral service.
// The system OpenSSH client is used so ~/.ssh/config aliases, agents, and jump
// hosts work unchanged. Attach-only: this never starts anything on the host.
class SpiralSshTunnel : public QObject
{
    Q_OBJECT
public:
    enum class FailureKind { MissingSshClient, HostKeyVerification, Authentication, Other };

    explicit SpiralSshTunnel(QObject* parent = nullptr);
    ~SpiralSshTunnel() override;

    void start(const QString& destination, int remoteServicePort);
    void stop();
    bool isRunning() const;
    int localPort() const { return _localPort; }

    // One-shot `ssh destination cat <path>` used to retrieve the service's
    // auto-generated API key file so the user never copies a credential.
    void readRemoteFile(const QString& destination, const QString& remotePath,
                        std::function<void(const QString& contents, const QString& error)> done);

signals:
    void ready(int localPort);
    void failed(SpiralSshTunnel::FailureKind kind, const QString& message);
    void collapsed(const QString& message);
    void logMessage(const QString& message);

private:
    static QString sshExecutable();
    static FailureKind classifyFailure(const QString& stderrText);
    static int pickFreeLocalPort();
    void probeReadiness();

    QProcess* _process = nullptr;
    QTimer* _probeTimer = nullptr;
    QString _stderrText;
    int _localPort = 0;
    int _probeAttempts = 0;
    bool _announcedReady = false;
    bool _stopping = false;
};
