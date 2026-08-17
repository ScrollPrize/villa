#include "SpiralSshTunnel.hpp"

#include <QStandardPaths>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTimer>

namespace {
constexpr int kProbeIntervalMs = 250;
constexpr int kMaxProbeAttempts = 120; // 30 seconds to authenticate and forward
}

SpiralSshTunnel::SpiralSshTunnel(QObject* parent) : QObject(parent)
{
    _probeTimer = new QTimer(this);
    _probeTimer->setInterval(kProbeIntervalMs);
    connect(_probeTimer, &QTimer::timeout, this, &SpiralSshTunnel::probeReadiness);
}

SpiralSshTunnel::~SpiralSshTunnel() { stop(); }

QString SpiralSshTunnel::sshExecutable()
{
    // On Windows the system OpenSSH client (ssh.exe) is the supported client.
    return QStandardPaths::findExecutable(QStringLiteral("ssh"));
}

SpiralSshTunnel::FailureKind SpiralSshTunnel::classifyFailure(const QString& stderrText)
{
    // Host key verification is a separate failure mode from authentication:
    // with BatchMode=yes and no TTY an unknown host key fails instead of
    // prompting, and the remedy is one interactive `ssh <destination>`.
    if (stderrText.contains(QStringLiteral("Host key verification failed"))
        || stderrText.contains(QStringLiteral("REMOTE HOST IDENTIFICATION HAS CHANGED"))
        || stderrText.contains(QStringLiteral("No ED25519 host key is known"))
        || stderrText.contains(QStringLiteral("host key for")))
        return FailureKind::HostKeyVerification;
    if (stderrText.contains(QStringLiteral("Permission denied"))
        || stderrText.contains(QStringLiteral("Too many authentication failures")))
        return FailureKind::Authentication;
    return FailureKind::Other;
}

int SpiralSshTunnel::pickFreeLocalPort()
{
    QTcpServer server;
    if (!server.listen(QHostAddress::LocalHost, 0)) return 0;
    const int port = server.serverPort();
    server.close();
    return port;
}

bool SpiralSshTunnel::isRunning() const
{
    return _process && _process->state() != QProcess::NotRunning;
}

void SpiralSshTunnel::start(const QString& destination, int remoteServicePort)
{
    stop();
    const QString ssh = sshExecutable();
    if (ssh.isEmpty()) {
        emit failed(FailureKind::MissingSshClient,
                    tr("No OpenSSH client (ssh) was found on this computer. Install "
                       "OpenSSH to use an SSH Spiral profile."));
        return;
    }
    _localPort = pickFreeLocalPort();
    if (_localPort == 0) {
        emit failed(FailureKind::Other, tr("Could not allocate a free local port for the SSH tunnel"));
        return;
    }
    _stderrText.clear();
    _announcedReady = false;
    _stopping = false;
    _probeAttempts = 0;

    _process = new QProcess(this);
    connect(_process, &QProcess::readyReadStandardError, this, [this]() {
        const QString text = QString::fromUtf8(_process->readAllStandardError());
        _stderrText += text;
        for (const QString& line : text.split('\n', Qt::SkipEmptyParts))
            emit logMessage(QStringLiteral("ssh: ") + line);
    });
    connect(_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (_stopping) return;
        if (error == QProcess::FailedToStart) {
            _probeTimer->stop();
            emit failed(FailureKind::MissingSshClient,
                        tr("The ssh client failed to start: %1").arg(_process->errorString()));
        }
    });
    connect(_process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this](int code, QProcess::ExitStatus) {
                _probeTimer->stop();
                if (_stopping) return;
                const QString detail = _stderrText.trimmed();
                if (_announcedReady) {
                    emit collapsed(tr("The SSH tunnel closed (exit code %1). %2").arg(code).arg(detail));
                    return;
                }
                const FailureKind kind = classifyFailure(detail);
                QString message;
                switch (kind) {
                case FailureKind::HostKeyVerification:
                    message = tr("SSH does not trust this host key yet. Run `ssh %1` once in "
                                 "a terminal to verify and accept the host key, then reconnect.\n%2")
                                  .arg(_process->program().isEmpty() ? QString() : _process->arguments().last(), detail);
                    break;
                case FailureKind::Authentication:
                    message = tr("SSH authentication failed. Spiral SSH profiles require "
                                 "non-interactive authentication (keys or an agent).\n%1").arg(detail);
                    break;
                default:
                    message = tr("The SSH tunnel exited with code %1.\n%2").arg(code).arg(detail);
                    break;
                }
                emit failed(kind, message);
            });

    const QStringList arguments{
        QStringLiteral("-N"),
        QStringLiteral("-o"), QStringLiteral("BatchMode=yes"),
        QStringLiteral("-o"), QStringLiteral("ServerAliveInterval=15"),
        QStringLiteral("-o"), QStringLiteral("ServerAliveCountMax=3"),
        QStringLiteral("-o"), QStringLiteral("ExitOnForwardFailure=yes"),
        QStringLiteral("-L"),
        QStringLiteral("%1:127.0.0.1:%2").arg(_localPort).arg(remoteServicePort),
        destination,
    };
    emit logMessage(tr("Starting SSH tunnel: ssh %1").arg(arguments.join(QLatin1Char(' '))));
    _process->start(ssh, arguments);
    _probeTimer->start();
}

void SpiralSshTunnel::probeReadiness()
{
    if (!isRunning()) { _probeTimer->stop(); return; }
    if (++_probeAttempts > kMaxProbeAttempts) {
        _probeTimer->stop();
        emit failed(FailureKind::Other, tr("Timed out waiting for the SSH tunnel to come up"));
        stop();
        return;
    }
    auto* socket = new QTcpSocket(this);
    connect(socket, &QTcpSocket::connected, this, [this, socket]() {
        socket->disconnectFromHost();
        socket->deleteLater();
        if (!_announcedReady) {
            _announcedReady = true;
            _probeTimer->stop();
            emit ready(_localPort);
        }
    });
    connect(socket, &QTcpSocket::errorOccurred, socket, [socket](QAbstractSocket::SocketError) {
        socket->deleteLater();
    });
    socket->connectToHost(QStringLiteral("127.0.0.1"), static_cast<quint16>(_localPort));
}

void SpiralSshTunnel::stop()
{
    _probeTimer->stop();
    if (!_process) return;
    _stopping = true;
    if (_process->state() != QProcess::NotRunning) {
        _process->terminate();
        if (!_process->waitForFinished(2000)) {
            _process->kill();
            _process->waitForFinished(1000);
        }
    }
    _process->deleteLater();
    _process = nullptr;
    _announcedReady = false;
    _localPort = 0;
}

void SpiralSshTunnel::readRemoteFile(const QString& destination, const QString& remotePath,
                                     std::function<void(const QString&, const QString&)> done)
{
    const QString ssh = sshExecutable();
    if (ssh.isEmpty()) {
        done({}, tr("No OpenSSH client (ssh) was found on this computer."));
        return;
    }
    auto* process = new QProcess(this);
    connect(process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [process, done, remotePath](int code, QProcess::ExitStatus) {
                const QString output = QString::fromUtf8(process->readAllStandardOutput()).trimmed();
                const QString errors = QString::fromUtf8(process->readAllStandardError()).trimmed();
                process->deleteLater();
                if (code != 0 || output.isEmpty()) {
                    const FailureKind kind = classifyFailure(errors);
                    QString message = errors.isEmpty()
                        ? tr("Could not read %1 on the service host").arg(remotePath)
                        : errors;
                    if (kind == FailureKind::HostKeyVerification)
                        message = tr("SSH does not trust this host key yet. Run `ssh <destination>` "
                                     "once in a terminal to accept the host key.\n%1").arg(errors);
                    done({}, message);
                    return;
                }
                done(output, {});
            });
    process->start(ssh, {QStringLiteral("-o"), QStringLiteral("BatchMode=yes"),
                         destination, QStringLiteral("cat"), remotePath});
}
