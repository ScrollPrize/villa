#pragma once

#include <QObject>
#include <QProcess>
#include <QString>

#include <nlohmann/json.hpp>

#include <memory>
#include <mutex>
#include <string>

// The Python fiber-follow tracing service (vesuvius
// neural_tracing.tracers.fibers.tracing.service) used by Line Annotation's
// fiber model: one process per checkpoint, started on first use from the
// Python environment VC3D was started from (or the configured interpreter),
// kept alive while VC3D runs, and stopped with it. Requests go over a unix
// socket, one connection each, so they may be sent from solver threads.
class FiberFollowService : public QObject
{
    Q_OBJECT

public:
    static FiberFollowService& instance();

    // GUI thread. Starts (or restarts, for another checkpoint or interpreter)
    // the service and waits until it listens, keeping the UI responsive.
    // An empty checkpoint uses the service's default Hugging Face model.
    // Returns false with *error set on failure.
    bool ensureRunning(const QString& checkpoint, const QString& python, QString* error = nullptr);
    void stop();
    [[nodiscard]] bool isRunning() const;
    [[nodiscard]] QString checkpoint() const { return _checkpoint; }
    [[nodiscard]] QString python() const { return _python; }

    // Any thread, while running: one request (see service.py). Throws
    // std::runtime_error on transport failure or an {"ok": false} reply.
    [[nodiscard]] nlohmann::json request(const nlohmann::json& request) const;
    [[nodiscard]] std::string socketPath() const;

signals:
    void statusMessage(const QString& message);

private:
    explicit FiberFollowService(QObject* parent = nullptr);
    ~FiberFollowService() override;

    std::unique_ptr<QProcess> _process;
    QString _checkpoint;
    QString _python;
    QString _output;
    mutable std::mutex _mutex;  // guards _socketPath for solver threads
    std::string _socketPath;
};
