#pragma once

#include "ToolOutput.hpp"

#include <QFile>
#include <QObject>
#include <QProcess>
#include <QProcessEnvironment>
#include <QString>
#include <QStringList>
#include <QTextStream>

#include <memory>

namespace vc3d {

// One run of a command-line tool: VC3D's only way of launching one. Resolves
// the executable, applies the run's environment, logs the run, splits its
// merged output into lines, collects the surfaces it reports, and delivers
// exactly one finished() however the run ends (failed start, crash, exit,
// cancel).
class ToolProcess : public QObject {
    Q_OBJECT
public:
    struct Launch {
        // A tool name (vc3d::findToolExecutable) or an executable path.
        QString tool;
        QStringList args;
        // The store contract: `--store`, `--volume-uuid` and (when not 1)
        // `--voxel-scale` follow `args` when set (tools take options anywhere;
        // a script's path stays first).
        QString store;
        QString volume;
        double voxelScale = 1;
        // OMP_NUM_THREADS for the run, when positive.
        int ompThreads = -1;
        // Variables added to the system environment.
        QProcessEnvironment environment;
        // Names of variables in `environment` logged as "<set>" (credentials).
        QStringList secretVariables;
        // The run's log file (header, output, exit); none when empty.
        QString logPath;
        QString workingDirectory;
        // Run at the lowest CPU (nice) and, on Linux, I/O (ionice) priority
        // when those wrappers exist.
        bool lowPriority = false;
    };
    struct Result {
        bool ok = false;
        bool cancelled = false;
        bool failedToStart = false;
        bool crashed = false;
        int exitCode = -1;
        // Why it failed (start failure, crash, exit code); empty when ok or cancelled.
        QString error;
        // The `surface_uuid:` lines it printed, in order.
        QStringList surfaces;
        // The last lines of its output, for error reports.
        QString tail;
    };

    explicit ToolProcess(QObject* parent = nullptr);

    // The arguments a launch runs with: its own, then the store contract.
    static QStringList arguments(const Launch& launch);
    ~ToolProcess() override;

    // False (and no finished()) when the executable is not found; error() says why.
    bool start(const Launch& launch);
    // Terminate, then kill after a grace period; finished() reports it cancelled.
    void cancel();
    bool running() const;
    // The executable and the command line as run (with OMP_NUM_THREADS).
    const QString& executable() const { return executable_; }
    const QString& command() const { return command_; }
    const QString& error() const { return error_; }
    // The surfaces reported so far.
    const QStringList& surfaces() const { return surfaces_; }

signals:
    void started();
    // Output as it arrives, and as complete lines.
    void output(const QString& chunk);
    void line(const QString& line);
    void finished(const vc3d::ToolProcess::Result& result);

private:
    void readOutput();
    void handleLines(const QStringList& lines);
    void finish(Result result);
    void log(const QString& text);

    QProcess* process_ = nullptr;
    QString executable_, command_, error_;
    QStringList surfaces_;
    QString tail_;
    ToolOutputLines lines_;
    std::unique_ptr<QFile> logFile_;
    std::unique_ptr<QTextStream> logStream_;
    bool cancelled_ = false;
    bool finished_ = false;
};

} // namespace vc3d
