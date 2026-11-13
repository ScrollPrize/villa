#include "ToolProcess.hpp"

#include "ToolExecutable.hpp"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>
#include <QTimer>

#include <algorithm>

namespace vc3d {
namespace {
constexpr int kTailBytes = 8192;
constexpr int kKillGraceMs = 3000;

QString quoteArg(const QString& arg)
{
    const bool needsQuotes = std::any_of(arg.begin(), arg.end(), [](QChar ch) {
        return ch.isSpace() || ch == '"' || ch == '\'' || ch == '\\';
    });
    if (!needsQuotes) return arg;
    QString escaped = arg;
    escaped.replace('\\', "\\\\");
    escaped.replace('"', "\\\"");
    return "\"" + escaped + "\"";
}

// The Linux AppImage bundles glibc and runs VC3D through the bundled loader
// (scripts/appimage/AppRun). A bundled tool started directly is loaded by the
// host's loader, which then picks up the bundled libc through the tool's
// RUNPATH and fails unless the host glibc is the same version. Start it
// through the bundled loader, as AppRun does.
void useBundledLoader(QString& program, QStringList& args)
{
    const QString appDir = qEnvironmentVariable("APPDIR");
    if (appDir.isEmpty()) return;
    const QDir usr(QDir(appDir).filePath(QStringLiteral("usr")));
    const QString libDir = usr.filePath(QStringLiteral("lib"));
    const QString loader = QDir(libDir).filePath(QStringLiteral("ld-linux-x86-64.so.2"));
    const QString binDir = QFileInfo(usr.filePath(QStringLiteral("bin"))).canonicalFilePath();
    if (binDir.isEmpty() || !QFileInfo(loader).isFile() || QFileInfo(program).canonicalPath() != binDir) return;
    args.prepend(program);
    args.prepend(libDir);
    args.prepend(QStringLiteral("--library-path"));
    program = loader;
}
}

ToolProcess::ToolProcess(QObject* parent) : QObject(parent) {}

QStringList ToolProcess::arguments(const Launch& launch)
{
    QStringList args = launch.args;
    if (!launch.store.isEmpty()) args << QStringLiteral("--store") << launch.store;
    if (!launch.volume.isEmpty()) {
        args << QStringLiteral("--volume-uuid") << launch.volume;
        if (launch.voxelScale != 1)
            args << QStringLiteral("--voxel-scale") << QString::number(launch.voxelScale, 'g', 17);
    }
    return args;
}

ToolProcess::~ToolProcess()
{
    if (process_ && process_->state() != QProcess::NotRunning) {
        process_->disconnect(this);
        process_->kill();
        process_->waitForFinished(kKillGraceMs);
    }
}

bool ToolProcess::start(const Launch& launch)
{
    const QFileInfo given(launch.tool);
    executable_ = given.isAbsolute() || launch.tool.contains('/') ? launch.tool : findToolExecutable(launch.tool);
    if (executable_.isEmpty() || !QFileInfo(executable_).isExecutable()) {
        error_ = tr("Tool executable not found or not executable: %1").arg(executable_.isEmpty() ? launch.tool : executable_);
        return false;
    }
    QString program = executable_;
    QStringList args = arguments(launch);
    useBundledLoader(program, args);
    if (launch.lowPriority) {
#if defined(Q_OS_UNIX)
        if (const auto nice = QStandardPaths::findExecutable(QStringLiteral("nice")); !nice.isEmpty()) {
            QStringList wrapped{QStringLiteral("-n"), QStringLiteral("19")};
#if defined(Q_OS_LINUX)
            if (const auto ionice = QStandardPaths::findExecutable(QStringLiteral("ionice")); !ionice.isEmpty())
                wrapped << ionice << QStringLiteral("-c") << QStringLiteral("3");
#endif
            args = wrapped + QStringList{program} + args;
            program = nice;
        }
#endif
    }
    QStringList quoted;
    for (const auto& arg : args) quoted << quoteArg(arg);
    command_ = QStringLiteral("%1 %2").arg(program, quoted.join(' '));
    if (launch.ompThreads > 0) command_ = QStringLiteral("OMP_NUM_THREADS=%1 %2").arg(launch.ompThreads).arg(command_);

    if (!launch.logPath.isEmpty()) {
        logFile_ = std::make_unique<QFile>(launch.logPath);
        if (logFile_->open(QIODevice::WriteOnly | QIODevice::Text)) {
            logStream_ = std::make_unique<QTextStream>(logFile_.get());
        } else {
            logFile_.reset();
        }
    }
    log(QStringLiteral("Tool: %1\nStarted: %2\nCommand: %3\n").arg(
        executable_, QDateTime::currentDateTime().toString(Qt::ISODate), command_));

    auto environment = QProcessEnvironment::systemEnvironment();
    if (launch.ompThreads > 0) environment.insert(QStringLiteral("OMP_NUM_THREADS"), QString::number(launch.ompThreads));
    for (const auto& key : launch.environment.keys()) {
        environment.insert(key, launch.environment.value(key));
        log(QStringLiteral("ENV: %1=%2\n").arg(key, launch.secretVariables.contains(key) ? QStringLiteral("<set>")
                                                                                           : launch.environment.value(key)));
    }
    log(QStringLiteral("===================================\n\n"));

    process_ = new QProcess(this);
    process_->setProcessChannelMode(QProcess::MergedChannels);
    process_->setProcessEnvironment(environment);
    if (!launch.workingDirectory.isEmpty()) process_->setWorkingDirectory(launch.workingDirectory);
    connect(process_, &QProcess::started, this, &ToolProcess::started);
    connect(process_, &QProcess::readyRead, this, &ToolProcess::readOutput);
    connect(process_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
            [this](int exitCode, QProcess::ExitStatus status) {
                readOutput();
                handleLines(lines_.finish());
                Result result;
                result.exitCode = exitCode;
                result.cancelled = cancelled_;
                result.crashed = status != QProcess::NormalExit;
                if (!cancelled_) {
                    if (result.crashed) result.error = tr("%1 crashed").arg(QFileInfo(executable_).baseName());
                    else if (exitCode != 0)
                        result.error = tr("%1 failed with exit code %2").arg(QFileInfo(executable_).baseName()).arg(exitCode);
                    result.ok = result.error.isEmpty();
                }
                finish(std::move(result));
            });
    // Qt emits no finished() for a process that never started.
    connect(process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (error != QProcess::FailedToStart) return;
        Result result;
        result.failedToStart = true;
        result.error = tr("%1 could not be started: %2").arg(executable_, process_->errorString());
        finish(std::move(result));
    });
    process_->start(program, args);
    return true;
}

void ToolProcess::cancel()
{
    if (!running()) return;
    cancelled_ = true;
    process_->terminate();
    QTimer::singleShot(kKillGraceMs, process_, [process = process_] {
        if (process->state() != QProcess::NotRunning) process->kill();
    });
}

bool ToolProcess::running() const
{
    return process_ && process_->state() != QProcess::NotRunning;
}

void ToolProcess::readOutput()
{
    const QString chunk = QString::fromUtf8(process_->readAll());
    if (chunk.isEmpty()) return;
    tail_ = (tail_ + chunk).right(kTailBytes);
    log(chunk);
    emit output(chunk);
    handleLines(lines_.add(chunk));
}

void ToolProcess::handleLines(const QStringList& lines)
{
    for (const auto& line : lines) {
        if (const auto uuid = reportedSurfaceUuid(line); !uuid.isEmpty()) surfaces_ << uuid;
        emit this->line(line);
    }
}

void ToolProcess::finish(Result result)
{
    if (finished_) return;
    finished_ = true;
    result.surfaces = surfaces_;
    result.tail = tail_.trimmed();
    log(QStringLiteral("\n===================================\nFinished: %1\nExit code: %2\n%3")
            .arg(QDateTime::currentDateTime().toString(Qt::ISODate))
            .arg(result.exitCode)
            .arg(result.cancelled ? QStringLiteral("Cancelled\n") : result.error.isEmpty() ? QString() : result.error + "\n"));
    logStream_.reset();
    logFile_.reset();
    emit finished(result);
}

void ToolProcess::log(const QString& text)
{
    if (!logStream_) return;
    *logStream_ << text;
    logStream_->flush();
}

} // namespace vc3d
