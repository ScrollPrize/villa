#include "CWindow.hpp"
#include "CSurfaceCollection.hpp"
#include "SurfacePanelController.hpp"

#include <functional>
#include <algorithm>
#include <iostream>

#include <QSettings>
#include <QMessageBox>
#include <QProcess>
#include <QDir>
#include <QFileInfo>
#include <QCoreApplication>
#include <QDateTime>
#include <QJsonDocument>
#include <QJsonObject>
#include <QInputDialog>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QFile>
#include <QTextStream>
#include <QtGlobal>
#include <QProcessEnvironment>
#include <QProgressDialog>
#include <QPointer>
#include <QTimer>
#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
#include <QStandardPaths>
#endif

#include "CommandLineToolRunner.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "ToolDialogs.hpp"

// --------- local helpers for running external tools -------------------------
static bool runProcessBlocking(const QString& program,
                               const QStringList& args,
                               const QString& workDir,
                               QString* out=nullptr,
                               QString* err=nullptr)
{
    QProcess p;
    if (!workDir.isEmpty()) p.setWorkingDirectory(workDir);
    p.setProcessChannelMode(QProcess::SeparateChannels);

    std::cout << "Running: " << program.toStdString();
    for (const QString& arg : args) std::cout << " " << arg.toStdString();
    std::cout << std::endl;

    p.start(program, args);
    if (!p.waitForStarted()) { if (err) *err = QObject::tr("Failed to start %1").arg(program); return false; }
    if (!p.waitForFinished(-1)) { if (err) *err = QObject::tr("Timeout running %1").arg(program); return false; }
    if (out) *out = QString::fromLocal8Bit(p.readAllStandardOutput());
    if (err) *err = QString::fromLocal8Bit(p.readAllStandardError());
    return (p.exitStatus()==QProcess::NormalExit && p.exitCode()==0);
}

// --------- locate generic vc_* executables -----------------------------------
static QString findVcTool(const char* name)
{
    QSettings settings("VC.ini", QSettings::IniFormat);
    const QString key1 = QStringLiteral("tools/%1_path").arg(name);
    const QString key2 = QStringLiteral("tools/%1").arg(name);
    const QString iniPath =
        settings.value(key1, settings.value(key2)).toString().trimmed();
    if (!iniPath.isEmpty()) {
        QFileInfo fi(iniPath);
        if (fi.exists() && fi.isFile() && fi.isExecutable())
            return fi.absoluteFilePath();
    }

#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
    const QString onPath = QStandardPaths::findExecutable(QString::fromLatin1(name));
    if (!onPath.isEmpty()) return onPath;
#else
    const QStringList pathDirs =
        QProcessEnvironment::systemEnvironment().value("PATH")
            .split(QDir::listSeparator(), Qt::SkipEmptyParts);
    for (const QString& dir : pathDirs) {
        const QString candidate = QDir(dir).filePath(QString::fromLatin1(name));
        QFileInfo fi(candidate);
        if (fi.exists() && fi.isFile() && fi.isExecutable())
            return fi.absoluteFilePath();
    }
#endif
    return {};
}

namespace { // -------------------- anonymous namespace -------------------------

// Owns the lifecycle for the async SLIM run; deletes itself on finish/cancel
class SlimJob : public QObject {
public:
    SlimJob(CWindow* win,
            const QString& segDir,
            const QString& segmentStem,
            const QString& flatboiExe)
    : QObject(win)
    , w_(win)
    , segDir_(segDir)
    , stem_(segmentStem)
    , objPath_(QDir(segDir).filePath(segmentStem + ".obj"))
    , flatObj_(QDir(segDir).filePath(segmentStem + "_flatboi.obj"))
    , outFinal_(segDir.endsWith("_flatboi") ? segDir : (segDir + "_flatboi"))
    , outTemp_ (segDir.endsWith("_flatboi") ? (segDir + "__rebuild_tmp__") : outFinal_)
    , flatboiExe_(flatboiExe)
    , inputIsAlreadyFlat_(segDir.endsWith("_flatboi"))
    , proc_(new QProcess(this))
    , progress_(new QProgressDialog(QObject::tr("Preparing SLIM…"), QObject::tr("Cancel"), 0, 0, win))
    , itRe_(R"(^\s*\[it\s+(\d+)\])", QRegularExpression::CaseInsensitiveOption)
    , progRe_(R"(^\s*PROGRESS\s+(\d+)\s*/\s*(\d+)\s*$)", QRegularExpression::CaseInsensitiveOption)
    {
        QSettings s("VC.ini", QSettings::IniFormat);
        iters_ = s.value("tools/flatboi_iters", 20).toInt();
        if (iters_ <= 0) iters_ = 20;

        tifxyz2objExe_ = findVcTool("vc_tifxyz2obj");
        obj2tifxyzExe_ = findVcTool("vc_obj2tifxyz");

        // never create outTemp_ here; we'll let vc_obj2tifxyz create it later
        if (QFileInfo::exists(outTemp_)) {
            QDir(outTemp_).removeRecursively();
        }

        proc_->setWorkingDirectory(segDir_);
        proc_->setProcessChannelMode(QProcess::MergedChannels);

        progress_->setWindowModality(Qt::WindowModal);
        progress_->setAutoClose(false);
        progress_->setAutoReset(true);
        progress_->setMinimumDuration(0);
        progress_->setMaximum(1 + iters_ + 1);
        progress_->setValue(0);
        progress_->setAttribute(Qt::WA_DeleteOnClose);

        QObject::connect(progress_, &QProgressDialog::canceled,
                         this, &SlimJob::onCanceled_);
        QObject::connect(proc_, &QProcess::readyReadStandardOutput,
                         this, &SlimJob::onStdout_);
        QObject::connect(proc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
                         this, &SlimJob::onFinished_);
        QObject::connect(proc_, &QProcess::errorOccurred,
                         this, &SlimJob::onProcError_);

        w_->statusBar()->showMessage(QObject::tr("Converting TIFXYZ to OBJ…"), 0);
        startToObj_();
    }

private:
    enum class Phase { ToObj, Flatboi, ToTifxyz, Swap, Done };

    void startToObj_() {
        if (tifxyz2objExe_.isEmpty()) { showImmediateToolNotFound_("vc_tifxyz2obj"); return; }
        phase_ = Phase::ToObj;
        progress_->setLabelText(QObject::tr("Converting TIFXYZ → OBJ…"));
        progress_->setMaximum(1 + iters_ + 1);
        progress_->setValue(0);
        ioLog_.clear();
        QStringList args; args << segDir_ << objPath_;
        ioLog_ += QStringLiteral("Running: %1 %2\n").arg(tifxyz2objExe_, args.join(' '));
        proc_->start(tifxyz2objExe_, args);
    }

    void startFlatboi_() {
        phase_ = Phase::Flatboi;
        lastIterSeen_ = 0;
        progress_->setLabelText(QObject::tr("Running SLIM (flatboi)…"));
        progress_->setValue(1);
        ioLog_.clear();
        QStringList args; args << objPath_ << QString::number(iters_);
        ioLog_ += QStringLiteral("Running: %1 %2\n").arg(flatboiExe_, args.join(' '));
        proc_->start(flatboiExe_, args);
    }

    void startToTifxyz_() {
        if (obj2tifxyzExe_.isEmpty()) { showImmediateToolNotFound_("vc_obj2tifxyz"); return; }
        phase_ = Phase::ToTifxyz;
        progress_->setLabelText(QObject::tr("Converting flattened OBJ → TIFXYZ…"));
        progress_->setValue(1 + iters_);

        // IMPORTANT: vc_obj2tifxyz expects the target directory NOT to exist.
        if (QFileInfo::exists(outTemp_)) {
            ioLog_ += QStringLiteral("Removing existing output dir: %1\n").arg(outTemp_);
            if (!QDir(outTemp_).removeRecursively()) {
                QMessageBox::critical(w_, QObject::tr("Error"),
                                      QObject::tr("Output directory already exists and cannot be removed:\n%1")
                                      .arg(outTemp_));
                cleanupAndDelete_();
                return;
            }
        }

        // Ensure parent directory exists; vc_obj2tifxyz will create outTemp_ itself
        const QString parentPath = QFileInfo(outTemp_).absolutePath();
        QDir parent(parentPath);
        if (!parent.exists() && !parent.mkpath(".")) {
            QMessageBox::critical(w_, QObject::tr("Error"),
                                  QObject::tr("Cannot create parent directory: %1").arg(parentPath));
            cleanupAndDelete_();
            return;
        }

        ioLog_.clear();
        QStringList args; args << flatObj_ << outTemp_;
        ioLog_ += QStringLiteral("Running: %1 %2\n").arg(obj2tifxyzExe_, args.join(' '));
        proc_->start(obj2tifxyzExe_, args);
    }

    void finishSwapIfNeeded_() {
        if (inputIsAlreadyFlat_) {
            QDir orig(segDir_);
            orig.removeRecursively();

            const QFileInfo tmpInfo(outTemp_);
            QDir parent(tmpInfo.absolutePath());
            if (!parent.rename(tmpInfo.fileName(), QFileInfo(outFinal_).fileName())) {
                QMessageBox* warn = new QMessageBox(QMessageBox::Warning,
                    QObject::tr("Warning"),
                    QObject::tr("Rebuilt directory created, but failed to overwrite original.\n"
                                "Kept temporary at:\n%1").arg(outTemp_),
                    QMessageBox::Ok, w_);
                warn->setAttribute(Qt::WA_DeleteOnClose);
                warn->open();
            }
        }
    }

    void showDoneAndCleanup_() {
        if (progress_) {
            progress_->setValue(progress_->maximum());
            progress_->close();
        }

        QMessageBox* box = new QMessageBox(QMessageBox::Information,
                                           QObject::tr("SLIM-flatten"),
                                           QObject::tr("Flattened segment written to:\n%1").arg(outFinal_),
                                           QMessageBox::Ok, w_);
        box->setAttribute(Qt::WA_DeleteOnClose);
        QObject::connect(box, &QMessageBox::finished, this, [this]() {
            if (progress_) progress_->deleteLater();
            this->deleteLater();
        });
        box->open();
    }

    void cleanupAndDelete_() {
        if (QFileInfo::exists(outTemp_) && outTemp_ != outFinal_) {
            QDir(outTemp_).removeRecursively();
        }
        if (progress_) { progress_->close(); progress_->deleteLater(); }
        QTimer::singleShot(0, this, [this](){ this->deleteLater(); });
    }


    void onCanceled_() {
        if (proc_->state() != QProcess::NotRunning) {
            proc_->kill();
            proc_->waitForFinished(3000);
            
            // Ensure the process is actually terminated before proceeding
            if (proc_->state() != QProcess::NotRunning) {
                return; // Don't proceed with cleanup if process is still running
            }
        }
        if (QFileInfo::exists(outTemp_) && outTemp_ != outFinal_) {
            QDir(outTemp_).removeRecursively();
        }

        w_->statusBar()->showMessage(QObject::tr("SLIM-flatten cancelled"), 5000);
        progress_->close();
        progress_->deleteLater();
        QTimer::singleShot(0, this, [this](){ this->deleteLater(); });
    }

    void onStdout_() {
        const QString chunk = QString::fromLocal8Bit(proc_->readAllStandardOutput());
        ioLog_ += chunk;
        const QStringList lines = chunk.split('\n', Qt::SkipEmptyParts);
        for (const QString& raw : lines) {
            const QString line = raw.trimmed();

            if (phase_ == Phase::Flatboi) {
                if (auto m = progRe_.match(line); m.hasMatch()) {
                    const int cur = m.captured(1).toInt();
                    const int tot = m.captured(2).toInt();
                    if (tot > 0 && tot != iters_) {
                        iters_ = tot;
                        progress_->setMaximum(1 + iters_ + 1);
                    }
                    progress_->setLabelText(QObject::tr("SLIM iterations: %1 / %2").arg(cur).arg(iters_));
                    progress_->setValue(1 + std::max(0, std::min(cur, iters_)));
                    lastIterSeen_ = std::max(lastIterSeen_, cur);
                    continue;
                }
                if (auto m = itRe_.match(line); m.hasMatch()) {
                    const int n = m.captured(1).toInt();
                    lastIterSeen_ = std::max(lastIterSeen_, n);
                    progress_->setLabelText(QObject::tr("SLIM iterations: %1 / %2").arg(lastIterSeen_).arg(iters_));
                    progress_->setValue(1 + std::max(0, std::min(lastIterSeen_, iters_)));
                    continue;
                }
            }

            if (line.startsWith("Final stretch") || line.startsWith("Wrote:")) {
                w_->statusBar()->showMessage(line, 0);
            }
        }
    }

    void onProcError_(QProcess::ProcessError e) {
        if (errorShown_) return;
        errorShown_ = true;
        QString why;
        switch (e) {
            case QProcess::FailedToStart: why = QObject::tr("Program not found or not executable."); break;
            case QProcess::Crashed:       why = QObject::tr("Process crashed."); break;
            default:                      why = QObject::tr("Process error (%1).").arg(int(e)); break;
        }
        QString what;
        switch (phase_) {
            case Phase::ToObj:    what = QObject::tr("vc_tifxyz2obj failed to start."); break;
            case Phase::Flatboi:  what = QObject::tr("flatboi failed to start.");       break;
            case Phase::ToTifxyz: what = QObject::tr("vc_obj2tifxyz failed to start."); break;
            default: break;
        }
        QMessageBox* box = new QMessageBox(QMessageBox::Critical, QObject::tr("Error"),
                                           what + "\n\n" + ioLog_.trimmed() + "\n\n" + why,
                                           QMessageBox::Ok, w_);
        box->setAttribute(Qt::WA_DeleteOnClose);
        QObject::connect(box, &QMessageBox::finished, this, [this]() { cleanupAndDelete_(); });
        box->open();
        w_->statusBar()->showMessage(QObject::tr("SLIM-flatten failed"), 5000);
    }

    void onFinished_(int exitCode, QProcess::ExitStatus st) {
        if (errorShown_) return;

        // Error path
        if (st != QProcess::NormalExit || exitCode != 0) {
            const QString err = ioLog_.trimmed();
            QString what;
            switch (phase_) {
                case Phase::ToObj:    what = QObject::tr("vc_tifxyz2obj failed."); break;
                case Phase::Flatboi:  what = QObject::tr("flatboi failed.");       break;
                case Phase::ToTifxyz: what = QObject::tr("vc_obj2tifxyz failed."); break;
                default: break;
            }
            QMessageBox* box = new QMessageBox(QMessageBox::Critical, QObject::tr("Error"),
                                               what + (err.isEmpty()? QString() : ("\n\n" + err)),
                                               QMessageBox::Ok, w_);
            errorShown_ = true;  // Prevent duplicate error dialogs
            box->setAttribute(Qt::WA_DeleteOnClose);
            QObject::connect(box, &QMessageBox::finished, this, [this]() {
                if (QFileInfo::exists(outTemp_) && outTemp_ != outFinal_) {
                    QDir(outTemp_).removeRecursively();
                }
                if (progress_) { progress_->close(); progress_->deleteLater(); }
                this->deleteLater();
            });
            box->open();
            w_->statusBar()->showMessage(QObject::tr("SLIM-flatten failed"), 5000);
            return;
        }

        // Success: advance phases
        if (phase_ == Phase::ToObj) {
            if (!QFileInfo::exists(objPath_)) { onFinished_(1, QProcess::NormalExit); return; }
            if (progress_) progress_->setValue(1);
            startFlatboi_();
            return;
        }

        if (phase_ == Phase::Flatboi) {
            if (!QFileInfo::exists(flatObj_)) { onFinished_(1, QProcess::NormalExit); return; }
            startToTifxyz_();
            return;
        }

        if (phase_ == Phase::ToTifxyz) {
            if (!QFileInfo::exists(outTemp_) || !QFileInfo(outTemp_).isDir()) {
                onFinished_(1, QProcess::NormalExit); return;
            }
            phase_ = Phase::Swap;
            finishSwapIfNeeded_();
            phase_ = Phase::Done;

            w_->statusBar()->showMessage(QObject::tr("SLIM-flatten complete: %1").arg(outFinal_), 5000);
            showDoneAndCleanup_();
            return;
        }
    }

    static void removeDirIfExists_(const QString& p){
        if (QFileInfo::exists(p)) { QDir d(p); d.removeRecursively(); }
    }

private:
    CWindow* w_ = nullptr;

    // paths & flags
    QString segDir_;
    QString stem_;
    QString objPath_;
    QString flatObj_;
    QString outFinal_;
    QString outTemp_;
    QString flatboiExe_;
    bool    inputIsAlreadyFlat_ = false;

    // process & progress
    QProcess* proc_ = nullptr;
    QPointer<QProgressDialog> progress_;
    Phase   phase_ = Phase::ToObj;

    // iteration tracking
    int iters_ = 20;
    int lastIterSeen_ = 0;
    QRegularExpression itRe_;
    QRegularExpression progRe_;

    // buffered output for error reporting
    QString ioLog_;

    // resolved executables
    QString tifxyz2objExe_;
    QString obj2tifxyzExe_;

    bool errorShown_ = false;

    void showImmediateToolNotFound_(const char* tool) {
        QMessageBox::critical(w_, QObject::tr("Error"),
            QObject::tr("Could not find the '%1' executable.\n"
                        "Tip: set VC.ini [tools] %1_path or ensure it's on PATH.").arg(tool));
        cleanupAndDelete_();
    }
};

// --------- locate 'flatboi' executable --------------------------------------
static QString findFlatboiExecutable()
{
    const QByteArray envFlatboi = qgetenv("FLATBOI");
    if (!envFlatboi.isEmpty()) {
        const QString p = QString::fromLocal8Bit(envFlatboi);
        QFileInfo fi(p);
        if (fi.exists() && fi.isFile() && fi.isExecutable())
            return fi.absoluteFilePath();
    }

    {
        QSettings settings("VC.ini", QSettings::IniFormat);
        const QString iniPath = settings.value("tools/flatboi_path",
                                               settings.value("tools/flatboi")).toString().trimmed();
        if (!iniPath.isEmpty()) {
            QFileInfo fi(iniPath);
            if (fi.exists() && fi.isFile() && fi.isExecutable())
                return fi.absoluteFilePath();
        }
    }

    const QStringList known = {
        "/usr/local/bin/flatboi",
        "/home/builder/vc-dependencies/bin/flatboi"
    };
    for (const QString& p : known) {
        QFileInfo fi(p);
        if (fi.exists() && fi.isFile() && fi.isExecutable())
            return fi.absoluteFilePath();
    }

#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
    const QString onPath = QStandardPaths::findExecutable("flatboi");
    if (!onPath.isEmpty()) return onPath;
#else
    const QStringList pathDirs =
        QProcessEnvironment::systemEnvironment().value("PATH")
            .split(QDir::listSeparator(), Qt::SkipEmptyParts);
    for (const QString& dir : pathDirs) {
        const QString candidate = QDir(dir).filePath("flatboi");
        QFileInfo fi(candidate);
        if (fi.exists() && fi.isFile() && fi.isExecutable())
            return fi.absoluteFilePath();
    }
#endif

    return {};
}

} // -------------------- end anonymous namespace ------------------------------

// ====================== CWindow member functions ==============================

void CWindow::onRenderSegment(const std::string& segmentId)
{
    auto surfMeta = fVpkg ? fVpkg->getSurface(segmentId) : nullptr;
    if (currentVolume == nullptr || !surfMeta) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot render segment: No volume or invalid segment selected"));
        return;
    }

    QSettings settings("VC.ini", QSettings::IniFormat);

    const QString volumePath = getCurrentVolumePath();
    const QString segmentPath = QString::fromStdString(surfMeta->path.string());
    const QString segmentOutDir = QString::fromStdString(surfMeta->path.string());
    const QString outputFormat = "%s/layers/%02d.tif";
    const float scale = 1.0f;
    const int resolution = 0;
    const int layers = 31;
    const QString outputPattern = QString(outputFormat).replace("%s", segmentOutDir);

    RenderParamsDialog dlg(this, volumePath, segmentPath, outputPattern, scale, resolution, layers);
    if (dlg.exec() != QDialog::Accepted) {
        statusBar()->showMessage(tr("Render cancelled"), 3000);
        return;
    }

    if (!_cmdRunner) {
        _cmdRunner = new CommandLineToolRunner(statusBar(), this, this);
        connect(_cmdRunner, &CommandLineToolRunner::toolStarted,
                [this](CommandLineToolRunner::Tool /*tool*/, const QString& message) {
                    statusBar()->showMessage(message, 0);
                });
        connect(_cmdRunner, &CommandLineToolRunner::toolFinished,
                [this](CommandLineToolRunner::Tool /*tool*/, bool success, const QString& message,
                       const QString& /*outputPath*/, bool copyToClipboard) {
                    if (success) {
                        QString displayMsg = message;
                        if (copyToClipboard) displayMsg += tr(" - Path copied to clipboard");
                        statusBar()->showMessage(displayMsg, 5000);
                        QMessageBox::information(this, tr("Rendering Complete"), displayMsg);
                    } else {
                        statusBar()->showMessage(tr("Rendering failed"), 5000);
                        QMessageBox::critical(this, tr("Rendering Error"), message);
                    }
                });
    }

    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    _cmdRunner->setSegmentPath(dlg.segmentPath());
    _cmdRunner->setOutputPattern(dlg.outputPattern());
    _cmdRunner->setRenderParams(static_cast<float>(dlg.scale()), dlg.groupIdx(), dlg.numSlices());
    _cmdRunner->setOmpThreads(dlg.ompThreads());
    _cmdRunner->setVolumePath(dlg.volumePath());
    _cmdRunner->setRenderAdvanced(
        dlg.cropX(), dlg.cropY(), dlg.cropWidth(), dlg.cropHeight(),
        dlg.affinePath(), dlg.invertAffine(),
        static_cast<float>(dlg.scaleSegmentation()), dlg.rotateDegrees(), dlg.flipAxis());
    _cmdRunner->setIncludeTifs(dlg.includeTifs());

    _cmdRunner->execute(CommandLineToolRunner::Tool::RenderTifXYZ);
    statusBar()->showMessage(tr("Rendering segment: %1").arg(QString::fromStdString(segmentId)), 5000);
}

void CWindow::onSlimFlatten(const std::string& segmentId)
{
    auto surfMeta = fVpkg ? fVpkg->getSurface(segmentId) : nullptr;
    if (currentVolume == nullptr || !surfMeta) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot SLIM-flatten: No volume or invalid segment selected"));
        return;
    }
    if (_cmdRunner && _cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    const std::filesystem::path segDirFs = surfMeta->path; // tifxyz folder
    const QString segDir = QString::fromStdString(segDirFs.string());
    const QString segmentStem = QString::fromStdString(segmentId);

    const QString flatboiExe = findFlatboiExecutable();
    if (flatboiExe.isEmpty()) {
        const QString msg =
            tr("Could not find the 'flatboi' executable.\n"
               "Looked in known locations and PATH.\n\n"
               "Tip: set an override via VC.ini [tools] flatboi_path or FLATBOI env var.");
        QMessageBox::critical(this, tr("Error"), msg);
        statusBar()->showMessage(tr("SLIM-flatten failed"), 5000);
        return;
    }

    new SlimJob(this, segDir, segmentStem, flatboiExe);
}

void CWindow::onGrowSegmentFromSegment(const std::string& segmentId)
{
    if (currentVolume == nullptr || !fVpkg) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot grow segment: No volume package loaded"));
        return;
    }

    auto surfMeta = fVpkg->getSurface(segmentId);
    if (!surfMeta) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot grow segment: Invalid segment or segment not loaded"));
        return;
    }

    if (!initializeCommandLineRunner()) return;
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    QString srcSegment = QString::fromStdString(surfMeta->path.string());

    std::filesystem::path volpkgPath = std::filesystem::path(fVpkgPath.toStdString());
    std::filesystem::path tracesDir = volpkgPath / "traces";
    std::filesystem::path jsonParamsPath = volpkgPath / "trace_params.json";
    std::filesystem::path pathsDir = volpkgPath / "paths";

    statusBar()->showMessage(tr("Preparing to run grow_seg_from_segment..."), 2000);

    if (!std::filesystem::exists(tracesDir)) {
        try { std::filesystem::create_directory(tracesDir); }
        catch (const std::exception& e) {
            QMessageBox::warning(this, tr("Error"), tr("Failed to create traces directory: %1").arg(e.what()));
            return;
        }
    }

    if (!std::filesystem::exists(jsonParamsPath)) {
        QMessageBox::warning(this, tr("Error"), tr("trace_params.json not found in the volpkg"));
        return;
    }

    TraceParamsDialog dlg(this,
                          getCurrentVolumePath(),
                          QString::fromStdString(pathsDir.string()),
                          QString::fromStdString(tracesDir.string()),
                          QString::fromStdString(jsonParamsPath.string()),
                          srcSegment);
    if (dlg.exec() != QDialog::Accepted) {
        statusBar()->showMessage(tr("Run trace cancelled"), 3000);
        return;
    }

    QJsonObject base;
    {
        QFile f(dlg.jsonParams());
        if (f.open(QIODevice::ReadOnly)) {
            const auto doc = QJsonDocument::fromJson(f.readAll());
            f.close();
            if (doc.isObject()) base = doc.object();
        }
    }
    const QJsonObject ui = dlg.makeParamsJson();
    for (auto it = ui.begin(); it != ui.end(); ++it) base[it.key()] = it.value();

    const QString mergedJsonPath = QDir(dlg.tgtDir()).filePath(QString("trace_params_ui.json"));
    {
        QFile f(mergedJsonPath);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            QMessageBox::warning(this, tr("Error"), tr("Failed to write params JSON: %1").arg(mergedJsonPath));
            return;
        }
        f.write(QJsonDocument(base).toJson(QJsonDocument::Indented));
        f.close();
    }

    _cmdRunner->setTraceParams(
        dlg.volumePath(),
        dlg.srcDir(),
        dlg.tgtDir(),
        mergedJsonPath,
        dlg.srcSegment());
    _cmdRunner->setOmpThreads(dlg.ompThreads());

    _cmdRunner->showConsoleOutput();
    _cmdRunner->execute(CommandLineToolRunner::Tool::GrowSegFromSegment);
    statusBar()->showMessage(tr("Growing segment from: %1").arg(QString::fromStdString(segmentId)), 5000);
}

void CWindow::onAddOverlap(const std::string& segmentId)
{
    if (currentVolume == nullptr || !fVpkg) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot add overlap: No volume package loaded"));
        return;
    }

    auto surfMeta = fVpkg->getSurface(segmentId);
    if (!surfMeta) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot add overlap: Invalid segment or segment not loaded"));
        return;
    }

    if (!initializeCommandLineRunner()) return;
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    std::filesystem::path volpkgPath = std::filesystem::path(fVpkgPath.toStdString());
    std::filesystem::path pathsDir = volpkgPath / "paths";
    QString tifxyzPath = QString::fromStdString(surfMeta->path.string());

    _cmdRunner->setAddOverlapParams(QString::fromStdString(pathsDir.string()), tifxyzPath);
    _cmdRunner->execute(CommandLineToolRunner::Tool::SegAddOverlap);
    statusBar()->showMessage(tr("Adding overlap for segment: %1").arg(QString::fromStdString(segmentId)), 5000);
}

void CWindow::onConvertToObj(const std::string& segmentId)
{
    if (currentVolume == nullptr || !fVpkg) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot convert to OBJ: No volume package loaded"));
        return;
    }

    auto surfMeta = fVpkg->getSurface(segmentId);
    if (!surfMeta) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot convert to OBJ: Invalid segment or segment not loaded"));
        return;
    }

    if (!initializeCommandLineRunner()) return;
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    std::filesystem::path tifxyzPath = surfMeta->path;
    std::filesystem::path objPath = tifxyzPath / (segmentId + ".obj");

    ConvertToObjDialog dlg(this,
                           QString::fromStdString(tifxyzPath.string()),
                           QString::fromStdString(objPath.string()));
    if (dlg.exec() != QDialog::Accepted) {
        statusBar()->showMessage(tr("Convert to OBJ cancelled"), 3000);
        return;
    }

    _cmdRunner->setToObjParams(dlg.tifxyzPath(), dlg.objPath());
    _cmdRunner->setOmpThreads(dlg.ompThreads());
    _cmdRunner->setToObjOptions(dlg.normalizeUV(), dlg.alignGrid(), dlg.decimateIterations(), dlg.cleanSurface(), static_cast<float>(dlg.cleanK()));
    _cmdRunner->execute(CommandLineToolRunner::Tool::tifxyz2obj);
    statusBar()->showMessage(tr("Converting segment to OBJ: %1").arg(QString::fromStdString(segmentId)), 5000);
}

void CWindow::onGrowSeeds(const std::string& segmentId, bool isExpand, bool isRandomSeed)
{
    if (currentVolume == nullptr) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot grow seeds: No volume loaded"));
        return;
    }

    if (!initializeCommandLineRunner()) return;
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    std::filesystem::path volpkgPath = std::filesystem::path(fVpkgPath.toStdString());
    std::filesystem::path pathsDir = volpkgPath / "paths";

    if (!std::filesystem::exists(pathsDir)) {
        QMessageBox::warning(this, tr("Error"), tr("Paths directory not found in the volpkg"));
        return;
    }

    QString jsonFileName = isExpand ? "expand.json" : "seed.json";
    std::filesystem::path jsonParamsPath = volpkgPath / jsonFileName.toStdString();

    if (!std::filesystem::exists(jsonParamsPath)) {
        QMessageBox::warning(this, tr("Error"), tr("%1 not found in the volpkg").arg(jsonFileName));
        return;
    }

    int seedX = 0, seedY = 0, seedZ = 0;
    if (!isExpand && !isRandomSeed) {
        POI *poi = _surf_col->poi("focus");
        if (!poi) {
            QMessageBox::warning(this, tr("Error"), tr("No focus point selected. Click on a volume with Ctrl key to set a seed point."));
            return;
        }
        seedX = static_cast<int>(poi->p[0]);
        seedY = static_cast<int>(poi->p[1]);
        seedZ = static_cast<int>(poi->p[2]);
    }

    _cmdRunner->setGrowParams(
        QString(),
        QString::fromStdString(pathsDir.string()),
        QString::fromStdString(jsonParamsPath.string()),
        seedX, seedY, seedZ,
        isExpand, isRandomSeed
    );

    _cmdRunner->execute(CommandLineToolRunner::Tool::GrowSegFromSeeds);

    QString modeDesc = isExpand ? "expand mode" :
                      (isRandomSeed ? "random seed mode" : "seed mode");
    statusBar()->showMessage(tr("Growing segment using %1 in %2").arg(jsonFileName).arg(modeDesc), 5000);
}

bool CWindow::initializeCommandLineRunner()
{
    if (!_cmdRunner) {
        _cmdRunner = new CommandLineToolRunner(statusBar(), this, this);

        QSettings settings("VC.ini", QSettings::IniFormat);
        int parallelProcesses = settings.value("perf/parallel_processes", 8).toInt();
        int iterationCount = settings.value("perf/iteration_count", 1000).toInt();

        _cmdRunner->setParallelProcesses(parallelProcesses);
        _cmdRunner->setIterationCount(iterationCount);

        connect(_cmdRunner, &CommandLineToolRunner::toolStarted,
                [this](CommandLineToolRunner::Tool /*tool*/, const QString& message) {
                    statusBar()->showMessage(message, 0);
                });
        connect(_cmdRunner, &CommandLineToolRunner::toolFinished,
                [this](CommandLineToolRunner::Tool /*tool*/, bool success, const QString& message,
                       const QString& outputPath, bool copyToClipboard) {
                    Q_UNUSED(outputPath);
                    if (success) {
                        QString displayMsg = message;
                        if (copyToClipboard) displayMsg += tr(" - Path copied to clipboard");
                        statusBar()->showMessage(displayMsg, 5000);
                        QMessageBox::information(this, tr("Operation Complete"), displayMsg);
                    } else {
                        statusBar()->showMessage(tr("Operation failed"), 5000);
                        QMessageBox::critical(this, tr("Error"), message);
                    }
                });
    }
    return true;
}

void CWindow::onAWSUpload(const std::string& segmentId)
{
    auto surfMeta = fVpkg ? fVpkg->getSurface(segmentId) : nullptr;
    if (currentVolume == nullptr || !surfMeta) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot upload to AWS: No volume or invalid segment selected"));
        return;
    }
    if (_cmdRunner && _cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    const std::filesystem::path segDirFs = surfMeta->path;
    const QString  segDir   = QString::fromStdString(segDirFs.string());
    const QString  objPath  = QDir(segDir).filePath(QString::fromStdString(segmentId) + ".obj");
    const QString  flatObj  = QDir(segDir).filePath(QString::fromStdString(segmentId) + "_flatboi.obj");
    QString        outTifxyz= segDir + "_flatboi";

    if (!QFileInfo::exists(segDir)) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot upload to AWS: Segment directory not found"));
        return;
    }

    QStringList scrollOptions;
    scrollOptions << "PHerc0172" << "PHerc0343P" << "PHerc0500P2";

    bool ok;
    QString selectedScroll = QInputDialog::getItem(
        this,
        tr("Select Scroll for Upload"),
        tr("Select the target scroll directory:"),
        scrollOptions,
        0, false, &ok
    );

    if (!ok || selectedScroll.isEmpty()) {
        statusBar()->showMessage(tr("AWS upload cancelled by user"), 3000);
        return;
    }

    QSettings settings("VC.ini", QSettings::IniFormat);
    QString defaultProfile = settings.value("aws/default_profile", "").toString();

    QString awsProfile = QInputDialog::getText(
        this, tr("AWS Profile"),
        tr("Enter AWS profile name (leave empty for default credentials):"),
        QLineEdit::Normal, defaultProfile, &ok
    );

    if (!ok) {
        statusBar()->showMessage(tr("AWS upload cancelled by user"), 3000);
        return;
    }

    if (!awsProfile.isEmpty()) settings.setValue("aws/default_profile", awsProfile);

    QStringList uploadedFiles;
    QStringList failedFiles;

    auto uploadFileWithProgress = [&](const QString& localPath, const QString& s3Path, const QString& description, bool isDirectory = false) {
        if (!QFileInfo::exists(localPath)) return;
        if (isDirectory && !QFileInfo(localPath).isDir()) return;

        QStringList awsArgs;
        awsArgs << "s3" << "cp" << localPath << s3Path;
        if (isDirectory) awsArgs << "--recursive";
        if (!awsProfile.isEmpty()) { awsArgs << "--profile" << awsProfile; }

        statusBar()->showMessage(tr("Uploading %1...").arg(description), 0);

        QProcess p;
        p.setWorkingDirectory(segDir);
        p.setProcessChannelMode(QProcess::MergedChannels);
        p.start("aws", awsArgs);
        if (!p.waitForStarted()) { failedFiles << QString("%1: Failed to start AWS CLI").arg(description); return; }

        while (p.state() != QProcess::NotRunning) {
            if (p.waitForReadyRead(100)) {
                QString output = QString::fromLocal8Bit(p.readAllStandardOutput());
                if (!output.isEmpty()) {
                    const QStringList lines = output.split('\n', Qt::SkipEmptyParts);
                    for (const QString& line : lines) {
                        if (line.contains("Completed") || line.contains("upload:")) {
                            statusBar()->showMessage(QString("Uploading %1: %2").arg(description, line.trimmed()), 0);
                        }
                    }
                }
            }
            QCoreApplication::processEvents();
        }

        p.waitForFinished(-1);
        if (p.exitStatus() == QProcess::NormalExit && p.exitCode() == 0) {
            uploadedFiles << description;
        } else {
            QString error = QString::fromLocal8Bit(p.readAllStandardError());
            if (error.isEmpty()) error = QString::fromLocal8Bit(p.readAllStandardOutput());
            failedFiles << QString("%1: %2").arg(description, error);
        }
    };

    auto uploadSegmentContents = [&](const QString& targetDir, const QString& segmentSuffix) {
        QString segmentName = QString::fromStdString(segmentId) + segmentSuffix;

        QString meshPath = QString("s3://vesuvius-challenge/%1/segments/meshes/%2/")
            .arg(selectedScroll).arg(segmentName);

        QString objFile = QDir(targetDir).filePath(segmentName + ".obj");
        uploadFileWithProgress(objFile, meshPath, QString("%1.obj").arg(segmentName));

        QString flatboiObjFile = QDir(targetDir).filePath(segmentName + "_flatboi.obj");
        uploadFileWithProgress(flatboiObjFile, meshPath, QString("%1_flatboi.obj").arg(segmentName));

        QString xTif = QDir(targetDir).filePath("x.tif");
        QString yTif = QDir(targetDir).filePath("y.tif");
        QString zTif = QDir(targetDir).filePath("z.tif");
        QString metaJson = QDir(targetDir).filePath("meta.json");

        if (QFileInfo::exists(xTif) && QFileInfo::exists(yTif) &&
            QFileInfo::exists(zTif) && QFileInfo::exists(metaJson)) {
            uploadFileWithProgress(xTif, meshPath, QString("%1/x.tif").arg(segmentName));
            uploadFileWithProgress(yTif, meshPath, QString("%1/y.tif").arg(segmentName));
            uploadFileWithProgress(zTif, meshPath, QString("%1/z.tif").arg(segmentName));
            uploadFileWithProgress(metaJson, meshPath, QString("%1/meta.json").arg(segmentName));
        }

        QString overlappingJson = QDir(targetDir).filePath("overlapping.json");
        uploadFileWithProgress(overlappingJson, meshPath, QString("%1/overlapping.json").arg(segmentName));

        QString layersDir = QDir(targetDir).filePath("layers");
        if (QFileInfo::exists(layersDir) && QFileInfo(layersDir).isDir()) {
            QString surfaceVolPath = QString("s3://vesuvius-challenge/%1/segments/surface-volumes/%2/layers/")
                .arg(selectedScroll).arg(segmentName);
            uploadFileWithProgress(layersDir, surfaceVolPath, QString("%1/layers").arg(segmentName), true);
        }
    };

    QProgressDialog progressDlg(tr("Uploading to AWS S3..."), tr("Cancel"), 0, 0, this);
    progressDlg.setWindowModality(Qt::WindowModal);
    progressDlg.setAutoClose(false);
    progressDlg.show();

    uploadSegmentContents(segDir, "");
    if (progressDlg.wasCanceled()) { statusBar()->showMessage(tr("AWS upload cancelled"), 3000); return; }
    if (QFileInfo::exists(outTifxyz) && QFileInfo(outTifxyz).isDir()) {
        uploadSegmentContents(outTifxyz, "_flatboi");
    }

    progressDlg.close();

    if (!uploadedFiles.isEmpty() && failedFiles.isEmpty()) {
        QMessageBox::information(this, tr("Upload Complete"),
            tr("Successfully uploaded to S3:\n\n%1").arg(uploadedFiles.join("\n")));
        statusBar()->showMessage(tr("AWS upload complete"), 5000);
    } else if (!uploadedFiles.isEmpty() && !failedFiles.isEmpty()) {
        QMessageBox::warning(this, tr("Partial Upload"),
            tr("Uploaded:\n%1\n\nFailed:\n%2").arg(uploadedFiles.join("\n"), failedFiles.join("\n")));
        statusBar()->showMessage(tr("AWS upload partially complete"), 5000);
    } else if (uploadedFiles.isEmpty() && !failedFiles.isEmpty()) {
        QMessageBox::critical(this, tr("Upload Failed"),
            tr("All uploads failed:\n\n%1\n\nPlease check:\n"
               "- AWS CLI is installed\n"
               "- AWS credentials are configured\n"
               "- You have internet connection\n"
               "- You have permissions for the S3 bucket").arg(failedFiles.join("\n")));
        statusBar()->showMessage(tr("AWS upload failed"), 5000);
    } else {
        QMessageBox::information(this, tr("No Files to Upload"),
            tr("No files found to upload for segment: %1").arg(QString::fromStdString(segmentId)));
        statusBar()->showMessage(tr("No files to upload"), 3000);
    }
}
