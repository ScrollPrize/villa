#include "CWindow.hpp"
#include "CSurfaceCollection.hpp"
#include "SurfacePanelController.hpp"
#include "VCSettings.hpp"

#include <functional>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <optional>
#include <vector>

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
#include <QTemporaryFile>
#include <QSet>
#include <QVector>
#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
#include <QStandardPaths>
#endif

#include "CommandLineToolRunner.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "ToolDialogs.hpp"
#include <nlohmann/json.hpp>

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
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
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

bool isValidSurfacePoint(const cv::Vec3f& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]) &&
           !(point[0] == -1.f && point[1] == -1.f && point[2] == -1.f);
}

std::optional<cv::Rect> computeValidSurfaceBounds(const cv::Mat_<cv::Vec3f>& points)
{
    if (points.empty()) {
        return std::nullopt;
    }

    int minRow = points.rows;
    int maxRow = -1;
    int minCol = points.cols;
    int maxCol = -1;

    for (int r = 0; r < points.rows; ++r) {
        for (int c = 0; c < points.cols; ++c) {
            if (!isValidSurfacePoint(points(r, c))) {
                continue;
            }
            minRow = std::min(minRow, r);
            maxRow = std::max(maxRow, r);
            minCol = std::min(minCol, c);
            maxCol = std::max(maxCol, c);
        }
    }

    if (maxRow < 0 || maxCol < 0) {
        return std::nullopt;
    }

    return cv::Rect(minCol,
                    minRow,
                    maxCol - minCol + 1,
                    maxRow - minRow + 1);
}

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
        QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
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
    // Write (or update) meta.json in 'dir' so that it contains:
    //   "scale": [sx, sy]
    // Returns true on success; leaves other JSON keys intact if meta.json exists.
    static bool overwriteMetaScale_(const QString& dir, double sx, double sy) {
        const QString metaPath = QDir(dir).filePath(QStringLiteral("meta.json"));
        QJsonObject root;

        // Try to read existing meta.json (optional).
        if (QFileInfo::exists(metaPath)) {
            QFile in(metaPath);
            if (in.open(QIODevice::ReadOnly)) {
                const auto doc = QJsonDocument::fromJson(in.readAll());
                if (doc.isObject()) root = doc.object();
                in.close();
            }
        }

        QJsonArray scaleArr; scaleArr.append(sx); scaleArr.append(sy);
        root.insert(QStringLiteral("scale"), scaleArr);

        QFile out(metaPath);
        if (!out.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            return false;
        }
        out.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
        out.close();
        return true;
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
        QStringList args;
        args << flatObj_
             << outTemp_
             // Downsample UV grid by 20× per axis to reduce compute/memory.
             << QStringLiteral("--uv-downsample=20");
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

            // Ensure the new tifxyz has a deterministic pixel size in meta.json
            // Requested: "scale": [0.05, 0.05]
            if (!overwriteMetaScale_(outTemp_, 0.05, 0.05)) {
                // Non-fatal: warn but continue with swap and completion.
                QMessageBox* warn = new QMessageBox(QMessageBox::Warning,
                    QObject::tr("Warning"),
                    QObject::tr("Converted directory created, but failed to update meta.json scale in:\n%1")
                        .arg(outTemp_),
                    QMessageBox::Ok, w_);
                warn->setAttribute(Qt::WA_DeleteOnClose);
                warn->open();
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
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
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

static QSet<QString> snapshotDirectoryEntries(const QString& dirPath)
{
    QSet<QString> entries;
    QDir dir(dirPath);
    if (!dir.exists()) {
        return entries;
    }
    const QFileInfoList infoList = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (const QFileInfo& info : infoList) {
        entries.insert(info.fileName());
    }
    return entries;
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

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

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

    if (!initializeCommandLineRunner()) return;

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

void CWindow::onNeighborCopyRequested(const QString& segmentId, bool copyOut)
{
    if (!fVpkg) {
        QMessageBox::warning(this, tr("Error"), tr("No volume package loaded."));
        return;
    }

    if (!initializeCommandLineRunner()) return;
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    if (_neighborCopyJob && _neighborCopyJob->stage != NeighborCopyJob::Stage::None) {
        QMessageBox::warning(this, tr("Warning"), tr("Another neighbor copy request is already running."));
        return;
    }

    auto surfMeta = fVpkg->getSurface(segmentId.toStdString());
    if (!surfMeta) {
        QMessageBox::warning(this, tr("Error"), tr("Invalid surface selected."));
        return;
    }

    QVector<NeighborCopyVolumeOption> volumeOptions;
    for (const auto& volumeId : fVpkg->volumeIDs()) {
        auto volume = fVpkg->volume(volumeId);
        if (!volume) {
            continue;
        }
        NeighborCopyVolumeOption option;
        option.id = QString::fromStdString(volumeId);
        option.name = QString::fromStdString(volume->name());
        option.path = QString::fromStdString(volume->path().string());
        volumeOptions.push_back(option);
    }

    if (volumeOptions.isEmpty()) {
        QMessageBox::warning(this, tr("Error"), tr("No volumes available in the volume package."));
        return;
    }

    QString defaultVolumeId = volumeOptions.front().id;
    if (!currentVolumeId.empty()) {
        const QString currentId = QString::fromStdString(currentVolumeId);
        for (const auto& opt : volumeOptions) {
            if (opt.id == currentId) {
                defaultVolumeId = currentId;
                break;
            }
        }
    }

    const QString surfacePath = QString::fromStdString(surfMeta->path.string());
    QString volpkgRoot = fVpkgPath;
    if (volpkgRoot.isEmpty()) {
        volpkgRoot = QString::fromStdString(fVpkg->getVolpkgDirectory());
    }
    QString defaultOutputDir = QDir(volpkgRoot).filePath(QStringLiteral("paths"));

    NeighborCopyDialog dlg(this, surfacePath, volumeOptions, defaultVolumeId, defaultOutputDir);
    if (dlg.exec() != QDialog::Accepted) {
        statusBar()->showMessage(tr("Copy %1 cancelled").arg(copyOut ? tr("out") : tr("in")), 3000);
        return;
    }

    QString selectedVolumePath = dlg.selectedVolumePath();
    if (selectedVolumePath.isEmpty()) {
        QMessageBox::warning(this, tr("Error"), tr("No target volume selected."));
        return;
    }

    QString outputDirPath = dlg.outputPath().trimmed();
    if (outputDirPath.isEmpty()) {
        QMessageBox::warning(this, tr("Error"), tr("Output path cannot be empty."));
        return;
    }
    QDir outDir(outputDirPath);
    if (!outDir.exists() && !outDir.mkpath(".")) {
        QMessageBox::warning(this, tr("Error"), tr("Failed to create output directory: %1").arg(outputDirPath));
        return;
    }
    outputDirPath = outDir.absolutePath();

    const QString normalGridPath = QDir(volpkgRoot).filePath(QStringLiteral("normal_grids"));

    QJsonObject pass1Params;
    pass1Params["normal_grid_path"] = normalGridPath;
    pass1Params["neighbor_dir"] = copyOut ? QStringLiteral("out") : QStringLiteral("in");
    pass1Params["neighbor_max_distance"] = 50;
    pass1Params["mode"] = QStringLiteral("gen_neighbor");
    pass1Params["neighbor_min_clearance"] = 4;
    pass1Params["neighbor_fill"] = true;
    pass1Params["neighbor_interp_window"] = 5;
    pass1Params["generations"] = 2;
    pass1Params["neighbor_spike_window"] = 2;

    auto pass1JsonFile = std::make_unique<QTemporaryFile>(QDir::temp().filePath("neighbor_copy_pass1_XXXXXX.json"));
    if (!pass1JsonFile->open()) {
        QMessageBox::warning(this, tr("Error"), tr("Failed to create temporary params file."));
        return;
    }
    pass1JsonFile->write(QJsonDocument(pass1Params).toJson(QJsonDocument::Indented));
    pass1JsonFile->flush();

    QJsonObject pass2Params;
    pass2Params["normal_grid_path"] = normalGridPath;
    pass2Params["max_gen"] = 1;
    pass2Params["generations"] = 1;
    pass2Params["resume_local_opt_step"] = dlg.resumeLocalOptStep();
    pass2Params["resume_local_opt_radius"] = dlg.resumeLocalOptRadius();
    pass2Params["resume_local_max_iters"] = dlg.resumeLocalMaxIters();
    pass2Params["resume_local_dense_qr"] = dlg.resumeLocalDenseQr();

    auto pass2JsonFile = std::make_unique<QTemporaryFile>(QDir::temp().filePath("neighbor_copy_pass2_XXXXXX.json"));
    if (!pass2JsonFile->open()) {
        QMessageBox::warning(this, tr("Error"), tr("Failed to create temporary params file for pass 2."));
        return;
    }
    pass2JsonFile->write(QJsonDocument(pass2Params).toJson(QJsonDocument::Indented));
    pass2JsonFile->flush();

    _neighborCopyJob = NeighborCopyJob{};
    auto& job = *_neighborCopyJob;
    job.stage = NeighborCopyJob::Stage::FirstPass;
    job.segmentId = segmentId;
    job.volumePath = selectedVolumePath;
    job.resumeSurfacePath = surfacePath;
    job.outputDir = outputDirPath;
    job.pass1JsonPath = pass1JsonFile->fileName();
    job.pass2JsonPath = pass2JsonFile->fileName();
    job.directoryPrefix = copyOut ? QStringLiteral("neighbor_out_") : QStringLiteral("neighbor_in_");
    job.copyOut = copyOut;
    job.baselineEntries = snapshotDirectoryEntries(outputDirPath);
    job.pass1JsonFile = std::move(pass1JsonFile);
    job.pass2JsonFile = std::move(pass2JsonFile);
    job.generatedSurfacePath.clear();

    if (!startNeighborCopyPass(job.pass1JsonPath,
                               job.resumeSurfacePath,
                               QStringLiteral("skip"),
                               -1)) {
        QMessageBox::warning(this, tr("Error"), tr("Failed to launch neighbor copy pass."));
        _neighborCopyJob.reset();
        _cmdRunner->setOmpThreads(-1);
        return;
    }

    const QString dirName = QFileInfo(job.resumeSurfacePath).fileName();
    statusBar()->showMessage(tr("Copy %1 started for %2")
                                 .arg(copyOut ? tr("out") : tr("in"))
                                 .arg(dirName.isEmpty() ? segmentId : dirName),
                             5000);
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
    _cmdRunner->setToObjOptions(dlg.normalizeUV(), dlg.alignGrid());
    _cmdRunner->execute(CommandLineToolRunner::Tool::tifxyz2obj);
    statusBar()->showMessage(tr("Converting segment to OBJ: %1").arg(QString::fromStdString(segmentId)), 5000);
}

void CWindow::onCropSurfaceToValidRegion(const std::string& segmentId)
{
    if (currentVolume == nullptr || !fVpkg) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot crop surface: No volume package loaded"));
        return;
    }

    auto surfMeta = fVpkg->getSurface(segmentId);
    if (!surfMeta) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot crop surface: Invalid segment or segment not loaded"));
        return;
    }

    QuadSurface* surface = surfMeta->surface();
    if (!surface) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot crop surface: Failed to load data"));
        return;
    }

    cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot crop surface: Missing coordinate grid"));
        return;
    }

    const int origCols = points->cols;
    const int origRows = points->rows;

    const auto boundsOpt = computeValidSurfaceBounds(*points);
    if (!boundsOpt) {
        QMessageBox::warning(this,
                             tr("Crop failed"),
                             tr("Surface %1 does not contain any valid vertices to crop.")
                                 .arg(QString::fromStdString(segmentId)));
        return;
    }

    const cv::Rect roi = *boundsOpt;
    if (roi.x == 0 && roi.y == 0 && roi.width == origCols && roi.height == origRows) {
        if (statusBar()) {
            statusBar()->showMessage(
                tr("Surface %1 already occupies the tightest bounds.")
                    .arg(QString::fromStdString(segmentId)),
                4000);
        }
        return;
    }

    struct CroppedChannel {
        std::string name;
        cv::Mat data;
    };
    std::vector<CroppedChannel> croppedChannels;
    croppedChannels.reserve(surface->channelNames().size());

    const auto channelNames = surface->channelNames();
    for (const auto& name : channelNames) {
        cv::Mat channelData = surface->channel(name, SURF_CHANNEL_NORESIZE);
        if (channelData.empty()) {
            continue;
        }
        if (channelData.cols % origCols != 0 || channelData.rows % origRows != 0) {
            QMessageBox::warning(this,
                                 tr("Crop failed"),
                                 tr("Channel '%1' has size %2×%3, which is not divisible by the surface grid %4×%5.")
                                     .arg(QString::fromStdString(name))
                                     .arg(channelData.cols)
                                     .arg(channelData.rows)
                                     .arg(origCols)
                                     .arg(origRows));
            return;
        }

        const int scaleX = channelData.cols / origCols;
        const int scaleY = channelData.rows / origRows;
        const cv::Rect chanRect(roi.x * scaleX,
                                roi.y * scaleY,
                                roi.width * scaleX,
                                roi.height * scaleY);
        if (chanRect.x < 0 || chanRect.y < 0 ||
            chanRect.x + chanRect.width > channelData.cols ||
            chanRect.y + chanRect.height > channelData.rows) {
            QMessageBox::warning(this,
                                 tr("Crop failed"),
                                 tr("Computed crop exceeds the bounds of channel '%1'.")
                                     .arg(QString::fromStdString(name)));
            return;
        }

        croppedChannels.push_back({name, channelData(chanRect).clone()});
    }

    cv::Mat_<cv::Vec3f> croppedPoints = (*points)(roi).clone();

    std::unique_ptr<QuadSurface> tempSurface;
    try {
        tempSurface = std::make_unique<QuadSurface>(croppedPoints, surface->scale());
        tempSurface->path = surface->path;
        tempSurface->id = surface->id;
        if (surface->meta) {
            tempSurface->meta = new nlohmann::json(*surface->meta);
        }
        for (const auto& ch : croppedChannels) {
            tempSurface->setChannel(ch.name, ch.data);
        }
        tempSurface->save(surface->path.string(), surface->id, true);
    } catch (const std::exception& ex) {
        QMessageBox::critical(this,
                              tr("Crop failed"),
                              tr("Failed to crop %1: %2")
                                  .arg(QString::fromStdString(segmentId))
                                  .arg(QString::fromUtf8(ex.what())));
        return;
    }

    croppedPoints.copyTo(*points);
    for (const auto& ch : croppedChannels) {
        surface->setChannel(ch.name, ch.data);
    }
    surface->invalidateCache();

    if (tempSurface && tempSurface->meta) {
        if (!surface->meta) {
            surface->meta = new nlohmann::json(*tempSurface->meta);
        } else {
            *surface->meta = *tempSurface->meta;
        }
        if (surface->meta) {
            if (surfMeta->meta) {
                *surfMeta->meta = *surface->meta;
            } else {
                surfMeta->meta = new nlohmann::json(*surface->meta);
            }
        }
    }

    surfMeta->bbox = surface->bbox();

    if (_surf_col) {
        _surf_col->setSurface(segmentId, surface, false, false);
        if (_surfID == segmentId) {
            _surf_col->setSurface("segmentation", surface, false, false);
        }
    }
    if (_surfacePanel) {
        _surfacePanel->refreshSurfaceMetrics(segmentId);
    }

    const QString segLabel = QString::fromStdString(segmentId);
    if (statusBar()) {
        statusBar()->showMessage(
            tr("Cropped %1 to %2×%3 (offset %4,%5)")
                .arg(segLabel)
                .arg(roi.width)
                .arg(roi.height)
                .arg(roi.x)
                .arg(roi.y),
            5000);
    }
}

void CWindow::onAlphaCompRefine(const std::string& segmentId)
{
    if (currentVolume == nullptr || !fVpkg) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot refine surface: No volume package loaded"));
        return;
    }

    auto surfMeta = fVpkg->getSurface(segmentId);
    if (!surfMeta) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot refine surface: Invalid segment or segment not loaded"));
        return;
    }

    if (!initializeCommandLineRunner()) return;
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    QString volumePath = getCurrentVolumePath();
    if (volumePath.isEmpty()) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot refine surface: Unable to determine volume path"));
        return;
    }

    QString srcPath = QString::fromStdString(surfMeta->path.string());
    QFileInfo srcInfo(srcPath);

    QString defaultOutput;
    if (srcInfo.isDir()) {
        defaultOutput = srcInfo.absoluteFilePath() + "_refined";
    } else {
        const QString base = srcInfo.completeBaseName();
        const QString suffix = srcInfo.completeSuffix();
        QString candidate = srcInfo.absolutePath() + "/" + base + "_refined";
        if (!suffix.isEmpty()) {
            candidate += "." + suffix;
        }
        defaultOutput = candidate;
    }

    AlphaCompRefineDialog dlg(this, volumePath, srcPath, defaultOutput);
    if (dlg.exec() != QDialog::Accepted) {
        statusBar()->showMessage(tr("Alpha-comp refinement cancelled"), 3000);
        return;
    }

    if (dlg.volumePath().isEmpty() || dlg.srcPath().isEmpty() || dlg.dstPath().isEmpty()) {
        QMessageBox::warning(this, tr("Error"), tr("Volume, source, and output paths must be specified"));
        return;
    }

    QJsonObject paramsJson = dlg.paramsJson();
    QString paramsPath = QDir(QDir::tempPath()).filePath(
        QStringLiteral("vc_objrefine_%1.json").arg(QDateTime::currentMSecsSinceEpoch()));

    QFile paramsFile(paramsPath);
    if (!paramsFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        QMessageBox::warning(this, tr("Error"), tr("Failed to write params JSON: %1").arg(paramsPath));
        return;
    }
    paramsFile.write(QJsonDocument(paramsJson).toJson(QJsonDocument::Indented));
    paramsFile.close();

    _cmdRunner->setObjRefineParams(dlg.volumePath(), dlg.srcPath(), dlg.dstPath(), paramsPath);
    _cmdRunner->setOmpThreads(dlg.ompThreads());
    _cmdRunner->execute(CommandLineToolRunner::Tool::AlphaCompRefine);
    statusBar()->showMessage(tr("Refining segment: %1").arg(QString::fromStdString(segmentId)), 5000);
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

        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        int parallelProcesses = settings.value("perf/parallel_processes", 8).toInt();
        int iterationCount = settings.value("perf/iteration_count", 1000).toInt();

        _cmdRunner->setParallelProcesses(parallelProcesses);
        _cmdRunner->setIterationCount(iterationCount);

        connect(_cmdRunner, &CommandLineToolRunner::toolStarted,
                [this](CommandLineToolRunner::Tool /*tool*/, const QString& message) {
                    statusBar()->showMessage(message, 0);
                });
        connect(_cmdRunner, &CommandLineToolRunner::toolFinished,
                [this](CommandLineToolRunner::Tool tool, bool success, const QString& message,
                       const QString& outputPath, bool copyToClipboard) {
                    Q_UNUSED(outputPath);
                    const bool neighborJobActive = _neighborCopyJob.has_value() &&
                        tool == CommandLineToolRunner::Tool::NeighborCopy;

                    bool suppressDialogs = neighborJobActive && success &&
                                           _neighborCopyJob->stage == NeighborCopyJob::Stage::FirstPass;

                    if (!suppressDialogs) {
                        if (success) {
                            QString displayMsg = message;
                            if (copyToClipboard) displayMsg += tr(" - Path copied to clipboard");
                            statusBar()->showMessage(displayMsg, 5000);
                            QMessageBox::information(this, tr("Operation Complete"), displayMsg);
                        } else {
                            statusBar()->showMessage(tr("Operation failed"), 5000);
                            QMessageBox::critical(this, tr("Error"), message);
                        }
                    } else {
                        statusBar()->showMessage(tr("Neighbor copy pass 1 complete"), 2000);
                    }

                    if (neighborJobActive) {
                        handleNeighborCopyToolFinished(success);
                    }
                });
    }
    return true;
}

void CWindow::handleNeighborCopyToolFinished(bool success)
{
    if (!_neighborCopyJob) {
        return;
    }

    auto& job = *_neighborCopyJob;
    if (!success) {
        _cmdRunner->setOmpThreads(-1);
        _neighborCopyJob.reset();
        return;
    }

    if (job.stage == NeighborCopyJob::Stage::FirstPass) {
        const QString newSurface = findNewNeighborSurface(job);
        if (newSurface.isEmpty()) {
            QMessageBox::warning(this, tr("Error"),
                                 tr("Could not locate the newly generated neighbor surface in %1.")
                                     .arg(job.outputDir));
            _cmdRunner->setOmpThreads(-1);
            _neighborCopyJob.reset();
            return;
        }

        job.generatedSurfacePath = newSurface;
        job.baselineEntries.insert(QFileInfo(newSurface).fileName());
        job.stage = NeighborCopyJob::Stage::SecondPass;

        statusBar()->showMessage(
            tr("Neighbor copy pass 1 complete: %1")
                .arg(QFileInfo(newSurface).fileName()),
            3000);

        launchNeighborCopySecondPass();
        return;
    }

    const bool copyOut = job.copyOut;
    const QString surfaceName = QFileInfo(job.generatedSurfacePath).fileName();
    _neighborCopyJob.reset();
    _cmdRunner->setOmpThreads(-1);

    if (_surfacePanel) {
        _surfacePanel->reloadSurfacesFromDisk();
    }

    statusBar()->showMessage(tr("Copy %1 complete: %2")
                                 .arg(copyOut ? tr("out") : tr("in"))
                                 .arg(surfaceName),
                             5000);
}

QString CWindow::findNewNeighborSurface(const NeighborCopyJob& job) const
{
    QDir dir(job.outputDir);
    if (!dir.exists()) {
        return QString();
    }

    const QFileInfoList infoList = dir.entryInfoList(
        QDir::Dirs | QDir::NoDotAndDotDot,
        QDir::Time);

    QFileInfo newest;
    bool found = false;
    for (const QFileInfo& info : infoList) {
        const QString name = info.fileName();
        if (!name.startsWith(job.directoryPrefix)) {
            continue;
        }
        if (job.baselineEntries.contains(name)) {
            continue;
        }
        if (!found || info.lastModified() > newest.lastModified()) {
            newest = info;
            found = true;
        }
    }

    return found ? newest.absoluteFilePath() : QString();
}

bool CWindow::startNeighborCopyPass(const QString& paramsPath,
                                    const QString& resumeSurface,
                                    const QString& resumeOpt,
                                    int ompThreads)
{
    if (!_cmdRunner || !_neighborCopyJob) {
        return false;
    }

    auto& job = *_neighborCopyJob;
    _cmdRunner->setNeighborCopyParams(
        job.volumePath,
        paramsPath,
        resumeSurface,
        job.outputDir,
        resumeOpt);
    _cmdRunner->setOmpThreads(ompThreads);
    _cmdRunner->showConsoleOutput();
    return _cmdRunner->execute(CommandLineToolRunner::Tool::NeighborCopy);
}

void CWindow::launchNeighborCopySecondPass()
{
    if (!_neighborCopyJob) {
        return;
    }

    const QString resumeSurface = _neighborCopyJob->generatedSurfacePath;
    const bool copyOut = _neighborCopyJob->copyOut;

    QTimer::singleShot(0, this, [this, resumeSurface, copyOut]() {
        if (!_neighborCopyJob || _neighborCopyJob->stage != NeighborCopyJob::Stage::SecondPass) {
            return;
        }
        if (!startNeighborCopyPass(_neighborCopyJob->pass2JsonPath,
                                   resumeSurface,
                                   QStringLiteral("local"),
                                   12)) {
            _cmdRunner->setOmpThreads(-1);
            QMessageBox::warning(this, tr("Error"), tr("Failed to launch the second neighbor copy pass."));
            _neighborCopyJob.reset();
            return;
        }

        statusBar()->showMessage(
            tr("Copy %1 pass 2 running").arg(copyOut ? tr("out") : tr("in")),
            3000);
    });
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

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
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

void CWindow::onExportWidthChunks(const std::string& segmentId)
{
    auto surfMeta = fVpkg ? fVpkg->getSurface(segmentId) : nullptr;
    if (currentVolume == nullptr || !surfMeta) {
        QMessageBox::warning(this, tr("Error"),
                             tr("Cannot export: No volume or invalid segment selected"));
        return;
    }

    QuadSurface* surf = surfMeta->surface();
    if (!surf) {
        QMessageBox::warning(this, tr("Error"),
                             tr("Cannot export: Failed to load segment surface"));
        return;
    }

    // Pull points and get dimensions early so we can show them in the dialog
    cv::Mat_<cv::Vec3f> points = surf->rawPoints();
    const int W = points.cols;
    const int H = points.rows;
    const cv::Vec2f sc = surf->scale();
    const double sx = (std::isfinite(sc[0]) && sc[0] > 0.0f) ? double(sc[0]) : 1.0; // guard

    if (W <= 0 || H <= 0) {
        QMessageBox::warning(this, tr("Error"),
                             tr("Surface has invalid dimensions (%1 x %2)").arg(W).arg(H));
        return;
    }

    // Show dialog to get export parameters
    ExportChunksDialog dlg(this, W, sx);
    if (dlg.exec() != QDialog::Accepted) {
        return;
    }

    const int chunkWidthReal = dlg.chunkWidth();
    const int overlapReal = dlg.overlapPerSide();
    const bool overwrite = dlg.overwrite();

    // Determine export root directory: <volpkg>/export (not inside paths)
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QString configuredRoot = settings.value("export/dir", "").toString().trimmed();
    const QString segDir  = QString::fromStdString(surfMeta->path.string());
    const QString segName = QString::fromStdString(segmentId);

    QString volpkgRoot = fVpkg ? QString::fromStdString(fVpkg->getVolpkgDirectory()) : QString();
    if (volpkgRoot.isEmpty()) {
        QDir d(QFileInfo(segDir).absoluteDir());   // start at parent of the segment folder
        while (!d.isRoot() && !d.dirName().endsWith(".volpkg")) d.cdUp();
        volpkgRoot = d.dirName().endsWith(".volpkg") ? d.absolutePath()
                                                    : QFileInfo(segDir).absolutePath();
    }
    const QString exportRoot = configuredRoot.isEmpty()
        ? QDir(volpkgRoot).filePath("export")
        : configuredRoot;

    QDir outRoot(exportRoot);
    if (!outRoot.exists() && !outRoot.mkpath(".")) {
        QMessageBox::critical(this, tr("Error"),
                              tr("Cannot create export directory:\n%1").arg(exportRoot));
        return;
    }

    // Convert real pixels to grid columns
    // Example: 40k real px with scale 0.05 → 2,000 columns per chunk
    const int chunkCols = std::max(1, int(std::llround(double(chunkWidthReal) * sx)));
    const int overlapCols = int(std::llround(double(overlapReal) * sx));

    // Calculate number of chunks: step through by chunkCols (the core width)
    const int nChunks = (W + chunkCols - 1) / chunkCols; // ceil-div purely in grid space

    if (nChunks <= 0) {
        QMessageBox::information(this, tr("Export"), tr("Nothing to export."));
        return;
    }

    // Progress dialog
    QProgressDialog prog(tr("Exporting width-chunks…"), tr("Cancel"), 0, nChunks, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setAutoClose(false);
    prog.setAutoReset(true);
    prog.setMinimumDuration(0);

    // Helper to generate a unique directory name if overwrite is false and target exists
    auto uniqueName = [&](const QString& base)->QString {
        if (!QFileInfo(outRoot.filePath(base)).exists()) return base;
        int k = 1;
        while (QFileInfo(outRoot.filePath(QString("%1_%2").arg(base).arg(k))).exists()) ++k;
        return QString("%1_%2").arg(base).arg(k);
    };

    // Zero-pad for nicer sorting
    auto padded = [nChunks](int idx)->QString {
        const int digits = (nChunks < 10) ? 1 : (nChunks < 100) ? 2 : (nChunks < 1000) ? 3 : 4;
        return QString("%1").arg(idx, digits, 10, QChar('0'));
    };

    // Export loop
    int exported = 0;
    QStringList results;
    QStringList failures;

    for (int c = 0; c < nChunks; ++c) {
        if (prog.wasCanceled()) break;
        prog.setLabelText(tr("Exporting chunk %1 / %2…").arg(c+1).arg(nChunks));
        prog.setValue(c);
        QCoreApplication::processEvents();

        // Core region for chunk c starts at c * chunkCols
        const int coreStart = c * chunkCols;

        // Calculate actual region with overlap:
        // - Left overlap: only if not the first chunk
        // - Right overlap: only if not the last chunk
        const int leftOverlap = (c == 0) ? 0 : overlapCols;
        const int rightOverlap = (c == nChunks - 1) ? 0 : overlapCols;

        // x0 = start of region (core start minus left overlap, clamped to 0)
        const int x0 = std::max(0, coreStart - leftOverlap);
        // x1 = end of region (core end plus right overlap, clamped to W)
        const int coreEnd = std::min(coreStart + chunkCols, W);
        const int x1 = std::min(coreEnd + rightOverlap, W);
        const int dx = x1 - x0;

        if (dx <= 0) continue;

        // ROI [all rows, x0:x1)
        cv::Mat_<cv::Vec3f> roi(points, cv::Range::all(), cv::Range(x0, x1));
        cv::Mat_<cv::Vec3f> roiCopy = roi.clone();  // ensure contiguous, independent buffer

        // Create a temp surface for this chunk; scale is preserved.
        QuadSurface chunkSurf(roiCopy, surf->scale());

        // Build target dir under exportRoot, name "<segName>_<indexPadded>"
        const QString baseName = QString("%1_%2").arg(segName, padded(c));
        QString outDirName = baseName;
        bool forceOverwrite = false;
        if (QFileInfo(outRoot.filePath(outDirName)).exists()) {
            if (overwrite) {
                forceOverwrite = true;
            } else {
                outDirName = uniqueName(baseName);
            }
        }
        const QString outAbs = outRoot.filePath(outDirName);
        const std::string outPath = outAbs.toStdString();
        const std::string uuid    = outDirName.toStdString();  // uuid ~ folder name

        try {
            chunkSurf.save(outPath, uuid, forceOverwrite);
            ++exported;
            results << outAbs;
        } catch (const std::exception& e) {
            failures << QString("%1 — %2").arg(outAbs, e.what());
        }

        QCoreApplication::processEvents();
    }
    prog.setValue(nChunks);

    // Summarize
    if (exported > 0 && failures.isEmpty()) {
        QMessageBox::information(this, tr("Export complete"),
                                 tr("Exported %1 chunk(s) to:\n%2")
                                 .arg(exported)
                                 .arg(QDir::toNativeSeparators(exportRoot)));
        statusBar()->showMessage(tr("Exported %1 chunk(s) → %2")
                                 .arg(exported)
                                 .arg(QDir::toNativeSeparators(exportRoot)),
                                 5000);
    } else if (exported > 0 && !failures.isEmpty()) {
        QMessageBox::warning(this, tr("Partial export"),
                             tr("Exported %1 chunk(s), but failed:\n\n%2")
                             .arg(exported)
                             .arg(failures.join('\n')));
        statusBar()->showMessage(tr("Export partially complete"), 5000);
    } else if (!failures.isEmpty()) {
        QMessageBox::critical(this, tr("Export failed"),
                              tr("All chunks failed:\n\n%1").arg(failures.join('\n')));
        statusBar()->showMessage(tr("Export failed"), 5000);
    } else {
        statusBar()->showMessage(tr("Export cancelled"), 3000);
    }
}
