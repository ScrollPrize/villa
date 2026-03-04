#include "CWindow.hpp"
#include "SurfacePanelController.hpp"
#include "VCSettings.hpp"
#include "SegmentationCommandHandler.hpp"

#include <QDialogButtonBox>
#include <QSettings>
#include <QMessageBox>
#include <QStandardPaths>
#include <QtConcurrent>
#include <QProgressDialog>

#include "CommandLineToolRunner.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/flattening/ABFFlattening.hpp"
#include "ToolDialogs.hpp"
#include "elements/VolumeSelector.hpp"
#include "elements/JsonProfilePresets.hpp"
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

bool selectResumeLocalTracerParams(QWidget* parent,
                                   const QVector<VolumeSelector::VolumeOption>& volumes,
                                   const QString& defaultVolumeId,
                                   QString* selectedVolumePath,
                                   std::optional<QJsonObject>* paramsOut,
                                   int* ompThreadsOut)
{
    if (!paramsOut || !selectedVolumePath || !ompThreadsOut) {
        return false;
    }

    QDialog dlg(parent);
    dlg.setWindowTitle(QObject::tr("Resume-opt Local (GrowPatch)"));

    auto* main = new QVBoxLayout(&dlg);
    auto* volumeSelector = new VolumeSelector(&dlg);
    volumeSelector->setVolumes(volumes, defaultVolumeId);
    main->addWidget(volumeSelector);

    auto* ompRow = new QWidget(&dlg);
    auto* ompLayout = new QHBoxLayout(ompRow);
    ompLayout->setContentsMargins(0, 0, 0, 0);
    auto* ompLabel = new QLabel(QObject::tr("OMP Threads:"), ompRow);
    auto* ompSpin = new QSpinBox(ompRow);
    ompSpin->setRange(0, 256);
    ompSpin->setToolTip(QObject::tr("If greater than 0, sets OMP_NUM_THREADS for the reoptimization run."));
    ompLayout->addWidget(ompLabel);
    ompLayout->addWidget(ompSpin, 1);
    main->addWidget(ompRow);

    auto* editor = new JsonProfileEditor(QObject::tr("Tracer Params"), &dlg);
    editor->setDescription(QObject::tr(
        "Additional JSON fields merge into the tracer params used for resume-local optimization."));
    editor->setPlaceholderText(QStringLiteral("{\n    \"example_param\": 1\n}"));

    const auto profiles = vc3d::json_profiles::tracerParamProfiles(
        [](const char* text) { return QObject::tr(text); });

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QString savedProfile = settings.value(
        vc3d::settings::neighbor_copy::PASS2_PARAMS_PROFILE,
        QStringLiteral("default")).toString();
    const QString savedText = settings.value(
        vc3d::settings::neighbor_copy::PASS2_PARAMS_TEXT,
        QString()).toString();
    const int savedOmpThreads = settings.value(
        vc3d::settings::neighbor_copy::RESUME_LOCAL_OMP_THREADS,
        vc3d::settings::neighbor_copy::RESUME_LOCAL_OMP_THREADS_DEFAULT).toInt();

    editor->setCustomText(savedText);
    editor->setProfiles(profiles, savedProfile);
    ompSpin->setValue(savedOmpThreads);
    main->addWidget(editor);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dlg, [&]() {
        if (!editor->isValid()) {
            const QString error = editor->errorText();
            QMessageBox::warning(&dlg,
                                 QObject::tr("Error"),
                                 error.isEmpty()
                                     ? QObject::tr("Tracer params JSON is invalid.")
                                     : error);
            return;
        }
        settings.setValue(vc3d::settings::neighbor_copy::PASS2_PARAMS_PROFILE,
                          editor->profile());
        settings.setValue(vc3d::settings::neighbor_copy::PASS2_PARAMS_TEXT,
                          editor->customText());
        settings.setValue(vc3d::settings::neighbor_copy::RESUME_LOCAL_OMP_THREADS,
                          ompSpin->value());
        dlg.accept();
    });
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
    main->addWidget(buttons);

    if (dlg.exec() != QDialog::Accepted) {
        return false;
    }

    *selectedVolumePath = volumeSelector->selectedVolumePath();
    *ompThreadsOut = ompSpin->value();

    QString error;
    auto extra = editor->jsonObject(&error);
    if (!error.isEmpty()) {
        QMessageBox::warning(parent, QObject::tr("Error"), error);
        return false;
    }

    *paramsOut = extra;
    return true;
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
        const QString iniPath = settings.value(vc3d::settings::tools::FLATBOI_PATH,
                                               settings.value(vc3d::settings::tools::FLATBOI)).toString().trimmed();
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

using ProgressCallback = std::function<void(const QString&)>;

struct ABFFlattenTaskConfig {
    QString inputPath;
    QString outputPath;
    int iterations{10};
    int downsampleFactor{1};
    std::shared_ptr<std::atomic_bool> cancelFlag;
};

struct ABFFlattenResult {
    bool success{false};
    bool canceled{false};
    QString errorMsg;
};

static ABFFlattenResult runAbfFlattenTask(const ABFFlattenTaskConfig& cfg, const ProgressCallback& onProgress)
{
    auto emitProgress = [&](const QString& msg) {
        if (onProgress) onProgress(msg);
    };
    auto isCanceled = [&]() -> bool {
        return cfg.cancelFlag && cfg.cancelFlag->load(std::memory_order_relaxed);
    };

    ABFFlattenResult result;
    try {
        if (isCanceled()) {
            result.canceled = true;
            return result;
        }

        emitProgress(QObject::tr("Loading surface..."));
        auto surf = load_quad_from_tifxyz(cfg.inputPath.toStdString());
        if (!surf) {
            result.errorMsg = QObject::tr("Failed to load surface from: %1").arg(cfg.inputPath);
            return result;
        }

        if (isCanceled()) {
            result.canceled = true;
            return result;
        }

        emitProgress(QObject::tr("Running ABF++ flattening..."));
        vc::ABFConfig config;
        config.maxIterations = static_cast<std::size_t>(std::max(1, cfg.iterations));
        config.downsampleFactor = std::max(1, cfg.downsampleFactor);
        config.useABF = true;
        config.scaleToOriginalArea = true;
        config.alignToInputGrid = true;
        config.rotateHighZToTop = false;

        std::unique_ptr<QuadSurface> flatSurf(vc::abfFlattenToNewSurface(*surf, config));
        if (!flatSurf) {
            result.errorMsg = QObject::tr("ABF++ flattening failed");
            return result;
        }

        if (isCanceled()) {
            result.canceled = true;
            return result;
        }

        emitProgress(QObject::tr("Saving flattened surface..."));
        std::filesystem::path outPath(cfg.outputPath.toStdString());
        std::filesystem::create_directories(outPath);
        flatSurf->save(outPath, true);

        result.success = true;
    } catch (const std::exception& e) {
        result.errorMsg = QObject::tr("Error: %1").arg(e.what());
    }

    return result;
}

class ABFJob : public QObject {
    Q_OBJECT
public:
    ABFJob(CWindow* win, SurfacePanelController* surfacePanel, const QString& segDir, const QString& segmentStem, int iterations, int downsampleFactor = 1)
        : QObject(win)
        , w_(win)
        , surfacePanel_(surfacePanel)
        , segDir_(segDir)
        , stem_(segmentStem)
        , outDir_(segDir.endsWith("_abf") ? segDir : (segDir + "_abf"))
        , iterations_(std::max(1, iterations))
        , downsampleFactor_(std::max(1, downsampleFactor))
        , cancelFlag_(std::make_shared<std::atomic_bool>(false))
        , watcher_(this)
        , progress_(new QProgressDialog(QObject::tr("ABF++ Flattening..."), QObject::tr("Cancel"), 0, 0, win))
    {
        progress_->setWindowModality(Qt::NonModal);
        progress_->setMinimumDuration(0);
        progress_->setRange(0, 0); // indeterminate
        progress_->setAttribute(Qt::WA_DeleteOnClose);

        connect(progress_, &QProgressDialog::canceled, this, &ABFJob::onCanceledRequested_);
        connect(&watcher_, &QFutureWatcher<ABFFlattenResult>::finished, this, &ABFJob::onFinished_);

        startTask_();
    }

    ~ABFJob() override {
        if (cancelFlag_) {
            cancelFlag_->store(true, std::memory_order_relaxed);
        }
    }

private slots:
    void onCanceledRequested_() {
        if (cancelFlag_) {
            cancelFlag_->store(true, std::memory_order_relaxed);
        }
        if (progress_) {
            progress_->setLabelText(QObject::tr("Canceling…"));
        }
    }

    void onFinished_() {
        if (progress_) {
            progress_->close();
        }

        if (!watcher_.isFinished()) {
            deleteLater();
            return;
        }

        const ABFFlattenResult result = watcher_.result();

        if (result.canceled) {
            if (w_) {
                w_->statusBar()->showMessage(QObject::tr("ABF++ flatten cancelled"), 5000);
            }
            deleteLater();
            return;
        }

        if (!result.success) {
            if (w_) {
                w_->statusBar()->showMessage(QObject::tr("ABF++ flatten failed"), 5000);
                const QString errorMsg = result.errorMsg.isEmpty()
                    ? QObject::tr("ABF++ flattening failed")
                    : result.errorMsg;
                QMessageBox::critical(w_, QObject::tr("ABF++ Flatten Failed"), errorMsg);
            }
            deleteLater();
            return;
        }

        const QString label = !stem_.isEmpty() ? stem_ : outDir_;

        if (w_) {
            w_->statusBar()->showMessage(QObject::tr("ABF++ flatten complete: %1").arg(label), 5000);
            QMessageBox::information(w_, QObject::tr("ABF++ Flatten Complete"),
                QObject::tr("Flattened surface saved to:\n%1").arg(outDir_));
        }

        if (surfacePanel_) {
            QMetaObject::invokeMethod(surfacePanel_.data(),
                                      &SurfacePanelController::reloadSurfacesFromDisk,
                                      Qt::QueuedConnection);
        }

        deleteLater();
    }

private:
    void startTask_() {
        const ABFFlattenTaskConfig cfg{
            segDir_,
            outDir_,
            iterations_,
            downsampleFactor_,
            cancelFlag_
        };

        QPointer<ABFJob> guard(this);
        auto progressCb = [guard](const QString& msg) {
            if (!guard) return;
            QMetaObject::invokeMethod(guard, [guard, msg]() {
                if (guard && guard->progress_) {
                    guard->progress_->setLabelText(msg);
                }
            }, Qt::QueuedConnection);
        };

        watcher_.setFuture(QtConcurrent::run([cfg, progressCb]() {
            return runAbfFlattenTask(cfg, progressCb);
        }));
    }

    QPointer<CWindow> w_;
    QPointer<SurfacePanelController> surfacePanel_;
    QString segDir_;
    QString stem_;
    QString outDir_;
    int iterations_;
    int downsampleFactor_;
    std::shared_ptr<std::atomic_bool> cancelFlag_;
    QFutureWatcher<ABFFlattenResult> watcher_;
    QPointer<QProgressDialog> progress_;
};

} // -------------------- end anonymous namespace ------------------------------

// ====================== CWindow member functions ==============================

bool CWindow::initializeCommandLineRunner()
{
    if (!_cmdRunner) {
        _cmdRunner = new CommandLineToolRunner(statusBar(), this, this);

        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        int parallelProcesses = settings.value(vc3d::settings::perf::PARALLEL_PROCESSES,
                                               vc3d::settings::perf::PARALLEL_PROCESSES_DEFAULT).toInt();
        int iterationCount = settings.value(vc3d::settings::perf::ITERATION_COUNT,
                                            vc3d::settings::perf::ITERATION_COUNT_DEFAULT).toInt();

        _cmdRunner->setParallelProcesses(parallelProcesses);
        _cmdRunner->setIterationCount(iterationCount);

        if (_segmentationCommandHandler) {
            _segmentationCommandHandler->setCmdRunner(_cmdRunner);
        }

        connect(_cmdRunner, &CommandLineToolRunner::toolStarted,
                [this](CommandLineToolRunner::Tool /*tool*/, const QString& message) {
                    statusBar()->showMessage(message, 0);
                });
        connect(_cmdRunner, &CommandLineToolRunner::toolFinished,
                [this](CommandLineToolRunner::Tool tool, bool success, const QString& message,
                       const QString& outputPath, bool copyToClipboard) {
                    Q_UNUSED(outputPath);
                    const bool neighborJobActive = _segmentationCommandHandler &&
                        _segmentationCommandHandler->neighborCopyJob().has_value() &&
                        tool == CommandLineToolRunner::Tool::NeighborCopy;
                    const bool resumeLocalActive = _segmentationCommandHandler &&
                        !_segmentationCommandHandler->neighborCopyJob() &&
                        _segmentationCommandHandler->resumeLocalJob().has_value() &&
                        tool == CommandLineToolRunner::Tool::NeighborCopy;

                    bool suppressDialogs = neighborJobActive && success &&
                                           _segmentationCommandHandler->neighborCopyJob()->stage ==
                                               SegmentationCommandHandler::NeighborCopyJob::Stage::FirstPass;

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
                        _segmentationCommandHandler->handleNeighborCopyToolFinished(success);
                    }
                    if (resumeLocalActive) {
                        if (success && _surfacePanel) {
                            _surfacePanel->reloadSurfacesFromDisk();
                        }
                        _cmdRunner->setOmpThreads(-1);
                        _segmentationCommandHandler->resumeLocalJob().reset();
                    }
                });
    }
    return true;
}

#include "CWindowContextMenu.moc"
