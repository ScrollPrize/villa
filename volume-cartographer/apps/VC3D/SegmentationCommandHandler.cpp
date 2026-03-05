#include "SegmentationCommandHandler.hpp"
#include "CState.hpp"
#include "SurfacePanelController.hpp"
#include "VCSettings.hpp"

#include <functional>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <optional>
#include <atomic>
#include <vector>
#include <memory>
#include <filesystem>

#include <QSettings>
#include <QMessageBox>
#include <QProcess>
#include <QDir>
#include <QFileInfo>
#include <QCoreApplication>
#include <QDateTime>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QInputDialog>
#include <QDialog>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QVBoxLayout>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QFile>
#include <QTextStream>
#include <QtGlobal>
#include <QProcessEnvironment>
#include <QProgressDialog>
#include <QFutureWatcher>
#include <QPointer>
#include <QTimer>
#include <QTemporaryFile>
#include <QSet>
#include <QVector>
#include <QSpinBox>
#include <QLineEdit>
#include <QtConcurrent/QtConcurrentRun>
#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
#include <QStandardPaths>
#endif

#include "CommandLineToolRunner.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Segmentation.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/flattening/ABFFlattening.hpp"
#include "ToolDialogs.hpp"
#include "elements/VolumeSelector.hpp"
#include "elements/JsonProfilePresets.hpp"
#include <nlohmann/json.hpp>

// --------- locate executables (unified helper) --------------------------------

static QString findExecutable(
    const QString& name,
    const QStringList& extraPaths = {},
    const QString& envVar = {})
{
    // 1. Environment variable override (if provided)
    if (!envVar.isEmpty()) {
        const QByteArray envVal = qgetenv(envVar.toUtf8().constData());
        if (!envVal.isEmpty()) {
            QFileInfo fi(QString::fromLocal8Bit(envVal));
            if (fi.exists() && fi.isFile() && fi.isExecutable())
                return fi.absoluteFilePath();
        }
    }

    // 2. INI settings (tools/<name>_path, tools/<name>)
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
    }

    // 3. Extra hard-coded paths (caller-supplied)
    for (const QString& p : extraPaths) {
        QFileInfo fi(p);
        if (fi.exists() && fi.isFile() && fi.isExecutable())
            return fi.absoluteFilePath();
    }

    // 4. QStandardPaths / manual PATH walk
#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
    const QString onPath = QStandardPaths::findExecutable(name);
    if (!onPath.isEmpty()) return onPath;
#else
    const QStringList pathDirs =
        QProcessEnvironment::systemEnvironment().value("PATH")
            .split(QDir::listSeparator(), Qt::SkipEmptyParts);
    for (const QString& dir : pathDirs) {
        const QString candidate = QDir(dir).filePath(name);
        QFileInfo fi(candidate);
        if (fi.exists() && fi.isFile() && fi.isExecutable())
            return fi.absoluteFilePath();
    }
#endif
    return {};
}

// Convenience wrappers matching the old signatures
static QString findVcTool(const char* name)
{
    return findExecutable(QString::fromLatin1(name));
}

static QString findFlatboiExecutable()
{
    return findExecutable(
        QStringLiteral("flatboi"),
        {QStringLiteral("/usr/local/bin/flatboi"),
         QStringLiteral("/home/builder/vc-dependencies/bin/flatboi")},
        QStringLiteral("FLATBOI"));
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

static bool hasTifxyzMeshFiles(const std::filesystem::path& dir)
{
    return std::filesystem::is_directory(dir)
        && std::filesystem::is_regular_file(dir / "x.tif")
        && std::filesystem::is_regular_file(dir / "y.tif")
        && std::filesystem::is_regular_file(dir / "z.tif");
}

static QJsonObject readJsonObject(const QString& path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return {};

    const auto doc = QJsonDocument::fromJson(f.readAll());
    if (!doc.isObject()) return {};
    return doc.object();
}

static bool writeJsonObject(const QString& path, const QJsonObject& obj)
{
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return false;
    f.write(QJsonDocument(obj).toJson(QJsonDocument::Indented));
    return true;
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
    SlimJob(QWidget* parentWidget,
            const QString& segDir,
            const QString& segmentStem,
            const QString& flatboiExe,
            SegmentationCommandHandler* handler)
    : QObject(handler)
    , parentWidget_(parentWidget)
    , handler_(handler)
    , segDir_(segDir)
    , stem_(segmentStem)
    , objPath_(QDir(segDir).filePath(segmentStem + ".obj"))
    , flatObj_(QDir(segDir).filePath(segmentStem + "_flatboi.obj"))
    , outFinal_(segDir.endsWith("_flatboi") ? segDir : (segDir + "_flatboi"))
    , outTemp_ (segDir.endsWith("_flatboi") ? (segDir + "__rebuild_tmp__") : outFinal_)
    , flatboiExe_(flatboiExe)
    , inputIsAlreadyFlat_(segDir.endsWith("_flatboi"))
    , proc_(new QProcess(this))
    , progress_(new QProgressDialog(QObject::tr("Preparing SLIM..."), QObject::tr("Cancel"), 0, 0, parentWidget))
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

        if (handler_) emit handler_->statusMessage(QObject::tr("Converting TIFXYZ to OBJ..."), 0);
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
        progress_->setLabelText(QObject::tr("Converting TIFXYZ -> OBJ..."));
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
        progress_->setLabelText(QObject::tr("Running SLIM (flatboi)..."));
        progress_->setValue(1);
        ioLog_.clear();
        QStringList args; args << objPath_ << QString::number(iters_);
        ioLog_ += QStringLiteral("Running: %1 %2\n").arg(flatboiExe_, args.join(' '));
        proc_->start(flatboiExe_, args);
    }

    void startToTifxyz_() {
        if (obj2tifxyzExe_.isEmpty()) { showImmediateToolNotFound_("vc_obj2tifxyz"); return; }
        phase_ = Phase::ToTifxyz;
        progress_->setLabelText(QObject::tr("Converting flattened OBJ -> TIFXYZ..."));
        progress_->setValue(1 + iters_);

        // IMPORTANT: vc_obj2tifxyz expects the target directory NOT to exist.
        if (QFileInfo::exists(outTemp_)) {
            ioLog_ += QStringLiteral("Removing existing output dir: %1\n").arg(outTemp_);
            if (!QDir(outTemp_).removeRecursively()) {
                QMessageBox::critical(parentWidget_, QObject::tr("Error"),
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
            QMessageBox::critical(parentWidget_, QObject::tr("Error"),
                                  QObject::tr("Cannot create parent directory: %1").arg(parentPath));
            cleanupAndDelete_();
            return;
        }

        ioLog_.clear();
        QStringList args;
        args << flatObj_
             << outTemp_
             // Downsample UV grid by 20x per axis to reduce compute/memory.
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
                    QMessageBox::Ok, parentWidget_);
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
                                           QMessageBox::Ok, parentWidget_);
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

        if (handler_) emit handler_->statusMessage(QObject::tr("SLIM-flatten cancelled"), 5000);
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
                if (handler_) emit handler_->statusMessage(line, 0);
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
                                           QMessageBox::Ok, parentWidget_);
        box->setAttribute(Qt::WA_DeleteOnClose);
        QObject::connect(box, &QMessageBox::finished, this, [this]() { cleanupAndDelete_(); });
        box->open();
        if (handler_) emit handler_->statusMessage(QObject::tr("SLIM-flatten failed"), 5000);
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
                                               QMessageBox::Ok, parentWidget_);
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
            if (handler_) emit handler_->statusMessage(QObject::tr("SLIM-flatten failed"), 5000);
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
                    QMessageBox::Ok, parentWidget_);
                warn->setAttribute(Qt::WA_DeleteOnClose);
                warn->open();
            }

            phase_ = Phase::Swap;
            finishSwapIfNeeded_();
            phase_ = Phase::Done;

            if (handler_) emit handler_->statusMessage(QObject::tr("SLIM-flatten complete: %1").arg(outFinal_), 5000);
            showDoneAndCleanup_();
            return;
        }
    }

private:
    QPointer<QWidget> parentWidget_;
    QPointer<SegmentationCommandHandler> handler_;

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
        QMessageBox::critical(parentWidget_, QObject::tr("Error"),
            QObject::tr("Could not find the '%1' executable.\n"
                        "Tip: set VC.ini [tools] %1_path or ensure it's on PATH.").arg(tool));
        cleanupAndDelete_();
    }
};

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
    ABFJob(QWidget* parentWidget, SurfacePanelController* surfacePanel,
           SegmentationCommandHandler* handler,
           const QString& segDir, const QString& segmentStem,
           int iterations, int downsampleFactor = 1)
        : QObject(handler)
        , parentWidget_(parentWidget)
        , handler_(handler)
        , surfacePanel_(surfacePanel)
        , segDir_(segDir)
        , stem_(segmentStem)
        , outDir_(segDir.endsWith("_abf") ? segDir : (segDir + "_abf"))
        , iterations_(std::max(1, iterations))
        , downsampleFactor_(std::max(1, downsampleFactor))
        , cancelFlag_(std::make_shared<std::atomic_bool>(false))
        , watcher_(this)
        , progress_(new QProgressDialog(QObject::tr("ABF++ Flattening..."), QObject::tr("Cancel"), 0, 0, parentWidget))
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
            progress_->setLabelText(QObject::tr("Canceling..."));
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
            if (handler_) {
                emit handler_->statusMessage(QObject::tr("ABF++ flatten cancelled"), 5000);
            }
            deleteLater();
            return;
        }

        if (!result.success) {
            if (handler_) {
                emit handler_->statusMessage(QObject::tr("ABF++ flatten failed"), 5000);
                const QString errorMsg = result.errorMsg.isEmpty()
                    ? QObject::tr("ABF++ flattening failed")
                    : result.errorMsg;
                QMessageBox::critical(parentWidget_, QObject::tr("ABF++ Flatten Failed"), errorMsg);
            }
            deleteLater();
            return;
        }

        const QString label = !stem_.isEmpty() ? stem_ : outDir_;

        if (handler_) {
            emit handler_->statusMessage(QObject::tr("ABF++ flatten complete: %1").arg(label), 5000);
            QMessageBox::information(parentWidget_, QObject::tr("ABF++ Flatten Complete"),
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

    QWidget* parentWidget_ = nullptr;
    QPointer<SegmentationCommandHandler> handler_;
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

// ====================== SegmentationCommandHandler ============================

SegmentationCommandHandler::SegmentationCommandHandler(QWidget* parentWidget,
                                                       CState* state,
                                                       QObject* parent)
    : QObject(parent)
    , _parentWidget(parentWidget)
    , _state(state)
{
}

QString SegmentationCommandHandler::getCurrentVolumePath() const
{
    if (_state->currentVolume() == nullptr) {
        return QString();
    }
    return QString::fromStdString(_state->currentVolume()->path().string());
}

QuadSurface* SegmentationCommandHandler::requireSurfaceAndRunner(
    const std::string& segmentId,
    bool checkRunner)
{
    if (_state->currentVolume() == nullptr || !_state->vpkg()) {
        QMessageBox::warning(_parentWidget, tr("Error"),
                             tr("No volume package or volume loaded."));
        return nullptr;
    }

    auto surf = _state->vpkg()->getSurface(segmentId);
    if (!surf) {
        QMessageBox::warning(_parentWidget, tr("Error"),
                             tr("Invalid segment or segment not loaded: %1")
                                 .arg(QString::fromStdString(segmentId)));
        return nullptr;
    }

    if (checkRunner) {
        if (!_cmdRunner) {
            emit statusMessage(tr("Command line tools not available"), 3000);
            return nullptr;
        }
        if (_cmdRunner->isRunning()) {
            QMessageBox::warning(_parentWidget, tr("Warning"),
                                 tr("A command line tool is already running."));
            return nullptr;
        }
    }

    // Safe to return raw pointer: getSurface() returns a shared_ptr backed by
    // Segmentation::surface_ (a cached member), so the pointed-to object remains
    // alive as long as the Segmentation exists in the VolumePkg.
    return surf.get();
}

QVector<VolumeSelector::VolumeOption>
SegmentationCommandHandler::buildVolumeOptionList(QString* defaultOut)
{
    QVector<VolumeSelector::VolumeOption> options;
    if (!_state->vpkg()) {
        return options;
    }

    for (const auto& volumeId : _state->vpkg()->volumeIDs()) {
        auto volume = _state->vpkg()->volume(volumeId);
        if (!volume) {
            continue;
        }
        VolumeSelector::VolumeOption opt;
        opt.id = QString::fromStdString(volumeId);
        opt.name = QString::fromStdString(volume->name());
        opt.path = QString::fromStdString(volume->path().string());
        options.push_back(opt);
    }

    if (defaultOut && !options.isEmpty()) {
        *defaultOut = options.front().id;
        if (!_state->currentVolumeId().empty()) {
            const QString currentId = QString::fromStdString(_state->currentVolumeId());
            for (const auto& opt : options) {
                if (opt.id == currentId) {
                    *defaultOut = currentId;
                    break;
                }
            }
        }
    }

    return options;
}

void SegmentationCommandHandler::onRenderSegment(const std::string& segmentId)
{
    auto* surface = requireSurfaceAndRunner(segmentId, false);
    if (!surface) return;

    auto surf = _state->vpkg()->getSurface(segmentId);

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    const QString volumePath = getCurrentVolumePath();
    const QString segmentPath = QString::fromStdString(surf->path.string());
    const QString segmentOutDir = QString::fromStdString(surf->path.string());
    const QString outputFormat = "%s/layers/%02d.tif";
    const float scale = 1.0f;
    const int resolution = 0;
    const int layers = 31;
    const QString outputPattern = QString(outputFormat).replace("%s", segmentOutDir);

    RenderParamsDialog dlg(_parentWidget, volumePath, segmentPath, outputPattern, scale, resolution, layers);
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Render cancelled"), 3000);
        return;
    }

    if (!_cmdRunner) {
        emit statusMessage(tr("Command line tools not available"), 3000);
        return;
    }

    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(_parentWidget, tr("Warning"), tr("A command line tool is already running."));
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
    _cmdRunner->setFlattenOptions(dlg.flatten(), dlg.flattenIterations(), dlg.flattenDownsample());

    _cmdRunner->execute(CommandLineToolRunner::Tool::RenderTifXYZ);
    emit statusMessage(tr("Rendering segment: %1").arg(QString::fromStdString(segmentId)), 5000);
}

void SegmentationCommandHandler::onSlimFlatten(const std::string& segmentId)
{
    auto* surface = requireSurfaceAndRunner(segmentId, false);
    if (!surface) return;
    if (_cmdRunner && _cmdRunner->isRunning()) {
        QMessageBox::warning(_parentWidget, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    const std::filesystem::path segDirFs = surface->path; // tifxyz folder
    const QString segDir = QString::fromStdString(segDirFs.string());
    const QString segmentStem = QString::fromStdString(segmentId);

    const QString flatboiExe = findFlatboiExecutable();
    if (flatboiExe.isEmpty()) {
        const QString msg =
            tr("Could not find the 'flatboi' executable.\n"
               "Looked in known locations and PATH.\n\n"
               "Tip: set an override via VC.ini [tools] flatboi_path or FLATBOI env var.");
        QMessageBox::critical(_parentWidget, tr("Error"), msg);
        emit statusMessage(tr("SLIM-flatten failed"), 5000);
        return;
    }

    new SlimJob(_parentWidget, segDir, segmentStem, flatboiExe, this);
}

void SegmentationCommandHandler::onABFFlatten(const std::string& segmentId)
{
    auto surf = _state->vpkg() ? _state->vpkg()->getSurface(segmentId) : nullptr;
    if (!surf) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Cannot ABF++ flatten: Invalid segment selected"));
        return;
    }

    const std::filesystem::path segDirFs = surf->path;
    const QString segDir = QString::fromStdString(segDirFs.string());
    const QString segmentStem = QString::fromStdString(segmentId);

    // Show ABF++ flatten dialog
    ABFFlattenDialog dlg(_parentWidget);
    if (dlg.exec() != QDialog::Accepted) {
        return;
    }

    new ABFJob(_parentWidget, _surfacePanel, this, segDir, segmentStem, dlg.iterations(), dlg.downsampleFactor());
}

void SegmentationCommandHandler::onGrowSegmentFromSegment(const std::string& segmentId)
{
    auto* surface = requireSurfaceAndRunner(segmentId, true);
    if (!surface) return;

    QString srcSegment = QString::fromStdString(surface->path.string());

    std::filesystem::path volpkgPath = std::filesystem::path(_state->vpkgPath().toStdString());
    std::filesystem::path tracesDir = volpkgPath / "traces";
    std::filesystem::path jsonParamsPath = volpkgPath / "trace_params.json";
    std::filesystem::path pathsDir = volpkgPath / "paths";

    emit statusMessage(tr("Preparing to run grow_seg_from_segment..."), 2000);

    if (!std::filesystem::exists(tracesDir)) {
        try { std::filesystem::create_directory(tracesDir); }
        catch (const std::exception& e) {
            QMessageBox::warning(_parentWidget, tr("Error"), tr("Failed to create traces directory: %1").arg(e.what()));
            return;
        }
    }

    if (!std::filesystem::exists(jsonParamsPath)) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("trace_params.json not found in the volpkg"));
        return;
    }

    TraceParamsDialog dlg(_parentWidget,
                          getCurrentVolumePath(),
                          QString::fromStdString(pathsDir.string()),
                          QString::fromStdString(tracesDir.string()),
                          QString::fromStdString(jsonParamsPath.string()),
                          srcSegment);
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Run trace cancelled"), 3000);
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
            QMessageBox::warning(_parentWidget, tr("Error"), tr("Failed to write params JSON: %1").arg(mergedJsonPath));
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
    emit statusMessage(tr("Growing segment from: %1").arg(QString::fromStdString(segmentId)), 5000);
}

void SegmentationCommandHandler::onAddOverlap(const std::string& segmentId)
{
    auto* surface = requireSurfaceAndRunner(segmentId, true);
    if (!surface) return;

    std::filesystem::path volpkgPath = std::filesystem::path(_state->vpkgPath().toStdString());
    std::filesystem::path pathsDir = volpkgPath / "paths";
    QString tifxyzPath = QString::fromStdString(surface->path.string());

    _cmdRunner->setAddOverlapParams(QString::fromStdString(pathsDir.string()), tifxyzPath);
    _cmdRunner->execute(CommandLineToolRunner::Tool::SegAddOverlap);
    emit statusMessage(tr("Adding overlap for segment: %1").arg(QString::fromStdString(segmentId)), 5000);
}

void SegmentationCommandHandler::onNeighborCopyRequested(const QString& segmentId, bool copyOut)
{
    if (!_state->vpkg()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("No volume package loaded."));
        return;
    }

    if (!_cmdRunner) {
        emit statusMessage(tr("Command line tools not available"), 3000);
        return;
    }
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(_parentWidget, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    if (_neighborCopyJob && _neighborCopyJob->stage != NeighborCopyJob::Stage::None) {
        QMessageBox::warning(_parentWidget, tr("Warning"), tr("Another neighbor copy request is already running."));
        return;
    }

    auto surf = _state->vpkg()->getSurface(segmentId.toStdString());
    if (!surf) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Invalid surface selected."));
        return;
    }

    QString defaultVolumeId;
    const auto volOpts = buildVolumeOptionList(&defaultVolumeId);
    if (volOpts.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("No volumes available in the volume package."));
        return;
    }

    // Convert to NeighborCopyVolumeOption for the dialog
    QVector<NeighborCopyVolumeOption> volumeOptions;
    volumeOptions.reserve(volOpts.size());
    for (const auto& v : volOpts) {
        NeighborCopyVolumeOption opt;
        opt.id = v.id;
        opt.name = v.name;
        opt.path = v.path;
        volumeOptions.push_back(opt);
    }

    const QString surfacePath = QString::fromStdString(surf->path.string());
    QString volpkgRoot = _state->vpkgPath();
    if (volpkgRoot.isEmpty()) {
        volpkgRoot = QString::fromStdString(_state->vpkg()->getVolpkgDirectory());
    }
    QString defaultOutputDir = QDir(volpkgRoot).filePath(QStringLiteral("paths"));

    NeighborCopyDialog dlg(_parentWidget, surfacePath, volumeOptions, defaultVolumeId, defaultOutputDir);
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Copy %1 cancelled").arg(copyOut ? tr("out") : tr("in")), 3000);
        return;
    }

    QString selectedVolumePath = dlg.selectedVolumePath();
    if (selectedVolumePath.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("No target volume selected."));
        return;
    }

    QString outputDirPath = dlg.outputPath().trimmed();
    if (outputDirPath.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Output path cannot be empty."));
        return;
    }
    QDir outDir(outputDirPath);
    if (!outDir.exists() && !outDir.mkpath(".")) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Failed to create output directory: %1").arg(outputDirPath));
        return;
    }
    outputDirPath = outDir.absolutePath();

    const QString normalGridPath = QDir(volpkgRoot).filePath(QStringLiteral("normal_grids"));

    QJsonObject pass1Params;
    pass1Params["normal_grid_path"] = normalGridPath;
    pass1Params["neighbor_dir"] = copyOut ? QStringLiteral("out") : QStringLiteral("in");
    pass1Params["neighbor_max_distance"] = dlg.neighborMaxDistance();
    pass1Params["mode"] = QStringLiteral("gen_neighbor");
    pass1Params["neighbor_min_clearance"] = dlg.neighborMinClearance();
    pass1Params["neighbor_fill"] = dlg.neighborFill();
    pass1Params["neighbor_interp_window"] = dlg.neighborInterpWindow();
    pass1Params["generations"] = dlg.generations();
    pass1Params["neighbor_spike_window"] = dlg.neighborSpikeWindow();

    auto pass1JsonFile = std::make_unique<QTemporaryFile>(QDir::temp().filePath("neighbor_copy_pass1_XXXXXX.json"));
    if (!pass1JsonFile->open()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Failed to create temporary params file."));
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

    {
        QString pass2Error;
        auto extraParams = dlg.pass2TracerParamsJson(&pass2Error);
        if (!pass2Error.isEmpty()) {
            QMessageBox::warning(_parentWidget, tr("Error"), pass2Error);
            return;
        }
        if (extraParams) {
            for (auto it = extraParams->begin(); it != extraParams->end(); ++it) {
                pass2Params.insert(it.key(), it.value());
            }
        }
    }

    auto pass2JsonFile = std::make_unique<QTemporaryFile>(QDir::temp().filePath("neighbor_copy_pass2_XXXXXX.json"));
    if (!pass2JsonFile->open()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Failed to create temporary params file for pass 2."));
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
    job.pass2OmpThreads = dlg.pass2OmpThreads();
    // Snapshot current directory entries so we can detect the newly-created
    // surface after the first pass completes.
    {
        QSet<QString> entries;
        const QDir dir(outputDirPath);
        if (dir.exists()) {
            for (const auto& fi : dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot)) {
                entries.insert(fi.fileName());
            }
        }
        job.baselineEntries = std::move(entries);
    }
    job.pass1JsonFile = std::move(pass1JsonFile);
    job.pass2JsonFile = std::move(pass2JsonFile);
    job.generatedSurfacePath.clear();

    if (!startNeighborCopyPass(job.pass1JsonPath,
                               job.resumeSurfacePath,
                               QStringLiteral("skip"),
                               -1)) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Failed to launch neighbor copy pass."));
        _neighborCopyJob.reset();
        _cmdRunner->setOmpThreads(-1);
        return;
    }

    const QString dirName = QFileInfo(job.resumeSurfacePath).fileName();
    emit statusMessage(tr("Copy %1 started for %2")
                                 .arg(copyOut ? tr("out") : tr("in"))
                                 .arg(dirName.isEmpty() ? segmentId : dirName),
                             5000);
}

void SegmentationCommandHandler::onResumeLocalGrowPatchRequested(const QString& segmentId)
{
    if (!_state->vpkg()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("No volume package loaded."));
        return;
    }

    if (_state->currentVolume() == nullptr) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("No volume loaded."));
        return;
    }

    if (!_cmdRunner) {
        emit statusMessage(tr("Command line tools not available"), 3000);
        return;
    }
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(_parentWidget, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    if (_neighborCopyJob && _neighborCopyJob->stage != NeighborCopyJob::Stage::None) {
        QMessageBox::warning(_parentWidget, tr("Warning"), tr("Another neighbor copy request is already running."));
        return;
    }

    if (_resumeLocalJob) {
        QMessageBox::warning(_parentWidget, tr("Warning"), tr("A resume-opt local GrowPatch run is already active."));
        return;
    }

    auto surf = _state->vpkg()->getSurface(segmentId.toStdString());
    if (!surf) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Invalid surface selected."));
        return;
    }

    QString defaultVolumeId;
    const auto volumeOptions = buildVolumeOptionList(&defaultVolumeId);
    if (volumeOptions.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("No volumes available in the volume package."));
        return;
    }

    QString selectedVolumePath;
    std::optional<QJsonObject> extraParams;
    int ompThreads = vc3d::settings::neighbor_copy::RESUME_LOCAL_OMP_THREADS_DEFAULT;
    if (!selectResumeLocalTracerParams(_parentWidget,
                                       volumeOptions,
                                       defaultVolumeId,
                                       &selectedVolumePath,
                                       &extraParams,
                                       &ompThreads)) {
        emit statusMessage(tr("Resume-opt local GrowPatch cancelled"), 3000);
        return;
    }

    if (selectedVolumePath.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("No target volume selected."));
        return;
    }

    QString volpkgRoot = _state->vpkgPath();
    if (volpkgRoot.isEmpty()) {
        volpkgRoot = QString::fromStdString(_state->vpkg()->getVolpkgDirectory());
    }

    std::filesystem::path outputDirFs = surf->path.parent_path();
    QString outputDirPath = QString::fromStdString(outputDirFs.string());
    if (outputDirPath.isEmpty()) {
        outputDirPath = QDir(volpkgRoot).filePath(QStringLiteral("paths"));
    }
    QDir outDir(outputDirPath);
    if (!outDir.exists() && !outDir.mkpath(".")) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Failed to create output directory: %1").arg(outputDirPath));
        return;
    }
    outputDirPath = outDir.absolutePath();

    const QString normalGridPath = QDir(volpkgRoot).filePath(QStringLiteral("normal_grids"));

    QJsonObject params;
    params["normal_grid_path"] = normalGridPath;
    if (_normal3dZarrPathGetter) {
        const QString n3dPath = _normal3dZarrPathGetter();
        if (!n3dPath.isEmpty()) {
            params["normal3d_zarr_path"] = n3dPath;
        }
    }
    params["max_gen"] = 1;
    params["generations"] = 1;
    params["resume_local_opt_step"] = 20;
    params["resume_local_opt_radius"] = 40;
    params["resume_local_max_iters"] = 1000;
    params["resume_local_dense_qr"] = false;

    if (extraParams) {
        for (auto it = extraParams->begin(); it != extraParams->end(); ++it) {
            params.insert(it.key(), it.value());
        }
    }

    // Check if merged params require normal3d but we don't have it
    bool needsNormal3d = false;
    if (params.contains("normal3dline_weight")) {
        const double w = params["normal3dline_weight"].toDouble(0.0);
        needsNormal3d = (w > 0.0);
    }

    if (needsNormal3d && !params.contains("normal3d_zarr_path")) {
        auto reply = QMessageBox::warning(
            _parentWidget, tr("Missing Normal3D"),
            tr("The selected tracer profile uses normal3dline_weight > 0, "
               "but no normal3d zarr path is available.\n\n"
               "The normal3d line constraint will have no effect.\n\n"
               "To fix this, select a normal3d dataset in the segmentation panel, "
               "or use a profile without normal3dline_weight.\n\n"
               "Continue anyway?"),
            QMessageBox::Yes | QMessageBox::No,
            QMessageBox::No);
        if (reply != QMessageBox::Yes) {
            emit statusMessage(tr("Resume-opt local GrowPatch cancelled"), 3000);
            return;
        }
    }

    auto paramsFile = std::make_unique<QTemporaryFile>(QDir::temp().filePath("growpatch_resume_local_XXXXXX.json"));
    if (!paramsFile->open()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Failed to create temporary params file."));
        return;
    }
    paramsFile->write(QJsonDocument(params).toJson(QJsonDocument::Indented));
    paramsFile->flush();

    _resumeLocalJob = ResumeLocalJob{};
    auto& job = *_resumeLocalJob;
    job.segmentId = segmentId;
    job.outputDir = outputDirPath;
    job.paramsPath = paramsFile->fileName();
    job.paramsFile = std::move(paramsFile);

    _cmdRunner->setNeighborCopyParams(selectedVolumePath,
                                      job.paramsPath,
                                      QString::fromStdString(surf->path.string()),
                                      outputDirPath,
                                      QStringLiteral("local"));
    _cmdRunner->setOmpThreads(ompThreads);
    _cmdRunner->showConsoleOutput();
    _cmdRunner->execute(CommandLineToolRunner::Tool::NeighborCopy);
    emit statusMessage(tr("Resume-opt local GrowPatch started for %1").arg(segmentId), 5000);
}

void SegmentationCommandHandler::onConvertToObj(const std::string& segmentId)
{
    auto* surface = requireSurfaceAndRunner(segmentId, true);
    if (!surface) return;

    std::filesystem::path tifxyzPath = surface->path;
    std::filesystem::path objPath = tifxyzPath / (segmentId + ".obj");

    ConvertToObjDialog dlg(_parentWidget,
                           QString::fromStdString(tifxyzPath.string()),
                           QString::fromStdString(objPath.string()));
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Convert to OBJ cancelled"), 3000);
        return;
    }

    _cmdRunner->setToObjParams(dlg.tifxyzPath(), dlg.objPath());
    _cmdRunner->setOmpThreads(dlg.ompThreads());
    _cmdRunner->setToObjOptions(dlg.normalizeUV(), dlg.alignGrid());
    _cmdRunner->execute(CommandLineToolRunner::Tool::tifxyz2obj);
    emit statusMessage(tr("Converting segment to OBJ: %1").arg(QString::fromStdString(segmentId)), 5000);
}

void SegmentationCommandHandler::onCropSurfaceToValidRegion(const std::string& segmentId)
{
    auto* surface = requireSurfaceAndRunner(segmentId, false);
    if (!surface) return;

    auto surf = _state->vpkg()->getSurface(segmentId);

    cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Cannot crop surface: Missing coordinate grid"));
        return;
    }

    const int origCols = points->cols;
    const int origRows = points->rows;

    const auto boundsOpt = computeValidSurfaceBounds(*points);
    if (!boundsOpt) {
        QMessageBox::warning(_parentWidget,
                             tr("Crop failed"),
                             tr("Surface %1 does not contain any valid vertices to crop.")
                                 .arg(QString::fromStdString(segmentId)));
        return;
    }

    const cv::Rect roi = *boundsOpt;
    if (roi.x == 0 && roi.y == 0 && roi.width == origCols && roi.height == origRows) {
        emit statusMessage(
            tr("Surface %1 already occupies the tightest bounds.")
                .arg(QString::fromStdString(segmentId)),
            4000);
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
            QMessageBox::warning(_parentWidget,
                                 tr("Crop failed"),
                                 tr("Channel '%1' has size %2x%3, which is not divisible by the surface grid %4x%5.")
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
            QMessageBox::warning(_parentWidget,
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
            tempSurface->meta = std::make_unique<nlohmann::json>(*surface->meta);
        }
        for (const auto& ch : croppedChannels) {
            tempSurface->setChannel(ch.name, ch.data);
        }
        tempSurface->save(surface->path.string(), surface->id, true);
    } catch (const std::exception& ex) {
        QMessageBox::critical(_parentWidget,
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
            surface->meta = std::make_unique<nlohmann::json>(*tempSurface->meta);
        } else {
            *surface->meta = *tempSurface->meta;
        }
        if (surface->meta) {
            if (surf->meta) {
                *surf->meta = *surface->meta;
            } else {
                surf->meta = std::make_unique<nlohmann::json>(*surface->meta);
            }
        }
    }

    // Bbox will be recalculated lazily (invalidateCache was already called)

    if (_state) {
        _state->setSurface(segmentId, surf, false, false);
        if (_state->activeSurfaceId() == segmentId) {
            _state->setSurface("segmentation", surf, false, false);
        }
    }
    if (_surfacePanel) {
        _surfacePanel->refreshSurfaceMetrics(segmentId);
    }

    const QString segLabel = QString::fromStdString(segmentId);
    emit statusMessage(
        tr("Cropped %1 to %2x%3 (offset %4,%5)")
            .arg(segLabel)
            .arg(roi.width)
            .arg(roi.height)
            .arg(roi.x)
            .arg(roi.y),
        5000);
}

void SegmentationCommandHandler::onFlipSurface(const std::string& segmentId, bool flipU)
{
    auto* surface = requireSurfaceAndRunner(segmentId, false);
    if (!surface) return;

    auto surf = _state->vpkg()->getSurface(segmentId);

    if (flipU) {
        surface->flipU();
    } else {
        surface->flipV();
    }

    try {
        surface->save(surface->path.string(), surface->id, true);
    } catch (const std::exception& ex) {
        QMessageBox::critical(_parentWidget,
                              tr("Flip failed"),
                              tr("Failed to save flipped surface %1: %2")
                                  .arg(QString::fromStdString(segmentId))
                                  .arg(QString::fromUtf8(ex.what())));
        return;
    }

    if (_state) {
        _state->setSurface(segmentId, surf, false, false);
        if (_state->activeSurfaceId() == segmentId) {
            _state->setSurface("segmentation", surf, false, false);
        }
    }

    const QString axisLabel = flipU ? tr("U") : tr("V");
    emit statusMessage(
        tr("Flipped %1 over %2 axis")
            .arg(QString::fromStdString(segmentId))
            .arg(axisLabel),
        5000);
}

void SegmentationCommandHandler::onRotateSurface(const std::string& segmentId)
{
    auto* surface = requireSurfaceAndRunner(segmentId, false);
    if (!surface) return;

    auto surf = _state->vpkg()->getSurface(segmentId);
    surface->rotate(90.0f);

    try {
        surface->save(surface->path.string(), surface->id, true);
    } catch (const std::exception& ex) {
        QMessageBox::critical(_parentWidget,
                              tr("Rotate failed"),
                              tr("Failed to save rotated surface %1: %2")
                                  .arg(QString::fromStdString(segmentId))
                                  .arg(QString::fromUtf8(ex.what())));
        return;
    }

    if (_state) {
        _state->setSurface(segmentId, surf, false, false);
        if (_state->activeSurfaceId() == segmentId) {
            _state->setSurface("segmentation", surf, false, false);
        }
    }

    emit statusMessage(
        tr("Rotated %1 by 90 degrees clockwise")
            .arg(QString::fromStdString(segmentId)),
        5000);
}

void SegmentationCommandHandler::onAlphaCompRefine(const std::string& segmentId)
{
    auto* surface = requireSurfaceAndRunner(segmentId, true);
    if (!surface) return;

    QString volumePath = getCurrentVolumePath();
    if (volumePath.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Cannot refine surface: Unable to determine volume path"));
        return;
    }

    QString srcPath = QString::fromStdString(surface->path.string());
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

    AlphaCompRefineDialog dlg(_parentWidget, volumePath, srcPath, defaultOutput);
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Alpha-comp refinement cancelled"), 3000);
        return;
    }

    if (dlg.volumePath().isEmpty() || dlg.srcPath().isEmpty() || dlg.dstPath().isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Volume, source, and output paths must be specified"));
        return;
    }

    QJsonObject paramsJson = dlg.paramsJson();

    auto paramsFile = std::make_unique<QTemporaryFile>(QDir::temp().filePath("vc_objrefine_XXXXXX.json"));
    if (!paramsFile->open()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Failed to create temporary params JSON file"));
        return;
    }
    paramsFile->write(QJsonDocument(paramsJson).toJson(QJsonDocument::Indented));
    paramsFile->flush();
    QString paramsPath = paramsFile->fileName();
    paramsFile->setAutoRemove(false); // CommandLineToolRunner will use the file after this scope
    paramsFile->close();

    _cmdRunner->setObjRefineParams(dlg.volumePath(), dlg.srcPath(), dlg.dstPath(), paramsPath);
    _cmdRunner->setOmpThreads(dlg.ompThreads());
    _cmdRunner->execute(CommandLineToolRunner::Tool::AlphaCompRefine);
    emit statusMessage(tr("Refining segment: %1").arg(QString::fromStdString(segmentId)), 5000);
}

void SegmentationCommandHandler::onGrowSeeds(const std::string& segmentId, bool isExpand, bool isRandomSeed)
{
    if (_state->currentVolume() == nullptr) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Cannot grow seeds: No volume loaded"));
        return;
    }

    if (!_cmdRunner) {
        emit statusMessage(tr("Command line tools not available"), 3000);
        return;
    }
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(_parentWidget, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    std::filesystem::path volpkgPath = std::filesystem::path(_state->vpkgPath().toStdString());
    std::filesystem::path pathsDir = volpkgPath / "paths";

    if (!std::filesystem::exists(pathsDir)) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Paths directory not found in the volpkg"));
        return;
    }

    QString jsonFileName = isExpand ? "expand.json" : "seed.json";
    std::filesystem::path jsonParamsPath = volpkgPath / jsonFileName.toStdString();

    if (!std::filesystem::exists(jsonParamsPath)) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("%1 not found in the volpkg").arg(jsonFileName));
        return;
    }

    int seedX = 0, seedY = 0, seedZ = 0;
    if (!isExpand && !isRandomSeed) {
        POI *poi = _state->poi("focus");
        if (!poi) {
            QMessageBox::warning(_parentWidget, tr("Error"), tr("No focus point selected. Click on a volume with Ctrl key to set a seed point."));
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
    emit statusMessage(tr("Growing segment using %1 in %2").arg(jsonFileName).arg(modeDesc), 5000);
}

void SegmentationCommandHandler::handleNeighborCopyToolFinished(bool success)
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
            QMessageBox::warning(_parentWidget, tr("Error"),
                                 tr("Could not locate the newly generated neighbor surface in %1.")
                                     .arg(job.outputDir));
            _cmdRunner->setOmpThreads(-1);
            _neighborCopyJob.reset();
            return;
        }

        job.generatedSurfacePath = newSurface;
        job.baselineEntries.insert(QFileInfo(newSurface).fileName());
        job.stage = NeighborCopyJob::Stage::SecondPass;

        emit statusMessage(
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

    emit statusMessage(tr("Copy %1 complete: %2")
                                 .arg(copyOut ? tr("out") : tr("in"))
                                 .arg(surfaceName),
                             5000);
}

QString SegmentationCommandHandler::findNewNeighborSurface(const NeighborCopyJob& job) const
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

bool SegmentationCommandHandler::startNeighborCopyPass(const QString& paramsPath,
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

void SegmentationCommandHandler::launchNeighborCopySecondPass()
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
        _cmdRunner->setPreserveConsoleOutput(true);
        if (!startNeighborCopyPass(_neighborCopyJob->pass2JsonPath,
                                   resumeSurface,
                                   QStringLiteral("local"),
                                   std::max(1, _neighborCopyJob->pass2OmpThreads))) {
            _cmdRunner->setOmpThreads(-1);
            QMessageBox::warning(_parentWidget, tr("Error"), tr("Failed to launch the second neighbor copy pass."));
            _neighborCopyJob.reset();
            return;
        }

        emit statusMessage(
            tr("Copy %1 pass 2 running").arg(copyOut ? tr("out") : tr("in")),
            3000);
    });
}

// ---------------------------------------------------------------------------
// AWSUploadJob -- async, signal-driven S3 upload that does NOT block the GUI.
// Uploads are queued and executed one at a time via QProcess::finished.
// ---------------------------------------------------------------------------
class AWSUploadJob : public QObject {
public:
    struct UploadTask {
        QString localPath;
        QString s3Path;
        QString description;
        bool    isDirectory{false};
    };

    AWSUploadJob(QWidget* parentWidget,
                 const QString& segDir,
                 const QString& awsProfile,
                 QList<UploadTask> tasks,
                 SegmentationCommandHandler* handler)
        : QObject(handler)
        , parentWidget_(parentWidget)
        , handler_(handler)
        , segDir_(segDir)
        , awsProfile_(awsProfile)
        , tasks_(std::move(tasks))
        , proc_(new QProcess(this))
        , progress_(new QProgressDialog(
              QObject::tr("Uploading to AWS S3..."),
              QObject::tr("Cancel"), 0,
              std::max(1, static_cast<int>(tasks_.size())),
              parentWidget))
    {
        proc_->setWorkingDirectory(segDir_);
        proc_->setProcessChannelMode(QProcess::MergedChannels);

        progress_->setWindowModality(Qt::WindowModal);
        progress_->setAutoClose(false);
        progress_->setValue(0);
        progress_->setAttribute(Qt::WA_DeleteOnClose);

        QObject::connect(progress_, &QProgressDialog::canceled,
                         this, &AWSUploadJob::onCanceled_);
        QObject::connect(proc_, &QProcess::readyReadStandardOutput,
                         this, &AWSUploadJob::onStdout_);
        QObject::connect(proc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
                         this, &AWSUploadJob::onFinished_);
        QObject::connect(proc_, &QProcess::errorOccurred,
                         this, &AWSUploadJob::onProcError_);

        if (tasks_.isEmpty()) {
            showSummary_();
            return;
        }

        startNext_();
    }

private:
    void startNext_() {
        if (canceled_) return;
        lastOutput_.clear();

        // Skip tasks whose source files don't exist.
        while (taskIndex_ < tasks_.size()) {
            const auto& t = tasks_[taskIndex_];
            if (QFileInfo::exists(t.localPath) &&
                (!t.isDirectory || QFileInfo(t.localPath).isDir())) {
                break;
            }
            ++taskIndex_;
        }

        if (taskIndex_ >= tasks_.size()) {
            showSummary_();
            return;
        }

        const auto& task = tasks_[taskIndex_];

        QStringList args;
        args << "s3" << "cp" << task.localPath << task.s3Path;
        if (task.isDirectory) args << "--recursive";
        if (!awsProfile_.isEmpty()) { args << "--profile" << awsProfile_; }

        if (progress_) {
            progress_->setLabelText(QObject::tr("Uploading %1...").arg(task.description));
        }
        if (handler_) emit handler_->statusMessage(QObject::tr("Uploading %1...").arg(task.description), 0);

        proc_->start("aws", args);
    }

    void onStdout_() {
        const QByteArray raw = proc_->readAllStandardOutput();
        lastOutput_ += raw;
        const QString output = QString::fromLocal8Bit(raw);
        if (output.isEmpty()) return;
        const QStringList lines = output.split('\n', Qt::SkipEmptyParts);
        for (const QString& line : lines) {
            if (line.contains("Completed") || line.contains("upload:")) {
                const QString& desc = tasks_[taskIndex_].description;
                if (handler_) emit handler_->statusMessage(
                    QString("Uploading %1: %2").arg(desc, line.trimmed()), 0);
            }
        }
    }

    void onFinished_(int exitCode, QProcess::ExitStatus st) {
        if (canceled_) return;

        const auto& task = tasks_[taskIndex_];
        if (st == QProcess::NormalExit && exitCode == 0) {
            uploadedFiles_ << task.description;
        } else {
            const QString error = QString::fromLocal8Bit(lastOutput_).trimmed();
            failedFiles_ << QString("%1: %2").arg(task.description, error);
        }

        ++taskIndex_;
        if (progress_) progress_->setValue(taskIndex_);

        if (taskIndex_ < tasks_.size()) {
            startNext_();
        } else {
            showSummary_();
        }
    }

    void onProcError_(QProcess::ProcessError err) {
        if (canceled_) return;
        if (err == QProcess::FailedToStart) {
            const auto& task = tasks_[taskIndex_];
            failedFiles_ << QString("%1: Failed to start AWS CLI").arg(task.description);
            ++taskIndex_;
            if (progress_) progress_->setValue(taskIndex_);
            if (taskIndex_ < tasks_.size()) {
                startNext_();
            } else {
                showSummary_();
            }
        }
    }

    void onCanceled_() {
        canceled_ = true;
        if (proc_->state() != QProcess::NotRunning) {
            proc_->kill();
            proc_->waitForFinished(3000);
        }
        if (progress_) { progress_->close(); }
        if (handler_) emit handler_->statusMessage(QObject::tr("AWS upload cancelled"), 3000);
        deleteLater();
    }

    void showSummary_() {
        if (progress_) {
            progress_->setValue(progress_->maximum());
            progress_->close();
        }

        if (!uploadedFiles_.isEmpty() && failedFiles_.isEmpty()) {
            QMessageBox::information(parentWidget_, QObject::tr("Upload Complete"),
                QObject::tr("Successfully uploaded to S3:\n\n%1").arg(uploadedFiles_.join("\n")));
            if (handler_) emit handler_->statusMessage(QObject::tr("AWS upload complete"), 5000);
        } else if (!uploadedFiles_.isEmpty() && !failedFiles_.isEmpty()) {
            QMessageBox::warning(parentWidget_, QObject::tr("Partial Upload"),
                QObject::tr("Uploaded:\n%1\n\nFailed:\n%2").arg(uploadedFiles_.join("\n"), failedFiles_.join("\n")));
            if (handler_) emit handler_->statusMessage(QObject::tr("AWS upload partially complete"), 5000);
        } else if (uploadedFiles_.isEmpty() && !failedFiles_.isEmpty()) {
            QMessageBox::critical(parentWidget_, QObject::tr("Upload Failed"),
                QObject::tr("All uploads failed:\n\n%1\n\nPlease check:\n"
                   "- AWS CLI is installed\n"
                   "- AWS credentials are configured\n"
                   "- You have internet connection\n"
                   "- You have permissions for the S3 bucket").arg(failedFiles_.join("\n")));
            if (handler_) emit handler_->statusMessage(QObject::tr("AWS upload failed"), 5000);
        } else {
            QMessageBox::information(parentWidget_, QObject::tr("No Files to Upload"),
                QObject::tr("No files were uploaded."));
            if (handler_) emit handler_->statusMessage(QObject::tr("No files to upload"), 3000);
        }

        deleteLater();
    }

    QPointer<QWidget> parentWidget_;
    QPointer<SegmentationCommandHandler> handler_;
    QString segDir_;
    QString awsProfile_;
    QList<UploadTask> tasks_;
    QProcess* proc_;
    QProgressDialog* progress_;
    QStringList uploadedFiles_;
    QStringList failedFiles_;
    QByteArray lastOutput_;
    int taskIndex_{0};
    bool canceled_{false};
};

// Helper: enqueue upload tasks for one segment directory.
static void enqueueSegmentUploads(
    QList<AWSUploadJob::UploadTask>& tasks,
    const QString& targetDir,
    const QString& segmentId,
    const QString& segmentSuffix,
    const QString& selectedScroll)
{
    const QString segmentName = segmentId + segmentSuffix;
    const QString meshPath = QString("s3://vesuvius-challenge/%1/segments/meshes/%2/")
        .arg(selectedScroll, segmentName);

    const QString objFile = QDir(targetDir).filePath(segmentName + ".obj");
    tasks.append({objFile, meshPath, QString("%1.obj").arg(segmentName), false});

    const QString flatboiObjFile = QDir(targetDir).filePath(segmentName + "_flatboi.obj");
    tasks.append({flatboiObjFile, meshPath, QString("%1_flatboi.obj").arg(segmentName), false});

    const QString xTif = QDir(targetDir).filePath("x.tif");
    const QString yTif = QDir(targetDir).filePath("y.tif");
    const QString zTif = QDir(targetDir).filePath("z.tif");
    const QString metaJson = QDir(targetDir).filePath("meta.json");

    if (QFileInfo::exists(xTif) && QFileInfo::exists(yTif) &&
        QFileInfo::exists(zTif) && QFileInfo::exists(metaJson)) {
        tasks.append({xTif, meshPath, QString("%1/x.tif").arg(segmentName), false});
        tasks.append({yTif, meshPath, QString("%1/y.tif").arg(segmentName), false});
        tasks.append({zTif, meshPath, QString("%1/z.tif").arg(segmentName), false});
        tasks.append({metaJson, meshPath, QString("%1/meta.json").arg(segmentName), false});
    }

    const QString overlappingJson = QDir(targetDir).filePath("overlapping.json");
    tasks.append({overlappingJson, meshPath, QString("%1/overlapping.json").arg(segmentName), false});

    const QString layersDir = QDir(targetDir).filePath("layers");
    if (QFileInfo::exists(layersDir) && QFileInfo(layersDir).isDir()) {
        const QString surfaceVolPath = QString("s3://vesuvius-challenge/%1/segments/surface-volumes/%2/layers/")
            .arg(selectedScroll, segmentName);
        tasks.append({layersDir, surfaceVolPath, QString("%1/layers").arg(segmentName), true});
    }
}

void SegmentationCommandHandler::onAWSUpload(const std::string& segmentId)
{
    auto* surface = requireSurfaceAndRunner(segmentId, false);
    if (!surface) return;
    if (_cmdRunner && _cmdRunner->isRunning()) {
        QMessageBox::warning(_parentWidget, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    const std::filesystem::path segDirFs = surface->path;
    const QString  segDir   = QString::fromStdString(segDirFs.string());
    const QString  outTifxyz= segDir + "_flatboi";

    if (!QFileInfo::exists(segDir)) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Cannot upload to AWS: Segment directory not found"));
        return;
    }

    QStringList scrollOptions;
    scrollOptions << "PHerc0172" << "PHerc0343P" << "PHerc0500P2";

    bool ok;
    QString selectedScroll = QInputDialog::getItem(
        _parentWidget,
        tr("Select Scroll for Upload"),
        tr("Select the target scroll directory:"),
        scrollOptions,
        0, false, &ok
    );

    if (!ok || selectedScroll.isEmpty()) {
        emit statusMessage(tr("AWS upload cancelled by user"), 3000);
        return;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    QString defaultProfile = settings.value(vc3d::settings::aws::DEFAULT_PROFILE,
                                            vc3d::settings::aws::DEFAULT_PROFILE_DEFAULT).toString();

    QString awsProfile = QInputDialog::getText(
        _parentWidget, tr("AWS Profile"),
        tr("Enter AWS profile name (leave empty for default credentials):"),
        QLineEdit::Normal, defaultProfile, &ok
    );

    if (!ok) {
        emit statusMessage(tr("AWS upload cancelled by user"), 3000);
        return;
    }

    if (!awsProfile.isEmpty()) settings.setValue(vc3d::settings::aws::DEFAULT_PROFILE, awsProfile);

    // Build the upload queue (non-existent files are skipped at upload time).
    const QString segIdStr = QString::fromStdString(segmentId);
    QList<AWSUploadJob::UploadTask> tasks;
    enqueueSegmentUploads(tasks, segDir, segIdStr, QString(), selectedScroll);
    if (QFileInfo::exists(outTifxyz) && QFileInfo(outTifxyz).isDir()) {
        enqueueSegmentUploads(tasks, outTifxyz, segIdStr, QStringLiteral("_flatboi"), selectedScroll);
    }

    // AWSUploadJob is self-deleting (calls deleteLater() when finished).
    new AWSUploadJob(_parentWidget, segDir, awsProfile, std::move(tasks), this);
}

void SegmentationCommandHandler::onRasterizeSegments(const QStringList& segmentIds)
{
    if (!_state || !_state->vpkg()) {
        emit statusMessage(tr("No volume package loaded"), 3000);
        return;
    }

    if (_cmdRunner && _cmdRunner->isRunning()) {
        QMessageBox::warning(_parentWidget, tr("Warning"),
                             tr("A command line tool is already running."));
        return;
    }

    QStringList requestedIds = segmentIds;
    requestedIds.removeAll(QString());
    if (requestedIds.isEmpty()) {
        emit statusMessage(tr("No segments selected"), 3000);
        return;
    }

    QStringList rasterIds;
    QStringList validIds;
    QSet<QString> seenIds;
    for (const QString& id : requestedIds) {
        const QString normalized = id.trimmed();
        if (normalized.isEmpty()) {
            continue;
        }
        if (seenIds.contains(normalized)) {
            continue;
        }
        seenIds.insert(normalized);
        rasterIds << normalized;
    }

    if (rasterIds.isEmpty()) {
        emit statusMessage(tr("No valid segments selected"), 3000);
        return;
    }

    QStringList segmentPaths;
    QStringList missingIds;

    for (const QString& segmentId : rasterIds) {
        auto seg = _state->vpkg()->segmentation(segmentId.toStdString());
        if (!seg) {
            missingIds << segmentId;
            continue;
        }
        const auto segPath = seg->path();
        const QString segPathStr = QString::fromStdString(segPath.string());
        if (!std::filesystem::is_directory(segPath)) {
            missingIds << segmentId;
            continue;
        }
        if (!hasTifxyzMeshFiles(segPath)) {
            missingIds << segmentId;
            continue;
        }
        validIds << segmentId;
        segmentPaths << segPathStr;
    }

    if (segmentPaths.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"),
                             tr("Selected segments are not tifxyz meshes: %1")
                                 .arg(missingIds.join(QStringLiteral(", "))));
        return;
    }
    if (!missingIds.isEmpty()) {
        emit statusMessage(
            tr("Ignoring %1 segment(s) without tifxyz meshes.")
                .arg(missingIds.size()),
            3000);
    }

    const QString timestamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMddHHmmss"));
    const auto baseOutputName = QStringLiteral("labels_%1.zarr").arg(timestamp);

    std::error_code ec;
    const std::filesystem::path volpkgDir(_state->vpkg()->getVolpkgDirectory());
    const std::filesystem::path volumesDir = volpkgDir / "volumes";
    std::filesystem::create_directories(volumesDir, ec);
    if (ec) {
        QMessageBox::critical(_parentWidget, tr("Error"),
                              tr("Cannot create volumes directory: %1").arg(QString::fromStdString(ec.message())));
        return;
    }

    std::filesystem::path finalOutputRoot = volumesDir / baseOutputName.toStdString();
    for (int suffix = 1; std::filesystem::exists(finalOutputRoot, ec) && suffix < 1000; ++suffix) {
        finalOutputRoot = volumesDir /
            QStringLiteral("labels_%1_%2.zarr").arg(timestamp).arg(suffix).toStdString();
    }
    if (std::filesystem::exists(finalOutputRoot)) {
        QMessageBox::critical(_parentWidget, tr("Error"),
                              tr("Unable to reserve output directory after retries: %1")
                                  .arg(QString::fromStdString(finalOutputRoot.string())));
        return;
    }

    std::filesystem::path stagedOutputRoot =
        volumesDir / (".vc3d_rasterize_" + timestamp.toStdString());
    for (int suffix = 1; std::filesystem::exists(stagedOutputRoot, ec) && suffix < 1000; ++suffix) {
        stagedOutputRoot = volumesDir /
            QStringLiteral(".vc3d_rasterize_%1_%2").arg(timestamp).arg(suffix).toStdString();
    }

    std::filesystem::path tempRoot =
        std::filesystem::temp_directory_path() / ("vc3d_rasterize_" + timestamp.toStdString());
    if (std::filesystem::exists(tempRoot, ec)) {
        const std::string ts = timestamp.toStdString();
        for (int suffix = 1; suffix < 1000; ++suffix) {
            tempRoot = std::filesystem::temp_directory_path() /
                ("vc3d_rasterize_" + ts + "_" + std::to_string(suffix));
            if (!std::filesystem::exists(tempRoot, ec)) break;
        }
    }
    if (std::filesystem::exists(tempRoot, ec)) {
        QMessageBox::critical(_parentWidget, tr("Error"),
                              tr("Unable to reserve temporary input directory: %1")
                                  .arg(QString::fromStdString(tempRoot.string())));
        return;
    }

    if (!std::filesystem::create_directories(tempRoot, ec) || ec) {
        QMessageBox::critical(_parentWidget, tr("Error"),
                              tr("Cannot create temporary input directory: %1")
                                  .arg(QString::fromStdString(ec.message())));
        return;
    }

    for (int i = 0; i < validIds.size(); ++i) {
        const auto sourceSeg = std::filesystem::path(segmentPaths[i].toStdString());
        const auto targetSeg = tempRoot / sourceSeg.filename();

        std::error_code linkErr;
        std::filesystem::create_directory_symlink(sourceSeg, targetSeg, linkErr);
        if (linkErr) {
            std::error_code copyErr;
            std::filesystem::copy(sourceSeg, targetSeg,
                                  std::filesystem::copy_options::recursive, copyErr);
            if (copyErr) {
                QMessageBox::critical(_parentWidget, tr("Error"),
                                      tr("Failed to stage segment '%1': %2")
                                          .arg(validIds.at(i))
                                          .arg(QString::fromStdString(copyErr.message())));
                std::filesystem::remove_all(stagedOutputRoot, ec);
                std::filesystem::remove_all(tempRoot);
                return;
            }
        }
    }

    QString referenceZarr = !_normal3dZarrPathGetter ? QString() : _normal3dZarrPathGetter();
    if (referenceZarr.isEmpty()) {
        referenceZarr = getCurrentVolumePath();
    }
    if (referenceZarr.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"),
                             tr("Missing reference OME-Zarr. Load a normal3d/volume first."));
        std::filesystem::remove_all(tempRoot);
        return;
    }

    const QString executable = findVcTool("vc_tifxyz2zarr_sparse");
    if (executable.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"),
                             tr("vc_tifxyz2zarr_sparse tool not found. Configure tools/vc_tifxyz2zarr_sparse path."));
        std::filesystem::remove_all(tempRoot);
        return;
    }

    const QString tempRootStr = QString::fromStdString(tempRoot.string());
    const QString stagedOutputRootStr = QString::fromStdString(stagedOutputRoot.string());
    const QString finalOutputRootStr = QString::fromStdString(finalOutputRoot.string());
    QStringList args;
    args << tempRootStr
         << stagedOutputRootStr
         << QStringLiteral("--reference-zarr")
         << referenceZarr
         << QStringLiteral("--overwrite");
    for (const QString& segmentId : validIds) {
        args << QStringLiteral("--source-segment") << segmentId;
    }
    for (const QString& segmentPath : segmentPaths) {
        args << QStringLiteral("--source-mesh") << segmentPath;
    }

    auto runner = _cmdRunner;
    if (!runner) {
        QMessageBox::critical(_parentWidget, tr("Error"),
                              tr("Command runner is not available."));
        std::filesystem::remove_all(stagedOutputRoot);
        std::filesystem::remove_all(tempRoot);
        return;
    }

    QPointer<SegmentationCommandHandler> guard(this);
    auto connection = std::make_shared<QMetaObject::Connection>();
    *connection = connect(runner, &CommandLineToolRunner::toolFinished,
                         this,
                         [this, guard, connection, runner,
                          tempRootStr, stagedOutputRootStr, finalOutputRootStr,
                          validIds, segmentPaths](CommandLineToolRunner::Tool tool,
                                                  bool success,
                                                  const QString& message,
                                                  const QString&,
                                                  bool) {
        if (!guard) {
            disconnect(*connection);
            return;
        }
        if (tool != CommandLineToolRunner::Tool::CustomCommand) {
            return;
        }
        disconnect(*connection);

        bool finalizeOutput = false;
        if (!success) {
            QMessageBox::critical(_parentWidget, tr("Error"),
                                  tr("vc_tifxyz2zarr_sparse failed.\n%1")
                                      .arg(message));
            emit statusMessage(tr("Rasterize failed"), 3000);
        } else if (!appendRasterizationMetadata(stagedOutputRootStr, validIds, segmentPaths)) {
            emit showWarning(tr("Warning"), tr("Rasterization completed but metadata update failed"));
            emit statusMessage(tr("Rasterize complete, but metadata update failed"), 5000);
        } else {
            std::error_code renameErr;
            std::filesystem::rename(stagedOutputRootStr.toStdString(), finalOutputRootStr.toStdString(), renameErr);
            if (renameErr) {
                emit showWarning(tr("Warning"),
                                 tr("Rasterization completed, but finalizing output folder failed: %1")
                                     .arg(QString::fromStdString(renameErr.message())));
                emit statusMessage(tr("Rasterize complete, but finalizing output failed"), 5000);
            } else {
                emit statusMessage(
                    tr("Rasterized %1 segment(s) -> %2")
                        .arg(validIds.size())
                        .arg(QDir::toNativeSeparators(finalOutputRootStr)),
                    5000);
                finalizeOutput = true;
            }
        }

        std::error_code cleanupErr;
        if (!finalizeOutput) {
            std::filesystem::remove_all(std::filesystem::path(stagedOutputRootStr.toStdString()), cleanupErr);
        }
        std::filesystem::remove_all(std::filesystem::path(tempRootStr.toStdString()), cleanupErr);
    });

    if (!runner->executeCustomCommand(executable, args, QStringLiteral("vc_tifxyz2zarr_sparse"))) {
        QObject::disconnect(*connection);
        QMessageBox::critical(_parentWidget, tr("Error"),
                              tr("Failed to start vc_tifxyz2zarr_sparse."));
        std::error_code cleanupErr;
        std::filesystem::remove_all(stagedOutputRoot, cleanupErr);
        std::filesystem::remove_all(tempRoot);
        return;
    }

    emit statusMessage(
        tr("Rasterization started for %1 segment(s)...").arg(validIds.size()), 0);
}

void SegmentationCommandHandler::onExportWidthChunks(const std::string& segmentId)
{
    auto surf = _state->vpkg() ? _state->vpkg()->getSurface(segmentId) : nullptr;
    if (_state->currentVolume() == nullptr || !surf) {
        QMessageBox::warning(_parentWidget, tr("Error"),
                             tr("Cannot export: No volume or invalid segment selected"));
        return;
    }

    // Pull points and get dimensions early so we can show them in the dialog
    cv::Mat_<cv::Vec3f> points = surf->rawPoints();
    const int W = points.cols;
    const int H = points.rows;
    const cv::Vec2f sc = surf->scale();
    const double sx = (std::isfinite(sc[0]) && sc[0] > 0.0f) ? double(sc[0]) : 1.0; // guard

    if (W <= 0 || H <= 0) {
        QMessageBox::warning(_parentWidget, tr("Error"),
                             tr("Surface has invalid dimensions (%1 x %2)").arg(W).arg(H));
        return;
    }

    // Show dialog to get export parameters
    ExportChunksDialog dlg(_parentWidget, W, sx);
    if (dlg.exec() != QDialog::Accepted) {
        return;
    }

    const int chunkWidthReal = dlg.chunkWidth();
    const int overlapReal = dlg.overlapPerSide();
    const bool overwrite = dlg.overwrite();

    // Determine export root directory: <volpkg>/export (not inside paths)
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QString configuredRoot = settings.value(vc3d::settings::export_::DIR,
                                                   vc3d::settings::export_::DIR_DEFAULT).toString().trimmed();
    const QString segDir  = QString::fromStdString(surf->path.string());
    const QString segName = QString::fromStdString(segmentId);

    QString volpkgRoot = _state->vpkg() ? QString::fromStdString(_state->vpkg()->getVolpkgDirectory()) : QString();
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
        QMessageBox::critical(_parentWidget, tr("Error"),
                              tr("Cannot create export directory:\n%1").arg(exportRoot));
        return;
    }

    // Convert real pixels to grid columns
    // Example: 40k real px with scale 0.05 -> 2,000 columns per chunk
    const int chunkCols = std::max(1, int(std::llround(double(chunkWidthReal) * sx)));
    const int overlapCols = int(std::llround(double(overlapReal) * sx));

    // Calculate number of chunks: step through by chunkCols (the core width)
    const int nChunks = (W + chunkCols - 1) / chunkCols; // ceil-div purely in grid space

    if (nChunks <= 0) {
        QMessageBox::information(_parentWidget, tr("Export"), tr("Nothing to export."));
        return;
    }

    // Progress dialog
    QProgressDialog prog(tr("Exporting width-chunks..."), tr("Cancel"), 0, nChunks, _parentWidget);
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
        prog.setLabelText(tr("Exporting chunk %1 / %2...").arg(c+1).arg(nChunks));
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
            failures << QString("%1 -- %2").arg(outAbs, e.what());
        }

        QCoreApplication::processEvents();
    }
    prog.setValue(nChunks);

    // Summarize
    if (exported > 0 && failures.isEmpty()) {
        QMessageBox::information(_parentWidget, tr("Export complete"),
                                 tr("Exported %1 chunk(s) to:\n%2")
                                 .arg(exported)
                                 .arg(QDir::toNativeSeparators(exportRoot)));
        emit statusMessage(tr("Exported %1 chunk(s) -> %2")
                                 .arg(exported)
                                 .arg(QDir::toNativeSeparators(exportRoot)),
                                 5000);
    } else if (exported > 0 && !failures.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Partial export"),
                             tr("Exported %1 chunk(s), but failed:\n\n%2")
                             .arg(exported)
                             .arg(failures.join('\n')));
        emit statusMessage(tr("Export partially complete"), 5000);
    } else if (!failures.isEmpty()) {
        QMessageBox::critical(_parentWidget, tr("Export failed"),
                              tr("All chunks failed:\n\n%1").arg(failures.join('\n')));
        emit statusMessage(tr("Export failed"), 5000);
    } else {
        emit statusMessage(tr("Export cancelled"), 3000);
    }
}

bool SegmentationCommandHandler::appendRasterizationMetadata(const QString& outputZarrPath,
                                                           const QStringList& segmentIds,
                                                           const QStringList& segmentPaths) const
{
    if (outputZarrPath.isEmpty()) {
        return false;
    }

    const QDir outDir(outputZarrPath);
    if (!outDir.exists()) {
        return false;
    }

    const QString metaJsonPath = outDir.filePath(QStringLiteral("meta.json"));

    QJsonObject metaJson = readJsonObject(metaJsonPath);
    if (metaJson.isEmpty()) {
        metaJson["type"] = QStringLiteral("vol");
        metaJson["uuid"] = outDir.dirName();
        metaJson["name"] = outDir.dirName();
        metaJson["width"] = 0;
        metaJson["height"] = 0;
        metaJson["slices"] = 0;
        metaJson["voxelsize"] = 0.0;
        metaJson["min"] = 0.0;
        metaJson["max"] = 255.0;
        metaJson["format"] = QStringLiteral("zarr");
    }

    QJsonArray idArray;
    QJsonArray pathArray;
    for (const QString& id : segmentIds) {
        idArray.append(id);
    }
    for (const QString& p : segmentPaths) {
        pathArray.append(p);
    }

    const QJsonValue rasterizedAt = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    metaJson.insert(QStringLiteral("label_volume"), QStringLiteral("rasterized"));
    metaJson.insert(QStringLiteral("source_segments"), idArray);
    metaJson.insert(QStringLiteral("source_meshes"), pathArray);
    metaJson.insert(QStringLiteral("source_mesh_count"), static_cast<int>(segmentIds.size()));
    metaJson.insert(QStringLiteral("rasterized_at"), rasterizedAt);
    metaJson.insert(QStringLiteral("rasterizer"), QStringLiteral("vc_tifxyz2zarr_sparse"));

    if (!writeJsonObject(metaJsonPath, metaJson)) {
        return false;
    }

    return true;
}

void SegmentationCommandHandler::onReloadFromBackup(const QString& segmentId, int backupIndex)
{
    if (!_state->vpkg()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("No volume package loaded."));
        return;
    }

    const std::string segIdStd = segmentId.toStdString();
    auto surf = _state->vpkg()->getSurface(segIdStd);
    if (!surf) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Surface not found: %1").arg(segmentId));
        return;
    }

    // Build paths
    namespace fs = std::filesystem;
    fs::path volpkgRoot = _state->vpkg()->getVolpkgDirectory();
    fs::path backupDir = volpkgRoot / "backups" / segIdStd / std::to_string(backupIndex);
    fs::path segmentDir = surf->path;

    if (!fs::exists(backupDir)) {
        QMessageBox::warning(_parentWidget, tr("Error"),
            tr("Backup directory does not exist: %1").arg(QString::fromStdString(backupDir.string())));
        return;
    }

    if (!fs::exists(segmentDir)) {
        QMessageBox::warning(_parentWidget, tr("Error"),
            tr("Segment directory does not exist: %1").arg(QString::fromStdString(segmentDir.string())));
        return;
    }

    // Confirm with user
    QMessageBox::StandardButton reply = QMessageBox::question(
        _parentWidget,
        tr("Confirm Reload from Backup"),
        tr("This will replace the current segment '%1' with backup %2.\n\n"
           "The current segment data will be overwritten. Continue?")
           .arg(segmentId).arg(backupIndex),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);

    if (reply != QMessageBox::Yes) {
        emit statusMessage(tr("Reload from backup cancelled"), 3000);
        return;
    }

    // Files to copy from backup
    std::vector<std::string> filesToCopy = {
        "mesh.ply",
        "mask.tif",
        "meta.json",
        "generations.tif"
    };

    std::error_code ec;
    int copiedCount = 0;

    for (const auto& filename : filesToCopy) {
        fs::path srcFile = backupDir / filename;
        fs::path dstFile = segmentDir / filename;

        if (fs::exists(srcFile)) {
            // Remove existing file first
            if (fs::exists(dstFile)) {
                fs::remove(dstFile, ec);
                if (ec) {
                    QMessageBox::warning(_parentWidget, tr("Error"),
                        tr("Failed to remove existing file %1: %2")
                           .arg(QString::fromStdString(dstFile.string()))
                           .arg(QString::fromStdString(ec.message())));
                    return;
                }
            }

            // Copy from backup
            fs::copy_file(srcFile, dstFile, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                QMessageBox::warning(_parentWidget, tr("Error"),
                    tr("Failed to copy %1: %2")
                       .arg(QString::fromStdString(filename))
                       .arg(QString::fromStdString(ec.message())));
                return;
            }
            copiedCount++;
        }
    }

    if (copiedCount == 0) {
        QMessageBox::warning(_parentWidget, tr("Error"),
            tr("No files found in backup directory."));
        return;
    }

    // Reload the surface
    bool wasSelected = (_state->activeSurfaceId() == segIdStd);

    if (_state->vpkg()->reloadSingleSegmentation(segIdStd)) {
        try {
            auto reloadedSurf = _state->vpkg()->loadSurface(segIdStd);
            if (reloadedSurf) {
                if (_state) {
                    _state->setSurface(segIdStd, reloadedSurf, false, false);
                }

                if (_surfacePanel) {
                    _surfacePanel->refreshSurfaceMetrics(segIdStd);
                }

                if (wasSelected) {
                    _state->setActiveSurface(segIdStd, std::dynamic_pointer_cast<QuadSurface>(reloadedSurf));

                    if (_state) {
                        _state->setSurface("segmentation", reloadedSurf, false, false);
                    }

                    if (_surfacePanel) {
                        _surfacePanel->syncSelectionUi(segIdStd, reloadedSurf.get());
                    }
                }

                emit statusMessage(
                    tr("Restored '%1' from backup %2 (%3 files)")
                       .arg(segmentId).arg(backupIndex).arg(copiedCount),
                    5000);
            }
        } catch (const std::exception& e) {
            QMessageBox::critical(_parentWidget, tr("Error"),
                tr("Failed to reload surface after restore: %1")
                   .arg(QString::fromUtf8(e.what())));
        }
    } else {
        QMessageBox::warning(_parentWidget, tr("Warning"),
            tr("Files were copied but failed to reload the segmentation. "
               "Try using the reload button."));
    }
}

void SegmentationCommandHandler::onMoveSegmentToPaths(const QString& segmentId)
{
    if (!_state->vpkg()) {
        emit statusMessage(tr("No volume package loaded"), 3000);
        return;
    }

    // Verify we're in traces directory
    if (_state->vpkg()->getSegmentationDirectory() != "traces") {
        emit statusMessage(tr("Can only move segments from traces directory"), 3000);
        return;
    }

    // Get the segment
    auto seg = _state->vpkg()->segmentation(segmentId.toStdString());
    if (!seg) {
        emit statusMessage(tr("Segment not found: %1").arg(segmentId), 3000);
        return;
    }

    // Build paths
    std::filesystem::path volpkgPath(_state->vpkg()->getVolpkgDirectory());
    std::filesystem::path currentPath = seg->path();
    std::filesystem::path newPath = volpkgPath / "paths" / currentPath.filename();

    // Check if destination exists
    if (std::filesystem::exists(newPath)) {
        QMessageBox::StandardButton reply = QMessageBox::question(
            _parentWidget,
            tr("Destination Exists"),
            tr("Segment '%1' already exists in paths/.\nDo you want to replace it?").arg(segmentId),
            QMessageBox::Yes | QMessageBox::No,
            QMessageBox::No
        );

        if (reply != QMessageBox::Yes) {
            return;
        }

        // Remove the existing one
        try {
            std::filesystem::remove_all(newPath);
        } catch (const std::exception& e) {
            QMessageBox::critical(_parentWidget, tr("Error"),
                tr("Failed to remove existing segment: %1").arg(e.what()));
            return;
        }
    }

    // Confirm the move
    QMessageBox::StandardButton reply = QMessageBox::question(
        _parentWidget,
        tr("Move to Paths"),
        tr("Move segment '%1' from traces/ to paths/?\n\n"
           "Note: The segment will be closed if currently open.").arg(segmentId),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::Yes
    );

    if (reply != QMessageBox::Yes) {
        return;
    }

    // === CRITICAL: Clean up the segment before moving ===
    std::string idStd = segmentId.toStdString();

    // Check if this is the currently selected segment
    bool wasSelected = (_state->activeSurfaceId() == idStd);

    // Clear from surface collection (including "segmentation" if it matches)
    if (_state) {
        auto currentSurface = _state->surface(idStd);
        auto segmentationSurface = _state->surface("segmentation");

        // If this surface is currently shown as "segmentation", clear it
        if (currentSurface && segmentationSurface && currentSurface == segmentationSurface) {
            _state->setSurface("segmentation", nullptr, false, false);
        }

        // Clear the surface from the collection
        _state->setSurface(idStd, nullptr, false, false);
    }

    // Unload the surface from VolumePkg
    _state->vpkg()->unloadSurface(idStd);

    // Clear selection if this was selected
    if (wasSelected) {
        if (_clearSelectionCallback) {
            _clearSelectionCallback();
        }
    }

    // Perform the move
    try {
        std::filesystem::rename(currentPath, newPath);

        // Remove from VolumePkg's internal tracking for traces
        _state->vpkg()->removeSingleSegmentation(idStd);

        // The inotify system will pick up the IN_MOVED_TO in paths/
        // and handle adding it there if the user switches to that directory

        if (_surfacePanel) {
            _surfacePanel->removeSingleSegmentation(idStd);
        }

        emit statusMessage(
            tr("Moved %1 from traces/ to paths/. Switch to paths directory to see it.").arg(segmentId), 5000);

    } catch (const std::exception& e) {
        // If move failed, we might want to reload the segment
        // but it's probably safer to leave it unloaded
        QMessageBox::critical(_parentWidget, tr("Error"),
            tr("Failed to move segment: %1\n\n"
               "The segment has been unloaded from the viewer.").arg(e.what()));
    }
}

void SegmentationCommandHandler::onRenameSurface(const QString& segmentId)
{
    if (!_state->vpkg()) {
        emit statusMessage(tr("No volume package loaded"), 3000);
        return;
    }

    // Block if surface is currently being edited
    if (_isEditingCheck && _isEditingCheck()) {
        QMessageBox::warning(_parentWidget, tr("Cannot Rename"),
            tr("Cannot rename surface while editing is in progress.\n"
               "Please finish or cancel editing first."));
        return;
    }

    // Get the segment
    std::string oldId = segmentId.toStdString();
    auto seg = _state->vpkg()->segmentation(oldId);
    if (!seg) {
        emit statusMessage(tr("Segment not found: %1").arg(segmentId), 3000);
        return;
    }

    // Show input dialog to get new name
    bool ok = false;
    QString newName = QInputDialog::getText(
        _parentWidget,
        tr("Rename Surface"),
        tr("Enter new name for '%1':").arg(segmentId),
        QLineEdit::Normal,
        segmentId,
        &ok);

    if (!ok || newName.isEmpty()) {
        return;
    }

    std::string newId = newName.toStdString();

    // Validate new name: alphanumeric + underscore + hyphen only
    static const QRegularExpression validNameRegex(QStringLiteral("^[a-zA-Z0-9_-]+$"));
    if (!validNameRegex.match(newName).hasMatch()) {
        QMessageBox::warning(_parentWidget, tr("Invalid Name"),
            tr("Surface name can only contain letters, numbers, underscores, and hyphens."));
        return;
    }

    // Check if name is unchanged
    if (newId == oldId) {
        return;
    }

    // Check for name collision
    std::filesystem::path volpkgPath(_state->vpkg()->getVolpkgDirectory());
    std::filesystem::path currentPath = seg->path();
    std::filesystem::path parentDir = currentPath.parent_path();
    std::filesystem::path newPath = parentDir / newId;

    if (std::filesystem::exists(newPath)) {
        QMessageBox::warning(_parentWidget, tr("Name Exists"),
            tr("A surface with the name '%1' already exists.").arg(newName));
        return;
    }

    // Check if this is the currently selected segment
    bool wasSelected = (_state->activeSurfaceId() == oldId);

    // Store the old UUID for rollback if needed
    std::string oldUuid = seg->id();

    // === Clean up the segment before renaming ===

    // Wait for any pending index rebuild
    if (_waitForIndexRebuildCallback) {
        _waitForIndexRebuildCallback();
    }

    // Clear from surface collection (including "segmentation" if it matches)
    if (_state) {
        auto currentSurface = _state->surface(oldId);
        auto segmentationSurface = _state->surface("segmentation");

        // If this surface is currently shown as "segmentation", clear it
        if (currentSurface && segmentationSurface && currentSurface == segmentationSurface) {
            _state->setSurface("segmentation", nullptr, false, false);
        }

        // Clear the surface from the collection
        _state->setSurface(oldId, nullptr, false, false);
    }

    // Unload the surface from VolumePkg
    _state->vpkg()->unloadSurface(oldId);

    // Clear selection if this was selected
    if (wasSelected) {
        if (_clearSelectionCallback) {
            _clearSelectionCallback();
        }
    }

    // Update meta.json UUID
    try {
        seg->setId(newId);
        seg->saveMetadata();
    } catch (const std::exception& e) {
        QMessageBox::critical(_parentWidget, tr("Error"),
            tr("Failed to update metadata: %1").arg(e.what()));
        // Reload the old segment
        _state->vpkg()->refreshSegmentations();
        if (_surfacePanel) {
            _surfacePanel->reloadSurfacesFromDisk();
        }
        return;
    }

    // Perform the folder rename
    try {
        std::filesystem::rename(currentPath, newPath);

        // Remove old ID from VolumePkg's internal tracking
        _state->vpkg()->removeSingleSegmentation(oldId);

        // Remove from surface panel
        if (_surfacePanel) {
            _surfacePanel->removeSingleSegmentation(oldId);
        }

        // Refresh segmentations to pick up the new ID
        _state->vpkg()->refreshSegmentations();

        // Add the new segment
        if (_surfacePanel) {
            _surfacePanel->addSingleSegmentation(newId);
        }

        // Restore selection if it was the selected surface
        if (wasSelected && _restoreSelectionCallback) {
            _restoreSelectionCallback(newId);
        }

        emit statusMessage(
            tr("Renamed '%1' to '%2'").arg(segmentId, newName), 5000);

    } catch (const std::exception& e) {
        // Attempt to rollback metadata change
        try {
            seg->setId(oldUuid);
            seg->saveMetadata();
        } catch (...) {
            // Rollback failed - metadata is now inconsistent
        }

        QMessageBox::critical(_parentWidget, tr("Error"),
            tr("Failed to rename folder: %1\n\n"
               "The segment has been unloaded. Please reload surfaces.").arg(e.what()));

        // Refresh to get back to a consistent state
        _state->vpkg()->refreshSegmentations();
        if (_surfacePanel) {
            _surfacePanel->reloadSurfacesFromDisk();
        }
    }
}

void SegmentationCommandHandler::onCopySurfaceRequested(const QString& segmentId)
{
    if (!_state->vpkg()) {
        emit statusMessage(tr("No volume package loaded"), 3000);
        return;
    }

    // Block if surface is currently being edited
    if (_isEditingCheck && _isEditingCheck()) {
        QMessageBox::warning(_parentWidget, tr("Cannot Copy"),
            tr("Cannot copy surface while editing is in progress.\n"
               "Please finish or cancel editing first."));
        return;
    }

    // Get the segment
    std::string oldId = segmentId.toStdString();
    auto seg = _state->vpkg()->segmentation(oldId);
    if (!seg) {
        emit statusMessage(tr("Segment not found: %1").arg(segmentId), 3000);
        return;
    }

    std::filesystem::path currentPath = seg->path();
    std::filesystem::path parentDir = currentPath.parent_path();

    QString baseName = segmentId + "_copy";
    QString suggestedName = baseName;
    int suffix = 1;
    while (std::filesystem::exists(parentDir / suggestedName.toStdString())) {
        ++suffix;
        suggestedName = QString("%1_%2").arg(baseName).arg(suffix);
    }

    bool ok = false;
    QString newName = QInputDialog::getText(
        _parentWidget,
        tr("Copy Surface"),
        tr("Enter name for copy of '%1':").arg(segmentId),
        QLineEdit::Normal,
        suggestedName,
        &ok);

    if (!ok) {
        return;
    }

    newName = newName.trimmed();
    if (newName.isEmpty()) {
        return;
    }

    // Validate new name: alphanumeric + underscore + hyphen only
    static const QRegularExpression validNameRegex(QStringLiteral("^[a-zA-Z0-9_-]+$"));
    if (!validNameRegex.match(newName).hasMatch()) {
        QMessageBox::warning(_parentWidget, tr("Invalid Name"),
            tr("Surface name can only contain letters, numbers, underscores, and hyphens."));
        return;
    }

    std::string newId = newName.toStdString();
    if (newId == oldId) {
        return;
    }

    std::filesystem::path newPath = parentDir / newId;
    if (std::filesystem::exists(newPath)) {
        QMessageBox::warning(_parentWidget, tr("Name Exists"),
            tr("A surface with the name '%1' already exists.").arg(newName));
        return;
    }

    try {
        std::filesystem::copy(currentPath, newPath, std::filesystem::copy_options::recursive);
    } catch (const std::exception& e) {
        QMessageBox::critical(_parentWidget, tr("Error"),
            tr("Failed to copy surface: %1").arg(e.what()));
        return;
    }

    try {
        auto copiedSeg = Segmentation::New(newPath);
        copiedSeg->setId(newId);
        copiedSeg->setName(newId);
        copiedSeg->saveMetadata();
    } catch (const std::exception& e) {
        try {
            std::filesystem::remove_all(newPath);
        } catch (...) {
            // Best-effort cleanup only
        }
        QMessageBox::critical(_parentWidget, tr("Error"),
            tr("Failed to update metadata for copied surface: %1").arg(e.what()));
        return;
    }

    if (_state->vpkg()->addSingleSegmentation(newId)) {
        if (_surfacePanel) {
            _surfacePanel->addSingleSegmentation(newId);
        }
    } else {
        _state->vpkg()->refreshSegmentations();
        if (_surfacePanel) {
            _surfacePanel->reloadSurfacesFromDisk();
        }
    }

    emit statusMessage(
        tr("Copied '%1' to '%2'").arg(segmentId, newName), 5000);
}

// Include the MOC file for Q_OBJECT classes in anonymous namespace
#include "SegmentationCommandHandler.moc"
