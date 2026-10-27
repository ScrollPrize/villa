#include "SegmentationCommandHandler.hpp"

#include "CState.hpp"
#include "SurfacePanelController.hpp"
#include "SurfaceCatalog.hpp"
#include "SurfaceAreaCalculator.hpp"
#include "VCSettings.hpp"
#include "OpenDataNormalGrids.hpp"
#include "OpenDataManifest.hpp"
#include <vc_object_store/Uuid.hpp>

#include <functional>
#include <algorithm>
#include <iostream>
#include <thread>
#include <cmath>
#include <optional>
#include <atomic>
#include <vector>
#include <memory>
#include <filesystem>
#include <limits>
#include <mutex>

#include <QSettings>
#include <QMessageBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QCoreApplication>
#include <QDateTime>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QInputDialog>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QVBoxLayout>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QFile>
#include <QTextStream>
#include <QtGlobal>
#include <QProgressDialog>
#include <QPointer>
#include <QTimer>
#include <QTemporaryFile>
#include <QHash>
#include <QSet>
#include <QVector>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QComboBox>
#include <QToolButton>
#include <QVector3D>
#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
#include <QStandardPaths>
#endif

#include "CommandLineToolRunner.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include "ToolDialogs.hpp"
#include "ToolExecutable.hpp"
#include "ToolProcess.hpp"
#include "elements/VolumeSelector.hpp"
#include "elements/JsonProfilePresets.hpp"
#include "utils/Json.hpp"

static QString commandPathForVolume(const std::shared_ptr<Volume>& volume)
{
    if (!volume) {
        return QString();
    }
    if (volume->isRemote()) {
        return QString::fromStdString(volume->remoteLocator());
    }
    return QString::fromStdString(volume->path().string());
}

struct CommandLaunchErrorSetter {
    CommandLaunchError* error;

    void operator()(
        const QString& message,
        CommandLaunchError::Kind kind = CommandLaunchError::Other) const
    {
        if (error)
            *error = {kind, message};
    }
};

static QString findFlatboiExecutable()
{
    return vc3d::findToolExecutable(
        QStringLiteral("flatboi"),
        {QStringLiteral("/usr/local/bin/flatboi"),
         QStringLiteral("/home/builder/vc-dependencies/bin/flatboi")},
        QStringLiteral("FLATBOI"));
}

namespace { // -------------------- anonymous namespace -------------------------


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
            if (!vc::geometry::validSurfacePoint(points(r, c))) {
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

bool selectGrowPatchSeedParams(QWidget* parent,
                               const QVector<VolumeSelector::VolumeOption>& volumeOptions,
                               QString* selectedVolumeId,
                               QString* selectedVolumePath,
                               int* iterations,
                               double* minAreaCm,
                               QString* outputTag)
{
    QDialog dlg(parent);
    dlg.setWindowTitle(QObject::tr("Create Segment (GrowPatch)"));

    auto* layout = new QVBoxLayout(&dlg);
    auto* form = new QFormLayout();

    auto* volumeCombo = new QComboBox(&dlg);
    for (const auto& option : volumeOptions) {
        const QString label = option.name.isEmpty()
            ? option.id
            : QStringLiteral("%1 (%2)").arg(option.name, option.id);
        volumeCombo->addItem(label, option.id);
        volumeCombo->setItemData(volumeCombo->count() - 1, option.path, Qt::UserRole + 1);
    }
    if (selectedVolumeId && !selectedVolumeId->isEmpty()) {
        const int idx = volumeCombo->findData(*selectedVolumeId);
        if (idx >= 0) {
            volumeCombo->setCurrentIndex(idx);
        }
    }
    form->addRow(QObject::tr("Volume:"), volumeCombo);

    auto* iterationsSpin = new QSpinBox(&dlg);
    iterationsSpin->setRange(1, 100000);
    iterationsSpin->setValue(iterations ? std::clamp(*iterations, 1, 100000) : 200);
    form->addRow(QObject::tr("Iterations:"), iterationsSpin);

    auto* minAreaSpin = new QDoubleSpinBox(&dlg);
    minAreaSpin->setRange(0.0, 1000000.0);
    minAreaSpin->setDecimals(6);
    minAreaSpin->setSingleStep(0.001);
    minAreaSpin->setValue(minAreaCm ? std::clamp(*minAreaCm, 0.0, 1000000.0) : 0.002);
    minAreaSpin->setToolTip(QObject::tr("Writes the min_area_cm parameter for vc_grow_seg_from_seed."));
    form->addRow(QObject::tr("Min size:"), minAreaSpin);

    auto* tagEdit = new QLineEdit(outputTag ? *outputTag : QStringLiteral("paths"), &dlg);
    tagEdit->setToolTip(QObject::tr("Tag for the new surface."));
    form->addRow(QObject::tr("Output tag:"), tagEdit);

    layout->addLayout(form);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
    layout->addWidget(buttons);

    QObject::connect(buttons, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dlg, [&]() {
        const QString tag = tagEdit->text().trimmed();
        if (tag.isEmpty()) {
            QMessageBox::warning(&dlg, QObject::tr("Create Segment"), QObject::tr("Output tag cannot be empty."));
            return;
        }
        if (volumeCombo->currentIndex() < 0) {
            QMessageBox::warning(&dlg, QObject::tr("Create Segment"), QObject::tr("Select a volume."));
            return;
        }
        if (selectedVolumeId) {
            *selectedVolumeId = volumeCombo->currentData().toString();
        }
        if (selectedVolumePath) {
            *selectedVolumePath = volumeCombo->currentData(Qt::UserRole + 1).toString();
        }
        if (iterations) {
            *iterations = iterationsSpin->value();
        }
        if (minAreaCm) {
            *minAreaCm = minAreaSpin->value();
        }
        if (outputTag) {
            *outputTag = tag;
        }
        dlg.accept();
    });

    return dlg.exec() == QDialog::Accepted;
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

struct RasterizeDialogResult {
    int chunkSize{128};
    int zMin{0};
    int zMax{-1};
};

static bool selectRasterizeParams(QWidget* parent, RasterizeDialogResult* out)
{
    if (!out) {
        return false;
    }

    QDialog dlg(parent);
    dlg.setWindowTitle(QObject::tr("Rasterize"));

    auto* main = new QVBoxLayout(&dlg);
    auto* form = new QFormLayout();

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const int savedChunkSize = std::clamp(
        settings.value(QStringLiteral("tools/rasterize_chunk_size"), 128).toInt(),
        1, 4096);
    const int savedZMin = std::max(0, settings.value(QStringLiteral("tools/rasterize_z_min"), 0).toInt());
    const int savedZMax = settings.value(QStringLiteral("tools/rasterize_z_max"), -1).toInt();

    auto* spChunkSize = new QSpinBox(&dlg);
    spChunkSize->setRange(1, 4096);
    spChunkSize->setValue(savedChunkSize);
    spChunkSize->setToolTip(QObject::tr("Isotropic chunk size applied to all pyramid levels."));
    form->addRow(QObject::tr("Chunk size:"), spChunkSize);

    auto* spZMin = new QSpinBox(&dlg);
    spZMin->setRange(0, 1000000000);
    spZMin->setValue(savedZMin);
    form->addRow(QObject::tr("Level-0 Z min:"), spZMin);

    auto* spZMax = new QSpinBox(&dlg);
    spZMax->setRange(-1, 1000000000);
    spZMax->setValue(savedZMax);
    spZMax->setToolTip(QObject::tr("-1 means the end of the level-0 volume."));
    form->addRow(QObject::tr("Level-0 Z max (exclusive):"), spZMax);

    main->addLayout(form);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dlg, [&]() {
        if (spZMax->value() >= 0 && spZMin->value() >= spZMax->value()) {
            QMessageBox::warning(&dlg,
                                 QObject::tr("Error"),
                                 QObject::tr("Z min must be smaller than Z max unless Z max is -1."));
            return;
        }
        settings.setValue(QStringLiteral("tools/rasterize_chunk_size"), spChunkSize->value());
        settings.setValue(QStringLiteral("tools/rasterize_z_min"), spZMin->value());
        settings.setValue(QStringLiteral("tools/rasterize_z_max"), spZMax->value());
        dlg.accept();
    });
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
    main->addWidget(buttons);

    if (dlg.exec() != QDialog::Accepted) {
        return false;
    }

    out->chunkSize = spChunkSize->value();
    out->zMin = spZMin->value();
    out->zMax = spZMax->value();
    return true;
}

static bool isRasterizedLabelVolumePath(const QString& volumePath)
{
    if (volumePath.isEmpty()) {
        return false;
    }
    const QString metaPath = QDir(volumePath).filePath(QStringLiteral("meta.json"));
    const QJsonObject meta = readJsonObject(metaPath);
    return meta.value(QStringLiteral("label_volume")).toString() == QStringLiteral("rasterized");
}

struct IgnoreLabelDialogResult {
    QString volumePath;
    QString outputName;
    int ignoreValue{2};
    double chunkAlphaL0{64.0};
    int workers{0};
    int zMin{0};
    int zMax{-1};
};

struct IgnoreLabelProgressState {
    QString pendingOutput;
    QString stageName;
    int stageLevel{-1};
    int totalSteps{0};
};

static bool parseStructuredProgressLine(const QString& line,
                                        QString* kindOut,
                                        QHash<QString, QString>* fieldsOut)
{
    if (!kindOut || !fieldsOut) {
        return false;
    }

    const QString trimmed = line.trimmed();
    if (!trimmed.startsWith(QStringLiteral("VC_"))) {
        return false;
    }

    const QStringList tokens = trimmed.split(QChar::Space, Qt::SkipEmptyParts);
    if (tokens.isEmpty()) {
        return false;
    }

    *kindOut = tokens.front();
    fieldsOut->clear();
    for (int i = 1; i < tokens.size(); ++i) {
        const QString& token = tokens.at(i);
        const int eq = token.indexOf(QChar('='));
        if (eq <= 0 || eq >= token.size() - 1) {
            continue;
        }
        fieldsOut->insert(token.left(eq), token.mid(eq + 1));
    }
    return true;
}

static int clampProgressCount(qint64 value)
{
    if (value <= 0) {
        return 0;
    }
    if (value > std::numeric_limits<int>::max()) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(value);
}

static QString ignoreLabelStageText(const QString& stageName,
                                    int stageLevel,
                                    int current,
                                    int total)
{
    QString base;
    if (stageName == QStringLiteral("copy")) {
        base = QObject::tr("Copying input tree");
    } else if (stageName == QStringLiteral("reuse")) {
        base = QObject::tr("Reusing existing output tree");
    } else if (stageName == QStringLiteral("wrap")) {
        base = QObject::tr("Wrapping level-0 chunks");
    } else if (stageName == QStringLiteral("pyramid")) {
        base = stageLevel >= 0
            ? QObject::tr("Building pyramid level %1").arg(stageLevel)
            : QObject::tr("Building pyramid");
    } else if (stageName == QStringLiteral("slice")) {
        base = QObject::tr("Processing slices");
    } else {
        base = QObject::tr("Running %1").arg(stageName);
    }

    if (total > 0 && current >= 0) {
        return QObject::tr("%1 (%2 / %3)").arg(base).arg(current).arg(total);
    }
    return base;
}

bool selectIgnoreLabelParams(QWidget* parent,
                             const QVector<VolumeSelector::VolumeOption>& volumes,
                             const QString& defaultVolumeId,
                             IgnoreLabelDialogResult* out)
{
    if (!out) {
        return false;
    }

    QDialog dlg(parent);
    dlg.setWindowTitle(QObject::tr("Add Ignore Label"));

    auto* main = new QVBoxLayout(&dlg);
    auto* volumeSelector = new VolumeSelector(&dlg);
    volumeSelector->setVolumes(volumes, defaultVolumeId);
    main->addWidget(volumeSelector);

    auto* form = new QFormLayout();

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QString timestamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMddHHmmss"));
    QString defaultOut = settings.value(QStringLiteral("tools/add_ignore_label_output_name"),
                                        QStringLiteral("labels_ignore_%1.zarr").arg(timestamp))
                             .toString()
                             .trimmed();
    if (defaultOut.isEmpty()) {
        defaultOut = QStringLiteral("labels_ignore_%1.zarr").arg(timestamp);
    }
    if (!defaultOut.endsWith(QStringLiteral(".zarr"), Qt::CaseInsensitive)) {
        defaultOut += QStringLiteral(".zarr");
    }

    auto* edtOutput = new QLineEdit(defaultOut, &dlg);
    form->addRow(QObject::tr("Output folder name:"), edtOutput);

    auto* spIgnore = new QSpinBox(&dlg);
    spIgnore->setRange(1, 254);
    spIgnore->setValue(settings.value(QStringLiteral("tools/add_ignore_label_ignore_value"), 2).toInt());
    form->addRow(QObject::tr("Ignore value:"), spIgnore);

    auto* lblMode = new QLabel(
        QObject::tr("Runs 3D chunk alpha-wrap on level 0 and labels only the outer region."),
        &dlg);
    lblMode->setWordWrap(true);
    form->addRow(QString(), lblMode);

    auto* spChunkAlpha = new QDoubleSpinBox(&dlg);
    spChunkAlpha->setDecimals(1);
    spChunkAlpha->setSingleStep(1.0);
    spChunkAlpha->setRange(1.0, 4096.0);
    spChunkAlpha->setValue(
        settings.value(QStringLiteral("tools/add_ignore_label_chunk_alpha_l0"), 64.0).toDouble());
    spChunkAlpha->setToolTip(
        QObject::tr("Absolute alpha-wrap radius in level-0 voxels."));
    form->addRow(QObject::tr("Wrap radius (L0 voxels):"), spChunkAlpha);

    auto* spWorkers = new QSpinBox(&dlg);
    spWorkers->setRange(0, 256);
    spWorkers->setValue(settings.value(QStringLiteral("tools/add_ignore_label_workers"), 0).toInt());
    spWorkers->setToolTip(QObject::tr("0 uses all available hardware threads."));
    form->addRow(QObject::tr("Workers (0=auto):"), spWorkers);

    auto* spZMin = new QSpinBox(&dlg);
    spZMin->setRange(0, 1000000000);
    spZMin->setValue(settings.value(QStringLiteral("tools/add_ignore_label_z_min"), 0).toInt());
    form->addRow(QObject::tr("Level-0 Z min:"), spZMin);

    auto* spZMax = new QSpinBox(&dlg);
    spZMax->setRange(-1, 1000000000);
    spZMax->setValue(settings.value(QStringLiteral("tools/add_ignore_label_z_max"), -1).toInt());
    spZMax->setToolTip(QObject::tr("-1 means the end of the level-0 volume."));
    form->addRow(QObject::tr("Level-0 Z max (exclusive):"), spZMax);

    main->addLayout(form);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dlg, [&]() {
        const QString outName = edtOutput->text().trimmed();
        if (outName.isEmpty()) {
            QMessageBox::warning(&dlg,
                                 QObject::tr("Error"),
                                 QObject::tr("Output folder name cannot be empty."));
            return;
        }
        settings.setValue(QStringLiteral("tools/add_ignore_label_output_name"), outName);
        settings.setValue(QStringLiteral("tools/add_ignore_label_ignore_value"), spIgnore->value());
        settings.setValue(QStringLiteral("tools/add_ignore_label_chunk_alpha_l0"), spChunkAlpha->value());
        settings.setValue(QStringLiteral("tools/add_ignore_label_workers"), spWorkers->value());
        settings.setValue(QStringLiteral("tools/add_ignore_label_z_min"), spZMin->value());
        settings.setValue(QStringLiteral("tools/add_ignore_label_z_max"), spZMax->value());
        dlg.accept();
    });
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
    main->addWidget(buttons);

    if (dlg.exec() != QDialog::Accepted) {
        return false;
    }

    out->volumePath = volumeSelector->selectedVolumePath();
    out->outputName = edtOutput->text().trimmed();
    if (!out->outputName.endsWith(QStringLiteral(".zarr"), Qt::CaseInsensitive)) {
        out->outputName += QStringLiteral(".zarr");
    }
    out->ignoreValue = spIgnore->value();
    out->chunkAlphaL0 = spChunkAlpha->value();
    out->workers = spWorkers->value();
    out->zMin = spZMin->value();
    out->zMax = spZMax->value();
    return !out->volumePath.isEmpty();
}

bool selectResumeLocalTracerParams(QWidget* parent,
                                   const QVector<VolumeSelector::VolumeOption>& volumes,
                                   const QString& defaultVolumeId,
                                   QString* selectedVolumeId,
                                   std::optional<QJsonObject>* paramsOut,
                                   int* ompThreadsOut)
{
    if (!paramsOut || !selectedVolumeId || !ompThreadsOut) {
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

    *selectedVolumeId = volumeSelector->selectedVolumeId();
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

// A flattening tool run on a stored surface (SLIM by flatboi, ABF++ by
// vc_flatten, vc_straighten): the tool reads the surface from the store and
// creates the result there. Shows a progress dialog, reports the run through
// flattenJobStarted/flattenJobFinished exactly once (an empty message means
// cancelled), and deletes itself.
class FlattenJob : public QObject {
public:
    struct Spec {
        QString tool;         // executable path
        QString store, volume;  // the store contract (ToolProcess::Launch)
        QStringList args;
        QString kind;         // flattenJobStarted's kind, e.g. "flatten.slim"
        QString title;        // "SLIM-flatten", "ABF++ flatten", "Straighten"
        QString running;      // the progress dialog's first label
        QString outputName;
        // A determinate bar over this many iterations, fed by "PROGRESS n / m"
        // and "[it n]" lines; 0: indeterminate, the label follows the output.
        int iterations = 0;
        bool modal = false;
        // Recompute the created surface's area (the surface list's column).
        bool computeArea = false;
    };

    FlattenJob(QWidget* parentWidget, SegmentationCommandHandler* handler, SurfaceCatalog* catalog, Spec spec,
               bool suppressDialogs)
        : QObject(handler)
        , parentWidget_(parentWidget)
        , handler_(handler)
        , catalog_(catalog)
        , spec_(std::move(spec))
        , suppressDialogs_(suppressDialogs)
        , run_(new vc3d::ToolProcess(this))
        , progress_(new QProgressDialog(spec_.running, QObject::tr("Cancel"), 0, 0, parentWidget))
    {
        progress_->setWindowModality(spec_.modal ? Qt::WindowModal : Qt::NonModal);
        progress_->setAutoClose(false);
        // Suppressed dialogs use a huge minimum duration so the auto-show timer
        // never fires. Updates below remain harmless.
        progress_->setMinimumDuration(suppressDialogs_ ? std::numeric_limits<int>::max() : 0);
        progress_->setRange(0, spec_.iterations > 0 ? spec_.iterations + 1 : 0);
        progress_->setAttribute(Qt::WA_DeleteOnClose);
        connect(progress_, &QProgressDialog::canceled, run_, &vc3d::ToolProcess::cancel);
        connect(run_, &vc3d::ToolProcess::output, this, [](const QString& chunk) { std::cout << chunk.toStdString() << std::flush; });
        connect(run_, &vc3d::ToolProcess::line, this, [this](const QString& line) { onLine_(line.trimmed()); });
        connect(run_, &vc3d::ToolProcess::finished, this, [this](const vc3d::ToolProcess::Result& result) { onFinished_(result); });

        if (handler_) {
            emit handler_->statusMessage(spec_.running, 0);
            emit handler_->flattenJobStarted(spec_.kind, spec_.outputName);
        }
        vc3d::ToolProcess::Launch launch;
        launch.tool = spec_.tool;
        launch.args = spec_.args;
        launch.store = spec_.store;
        launch.volume = spec_.volume;
        if (!run_->start(launch)) {
            vc3d::ToolProcess::Result failed;
            failed.error = run_->error();
            onFinished_(failed);
            return;
        }
        std::cout << "[" << spec_.kind.toStdString() << "] launching: " << run_->command().toStdString() << std::endl;
    }

private:
    void onLine_(const QString& line) {
        if (line.isEmpty() || !progress_) return;
        static const QRegularExpression progressRe(R"(^PROGRESS\s+(\d+)\s*/\s*(\d+)$)", QRegularExpression::CaseInsensitiveOption);
        static const QRegularExpression iterationRe(R"(^\[it\s+(\d+)\])", QRegularExpression::CaseInsensitiveOption);
        if (spec_.iterations <= 0) {
            progress_->setLabelText(line);
            if (handler_) emit handler_->statusMessage(line, 0);
        } else if (const auto m = progressRe.match(line); m.hasMatch()) {
            const int total = m.captured(2).toInt();
            if (total > 0 && total != spec_.iterations) {
                spec_.iterations = total;
                progress_->setMaximum(total + 1);
            }
            showIteration_(m.captured(1).toInt());
        } else if (const auto m = iterationRe.match(line); m.hasMatch()) {
            lastIteration_ = std::max(lastIteration_, m.captured(1).toInt());
            showIteration_(lastIteration_);
        } else if (line.startsWith(QStringLiteral("Final stretch"))) {
            progress_->setLabelText(QObject::tr("Creating the flattened surface..."));
            if (handler_) emit handler_->statusMessage(line, 0);
        }
    }

    void showIteration_(int iteration) {
        progress_->setLabelText(QObject::tr("Iterations: %1 / %2").arg(iteration).arg(spec_.iterations));
        progress_->setValue(std::clamp(iteration, 0, spec_.iterations));
    }

    void onFinished_(const vc3d::ToolProcess::Result& result) {
        if (progress_) progress_->close();
        const QString created = result.surfaces.isEmpty() ? QString() : result.surfaces.back();
        if (result.cancelled) {
            report_(false, QString(), QString(), QObject::tr("%1 cancelled").arg(spec_.title));
            return;
        }
        if (!result.ok || created.isEmpty()) {
            const QString why = result.ok ? QObject::tr("%1 reported no surface.").arg(QFileInfo(spec_.tool).baseName())
                                          : result.error;
            const QString detail = why + (result.tail.isEmpty() ? QString() : "\n\n" + result.tail);
            report_(false, detail, QString(), QObject::tr("%1 failed").arg(spec_.title));
            if (!suppressDialogs_) QMessageBox::critical(parentWidget_, QObject::tr("%1 Failed").arg(spec_.title), detail);
            return;
        }
        if (spec_.computeArea && catalog_) {
            for (const auto& area : SurfaceAreaCalculator::calculateAreas(*catalog_, {created.toStdString()}))
                if (!area.success)
                    std::cout << "[" << spec_.kind.toStdString() << "] warning: no area for " << created.toStdString()
                              << ": " << area.errorReason << std::endl;
        }
        const QString done = QObject::tr("%1 complete: %2").arg(spec_.title, spec_.outputName);
        report_(true, done, created, done);
        if (!suppressDialogs_)
            QMessageBox::information(parentWidget_, QObject::tr("%1 Complete").arg(spec_.title),
                                     QObject::tr("Surface created: %1").arg(spec_.outputName));
    }

    void report_(bool success, const QString& message, const QString& created, const QString& status) {
        if (handler_) {
            emit handler_->statusMessage(status, 5000);
            emit handler_->flattenJobFinished(success, message, created);
        }
        deleteLater();
    }

    QPointer<QWidget> parentWidget_;
    QPointer<SegmentationCommandHandler> handler_;
    QPointer<SurfaceCatalog> catalog_;
    Spec spec_;
    bool suppressDialogs_ = false;
    int lastIteration_ = 0;
    vc3d::ToolProcess* run_ = nullptr;
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
    return commandPathForVolume(_state->currentVolume());
}

namespace {
// Whole-surface transforms rewrite every channel: load the ones still in the store.
void loadAllChannels(QuadSurface& surface)
{
    for (const auto& name : surface.channelNames()) {
        (void)surface.channel(name, SURF_CHANNEL_NORESIZE);
    }
}
}

std::shared_ptr<QuadSurface> SegmentationCommandHandler::activeSurface(const std::string& segmentId) const
{
    auto surface = _state ? _state->activeSurface().lock() : nullptr;
    return surface && surface->id == segmentId ? surface : nullptr;
}

bool SegmentationCommandHandler::publishSurface(QuadSurface* surface, const QString& action)
{
    if (_surfaceSaver && _surfaceSaver(surface)) {
        return true;
    }
    QMessageBox::critical(_parentWidget, tr("%1 failed").arg(action),
                          tr("Could not save surface %1 to the object store.").arg(QString::fromStdString(surface->id)));
    return false;
}

QuadSurface* SegmentationCommandHandler::requireSurfaceAndRunner(
    const std::string& segmentId,
    bool checkRunner)
{
    if (_state->currentVolume() == nullptr) {
        QMessageBox::warning(_parentWidget, tr("Error"),
                             tr("No volume loaded."));
        return nullptr;
    }

    auto surf = activeSurface(segmentId);
    if (!surf) {
        QMessageBox::warning(_parentWidget, tr("Error"),
                             tr("Select surface %1 first.")
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

    if (_editingDestinationResolver && !_editingDestinationResolver(surf)) return nullptr;

    // The active surface stays alive while it is CState's active surface.
    return surf.get();
}

std::optional<SegmentationCommandHandler::StoredSurface>
SegmentationCommandHandler::storeForTools(CommandLaunchError* error) const
{
    const CommandLaunchErrorSetter setErr{error};
    auto* catalog = _surfacePanel ? _surfacePanel->catalog() : nullptr;
    if (!catalog) {
        setErr(tr("Attach an object store and open a volume first."), CommandLaunchError::InvalidState);
        return std::nullopt;
    }
    try {
        const auto tools = catalog->toolStore();
        StoredSurface stored;
        stored.volumeUuid = QString::fromStdString(tools.volume);
        stored.store = QString::fromStdString(tools.store.string());
        return stored;
    } catch (const std::exception& e) {
        setErr(QString::fromUtf8(e.what()), CommandLaunchError::InvalidState);
        return std::nullopt;
    }
}

std::optional<SegmentationCommandHandler::StoredSurface>
SegmentationCommandHandler::storedSurface(const std::string& id, CommandLaunchError* error, bool throughView) const
{
    const CommandLaunchErrorSetter setErr{error};
    auto stored = storeForTools(error);
    if (!stored) return std::nullopt;
    const auto* row = _surfacePanel->catalog()->row(id);
    if (!row) {
        setErr(tr("Surface not in the object store: %1").arg(QString::fromStdString(id)),
               CommandLaunchError::SegmentNotFound);
        return std::nullopt;
    }
    if (row->catalog) {
        setErr(tr("%1 is a read-only catalog surface; copy it to your store first.")
                   .arg(QString::fromStdString(row->record.name.empty() ? id : row->record.name)),
               CommandLaunchError::InvalidState);
        return std::nullopt;
    }
    if (throughView ? !row->bound() : !row->toolInput()) {
        setErr((throughView ? tr("Surface %1 does not open in the current volume.")
                            : tr("Surface %1 is not stored in the current volume's voxels; tools read stored coordinates."))
                   .arg(QString::fromStdString(id)),
               CommandLaunchError::InvalidState);
        return std::nullopt;
    }
    stored->uuid = id;
    stored->name = QString::fromStdString(row->record.name.empty() ? id : row->record.name);
    return stored;
}

std::optional<SegmentationCommandHandler::StoredSurface>
SegmentationCommandHandler::requireStoredSurface(const std::string& id, bool checkRunner, bool throughView)
{
    CommandLaunchError error;
    auto stored = storedSurface(id, &error, throughView);
    if (!stored) {
        QMessageBox::warning(_parentWidget, tr("Error"), error.message);
        return std::nullopt;
    }
    if (checkRunner) {
        if (!_cmdRunner) {
            emit statusMessage(tr("Command line tools not available"), 3000);
            return std::nullopt;
        }
        if (_cmdRunner->isRunning()) {
            QMessageBox::warning(_parentWidget, tr("Warning"), tr("A command line tool is already running."));
            return std::nullopt;
        }
    }
    return stored;
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
        opt.path = commandPathForVolume(volume);
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

void SegmentationCommandHandler::configureCommandRunnerForVolume(const std::string& volumeId)
{
    if (!_cmdRunner) {
        return;
    }
    _cmdRunner->setRemoteVolumeUrl(QString());
    _cmdRunner->setRemoteVolumeAuth(QString(), QString(), QString(), QString());
    if (!_state || !_state->vpkg() || !_state->vpkg()->hasVolume(volumeId)) {
        return;
    }
    const auto volume = _state->vpkg()->volume(volumeId);
    if (!volume->isRemote()) {
        return;
    }
    const auto& auth = volume->remoteAuth();
    _cmdRunner->setRemoteVolumeUrl(QString::fromStdString(volume->remoteLocator()));
    _cmdRunner->setRemoteVolumeAuth(QString::fromStdString(auth.access_key),
                                    QString::fromStdString(auth.secret_key),
                                    QString::fromStdString(auth.session_token),
                                    QString::fromStdString(auth.region));
}

void SegmentationCommandHandler::onRenderSegment(const std::string& segmentId)
{
    const auto stored = requireStoredSurface(segmentId, false, true);
    if (!stored) return;

    QString artifacts;
    try {
        auto* catalog = _surfacePanel->catalog();
        artifacts = QString::fromStdString(catalog->session()->artifactDirectory(*catalog->attachment(), stored->uuid).string());
    } catch (const std::exception& e) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("No artifact directory for the surface: %1").arg(e.what()));
        return;
    }
    const float scale = 1.0f;
    const int resolution = 0;
    const int layers = 31;
    const QString outputPattern = QDir(artifacts).filePath(QStringLiteral("layers"));

    QString defaultVolumeId;
    const auto volumes = buildVolumeOptionList(&defaultVolumeId);
    RenderParamsDialog dlg(_parentWidget, volumes, defaultVolumeId, stored->name, outputPattern, scale, resolution, layers);
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Render cancelled"), 3000);
        return;
    }

    RenderSegmentParams params;
    params.volumeId = dlg.volumeId();
    // Trim once: the dialog enables its Zarr options on the trimmed path, so the
    // command has to use the same value. A .zarr output path asks for the zarr
    // store (which "Also write TIFF slices (Zarr)" attaches to), else a TIFF stack.
    params.outputPattern = dlg.outputPattern().trimmed();
    params.outputFormat = params.outputPattern.endsWith(QStringLiteral(".zarr"), Qt::CaseInsensitive)
        ? CommandLineToolRunner::RenderOutputFormat::Zarr
        : CommandLineToolRunner::RenderOutputFormat::TifStack;
    params.scale = static_cast<float>(dlg.scale());
    params.groupIdx = dlg.groupIdx();
    params.numSlices = dlg.numSlices();
    params.ompThreads = dlg.ompThreads();
    params.cropX = dlg.cropX();
    params.cropY = dlg.cropY();
    params.cropWidth = dlg.cropWidth();
    params.cropHeight = dlg.cropHeight();
    params.rotateDegrees = dlg.rotateDegrees();
    params.flipAxis = dlg.flipAxis();
    params.includeTifs = dlg.includeTifs();
    params.flatten = dlg.flatten();
    params.flattenIterations = dlg.flattenIterations();
    params.flattenDownsample = dlg.flattenDownsample();
    CommandLaunchError error;
    if (!startRenderImpl(segmentId, params, /*interactive=*/true, &error, nullptr)) {
        QMessageBox::warning(_parentWidget, tr("Error"), error.message);
        return;
    }
    emit statusMessage(tr("Rendering surface: %1").arg(stored->name), 5000);
}

void SegmentationCommandHandler::onSlimFlatten(const std::string& segmentId)
{
    const auto stored = requireStoredSurface(segmentId, false, true);
    if (!stored) return;

    SlimFlattenDialog dlg(_parentWidget, stored->name + QStringLiteral("_flatboi"));
    if (dlg.exec() != QDialog::Accepted) {
        return;
    }
    SlimFlattenParams params;
    params.iterations = dlg.maxIterations();
    params.tolerance = dlg.tolerance();
    params.energyType = dlg.energyType();
    params.keepPercent = dlg.keepPercent();
    params.inpaintHoles = dlg.inpaintHoles();
    params.outputName = dlg.outputName();

    const QByteArray pastixEnv = qgetenv("PASTIX_NUM_THREADS");
    std::cout << "[slim-flatten] surface=" << segmentId
              << " out=" << params.outputName.toStdString()
              << " iters=" << params.iterations
              << " tol=" << params.tolerance
              << " energy=" << params.energyType.toStdString()
              << " keep_percent=" << params.keepPercent
              << " inpaint=" << (params.inpaintHoles ? "true" : "false")
              << " PASTIX_NUM_THREADS=" << (pastixEnv.isEmpty() ? "<unset, PaStiX auto>" : pastixEnv.toStdString())
              << " hardware_concurrency=" << std::thread::hardware_concurrency()
              << std::endl;

    CommandLaunchError error;
    if (!startSlimFlattenImpl(segmentId, params, /*interactive=*/true, &error, nullptr)) {
        QMessageBox::critical(_parentWidget, tr("Error"), error.message);
        emit statusMessage(tr("SLIM-flatten failed"), 5000);
    }
}

void SegmentationCommandHandler::onStraighten(const std::string& segmentId)
{
    const auto stored = requireStoredSurface(segmentId, false, true);
    if (!stored) return;

    StraightenDialog dlg(_parentWidget, stored->name + QStringLiteral("_straightened"));
    if (dlg.exec() != QDialog::Accepted) {
        return;
    }
    CommandLaunchError error;
    if (!startStraightenImpl(segmentId, dlg.request(), /*interactive=*/true, &error, nullptr)) {
        QMessageBox::critical(_parentWidget, tr("Error"), error.message);
        emit statusMessage(tr("Straighten failed"), 5000);
    }
}

void SegmentationCommandHandler::onABFFlatten(const std::string& segmentId)
{
    const auto stored = requireStoredSurface(segmentId, false, true);
    if (!stored) return;

    ABFFlattenDialog dlg(_parentWidget);
    if (dlg.exec() != QDialog::Accepted) {
        return;
    }
    startAbfFlattenImpl(segmentId, dlg.iterations(), dlg.downsampleFactor(), /*interactive=*/true, nullptr, nullptr);
}

void SegmentationCommandHandler::onGrowSegmentFromSegment(const std::string& segmentId)
{
    // The dialog collects inputs; startRunTrace owns validation and launch.
    const auto stored = requireStoredSurface(segmentId, true, true);
    if (!stored) return;
    if (_state && _state->currentVolume() && _state->currentVolume()->isRemote()) {
        QMessageBox::warning(
            _parentWidget,
            tr("Unsupported Remote Volume"),
            tr("Run Trace uses vc_grow_seg_from_segments, which accepts only local volumes. "
               "The remote volume locator was not modified or passed to the tool."));
        return;
    }

    const std::filesystem::path jsonParamsPath =
        std::filesystem::path(_state->vpkgPath().toStdString()) / "trace_params.json";
    if (!std::filesystem::exists(jsonParamsPath)) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("trace_params.json not found in the volpkg"));
        return;
    }

    RunTraceParams params;
    TraceParamsDialog dlg(_parentWidget,
                          getCurrentVolumePath(),
                          params.sourceTag,
                          params.outputTag,
                          QString::fromStdString(jsonParamsPath.string()),
                          stored->name);
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Run trace cancelled"), 3000);
        return;
    }

    params.paramOverrides = dlg.makeParamsJson();
    params.ompThreads = dlg.ompThreads();
    params.sourceTag = dlg.sourceTag();
    params.outputTag = dlg.outputTag();

    CommandLaunchError error;
    if (!startRunTraceImpl(segmentId, params, /*interactive=*/true, &error)) {
        QMessageBox::warning(_parentWidget, tr("Error"), error.message);
        return;
    }

    emit statusMessage(tr("Growing surface from: %1").arg(stored->name), 5000);
}

bool SegmentationCommandHandler::startRunTrace(const std::string& segmentId,
                                              const RunTraceParams& params,
                                              CommandLaunchError* error)
{
    return startRunTraceImpl(segmentId, params, /*interactive=*/false, error);
}

bool SegmentationCommandHandler::startRunTraceImpl(
    const std::string& segmentId,
    const RunTraceParams& params,
    bool interactive,
    CommandLaunchError* error)
{
    const CommandLaunchErrorSetter setErr{error};

    // Validate without opening dialogs.
    if (!_state || _state->currentVolume() == nullptr || !_state->vpkg()) {
        setErr(tr("No volume package or volume loaded."), CommandLaunchError::InvalidState);
        return false;
    }
    const auto stored = storedSurface(segmentId, error, true);
    if (!stored) return false;
    if (!_cmdRunner) {
        setErr(tr("Command line tools not available"), CommandLaunchError::ToolUnavailable);
        return false;
    }
    if (_cmdRunner->isRunning()) {
        setErr(tr("A command line tool is already running."), CommandLaunchError::Busy);
        return false;
    }
    if (_state->currentVolume()->isRemote()) {
        setErr(tr("Run Trace uses vc_grow_seg_from_segments, which accepts only "
                  "local volumes (remote current volume rejected)."),
               CommandLaunchError::RemoteVolume);
        return false;
    }
    if (params.sourceTag.isEmpty() || params.outputTag.isEmpty()) {
        setErr(tr("Run Trace needs a source tag and an output tag."));
        return false;
    }

    const std::filesystem::path jsonParamsPath =
        std::filesystem::path(_state->vpkgPath().toStdString()) / "trace_params.json";
    if (!std::filesystem::exists(jsonParamsPath)) {
        setErr(tr("trace_params.json not found in the volpkg"),
               CommandLaunchError::InputNotFound);
        return false;
    }

    // Merge overrides over trace_params.json and write the launch copy.
    QJsonObject base;
    {
        QFile f(QString::fromStdString(jsonParamsPath.string()));
        if (f.open(QIODevice::ReadOnly)) {
            const auto doc = QJsonDocument::fromJson(f.readAll());
            f.close();
            if (doc.isObject()) base = doc.object();
        }
    }
    for (auto it = params.paramOverrides.begin(); it != params.paramOverrides.end(); ++it)
        base[it.key()] = it.value();

    auto paramsFile = std::make_unique<QTemporaryFile>(
        QDir::temp().filePath(QStringLiteral("vc_trace_params_XXXXXX.json")));
    paramsFile->setAutoRemove(true);
    if (!paramsFile->open()) {
        setErr(tr("Failed to write params JSON: %1").arg(paramsFile->errorString()));
        return false;
    }
    paramsFile->write(QJsonDocument(base).toJson(QJsonDocument::Indented));
    paramsFile->flush();
    _traceParamsFile = std::move(paramsFile);

    _cmdRunner->setStore(stored->store);
    _cmdRunner->setTraceParams(
        getCurrentVolumePath(),
        stored->volumeUuid,
        params.sourceTag,
        params.outputTag,
        _traceParamsFile->fileName(),
        QString::fromStdString(stored->uuid));
    _cmdRunner->setNextOmpThreads(params.ompThreads);
    const auto options = interactive
        ? CommandLineToolRunner::ExecutionOptions{}
        : CommandLineToolRunner::ExecutionOptions::silent();
    if (!_cmdRunner->execute(
            CommandLineToolRunner::Tool::GrowSegFromSegment, options)) {
        setErr(tr("Failed to start Run Trace."));
        return false;
    }
    return true;
}

bool SegmentationCommandHandler::startRenderSegment(const std::string& segmentId,
                                                    const RenderSegmentParams& params,
                                                    CommandLaunchError* error,
                                                    QString* resolvedOutputDir)
{
    return startRenderImpl(segmentId, params, /*interactive=*/false, error, resolvedOutputDir);
}

bool SegmentationCommandHandler::startRenderImpl(const std::string& segmentId,
                                                 const RenderSegmentParams& params,
                                                 bool interactive,
                                                 CommandLaunchError* error,
                                                 QString* resolvedOutputDir)
{
    const CommandLaunchErrorSetter setErr{error};

    // Validate without opening dialogs.
    if (!_state || !_state->vpkg()) {
        setErr(tr("No volume package or volume loaded."), CommandLaunchError::InvalidState);
        return false;
    }
    if (!_state->currentVolume()) {
        setErr(tr("No volume loaded."), CommandLaunchError::InvalidState);
        return false;
    }
    const auto stored = storedSurface(segmentId, error, true);
    if (!stored) return false;
    if (!_cmdRunner) {
        setErr(tr("Command line tools not available"), CommandLaunchError::ToolUnavailable);
        return false;
    }
    if (_cmdRunner->isRunning()) {
        setErr(tr("A command line tool is already running."), CommandLaunchError::Busy);
        return false;
    }

    // Return missing-tool errors before execute() reaches its dialog path.
    if (vc3d::findToolExecutable(QStringLiteral("vc_render_tifxyz")).isEmpty()) {
        setErr(tr("vc_render_tifxyz not found or not executable"), CommandLaunchError::ToolUnavailable);
        return false;
    }

    // Rendering accepts local paths and remote locators.
    std::shared_ptr<Volume> renderVolume;
    if (params.volumeId.isEmpty()) {
        renderVolume = _state->currentVolume();
    } else {
        const auto ids = _state->vpkg()->volumeIDs();
        if (std::find(ids.begin(), ids.end(), params.volumeId.toStdString()) == ids.end()) {
            setErr(tr("Unknown volume id: %1").arg(params.volumeId),
                   CommandLaunchError::VolumeNotFound);
            return false;
        }
        renderVolume = _state->vpkg()->volume(params.volumeId.toStdString());
    }
    const QString volumePath = commandPathForVolume(renderVolume);
    if (volumePath.isEmpty()) {
        setErr(tr("Could not resolve a volume path for rendering."));
        return false;
    }

    // An explicit output pattern, else one in the output base (relative to the
    // package, or the surface's artifacts), created if needed.
    QString outputPattern = params.outputPattern;
    if (outputPattern.isEmpty()) {
        const std::filesystem::path volpkgPath(_state->vpkgPath().toStdString());
        std::filesystem::path baseDir;
        try {
            if (params.outputDir.isEmpty()) {
                auto* catalog = _surfacePanel->catalog();
                baseDir = catalog->session()->artifactDirectory(*catalog->attachment(), stored->uuid);
            } else {
                std::filesystem::path requested(params.outputDir.toStdString());
                baseDir = requested.is_absolute() ? requested : (volpkgPath / requested);
            }
            std::filesystem::create_directories(baseDir);
        } catch (const std::exception& e) {
            setErr(tr("Failed to create output directory: %1").arg(e.what()));
            return false;
        }
        const bool wantZarr = params.outputFormat == CommandLineToolRunner::RenderOutputFormat::Zarr;
        outputPattern = QString::fromStdString((baseDir / (wantZarr ? "surface.zarr" : "layers")).string());
    }

    // Every option is set, so earlier settings cannot leak in.
    _cmdRunner->setStore(stored->store);
    _cmdRunner->setSurface(QString::fromStdString(stored->uuid));
    _cmdRunner->setOutputPattern(outputPattern);
    _cmdRunner->setRenderOutputFormat(params.outputFormat);
    _cmdRunner->setRenderParams(params.scale, params.groupIdx, params.numSlices);

    if (params.hasVoxelSize) {
        _cmdRunner->setRenderVoxelSize(params.voxelSizeUm, true);
    } else {
        _cmdRunner->setRenderVoxelSize(
            renderVolume ? renderVolume->voxelSize() : 0.0,
            renderVolume &&
                (renderVolume->baseScaleLevel() > 0 ||
                 renderVolume->hasExplicitVoxelSizeOverride()));
    }

    _cmdRunner->setVolumePath(volumePath);
    configureCommandRunnerForVolume(renderVolume->id());

    _cmdRunner->setRenderVolume(QString::fromStdString(renderVolume->id()));
    _cmdRunner->setNextOmpThreads(params.ompThreads);
    _cmdRunner->setRenderAdvanced(params.cropX, params.cropY, params.cropWidth, params.cropHeight,
                                  params.rotateDegrees, params.flipAxis);
    _cmdRunner->setIncludeTifs(params.includeTifs);
    _cmdRunner->setFlattenOptions(params.flatten, params.flattenIterations, params.flattenDownsample);
    if (!_cmdRunner->execute(
            CommandLineToolRunner::Tool::RenderTifXYZ,
            interactive ? CommandLineToolRunner::ExecutionOptions{}
                        : CommandLineToolRunner::ExecutionOptions::silent())) {
        setErr(tr("Failed to start render."));
        return false;
    }

    if (resolvedOutputDir)
        *resolvedOutputDir = outputPattern;
    return true;
}

bool SegmentationCommandHandler::startSlimFlatten(const std::string& segmentId,
                                                  const SlimFlattenParams& params,
                                                  CommandLaunchError* error,
                                                  QString* resolvedOutputName)
{
    return startSlimFlattenImpl(segmentId, params, /*interactive=*/false, error, resolvedOutputName);
}

bool SegmentationCommandHandler::startSlimFlattenImpl(const std::string& segmentId,
                                                      const SlimFlattenParams& params,
                                                      bool interactive,
                                                      CommandLaunchError* error,
                                                      QString* resolvedOutputName)
{
    const CommandLaunchErrorSetter setErr{error};

    const auto stored = storedSurface(segmentId, error, true);
    if (!stored) return false;

    const QString flatboiExe = findFlatboiExecutable();
    if (flatboiExe.isEmpty()) {
        setErr(tr("flatboi not found or not executable (checked known locations, "
                  "PATH, VC.ini [tools] flatboi_path, and $FLATBOI)."),
               CommandLaunchError::ToolUnavailable);
        return false;
    }

    const QString outputName = params.outputName.isEmpty() ? stored->name + QStringLiteral("_flatboi") : params.outputName;
    const int iters = params.iterations > 0 ? params.iterations : 50;
    QStringList args{QStringLiteral("--surface"), QString::fromStdString(stored->uuid),
                     QStringLiteral("--name"), outputName};
    if (params.keepPercent < 100.0) args << QStringLiteral("--keep=%1").arg(params.keepPercent, 0, 'f', 4);
    if (params.inpaintHoles) args << QStringLiteral("--inpaint");
    if (params.tolerance > 0.0) args << QStringLiteral("--tol=%1").arg(params.tolerance, 0, 'g', 8);
    args << QString::number(iters) << params.energyType;

    new FlattenJob(_parentWidget, this, _surfacePanel->catalog(),
                   {flatboiExe, stored->store, stored->volumeUuid, args, QStringLiteral("flatten.slim"), tr("SLIM-flatten"), tr("Running SLIM (flatboi)..."),
                    outputName, iters, /*modal=*/true, /*computeArea=*/true},
                   /*suppressDialogs=*/!interactive);

    if (resolvedOutputName) *resolvedOutputName = outputName;
    return true;
}

bool SegmentationCommandHandler::startAbfFlatten(const std::string& segmentId,
                                                 int iterations,
                                                 int downsampleFactor,
                                                 CommandLaunchError* error,
                                                 QString* resolvedOutputName)
{
    return startAbfFlattenImpl(segmentId, iterations, downsampleFactor, /*interactive=*/false, error, resolvedOutputName);
}

bool SegmentationCommandHandler::startAbfFlattenImpl(const std::string& segmentId,
                                                     int iterations,
                                                     int downsampleFactor,
                                                     bool interactive,
                                                     CommandLaunchError* error,
                                                     QString* resolvedOutputName)
{
    const CommandLaunchErrorSetter setErr{error};
    const auto stored = storedSurface(segmentId, error, true);
    if (!stored) return false;

    const QString exe = vc3d::findToolExecutable(QStringLiteral("vc_flatten"));
    if (exe.isEmpty()) {
        setErr(tr("Could not find the 'vc_flatten' executable (known locations and PATH). "
                  "Set it in VC.ini [tools] vc_flatten, or put it on PATH."),
               CommandLaunchError::ToolUnavailable);
        return false;
    }
    const QString outputName = stored->name + QStringLiteral("_abf");
    const QStringList args{QStringLiteral("--surface"), QString::fromStdString(stored->uuid),
                           QStringLiteral("--name"), outputName,
                           QStringLiteral("--iterations"), QString::number(std::max(1, iterations)),
                           QStringLiteral("--downsample"), QString::number(std::max(1, downsampleFactor))};
    new FlattenJob(_parentWidget, this, _surfacePanel->catalog(),
                   {exe, stored->store, stored->volumeUuid, args, QStringLiteral("flatten.abf"), tr("ABF++ flatten"), tr("ABF++ Flattening..."), outputName},
                   /*suppressDialogs=*/!interactive);

    if (resolvedOutputName) *resolvedOutputName = outputName;
    return true;
}

bool SegmentationCommandHandler::startStraighten(const std::string& segmentId,
                                                 const StraightenParams& params,
                                                 CommandLaunchError* error,
                                                 QString* resolvedOutputName)
{
    return startStraightenImpl(segmentId, params, /*interactive=*/false, error, resolvedOutputName);
}

bool SegmentationCommandHandler::startStraightenImpl(const std::string& segmentId,
                                                     const StraightenParams& params,
                                                     bool interactive,
                                                     CommandLaunchError* error,
                                                     QString* resolvedOutputName)
{
    const CommandLaunchErrorSetter setErr{error};

    const auto stored = storedSurface(segmentId, error, true);
    if (!stored) return false;

    const QString exe = vc3d::findToolExecutable(QStringLiteral("vc_straighten"));
    if (exe.isEmpty()) {
        setErr(tr("Could not find the 'vc_straighten' executable (known locations and PATH). "
                  "Set it in VC.ini [tools] vc_straighten, or put it on PATH."),
               CommandLaunchError::ToolUnavailable);
        return false;
    }

    const QString outputName =
        params.outputName.isEmpty() ? stored->name + QStringLiteral("_straightened") : params.outputName;

    const QStringList args = straightenArgs(params);
    std::cout << "[straighten] surface=" << segmentId << " out=" << outputName.toStdString()
              << " exe=" << exe.toStdString() << " args=" << args.join(' ').toStdString() << std::endl;
    new FlattenJob(_parentWidget, this, _surfacePanel->catalog(),
                   {exe, stored->store, stored->volumeUuid,
                    QStringList{QStringLiteral("--surface"), QString::fromStdString(stored->uuid),
                                QStringLiteral("--name"), outputName} + args,
                    QStringLiteral("flatten.straighten"), tr("Straighten"), tr("Running vc_straighten..."), outputName},
                   /*suppressDialogs=*/!interactive);

    if (resolvedOutputName) *resolvedOutputName = outputName;
    return true;
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

    if (_neighborCopyJob) {
        QMessageBox::warning(_parentWidget, tr("Warning"), tr("Another neighbor copy request is already running."));
        return;
    }

    const auto stored = requireStoredSurface(segmentId.toStdString(), false, true);
    if (!stored) return;

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

    NeighborCopyDialog dlg(_parentWidget, stored->name, volumeOptions, defaultVolumeId, QStringLiteral("paths"));
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Copy %1 cancelled").arg(copyOut ? tr("out") : tr("in")), 3000);
        return;
    }

    QString selectedVolumePath = dlg.selectedVolumePath();
    if (selectedVolumePath.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("No target volume selected."));
        return;
    }

    const QString outputTag = dlg.outputTag();
    if (outputTag.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Output tag cannot be empty."));
        return;
    }

    // The normal grids paired with the open volume (CWindow's catalog-aware lookup).
    const QString normalGridPath = _normalGridPathGetter ? _normalGridPathGetter() : QString();

    QJsonObject refine;
    refine["normal_grid_path"] = normalGridPath;
    refine["max_gen"] = 1;
    refine["generations"] = 1;
    refine["resume_local_opt_step"] = dlg.resumeLocalOptStep();
    refine["resume_local_opt_radius"] = dlg.resumeLocalOptRadius();
    refine["resume_local_max_iters"] = dlg.resumeLocalMaxIters();
    refine["resume_local_dense_qr"] = dlg.resumeLocalDenseQr();
    refine["thread_limit"] = std::max(1, dlg.pass2OmpThreads());
    {
        QString refineError;
        auto extraParams = dlg.pass2TracerParamsJson(&refineError);
        if (!refineError.isEmpty()) {
            QMessageBox::warning(_parentWidget, tr("Error"), refineError);
            return;
        }
        if (extraParams) {
            for (auto it = extraParams->begin(); it != extraParams->end(); ++it) {
                refine.insert(it.key(), it.value());
            }
        }
    }

    QJsonObject params;
    params["normal_grid_path"] = normalGridPath;
    params["neighbor_dir"] = copyOut ? QStringLiteral("out") : QStringLiteral("in");
    params["neighbor_max_distance"] = dlg.neighborMaxDistance();
    params["mode"] = QStringLiteral("gen_neighbor");
    params["neighbor_min_clearance"] = dlg.neighborMinClearance();
    params["neighbor_fill"] = dlg.neighborFill();
    params["neighbor_interp_window"] = dlg.neighborInterpWindow();
    params["generations"] = dlg.generations();
    params["neighbor_spike_window"] = dlg.neighborSpikeWindow();
    params["refine"] = refine;

    auto paramsFile = std::make_unique<QTemporaryFile>(QDir::temp().filePath("neighbor_copy_XXXXXX.json"));
    if (!paramsFile->open()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Failed to create temporary params file."));
        return;
    }
    paramsFile->write(QJsonDocument(params).toJson(QJsonDocument::Indented));
    paramsFile->flush();

    configureCommandRunnerForVolume(dlg.selectedVolumeId().toStdString());
    _cmdRunner->setStore(stored->store);
    _cmdRunner->setNeighborCopyParams(selectedVolumePath, paramsFile->fileName(), segmentId,
                                      dlg.selectedVolumeId(), outputTag, QString());
    _cmdRunner->showConsoleOutput();
    _neighborCopyJob = NeighborCopyJob{copyOut, std::move(paramsFile)};
    if (!_cmdRunner->execute(CommandLineToolRunner::Tool::NeighborCopy)) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Failed to launch neighbor copy."));
        _neighborCopyJob.reset();
        return;
    }

    emit statusMessage(tr("Copy %1 started for %2")
                                 .arg(copyOut ? tr("out") : tr("in"))
                                 .arg(stored->name),
                             5000);
}

QJsonObject SegmentationCommandHandler::buildResumeLocalBaseParamsJson() const
{
    // The normal grids paired with the open volume (CWindow's catalog-aware lookup).
    const QString normalGridPath = _normalGridPathGetter ? _normalGridPathGetter() : QString();

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
    return params;
}

QJsonObject SegmentationCommandHandler::buildResumeLocalParamsJson(
    const QJsonObject& overrides) const
{
    QJsonObject params = buildResumeLocalBaseParamsJson();
    for (auto it = overrides.begin(); it != overrides.end(); ++it) {
        params.insert(it.key(), it.value());
    }
    return params;
}

void SegmentationCommandHandler::onResumeLocalGrowPatchRequested(const QString& segmentId)
{
    if (!_state || !_state->vpkg()) {
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

    if (_neighborCopyJob) {
        QMessageBox::warning(_parentWidget, tr("Warning"), tr("Another neighbor copy request is already running."));
        return;
    }

    if (_resumeLocalJob) {
        QMessageBox::warning(_parentWidget, tr("Warning"), tr("A resume-opt local GrowPatch run is already active."));
        return;
    }

    if (!requireStoredSurface(segmentId.toStdString(), false, true)) return;

    QString defaultVolumeId;
    const auto volumeOptions = buildVolumeOptionList(&defaultVolumeId);
    if (volumeOptions.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("No volumes available in the volume package."));
        return;
    }

    ResumeLocalGrowParams params;
    std::optional<QJsonObject> extraParams;
    params.ompThreads =
        vc3d::settings::neighbor_copy::RESUME_LOCAL_OMP_THREADS_DEFAULT;
    if (!selectResumeLocalTracerParams(_parentWidget,
                                       volumeOptions,
                                       defaultVolumeId,
                                       &params.volumeId,
                                       &extraParams,
                                       &params.ompThreads)) {
        emit statusMessage(tr("Resume-opt local GrowPatch cancelled"), 3000);
        return;
    }

    if (params.volumeId.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("No target volume selected."));
        return;
    }
    if (extraParams) {
        params.paramOverrides = *extraParams;
    }

    // Check if merged params require normal3d but we don't have it
    const QJsonObject paramsJson =
        buildResumeLocalParamsJson(params.paramOverrides);
    bool needsNormal3d = false;
    if (paramsJson.contains("normal3dline_weight")) {
        const double w = paramsJson["normal3dline_weight"].toDouble(0.0);
        needsNormal3d = (w > 0.0);
    }

    if (needsNormal3d && !paramsJson.contains("normal3d_zarr_path")) {
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

    CommandLaunchError error;
    if (!startResumeLocalGrowPatchImpl(segmentId.toStdString(), params,
                                       /*interactive=*/true, &error)) {
        QMessageBox::warning(_parentWidget, tr("Error"), error.message);
        return;
    }
    emit statusMessage(tr("Resume-opt local GrowPatch started for %1").arg(segmentId), 5000);
}

bool SegmentationCommandHandler::startResumeLocalGrowPatch(const std::string& segmentId,
                                                          const ResumeLocalGrowParams& params,
                                                          CommandLaunchError* error)
{
    return startResumeLocalGrowPatchImpl(segmentId, params, /*interactive=*/false, error);
}

bool SegmentationCommandHandler::startResumeLocalGrowPatchImpl(
    const std::string& segmentId,
    const ResumeLocalGrowParams& params,
    bool interactive,
    CommandLaunchError* error)
{
    auto fail = [&](const QString& message,
                    CommandLaunchError::Kind kind = CommandLaunchError::Other) -> bool {
        if (error) *error = {kind, message};
        return false;
    };

    if (!_state || !_state->vpkg()) {
        return fail(tr("No volume package loaded."), CommandLaunchError::InvalidState);
    }
    if (_state->currentVolume() == nullptr) {
        return fail(tr("No volume loaded."), CommandLaunchError::InvalidState);
    }
    if (!_cmdRunner) {
        return fail(tr("Command line tools are not available."),
                    CommandLaunchError::ToolUnavailable);
    }
    if (_cmdRunner->isRunning()) {
        return fail(tr("A command line tool is already running."), CommandLaunchError::Busy);
    }
    // Return a typed launch error before the runner reaches its interactive
    // missing-tool warning. Tool::NeighborCopy launches vc_grow_seg_from_seed.
    if (vc3d::findToolExecutable(QStringLiteral("vc_grow_seg_from_seed")).isEmpty()) {
        return fail(tr("vc_grow_seg_from_seed not found or not executable"), CommandLaunchError::ToolUnavailable);
    }
    if (_neighborCopyJob) {
        return fail(tr("A neighbor copy request is already active."), CommandLaunchError::Busy);
    }
    if (_resumeLocalJob) {
        return fail(tr("A resume-opt local GrowPatch run is already active."),
                    CommandLaunchError::Busy);
    }

    const auto stored = storedSurface(segmentId, error, true);
    if (!stored) return false;

    QString defaultVolumeId;
    const auto volumeOptions = buildVolumeOptionList(&defaultVolumeId);
    if (volumeOptions.isEmpty()) {
        return fail(tr("No volumes available in the volume package."));
    }

    QString selectedVolumeId = params.volumeId.isEmpty() ? defaultVolumeId : params.volumeId;
    QString selectedVolumePath;
    for (const auto& option : volumeOptions) {
        if (option.id == selectedVolumeId) {
            selectedVolumePath = option.path;
            break;
        }
    }
    if (selectedVolumePath.isEmpty()) {
        return fail(tr("Unknown volume id: %1").arg(selectedVolumeId),
                    CommandLaunchError::VolumeNotFound);
    }

    const QJsonObject paramsJson =
        buildResumeLocalParamsJson(params.paramOverrides);

    auto paramsFile = std::make_unique<QTemporaryFile>(
        QDir::temp().filePath("growpatch_resume_local_XXXXXX.json"));
    if (!paramsFile->open()) {
        return fail(tr("Failed to create temporary params file."));
    }
    paramsFile->write(QJsonDocument(paramsJson).toJson(QJsonDocument::Indented));
    paramsFile->flush();

    _resumeLocalJob = ResumeLocalJob{};
    auto& job = *_resumeLocalJob;
    job.segmentId = QString::fromStdString(segmentId);
    job.paramsPath = paramsFile->fileName();
    job.paramsFile = std::move(paramsFile);

    // The resumed surface is a new surface tagged like a grown patch.
    _cmdRunner->setStore(stored->store);
    _cmdRunner->setNeighborCopyParams(selectedVolumePath,
                                      job.paramsPath,
                                      QString::fromStdString(segmentId),
                                      selectedVolumeId,
                                      QStringLiteral("paths"),
                                      QStringLiteral("local"));
    configureCommandRunnerForVolume(selectedVolumeId.toStdString());
    _cmdRunner->setNextOmpThreads(params.ompThreads);
    const auto options = interactive
        ? CommandLineToolRunner::ExecutionOptions{}
        : CommandLineToolRunner::ExecutionOptions::silent();
    const bool started =
        _cmdRunner->execute(CommandLineToolRunner::Tool::NeighborCopy, options);
    if (!started) {
        // Nothing was launched, so toolFinished (which clears resumeLocalJob()
        // in the CWindow slot) will never fire: undo our own bookkeeping here.
        _resumeLocalJob.reset();
        return fail(tr("Failed to start resume-opt local GrowPatch."));
    }
    return true;
}

bool SegmentationCommandHandler::startGrowPatchFromSeed(const QVector3D& seedPoint,
                                                       const GrowPatchSeedParams& params,
                                                       CommandLaunchError* error)
{
    return startGrowPatchFromSeedImpl(seedPoint, params, /*interactive=*/false, error);
}

bool SegmentationCommandHandler::startGrowPatchFromSeedImpl(
    const QVector3D& seedPoint,
    const GrowPatchSeedParams& params,
    bool interactive,
    CommandLaunchError* error)
{
    // Never opens a dialog; callers receive typed launch failures.
    auto fail = [&](const QString& message,
                    CommandLaunchError::Kind kind = CommandLaunchError::Other) -> bool {
        if (error) *error = {kind, message};
        return false;
    };

    if (!_state || !_state->vpkg()) {
        return fail(tr("No volume package loaded."), CommandLaunchError::InvalidState);
    }
    if (_state->currentVolume() == nullptr) {
        return fail(tr("No volume loaded."), CommandLaunchError::InvalidState);
    }
    if (!_cmdRunner) {
        return fail(tr("Command line tools are not available."),
                    CommandLaunchError::ToolUnavailable);
    }
    if (_cmdRunner->isRunning()) {
        return fail(tr("A command line tool is already running."), CommandLaunchError::Busy);
    }
    if (_growPatchSeedJob) {
        return fail(tr("A GrowPatch seed run is already active."), CommandLaunchError::Busy);
    }

    const QString executable = vc3d::findToolExecutable(QStringLiteral("vc_grow_seg_from_seed"));
    if (executable.isEmpty()) {
        return fail(tr("Could not find the 'vc_grow_seg_from_seed' executable."),
                    CommandLaunchError::ToolUnavailable);
    }

    QString defaultVolumeId;
    const auto volumeOptions = buildVolumeOptionList(&defaultVolumeId);
    if (volumeOptions.isEmpty()) {
        return fail(tr("No volumes available in the volume package."));
    }

    QString selectedVolumeId = params.volumeId;
    if (selectedVolumeId.isEmpty()) {
        selectedVolumeId = defaultVolumeId;
    }
    QString selectedVolumePath;
    for (const auto& option : volumeOptions) {
        if (option.id == selectedVolumeId) {
            selectedVolumePath = option.path;
            break;
        }
    }
    if (selectedVolumePath.isEmpty()) {
        return fail(tr("Unknown volume id: %1").arg(selectedVolumeId),
                    CommandLaunchError::VolumeNotFound);
    }

    // The normal grids paired with the open volume (CWindow's catalog-aware lookup).
    const QString normalGridPath = _normalGridPathGetter ? _normalGridPathGetter() : QString();
    if (normalGridPath.isEmpty()) {
        return fail(tr("No normal grid is available for GrowPatch."),
                    CommandLaunchError::InvalidState);
    }

    const auto store = storeForTools(error);
    if (!store) return false;
    if (params.outputTag.trimmed().isEmpty()) {
        return fail(tr("GrowPatch needs an output tag."));
    }

    QString volpkgRoot = _state->vpkgPath();
    if (volpkgRoot.isEmpty()) {
        volpkgRoot = QString::fromStdString(_state->vpkg()->getVolpkgDirectory());
    }

    const int iterations = std::clamp(params.iterations, 1, 100000);
    const double minAreaCm = std::max(0.0, params.minAreaCm);

    double selectedVoxelSize = 0.0;
    if (auto selectedVolume = _state->vpkg()->volume(selectedVolumeId.toStdString())) {
        selectedVoxelSize = selectedVolume->voxelSize();
    }
    // vc_grow_seg_from_seed refuses a physical min_area_cm without a voxel
    // size. Say so here, where the message reaches the user and the fix is
    // one field away, instead of surfacing only as "exit code 1" along with
    // a hint naming a params file this dialog has already deleted.
    if (minAreaCm > 0.0 && !(std::isfinite(selectedVoxelSize) && selectedVoxelSize > 0.0)) {
        return fail(tr("Volume '%1' reports no voxel size, so the minimum patch area of %2 cm² "
                       "cannot be evaluated. Set the minimum area to 0 to grow without a "
                       "size threshold, or add \"voxelsize\" (µm per voxel) to the volume's "
                       "metadata.")
                        .arg(selectedVolumeId)
                        .arg(minAreaCm),
                    CommandLaunchError::InvalidState);
    }

    QJsonObject paramsJson;
    paramsJson.insert(QStringLiteral("mode"), QStringLiteral("seed"));
    paramsJson.insert(QStringLiteral("step_size"), 20);
    paramsJson.insert(QStringLiteral("min_area_cm"), minAreaCm);
    paramsJson.insert(QStringLiteral("generations"), iterations);
    paramsJson.insert(QStringLiteral("thread_limit"), 1);
    paramsJson.insert(QStringLiteral("normal_grid_path"), normalGridPath);
    paramsJson.insert(QStringLiteral("cache_root"), QDir(volpkgRoot).filePath(QStringLiteral("cache")));
    if (std::isfinite(selectedVoxelSize) && selectedVoxelSize > 0.0) {
        paramsJson.insert(QStringLiteral("voxelsize"), selectedVoxelSize);
    }
    if (_normal3dZarrPathGetter) {
        const QString normal3dPath = _normal3dZarrPathGetter();
        if (!normal3dPath.isEmpty()) {
            paramsJson.insert(QStringLiteral("normal3d_zarr_path"), normal3dPath);
        }
    }

    auto paramsFile = std::make_unique<QTemporaryFile>(QDir::temp().filePath("growpatch_seed_XXXXXX.json"));
    if (!paramsFile->open()) {
        return fail(tr("Failed to create temporary params file."));
    }
    paramsFile->write(QJsonDocument(paramsJson).toJson(QJsonDocument::Indented));
    paramsFile->flush();

    _growPatchSeedJob = GrowPatchSeedJob{};
    _growPatchSeedJob->paramsPath = paramsFile->fileName();
    _growPatchSeedJob->paramsFile = std::move(paramsFile);

    // The new surface is bound to the selected volume.
    QStringList args;
    args << QStringLiteral("-v") << selectedVolumePath
         << QStringLiteral("-p") << _growPatchSeedJob->paramsPath
         << QStringLiteral("--tag") << params.outputTag.trimmed()
         << QStringLiteral("--seed")
         << QString::number(seedPoint.x(), 'g', 10)
         << QString::number(seedPoint.y(), 'g', 10)
         << QString::number(seedPoint.z(), 'g', 10);

    configureCommandRunnerForVolume(selectedVolumeId.toStdString());

    QPointer<SegmentationCommandHandler> guard(this);
    auto connection = std::make_shared<QMetaObject::Connection>();
    *connection = connect(_cmdRunner,
                          &CommandLineToolRunner::toolFinished,
                          this,
                          [this, guard, connection](CommandLineToolRunner::Tool tool,
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
        _growPatchSeedJob.reset();

        // The runner's own completion handler reports a failure.
        if (!success) {
            emit statusMessage(tr("Create segment failed"), 3000);
            return;
        }

        emit statusMessage(
            tr("GrowPatch surface created: %1").arg(_cmdRunner->createdSurfaces().join(QStringLiteral(", "))),
            5000);
    });

    const auto options = interactive
        ? CommandLineToolRunner::ExecutionOptions{}
        : CommandLineToolRunner::ExecutionOptions::silent();
    if (!_cmdRunner->executeCustomCommand(
            executable, args, QStringLiteral("vc_grow_seg_from_seed"), options, store->store, selectedVolumeId)) {
        QObject::disconnect(*connection);
        _growPatchSeedJob.reset();
        return fail(tr("Failed to start vc_grow_seg_from_seed."));
    }

    emit statusMessage(
        tr("GrowPatch seed started at (%1, %2, %3)")
            .arg(seedPoint.x(), 0, 'g', 6)
            .arg(seedPoint.y(), 0, 'g', 6)
            .arg(seedPoint.z(), 0, 'g', 6),
        5000);
    return true;
}

void SegmentationCommandHandler::onCreateSegmentGrowPatchFromSeed(const QVector3D& seedPoint)
{
    // Gather interactive defaults before using the shared launcher.
    if (!_state || !_state->vpkg()) {
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

    if (_growPatchSeedJob) {
        QMessageBox::warning(_parentWidget, tr("Warning"), tr("A GrowPatch seed run is already active."));
        return;
    }

    if (vc3d::findToolExecutable(QStringLiteral("vc_grow_seg_from_seed")).isEmpty()) {
        QMessageBox::critical(_parentWidget, tr("Error"),
            tr("Could not find the 'vc_grow_seg_from_seed' executable.\n"
               "Looked in known locations and PATH.\n\n"
               "Tip: set an override via VC.ini [tools] vc_grow_seg_from_seed, or put "
               "the binary on PATH."));
        return;
    }

    QString defaultVolumeId;
    const auto volumeOptions = buildVolumeOptionList(&defaultVolumeId);
    if (volumeOptions.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("No volumes available in the volume package."));
        return;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    QString selectedVolumeId = settings.value(QStringLiteral("growpatch_seed/volume_id")).toString();
    if (selectedVolumeId.isEmpty() || std::none_of(volumeOptions.begin(), volumeOptions.end(),
                                                   [&](const auto& option) { return option.id == selectedVolumeId; })) {
        selectedVolumeId = defaultVolumeId;
    }
    QString selectedVolumePath;
    for (const auto& option : volumeOptions) {
        if (option.id == selectedVolumeId) {
            selectedVolumePath = option.path;
            break;
        }
    }

    int iterations = 200;
    double minAreaCm = 0.002;
    QString outputTag = settings.value(QStringLiteral("growpatch_seed/output_tag"), QStringLiteral("paths")).toString();
    if (!selectGrowPatchSeedParams(_parentWidget,
                                   volumeOptions,
                                   &selectedVolumeId,
                                   &selectedVolumePath,
                                   &iterations,
                                   &minAreaCm,
                                   &outputTag)) {
        emit statusMessage(tr("Create segment cancelled"), 3000);
        return;
    }
    settings.setValue(QStringLiteral("growpatch_seed/output_tag"), outputTag);
    settings.setValue(QStringLiteral("growpatch_seed/volume_id"), selectedVolumeId);

    GrowPatchSeedParams patchParams;
    patchParams.volumeId = selectedVolumeId;
    patchParams.iterations = iterations;
    patchParams.minAreaCm = minAreaCm;
    patchParams.outputTag = outputTag;

    CommandLaunchError error;
    if (!startGrowPatchFromSeedImpl(
            seedPoint, patchParams, /*interactive=*/true, &error)) {
        QMessageBox::warning(_parentWidget, tr("Error"), error.message);
    }
}

void SegmentationCommandHandler::onConvertToObj(const std::string& segmentId)
{
    const auto stored = requireStoredSurface(segmentId, true, true);
    if (!stored) return;

    ConvertToObjDialog dlg(_parentWidget, stored->name, stored->name + QStringLiteral(".obj"));
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Convert to OBJ cancelled"), 3000);
        return;
    }
    const QString objName = dlg.objName();
    if (objName.isEmpty() || objName.contains('/') || objName.contains('\\')) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("The OBJ name is a file name, without directories."));
        return;
    }

    // The tool writes into the surface's artifact directory and reports the path.
    _cmdRunner->setStore(stored->store);
    _cmdRunner->setToObjParams(QString::fromStdString(stored->uuid), objName, stored->volumeUuid);
    _cmdRunner->setNextOmpThreads(dlg.ompThreads());
    _cmdRunner->setToObjOptions(dlg.normalizeUV(), dlg.alignGrid());
    _cmdRunner->execute(CommandLineToolRunner::Tool::tifxyz2obj);
    emit statusMessage(tr("Converting surface to OBJ: %1").arg(stored->name), 5000);
}

void SegmentationCommandHandler::onCropSurfaceToValidRegion(const std::string& segmentId)
{
    // Interactive wrapper: run the dialog-free core and surface failures via
    // QMessageBox as before; success is reported by the core's own statusMessage.
    QString err;
    if (!cropSurfaceToValidRegion(segmentId, &err) && !err.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Crop failed"), err);
    }
}

bool SegmentationCommandHandler::cropSurfaceToValidRegion(const std::string& segmentId,
                                                          QString* errorMessage)
{
    // Crop to the tightest valid bounds and persist the surface. An already
    // tight surface is a successful no-op.
    auto fail = [&](const QString& message) -> bool {
        if (errorMessage) {
            *errorMessage = message;
        }
        return false;
    };

    // Inlined from requireSurfaceAndRunner (which pops dialogs) so this path stays
    // dialog-free; runner state is irrelevant to a synchronous crop.
    auto surf = activeSurface(segmentId);
    if (!surf) {
        return fail(tr("Select surface %1 first.").arg(QString::fromStdString(segmentId)));
    }
    if (_editingDestinationResolver && !_editingDestinationResolver(surf))
        return fail(tr("Could not prepare the surface for editing"));
    QuadSurface* surface = surf.get();

    cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return fail(tr("Cannot crop surface: Missing coordinate grid"));
    }

    const int origCols = points->cols;
    const int origRows = points->rows;

    const auto boundsOpt = computeValidSurfaceBounds(*points);
    if (!boundsOpt) {
        return fail(tr("Surface %1 does not contain any valid vertices to crop.")
                        .arg(QString::fromStdString(segmentId)));
    }

    const cv::Rect roi = *boundsOpt;
    if (roi.x == 0 && roi.y == 0 && roi.width == origCols && roi.height == origRows) {
        emit statusMessage(
            tr("Surface %1 already occupies the tightest bounds.")
                .arg(QString::fromStdString(segmentId)),
            4000);
        return true;
    }

    struct CroppedChannel {
        std::string name;
        cv::Mat data;
    };
    std::vector<CroppedChannel> croppedChannels;
    croppedChannels.reserve(surface->channelNames().size());
    bool droppedGenerationsChannel = false;

    const auto channelNames = surface->channelNames();
    for (const auto& name : channelNames) {
        cv::Mat channelData = surface->channel(name, SURF_CHANNEL_NORESIZE);
        if (channelData.empty()) {
            continue;
        }
        if (channelData.cols % origCols != 0 || channelData.rows % origRows != 0) {
            if (name == "generations") {
                droppedGenerationsChannel = true;
                continue;
            }
            return fail(tr("Channel '%1' has size %2x%3, which is not divisible by the surface grid %4x%5.")
                            .arg(QString::fromStdString(name))
                            .arg(channelData.cols)
                            .arg(channelData.rows)
                            .arg(origCols)
                            .arg(origRows));
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
            return fail(tr("Computed crop exceeds the bounds of channel '%1'.")
                            .arg(QString::fromStdString(name)));
        }

        croppedChannels.push_back({name, channelData(chanRect).clone()});
    }

    cv::Mat_<cv::Vec3f> croppedPoints = (*points)(roi).clone();

    // The cropped grid replaces the stored geometry (a dropped channel goes with it).
    croppedPoints.copyTo(*points);
    for (const auto& ch : croppedChannels) {
        surface->setChannel(ch.name, ch.data);
    }
    if (droppedGenerationsChannel) {
        surface->setChannel("generations", cv::Mat());
    }
    surface->invalidateCache();
    if (!_surfaceSaver || !_surfaceSaver(surface)) {
        return fail(tr("Failed to save the cropped surface %1.").arg(QString::fromStdString(segmentId)));
    }

    // Bbox will be recalculated lazily (invalidateCache was already called)

    if (_state) {
        _state->setSurface("segmentation", surf, false, false);
    }

    const QString segLabel = QString::fromStdString(segmentId);
    emit statusMessage(
        (droppedGenerationsChannel
             ? tr("Cropped %1 to %2x%3 (offset %4,%5); removed the mismatched generations channel")
             : tr("Cropped %1 to %2x%3 (offset %4,%5)"))
            .arg(segLabel)
            .arg(roi.width)
            .arg(roi.height)
            .arg(roi.x)
            .arg(roi.y),
        5000);
    return true;
}

void SegmentationCommandHandler::onFlipSurface(const std::string& segmentId, bool flipU)
{
    auto* surface = requireSurfaceAndRunner(segmentId, false);
    if (!surface) return;

    auto surf = activeSurface(segmentId);
    loadAllChannels(*surface);

    if (flipU) {
        surface->flipU();
    } else {
        surface->flipV();
    }

    if (!publishSurface(surface, tr("Flip"))) {
        return;
    }

    if (_state) {
        _state->setSurface("segmentation", surf, false, false);
    }

    const QString directionLabel = flipU ? tr("vertically") : tr("horizontally");
    emit statusMessage(
        tr("Flipped %1 %2")
            .arg(QString::fromStdString(segmentId))
            .arg(directionLabel),
        5000);
}

void SegmentationCommandHandler::onRotateSurface(const std::string& segmentId)
{
    auto* surface = requireSurfaceAndRunner(segmentId, false);
    if (!surface) return;

    auto surf = activeSurface(segmentId);
    loadAllChannels(*surface);
    surface->rotate(90.0f);

    if (!publishSurface(surface, tr("Rotate"))) {
        return;
    }

    if (_state) {
        _state->setSurface("segmentation", surf, false, false);
    }

    emit statusMessage(
        tr("Rotated %1 by 90 degrees clockwise")
            .arg(QString::fromStdString(segmentId)),
        5000);
}

void SegmentationCommandHandler::onAlphaCompRefine(const std::string& segmentId)
{
    const auto stored = requireStoredSurface(segmentId, true, true);
    if (!stored) return;

    QString defaultVolumeId;
    const auto volumes = buildVolumeOptionList(&defaultVolumeId);
    AlphaCompRefineDialog dlg(_parentWidget, volumes, defaultVolumeId, stored->name, stored->name + QStringLiteral("_refined"));
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Alpha-comp refinement cancelled"), 3000);
        return;
    }

    CommandLaunchError error;
    if (!startAlphaCompRefineImpl(segmentId, dlg.request(), /*interactive=*/true, &error)) {
        QMessageBox::warning(_parentWidget, tr("Error"), error.message);
        return;
    }
    emit statusMessage(tr("Refining surface: %1").arg(stored->name), 5000);
}

bool SegmentationCommandHandler::startAlphaCompRefine(const std::string& segmentId,
                                                     const AlphaCompRefineParams& params,
                                                     CommandLaunchError* error)
{
    return startAlphaCompRefineImpl(segmentId, params, /*interactive=*/false, error);
}

QJsonObject SegmentationCommandHandler::alphaCompRefineParamsJson(
    const AlphaCompRefineParams& params)
{
    return {
        {QStringLiteral("refine"), params.refine},
        {QStringLiteral("start"), params.start},
        {QStringLiteral("stop"), params.stop},
        {QStringLiteral("step"), params.step},
        {QStringLiteral("low"), params.low},
        {QStringLiteral("high"), params.high},
        {QStringLiteral("border_off"), params.borderOff},
        {QStringLiteral("r"), params.radius},
        {QStringLiteral("gen_vertexcolor"), params.genVertexColor},
        {QStringLiteral("reader_scale"), params.readerScale},
        {QStringLiteral("scale_group"),
         params.scaleGroup.isEmpty() ? QStringLiteral("1") : params.scaleGroup},
    };
}

bool SegmentationCommandHandler::startAlphaCompRefineImpl(
    const std::string& segmentId,
    const AlphaCompRefineParams& params,
    bool interactive,
    CommandLaunchError* error)
{
    auto fail = [&](const QString& message,
                    CommandLaunchError::Kind kind = CommandLaunchError::Other) -> bool {
        if (error) *error = {kind, message};
        return false;
    };

    if (!_state || _state->currentVolume() == nullptr || !_state->vpkg()) {
        return fail(tr("No volume package or volume loaded."),
                    CommandLaunchError::InvalidState);
    }
    const auto stored = storedSurface(segmentId, error, !params.inPlace);
    if (!stored) return false;
    if (!_cmdRunner) {
        return fail(tr("Command line tools are not available."),
                    CommandLaunchError::ToolUnavailable);
    }
    if (_cmdRunner->isRunning()) {
        return fail(tr("A command line tool is already running."), CommandLaunchError::Busy);
    }
    // The surface is read in the voxels of the volume the refinement samples.
    const std::string sampledVolume = params.volumeId.isEmpty() ? _state->currentVolumeId() : params.volumeId.toStdString();
    if (!_state->vpkg()->hasVolume(sampledVolume)) {
        return fail(tr("%1 is not a volume of this project.").arg(QString::fromStdString(sampledVolume)));
    }
    const auto sampled = _state->vpkg()->volume(sampledVolume);
    if (sampled->isRemote()) {
        return fail(tr("Alpha-comp refinement accepts only local volumes."), CommandLaunchError::RemoteVolume);
    }
    const QString volumePath = commandPathForVolume(sampled);
    // Return a typed launch error before the runner reaches its interactive
    // missing-tool warning. Tool::AlphaCompRefine launches vc_objrefine.
    if (vc3d::findToolExecutable(QStringLiteral("vc_objrefine")).isEmpty()) {
        return fail(tr("vc_objrefine not found or not executable"), CommandLaunchError::ToolUnavailable);
    }

    // The output is a new surface's name, or none to refine in place.
    const QString newName = params.inPlace ? QString()
        : params.outputName.trimmed().isEmpty() ? stored->name + QStringLiteral("_refined")
                                                : params.outputName.trimmed();

    auto paramsFile = std::make_unique<QTemporaryFile>(
        QDir::temp().filePath("vc_objrefine_XXXXXX.json"));
    if (!paramsFile->open()) {
        return fail(tr("Failed to create temporary params JSON file."));
    }
    paramsFile->write(
        QJsonDocument(alphaCompRefineParamsJson(params)).toJson(QJsonDocument::Indented));
    paramsFile->flush();
    const QString paramsPath = paramsFile->fileName();
    paramsFile->close();

    _alphaCompJob = AlphaCompJob{std::move(paramsFile)};
    auto completion = std::make_shared<QMetaObject::Connection>();
    *completion = connect(
        _cmdRunner, &CommandLineToolRunner::toolFinished, this,
        [this, completion](CommandLineToolRunner::Tool tool, bool, const QString&,
                           const QString&, bool) {
            if (tool != CommandLineToolRunner::Tool::AlphaCompRefine) {
                return;
            }
            disconnect(*completion);
            _alphaCompJob.reset();
        });

    _cmdRunner->setStore(stored->store);
    _cmdRunner->setObjRefineParams(volumePath, QString::fromStdString(stored->uuid), newName, paramsPath,
                                   QString::fromStdString(sampledVolume));
    _cmdRunner->setNextOmpThreads(params.ompThreads);
    const auto options = interactive
        ? CommandLineToolRunner::ExecutionOptions{}
        : CommandLineToolRunner::ExecutionOptions::silent();
    const bool started =
        _cmdRunner->execute(CommandLineToolRunner::Tool::AlphaCompRefine, options);
    if (!started) {
        disconnect(*completion);
        _alphaCompJob.reset();
        return fail(tr("Failed to start alpha-comp refinement."));
    }
    return true;
}

void SegmentationCommandHandler::handleNeighborCopyToolFinished(bool success)
{
    if (!_neighborCopyJob) {
        return;
    }
    const bool copyOut = _neighborCopyJob->copyOut;
    _neighborCopyJob.reset();
    if (!success) {
        return;
    }
    const QStringList created = _cmdRunner ? _cmdRunner->createdSurfaces() : QStringList{};
    emit statusMessage(tr("Copy %1 complete: %2")
                                 .arg(copyOut ? tr("out") : tr("in"))
                                 .arg(created.join(QStringLiteral(", "))),
                             5000);
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

    std::optional<StoredSurface> store;
    QStringList missingIds;
    for (const QString& segmentId : rasterIds) {
        CommandLaunchError error;
        if (auto stored = storedSurface(segmentId.toStdString(), &error, true)) {
            store = stored;
            validIds << segmentId;
        } else {
            missingIds << QStringLiteral("%1 (%2)").arg(segmentId, error.message);
        }
    }

    if (validIds.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"),
                             tr("None of the selected surfaces can be rasterized: %1")
                                 .arg(missingIds.join(QStringLiteral(", "))));
        return;
    }
    if (!missingIds.isEmpty()) {
        emit statusMessage(
            tr("Ignoring %1 surface(s) that cannot be rasterized.")
                .arg(missingIds.size()),
            3000);
    }

    QString referenceZarr = !_normal3dZarrPathGetter ? QString() : _normal3dZarrPathGetter();
    if (referenceZarr.isEmpty()) {
        referenceZarr = getCurrentVolumePath();
    }
    if (referenceZarr.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"),
                             tr("Missing reference OME-Zarr. Load a normal3d/volume first."));
        return;
    }

    const QString executable = vc3d::findToolExecutable(QStringLiteral("vc_tifxyz2zarr_sparse"));
    if (executable.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"),
                             tr("vc_tifxyz2zarr_sparse tool not found. Configure tools/vc_tifxyz2zarr_sparse path."));
        return;
    }

    RasterizeDialogResult rasterizeParams;
    if (!selectRasterizeParams(_parentWidget, &rasterizeParams)) {
        return;
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

    // The tool publishes the volume (staged, then renamed) with its meta.json.
    const QString finalOutputRootStr = QString::fromStdString(finalOutputRoot.string());
    QStringList args;
    args << finalOutputRootStr
         << QStringLiteral("--reference-zarr")
         << referenceZarr
         << QStringLiteral("--chunk-size")
         << QString::number(rasterizeParams.chunkSize)
         << QStringLiteral("--z-min")
         << QString::number(rasterizeParams.zMin)
         << QStringLiteral("--raster-mode")
         << QStringLiteral("zyx-integer")
         << QStringLiteral("--overwrite");
    if (rasterizeParams.zMax >= 0) {
        args << QStringLiteral("--z-max") << QString::number(rasterizeParams.zMax);
    }
    for (const QString& segmentId : validIds) {
        args << QStringLiteral("--surface") << segmentId;
    }

    auto runner = _cmdRunner;
    if (!runner) {
        QMessageBox::critical(_parentWidget, tr("Error"),
                              tr("Command runner is not available."));
        return;
    }

    QPointer<SegmentationCommandHandler> guard(this);
    auto connection = std::make_shared<QMetaObject::Connection>();
    *connection = connect(runner, &CommandLineToolRunner::toolFinished,
                         this,
                         [this, guard, connection, runner, finalOutputRootStr,
                          validIds](CommandLineToolRunner::Tool tool,
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

        // The runner's completion handler reports a failure.
        if (success) {
            emit statusMessage(tr("Rasterized %1 segment(s) -> %2")
                                   .arg(validIds.size())
                                   .arg(QDir::toNativeSeparators(finalOutputRootStr)),
                               5000);
        } else {
            emit statusMessage(tr("Rasterize failed"), 3000);
        }
    });

    if (!runner->executeCustomCommand(executable, args, QStringLiteral("vc_tifxyz2zarr_sparse"), {}, store->store,
                                      store->volumeUuid)) {
        QObject::disconnect(*connection);
        QMessageBox::critical(_parentWidget, tr("Error"),
                              tr("Failed to start vc_tifxyz2zarr_sparse."));
        return;
    }

    emit statusMessage(
        tr("Rasterization started for %1 surface(s)...").arg(validIds.size()), 0);
}

std::vector<const SurfaceCatalog::Row*> SegmentationCommandHandler::toolSurfaces(bool throughView) const
{
    std::vector<const SurfaceCatalog::Row*> rows;
    if (auto* catalog = _surfacePanel ? _surfacePanel->catalog() : nullptr) {
        for (const auto& [uuid, row] : catalog->rows())
            if (throughView ? !row.catalog && row.bound() : row.toolInput()) rows.push_back(&row);
    }
    return rows;
}

void SegmentationCommandHandler::onMergeSurfaces(const QStringList& segmentIds)
{
    if (!_cmdRunner) {
        QMessageBox::warning(_parentWidget, tr("Merge TIFXYZ"),
                             tr("Command runner is not initialized."));
        return;
    }
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(_parentWidget, tr("Merge TIFXYZ"),
                             tr("A command-line tool is already running."));
        return;
    }
    CommandLaunchError error;
    const auto store = storeForTools(&error);
    if (!store) {
        QMessageBox::warning(_parentWidget, tr("Merge TIFXYZ"), error.message);
        return;
    }

    QMap<QString, QString> available;
    for (const auto* row : toolSurfaces(true)) {
        const auto& record = row->record;
        available.insert(QString::fromStdString(record.uuid),
                         QString::fromStdString(record.name.empty() ? record.uuid : record.name));
    }
    if (available.size() < 2) {
        QMessageBox::warning(_parentWidget, tr("Merge TIFXYZ"),
                             tr("The current volume has fewer than 2 surfaces; merge needs at least 2."));
        return;
    }

    MergeSurfacesDialog dlg(_parentWidget, segmentIds, available);
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Merge cancelled"), 3000);
        return;
    }

    auto mergeJson = std::make_shared<QTemporaryFile>(QDir::temp().filePath(QStringLiteral("vc3d_XXXXXX.merge.json")));
    if (!mergeJson->open()) {
        QMessageBox::critical(_parentWidget, tr("Merge TIFXYZ"), tr("Failed to write merge.json."));
        return;
    }
    mergeJson->write(dlg.mergeJsonText().toUtf8());
    mergeJson->flush();

    // The merge is a new surface bound to the current volume.
    _cmdRunner->setStore(store->store);
    _cmdRunner->setMergeParams(mergeJson->fileName(),
                               store->volumeUuid,
                               QStringLiteral("paths"),
                               dlg.refSurface(),
                               dlg.ransacIters(),
                               dlg.ransacMinThresh(),
                               dlg.ransacMaxThresh(),
                               dlg.ransacMadK(),
                               dlg.ransacSeed(),
                               dlg.anchorCap(),
                               dlg.stripCols());
    if (dlg.ompThreads() > 0) _cmdRunner->setNextOmpThreads(dlg.ompThreads());
    // The merge.json lives until the tool finishes.
    auto connection = std::make_shared<QMetaObject::Connection>();
    *connection = connect(_cmdRunner, &CommandLineToolRunner::toolFinished, this,
                          [connection, mergeJson](CommandLineToolRunner::Tool tool, bool, const QString&, const QString&, bool) {
                              if (tool == CommandLineToolRunner::Tool::MergeTifxyz) disconnect(*connection);
                          });
    _cmdRunner->showConsoleOutput();
    if (!_cmdRunner->execute(CommandLineToolRunner::Tool::MergeTifxyz)) {
        disconnect(*connection);
        return;
    }
    emit statusMessage(tr("Merging surfaces..."), 0);
}

void SegmentationCommandHandler::onMergePatch(const QStringList& segmentIds)
{
    if (!_cmdRunner) {
        QMessageBox::warning(_parentWidget, tr("Patch Surface"),
                             tr("Command runner is not initialized."));
        return;
    }
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(_parentWidget, tr("Patch Surface"),
                             tr("A command-line tool is already running."));
        return;
    }
    CommandLaunchError error;
    const auto store = storeForTools(&error);
    if (!store) {
        QMessageBox::warning(_parentWidget, tr("Patch Surface"), error.message);
        return;
    }

    QMap<QString, MergePatchDialog::Choice> available;
    for (const auto* row : toolSurfaces()) {
        const auto& record = row->record;
        const auto meta = nlohmann::json::parse(record.meta.empty() ? std::string("{}") : record.meta, nullptr, false);
        const auto area = meta.is_object() ? meta.find("area_vx2") : meta.end();
        available.insert(QString::fromStdString(record.uuid),
                         {QString::fromStdString(record.name.empty() ? record.uuid : record.name),
                          meta.is_object() && area != meta.end() && area->is_number() ? area->get<double>() : -1.0});
    }
    if (available.size() < 2) {
        QMessageBox::warning(_parentWidget, tr("Patch Surface"),
                             tr("The current volume has fewer than 2 surfaces; "
                                "patch needs a parent + child."));
        return;
    }

    MergePatchDialog dlg(_parentWidget, segmentIds, available);
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Patch cancelled"), 3000);
        return;
    }

    _cmdRunner->setStore(store->store);
    _cmdRunner->setMergePatchParams(dlg.parentSurface(),
                                    dlg.childSurface(),
                                    dlg.explicitRoles(),
                                    dlg.borderCells(),
                                    dlg.blendCells(),
                                    dlg.idwK(),
                                    dlg.ransacIters(),
                                    dlg.ransacMinThresh(),
                                    dlg.ransacMaxThresh(),
                                    dlg.ransacMadK(),
                                    dlg.ransacSeed(),
                                    dlg.anchorCap());
    if (dlg.ompThreads() > 0) _cmdRunner->setNextOmpThreads(dlg.ompThreads());

    // The parent's new revision reaches the panel and an opened copy through
    // the store's commit events.
    QPointer<SegmentationCommandHandler> guard(this);
    auto connection = std::make_shared<QMetaObject::Connection>();
    *connection = connect(_cmdRunner, &CommandLineToolRunner::toolFinished,
                          this,
                          [this, guard, connection]
                          (CommandLineToolRunner::Tool tool,
                           bool success,
                           const QString& message,
                           const QString& outputPath,
                           bool) {
        if (!guard) {
            disconnect(*connection);
            return;
        }
        if (tool != CommandLineToolRunner::Tool::MergePatch) {
            return;
        }
        disconnect(*connection);
        if (success) {
            emit statusMessage(tr("Patch applied to %1").arg(outputPath), 5000);
        } else {
            QMessageBox::critical(_parentWidget, tr("Patch Surface"),
                                  tr("vc_merge_patch failed.\n%1").arg(message));
            emit statusMessage(tr("Patch failed"), 5000);
        }
    });

    _cmdRunner->showConsoleOutput();
    // If execute() rejects the launch (preflight failure, missing binary),
    // it never emits toolFinished -- so the one-shot lambda above would
    // stay attached and fire on the NEXT tool's toolFinished. Disconnect
    // immediately on launch failure to keep the runner's signal table
    // clean across subsequent runs.
    if (!_cmdRunner->execute(CommandLineToolRunner::Tool::MergePatch)) {
        QObject::disconnect(*connection);
        emit statusMessage(tr("Failed to start vc_merge_patch"), 5000);
        return;
    }
    emit statusMessage(tr("Patching surface..."), 0);
}

void SegmentationCommandHandler::onGrowTrackPatches()
{
    if (!_cmdRunner) {
        QMessageBox::warning(_parentWidget, tr("Grow Track Patches"),
                             tr("Command runner is not initialized."));
        return;
    }
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(_parentWidget, tr("Grow Track Patches"),
                             tr("A command-line tool is already running."));
        return;
    }
    CommandLaunchError error;
    const auto store = storeForTools(&error);
    if (!store) {
        QMessageBox::warning(_parentWidget, tr("Grow Track Patches"), error.message);
        return;
    }

    GrowTrackPatchesDialog dlg(_parentWidget);
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Grow Track Patches cancelled"), 3000);
        return;
    }

    const QString python = dlg.resolvedPython();
    const QString script = dlg.scriptPath();
    const QStringList scriptArgs = dlg.buildArgs();

    // Validate python executable once more (dialog already did)
    QFileInfo pyInfo(python);
    if (!pyInfo.exists() || (!pyInfo.isExecutable() && !pyInfo.isFile())) {
        QMessageBox::warning(_parentWidget, tr("Grow Track Patches"),
                             tr("Python executable not found or not executable: %1").arg(python));
        return;
    }
    if (!QFileInfo(script).isFile()) {
        QMessageBox::warning(_parentWidget, tr("Grow Track Patches"),
                             tr("Script not found: %1").arg(script));
        return;
    }

    // The patches become surfaces of the current volume, like traced ones.
    QStringList args;
    args << script << scriptArgs << QStringLiteral("--tag") << QStringLiteral("paths");

    if (dlg.ompThreads() > 0) {
        _cmdRunner->setNextOmpThreads(dlg.ompThreads());
    }

    // Connect one-shot completion handler to surface status messages.
    // Use CustomCommand tool identifier.
    QPointer<SegmentationCommandHandler> guard(this);
    auto connection = std::make_shared<QMetaObject::Connection>();
    *connection = connect(_cmdRunner, &CommandLineToolRunner::toolFinished,
                          this,
                          [this, guard, connection]
                          (CommandLineToolRunner::Tool tool,
                           bool success,
                           const QString& message,
                           const QString& /*outputPath*/,
                           bool) {
        if (!guard) {
            disconnect(*connection);
            return;
        }
        if (tool != CommandLineToolRunner::Tool::CustomCommand) {
            return;
        }
        disconnect(*connection);
        if (success) {
            emit statusMessage(tr("Grow Track Patches finished"), 5000);
        } else {
            QMessageBox::critical(_parentWidget, tr("Grow Track Patches"),
                                  tr("grow_track_graph.py failed.\n%1").arg(message));
            emit statusMessage(tr("Grow Track Patches failed"), 5000);
        }
    });

    _cmdRunner->showConsoleOutput();
    const bool started = _cmdRunner->executeCustomCommand(python, args, tr("Grow Track Patches"), {}, store->store,
                                                          store->volumeUuid);
    if (!started) {
        QObject::disconnect(*connection);
        emit statusMessage(tr("Failed to start grow_track_graph.py"), 5000);
        return;
    }
    emit statusMessage(tr("Running grow_track_graph.py..."), 0);
}

void SegmentationCommandHandler::onAddIgnoreLabel()
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

    QVector<VolumeSelector::VolumeOption> rasterizedVolumes;
    QString defaultVolumeId;
    const QString currentVolumeId = QString::fromStdString(_state->currentVolumeId());

    for (const auto& volumeId : _state->vpkg()->volumeIDs()) {
        auto volume = _state->vpkg()->volume(volumeId);
        if (!volume) {
            continue;
        }
        const QString id = QString::fromStdString(volumeId);
        const QString path = QString::fromStdString(volume->path().string());
        if (!isRasterizedLabelVolumePath(path)) {
            continue;
        }

        VolumeSelector::VolumeOption opt;
        opt.id = id;
        opt.name = QString::fromStdString(volume->name());
        opt.path = path;
        rasterizedVolumes.push_back(opt);

        if (defaultVolumeId.isEmpty()) {
            defaultVolumeId = id;
        }
        if (!currentVolumeId.isEmpty() && id == currentVolumeId) {
            defaultVolumeId = id;
        }
    }

    if (rasterizedVolumes.isEmpty()) {
        QMessageBox::warning(_parentWidget,
                             tr("Error"),
                             tr("No rasterized label volumes found.\n"
                                "Only volumes with meta.json field \"label_volume\": \"rasterized\" are supported."));
        return;
    }

    IgnoreLabelDialogResult params;
    if (!selectIgnoreLabelParams(_parentWidget, rasterizedVolumes, defaultVolumeId, &params)) {
        emit statusMessage(tr("Add ignore label cancelled"), 3000);
        return;
    }

    if (!isRasterizedLabelVolumePath(params.volumePath)) {
        QMessageBox::warning(_parentWidget,
                             tr("Error"),
                             tr("Selected volume is not a rasterized label volume."));
        return;
    }

    const QString executable = vc3d::findToolExecutable(QStringLiteral("vc_add_ignore_label"));
    if (executable.isEmpty()) {
        QMessageBox::warning(_parentWidget,
                             tr("Error"),
                             tr("vc_add_ignore_label tool not found. Configure tools/vc_add_ignore_label path."));
        return;
    }

    std::error_code ec;
    const std::filesystem::path volpkgDir(_state->vpkg()->getVolpkgDirectory());
    const std::filesystem::path volumesDir = volpkgDir / "volumes";
    std::filesystem::create_directories(volumesDir, ec);
    if (ec) {
        QMessageBox::critical(_parentWidget,
                              tr("Error"),
                              tr("Cannot create volumes directory: %1")
                                  .arg(QString::fromStdString(ec.message())));
        return;
    }

    QString outName = QFileInfo(params.outputName).fileName().trimmed();
    if (outName.isEmpty()) {
        outName = QStringLiteral("labels_ignore_%1.zarr")
                      .arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMddHHmmss")));
    }
    if (!outName.endsWith(QStringLiteral(".zarr"), Qt::CaseInsensitive)) {
        outName += QStringLiteral(".zarr");
    }

    const std::filesystem::path finalOutputRoot = volumesDir / outName.toStdString();
    const bool outputExists = std::filesystem::exists(finalOutputRoot, ec);
    if (ec) {
        QMessageBox::critical(_parentWidget,
                              tr("Error"),
                              tr("Cannot inspect output path: %1")
                                  .arg(QString::fromStdString(ec.message())));
        return;
    }
    if (outputExists && !std::filesystem::is_directory(finalOutputRoot, ec)) {
        QMessageBox::critical(_parentWidget,
                              tr("Error"),
                              tr("Output path exists and is not a directory: %1")
                                  .arg(QString::fromStdString(finalOutputRoot.string())));
        return;
    }

    // An existing output is updated in place; a new one the tool publishes
    // (staged, then renamed) with its meta.json.
    const bool reuseExistingOutput = outputExists;
    const QString finalOutputRootStr = QString::fromStdString(finalOutputRoot.string());

    QStringList args;
    args << params.volumePath
         << finalOutputRootStr
         << QStringLiteral("--mode") << QStringLiteral("chunk-alpha-wrap")
         << QStringLiteral("--chunk-alpha") << QString::number(params.chunkAlphaL0, 'g', 10)
         << QStringLiteral("--compute-level") << QStringLiteral("0")
         << QStringLiteral("--output-level") << QStringLiteral("0")
         << QStringLiteral("--skip-inner")
         << QStringLiteral("--verbose")
         << QStringLiteral("--ignore-value") << QString::number(params.ignoreValue)
         << QStringLiteral("--z-min") << QString::number(params.zMin);

    if (params.zMax >= 0) {
        args << QStringLiteral("--z-max") << QString::number(params.zMax);
    }
    if (params.workers > 0) {
        args << QStringLiteral("--workers") << QString::number(params.workers);
    }
    if (reuseExistingOutput) {
        args << QStringLiteral("--reuse-output-tree");
    }

    auto* runner = _cmdRunner;
    if (!runner) {
        QMessageBox::critical(_parentWidget, tr("Error"), tr("Command runner is not available."));
        return;
    }

    QPointer<QProgressDialog> progressDialog = new QProgressDialog(
        tr("Preparing add ignore label..."),
        tr("Cancel"),
        0,
        0,
        _parentWidget);
    progressDialog->setWindowTitle(tr("Add Ignore Label"));
    progressDialog->setWindowModality(Qt::WindowModal);
    progressDialog->setAutoClose(false);
    progressDialog->setAutoReset(false);
    progressDialog->setMinimumDuration(0);
    progressDialog->setRange(0, 0);
    progressDialog->setValue(0);
    progressDialog->setAttribute(Qt::WA_DeleteOnClose);
    progressDialog->show();

    QPointer<SegmentationCommandHandler> guard(this);
    auto progressState = std::make_shared<IgnoreLabelProgressState>();
    auto finishConnection = std::make_shared<QMetaObject::Connection>();
    auto outputConnection = std::make_shared<QMetaObject::Connection>();

    QObject::connect(progressDialog,
                     &QProgressDialog::canceled,
                     this,
                     [this, runner, progressDialog]() {
        if (progressDialog) {
            progressDialog->setRange(0, 0);
            progressDialog->setLabelText(tr("Canceling add ignore label..."));
        }
        runner->cancel();
        emit statusMessage(tr("Canceling add ignore label..."), 0);
    });

    *outputConnection = connect(runner,
                                &CommandLineToolRunner::consoleOutputReceived,
                                this,
                                [this, guard, progressDialog, progressState](const QString& output) {
        if (!guard) {
            return;
        }
        progressState->pendingOutput += output;

        while (true) {
            const int newline = progressState->pendingOutput.indexOf(QChar('\n'));
            if (newline < 0) {
                break;
            }

            QString line = progressState->pendingOutput.left(newline);
            progressState->pendingOutput.remove(0, newline + 1);
            line = line.trimmed();
            if (line.isEmpty()) {
                continue;
            }

            QString kind;
            QHash<QString, QString> fields;
            if (!parseStructuredProgressLine(line, &kind, &fields)) {
                continue;
            }

            if (kind == QStringLiteral("VC_STAGE")) {
                progressState->stageName = fields.value(QStringLiteral("name"));
                progressState->stageLevel = fields.value(QStringLiteral("level"), QStringLiteral("-1")).toInt();
                const int total = std::max(
                    1,
                    clampProgressCount(fields.value(QStringLiteral("total"), QStringLiteral("0")).toLongLong()));
                progressState->totalSteps = total;

                if (progressDialog) {
                    progressDialog->setRange(0, total);
                    progressDialog->setValue(0);
                    progressDialog->setLabelText(
                        ignoreLabelStageText(progressState->stageName,
                                             progressState->stageLevel,
                                             0,
                                             total));
                }
                emit statusMessage(ignoreLabelStageText(progressState->stageName,
                                                       progressState->stageLevel,
                                                       0,
                                                       total),
                                   0);
                continue;
            }

            if (kind == QStringLiteral("VC_PROGRESS")) {
                const QString stageName =
                    fields.value(QStringLiteral("name"), progressState->stageName);
                const int stageLevel =
                    fields.contains(QStringLiteral("level"))
                        ? fields.value(QStringLiteral("level")).toInt()
                        : progressState->stageLevel;
                const int total = std::max(
                    1,
                    clampProgressCount(fields.value(QStringLiteral("total"),
                                                    QString::number(progressState->totalSteps))
                                           .toLongLong()));
                const int current = std::clamp(
                    clampProgressCount(fields.value(QStringLiteral("current"), QStringLiteral("0")).toLongLong()),
                    0,
                    total);

                progressState->stageName = stageName;
                progressState->stageLevel = stageLevel;
                progressState->totalSteps = total;

                if (progressDialog) {
                    progressDialog->setRange(0, total);
                    progressDialog->setValue(current);
                    progressDialog->setLabelText(
                        ignoreLabelStageText(stageName, stageLevel, current, total));
                }
                continue;
            }

            if (kind == QStringLiteral("VC_SUMMARY") && progressDialog) {
                progressDialog->setRange(0, 0);
                progressDialog->setLabelText(tr("Finalizing output..."));
            }
        }
    });

    *finishConnection = connect(runner,
                          &CommandLineToolRunner::toolFinished,
                          this,
                          [this, guard, finishConnection, outputConnection, progressDialog,
                           reuseExistingOutput, finalOutputRootStr]
                          (CommandLineToolRunner::Tool tool,
                           bool success,
                           const QString& message,
                           const QString&,
                           bool) {
        if (!guard) {
            disconnect(*finishConnection);
            disconnect(*outputConnection);
            return;
        }
        if (tool != CommandLineToolRunner::Tool::CustomCommand) {
            return;
        }
        disconnect(*finishConnection);
        disconnect(*outputConnection);

        // The runner's completion handler reports a failure.
        if (!success) {
            emit statusMessage(tr("Add ignore label failed"), 3000);
        } else {
            emit statusMessage((reuseExistingOutput ? tr("Ignore label volume updated -> %1")
                                                    : tr("Ignore label volume created -> %1"))
                                   .arg(QDir::toNativeSeparators(finalOutputRootStr)),
                               5000);
        }
        if (progressDialog) {
            progressDialog->close();
        }
    });

    if (!runner->executeCustomCommand(executable, args, QStringLiteral("vc_add_ignore_label"))) {
        QObject::disconnect(*finishConnection);
        QObject::disconnect(*outputConnection);
        if (progressDialog) {
            progressDialog->close();
        }
        QMessageBox::critical(_parentWidget,
                              tr("Error"),
                              tr("Failed to start vc_add_ignore_label."));
        return;
    }

    emit statusMessage(tr("Add ignore label started at level 0..."), 0);
}
