#include "SegmentationCommandHandler.hpp"

#include "OpenDataCoordinateIdentity.hpp"
#include "CState.hpp"
#include "SurfacePanelController.hpp"
#include "VCSettings.hpp"
#include "OpenDataNormalGrids.hpp"
#include "OpenDataSegmentCache.hpp"

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
#include <QProcess>
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
#include <QProcessEnvironment>
#include <QProgressDialog>
#include <QFutureWatcher>
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
#include <QtConcurrent/QtConcurrentRun>
#include <QToolButton>
#include <QVector3D>
#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
#include <QStandardPaths>
#endif

#include "CommandLineToolRunner.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Segmentation.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include "vc/flattening/ABFFlattening.hpp"
#include "ToolDialogs.hpp"
#include "elements/VolumeSelector.hpp"
#include "elements/JsonProfilePresets.hpp"
#include "utils/Json.hpp"

// --------- locate executables (unified helper) --------------------------------

static bool isExecutableFile(const QString& path)
{
    QFileInfo fi(path);
    return fi.exists() && fi.isFile() && fi.isExecutable();
}

static QStringList applicationRelativeExecutablePaths(const QString& name)
{
#ifdef _WIN32
    const QString executableName = name.endsWith(QStringLiteral(".exe"), Qt::CaseInsensitive)
        ? name
        : name + QStringLiteral(".exe");
#else
    const QString executableName = name;
#endif

    const QString appDir = QCoreApplication::applicationDirPath();
    QStringList candidates{
        QDir(appDir).filePath(executableName),
        QDir(appDir).filePath(QStringLiteral("../") + executableName),
        QDir(appDir).filePath(QStringLiteral("../bin/") + executableName),
        QDir(appDir).filePath(QStringLiteral("../../bin/") + executableName),
        QDir(appDir).filePath(QStringLiteral("../libexec/") + executableName),
        QDir(appDir).filePath(QStringLiteral("../Resources/bin/") + executableName),
    };
    candidates.removeDuplicates();
    return candidates;
}

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

static QString findExecutable(
    const QString& name,
    const QStringList& extraPaths = {},
    const QString& envVar = {})
{
    // 1. Environment variable override (if provided)
    if (!envVar.isEmpty()) {
        const QByteArray envVal = qgetenv(envVar.toUtf8().constData());
        if (!envVal.isEmpty()) {
            const QString envPath = QString::fromLocal8Bit(envVal);
            if (isExecutableFile(envPath))
                return QFileInfo(envPath).absoluteFilePath();
        }
    }

    // 2. INI settings (tools/<name>_path, tools/<name>)
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        const QString key1 = QStringLiteral("tools/%1_path").arg(name);
        const QString key2 = QStringLiteral("tools/%1").arg(name);
        const QString iniPath =
            settings.value(key1, settings.value(key2)).toString().trimmed();
        if (!iniPath.isEmpty() && isExecutableFile(iniPath)) {
            return QFileInfo(iniPath).absoluteFilePath();
        }
    }

    // 3. Extra hard-coded paths (caller-supplied)
    for (const QString& p : extraPaths) {
        if (isExecutableFile(p))
            return QFileInfo(p).absoluteFilePath();
    }

    // 4. Binaries colocated with the GUI app or common install layouts
    for (const QString& p : applicationRelativeExecutablePaths(name)) {
        if (isExecutableFile(p))
            return QFileInfo(p).absoluteFilePath();
    }

    // 5. QStandardPaths / manual PATH walk
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

QString segmentsEntryLocationForPath(const QString& outputDir, const QString& volpkgRoot)
{
    const std::filesystem::path outputAbs =
        std::filesystem::absolute(std::filesystem::path(outputDir.toStdString())).lexically_normal();
    const std::filesystem::path rootAbs =
        std::filesystem::absolute(std::filesystem::path(volpkgRoot.toStdString())).lexically_normal();

    std::error_code relEc;
    const std::filesystem::path rel = std::filesystem::relative(outputAbs, rootAbs, relEc);
    if (!relEc && !rel.empty()) {
        const std::string relStr = rel.generic_string();
        if (relStr != "." && relStr.rfind("../", 0) != 0 && relStr != "..") {
            return QString::fromStdString(relStr);
        }
    }
    return QString::fromStdString(outputAbs.string());
}

std::optional<QString> openDataPatchesRootForVolume(const VolumePkg& pkg,
                                                     const QString& loadedVolumeId)
{
    if (loadedVolumeId.isEmpty() || !pkg.hasRemoteCacheRoot()) {
        return std::nullopt;
    }

    const auto tags = pkg.volumeTags(loadedVolumeId.toStdString());
    for (const auto& tag : tags) {
        if (tag.rfind(vc3d::opendata::kOpenDataSampleIdTagPrefix, 0) == 0) {
            const auto path = vc3d::opendata::openDataPatchesRoot(
                pkg.remoteCacheRootOrEmpty(),
                tag.substr(vc3d::opendata::kOpenDataSampleIdTagPrefix.size()));
            return QString::fromStdString(path.string());
        }
    }
    return std::nullopt;
}

bool selectGrowPatchSeedParams(QWidget* parent,
                               const QString& volpkgRoot,
                               const QVector<VolumeSelector::VolumeOption>& volumeOptions,
                               const QStringList& outputDirChoices,
                               const QHash<QString, QString>& openDataOutputRootsByVolumeId,
                               QString* selectedVolumeId,
                               QString* selectedVolumePath,
                               int* iterations,
                               double* minAreaCm,
                               QString* outputDir)
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

    auto* outputCombo = new QComboBox(&dlg);
    outputCombo->setEditable(true);
    outputCombo->addItems(outputDirChoices);
    if (outputDir && !outputDir->isEmpty()) {
        outputCombo->setCurrentText(*outputDir);
    } else if (!outputDirChoices.isEmpty()) {
        outputCombo->setCurrentText(outputDirChoices.front());
    }
    outputCombo->setProperty("vc_user_edited", false);
    QObject::connect(outputCombo, &QComboBox::editTextChanged, &dlg, [outputCombo]() {
        if (outputCombo->hasFocus()) {
            outputCombo->setProperty("vc_user_edited", true);
        }
    });

    auto selectedOpenDataOutputRoot = [&]() -> std::optional<QString> {
        const QString volumeId = volumeCombo->currentData().toString();
        const auto it = openDataOutputRootsByVolumeId.constFind(volumeId);
        if (it == openDataOutputRootsByVolumeId.cend()) {
            return std::nullopt;
        }
        return it.value();
    };

    QObject::connect(volumeCombo, &QComboBox::currentIndexChanged, &dlg, [=, &selectedOpenDataOutputRoot](int) {
        if (outputCombo->property("vc_user_edited").toBool()) {
            return;
        }
        if (const auto preferredOutput = selectedOpenDataOutputRoot()) {
            outputCombo->setCurrentText(*preferredOutput);
        }
    });

    auto* outputRow = new QWidget(&dlg);
    auto* outputLayout = new QHBoxLayout(outputRow);
    outputLayout->setContentsMargins(0, 0, 0, 0);
    outputLayout->addWidget(outputCombo, 1);
    auto* browseButton = new QToolButton(outputRow);
    browseButton->setText(QObject::tr("..."));
    outputLayout->addWidget(browseButton);
    form->addRow(QObject::tr("Output folder:"), outputRow);

    layout->addLayout(form);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
    layout->addWidget(buttons);

    QObject::connect(browseButton, &QToolButton::clicked, &dlg, [&]() {
        QString startDir;
        if (const auto openDataOutputRoot = selectedOpenDataOutputRoot()) {
            startDir = *openDataOutputRoot;
            // Catalog segment directories are immutable. Ensure the writable
            // sample patches directory exists so QFileDialog opens there.
            QDir().mkpath(startDir);
        } else {
            startDir = outputCombo->currentText().trimmed().isEmpty()
                ? volpkgRoot
                : outputCombo->currentText().trimmed();
        }
        const QString selected = QFileDialog::getExistingDirectory(
            &dlg,
            QObject::tr("Choose Output Folder"),
            startDir,
            QFileDialog::ShowDirsOnly);
        if (!selected.isEmpty()) {
            outputCombo->setCurrentText(selected);
        }
    });

    QObject::connect(buttons, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dlg, [&]() {
        const QString selected = outputCombo->currentText().trimmed();
        if (selected.isEmpty()) {
            QMessageBox::warning(&dlg, QObject::tr("Create Segment"), QObject::tr("Output folder cannot be empty."));
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
        if (outputDir) {
            *outputDir = selected;
        }
        dlg.accept();
    });

    return dlg.exec() == QDialog::Accepted;
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

static QJsonObject coordinateIdentityObject(
    const std::optional<vc3d::opendata::CoordinateIdentity>& identity)
{
    QJsonObject metadata;
    if (!identity)
        return metadata;
    metadata.insert(QStringLiteral("vc_open_data_coordinate_space"),
                    QString::fromStdString(identity->coordinateSpace));
    metadata.insert(QStringLiteral("vc_open_data_source_path"),
                    QString::fromStdString(identity->sourcePath));
    metadata.insert(QStringLiteral("vc_open_data_source_coordinate_level"),
                    identity->sourceCoordinateLevel);
    metadata.insert(QStringLiteral("vc_open_data_source_coordinate_scale_factor"),
                    static_cast<int>(identity->sourceCoordinateScaleFactor));
    metadata.insert(QStringLiteral("vc_open_data_source_original_resolution"),
                    identity->sourceOriginalResolution);
    return metadata;
}

static void copyCoordinateIdentity(QJsonObject& target,
                                   const QJsonObject& coordinateIdentity)
{
    for (auto it = coordinateIdentity.begin(); it != coordinateIdentity.end(); ++it)
        target.insert(it.key(), it.value());
}

static QJsonObject coordinateIdentityFromJson(const QJsonObject& source)
{
    QJsonObject metadata;
    static const std::array<QString, 5> keys{
        QStringLiteral("vc_open_data_coordinate_space"),
        QStringLiteral("vc_open_data_source_path"),
        QStringLiteral("vc_open_data_source_coordinate_level"),
        QStringLiteral("vc_open_data_source_coordinate_scale_factor"),
        QStringLiteral("vc_open_data_source_original_resolution"),
    };
    for (const auto& key : keys) {
        if (source.contains(key))
            metadata.insert(key, source.value(key));
    }
    return metadata;
}

static bool updateVolumeIdentityMetadata(
    const QString& volumePath,
    const QJsonObject& coordinateIdentity = {})
{
    if (volumePath.isEmpty()) {
        return false;
    }

    const QDir dir(volumePath);
    if (!dir.exists()) {
        return false;
    }

    const QString volumeId = QFileInfo(dir.path()).fileName();
    if (volumeId.isEmpty()) {
        return false;
    }

    const QString metaPath = dir.filePath(QStringLiteral("meta.json"));
    QJsonObject meta = readJsonObject(metaPath);
    if (meta.isEmpty()) {
        return false;
    }

    meta.insert(QStringLiteral("uuid"), volumeId);
    meta.insert(QStringLiteral("name"), volumeId);
    copyCoordinateIdentity(meta, coordinateIdentity);
    return writeJsonObject(metaPath, meta);
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

// Owns the lifecycle for the async SLIM run; deletes itself on finish/cancel
class SlimJob : public QObject {
public:
    SlimJob(QWidget* parentWidget,
            const QString& segDir,
            const QString& segmentStem,
            const QString& flatboiExe,
            SegmentationCommandHandler* handler,
            int iters,
            double tolerance,
            const QString& energy,
            double keepPercent,
            bool inpaintHoles,
            const QString& outputDir,
            double voxelSize,
            bool suppressDialogs = false)
    : QObject(handler)
    , parentWidget_(parentWidget)
    , handler_(handler)
    , segDir_(segDir)
    , stem_(segmentStem)
    , objPath_(QDir(segDir).filePath(segmentStem + (keepPercent < 100.0 ? "_coarse.obj" : ".obj")))
    , objFine_(QDir(segDir).filePath(segmentStem + ".obj"))
    , flatObj_(QDir(segDir).filePath(segmentStem + (keepPercent < 100.0 ? "_coarse_flatboi.obj" : "_flatboi.obj")))
    , liftedObj_(QDir(segDir).filePath(segmentStem + "_lifted.obj"))
    , outFinal_(outputDir)
    , outTemp_ (outputDir == segDir ? (segDir + "__rebuild_tmp__") : outputDir)
    , flatboiExe_(flatboiExe)
    , inputIsAlreadyFlat_(outputDir == segDir)
    , tolerance_(tolerance)
    , energy_(energy)
    , keepPercent_(keepPercent)
    , inpaintHoles_(inpaintHoles)
    , voxelSize_(voxelSize)
    , suppressDialogs_(suppressDialogs)
    , proc_(new QProcess(this))
    , progress_(new QProgressDialog(QObject::tr("Preparing SLIM..."), QObject::tr("Cancel"), 0, 0, parentWidget))
    , itRe_(R"(^\s*\[it\s+(\d+)\])", QRegularExpression::CaseInsensitiveOption)
    , progRe_(R"(^\s*PROGRESS\s+(\d+)\s*/\s*(\d+)\s*$)", QRegularExpression::CaseInsensitiveOption)
    {
        iters_ = iters > 0 ? iters : 20;

        tifxyz2objExe_ = findVcTool("vc_tifxyz2obj");
        obj2tifxyzExe_ = findVcTool("vc_obj2tifxyz");
        uvLiftExe_     = findVcTool("vc_obj_uv_lift");

        // never create outTemp_ here; we'll let vc_obj2tifxyz create it later
        if (QFileInfo::exists(outTemp_)) {
            QDir(outTemp_).removeRecursively();
        }

        proc_->setWorkingDirectory(segDir_);
        proc_->setProcessChannelMode(QProcess::MergedChannels);

        progress_->setWindowModality(Qt::WindowModal);
        progress_->setAutoClose(false);
        progress_->setAutoReset(true);
        // Suppressed dialogs use a huge minimum duration so the auto-show timer
        // never fires. Updates below remain harmless.
        progress_->setMinimumDuration(suppressDialogs_ ? std::numeric_limits<int>::max() : 0);
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

        if (handler_) {
            emit handler_->statusMessage(QObject::tr("Converting TIFXYZ to OBJ..."), 0);
            // Publish the shared flattening lifecycle.
            emit handler_->flattenJobStarted(QStringLiteral("flatten.slim"), stem_);
        }
        startToObj_();
    }

private:
    // Write (or update) meta.json in 'dir' so that it contains tifxyz identity
    // fields plus:
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
        if (!root.contains(QStringLiteral("type"))) {
            root.insert(QStringLiteral("type"), QStringLiteral("seg"));
        }
        if (!root.contains(QStringLiteral("uuid"))) {
            QString uuid = QFileInfo(dir).fileName();
            if (uuid.isEmpty()) {
                uuid = QDir(dir).dirName();
            }
            root.insert(QStringLiteral("uuid"), uuid);
        }
        if (!root.contains(QStringLiteral("format"))) {
            root.insert(QStringLiteral("format"), QStringLiteral("tifxyz"));
        }

        QFile out(metaPath);
        if (!out.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            return false;
        }
        out.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
        out.close();
        return true;
    }

    // Compute area_vx2 from the freshly-written tifxyz at 'dir' and write
    // both area_vx2 and area_cm2 into its meta.json. voxelSize is in micrometers
    // per voxel (same units as Volume::voxelSize()); area_cm2 = vx2 * vs^2 / 1e8
    // matches SurfaceAreaCalculator's convention. Returns true on success;
    // failures are non-fatal — caller may continue without updated area.
    static bool updateAreaInMeta_(const QString& dir, double voxelSize) {
        std::unique_ptr<QuadSurface> qs;
        try {
            qs = load_quad_from_tifxyz(dir.toStdString());
        } catch (...) {
            return false;
        }
        if (!qs) return false;

        const double area_vx2 = vc::surface::computeSurfaceAreaVox2(*qs);
        if (!std::isfinite(area_vx2) || area_vx2 <= 0.0) return false;

        double area_cm2 = std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(voxelSize) && voxelSize > 0.0) {
            area_cm2 = area_vx2 * voxelSize * voxelSize / 1e8;
        }

        const QString metaPath = QDir(dir).filePath(QStringLiteral("meta.json"));
        QJsonObject root;
        if (QFileInfo::exists(metaPath)) {
            QFile in(metaPath);
            if (in.open(QIODevice::ReadOnly)) {
                const auto doc = QJsonDocument::fromJson(in.readAll());
                if (doc.isObject()) root = doc.object();
                in.close();
            }
        }
        root.insert(QStringLiteral("area_vx2"), area_vx2);
        if (std::isfinite(area_cm2)) {
            root.insert(QStringLiteral("area_cm2"), area_cm2);
        }

        QFile out(metaPath);
        if (!out.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            return false;
        }
        out.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
        out.close();
        return true;
    }

private:
    enum class Phase { ToObj, Flatboi, ToObjFine, UVLift, ToTifxyz, Swap, Done };

    void startToObj_() {
        if (tifxyz2objExe_.isEmpty()) { showImmediateToolNotFound_("vc_tifxyz2obj"); return; }
        phase_ = Phase::ToObj;
        const bool decimating = keepPercent_ < 100.0;
        progress_->setLabelText(decimating
            ? QObject::tr("Converting TIFXYZ -> coarse OBJ (keep %1%)...")
                  .arg(keepPercent_, 0, 'f', 2)
            : QObject::tr("Converting TIFXYZ -> OBJ..."));
        progress_->setMaximum(1 + iters_ + 1);
        progress_->setValue(0);
        ioLog_.clear();
        QStringList args; args << segDir_ << objPath_;
        if (decimating) {
            args << QStringLiteral("--keep=%1").arg(keepPercent_, 0, 'f', 4);
        }
        if (inpaintHoles_) {
            args << QStringLiteral("--inpaint");
        }
        ioLog_ += QStringLiteral("Running: %1 %2\n").arg(tifxyz2objExe_, args.join(' '));
        proc_->start(tifxyz2objExe_, args);
    }

    void startToObjFine_() {
        if (tifxyz2objExe_.isEmpty()) { showImmediateToolNotFound_("vc_tifxyz2obj"); return; }
        phase_ = Phase::ToObjFine;
        progress_->setLabelText(QObject::tr("Converting TIFXYZ -> full-res OBJ for UV lift..."));
        ioLog_.clear();
        QStringList args; args << segDir_ << objFine_;
        if (inpaintHoles_) {
            args << QStringLiteral("--inpaint");
        }
        ioLog_ += QStringLiteral("Running: %1 %2\n").arg(tifxyz2objExe_, args.join(' '));
        proc_->start(tifxyz2objExe_, args);
    }

    void startUVLift_() {
        if (uvLiftExe_.isEmpty()) { showImmediateToolNotFound_("vc_obj_uv_lift"); return; }
        phase_ = Phase::UVLift;
        progress_->setLabelText(QObject::tr("Lifting UVs onto full-res mesh..."));
        ioLog_.clear();
        // Grid-space lift: needs the un-flattened coarse OBJ (grid UVs from
        // vc_tifxyz2obj) plus the flatboi output. 3D-nearest lift produced
        // streaks where the surface passes close to itself in voxel space.
        QStringList args; args << objPath_ << flatObj_ << objFine_ << liftedObj_;
        ioLog_ += QStringLiteral("Running: %1 %2\n").arg(uvLiftExe_, args.join(' '));
        proc_->start(uvLiftExe_, args);
    }

    void startFlatboi_() {
        phase_ = Phase::Flatboi;
        lastIterSeen_ = 0;
        progress_->setLabelText(QObject::tr("Running SLIM (flatboi)..."));
        progress_->setValue(1);
        ioLog_.clear();
        QStringList args;
        args << objPath_ << QString::number(iters_) << energy_;
        if (tolerance_ > 0.0) {
            args << QStringLiteral("--tol=%1").arg(tolerance_, 0, 'g', 8);
        }
        ioLog_ += QStringLiteral("Running: %1 %2\n").arg(flatboiExe_, args.join(' '));
        std::cout << "[slim-flatten] launching: " << flatboiExe_.toStdString()
                  << " " << args.join(' ').toStdString() << std::endl;
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
                const QString msg = QObject::tr("Output directory already exists and cannot be removed:\n%1")
                                      .arg(outTemp_);
                emitFinishedOnce_(false, msg, QString());
                if (!suppressDialogs_)
                    QMessageBox::critical(parentWidget_, QObject::tr("Error"), msg);
                cleanupAndDelete_();
                return;
            }
        }

        // Ensure parent directory exists; vc_obj2tifxyz will create outTemp_ itself
        const QString parentPath = QFileInfo(outTemp_).absolutePath();
        QDir parent(parentPath);
        if (!parent.exists() && !parent.mkpath(".")) {
            const QString msg = QObject::tr("Cannot create parent directory: %1").arg(parentPath);
            emitFinishedOnce_(false, msg, QString());
            if (!suppressDialogs_)
                QMessageBox::critical(parentWidget_, QObject::tr("Error"), msg);
            cleanupAndDelete_();
            return;
        }

        ioLog_.clear();
        // If decimation is in play, the UVs have been lifted onto the
        // full-res mesh (liftedObj_); otherwise flatObj_ itself is the
        // flattened full-res mesh.
        const QString srcObj = (keepPercent_ < 100.0) ? liftedObj_ : flatObj_;
        QStringList args;
        args << srcObj
             << outTemp_
             // The original tifxyz: its meta scale sizes the output grid to the
             // input sampling density (so the flattened tifxyz keeps the input
             // scale, no 1/scale blowup), and its approval.tif is resampled onto
             // the new grid via the grid-UV sidecar carried alongside srcObj.
             << QStringLiteral("--tifxyz-source=%1").arg(segDir_);
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
                if (!suppressDialogs_) {
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
    }

    void showDoneAndCleanup_() {
        if (progress_) {
            progress_->setValue(progress_->maximum());
            progress_->close();
        }

        emitFinishedOnce_(true,
                          QObject::tr("Flattened segment written to: %1").arg(outFinal_),
                          outFinal_);

        if (suppressDialogs_) {
            // Direct callers suppress the completion dialog.
            if (progress_) progress_->deleteLater();
            this->deleteLater();
            return;
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
        emitFinishedOnce_(false, QString(), QString());  // empty msg => cancel
        progress_->close();
        progress_->deleteLater();
        QTimer::singleShot(0, this, [this](){ this->deleteLater(); });
    }

    void onStdout_() {
        const QString chunk = QString::fromLocal8Bit(proc_->readAllStandardOutput());
        ioLog_ += chunk;
        std::cout << chunk.toStdString() << std::flush;
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
            case Phase::ToObj:     what = QObject::tr("vc_tifxyz2obj failed to start."); break;
            case Phase::Flatboi:   what = QObject::tr("flatboi failed to start.");       break;
            case Phase::ToObjFine: what = QObject::tr("vc_tifxyz2obj (full-res) failed to start."); break;
            case Phase::UVLift:    what = QObject::tr("vc_obj_uv_lift failed to start.");break;
            case Phase::ToTifxyz:  what = QObject::tr("vc_obj2tifxyz failed to start."); break;
            default: break;
        }
        const QString detail = what + "\n\n" + ioLog_.trimmed() + "\n\n" + why;
        emitFinishedOnce_(false, detail, QString());
        if (handler_) emit handler_->statusMessage(QObject::tr("SLIM-flatten failed"), 5000);
        if (suppressDialogs_) {
            cleanupAndDelete_();
            return;
        }
        QMessageBox* box = new QMessageBox(QMessageBox::Critical, QObject::tr("Error"),
                                           detail,
                                           QMessageBox::Ok, parentWidget_);
        box->setAttribute(Qt::WA_DeleteOnClose);
        QObject::connect(box, &QMessageBox::finished, this, [this]() { cleanupAndDelete_(); });
        box->open();
    }

    void onFinished_(int exitCode, QProcess::ExitStatus st) {
        if (errorShown_) return;

        // Error path
        if (st != QProcess::NormalExit || exitCode != 0) {
            const QString err = ioLog_.trimmed();
            QString what;
            switch (phase_) {
                case Phase::ToObj:     what = QObject::tr("vc_tifxyz2obj failed."); break;
                case Phase::Flatboi:   what = QObject::tr("flatboi failed.");       break;
                case Phase::ToObjFine: what = QObject::tr("vc_tifxyz2obj (full-res) failed."); break;
                case Phase::UVLift:    what = QObject::tr("vc_obj_uv_lift failed.");break;
                case Phase::ToTifxyz:  what = QObject::tr("vc_obj2tifxyz failed."); break;
                default: break;
            }
            errorShown_ = true;  // Prevent duplicate error dialogs
            const QString detail = what + (err.isEmpty()? QString() : ("\n\n" + err));
            emitFinishedOnce_(false, detail, QString());
            if (handler_) emit handler_->statusMessage(QObject::tr("SLIM-flatten failed"), 5000);
            if (suppressDialogs_) {
                if (QFileInfo::exists(outTemp_) && outTemp_ != outFinal_) {
                    QDir(outTemp_).removeRecursively();
                }
                if (progress_) { progress_->close(); progress_->deleteLater(); }
                this->deleteLater();
                return;
            }
            QMessageBox* box = new QMessageBox(QMessageBox::Critical, QObject::tr("Error"),
                                               detail,
                                               QMessageBox::Ok, parentWidget_);
            box->setAttribute(Qt::WA_DeleteOnClose);
            QObject::connect(box, &QMessageBox::finished, this, [this]() {
                if (QFileInfo::exists(outTemp_) && outTemp_ != outFinal_) {
                    QDir(outTemp_).removeRecursively();
                }
                if (progress_) { progress_->close(); progress_->deleteLater(); }
                this->deleteLater();
            });
            box->open();
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
            if (keepPercent_ < 100.0) {
                startToObjFine_();
            } else {
                startToTifxyz_();
            }
            return;
        }

        if (phase_ == Phase::ToObjFine) {
            if (!QFileInfo::exists(objFine_)) { onFinished_(1, QProcess::NormalExit); return; }
            startUVLift_();
            return;
        }

        if (phase_ == Phase::UVLift) {
            if (!QFileInfo::exists(liftedObj_)) { onFinished_(1, QProcess::NormalExit); return; }
            startToTifxyz_();
            return;
        }

        if (phase_ == Phase::ToTifxyz) {
            if (!QFileInfo::exists(outTemp_) || !QFileInfo(outTemp_).isDir()) {
                onFinished_(1, QProcess::NormalExit); return;
            }

            // Ensure the new tifxyz has a deterministic pixel size in meta.json
            // Requested: "scale": [0.05, 0.05]
            if (!overwriteMetaScale_(outTemp_, 0.05, 0.05) && !suppressDialogs_) {
                // Non-fatal: warn but continue with swap and completion.
                QMessageBox* warn = new QMessageBox(QMessageBox::Warning,
                    QObject::tr("Warning"),
                    QObject::tr("Converted directory created, but failed to update meta.json scale in:\n%1")
                        .arg(outTemp_),
                    QMessageBox::Ok, parentWidget_);
                warn->setAttribute(Qt::WA_DeleteOnClose);
                warn->open();
            }

            // Recompute area_vx2 / area_cm2 from the flattened tifxyz so the
            // segment list area column reflects the new geometry. Non-fatal:
            // if it fails, the meta.json keeps whatever area was already there.
            if (!updateAreaInMeta_(outTemp_, voxelSize_)) {
                std::cout << "[slim-flatten] warning: failed to compute area for "
                          << outTemp_.toStdString() << std::endl;
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
    QString objPath_;     // mesh fed to flatboi (coarse if decimate>0, else full-res)
    QString objFine_;     // full-res mesh used as UV-lift target when decimating
    QString flatObj_;     // flatboi output (coarse-flat when decimate>0)
    QString liftedObj_;   // full-res mesh with UVs lifted from flatObj_
    QString outFinal_;
    QString outTemp_;
    QString flatboiExe_;
    bool    inputIsAlreadyFlat_ = false;
    double  tolerance_ = 0.0;
    QString energy_ = QStringLiteral("symmetric_dirichlet");
    double  keepPercent_ = 100.0;
    bool    inpaintHoles_ = false;
    double  voxelSize_ = 0.0;
    bool    suppressDialogs_ = false;   // capture failures without presenting UI
    bool    finishedEmitted_ = false;   // guards single flattenJobFinished emit

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
    QString uvLiftExe_;

    bool errorShown_ = false;

    // Emit exactly one terminal event from every completion path.
    void emitFinishedOnce_(bool success, const QString& message,
                           const QString& outputPath) {
        if (finishedEmitted_) return;
        finishedEmitted_ = true;
        if (handler_) emit handler_->flattenJobFinished(success, message, outputPath);
    }

    void showImmediateToolNotFound_(const char* tool) {
        const QString msg = QObject::tr("Could not find the '%1' executable.\n"
                        "Tip: set VC.ini [tools] %1_path or ensure it's on PATH.").arg(tool);
        emitFinishedOnce_(false, msg, QString());
        if (!suppressDialogs_) {
            QMessageBox::critical(parentWidget_, QObject::tr("Error"), msg);
        }
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
           int iterations, int downsampleFactor = 1,
           bool suppressDialogs = false)
        : QObject(handler)
        , parentWidget_(parentWidget)
        , handler_(handler)
        , surfacePanel_(surfacePanel)
        , segDir_(segDir)
        , stem_(segmentStem)
        , outDir_(segDir.endsWith("_abf") ? segDir : (segDir + "_abf"))
        , iterations_(std::max(1, iterations))
        , downsampleFactor_(std::max(1, downsampleFactor))
        , suppressDialogs_(suppressDialogs)
        , cancelFlag_(std::make_shared<std::atomic_bool>(false))
        , watcher_(this)
        , progress_(new QProgressDialog(QObject::tr("ABF++ Flattening..."), QObject::tr("Cancel"), 0, 0, parentWidget))
    {
        progress_->setWindowModality(Qt::NonModal);
        // Suppressed dialogs must not be auto-shown.
        progress_->setMinimumDuration(suppressDialogs_ ? std::numeric_limits<int>::max() : 0);
        progress_->setRange(0, 0); // indeterminate
        progress_->setAttribute(Qt::WA_DeleteOnClose);

        connect(progress_, &QProgressDialog::canceled, this, &ABFJob::onCanceledRequested_);
        connect(&watcher_, &QFutureWatcher<ABFFlattenResult>::finished, this, &ABFJob::onFinished_);

        // Publish the shared flattening lifecycle.
        if (handler_)
            emit handler_->flattenJobStarted(QStringLiteral("flatten.abf"),
                                             stem_.isEmpty() ? outDir_ : stem_);

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
            emitFinishedOnce_(false, QString(), QString());  // empty msg => cancel
            deleteLater();
            return;
        }

        if (!result.success) {
            const QString errorMsg = result.errorMsg.isEmpty()
                ? QObject::tr("ABF++ flattening failed")
                : result.errorMsg;
            emitFinishedOnce_(false, errorMsg, QString());
            if (handler_) {
                emit handler_->statusMessage(QObject::tr("ABF++ flatten failed"), 5000);
                if (!suppressDialogs_)
                    QMessageBox::critical(parentWidget_, QObject::tr("ABF++ Flatten Failed"), errorMsg);
            }
            deleteLater();
            return;
        }

        const QString label = !stem_.isEmpty() ? stem_ : outDir_;

        emitFinishedOnce_(true,
                          QObject::tr("ABF++ flatten complete: %1").arg(label),
                          outDir_);
        if (handler_) {
            emit handler_->statusMessage(QObject::tr("ABF++ flatten complete: %1").arg(label), 5000);
            if (!suppressDialogs_)
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

    void emitFinishedOnce_(bool success, const QString& message,
                           const QString& outputPath) {
        if (finishedEmitted_) return;
        finishedEmitted_ = true;
        if (handler_) emit handler_->flattenJobFinished(success, message, outputPath);
    }

    QWidget* parentWidget_ = nullptr;
    QPointer<SegmentationCommandHandler> handler_;
    QPointer<SurfacePanelController> surfacePanel_;
    QString segDir_;
    QString stem_;
    QString outDir_;
    int iterations_;
    int downsampleFactor_;
    bool suppressDialogs_ = false;
    bool finishedEmitted_ = false;
    std::shared_ptr<std::atomic_bool> cancelFlag_;
    QFutureWatcher<ABFFlattenResult> watcher_;
    QPointer<QProgressDialog> progress_;
};

// vc_straighten runs directly on tifxyz dirs (no OBJ round-trip), so this is a
// single subprocess: stream its stdout into the progress label, then reload
// the surface list so the new surface appears.
class StraightenJob : public QObject {
    Q_OBJECT
public:
    StraightenJob(QWidget* parentWidget, SurfacePanelController* surfacePanel,
                  SegmentationCommandHandler* handler, const QString& straightenExe,
                  const QString& segDir, const QString& segmentStem,
                  const QString& outputDir, const QStringList& extraArgs,
                  bool suppressDialogs = false)
        : QObject(handler)
        , parentWidget_(parentWidget)
        , handler_(handler)
        , surfacePanel_(surfacePanel)
        , stem_(segmentStem)
        , outDir_(outputDir)
        , suppressDialogs_(suppressDialogs)
        , proc_(new QProcess(this))
        , progress_(new QProgressDialog(QObject::tr("Straightening..."), QObject::tr("Cancel"), 0, 0, parentWidget))
    {
        progress_->setWindowModality(Qt::NonModal);
        // Suppressed dialogs must not be auto-shown.
        progress_->setMinimumDuration(suppressDialogs_ ? std::numeric_limits<int>::max() : 0);
        progress_->setRange(0, 0); // indeterminate; vc_straighten emits no PROGRESS
        progress_->setAttribute(Qt::WA_DeleteOnClose);
        connect(progress_, &QProgressDialog::canceled, this, &StraightenJob::onCanceled_);

        proc_->setProcessChannelMode(QProcess::MergedChannels);
        connect(proc_, &QProcess::readyReadStandardOutput, this, &StraightenJob::onStdout_);
        connect(proc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
                this, &StraightenJob::onFinished_);
        connect(proc_, &QProcess::errorOccurred, this, &StraightenJob::onProcError_);

        QStringList args;
        args << segDir << outputDir << extraArgs;
        std::cout << "[straighten] launching: " << straightenExe.toStdString()
                  << " " << args.join(' ').toStdString() << std::endl;
        if (handler_) {
            emit handler_->statusMessage(QObject::tr("Running vc_straighten..."), 0);
            // Publish the shared flattening lifecycle.
            emit handler_->flattenJobStarted(QStringLiteral("flatten.straighten"),
                                             stem_.isEmpty() ? outDir_ : stem_);
        }
        proc_->start(straightenExe, args);
    }

private slots:
    void onStdout_() {
        const QString chunk = QString::fromLocal8Bit(proc_->readAllStandardOutput());
        const QStringList lines = chunk.split('\n', Qt::SkipEmptyParts);
        for (const QString& raw : lines) {
            const QString line = raw.trimmed();
            if (line.isEmpty()) continue;
            std::cout << "[straighten] " << line.toStdString() << std::endl;
            if (progress_) progress_->setLabelText(line);
            if (handler_) emit handler_->statusMessage(line, 0);
        }
    }

    void onCanceled_() {
        if (proc_ && proc_->state() != QProcess::NotRunning) proc_->kill();
    }

    void onProcError_(QProcess::ProcessError e) {
        // A crash is reported through finished(); only the hard start failure
        // needs handling here.
        if (e != QProcess::FailedToStart || done_) return;
        done_ = true;
        const QString msg = QObject::tr("Could not launch vc_straighten.");
        emitFinishedOnce_(false, msg, QString());
        if (handler_) emit handler_->statusMessage(QObject::tr("Straighten failed"), 5000);
        if (progress_) progress_->close();
        if (!suppressDialogs_)
            QMessageBox::critical(parentWidget_, QObject::tr("Straighten Failed"), msg);
        deleteLater();
    }

    void onFinished_(int code, QProcess::ExitStatus status) {
        if (done_) return;
        done_ = true;
        if (progress_) progress_->close();
        if (status != QProcess::NormalExit || code != 0) {
            const QString msg = QObject::tr("vc_straighten exited with code %1.\nSee the console for details.").arg(code);
            emitFinishedOnce_(false, msg, QString());
            if (handler_) emit handler_->statusMessage(QObject::tr("Straighten failed"), 5000);
            if (!suppressDialogs_)
                QMessageBox::critical(parentWidget_, QObject::tr("Straighten Failed"), msg);
            deleteLater();
            return;
        }
        const QString label = !stem_.isEmpty() ? stem_ : outDir_;
        emitFinishedOnce_(true,
                          QObject::tr("Straighten complete: %1").arg(label), outDir_);
        if (handler_) emit handler_->statusMessage(QObject::tr("Straighten complete: %1").arg(label), 5000);
        if (!suppressDialogs_)
            QMessageBox::information(parentWidget_, QObject::tr("Straighten Complete"),
                QObject::tr("Straightened surface saved to:\n%1").arg(outDir_));
        if (surfacePanel_) {
            QMetaObject::invokeMethod(surfacePanel_.data(),
                                      &SurfacePanelController::reloadSurfacesFromDisk,
                                      Qt::QueuedConnection);
        }
        deleteLater();
    }

private:
    void emitFinishedOnce_(bool success, const QString& message,
                           const QString& outputPath) {
        if (finishedEmitted_) return;
        finishedEmitted_ = true;
        if (handler_) emit handler_->flattenJobFinished(success, message, outputPath);
    }

    QWidget* parentWidget_ = nullptr;
    QPointer<SegmentationCommandHandler> handler_;
    QPointer<SurfacePanelController> surfacePanel_;
    QString stem_;
    QString outDir_;
    bool suppressDialogs_ = false;
    bool finishedEmitted_ = false;
    QProcess* proc_ = nullptr;
    QPointer<QProgressDialog> progress_;
    bool done_ = false;
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

QString SegmentationCommandHandler::getCurrentRenderVolumePath(QString* remoteUrlOut) const
{
    if (remoteUrlOut) {
        remoteUrlOut->clear();
    }

    auto volume = _state ? _state->currentVolume() : nullptr;
    if (!volume) {
        return QString();
    }

    if (volume->isRemote()) {
        const QString remoteUrl = QString::fromStdString(volume->remoteLocator());
        if (remoteUrlOut) {
            *remoteUrlOut = remoteUrl;
        }
        return remoteUrl;
    }

    return QString::fromStdString(volume->path().string());
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

void SegmentationCommandHandler::configureCommandRunnerRemoteAuthForVolumePath(const QString& volumePath)
{
    if (!_cmdRunner) {
        return;
    }

    _cmdRunner->setRemoteVolumeUrl(QString());
    _cmdRunner->setRemoteVolumeAuth(QString(), QString(), QString(), QString());

    if (!_state || !_state->vpkg() || volumePath.isEmpty()) {
        return;
    }

    for (const auto& volumeId : _state->vpkg()->volumeIDs()) {
        auto volume = _state->vpkg()->volume(volumeId);
        if (!volume || !volume->isRemote()) {
            continue;
        }
        if (commandPathForVolume(volume) != volumePath) {
            continue;
        }

        const auto& auth = volume->remoteAuth();
        _cmdRunner->setRemoteVolumeUrl(QString::fromStdString(volume->remoteLocator()));
        _cmdRunner->setRemoteVolumeAuth(QString::fromStdString(auth.access_key),
                                        QString::fromStdString(auth.secret_key),
                                        QString::fromStdString(auth.session_token),
                                        QString::fromStdString(auth.region));
        return;
    }
}

void SegmentationCommandHandler::onRenderSegment(const std::string& segmentId)
{
    auto* surface = requireSurfaceAndRunner(segmentId, false);
    if (!surface) return;

    auto surf = _state->vpkg()->getSurface(segmentId);

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    auto renderVolume = _state ? _state->currentVolume() : nullptr;
    QString remoteVolumeUrl;
    const QString volumePath = getCurrentRenderVolumePath(&remoteVolumeUrl);
    const QString segmentPath = QString::fromStdString(surf->path.string());
    const QString segmentOutDir = QString::fromStdString(surf->path.string());
    const QString outputFormat = "%s/layers";
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
    // Interactive renders always start from the TIFF-stack default.
    _cmdRunner->setRenderOutputFormat(CommandLineToolRunner::RenderOutputFormat::TifStack);
    _cmdRunner->setRenderParams(static_cast<float>(dlg.scale()), dlg.groupIdx(), dlg.numSlices());
    _cmdRunner->setRenderVoxelSize(
        renderVolume ? renderVolume->voxelSize() : 0.0,
        renderVolume &&
            (renderVolume->baseScaleLevel() > 0 ||
             renderVolume->hasExplicitVoxelSizeOverride()));
    _cmdRunner->setNextOmpThreads(dlg.ompThreads());
    _cmdRunner->setVolumePath(dlg.volumePath());
    const bool useRemoteVolume = dlg.volumePath() == volumePath && !remoteVolumeUrl.isEmpty();
    _cmdRunner->setRemoteVolumeUrl(useRemoteVolume ? remoteVolumeUrl : QString());
    if (useRemoteVolume && renderVolume && renderVolume->isRemote()) {
        const auto& auth = renderVolume->remoteAuth();
        _cmdRunner->setRemoteVolumeAuth(QString::fromStdString(auth.access_key),
                                        QString::fromStdString(auth.secret_key),
                                        QString::fromStdString(auth.session_token),
                                        QString::fromStdString(auth.region));
    } else {
        _cmdRunner->setRemoteVolumeAuth(QString(), QString(), QString(), QString());
    }
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

    const QString defaultOutput = segDir.endsWith("_flatboi") ? segDir : (segDir + "_flatboi");
    SlimFlattenDialog dlg(_parentWidget, defaultOutput);
    if (dlg.exec() != QDialog::Accepted) {
        return;
    }
    const int iters = dlg.maxIterations();
    const double tol = dlg.tolerance();
    const QString energy = dlg.energyType();
    const double keepPercent = dlg.keepPercent();
    const bool inpaintHoles = dlg.inpaintHoles();
    const QString outputDir = dlg.outputPath();

    const QByteArray pastixEnv = qgetenv("PASTIX_NUM_THREADS");
    const unsigned hwConc = std::thread::hardware_concurrency();
    std::cout << "[slim-flatten] segment=" << segmentId
              << " dir=" << segDirFs.string()
              << " out=" << outputDir.toStdString()
              << " flatboi=" << flatboiExe.toStdString()
              << " iters=" << iters
              << " tol=" << tol
              << " energy=" << energy.toStdString()
              << " keep_percent=" << keepPercent
              << " inpaint=" << (inpaintHoles ? "true" : "false")
              << " PASTIX_NUM_THREADS=" << (pastixEnv.isEmpty() ? "<unset, PaStiX auto>" : pastixEnv.toStdString())
              << " hardware_concurrency=" << hwConc
              << std::endl;

    double voxelSize = 0.0;
    try {
        if (auto volume = _state ? _state->currentVolume() : nullptr) {
            voxelSize = volume->voxelSize();
        }
    } catch (...) {}
    if (!std::isfinite(voxelSize) || voxelSize <= 0.0) voxelSize = 0.0;

    new SlimJob(_parentWidget, segDir, segmentStem, flatboiExe, this, iters, tol, energy, keepPercent, inpaintHoles, outputDir, voxelSize);
}

void SegmentationCommandHandler::onStraighten(const std::string& segmentId)
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

    const QString exe = findVcTool("vc_straighten");
    if (exe.isEmpty()) {
        QMessageBox::critical(_parentWidget, tr("Error"),
            tr("Could not find the 'vc_straighten' executable.\n"
               "Looked in known locations and PATH.\n\n"
               "Tip: set an override via VC.ini [tools] vc_straighten, or put "
               "the binary on PATH."));
        emit statusMessage(tr("Straighten failed"), 5000);
        return;
    }

    const QString defaultOutput = segDir.endsWith("_straightened")
        ? (segDir + "_v2") : (segDir + "_straightened");
    StraightenDialog dlg(_parentWidget, defaultOutput);
    if (dlg.exec() != QDialog::Accepted) {
        return;
    }
    const QString outputDir = dlg.outputPath();
    if (QFileInfo::exists(outputDir)) {
        QMessageBox::warning(_parentWidget, tr("Straighten"),
            tr("Output directory already exists:\n%1\n\nvc_straighten will not "
               "overwrite it; choose a different name.").arg(outputDir));
        return;
    }
    const QStringList args = dlg.toArgs();
    std::cout << "[straighten] segment=" << segmentId
              << " dir=" << segDirFs.string()
              << " out=" << outputDir.toStdString()
              << " exe=" << exe.toStdString()
              << " args=" << args.join(' ').toStdString()
              << std::endl;

    new StraightenJob(_parentWidget, _surfacePanel, this, exe, segDir, segmentStem, outputDir, args);
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
    // The dialog collects inputs; startRunTrace owns validation and launch.
    auto* surface = requireSurfaceAndRunner(segmentId, true);
    if (!surface) return;
    if (_state && _state->currentVolume() && _state->currentVolume()->isRemote()) {
        QMessageBox::warning(
            _parentWidget,
            tr("Unsupported Remote Volume"),
            tr("Run Trace uses vc_grow_seg_from_segments, which accepts only local volumes. "
               "The remote volume locator was not modified or passed to the tool."));
        return;
    }

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

    RunTraceParams params;
    params.paramOverrides = dlg.makeParamsJson();
    params.ompThreads = dlg.ompThreads();
    params.tgtDir = dlg.tgtDir();

    CommandLaunchError error;
    if (!startRunTraceImpl(segmentId, params, /*interactive=*/true, &error, nullptr)) {
        QMessageBox::warning(_parentWidget, tr("Error"), error.message);
        return;
    }

    emit statusMessage(tr("Growing segment from: %1").arg(QString::fromStdString(segmentId)), 5000);
}

bool SegmentationCommandHandler::startRunTrace(const std::string& segmentId,
                                              const RunTraceParams& params,
                                              CommandLaunchError* error,
                                              QString* resolvedOutputDir)
{
    return startRunTraceImpl(segmentId, params, /*interactive=*/false,
                             error, resolvedOutputDir);
}

bool SegmentationCommandHandler::startRunTraceImpl(
    const std::string& segmentId,
    const RunTraceParams& params,
    bool interactive,
    CommandLaunchError* error,
    QString* resolvedOutputDir)
{
    const CommandLaunchErrorSetter setErr{error};

    // Validate without opening dialogs.
    if (!_state || _state->currentVolume() == nullptr || !_state->vpkg()) {
        setErr(tr("No volume package or volume loaded."), CommandLaunchError::InvalidState);
        return false;
    }
    auto surf = _state->vpkg()->getSurface(segmentId);
    if (!surf) {
        setErr(tr("Invalid segment or segment not loaded: %1")
                   .arg(QString::fromStdString(segmentId)),
               CommandLaunchError::SegmentNotFound);
        return false;
    }
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

    const QString srcSegment = QString::fromStdString(surf->path.string());

    const std::filesystem::path volpkgPath =
        std::filesystem::path(_state->vpkgPath().toStdString());
    const std::filesystem::path tracesDir = volpkgPath / "traces";
    const std::filesystem::path jsonParamsPath = volpkgPath / "trace_params.json";
    const std::filesystem::path pathsDir = volpkgPath / "paths";

    // Resolve target dir: params.tgtDir (absolute or relative to volpkg root) or
    // <volpkg>/traces by default; created if missing.
    std::filesystem::path tgtDir = tracesDir;
    if (!params.tgtDir.isEmpty()) {
        std::filesystem::path requested(params.tgtDir.toStdString());
        tgtDir = requested.is_absolute() ? requested : (volpkgPath / requested);
    }
    if (!std::filesystem::exists(tgtDir)) {
        try { std::filesystem::create_directories(tgtDir); }
        catch (const std::exception& e) {
            setErr(tr("Failed to create traces directory: %1").arg(e.what()));
            return false;
        }
    }

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

    const QString mergedJsonPath =
        QDir(QString::fromStdString(tgtDir.string())).filePath(QStringLiteral("trace_params_ui.json"));
    {
        QFile f(mergedJsonPath);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            setErr(tr("Failed to write params JSON: %1").arg(mergedJsonPath));
            return false;
        }
        f.write(QJsonDocument(base).toJson(QJsonDocument::Indented));
        f.close();
    }

    _cmdRunner->setTraceParams(
        getCurrentVolumePath(),
        QString::fromStdString(pathsDir.string()),
        QString::fromStdString(tgtDir.string()),
        mergedJsonPath,
        srcSegment);
    _cmdRunner->setNextOmpThreads(params.ompThreads);
    const auto options = interactive
        ? CommandLineToolRunner::ExecutionOptions{}
        : CommandLineToolRunner::ExecutionOptions::silent();
    if (!_cmdRunner->execute(
            CommandLineToolRunner::Tool::GrowSegFromSegment, options)) {
        setErr(tr("Failed to start Run Trace."));
        return false;
    }

    if (resolvedOutputDir)
        *resolvedOutputDir = QString::fromStdString(tgtDir.string());
    return true;
}

bool SegmentationCommandHandler::startRenderSegment(const std::string& segmentId,
                                                    const RenderSegmentParams& params,
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
    auto surf = _state->vpkg()->getSurface(segmentId);
    if (!surf) {
        setErr(tr("Invalid segment or segment not loaded: %1")
                   .arg(QString::fromStdString(segmentId)),
               CommandLaunchError::SegmentNotFound);
        return false;
    }
    if (!_cmdRunner) {
        setErr(tr("Command line tools not available"), CommandLaunchError::ToolUnavailable);
        return false;
    }
    if (_cmdRunner->isRunning()) {
        setErr(tr("A command line tool is already running."), CommandLaunchError::Busy);
        return false;
    }

    // Return missing-tool errors before execute() reaches its dialog path.
    const QString toolPath =
        QCoreApplication::applicationDirPath() + QStringLiteral("/vc_render_tifxyz");
    QFileInfo toolInfo(toolPath);
    if (!toolInfo.exists() || !toolInfo.isExecutable()) {
        setErr(tr("vc_render_tifxyz not found or not executable: %1").arg(toolPath),
               CommandLaunchError::ToolUnavailable);
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

    // Resolve the output base relative to the package and create it if needed.
    const std::filesystem::path volpkgPath(_state->vpkgPath().toStdString());
    std::filesystem::path baseDir;
    if (params.outputDir.isEmpty()) {
        baseDir = surf->path;  // the segment folder
    } else {
        std::filesystem::path requested(params.outputDir.toStdString());
        baseDir = requested.is_absolute() ? requested : (volpkgPath / requested);
    }
    if (!std::filesystem::exists(baseDir)) {
        try { std::filesystem::create_directories(baseDir); }
        catch (const std::exception& e) {
            setErr(tr("Failed to create output directory: %1").arg(e.what()));
            return false;
        }
    }

    const bool wantZarr =
        params.outputFormat == CommandLineToolRunner::RenderOutputFormat::Zarr;
    const std::filesystem::path outputArtifact =
        baseDir / (wantZarr ? "surface.zarr" : "layers");
    const QString outputPattern = QString::fromStdString(outputArtifact.string());

    // Reset options omitted from this API so prior UI settings cannot leak in.
    _cmdRunner->setSegmentPath(QString::fromStdString(surf->path.string()));
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
    configureCommandRunnerRemoteAuthForVolumePath(volumePath);

    _cmdRunner->setRenderAdvanced(0, 0, 0, 0, QString(), false, 1.0f, 0.0, -1);
    _cmdRunner->setIncludeTifs(false);
    _cmdRunner->setFlattenOptions(false, 10, 1);
    if (!_cmdRunner->execute(
            CommandLineToolRunner::Tool::RenderTifXYZ,
            CommandLineToolRunner::ExecutionOptions::silent())) {
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
                                                  QString* resolvedOutputDir)
{
    const CommandLaunchErrorSetter setErr{error};

    if (!_state || !_state->vpkg() || !_state->currentVolume()) {
        setErr(tr("No volume package or volume loaded."), CommandLaunchError::InvalidState);
        return false;
    }
    auto surf = _state->vpkg()->getSurface(segmentId);
    if (!surf) {
        setErr(tr("Invalid segment or segment not loaded: %1")
                   .arg(QString::fromStdString(segmentId)),
               CommandLaunchError::SegmentNotFound);
        return false;
    }

    // Resolve the full pipeline before constructing a job; vc_obj_uv_lift is
    // needed only when decimating.
    const QString flatboiExe = findFlatboiExecutable();
    if (flatboiExe.isEmpty()) {
        setErr(tr("flatboi not found or not executable (checked known locations, "
                  "PATH, VC.ini [tools] flatboi_path, and $FLATBOI)."),
               CommandLaunchError::ToolUnavailable);
        return false;
    }
    if (findVcTool("vc_tifxyz2obj").isEmpty()) {
        setErr(tr("vc_tifxyz2obj not found or not executable."),
               CommandLaunchError::ToolUnavailable);
        return false;
    }
    if (findVcTool("vc_obj2tifxyz").isEmpty()) {
        setErr(tr("vc_obj2tifxyz not found or not executable."),
               CommandLaunchError::ToolUnavailable);
        return false;
    }
    const bool decimating = params.keepPercent < 100.0;
    if (decimating && findVcTool("vc_obj_uv_lift").isEmpty()) {
        setErr(tr("vc_obj_uv_lift not found or not executable (required when "
                  "keepPercent < 100)."),
               CommandLaunchError::ToolUnavailable);
        return false;
    }

    const std::filesystem::path segDirFs = surf->path;
    const QString segDir = QString::fromStdString(segDirFs.string());
    const QString segmentStem = QString::fromStdString(segmentId);

    // Relative output paths are rooted at the volume package.
    QString outputDir;
    if (params.outputDir.isEmpty()) {
        outputDir = segDir.endsWith("_flatboi") ? segDir : (segDir + "_flatboi");
    } else {
        std::filesystem::path requested(params.outputDir.toStdString());
        const std::filesystem::path volpkgPath(_state->vpkgPath().toStdString());
        outputDir = QString::fromStdString(
            (requested.is_absolute() ? requested : (volpkgPath / requested)).string());
    }

    double voxelSize = 0.0;
    try {
        if (auto volume = _state->currentVolume()) voxelSize = volume->voxelSize();
    } catch (...) {}
    if (!std::isfinite(voxelSize) || voxelSize <= 0.0) voxelSize = 0.0;

    const int iters = params.iterations > 0 ? params.iterations : 50;

    // The job reports completion through flattenJobFinished and self-deletes.
    new SlimJob(_parentWidget, segDir, segmentStem, flatboiExe, this, iters,
                params.tolerance, params.energyType, params.keepPercent,
                params.inpaintHoles, outputDir, voxelSize,
                /*suppressDialogs=*/true);

    if (resolvedOutputDir) *resolvedOutputDir = outputDir;
    return true;
}

bool SegmentationCommandHandler::startAbfFlatten(const std::string& segmentId,
                                                 int iterations,
                                                 int downsampleFactor,
                                                 CommandLaunchError* error,
                                                 QString* resolvedOutputDir)
{
    const CommandLaunchErrorSetter setErr{error};

    if (!_state || !_state->vpkg()) {
        setErr(tr("No volume package loaded."), CommandLaunchError::InvalidState);
        return false;
    }
    auto surf = _state->vpkg()->getSurface(segmentId);
    if (!surf) {
        setErr(tr("Invalid segment or segment not loaded: %1")
                   .arg(QString::fromStdString(segmentId)),
               CommandLaunchError::SegmentNotFound);
        return false;
    }

    const std::filesystem::path segDirFs = surf->path;
    const QString segDir = QString::fromStdString(segDirFs.string());
    const QString segmentStem = QString::fromStdString(segmentId);
    const QString outDir = segDir.endsWith("_abf") ? segDir : (segDir + "_abf");

    // ABF++ runs in-process and reports completion through flattenJobFinished.
    new ABFJob(_parentWidget, _surfacePanel, this, segDir, segmentStem,
               std::max(1, iterations), std::max(1, downsampleFactor),
               /*suppressDialogs=*/true);

    if (resolvedOutputDir) *resolvedOutputDir = outDir;
    return true;
}

bool SegmentationCommandHandler::startStraighten(const std::string& segmentId,
                                                 const StraightenParams& params,
                                                 CommandLaunchError* error,
                                                 QString* resolvedOutputDir)
{
    const CommandLaunchErrorSetter setErr{error};

    if (!_state || !_state->vpkg() || !_state->currentVolume()) {
        setErr(tr("No volume package or volume loaded."), CommandLaunchError::InvalidState);
        return false;
    }
    auto surf = _state->vpkg()->getSurface(segmentId);
    if (!surf) {
        setErr(tr("Invalid segment or segment not loaded: %1")
                   .arg(QString::fromStdString(segmentId)),
               CommandLaunchError::SegmentNotFound);
        return false;
    }

    const QString exe = findVcTool("vc_straighten");
    if (exe.isEmpty()) {
        setErr(tr("vc_straighten not found or not executable."),
               CommandLaunchError::ToolUnavailable);
        return false;
    }

    const std::filesystem::path segDirFs = surf->path;
    const QString segDir = QString::fromStdString(segDirFs.string());
    const QString segmentStem = QString::fromStdString(segmentId);

    QString outputDir;
    if (params.outputDir.isEmpty()) {
        outputDir = segDir.endsWith("_straightened") ? (segDir + "_v2")
                                                     : (segDir + "_straightened");
    } else {
        std::filesystem::path requested(params.outputDir.toStdString());
        const std::filesystem::path volpkgPath(_state->vpkgPath().toStdString());
        outputDir = QString::fromStdString(
            (requested.is_absolute() ? requested : (volpkgPath / requested)).string());
    }

    // vc_straighten refuses to overwrite an existing output directory.
    if (QFileInfo::exists(outputDir)) {
        setErr(tr("Output directory already exists: %1 (vc_straighten will not "
                  "overwrite it; choose a different name).").arg(outputDir));
        return false;
    }

    // Keep argument construction aligned with StraightenDialog::toArgs().
    QStringList args;
    if (params.unbend) {
        args << QStringLiteral("--unbend");
        args << QStringLiteral("--unbend-smooth-cols")
             << QString::number(params.unbendSmoothCols, 'f', 0);
    }
    args << QStringLiteral("--overlap-pairs") << QString::number(params.overlapPasses);
    if (params.orthogonalize) args << QStringLiteral("--orthogonalize");
    if (params.trim) {
        args << QStringLiteral("--trim");
        args << QStringLiteral("--trim-max-edge")
             << QString::number(params.trimMaxEdge, 'f', 0);
    }

    new StraightenJob(_parentWidget, _surfacePanel, this, exe, segDir, segmentStem,
                      outputDir, args, /*suppressDialogs=*/true);

    if (resolvedOutputDir) *resolvedOutputDir = outputDir;
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

    QString normalGridPath = _normalGridPathGetter ? _normalGridPathGetter() : QString();
    if (normalGridPath.isEmpty() && _state && _state->vpkg()) {
        const auto paths = _state->vpkg()->normalGridPaths();
        if (!paths.empty()) normalGridPath = QString::fromStdString(paths.front().string());
    }

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
        return;
    }

    const QString dirName = QFileInfo(job.resumeSurfacePath).fileName();
    emit statusMessage(tr("Copy %1 started for %2")
                                 .arg(copyOut ? tr("out") : tr("in"))
                                 .arg(dirName.isEmpty() ? segmentId : dirName),
                             5000);
}

QJsonObject SegmentationCommandHandler::buildResumeLocalBaseParamsJson() const
{
    QString normalGridPath = _normalGridPathGetter ? _normalGridPathGetter() : QString();
    if (normalGridPath.isEmpty() && _state && _state->vpkg()) {
        const auto paths = _state->vpkg()->normalGridPaths();
        if (!paths.empty()) normalGridPath = QString::fromStdString(paths.front().string());
    }

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

QString SegmentationCommandHandler::resolveSegmentOutputDir(
    const std::filesystem::path& surfacePath, QString* errorMessage) const
{
    QString volpkgRoot = _state->vpkgPath();
    if (volpkgRoot.isEmpty()) {
        volpkgRoot = QString::fromStdString(_state->vpkg()->getVolpkgDirectory());
    }

    QString outputDirPath = QString::fromStdString(surfacePath.parent_path().string());
    if (outputDirPath.isEmpty()) {
        outputDirPath = QDir(volpkgRoot).filePath(QStringLiteral("paths"));
    }
    QDir outDir(outputDirPath);
    if (!outDir.exists() && !outDir.mkpath(QStringLiteral("."))) {
        if (errorMessage) {
            *errorMessage = tr("Failed to create output directory: %1").arg(outputDirPath);
        }
        return QString();
    }
    return outDir.absolutePath();
}

QString SegmentationCommandHandler::defaultRefinedOutputPath(const QFileInfo& srcInfo)
{
    if (srcInfo.isDir()) {
        return srcInfo.absoluteFilePath() + "_refined";
    }
    const QString base = srcInfo.completeBaseName();
    const QString suffix = srcInfo.completeSuffix();
    QString candidate = srcInfo.absolutePath() + "/" + base + "_refined";
    if (!suffix.isEmpty()) {
        candidate += "." + suffix;
    }
    return candidate;
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
                                       /*interactive=*/true, &error, nullptr)) {
        QMessageBox::warning(_parentWidget, tr("Error"), error.message);
        return;
    }
    emit statusMessage(tr("Resume-opt local GrowPatch started for %1").arg(segmentId), 5000);
}

bool SegmentationCommandHandler::startResumeLocalGrowPatch(const std::string& segmentId,
                                                          const ResumeLocalGrowParams& params,
                                                          CommandLaunchError* error,
                                                          QString* resolvedOutputDir)
{
    return startResumeLocalGrowPatchImpl(segmentId, params, /*interactive=*/false,
                                         error, resolvedOutputDir);
}

bool SegmentationCommandHandler::startResumeLocalGrowPatchImpl(
    const std::string& segmentId,
    const ResumeLocalGrowParams& params,
    bool interactive,
    CommandLaunchError* error,
    QString* resolvedOutputDir)
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
    const QString toolPath =
        QCoreApplication::applicationDirPath() + QStringLiteral("/vc_grow_seg_from_seed");
    if (const QFileInfo toolInfo(toolPath); !toolInfo.exists() || !toolInfo.isExecutable()) {
        return fail(tr("vc_grow_seg_from_seed not found or not executable: %1").arg(toolPath),
                    CommandLaunchError::ToolUnavailable);
    }
    if (_neighborCopyJob && _neighborCopyJob->stage != NeighborCopyJob::Stage::None) {
        return fail(tr("A neighbor copy request is already active."), CommandLaunchError::Busy);
    }
    if (_resumeLocalJob) {
        return fail(tr("A resume-opt local GrowPatch run is already active."),
                    CommandLaunchError::Busy);
    }

    auto surf = _state->vpkg()->getSurface(segmentId);
    if (!surf) {
        return fail(tr("Invalid segment or segment not loaded: %1")
                        .arg(QString::fromStdString(segmentId)),
                    CommandLaunchError::SegmentNotFound);
    }

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

    QString outputError;
    QString outputDirPath = resolveSegmentOutputDir(surf->path, &outputError);
    if (outputDirPath.isEmpty()) {
        return fail(outputError);
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
    job.outputDir = outputDirPath;
    job.paramsPath = paramsFile->fileName();
    job.paramsFile = std::move(paramsFile);

    _cmdRunner->setNeighborCopyParams(selectedVolumePath,
                                      job.paramsPath,
                                      QString::fromStdString(surf->path.string()),
                                      outputDirPath,
                                      QStringLiteral("local"));
    configureCommandRunnerRemoteAuthForVolumePath(selectedVolumePath);
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

    if (resolvedOutputDir) {
        *resolvedOutputDir = outputDirPath;
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

    const QString executable = findVcTool("vc_grow_seg_from_seed");
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

    QString normalGridPath = _normalGridPathGetter ? _normalGridPathGetter() : QString();
    if (normalGridPath.isEmpty()) {
        const auto paths = _state->vpkg()->normalGridPaths();
        if (!paths.empty()) {
            normalGridPath = QString::fromStdString(paths.front().string());
        }
    }
    if (normalGridPath.isEmpty()) {
        return fail(tr("No normal grid is available for GrowPatch."),
                    CommandLaunchError::InvalidState);
    }

    QString volpkgRoot = _state->vpkgPath();
    if (volpkgRoot.isEmpty()) {
        volpkgRoot = QString::fromStdString(_state->vpkg()->getVolpkgDirectory());
    }
    if (volpkgRoot.isEmpty()) {
        return fail(tr("Cannot determine volume package folder."));
    }

    // Resolve the output directory. An empty request uses the same default the
    // interactive dialog offers: the head of the output-choice list.
    QString outputDirPath = params.outputDir.trimmed();
    if (outputDirPath.isEmpty()) {
        QStringList outputChoices;
        auto appendChoice = [&outputChoices](const QString& path) {
            const QString cleaned = QDir::cleanPath(path);
            if (!cleaned.isEmpty() && !outputChoices.contains(cleaned)) {
                outputChoices << cleaned;
            }
        };
        if (const auto patchesRoot =
                openDataPatchesRootForVolume(*_state->vpkg(), selectedVolumeId)) {
            appendChoice(*patchesRoot);
        }
        const auto outputSegmentsPath = _state->vpkg()->outputSegmentsPath();
        if (!outputSegmentsPath.empty()) {
            appendChoice(QString::fromStdString(outputSegmentsPath.string()));
        }
        for (const auto& path : _state->vpkg()->availableSegmentPaths()) {
            appendChoice(QString::fromStdString(path.string()));
        }
        appendChoice(QDir(volpkgRoot).filePath(QStringLiteral("paths")));
        appendChoice(QDir(volpkgRoot).filePath(QStringLiteral("traces")));
        outputDirPath = outputChoices.isEmpty()
            ? QDir(volpkgRoot).filePath(QStringLiteral("paths"))
            : outputChoices.front();
    }
    if (QDir::isRelativePath(outputDirPath)) {
        outputDirPath = QDir(volpkgRoot).filePath(outputDirPath);
    }
    QDir outDir(outputDirPath);
    if (!outDir.exists() && !outDir.mkpath(QStringLiteral("."))) {
        return fail(tr("Failed to create output folder: %1").arg(outputDirPath));
    }
    outputDirPath = outDir.absolutePath();

    const int iterations = std::clamp(params.iterations, 1, 100000);
    const double minAreaCm = std::max(0.0, params.minAreaCm);

    double selectedVoxelSize = 0.0;
    if (auto selectedVolume = _state->vpkg()->volume(selectedVolumeId.toStdString())) {
        selectedVoxelSize = selectedVolume->voxelSize();
    }

    const QString segmentsEntry = segmentsEntryLocationForPath(outputDirPath, volpkgRoot);
    _state->vpkg()->addSegmentsEntry(segmentsEntry.toStdString(), {"growpatch"});
    _state->vpkg()->setOutputSegments(segmentsEntry.toStdString());

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
    _growPatchSeedJob->outputDir = outputDirPath;
    _growPatchSeedJob->paramsPath = paramsFile->fileName();
    _growPatchSeedJob->paramsFile = std::move(paramsFile);

    QStringList args;
    args << selectedVolumePath
         << outputDirPath
         << _growPatchSeedJob->paramsPath
         << QString::number(seedPoint.x(), 'g', 10)
         << QString::number(seedPoint.y(), 'g', 10)
         << QString::number(seedPoint.z(), 'g', 10);

    configureCommandRunnerRemoteAuthForVolumePath(selectedVolumePath);

    const auto outputCoordinateIdentity =
        vc3d::opendata::coordinateIdentityForVolume(
            *_state->vpkg(), selectedVolumeId.toStdString());

    QPointer<SegmentationCommandHandler> guard(this);
    auto connection = std::make_shared<QMetaObject::Connection>();
    *connection = connect(_cmdRunner,
                          &CommandLineToolRunner::toolFinished,
                          this,
                          [this, guard, connection, outputDirPath,
                           outputCoordinateIdentity](CommandLineToolRunner::Tool tool,
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

        const bool silentExecution =
            _cmdRunner && _cmdRunner->currentExecutionIsSilent();

        if (!success) {
            if (!silentExecution) {
                QMessageBox::critical(_parentWidget,
                                      tr("Error"),
                                      tr("vc_grow_seg_from_seed failed.\n%1").arg(message));
            }
            emit statusMessage(tr("Create segment failed"), 3000);
            return;
        }

        if (outputCoordinateIdentity) {
            const QString metaPath = QDir(outputDirPath).filePath(QStringLiteral("meta.json"));
            QFile metaFile(metaPath);
            if (metaFile.open(QIODevice::ReadOnly)) {
                auto document = QJsonDocument::fromJson(metaFile.readAll());
                metaFile.close();
                if (document.isObject()) {
                    auto object = document.object();
                    copyCoordinateIdentity(
                        object, coordinateIdentityObject(outputCoordinateIdentity));
                    if (metaFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
                        metaFile.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
                        metaFile.close();
                    }
                }
            }
        }

        if (_state && _state->vpkg()) {
            _state->vpkg()->refreshSegmentations();
        }
        if (_surfacePanel) {
            _surfacePanel->reloadSurfacesFromDisk();
        }
        emit statusMessage(
            tr("GrowPatch segment created in %1").arg(QDir::toNativeSeparators(outputDirPath)),
            5000);
    });

    const auto options = interactive
        ? CommandLineToolRunner::ExecutionOptions{}
        : CommandLineToolRunner::ExecutionOptions::silent();
    if (!_cmdRunner->executeCustomCommand(
            executable, args, QStringLiteral("vc_grow_seg_from_seed"), options)) {
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

    if (findVcTool("vc_grow_seg_from_seed").isEmpty()) {
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

    QHash<QString, QString> openDataPatchesRootsByVolumeId;
    for (const auto& option : volumeOptions) {
        if (const auto patchesRoot = openDataPatchesRootForVolume(*_state->vpkg(), option.id)) {
            openDataPatchesRootsByVolumeId.insert(option.id, *patchesRoot);
        }
    }

    QString volpkgRoot = _state->vpkgPath();
    if (volpkgRoot.isEmpty()) {
        volpkgRoot = QString::fromStdString(_state->vpkg()->getVolpkgDirectory());
    }

    QStringList outputChoices;
    auto appendChoice = [&outputChoices](const QString& path) {
        const QString cleaned = QDir::cleanPath(path);
        if (!cleaned.isEmpty() && !outputChoices.contains(cleaned)) {
            outputChoices << cleaned;
        }
    };

    const auto outputSegmentsPath = _state->vpkg()->outputSegmentsPath();
    if (const auto it = openDataPatchesRootsByVolumeId.constFind(selectedVolumeId);
        it != openDataPatchesRootsByVolumeId.cend()) {
        appendChoice(it.value());
    }
    if (!outputSegmentsPath.empty()) {
        appendChoice(QString::fromStdString(outputSegmentsPath.string()));
    }
    for (const auto& path : _state->vpkg()->availableSegmentPaths()) {
        appendChoice(QString::fromStdString(path.string()));
    }
    appendChoice(QDir(volpkgRoot).filePath(QStringLiteral("paths")));
    appendChoice(QDir(volpkgRoot).filePath(QStringLiteral("traces")));

    int iterations = 200;
    double minAreaCm = 0.002;
    QString outputDirPath = outputChoices.isEmpty()
        ? QDir(volpkgRoot).filePath(QStringLiteral("paths"))
        : outputChoices.front();
    if (!selectGrowPatchSeedParams(_parentWidget,
                                   volpkgRoot,
                                   volumeOptions,
                                   outputChoices,
                                   openDataPatchesRootsByVolumeId,
                                   &selectedVolumeId,
                                   &selectedVolumePath,
                                   &iterations,
                                   &minAreaCm,
                                   &outputDirPath)) {
        emit statusMessage(tr("Create segment cancelled"), 3000);
        return;
    }
    settings.setValue(QStringLiteral("growpatch_seed/volume_id"), selectedVolumeId);

    GrowPatchSeedParams patchParams;
    patchParams.volumeId = selectedVolumeId;
    patchParams.iterations = iterations;
    patchParams.minAreaCm = minAreaCm;
    patchParams.outputDir = outputDirPath;

    CommandLaunchError error;
    if (!startGrowPatchFromSeedImpl(
            seedPoint, patchParams, /*interactive=*/true, &error)) {
        QMessageBox::warning(_parentWidget, tr("Error"), error.message);
    }
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
    _cmdRunner->setNextOmpThreads(dlg.ompThreads());
    _cmdRunner->setToObjOptions(dlg.normalizeUV(), dlg.alignGrid());
    _cmdRunner->execute(CommandLineToolRunner::Tool::tifxyz2obj);
    emit statusMessage(tr("Converting segment to OBJ: %1").arg(QString::fromStdString(segmentId)), 5000);
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
    if (!_state || _state->currentVolume() == nullptr || !_state->vpkg()) {
        return fail(tr("No volume package or volume loaded."));
    }
    auto surf = _state->vpkg()->getSurface(segmentId);
    if (!surf) {
        return fail(tr("Invalid segment or segment not loaded: %1")
                        .arg(QString::fromStdString(segmentId)));
    }
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

    std::unique_ptr<QuadSurface> tempSurface;
    try {
        tempSurface = std::make_unique<QuadSurface>(croppedPoints, surface->scale());
        tempSurface->path = surface->path;
        tempSurface->id = surface->id;
        if (!surface->meta.is_null()) {
            tempSurface->meta = surface->meta;
        }
        for (const auto& ch : croppedChannels) {
            tempSurface->setChannel(ch.name, ch.data);
        }
        tempSurface->save(surface->path.string(), surface->id, true);
    } catch (const std::exception& ex) {
        return fail(tr("Failed to crop %1: %2")
                        .arg(QString::fromStdString(segmentId))
                        .arg(QString::fromUtf8(ex.what())));
    }

    croppedPoints.copyTo(*points);
    for (const auto& ch : croppedChannels) {
        surface->setChannel(ch.name, ch.data);
    }
    if (droppedGenerationsChannel) {
        surface->setChannel("generations", cv::Mat());
        std::error_code ec;
        std::filesystem::remove(surface->path / "generations.tif", ec);
    }
    surface->invalidateCache();

    if (tempSurface && !tempSurface->meta.is_null()) {
        surface->meta = tempSurface->meta;
        surf->meta = surface->meta;
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
        (droppedGenerationsChannel
             ? tr("Cropped %1 to %2x%3 (offset %4,%5); removed mismatched generations.tif")
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
    if (_state && _state->currentVolume() && _state->currentVolume()->isRemote()) {
        QMessageBox::warning(
            _parentWidget, tr("Unsupported Remote Volume"),
            tr("Alpha-comp refinement accepts only local volumes. The remote "
               "locator was not modified or passed to the tool."));
        return;
    }

    QString volumePath = getCurrentVolumePath();
    if (volumePath.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"), tr("Cannot refine surface: Unable to determine volume path"));
        return;
    }

    QString srcPath = QString::fromStdString(surface->path.string());
    QFileInfo srcInfo(srcPath);

    const QString defaultOutput = defaultRefinedOutputPath(srcInfo);

    AlphaCompRefineDialog dlg(_parentWidget, volumePath, srcPath, defaultOutput);
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Alpha-comp refinement cancelled"), 3000);
        return;
    }

    CommandLaunchError error;
    if (!startAlphaCompRefineImpl(segmentId, dlg.request(), /*interactive=*/true,
                                  &error, nullptr)) {
        QMessageBox::warning(_parentWidget, tr("Error"), error.message);
        return;
    }
    emit statusMessage(tr("Refining segment: %1").arg(QString::fromStdString(segmentId)), 5000);
}

bool SegmentationCommandHandler::startAlphaCompRefine(const std::string& segmentId,
                                                     const AlphaCompRefineParams& params,
                                                     CommandLaunchError* error,
                                                     QString* resolvedOutputDir)
{
    return startAlphaCompRefineImpl(segmentId, params, /*interactive=*/false,
                                    error, resolvedOutputDir);
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
        {QStringLiteral("overwrite"), params.overwrite},
        {QStringLiteral("reader_scale"), params.readerScale},
        {QStringLiteral("scale_group"),
         params.scaleGroup.isEmpty() ? QStringLiteral("1") : params.scaleGroup},
    };
}

bool SegmentationCommandHandler::startAlphaCompRefineImpl(
    const std::string& segmentId,
    const AlphaCompRefineParams& params,
    bool interactive,
    CommandLaunchError* error,
    QString* resolvedOutputDir)
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
    auto surf = _state->vpkg()->getSurface(segmentId);
    if (!surf) {
        return fail(tr("Invalid segment or segment not loaded: %1")
                        .arg(QString::fromStdString(segmentId)),
                    CommandLaunchError::SegmentNotFound);
    }
    if (!_cmdRunner) {
        return fail(tr("Command line tools are not available."),
                    CommandLaunchError::ToolUnavailable);
    }
    if (_cmdRunner->isRunning()) {
        return fail(tr("A command line tool is already running."), CommandLaunchError::Busy);
    }
    if (_state->currentVolume()->isRemote()) {
        return fail(tr("Alpha-comp refinement accepts only local volumes."),
                    CommandLaunchError::RemoteVolume);
    }
    // Return a typed launch error before the runner reaches its interactive
    // missing-tool warning. Tool::AlphaCompRefine launches vc_objrefine.
    const QString toolPath =
        QCoreApplication::applicationDirPath() + QStringLiteral("/vc_objrefine");
    if (const QFileInfo toolInfo(toolPath); !toolInfo.exists() || !toolInfo.isExecutable()) {
        return fail(tr("vc_objrefine not found or not executable: %1").arg(toolPath),
                    CommandLaunchError::ToolUnavailable);
    }

    const QString volumePath = params.volumePath.isEmpty()
        ? getCurrentVolumePath()
        : params.volumePath;
    if (volumePath.isEmpty()) {
        return fail(tr("Unable to determine volume path."));
    }

    const QString srcPath = params.sourcePath.isEmpty()
        ? QString::fromStdString(surf->path.string())
        : params.sourcePath;
    const QFileInfo srcInfo(srcPath);

    QString dstPath = params.outputDir.trimmed();
    if (dstPath.isEmpty()) {
        dstPath = defaultRefinedOutputPath(srcInfo);
    } else if (!interactive && QDir::isRelativePath(dstPath)) {
        QString volpkgRoot = _state->vpkgPath();
        if (volpkgRoot.isEmpty()) {
            volpkgRoot = QString::fromStdString(_state->vpkg()->getVolpkgDirectory());
        }
        dstPath = QDir(volpkgRoot).filePath(dstPath);
    }

    if (volumePath.isEmpty() || srcPath.isEmpty() || dstPath.isEmpty()) {
        return fail(tr("Volume, source, and output paths must be specified."));
    }

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

    _cmdRunner->setObjRefineParams(volumePath, srcPath, dstPath, paramsPath);
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

    if (resolvedOutputDir) {
        *resolvedOutputDir = dstPath;
    }
    return true;
}

void SegmentationCommandHandler::handleNeighborCopyToolFinished(bool success)
{
    if (!_neighborCopyJob) {
        return;
    }

    auto& job = *_neighborCopyJob;
    if (!success) {
        _neighborCopyJob.reset();
        return;
    }

    if (job.stage == NeighborCopyJob::Stage::FirstPass) {
        const QString newSurface = findNewNeighborSurface(job);
        if (newSurface.isEmpty()) {
            QMessageBox::warning(_parentWidget, tr("Error"),
                                 tr("Could not locate the newly generated neighbor surface in %1.")
                                     .arg(job.outputDir));
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
    configureCommandRunnerRemoteAuthForVolumePath(job.volumePath);
    _cmdRunner->setNeighborCopyParams(
        job.volumePath,
        paramsPath,
        resumeSurface,
        job.outputDir,
        resumeOpt);
    _cmdRunner->setNextOmpThreads(ompThreads);
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
            QMessageBox::warning(_parentWidget, tr("Error"), tr("Failed to launch the second neighbor copy pass."));
            _neighborCopyJob.reset();
            return;
        }

        emit statusMessage(
            tr("Copy %1 pass 2 running").arg(copyOut ? tr("out") : tr("in")),
            3000);
    });
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

    QString referenceZarr = !_normal3dZarrPathGetter ? QString() : _normal3dZarrPathGetter();
    if (referenceZarr.isEmpty()) {
        referenceZarr = getCurrentVolumePath();
    }
    if (referenceZarr.isEmpty()) {
        QMessageBox::warning(_parentWidget, tr("Error"),
                             tr("Missing reference OME-Zarr. Load a normal3d/volume first."));
        return;
    }

    const QString executable = findVcTool("vc_tifxyz2zarr_sparse");
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

    const QString tempRootStr = QString::fromStdString(tempRoot.string());
    const QString stagedOutputRootStr = QString::fromStdString(stagedOutputRoot.string());
    const QString finalOutputRootStr = QString::fromStdString(finalOutputRoot.string());
    const QJsonObject outputCoordinateIdentity = coordinateIdentityObject(
        vc3d::opendata::coordinateIdentityForVolume(
            *_state->vpkg(), _state->currentVolumeId()));
    QStringList args;
    args << tempRootStr
         << stagedOutputRootStr
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
                          validIds, segmentPaths, outputCoordinateIdentity](CommandLineToolRunner::Tool tool,
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
                finalizeOutput = true;
                if (!updateVolumeIdentityMetadata(
                        finalOutputRootStr, outputCoordinateIdentity)) {
                    emit showWarning(
                        tr("Warning"),
                        tr("Rasterized volume created, but updating meta.json identity failed."));
                    emit statusMessage(
                        tr("Rasterized volume created, but metadata update failed -> %1")
                            .arg(QDir::toNativeSeparators(finalOutputRootStr)),
                        5000);
                } else {
                    emit statusMessage(
                        tr("Rasterized %1 segment(s) -> %2")
                            .arg(validIds.size())
                            .arg(QDir::toNativeSeparators(finalOutputRootStr)),
                        5000);
                }
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

void SegmentationCommandHandler::onMergeTifxyz(const QStringList& segmentIds)
{
    if (!_state || !_state->vpkg()) {
        QMessageBox::warning(_parentWidget, tr("Merge TIFXYZ"),
                             tr("Open a volpkg first."));
        return;
    }
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

    auto vpkg = _state->vpkg();
    // Use the volpkg's resolved output-segments path so it works for
    // both the `<volpkg-dir>/<segdir>` layout and the freshly-introduced
    // .volpkg.json form where the project file lives outside the data
    // directory and segment locations are relative.
    const std::filesystem::path resolvedSeg = vpkg->outputSegmentsPath();
    if (resolvedSeg.empty() || !std::filesystem::is_directory(resolvedSeg)) {
        QMessageBox::warning(_parentWidget, tr("Merge TIFXYZ"),
                             tr("No active segmentation directory; pick one "
                                "before running merge."));
        return;
    }
    const QString pathsDir = QString::fromStdString(resolvedSeg.string());
    // The merge.json + output dir live alongside the resolved segments
    // dir (the actual volpkg data root), not the .volpkg.json directory.
    const QString volpkgDir = QString::fromStdString(
        resolvedSeg.parent_path().string());
    const std::string segDirName = vpkg->getSegmentationDirectory();

    QStringList availableSegments;
    {
        const auto ids = vpkg->segmentationIDs();
        availableSegments.reserve(static_cast<int>(ids.size()));
        for (const auto& s : ids) availableSegments << QString::fromStdString(s);
    }
    if (availableSegments.size() < 2) {
        QMessageBox::warning(_parentWidget, tr("Merge TIFXYZ"),
                             tr("This volpkg has fewer than 2 segments in '%1'; "
                                "merge needs at least 2.")
                                 .arg(QString::fromStdString(segDirName)));
        return;
    }

    MergeTifxyzDialog dlg(_parentWidget, segmentIds, availableSegments,
                          volpkgDir, pathsDir);
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Merge cancelled"), 3000);
        return;
    }

    _cmdRunner->setMergeParams(dlg.mergeJsonPath(),
                               pathsDir,
                               dlg.refSurface(),
                               dlg.ransacIters(),
                               dlg.ransacMinThresh(),
                               dlg.ransacMaxThresh(),
                               dlg.ransacMadK(),
                               dlg.ransacSeed(),
                               dlg.anchorCap(),
                               dlg.stripCols());
    if (dlg.ompThreads() > 0) _cmdRunner->setNextOmpThreads(dlg.ompThreads());
    const QJsonObject outputCoordinateIdentity = coordinateIdentityObject(
        vc3d::opendata::coordinateIdentityForVolume(
            *vpkg, _state->currentVolumeId()));
    QPointer<SegmentationCommandHandler> guard(this);
    auto connection = std::make_shared<QMetaObject::Connection>();
    *connection = connect(
        _cmdRunner, &CommandLineToolRunner::toolFinished, this,
        [this, guard, connection, outputCoordinateIdentity](
            CommandLineToolRunner::Tool tool,
            bool success,
            const QString&,
            const QString& outputPath,
            bool) {
            if (!guard) {
                disconnect(*connection);
                return;
            }
            if (tool != CommandLineToolRunner::Tool::MergeTifxyz)
                return;
            disconnect(*connection);
            if (success && !updateVolumeIdentityMetadata(
                               outputPath, outputCoordinateIdentity)) {
                emit showWarning(
                    tr("Warning"),
                    tr("Merged tifxyz created, but coordinate metadata update failed."));
            }
        });
    _cmdRunner->showConsoleOutput();
    if (!_cmdRunner->execute(CommandLineToolRunner::Tool::MergeTifxyz)) {
        disconnect(*connection);
        return;
    }
    emit statusMessage(tr("Merging tifxyz surfaces..."), 0);
}

void SegmentationCommandHandler::onMergePatch(const QStringList& segmentIds)
{
    if (!_state || !_state->vpkg()) {
        QMessageBox::warning(_parentWidget, tr("Patch tifxyz"),
                             tr("Open a volpkg first."));
        return;
    }
    if (!_cmdRunner) {
        QMessageBox::warning(_parentWidget, tr("Patch tifxyz"),
                             tr("Command runner is not initialized."));
        return;
    }
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(_parentWidget, tr("Patch tifxyz"),
                             tr("A command-line tool is already running."));
        return;
    }

    auto vpkg = _state->vpkg();
    const std::filesystem::path resolvedSeg = vpkg->outputSegmentsPath();
    if (resolvedSeg.empty() || !std::filesystem::is_directory(resolvedSeg)) {
        QMessageBox::warning(_parentWidget, tr("Patch tifxyz"),
                             tr("No active segmentation directory; pick one "
                                "before running patch."));
        return;
    }
    const QString pathsDir  = QString::fromStdString(resolvedSeg.string());
    const QString volpkgDir = QString::fromStdString(
        resolvedSeg.parent_path().string());

    QStringList availableSegments;
    {
        const auto ids = vpkg->segmentationIDs();
        availableSegments.reserve(static_cast<int>(ids.size()));
        for (const auto& s : ids) availableSegments << QString::fromStdString(s);
    }
    if (availableSegments.size() < 2) {
        QMessageBox::warning(_parentWidget, tr("Patch tifxyz"),
                             tr("This volpkg has fewer than 2 segments; "
                                "patch needs a parent + child."));
        return;
    }

    MergePatchDialog dlg(_parentWidget, segmentIds, availableSegments,
                         vpkg, volpkgDir, pathsDir);
    if (dlg.exec() != QDialog::Accepted) {
        emit statusMessage(tr("Patch cancelled"), 3000);
        return;
    }

    _cmdRunner->setMergePatchParams(dlg.parentPath(),
                                    dlg.childPath(),
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

    // One-shot handler: on success, force-refresh the parent (and any
    // other modified) surface from disk. The mask.tif mtime drives the
    // per-surface reload inside detectSurfaceChanges, so the in-memory
    // QuadSurface picks up the patched x/y/z without per-surface
    // bookkeeping here.
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
            if (_surfacePanel) _surfacePanel->reloadSurfacesFromDisk();
            emit statusMessage(
                tr("Patch applied to %1")
                    .arg(QDir::toNativeSeparators(outputPath)), 5000);
        } else {
            QMessageBox::critical(_parentWidget, tr("Patch tifxyz"),
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
    emit statusMessage(tr("Patching tifxyz..."), 0);
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

    const QString executable = findVcTool("vc_add_ignore_label");
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

    const bool reuseExistingOutput = outputExists;
    const bool useStaging = !reuseExistingOutput;

    std::filesystem::path stagedOutputRoot;
    if (useStaging) {
        const QString timestamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMddHHmmss"));
        stagedOutputRoot = volumesDir / (".vc3d_ignore_" + timestamp.toStdString());
        for (int idx = 1; std::filesystem::exists(stagedOutputRoot, ec) && idx < 1000; ++idx) {
            const QString candidate = QStringLiteral(".vc3d_ignore_%1_%2").arg(timestamp).arg(idx);
            stagedOutputRoot = volumesDir / candidate.toStdString();
        }
        if (std::filesystem::exists(stagedOutputRoot, ec)) {
            QMessageBox::critical(_parentWidget,
                                  tr("Error"),
                                  tr("Unable to reserve staging directory after retries: %1")
                                      .arg(QString::fromStdString(stagedOutputRoot.string())));
            return;
        }
    }

    const QString stagedOutputRootStr = useStaging
        ? QString::fromStdString(stagedOutputRoot.string())
        : QString();
    const QString finalOutputRootStr = QString::fromStdString(finalOutputRoot.string());
    const QString runOutputRootStr = useStaging ? stagedOutputRootStr : finalOutputRootStr;
    const QJsonObject outputCoordinateIdentity = coordinateIdentityFromJson(
        readJsonObject(QDir(params.volumePath).filePath(QStringLiteral("meta.json"))));

    QStringList args;
    args << params.volumePath
         << runOutputRootStr
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
                           useStaging, stagedOutputRootStr, finalOutputRootStr,
                           outputCoordinateIdentity]
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

        bool finalized = false;
        if (!success) {
            if (progressDialog) {
                progressDialog->close();
            }
            QMessageBox::critical(_parentWidget,
                                  tr("Error"),
                                  tr("vc_add_ignore_label failed.\n%1").arg(message));
            emit statusMessage(tr("Add ignore label failed"), 3000);
        } else if (useStaging) {
            if (progressDialog) {
                progressDialog->setRange(0, 0);
                progressDialog->setLabelText(tr("Finalizing output..."));
            }
            std::error_code renameErr;
            std::filesystem::rename(stagedOutputRootStr.toStdString(),
                                    finalOutputRootStr.toStdString(),
                                    renameErr);
            if (renameErr) {
                emit showWarning(tr("Warning"),
                                 tr("Ignore label generation completed, but finalizing output folder failed: %1")
                                     .arg(QString::fromStdString(renameErr.message())));
                emit statusMessage(tr("Ignore label complete, but finalizing output failed"), 5000);
            } else {
                finalized = true;
                if (!updateVolumeIdentityMetadata(
                        finalOutputRootStr, outputCoordinateIdentity)) {
                    emit showWarning(
                        tr("Warning"),
                        tr("Ignore label volume created, but updating meta.json identity failed."));
                    emit statusMessage(
                        tr("Ignore label volume created, but metadata update failed -> %1")
                            .arg(QDir::toNativeSeparators(finalOutputRootStr)),
                        5000);
                } else {
                    emit statusMessage(
                        tr("Ignore label volume created -> %1")
                            .arg(QDir::toNativeSeparators(finalOutputRootStr)),
                        5000);
                }
            }
        } else {
            if (progressDialog) {
                progressDialog->setRange(0, 0);
                progressDialog->setLabelText(tr("Finishing..."));
            }
            finalized = true;
            emit statusMessage(
                tr("Ignore label volume updated -> %1")
                    .arg(QDir::toNativeSeparators(finalOutputRootStr)),
                5000);
        }

        if (!finalized && useStaging) {
            std::error_code cleanupErr;
            std::filesystem::remove_all(std::filesystem::path(stagedOutputRootStr.toStdString()), cleanupErr);
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
        if (useStaging) {
            std::error_code cleanupErr;
            std::filesystem::remove_all(stagedOutputRoot, cleanupErr);
        }
        return;
    }

    emit statusMessage(tr("Add ignore label started at level 0..."), 0);
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
        chunkSurf.meta = surf->meta;
        if (_state && _state->vpkg()) {
            vc3d::opendata::copyVolumeCoordinateIdentityToSurface(
                chunkSurf, *_state->vpkg(), _state->currentVolumeId());
        }

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
        metaJson["max"] = 1.0;
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

    if (_state && _state->vpkg()) {
        copyCoordinateIdentity(
            metaJson,
            coordinateIdentityObject(vc3d::opendata::coordinateIdentityForVolume(
                *_state->vpkg(), _state->currentVolumeId())));
    }

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

    // Backups live under <backupRoot>/backups/<id>/<index> (backupRoot is the
    // volpkg.json's directory), matching QuadSurface::saveSnapshot().
    namespace fs = std::filesystem;
    fs::path segmentDir = surf->path;
    fs::path backupRoot = surf->backupRoot.empty() ? segmentDir.parent_path() : surf->backupRoot;
    fs::path backupDir = backupRoot / "backups" / segmentDir.filename() / std::to_string(backupIndex);

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

    // Dialog-free core carries the validation, rename, and rollback. Map its
    // distinct failure sentences back to the interactive message boxes.
    QString err;
    if (renameSurfaceHeadless(segmentId, newName, &err)) {
        emit statusMessage(tr("Renamed '%1' to '%2'").arg(segmentId, newName), 5000);
        return;
    }
    if (err == QLatin1String("name unchanged"))
        return;
    if (err == QLatin1String("invalid name")) {
        QMessageBox::warning(_parentWidget, tr("Invalid Name"),
            tr("Surface name can only contain letters, numbers, underscores, and hyphens."));
    } else if (err == QLatin1String("name exists")) {
        QMessageBox::warning(_parentWidget, tr("Name Exists"),
            tr("A surface with the name '%1' already exists.").arg(newName));
    } else if (err == QLatin1String("segment not found")) {
        emit statusMessage(tr("Segment not found: %1").arg(segmentId), 3000);
    } else if (!err.isEmpty()) {
        QMessageBox::critical(_parentWidget, tr("Error"), err);
    }
}

bool SegmentationCommandHandler::renameSurfaceHeadless(const QString& segmentIdQ,
                                                       const QString& newName,
                                                       QString* err)
{
    // The short sentinels below are stable machine-readable tokens used to
    // classify failures, so they must stay untranslated.
    auto fail = [&](const QString& msg) {
        if (err) *err = msg;
        return false;
    };

    if (!_state || !_state->vpkg())
        return fail(QStringLiteral("no volume package"));

    // Block if surface is currently being edited.
    if (_isEditingCheck && _isEditingCheck())
        return fail(QStringLiteral("editing in progress"));

    const std::string oldId = segmentIdQ.toStdString();
    auto seg = _state->vpkg()->segmentation(oldId);
    if (!seg)
        return fail(QStringLiteral("segment not found"));

    const std::string newId = newName.toStdString();

    // Validate new name: alphanumeric + underscore + hyphen only.
    static const QRegularExpression validNameRegex(QStringLiteral("^[a-zA-Z0-9_-]+$"));
    if (!validNameRegex.match(newName).hasMatch())
        return fail(QStringLiteral("invalid name"));

    // Check if name is unchanged.
    if (newId == oldId)
        return fail(QStringLiteral("name unchanged"));

    // Check for name collision.
    std::filesystem::path currentPath = seg->path();
    std::filesystem::path parentDir = currentPath.parent_path();
    std::filesystem::path newPath = parentDir / newId;

    if (std::filesystem::exists(newPath))
        return fail(QStringLiteral("name exists"));

    // Check if this is the currently selected segment.
    bool wasSelected = (_state->activeSurfaceId() == oldId);

    // Store the old UUID for rollback if needed.
    std::string oldUuid = seg->id();

    // === Clean up the segment before renaming ===

    // Clear from surface collection (including "segmentation" if it matches).
    if (_state) {
        auto currentSurface = _state->surface(oldId);
        auto segmentationSurface = _state->surface("segmentation");

        // If this surface is currently shown as "segmentation", clear it.
        if (currentSurface && segmentationSurface && currentSurface == segmentationSurface) {
            _state->setSurface("segmentation", nullptr, false, false);
        }

        // Clear the surface from the collection.
        _state->setSurface(oldId, nullptr, false, false);
    }

    // Unload the surface from VolumePkg.
    _state->vpkg()->unloadSurface(oldId);

    // Clear selection if this was selected.
    if (wasSelected) {
        if (_clearSelectionCallback) {
            _clearSelectionCallback();
        }
    }

    // Update meta.json UUID.
    try {
        seg->setId(newId);
        seg->saveMetadata();
    } catch (const std::exception& e) {
        // Reload the old segment.
        _state->vpkg()->refreshSegmentations();
        if (_surfacePanel) {
            _surfacePanel->reloadSurfacesFromDisk();
        }
        return fail(tr("Failed to update metadata: %1").arg(e.what()));
    }

    // Perform the folder rename.
    try {
        std::filesystem::rename(currentPath, newPath);

        // Remove old ID from VolumePkg's internal tracking.
        _state->vpkg()->removeSingleSegmentation(oldId);

        // Remove from surface panel.
        if (_surfacePanel) {
            _surfacePanel->removeSingleSegmentation(oldId);
        }

        // Refresh segmentations to pick up the new ID.
        _state->vpkg()->refreshSegmentations();

        // Add the new segment.
        if (_surfacePanel) {
            _surfacePanel->addSingleSegmentation(newId);
        }

        // Restore selection if it was the selected surface.
        if (wasSelected && _restoreSelectionCallback) {
            _restoreSelectionCallback(newId);
        }

        return true;

    } catch (const std::exception& e) {
        // Attempt to rollback metadata change.
        try {
            seg->setId(oldUuid);
            seg->saveMetadata();
        } catch (...) {
            // Rollback failed - metadata is now inconsistent.
        }

        // Refresh to get back to a consistent state.
        _state->vpkg()->refreshSegmentations();
        if (_surfacePanel) {
            _surfacePanel->reloadSurfacesFromDisk();
        }
        return fail(tr("Failed to rename folder: %1\n\n"
                       "The segment has been unloaded. Please reload surfaces.").arg(e.what()));
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
