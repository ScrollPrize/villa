#include "SettingsDialog.hpp"

#include "VCSettings.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>
#include <QCheckBox>
#include <QComboBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QToolTip>



SettingsDialog::SettingsDialog(std::shared_ptr<VolumePkg> volumePackage,
                               QWidget *parent)
    : QDialog(parent)
    , _volumePackage(std::move(volumePackage))
{
    setupUi(this);

    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    edtDefaultPathVolpkg->setText(settings.value(project::DEFAULT_PATH).toString());
    chkAutoOpenVolpkg->setChecked(settings.value(project::AUTO_OPEN, project::AUTO_OPEN_DEFAULT).toInt() != 0);
    setupOutputSegmentsControl();

    spinFwdBackStepMs->setValue(settings.value(viewer::FWD_BACK_STEP_MS, viewer::FWD_BACK_STEP_MS_DEFAULT).toInt());
    chkCenterOnZoom->setChecked(settings.value(viewer::CENTER_ON_ZOOM, viewer::CENTER_ON_ZOOM_DEFAULT).toInt() != 0);
    edtImpactRange->setText(settings.value(viewer::IMPACT_RANGE_STEPS, viewer::IMPACT_RANGE_STEPS_DEFAULT).toString());
    edtScanRange->setText(settings.value(viewer::SCAN_RANGE_STEPS, viewer::SCAN_RANGE_STEPS_DEFAULT).toString());
    spinScrollSpeed->setValue(settings.value(viewer::SCROLL_SPEED, viewer::SCROLL_SPEED_DEFAULT).toInt());
    spinZoomSensitivity->setValue(settings.value(viewer::ZOOM_SENSITIVITY, viewer::ZOOM_SENSITIVITY_DEFAULT).toDouble());
    spinDisplayOpacity->setValue(settings.value(viewer::DISPLAY_SEGMENT_OPACITY, viewer::DISPLAY_SEGMENT_OPACITY_DEFAULT).toInt());
    chkPlaySoundAfterSegRun->setChecked(settings.value(viewer::PLAY_SOUND_AFTER_SEG_RUN, viewer::PLAY_SOUND_AFTER_SEG_RUN_DEFAULT).toInt() != 0);
    edtUsername->setText(settings.value(viewer::USERNAME, viewer::USERNAME_DEFAULT).toString());
    chkResetViewOnSurfaceChange->setChecked(settings.value(viewer::RESET_VIEW_ON_SURFACE_CHANGE, viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toInt() != 0);
    if (auto* chk = findChild<QCheckBox*>("chkShowPlaneIntersectionLines")) {
        chk->setChecked(settings.value(viewer::SHOW_PLANE_INTERSECTION_LINES, viewer::SHOW_PLANE_INTERSECTION_LINES_DEFAULT).toInt() != 0);
    }
    if (auto* cmb = findChild<QComboBox*>("cmbInterpolation")) {
        cmb->setCurrentIndex(std::clamp(settings.value(perf::INTERPOLATION_METHOD, perf::INTERPOLATION_METHOD_DEFAULT).toInt(), 0, 1));
    }
    if (auto* spin = findChild<QSpinBox*>("spinIntersectionOpacity")) {
        spin->setValue(std::clamp(settings.value(viewer::INTERSECTION_OPACITY, viewer::INTERSECTION_OPACITY_DEFAULT).toInt(), 0, 100));
    }
    if (auto* spin = findChild<QSpinBox*>("spinAxisOverlayOpacity")) {
        spin->setValue(std::clamp(settings.value(viewer::AXIS_OVERLAY_OPACITY, viewer::AXIS_OVERLAY_OPACITY_DEFAULT).toInt(), 0, 100));
    }
    // Show direction hints (flip_x arrows)
    if (findChild<QCheckBox*>("chkShowDirectionHints")) {
        findChild<QCheckBox*>("chkShowDirectionHints")->setChecked(settings.value(viewer::SHOW_DIRECTION_HINTS, viewer::SHOW_DIRECTION_HINTS_DEFAULT).toInt() != 0);
    }
    // Direction step size default
    if (auto* spin = findChild<QDoubleSpinBox*>("spinDirectionStep")) {
        spin->setValue(settings.value(viewer::DIRECTION_STEP, viewer::DIRECTION_STEP_DEFAULT).toDouble());
    }
    // Use segmentation step for hints
    if (auto* chk = findChild<QCheckBox*>("chkUseSegStepForHints")) {
        chk->setChecked(settings.value(viewer::USE_SEG_STEP_FOR_HINTS, viewer::USE_SEG_STEP_FOR_HINTS_DEFAULT).toInt() != 0);
    }
    // Number of step points per direction
    if (auto* spin = findChild<QSpinBox*>("spinDirectionStepPoints")) {
        spin->setValue(settings.value(viewer::DIRECTION_STEP_POINTS, viewer::DIRECTION_STEP_POINTS_DEFAULT).toInt());
    }

    spinPreloadedSlices->setValue(settings.value(perf::PRELOADED_SLICES, perf::PRELOADED_SLICES_DEFAULT).toInt());
    spinParallelProcesses->setValue(settings.value(perf::PARALLEL_PROCESSES, perf::PARALLEL_PROCESSES_DEFAULT).toInt());
    spinIterationCount->setValue(settings.value(perf::ITERATION_COUNT, perf::ITERATION_COUNT_DEFAULT).toInt());
    cmbDownscaleOverride->setCurrentIndex(settings.value(perf::DOWNSCALE_OVERRIDE, perf::DOWNSCALE_OVERRIDE_DEFAULT).toInt());
    chkEnableFileWatching->setChecked(settings.value(perf::ENABLE_FILE_WATCHING, perf::ENABLE_FILE_WATCHING_DEFAULT).toBool());

    // Cache settings
    spinRamCacheSizeGB->setValue(settings.value(perf::RAM_CACHE_SIZE_GB, perf::RAM_CACHE_SIZE_GB_DEFAULT).toInt());
    spinViewerSurfaceCacheGB->setValue(
        settings.value(viewer_cache::SURFACE_CACHE_GB,
                       viewer_cache::SURFACE_CACHE_GB_DEFAULT).toInt());
    spinViewerOverlaySurfaceCacheGB->setValue(
        settings.value(viewer_cache::OVERLAY_SURFACE_CACHE_GB,
                       viewer_cache::OVERLAY_SURFACE_CACHE_GB_DEFAULT).toInt());
    {
        const QString stored =
            settings.value(viewer::REMOTE_CACHE_DIR).toString();
        const QString active = vc3d::remoteCachePath(stored);
        edtRemoteCachePath->setText(active);
        _activeRemoteCacheRoot = active.toStdString();
    }
    spinRemoteCacheMaximumGiB->setValue(static_cast<int>(settings.value(
        perf::REMOTE_CACHE_MAX_GIB, perf::REMOTE_CACHE_MAX_GIB_DEFAULT).toULongLong()));
    spinRemoteCacheMinimumFreeGiB->setValue(static_cast<int>(settings.value(
        perf::REMOTE_CACHE_MIN_FREE_GIB, perf::REMOTE_CACHE_MIN_FREE_GIB_DEFAULT).toULongLong()));

    // Per-segment rotating-backup count.
    if (spinSegmentBackupCount) {
        spinSegmentBackupCount->setValue(
            settings.value(backup::SEGMENT_COUNT, backup::SEGMENT_COUNT_DEFAULT).toInt());
    }

    const bool automaticDownloads = settings.value(
        perf::REMOTE_DOWNLOAD_AUTOMATIC,
        perf::REMOTE_DOWNLOAD_AUTOMATIC_DEFAULT).toBool();
    chkAutoDownloadParallelism->setChecked(automaticDownloads);
    spinIOThreads->setRange(1, perf::REMOTE_DOWNLOAD_WORKER_CAPACITY);
    spinIOThreads->setValue(std::clamp(
        settings.value(
            perf::REMOTE_DOWNLOAD_PARALLELISM,
            perf::REMOTE_DOWNLOAD_PARALLELISM_DEFAULT).toInt(),
        1, perf::REMOTE_DOWNLOAD_WORKER_CAPACITY));
    spinIOThreads->setEnabled(!automaticDownloads);
    connect(chkAutoDownloadParallelism, &QCheckBox::toggled,
            spinIOThreads, [this](bool automatic) {
                spinIOThreads->setEnabled(!automatic);
            });

    setupCacheActionControls();
    _remoteCacheDelta3dCheckBox->setChecked(settings.value(
        perf::REMOTE_CACHE_DELTA3D,
        perf::REMOTE_CACHE_DELTA3D_DEFAULT).toBool());

    connect(btnBrowseRemoteCachePath, &QPushButton::clicked, this, [this]{
        QString dir = QFileDialog::getExistingDirectory(this, tr("Select Remote Cache Directory"),
            edtRemoteCachePath->text());
        if (!dir.isEmpty()) {
            edtRemoteCachePath->setText(dir);
        }
    });

    connect(btnHelpDownscaleOverride, &QPushButton::clicked, this, [this]{ QToolTip::showText(QCursor::pos(), btnHelpDownscaleOverride->toolTip()); });
    connect(btnHelpScrollSpeed, &QPushButton::clicked, this, [this]{ QToolTip::showText(QCursor::pos(), btnHelpScrollSpeed->toolTip()); });
    connect(btnHelpDisplayOpacity, &QPushButton::clicked, this, [this]{ QToolTip::showText(QCursor::pos(), btnHelpDisplayOpacity->toolTip()); });
    connect(btnHelpPreloadedSlices, &QPushButton::clicked, this, [this]{ QToolTip::showText(QCursor::pos(), btnHelpPreloadedSlices->toolTip()); });
    connect(btnHelpRamCacheSize, &QPushButton::clicked, this, [this]{ QToolTip::showText(QCursor::pos(), btnHelpRamCacheSize->toolTip()); });
}

void SettingsDialog::setupCacheActionControls()
{
    _remoteCacheDelta3dCheckBox = new QCheckBox(
        tr("Compress remote volume disk cache with VC-Delta3D (lossless)."),
        groupBox_5);
    _remoteCacheDelta3dCheckBox->setObjectName(
        QStringLiteral("chkRemoteCacheDelta3d"));
    _remoteCacheDelta3dCheckBox->setToolTip(tr(
        "Requires restart. On the next use of each remote volume, VC3D replaces "
        "incompatible disposable cache contents with the selected lossless format."));

    if (gridLayout_5)
        gridLayout_5->addWidget(_remoteCacheDelta3dCheckBox, 2, 0, 1, 3);
}

void SettingsDialog::setupOutputSegmentsControl()
{
    auto* layout = qobject_cast<QGridLayout*>(gridLayout);
    if (!layout) {
        return;
    }

    auto* label = new QLabel(tr("Output segments"), groupBox_2);
    _outputSegmentsCombo = new QComboBox(groupBox_2);
    _outputSegmentsCombo->setObjectName(QStringLiteral("cmbOutputSegments"));

    layout->addWidget(label, 2, 0);
    layout->addWidget(_outputSegmentsCombo, 2, 1);

    if (!_volumePackage) {
        _outputSegmentsCombo->addItem(tr("Open or create a project first"), QString());
        _outputSegmentsCombo->setEnabled(false);
        return;
    }

    const auto& entries = _volumePackage->segmentEntries();
    if (entries.empty()) {
        _outputSegmentsCombo->addItem(tr("Attach a segments source first"), QString());
        _outputSegmentsCombo->setEnabled(false);
        return;
    }

    int currentIdx = 0;
    const QString current = _volumePackage->hasOutputSegments()
        ? QString::fromStdString(_volumePackage->outputSegmentsPath().string())
        : QString();
    for (const auto& entry : entries) {
        const QString location = QString::fromStdString(entry.location);
        _outputSegmentsCombo->addItem(location, location);
        if (!current.isEmpty() && location == current) {
            currentIdx = _outputSegmentsCombo->count() - 1;
        }
    }
    _outputSegmentsCombo->setCurrentIndex(currentIdx);
}

void SettingsDialog::accept()
{
    // Store the settings
    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    settings.setValue(project::DEFAULT_PATH, edtDefaultPathVolpkg->text());
    settings.setValue(project::AUTO_OPEN, chkAutoOpenVolpkg->isChecked() ? "1" : "0");
    if (_volumePackage && _outputSegmentsCombo && _outputSegmentsCombo->isEnabled()) {
        const QString chosen = _outputSegmentsCombo->currentData().toString();
        if (!chosen.isEmpty()) {
            const QString current = _volumePackage->hasOutputSegments()
                ? QString::fromStdString(_volumePackage->outputSegmentsPath().string())
                : QString();
            if (chosen != current) {
                _volumePackage->setOutputSegments(chosen.toStdString());
                _outputSegmentsChanged = true;
            }
        }
    }

    settings.setValue(viewer::FWD_BACK_STEP_MS, spinFwdBackStepMs->value());
    settings.setValue(viewer::CENTER_ON_ZOOM, chkCenterOnZoom->isChecked() ? "1" : "0");
    settings.setValue(viewer::IMPACT_RANGE_STEPS, edtImpactRange->text());
    settings.setValue(viewer::SCAN_RANGE_STEPS, edtScanRange->text());
    settings.setValue(viewer::SCROLL_SPEED, spinScrollSpeed->value());
    settings.setValue(viewer::ZOOM_SENSITIVITY, spinZoomSensitivity->value());
    settings.setValue(viewer::DISPLAY_SEGMENT_OPACITY, spinDisplayOpacity->value());
    settings.setValue(viewer::PLAY_SOUND_AFTER_SEG_RUN, chkPlaySoundAfterSegRun->isChecked() ? "1" : "0");
    settings.setValue(viewer::USERNAME, edtUsername->text());
    settings.setValue(viewer::RESET_VIEW_ON_SURFACE_CHANGE, chkResetViewOnSurfaceChange->isChecked() ? "1" : "0");
    if (auto* chk = findChild<QCheckBox*>("chkShowPlaneIntersectionLines")) {
        settings.setValue(viewer::SHOW_PLANE_INTERSECTION_LINES, chk->isChecked() ? "1" : "0");
    }
    if (auto* cmb = findChild<QComboBox*>("cmbInterpolation")) {
        settings.setValue(perf::INTERPOLATION_METHOD, cmb->currentIndex());
    }
    if (auto* spin = findChild<QSpinBox*>("spinIntersectionOpacity")) {
        settings.setValue(viewer::INTERSECTION_OPACITY, spin->value());
    }
    if (auto* spin = findChild<QSpinBox*>("spinAxisOverlayOpacity")) {
        settings.setValue(viewer::AXIS_OVERLAY_OPACITY, spin->value());
    }
    if (findChild<QCheckBox*>("chkShowDirectionHints")) {
        settings.setValue(viewer::SHOW_DIRECTION_HINTS, findChild<QCheckBox*>("chkShowDirectionHints")->isChecked() ? "1" : "0");
    }
    if (auto* spin = findChild<QDoubleSpinBox*>("spinDirectionStep")) {
        settings.setValue(viewer::DIRECTION_STEP, spin->value());
    }
    if (auto* chk = findChild<QCheckBox*>("chkUseSegStepForHints")) {
        settings.setValue(viewer::USE_SEG_STEP_FOR_HINTS, chk->isChecked() ? "1" : "0");
    }
    if (auto* spin = findChild<QSpinBox*>("spinDirectionStepPoints")) {
        settings.setValue(viewer::DIRECTION_STEP_POINTS, spin->value());
    }

    settings.setValue(perf::PRELOADED_SLICES, spinPreloadedSlices->value());
    settings.setValue(perf::PARALLEL_PROCESSES, spinParallelProcesses->value());
    settings.setValue(perf::ITERATION_COUNT, spinIterationCount->value());
    settings.setValue(perf::DOWNSCALE_OVERRIDE, cmbDownscaleOverride->currentIndex());
    settings.setValue(perf::ENABLE_FILE_WATCHING, chkEnableFileWatching->isChecked() ? "1" : "0");

    // Cache settings
    settings.setValue(perf::RAM_CACHE_SIZE_GB, spinRamCacheSizeGB->value());
    settings.setValue(viewer_cache::SURFACE_CACHE_GB, spinViewerSurfaceCacheGB->value());
    settings.setValue(viewer_cache::OVERLAY_SURFACE_CACHE_GB,
                      spinViewerOverlaySurfaceCacheGB->value());
    settings.setValue(viewer::REMOTE_CACHE_DIR, edtRemoteCachePath->text());
    settings.setValue(
        perf::REMOTE_CACHE_DELTA3D,
        _remoteCacheDelta3dCheckBox->isChecked());
    const bool automaticDownloads = chkAutoDownloadParallelism->isChecked();
    const int downloadParallelism = spinIOThreads->value();
    settings.setValue(
        perf::REMOTE_DOWNLOAD_AUTOMATIC, automaticDownloads);
    settings.setValue(
        perf::REMOTE_DOWNLOAD_PARALLELISM, downloadParallelism);
    vc::render::processChunkCacheService()->configureFetchConcurrency(
        automaticDownloads
            ? static_cast<std::size_t>(perf::REMOTE_DOWNLOAD_WORKER_CAPACITY)
            : static_cast<std::size_t>(downloadParallelism),
        automaticDownloads);
    settings.setValue(perf::REMOTE_CACHE_MAX_GIB, spinRemoteCacheMaximumGiB->value());
    settings.setValue(perf::REMOTE_CACHE_MIN_FREE_GIB, spinRemoteCacheMinimumFreeGiB->value());
    constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
    vc::render::PersistentZarrCacheBudget::Limits limits;
    if (spinRemoteCacheMaximumGiB->value() > 0)
        limits.maximumBytes = static_cast<std::uint64_t>(spinRemoteCacheMaximumGiB->value()) * gib;
    limits.minimumFreeBytes =
        static_cast<std::uint64_t>(spinRemoteCacheMinimumFreeGiB->value()) * gib;
    vc::render::PersistentZarrCacheBudget::configure(_activeRemoteCacheRoot, limits);
    vc::render::PersistentZarrCacheBudget::updateAllConfiguredLimits(limits);

    // Per-segment backup count: persist and apply live (no restart needed).
    if (spinSegmentBackupCount) {
        const int backupCount = spinSegmentBackupCount->value();
        settings.setValue(backup::SEGMENT_COUNT, backupCount);
        QuadSurface::setBackupCount(backupCount);
    }

    QMessageBox::information(this, tr("Restart required"), tr("Note: Some settings only take effect once you restarted the app."));

    QDialog::accept();
}

// Expand string that contains a range definition from the user settings into an integer vector
std::vector<int> SettingsDialog::expandSettingToIntRange(const QString& setting)
{
    std::vector<int> res;
    if (setting.isEmpty()) {
        return res;
    }

    auto value = setting.simplified();
    value.replace(" ", "");
    auto commaSplit = value.split(",");
    for(auto str : commaSplit) {
        if (str.contains("-")) {
            // Expand the range to distinct values
            auto dashSplit = str.split("-");
            // We need to have two split results (before and after the dash), otherwise skip
            if (dashSplit.size() == 2) {
                for(int i = dashSplit.at(0).toInt(); i <= dashSplit.at(1).toInt(); i++) {
                    res.push_back(i);
                }
            }
        } else {
            res.push_back(str.toInt());
        }
    }

    return res;
}
