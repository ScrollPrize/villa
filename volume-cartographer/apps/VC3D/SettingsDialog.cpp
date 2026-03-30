#include "SettingsDialog.hpp"

#include "VCSettings.hpp"

#include <algorithm>
#include <QDir>
#include <QFileDialog>
#include <QSettings>
#include <QMessageBox>
#include <QToolTip>



SettingsDialog::SettingsDialog(QWidget *parent) : QDialog(parent)
{
    setupUi(this);

    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    edtDefaultPathVolpkg->setText(settings.value(volpkg::DEFAULT_PATH).toString());
    chkAutoOpenVolpkg->setChecked(settings.value(volpkg::AUTO_OPEN, volpkg::AUTO_OPEN_DEFAULT).toInt() != 0);

    spinFwdBackStepMs->setValue(settings.value(viewer::FWD_BACK_STEP_MS, viewer::FWD_BACK_STEP_MS_DEFAULT).toInt());
    chkCenterOnZoom->setChecked(settings.value(viewer::CENTER_ON_ZOOM, viewer::CENTER_ON_ZOOM_DEFAULT).toInt() != 0);
    edtImpactRange->setText(settings.value(viewer::IMPACT_RANGE_STEPS, viewer::IMPACT_RANGE_STEPS_DEFAULT).toString());
    edtScanRange->setText(settings.value(viewer::SCAN_RANGE_STEPS, viewer::SCAN_RANGE_STEPS_DEFAULT).toString());
    spinScrollSpeed->setValue(settings.value(viewer::SCROLL_SPEED, viewer::SCROLL_SPEED_DEFAULT).toInt());
    spinDisplayOpacity->setValue(settings.value(viewer::DISPLAY_SEGMENT_OPACITY, viewer::DISPLAY_SEGMENT_OPACITY_DEFAULT).toInt());
    chkPlaySoundAfterSegRun->setChecked(settings.value(viewer::PLAY_SOUND_AFTER_SEG_RUN, viewer::PLAY_SOUND_AFTER_SEG_RUN_DEFAULT).toInt() != 0);
    edtUsername->setText(settings.value(viewer::USERNAME, viewer::USERNAME_DEFAULT).toString());
    chkResetViewOnSurfaceChange->setChecked(settings.value(viewer::RESET_VIEW_ON_SURFACE_CHANGE, viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toInt() != 0);
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
    chkFastInterpolation->setChecked(settings.value(perf::FAST_INTERPOLATION, perf::FAST_INTERPOLATION_DEFAULT).toBool());
    chkEnableFileWatching->setChecked(settings.value(perf::ENABLE_FILE_WATCHING, perf::ENABLE_FILE_WATCHING_DEFAULT).toBool());

    // Cache settings
    spinRamCacheSizeGB->setValue(settings.value(perf::RAM_CACHE_SIZE_GB, perf::RAM_CACHE_SIZE_GB_DEFAULT).toInt());
    spinDiskCacheSizeGB->setValue(settings.value(perf::DISK_CACHE_SIZE_GB, perf::DISK_CACHE_SIZE_GB_DEFAULT).toInt());
    {
        QString defaultCache = QDir::homePath() + "/.VC3D/remote_cache";
        edtRemoteCachePath->setText(settings.value(viewer::REMOTE_CACHE_DIR, defaultCache).toString());
    }

    // Video codec recompression settings
    chkVideoRecompress->setChecked(settings.value(perf::VIDEO_RECOMPRESS_ENABLED, perf::VIDEO_RECOMPRESS_ENABLED_DEFAULT).toBool());
    {
        // Map codec type value to combo index: 0=H264→0, 1=H265→1, 3=C3D→2
        int codecType = settings.value(perf::VIDEO_CODEC_TYPE, perf::VIDEO_CODEC_TYPE_DEFAULT).toInt();
        int comboIdx = (codecType == 3) ? 2 : codecType;
        cmbVideoCodecType->setCurrentIndex(std::clamp(comboIdx, 0, 2));
    }
    cmbVideoQualityPreset->setCurrentIndex(std::clamp(
        settings.value(perf::VIDEO_QUALITY_PRESET, perf::VIDEO_QUALITY_PRESET_DEFAULT).toInt(),
        0, perf::PRESET_COUNT - 1));
    chkRechunk32->setChecked(settings.value(perf::VIDEO_RECHUNK_32, perf::VIDEO_RECHUNK_32_DEFAULT).toBool());
    spinPrefetchLevels->setValue(settings.value(perf::PREFETCH_LEVELS, perf::PREFETCH_LEVELS_DEFAULT).toInt());
    spinIOThreads->setValue(settings.value(perf::IO_THREADS, perf::IO_THREADS_DEFAULT).toInt());

    // Enable/disable codec options based on checkbox
    auto updateVideoCodecEnabled = [this]{
        bool enabled = chkVideoRecompress->isChecked();
        cmbVideoCodecType->setEnabled(enabled);
        cmbVideoQualityPreset->setEnabled(enabled);
        chkRechunk32->setEnabled(enabled);
    };
    updateVideoCodecEnabled();
    connect(chkVideoRecompress, &QCheckBox::toggled, this, updateVideoCodecEnabled);

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
    connect(btnHelpFastInterpolation, &QPushButton::clicked, this, [this]{ QToolTip::showText(QCursor::pos(), btnHelpFastInterpolation->toolTip()); });
    connect(btnHelpRamCacheSize, &QPushButton::clicked, this, [this]{ QToolTip::showText(QCursor::pos(), btnHelpRamCacheSize->toolTip()); });
    connect(btnHelpDiskCacheSize, &QPushButton::clicked, this, [this]{ QToolTip::showText(QCursor::pos(), btnHelpDiskCacheSize->toolTip()); });
}

void SettingsDialog::accept()
{
    // Store the settings
    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    settings.setValue(volpkg::DEFAULT_PATH, edtDefaultPathVolpkg->text());
    settings.setValue(volpkg::AUTO_OPEN, chkAutoOpenVolpkg->isChecked() ? "1" : "0");

    settings.setValue(viewer::FWD_BACK_STEP_MS, spinFwdBackStepMs->value());
    settings.setValue(viewer::CENTER_ON_ZOOM, chkCenterOnZoom->isChecked() ? "1" : "0");
    settings.setValue(viewer::IMPACT_RANGE_STEPS, edtImpactRange->text());
    settings.setValue(viewer::SCAN_RANGE_STEPS, edtScanRange->text());
    settings.setValue(viewer::SCROLL_SPEED, spinScrollSpeed->value());
    settings.setValue(viewer::DISPLAY_SEGMENT_OPACITY, spinDisplayOpacity->value());
    settings.setValue(viewer::PLAY_SOUND_AFTER_SEG_RUN, chkPlaySoundAfterSegRun->isChecked() ? "1" : "0");
    settings.setValue(viewer::USERNAME, edtUsername->text());
    settings.setValue(viewer::RESET_VIEW_ON_SURFACE_CHANGE, chkResetViewOnSurfaceChange->isChecked() ? "1" : "0");
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
    settings.setValue(perf::FAST_INTERPOLATION, chkFastInterpolation->isChecked() ? "1" : "0");
    settings.setValue(perf::ENABLE_FILE_WATCHING, chkEnableFileWatching->isChecked() ? "1" : "0");

    // Cache settings
    settings.setValue(perf::RAM_CACHE_SIZE_GB, spinRamCacheSizeGB->value());
    settings.setValue(perf::DISK_CACHE_SIZE_GB, spinDiskCacheSizeGB->value());
    settings.setValue(viewer::REMOTE_CACHE_DIR, edtRemoteCachePath->text());

    // Video codec recompression
    settings.setValue(perf::VIDEO_RECOMPRESS_ENABLED, chkVideoRecompress->isChecked());
    {
        // Map combo index to codec type value: 0→0(H264), 1→1(H265), 2→3(C3D)
        static constexpr int comboToCodec[] = {0, 1, 3};
        int idx = std::clamp(cmbVideoCodecType->currentIndex(), 0, 2);
        settings.setValue(perf::VIDEO_CODEC_TYPE, comboToCodec[idx]);
    }
    settings.setValue(perf::VIDEO_QUALITY_PRESET, cmbVideoQualityPreset->currentIndex());
    settings.setValue(perf::VIDEO_RECHUNK_32, chkRechunk32->isChecked());
    settings.setValue(perf::PREFETCH_LEVELS, spinPrefetchLevels->value());
    settings.setValue(perf::IO_THREADS, spinIOThreads->value());

    QMessageBox::information(this, tr("Restart required"), tr("Note: Some settings only take effect once you restarted the app."));

    close();
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

