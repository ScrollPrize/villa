#include "SettingsDialog.hpp"

#include "VCSettings.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/CacheCompression.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <future>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QFileDialog>
#include <QGridLayout>
#include <QLabel>
#include <QProgressDialog>
#include <QSettings>
#include <QMessageBox>
#include <QToolTip>



SettingsDialog::SettingsDialog(std::shared_ptr<VolumePkg> volumePackage,
                               std::filesystem::path currentVolumeCacheDir,
                               CacheChunkLayout currentVolumeChunkLayout,
                               QWidget *parent)
    : QDialog(parent)
    , _volumePackage(std::move(volumePackage))
    , _currentVolumeCacheDir(std::move(currentVolumeCacheDir))
    , _currentVolumeChunkLayout(std::move(currentVolumeChunkLayout))
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
    {
        const QString stored =
            settings.value(viewer::REMOTE_CACHE_DIR).toString();
        edtRemoteCachePath->setText(vc3d::remoteCachePath(stored));
    }

    // Per-segment rotating-backup count.
    if (spinSegmentBackupCount) {
        spinSegmentBackupCount->setValue(
            settings.value(backup::SEGMENT_COUNT, backup::SEGMENT_COUNT_DEFAULT).toInt());
    }

    // IO threads is no longer user-configurable (tracks hardware_concurrency).
    if (spinIOThreads) {
        spinIOThreads->setEnabled(false);
        spinIOThreads->setValue(static_cast<int>(std::thread::hardware_concurrency()));
    }
    if (auto* lbl = findChild<QLabel*>("labelIOThreads")) lbl->hide();
    if (spinIOThreads) spinIOThreads->hide();

    chkCompressRemoteCache->setChecked(
        settings.value(perf::REMOTE_CACHE_COMPRESSION, perf::REMOTE_CACHE_COMPRESSION_DEFAULT).toBool());

    // Compacting an existing cache needs a currently shown remote volume.
    btnCompressExistingCache->setEnabled(!_currentVolumeCacheDir.empty());
    if (_currentVolumeCacheDir.empty()) {
        btnCompressExistingCache->setToolTip(
            tr("Open a remote volume first — compression applies to the disk cache of the currently shown volume."));
    }
    connect(btnCompressExistingCache, &QPushButton::clicked, this,
            &SettingsDialog::compressExistingCache);

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
    settings.setValue(viewer::REMOTE_CACHE_DIR, edtRemoteCachePath->text());
    settings.setValue(perf::REMOTE_CACHE_COMPRESSION, chkCompressRemoteCache->isChecked());

    // Per-segment backup count: persist and apply live (no restart needed).
    if (spinSegmentBackupCount) {
        const int backupCount = spinSegmentBackupCount->value();
        settings.setValue(backup::SEGMENT_COUNT, backupCount);
        QuadSurface::setBackupCount(backupCount);
    }

    // IO_THREADS setting removed — see CState::applyCacheBudget.

    QMessageBox::information(this, tr("Restart required"), tr("Note: Some settings only take effect once you restarted the app."));

    QDialog::accept();
}

namespace {

// Compress one raw cache chunk in place: <name>.bin -> <name>.zst
// (atomic tmp+rename, source removed on success). Returns false on failure;
// the original file is left untouched in that case. shapeZYX/elemSize enable
// the delta-zyx filter; a mismatched or unknown shape falls back to a plain
// zstd frame inside cacheCompress.
bool compressCacheFile(const std::filesystem::path& binPath,
                       std::array<int, 3> shapeZYX,
                       std::size_t elemSize,
                       std::uint64_t& bytesIn,
                       std::uint64_t& bytesOut)
{
    namespace fs = std::filesystem;

    std::vector<std::byte> input;
    {
        std::ifstream file(binPath, std::ios::binary | std::ios::ate);
        if (!file)
            return false;
        const auto size = file.tellg();
        if (size < 0)
            return false;
        input.resize(static_cast<std::size_t>(size));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(input.data()), size);
        if (!file)
            return false;
    }

    std::vector<std::byte> compressed;
    try {
        compressed = vc::cacheCompress(
            std::span<const std::byte>(input.data(), input.size()),
            shapeZYX,
            elemSize);
    } catch (const std::exception&) {
        return false;
    }

    fs::path zstPath = binPath;
    zstPath.replace_extension(vc::kCompressedCacheExtension);
    const fs::path tmpPath = zstPath.string() + ".tmp";
    {
        std::ofstream file(tmpPath, std::ios::binary | std::ios::trunc);
        if (!file)
            return false;
        file.write(reinterpret_cast<const char*>(compressed.data()),
                   static_cast<std::streamsize>(compressed.size()));
        if (!file) {
            std::error_code ec;
            fs::remove(tmpPath, ec);
            return false;
        }
    }
    std::error_code ec;
    fs::rename(tmpPath, zstPath, ec);
    if (ec) {
        fs::remove(zstPath, ec);
        ec.clear();
        fs::rename(tmpPath, zstPath, ec);
    }
    if (ec) {
        fs::remove(tmpPath, ec);
        return false;
    }
    fs::remove(binPath, ec);

    bytesIn += input.size();
    bytesOut += compressed.size();
    return true;
}

} // namespace

void SettingsDialog::compressExistingCache()
{
    namespace fs = std::filesystem;

    const fs::path cacheDir = _currentVolumeCacheDir;
    std::error_code ec;
    if (cacheDir.empty() || !fs::is_directory(cacheDir, ec)) {
        QMessageBox::information(this, tr("Compress cache"),
            tr("No disk cache found for the currently shown volume."));
        return;
    }

    // Uncompressed chunks are stored as ".bin"; ".zst" is already
    // compressed and ".c3d"/".empty" entries have nothing to gain.
    // Each file's pyramid level ("level_N" path component) selects the
    // chunk shape used by the delta-zyx compression filter.
    const auto& layout = _currentVolumeChunkLayout;
    auto shapeForFile = [&](const fs::path& path) -> std::array<int, 3> {
        const auto rel = fs::relative(path, cacheDir, ec);
        if (ec || rel.empty())
            return {0, 0, 0};
        const std::string top = rel.begin()->string();
        constexpr std::string_view prefix{"level_"};
        if (top.rfind(prefix, 0) != 0)
            return {0, 0, 0};
        const std::size_t level = std::strtoul(top.c_str() + prefix.size(), nullptr, 10);
        if (level >= layout.levelChunkShapes.size())
            return {0, 0, 0};
        return layout.levelChunkShapes[level];
    };

    std::vector<std::pair<fs::path, std::array<int, 3>>> files;
    for (auto it = fs::recursive_directory_iterator(
             cacheDir, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec)
            break;
        if (it->is_regular_file(ec) && it->path().extension() == ".bin")
            files.emplace_back(it->path(), shapeForFile(it->path()));
    }
    if (files.empty()) {
        QMessageBox::information(this, tr("Compress cache"),
            tr("The cache for the currently shown volume has no uncompressed chunks."));
        return;
    }

    QProgressDialog progress(
        tr("Compressing %1 cached chunks...").arg(files.size()),
        tr("Cancel"), 0, static_cast<int>(files.size()), this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    progress.setValue(0);

    std::atomic<std::size_t> nextIndex{0};
    std::atomic<std::size_t> done{0};
    std::atomic<std::size_t> failures{0};
    std::atomic<std::uint64_t> totalIn{0};
    std::atomic<std::uint64_t> totalOut{0};
    std::atomic<bool> cancelled{false};

    // std::async workers: VC3D pins OpenMP/cv::parallel_for_ to one thread,
    // so explicit fan-out is the supported parallelism route here.
    const std::size_t workerCount = std::min<std::size_t>(
        files.size(),
        std::max<std::size_t>(2, std::thread::hardware_concurrency() / 2));
    std::vector<std::future<void>> workers;
    workers.reserve(workerCount);
    for (std::size_t w = 0; w < workerCount; ++w) {
        workers.push_back(std::async(std::launch::async, [&]{
            std::uint64_t bytesIn = 0;
            std::uint64_t bytesOut = 0;
            std::size_t localFailures = 0;
            while (!cancelled.load(std::memory_order_relaxed)) {
                const auto i = nextIndex.fetch_add(1, std::memory_order_relaxed);
                if (i >= files.size())
                    break;
                if (!compressCacheFile(files[i].first, files[i].second,
                                       layout.elemSize, bytesIn, bytesOut))
                    ++localFailures;
                done.fetch_add(1, std::memory_order_relaxed);
            }
            totalIn += bytesIn;
            totalOut += bytesOut;
            failures += localFailures;
        }));
    }

    while (done.load() < files.size() && !progress.wasCanceled()) {
        progress.setValue(static_cast<int>(done.load()));
        QCoreApplication::processEvents(QEventLoop::AllEvents, 50);
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    cancelled = progress.wasCanceled();
    for (auto& worker : workers)
        worker.wait();
    progress.setValue(static_cast<int>(files.size()));

    const auto gib = [](std::uint64_t bytes) {
        return QString::number(static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), 'f', 2);
    };
    const std::size_t compressed = done.load() - failures.load();
    QString summary =
        tr("Compressed %1 of %2 chunks: %3 GiB -> %4 GiB.")
            .arg(compressed)
            .arg(files.size())
            .arg(gib(totalIn.load()))
            .arg(gib(totalOut.load()));
    if (cancelled)
        summary += tr("\nCancelled — the remaining chunks are unchanged and can be compressed later.");
    if (failures.load() > 0)
        summary += tr("\n%1 chunks could not be compressed and were left as-is.").arg(failures.load());
    QMessageBox::information(this, tr("Compress cache"), summary);
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
