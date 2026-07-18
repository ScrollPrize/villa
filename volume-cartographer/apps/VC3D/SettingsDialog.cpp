#include "SettingsDialog.hpp"

#include "VCSettings.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/types/Volume.hpp"
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
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QFileDialog>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QProgressDialog>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QToolTip>



SettingsDialog::SettingsDialog(std::shared_ptr<VolumePkg> volumePackage,
                               std::shared_ptr<Volume> currentVolume,
                               std::filesystem::path currentVolumeCacheDir,
                               CacheChunkLayout currentVolumeChunkLayout,
                               QWidget *parent)
    : QDialog(parent)
    , _volumePackage(std::move(volumePackage))
    , _currentVolume(std::move(currentVolume))
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

    // IO threads is no longer user-configurable (tracks hardware_concurrency).
    if (spinIOThreads) {
        spinIOThreads->setEnabled(false);
        spinIOThreads->setValue(static_cast<int>(std::thread::hardware_concurrency()));
    }
    if (auto* lbl = findChild<QLabel*>("labelIOThreads")) lbl->hide();
    if (spinIOThreads) spinIOThreads->hide();

    chkCompressRemoteCache->setChecked(
        settings.value(perf::REMOTE_CACHE_COMPRESSION, perf::REMOTE_CACHE_COMPRESSION_DEFAULT).toBool());

    cmbCacheQuantization->addItem(tr("Lossless"), vc::kCacheQuantLossless);
    cmbCacheQuantization->addItem(tr("Near-lossless (max error ±1)"),
                                  vc::kCacheQuantMaxErr1);
    cmbCacheQuantization->addItem(tr("Near-lossless (max error ±2)"),
                                  vc::kCacheQuantMaxErr2);
    {
        const int width = settings.value(perf::REMOTE_CACHE_QUANTIZATION,
                                         perf::REMOTE_CACHE_QUANTIZATION_DEFAULT).toInt();
        const int idx = cmbCacheQuantization->findData(width);
        cmbCacheQuantization->setCurrentIndex(idx >= 0 ? idx : 0);
    }

    setupCacheActionControls();

    // Compacting an existing cache needs a currently shown remote volume.
    btnCompressExistingCache->setEnabled(!_currentVolumeCacheDir.empty());
    if (_currentVolumeCacheDir.empty()) {
        btnCompressExistingCache->setToolTip(
            tr("Open a remote volume first — compression applies to the disk cache of the currently shown volume."));
    }
    _redownloadCacheButton->setEnabled(!_currentVolumeCacheDir.empty() &&
                                       _currentVolume &&
                                       _currentVolume->isRemote());
    if (!_redownloadCacheButton->isEnabled()) {
        _redownloadCacheButton->setToolTip(
            tr("Open a remote volume first — redownload applies to the disk cache of the currently shown volume."));
    }
    connect(btnCompressExistingCache, &QPushButton::clicked, this,
            &SettingsDialog::compressExistingCache);
    connect(_redownloadCacheButton, &QPushButton::clicked, this,
            &SettingsDialog::redownloadExistingCache);

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
    _redownloadCacheButton = new QPushButton(tr("Redownload cache..."), groupBox_5);
    _redownloadCacheButton->setObjectName(QStringLiteral("btnRedownloadCache"));
    _redownloadCacheButton->setToolTip(
        tr("Fetch fresh versions of already-downloaded raw cache chunks for the currently shown volume, then write them using the selected compression setting."));

    _cacheActionWorkersSpin = new QSpinBox(groupBox_5);
    _cacheActionWorkersSpin->setObjectName(QStringLiteral("spinCacheActionWorkers"));
    _cacheActionWorkersSpin->setRange(1, 100);
    const int defaultWorkers = std::clamp(
        static_cast<int>(std::max(1u, std::thread::hardware_concurrency() / 2)),
        1,
        100);
    _cacheActionWorkersSpin->setValue(defaultWorkers);
    _cacheActionWorkersSpin->setToolTip(
        tr("Worker threads used by Compress existing cache and Redownload cache."));

    auto* label = new QLabel(tr("Workers"), groupBox_5);
    label->setObjectName(QStringLiteral("labelCacheActionWorkers"));
    label->setBuddy(_cacheActionWorkersSpin);

    auto* row = new QWidget(groupBox_5);
    row->setObjectName(QStringLiteral("cacheActionControls"));
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);
    if (gridLayout_5)
        gridLayout_5->removeWidget(btnCompressExistingCache);
    layout->addWidget(btnCompressExistingCache);
    layout->addWidget(_redownloadCacheButton);
    layout->addWidget(label);
    layout->addWidget(_cacheActionWorkersSpin);

    if (gridLayout_5)
        gridLayout_5->addWidget(row, 2, 2, 1, 1);
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
    settings.setValue(perf::REMOTE_CACHE_QUANTIZATION,
                      cmbCacheQuantization->currentData().toInt());
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

    // IO_THREADS setting removed — see CState::applyCacheBudget.

    QMessageBox::information(this, tr("Restart required"), tr("Note: Some settings only take effect once you restarted the app."));

    QDialog::accept();
}

namespace {

enum class RecompressResult { Done, Skipped, Failed };
enum class RedownloadResult { Done, Missing, Failed };

struct CacheFileEntry {
    vc::render::ChunkKey key;
    std::array<int, 3> shapeZYX{};
};

std::optional<int> parseLevelDirectory(const std::string& name)
{
    constexpr std::string_view prefix{"level_"};
    if (name.rfind(prefix, 0) != 0)
        return std::nullopt;
    char* end = nullptr;
    const long value = std::strtol(name.c_str() + prefix.size(), &end, 10);
    if (!end || *end != '\0' || value < 0 || value > std::numeric_limits<int>::max())
        return std::nullopt;
    return static_cast<int>(value);
}

std::optional<int> parseNonNegativeInt(const std::string& value)
{
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (!end || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max())
        return std::nullopt;
    return static_cast<int>(parsed);
}

std::optional<CacheFileEntry> cacheFileEntryForPath(
    const std::filesystem::path& cacheDir,
    const std::filesystem::path& path,
    const CacheChunkLayout& layout)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    const auto rel = fs::relative(path, cacheDir, ec);
    if (ec)
        return std::nullopt;

    std::vector<std::string> parts;
    for (const auto& part : rel)
        parts.push_back(part.string());
    if (parts.size() != 4)
        return std::nullopt;

    auto level = parseLevelDirectory(parts[0]);
    auto iz = parseNonNegativeInt(parts[1]);
    auto iy = parseNonNegativeInt(parts[2]);
    const auto stem = fs::path(parts[3]).stem().string();
    auto ix = parseNonNegativeInt(stem);
    if (!level || !iz || !iy || !ix)
        return std::nullopt;
    if (static_cast<std::size_t>(*level) >= layout.levelChunkShapes.size())
        return std::nullopt;

    CacheFileEntry entry;
    entry.key = {*level, *iz, *iy, *ix};
    entry.shapeZYX = layout.levelChunkShapes[static_cast<std::size_t>(*level)];
    if (entry.shapeZYX[0] <= 0 || entry.shapeZYX[1] <= 0 || entry.shapeZYX[2] <= 0)
        return std::nullopt;
    return entry;
}

std::vector<CacheFileEntry> collectRawCacheEntries(
    const std::filesystem::path& cacheDir,
    const CacheChunkLayout& layout,
    bool includeCompressed)
{
    namespace fs = std::filesystem;
    std::vector<CacheFileEntry> entries;
    std::set<vc::render::ChunkKey, bool(*)(const vc::render::ChunkKey&, const vc::render::ChunkKey&)> seen(
        [](const vc::render::ChunkKey& a, const vc::render::ChunkKey& b) {
            return std::tie(a.level, a.iz, a.iy, a.ix) <
                   std::tie(b.level, b.iz, b.iy, b.ix);
        });

    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(
             cacheDir, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec)
            break;
        if (!it->is_regular_file(ec))
            continue;
        const auto ext = it->path().extension();
        if (ext != ".bin" && ext != ".empty" &&
            !(includeCompressed && ext == vc::kCompressedCacheExtension)) {
            continue;
        }
        auto entry = cacheFileEntryForPath(cacheDir, it->path(), layout);
        if (!entry || !seen.insert(entry->key).second)
            continue;
        entries.push_back(*entry);
    }
    return entries;
}

std::filesystem::path rawCachePath(const std::filesystem::path& cacheDir,
                                   const vc::render::ChunkKey& key)
{
    return cacheDir / ("level_" + std::to_string(key.level)) /
           std::to_string(key.iz) / std::to_string(key.iy) /
           (std::to_string(key.ix) + ".bin");
}

std::filesystem::path compressedCachePath(const std::filesystem::path& cacheDir,
                                          const vc::render::ChunkKey& key)
{
    return cacheDir / ("level_" + std::to_string(key.level)) /
           std::to_string(key.iz) / std::to_string(key.iy) /
           (std::to_string(key.ix) + vc::kCompressedCacheExtension);
}

std::filesystem::path emptyCachePath(const std::filesystem::path& cacheDir,
                                     const vc::render::ChunkKey& key)
{
    return cacheDir / ("level_" + std::to_string(key.level)) /
           std::to_string(key.iz) / std::to_string(key.iy) /
           (std::to_string(key.ix) + ".empty");
}

bool writeFileAtomically(const std::filesystem::path& path,
                         std::span<const std::byte> bytes)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec)
        return false;
    const auto tmp = path.string() + ".tmp";
    {
        std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
        if (!file)
            return false;
        file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        if (!file) {
            fs::remove(tmp, ec);
            return false;
        }
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(path, ec);
        ec.clear();
        fs::rename(tmp, path, ec);
    }
    if (ec) {
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

RedownloadResult redownloadCacheEntry(
    vc::render::IChunkedArray& source,
    const std::shared_ptr<vc::render::PersistentZarrCacheBudget>& budget,
    const std::filesystem::path& cacheDir,
    const CacheFileEntry& entry,
    std::size_t elemSize,
    bool compress,
    int quantBinWidth,
    std::uint64_t& bytesOut)
{
    namespace fs = std::filesystem;

    const auto chunk = source.getChunkBlocking(
        entry.key.level, entry.key.iz, entry.key.iy, entry.key.ix);
    if (chunk.status == vc::render::ChunkStatus::Missing ||
        chunk.status == vc::render::ChunkStatus::AllFill) {
        const auto path = emptyCachePath(cacheDir, entry.key);
        std::vector<fs::path> replacements{
            rawCachePath(cacheDir, entry.key),
            compressedCachePath(cacheDir, entry.key)};
        auto reservation = budget
            ? budget->reserveWrite(path, 1, replacements)
            : vc::render::PersistentZarrCacheBudget::WriteReservation{};
        if (budget && !reservation)
            return RedownloadResult::Failed;
        std::error_code ec;
        fs::remove(replacements[0], ec);
        ec.clear();
        fs::remove(replacements[1], ec);
        ec.clear();
        fs::remove(path, ec);
        const std::byte newline{static_cast<unsigned char>('\n')};
        if (!writeFileAtomically(path, std::span<const std::byte>(&newline, 1))) {
            if (budget)
                reservation.commit();
            return RedownloadResult::Failed;
        }
        if (budget)
            reservation.commit();
        bytesOut += 1;
        return RedownloadResult::Missing;
    }
    if (chunk.status != vc::render::ChunkStatus::Data || !chunk.bytes)
        return RedownloadResult::Failed;

    std::span<const std::byte> payload(chunk.bytes->data(), chunk.bytes->size());
    std::vector<std::byte> compressed;
    fs::path path = rawCachePath(cacheDir, entry.key);
    if (compress) {
        try {
            compressed = vc::cacheCompress(
                payload,
                entry.shapeZYX,
                elemSize,
                vc::kCacheCompressionLevel,
                quantBinWidth);
        } catch (const std::exception&) {
            return RedownloadResult::Failed;
        }
        payload = std::span<const std::byte>(compressed.data(), compressed.size());
        path = compressedCachePath(cacheDir, entry.key);
    }
    std::vector<fs::path> replacements;
    for (const auto& candidate : {rawCachePath(cacheDir, entry.key),
                                  compressedCachePath(cacheDir, entry.key),
                                  emptyCachePath(cacheDir, entry.key)}) {
        if (candidate != path)
            replacements.push_back(candidate);
    }
    auto reservation = budget
        ? budget->reserveWrite(path, payload.size(), replacements)
        : vc::render::PersistentZarrCacheBudget::WriteReservation{};
    if (budget && !reservation)
        return RedownloadResult::Failed;
    std::error_code ec;
    for (const auto& replacement : replacements) {
        fs::remove(replacement, ec);
        ec.clear();
    }
    if (!writeFileAtomically(path, payload)) {
        if (budget)
            reservation.commit();
        return RedownloadResult::Failed;
    }
    if (budget)
        reservation.commit();
    bytesOut += payload.size();
    return RedownloadResult::Done;
}

// Recompress one cache chunk in place with the requested quantization
// (atomic tmp+rename; a replaced ".bin" source is removed on success).
// Raw ".bin" chunks are always encoded; ".zst" chunks are re-encoded when
// their recorded quantization width is below the requested one, when they
// carry an outdated entropy codec, or when they predate per-chunk delta
// filter selection (re-quantizing at or below the recorded width is a
// lossless no-op, so a codec or filter upgrade never loses data; the
// recorded width is kept when it exceeds the requested one). Returns Failed
// on any error — including a mismatched or unknown chunk shape, which
// cacheCompress rejects — and the original file is left untouched then.
RecompressResult compressCacheFile(
    const std::shared_ptr<vc::render::PersistentZarrCacheBudget>& budget,
    const std::filesystem::path& srcPath,
    std::array<int, 3> shapeZYX,
    std::size_t elemSize,
    int quantBinWidth,
    std::uint64_t& bytesIn,
    std::uint64_t& bytesOut)
{
    namespace fs = std::filesystem;

    auto readPin = budget
        ? budget->pinRead(srcPath)
        : vc::render::PersistentZarrCacheBudget::ReadPin{};
    std::vector<std::byte> input;
    {
        std::ifstream file(srcPath, std::ios::binary | std::ios::ate);
        if (!file)
            return RecompressResult::Failed;
        const auto size = file.tellg();
        if (size < 0)
            return RecompressResult::Failed;
        input.resize(static_cast<std::size_t>(size));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(input.data()), size);
        if (!file)
            return RecompressResult::Failed;
    }
    if (budget)
        readPin.complete(true);

    const bool alreadyCompressed =
        srcPath.extension() == vc::kCompressedCacheExtension;
    std::vector<std::byte> raw;
    std::span<const std::byte> rawSpan(input.data(), input.size());
    if (alreadyCompressed) {
        const std::span<const std::byte> inputSpan(input.data(), input.size());
        const auto existingWidth = vc::cacheQuantBinWidth(inputSpan);
        if (!existingWidth)
            return RecompressResult::Failed;
        if (*existingWidth >= quantBinWidth &&
            vc::cacheCodec(inputSpan) == vc::kCacheDefaultCodec &&
            vc::cacheDeltaMask(inputSpan).has_value())
            return RecompressResult::Skipped;
        quantBinWidth = std::max(quantBinWidth, *existingWidth);
        const std::size_t expectedSize =
            static_cast<std::size_t>(shapeZYX[0]) * shapeZYX[1] * shapeZYX[2] *
            elemSize;
        auto decoded = vc::cacheDecompress(
            std::span<const std::byte>(input.data(), input.size()), expectedSize);
        if (!decoded)
            return RecompressResult::Failed;
        raw = std::move(*decoded);
        rawSpan = {raw.data(), raw.size()};
    }

    std::vector<std::byte> compressed;
    try {
        compressed = vc::cacheCompress(rawSpan, shapeZYX, elemSize,
                                       vc::kCacheCompressionLevel, quantBinWidth);
    } catch (const std::exception&) {
        return RecompressResult::Failed;
    }

    fs::path zstPath = srcPath;
    zstPath.replace_extension(vc::kCompressedCacheExtension);
    std::vector<fs::path> replacements;
    if (!alreadyCompressed)
        replacements.push_back(srcPath);
    auto reservation = budget
        ? budget->reserveWrite(zstPath, compressed.size(), replacements)
        : vc::render::PersistentZarrCacheBudget::WriteReservation{};
    if (budget && !reservation)
        return RecompressResult::Failed;
    const fs::path tmpPath = zstPath.string() + ".tmp";
    {
        std::ofstream file(tmpPath, std::ios::binary | std::ios::trunc);
        if (!file) {
            if (budget)
                reservation.commit();
            return RecompressResult::Failed;
        }
        file.write(reinterpret_cast<const char*>(compressed.data()),
                   static_cast<std::streamsize>(compressed.size()));
        if (!file) {
            std::error_code ec;
            fs::remove(tmpPath, ec);
            if (budget)
                reservation.commit();
            return RecompressResult::Failed;
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
        if (budget)
            reservation.commit();
        return RecompressResult::Failed;
    }
    if (!alreadyCompressed)
        fs::remove(srcPath, ec);
    if (budget)
        reservation.commit();

    bytesIn += input.size();
    bytesOut += compressed.size();
    return RecompressResult::Done;
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

    // Uncompressed chunks are stored as ".bin". When a near-lossless mode
    // is selected, already-compressed ".zst" chunks are candidates too —
    // the worker re-encodes only those recorded with a narrower
    // quantization than requested. ".c3d"/".empty" entries have nothing to
    // gain. Each file's pyramid level ("level_N" path component) selects
    // the chunk shape used by the delta-zyx compression filter.
    const int quantBinWidth = cmbCacheQuantization->currentData().toInt();
    const auto& layout = _currentVolumeChunkLayout;
    const auto budget =
        vc::render::PersistentZarrCacheBudget::findForPath(cacheDir);
    struct RecompressFile {
        fs::path path;
        std::array<int, 3> shapeZYX;
    };
    std::vector<RecompressFile> files;
    auto keyLess = [](const vc::render::ChunkKey& a,
                      const vc::render::ChunkKey& b) {
        return std::tie(a.level, a.iz, a.iy, a.ix) <
               std::tie(b.level, b.iz, b.iy, b.ix);
    };
    std::map<vc::render::ChunkKey, std::size_t, decltype(keyLess)> fileByKey(keyLess);
    for (auto it = fs::recursive_directory_iterator(
             cacheDir, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec)
            break;
        if (!it->is_regular_file(ec))
            continue;
        const auto ext = it->path().extension();
        if (ext == ".bin" ||
            (quantBinWidth > vc::kCacheQuantLossless &&
             ext == vc::kCompressedCacheExtension)) {
            if (auto entry = cacheFileEntryForPath(cacheDir, it->path(), layout)) {
                if (auto [pos, inserted] = fileByKey.emplace(entry->key, files.size());
                    inserted) {
                    files.push_back({it->path(), entry->shapeZYX});
                } else if (ext == ".bin") {
                    files[pos->second] = {it->path(), entry->shapeZYX};
                }
            }
        }
    }
    if (files.empty()) {
        QMessageBox::information(this, tr("Compress cache"),
            tr("The cache for the currently shown volume has no chunks to compress."));
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
    std::atomic<std::size_t> skipped{0};
    std::atomic<std::uint64_t> totalIn{0};
    std::atomic<std::uint64_t> totalOut{0};
    std::atomic<bool> cancelled{false};

    // std::async workers: VC3D pins OpenMP/cv::parallel_for_ to one thread,
    // so explicit fan-out is the supported parallelism route here.
    const std::size_t workerCount = std::min<std::size_t>(
        files.size(),
        static_cast<std::size_t>(_cacheActionWorkersSpin->value()));
    std::vector<std::future<void>> workers;
    workers.reserve(workerCount);
    for (std::size_t w = 0; w < workerCount; ++w) {
        workers.push_back(std::async(std::launch::async, [&]{
            std::uint64_t bytesIn = 0;
            std::uint64_t bytesOut = 0;
            std::size_t localFailures = 0;
            std::size_t localSkipped = 0;
            while (!cancelled.load(std::memory_order_relaxed)) {
                const auto i = nextIndex.fetch_add(1, std::memory_order_relaxed);
                if (i >= files.size())
                    break;
                switch (compressCacheFile(budget, files[i].path, files[i].shapeZYX,
                                          layout.elemSize, quantBinWidth,
                                          bytesIn, bytesOut)) {
                case RecompressResult::Failed: ++localFailures; break;
                case RecompressResult::Skipped: ++localSkipped; break;
                case RecompressResult::Done: break;
                }
                done.fetch_add(1, std::memory_order_relaxed);
            }
            totalIn += bytesIn;
            totalOut += bytesOut;
            failures += localFailures;
            skipped += localSkipped;
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
    const std::size_t compressed =
        done.load() - failures.load() - skipped.load();
    QString summary =
        tr("Compressed %1 of %2 chunks: %3 GiB -> %4 GiB.")
            .arg(compressed)
            .arg(files.size())
            .arg(gib(totalIn.load()))
            .arg(gib(totalOut.load()));
    if (cancelled)
        summary += tr("\nCancelled — the remaining chunks are unchanged and can be compressed later.");
    if (skipped.load() > 0)
        summary += tr("\n%1 chunks already matched the selected accuracy and were left as-is.").arg(skipped.load());
    if (failures.load() > 0)
        summary += tr("\n%1 chunks could not be compressed and were left as-is.").arg(failures.load());
    QMessageBox::information(this, tr("Compress cache"), summary);
}

void SettingsDialog::redownloadExistingCache()
{
    namespace fs = std::filesystem;

    const fs::path cacheDir = _currentVolumeCacheDir;
    std::error_code ec;
    if (cacheDir.empty() || !fs::is_directory(cacheDir, ec) ||
        !_currentVolume || !_currentVolume->isRemote()) {
        QMessageBox::information(this, tr("Redownload cache"),
            tr("No remote disk cache found for the currently shown volume."));
        return;
    }

    const auto entries = collectRawCacheEntries(
        cacheDir, _currentVolumeChunkLayout, true);
    if (entries.empty()) {
        QMessageBox::information(this, tr("Redownload cache"),
            tr("The cache for the currently shown volume has no raw chunks to redownload."));
        return;
    }

    const bool compress = chkCompressRemoteCache->isChecked();
    const int quantBinWidth = cmbCacheQuantization->currentData().toInt();
    const auto budget =
        vc::render::PersistentZarrCacheBudget::findForPath(cacheDir);
    const std::size_t workerCount = std::min<std::size_t>(
        entries.size(),
        static_cast<std::size_t>(_cacheActionWorkersSpin->value()));

    QProgressDialog progress(
        tr("Redownloading %1 cached chunks...").arg(entries.size()),
        tr("Cancel"), 0, static_cast<int>(entries.size()), this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    progress.setValue(0);

    std::shared_ptr<vc::render::ChunkCache> source;
    try {
        auto freshVolume = Volume::NewFromUrl(
            _currentVolume->remoteLocator(), {}, _currentVolume->remoteAuth());
        vc::render::ChunkCache::Options options;
        options.maxConcurrentReads = workerCount;
        options.compressPersistentCache = false;
        options.decodedByteCapacity = 512ULL * 1024ULL * 1024ULL;
        source = freshVolume->createChunkCache(std::move(options));
    } catch (const std::exception& e) {
        QMessageBox::warning(this, tr("Redownload cache"),
            tr("Could not open the remote volume for redownload:\n%1")
                .arg(QString::fromUtf8(e.what())));
        return;
    }
    if (!source) {
        QMessageBox::warning(this, tr("Redownload cache"),
            tr("Could not create a remote chunk reader for redownload."));
        return;
    }

    std::atomic<std::size_t> nextIndex{0};
    std::atomic<std::size_t> done{0};
    std::atomic<std::size_t> refreshed{0};
    std::atomic<std::size_t> missing{0};
    std::atomic<std::size_t> failures{0};
    std::atomic<std::uint64_t> totalOut{0};
    std::atomic<bool> cancelled{false};

    std::vector<std::future<void>> workers;
    workers.reserve(workerCount);
    for (std::size_t w = 0; w < workerCount; ++w) {
        workers.push_back(std::async(std::launch::async, [&]{
            std::uint64_t bytesOut = 0;
            std::size_t localRefreshed = 0;
            std::size_t localMissing = 0;
            std::size_t localFailures = 0;
            while (!cancelled.load(std::memory_order_relaxed)) {
                const auto i = nextIndex.fetch_add(1, std::memory_order_relaxed);
                if (i >= entries.size())
                    break;
                switch (redownloadCacheEntry(*source, budget, cacheDir, entries[i],
                                             _currentVolumeChunkLayout.elemSize,
                                             compress, quantBinWidth, bytesOut)) {
                case RedownloadResult::Done: ++localRefreshed; break;
                case RedownloadResult::Missing: ++localMissing; break;
                case RedownloadResult::Failed: ++localFailures; break;
                }
                done.fetch_add(1, std::memory_order_relaxed);
            }
            totalOut += bytesOut;
            refreshed += localRefreshed;
            missing += localMissing;
            failures += localFailures;
        }));
    }

    while (done.load() < entries.size() && !progress.wasCanceled()) {
        progress.setValue(static_cast<int>(done.load()));
        QCoreApplication::processEvents(QEventLoop::AllEvents, 50);
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    cancelled = progress.wasCanceled();
    for (auto& worker : workers)
        worker.wait();
    progress.setValue(static_cast<int>(entries.size()));

    const auto gib = [](std::uint64_t bytes) {
        return QString::number(static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), 'f', 2);
    };
    QString summary =
        tr("Redownloaded %1 of %2 chunks; wrote %3 GiB using %4.")
            .arg(refreshed.load())
            .arg(entries.size())
            .arg(gib(totalOut.load()))
            .arg(compress ? tr("the selected compression setting")
                          : tr("uncompressed cache files"));
    if (missing.load() > 0)
        summary += tr("\n%1 chunks are currently missing remotely and were stored as empty markers.").arg(missing.load());
    if (cancelled)
        summary += tr("\nCancelled — chunks not yet processed are unchanged.");
    if (failures.load() > 0)
        summary += tr("\n%1 chunks could not be redownloaded and were left unchanged.").arg(failures.load());
    QMessageBox::information(this, tr("Redownload cache"), summary);
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
