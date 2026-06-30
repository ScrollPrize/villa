#include "OpenDataCatalogWindow.hpp"

#include "OpenDataSegmentCache.hpp"
#include "VCSettings.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/HttpFetch.hpp"

#include <nlohmann/json.hpp>

#include <QCheckBox>
#include <QClipboard>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QGuiApplication>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QIODevice>
#include <QItemSelectionModel>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSaveFile>
#include <QSignalBlocker>
#include <QSplitter>
#include <QStandardPaths>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTabWidget>
#include <QUrl>
#include <QVBoxLayout>
#include <QtConcurrent>

#include <exception>

namespace vc3d::opendata {
namespace {

QString qstr(const std::string& value)
{
    return QString::fromStdString(value);
}

QString optionalNumber(std::optional<double> value, char format = 'f', int precision = 3)
{
    return value ? QString::number(*value, format, precision) : QString();
}

QString optionalInt(std::optional<int> value)
{
    return value ? QString::number(*value) : QString();
}

QString yesNo(bool value)
{
    return value ? QStringLiteral("Yes") : QStringLiteral("No");
}

QTableWidgetItem* item(const QString& text)
{
    auto* tableItem = new QTableWidgetItem(text);
    tableItem->setFlags(tableItem->flags() & ~Qt::ItemIsEditable);
    return tableItem;
}

QTableWidgetItem* numericItem(qulonglong value)
{
    auto* tableItem = item(QString::number(value));
    tableItem->setData(Qt::UserRole, value);
    return tableItem;
}

QString firstArtifactUrl(const std::vector<OpenDataArtifact>& artifacts, const QString& preferredType = {})
{
    const OpenDataArtifact* artifact = nullptr;
    if (!preferredType.isEmpty()) {
        artifact = findArtifact(artifacts, preferredType.toStdString());
    }
    if (!artifact && !artifacts.empty()) {
        artifact = &artifacts.front();
    }
    if (!artifact) {
        return {};
    }
    return qstr(artifact->resolvedUrl.empty() ? artifact->sourcePath : artifact->resolvedUrl);
}

QString artifactUrl(const OpenDataArtifact& artifact)
{
    return qstr(artifact.resolvedUrl.empty() ? artifact.sourcePath : artifact.resolvedUrl);
}

QString manifestJsonError(const std::exception& e)
{
    return QStringLiteral("Could not parse cached open-data manifest: %1").arg(QString::fromUtf8(e.what()));
}

} // namespace

OpenDataCatalogWindow::OpenDataCatalogWindow(QWidget* parent)
    : QDialog(parent)
{
    buildUi();
    loadCachedManifestIfAvailable();
    reloadManifest();
}

OpenDataCatalogWindow::~OpenDataCatalogWindow()
{
    if (_fetchWatcher) {
        _fetchWatcher->cancel();
        _fetchWatcher->waitForFinished();
    }
}

void OpenDataCatalogWindow::setOpenSampleHandler(std::function<void(const OpenDataSample&)> handler)
{
    _openSampleHandler = std::move(handler);
    updateActionButtons();
}

void OpenDataCatalogWindow::buildUi()
{
    setWindowTitle(tr("Open Data Catalog"));
    resize(1120, 760);

    auto* mainLayout = new QVBoxLayout(this);

    auto* topRow = new QHBoxLayout;
    _searchEdit = new QLineEdit(this);
    _searchEdit->setPlaceholderText(tr("Search sample ID"));
    _segmentsOnlyCheck = new QCheckBox(tr("Segments"), this);
    _tifxyzOnlyCheck = new QCheckBox(tr("TIFXYZ"), this);
    _inkOnlyCheck = new QCheckBox(tr("Ink"), this);
    _refreshButton = new QPushButton(tr("Refresh"), this);
    topRow->addWidget(_searchEdit, 1);
    topRow->addWidget(_segmentsOnlyCheck);
    topRow->addWidget(_tifxyzOnlyCheck);
    topRow->addWidget(_inkOnlyCheck);
    topRow->addWidget(_refreshButton);
    mainLayout->addLayout(topRow);

    auto* splitter = new QSplitter(Qt::Horizontal, this);
    _sampleTable = new QTableWidget(splitter);
    _sampleTable->setColumnCount(7);
    _sampleTable->setHorizontalHeaderLabels({
        tr("Sample ID"),
        tr("Type"),
        tr("Scans"),
        tr("Volumes"),
        tr("Segments"),
        tr("TIFXYZ"),
        tr("Ink")
    });
    _sampleTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _sampleTable->setSelectionMode(QAbstractItemView::SingleSelection);
    _sampleTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _sampleTable->setSortingEnabled(true);
    _sampleTable->horizontalHeader()->setStretchLastSection(true);
    _sampleTable->verticalHeader()->hide();

    _tabs = new QTabWidget(splitter);
    _overviewLabel = new QLabel(_tabs);
    _overviewLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _overviewLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    _overviewLabel->setWordWrap(true);
    _tabs->addTab(_overviewLabel, tr("Overview"));

    _scansTable = new QTableWidget(_tabs);
    _scansTable->setColumnCount(4);
    _scansTable->setHorizontalHeaderLabels({tr("Scan ID"), tr("Suffix"), tr("Created"), tr("Artifacts")});
    _scansTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _scansTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _scansTable->horizontalHeader()->setStretchLastSection(true);
    _scansTable->verticalHeader()->hide();
    _tabs->addTab(_scansTable, tr("Scans"));

    auto* volumesPage = new QWidget(_tabs);
    auto* volumesLayout = new QVBoxLayout(volumesPage);
    _volumesTable = new QTableWidget(volumesPage);
    _volumesTable->setColumnCount(8);
    _volumesTable->setHorizontalHeaderLabels({
        tr("Volume ID"),
        tr("Scan ID"),
        tr("Suffix"),
        tr("Resolution"),
        tr("Energy"),
        tr("Detector distance"),
        tr("Format"),
        tr("Created")
    });
    _volumesTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _volumesTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _volumesTable->setSelectionMode(QAbstractItemView::SingleSelection);
    _volumesTable->horizontalHeader()->setStretchLastSection(true);
    _volumesTable->verticalHeader()->hide();
    auto* volumeActions = new QHBoxLayout;
    _copyVolumeUrlButton = new QPushButton(tr("Copy URL"), volumesPage);
    _openVolumeUrlButton = new QPushButton(tr("Open URL"), volumesPage);
    volumeActions->addStretch(1);
    volumeActions->addWidget(_copyVolumeUrlButton);
    volumeActions->addWidget(_openVolumeUrlButton);
    volumesLayout->addWidget(_volumesTable, 1);
    volumesLayout->addLayout(volumeActions);
    _tabs->addTab(volumesPage, tr("Volumes"));

    auto* segmentsPage = new QWidget(_tabs);
    auto* segmentsLayout = new QVBoxLayout(segmentsPage);
    _segmentsTable = new QTableWidget(segmentsPage);
    _segmentsTable->setColumnCount(10);
    _segmentsTable->setHorizontalHeaderLabels({
        tr("Segment ID"),
        tr("Suffix"),
        tr("Base volume"),
        tr("Width"),
        tr("Height"),
        tr("TIFXYZ"),
        tr("Ink"),
        tr("Layers Zarr"),
        tr("Cache"),
        tr("Created")
    });
    _segmentsTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _segmentsTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _segmentsTable->setSelectionMode(QAbstractItemView::SingleSelection);
    _segmentsTable->horizontalHeader()->setStretchLastSection(true);
    _segmentsTable->verticalHeader()->hide();
    auto* segmentActions = new QHBoxLayout;
    _cacheSegmentButton = new QPushButton(tr("Cache Selected"), segmentsPage);
    _openSegmentCacheFolderButton = new QPushButton(tr("Open Cache Folder"), segmentsPage);
    _copySegmentUrlButton = new QPushButton(tr("Copy Source URL"), segmentsPage);
    _openSegmentUrlButton = new QPushButton(tr("Open Source URL"), segmentsPage);
    segmentActions->addStretch(1);
    segmentActions->addWidget(_cacheSegmentButton);
    segmentActions->addWidget(_openSegmentCacheFolderButton);
    segmentActions->addWidget(_copySegmentUrlButton);
    segmentActions->addWidget(_openSegmentUrlButton);
    segmentsLayout->addWidget(_segmentsTable, 1);
    segmentsLayout->addLayout(segmentActions);
    _tabs->addTab(segmentsPage, tr("Segments"));

    splitter->addWidget(_sampleTable);
    splitter->addWidget(_tabs);
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 3);
    mainLayout->addWidget(splitter, 1);

    auto* bottomRow = new QHBoxLayout;
    _statusLabel = new QLabel(this);
    _statusLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _openSampleButton = new QPushButton(tr("Open Sample"), this);
    auto* closeButton = new QPushButton(tr("Close"), this);
    bottomRow->addWidget(_statusLabel, 1);
    bottomRow->addWidget(_openSampleButton);
    bottomRow->addWidget(closeButton);
    mainLayout->addLayout(bottomRow);

    connect(_refreshButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::reloadManifest);
    connect(_searchEdit, &QLineEdit::textChanged, this, &OpenDataCatalogWindow::updateSampleFilter);
    connect(_segmentsOnlyCheck, &QCheckBox::toggled, this, &OpenDataCatalogWindow::updateSampleFilter);
    connect(_tifxyzOnlyCheck, &QCheckBox::toggled, this, &OpenDataCatalogWindow::updateSampleFilter);
    connect(_inkOnlyCheck, &QCheckBox::toggled, this, &OpenDataCatalogWindow::updateSampleFilter);
    connect(_sampleTable->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &OpenDataCatalogWindow::updateSelectedSample);
    connect(_volumesTable->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &OpenDataCatalogWindow::updateActionButtons);
    connect(_segmentsTable->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &OpenDataCatalogWindow::updateActionButtons);
    connect(_copyVolumeUrlButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::copySelectedVolumeUrl);
    connect(_openVolumeUrlButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::openSelectedVolumeUrl);
    connect(_copySegmentUrlButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::copySelectedSegmentUrl);
    connect(_openSegmentUrlButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::openSelectedSegmentUrl);
    connect(_cacheSegmentButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::cacheSelectedSegment);
    connect(_openSegmentCacheFolderButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::openSelectedSegmentCacheFolder);
    connect(_openSampleButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::openSelectedSample);
    connect(closeButton, &QPushButton::clicked, this, &QDialog::accept);

    clearDetails();
    updateActionButtons();
}

void OpenDataCatalogWindow::reloadManifest()
{
    if (_fetchWatcher && _fetchWatcher->isRunning()) {
        return;
    }

    setLoading(true);

    _fetchWatcher = new QFutureWatcher<ManifestLoadResult>(this);
    connect(_fetchWatcher, &QFutureWatcher<ManifestLoadResult>::finished,
            this, &OpenDataCatalogWindow::onFetchFinished);

    const auto manifestUrl = std::string(kDefaultManifestUrl);
    _fetchWatcher->setFuture(QtConcurrent::run([manifestUrl]() {
        ManifestLoadResult result;
        result.sourceLabel = QString::fromStdString(manifestUrl);
        try {
            const auto body = vc::httpGetString(manifestUrl);
            if (body.empty()) {
                result.error = QStringLiteral("Open-data manifest fetch returned an empty response.");
                return result;
            }
            result.jsonText = QString::fromStdString(body);
            result.manifest = parseOpenDataManifest(body, manifestUrl);
        } catch (const std::exception& e) {
            result.error = QString::fromUtf8(e.what());
        } catch (...) {
            result.error = QStringLiteral("Unknown error while fetching open-data manifest.");
        }
        return result;
    }));
}

void OpenDataCatalogWindow::onFetchFinished()
{
    QFutureWatcher<ManifestLoadResult>* watcher = _fetchWatcher;
    _fetchWatcher = nullptr;
    const ManifestLoadResult result = watcher->result();
    watcher->deleteLater();
    setLoading(false);
    if (!result.error.isEmpty()) {
        if (_manifest) {
            setStatus(tr("Using cached catalog. Live refresh failed: %1").arg(result.error));
        } else {
            setStatus(tr("Live catalog refresh failed: %1").arg(result.error));
        }
        return;
    }

    persistFetchedManifest(result);
    applyManifest(result.manifest, result.sourceLabel, false);
}

void OpenDataCatalogWindow::loadCachedManifestIfAvailable()
{
    const auto path = cachedManifestPath();
    if (!std::filesystem::is_regular_file(path)) {
        return;
    }
    try {
        auto manifest = loadOpenDataManifestFile(path, std::string(kDefaultManifestUrl));
        applyManifest(std::move(manifest), qstr(path.string()), true);
    } catch (const std::exception& e) {
        setStatus(manifestJsonError(e));
    }
}

void OpenDataCatalogWindow::applyManifest(OpenDataManifest manifest, QString sourceLabel, bool fromCache)
{
    _manifest = std::move(manifest);
    populateSamples();
    setStatus(tr("%1 catalog: %2 samples, %3 models. Source: %4")
                  .arg(fromCache ? tr("Cached") : tr("Live"))
                  .arg(_manifest->samples.size())
                  .arg(_manifest->models.size())
                  .arg(sourceLabel));
}

void OpenDataCatalogWindow::setStatus(const QString& text)
{
    if (_statusLabel) {
        _statusLabel->setText(text);
    }
}

void OpenDataCatalogWindow::setLoading(bool loading)
{
    if (_refreshButton) {
        _refreshButton->setEnabled(!loading);
        _refreshButton->setText(loading ? tr("Refreshing...") : tr("Refresh"));
    }
    if (loading && !_manifest) {
        setStatus(tr("Fetching open-data catalog..."));
    }
    updateActionButtons();
}

void OpenDataCatalogWindow::persistFetchedManifest(const ManifestLoadResult& result) const
{
    if (result.jsonText.isEmpty()) {
        return;
    }

    const auto root = cacheRoot();
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) {
        return;
    }

    QSaveFile manifestFile(QString::fromStdString(cachedManifestPath().string()));
    if (manifestFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        manifestFile.write(result.jsonText.toUtf8());
        manifestFile.commit();
    }

    nlohmann::json meta = {
        {"manifest_url", std::string(kDefaultManifestUrl)},
        {"fetched_at_utc", QDateTime::currentDateTimeUtc().toString(Qt::ISODate).toStdString()},
        {"sample_count", result.manifest.samples.size()},
        {"model_count", result.manifest.models.size()}
    };
    QSaveFile metadataFile(QString::fromStdString(cacheMetadataPath().string()));
    if (metadataFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        const auto bytes = QByteArray::fromStdString(meta.dump(2));
        metadataFile.write(bytes);
        metadataFile.commit();
    }
}

void OpenDataCatalogWindow::populateSamples()
{
    _visibleSampleIndexes.clear();
    if (!_manifest) {
        _sampleTable->setRowCount(0);
        clearDetails();
        return;
    }

    const QString needle = _searchEdit->text().trimmed();
    for (std::size_t i = 0; i < _manifest->samples.size(); ++i) {
        const auto& sample = _manifest->samples[i];
        if (!needle.isEmpty() &&
            !qstr(sample.id).contains(needle, Qt::CaseInsensitive)) {
            continue;
        }
        if (_segmentsOnlyCheck->isChecked() && sample.segmentCount() == 0) {
            continue;
        }
        if (_tifxyzOnlyCheck->isChecked() && sample.tifxyzSegmentCount() == 0) {
            continue;
        }
        if (_inkOnlyCheck->isChecked() && sample.inkDetectionSegmentCount() == 0) {
            continue;
        }
        _visibleSampleIndexes.push_back(i);
    }

    QSignalBlocker blocker(_sampleTable);
    _sampleTable->setSortingEnabled(false);
    _sampleTable->setRowCount(static_cast<int>(_visibleSampleIndexes.size()));
    for (int row = 0; row < static_cast<int>(_visibleSampleIndexes.size()); ++row) {
        const auto sampleIndex = _visibleSampleIndexes[static_cast<std::size_t>(row)];
        const auto& sample = _manifest->samples[sampleIndex];
        auto* idItem = item(qstr(sample.id));
        idItem->setData(Qt::UserRole, QVariant::fromValue<qulonglong>(sampleIndex));
        _sampleTable->setItem(row, 0, idItem);
        _sampleTable->setItem(row, 1, item(qstr(sample.type)));
        _sampleTable->setItem(row, 2, numericItem(sample.scanCount()));
        _sampleTable->setItem(row, 3, numericItem(sample.volumeCount()));
        _sampleTable->setItem(row, 4, numericItem(sample.segmentCount()));
        _sampleTable->setItem(row, 5, numericItem(sample.tifxyzSegmentCount()));
        _sampleTable->setItem(row, 6, numericItem(sample.inkDetectionSegmentCount()));
    }
    _sampleTable->setSortingEnabled(true);
    _sampleTable->resizeColumnsToContents();

    if (_sampleTable->rowCount() > 0) {
        _sampleTable->selectRow(0);
    } else {
        clearDetails();
    }
}

void OpenDataCatalogWindow::updateSampleFilter()
{
    populateSamples();
}

void OpenDataCatalogWindow::updateSelectedSample()
{
    populateDetails(selectedSample());
}

void OpenDataCatalogWindow::populateDetails(const OpenDataSample* sample)
{
    if (!sample) {
        clearDetails();
        return;
    }

    _overviewLabel->setText(
        tr("<b>%1</b><br>Type: %2<br>Scans: %3<br>Volumes: %4<br>Segments: %5<br>"
           "TIFXYZ segments: %6<br>Ink detections: %7<br><br>%8")
            .arg(qstr(sample->id).toHtmlEscaped())
            .arg(qstr(sample->type).toHtmlEscaped())
            .arg(sample->scanCount())
            .arg(sample->volumeCount())
            .arg(sample->segmentCount())
            .arg(sample->tifxyzSegmentCount())
            .arg(sample->inkDetectionSegmentCount())
            .arg(qstr(sample->description).toHtmlEscaped()));

    _scansTable->setRowCount(static_cast<int>(sample->scans.size()));
    for (int row = 0; row < static_cast<int>(sample->scans.size()); ++row) {
        const auto& scan = sample->scans[static_cast<std::size_t>(row)];
        _scansTable->setItem(row, 0, item(qstr(scan.id)));
        _scansTable->setItem(row, 1, item(qstr(scan.suffix)));
        _scansTable->setItem(row, 2, item(qstr(scan.createdAt)));
        _scansTable->setItem(row, 3, numericItem(scan.artifacts.size()));
    }
    _scansTable->resizeColumnsToContents();

    _volumesTable->setRowCount(static_cast<int>(sample->volumes.size()));
    for (int row = 0; row < static_cast<int>(sample->volumes.size()); ++row) {
        const auto& volume = sample->volumes[static_cast<std::size_t>(row)];
        auto* idItem = item(qstr(volume.id));
        idItem->setData(Qt::UserRole, row);
        _volumesTable->setItem(row, 0, idItem);
        _volumesTable->setItem(row, 1, item(qstr(volume.scanId)));
        _volumesTable->setItem(row, 2, item(qstr(volume.suffix)));
        _volumesTable->setItem(row, 3, item(optionalNumber(volume.pixelSizeUm)));
        _volumesTable->setItem(row, 4, item(optionalNumber(volume.energyKeV, 'f', 1)));
        _volumesTable->setItem(row, 5, item(optionalNumber(volume.detectorDistanceMm, 'f', 1)));
        _volumesTable->setItem(row, 6, item(qstr(volume.dataFormat)));
        _volumesTable->setItem(row, 7, item(qstr(volume.createdAt)));
    }
    _volumesTable->resizeColumnsToContents();

    _segmentsTable->setRowCount(static_cast<int>(sample->segments.size()));
    for (int row = 0; row < static_cast<int>(sample->segments.size()); ++row) {
        const auto& segment = sample->segments[static_cast<std::size_t>(row)];
        auto* idItem = item(qstr(segment.id));
        idItem->setData(Qt::UserRole, row);
        _segmentsTable->setItem(row, 0, idItem);
        _segmentsTable->setItem(row, 1, item(qstr(segment.suffix)));
        _segmentsTable->setItem(row, 2, item(qstr(segment.originalVolumeId)));
        _segmentsTable->setItem(row, 3, item(optionalInt(segment.width)));
        _segmentsTable->setItem(row, 4, item(optionalInt(segment.height)));
        _segmentsTable->setItem(row, 5, item(yesNo(segment.hasTifxyz())));
        _segmentsTable->setItem(row, 6, item(yesNo(segment.hasInkDetection())));
        _segmentsTable->setItem(row, 7, item(yesNo(segment.hasLayersZarr())));
        const auto state = cacheStateForSegment(
            std::filesystem::path(vc3d::remoteCachePath().toStdString()),
            *sample,
            segment);
        _segmentsTable->setItem(row, 8, item(QString::fromLatin1(cacheStateName(state))));
        _segmentsTable->setItem(row, 9, item(qstr(segment.createdAt)));
    }
    _segmentsTable->resizeColumnsToContents();
    updateActionButtons();
}

void OpenDataCatalogWindow::clearDetails()
{
    if (_overviewLabel) {
        _overviewLabel->setText(tr("No sample selected."));
    }
    if (_scansTable) {
        _scansTable->setRowCount(0);
    }
    if (_volumesTable) {
        _volumesTable->setRowCount(0);
    }
    if (_segmentsTable) {
        _segmentsTable->setRowCount(0);
    }
    updateActionButtons();
}

const OpenDataSample* OpenDataCatalogWindow::selectedSample() const
{
    if (!_manifest || !_sampleTable || !_sampleTable->currentItem()) {
        return nullptr;
    }
    const int row = _sampleTable->currentRow();
    const auto* idItem = _sampleTable->item(row, 0);
    if (!idItem) {
        return nullptr;
    }
    const auto sampleIndex = idItem->data(Qt::UserRole).toULongLong();
    if (sampleIndex >= _manifest->samples.size()) {
        return nullptr;
    }
    return &_manifest->samples[static_cast<std::size_t>(sampleIndex)];
}

const OpenDataVolume* OpenDataCatalogWindow::selectedVolume() const
{
    const auto* sample = selectedSample();
    if (!sample || !_volumesTable || _volumesTable->currentRow() < 0) {
        return nullptr;
    }
    const int row = _volumesTable->currentRow();
    const auto* idItem = _volumesTable->item(row, 0);
    if (!idItem) {
        return nullptr;
    }
    const int volumeIndex = idItem->data(Qt::UserRole).toInt();
    if (volumeIndex < 0 || volumeIndex >= static_cast<int>(sample->volumes.size())) {
        return nullptr;
    }
    return &sample->volumes[static_cast<std::size_t>(volumeIndex)];
}

const OpenDataSegment* OpenDataCatalogWindow::selectedSegment() const
{
    const auto* sample = selectedSample();
    if (!sample || !_segmentsTable || _segmentsTable->currentRow() < 0) {
        return nullptr;
    }
    const int row = _segmentsTable->currentRow();
    const auto* idItem = _segmentsTable->item(row, 0);
    if (!idItem) {
        return nullptr;
    }
    const int segmentIndex = idItem->data(Qt::UserRole).toInt();
    if (segmentIndex < 0 || segmentIndex >= static_cast<int>(sample->segments.size())) {
        return nullptr;
    }
    return &sample->segments[static_cast<std::size_t>(segmentIndex)];
}

QString OpenDataCatalogWindow::selectedVolumeUrl() const
{
    const auto* volume = selectedVolume();
    if (!volume) {
        return {};
    }
    const auto* artifact = preferredVolumeArtifact(*volume);
    if (!artifact) {
        return {};
    }
    return qstr(artifact->resolvedUrl.empty() ? artifact->sourcePath : artifact->resolvedUrl);
}

QString OpenDataCatalogWindow::selectedSegmentUrl() const
{
    const auto* segment = selectedSegment();
    if (!segment) {
        return {};
    }
    const auto* artifact = preferredTifxyzArtifact(*segment);
    if (!artifact) {
        return firstArtifactUrl(segment->artifacts);
    }
    return artifactUrl(*artifact);
}

std::filesystem::path OpenDataCatalogWindow::selectedSegmentCacheDir() const
{
    const auto* sample = selectedSample();
    const auto* segment = selectedSegment();
    if (!sample || !segment) {
        return {};
    }
    return openDataSegmentCacheDirectory(
        std::filesystem::path(vc3d::remoteCachePath().toStdString()),
        *sample,
        *segment);
}

void OpenDataCatalogWindow::updateActionButtons()
{
    const bool hasVolumeUrl = !selectedVolumeUrl().isEmpty();
    const bool hasSegmentUrl = !selectedSegmentUrl().isEmpty();
    const auto* segment = selectedSegment();
    const bool canCacheSegment = segment && segment->hasTifxyz();
    const auto cacheDir = selectedSegmentCacheDir();
    const bool hasCacheDir = !cacheDir.empty() && std::filesystem::is_directory(cacheDir);
    const bool canOpenSample = selectedSample() != nullptr &&
                               static_cast<bool>(_openSampleHandler) &&
                               !(_fetchWatcher && _fetchWatcher->isRunning() && !_manifest);
    if (_openSampleButton) {
        _openSampleButton->setEnabled(canOpenSample);
    }
    if (_copyVolumeUrlButton) {
        _copyVolumeUrlButton->setEnabled(hasVolumeUrl);
    }
    if (_openVolumeUrlButton) {
        _openVolumeUrlButton->setEnabled(hasVolumeUrl);
    }
    if (_cacheSegmentButton) {
        _cacheSegmentButton->setEnabled(canCacheSegment);
    }
    if (_openSegmentCacheFolderButton) {
        _openSegmentCacheFolderButton->setEnabled(hasCacheDir);
    }
    if (_copySegmentUrlButton) {
        _copySegmentUrlButton->setEnabled(hasSegmentUrl);
    }
    if (_openSegmentUrlButton) {
        _openSegmentUrlButton->setEnabled(hasSegmentUrl);
    }
}

void OpenDataCatalogWindow::openSelectedSample()
{
    const auto* sample = selectedSample();
    if (!sample || !_openSampleHandler) {
        return;
    }
    _openSampleHandler(*sample);
}

void OpenDataCatalogWindow::copySelectedVolumeUrl()
{
    const QString url = selectedVolumeUrl();
    if (!url.isEmpty() && QGuiApplication::clipboard()) {
        QGuiApplication::clipboard()->setText(url);
        setStatus(tr("Copied volume URL."));
    }
}

void OpenDataCatalogWindow::openSelectedVolumeUrl()
{
    const QString url = selectedVolumeUrl();
    if (!url.isEmpty()) {
        QDesktopServices::openUrl(QUrl(url));
    }
}

void OpenDataCatalogWindow::copySelectedSegmentUrl()
{
    const QString url = selectedSegmentUrl();
    if (!url.isEmpty() && QGuiApplication::clipboard()) {
        QGuiApplication::clipboard()->setText(url);
        setStatus(tr("Copied segment source URL."));
    }
}

void OpenDataCatalogWindow::openSelectedSegmentUrl()
{
    const QString url = selectedSegmentUrl();
    if (!url.isEmpty()) {
        QDesktopServices::openUrl(QUrl(url));
    }
}

void OpenDataCatalogWindow::cacheSelectedSegment()
{
    const auto* sample = selectedSample();
    const auto* segment = selectedSegment();
    if (!sample || !segment || !segment->hasTifxyz()) {
        return;
    }

    setStatus(tr("Caching segment %1...").arg(qstr(segment->id)));
    OpenDataSample oneSegmentSample;
    oneSegmentSample.id = sample->id;
    oneSegmentSample.type = sample->type;
    oneSegmentSample.description = sample->description;
    oneSegmentSample.properties = sample->properties;
    oneSegmentSample.segments.push_back(*segment);

    auto pkg = VolumePkg::newEmpty();
    const std::filesystem::path remoteRoot(vc3d::remoteCachePath().toStdString());
    auto result = reconcileOpenDataSampleSegments(
        *pkg,
        oneSegmentSample,
        remoteRoot,
        [&](const OpenDataSampleDownloadProgress& progress) {
            if (!progress.segmentId.empty()) {
                setStatus(tr("Caching %1: %2/%3 files")
                              .arg(qstr(progress.segmentId))
                              .arg(progress.completedFiles)
                              .arg(progress.totalFiles));
            }
        });

    populateDetails(sample);
    QString message = tr("Cached %1 of %2 selected tifxyz segment(s).")
                          .arg(result.cachedTifxyzSegments)
                          .arg(result.supportedTifxyzSegments);
    if (result.failedTifxyzSegments > 0 && !result.messages.empty()) {
        message += tr(" %1").arg(qstr(result.messages.front()));
    }
    setStatus(message);
}

void OpenDataCatalogWindow::openSelectedSegmentCacheFolder()
{
    const auto dir = selectedSegmentCacheDir();
    if (!dir.empty() && std::filesystem::is_directory(dir)) {
        QDesktopServices::openUrl(QUrl::fromLocalFile(QString::fromStdString(dir.string())));
    }
}

std::filesystem::path OpenDataCatalogWindow::cacheRoot() const
{
    QString base = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
    if (base.isEmpty()) {
        base = QDir::home().filePath(QStringLiteral(".VC3D"));
    }
    return std::filesystem::path(base.toStdString()) / "open-data-catalog";
}

std::filesystem::path OpenDataCatalogWindow::cachedManifestPath() const
{
    return cacheRoot() / "metadata.json";
}

std::filesystem::path OpenDataCatalogWindow::cacheMetadataPath() const
{
    return cacheRoot() / "metadata.cache.json";
}

} // namespace vc3d::opendata
