#include "OpenDataCatalogWindow.hpp"

#include "OpenDataLasagna.hpp"
#include "OpenDataNormalGrids.hpp"
#include "ProjectCreationDefaults.hpp"
#include "VCSettings.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/HttpFetch.hpp"

#include <nlohmann/json.hpp>

#include <QCheckBox>
#include <QClipboard>
#include <QColor>
#include <QCoreApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QMessageBox>
#include <QTreeWidget>
#include <QGuiApplication>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QImage>
#include <QIODevice>
#include <QItemSelectionModel>
#include <QLabel>
#include <QLineEdit>
#include <QMetaObject>
#include <QPixmap>
#include <QPointer>
#include <QProgressDialog>
#include <QPushButton>
#include <QResizeEvent>
#include <QSaveFile>
#include <QSettings>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSplitter>
#include <QStringList>
#include <QStandardPaths>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTabWidget>
#include <QPalette>
#include <QUrl>
#include <QVBoxLayout>
#include <QtConcurrent>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cmath>
#include <memory>
#include <exception>
#include <limits>
#include <utility>

namespace vc3d::opendata {
namespace {

constexpr int kOverviewPhotoMaxHeight = 360;
constexpr int kOverviewPhotoDefaultWidth = 420;
constexpr int kCatalogDefaultWidth = 1520;
constexpr int kCatalogDefaultHeight = 760;
constexpr int kSampleTableInitialWidth = 940;
constexpr int kDetailsInitialWidth = 520;
constexpr int kRepresentationVolumeIndexRole = Qt::UserRole;
constexpr int kRepresentationArtifactIndexRole = Qt::UserRole + 1;

class OverviewPhotoLabel : public QLabel {
public:
    using QLabel::QLabel;

    void setSourceImage(QImage image)
    {
        _sourceImage = std::move(image);
        updateScaledPixmap();
    }

    void clearSourceImage()
    {
        _sourceImage = {};
        clear();
    }

    [[nodiscard]] bool hasHeightForWidth() const override
    {
        return true;
    }

    [[nodiscard]] int heightForWidth(int width) const override
    {
        if (_sourceImage.isNull() || width <= 0 || _sourceImage.width() <= 0) {
            return QLabel::heightForWidth(width);
        }
        const int scaledHeight = static_cast<int>(
            std::ceil(static_cast<double>(_sourceImage.height()) *
                      static_cast<double>(width) /
                      static_cast<double>(_sourceImage.width())));
        return std::min(scaledHeight, kOverviewPhotoMaxHeight);
    }

    [[nodiscard]] QSize sizeHint() const override
    {
        if (_sourceImage.isNull()) {
            return QLabel::sizeHint();
        }
        const int width = std::min(_sourceImage.width(), kOverviewPhotoDefaultWidth);
        return {width, heightForWidth(width)};
    }

    [[nodiscard]] QSize minimumSizeHint() const override
    {
        return {0, 0};
    }

protected:
    void resizeEvent(QResizeEvent* event) override
    {
        QLabel::resizeEvent(event);
        updateScaledPixmap();
    }

private:
    void updateScaledPixmap()
    {
        if (_sourceImage.isNull()) {
            return;
        }
        const int targetWidth = std::max(1, width());
        const QImage scaled = _sourceImage.scaled(
            QSize(targetWidth, kOverviewPhotoMaxHeight),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation);
        QLabel::setPixmap(QPixmap::fromImage(scaled));
        setMinimumHeight(scaled.height());
    }

    QImage _sourceImage;
};

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

QString formatByteSize(std::uint64_t bytes)
{
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    if (bytes >= static_cast<std::uint64_t>(kGiB)) {
        return QStringLiteral("%1 GB").arg(bytes / kGiB, 0, 'f', 1);
    }
    if (bytes >= static_cast<std::uint64_t>(kMiB)) {
        return QStringLiteral("%1 MB").arg(bytes / kMiB, 0, 'f', 1);
    }
    return QStringLiteral("%1 KB").arg(bytes / 1024.0, 0, 'f', 1);
}

QString normalGridsStatusDisplay(const std::filesystem::path& remoteRoot,
                                 const std::string& sampleId,
                                 const vcr::catalog::OpenDataVolume& volume)
{
    const auto state = normalGridsCacheState(remoteRoot, sampleId, volume);
    if (!state.hasArtifact) {
        return QStringLiteral("—");
    }
    if (state.complete) {
        return QObject::tr("All (%1)").arg(formatByteSize(state.totalBytes));
    }
    if (state.cachedFiles > 0) {
        return QObject::tr("%1 cached (%2 files)")
            .arg(formatByteSize(state.cachedBytes))
            .arg(state.cachedFiles);
    }
    return QObject::tr("Not cached");
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

QString firstArtifactUrl(const std::vector<vcr::catalog::OpenDataArtifact>& artifacts)
{
    return artifacts.empty() ? QString() : qstr(artifacts.front().url());
}

QString volumeSourceFilename(const vcr::catalog::OpenDataVolume& volume)
{
    const auto* artifact = vcr::catalog::findArtifact(volume.artifacts, "ome-zarr");
    QStringList filenames;
    if (artifact) {
        for (const auto& origin : artifact->origins) {
            const auto parts = qstr(origin.copySourcePath).split('/', Qt::SkipEmptyParts);
            if (!parts.isEmpty() && !filenames.contains(parts.last()))
                filenames.push_back(parts.last());
        }
    }
    return filenames.isEmpty() ? QStringLiteral("—") : filenames.join(QStringLiteral("\n"));
}

QString predictionTypeCell(const vcr::catalog::OpenDataArtifact& artifact)
{
    return artifact.predictionType.empty() ? QStringLiteral("—") : qstr(artifact.predictionType);
}

// Whether the project's attach for the representation's kind takes it, at a
// public URL.
bool canAttachRepresentation(const vcr::catalog::OpenDataSample& sample,
                             const vcr::catalog::OpenDataRepresentationRef& ref)
{
    const auto& volume = sample.volumes[ref.volumeIndex];
    const auto& artifact = volume.artifacts[ref.artifactIndex];
    if (!artifact.hasResolvedUrl()) return false;
    switch (ref.kind) {
    case vcr::catalog::OpenDataRepresentationKind::NormalGrids: {
        const auto grids = normalGridsArtifacts(sample.id, volume);
        return std::any_of(grids.begin(), grids.end(),
                           [&](const auto& info) { return info.artifactIndex == ref.artifactIndex; });
    }
    case vcr::catalog::OpenDataRepresentationKind::Lasagna: {
        const auto lasagna = lasagnaArtifacts(volume);
        return std::find(lasagna.begin(), lasagna.end(), ref.artifactIndex) != lasagna.end();
    }
    case vcr::catalog::OpenDataRepresentationKind::Prediction:
        // Prediction artifacts are not necessarily volumes (e.g. meshes).
        return vcr::catalog::isZarrArtifact(artifact);
    }
    return false;
}

QString representationCoordinates(const vcr::catalog::OpenDataArtifact& artifact,
                                  vcr::catalog::OpenDataRepresentationKind kind)
{
    if (artifact.sourceCoordinateLevel)
        return QStringLiteral("L%1").arg(*artifact.sourceCoordinateLevel);
    if (kind == vcr::catalog::OpenDataRepresentationKind::Lasagna)
        return QObject::tr("Base L0 (shape matched)");
    return QObject::tr("Unspecified");
}

QString representationParameters(const vcr::catalog::OpenDataArtifact& artifact)
{
    nlohmann::json details = artifact.parameters;
    if (!details.is_object()) details = nlohmann::json::object();
    if (!artifact.creationInfo.empty())
        details["creation_info"] = artifact.creationInfo;
    return details.empty() ? QStringLiteral("—") : qstr(details.dump());
}

QStringList volumeCells(const std::filesystem::path& remoteRoot,
                        const vcr::catalog::OpenDataSample& sample, const vcr::catalog::OpenDataVolume& volume)
{
    return {qstr(volume.id), qstr(volume.scanId), qstr(volume.suffix),
            optionalNumber(volume.pixelSizeUm), optionalNumber(volume.energyKeV, 'f', 1),
            optionalNumber(volume.detectorDistanceMm, 'f', 1), qstr(volume.dataFormat),
            normalGridsStatusDisplay(remoteRoot, sample.id, volume), qstr(volume.createdAt),
            volumeSourceFilename(volume)};
}

QStringList representationCells(const vcr::catalog::OpenDataVolume& volume, const vcr::catalog::OpenDataArtifact& artifact,
                                vcr::catalog::OpenDataRepresentationKind kind)
{
    return {qstr(volume.id), predictionTypeCell(artifact), qstr(artifact.type),
            qstr(artifact.modelId.value_or(std::string{})), representationCoordinates(artifact, kind),
            qstr(artifact.accessUsage), representationParameters(artifact), qstr(artifact.url()),
            artifact.sourceCoordinateLevel ? QString::number(*artifact.sourceCoordinateLevel)
                                           : QStringLiteral("—")};
}

QStringList segmentCells(const vcr::catalog::OpenDataSample& sample, const vcr::catalog::OpenDataSegment& segment)
{
    return {qstr(segment.id), qstr(segment.suffix), qstr(segment.originalVolumeId),
            optionalInt(segment.width), optionalInt(segment.height), yesNo(!sample.surfaces(segment).empty()),
            yesNo(segment.hasInkDetection()), yesNo(segment.hasLayersZarr()), qstr(segment.createdAt)};
}

void setCatalogRow(QTableWidget* table, int row, const QStringList& cells)
{
    for (int col = 0; col < cells.size(); ++col) {
        auto* cell = item(cells[col]);
        cell->setToolTip(cells[col]);
        table->setItem(row, col, cell);
    }
}

QImage imageFromBytes(const std::vector<std::byte>& bytes)
{
    if (bytes.empty() || bytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return {};
    }
    return QImage::fromData(
        reinterpret_cast<const uchar*>(bytes.data()),
        static_cast<int>(bytes.size()));
}

QString cssColor(const QColor& color)
{
    return QStringLiteral("rgba(%1, %2, %3, %4)")
        .arg(color.red())
        .arg(color.green())
        .arg(color.blue())
        .arg(color.alpha());
}

QString catalogTabStyle(const QPalette& palette)
{
    const QColor inactiveBackground = palette.color(QPalette::Button);
    const QColor inactiveText = palette.color(QPalette::ButtonText);
    const QColor selectedBackground = palette.color(QPalette::Highlight);
    const QColor selectedText = palette.color(QPalette::HighlightedText);
    const QColor border = palette.color(QPalette::Mid);

    return QStringLiteral(
        "QTabWidget#openDataCatalogTabs::pane {"
        "  border: 1px solid %1;"
        "  top: -1px;"
        "}"
        "QTabWidget#openDataCatalogTabs QTabBar::tab {"
        "  background: %2;"
        "  color: %3;"
        "  border: 1px solid %1;"
        "  border-bottom-color: %1;"
        "  padding: 6px 14px;"
        "  min-width: 86px;"
        "  margin-right: 2px;"
        "}"
        "QTabWidget#openDataCatalogTabs QTabBar::tab:!selected {"
        "  margin-top: 3px;"
        "}"
        "QTabWidget#openDataCatalogTabs QTabBar::tab:selected {"
        "  background: %4;"
        "  color: %5;"
        "  border-color: %4;"
        "  border-bottom-color: %4;"
        "  font-weight: 700;"
        "}"
        "QTabWidget#openDataCatalogTabs QTabBar::tab:hover:!selected {"
        "  background: %6;"
        "}")
        .arg(cssColor(border),
             cssColor(inactiveBackground),
             cssColor(inactiveText),
             cssColor(selectedBackground),
             cssColor(selectedText),
             cssColor(inactiveBackground.lighter(112)));
}

} // namespace

OpenDataCatalogWindow::OpenDataCatalogWindow(QWidget* parent)
    : QDialog(parent)
{
    buildUi();
    loadCachedManifestIfAvailable();
    reloadManifest();
}

OpenDataCatalogWindow::OpenDataCatalogWindow(const vcr::catalog::OpenDataManifest& manifest, QWidget* parent)
    : QDialog(parent)
{
    buildUi();
    applyManifest(manifest, tr("Provided catalog"), false);
}

OpenDataCatalogWindow::~OpenDataCatalogWindow()
{
    if (_fetchWatcher) {
        _fetchWatcher->cancel();
        _fetchWatcher->waitForFinished();
    }
    for (auto* watcher : _photoWatchers) {
        watcher->cancel();
        watcher->waitForFinished();
    }
}

void OpenDataCatalogWindow::setOpenSampleHandler(std::function<bool(const vcr::catalog::OpenDataSample&)> handler)
{
    _openSampleHandler = std::move(handler);
    updateActionButtons();
}

void OpenDataCatalogWindow::setCreateProjectHandler(CreateProjectHandler handler)
{
    _createProjectHandler = std::move(handler);
    updateActionButtons();
}

void OpenDataCatalogWindow::setAttachRepresentationHandler(AttachRepresentationHandler handler)
{
    _attachRepresentationHandler = std::move(handler);
    updateActionButtons();
}

void OpenDataCatalogWindow::setRefreshSampleHandler(RefreshSampleHandler handler)
{
    _refreshSampleHandler = std::move(handler);
    updateActionButtons();
}

void OpenDataCatalogWindow::createSelectedProject()
{
    if (_manifestRefreshPending || !selectedSample() || !_createProjectHandler) return;
    // Snapshot indices and metadata while the modal dialog is open.
    const auto sample = *selectedSample();
    const auto remoteRoot = vc3d::remoteCachePathFs();
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Create Project - %1").arg(QString::fromStdString(sample.id)));
    dialog.resize(1400, 700);
    auto* layout = new QVBoxLayout(&dialog);
    auto* form = new QFormLayout;
    auto* name = new QLineEdit(QString::fromStdString(sample.id), &dialog);
    name->setObjectName(QStringLiteral("catalogProjectName"));
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const auto defaultDirectory = vc3d::defaultNewProjectDirectory(settings);
    const auto defaultPath = QDir(defaultDirectory).filePath(
        QString::fromStdString(sample.id) + ".volpkg.json");
    auto* path = new QLineEdit(defaultPath, &dialog);
    path->setObjectName(QStringLiteral("catalogProjectPath"));
    bool nameEdited = false;
    connect(name, &QLineEdit::textEdited, &dialog, [&]() { nameEdited = true; });
    connect(path, &QLineEdit::textChanged, &dialog, [&](const QString& value) {
        if (nameEdited) return;
        auto filename = QFileInfo(value.trimmed()).fileName();
        if (filename.endsWith(".volpkg.json", Qt::CaseInsensitive))
            filename.chop(int(QStringLiteral(".volpkg.json").size()));
        name->setText(filename);
    });
    auto* browse = new QPushButton(tr("Browse..."), &dialog);
    auto* pathRow = new QHBoxLayout;
    pathRow->addWidget(path);
    pathRow->addWidget(browse);
    form->addRow(tr("Project name"), name);
    form->addRow(tr("Project JSON"), pathRow);
    layout->addLayout(form);
    auto* tabs = new QTabWidget(&dialog);
    layout->addWidget(tabs, 1);
    auto makeTree = [&](const QString& label, const QTableWidget* catalog) {
        auto* tree = new QTreeWidget(tabs);
        QStringList headers;
        for (int i = 0; i < catalog->columnCount(); ++i)
            headers.push_back(catalog->horizontalHeaderItem(i)->text());
        tree->setHeaderLabels(headers);
        tree->header()->setStretchLastSection(false);
        for (int visual = 0; visual < catalog->columnCount(); ++visual) {
            const int logical = catalog->horizontalHeader()->logicalIndex(visual);
            tree->header()->moveSection(tree->header()->visualIndex(logical), visual);
            tree->setColumnWidth(logical, std::clamp(catalog->columnWidth(logical), 90, 360));
        }
        tree->setColumnWidth(0, 260);
        tabs->addTab(tree, label);
        return tree;
    };
    auto* volumeTree = makeTree(tr("Volumes"), _volumesTable);
    auto* representationTree = makeTree(tr("Representations"), _representationsTable);
    auto group = [](QTreeWidget* tree, const QString& label) {
        auto* item = new QTreeWidgetItem(tree, {label});
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable | Qt::ItemIsAutoTristate);
        item->setCheckState(0, Qt::Unchecked);
        return item;
    };
    auto leaf = [](QTreeWidgetItem* parent, const QStringList& labels) {
        auto* item = new QTreeWidgetItem(parent, labels);
        for (int col = 0; col < labels.size(); ++col)
            item->setToolTip(col, labels[col]);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(0, Qt::Unchecked);
        return item;
    };
    std::vector<std::pair<QTreeWidgetItem*, std::string>> volumes;
    std::vector<std::pair<QTreeWidgetItem*, vcr::catalog::OpenDataRepresentationRef>> representations;
    auto* sourceGroup = group(volumeTree, tr("Source volumes"));
    for (std::size_t row = 0; row < sample.volumes.size(); ++row) {
        const auto& volume = sample.volumes[row];
        if (std::none_of(volume.artifacts.begin(), volume.artifacts.end(),
                        [](const auto& a) { return !vcr::catalog::classifyDerivedRepresentation(a) && a.hasResolvedUrl(); }))
            continue;
        volumes.emplace_back(leaf(sourceGroup, volumeCells(remoteRoot, sample, volume)), volume.id);
    }
    QMap<QString, QTreeWidgetItem*> groups;
    const auto refs = vcr::catalog::derivedRepresentations(sample);
    for (std::size_t row = 0; row < refs.size(); ++row) {
        const auto& ref = refs[row];
        const auto& volume = sample.volumes.at(ref.volumeIndex);
        const auto& artifact = volume.artifacts.at(ref.artifactIndex);
        const QString type = QString::fromUtf8(vcr::catalog::representationKindName(ref.kind).data(),
                                              int(vcr::catalog::representationKindName(ref.kind).size()));
        const QString prediction = qstr(artifact.predictionType);
        const QString label = prediction.isEmpty() ? type : type + " / " + prediction;
        if (!groups.contains(label)) groups[label] = group(representationTree, label);
        auto* item = leaf(groups[label], representationCells(volume, artifact, ref.kind));
        representations.emplace_back(item, ref);
    }
    const std::array trees{volumeTree, representationTree};
    for (auto* tree : trees) tree->expandAll();
    auto* selectionRow = new QHBoxLayout;
    for (const auto& [label, checked] : {std::pair{tr("Select all"), true},
                                       std::pair{tr("Select none"), false}}) {
        auto* button = new QPushButton(label, &dialog);
        selectionRow->addWidget(button);
        connect(button, &QPushButton::clicked, &dialog, [trees, checked]() {
            for (auto* tree : trees)
                for (int i = 0; i < tree->topLevelItemCount(); ++i)
                    tree->topLevelItem(i)->setCheckState(0, checked ? Qt::Checked : Qt::Unchecked);
        });
    }
    selectionRow->addStretch();
    layout->addLayout(selectionRow);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, &dialog);
    buttons->button(QDialogButtonBox::Save)->setText(tr("Create and Open"));
    layout->addWidget(buttons);
    connect(browse, &QPushButton::clicked, &dialog, [&]() {
        const auto destination = QFileDialog::getSaveFileName(&dialog, tr("New Project"),
            path->text().isEmpty() ? QDir(defaultDirectory).filePath(name->text() + ".volpkg.json") : path->text(),
            tr("VC3D project (*.volpkg.json)"), nullptr, QFileDialog::DontConfirmOverwrite);
        if (!destination.isEmpty()) path->setText(destination);
    });
    auto validate = [&]() {
        buttons->button(QDialogButtonBox::Save)->setEnabled(
            !name->text().trimmed().isEmpty() && !path->text().trimmed().isEmpty());
    };
    connect(name, &QLineEdit::textChanged, &dialog, validate);
    connect(path, &QLineEdit::textChanged, &dialog, validate);
    validate();
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, [&]() {
        if (!path->text().trimmed().endsWith(".volpkg.json", Qt::CaseInsensitive))
            path->setText(path->text().trimmed() + ".volpkg.json");
        if (QFileInfo::exists(path->text()) && QMessageBox::question(&dialog, tr("Overwrite Project File"),
                tr("A project file already exists at:\n%1\n\nOverwrite this file on disk?").arg(path->text()), QMessageBox::Yes | QMessageBox::No,
                QMessageBox::No) != QMessageBox::Yes) return;
        dialog.accept();
    });
    if (dialog.exec() != QDialog::Accepted) return;
    OpenDataResourceSelection selection;
    selection.rawVolumeIds.emplace();
    selection.representations.emplace();
    for (const auto& [item, id] : volumes)
        if (item->checkState(0) == Qt::Checked) selection.rawVolumeIds->push_back(id);
    for (const auto& [item, ref] : representations)
        if (item->checkState(0) == Qt::Checked) selection.representations->push_back(ref);
    const OpenDataNewProject project{QFileInfo(path->text()).absoluteFilePath().toStdString(),
                                     name->text().trimmed().toStdString()};
    if (_createProjectHandler(sample, selection, project)) accept();
}

void OpenDataCatalogWindow::buildUi()
{
    setWindowTitle(tr("Open Data Catalog"));
    resize(kCatalogDefaultWidth, kCatalogDefaultHeight);

    auto* mainLayout = new QVBoxLayout(this);

    auto* topRow = new QHBoxLayout;
    _searchEdit = new QLineEdit(this);
    _searchEdit->setPlaceholderText(tr("Filter sample ID"));
    _searchEdit->setAccessibleName(tr("Filter"));
    _segmentsOnlyCheck = new QCheckBox(tr("Segments"), this);
    _inkOnlyCheck = new QCheckBox(tr("Ink"), this);
    _refreshButton = new QPushButton(tr("Refresh"), this);
    _refreshButton->setToolTip(tr("Refresh the open-data catalog manifest from the remote source."));
    _refreshButton->setStatusTip(tr("Refreshes the open-data catalog listing, including samples, scans, volumes, representations, and segments."));
    topRow->addWidget(_searchEdit, 1);
    topRow->addWidget(_segmentsOnlyCheck);
    topRow->addWidget(_inkOnlyCheck);
    topRow->addWidget(_refreshButton);
    mainLayout->addLayout(topRow);

    auto* splitter = new QSplitter(Qt::Horizontal, this);
    _sampleTable = new QTableWidget(splitter);
    _sampleTable->setColumnCount(5);
    _sampleTable->setHorizontalHeaderLabels({
        tr("Sample ID"),
        tr("Scans"),
        tr("Volumes"),
        tr("Segments"),
        tr("Ink")
    });
    _sampleTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _sampleTable->setSelectionMode(QAbstractItemView::SingleSelection);
    _sampleTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _sampleTable->setSortingEnabled(true);
    _sampleTable->setMinimumWidth(kSampleTableInitialWidth);
    _sampleTable->horizontalHeader()->setSectionsClickable(true);
    _sampleTable->horizontalHeader()->setSortIndicatorShown(true);
    _sampleTable->horizontalHeader()->setSortIndicator(0, Qt::AscendingOrder);
    _sampleTable->horizontalHeader()->setToolTip(tr("Click a column header to sort."));
    _sampleTable->horizontalHeader()->setStretchLastSection(true);
    _sampleTable->verticalHeader()->hide();

    _tabs = new QTabWidget(splitter);
    _tabs->setObjectName(QStringLiteral("openDataCatalogTabs"));
    _tabs->setStyleSheet(catalogTabStyle(_tabs->palette()));
    auto* overviewPage = new QWidget(_tabs);
    auto* overviewLayout = new QVBoxLayout(overviewPage);
    _overviewLabel = new QLabel(overviewPage);
    _overviewLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _overviewLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    _overviewLabel->setWordWrap(true);
    _overviewPhotoLabel = new OverviewPhotoLabel(overviewPage);
    _overviewPhotoLabel->setAlignment(Qt::AlignCenter);
    _overviewPhotoLabel->setWordWrap(true);
    _overviewPhotoLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _overviewPhotoLabel->setMaximumHeight(kOverviewPhotoMaxHeight);
    _overviewPhotoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    _overviewPhotoLabel->hide();
    overviewLayout->addWidget(_overviewLabel, 0);
    overviewLayout->addWidget(_overviewPhotoLabel, 0, Qt::AlignLeft | Qt::AlignTop);
    overviewLayout->addStretch(1);
    _tabs->addTab(overviewPage, tr("Overview"));

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
    _volumesTable->setColumnCount(10);
    _volumesTable->setHorizontalHeaderLabels({
        tr("Volume ID"),
        tr("Scan ID"),
        tr("Suffix"),
        tr("Resolution"),
        tr("Energy (keV)"),
        tr("Detector distance"),
        tr("Format"),
        tr("Normal grids"),
        tr("Export date"),
        tr("Source filename")
    });
    _volumesTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _volumesTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _volumesTable->setSelectionMode(QAbstractItemView::SingleSelection);
    _volumesTable->horizontalHeader()->setStretchLastSection(true);
    _volumesTable->horizontalHeader()->moveSection(9, 2);
    _volumesTable->verticalHeader()->hide();
    auto* volumeActions = new QHBoxLayout;
    _downloadNormalGridsButton = new QPushButton(tr("Download Normal Grids"), volumesPage);
    _downloadNormalGridsButton->setToolTip(
        tr("Download the volume's full normal-grid store into the remote cache. "
           "Not required: normal grids stream on demand as they are used."));
    _copyVolumeUrlButton = new QPushButton(tr("Copy URL"), volumesPage);
    _attachVolumeButton = new QPushButton(tr("Attach to Current Project..."), volumesPage);
    _openVolumeUrlButton = new QPushButton(tr("Open URL"), volumesPage);
    volumeActions->addStretch(1);
    volumeActions->addWidget(_downloadNormalGridsButton);
    volumeActions->addWidget(_copyVolumeUrlButton);
    volumeActions->addWidget(_attachVolumeButton);
    volumeActions->addWidget(_openVolumeUrlButton);
    volumesLayout->addWidget(_volumesTable, 1);
    volumesLayout->addLayout(volumeActions);
    _tabs->addTab(volumesPage, tr("Volumes"));

    auto* representationsPage = new QWidget(_tabs);
    auto* representationsLayout = new QVBoxLayout(representationsPage);
    auto* representationsHelp = new QLabel(
        tr("Manifest-declared derived data for this sample. Parent volume and "
           "coordinate metadata are shown explicitly; these rows are not source volumes."),
        representationsPage);
    representationsHelp->setWordWrap(true);
    _representationsTable = new QTableWidget(representationsPage);
    _representationsTable->setObjectName("catalogRepresentations");
    _representationsTable->setColumnCount(9);
    _representationsTable->setHorizontalHeaderLabels({
        tr("Parent volume"),
        tr("Prediction type"),
        tr("Artifact type"),
        tr("Model"),
        tr("Coordinates"),
        tr("Access"),
        tr("Parameters"),
        tr("URL"),
        tr("Level")
    });
    _representationsTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _representationsTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _representationsTable->setSelectionMode(QAbstractItemView::SingleSelection);
    _representationsTable->horizontalHeader()->setStretchLastSection(true);
    _representationsTable->horizontalHeader()->moveSection(8, 3);
    _representationsTable->verticalHeader()->hide();
    auto* representationActions = new QHBoxLayout;
    _copyRepresentationUrlButton =
        new QPushButton(tr("Copy URL"), representationsPage);
    _openRepresentationUrlButton =
        new QPushButton(tr("Open URL"), representationsPage);
    _attachRepresentationButton =
        new QPushButton(tr("Attach to Current Project..."), representationsPage);
    representationActions->addStretch(1);
    representationActions->addWidget(_copyRepresentationUrlButton);
    representationActions->addWidget(_attachRepresentationButton);
    representationActions->addWidget(_openRepresentationUrlButton);
    representationsLayout->addWidget(representationsHelp);
    representationsLayout->addWidget(_representationsTable, 1);
    representationsLayout->addLayout(representationActions);
    _tabs->addTab(representationsPage, tr("Representations"));

    auto* segmentsPage = new QWidget(_tabs);
    auto* segmentsLayout = new QVBoxLayout(segmentsPage);
    _segmentsTable = new QTableWidget(segmentsPage);
    _segmentsTable->setColumnCount(9);
    _segmentsTable->setHorizontalHeaderLabels({
        tr("Segment ID"),
        tr("Suffix"),
        tr("Base volume"),
        tr("Width"),
        tr("Height"),
        tr("TIFXYZ"),
        tr("Ink"),
        tr("Layers Zarr"),
        tr("Created")
    });
    _segmentsTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _segmentsTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _segmentsTable->setSelectionMode(QAbstractItemView::SingleSelection);
    _segmentsTable->horizontalHeader()->setStretchLastSection(true);
    _segmentsTable->verticalHeader()->hide();
    auto* segmentActions = new QHBoxLayout;
    _copySegmentUrlButton = new QPushButton(tr("Copy Source URL"), segmentsPage);
    _openSegmentUrlButton = new QPushButton(tr("Open Source URL"), segmentsPage);
    segmentActions->addStretch(1);
    segmentActions->addWidget(_copySegmentUrlButton);
    segmentActions->addWidget(_openSegmentUrlButton);
    segmentsLayout->addWidget(_segmentsTable, 1);
    segmentsLayout->addLayout(segmentActions);
    _tabs->addTab(segmentsPage, tr("Segments"));

    splitter->addWidget(_sampleTable);
    splitter->addWidget(_tabs);
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 3);
    splitter->setSizes({kSampleTableInitialWidth, kDetailsInitialWidth});
    mainLayout->addWidget(splitter, 1);

    auto* bottomRow = new QHBoxLayout;
    _statusLabel = new QLabel(this);
    _statusLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _refreshSampleButton = new QPushButton(tr("Refresh from Open Data"), this);
    _refreshSampleButton->setToolTip(tr("Re-read the published catalog into this sample's surface store: new and "
                                        "republished surfaces. Locally edited surfaces are preserved."));
    _openSampleButton = new QPushButton(tr("Open Sample"), this);
    _createProjectButton = new QPushButton(tr("Create Project..."), this);
    _createProjectButton->setToolTip(tr("Choose resources for a new project"));
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    _doNotShowOnNextOpenCheck = new QCheckBox(tr("Do not show on next open"), this);
    _doNotShowOnNextOpenCheck->setChecked(
        !settings.value(vc3d::settings::project::SHOW_OPEN_DATA_CATALOG_ON_STARTUP,
                        vc3d::settings::project::SHOW_OPEN_DATA_CATALOG_ON_STARTUP_DEFAULT).toBool());
    auto* closeButton = new QPushButton(tr("Close"), this);
    bottomRow->addWidget(_statusLabel, 1);
    bottomRow->addWidget(_doNotShowOnNextOpenCheck);
    bottomRow->addWidget(_refreshSampleButton);
    bottomRow->addWidget(_openSampleButton);
    bottomRow->addWidget(_createProjectButton);
    bottomRow->addWidget(closeButton);
    mainLayout->addLayout(bottomRow);

    connect(_refreshButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::reloadManifest);
    connect(_searchEdit, &QLineEdit::textChanged, this, &OpenDataCatalogWindow::updateSampleFilter);
    connect(_segmentsOnlyCheck, &QCheckBox::toggled, this, &OpenDataCatalogWindow::updateSampleFilter);
    connect(_inkOnlyCheck, &QCheckBox::toggled, this, &OpenDataCatalogWindow::updateSampleFilter);
    connect(_sampleTable, &QTableWidget::cellDoubleClicked,
            this, [this](int row, int column) {
                _sampleTable->setCurrentCell(row, std::max(column, 0));
                openSelectedSample();
            });
    connect(_sampleTable->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &OpenDataCatalogWindow::updateSelectedSample);
    connect(_volumesTable->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &OpenDataCatalogWindow::updateActionButtons);
    connect(_representationsTable->selectionModel(),
            &QItemSelectionModel::selectionChanged,
            this,
            &OpenDataCatalogWindow::updateActionButtons);
    connect(_representationsTable,
            &QTableWidget::cellDoubleClicked,
            this,
            [this](int row, int column) {
                _representationsTable->setCurrentCell(row, std::max(column, 0));
                openSelectedRepresentationUrl();
            });
    connect(_segmentsTable->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &OpenDataCatalogWindow::updateActionButtons);
    connect(_copyVolumeUrlButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::copySelectedVolumeUrl);
    connect(_attachVolumeButton, &QPushButton::clicked, this, [this]() {
        const QString url = selectedVolumeUrl();
        if (!url.isEmpty()) emit attachVolumeRequested(url);
    });
    connect(_attachRepresentationButton, &QPushButton::clicked, this, [this]() {
        const auto* sample = selectedSample();
        const auto ref = selectedRepresentation();
        if (_manifestRefreshPending || !sample || !ref || !_attachRepresentationHandler ||
            !canAttachRepresentation(*sample, *ref))
            return;
        // The full sample gives the attach its catalog context (source volumes,
        // shapes, levels); the handler attaches only this representation.
        const auto sampleCopy = *sample;
        _attachRepresentationHandler(sampleCopy, *ref);
    });
    _attachRepresentationButton->setObjectName("attachCatalogRepresentation");
    connect(_openVolumeUrlButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::openSelectedVolumeUrl);
    connect(_copyRepresentationUrlButton,
            &QPushButton::clicked,
            this,
            &OpenDataCatalogWindow::copySelectedRepresentationUrl);
    connect(_openRepresentationUrlButton,
            &QPushButton::clicked,
            this,
            &OpenDataCatalogWindow::openSelectedRepresentationUrl);
    connect(_downloadNormalGridsButton, &QPushButton::clicked,
            this, &OpenDataCatalogWindow::downloadSelectedVolumeNormalGrids);
    connect(_copySegmentUrlButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::copySelectedSegmentUrl);
    connect(_openSegmentUrlButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::openSelectedSegmentUrl);
    connect(_refreshSampleButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::refreshSelectedSample);
    connect(_openSampleButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::openSelectedSample);
    connect(_createProjectButton, &QPushButton::clicked, this, &OpenDataCatalogWindow::createSelectedProject);
    connect(_doNotShowOnNextOpenCheck, &QCheckBox::toggled, this, [](bool checked) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue(vc3d::settings::project::SHOW_OPEN_DATA_CATALOG_ON_STARTUP,
                          checked ? QStringLiteral("0") : QStringLiteral("1"));
    });
    connect(closeButton, &QPushButton::clicked, this, &QDialog::accept);

    clearDetails();
    updateActionButtons();
}

void OpenDataCatalogWindow::reloadManifest()
{
    if (_manifestRefreshPending) {
        return;
    }

    _manifestRefreshPending = true;
    setLoading(true);

    _fetchWatcher = new QFutureWatcher<ManifestLoadResult>(this);
    connect(_fetchWatcher, &QFutureWatcher<ManifestLoadResult>::finished,
            this, &OpenDataCatalogWindow::onFetchFinished);

    const auto manifestUrl = std::string(vcr::catalog::kDefaultManifestUrl);
    _fetchWatcher->setFuture(QtConcurrent::run([manifestUrl]() {
        ManifestLoadResult result;
        result.sourceLabel = QString::fromStdString(manifestUrl);
        try {
            auto loaded = vcr::catalog::loadCatalog(manifestUrl,
                {cachedOpenDataManifestPath(), true});
            result.manifest = std::move(loaded.manifest);
            if (loaded.fromCache) result.sourceLabel += QStringLiteral(" (cached)");
            if (!loaded.warning.empty()) result.sourceLabel += QStringLiteral(": ") + QString::fromStdString(loaded.warning);
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
    _manifestRefreshPending = false;
    setLoading(false);
    if (!result.error.isEmpty()) {
        if (_manifest) {
            setStatus(tr("Using cached catalog. Live refresh failed: %1").arg(result.error));
        } else {
            setStatus(tr("Live catalog refresh failed: %1").arg(result.error));
        }
        return;
    }

    applyManifest(result.manifest, result.sourceLabel, false);
}

void OpenDataCatalogWindow::loadCachedManifestIfAvailable()
{
    const auto path = cachedOpenDataManifestPath();
    if (!std::filesystem::is_regular_file(path)) {
        return;
    }
    if (const auto manifest = cachedCatalog())
        applyManifest(*manifest, qstr(path.string()), true);
    else
        setStatus(tr("Could not parse cached open-data manifest %1").arg(qstr(path.string())));
}

void OpenDataCatalogWindow::applyManifest(vcr::catalog::OpenDataManifest manifest, QString sourceLabel, bool fromCache)
{
    _manifest = std::move(manifest);
    populateSamples();
    setStatus(tr("%1 catalog: %2 samples. Source: %3")
                  .arg(fromCache ? tr("Cached") : tr("Live"))
                  .arg(_manifest->samples.size())
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
    if (loading) {
        setStatus(_manifest
            ? tr("Refreshing the live open-data catalog; cached data is preview-only...")
            : tr("Fetching open-data catalog..."));
    }
    updateActionButtons();
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
        if (_inkOnlyCheck->isChecked() && sample.inkDetectionSegmentCount() == 0) {
            continue;
        }
        _visibleSampleIndexes.push_back(i);
    }

    QSignalBlocker blocker(_sampleTable);
    QSignalBlocker selectionBlocker(_sampleTable->selectionModel());
    _sampleTable->setSortingEnabled(false);
    _sampleTable->setRowCount(static_cast<int>(_visibleSampleIndexes.size()));
    for (int row = 0; row < static_cast<int>(_visibleSampleIndexes.size()); ++row) {
        const auto sampleIndex = _visibleSampleIndexes[static_cast<std::size_t>(row)];
        const auto& sample = _manifest->samples[sampleIndex];
        auto* idItem = item(qstr(sample.id));
        idItem->setData(Qt::UserRole, QVariant::fromValue<qulonglong>(sampleIndex));
        _sampleTable->setItem(row, 0, idItem);
        _sampleTable->setItem(row, 1, numericItem(sample.scanCount()));
        _sampleTable->setItem(row, 2, numericItem(sample.volumeCount()));
        _sampleTable->setItem(row, 3, numericItem(sample.segmentCount()));
        _sampleTable->setItem(row, 4, numericItem(sample.inkDetectionSegmentCount()));
    }
    _sampleTable->setSortingEnabled(true);
    _sampleTable->resizeColumnsToContents();

    if (_sampleTable->rowCount() > 0) {
        _sampleTable->selectRow(0);
        // Filtering can retain row 0 without emitting a new selection signal.
        updateSelectedSample();
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

void OpenDataCatalogWindow::populateDetails(const vcr::catalog::OpenDataSample* sample)
{
    if (!sample) {
        clearDetails();
        return;
    }

    const auto remoteRoot = vc3d::remoteCachePathFs();
    const auto representations = vcr::catalog::derivedRepresentations(*sample);
    _overviewLabel->setText(
        tr("<b>%1</b><br>Type: %2<br>Scans: %3<br>Volumes: %4<br>Segments: %5<br>"
           "Representations: %6<br>TIFXYZ segments: %7<br>Ink detections: %8<br>Photos: %9<br><br>%10")
            .arg(qstr(sample->id).toHtmlEscaped())
            .arg(qstr(sample->type).toHtmlEscaped())
            .arg(sample->scanCount())
            .arg(sample->volumeCount())
            .arg(sample->segmentCount())
            .arg(representations.size())
            .arg(sample->surfaceSegmentCount())
            .arg(sample->inkDetectionSegmentCount())
            .arg(sample->artifacts.size())
            .arg(qstr(sample->description).toHtmlEscaped()));
    loadOverviewPhoto(*sample);

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
        setCatalogRow(_volumesTable, row, volumeCells(remoteRoot, *sample, volume));
        _volumesTable->item(row, 0)->setData(Qt::UserRole, row);
    }
    _volumesTable->resizeColumnsToContents();
    _volumesTable->setColumnWidth(9, std::min(_volumesTable->columnWidth(9), 420));

    _representationsTable->setRowCount(static_cast<int>(representations.size()));
    for (int row = 0; row < static_cast<int>(representations.size()); ++row) {
        const auto& ref = representations[static_cast<std::size_t>(row)];
        const auto& volume = sample->volumes[ref.volumeIndex];
        const auto& artifact = volume.artifacts[ref.artifactIndex];
        setCatalogRow(_representationsTable, row, representationCells(volume, artifact, ref.kind));
        auto* volumeItem = _representationsTable->item(row, 0);
        volumeItem->setData(kRepresentationVolumeIndexRole,
                            static_cast<qulonglong>(ref.volumeIndex));
        volumeItem->setData(kRepresentationArtifactIndexRole,
                            static_cast<qulonglong>(ref.artifactIndex));
    }
    _representationsTable->resizeColumnsToContents();
    _representationsTable->setColumnWidth(
        2, std::min(_representationsTable->columnWidth(2), 240));
    _representationsTable->setColumnWidth(
        6, std::min(_representationsTable->columnWidth(6), 320));
    _representationsTable->setColumnWidth(
        7, std::min(_representationsTable->columnWidth(7), 360));

    _segmentsTable->setRowCount(static_cast<int>(sample->segments.size()));
    for (int row = 0; row < static_cast<int>(sample->segments.size()); ++row) {
        const auto& segment = sample->segments[static_cast<std::size_t>(row)];
        setCatalogRow(_segmentsTable, row, segmentCells(*sample, segment));
        _segmentsTable->item(row, 0)->setData(Qt::UserRole, row);
    }
    _segmentsTable->resizeColumnsToContents();
    updateActionButtons();
}

void OpenDataCatalogWindow::clearDetails()
{
    if (_overviewLabel) {
        _overviewLabel->setText(tr("No sample selected."));
    }
    _pendingPhotoSampleId.clear();
    for (auto* watcher : _photoWatchers) {
        if (watcher && watcher->isRunning()) {
            watcher->cancel();
        }
    }
    if (_overviewPhotoLabel) {
        static_cast<OverviewPhotoLabel*>(_overviewPhotoLabel)->clearSourceImage();
        _overviewPhotoLabel->hide();
    }
    if (_scansTable) {
        _scansTable->setRowCount(0);
    }
    if (_volumesTable) {
        _volumesTable->setRowCount(0);
    }
    if (_representationsTable) {
        _representationsTable->setRowCount(0);
    }
    if (_segmentsTable) {
        _segmentsTable->setRowCount(0);
    }
    updateActionButtons();
}

void OpenDataCatalogWindow::loadOverviewPhoto(const vcr::catalog::OpenDataSample& sample)
{
    _pendingPhotoSampleId = qstr(sample.id);
    const auto* artifact = vcr::catalog::preferredPhotoArtifact(sample);
    if (!artifact) {
        if (_overviewPhotoLabel) {
            static_cast<OverviewPhotoLabel*>(_overviewPhotoLabel)->clearSourceImage();
            _overviewPhotoLabel->hide();
        }
        return;
    }

    const QString url = qstr(artifact->url());
    if (url.isEmpty()) {
        if (_overviewPhotoLabel) {
            static_cast<OverviewPhotoLabel*>(_overviewPhotoLabel)->clearSourceImage();
            _overviewPhotoLabel->hide();
        }
        return;
    }

    for (auto* activeWatcher : _photoWatchers) {
        if (activeWatcher && activeWatcher->isRunning()) {
            activeWatcher->cancel();
        }
    }
    setOverviewPhotoText(tr("Loading photo..."));

    auto* watcher = new QFutureWatcher<PhotoLoadResult>(this);
    _photoWatchers.push_back(watcher);
    connect(watcher, &QFutureWatcher<PhotoLoadResult>::finished, this, [this, watcher]() {
        onOverviewPhotoFinished(watcher);
    });

    const QString sampleId = _pendingPhotoSampleId;
    watcher->setFuture(QtConcurrent::run([sampleId, url]() {
        PhotoLoadResult result;
        result.sampleId = sampleId;
        result.url = url;
        try {
            result.image = imageFromBytes(vc::httpGetBytes(url.toStdString()));
            if (result.image.isNull()) {
                result.error = QStringLiteral("Could not decode photo.");
            }
        } catch (const std::exception& e) {
            result.error = QString::fromUtf8(e.what());
        } catch (...) {
            result.error = QStringLiteral("Unknown error while fetching photo.");
        }
        return result;
    }));
}

void OpenDataCatalogWindow::onOverviewPhotoFinished(QFutureWatcher<PhotoLoadResult>* watcher)
{
    if (!watcher) {
        return;
    }
    _photoWatchers.erase(
        std::remove(_photoWatchers.begin(), _photoWatchers.end(), watcher),
        _photoWatchers.end());
    const auto future = watcher->future();
    if (future.resultCount() == 0) {
        watcher->deleteLater();
        return;
    }
    const PhotoLoadResult result = future.result();
    watcher->deleteLater();

    if (result.sampleId != _pendingPhotoSampleId) {
        return;
    }
    if (!result.error.isEmpty()) {
        setOverviewPhotoText(tr("Photo unavailable"));
        return;
    }
    setOverviewPhotoImage(result.image);
}

void OpenDataCatalogWindow::setOverviewPhotoText(const QString& text)
{
    if (!_overviewPhotoLabel) {
        return;
    }
    static_cast<OverviewPhotoLabel*>(_overviewPhotoLabel)->clearSourceImage();
    _overviewPhotoLabel->setText(text);
    _overviewPhotoLabel->show();
}

void OpenDataCatalogWindow::setOverviewPhotoImage(const QImage& image)
{
    if (!_overviewPhotoLabel || image.isNull()) {
        return;
    }
    _overviewPhotoLabel->setText({});
    static_cast<OverviewPhotoLabel*>(_overviewPhotoLabel)->setSourceImage(image);
    _overviewPhotoLabel->show();
}

const vcr::catalog::OpenDataSample* OpenDataCatalogWindow::selectedSample() const
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

const vcr::catalog::OpenDataVolume* OpenDataCatalogWindow::selectedVolume() const
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

std::optional<vcr::catalog::OpenDataRepresentationRef> OpenDataCatalogWindow::selectedRepresentation() const
{
    const auto* sample = selectedSample();
    if (!sample || !_representationsTable || _representationsTable->currentRow() < 0) {
        return std::nullopt;
    }
    const auto* rowItem = _representationsTable->item(_representationsTable->currentRow(), 0);
    if (!rowItem) return std::nullopt;
    const auto volumeIndex = static_cast<std::size_t>(rowItem->data(kRepresentationVolumeIndexRole).toULongLong());
    const auto artifactIndex = static_cast<std::size_t>(rowItem->data(kRepresentationArtifactIndexRole).toULongLong());
    if (volumeIndex >= sample->volumes.size() || artifactIndex >= sample->volumes[volumeIndex].artifacts.size()) {
        return std::nullopt;
    }
    const auto kind =
        vcr::catalog::classifyDerivedRepresentation(sample->volumes[volumeIndex].artifacts[artifactIndex]);
    if (!kind) return std::nullopt;
    return vcr::catalog::OpenDataRepresentationRef{volumeIndex, artifactIndex, *kind};
}

const vcr::catalog::OpenDataArtifact* OpenDataCatalogWindow::selectedRepresentationArtifact() const
{
    const auto ref = selectedRepresentation();
    return ref ? &selectedSample()->volumes[ref->volumeIndex].artifacts[ref->artifactIndex] : nullptr;
}

const vcr::catalog::OpenDataSegment* OpenDataCatalogWindow::selectedSegment() const
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
    const auto* artifact = vcr::catalog::preferredVolumeArtifact(*volume);
    if (!artifact) {
        return {};
    }
    return qstr(artifact->url());
}

QString OpenDataCatalogWindow::selectedRepresentationUrl() const
{
    const auto* artifact = selectedRepresentationArtifact();
    return artifact ? qstr(artifact->url()) : QString{};
}

QString OpenDataCatalogWindow::selectedSegmentUrl() const
{
    const auto* sample = selectedSample();
    const auto* segment = selectedSegment();
    if (!sample || !segment) {
        return {};
    }
    // The canonical surface: the representation opening the sample imports.
    const auto surfaces = sample->surfaces(*segment);
    if (surfaces.empty()) {
        return firstArtifactUrl(segment->artifacts);
    }
    return qstr(surfaces.front().artifact->url());
}

void OpenDataCatalogWindow::updateActionButtons()
{
    const bool manifestReady = !_manifestRefreshPending;
    const bool hasVolumeUrl = !selectedVolumeUrl().isEmpty();
    const bool hasRepresentationUrl = !selectedRepresentationUrl().isEmpty();
    if (_attachRepresentationButton) {
        const auto* sample = selectedSample();
        const auto representation = selectedRepresentation();
        _attachRepresentationButton->setEnabled(manifestReady && sample && representation &&
                                                bool(_attachRepresentationHandler) &&
                                                canAttachRepresentation(*sample, *representation));
    }
    const bool hasSegmentUrl = !selectedSegmentUrl().isEmpty();
    const auto* sample = selectedSample();
    const auto* segment = selectedSegment();
    const auto remoteRoot = vc3d::remoteCachePathFs();
    const bool canOpenSample = sample != nullptr &&
                               static_cast<bool>(_openSampleHandler) &&
                               manifestReady;
    if (_openSampleButton) {
        _openSampleButton->setEnabled(canOpenSample);
    }
    if (_createProjectButton)
        _createProjectButton->setEnabled(manifestReady && sample && bool(_createProjectHandler));
    if (_refreshSampleButton)
        _refreshSampleButton->setEnabled(manifestReady && sample && sample->surfaceSegmentCount() > 0 &&
                                         bool(_refreshSampleHandler));
    if (_downloadNormalGridsButton) {
        const auto* volume = selectedVolume();
        bool canDownload = false;
        QString text = tr("Download Normal Grids");
        if (sample && volume) {
            const auto gridsState = normalGridsCacheState(remoteRoot, sample->id, *volume);
            if (gridsState.complete) {
                text = tr("Normal Grids Downloaded");
            } else {
                canDownload = manifestReady && gridsState.hasArtifact;
            }
        }
        _downloadNormalGridsButton->setEnabled(canDownload);
        _downloadNormalGridsButton->setText(text);
    }
    if (_copyVolumeUrlButton) {
        _copyVolumeUrlButton->setEnabled(hasVolumeUrl);
    }
    if (_attachVolumeButton) {
        _attachVolumeButton->setEnabled(manifestReady && hasVolumeUrl);
    }
    if (_openVolumeUrlButton) {
        _openVolumeUrlButton->setEnabled(hasVolumeUrl);
    }
    if (_copyRepresentationUrlButton) {
        _copyRepresentationUrlButton->setEnabled(
            manifestReady && hasRepresentationUrl);
    }
    if (_openRepresentationUrlButton) {
        _openRepresentationUrlButton->setEnabled(
            manifestReady && hasRepresentationUrl);
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
    if (_manifestRefreshPending || !sample || !_openSampleHandler) {
        return;
    }
    if (_openSampleHandler(*sample)) {
        accept();
    }
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

void OpenDataCatalogWindow::copySelectedRepresentationUrl()
{
    const QString url = selectedRepresentationUrl();
    if (!url.isEmpty() && QGuiApplication::clipboard()) {
        QGuiApplication::clipboard()->setText(url);
        setStatus(tr("Copied representation URL."));
    }
}

void OpenDataCatalogWindow::openSelectedRepresentationUrl()
{
    const QString url = selectedRepresentationUrl();
    if (!url.isEmpty()) QDesktopServices::openUrl(QUrl(url));
}

void OpenDataCatalogWindow::downloadSelectedVolumeNormalGrids()
{
    const auto* sample = selectedSample();
    const auto* volume = selectedVolume();
    if (!sample || !volume) {
        return;
    }

    const auto infos = normalGridsArtifacts(sample->id, *volume);
    if (infos.empty()) {
        return;
    }
    const auto remoteRoot = vc3d::remoteCachePathFs();
    const QString volumeLabel = qstr(volume->id);
    const int volumeRow = _volumesTable ? _volumesTable->currentRow() : -1;

    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    auto* dialog = new QProgressDialog(
        tr("Listing normal grid files for volume %1...").arg(volumeLabel),
        tr("Cancel"), 0, 0, this);
    dialog->setWindowTitle(tr("Downloading Normal Grids"));
    dialog->setWindowModality(Qt::WindowModal);
    dialog->setMinimumDuration(0);
    dialog->setAutoClose(false);
    dialog->setAutoReset(false);
    dialog->show();
    QPointer<QProgressDialog> dialogGuard(dialog);
    connect(dialog, &QProgressDialog::canceled, this, [cancelFlag]() {
        cancelFlag->store(true, std::memory_order_release);
    });

    auto progressCallback =
        [dialogGuard, volumeLabel](const NormalGridsDownloadProgress& progress) {
            if (!dialogGuard) {
                return;
            }
            QMetaObject::invokeMethod(
                dialogGuard.data(),
                [dialogGuard, volumeLabel, progress]() {
                    if (!dialogGuard) {
                        return;
                    }
                    if (progress.totalFiles <= 0) {
                        dialogGuard->setLabelText(
                            tr("Listing normal grid files for volume %1...").arg(volumeLabel));
                        return;
                    }
                    const int done = progress.completedFiles + progress.failedFiles;
                    QString label =
                        tr("Downloading normal grids for volume %1 with %2 worker(s): "
                           "%3/%4 files (%5 / %6).")
                            .arg(volumeLabel)
                            .arg(kNormalGridsDownloadWorkers)
                            .arg(done)
                            .arg(progress.totalFiles)
                            .arg(formatByteSize(progress.completedBytes),
                                 formatByteSize(progress.totalBytes));
                    if (progress.failedFiles > 0) {
                        label += tr("\nFailures: %1").arg(progress.failedFiles);
                    }
                    dialogGuard->setMaximum(progress.totalFiles);
                    dialogGuard->setValue(std::min(done, progress.totalFiles));
                    dialogGuard->setLabelText(label);
                },
                Qt::QueuedConnection);
        };

    struct DownloadResult {
        std::vector<std::filesystem::path> dirs;
        std::string error;
    };

    QFutureWatcher<DownloadResult> watcher;
    QEventLoop loop;
    connect(&watcher, &QFutureWatcher<DownloadResult>::finished, &loop, &QEventLoop::quit);
    watcher.setFuture(QtConcurrent::run([infos, remoteRoot, progressCallback, cancelFlag]() {
        DownloadResult result;
        for (const auto& info : infos) {
            std::string error;
            auto dir = downloadOpenDataNormalGrids(
                info, remoteRoot, kNormalGridsDownloadWorkers,
                progressCallback, cancelFlag.get(), &error);
            if (dir.empty()) {
                result.error = error;
                break;
            }
            result.dirs.push_back(std::move(dir));
        }
        return result;
    }));
    if (!watcher.isFinished()) {
        loop.exec();
    }
    const DownloadResult result = watcher.result();
    if (dialogGuard) {
        dialogGuard->close();
        dialogGuard->deleteLater();
    }

    if (result.dirs.size() != infos.size()) {
        setStatus(cancelFlag->load(std::memory_order_acquire)
                      ? tr("Normal grid download for %1 cancelled.").arg(volumeLabel)
                      : tr("Normal grid download for %1 failed: %2")
                            .arg(volumeLabel, qstr(result.error)));
    } else {
        setStatus(tr("%1 normal-grid artifact(s) for %2 downloaded.")
                      .arg(result.dirs.size()).arg(volumeLabel));
    }

    // Refresh the volume's status cell and the button state.
    if (_volumesTable && volumeRow >= 0 && volumeRow < _volumesTable->rowCount() &&
        selectedSample() == sample) {
        _volumesTable->setItem(
            volumeRow, 7, item(normalGridsStatusDisplay(remoteRoot, sample->id, *volume)));
    }
    updateActionButtons();
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

void OpenDataCatalogWindow::refreshSelectedSample()
{
    const auto* sample = selectedSample();
    if (!sample || !_refreshSampleHandler) return;
    setStatus(tr("Refreshing %1 from Open Data...").arg(qstr(sample->id)));
    setStatus(_refreshSampleHandler(*sample));
}

} // namespace vc3d::opendata
