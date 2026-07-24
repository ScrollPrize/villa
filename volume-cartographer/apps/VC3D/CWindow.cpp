#include "CWindow.hpp"
#include "OpenDataCoordinateIdentity.hpp"
#include "OpenDataLasagna.hpp"

#include "RenderBenchRecorder.hpp"
#include "RenderBenchReplay.hpp"
#include "StatusDockPanelHost.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/AffineTransform.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "vc/atlas/Atlas.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <iostream>

#include <array>
#include <functional>
#include <optional>

#include "VCSettings.hpp"
#include "RemoteVolumeCachePaths.hpp"
#include "Keybinds.hpp"
#include "OpenDataNormalGrids.hpp"
#include "viewer_controls/panels/ViewerCompositePanel.hpp"
#include "viewer_controls/panels/ViewerInkDetectionPanel.hpp"
#include <QGridLayout>
#include <QAction>
#include <QVBoxLayout>
#include <QCursor>
#include <QEvent>
#include <QKeyEvent>
#include <QResizeEvent>
#include <QSettings>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QApplication>
#include <QGuiApplication>
#include <QBrush>
#include <QDialog>
#include <QDialogButtonBox>
#include <QStyleHints>
#include <QWindow>
#include <QScreen>
#include <QDesktopServices>
#include <QUrl>
#include <QClipboard>
#include <QFileDialog>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPointF>
#include <QMessageBox>
#include <QtConcurrent/QtConcurrent>
#include <QComboBox>
#include <QFutureWatcher>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QDockWidget>
#include <QLabel>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QStatusBar>
#include <QSizePolicy>
#include <QTimer>
#include <QSize>
#include <QVector>
#include <QVector3D>
#include <QLoggingCategory>
#include <QDebug>
#include <QFrame>
#include <QMouseEvent>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QStandardPaths>
#include <QMenu>
#include <QMainWindow>
#include <QTabBar>
#include <QTabWidget>
#include <QTreeWidget>
#include <QHeaderView>
#include <QFormLayout>
#include <QProgressBar>
#include <QColorDialog>
#include <QListView>
#include <QTreeView>
#include <QPainter>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QStyledItemDelegate>
#include <QStyleOptionButton>
#include <QPersistentModelIndex>
#include <QSet>
#include <QPushButton>
#include <QItemSelectionModel>
#include <QItemSelection>
#include <QUnhandledException>
#include "utils/Json.hpp"
#include <QPointer>
#include <QListView>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include "vc/core/types/Segmentation.hpp"
#include <limits>
#include <optional>
#include <sstream>
#include <cctype>
#include <string_view>
#include <utility>
#include <filesystem>
#include <system_error>
#include <unordered_map>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QStringList>
#include <array>

#include "volume_viewers/CVolumeViewerView.hpp"
#include "viewer_controls/ViewerControlsPanel.hpp"
#include "viewer_controls/panels/ViewerTransformsPanel.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "SettingsDialog.hpp"
#include "elements/VolumeSelector.hpp"
#include "elements/ViewerSplitGrid.hpp"
#include "CPointCollectionWidget.hpp"
#include "WrapAnnotationWidget.hpp"
#include "AtlasControlPointsDock.hpp"
#include "CFiberWidget.hpp"
#include "FiberAnnotationController.hpp"
#include "overlays/FiberOverlayController.hpp"
#include "LineAnnotationController.hpp"
#include "LineAnnotationDialog.hpp"
#include "SurfaceTreeWidget.hpp"
#include "SeedingWidget.hpp"
#include "CommandLineToolRunner.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"
#include "segmentation/growth/SegmentationGrower.hpp"
#include "SurfacePanelController.hpp"
#include "elements/DropdownChecklistButton.hpp"
#include "MenuActionController.hpp"
#include "FileWatcherService.hpp"
#include "AxisAlignedSliceController.hpp"
#include "SurfaceAreaCalculator.hpp"
#include "SegmentationCommandHandler.hpp"
#include "LasagnaServiceManager.hpp"
#include "SpiralWorkspace.hpp"
#include "SurfaceOverlayColors.hpp"
#include "segmentation/panels/SegmentationLasagnaPanel.hpp"
#include "vc/core/Version.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Render.hpp"
#include "vc/core/util/Tiff.hpp"
#include "vc/atlas/Atlas.hpp"
#include "FiberSliceGeometry.hpp"
#include "vc/atlas/FiberIntersections.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include <utils/zarr.hpp>





Q_LOGGING_CATEGORY(lcSegGrowth, "vc.segmentation.growth");

using qga = QGuiApplication;
using PathBrushShape = ViewerOverlayControllerBase::PathBrushShape;
namespace
{
constexpr auto WORKSPACE_TAB_SETTING = "mainWin/workspace_tab";
constexpr auto WORKSPACE_ID_SETTING = "mainWin/workspace_id";
constexpr auto MAIN_VIEWER_SPLIT_X_SETTING = "mainWin/main_viewer_split_x";
constexpr auto MAIN_VIEWER_SPLIT_Y_SETTING = "mainWin/main_viewer_split_y";
constexpr auto MAIN_VIEWER_LAYOUT_SURFACES_SETTING = "mainWin/main_viewer_layout_surfaces";
constexpr auto MAIN_VIEWER_LAYOUT_HIDDEN_SETTING = "mainWin/main_viewer_layout_hidden";
constexpr auto ATLAS_INTERNAL_SURFACE_NAME = "__atlas_workspace_base_mesh";
constexpr int ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS = 0;
constexpr int ATLAS_SEARCH_MODE_NON_ATLAS_ONLY = 1;
constexpr int ATLAS_SEARCH_PHASE_COUNT = 5;
constexpr int ATLAS_SEARCH_RESULT_INDEX_ROLE = Qt::UserRole;
constexpr int ATLAS_SEARCH_FIBER_ID_ROLE = Qt::UserRole + 1;
constexpr int ATLAS_FIBER_ID_ROLE = Qt::UserRole;
constexpr int ATLAS_FIBER_PATH_KEY_ROLE = Qt::UserRole + 1;
constexpr int ATLAS_CONTROL_INDEX_ROLE = Qt::UserRole + 2;
constexpr int ATLAS_CONTROL_SOURCE_INDEX_ROLE = Qt::UserRole + 3;
constexpr int ATLAS_SURFACE_X_ROLE = Qt::UserRole + 4;
constexpr int ATLAS_SURFACE_Y_ROLE = Qt::UserRole + 5;
constexpr int SEGMENT_DIR_NAME_ROLE = Qt::UserRole + 100;
constexpr int SEGMENT_DIR_COLOR_ROLE = Qt::UserRole + 101;
constexpr int SEGMENT_DIR_DEFAULT_PALETTE_ROLE = Qt::UserRole + 102;
constexpr int SURFACE_OVERLAY_KIND_ROLE = Qt::UserRole + 200;
constexpr int SURFACE_OVERLAY_NAME_ROLE = Qt::UserRole + 201;
constexpr int SURFACE_OVERLAY_FOLDER_ROLE = Qt::UserRole + 202;
constexpr double ATLAS_SEARCH_CLOSE_WINDING_THRESHOLD = 0.5;
constexpr std::string_view OPEN_DATA_VOLUME_ID_TAG_PREFIX = "vc-open-data-volume-id:";

std::optional<vc3d::opendata::ResolvedOpenDataLasagna>
resolvedLasagnaForState(const CState* state)
{
    if (!state || !state->vpkg()) return std::nullopt;
    return vc3d::opendata::resolveLasagnaForVolume(
        *state->vpkg(), state->currentVolumeId());
}

enum class SurfaceOverlayItemKind {
    Folder,
    Surface,
};

SurfaceOverlayItemKind surfaceOverlayItemKind(const QStandardItem* item)
{
    return static_cast<SurfaceOverlayItemKind>(
        item ? item->data(SURFACE_OVERLAY_KIND_ROLE).toInt()
             : static_cast<int>(SurfaceOverlayItemKind::Surface));
}

bool isSurfaceOverlayFolder(const QStandardItem* item)
{
    return surfaceOverlayItemKind(item) == SurfaceOverlayItemKind::Folder;
}

bool isSurfaceOverlaySurface(const QStandardItem* item)
{
    return surfaceOverlayItemKind(item) == SurfaceOverlayItemKind::Surface;
}

QStandardItem* findSurfaceOverlayFolder(QStandardItemModel* model, const QString& folderName)
{
    if (!model) {
        return nullptr;
    }
    for (int row = 0; row < model->rowCount(); ++row) {
        QStandardItem* item = model->item(row);
        if (isSurfaceOverlayFolder(item) &&
            item->data(SURFACE_OVERLAY_FOLDER_ROLE).toString() == folderName) {
            return item;
        }
    }
    return nullptr;
}

QList<QStandardItem*> surfaceOverlaySurfaceItems(QStandardItemModel* model)
{
    QList<QStandardItem*> items;
    if (!model) {
        return items;
    }
    for (int folderRow = 0; folderRow < model->rowCount(); ++folderRow) {
        QStandardItem* folder = model->item(folderRow);
        if (!isSurfaceOverlayFolder(folder)) {
            continue;
        }
        for (int row = 0; row < folder->rowCount(); ++row) {
            QStandardItem* child = folder->child(row);
            if (isSurfaceOverlaySurface(child)) {
                items.append(child);
            }
        }
    }
    return items;
}

QString surfaceOverlayFolderName(const std::string& surfaceName,
                                 const Surface* surface,
                                 const QString& currentDir)
{
    if (surface && !surface->meta.is_null()) {
        const auto folder = vc::json::string_or(
            surface->meta,
            "vc3d_segment_folder",
            std::string{});
        if (!folder.empty()) {
            return QString::fromStdString(folder);
        }
    }

    const QString name = QString::fromStdString(surfaceName);
    const int slash = name.indexOf(QLatin1Char('/'));
    if (slash > 0) {
        return name.left(slash);
    }
    if (!currentDir.isEmpty()) {
        return currentDir;
    }
    return QObject::tr("Segments");
}

QString surfaceOverlayLeafLabel(const std::string& surfaceName, const QString& folderName)
{
    const QString name = QString::fromStdString(surfaceName);
    const QString prefix = folderName + QLatin1Char('/');
    if (name.startsWith(prefix)) {
        return name.mid(prefix.size());
    }
    return name;
}

void updateSurfaceOverlayFolderState(QStandardItem* folder)
{
    if (!folder || !isSurfaceOverlayFolder(folder)) {
        return;
    }

    int checkedCount = 0;
    int surfaceCount = 0;
    for (int row = 0; row < folder->rowCount(); ++row) {
        QStandardItem* child = folder->child(row);
        if (!isSurfaceOverlaySurface(child)) {
            continue;
        }
        ++surfaceCount;
        if (child->checkState() == Qt::Checked) {
            ++checkedCount;
        }
    }

    if (checkedCount == 0) {
        folder->setCheckState(Qt::Unchecked);
    } else if (checkedCount == surfaceCount && surfaceCount > 0) {
        folder->setCheckState(Qt::Checked);
    } else {
        folder->setCheckState(Qt::PartiallyChecked);
    }
}

bool surfaceOverlayItemMatchesFilter(const QStandardItem* item, const QString& filter)
{
    if (!item || filter.isEmpty()) {
        return true;
    }
    const QString needle = filter.toCaseFolded();
    if (item->text().toCaseFolded().contains(needle) ||
        item->data(SURFACE_OVERLAY_NAME_ROLE).toString().toCaseFolded().contains(needle) ||
        item->data(SURFACE_OVERLAY_FOLDER_ROLE).toString().toCaseFolded().contains(needle)) {
        return true;
    }
    for (int row = 0; row < item->rowCount(); ++row) {
        if (surfaceOverlayItemMatchesFilter(item->child(row), filter)) {
            return true;
        }
    }
    return false;
}

VolumeViewerBase* baseViewerFromWidget(QWidget* widget);

QRect segmentFolderCheckRect(const QRect& row)
{
    return QRect(row.left() + 4, row.top() + (row.height() - 18) / 2, 18, 18);
}

QRect segmentFolderPaletteRect(const QRect& row)
{
    return QRect(row.right() - 26, row.top() + (row.height() - 20) / 2, 20, 20);
}

enum class SegmentFolderControl {
    None,
    CheckBox,
    Palette,
};

SegmentFolderControl segmentFolderControlAt(const QRect& row, const QPoint& pos)
{
    if (segmentFolderCheckRect(row).contains(pos)) {
        return SegmentFolderControl::CheckBox;
    }
    if (segmentFolderPaletteRect(row).contains(pos)) {
        return SegmentFolderControl::Palette;
    }
    return SegmentFolderControl::None;
}

class SegmentFolderDelegate final : public QStyledItemDelegate
{
public:
    using PaletteCallback = std::function<void(int)>;

    explicit SegmentFolderDelegate(PaletteCallback paletteCallback, QObject* parent = nullptr)
        : QStyledItemDelegate(parent)
        , _paletteCallback(std::move(paletteCallback))
    {
    }

    void paint(QPainter* painter,
               const QStyleOptionViewItem& option,
               const QModelIndex& index) const override
    {
        painter->save();

        QStyleOptionViewItem opt(option);
        initStyleOption(&opt, index);
        opt.text.clear();
        QStyle* style = opt.widget ? opt.widget->style() : QApplication::style();
        style->drawControl(QStyle::CE_ItemViewItem, &opt, painter, opt.widget);

        QStyleOptionButton check;
        check.state = QStyle::State_Enabled;
        check.state |= index.data(Qt::CheckStateRole).toInt() == Qt::Checked
            ? QStyle::State_On
            : QStyle::State_Off;
        check.rect = segmentFolderCheckRect(option.rect);
        style->drawPrimitive(QStyle::PE_IndicatorCheckBox, &check, painter, opt.widget);

        const QRect icon = segmentFolderPaletteRect(option.rect);
        const bool defaultPalette = index.data(SEGMENT_DIR_DEFAULT_PALETTE_ROLE).toBool();
        const QColor color = index.data(SEGMENT_DIR_COLOR_ROLE).value<QColor>();
        if (defaultPalette) {
            const std::array<QColor, 4> colors = {
                QColor(255, 120, 120),
                QColor(120, 200, 255),
                QColor(120, 255, 140),
                QColor(255, 220, 100),
            };
            const int w = icon.width() / 2;
            const int h = icon.height() / 2;
            for (int i = 0; i < 4; ++i) {
                painter->fillRect(QRect(icon.left() + (i % 2) * w,
                                        icon.top() + (i / 2) * h,
                                        w,
                                        h),
                                  colors[i]);
            }
            painter->setPen(option.palette.mid().color());
            painter->drawRect(icon.adjusted(0, 0, -1, -1));
        } else {
            painter->setRenderHint(QPainter::Antialiasing, true);
            painter->setPen(option.palette.mid().color());
            painter->setBrush(color.isValid() ? color : QColor(120, 200, 255));
            painter->drawEllipse(icon.adjusted(2, 2, -2, -2));
        }

        QRect text = option.rect.adjusted(check.rect.width() + 10, 0, -(icon.width() + 10), 0);
        painter->setPen(option.palette.text().color());
        painter->drawText(text, Qt::AlignVCenter | Qt::AlignLeft, index.data(Qt::DisplayRole).toString());

        painter->restore();
    }

    bool editorEvent(QEvent* event,
                     QAbstractItemModel* model,
                     const QStyleOptionViewItem& option,
                     const QModelIndex& index) override
    {
        if (!event || !model || !index.isValid() || event->type() != QEvent::MouseButtonRelease) {
            return QStyledItemDelegate::editorEvent(event, model, option, index);
        }
        const auto* mouse = static_cast<QMouseEvent*>(event);
        if (segmentFolderCheckRect(option.rect).contains(mouse->pos())) {
            const bool checked = index.data(Qt::CheckStateRole).toInt() == Qt::Checked;
            model->setData(index, checked ? Qt::Unchecked : Qt::Checked, Qt::CheckStateRole);
            return true;
        }
        if (segmentFolderPaletteRect(option.rect).contains(mouse->pos())) {
            if (_paletteCallback) {
                _paletteCallback(index.row());
            }
            return true;
        }
        return QStyledItemDelegate::editorEvent(event, model, option, index);
    }

private:
    PaletteCallback _paletteCallback;
};

class SegmentFolderListView final : public QListView
{
public:
    using PaletteCallback = std::function<void(int)>;

    explicit SegmentFolderListView(PaletteCallback paletteCallback, QWidget* parent = nullptr)
        : QListView(parent)
        , _paletteCallback(std::move(paletteCallback))
    {
    }

protected:
    void showEvent(QShowEvent* event) override
    {
        // QComboBox's popup container filters viewport mouse events and closes
        // the popup / selects the hovered row on release. Installing this
        // filter on every show keeps it ahead of the container's, so clicks on
        // the checkbox or palette controls can be consumed before the combo
        // reacts to them.
        viewport()->removeEventFilter(this);
        viewport()->installEventFilter(this);
        QListView::showEvent(event);
    }

    bool eventFilter(QObject* watched, QEvent* event) override
    {
        if (watched != viewport()) {
            return QListView::eventFilter(watched, event);
        }

        switch (event->type()) {
        case QEvent::MouseButtonPress:
        case QEvent::MouseButtonDblClick: {
            const auto* mouse = static_cast<QMouseEvent*>(event);
            if (mouse->button() != Qt::LeftButton) {
                break;
            }
            const QModelIndex index = indexAt(mouse->pos());
            const SegmentFolderControl control =
                index.isValid() ? segmentFolderControlAt(visualRect(index), mouse->pos())
                                : SegmentFolderControl::None;
            if (control != SegmentFolderControl::None) {
                _pressedIndex = QPersistentModelIndex(index);
                _pressedControl = control;
                return true;
            }
            break;
        }
        case QEvent::MouseButtonRelease: {
            const auto* mouse = static_cast<QMouseEvent*>(event);
            if (mouse->button() != Qt::LeftButton || !_pressedIndex.isValid()) {
                break;
            }
            const QModelIndex index = indexAt(mouse->pos());
            const SegmentFolderControl control =
                index == QModelIndex(_pressedIndex)
                    ? segmentFolderControlAt(visualRect(index), mouse->pos())
                    : SegmentFolderControl::None;
            const QModelIndex pressed(_pressedIndex);
            const bool activate = control == _pressedControl;
            _pressedIndex = QPersistentModelIndex();
            _pressedControl = SegmentFolderControl::None;
            if (activate) {
                activateControl(pressed, control);
            }
            // Swallow the release even on a cancelled press so the combo
            // neither switches rows nor closes the popup.
            return true;
        }
        default:
            break;
        }
        return QListView::eventFilter(watched, event);
    }

private:
    void activateControl(const QModelIndex& index, SegmentFolderControl control)
    {
        if (!index.isValid()) {
            return;
        }

        if (control == SegmentFolderControl::CheckBox) {
            if (QAbstractItemModel* itemModel = model()) {
                const bool checked = index.data(Qt::CheckStateRole).toInt() == Qt::Checked;
                itemModel->setData(index, checked ? Qt::Unchecked : Qt::Checked, Qt::CheckStateRole);
            }
            return;
        }

        if (control == SegmentFolderControl::Palette && _paletteCallback) {
            // Defer so the menu's nested event loop does not run inside this
            // event filter while the popup is still dispatching the release.
            const int row = index.row();
            QTimer::singleShot(0, this, [this, row]() { _paletteCallback(row); });
        }
    }

    PaletteCallback _paletteCallback;
    QPersistentModelIndex _pressedIndex;
    SegmentFolderControl _pressedControl{SegmentFolderControl::None};
};

std::filesystem::path openDataCatalogManifestCachePath()
{
    QString base = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
    if (base.isEmpty()) {
        base = QDir::home().filePath(QStringLiteral(".VC3D"));
    }
    return std::filesystem::path(base.toStdString()) / "open-data-catalog" / "metadata.json";
}

QString formatAtlasCoveredSize(const vc::atlas::AtlasCoveredSize& size)
{
    if (!size.valid) {
        return QObject::tr("No valid footprint");
    }
    auto formatValue = [](double value) {
        if (std::abs(value - std::round(value)) < 1.0e-6) {
            return QString::number(static_cast<qint64>(std::llround(value)));
        }
        return QString::number(value, 'f', 2);
    };
    return QStringLiteral("%1 x %2 vx").arg(formatValue(size.width), formatValue(size.height));
}

QString openDataCatalogVolumeIdForLoadedVolume(const VolumePkg& pkg, const std::string& loadedVolumeId)
{
    constexpr std::string_view prefix = "vc-open-data-volume-id:";
    for (const auto& tag : pkg.volumeTags(loadedVolumeId)) {
        if (tag.rfind(prefix, 0) == 0) {
            return QString::fromStdString(tag.substr(prefix.size()));
        }
    }
    return {};
}

std::optional<int> openDataCoordinateLevelForLoadedVolume(
    const VolumePkg& pkg,
    const std::string& loadedVolumeId)
{
    constexpr std::string_view prefix = "vc-open-data-source-coordinate-level:";
    for (const auto& tag : pkg.volumeTags(loadedVolumeId)) {
        if (tag.rfind(prefix, 0) != 0)
            continue;
        const auto value = tag.substr(prefix.size());
        try {
            std::size_t consumed = 0;
            const int level = std::stoi(value, &consumed);
            if (consumed == value.size() && level >= 0 && level <= 5)
                return level;
        } catch (...) {
        }
    }
    return std::nullopt;
}

std::string openDataCoordinateSpaceForLoadedVolume(
    const VolumePkg& pkg,
    const std::string& loadedVolumeId)
{
    constexpr std::string_view prefix = "vc-open-data-coordinate-space:";
    for (const auto& tag : pkg.volumeTags(loadedVolumeId)) {
        if (tag.rfind(prefix, 0) == 0)
            return tag.substr(prefix.size());
    }
    return {};
}

cv::Matx44d coordinateLevelScale(int level)
{
    const double scale = std::ldexp(1.0, level);
    return cv::Matx44d(
        scale, 0, 0, 0,
        0, scale, 0, 0,
        0, 0, scale, 0,
        0, 0, 0, 1);
}

bool isOpenDataSegmentsEntry(const vc::project::Entry& entry)
{
    if (std::find(entry.tags.begin(), entry.tags.end(), "open-data") != entry.tags.end()) {
        return true;
    }
    return std::any_of(entry.tags.begin(), entry.tags.end(), [](const std::string& tag) {
        constexpr std::string_view sourcePrefix = "vc-open-data-source-volume-id:";
        constexpr std::string_view targetPrefix = "vc-open-data-target-volume-id:";
        return tag.rfind(sourcePrefix, 0) == 0 || tag.rfind(targetPrefix, 0) == 0;
    });
}

bool isAvailableOpenDataSegmentsEntry(const VolumePkg& pkg,
                                      const vc::project::Entry& entry)
{
    if (!isOpenDataSegmentsEntry(entry)) {
        return false;
    }
    if (vc::project::isLocationRemote(entry.location)) {
        return true;
    }

    std::error_code ec;
    const auto path = vc::project::resolveLocalPath(
        entry.location,
        pkg.path().parent_path());
    if (!std::filesystem::is_directory(path, ec) || ec) {
        return false;
    }
    if (std::find(entry.tags.begin(), entry.tags.end(),
                  "vc-open-data-segment-aggregate") != entry.tags.end()) {
        return true;
    }
    return vc::project::validateLocation(
        vc::project::Category::Segments,
        path.string()).empty();
}

std::vector<QString> openDataCatalogVolumeIdCandidates(const VolumePkg& pkg,
                                                       const std::string& loadedVolumeId)
{
    std::vector<QString> candidates;
    constexpr std::string_view prefix = "vc-open-data-volume-id:";

    auto addCandidate = [&candidates](QString candidate) {
        if (candidate.isEmpty()) {
            return;
        }
        if (std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
            candidates.push_back(std::move(candidate));
        }
    };

    for (const auto& tag : pkg.volumeTags(loadedVolumeId)) {
        if (tag.rfind(prefix, 0) == 0) {
            addCandidate(QString::fromStdString(tag.substr(prefix.size())));
        }
    }
    addCandidate(QString::fromStdString(loadedVolumeId));

    return candidates;
}

const vc::project::Entry* findOpenDataSegmentsEntryForVolume(const VolumePkg& pkg,
                                                             const QString& catalogVolumeId)
{
    if (catalogVolumeId.isEmpty()) {
        return nullptr;
    }

    const std::string targetTag = "vc-open-data-target-volume-id:" + catalogVolumeId.toStdString();
    const std::string sourceTag = "vc-open-data-source-volume-id:" + catalogVolumeId.toStdString();
    const vc::project::Entry* sourceMatch = nullptr;

    for (const auto& entry : pkg.segmentEntries()) {
        if (!isAvailableOpenDataSegmentsEntry(pkg, entry)) {
            continue;
        }
        if (std::find(entry.tags.begin(), entry.tags.end(), targetTag) != entry.tags.end()) {
            return &entry;
        }
        if (!sourceMatch &&
            std::find(entry.tags.begin(), entry.tags.end(), sourceTag) != entry.tags.end()) {
            sourceMatch = &entry;
        }
    }

    return sourceMatch;
}

const vc::project::Entry* findOpenDataSegmentsEntryForLoadedVolume(const VolumePkg& pkg,
                                                                   const std::string& loadedVolumeId,
                                                                   QString* matchedCatalogVolumeId = nullptr)
{
    const auto coordinateSpace =
        openDataCoordinateSpaceForLoadedVolume(pkg, loadedVolumeId);
    if (!coordinateSpace.empty()) {
        const std::string coordinateTag =
            "vc-open-data-coordinate-space:" + coordinateSpace;
        for (const auto& entry : pkg.segmentEntries()) {
            if (isAvailableOpenDataSegmentsEntry(pkg, entry) &&
                std::find(entry.tags.begin(), entry.tags.end(), coordinateTag) !=
                    entry.tags.end()) {
                if (matchedCatalogVolumeId) {
                    *matchedCatalogVolumeId =
                        openDataCatalogVolumeIdForLoadedVolume(pkg, loadedVolumeId);
                }
                return &entry;
            }
        }
        // Explicitly identified assets must never fall back to lineage-only
        // association, because native and virtual views share that lineage.
        return nullptr;
    }

    for (const QString& candidate : openDataCatalogVolumeIdCandidates(pkg, loadedVolumeId)) {
        if (const auto* entry = findOpenDataSegmentsEntryForVolume(pkg, candidate)) {
            if (matchedCatalogVolumeId) {
                *matchedCatalogVolumeId = candidate;
            }
            return entry;
        }
    }
    return nullptr;
}

bool packageHasOpenDataSegments(const VolumePkg& pkg)
{
    for (const auto& entry : pkg.segmentEntries()) {
        if (isAvailableOpenDataSegmentsEntry(pkg, entry)) {
            return true;
        }
    }
    return false;
}

bool volumeHasOpenDataSegmentsEntry(const VolumePkg& pkg, const std::string& loadedVolumeId)
{
    return findOpenDataSegmentsEntryForLoadedVolume(pkg, loadedVolumeId) != nullptr;
}

class DockMenuMainWindow : public QMainWindow
{
public:
    using DockMenuBuilder = std::function<void(QMenu*)>;

    explicit DockMenuMainWindow(QWidget* parent = nullptr)
        : QMainWindow(parent)
    {
    }

    void setDockMenuBuilder(DockMenuBuilder builder)
    {
        _dockMenuBuilder = std::move(builder);
    }

protected:
    QMenu* createPopupMenu() override
    {
        auto* menu = new QMenu(this);
        if (_dockMenuBuilder) {
            _dockMenuBuilder(menu);
        }
        return menu;
    }

private:
    DockMenuBuilder _dockMenuBuilder;
};

struct MainViewerSpec {
    std::string surfaceName;
    QString title;
    std::set<std::string> intersects;
};

const std::array<const char*, 4> kDefaultMainViewerSurfaces{
    "segmentation",
    "xy plane",
    "seg xz",
    "seg yz",
};

const std::array<bool, 4> kDefaultMainViewerHidden{
    false,
    false,
    true,
    false,
};

const std::array<const char*, 4> kMainViewerPaneLabels{
    "Top left",
    "Top right",
    "Bottom left",
    "Bottom right",
};

std::optional<MainViewerSpec> mainViewerSpecForSurface(const std::string& surfaceName)
{
    if (surfaceName == "segmentation") {
        return MainViewerSpec{surfaceName, QObject::tr("Surface"), {"seg xz", "seg yz"}};
    }
    if (surfaceName == "xy plane") {
        return MainViewerSpec{surfaceName, QObject::tr("XY"), {"segmentation"}};
    }
    if (surfaceName == "seg xz") {
        return MainViewerSpec{surfaceName, QObject::tr("XZ"), {"segmentation"}};
    }
    if (surfaceName == "seg yz") {
        return MainViewerSpec{surfaceName, QObject::tr("YZ"), {"segmentation"}};
    }
    return std::nullopt;
}

QString mainViewerDisplayName(const std::string& surfaceName)
{
    if (surfaceName == "segmentation") {
        return QObject::tr("Surface");
    }
    if (surfaceName == "xy plane") {
        return QObject::tr("XY");
    }
    if (surfaceName == "seg xz") {
        return QObject::tr("XZ");
    }
    if (surfaceName == "seg yz") {
        return QObject::tr("YZ");
    }
    return QString::fromStdString(surfaceName);
}

std::array<std::string, 4> readMainViewerSurfaceLayout()
{
    std::array<std::string, 4> layout{};
    for (int i = 0; i < 4; ++i) {
        layout[i] = kDefaultMainViewerSurfaces[i];
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QStringList saved = settings.value(MAIN_VIEWER_LAYOUT_SURFACES_SETTING).toStringList();
    if (saved.size() != 4) {
        return layout;
    }

    std::set<std::string> seen;
    std::array<std::string, 4> candidate{};
    for (int i = 0; i < 4; ++i) {
        const std::string name = saved[i].toStdString();
        if (!mainViewerSpecForSurface(name) || !seen.insert(name).second) {
            return layout;
        }
        candidate[i] = name;
    }
    return candidate;
}

std::array<bool, 4> readMainViewerHiddenLayout()
{
    std::array<bool, 4> hidden = kDefaultMainViewerHidden;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QStringList saved = settings.value(MAIN_VIEWER_LAYOUT_HIDDEN_SETTING).toStringList();
    if (saved.size() != 4) {
        return hidden;
    }

    int visibleCount = 0;
    for (int i = 0; i < 4; ++i) {
        hidden[i] = saved[i].toInt() != 0;
        if (!hidden[i]) {
            ++visibleCount;
        }
    }
    if (visibleCount == 0) {
        hidden[0] = false;
    }
    return hidden;
}

void persistMainViewerLayout(ViewerSplitGrid* grid)
{
    if (!grid) {
        return;
    }

    QStringList surfaces;
    QStringList hidden;
    for (int i = 0; i < 4; ++i) {
        std::string surfaceName = kDefaultMainViewerSurfaces[i];
        if (auto* viewer = baseViewerFromWidget(grid->viewer(i))) {
            surfaceName = viewer->surfName();
        }
        surfaces.push_back(QString::fromStdString(surfaceName));
        hidden.push_back(grid->paneHidden(i) ? QStringLiteral("1") : QStringLiteral("0"));
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(MAIN_VIEWER_LAYOUT_SURFACES_SETTING, surfaces);
    settings.setValue(MAIN_VIEWER_LAYOUT_HIDDEN_SETTING, hidden);
}

ViewerSplitGrid* mainViewerSplitGrid(QWidget* parent)
{
    auto* widget = parent
        ? parent->findChild<QWidget*>(QStringLiteral("mainViewerSplitGrid"))
        : nullptr;
    return dynamic_cast<ViewerSplitGrid*>(widget);
}

void applyMainViewerLayout(ViewerSplitGrid* grid, ViewerManager* manager)
{
    if (!grid || !manager) {
        return;
    }

    const auto surfaces = readMainViewerSurfaceLayout();
    const auto hidden = readMainViewerHiddenLayout();
    for (int pane = 0; pane < 4; ++pane) {
        QWidget* widget = nullptr;
        for (auto* viewer : manager->baseViewers()) {
            if (!viewer || viewer->surfName() != surfaces[pane]) {
                continue;
            }
            widget = qobject_cast<QWidget*>(viewer->asQObject());
            if (widget && widget->parentWidget() == grid) {
                break;
            }
            widget = nullptr;
        }
        grid->setViewer(pane, widget);
        grid->setPaneHidden(pane, hidden[pane]);
    }
}

QString atlasSearchIdListString(const std::vector<uint64_t>& ids)
{
    QStringList parts;
    parts.reserve(static_cast<int>(ids.size()));
    for (uint64_t id : ids) {
        parts.push_back(QString::number(id));
    }
    return parts.join(QStringLiteral(","));
}

bool atlasSearchDebugEnabled()
{
    const char* value = std::getenv("VC_ATLAS_SEARCH_DEBUG");
    return value && *value != '\0' && std::string_view(value) != "0";
}

QString extractFutureExceptionMessage(const std::exception& e)
{
    if (auto* unhandled = dynamic_cast<const QUnhandledException*>(&e)) {
        const std::exception_ptr ptr = unhandled->exception();
        if (ptr) {
            try {
                std::rethrow_exception(ptr);
            } catch (const std::exception& inner) {
                return QString::fromStdString(inner.what());
            } catch (...) {
                return QObject::tr("Unknown non-standard exception");
            }
        }
    }
    return QString::fromStdString(e.what());
}

QString formatOptionalDouble(std::optional<double> value, int precision = 3)
{
    return (value && std::isfinite(*value))
        ? QString::number(*value, 'f', precision)
        : QStringLiteral("-");
}

QString formatAtlasVec3(const cv::Vec3d& point, int precision = 1)
{
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) {
        return QStringLiteral("-");
    }
    return QStringLiteral("%1, %2, %3")
        .arg(point[0], 0, 'f', precision)
        .arg(point[1], 0, 'f', precision)
        .arg(point[2], 0, 'f', precision);
}

QString predSnapDirectionLabel(std::optional<vc::atlas::AtlasPredSnapDirection> direction)
{
    if (!direction) {
        return QStringLiteral("-");
    }
    return *direction == vc::atlas::AtlasPredSnapDirection::Inside
        ? QObject::tr("inside")
        : QObject::tr("outside");
}

bool isResolvedAtlasPredSnap(const vc::atlas::AtlasPredSnapPoint& point)
{
    return point.predSnapPoint.has_value() &&
           (point.source == vc::atlas::AtlasPredSnapSource::Manual ||
            point.source == vc::atlas::AtlasPredSnapSource::Optimized);
}

const vc::atlas::AtlasPredSnapPoint* predSnapPointForAnchor(
    const std::unordered_map<std::string, const vc::atlas::AtlasPredSnapPoint*>& snapsByControl,
    const vc::atlas::AtlasAnchor& anchor)
{
    const auto it = snapsByControl.find(vc::atlas::atlasPredSnapControlPointKey(anchor.world));
    return it == snapsByControl.end() ? nullptr : it->second;
}

std::optional<double> predSnapDisplayWinding(const vc::atlas::AtlasPredSnapPoint& point)
{
    if (point.selectedCandidateIndex &&
        *point.selectedCandidateIndex >= 0 &&
        *point.selectedCandidateIndex < static_cast<int>(point.candidates.size())) {
        const auto& candidate = point.candidates[static_cast<size_t>(*point.selectedCandidateIndex)];
        if (candidate.windingDistance) {
            return candidate.windingDistance;
        }
    }
    return point.weightedFirstHitWindingDistance;
}

QString predSnapStatusLabel(const vc::atlas::AtlasPredSnapPoint* point)
{
    if (!point) {
        return QObject::tr("missing");
    }
    if (!point->status.empty()) {
        return QString::fromStdString(point->status);
    }
    if (isResolvedAtlasPredSnap(*point)) {
        return point->source == vc::atlas::AtlasPredSnapSource::Manual
            ? QObject::tr("manual")
            : QObject::tr("optimized");
    }
    if (point->candidates.empty()) {
        return QObject::tr("no_candidates");
    }
    return QObject::tr("unassigned");
}

QString predSnapStatusReason(const vc::atlas::AtlasPredSnapPoint* point)
{
    if (!point) {
        return QObject::tr("No pred-snap attachment record for this control point.");
    }
    if (!point->statusReason.empty()) {
        return QString::fromStdString(point->statusReason);
    }
    if (isResolvedAtlasPredSnap(*point)) {
        return QObject::tr("Snap point is assigned.");
    }
    return QObject::tr("Snap point is not assigned.");
}

std::set<std::string> atlasSelectedFiberPathKeys(QTreeWidget* tree)
{
    std::set<std::string> pathKeys;
    if (!tree) {
        return pathKeys;
    }
    for (QTreeWidgetItem* item : tree->selectedItems()) {
        while (item && item->data(0, ATLAS_FIBER_PATH_KEY_ROLE).toString().isEmpty()) {
            item = item->parent();
        }
        if (!item) {
            continue;
        }
        const QString key = item->data(0, ATLAS_FIBER_PATH_KEY_ROLE).toString();
        if (!key.isEmpty()) {
            pathKeys.insert(key.toStdString());
        }
    }
    return pathKeys;
}

QJsonObject toQtJsonObject(const nlohmann::json& json)
{
    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(
        QByteArray::fromStdString(json.dump()), &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        throw std::runtime_error("failed to convert JSON object for Qt");
    }
    return doc.object();
}

nlohmann::json fromQtJsonObject(const QJsonObject& object)
{
    const QByteArray bytes = QJsonDocument(object).toJson(QJsonDocument::Compact);
    return nlohmann::json::parse(bytes.constData());
}

QString atlasSearchModeName(int searchMode)
{
    return searchMode == ATLAS_SEARCH_MODE_NON_ATLAS_ONLY
        ? QStringLiteral("non_atlas_only")
        : QStringLiteral("atlas_to_non_atlas");
}

int atlasSearchPhaseNumber(vc::atlas::AtlasSearchProgressPhase phase)
{
    return static_cast<int>(phase) + 1;
}

QString atlasSearchPhaseAction(vc::atlas::AtlasSearchProgressPhase phase)
{
    switch (phase) {
    case vc::atlas::AtlasSearchProgressPhase::PrepareInputs:
        return QObject::tr("Preparing inputs");
    case vc::atlas::AtlasSearchProgressPhase::BuildSpatialIndex:
        return QObject::tr("Building spatial index");
    case vc::atlas::AtlasSearchProgressPhase::SearchPairs:
        return QObject::tr("Searching pairs");
    case vc::atlas::AtlasSearchProgressPhase::PrepareSigningSurface:
        return QObject::tr("Preparing signing surface");
    case vc::atlas::AtlasSearchProgressPhase::FinishResults:
        return QObject::tr("Finishing results");
    }
    return QObject::tr("Searching");
}

QStringList atlasSearchTagList(const QString& text)
{
    QStringList tags;
    const auto parts = text.split(QRegularExpression(QStringLiteral("[,\\s]+")),
                                  Qt::SkipEmptyParts);
    for (const QString& part : parts) {
        const QString tag = part.trimmed();
        if (!tag.isEmpty() && !tags.contains(tag)) {
            tags.push_back(tag);
        }
    }
    tags.sort();
    return tags;
}

bool atlasSearchFiberMatchesTags(const std::vector<std::string>& fiberTags,
                                 const QStringList& requiredTags)
{
    if (requiredTags.isEmpty()) {
        return true;
    }
    std::set<QString> normalizedFiberTags;
    for (const auto& tag : fiberTags) {
        normalizedFiberTags.insert(QString::fromStdString(tag).trimmed());
    }
    for (const QString& tag : requiredTags) {
        if (normalizedFiberTags.find(tag) == normalizedFiberTags.end()) {
            return false;
        }
    }
    return true;
}

bool atlasSearchFiberHasAnyTag(const std::vector<std::string>& fiberTags,
                               const QStringList& excludedTags)
{
    if (excludedTags.isEmpty()) {
        return false;
    }
    std::set<QString> normalizedFiberTags;
    for (const auto& tag : fiberTags) {
        normalizedFiberTags.insert(QString::fromStdString(tag).trimmed());
    }
    for (const QString& tag : excludedTags) {
        if (normalizedFiberTags.find(tag) != normalizedFiberTags.end()) {
            return true;
        }
    }
    return false;
}

std::optional<int> atlasSearchResultIndexForItem(const QTreeWidgetItem* item)
{
    if (!item) {
        return std::nullopt;
    }
    bool ok = false;
    const int resultIndex = item->data(0, ATLAS_SEARCH_RESULT_INDEX_ROLE).toInt(&ok);
    return ok ? std::optional<int>(resultIndex) : std::nullopt;
}

std::optional<uint64_t> atlasSearchFiberIdForItem(QTreeWidgetItem* item)
{
    while (item) {
        bool ok = false;
        const qulonglong fiberId = item->data(0, ATLAS_SEARCH_FIBER_ID_ROLE).toULongLong(&ok);
        if (ok && fiberId != 0) {
            return static_cast<uint64_t>(fiberId);
        }
        item = item->parent();
    }
    return std::nullopt;
}

template <typename Fn>
void forEachAtlasSearchResultItem(QTreeWidget* tree, Fn&& fn)
{
    if (!tree) {
        return;
    }
    std::vector<QTreeWidgetItem*> stack;
    stack.reserve(static_cast<size_t>(tree->topLevelItemCount()));
    for (int i = 0; i < tree->topLevelItemCount(); ++i) {
        stack.push_back(tree->topLevelItem(i));
    }
    while (!stack.empty()) {
        QTreeWidgetItem* item = stack.back();
        stack.pop_back();
        if (!item) {
            continue;
        }
        if (atlasSearchResultIndexForItem(item)) {
            fn(item);
        }
        for (int i = 0; i < item->childCount(); ++i) {
            stack.push_back(item->child(i));
        }
    }
}

std::vector<cv::Vec3d> linePointsFromPolyline(const vc::atlas::FiberPolyline& fiber)
{
    std::vector<cv::Vec3d> points;
    points.reserve(fiber.points.size());
    for (const auto& point : fiber.points) {
        points.push_back(point.position);
    }
    return points;
}

vc::atlas::FiberInput fiberInputFromSnapshot(const std::filesystem::path& fiberPath,
                                             const vc::atlas::FiberPolyline& fiber)
{
    vc::atlas::FiberInput input;
    input.fiberPath = fiberPath;
    input.controlPoints = fiber.controlPoints;
    input.linePoints = linePointsFromPolyline(fiber);
    vc::atlas::validateFiberInputControlPoints(input);
    return input;
}

const vc::atlas::FiberMapping* findAtlasMappingForPath(
    const vc::atlas::Atlas& atlas,
    const std::filesystem::path& fiberPath)
{
    const std::string pathKey = vc::atlas::atlasFiberPathKey(fiberPath);
    for (const auto& mapping : atlas.fibers) {
        if (vc::atlas::atlasFiberPathKey(mapping.fiberPath) == pathKey) {
            return &mapping;
        }
    }
    return nullptr;
}

vc::atlas::FiberMapping* findAtlasMappingForPath(
    vc::atlas::Atlas& atlas,
    const std::filesystem::path& fiberPath)
{
    const std::string pathKey = vc::atlas::atlasFiberPathKey(fiberPath);
    for (auto& mapping : atlas.fibers) {
        if (vc::atlas::atlasFiberPathKey(mapping.fiberPath) == pathKey) {
            return &mapping;
        }
    }
    return nullptr;
}

const vc::atlas::AtlasAnchor* nearestLineAnchorForPosition(
    const vc::atlas::FiberMapping& mapping,
    double linePosition)
{
    if (!std::isfinite(linePosition)) {
        return nullptr;
    }
    const vc::atlas::AtlasAnchor* best = nullptr;
    double bestDelta = std::numeric_limits<double>::infinity();
    for (const auto& anchor : mapping.lineAnchors) {
        const double delta = std::abs(static_cast<double>(anchor.sourceIndex) - linePosition);
        if (delta < bestDelta) {
            best = &anchor;
            bestDelta = delta;
        }
    }
    return best;
}

std::optional<cv::Vec2f> atlasSearchCandidateSurfaceCoord(
    const vc::atlas::Atlas& atlas,
    const QuadSurface& displaySurface,
    const vc::atlas::AtlasDisplayRange& displayRange,
    const vc::atlas::FiberIntersectionResult& result,
    const std::unordered_map<uint64_t, AtlasSearchFiberSnapshot>& snapshotsById)
{
    const auto sourceIt = snapshotsById.find(result.sourceFiberId);
    if (sourceIt == snapshotsById.end()) {
        return std::nullopt;
    }
    const auto* mapping = findAtlasMappingForPath(atlas, sourceIt->second.fiberPath);
    if (!mapping) {
        return std::nullopt;
    }
    const auto linePoints = linePointsFromPolyline(sourceIt->second.fiber);
    const auto sample = vc3d::fiber_slice::samplePolylineAtArclength(linePoints,
                                                                     result.sourceArclength);
    if (!sample.valid) {
        return std::nullopt;
    }
    const auto* anchor = nearestLineAnchorForPosition(*mapping, sample.linePosition);
    if (!anchor) {
        return std::nullopt;
    }
    const double atlasU = vc::atlas::actualAtlasU(*anchor, *mapping, displayRange.baseColumns);
    const cv::Vec2f surfaceCoord =
        vc::atlas::atlasGridToSurfaceCoords(atlasU,
                                            anchor->atlasV,
                                            displaySurface,
                                            displayRange.atlasUOffset);
    if (!std::isfinite(surfaceCoord[0]) || !std::isfinite(surfaceCoord[1])) {
        return std::nullopt;
    }
    return surfaceCoord;
}

struct AtlasSearchSignedWindingRow {
    double signedWinding = 0.0;
    uint64_t hFiberId = 0;
    uint64_t vFiberId = 0;
    QString orientationSource;
};

struct AtlasSearchSigningContext {
    vc::atlas::Atlas atlas;
    bool haveAtlas = false;
    vc::atlas::AtlasBaseMappingContext baseMapping;
    QString orientationSource;
};

struct AtlasSearchSignedResult {
    vc::atlas::FiberIntersectionResult result;
    AtlasSearchSignedWindingRow row;
};

struct AtlasSearchWorkerResult {
    std::vector<vc::atlas::FiberIntersectionResult> results;
    std::vector<double> signedWindings;
    std::size_t rawResultCount = 0;
    std::size_t skippedSigningCount = 0;
};

uint64_t atlasSearchTieBreakFiberId(const AtlasSearchFiberSnapshot& snapshot,
                                    uint64_t runtimeFiberId)
{
    return snapshot.storedFiberId != 0 ? snapshot.storedFiberId : runtimeFiberId;
}

const AtlasSearchFiberSnapshot& atlasSearchSnapshotFor(
    const std::unordered_map<uint64_t, AtlasSearchFiberSnapshot>& snapshotsById,
    uint64_t runtimeFiberId)
{
    const auto it = snapshotsById.find(runtimeFiberId);
    if (it == snapshotsById.end()) {
        throw std::runtime_error("Atlas search result refers to a fiber snapshot that is no longer available");
    }
    return it->second;
}

bool atlasSearchSourceDisplaysAsH(
    const vc::atlas::FiberIntersectionResult& result,
    const std::unordered_map<uint64_t, AtlasSearchFiberSnapshot>& snapshotsById)
{
    const auto& source = atlasSearchSnapshotFor(snapshotsById, result.sourceFiberId);
    const auto& target = atlasSearchSnapshotFor(snapshotsById, result.targetFiberId);
    return vc3d::line_annotation::firstFiberDisplaysAsH(
        source.hvClassification,
        source.manualHvTag,
        target.hvClassification,
        target.manualHvTag,
        atlasSearchTieBreakFiberId(source, result.sourceFiberId) <
            atlasSearchTieBreakFiberId(target, result.targetFiberId));
}

AtlasSearchSigningContext prepareAtlasSearchSigningContext(
    const std::vector<vc::atlas::FiberIntersectionResult>& seedOrderedResults,
    const std::unordered_map<uint64_t, AtlasSearchFiberSnapshot>& snapshotsById,
    const std::optional<std::filesystem::path>& atlasDir,
    const vc::lasagna::LasagnaDataset& dataset,
    vc::lasagna::LasagnaNormalSampler& sampler,
    const vc::atlas::FiberIntersectionProgressCallback& progressCallback)
{
    AtlasSearchSigningContext context;
    if (seedOrderedResults.empty()) {
        return context;
    }

    if (atlasDir) {
        constexpr std::size_t total = 2;
        if (progressCallback) {
            progressCallback(vc::atlas::AtlasSearchProgressPhase::PrepareSigningSurface, 0, total);
        }
        context.atlas = vc::atlas::Atlas::load(*atlasDir);
        if (progressCallback) {
            progressCallback(vc::atlas::AtlasSearchProgressPhase::PrepareSigningSurface, 1, total);
        }
        context.baseMapping = vc::atlas::loadAtlasBaseMappingContext(*atlasDir, context.atlas);
        context.haveAtlas = true;
        context.orientationSource = QStringLiteral("selected_atlas");
        if (progressCallback) {
            progressCallback(vc::atlas::AtlasSearchProgressPhase::PrepareSigningSurface, total, total);
        }
        return context;
    }

    constexpr std::size_t total = 3;
    if (progressCallback) {
        progressCallback(vc::atlas::AtlasSearchProgressPhase::PrepareSigningSurface, 0, total);
    }
    const auto initShellDir = vc::atlas::initShellDirectoryFromManifest(dataset.manifest());
    auto surfaces = vc::atlas::loadInitShellCandidates(initShellDir);
    if (progressCallback) {
        progressCallback(vc::atlas::AtlasSearchProgressPhase::PrepareSigningSurface, 1, total);
    }
    std::vector<std::shared_ptr<QuadSurface>> shellSurfaces;
    shellSurfaces.reserve(surfaces.size());
    for (const auto& candidate : surfaces) {
        if (candidate.surface) {
            shellSurfaces.push_back(candidate.surface);
        }
    }
    if (shellSurfaces.empty()) {
        throw std::runtime_error("Lasagna init shells contain no loadable base surfaces");
    }
    SurfacePatchIndex shellIndex;
    shellIndex.rebuild(shellSurfaces);
    if (progressCallback) {
        progressCallback(vc::atlas::AtlasSearchProgressPhase::PrepareSigningSurface, 2, total);
    }

    const auto& seedResult = seedOrderedResults.front();
    const bool seedSourceIsH = atlasSearchSourceDisplaysAsH(seedResult, snapshotsById);
    const auto& seedSnapshot = atlasSearchSnapshotFor(
        snapshotsById,
        seedSourceIsH ? seedResult.sourceFiberId : seedResult.targetFiberId);
    const vc::atlas::FiberInput seedInput =
        fiberInputFromSnapshot(seedSnapshot.fiberPath, seedSnapshot.fiber);
    const auto selection =
        vc::atlas::selectBaseSurfaceBySeedRay(seedInput, surfaces, shellIndex, sampler);
    if (selection.surfaceIndex < 0 ||
        selection.surfaceIndex >= static_cast<int>(surfaces.size()) ||
        !surfaces[static_cast<size_t>(selection.surfaceIndex)].surface) {
        throw std::runtime_error("Lasagna init-shell base selection did not return a surface");
    }
    context.baseMapping = vc::atlas::atlasBaseMappingContextFromSurface(
        surfaces[static_cast<size_t>(selection.surfaceIndex)].surface);
    context.orientationSource = QStringLiteral("init_shell:%1")
        .arg(QString::fromStdString(selection.surfaceName));
    if (progressCallback) {
        progressCallback(vc::atlas::AtlasSearchProgressPhase::PrepareSigningSurface, total, total);
    }
    return context;
}

std::vector<AtlasSearchSignedResult> signAtlasSearchResults(
    const std::vector<vc::atlas::FiberIntersectionResult>& results,
    const std::unordered_map<uint64_t, AtlasSearchFiberSnapshot>& snapshotsById,
    AtlasSearchSigningContext& context,
    vc::lasagna::LasagnaNormalSampler& sampler,
    std::size_t& skippedSigningCount)
{
    std::vector<AtlasSearchSignedResult> signedResults;
    signedResults.reserve(results.size());
    skippedSigningCount = 0;
    if (results.empty()) {
        return signedResults;
    }
    if (!context.baseMapping.baseSurface || !context.baseMapping.baseIndex) {
        skippedSigningCount = results.size();
        qWarning().noquote() << QStringLiteral("[atlas-search] signing_skipped_all reason=no signing base surface/index");
        return signedResults;
    }

    std::map<uint64_t, vc::atlas::FiberMapping> temporaryMappings;
    std::map<uint64_t, std::string> mappingFailures;
    auto mappingFor = [&](uint64_t runtimeFiberId) -> const vc::atlas::FiberMapping& {
        const auto& snapshot = atlasSearchSnapshotFor(snapshotsById, runtimeFiberId);
        if (context.haveAtlas) {
            if (const auto* mapping = findAtlasMappingForPath(context.atlas, snapshot.fiberPath)) {
                return *mapping;
            }
        }
        if (const auto failed = mappingFailures.find(runtimeFiberId);
            failed != mappingFailures.end()) {
            throw std::runtime_error(failed->second);
        }
        const auto cached = temporaryMappings.find(runtimeFiberId);
        if (cached != temporaryMappings.end()) {
            return cached->second;
        }
        try {
            const vc::atlas::FiberInput input =
                fiberInputFromSnapshot(snapshot.fiberPath, snapshot.fiber);
            auto mapping = vc::atlas::mapFiberToBaseSurface(
                input,
                *context.baseMapping.baseSurface,
                *context.baseMapping.baseIndex,
                sampler);
            const auto [it, inserted] =
                temporaryMappings.emplace(runtimeFiberId, std::move(mapping));
            (void)inserted;
            return it->second;
        } catch (const std::exception& ex) {
            const std::string reason = ex.what();
            mappingFailures.emplace(runtimeFiberId, reason);
            throw std::runtime_error(reason);
        }
    };

    for (const auto& result : results) {
        try {
            const auto& sourceSnapshot = atlasSearchSnapshotFor(snapshotsById, result.sourceFiberId);
            const auto& targetSnapshot = atlasSearchSnapshotFor(snapshotsById, result.targetFiberId);
            const auto sourceSample = vc3d::fiber_slice::samplePolylineAtArclength(
                linePointsFromPolyline(sourceSnapshot.fiber),
                result.sourceArclength);
            const auto targetSample = vc3d::fiber_slice::samplePolylineAtArclength(
                linePointsFromPolyline(targetSnapshot.fiber),
                result.targetArclength);
            if (!sourceSample.valid || !targetSample.valid) {
                throw std::runtime_error("Could not sample atlas search intersection arclengths");
            }

            const bool sourceIsH = atlasSearchSourceDisplaysAsH(result, snapshotsById);
            const auto display = vc::atlas::signedAtlasSearchWindingDisplay(
                result.windingDistance,
                sourceIsH,
                sourceSample.linePosition,
                targetSample.linePosition,
                result.sourcePoint,
                result.targetPoint,
                mappingFor(result.sourceFiberId),
                mappingFor(result.targetFiberId),
                *context.baseMapping.baseSurface);
            signedResults.push_back(AtlasSearchSignedResult{
                result,
                AtlasSearchSignedWindingRow{
                    display.signedWindingDistance,
                    sourceIsH ? result.sourceFiberId : result.targetFiberId,
                    sourceIsH ? result.targetFiberId : result.sourceFiberId,
                    context.orientationSource,
                },
            });
        } catch (const std::exception& ex) {
            ++skippedSigningCount;
            qWarning().noquote() << QStringLiteral("[atlas-search] signing_skipped source_id=%1 target_id=%2 raw_winding=%3 reason=%4 orientation_source=%5")
                .arg(QString::number(result.sourceFiberId),
                     QString::number(result.targetFiberId),
                     QString::number(result.windingDistance, 'g', 12),
                     QString::fromStdString(ex.what()),
                     context.orientationSource);
        }
    }
    return signedResults;
}

AtlasSearchWorkerResult buildSignedAtlasSearchResults(
    std::vector<vc::atlas::FiberIntersectionResult> rawResults,
    const std::unordered_map<uint64_t, AtlasSearchFiberSnapshot>& snapshotsById,
    const std::optional<std::filesystem::path>& atlasDir,
    const std::filesystem::path& lasagnaManifestPath,
    double workingToBaseScale,
    const vc::atlas::FiberIntersectionProgressCallback& progressCallback,
    bool debugSearch)
{
    AtlasSearchWorkerResult workerResult;
    workerResult.rawResultCount = rawResults.size();
    std::sort(rawResults.begin(), rawResults.end(), [&](const auto& a, const auto& b) {
        const double da = std::abs(a.windingDistance);
        const double db = std::abs(b.windingDistance);
        if (da != db) return da < db;
        if (a.sourceFiberId != b.sourceFiberId) return a.sourceFiberId < b.sourceFiberId;
        return a.targetFiberId < b.targetFiberId;
    });
    if (rawResults.empty()) {
        return workerResult;
    }

    try {
        vc::lasagna::LasagnaDataset dataset =
            vc::lasagna::LasagnaDataset::open(
                lasagnaManifestPath, {workingToBaseScale});
        vc::lasagna::LasagnaNormalSampler sampler(dataset);
        auto context = prepareAtlasSearchSigningContext(
            rawResults, snapshotsById, atlasDir, dataset, sampler, progressCallback);
        std::vector<AtlasSearchSignedResult> signedResults = signAtlasSearchResults(
            rawResults, snapshotsById, context, sampler, workerResult.skippedSigningCount);
        std::sort(signedResults.begin(), signedResults.end(), [](const auto& a, const auto& b) {
            const double da = a.row.signedWinding;
            const double db = b.row.signedWinding;
            if (da != db) return da < db;
            const double rawA = std::abs(a.result.windingDistance);
            const double rawB = std::abs(b.result.windingDistance);
            if (rawA != rawB) return rawA < rawB;
            if (a.result.sourceFiberId != b.result.sourceFiberId) {
                return a.result.sourceFiberId < b.result.sourceFiberId;
            }
            return a.result.targetFiberId < b.result.targetFiberId;
        });
        workerResult.results.reserve(signedResults.size());
        workerResult.signedWindings.reserve(signedResults.size());
        for (int row = 0; row < static_cast<int>(signedResults.size()); ++row) {
            const auto& signedResult = signedResults[static_cast<size_t>(row)];
            if (debugSearch) {
                qInfo().noquote() << QStringLiteral("[atlas-search] signed_result row=%1 source_id=%2 target_id=%3 raw_winding=%4 signed_winding=%5 h_id=%6 v_id=%7 orientation_source=%8")
                    .arg(row)
                    .arg(QString::number(signedResult.result.sourceFiberId),
                         QString::number(signedResult.result.targetFiberId),
                         QString::number(signedResult.result.windingDistance, 'g', 12),
                         QString::number(signedResult.row.signedWinding, 'g', 12),
                         QString::number(signedResult.row.hFiberId),
                         QString::number(signedResult.row.vFiberId),
                         signedResult.row.orientationSource);
            }
            workerResult.results.push_back(signedResult.result);
            workerResult.signedWindings.push_back(signedResult.row.signedWinding);
        }
    } catch (const std::exception& ex) {
        workerResult.skippedSigningCount = rawResults.size();
        qWarning().noquote() << QStringLiteral("[atlas-search] signing_skipped_all count=%1 reason=%2 atlas_mode=%3")
            .arg(static_cast<qulonglong>(rawResults.size()))
            .arg(QString::fromStdString(ex.what()),
                 atlasDir ? QStringLiteral("selected_atlas") : QStringLiteral("init_shell"));
        if (progressCallback) {
            progressCallback(vc::atlas::AtlasSearchProgressPhase::PrepareSigningSurface, 1, 1);
        }
    }
    return workerResult;
}

class FunctionEventFilter final : public QObject
{
public:
    using Handler = std::function<bool(QObject*, QEvent*)>;

    FunctionEventFilter(Handler handler, QObject* parent)
        : QObject(parent)
        , handler_(std::move(handler))
    {
    }

protected:
    bool eventFilter(QObject* watched, QEvent* event) override
    {
        return handler_ ? handler_(watched, event) : false;
    }

private:
    Handler handler_;
};

VolumeViewerBase* baseViewerFromWidget(QWidget* widget)
{
    if (auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(widget)) {
        return chunkedViewer;
    }
    return nullptr;
}

bool isChunkedViewer(VolumeViewerBase* viewer)
{
    return viewer && qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject());
}

bool moveOnSurfaceChangeEnabled()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    return settings.value(vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE,
                          vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toBool();
}

void centerViewerOnSurfacePointForNavigation(VolumeViewerBase* viewer, const cv::Vec2f& position)
{
    if (!viewer) {
        return;
    }
    viewer->centerOnSurfacePoint(position, !isChunkedViewer(viewer));
}

std::optional<cv::Vec2f> atlasControlGridToSurface(VolumeViewerBase* viewer,
                                                   const AtlasControlPointResult& point)
{
    auto* quad = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    if (!quad ||
        !std::isfinite(point.modelH) ||
        !std::isfinite(point.modelW)) {
        return std::nullopt;
    }
    const cv::Vec2f scale = quad->scale();
    if (std::abs(scale[0]) < 1e-6f || std::abs(scale[1]) < 1e-6f) {
        return std::nullopt;
    }
    const float modelH = point.snapValid && std::isfinite(point.snapModelH)
        ? point.snapModelH
        : point.modelH;
    const float modelW = point.snapValid && std::isfinite(point.snapModelW)
        ? point.snapModelW
        : point.modelW;
    const cv::Vec3f center = quad->center();
    const cv::Vec2f surface{
        modelW / scale[0] - center[0],
        modelH / scale[1] - center[1],
    };
    if (!std::isfinite(surface[0]) || !std::isfinite(surface[1])) {
        return std::nullopt;
    }
    return surface;
}

struct SurfaceFocusPoint {
    cv::Vec3f world{0, 0, 0};
    cv::Vec3f ptr{0, 0, 0};
    int row = -1;
    int col = -1;
};

bool isValidSurfacePoint(const cv::Vec3f& point)
{
    return point[0] != -1.0f && point[1] != -1.0f && point[2] != -1.0f
        && std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

bool isFiniteVec3(const cv::Vec3f& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

std::optional<SurfaceFocusPoint> focusPointAtGrid(QuadSurface& surface, int row, int col)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || row < 0 || row >= points->rows || col < 0 || col >= points->cols) {
        return std::nullopt;
    }
    if (row <= 0 || col <= 0 || row >= points->rows - 1 || col >= points->cols - 1) {
        return std::nullopt;
    }
    if (!surface.isQuadValid(row, col)) {
        return std::nullopt;
    }
    const cv::Vec3f normal = surface.gridNormal(row, col);
    if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) || !std::isfinite(normal[2])) {
        return std::nullopt;
    }

    const cv::Vec3f& point = (*points)(row, col);
    if (!isValidSurfacePoint(point)) {
        return std::nullopt;
    }

    const cv::Vec3f center = surface.center();
    const cv::Vec2f scale = surface.scale();
    return SurfaceFocusPoint{
        point,
        cv::Vec3f(static_cast<float>(col) - center[0] * scale[0],
                  static_cast<float>(row) - center[1] * scale[1],
                  0.0f),
        row,
        col,
    };
}

std::optional<SurfaceFocusPoint> findSegmentFocusPoint(QuadSurface& surface)
{
    surface.ensureLoaded();
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    const cv::Vec2f centerGrid = surface.ptrToGrid({0, 0, 0});
    const int centerRow = std::clamp(static_cast<int>(std::lround(centerGrid[1])), 0, points->rows - 1);
    const int centerCol = std::clamp(static_cast<int>(std::lround(centerGrid[0])), 0, points->cols - 1);

    if (auto focus = focusPointAtGrid(surface, centerRow, centerCol)) {
        return focus;
    }

    const int maxHorizontalRadius = std::max(centerCol, points->cols - 1 - centerCol);
    for (int radius = 1; radius <= maxHorizontalRadius; ++radius) {
        if (auto focus = focusPointAtGrid(surface, centerRow, centerCol - radius)) {
            return focus;
        }
        if (auto focus = focusPointAtGrid(surface, centerRow, centerCol + radius)) {
            return focus;
        }
    }

    const int maxRadius = std::max({centerRow, centerCol,
                                    points->rows - 1 - centerRow,
                                    points->cols - 1 - centerCol});
    for (int radius = 1; radius <= maxRadius; ++radius) {
        const int rowMin = std::max(0, centerRow - radius);
        const int rowMax = std::min(points->rows - 1, centerRow + radius);
        const int colMin = std::max(0, centerCol - radius);
        const int colMax = std::min(points->cols - 1, centerCol + radius);

        for (int col = colMin; col <= colMax; ++col) {
            if (auto focus = focusPointAtGrid(surface, rowMin, col)) {
                return focus;
            }
            if (rowMax != rowMin) {
                if (auto focus = focusPointAtGrid(surface, rowMax, col)) {
                    return focus;
                }
            }
        }
        for (int row = rowMin + 1; row < rowMax; ++row) {
            if (auto focus = focusPointAtGrid(surface, row, colMin)) {
                return focus;
            }
            if (colMax != colMin) {
                if (auto focus = focusPointAtGrid(surface, row, colMax)) {
                    return focus;
                }
            }
        }
    }

    return std::nullopt;
}

void ensureDockWidgetFeatures(QDockWidget* dock)
{
    if (!dock) {
        return;
    }

    auto features = dock->features();
    features |= QDockWidget::DockWidgetMovable;
    features |= QDockWidget::DockWidgetFloatable;
    features |= QDockWidget::DockWidgetClosable;
    dock->setFeatures(features);
}

QString normalGridDirectoryForVolumePkg(const std::shared_ptr<VolumePkg>& pkg,
                                        const std::string& loadedVolumeId,
                                        QString* checkedPath)
{
    if (checkedPath) {
        *checkedPath = QString();
    }

    if (!pkg) {
        qCInfo(lcSegGrowth) << "Normal grid lookup skipped (no volume package loaded)";
        return QString();
    }

    auto paths = pkg->normalGridPaths();
    if (paths.empty()) {
        qCInfo(lcSegGrowth) << "Normal grid lookup: no normal_grids entries in project";
        return QString();
    }

    // Coordinate-tagged entries apply only to an exact coordinate view;
    // untagged entries keep the legacy behavior for untagged volumes.
    const auto catalogIds =
        openDataCatalogVolumeIdCandidates(*pkg, loadedVolumeId);
    const auto coordinateSpace =
        openDataCoordinateSpaceForLoadedVolume(*pkg, loadedVolumeId);
    const std::string coordinateTag = coordinateSpace.empty()
        ? std::string{}
        : "vc-open-data-coordinate-space:" + coordinateSpace;
    constexpr std::string_view volumeIdTagPrefix = "vc-open-data-volume-id:";

    QString untaggedFallback;
    for (const auto& path : paths) {
        const auto pathStr = path.string();
        const QString candidateStr = QString::fromStdString(pathStr);
        const vc::project::Entry* owningEntry = nullptr;
        for (const auto& entry : pkg->normalGridEntries()) {
            if (pathStr == entry.location ||
                pathStr.rfind(entry.location + "/", 0) == 0) {
                owningEntry = &entry;
                break;
            }
        }

        std::vector<QString> entryVolumeIds;
        if (owningEntry) {
            if (!coordinateTag.empty()) {
                if (std::find(owningEntry->tags.begin(), owningEntry->tags.end(),
                              coordinateTag) != owningEntry->tags.end()) {
                    if (checkedPath) *checkedPath = candidateStr;
                    qCInfo(lcSegGrowth) << "Normal grid resolved by coordinate space to"
                                        << candidateStr;
                    return candidateStr;
                }
                continue;
            }
            for (const auto& tag : owningEntry->tags) {
                if (tag.rfind(volumeIdTagPrefix, 0) == 0) {
                    entryVolumeIds.push_back(
                        QString::fromStdString(tag.substr(volumeIdTagPrefix.size())));
                }
            }
        }

        if (entryVolumeIds.empty()) {
            if (untaggedFallback.isEmpty()) {
                untaggedFallback = candidateStr;
            }
            continue;
        }
        const bool matchesVolume = std::any_of(
            entryVolumeIds.begin(), entryVolumeIds.end(), [&](const QString& id) {
                return std::find(catalogIds.begin(), catalogIds.end(), id) !=
                       catalogIds.end();
            });
        if (matchesVolume) {
            if (checkedPath) *checkedPath = candidateStr;
            qCInfo(lcSegGrowth) << "Normal grid resolved to" << candidateStr
                                << "for volume" << QString::fromStdString(loadedVolumeId);
            return candidateStr;
        }
    }

    if (coordinateTag.empty() && !untaggedFallback.isEmpty()) {
        if (checkedPath) *checkedPath = untaggedFallback;
        qCInfo(lcSegGrowth) << "Normal grid resolved to" << untaggedFallback;
        return untaggedFallback;
    }

    qCInfo(lcSegGrowth) << "Normal grid lookup: no store paired with volume"
                        << QString::fromStdString(loadedVolumeId);
    return QString();
}

QStringList normal3dZarrCandidatesForVolumePkg(const std::shared_ptr<VolumePkg>& pkg,
                                               QString* hint)
{
    if (hint) {
        *hint = QString();
    }
    if (!pkg) {
        if (hint) {
            *hint = QObject::tr("Normal3D lookup skipped (no volume package loaded)");
        }
        return {};
    }
    auto paths = pkg->normal3dZarrPaths();
    QStringList candidates;
    for (const auto& p : paths) candidates.push_back(QString::fromStdString(p.string()));
    candidates.sort();
    if (hint) {
        *hint = candidates.isEmpty()
            ? QObject::tr("No volumes tagged 'normal3d' in project")
            : QObject::tr("%1 normal3d zarr(s) tagged").arg(candidates.size());
    }
    return candidates;
}

QString absoluteSegmentPathForClipboard(const std::filesystem::path& segmentPath,
                                        const std::shared_ptr<VolumePkg>& pkg)
{
    auto path = segmentPath;
    if (!path.is_absolute() && pkg) {
        const auto projectPath = pkg->path();
        const auto projectDir = projectPath.has_parent_path()
            ? projectPath.parent_path()
            : std::filesystem::current_path();
        path = projectDir / path;
    }

    std::error_code ec;
    const auto absolutePath = std::filesystem::absolute(path, ec);
    if (!ec) {
        path = absolutePath;
    }
    return QString::fromStdString(path.lexically_normal().string());
}

QDockWidget* createAtlasOverviewDock(QWidget* parent)
{
    auto* dock = new QDockWidget(QObject::tr("Atlas Overview"), parent);
    auto* content = new QWidget(dock);
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(6);

    auto* actions = new QHBoxLayout();
    actions->setContentsMargins(0, 0, 0, 0);
    actions->setSpacing(6);

    auto* remap = new QPushButton(QObject::tr("Remap"), content);
    remap->setObjectName(QStringLiteral("atlasRemapButton"));
    remap->setToolTip(QObject::tr(
        "Rebuild atlas fiber mappings from the saved source fibers using the selected Lasagna dataset."));
    remap->setEnabled(false);
    actions->addWidget(remap);

    auto* rankSnap = new QPushButton(QObject::tr("Rank via Fit-Service"), content);
    rankSnap->setObjectName(QStringLiteral("atlasRankSnapButton"));
    rankSnap->setToolTip(QObject::tr(
        "Run atlas pred-snap ranking through the Lasagna fit-service job queue."));
    rankSnap->setEnabled(false);
    actions->addWidget(rankSnap);

    layout->addLayout(actions);

    auto* atlasTree = new QTreeWidget(content);
    atlasTree->setObjectName(QStringLiteral("atlasOverviewTree"));
    atlasTree->setColumnCount(2);
    atlasTree->setHeaderLabels({QObject::tr("Atlas"), QObject::tr("Value")});
    atlasTree->setAlternatingRowColors(true);
    atlasTree->setRootIsDecorated(true);
    atlasTree->setSelectionMode(QAbstractItemView::SingleSelection);
    atlasTree->header()->setStretchLastSection(true);
    layout->addWidget(atlasTree, 1);

    dock->setWidget(content);
    return dock;
}

QDockWidget* createAtlasFiberDock(QWidget* parent)
{
    auto* dock = new QDockWidget(QObject::tr("Atlas Fibers"), parent);
    auto* content = new QWidget(dock);
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(6);

    auto* current = new QLabel(QObject::tr("No atlas selected"), content);
    current->setObjectName(QStringLiteral("atlasFiberCurrentLabel"));
    current->setWordWrap(true);
    layout->addWidget(current);

    auto* optimize = new QPushButton(QObject::tr("Optimize snap cands"), content);
    optimize->setObjectName(QStringLiteral("atlasOptimizeSnapCandidatesButton"));
    optimize->setEnabled(false);
    layout->addWidget(optimize);

    auto* tree = new QTreeWidget(content);
    tree->setObjectName(QStringLiteral("atlasFiberTree"));
    tree->setColumnCount(9);
    tree->setHeaderLabels({
        QObject::tr("Fiber"),
        QObject::tr("Source"),
        QObject::tr("Anchor dist"),
        QObject::tr("Snap"),
        QObject::tr("Snap wind"),
        QObject::tr("Snap XYZ"),
        QObject::tr("Status"),
        QObject::tr("Reason"),
        QObject::tr("Saved fiber"),
    });
    tree->setAlternatingRowColors(true);
    tree->setSelectionBehavior(QAbstractItemView::SelectRows);
    tree->setSelectionMode(QAbstractItemView::ExtendedSelection);
    tree->setEditTriggers(QAbstractItemView::NoEditTriggers);
    tree->setSortingEnabled(false);
    tree->setRootIsDecorated(true);
    tree->setContextMenuPolicy(Qt::CustomContextMenu);
    tree->header()->setStretchLastSection(false);
    tree->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
    tree->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    layout->addWidget(tree, 1);

    dock->setWidget(content);
    return dock;
}

QDockWidget* createAtlasSearchDock(QWidget* parent)
{
    auto* dock = new QDockWidget(QObject::tr("Atlas Object Search"), parent);
    auto* content = new QWidget(dock);
    content->setObjectName(QStringLiteral("atlasSearchContent"));
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(6);

    auto* current = new QLabel(QObject::tr("No atlas selected"), content);
    current->setObjectName(QStringLiteral("atlasSearchCurrentLabel"));
    current->setWordWrap(true);
    layout->addWidget(current);

    auto* form = new QFormLayout();
    form->setContentsMargins(0, 0, 0, 0);
    auto* searchType = new QComboBox(content);
    searchType->setObjectName(QStringLiteral("atlasSearchTypeCombo"));
    searchType->addItem(QObject::tr("Atlas fibers -> non-atlas fibers"),
                        ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS);
    searchType->addItem(QObject::tr("Non-atlas fibers only"),
                        ATLAS_SEARCH_MODE_NON_ATLAS_ONLY);
    searchType->setToolTip(QObject::tr(
        "Choose whether to search from mapped atlas fibers to non-atlas fibers, or only between non-atlas fibers."));
    form->addRow(QObject::tr("Search type"), searchType);

    auto* tagFilter = new QLineEdit(content);
    tagFilter->setObjectName(QStringLiteral("atlasSearchTagFilterEdit"));
    tagFilter->setPlaceholderText(QObject::tr("tag-a, tag-b"));
    tagFilter->setClearButtonEnabled(true);
    tagFilter->setToolTip(QObject::tr(
        "Only consider fibers that have all listed tags. Separate tags with commas or spaces."));
    form->addRow(QObject::tr("Fiber tags"), tagFilter);

    auto* excludeTagFilter = new QLineEdit(content);
    excludeTagFilter->setObjectName(QStringLiteral("atlasSearchExcludeTagFilterEdit"));
    excludeTagFilter->setPlaceholderText(QObject::tr("bad, skip"));
    excludeTagFilter->setClearButtonEnabled(true);
    excludeTagFilter->setToolTip(QObject::tr(
        "Exclude fibers that have any listed tag. Separate tags with commas or spaces."));
    form->addRow(QObject::tr("Exclude tags"), excludeTagFilter);

    auto* maxDistance = new QDoubleSpinBox(content);
    maxDistance->setObjectName(QStringLiteral("atlasSearchMaxDistanceSpin"));
    maxDistance->setRange(1.0, 100000.0);
    maxDistance->setDecimals(2);
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        maxDistance->setValue(settings.value(
            vc3d::settings::atlas::SEARCH_MAX_DISTANCE,
            vc::atlas::FiberIntersectionBroadPhaseOptions{}.maxDistance).toDouble());
    }
    maxDistance->setSuffix(QObject::tr(" vx"));
    maxDistance->setToolTip(QObject::tr(
        "Broad-phase segment radius in original voxel coordinates. Segment pairs farther apart than this are not sent to Ceres."));
    form->addRow(QObject::tr("Max straight distance"), maxDistance);

    auto* groupByFiber = new QCheckBox(QObject::tr("Group results by fiber"), content);
    groupByFiber->setObjectName(QStringLiteral("atlasSearchGroupByFiberCheck"));
    groupByFiber->setChecked(true);
    groupByFiber->setToolTip(QObject::tr(
        "Show each fiber as a group with all matching fiber pairs beneath it."));
    form->addRow(QString(), groupByFiber);
    layout->addLayout(form);

    auto* buttons = new QHBoxLayout();
    auto* run = new QPushButton(QObject::tr("Search"), content);
    run->setObjectName(QStringLiteral("atlasSearchRunButton"));
    auto* cancel = new QPushButton(QObject::tr("Cancel"), content);
    cancel->setObjectName(QStringLiteral("atlasSearchCancelButton"));
    cancel->setEnabled(false);
    buttons->addWidget(run);
    buttons->addWidget(cancel);
    buttons->addStretch(1);
    layout->addLayout(buttons);

    auto* progress = new QProgressBar(content);
    progress->setObjectName(QStringLiteral("atlasSearchProgressBar"));
    progress->setRange(0, 100);
    progress->setValue(0);
    layout->addWidget(progress);

    auto* tree = new QTreeWidget(content);
    tree->setObjectName(QStringLiteral("atlasSearchResultTree"));
    tree->setColumnCount(5);
    tree->setHeaderLabels({
        QObject::tr("Distance (windings)"),
        QObject::tr("Source fiber"),
        QObject::tr("Target fiber"),
        QObject::tr("Src idx"),
        QObject::tr("Tgt idx"),
    });
    tree->setEditTriggers(QAbstractItemView::NoEditTriggers);
    tree->setSelectionBehavior(QAbstractItemView::SelectRows);
    tree->setSelectionMode(QAbstractItemView::ExtendedSelection);
    tree->setMouseTracking(true);
    tree->setAlternatingRowColors(true);
    tree->setRootIsDecorated(true);
    tree->setContextMenuPolicy(Qt::CustomContextMenu);
    tree->header()->setStretchLastSection(true);
    layout->addWidget(tree, 1);

    dock->setWidget(content);
    return dock;
}

constexpr float kEpsilon = 1e-6f;

cv::Vec3f projectVectorOntoPlane(const cv::Vec3f& v, const cv::Vec3f& normal)
{
    const float dot = v.dot(normal);
    return v - normal * dot;
}

cv::Vec3f normalizeOrZero(const cv::Vec3f& v)
{
    const float magnitude = cv::norm(v);
    if (magnitude <= kEpsilon) {
        return cv::Vec3f(0.0f, 0.0f, 0.0f);
    }
    return v * (1.0f / magnitude);
}

cv::Vec3f crossProduct(const cv::Vec3f& a, const cv::Vec3f& b)
{
    return cv::Vec3f(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]);
}

float signedAngleBetween(const cv::Vec3f& from, const cv::Vec3f& to, const cv::Vec3f& axis)
{
    cv::Vec3f fromNorm = normalizeOrZero(from);
    cv::Vec3f toNorm = normalizeOrZero(to);
    if (cv::norm(fromNorm) <= kEpsilon || cv::norm(toNorm) <= kEpsilon) {
        return 0.0f;
    }

    float dot = fromNorm.dot(toNorm);
    dot = std::clamp(dot, -1.0f, 1.0f);
    cv::Vec3f cross = crossProduct(fromNorm, toNorm);
    float angle = std::atan2(cv::norm(cross), dot);
    float sign = cross.dot(axis) >= 0.0f ? 1.0f : -1.0f;
    return angle * sign;
}

} // namespace

// Dark mode detection - works on all Qt 6.x versions
static bool isDarkMode() {
#if QT_VERSION >= QT_VERSION_CHECK(6, 5, 0)
    if (QGuiApplication::styleHints()->colorScheme() == Qt::ColorScheme::Dark)
        return true;
#endif
    // Fallback: check system palette brightness
    const auto windowColor = QGuiApplication::palette().color(QPalette::Window);
    return windowColor.lightness() < 128;
}

// Apply a consistent dark palette application-wide
static void applyDarkPalette() {
    QPalette p;
    p.setColor(QPalette::Window, QColor(53, 53, 53));
    p.setColor(QPalette::WindowText, Qt::white);
    p.setColor(QPalette::Base, QColor(42, 42, 42));
    p.setColor(QPalette::AlternateBase, QColor(66, 66, 66));
    p.setColor(QPalette::ToolTipBase, QColor(53, 53, 53));
    p.setColor(QPalette::ToolTipText, Qt::white);
    p.setColor(QPalette::Text, Qt::white);
    p.setColor(QPalette::Button, QColor(53, 53, 53));
    p.setColor(QPalette::ButtonText, Qt::white);
    p.setColor(QPalette::BrightText, Qt::red);
    p.setColor(QPalette::Link, QColor(42, 130, 218));
    p.setColor(QPalette::Highlight, QColor(42, 130, 218));
    p.setColor(QPalette::HighlightedText, Qt::black);
    p.setColor(QPalette::Disabled, QPalette::Text, QColor(127, 127, 127));
    p.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(127, 127, 127));
    QApplication::setPalette(p);
}

static QString windowStateScreenSignature()
{
    QStringList parts;
    parts << qga::platformName();
    const auto screens = qga::screens();
    parts << QString::number(screens.size());
    for (const QScreen* screen : screens) {
        if (!screen) {
            continue;
        }
        const QRect geom = screen->geometry();
        const qreal dpr = screen->devicePixelRatio();
        const QString name = screen->name().isEmpty() ? QStringLiteral("screen") : screen->name();
        parts << QString("%1:%2x%3+%4+%5@%6")
                     .arg(name)
                     .arg(geom.width())
                     .arg(geom.height())
                     .arg(geom.x())
                     .arg(geom.y())
                     .arg(dpr, 0, 'f', 2);
    }
    return parts.join("|");
}

static QString windowStateQtVersion()
{
    return QString::fromUtf8(qVersion());
}

static QString windowStateAppVersion()
{
    return QString::fromStdString(ProjectInfo::VersionString());
}

static void writeWindowStateMeta(QSettings& settings,
                                 const QString& screenSignature,
                                 const QString& qtVersion,
                                 const QString& appVersion)
{
    settings.setValue(vc3d::settings::window::STATE_META_SCREEN_SIGNATURE, screenSignature);
    settings.setValue(vc3d::settings::window::STATE_META_QT_VERSION, qtVersion);
    settings.setValue(vc3d::settings::window::STATE_META_APP_VERSION, appVersion);
}

static bool windowStateMetaMatches(const QSettings& settings,
                                   const QString& screenSignature,
                                   const QString& qtVersion,
                                   const QString& appVersion)
{
    const QString savedSignature =
        settings.value(vc3d::settings::window::STATE_META_SCREEN_SIGNATURE).toString();
    const QString savedQtVersion =
        settings.value(vc3d::settings::window::STATE_META_QT_VERSION).toString();
    const QString savedAppVersion =
        settings.value(vc3d::settings::window::STATE_META_APP_VERSION).toString();

    if (savedSignature.isEmpty() || savedQtVersion.isEmpty()) {
        return false;
    }

    Q_UNUSED(savedAppVersion);
    Q_UNUSED(appVersion);
    return savedSignature == screenSignature
        && savedQtVersion == qtVersion;
}

// Constructor
CWindow::CWindow(size_t cacheSizeGB, RenderBenchOptions benchOptions) :
    _cmdRunner(nullptr),
    _benchOptions(std::move(benchOptions)),
    _seedingWidget(nullptr),
    _point_collection_widget(nullptr)
{
    // Initialize timer for debounced window state saving (500ms delay)
    _windowStateSaveTimer = new QTimer(this);
    _windowStateSaveTimer->setSingleShot(true);
    _windowStateSaveTimer->setInterval(500);
    connect(_windowStateSaveTimer, &QTimer::timeout, this, &CWindow::saveWindowState);

    const QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    _mirrorCursorToSegmentation = settings.value(vc3d::settings::viewer::MIRROR_CURSOR_TO_SEGMENTATION,
                                                  vc3d::settings::viewer::MIRROR_CURSOR_TO_SEGMENTATION_DEFAULT).toBool();
    setWindowIcon(QPixmap(":/images/logo.png"));
    ui.setupUi(this);
    QWidget* segmentCentralWidget = takeCentralWidget();
    auto* segmentWorkspaceWindow = new DockMenuMainWindow(this);
    segmentWorkspaceWindow->setDockMenuBuilder([this](QMenu* menu) {
        populateDockToggleMenu(menu);
    });
    _segmentWorkspaceWindow = segmentWorkspaceWindow;
    _segmentWorkspaceWindow->setObjectName(QStringLiteral("segmentWorkspaceWindow"));
    _segmentWorkspaceWindow->setDockOptions(dockOptions());
    _statusDockPanelHost = new StatusDockPanelHost(segmentCentralWidget, _segmentWorkspaceWindow);
    _segmentWorkspaceWindow->setCentralWidget(_statusDockPanelHost);

    auto moveExistingDockToSegment = [this](QDockWidget* dock, Qt::DockWidgetArea area) {
        if (!dock || !_segmentWorkspaceWindow) {
            return;
        }
        removeDockWidget(dock);
        _segmentWorkspaceWindow->addDockWidget(area, dock);
    };
    moveExistingDockToSegment(ui.dockWidgetVolumes, Qt::LeftDockWidgetArea);
    moveExistingDockToSegment(ui.dockWidgetSegmentation, Qt::RightDockWidgetArea);
    moveExistingDockToSegment(ui.dockWidgetDistanceTransform, Qt::RightDockWidgetArea);
    moveExistingDockToSegment(ui.dockWidgetViewerControls, Qt::LeftDockWidgetArea);
    for (QDockWidget* dock : {ui.dockWidgetNormalVis,
                              ui.dockWidgetView,
                              ui.dockWidgetOverlay,
                              ui.dockWidgetRenderSettings,
                              ui.dockWidgetComposite}) {
        moveExistingDockToSegment(dock, Qt::LeftDockWidgetArea);
    }

    _lasagnaWorkspaceWindow = new QMainWindow(this);
    _lasagnaWorkspaceWindow->setObjectName(QStringLiteral("lasagnaWorkspaceWindow"));

    _atlasWorkspaceWindow = new QMainWindow(this);
    _atlasWorkspaceWindow->setObjectName(QStringLiteral("atlasWorkspaceWindow"));
    _atlasWorkspaceWindow->setDockOptions(dockOptions());

    _fiberSliceWorkspaceWindow = new QMainWindow(this);
    _fiberSliceWorkspaceWindow->setObjectName(QStringLiteral("fiberSliceWorkspaceWindow"));
    _fiberSliceWorkspaceWindow->setDockOptions(dockOptions());

    _intersectionsWorkspaceWindow = new QMainWindow(this);
    _intersectionsWorkspaceWindow->setObjectName(QStringLiteral("intersectionsWorkspaceWindow"));
    _intersectionsWorkspaceWindow->setDockOptions(dockOptions());

    // The real Spiral workspace is installed after Main's CState exists. Keep
    // an inert shell here so it can be appended without shifting legacy tabs.
    _spiralWorkspaceWindow = new QMainWindow(this);
    _spiralWorkspaceWindow->setObjectName(QStringLiteral("spiralWorkspaceShell"));

    _workspaceTabs = new QTabWidget(this);
    _workspaceTabs->setObjectName(QStringLiteral("workspaceTabs"));
    _workspaceTabs->setTabsClosable(true);
    _workspaceTabs->addTab(_segmentWorkspaceWindow, tr("main"));
    _workspaceTabs->addTab(_lasagnaWorkspaceWindow, tr("Lasagna"));
    _workspaceTabs->addTab(_atlasWorkspaceWindow, tr("Atlas"));
    _workspaceTabs->addTab(_fiberSliceWorkspaceWindow, tr("Fiber Slice"));
    _workspaceTabs->addTab(_intersectionsWorkspaceWindow, tr("Intersections"));
    _workspaceTabs->addTab(_spiralWorkspaceWindow, tr("Spiral"));
    _segmentWorkspaceWindow->setProperty("workspaceId", QStringLiteral("main"));
    _lasagnaWorkspaceWindow->setProperty("workspaceId", QStringLiteral("lasagna"));
    _atlasWorkspaceWindow->setProperty("workspaceId", QStringLiteral("atlas"));
    _fiberSliceWorkspaceWindow->setProperty("workspaceId", QStringLiteral("fiber-slice"));
    _intersectionsWorkspaceWindow->setProperty("workspaceId", QStringLiteral("intersections"));
    _spiralWorkspaceWindow->setProperty("workspaceId", QStringLiteral("spiral"));
    if (auto* tabBar = _workspaceTabs->tabBar()) {
        for (int i = 0; i < _workspaceTabs->count(); ++i) {
            tabBar->setTabButton(i, QTabBar::RightSide, nullptr);
        }
    }
    auto* workspaceContainer = new QWidget(this);
    workspaceContainer->setObjectName(QStringLiteral("workspaceContainer"));
    auto* workspaceLayout = new QVBoxLayout(workspaceContainer);
    workspaceLayout->setContentsMargins(0, 0, 0, 0);
    workspaceLayout->setSpacing(0);
    _persistentCacheWarningBanner = new QFrame(workspaceContainer);
    _persistentCacheWarningBanner->setObjectName(QStringLiteral("persistentCacheWarningBanner"));
    _persistentCacheWarningBanner->setStyleSheet(
        QStringLiteral("QFrame#persistentCacheWarningBanner { background: #fff3cd; color: #664d03; border-bottom: 1px solid #ffda6a; }"));
    auto* warningLayout = new QHBoxLayout(_persistentCacheWarningBanner);
    warningLayout->setContentsMargins(10, 5, 8, 5);
    _persistentCacheWarningText = new QLabel(_persistentCacheWarningBanner);
    _persistentCacheWarningText->setObjectName(QStringLiteral("persistentCacheWarningText"));
    _persistentCacheWarningText->setWordWrap(true);
    warningLayout->addWidget(_persistentCacheWarningText, 1);
    auto* dismissCacheWarning = new QPushButton(tr("Dismiss"), _persistentCacheWarningBanner);
    dismissCacheWarning->setObjectName(QStringLiteral("dismissPersistentCacheWarning"));
    warningLayout->addWidget(dismissCacheWarning);
    connect(dismissCacheWarning, &QPushButton::clicked,
            _persistentCacheWarningBanner, &QWidget::hide);
    _persistentCacheWarningBanner->hide();
    workspaceLayout->addWidget(_persistentCacheWarningBanner);
    workspaceLayout->addWidget(_workspaceTabs, 1);
    setCentralWidget(workspaceContainer);
    connect(_workspaceTabs, &QTabWidget::currentChanged, this, [this]() {
        scheduleWindowStateSave();
        updateActiveWorkspaceViewerControls();
    });
    connect(_workspaceTabs, &QTabWidget::tabCloseRequested, this, [this](int index) {
        if (!_workspaceTabs) {
            return;
        }
        auto* dialog = qobject_cast<LineAnnotationDialog*>(_workspaceTabs->widget(index));
        if (dialog) {
            dialog->close();
        }
    });
    auto* lasagnaEscapeShortcut = new QShortcut(QKeySequence(Qt::Key_Escape), _lasagnaWorkspaceWindow);
    lasagnaEscapeShortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(lasagnaEscapeShortcut, &QShortcut::activated, this, &CWindow::switchToMainWorkspace);

    const QString baseTitle = windowTitle();
    const QString repoShortHash = QString::fromStdString(ProjectInfo::RepositoryShortHash()).trimmed();
    if (!repoShortHash.isEmpty() && !repoShortHash.startsWith('@')
        && repoShortHash.compare("Untracked", Qt::CaseInsensitive) != 0) {
        setWindowTitle(QString("%1 %2").arg(baseTitle, repoShortHash));
    }
    // setAttribute(Qt::WA_DeleteOnClose);

    _cacheSizeBytes = cacheSizeGB * 1024ULL * 1024ULL * 1024ULL;
    std::cout << "chunk cache budget is " << cacheSizeGB << " gigabytes" << std::endl;

    _state = new CState(_cacheSizeBytes, this);
    connect(_state, &CState::poiChanged, this, &CWindow::onFocusPOIChanged);
    connect(_state, &CState::surfaceWillBeDeleted, this, &CWindow::onSurfaceWillBeDeleted);
    connect(_state, &CState::vpkgChanged, this,
            [this](std::shared_ptr<VolumePkg> pkg) {
                _currentAtlasDir.reset();
                _currentAtlasName.clear();
                if (_lineAnnotationController) {
                    _lineAnnotationController->setCurrentAtlasDirectory(std::nullopt);
                }
                _fiberIntersectionCache.clear();
                refreshAtlasOverviewDocks();
                updateAtlasFiberDocks();
                updateAtlasSearchDocks();
                if (!pkg) return;
                pkg->setSegmentsChangedCallback(
                    [self = QPointer<CWindow>(this)]() {
                        QMetaObject::invokeMethod(self.data(), [self]() {
                            auto* w = self.data();
                            if (!w || !w->_surfacePanel || !w->_state || !w->_state->vpkg()) return;
                            w->_surfacePanel->setVolumePkg(w->_state->vpkg());
                            w->_surfacePanel->refreshSurfaceList();
                        }, Qt::QueuedConnection);
                    });
            });

    _fileWatcher = std::make_unique<FileWatcherService>(_state, this);
    connect(_fileWatcher.get(), &FileWatcherService::statusMessage,
            this, &CWindow::onShowStatusMessage);
    connect(_fileWatcher.get(), &FileWatcherService::volumeCatalogChanged,
            this, [this](const QString& preferredVolumeId) {
                refreshCurrentVolumePackageUi(preferredVolumeId, false);
            });

    _axisAlignedSliceController = std::make_unique<AxisAlignedSliceController>(_state, this);

    _viewerManager = std::make_unique<ViewerManager>(_state, _state->pointCollection(), this);
    _viewerManager->setSegmentationCursorMirroring(_mirrorCursorToSegmentation);
    if (_workspaceTabs && _spiralWorkspaceWindow) {
        const int spiralIndex = _workspaceTabs->indexOf(_spiralWorkspaceWindow);
        auto* shell = _spiralWorkspaceWindow;
        _workspaceTabs->removeTab(spiralIndex);
        _spiralWorkspace = new SpiralWorkspace(_state, this);
        _spiralWorkspace->setProperty("workspaceId", QStringLiteral("spiral"));
        _spiralWorkspaceWindow = _spiralWorkspace;
        _workspaceTabs->insertTab(spiralIndex, _spiralWorkspace, tr("Spiral"));
        if (auto* tabBar = _workspaceTabs->tabBar()) tabBar->setTabButton(spiralIndex, QTabBar::RightSide, nullptr);
        shell->deleteLater();
        // The "Add to current spiral fit" context actions are greyed out
        // unless a Spiral session is active on the connected service.
        connect(_spiralWorkspace, &SpiralWorkspace::spiralSessionActiveChanged, this, [this](bool active) {
            if (_surfacePanel) _surfacePanel->setSpiralFitAvailable(active);
            if (_fiberWidget) _fiberWidget->setSpiralFitAvailable(active);
        });
    }
    _lineAnnotationController = std::make_unique<LineAnnotationController>(_state,
                                                                           _viewerManager.get(),
                                                                           this,
                                                                           this);
    _lineAnnotationController->setVolumeSelectorFactory(
        [this](QWidget* parent) { return createAnnotationVolumeSelector(parent); });
    connect(_lineAnnotationController.get(),
            &LineAnnotationController::atlasCreated,
            this,
            &CWindow::displayAtlasFromDirectory);
    connect(_lineAnnotationController.get(),
            &LineAnnotationController::lineAnnotationWorkspaceRequested,
            this,
            &CWindow::openLineAnnotationWorkspace);
    connect(_lineAnnotationController.get(),
            &LineAnnotationController::fiberSaved,
            this,
            [this](uint64_t fiberId, uint64_t) {
                _fiberIntersectionCache.pruneFiber(fiberId);
                updateAtlasSearchDocks();
            });
    connect(_lineAnnotationController.get(),
            &LineAnnotationController::fibersDeleted,
            this,
            [this](std::vector<uint64_t> fiberIds) {
                for (uint64_t fiberId : fiberIds) {
                    _fiberIntersectionCache.pruneFiber(fiberId);
                }
                updateAtlasSearchDocks();
            });
    connect(_viewerManager.get(), &ViewerManager::baseViewerCreated, this, [this](VolumeViewerBase* viewer) {
        if (!viewer) {
            return;
        }
        if (auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject())) {
            configureChunkedViewerConnections(chunkedViewer);
        }
    });
    // Fiber annotation gets first refusal on volume clicks before the shared
    // Ctrl+click-to-focus policy runs.
    _viewerManager->setVolumeClickInterceptor(
        [this](const cv::Vec3f& volLoc, const cv::Vec3f& normal, Surface* surf,
               Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
            return _fiberController &&
                   _fiberController->handleVolumeClick(volLoc, normal, surf, button, modifiers);
        });
    connect(_viewerManager.get(), &ViewerManager::sharedCacheStatsChanged,
            this, &CWindow::onSharedCacheStatsChanged);

    _sharedCacheStatsLabel = new QLabel(this);
    _sharedCacheStatsLabel->setContentsMargins(8, 0, 8, 0);
    _sharedCacheStatsLabel->setMinimumWidth(320);
    _sharedCacheStatsLabel->setText(tr("RAM --  disk --  network --"));
    statusBar()->addPermanentWidget(_sharedCacheStatsLabel);

    _persistentCacheLowSpaceLabel = new QLabel(this);
    _persistentCacheLowSpaceLabel->setObjectName(QStringLiteral("persistentCacheLowSpaceLabel"));
    _persistentCacheLowSpaceLabel->setStyleSheet(QStringLiteral("color: #d28b00; font-weight: 600;"));
    _persistentCacheLowSpaceLabel->hide();
    statusBar()->addPermanentWidget(_persistentCacheLowSpaceLabel);

    _persistentCacheSpaceTimer = new QTimer(this);
    _persistentCacheSpaceTimer->setInterval(5000);
    auto updatePersistentCacheSpace = [this]() {
        auto root = vc3d::remoteCacheRootForState(_state);
        if (_state) {
            if (const auto volume = _state->currentVolume();
                volume && !volume->remoteCacheRoot().empty()) {
                root = volume->remoteCacheRoot();
            }
        }
        auto budget = vc::render::PersistentZarrCacheBudget::findForPath(root);
        if (!budget)
            return;
        const auto stats = budget->stats();
        if (stats.lowSpace) {
            const double freeGiB = static_cast<double>(stats.freeBytes) /
                                   (1024.0 * 1024.0 * 1024.0);
            const QString text = tr("Low disk space: %1 GiB free; remote Zarr cache growth is paused.")
                                     .arg(freeGiB, 0, 'f', 1);
            _persistentCacheLowSpaceLabel->setText(QStringLiteral("⚠ ") + text);
            _persistentCacheLowSpaceLabel->show();
            if (!_persistentCacheBannerShownThisSession) {
                _persistentCacheBannerShownThisSession = true;
                _persistentCacheWarningText->setText(text);
                _persistentCacheWarningBanner->show();
            }
        } else {
            _persistentCacheLowSpaceLabel->hide();
            _persistentCacheWarningBanner->hide();
        }
    };
    connect(_persistentCacheSpaceTimer, &QTimer::timeout, this,
            updatePersistentCacheSpace);
    _persistentCacheSpaceTimer->start();
    updatePersistentCacheSpace();

    // Z-scroll sensitivity label in status bar
    _sliceStepLabel = new QLabel(this);
    _sliceStepLabel->setContentsMargins(4, 0, 4, 0);
    double initialSensitivity = _viewerManager->zScrollSensitivity();
    _sliceStepLabel->setText(tr("Z sens: %1").arg(initialSensitivity, 0, 'f', 1));
    _sliceStepLabel->setToolTip(tr("Z-scroll sensitivity: use Shift+G / Shift+H to adjust"));
    statusBar()->addPermanentWidget(_sliceStepLabel);

    _pointsOverlay = std::make_unique<PointsOverlayController>(_state->pointCollection(), this);
    _viewerManager->setPointsOverlay(_pointsOverlay.get());

    _fiberOverlay = std::make_unique<FiberOverlayController>(this);
    _fiberOverlay->bindToViewerManager(_viewerManager.get());

    _rawPointsOverlay = std::make_unique<RawPointsOverlayController>(_state, this);
    _viewerManager->setRawPointsOverlay(_rawPointsOverlay.get());

    _pathsOverlay = std::make_unique<PathsOverlayController>(this);
    _viewerManager->setPathsOverlay(_pathsOverlay.get());

    _bboxOverlay = std::make_unique<BBoxOverlayController>(this);
    _viewerManager->setBBoxOverlay(_bboxOverlay.get());

    _vectorOverlay = std::make_unique<VectorOverlayController>(_state, this);
    _viewerManager->setVectorOverlay(_vectorOverlay.get());

    _inkDetectionOverlay = std::make_unique<InkDetectionOverlayController>(_state, this);
    _viewerManager->setInkDetectionOverlay(_inkDetectionOverlay.get());

    _atlasControlOverlay = std::make_unique<AtlasControlPointsOverlayController>(this);
    _atlasControlOverlay->bindToViewerManager(_viewerManager.get());

    _planeSlicingOverlay = std::make_unique<PlaneSlicingOverlayController>(_state, this);
    _planeSlicingOverlay->bindToViewerManager(_viewerManager.get());
    _planeSlicingOverlay->setRotationSetter([this](const std::string& planeName, float degrees) {
        _axisAlignedSliceController->setRotationDegrees(planeName, degrees);
        _axisAlignedSliceController->scheduleOrientationUpdate();
    });
    _planeSlicingOverlay->setRotationFinishedCallback([this]() {
        _axisAlignedSliceController->flushOrientationUpdate();
    });
    _planeSlicingOverlay->setAxisAlignedEnabled(_axisAlignedSliceController && _axisAlignedSliceController->isEnabled());

    _axisAlignedSliceController->setPlaneSlicingOverlay(_planeSlicingOverlay.get());
    _axisAlignedSliceController->setViewerManager(_viewerManager.get());

    _volumeOverlay = std::make_unique<VolumeOverlayController>(_viewerManager.get(), this);
    connect(_volumeOverlay.get(), &VolumeOverlayController::requestStatusMessage, this,
            [this](const QString& message, int timeout) {
                if (statusBar()) {
                    showStatusBarMessage(message, timeout);
                }
            });
    connect(_state, &CState::volumeChanged, _volumeOverlay.get(),
            [this](const std::shared_ptr<Volume>&, const std::string&) {
                if (_volumeOverlay && activeWorkspaceViewerManager() == _viewerManager.get()) {
                    _volumeOverlay->refreshForCurrentVolume();
                }
            });
    _viewerManager->setVolumeOverlay(_volumeOverlay.get());

    if (_statusDockPanelHost) {
        if (auto* statusDockBar = _statusDockPanelHost->takeBarWidget()) {
            statusDockBar->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);
            statusBar()->insertWidget(0, statusDockBar, 0);
        }
    }

    _statusMessageLabel = new QLabel(this);
    _statusMessageLabel->setObjectName(QStringLiteral("statusMessageLabel"));
    _statusMessageLabel->setContentsMargins(8, 0, 8, 0);
    _statusMessageLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _statusMessageLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    _statusMessageLabel->setMinimumWidth(160);
    _statusMessageLabel->setWordWrap(false);
    statusBar()->addWidget(_statusMessageLabel, 1);

    _statusMessageTimer = new QTimer(this);
    _statusMessageTimer->setSingleShot(true);
    connect(_statusMessageTimer, &QTimer::timeout, this, &CWindow::clearStatusBarMessage);
    connect(statusBar(), &QStatusBar::messageChanged, this, [this](const QString& message) {
        if (_relayingNativeStatusMessage || message.isEmpty()) {
            return;
        }

        _relayingNativeStatusMessage = true;
        showStatusBarMessage(message, 0);
        statusBar()->clearMessage();
        _relayingNativeStatusMessage = false;
    });

    // create UI widgets
    CreateWidgets();

    // create menus/actions controller
    _menuController = std::make_unique<MenuActionController>(this);
    _menuController->populateMenus(menuBar());
    // Wire the Actions -> Merge tifxyz... menu entry to the handler.
    // Has to happen here (not inside CreateWidgets) because
    // _menuController is created after CreateWidgets() returns.
    connect(_menuController.get(), &MenuActionController::mergeTifxyzFromMenuRequested,
            this, [this]() {
                _segmentationCommandHandler->onMergeTifxyz(QStringList{});
            });
    connect(_menuController.get(), &MenuActionController::mergePatchFromMenuRequested,
            this, [this]() {
                _segmentationCommandHandler->onMergePatch(QStringList{});
            });
    connect(_menuController.get(), &MenuActionController::openDataCatalogVisibilityChanged,
            this, [this](bool) {
                updateVolumePackageEmptyState();
            });
    connect(ui.btnOpenDataCatalog, &QPushButton::clicked, this, [this]() {
        if (_menuController) {
            _menuController->showOpenDataCatalog();
            updateVolumePackageEmptyState();
        }
    });

    if (isDarkMode()) {
        applyDarkPalette();
        const auto style = "QMenuBar { background: qlineargradient( x0:0 y0:0, x1:1 y1:0, stop:0 rgb(55, 80, 170), stop:0.8 rgb(225, 90, 80), stop:1 rgb(225, 150, 0)); }"
            "QMenuBar::item { background: transparent; }"
            "QMenuBar::item:selected { background: rgb(235, 180, 30); }"
            "QWidget#dockWidgetVolumesContent { background: rgb(55, 55, 55); }"
            "QWidget#dockWidgetSegmentationContent { background: rgb(55, 55, 55); }"
            "QWidget#dockWidgetAnnotationsContent { background: rgb(55, 55, 55); }"
            "QDockWidget::title { padding-top: 6px; background: rgb(60, 60, 75); }"
            "QTabBar::tab { background: rgb(60, 60, 75); }"
            "QWidget#tabSegment { background: rgb(55, 55, 55); }";
        setStyleSheet(style);
    } else {
        const auto style = "QMenuBar { background: qlineargradient( x0:0 y0:0, x1:1 y1:0, stop:0 rgb(85, 110, 200), stop:0.8 rgb(255, 120, 110), stop:1 rgb(255, 180, 30)); }"
            "QMenuBar::item { background: transparent; }"
            "QMenuBar::item:selected { background: rgb(255, 200, 50); }"
            "QWidget#dockWidgetVolumesContent { background: rgb(245, 245, 255); }"
            "QWidget#dockWidgetSegmentationContent { background: rgb(245, 245, 255); }"
            "QWidget#dockWidgetAnnotationsContent { background: rgb(245, 245, 255); }"
            "QDockWidget::title { padding-top: 6px; background: rgb(205, 210, 240); }"
            "QTabBar::tab { background: rgb(205, 210, 240); }"
            "QWidget#tabSegment { background: rgb(245, 245, 255); }"
            "QRadioButton:disabled { color: gray; }";
        setStyleSheet(style);
    }

    // Restore geometry / sizes
    QSettings geometry(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QString currentScreenSignature = windowStateScreenSignature();
    const QString currentQtVersion = windowStateQtVersion();
    const QString currentAppVersion = windowStateAppVersion();

    const bool restoreDisabled =
        geometry.value(vc3d::settings::window::RESTORE_DISABLED, false).toBool();
    const bool restoreInProgress =
        geometry.value(vc3d::settings::window::RESTORE_IN_PROGRESS, false).toBool();

    auto clearSavedWindowState = [&geometry]() {
        geometry.remove(vc3d::settings::window::GEOMETRY);
        geometry.remove(vc3d::settings::window::STATE);
    };

    bool allowRestore = !restoreDisabled && !restoreInProgress;
    if (restoreInProgress) {
        Logger()->warn("Previous window-state restore did not complete; clearing saved state");
        clearSavedWindowState();
        geometry.setValue(vc3d::settings::window::RESTORE_DISABLED, true);
        geometry.setValue(vc3d::settings::window::RESTORE_IN_PROGRESS, false);
        geometry.sync();
        allowRestore = false;
    }
    const bool hasStateMeta =
        geometry.contains(vc3d::settings::window::STATE_META_SCREEN_SIGNATURE)
        && geometry.contains(vc3d::settings::window::STATE_META_QT_VERSION)
        && geometry.contains(vc3d::settings::window::STATE_META_APP_VERSION);
    if (allowRestore && hasStateMeta
        && !windowStateMetaMatches(geometry,
                                   currentScreenSignature,
                                   currentQtVersion,
                                   currentAppVersion)) {
        Logger()->warn("Window state metadata mismatch; skipping restore");
        clearSavedWindowState();
        writeWindowStateMeta(geometry, currentScreenSignature, currentQtVersion, currentAppVersion);
        geometry.setValue(vc3d::settings::window::RESTORE_IN_PROGRESS, false);
        geometry.sync();
        allowRestore = false;
    }

    if (allowRestore) {
        geometry.setValue(vc3d::settings::window::RESTORE_IN_PROGRESS, true);
        geometry.sync();
    }

    bool restoredGeometry = false;
    bool restoredState = false;
    if (allowRestore) {
        const QByteArray savedGeometry = geometry.value(vc3d::settings::window::GEOMETRY).toByteArray();
        if (!savedGeometry.isEmpty()) {
            restoredGeometry = restoreGeometry(savedGeometry);
            if (!restoredGeometry) {
                Logger()->warn("Failed to restore main window geometry; clearing saved geometry");
                geometry.remove(vc3d::settings::window::GEOMETRY);
                geometry.sync();
            }
        }
        const QByteArray savedState = geometry.value(vc3d::settings::window::STATE).toByteArray();
        if (!savedState.isEmpty()) {
            restoredState = _segmentWorkspaceWindow && _segmentWorkspaceWindow->restoreState(savedState);
            if (!restoredState) {
                Logger()->warn("Failed to restore main window state; clearing saved state");
                geometry.remove(vc3d::settings::window::STATE);
                geometry.sync();
            }
        }
    }
    if (allowRestore) {
        QTimer::singleShot(1500, this, [currentScreenSignature, currentQtVersion, currentAppVersion]() {
            QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
            if (settings.value(vc3d::settings::window::RESTORE_DISABLED, false).toBool()) {
                settings.setValue(vc3d::settings::window::RESTORE_IN_PROGRESS, false);
                settings.sync();
                return;
            }
            writeWindowStateMeta(settings, currentScreenSignature, currentQtVersion, currentAppVersion);
            settings.setValue(vc3d::settings::window::RESTORE_IN_PROGRESS, false);
            settings.sync();
        });
    }
    // Ensure right-side tabified docks have a usable minimum size
    for (QDockWidget* dock : { ui.dockWidgetSegmentation,
                               _lasagnaDock,
                               ui.dockWidgetDistanceTransform,
                               _atlasOverviewDock,
                               _atlasSearchDock,
                               ui.dockWidgetOverlay,
                               _inkDetectionDock,
                               _transformsDock,
                               static_cast<QDockWidget*>(_atlasControlDock),
                               _atlasWorkspaceOverviewDock,
                               _atlasWorkspaceSearchDock,
                               static_cast<QDockWidget*>(_atlasWorkspaceFiberDock),
                               static_cast<QDockWidget*>(_fiberSliceWidget) }) {
        if (dock) {
            dock->setMinimumWidth(250);
            dock->setMinimumHeight(120);
        }
    }
    if (!restoredState) {
        // No saved state - set sensible default sizes for dock widgets
        _segmentWorkspaceWindow->resizeDocks({ui.dockWidgetVolumes}, {300}, Qt::Horizontal);
        _segmentWorkspaceWindow->resizeDocks({ui.dockWidgetVolumes}, {400}, Qt::Vertical);
        _segmentWorkspaceWindow->resizeDocks({ui.dockWidgetSegmentation}, {350}, Qt::Horizontal);
    }

    if (_workspaceTabs) {
        int workspaceIndex = -1;
        const QString workspaceId = geometry.value(WORKSPACE_ID_SETTING).toString();
        if (!workspaceId.isEmpty()) {
            for (int i = 0; i < _workspaceTabs->count(); ++i) {
                if (_workspaceTabs->widget(i)->property("workspaceId").toString() == workspaceId) {
                    workspaceIndex = i;
                    break;
                }
            }
        }
        if (workspaceIndex < 0) workspaceIndex = geometry.value(WORKSPACE_TAB_SETTING, 0).toInt();
        if (workspaceIndex >= 0 && workspaceIndex < _workspaceTabs->count()) {
            _workspaceTabs->setCurrentIndex(workspaceIndex);
        }
    }

    for (QDockWidget* dock : { ui.dockWidgetSegmentation,
                               _lasagnaDock,
                               ui.dockWidgetDistanceTransform,
                               static_cast<QDockWidget*>(_wrapAnnotationWidget),
                               ui.dockWidgetVolumes,
                               ui.dockWidgetViewerControls,
                               ui.dockWidgetOverlay,
                               ui.dockWidgetRenderSettings,
                               _inkDetectionDock,
                               _transformsDock,
                               _atlasOverviewDock,
                               _atlasSearchDock,
                               static_cast<QDockWidget*>(_atlasControlDock),
                               _atlasWorkspaceOverviewDock,
                               _atlasWorkspaceSearchDock,
                               static_cast<QDockWidget*>(_atlasWorkspaceFiberDock),
                               static_cast<QDockWidget*>(_fiberSliceWidget)  }) {
        ensureDockWidgetFeatures(dock);
        // Connect dock widget signals to trigger state saving
        if (!dock) continue;
        connect(dock, &QDockWidget::topLevelChanged, this, &CWindow::scheduleWindowStateSave);
        connect(dock, &QDockWidget::dockLocationChanged, this, &CWindow::scheduleWindowStateSave);
    }
    ensureDockWidgetFeatures(_point_collection_widget);
    connect(_point_collection_widget, &QDockWidget::topLevelChanged, this, &CWindow::scheduleWindowStateSave);
    connect(_point_collection_widget, &QDockWidget::dockLocationChanged, this, &CWindow::scheduleWindowStateSave);

    // Wayland workaround: dock drags trigger grabMouse() which fails,
    // leaving Qt's internal state stuck so all mouse events stop.
    // Synthesize a mouse release to clear the stuck button state.
    if (QGuiApplication::platformName() == QLatin1String("wayland")) {
        auto fixGrab = [this](){
            // Defer to after Qt finishes its internal dock drag processing.
            QTimer::singleShot(100, this, [](){
                if (auto* g = QWidget::mouseGrabber())
                    g->releaseMouse();
                for (auto* w : QGuiApplication::topLevelWindows())
                    w->setMouseGrabEnabled(false);
            });
        };
        for (QDockWidget* dock : { ui.dockWidgetSegmentation,
                                   _lasagnaDock,
                                   ui.dockWidgetDistanceTransform,
                                   static_cast<QDockWidget*>(_wrapAnnotationWidget),
                                   ui.dockWidgetVolumes,
                                   ui.dockWidgetViewerControls,
                                   ui.dockWidgetOverlay,
                                   ui.dockWidgetComposite,
                                   ui.dockWidgetRenderSettings,
                                   _inkDetectionDock,
                                   _transformsDock,
                                   _atlasOverviewDock,
                                   _atlasSearchDock,
                                   static_cast<QDockWidget*>(_atlasControlDock),
                                   _atlasWorkspaceOverviewDock,
                                   _atlasWorkspaceSearchDock,
                                   static_cast<QDockWidget*>(_atlasWorkspaceFiberDock),
                                   static_cast<QDockWidget*>(_fiberSliceWidget),
                                   static_cast<QDockWidget*>(_point_collection_widget) }) {
            if (!dock) continue;
            connect(dock, &QDockWidget::topLevelChanged, this, fixGrab);
            connect(dock, &QDockWidget::dockLocationChanged, this, fixGrab);
        }
    }

    if (_statusDockPanelHost) {
        for (QDockWidget* dock : {ui.dockWidgetVolumes,
                                  ui.dockWidgetViewerControls,
                                  ui.dockWidgetComposite,
                                  ui.dockWidgetRenderSettings,
                                  ui.dockWidgetOverlay,
                                  _inkDetectionDock,
                                  _transformsDock,
                                  ui.dockWidgetSegmentation,
                                  _lasagnaDock,
                                  ui.dockWidgetDistanceTransform,
                                  static_cast<QDockWidget*>(_point_collection_widget),
                                  static_cast<QDockWidget*>(_wrapAnnotationWidget),
                                  _atlasOverviewDock,
                                  _atlasSearchDock,
                                  static_cast<QDockWidget*>(_atlasControlDock),
                                  static_cast<QDockWidget*>(_fiberWidget)}) {
            _statusDockPanelHost->addDock(dock);
        }
    }

    const QSize minWindowSize(960, 640);
    setMinimumSize(minWindowSize);
    if (width() < minWindowSize.width() || height() < minWindowSize.height()) {
        resize(std::max(width(), minWindowSize.width()),
               std::max(height(), minWindowSize.height()));
    }
    if (!restoredGeometry) {
        setWindowState(windowState() | Qt::WindowMaximized);
    }

    bool scheduledStartupAction = false;

    // If enabled, auto open the last used local volume package.
    if (settings.value(vc3d::settings::project::AUTO_OPEN, vc3d::settings::project::AUTO_OPEN_DEFAULT).toInt() != 0) {

        QStringList files = settings.value(vc3d::settings::project::RECENT).toStringList();

        if (!files.empty() && !files.at(0).isEmpty()) {
            QString path = files[0];
            scheduledStartupAction = true;
            QTimer::singleShot(0, this, [this, path]() {
                if (_menuController) {
                    _menuController->openVolpkgAt(path);
                }
            });
        }
    }

    if (!scheduledStartupAction &&
        settings.value(vc3d::settings::project::SHOW_OPEN_DATA_CATALOG_ON_STARTUP,
                       vc3d::settings::project::SHOW_OPEN_DATA_CATALOG_ON_STARTUP_DEFAULT).toBool()) {
        QTimer::singleShot(0, this, [this]() {
            if (_menuController && !_state->hasVpkg()) {
                _menuController->showOpenDataCatalog();
                updateVolumePackageEmptyState();
            }
        });
    }

    // Create application-wide keyboard shortcuts
    fCompositeViewShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::CompositeView), this);
    fCompositeViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fCompositeViewShortcut, &QShortcut::activated, [this]() {
        if (_viewerCompositePanel) {
            _viewerCompositePanel->toggleSegmentationComposite();
        }
    });

    // Toggle direction hints overlay (Ctrl+T)
    fDirectionHintsShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::DirectionHints), this);
    fDirectionHintsShortcut->setContext(Qt::ApplicationShortcut);
    connect(fDirectionHintsShortcut, &QShortcut::activated, [this]() {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool current = settings.value(viewer::SHOW_DIRECTION_HINTS, viewer::SHOW_DIRECTION_HINTS_DEFAULT).toBool();
        bool next = !current;
        settings.setValue(viewer::SHOW_DIRECTION_HINTS, next ? "1" : "0");
        for (auto* manager : ViewerManager::allManagers()) {
            manager->setShowDirectionHints(next);
        }
    });

    // Toggle surface normals visualization (Ctrl+N)
    fSurfaceNormalsShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::SurfaceNormals), this);
    fSurfaceNormalsShortcut->setContext(Qt::ApplicationShortcut);
    connect(fSurfaceNormalsShortcut, &QShortcut::activated, [this]() {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool current = settings.value(viewer::SHOW_SURFACE_NORMALS, viewer::SHOW_SURFACE_NORMALS_DEFAULT).toBool();
        bool next = !current;
        settings.setValue(viewer::SHOW_SURFACE_NORMALS, next ? "1" : "0");
        for (auto* manager : ViewerManager::allManagers()) {
            manager->setShowSurfaceNormals(next);
        }
        showStatusBarMessage(next ? tr("Surface normals: ON") : tr("Surface normals: OFF"), 2000);
    });

    fAxisAlignedSlicesShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::AxisAlignedSlices), this);
    fAxisAlignedSlicesShortcut->setContext(Qt::ApplicationShortcut);
    connect(fAxisAlignedSlicesShortcut, &QShortcut::activated, [this]() {
        if (chkAxisAlignedSlices) {
            chkAxisAlignedSlices->toggle();
        }
    });

    // Raw points overlay shortcut (P key)
    auto* rawPointsShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::RawPointsOverlay), this);
    rawPointsShortcut->setContext(Qt::ApplicationShortcut);
    connect(rawPointsShortcut, &QShortcut::activated, [this]() {
        if (_rawPointsOverlay) {
            bool newEnabled = !_rawPointsOverlay->isEnabled();
            _rawPointsOverlay->setEnabled(newEnabled);
            showStatusBarMessage(
                newEnabled ? tr("Raw points overlay enabled") : tr("Raw points overlay disabled"),
                2000);
        }
    });

    // Zoom shortcuts (Shift+= for zoom in, Shift+- for zoom out)
    // Use 15% steps for smooth, proportional zooming - only affects active viewer
    constexpr float ZOOM_FACTOR = 1.15f;
    fZoomInShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::ZoomIn), this);
    fZoomInShortcut->setContext(Qt::ApplicationShortcut);
    connect(fZoomInShortcut, &QShortcut::activated, [this]() {
        if (auto* viewer = activeBaseViewer()) {
            viewer->adjustZoomByFactor(ZOOM_FACTOR);
        }
    });

    fZoomOutShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::ZoomOut), this);
    fZoomOutShortcut->setContext(Qt::ApplicationShortcut);
    connect(fZoomOutShortcut, &QShortcut::activated, [this]() {
        if (auto* viewer = activeBaseViewer()) {
            viewer->adjustZoomByFactor(1.0f / ZOOM_FACTOR);
        }
    });

    // Reset view shortcut (m to fit surface in view and reset all offsets)
    fResetViewShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::ResetView), this);
    fResetViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fResetViewShortcut, &QShortcut::activated, [this]() {
        if (auto* viewer = activeBaseViewer()) {
            viewer->resetSurfaceOffsets();
            viewer->fitSurfaceInView();
            viewer->renderVisible(true);
        }
    });

    // Z offset: Ctrl+. = +Z (further/deeper), Ctrl+, = -Z (closer)
    fWorldOffsetZPosShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::WorldOffsetZPos), this);
    fWorldOffsetZPosShortcut->setContext(Qt::ApplicationShortcut);
    connect(fWorldOffsetZPosShortcut, &QShortcut::activated, [this]() {
        if (auto* viewer = activeBaseViewer()) {
            viewer->adjustSurfaceOffset(1.0f);
        }
    });

    fWorldOffsetZNegShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::WorldOffsetZNeg), this);
    fWorldOffsetZNegShortcut->setContext(Qt::ApplicationShortcut);
    connect(fWorldOffsetZNegShortcut, &QShortcut::activated, [this]() {
        if (auto* viewer = activeBaseViewer()) {
            viewer->adjustSurfaceOffset(-1.0f);
        }
    });

    // Segment cycling shortcuts (] for next, [ for previous)
    fCycleNextSegmentShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::CycleNextSegment), this);
    fCycleNextSegmentShortcut->setContext(Qt::ApplicationShortcut);
    connect(fCycleNextSegmentShortcut, &QShortcut::activated, [this]() {
        if (!_surfacePanel) {
            return;
        }

        const bool preserveEditing = _segmentationWidget && _segmentationWidget->isEditingEnabled();
        bool previousIgnore = false;
        if (preserveEditing && _segmentationModule) {
            previousIgnore = _segmentationModule->ignoreSegSurfaceChange();
            _segmentationModule->setIgnoreSegSurfaceChange(true);
        }

        _surfacePanel->cycleToNextVisibleSegment();

        if (preserveEditing && _segmentationModule) {
            _segmentationModule->setIgnoreSegSurfaceChange(previousIgnore);
        }
    });

    fCyclePrevSegmentShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::CyclePrevSegment), this);
    fCyclePrevSegmentShortcut->setContext(Qt::ApplicationShortcut);
    connect(fCyclePrevSegmentShortcut, &QShortcut::activated, [this]() {
        if (!_surfacePanel) {
            return;
        }

        const bool preserveEditing = _segmentationWidget && _segmentationWidget->isEditingEnabled();
        bool previousIgnore = false;
        if (preserveEditing && _segmentationModule) {
            previousIgnore = _segmentationModule->ignoreSegSurfaceChange();
            _segmentationModule->setIgnoreSegSurfaceChange(true);
        }

        _surfacePanel->cycleToPreviousVisibleSegment();

        if (preserveEditing && _segmentationModule) {
            _segmentationModule->setIgnoreSegSurfaceChange(previousIgnore);
        }
    });

    fApplyApprovedTagShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::ApplyApprovedTag), this);
    fApplyApprovedTagShortcut->setContext(Qt::ApplicationShortcut);
    connect(fApplyApprovedTagShortcut, &QShortcut::activated, [this]() {
        if (_surfacePanel &&
            _surfacePanel->setTagChecked(SurfacePanelController::Tag::Approved, true)) {
            showStatusBarMessage(tr("Applied Approved tag"), 2000);
        }
    });

    fApplyDefectiveTagShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::ApplyDefectiveTag), this);
    fApplyDefectiveTagShortcut->setContext(Qt::ApplicationShortcut);
    connect(fApplyDefectiveTagShortcut, &QShortcut::activated, [this]() {
        if (_surfacePanel &&
            _surfacePanel->setTagChecked(SurfacePanelController::Tag::Defective, true)) {
            showStatusBarMessage(tr("Applied Defective tag"), 2000);
        }
    });

    // Focused view toggle (Shift+Ctrl+F) - hides dock widgets, keeps all viewers
    fFocusedViewShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::FocusedView), this);
    fFocusedViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fFocusedViewShortcut, &QShortcut::activated, this, &CWindow::toggleFocusedView);

    fOpenLasagnaWorkspaceShortcut = new QShortcut(QKeySequence(QStringLiteral("Ctrl+L")), this);
    fOpenLasagnaWorkspaceShortcut->setContext(Qt::ApplicationShortcut);
    connect(fOpenLasagnaWorkspaceShortcut, &QShortcut::activated,
            this, &CWindow::switchToLasagnaWorkspace);

    fRepeatLasagnaActionShortcut = new QShortcut(QKeySequence(QStringLiteral("Ctrl+Shift+L")), this);
    fRepeatLasagnaActionShortcut->setContext(Qt::ApplicationShortcut);
    connect(fRepeatLasagnaActionShortcut, &QShortcut::activated,
            this, &CWindow::repeatLastLasagnaAction);

    connect(_surfacePanel.get(), &SurfacePanelController::moveToPathsRequested,
            _segmentationCommandHandler.get(), &SegmentationCommandHandler::onMoveSegmentToPaths);
    connect(_surfacePanel.get(), &SurfacePanelController::renameSurfaceRequested,
            _segmentationCommandHandler.get(), &SegmentationCommandHandler::onRenameSurface);
    connect(_surfacePanel.get(), &SurfacePanelController::copySurfaceRequested,
            _segmentationCommandHandler.get(), &SegmentationCommandHandler::onCopySurfaceRequested);

    // Render-bench: kick off replay once the event loop is running, or arm the
    // recorder to attach when a volume+segment becomes active.
    if (!_benchOptions.replayPath.isEmpty()) {
        _benchReplay = std::make_unique<RenderBenchReplay>();
        _benchReplay->setReplayLimit(_benchOptions.replayLimit);
        if (_benchReplay->load(_benchOptions.replayPath)) {
            _benchReplay->setWarmPass(_benchOptions.replayWarm);
            _benchReplay->setOffscreen4k(_benchOptions.replayOffscreen4k);
            _benchReplay->setSkipChunkComplete(_benchOptions.replaySkipChunkComplete);
            _benchReplay->setSkipFastRender(_benchOptions.replaySkipFastRender);
            _benchReplay->setTimedProfile(_benchOptions.replayTimedProfile);
            _benchReplay->setTimedProfilePeriodMs(_benchOptions.replayTimedProfilePeriodMs);
            QTimer::singleShot(0, this, [this] { _benchReplay->run(*this); });
        } else {
            _benchReplay.reset();
        }
    } else if (!_benchOptions.recordPath.isEmpty()) {
        _benchRecorder = std::make_unique<RenderBenchRecorder>(_benchOptions.recordPath);
    }
}

// Destructor
CWindow::~CWindow()
{
    _destroyingWindow = true;
    if (qApp) {
        qApp->removeEventFilter(this);
    }

    // Backstop in case ~CWindow is reached without closeEvent firing (e.g.
    // if the app is torn down programmatically). Same rationale as the
    // closeEvent hook — skip SurfacePatchIndex removal during teardown.
    if (_viewerManager) {
        _viewerManager->beginShutdown();
    }
    if (_fileWatcher) {
        _fileWatcher->stopWatching();
    }
    setStatusBar(nullptr);

    if (_lineAnnotationController) {
        _lineAnnotationController->saveOpenFibers();
    }
    CloseVolume();
}

void CWindow::populateDockToggleMenu(QMenu* menu) const
{
    if (!menu) {
        return;
    }

    auto addDock = [](QMenu* targetMenu, QDockWidget* dock) {
        if (dock) {
            targetMenu->addAction(dock->toggleViewAction());
        }
    };

    addDock(menu, ui.dockWidgetVolumes);
    addDock(menu, ui.dockWidgetSegmentation);
    addDock(menu, ui.dockWidgetDistanceTransform);
    addDock(menu, ui.dockWidgetViewerControls);
    addDock(menu, ui.dockWidgetNormalVis);
    addDock(menu, ui.dockWidgetView);
    addDock(menu, ui.dockWidgetOverlay);
    addDock(menu, _inkDetectionDock);
    addDock(menu, _transformsDock);
    addDock(menu, ui.dockWidgetRenderSettings);
    addDock(menu, ui.dockWidgetComposite);
    addDock(menu, _lasagnaDock);

    if (_atlasOverviewDock || _atlasSearchDock || _atlasControlDock || _atlasWorkspaceFiberDock) {
        auto* atlasMenu = menu->addMenu(tr("Atlas"));
        addDock(atlasMenu, _atlasOverviewDock);
        addDock(atlasMenu, _atlasSearchDock);
        addDock(atlasMenu, _atlasControlDock);
        addDock(atlasMenu, _atlasWorkspaceFiberDock);
    }

    addDock(menu, _point_collection_widget);
    addDock(menu, _wrapAnnotationWidget);

    if (_fiberWidget || _fiberSliceWidget) {
        auto* fiberMenu = menu->addMenu(tr("Fibers"));
        addDock(fiberMenu, _fiberWidget);
        addDock(fiberMenu, _fiberSliceWidget);
    }
}

VolumeViewerBase *CWindow::newConnectedViewer(std::string surfaceName, QString title, QMdiArea *mdiArea)
{
    if (!_viewerManager) {
        return nullptr;
    }

    VolumeViewerBase* viewer = _viewerManager->createViewer(surfaceName, title, mdiArea);
    if (!viewer) {
        return nullptr;
    }

    if (auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject())) {
        configureChunkedViewerConnections(chunkedViewer);
    }
    return viewer;
}

VolumeViewerBase* CWindow::newConnectedViewerInWidget(std::string surfaceName, QString title, QWidget* parent)
{
    if (!_viewerManager || !parent) {
        return nullptr;
    }

    VolumeViewerBase* viewer = _viewerManager->createViewerInWidget(surfaceName, parent);
    if (!viewer) {
        return nullptr;
    }

    if (auto* viewerObject = viewer->asQObject())
        viewerObject->setProperty("vc_viewer_label", title);

    if (auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject())) {
        configureChunkedViewerConnections(chunkedViewer);
    }
    return viewer;
}

void CWindow::configureChunkedViewerConnections(CChunkedVolumeViewer* viewer)
{
    if (!viewer) {
        return;
    }

    // Volume-change propagation, cache-stats aggregation, the default
    // volume-click policy and active-viewer tracking are wired by
    // ViewerManager::initializeChunkedViewer for every workspace.
    const bool annotationViewer = viewer->property("vc_viewer_role").toString() == QStringLiteral("annotation");

    if (annotationViewer && !viewer->property("vc_annotation_focus_bound").toBool()) {
        const std::string surfaceName = viewer->surfName();
        const bool atlasFocusViewer = surfaceName == ATLAS_INTERNAL_SURFACE_NAME;
        const bool lineFocusViewer = surfaceName.rfind("line-", 0) == 0 ||
                                     surfaceName.rfind("line_annotation_slice_", 0) == 0;
        if (atlasFocusViewer || lineFocusViewer) {
            connect(viewer,
                    &CChunkedVolumeViewer::sendVolumeClicked,
                    this,
                    [this, atlasFocusViewer, lineFocusViewer](cv::Vec3f volLoc,
                                                              cv::Vec3f normal,
                                                              Surface*,
                                                              Qt::MouseButton button,
                                                              Qt::KeyboardModifiers modifiers) {
                        if (button != Qt::LeftButton || !modifiers.testFlag(Qt::ControlModifier)) {
                            return;
                        }
                        std::string sourceId;
                        if (lineFocusViewer) {
                            sourceId = "segmentation";
                        }
                        const bool focused = centerFocusAt(volLoc, normal, sourceId);
                        if (atlasFocusViewer || (lineFocusViewer && focused)) {
                            switchToMainWorkspace();
                            raise();
                            activateWindow();
                        }
                    });
            viewer->setProperty("vc_annotation_focus_bound", true);
        }
    }

    if (auto* graphicsView = viewer->graphicsView()) {
        if (!viewer->property("vc_annotation_context_bound").toBool()) {
            connect(graphicsView,
                    &CVolumeViewerView::sendAnnotationContextMenuRequested,
                    this,
                    [this, viewer](QPointF scenePoint, QPoint globalPos, Qt::KeyboardModifiers) {
                        if (_lineAnnotationController &&
                            _lineAnnotationController->showGeneratedControlPointContextMenu(
                                viewer,
                                scenePoint,
                                globalPos)) {
                            return;
                        }

                        const cv::Vec3f volumePoint = viewer->sceneToVolume(scenePoint);
                        const bool validVolumePoint =
                            std::isfinite(volumePoint[0]) &&
                            std::isfinite(volumePoint[1]) &&
                            std::isfinite(volumePoint[2]);
                        const int seedX = validVolumePoint
                            ? static_cast<int>(std::lround(volumePoint[0]))
                            : 0;
                        const int seedY = validVolumePoint
                            ? static_cast<int>(std::lround(volumePoint[1]))
                            : 0;
                        const int seedZ = validVolumePoint
                            ? static_cast<int>(std::lround(volumePoint[2]))
                            : 0;

                        QPointer<SegmentationLasagnaPanel> lasagnaPanel = _segmentationWidget
                            ? _segmentationWidget->lasagnaPanel()
                            : nullptr;

                        bool activeSegmentHasLasagnaModel = false;
                        if (_state && _state->vpkg()) {
                            auto activeSurface = std::dynamic_pointer_cast<QuadSurface>(
                                _state->surface("segmentation"));
                            if (activeSurface && !activeSurface->path.empty()) {
                                std::filesystem::path segPath = activeSurface->path;
                                if (segPath.is_relative()) {
                                    std::filesystem::path outputSegmentsPath =
                                        _state->vpkg()->outputSegmentsPath();
                                    if (outputSegmentsPath.empty()) {
                                        outputSegmentsPath = _state->vpkg()->findSegmentPathByName(
                                            _state->vpkg()->getSegmentationDirectory());
                                    }
                                    if (outputSegmentsPath.empty()) {
                                        outputSegmentsPath =
                                            std::filesystem::path(_state->vpkg()->getVolpkgDirectory()) /
                                            "paths";
                                    }
                                    segPath = outputSegmentsPath / segPath.filename();
                                }
                                std::error_code ec;
                                activeSegmentHasLasagnaModel =
                                    std::filesystem::exists(segPath / "model.pt", ec);
                            }
                        }

                        QMenu menu(this);

                        QAction* newLineAnnotationAction = menu.addAction(tr("New line annotation"));
                        newLineAnnotationAction->setEnabled(
                            _lineAnnotationController &&
                            _lineAnnotationController->canLaunchFromViewer(viewer) &&
                            viewer->sampleSceneVolume(scenePoint).has_value());

                        QAction* createGrowPatchAction = menu.addAction(tr("Create Segment (GrowPatch)"));
                        createGrowPatchAction->setEnabled(validVolumePoint &&
                                                          _segmentationCommandHandler &&
                                                          _state &&
                                                          _state->vpkg() &&
                                                          _state->currentVolume());
                        connect(createGrowPatchAction, &QAction::triggered, this,
                                [this, volumePoint]() {
                                    if (!_segmentationCommandHandler) {
                                        return;
                                    }
                                    _segmentationCommandHandler->onCreateSegmentGrowPatchFromSeed(
                                        QVector3D(volumePoint[0], volumePoint[1], volumePoint[2]));
                                });

                        menu.addSeparator();

                        auto* viewerWidget = qobject_cast<QWidget*>(viewer->asQObject());
                        auto* viewerGrid = mainViewerSplitGrid(ui.tabSegment);
                        const int mainViewerPane = viewerGrid ? viewerGrid->indexOf(viewerWidget) : -1;
                        if (viewerGrid && mainViewerPane >= 0) {
                            const bool fullSizeActive = viewerGrid->fullSizeActive();
                            QAction* fullSizeAction = menu.addAction(
                                viewerGrid->fullSizeActiveForPane(mainViewerPane)
                                    ? tr("Exit full size")
                                    : tr("Full size viewer"));
                            connect(fullSizeAction, &QAction::triggered, this,
                                    [viewerGrid, mainViewerPane]() {
                                        if (viewerGrid->fullSizeActiveForPane(mainViewerPane)) {
                                            viewerGrid->exitFullSize();
                                        } else {
                                            viewerGrid->setFullSizePane(mainViewerPane);
                                        }
                                    });

                            QAction* closeAction = menu.addAction(tr("Close viewer"));
                            closeAction->setEnabled(!fullSizeActive && !viewerGrid->paneHidden(mainViewerPane));
                            connect(closeAction, &QAction::triggered, this, [viewerGrid, mainViewerPane]() {
                                viewerGrid->setPaneHidden(mainViewerPane, true);
                                persistMainViewerLayout(viewerGrid);
                            });

                            auto* moveMenu = menu.addMenu(tr("Move viewer to"));
                            for (int pane = 0; pane < 4; ++pane) {
                                const QString paneLabel = QObject::tr(kMainViewerPaneLabels[pane]);
                                QAction* paneAction = moveMenu->addAction(paneLabel);
                                paneAction->setEnabled(!fullSizeActive && pane != mainViewerPane);
                                connect(paneAction, &QAction::triggered, this,
                                        [viewerGrid, mainViewerPane, pane]() {
                                            const bool targetWasHidden = viewerGrid->paneHidden(pane);
                                            viewerGrid->swapViewers(mainViewerPane, pane);
                                            if (targetWasHidden) {
                                                viewerGrid->setPaneHidden(pane, false);
                                                viewerGrid->setPaneHidden(mainViewerPane, true);
                                            }
                                            persistMainViewerLayout(viewerGrid);
                                        });
                            }

                            auto* showMenu = menu.addMenu(tr("Show closed viewer here"));
                            bool hasClosedViewer = false;
                            for (int pane = 0; pane < 4; ++pane) {
                                if (!viewerGrid->paneHidden(pane) || !viewerGrid->viewer(pane)) {
                                    continue;
                                }
                                auto* hiddenViewer = baseViewerFromWidget(viewerGrid->viewer(pane));
                                const QString label = hiddenViewer
                                    ? mainViewerDisplayName(hiddenViewer->surfName())
                                    : QObject::tr(kMainViewerPaneLabels[pane]);
                                QAction* showAction = showMenu->addAction(label);
                                hasClosedViewer = true;
                                connect(showAction, &QAction::triggered, this,
                                        [viewerGrid, mainViewerPane, pane]() {
                                            viewerGrid->swapViewers(mainViewerPane, pane);
                                            viewerGrid->setPaneHidden(mainViewerPane, false);
                                            viewerGrid->setPaneHidden(pane, true);
                                            persistMainViewerLayout(viewerGrid);
                                        });
                            }
                            showMenu->setEnabled(!fullSizeActive && hasClosedViewer);
                            menu.addSeparator();
                        }

                        QPointer<QMdiSubWindow> subWindow = viewerWidget
                            ? qobject_cast<QMdiSubWindow*>(viewerWidget->parentWidget())
                            : nullptr;
                        if (subWindow) {
                            QAction* maximizeAction = menu.addAction(
                                subWindow->isMaximized() ? tr("Restore viewer") : tr("Maximize viewer"));
                            connect(maximizeAction, &QAction::triggered, this, [subWindow]() {
                                if (!subWindow)
                                    return;
                                if (subWindow->isMaximized())
                                    subWindow->showNormal();
                                else
                                    subWindow->showMaximized();
                            });

                            QAction* closeAction = menu.addAction(tr("Close viewer"));
                            connect(closeAction, &QAction::triggered, this, [subWindow]() {
                                if (subWindow)
                                    subWindow->close();
                            });
                            menu.addSeparator();
                        }

                        if (viewer->measurementSupported()) {
                            QAction* measureAction = menu.addAction(tr("Measure"));
                            connect(measureAction, &QAction::triggered, this, [viewer]() {
                                viewer->startMeasurementMode();
                            });
                            menu.addSeparator();
                        }

                        auto addLasagnaLaunchAction =
                            [this, lasagnaPanel, &menu, seedX, seedY, seedZ, validVolumePoint,
                             activeSegmentHasLasagnaModel](
                                const QString& verb,
                                SegmentationLasagnaPanel::LasagnaMode mode,
                                bool requiresLasagnaModel,
                                bool showSeedInLabel) {
                                const QString configPath = lasagnaPanel
                                    ? lasagnaPanel->selectedLasagnaConfigPathForMode(mode)
                                    : QString();
                                const QString configName = QFileInfo(configPath).fileName();
                                const QString seedText = tr("%1,%2,%3").arg(seedX).arg(seedY).arg(seedZ);
                                QString label;
                                if (configName.isEmpty()) {
                                    label = showSeedInLabel
                                        ? tr("%1 l3d (%2)").arg(verb, seedText)
                                        : tr("%1 l3d").arg(verb);
                                } else {
                                    label = showSeedInLabel
                                        ? tr("%1 l3d (%2 %3)").arg(verb, configName, seedText)
                                        : tr("%1 l3d (%2)").arg(verb, configName);
                                }
                                QAction* launchAction = menu.addAction(label);
                                launchAction->setEnabled(
                                    validVolumePoint &&
                                    !configPath.isEmpty() &&
                                    (!requiresLasagnaModel || activeSegmentHasLasagnaModel));
                                launchAction->setToolTip(configPath);
                                connect(launchAction, &QAction::triggered, this,
                                        [this, lasagnaPanel, mode, configPath, seedX, seedY, seedZ]() {
                                            if (!lasagnaPanel) {
                                                return;
                                            }
                                            lasagnaPanel->startOptimizationAtSeed(
                                                _state,
                                                statusBar(),
                                                mode,
                                                configPath,
                                                seedX,
                                                seedY,
                                                seedZ);
                                        });
                            };

                        addLasagnaLaunchAction(
                            tr("new"), SegmentationLasagnaPanel::LasagnaMode::NewModel, false, true);
                        addLasagnaLaunchAction(
                            tr("reopt"), SegmentationLasagnaPanel::LasagnaMode::ReOptimize, true, false);

                        auto* configMenu = menu.addMenu(
                            tr("l3d config (%1,%2,%3)").arg(seedX).arg(seedY).arg(seedZ));
                        configMenu->setEnabled(validVolumePoint && lasagnaPanel);
                        if (lasagnaPanel) {
                            QStringList configs = lasagnaPanel->lasagnaConfigPathsForMode(
                                SegmentationLasagnaPanel::LasagnaMode::NewModel);
                            const QStringList reoptConfigs = lasagnaPanel->lasagnaConfigPathsForMode(
                                SegmentationLasagnaPanel::LasagnaMode::ReOptimize);
                            for (const QString& configPath : reoptConfigs) {
                                if (!configs.contains(configPath)) {
                                    configs.append(configPath);
                                }
                            }
                            if (configs.isEmpty()) {
                                QAction* none = configMenu->addAction(tr("No config selected"));
                                none->setEnabled(false);
                            }
                            for (const QString& configPath : configs) {
                                QAction* configAction =
                                    configMenu->addAction(QFileInfo(configPath).fileName());
                                configAction->setToolTip(configPath);
                                connect(configAction, &QAction::triggered, this,
                                        [this, lasagnaPanel, configPath, seedX, seedY, seedZ]() {
                                            if (!lasagnaPanel) {
                                                return;
                                            }
                                            lasagnaPanel->startOptimizationAtSeed(
                                                _state,
                                                statusBar(),
                                                SegmentationLasagnaPanel::LasagnaMode::NewModel,
                                                configPath,
                                                seedX,
                                                seedY,
                                                seedZ);
                                        });
                            }
                        }

                        QAction* selected = menu.exec(globalPos);
                        if (selected == newLineAnnotationAction &&
                            newLineAnnotationAction->isEnabled() &&
                            _lineAnnotationController) {
                            _lineAnnotationController->launchFromViewerAtPoint(viewer, scenePoint);
                        }
                    });
            viewer->setProperty("vc_annotation_context_bound", true);
        }

        if (!annotationViewer) {
            connect(graphicsView, &CVolumeViewerView::sendMousePress,
                    viewer, &CChunkedVolumeViewer::onMousePress, Qt::UniqueConnection);
            connect(graphicsView, &CVolumeViewerView::sendMouseMove,
                    viewer, &CChunkedVolumeViewer::onMouseMove, Qt::UniqueConnection);
            connect(graphicsView, &CVolumeViewerView::sendMouseRelease,
                    viewer, &CChunkedVolumeViewer::onMouseRelease, Qt::UniqueConnection);
        }
    }

    if (annotationViewer) {
        return;
    }

    if (_seedingWidget && !viewer->property("vc_seeding_bound").toBool()) {
        connect(_seedingWidget, &SeedingWidget::sendPathsChanged,
                viewer, &CChunkedVolumeViewer::onPathsChanged, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::sendMousePressVolume,
                _seedingWidget, &SeedingWidget::onMousePress, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::sendMouseMoveVolume,
                _seedingWidget, &SeedingWidget::onMouseMove, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::sendMouseReleaseVolume,
                _seedingWidget, &SeedingWidget::onMouseRelease, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::sendZSliceChanged,
                _seedingWidget, &SeedingWidget::updateCurrentZSlice, Qt::UniqueConnection);
        viewer->setProperty("vc_seeding_bound", true);
    }

    if (_point_collection_widget && !viewer->property("vc_points_bound").toBool()) {
        connect(_point_collection_widget, &CPointCollectionWidget::collectionSelected,
                viewer, &CChunkedVolumeViewer::onCollectionSelected, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::sendCollectionSelected,
                _point_collection_widget, &CPointCollectionWidget::selectCollection, Qt::UniqueConnection);
        connect(_point_collection_widget, &CPointCollectionWidget::pointSelected,
                viewer, &CChunkedVolumeViewer::onPointSelected, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::pointSelected,
                _point_collection_widget, &CPointCollectionWidget::selectPoint, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::pointClicked,
                _point_collection_widget, &CPointCollectionWidget::selectPoint, Qt::UniqueConnection);
        viewer->setProperty("vc_points_bound", true);
    }

    if (_wrapAnnotationWidget && !viewer->property("vc_wrap_annotation_bound").toBool()) {
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::collectionSelected,
                viewer, &CChunkedVolumeViewer::onCollectionSelected, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::sendCollectionSelected,
                _wrapAnnotationWidget, &WrapAnnotationWidget::selectCollection, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::pointSelected,
                viewer, &CChunkedVolumeViewer::onPointSelected, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::pointSelected,
                _wrapAnnotationWidget, &WrapAnnotationWidget::selectPoint, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::pointClicked,
                _wrapAnnotationWidget, &WrapAnnotationWidget::selectPoint, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationToggled,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationMode, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationSpacingChanged,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationSpacing, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationPolylineOpacityChanged,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationPolylineOpacity, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationMergeToggled,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationMergeExisting, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationPathTypeChanged,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationPathType, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationFilterTypeChanged,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationFilterType, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationFilterKernelSizeChanged,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationFilterKernelSize, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationClearRequested,
                viewer, &CChunkedVolumeViewer::clearSameWrapAnnotationPreview, Qt::UniqueConnection);
        viewer->setSameWrapAnnotationSpacing(_wrapAnnotationWidget->sameWrapAnnotationSpacing());
        viewer->setSameWrapAnnotationPolylineOpacity(_wrapAnnotationWidget->sameWrapAnnotationPolylineOpacity());
        viewer->setSameWrapAnnotationMergeExisting(_wrapAnnotationWidget->sameWrapAnnotationMergeEnabled());
        viewer->setSameWrapAnnotationPathType(_wrapAnnotationWidget->sameWrapAnnotationPathType());
        viewer->setSameWrapAnnotationFilterKernelSize(_wrapAnnotationWidget->sameWrapAnnotationFilterKernelSize());
        viewer->setSameWrapAnnotationFilterType(_wrapAnnotationWidget->sameWrapAnnotationFilterType());
        viewer->setSameWrapAnnotationMode(_wrapAnnotationWidget->sameWrapAnnotationEnabled());
        viewer->setProperty("vc_wrap_annotation_bound", true);
    }

    // Axis-aligned rotation/tilt wiring is attached per viewer by
    // AxisAlignedSliceController via ViewerManager::baseViewerCreated.
}

CChunkedVolumeViewer* CWindow::segmentationViewer() const
{
    if (!_viewerManager) {
        return nullptr;
    }
    for (auto* viewer : _viewerManager->baseViewers()) {
        if (viewer && viewer->surfName() == "segmentation") {
            return qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject());
        }
    }
    return nullptr;
}

VolumeViewerBase* CWindow::segmentationBaseViewer() const
{
    if (!_viewerManager) {
        return nullptr;
    }
    for (auto* viewer : _viewerManager->baseViewers()) {
        if (viewer && viewer->surfName() == "segmentation") {
            return viewer;
        }
    }
    return nullptr;
}

VolumeViewerBase* CWindow::activeBaseViewer() const
{
    auto* manager = activeWorkspaceViewerManager();
    if (!manager) {
        return nullptr;
    }

    if (auto* viewer = manager->activeViewer()) {
        return viewer;
    }

    if (manager != _viewerManager.get()) {
        QWidget* focus = QApplication::focusWidget();
        for (auto* viewer : manager->baseViewers()) {
            auto* widget = viewer ? qobject_cast<QWidget*>(viewer->asQObject()) : nullptr;
            if (widget && focus && (widget == focus || widget->isAncestorOf(focus))) {
                return viewer;
            }
        }
        for (auto* viewer : manager->baseViewers()) {
            auto* widget = viewer ? qobject_cast<QWidget*>(viewer->asQObject()) : nullptr;
            if (widget && widget->isVisible() && widget->underMouse()) {
                return viewer;
            }
        }
        for (auto* viewer : manager->baseViewers()) {
            auto* widget = viewer ? qobject_cast<QWidget*>(viewer->asQObject()) : nullptr;
            if (widget && widget->isVisible()) {
                return viewer;
            }
        }
        return nullptr;
    }

    if (!mdiArea) {
        return nullptr;
    }
    auto* subWindow = mdiArea->activeSubWindow();
    if (!subWindow) {
        return nullptr;
    }
    return baseViewerFromWidget(subWindow->widget());
}

ViewerManager* CWindow::activeWorkspaceViewerManager() const
{
    if (_workspaceTabs && _spiralWorkspace &&
        _workspaceTabs->currentWidget() == _spiralWorkspace) {
        return _spiralWorkspace->viewerManager();
    }
    return _viewerManager.get();
}

void CWindow::updateActiveWorkspaceViewerControls()
{
    auto* manager = activeWorkspaceViewerManager();
    if (_viewerControlsPanel) {
        _viewerControlsPanel->setViewerManager(manager);
    }
    if (_viewerCompositePanel) {
        _viewerCompositePanel->setViewerManager(manager);
    }
    if (_volumeOverlay) {
        _volumeOverlay->setViewerManager(manager);
    }
}

void CWindow::resetSegmentationViews(bool persistLayout)
{
    auto* viewerGrid = mainViewerSplitGrid(ui.tabSegment);
    if (!viewerGrid) {
        return;
    }

    auto ensureDefaultViewer = [this, viewerGrid](const std::string& surfaceName,
                                                  const QString& title,
                                                  const std::set<std::string>& intersects) -> VolumeViewerBase* {
        if (_viewerManager) {
            for (auto* viewer : _viewerManager->baseViewers()) {
                auto* viewerWidget = viewer ? qobject_cast<QWidget*>(viewer->asQObject()) : nullptr;
                if (viewer && viewerWidget && viewer->surfName() == surfaceName && viewerWidget->parentWidget() == viewerGrid) {
                    return viewer;
                }
            }
        }

        auto* viewer = newConnectedViewerInWidget(surfaceName, title, viewerGrid);
        if (viewer) {
            viewer->setIntersects(intersects);
        }
        return viewer;
    };

    auto* surfaceViewer = ensureDefaultViewer("segmentation", tr("Surface"), {"seg xz", "seg yz"});
    auto* xyViewer = ensureDefaultViewer("xy plane", tr("XY"), {"segmentation"});
    auto* xzViewer = ensureDefaultViewer("seg xz", tr("XZ"), {"segmentation"});
    auto* yzViewer = ensureDefaultViewer("seg yz", tr("YZ"), {"segmentation"});

    viewerGrid->setViewer(0, surfaceViewer ? qobject_cast<QWidget*>(surfaceViewer->asQObject()) : nullptr);
    viewerGrid->setViewer(1, xyViewer ? qobject_cast<QWidget*>(xyViewer->asQObject()) : nullptr);
    viewerGrid->setViewer(2, xzViewer ? qobject_cast<QWidget*>(xzViewer->asQObject()) : nullptr);
    viewerGrid->setViewer(3, yzViewer ? qobject_cast<QWidget*>(yzViewer->asQObject()) : nullptr);

    for (auto* viewer : {surfaceViewer, xzViewer, xyViewer, yzViewer}) {
        if (auto* widget = viewer ? qobject_cast<QWidget*>(viewer->asQObject()) : nullptr) {
            widget->show();
        }
    }

    viewerGrid->setPaneHidden(0, kDefaultMainViewerHidden[0]);
    viewerGrid->setPaneHidden(1, kDefaultMainViewerHidden[1]);
    viewerGrid->setPaneHidden(2, kDefaultMainViewerHidden[2]);
    viewerGrid->setPaneHidden(3, kDefaultMainViewerHidden[3]);
    viewerGrid->resetSplits();
    if (persistLayout) {
        persistMainViewerLayout(viewerGrid);
    }
}

void CWindow::clearSurfaceSelection()
{
    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->clearPreview(true);
    }
    _state->clearActiveSurface();

    if (_surfacePanel) {
        _surfacePanel->resetTagUi();
    }

    if (auto* viewer = segmentationViewer()) {
        viewer->setWindowTitle(tr("Surface"));
    }

    if (treeWidgetSurfaces) {
        treeWidgetSurfaces->clearSelection();
    }

    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }
    if (_atlasControlDock) {
        _atlasControlDock->clearResults();
    }
}

bool CWindow::restoreActiveSurfaceAfterSurfaceReload(const std::string& surfaceId)
{
    if (!_state || surfaceId.empty()) {
        return false;
    }

    auto surf = std::dynamic_pointer_cast<QuadSurface>(_state->surface(surfaceId));
    if (!surf && _state->vpkg()) {
        surf = _state->vpkg()->getSurface(surfaceId);
    }
    if (!surf) {
        return false;
    }

    if (_viewerManager) {
        _viewerManager->setSegmentationResetViewSuppressed(true);
    }

    _state->setActiveSurface(surfaceId, surf);
    _state->setSurface("segmentation", surf, false, true);

    if (_viewerManager) {
        _viewerManager->setSegmentationResetViewSuppressed(false);
    }

    if (_surfacePanel) {
        _surfacePanel->selectSurfaceById(surfaceId);
    }

    if (_segmentationModule) {
        try {
            _segmentationModule->onActiveSegmentChanged(surf.get());
        } catch (const std::exception& e) {
            qWarning() << "Failed to reactivate segment"
                       << QString::fromStdString(surfaceId)
                       << "after reloading surfaces:"
                       << e.what();
            return false;
        }
    }

    if (_point_collection_widget) {
        if (!surf->path.empty()) {
            _point_collection_widget->loadCorrPointsResults(surf->path / "corr_points_results.json");
        } else {
            _point_collection_widget->clearCorrPointsResults();
        }
    }
    if (_atlasControlDock) {
        if (!surf->path.empty()) {
            _atlasControlDock->loadResults(surf->path / "atlas_control_points_results.json");
        } else {
            _atlasControlDock->clearResults();
        }
    }

    if (_viewerManager) {
        _viewerManager->forEachBaseViewer([](VolumeViewerBase* viewer) {
            if (!viewer) {
                return;
            }
            viewer->invalidateIntersect("segmentation");
            viewer->renderIntersections("active segment restored after surface reload");
            viewer->requestRender("active segment restored after surface reload");
        });
    }

    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }
    maybeAttachBenchRecorder();
    return true;
}

void CWindow::setVolume(std::shared_ptr<Volume> newvol)
{
    const bool hadVolume = static_cast<bool>(_state->currentVolume());
    const std::string previousVolumeId = _state ? _state->currentVolumeId() : std::string{};
    std::string targetVolumeId;
    if (_state && _state->vpkg() && newvol) {
        for (const auto& id : _state->vpkg()->volumeIDs()) {
            if (_state->vpkg()->volume(id) == newvol) {
                targetVolumeId = id;
                break;
            }
        }
    }
    if (targetVolumeId.empty() && newvol) {
        targetVolumeId = newvol->id();
    }

    const auto navigationTransform =
        (hadVolume && newvol && previousVolumeId != targetVolumeId)
            ? openDataVolumeTransformForSwitch(previousVolumeId, targetVolumeId)
            : std::optional<cv::Matx44d>{};

    // Volume assignment, focus rederivation and viewer-navigation retargeting
    // are shared workspace policy.
    if (_viewerManager) {
        _viewerManager->switchVolume(newvol, navigationTransform);
    } else {
        _state->setCurrentVolume(newvol);
    }

    if (_viewerManager && _viewerManager->overlayVolume()) {
        _viewerManager->setOverlayVolume(
            _viewerManager->overlayVolume(),
            _viewerManager->overlayVolumeId());
    }
    if (_state->currentVolume() && !_state->currentVolumeId().empty()) {
        rememberCurrentVolumeForPackage(QString::fromStdString(_state->currentVolumeId()));
    }

    const bool growthVolumeValid = _state->hasVpkg() && !_state->segmentationGrowthVolumeId().empty() &&
                                   _state->vpkg()->hasVolume(_state->segmentationGrowthVolumeId());
    if (!growthVolumeValid) {
        _state->setSegmentationGrowthVolumeId(_state->currentVolumeId());
        if (_segmentationWidget) {
            _segmentationWidget->setActiveVolume(QString::fromStdString(_state->currentVolumeId()));
        }
    }

    updateNormalGridAvailability();

    _axisAlignedSliceController->applyOrientation(_state ? _state->surface("segmentation").get() : nullptr);
    syncVolumeSelectionControls();
    updateOpenDataSegmentTransformState(true);
}

CWindow::VolumeAttachResult CWindow::attachVolumeToCurrentPackage(
    const std::shared_ptr<Volume>& volume,
    const QString& location,
    std::vector<std::string> tags,
    const QString& remoteCacheRoot,
    const QString& preferredVolumeId)
{
    if (!_state || !_state->vpkg() || !volume) {
        throw std::logic_error("cannot attach a volume without an open project");
    }

    const auto result = _state->vpkg()->attachPreparedVolume(
        location.toStdString(),
        std::move(tags),
        volume,
        remoteCacheRoot.toStdString());
    if (result == VolumePkg::AttachVolumeResult::VolumeIdConflict)
        return VolumeAttachResult::VolumeIdConflict;

    const bool needSurfaceLoad = _surfacePanel && !_surfacePanel->hasSurfaces();
    refreshCurrentVolumePackageUi(preferredVolumeId, needSurfaceLoad);
    UpdateView();
    return result == VolumePkg::AttachVolumeResult::Attached
        ? VolumeAttachResult::Attached
        : VolumeAttachResult::AlreadyAttached;
}

void CWindow::refreshCurrentVolumePackageUi(const QString& preferredVolumeId,
                                            bool reloadSurfaces)
{
    if (!_state || !_state->vpkg()) {
        return;
    }

    if (_segmentationWidget) {
        _segmentationWidget->setVolumePackagePath(_state->vpkgPath());
    }

    updateNormalGridAvailability();

    refreshVolumeSelectionUi(preferredVolumeId);
    if (!_state->vpkg()->hasVolumes()) {
        Logger()->info("Opened volpkg '{}' with no volumes", _state->vpkgPath().toStdString());
        showStatusBarMessage(tr("Opened volume package with no volumes."), 5000);
    }

    if (_volumeOverlay) {
        _volumeOverlay->setVolumePkg(_state->vpkg(), _state->vpkgPath());
    }

    refreshSegmentationDirectoryDropdown();

    if (_surfacePanel) {
        _surfacePanel->setVolumePkg(_state->vpkg());
        if (_viewerManager) {
            _viewerManager->resetStrideUserOverride();
        }
        if (reloadSurfaces) {
            _surfacePanel->loadSurfaces(false);
            _surfacePanel->refreshPointSetFilterOptions();
        }
    }

    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }
    updateOpenDataSegmentTransformState(false);
}

void CWindow::updateNormalGridAvailability()
{
    QString checkedPath;
    const QString path = normalGridDirectoryForVolumePkg(
        _state->vpkg(), _state->currentVolumeId(), &checkedPath);
    const bool available = !path.isEmpty();

    _normalGridAvailable = available;
    _normalGridPath = path;

    if (_segmentationWidget) {
        _segmentationWidget->setNormalGridAvailable(_normalGridAvailable);
        _segmentationWidget->setNormalGridPath(_normalGridPath);
        QString hint;
        if (!_normalGridAvailable && _state->vpkg()) {
            const auto info = vc3d::opendata::normalGridsInfoFromTags(
                _state->vpkg()->volumeTags(_state->currentVolumeId()));
            if (info) {
                hint = tr("No normal grids are published for this open-data volume.");
            }
        }
        if (_normalGridAvailable) {
        } else if (!hint.isEmpty()) {
        } else if (!checkedPath.isEmpty()) {
            hint = tr("Checked: %1").arg(checkedPath);
        } else {
            hint = tr("No volume package loaded.");
        }
        _segmentationWidget->setNormalGridPathHint(hint);

        QString normal3dHint;
        const QStringList normal3d = normal3dZarrCandidatesForVolumePkg(_state->vpkg(), &normal3dHint);
        _segmentationWidget->setNormal3dZarrCandidates(normal3d, normal3dHint);
    }
}

void CWindow::toggleVolumeOverlayVisibility()
{
    if (_volumeOverlay) {
        _volumeOverlay->toggleVisibility();
    }
    if (_inkDetectionOverlay && _inkDetectionOverlay->hasLoadedSelection()) {
        _inkDetectionOverlay->toggleVisibility();
    }
}

void CWindow::toggleFocusedView()
{
    if (_focusedViewActive) {
        for (const auto& [dock, state] : _savedDockStates) {
            if (dock) {
                dock->setVisible(state.visible);
            }
        }
        for (const auto& [dock, state] : _savedDockStates) {
            if (dock && state.wasRaised) {
                dock->raise();
            }
        }
        _savedDockStates.clear();
        _focusedViewActive = false;
        showStatusBarMessage(tr("Restored full view"), 2000);
    } else {
        _savedDockStates.clear();
        const QList<QDockWidget*> docks = findChildren<QDockWidget*>();
        for (QDockWidget* dock : docks) {
            bool wasRaised = false;
            if (dock->isVisible() && !dock->isFloating()) {
                if (QWidget* content = dock->widget()) {
                    wasRaised = !content->visibleRegion().isEmpty();
                }
            }
            _savedDockStates[dock] = {dock->isVisible(), dock->isFloating(), wasRaised};
            dock->hide();
        }
        _focusedViewActive = true;
        showStatusBarMessage(tr("Focused view (Shift+Ctrl+F to restore)"), 2000);
    }
}

void CWindow::switchToLasagnaWorkspace()
{
    if (!_workspaceTabs || !_lasagnaWorkspaceWindow) {
        return;
    }
    const int index = _workspaceTabs->indexOf(_lasagnaWorkspaceWindow);
    if (index >= 0) {
        _workspaceTabs->setCurrentIndex(index);
        _lasagnaWorkspaceWindow->raise();
        _lasagnaWorkspaceWindow->setFocus(Qt::ShortcutFocusReason);
    }
}

void CWindow::switchToMainWorkspace()
{
    if (!_workspaceTabs || !_segmentWorkspaceWindow) {
        return;
    }
    const int index = _workspaceTabs->indexOf(_segmentWorkspaceWindow);
    if (index >= 0) {
        _workspaceTabs->setCurrentIndex(index);
        _segmentWorkspaceWindow->raise();
        _segmentWorkspaceWindow->setFocus(Qt::ShortcutFocusReason);
    }
}

void CWindow::switchToFiberSliceWorkspace()
{
    if (!_workspaceTabs || !_fiberSliceWorkspaceWindow) {
        return;
    }
    const int index = _workspaceTabs->indexOf(_fiberSliceWorkspaceWindow);
    if (index >= 0) {
        _workspaceTabs->setCurrentIndex(index);
        _fiberSliceWorkspaceWindow->raise();
        _fiberSliceWorkspaceWindow->setFocus(Qt::ShortcutFocusReason);
    }
}

void CWindow::openLineAnnotationWorkspace(LineAnnotationDialog* dialog, const QString& title)
{
    if (!_workspaceTabs || !dialog) {
        return;
    }

    dialog->setWorkspaceEmbedded(true);

    int index = _workspaceTabs->indexOf(dialog);
    if (index < 0) {
        index = _workspaceTabs->addTab(dialog, title.isEmpty() ? tr("Line Annotation") : title);
        auto* escapeShortcut = new QShortcut(QKeySequence(Qt::Key_Escape), dialog);
        escapeShortcut->setContext(Qt::WidgetWithChildrenShortcut);
        connect(escapeShortcut, &QShortcut::activated, dialog, &QWidget::close);
        QWidget* tabWidget = dialog;
        connect(dialog, &QObject::destroyed, this, [this, tabWidget]() {
            if (_destroyingWindow) {
                return;
            }
            if (!_workspaceTabs) {
                return;
            }
            for (int i = 0; i < _workspaceTabs->count(); ++i) {
                if (_workspaceTabs->widget(i) == tabWidget) {
                    const bool wasCurrent = _workspaceTabs->currentIndex() == i;
                    _workspaceTabs->removeTab(i);
                    if (wasCurrent) {
                        switchToMainWorkspace();
                    }
                    break;
                }
            }
        });
    }

    _workspaceTabs->setCurrentIndex(index);
    dialog->show();
    dialog->raise();
    dialog->setFocus(Qt::ShortcutFocusReason);
}

void CWindow::repeatLastLasagnaAction()
{
    if (_segmentationWidget && _segmentationWidget->lasagnaPanel()) {
        _segmentationWidget->lasagnaPanel()->repeatLastLasagnaAction();
    }
}

void CWindow::selectLasagnaOutputSegment(const QString& outputName)
{
    switchToMainWorkspace();
    if (!treeWidgetSurfaces) {
        return;
    }

    const QString trimmed = outputName.trimmed();
    if (trimmed.isEmpty()) {
        return;
    }

    QStringList candidates;
    auto addCandidate = [&candidates](const QString& value) {
        const QString candidate = value.trimmed();
        if (!candidate.isEmpty() && !candidates.contains(candidate)) {
            candidates << candidate;
        }
    };

    addCandidate(trimmed);
    addCandidate(QFileInfo(trimmed).fileName());
    const QStringList baseCandidates = candidates;
    for (const QString& candidate : baseCandidates) {
        if (candidate.endsWith(QStringLiteral(".tifxyz"))) {
            addCandidate(candidate.left(candidate.size() - 7));
        } else {
            addCandidate(candidate + QStringLiteral(".tifxyz"));
        }
    }

    auto selectCandidate = [this, &candidates]() -> bool {
        QTreeWidgetItemIterator it(treeWidgetSurfaces);
        while (*it) {
            const QString id = (*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
            if (candidates.contains(id)) {
                treeWidgetSurfaces->setCurrentItem(*it);
                treeWidgetSurfaces->scrollToItem(*it);
                ui.dockWidgetVolumes->raise();
                return true;
            }
            ++it;
        }
        return false;
    };

    if (selectCandidate()) {
        return;
    }

    if (_surfacePanel) {
        _surfacePanel->reloadSurfacesFromDisk();
        selectCandidate();
    }
}

bool CWindow::centerFocusAt(const cv::Vec3f& position, const cv::Vec3f& normal, const std::string& sourceId)
{
    return _viewerManager && _viewerManager->centerFocusAt(position, normal, sourceId);
}

void CWindow::recenterPlaneViewersOn(const cv::Vec3f& position)
{
    if (_viewerManager) {
        _viewerManager->recenterPlaneViewersOn(position);
    }
}

void CWindow::recenterSegmentationViewerNear(const cv::Vec3f& position)
{
    if (_viewerManager) {
        _viewerManager->recenterSegmentationViewerNear(position);
    }
}

bool CWindow::recenterViewersOnCurrentFocus()
{
    auto* manager = activeWorkspaceViewerManager();
    return manager && manager->recenterViewersOnCurrentFocus();
}

bool CWindow::centerFocusOnCursor()
{
    auto* manager = activeWorkspaceViewerManager();
    return manager && manager->centerFocusOnCursor();
}

void CWindow::setSegmentationCursorMirroring(bool enabled)
{
    if (_mirrorCursorToSegmentation == enabled) {
        return;
    }

    _mirrorCursorToSegmentation = enabled;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::MIRROR_CURSOR_TO_SEGMENTATION, enabled ? "1" : "0");

    if (_viewerManager) {
        _viewerManager->setSegmentationCursorMirroring(enabled);
    }

    if (statusBar()) {
        showStatusBarMessage(enabled ? tr("Syncing cursor across views enabled")
                                         : tr("Syncing cursor across views disabled"),
                                  2000);
    }
}

void CWindow::createAtlasWorkspace()
{
    if (!_atlasWorkspaceWindow || !_segmentWorkspaceWindow) {
        return;
    }

    if (!_atlasWorkspaceWindow->centralWidget()) {
        auto* central = new QWidget(_atlasWorkspaceWindow);
        auto* layout = new QVBoxLayout(central);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(0);
        _atlasWorkspaceWindow->setCentralWidget(central);
    }

    if (!_atlasViewer && _viewerManager) {
        auto* central = _atlasWorkspaceWindow->centralWidget();
        _atlasViewer = _viewerManager->createViewerInWidget(ATLAS_INTERNAL_SURFACE_NAME,
                                                            central,
                                                            ViewerManager::ViewerRole::Annotation);
        if (auto* layout = central ? dynamic_cast<QVBoxLayout*>(central->layout()) : nullptr) {
            if (auto* viewerWidget = _atlasViewer ? qobject_cast<QWidget*>(_atlasViewer->asQObject()) : nullptr) {
                layout->addWidget(viewerWidget);
            }
        }
        if (!_atlasOverlay) {
            _atlasOverlay = std::make_unique<AtlasOverlayController>(this);
        }
        _atlasOverlay->attachViewer(_atlasViewer);
    }

    _atlasWorkspaceOverviewDock = createAtlasOverviewDock(this);
    _atlasWorkspaceOverviewDock->setObjectName(QStringLiteral("dockWidgetAtlasOverviewAtlas"));
    _atlasWorkspaceWindow->addDockWidget(Qt::LeftDockWidgetArea, _atlasWorkspaceOverviewDock);

    _atlasWorkspaceFiberDock = createAtlasFiberDock(this);
    _atlasWorkspaceFiberDock->setObjectName(QStringLiteral("dockWidgetAtlasFibersAtlas"));
    _atlasWorkspaceWindow->addDockWidget(Qt::LeftDockWidgetArea, _atlasWorkspaceFiberDock);
    if (auto* optimize = _atlasWorkspaceFiberDock->widget()
            ? _atlasWorkspaceFiberDock->widget()->findChild<QPushButton*>(
                  QStringLiteral("atlasOptimizeSnapCandidatesButton"))
            : nullptr) {
        connect(optimize, &QPushButton::clicked,
                this, &CWindow::optimizeAtlasSnapCandidates);
    }
    _atlasWorkspaceWindow->tabifyDockWidget(_atlasWorkspaceOverviewDock, _atlasWorkspaceFiberDock);
    _atlasWorkspaceFiberDock->raise();

    _atlasWorkspaceSearchDock = createAtlasSearchDock(this);
    _atlasWorkspaceSearchDock->setObjectName(QStringLiteral("dockWidgetAtlasSearchAtlas"));
    _atlasWorkspaceWindow->addDockWidget(Qt::RightDockWidgetArea, _atlasWorkspaceSearchDock);

    _atlasOverviewDock = createAtlasOverviewDock(this);
    _atlasOverviewDock->setObjectName(QStringLiteral("dockWidgetAtlasOverview"));
    _segmentWorkspaceWindow->addDockWidget(Qt::LeftDockWidgetArea, _atlasOverviewDock);

    _atlasSearchDock = createAtlasSearchDock(this);
    _atlasSearchDock->setObjectName(QStringLiteral("dockWidgetAtlasSearch"));
    _segmentWorkspaceWindow->addDockWidget(Qt::RightDockWidgetArea, _atlasSearchDock);

    _atlasOverviewDock->hide();
    _atlasSearchDock->hide();

    auto connectOverview = [this](QDockWidget* dock) {
        if (!dock || !dock->widget()) {
            return;
        }
        auto* tree = dock->widget()->findChild<QTreeWidget*>(QStringLiteral("atlasOverviewTree"));
        if (!tree) {
            return;
        }
        if (auto* rankSnap = dock->widget()->findChild<QPushButton*>(
                QStringLiteral("atlasRankSnapButton"))) {
            connect(rankSnap, &QPushButton::clicked, this, &CWindow::optimizeAtlasSnapCandidates);
        }
        if (auto* remap = dock->widget()->findChild<QPushButton*>(
                QStringLiteral("atlasRemapButton"))) {
            connect(remap, &QPushButton::clicked, this, &CWindow::remapCurrentAtlas);
        }
        connect(tree, &QTreeWidget::itemActivated, this, [this](QTreeWidgetItem* item, int) {
            while (item && item->data(0, Qt::UserRole).toString().isEmpty()) {
                item = item->parent();
            }
            if (!item) {
                return;
            }
            const QString path = item->data(0, Qt::UserRole).toString();
            if (!path.isEmpty()) {
                displayAtlasFromDirectory(std::filesystem::path(path.toStdString()));
            }
        });
    };
    connectOverview(_atlasOverviewDock);
    connectOverview(_atlasWorkspaceOverviewDock);

    if (_atlasWorkspaceFiberDock && _atlasWorkspaceFiberDock->widget()) {
        if (auto* tree = _atlasWorkspaceFiberDock->widget()->findChild<QTreeWidget*>(QStringLiteral("atlasFiberTree"))) {
            auto itemFiberId = [](QTreeWidgetItem* item) -> uint64_t {
                while (item) {
                    bool ok = false;
                    const qulonglong fiberId = item->data(0, ATLAS_FIBER_ID_ROLE).toULongLong(&ok);
                    if (ok && fiberId != 0) {
                        return static_cast<uint64_t>(fiberId);
                    }
                    item = item->parent();
                }
                return 0;
            };
            auto focusAtlasControlItem = [this](QTreeWidgetItem* item) {
                if (!item || !_atlasOverlay) {
                    return;
                }
                const QString pathKey = item->data(0, ATLAS_FIBER_PATH_KEY_ROLE).toString();
                const int sourceIndex = item->data(0, ATLAS_CONTROL_SOURCE_INDEX_ROLE).toInt();
                if (!pathKey.isEmpty() && sourceIndex >= 0) {
                    _atlasOverlay->setSelectedControlPoint(
                        std::make_pair(pathKey.toStdString(), sourceIndex));
                }
                bool okX = false;
                bool okY = false;
                const float x = item->data(0, ATLAS_SURFACE_X_ROLE).toFloat(&okX);
                const float y = item->data(0, ATLAS_SURFACE_Y_ROLE).toFloat(&okY);
                if (okX && okY && std::isfinite(x) && std::isfinite(y)) {
                    centerViewerOnSurfacePointForNavigation(_atlasViewer, cv::Vec2f{x, y});
                }
            };
            auto openFromItem = [this, tree, itemFiberId, focusAtlasControlItem](QTreeWidgetItem* item) {
                if (!item ||
                    tree->property("vc_atlas_opening_fiber").toBool()) {
                    return;
                }
                bool hasControlIndex = false;
                const int controlIndex = item->data(0, ATLAS_CONTROL_INDEX_ROLE).toInt(&hasControlIndex);
                if (hasControlIndex && controlIndex >= 0) {
                    focusAtlasControlItem(item);
                    return;
                }
                if (!_lineAnnotationController) {
                    return;
                }
                const uint64_t fiberId = itemFiberId(item);
                if (fiberId == 0) {
                    return;
                }
                tree->setProperty("vc_atlas_opening_fiber", true);
                QTimer::singleShot(0, tree, [tree]() {
                    tree->setProperty("vc_atlas_opening_fiber", false);
                });
                if (_fiberWidget) {
                    _fiberWidget->selectFiber(fiberId);
                }
                if (_fiberSliceWidget) {
                    _fiberSliceWidget->selectFiber(fiberId);
                }
                _lineAnnotationController->openFiber(fiberId);
            };
            connect(tree,
                    &QTreeWidget::itemSelectionChanged,
                    this,
                    [this, tree]() {
                        if (_atlasOverlay) {
                            _atlasOverlay->setSelectedFiberPaths(atlasSelectedFiberPathKeys(tree));
                        }
                    });
            connect(tree,
                    &QTreeWidget::itemDoubleClicked,
                    this,
                    [openFromItem](QTreeWidgetItem* item, int) {
                        openFromItem(item);
                    });
            connect(tree,
                    &QTreeWidget::customContextMenuRequested,
                    this,
                    [this, tree, itemFiberId, focusAtlasControlItem](const QPoint& pos) {
                        QTreeWidgetItem* item = tree->itemAt(pos);
                        if (!item) {
                            return;
                        }
                        if (!tree->selectionModel()->isSelected(tree->indexFromItem(item))) {
                            tree->setCurrentItem(item);
                        }
                        bool hasControlIndex = false;
                        const int controlIndex = item->data(0, ATLAS_CONTROL_INDEX_ROLE).toInt(&hasControlIndex);
                        if (!hasControlIndex || controlIndex < 0) {
                            return;
                        }
                        bool hasSourceIndex = false;
                        const int sourceIndex = item->data(0, ATLAS_CONTROL_SOURCE_INDEX_ROLE).toInt(&hasSourceIndex);
                        const uint64_t fiberId = itemFiberId(item);
                        QMenu menu(tree);
                        auto* showAction = menu.addAction(tr("Show in line annotation"));
                        showAction->setEnabled(fiberId != 0 && _lineAnnotationController != nullptr);
                        QAction* selected = menu.exec(tree->viewport()->mapToGlobal(pos));
                        if (selected == showAction && fiberId != 0 && _lineAnnotationController) {
                            focusAtlasControlItem(item);
                            if (_fiberWidget) {
                                _fiberWidget->selectFiber(fiberId);
                            }
                            if (_fiberSliceWidget) {
                                _fiberSliceWidget->selectFiber(fiberId);
                            }
                            if (hasSourceIndex && sourceIndex >= 0) {
                                _lineAnnotationController->openFiberAtLinePointIndex(fiberId, sourceIndex);
                            } else {
                                _lineAnnotationController->openFiberAtControlPoint(fiberId, controlIndex);
                            }
                        }
                    });
        }
    }

    auto connectSearchDock = [this](QDockWidget* dock) {
        if (!dock || !dock->widget()) {
            return;
        }
        if (auto* run = dock->widget()->findChild<QPushButton*>(QStringLiteral("atlasSearchRunButton"))) {
            connect(run, &QPushButton::clicked, this, &CWindow::startAtlasFiberIntersectionSearch);
        }
        if (auto* cancel = dock->widget()->findChild<QPushButton*>(QStringLiteral("atlasSearchCancelButton"))) {
            connect(cancel, &QPushButton::clicked, this, &CWindow::cancelAtlasFiberIntersectionSearch);
        }
        if (auto* combo = dock->widget()->findChild<QComboBox*>(QStringLiteral("atlasSearchTypeCombo"))) {
            connect(combo,
                    QOverload<int>::of(&QComboBox::currentIndexChanged),
                    this,
                    [this]() { updateAtlasSearchDocks(); });
        }
        if (auto* maxDistance =
                dock->widget()->findChild<QDoubleSpinBox*>(QStringLiteral("atlasSearchMaxDistanceSpin"))) {
            connect(maxDistance,
                    QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                    this,
                    [this, dock](double value) {
                        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                        settings.setValue(vc3d::settings::atlas::SEARCH_MAX_DISTANCE, value);
                        for (auto* otherDock : {_atlasSearchDock, _atlasWorkspaceSearchDock}) {
                            if (!otherDock || otherDock == dock || !otherDock->widget()) {
                                continue;
                            }
                            if (auto* otherSpin = otherDock->widget()->findChild<QDoubleSpinBox*>(
                                    QStringLiteral("atlasSearchMaxDistanceSpin"))) {
                                const QSignalBlocker blocker(otherSpin);
                                otherSpin->setValue(value);
                            }
                        }
                    });
        }
        if (auto* groupByFiber =
                dock->widget()->findChild<QCheckBox*>(QStringLiteral("atlasSearchGroupByFiberCheck"))) {
            connect(groupByFiber,
                    &QCheckBox::toggled,
                    this,
                    [this, dock](bool checked) {
                        for (auto* otherDock : {_atlasSearchDock, _atlasWorkspaceSearchDock}) {
                            if (!otherDock || otherDock == dock || !otherDock->widget()) {
                                continue;
                            }
                            if (auto* otherCheck = otherDock->widget()->findChild<QCheckBox*>(
                                    QStringLiteral("atlasSearchGroupByFiberCheck"))) {
                                const QSignalBlocker blocker(otherCheck);
                                otherCheck->setChecked(checked);
                            }
                        }
                        if (!_atlasSearchCancelFlag && !_atlasSearchResults.empty()) {
                            populateAtlasSearchResults(_atlasSearchResults);
                        }
                    });
        }
        if (auto* tree = dock->widget()->findChild<QTreeWidget*>(QStringLiteral("atlasSearchResultTree"))) {
            auto openFromItem = [this, tree](QTreeWidgetItem* item) {
                if (!item || tree->property("vc_atlas_opening_result").toBool()) {
                    return;
                }
                const auto resultIndex = atlasSearchResultIndexForItem(item);
                if (!resultIndex) {
                    return;
                }
                tree->setProperty("vc_atlas_opening_result", true);
                QTimer::singleShot(0, tree, [tree]() {
                    tree->setProperty("vc_atlas_opening_result", false);
                });
                openAtlasSearchResult(*resultIndex);
            };
            connect(tree,
                    &QTreeWidget::itemDoubleClicked,
                    this,
                    openFromItem);
            connect(tree,
                    &QTreeWidget::itemEntered,
                    this,
                    [this](QTreeWidgetItem* item, int /*column*/) {
                        setAtlasSearchHoverResult(atlasSearchResultIndexForItem(item));
                    });
            if (auto* viewport = tree->viewport()) {
                viewport->setMouseTracking(true);
                viewport->installEventFilter(new FunctionEventFilter(
                    [this](QObject*, QEvent* event) {
                        if (event && event->type() == QEvent::Leave) {
                            setAtlasSearchHoverResult(std::nullopt);
                        }
                        return false;
                    },
                    viewport));
            }
            connect(tree,
                    &QTreeWidget::itemSelectionChanged,
                    this,
                    [this, tree]() {
                        if (tree->property("vc_atlas_syncing_selection").toBool()) {
                            return;
                        }
                        updateAtlasSearchSelectionFromTree(tree);
                    });
            connect(tree,
                    &QWidget::customContextMenuRequested,
                    this,
                    [this, tree](const QPoint& pos) {
                        if (!_lineAnnotationController) {
                            return;
                        }
                        QTreeWidgetItem* item = tree->itemAt(pos);
                        const auto runtimeFiberId = atlasSearchFiberIdForItem(item);
                        if (!runtimeFiberId) {
                            return;
                        }
                        const auto snapshotIt =
                            _atlasSearchFiberSnapshotsByRuntimeId.find(*runtimeFiberId);
                        if (snapshotIt == _atlasSearchFiberSnapshotsByRuntimeId.end() ||
                            snapshotIt->second.storedFiberId == 0) {
                            return;
                        }

                        QMenu menu(tree);
                        QAction* createAtlas = menu.addAction(tr("Create atlas from fiber"));
                        QAction* chosen = menu.exec(tree->viewport()->mapToGlobal(pos));
                        if (chosen == createAtlas) {
                            _lineAnnotationController->createAtlasFromFiber(
                                snapshotIt->second.storedFiberId);
                        }
                    });
        }
    };
    connectSearchDock(_atlasSearchDock);
    connectSearchDock(_atlasWorkspaceSearchDock);
    refreshAtlasOverviewDocks();
    updateAtlasFiberDocks();
    updateAtlasSearchDocks();
}

void CWindow::refreshAtlasOverviewDocks()
{
    auto vpkg = _state ? _state->vpkg() : nullptr;
    const std::filesystem::path volpkgRoot = vpkg && !vpkg->path().empty()
        ? vpkg->path().parent_path()
        : std::filesystem::path{};

    auto populate = [this, &volpkgRoot](QDockWidget* dock) {
        if (!dock || !dock->widget()) {
            return;
        }
        auto* tree = dock->widget()->findChild<QTreeWidget*>(QStringLiteral("atlasOverviewTree"));
        if (!tree) {
            return;
        }
        tree->clear();
        for (const auto& atlasInfo : vc::atlas::discoverAtlasDirectories(volpkgRoot)) {
            const std::filesystem::path& atlasDir = atlasInfo.path;
            try {
                const auto atlas = vc::atlas::Atlas::load(atlasDir);
                auto* item = new QTreeWidgetItem(tree);
                item->setText(0, QString::fromStdString(atlas.metadata.name));
                item->setData(0, Qt::UserRole, QString::fromStdString(atlasDir.string()));

                auto* fiberCount = new QTreeWidgetItem(item);
                fiberCount->setText(0, tr("Fiber count"));
                fiberCount->setText(1, QString::number(static_cast<int>(atlas.fibers.size())));

                auto* fibersRoot = new QTreeWidgetItem(item);
                fibersRoot->setText(0, tr("Fibers"));
                fibersRoot->setText(1, QString::number(static_cast<int>(atlas.fibers.size())));
                for (const auto& mapping : atlas.fibers) {
                    auto* fiberItem = new QTreeWidgetItem(fibersRoot);
                    fiberItem->setText(0, QString::fromStdString(mapping.fiberPath.generic_string()));
                    fiberItem->setText(
                        1,
                        tr("%1 line anchors, %2 control anchors")
                            .arg(static_cast<int>(mapping.lineAnchors.size()))
                            .arg(static_cast<int>(mapping.controlAnchors.size())));
                }

                auto* coveredSize = new QTreeWidgetItem(item);
                coveredSize->setText(0, tr("Object covered atlas size"));
                const std::filesystem::path basePath = atlasDir / atlas.metadata.baseMeshPath;
                const QuadSurface baseSurface(basePath);
                const auto* basePoints = baseSurface.rawPointsPtr();
                if (!basePoints || basePoints->empty() || basePoints->cols <= 0) {
                    throw std::runtime_error("atlas base mesh has no valid grid");
                }
                coveredSize->setText(1, formatAtlasCoveredSize(
                    vc::atlas::mappedObjectCoveredAtlasSize(
                        atlas,
                        baseSurface.scale(),
                        vc::atlas::atlasHorizontalPeriodColumns(baseSurface))));
            } catch (const std::exception& ex) {
                auto* item = new QTreeWidgetItem(tree);
                item->setText(0, QString::fromStdString(
                    atlasInfo.name.empty() ? atlasDir.filename().string() : atlasInfo.name));
                item->setText(1, QString::fromStdString(ex.what()));
                item->setData(0, Qt::UserRole, QString::fromStdString(atlasDir.string()));
            }
        }
        tree->expandAll();
    };

    populate(_atlasOverviewDock);
    populate(_atlasWorkspaceOverviewDock);
}

void CWindow::updateAtlasFiberDocks()
{
    if (!_atlasWorkspaceFiberDock || !_atlasWorkspaceFiberDock->widget()) {
        return;
    }

    auto* label = _atlasWorkspaceFiberDock->widget()->findChild<QLabel*>(QStringLiteral("atlasFiberCurrentLabel"));
    auto* tree = _atlasWorkspaceFiberDock->widget()->findChild<QTreeWidget*>(QStringLiteral("atlasFiberTree"));
    auto* optimize = _atlasWorkspaceFiberDock->widget()->findChild<QPushButton*>(
        QStringLiteral("atlasOptimizeSnapCandidatesButton"));
    auto updateOptimizeEnabled = [&]() {
        bool hasManifest = false;
        if (_state && _state->vpkg()) {
            try {
                const auto resolved = resolvedLasagnaForState(_state);
                hasManifest = resolved && !resolved->manifestPath.empty() &&
                              std::filesystem::exists(resolved->manifestPath);
            } catch (...) {
                hasManifest = false;
            }
        }
        if (optimize) {
            optimize->setEnabled(_currentAtlasDir.has_value() &&
                                 hasManifest &&
                                 LasagnaServiceManager::instance().isRunning());
        }
    };
    if (!tree) {
        updateOptimizeEnabled();
        return;
    }

    const QSignalBlocker treeBlocker(tree);
    tree->setSortingEnabled(false);
    tree->clear();

    if (!_currentAtlasDir) {
        if (label) {
            label->setText(tr("No atlas selected"));
        }
        updateOptimizeEnabled();
        return;
    }

    vc::atlas::Atlas atlas;
    try {
        auto vpkg = _state ? _state->vpkg() : nullptr;
        std::filesystem::path volpkgRoot;
        if (vpkg) {
            volpkgRoot = vpkg->path().empty()
                ? std::filesystem::path(vpkg->getVolpkgDirectory())
                : vpkg->path().parent_path();
        }
        atlas = volpkgRoot.empty()
            ? vc::atlas::Atlas::load(*_currentAtlasDir)
            : vc::atlas::Atlas::load(*_currentAtlasDir, volpkgRoot);
    } catch (const std::exception& ex) {
        if (label) {
            label->setText(tr("Atlas: %1 (could not load: %2)")
                .arg(QString::fromStdString(_currentAtlasName.empty()
                         ? _currentAtlasDir->filename().string()
                         : _currentAtlasName),
                     QString::fromStdString(ex.what())));
        }
        updateOptimizeEnabled();
        return;
    }

    if (label) {
        label->setText(tr("Atlas: %1 (%2 mapped fibers)")
            .arg(QString::fromStdString(atlas.metadata.name.empty()
                     ? _currentAtlasDir->filename().string()
                     : atlas.metadata.name))
            .arg(static_cast<int>(atlas.fibers.size())));
    }

    for (const auto& mapping : atlas.fibers) {
        const std::optional<uint64_t> matchedFiberId = _lineAnnotationController
            ? _lineAnnotationController->fiberIdForAtlasPath(mapping.fiberPath)
            : std::nullopt;
        const uint64_t fiberId = matchedFiberId.value_or(0);
        const QString fiberPathKey = QString::fromStdString(
            vc::atlas::atlasFiberPathKey(mapping.fiberPath));

        int snapTotal = 0;
        int snapFound = 0;
        vc::atlas::AtlasPredSnapSet predSnapSet;
        std::unordered_map<std::string, const vc::atlas::AtlasPredSnapPoint*> snapsByControl;
        try {
            predSnapSet = vc::atlas::loadAtlasPredSnapSet(
                vc::atlas::atlasPredSnapAttachmentPath(*_currentAtlasDir, mapping.fiberPath));
            snapTotal = static_cast<int>(predSnapSet.points.size());
            snapFound = static_cast<int>(std::count_if(predSnapSet.points.begin(),
                                                       predSnapSet.points.end(),
                                                       [](const vc::atlas::AtlasPredSnapPoint& point) {
                                                           return isResolvedAtlasPredSnap(point);
                                                       }));
            for (const auto& point : predSnapSet.points) {
                snapsByControl[vc::atlas::atlasPredSnapControlPointKey(point.controlPoint)] = &point;
            }
        } catch (const std::exception&) {
            snapTotal = -1;
            snapFound = -1;
        }

        const std::filesystem::path fiberNamePath = mapping.fiberPath.filename();
        const std::string fiberLabel = fiberNamePath.empty()
            ? mapping.fiberPath.generic_string()
            : fiberNamePath.string();
        auto* fiberItem = new QTreeWidgetItem(tree);
        fiberItem->setText(0, QString::fromStdString(fiberLabel));
        fiberItem->setText(1, tr("%1 controls").arg(static_cast<int>(mapping.controlAnchors.size())));
        fiberItem->setText(3, snapTotal < 0 ? tr("missing") : tr("%1/%2").arg(snapFound).arg(snapTotal));
        fiberItem->setText(6, snapTotal < 0 ? tr("missing") : tr("%1 records").arg(snapTotal));
        fiberItem->setText(8, fiberId == 0 ? tr("not loaded") : QString::number(fiberId));
        fiberItem->setToolTip(0, QString::fromStdString(mapping.fiberPath.generic_string()));
        fiberItem->setData(0, ATLAS_FIBER_ID_ROLE, QVariant::fromValue<qulonglong>(fiberId));
        fiberItem->setData(0, ATLAS_FIBER_PATH_KEY_ROLE, fiberPathKey);
        fiberItem->setData(0, ATLAS_CONTROL_INDEX_ROLE, -1);
        fiberItem->setData(0, ATLAS_CONTROL_SOURCE_INDEX_ROLE, -1);

        for (int controlIndex = 0;
             controlIndex < static_cast<int>(mapping.controlAnchors.size());
             ++controlIndex) {
            const auto& anchor = mapping.controlAnchors[static_cast<size_t>(controlIndex)];
            const auto surfaceCoord = _atlasOverlay
                ? _atlasOverlay->atlasAnchorToSurface(anchor, mapping)
                : std::optional<cv::Vec2f>{};
            const vc::atlas::AtlasPredSnapPoint* snap =
                predSnapPointForAnchor(snapsByControl, anchor);
            const bool snapResolved = snap && isResolvedAtlasPredSnap(*snap);

            auto* row = new QTreeWidgetItem(fiberItem);
            row->setText(0, tr("CP %1").arg(controlIndex));
            row->setText(1, QString::number(anchor.sourceIndex));
            row->setText(2, QString::number(anchor.distance, 'f', 3));
            row->setText(3, snapResolved
                ? predSnapDirectionLabel(snap->direction)
                : (snap ? tr("none") : tr("-")));
            row->setText(4, snapResolved ? formatOptionalDouble(predSnapDisplayWinding(*snap), 3)
                                         : QStringLiteral("-"));
            row->setText(5, snapResolved
                ? formatAtlasVec3(*snap->predSnapPoint)
                : QStringLiteral("-"));
            row->setText(6, predSnapStatusLabel(snap));
            row->setText(7, predSnapStatusReason(snap));
            row->setText(8, fiberId == 0 ? tr("not loaded") : QString::number(fiberId));
            row->setData(0, ATLAS_FIBER_ID_ROLE, QVariant::fromValue<qulonglong>(fiberId));
            row->setData(0, ATLAS_FIBER_PATH_KEY_ROLE, fiberPathKey);
            row->setData(0, ATLAS_CONTROL_INDEX_ROLE, controlIndex);
            row->setData(0, ATLAS_CONTROL_SOURCE_INDEX_ROLE, anchor.sourceIndex);
            if (surfaceCoord) {
                row->setData(0, ATLAS_SURFACE_X_ROLE, (*surfaceCoord)[0]);
                row->setData(0, ATLAS_SURFACE_Y_ROLE, (*surfaceCoord)[1]);
            }
            row->setToolTip(0, tr("Control point: %1").arg(formatAtlasVec3(anchor.world)));
            row->setToolTip(6, predSnapStatusReason(snap));
            row->setToolTip(7, predSnapStatusReason(snap));
            if (snapResolved) {
                row->setToolTip(5, tr("Snap target: %1").arg(formatAtlasVec3(*snap->predSnapPoint)));
            }
        }
    }

    tree->collapseAll();
    updateOptimizeEnabled();
}

void CWindow::updateAtlasSearchDocks()
{
    const auto snapshots = _lineAnnotationController
        ? _lineAnnotationController->fiberSnapshotsFromStorage()
        : std::vector<vc::atlas::FiberPolyline>{};
    int savedFiberCount = 0;
    QString atlasLoadError;
    if (_currentAtlasDir) {
        try {
            const vc::atlas::Atlas atlas = vc::atlas::Atlas::load(*_currentAtlasDir);
            savedFiberCount = static_cast<int>(atlas.fibers.size());
        } catch (const std::exception& ex) {
            atlasLoadError = QString::fromStdString(ex.what());
        }
    }
    const bool hasSnapshots = !snapshots.empty();
    const bool atlasUsable = _currentAtlasDir.has_value() && atlasLoadError.isEmpty();
    const QString atlasName = _currentAtlasDir
        ? QString::fromStdString(_currentAtlasName.empty()
              ? _currentAtlasDir->filename().string()
              : _currentAtlasName)
        : QString{};
    const QString atlasText = !_currentAtlasDir
        ? tr("No atlas selected (non-atlas search available)")
        : (!atlasLoadError.isEmpty()
              ? tr("Atlas: %1 (could not load: %2)").arg(atlasName, atlasLoadError)
              : tr("Atlas: %1 (%2 saved fibers)").arg(atlasName).arg(savedFiberCount));

    for (auto* dock : {_atlasSearchDock, _atlasWorkspaceSearchDock}) {
        if (!dock || !dock->widget()) {
            continue;
        }
        if (auto* label = dock->widget()->findChild<QLabel*>(QStringLiteral("atlasSearchCurrentLabel"))) {
            label->setText(atlasText);
        }
        int searchMode = ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS;
        if (auto* combo = dock->widget()->findChild<QComboBox*>(QStringLiteral("atlasSearchTypeCombo"))) {
            combo->setEnabled(hasSnapshots);
            searchMode = combo->currentData().toInt();
        }
        if (auto* run = dock->widget()->findChild<QPushButton*>(QStringLiteral("atlasSearchRunButton"))) {
            run->setEnabled(hasSnapshots &&
                            (searchMode == ATLAS_SEARCH_MODE_NON_ATLAS_ONLY || atlasUsable));
        }
        if (auto* tagFilter = dock->widget()->findChild<QLineEdit*>(
                QStringLiteral("atlasSearchTagFilterEdit"))) {
            tagFilter->setEnabled(hasSnapshots);
        }
        if (auto* excludeTagFilter = dock->widget()->findChild<QLineEdit*>(
                QStringLiteral("atlasSearchExcludeTagFilterEdit"))) {
            excludeTagFilter->setEnabled(hasSnapshots);
        }
        if (auto* cancel = dock->widget()->findChild<QPushButton*>(QStringLiteral("atlasSearchCancelButton"))) {
            cancel->setEnabled(false);
        }
    }

    for (auto* dock : {_atlasOverviewDock, _atlasWorkspaceOverviewDock}) {
        if (!dock || !dock->widget()) {
            continue;
        }
        if (auto* rankSnap = dock->widget()->findChild<QPushButton*>(QStringLiteral("atlasRankSnapButton"))) {
            rankSnap->setEnabled(atlasUsable);
        }
        if (auto* remap = dock->widget()->findChild<QPushButton*>(QStringLiteral("atlasRemapButton"))) {
            remap->setEnabled(atlasUsable);
        }
    }
}

// Dialog-free core shared with remapCurrentAtlas.
bool CWindow::startAtlasRemapHeadless(QString* errorMessage,
                                      std::function<void(bool success, const QString& detail)> onFinished)
{
    auto fail = [errorMessage](const QString& message) {
        if (errorMessage) {
            *errorMessage = message;
        }
        return false;
    };
    if (!_currentAtlasDir || !_state || !_state->vpkg()) {
        return fail(tr("Load an atlas before remapping."));
    }
    auto vpkg = _state->vpkg();
    std::optional<vc3d::opendata::ResolvedOpenDataLasagna> resolvedLasagna;
    try {
        resolvedLasagna = resolvedLasagnaForState(_state);
    } catch (const std::exception& ex) {
        return fail(QString::fromStdString(ex.what()));
    }
    if (!resolvedLasagna || resolvedLasagna->manifestPath.empty() ||
        !std::filesystem::exists(resolvedLasagna->manifestPath)) {
        return fail(tr("No Lasagna dataset matches the active volume."));
    }
    const std::filesystem::path manifestPath = resolvedLasagna->manifestPath;
    const double workingToBaseScale = resolvedLasagna->workingToBaseScale;

    const std::filesystem::path atlasDir = *_currentAtlasDir;
    const std::filesystem::path volpkgRoot = vpkg->path().empty()
        ? std::filesystem::path(vpkg->getVolpkgDirectory())
        : vpkg->path().parent_path();

    if (statusBar()) {
        showStatusBarMessage(tr("Remapping atlas fibers..."), 3000);
    }
    for (auto* dock : {_atlasOverviewDock, _atlasWorkspaceOverviewDock}) {
        if (!dock || !dock->widget()) {
            continue;
        }
        if (auto* button = dock->widget()->findChild<QPushButton*>(
                QStringLiteral("atlasRemapButton"))) {
            button->setEnabled(false);
        }
        if (auto* button = dock->widget()->findChild<QPushButton*>(
                QStringLiteral("atlasRankSnapButton"))) {
            button->setEnabled(false);
        }
    }

    auto* watcher = new QFutureWatcher<QString>(this);
    connect(watcher,
            &QFutureWatcher<QString>::finished,
            this,
            [this, watcher, atlasDir, onFinished = std::move(onFinished)]() {
        watcher->deleteLater();
        try {
            const QString summary = watcher->result();
            // A remap cannot require rebuilding; report redisplay failures
            // through the completion callback.
            QString displayError;
            if (!displayAtlasFromDirectoryHeadless(atlasDir, &displayError)) {
                throw std::runtime_error(displayError.toStdString());
            }
            if (statusBar()) {
                showStatusBarMessage(tr("Remapped atlas fibers. %1").arg(summary), 7000);
            }
            if (onFinished) {
                onFinished(true, summary);
            }
        } catch (const std::exception& ex) {
            refreshAtlasOverviewDocks();
            updateAtlasFiberDocks();
            updateAtlasSearchDocks();
            const QString detail = QString::fromStdString(ex.what());
            if (statusBar()) {
                showStatusBarMessage(tr("Could not remap atlas: %1").arg(detail), 7000);
            }
            if (onFinished) {
                onFinished(false, detail);
            }
        }
    });

    watcher->setFuture(QtConcurrent::run(
        [atlasDir, volpkgRoot, manifestPath, workingToBaseScale]() -> QString {
        std::cerr << "[atlas-remap] start"
                  << " atlas=" << atlasDir.string()
                  << " volpkg_root=" << volpkgRoot.string()
                  << " manifest=" << manifestPath.string()
                  << std::endl;
        vc::lasagna::LasagnaDataset dataset =
            vc::lasagna::LasagnaDataset::open(manifestPath, {workingToBaseScale});
        vc::lasagna::LasagnaNormalSampler sampler(dataset);
        const vc::atlas::Atlas rebuilt =
            vc::atlas::rebuildAtlasFromSourceFibers(atlasDir, volpkgRoot, sampler);

        std::ostringstream summary;
        summary.imbue(std::locale::classic());
        summary << "fibers=" << rebuilt.fibers.size();
        for (const auto& mapping : rebuilt.fibers) {
            std::cerr << "[atlas-remap] fiber="
                      << mapping.fiberPath.generic_string()
                      << " line_anchors=" << mapping.lineAnchors.size()
                      << " control_anchors=" << mapping.controlAnchors.size()
                      << std::endl;
            summary << " "
                    << mapping.fiberPath.filename().string()
                    << ":"
                    << mapping.controlAnchors.size()
                    << "cp";
        }
        std::cerr << "[atlas-remap] finished" << std::endl;
        return QString::fromStdString(summary.str());
    }));
    return true;
}

void CWindow::remapCurrentAtlas()
{
    QString errorMessage;
    const bool started = startAtlasRemapHeadless(
        &errorMessage,
        [this](bool success, const QString& detail) {
            if (!success) {
                QMessageBox::warning(this,
                                     tr("Atlas Remap"),
                                     tr("Could not remap atlas: %1").arg(detail));
            }
        });
    if (!started) {
        QMessageBox::warning(this, tr("Atlas"), errorMessage);
    }
}

// Dialog-free core shared with optimizeAtlasSnapCandidates.
bool CWindow::optimizeAtlasSnapCandidatesHeadless(QString* errorMessage,
                                                  std::function<void(const QString& detail)> onAsyncError)
{
    auto fail = [errorMessage](const QString& message) {
        if (errorMessage) {
            *errorMessage = message;
        }
        return false;
    };
    if (!_currentAtlasDir || !_state || !_state->vpkg()) {
        return fail(tr("Load an atlas before ranking snap candidates."));
    }
    auto vpkg = _state->vpkg();
    std::optional<vc3d::opendata::ResolvedOpenDataLasagna> resolvedLasagna;
    try {
        resolvedLasagna = resolvedLasagnaForState(_state);
    } catch (const std::exception& ex) {
        return fail(QString::fromStdString(ex.what()));
    }
    if (!resolvedLasagna || resolvedLasagna->manifestPath.empty() ||
        !std::filesystem::exists(resolvedLasagna->manifestPath)) {
        return fail(tr("No Lasagna dataset matches the active volume."));
    }
    const std::filesystem::path manifestPath = resolvedLasagna->manifestPath;
    const double workingToBaseScale = resolvedLasagna->workingToBaseScale;

    auto& manager = LasagnaServiceManager::instance();
    if (manager.isExternal()) {
        if (resolvedLasagna->manifestBacked) {
            return fail(tr("Manifest-backed Lasagna is available to local tools only. "
                           "The external service must use a dataset installed on that service."));
        }
        if (!manager.isRunning()) {
            return fail(tr("Connect the external Lasagna service before ranking snap candidates."));
        }
    } else if (!manager.ensureServiceRunning()) {
        return fail(tr("Failed to start Lasagna service: %1").arg(manager.lastError()));
    }

    // Shared failure reporting for the completion paths below. Interactive
    // callers can add a message box through onAsyncError.
    auto reportAsyncError = [this, onAsyncError = std::move(onAsyncError)](const QString& message) {
        if (statusBar()) {
            showStatusBarMessage(tr("Could not rank snap candidates: %1").arg(message), 7000);
        }
        if (onAsyncError) {
            onAsyncError(message);
        }
    };

    const std::filesystem::path atlasDir = *_currentAtlasDir;
    const std::filesystem::path volpkgRoot = vpkg->path().empty()
        ? std::filesystem::path(vpkg->getVolpkgDirectory())
        : vpkg->path().parent_path();
    const std::string serviceManifestPath = manager.isExternal()
        ? manifestPath.filename().generic_string()
        : manifestPath.string();
    if (statusBar()) {
        showStatusBarMessage(tr("Preparing atlas snap candidates..."), 3000);
    }
    auto setRankButtonsEnabled = [this](bool enabled) {
        for (auto* dock : {_atlasOverviewDock, _atlasWorkspaceOverviewDock}) {
            if (!dock || !dock->widget()) {
                continue;
            }
            if (auto* button = dock->widget()->findChild<QPushButton*>(
                    QStringLiteral("atlasRankSnapButton"))) {
                button->setEnabled(enabled);
            }
        }
    };
    setRankButtonsEnabled(false);

    QPointer<CWindow> self(this);
    auto startFinish =
        [this, self, atlasDir, setRankButtonsEnabled, reportAsyncError](
            vc::atlas::AtlasSnapPreparedCandidates prepared,
            nlohmann::json rankResponse) {
        if (!self) {
            return;
        }
        if (statusBar()) {
            showStatusBarMessage(tr("Applying atlas snap candidate ranking..."), 3000);
        }
        auto* finishWatcher = new QFutureWatcher<vc::atlas::AtlasSnapOptimizeReport>(this);
        connect(finishWatcher,
                &QFutureWatcher<vc::atlas::AtlasSnapOptimizeReport>::finished,
                this,
                [this, finishWatcher, atlasDir, setRankButtonsEnabled, reportAsyncError]() {
            finishWatcher->deleteLater();
            setRankButtonsEnabled(true);
            try {
                const vc::atlas::AtlasSnapOptimizeReport report = finishWatcher->result();
                refreshAtlasOverviewDocks();
                // Redisplay failures enter the catch below.
                loadAndDisplayAtlas(atlasDir);
                if (statusBar()) {
                    showStatusBarMessage(
                        tr("Ranked snap candidates: %1 controls, terms %2 ok / %3 zero / %4 skipped (%5 total), %6 queued, %7 cached.")
                            .arg(report.controls)
                            .arg(report.successfulPairTerms)
                            .arg(report.zeroContributionTerms)
                            .arg(report.skippedPairTerms)
                            .arg(report.pairTerms)
                            .arg(report.rankJobsRequested)
                            .arg(report.cacheHits),
                        6000);
                }
                std::cerr << "[atlas-snap] rank finished"
                          << " controls=" << report.controls
                          << " variables=" << report.variableControls
                          << " fixed=" << report.fixedControls
                          << " manual=" << report.manualControls
                          << " singleton_auto=" << report.singletonControls
                          << " links=" << report.links
                          << " pair_terms=" << report.pairTerms
                          << " successful_terms=" << report.successfulPairTerms
                          << " zero_contribution_terms=" << report.zeroContributionTerms
                          << " skipped_terms=" << report.skippedPairTerms
                          << " queued=" << report.rankJobsRequested
                          << " cached=" << report.cacheHits
                          << " objective=" << report.objective
                          << std::endl;
            } catch (const std::exception& ex) {
                refreshAtlasOverviewDocks();
                updateAtlasSearchDocks();
                const QString message = extractFutureExceptionMessage(ex);
                std::cerr << "[atlas-snap] finish failed: "
                          << message.toStdString() << std::endl;
                reportAsyncError(message);
            }
        });
        finishWatcher->setFuture(QtConcurrent::run(
            [prepared = std::move(prepared), rankResponse = std::move(rankResponse)]() mutable {
                return vc::atlas::finishAtlasPredSnapCandidates(prepared, rankResponse);
            }));
    };

    auto* prepareWatcher =
        new QFutureWatcher<vc::atlas::AtlasSnapPreparedCandidates>(this);
    connect(prepareWatcher,
            &QFutureWatcher<vc::atlas::AtlasSnapPreparedCandidates>::finished,
            this,
            [this,
             prepareWatcher,
             startFinish,
             self,
             serviceManifestPath,
             workingToBaseScale,
             setRankButtonsEnabled,
             reportAsyncError]() mutable {
        prepareWatcher->deleteLater();
        try {
            vc::atlas::AtlasSnapPreparedCandidates prepared = prepareWatcher->result();
            const size_t jobCount =
                prepared.rankRequest.value("jobs", nlohmann::json::array()).size();
            if (jobCount == 0) {
                std::cerr << "[atlas-snap] no laplace rank jobs required; finishing from cache"
                          << std::endl;
                startFinish(std::move(prepared), nlohmann::json::object());
                return;
            }

            nlohmann::json serviceRequest = prepared.rankRequest;
            serviceRequest["manifest"] = serviceManifestPath;
            serviceRequest["working_to_base_scale"] = workingToBaseScale;
            std::cerr << "[atlas-snap] requesting laplace rank jobs="
                      << jobCount
                      << " service_manifest=" << serviceManifestPath
                      << std::endl;
            QJsonObject qtRequest = toQtJsonObject(serviceRequest);
            if (statusBar()) {
                showStatusBarMessage(tr("Ranking atlas snap candidates..."), 3000);
            }
            LasagnaServiceManager::instance().rankLaplaceSnapPairs(
                qtRequest,
                [startFinish, prepared](const QJsonObject& response) mutable {
                    std::cerr << "[atlas-snap] laplace rank response received"
                              << std::endl;
                    startFinish(prepared, fromQtJsonObject(response));
                },
                [self, setRankButtonsEnabled, reportAsyncError](const QString& message) {
                    if (!self) {
                        return;
                    }
                    setRankButtonsEnabled(true);
                    self->refreshAtlasOverviewDocks();
                    self->updateAtlasSearchDocks();
                    std::cerr << "[atlas-snap] laplace rank failed: "
                              << message.toStdString() << std::endl;
                    reportAsyncError(message);
                },
                [self, prepared](int index, const QJsonObject& result) mutable {
                    if (!self) {
                        return;
                    }
                    try {
                        vc::atlas::cacheAtlasPredSnapRankResult(
                            prepared,
                            static_cast<size_t>(index),
                            fromQtJsonObject(result));
                    } catch (const std::exception& ex) {
                        std::cerr << "[atlas-snap] partial rank result callback failed: "
                                  << ex.what()
                                  << std::endl;
                    } catch (...) {
                        std::cerr << "[atlas-snap] partial rank result callback failed: "
                                  << "unknown non-standard exception"
                                  << std::endl;
                    }
                });
        } catch (const std::exception& ex) {
            setRankButtonsEnabled(true);
            refreshAtlasOverviewDocks();
            updateAtlasSearchDocks();
            const QString message = extractFutureExceptionMessage(ex);
            std::cerr << "[atlas-snap] prepare failed: "
                      << message.toStdString() << std::endl;
            reportAsyncError(message);
        }
    });

    prepareWatcher->setFuture(QtConcurrent::run([atlasDir,
                                                  volpkgRoot,
                                                  manifestPath,
                                                  workingToBaseScale]() {
        try {
            std::cerr << "[atlas-snap] prepare start"
                      << " atlas=" << atlasDir.string()
                      << " volpkg_root=" << volpkgRoot.string()
                      << " manifest=" << manifestPath.string()
                      << std::endl;
            vc::lasagna::LasagnaDataset dataset =
                vc::lasagna::LasagnaDataset::open(
                    manifestPath, {workingToBaseScale});
            vc::lasagna::LasagnaNormalSampler sampler(dataset);
            if (!sampler.hasPredDtChannel()) {
                throw std::runtime_error("selected Lasagna dataset has no pred_dt channel: " +
                                         manifestPath.string());
            }

            vc::atlas::AtlasSnapOptimizeOptions options;
            options.predDtThreshold = 110;
            options.rankOptions = {
                {"threshold", options.predDtThreshold},
                {"margin_base_voxels", 1000},
                {"source_depth", 0},
                {"amgx_config", nullptr},
            };

            vc::atlas::AtlasSnapPreparedCandidates prepared =
                vc::atlas::prepareAtlasPredSnapCandidates(
                    atlasDir,
                    volpkgRoot,
                    manifestPath,
                    sampler,
                    options);
            const size_t jobCount =
                prepared.rankRequest.value("jobs", nlohmann::json::array()).size();
            std::cerr << "[atlas-snap] prepare finished"
                      << " queued=" << jobCount
                      << std::endl;
            return prepared;
        } catch (const std::exception& ex) {
            std::cerr << "[atlas-snap] prepare exception: "
                      << ex.what() << std::endl;
            throw;
        } catch (...) {
            std::cerr << "[atlas-snap] prepare exception: unknown non-standard exception"
                      << std::endl;
            throw;
        }
    }));
    return true;
}

void CWindow::optimizeAtlasSnapCandidates()
{
    QString errorMessage;
    const bool started = optimizeAtlasSnapCandidatesHeadless(
        &errorMessage,
        [this](const QString& detail) {
            QMessageBox::warning(this,
                                 tr("Atlas Snap Candidates"),
                                 tr("Could not rank snap candidates: %1").arg(detail));
        });
    if (!started) {
        QMessageBox::warning(this, tr("Atlas"), errorMessage);
    }
}

void CWindow::cancelAtlasFiberIntersectionSearch()
{
    _atlasSearchCancelRequested = true;
    if (_atlasSearchCancelFlag) {
        _atlasSearchCancelFlag->store(true, std::memory_order_relaxed);
    }
    qInfo().noquote() << QStringLiteral("[atlas-search] cancel requested phase=%1 completed=%2 total=%3")
        .arg(atlasSearchPhaseNumber(_atlasSearchProgressPhase))
        .arg(static_cast<qulonglong>(_atlasSearchPhaseCompleted))
        .arg(static_cast<qulonglong>(_atlasSearchPhaseTotal));
    clearAtlasSearchPreviewState();
    for (auto* dock : {_atlasSearchDock, _atlasWorkspaceSearchDock}) {
        if (!dock || !dock->widget()) {
            continue;
        }
        if (auto* progress = dock->widget()->findChild<QProgressBar*>(QStringLiteral("atlasSearchProgressBar"))) {
            progress->setRange(0, 100);
            progress->setValue(vc::atlas::atlasSearchPhaseProgressPercent(
                _atlasSearchProgressPhase,
                _atlasSearchPhaseCompleted,
                _atlasSearchPhaseTotal));
            progress->setFormat(tr("Canceling: phase %1/%2 %3 (%4 / %5)")
                                    .arg(atlasSearchPhaseNumber(_atlasSearchProgressPhase))
                                    .arg(ATLAS_SEARCH_PHASE_COUNT)
                                    .arg(atlasSearchPhaseAction(_atlasSearchProgressPhase))
                                    .arg(static_cast<qulonglong>(_atlasSearchPhaseCompleted))
                                    .arg(static_cast<qulonglong>(_atlasSearchPhaseTotal)));
        }
        if (auto* cancel = dock->widget()->findChild<QPushButton*>(QStringLiteral("atlasSearchCancelButton"))) {
            cancel->setEnabled(false);
        }
        if (auto* tree = dock->widget()->findChild<QTreeWidget*>(QStringLiteral("atlasSearchResultTree"))) {
            tree->clear();
        }
    }
}

void CWindow::updateAtlasSearchProgress(vc::atlas::AtlasSearchProgressPhase phase,
                                        std::size_t completed,
                                        std::size_t total)
{
    _atlasSearchProgressPhase = phase;
    _atlasSearchPhaseCompleted = total == 0 ? completed : std::min(completed, total);
    _atlasSearchPhaseTotal = total;

    const int percent = vc::atlas::atlasSearchPhaseProgressPercent(
        phase,
        _atlasSearchPhaseCompleted,
        total);
    const QString action = atlasSearchPhaseAction(phase);
    const QString format = _atlasSearchCancelRequested
        ? tr("Canceling: phase %1/%2 %3 (%4 / %5)")
              .arg(atlasSearchPhaseNumber(phase))
              .arg(ATLAS_SEARCH_PHASE_COUNT)
              .arg(action)
              .arg(static_cast<qulonglong>(_atlasSearchPhaseCompleted))
              .arg(static_cast<qulonglong>(total))
        : tr("Phase %1/%2: %3 (%4 / %5)")
              .arg(atlasSearchPhaseNumber(phase))
              .arg(ATLAS_SEARCH_PHASE_COUNT)
              .arg(action)
              .arg(static_cast<qulonglong>(_atlasSearchPhaseCompleted))
              .arg(static_cast<qulonglong>(total));

    for (auto* dock : {_atlasSearchDock, _atlasWorkspaceSearchDock}) {
        if (!dock || !dock->widget()) {
            continue;
        }
        if (auto* progress = dock->widget()->findChild<QProgressBar*>(QStringLiteral("atlasSearchProgressBar"))) {
            progress->setRange(0, 100);
            progress->setValue(percent);
            progress->setFormat(format);
        }
    }

    // Emit the same normalized progress used by non-widget consumers.
    double fraction = 0.0;
    if (total > 0) {
        fraction = static_cast<double>(_atlasSearchPhaseCompleted) /
                   static_cast<double>(total);
    } else if (completed > 0) {
        fraction = 1.0;
    }
    fraction = std::clamp(fraction, 0.0, 1.0);
    emit atlasSearchProgressChanged(atlasSearchPhaseNumber(phase), fraction);
}

void CWindow::startAtlasFiberIntersectionSearch()
{
    if (!_lineAnnotationController) {
        updateAtlasSearchDocks();
        return;
    }

    // Build the shared search parameters from the widgets.
    AtlasFiberSearchParams params;
    params.searchMode = ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS;
    auto readSearchControls = [&](QDockWidget* dock) {
        if (!dock || !dock->widget()) {
            return false;
        }
        bool found = false;
        if (auto* combo = dock->widget()->findChild<QComboBox*>(
                QStringLiteral("atlasSearchTypeCombo"))) {
            params.searchMode = combo->currentData().toInt();
            found = true;
        }
        if (auto* tagFilter = dock->widget()->findChild<QLineEdit*>(
                QStringLiteral("atlasSearchTagFilterEdit"))) {
            params.requiredTags = atlasSearchTagList(tagFilter->text());
            found = true;
        }
        if (auto* excludeTagFilter = dock->widget()->findChild<QLineEdit*>(
                QStringLiteral("atlasSearchExcludeTagFilterEdit"))) {
            params.excludedTags = atlasSearchTagList(excludeTagFilter->text());
            found = true;
        }
        if (auto* spin = dock->widget()->findChild<QDoubleSpinBox*>(
                QStringLiteral("atlasSearchMaxDistanceSpin"))) {
            params.maxDistance = spin->value();
            found = true;
        }
        return found;
    };
    bool readControls = false;
    for (auto* dock : {_atlasSearchDock, _atlasWorkspaceSearchDock}) {
        if (dock && dock->isVisible()) {
            readControls = readSearchControls(dock) || readControls;
        }
    }
    if (!readControls) {
        for (auto* dock : {_atlasSearchDock, _atlasWorkspaceSearchDock}) {
            if (readSearchControls(dock)) {
                break;
            }
        }
    }

    // No saved fibers: silent no-op in the interactive flow (unchanged: no dialog,
    // just refresh the docks).
    if (_lineAnnotationController->fiberSnapshotsFromStorageWithPaths().empty()) {
        updateAtlasSearchDocks();
        return;
    }

    QString errorMessage;
    if (!startAtlasFiberIntersectionSearchHeadless(params, &errorMessage)) {
        QMessageBox::warning(this, tr("Atlas Object Search"), errorMessage);
    }
}

// Dialog-free core shared with startAtlasFiberIntersectionSearch.
bool CWindow::startAtlasFiberIntersectionSearchHeadless(const AtlasFiberSearchParams& params,
                                                        QString* errorMessage)
{
    auto fail = [errorMessage](const QString& message) {
        if (errorMessage) {
            *errorMessage = message;
        }
        return false;
    };
    if (!_lineAnnotationController) {
        updateAtlasSearchDocks();
        return fail(tr("Line annotation is not available."));
    }
    // A live cancel flag means a search is in flight (created at launch and
    // reset by the finished handler).
    if (_atlasSearchCancelFlag) {
        return fail(tr("An atlas object search is already running."));
    }

    vc::atlas::FiberIntersectionBroadPhaseOptions broad;
    if (params.maxDistance) {
        broad.maxDistance = *params.maxDistance;
    } else {
        // The spin box persists every change, so QSettings holds its current
        // value without requiring a widget read.
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        broad.maxDistance = settings.value(vc3d::settings::atlas::SEARCH_MAX_DISTANCE,
                                           broad.maxDistance).toDouble();
    }
    const int searchMode = params.searchMode;
    const QStringList requiredTags = params.requiredTags;
    const QStringList excludedTags = params.excludedTags;
    if (searchMode != ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS &&
        searchMode != ATLAS_SEARCH_MODE_NON_ATLAS_ONLY) {
        return fail(tr("Unsupported atlas search type."));
    }

    if (!_currentAtlasDir && searchMode == ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS) {
        return fail(tr("Select an atlas, or choose \"Non-atlas fibers only\"."));
    }

    vc::atlas::Atlas atlas;
    bool haveAtlas = false;
    if (_currentAtlasDir) {
        try {
            atlas = vc::atlas::Atlas::load(*_currentAtlasDir);
            haveAtlas = true;
        } catch (const std::exception& ex) {
            return fail(tr("Could not load selected atlas: %1")
                            .arg(QString::fromStdString(ex.what())));
        }
    }

    const auto fiberSnapshots = _lineAnnotationController->fiberSnapshotsFromStorageWithPaths();
    if (fiberSnapshots.empty()) {
        updateAtlasSearchDocks();
        return fail(tr("No saved fibers are available to search."));
    }

    std::vector<std::string> atlasFiberPaths = haveAtlas
        ? vc::atlas::atlasMappedFiberPathKeys(atlas)
        : std::vector<std::string>{};
    if (searchMode == ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS && atlasFiberPaths.empty()) {
        return fail(tr("Selected atlas has no saved fiber mappings."));
    }

    const bool debugSearch = atlasSearchDebugEnabled();
    if (debugSearch) {
        qInfo().noquote() << QStringLiteral("[atlas-search] atlas_dir=%1 name=%2 mapping_count=%3")
            .arg(_currentAtlasDir
                     ? QString::fromStdString(_currentAtlasDir->string())
                     : QStringLiteral("<none>"),
                 QString::fromStdString(atlas.metadata.name))
            .arg(static_cast<int>(atlas.fibers.size()));
        for (const auto& mapping : atlas.fibers) {
            qInfo().noquote() << QStringLiteral("[atlas-search] atlas_mapping fiber_path=%1 key=%2 line_anchors=%3 control_anchors=%4")
                .arg(QString::fromStdString(mapping.fiberPath.generic_string()),
                     QString::fromStdString(vc::atlas::atlasFiberPathKey(mapping.fiberPath)))
                .arg(static_cast<int>(mapping.lineAnchors.size()))
                .arg(static_cast<int>(mapping.controlAnchors.size()));
        }
        qInfo().noquote() << QStringLiteral("[atlas-search] saved_fiber_count=%1")
            .arg(static_cast<int>(fiberSnapshots.size()));
    }

    std::vector<std::filesystem::path> canonicalPaths;
    canonicalPaths.reserve(fiberSnapshots.size());
    for (const auto& snapshot : fiberSnapshots) {
        canonicalPaths.push_back(snapshot.fiberPath);
    }
    const auto runtimeIds = vc::atlas::makeFiberRuntimeIdentityMap(canonicalPaths);
    const auto searchSets = vc::atlas::atlasFiberSearchSets(atlas, runtimeIds);
    std::vector<vc::atlas::FiberPolyline> fibers;
    std::vector<uint64_t> sourceFiberIds = searchMode == ATLAS_SEARCH_MODE_NON_ATLAS_ONLY
        ? searchSets.targetFiberIds
        : searchSets.sourceFiberIds;
    std::vector<uint64_t> targetFiberIds = searchSets.targetFiberIds;
    std::unordered_map<uint64_t, std::string> fiberPathById;
    std::unordered_map<uint64_t, AtlasSearchFiberSnapshot> snapshotsByRuntimeId;
    fibers.reserve(fiberSnapshots.size());
    snapshotsByRuntimeId.reserve(fiberSnapshots.size());
    _atlasSearchCancelRequested = false;
    qInfo().noquote() << QStringLiteral("[atlas-search] phase=1 prepare_inputs start total=%1")
        .arg(static_cast<qulonglong>(fiberSnapshots.size()));
    updateAtlasSearchProgress(vc::atlas::AtlasSearchProgressPhase::PrepareInputs,
                              0,
                              fiberSnapshots.size());
    std::size_t preparedFibers = 0;
    for (const auto& snapshot : fiberSnapshots) {
        const uint64_t fiberId = runtimeIds.idForPath(snapshot.fiberPath);
        auto fiber = snapshot.fiber;
        fiber.id = fiberId;
        fibers.push_back(std::move(fiber));
        auto previewFiber = snapshot.fiber;
        previewFiber.id = fiberId;
        snapshotsByRuntimeId.emplace(
            fiberId,
            AtlasSearchFiberSnapshot{
                snapshot.fiberPath,
                std::move(previewFiber),
                snapshot.storedFiberId,
                snapshot.hvClassification,
                snapshot.manualHvTag,
                snapshot.tags});
        const std::string pathKey = vc::atlas::atlasFiberPathKey(snapshot.fiberPath);
        const bool inAtlas = std::binary_search(atlasFiberPaths.begin(), atlasFiberPaths.end(), pathKey);
        fiberPathById.emplace(fiberId, pathKey);
        if (debugSearch) {
            qInfo().noquote() << QStringLiteral("[atlas-search] saved_fiber path=%1 key=%2 runtime_id=%3 generation=%4 points=%5 controls=%6 side=%7")
                .arg(QString::fromStdString(snapshot.fiberPath.generic_string()),
                     QString::fromStdString(pathKey),
                     QString::number(fiberId),
                     QString::number(snapshot.fiber.generation),
                     QString::number(static_cast<int>(snapshot.fiber.points.size())),
                     QString::number(static_cast<int>(snapshot.fiber.controlPoints.size())),
                     inAtlas ? QStringLiteral("source_atlas") : QStringLiteral("target_non_atlas"));
        }
        ++preparedFibers;
        updateAtlasSearchProgress(vc::atlas::AtlasSearchProgressPhase::PrepareInputs,
                                  preparedFibers,
                                  fiberSnapshots.size());
    }
    qInfo().noquote() << QStringLiteral("[atlas-search] phase=1 prepare_inputs end completed=%1 total=%2")
        .arg(static_cast<qulonglong>(preparedFibers))
        .arg(static_cast<qulonglong>(fiberSnapshots.size()));
    if (!requiredTags.isEmpty() || !excludedTags.isEmpty()) {
        auto keepTagged = [&snapshotsByRuntimeId, &requiredTags, &excludedTags](std::vector<uint64_t>& ids) {
            ids.erase(std::remove_if(ids.begin(),
                                     ids.end(),
                                     [&snapshotsByRuntimeId, &requiredTags, &excludedTags](uint64_t id) {
                                         const auto it = snapshotsByRuntimeId.find(id);
                                         return it == snapshotsByRuntimeId.end() ||
                                                !atlasSearchFiberMatchesTags(it->second.tags,
                                                                             requiredTags) ||
                                                atlasSearchFiberHasAnyTag(it->second.tags,
                                                                          excludedTags);
                                     }),
                      ids.end());
        };
        keepTagged(sourceFiberIds);
        keepTagged(targetFiberIds);
    }
    if (sourceFiberIds.empty()) {
        return fail(searchMode == ATLAS_SEARCH_MODE_NON_ATLAS_ONLY
                        ? tr("No saved non-atlas fibers match the search filters.")
                        : tr("None of the selected atlas fibers are available in saved fiber files or match the search filters."));
    }
    if (targetFiberIds.empty()) {
        return fail(tr("No saved non-atlas fibers are available to search or match the search filters."));
    }
    if (debugSearch) {
        qInfo().noquote() << QStringLiteral("[atlas-search] split source_ids=%1 target_ids=%2")
            .arg(atlasSearchIdListString(sourceFiberIds),
                 atlasSearchIdListString(targetFiberIds));
    }
    const std::size_t pairTotal = sourceFiberIds.size() * targetFiberIds.size();
    qInfo().noquote() << QStringLiteral("[atlas-search] start mode=%1 saved_fibers=%2 sources=%3 targets=%4 pairs=%5")
        .arg(atlasSearchModeName(searchMode))
        .arg(static_cast<qulonglong>(fiberSnapshots.size()))
        .arg(static_cast<qulonglong>(sourceFiberIds.size()))
        .arg(static_cast<qulonglong>(targetFiberIds.size()))
        .arg(static_cast<qulonglong>(pairTotal));

    vc::atlas::FiberIntersectionCeresOptions ceres;

    std::filesystem::path lasagnaManifestPath;
    double lasagnaWorkingToBaseScale = 1.0;
    try {
        if (const auto resolved = resolvedLasagnaForState(_state)) {
            lasagnaManifestPath = resolved->manifestPath;
            lasagnaWorkingToBaseScale = resolved->workingToBaseScale;
        }
    } catch (const std::exception& ex) {
        return fail(QString::fromStdString(ex.what()));
    }
    if (lasagnaManifestPath.empty()) {
        return fail(tr("Select a local Lasagna dataset before searching. "
                       "Intersection distance is measured in grad_mag winding-integral space."));
    }

    clearAtlasSearchPreviewState();
    auto snapshotsByRuntimeIdForWorker = snapshotsByRuntimeId;
    _atlasSearchFiberSnapshotsByRuntimeId = std::move(snapshotsByRuntimeId);
    _atlasSearchLasagnaManifestPath = lasagnaManifestPath;
    _atlasSearchLasagnaWorkingToBaseScale = lasagnaWorkingToBaseScale;
    _atlasSearchCancelRequested = false;
    _atlasSearchCancelFlag = std::make_shared<std::atomic_bool>(false);
    for (auto* dock : {_atlasSearchDock, _atlasWorkspaceSearchDock}) {
        if (!dock || !dock->widget()) {
            continue;
        }
        if (auto* run = dock->widget()->findChild<QPushButton*>(QStringLiteral("atlasSearchRunButton"))) {
            run->setEnabled(false);
        }
        if (auto* cancel = dock->widget()->findChild<QPushButton*>(QStringLiteral("atlasSearchCancelButton"))) {
            cancel->setEnabled(true);
        }
        if (auto* tree = dock->widget()->findChild<QTreeWidget*>(QStringLiteral("atlasSearchResultTree"))) {
            tree->clear();
        }
    }
    _atlasSearchResults.clear();
    _atlasSearchSignedWindings.clear();

    const auto searchStart = std::chrono::steady_clock::now();
    auto* watcher = new QFutureWatcher<AtlasSearchWorkerResult>(this);
    connect(watcher,
            &QFutureWatcher<AtlasSearchWorkerResult>::finished,
            this,
            [this,
             watcher,
             cancelFlag = _atlasSearchCancelFlag,
             searchStart]() {
                watcher->deleteLater();
                // result() can rethrow worker failures. Convert them to a failed
                // search so the exception cannot escape the event loop.
                AtlasSearchWorkerResult workerResult;
                bool workerFailed = false;
                QString workerError;
                try {
                    workerResult = watcher->result();
                } catch (const std::exception& ex) {
                    workerFailed = true;
                    workerError = QString::fromStdString(ex.what());
                } catch (...) {
                    workerFailed = true;
                    workerError = QStringLiteral("unknown error");
                }
                const bool canceled = cancelFlag && cancelFlag->load(std::memory_order_relaxed);
                const bool searchSucceeded = !workerFailed && !canceled && !_atlasSearchCancelRequested;
                if (searchSucceeded) {
                    populateAtlasSearchResults(workerResult.results, workerResult.signedWindings);
                    if (statusBar()) {
                        const QString message = workerResult.skippedSigningCount == 0
                            ? tr("Atlas object search found %1 result(s)")
                                  .arg(static_cast<int>(workerResult.results.size()))
                            : tr("Atlas object search found %1 signed result(s); skipped %2 result(s)")
                                  .arg(static_cast<int>(workerResult.results.size()))
                                  .arg(static_cast<int>(workerResult.skippedSigningCount));
                        showStatusBarMessage(message, 3000);
                    }
                } else {
                    qInfo().noquote() << QStringLiteral("[atlas-search] phase=%1 finishing_results start total=1")
                        .arg(atlasSearchPhaseNumber(vc::atlas::AtlasSearchProgressPhase::FinishResults));
                    updateAtlasSearchProgress(vc::atlas::AtlasSearchProgressPhase::FinishResults,
                                              1,
                                              1);
                    qInfo().noquote() << QStringLiteral("[atlas-search] phase=%1 finishing_results end completed=1 total=1")
                        .arg(atlasSearchPhaseNumber(vc::atlas::AtlasSearchProgressPhase::FinishResults));
                    for (auto* dock : {_atlasSearchDock, _atlasWorkspaceSearchDock}) {
                        if (!dock || !dock->widget()) {
                            continue;
                        }
                        if (auto* progress = dock->widget()->findChild<QProgressBar*>(
                                QStringLiteral("atlasSearchProgressBar"))) {
                            progress->setFormat(workerFailed ? tr("Failed") : tr("Canceled"));
                        }
                    }
                    if (statusBar()) {
                        showStatusBarMessage(workerFailed
                                                 ? tr("Atlas object search failed: %1").arg(workerError)
                                                 : tr("Atlas object search canceled"),
                                             3000);
                    }
                }
                const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - searchStart).count();
                qInfo().noquote() << QStringLiteral("[atlas-search] final raw_result_count=%1 signed_result_count=%2 skipped_signing=%3 canceled=%4 elapsed_ms=%5")
                    .arg(static_cast<qulonglong>(workerResult.rawResultCount))
                    .arg(static_cast<qulonglong>(workerResult.results.size()))
                    .arg(static_cast<qulonglong>(workerResult.skippedSigningCount))
                    .arg(canceled || _atlasSearchCancelRequested ? QStringLiteral("yes") : QStringLiteral("no"))
                    .arg(static_cast<qlonglong>(elapsedMs));
                _atlasSearchCancelRequested = false;
                if (_atlasSearchCancelFlag == cancelFlag) {
                    _atlasSearchCancelFlag.reset();
                }
                updateAtlasSearchDocks();
                // success=false covers cancellation and worker failure.
                emit atlasSearchFinished(searchSucceeded,
                                         static_cast<int>(_atlasSearchResults.size()));
            });

    vc::atlas::FiberIntersectionCache* cache = &_fiberIntersectionCache;
    QPointer<CWindow> self(this);
    auto cancelFlag = _atlasSearchCancelFlag;
    const auto atlasDirForWorker = _currentAtlasDir;
    watcher->setFuture(QtConcurrent::run([fibers = std::move(fibers),
                                          sourceFiberIds = std::move(sourceFiberIds),
                                          targetFiberIds = std::move(targetFiberIds),
                                          fiberPathById = std::move(fiberPathById),
                                          snapshotsByRuntimeId = std::move(snapshotsByRuntimeIdForWorker),
                                          lasagnaManifestPath = std::move(lasagnaManifestPath),
                                          lasagnaWorkingToBaseScale,
                                          atlasDir = atlasDirForWorker,
                                          cache,
                                          broad,
                                          ceres,
                                          self,
                                          cancelFlag,
                                          debugSearch]() mutable {
        vc::lasagna::LasagnaDataset dataset =
            vc::lasagna::LasagnaDataset::open(
                lasagnaManifestPath, {lasagnaWorkingToBaseScale});
        vc::lasagna::LasagnaNormalSampler windingSampler(dataset);
        bool phase2Started = false;
        bool phase2Ended = false;
        bool phase3Started = false;
        bool phase3Ended = false;
        bool phase4Started = false;
        bool phase4Ended = false;
        auto progressCallback = [self,
                                 cancelFlag,
                                 &phase2Started,
                                 &phase2Ended,
                                 &phase3Started,
                                 &phase3Ended,
                                 &phase4Started,
                                 &phase4Ended](vc::atlas::AtlasSearchProgressPhase phase,
                                               std::size_t completed,
                                               std::size_t total) {
            bool* started = nullptr;
            bool* ended = nullptr;
            if (phase == vc::atlas::AtlasSearchProgressPhase::BuildSpatialIndex) {
                started = &phase2Started;
                ended = &phase2Ended;
            } else if (phase == vc::atlas::AtlasSearchProgressPhase::SearchPairs) {
                started = &phase3Started;
                ended = &phase3Ended;
            } else if (phase == vc::atlas::AtlasSearchProgressPhase::PrepareSigningSurface) {
                started = &phase4Started;
                ended = &phase4Ended;
            }
            if (started && !*started && completed == 0) {
                *started = true;
                qInfo().noquote() << QStringLiteral("[atlas-search] phase=%1 %2 start total=%3")
                    .arg(atlasSearchPhaseNumber(phase))
                    .arg(atlasSearchPhaseAction(phase).toLower().replace(QChar(' '), QChar('_')))
                    .arg(static_cast<qulonglong>(total));
            }
            if (ended && !*ended && (completed >= total)) {
                *ended = true;
                qInfo().noquote() << QStringLiteral("[atlas-search] phase=%1 %2 end completed=%3 total=%4")
                    .arg(atlasSearchPhaseNumber(phase))
                    .arg(atlasSearchPhaseAction(phase).toLower().replace(QChar(' '), QChar('_')))
                    .arg(static_cast<qulonglong>(completed))
                    .arg(static_cast<qulonglong>(total));
            }
            if (!self) {
                return;
            }
            QMetaObject::invokeMethod(self.data(), [self, cancelFlag, phase, completed, total]() {
                if (!self || self->_atlasSearchCancelFlag != cancelFlag) {
                    return;
                }
                self->updateAtlasSearchProgress(phase, completed, total);
            }, Qt::QueuedConnection);
        };
        auto cancelCallback = [cancelFlag]() {
            return cancelFlag && cancelFlag->load(std::memory_order_relaxed);
        };
        auto rawResults = vc::atlas::searchFiberIntersections(fibers,
                                                              sourceFiberIds,
                                                              targetFiberIds,
                                                              cache,
                                                              broad,
                                                              ceres,
                                                              &windingSampler,
                                                              progressCallback,
                                                              cancelCallback);
        if (debugSearch) {
            qInfo().noquote() << QStringLiteral("[atlas-search] finished source_ids=%1 target_ids=%2 result_count=%3")
                .arg(atlasSearchIdListString(sourceFiberIds),
                     atlasSearchIdListString(targetFiberIds))
                .arg(static_cast<int>(rawResults.size()));
            for (const auto& result : rawResults) {
                const auto sourcePath = fiberPathById.find(result.sourceFiberId);
                const auto targetPath = fiberPathById.find(result.targetFiberId);
                qInfo().noquote() << QStringLiteral("[atlas-search] result source_id=%1 source_path=%2 target_id=%3 target_path=%4 winding=%5 candidate=%6 source_s=%7 target_s=%8")
                    .arg(QString::number(result.sourceFiberId),
                         QString::fromStdString(sourcePath == fiberPathById.end() ? std::string("<missing>") : sourcePath->second),
                         QString::number(result.targetFiberId),
                         QString::fromStdString(targetPath == fiberPathById.end() ? std::string("<missing>") : targetPath->second),
                         QString::number(result.windingDistance, 'g', 12),
                         QString::number(result.candidateDistance, 'g', 12),
                         QString::number(result.sourceArclength, 'g', 12),
                         QString::number(result.targetArclength, 'g', 12));
            }
        }
        if (cancelCallback()) {
            AtlasSearchWorkerResult canceledResult;
            canceledResult.rawResultCount = rawResults.size();
            return canceledResult;
        }
        return buildSignedAtlasSearchResults(std::move(rawResults),
                                             snapshotsByRuntimeId,
                                             atlasDir,
                                             lasagnaManifestPath,
                                             lasagnaWorkingToBaseScale,
                                             progressCallback,
                                             debugSearch);
    }));
    return true;
}

void CWindow::populateAtlasSearchResults(const std::vector<vc::atlas::FiberIntersectionResult>& results,
                                         std::vector<double> signedWindings)
{
    struct FiberDisplayInfo {
        QString label;
        double lengthVx = 0.0;
    };
    std::unordered_map<uint64_t, FiberDisplayInfo> fiberInfo;
    if (_lineAnnotationController) {
        for (const auto& fiber : _lineAnnotationController->fiberSummaries()) {
            QString label = QString::number(fiber.id);
            if (!fiber.name.empty()) {
                label = tr("%1 (%2)")
                    .arg(QString::fromStdString(fiber.name))
                    .arg(fiber.id);
            }
            fiberInfo[fiber.id] = FiberDisplayInfo{label, fiber.lengthVx};
        }
    }

    auto fiberLabel = [&fiberInfo](uint64_t fiberId) {
        const auto it = fiberInfo.find(fiberId);
        if (it != fiberInfo.end()) {
            return it->second.label;
        }
        return QString::number(fiberId);
    };
    auto arclengthFraction = [&fiberInfo](uint64_t fiberId, double arclength) {
        const auto it = fiberInfo.find(fiberId);
        if (it == fiberInfo.end() || !std::isfinite(it->second.lengthVx) ||
            it->second.lengthVx <= 0.0 || !std::isfinite(arclength)) {
            return 0.0;
        }
        return std::clamp(arclength / it->second.lengthVx, 0.0, 1.0);
    };

    if (signedWindings.empty() &&
        results.size() == _atlasSearchResults.size() &&
        _atlasSearchSignedWindings.size() == _atlasSearchResults.size()) {
        signedWindings = _atlasSearchSignedWindings;
    }
    if (signedWindings.size() != results.size()) {
        qWarning().noquote() << QStringLiteral("[atlas-search] render skipped: result_count=%1 signed_count=%2")
            .arg(static_cast<qulonglong>(results.size()))
            .arg(static_cast<qulonglong>(signedWindings.size()));
        return;
    }

    _atlasSearchResults = results;
    _atlasSearchSignedWindings = std::move(signedWindings);
    ++_atlasSearchPreviewGeneration;
    _atlasSearchHoveredResult.reset();
    _atlasSearchSelectedResults.clear();
    _atlasSearchPreviewRequestedResults.clear();

    const std::size_t finishTotal = std::max<std::size_t>(1, _atlasSearchResults.size());
    qInfo().noquote() << QStringLiteral("[atlas-search] phase=%1 finishing_results start total=%2")
        .arg(atlasSearchPhaseNumber(vc::atlas::AtlasSearchProgressPhase::FinishResults))
        .arg(static_cast<qulonglong>(finishTotal));
    updateAtlasSearchProgress(vc::atlas::AtlasSearchProgressPhase::FinishResults,
                              0,
                              finishTotal);
    bool countedResultRows = false;
    for (auto* dock : {_atlasSearchDock, _atlasWorkspaceSearchDock}) {
        if (!dock || !dock->widget()) {
            continue;
        }
        auto* tree = dock->widget()->findChild<QTreeWidget*>(QStringLiteral("atlasSearchResultTree"));
        if (!tree) {
            continue;
        }
        tree->clear();
        tree->setColumnCount(5);
        tree->setSelectionMode(QAbstractItemView::ExtendedSelection);
        tree->setSelectionBehavior(QAbstractItemView::SelectRows);
        tree->setMouseTracking(true);
        tree->setHeaderLabels({
            tr("Distance (windings)"),
            tr("Source fiber"),
            tr("Target fiber"),
            tr("Src idx"),
            tr("Tgt idx"),
        });
        const bool groupByFiber = [&]() {
            if (auto* check = dock->widget()->findChild<QCheckBox*>(
                    QStringLiteral("atlasSearchGroupByFiberCheck"))) {
                return check->isChecked();
            }
            return true;
        }();

        auto fillResultRow = [&](QTreeWidgetItem* row, int resultIndex) {
            const auto& result = _atlasSearchResults[static_cast<size_t>(resultIndex)];
            row->setData(0, ATLAS_SEARCH_RESULT_INDEX_ROLE, resultIndex);
            row->setText(0, QString::number(_atlasSearchSignedWindings[static_cast<size_t>(resultIndex)], 'f', 3));
            row->setTextAlignment(0, Qt::AlignRight | Qt::AlignVCenter);
            row->setText(1, fiberLabel(result.sourceFiberId));
            row->setText(2, fiberLabel(result.targetFiberId));
            row->setText(3, QString::number(arclengthFraction(result.sourceFiberId,
                                                              result.sourceArclength),
                                            'f',
                                            6));
            row->setTextAlignment(3, Qt::AlignRight | Qt::AlignVCenter);
            row->setText(4, QString::number(arclengthFraction(result.targetFiberId,
                                                              result.targetArclength),
                                            'f',
                                            6));
            row->setTextAlignment(4, Qt::AlignRight | Qt::AlignVCenter);
        };

        if (!groupByFiber) {
            for (int rowIndex = 0; rowIndex < static_cast<int>(_atlasSearchResults.size()); ++rowIndex) {
                auto* row = new QTreeWidgetItem(tree);
                fillResultRow(row, rowIndex);
                if (!countedResultRows) {
                    updateAtlasSearchProgress(vc::atlas::AtlasSearchProgressPhase::FinishResults,
                                              static_cast<std::size_t>(rowIndex) + 1,
                                              finishTotal);
                }
            }
            if (_atlasSearchResults.empty() && !countedResultRows) {
                updateAtlasSearchProgress(vc::atlas::AtlasSearchProgressPhase::FinishResults,
                                          1,
                                          finishTotal);
            }
            countedResultRows = true;
            tree->resizeColumnToContents(0);
            tree->resizeColumnToContents(1);
            tree->resizeColumnToContents(2);
            continue;
        }

        struct FiberGroup {
            uint64_t fiberId = 0;
            std::vector<int> resultIndices;
            std::set<uint64_t> partnerFiberIds;
            std::set<uint64_t> closePartnerFiberIds;
        };
        std::map<uint64_t, FiberGroup> groupsByFiber;
        for (int row = 0; row < static_cast<int>(_atlasSearchResults.size()); ++row) {
            const auto& result = _atlasSearchResults[static_cast<size_t>(row)];
            auto addToGroup = [&](uint64_t fiberId, uint64_t partnerFiberId) {
                auto& group = groupsByFiber[fiberId];
                group.fiberId = fiberId;
                group.resultIndices.push_back(row);
                group.partnerFiberIds.insert(partnerFiberId);
                if (std::abs(result.windingDistance) < ATLAS_SEARCH_CLOSE_WINDING_THRESHOLD) {
                    group.closePartnerFiberIds.insert(partnerFiberId);
                }
            };
            addToGroup(result.sourceFiberId, result.targetFiberId);
            addToGroup(result.targetFiberId, result.sourceFiberId);
            if (!countedResultRows) {
                updateAtlasSearchProgress(vc::atlas::AtlasSearchProgressPhase::FinishResults,
                                          static_cast<std::size_t>(row) + 1,
                                          finishTotal);
            }
        }

        std::vector<FiberGroup> groups;
        groups.reserve(groupsByFiber.size());
        for (auto& [fiberId, group] : groupsByFiber) {
            (void)fiberId;
            std::sort(group.resultIndices.begin(),
                      group.resultIndices.end(),
                      [&](int a, int b) {
                          const auto& ra = _atlasSearchResults[static_cast<size_t>(a)];
                          const auto& rb = _atlasSearchResults[static_cast<size_t>(b)];
                          const double da = _atlasSearchSignedWindings[static_cast<size_t>(a)];
                          const double db = _atlasSearchSignedWindings[static_cast<size_t>(b)];
                          if (da != db) return da < db;
                          const double rawA = std::abs(ra.windingDistance);
                          const double rawB = std::abs(rb.windingDistance);
                          if (rawA != rawB) return rawA < rawB;
                          if (ra.sourceFiberId != rb.sourceFiberId) {
                              return ra.sourceFiberId < rb.sourceFiberId;
                          }
                          return ra.targetFiberId < rb.targetFiberId;
                      });
            groups.push_back(std::move(group));
        }
        std::sort(groups.begin(), groups.end(), [&](const FiberGroup& a, const FiberGroup& b) {
            if (a.closePartnerFiberIds.size() != b.closePartnerFiberIds.size()) {
                return a.closePartnerFiberIds.size() > b.closePartnerFiberIds.size();
            }
            if (a.partnerFiberIds.size() != b.partnerFiberIds.size()) {
                return a.partnerFiberIds.size() > b.partnerFiberIds.size();
            }
            return fiberLabel(a.fiberId) < fiberLabel(b.fiberId);
        });

        for (const FiberGroup& group : groups) {
            auto* root = new QTreeWidgetItem(tree);
            root->setText(
                0,
                tr("%1 - %2 pairs below 0.5 windings, %3 total pairs, %4 intersections")
                    .arg(fiberLabel(group.fiberId))
                    .arg(static_cast<int>(group.closePartnerFiberIds.size()))
                    .arg(static_cast<int>(group.partnerFiberIds.size()))
                    .arg(static_cast<int>(group.resultIndices.size())));
            root->setData(0,
                          ATLAS_SEARCH_FIBER_ID_ROLE,
                          QVariant::fromValue<qulonglong>(group.fiberId));
            root->setFirstColumnSpanned(true);
            root->setFlags(root->flags() & ~Qt::ItemIsSelectable);

            for (int resultIndex : group.resultIndices) {
                auto* row = new QTreeWidgetItem(root);
                fillResultRow(row, resultIndex);
            }
        }
        if (_atlasSearchResults.empty() && !countedResultRows) {
            updateAtlasSearchProgress(vc::atlas::AtlasSearchProgressPhase::FinishResults,
                                      1,
                                      finishTotal);
        }
        countedResultRows = true;
        tree->expandAll();
        tree->resizeColumnToContents(0);
        tree->resizeColumnToContents(1);
        tree->resizeColumnToContents(2);
    }
    updateAtlasSearchProgress(vc::atlas::AtlasSearchProgressPhase::FinishResults,
                              finishTotal,
                              finishTotal);
    qInfo().noquote() << QStringLiteral("[atlas-search] phase=%1 finishing_results end completed=%2 total=%3")
        .arg(atlasSearchPhaseNumber(vc::atlas::AtlasSearchProgressPhase::FinishResults))
        .arg(static_cast<qulonglong>(finishTotal))
        .arg(static_cast<qulonglong>(finishTotal));
    for (auto* dock : {_atlasSearchDock, _atlasWorkspaceSearchDock}) {
        if (!dock || !dock->widget()) {
            continue;
        }
        if (auto* progress = dock->widget()->findChild<QProgressBar*>(QStringLiteral("atlasSearchProgressBar"))) {
            progress->setFormat(tr("Done"));
        }
    }
    updateAtlasSearchPreviewCandidates();
}

void CWindow::openAtlasSearchResult(int sortedResultIndex)
{
    if (sortedResultIndex < 0 ||
        sortedResultIndex >= static_cast<int>(_atlasSearchResults.size())) {
        return;
    }
    if (!_lineAnnotationController || !_intersectionsMdiArea) {
        QMessageBox::warning(this,
                             tr("Intersections"),
                             tr("Intersections workspace is not available."));
        return;
    }
    if (!_currentAtlasDir) {
        QMessageBox::information(this,
                                 tr("Intersections"),
                                 tr("Create or select an atlas before inspecting intersections."));
        return;
    }
    if (_workspaceTabs && _intersectionsWorkspaceWindow) {
        _workspaceTabs->setCurrentWidget(_intersectionsWorkspaceWindow);
    }
    _lineAnnotationController->showIntersectionInspection(
        _atlasSearchResults[static_cast<size_t>(sortedResultIndex)],
        _intersectionsMdiArea,
        _currentAtlasDir);
}

void CWindow::clearAtlasSearchPreviewState()
{
    ++_atlasSearchPreviewGeneration;
    _atlasSearchResults.clear();
    _atlasSearchSignedWindings.clear();
    _atlasSearchFiberSnapshotsByRuntimeId.clear();
    _atlasSearchLasagnaManifestPath.reset();
    _atlasSearchLasagnaWorkingToBaseScale = 1.0;
    _atlasSearchHoveredResult.reset();
    _atlasSearchSelectedResults.clear();
    _atlasSearchPreviewRequestedResults.clear();
    if (_atlasOverlay) {
        _atlasOverlay->clearSearchPreviews();
    }
}

void CWindow::updateAtlasSearchPreviewCandidates()
{
    if (!_atlasOverlay || !_currentAtlasDir || _atlasSearchResults.empty()) {
        if (_atlasOverlay) {
            _atlasOverlay->setSearchPreviewCandidates({});
        }
        return;
    }

    try {
        vc::atlas::Atlas atlas = vc::atlas::Atlas::load(*_currentAtlasDir);
        const std::filesystem::path basePath = *_currentAtlasDir / atlas.metadata.baseMeshPath;
        QuadSurface baseSurface(basePath);
        const auto* points = baseSurface.rawPointsPtr();
        if (!points || points->empty() || points->cols <= 0) {
            throw std::runtime_error("atlas base mesh has no valid grid");
        }
        const int periodColumns = vc::atlas::atlasHorizontalPeriodColumns(baseSurface);
        (void)vc::atlas::layoutAtlasObjects(atlas, periodColumns);
        const vc::atlas::AtlasDisplayRange displayRange =
            vc::atlas::atlasDisplayRange(atlas, periodColumns);
        const auto displaySurface =
            vc::atlas::repeatedAtlasDisplaySurface(baseSurface,
                                                   displayRange.unwrapCount,
                                                   atlas.metadata.zeroWindingColumn);

        std::vector<AtlasOverlayController::SearchPreviewCandidate> candidates;
        candidates.reserve(_atlasSearchResults.size());
        for (int row = 0; row < static_cast<int>(_atlasSearchResults.size()); ++row) {
            const auto coord = atlasSearchCandidateSurfaceCoord(
                atlas,
                *displaySurface,
                displayRange,
                _atlasSearchResults[static_cast<size_t>(row)],
                _atlasSearchFiberSnapshotsByRuntimeId);
            if (!coord) {
                continue;
            }
            candidates.push_back({row, *coord});
        }
        _atlasOverlay->setSearchPreviewCandidates(std::move(candidates));
    } catch (const std::exception& ex) {
        qWarning().noquote() << QStringLiteral("Could not build atlas search preview crosses: %1")
            .arg(QString::fromStdString(ex.what()));
        _atlasOverlay->setSearchPreviewCandidates({});
    }
}

void CWindow::setAtlasSearchHoverResult(std::optional<int> sortedResultIndex)
{
    if (sortedResultIndex &&
        (*sortedResultIndex < 0 ||
         *sortedResultIndex >= static_cast<int>(_atlasSearchResults.size()))) {
        sortedResultIndex.reset();
    }
    if (_atlasSearchHoveredResult == sortedResultIndex) {
        return;
    }
    _atlasSearchHoveredResult = sortedResultIndex;
    if (_atlasOverlay) {
        _atlasOverlay->setSearchPreviewHover(_atlasSearchHoveredResult);
    }
    updateAtlasSearchPreviewRequests();
}

void CWindow::updateAtlasSearchSelectionFromTree(QTreeWidget* sourceTree)
{
    std::set<int> selected;
    if (sourceTree) {
        for (const auto* item : sourceTree->selectedItems()) {
            const auto resultIndex = atlasSearchResultIndexForItem(item);
            if (!resultIndex) {
                continue;
            }
            if (*resultIndex >= 0 &&
                *resultIndex < static_cast<int>(_atlasSearchResults.size())) {
                selected.insert(*resultIndex);
            }
        }
    }

    if (_atlasSearchSelectedResults == selected) {
        return;
    }
    _atlasSearchSelectedResults = std::move(selected);
    syncAtlasSearchTreeSelection(nullptr);
    if (_atlasOverlay) {
        _atlasOverlay->setSearchPreviewSelection(_atlasSearchSelectedResults);
    }
    updateAtlasSearchPreviewRequests();
}

void CWindow::syncAtlasSearchTreeSelection(QTreeWidget* sourceTree)
{
    for (auto* dock : {_atlasSearchDock, _atlasWorkspaceSearchDock}) {
        if (!dock || !dock->widget()) {
            continue;
        }
        auto* tree = dock->widget()->findChild<QTreeWidget*>(QStringLiteral("atlasSearchResultTree"));
        if (!tree || tree == sourceTree) {
            continue;
        }
        QSignalBlocker blocker(tree);
        tree->setProperty("vc_atlas_syncing_selection", true);
        tree->clearSelection();
        if (auto* selection = tree->selectionModel()) {
            forEachAtlasSearchResultItem(tree, [&](QTreeWidgetItem* item) {
                const auto resultIndex = atlasSearchResultIndexForItem(item);
                if (!resultIndex ||
                    _atlasSearchSelectedResults.find(*resultIndex) ==
                        _atlasSearchSelectedResults.end()) {
                    return;
                }
                const QModelIndex left = tree->indexFromItem(item, 0);
                const QModelIndex right = tree->indexFromItem(item, tree->columnCount() - 1);
                selection->select(QItemSelection(left, right),
                                  QItemSelectionModel::Select | QItemSelectionModel::Rows);
            });
        }
        tree->setProperty("vc_atlas_syncing_selection", false);
    }
}

void CWindow::updateAtlasSearchPreviewRequests()
{
    std::set<int> wanted = _atlasSearchSelectedResults;
    if (_atlasSearchHoveredResult) {
        wanted.insert(*_atlasSearchHoveredResult);
    }
    for (const int resultIndex : wanted) {
        requestAtlasSearchPreviewLine(resultIndex);
    }
}

void CWindow::requestAtlasSearchPreviewLine(int sortedResultIndex)
{
    if (sortedResultIndex < 0 ||
        sortedResultIndex >= static_cast<int>(_atlasSearchResults.size()) ||
        !_currentAtlasDir ||
        !_atlasSearchLasagnaManifestPath ||
        _atlasSearchPreviewRequestedResults.find(sortedResultIndex) !=
            _atlasSearchPreviewRequestedResults.end()) {
        return;
    }

    const auto result = _atlasSearchResults[static_cast<size_t>(sortedResultIndex)];
    const auto sourceIt = _atlasSearchFiberSnapshotsByRuntimeId.find(result.sourceFiberId);
    const auto targetIt = _atlasSearchFiberSnapshotsByRuntimeId.find(result.targetFiberId);
    if (sourceIt == _atlasSearchFiberSnapshotsByRuntimeId.end() ||
        targetIt == _atlasSearchFiberSnapshotsByRuntimeId.end()) {
        return;
    }

    _atlasSearchPreviewRequestedResults.insert(sortedResultIndex);
    const int generation = _atlasSearchPreviewGeneration;
    auto* watcher =
        new QFutureWatcher<std::optional<AtlasOverlayController::SearchPreviewFiber>>(this);
    connect(watcher,
            &QFutureWatcher<std::optional<AtlasOverlayController::SearchPreviewFiber>>::finished,
            this,
            [this, watcher, generation, sortedResultIndex]() {
                const auto preview = watcher->result();
                watcher->deleteLater();
                if (generation != _atlasSearchPreviewGeneration || !preview || !_atlasOverlay) {
                    return;
                }
                _atlasOverlay->setSearchPreviewFiber(*preview);
            });

    watcher->setFuture(QtConcurrent::run([
        atlasDir = *_currentAtlasDir,
        manifestPath = *_atlasSearchLasagnaManifestPath,
        workingToBaseScale = _atlasSearchLasagnaWorkingToBaseScale,
        result,
        sourceSnapshot = sourceIt->second,
        targetSnapshot = targetIt->second,
        sortedResultIndex
    ]() -> std::optional<AtlasOverlayController::SearchPreviewFiber> {
        try {
            vc::atlas::Atlas atlas = vc::atlas::Atlas::load(atlasDir);
            const std::filesystem::path basePath = atlasDir / atlas.metadata.baseMeshPath;
            auto baseSurface = std::make_shared<QuadSurface>(basePath);
            const auto* points = baseSurface->rawPointsPtr();
            if (!points || points->empty() || points->cols <= 0) {
                throw std::runtime_error("atlas base mesh has no valid grid");
            }

            vc::lasagna::LasagnaDataset dataset =
                vc::lasagna::LasagnaDataset::open(
                    manifestPath, {workingToBaseScale});
            vc::lasagna::LasagnaNormalSampler sampler(dataset);
            SurfacePatchIndex baseIndex;
            baseIndex.rebuild({baseSurface});
            const int periodColumns = vc::atlas::atlasHorizontalPeriodColumns(*baseSurface);
            (void)vc::atlas::layoutAtlasObjects(atlas, periodColumns);

            const vc::atlas::FiberInput sourceInput =
                fiberInputFromSnapshot(sourceSnapshot.fiberPath, sourceSnapshot.fiber);
            const vc::atlas::FiberInput targetInput =
                fiberInputFromSnapshot(targetSnapshot.fiberPath, targetSnapshot.fiber);

            if (!findAtlasMappingForPath(atlas, sourceInput.fiberPath)) {
                atlas.fibers.push_back(vc::atlas::mapFiberToBaseSurface(
                    sourceInput, *baseSurface, baseIndex, sampler));
            }
            if (!findAtlasMappingForPath(atlas, targetInput.fiberPath)) {
                atlas.fibers.push_back(vc::atlas::mapFiberToBaseSurface(
                    targetInput, *baseSurface, baseIndex, sampler));
            }

            const auto* sourceMapping = findAtlasMappingForPath(atlas, sourceInput.fiberPath);
            auto* targetMapping = findAtlasMappingForPath(atlas, targetInput.fiberPath);
            if (!sourceMapping || !targetMapping) {
                throw std::runtime_error("Could not map both preview fibers into the atlas");
            }

            const auto sourceSample = vc3d::fiber_slice::samplePolylineAtArclength(
                sourceInput.linePoints,
                result.sourceArclength);
            const auto targetSample = vc3d::fiber_slice::samplePolylineAtArclength(
                targetInput.linePoints,
                result.targetArclength);
            if (!sourceSample.valid || !targetSample.valid) {
                throw std::runtime_error("Could not sample preview intersection arclengths");
            }

            auto endpointFor = [](const vc::atlas::FiberMapping& mapping,
                                  double arclength,
                                  double linePosition) {
                const auto* best = nearestLineAnchorForPosition(mapping, linePosition);
                if (!best) {
                    throw std::runtime_error("Mapped preview fiber has no line anchor");
                }
                vc::atlas::AtlasLinkEndpoint endpoint;
                endpoint.fiberPath = mapping.fiberPath;
                endpoint.sourceIndex = best->sourceIndex;
                endpoint.arclength = arclength;
                endpoint.atlasU = best->atlasU;
                endpoint.atlasV = best->atlasV;
                return endpoint;
            };

            vc::atlas::AtlasLink link;
            link.first = endpointFor(*sourceMapping,
                                     result.sourceArclength,
                                     sourceSample.linePosition);
            link.second = endpointFor(*targetMapping,
                                      result.targetArclength,
                                      targetSample.linePosition);
            link.desiredWindingDelta = 0;
            const int delta = vc::atlas::atlasLinkWindingOffsetDelta(
                link, periodColumns, atlas.metadata.zeroWindingColumn);
            targetMapping->windingOffset = sourceMapping->windingOffset + delta;
            return AtlasOverlayController::SearchPreviewFiber{
                sortedResultIndex,
                *targetMapping,
            };
        } catch (const std::exception& ex) {
            qWarning().noquote() << QStringLiteral("Could not build atlas search preview line: %1")
                .arg(QString::fromStdString(ex.what()));
            return std::nullopt;
        }
    }));
}

// Dialog-free atlas loading shared by the interactive and direct entry points.
void CWindow::loadAndDisplayAtlas(const std::filesystem::path& atlasDir)
{
    if (!_atlasWorkspaceWindow || !_atlasViewer) {
        createAtlasWorkspace();
    }
    auto vpkg = _state ? _state->vpkg() : nullptr;
    if (!vpkg) {
        throw std::runtime_error("No volume package is loaded");
    }
    const auto resolvedLasagna = resolvedLasagnaForState(_state);
    if (!resolvedLasagna || resolvedLasagna->manifestPath.empty()) {
        throw std::runtime_error(
            "No Lasagna dataset matches the active volume; atlas pred-snap attachments are required");
    }
    const std::filesystem::path manifestPath = resolvedLasagna->manifestPath;
    if (!std::filesystem::exists(manifestPath)) {
        throw std::runtime_error("Selected Lasagna dataset does not exist");
    }
    const std::filesystem::path volpkgRoot = vpkg->path().empty()
        ? std::filesystem::path(vpkg->getVolpkgDirectory())
        : vpkg->path().parent_path();
    auto atlas = vc::atlas::Atlas::load(atlasDir, volpkgRoot);
    vc::lasagna::LasagnaDataset dataset =
        vc::lasagna::LasagnaDataset::open(
            manifestPath,
            vc::lasagna::LasagnaDatasetOpenOptions{
                resolvedLasagna->workingToBaseScale});
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
    (void)vc::atlas::ensureAtlasPredSnapAttachments(atlasDir, volpkgRoot, sampler);
    atlas = vc::atlas::Atlas::load(atlasDir, volpkgRoot);
    _currentAtlasDir = atlasDir;
    _currentAtlasName = atlas.metadata.name;
    if (_lineAnnotationController) {
        _lineAnnotationController->setCurrentAtlasDirectory(_currentAtlasDir);
    }
    if (_segmentationWidget && _segmentationWidget->lasagnaPanel()) {
        _segmentationWidget->lasagnaPanel()->setSelectedAtlasPath(
            QString::fromStdString(atlasDir.string()));
    }
    const std::filesystem::path basePath = atlasDir / atlas.metadata.baseMeshPath;
    auto baseSurface = std::make_shared<QuadSurface>(basePath);
    const auto* points = baseSurface->rawPointsPtr();
    if (!points || points->empty() || points->cols <= 0) {
        throw std::runtime_error("atlas base mesh has no valid grid");
    }
    const cv::Vec2f baseScale = baseSurface->scale();
    if (!std::isfinite(baseScale[0]) || !std::isfinite(baseScale[1]) ||
        baseScale[0] <= 0.0f || baseScale[1] <= 0.0f) {
        throw std::runtime_error("atlas base mesh has invalid scale");
    }

    const int periodColumns = vc::atlas::atlasHorizontalPeriodColumns(*baseSurface);
    (void)vc::atlas::layoutAtlasObjects(atlas, periodColumns);
    const vc::atlas::AtlasDisplayRange displayRange =
        vc::atlas::atlasDisplayRange(atlas, periodColumns);
    std::shared_ptr<QuadSurface> displaySurface =
        vc::atlas::repeatedAtlasDisplaySurface(*baseSurface,
                                               displayRange.unwrapCount,
                                               atlas.metadata.zeroWindingColumn);
    displaySurface->id = ATLAS_INTERNAL_SURFACE_NAME;
    if (_state) {
        _state->setSurface(ATLAS_INTERNAL_SURFACE_NAME, displaySurface);
    }

    if (!_atlasOverlay) {
        _atlasOverlay = std::make_unique<AtlasOverlayController>(this);
    }
    if (_atlasViewer) {
        _atlasOverlay->attachViewer(_atlasViewer);
    }
    _atlasOverlay->setAtlas(atlas, displaySurface, displayRange);
    clearAtlasSearchPreviewState();
    for (auto* dock : {_atlasSearchDock, _atlasWorkspaceSearchDock}) {
        if (!dock || !dock->widget()) {
            continue;
        }
        if (auto* tree = dock->widget()->findChild<QTreeWidget*>(
                QStringLiteral("atlasSearchResultTree"))) {
            tree->clear();
        }
        if (auto* progress = dock->widget()->findChild<QProgressBar*>(
                QStringLiteral("atlasSearchProgressBar"))) {
            progress->setRange(0, 100);
            progress->setValue(0);
            progress->setFormat(QString());
        }
    }

    if (_atlasViewer) {
        if (const auto bounds = _atlasOverlay->surfaceBounds()) {
            _atlasViewer->centerOnSurfacePoint({
                static_cast<float>(bounds->center().x()),
                static_cast<float>(bounds->center().y()),
            }, false);
        } else {
            _atlasViewer->fitSurfaceInView();
        }
    }
    refreshAtlasOverviewDocks();
    updateAtlasFiberDocks();
    updateAtlasSearchDocks();
    if (_workspaceTabs && _atlasWorkspaceWindow) {
        _workspaceTabs->setCurrentWidget(_atlasWorkspaceWindow);
    }
    if (statusBar()) {
        showStatusBarMessage(tr("Displayed atlas %1")
                                 .arg(QString::fromStdString(atlas.metadata.name)),
                             3000);
    }
}

// Opens an atlas without dialogs or the interactive rebuild prompt.
bool CWindow::displayAtlasFromDirectoryHeadless(const std::filesystem::path& atlasDir,
                                                QString* errorMessage)
{
    try {
        loadAndDisplayAtlas(atlasDir);
        return true;
    } catch (const std::exception& ex) {
        if (errorMessage) {
            *errorMessage = QString::fromStdString(ex.what());
        }
        return false;
    }
}

void CWindow::displayAtlasFromDirectory(const std::filesystem::path& atlasDir)
{
    try {
        loadAndDisplayAtlas(atlasDir);
    } catch (const std::exception& ex) {
        if (vc::atlas::atlasLoadErrorRequiresRebuild(ex)) {
            const auto choice = QMessageBox::question(
                this,
                tr("Atlas Rebuild Required"),
                tr("This atlas was saved with an older or stale mapping format and must be rebuilt "
                   "from its unchanged source fiber JSON using the selected Lasagna dataset.\n\n"
                   "Rebuild now?"),
                QMessageBox::Yes | QMessageBox::No,
                QMessageBox::No);
            if (choice != QMessageBox::Yes) {
                return;
            }
            try {
                auto vpkg = _state ? _state->vpkg() : nullptr;
                if (!vpkg) {
                    throw std::runtime_error("No volume package is loaded");
                }
                const auto resolvedLasagna = resolvedLasagnaForState(_state);
                if (!resolvedLasagna || resolvedLasagna->manifestPath.empty())
                    throw std::runtime_error("No Lasagna dataset matches the active volume");
                const std::filesystem::path manifestPath = resolvedLasagna->manifestPath;
                if (!std::filesystem::exists(manifestPath)) {
                    throw std::runtime_error("Selected Lasagna dataset does not exist");
                }
                const std::filesystem::path volpkgRoot = vpkg->path().empty()
                    ? std::filesystem::path{}
                    : vpkg->path().parent_path();
                vc::lasagna::LasagnaDataset dataset =
                    vc::lasagna::LasagnaDataset::open(
                        manifestPath,
                        vc::lasagna::LasagnaDatasetOpenOptions{
                            resolvedLasagna->workingToBaseScale});
                vc::lasagna::LasagnaNormalSampler sampler(dataset);
                if (!sampler.hasPredDtChannel()) {
                    throw std::runtime_error(
                        "Selected Lasagna dataset has no pred_dt channel; atlas pred-snap attachments are required");
                }
                vc::atlas::rebuildAtlasFromSourceFibers(atlasDir, volpkgRoot, sampler);
                displayAtlasFromDirectory(atlasDir);
            } catch (const std::exception& rebuildEx) {
                QMessageBox::warning(
                    this,
                    tr("Atlas Rebuild"),
                    tr("Could not rebuild atlas: %1")
                        .arg(QString::fromStdString(rebuildEx.what())));
            }
            return;
        }
        QMessageBox::warning(this,
                             tr("Atlas"),
                             tr("Could not display atlas: %1").arg(QString::fromStdString(ex.what())));
    }
}

// Create widgets
void CWindow::CreateWidgets(void)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    // add volume viewer
    auto aWidgetLayout = new QVBoxLayout;
    aWidgetLayout->setContentsMargins(0, 0, 0, 0);
    aWidgetLayout->setSpacing(0);
    ui.tabSegment->setLayout(aWidgetLayout);

    mdiArea = nullptr;

    auto* viewerGrid = new ViewerSplitGrid(ui.tabSegment);
    viewerGrid->setObjectName(QStringLiteral("mainViewerSplitGrid"));
    aWidgetLayout->addWidget(viewerGrid, 1);

    {
        resetSegmentationViews(false);
        applyMainViewerLayout(viewerGrid, _viewerManager.get());
        viewerGrid->setSplits(settings.value(MAIN_VIEWER_SPLIT_X_SETTING, 0.5).toDouble(),
                              settings.value(MAIN_VIEWER_SPLIT_Y_SETTING, 0.5).toDouble());
    }

    viewerGrid->onSplitChanged = [viewerGrid]() {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue(MAIN_VIEWER_SPLIT_X_SETTING, viewerGrid->splitX());
        settings.setValue(MAIN_VIEWER_SPLIT_Y_SETTING, viewerGrid->splitY());
    };

    if (_fiberSliceWorkspaceWindow) {
        _fiberSliceMdiArea = new QMdiArea(_fiberSliceWorkspaceWindow);
        _fiberSliceMdiArea->setObjectName(QStringLiteral("fiberSliceMdiArea"));
        _fiberSliceWorkspaceWindow->setCentralWidget(_fiberSliceMdiArea);
        connect(_fiberSliceMdiArea, &QMdiArea::subWindowActivated, [](QMdiSubWindow* subWindow) {
            if (subWindow) {
                if (auto* viewer = dynamic_cast<VolumeViewerBase*>(subWindow->widget())) {
                    if (auto* graphicsView = viewer->graphicsView()) {
                        graphicsView->setFocus();
                    }
                }
            }
        });
    }

    if (_intersectionsWorkspaceWindow) {
        _intersectionsMdiArea = new QMdiArea(_intersectionsWorkspaceWindow);
        _intersectionsMdiArea->setObjectName(QStringLiteral("intersectionsMdiArea"));
        _intersectionsWorkspaceWindow->setCentralWidget(_intersectionsMdiArea);
        connect(_intersectionsMdiArea, &QMdiArea::subWindowActivated, [](QMdiSubWindow* subWindow) {
            if (subWindow) {
                if (auto* viewer = dynamic_cast<VolumeViewerBase*>(subWindow->widget())) {
                    if (auto* graphicsView = viewer->graphicsView()) {
                        graphicsView->setFocus();
                    }
                }
            }
        });

    }

    createAtlasWorkspace();

    treeWidgetSurfaces = ui.treeWidgetSurfaces;
    treeWidgetSurfaces->setSelectionMode(QAbstractItemView::ExtendedSelection);
    btnReloadSurfaces = ui.btnReloadSurfaces;

    SurfacePanelController::UiRefs surfaceUi{
        .treeWidget = treeWidgetSurfaces,
        .reloadButton = btnReloadSurfaces,
    };
    _surfacePanel = std::make_unique<SurfacePanelController>(
        surfaceUi,
        _state,
        _viewerManager.get(),
        [this]() { return segmentationViewer(); },
        std::function<void()>{},
        this);
    if (_segmentationGrower) {
        _segmentationGrower->setSurfacePanel(_surfacePanel.get());
    }
    if (_lineAnnotationController) {
        _lineAnnotationController->setSurfacePanel(_surfacePanel.get());
    }
    connect(_surfacePanel.get(), &SurfacePanelController::surfacesLoaded, this, [this]() {
        emit _state->surfacesLoaded();
        // Update surface overlay dropdown when surfaces are loaded
        updateSurfaceOverlayDropdown();
        if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }
    });
    connect(_surfacePanel.get(), &SurfacePanelController::surfaceSelectionCleared, this, [this]() {
        clearSurfaceSelection();
    });
    connect(_surfacePanel.get(), &SurfacePanelController::filtersApplied, this, [this](int filterCount) {
        UpdateVolpkgLabel(filterCount);
    });
    connect(_surfacePanel.get(), &SurfacePanelController::copySegmentPathRequested,
            this, [this](const QString& segmentId) {
                if (!_state->vpkg()) {
                    return;
                }
                auto surf = std::dynamic_pointer_cast<QuadSurface>(_state->surface(segmentId.toStdString()));
                if (!surf) {
                    surf = _state->vpkg()->getSurface(segmentId.toStdString());
                }
                if (!surf) {
                    return;
                }
                const QString path = absoluteSegmentPathForClipboard(surf->path, _state->vpkg());
                QApplication::clipboard()->setText(path);
                showStatusBarMessage(tr("Copied segment path to clipboard: %1").arg(path), 3000);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::addSurfaceToSpiralFitRequested,
            this, [this](const QString& segmentId) {
                if (!_spiralWorkspace) {
                    return;
                }
                auto surf = std::dynamic_pointer_cast<QuadSurface>(_state->surface(segmentId.toStdString()));
                if (!surf && _state->vpkg()) {
                    surf = _state->vpkg()->getSurface(segmentId.toStdString());
                }
                if (!surf || surf->path.empty()) {
                    showStatusBarMessage(tr("Cannot resolve an on-disk TIFXYZ directory for %1").arg(segmentId), 5000);
                    return;
                }
                _spiralWorkspace->addPatchToCurrentFit(
                    QString::fromStdString(surf->path.string()), surf);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::renderSegmentRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onRenderSegment(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::growSegmentRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onGrowSegmentFromSegment(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::neighborCopyRequested,
            this, [this](const QString& segmentId, bool copyOut) {
                _segmentationCommandHandler->onNeighborCopyRequested(segmentId, copyOut);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::resumeLocalGrowPatchRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onResumeLocalGrowPatchRequested(segmentId);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::reloadFromBackupRequested,
            this, [this](const QString& segmentId, int backupIndex) {
                _segmentationCommandHandler->onReloadFromBackup(segmentId, backupIndex);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::convertToObjRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onConvertToObj(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::mergeTifxyzRequested,
            this, [this](const QStringList& segmentIds) {
                _segmentationCommandHandler->onMergeTifxyz(segmentIds);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::mergePatchRequested,
            this, [this](const QStringList& segmentIds) {
                _segmentationCommandHandler->onMergePatch(segmentIds);
            });
    // Note: the Actions -> Merge tifxyz... menu wiring lives in the
    // constructor after _menuController is initialized -- _menuController
    // is null inside CreateWidgets().
    connect(_surfacePanel.get(), &SurfacePanelController::visLasagnaObjRequested,
            this, [this](const QString& segmentId) {
                onVisLasagnaObj(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::cropBoundsRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onCropSurfaceToValidRegion(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::flipURequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onFlipSurface(segmentId.toStdString(), true);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::flipVRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onFlipSurface(segmentId.toStdString(), false);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::rotateSurfaceRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onRotateSurface(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::focusSurfaceRequested,
            this, [this](const QString& segmentId) {
                if (!_state || !_state->vpkg()) return;
                auto surf = std::dynamic_pointer_cast<QuadSurface>(_state->surface(segmentId.toStdString()));
                if (!surf) {
                    surf = _state->vpkg()->getSurface(segmentId.toStdString());
                }
                auto* quad = dynamic_cast<QuadSurface*>(surf.get());
                if (!quad) return;
                const auto focusPoint = findSegmentFocusPoint(*quad);
                if (!focusPoint) return;
                cv::Vec3f normal = quad->normal(focusPoint->ptr, {0, 0, 0});
                if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) || !std::isfinite(normal[2])) {
                    normal = cv::Vec3f(0, 0, 1);
                }
                if (auto vol = _state->currentVolume()) {
                    auto [w, h, d] = vol->shapeXyz();
                    cv::Vec3f clamped = focusPoint->world;
                    clamped[0] = std::clamp(clamped[0], 0.0f, static_cast<float>(w - 1));
                    clamped[1] = std::clamp(clamped[1], 0.0f, static_cast<float>(h - 1));
                    clamped[2] = std::clamp(clamped[2], 0.0f, static_cast<float>(d - 1));
                    POI* poi = new POI;
                    poi->p = clamped;
                    poi->n = normal;
                    poi->surfaceId = segmentId.toStdString();
                    poi->surfacePtr = focusPoint->ptr;
                    _state->setPOI("focus", poi);
                }
            });
    connect(_surfacePanel.get(), &SurfacePanelController::alphaCompRefineRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onAlphaCompRefine(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::slimFlattenRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onSlimFlatten(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::straightenRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onStraighten(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::abfFlattenRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onABFFlatten(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::exportTifxyzChunksRequested,
        this, [this](const QString& segmentId) {
            _segmentationCommandHandler->onExportWidthChunks(segmentId.toStdString());
        });
    connect(_surfacePanel.get(), &SurfacePanelController::rasterizeSegmentsRequested,
        this, [this](const QStringList& segmentIds) {
            _segmentationCommandHandler->onRasterizeSegments(segmentIds);
        });
    connect(_surfacePanel.get(), &SurfacePanelController::generateSegmentMaskRequested,
        this, &CWindow::onEditMaskPressed);
    connect(_surfacePanel.get(), &SurfacePanelController::appendSegmentMaskRequested,
        this, &CWindow::onAppendMaskPressed);
    connect(_surfacePanel.get(), &SurfacePanelController::addIgnoreLabelRequested,
        this, [this]() {
            _segmentationCommandHandler->onAddIgnoreLabel();
        });
    connect(_surfacePanel.get(), &SurfacePanelController::recalcAreaRequested,
            this, [this](const QStringList& segmentIds) {
                if (segmentIds.isEmpty()) return;
                std::vector<std::string> ids;
                ids.reserve(segmentIds.size());
                for (const auto& id : segmentIds) {
                    ids.push_back(id.toStdString());
                }
                auto results = SurfaceAreaCalculator::calculateAreas(_state->vpkg(), _state->currentVolume(), ids);
                int okCount = 0, failCount = 0;
                QStringList skippedIds;
                for (const auto& r : results) {
                    if (r.success) {
                        ++okCount;
                        // Update tree widget
                        QTreeWidgetItemIterator it(treeWidgetSurfaces);
                        while (*it) {
                            if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() == r.segmentId) {
                                (*it)->setText(2, QString::number(r.areaCm2, 'f', 3));
                                break;
                            }
                            ++it;
                        }
                    } else {
                        ++failCount;
                        skippedIds << QString::fromStdString(r.segmentId + " (" + r.errorReason + ")");
                    }
                }
                if (okCount > 0) {
                    showStatusBarMessage(
                        tr("Recalculated area for %1 segment(s).").arg(okCount), 5000);
                }
                if (failCount > 0) {
                    QMessageBox::warning(this, tr("Area Recalculation"),
                        tr("Updated: %1\nSkipped: %2\n\n%3").arg(okCount).arg(failCount).arg(skippedIds.join("\n")));
                }
            });
    connect(_surfacePanel.get(), &SurfacePanelController::statusMessageRequested,
            this, [this](const QString& message, int timeoutMs) {
                showStatusBarMessage(message, timeoutMs);
            });

    const auto attachScrollAreaToDock = [](QDockWidget* dock, QWidget* content, const QString& objectName) {
        if (!dock || !content) {
            return;
        }

        // Delete any existing widget from the .ui file to prevent ghosting
        if (auto* oldWidget = dock->widget()) {
            delete oldWidget;
        }

        auto* container = new QWidget(dock);
        container->setObjectName(objectName);
        auto* layout = new QVBoxLayout(container);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(0);
        layout->addWidget(content);
        layout->addStretch(1);

        auto* scrollArea = new QScrollArea(dock);
        scrollArea->setFrameShape(QFrame::NoFrame);
        scrollArea->setWidgetResizable(true);
        scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        scrollArea->setWidget(container);

        dock->setWidget(scrollArea);
    };


    // Create Segmentation widget
    _segmentationWidget = new SegmentationWidget();
    _segmentationWidget->setNormalGridAvailable(_normalGridAvailable);
    _segmentationWidget->setNormalGridPath(_normalGridPath);
    const QString initialHint = _normalGridAvailable
        ? tr("Normal grids directory found.")
        : tr("No volume package loaded.");
    _segmentationWidget->setNormalGridPathHint(initialHint);
    attachScrollAreaToDock(ui.dockWidgetSegmentation, _segmentationWidget, QStringLiteral("dockWidgetSegmentationContent"));

    // Create Lasagna dock from the panel already constructed by SegmentationWidget
    {
        auto* panel = _segmentationWidget->lasagnaPanel();
        panel->setState(_state);
        panel->setVisible(true);
        _lasagnaWorkspaceWindow->setCentralWidget(panel);

        _lasagnaDock = new QDockWidget(tr("Lasagna"), this);
        _lasagnaDock->setObjectName(QStringLiteral("dockWidgetLasagna"));
        attachScrollAreaToDock(_lasagnaDock,
                               panel->createCompactView(_lasagnaDock),
                               QStringLiteral("dockWidgetLasagnaContent"));
        _segmentWorkspaceWindow->addDockWidget(Qt::RightDockWidgetArea, _lasagnaDock);
        connect(panel, &SegmentationLasagnaPanel::openLasagnaWorkspaceRequested,
                this, &CWindow::switchToLasagnaWorkspace);
        connect(panel, &SegmentationLasagnaPanel::lasagnaOutputActivated,
                this, &CWindow::selectLasagnaOutputSegment);
    }

    _segmentationEdit = std::make_unique<SegmentationEditManager>(this);
    _segmentationEdit->setViewerManager(_viewerManager.get());
    _segmentationOverlay = std::make_unique<SegmentationOverlayController>(_state, this);
    _segmentationOverlay->setEditManager(_segmentationEdit.get());
    _segmentationOverlay->setViewerManager(_viewerManager.get());
    _surfaceRotationOverlay = std::make_unique<SurfaceRotationOverlayController>(_state, this);
    _surfaceRotationOverlay->setViewerManager(_viewerManager.get());

    _segmentationModule = std::make_unique<SegmentationModule>(
        _segmentationWidget,
        _segmentationEdit.get(),
        _segmentationOverlay.get(),
        _viewerManager.get(),
        _state,
        _state->pointCollection(),
        _segmentationWidget->isEditingEnabled(),
        this);

    if (_segmentationModule && _planeSlicingOverlay) {
        QPointer<PlaneSlicingOverlayController> overlayPtr(_planeSlicingOverlay.get());
        _segmentationModule->setRotationHandleHitTester(
            [overlayPtr](VolumeViewerBase* viewer, const cv::Vec3f& worldPos) {
                if (!overlayPtr) {
                    return false;
                }
                return overlayPtr->isVolumePointNearRotationHandle(viewer, worldPos, 1.5);
            });
    }

    if (_viewerManager) {
        _viewerManager->setSegmentationOverlay(_segmentationOverlay.get());
    }

    // Wire annotate mode: module -> header row (via SegmentationWidget)
    // NOTE: connections involving _point_collection_widget are below, after it is created.
    connect(_segmentationModule.get(), &SegmentationModule::annotateModeChanged,
            _segmentationWidget, &SegmentationWidget::setAnnotateChecked);
    connect(_segmentationModule.get(), &SegmentationModule::annotationPointFocused,
            this, &CWindow::onPointDoubleClicked);

    connect(_segmentationModule.get(), &SegmentationModule::editingEnabledChanged,
            this, &CWindow::onSegmentationEditingModeChanged);
    connect(_segmentationModule.get(), &SegmentationModule::segmentationFolderChanged,
            this, [this](const QString& surfaceId) {
                refreshSegmentationDirectoryDropdown();
                if (!_state || !_state->vpkg() || !_surfacePanel) {
                    return;
                }

                _surfacePanel->setVolumePkg(_state->vpkg());
                _surfacePanel->resetTagUi();
                _surfacePanel->loadSurfaces(true);
                _surfacePanel->refreshPointSetFilterOptions();

                if (!restoreActiveSurfaceAfterSurfaceReload(surfaceId.toStdString())) {
                    clearSurfaceSelection();
                    _state->setSurface("segmentation", nullptr, true);
                }
            });
    connect(_segmentationModule.get(), &SegmentationModule::statusMessageRequested,
            this, &CWindow::onShowStatusMessage);
    connect(_segmentationModule.get(), &SegmentationModule::stopToolsRequested,
            this, &CWindow::onSegmentationStopToolsRequested);
    connect(_segmentationModule.get(), &SegmentationModule::growthInProgressChanged,
            this, &CWindow::onSegmentationGrowthStatusChanged);
    connect(_segmentationModule.get(), &SegmentationModule::focusPoiRequested,
            this, [this](const cv::Vec3f& position, QuadSurface* base) {
                Q_UNUSED(position);
                _axisAlignedSliceController->applyOrientation(base);
            });
    connect(_segmentationModule.get(), &SegmentationModule::growSurfaceRequested,
            this, &CWindow::onGrowSegmentationSurface);
    connect(_segmentationModule.get(), &SegmentationModule::approvalMaskSaved,
            _fileWatcher.get(), &FileWatcherService::markSegmentRecentlyEdited);

    SegmentationGrower::Context growerContext{
        _segmentationModule.get(),
        _segmentationWidget,
        _state,
        _viewerManager.get()
    };
    SegmentationGrower::UiCallbacks growerCallbacks{
        [this](const QString& text, int timeout) {
            if (statusBar()) {
                showStatusBarMessage(text, timeout);
            }
        },
        [this](QuadSurface* surface) {
            _axisAlignedSliceController->applyOrientation(surface);
        }
    };
    _segmentationGrower = std::make_unique<SegmentationGrower>(growerContext, growerCallbacks, this);

    _segmentationCommandHandler = std::make_unique<SegmentationCommandHandler>(this, _state, this);
    _segmentationCommandHandler->setCmdRunner(_cmdRunner);
    _segmentationCommandHandler->setSurfacePanel(_surfacePanel.get());
    _segmentationCommandHandler->setSegmentationGrower(_segmentationGrower.get());
    initializeCommandLineRunner();
    _segmentationCommandHandler->setIsEditingCheck([this]() -> bool {
        return _segmentationModule && _segmentationModule->isEditingApprovalMask();
    });
    _segmentationCommandHandler->setClearSelectionCallback([this]() {
        clearSurfaceSelection();
    });
    _segmentationCommandHandler->setRestoreSelectionCallback([this](const std::string& id) {
        if (treeWidgetSurfaces) {
            QTreeWidgetItemIterator it(treeWidgetSurfaces);
            while (*it) {
                if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() == id) {
                    treeWidgetSurfaces->setCurrentItem(*it);
                    break;
                }
                ++it;
            }
        }
    });
    _segmentationCommandHandler->setNormal3dZarrPathGetter([this]() -> QString {
        return _segmentationWidget ? _segmentationWidget->normal3dZarrPath() : QString();
    });
    _segmentationCommandHandler->setNormalGridPathGetter([this]() -> QString {
        updateNormalGridAvailability();
        return _normalGridPath;
    });
    connect(_segmentationCommandHandler.get(), &SegmentationCommandHandler::statusMessage,
            this, &CWindow::onShowStatusMessage);

    _fileWatcher->setSurfacePanel(_surfacePanel.get());
    _fileWatcher->setSegmentationModule(_segmentationModule.get());
    _fileWatcher->setTreeWidget(treeWidgetSurfaces);

    connect(_segmentationWidget, &SegmentationWidget::copyWithNtRequested,
            this, &CWindow::onCopyWithNtRequested);
    connect(_segmentationWidget, &SegmentationWidget::volumeSelectionChanged, this, [this](const QString& volumeId) {
        if (!_state->vpkg()) {
            showStatusBarMessage(tr("No volume package loaded."), 4000);
            if (_segmentationWidget) {
                const QString fallbackId = QString::fromStdString(!_state->segmentationGrowthVolumeId().empty()
                                                                   ? _state->segmentationGrowthVolumeId()
                                                                   : _state->currentVolumeId());
                _segmentationWidget->setActiveVolume(fallbackId);
            }
            return;
        }

        const std::string requestedId = volumeId.toStdString();
        try {
            auto vol = _state->vpkg()->volume(requestedId);
            _state->setSegmentationGrowthVolumeId(requestedId);
            // Set volume zarr path for neural tracing
            if (_segmentationWidget && vol) {
                _segmentationWidget->setVolumeZarrPath(QString::fromStdString(vol->path().string()));
            }
            showStatusBarMessage(tr("Using volume '%1' for surface growth.").arg(volumeId), 2500);
        } catch (const std::out_of_range&) {
            showStatusBarMessage(tr("Volume '%1' not found in this package.").arg(volumeId), 4000);
            if (_segmentationWidget) {
                const QString fallbackId = QString::fromStdString(!_state->currentVolumeId().empty()
                                                                   ? _state->currentVolumeId()
                                                                   : std::string{});
                _segmentationWidget->setActiveVolume(fallbackId);
                _state->setSegmentationGrowthVolumeId(_state->currentVolumeId());
            }
        }
    });

    // -- Lasagna connections --
    connect(_segmentationWidget, &SegmentationWidget::seedFromFocusRequested, this, [this]() {
        POI* focus = _state ? _state->poi("focus") : nullptr;
        if (focus)
            _segmentationWidget->setSeedFromFocus(
                static_cast<int>(focus->p[0]),
                static_cast<int>(focus->p[1]),
                static_cast<int>(focus->p[2]));
    });

    connect(_segmentationWidget, &SegmentationWidget::lasagnaOptimizeRequested, this, [this]() {
        if (auto* panel = _segmentationWidget->lasagnaPanel()) {
            panel->startOptimization(_state, statusBar());
        }
    });

    connect(_segmentationWidget, &SegmentationWidget::lasagnaStopRequested, this, [this]() {
        LasagnaServiceManager::instance().stopOptimization();
        showStatusBarMessage(tr("Lasagna optimization stop requested."), 3000);
    });
    connect(&LasagnaServiceManager::instance(), &LasagnaServiceManager::serviceStarted,
            this, &CWindow::updateAtlasFiberDocks);
    connect(&LasagnaServiceManager::instance(), &LasagnaServiceManager::serviceStopped,
            this, &CWindow::updateAtlasFiberDocks);

    // Add only the segments placed by lasagna instead of rescanning every surface.
    connect(&LasagnaServiceManager::instance(), &LasagnaServiceManager::resultsPlaced,
            this, [this](const QString& outputDir, const QStringList& segmentNames) {
        showStatusBarMessage(
            tr("Lasagna optimization finished. Added %1 segment(s) from %2")
                .arg(segmentNames.size())
                .arg(outputDir), 5000);
        if (_surfacePanel) {
            for (const QString& segmentName : segmentNames) {
                _surfacePanel->addSingleSegmentation(segmentName.toStdString());
            }
            _surfacePanel->refreshFiltersOnly();
        }
        // corr_points_results will be loaded when the new segment is activated
    });

    // Create Seeding widget
    _seedingWidget = new SeedingWidget(_state->pointCollection(), _state);
    _seedingWidget->setViewerManager(_viewerManager.get());
    attachScrollAreaToDock(ui.dockWidgetDistanceTransform, _seedingWidget, QStringLiteral("dockWidgetDistanceTransformContent"));

    _seedingWidget->setState(_state);
    connect(_state, &CState::volumeChanged, _seedingWidget,
            static_cast<void (SeedingWidget::*)(std::shared_ptr<Volume>, const std::string&)>(&SeedingWidget::onVolumeChanged));
    connect(_state, &CState::volumeChanged, this,
            [this](std::shared_ptr<Volume>, const std::string&) {
                if (_surfaceAffineTransforms) {
                    _surfaceAffineTransforms->refresh();
                }
            });
    connect(_seedingWidget, &SeedingWidget::sendStatusMessageAvailable, this, &CWindow::onShowStatusMessage);
    connect(_state, &CState::surfacesLoaded, _seedingWidget, &SeedingWidget::onSurfacesLoaded);

    _wrapAnnotationWidget = new WrapAnnotationWidget(_state->pointCollection(), this);
    _wrapAnnotationWidget->setObjectName("wrapAnnotationDock");
    _segmentWorkspaceWindow->addDockWidget(Qt::RightDockWidgetArea, _wrapAnnotationWidget);
    connect(_wrapAnnotationWidget, &WrapAnnotationWidget::relWindingAnnotationToggled,
            _seedingWidget, &SeedingWidget::setRelWindingAnnotationMode);
    connect(_wrapAnnotationWidget, &WrapAnnotationWidget::relWindingIntersectionSourceChanged,
            _seedingWidget, &SeedingWidget::setRelWindingIntersectionSource);
    connect(_wrapAnnotationWidget, &WrapAnnotationWidget::relWindingPatchToleranceChanged,
            _seedingWidget, &SeedingWidget::setRelWindingPatchTolerance);
    connect(_seedingWidget, &SeedingWidget::relWindingAnnotationModeChanged,
            _wrapAnnotationWidget, &WrapAnnotationWidget::setRelWindingAnnotationChecked);
    _seedingWidget->setRelWindingIntersectionSource(_wrapAnnotationWidget->relWindingIntersectionSource());
    _seedingWidget->setRelWindingPatchTolerance(_wrapAnnotationWidget->relWindingPatchTolerance());

    // Create and add the point collection widget
    _point_collection_widget = new CPointCollectionWidget(_state->pointCollection(), this);
    _point_collection_widget->setObjectName("pointCollectionDock");
    _segmentWorkspaceWindow->addDockWidget(Qt::RightDockWidgetArea, _point_collection_widget);

    _atlasControlDock = new AtlasControlPointsDock(this);
    _segmentWorkspaceWindow->addDockWidget(Qt::RightDockWidgetArea, _atlasControlDock);
    connect(_atlasControlDock, &AtlasControlPointsDock::resultsChanged,
            this, [this](const AtlasControlPointResults& results) {
                if (_atlasControlOverlay) {
                    _atlasControlOverlay->setResults(results);
                    _atlasControlOverlay->setOverlayEnabled(_atlasControlDock && _atlasControlDock->overlayChecked());
                }
            });
    connect(_atlasControlDock, &AtlasControlPointsDock::overlayToggled,
            this, [this](bool enabled) {
                if (_atlasControlOverlay) {
                    _atlasControlOverlay->setOverlayEnabled(enabled);
                }
            });
    connect(_atlasControlDock, &AtlasControlPointsDock::controlPointSelected,
            this, [this](const AtlasControlPointResult& point) {
                if (_atlasControlOverlay) {
                    _atlasControlOverlay->setSelectedPoint(point.fiberId, point.controlIndex);
                }
            });
    connect(_atlasControlDock, &AtlasControlPointsDock::controlPointActivated,
            this, [this](const AtlasControlPointResult& point) {
                if (_atlasControlOverlay) {
                    _atlasControlOverlay->setSelectedPoint(point.fiberId, point.controlIndex);
                }
                const cv::Vec3f* focusPoint = nullptr;
                if (isFiniteVec3(point.snapTargetXyz)) {
                    focusPoint = &point.snapTargetXyz;
                } else if (isFiniteVec3(point.meshXyz)) {
                    focusPoint = &point.meshXyz;
                }

                if (focusPoint) {
                    const std::string sourceId = _state ? _state->activeSurfaceId() : std::string{};
                    centerFocusAt(*focusPoint, cv::Vec3f(0.0f, 0.0f, 0.0f), sourceId);
                    recenterPlaneViewersOn(*focusPoint);
                } else if (const auto surfacePoint = atlasControlGridToSurface(segmentationBaseViewer(), point)) {
                    centerViewerOnSurfacePointForNavigation(segmentationBaseViewer(), *surfacePoint);
                }
            });

    // Selection dock (removed per request; selection actions remain in the menu)
    if (_viewerManager) {
        for (auto* viewer : _viewerManager->baseViewers()) {
            if (viewer) {
                if (auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject())) {
                    configureChunkedViewerConnections(chunkedViewer);
                }
            }
        }
    }
    connect(_point_collection_widget, &CPointCollectionWidget::pointDoubleClicked, this, &CWindow::onPointDoubleClicked);
    connect(_point_collection_widget, &CPointCollectionWidget::convertPointToAnchorRequested, this, &CWindow::onConvertPointToAnchor);
    connect(_point_collection_widget, &CPointCollectionWidget::focusViewsRequested, this, &CWindow::onFocusViewsRequested);
    connect(_wrapAnnotationWidget, &WrapAnnotationWidget::pointDoubleClicked, this, &CWindow::onPointDoubleClicked);
    connect(_wrapAnnotationWidget, &WrapAnnotationWidget::focusViewsRequested, this, &CWindow::onFocusViewsRequested);
    if (_pointsOverlay) {
        _pointsOverlay->setViewTolerance(_point_collection_widget->pointViewTolerance());
        connect(_point_collection_widget, &CPointCollectionWidget::pointViewToleranceChanged,
                _pointsOverlay.get(), &PointsOverlayController::setViewTolerance);
    }

    // Tab the docks - keep Segmentation, Lasagna, Seeding, and Point Collections together
    // Wire annotate mode & annotation selection: dock widget <-> segmentation module
    // (must be after _point_collection_widget creation)
    connect(_point_collection_widget, &CPointCollectionWidget::annotateToggled,
            _segmentationModule.get(), &SegmentationModule::setAnnotateMode);
    connect(_segmentationModule.get(), &SegmentationModule::annotateModeChanged,
            _point_collection_widget, &CPointCollectionWidget::setAnnotateChecked);
    connect(_segmentationModule.get(), &SegmentationModule::annotationPointSelected,
            _point_collection_widget, &CPointCollectionWidget::selectPoint);
    connect(_segmentationModule.get(), &SegmentationModule::annotationCollectionSelected,
            _point_collection_widget, &CPointCollectionWidget::selectCollection);
    connect(_point_collection_widget, &CPointCollectionWidget::collectionSelected,
            _segmentationModule.get(), &SegmentationModule::setSelectedAnnotationCollection);

    // Create fiber annotation controller and dock
    _fiberController = std::make_unique<FiberAnnotationController>(
        _state, _state->pointCollection(), this);
    _fiberController->setMdiArea(_fiberSliceMdiArea);

    _fiberWidget = new CFiberWidget(this);
    _fiberWidget->setObjectName("fiberDock");
    _segmentWorkspaceWindow->addDockWidget(Qt::RightDockWidgetArea, _fiberWidget);

    if (_fiberSliceWorkspaceWindow) {
        _fiberSliceWidget = new CFiberWidget(_fiberSliceWorkspaceWindow);
        _fiberSliceWidget->setObjectName(QStringLiteral("fiberSliceDock"));
        _fiberSliceWidget->setWindowTitle(tr("Fiber Slice Fibers"));
        _fiberSliceWorkspaceWindow->addDockWidget(Qt::LeftDockWidgetArea, _fiberSliceWidget);
    }

    auto syncShowFiberChecks = [this](bool checked) {
        if (_fiberWidget) {
            _fiberWidget->setShowFibersChecked(checked);
        }
        if (_fiberSliceWidget) {
            _fiberSliceWidget->setShowFibersChecked(checked);
        }
    };
    auto syncFiberViewDistances = [this](double distance) {
        if (_fiberWidget) {
            _fiberWidget->setFiberViewDistance(distance);
        }
        if (_fiberSliceWidget) {
            _fiberSliceWidget->setFiberViewDistance(distance);
        }
    };
    auto setFiberOverlayVisible = [this, syncShowFiberChecks](bool visible) {
        if (!_fiberOverlay || !_fiberOverlay->hasChains()) {
            visible = false;
        }
        if (_fiberOverlay) {
            _fiberOverlay->setVisible(visible);
        }
        syncShowFiberChecks(visible);
    };
    auto setFiberViewDistance = [this, syncFiberViewDistances](double distance) {
        if (_fiberOverlay) {
            _fiberOverlay->setViewDistance(distance);
        }
        if (_spiralWorkspace) {
            _spiralWorkspace->setFiberViewDistance(distance);
        }
        syncFiberViewDistances(distance);
    };
    if (_fiberWidget) {
        connect(_fiberWidget, &CFiberWidget::showFibersToggled,
                this, setFiberOverlayVisible);
        connect(_fiberWidget, &CFiberWidget::fiberViewDistanceChanged,
                this, setFiberViewDistance);
    }
    if (_fiberSliceWidget) {
        connect(_fiberSliceWidget, &CFiberWidget::showFibersToggled,
                this, setFiberOverlayVisible);
        connect(_fiberSliceWidget, &CFiberWidget::fiberViewDistanceChanged,
                this, setFiberViewDistance);
    }
    setFiberViewDistance(_fiberWidget->fiberViewDistance());

    if (_lineAnnotationController) {
        auto toFiberWidgetAlignment =
            [](const LineAnnotationController::FiberSummary::AlignmentMetrics& source) {
                CFiberWidget::FiberEntry::AlignmentMetrics alignment;
                alignment.available = source.available;
                alignment.pending = source.pending;
                alignment.sampleCount = source.sampleCount;
                alignment.meanErrorDegrees = source.meanErrorDegrees;
                alignment.maxErrorDegrees = source.maxErrorDegrees;
                alignment.error = source.error;
                return alignment;
            };
        auto updateFiberList =
            [this, toFiberWidgetAlignment, syncShowFiberChecks](
                const std::vector<LineAnnotationController::FiberSummary>& fibers) {
            std::vector<CFiberWidget::FiberEntry> entries;
            entries.reserve(fibers.size());
            for (const auto& fiber : fibers) {
                CFiberWidget::FiberEntry::AlignmentMetrics alignment =
                    toFiberWidgetAlignment(fiber.alignment);

                std::vector<CFiberWidget::FiberEntry::SpanEntry> spans;
                spans.reserve(fiber.spans.size());
                for (const auto& span : fiber.spans) {
                    CFiberWidget::FiberEntry::AlignmentMetrics spanAlignment =
                        toFiberWidgetAlignment(span.alignment);
                    spans.push_back(CFiberWidget::FiberEntry::SpanEntry{
                        span.spanIndex,
                        span.firstControlIndex,
                        span.secondControlIndex,
                        span.controlPointCount,
                        span.linePointCount,
                        span.lengthVx,
                        spanAlignment,
                    });
                }

                entries.push_back(CFiberWidget::FiberEntry{
                    fiber.id,
                    fiber.name,
                    fiber.controlPointCount,
                    fiber.linePointCount,
                    fiber.lengthVx,
                    alignment,
                    spans,
                    fiber.hvZDistance,
                    fiber.hvFiberLength,
                    fiber.horizontalScore,
                    fiber.verticalScore,
                    fiber.automaticCertainty,
                    fiber.automaticHvTag,
                    fiber.manualHvTag,
                    fiber.tags,
                });
            }
            if (_fiberWidget) {
                _fiberWidget->setFibers(entries);
                _fiberWidget->setKnownTags(_lineAnnotationController->knownFiberTags());
            }
            if (_fiberSliceWidget) {
                _fiberSliceWidget->setFibers(entries);
                _fiberSliceWidget->setKnownTags(_lineAnnotationController->knownFiberTags());
            }

            std::vector<FiberOverlayController::Chain> chains;
            for (const auto& snapshot : _lineAnnotationController->fiberSnapshots()) {
                if (snapshot.controlPoints.empty()) {
                    continue;
                }
                FiberOverlayController::Chain chain;
                chain.id = snapshot.id;
                chain.points.reserve(snapshot.controlPoints.size());
                for (const cv::Vec3d& point : snapshot.controlPoints) {
                    chain.points.emplace_back(static_cast<float>(point[0]),
                                              static_cast<float>(point[1]),
                                              static_cast<float>(point[2]));
                }
                chains.push_back(std::move(chain));
            }
            const bool fibersAvailable = !chains.empty();
            if (_fiberOverlay) {
                _fiberOverlay->setChains(std::move(chains));
            }
            if (_fiberWidget) {
                _fiberWidget->setShowFibersAvailable(fibersAvailable);
            }
            if (_fiberSliceWidget) {
                _fiberSliceWidget->setShowFibersAvailable(fibersAvailable);
            }
            if (!fibersAvailable) {
                syncShowFiberChecks(false);
            }
            updateAtlasFiberDocks();
        };
        auto updateFiberMetricRows =
            [this, toFiberWidgetAlignment](
                uint64_t fiberId,
                LineAnnotationController::FiberSummary::AlignmentMetrics alignment,
                const std::vector<LineAnnotationController::FiberSummary::AlignmentMetrics>& spanAlignments) {
                std::vector<CFiberWidget::FiberEntry::AlignmentMetrics> widgetSpanAlignments;
                widgetSpanAlignments.reserve(spanAlignments.size());
                for (const auto& spanAlignment : spanAlignments) {
                    widgetSpanAlignments.push_back(toFiberWidgetAlignment(spanAlignment));
                }
                const CFiberWidget::FiberEntry::AlignmentMetrics widgetAlignment =
                    toFiberWidgetAlignment(alignment);
                if (_fiberWidget) {
                    _fiberWidget->updateAlignmentMetrics(fiberId,
                                                         widgetAlignment,
                                                         widgetSpanAlignments);
                }
                if (_fiberSliceWidget) {
                    _fiberSliceWidget->updateAlignmentMetrics(fiberId,
                                                              widgetAlignment,
                                                              widgetSpanAlignments);
                }
            };
        auto setFiberMetricsPending = [this](bool pending) {
            if (_fiberWidget) {
                _fiberWidget->setAlignmentMetricsPending(pending);
            }
            if (_fiberSliceWidget) {
                _fiberSliceWidget->setAlignmentMetricsPending(pending);
            }
        };
        auto connectFiberWidget = [this](CFiberWidget* widget) {
            if (!widget || !_lineAnnotationController) {
                return;
            }
            connect(widget,
                    &CFiberWidget::fiberOpenRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::openFiber);
            connect(widget,
                    &CFiberWidget::fiberSpanOpenRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::openFiberSpan);
            connect(widget,
                    &CFiberWidget::deleteFibersRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::deleteFibers);
            connect(widget,
                    &CFiberWidget::renameFiberFileRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::renameFiberFile);
            connect(widget,
                    &CFiberWidget::importFibersRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::importFibers);
            connect(widget,
                    &CFiberWidget::exportFibersRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::exportFibers);
            connect(widget,
                    &CFiberWidget::metricsCalculationRequested,
                    _lineAnnotationController.get(),
                    [this](std::vector<uint64_t> orderedFiberIds) {
                        if (_lineAnnotationController) {
                            _lineAnnotationController->calculateFiberAlignmentMetrics(
                                std::move(orderedFiberIds));
                        }
                    });
            connect(widget,
                    &CFiberWidget::manualHvTagChanged,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::setFiberManualHvTag);
            connect(widget,
                    &CFiberWidget::fiberTagChanged,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::setFiberTag);
            connect(widget,
                    &CFiberWidget::hvScoreRecalculationRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::recalculateFiberHvClassification);
            connect(widget,
                    &CFiberWidget::newAtlasFromFiberRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::createAtlasFromFiber);
            connect(widget,
                    &CFiberWidget::addFibersToPointCollectionsRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::addFibersToPointCollections);
            connect(widget,
                    &CFiberWidget::addFibersToSpiralFitRequested,
                    this,
                    [this](std::vector<uint64_t> fiberIds) {
                        if (!_spiralWorkspace || !_lineAnnotationController) {
                            return;
                        }
                        for (uint64_t fiberId : fiberIds) {
                            const auto path = _lineAnnotationController->fiberFilePath(fiberId);
                            if (path.empty()) {
                                showStatusBarMessage(tr("Fiber %1 has no saved JSON file yet").arg(fiberId), 5000);
                                continue;
                            }
                            _spiralWorkspace->addFiberToCurrentFit(QString::fromStdString(path.string()));
                        }
                    });
            connect(widget,
                    &CFiberWidget::fiberSliceRequested,
                    this,
                    [this](uint64_t fiberId) {
                        if (_fiberWidget) {
                            _fiberWidget->selectFiber(fiberId);
                        }
                        if (_fiberSliceWidget) {
                            _fiberSliceWidget->selectFiber(fiberId);
                        }
                        switchToFiberSliceWorkspace();
                        if (_lineAnnotationController) {
                            _lineAnnotationController->showFiberSlice(fiberId, _fiberSliceMdiArea);
                        }
                    });
        };
        connect(_lineAnnotationController.get(),
                &LineAnnotationController::fibersChanged,
                this,
                updateFiberList);
        connect(_lineAnnotationController.get(),
                &LineAnnotationController::fiberAlignmentMetricsReset,
                this,
                setFiberMetricsPending);
        connect(_lineAnnotationController.get(),
                &LineAnnotationController::fiberAlignmentMetricsUpdated,
                this,
                updateFiberMetricRows);
        connectFiberWidget(_fiberWidget);
        connectFiberWidget(_fiberSliceWidget);
        updateFiberList(_lineAnnotationController->fiberSummaries());
    }
    connect(_fiberController.get(), &FiberAnnotationController::crosshairModeChanged,
            this, &CWindow::onFiberCrosshairModeChanged);
    connect(_fiberController.get(), &FiberAnnotationController::requestFiberViewers,
            this, &CWindow::onFiberViewersRequested);
    connect(_fiberController.get(), &FiberAnnotationController::annotationFinished,
            this, &CWindow::onFiberAnnotationFinished);

    ensureDockWidgetFeatures(_fiberWidget);
    connect(_fiberWidget, &QDockWidget::topLevelChanged, this, &CWindow::scheduleWindowStateSave);
    connect(_fiberWidget, &QDockWidget::dockLocationChanged, this, &CWindow::scheduleWindowStateSave);
    ensureDockWidgetFeatures(_fiberSliceWidget);
    if (_fiberSliceWidget) {
        connect(_fiberSliceWidget, &QDockWidget::topLevelChanged, this, &CWindow::scheduleWindowStateSave);
        connect(_fiberSliceWidget, &QDockWidget::dockLocationChanged, this, &CWindow::scheduleWindowStateSave);
    }
    ensureDockWidgetFeatures(_atlasWorkspaceFiberDock);
    if (_atlasWorkspaceFiberDock) {
        connect(_atlasWorkspaceFiberDock, &QDockWidget::topLevelChanged, this, &CWindow::scheduleWindowStateSave);
        connect(_atlasWorkspaceFiberDock, &QDockWidget::dockLocationChanged, this, &CWindow::scheduleWindowStateSave);
    }

    // Tab the docks - keep Segmentation, Lasagna, Seeding, Point Collections, and Fibers together
    _segmentWorkspaceWindow->tabifyDockWidget(ui.dockWidgetSegmentation, _lasagnaDock);
    _segmentWorkspaceWindow->tabifyDockWidget(ui.dockWidgetSegmentation, ui.dockWidgetDistanceTransform);
    _segmentWorkspaceWindow->tabifyDockWidget(ui.dockWidgetSegmentation, _point_collection_widget);
    _segmentWorkspaceWindow->tabifyDockWidget(ui.dockWidgetSegmentation, _atlasControlDock);
    _segmentWorkspaceWindow->tabifyDockWidget(ui.dockWidgetSegmentation, _fiberWidget);

    // Make Segmentation dock the active tab by default
    ui.dockWidgetSegmentation->raise();

    ViewerControlsPanel::UiRefs viewerControlsUi{
        .contents = ui.dockWidgetViewerControlsContents,
        .viewScrollArea = ui.scrollAreaView,
        .viewContents = ui.dockWidgetViewContents,
        .overlayScrollArea = ui.scrollAreaOverlay,
        .overlayContents = ui.dockWidgetOverlayContents,
        .normalVisualizationContents = ui.dockWidgetNormalVisContents,
        .showSurfaceNormals = ui.chkShowSurfaceNormals,
        .normalArrowLengthLabel = ui.labelNormalArrowLength,
        .normalArrowLengthSlider = ui.sliderNormalArrowLength,
        .normalArrowLengthValueLabel = ui.labelNormalArrowLengthValue,
        .normalMaxArrowsLabel = ui.labelNormalMaxArrows,
        .normalMaxArrowsSlider = ui.sliderNormalMaxArrows,
        .normalMaxArrowsValueLabel = ui.labelNormalMaxArrowsValue,
        .zoomInButton = ui.btnZoomIn,
        .zoomOutButton = ui.btnZoomOut,
        .volumeWindowContainer = ui.volumeWindowContainer,
        .overlayWindowContainer = ui.overlayWindowContainer,
        .intersectionThicknessSpin = ui.doubleSpinIntersectionThickness,
    };
    _viewerControlsPanel = std::make_unique<ViewerControlsPanel>(viewerControlsUi,
                                                                 _viewerManager.get(),
                                                                 ui.dockWidgetViewerControlsContents);
    ViewerCompositePanel::UiRefs compositeUi{
        .scrollArea = ui.scrollAreaComposite,
        .contents = ui.dockWidgetCompositeContents,
        .compositeEnabled = ui.chkCompositeEnabled,
        .compositeMode = ui.cmbCompositeMode,
        .layersInFront = ui.spinLayersInFront,
        .layersBehind = ui.spinLayersBehind,
        .alphaMinLabel = ui.lblAlphaMin,
        .alphaMin = ui.spinAlphaMin,
        .alphaMaxLabel = ui.lblAlphaMax,
        .alphaMax = ui.spinAlphaMax,
        .alphaThresholdLabel = ui.lblAlphaThreshold,
        .alphaThreshold = ui.spinAlphaThreshold,
        .materialLabel = ui.lblMaterial,
        .material = ui.spinMaterial,
        .reverseDirection = ui.chkReverseDirection,
        .planeCompositeXY = ui.chkPlaneCompositeXY,
        .planeCompositeXZ = ui.chkPlaneCompositeXZ,
        .planeCompositeYZ = ui.chkPlaneCompositeYZ,
        .planeLayersFront = ui.spinPlaneLayersFront,
        .planeLayersBehind = ui.spinPlaneLayersBehind,
    };
    _viewerCompositePanel = new ViewerCompositePanel(compositeUi, _viewerManager.get(), ui.dockWidgetComposite);
    attachScrollAreaToDock(ui.dockWidgetComposite,
                           _viewerCompositePanel,
                           QStringLiteral("dockWidgetCompositeContent"));

    _inkDetectionDock = new QDockWidget(tr("Ink Detection"), this);
    _inkDetectionDock->setObjectName(QStringLiteral("dockWidgetInkDetection"));
    attachScrollAreaToDock(_inkDetectionDock,
                           new ViewerInkDetectionPanel(_viewerManager.get(), _inkDetectionDock),
                           QStringLiteral("dockWidgetInkDetectionContent"));

    _transformsDock = new QDockWidget(tr("Transforms"), this);
    _transformsDock->setObjectName(QStringLiteral("dockWidgetTransforms"));
    auto* transformsPanel = new ViewerTransformsPanel(_transformsDock);
    attachScrollAreaToDock(_transformsDock,
                           transformsPanel,
                           QStringLiteral("dockWidgetTransformsContent"));
    _viewerControlsPanel->setTransformsPanel(transformsPanel);

    connect(_viewerControlsPanel.get(), &ViewerControlsPanel::zoomInRequested,
            this, &CWindow::onZoomIn);
    connect(_viewerControlsPanel.get(), &ViewerControlsPanel::zoomOutRequested,
            this, &CWindow::onZoomOut);
    connect(_viewerManager.get(), &ViewerManager::zScrollSensitivityChanged,
            this, &CWindow::onZScrollSensitivityChanged);
    connect(_viewerControlsPanel.get(), &ViewerControlsPanel::statusMessageRequested,
            this, &CWindow::onShowStatusMessage);
    if (_viewerControlsPanel) {
        _viewerControlsPanel->setViewControlsEnabled(!ui.grpVolManager || ui.grpVolManager->isEnabled());
    }

    _surfaceAffineTransforms = std::make_unique<SurfaceAffineTransformController>(
        SurfaceAffineTransformController::Deps{
            .state = _state,
            .viewerControlsPanel = _viewerControlsPanel.get(),
            .viewerManager = _viewerManager.get(),
            .segmentationModule = _segmentationModule.get(),
            .surfacePanel = _surfacePanel.get(),
            .axisAlignedSliceController = _axisAlignedSliceController.get(),
            .dialogParent = this,
            .showStatus = [this](const QString& text, int timeoutMs) {
                onShowStatusMessage(text, timeoutMs);
            },
        },
        this);

    _segmentWorkspaceWindow->addDockWidget(Qt::LeftDockWidgetArea, ui.dockWidgetViewerControls);
    _segmentWorkspaceWindow->splitDockWidget(ui.dockWidgetVolumes, ui.dockWidgetViewerControls, Qt::Vertical);
    _segmentWorkspaceWindow->addDockWidget(Qt::LeftDockWidgetArea, ui.dockWidgetOverlay);
    _segmentWorkspaceWindow->addDockWidget(Qt::LeftDockWidgetArea, _inkDetectionDock);
    _segmentWorkspaceWindow->addDockWidget(Qt::LeftDockWidgetArea, _transformsDock);
    _segmentWorkspaceWindow->splitDockWidget(ui.dockWidgetViewerControls, ui.dockWidgetOverlay, Qt::Vertical);
    _segmentWorkspaceWindow->splitDockWidget(ui.dockWidgetOverlay, _inkDetectionDock, Qt::Vertical);
    _segmentWorkspaceWindow->splitDockWidget(_inkDetectionDock, _transformsDock, Qt::Vertical);

    auto hideLegacyViewerDocks = [this]() {
        for (QDockWidget* dock : { ui.dockWidgetNormalVis,
                                   ui.dockWidgetView,
                                   ui.dockWidgetRenderSettings,
                                   ui.dockWidgetComposite }) {
            if (!dock) {
                continue;
            }
            _segmentWorkspaceWindow->removeDockWidget(dock);
            dock->setVisible(false);
        }
    };
    hideLegacyViewerDocks();
    QTimer::singleShot(0, this, [hideLegacyViewerDocks]() {
        hideLegacyViewerDocks();
    });

    connect(_surfacePanel.get(), &SurfacePanelController::surfaceActivated,
            this, &CWindow::onSurfaceActivated);
    connect(_surfacePanel.get(), &SurfacePanelController::surfaceActivatedPreserveEditing,
            this, &CWindow::onSurfaceActivatedPreserveEditing);
    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }

    // new and remove path buttons
    // connect(ui.btnNewPath, SIGNAL(clicked()), this, SLOT(OnNewPathClicked()));
    // connect(ui.btnRemovePath, SIGNAL(clicked()), this, SLOT(OnRemovePathClicked()));

    // TODO CHANGE VOLUME LOADING; FIRST CHECK FOR OTHER VOLUMES IN THE STRUCTS
    if (ui.volSelect) {
        ui.volSelect->setLabelVisible(false);
        volSelect = ui.volSelect->comboBox();
    } else {
        volSelect = nullptr;
    }

    QComboBox* overlayVolumeSelect = nullptr;
    if (ui.overlayVolumeSelect) {
        ui.overlayVolumeSelect->setLabelVisible(false);
        overlayVolumeSelect = ui.overlayVolumeSelect->comboBox();
    }

    if (_volumeOverlay) {
        VolumeOverlayController::UiRefs overlayUi{
            .volumeSelect = overlayVolumeSelect,
            .colormapSelect = ui.overlayColormapSelect,
            .samplingMethodSelect = ui.overlayInterpolationSelect,
            .opacitySpin = ui.overlayOpacitySpin,
            .thresholdSpin = ui.overlayThresholdSpin,
            .maxDisplayedResolutionSpin = ui.overlayMaxDisplayedResolutionSpin,
            .compositeEnabledCheck = ui.chkOverlayComposite,
            .compositeMethodSelect = ui.cmbOverlayCompositeMethod,
            .compositeLayersFrontSpin = ui.spinOverlayCompositeLayersFront,
            .compositeLayersBehindSpin = ui.spinOverlayCompositeLayersBehind,
        };
        _volumeOverlay->setUi(overlayUi);
        if (_viewerControlsPanel) {
            _viewerControlsPanel->setOverlayWindowAvailable(_volumeOverlay->hasOverlaySelection());
        }
        updateActiveWorkspaceViewerControls();
    }

    // Setup surface overlay controls
    connect(ui.chkSurfaceOverlay, &QCheckBox::toggled, [this](bool checked) {
        if (!_viewerManager) return;
        _viewerManager->setSurfaceOverlaysEnabled(checked);
        ui.surfaceOverlaySelect->setEnabled(checked);
        ui.spinOverlapThreshold->setEnabled(checked);
    });

    connect(ui.surfaceOverlaySelect, &QPushButton::clicked,
            this, &CWindow::showSurfaceOverlaySelectionDialog);

    connect(ui.spinOverlapThreshold, qOverload<double>(&QDoubleSpinBox::valueChanged), [this](double value) {
        if (!_viewerManager) return;
        _viewerManager->setSurfaceOverlapThreshold(static_cast<float>(value));
    });

    // Initially disable surface overlay controls
    ui.surfaceOverlaySelect->setEnabled(false);
    ui.spinOverlapThreshold->setEnabled(false);

    // Initialize surface overlay selection model (will be populated when surfaces load)
    updateSurfaceOverlayDropdown();

    connectVolumeSelector(volSelect);

    auto* filterDropdown = ui.btnFilterDropdown;
    auto* cmbPointSetFilter = new QComboBox(this);
    auto* btnPointSetFilterAll = new QPushButton(tr("All"), this);
    auto* btnPointSetFilterNone = new QPushButton(tr("None"), this);
    auto* cmbPointSetFilterMode = new QComboBox(this);
    auto* spinFocusPointFilterDistance = new QDoubleSpinBox(this);
    auto* spinSurfaceZLowerBound = new QDoubleSpinBox(this);
    auto* spinSurfaceZUpperBound = new QDoubleSpinBox(this);
    cmbPointSetFilterMode->addItem("Any (OR)");
    cmbPointSetFilterMode->addItem("All (AND)");

    SurfacePanelController::FilterUiRefs filterUi;
    filterUi.dropdown = filterDropdown;
    filterUi.currentOnly = ui.chkFilterCurrentOnly;
    filterUi.pointSet = cmbPointSetFilter;
    filterUi.pointSetAll = btnPointSetFilterAll;
    filterUi.pointSetNone = btnPointSetFilterNone;
    filterUi.pointSetMode = cmbPointSetFilterMode;
    filterUi.surfaceIdFilter = ui.lineEditSurfaceFilter;
    filterUi.focusPointDistance = spinFocusPointFilterDistance;
    filterUi.zLowerBound = spinSurfaceZLowerBound;
    filterUi.zUpperBound = spinSurfaceZUpperBound;
    _surfacePanel->configureFilters(filterUi, _state->pointCollection());

    auto* tagDropdown = ui.btnTagDropdown;
    SurfacePanelController::TagUiRefs tagUi{
        .dropdown = tagDropdown,
        .approved = tagDropdown->addOption(tr("Approved"), QStringLiteral("chkApproved")),
        .defective = tagDropdown->addOption(tr("Defective"), QStringLiteral("chkDefective")),
        .reviewed = tagDropdown->addOption(tr("Reviewed"), QStringLiteral("chkReviewed")),
        .inspect = tagDropdown->addOption(tr("Inspect"), QStringLiteral("chkInspect")),
    };
    _surfacePanel->configureTags(tagUi);

    cmbSegmentationDir = ui.cmbSegmentationDir;
    _segmentDirModel = new QStandardItemModel(cmbSegmentationDir);
    cmbSegmentationDir->setModel(_segmentDirModel);
    cmbSegmentationDir->setView(
        new SegmentFolderListView([this](int row) { showSegmentFolderPaletteMenu(row); },
                                  cmbSegmentationDir));
    cmbSegmentationDir->view()->setItemDelegate(
        new SegmentFolderDelegate([this](int row) { showSegmentFolderPaletteMenu(row); },
                                  cmbSegmentationDir->view()));
    connect(_segmentDirModel, &QStandardItemModel::dataChanged,
            this, [this](const QModelIndex&, const QModelIndex&, const QVector<int>& roles) {
        if (_updatingSegmentDirUi) {
            return;
        }
        if (roles.isEmpty() || roles.contains(Qt::CheckStateRole)) {
            applySegmentFolderSelection(true);
        }
    });
    connect(cmbSegmentationDir, &QComboBox::currentIndexChanged, this, &CWindow::onSegmentationDirChanged);

    // Location input element (single QLineEdit for comma-separated values)
    lblLocFocus = ui.sliceFocus;

    // Set up validator for location input (accepts digits, commas, and spaces)
    QRegularExpressionValidator* validator = new QRegularExpressionValidator(
        QRegularExpression("^\\s*\\d+\\s*,\\s*\\d+\\s*,\\s*\\d+\\s*$"), this);
    lblLocFocus->setValidator(validator);
    connect(lblLocFocus, &QLineEdit::editingFinished, this, &CWindow::onManualLocationChanged);

    QPushButton* btnCopyCoords = ui.btnCopyCoords;
    connect(btnCopyCoords, &QPushButton::clicked, this, &CWindow::onCopyCoordinates);

    if (auto* chkAxisOverlays = ui.chkAxisOverlays) {
        bool showOverlays = settings.value(vc3d::settings::viewer::SHOW_AXIS_OVERLAYS,
                                           vc3d::settings::viewer::SHOW_AXIS_OVERLAYS_DEFAULT).toBool();
        QSignalBlocker blocker(chkAxisOverlays);
        chkAxisOverlays->setChecked(showOverlays);
        connect(chkAxisOverlays, &QCheckBox::toggled, this, &CWindow::onAxisOverlayVisibilityToggled);
    }
    if (auto* btnResetRot = ui.btnResetAxisRotations) {
        connect(btnResetRot, &QPushButton::clicked, this, &CWindow::onResetAxisAlignedRotations);
    }

    chkAxisAlignedSlices = ui.chkAxisAlignedSlices;
    if (chkAxisAlignedSlices) {
        bool useAxisAligned = settings.value(vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES,
                                             vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES_DEFAULT).toBool();
        QSignalBlocker blocker(chkAxisAlignedSlices);
        chkAxisAlignedSlices->setChecked(useAxisAligned);
        connect(chkAxisAlignedSlices, &QCheckBox::toggled, this, &CWindow::onAxisAlignedSlicesToggled);
    }

    if (chkAxisAlignedSlices) {
        onAxisAlignedSlicesToggled(chkAxisAlignedSlices->isChecked());
    }
    onAxisOverlayOpacityChanged(settings.value(vc3d::settings::viewer::AXIS_OVERLAY_OPACITY,
                                               vc3d::settings::viewer::AXIS_OVERLAY_OPACITY_DEFAULT).toInt());
    if (auto* chkAxisOverlays = ui.chkAxisOverlays) {
        onAxisOverlayVisibilityToggled(chkAxisOverlays->isChecked());
    }
    onMoveOnSurfaceChangedToggled(moveOnSurfaceChangeEnabled());

}

// Create menus
// Create actions
void CWindow::keyPressEvent(QKeyEvent* event)
{
    // Let fiber controller handle Escape first
    if (event->key() == Qt::Key_Escape && _fiberController && _fiberController->handleEscape()) {
        event->accept();
        return;
    }

    // Fiber animation (P key)
    if (_fiberController && _fiberController->handleKeyPress(event))
        return;

    if (event->key() == vc3d::keybinds::keypress::ToggleVolumeOverlay.key &&
        event->modifiers() == vc3d::keybinds::keypress::ToggleVolumeOverlay.modifiers) {
        toggleVolumeOverlayVisibility();
        event->accept();
        return;
    }

    if (event->key() == vc3d::keybinds::keypress::CenterFocusOnCursor.key &&
        event->modifiers() == vc3d::keybinds::keypress::CenterFocusOnCursor.modifiers) {
        if (centerFocusOnCursor()) {
            event->accept();
            return;
        }
    }

    if (event->key() == vc3d::keybinds::keypress::RecenterFocus.key &&
        event->modifiers() == vc3d::keybinds::keypress::RecenterFocus.modifiers) {
        if (recenterViewersOnCurrentFocus()) {
            event->accept();
            return;
        }
    }

    if (_viewerManager && _wrapAnnotationWidget && _wrapAnnotationWidget->sameWrapAnnotationEnabled()) {
        if (event->key() == Qt::Key_E && event->modifiers() == Qt::ShiftModifier) {
            bool committed = false;
            _viewerManager->forEachBaseViewer([&committed](VolumeViewerBase* baseViewer) {
                if (committed || !baseViewer) {
                    return;
                }
                auto* viewer = qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject());
                if (!viewer) {
                    return;
                }
                committed = viewer->commitSameWrapAnnotationPreview();
            });
            if (committed) {
                event->accept();
                return;
            }
        }
        if (event->key() == Qt::Key_Z && event->modifiers() == Qt::ControlModifier) {
            bool clearedPreview = false;
            _viewerManager->forEachBaseViewer([&clearedPreview](VolumeViewerBase* baseViewer) {
                if (!baseViewer) {
                    return;
                }
                if (auto* viewer = qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject())) {
                    if (viewer->hasSameWrapAnnotationPreview()) {
                        viewer->clearSameWrapAnnotationPreview();
                        clearedPreview = true;
                    }
                }
            });
            if (clearedPreview) {
                event->accept();
                return;
            }

            bool undone = false;
            _viewerManager->forEachBaseViewer([&undone](VolumeViewerBase* baseViewer) {
                if (undone || !baseViewer) {
                    return;
                }
                if (auto* viewer = qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject())) {
                    undone = viewer->undoSameWrapAnnotation();
                }
            });
            if (undone) {
                event->accept();
                return;
            }
        }
    }

    // Shift+G decreases Z-scroll sensitivity, Shift+H increases it. The
    // preference is global, so keep every workspace's viewers in sync.
    if (event->modifiers() == vc3d::keybinds::keypress::SliceStepDecrease.modifiers && _viewerManager) {
        if (event->key() == vc3d::keybinds::keypress::SliceStepDecrease.key) {
            const double sensitivity = _viewerManager->zScrollSensitivity() - 0.1;
            for (auto* manager : ViewerManager::allManagers()) {
                manager->setZScrollSensitivity(sensitivity);
            }
            event->accept();
            return;
        } else if (event->key() == vc3d::keybinds::keypress::SliceStepIncrease.key) {
            const double sensitivity = _viewerManager->zScrollSensitivity() + 0.1;
            for (auto* manager : ViewerManager::allManagers()) {
                manager->setZScrollSensitivity(sensitivity);
            }
            event->accept();
            return;
        }
    }

    if (_segmentationModule && _segmentationModule->handleKeyPress(event)) {
        return;
    }

    QMainWindow::keyPressEvent(event);
}

void CWindow::keyReleaseEvent(QKeyEvent* event)
{
    if (_viewerManager && _wrapAnnotationWidget &&
        _wrapAnnotationWidget->sameWrapAnnotationEnabled() &&
        event->key() == Qt::Key_Shift) {
        _viewerManager->forEachBaseViewer([event](VolumeViewerBase* baseViewer) {
            if (!baseViewer) {
                return;
            }
            if (auto* viewer = qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject())) {
                viewer->onKeyRelease(event->key(), event->modifiers());
            }
        });
    }

    if (_segmentationModule && _segmentationModule->handleKeyRelease(event)) {
        return;
    }

    QMainWindow::keyReleaseEvent(event);
}

void CWindow::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);
    scheduleWindowStateSave();
}

void CWindow::scheduleWindowStateSave()
{
    // Restart the timer - this debounces rapid changes
    if (_windowStateSaveTimer) {
        _windowStateSaveTimer->start();
    }
}

void CWindow::saveWindowState()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::window::GEOMETRY, saveGeometry());
    if (_segmentWorkspaceWindow) {
        settings.setValue(vc3d::settings::window::STATE, _segmentWorkspaceWindow->saveState());
    }
    if (_workspaceTabs) {
        settings.setValue(WORKSPACE_TAB_SETTING, _workspaceTabs->currentIndex());
        if (auto* workspace = _workspaceTabs->currentWidget())
            settings.setValue(WORKSPACE_ID_SETTING, workspace->property("workspaceId").toString());
    }
    settings.setValue(vc3d::settings::window::RESTORE_DISABLED, false);
    settings.setValue(vc3d::settings::window::RESTORE_IN_PROGRESS, false);
    writeWindowStateMeta(settings,
                         windowStateScreenSignature(),
                         windowStateQtVersion(),
                         windowStateAppVersion());
    settings.sync();
}

void CWindow::closeEvent(QCloseEvent* event)
{
    _destroyingWindow = true;
    // Flush a render-bench recording (if any) before teardown.
    if (_benchRecorder && _benchRecorder->attached()) {
        _benchRecorder->save();
    }
    // Tell ViewerManager to stop maintaining the SurfacePatchIndex. The
    // CState teardown below iterates every tracked surface and sets it to
    // nullptr, which would otherwise trigger an O(N) rtree->remove() per
    // surface — easily 10+ seconds on a flattened segment with millions
    // of cells.
    if (_viewerManager) {
        _viewerManager->beginShutdown();
    }
    // Flush any pending debounced approval-mask save before teardown so the
    // last few seconds of approvals aren't lost on exit. No-op if nothing
    // is pending.
    if (_segmentationModule && _segmentationModule->overlay()) {
        _segmentationModule->overlay()->flushPendingApprovalMaskSave();
    }
    if (_lineAnnotationController) {
        _lineAnnotationController->saveOpenFibers();
    }
    if (_state && _state->vpkg()) {
        try { _state->vpkg()->saveAutosave(); } catch (...) {}
    }
    saveWindowState();
    event->accept();
}

void CWindow::setWidgetsEnabled(bool state)
{
    ui.grpVolManager->setEnabled(state);
    ui.btnOpenDataCatalog->setEnabled(!state);
    if (_viewerControlsPanel) {
        _viewerControlsPanel->setViewControlsEnabled(state);
        _viewerControlsPanel->setOverlayWindowAvailable(_volumeOverlay && _volumeOverlay->hasOverlaySelection());
    }
}

// Update the widgets
void CWindow::UpdateView(void)
{
    if (!_state->hasVpkg() && _state->currentVolume() == nullptr) {
        setWidgetsEnabled(false);  // Disable Widgets for User
        ui.lblVpkgName->setText("[ No Volume Package Loaded ]");
        updateVolumePackageEmptyState();
        return;
    }

    setWidgetsEnabled(true);  // Enable Widgets for User
    updateVolumePackageEmptyState();

    // show volume package name
    UpdateVolpkgLabel(0);

    syncVolumeSelectionControls();

    update();
}

void CWindow::updateVolumePackageEmptyState()
{
    const bool hasOpenPackage = _state && (_state->hasVpkg() || _state->currentVolume() != nullptr);
    const bool catalogVisible = _menuController && _menuController->isOpenDataCatalogVisible();
    ui.btnOpenDataCatalog->setVisible(!hasOpenPackage && !catalogVisible);
}

void CWindow::UpdateVolpkgLabel(int filterCounter)
{
    if (_state->vpkg()) {
        QString label = tr("%1").arg(QString::fromStdString(_state->vpkg()->name()));
        ui.lblVpkgName->setText(label);
    } else if (_state->currentVolume()) {
        QString label = tr("Remote: %1").arg(QString::fromStdString(_state->currentVolumeId()));
        ui.lblVpkgName->setText(label);
    }
}

void CWindow::onShowStatusMessage(QString text, int timeout)
{
    showStatusBarMessage(text, timeout);
}

void CWindow::showStatusBarMessage(const QString& text, int timeout)
{
    if (!_statusMessageLabel) {
        if (statusBar()) {
            statusBar()->showMessage(text, timeout);
        }
        return;
    }

    _statusMessageLabel->setText(text);
    _statusMessageLabel->setToolTip(text);

    if (_statusMessageTimer) {
        _statusMessageTimer->stop();
        if (timeout > 0) {
            _statusMessageTimer->start(timeout);
        }
    }
}

void CWindow::clearStatusBarMessage()
{
    if (_statusMessageTimer) {
        _statusMessageTimer->stop();
    }
    if (_statusMessageLabel) {
        _statusMessageLabel->clear();
        _statusMessageLabel->setToolTip(QString());
    } else if (statusBar()) {
        statusBar()->clearMessage();
    }
}

void CWindow::onSegmentationGrowthStatusChanged(bool running)
{
    if (!statusBar()) {
        return;
    }

    if (_surfacePanel) {
        _surfacePanel->setSelectionLocked(running);
    }

    if (running) {
        if (!_segmentationGrowthWarning) {
            _segmentationGrowthWarning = new QLabel(statusBar());
            _segmentationGrowthWarning->setObjectName(QStringLiteral("segmentationGrowthWarning"));
            _segmentationGrowthWarning->setStyleSheet(QStringLiteral("color: #c62828; font-weight: 600;"));
            _segmentationGrowthWarning->setContentsMargins(8, 0, 8, 0);
            _segmentationGrowthWarning->setAlignment(Qt::AlignCenter);
            _segmentationGrowthWarning->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
            _segmentationGrowthWarning->setMinimumWidth(260);
            _segmentationGrowthWarning->hide();
            statusBar()->addPermanentWidget(_segmentationGrowthWarning, 1);
        }
        _segmentationGrowthStatusText = tr("Surface growth in progress - surface selection locked");
        _segmentationGrowthWarning->setText(_segmentationGrowthStatusText);
        _segmentationGrowthWarning->setVisible(true);
        showStatusBarMessage(_segmentationGrowthStatusText, 0);
    } else if (_segmentationGrowthWarning) {
        _segmentationGrowthWarning->clear();
        _segmentationGrowthWarning->setVisible(false);
        if (_statusMessageLabel && _statusMessageLabel->text() == _segmentationGrowthStatusText) {
            clearStatusBarMessage();
        } else if (statusBar()->currentMessage() == _segmentationGrowthStatusText) {
            clearStatusBarMessage();
        }
        _segmentationGrowthStatusText.clear();
    }
}

void CWindow::onZScrollSensitivityChanged(double sensitivity)
{
    // Update status bar label. Persistence + viewer refresh happen in
    // ViewerManager::setZScrollSensitivity (the single source of truth).
    if (_sliceStepLabel) {
        _sliceStepLabel->setText(tr("Z sens: %1").arg(sensitivity, 0, 'f', 1));
    }
}

void CWindow::onSharedCacheStatsChanged(const QStringList& items)
{
    if (!_sharedCacheStatsLabel || items.isEmpty()) {
        return;
    }
    _sharedCacheStatsLabel->setText(items.join(QStringLiteral("  ")));
}

// Open volume package
bool CWindow::OpenVolume(const QString& path,
                         bool interactive,
                         QString* errorMessage,
                         const QString& preferredVolumeId,
                         VolumeOpenError* openError)
{
    if (errorMessage) {
        errorMessage->clear();
    }
    if (openError) {
        *openError = VolumeOpenError::None;
    }
    QString aVpkgPath = path;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    if (aVpkgPath.isEmpty()) {
        if (!interactive) {
            if (errorMessage) {
                *errorMessage = tr("Project path is empty.");
            }
            if (openError) {
                *openError = VolumeOpenError::PackageLoadFailed;
            }
            return false;
        }
        aVpkgPath = QFileDialog::getOpenFileName(
            this, tr("Open Project"), settings.value(vc3d::settings::project::DEFAULT_PATH).toString(),
            tr("Project (*.volpkg.json);;All files (*.*)"),
            nullptr, QFileDialog::DontResolveSymlinks | QFileDialog::ReadOnly | QFileDialog::DontUseNativeDialog);
        if (aVpkgPath.isEmpty()) {
            Logger()->info("Open project canceled");
            return false;
        }
    }

    std::shared_ptr<VolumePkg> package;
    QString loadError;
    try {
        package = VolumePkg::load(aVpkgPath.toStdString());
    } catch (const std::exception& e) {
        Logger()->error("Failed to initialize volpkg: {}", e.what());
        loadError = QString::fromUtf8(e.what());
    }

    if (!package) {
        Logger()->error("Cannot open project: {}", aVpkgPath.toStdString());
        const QString message = loadError.isEmpty()
            ? tr("Project failed to load.")
            : tr("Project failed to load: %1").arg(loadError);
        if (errorMessage) {
            *errorMessage = message;
        }
        if (openError) {
            *openError = VolumeOpenError::PackageLoadFailed;
        }
        if (interactive) {
            QMessageBox::warning(this, tr("Error"), message);
        }
        return false;
    }

    // Check version number
    if (package->version() < VOLPKG_MIN_VERSION) {
        const auto msg = "Volume package is version " +
                         std::to_string(package->version()) +
                         " but this program requires version " +
                         std::to_string(VOLPKG_MIN_VERSION) + "+.";
        Logger()->error(msg);
        if (errorMessage) {
            *errorMessage = QString::fromStdString(msg);
        }
        if (interactive) {
            QMessageBox::warning(this, tr("ERROR"), QString::fromStdString(msg));
        }
        if (openError) {
            *openError = VolumeOpenError::PackageLoadFailed;
        }
        return false;
    }

    if (!preferredVolumeId.isEmpty()) {
        const auto ids = package->volumeIDs();
        const auto id = preferredVolumeId.toStdString();
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
            if (errorMessage) {
                *errorMessage = tr("Unknown volume id: %1").arg(preferredVolumeId);
            }
            if (openError) {
                *openError = VolumeOpenError::VolumeNotFound;
            }
            return false;
        }
        try {
            (void)package->volume(id);
        } catch (const std::exception& e) {
            if (errorMessage) {
                *errorMessage = tr("Failed to load volume %1: %2")
                                    .arg(preferredVolumeId, QString::fromUtf8(e.what()));
            }
            if (openError) {
                *openError = VolumeOpenError::PackageLoadFailed;
            }
            return false;
        }
    }

    CloseVolume();
    _state->setVpkg(std::move(package));
    refreshCurrentVolumePackageUi(preferredVolumeId, true);
    if (_menuController) {
        _menuController->updateRecentVolpkgList(aVpkgPath);
    }

    if (_fileWatcher) {
        _fileWatcher->startWatching();
    }
    return true;
}

bool CWindow::openVolumePackage(const QString& path,
                                bool interactive,
                                QString* errorMessage,
                                const QString& preferredVolumeId,
                                VolumeOpenError* openError)
{
    const bool opened = OpenVolume(
        path, interactive, errorMessage, preferredVolumeId, openError);
    UpdateView();
    return opened;
}

std::vector<QComboBox*> CWindow::volumeSelectionControls() const
{
    std::vector<QComboBox*> selectors;
    if (volSelect) {
        selectors.push_back(volSelect);
    }
    for (const auto& selector : _annotationVolumeSelects) {
        if (selector) {
            selectors.push_back(selector.data());
        }
    }
    return selectors;
}

void CWindow::connectVolumeSelector(QComboBox* selector)
{
    if (!selector) {
        return;
    }
    connect(selector,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            [this, selector](const int& index) {
                auto vpkg = _state->vpkg();
                if (vpkg && index >= 0) {
                    const QString volumeId = selector->currentData().toString();
                    std::shared_ptr<Volume> newVolume;
                    try {
                        newVolume = vpkg->volume(volumeId.toStdString());
                    } catch (const std::out_of_range&) {
                        QMessageBox::warning(this, "Error", "Could not load volume.");
                        syncVolumeSelectionControls();
                        return;
                    }
                    setVolume(newVolume);
                    syncVolumeSelectionControls(volumeId);
                }
            });
}

QString CWindow::lastVolumeSettingKeyForCurrentPackage() const
{
    if (!_state || !_state->vpkg()) {
        return {};
    }

    QString projectPath = QString::fromStdString(_state->vpkg()->path().string());
    if (projectPath.isEmpty()) {
        return {};
    }

    const QFileInfo info(projectPath);
    QString stablePath = info.canonicalFilePath();
    if (stablePath.isEmpty()) {
        stablePath = info.absoluteFilePath();
    }
    if (stablePath.isEmpty()) {
        stablePath = projectPath;
    }

    return QStringLiteral("%1%2").arg(
        QString::fromLatin1(vc3d::settings::project::LAST_VOLUME_PREFIX),
        QString::fromLatin1(QUrl::toPercentEncoding(stablePath)));
}

QString CWindow::rememberedVolumeIdForCurrentPackage() const
{
    const QString key = lastVolumeSettingKeyForCurrentPackage();
    if (key.isEmpty()) {
        return {};
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    return settings.value(key).toString();
}

void CWindow::rememberCurrentVolumeForPackage(const QString& volumeId) const
{
    if (volumeId.isEmpty() || !_state || !_state->vpkg() ||
        !_state->vpkg()->hasVolume(volumeId.toStdString())) {
        return;
    }

    const QString key = lastVolumeSettingKeyForCurrentPackage();
    if (key.isEmpty()) {
        return;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(key, volumeId);
}

const vc3d::opendata::OpenDataManifest* CWindow::cachedOpenDataManifest() const
{
    if (_openDataManifestCache) {
        return &*_openDataManifestCache;
    }
    if (_openDataManifestLoadAttempted) {
        return nullptr;
    }
    _openDataManifestLoadAttempted = true;

    const auto manifestPath = openDataCatalogManifestCachePath();
    if (manifestPath.empty() || !std::filesystem::is_regular_file(manifestPath)) {
        return nullptr;
    }

    try {
        _openDataManifestCache = vc3d::opendata::loadOpenDataManifestFile(
            manifestPath,
            std::string(vc3d::opendata::kDefaultManifestUrl));
        return &*_openDataManifestCache;
    } catch (const std::exception& ex) {
        Logger()->warn("Failed to load cached open-data manifest '{}': {}",
                       manifestPath.string(), ex.what());
    } catch (...) {
        Logger()->warn("Failed to load cached open-data manifest '{}': unknown error",
                       manifestPath.string());
    }

    return nullptr;
}

std::string CWindow::openDataVolumeIdForLoadedVolumeId(const std::string& volumeId) const
{
    if (volumeId.empty() || !_state || !_state->vpkg()) {
        return {};
    }

    for (const auto& tag : _state->vpkg()->volumeTags(volumeId)) {
        if (tag.rfind(OPEN_DATA_VOLUME_ID_TAG_PREFIX, 0) != 0) {
            continue;
        }
        return tag.substr(OPEN_DATA_VOLUME_ID_TAG_PREFIX.size());
    }

    return {};
}

std::optional<cv::Matx44d> CWindow::openDataVolumeTransformForSwitch(
    const std::string& fromLoadedVolumeId,
    const std::string& toLoadedVolumeId) const
{
    const std::string fromCatalogId = openDataVolumeIdForLoadedVolumeId(fromLoadedVolumeId);
    const std::string toCatalogId = openDataVolumeIdForLoadedVolumeId(toLoadedVolumeId);
    if (fromCatalogId.empty() || toCatalogId.empty()) {
        return std::nullopt;
    }

    const auto& pkg = *_state->vpkg();
    const auto fromLevel = openDataCoordinateLevelForLoadedVolume(pkg, fromLoadedVolumeId);
    const auto toLevel = openDataCoordinateLevelForLoadedVolume(pkg, toLoadedVolumeId);
    const auto fromSpace = openDataCoordinateSpaceForLoadedVolume(pkg, fromLoadedVolumeId);
    const auto toSpace = openDataCoordinateSpaceForLoadedVolume(pkg, toLoadedVolumeId);
    const bool explicitCoordinates = fromLevel && toLevel &&
                                     !fromSpace.empty() && !toSpace.empty();
    if (explicitCoordinates) {
        if (fromSpace == toSpace)
            return cv::Matx44d::eye();
        if (fromCatalogId == toCatalogId) {
            const double factor = std::ldexp(1.0, *fromLevel - *toLevel);
            return cv::Matx44d(
                factor, 0, 0, 0,
                0, factor, 0, 0,
                0, 0, factor, 0,
                0, 0, 0, 1);
        }
    } else if (fromCatalogId == toCatalogId) {
        // Preserve legacy/manual same-lineage behavior when explicit
        // coordinate identity is unavailable.
        return std::nullopt;
    }

    const auto* manifest = cachedOpenDataManifest();
    if (!manifest) {
        return std::nullopt;
    }

    for (const auto& sample : manifest->samples) {
        if (auto matrix = vc3d::opendata::findSampleVolumeTransform(
                sample, fromCatalogId, toCatalogId)) {
            if (!explicitCoordinates)
                return matrix;
            const auto sourceScale = coordinateLevelScale(*fromLevel);
            const auto targetScaleInv = coordinateLevelScale(-*toLevel);
            return targetScaleInv * *matrix * sourceScale;
        }
    }

    return std::nullopt;
}

void CWindow::updateOpenDataSegmentTransformState(bool showDialog)
{
    if (!_state || !_state->vpkg()) {
        return;
    }

    auto vpkg = _state->vpkg();
    const std::string loadedVolumeId = _state->currentVolumeId();
    QString catalogVolumeId = openDataCatalogVolumeIdForLoadedVolume(*vpkg, loadedVolumeId);
    const bool hasOpenDataSegments = packageHasOpenDataSegments(*vpkg);
    const auto* matchingEntry = findOpenDataSegmentsEntryForLoadedVolume(
        *vpkg,
        loadedVolumeId,
        &catalogVolumeId);

    auto setWarning = [&](bool enabled) {
        const QString warningText = tr("Current segments have no available transforms to selected volume.");
        if (!_segmentTransformWarning && statusBar()) {
            _segmentTransformWarning = new QLabel(statusBar());
            _segmentTransformWarning->setObjectName(QStringLiteral("segmentTransformWarning"));
            _segmentTransformWarning->setStyleSheet(QStringLiteral("color: #c62828; font-weight: 600;"));
            _segmentTransformWarning->setContentsMargins(8, 0, 8, 0);
            _segmentTransformWarning->setAlignment(Qt::AlignCenter);
            _segmentTransformWarning->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
            _segmentTransformWarning->setMinimumWidth(320);
            _segmentTransformWarning->hide();
            statusBar()->addPermanentWidget(_segmentTransformWarning, 1);
        }

        if (_surfacePanel) {
            _surfacePanel->setTransformWarning(enabled ? warningText : QString());
        }
        if (_segmentTransformWarning) {
            _segmentTransformWarning->setText(enabled ? warningText : QString());
            _segmentTransformWarning->setVisible(enabled);
        }
        if (enabled) {
            showStatusBarMessage(warningText, 0);
        } else if (_statusMessageLabel && _statusMessageLabel->text() == warningText) {
            clearStatusBarMessage();
        }
    };

    if (!hasOpenDataSegments || loadedVolumeId.empty()) {
        setWarning(false);
        _lastSegmentTransformWarningVolumeId.clear();
        return;
    }

    if (matchingEntry) {
        const auto currentPath = vpkg->outputSegmentsPath().lexically_normal();
        const auto targetPath = vc::project::resolveLocalPath(
            matchingEntry->location,
            vpkg->path().parent_path()).lexically_normal();
        if (currentPath != targetPath) {
            const std::string previousActiveSurfaceId = _state->activeSurfaceId();
            vpkg->setOutputSegments(matchingEntry->location);
            vpkg->refreshSegmentations();
            refreshSegmentationDirectoryDropdown();
            if (_surfacePanel) {
                _surfacePanel->setVolumePkg(vpkg);
                _surfacePanel->loadSurfaces(true);
                _surfacePanel->refreshPointSetFilterOptions();
            }
            if (!restoreActiveSurfaceAfterSurfaceReload(previousActiveSurfaceId)) {
                clearSurfaceSelection();
                _state->setSurface("segmentation", nullptr, true);
            }
        }
        setWarning(false);
        _lastSegmentTransformWarningVolumeId.clear();
        return;
    }

    setWarning(true);
    const QString warningVolumeId = catalogVolumeId.isEmpty()
        ? QString::fromStdString(loadedVolumeId)
        : catalogVolumeId;
    if (showDialog && _lastSegmentTransformWarningVolumeId != warningVolumeId) {
        _lastSegmentTransformWarningVolumeId = warningVolumeId;
        QMessageBox::information(
            this,
            tr("Segments unavailable"),
            tr("Current segments have no available transforms to selected volume."));
    }
}

QWidget* CWindow::createAnnotationVolumeSelector(QWidget* parent)
{
    auto* volumeSelector = new VolumeSelector(parent);
    volumeSelector->setLabelVisible(false);
    volumeSelector->setBrowseEnabled(false);
    if (auto* combo = volumeSelector->comboBox()) {
        combo->setObjectName(QStringLiteral("annotationVolumeSelect"));
        combo->setMinimumWidth(130);
        combo->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        _annotationVolumeSelects.push_back(QPointer<QComboBox>(combo));
        connectVolumeSelector(combo);
    }
    refreshVolumeSelectionUi();
    return volumeSelector;
}

void CWindow::syncVolumeSelectionControls(const QString& activeVolumeId)
{
    QString targetId = activeVolumeId;
    if (targetId.isEmpty() && _state) {
        targetId = QString::fromStdString(_state->currentVolumeId());
    }

    for (QComboBox* selector : volumeSelectionControls()) {
        if (!selector) {
            continue;
        }
        const int index = targetId.isEmpty() ? -1 : selector->findData(targetId);
        if (index >= 0 && selector->currentIndex() != index) {
            const QSignalBlocker blocker(selector);
            selector->setCurrentIndex(index);
        }
        selector->setEnabled(can_change_volume_());
    }
}

void CWindow::refreshVolumeSelectionUi(const QString& preferredVolumeId)
{
    const auto selectors = volumeSelectionControls();
    if (selectors.empty() || !_state || !_state->vpkg()) {
        return;
    }

    QVector<QPair<QString, QString>> volumeEntries;
    QVector<QPair<QString, QString>> openDataVolumeIdMap;
    std::set<QString> preferredOpenDataSourceIds;
    std::set<QString> openDataVolumesWithoutSegments;
    std::vector<QString> orderedIds;
    QString activeCandidate = preferredVolumeId;
    const bool hasExplicitPreferredVolume = !activeCandidate.isEmpty();
    QString currentComboId;
    for (QComboBox* selector : selectors) {
        if (selector && selector->currentIndex() >= 0) {
            currentComboId = selector->currentData().toString();
            if (!currentComboId.isEmpty()) {
                break;
            }
        }
    }
    const QString currentVolumeId = QString::fromStdString(_state->currentVolumeId());

    auto hasVolume = [&](const QString& volumeId) {
        for (const auto& id : orderedIds) {
            if (id == volumeId) {
                return true;
            }
        }
        return false;
    };

    auto openDataVolumeIdMappedToLoadedId = [&](const QString& catalogVolumeId) {
        for (const auto& mapping : openDataVolumeIdMap) {
            if (mapping.first == catalogVolumeId) {
                return mapping.second;
            }
        }
        return QString{};
    };

    QString bestGrowthVolumeId;
    bool preferredVolumeFound = false;
    const bool hasOpenDataSegments = packageHasOpenDataSegments(*_state->vpkg());
    const auto volumeIds = _state->vpkg()->volumeIDs();
    for (const auto& id : volumeIds) {
        try {
            auto vol = _state->vpkg()->volume(id);
            const QString idStr = QString::fromStdString(id);
            const QString nameStr = QString::fromStdString(vol->name());
            const QString label = nameStr.isEmpty() ? idStr : QStringLiteral("%1 (%2)").arg(nameStr, idStr);

            orderedIds.push_back(idStr);
            volumeEntries.append({idStr, label});
            if (hasOpenDataSegments && !volumeHasOpenDataSegmentsEntry(*_state->vpkg(), id)) {
                openDataVolumesWithoutSegments.insert(idStr);
            }
            const auto loadedVolumeTags = _state->vpkg()->volumeTags(id);
            const bool preferredOpenDataSource =
                std::find(loadedVolumeTags.begin(), loadedVolumeTags.end(),
                          "vc-open-data-preferred-source") != loadedVolumeTags.end();
            for (const auto& tag : loadedVolumeTags) {
                constexpr std::string_view prefix = "vc-open-data-volume-id:";
                if (tag.rfind(prefix, 0) != 0) {
                    continue;
                }
                const QString catalogVolumeId =
                    QString::fromStdString(tag.substr(prefix.size()));
                if (!catalogVolumeId.isEmpty()) {
                    const auto existing = std::find_if(
                        openDataVolumeIdMap.begin(), openDataVolumeIdMap.end(),
                        [&](const auto& mapping) {
                            return mapping.first == catalogVolumeId;
                        });
                    if (existing == openDataVolumeIdMap.end()) {
                        openDataVolumeIdMap.append({catalogVolumeId, idStr});
                    } else if (preferredOpenDataSource &&
                               !preferredOpenDataSourceIds.contains(catalogVolumeId)) {
                        existing->second = idStr;
                    }
                    if (preferredOpenDataSource)
                        preferredOpenDataSourceIds.insert(catalogVolumeId);
                }
            }

            const QString loweredName = nameStr.toLower();
            const QString loweredId = idStr.toLower();
            const bool matchesPreferred = loweredName.contains(QStringLiteral("surface")) ||
                                          loweredName.contains(QStringLiteral("surf")) ||
                                          loweredId.contains(QStringLiteral("surface")) ||
                                          loweredId.contains(QStringLiteral("surf"));

            if (!preferredVolumeFound && matchesPreferred) {
                bestGrowthVolumeId = idStr;
                preferredVolumeFound = true;
            }
        } catch (...) {
            continue;
        }
    }

    if (bestGrowthVolumeId.isEmpty() && !volumeEntries.isEmpty()) {
        bestGrowthVolumeId = orderedIds.front();
    }

    if (!activeCandidate.isEmpty() && !hasVolume(activeCandidate)) {
        activeCandidate = openDataVolumeIdMappedToLoadedId(activeCandidate);
    }
    if (activeCandidate.isEmpty() && !hasExplicitPreferredVolume) {
        const QString rememberedVolumeId = rememberedVolumeIdForCurrentPackage();
        if (!rememberedVolumeId.isEmpty() && hasVolume(rememberedVolumeId)) {
            activeCandidate = rememberedVolumeId;
        }
    }
    if (activeCandidate.isEmpty() && !currentComboId.isEmpty() && hasVolume(currentComboId)) {
        activeCandidate = currentComboId;
    }
    if (activeCandidate.isEmpty() && !currentVolumeId.isEmpty() && hasVolume(currentVolumeId)) {
        activeCandidate = currentVolumeId;
    }
    if (activeCandidate.isEmpty() && !volumeEntries.isEmpty()) {
        activeCandidate = orderedIds.front();
    }

    for (QComboBox* selector : selectors) {
        if (!selector) {
            continue;
        }
        const QSignalBlocker blocker{selector};
        selector->clear();
        for (const auto& [id, label] : volumeEntries) {
            selector->addItem(label, QVariant(id));
            const int row = selector->count() - 1;
            if (openDataVolumesWithoutSegments.find(id) != openDataVolumesWithoutSegments.end()) {
                selector->setItemData(row, QBrush(QColor(245, 124, 0)), Qt::ForegroundRole);
                selector->setItemData(row,
                                      tr("No segment transform is available for this volume."),
                                      Qt::ToolTipRole);
            }
        }
        if (activeCandidate.isEmpty()) {
            if (selector->count() > 0) {
                selector->setCurrentIndex(0);
            }
        } else {
            const int activeIndex = selector->findData(activeCandidate);
            selector->setCurrentIndex(activeIndex >= 0 ? activeIndex : 0);
        }
    }

    QString activeId;
    for (QComboBox* selector : selectors) {
        if (selector && selector->count() > 0) {
            activeId = selector->currentData().toString();
            break;
        }
    }
    syncVolumeSelectionControls(activeId);

    QString growthVolumeId = QString::fromStdString(_state->segmentationGrowthVolumeId());
    if (!growthVolumeId.isEmpty() && !hasVolume(growthVolumeId)) {
        growthVolumeId.clear();
    }
    if (growthVolumeId.isEmpty()) {
        growthVolumeId = bestGrowthVolumeId;
    }
    if (growthVolumeId.isEmpty()) {
        growthVolumeId = activeId;
    }

    if (!activeId.isEmpty()) {
        if (!_state->currentVolume() || _state->currentVolumeId() != activeId.toStdString()) {
            try {
                auto newVolume = _state->vpkg()->volume(activeId.toStdString());
                setVolume(newVolume);
            } catch (...) {
                // Ignore errors - keep existing volume selection if invalid.
            }
        }

        _state->setSegmentationGrowthVolumeId(growthVolumeId.toStdString());

        if (_segmentationWidget) {
            _segmentationWidget->setAvailableVolumes(volumeEntries, growthVolumeId);
            if (!growthVolumeId.isEmpty()) {
                _segmentationWidget->setActiveVolume(growthVolumeId);
            }
            try {
                auto growthVolume = _state->vpkg()->volume(growthVolumeId.toStdString());
                if (growthVolume) {
                    _segmentationWidget->setVolumeZarrPath(QString::fromStdString(growthVolume->path().string()));
                }
            } catch (...) {
                // Ignore errors - neural growth path update is non-critical.
            }
        }
    } else {
        setVolume(nullptr);
        _state->setSegmentationGrowthVolumeId({});
        if (_segmentationWidget) {
            _segmentationWidget->setAvailableVolumes(QVector<QPair<QString, QString>>{}, {});
            _segmentationWidget->setActiveVolume({});
            _segmentationWidget->setVolumeZarrPath({});
        }
    }
}

void CWindow::CloseVolume(void)
{
    if (_fileWatcher) {
        _fileWatcher->stopWatching();
    }

    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->clearPreview(false);
    }

    // Tear down active segmentation editing before surfaces disappear to avoid
    // dangling pointers inside the edit manager when the underlying surfaces
    // are unloaded (reloading with editing enabled previously triggered a
    // use-after-free crash).
    if (_segmentationModule) {
        if (_segmentationModule->editingEnabled()) {
            _segmentationModule->setEditingEnabled(false);
        } else if (_segmentationModule->hasActiveSession()) {
            _segmentationModule->endEditingSession();
        }
    }

    // CState::closeAll emits volumeClosing, clears surfaces, vpkg, volume, points
    _state->closeAll();

    updateNormalGridAvailability();
    if (_segmentationWidget) {
        _segmentationWidget->setAvailableVolumes({}, QString());
        _segmentationWidget->setVolumePackagePath(QString());
    }

    if (_surfacePanel) {
        _surfacePanel->clear();
        _surfacePanel->setVolumePkg(nullptr);
        _surfacePanel->resetTagUi();
    }

    // Update UI
    UpdateView();
    if (treeWidgetSurfaces) {
        treeWidgetSurfaces->clear();
    }

    if (_volumeOverlay) {
        _volumeOverlay->clearVolumePkg();
    }
    if (_viewerManager) {
        _viewerManager->setOverlayVolume(nullptr, {});
    }
    if (_spiralWorkspace && _spiralWorkspace->viewerManager()) {
        _spiralWorkspace->viewerManager()->setOverlayVolume(nullptr, {});
    }

    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }
}

// Handle open request
auto CWindow::can_change_volume_() -> bool
{
    if (_state->hasVpkg() && _state->vpkg()->numberOfVolumes() > 1) {
        return true;
    }
    // Also allow switching when volSelect has multiple remote volumes
    if (_state && _state->currentVolume()) {
        for (QComboBox* selector : volumeSelectionControls()) {
            if (selector && selector->count() > 1) {
                return true;
            }
        }
    }
    return false;
}

void CWindow::onSurfaceActivated(const QString& surfaceId, QuadSurface* surface)
{
    const std::string previousSurfId = _state->activeSurfaceId();
    const std::string newSurfId = surfaceId.toStdString();

    // Look up the shared_ptr by display ID. Multi-folder entries are keyed only
    // in CState, while current-folder entries are still owned by VolumePkg.
    if (_state && !newSurfId.empty()) {
        auto stateSurface = std::dynamic_pointer_cast<QuadSurface>(_state->surface(newSurfId));
        if (!stateSurface && _state->vpkg()) {
            stateSurface = _state->vpkg()->getSurface(newSurfId);
        }
        _state->setActiveSurface(newSurfId, stateSurface);
    } else {
        _state->clearActiveSurface();
    }

    auto surf = _state->activeSurface().lock();

    _state->setSurface("segmentation", surf, false, false);

    if (newSurfId != previousSurfId) {
        if (_segmentationModule && _segmentationModule->editingEnabled()) {
            _segmentationModule->setEditingEnabled(false);
        } else if (_segmentationWidget && _segmentationWidget->isEditingEnabled()) {
            _segmentationWidget->setEditingEnabled(false);
        }

        if (_segmentationModule) {
            try {
                _segmentationModule->onActiveSegmentChanged(surf.get());
            } catch (const std::exception& e) {
                qWarning() << "Failed to activate surface"
                           << surfaceId
                           << "while it may still be writing:"
                           << e.what();
                _state->clearActiveSurface();
                _state->setSurface("segmentation", nullptr, false, false);
                surf.reset();
            }
        }

        // Load corr_points_results for the new segment
        if (_point_collection_widget) {
            auto quadSurf = std::dynamic_pointer_cast<QuadSurface>(surf);
            if (quadSurf && !quadSurf->path.empty()) {
                _point_collection_widget->loadCorrPointsResults(
                    quadSurf->path / "corr_points_results.json");
            } else {
                _point_collection_widget->clearCorrPointsResults();
            }
        }
        if (_atlasControlDock) {
            auto quadSurf = std::dynamic_pointer_cast<QuadSurface>(surf);
            if (quadSurf && !quadSurf->path.empty()) {
                _atlasControlDock->loadResults(quadSurf->path / "atlas_control_points_results.json");
            } else {
                _atlasControlDock->clearResults();
            }
        }
    }

    const bool moveOnSurfaceChange = moveOnSurfaceChangeEnabled();
    const bool activatingAxisAlignedPlane =
        newSurfId == "xy plane" || newSurfId == "seg xz" || newSurfId == "seg yz";
    if (moveOnSurfaceChange && _axisAlignedSliceController && !activatingAxisAlignedPlane) {
        _axisAlignedSliceController->resetAll();
    }

    if (moveOnSurfaceChange) {
        if (auto quadSurf = std::dynamic_pointer_cast<QuadSurface>(surf)) {
            try {
                quadSurf->ensureLoaded();
                const cv::Vec3f worldCenter = quadSurf->coord({0, 0, 0}, {0, 0, 0});
                const bool centerValid = std::isfinite(worldCenter[0])
                    && std::isfinite(worldCenter[1])
                    && std::isfinite(worldCenter[2])
                    && worldCenter[0] >= 0.0f;
                if (centerValid) {
                    if (auto vol = _state->currentVolume()) {
                        auto [w, h, d] = vol->shapeXyz();
                        cv::Vec3f clamped = worldCenter;
                        clamped[0] = std::clamp(clamped[0], 0.0f, static_cast<float>(w - 1));
                        clamped[1] = std::clamp(clamped[1], 0.0f, static_cast<float>(h - 1));
                        clamped[2] = std::clamp(clamped[2], 0.0f, static_cast<float>(d - 1));
                        POI* poi = new POI;
                        poi->p = clamped;
                        poi->n = cv::Vec3f(0, 0, 0);
                        poi->surfaceId = newSurfId;
                        _state->setPOI("focus", poi);
                    }
                }
            } catch (const std::exception& e) {
                qWarning() << "Could not compute world center for"
                           << surfaceId << ":" << e.what();
            }
        }
    }

    if (moveOnSurfaceChange || activatingAxisAlignedPlane) {
        try {
            if (surf) {
                _axisAlignedSliceController->applyOrientation(surf.get());
            } else {
                _axisAlignedSliceController->applyOrientation();
            }
        } catch (const std::exception& e) {
            qWarning() << "Failed to apply surface orientation for"
                       << surfaceId
                       << "while it may still be writing:"
                       << e.what();
            _state->clearActiveSurface();
            _state->setSurface("segmentation", nullptr, false, false);
            _axisAlignedSliceController->applyOrientation();
        }
    }

    if (_surfacePanel && _surfacePanel->isCurrentOnlyFilterEnabled()) {
        _surfacePanel->refreshFiltersOnly();
    }

    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }

    maybeAttachBenchRecorder();
}

void CWindow::maybeAttachBenchRecorder()
{
    if (!_benchRecorder || _benchRecorder->attached())
        return;
    auto* viewer = segmentationViewer();
    if (!viewer || !viewer->currentVolume() || !viewer->currentSurface())
        return;

    RenderBenchRecorder::Header h;
    // The replay driver reopens via OpenVolume(), which wants the .volpkg.json
    // file path (vpkg()->path()), not the parent directory vpkgPath() returns.
    auto vpkg = _state ? _state->vpkg() : nullptr;
    h.volpkgPath = vpkg ? QString::fromStdString(vpkg->path().string()) : QString();
    h.volumeId = QString::fromStdString(viewer->currentVolume()->id());
    h.volpkgIsRemote = viewer->currentVolume()->isRemote();
    h.segmentId = QString::fromStdString(viewer->surfName() == "segmentation"
                                             ? _state->activeSurfaceId()
                                             : viewer->surfName());
    if (auto* gv = viewer->graphicsView()) {
        h.viewportW = gv->viewport()->width();
        h.viewportH = gv->viewport()->height();
    }
    h.cacheSizeGB = _cacheSizeBytes / (1024ULL * 1024ULL * 1024ULL);
    h.vc3dCommit = QString::fromStdString(ProjectInfo::RepositoryShortHash()).trimmed();
    _benchRecorder->attach(viewer, std::move(h));
}

void CWindow::onSurfaceActivatedPreserveEditing(const QString& surfaceId, QuadSurface* surface)
{
    const std::string previousSurfId = _state->activeSurfaceId();
    const std::string newSurfId = surfaceId.toStdString();

    if (_state && !newSurfId.empty()) {
        auto stateSurface = std::dynamic_pointer_cast<QuadSurface>(_state->surface(newSurfId));
        if (!stateSurface && _state->vpkg()) {
            stateSurface = _state->vpkg()->getSurface(newSurfId);
        }
        _state->setActiveSurface(newSurfId, stateSurface);
    } else {
        _state->clearActiveSurface();
    }

    auto surf = _state->activeSurface().lock();

    _state->setSurface("segmentation", surf, false, false);

    if (newSurfId != previousSurfId && _segmentationModule) {
        try {
            _segmentationModule->onActiveSegmentChanged(surf.get());
        } catch (const std::exception& e) {
            qWarning() << "Failed to activate surface"
                       << surfaceId
                       << "while it may still be writing:"
                       << e.what();
            _state->clearActiveSurface();
            _state->setSurface("segmentation", nullptr, false, false);
            surf.reset();
        }

        // Load corr_points_results for the new segment
        if (_point_collection_widget) {
            auto quadSurf = std::dynamic_pointer_cast<QuadSurface>(surf);
            if (quadSurf && !quadSurf->path.empty()) {
                _point_collection_widget->loadCorrPointsResults(
                    quadSurf->path / "corr_points_results.json");
            } else {
                _point_collection_widget->clearCorrPointsResults();
            }
        }
        if (_atlasControlDock) {
            auto quadSurf = std::dynamic_pointer_cast<QuadSurface>(surf);
            if (quadSurf && !quadSurf->path.empty()) {
                _atlasControlDock->loadResults(quadSurf->path / "atlas_control_points_results.json");
            } else {
                _atlasControlDock->clearResults();
            }
        }

        const bool wantsEditing = _segmentationWidget && _segmentationWidget->isEditingEnabled();
        if (wantsEditing) {
            if (!_segmentationModule->editingEnabled()) {
                _segmentationModule->setEditingEnabled(true);
            } else if (_state) {
                auto targetSurface = surf;
                if (!targetSurface) {
                    targetSurface = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
                }

                if (targetSurface) {
                    _segmentationModule->endEditingSession();
                    if (_segmentationModule->beginEditingSession(targetSurface) && _viewerManager) {
                        _viewerManager->forEachBaseViewer([](VolumeViewerBase* viewer) {
                            if (viewer) {
                                viewer->clearOverlayGroup("segmentation_radius_indicator");
                            }
                        });
                    }
                }
            }
        }
    }

    if (moveOnSurfaceChangeEnabled()) {
        try {
            if (surf) {
                _axisAlignedSliceController->applyOrientation(surf.get());
            } else {
                _axisAlignedSliceController->applyOrientation();
            }
        } catch (const std::exception& e) {
            qWarning() << "Failed to apply surface orientation for"
                       << surfaceId
                       << "while it may still be writing:"
                       << e.what();
            _state->clearActiveSurface();
            _state->setSurface("segmentation", nullptr, false, false);
            _axisAlignedSliceController->applyOrientation();
        }
    }

    if (_surfacePanel && _surfacePanel->isCurrentOnlyFilterEnabled()) {
        _surfacePanel->refreshFiltersOnly();
    }

    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }
}

void CWindow::onSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf)
{
    // Called BEFORE surface deletion - clear all references to prevent use-after-free

    // Clear if this is our current active surface
    auto currentSurf = _state->activeSurface().lock();
    if (currentSurf && currentSurf == surf) {
        _state->clearActiveSurface();
    }

    // Focus history uses string IDs now, so no cleanup needed for surface pointers
    // (the ID remains valid for lookup - will just return nullptr if surface is gone)
}

// Renders <segment>/mask.tif on the calling (worker) thread with no GUI side
// effects. append=false writes a single-layer binary mask; append=true appends a
// surface-image layer to an existing mask (or creates a two-layer mask+image file).
// Throws std::runtime_error on failure; returns a status string on success.
static QString renderSegmentMaskToFile(const std::shared_ptr<QuadSurface>& surf,
                                       const std::shared_ptr<Volume>& volume,
                                       const std::filesystem::path& path,
                                       bool append)
{
    cv::Mat_<uint8_t> mask;
    if (!append) {
        cv::Mat_<cv::Vec3f> coords;
        render_binary_mask(surf.get(), mask, coords, 1.0f);
        // cv::imwrite returns false (rather than throwing) on most write
        // failures; treat that as an error so no caller reports a false success.
        if (!cv::imwrite(path.string(), mask)) {
            throw std::runtime_error("Failed to write mask to " + path.string());
        }
        surf->meta["date_last_modified"] = get_surface_time_str();
        surf->save_meta();
        return QStringLiteral("Mask saved");
    }

    cv::Mat_<uint8_t> img;
    std::vector<cv::Mat> existing_layers;
    if (std::filesystem::exists(path)) {
        cv::imreadmulti(path.string(), existing_layers, cv::IMREAD_UNCHANGED);
        if (existing_layers.empty())
            throw std::runtime_error("Could not read existing mask file.");

        mask = existing_layers[0];
        const cv::Size maskSize = mask.size();
        {
            const cv::Size rawSize = surf->rawPointsPtr()->size();
            const cv::Vec3f ptr(0, 0, 0);
            const cv::Vec3f offset(-rawSize.width / 2.0f, -rawSize.height / 2.0f, 0);
            const float surfScale = surf->scale()[0];
            cv::Mat_<cv::Vec3f> coords;
            surf->gen(&coords, nullptr, maskSize, ptr, surfScale, offset);
            img.create(coords.size());
            render_image_from_coords(coords, img, volume.get());
        }
        cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8U);
        existing_layers.push_back(img);
        atomicImwriteMulti(path, existing_layers);

        surf->meta["date_last_modified"] = get_surface_time_str();
        surf->save_meta();
        return QString("Appended surface image to existing mask (now %1 layers)")
            .arg(existing_layers.size());
    }

    cv::Mat_<cv::Vec3f> coords;
    render_binary_mask(surf.get(), mask, coords, 1.0f);
    render_surface_image(surf.get(), mask, img, volume.get(), 0, 1.0f);
    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8U);
    std::vector<cv::Mat> layers = {mask, img};
    atomicImwriteMulti(path, layers);

    surf->meta["date_last_modified"] = get_surface_time_str();
    surf->save_meta();
    return QString("Created new surface mask with image data");
}

void CWindow::onEditMaskPressed(const QString& segmentId)
{
    auto surf = (_state && _state->vpkg())
        ? _state->vpkg()->getSurface(segmentId.toStdString())
        : nullptr;
    if (!surf) {
        QMessageBox::warning(this, tr("Error"), tr("No surface selected."));
        return;
    }

    std::filesystem::path path = surf->path/"mask.tif";

    // If mask already exists, just open it
    if (std::filesystem::exists(path)) {
        QDesktopServices::openUrl(QUrl::fromLocalFile(path.string().c_str()));
        return;
    }
    if (_maskRenderInProgress) {
        return;
    }

    QString error;
    if (!startMaskRender(
            segmentId, false,
            [this, path](bool success, const QString& message) {
                if (success) {
                    showStatusBarMessage(tr("Mask saved"), 3000);
                    QDesktopServices::openUrl(QUrl::fromLocalFile(
                        QString::fromStdString(path.string())));
                } else {
                    QMessageBox::critical(this, tr("Error"),
                                          tr("Failed to render surface: %1").arg(message));
                    clearStatusBarMessage();
                }
            },
            &error)) {
        QMessageBox::warning(this, tr("Error"), error);
    }
}

void CWindow::onAppendMaskPressed(const QString& segmentId)
{
    auto surf = (_state && _state->vpkg())
        ? _state->vpkg()->getSurface(segmentId.toStdString())
        : nullptr;
    if (!surf || !_state->currentVolume()) {
        if (!surf) {
            QMessageBox::warning(this, tr("Error"), tr("No surface selected."));
        } else {
            QMessageBox::warning(this, tr("Error"), tr("No volume loaded."));
        }
        return;
    }
    if (_maskRenderInProgress) {
        return;
    }

    std::filesystem::path path = surf->path/"mask.tif";
    QString error;
    if (!startMaskRender(
            segmentId, true,
            [this, path](bool success, const QString& message) {
                if (success) {
                    showStatusBarMessage(message, 3000);
                    QDesktopServices::openUrl(QUrl::fromLocalFile(
                        QString::fromStdString(path.string())));
                } else {
                    QMessageBox::critical(this, tr("Error"),
                                          tr("Failed to render surface: %1").arg(message));
                    clearStatusBarMessage();
                }
            },
            &error)) {
        QMessageBox::warning(this, tr("Error"), error);
    }
}

bool CWindow::startMaskRender(const QString& segmentId, bool append,
                              std::function<void(bool, QString)> onFinished,
                              QString* errorMessage)
{
    auto fail = [&](const QString& message) -> bool {
        if (errorMessage) {
            *errorMessage = message;
        }
        return false;
    };

    auto surf = (_state && _state->vpkg())
        ? _state->vpkg()->getSurface(segmentId.toStdString())
        : nullptr;
    if (!surf) {
        return fail(tr("No volume package loaded or invalid segment."));
    }
    auto volume = _state ? _state->currentVolume() : nullptr;
    if (append && !volume) {
        return fail(tr("No volume loaded."));
    }
    if (_maskRenderInProgress) {
        return fail(tr("A mask render is already in progress."));
    }
    _maskRenderInProgress = true;
    showStatusBarMessage(tr("Rendering mask..."));

    // The worker's finished slot clears _maskRenderInProgress on completion, but
    // the setup below (coordinate-identity copy, watcher allocation, future
    // launch) can throw before the worker is ever queued. Arm a guard that
    // restores the flag on any such early failure, so a subsequent mask op is not
    // permanently blocked; it is dismissed once the worker owns the flag.
    bool workerLaunched = false;
    struct InProgressGuard {
        bool& flag;
        const bool& launched;
        ~InProgressGuard() { if (!launched) flag = false; }
    } inProgressGuard{_maskRenderInProgress, workerLaunched};

    const std::filesystem::path path = surf->path / "mask.tif";
    vc3d::opendata::copyVolumeCoordinateIdentityToSurface(
        *surf, *_state->vpkg(), _state->currentVolumeId());

    auto* watcher = new QFutureWatcher<QString>(this);
    connect(watcher, &QFutureWatcher<QString>::finished, this,
            [this, watcher, onFinished = std::move(onFinished)]() {
                watcher->deleteLater();
                _maskRenderInProgress = false;
                clearStatusBarMessage();
                bool success = false;
                QString message;
                try {
                    message = watcher->result();
                    success = true;
                } catch (const std::exception& e) {
                    message = QString::fromUtf8(e.what());
                }
                if (onFinished) {
                    onFinished(success, message);
                }
            });

    watcher->setFuture(QtConcurrent::run([surf, volume, path, append]() -> QString {
        return renderSegmentMaskToFile(surf, volume, path, append);
    }));

    workerLaunched = true;  // worker now owns _maskRenderInProgress; disarm the guard
    return true;
}

QString CWindow::getCurrentVolumePath() const
{
    auto volume = _state->currentVolume();
    if (volume == nullptr) {
        return QString();
    }
    if (volume->isRemote()) {
        return QString::fromStdString(volume->remoteLocator());
    }
    return QString::fromStdString(volume->path().string());
}

QColor CWindow::defaultSegmentFolderColor(const QString& dirName) const
{
    static const std::array<QColor, 10> colors = {
        QColor(0, 180, 255),
        QColor(255, 170, 0),
        QColor(120, 220, 120),
        QColor(255, 100, 150),
        QColor(180, 120, 255),
        QColor(70, 210, 190),
        QColor(255, 230, 90),
        QColor(240, 120, 70),
        QColor(150, 210, 255),
        QColor(210, 210, 210),
    };
    return colors[static_cast<size_t>(qHash(dirName)) % colors.size()];
}

QColor segmentFolderSolidColorOr(const std::map<QString, QColor>& colors,
                                 const QString& dirName,
                                 const QColor& fallback)
{
    auto it = colors.find(dirName);
    return it != colors.end() ? it->second : fallback;
}

QString CWindow::effectiveDefaultSegmentFolderDir() const
{
    if (!_state || !_state->vpkg()) {
        return {};
    }

    const QString currentDir = QString::fromStdString(_state->vpkg()->getSegmentationDirectory());
    const auto availableDirs = _state->vpkg()->getAvailableSegmentationDirectories();
    const auto containsDir = [&availableDirs](const QString& dirName) {
        return std::any_of(availableDirs.begin(), availableDirs.end(), [&dirName](const std::string& available) {
            return QString::fromStdString(available) == dirName;
        });
    };

    if (!_segmentFolderDefaultPaletteDir.isEmpty() && containsDir(_segmentFolderDefaultPaletteDir)) {
        return _segmentFolderDefaultPaletteDir;
    }

    if (!currentDir.isEmpty() && !_segmentFolderSolidColors.contains(currentDir)) {
        return currentDir;
    }
    return {};
}

void CWindow::refreshSegmentationDirectoryDropdown()
{
    if (!_state || !_state->vpkg() || !cmbSegmentationDir || !_segmentDirModel) {
        return;
    }

    QSet<QString> previouslyChecked;
    for (int row = 0; row < _segmentDirModel->rowCount(); ++row) {
        auto* item = _segmentDirModel->item(row);
        if (item && item->checkState() == Qt::Checked) {
            previouslyChecked.insert(item->data(SEGMENT_DIR_NAME_ROLE).toString());
        }
    }

    const QString currentDir = QString::fromStdString(_state->vpkg()->getSegmentationDirectory());
    if (!currentDir.isEmpty()) {
        previouslyChecked.insert(currentDir);
    }

    const QSignalBlocker blocker{cmbSegmentationDir};
    _updatingSegmentDirUi = true;
    _segmentDirModel->clear();

    const auto availableDirs = _state->vpkg()->getAvailableSegmentationDirectories();
    const QString defaultPaletteDir = effectiveDefaultSegmentFolderDir();
    int currentIndex = -1;
    for (const auto& dirNameStd : availableDirs) {
        const QString dirName = QString::fromStdString(dirNameStd);
        auto* item = new QStandardItem(dirName);
        item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable);
        item->setCheckable(true);
        item->setCheckState(previouslyChecked.contains(dirName) ? Qt::Checked : Qt::Unchecked);
        item->setData(dirName, SEGMENT_DIR_NAME_ROLE);
        item->setData(segmentFolderSolidColorOr(_segmentFolderSolidColors, dirName, QColor()),
                      SEGMENT_DIR_COLOR_ROLE);
        item->setData(dirName == defaultPaletteDir, SEGMENT_DIR_DEFAULT_PALETTE_ROLE);
        _segmentDirModel->appendRow(item);
        if (dirName == currentDir) {
            currentIndex = _segmentDirModel->rowCount() - 1;
        }
    }

    if (currentIndex >= 0) {
        cmbSegmentationDir->setCurrentIndex(currentIndex);
    }
    _updatingSegmentDirUi = false;

    applySegmentFolderSelection(false);
}

void CWindow::applySegmentFolderSelection(bool reloadSurfaces)
{
    if (!_state || !_state->vpkg() || !_surfacePanel || !_segmentDirModel) {
        return;
    }

    const QString currentDir = QString::fromStdString(_state->vpkg()->getSegmentationDirectory());
    const QString defaultPaletteDir = effectiveDefaultSegmentFolderDir();
    std::vector<SurfacePanelController::SegmentFolderSelection> folders;
    folders.reserve(static_cast<size_t>(_segmentDirModel->rowCount()));

    _updatingSegmentDirUi = true;
    for (int row = 0; row < _segmentDirModel->rowCount(); ++row) {
        auto* item = _segmentDirModel->item(row);
        if (!item) {
            continue;
        }
        const QString dirName = item->data(SEGMENT_DIR_NAME_ROLE).toString();
        if (dirName == currentDir && item->checkState() != Qt::Checked) {
            item->setCheckState(Qt::Checked);
        }
        const bool checked = item->checkState() == Qt::Checked;
        const bool isCurrent = dirName == currentDir;
        const bool defaultPalette = dirName == defaultPaletteDir;
        const QColor color = defaultPalette
            ? QColor()
            : segmentFolderSolidColorOr(_segmentFolderSolidColors,
                                        dirName,
                                        defaultSegmentFolderColor(dirName));
        item->setData(color, SEGMENT_DIR_COLOR_ROLE);
        item->setData(defaultPalette, SEGMENT_DIR_DEFAULT_PALETTE_ROLE);
        if (!checked) {
            continue;
        }
        folders.push_back(SurfacePanelController::SegmentFolderSelection{
            dirName.toStdString(),
            _state->vpkg()->findSegmentPathByName(dirName.toStdString()),
            isCurrent,
            defaultPalette,
            color,
        });
    }
    _updatingSegmentDirUi = false;

    _surfacePanel->setVisibleSegmentFolders(std::move(folders));
    if (reloadSurfaces) {
        const std::string previousActiveSurfaceId = _state->activeSurfaceId();
        _surfacePanel->resetTagUi();
        _surfacePanel->loadSurfaces(true);
        restoreActiveSurfaceAfterSurfaceReload(previousActiveSurfaceId);
    }
}

void CWindow::showSegmentFolderPaletteMenu(int row)
{
    if (!_segmentDirModel || row < 0 || row >= _segmentDirModel->rowCount()) {
        return;
    }
    auto* item = _segmentDirModel->item(row);
    if (!item) {
        return;
    }
    const QString dirName = item->data(SEGMENT_DIR_NAME_ROLE).toString();
    const QString previousDefaultDir = effectiveDefaultSegmentFolderDir();

    QMenu menu(cmbSegmentationDir);
    QAction* defaultAction = menu.addAction(tr("Default palette"));
    defaultAction->setCheckable(true);
    defaultAction->setChecked(dirName == previousDefaultDir);
    connect(defaultAction, &QAction::triggered, this, [this, dirName, previousDefaultDir]() {
        if (!previousDefaultDir.isEmpty() && previousDefaultDir != dirName &&
            !_segmentFolderSolidColors.contains(previousDefaultDir)) {
            _segmentFolderSolidColors[previousDefaultDir] = defaultSegmentFolderColor(previousDefaultDir);
        }
        _segmentFolderSolidColors.erase(dirName);
        _segmentFolderDefaultPaletteDir = dirName;
        applySegmentFolderSelection(true);
        if (cmbSegmentationDir) {
            cmbSegmentationDir->view()->viewport()->update();
        }
    });
    menu.addSeparator();

    QAction* chooseAction = menu.addAction(tr("Solid color..."));
    connect(chooseAction, &QAction::triggered, this, [this, dirName]() {
        const QColor initial = segmentFolderSolidColorOr(_segmentFolderSolidColors,
                                                        dirName,
                                                        defaultSegmentFolderColor(dirName));
        const QColor chosen = QColorDialog::getColor(initial, this, tr("Segment Folder Color"));
        if (!chosen.isValid()) {
            return;
        }
        _segmentFolderSolidColors[dirName] = chosen;
        if (effectiveDefaultSegmentFolderDir() == dirName) {
            _segmentFolderDefaultPaletteDir.clear();
        }
        applySegmentFolderSelection(true);
        if (cmbSegmentationDir) {
            cmbSegmentationDir->view()->viewport()->update();
        }
    });
    menu.exec(QCursor::pos());
}

void CWindow::onSegmentationDirChanged(int index)
{
    if (!_state->vpkg() || index < 0 || !cmbSegmentationDir) {
        return;
    }

    std::string newDir;
    if (_segmentDirModel && index < _segmentDirModel->rowCount()) {
        newDir = _segmentDirModel->item(index)->data(SEGMENT_DIR_NAME_ROLE).toString().toStdString();
    } else {
        newDir = cmbSegmentationDir->itemText(index).toStdString();
    }

    // Only reload if the directory actually changed
    if (newDir != _state->vpkg()->getSegmentationDirectory()) {
        const std::string previousActiveSurfaceId = _state->activeSurfaceId();

        // Set the new directory in the VolumePkg
        _state->vpkg()->setSegmentationDirectory(newDir);
        _state->vpkg()->refreshSegmentations();

        // Reset stride user override for the new directory.
        if (_viewerManager) {
            _viewerManager->resetStrideUserOverride();
        }
        applySegmentFolderSelection(false);
        if (_surfacePanel) {
            _surfacePanel->loadSurfaces(true);
        }

        const bool restoredActiveSurface =
            restoreActiveSurfaceAfterSurfaceReload(previousActiveSurfaceId);

        if (!restoredActiveSurface) {
            _state->clearActiveSurface();
            _state->setSurface("segmentation", nullptr, true);
            if (treeWidgetSurfaces) {
                treeWidgetSurfaces->clearSelection();
            }
            if (_surfacePanel) {
                _surfacePanel->resetTagUi();
            }
        }

        // Update the status bar to show the change
        showStatusBarMessage(tr("Switched to %1 directory").arg(QString::fromStdString(newDir)), 3000);
    }
}


void CWindow::onManualLocationChanged()
{
    // Check if we have a valid volume loaded
    if (!_state->currentVolume()) {
        return;
    }

    // Parse the comma-separated values
    QString text = lblLocFocus->text().trimmed();
    QStringList parts = text.split(',');

    // Validate we have exactly 3 parts
    if (parts.size() != 3) {
        // Invalid input - restore the previous values
        POI* poi = _state->poi("focus");
        if (poi) {
            lblLocFocus->setText(QString("%1, %2, %3")
                .arg(static_cast<int>(poi->p[0]))
                .arg(static_cast<int>(poi->p[1]))
                .arg(static_cast<int>(poi->p[2])));
        }
        return;
    }

    // Parse each coordinate
    bool ok[3];
    int x = parts[0].trimmed().toInt(&ok[0]);
    int y = parts[1].trimmed().toInt(&ok[1]);
    int z = parts[2].trimmed().toInt(&ok[2]);

    // Validate the input
    if (!ok[0] || !ok[1] || !ok[2]) {
        // Invalid input - restore the previous values
        POI* poi = _state->poi("focus");
        if (poi) {
            lblLocFocus->setText(QString("%1, %2, %3")
                .arg(static_cast<int>(poi->p[0]))
                .arg(static_cast<int>(poi->p[1]))
                .arg(static_cast<int>(poi->p[2])));
        }
        return;
    }

    // Clamp values to physical volume bounds
    auto [w, h, d] = _state->currentVolume()->shapeXyz();
    int cx0 = 0, cy0 = 0, cz0 = 0;
    int cx1 = w - 1, cy1 = h - 1, cz1 = d - 1;

    x = std::max(cx0, std::min(x, cx1));
    y = std::max(cy0, std::min(y, cy1));
    z = std::max(cz0, std::min(z, cz1));

    // Update the line edit with clamped values
    lblLocFocus->setText(QString("%1, %2, %3").arg(x).arg(y).arg(z));

    // Route through centerFocusAt so slice planes reorient (canonical fallback
    // when no segment is loaded, segment-tangent otherwise) — same behaviour
    // as ctrl-click in a viewer.
    centerFocusAt(cv::Vec3f(x, y, z), cv::Vec3f(0, 0, 1), std::string());

    if (_surfacePanel) {
        _surfacePanel->refreshFiltersOnly();
    }
}

void CWindow::onZoomIn()
{
    if (auto* viewer = activeBaseViewer()) {
        viewer->adjustZoomByFactor(1.15f);
    }
}

void CWindow::onFocusPOIChanged(std::string name, POI* poi)
{
    // Slice-plane reorientation and viewer recentering are handled centrally
    // by ViewerManager::handleFocusPoiChanged; only Main-specific UI here.
    if (name == "focus" && poi) {
        lblLocFocus->setText(QString("%1, %2, %3")
            .arg(static_cast<int>(poi->p[0]))
            .arg(static_cast<int>(poi->p[1]))
            .arg(static_cast<int>(poi->p[2])));

        if (_surfacePanel) {
            _surfacePanel->refreshFiltersOnly();
        }
    }
}

void CWindow::onPointDoubleClicked(uint64_t pointId)
{
    auto point_opt = _state->pointCollection()->getPoint(pointId);
    if (point_opt) {
        centerFocusAt(point_opt->p, cv::Vec3f(0, 0, 0), "");
    }
}

void CWindow::onConvertPointToAnchor(uint64_t pointId, uint64_t collectionId)
{
    auto point_opt = _state->pointCollection()->getPoint(pointId);
    if (!point_opt) {
        showStatusBarMessage(tr("Point not found"), 2000);
        return;
    }

    // Get the segmentation surface to project the point onto
    auto seg_surface = _state->surface("segmentation");
    auto* quad_surface = dynamic_cast<QuadSurface*>(seg_surface.get());
    if (!quad_surface) {
        showStatusBarMessage(tr("No active segmentation surface for anchor conversion"), 3000);
        return;
    }

    // Find the 2D grid location of this point on the surface
    cv::Vec3f ptr(0, 0, 0);
    auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
    float dist = quad_surface->pointTo(ptr, point_opt->p, 4.0, 1000, patchIndex);

    if (dist > 10.0) {
        showStatusBarMessage(tr("Point is too far from surface (distance: %1)").arg(dist), 3000);
        return;
    }

    // Get the raw grid location (internal coordinates)
    cv::Vec3f loc_3d = quad_surface->loc_raw(ptr);
    cv::Vec2f anchor2d(loc_3d[0], loc_3d[1]);

    // Set the anchor2d on the collection
    _state->pointCollection()->setCollectionAnchor2d(collectionId, anchor2d);

    // Remove the point (it's now represented by the anchor)
    _state->pointCollection()->removePoint(pointId);

    showStatusBarMessage(tr("Converted point to anchor at grid position (%1, %2)").arg(anchor2d[0]).arg(anchor2d[1]), 3000);
}

void CWindow::onZoomOut()
{
    if (auto* viewer = activeBaseViewer()) {
        viewer->adjustZoomByFactor(1.0f / 1.15f);
    }
}

void CWindow::onCopyCoordinates()
{
    QString coords = lblLocFocus->text().trimmed();
    if (!coords.isEmpty()) {
        QApplication::clipboard()->setText(coords);
        showStatusBarMessage(tr("Coordinates copied to clipboard: %1").arg(coords), 2000);
    }
}

void CWindow::onResetAxisAlignedRotations()
{
    _axisAlignedSliceController->resetRotations();
    _axisAlignedSliceController->applyOrientation();
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->refreshAll();
    }
    showStatusBarMessage(tr("All plane rotations reset"), 2000);
}

void CWindow::onAxisOverlayVisibilityToggled(bool enabled)
{
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->setAxisAlignedEnabled(enabled && _axisAlignedSliceController->isEnabled());
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::SHOW_AXIS_OVERLAYS, enabled ? "1" : "0");
}

void CWindow::onAxisOverlayOpacityChanged(int value)
{
    float normalized = std::clamp(static_cast<float>(value) / 100.0f, 0.0f, 1.0f);
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->setAxisAlignedOverlayOpacity(normalized);
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::AXIS_OVERLAY_OPACITY, value);
}

void CWindow::onAxisAlignedSlicesToggled(bool enabled)
{
    _axisAlignedSliceController->setEnabled(enabled, ui.chkAxisOverlays);
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES, enabled ? "1" : "0");
}

void CWindow::onMoveOnSurfaceChangedToggled(bool enabled)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE, enabled ? "1" : "0");

    for (auto* manager : ViewerManager::allManagers()) {
        manager->setResetViewOnSurfaceChangeDefault(enabled);
    }
}

void CWindow::onPlaneIntersectionLinesToggled(bool enabled)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::SHOW_PLANE_INTERSECTION_LINES, enabled ? "1" : "0");

    for (auto* manager : ViewerManager::allManagers()) {
        manager->setPlaneIntersectionLinesVisible(enabled);
    }
}

void CWindow::onSegmentationEditingModeChanged(bool enabled)
{
    if (!_segmentationModule) {
        return;
    }

    const bool already = _segmentationModule->editingEnabled();
    if (already != enabled) {
        // Update widget to reflect actual module state to avoid drift.
        if (_segmentationWidget && _segmentationWidget->isEditingEnabled() != already) {
            _segmentationWidget->setEditingEnabled(already);
        }
        enabled = already;
    }

    std::optional<std::string> recentlyEditedId;
    if (!enabled) {
        if (auto* activeSurface = _segmentationModule->activeBaseSurface()) {
            recentlyEditedId = activeSurface->id;
        }
    }

    // Set flag BEFORE beginEditingSession so the surface change doesn't reset view
    if (_viewerManager) {
        _viewerManager->setSegmentationResetViewSuppressed(enabled);
    }

    if (enabled) {
        auto activeSurfaceShared = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));

        if (!_segmentationModule->beginEditingSession(activeSurfaceShared)) {
            showStatusBarMessage(tr("Unable to start segmentation editing"), 3000);
            if (_segmentationWidget && _segmentationWidget->isEditingEnabled()) {
                QSignalBlocker blocker(_segmentationWidget);
                _segmentationWidget->setEditingEnabled(false);
            }
            _segmentationModule->setEditingEnabled(false);
            return;
        }

        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([](VolumeViewerBase* viewer) {
                if (viewer) {
                    viewer->clearOverlayGroup("segmentation_radius_indicator");
                }
            });
        }
    } else {
        _segmentationModule->endEditingSession();

        if (recentlyEditedId && !recentlyEditedId->empty()) {
            _fileWatcher->markSegmentRecentlyEdited(*recentlyEditedId);
        }
    }

    const QString message = enabled
        ? tr("Segmentation editing enabled")
        : tr("Segmentation editing disabled");
    showStatusBarMessage(message, 2000);
    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }
}

void CWindow::onSegmentationStopToolsRequested()
{
    if (!initializeCommandLineRunner()) {
        return;
    }
    if (_cmdRunner) {
        _cmdRunner->cancel();
        showStatusBarMessage(tr("Cancelling running tools..."), 3000);
    }
}

void CWindow::onGrowSegmentationSurface(SegmentationGrowthMethod method,
                                        SegmentationGrowthDirection direction,
                                        int steps,
                                        bool inpaintOnly)
{
    if (!_segmentationGrower) {
        showStatusBarMessage(tr("Segmentation growth is unavailable."), 4000);
        return;
    }

    updateNormalGridAvailability();

    SegmentationGrower::Context context{
        _segmentationModule.get(),
        _segmentationWidget,
        _state,
        _viewerManager.get(),
    };
    _segmentationGrower->updateContext(context);

    SegmentationGrower::VolumeContext volumeContext{
        _state->vpkg(),
        _state->currentVolume(),
        _state->currentVolumeId(),
        _state->segmentationGrowthVolumeId().empty() ? _state->currentVolumeId() : _state->segmentationGrowthVolumeId(),
        _normalGridPath,
        _segmentationWidget ? _segmentationWidget->normal3dZarrPath() : QString()
    };

    if (!_segmentationGrower->start(volumeContext, method, direction, steps, inpaintOnly)) {
        return;
    }
}

void CWindow::updateSurfaceOverlayDropdown()
{
    if (!ui.surfaceOverlaySelect) {
        return;
    }

    QSet<QString> previouslyChecked;
    if (_surfaceOverlayModel) {
        for (auto* item : surfaceOverlaySurfaceItems(_surfaceOverlayModel)) {
            if (item && item->checkState() == Qt::Checked) {
                previouslyChecked.insert(item->data(SURFACE_OVERLAY_NAME_ROLE).toString());
            }
        }
        disconnect(_surfaceOverlayModel, &QStandardItemModel::dataChanged,
                   this, &CWindow::onSurfaceOverlaySelectionChanged);
        _surfaceOverlayModel->deleteLater();
    }

    _surfaceOverlayModel = new QStandardItemModel(this);
    _surfaceOverlayModel->setHorizontalHeaderLabels({tr("Surface")});

    QString currentDir;
    if (_state->vpkg()) {
        currentDir = QString::fromStdString(_state->vpkg()->getSegmentationDirectory());
    }

    if (_state) {
        const auto names = _state->surfaceNames();
        for (const auto& name : names) {
            auto surf = _state->surface(name);
            auto* quadSurf = dynamic_cast<QuadSurface*>(surf.get());
            if (!quadSurf) {
                continue;
            }

            const QString folderName = surfaceOverlayFolderName(name, surf.get(), currentDir);
            QStandardItem* folderItem = findSurfaceOverlayFolder(_surfaceOverlayModel, folderName);
            if (!folderItem) {
                folderItem = new QStandardItem(folderName);
                folderItem->setFlags(Qt::ItemIsUserCheckable |
                                     Qt::ItemIsEnabled |
                                     Qt::ItemIsSelectable |
                                     Qt::ItemIsAutoTristate);
                folderItem->setCheckable(true);
                folderItem->setCheckState(Qt::Unchecked);
                folderItem->setData(static_cast<int>(SurfaceOverlayItemKind::Folder),
                                    SURFACE_OVERLAY_KIND_ROLE);
                folderItem->setData(folderName, SURFACE_OVERLAY_FOLDER_ROLE);
                _surfaceOverlayModel->appendRow(folderItem);
            }

            if (_surfaceOverlayColorAssignments.find(name) == _surfaceOverlayColorAssignments.end()) {
                _surfaceOverlayColorAssignments[name] = _nextSurfaceOverlayColorIndex++;
            }
            size_t colorIdx = _surfaceOverlayColorAssignments[name];

            auto* item = new QStandardItem(surfaceOverlayLeafLabel(name, folderName));
            item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
            item->setCheckable(true);
            item->setCheckState(previouslyChecked.contains(QString::fromStdString(name))
                                    ? Qt::Checked
                                    : Qt::Unchecked);
            item->setData(static_cast<int>(SurfaceOverlayItemKind::Surface),
                          SURFACE_OVERLAY_KIND_ROLE);
            item->setData(QString::fromStdString(name), SURFACE_OVERLAY_NAME_ROLE);
            item->setData(folderName, SURFACE_OVERLAY_FOLDER_ROLE);

            QPixmap swatch(16, 16);
            swatch.fill(getOverlayColor(colorIdx));
            item->setIcon(QIcon(swatch));

            folderItem->appendRow(item);
        }
    }

    for (int row = 0; row < _surfaceOverlayModel->rowCount(); ++row) {
        updateSurfaceOverlayFolderState(_surfaceOverlayModel->item(row));
    }

    connect(_surfaceOverlayModel, &QStandardItemModel::dataChanged,
            this, &CWindow::onSurfaceOverlaySelectionChanged);

    applySurfaceOverlaySelection();
}

void CWindow::onSurfaceOverlaySelectionChanged(const QModelIndex& topLeft,
                                                const QModelIndex& bottomRight,
                                                const QVector<int>& roles)
{
    if ((!roles.isEmpty() && !roles.contains(Qt::CheckStateRole)) ||
        !_surfaceOverlayModel ||
        !_viewerManager) {
        return;
    }

    {
        QSignalBlocker blocker(_surfaceOverlayModel);
        for (int row = topLeft.row(); row <= bottomRight.row(); ++row) {
            const QModelIndex idx = topLeft.sibling(row, topLeft.column());
            QStandardItem* changedItem = _surfaceOverlayModel->itemFromIndex(idx);
            if (!changedItem) {
                continue;
            }
            if (isSurfaceOverlayFolder(changedItem)) {
                const Qt::CheckState state = changedItem->checkState() == Qt::Checked
                    ? Qt::Checked
                    : Qt::Unchecked;
                for (int childRow = 0; childRow < changedItem->rowCount(); ++childRow) {
                    QStandardItem* child = changedItem->child(childRow);
                    if (isSurfaceOverlaySurface(child)) {
                        child->setCheckState(state);
                    }
                }
            } else if (isSurfaceOverlaySurface(changedItem) && changedItem->parent()) {
                updateSurfaceOverlayFolderState(changedItem->parent());
            }
        }
    }

    applySurfaceOverlaySelection();
}

void CWindow::applySurfaceOverlaySelection()
{
    if (!_surfaceOverlayModel || !_viewerManager) {
        updateSurfaceOverlayButtonText();
        return;
    }

    std::map<std::string, cv::Vec3b> selectedSurfaces;
    for (auto* item : surfaceOverlaySurfaceItems(_surfaceOverlayModel)) {
        if (!item) {
            continue;
        }
        if (item->checkState() == Qt::Checked) {
            std::string name = item->data(SURFACE_OVERLAY_NAME_ROLE).toString().toStdString();
            size_t colorIdx = _surfaceOverlayColorAssignments[name];
            selectedSurfaces[name] = getOverlayColorBGR(colorIdx);
        }
    }

    _viewerManager->setSurfaceOverlays(selectedSurfaces);

    updateSurfaceOverlayButtonText();
}

void CWindow::updateSurfaceOverlayButtonText()
{
    if (!ui.surfaceOverlaySelect) {
        return;
    }

    int checkedCount = 0;
    int totalCount = 0;
    if (_surfaceOverlayModel) {
        for (auto* item : surfaceOverlaySurfaceItems(_surfaceOverlayModel)) {
            if (!item) {
                continue;
            }
            ++totalCount;
            if (item->checkState() == Qt::Checked) {
                ++checkedCount;
            }
        }
    }

    if (totalCount == 0) {
        ui.surfaceOverlaySelect->setText(tr("No surfaces"));
    } else if (checkedCount == 0) {
        ui.surfaceOverlaySelect->setText(tr("Select..."));
    } else if (checkedCount == totalCount) {
        ui.surfaceOverlaySelect->setText(tr("All surfaces (%1)").arg(totalCount));
    } else {
        ui.surfaceOverlaySelect->setText(tr("%1 surfaces").arg(checkedCount));
    }
}

void CWindow::showSurfaceOverlaySelectionDialog()
{
    if (!_surfaceOverlayModel) {
        updateSurfaceOverlayDropdown();
    }
    if (!_surfaceOverlayModel) {
        return;
    }

    QDialog dialog(this);
    dialog.setWindowTitle(tr("Select Overlap Surfaces"));
    dialog.resize(460, 560);

    auto* layout = new QVBoxLayout(&dialog);
    auto* filterEdit = new QLineEdit(&dialog);
    filterEdit->setPlaceholderText(tr("Filter surfaces or folders"));
    layout->addWidget(filterEdit);

    auto* tree = new QTreeView(&dialog);
    tree->setModel(_surfaceOverlayModel);
    tree->setHeaderHidden(true);
    tree->setAlternatingRowColors(true);
    tree->setSelectionMode(QAbstractItemView::NoSelection);
    tree->expandAll();
    layout->addWidget(tree, 1);

    auto* bulkRow = new QHBoxLayout();
    auto* checkAll = new QPushButton(tr("Check All"), &dialog);
    auto* deselectAll = new QPushButton(tr("Deselect All"), &dialog);
    bulkRow->addWidget(checkAll);
    bulkRow->addWidget(deselectAll);
    bulkRow->addStretch(1);
    layout->addLayout(bulkRow);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, &dialog);
    layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

    const auto refreshFilter = [tree, filterEdit, this]() {
        const QString filter = filterEdit->text().trimmed();
        for (int folderRow = 0; folderRow < _surfaceOverlayModel->rowCount(); ++folderRow) {
            QStandardItem* folder = _surfaceOverlayModel->item(folderRow);
            const bool folderVisible = surfaceOverlayItemMatchesFilter(folder, filter);
            tree->setRowHidden(folderRow, QModelIndex(), !folderVisible);
            for (int row = 0; folder && row < folder->rowCount(); ++row) {
                QStandardItem* child = folder->child(row);
                const bool childVisible = filter.isEmpty() ||
                    surfaceOverlayItemMatchesFilter(child, filter) ||
                    folder->text().toCaseFolded().contains(filter.toCaseFolded());
                tree->setRowHidden(row, folder->index(), !childVisible);
            }
        }
        tree->expandAll();
    };

    const auto setVisibleSurfaces = [tree, this](Qt::CheckState state) {
        if (!_surfaceOverlayModel) {
            return;
        }
        QSignalBlocker blocker(_surfaceOverlayModel);
        for (int folderRow = 0; folderRow < _surfaceOverlayModel->rowCount(); ++folderRow) {
            if (tree->isRowHidden(folderRow, QModelIndex())) {
                continue;
            }
            QStandardItem* folder = _surfaceOverlayModel->item(folderRow);
            for (int row = 0; folder && row < folder->rowCount(); ++row) {
                if (tree->isRowHidden(row, folder->index())) {
                    continue;
                }
                QStandardItem* child = folder->child(row);
                if (isSurfaceOverlaySurface(child)) {
                    child->setCheckState(state);
                }
            }
            updateSurfaceOverlayFolderState(folder);
        }
        applySurfaceOverlaySelection();
    };

    connect(filterEdit, &QLineEdit::textChanged, &dialog, refreshFilter);
    connect(checkAll, &QPushButton::clicked, &dialog, [setVisibleSurfaces]() {
        setVisibleSurfaces(Qt::Checked);
    });
    connect(deselectAll, &QPushButton::clicked, &dialog, [setVisibleSurfaces]() {
        setVisibleSurfaces(Qt::Unchecked);
    });

    refreshFilter();
    dialog.exec();
}

QColor CWindow::getOverlayColor(size_t index) const
{
    return vc3d::surfaceOverlayColor(index);
}

cv::Vec3b CWindow::getOverlayColorBGR(size_t index) const
{
    return vc3d::surfaceOverlayColorBgr(index);
}

void CWindow::onCopyWithNtRequested()
{
    if (!_segmentationGrower) {
        showStatusBarMessage(tr("Segmentation growth is unavailable."), 4000);
        return;
    }

    updateNormalGridAvailability();

    SegmentationGrower::Context context{
        _segmentationModule.get(),
        _segmentationWidget,
        _state,
        _viewerManager.get(),
    };
    _segmentationGrower->updateContext(context);

    SegmentationGrower::VolumeContext volumeContext{
        _state->vpkg(),
        _state->currentVolume(),
        _state->currentVolumeId(),
        _state->segmentationGrowthVolumeId().empty() ? _state->currentVolumeId() : _state->segmentationGrowthVolumeId(),
        _normalGridPath,
        _segmentationWidget ? _segmentationWidget->normal3dZarrPath() : QString()
    };

    if (!_segmentationGrower->startCopyWithNt(volumeContext)) {
        return;
    }
}

void CWindow::onFocusViewsRequested(uint64_t collectionId, uint64_t pointId)
{
    if (!_state) return;
    auto* pointCollection = _state->pointCollection();
    if (!pointCollection) return;

    const auto& collections = pointCollection->getAllCollections();
    auto it = collections.find(collectionId);
    if (it == collections.end()) return;

    const auto& collection = it->second;
    if (collection.points.empty()) return;

    // Gather all 3D points
    std::vector<cv::Vec3f> pts;
    pts.reserve(collection.points.size());
    for (const auto& pair : collection.points) {
        pts.push_back(pair.second.p);
    }

    // Compute centroid
    cv::Vec3f centroid(0, 0, 0);
    for (const auto& p : pts) centroid += p;
    centroid *= 1.0f / pts.size();

    // Determine focus position
    cv::Vec3f focusPos = centroid;
    if (pointId != 0) {
        auto point_opt = pointCollection->getPoint(pointId);
        if (point_opt) focusPos = point_opt->p;
    }

    // Compute plane normal via PCA (only if >= 3 points)
    cv::Vec3f N(0, 0, 1); // default
    if (pts.size() >= 3) {
        // Build 3x3 covariance matrix from centered points
        cv::Matx33f cov = cv::Matx33f::zeros();
        for (const auto& p : pts) {
            cv::Vec3f d = p - centroid;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    cov(r, c) += d[r] * d[c];
        }
        cv::Mat eigenvalues, eigenvectors;
        cv::eigen(cv::Mat(cov), eigenvalues, eigenvectors);
        // Eigenvectors are sorted descending by eigenvalue.
        // Smallest eigenvalue's eigenvector (row 2) = plane normal.
        N = cv::Vec3f(eigenvectors.at<float>(2, 0),
                      eigenvectors.at<float>(2, 1),
                      eigenvectors.at<float>(2, 2));
        N = normalizeOrZero(N);
        if (cv::norm(N) < kEpsilon) N = cv::Vec3f(0, 0, 1);
    } else if (pts.size() == 2) {
        cv::Vec3f d = normalizeOrZero(pts[1] - pts[0]);
        if (cv::norm(d) > kEpsilon) {
            // Pick N perpendicular to d and closest to a canonical axis
            cv::Vec3f candidates[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
            float bestDot = 1.0f;
            cv::Vec3f bestN(0, 0, 1);
            for (auto& axis : candidates) {
                float absDot = std::abs(d.dot(axis));
                if (absDot < bestDot) {
                    bestDot = absDot;
                    cv::Vec3f proj = normalizeOrZero(axis - d * d.dot(axis));
                    if (cv::norm(proj) > kEpsilon) bestN = proj;
                }
            }
            N = bestN;
        }
    } else {
        // 1 point: just center, don't change orientation
        centerFocusAt(focusPos, cv::Vec3f(0, 0, 1), "");
        return;
    }

    // Choose which viewer gets the primary plane
    const cv::Vec3f segYZCanonical(1, 0, 0);
    const cv::Vec3f segXZCanonical(0, 1, 0);

    std::string primaryName, secondaryName;
    cv::Vec3f secondaryCanonical;

    if (std::abs(N.dot(segYZCanonical)) >= std::abs(N.dot(segXZCanonical))) {
        primaryName = "seg yz";
        secondaryName = "seg xz";
        secondaryCanonical = segXZCanonical;
    } else {
        primaryName = "seg xz";
        secondaryName = "seg yz";
        secondaryCanonical = segYZCanonical;
    }

    // Helper to configure a plane with Z-up in-plane rotation
    const auto configureFocusPlane = [&](const std::string& planeName,
                                         const cv::Vec3f& normal) {
        auto planeShared = std::dynamic_pointer_cast<PlaneSurface>(_state->surface(planeName));
        if (!planeShared) {
            planeShared = std::make_shared<PlaneSurface>();
        }
        planeShared->setOrigin(focusPos);
        planeShared->setNormal(normal);
        planeShared->setInPlaneRotation(0.0f);

        // Adjust in-plane rotation so Z projects "up"
        const cv::Vec3f upAxis(0.0f, 0.0f, 1.0f);
        const cv::Vec3f projectedUp = projectVectorOntoPlane(upAxis, normal);
        const cv::Vec3f desiredUp = normalizeOrZero(projectedUp);
        if (cv::norm(desiredUp) > kEpsilon) {
            const cv::Vec3f currentUp = planeShared->basisY();
            const float delta = signedAngleBetween(currentUp, desiredUp, normal);
            if (std::abs(delta) > kEpsilon) {
                planeShared->setInPlaneRotation(delta);
            }
        }

        _state->setSurface(planeName, planeShared);
    };

    // Set focus POI first — this triggers applySlicePlaneOrientation() which
    // overwrites slice planes. We set our custom planes after.
    POI* focus = _state->poi("focus");
    if (!focus) {
        focus = new POI;
    }
    focus->p = focusPos;
    focus->n = N;
    focus->surfacePtr.reset();
    _state->setPOI("focus", focus);

    // Now set our PCA-derived planes (overriding what applySlicePlaneOrientation set)
    configureFocusPlane(primaryName, N);

    // Set secondary plane: component of other canonical axis orthogonal to N
    cv::Vec3f secNormal = normalizeOrZero(secondaryCanonical - N * N.dot(secondaryCanonical));
    if (cv::norm(secNormal) < kEpsilon) {
        // Fallback: use cross product
        secNormal = normalizeOrZero(crossProduct(N, cv::Vec3f(0, 0, 1)));
        if (cv::norm(secNormal) < kEpsilon) {
            secNormal = normalizeOrZero(crossProduct(N, cv::Vec3f(0, 1, 0)));
        }
    }
    configureFocusPlane(secondaryName, secNormal);

    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->refreshAll();
    }

    showStatusBarMessage(tr("Focused & aligned view to %1 points").arg(pts.size()), 3000);
}

// ---- Fiber annotation slots -------------------------------------------------

void CWindow::onNewFiberRequested()
{
    if (_fiberController) {
        _fiberController->beginNewFiber();
    }
}

void CWindow::onFiberCrosshairModeChanged(bool active)
{
    if (!_viewerManager) return;
    _viewerManager->forEachBaseViewer([active](VolumeViewerBase* v) {
        if (v->graphicsView()) {
            v->graphicsView()->setCursor(active ? Qt::CrossCursor : Qt::ArrowCursor);
        }
    });
}

void CWindow::onFiberViewersRequested()
{
    if (!_fiberController) return;

    constexpr int N = FiberAnnotationController::kNumViews;
    QMdiArea* targetMdiArea = _fiberSliceMdiArea ? _fiberSliceMdiArea : mdiArea;
    if (!targetMdiArea) {
        return;
    }

    QMdiSubWindow* subWindows[N] = {};

    for (int i = 0; i < N; ++i) {
        QString title = (i == 0) ? tr("Fiber Ref") : tr("Fiber Annotate");

        auto* baseViewer = newConnectedViewer(
            FiberAnnotationController::fiberSurfaceName(i), title, targetMdiArea);
        auto* viewer = baseViewer
            ? qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject())
            : nullptr;
        if (!viewer) continue;

        _fiberController->setFiberViewer(i, viewer);

        // Isolate from segmentation module and global focus POI
        disconnect(_state, &CState::poiChanged, viewer, &CTiledVolumeViewer::onPOIChanged);
        if (_segmentationModule)
            disconnect(viewer, nullptr, _segmentationModule.get(), nullptr);

        // Only the last view (annotation) handles clicks
        if (i == N - 1) {
            connect(viewer, &CTiledVolumeViewer::sendVolumeClicked,
                    _fiberController.get(), &FiberAnnotationController::onAnnotationViewerClicked);
        }

        if (_state->currentVolume())
            viewer->OnVolumeChanged(_state->currentVolume());

        subWindows[i] = qobject_cast<QMdiSubWindow*>(viewer->parentWidget());
        if (subWindows[i])
            subWindows[i]->show();
    }

    // Layout: 2 columns × 1 row — ref on the left, annotate on the right.
    QRect area = targetMdiArea->contentsRect();
    int colW = area.width() / 2;
    int rowH = area.height();

    for (int i = 0; i < N; ++i) {
        if (!subWindows[i]) continue;
        int x = area.x() + i * colW;
        int y = area.y();
        subWindows[i]->setGeometry(x, y, colW, rowH);
    }
}

void CWindow::onFiberAnnotationFinished(uint64_t fiberId)
{
    if (_fiberWidget) {
        _fiberWidget->selectFiber(fiberId);
    }
    if (_fiberSliceWidget) {
        _fiberSliceWidget->selectFiber(fiberId);
    }
}
