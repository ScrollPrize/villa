#include "CWindow.hpp"

#include "RenderBenchRecorder.hpp"
#include "RenderBenchReplay.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/atlas/Atlas.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <iostream>

#include <functional>

#include "VCSettings.hpp"
#include "Keybinds.hpp"
#include <QGridLayout>
#include <QAction>
#include <QVBoxLayout>
#include <QCursor>
#include <QEvent>
#include <QEventLoop>
#include <QKeyEvent>
#include <QResizeEvent>
#include <QSettings>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QApplication>
#include <QGuiApplication>
#include <QStyleHints>
#include <QWindow>
#include <QScreen>
#include <QDesktopServices>
#include <QUrl>
#include <QClipboard>
#include <QEventLoop>
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
#include <QSizePolicy>
#include <QTimer>
#include <QSize>
#include <QVector>
#include <QLoggingCategory>
#include <QDebug>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QMenu>
#include <QMainWindow>
#include <QTabWidget>
#include <QTreeWidget>
#include <QHeaderView>
#include <QFormLayout>
#include <QProgressBar>
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

#include "volume_viewers/CVolumeViewerView.hpp"
#include "viewer_controls/ViewerControlsPanel.hpp"
#include "viewer_controls/panels/ViewerTransformsPanel.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "SettingsDialog.hpp"
#include "elements/VolumeSelector.hpp"
#include "CPointCollectionWidget.hpp"
#include "AtlasControlPointsDock.hpp"
#include "CFiberWidget.hpp"
#include "FiberAnnotationController.hpp"
#include "LineAnnotationController.hpp"
#include "SurfaceTreeWidget.hpp"
#include "SeedingWidget.hpp"
#include "CommandLineToolRunner.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"
#include "segmentation/growth/SegmentationGrower.hpp"
#include "SurfacePanelController.hpp"
#include "MenuActionController.hpp"
#include "FileWatcherService.hpp"
#include "AxisAlignedSliceController.hpp"
#include "SurfaceAreaCalculator.hpp"
#include "SegmentationCommandHandler.hpp"
#include "LasagnaServiceManager.hpp"
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
constexpr auto ATLAS_INTERNAL_SURFACE_NAME = "__atlas_workspace_base_mesh";
constexpr int ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS = 0;
constexpr int ATLAS_SEARCH_MODE_NON_ATLAS_ONLY = 1;
constexpr int ATLAS_SEARCH_RESULT_INDEX_ROLE = Qt::UserRole;
constexpr int ATLAS_SEARCH_FIBER_ID_ROLE = Qt::UserRole + 1;
constexpr int ATLAS_FIBER_ID_ROLE = Qt::UserRole;
constexpr int ATLAS_FIBER_PATH_KEY_ROLE = Qt::UserRole + 1;
constexpr int ATLAS_CONTROL_INDEX_ROLE = Qt::UserRole + 2;
constexpr int ATLAS_CONTROL_SOURCE_INDEX_ROLE = Qt::UserRole + 3;
constexpr int ATLAS_SURFACE_X_ROLE = Qt::UserRole + 4;
constexpr int ATLAS_SURFACE_Y_ROLE = Qt::UserRole + 5;
constexpr double ATLAS_SEARCH_CLOSE_WINDING_THRESHOLD = 0.5;

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

void centerViewerOnVolumePointForNavigation(VolumeViewerBase* viewer, const cv::Vec3f& position)
{
    if (!viewer) {
        return;
    }
    viewer->centerOnVolumePoint(position, !isChunkedViewer(viewer));
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
    const QString candidateStr = QString::fromStdString(paths.front().string());
    if (checkedPath) *checkedPath = candidateStr;
    qCInfo(lcSegGrowth) << "Normal grid resolved to" << candidateStr;
    return candidateStr;
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
    tree->setColumnCount(7);
    tree->setHeaderLabels({
        QObject::tr("Fiber"),
        QObject::tr("Source"),
        QObject::tr("Anchor dist"),
        QObject::tr("Snap"),
        QObject::tr("Snap wind"),
        QObject::tr("Snap XYZ"),
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
    _segmentWorkspaceWindow->setCentralWidget(segmentCentralWidget);

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
    for (QDockWidget* dock : {ui.dockWidgetPreprocessing,
                              ui.dockWidgetNormalVis,
                              ui.dockWidgetView,
                              ui.dockWidgetOverlay,
                              ui.dockWidgetRenderSettings,
                              ui.dockWidgetComposite,
                              ui.dockWidgetPostprocessing}) {
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

    _workspaceTabs = new QTabWidget(this);
    _workspaceTabs->setObjectName(QStringLiteral("workspaceTabs"));
    _workspaceTabs->addTab(_segmentWorkspaceWindow, tr("main"));
    _workspaceTabs->addTab(_lasagnaWorkspaceWindow, tr("Lasagna"));
    _workspaceTabs->addTab(_atlasWorkspaceWindow, tr("Atlas"));
    _workspaceTabs->addTab(_fiberSliceWorkspaceWindow, tr("Fiber Slice"));
    _workspaceTabs->addTab(_intersectionsWorkspaceWindow, tr("Intersections"));
    setCentralWidget(_workspaceTabs);
    connect(_workspaceTabs, &QTabWidget::currentChanged, this, &CWindow::scheduleWindowStateSave);
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
    _lineAnnotationController = std::make_unique<LineAnnotationController>(_state,
                                                                           _viewerManager.get(),
                                                                           this,
                                                                           this);
    connect(_lineAnnotationController.get(),
            &LineAnnotationController::atlasCreated,
            this,
            &CWindow::displayAtlasFromDirectory);
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

    // Slice step size label in status bar
    _sliceStepLabel = new QLabel(this);
    _sliceStepLabel->setContentsMargins(4, 0, 4, 0);
    int initialStepSize = _viewerManager->sliceStepSize();
    _sliceStepLabel->setText(tr("Step: %1").arg(initialStepSize));
    _sliceStepLabel->setToolTip(tr("Slice step size: use Shift+G / Shift+H to adjust"));
    statusBar()->addPermanentWidget(_sliceStepLabel);

    _pointsOverlay = std::make_unique<PointsOverlayController>(_state->pointCollection(), this);
    _viewerManager->setPointsOverlay(_pointsOverlay.get());

    _rawPointsOverlay = std::make_unique<RawPointsOverlayController>(_state, this);
    _viewerManager->setRawPointsOverlay(_rawPointsOverlay.get());

    _pathsOverlay = std::make_unique<PathsOverlayController>(this);
    _viewerManager->setPathsOverlay(_pathsOverlay.get());

    _bboxOverlay = std::make_unique<BBoxOverlayController>(this);
    _viewerManager->setBBoxOverlay(_bboxOverlay.get());

    _vectorOverlay = std::make_unique<VectorOverlayController>(_state, this);
    _viewerManager->setVectorOverlay(_vectorOverlay.get());

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
                    statusBar()->showMessage(message, timeout);
                }
            });
    _viewerManager->setVolumeOverlay(_volumeOverlay.get());

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
        const int workspaceIndex = geometry.value(WORKSPACE_TAB_SETTING, 0).toInt();
        if (workspaceIndex >= 0 && workspaceIndex < _workspaceTabs->count()) {
            _workspaceTabs->setCurrentIndex(workspaceIndex);
        }
    }

    for (QDockWidget* dock : { ui.dockWidgetSegmentation,
                               _lasagnaDock,
                               ui.dockWidgetDistanceTransform,
                               ui.dockWidgetVolumes,
                               ui.dockWidgetViewerControls,
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
                                   ui.dockWidgetVolumes,
                                   ui.dockWidgetViewerControls,
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

    const QSize minWindowSize(960, 640);
    setMinimumSize(minWindowSize);
    if (width() < minWindowSize.width() || height() < minWindowSize.height()) {
        resize(std::max(width(), minWindowSize.width()),
               std::max(height(), minWindowSize.height()));
    }

    // If enabled, auto open the last used local volume package.
    if (settings.value(vc3d::settings::project::AUTO_OPEN, vc3d::settings::project::AUTO_OPEN_DEFAULT).toInt() != 0) {

        QStringList files = settings.value(vc3d::settings::project::RECENT).toStringList();

        if (!files.empty() && !files.at(0).isEmpty()) {
            QString path = files[0];
            QTimer::singleShot(0, this, [this, path]() {
                if (_menuController) {
                    _menuController->openVolpkgAt(path);
                }
            });
        }
    }

    // Create application-wide keyboard shortcuts
    fCompositeViewShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::CompositeView), this);
    fCompositeViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fCompositeViewShortcut, &QShortcut::activated, [this]() {
        if (_viewerControlsPanel) {
            _viewerControlsPanel->toggleSegmentationComposite();
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
        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([next](VolumeViewerBase* viewer) {
                if (viewer) {
                    viewer->setShowDirectionHints(next);
                }
            });
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
        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([next](VolumeViewerBase* viewer) {
                if (viewer) {
                    viewer->setShowSurfaceNormals(next);
                }
            });
        }
        statusBar()->showMessage(next ? tr("Surface normals: ON") : tr("Surface normals: OFF"), 2000);
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
            statusBar()->showMessage(
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
        if (_benchReplay->load(_benchOptions.replayPath)) {
            _benchReplay->setWarmPass(_benchOptions.replayWarm);
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

    auto addDock = [menu](QDockWidget* dock) {
        if (dock) {
            menu->addAction(dock->toggleViewAction());
        }
    };

    addDock(ui.dockWidgetVolumes);
    addDock(ui.dockWidgetSegmentation);
    addDock(ui.dockWidgetDistanceTransform);
    addDock(ui.dockWidgetViewerControls);
    addDock(ui.dockWidgetPreprocessing);
    addDock(ui.dockWidgetNormalVis);
    addDock(ui.dockWidgetView);
    addDock(ui.dockWidgetOverlay);
    addDock(ui.dockWidgetRenderSettings);
    addDock(ui.dockWidgetComposite);
    addDock(ui.dockWidgetPostprocessing);
    addDock(_lasagnaDock);
    addDock(_atlasOverviewDock);
    addDock(_atlasSearchDock);
    addDock(_atlasControlDock);
    addDock(_atlasWorkspaceFiberDock);
    addDock(_point_collection_widget);
    addDock(_fiberWidget);
    addDock(_fiberSliceWidget);
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

void CWindow::configureChunkedViewerConnections(CChunkedVolumeViewer* viewer)
{
    if (!viewer) {
        return;
    }

    const bool annotationViewer = viewer->property("vc_viewer_role").toString() == QStringLiteral("annotation");

    connect(_state, &CState::volumeChanged, viewer, &CChunkedVolumeViewer::OnVolumeChanged, Qt::UniqueConnection);
    connect(_state, &CState::volumeClosing, viewer, &CChunkedVolumeViewer::onVolumeClosing, Qt::UniqueConnection);
    if (!annotationViewer) {
        connect(viewer, &CChunkedVolumeViewer::sendVolumeClicked, this, &CWindow::onVolumeClicked, Qt::UniqueConnection);
    } else if (!viewer->property("vc_annotation_focus_bound").toBool()) {
        const std::string surfaceName = viewer->surfName();
        const bool atlasFocusViewer = surfaceName == ATLAS_INTERNAL_SURFACE_NAME;
        const bool lineFocusViewer = surfaceName.rfind("line-", 0) == 0;
        if (atlasFocusViewer || lineFocusViewer) {
            connect(viewer,
                    &CChunkedVolumeViewer::sendVolumeClicked,
                    this,
                    [this, atlasFocusViewer, lineFocusViewer, surfaceName](cv::Vec3f volLoc,
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
                        }
                        if (lineFocusViewer && focused) {
                            QTimer::singleShot(0, this, [this, surfaceName]() {
                                if (_lineAnnotationController) {
                                    _lineAnnotationController->closeFiberWindowForSurface(surfaceName);
                                }
                            });
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

                        QAction* action = menu.addAction(tr("New line annotation"));
                        action->setEnabled(_lineAnnotationController &&
                                           _lineAnnotationController->canLaunchFromViewer(viewer) &&
                                           viewer->sampleSceneVolume(scenePoint).has_value());
                        QAction* selected = menu.exec(globalPos);
                        if (selected == action && action->isEnabled() && _lineAnnotationController) {
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
        connect(_point_collection_widget, &CPointCollectionWidget::sameWrapAnnotationToggled,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationMode, Qt::UniqueConnection);
        connect(_point_collection_widget, &CPointCollectionWidget::sameWrapAnnotationSpacingChanged,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationSpacing, Qt::UniqueConnection);
        connect(_point_collection_widget, &CPointCollectionWidget::sameWrapAnnotationMergeToggled,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationMergeExisting, Qt::UniqueConnection);
        connect(_point_collection_widget, &CPointCollectionWidget::sameWrapAnnotationPathTypeChanged,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationPathType, Qt::UniqueConnection);
        connect(_point_collection_widget, &CPointCollectionWidget::sameWrapAnnotationFilterTypeChanged,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationFilterType, Qt::UniqueConnection);
        connect(_point_collection_widget, &CPointCollectionWidget::sameWrapAnnotationFilterKernelSizeChanged,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationFilterKernelSize, Qt::UniqueConnection);
        connect(_point_collection_widget, &CPointCollectionWidget::sameWrapAnnotationClearRequested,
                viewer, &CChunkedVolumeViewer::clearSameWrapAnnotationPreview, Qt::UniqueConnection);
        viewer->setSameWrapAnnotationSpacing(_point_collection_widget->sameWrapAnnotationSpacing());
        viewer->setSameWrapAnnotationMergeExisting(_point_collection_widget->sameWrapAnnotationMergeEnabled());
        viewer->setSameWrapAnnotationPathType(_point_collection_widget->sameWrapAnnotationPathType());
        viewer->setSameWrapAnnotationFilterKernelSize(_point_collection_widget->sameWrapAnnotationFilterKernelSize());
        viewer->setSameWrapAnnotationFilterType(_point_collection_widget->sameWrapAnnotationFilterType());
        viewer->setSameWrapAnnotationMode(_point_collection_widget->sameWrapAnnotationEnabled());
        viewer->setProperty("vc_points_bound", true);
    }

    const std::string& surfName = viewer->surfName();
    if ((surfName == "seg xz" || surfName == "seg yz") && !viewer->property("vc_axisaligned_bound").toBool()) {
        if (auto* graphicsView = viewer->graphicsView()) {
            graphicsView->setMiddleButtonPanEnabled(!_axisAlignedSliceController->isEnabled());
        }

        connect(viewer, &CChunkedVolumeViewer::sendMousePressVolume,
                this, [this, viewer](cv::Vec3f volLoc, cv::Vec3f /*normal*/, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                    _axisAlignedSliceController->onMousePress(viewer, volLoc, button, modifiers);
                });

        connect(viewer, &CChunkedVolumeViewer::sendMouseMoveVolume,
                this, [this, viewer](cv::Vec3f volLoc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers) {
                    _axisAlignedSliceController->onMouseMove(viewer, volLoc, buttons, modifiers);
                });

        connect(viewer, &CChunkedVolumeViewer::sendMouseReleaseVolume,
                this, [this, viewer](cv::Vec3f /*volLoc*/, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                    _axisAlignedSliceController->onMouseRelease(viewer, button, modifiers);
                });

        viewer->setProperty("vc_axisaligned_bound", true);
    }
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
    if (!mdiArea) {
        return nullptr;
    }
    auto* subWindow = mdiArea->activeSubWindow();
    if (!subWindow) {
        return nullptr;
    }
    return baseViewerFromWidget(subWindow->widget());
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

void CWindow::setVolume(std::shared_ptr<Volume> newvol)
{
    const bool hadVolume = static_cast<bool>(_state->currentVolume());
    POI* existingFocusPoi = _state ? _state->poi("focus") : nullptr;

    // CState handles cache budget and volume ID resolution, and emits volumeChanged
    _state->setCurrentVolume(newvol);

    const bool growthVolumeValid = _state->hasVpkg() && !_state->segmentationGrowthVolumeId().empty() &&
                                   _state->vpkg()->hasVolume(_state->segmentationGrowthVolumeId());
    if (!growthVolumeValid) {
        _state->setSegmentationGrowthVolumeId(_state->currentVolumeId());
        if (_segmentationWidget) {
            _segmentationWidget->setActiveVolume(QString::fromStdString(_state->currentVolumeId()));
        }
    }

    updateNormalGridAvailability();

    if (_state->currentVolume() && _state) {
        auto [w, h, d] = _state->currentVolume()->shapeXyz();
        float x0 = 0, y0 = 0, z0 = 0;
        float x1 = static_cast<float>(w - 1), y1 = static_cast<float>(h - 1), z1 = static_cast<float>(d - 1);

        POI* poi = existingFocusPoi;
        const bool createdPoi = (poi == nullptr);
        if (!poi) {
            poi = new POI;
            poi->n = cv::Vec3f(0, 0, 1);
        }

        if (createdPoi || !hadVolume) {
            poi->p = cv::Vec3f((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, (z0 + z1) * 0.5f);
        } else {
            poi->p[0] = std::clamp(poi->p[0], x0, x1);
            poi->p[1] = std::clamp(poi->p[1], y0, y1);
            poi->p[2] = std::clamp(poi->p[2], z0, z1);
        }
        poi->surfacePtr.reset();

        _state->setPOI("focus", poi);
    }

    _axisAlignedSliceController->applyOrientation(_state ? _state->surface("segmentation").get() : nullptr);
}

bool CWindow::attachVolumeToCurrentPackage(const std::shared_ptr<Volume>& volume,
                                           const QString& preferredVolumeId)
{
    if (!_state || !_state->vpkg() || !volume) {
        return false;
    }

    if (!_state->vpkg()->addVolume(volume)) {
        return false;
    }

    const bool needSurfaceLoad = _surfacePanel && !_surfacePanel->hasSurfaces();
    refreshCurrentVolumePackageUi(preferredVolumeId.isEmpty()
                                      ? QString::fromStdString(volume->id())
                                      : preferredVolumeId,
                                  needSurfaceLoad);
    UpdateView();
    return true;
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
        statusBar()->showMessage(tr("Opened volume package with no volumes."), 5000);
    }

    if (_volumeOverlay) {
        _volumeOverlay->setVolumePkg(_state->vpkg(), _state->vpkgPath());
    }

    {
        const QSignalBlocker blocker{cmbSegmentationDir};
        cmbSegmentationDir->clear();

        auto availableDirs = _state->vpkg()->getAvailableSegmentationDirectories();
        for (const auto& dirName : availableDirs) {
            cmbSegmentationDir->addItem(QString::fromStdString(dirName));
        }

        int currentIndex = cmbSegmentationDir->findText(
            QString::fromStdString(_state->vpkg()->getSegmentationDirectory()));
        if (currentIndex >= 0) {
            cmbSegmentationDir->setCurrentIndex(currentIndex);
        }
    }

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
}

void CWindow::updateNormalGridAvailability()
{
    QString checkedPath;
    const QString path = normalGridDirectoryForVolumePkg(_state->vpkg(), &checkedPath);
    const bool available = !path.isEmpty();

    _normalGridAvailable = available;
    _normalGridPath = path;

    if (_segmentationWidget) {
        _segmentationWidget->setNormalGridAvailable(_normalGridAvailable);
        _segmentationWidget->setNormalGridPath(_normalGridPath);
        QString hint;
        if (_normalGridAvailable) {
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
        statusBar()->showMessage(tr("Restored full view"), 2000);
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
        statusBar()->showMessage(tr("Focused view (Shift+Ctrl+F to restore)"), 2000);
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
    if (!_state) {
        return false;
    }

    POI* focus = _state->poi("focus");
    if (!focus) {
        focus = new POI;
    }

    focus->p = position;
    if (cv::norm(normal) > 0.0) {
        focus->n = normal;
    }
    if (!sourceId.empty()) {
        focus->surfaceId = sourceId;
    } else if (focus->surfaceId.empty()) {
        focus->surfaceId = "segmentation";
    }
    focus->surfacePtr.reset();

    focus->suppressTransientPlaneIntersections = true;
    _state->setPOI("focus", focus);
    recenterSegmentationViewerNear(position);

    // Get surface for orientation - look up by ID
    Surface* orientationSource = _state->surfaceRaw(focus->surfaceId);
    if (!orientationSource) {
        orientationSource = _state->surfaceRaw("segmentation");
    }
    _axisAlignedSliceController->applyOrientation(orientationSource);

    return true;
}

void CWindow::recenterPlaneViewersOn(const cv::Vec3f& position)
{
    if (!_viewerManager) {
        return;
    }

    _viewerManager->forEachBaseViewer([&position](VolumeViewerBase* viewer) {
        if (!viewer) {
            return;
        }

        const std::string name = viewer->surfName();
        if (name == "xy plane" || name == "seg xz" || name == "seg yz" ||
            name.rfind("line_annotation_slice_", 0) == 0) {
            centerViewerOnVolumePointForNavigation(viewer, position);
        }
    });
}

void CWindow::recenterSegmentationViewerNear(const cv::Vec3f& position)
{
    static constexpr float kMaxDistanceVoxels = 100.0f;

    if (!_viewerManager) {
        return;
    }

    auto* viewer = segmentationViewer();
    if (!viewer) {
        return;
    }

    auto activeSurface = _segmentationModule ? _segmentationModule->activeBaseSurfaceShared() : nullptr;
    if (!activeSurface) {
        activeSurface = std::dynamic_pointer_cast<QuadSurface>(_state ? _state->surface("segmentation") : nullptr);
    }
    if (!activeSurface) {
        return;
    }

    auto* patchIndex = _viewerManager->surfacePatchIndex();
    if (!patchIndex || !patchIndex->containsSurface(activeSurface)) {
        return;
    }

    SurfacePatchIndex::PointQuery query;
    query.worldPoint = position;
    query.tolerance = kMaxDistanceVoxels;
    query.surfaces.only = activeSurface;
    auto hit = patchIndex->locate(query);
    if (hit && hit->distance <= kMaxDistanceVoxels) {
        const cv::Vec3f loc = activeSurface->loc(hit->ptr);
        centerViewerOnSurfacePointForNavigation(viewer, {loc[0], loc[1]});
    }
}

bool CWindow::recenterViewersOnCurrentFocus()
{
    if (!_state || !_viewerManager) {
        return false;
    }

    POI* focus = _state->poi("focus");
    if (!focus) {
        return false;
    }

    const cv::Vec3f position = focus->p;
    _viewerManager->forEachBaseViewer([&position](VolumeViewerBase* viewer) {
        if (viewer) {
            centerViewerOnVolumePointForNavigation(viewer, position);
        }
    });

    return true;
}

bool CWindow::centerFocusOnCursor()
{
    if (!_state || !mdiArea) {
        return false;
    }

    const QPoint globalPos = QCursor::pos();
    auto tryCenterFromViewer = [&](VolumeViewerBase* viewer) -> bool {
        if (!viewer) {
            return false;
        }

        auto* viewerObject = viewer->asQObject();
        auto* viewerWidget = qobject_cast<QWidget*>(viewerObject);
        if (viewerWidget && !viewerWidget->isVisible()) {
            return false;
        }

        auto* gv = viewer->graphicsView();
        auto* viewport = gv ? gv->viewport() : nullptr;
        if (!viewport) {
            return false;
        }

        const QPoint viewportPos = viewport->mapFromGlobal(globalPos);
        if (!viewport->rect().contains(viewportPos)) {
            return false;
        }

        const QPointF scenePos = gv->mapToScene(viewportPos);
        cv::Vec3f p = viewer->sceneToVolume(scenePos);
        cv::Vec3f n(0, 0, 1);
        if (auto* plane = dynamic_cast<PlaneSurface*>(viewer->currentSurface())) {
            n = plane->normal(cv::Vec3f(0, 0, 0), {});
        }

        return centerFocusAt(p, n, viewer->surfName());
    };

    // Prefer the viewer actually under the mouse cursor. With tiled MDI
    // windows, the active subwindow can lag behind the hovered viewer, which
    // makes the focus jump use the wrong scene transform.
    if (QWidget* hoveredWidget = QApplication::widgetAt(globalPos)) {
        for (QWidget* widget = hoveredWidget; widget; widget = widget->parentWidget()) {
            if (auto* viewer = baseViewerFromWidget(widget)) {
                if (tryCenterFromViewer(viewer)) {
                    return true;
                }
                break;
            }
        }
    }

    if (_viewerManager) {
        for (auto* viewer : _viewerManager->baseViewers()) {
            if (tryCenterFromViewer(viewer)) {
                return true;
            }
        }
    }

    // Fall back to the active viewer if the cursor isn't currently over any
    // tiled viewport.
    if (auto* subWindow = mdiArea->activeSubWindow()) {
        if (auto* viewer = baseViewerFromWidget(subWindow->widget())) {
            if (tryCenterFromViewer(viewer)) {
                return true;
            }
        }
    }

    // Fallback to stored cursor POI if no active viewer or cursor is outside
    POI* cursor = _state->poi("cursor");
    if (!cursor) {
        return false;
    }

    return centerFocusAt(cursor->p, cursor->n, cursor->surfaceId);
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
        statusBar()->showMessage(enabled ? tr("Mirroring cursor to Surface view enabled")
                                         : tr("Mirroring cursor to Surface view disabled"),
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
                            _lineAnnotationController->openFiberAtControlPoint(fiberId, controlIndex);
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
            const auto manifestPath = _state->vpkg()->selectedLasagnaDatasetPath();
            hasManifest = !manifestPath.empty() && std::filesystem::exists(manifestPath);
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
        fiberItem->setText(6, fiberId == 0 ? tr("not loaded") : QString::number(fiberId));
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
            row->setText(6, fiberId == 0 ? tr("not loaded") : QString::number(fiberId));
            row->setData(0, ATLAS_FIBER_ID_ROLE, QVariant::fromValue<qulonglong>(fiberId));
            row->setData(0, ATLAS_FIBER_PATH_KEY_ROLE, fiberPathKey);
            row->setData(0, ATLAS_CONTROL_INDEX_ROLE, controlIndex);
            row->setData(0, ATLAS_CONTROL_SOURCE_INDEX_ROLE, anchor.sourceIndex);
            if (surfaceCoord) {
                row->setData(0, ATLAS_SURFACE_X_ROLE, (*surfaceCoord)[0]);
                row->setData(0, ATLAS_SURFACE_Y_ROLE, (*surfaceCoord)[1]);
            }
            row->setToolTip(0, tr("Control point: %1").arg(formatAtlasVec3(anchor.world)));
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

void CWindow::remapCurrentAtlas()
{
    if (!_currentAtlasDir || !_state || !_state->vpkg()) {
        QMessageBox::warning(this, tr("Atlas"), tr("Load an atlas before remapping."));
        return;
    }
    auto vpkg = _state->vpkg();
    const std::filesystem::path manifestPath = vpkg->selectedLasagnaDatasetPath();
    if (manifestPath.empty() || !std::filesystem::exists(manifestPath)) {
        QMessageBox::warning(this,
                             tr("Atlas"),
                             tr("Select a local Lasagna dataset before remapping."));
        return;
    }

    const std::filesystem::path atlasDir = *_currentAtlasDir;
    const std::filesystem::path volpkgRoot = vpkg->path().empty()
        ? std::filesystem::path(vpkg->getVolpkgDirectory())
        : vpkg->path().parent_path();

    if (statusBar()) {
        statusBar()->showMessage(tr("Remapping atlas fibers..."), 3000);
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
            [this, watcher, atlasDir]() {
        watcher->deleteLater();
        try {
            const QString summary = watcher->result();
            displayAtlasFromDirectory(atlasDir);
            if (statusBar()) {
                statusBar()->showMessage(tr("Remapped atlas fibers. %1").arg(summary), 7000);
            }
        } catch (const std::exception& ex) {
            refreshAtlasOverviewDocks();
            updateAtlasFiberDocks();
            updateAtlasSearchDocks();
            QMessageBox::warning(
                this,
                tr("Atlas Remap"),
                tr("Could not remap atlas: %1").arg(QString::fromStdString(ex.what())));
        }
    });

    watcher->setFuture(QtConcurrent::run([atlasDir, volpkgRoot, manifestPath]() -> QString {
        std::cerr << "[atlas-remap] start"
                  << " atlas=" << atlasDir.string()
                  << " volpkg_root=" << volpkgRoot.string()
                  << " manifest=" << manifestPath.string()
                  << std::endl;
        vc::lasagna::LasagnaDataset dataset =
            vc::lasagna::LasagnaDataset::open(manifestPath);
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
}

void CWindow::optimizeAtlasSnapCandidates()
{
    if (!_currentAtlasDir || !_state || !_state->vpkg()) {
        QMessageBox::warning(this, tr("Atlas"), tr("Load an atlas before ranking snap candidates."));
        return;
    }
    auto vpkg = _state->vpkg();
    const std::filesystem::path manifestPath = vpkg->selectedLasagnaDatasetPath();
    if (manifestPath.empty() || !std::filesystem::exists(manifestPath)) {
        QMessageBox::warning(this,
                             tr("Atlas"),
                             tr("Select a local Lasagna dataset before ranking snap candidates."));
        return;
    }

    auto& manager = LasagnaServiceManager::instance();
    if (manager.isExternal()) {
        if (!manager.isRunning()) {
            QMessageBox::warning(this,
                                 tr("Atlas"),
                                 tr("Connect the external Lasagna service before ranking snap candidates."));
            return;
        }
    } else if (!manager.ensureServiceRunning()) {
        QMessageBox::warning(this,
                             tr("Atlas"),
                             tr("Failed to start Lasagna service: %1").arg(manager.lastError()));
        return;
    }

    const std::filesystem::path atlasDir = *_currentAtlasDir;
    const std::filesystem::path volpkgRoot = vpkg->path().empty()
        ? std::filesystem::path(vpkg->getVolpkgDirectory())
        : vpkg->path().parent_path();
    const std::string serviceManifestPath = manager.isExternal()
        ? manifestPath.filename().generic_string()
        : manifestPath.string();
    if (statusBar()) {
        statusBar()->showMessage(tr("Ranking atlas snap candidates..."), 3000);
    }
    for (auto* dock : {_atlasOverviewDock, _atlasWorkspaceOverviewDock}) {
        if (!dock || !dock->widget()) {
            continue;
        }
        if (auto* button = dock->widget()->findChild<QPushButton*>(
                QStringLiteral("atlasRankSnapButton"))) {
            button->setEnabled(false);
        }
    }

    auto* watcher = new QFutureWatcher<vc::atlas::AtlasSnapOptimizeReport>(this);
    connect(watcher,
            &QFutureWatcher<vc::atlas::AtlasSnapOptimizeReport>::finished,
            this,
            [this, watcher, atlasDir]() {
        watcher->deleteLater();
        try {
            const vc::atlas::AtlasSnapOptimizeReport report = watcher->result();
            refreshAtlasOverviewDocks();
            displayAtlasFromDirectory(atlasDir);
            if (statusBar()) {
                statusBar()->showMessage(
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
        } catch (const std::exception& ex) {
            refreshAtlasOverviewDocks();
            updateAtlasSearchDocks();
            const QString message = extractFutureExceptionMessage(ex);
            std::cerr << "[atlas-snap] rank failed: "
                      << message.toStdString() << std::endl;
            QMessageBox::warning(
                this,
                tr("Atlas Snap Candidates"),
                tr("Could not rank snap candidates: %1").arg(message));
        }
    });

    QPointer<CWindow> self(this);
    watcher->setFuture(QtConcurrent::run([self,
                                           atlasDir,
                                           volpkgRoot,
                                           manifestPath,
                                           serviceManifestPath]() {
        try {
            std::cerr << "[atlas-snap] rank start"
                      << " atlas=" << atlasDir.string()
                      << " volpkg_root=" << volpkgRoot.string()
                      << " manifest=" << manifestPath.string()
                      << " service_manifest=" << serviceManifestPath
                      << std::endl;
            vc::lasagna::LasagnaDataset dataset =
                vc::lasagna::LasagnaDataset::open(manifestPath);
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
            const auto ranker =
                [self, serviceManifestPath](
                    const nlohmann::json& request,
                    const vc::atlas::AtlasSnapPairRankProgress& onResult) -> nlohmann::json {
                    if (!self) {
                        throw std::runtime_error("window closed before rank request completed");
                    }
                    nlohmann::json serviceRequest = request;
                    serviceRequest["manifest"] = serviceManifestPath;
                    const size_t jobCount =
                        serviceRequest.value("jobs", nlohmann::json::array()).size();
                    std::cerr << "[atlas-snap] requesting laplace rank jobs="
                              << jobCount
                              << " service_manifest=" << serviceManifestPath
                              << std::endl;
                    QJsonObject qtRequest = toQtJsonObject(serviceRequest);
                    QJsonObject qtResponse;
                    QString error;
                    QMetaObject::invokeMethod(
                        self.data(),
                        [&qtRequest, &qtResponse, &error, &onResult]() {
                            QEventLoop loop;
                            LasagnaServiceManager::instance().rankLaplaceSnapPairs(
                                qtRequest,
                                [&](const QJsonObject& response) {
                                    qtResponse = response;
                                    loop.quit();
                                },
                                [&](const QString& message) {
                                    error = message;
                                    loop.quit();
                                },
                                [&](int index, const QJsonObject& result) {
                                    if (!onResult) {
                                        return;
                                    }
                                    try {
                                        onResult(static_cast<size_t>(index),
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
                            loop.exec();
                        },
                        Qt::BlockingQueuedConnection);
                    if (!error.isEmpty()) {
                        std::cerr << "[atlas-snap] laplace rank failed: "
                                  << error.toStdString() << std::endl;
                        throw std::runtime_error(error.toStdString());
                    }
                    std::cerr << "[atlas-snap] laplace rank response received"
                              << std::endl;
                    return fromQtJsonObject(qtResponse);
                };

            vc::atlas::AtlasSnapOptimizeReport report =
                vc::atlas::optimizeAtlasPredSnapCandidates(
                    atlasDir,
                    volpkgRoot,
                    manifestPath,
                    sampler,
                    ranker,
                    options);
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
            return report;
        } catch (const std::exception& ex) {
            std::cerr << "[atlas-snap] rank exception: "
                      << ex.what() << std::endl;
            throw;
        } catch (...) {
            std::cerr << "[atlas-snap] rank exception: unknown non-standard exception"
                      << std::endl;
            throw;
        }
    }));
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
            progress->setFormat(tr("Canceling: phase %1/4 %2 (%3 / %4)")
                                    .arg(atlasSearchPhaseNumber(_atlasSearchProgressPhase))
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
        ? tr("Canceling: phase %1/4 %2 (%3 / %4)")
              .arg(atlasSearchPhaseNumber(phase))
              .arg(action)
              .arg(static_cast<qulonglong>(_atlasSearchPhaseCompleted))
              .arg(static_cast<qulonglong>(total))
        : tr("Phase %1/4: %2 (%3 / %4)")
              .arg(atlasSearchPhaseNumber(phase))
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
}

void CWindow::startAtlasFiberIntersectionSearch()
{
    if (!_lineAnnotationController) {
        updateAtlasSearchDocks();
        return;
    }

    vc::atlas::FiberIntersectionBroadPhaseOptions broad;
    int searchMode = ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS;
    QStringList requiredTags;
    auto readSearchControls = [&](QDockWidget* dock) {
        if (!dock || !dock->widget()) {
            return false;
        }
        bool found = false;
        if (auto* combo = dock->widget()->findChild<QComboBox*>(
                QStringLiteral("atlasSearchTypeCombo"))) {
            searchMode = combo->currentData().toInt();
            found = true;
        }
        if (auto* tagFilter = dock->widget()->findChild<QLineEdit*>(
                QStringLiteral("atlasSearchTagFilterEdit"))) {
            requiredTags = atlasSearchTagList(tagFilter->text());
            found = true;
        }
        if (auto* spin = dock->widget()->findChild<QDoubleSpinBox*>(
                QStringLiteral("atlasSearchMaxDistanceSpin"))) {
            broad.maxDistance = spin->value();
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
    if (searchMode != ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS &&
        searchMode != ATLAS_SEARCH_MODE_NON_ATLAS_ONLY) {
        QMessageBox::warning(this,
                             tr("Atlas Object Search"),
                             tr("Unsupported atlas search type."));
        return;
    }

    if (!_currentAtlasDir && searchMode == ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS) {
        QMessageBox::information(this,
                                 tr("Atlas Object Search"),
                                 tr("Select an atlas, or choose \"Non-atlas fibers only\"."));
        return;
    }

    vc::atlas::Atlas atlas;
    bool haveAtlas = false;
    if (_currentAtlasDir) {
        try {
            atlas = vc::atlas::Atlas::load(*_currentAtlasDir);
            haveAtlas = true;
        } catch (const std::exception& ex) {
            QMessageBox::warning(this,
                                 tr("Atlas Object Search"),
                                 tr("Could not load selected atlas: %1")
                                     .arg(QString::fromStdString(ex.what())));
            return;
        }
    }

    const auto fiberSnapshots = _lineAnnotationController->fiberSnapshotsFromStorageWithPaths();
    if (fiberSnapshots.empty()) {
        updateAtlasSearchDocks();
        return;
    }

    std::vector<std::string> atlasFiberPaths = haveAtlas
        ? vc::atlas::atlasMappedFiberPathKeys(atlas)
        : std::vector<std::string>{};
    if (searchMode == ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS && atlasFiberPaths.empty()) {
        QMessageBox::information(this,
                                 tr("Atlas Object Search"),
                                 tr("Selected atlas has no saved fiber mappings."));
        return;
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
                snapshot.fiberPath, std::move(previewFiber), snapshot.storedFiberId, snapshot.tags});
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
    if (!requiredTags.isEmpty()) {
        auto keepTagged = [&snapshotsByRuntimeId, &requiredTags](std::vector<uint64_t>& ids) {
            ids.erase(std::remove_if(ids.begin(),
                                     ids.end(),
                                     [&snapshotsByRuntimeId, &requiredTags](uint64_t id) {
                                         const auto it = snapshotsByRuntimeId.find(id);
                                         return it == snapshotsByRuntimeId.end() ||
                                                !atlasSearchFiberMatchesTags(it->second.tags,
                                                                             requiredTags);
                                     }),
                      ids.end());
        };
        keepTagged(sourceFiberIds);
        keepTagged(targetFiberIds);
    }
    if (sourceFiberIds.empty()) {
        QMessageBox::information(this,
                                 tr("Atlas Object Search"),
                                 searchMode == ATLAS_SEARCH_MODE_NON_ATLAS_ONLY
                                     ? tr("No saved non-atlas fibers match the search filters.")
                                     : tr("None of the selected atlas fibers are available in saved fiber files or match the search filters."));
        return;
    }
    if (targetFiberIds.empty()) {
        QMessageBox::information(this,
                                 tr("Atlas Object Search"),
                                 tr("No saved non-atlas fibers are available to search or match the search filters."));
        return;
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
    if (_state && _state->vpkg()) {
        lasagnaManifestPath = _state->vpkg()->selectedLasagnaDatasetPath();
    }
    if (lasagnaManifestPath.empty()) {
        QMessageBox::warning(this,
                             tr("Atlas Object Search"),
                             tr("Select a local Lasagna dataset before searching. "
                                "Intersection distance is measured in grad_mag winding-integral space."));
        return;
    }

    clearAtlasSearchPreviewState();
    _atlasSearchFiberSnapshotsByRuntimeId = std::move(snapshotsByRuntimeId);
    _atlasSearchLasagnaManifestPath = lasagnaManifestPath;
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

    const auto searchStart = std::chrono::steady_clock::now();
    auto* watcher = new QFutureWatcher<std::vector<vc::atlas::FiberIntersectionResult>>(this);
    connect(watcher,
            &QFutureWatcher<std::vector<vc::atlas::FiberIntersectionResult>>::finished,
            this,
            [this,
             watcher,
             sourceFiberIds,
             targetFiberIds,
             fiberPathById = std::move(fiberPathById),
             cancelFlag = _atlasSearchCancelFlag,
             searchStart,
             debugSearch]() {
                const auto results = watcher->result();
                if (debugSearch) {
                    qInfo().noquote() << QStringLiteral("[atlas-search] finished source_ids=%1 target_ids=%2 result_count=%3")
                        .arg(atlasSearchIdListString(sourceFiberIds),
                             atlasSearchIdListString(targetFiberIds))
                        .arg(static_cast<int>(results.size()));
                    for (const auto& result : results) {
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
                watcher->deleteLater();
                const bool canceled = cancelFlag && cancelFlag->load(std::memory_order_relaxed);
                if (!canceled && !_atlasSearchCancelRequested) {
                    populateAtlasSearchResults(results);
                    if (statusBar()) {
                        statusBar()->showMessage(tr("Atlas object search found %1 result(s)")
                                                     .arg(static_cast<int>(results.size())),
                                                 3000);
                    }
                } else {
                    qInfo().noquote() << QStringLiteral("[atlas-search] phase=4 finishing_results start total=1");
                    updateAtlasSearchProgress(vc::atlas::AtlasSearchProgressPhase::FinishResults,
                                              1,
                                              1);
                    qInfo().noquote() << QStringLiteral("[atlas-search] phase=4 finishing_results end completed=1 total=1");
                    for (auto* dock : {_atlasSearchDock, _atlasWorkspaceSearchDock}) {
                        if (!dock || !dock->widget()) {
                            continue;
                        }
                        if (auto* progress = dock->widget()->findChild<QProgressBar*>(
                                QStringLiteral("atlasSearchProgressBar"))) {
                            progress->setFormat(tr("Canceled"));
                        }
                    }
                    if (statusBar()) {
                        statusBar()->showMessage(tr("Atlas object search canceled"), 3000);
                    }
                }
                const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - searchStart).count();
                qInfo().noquote() << QStringLiteral("[atlas-search] final result_count=%1 canceled=%2 elapsed_ms=%3")
                    .arg(static_cast<qulonglong>(results.size()))
                    .arg(canceled || _atlasSearchCancelRequested ? QStringLiteral("yes") : QStringLiteral("no"))
                    .arg(static_cast<qlonglong>(elapsedMs));
                _atlasSearchCancelRequested = false;
                if (_atlasSearchCancelFlag == cancelFlag) {
                    _atlasSearchCancelFlag.reset();
                }
                updateAtlasSearchDocks();
            });

    vc::atlas::FiberIntersectionCache* cache = &_fiberIntersectionCache;
    QPointer<CWindow> self(this);
    auto cancelFlag = _atlasSearchCancelFlag;
    watcher->setFuture(QtConcurrent::run([fibers = std::move(fibers),
                                          sourceFiberIds = std::move(sourceFiberIds),
                                          targetFiberIds = std::move(targetFiberIds),
                                          lasagnaManifestPath = std::move(lasagnaManifestPath),
                                          cache,
                                          broad,
                                          ceres,
                                          self,
                                          cancelFlag]() mutable {
        vc::lasagna::LasagnaDataset dataset =
            vc::lasagna::LasagnaDataset::open(lasagnaManifestPath);
        vc::lasagna::LasagnaNormalSampler windingSampler(dataset);
        bool phase2Started = false;
        bool phase2Ended = false;
        bool phase3Started = false;
        bool phase3Ended = false;
        auto progressCallback = [self,
                                 cancelFlag,
                                 &phase2Started,
                                 &phase2Ended,
                                 &phase3Started,
                                 &phase3Ended](vc::atlas::AtlasSearchProgressPhase phase,
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
        return vc::atlas::searchFiberIntersections(fibers,
                                                   sourceFiberIds,
                                                   targetFiberIds,
                                                   cache,
                                                   broad,
                                                   ceres,
                                                   &windingSampler,
                                                   std::move(progressCallback),
                                                   std::move(cancelCallback));
    }));
}

void CWindow::populateAtlasSearchResults(const std::vector<vc::atlas::FiberIntersectionResult>& results)
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

    auto displayedDistance = [](const vc::atlas::FiberIntersectionResult& result) {
        return result.windingDistance;
    };
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

    std::vector<vc::atlas::FiberIntersectionResult> sortedResults = results;
    std::sort(sortedResults.begin(), sortedResults.end(), [&](const auto& a, const auto& b) {
        const double da = displayedDistance(a);
        const double db = displayedDistance(b);
        if (da != db) return da < db;
        if (a.sourceFiberId != b.sourceFiberId) return a.sourceFiberId < b.sourceFiberId;
        return a.targetFiberId < b.targetFiberId;
    });
    _atlasSearchResults = sortedResults;
    ++_atlasSearchPreviewGeneration;
    _atlasSearchHoveredResult.reset();
    _atlasSearchSelectedResults.clear();
    _atlasSearchPreviewRequestedResults.clear();

    const std::size_t finishTotal = std::max<std::size_t>(1, _atlasSearchResults.size());
    qInfo().noquote() << QStringLiteral("[atlas-search] phase=4 finishing_results start total=%1")
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
            row->setText(0, QString::number(displayedDistance(result), 'f', 3));
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
                if (displayedDistance(result) < ATLAS_SEARCH_CLOSE_WINDING_THRESHOLD) {
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
                          const double da = displayedDistance(ra);
                          const double db = displayedDistance(rb);
                          if (da != db) return da < db;
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
    qInfo().noquote() << QStringLiteral("[atlas-search] phase=4 finishing_results end completed=%1 total=%2")
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
    _atlasSearchFiberSnapshotsByRuntimeId.clear();
    _atlasSearchLasagnaManifestPath.reset();
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
        const vc::atlas::Atlas atlas = vc::atlas::Atlas::load(*_currentAtlasDir);
        const std::filesystem::path basePath = *_currentAtlasDir / atlas.metadata.baseMeshPath;
        QuadSurface baseSurface(basePath);
        const auto* points = baseSurface.rawPointsPtr();
        if (!points || points->empty() || points->cols <= 0) {
            throw std::runtime_error("atlas base mesh has no valid grid");
        }
        const vc::atlas::AtlasDisplayRange displayRange =
            vc::atlas::atlasDisplayRange(atlas, vc::atlas::atlasHorizontalPeriodColumns(baseSurface));
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
                vc::lasagna::LasagnaDataset::open(manifestPath);
            vc::lasagna::LasagnaNormalSampler sampler(dataset);
            SurfacePatchIndex baseIndex;
            baseIndex.rebuild({baseSurface});
            const int periodColumns = vc::atlas::atlasHorizontalPeriodColumns(*baseSurface);

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
            const auto* targetMapping = findAtlasMappingForPath(atlas, targetInput.fiberPath);
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
            atlas.links.push_back(std::move(link));
            vc::atlas::layoutAtlasObjects(atlas, periodColumns);

            const auto* laidOutTarget = findAtlasMappingForPath(atlas, targetInput.fiberPath);
            if (!laidOutTarget) {
                throw std::runtime_error("Could not find laid-out preview target fiber");
            }
            return AtlasOverlayController::SearchPreviewFiber{
                sortedResultIndex,
                *laidOutTarget,
            };
        } catch (const std::exception& ex) {
            qWarning().noquote() << QStringLiteral("Could not build atlas search preview line: %1")
                .arg(QString::fromStdString(ex.what()));
            return std::nullopt;
        }
    }));
}

void CWindow::displayAtlasFromDirectory(const std::filesystem::path& atlasDir)
{
    try {
        if (!_atlasWorkspaceWindow || !_atlasViewer) {
            createAtlasWorkspace();
        }
        auto vpkg = _state ? _state->vpkg() : nullptr;
        if (!vpkg) {
            throw std::runtime_error("No volume package is loaded");
        }
        const std::filesystem::path manifestPath = vpkg->selectedLasagnaDatasetPath();
        if (manifestPath.empty()) {
            throw std::runtime_error(
                "No local Lasagna normal dataset is selected; atlas pred-snap attachments are required");
        }
        if (!std::filesystem::exists(manifestPath)) {
            throw std::runtime_error("Selected Lasagna normal dataset does not exist");
        }
        const std::filesystem::path volpkgRoot = vpkg->path().empty()
            ? std::filesystem::path(vpkg->getVolpkgDirectory())
            : vpkg->path().parent_path();
        auto atlas = vc::atlas::Atlas::load(atlasDir, volpkgRoot);
        vc::lasagna::LasagnaDataset dataset =
            vc::lasagna::LasagnaDataset::open(manifestPath);
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

        const vc::atlas::AtlasDisplayRange displayRange =
            vc::atlas::atlasDisplayRange(atlas, vc::atlas::atlasHorizontalPeriodColumns(*baseSurface));
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
            if (auto* tree = dock->widget()->findChild<QTreeWidget*>(QStringLiteral("atlasSearchResultTree"))) {
                tree->clear();
            }
            if (auto* progress = dock->widget()->findChild<QProgressBar*>(QStringLiteral("atlasSearchProgressBar"))) {
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
            statusBar()->showMessage(tr("Displayed atlas %1")
                                         .arg(QString::fromStdString(atlas.metadata.name)),
                                     3000);
        }
    } catch (const std::exception& ex) {
        if (vc::atlas::atlasLoadErrorRequiresRebuild(ex)) {
            const auto choice = QMessageBox::question(
                this,
                tr("Atlas Rebuild Required"),
                tr("This atlas was saved with an older or stale mapping format and must be rebuilt "
                   "from its unchanged source fiber JSON using the selected Lasagna normal dataset.\n\n"
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
                const std::filesystem::path manifestPath = vpkg->selectedLasagnaDatasetPath();
                if (manifestPath.empty()) {
                    throw std::runtime_error("No local Lasagna normal dataset is selected");
                }
                if (!std::filesystem::exists(manifestPath)) {
                    throw std::runtime_error("Selected Lasagna normal dataset does not exist");
                }
                const std::filesystem::path volpkgRoot = vpkg->path().empty()
                    ? std::filesystem::path{}
                    : vpkg->path().parent_path();
                vc::lasagna::LasagnaDataset dataset =
                    vc::lasagna::LasagnaDataset::open(manifestPath);
                vc::lasagna::LasagnaNormalSampler sampler(dataset);
                if (!sampler.hasPredDtChannel()) {
                    throw std::runtime_error(
                        "Selected Lasagna dataset has no pred_dt channel; atlas pred-snap attachments are required");
                }
                vc::atlas::rebuildAtlasFromSourceFibers(
                    atlasDir, volpkgRoot, sampler);
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
    ui.tabSegment->setLayout(aWidgetLayout);

    mdiArea = new QMdiArea(ui.tabSegment);
    aWidgetLayout->addWidget(mdiArea);

    // Ensure the viewer's graphics view gets focus when subwindow is activated
    connect(mdiArea, &QMdiArea::subWindowActivated, [](QMdiSubWindow* subWindow) {
        if (subWindow) {
            if (auto* viewer = dynamic_cast<VolumeViewerBase*>(subWindow->widget())) {
                if (auto* graphicsView = viewer->graphicsView()) {
                    graphicsView->setFocus();
                }
            }
        }
    });

    {
        newConnectedViewer("seg xz", tr("Segmentation XZ"), mdiArea)->setIntersects({"segmentation"});
        newConnectedViewer("seg yz", tr("Segmentation YZ"), mdiArea)->setIntersects({"segmentation"});
        newConnectedViewer("xy plane", tr("XY / Slices"), mdiArea)->setIntersects({"segmentation"});
        newConnectedViewer("segmentation", tr("Surface"), mdiArea)->setIntersects({"seg xz","seg yz"});
    }
    mdiArea->tileSubWindows();

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
                auto surf = _state->vpkg()->getSurface(segmentId.toStdString());
                if (!surf) {
                    return;
                }
                const QString path = absoluteSegmentPathForClipboard(surf->path, _state->vpkg());
                QApplication::clipboard()->setText(path);
                statusBar()->showMessage(tr("Copied segment path to clipboard: %1").arg(path), 3000);
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
                auto surf = _state->vpkg()->getSurface(segmentId.toStdString());
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
                    statusBar()->showMessage(
                        tr("Recalculated area for %1 segment(s).").arg(okCount), 5000);
                }
                if (failCount > 0) {
                    QMessageBox::warning(this, tr("Area Recalculation"),
                        tr("Updated: %1\nSkipped: %2\n\n%3").arg(okCount).arg(failCount).arg(skippedIds.join("\n")));
                }
            });
    connect(_surfacePanel.get(), &SurfacePanelController::statusMessageRequested,
            this, [this](const QString& message, int timeoutMs) {
                statusBar()->showMessage(message, timeoutMs);
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
                statusBar()->showMessage(text, timeout);
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
    connect(_segmentationCommandHandler.get(), &SegmentationCommandHandler::statusMessage,
            this, &CWindow::onShowStatusMessage);

    _fileWatcher->setSurfacePanel(_surfacePanel.get());
    _fileWatcher->setSegmentationModule(_segmentationModule.get());
    _fileWatcher->setTreeWidget(treeWidgetSurfaces);

    connect(_segmentationWidget, &SegmentationWidget::copyWithNtRequested,
            this, &CWindow::onCopyWithNtRequested);
    connect(_segmentationWidget, &SegmentationWidget::volumeSelectionChanged, this, [this](const QString& volumeId) {
        if (!_state->vpkg()) {
            statusBar()->showMessage(tr("No volume package loaded."), 4000);
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
            statusBar()->showMessage(tr("Using volume '%1' for surface growth.").arg(volumeId), 2500);
        } catch (const std::out_of_range&) {
            statusBar()->showMessage(tr("Volume '%1' not found in this package.").arg(volumeId), 4000);
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
        statusBar()->showMessage(tr("Lasagna optimization stop requested."), 3000);
    });
    connect(&LasagnaServiceManager::instance(), &LasagnaServiceManager::serviceStarted,
            this, &CWindow::updateAtlasFiberDocks);
    connect(&LasagnaServiceManager::instance(), &LasagnaServiceManager::serviceStopped,
            this, &CWindow::updateAtlasFiberDocks);

    // Add only the segments placed by lasagna instead of rescanning every surface.
    connect(&LasagnaServiceManager::instance(), &LasagnaServiceManager::resultsPlaced,
            this, [this](const QString& outputDir, const QStringList& segmentNames) {
        statusBar()->showMessage(
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
    _fiberController->setMdiArea(mdiArea);

    _fiberWidget = new CFiberWidget(this);
    _fiberWidget->setObjectName("fiberDock");
    _segmentWorkspaceWindow->addDockWidget(Qt::RightDockWidgetArea, _fiberWidget);

    if (_fiberSliceWorkspaceWindow) {
        _fiberSliceWidget = new CFiberWidget(_fiberSliceWorkspaceWindow);
        _fiberSliceWidget->setObjectName(QStringLiteral("fiberSliceDock"));
        _fiberSliceWorkspaceWindow->addDockWidget(Qt::LeftDockWidgetArea, _fiberSliceWidget);
    }

    if (_lineAnnotationController) {
        auto updateFiberList = [this](const std::vector<LineAnnotationController::FiberSummary>& fibers) {
            std::vector<CFiberWidget::FiberEntry> entries;
            entries.reserve(fibers.size());
            for (const auto& fiber : fibers) {
                entries.push_back(CFiberWidget::FiberEntry{
                    fiber.id,
                    fiber.name,
                    fiber.controlPointCount,
                    fiber.linePointCount,
                    fiber.lengthVx,
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
            updateAtlasFiberDocks();
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
                    &CFiberWidget::deleteFibersRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::deleteFibers);
            connect(widget,
                    &CFiberWidget::renameFiberFileRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::renameFiberFile);
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
        .compositeScrollArea = ui.scrollAreaComposite,
        .compositeContents = ui.dockWidgetCompositeContents,
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
        .methodScaleLabel = ui.lblMethodScale,
        .methodScale = ui.sliderMethodScale,
        .methodScaleValue = ui.lblMethodScaleValue,
        .methodParamLabel = ui.lblMethodParam,
        .methodParam = ui.sliderMethodParam,
        .methodParamValue = ui.lblMethodParamValue,
        .blExtinctionLabel = ui.lblBLExtinction,
        .blExtinction = ui.spinBLExtinction,
        .blEmissionLabel = ui.lblBLEmission,
        .blEmission = ui.spinBLEmission,
        .blAmbientLabel = ui.lblBLAmbient,
        .blAmbient = ui.spinBLAmbient,
        .lightingEnabled = ui.chkLightingEnabled,
        .lightAzimuthLabel = ui.lblLightAzimuth,
        .lightAzimuth = ui.spinLightAzimuth,
        .lightElevationLabel = ui.lblLightElevation,
        .lightElevation = ui.spinLightElevation,
        .lightDiffuseLabel = ui.lblLightDiffuse,
        .lightDiffuse = ui.spinLightDiffuse,
        .lightAmbientLabel = ui.lblLightAmbient,
        .lightAmbient = ui.spinLightAmbient,
        .useVolumeGradients = ui.chkUseVolumeGradients,
        .shadowStepsLabel = ui.lblShadowSteps,
        .shadowSteps = ui.spinShadowSteps,
        .rakingEnabled = ui.chkRakingEnabled,
        .rakingAzimuthLabel = ui.lblRakingAzimuth,
        .rakingAzimuth = ui.spinRakingAzimuth,
        .rakingElevationLabel = ui.lblRakingElevation,
        .rakingElevation = ui.spinRakingElevation,
        .rakingStrengthLabel = ui.lblRakingStrength,
        .rakingStrength = ui.spinRakingStrength,
        .rakingDepthLabel = ui.lblRakingDepth,
        .rakingDepthScale = ui.spinRakingDepthScale,
        .preNormalizeLayers = ui.chkPreNormalizeLayers,
        .preHistEqLayers = ui.chkPreHistEqLayers,
        .preTfEnabled = ui.chkPreTfEnabled,
        .preTfX1 = ui.spinPreTfX1,
        .preTfY1 = ui.spinPreTfY1,
        .preTfKnot2Label = ui.lblPreTfKnot2,
        .preTfX2 = ui.spinPreTfX2,
        .preTfY2 = ui.spinPreTfY2,
        .postTfEnabled = ui.chkPostTfEnabled,
        .postTfX1 = ui.spinPostTfX1,
        .postTfY1 = ui.spinPostTfY1,
        .postTfKnot2Label = ui.lblPostTfKnot2,
        .postTfX2 = ui.spinPostTfX2,
        .postTfY2 = ui.spinPostTfY2,
        .dvrAmbientLabel = ui.lblDvrAmbient,
        .dvrAmbient = ui.spinDvrAmbient,
        .pbrRoughnessLabel = ui.lblPbrRoughness,
        .pbrRoughness = ui.spinPbrRoughness,
        .pbrMetallicLabel = ui.lblPbrMetallic,
        .pbrMetallic = ui.spinPbrMetallic,
        .planeCompositeXY = ui.chkPlaneCompositeXY,
        .planeCompositeXZ = ui.chkPlaneCompositeXZ,
        .planeCompositeYZ = ui.chkPlaneCompositeYZ,
        .planeLayersFront = ui.spinPlaneLayersFront,
        .planeLayersBehind = ui.spinPlaneLayersBehind,
        .renderSettingsScrollArea = ui.scrollAreaRenderSettings,
        .renderSettingsContents = ui.dockWidgetRenderSettingsContents,
        .normalVisualizationContents = ui.dockWidgetNormalVisContents,
        .showSurfaceNormals = ui.chkShowSurfaceNormals,
        .normalArrowLengthLabel = ui.labelNormalArrowLength,
        .normalArrowLengthSlider = ui.sliderNormalArrowLength,
        .normalArrowLengthValueLabel = ui.labelNormalArrowLengthValue,
        .normalMaxArrowsLabel = ui.labelNormalMaxArrows,
        .normalMaxArrowsSlider = ui.sliderNormalMaxArrows,
        .normalMaxArrowsValueLabel = ui.labelNormalMaxArrowsValue,
        .preprocessingScrollArea = ui.scrollAreaPreprocessing,
        .preprocessingContents = ui.dockWidgetPreprocessingContents,
        .isoCutoff = ui.sliderIsoCutoff,
        .isoCutoffValue = ui.lblIsoCutoffValue,
        .postprocessingScrollArea = ui.scrollAreaPostprocessing,
        .postprocessingContents = ui.dockWidgetPostprocessingContents,
        .baseColormap = ui.baseColormapSelect,
        .stretchValuesPost = ui.chkStretchValuesPost,
        .removeSmallComponents = ui.chkRemoveSmallComponents,
        .minComponentSizeLabel = ui.lblMinComponentSize,
        .minComponentSize = ui.spinMinComponentSize,
        .claheEnabled = ui.chkClaheEnabled,
        .claheClipLimitLabel = ui.lblClaheClipLimit,
        .claheClipLimit = ui.spinClaheClipLimit,
        .claheTileSizeLabel = ui.lblClaheTileSize,
        .claheTileSize = ui.spinClaheTileSize,
        .zoomInButton = ui.btnZoomIn,
        .zoomOutButton = ui.btnZoomOut,
        .sliceStepSizeSpin = ui.spinSliceStepSize,
        .volumeWindowContainer = ui.volumeWindowContainer,
        .overlayWindowContainer = ui.overlayWindowContainer,
        .intersectionOpacitySpin = ui.spinIntersectionOpacity,
        .intersectionThicknessSpin = ui.doubleSpinIntersectionThickness,
    };
    _viewerControlsPanel = std::make_unique<ViewerControlsPanel>(viewerControlsUi,
                                                                 _viewerManager.get(),
                                                                 ui.dockWidgetViewerControlsContents);
    connect(_viewerControlsPanel.get(), &ViewerControlsPanel::zoomInRequested,
            this, &CWindow::onZoomIn);
    connect(_viewerControlsPanel.get(), &ViewerControlsPanel::zoomOutRequested,
            this, &CWindow::onZoomOut);
    connect(_viewerManager.get(), &ViewerManager::sliceStepSizeChanged,
            this, &CWindow::onSliceStepSizeChanged);
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

    auto hideLegacyViewerDocks = [this]() {
        for (QDockWidget* dock : { ui.dockWidgetPreprocessing,
                                   ui.dockWidgetNormalVis,
                                   ui.dockWidgetView,
                                   ui.dockWidgetOverlay,
                                   ui.dockWidgetRenderSettings,
                                   ui.dockWidgetComposite,
                                   ui.dockWidgetPostprocessing }) {
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
            .opacitySpin = ui.overlayOpacitySpin,
            .thresholdSpin = ui.overlayThresholdSpin,
        };
        _volumeOverlay->setUi(overlayUi);
        if (_viewerControlsPanel) {
            _viewerControlsPanel->setOverlayWindowAvailable(_volumeOverlay->hasOverlaySelection());
        }
    }

    // Setup surface overlay controls
    connect(ui.chkSurfaceOverlay, &QCheckBox::toggled, [this](bool checked) {
        if (!_viewerManager) return;
        _viewerManager->forEachBaseViewer([checked](VolumeViewerBase* viewer) {
            viewer->setSurfaceOverlayEnabled(checked);
        });
        ui.surfaceOverlaySelect->setEnabled(checked);
        ui.spinOverlapThreshold->setEnabled(checked);
    });

    connect(ui.spinOverlapThreshold, qOverload<double>(&QDoubleSpinBox::valueChanged), [this](double value) {
        if (!_viewerManager) return;
        _viewerManager->forEachBaseViewer([value](VolumeViewerBase* viewer) {
            viewer->setSurfaceOverlapThreshold(static_cast<float>(value));
        });
    });

    // Initially disable surface overlay controls
    ui.surfaceOverlaySelect->setEnabled(false);
    ui.spinOverlapThreshold->setEnabled(false);

    // Initialize surface overlay dropdown (will be populated when surfaces load)
    updateSurfaceOverlayDropdown();



    connect(
        volSelect, &QComboBox::currentIndexChanged, [this](const int& index) {
            auto vpkg = _state->vpkg();
            if (vpkg && index >= 0) {
                std::shared_ptr<Volume> newVolume;
                try {
                    newVolume = vpkg->volume(volSelect->currentData().toString().toStdString());
                } catch (const std::out_of_range& e) {
                    QMessageBox::warning(this, "Error", "Could not load volume.");
                    return;
                }
                setVolume(newVolume);
            }
        });

    auto* filterDropdown = ui.btnFilterDropdown;
    auto* cmbPointSetFilter = ui.cmbPointSetFilter;
    auto* btnPointSetFilterAll = ui.btnPointSetFilterAll;
    auto* btnPointSetFilterNone = ui.btnPointSetFilterNone;
    auto* cmbPointSetFilterMode = new QComboBox();
    cmbPointSetFilterMode->addItem("Any (OR)");
    cmbPointSetFilterMode->addItem("All (AND)");
    ui.pointSetFilterLayout->insertWidget(1, cmbPointSetFilterMode);

    SurfacePanelController::FilterUiRefs filterUi;
    filterUi.dropdown = filterDropdown;
    filterUi.pointSet = cmbPointSetFilter;
    filterUi.pointSetAll = btnPointSetFilterAll;
    filterUi.pointSetNone = btnPointSetFilterNone;
    filterUi.pointSetMode = cmbPointSetFilterMode;
    filterUi.surfaceIdFilter = ui.lineEditSurfaceFilter;
    filterUi.focusPointDistance = ui.spinFocusPointFilterDistance;
    _surfacePanel->configureFilters(filterUi, _state->pointCollection());

    SurfacePanelController::TagUiRefs tagUi{
        .approved = ui.chkApproved,
        .defective = ui.chkDefective,
        .reviewed = ui.chkReviewed,
        .inspect = ui.chkInspect,
    };
    _surfacePanel->configureTags(tagUi);

    cmbSegmentationDir = ui.cmbSegmentationDir;
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
    if (auto* chkMoveOnSurfaceChanged = ui.chkMoveOnSurfaceChanged) {
        bool moveOnSurfaceChanged = settings.value(vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE,
                                                   vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toBool();
        QSignalBlocker blocker(chkMoveOnSurfaceChanged);
        chkMoveOnSurfaceChanged->setChecked(moveOnSurfaceChanged);
        connect(chkMoveOnSurfaceChanged, &QCheckBox::toggled, this, &CWindow::onMoveOnSurfaceChangedToggled);
    }
    if (auto* chkPlaneIntersectionLines = ui.chkPlaneIntersectionLines) {
        bool showPlaneIntersectionLines = settings.value(vc3d::settings::viewer::SHOW_PLANE_INTERSECTION_LINES,
                                                         vc3d::settings::viewer::SHOW_PLANE_INTERSECTION_LINES_DEFAULT).toBool();
        QSignalBlocker blocker(chkPlaneIntersectionLines);
        chkPlaneIntersectionLines->setChecked(showPlaneIntersectionLines);
        connect(chkPlaneIntersectionLines, &QCheckBox::toggled,
                this, &CWindow::onPlaneIntersectionLinesToggled);
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        int storedOpacity = settings.value(vc3d::settings::viewer::AXIS_OVERLAY_OPACITY,
                                           spinAxisOverlayOpacity->value()).toInt();
        storedOpacity = std::clamp(storedOpacity, spinAxisOverlayOpacity->minimum(), spinAxisOverlayOpacity->maximum());
        QSignalBlocker blocker(spinAxisOverlayOpacity);
        spinAxisOverlayOpacity->setValue(storedOpacity);
        connect(spinAxisOverlayOpacity, qOverload<int>(&QSpinBox::valueChanged), this, &CWindow::onAxisOverlayOpacityChanged);
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

    connect(ui.btnEditMask, &QPushButton::pressed, this, &CWindow::onEditMaskPressed);
    connect(ui.btnAppendMask, &QPushButton::pressed, this, &CWindow::onAppendMaskPressed);  // Add this

    if (chkAxisAlignedSlices) {
        onAxisAlignedSlicesToggled(chkAxisAlignedSlices->isChecked());
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        onAxisOverlayOpacityChanged(spinAxisOverlayOpacity->value());
    }
    if (auto* chkAxisOverlays = ui.chkAxisOverlays) {
        onAxisOverlayVisibilityToggled(chkAxisOverlays->isChecked());
    }
    if (auto* chkMoveOnSurfaceChanged = ui.chkMoveOnSurfaceChanged) {
        onMoveOnSurfaceChangedToggled(chkMoveOnSurfaceChanged->isChecked());
    }
    if (auto* chkPlaneIntersectionLines = ui.chkPlaneIntersectionLines) {
        onPlaneIntersectionLinesToggled(chkPlaneIntersectionLines->isChecked());
    }

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

    if (_viewerManager && _point_collection_widget && _point_collection_widget->sameWrapAnnotationEnabled()) {
        if (event->key() == Qt::Key_E && event->modifiers() == Qt::ShiftModifier) {
            bool committed = false;
            _viewerManager->forEachBaseViewer([&committed](VolumeViewerBase* baseViewer) {
                if (committed || !baseViewer) {
                    return;
                }
                if (auto* viewer = qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject())) {
                    committed = viewer->commitSameWrapAnnotationPreview();
                }
            });
            if (committed) {
                event->accept();
                return;
            }
        }
        if (event->key() == Qt::Key_Z && event->modifiers() == Qt::ControlModifier) {
            _viewerManager->forEachBaseViewer([](VolumeViewerBase* baseViewer) {
                if (!baseViewer) {
                    return;
                }
                if (auto* viewer = qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject())) {
                    viewer->clearSameWrapAnnotationPreview();
                }
            });
            event->accept();
            return;
        }
    }

    // Shift+G decreases slice step size, Shift+H increases it
    if (event->modifiers() == vc3d::keybinds::keypress::SliceStepDecrease.modifiers && _viewerManager) {
        if (event->key() == vc3d::keybinds::keypress::SliceStepDecrease.key) {
            int currentStep = _viewerManager->sliceStepSize();
            int newStep = std::max(1, currentStep - 1);
            _viewerManager->setSliceStepSize(newStep);
            event->accept();
            return;
        } else if (event->key() == vc3d::keybinds::keypress::SliceStepIncrease.key) {
            int currentStep = _viewerManager->sliceStepSize();
            int newStep = std::min(100, currentStep + 1);
            _viewerManager->setSliceStepSize(newStep);
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
    if (_viewerManager && _point_collection_widget &&
        _point_collection_widget->sameWrapAnnotationEnabled() &&
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
    if (_viewerControlsPanel) {
        _viewerControlsPanel->setViewControlsEnabled(state);
        _viewerControlsPanel->setOverlayWindowAvailable(_volumeOverlay && _volumeOverlay->hasOverlaySelection());
    }
}

auto CWindow::InitializeVolumePkg(const std::string& nVpkgPath) -> bool
{
    _state->setVpkg(nullptr);
    updateNormalGridAvailability();
    if (_segmentationModule && _segmentationModule->editingEnabled()) {
        _segmentationModule->setEditingEnabled(false);
    }
    if (_segmentationModule) {
        _segmentationModule->setAnnotateMode(false);
    }
    if (_segmentationWidget) {
        if (!_segmentationModule || _segmentationWidget->isEditingEnabled()) {
            _segmentationWidget->setEditingEnabled(false);
        }
        _segmentationWidget->setAvailableVolumes({}, QString());
        _segmentationWidget->setVolumePackagePath(QString());
    }

    try {
        _state->setVpkg(VolumePkg::load(nVpkgPath));
    } catch (const std::exception& e) {
        Logger()->error("Failed to initialize volpkg: {}", e.what());
    }

    if (_state->vpkg() == nullptr) {
        Logger()->error("Cannot open project: {}", nVpkgPath);
        QMessageBox::warning(
            this, "Error",
            "Project failed to load. Falling back to a new blank project.");
        _state->setVpkg(VolumePkg::newEmpty());
        return false;
    }
    return true;
}

// Update the widgets
void CWindow::UpdateView(void)
{
    if (!_state->hasVpkg() && _state->currentVolume() == nullptr) {
        setWidgetsEnabled(false);  // Disable Widgets for User
        ui.lblVpkgName->setText("[ No Volume Package Loaded ]");
        return;
    }

    setWidgetsEnabled(true);  // Enable Widgets for User

    // show volume package name
    UpdateVolpkgLabel(0);

    volSelect->setEnabled(can_change_volume_());

    update();
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
    statusBar()->showMessage(text, timeout);
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
        statusBar()->showMessage(_segmentationGrowthStatusText, 0);
    } else if (_segmentationGrowthWarning) {
        _segmentationGrowthWarning->clear();
        _segmentationGrowthWarning->setVisible(false);
        if (statusBar()->currentMessage() == _segmentationGrowthStatusText) {
            statusBar()->clearMessage();
        }
        _segmentationGrowthStatusText.clear();
    }
}

void CWindow::onSliceStepSizeChanged(int newSize)
{
    // Update status bar label
    if (_sliceStepLabel) {
        _sliceStepLabel->setText(tr("Step: %1").arg(newSize));
    }

    // Save to settings
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::SLICE_STEP_SIZE, newSize);

    if (_viewerControlsPanel) {
        _viewerControlsPanel->setSliceStepSize(newSize);
    }
}

// Open volume package
void CWindow::OpenVolume(const QString& path)
{
    QString aVpkgPath = path;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    if (aVpkgPath.isEmpty()) {
        aVpkgPath = QFileDialog::getOpenFileName(
            this, tr("Open Project"), settings.value(vc3d::settings::project::DEFAULT_PATH).toString(),
            tr("Project (*.volpkg.json);;All files (*.*)"),
            nullptr, QFileDialog::DontResolveSymlinks | QFileDialog::ReadOnly | QFileDialog::DontUseNativeDialog);
        if (aVpkgPath.isEmpty()) {
            Logger()->info("Open project canceled");
            return;
        }
    }

    if (!InitializeVolumePkg(aVpkgPath.toStdString())) {
        return;
    }

    // Check version number
    if (_state->vpkg()->version() < VOLPKG_MIN_VERSION) {
        const auto msg = "Volume package is version " +
                         std::to_string(_state->vpkg()->version()) +
                         " but this program requires version " +
                         std::to_string(VOLPKG_MIN_VERSION) + "+.";
        Logger()->error(msg);
        QMessageBox::warning(this, tr("ERROR"), QString(msg.c_str()));
        _state->setVpkg(nullptr);
        updateNormalGridAvailability();
        return;
    }

    refreshCurrentVolumePackageUi(QString(), true);
    if (_menuController) {
        _menuController->updateRecentVolpkgList(aVpkgPath);
    }

    if (_fileWatcher) {
        _fileWatcher->startWatching();
    }
}

void CWindow::refreshVolumeSelectionUi(const QString& preferredVolumeId)
{
    if (!volSelect || !_state || !_state->vpkg()) {
        return;
    }

    QVector<QPair<QString, QString>> volumeEntries;
    std::vector<QString> orderedIds;
    QString activeCandidate = preferredVolumeId;
    const QString currentComboId = volSelect->currentData().toString();
    const QString currentVolumeId = QString::fromStdString(_state->currentVolumeId());

    auto hasVolume = [&](const QString& volumeId) {
        for (const auto& id : orderedIds) {
            if (id == volumeId) {
                return true;
            }
        }
        return false;
    };

    QString bestGrowthVolumeId;
    bool preferredVolumeFound = false;
    const auto volumeIds = _state->vpkg()->volumeIDs();
    for (const auto& id : volumeIds) {
        try {
            auto vol = _state->vpkg()->volume(id);
            const QString idStr = QString::fromStdString(id);
            const QString nameStr = QString::fromStdString(vol->name());
            const QString label = nameStr.isEmpty() ? idStr : QStringLiteral("%1 (%2)").arg(nameStr, idStr);

            orderedIds.push_back(idStr);
            volumeEntries.append({idStr, label});

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
        activeCandidate.clear();
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

    {
        const QSignalBlocker blocker{volSelect};
        volSelect->clear();
        for (const auto& [id, label] : volumeEntries) {
            volSelect->addItem(label, QVariant(id));
        }
        if (activeCandidate.isEmpty()) {
            if (volSelect->count() > 0) {
                volSelect->setCurrentIndex(0);
            }
        } else {
            volSelect->setCurrentIndex(volSelect->findData(activeCandidate));
        }
    }

    QString activeId = volSelect->count() > 0 ? volSelect->currentData().toString() : QString();

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
    if (volSelect && volSelect->count() > 1) {
        return true;
    }
    return false;
}

void CWindow::onVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface *surf, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    // Let fiber annotation controller consume the click if in WaitingForFirstClick mode
    if (_fiberController && _fiberController->handleVolumeClick(vol_loc, normal, surf, buttons, modifiers))
        return;

    if (modifiers & Qt::ShiftModifier) {
        return;
    }
    else if (modifiers & Qt::ControlModifier) {
        std::cout << "clicked on vol loc " << vol_loc << std::endl;
        // Get the surface ID from the surface collection
        std::string surfId;
        if (_state && surf) {
            surfId = _state->findSurfaceId(surf);
        }
        centerFocusAt(vol_loc, normal, surfId);
    }
    else {
    }
}

void CWindow::onSurfaceActivated(const QString& surfaceId, QuadSurface* surface)
{
    const std::string previousSurfId = _state->activeSurfaceId();
    const std::string newSurfId = surfaceId.toStdString();

    // Look up the shared_ptr by ID
    if (_state->vpkg() && !newSurfId.empty()) {
        _state->setActiveSurface(newSurfId, _state->vpkg()->getSurface(newSurfId));
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

    const bool activatingAxisAlignedPlane =
        newSurfId == "xy plane" || newSurfId == "seg xz" || newSurfId == "seg yz";
    if (_axisAlignedSliceController && !activatingAxisAlignedPlane) {
        _axisAlignedSliceController->resetAll();
    }

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

    if (_state->vpkg() && !newSurfId.empty()) {
        _state->setActiveSurface(newSurfId, _state->vpkg()->getSurface(newSurfId));
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

void CWindow::onEditMaskPressed(void)
{
    auto surf = _state->activeSurface().lock();
    if (!surf)
        return;

    std::filesystem::path path = surf->path/"mask.tif";

    // If mask already exists, just open it
    if (std::filesystem::exists(path)) {
        QDesktopServices::openUrl(QUrl::fromLocalFile(path.string().c_str()));
        return;
    }

    if (_maskRenderInProgress)
        return;
    _maskRenderInProgress = true;
    ui.btnEditMask->setEnabled(false);
    ui.btnAppendMask->setEnabled(false);
    statusBar()->showMessage(tr("Rendering mask..."));

    auto* watcher = new QFutureWatcher<void>(this);
    connect(watcher, &QFutureWatcher<void>::finished, this,
            [this, watcher, surf, path]() {
                watcher->deleteLater();
                _maskRenderInProgress = false;
                ui.btnEditMask->setEnabled(true);
                ui.btnAppendMask->setEnabled(true);

                statusBar()->showMessage(tr("Mask saved"), 3000);
                QDesktopServices::openUrl(QUrl::fromLocalFile(
                    QString::fromStdString(path.string())));
            });

    watcher->setFuture(QtConcurrent::run([surf, path]() {
        cv::Mat_<uint8_t> mask;
        cv::Mat_<cv::Vec3f> coords;
        render_binary_mask(surf.get(), mask, coords, 1.0f);
        cv::imwrite(path.string(), mask);

        surf->meta["date_last_modified"] = get_surface_time_str();
        surf->save_meta();
    }));
}

void CWindow::onAppendMaskPressed(void)
{
    auto surf = _state->activeSurface().lock();
    if (!surf || !_state->currentVolume()) {
        if (!surf) {
            QMessageBox::warning(this, tr("Error"), tr("No surface selected."));
        } else {
            QMessageBox::warning(this, tr("Error"), tr("No volume loaded."));
        }
        return;
    }

    if (_maskRenderInProgress)
        return;
    _maskRenderInProgress = true;
    ui.btnEditMask->setEnabled(false);
    ui.btnAppendMask->setEnabled(false);
    statusBar()->showMessage(tr("Rendering mask..."));

    std::filesystem::path path = surf->path/"mask.tif";
    auto volume = _state->currentVolume();

    auto* watcher = new QFutureWatcher<QString>(this);
    connect(watcher, &QFutureWatcher<QString>::finished, this,
            [this, watcher, path]() {
                watcher->deleteLater();
                _maskRenderInProgress = false;
                ui.btnEditMask->setEnabled(true);
                ui.btnAppendMask->setEnabled(true);

                try {
                    QString msg = watcher->result();
                    statusBar()->showMessage(msg, 3000);
                    QDesktopServices::openUrl(QUrl::fromLocalFile(
                        QString::fromStdString(path.string())));
                } catch (const std::exception& e) {
                    QMessageBox::critical(this, tr("Error"),
                                         tr("Failed to render surface: %1").arg(e.what()));
                    statusBar()->clearMessage();
                }
            });

    watcher->setFuture(QtConcurrent::run([surf, volume, path]() -> QString {
        cv::Mat_<uint8_t> mask;
        cv::Mat_<uint8_t> img;
        std::vector<cv::Mat> existing_layers;

        if (std::filesystem::exists(path)) {
            cv::imreadmulti(path.string(), existing_layers, cv::IMREAD_UNCHANGED);

            if (existing_layers.empty())
                throw std::runtime_error("Could not read existing mask file.");

            mask = existing_layers[0];
            cv::Size maskSize = mask.size();

            {
                cv::Size rawSize = surf->rawPointsPtr()->size();
                cv::Vec3f ptr(0, 0, 0);
                cv::Vec3f offset(-rawSize.width/2.0f, -rawSize.height/2.0f, 0);
                float surfScale = surf->scale()[0];
                cv::Mat_<cv::Vec3f> coords;
                surf->gen(&coords, nullptr, maskSize, ptr, surfScale, offset);
                img.create(coords.size());
                render_image_from_coords(coords, img, volume.get());
            }
            cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8U);

            existing_layers.push_back(img);
            atomicImwriteMulti(path, existing_layers);

            QString msg = QString("Appended surface image to existing mask (now %1 layers)")
                              .arg(existing_layers.size());

            surf->meta["date_last_modified"] = get_surface_time_str();
            surf->save_meta();
            return msg;

        } else {
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
    }));
}

QString CWindow::getCurrentVolumePath() const
{
    auto volume = _state->currentVolume();
    if (volume == nullptr) {
        return QString();
    }
    if (volume->isRemote()) {
        return QString::fromStdString(volume->remoteUrl());
    }
    return QString::fromStdString(volume->path().string());
}

void CWindow::onSegmentationDirChanged(int index)
{
    if (!_state->vpkg() || index < 0 || !cmbSegmentationDir) {
        return;
    }

    std::string newDir = cmbSegmentationDir->itemText(index).toStdString();

    // Only reload if the directory actually changed
    if (newDir != _state->vpkg()->getSegmentationDirectory()) {
        // Clear the current segmentation surface first to ensure viewers update
        _state->setSurface("segmentation", nullptr, true);

        // Clear current surface selection
        _state->clearActiveSurface();
        treeWidgetSurfaces->clearSelection();

        if (_surfacePanel) {
            _surfacePanel->resetTagUi();
        }

        // Set the new directory in the VolumePkg
        _state->vpkg()->setSegmentationDirectory(newDir);

        // Reset stride user override for the new directory.
        if (_viewerManager) {
            _viewerManager->resetStrideUserOverride();
        }
        if (_surfacePanel) {
            _surfacePanel->loadSurfaces(true);
        }

        // Update the status bar to show the change
        statusBar()->showMessage(tr("Switched to %1 directory").arg(QString::fromStdString(newDir)), 3000);
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
    if (name == "focus" && poi) {
        lblLocFocus->setText(QString("%1, %2, %3")
            .arg(static_cast<int>(poi->p[0]))
            .arg(static_cast<int>(poi->p[1]))
            .arg(static_cast<int>(poi->p[2])));

        if (_surfacePanel) {
            _surfacePanel->refreshFiltersOnly();
        }

        _axisAlignedSliceController->applyOrientation();

        if (!poi->suppressViewerRecenter) {
            const cv::Vec3f focusPosition = poi->p;
            QTimer::singleShot(0, this, [this, focusPosition]() {
                recenterPlaneViewersOn(focusPosition);
            });
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
        statusBar()->showMessage(tr("Point not found"), 2000);
        return;
    }

    // Get the segmentation surface to project the point onto
    auto seg_surface = _state->surface("segmentation");
    auto* quad_surface = dynamic_cast<QuadSurface*>(seg_surface.get());
    if (!quad_surface) {
        statusBar()->showMessage(tr("No active segmentation surface for anchor conversion"), 3000);
        return;
    }

    // Find the 2D grid location of this point on the surface
    cv::Vec3f ptr(0, 0, 0);
    auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
    float dist = quad_surface->pointTo(ptr, point_opt->p, 4.0, 1000, patchIndex);

    if (dist > 10.0) {
        statusBar()->showMessage(tr("Point is too far from surface (distance: %1)").arg(dist), 3000);
        return;
    }

    // Get the raw grid location (internal coordinates)
    cv::Vec3f loc_3d = quad_surface->loc_raw(ptr);
    cv::Vec2f anchor2d(loc_3d[0], loc_3d[1]);

    // Set the anchor2d on the collection
    _state->pointCollection()->setCollectionAnchor2d(collectionId, anchor2d);

    // Remove the point (it's now represented by the anchor)
    _state->pointCollection()->removePoint(pointId);

    statusBar()->showMessage(tr("Converted point to anchor at grid position (%1, %2)").arg(anchor2d[0]).arg(anchor2d[1]), 3000);
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
        statusBar()->showMessage(tr("Coordinates copied to clipboard: %1").arg(coords), 2000);
    }
}

void CWindow::onResetAxisAlignedRotations()
{
    _axisAlignedSliceController->resetRotations();
    _axisAlignedSliceController->applyOrientation();
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->refreshAll();
    }
    statusBar()->showMessage(tr("All plane rotations reset"), 2000);
}

void CWindow::onAxisOverlayVisibilityToggled(bool enabled)
{
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->setAxisAlignedEnabled(enabled && _axisAlignedSliceController->isEnabled());
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        spinAxisOverlayOpacity->setEnabled(_axisAlignedSliceController->isEnabled() && enabled);
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
    _axisAlignedSliceController->setEnabled(enabled, ui.chkAxisOverlays, ui.spinAxisOverlayOpacity);
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES, enabled ? "1" : "0");
}

void CWindow::onMoveOnSurfaceChangedToggled(bool enabled)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE, enabled ? "1" : "0");

    if (!_viewerManager) {
        return;
    }

    const bool editingActive = _segmentationModule && _segmentationModule->editingEnabled();
    _viewerManager->forEachBaseViewer([this, enabled, editingActive](VolumeViewerBase* viewer) {
        if (!viewer) {
            return;
        }
        _viewerManager->setResetDefaultFor(viewer, enabled);
        if (editingActive && viewer->surfName() == "segmentation") {
            viewer->setResetViewOnSurfaceChange(false);
            return;
        }
        viewer->setResetViewOnSurfaceChange(enabled);
    });
}

void CWindow::onPlaneIntersectionLinesToggled(bool enabled)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::SHOW_PLANE_INTERSECTION_LINES, enabled ? "1" : "0");

    if (!_viewerManager) {
        return;
    }

    _viewerManager->forEachBaseViewer([enabled](VolumeViewerBase* viewer) {
        if (viewer) {
            viewer->setPlaneIntersectionLinesVisible(enabled);
        }
    });
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
        _viewerManager->forEachBaseViewer([this, enabled](VolumeViewerBase* viewer) {
            if (!viewer) {
                return;
            }
            if (viewer->surfName() == "segmentation") {
                bool defaultReset = _viewerManager->resetDefaultFor(viewer);
                if (enabled) {
                    viewer->setResetViewOnSurfaceChange(false);
                } else {
                    viewer->setResetViewOnSurfaceChange(defaultReset);
                }
            }
        });
    }

    if (enabled) {
        auto activeSurfaceShared = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));

        if (!_segmentationModule->beginEditingSession(activeSurfaceShared)) {
            statusBar()->showMessage(tr("Unable to start segmentation editing"), 3000);
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
    statusBar()->showMessage(message, 2000);
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
        statusBar()->showMessage(tr("Cancelling running tools..."), 3000);
    }
}

void CWindow::onGrowSegmentationSurface(SegmentationGrowthMethod method,
                                        SegmentationGrowthDirection direction,
                                        int steps,
                                        bool inpaintOnly)
{
    if (!_segmentationGrower) {
        statusBar()->showMessage(tr("Segmentation growth is unavailable."), 4000);
        return;
    }

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

    // Disconnect previous model's signals if any
    if (_surfaceOverlayModel) {
        disconnect(_surfaceOverlayModel, &QStandardItemModel::dataChanged,
                   this, &CWindow::onSurfaceOverlaySelectionChanged);
    }

    // Create new model
    _surfaceOverlayModel = new QStandardItemModel(ui.surfaceOverlaySelect);
    ui.surfaceOverlaySelect->setModel(_surfaceOverlayModel);

    // Use a QListView to properly show checkboxes
    auto* listView = new QListView(ui.surfaceOverlaySelect);
    ui.surfaceOverlaySelect->setView(listView);

    // Get current segmentation directory for filtering
    std::string currentDir;
    if (_state->vpkg()) {
        currentDir = _state->vpkg()->getSegmentationDirectory();
    }

    // Add "All" item at the top
    auto* allItem = new QStandardItem(tr("All"));
    allItem->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
    allItem->setData(Qt::Unchecked, Qt::CheckStateRole);
    allItem->setData(QStringLiteral("__all__"), Qt::UserRole);
    _surfaceOverlayModel->appendRow(allItem);

    if (_state) {
        const auto names = _state->surfaceNames();
        for (const auto& name : names) {
            // Only add QuadSurfaces (actual segmentations), skip PlaneSurfaces
            auto surf = _state->surface(name);
            auto* quadSurf = dynamic_cast<QuadSurface*>(surf.get());
            if (!quadSurf) {
                continue;
            }

            // Filter by current segmentation directory
            if (!currentDir.empty() && !surf->path.empty()) {
                std::string surfDir = surf->path.parent_path().filename().string();
                if (surfDir != currentDir) {
                    continue;
                }
            }

            auto* item = new QStandardItem(QString::fromStdString(name));
            item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
            item->setData(Qt::Unchecked, Qt::CheckStateRole);
            item->setData(QString::fromStdString(name), Qt::UserRole);

            // Assign persistent color if not already assigned
            if (_surfaceOverlayColorAssignments.find(name) == _surfaceOverlayColorAssignments.end()) {
                _surfaceOverlayColorAssignments[name] = _nextSurfaceOverlayColorIndex++;
            }
            size_t colorIdx = _surfaceOverlayColorAssignments[name];

            // Create color swatch icon (16x16 colored square)
            QPixmap swatch(16, 16);
            swatch.fill(getOverlayColor(colorIdx));
            item->setIcon(QIcon(swatch));

            _surfaceOverlayModel->appendRow(item);
        }
    }

    // Connect model's dataChanged signal for checkbox state changes
    connect(_surfaceOverlayModel, &QStandardItemModel::dataChanged,
            this, &CWindow::onSurfaceOverlaySelectionChanged);
}

void CWindow::onSurfaceOverlaySelectionChanged(const QModelIndex& topLeft,
                                                const QModelIndex& /*bottomRight*/,
                                                const QVector<int>& roles)
{
    if (!roles.contains(Qt::CheckStateRole) || !_surfaceOverlayModel || !_viewerManager) {
        return;
    }

    // Check if "All" was toggled (row 0)
    QStandardItem* changedItem = _surfaceOverlayModel->itemFromIndex(topLeft);
    if (changedItem && changedItem->data(Qt::UserRole).toString() == QStringLiteral("__all__")) {
        bool allChecked = changedItem->checkState() == Qt::Checked;

        // Block signals while updating all items
        {
            QSignalBlocker blocker(_surfaceOverlayModel);
            for (int row = 1; row < _surfaceOverlayModel->rowCount(); ++row) {
                QStandardItem* item = _surfaceOverlayModel->item(row);
                if (item) {
                    item->setCheckState(allChecked ? Qt::Checked : Qt::Unchecked);
                }
            }
        }
    }

    // Build map of selected surfaces with colors
    std::map<std::string, cv::Vec3b> selectedSurfaces;
    int checkedCount = 0;
    int totalSurfaces = 0;

    for (int row = 1; row < _surfaceOverlayModel->rowCount(); ++row) {
        QStandardItem* item = _surfaceOverlayModel->item(row);
        if (!item) continue;

        totalSurfaces++;
        if (item->checkState() == Qt::Checked) {
            checkedCount++;
            std::string name = item->data(Qt::UserRole).toString().toStdString();
            size_t colorIdx = _surfaceOverlayColorAssignments[name];
            selectedSurfaces[name] = getOverlayColorBGR(colorIdx);
        }
    }

    // Update "All" checkbox state (partial/full/none) without triggering recursion
    {
        QSignalBlocker blocker(_surfaceOverlayModel);
        QStandardItem* allItem = _surfaceOverlayModel->item(0);
        if (allItem) {
            if (checkedCount == 0) {
                allItem->setCheckState(Qt::Unchecked);
            } else if (checkedCount == totalSurfaces && totalSurfaces > 0) {
                allItem->setCheckState(Qt::Checked);
            } else {
                allItem->setCheckState(Qt::PartiallyChecked);
            }
        }
    }

    // Propagate to all viewers
    _viewerManager->forEachBaseViewer([&selectedSurfaces](VolumeViewerBase* viewer) {
        viewer->setSurfaceOverlays(selectedSurfaces);
    });
}

QColor CWindow::getOverlayColor(size_t index) const
{
    static const std::vector<QColor> palette = {
        QColor(80, 180, 255),   // sky blue
        QColor(180, 80, 220),   // violet
        QColor(80, 220, 200),   // aqua/teal
        QColor(220, 80, 180),   // magenta
        QColor(80, 130, 255),   // medium blue
        QColor(160, 80, 255),   // purple
        QColor(80, 255, 220),   // cyan
        QColor(255, 80, 200),   // hot pink
        QColor(120, 220, 80),   // lime green
        QColor(80, 180, 120),   // spring green
        QColor(150, 200, 255),  // light sky blue
        QColor(200, 150, 230),  // light violet
    };
    return palette[index % palette.size()];
}

cv::Vec3b CWindow::getOverlayColorBGR(size_t index) const
{
    QColor c = getOverlayColor(index);
    return cv::Vec3b(c.blue(), c.green(), c.red());
}

void CWindow::onCopyWithNtRequested()
{
    if (!_segmentationGrower) {
        statusBar()->showMessage(tr("Segmentation growth is unavailable."), 4000);
        return;
    }

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

    statusBar()->showMessage(tr("Focused & aligned view to %1 points").arg(pts.size()), 3000);
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

    QMdiSubWindow* subWindows[N] = {};

    for (int i = 0; i < N; ++i) {
        QString title = (i == 0) ? tr("Fiber Ref") : tr("Fiber Annotate");

        auto* baseViewer = newConnectedViewer(
            FiberAnnotationController::fiberSurfaceName(i), title, mdiArea);
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
    QRect area = mdiArea->contentsRect();
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
