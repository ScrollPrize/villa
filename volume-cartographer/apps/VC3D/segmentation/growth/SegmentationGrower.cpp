#include "SegmentationGrower.hpp"

#include "../../NeuralTraceServiceManager.hpp"
#include "ExtrapolationGrowth.hpp"
#include "../SegmentationModule.hpp"
#include "../SegmentationWidget.hpp"

#include "../../CVolumeViewer.hpp"
#include "../../CSurfaceCollection.hpp"
#include "../../ViewerManager.hpp"
#include "../../SurfacePanelController.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

#include <QtConcurrent/QtConcurrent>

#include <QDir>
#include <QFileInfo>
#include <QLoggingCategory>
#include <QUuid>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <utility>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <set>
#include <unordered_set>

#include <opencv2/core.hpp>

#include <nlohmann/json.hpp>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

Q_DECLARE_LOGGING_CATEGORY(lcSegGrowth);

namespace
{
const QString kDenseLatestSentinel = QStringLiteral("extrap_displacement_latest");
const QString kCopyLatestSentinel = QStringLiteral("copy_displacement_latest");

QString cacheRootForVolumePkg(const std::shared_ptr<VolumePkg>& pkg)
{
    if (!pkg) {
        return QString();
    }

    const QString base = QString::fromStdString(pkg->getVolpkgDirectory());
    return QDir(base).filePath(QStringLiteral("cache"));
}

void ensureGenerationsChannel(QuadSurface* surface)
{
    if (!surface) {
        return;
    }

    cv::Mat generations = surface->channel("generations");
    if (!generations.empty()) {
        return;
    }

    cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }

    cv::Mat_<uint16_t> seeded(points->rows, points->cols, static_cast<uint16_t>(1));
    surface->setChannel("generations", seeded);
}

void ensureSurfaceMetaObject(QuadSurface* surface)
{
    if (!surface) {
        return;
    }

    if (surface->meta && surface->meta->is_object()) {
        return;
    }

    surface->meta = std::make_unique<nlohmann::json>(nlohmann::json::object());
}


bool isInvalidPoint(const cv::Vec3f& value)
{
    return !std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]) ||
           (value[0] == -1.0f && value[1] == -1.0f && value[2] == -1.0f);
}

std::optional<std::pair<int, int>> worldToGridIndexApprox(QuadSurface* surface,
                                                          const cv::Vec3f& worldPos,
                                                          cv::Vec3f& pointerSeed,
                                                          bool& pointerSeedValid,
                                                          SurfacePatchIndex* patchIndex = nullptr)
{
    if (!surface) {
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    if (!pointerSeedValid) {
        pointerSeed = surface->pointer();
        pointerSeedValid = true;
    }

    surface->pointTo(pointerSeed, worldPos, std::numeric_limits<float>::max(), 400, patchIndex);
    cv::Vec3f raw = surface->loc_raw(pointerSeed);

    const int rows = points->rows;
    const int cols = points->cols;
    if (rows <= 0 || cols <= 0) {
        return std::nullopt;
    }

    int approxCol = static_cast<int>(std::lround(raw[0]));
    int approxRow = static_cast<int>(std::lround(raw[1]));
    approxRow = std::clamp(approxRow, 0, rows - 1);
    approxCol = std::clamp(approxCol, 0, cols - 1);

    if (isInvalidPoint((*points)(approxRow, approxCol))) {
        constexpr int kMaxRadius = 12;
        float bestDistSq = std::numeric_limits<float>::max();
        int bestRow = -1;
        int bestCol = -1;

        auto accumulateCandidate = [&](int r, int c) {
            const cv::Vec3f& candidate = (*points)(r, c);
            if (isInvalidPoint(candidate)) {
                return;
            }
            const cv::Vec3f diff = candidate - worldPos;
            const float distSq = diff.dot(diff);
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestRow = r;
                bestCol = c;
            }
        };

        for (int radius = 1; radius <= kMaxRadius; ++radius) {
            const int rowStart = std::max(0, approxRow - radius);
            const int rowEnd = std::min(rows - 1, approxRow + radius);
            const int colStart = std::max(0, approxCol - radius);
            const int colEnd = std::min(cols - 1, approxCol + radius);
            for (int r = rowStart; r <= rowEnd; ++r) {
                for (int c = colStart; c <= colEnd; ++c) {
                    accumulateCandidate(r, c);
                }
            }
            if (bestRow != -1) {
                approxRow = bestRow;
                approxCol = bestCol;
                break;
            }
        }

        if (bestRow == -1) {
            return std::nullopt;
        }
    }

    return std::make_pair(approxRow, approxCol);
}

std::optional<std::pair<int, int>> locateGridIndexWithPatchIndex(QuadSurface* surface,
                                                                 SurfacePatchIndex* patchIndex,
                                                                 const cv::Vec3f& worldPos,
                                                                 cv::Vec3f& pointerSeed,
                                                                 bool& pointerSeedValid)
{
    if (!surface) {
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    if (patchIndex) {
        cv::Vec2f gridScale = surface->scale();
        float scale = std::max(gridScale[0], gridScale[1]);
        if (!std::isfinite(scale) || scale <= 0.0f) {
            scale = 1.0f;
        }
        const float tolerance = std::max(8.0f, scale * 8.0f);
        // Query without surface filter, then verify result matches our surface
        if (auto hit = patchIndex->locate(worldPos, tolerance)) {
            if (hit->surface.get() == surface) {
                const int col = std::clamp(static_cast<int>(std::round(hit->ptr[0])),
                                           0,
                                           points->cols - 1);
                const int row = std::clamp(static_cast<int>(std::round(hit->ptr[1])),
                                           0,
                                           points->rows - 1);
                return std::make_pair(row, col);
            }
        }
    }

    return worldToGridIndexApprox(surface, worldPos, pointerSeed, pointerSeedValid, patchIndex);
}

std::optional<cv::Rect> computeCorrectionsAffectedBounds(QuadSurface* surface,
                                                      const SegmentationCorrectionsPayload& corrections,
                                                      ViewerManager* viewerManager)
{
    if (!surface || corrections.empty()) {
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    SurfacePatchIndex* patchIndex = viewerManager ? viewerManager->surfacePatchIndex() : nullptr;

    cv::Vec3f pointerSeed{0.0f, 0.0f, 0.0f};
    bool pointerSeedValid = false;

    const int rows = points->rows;
    const int cols = points->cols;

    constexpr int kPaddingCells = 12;

    bool anyMapped = false;
    int unionRowStart = rows;
    int unionRowEnd = 0;
    int unionColStart = cols;
    int unionColEnd = 0;

    for (const auto& collection : corrections.collections) {
        if (collection.points.empty()) {
            continue;
        }

        int collectionMinRow = rows;
        int collectionMaxRow = -1;
        int collectionMinCol = cols;
        int collectionMaxCol = -1;

        for (const auto& colPoint : collection.points) {
            auto gridIndex = locateGridIndexWithPatchIndex(surface,
                                                           patchIndex,
                                                           colPoint.p,
                                                           pointerSeed,
                                                           pointerSeedValid);
            if (!gridIndex) {
                continue;
            }
            const auto [row, col] = *gridIndex;
            collectionMinRow = std::min(collectionMinRow, row);
            collectionMaxRow = std::max(collectionMaxRow, row);
            collectionMinCol = std::min(collectionMinCol, col);
            collectionMaxCol = std::max(collectionMaxCol, col);
        }

        if (collectionMinRow > collectionMaxRow || collectionMinCol > collectionMaxCol) {
            continue;
        }

        anyMapped = true;
        const int rowStart = std::max(0, collectionMinRow - kPaddingCells);
        const int rowEndExclusive = std::min(rows, collectionMaxRow + kPaddingCells + 1);
        const int colStart = std::max(0, collectionMinCol - kPaddingCells);
        const int colEndExclusive = std::min(cols, collectionMaxCol + kPaddingCells + 1);

        unionRowStart = std::min(unionRowStart, rowStart);
        unionRowEnd = std::max(unionRowEnd, rowEndExclusive);
        unionColStart = std::min(unionColStart, colStart);
        unionColEnd = std::max(unionColEnd, colEndExclusive);
    }

    if (!anyMapped) {
        return cv::Rect(0, 0, cols, rows);
    }

    const int width = std::max(0, unionColEnd - unionColStart);
    const int height = std::max(0, unionRowEnd - unionRowStart);
    if (width == 0 || height == 0) {
        return cv::Rect(0, 0, cols, rows);
    }

    return cv::Rect(unionColStart, unionRowStart, width, height);
}

// Compute 3D world-space bounding box from correction points (min 512^3, centered)
// Returns the world-space box and the corresponding 2D grid region
std::optional<CorrectionsBounds> computeCorrectionsBounds(
    const SegmentationCorrectionsPayload& corrections,
    QuadSurface* surface,
    float minWorldSize = 512.0f)
{
    if (!surface || corrections.empty()) {
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    // Find min/max of all correction point positions (3D world coords)
    cv::Vec3f worldMin(std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max());
    cv::Vec3f worldMax(std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest());

    bool hasPoints = false;
    for (const auto& collection : corrections.collections) {
        for (const auto& colPoint : collection.points) {
            hasPoints = true;
            worldMin[0] = std::min(worldMin[0], colPoint.p[0]);
            worldMin[1] = std::min(worldMin[1], colPoint.p[1]);
            worldMin[2] = std::min(worldMin[2], colPoint.p[2]);
            worldMax[0] = std::max(worldMax[0], colPoint.p[0]);
            worldMax[1] = std::max(worldMax[1], colPoint.p[1]);
            worldMax[2] = std::max(worldMax[2], colPoint.p[2]);
        }
    }

    if (!hasPoints) {
        return std::nullopt;
    }

    // Compute center and expand to at least minWorldSize in each dimension
    cv::Vec3f center = (worldMin + worldMax) * 0.5f;
    cv::Vec3f halfSize;
    for (int i = 0; i < 3; ++i) {
        float extent = worldMax[i] - worldMin[i];
        halfSize[i] = std::max(extent * 0.5f, minWorldSize * 0.5f);
    }

    worldMin = center - halfSize;
    worldMax = center + halfSize;

    // Find all grid cells whose 3D positions fall within this world-space box
    const int rows = points->rows;
    const int cols = points->cols;

    int gridRowMin = rows;
    int gridRowMax = -1;
    int gridColMin = cols;
    int gridColMax = -1;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const cv::Vec3f& pt = (*points)(r, c);
            if (isInvalidPoint(pt)) {
                continue;
            }

            // Check if point is within the world-space bounding box
            if (pt[0] >= worldMin[0] && pt[0] <= worldMax[0] &&
                pt[1] >= worldMin[1] && pt[1] <= worldMax[1] &&
                pt[2] >= worldMin[2] && pt[2] <= worldMax[2]) {
                gridRowMin = std::min(gridRowMin, r);
                gridRowMax = std::max(gridRowMax, r);
                gridColMin = std::min(gridColMin, c);
                gridColMax = std::max(gridColMax, c);
            }
        }
    }

    if (gridRowMin > gridRowMax || gridColMin > gridColMax) {
        // No grid cells found within bounds, return full grid
        return CorrectionsBounds{worldMin, worldMax, cv::Rect(0, 0, cols, rows)};
    }

    // Clip to surface grid bounds (already implicitly done by loop bounds)
    int width = gridColMax - gridColMin + 1;
    int height = gridRowMax - gridRowMin + 1;

    CorrectionsBounds bounds;
    bounds.worldMin = worldMin;
    bounds.worldMax = worldMax;
    bounds.gridRegion = cv::Rect(gridColMin, gridRowMin, width, height);

    return bounds;
}

// Crop a QuadSurface to a 2D grid region
std::unique_ptr<QuadSurface> cropSurfaceToGridRegion(
    const QuadSurface* surface,
    const cv::Rect& gridRegion)
{
    if (!surface) {
        return nullptr;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return nullptr;
    }

    const int rows = points->rows;
    const int cols = points->cols;

    // Clamp region to valid bounds
    int x0 = std::clamp(gridRegion.x, 0, cols - 1);
    int y0 = std::clamp(gridRegion.y, 0, rows - 1);
    int x1 = std::clamp(gridRegion.x + gridRegion.width, 1, cols);
    int y1 = std::clamp(gridRegion.y + gridRegion.height, 1, rows);

    if (x1 <= x0 || y1 <= y0) {
        return nullptr;
    }

    // Extract ROI
    cv::Mat_<cv::Vec3f> roi(*points, cv::Range(y0, y1), cv::Range(x0, x1));
    cv::Mat_<cv::Vec3f> roiClone = roi.clone();

    // Create new QuadSurface with the cropped data
    auto cropped = std::make_unique<QuadSurface>(roiClone, surface->scale());

    return cropped;
}

// Generate ISO 8601 timestamp string for folder naming
std::string generateTimestampString()
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&time, &tm);

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H-%M-%S", &tm);
    return std::string(buffer);
}

// Save correction annotation data
void saveCorrectionsAnnotation(
    const std::filesystem::path& volpkgRoot,
    const std::string& segmentId,
    const QuadSurface* beforeCrop,
    const QuadSurface* afterCrop,
    const SegmentationCorrectionsPayload& corrections,
    const std::vector<std::string>& volumeIds,
    const std::string& growthVolumeId,
    const CorrectionsBounds& bounds)
{
    if (!beforeCrop || !afterCrop || corrections.empty()) {
        return;
    }

    // Create timestamped folder
    std::string timestamp = generateTimestampString();
    std::filesystem::path correctionsDir = volpkgRoot / "corrections" / timestamp;

    std::error_code ec;
    std::filesystem::create_directories(correctionsDir, ec);
    if (ec) {
        qCWarning(lcSegGrowth) << "Failed to create corrections directory:" << ec.message().c_str();
        return;
    }

    // Save before tifxyz
    std::filesystem::path beforePath = correctionsDir / "before";
    try {
        auto* mutableBefore = const_cast<QuadSurface*>(beforeCrop);
        mutableBefore->save(beforePath.string(), "before", true);
    } catch (const std::exception& ex) {
        qCWarning(lcSegGrowth) << "Failed to save before tifxyz:" << ex.what();
    }

    // Save after tifxyz
    std::filesystem::path afterPath = correctionsDir / "after";
    try {
        auto* mutableAfter = const_cast<QuadSurface*>(afterCrop);
        mutableAfter->save(afterPath.string(), "after", true);
    } catch (const std::exception& ex) {
        qCWarning(lcSegGrowth) << "Failed to save after tifxyz:" << ex.what();
    }

    // Build corrections.json
    nlohmann::json j;
    j["timestamp"] = timestamp;
    j["segment_id"] = segmentId;
    j["volumes"] = volumeIds;
    j["volume_used"] = growthVolumeId;
    j["bbox"] = {
        {"min", {bounds.worldMin[0], bounds.worldMin[1], bounds.worldMin[2]}},
        {"max", {bounds.worldMax[0], bounds.worldMax[1], bounds.worldMax[2]}}
    };

    // Build collections array with points sorted by creation_time
    nlohmann::json collectionsJson = nlohmann::json::array();
    for (const auto& collection : corrections.collections) {
        nlohmann::json collJson;
        collJson["id"] = collection.id;
        collJson["name"] = collection.name;
        collJson["color"] = {collection.color[0], collection.color[1], collection.color[2]};

        // Sort points by creation_time to preserve placement order
        std::vector<ColPoint> sortedPoints = collection.points;
        std::sort(sortedPoints.begin(), sortedPoints.end(),
                  [](const ColPoint& a, const ColPoint& b) {
                      return a.creation_time < b.creation_time;
                  });

        nlohmann::json pointsJson = nlohmann::json::array();
        for (const auto& pt : sortedPoints) {
            nlohmann::json ptJson;
            ptJson["id"] = pt.id;
            ptJson["position"] = {pt.p[0], pt.p[1], pt.p[2]};
            ptJson["creation_time"] = pt.creation_time;
            pointsJson.push_back(ptJson);
        }
        collJson["points"] = pointsJson;

        collectionsJson.push_back(collJson);
    }
    j["collections"] = collectionsJson;

    // Write corrections.json
    std::filesystem::path jsonPath = correctionsDir / "corrections.json";
    try {
        std::ofstream ofs(jsonPath);
        if (ofs.is_open()) {
            ofs << j.dump(2);
            ofs.close();
            qCInfo(lcSegGrowth) << "Saved corrections annotation to" << jsonPath.c_str();
        } else {
            qCWarning(lcSegGrowth) << "Failed to open corrections.json for writing";
        }
    } catch (const std::exception& ex) {
        qCWarning(lcSegGrowth) << "Failed to write corrections.json:" << ex.what();
    }
}

void queueIndexUpdateForBounds(SurfacePatchIndex* index,
                               const SurfacePatchIndex::SurfacePtr& surface,
                               const cv::Rect& vertexRect)
{
    if (!index || !surface || vertexRect.width <= 0 || vertexRect.height <= 0) {
        return;
    }

    const int rowStart = vertexRect.y;
    const int rowEnd = vertexRect.y + vertexRect.height;
    const int colStart = vertexRect.x;
    const int colEnd = vertexRect.x + vertexRect.width;

    index->queueCellRangeUpdate(surface, rowStart, rowEnd, colStart, colEnd);
}

void synchronizeSurfaceMeta(const std::shared_ptr<VolumePkg>& pkg,
                            QuadSurface* surface,
                            SurfacePanelController* panel)
{
    if (!pkg || !surface) {
        return;
    }

    // getSurface now returns the QuadSurface directly, so we just need to refresh the panel
    const auto loadedIds = pkg->getLoadedSurfaceIDs();
    for (const auto& id : loadedIds) {
        auto loadedSurface = pkg->getSurface(id);
        if (!loadedSurface) {
            continue;
        }

        if (loadedSurface->path == surface->path) {
            // Sync metadata if needed
            if (surface->meta) {
                if (!loadedSurface->meta) {
                    loadedSurface->meta = std::make_unique<nlohmann::json>(nlohmann::json::object());
                }
                *loadedSurface->meta = *surface->meta;
            } else if (loadedSurface->meta) {
                loadedSurface->meta->clear();
            }

            if (panel) {
                panel->refreshSurfaceMetrics(id);
            }
        }
    }
}

void refreshSegmentationViewers(ViewerManager* manager)
{
    if (!manager) {
        return;
    }

    manager->forEachViewer([](CVolumeViewer* viewer) {
        if (!viewer) {
            return;
        }

        if (viewer->surfName() == "segmentation") {
            viewer->invalidateVis();
            viewer->renderVisible(true);
        }
    });
}

struct DenseDisplacementJob
{
    QString socketPath;
    QString tifxyzInputPath;
    QString volumeZarrPath;
    int volumeScale{0};
    int iterations{1};
    QString checkpointPath;
    DenseTtaMode ttaMode{DenseTtaMode::Mirror};
    QString ttaMergeMethod{QStringLiteral("vector_geomedian")};
    double ttaOutlierDropThresh{1.25};
    std::optional<nlohmann::json> customParams;
    std::vector<SegmentationGrowthDirection> directions;
};

struct CopyDisplacementJob
{
    QString socketPath;
    QString tifxyzInputPath;
    QString volumeZarrPath;
    int volumeScale{0};
    QString checkpointPath;
    DenseTtaMode ttaMode{DenseTtaMode::Mirror};
    QString ttaMergeMethod{QStringLiteral("vector_geomedian")};
    double ttaOutlierDropThresh{1.25};
    std::optional<nlohmann::json> customParams;
};

std::optional<std::string> denseDirectionToToken(SegmentationGrowthDirection direction)
{
    switch (direction) {
    case SegmentationGrowthDirection::Left:
        return std::string("left");
    case SegmentationGrowthDirection::Right:
        return std::string("right");
    case SegmentationGrowthDirection::Up:
        return std::string("up");
    case SegmentationGrowthDirection::Down:
        return std::string("down");
    case SegmentationGrowthDirection::All:
        return std::string("all");
    default:
        return std::nullopt;
    }
}

std::vector<SegmentationGrowthDirection> resolveDenseDirections(
    SegmentationGrowthDirection primaryDirection,
    const std::vector<SegmentationGrowthDirection>& allowedDirections)
{
    if (primaryDirection != SegmentationGrowthDirection::All) {
        return {primaryDirection};
    }

    const std::vector<SegmentationGrowthDirection> order = {
        SegmentationGrowthDirection::Left,
        SegmentationGrowthDirection::Right,
        SegmentationGrowthDirection::Up,
        SegmentationGrowthDirection::Down,
    };

    std::set<SegmentationGrowthDirection> allowedSet;
    for (auto dir : allowedDirections) {
        if (dir == SegmentationGrowthDirection::All) {
            allowedSet.insert(order.begin(), order.end());
            break;
        }
        if (dir == SegmentationGrowthDirection::Left ||
            dir == SegmentationGrowthDirection::Right ||
            dir == SegmentationGrowthDirection::Up ||
            dir == SegmentationGrowthDirection::Down) {
            allowedSet.insert(dir);
        }
    }
    if (allowedSet.empty()) {
        allowedSet.insert(order.begin(), order.end());
    }

    if (allowedSet.size() == order.size()) {
        return {SegmentationGrowthDirection::All};
    }

    std::vector<SegmentationGrowthDirection> resolved;
    resolved.reserve(4);
    for (auto dir : order) {
        if (allowedSet.count(dir) > 0) {
            resolved.push_back(dir);
        }
    }
    return resolved;
}

QString createDenseSnapshotPath(const QString& preferredDir)
{
    QString outputDir = preferredDir.trimmed();
    if (outputDir.isEmpty()) {
        outputDir = QDir::tempPath();
    }

    QDir dir(outputDir);
    if (!dir.exists() && !dir.mkpath(QStringLiteral("."))) {
        dir = QDir(QDir::tempPath());
    }

    const QString suffix = QUuid::createUuid().toString(QUuid::WithoutBraces);
    return dir.filePath(QStringLiteral("vc_dense_input_%1").arg(suffix));
}

nlohmann::json sendSocketJsonRequest(const QString& socketPath, const nlohmann::json& request)
{
    const std::string socketStd = socketPath.toStdString();
    int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        throw std::runtime_error("Failed to create UNIX socket.");
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (socketStd.size() >= sizeof(addr.sun_path)) {
        ::close(sock);
        throw std::runtime_error("Neural socket path is too long.");
    }
    std::strncpy(addr.sun_path, socketStd.c_str(), sizeof(addr.sun_path) - 1);

    if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(sock);
        throw std::runtime_error("Failed to connect to neural trace socket.");
    }

    const std::string requestPayload = request.dump() + "\n";
    if (::send(sock, requestPayload.c_str(), requestPayload.size(), 0) < 0) {
        ::close(sock);
        throw std::runtime_error("Failed to send dense displacement request.");
    }

    std::string responsePayload;
    char ch = 0;
    while (::recv(sock, &ch, 1, 0) == 1) {
        if (ch == '\n') {
            break;
        }
        responsePayload.push_back(ch);
    }
    ::close(sock);

    if (responsePayload.empty()) {
        throw std::runtime_error("Dense displacement service returned an empty response.");
    }

    return nlohmann::json::parse(responsePayload);
}

std::string sanitizeSegmentId(const std::string& value)
{
    std::string out;
    out.reserve(value.size());
    for (unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '_' || ch == '-') {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('_');
        }
    }
    while (!out.empty() && out.back() == '_') {
        out.pop_back();
    }
    if (out.empty()) {
        out = "dense_displacement";
    }
    return out;
}

std::string makeUniqueSegmentId(const std::filesystem::path& pathsDir, const std::string& base)
{
    std::string candidate = base;
    int suffix = 1;
    while (std::filesystem::exists(pathsDir / candidate)) {
        candidate = base + "_" + std::to_string(suffix++);
    }
    return candidate;
}

TracerGrowthResult runDenseDisplacementGrowth(const DenseDisplacementJob& job)
{
    TracerGrowthResult result;
    std::vector<std::filesystem::path> temporaryPaths;
    temporaryPaths.reserve(job.directions.size() + 1);
    if (!job.tifxyzInputPath.isEmpty()) {
        temporaryPaths.emplace_back(job.tifxyzInputPath.toStdString());
    }

    auto finalizeResult = [&]() -> TracerGrowthResult {
        result.temporarySurfacePaths.clear();
        result.temporarySurfacePaths.reserve(temporaryPaths.size());
        for (const auto& path : temporaryPaths) {
            if (!path.empty()) {
                result.temporarySurfacePaths.push_back(path);
            }
        }
        return result;
    };

    QString currentInputPath = job.tifxyzInputPath;
    QString finalOutputPath = job.tifxyzInputPath;

    for (auto direction : job.directions) {
        const auto directionToken = denseDirectionToToken(direction);
        if (!directionToken.has_value()) {
            continue;
        }

        nlohmann::json request;
        request["request_type"] = "dense_displacement_grow";
        request["tifxyz_path"] = currentInputPath.toStdString();
        request["grow_direction"] = *directionToken;
        request["iterations"] = std::max(1, job.iterations);
        request["edge_input_rowscols"] = 40;
        request["volume_path"] = job.volumeZarrPath.toStdString();
        request["volume_scale"] = std::max(0, job.volumeScale);
        request["checkpoint_path"] = job.checkpointPath.toStdString();
        request["tta_merge_method"] = job.ttaMergeMethod.toStdString();
        request["tta_outlier_drop_thresh"] = job.ttaOutlierDropThresh;
        switch (job.ttaMode) {
        case DenseTtaMode::Rotate3:
            request["tta"] = true;
            request["tta_transform"] = "rotate3";
            break;
        case DenseTtaMode::None:
            request["tta"] = false;
            break;
        case DenseTtaMode::Mirror:
        default:
            request["tta"] = true;
            request["tta_transform"] = "mirror";
            break;
        }

        if (job.customParams) {
            auto denseArgsIt = job.customParams->find("dense_args");
            if (denseArgsIt != job.customParams->end() && denseArgsIt->is_object()) {
                request["dense_args"] = *denseArgsIt;
            }
            auto denseOverridesIt = job.customParams->find("dense_overrides");
            if (denseOverridesIt != job.customParams->end() && denseOverridesIt->is_object()) {
                request["overrides"] = *denseOverridesIt;
            }
        }

        nlohmann::json response;
        try {
            response = sendSocketJsonRequest(job.socketPath, request);
        } catch (const std::exception& ex) {
            result.error = QStringLiteral("Dense displacement request failed: %1").arg(ex.what());
            return finalizeResult();
        }

        if (response.contains("error")) {
            QString serviceError;
            if (response["error"].is_string()) {
                serviceError = QString::fromStdString(response["error"].get<std::string>());
            } else {
                serviceError = QString::fromStdString(response["error"].dump());
            }
            result.error = QStringLiteral("Dense displacement service error: %1").arg(serviceError);
            return finalizeResult();
        }
        if (!response.contains("output_tifxyz_path") || !response["output_tifxyz_path"].is_string()) {
            result.error = QStringLiteral("Dense displacement response missing output_tifxyz_path.");
            return finalizeResult();
        }

        finalOutputPath = QString::fromStdString(response["output_tifxyz_path"].get<std::string>());
        currentInputPath = finalOutputPath;
        if (!finalOutputPath.isEmpty()) {
            temporaryPaths.emplace_back(finalOutputPath.toStdString());
        }
    }

    if (finalOutputPath.isEmpty()) {
        result.error = QStringLiteral("Dense displacement did not produce an output path.");
        return finalizeResult();
    }

    std::unique_ptr<QuadSurface> loaded;
    try {
        loaded = load_quad_from_tifxyz(finalOutputPath.toStdString());
    } catch (const std::exception& ex) {
        result.error = QStringLiteral("Failed to load dense displacement output: %1").arg(ex.what());
        return finalizeResult();
    }
    if (!loaded) {
        result.error = QStringLiteral("Dense displacement output could not be loaded.");
        return finalizeResult();
    }

    result.surface = loaded.release();
    result.statusMessage = QStringLiteral("Dense displacement growth completed");
    return finalizeResult();
}

TracerGrowthResult runCopyDisplacementGrowth(const CopyDisplacementJob& job)
{
    TracerGrowthResult result;
    std::vector<std::filesystem::path> temporaryPaths;
    temporaryPaths.reserve(3);
    if (!job.tifxyzInputPath.isEmpty()) {
        temporaryPaths.emplace_back(job.tifxyzInputPath.toStdString());
    }

    auto finalizeResult = [&]() -> TracerGrowthResult {
        result.temporarySurfacePaths.clear();
        result.temporarySurfacePaths.reserve(temporaryPaths.size());
        for (const auto& path : temporaryPaths) {
            if (!path.empty()) {
                result.temporarySurfacePaths.push_back(path);
            }
        }
        return result;
    };

    nlohmann::json request;
    request["request_type"] = "displacement_copy_grow";
    request["tifxyz_path"] = job.tifxyzInputPath.toStdString();
    request["volume_path"] = job.volumeZarrPath.toStdString();
    request["volume_scale"] = std::max(0, job.volumeScale);
    request["checkpoint_path"] = job.checkpointPath.toStdString();
    request["tta_merge_method"] = job.ttaMergeMethod.toStdString();
    request["tta_outlier_drop_thresh"] = job.ttaOutlierDropThresh;

    switch (job.ttaMode) {
    case DenseTtaMode::Rotate3:
        request["tta"] = true;
        request["tta_transform"] = "rotate3";
        break;
    case DenseTtaMode::None:
        request["tta"] = false;
        break;
    case DenseTtaMode::Mirror:
    default:
        request["tta"] = true;
        request["tta_transform"] = "mirror";
        break;
    }

    if (job.customParams) {
        auto copyArgsIt = job.customParams->find("copy_args");
        if (copyArgsIt != job.customParams->end() && copyArgsIt->is_object()) {
            request["copy_args"] = *copyArgsIt;
        }
        auto copyOverridesIt = job.customParams->find("copy_overrides");
        if (copyOverridesIt != job.customParams->end() && copyOverridesIt->is_object()) {
            request["overrides"] = *copyOverridesIt;
        }
    }

    nlohmann::json response;
    try {
        response = sendSocketJsonRequest(job.socketPath, request);
    } catch (const std::exception& ex) {
        result.error = QStringLiteral("Displacement copy request failed: %1").arg(ex.what());
        return finalizeResult();
    }

    if (response.contains("error")) {
        QString serviceError;
        if (response["error"].is_string()) {
            serviceError = QString::fromStdString(response["error"].get<std::string>());
        } else {
            serviceError = QString::fromStdString(response["error"].dump());
        }
        result.error = QStringLiteral("Displacement copy service error: %1").arg(serviceError);
        return finalizeResult();
    }

    if (!response.contains("output_tifxyz_paths") || !response["output_tifxyz_paths"].is_object()) {
        result.error = QStringLiteral("Displacement copy response missing output_tifxyz_paths.");
        return finalizeResult();
    }

    const auto& outputs = response["output_tifxyz_paths"];
    if (!outputs.contains("front") || !outputs["front"].is_string()) {
        result.error = QStringLiteral("Displacement copy response missing front output.");
        return finalizeResult();
    }
    if (!outputs.contains("back") || !outputs["back"].is_string()) {
        result.error = QStringLiteral("Displacement copy response missing back output.");
        return finalizeResult();
    }

    const QString frontPath = QString::fromStdString(outputs["front"].get<std::string>());
    const QString backPath = QString::fromStdString(outputs["back"].get<std::string>());
    if (!frontPath.isEmpty()) {
        temporaryPaths.emplace_back(frontPath.toStdString());
    }
    if (!backPath.isEmpty()) {
        temporaryPaths.emplace_back(backPath.toStdString());
    }

    result.statusMessage = QStringLiteral("Displacement copy completed");
    return finalizeResult();
}
} // namespace

SegmentationGrower::SegmentationGrower(Context context,
                                       UiCallbacks callbacks,
                                       QObject* parent)
    : QObject(parent)
    , _context(std::move(context))
    , _callbacks(std::move(callbacks))
{
}

void SegmentationGrower::updateContext(Context context)
{
    _context = std::move(context);
}

void SegmentationGrower::updateUiCallbacks(UiCallbacks callbacks)
{
    _callbacks = std::move(callbacks);
}

void SegmentationGrower::setSurfacePanel(SurfacePanelController* panel)
{
    _surfacePanel = panel;
}

bool SegmentationGrower::start(const VolumeContext& volumeContext,
                               SegmentationGrowthMethod method,
                               SegmentationGrowthDirection direction,
                               int steps,
                               bool inpaintOnly)
{
    auto showStatus = [&](const QString& text, int timeout) {
        if (_callbacks.showStatus) {
            _callbacks.showStatus(text, timeout);
        }
    };

    if (_running) {
        qCInfo(lcSegGrowth) << "Rejecting growth because another operation is running";
        showStatus(tr("A surface growth operation is already running."), kStatusMedium);
        return false;
    }

    if (!_context.module || !_context.widget || !_context.surfaces) {
        showStatus(tr("Segmentation growth is unavailable."), kStatusLong);
        return false;
    }

    auto segmentationSurface = std::dynamic_pointer_cast<QuadSurface>(_context.surfaces->surface("segmentation"));
    if (!segmentationSurface) {
        qCInfo(lcSegGrowth) << "Rejecting growth because segmentation surface is missing";
        showStatus(tr("Segmentation surface is not available."), kStatusMedium);
        return false;
    }

    ensureGenerationsChannel(segmentationSurface.get());

    // Handle Extrapolation method separately - it doesn't need volume data
    if (method == SegmentationGrowthMethod::Extrapolation) {
        const int sanitizedSteps = std::max(1, steps);
        const SegmentationGrowthDirection effectiveDirection = direction;

        // Get extrapolation parameters from widget
        int pointCount = _context.widget->extrapolationPointCount();
        ExtrapolationType extrapType = _context.widget->extrapolationType();

        // Check for JSON custom params overrides
        if (_context.widget->customParamsValid()) {
            if (auto customParams = _context.widget->customParamsJson()) {
                if (customParams->contains("extrapolation_point_count") &&
                    (*customParams)["extrapolation_point_count"].is_number()) {
                    pointCount = (*customParams)["extrapolation_point_count"].get<int>();
                    pointCount = std::clamp(pointCount, 3, 20);
                }
                if (customParams->contains("extrapolation_type") &&
                    (*customParams)["extrapolation_type"].is_string()) {
                    const std::string typeStr = (*customParams)["extrapolation_type"].get<std::string>();
                    if (typeStr == "quadratic") {
                        extrapType = ExtrapolationType::Quadratic;
                    } else {
                        extrapType = ExtrapolationType::Linear;
                    }
                }
            }
        }

        showStatus(tr("Running extrapolation-based surface growth..."), kStatusMedium);

        // Set up SDT context for LinearFit mode or custom params override
        SDTContext sdtContext;
        SDTContext* sdtContextPtr = nullptr;
        std::shared_ptr<Volume> sdtVolume;

        // For LinearFit, automatically use the dropdown-selected volume for SDT refinement
        if (extrapType == ExtrapolationType::LinearFit && volumeContext.package) {
            std::string sdtVolumeId = volumeContext.requestedVolumeId;
            if (sdtVolumeId.empty()) {
                sdtVolumeId = volumeContext.activeVolumeId;
            }
            if (!sdtVolumeId.empty()) {
                try {
                    sdtVolume = volumeContext.package->volume(sdtVolumeId);
                    if (sdtVolume) {
                        sdtContext.binaryDataset = sdtVolume->zarrDataset(0);
                        sdtContext.cache = _context.chunkCache;
                        sdtContextPtr = &sdtContext;
                        qCInfo(lcSegGrowth) << "Linear+Fit: SDT refinement using volume"
                                            << QString::fromStdString(sdtVolumeId);
                    }
                } catch (const std::exception& e) {
                    qCWarning(lcSegGrowth) << "Failed to set up SDT for Linear+Fit:" << e.what();
                }
            }
        }

        // Apply SDT params from widget UI as defaults (applied to sdtContext regardless of source)
        sdtContext.params.maxSteps = _context.widget->sdtMaxSteps();
        sdtContext.params.stepSize = _context.widget->sdtStepSize();
        sdtContext.params.convergenceThreshold = _context.widget->sdtConvergence();
        sdtContext.params.chunkSize = _context.widget->sdtChunkSize();

        // Check custom params for SDT settings overrides (or explicit volume for other modes)
        if (_context.widget->customParamsValid()) {
            if (auto customParams = _context.widget->customParamsJson()) {
                // Allow custom params to override/specify SDT volume for any mode
                if (!sdtContextPtr && customParams->contains("sdt_volume_id") &&
                    (*customParams)["sdt_volume_id"].is_string()) {
                    const std::string sdtVolumeId = (*customParams)["sdt_volume_id"].get<std::string>();
                    if (!sdtVolumeId.empty() && volumeContext.package) {
                        try {
                            sdtVolume = volumeContext.package->volume(sdtVolumeId);
                            if (sdtVolume) {
                                sdtContext.binaryDataset = sdtVolume->zarrDataset(0);
                                sdtContext.cache = _context.chunkCache;
                                sdtContextPtr = &sdtContext;
                                qCInfo(lcSegGrowth) << "SDT refinement enabled with volume"
                                                    << QString::fromStdString(sdtVolumeId);
                            }
                        } catch (const std::exception& e) {
                            qCWarning(lcSegGrowth) << "Failed to load SDT volume:" << e.what();
                        }
                    }
                }

                // Apply Newton refinement param overrides from JSON if SDT is enabled
                // (JSON overrides widget UI values for advanced users)
                if (sdtContextPtr) {
                    if (customParams->contains("sdt_max_steps") &&
                        (*customParams)["sdt_max_steps"].is_number()) {
                        sdtContext.params.maxSteps = std::clamp(
                            (*customParams)["sdt_max_steps"].get<int>(), 1, 10);
                    }
                    if (customParams->contains("sdt_step_size") &&
                        (*customParams)["sdt_step_size"].is_number()) {
                        sdtContext.params.stepSize = std::clamp(
                            (*customParams)["sdt_step_size"].get<float>(), 0.1f, 2.0f);
                    }
                    if (customParams->contains("sdt_convergence") &&
                        (*customParams)["sdt_convergence"].is_number()) {
                        sdtContext.params.convergenceThreshold = std::clamp(
                            (*customParams)["sdt_convergence"].get<float>(), 0.1f, 2.0f);
                    }
                    if (customParams->contains("sdt_chunk_size") &&
                        (*customParams)["sdt_chunk_size"].is_number()) {
                        sdtContext.params.chunkSize = std::clamp(
                            (*customParams)["sdt_chunk_size"].get<int>(), 32, 256);
                    }
                }
            }
        }

        // Set up SkeletonPath context when using SkeletonPath extrapolation
        SkeletonPathContext skeletonContext;
        SkeletonPathContext* skeletonContextPtr = nullptr;
        std::shared_ptr<Volume> skeletonVolume;

        if (extrapType == ExtrapolationType::SkeletonPath && volumeContext.package) {
            std::string skeletonVolumeId = volumeContext.requestedVolumeId;
            if (skeletonVolumeId.empty()) {
                skeletonVolumeId = volumeContext.activeVolumeId;
            }
            if (!skeletonVolumeId.empty()) {
                try {
                    skeletonVolume = volumeContext.package->volume(skeletonVolumeId);
                    if (skeletonVolume) {
                        skeletonContext.binaryDataset = skeletonVolume->zarrDataset(0);
                        skeletonContext.cache = _context.chunkCache;
                        skeletonContextPtr = &skeletonContext;
                        qCInfo(lcSegGrowth) << "Skeleton Path: using volume"
                                            << QString::fromStdString(skeletonVolumeId);
                    }
                } catch (const std::exception& e) {
                    qCWarning(lcSegGrowth) << "Failed to set up Skeleton Path volume:" << e.what();
                }
            }

            // Apply skeleton params from widget UI
            int connectivity = _context.widget->skeletonConnectivity();
            if (connectivity == 6) {
                skeletonContext.params.connectivity = DijkstraConnectivity::Conn6;
            } else if (connectivity == 18) {
                skeletonContext.params.connectivity = DijkstraConnectivity::Conn18;
            } else {
                skeletonContext.params.connectivity = DijkstraConnectivity::Conn26;
            }

            int sliceOrientation = _context.widget->skeletonSliceOrientation();
            skeletonContext.params.sliceOrientation = sliceOrientation == 0
                ? SkeletonSliceOrientation::X
                : SkeletonSliceOrientation::Y;
            skeletonContext.params.chunkSize = _context.widget->skeletonChunkSize();
            skeletonContext.params.searchRadius = _context.widget->skeletonSearchRadius();
        }

        qCInfo(lcSegGrowth) << "Segmentation extrapolation requested"
                            << segmentationGrowthDirectionToString(effectiveDirection)
                            << "steps" << sanitizedSteps
                            << "pointCount" << pointCount
                            << "type" << extrapolationTypeToString(extrapType)
                            << "sdtRefinement" << (sdtContextPtr != nullptr)
                            << "skeletonPath" << (skeletonContextPtr != nullptr);

        _running = true;
        _context.module->setGrowthInProgress(true);

        // Run extrapolation (synchronous since it's fast)
        auto result = runExtrapolationGrowth(
            segmentationSurface.get(),
            effectiveDirection,
            sanitizedSteps,
            pointCount,
            extrapType,
            sdtContextPtr,
            skeletonContextPtr);

        if (!result.error.isEmpty()) {
            handleFailure(result.error);
            return false;
        }

        if (!result.surface) {
            handleFailure(tr("Extrapolation produced no result surface."));
            return false;
        }

        // Get the active volume for voxel size (for metadata update)
        double voxelSize = 1.0;
        if (volumeContext.activeVolume) {
            voxelSize = volumeContext.activeVolume->voxelSize();
        }

        // Swap points into the existing surface
        cv::Mat_<cv::Vec3f>* newPoints = result.surface->rawPointsPtr();
        cv::Mat_<cv::Vec3f>* oldPoints = segmentationSurface->rawPointsPtr();
        if (newPoints && oldPoints) {
            *oldPoints = newPoints->clone();
        }

        // Copy channels from result to existing surface
        cv::Mat newGen = result.surface->channel("generations");
        if (!newGen.empty()) {
            segmentationSurface->setChannel("generations", newGen);
        }

        cv::Mat newApproval = result.surface->channel("approval", SURF_CHANNEL_NORESIZE);
        if (!newApproval.empty()) {
            segmentationSurface->setChannel("approval", newApproval);
        }

        // Copy metadata
        if (result.surface->meta && result.surface->meta->is_object()) {
            ensureSurfaceMetaObject(segmentationSurface.get());
            *segmentationSurface->meta = *result.surface->meta;
        }

        // Update metadata
        updateSegmentationSurfaceMetadata(segmentationSurface.get(), voxelSize);

        // Clean up
        delete result.surface;

        // Invalidate caches and refresh
        segmentationSurface->invalidateCache();

        // Update the surface patch index (used for intersection rendering)
        // Must be done before setSurface to ensure index is current when viewers refresh
        if (_context.viewerManager) {
            _context.viewerManager->refreshSurfacePatchIndex(segmentationSurface);
        }

        // Re-set the surface in the collection to trigger proper viewer refresh
        // (same pattern used by Tracer growth)
        if (_context.surfaces) {
            _context.surfaces->setSurface("segmentation", segmentationSurface, false, true);
        }
        refreshSegmentationViewers(_context.viewerManager);
        if (_context.module && _context.module->hasActiveSession()) {
            QuadSurface* baseSurface = _context.module->activeBaseSurface();
            if (baseSurface) {
                _context.module->refreshSessionFromSurface(baseSurface);
            }
        }

        if (_surfacePanel) {
            _surfacePanel->refreshSurfaceMetrics("segmentation");
        }

        showStatus(result.statusMessage.isEmpty()
                       ? tr("Extrapolation growth completed.")
                       : result.statusMessage,
                   kStatusMedium);

        finalize(true);
        return true;
    }

    std::shared_ptr<Volume> growthVolume;
    std::string growthVolumeId = volumeContext.requestedVolumeId;

    if (volumeContext.package && !volumeContext.requestedVolumeId.empty()) {
        try {
            growthVolume = volumeContext.package->volume(volumeContext.requestedVolumeId);
        } catch (const std::out_of_range&) {
            growthVolume.reset();
        }
    }

    if (!growthVolume) {
        growthVolume = volumeContext.activeVolume;
        growthVolumeId = volumeContext.activeVolumeId;
    }

    if (!growthVolume) {
        qCInfo(lcSegGrowth) << "Rejecting growth because no usable volume is available";
        showStatus(tr("No volume available for growth."), kStatusMedium);
        return false;
    }

    if (!volumeContext.requestedVolumeId.empty() &&
        volumeContext.requestedVolumeId != growthVolumeId) {
        showStatus(tr("Selected growth volume unavailable; using the active volume instead."), kStatusMedium);
    }

    SegmentationCorrectionsPayload corrections = _context.module->buildCorrectionsPayload();
    const bool hasCorrections = !corrections.empty();
    const bool usingCorrections = method == SegmentationGrowthMethod::Corrections && hasCorrections;

    if (method == SegmentationGrowthMethod::Corrections && !hasCorrections) {
        qCInfo(lcSegGrowth) << "Corrections growth requested without correction points; continuing with tracer behavior.";
    }

    if (usingCorrections) {
        qCInfo(lcSegGrowth) << "Including" << corrections.collections.size() << "correction set(s)";
    }

    if (_context.module->growthInProgress()) {
        showStatus(tr("Surface growth already in progress"), kStatusMedium);
        return false;
    }

    const bool allowZeroSteps = inpaintOnly || method == SegmentationGrowthMethod::Corrections;
    int sanitizedSteps = allowZeroSteps ? std::max(0, steps) : std::max(1, steps);
    if (usingCorrections) {
        // Correction-guided tracer should not advance additional steps.
        sanitizedSteps = 0;
    }

    const SegmentationGrowthDirection effectiveDirection = inpaintOnly
        ? SegmentationGrowthDirection::All
        : direction;

    SegmentationGrowthRequest request;
    request.method = method;
    request.direction = effectiveDirection;
    request.steps = sanitizedSteps;
    request.inpaintOnly = inpaintOnly;

    if (inpaintOnly) {
        // Consume any pending overrides; inpainting ignores directional constraints.
        const auto pendingOverride = _context.module->takeShortcutDirectionOverride();
        if (pendingOverride) {
            qCInfo(lcSegGrowth) << "Ignoring direction override for inpaint request.";
        }
        request.allowedDirections = {SegmentationGrowthDirection::All};
    } else {
        if (auto overrideDirs = _context.module->takeShortcutDirectionOverride()) {
            request.allowedDirections = std::move(*overrideDirs);
        }
        if (request.allowedDirections.empty()) {
            request.allowedDirections = _context.widget->allowedGrowthDirections();
            if (request.allowedDirections.empty()) {
                request.allowedDirections = {
                    SegmentationGrowthDirection::Up,
                    SegmentationGrowthDirection::Down,
                    SegmentationGrowthDirection::Left,
                    SegmentationGrowthDirection::Right
                };
            }
        }
    }

    request.directionFields = _context.widget->directionFieldConfigs();

    if (!_context.widget->customParamsValid()) {
        const QString errorText = _context.widget->customParamsError();
        const QString message = errorText.isEmpty()
            ? tr("Custom params JSON is invalid. Fix the contents and try again.")
            : tr("Custom params JSON is invalid: %1").arg(errorText);
        showStatus(message, kStatusLong);
        return false;
    }
    if (auto customParams = _context.widget->customParamsJson()) {
        request.customParams = std::move(*customParams);
    }
    if (method == SegmentationGrowthMethod::Corrections &&
        _context.module && _context.module->cellReoptCollectionPending()) {
        if (!request.customParams) {
            request.customParams = nlohmann::json::object();
        }
        (*request.customParams)["cell_reopt_mode"] = true;
        qCInfo(lcSegGrowth) << "Cell reoptimization mode enabled for tracer params.";
    }

    const bool neuralTracerEnabled = _context.widget->neuralTracerEnabled();
    const bool denseMode = neuralTracerEnabled &&
        _context.widget->neuralModelType() == NeuralTracerModelType::DenseDisplacement;

    if (denseMode) {
        const QString denseCheckpointPath = _context.widget->denseCheckpointPath().trimmed();
        const QString pythonPath = _context.widget->neuralPythonPath();
        const QString volumeZarr = _context.widget->volumeZarrPath().trimmed();
        const int volumeScale = _context.widget->neuralVolumeScale();
        const DenseTtaMode denseTtaMode = _context.widget->denseTtaMode();
        const QString denseTtaMergeMethod = _context.widget->denseTtaMergeMethod().trimmed();
        const double denseTtaOutlierDropThresh = _context.widget->denseTtaOutlierDropThresh();
        const auto outputMode = _context.widget->neuralOutputMode();

        if (denseCheckpointPath.isEmpty()) {
            showStatus(tr("Dense displacement requires a dense checkpoint path."), kStatusLong);
            return false;
        }
        const bool usingDenseLatestPreset = denseCheckpointPath == kDenseLatestSentinel;
        if (!usingDenseLatestPreset &&
            (!QFileInfo::exists(denseCheckpointPath) || !QFileInfo(denseCheckpointPath).isFile())) {
            showStatus(tr("Dense checkpoint does not exist: %1").arg(denseCheckpointPath), kStatusLong);
            return false;
        }
        if (volumeZarr.isEmpty()) {
            showStatus(tr("Dense displacement requires a volume zarr path."), kStatusLong);
            return false;
        }

        auto& serviceManager = NeuralTraceServiceManager::instance();
        showStatus(tr("Starting neural trace service for dense displacement..."), kStatusLong);
        if (!serviceManager.ensureServiceRunning(denseCheckpointPath, volumeZarr, volumeScale, pythonPath)) {
            const QString error = serviceManager.lastError();
            showStatus(tr("Failed to start neural trace service: %1").arg(error), kStatusLong);
            return false;
        }
        const QString socketPath = serviceManager.socketPath();
        if (socketPath.isEmpty()) {
            showStatus(tr("Neural trace service is running but socket path is unavailable."), kStatusLong);
            return false;
        }

        const auto denseDirections = resolveDenseDirections(effectiveDirection, request.allowedDirections);
        if (denseDirections.empty()) {
            showStatus(tr("No valid dense growth directions are enabled."), kStatusLong);
            return false;
        }

        const QString segmentationPath = QString::fromStdString(segmentationSurface->path.string());
        const QString denseSnapshotDir = segmentationPath.isEmpty()
            ? QString()
            : QFileInfo(segmentationPath).dir().absolutePath();
        const QString denseInputPath = createDenseSnapshotPath(denseSnapshotDir);
        try {
            const QFileInfo snapshotInfo(denseInputPath);
            const std::string snapshotId = snapshotInfo.fileName().toStdString();

            // QuadSurface::save mutates the instance path/id/meta. Preserve and restore
            // so taking a dense snapshot does not retarget the live segmentation surface.
            const std::filesystem::path originalSurfacePath = segmentationSurface->path;
            const std::string originalSurfaceId = segmentationSurface->id;
            std::unique_ptr<nlohmann::json> originalSurfaceMeta;
            if (segmentationSurface->meta) {
                originalSurfaceMeta = std::make_unique<nlohmann::json>(*segmentationSurface->meta);
            }

            const auto restoreLiveSurfaceState = [&]() {
                segmentationSurface->path = originalSurfacePath;
                segmentationSurface->id = originalSurfaceId;
                if (originalSurfaceMeta) {
                    segmentationSurface->meta = std::make_unique<nlohmann::json>(*originalSurfaceMeta);
                } else {
                    segmentationSurface->meta.reset();
                }
            };

            try {
                segmentationSurface->save(denseInputPath.toStdString(), snapshotId, false);
            } catch (...) {
                restoreLiveSurfaceState();
                throw;
            }
            restoreLiveSurfaceState();
        } catch (const std::exception& ex) {
            showStatus(tr("Failed to prepare dense displacement input: %1").arg(ex.what()), kStatusLong);
            return false;
        }

        DenseDisplacementJob denseJob;
        denseJob.socketPath = socketPath;
        denseJob.tifxyzInputPath = denseInputPath;
        denseJob.volumeZarrPath = volumeZarr;
        denseJob.volumeScale = volumeScale;
        denseJob.iterations = std::max(1, sanitizedSteps);
        denseJob.checkpointPath = denseCheckpointPath;
        denseJob.ttaMode = denseTtaMode;
        denseJob.ttaMergeMethod = denseTtaMergeMethod.isEmpty()
            ? QStringLiteral("vector_geomedian")
            : denseTtaMergeMethod;
        denseJob.ttaOutlierDropThresh = std::max(0.01, denseTtaOutlierDropThresh);
        denseJob.customParams = request.customParams;
        denseJob.directions = denseDirections;

        qCInfo(lcSegGrowth) << "Dense displacement enabled:"
                            << "socket" << socketPath
                            << "iterations" << denseJob.iterations
                            << "directions" << static_cast<int>(denseDirections.size());

        _running = true;
        _context.module->setGrowthInProgress(true);

        ActiveRequest pending;
        pending.volumeContext = volumeContext;
        pending.growthVolume = growthVolume;
        pending.growthVolumeId = growthVolumeId;
        pending.segmentationSurface = segmentationSurface;
        pending.growthVoxelSize = growthVolume->voxelSize();
        pending.usingCorrections = false;
        pending.inpaintOnly = inpaintOnly;
        pending.denseDisplacement = true;
        pending.denseCreateNewSegment = outputMode == NeuralTracerOutputMode::CreateNewSegment;
        _activeRequest = std::move(pending);

        showStatus(
            outputMode == NeuralTracerOutputMode::CreateNewSegment
                ? tr("Running dense displacement (create new segment)...")
                : tr("Running dense displacement (overwrite current segment)..."),
            kStatusMedium);

        auto future = QtConcurrent::run(runDenseDisplacementGrowth, denseJob);
        _watcher = std::make_unique<QFutureWatcher<TracerGrowthResult>>(this);
        connect(_watcher.get(), &QFutureWatcher<TracerGrowthResult>::finished,
                this, &SegmentationGrower::onFutureFinished);
        _watcher->setFuture(future);
        return true;
    }

    // Heatmap neural tracer integration - pass neural_socket to GrowPatch when enabled.
    if (neuralTracerEnabled) {
        const QString checkpointPath = _context.widget->neuralCheckpointPath().trimmed();
        const QString pythonPath = _context.widget->neuralPythonPath();
        const QString volumeZarr = _context.widget->volumeZarrPath().trimmed();
        const int volumeScale = _context.widget->neuralVolumeScale();
        const int batchSize = _context.widget->neuralBatchSize();

        if (checkpointPath.isEmpty()) {
            showStatus(tr("Neural tracer enabled but no checkpoint path specified."), kStatusLong);
            return false;
        }
        if (volumeZarr.isEmpty()) {
            showStatus(tr("Neural tracer enabled but no volume zarr path available."), kStatusLong);
            return false;
        }

        auto& serviceManager = NeuralTraceServiceManager::instance();
        showStatus(tr("Starting neural trace service..."), kStatusLong);

        if (!serviceManager.ensureServiceRunning(checkpointPath, volumeZarr, volumeScale, pythonPath)) {
            const QString error = serviceManager.lastError();
            showStatus(tr("Failed to start neural trace service: %1").arg(error), kStatusLong);
            return false;
        }

        const QString socketPath = serviceManager.socketPath();
        if (socketPath.isEmpty()) {
            showStatus(tr("Neural trace service is running but socket path is unavailable."), kStatusLong);
            return false;
        }

        if (!request.customParams) {
            request.customParams = nlohmann::json::object();
        }
        (*request.customParams)["neural_socket"] = socketPath.toStdString();
        (*request.customParams)["neural_batch_size"] = batchSize;

        qCInfo(lcSegGrowth) << "Neural tracer enabled:"
                            << "socket" << socketPath
                            << "batch_size" << batchSize;
    }

    request.corrections = corrections;
    if (method == SegmentationGrowthMethod::Corrections) {
        if (auto zRange = _context.module->correctionsZRange()) {
            request.correctionsZRange = zRange;
        }
    }

    std::optional<cv::Rect> correctionAffectedBounds;
    if (usingCorrections) {
        correctionAffectedBounds = computeCorrectionsAffectedBounds(segmentationSurface.get(),
                                                                    corrections,
                                                                    _context.viewerManager);
        if (correctionAffectedBounds) {
            const int rowEnd = correctionAffectedBounds->y + correctionAffectedBounds->height;
            const int colEnd = correctionAffectedBounds->x + correctionAffectedBounds->width;
            qCInfo(lcSegGrowth) << "Computed correction affected bounds:"
                                << "rows" << correctionAffectedBounds->y << "to" << rowEnd
                                << "cols" << correctionAffectedBounds->x << "to" << colEnd;
        } else {
            qCInfo(lcSegGrowth) << "Unable to compute correction affected bounds; falling back to full surface rebuild.";
        }
    }

    TracerGrowthContext ctx;
    ctx.resumeSurface = segmentationSurface.get();
    ctx.volume = growthVolume.get();
    ctx.cache = _context.chunkCache;
    ctx.cacheRoot = cacheRootForVolumePkg(volumeContext.package);
    ctx.voxelSize = growthVolume->voxelSize();
    ctx.normalGridPath = volumeContext.normalGridPath;
    ctx.normal3dZarrPath = volumeContext.normal3dZarrPath;

    // Populate fields for corrections annotation saving
    if (volumeContext.package) {
        ctx.volpkgRoot = std::filesystem::path(volumeContext.package->getVolpkgDirectory());
        ctx.volumeIds = volumeContext.package->volumeIDs();
    }
    ctx.growthVolumeId = growthVolumeId;

    if (ctx.cacheRoot.isEmpty()) {
        const auto volumePath = growthVolume->path();
        ctx.cacheRoot = QDir(QString::fromStdString(volumePath.parent_path().string()))
                            .filePath(QStringLiteral("cache"));
    }

    if (ctx.cacheRoot.isEmpty()) {
        qCInfo(lcSegGrowth) << "Tracer growth aborted because cache root is empty";
        showStatus(tr("Cache root unavailable for tracer growth."), kStatusLong);
        return false;
    }

    if (method == SegmentationGrowthMethod::Corrections) {
        if (usingCorrections) {
            showStatus(tr("Applying correction-guided tracer growth..."), kStatusMedium);
        } else {
            showStatus(tr("No correction points provided; running tracer growth..."), kStatusMedium);
        }
    } else {
        const QString status = inpaintOnly
            ? tr("Running tracer inpainting...")
            : tr("Running tracer-based surface growth...");
        showStatus(status, kStatusMedium);
    }

    qCInfo(lcSegGrowth) << "Segmentation growth requested"
                        << segmentationGrowthMethodToString(method)
                        << segmentationGrowthDirectionToString(effectiveDirection)
                        << "steps" << sanitizedSteps
                        << "inpaintOnly" << inpaintOnly;
    qCInfo(lcSegGrowth) << "Growth volume ID" << QString::fromStdString(growthVolumeId);
    qCInfo(lcSegGrowth) << "Starting tracer growth";

    _running = true;
    _context.module->setGrowthInProgress(true);

    ActiveRequest pending;
    pending.volumeContext = volumeContext;
    pending.growthVolume = growthVolume;
    pending.growthVolumeId = growthVolumeId;
    pending.segmentationSurface = segmentationSurface;
    pending.growthVoxelSize = growthVolume->voxelSize();
    pending.usingCorrections = usingCorrections;
    pending.inpaintOnly = inpaintOnly;
    pending.correctionsAffectedBounds = correctionAffectedBounds;

    // Compute corrections bounds and snapshot "before" surface for annotation saving
    if (usingCorrections) {
        pending.corrections = corrections;
        auto bounds = computeCorrectionsBounds(corrections, segmentationSurface.get());
        if (bounds) {
            pending.correctionsBounds = bounds;
            pending.beforeCrop = cropSurfaceToGridRegion(segmentationSurface.get(), bounds->gridRegion);
            if (pending.beforeCrop) {
                qCInfo(lcSegGrowth) << "Captured before-crop for corrections annotation:"
                                    << bounds->gridRegion.width << "x" << bounds->gridRegion.height;
            }
        }
    }

    _activeRequest = std::move(pending);

    auto future = QtConcurrent::run(runTracerGrowth, request, ctx);
    _watcher = std::make_unique<QFutureWatcher<TracerGrowthResult>>(this);
    connect(_watcher.get(), &QFutureWatcher<TracerGrowthResult>::finished,
            this, &SegmentationGrower::onFutureFinished);
    _watcher->setFuture(future);

    return true;
}

bool SegmentationGrower::startCopyWithNt(const VolumeContext& volumeContext)
{
    auto showStatus = [&](const QString& text, int timeout) {
        if (_callbacks.showStatus) {
            _callbacks.showStatus(text, timeout);
        }
    };

    if (_running) {
        showStatus(tr("A surface growth operation is already running."), kStatusMedium);
        return false;
    }
    if (!_context.module || !_context.widget || !_context.surfaces) {
        showStatus(tr("Segmentation growth is unavailable."), kStatusLong);
        return false;
    }
    if (!_context.widget->neuralTracerEnabled()) {
        showStatus(tr("Enable neural tracer before using Copy with NT."), kStatusLong);
        return false;
    }
    if (_context.widget->neuralModelType() != NeuralTracerModelType::DisplacementCopy) {
        showStatus(tr("Select 'Displacement Copy' model type to use Copy with NT."), kStatusLong);
        return false;
    }
    if (!_context.widget->customParamsValid()) {
        const QString errorText = _context.widget->customParamsError();
        const QString message = errorText.isEmpty()
            ? tr("Custom params JSON is invalid. Fix the contents and try again.")
            : tr("Custom params JSON is invalid: %1").arg(errorText);
        showStatus(message, kStatusLong);
        return false;
    }

    auto segmentationSurface = std::dynamic_pointer_cast<QuadSurface>(_context.surfaces->surface("segmentation"));
    if (!segmentationSurface) {
        showStatus(tr("Segmentation surface is not available."), kStatusMedium);
        return false;
    }

    std::shared_ptr<Volume> growthVolume;
    std::string growthVolumeId = volumeContext.requestedVolumeId;
    if (volumeContext.package && !volumeContext.requestedVolumeId.empty()) {
        try {
            growthVolume = volumeContext.package->volume(volumeContext.requestedVolumeId);
        } catch (const std::out_of_range&) {
            growthVolume.reset();
        }
    }
    if (!growthVolume) {
        growthVolume = volumeContext.activeVolume;
        growthVolumeId = volumeContext.activeVolumeId;
    }
    if (!growthVolume) {
        showStatus(tr("No volume available for growth."), kStatusMedium);
        return false;
    }

    const QString copyCheckpointPath = _context.widget->copyCheckpointPath().trimmed();
    const QString pythonPath = _context.widget->neuralPythonPath();
    const QString volumeZarr = _context.widget->volumeZarrPath().trimmed();
    const int volumeScale = _context.widget->neuralVolumeScale();
    const DenseTtaMode ttaMode = _context.widget->denseTtaMode();
    const QString ttaMergeMethod = _context.widget->denseTtaMergeMethod().trimmed();
    const double ttaOutlierDropThresh = _context.widget->denseTtaOutlierDropThresh();
    const bool usingCopyLatestPreset = copyCheckpointPath == kCopyLatestSentinel;

    if (copyCheckpointPath.isEmpty()) {
        showStatus(tr("Displacement copy requires a checkpoint path."), kStatusLong);
        return false;
    }
    if (!usingCopyLatestPreset &&
        (!QFileInfo::exists(copyCheckpointPath) || !QFileInfo(copyCheckpointPath).isFile())) {
        showStatus(tr("Copy checkpoint does not exist: %1").arg(copyCheckpointPath), kStatusLong);
        return false;
    }
    if (volumeZarr.isEmpty()) {
        showStatus(tr("Displacement copy requires a volume zarr path."), kStatusLong);
        return false;
    }

    auto& serviceManager = NeuralTraceServiceManager::instance();
    showStatus(tr("Starting neural trace service for displacement copy..."), kStatusLong);
    if (!serviceManager.ensureServiceRunning(copyCheckpointPath, volumeZarr, volumeScale, pythonPath)) {
        const QString error = serviceManager.lastError();
        showStatus(tr("Failed to start neural trace service: %1").arg(error), kStatusLong);
        return false;
    }

    const QString socketPath = serviceManager.socketPath();
    if (socketPath.isEmpty()) {
        showStatus(tr("Neural trace service is running but socket path is unavailable."), kStatusLong);
        return false;
    }

    const QString segmentationPath = QString::fromStdString(segmentationSurface->path.string());
    const QString snapshotDir = segmentationPath.isEmpty()
        ? QString()
        : QFileInfo(segmentationPath).dir().absolutePath();
    const QString snapshotPath = createDenseSnapshotPath(snapshotDir);
    try {
        const QFileInfo snapshotInfo(snapshotPath);
        const std::string snapshotId = snapshotInfo.fileName().toStdString();

        const std::filesystem::path originalSurfacePath = segmentationSurface->path;
        const std::string originalSurfaceId = segmentationSurface->id;
        std::unique_ptr<nlohmann::json> originalSurfaceMeta;
        if (segmentationSurface->meta) {
            originalSurfaceMeta = std::make_unique<nlohmann::json>(*segmentationSurface->meta);
        }

        const auto restoreLiveSurfaceState = [&]() {
            segmentationSurface->path = originalSurfacePath;
            segmentationSurface->id = originalSurfaceId;
            if (originalSurfaceMeta) {
                segmentationSurface->meta = std::make_unique<nlohmann::json>(*originalSurfaceMeta);
            } else {
                segmentationSurface->meta.reset();
            }
        };

        try {
            segmentationSurface->save(snapshotPath.toStdString(), snapshotId, false);
        } catch (...) {
            restoreLiveSurfaceState();
            throw;
        }
        restoreLiveSurfaceState();
    } catch (const std::exception& ex) {
        showStatus(tr("Failed to prepare displacement copy input: %1").arg(ex.what()), kStatusLong);
        return false;
    }

    std::optional<nlohmann::json> customParams;
    if (auto parsed = _context.widget->customParamsJson()) {
        customParams = std::move(*parsed);
    }

    CopyDisplacementJob copyJob;
    copyJob.socketPath = socketPath;
    copyJob.tifxyzInputPath = snapshotPath;
    copyJob.volumeZarrPath = volumeZarr;
    copyJob.volumeScale = volumeScale;
    copyJob.checkpointPath = copyCheckpointPath;
    copyJob.ttaMode = ttaMode;
    copyJob.ttaMergeMethod = ttaMergeMethod.isEmpty()
        ? QStringLiteral("vector_geomedian")
        : ttaMergeMethod;
    copyJob.ttaOutlierDropThresh = std::max(0.01, ttaOutlierDropThresh);
    copyJob.customParams = customParams;

    _running = true;
    _context.module->setGrowthInProgress(true);

    ActiveRequest pending;
    pending.volumeContext = volumeContext;
    pending.growthVolume = growthVolume;
    pending.growthVolumeId = growthVolumeId;
    pending.segmentationSurface = segmentationSurface;
    pending.growthVoxelSize = growthVolume->voxelSize();
    pending.usingCorrections = false;
    pending.inpaintOnly = false;
    pending.denseDisplacement = false;
    pending.denseCreateNewSegment = true;
    pending.copyDisplacement = true;
    _activeRequest = std::move(pending);

    showStatus(tr("Running displacement copy (creating front/back segments)..."), kStatusMedium);

    auto future = QtConcurrent::run(runCopyDisplacementGrowth, copyJob);
    _watcher = std::make_unique<QFutureWatcher<TracerGrowthResult>>(this);
    connect(_watcher.get(), &QFutureWatcher<TracerGrowthResult>::finished,
            this, &SegmentationGrower::onFutureFinished);
    _watcher->setFuture(future);

    return true;
}

void SegmentationGrower::finalize(bool ok)
{
    if (_context.module) {
        _context.module->setGrowthInProgress(false);
    }
    _running = false;
    _activeRequest.reset();
}

void SegmentationGrower::handleFailure(const QString& message)
{
    auto showStatus = [&](const QString& text, int timeout) {
        if (_callbacks.showStatus) {
            _callbacks.showStatus(text, timeout);
        }
    };
    if (!message.isEmpty()) {
        showStatus(message, kStatusLong);
    }
    finalize(false);
}

// Compute bounding box of cells that changed between two point matrices
// Returns bounds in the NEW surface's coordinate system
static cv::Rect computeChangedBounds(const cv::Mat_<cv::Vec3f>& oldPts,
                                     const cv::Mat_<cv::Vec3f>& newPts)
{
    // Handle size differences - the new surface may have padding around it
    // The old content is centered in the new surface
    const int padX = (newPts.cols - oldPts.cols) / 2;
    const int padY = (newPts.rows - oldPts.rows) / 2;

    int minRow = newPts.rows, maxRow = -1;
    int minCol = newPts.cols, maxCol = -1;

    // Compare the overlapping region (old surface embedded in new)
    for (int oldRow = 0; oldRow < oldPts.rows; ++oldRow) {
        const int newRow = oldRow + padY;
        if (newRow < 0 || newRow >= newPts.rows) continue;

        for (int oldCol = 0; oldCol < oldPts.cols; ++oldCol) {
            const int newCol = oldCol + padX;
            if (newCol < 0 || newCol >= newPts.cols) continue;

            const auto& o = oldPts(oldRow, oldCol);
            const auto& n = newPts(newRow, newCol);
            if (o[0] != n[0] || o[1] != n[1] || o[2] != n[2]) {
                minRow = std::min(minRow, newRow);
                maxRow = std::max(maxRow, newRow);
                minCol = std::min(minCol, newCol);
                maxCol = std::max(maxCol, newCol);
            }
        }
    }

    // Check padding cells for valid (non-empty) content that was added
    // Invalid points have x == -1
    auto isValid = [](const cv::Vec3f& p) { return p[0] != -1.0f; };

    // Check top/bottom padding rows
    for (int row = 0; row < padY && row < newPts.rows; ++row) {
        for (int col = 0; col < newPts.cols; ++col) {
            if (isValid(newPts(row, col))) {
                minRow = std::min(minRow, row);
                maxRow = std::max(maxRow, row);
                minCol = std::min(minCol, col);
                maxCol = std::max(maxCol, col);
            }
        }
    }
    for (int row = newPts.rows - padY; row < newPts.rows; ++row) {
        if (row < 0) continue;
        for (int col = 0; col < newPts.cols; ++col) {
            if (isValid(newPts(row, col))) {
                minRow = std::min(minRow, row);
                maxRow = std::max(maxRow, row);
                minCol = std::min(minCol, col);
                maxCol = std::max(maxCol, col);
            }
        }
    }
    // Check left/right padding cols
    for (int row = 0; row < newPts.rows; ++row) {
        for (int col = 0; col < padX && col < newPts.cols; ++col) {
            if (isValid(newPts(row, col))) {
                minRow = std::min(minRow, row);
                maxRow = std::max(maxRow, row);
                minCol = std::min(minCol, col);
                maxCol = std::max(maxCol, col);
            }
        }
        for (int col = newPts.cols - padX; col < newPts.cols; ++col) {
            if (col < 0) continue;
            if (isValid(newPts(row, col))) {
                minRow = std::min(minRow, row);
                maxRow = std::max(maxRow, row);
                minCol = std::min(minCol, col);
                maxCol = std::max(maxCol, col);
            }
        }
    }

    if (maxRow < 0) {
        return cv::Rect();  // No changes
    }

    return cv::Rect(minCol, minRow, maxCol - minCol + 1, maxRow - minRow + 1);
}

void SegmentationGrower::onFutureFinished()
{
    if (!_watcher) {
        finalize(false);
        return;
    }

    const TracerGrowthResult result = _watcher->result();
    _watcher.reset();

    if (!_activeRequest) {
        finalize(false);
        return;
    }

    auto showStatus = [&](const QString& text, int timeout) {
        if (_callbacks.showStatus) {
            _callbacks.showStatus(text, timeout);
        }
    };

    ActiveRequest request = std::move(*_activeRequest);
    _activeRequest.reset();

    auto cleanupDisplacementTemporarySurfaces = [&](const std::filesystem::path& preservePath = std::filesystem::path()) {
        if (!request.denseDisplacement && !request.copyDisplacement) {
            return;
        }
        if (result.temporarySurfacePaths.empty()) {
            return;
        }

        const std::filesystem::path preserved = preservePath.lexically_normal();
        std::unordered_set<std::string> visited;
        for (const auto& rawPath : result.temporarySurfacePaths) {
            if (rawPath.empty()) {
                continue;
            }
            const std::filesystem::path candidate = rawPath.lexically_normal();
            if (!preserved.empty() && candidate == preserved) {
                continue;
            }

            const std::string key = candidate.string();
            if (!visited.insert(key).second) {
                continue;
            }

            std::error_code existsEc;
            if (!std::filesystem::exists(candidate, existsEc) || existsEc) {
                continue;
            }

            std::error_code rmEc;
            std::filesystem::remove_all(candidate, rmEc);
            if (rmEc) {
                qCInfo(lcSegGrowth) << "Failed to remove displacement temporary surface"
                                    << QString::fromStdString(candidate.string())
                                    << QString::fromStdString(rmEc.message());
            }
        }
    };

    if (!result.error.isEmpty()) {
        qCInfo(lcSegGrowth) << "Tracer growth error" << result.error;
        cleanupDisplacementTemporarySurfaces(request.segmentationSurface
                                                 ? request.segmentationSurface->path
                                                 : std::filesystem::path());
        showStatus(result.error, kStatusLong);
        finalize(false);
        return;
    }

    if (request.copyDisplacement) {
        if (!request.volumeContext.package) {
            showStatus(tr("Displacement copy requires an active volume package."), kStatusLong);
            cleanupDisplacementTemporarySurfaces();
            finalize(false);
            return;
        }

        if (result.temporarySurfacePaths.size() < 3) {
            showStatus(tr("Displacement copy did not produce front/back outputs."), kStatusLong);
            cleanupDisplacementTemporarySurfaces();
            finalize(false);
            return;
        }

        const std::filesystem::path frontPath = result.temporarySurfacePaths[result.temporarySurfacePaths.size() - 2];
        const std::filesystem::path backPath = result.temporarySurfacePaths.back();
        if (frontPath.empty() || backPath.empty()) {
            showStatus(tr("Displacement copy did not produce front/back outputs."), kStatusLong);
            cleanupDisplacementTemporarySurfaces();
            finalize(false);
            return;
        }

        std::unique_ptr<QuadSurface> frontSurface;
        std::unique_ptr<QuadSurface> backSurface;
        try {
            frontSurface = load_quad_from_tifxyz(frontPath.string());
            backSurface = load_quad_from_tifxyz(backPath.string());
        } catch (const std::exception& ex) {
            showStatus(tr("Failed to load displacement copy outputs: %1").arg(ex.what()), kStatusLong);
            cleanupDisplacementTemporarySurfaces();
            finalize(false);
            return;
        }
        if (!frontSurface || !backSurface) {
            showStatus(tr("Failed to load displacement copy outputs."), kStatusLong);
            cleanupDisplacementTemporarySurfaces();
            finalize(false);
            return;
        }

        const std::filesystem::path volpkgRoot(request.volumeContext.package->getVolpkgDirectory());
        const std::string segmentationDir = request.volumeContext.package->getSegmentationDirectory();
        const std::filesystem::path segmentsDir = volpkgRoot / segmentationDir;
        std::error_code mkdirEc;
        std::filesystem::create_directories(segmentsDir, mkdirEc);
        if (mkdirEc) {
            showStatus(tr("Failed to create %1 directory for displacement copy: %2")
                           .arg(QString::fromStdString(segmentationDir))
                           .arg(QString::fromStdString(mkdirEc.message())),
                       kStatusLong);
            cleanupDisplacementTemporarySurfaces();
            finalize(false);
            return;
        }

        auto persistAsNewSegment = [&](QuadSurface* surface, const std::string& fallbackBaseId) -> std::optional<std::string> {
            if (!surface) {
                return std::nullopt;
            }
            const std::string baseId = sanitizeSegmentId(surface->id.empty() ? fallbackBaseId : surface->id);
            const std::string newSegmentId = makeUniqueSegmentId(segmentsDir, baseId);
            const std::filesystem::path newSegmentPath = segmentsDir / newSegmentId;
            try {
                surface->save(newSegmentPath.string(), newSegmentId, false);
            } catch (const std::exception& ex) {
                showStatus(tr("Failed to save displacement copy result: %1").arg(ex.what()), kStatusLong);
                return std::nullopt;
            }

            if (!request.volumeContext.package->addSingleSegmentation(newSegmentId)) {
                request.volumeContext.package->refreshSegmentations();
            }
            return newSegmentId;
        };

        const auto frontId = persistAsNewSegment(frontSurface.get(), "copy_displacement_front");
        const auto backId = persistAsNewSegment(backSurface.get(), "copy_displacement_back");
        if (!frontId || !backId) {
            cleanupDisplacementTemporarySurfaces();
            finalize(false);
            return;
        }

        cleanupDisplacementTemporarySurfaces();

        if (_surfacePanel) {
            _surfacePanel->addSingleSegmentation(*frontId);
            _surfacePanel->refreshSurfaceMetrics(*frontId);
            _surfacePanel->addSingleSegmentation(*backId);
            _surfacePanel->refreshSurfaceMetrics(*backId);
        } else if (_context.surfaces) {
            auto loadedFront = request.volumeContext.package->loadSurface(*frontId);
            if (loadedFront) {
                _context.surfaces->setSurface(*frontId, loadedFront, true, false);
            }
            auto loadedBack = request.volumeContext.package->loadSurface(*backId);
            if (loadedBack) {
                _context.surfaces->setSurface(*backId, loadedBack, true, false);
            }
        }

        refreshSegmentationViewers(_context.viewerManager);
        showStatus(
            tr("Displacement copy created new segments '%1' and '%2'.")
                .arg(QString::fromStdString(*frontId))
                .arg(QString::fromStdString(*backId)),
            kStatusLong);
        finalize(true);
        return;
    }

    if (!result.surface) {
        qCInfo(lcSegGrowth) << "Tracer growth returned null surface";
        cleanupDisplacementTemporarySurfaces(request.segmentationSurface
                                                 ? request.segmentationSurface->path
                                                 : std::filesystem::path());
        showStatus(tr("Tracer growth did not return a surface."), kStatusMedium);
        finalize(false);
        return;
    }

    if (request.denseDisplacement && request.denseCreateNewSegment) {
        if (!request.volumeContext.package) {
            showStatus(tr("Dense displacement new-segment mode requires an active volume package."), kStatusLong);
            cleanupDisplacementTemporarySurfaces();
            delete result.surface;
            finalize(false);
            return;
        }

        const std::filesystem::path volpkgRoot(request.volumeContext.package->getVolpkgDirectory());
        const std::string segmentationDir = request.volumeContext.package->getSegmentationDirectory();
        const std::filesystem::path segmentsDir = volpkgRoot / segmentationDir;
        std::error_code mkdirEc;
        std::filesystem::create_directories(segmentsDir, mkdirEc);
        if (mkdirEc) {
            showStatus(tr("Failed to create %1 directory for new segment: %2")
                           .arg(QString::fromStdString(segmentationDir))
                           .arg(QString::fromStdString(mkdirEc.message())),
                       kStatusLong);
            cleanupDisplacementTemporarySurfaces();
            delete result.surface;
            finalize(false);
            return;
        }

        const std::string baseId = sanitizeSegmentId(
            result.surface->id.empty() ? std::string("dense_displacement") : result.surface->id);
        const std::string newSegmentId = makeUniqueSegmentId(segmentsDir, baseId);
        const std::filesystem::path newSegmentPath = segmentsDir / newSegmentId;

        try {
            result.surface->save(newSegmentPath.string(), newSegmentId, false);
        } catch (const std::exception& ex) {
            showStatus(tr("Failed to save dense displacement result: %1").arg(ex.what()), kStatusLong);
            cleanupDisplacementTemporarySurfaces();
            delete result.surface;
            finalize(false);
            return;
        }

        // The new directory exists on disk now, but VolumePkg may not know this ID yet.
        // Register it before any loadSurface call.
        if (!request.volumeContext.package->addSingleSegmentation(newSegmentId)) {
            request.volumeContext.package->refreshSegmentations();
        }

        cleanupDisplacementTemporarySurfaces(newSegmentPath);
        delete result.surface;

        if (_surfacePanel) {
            _surfacePanel->addSingleSegmentation(newSegmentId);
            _surfacePanel->refreshSurfaceMetrics(newSegmentId);
        } else if (_context.surfaces) {
            auto loadedSurface = request.volumeContext.package->loadSurface(newSegmentId);
            if (loadedSurface) {
                _context.surfaces->setSurface(newSegmentId, loadedSurface, true, false);
            }
        }

        refreshSegmentationViewers(_context.viewerManager);
        showStatus(
            tr("Dense displacement created new segment '%1'.")
                .arg(QString::fromStdString(newSegmentId)),
            kStatusLong);
        finalize(true);
        return;
    }

    const double voxelSize = request.growthVoxelSize;
    cv::Mat generations = result.surface->channel("generations");

    std::vector<SurfacePatchIndex::SurfacePtr> surfacesToUpdate;
    surfacesToUpdate.reserve(3);
    auto appendUniqueSurface = [&](const SurfacePatchIndex::SurfacePtr& surface) {
        if (!surface) {
            return;
        }
        if (std::find(surfacesToUpdate.begin(), surfacesToUpdate.end(), surface) == surfacesToUpdate.end()) {
            surfacesToUpdate.push_back(surface);
        }
    };

    appendUniqueSurface(request.segmentationSurface);
    if (_context.module && _context.module->hasActiveSession()) {
        appendUniqueSurface(_context.module->activeBaseSurfaceShared());
    }

    QuadSurface* primarySurface = surfacesToUpdate.empty() ? nullptr : surfacesToUpdate.front().get();
    cv::Mat_<cv::Vec3f>* primaryPoints = primarySurface ? primarySurface->rawPointsPtr() : nullptr;
    cv::Mat_<cv::Vec3f>* resultPoints = result.surface->rawPointsPtr();

    // Compute the changed region before swapping points (for efficient R-tree update)
    // Only use region update when sizes match; otherwise grid coordinates shift
    cv::Rect changedBounds;
    bool sizeChanged = false;
    if (primarySurface && primaryPoints && resultPoints) {
        sizeChanged = (primaryPoints->size() != resultPoints->size());
        if (!sizeChanged) {
            changedBounds = computeChangedBounds(*primaryPoints, *resultPoints);
            qCInfo(lcSegGrowth) << "Changed bounds:" << changedBounds.x << changedBounds.y
                                << changedBounds.width << "x" << changedBounds.height
                                << "(surface:" << resultPoints->cols << "x" << resultPoints->rows << ")";
        } else {
            qCInfo(lcSegGrowth) << "Surface size changed:" << primaryPoints->cols << "x" << primaryPoints->rows
                                << "->" << resultPoints->cols << "x" << resultPoints->rows
                                << "- using full update";
        }
    }

    if (primarySurface && primaryPoints && resultPoints) {
        std::swap(*primaryPoints, *resultPoints);
        primarySurface->invalidateCache();
    } else if (primarySurface && primaryPoints) {
        result.surface->rawPoints().copyTo(*primaryPoints);
        primarySurface->invalidateCache();
    }

    for (const auto& targetSurfacePtr : surfacesToUpdate) {
        QuadSurface* targetSurface = targetSurfacePtr.get();
        if (!targetSurface) {
            continue;
        }

        if (targetSurface != primarySurface) {
            if (auto* destPoints = targetSurface->rawPointsPtr()) {
                if (primaryPoints) {
                    primaryPoints->copyTo(*destPoints);
                } else if (resultPoints) {
                    resultPoints->copyTo(*destPoints);
                } else {
                    result.surface->rawPoints().copyTo(*destPoints);
                }
            }
            targetSurface->invalidateCache();
        }

        nlohmann::json preservedTags = nlohmann::json::object();
        bool hadPreservedTags = false;
        if (targetSurface->meta && targetSurface->meta->is_object()) {
            auto tagsIt = targetSurface->meta->find("tags");
            if (tagsIt != targetSurface->meta->end() && tagsIt->is_object()) {
                preservedTags = *tagsIt;
                hadPreservedTags = true;
            }
        }

        if (!generations.empty()) {
            targetSurface->setChannel("generations", generations);
        }

        // Copy preserved approval mask from result surface
        cv::Mat approval = result.surface->channel("approval", SURF_CHANNEL_NORESIZE);
        if (!approval.empty()) {
            targetSurface->setChannel("approval", approval);
        }

        if (result.surface->meta) {
            targetSurface->meta = std::make_unique<nlohmann::json>(*result.surface->meta);
        } else {
            ensureSurfaceMetaObject(targetSurface);
        }

        if (hadPreservedTags && targetSurface->meta && targetSurface->meta->is_object()) {
            nlohmann::json mergedTags = preservedTags;
            auto tagsIt = targetSurface->meta->find("tags");
            if (tagsIt != targetSurface->meta->end() && tagsIt->is_object()) {
                mergedTags.update(*tagsIt);
            }
            (*targetSurface->meta)["tags"] = mergedTags;
        }

        updateSegmentationSurfaceMetadata(targetSurface, voxelSize);

        // Refresh intersection index for this surface so renderIntersections() has up-to-date data
        if (_context.viewerManager) {
            if (!changedBounds.empty()) {
                _context.viewerManager->refreshSurfacePatchIndex(targetSurfacePtr, changedBounds);
            } else {
                _context.viewerManager->refreshSurfacePatchIndex(targetSurfacePtr);
            }
        }
    }

    QuadSurface* surfaceToPersist = nullptr;
    const bool sessionActive = _context.module && _context.module->hasActiveSession();
    if (sessionActive) {
        surfaceToPersist = _context.module->activeBaseSurface();
    }
    if (!surfaceToPersist) {
        surfaceToPersist = request.segmentationSurface.get();
    }

    // Mask is no longer valid after growth/inpainting
    if (surfaceToPersist) {
        surfaceToPersist->invalidateMask();
    }

    if (!sessionActive) {
        try {
            if (surfaceToPersist) {
                ensureSurfaceMetaObject(surfaceToPersist);
                surfaceToPersist->saveOverwrite();

                // Dense overwrite mode persists an updated segment in-place;
                // force reload from disk so in-app references observe the new geometry.
                if (request.denseDisplacement && !request.denseCreateNewSegment) {
                    std::shared_ptr<QuadSurface> reloadedSurface;
                    const std::string persistedId = surfaceToPersist->id;

                    if (request.volumeContext.package && !persistedId.empty()) {
                        request.volumeContext.package->unloadSurface(persistedId);
                        reloadedSurface = request.volumeContext.package->loadSurface(persistedId);
                    }

                    if (!reloadedSurface && !surfaceToPersist->path.empty()) {
                        try {
                            auto loadedFromDisk = load_quad_from_tifxyz(surfaceToPersist->path.string());
                            if (loadedFromDisk) {
                                reloadedSurface.reset(loadedFromDisk.release());
                            }
                        } catch (const std::exception& ex) {
                            qCInfo(lcSegGrowth) << "Dense overwrite reload fallback failed" << ex.what();
                        }
                    }

                    if (reloadedSurface) {
                        if (_context.surfaces && !persistedId.empty() && _context.surfaces->surface(persistedId)) {
                            _context.surfaces->setSurface(persistedId, reloadedSurface, true, false);
                        }
                        request.segmentationSurface = reloadedSurface;
                        surfaceToPersist = reloadedSurface.get();
                    } else {
                        qCInfo(lcSegGrowth) << "Dense overwrite reload skipped; using in-memory geometry";
                    }
                }
            }
        } catch (const std::exception& ex) {
            qCInfo(lcSegGrowth) << "Failed to save tracer result" << ex.what();
            showStatus(tr("Failed to save segmentation: %1").arg(ex.what()), kStatusLong);
        }
    } else if (_context.module) {
        _context.module->requestAutosaveFromGrowth();
    }

    std::vector<std::pair<CVolumeViewer*, bool>> resetDefaults;
    if (_context.viewerManager) {
        ViewerManager* manager = _context.viewerManager;
        manager->forEachViewer([manager, &resetDefaults](CVolumeViewer* viewer) {
            if (!viewer || viewer->surfName() != "segmentation") {
                return;
            }
            const bool defaultReset = manager->resetDefaultFor(viewer);
            resetDefaults.emplace_back(viewer, defaultReset);
            viewer->setResetViewOnSurfaceChange(false);
        });
    }

    if (_context.surfaces) {
        _context.surfaces->setSurface("segmentation", request.segmentationSurface, false, true);
        // Note: SurfacePatchIndex is automatically updated via handleSurfaceChanged signal
    }

    if (!resetDefaults.empty()) {
        const bool editingActive = _context.module && _context.module->editingEnabled();
        for (auto& entry : resetDefaults) {
            auto* viewer = entry.first;
            if (!viewer) {
                continue;
            }
            if (editingActive) {
                viewer->setResetViewOnSurfaceChange(false);
            } else {
                viewer->setResetViewOnSurfaceChange(entry.second);
            }
        }
    }

    if (sessionActive && _context.module) {
        _context.module->markNextHandlesFromGrowth();
        bool appliedIncremental = false;
        if (request.correctionsAffectedBounds) {
            appliedIncremental = _context.module->applySurfaceUpdateFromGrowth(*request.correctionsAffectedBounds);
        }
        if (!appliedIncremental) {
            qCInfo(lcSegGrowth) << "Refreshing active segmentation session after tracer growth";
            _context.module->refreshSessionFromSurface(surfaceToPersist);
        }
    }

    QuadSurface* currentSegSurface = nullptr;
    std::shared_ptr<Surface> currentSegSurfaceHolder;  // Keep surface alive during this scope
    if (_context.surfaces) {
        currentSegSurfaceHolder = _context.surfaces->surface("segmentation");
        currentSegSurface = dynamic_cast<QuadSurface*>(currentSegSurfaceHolder.get());
    }
    if (!currentSegSurface) {
        currentSegSurface = request.segmentationSurface.get();
    }

    // Update approval tool after surface replacement (handles case with no active editing session)
    if (_context.module) {
        _context.module->updateApprovalToolAfterGrowth(currentSegSurface);
    }

    QuadSurface* metaSurface = surfaceToPersist ? surfaceToPersist : request.segmentationSurface.get();
    synchronizeSurfaceMeta(request.volumeContext.package, metaSurface, _surfacePanel);

    if (_surfacePanel) {
        std::vector<std::string> idsToRefresh;
        idsToRefresh.reserve(surfacesToUpdate.size() + 1);

        auto maybeAddId = [&idsToRefresh](QuadSurface* surface) {
            if (!surface) {
                return;
            }
            const std::string& surfaceId = surface->id;
            if (surfaceId.empty()) {
                return;
            }
            if (std::find(idsToRefresh.begin(), idsToRefresh.end(), surfaceId) == idsToRefresh.end()) {
                idsToRefresh.push_back(surfaceId);
            }
        };

        for (const auto& surface : surfacesToUpdate) {
            maybeAddId(surface.get());
        }
        maybeAddId(currentSegSurface);
        if (_context.module && _context.module->hasActiveSession()) {
            maybeAddId(_context.module->activeBaseSurfaceShared().get());
        }

        for (const auto& id : idsToRefresh) {
            _surfacePanel->refreshSurfaceMetrics(id);
        }
    }

    if (_callbacks.applySliceOrientation) {
        _callbacks.applySliceOrientation(currentSegSurface);
    }

    refreshSegmentationViewers(_context.viewerManager);

    // Save corrections annotation if we have a before-crop and bounds
    if (request.usingCorrections && request.beforeCrop && request.correctionsBounds) {
        // Crop the "after" surface using the same grid region
        auto afterCrop = cropSurfaceToGridRegion(primarySurface ? primarySurface : request.segmentationSurface.get(),
                                                  request.correctionsBounds->gridRegion);
        if (afterCrop) {
            // Get volpkg root from the package
            std::filesystem::path volpkgRoot;
            std::vector<std::string> volumeIds;
            if (request.volumeContext.package) {
                volpkgRoot = std::filesystem::path(request.volumeContext.package->getVolpkgDirectory());
                volumeIds = request.volumeContext.package->volumeIDs();
            }

            if (!volpkgRoot.empty()) {
                saveCorrectionsAnnotation(
                    volpkgRoot,
                    request.segmentationSurface ? request.segmentationSurface->id : "",
                    request.beforeCrop.get(),
                    afterCrop.get(),
                    request.corrections,
                    volumeIds,
                    request.growthVolumeId,
                    *request.correctionsBounds);
            }
        }
    }

    // Apply grid offset to correction anchors (for surface growth remapping)
    // and save immediately to persist updated positions
    if (result.surface && result.surface->meta && _context.module) {
        auto offsetIt = result.surface->meta->find("grid_offset");
        if (offsetIt != result.surface->meta->end() && offsetIt->is_array() && offsetIt->size() == 2) {
            float offsetX = (*offsetIt)[0].get<float>();
            float offsetY = (*offsetIt)[1].get<float>();
            if (offsetX != 0.0f || offsetY != 0.0f) {
                _context.module->applyCorrectionAnchorOffset(offsetX, offsetY);
                // Save corrections with updated anchors to the new surface path
                if (surfaceToPersist && !surfaceToPersist->path.empty()) {
                    _context.module->saveCorrectionPoints(surfaceToPersist->path);
                }
            }
        }
    }

    if (request.usingCorrections && _context.module) {
        _context.module->clearPendingCorrections();
    }

    qCInfo(lcSegGrowth) << "Tracer growth completed successfully";

    cleanupDisplacementTemporarySurfaces(surfaceToPersist ? surfaceToPersist->path : std::filesystem::path());
    delete result.surface;

    QString message;
    if (!result.statusMessage.isEmpty()) {
        message = result.statusMessage;
    } else if (request.usingCorrections) {
        message = tr("Corrections applied; tracer growth complete.");
    } else if (request.inpaintOnly) {
        message = tr("Tracer inpainting complete.");
    } else {
        message = tr("Tracer growth complete.");
    }
    showStatus(message, kStatusLong);

    finalize(true);
}
