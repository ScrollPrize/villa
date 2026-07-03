#include "volume_viewers/annotation_tools/SameWrapAnnotationTool.hpp"

#include "vc/ui/VCCollection.hpp"

#include "dijkstra3d.hpp"

#include <QBrush>
#include <QColor>
#include <QGraphicsEllipseItem>
#include <QGraphicsPathItem>
#include <QPainterPath>
#include <QPen>
#include <QRectF>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iterator>
#include <optional>
#include <queue>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

namespace {
constexpr const char* kSameWrapAnnotationOverlayKey = "same_wrap_annotation_preview";
constexpr qreal kNearbyCollectionColorRadiusPx = 160.0;

bool isSameWrapCollectionName(const std::string& name)
{
    return std::string_view(name).rfind("same_wrap", 0) == 0;
}

std::vector<ColPoint> orderedCollectionPoints(const VCCollection::Collection& collection)
{
    std::vector<ColPoint> points;
    points.reserve(collection.points.size());
    for (const auto& [id, point] : collection.points) {
        points.push_back(point);
    }
    std::sort(points.begin(), points.end(), [](const ColPoint& a, const ColPoint& b) {
        if (a.creation_time != b.creation_time) {
            return a.creation_time < b.creation_time;
        }
        return a.id < b.id;
    });
    return points;
}

std::vector<cv::Vec3f> orderedCollectionPositions(const VCCollection::Collection& collection)
{
    std::vector<ColPoint> points = orderedCollectionPoints(collection);
    std::vector<cv::Vec3f> positions;
    positions.reserve(points.size());
    for (const ColPoint& point : points) {
        positions.push_back(point.p);
    }
    return positions;
}

float endpointGap(const std::vector<cv::Vec3f>& first, const std::vector<cv::Vec3f>& second)
{
    if (first.empty() || second.empty()) {
        return std::numeric_limits<float>::infinity();
    }
    return cv::norm(first.back() - second.front());
}

std::vector<cv::Vec3f> reversedCopy(std::vector<cv::Vec3f> points)
{
    std::reverse(points.begin(), points.end());
    return points;
}

std::vector<cv::Vec3f> bestEndpointMergeOrder(const VCCollection::Collection& firstCollection,
                                              const VCCollection::Collection& secondCollection)
{
    const std::vector<cv::Vec3f> firstOrdered = orderedCollectionPositions(firstCollection);
    const std::vector<cv::Vec3f> secondOrdered = orderedCollectionPositions(secondCollection);
    if (firstOrdered.empty() || secondOrdered.empty()) {
        return {};
    }

    std::vector<cv::Vec3f> bestFirst;
    std::vector<cv::Vec3f> bestSecond;
    float bestGap = std::numeric_limits<float>::infinity();

    const std::array<std::vector<cv::Vec3f>, 2> firstVariants = {
        firstOrdered,
        reversedCopy(firstOrdered)
    };
    const std::array<std::vector<cv::Vec3f>, 2> secondVariants = {
        secondOrdered,
        reversedCopy(secondOrdered)
    };

    for (const std::vector<cv::Vec3f>& first : firstVariants) {
        for (const std::vector<cv::Vec3f>& second : secondVariants) {
            const float gap = endpointGap(first, second);
            if (gap < bestGap) {
                bestGap = gap;
                bestFirst = first;
                bestSecond = second;
            }
        }
    }

    if (bestFirst.empty() || bestSecond.empty()) {
        return {};
    }
    if (bestGap <= 0.5f) {
        bestSecond.erase(bestSecond.begin());
    }
    bestFirst.insert(bestFirst.end(), bestSecond.begin(), bestSecond.end());
    return bestFirst;
}

std::vector<cv::Vec3f> removeClosePoints(const std::vector<cv::Vec3f>& points, float minSpacing)
{
    if (points.size() < 2) {
        return points;
    }

    minSpacing = std::max(0.0f, minSpacing);
    const float minSpacing2 = minSpacing * minSpacing;
    std::vector<cv::Vec3f> filtered;
    filtered.reserve(points.size());
    for (const cv::Vec3f& point : points) {
        if (filtered.empty()) {
            filtered.push_back(point);
            continue;
        }
        const cv::Vec3f delta = point - filtered.back();
        if (delta.dot(delta) >= minSpacing2) {
            filtered.push_back(point);
        }
    }
    return filtered;
}

std::string dominantDirectionKey(const std::vector<cv::Vec3f>& path)
{
    float xyMotion = 0.0f;
    float zMotion = 0.0f;
    for (size_t i = 1; i < path.size(); ++i) {
        const cv::Vec3f delta = path[i] - path[i - 1];
        xyMotion += std::hypot(delta[0], delta[1]);
        zMotion += std::abs(delta[2]);
    }
    return zMotion > xyMotion ? "z" : "xy";
}

float rgbDistance2(const cv::Vec3f& a, const cv::Vec3f& b)
{
    const cv::Vec3f delta = a - b;
    return delta.dot(delta);
}

cv::Vec3f qColorToVec3f(const QColor& color)
{
    return {
        static_cast<float>(color.redF()),
        static_cast<float>(color.greenF()),
        static_cast<float>(color.blueF())
    };
}

struct WeightedColor {
    cv::Vec3f color;
    float weight = 1.0f;
};

std::vector<cv::Vec3f> sameWrapCandidateColors()
{
    static constexpr std::array<QRgb, 16> palette = {
        qRgb(0, 255, 255),    // cyan
        qRgb(255, 230, 0),    // yellow
        qRgb(0, 255, 70),     // green
        qRgb(0, 120, 255),    // blue
        qRgb(255, 128, 0),    // orange
        qRgb(180, 0, 255),    // violet
        qRgb(255, 0, 180),    // magenta
        qRgb(255, 40, 40),    // red
        qRgb(64, 255, 160),   // mint
        qRgb(120, 180, 255),  // sky
        qRgb(255, 160, 220),  // pink
        qRgb(180, 255, 80),   // lime
        qRgb(0, 190, 160),    // teal
        qRgb(255, 180, 80),   // amber
        qRgb(120, 80, 255),   // indigo
        qRgb(255, 80, 120)    // coral
    };

    std::vector<cv::Vec3f> colors;
    colors.reserve(palette.size());
    for (const QRgb rgb : palette) {
        colors.push_back(qColorToVec3f(QColor::fromRgb(rgb)));
    }
    return colors;
}

float circularHueDistance(float a, float b)
{
    if (a < 0.0f || b < 0.0f) {
        return 0.0f;
    }
    const float delta = std::abs(a - b);
    return std::min(delta, 1.0f - delta);
}

float visualColorDistance(const cv::Vec3f& a, const cv::Vec3f& b)
{
    const QColor qa = QColor::fromRgbF(
        std::clamp(a[0], 0.0f, 1.0f),
        std::clamp(a[1], 0.0f, 1.0f),
        std::clamp(a[2], 0.0f, 1.0f));
    const QColor qb = QColor::fromRgbF(
        std::clamp(b[0], 0.0f, 1.0f),
        std::clamp(b[1], 0.0f, 1.0f),
        std::clamp(b[2], 0.0f, 1.0f));

    const float hueDistance = circularHueDistance(qa.hsvHueF(), qb.hsvHueF());
    const float saturationDistance = static_cast<float>(qa.hsvSaturationF() - qb.hsvSaturationF());
    const float valueDistance = static_cast<float>(qa.valueF() - qb.valueF());
    return 4.0f * hueDistance * hueDistance +
           0.75f * saturationDistance * saturationDistance +
           0.5f * valueDistance * valueDistance +
           1.25f * rgbDistance2(a, b);
}

std::vector<int> sameWrapPaletteUsage(const VCCollection* pointCollection,
                                      const std::unordered_set<uint64_t>& excludedCollectionIds,
                                      const std::vector<cv::Vec3f>& candidates)
{
    std::vector<int> usage(candidates.size(), 0);
    if (!pointCollection) {
        return usage;
    }

    constexpr float kSameColorDistance = 0.03f;
    for (const auto& [collectionId, collection] : pointCollection->getAllCollections()) {
        if (excludedCollectionIds.count(collectionId) != 0 || !isSameWrapCollectionName(collection.name)) {
            continue;
        }

        auto bestIt = candidates.end();
        float bestDistance = std::numeric_limits<float>::infinity();
        for (auto it = candidates.begin(); it != candidates.end(); ++it) {
            const float distance = visualColorDistance(collection.color, *it);
            if (distance < bestDistance) {
                bestIt = it;
                bestDistance = distance;
            }
        }
        if (bestIt != candidates.end() && bestDistance <= kSameColorDistance) {
            const auto idx = static_cast<std::size_t>(std::distance(candidates.begin(), bestIt));
            ++usage[idx];
        }
    }
    return usage;
}

std::optional<cv::Vec3f> mostRecentSameWrapColor(
    const VCCollection* pointCollection,
    const std::unordered_set<uint64_t>& excludedCollectionIds)
{
    if (!pointCollection) {
        return std::nullopt;
    }

    std::optional<cv::Vec3f> color;
    int64_t latestCreationTime = std::numeric_limits<int64_t>::min();
    uint64_t latestCollectionId = 0;
    for (const auto& [collectionId, collection] : pointCollection->getAllCollections()) {
        if (excludedCollectionIds.count(collectionId) != 0 || !isSameWrapCollectionName(collection.name)) {
            continue;
        }

        int64_t collectionCreationTime = std::numeric_limits<int64_t>::min();
        for (const auto& [pointId, point] : collection.points) {
            (void)pointId;
            collectionCreationTime = std::max(collectionCreationTime, point.creation_time);
        }
        if (collectionCreationTime > latestCreationTime ||
            (collectionCreationTime == latestCreationTime && collectionId > latestCollectionId)) {
            color = collection.color;
            latestCreationTime = collectionCreationTime;
            latestCollectionId = collectionId;
        }
    }
    return color;
}

cv::Vec3f mostDistinctColor(const std::vector<WeightedColor>& avoidedColors,
                            const std::vector<int>& paletteUsage)
{
    const std::vector<cv::Vec3f> candidates = sameWrapCandidateColors();
    cv::Vec3f best = candidates.front();
    float bestMinDistance = -1.0f;
    float bestTotalDistance = -1.0f;
    int bestUsage = std::numeric_limits<int>::max();

    int minUsage = 0;
    if (paletteUsage.size() == candidates.size()) {
        minUsage = *std::min_element(paletteUsage.begin(), paletteUsage.end());
    }

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const int usage = paletteUsage.size() == candidates.size() ? paletteUsage[i] : 0;
        if (usage != minUsage) {
            continue;
        }

        const cv::Vec3f& candidate = candidates[i];
        float minDistance = std::numeric_limits<float>::infinity();
        float totalDistance = 0.0f;
        for (const WeightedColor& avoidedColor : avoidedColors) {
            const float distance = visualColorDistance(candidate, avoidedColor.color);
            minDistance = std::min(minDistance, distance);
            totalDistance += distance * avoidedColor.weight;
        }
        if (usage < bestUsage ||
            (usage == bestUsage && minDistance > bestMinDistance) ||
            (minDistance == bestMinDistance && totalDistance > bestTotalDistance)) {
            best = candidate;
            bestUsage = usage;
            bestMinDistance = minDistance;
            bestTotalDistance = totalDistance;
        }
    }
    return best;
}

std::optional<cv::Vec3f> colorDistinctFromVisibleCollections(
    const VCCollection* pointCollection,
    const std::vector<cv::Vec3f>& points,
    const std::unordered_set<uint64_t>& excludedCollectionIds,
    const std::vector<cv::Vec3f>& extraAvoidedColors,
    const SameWrapAnnotationTool::VolumeToSceneFn& volumeToScene,
    const QRectF& visibleSceneRect)
{
    if (!pointCollection || points.empty()) {
        return std::nullopt;
    }

    const std::vector<cv::Vec3f> candidates = sameWrapCandidateColors();
    const std::vector<int> paletteUsage =
        sameWrapPaletteUsage(pointCollection, excludedCollectionIds, candidates);

    std::vector<WeightedColor> avoidedColors;
    avoidedColors.reserve(extraAvoidedColors.size() + 1);
    for (const cv::Vec3f& color : extraAvoidedColors) {
        avoidedColors.push_back({color, 24.0f});
    }
    if (const auto previousColor = mostRecentSameWrapColor(pointCollection, excludedCollectionIds)) {
        avoidedColors.push_back({*previousColor, 32.0f});
    }

    if (!volumeToScene || visibleSceneRect.isEmpty()) {
        return mostDistinctColor(avoidedColors, paletteUsage);
    }

    std::vector<QPointF> visibleNewPoints;
    visibleNewPoints.reserve(points.size());
    for (const cv::Vec3f& point : points) {
        const QPointF scenePoint = volumeToScene(point);
        if (std::isfinite(scenePoint.x()) &&
            std::isfinite(scenePoint.y()) &&
            visibleSceneRect.contains(scenePoint)) {
            visibleNewPoints.push_back(scenePoint);
        }
    }
    if (visibleNewPoints.empty()) {
        return mostDistinctColor(avoidedColors, paletteUsage);
    }

    constexpr qreal radius2 = kNearbyCollectionColorRadiusPx * kNearbyCollectionColorRadiusPx;
    constexpr qreal cellSize = kNearbyCollectionColorRadiusPx;
    struct CellKey {
        int x = 0;
        int y = 0;
        bool operator==(const CellKey& other) const
        {
            return x == other.x && y == other.y;
        }
    };
    struct CellKeyHash {
        std::size_t operator()(const CellKey& key) const
        {
            const std::uint64_t ux = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.x));
            const std::uint64_t uy = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.y));
            return static_cast<std::size_t>((ux << 32) ^ uy);
        }
    };
    const auto cellForPoint = [&](const QPointF& point) {
        return CellKey{
            static_cast<int>(std::floor(point.x() / cellSize)),
            static_cast<int>(std::floor(point.y() / cellSize))
        };
    };

    std::unordered_map<CellKey, std::vector<QPointF>, CellKeyHash> newPointGrid;
    newPointGrid.reserve(visibleNewPoints.size());
    for (const QPointF& newScenePoint : visibleNewPoints) {
        newPointGrid[cellForPoint(newScenePoint)].push_back(newScenePoint);
    }

    const auto isNearNewPoint = [&](const QPointF& scenePoint) {
        const CellKey base = cellForPoint(scenePoint);
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const auto it = newPointGrid.find(CellKey{base.x + dx, base.y + dy});
                if (it == newPointGrid.end()) {
                    continue;
                }
                for (const QPointF& newScenePoint : it->second) {
                    const qreal xDelta = scenePoint.x() - newScenePoint.x();
                    const qreal yDelta = scenePoint.y() - newScenePoint.y();
                    if (xDelta * xDelta + yDelta * yDelta <= radius2) {
                        return true;
                    }
                }
            }
        }
        return false;
    };

    std::unordered_set<uint64_t> avoidedCollectionIds;
    const auto& collections = pointCollection->getAllCollections();
    for (const auto& [collectionId, collection] : collections) {
        if (excludedCollectionIds.count(collectionId) != 0) {
            continue;
        }
        bool isVisible = false;
        bool isNearby = false;
        for (const auto& [pointId, point] : collection.points) {
            (void)pointId;
            const QPointF scenePoint = volumeToScene(point.p);
            if (!std::isfinite(scenePoint.x()) ||
                !std::isfinite(scenePoint.y()) ||
                !visibleSceneRect.contains(scenePoint)) {
                continue;
            }
            isVisible = true;
            isNearby = isNearNewPoint(scenePoint);
            if (isNearby) {
                break;
            }
        }
        if (isVisible && avoidedCollectionIds.insert(collectionId).second) {
            avoidedColors.push_back({collection.color, isNearby ? 8.0f : 1.0f});
        }
    }

    return mostDistinctColor(avoidedColors, paletteUsage);
}

int nearestSkeletonPixel(const cv::Mat& skeleton, int x, int y, int searchRadius)
{
    int bestKey = -1;
    int bestDist2 = std::numeric_limits<int>::max();
    for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
        const int py = y + dy;
        if (py < 0 || py >= skeleton.rows) {
            continue;
        }
        const uint8_t* row = skeleton.ptr<uint8_t>(py);
        for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
            const int px = x + dx;
            if (px < 0 || px >= skeleton.cols || row[px] == 0) {
                continue;
            }
            const int dist2 = dx * dx + dy * dy;
            if (dist2 < bestDist2) {
                bestDist2 = dist2;
                bestKey = py * skeleton.cols + px;
            }
        }
    }
    return bestKey;
}

// Same-wrap collections mark points on a single wrap, not winding annotations.
// They must not carry an absolute winding number (the metadata default), or
// downstream winding consumers misinterpret them.
void markSameWrapCollectionMetadata(VCCollection* pointCollection, uint64_t collectionId)
{
    if (!pointCollection || collectionId == 0) {
        return;
    }
    CollectionMetadata metadata;
    metadata.absolute_winding_number = false;
    pointCollection->setCollectionMetadata(collectionId, metadata);
}

std::vector<cv::Vec3f> sampleScenePath(const std::vector<QPointF>& scenePath,
                                       float spacingPx,
                                       const SameWrapAnnotationTool::SceneToVolumeFn& sceneToVolume)
{
    std::vector<cv::Vec3f> sampled;
    if (scenePath.empty()) {
        return sampled;
    }

    sampled.push_back(sceneToVolume(scenePath.front()));
    float sinceLast = 0.0f;
    QPointF prev = scenePath.front();
    for (size_t i = 1; i < scenePath.size(); ++i) {
        QPointF cur = scenePath[i];
        float segmentLen = static_cast<float>(std::hypot(cur.x() - prev.x(), cur.y() - prev.y()));
        while (sinceLast + segmentLen >= spacingPx && segmentLen > 0.0f) {
            const float t = (spacingPx - sinceLast) / segmentLen;
            const QPointF sample(prev.x() + (cur.x() - prev.x()) * t,
                                 prev.y() + (cur.y() - prev.y()) * t);
            sampled.push_back(sceneToVolume(sample));
            prev = sample;
            segmentLen = static_cast<float>(std::hypot(cur.x() - prev.x(), cur.y() - prev.y()));
            sinceLast = 0.0f;
        }
        sinceLast += segmentLen;
        prev = cur;
    }
    if (sampled.empty() || cv::norm(sampled.back() - sceneToVolume(scenePath.back())) > 0.5f) {
        sampled.push_back(sceneToVolume(scenePath.back()));
    }
    return sampled;
}
}

void SameWrapAnnotationTool::setEnabled(bool enabled)
{
    _state.enabled = enabled;
}

void SameWrapAnnotationTool::setSpacing(double spacingVx)
{
    _state.spacingVx = std::max(1.0f, static_cast<float>(spacingVx));
}

void SameWrapAnnotationTool::setMergeExistingAnnotations(bool enabled)
{
    _state.mergeExistingAnnotations = enabled;
}

void SameWrapAnnotationTool::setPathType(PathType pathType)
{
    if (_state.pathType == pathType) {
        return;
    }
    _state.pathType = pathType;
    _state.hasShortestPathSource = false;
}

void SameWrapAnnotationTool::setImageFilter(ImageFilterType filterType, int kernelSize)
{
    _state.imageFilterType = filterType;
    _state.imageFilterKernelSize = std::max(3, kernelSize | 1);
    _state.hasShortestPathSource = false;
}

void SameWrapAnnotationTool::setImageFilterType(ImageFilterType filterType)
{
    _state.imageFilterType = filterType;
    _state.hasShortestPathSource = false;
}

void SameWrapAnnotationTool::setImageFilterKernelSize(int kernelSize)
{
    _state.imageFilterKernelSize = std::max(3, kernelSize | 1);
    _state.hasShortestPathSource = false;
}

void SameWrapAnnotationTool::noteShiftReleased()
{
    _state.shiftReleasedSincePreview = true;
}

void SameWrapAnnotationTool::clear(const ClearOverlayGroupFn& clearOverlayGroup)
{
    _state.componentScenePath.clear();
    _state.componentVolumePath.clear();
    _state.sampledVolumePoints.clear();
    _state.clickVolumePos = {0.0f, 0.0f, 0.0f};
    _state.pendingMergeCollectionId = 0;
    _state.pendingMergePointId = 0;
    _state.hasShortestPathSource = false;
    _state.shortestPathSourceScenePos = QPointF();
    _state.shortestPathSourceVolumePos = {0.0f, 0.0f, 0.0f};
    _state.hasPreview = false;
    _state.shiftReleasedSincePreview = true;
    clearOverlayGroup(kSameWrapAnnotationOverlayKey);
}

bool SameWrapAnnotationTool::commit(VCCollection* pointCollection,
                                    const VolumeToSceneFn& volumeToScene,
                                    const QRectF& visibleSceneRect,
                                    const ClearOverlayGroupFn& clearOverlayGroup)
{
    if (!_state.enabled || !_state.hasPreview ||
        _state.sampledVolumePoints.empty() || !pointCollection) {
        return false;
    }

    std::vector<uint64_t> committedCollectionIds;
    const std::string collectionName = pointCollection->generateNewCollectionName("same_wrap");
    pointCollection->addPoints(collectionName, _state.sampledVolumePoints);
    const uint64_t collectionId = pointCollection->getCollectionId(collectionName);
    if (collectionId != 0) {
        markSameWrapCollectionMetadata(pointCollection, collectionId);
        if (const auto color = colorDistinctFromVisibleCollections(
                pointCollection,
                _state.sampledVolumePoints,
                std::unordered_set<uint64_t>{collectionId},
                {},
                volumeToScene,
                visibleSceneRect)) {
            pointCollection->setCollectionColor(collectionId, *color);
        }
        committedCollectionIds.push_back(collectionId);
    }
    if (!committedCollectionIds.empty()) {
        _committedCollectionHistory.push_back(std::move(committedCollectionIds));
    }
    clear(clearOverlayGroup);
    return true;
}

bool SameWrapAnnotationTool::undoLastCommit(VCCollection* pointCollection)
{
    if (!pointCollection) {
        return false;
    }

    while (!_committedCollectionHistory.empty()) {
        std::vector<uint64_t> collectionIds = std::move(_committedCollectionHistory.back());
        _committedCollectionHistory.pop_back();

        bool removedAny = false;
        for (uint64_t collectionId : collectionIds) {
            if (pointCollection->getAllCollections().count(collectionId) == 0) {
                continue;
            }
            pointCollection->clearCollection(collectionId);
            removedAny = true;
        }
        if (removedAny) {
            return true;
        }
    }
    return false;
}

bool SameWrapAnnotationTool::manualMergePointClicked(VCCollection* pointCollection,
                                                     uint64_t collectionId,
                                                     uint64_t pointId,
                                                     const VolumeToSceneFn& volumeToScene,
                                                     const QRectF& visibleSceneRect,
                                                     const ConfirmMixedDirectionMergeFn& confirmMixedDirectionMerge)
{
    if (!_state.enabled || !_state.mergeExistingAnnotations || !pointCollection ||
        collectionId == 0 || pointId == 0) {
        return false;
    }

    const auto& collections = pointCollection->getAllCollections();
    const auto collectionIt = collections.find(collectionId);
    if (collectionIt == collections.end() ||
        !isSameWrapCollectionName(collectionIt->second.name) ||
        collectionIt->second.points.count(pointId) == 0) {
        _state.pendingMergeCollectionId = 0;
        _state.pendingMergePointId = 0;
        return false;
    }

    if (_state.pendingMergeCollectionId == 0 ||
        _state.pendingMergePointId == 0 ||
        _state.pendingMergeCollectionId == collectionId) {
        _state.pendingMergeCollectionId = collectionId;
        _state.pendingMergePointId = pointId;
        return true;
    }

    const uint64_t firstCollectionId = _state.pendingMergeCollectionId;
    const uint64_t firstPointId = _state.pendingMergePointId;
    _state.pendingMergeCollectionId = 0;
    _state.pendingMergePointId = 0;

    const auto firstIt = collections.find(firstCollectionId);
    if (firstIt == collections.end() ||
        !isSameWrapCollectionName(firstIt->second.name) ||
        firstIt->second.points.count(firstPointId) == 0) {
        _state.pendingMergeCollectionId = collectionId;
        _state.pendingMergePointId = pointId;
        return true;
    }

    const std::string firstDirectionKey = dominantDirectionKey(orderedCollectionPositions(firstIt->second));
    const std::string secondDirectionKey = dominantDirectionKey(orderedCollectionPositions(collectionIt->second));
    if (firstDirectionKey != secondDirectionKey && confirmMixedDirectionMerge) {
        const MixedDirectionMergeWarning warning{
            firstIt->second.name,
            firstDirectionKey,
            collectionIt->second.name,
            secondDirectionKey
        };
        if (!confirmMixedDirectionMerge(warning)) {
            _state.pendingMergeCollectionId = firstCollectionId;
            _state.pendingMergePointId = firstPointId;
            return true;
        }
    }

    std::vector<cv::Vec3f> mergedPoints =
        bestEndpointMergeOrder(firstIt->second, collectionIt->second);
    mergedPoints = removeClosePoints(mergedPoints, _state.spacingVx);
    if (mergedPoints.size() < 2) {
        return false;
    }

    const std::string mergedCollectionName = firstIt->second.name;
    const std::string directionKey = dominantDirectionKey(mergedPoints);
    const std::vector<cv::Vec3f> sourceColorsToAvoid{
        firstIt->second.color,
        collectionIt->second.color
    };
    pointCollection->clearCollection(firstCollectionId);
    pointCollection->clearCollection(collectionId);
    pointCollection->addPoints(mergedCollectionName, mergedPoints);
    const uint64_t mergedCollectionId = pointCollection->getCollectionId(mergedCollectionName);
    if (mergedCollectionId != 0) {
        markSameWrapCollectionMetadata(pointCollection, mergedCollectionId);
        pointCollection->setCollectionTag(mergedCollectionId, "same_wrap_direction", directionKey);
        if (const auto color = colorDistinctFromVisibleCollections(
                pointCollection,
                mergedPoints,
                std::unordered_set<uint64_t>{mergedCollectionId},
                sourceColorsToAvoid,
                volumeToScene,
                visibleSceneRect)) {
            pointCollection->setCollectionColor(mergedCollectionId, *color);
        }
    }
    return true;
}

bool SameWrapAnnotationTool::generatePreview(const QImage& framebuffer,
                                             const QPointF& scenePos,
                                             bool appendToPreview,
                                             float viewScale,
                                             const VCCollection* pointCollection,
                                             const SceneToVolumeFn& sceneToVolume,
                                             const VolumeToSceneFn& volumeToScene,
                                             const SetOverlayGroupFn& setOverlayGroup,
                                             const ClearOverlayGroupFn& clearOverlayGroup)
{
    if (_state.pathType == PathType::Manual) {
        return false;
    }

    std::vector<QPointF> previousScenePath;
    std::vector<cv::Vec3f> previousVolumePath;
    std::vector<cv::Vec3f> previousSampledPoints;
    if (appendToPreview && _state.hasPreview) {
        previousScenePath = _state.componentScenePath;
        previousVolumePath = _state.componentVolumePath;
        previousSampledPoints = _state.sampledVolumePoints;
    } else {
        if (!(_state.pathType == PathType::ShortestPath && _state.hasShortestPathSource)) {
            clear(clearOverlayGroup);
        }
        appendToPreview = false;
    }

    cv::Mat gray;
    if (!sampleSourceImage(framebuffer, gray)) {
        if (appendToPreview) {
            _state.componentScenePath = std::move(previousScenePath);
            _state.componentVolumePath = std::move(previousVolumePath);
            _state.sampledVolumePoints = std::move(previousSampledPoints);
            _state.hasPreview = !_state.sampledVolumePoints.empty();
        }
        return false;
    }
    if (_state.imageFilterType == ImageFilterType::Median) {
        cv::medianBlur(gray, gray, std::max(3, _state.imageFilterKernelSize | 1));
    } else if (_state.imageFilterType == ImageFilterType::Gaussian) {
        const int kernelSize = std::max(3, _state.imageFilterKernelSize | 1);
        cv::GaussianBlur(gray, gray, cv::Size(kernelSize, kernelSize), 0.0);
    }

    const int w = framebuffer.width();
    const int h = framebuffer.height();
    const int clickX = std::clamp(static_cast<int>(std::lround(scenePos.x())), 0, w - 1);
    const int clickY = std::clamp(static_cast<int>(std::lround(scenePos.y())), 0, h - 1);

    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat skeleton;
    cv::ximgproc::thinning(binary, skeleton, cv::ximgproc::THINNING_GUOHALL);

    if (_state.pathType == PathType::ShortestPath) {
        constexpr int kSearchRadius = 10;
        const int selectedKey = nearestSkeletonPixel(skeleton, clickX, clickY, kSearchRadius);
        if (selectedKey < 0) {
            return false;
        }

        const QPointF snappedScenePos(static_cast<qreal>(selectedKey % w),
                                      static_cast<qreal>(selectedKey / w));
        if (!_state.hasShortestPathSource) {
            clear(clearOverlayGroup);
            _state.hasShortestPathSource = true;
            _state.shortestPathSourceScenePos = snappedScenePos;
            _state.shortestPathSourceVolumePos = sceneToVolume(snappedScenePos);
            _state.clickVolumePos = _state.shortestPathSourceVolumePos;
            _state.hasPreview = true;
            _state.shiftReleasedSincePreview = false;
            updateOverlay(volumeToScene, setOverlayGroup, clearOverlayGroup);
            return true;
        }

        _state.shortestPathSourceScenePos = volumeToScene(_state.shortestPathSourceVolumePos);
        const int sourceKey = nearestSkeletonPixel(
            skeleton,
            static_cast<int>(std::lround(_state.shortestPathSourceScenePos.x())),
            static_cast<int>(std::lround(_state.shortestPathSourceScenePos.y())),
            kSearchRadius);
        const int targetKey = selectedKey;
        if (sourceKey < 0 || sourceKey == targetKey) {
            return false;
        }

        std::vector<float> weights(static_cast<size_t>(w) * static_cast<size_t>(h), 1000000.0f);
        for (int y = 0; y < h; ++y) {
            const uint8_t* row = skeleton.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x) {
                if (row[x] > 0) {
                    weights[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = 1.0f;
                }
            }
        }

        std::vector<uint32_t> path = dijkstra::dijkstra2d<float, uint32_t>(
            weights.data(),
            static_cast<size_t>(w),
            static_cast<size_t>(h),
            static_cast<size_t>(sourceKey),
            static_cast<size_t>(targetKey),
            8);
        if (path.size() < 2) {
            return false;
        }
        std::reverse(path.begin(), path.end());

        std::vector<QPointF> scenePath;
        scenePath.reserve(path.size());
        for (uint32_t key : path) {
            const int x = static_cast<int>(key % static_cast<uint32_t>(w));
            const int y = static_cast<int>(key / static_cast<uint32_t>(w));
            if (x < 0 || x >= w || y < 0 || y >= h || skeleton.at<uint8_t>(y, x) == 0) {
                return false;
            }
            scenePath.emplace_back(static_cast<qreal>(x), static_cast<qreal>(y));
        }

        std::vector<cv::Vec3f> volumePath;
        volumePath.reserve(scenePath.size());
        for (const QPointF& point : scenePath) {
            volumePath.push_back(sceneToVolume(point));
        }

        const float spacingPx = std::max(1.0f, _state.spacingVx * viewScale);
        std::vector<cv::Vec3f> sampled = sampleScenePath(scenePath, spacingPx, sceneToVolume);
        if (sampled.size() < 2) {
            return false;
        }

        _state.componentScenePath = std::move(scenePath);
        _state.componentVolumePath = std::move(volumePath);
        _state.sampledVolumePoints = std::move(sampled);
        _state.clickVolumePos = sceneToVolume(snappedScenePos);
        _state.hasPreview = true;
        _state.hasShortestPathSource = false;
        _state.shiftReleasedSincePreview = false;
        updateOverlay(volumeToScene, setOverlayGroup, clearOverlayGroup);
        return true;
    }

    cv::Mat labels;
    const int componentCount = cv::connectedComponents(skeleton, labels, 8, CV_32S);
    if (componentCount <= 1) {
        return false;
    }

    int selectedLabel = labels.at<int>(clickY, clickX);
    if (selectedLabel == 0) {
        constexpr int kSearchRadius = 10;
        int bestDist2 = std::numeric_limits<int>::max();
        for (int dy = -kSearchRadius; dy <= kSearchRadius; ++dy) {
            const int y = clickY + dy;
            if (y < 0 || y >= h) {
                continue;
            }
            for (int dx = -kSearchRadius; dx <= kSearchRadius; ++dx) {
                const int x = clickX + dx;
                if (x < 0 || x >= w) {
                    continue;
                }
                const int label = labels.at<int>(y, x);
                if (label == 0) {
                    continue;
                }
                const int dist2 = dx * dx + dy * dy;
                if (dist2 < bestDist2) {
                    bestDist2 = dist2;
                    selectedLabel = label;
                }
            }
        }
    }
    if (selectedLabel == 0) {
        return false;
    }

    std::vector<int> pixels;
    pixels.reserve(1024);
    std::unordered_map<int, int> pixelToNode;
    for (int y = 0; y < h; ++y) {
        const int* labelRow = labels.ptr<int>(y);
        for (int x = 0; x < w; ++x) {
            if (labelRow[x] == selectedLabel) {
                const int key = y * w + x;
                pixelToNode.emplace(key, static_cast<int>(pixels.size()));
                pixels.push_back(key);
            }
        }
    }
    if (pixels.size() < 2) {
        return false;
    }

    std::vector<std::vector<int>> adjacency(pixels.size());
    static constexpr std::array<std::pair<int, int>, 8> kNeighbors{{
        {-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}
    }};
    for (size_t i = 0; i < pixels.size(); ++i) {
        const int x = pixels[i] % w;
        const int y = pixels[i] / w;
        for (const auto& [dx, dy] : kNeighbors) {
            const int nx = x + dx;
            const int ny = y + dy;
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) {
                continue;
            }
            auto it = pixelToNode.find(ny * w + nx);
            if (it != pixelToNode.end()) {
                adjacency[i].push_back(it->second);
            }
        }
    }

    auto farthestFrom = [&](int start, std::vector<int>* outParent) {
        std::vector<float> dist(pixels.size(), -1.0f);
        std::vector<int> parent(pixels.size(), -1);
        struct QueueEntry {
            float dist;
            int node;
            bool operator>(const QueueEntry& other) const { return dist > other.dist; }
        };
        std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> queue;
        dist[start] = 0.0f;
        queue.push({0.0f, start});
        while (!queue.empty()) {
            const auto [curDist, node] = queue.top();
            queue.pop();
            if (curDist != dist[node]) {
                continue;
            }
            const int x = pixels[node] % w;
            const int y = pixels[node] / w;
            for (int next : adjacency[node]) {
                const int nx = pixels[next] % w;
                const int ny = pixels[next] / w;
                const float step = (x == nx || y == ny) ? 1.0f : std::sqrt(2.0f);
                const float nd = curDist + step;
                if (dist[next] < 0.0f || nd < dist[next]) {
                    dist[next] = nd;
                    parent[next] = node;
                    queue.push({nd, next});
                }
            }
        }
        int best = start;
        for (int i = 0; i < static_cast<int>(dist.size()); ++i) {
            if (dist[i] > dist[best]) {
                best = i;
            }
        }
        if (outParent) {
            *outParent = std::move(parent);
        }
        return best;
    };

    int seed = 0;
    for (int i = 0; i < static_cast<int>(adjacency.size()); ++i) {
        if (adjacency[i].size() <= 1) {
            seed = i;
            break;
        }
    }
    const int start = farthestFrom(seed, nullptr);
    std::vector<int> parent;
    const int end = farthestFrom(start, &parent);

    std::vector<int> orderedNodes;
    for (int node = end; node >= 0; node = parent[node]) {
        orderedNodes.push_back(node);
        if (node == start) {
            break;
        }
    }
    if (orderedNodes.size() < 2 || orderedNodes.back() != start) {
        return false;
    }
    std::reverse(orderedNodes.begin(), orderedNodes.end());

    std::vector<QPointF> scenePath;
    scenePath.reserve(orderedNodes.size());
    for (int node : orderedNodes) {
        const int x = pixels[node] % w;
        const int y = pixels[node] / w;
        scenePath.emplace_back(static_cast<qreal>(x), static_cast<qreal>(y));
    }

    std::vector<cv::Vec3f> volumePath;
    volumePath.reserve(scenePath.size());
    for (const QPointF& point : scenePath) {
        volumePath.push_back(sceneToVolume(point));
    }

    if (appendToPreview && !previousSampledPoints.empty() && volumePath.size() >= 2) {
        const cv::Vec3f previousLast = previousSampledPoints.back();
        int closestIdx = 0;
        float closestDist = std::numeric_limits<float>::max();
        for (int i = 0; i < static_cast<int>(volumePath.size()); ++i) {
            const float dist = cv::norm(volumePath[i] - previousLast);
            if (dist < closestDist) {
                closestDist = dist;
                closestIdx = i;
            }
        }

        bool walkForward = (closestIdx < static_cast<int>(volumePath.size()) - 1);
        if (closestIdx > 0 && closestIdx < static_cast<int>(volumePath.size()) - 1 &&
            previousSampledPoints.size() >= 2) {
            const cv::Vec3f previousTangent = previousSampledPoints.back() -
                                              previousSampledPoints[previousSampledPoints.size() - 2];
            const float tangentNorm = cv::norm(previousTangent);
            if (tangentNorm > 1e-4f) {
                const cv::Vec3f tangent = previousTangent / tangentNorm;
                const cv::Vec3f forwardStep = volumePath[closestIdx + 1] - volumePath[closestIdx];
                const cv::Vec3f backwardStep = volumePath[closestIdx - 1] - volumePath[closestIdx];
                const float forwardNorm = cv::norm(forwardStep);
                const float backwardNorm = cv::norm(backwardStep);
                const float forwardDot = forwardNorm > 1e-4f ? tangent.dot(forwardStep / forwardNorm)
                                                             : -std::numeric_limits<float>::infinity();
                const float backwardDot = backwardNorm > 1e-4f ? tangent.dot(backwardStep / backwardNorm)
                                                               : -std::numeric_limits<float>::infinity();
                walkForward = forwardDot >= backwardDot;
            }
        } else if (closestIdx == static_cast<int>(volumePath.size()) - 1) {
            walkForward = false;
        }

        std::vector<QPointF> continuedScenePath;
        std::vector<cv::Vec3f> continuedVolumePath;
        if (walkForward) {
            continuedScenePath.assign(scenePath.begin() + closestIdx, scenePath.end());
            continuedVolumePath.assign(volumePath.begin() + closestIdx, volumePath.end());
        } else {
            continuedScenePath.reserve(static_cast<size_t>(closestIdx) + 1);
            continuedVolumePath.reserve(static_cast<size_t>(closestIdx) + 1);
            for (int i = closestIdx; i >= 0; --i) {
                continuedScenePath.push_back(scenePath[i]);
                continuedVolumePath.push_back(volumePath[i]);
            }
        }
        scenePath = std::move(continuedScenePath);
        volumePath = std::move(continuedVolumePath);
    }

    const float spacingPx = std::max(1.0f, _state.spacingVx * viewScale);
    std::vector<cv::Vec3f> sampled = sampleScenePath(scenePath, spacingPx, sceneToVolume);
    if (sampled.size() < 2) {
        return false;
    }

    if (appendToPreview && !previousSampledPoints.empty() && !sampled.empty()) {
        if (cv::norm(sampled.front() - previousSampledPoints.back()) <
            std::max(0.5f, _state.spacingVx * 0.5f)) {
            sampled.erase(sampled.begin());
        }
        if (sampled.empty()) {
            return false;
        }

        if (!previousScenePath.empty() && !scenePath.empty()) {
            previousScenePath.push_back(scenePath.front());
            previousScenePath.insert(previousScenePath.end(), scenePath.begin() + 1, scenePath.end());
        } else {
            previousScenePath.insert(previousScenePath.end(), scenePath.begin(), scenePath.end());
        }
        if (!previousVolumePath.empty() && !volumePath.empty()) {
            previousVolumePath.push_back(volumePath.front());
            previousVolumePath.insert(previousVolumePath.end(), volumePath.begin() + 1, volumePath.end());
        } else {
            previousVolumePath.insert(previousVolumePath.end(), volumePath.begin(), volumePath.end());
        }

        previousSampledPoints.insert(previousSampledPoints.end(), sampled.begin(), sampled.end());
        _state.componentScenePath = std::move(previousScenePath);
        _state.componentVolumePath = std::move(previousVolumePath);
        _state.sampledVolumePoints = std::move(previousSampledPoints);
    } else {
        _state.componentScenePath = std::move(scenePath);
        _state.componentVolumePath = std::move(volumePath);
        _state.sampledVolumePoints = std::move(sampled);
    }
    _state.clickVolumePos = sceneToVolume(QPointF(clickX, clickY));
    _state.hasPreview = true;
    _state.shiftReleasedSincePreview = false;
    updateOverlay(volumeToScene, setOverlayGroup, clearOverlayGroup);
    return true;
}

bool SameWrapAnnotationTool::beginManualPreview(const QPointF& scenePos,
                                                bool appendToPreview,
                                                const SceneToVolumeFn& sceneToVolume,
                                                const VolumeToSceneFn& volumeToScene,
                                                const SetOverlayGroupFn& setOverlayGroup,
                                                const ClearOverlayGroupFn& clearOverlayGroup)
{
    if (!_state.enabled || _state.pathType != PathType::Manual) {
        return false;
    }

    if (!appendToPreview || !_state.hasPreview) {
        clear(clearOverlayGroup);
    }

    if (!_state.componentScenePath.empty()) {
        const QPointF& previous = _state.componentScenePath.back();
        if (std::hypot(scenePos.x() - previous.x(), scenePos.y() - previous.y()) <= 0.5) {
            return true;
        }
    }

    _state.componentScenePath.push_back(scenePos);
    _state.componentVolumePath.push_back(sceneToVolume(scenePos));
    _state.clickVolumePos = _state.componentVolumePath.back();
    _state.hasPreview = true;
    _state.shiftReleasedSincePreview = false;
    updateOverlay(volumeToScene, setOverlayGroup, clearOverlayGroup);
    return true;
}

bool SameWrapAnnotationTool::appendManualPreview(const QPointF& scenePos,
                                                 float viewScale,
                                                 const VCCollection* pointCollection,
                                                 const SceneToVolumeFn& sceneToVolume,
                                                 const VolumeToSceneFn& volumeToScene,
                                                 const SetOverlayGroupFn& setOverlayGroup,
                                                 const ClearOverlayGroupFn& clearOverlayGroup)
{
    if (!_state.enabled || _state.pathType != PathType::Manual || !_state.hasPreview) {
        return false;
    }

    if (!_state.componentScenePath.empty()) {
        const QPointF& previous = _state.componentScenePath.back();
        if (std::hypot(scenePos.x() - previous.x(), scenePos.y() - previous.y()) <= 0.5) {
            return true;
        }
    }

    _state.componentScenePath.push_back(scenePos);
    _state.componentVolumePath.push_back(sceneToVolume(scenePos));
    _state.clickVolumePos = _state.componentVolumePath.back();

    const float spacingPx = std::max(1.0f, _state.spacingVx * viewScale);
    std::vector<cv::Vec3f> sampled = sampleScenePath(_state.componentScenePath, spacingPx, sceneToVolume);
    if (sampled.size() < 2) {
        _state.sampledVolumePoints.clear();
        updateOverlay(volumeToScene, setOverlayGroup, clearOverlayGroup);
        return true;
    }

    _state.sampledVolumePoints = std::move(sampled);
    _state.shiftReleasedSincePreview = false;
    updateOverlay(volumeToScene, setOverlayGroup, clearOverlayGroup);
    return true;
}

void SameWrapAnnotationTool::refreshOverlay(const VolumeToSceneFn& volumeToScene,
                                            const SetOverlayGroupFn& setOverlayGroup,
                                            const ClearOverlayGroupFn& clearOverlayGroup)
{
    if (_state.hasShortestPathSource) {
        _state.shortestPathSourceScenePos = volumeToScene(_state.shortestPathSourceVolumePos);
    }
    updateOverlay(volumeToScene, setOverlayGroup, clearOverlayGroup);
}

bool SameWrapAnnotationTool::sampleSourceImage(const QImage& framebuffer, cv::Mat& gray) const
{
    if (framebuffer.isNull() || framebuffer.width() <= 0 || framebuffer.height() <= 0) {
        return false;
    }

    const QImage image = framebuffer.convertToFormat(QImage::Format_RGB32);
    gray.create(image.height(), image.width(), CV_8U);
    for (int y = 0; y < image.height(); ++y) {
        const auto* src = reinterpret_cast<const QRgb*>(image.constScanLine(y));
        auto* dst = gray.ptr<uint8_t>(y);
        for (int x = 0; x < image.width(); ++x) {
            dst[x] = static_cast<uint8_t>(qGray(src[x]));
        }
    }
    return true;
}

void SameWrapAnnotationTool::updateOverlay(const VolumeToSceneFn& volumeToScene,
                                           const SetOverlayGroupFn& setOverlayGroup,
                                           const ClearOverlayGroupFn& clearOverlayGroup)
{
    if (!_state.hasPreview) {
        clearOverlayGroup(kSameWrapAnnotationOverlayKey);
        return;
    }

    std::vector<QGraphicsItem*> items;
    if (_state.componentVolumePath.size() >= 2) {
        QPainterPath path(volumeToScene(_state.componentVolumePath.front()));
        for (size_t i = 1; i < _state.componentVolumePath.size(); ++i) {
            path.lineTo(volumeToScene(_state.componentVolumePath[i]));
        }
        auto* pathItem = new QGraphicsPathItem(path);
        QPen pen(QColor(255, 0, 0, 230));
        pen.setWidthF(3.0);
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        pathItem->setPen(pen);
        pathItem->setZValue(130.0);
        items.push_back(pathItem);
    }

    if (_state.hasShortestPathSource) {
        const QPointF scenePoint = volumeToScene(_state.shortestPathSourceVolumePos);
        _state.shortestPathSourceScenePos = scenePoint;
        auto* marker = new QGraphicsEllipseItem(scenePoint.x() - 5.0, scenePoint.y() - 5.0, 10.0, 10.0);
        marker->setPen(QPen(QColor(0, 255, 255, 240), 2.0));
        marker->setBrush(QBrush(QColor(0, 255, 255, 110)));
        marker->setZValue(134.0);
        items.push_back(marker);
    }

    for (const auto& point : _state.sampledVolumePoints) {
        const QPointF scenePoint = volumeToScene(point);
        auto* marker = new QGraphicsEllipseItem(scenePoint.x() - 3.0, scenePoint.y() - 3.0, 6.0, 6.0);
        marker->setPen(QPen(QColor(0, 255, 0, 230), 1.5));
        marker->setBrush(QBrush(QColor(0, 255, 0, 210)));
        marker->setZValue(132.0);
        items.push_back(marker);
    }

    const QPointF clickScenePos = volumeToScene(_state.clickVolumePos);
    auto* clickMarker = new QGraphicsEllipseItem(clickScenePos.x() - 6.0,
                                                 clickScenePos.y() - 6.0,
                                                 12.0,
                                                 12.0);
    clickMarker->setPen(QPen(QColor(0, 255, 255, 240), 2.0));
    clickMarker->setBrush(Qt::NoBrush);
    clickMarker->setZValue(133.0);
    items.push_back(clickMarker);

    setOverlayGroup(kSameWrapAnnotationOverlayKey, items);
}
