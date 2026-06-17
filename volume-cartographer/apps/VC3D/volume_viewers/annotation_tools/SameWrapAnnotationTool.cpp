#include "volume_viewers/annotation_tools/SameWrapAnnotationTool.hpp"

#include "vc/ui/VCCollection.hpp"

#include "dijkstra3d.hpp"

#include <QBrush>
#include <QColor>
#include <QGraphicsEllipseItem>
#include <QGraphicsPathItem>
#include <QPainterPath>
#include <QPen>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

namespace {
constexpr const char* kSameWrapAnnotationOverlayKey = "same_wrap_annotation_preview";
constexpr float kSameWrapMergeContactToleranceVx = 1.0f;

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

struct PathProjection {
    float distance = std::numeric_limits<float>::infinity();
    float arclength = 0.0f;
};

PathProjection projectPointToPath(const cv::Vec3f& point,
                                  const std::vector<cv::Vec3f>& path,
                                  const std::vector<float>& cumulativeLength)
{
    PathProjection best;
    if (path.empty()) {
        return best;
    }
    if (path.size() == 1) {
        best.distance = cv::norm(point - path.front());
        return best;
    }

    for (size_t i = 1; i < path.size(); ++i) {
        const cv::Vec3f a = path[i - 1];
        const cv::Vec3f ab = path[i] - a;
        const float len2 = ab.dot(ab);
        const float t = len2 > 1e-8f ? std::clamp((point - a).dot(ab) / len2, 0.0f, 1.0f) : 0.0f;
        const cv::Vec3f projected = a + ab * t;
        const float distance = cv::norm(point - projected);
        if (distance < best.distance) {
            best.distance = distance;
            best.arclength = cumulativeLength[i - 1] + std::sqrt(len2) * t;
        }
    }
    return best;
}

std::vector<float> cumulativePathLength(const std::vector<cv::Vec3f>& path)
{
    std::vector<float> cumulative(path.size(), 0.0f);
    for (size_t i = 1; i < path.size(); ++i) {
        cumulative[i] = cumulative[i - 1] + cv::norm(path[i] - path[i - 1]);
    }
    return cumulative;
}

void appendIfDistinct(std::vector<cv::Vec3f>& points, const cv::Vec3f& point, float tolerance)
{
    if (!points.empty() && cv::norm(points.back() - point) <= tolerance) {
        return;
    }
    points.push_back(point);
}

struct ProjectedPoint {
    float arclength = 0.0f;
    uint64_t collectionId = 0;
    uint64_t pointId = 0;
    cv::Vec3f point;
};

bool mergeExistingSameWrapAnnotations(const VCCollection* pointCollection,
                                      const std::vector<cv::Vec3f>& volumePath,
                                      float spacingVx,
                                      std::vector<cv::Vec3f>& sampled,
                                      std::vector<uint64_t>& mergeCollectionIds,
                                      std::string& mergeCollectionName)
{
    if (!pointCollection || volumePath.size() < 2) {
        return true;
    }

    std::vector<ProjectedPoint> projectedPoints;
    const std::vector<float> cumulativeLength = cumulativePathLength(volumePath);
    const float mergeTolerance = kSameWrapMergeContactToleranceVx;
    std::unordered_set<uint64_t> seenCollections(mergeCollectionIds.begin(), mergeCollectionIds.end());

    for (const auto& [collectionId, collection] : pointCollection->getAllCollections()) {
        if (!isSameWrapCollectionName(collection.name) || collection.points.empty() ||
            seenCollections.count(collectionId) > 0) {
            continue;
        }

        const std::vector<ColPoint> orderedPoints = orderedCollectionPoints(collection);
        std::vector<ProjectedPoint> collectionProjectedPoints;
        collectionProjectedPoints.reserve(orderedPoints.size());
        bool collectionMatchesPath = false;
        for (const ColPoint& point : orderedPoints) {
            const PathProjection projection = projectPointToPath(point.p, volumePath, cumulativeLength);
            if (projection.distance <= mergeTolerance) {
                collectionMatchesPath = true;
            }
            collectionProjectedPoints.push_back({projection.arclength, collectionId, point.id, point.p});
        }

        if (!collectionMatchesPath) {
            continue;
        }
        if (mergeCollectionName.empty()) {
            mergeCollectionName = collection.name;
        }
        mergeCollectionIds.push_back(collectionId);
        seenCollections.insert(collectionId);
        projectedPoints.insert(projectedPoints.end(),
                               collectionProjectedPoints.begin(),
                               collectionProjectedPoints.end());
    }

    if (projectedPoints.empty()) {
        return true;
    }

    for (const cv::Vec3f& point : sampled) {
        const PathProjection projection = projectPointToPath(point, volumePath, cumulativeLength);
        projectedPoints.push_back({projection.arclength, 0, 0, point});
    }

    std::sort(projectedPoints.begin(), projectedPoints.end(),
              [](const ProjectedPoint& a, const ProjectedPoint& b) {
                  if (a.arclength != b.arclength) {
                      return a.arclength < b.arclength;
                  }
                  if (a.collectionId != b.collectionId) {
                      return a.collectionId < b.collectionId;
                  }
                  return a.pointId < b.pointId;
              });

    sampled.clear();
    for (const ProjectedPoint& projected : projectedPoints) {
        appendIfDistinct(sampled, projected.point, std::max(0.5f, spacingVx * 0.5f));
    }
    return sampled.size() >= 2;
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
    _state.mergeCollectionIds.clear();
    _state.mergeCollectionName.clear();
    _state.clickVolumePos = {0.0f, 0.0f, 0.0f};
    _state.hasShortestPathSource = false;
    _state.shortestPathSourceScenePos = QPointF();
    _state.shortestPathSourceVolumePos = {0.0f, 0.0f, 0.0f};
    _state.hasPreview = false;
    _state.shiftReleasedSincePreview = true;
    clearOverlayGroup(kSameWrapAnnotationOverlayKey);
}

bool SameWrapAnnotationTool::commit(VCCollection* pointCollection,
                                    const ClearOverlayGroupFn& clearOverlayGroup)
{
    if (!_state.enabled || !_state.hasPreview ||
        _state.sampledVolumePoints.empty() || !pointCollection) {
        return false;
    }

    std::string collectionName = pointCollection->generateNewCollectionName("same_wrap");
    if (_state.mergeExistingAnnotations && !_state.mergeCollectionIds.empty() &&
        !_state.mergeCollectionName.empty()) {
        collectionName = _state.mergeCollectionName;
        const std::vector<uint64_t> mergeCollectionIds = _state.mergeCollectionIds;
        for (uint64_t collectionId : mergeCollectionIds) {
            pointCollection->clearCollection(collectionId);
        }
    }
    pointCollection->addPoints(collectionName, _state.sampledVolumePoints);

    // Same-wrap annotations use relative winding numbers; ensure the collection is
    // not flagged as absolute (which would tick the "Absolute Winding Number" checkbox).
    const uint64_t collectionId = pointCollection->getCollectionId(collectionName);
    if (collectionId != 0) {
        const auto& collections = pointCollection->getAllCollections();
        if (collections.count(collectionId)) {
            CollectionMetadata metadata = collections.at(collectionId).metadata;
            metadata.absolute_winding_number = false;
            pointCollection->setCollectionMetadata(collectionId, metadata);
        }
    }

    clear(clearOverlayGroup);
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
    std::vector<QPointF> previousScenePath;
    std::vector<cv::Vec3f> previousVolumePath;
    std::vector<cv::Vec3f> previousSampledPoints;
    std::vector<uint64_t> previousMergeCollectionIds;
    std::string previousMergeCollectionName;
    if (appendToPreview && _state.hasPreview) {
        previousScenePath = _state.componentScenePath;
        previousVolumePath = _state.componentVolumePath;
        previousSampledPoints = _state.sampledVolumePoints;
        previousMergeCollectionIds = _state.mergeCollectionIds;
        previousMergeCollectionName = _state.mergeCollectionName;
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
            _state.mergeCollectionIds = std::move(previousMergeCollectionIds);
            _state.mergeCollectionName = std::move(previousMergeCollectionName);
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

        std::vector<uint64_t> mergeCollectionIds;
        std::string mergeCollectionName;
        if (_state.mergeExistingAnnotations &&
            !mergeExistingSameWrapAnnotations(pointCollection,
                                              volumePath,
                                              _state.spacingVx,
                                              sampled,
                                              mergeCollectionIds,
                                              mergeCollectionName)) {
            return false;
        }

        _state.componentScenePath = std::move(scenePath);
        _state.componentVolumePath = std::move(volumePath);
        _state.sampledVolumePoints = std::move(sampled);
        _state.mergeCollectionIds = std::move(mergeCollectionIds);
        _state.mergeCollectionName = std::move(mergeCollectionName);
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

    std::vector<uint64_t> mergeCollectionIds;
    std::string mergeCollectionName;
    if (appendToPreview) {
        mergeCollectionIds = std::move(previousMergeCollectionIds);
        mergeCollectionName = std::move(previousMergeCollectionName);
    }

    if (_state.mergeExistingAnnotations &&
        !mergeExistingSameWrapAnnotations(pointCollection,
                                          volumePath,
                                          _state.spacingVx,
                                          sampled,
                                          mergeCollectionIds,
                                          mergeCollectionName)) {
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
        _state.mergeCollectionIds = std::move(mergeCollectionIds);
        _state.mergeCollectionName = std::move(mergeCollectionName);
    } else {
        _state.componentScenePath = std::move(scenePath);
        _state.componentVolumePath = std::move(volumePath);
        _state.sampledVolumePoints = std::move(sampled);
        _state.mergeCollectionIds = std::move(mergeCollectionIds);
        _state.mergeCollectionName = std::move(mergeCollectionName);
    }
    _state.clickVolumePos = sceneToVolume(QPointF(clickX, clickY));
    _state.hasPreview = true;
    _state.shiftReleasedSincePreview = false;
    updateOverlay(volumeToScene, setOverlayGroup, clearOverlayGroup);
    return true;
}

void SameWrapAnnotationTool::refreshOverlay(const VolumeToSceneFn& volumeToScene,
                                            const SetOverlayGroupFn& setOverlayGroup,
                                            const ClearOverlayGroupFn& clearOverlayGroup)
{
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
