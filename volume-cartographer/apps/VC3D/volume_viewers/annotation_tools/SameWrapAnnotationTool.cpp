#include "volume_viewers/annotation_tools/SameWrapAnnotationTool.hpp"

#include "vc/ui/VCCollection.hpp"

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
#include <unordered_map>

#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

namespace {
constexpr const char* kSameWrapAnnotationOverlayKey = "same_wrap_annotation_preview";
}

void SameWrapAnnotationTool::setEnabled(bool enabled)
{
    _state.enabled = enabled;
}

void SameWrapAnnotationTool::setSpacing(double spacingVx)
{
    _state.spacingVx = std::max(1.0f, static_cast<float>(spacingVx));
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

    const std::string collectionName = pointCollection->generateNewCollectionName("same_wrap");
    pointCollection->addPoints(collectionName, _state.sampledVolumePoints);
    clear(clearOverlayGroup);
    return true;
}

bool SameWrapAnnotationTool::generatePreview(const QImage& framebuffer,
                                             const QPointF& scenePos,
                                             bool appendToPreview,
                                             float viewScale,
                                             const SceneToVolumeFn& sceneToVolume,
                                             const VolumeToSceneFn& volumeToScene,
                                             const SetOverlayGroupFn& setOverlayGroup,
                                             const ClearOverlayGroupFn& clearOverlayGroup)
{
    std::vector<QPointF> previousScenePath;
    std::vector<cv::Vec3f> previousVolumePath;
    std::vector<cv::Vec3f> previousSampledPoints;
    if (appendToPreview && _state.hasPreview) {
        previousScenePath = _state.componentScenePath;
        previousVolumePath = _state.componentVolumePath;
        previousSampledPoints = _state.sampledVolumePoints;
    } else {
        clear(clearOverlayGroup);
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

    const int w = framebuffer.width();
    const int h = framebuffer.height();
    const int clickX = std::clamp(static_cast<int>(std::lround(scenePos.x())), 0, w - 1);
    const int clickY = std::clamp(static_cast<int>(std::lround(scenePos.y())), 0, h - 1);

    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat skeleton;
    cv::ximgproc::thinning(binary, skeleton, cv::ximgproc::THINNING_GUOHALL);

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
    std::vector<cv::Vec3f> sampled;
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
    _state.clickScenePos = QPointF(clickX, clickY);
    _state.hasPreview = true;
    _state.shiftReleasedSincePreview = false;
    updateOverlay(volumeToScene, setOverlayGroup, clearOverlayGroup);
    return true;
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

    for (const auto& point : _state.sampledVolumePoints) {
        const QPointF scenePoint = volumeToScene(point);
        auto* marker = new QGraphicsEllipseItem(scenePoint.x() - 3.0, scenePoint.y() - 3.0, 6.0, 6.0);
        marker->setPen(QPen(QColor(0, 255, 0, 230), 1.5));
        marker->setBrush(QBrush(QColor(0, 255, 0, 210)));
        marker->setZValue(132.0);
        items.push_back(marker);
    }

    auto* clickMarker = new QGraphicsEllipseItem(_state.clickScenePos.x() - 6.0,
                                                 _state.clickScenePos.y() - 6.0,
                                                 12.0,
                                                 12.0);
    clickMarker->setPen(QPen(QColor(0, 255, 255, 240), 2.0));
    clickMarker->setBrush(Qt::NoBrush);
    clickMarker->setZValue(133.0);
    items.push_back(clickMarker);

    setOverlayGroup(kSameWrapAnnotationOverlayKey, items);
}
