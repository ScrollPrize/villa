#pragma once

#include <QImage>
#include <QPointF>

#include <functional>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

class QGraphicsItem;
class VCCollection;

class SameWrapAnnotationTool {
public:
    enum class PathType {
        ConnectedComponents = 0,
        ShortestPath = 1
    };

    enum class ImageFilterType {
        None = 0,
        Median = 1,
        Gaussian = 2
    };

    using SceneToVolumeFn = std::function<cv::Vec3f(const QPointF&)>;
    using VolumeToSceneFn = std::function<QPointF(const cv::Vec3f&)>;
    using SetOverlayGroupFn = std::function<void(const std::string&, const std::vector<QGraphicsItem*>&)>;
    using ClearOverlayGroupFn = std::function<void(const std::string&)>;

    bool enabled() const { return _state.enabled; }
    bool hasPreview() const { return _state.hasPreview; }
    bool shiftReleasedSincePreview() const { return _state.shiftReleasedSincePreview; }

    void setEnabled(bool enabled);
    void setSpacing(double spacingVx);
    void setMergeExistingAnnotations(bool enabled);
    void setPathType(PathType pathType);
    void setImageFilter(ImageFilterType filterType, int kernelSize);
    void setImageFilterType(ImageFilterType filterType);
    void setImageFilterKernelSize(int kernelSize);
    void noteShiftReleased();
    void clear(const ClearOverlayGroupFn& clearOverlayGroup);
    bool commit(VCCollection* pointCollection, const ClearOverlayGroupFn& clearOverlayGroup);
    void refreshOverlay(const VolumeToSceneFn& volumeToScene,
                        const SetOverlayGroupFn& setOverlayGroup,
                        const ClearOverlayGroupFn& clearOverlayGroup);

    bool generatePreview(const QImage& framebuffer,
                         const QPointF& scenePos,
                         bool appendToPreview,
                         float viewScale,
                         const VCCollection* pointCollection,
                         const SceneToVolumeFn& sceneToVolume,
                         const VolumeToSceneFn& volumeToScene,
                         const SetOverlayGroupFn& setOverlayGroup,
                         const ClearOverlayGroupFn& clearOverlayGroup);

private:
    struct State {
        bool enabled = false;
        bool mergeExistingAnnotations = false;
        PathType pathType = PathType::ConnectedComponents;
        ImageFilterType imageFilterType = ImageFilterType::None;
        int imageFilterKernelSize = 3;
        float spacingVx = 20.0f;
        bool shiftReleasedSincePreview = true;
        bool hasShortestPathSource = false;
        QPointF shortestPathSourceScenePos;
        cv::Vec3f shortestPathSourceVolumePos{0.0f, 0.0f, 0.0f};
        std::vector<QPointF> componentScenePath;
        std::vector<cv::Vec3f> componentVolumePath;
        std::vector<cv::Vec3f> sampledVolumePoints;
        std::vector<uint64_t> mergeCollectionIds;
        std::string mergeCollectionName;
        cv::Vec3f clickVolumePos{0.0f, 0.0f, 0.0f};
        bool hasPreview = false;
    };

    bool sampleSourceImage(const QImage& framebuffer, cv::Mat& gray) const;
    void updateOverlay(const VolumeToSceneFn& volumeToScene,
                       const SetOverlayGroupFn& setOverlayGroup,
                       const ClearOverlayGroupFn& clearOverlayGroup);

    State _state;
};
