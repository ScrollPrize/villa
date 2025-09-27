#pragma once

#include <QObject>
#include <memory>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

class QuadSurface;
class PlaneSurface;

class SegmentationEditManager : public QObject
{
    Q_OBJECT

public:
    explicit SegmentationEditManager(QObject* parent = nullptr);

    bool beginSession(QuadSurface* baseSurface, int downsample);
    void endSession();

    [[nodiscard]] bool hasSession() const { return static_cast<bool>(_baseSurface); }
    [[nodiscard]] QuadSurface* baseSurface() const { return _baseSurface; }
    [[nodiscard]] QuadSurface* previewSurface() const { return _previewSurface.get(); }

    void setDownsample(int value);
    void setRadius(float radius);
    void setSigma(float sigma);
    [[nodiscard]] int downsample() const { return _downsample; }
    [[nodiscard]] float radius() const { return _radius; }
    [[nodiscard]] float sigma() const { return _sigma; }
    [[nodiscard]] bool hasPendingChanges() const { return _dirty; }

    void resetPreview();
    void applyPreview();

    struct Handle {
        int row;
        int col;
        cv::Vec3f originalWorld;
        cv::Vec3f currentWorld;
        bool isManual{false};
    };

    [[nodiscard]] const std::vector<Handle>& handles() const { return _handles; }
    void updateHandleWorldPosition(int row, int col, const cv::Vec3f& newWorldPos);
    Handle* findNearestHandle(const cv::Vec3f& world, float tolerance);
    std::optional<std::pair<int,int>> addHandleAtWorld(const cv::Vec3f& worldPos,
                                                       float tolerance = 40.0f,
                                                       PlaneSurface* plane = nullptr,
                                                       float planeTolerance = 0.0f);
    bool removeHandle(int row, int col);
    std::optional<cv::Vec3f> handleWorldPosition(int row, int col) const;
    std::optional<std::pair<int,int>> worldToGridIndex(const cv::Vec3f& worldPos, float* outDistance = nullptr) const;

private:
    void regenerateHandles();
    void applyHandleInfluence(const Handle& handle);
    Handle* findHandle(int row, int col);
    void syncPreviewFromBase();
    void reapplyAllHandles();

    QuadSurface* _baseSurface{nullptr};
    std::unique_ptr<cv::Mat_<cv::Vec3f>> _originalPoints;
    cv::Mat_<cv::Vec3f>* _previewPoints{nullptr};
    std::unique_ptr<QuadSurface> _previewSurface;
    std::vector<Handle> _handles;
    int _downsample{12};
    float _radius{1.0f};          // radius expressed in grid steps (Chebyshev distance)
    float _sigma{1.0f};           // strength multiplier applied to neighbouring grid points
    bool _dirty{false};
};
