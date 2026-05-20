#pragma once

#include "SegmentationEditManager.hpp"

#include <opencv2/core.hpp>

#include <optional>
#include <string>
#include <vector>

class ManualAddTool
{
public:
    enum class LinePreviewMode
    {
        VerticalOnly = 0,
        HorizontalOnly = 1,
        Cross = 2,
        CrossFill = 3,
    };

    enum class InterpolationMode
    {
        ThinPlateSpline = 0,
        TracerRestrictedToFill = 1,
    };

    struct Config
    {
        int maxPreviewSpan{256};
        int boundaryBand{2};
        double regularization{1e-4};
        int sampleCap{512};
        int previewThrottleMs{50};
        float tintOpacity{0.45f};
        double planeConstraintRadius{30.0};
        double planeConstraintReplacementRadius{16.0};
        LinePreviewMode linePreviewMode{LinePreviewMode::Cross};
        InterpolationMode interpolationMode{InterpolationMode::ThinPlateSpline};
        bool includeTouchedValidBorder{true};
        bool allowBoundarySmoothing{false};
    };

    struct GridPolyline
    {
        std::vector<cv::Point2i> vertices; // x=col, y=row
        bool committed{false};
        bool floodFillComponent{false};
    };

    struct Constraint3d
    {
        int row{-1};
        int col{-1};
        cv::Vec3f world{0.0f, 0.0f, 0.0f};
        enum class Source { Boundary, CommittedLine, PlaneUser };
        Source source{Source::Boundary};
    };

    bool begin(const cv::Mat_<cv::Vec3f>& points, Config config);
    void clear();
    bool clearPending(Config config);

    [[nodiscard]] bool active() const { return !_entrySnapshotPoints.empty(); }
    [[nodiscard]] const cv::Mat_<cv::Vec3f>& entrySnapshotPoints() const { return _entrySnapshotPoints; }
    [[nodiscard]] const cv::Mat_<cv::Vec3f>& previewPoints() const { return _previewPoints; }
    [[nodiscard]] const std::vector<GridPolyline>& hoverPolylines() const { return _hoverPolylines; }
    [[nodiscard]] std::optional<SegmentationEditManager::GridKey> hoverVertex() const { return _hoverVertex; }
    [[nodiscard]] const std::vector<SegmentationEditManager::GridKey>& hoverFillVertices() const { return _hoverFillVertices; }
    [[nodiscard]] const std::vector<GridPolyline>& committedPolylines() const { return _committedPolylines; }
    [[nodiscard]] const std::vector<SegmentationEditManager::GridKey>& fillVertices() const { return _fillVertices; }
    [[nodiscard]] const std::vector<SegmentationEditManager::GridKey>& changedVertices() const { return _changedVertices; }
    [[nodiscard]] const std::vector<Constraint3d>& userPlaneConstraints() const { return _userPlaneConstraints; }
    [[nodiscard]] bool initialFillCommitted() const { return _initialFillCommitted; }
    [[nodiscard]] uint64_t revision() const { return _revision; }

    bool updateHover(int row, int col);
    bool commitHover(std::string* status = nullptr);
    bool addOrReplacePlaneConstraint(int row, int col, const cv::Vec3f& world, std::string* status = nullptr);
    bool removePlaneConstraintNear(const cv::Vec3f& world, double radius, std::string* status = nullptr);
    bool removeLastPlaneConstraint(std::string* status = nullptr);
    bool recompute(std::string* status = nullptr);

    void setConfig(Config config);
    [[nodiscard]] Config config() const { return _config; }

    static bool isInvalidPoint(const cv::Vec3f& value);

private:
    using GridKey = SegmentationEditManager::GridKey;

    static Config sanitize(Config config);
    [[nodiscard]] bool inBounds(int row, int col) const;
    [[nodiscard]] bool isInvalid(int row, int col) const;
    [[nodiscard]] bool isValid(int row, int col) const;
    [[nodiscard]] std::optional<GridPolyline> discoverAxisLine(int row, int col, bool horizontal) const;
    [[nodiscard]] std::vector<GridKey> computeFillVerticesForLines(const std::vector<GridPolyline>& lines,
                                                                    bool includeUserConstraints) const;
    void extractFillAndBorder();
    std::vector<Constraint3d> buildFitSamples() const;
    std::vector<Constraint3d> downsampleSamples(std::vector<Constraint3d> samples) const;
    void touchRevision() { ++_revision; }

    Config _config;
    cv::Mat_<cv::Vec3f> _entrySnapshotPoints;
    cv::Mat_<cv::Vec3f> _previewPoints;
    std::vector<GridPolyline> _hoverPolylines;
    std::optional<GridKey> _hoverVertex;
    std::vector<GridKey> _hoverFillVertices;
    std::vector<GridPolyline> _committedPolylines;
    std::vector<GridKey> _fillVertices;
    std::vector<GridKey> _borderSampleVertices;
    std::vector<GridKey> _changedVertices;
    std::vector<Constraint3d> _userPlaneConstraints;
    bool _initialFillCommitted{false};
    uint64_t _revision{0};
};
