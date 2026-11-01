#pragma once

#include "SegmentationEditManager.hpp"

#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

class PlaneSurface;

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

    // Grid side an edge extension grows from. Up and Down move along rows
    // (row - 1, row + 1), Left and Right along columns, as in tracer growth.
    enum class ExtendSide
    {
        Up = 0,
        Down = 1,
        Left = 2,
        Right = 3,
    };

    // Invalid rows and columns to add around the grid before an extension.
    struct GridPadding
    {
        int top{0};
        int bottom{0};
        int left{0};
        int right{0};

        [[nodiscard]] bool any() const { return top > 0 || bottom > 0 || left > 0 || right > 0; }
    };

    // Inclusive range of rows (Left, Right) or columns (Up, Down) to extend.
    struct EdgeSpan
    {
        int first{0};
        int last{-1};
    };

    struct Config
    {
        int maxPreviewSpan{256};
        int boundaryBand{2};
        double regularization{1e-4};
        int sampleCap{512};
        // When the extension follows a guide instead of a straight line, a
        // click's correction fades out over about this many vertices around
        // it (0: it spreads over the whole strip, as for the straight guess).
        // 4 was best of 0, 2, 4 and 8 on the w035 edge windows.
        double guidedClickReach{4.0};
        // Re-walk each new extension along the CT sheet (SegmentationModule).
        bool followSheet{true};
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

    struct EdgePick
    {
        ExtendSide side{ExtendSide::Right};
        EdgeSpan span;
    };
    // The edge stretch a drag from `from` to `to` (grid cells, x = column,
    // y = row) runs along: a mostly horizontal drag picks the top or bottom
    // edge of the columns it covers, whichever the drag runs closer to, and a
    // mostly vertical one the left or right edge of its rows.
    static std::optional<EdgePick> edgeFromDrag(const cv::Mat_<cv::Vec3f>& points,
                                                cv::Point from,
                                                cv::Point to);
    static GridPadding paddingForExtension(const cv::Mat_<cv::Vec3f>& points,
                                           ExtendSide side,
                                           int cells,
                                           std::optional<EdgeSpan> span = std::nullopt);
    static cv::Mat_<cv::Vec3f> padPoints(const cv::Mat_<cv::Vec3f>& points, const GridPadding& padding);

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
    [[nodiscard]] const std::vector<SegmentationEditManager::GridKey>& borderSampleVertices() const { return _borderSampleVertices; }
    [[nodiscard]] const std::vector<SegmentationEditManager::GridKey>& changedVertices() const { return _changedVertices; }
    [[nodiscard]] const std::vector<Constraint3d>& userPlaneConstraints() const { return _userPlaneConstraints; }
    [[nodiscard]] bool initialFillCommitted() const { return _initialFillCommitted; }
    [[nodiscard]] uint64_t revision() const { return _revision; }

    bool updateHover(int row, int col);
    bool commitHover(std::string* status = nullptr);
    bool addOrReplacePlaneConstraint(int row, int col, const cv::Vec3f& world, std::string* status = nullptr);
    bool removePlaneConstraintNear(const cv::Vec3f& world, double radius, std::string* status = nullptr);
    // Fill vertex a Shift+click at `clickPlane` (plane coordinates) in a slice
    // view constrains: of the vertices within `radius` of the click in the
    // plane, the one nearest in 3D. Ranking by in-plane distance alone picks
    // vertices far in front of or behind the plane whenever several rows
    // project onto the same spot, as they do along an extended edge.
    [[nodiscard]] std::optional<SegmentationEditManager::GridKey>
    pickPlaneConstraintVertex(PlaneSurface& plane, const cv::Vec2f& clickPlane, double radius) const;
    bool removeLastPlaneConstraint(std::string* status = nullptr);
    bool recompute(std::string* status = nullptr);

    // Extends every row (Left, Right) or column (Up, Down) of the span by
    // `cells` new vertices past its last valid vertex on `side`. The preview
    // continues each line along its own direction at the edge; Shift+click
    // plane constraints then bend the new strip onto the sheet. The grid must
    // already have room (see paddingForExtension).
    bool beginExtension(ExtendSide side,
                        int cells,
                        std::optional<EdgeSpan> span = std::nullopt,
                        std::string* status = nullptr);
    [[nodiscard]] bool extensionActive() const { return _extension.has_value(); }
    [[nodiscard]] std::optional<ExtendSide> extensionSide() const;
    // Following the sheet. Each line of the pending extension is walked
    // again from its edge vertex in steps of its own length, each step turned
    // into the plane across the CT sheet normal at its midpoint, so the strip
    // bends with the papyrus instead of running straight. sheetWalk() copies
    // what the walk needs, so the volume reads can run off the GUI thread;
    // applySheetWalk() installs the result only if the same extension is
    // still pending.
    using SheetNormalFn = std::function<std::optional<cv::Vec3f>(const cv::Vec3f& world)>;
    struct SheetWalk
    {
        std::uint64_t extensionId{0};
        int cells{0};
        std::vector<cv::Vec3f> origins;
        std::vector<cv::Vec3f> steps;
    };
    [[nodiscard]] std::optional<SheetWalk> sheetWalk() const;
    // Walks line `line` of `walk`, writing its `cells` points into `guess`
    // (sized origins.size() * cells) at line * cells. Lines are independent.
    static void walkSheetLine(const SheetWalk& walk,
                              std::size_t line,
                              const SheetNormalFn& normal,
                              std::vector<cv::Vec3f>& guess);
    bool applySheetWalk(std::uint64_t extensionId, std::vector<cv::Vec3f> guess, std::string* status = nullptr);
    [[nodiscard]] bool extensionFollowsSheet() const { return _extension && _extension->guided; }

    // Replace the straight continuation of the pending extension with another
    // first guess, one point per fill vertex (in fillVertices() order); clicks
    // still bend the strip from there.
    bool setExtensionGuess(std::vector<cv::Vec3f> guess, std::string* status = nullptr);
    [[nodiscard]] std::optional<EdgeSpan> extensionSpan() const
    {
        return _extension ? _extension->span : std::nullopt;
    }

    void setConfig(Config config);
    [[nodiscard]] Config config() const { return _config; }


private:
    using GridKey = SegmentationEditManager::GridKey;

    static Config sanitize(Config config);
    [[nodiscard]] bool inBounds(int row, int col) const;
    [[nodiscard]] bool isInvalid(int row, int col) const;
    [[nodiscard]] bool isValid(int row, int col) const;
    [[nodiscard]] int flatIndex(int row, int col) const;
    [[nodiscard]] std::optional<GridPolyline> discoverAxisLine(int row, int col, bool horizontal) const;
    [[nodiscard]] std::vector<GridKey> computeFillVerticesForLines(const std::vector<GridPolyline>& lines,
                                                                    bool includeUserConstraints) const;
    void rebuildGridCache();
    void clearHoverFillCache();
    bool hoverFillCacheMatches(const std::vector<GridPolyline>& lines) const;
    void storeHoverFillCache(const std::vector<GridPolyline>& lines, const std::vector<GridKey>& fill);
    uint32_t nextMarkerEpoch() const;
    void extractFillAndBorder();
    std::vector<Constraint3d> buildFitSamples() const;
    std::vector<Constraint3d> downsampleSamples(std::vector<Constraint3d> samples) const;
    bool recomputeExtension(std::string* status);
    void touchRevision() { ++_revision; }

    struct Extension
    {
        ExtendSide side{ExtendSide::Right};
        int cells{0};
        std::optional<EdgeSpan> span;
        bool guided{false};
        std::uint64_t id{0};
        // Per extended line, in fill order: its edge vertex and the step of
        // its straight continuation.
        std::vector<cv::Vec3f> origins;
        std::vector<cv::Vec3f> steps;
        // Continuation of each line past its edge, parallel to _fillVertices.
        std::vector<cv::Vec3f> guess;
        // Valid vertices at and just inside the edge; they keep their place.
        std::vector<GridKey> anchors;
    };
    std::optional<Extension> _extension;
    std::uint64_t _nextExtensionId{1};

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

    cv::Rect _validBounds;
    bool _haveValidBounds{false};
    std::vector<uint8_t> _invalidMask;

    mutable std::vector<uint32_t> _fillSeenMarks;
    mutable std::vector<uint32_t> _barrierMarks;
    mutable std::vector<uint32_t> _sideMarks;
    mutable std::vector<uint32_t> _otherSideMarks;
    mutable uint32_t _markerEpoch{0};

    std::vector<GridPolyline> _cachedHoverFillLines;
    std::vector<GridKey> _cachedHoverFillVertices;
    LinePreviewMode _cachedHoverFillMode{LinePreviewMode::Cross};
    bool _hasCachedHoverFill{false};
};
