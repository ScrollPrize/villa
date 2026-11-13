#include <QtTest/QtTest>

#include "segmentation/SegmentationUndoHistory.hpp"
#include "segmentation/SurfaceGridEdit.hpp"
#include "segmentation/tools/AutoExtend.hpp"
#include "segmentation/tools/ManualAddTool.hpp"

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <filesystem>
#include <optional>
#include <limits>
#include <stdexcept>

namespace {

constexpr float kSpacing = 2.0f;
const cv::Vec3f kInvalid(-1.0f, -1.0f, -1.0f);

// A flat sheet at z = 100 with vertex (row, col) at (col * 2, row * 2),
// valid only inside `valid`.
cv::Mat_<cv::Vec3f> flatSheet(int rows, int cols, const cv::Rect& valid)
{
    cv::Mat_<cv::Vec3f> points(rows, cols, kInvalid);
    for (int row = valid.y; row < valid.y + valid.height; ++row) {
        for (int col = valid.x; col < valid.x + valid.width; ++col) {
            points(row, col) = cv::Vec3f(col * kSpacing, row * kSpacing, 100.0f);
        }
    }
    return points;
}

float distance(const cv::Vec3f& a, const cv::Vec3f& b)
{
    return static_cast<float>(cv::norm(a - b));
}

ManualAddTool::Config splineConfig()
{
    ManualAddTool::Config config;
    config.interpolationMode = ManualAddTool::InterpolationMode::ThinPlateSpline;
    return config;
}

} // namespace

class TestManualAddExtend : public QObject
{
    Q_OBJECT

private slots:
    void paddingCoversOnlyTheMissingRoom();
    void paddingRespectsSpan();
    void padPointsAddsInvalidBorder();
    void extensionContinuesEachLineStraight();
    void extensionFollowsAPlaneClick();
    void planeClickPicksTheRowOnThePlane();
    void extensionRefusesWithoutRoom();
    void clearPendingDropsTheExtension();
    void resizeSurfaceGridKeepsChannelsAligned();
    void resizeSurfaceGridRejectsLossyChannels();
    void maskCellsBecomeValid();
    void maskedFillSurvivesReload();
    void dragPicksTheNearestEdgeStretch();
    void extensionKeepsItsSpan();
    void undoHistoryFollowsGridPadding();
    void sheetWalkFollowsACurvedSheet();
    void sheetFrontDoesNotBridgeGridGaps();
    void streamlineConverges_data();
    void streamlineConverges();
    void streamlineReusesEndpointNormals();
    void streamlineDropsEndpointNormalAfterFallback();
    void substepsPreserveSpacingAndHandleMissingNormals();
    void projectionSubstepsReduceDrift();
    void guidedClicksStayLocal();
    void autoExtendGrowsAlongTheSheet();
    void autoExtendPreservesLazyChannelsWithoutResizing();
    void autoExtendReportsNormalFailures();
};

void TestManualAddExtend::paddingCoversOnlyTheMissingRoom()
{
    // Columns 2 to 7 are valid in a 10 wide grid: 2 columns of room on each side.
    const auto points = flatSheet(6, 10, cv::Rect(2, 0, 6, 6));
    auto padding = ManualAddTool::paddingForExtension(points, ManualAddTool::ExtendSide::Right, 5);
    QCOMPARE(padding.right, 3);
    QCOMPARE(padding.left + padding.top + padding.bottom, 0);

    padding = ManualAddTool::paddingForExtension(points, ManualAddTool::ExtendSide::Left, 2);
    QVERIFY(!padding.any());

    padding = ManualAddTool::paddingForExtension(points, ManualAddTool::ExtendSide::Up, 4);
    QCOMPARE(padding.top, 4);
}

void TestManualAddExtend::paddingRespectsSpan()
{
    // Row 0 reaches column 9; rows 1 to 5 stop at column 5.
    auto points = flatSheet(6, 10, cv::Rect(0, 0, 6, 6));
    for (int col = 6; col < 10; ++col) {
        points(0, col) = cv::Vec3f(col * kSpacing, 0.0f, 100.0f);
    }
    ManualAddTool::EdgeSpan span{1, 5};
    auto padding = ManualAddTool::paddingForExtension(points, ManualAddTool::ExtendSide::Right, 4, span);
    QVERIFY(!padding.any());
    padding = ManualAddTool::paddingForExtension(points, ManualAddTool::ExtendSide::Right, 4);
    QCOMPARE(padding.right, 4);
}

void TestManualAddExtend::padPointsAddsInvalidBorder()
{
    const auto points = flatSheet(3, 4, cv::Rect(0, 0, 4, 3));
    ManualAddTool::GridPadding padding;
    padding.top = 1;
    padding.left = 2;
    const auto padded = ManualAddTool::padPoints(points, padding);
    QCOMPARE(padded.rows, 4);
    QCOMPARE(padded.cols, 6);
    QVERIFY(!vc::geometry::validSurfacePoint(padded(0, 0)));
    QVERIFY(!vc::geometry::validSurfacePoint(padded(1, 1)));
    QCOMPARE(padded(1, 2), points(0, 0));
}

void TestManualAddExtend::extensionContinuesEachLineStraight()
{
    const auto points = flatSheet(8, 20, cv::Rect(0, 0, 10, 8));
    ManualAddTool tool;
    QVERIFY(tool.begin(points, splineConfig()));
    std::string status;
    QVERIFY2(tool.beginExtension(ManualAddTool::ExtendSide::Right, 6, std::nullopt, &status), status.c_str());
    QVERIFY(tool.extensionActive());
    QCOMPARE(static_cast<int>(tool.fillVertices().size()), 8 * 6);
    QCOMPARE(static_cast<int>(tool.changedVertices().size()), 8 * 6);

    const auto& preview = tool.previewPoints();
    for (int row = 0; row < 8; ++row) {
        for (int col = 10; col < 16; ++col) {
            const cv::Vec3f expected(col * kSpacing, row * kSpacing, 100.0f);
            QVERIFY2(distance(preview(row, col), expected) < 1e-3f,
                     qPrintable(QStringLiteral("row %1 col %2").arg(row).arg(col)));
        }
        QVERIFY(!vc::geometry::validSurfacePoint(preview(row, 16)));
        QCOMPARE(preview(row, 9), points(row, 9));
    }

    // The upward side continues columns the same way.
    ManualAddTool upTool;
    const auto lower = flatSheet(12, 6, cv::Rect(0, 5, 6, 7));
    QVERIFY(upTool.begin(lower, splineConfig()));
    QVERIFY(upTool.beginExtension(ManualAddTool::ExtendSide::Up, 3, std::nullopt, &status));
    QVERIFY(distance(upTool.previewPoints()(2, 4), cv::Vec3f(4 * kSpacing, 2 * kSpacing, 100.0f)) < 1e-3f);
    QVERIFY(!vc::geometry::validSurfacePoint(upTool.previewPoints()(1, 4)));
}

void TestManualAddExtend::extensionFollowsAPlaneClick()
{
    const auto points = flatSheet(9, 24, cv::Rect(0, 0, 12, 9));
    ManualAddTool tool;
    QVERIFY(tool.begin(points, splineConfig()));
    std::string status;
    QVERIFY(tool.beginExtension(ManualAddTool::ExtendSide::Right, 8, std::nullopt, &status));

    // The sheet actually curls up by 10 voxels at the far end of the middle row.
    const cv::Vec3f clicked(19 * kSpacing, 4 * kSpacing, 110.0f);
    QVERIFY2(tool.addOrReplacePlaneConstraint(4, 19, clicked, &status), status.c_str());
    const auto& preview = tool.previewPoints();
    QVERIFY(distance(preview(4, 19), clicked) < 0.5f);
    // The edge does not move, and the strip bends smoothly between them.
    QCOMPARE(preview(4, 11), points(4, 11));
    const float near = preview(4, 13)[2] - 100.0f;
    const float mid = preview(4, 16)[2] - 100.0f;
    QVERIFY2(near > 0.0f && near < mid && mid < 10.0f,
             qPrintable(QStringLiteral("near %1 mid %2").arg(near).arg(mid)));

    QVERIFY(tool.removeLastPlaneConstraint(&status));
    QVERIFY(std::abs(tool.previewPoints()(4, 19)[2] - 100.0f) < 1e-3f);
}

void TestManualAddExtend::planeClickPicksTheRowOnThePlane()
{
    const auto points = flatSheet(9, 24, cv::Rect(0, 0, 12, 9));
    ManualAddTool tool;
    QVERIFY(tool.begin(points, splineConfig()));
    std::string status;
    QVERIFY(tool.beginExtension(ManualAddTool::ExtendSide::Right, 8, std::nullopt, &status));

    // A slice through row 6 that every row of the strip projects onto at the
    // same spot: only the distance off the plane tells the rows apart.
    PlaneSurface plane(cv::Vec3f(0.0f, 6 * kSpacing, 0.0f), cv::Vec3f(0.0f, 1.0f, 0.0f));
    const cv::Vec3f click(17 * kSpacing, 6 * kSpacing, 104.0f);
    const cv::Vec3f inPlane = plane.project(click, 1.0f, 1.0f);
    const auto picked = tool.pickPlaneConstraintVertex(plane, cv::Vec2f(inPlane[0], inPlane[1]), 30.0);
    QVERIFY(picked.has_value());
    QCOMPARE(picked->row, 6);
    QCOMPARE(picked->col, 17);

    // Nothing within the radius in the plane: no vertex.
    const cv::Vec3f far(17 * kSpacing, 6 * kSpacing, 200.0f);
    const cv::Vec3f farInPlane = plane.project(far, 1.0f, 1.0f);
    QVERIFY(!tool.pickPlaneConstraintVertex(plane, cv::Vec2f(farInPlane[0], farInPlane[1]), 30.0));
}

void TestManualAddExtend::extensionRefusesWithoutRoom()
{
    const auto points = flatSheet(5, 8, cv::Rect(0, 0, 8, 5));
    ManualAddTool tool;
    QVERIFY(tool.begin(points, splineConfig()));
    std::string status;
    QVERIFY(!tool.beginExtension(ManualAddTool::ExtendSide::Right, 3, std::nullopt, &status));
    QVERIFY(!tool.extensionActive());
    QVERIFY(!status.empty());

    // A single valid column has no direction to continue.
    const auto line = flatSheet(5, 8, cv::Rect(0, 0, 1, 5));
    QVERIFY(tool.begin(line, splineConfig()));
    QVERIFY(!tool.beginExtension(ManualAddTool::ExtendSide::Right, 3, std::nullopt, &status));
}

void TestManualAddExtend::clearPendingDropsTheExtension()
{
    const auto points = flatSheet(6, 16, cv::Rect(0, 0, 8, 6));
    ManualAddTool tool;
    QVERIFY(tool.begin(points, splineConfig()));
    std::string status;
    QVERIFY(tool.beginExtension(ManualAddTool::ExtendSide::Right, 4, std::nullopt, &status));
    // A pending extension blocks hole-fill clicks and a second extension.
    QVERIFY(!tool.commitHover(&status));
    QVERIFY(!tool.beginExtension(ManualAddTool::ExtendSide::Right, 4, std::nullopt, &status));

    QVERIFY(tool.clearPending(splineConfig()));
    QVERIFY(!tool.extensionActive());
    QVERIFY(!tool.initialFillCommitted());
    QVERIFY(!vc::geometry::validSurfacePoint(tool.previewPoints()(0, 9)));
    QVERIFY(tool.beginExtension(ManualAddTool::ExtendSide::Right, 2, std::nullopt, &status));
}

void TestManualAddExtend::resizeSurfaceGridKeepsChannelsAligned()
{
    auto points = flatSheet(4, 5, cv::Rect(0, 0, 5, 4));
    QuadSurface surface(points, cv::Vec2f(0.5f, 0.5f));
    cv::Mat_<uint8_t> approval(4, 5, static_cast<uint8_t>(0));
    approval(0, 0) = 255;
    surface.setChannel("approval", approval);
    // A channel at twice the grid resolution keeps that resolution.
    cv::Mat_<uint8_t> fine(8, 10, static_cast<uint8_t>(0));
    fine(0, 0) = 7;
    surface.setChannel("fine", fine);

    QVERIFY(segmentation::resizeSurfaceGrid(surface, 1, 0, 2, 3));
    const auto* grown = surface.rawPointsPtr();
    QCOMPARE(grown->rows, 5);
    QCOMPARE(grown->cols, 10);
    QCOMPARE((*grown)(1, 2), points(0, 0));
    QVERIFY(!vc::geometry::validSurfacePoint((*grown)(0, 0)));
    cv::Mat grownApproval = surface.channel("approval", SURF_CHANNEL_NORESIZE);
    QCOMPARE(grownApproval.size(), cv::Size(10, 5));
    QCOMPARE(grownApproval.at<uint8_t>(1, 2), static_cast<uint8_t>(255));
    cv::Mat grownFine = surface.channel("fine", SURF_CHANNEL_NORESIZE);
    QCOMPARE(grownFine.size(), cv::Size(20, 10));
    QCOMPARE(grownFine.at<uint8_t>(2, 4), static_cast<uint8_t>(7));

    // Shrinking by the same amounts restores the original grid exactly.
    QVERIFY(segmentation::resizeSurfaceGrid(surface, -1, 0, -2, -3));
    QCOMPARE(surface.rawPointsPtr()->size(), points.size());
    QCOMPARE(cv::norm(*surface.rawPointsPtr() - points, cv::NORM_INF), 0.0);
    QCOMPARE(cv::norm(surface.channel("approval", SURF_CHANNEL_NORESIZE), approval, cv::NORM_INF), 0.0);

    QVERIFY(!segmentation::resizeSurfaceGrid(surface, -4, 0, 0, 0));
    QCOMPARE(surface.rawPointsPtr()->size(), points.size());
}

void TestManualAddExtend::resizeSurfaceGridRejectsLossyChannels()
{
    const auto points = flatSheet(4, 5, cv::Rect(0, 0, 5, 4));
    QuadSurface surface(points.clone(), cv::Vec2f(0.5f, 0.5f));
    cv::Mat_<uint8_t> aligned(8, 10, uint8_t{7});
    cv::Mat_<uint8_t> unaligned(7, 9);
    for (int y = 0; y < unaligned.rows; ++y)
        for (int x = 0; x < unaligned.cols; ++x)
            unaligned(y, x) = static_cast<uint8_t>(y * unaligned.cols + x);
    surface.setChannel("aligned", aligned.clone());
    surface.setChannel("unaligned", unaligned.clone());

    for (int sign : {1, -1}) {
        QVERIFY(!segmentation::resizeSurfaceGrid(surface, sign, 0, sign, 0));
        QCOMPARE(surface.rawPointsPtr()->size(), points.size());
        QCOMPARE(cv::norm(*surface.rawPointsPtr(), points, cv::NORM_INF), 0.0);
        for (const auto& name : {"aligned", "unaligned"}) {
            const cv::Mat expected = std::string(name) == "aligned" ? aligned : unaligned;
            const cv::Mat actual = surface.channel(name, SURF_CHANNEL_NORESIZE);
            QCOMPARE(actual.size(), expected.size());
            QCOMPARE(cv::norm(actual, expected, cv::NORM_INF), 0.0);
        }
    }
}

void TestManualAddExtend::maskCellsBecomeValid()
{
    QuadSurface surface(flatSheet(4, 4, cv::Rect(0, 0, 4, 4)), cv::Vec2f(0.5f, 0.5f));
    QVERIFY(!segmentation::markMaskCellsValid(surface, {cv::Point(1, 1)}));

    cv::Mat_<uint8_t> mask(8, 8, static_cast<uint8_t>(255));
    mask(cv::Rect(2, 2, 2, 2)).setTo(0);
    surface.setChannel("mask", mask);
    QVERIFY(segmentation::markMaskCellsValid(surface, {cv::Point(1, 1)}));
    cv::Mat updated = surface.channel("mask", SURF_CHANNEL_NORESIZE);
    QCOMPARE(cv::countNonZero(updated), 64);
}

void TestManualAddExtend::maskedFillSurvivesReload()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const std::filesystem::path path = std::filesystem::path(dir.path().toStdString()) / "segment";

    // A segment saved with a hole and a mask that marks the hole invalid.
    auto points = flatSheet(6, 6, cv::Rect(0, 0, 6, 6));
    points(2, 3) = kInvalid;
    {
        QuadSurface surface(points, cv::Vec2f(0.5f, 0.5f));
        surface.save(path);
        surface.writeValidMask();
    }

    const cv::Vec3f filled(3 * kSpacing, 2 * kSpacing, 100.0f);
    auto fillAndReload = [&](bool markMask) -> cv::Vec3f {
        QuadSurface surface(path);
        surface.ensureLoaded();
        (*surface.rawPointsPtr())(2, 3) = filled;
        surface.invalidateCache();
        if (markMask && !segmentation::markMaskCellsValid(surface, {cv::Point(3, 2)})) {
            return kInvalid;
        }
        surface.saveOverwrite();
        QuadSurface reloaded(path);
        reloaded.ensureLoaded();
        return reloaded.rawPoints()(2, 3);
    };

    // Without marking the mask, the reload drops the filled vertex.
    QVERIFY(!vc::geometry::validSurfacePoint(fillAndReload(false)));
    QVERIFY(distance(fillAndReload(true), filled) < 1e-4f);
}

void TestManualAddExtend::dragPicksTheNearestEdgeStretch()
{
    // Rows 1 to 4 and columns 2 to 7 are valid in a 6 x 10 grid.
    const auto points = flatSheet(6, 10, cv::Rect(2, 1, 6, 4));
    using Side = ManualAddTool::ExtendSide;

    auto pick = ManualAddTool::edgeFromDrag(points, cv::Point(3, 4), cv::Point(6, 5));
    QVERIFY(pick);
    QCOMPARE(pick->side, Side::Down);
    QCOMPARE(pick->span.first, 3);
    QCOMPARE(pick->span.last, 6);

    pick = ManualAddTool::edgeFromDrag(points, cv::Point(5, 0), cv::Point(2, 1));
    QVERIFY(pick);
    QCOMPARE(pick->side, Side::Up);
    QCOMPARE(pick->span.first, 2);
    QCOMPARE(pick->span.last, 5);

    pick = ManualAddTool::edgeFromDrag(points, cv::Point(8, 1), cv::Point(8, 3));
    QVERIFY(pick);
    QCOMPARE(pick->side, Side::Right);
    QCOMPARE(pick->span.first, 1);
    QCOMPARE(pick->span.last, 3);

    pick = ManualAddTool::edgeFromDrag(points, cv::Point(1, 2), cv::Point(2, 4));
    QVERIFY(pick);
    QCOMPARE(pick->side, Side::Left);

    // A single line, or only lines with no valid vertex, pick nothing.
    QVERIFY(!ManualAddTool::edgeFromDrag(points, cv::Point(4, 4), cv::Point(4, 4)));
    QVERIFY(!ManualAddTool::edgeFromDrag(points, cv::Point(8, 0), cv::Point(9, 0)));
}

void TestManualAddExtend::extensionKeepsItsSpan()
{
    const auto points = flatSheet(8, 12, cv::Rect(0, 0, 6, 8));
    ManualAddTool tool;
    QVERIFY(tool.begin(points, splineConfig()));
    QVERIFY(!tool.extensionSpan());
    std::string status;
    QVERIFY2(tool.beginExtension(ManualAddTool::ExtendSide::Right, 3, ManualAddTool::EdgeSpan{2, 5}, &status),
             status.c_str());
    QVERIFY(tool.extensionSpan());
    QCOMPARE(tool.extensionSpan()->first, 2);
    QCOMPARE(tool.extensionSpan()->last, 5);
    // Only rows 2 to 5 get new vertices.
    for (const auto& key : tool.fillVertices()) {
        QVERIFY(key.row >= 2 && key.row <= 5);
    }
}

void TestManualAddExtend::undoHistoryFollowsGridPadding()
{
    segmentation::UndoHistory history;
    cv::Mat_<cv::Vec3f> snapshot(2, 3, cv::Vec3f(1.0f, 2.0f, 3.0f));
    history.setGridSize(snapshot.size());
    QVERIFY(history.capture(snapshot));
    QVERIFY(history.captureDelta({{1, 2, cv::Vec3f(4.0f, 5.0f, 6.0f)}}));

    history.padGrid(1, 2, 3, 0);
    // The grown grid is the size the history now addresses, so it is kept.
    history.setGridSize(cv::Size(6, 5));
    QCOMPARE(history.size(), std::size_t{2});

    const auto deltas = history.takeUndo();
    QVERIFY(deltas && deltas->isDelta());
    QCOMPARE(deltas->deltas[0].row, 2);
    QCOMPARE(deltas->deltas[0].col, 5);
    const auto padded = history.takeUndo();
    QVERIFY(padded && !padded->isDelta());
    QCOMPARE(padded->points.rows, 5);
    QCOMPARE(padded->points.cols, 6);
    QCOMPARE(padded->points(1, 3), cv::Vec3f(1.0f, 2.0f, 3.0f));
    QCOMPARE(padded->points(0, 0), kInvalid);
    QCOMPARE(padded->points(4, 5), kInvalid);
    QCOMPARE(history.bytes(), std::size_t{0});
}

namespace {
// A sheet curled around the z axis: radius 40 in x and y, flat in z, with row
// 0 at z = 100 so that rows grown above it keep z > 0 (validSurfacePoint).
constexpr float kRadius = 40.0f;
constexpr float kCurvedZ0 = 100.0f;
cv::Mat_<cv::Vec3f> curvedSheet(int rows, int cols, int validCols)
{
    cv::Mat_<cv::Vec3f> points(rows, cols, kInvalid);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < validCols; ++col) {
            const float angle = col * kSpacing / kRadius;
            points(row, col) = cv::Vec3f(kRadius * std::cos(angle), kRadius * std::sin(angle), kCurvedZ0 + row * kSpacing);
        }
    }
    return points;
}
std::optional<cv::Vec3f> cylinderNormal(const cv::Vec3f& p)
{
    const float r = std::hypot(p[0], p[1]);
    if (r < 1e-3f) {
        return std::nullopt;
    }
    return cv::Vec3f(p[0] / r, p[1] / r, 0.0f);
}
} // namespace

void TestManualAddExtend::sheetWalkFollowsACurvedSheet()
{
    const int cells = 12;
    const auto points = curvedSheet(6, 30, 10);
    ManualAddTool tool;
    QVERIFY(tool.begin(points, splineConfig()));
    std::string status;
    QVERIFY2(tool.beginExtension(ManualAddTool::ExtendSide::Right, cells, std::nullopt, &status), status.c_str());

    // The straight continuation leaves the sheet...
    float straightWorst = 0.0f;
    for (const auto& key : tool.fillVertices()) {
        const auto& p = tool.previewPoints()(key.row, key.col);
        straightWorst = std::max(straightWorst, std::abs(std::hypot(p[0], p[1]) - kRadius));
    }
    QVERIFY(straightWorst > 3.0f);

    // ...and the walk along the sheet normals stays on it.
    const auto walk = tool.sheetWalk();
    QVERIFY(walk);
    std::vector<cv::Vec3f> guess(walk->origins.size() * static_cast<std::size_t>(walk->cells));
    for (std::size_t line = 0; line < walk->origins.size(); ++line) {
        ManualAddTool::walkSheetLine(*walk, line, cylinderNormal, guess);
    }
    QVERIFY(!tool.applySheetWalk(walk->extensionId + 1, guess));
    QVERIFY2(tool.applySheetWalk(walk->extensionId, guess, &status), status.c_str());
    QVERIFY(tool.extensionFollowsSheet());
    for (const auto& key : tool.fillVertices()) {
        const auto& p = tool.previewPoints()(key.row, key.col);
        QVERIFY2(std::abs(std::hypot(p[0], p[1]) - kRadius) < 1.0f,
                 qPrintable(QString("radius %1").arg(std::hypot(p[0], p[1]))));
        QVERIFY(std::abs(p[2] - (kCurvedZ0 + key.row * kSpacing)) < 1e-3f);
    }
}

void TestManualAddExtend::guidedClicksStayLocal()
{
    const auto points = flatSheet(20, 40, cv::Rect(0, 0, 10, 20));
    auto config = splineConfig();
    config.guidedClickReach = 2.0;
    config.planeConstraintReplacementRadius = 0.0;
    ManualAddTool tool;
    QVERIFY(tool.begin(points, config));
    std::string status;
    QVERIFY2(tool.beginExtension(ManualAddTool::ExtendSide::Right, 20, std::nullopt, &status), status.c_str());
    const auto straight = tool.previewPoints().clone();

    // Unguided: a click bends the strip far from it too.
    QVERIFY2(tool.addOrReplacePlaneConstraint(2, 25, straight(2, 25) + cv::Vec3f(0.0f, 0.0f, 10.0f), &status),
             status.c_str());
    QVERIFY(distance(tool.previewPoints()(17, 25), straight(17, 25)) > 0.5f);

    // Guided: the same click only corrects its neighborhood.
    const auto walk = tool.sheetWalk();
    QVERIFY(walk);
    std::vector<cv::Vec3f> guess;
    for (const auto& key : tool.fillVertices()) {
        guess.push_back(straight(key.row, key.col));
    }
    QVERIFY2(tool.applySheetWalk(walk->extensionId, guess, &status), status.c_str());
    QVERIFY(distance(tool.previewPoints()(2, 25), straight(2, 25) + cv::Vec3f(0.0f, 0.0f, 10.0f)) < 0.5f);
    QVERIFY(distance(tool.previewPoints()(17, 25), straight(17, 25)) < 1e-3f);
}

void TestManualAddExtend::autoExtendGrowsAlongTheSheet()
{
    QuadSurface surface(curvedSheet(6, 10, 10), cv::Vec2f(0.5f, 0.5f));
    segmentation::AutoExtendOptions options;
    options.sides = {ManualAddTool::ExtendSide::Right, ManualAddTool::ExtendSide::Up};
    options.cells = 5;
    options.rounds = 3;
    options.threads = 2;
    std::string status;
    const auto result = segmentation::autoExtendSurface(surface, cylinderNormal, options, &status);
    QVERIFY2(result, status.c_str());

    // Right then Up, three times: each round starts from the edge the last one left.
    QCOMPARE(static_cast<int>(result->steps.size()), 6);
    QCOMPARE(result->steps[0].added, 6 * 5);
    QCOMPARE(result->steps[1].added, 15 * 5);
    QCOMPARE(result->steps[2].added, 11 * 5);
    QCOMPARE(result->added, 30 + 75 + 55 + 100 + 80 + 125);
    const auto& points = *surface.rawPointsPtr();
    QCOMPARE(points.rows, 6 + 15);
    QCOMPARE(points.cols, 10 + 15);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            const auto& p = points(row, col);
            QVERIFY(vc::geometry::validSurfacePoint(p));
            QVERIFY2(std::abs(std::hypot(p[0], p[1]) - kRadius) < 1.0f,
                     qPrintable(QString("(%1, %2) radius %3").arg(row).arg(col).arg(std::hypot(p[0], p[1]))));
        }
    }
    for (const auto& step : result->steps) {
        QCOMPARE(step.stepsWithoutNormal, 0);
    }

    // Without normals every step counts as missing and the lines run straight.
    QuadSurface flat(flatSheet(6, 10, cv::Rect(0, 0, 10, 6)), cv::Vec2f(0.5f, 0.5f));
    options.sides = {ManualAddTool::ExtendSide::Right};
    options.rounds = 1;
    const auto none = segmentation::autoExtendSurface(
        flat, [](const cv::Vec3f&) { return std::optional<cv::Vec3f>(); }, options, &status);
    QVERIFY2(none, status.c_str());
    // Each vertex queries the front normal and then the projection midpoint.
    QCOMPARE(none->steps[0].stepsWithoutNormal, 2 * 6 * 5);
    QVERIFY(distance((*flat.rawPointsPtr())(3, 14), cv::Vec3f(14 * kSpacing, 3 * kSpacing, 100.0f)) < 1e-3f);
}

void TestManualAddExtend::autoExtendPreservesLazyChannelsWithoutResizing()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const auto root = std::filesystem::path(dir.path().toStdString());
    const auto input = root / "input";
    const auto output = root / "output";
    const auto points = flatSheet(4, 7, cv::Rect(0, 0, 4, 4));
    cv::Mat_<uint16_t> ink(points.rows * 2, points.cols * 2);
    for (int row = 0; row < ink.rows; ++row)
        for (int col = 0; col < ink.cols; ++col)
            ink(row, col) = static_cast<uint16_t>(row * ink.cols + col + 256);
    {
        QuadSurface surface(points, cv::Vec2f(0.5f, 0.5f));
        surface.setChannel("ink", ink);
        surface.save(input);
    }
    auto surface = load_quad_from_tifxyz(input);
    QVERIFY(!surface->channelLoaded("ink"));
    segmentation::AutoExtendOptions options;
    options.sides = {ManualAddTool::ExtendSide::Right};
    options.cells = 2;
    options.rounds = 1;
    options.threads = 1;
    std::string status;
    const auto result = segmentation::autoExtendSurface(*surface, {}, options, &status);
    QVERIFY2(result, status.c_str());
    QCOMPARE(result->added, 8);
    QCOMPARE(surface->rawPointsPtr()->size(), points.size());
    surface->saveAsNew(output);

    auto reloaded = load_quad_from_tifxyz(output);
    const cv::Mat savedInk = reloaded->channel("ink", SURF_CHANNEL_NORESIZE);
    QVERIFY(!savedInk.empty());
    QCOMPARE(savedInk.size(), ink.size());
    QCOMPARE(savedInk.type(), ink.type());
    QCOMPARE(cv::norm(savedInk, ink, cv::NORM_INF), 0.0);
}

void TestManualAddExtend::sheetFrontDoesNotBridgeGridGaps()
{
    for (bool staggered : {false, true}) {
        ManualAddTool::SheetWalk walk;
        walk.cells = 3;
        walk.origins = {{100, 100, 100}, {100, 102, 100}, {120, 200, 100}, {120, 202, 100}};
        walk.steps = {{2, 0, 0}, {2, 0, 0}, {-2, 0, 0}, {-2, 0, 0}};
        std::vector<SegmentationEditManager::GridKey> fill;
        for (int line = 0; line < 4; ++line) {
            const int row = staggered ? line : (line < 2 ? line : line + 8);
            const int start = staggered && line >= 2 ? 20 : 10;
            for (int k = 0; k < walk.cells; ++k) fill.push_back({row, start + k});
        }
        std::vector<cv::Vec3f> result(fill.size());
        ManualAddTool::walkSheetExtension(walk, fill,
            [](const cv::Vec3f&) { return std::optional<cv::Vec3f>({0, 0, 1}); }, result, 4);
        for (int line = 0; line < 4; ++line)
            for (int k = 0; k < walk.cells; ++k)
                QCOMPARE(result[line * walk.cells + k], walk.origins[line] + walk.steps[line] * float(k + 1));
    }
}

void TestManualAddExtend::streamlineConverges_data()
{
    QTest::addColumn<int>("method");
    QTest::newRow("midpoint") << 0;
    QTest::newRow("heun") << 1;
    QTest::newRow("rk4") << 2;
}

void TestManualAddExtend::streamlineConverges()
{
    QFETCH(int, method);
    // A unit-speed great circle on a sphere: both exact position and tangent
    // are known. This exercises direction transport, not just normal projection.
    constexpr double radius = 40;
    const auto run = [&](double h, bool flip) {
        ManualAddTool::SheetWalk walk;
        walk.cells = 8;
        walk.origins = {cv::Vec3f(radius, 0, 0)};
        walk.steps = {cv::Vec3f(0, 3, 4)};
        walk.options.maxStep = h;
        walk.options.mode = ManualAddTool::SheetWalkMode::Streamline;
        walk.options.integrator = static_cast<ManualAddTool::SheetIntegrator>(method);
        std::vector<cv::Vec3f> result(walk.cells);
        ManualAddTool::walkSheetLine(walk, 0, [flip](const cv::Vec3f& p) -> std::optional<cv::Vec3f> {
            const float sign = flip && static_cast<int>(std::floor(p[1])) % 2 ? -1.f : 1.f;
            return p * (sign / static_cast<float>(cv::norm(p)));
        }, result);
        return result.back();
    };
    const cv::Vec3f expected(radius * std::cos(1.0), radius * 0.6 * std::sin(1.0),
                             radius * 0.8 * std::sin(1.0));
    const auto coarse = run(5, false), fine = run(1.25, false);
    const float coarseError = distance(coarse, expected), fineError = distance(fine, expected);
    QVERIFY2(fineError < 0.01f, qPrintable(QString::number(fineError)));
    // RK4 reaches the fixed finite-difference / float-normal accuracy floor.
    QVERIFY2(fineError < coarseError || fineError < 0.001f,
             qPrintable(QString("coarse %1 fine %2").arg(coarseError).arg(fineError)));
    QVERIFY(distance(fine, run(1.25, true)) < 1e-5f);
}

void TestManualAddExtend::streamlineReusesEndpointNormals()
{
    for (int method = 0; method < 3; ++method) {
        ManualAddTool::SheetWalk walk;
        walk.cells = 4;
        walk.origins = {cv::Vec3f(40, 0, 0), cv::Vec3f(40, 0, 0)};
        walk.steps = {cv::Vec3f(0, 3, 4), cv::Vec3f(0, 3, 4)};
        walk.options.maxStep = 1.25; // Four substeps per output vertex.
        walk.options.mode = ManualAddTool::SheetWalkMode::Streamline;
        walk.options.integrator = static_cast<ManualAddTool::SheetIntegrator>(method);
        std::vector<cv::Vec3f> points(walk.cells * walk.origins.size());
        int calls = 0;
        int starts = 0;
        const auto normal = [&](const cv::Vec3f& p) -> std::optional<cv::Vec3f> {
            ++calls;
            if (p == walk.origins.front()) ++starts;
            return p / static_cast<float>(cv::norm(p));
        };
        ManualAddTool::walkSheetLines(walk, normal, points, 1);
        // The starting normal is requested once per line, not again for k1.
        // Subsequent substeps reuse their predecessor's endpoint normal.
        QCOMPARE(starts, 2);
        const int stages = method == 2 ? 4 : 2;
        QCOMPARE(calls, 2 * (1 + stages * 3 * 4 * walk.cells));
        for (int k = 0; k < walk.cells; ++k) QCOMPARE(points[k], points[k + walk.cells]);
        const auto serial = points;
        ManualAddTool::walkSheetLines(walk, [](const cv::Vec3f& p) -> std::optional<cv::Vec3f> {
            return p / static_cast<float>(cv::norm(p));
        }, points, 2);
        QCOMPARE(points, serial);
    }
}

void TestManualAddExtend::streamlineDropsEndpointNormalAfterFallback()
{
    ManualAddTool::SheetWalk walk;
    walk.cells = 3;
    walk.origins = {cv::Vec3f(0, 0, 0)};
    walk.steps = {cv::Vec3f(1, 0, 0)};
    walk.options.maxStep = 1;
    walk.options.mode = ManualAddTool::SheetWalkMode::Streamline;
    walk.options.integrator = ManualAddTool::SheetIntegrator::Midpoint;
    std::vector<cv::Vec3f> points(walk.cells);
    int afterFallbackQueries = 0;
    ManualAddTool::walkSheetLine(walk, 0, [&](const cv::Vec3f& p) -> std::optional<cv::Vec3f> {
        // Fail a derivative stage on the second substep. Projection takes us
        // to x=2, where the next step must fetch a fresh starting normal.
        if (p[0] == 1.25f) return std::nullopt;
        if (p[0] == 2.f) ++afterFallbackQueries;
        return cv::Vec3f(0, 0, 1);
    }, points);
    QCOMPARE(afterFallbackQueries, 1);
    for (int k = 0; k < walk.cells; ++k) QCOMPARE(points[k], cv::Vec3f(k + 1, 0, 0));
}

void TestManualAddExtend::substepsPreserveSpacingAndHandleMissingNormals()
{
    for (int mode = 0; mode < 2; ++mode) {
        for (int method = 0; method < 3; ++method) {
            ManualAddTool::SheetWalk walk;
            walk.cells = 3;
            walk.origins = {cv::Vec3f(10, 20, 30)};
            walk.steps = {cv::Vec3f(2, 0, 0)};
            walk.options.maxStep = 0.3;
            walk.options.mode = static_cast<ManualAddTool::SheetWalkMode>(mode);
            walk.options.integrator = static_cast<ManualAddTool::SheetIntegrator>(method);
            for (int field = 0; field < 4; ++field) {
                std::vector<cv::Vec3f> result(walk.cells);
                ManualAddTool::walkSheetLine(walk, 0, [field](const cv::Vec3f&) -> std::optional<cv::Vec3f> {
                    if (field == 0) return std::nullopt;
                    if (field == 1) return cv::Vec3f(0, 0, 1);
                    if (field == 2) return cv::Vec3f(1, 0, 0); // degenerate projection
                    return cv::Vec3f(std::numeric_limits<float>::quiet_NaN(), 0, 0);
                }, result);
                for (int k = 0; k < walk.cells; ++k)
                    QVERIFY(distance(result[k], walk.origins[0] + walk.steps[0] * float(k + 1)) < 1e-5f);
            }
            walk.steps[0] = cv::Vec3f(0, 0, 0);
            std::vector<cv::Vec3f> stationary(walk.cells);
            ManualAddTool::walkSheetLine(walk, 0, {}, stationary);
            for (const auto& point : stationary) QCOMPARE(point, walk.origins[0]);
        }
    }
}

void TestManualAddExtend::projectionSubstepsReduceDrift()
{
    const auto run = [](double h) {
        ManualAddTool::SheetWalk walk;
        walk.cells = 10;
        walk.origins = {cv::Vec3f(20, 0, 0)};
        walk.steps = {cv::Vec3f(0, 4, 0)};
        walk.options.maxStep = h;
        std::vector<cv::Vec3f> result(walk.cells);
        ManualAddTool::walkSheetLine(walk, 0, [](const cv::Vec3f& p) -> std::optional<cv::Vec3f> {
            return p / static_cast<float>(cv::norm(p));
        }, result);
        const cv::Vec3f expected(20 * std::cos(2.0), 20 * std::sin(2.0), 0);
        return distance(result.back(), expected);
    };
    QVERIFY(run(0.5) < run(0));
}

void TestManualAddExtend::autoExtendReportsNormalFailures()
{
    const auto points = flatSheet(8, 5, cv::Rect(0, 0, 5, 8));
    cv::Mat_<uint8_t> mask(16, 10, uint8_t{255});
    mask(0, 0) = 0;
    for (int threads : {1, 4}) {
        for (bool standardException : {true, false}) {
            QuadSurface surface(points.clone(), cv::Vec2f(0.5f, 0.5f));
            surface.setChannel("mask", mask.clone());
            segmentation::AutoExtendOptions options;
            options.sides = {ManualAddTool::ExtendSide::Right};
            options.cells = 2;
            options.rounds = 1;
            options.threads = threads;
            std::string status;
            const auto result = segmentation::autoExtendSurface(surface,
                [standardException](const cv::Vec3f&) -> std::optional<cv::Vec3f> {
                    if (standardException) throw std::runtime_error("chunk fetch failed");
                    throw 42;
                }, options, &status);
            QVERIFY(!result);
            QCOMPARE(status, standardException ? std::string("Sheet following failed: chunk fetch failed")
                                               : std::string("Sheet following failed: unknown error"));
            QCOMPARE(surface.rawPointsPtr()->size(), points.size());
            QCOMPARE(cv::norm(*surface.rawPointsPtr(), points, cv::NORM_INF), 0.0);
            const auto restoredMask = surface.channel("mask", SURF_CHANNEL_NORESIZE);
            QCOMPARE(restoredMask.size(), mask.size());
            QCOMPARE(cv::norm(restoredMask, mask, cv::NORM_INF), 0.0);

            const auto retry = segmentation::autoExtendSurface(surface,
                [](const cv::Vec3f&) { return cv::Vec3f(0, 0, 1); }, options);
            QVERIFY(retry);
            QCOMPARE(retry->added, points.rows * options.cells);
        }
    }
}

QTEST_APPLESS_MAIN(TestManualAddExtend)
#include "test_manual_add_extend.moc"
