#include <QtTest/QtTest>

#include "segmentation/SurfaceGridEdit.hpp"
#include "segmentation/tools/ManualAddTool.hpp"

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <filesystem>

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
    void maskCellsBecomeValid();
    void maskedFillSurvivesReload();
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
    QVERIFY(ManualAddTool::isInvalidPoint(padded(0, 0)));
    QVERIFY(ManualAddTool::isInvalidPoint(padded(1, 1)));
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
        QVERIFY(ManualAddTool::isInvalidPoint(preview(row, 16)));
        QCOMPARE(preview(row, 9), points(row, 9));
    }

    // The upward side continues columns the same way.
    ManualAddTool upTool;
    const auto lower = flatSheet(12, 6, cv::Rect(0, 5, 6, 7));
    QVERIFY(upTool.begin(lower, splineConfig()));
    QVERIFY(upTool.beginExtension(ManualAddTool::ExtendSide::Up, 3, std::nullopt, &status));
    QVERIFY(distance(upTool.previewPoints()(2, 4), cv::Vec3f(4 * kSpacing, 2 * kSpacing, 100.0f)) < 1e-3f);
    QVERIFY(ManualAddTool::isInvalidPoint(upTool.previewPoints()(1, 4)));
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
    QVERIFY(ManualAddTool::isInvalidPoint(tool.previewPoints()(0, 9)));
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
    QVERIFY(ManualAddTool::isInvalidPoint((*grown)(0, 0)));
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
    QVERIFY(ManualAddTool::isInvalidPoint(fillAndReload(false)));
    QVERIFY(distance(fillAndReload(true), filled) < 1e-4f);
}

QTEST_APPLESS_MAIN(TestManualAddExtend)
#include "test_manual_add_extend.moc"
