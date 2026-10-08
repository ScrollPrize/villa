// Coverage for buildGapField / gapFieldTiles in apps/VC3D/FiberMapGapField.cpp:
// the Fiber Map's stacked-sheet gap heat map. Layouts are constructed
// directly (the field reads only placed runs, anchors, the extent and the
// sheet model), so every case controls its own geometry exactly.

#include <QtTest/QtTest>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "FiberMapGapField.hpp"

using vc3d::fiber_map::GlobalAnchor;
using vc3d::fiber_map::GlobalPlacedFiber;
using vc3d::fiber_map::GlobalResult;
using vc3d::fiber_map::Run;
using vc3d::fiber_map::gaps::GapField;
using vc3d::fiber_map::gaps::GapFieldParams;
using vc3d::fiber_map::gaps::GapFieldTile;
using vc3d::fiber_map::gaps::buildGapField;
using vc3d::fiber_map::gaps::gapFieldTiles;
using vc3d::fiber_map::gaps::resampledToSheetDistance;
using vc3d::fiber_map::gaps::sameGapSettings;
using vc3d::fiber_map::sheetModelOf;

namespace
{

constexpr double kTwoPi = 2.0 * M_PI;

GlobalResult makeLayout(double rRef, double radius0, double pitch, double x0, double x1,
                        double yMin, double yMax)
{
    GlobalResult layout;
    layout.rRefVx = rRef;
    layout.sheetRadius0Vx = radius0;
    layout.sheetPitchVx = pitch;
    layout.x0Vx = x0;
    layout.x1Vx = x1;
    layout.yMinVx = yMin;
    layout.yMaxVx = yMax;
    return layout;
}

// A vertical (constant-x) fiber from yA to yB, one run.
void addVertical(GlobalResult& layout, double x, double yA, double yB, bool traced = true,
                 GlobalAnchor anchor = GlobalAnchor::Primary)
{
    GlobalPlacedFiber placed;
    placed.fiber.id = static_cast<uint64_t>(layout.fibers.size() + 1);
    placed.meta.anchor = anchor;
    Run run;
    run.traced = traced;
    run.points = {QPointF(x, yA), QPointF(x, yB)};
    placed.fiber.runs.push_back(run);
    layout.fibers.push_back(placed);
}

// A seed contained in one cell: two points a hair apart.
void addDot(GlobalResult& layout, double x, double y)
{
    GlobalPlacedFiber placed;
    placed.fiber.id = static_cast<uint64_t>(layout.fibers.size() + 1);
    placed.meta.anchor = GlobalAnchor::Primary;
    Run run;
    run.points = {QPointF(x, y), QPointF(x + 1e-3, y)};
    placed.fiber.runs.push_back(run);
    layout.fibers.push_back(placed);
}

float sampleAt(const GapField& field, double x, double y)
{
    const int col = static_cast<int>(std::floor((x - field.x0Vx) / field.cellVx));
    const int row = static_cast<int>(std::floor((y - field.y0Vx) / field.cellVx));
    if (col < 0 || col >= field.cols || row < 0 || row >= field.rows) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return field.at(row, col);
}

double sheetU(const GlobalResult& layout, double x)
{
    return vc3d::fiber_map::sheetDistanceVx(vc3d::fiber_map::sheetModelOf(layout), x);
}

// The re-gridding rule, spelled out independently: the smallest value of
// the source columns whose centres fall inside output cell (i, j), or the
// source column the cell's centre falls in when none does. NaN only when
// every candidate is NaN.
float expectedRegridded(const GapField& src, const vc3d::fiber_map::SheetModel& model,
                        const GapField& dst, int i, int j, bool* several)
{
    const double c = dst.cellVx;
    const double xa = vc3d::fiber_map::sheetXForDistanceMonotoneVx(model, dst.x0Vx + j * c);
    const double xb = vc3d::fiber_map::sheetXForDistanceMonotoneVx(model, dst.x0Vx + (j + 1) * c);
    const double xc = vc3d::fiber_map::sheetXForDistanceMonotoneVx(model, dst.x0Vx + (j + 0.5) * c);
    std::vector<int> candidates;
    for (int s = 0; s < src.cols; ++s) {
        const double centre = src.x0Vx + (s + 0.5) * src.cellVx;
        if (centre >= xa && centre < xb) {
            candidates.push_back(s);
        }
    }
    if (candidates.empty()) {
        candidates.push_back(std::clamp(static_cast<int>(std::floor((xc - src.x0Vx) / src.cellVx)),
                                        0, src.cols - 1));
    } else if (candidates.size() > 1 && several) {
        *several = true;
    }
    float value = std::numeric_limits<float>::quiet_NaN();
    for (const int s : candidates) {
        const float v = src.at(i, s);
        if (!std::isnan(v) && (std::isnan(value) || v < value)) {
            value = v;
        }
    }
    return value;
}

float finiteMinimum(const GapField& field)
{
    float best = std::numeric_limits<float>::infinity();
    for (const float v : field.distanceVx) {
        if (!std::isnan(v)) {
            best = std::min(best, v);
        }
    }
    return best;
}

}  // namespace

class TestFiberMapGapField : public QObject
{
    Q_OBJECT

private slots:
    void emptyLayoutYieldsEmptyField();
    void invalidParametersThrow();
    void unfoldedDistanceToVerticalLine();
    void farCellsReadExactlySaturation();
    void foldReachesFiberOnNextWinding();
    void foldReachesFiberOnPreviousWinding();
    void acrossWeightScalesAndZeroDisablesFold();
    void zeroPitchDisablesFold();
    void warpMeasuresArclengthAtTheCellsWinding();
    void interpolatedRunsSeedOnlyWhenAsked();
    void unresolvedAndSingletonRunsNeverSeed();
    void noSeedsReadsSaturationEverywhere();
    void outsideModelDomainIsNaN();
    void foldCapReportsTruncationOnlyWhenItBites();
    void cellCoarsensUnderBudget();
    void unachievableBudgetThrows();
    void interpolatesBetweenRasterColumns();
    void sameGapSettingsComparesWhatBuildsTheField();
    void transformIsExactDistanceToNearestSeedPixel();
    void degenerateGridsStillMeasure();
    void matchesBruteForceOracleAndIsDeterministic();
    void fadeTapersNeighbouringWindingsToNothing();
    void matchesBruteForceOracleWithFade();
    void tilesCoverTheFieldEdgeToEdge();
    void resamplesOntoSheetDistance();
    void resampledGridStaysUnderBudget();
};

void TestFiberMapGapField::emptyLayoutYieldsEmptyField()
{
    GlobalResult layout = makeLayout(1000.0, 1000.0, 0.0, 0.0, 1000.0, 0.0, 1000.0);
    QVERIFY(buildGapField(layout, GapFieldParams{}).empty());
    addVertical(layout, 100.0, 0.0, 500.0);
    layout.rRefVx = 0.0;
    QVERIFY(buildGapField(layout, GapFieldParams{}).empty());
    layout.rRefVx = 1000.0;
    layout.x1Vx = layout.x0Vx;
    QVERIFY(buildGapField(layout, GapFieldParams{}).empty());
    QVERIFY(gapFieldTiles(GapField{}, 8).empty());
}

void TestFiberMapGapField::invalidParametersThrow()
{
    GlobalResult layout = makeLayout(1000.0, 1000.0, 0.0, 0.0, 1000.0, 0.0, 1000.0);
    addVertical(layout, 100.0, 0.0, 500.0);
    GapFieldParams params;
    params.cellVx = 0.0;
    QVERIFY_THROWS_EXCEPTION(std::invalid_argument, buildGapField(layout, params));
    params = GapFieldParams{};
    params.saturationVx = std::numeric_limits<double>::infinity();
    QVERIFY_THROWS_EXCEPTION(std::invalid_argument, buildGapField(layout, params));
    params = GapFieldParams{};
    params.acrossWeight = -1.0;
    QVERIFY_THROWS_EXCEPTION(std::invalid_argument, buildGapField(layout, params));
    params = GapFieldParams{};
    params.maxFoldWindings = -1;
    QVERIFY_THROWS_EXCEPTION(std::invalid_argument, buildGapField(layout, params));
    params = GapFieldParams{};
    params.maxCells = 0;
    QVERIFY_THROWS_EXCEPTION(std::invalid_argument, buildGapField(layout, params));
    params = GapFieldParams{};
    params.fadeWindings = 0;
    QVERIFY_THROWS_EXCEPTION(std::invalid_argument, buildGapField(layout, params));
}

void TestFiberMapGapField::unachievableBudgetThrows()
{
    GlobalResult layout = makeLayout(4000.0, 4000.0, 0.0, 0.0, 20000.0, 0.0, 10000.0);
    addVertical(layout, 5000.0, 0.0, 10000.0);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 8000.0;
    for (const std::size_t budget : {std::size_t(1), std::size_t(2), std::size_t(3), std::size_t(4)}) {
        params.maxCells = budget;
        QVERIFY_THROWS_EXCEPTION(std::invalid_argument, buildGapField(layout, params));
    }
}

void TestFiberMapGapField::interpolatesBetweenRasterColumns()
{
    // pitch 0, so u == x and the raster geometry is known exactly: the seed
    // line lands in one raster column, and a query whose continuous column
    // coordinate is fractional must read the linear blend of its two
    // neighbouring transform columns, which for a single line is the exact
    // distance to the seed pixel's centre. A left-column sample would be off
    // by a large fraction of a cell here.
    const double cell = 100.0;
    const double seedX = 5030.0;
    const double saturation = 8000.0;
    // The extent reaches left past the raster's halo so its edge columns
    // are queryable.
    GlobalResult layout = makeLayout(4000.0, 4000.0, 0.0, -4000.0, 20000.0, 0.0, 10000.0);
    addVertical(layout, seedX, 0.0, 10000.0);
    GapFieldParams params;
    params.cellVx = cell;
    params.saturationVx = saturation;
    const GapField field = buildGapField(layout, params);
    const double halo = saturation + 2.0 * cell;
    const double uRaster0 = seedX - halo;
    const double seedPixelCentre = uRaster0 + (std::floor((seedX - uRaster0) / cell) + 0.5) * cell;
    QCOMPARE(seedPixelCentre, 5080.0);
    // Query cells whose centres sit at fractional raster columns: near the
    // raster's left edge (just under the saturation), mid-way, and near its
    // right edge (again just under the saturation).
    for (const double g : {2.7, 101.7, 160.7}) {
        const double u = uRaster0 + (g + 0.5) * cell;
        const int col = static_cast<int>(std::floor((u - field.x0Vx) / field.cellVx));
        QVERIFY(col >= 0 && col < field.cols);
        const double centre = field.x0Vx + (col + 0.5) * field.cellVx;
        const double expected = std::abs(centre - seedPixelCentre);
        QVERIFY2(expected < saturation, qPrintable(QString::number(expected)));
        const float got = field.at(field.rows / 2, col);
        QVERIFY2(std::abs(got - expected) <= 5.0,
                 qPrintable(QStringLiteral("g %1: field %2 expected %3").arg(g).arg(got).arg(expected)));
    }
}

void TestFiberMapGapField::sameGapSettingsComparesWhatBuildsTheField()
{
    GapFieldParams a;
    GapFieldParams b;
    QVERIFY(sameGapSettings(true, a, true, b));
    QVERIFY(sameGapSettings(false, a, false, b));
    // Off is off, whatever the parameters say.
    b.saturationVx *= 2.0;
    QVERIFY(sameGapSettings(false, a, false, b));
    QVERIFY(!sameGapSettings(true, a, true, b));
    QVERIFY(!sameGapSettings(true, a, false, a));
    b = a;
    b.acrossWeight = 0.5;
    QVERIFY(!sameGapSettings(true, a, true, b));
    b = a;
    b.cellVx += 1.0;
    QVERIFY(!sameGapSettings(true, a, true, b));
    b = a;
    b.seedInterpolated = !a.seedInterpolated;
    QVERIFY(!sameGapSettings(true, a, true, b));
    b = a;
    b.fade = !a.fade;
    QVERIFY(!sameGapSettings(true, a, true, b));
    // The fade count only matters while fading.
    a.fade = false;
    b = a;
    b.fadeWindings = a.fadeWindings + 1;
    QVERIFY(sameGapSettings(true, a, true, b));
    a.fade = true;
    b.fade = true;
    QVERIFY(!sameGapSettings(true, a, true, b));
}

void TestFiberMapGapField::unfoldedDistanceToVerticalLine()
{
    // pitch 0: u == x, no fold, the field is the plain distance to the line.
    GlobalResult layout = makeLayout(4000.0, 4000.0, 0.0, 0.0, 20000.0, 0.0, 10000.0);
    addVertical(layout, 5000.0, 0.0, 10000.0);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 8000.0;
    const GapField field = buildGapField(layout, params);
    QVERIFY(!field.empty());
    QVERIFY(!field.folded);
    QVERIFY(!field.cellCoarsened);
    QCOMPARE(field.seedFiberCount, 1);
    QCOMPARE(field.cols, 200);
    QCOMPARE(field.rows, 100);
    QVERIFY(std::abs(sampleAt(field, 7050.0, 5050.0) - 2050.0) <= params.cellVx);
    QVERIFY(std::abs(sampleAt(field, 2050.0, 9950.0) - 2950.0) <= params.cellVx);
    QVERIFY(sampleAt(field, 5050.0, 50.0) <= params.cellVx);
}

void TestFiberMapGapField::farCellsReadExactlySaturation()
{
    GlobalResult layout = makeLayout(4000.0, 4000.0, 0.0, 0.0, 20000.0, 0.0, 10000.0);
    addVertical(layout, 5000.0, 0.0, 10000.0);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 8000.0;
    const GapField field = buildGapField(layout, params);
    QCOMPARE(sampleAt(field, 15050.0, 5050.0), static_cast<float>(params.saturationVx));
    QCOMPARE(sampleAt(field, 19950.0, 50.0), static_cast<float>(params.saturationVx));
}

void TestFiberMapGapField::foldReachesFiberOnNextWinding()
{
    // r(W) = 4000 + 200 W; the only fiber sits on winding 1. The cell at the
    // same angle on winding 0 is a full turn away in the sheet but one pitch
    // away through it.
    const double rRef = 4000.0;
    const double period = kTwoPi * rRef;
    GlobalResult layout = makeLayout(rRef, 4000.0, 200.0, 0.0, 2.5 * period, 0.0, 10000.0);
    addVertical(layout, period + 3000.0, 0.0, 10000.0);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 5000.0;
    const GapField field = buildGapField(layout, params);
    QVERIFY(field.folded);
    QVERIFY(!field.foldTruncated);
    const float d = sampleAt(field, 3050.0, 5050.0);
    QVERIFY2(std::abs(d - 200.0) <= params.cellVx, qPrintable(QString::number(d)));
    // The fiber's own cell reads (near) zero.
    QVERIFY(sampleAt(field, period + 3050.0, 5050.0) <= params.cellVx);
    // Half a turn off in angle on winding 0 is genuinely far.
    QVERIFY(sampleAt(field, 3050.0 + 0.5 * period, 5050.0) > 1000.0f);
}

void TestFiberMapGapField::foldReachesFiberOnPreviousWinding()
{
    const double rRef = 4000.0;
    const double period = kTwoPi * rRef;
    GlobalResult layout = makeLayout(rRef, 4000.0, 200.0, 0.0, 3.5 * period, 0.0, 10000.0);
    addVertical(layout, period + 3000.0, 0.0, 10000.0);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 5000.0;
    const GapField field = buildGapField(layout, params);
    // Winding 2 at the fiber's angle: k = -1 wins with one pitch.
    const float d1 = sampleAt(field, 2.0 * period + 3050.0, 5050.0);
    QVERIFY2(std::abs(d1 - 200.0) <= params.cellVx, qPrintable(QString::number(d1)));
    // Winding 3: k = -2, two pitches.
    const float d2 = sampleAt(field, 3.0 * period + 3050.0, 5050.0);
    QVERIFY2(std::abs(d2 - 400.0) <= params.cellVx, qPrintable(QString::number(d2)));
}

void TestFiberMapGapField::acrossWeightScalesAndZeroDisablesFold()
{
    const double rRef = 4000.0;
    const double period = kTwoPi * rRef;
    GlobalResult layout = makeLayout(rRef, 4000.0, 200.0, 0.0, 2.5 * period, 0.0, 10000.0);
    addVertical(layout, period + 3000.0, 0.0, 10000.0);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 5000.0;
    params.acrossWeight = 2.0;
    const GapField doubled = buildGapField(layout, params);
    QVERIFY(doubled.folded);
    const float d = sampleAt(doubled, 3050.0, 5050.0);
    QVERIFY2(std::abs(d - 400.0) <= params.cellVx, qPrintable(QString::number(d)));
    params.acrossWeight = 0.0;
    const GapField unfolded = buildGapField(layout, params);
    QVERIFY(!unfolded.folded);
    QCOMPARE(sampleAt(unfolded, 3050.0, 5050.0), static_cast<float>(params.saturationVx));
}

void TestFiberMapGapField::zeroPitchDisablesFold()
{
    const double rRef = 4000.0;
    const double period = kTwoPi * rRef;
    GlobalResult layout = makeLayout(rRef, rRef, 0.0, 0.0, 2.5 * period, 0.0, 10000.0);
    addVertical(layout, period + 3000.0, 0.0, 10000.0);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 5000.0;
    const GapField field = buildGapField(layout, params);
    QVERIFY(!field.folded);
    QCOMPARE(sampleAt(field, 3050.0, 5050.0), static_cast<float>(params.saturationVx));
}

void TestFiberMapGapField::warpMeasuresArclengthAtTheCellsWinding()
{
    // The same map dx reads longer on an outer winding, in the ratio of the
    // modelled radii. Fold off so only the in-sheet term speaks.
    const double rRef = 4000.0;
    const double period = kTwoPi * rRef;
    const double radius0 = 4000.0;
    const double pitch = 500.0;
    GlobalResult layout = makeLayout(rRef, radius0, pitch, 0.0, 4.0 * period, 0.0, 10000.0);
    const double a = 0.25 * period;
    addVertical(layout, a, 0.0, 10000.0);
    addVertical(layout, a + 3.0 * period, 0.0, 10000.0);
    GapFieldParams params;
    params.cellVx = 50.0;
    params.saturationVx = 5000.0;
    params.acrossWeight = 0.0;
    const GapField field = buildGapField(layout, params);
    // Query at cell centres, so the only slack left is the seed's own
    // rasterisation (half a cell in u) plus the column interpolation.
    const auto centre = [&](double x) {
        return field.x0Vx + (std::floor((x - field.x0Vx) / field.cellVx) + 0.5) * field.cellVx;
    };
    const double dx = 2000.0;
    const double innerX = centre(a + dx);
    const double outerX = centre(a + 3.0 * period + dx);
    const double inner = sampleAt(field, innerX, 5025.0);
    const double outer = sampleAt(field, outerX, 5025.0);
    const double innerExpected = sheetU(layout, innerX) - sheetU(layout, a);
    const double outerExpected = sheetU(layout, outerX) - sheetU(layout, a + 3.0 * period);
    QVERIFY2(std::abs(inner - innerExpected) <= params.cellVx,
             qPrintable(QStringLiteral("%1 vs %2").arg(inner).arg(innerExpected)));
    QVERIFY2(std::abs(outer - outerExpected) <= params.cellVx,
             qPrintable(QStringLiteral("%1 vs %2").arg(outer).arg(outerExpected)));
    QVERIFY(outer > inner * 1.25);
}

void TestFiberMapGapField::interpolatedRunsSeedOnlyWhenAsked()
{
    GlobalResult layout = makeLayout(4000.0, 4000.0, 0.0, 0.0, 20000.0, 0.0, 10000.0);
    addVertical(layout, 5000.0, 0.0, 10000.0, /*traced=*/false);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 8000.0;
    params.seedInterpolated = false;
    const GapField without = buildGapField(layout, params);
    QCOMPARE(without.seedFiberCount, 0);
    QCOMPARE(sampleAt(without, 5050.0, 5050.0), static_cast<float>(params.saturationVx));
    params.seedInterpolated = true;
    const GapField with = buildGapField(layout, params);
    QCOMPARE(with.seedFiberCount, 1);
    QVERIFY(sampleAt(with, 5050.0, 5050.0) <= params.cellVx);
}

void TestFiberMapGapField::unresolvedAndSingletonRunsNeverSeed()
{
    GlobalResult layout = makeLayout(4000.0, 4000.0, 0.0, 0.0, 20000.0, 0.0, 10000.0);
    addVertical(layout, 5000.0, 0.0, 10000.0, true, GlobalAnchor::Unresolved);
    GlobalPlacedFiber singleton;
    singleton.fiber.id = 99;
    singleton.meta.anchor = GlobalAnchor::Primary;
    Run run;
    run.points = {QPointF(12000.0, 5000.0)};
    singleton.fiber.runs.push_back(run);
    layout.fibers.push_back(singleton);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 8000.0;
    const GapField field = buildGapField(layout, params);
    QCOMPARE(field.skippedUnresolvedCount, 1);
    QCOMPARE(field.seedFiberCount, 0);
    QCOMPARE(sampleAt(field, 5050.0, 5050.0), static_cast<float>(params.saturationVx));
    QCOMPARE(sampleAt(field, 12050.0, 5050.0), static_cast<float>(params.saturationVx));
}

void TestFiberMapGapField::noSeedsReadsSaturationEverywhere()
{
    GlobalResult layout = makeLayout(4000.0, 4000.0, 200.0, 0.0, 20000.0, 0.0, 10000.0);
    addVertical(layout, 5000.0, 0.0, 10000.0, true, GlobalAnchor::Unresolved);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 8000.0;
    const GapField field = buildGapField(layout, params);
    QVERIFY(!field.empty());
    QVERIFY(!field.folded);
    QVERIFY(std::all_of(field.distanceVx.begin(), field.distanceVx.end(),
                        [&](float v) { return v == static_cast<float>(params.saturationVx); }));
}

void TestFiberMapGapField::outsideModelDomainIsNaN()
{
    // r(W) = 1000 + 1000 W is positive only for W > -1: everything left of
    // x = -P has no sheet position.
    const double rRef = 1000.0;
    const double period = kTwoPi * rRef;
    GlobalResult layout = makeLayout(rRef, 1000.0, 1000.0, -2.0 * period, period, 0.0, 2000.0);
    addVertical(layout, 0.5 * period, 0.0, 2000.0);
    GapFieldParams params;
    params.cellVx = 50.0;
    params.saturationVx = 3000.0;
    const GapField field = buildGapField(layout, params);
    QVERIFY(std::isnan(sampleAt(field, -1.5 * period, 1000.0)));
    QVERIFY(std::isnan(sampleAt(field, -1.01 * period, 1000.0)));
    QVERIFY(std::isfinite(sampleAt(field, -0.9 * period, 1000.0)));
    QVERIFY(std::isfinite(sampleAt(field, 0.5 * period + 25.0, 1000.0)));
}

void TestFiberMapGapField::foldCapReportsTruncationOnlyWhenItBites()
{
    // The only fiber is three windings out; the cell at its angle on winding
    // 0 needs k = 3.
    const double rRef = 4000.0;
    const double period = kTwoPi * rRef;
    GlobalResult layout = makeLayout(rRef, 4000.0, 100.0, 0.0, 3.5 * period, 0.0, 10000.0);
    addVertical(layout, 3.0 * period + 3000.0, 0.0, 10000.0);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 5000.0;
    params.maxFoldWindings = 1;
    const GapField capped = buildGapField(layout, params);
    QVERIFY(capped.folded);
    QVERIFY(capped.foldTruncated);
    QCOMPARE(sampleAt(capped, 3050.0, 5050.0), static_cast<float>(params.saturationVx));
    params.maxFoldWindings = 8;
    const GapField full = buildGapField(layout, params);
    QVERIFY(!full.foldTruncated);
    const float d = sampleAt(full, 3050.0, 5050.0);
    QVERIFY2(std::abs(d - 300.0) <= params.cellVx, qPrintable(QString::number(d)));
    // A cap that is never reached because the saturation bounds k first.
    params.maxFoldWindings = 64;
    params.saturationVx = 250.0;
    const GapField saturated = buildGapField(layout, params);
    QVERIFY(!saturated.foldTruncated);
    QCOMPARE(sampleAt(saturated, 3050.0, 5050.0), static_cast<float>(params.saturationVx));
}

void TestFiberMapGapField::cellCoarsensUnderBudget()
{
    GlobalResult layout = makeLayout(4000.0, 4000.0, 0.0, 0.0, 20000.0, 0.0, 10000.0);
    addVertical(layout, 5000.0, 0.0, 10000.0);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 8000.0;
    params.maxCells = 6000;  // 200 x 100 = 20000 does not fit; 100 x 50 does
    const GapField field = buildGapField(layout, params);
    QVERIFY(field.cellCoarsened);
    QCOMPARE(field.cellVx, 200.0);
    QCOMPARE(field.cols, 100);
    QCOMPARE(field.rows, 50);
    QVERIFY(std::abs(sampleAt(field, 7100.0, 5100.0) - 2100.0) <= field.cellVx);
}

namespace
{

// Random dot seeds against an independent brute force over the seed pixels,
// with or without the fade; see the calling tests.
void checkAgainstBruteForce(bool fade, int fadeWindings)
{
    // Dot seeds each occupy exactly one raster pixel, so an independent
    // oracle knows the seed pixels: for each cell and each k, the distance
    // to the nearest seed pixel centre from the shifted query column. The
    // field interpolates the transform between columns, so it agrees to
    // within one cell.
    const double rRef = 3000.0;
    const double period = kTwoPi * rRef;
    const double radius0 = 3000.0;
    const double pitch = 150.0;
    GlobalResult layout = makeLayout(rRef, radius0, pitch, 0.0, 3.0 * period, 0.0, 3000.0);
    unsigned int state = 12345u;
    const auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<double>(state >> 8) / static_cast<double>(1u << 24);
    };
    for (int i = 0; i < 25; ++i) {
        addDot(layout, next() * 3.0 * period, next() * 3000.0);
    }
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 1500.0;
    params.fade = fade;
    params.fadeWindings = fadeWindings;
    const GapField field = buildGapField(layout, params);
    QVERIFY(field.folded);
    // Bitwise identical across thread counts: every cell is a function of
    // its own inputs only.
    const int threadsBefore = cv::getNumThreads();
    cv::setNumThreads(1);
    const GapField single = buildGapField(layout, params);
    cv::setNumThreads(4);
    const GapField quad = buildGapField(layout, params);
    cv::setNumThreads(threadsBefore);
    QCOMPARE(single.distanceVx, field.distanceVx);
    QCOMPARE(quad.distanceVx, field.distanceVx);

    const vc3d::fiber_map::SheetModel model = vc3d::fiber_map::sheetModelOf(layout);
    const double halo = params.saturationVx + 2.0 * params.cellVx;
    double uLo = std::numeric_limits<double>::infinity();
    struct Seed {
        double u;
        double y;
    };
    std::vector<Seed> seeds;
    for (const GlobalPlacedFiber& placed : layout.fibers) {
        const QPointF& p = placed.fiber.runs.front().points.front();
        seeds.push_back({vc3d::fiber_map::sheetDistanceVx(model, p.x()), p.y()});
        uLo = std::min(uLo, seeds.back().u);
    }
    const double uRaster0 = uLo - halo;
    const auto pixelCentreU = [&](double u) {
        return uRaster0 + (std::floor((u - uRaster0) / params.cellVx) + 0.5) * params.cellVx;
    };
    const auto pixelCentreY = [&](double y) {
        return layout.yMinVx + (std::floor((y - layout.yMinVx) / params.cellVx) + 0.5) * params.cellVx;
    };
    const double across = pitch;
    int kMax = static_cast<int>(std::ceil(params.saturationVx / across)) - 1;
    if (fade) {
        kMax = std::min(kMax, fadeWindings - 1);
    }
    const auto fadeIt = [&](int k, double value) {
        if (!fade) {
            return value;
        }
        const double f = std::min(1.0, static_cast<double>(std::abs(k)) / fadeWindings);
        return (1.0 - f) * value + f * params.saturationVx;
    };
    int checked = 0;
    for (int row = 0; row < field.rows; row += 3) {
        const double y = layout.yMinVx + (row + 0.5) * params.cellVx;
        for (int col = 0; col < field.cols; col += 7) {
            const double x = layout.x0Vx + (col + 0.5) * params.cellVx;
            double best = params.saturationVx;
            for (int k = -kMax; k <= kMax; ++k) {
                const double u = vc3d::fiber_map::sheetDistanceVx(model, x + k * period);
                for (const Seed& seed : seeds) {
                    const double du = u - pixelCentreU(seed.u);
                    const double dy = y - pixelCentreY(seed.y);
                    const double s = std::hypot(du, dy);
                    best = std::min(best, fadeIt(k, std::hypot(s, std::abs(k) * across)));
                }
            }
            const float got = field.at(row, col);
            QVERIFY2(std::abs(got - best) <= 1.0 * params.cellVx,
                     qPrintable(QStringLiteral("row %1 col %2: field %3 oracle %4")
                                    .arg(row).arg(col).arg(got).arg(best)));
            ++checked;
        }
    }
    QVERIFY(checked > 500);
}

}  // namespace

void TestFiberMapGapField::matchesBruteForceOracleAndIsDeterministic()
{
    checkAgainstBruteForce(false, 5);
}

void TestFiberMapGapField::matchesBruteForceOracleWithFade()
{
    checkAgainstBruteForce(true, 3);
}

void TestFiberMapGapField::fadeTapersNeighbouringWindingsToNothing()
{
    // r(W) = 4000 + 200 W. Fibers on windings 1, 2 and 5 only, at three
    // different angles so each is the sole candidate at its own angle. With
    // the fade at 5 windings, the cell at each angle on winding 0 reads the
    // candidate blended toward the saturation by k/5.
    const double rRef = 4000.0;
    const double period = kTwoPi * rRef;
    const double pitch = 200.0;
    const double saturation = 5000.0;
    GlobalResult layout = makeLayout(rRef, 4000.0, pitch, 0.0, 6.5 * period, 0.0, 10000.0);
    const double angle1 = 3000.0;
    const double angle2 = 3000.0 + 0.3 * period;
    const double angle5 = 3000.0 + 0.6 * period;
    addVertical(layout, 1.0 * period + angle1, 0.0, 10000.0);
    addVertical(layout, 2.0 * period + angle2, 0.0, 10000.0);
    addVertical(layout, 5.0 * period + angle5, 0.0, 10000.0);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = saturation;
    params.fade = true;
    params.fadeWindings = 5;
    const GapField field = buildGapField(layout, params);
    QVERIFY(field.folded);
    QVERIFY(field.faded);
    QVERIFY(!field.foldTruncated);
    const auto expect = [&](double x, double expected) {
        const float d = sampleAt(field, x, 5050.0);
        QVERIFY2(std::abs(d - expected) <= params.cellVx,
                 qPrintable(QStringLiteral("x %1: %2 vs %3").arg(x).arg(d).arg(expected)));
    };
    // k = 1: 0.8 * 200 + 0.2 * 5000.
    expect(angle1 + 50.0, 0.8 * pitch + 0.2 * saturation);
    // k = 2: 0.6 * 400 + 0.4 * 5000.
    expect(angle2 + 50.0, 0.6 * 2.0 * pitch + 0.4 * saturation);
    // k = 5: no influence at all.
    QCOMPARE(sampleAt(field, angle5 + 50.0, 5050.0), static_cast<float>(saturation));
    // The fiber's own winding is unaffected by the fade.
    QVERIFY(sampleAt(field, 1.0 * period + angle1 + 50.0, 5050.0) <= params.cellVx);
    // Without the fade the same cells read the plain fold.
    params.fade = false;
    const GapField plain = buildGapField(layout, params);
    QVERIFY(!plain.faded);
    QVERIFY(std::abs(sampleAt(plain, angle1 + 50.0, 5050.0) - pitch) <= params.cellVx);
    QVERIFY(std::abs(sampleAt(plain, angle5 + 50.0, 5050.0) - 5.0 * pitch) <= params.cellVx);
    // fadeWindings 1: only the cell's own winding; the fold still reports
    // as enabled.
    params.fade = true;
    params.fadeWindings = 1;
    const GapField own = buildGapField(layout, params);
    QVERIFY(own.folded);
    QCOMPARE(sampleAt(own, angle1 + 50.0, 5050.0), static_cast<float>(saturation));
}

void TestFiberMapGapField::tilesCoverTheFieldEdgeToEdge()
{
    GapField field;
    field.x0Vx = -700.0;
    field.y0Vx = 300.0;
    field.cellVx = 50.0;
    field.cols = 10;
    field.rows = 7;
    field.distanceVx.assign(70, 1.0f);
    const std::vector<GapFieldTile> tiles = gapFieldTiles(field, 4);
    QCOMPARE(tiles.size(), std::size_t(3));
    QCOMPARE(tiles[0].colBegin, 0);
    QCOMPARE(tiles[0].colEnd, 4);
    QCOMPARE(tiles[1].colBegin, 4);
    QCOMPARE(tiles[1].colEnd, 8);
    QCOMPARE(tiles[2].colBegin, 8);
    QCOMPARE(tiles[2].colEnd, 10);
    // Scene y is -z: the rect spans from the field's top row down to y0.
    QCOMPARE(tiles[0].sceneRect.top(), -(300.0 + 7 * 50.0));
    QCOMPARE(tiles[0].sceneRect.bottom(), -300.0);
    QCOMPARE(tiles[0].sceneRect.left(), -700.0);
    QCOMPARE(tiles[0].sceneRect.right(), -500.0);
    QCOMPARE(tiles[1].sceneRect.left(), tiles[0].sceneRect.right());
    QCOMPARE(tiles[2].sceneRect.left(), tiles[1].sceneRect.right());
    QCOMPARE(tiles[2].sceneRect.right(), -700.0 + 10 * 50.0);
    // A width below one is treated as one column per tile.
    QCOMPARE(gapFieldTiles(field, 0).size(), std::size_t(10));
}

void TestFiberMapGapField::transformIsExactDistanceToNearestSeedPixel()
{
    // pitch 0 and acrossWeight 0: u == x and no fold, so each cell reads the
    // distance transform directly. The extent's left edge is chosen so
    // every output cell centre lands on an integer raster column (no
    // interpolation), which makes the field the exact Euclidean distance
    // from the cell centre to the nearest seed pixel centre - checked
    // against brute force to float precision.
    const double cell = 100.0;
    const double saturation = 8000.0;
    const double halo = saturation + 2.0 * cell;
    const double firstSeedX = 5000.0;
    GlobalResult layout = makeLayout(4000.0, 4000.0, 0.0, 0.0, 20000.0, 0.0, 10000.0);
    addDot(layout, firstSeedX, 4321.0);
    unsigned int state = 99u;
    const auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<double>(state >> 8) / static_cast<double>(1u << 24);
    };
    for (int i = 0; i < 30; ++i) {
        addDot(layout, firstSeedX + next() * 14000.0, next() * 10000.0);
    }
    GapFieldParams params;
    params.cellVx = cell;
    params.saturationVx = saturation;
    params.acrossWeight = 0.0;
    const GapField field = buildGapField(layout, params);
    QVERIFY(!field.folded);
    const double uRaster0 = firstSeedX - halo;
    // (x0 - uRaster0) / cell must be an integer for the no-interpolation claim.
    QCOMPARE(std::fmod(layout.x0Vx - uRaster0, cell), 0.0);
    std::vector<QPointF> seedCentres;
    for (const GlobalPlacedFiber& placed : layout.fibers) {
        const QPointF& p = placed.fiber.runs.front().points.front();
        seedCentres.emplace_back(
            uRaster0 + (std::floor((p.x() - uRaster0) / cell) + 0.5) * cell,
            layout.yMinVx + (std::floor((p.y() - layout.yMinVx) / cell) + 0.5) * cell);
    }
    int checked = 0;
    for (int row = 0; row < field.rows; ++row) {
        const double y = layout.yMinVx + (row + 0.5) * cell;
        for (int col = 0; col < field.cols; ++col) {
            const double x = layout.x0Vx + (col + 0.5) * cell;
            double best = std::numeric_limits<double>::infinity();
            for (const QPointF& seed : seedCentres) {
                best = std::min(best, std::hypot(x - seed.x(), y - seed.y()));
            }
            const float got = field.at(row, col);
            if (best >= saturation) {
                QCOMPARE(got, static_cast<float>(saturation));
                continue;
            }
            QVERIFY2(std::abs(got - best) <= 0.01,
                     qPrintable(QStringLiteral("row %1 col %2: field %3 exact %4")
                                    .arg(row).arg(col).arg(got, 0, 'f', 4).arg(best, 0, 'f', 4)));
            ++checked;
        }
    }
    QVERIFY(checked > 10000);
}

void TestFiberMapGapField::degenerateGridsStillMeasure()
{
    // One row: the extent is thinner than a cell. Distances are horizontal.
    {
        GlobalResult layout = makeLayout(4000.0, 4000.0, 0.0, 0.0, 20000.0, 0.0, 40.0);
        addDot(layout, 5000.0, 20.0);
        GapFieldParams params;
        params.cellVx = 100.0;
        params.saturationVx = 8000.0;
        params.acrossWeight = 0.0;
        const GapField field = buildGapField(layout, params);
        QCOMPARE(field.rows, 1);
        QCOMPARE(field.cols, 200);
        // Cell centres sit on integer raster columns here too (see the
        // exactness test): 7050 vs the seed pixel centre 5050.
        QVERIFY(std::abs(sampleAt(field, 7050.0, 20.0) - 2000.0) <= 0.01);
        QVERIFY(std::abs(sampleAt(field, 2050.0, 20.0) - 3000.0) <= 0.01);
    }
    // One column: the extent is narrower than a cell. Distances are vertical.
    {
        // The seed at x = 0 puts the one cell's centre (x = 50) on an integer
        // raster column, so the reading is exact rather than interpolated.
        GlobalResult layout = makeLayout(4000.0, 4000.0, 0.0, 0.0, 40.0, 0.0, 20000.0);
        addDot(layout, 0.0, 5000.0);
        GapFieldParams params;
        params.cellVx = 100.0;
        params.saturationVx = 8000.0;
        params.acrossWeight = 0.0;
        const GapField field = buildGapField(layout, params);
        QCOMPARE(field.cols, 1);
        QCOMPARE(field.rows, 200);
        QVERIFY(std::abs(sampleAt(field, 20.0, 7050.0) - 2000.0) <= 0.01);
        QVERIFY(std::abs(sampleAt(field, 20.0, 2050.0) - 3000.0) <= 0.01);
    }
}

void TestFiberMapGapField::resamplesOntoSheetDistance()
{
    // Without a pitch the sheet distance is the map's x: the field comes back
    // as it was (at a negative origin too, where ceil of the rounded width
    // could otherwise add a column), and an empty field stays empty.
    {
        GlobalResult layout = makeLayout(1000.0, 1000.0, 0.0, -9167.0, 10833.0, 0.0, 1000.0);
        addVertical(layout, 5000.0, 0.0, 1000.0);
        GapFieldParams params;
        params.cellVx = 100.0;
        params.saturationVx = 3000.0;
        const GapField source = buildGapField(layout, params);
        const GapField same = resampledToSheetDistance(source, sheetModelOf(layout));
        QCOMPARE(same.cols, source.cols);
        QCOMPARE(same.rows, source.rows);
        QCOMPARE(same.x0Vx, source.x0Vx);
        QCOMPARE(same.cellVx, source.cellVx);
        QVERIFY(same.distanceVx == source.distanceVx);
        QVERIFY(resampledToSheetDistance(GapField{}, sheetModelOf(layout)).empty());
    }
    // rRef 1000, radius0 1000, pitch 200: winding W sits at x = W * 2pi*1000
    // on the map and at sheet distance 2pi * (1000 W + 100 W^2). The
    // re-gridded field spans the extent's sheet distance in the same cell,
    // and reads the source's in-sheet distance at each cell's own sheet
    // position, to within the source's resolution there.
    GlobalResult layout =
        makeLayout(1000.0, 1000.0, 200.0, 0.0, 3.0 * kTwoPi * 1000.0, 0.0, 400.0);
    const double xSeed = 2.0 * kTwoPi * 1000.0;
    addVertical(layout, xSeed, 0.0, 400.0);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 3000.0;
    params.acrossWeight = 0.0;
    const GapField source = buildGapField(layout, params);
    QVERIFY(!source.empty());
    const vc3d::fiber_map::SheetModel model = sheetModelOf(layout);
    const GapField field = resampledToSheetDistance(source, model);
    QCOMPARE(field.cellVx, source.cellVx);
    QCOMPARE(field.rows, source.rows);
    QCOMPARE(field.y0Vx, source.y0Vx);
    QCOMPARE(field.saturationVx, source.saturationVx);
    QCOMPARE(field.folded, source.folded);
    QCOMPARE(field.seedFiberCount, source.seedFiberCount);
    const double u0 = sheetU(layout, layout.x0Vx);
    const double u1 = sheetU(layout, layout.x1Vx);
    QCOMPARE(field.x0Vx, u0);
    QCOMPARE(field.cols, static_cast<int>(std::ceil((u1 - u0) / field.cellVx)));
    QVERIFY(field.cols > source.cols);
    // A source cell at winding 2 is 1.4 output cells of sheet distance wide,
    // and the source itself is exact to a cell: allow three cells.
    const double uSeed = sheetU(layout, xSeed);
    const double slack = 3.0 * params.cellVx;
    QVERIFY2(std::abs(sampleAt(field, uSeed, 200.0)) <= slack,
             qPrintable(QString::number(sampleAt(field, uSeed, 200.0))));
    for (const double offset : {-1500.0, -600.0, 600.0, 1500.0}) {
        const float value = sampleAt(field, uSeed + offset, 200.0);
        QVERIFY2(std::abs(value - std::abs(offset)) <= slack,
                 qPrintable(QStringLiteral("%1 at %2").arg(value).arg(offset)));
    }
    // Every output cell reads some source cell: nothing is left outside.
    QVERIFY(std::none_of(field.distanceVx.begin(), field.distanceVx.end(),
                         [](float v) { return std::isnan(v); }));
    // The rule cell by cell on this expanding fixture too: footprints here
    // are mostly empty (the output cell lies inside one source column and
    // reads the column its centre falls in), and the last output column
    // reaches past the source's edge and reads the edge column.
    {
        bool several = false;
        for (int j = 0; j < field.cols; ++j) {
            for (int i = 0; i < field.rows; ++i) {
                QCOMPARE(field.at(i, j), expectedRegridded(source, model, field, i, j, &several));
            }
        }
        QVERIFY(!several);
        const double lastCentre = field.x0Vx + (static_cast<double>(field.cols) - 0.5) * field.cellVx;
        const double lastX = vc3d::fiber_map::sheetXForDistanceMonotoneVx(model, lastCentre);
        QVERIFY(lastX > source.x0Vx + static_cast<double>(source.cols - 1) * source.cellVx);
        for (int i = 0; i < field.rows; ++i) {
            QCOMPARE(field.at(i, field.cols - 1), source.at(i, source.cols - 1));
        }
    }

    // A steep compression must not lose a fiber: rRef 4000 against a radius
    // of 300 at winding 0 squeezes about thirteen source columns into one
    // output column, and the seed's zero line is one source column wide.
    // The output's minimum is the source's (zero), at the seed's own sheet
    // position.
    {
        GlobalResult steep = makeLayout(4000.0, 300.0, 10.0, 0.0, 20800.0, 0.0, 600.0);
        addVertical(steep, 2600.0, 0.0, 600.0);
        GapFieldParams one;
        one.cellVx = 208.0;
        one.saturationVx = 4056.0;
        one.acrossWeight = 0.0;
        const GapField src = buildGapField(steep, one);
        QCOMPARE(finiteMinimum(src), 0.0f);
        const vc3d::fiber_map::SheetModel steepModel = sheetModelOf(steep);
        const GapField dst = resampledToSheetDistance(src, steepModel);
        QVERIFY(dst.cols * 10 < src.cols);
        QCOMPARE(finiteMinimum(dst), 0.0f);
        QCOMPARE(sampleAt(dst, sheetU(steep, 2600.0), 300.0), 0.0f);
        for (int j = 0; j < dst.cols; ++j) {
            for (int i = 0; i < dst.rows; ++i) {
                QCOMPARE(dst.at(i, j), expectedRegridded(src, steepModel, dst, i, j, nullptr));
            }
        }
    }
    // The selection rule, pinned cell by cell on a field whose rows differ,
    // including columns the source's out-of-domain NaN covers. rRef 1000,
    // radius0 500, pitch 200: the floor is winding -2.5, inside the extent,
    // so the leftmost source columns are NaN and stay NaN where they land.
    {
        GlobalResult sloped =
            makeLayout(1000.0, 500.0, 200.0, -20000.0, kTwoPi * 1000.0, 0.0, 600.0);
        addDot(sloped, 100.0, 100.0);
        addDot(sloped, 3000.0, 500.0);
        GapFieldParams two;
        two.cellVx = 100.0;
        two.saturationVx = 3000.0;
        two.acrossWeight = 0.0;
        const GapField src = buildGapField(sloped, two);
        const vc3d::fiber_map::SheetModel slopedModel = sheetModelOf(sloped);
        const double xFloor = vc3d::fiber_map::sheetDomainFloorXVx(slopedModel);
        QVERIFY(xFloor > sloped.x0Vx && xFloor < sloped.x1Vx);
        QVERIFY(std::isnan(sampleAt(src, sloped.x0Vx + 50.0, 300.0)));
        const GapField dst = resampledToSheetDistance(src, slopedModel);
        QCOMPARE(dst.rows, src.rows);
        QVERIFY(dst.rows > 1);
        QVERIFY(dst.cols < src.cols);  // mean radius below rRef: compressed
        bool sawNaN = false;
        bool sawDifferentRows = false;
        bool sawFootprintOfSeveral = false;
        for (int j = 0; j < dst.cols; ++j) {
            const auto expectedAt = [&](int i) {
                return expectedRegridded(src, slopedModel, dst, i, j, &sawFootprintOfSeveral);
            };
            for (int i = 0; i < dst.rows; ++i) {
                const float expected = expectedAt(i);
                const float actual = dst.at(i, j);
                if (std::isnan(expected)) {
                    QVERIFY2(std::isnan(actual), qPrintable(QStringLiteral("%1,%2").arg(i).arg(j)));
                    sawNaN = true;
                } else {
                    QCOMPARE(actual, expected);
                }
            }
            if (dst.rows > 1 && !std::isnan(dst.at(0, j)) && dst.at(0, j) != dst.at(dst.rows - 1, j)) {
                sawDifferentRows = true;
            }
        }
        QVERIFY(sawNaN);
        QVERIFY(sawDifferentRows);
        QVERIFY(sawFootprintOfSeveral);
        QCOMPARE(finiteMinimum(dst), finiteMinimum(src));
        // The tiles cover the re-gridded field edge to edge, image row 0 at
        // the field's top row, as for any field.
        const std::vector<GapFieldTile> tiles = gapFieldTiles(dst, 7);
        QVERIFY(tiles.size() > 1);
        QCOMPARE(tiles.front().sceneRect.left(), dst.x0Vx);
        QCOMPARE(tiles.back().sceneRect.right(),
                 dst.x0Vx + static_cast<double>(dst.cols) * dst.cellVx);
        QCOMPARE(tiles.back().colEnd, dst.cols);
    }
}

void TestFiberMapGapField::resampledGridStaysUnderBudget()
{
    // The reviewer's case: a model whose mean radius is well above rRef, so
    // re-gridding multiplies the columns. buildGapField counts those columns
    // against maxCells, so the cell coarsens until the re-gridded field
    // fits too, and re-gridding then allocates within the budget.
    GlobalResult layout = makeLayout(4000.0, 4000.0, 4000.0, 0.0, 20000.0, 0.0, 10000.0);
    addVertical(layout, 5000.0, 0.0, 10000.0);
    GapFieldParams params;
    params.cellVx = 100.0;
    params.saturationVx = 8000.0;
    params.maxCells = 6000;
    const GapField source = buildGapField(layout, params);
    QVERIFY(source.cellCoarsened);
    QVERIFY(source.distanceVx.size() <= params.maxCells);
    const GapField field = resampledToSheetDistance(source, sheetModelOf(layout));
    QVERIFY2(field.distanceVx.size() <= params.maxCells,
             qPrintable(QString::number(field.distanceVx.size())));
    QVERIFY(field.cols > source.cols);
    QCOMPARE(field.rows, source.rows);
    QCOMPARE(field.cellVx, source.cellVx);
    // The same layout with no pitch coarsens only for its own grid: the
    // re-gridding term never binds when it is the identity.
    layout.sheetPitchVx = 0.0;
    const GapField flat = buildGapField(layout, params);
    QCOMPARE(flat.cellVx, 200.0);
    QCOMPARE(flat.cols, 100);
    // The identity term is the grid's own column count, not a ceil of the
    // rounded sheet-width quotient: at this negative origin the quotient
    // rounds to 100.00000000000001, and exactly 1000 cells must fit 1000.
    {
        GlobalResult tight = makeLayout(1000.0, 1000.0, 0.0, -9167.0, 833.0, 0.0, 1000.0);
        addVertical(tight, 0.0, 0.0, 1000.0);
        GapFieldParams exact;
        exact.cellVx = 100.0;
        exact.saturationVx = 1000.0;
        exact.maxCells = 1000;
        const GapField fits = buildGapField(tight, exact);
        QVERIFY(!fits.cellCoarsened);
        QCOMPARE(fits.cols, 100);
        QCOMPARE(fits.rows, 10);
    }
}

QTEST_APPLESS_MAIN(TestFiberMapGapField)
#include "test_fiber_map_gap_field.moc"
