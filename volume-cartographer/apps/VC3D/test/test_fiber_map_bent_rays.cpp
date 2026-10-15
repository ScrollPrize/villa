#include "FiberMapBentRays.hpp"

#include <QtTest/QtTest>

#include <cmath>
#include <cstring>
#include <limits>

using namespace vc3d::fiber_map::bent;

namespace
{

constexpr double kPi = 3.14159265358979323846;

// The umbilicus is the z axis.
cv::Vec3d radialAt(const cv::Vec3d& p)
{
    const double r = std::hypot(p[0], p[1]);
    return r > 0.0 ? cv::Vec3d(p[0] / r, p[1] / r, 0.0) : cv::Vec3d(1.0, 0.0, 0.0);
}

UmbilicusFrame frame()
{
    UmbilicusFrame f;
    f.radialUnit = radialAt;
    f.radius = [](const cv::Vec3d& p) { return std::hypot(p[0], p[1]); };
    f.theta = [](const cv::Vec3d& p) { return std::atan2(p[1], p[0]); };
    // Exact along a segment: the horizontal offset is linear in the
    // parameter, its squared length a quadratic.
    f.minRadiusAlong = [](const cv::Vec3d& a, const cv::Vec3d& b) {
        const cv::Vec3d o0(a[0], a[1], 0.0);
        const cv::Vec3d d(b[0] - a[0], b[1] - a[1], 0.0);
        const double dd = d.dot(d);
        const double s = dd > 0.0 ? std::clamp(-o0.dot(d) / dd, 0.0, 1.0) : 0.0;
        const cv::Vec3d o = o0 + d * s;
        return std::min({std::sqrt(o.dot(o)), std::hypot(a[0], a[1]), std::hypot(b[0], b[1])});
    };
    return f;
}

// A cylindrical scroll: the sheet normal is radial everywhere.
class CylinderField final : public SheetNormalField {
public:
    [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
    {
        return radialAt(p);
    }
    [[nodiscard]] std::string identity() const override { return "cylinder"; }
};

// A shelf: between heights z0 and z1 the sheet is level (normal along z),
// outside it is cylindrical; the normal rotates linearly over a transition
// band so the field is continuous. `twist` makes the rotation continue to
// -e_r above instead of returning to +e_r: a fold, where the material
// normal above points back at the umbilicus.
class ShelfField final : public SheetNormalField {
public:
    ShelfField(double z0, double z1, double band, bool twist = false)
        : z0_(z0), z1_(z1), band_(band), twist_(twist)
    {
    }
    [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
    {
        const cv::Vec3d er = radialAt(p);
        const cv::Vec3d ez(0.0, 0.0, 1.0);
        if (!twist_) {
            double level = 0.0;
            if (p[2] >= z0_ && p[2] <= z1_) {
                level = 1.0;
            } else if (p[2] > z1_ && p[2] < z1_ + band_) {
                level = 1.0 - (p[2] - z1_) / band_;
            } else if (p[2] < z0_ && p[2] > z0_ - band_) {
                level = 1.0 - (z0_ - p[2]) / band_;
            }
            const double angle = level * kPi / 2.0;
            return er * std::cos(angle) + ez * std::sin(angle);
        }
        const double lo = z0_ - band_;
        const double hi = z1_ + band_;
        if (p[2] <= lo || p[2] >= hi) {
            return er;
        }
        const double angle = kPi * (p[2] - lo) / (hi - lo);
        return er * std::cos(angle) + ez * std::sin(angle);
    }
    [[nodiscard]] std::string identity() const override { return twist_ ? "twist" : "shelf"; }
private:
    double z0_;
    double z1_;
    double band_;
    bool twist_;
};

// A field with a hole: nullopt inside a slab of z.
class HoleField final : public SheetNormalField {
public:
    HoleField(double z0, double z1) : z0_(z0), z1_(z1) {}
    [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
    {
        if (p[2] >= z0_ && p[2] <= z1_) {
            return std::nullopt;
        }
        return radialAt(p);
    }
    [[nodiscard]] std::string identity() const override { return "hole"; }
private:
    double z0_;
    double z1_;
};

// An arc at radius r climbing from z0 to z1 over angles a0..a1.
std::vector<cv::Vec3d> climbingArc(double r, double z0, double z1, double a0, double a1, int count)
{
    std::vector<cv::Vec3d> out;
    for (int i = 0; i < count; ++i) {
        const double f = static_cast<double>(i) / (count - 1);
        const double a = a0 + (a1 - a0) * f;
        out.emplace_back(r * std::cos(a), r * std::sin(a), z0 + (z1 - z0) * f);
    }
    return out;
}

// A short polyline crossing the curtain the way a V fiber crosses an H: it
// runs radially (across the H) through the point `dz` above H sample `i`.
std::vector<cv::Vec3d> radialCrosser(const std::vector<cv::Vec3d>& h, std::size_t i, double dz,
                                     double half = 60.0)
{
    const cv::Vec3d er = radialAt(h[i]);
    const cv::Vec3d base = h[i] + cv::Vec3d(0.0, 0.0, dz);
    std::vector<cv::Vec3d> out;
    for (int k = 0; k <= 6; ++k) {
        out.push_back(base + er * (-half + 2.0 * half * k / 6.0));
    }
    return out;
}

// The H fiber of most tests: an arc at radius 1000 climbing from z 300 to
// 700 across the shelf [450, 550] (band 50), so its ends are well
// conditioned and its middle is a level run.
std::vector<cv::Vec3d> shelfH()
{
    return climbingArc(1000.0, 300.0, 700.0, 0.0, 0.6, 201);
}

BentRayParams testParams()
{
    BentRayParams p;
    p.stepVx = 4.0;
    p.maxLengthVx = 200.0;
    p.spacingVx = 6.0;
    return p;
}

constexpr double kMinRadius = 50.0;

BentCurtain curtainOf(const SheetNormalField& field, const std::vector<cv::Vec3d>& h,
                      const BentRayParams& params)
{
    const auto profile = conditioningProfile(field, h, frame());
    const auto orientation = orientFiber(h, profile, frame(), params);
    return traceCurtain(field, h, orientation, frame(), params, kMinRadius);
}

// The bit pattern of a double, for bit-for-bit comparisons (QCOMPARE on
// doubles is fuzzy).
uint64_t bits(double value)
{
    uint64_t out = 0;
    std::memcpy(&out, &value, sizeof out);
    return out;
}

std::size_t crossingCount(const std::vector<CurtainHit>& hits)
{
    std::size_t count = 0;
    for (const CurtainHit& hit : hits) {
        count += (!hit.touch && !hit.tangential) ? 1 : 0;
    }
    return count;
}

} // namespace

class TestFiberMapBentRays : public QObject {
    Q_OBJECT

private slots:
    void symmetricStartsAreReversalInvariant()
    {
        auto points = climbingArc(1000.0, 300.0, 700.0, 0.0, 0.6, 201);
        const auto forward = symmetricStarts(points, 0, points.size() - 1, 20.0);
        std::reverse(points.begin(), points.end());
        auto backward = symmetricStarts(points, 0, points.size() - 1, 20.0);
        for (auto& i : backward) {
            i = points.size() - 1 - i;
        }
        std::sort(backward.begin(), backward.end());
        QCOMPARE(forward, backward);
        QVERIFY(forward.size() > 2);
        QCOMPARE(forward.front(), std::size_t{0});
        QCOMPARE(forward.back(), points.size() - 1);
    }

    void profileReadsEverySample()
    {
        const auto h = shelfH();
        const BentRayParams params = testParams();
        CylinderField cylinder;
        const auto flat = conditioningProfile(cylinder, h, frame());
        QCOMPARE(flat.value.size(), h.size());
        for (const double v : flat.value) {
            QVERIFY(std::abs(v - 1.0) < 1e-9);
        }
        QVERIFY(illConditionedRuns(illConditionedSamples(flat, params.conditioningGate)).empty());
        ShelfField shelf(450.0, 550.0, 50.0);
        const auto level = conditioningProfile(shelf, h, frame());
        const auto runs = illConditionedRuns(illConditionedSamples(level, params.conditioningGate));
        QCOMPARE(runs.size(), std::size_t{1});
        QVERIFY(h[runs[0].first][2] > 400.0 && h[runs[0].first][2] < 460.0);
        QVERIFY(h[runs[0].second][2] > 540.0 && h[runs[0].second][2] < 600.0);
        // A one-sample dip is seen: every sample is read.
        class DipField final : public SheetNormalField {
        public:
            [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
            {
                if (std::abs(p[2] - 500.0) < 1.5) {
                    return cv::Vec3d(0.0, 0.0, 1.0);
                }
                return radialAt(p);
            }
            [[nodiscard]] std::string identity() const override { return "dip"; }
        };
        DipField dip;
        const auto dipped = conditioningProfile(dip, h, frame());
        QVERIFY(!illConditionedRuns(illConditionedSamples(dipped, params.conditioningGate)).empty());
        HoleField hole(400.0, 600.0);
        const auto holed = conditioningProfile(hole, h, frame());
        bool sawNaN = false;
        for (const double v : holed.value) {
            sawNaN = sawNaN || std::isnan(v);
        }
        QVERIFY(sawNaN);
        QVERIFY(illConditionedRuns(illConditionedSamples(holed, params.conditioningGate)).empty());
    }

    void orientationIsAnchoredByTheWholeStretch()
    {
        const auto h = shelfH();
        const BentRayParams params = testParams();
        ShelfField shelf(450.0, 550.0, 50.0);
        const auto profile = conditioningProfile(shelf, h, frame());
        const auto orientation = orientFiber(h, profile, frame(), params);
        QCOMPARE(orientation.stretches.size(), std::size_t{1});
        QCOMPARE(orientation.stretches[0].status, RunStatus::Oriented);
        QVERIFY(orientation.stretches[0].disagreeWeight == 0.0);
        QCOMPARE(orientation.runs.size(), std::size_t{1});
        QCOMPARE(orientation.runs[0].status, RunStatus::Oriented);
        // On the shelf the oriented normal is +z: the rotation from +e_r
        // below arrives at +z.
        for (std::size_t i = orientation.runs[0].firstSample; i <= orientation.runs[0].lastSample; ++i) {
            QVERIFY(orientation.normal[i].has_value());
            QVERIFY((*orientation.normal[i])[2] > 0.5);
        }
        // Well-conditioned samples carry +e_r.
        QVERIFY(orientation.normal[0]->dot(radialAt(h[0])) > 0.9);
    }

    void orientationFromOneEndSuffices()
    {
        const auto h = climbingArc(1000.0, 300.0, 500.0, 0.0, 0.3, 101);
        const BentRayParams params = testParams();
        ShelfField shelf(450.0, 550.0, 50.0);
        const auto profile = conditioningProfile(shelf, h, frame());
        const auto orientation = orientFiber(h, profile, frame(), params);
        QCOMPARE(orientation.runs.size(), std::size_t{1});
        QCOMPARE(orientation.runs[0].status, RunStatus::Oriented);
        QVERIFY((*orientation.normal[h.size() - 1])[2] > 0.5);
    }

    // On the fold the fiber's upper half runs on the returned limb: its
    // transported axis disagrees with +e_r there while the lower half's
    // agrees. The stretch is cut into two sections of opposite vote sign,
    // each anchored by its own unanimous vote, and the level run between
    // them is contested: nothing is read there without a witness.
    void foldedSheetIsContested()
    {
        const auto h = shelfH();
        const BentRayParams params = testParams();
        ShelfField twist(450.0, 550.0, 50.0, true);
        const auto profile = conditioningProfile(twist, h, frame());
        const auto orientation = orientFiber(h, profile, frame(), params);
        QCOMPARE(orientation.stretches.size(), std::size_t{3});
        std::vector<const OrientedStretch*> byStart;
        for (const auto& s : orientation.stretches) {
            byStart.push_back(&s);
        }
        std::sort(byStart.begin(), byStart.end(),
                  [](const OrientedStretch* a, const OrientedStretch* b) {
                      return a->firstSample < b->firstSample;
                  });
        QCOMPARE(byStart[0]->status, RunStatus::Oriented);
        QCOMPARE(byStart[1]->status, RunStatus::Unoriented);
        QCOMPARE(byStart[2]->status, RunStatus::Oriented);
        // Each voting section is unanimous, the two have opposite signs
        // (which one the transport agrees with depends on the canonical
        // end it started from), and both sections' normals read outward.
        for (const OrientedStretch* s : {byStart[0], byStart[2]}) {
            QVERIFY((s->agreeWeight > 0.0) != (s->disagreeWeight > 0.0));
            for (std::size_t k = s->firstSample; k <= s->lastSample; ++k) {
                QVERIFY(orientation.normal[k].has_value());
                QVERIFY((*orientation.normal[k]).dot(frame().radialUnit(h[k])) > -1e-9);
            }
        }
        QCOMPARE(byStart[0]->provisionalSign, -byStart[2]->provisionalSign);
        QCOMPARE(orientation.runs.size(), std::size_t{1});
        QCOMPARE(orientation.runs[0].status, RunStatus::Unoriented);
        // Traced all the same, with the status on the stretch: a link
        // witness may still anchor it at assembly; without one it is
        // withheld there.
        const BentCurtain curtain = traceCurtain(twist, h, orientation, frame(), params, kMinRadius);
        QCOMPARE(curtain.stretches.size(), std::size_t{1});
        QCOMPARE(curtain.stretches[0].status, RunStatus::Unoriented);
        QVERIFY(!curtain.stretches[0].rays.empty());
        QCOMPARE(curtain.unorientedRunCount, 1);
    }

    // A mostly radial field with a short level run keeps a unanimous vote
    // from the well-conditioned samples on either end.
    void radialStretchVotesAcrossALevelRun()
    {
        class RadialShelf final : public SheetNormalField {
        public:
            [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
            {
                // Level between 700 and 760, radial elsewhere (axis sign is
                // immaterial; the transport carries the material sign).
                const cv::Vec3d er = radialAt(p);
                const cv::Vec3d ez(0.0, 0.0, 1.0);
                double level = 0.0;
                if (p[2] >= 700.0 && p[2] <= 760.0) {
                    level = 1.0;
                } else if (p[2] > 760.0 && p[2] < 800.0) {
                    level = 1.0 - (p[2] - 760.0) / 40.0;
                } else if (p[2] < 700.0 && p[2] > 660.0) {
                    level = 1.0 - (700.0 - p[2]) / 40.0;
                }
                const double angle = level * kPi / 2.0;
                return er * std::cos(angle) + ez * std::sin(angle);
            }
            [[nodiscard]] std::string identity() const override { return "limb"; }
        };
        RadialShelf field;
        const auto h = climbingArc(1000.0, 620.0, 1000.0, 0.0, 0.6, 191);
        const BentRayParams params = testParams();
        const auto profile = conditioningProfile(field, h, frame());
        const auto orientation = orientFiber(h, profile, frame(), params);
        QCOMPARE(orientation.stretches.size(), std::size_t{1});
        QCOMPARE(orientation.stretches[0].status, RunStatus::Oriented);
        QVERIFY(orientation.stretches[0].disagreeWeight == 0.0);
    }

    void wholeFiberOnTheShelfIsUnsupported()
    {
        const auto h = climbingArc(1000.0, 480.0, 520.0, 0.0, 0.3, 101);
        const BentRayParams params = testParams();
        ShelfField shelf(450.0, 550.0, 50.0);
        const auto profile = conditioningProfile(shelf, h, frame());
        const auto orientation = orientFiber(h, profile, frame(), params);
        QCOMPARE(orientation.runs.size(), std::size_t{1});
        QCOMPARE(orientation.runs[0].status, RunStatus::Unsupported);
        const BentCurtain curtain = traceCurtain(shelf, h, orientation, frame(), params, kMinRadius);
        QCOMPARE(curtain.stretches.size(), std::size_t{1});
        QCOMPARE(curtain.stretches[0].status, RunStatus::Unsupported);
        QVERIFY(!curtain.stretches[0].rays.empty());
        QCOMPARE(curtain.unsupportedRunCount, 1);
        // Its readings exist, for a link witness to anchor.
        const std::size_t i = h.size() / 2;
        QCOMPARE(intersectCurtain(curtain, radialCrosser(h, i, 30.0)).size(), std::size_t{1});
    }

    // Rays from the shelf run climb (outward) and descend (inward); a
    // radial crosser 30 vx higher is read on the outward side, one 30 vx
    // lower on the inward side, each at its true arclength, once.
    void shelfRaysReadStackedFibers()
    {
        const auto h = shelfH();
        const BentRayParams params = testParams();
        ShelfField shelf(450.0, 550.0, 50.0);
        const BentCurtain curtain = curtainOf(shelf, h, params);
        QCOMPARE(curtain.stretches.size(), std::size_t{1});
        const BentStretch& stretch = curtain.stretches[0];
        QVERIFY(stretch.rays.size() >= 4);
        for (const BentRay& ray : stretch.rays) {
            QCOMPARE(ray.points.size(), ray.s.size());
            QCOMPARE(ray.points.size(), ray.theta.size());
            QCOMPARE(ray.s.front(), 0.0);
        }
        std::size_t checked = 0;
        for (std::size_t i = stretch.firstSample + 5; i + 5 <= stretch.lastSample; i += 7) {
            if (h[i][2] < 480.0 || h[i][2] > 520.0) {
                continue;
            }
            const auto above = intersectCurtain(curtain, radialCrosser(h, i, 30.0));
            const auto below = intersectCurtain(curtain, radialCrosser(h, i, -30.0));
            QCOMPARE(above.size(), std::size_t{1});
            QCOMPARE(below.size(), std::size_t{1});
            QVERIFY(!above[0].touch && !above[0].tangential);
            QCOMPARE(above[0].side, 1);
            QVERIFY(std::abs(above[0].s - 30.0) < 4.0);
            QVERIFY(above[0].transversality > 0.9);
            QVERIFY(above[0].seedA < above[0].seedB);
            QVERIFY(above[0].across >= 0.0 && above[0].across <= 1.0);
            QCOMPARE(below[0].side, -1);
            QVERIFY(std::abs(below[0].s - 30.0) < 4.0);
            ++checked;
        }
        QVERIFY(checked >= 3);
    }

    void cylinderHasNoCurtain()
    {
        CylinderField cylinder;
        const BentCurtain curtain = curtainOf(cylinder, shelfH(), testParams());
        QVERIFY(curtain.stretches.empty());
        QCOMPARE(curtain.unorientedRunCount, 0);
        QCOMPARE(curtain.unsupportedRunCount, 0);
    }

    // Every crossing of a strip is reported (the solver decides which are
    // one passage); a second polyline further along the rays is read on
    // its own.
    void allCrossingsReportedNoOcclusion()
    {
        const auto h = shelfH();
        const BentRayParams params = testParams();
        ShelfField shelf(450.0, 550.0, 50.0);
        const BentCurtain curtain = curtainOf(shelf, h, params);
        const BentStretch& stretch = curtain.stretches[0];
        const std::size_t i = stretch.firstSample + (stretch.lastSample - stretch.firstSample) / 2;
        std::vector<cv::Vec3d> doubling = radialCrosser(h, i, 20.0);
        auto back = radialCrosser(h, i, 60.0);
        std::reverse(back.begin(), back.end());
        doubling.insert(doubling.end(), back.begin(), back.end());
        const auto hits = intersectCurtain(curtain, doubling);
        QCOMPARE(crossingCount(hits), std::size_t{2});
        QVERIFY(std::abs(hits[0].s - 20.0) < 4.0);
        QVERIFY(std::abs(hits[1].s - 60.0) < 4.0);
        const auto near = intersectCurtain(curtain, radialCrosser(h, i, 20.0));
        const auto far = intersectCurtain(curtain, radialCrosser(h, i, 40.0));
        QCOMPARE(near.size(), std::size_t{1});
        QCOMPARE(far.size(), std::size_t{1});
        QVERIFY(std::abs(far.front().s - 40.0) < 4.0);
    }

    // A polyline that comes up to the curtain at a vertex and turns back is
    // a touch; one that passes through a vertex on the curtain is a
    // crossing, with the smaller of its segments' transversalities, the
    // same in either polyline order.
    void vertexTouchAndVertexCrossing()
    {
        const auto h = shelfH();
        const BentRayParams params = testParams();
        ShelfField shelf(450.0, 550.0, 50.0);
        const BentCurtain curtain = curtainOf(shelf, h, params);
        const BentStretch& stretch = curtain.stretches[0];
        const std::size_t a = stretch.starts[stretch.starts.size() / 2];
        const std::size_t b = stretch.starts[stretch.starts.size() / 2 + 1];
        const cv::Vec3d on = 0.5 * (h[a] + h[b]) + cv::Vec3d(0.0, 0.0, 20.0);
        const cv::Vec3d outward = radialAt(on) * 30.0;
        const cv::Vec3d alongH = (h[b] - h[a]) * (1.0 / std::sqrt((h[b] - h[a]).dot(h[b] - h[a])));
        const std::vector<cv::Vec3d> touching = {on + outward + cv::Vec3d(0.0, 0.0, -10.0), on,
                                                 on + outward + cv::Vec3d(0.0, 0.0, 10.0)};
        const auto touches = intersectCurtain(curtain, touching);
        QVERIFY(!touches.empty());
        for (const CurtainHit& hit : touches) {
            QVERIFY(hit.touch);
        }
        // Crossing with asymmetric incident segments: one steep (radial),
        // one shallow (mostly along the strip).
        std::vector<cv::Vec3d> crossing = {on - outward, on, on + outward * 0.1 + alongH * 40.0};
        const auto forward = intersectCurtain(curtain, crossing);
        std::reverse(crossing.begin(), crossing.end());
        const auto backward = intersectCurtain(curtain, crossing);
        QCOMPARE(crossingCount(forward), std::size_t{1});
        QCOMPARE(crossingCount(backward), std::size_t{1});
        const CurtainHit* f = nullptr;
        const CurtainHit* g = nullptr;
        for (const CurtainHit& hit : forward) {
            if (!hit.touch && !hit.tangential) {
                f = &hit;
            }
        }
        for (const CurtainHit& hit : backward) {
            if (!hit.touch && !hit.tangential) {
                g = &hit;
            }
        }
        QVERIFY(f && g);
        QVERIFY(std::abs(f->transversality - g->transversality) < 1e-9);
        QVERIFY(f->transversality < 0.5);
        const cv::Vec3d d = f->point - g->point;
        QVERIFY(std::sqrt(d.dot(d)) < 1e-6);
    }

    void coplanarSegmentIsTangential()
    {
        const auto h = shelfH();
        const BentRayParams params = testParams();
        ShelfField shelf(450.0, 550.0, 50.0);
        const BentCurtain curtain = curtainOf(shelf, h, params);
        const BentStretch& stretch = curtain.stretches[0];
        const std::size_t a = stretch.starts[stretch.starts.size() / 2];
        const std::size_t b = stretch.starts[stretch.starts.size() / 2 + 1];
        // Along the strip's own surface: the segment from A's first point to
        // B's second point lies in the first quad's plane (a vertical
        // ribbon over the H).
        const BentRay* rayA = nullptr;
        const BentRay* rayB = nullptr;
        for (const BentRay& ray : stretch.rays) {
            if (ray.side == 1 && ray.startSample == a) {
                rayA = &ray;
            }
            if (ray.side == 1 && ray.startSample == b) {
                rayB = &ray;
            }
        }
        QVERIFY(rayA && rayB);
        const std::vector<cv::Vec3d> inPlane = {rayA->points[0], rayB->points[1]};
        const auto hits = intersectCurtain(curtain, inPlane);
        QVERIFY(!hits.empty());
        bool tangential = false;
        for (const CurtainHit& hit : hits) {
            tangential = tangential || hit.tangential;
            QVERIFY(hit.touch || hit.tangential);
        }
        QVERIFY(tangential);
    }

    // The quad's triangulation is the shorter diagonal, so swapping which
    // ray is A does not move the surface.
    void triangulationIsOwnerInvariant()
    {
        BentRay a;
        a.side = 1;
        a.startSample = 0;
        a.points = {cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(0.0, 1.0, 0.0)};
        a.s = {0.0, 1.0};
        a.theta = {0.0, 0.0};
        BentRay b = a;
        b.startSample = 1;
        b.points = {cv::Vec3d(1.0, 0.0, 0.0), cv::Vec3d(1.0, 1.0, 1.0)};
        BentRay none;
        none.side = -1;
        const auto curtainWith = [&](const BentRay& first, const BentRay& second) {
            BentCurtain curtain;
            BentStretch stretch;
            stretch.rays = {first, none, second, none};
            curtain.stretches.push_back(stretch);
            return curtain;
        };
        const std::vector<cv::Vec3d> vertical = {cv::Vec3d(0.25, 0.5, -1.0), cv::Vec3d(0.25, 0.5, 2.0)};
        const auto ab = intersectCurtain(curtainWith(a, b), vertical);
        const auto ba = intersectCurtain(curtainWith(b, a), vertical);
        QCOMPARE(ab.size(), std::size_t{1});
        QCOMPARE(ba.size(), std::size_t{1});
        const cv::Vec3d d = ab[0].point - ba[0].point;
        QVERIFY(std::sqrt(d.dot(d)) < 1e-9);
        QVERIFY(std::abs(ab[0].transversality - ba[0].transversality) < 1e-9);
        QVERIFY(std::abs(ab[0].s - ba[0].s) < 1e-9);
    }

    // A ray stops before a hole: its last point is supported.
    void holeEndsTheRaySupported()
    {
        class ShelfWithHole final : public SheetNormalField {
        public:
            [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
            {
                if (p[2] > 560.0 && p[2] < 580.0) {
                    return std::nullopt;
                }
                return shelf_.axis(p);
            }
            [[nodiscard]] std::string identity() const override { return "shelf-hole"; }
        private:
            ShelfField shelf_{450.0, 550.0, 50.0};
        };
        ShelfWithHole field;
        const BentCurtain curtain = curtainOf(field, shelfH(), testParams());
        QVERIFY(!curtain.stretches.empty());
        for (const BentRay& ray : curtain.stretches[0].rays) {
            if (ray.side != 1) {
                continue;
            }
            for (const cv::Vec3d& p : ray.points) {
                QVERIFY(field.axis(p).has_value());
                QVERIFY(p[2] <= 560.0 + 1e-9);
            }
        }
    }

    void nearAxisCutsTheRay()
    {
        const auto h = climbingArc(80.0, 300.0, 700.0, 0.0, 0.6, 201);
        ShelfField shelf(450.0, 550.0, 50.0);
        const BentCurtain curtain = curtainOf(shelf, h, testParams());
        for (const BentStretch& stretch : curtain.stretches) {
            for (const BentRay& ray : stretch.rays) {
                for (const cv::Vec3d& p : ray.points) {
                    QVERIFY(std::hypot(p[0], p[1]) >= kMinRadius - 1e-9);
                }
            }
        }
    }

    void thetaIsAccumulatedAlongTheRay()
    {
        class SpokeField final : public SheetNormalField {
        public:
            [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
            {
                const cv::Vec3d er = radialAt(p);
                const cv::Vec3d et(-er[1], er[0], 0.0);
                double level = 0.0;
                if (p[2] >= 450.0 && p[2] <= 550.0) {
                    level = 1.0;
                } else if (p[2] > 550.0 && p[2] < 600.0) {
                    level = 1.0 - (p[2] - 550.0) / 50.0;
                } else if (p[2] < 450.0 && p[2] > 400.0) {
                    level = 1.0 - (450.0 - p[2]) / 50.0;
                }
                const double angle = level * kPi / 2.0;
                return er * std::cos(angle) + et * std::sin(angle);
            }
            [[nodiscard]] std::string identity() const override { return "spoke"; }
        };
        SpokeField spoke;
        // The SIGNED angle about the umbilicus the retained points sweep,
        // delta by delta (each shifted into (-pi, pi]), in both halves of
        // the shelf: at angle 0..0.6 and, rotated half a turn, across the
        // branch cut at +-pi, where the unwrapped sum must not jump.
        const auto sweepOf = [](const BentRay& ray) {
            double sum = 0.0;
            for (std::size_t k = 1; k < ray.points.size(); ++k) {
                double delta = std::atan2(ray.points[k][1], ray.points[k][0]) -
                               std::atan2(ray.points[k - 1][1], ray.points[k - 1][0]);
                if (delta > kPi) {
                    delta -= 2.0 * kPi;
                } else if (delta <= -kPi) {
                    delta += 2.0 * kPi;
                }
                sum += delta;
            }
            return sum;
        };
        for (const bool rotated : {false, true}) {
            std::vector<cv::Vec3d> h = shelfH();
            if (rotated) {
                for (cv::Vec3d& p : h) {
                    p = cv::Vec3d(-p[0], -p[1], p[2]);
                }
            }
            const BentCurtain curtain = curtainOf(spoke, h, testParams());
            QVERIFY(!curtain.stretches.empty());
            bool sawSweep = false;
            for (const BentRay& ray : curtain.stretches[0].rays) {
                if (ray.points.size() < 2) {
                    continue;
                }
                QCOMPARE(ray.theta.size(), ray.points.size());
                QCOMPARE(ray.theta.front(), 0.0);
                for (std::size_t k = 1; k < ray.points.size(); ++k) {
                    BentRay prefix = ray;
                    prefix.points.resize(k + 1);
                    QVERIFY(std::abs(ray.theta[k] - sweepOf(prefix)) < 1e-9);
                }
                // The field turns the ray tangentially: the sweep is about
                // the ray length over the radius (1000) and has the sign of
                // the tangential direction the field points to.
                const double expected = ray.s.back() / 1000.0;
                QVERIFY(std::abs(std::abs(ray.theta.back()) - expected) < 0.2 * expected + 1e-6);
                QVERIFY(std::abs(ray.theta.back()) < 1.0);
                sawSweep = sawSweep || std::abs(ray.theta.back()) > 0.1;
            }
            QVERIFY(sawSweep);
        }
    }


    // A ray that could not be traced breaks the curtain: no strip joins
    // its two neighbours across it.
    void deadRayBreaksTheCurtain()
    {
        const auto ray = [](double x, std::size_t start, int side, bool dead) {
            BentRay r;
            r.startSample = start;
            r.side = side;
            if (dead) {
                r.points = {cv::Vec3d(x, 0.0, 0.0)};
                r.s = {0.0};
                r.theta = {0.0};
                return r;
            }
            r.points = {cv::Vec3d(x, 0.0, 0.0), cv::Vec3d(x, 0.0, 8.0), cv::Vec3d(x, 0.0, 16.0)};
            r.s = {0.0, 8.0, 16.0};
            r.theta = {0.0, 0.0, 0.0};
            return r;
        };
        BentCurtain curtain;
        BentStretch stretch;
        stretch.rays = {ray(1000.0, 0, 1, false), ray(1000.0, 0, -1, true),
                        ray(1008.0, 1, 1, true), ray(1008.0, 1, -1, true),
                        ray(1016.0, 2, 1, false), ray(1016.0, 2, -1, true)};
        curtain.stretches.push_back(stretch);
        const std::vector<cv::Vec3d> through = {cv::Vec3d(1008.0, -5.0, 4.0), cv::Vec3d(1008.0, 5.0, 4.0)};
        QVERIFY(intersectCurtain(curtain, through).empty());
        // With the middle ray alive the same polyline is read.
        stretch.rays[2] = ray(1008.0, 1, 1, false);
        curtain.stretches[0] = stretch;
        QCOMPARE(intersectCurtain(curtain, through).size(), std::size_t{1});
    }

    // A segment crossing both triangles of a warped quad yields both hits,
    // the same ones whichever ray is A.
    void warpedQuadBothTrianglesOwnerInvariant()
    {
        BentRay a;
        a.side = 1;
        a.startSample = 0;
        a.points = {cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(0.0, 1.0, 0.0)};
        a.s = {0.0, 1.0};
        a.theta = {0.0, 0.0};
        BentRay b = a;
        b.startSample = 1;
        b.points = {cv::Vec3d(1.0, 0.0, 0.0), cv::Vec3d(1.0, 1.0, 1.0)};
        BentRay none;
        none.side = -1;
        const auto curtainWith = [&](const BentRay& first, const BentRay& second) {
            BentCurtain curtain;
            BentStretch stretch;
            stretch.rays = {first, none, second, none};
            curtain.stretches.push_back(stretch);
            return curtain;
        };
        const std::vector<cv::Vec3d> seg = {cv::Vec3d(0.0, 0.0, -0.05), cv::Vec3d(1.0, 1.0, 0.15)};
        const auto ab = intersectCurtain(curtainWith(a, b), seg);
        const auto ba = intersectCurtain(curtainWith(b, a), seg);
        QCOMPARE(ab.size(), std::size_t{2});
        QCOMPARE(ba.size(), std::size_t{2});
        QVERIFY(std::abs(ab[0].t - ab[1].t) > 1e-3);
        for (std::size_t i = 0; i < ab.size(); ++i) {
            const cv::Vec3d d = ab[i].point - ba[i].point;
            QVERIFY(std::sqrt(d.dot(d)) < 1e-9);
            QVERIFY(std::abs(ab[i].transversality - ba[i].transversality) < 1e-9);
            QVERIFY(std::abs(ab[i].t - ba[i].t) < 1e-9);
        }
    }

    // Repeated samples at a vertex on the curtain are one vertex whose
    // incident directions come from the nearest distinct samples: a
    // transversal crossing, in both orders.
    void repeatedVertexIsOneCrossing()
    {
        const auto h = shelfH();
        ShelfField shelf(450.0, 550.0, 50.0);
        const BentCurtain curtain = curtainOf(shelf, h, testParams());
        const BentStretch& stretch = curtain.stretches[0];
        const std::size_t a = stretch.starts[stretch.starts.size() / 2];
        const std::size_t b = stretch.starts[stretch.starts.size() / 2 + 1];
        const cv::Vec3d on = 0.5 * (h[a] + h[b]) + cv::Vec3d(0.0, 0.0, 20.0);
        const cv::Vec3d outward = radialAt(on) * 30.0;
        std::vector<cv::Vec3d> crossing = {on - outward, on, on, on, on + outward};
        const auto forward = intersectCurtain(curtain, crossing);
        std::reverse(crossing.begin(), crossing.end());
        const auto backward = intersectCurtain(curtain, crossing);
        QCOMPARE(crossingCount(forward), std::size_t{1});
        QCOMPARE(crossingCount(backward), std::size_t{1});
        for (const auto* list : {&forward, &backward}) {
            for (const CurtainHit& hit : *list) {
                if (!hit.touch && !hit.tangential) {
                    QVERIFY(hit.transversality > 0.9);
                }
            }
        }
    }

    // Distinct passages of a polyline through one point are distinct
    // records (their translates may differ).
    void revisitedPointKeepsBothPassages()
    {
        const auto h = shelfH();
        ShelfField shelf(450.0, 550.0, 50.0);
        const BentCurtain curtain = curtainOf(shelf, h, testParams());
        const BentStretch& stretch = curtain.stretches[0];
        const std::size_t i = stretch.firstSample + (stretch.lastSample - stretch.firstSample) / 2;
        // Out through the curtain at +20, around, and back in through the
        // very same point.
        const auto out = radialCrosser(h, i, 20.0);
        std::vector<cv::Vec3d> v = out;
        v.emplace_back(out.back() + cv::Vec3d(0.0, 0.0, 15.0));
        v.emplace_back(out.front() + cv::Vec3d(0.0, 0.0, 15.0));
        v.insert(v.end(), out.begin(), out.end());
        const auto hits = intersectCurtain(curtain, v);
        // Out at +20, back in at +35 on the way round, out again at +20
        // through the same point: three crossings, two of them at one point
        // on different segments.
        QCOMPARE(crossingCount(hits), std::size_t{3});
        std::size_t atTwenty = 0;
        for (const CurtainHit& hit : hits) {
            if (!hit.touch && !hit.tangential && std::abs(hit.s - 20.0) < 4.0) {
                ++atTwenty;
            }
        }
        QCOMPARE(atTwenty, std::size_t{2});
    }

    // Ray starts are the same physical samples in either order, also with
    // irregular spacing at exact stride boundaries.
    void symmetricStartsSurviveIrregularSpacing()
    {
        std::vector<cv::Vec3d> points = {cv::Vec3d(1000.0, 0.0, 0.0)};
        const double inc[][3] = {{0.0, 0.2, 0.0}, {0.5, 0.0, 0.0}, {0.2, 0.0, 0.0}, {0.3, 0.0, 0.0},
                                 {0.7, 0.0, 0.0}, {0.0, 0.1, 0.0}, {0.8, 0.0, 0.0}, {0.0, 0.0, 4.0},
                                 {0.5, 0.0, 0.0}, {0.0, 0.6, 0.0}, {0.0, 0.0, 0.2}, {0.0, 0.6, 0.0},
                                 {3.0, 0.0, 0.0}, {0.0, 0.0, 3.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}};
        for (const auto& d : inc) {
            points.push_back(points.back() + cv::Vec3d(d[0], d[1], d[2]));
        }
        const auto forward = symmetricStarts(points, 0, points.size() - 1, 8.0);
        std::vector<cv::Vec3d> reversed(points.rbegin(), points.rend());
        auto backward = symmetricStarts(reversed, 0, reversed.size() - 1, 8.0);
        for (auto& k : backward) {
            k = points.size() - 1 - k;
        }
        std::sort(backward.begin(), backward.end());
        QCOMPARE(forward, backward);
    }


    // A passage through a quad's diagonal is classified across the
    // surface, not a single plane: the reviewer's rays, with a V whose
    // vertex sits on the crease. Staying below both faces is a touch with no
    // confidence; ending above the far face is a crossing. Both polyline
    // orders, both ray owners.
    void creaseContactIsATouch()
    {
        for (int owner = 0; owner < 2; ++owner) {
            BentRay a;
            a.side = 1;
            a.startSample = 0;
            a.points = {cv::Vec3d(1000.0, 0.0, 0.0), cv::Vec3d(1000.0, 3.0, 0.0)};
            a.s = {0.0, 3.0};
            a.theta = {0.0, 0.0};
            BentRay b = a;
            b.startSample = 1;
            b.points = {cv::Vec3d(1003.0, 0.0, 0.0), cv::Vec3d(1004.0, 2.0, 2.0)};
            if (owner == 1) {
                std::swap(a, b);
            }
            BentRay none;
            none.side = -1;
            BentCurtain curtain;
            BentStretch stretch;
            stretch.rays = {a, none, b, none};
            curtain.stretches.push_back(stretch);
            std::vector<cv::Vec3d> below = {cv::Vec3d(1001.0, 1.0, -0.25), cv::Vec3d(1001.5, 1.5, 0.0),
                                            cv::Vec3d(1002.0, 2.0, 0.25)};
            std::vector<cv::Vec3d> through = {cv::Vec3d(1001.0, 1.0, -0.25), cv::Vec3d(1001.5, 1.5, 0.0),
                                              cv::Vec3d(1002.0, 2.0, 1.0)};
            for (int order = 0; order < 2; ++order) {
                const auto touches = intersectCurtain(curtain, below);
                QCOMPARE(touches.size(), std::size_t{1});
                QVERIFY(touches[0].touch);
                QCOMPARE(touches[0].transversality, 0.0);
                const auto crossings = intersectCurtain(curtain, through);
                QCOMPARE(crossings.size(), std::size_t{1});
                QVERIFY(!crossings[0].touch);
                QVERIFY(crossings[0].transversality > 0.1);
                QVERIFY(std::abs(crossings[0].point[2]) < 1e-9);
                std::reverse(below.begin(), below.end());
                std::reverse(through.begin(), through.end());
            }
        }
    }

    // A hit on a triangle's OUTER edge involves only that face: the
    // reviewer's rays, with a V through ray A's edge at (1000, 1.5, 0),
    // whose far endpoint happens to be nearer the quad's other triangle.
    // One crossing with transversality 1/sqrt(17), the same with a
    // collinear sample added, in both polyline orders and both ray owners.
    void outerEdgeHitUsesOnlyIncidentFaces()
    {
        for (int owner = 0; owner < 2; ++owner) {
            BentRay a;
            a.side = 1;
            a.startSample = 0;
            a.points = {cv::Vec3d(1000.0, 0.0, 0.0), cv::Vec3d(1000.0, 3.0, 0.0)};
            a.s = {0.0, 3.0};
            a.theta = {0.0, 0.0};
            BentRay b = a;
            b.startSample = 1;
            b.points = {cv::Vec3d(1003.0, 0.0, 0.0), cv::Vec3d(1004.0, 2.0, 2.0)};
            if (owner == 1) {
                std::swap(a, b);
            }
            BentRay none;
            none.side = -1;
            BentCurtain curtain;
            BentStretch stretch;
            stretch.rays = {a, none, b, none};
            curtain.stretches.push_back(stretch);
            std::vector<cv::Vec3d> coarse = {cv::Vec3d(999.0, 1.5, -0.25), cv::Vec3d(1003.0, 1.5, 0.75)};
            std::vector<cv::Vec3d> fine = {cv::Vec3d(999.0, 1.5, -0.25), cv::Vec3d(1001.0, 1.5, 0.25),
                                           cv::Vec3d(1003.0, 1.5, 0.75)};
            for (int order = 0; order < 2; ++order) {
                for (const auto& v : {coarse, fine}) {
                    const auto hits = intersectCurtain(curtain, v);
                    // The edge crossing at x = 1000 (s = 1.5) is one
                    // crossing; the segment goes on to cross the far face.
                    bool edge = false;
                    for (const CurtainHit& h : hits) {
                        if (std::abs(h.point[0] - 1000.0) < 1e-9) {
                            QVERIFY(!h.touch);
                            QVERIFY(std::abs(h.s - 1.5) < 1e-9);
                            QVERIFY(std::abs(h.transversality - 1.0 / std::sqrt(17.0)) < 1e-9);
                            QVERIFY(!edge);
                            edge = true;
                        }
                    }
                    QVERIFY(edge);
                }
                std::reverse(coarse.begin(), coarse.end());
                std::reverse(fine.begin(), fine.end());
            }
        }
    }

    // A hit on a ray shared by two strips: only the neighbouring strip's
    // triangle containing the point is incident. The reviewer's rays C, A,
    // B (strip CA planar, strip AB folded), with a V of very long segments
    // through ray A at (10000, 1.5, 0): one crossing with s = 1.5 and
    // transversality 1/sqrt(17), unchanged by a collinear sample, in both
    // polyline orders and both ray orders.
    void sharedRayHitUsesOnlyIncidentFaces()
    {
        for (int order = 0; order < 2; ++order) {
            BentRay c;
            c.side = 1;
            c.startSample = 0;
            c.points = {cv::Vec3d(9997.0, 0.0, 0.0), cv::Vec3d(9997.0, 3.0, 0.0)};
            c.s = {0.0, 3.0};
            c.theta = {0.0, 0.0};
            BentRay a = c;
            a.startSample = 1;
            a.points = {cv::Vec3d(10000.0, 0.0, 0.0), cv::Vec3d(10000.0, 3.0, 0.0)};
            BentRay b = c;
            b.startSample = 2;
            b.points = {cv::Vec3d(10003.0, 0.0, 0.0), cv::Vec3d(10004.0, 2.0, 2.0)};
            BentCurtain curtain;
            BentStretch stretch;
            stretch.rays = order == 0 ? std::vector<BentRay>{c, a, b} : std::vector<BentRay>{b, a, c};
            curtain.stretches.push_back(stretch);
            std::vector<cv::Vec3d> coarse = {cv::Vec3d(7000.0, 1.5, -750.0), cv::Vec3d(13000.0, 1.5, 750.0)};
            std::vector<cv::Vec3d> fine = {cv::Vec3d(7000.0, 1.5, -750.0), cv::Vec3d(10001.0, 1.5, 0.25),
                                           cv::Vec3d(13000.0, 1.5, 750.0)};
            for (int polyOrder = 0; polyOrder < 2; ++polyOrder) {
                for (const auto& v : {coarse, fine}) {
                    const auto hits = intersectCurtain(curtain, v);
                    int atA = 0;
                    for (const CurtainHit& h : hits) {
                        if (std::abs(h.point[0] - 10000.0) < 1e-6) {
                            QVERIFY(!h.touch);
                            QVERIFY(std::abs(h.s - 1.5) < 1e-6);
                            QVERIFY(std::abs(h.transversality - 1.0 / std::sqrt(17.0)) < 1e-9);
                            ++atA;
                        }
                    }
                    QCOMPARE(atA, 1);
                }
                std::reverse(coarse.begin(), coarse.end());
                std::reverse(fine.begin(), fine.end());
            }
        }
    }

    // The same on a row shared by two quads of one strip: the next row's
    // other (folded) triangle is not incident at the row edge.
    void sharedRowHitUsesOnlyIncidentFaces()
    {
        BentRay a;
        a.side = 1;
        a.startSample = 0;
        a.points = {cv::Vec3d(1000.0, 0.0, 0.0), cv::Vec3d(1000.0, 3.0, 0.0), cv::Vec3d(1000.0, 6.0, 0.0)};
        a.s = {0.0, 3.0, 6.0};
        a.theta = {0.0, 0.0, 0.0};
        BentRay b = a;
        b.startSample = 1;
        b.points = {cv::Vec3d(1003.0, 0.0, 0.0), cv::Vec3d(1003.0, 3.0, 0.0), cv::Vec3d(1004.0, 5.0, 2.0)};
        for (int owner = 0; owner < 2; ++owner) {
            BentCurtain curtain;
            BentStretch stretch;
            stretch.rays = owner == 0 ? std::vector<BentRay>{a, b} : std::vector<BentRay>{b, a};
            curtain.stretches.push_back(stretch);
            std::vector<cv::Vec3d> v = {cv::Vec3d(1001.5, -3000.0, -750.0), cv::Vec3d(1001.5, 3006.0, 750.0)};
            for (int order = 0; order < 2; ++order) {
                const auto hits = intersectCurtain(curtain, v);
                int atRow = 0;
                for (const CurtainHit& h : hits) {
                    if (std::abs(h.point[1] - 3.0) < 1e-6) {
                        QVERIFY(!h.touch);
                        QVERIFY(std::abs(h.s - 3.0) < 1e-6);
                        QVERIFY(std::abs(h.transversality - 750.0 / std::sqrt(3003.0 * 3003.0 + 750.0 * 750.0)) < 1e-9);
                        ++atRow;
                    }
                }
                QCOMPARE(atRow, 1);
                std::reverse(v.begin(), v.end());
            }
        }
    }

    // A hit at a mesh vertex gathers every quad meeting it, the diagonal
    // neighbour included: the reviewer's rays A, B, C (three samples each;
    // the curtain is locally z = max(x - 1000, 0) + max(y, 0)) with a V
    // through B's middle vertex, arriving below and leaving above: one
    // crossing, s = 3, transversality 1/sqrt(33) (the steepest incident
    // face), in both polyline orders and both ray orders.
    void vertexHitGathersAllIncidentQuads()
    {
        const double a = 3.0 / std::sqrt(2.0);
        for (int order = 0; order < 2; ++order) {
            BentRay A;
            A.side = 1;
            A.startSample = 0;
            A.points = {cv::Vec3d(997.0, -3.0, 0.0), cv::Vec3d(997.0, 0.0, 0.0), cv::Vec3d(997.0, a, a)};
            A.s = {0.0, 3.0, 6.0};
            A.theta = {0.0, 0.0, 0.0};
            BentRay B = A;
            B.startSample = 1;
            B.points = {cv::Vec3d(1000.0, -3.0, 0.0), cv::Vec3d(1000.0, 0.0, 0.0), cv::Vec3d(1000.0, a, a)};
            BentRay C = A;
            C.startSample = 2;
            C.points = {cv::Vec3d(1003.0, -3.0, 3.0), cv::Vec3d(1003.0, 0.0, 3.0), cv::Vec3d(1003.0, a, 3.0 + a)};
            BentCurtain curtain;
            BentStretch stretch;
            stretch.rays = order == 0 ? std::vector<BentRay>{A, B, C} : std::vector<BentRay>{C, B, A};
            curtain.stretches.push_back(stretch);
            std::vector<cv::Vec3d> v = {cv::Vec3d(999.0, -1.0, -0.5), cv::Vec3d(1000.0, 0.0, 0.0),
                                        cv::Vec3d(1001.0, 1.0, 3.0)};
            for (int polyOrder = 0; polyOrder < 2; ++polyOrder) {
                const auto hits = intersectCurtain(curtain, v);
                QCOMPARE(hits.size(), std::size_t{1});
                QVERIFY(!hits[0].touch);
                QVERIFY(std::abs(hits[0].s - 3.0) < 1e-9);
                QVERIFY(std::abs(hits[0].transversality - 1.0 / std::sqrt(33.0)) < 1e-9);
                std::reverse(v.begin(), v.end());
            }
        }
    }

    // Two crossings of one strip at the same s (the reviewer's rays A, B
    // and zigzag V): the canonical order puts the more transversal one
    // first in both polyline orders.
    void equalLengthHitsOrderByTransversality()
    {
        BentRay a;
        a.side = 1;
        a.startSample = 0;
        a.points = {cv::Vec3d(1000.0, -4.0, 0.0), cv::Vec3d(1000.0, -4.0, 8.0)};
        a.s = {0.0, 8.0};
        a.theta = {0.0, 0.0};
        BentRay b = a;
        b.startSample = 1;
        b.points = {cv::Vec3d(1000.0, 4.0, 0.0), cv::Vec3d(1000.0, 4.0, 8.0)};
        BentCurtain curtain;
        BentStretch stretch;
        stretch.rays = {a, b};
        curtain.stretches.push_back(stretch);
        std::vector<cv::Vec3d> v = {cv::Vec3d(999.0, -2.0, 4.0), cv::Vec3d(1001.0, -2.0, 4.0),
                                    cv::Vec3d(1001.0, -3.0, 4.0), cv::Vec3d(999.0, 5.0, 4.0)};
        for (int order = 0; order < 2; ++order) {
            const auto hits = intersectCurtain(curtain, v);
            QCOMPARE(hits.size(), std::size_t{2});
            QCOMPARE(hits[0].s, 4.0);
            QCOMPARE(hits[1].s, 4.0);
            QVERIFY(std::abs(hits[0].transversality - 1.0) < 1e-9);
            QVERIFY(std::abs(hits[0].point[1] + 2.0) < 1e-9);
            QVERIFY(std::abs(hits[1].transversality - 1.0 / std::sqrt(17.0)) < 1e-9);
            QVERIFY(std::abs(hits[1].point[1] - 1.0) < 1e-9);
            std::reverse(v.begin(), v.end());
        }
    }

    // Two strips of one side whose A rays start at the same point (a fiber
    // revisiting a seed position; a dead ray separates the strips) with
    // three passages: the order is by strip then s, identical in both
    // polyline orders, and the ordering key is consistent over every pair.
    void repeatedOriginStripsOrderConsistently()
    {
        BentRay a;
        a.side = 1;
        a.startSample = 0;
        a.points = {cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(0.0, 0.0, 8.0)};
        a.s = {0.0, 8.0};
        a.theta = {0.0, 0.0};
        BentRay b = a;
        b.startSample = 1;
        b.points = {cv::Vec3d(4.0, 0.0, 0.0), cv::Vec3d(4.0, 0.0, 8.0)};
        BentRay dead = a;
        dead.startSample = 2;
        dead.points.clear();
        dead.s.clear();
        dead.theta.clear();
        BentRay c = a;
        c.startSample = 3;
        c.points = {cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(0.0, 0.0, -8.0)};
        BentRay d = a;
        d.startSample = 4;
        d.points = {cv::Vec3d(5.0, 0.0, 0.0), cv::Vec3d(5.0, 0.0, -8.0)};
        BentCurtain curtain;
        BentStretch stretch;
        stretch.rays = {a, b, dead, c, d};
        curtain.stretches.push_back(stretch);
        std::vector<cv::Vec3d> v = {cv::Vec3d(2.0, -1.0, 2.0), cv::Vec3d(2.0, 1.0, 2.0),  cv::Vec3d(2.0, 1.0, -4.0),
                                    cv::Vec3d(2.0, -1.0, -4.0), cv::Vec3d(2.0, -1.0, 6.0), cv::Vec3d(2.0, 1.0, 6.0)};
        std::vector<std::vector<std::pair<std::size_t, double>>> seen;
        for (int order = 0; order < 2; ++order) {
            const auto hits = intersectCurtain(curtain, v);
            QCOMPARE(hits.size(), std::size_t{3});
            std::vector<std::pair<std::size_t, double>> key;
            for (const CurtainHit& h : hits) {
                key.emplace_back(h.rayA, h.s);
            }
            // Strip AB (its B starts at x = 4) before strip CD (x = 5),
            // each by s.
            QCOMPARE(key[0], (std::pair<std::size_t, double>{0, 2.0}));
            QCOMPARE(key[1], (std::pair<std::size_t, double>{0, 6.0}));
            QCOMPARE(key[2], (std::pair<std::size_t, double>{3, 4.0}));
            seen.push_back(key);
            std::reverse(v.begin(), v.end());
        }
        QCOMPARE(seen[0], seen[1]);
    }

    // The record order does not depend on the owner's storage order: the
    // reviewer's strips AB and CD (a dead ray between them), both crossed
    // at s = 4 by one V, listed as A, B, C, D and as D, C, B, A give the
    // same sequence of hits.
    void ownerReversalKeepsTheOrder()
    {
        const auto ray = [](std::size_t start, double x, double y) {
            BentRay r;
            r.side = 1;
            r.startSample = start;
            r.points = {cv::Vec3d(x, y, 0.0), cv::Vec3d(x, y, 8.0)};
            r.s = {0.0, 8.0};
            r.theta = {0.0, 0.0};
            return r;
        };
        const BentRay a = ray(0, 1000.0, 0.0);
        const BentRay b = ray(1, 1010.0, 0.0);
        BentRay dead = ray(2, 0.0, 0.0);
        dead.points.clear();
        dead.s.clear();
        dead.theta.clear();
        const BentRay c = ray(3, 1005.0, 1.0);
        const BentRay d = ray(4, 1006.0, 1.0);
        const std::vector<cv::Vec3d> v = {cv::Vec3d(1005.5, -1.0, 4.0), cv::Vec3d(1005.5, 2.0, 4.0)};
        std::vector<std::vector<cv::Vec3d>> orders;
        for (const auto& rays : {std::vector<BentRay>{a, b, dead, c, d}, std::vector<BentRay>{d, c, dead, b, a}}) {
            BentCurtain curtain;
            BentStretch stretch;
            stretch.rays = rays;
            curtain.stretches.push_back(stretch);
            const auto hits = intersectCurtain(curtain, v);
            QCOMPARE(hits.size(), std::size_t{2});
            std::vector<cv::Vec3d> points;
            for (const CurtainHit& h : hits) {
                QVERIFY(!h.touch);
                QCOMPARE(h.s, 4.0);
                points.push_back(h.point);
            }
            orders.push_back(points);
        }
        // Strip AB (starts from x = 1000) before strip CD (from x = 1005).
        QVERIFY(std::abs(orders[0][0][1]) < 1e-9);
        QVERIFY(std::abs(orders[0][1][1] - 1.0) < 1e-9);
        for (std::size_t i = 0; i < 2; ++i) {
            const cv::Vec3d diff = orders[0][i] - orders[1][i];
            QVERIFY(std::sqrt(diff.dot(diff)) < 1e-9);
        }
    }

    // A hit on a ray shared by two strips is owned by the strip whose
    // non-shared ray starts at the smaller point, whichever outer seed is
    // nearer the hit: the reviewer's rays A, B, C (each p, p+(0,0,8),
    // p+(0,8,8)) and a V crossing strip AB at s = 10 and ray B at s = 12.
    // Both records belong to strip AB in both ray orders.
    void sharedRayOwnershipByProvenance()
    {
        const auto ray = [](std::size_t start, const cv::Vec3d& p) {
            BentRay r;
            r.side = 1;
            r.startSample = start;
            r.points = {p, p + cv::Vec3d(0.0, 0.0, 8.0), p + cv::Vec3d(0.0, 8.0, 8.0)};
            r.s = {0.0, 8.0, 16.0};
            r.theta = {0.0, 0.0, 0.0};
            return r;
        };
        const BentRay a = ray(0, cv::Vec3d(999.0, 1.0, 0.0));
        const BentRay b = ray(1, cv::Vec3d(1000.0, 0.0, 0.0));
        const BentRay c = ray(2, cv::Vec3d(1001.0, 1.0, 0.0));
        std::vector<cv::Vec3d> v = {cv::Vec3d(999.5, 2.5, 7.0), cv::Vec3d(999.5, 2.5, 9.0), cv::Vec3d(1000.0, 4.0, 9.0),
                                    cv::Vec3d(1000.0, 4.0, 7.0)};
        for (int order = 0; order < 2; ++order) {
            const std::vector<BentRay> rays = order == 0 ? std::vector<BentRay>{a, b, c} : std::vector<BentRay>{c, b, a};
            const std::size_t indexA = order == 0 ? 0 : 2;
            const std::size_t indexB = 1;
            BentCurtain curtain;
            BentStretch stretch;
            stretch.rays = rays;
            curtain.stretches.push_back(stretch);
            for (int polyOrder = 0; polyOrder < 2; ++polyOrder) {
                const auto hits = intersectCurtain(curtain, v);
                QCOMPARE(hits.size(), std::size_t{2});
                for (const CurtainHit& h : hits) {
                    QVERIFY(!h.touch);
                    QVERIFY(std::abs(h.transversality - 1.0) < 1e-9);
                    QVERIFY((h.rayA == indexA && h.rayB == indexB) || (h.rayA == indexB && h.rayB == indexA));
                }
                QVERIFY(std::abs(hits[0].s - 10.0) < 1e-9);
                QVERIFY(std::abs(hits[1].s - 12.0) < 1e-9);
                std::reverse(v.begin(), v.end());
            }
        }
    }

    // Records of separate runs are in one geometric order whichever run
    // comes first in the owner's storage: the reviewer's two runs (strips
    // at x = 1000 and x = 1010, the +z curtain) crossed by one V, listed
    // in either order, give the same sequence.
    void runsOrderGeometrically()
    {
        const auto ray = [](std::size_t start, double x, double y) {
            BentRay r;
            r.side = 1;
            r.startSample = start;
            r.points = {cv::Vec3d(x, y, 0.0), cv::Vec3d(x, y, 8.0)};
            r.s = {0.0, 8.0};
            r.theta = {0.0, 0.0};
            return r;
        };
        BentStretch near;
        near.status = RunStatus::Oriented;
        near.rays = {ray(0, 1000.0, -2.0), ray(1, 1000.0, 2.0)};
        BentStretch far;
        far.status = RunStatus::Oriented;
        far.rays = {ray(3, 1010.0, -2.0), ray(4, 1010.0, 2.0)};
        std::vector<cv::Vec3d> v = {cv::Vec3d(999.0, 0.0, 4.0), cv::Vec3d(1001.0, 0.0, 4.0), cv::Vec3d(1009.0, 0.0, 4.0),
                                    cv::Vec3d(1011.0, 0.0, 4.0)};
        std::vector<std::vector<cv::Vec3d>> orders;
        for (int order = 0; order < 2; ++order) {
            BentCurtain curtain;
            curtain.stretches = order == 0 ? std::vector<BentStretch>{near, far} : std::vector<BentStretch>{far, near};
            for (int polyOrder = 0; polyOrder < 2; ++polyOrder) {
                const auto hits = intersectCurtain(curtain, v);
                QCOMPARE(hits.size(), std::size_t{2});
                std::vector<cv::Vec3d> points;
                for (const CurtainHit& h : hits) {
                    QVERIFY(!h.touch);
                    QCOMPARE(h.s, 4.0);
                    points.push_back(h.point);
                }
                // The strip at x = 1000 first.
                QVERIFY(std::abs(points[0][0] - 1000.0) < 1e-9);
                QVERIFY(std::abs(points[1][0] - 1010.0) < 1e-9);
                orders.push_back(points);
                std::reverse(v.begin(), v.end());
            }
        }
        for (std::size_t i = 1; i < orders.size(); ++i) {
            for (std::size_t k = 0; k < 2; ++k) {
                const cv::Vec3d d = orders[0][k] - orders[i][k];
                QVERIFY(std::sqrt(d.dot(d)) < 1e-9);
            }
        }
    }

    // The transport basis of a stretch the vote leaves unsupported does
    // not depend on the storage order: the reviewer's split field (axis
    // (0,-4,1)/sqrt17 for x < 1002, (0,4,1)/sqrt17 beyond, both with
    // positive z, no voters because the H lies radially) gives the same
    // provisional normal at every sample, and the same records in the same
    // order, for the H stored either way.
    void unsupportedBasisIsCanonicalUnderReversal()
    {
        class SplitField final : public SheetNormalField {
        public:
            [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
            {
                const double y = p[0] < 1002.0 ? -4.0 : 4.0;
                return cv::Vec3d(0.0, y, 1.0) * (1.0 / std::sqrt(17.0));
            }
            [[nodiscard]] std::string identity() const override { return "split"; }
        };
        const SplitField field;
        BentRayParams params = testParams();
        params.stepVx = 8.0;
        params.maxLengthVx = 8.0;
        params.spacingVx = 8.0;
        std::vector<cv::Vec3d> h = {cv::Vec3d(1000.0, 0.0, 0.0), cv::Vec3d(1001.0, 0.0, 0.0), cv::Vec3d(1003.0, 0.0, 0.0),
                                    cv::Vec3d(1004.0, 0.0, 0.0)};
        std::vector<cv::Vec3d> v = {cv::Vec3d(1002.0, -4.0, -4.0), cv::Vec3d(1002.0, -4.0, 4.0),
                                    cv::Vec3d(1002.0, 4.0, 4.0), cv::Vec3d(1002.0, 4.0, -4.0)};
        std::vector<std::optional<cv::Vec3d>> forwardNormals;
        std::vector<CurtainHit> forwardHits;
        for (int order = 0; order < 2; ++order) {
            const auto profile = conditioningProfile(field, h, frame());
            const auto orientation = orientFiber(h, profile, frame(), params);
            QCOMPARE(orientation.stretches.size(), std::size_t{1});
            QVERIFY(orientation.stretches[0].status == RunStatus::Unsupported);
            const auto curtain = traceCurtain(field, h, orientation, frame(), params, kMinRadius);
            const auto hits = intersectCurtain(curtain, v);
            QCOMPARE(hits.size(), std::size_t{2});
            if (order == 0) {
                forwardNormals = orientation.normal;
                forwardHits = hits;
            } else {
                for (std::size_t i = 0; i < h.size(); ++i) {
                    QVERIFY(forwardNormals[h.size() - 1 - i].has_value());
                    QVERIFY(orientation.normal[i].has_value());
                    const cv::Vec3d d = *forwardNormals[h.size() - 1 - i] - *orientation.normal[i];
                    QVERIFY(std::sqrt(d.dot(d)) < 1e-12);
                }
                for (std::size_t i = 0; i < hits.size(); ++i) {
                    const cv::Vec3d d = forwardHits[i].point - hits[i].point;
                    QVERIFY(std::sqrt(d.dot(d)) < 1e-9);
                    QCOMPARE(forwardHits[i].side, hits[i].side);
                    QVERIFY(std::abs(forwardHits[i].s - hits[i].s) < 1e-9);
                }
            }
            std::reverse(h.begin(), h.end());
        }
    }

    // The vote's sums are accumulated in the canonical traversal too:
    // axes (c, 0, sqrt(1 - c^2)) with c = .53, -.51, -.51, -.57, 0, 0
    // along a radial H form two sections. Their weights, status, sign and
    // every normal are the same for the H stored either way.
    void voteSumsAreCanonicalUnderReversal()
    {
        class RadialField final : public SheetNormalField {
        public:
            [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
            {
                const double c = cAt(p[0]);
                return cv::Vec3d(c, 0.0, std::sqrt(1.0 - c * c));
            }
            [[nodiscard]] std::string identity() const override { return "radial-vote"; }
            static double cAt(double x)
            {
                const double table[6] = {0.53, -0.51, -0.51, -0.57, 0.0, 0.0};
                const long i = std::lround(x - 1000.0);
                return i >= 0 && i < 6 ? table[i] : 0.0;
            }
        };
        const RadialField field;
        const BentRayParams params = testParams();
        std::vector<cv::Vec3d> h;
        for (int i = 0; i < 6; ++i) {
            h.emplace_back(1000.0 + i, 0.0, 0.0);
        }
        // Two sections (the sign changes between the first and second
        // voters), the same in either storage order: statuses, signs,
        // weights (summed in the canonical traversal) and sample bounds.
        std::vector<FiberOrientation> both;
        for (int order = 0; order < 2; ++order) {
            const auto profile = conditioningProfile(field, h, frame());
            both.push_back(orientFiber(h, profile, frame(), params));
            QCOMPARE(both.back().stretches.size(), std::size_t{2});
            std::reverse(h.begin(), h.end());
        }
        const auto sections = [&](const FiberOrientation& o, bool reversedStorage) {
            std::vector<std::tuple<std::size_t, std::size_t, int, int, double, double>> out;
            for (const OrientedStretch& s : o.stretches) {
                const std::size_t first = reversedStorage ? h.size() - 1 - s.lastSample : s.firstSample;
                const std::size_t last = reversedStorage ? h.size() - 1 - s.firstSample : s.lastSample;
                out.emplace_back(first, last, static_cast<int>(s.status), s.provisionalSign, s.agreeWeight,
                                 s.disagreeWeight);
            }
            std::sort(out.begin(), out.end());
            return out;
        };
        QVERIFY(sections(both[0], false) == sections(both[1], true));
        for (std::size_t i = 0; i < h.size(); ++i) {
            QVERIFY(both[0].normal[i].has_value());
            QVERIFY(both[1].normal[h.size() - 1 - i].has_value());
            const cv::Vec3d d = *both[0].normal[i] - *both[1].normal[h.size() - 1 - i];
            QVERIFY(std::sqrt(d.dot(d)) < 1e-12);
        }
    }

    // A step whose ends clear the cutoff but which dips inside between
    // them is rejected: the reviewer's seeds at radius 417.0002 with a +y
    // normal pass radius 416.999 one voxel in; no step survives.
    void stepDippingInsideTheCutoffIsRejected()
    {
        class PlusY final : public SheetNormalField {
        public:
            [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d&) const override
            {
                return cv::Vec3d(0.0, 1.0, 0.0);
            }
            [[nodiscard]] std::string identity() const override { return "+y"; }
        };
        const PlusY field;
        std::vector<cv::Vec3d> h = {cv::Vec3d(416.999, -1.0, -2.0), cv::Vec3d(416.999, -1.0, 2.0)};
        BentRayParams params = testParams();
        params.stepVx = 8.0;
        params.spacingVx = 1.0;
        const auto profile = conditioningProfile(field, h, frame());
        const auto orientation = orientFiber(h, profile, frame(), params);
        const BentCurtain curtain = traceCurtain(field, h, orientation, frame(), params, 417.0);
        QCOMPARE(curtain.stretches.size(), std::size_t{1});
        QVERIFY(!curtain.stretches[0].rays.empty());
        // The outward rays (+y, through radius 416.999 at y = 0) stop at
        // their seed; the inward rays (-y, radius growing) run on.
        const auto check = [&](const BentCurtain& c) {
            int outward = 0;
            for (const BentRay& ray : c.stretches[0].rays) {
                if (ray.side == 1) {
                    QCOMPARE(ray.points.size(), std::size_t{1});
                    ++outward;
                } else {
                    QVERIFY(ray.points.size() > 1);
                }
            }
            QVERIFY(outward >= 1);
        };
        check(curtain);
        // Without the exact minimum the sampled fallback catches it too.
        UmbilicusFrame sampled = frame();
        sampled.minRadiusAlong = nullptr;
        check(traceCurtain(field, h, orientation, sampled, params, 417.0));
    }

    // A ray start inside the umbilicus cutoff gets no ray (a dead ray that
    // breaks the curtain there); starts beyond it are traced.
    void seedInsideTheCutoffGetsNoRay()
    {
        ShelfField shelf(450.0, 550.0, 50.0);
        std::vector<cv::Vec3d> h;
        for (double r = 400.0; r <= 600.0 + 1e-9; r += 8.0) {
            h.emplace_back(r, 0.0, 500.0);
        }
        BentRayParams params = testParams();
        params.spacingVx = 8.0;
        const auto profile = conditioningProfile(shelf, h, frame());
        const auto orientation = orientFiber(h, profile, frame(), params);
        const double cutoff = 417.0;
        const BentCurtain curtain = traceCurtain(shelf, h, orientation, frame(), params, cutoff);
        QCOMPARE(curtain.stretches.size(), std::size_t{1});
        int dead = 0;
        int live = 0;
        for (const BentRay& ray : curtain.stretches[0].rays) {
            const double radius = std::hypot(h[ray.startSample][0], h[ray.startSample][1]);
            if (radius < cutoff) {
                QVERIFY(ray.points.empty());
                ++dead;
            } else {
                QVERIFY(ray.points.size() >= 2);
                ++live;
            }
        }
        QVERIFY(dead >= 2);
        QVERIFY(live >= 2);
    }

    // A transported basis that flips along the fiber (the field's axis
    // rotates half a turn against the radial direction over the fiber,
    // through an ill-conditioned run in the middle) cuts the stretch into
    // two sections of opposite vote sign, each with its own provisional
    // sign so both read outward, and the run between them is contested.
    void transportFlipSplitsTheStretch()
    {
        // e_r is +x to a good approximation far from the umbilicus; the axis
        // rotates from +x at x = 10000 to -x at x = 10800 in the xy plane.
        class RotatingField final : public SheetNormalField {
        public:
            [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
            {
                const double phi = std::clamp((p[0] - 10000.0) / 800.0, 0.0, 1.0) * M_PI;
                return cv::Vec3d(std::cos(phi), std::sin(phi), 0.0);
            }
            [[nodiscard]] std::string identity() const override { return "rotating"; }
        };
        const RotatingField field;
        std::vector<cv::Vec3d> h;
        for (double x = 10000.0; x <= 10800.0 + 1e-9; x += 8.0) {
            h.emplace_back(x, 0.0, 0.0);
        }
        const BentRayParams params = testParams();
        for (int order = 0; order < 2; ++order) {
            const auto profile = conditioningProfile(field, h, frame());
            const auto orientation = orientFiber(h, profile, frame(), params);
            QCOMPARE(orientation.stretches.size(), std::size_t{3});
            std::vector<const OrientedStretch*> byStart;
            for (const auto& s : orientation.stretches) {
                byStart.push_back(&s);
            }
            std::sort(byStart.begin(), byStart.end(),
                      [](const OrientedStretch* a, const OrientedStretch* b) {
                          return a->firstSample < b->firstSample;
                      });
            QCOMPARE(byStart[0]->status, RunStatus::Oriented);
            QCOMPARE(byStart[1]->status, RunStatus::Unoriented);
            QCOMPARE(byStart[2]->status, RunStatus::Oriented);
            QVERIFY(byStart[0]->agreeWeight > 0.0 || byStart[0]->disagreeWeight > 0.0);
            // The two voting sections have opposite provisional signs and
            // both normals point outward (+x).
            QCOMPARE(byStart[0]->provisionalSign, -byStart[2]->provisionalSign);
            for (const OrientedStretch* s : {byStart[0], byStart[2]}) {
                for (std::size_t k = s->firstSample; k <= s->lastSample; ++k) {
                    QVERIFY(orientation.normal[k].has_value());
                    QVERIFY((*orientation.normal[k]).dot(frame().radialUnit(h[k])) > -1e-9);
                }
            }
            QCOMPARE(orientation.runs.size(), std::size_t{1});
            QCOMPARE(orientation.runs[0].status, RunStatus::Unoriented);
            std::reverse(h.begin(), h.end());
        }
    }

    // A tangential overlap suppresses a crossing at its end only on a face
    // sharing that boundary: the reviewer's rays A, B (strip AB in the
    // plane z = 0) and C, D (strip CD in the plane x = 1000). The V lies in
    // AB until x = 1000 and crosses CD there: the CD crossing (across 0.5,
    // s = 2, transversality 1) stands.
    void overlapEndOnAnotherStripKeepsTheCrossing()
    {
        BentRay a;
        a.side = 1;
        a.startSample = 0;
        a.points = {cv::Vec3d(996.0, -2.0, 0.0), cv::Vec3d(996.0, 2.0, 0.0)};
        a.s = {0.0, 4.0};
        a.theta = {0.0, 0.0};
        BentRay b = a;
        b.startSample = 1;
        b.points = {cv::Vec3d(1000.0, -2.0, 0.0), cv::Vec3d(1000.0, 2.0, 0.0)};
        BentRay c = a;
        c.startSample = 2;
        c.points = {cv::Vec3d(1000.0, -4.0, -2.0), cv::Vec3d(1000.0, -4.0, 2.0)};
        BentRay d = a;
        d.startSample = 3;
        d.points = {cv::Vec3d(1000.0, 4.0, -2.0), cv::Vec3d(1000.0, 4.0, 2.0)};
        BentCurtain curtain;
        BentStretch stretch;
        stretch.rays = {a, b, c, d};
        curtain.stretches.push_back(stretch);
        std::vector<cv::Vec3d> v = {cv::Vec3d(995.0, 0.0, 0.0), cv::Vec3d(1005.0, 0.0, 0.0)};
        for (int order = 0; order < 2; ++order) {
            const auto hits = intersectCurtain(curtain, v);
            bool crossing = false;
            for (const CurtainHit& h : hits) {
                if (h.rayA == 2) {
                    QVERIFY(!h.touch);
                    QVERIFY(std::abs(h.across - 0.5) < 1e-9);
                    QVERIFY(std::abs(h.s - 2.0) < 1e-9);
                    QVERIFY(std::abs(h.transversality - 1.0) < 1e-9);
                    QVERIFY(!crossing);
                    crossing = true;
                } else {
                    QVERIFY(h.tangential || h.touch);
                }
            }
            QVERIFY(crossing);
            std::reverse(v.begin(), v.end());
        }
    }

    // Records at one polyline position from faces that do not share that
    // point are distinct: the reviewer's rays A, B, C, where the V passes
    // through AB's seed edge (one touch) and, at the same point, the
    // interior of strip BC (a crossing at across 0.5, s = 2).
    void samePointOnDifferentFacesStaysDistinct()
    {
        BentRay a;
        a.side = 1;
        a.startSample = 0;
        a.points = {cv::Vec3d(1000.0, -4.0, 0.0), cv::Vec3d(1000.0, -4.0, 8.0)};
        a.s = {0.0, 8.0};
        a.theta = {0.0, 0.0};
        BentRay b = a;
        b.startSample = 1;
        b.points = {cv::Vec3d(1000.0, 4.0, 0.0), cv::Vec3d(1000.0, 4.0, 8.0)};
        BentRay c = a;
        c.startSample = 2;
        c.points = {cv::Vec3d(1000.0, -4.0, -4.0), cv::Vec3d(1000.0, -4.0, 4.0)};
        BentCurtain curtain;
        BentStretch stretch;
        stretch.rays = {a, b, c};
        curtain.stretches.push_back(stretch);
        std::vector<cv::Vec3d> v = {cv::Vec3d(996.0, 0.0, 0.0), cv::Vec3d(1004.0, 0.0, 0.0)};
        for (int order = 0; order < 2; ++order) {
            const auto hits = intersectCurtain(curtain, v);
            QCOMPARE(hits.size(), std::size_t{2});
            const CurtainHit& seed = hits[0].rayA == 0 ? hits[0] : hits[1];
            const CurtainHit& inner = hits[0].rayA == 0 ? hits[1] : hits[0];
            QCOMPARE(seed.rayA, std::size_t{0});
            QVERIFY(seed.touch);
            QCOMPARE(seed.s, 0.0);
            QCOMPARE(inner.rayA, std::size_t{1});
            QVERIFY(!inner.touch);
            QVERIFY(std::abs(inner.across - 0.5) < 1e-9);
            QVERIFY(std::abs(inner.s - 2.0) < 1e-9);
            QVERIFY(std::abs(inner.transversality - 1.0) < 1e-9);
            std::reverse(v.begin(), v.end());
        }
    }


    // A polyline that passes through a vertex on the curtain, leaves,
    // wraps, and passes through the same place again: two records.
    void revisitedVertexKeepsBothPassages()
    {
        const auto h = shelfH();
        ShelfField shelf(450.0, 550.0, 50.0);
        const BentCurtain curtain = curtainOf(shelf, h, testParams());
        const BentStretch& stretch = curtain.stretches[0];
        const std::size_t a = stretch.starts[stretch.starts.size() / 2];
        const std::size_t b = stretch.starts[stretch.starts.size() / 2 + 1];
        const cv::Vec3d on = 0.5 * (h[a] + h[b]) + cv::Vec3d(0.0, 0.0, 20.0);
        const cv::Vec3d outward = radialAt(on) * 30.0;
        std::vector<cv::Vec3d> v = {on - outward, on, on + outward, on + outward + cv::Vec3d(0.0, 0.0, 15.0),
                                    on - outward + cv::Vec3d(0.0, 0.0, 15.0), on - outward, on, on + outward};
        for (int order = 0; order < 2; ++order) {
            const auto hits = intersectCurtain(curtain, v);
            std::size_t atVertex = 0;
            for (const CurtainHit& hit : hits) {
                if (!hit.touch && !hit.tangential && std::abs(hit.s - 20.0) < 4.0) {
                    ++atVertex;
                }
            }
            QCOMPARE(atVertex, std::size_t{2});
            std::reverse(v.begin(), v.end());
        }
    }

    // Equal end points: the canonical traversal is settled by the first
    // differing samples from the ends inward.
    void symmetricStartsWithEqualEnds()
    {
        std::vector<cv::Vec3d> points = {cv::Vec3d(1000.0, 0.0, 0.0), cv::Vec3d(1009.0, 0.0, 0.0),
                                         cv::Vec3d(1009.0, 2.0, 0.0), cv::Vec3d(1000.0, 0.0, 0.0)};
        const auto forward = symmetricStarts(points, 0, 3, 8.0);
        std::vector<cv::Vec3d> reversed(points.rbegin(), points.rend());
        auto backward = symmetricStarts(reversed, 0, 3, 8.0);
        for (auto& k : backward) {
            k = 3 - k;
        }
        std::sort(backward.begin(), backward.end());
        QCOMPARE(forward, backward);
    }


    // Two non-adjacent strips of one run whose curtains overlap in space
    // (an H turning back on itself): a polyline crossing both at one point
    // is two records, one per passage.
    void overlappingNonAdjacentStripsStayDistinct()
    {
        const auto ray = [](const cv::Vec3d& start, std::size_t sample) {
            BentRay r;
            r.startSample = sample;
            r.side = 1;
            for (int k = 0; k <= 4; ++k) {
                r.points.push_back(start + cv::Vec3d(0.0, 0.0, 4.0 * k));
                r.s.push_back(4.0 * k);
                r.theta.push_back(0.0);
            }
            return r;
        };
        BentRay none;
        none.side = -1;
        BentCurtain curtain;
        BentStretch stretch;
        // Strip 0 between samples 0 and 1 at z 0; strip 2 between samples
        // 4 and 5 at z 8 (a turn later), both over x = 1000, y in [-4, 4].
        stretch.rays = {ray(cv::Vec3d(1000.0, -4.0, 0.0), 0), none, ray(cv::Vec3d(1000.0, 4.0, 0.0), 1), none,
                        ray(cv::Vec3d(1200.0, 0.0, 0.0), 2), none, ray(cv::Vec3d(1200.0, 8.0, 0.0), 3), none,
                        ray(cv::Vec3d(1000.0, -4.0, 8.0), 4), none, ray(cv::Vec3d(1000.0, 4.0, 8.0), 5), none};
        curtain.stretches.push_back(stretch);
        std::vector<cv::Vec3d> v = {cv::Vec3d(996.0, 0.0, 12.0), cv::Vec3d(1004.0, 0.0, 12.0)};
        for (int order = 0; order < 2; ++order) {
            const auto hits = intersectCurtain(curtain, v);
            QCOMPARE(crossingCount(hits), std::size_t{2});
            bool saw12 = false;
            bool saw4 = false;
            for (const CurtainHit& hit : hits) {
                saw12 = saw12 || std::abs(hit.s - 12.0) < 1e-9;
                saw4 = saw4 || std::abs(hit.s - 4.0) < 1e-9;
            }
            QVERIFY(saw12 && saw4);
            std::reverse(v.begin(), v.end());
        }
    }

    // A segment lying in one face of a warped quad and leaving it across
    // the diagonal into the tilted face: a contact, not a crossing, in both
    // polyline orders and whichever ray is A.
    void coplanarSegmentLeavingAcrossACreaseIsAContact()
    {
        BentRay a;
        a.side = 1;
        a.startSample = 0;
        a.points = {cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(0.0, 1.0, 0.0)};
        a.s = {0.0, 1.0};
        a.theta = {0.0, 0.0};
        BentRay b = a;
        b.startSample = 1;
        b.points = {cv::Vec3d(1.0, 0.0, 0.0), cv::Vec3d(1.0, 1.0, 1.0)};
        BentRay none;
        none.side = -1;
        const auto curtainWith = [&](const BentRay& first, const BentRay& second) {
            BentCurtain curtain;
            BentStretch stretch;
            stretch.rays = {first, none, second, none};
            curtain.stretches.push_back(stretch);
            return curtain;
        };
        std::vector<cv::Vec3d> v = {cv::Vec3d(0.2, 0.5, 0.0), cv::Vec3d(0.8, 0.5, 0.0)};
        for (int order = 0; order < 2; ++order) {
            for (const auto& curtain : {curtainWith(a, b), curtainWith(b, a)}) {
                const auto hits = intersectCurtain(curtain, v);
                QVERIFY(!hits.empty());
                QCOMPARE(crossingCount(hits), std::size_t{0});
                for (const CurtainHit& hit : hits) {
                    QVERIFY(hit.touch || hit.tangential);
                    QCOMPARE(hit.transversality, 0.0);
                }
            }
            std::reverse(v.begin(), v.end());
        }
    }


    // The seed edge is shared by the outward and inward sides: a polyline
    // through it is one touch, not two crossings.
    void seedEdgeIsOneTouch()
    {
        const auto ray = [](const cv::Vec3d& start, std::size_t sample, int side) {
            BentRay r;
            r.startSample = sample;
            r.side = side;
            for (int k = 0; k <= 3; ++k) {
                r.points.push_back(start + cv::Vec3d(0.0, 0.0, 4.0 * k * side));
                r.s.push_back(4.0 * k);
                r.theta.push_back(0.0);
            }
            return r;
        };
        BentCurtain curtain;
        BentStretch stretch;
        stretch.rays = {ray(cv::Vec3d(1000.0, -4.0, 0.0), 0, 1), ray(cv::Vec3d(1000.0, -4.0, 0.0), 0, -1),
                        ray(cv::Vec3d(1000.0, 4.0, 0.0), 1, 1), ray(cv::Vec3d(1000.0, 4.0, 0.0), 1, -1)};
        curtain.stretches.push_back(stretch);
        std::vector<cv::Vec3d> v = {cv::Vec3d(996.0, 0.0, 0.0), cv::Vec3d(1004.0, 0.0, 0.0)};
        for (int order = 0; order < 2; ++order) {
            const auto hits = intersectCurtain(curtain, v);
            QCOMPARE(hits.size(), std::size_t{1});
            QVERIFY(hits[0].touch);
            QCOMPARE(hits[0].side, 1);
            QCOMPARE(hits[0].transversality, 0.0);
            std::reverse(v.begin(), v.end());
        }
        // Just above the edge it is a crossing of the outward side only.
        v = {cv::Vec3d(996.0, 0.0, 1.0), cv::Vec3d(1004.0, 0.0, 1.0)};
        const auto above = intersectCurtain(curtain, v);
        QCOMPARE(crossingCount(above), std::size_t{1});
        QCOMPARE(above[0].side, 1);
    }

    // A repeated owner sample seeds two identical rays: the zero-width strip
    // between them is no surface. A crossing on the shared ray next to it
    // is read from the real strip's faces alone (not turned into a touch by
    // zero-length incident edges), whichever way the owner and the crosser
    // are stored, from hand-made rays and from traced ones.
    void repeatedOwnerSampleDoesNotEraseTheCrossing()
    {
        const auto ray = [](const cv::Vec3d& start, std::size_t sample, int side) {
            BentRay r;
            r.startSample = sample;
            r.side = side;
            for (int k = 0; k <= 3; ++k) {
                r.points.push_back(start + cv::Vec3d(0.0, 0.0, 4.0 * k * side));
                r.s.push_back(4.0 * k);
                r.theta.push_back(0.0);
            }
            return r;
        };
        const auto check = [&](const BentCurtain& curtain, std::vector<cv::Vec3d> v) {
            for (int order = 0; order < 2; ++order) {
                const auto hits = intersectCurtain(curtain, v);
                QCOMPARE(crossingCount(hits), std::size_t{1});
                for (const CurtainHit& h : hits) {
                    if (h.touch || h.tangential) {
                        continue;
                    }
                    QCOMPARE(h.side, 1);
                    QVERIFY(std::abs(h.s - 4.0) < 1e-9);
                    QVERIFY(std::abs(h.transversality - 1.0) < 1e-9);
                    QVERIFY(std::abs(h.point[0] - 1000.0) < 1e-9);
                    QVERIFY(std::abs(h.point[1] - 4.0) < 1e-9);
                }
                std::reverse(v.begin(), v.end());
            }
        };
        const std::vector<cv::Vec3d> v = {cv::Vec3d(999.0, 4.0, 4.0), cv::Vec3d(1001.0, 4.0, 4.0)};
        for (int owner = 0; owner < 2; ++owner) {
            BentCurtain curtain;
            BentStretch stretch;
            std::vector<BentRay> rays = {ray(cv::Vec3d(1000.0, -4.0, 0.0), 0, 1), ray(cv::Vec3d(1000.0, -4.0, 0.0), 0, -1),
                                         ray(cv::Vec3d(1000.0, 4.0, 0.0), 1, 1), ray(cv::Vec3d(1000.0, 4.0, 0.0), 1, -1),
                                         ray(cv::Vec3d(1000.0, 4.0, 0.0), 2, 1), ray(cv::Vec3d(1000.0, 4.0, 0.0), 2, -1)};
            if (owner == 1) {
                std::reverse(rays.begin(), rays.end());
                for (BentRay& r : rays) {
                    r.startSample = 2 - r.startSample;
                }
            }
            stretch.rays = rays;
            curtain.stretches.push_back(stretch);
            check(curtain, v);
        }
        // Traced: the level field keeps every sample ill-conditioned, the
        // spacing keeps all three samples as seeds.
        class LevelField final : public SheetNormalField {
        public:
            [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d&) const override
            {
                return cv::Vec3d(0.0, 0.0, 1.0);
            }
            [[nodiscard]] std::string identity() const override { return "level"; }
        };
        const LevelField level;
        std::vector<cv::Vec3d> h = {cv::Vec3d(1000.0, -4.0, 0.0), cv::Vec3d(1000.0, 4.0, 0.0), cv::Vec3d(1000.0, 4.0, 0.0)};
        for (int owner = 0; owner < 2; ++owner) {
            const BentCurtain curtain = curtainOf(level, h, testParams());
            QCOMPARE(curtain.stretches.size(), std::size_t{1});
            QCOMPARE(curtain.stretches[0].starts.size(), std::size_t{3});
            check(curtain, v);
            std::reverse(h.begin(), h.end());
        }
    }

    // A crosser vertex exactly on the last ray of a stretch (no strip
    // beyond it): the last strip reads the crossing, from either direction.
    void vertexOnTheLastRayIsRead()
    {
        const auto ray = [](const cv::Vec3d& start, std::size_t sample, int side) {
            BentRay r;
            r.startSample = sample;
            r.side = side;
            for (int k = 0; k <= 3; ++k) {
                r.points.push_back(start + cv::Vec3d(0.0, 0.0, 4.0 * k * side));
                r.s.push_back(4.0 * k);
                r.theta.push_back(0.0);
            }
            return r;
        };
        for (int owner = 0; owner < 2; ++owner) {
            BentCurtain curtain;
            BentStretch stretch;
            stretch.rays = {ray(cv::Vec3d(1000.0, -4.0, 0.0), 0, 1), ray(cv::Vec3d(1000.0, -4.0, 0.0), 0, -1),
                            ray(cv::Vec3d(1000.0, 4.0, 0.0), 1, 1), ray(cv::Vec3d(1000.0, 4.0, 0.0), 1, -1)};
            if (owner == 1) {
                std::reverse(stretch.rays.begin(), stretch.rays.end());
                for (BentRay& r : stretch.rays) {
                    r.startSample = 1 - r.startSample;
                }
            }
            curtain.stretches.push_back(stretch);
            for (const double y : {4.0, -4.0}) {
                std::vector<cv::Vec3d> v = {cv::Vec3d(999.0, y, 4.0), cv::Vec3d(1000.0, y, 4.0),
                                            cv::Vec3d(1001.0, y, 4.0)};
                for (int order = 0; order < 2; ++order) {
                    const auto hits = intersectCurtain(curtain, v);
                    QCOMPARE(crossingCount(hits), std::size_t{1});
                    QCOMPARE(hits[0].side, 1);
                    QVERIFY(std::abs(hits[0].s - 4.0) < 1e-9);
                    std::reverse(v.begin(), v.end());
                }
            }
        }
    }

    // The reviewer's numerics (specialist S2). Endpoint membership: a hit a
    // hair past a crosser's start (canonical t ~ 1e-9 of a 7600 vx segment)
    // is a crossing in both storage directions - 1 - t rounds to the
    // endpoint band, the canonical parameter does not.
    void endpointMembershipIsCanonical()
    {
        const auto ray = [](const cv::Vec3d& start, std::size_t sample) {
            BentRay r;
            r.startSample = sample;
            r.side = 1;
            for (int k = 0; k <= 1; ++k) {
                r.points.push_back(start + cv::Vec3d(0.0, 0.0, 8.0 * k));
                r.s.push_back(8.0 * k);
                r.theta.push_back(0.0);
            }
            return r;
        };
        const double X = 100.029998779296875;
        BentCurtain curtain;
        BentStretch stretch;
        stretch.rays = {ray(cv::Vec3d(X, -4.0, 0.0), 0), ray(cv::Vec3d(X, 4.0, 0.0), 1)};
        curtain.stretches.push_back(stretch);
        std::vector<cv::Vec3d> v = {cv::Vec3d(100.02999114990234375, 0.0, 4.0), cv::Vec3d(7729.42431640625, 0.0, 4.0)};
        std::vector<uint64_t> sBits;
        for (int order = 0; order < 2; ++order) {
            const auto hits = intersectCurtain(curtain, v);
            QCOMPARE(crossingCount(hits), std::size_t{1});
            QCOMPARE(hits.size(), std::size_t{1});
            QVERIFY(!hits[0].touch);
            QVERIFY(std::abs(hits[0].s - 4.0) < 1e-9);
            QVERIFY(std::abs(hits[0].transversality - 1.0) < 1e-9);
            sBits.push_back(bits(hits[0].s));
            std::reverse(v.begin(), v.end());
        }
        QCOMPARE(sBits[0], sBits[1]);
    }

    // Incidence by the face's own scale: at z ~ 1e5 a coordinate-scaled
    // tolerance let a neighbouring strip's other triangle, passing a
    // thousandth of a voxel from a shared-ray hit, count as incident, and
    // its normal (perpendicular to the crosser) zeroed the transversality.
    void containmentIsJudgedAtTheFaceScale()
    {
        const auto ray = [](const cv::Vec3d& p0, const cv::Vec3d& p1, std::size_t sample) {
            BentRay r;
            r.startSample = sample;
            r.side = 1;
            r.points = {p0, p1};
            r.s = {0.0, 8.0};
            r.theta = {0.0, 0.0};
            return r;
        };
        BentCurtain curtain;
        BentStretch stretch;
        stretch.rays = {ray(cv::Vec3d(995.0, 0.0, 99991.796875), cv::Vec3d(995.0, 8.0, 99991.796875), 0),
                        ray(cv::Vec3d(1000.0, 0.0, 100000.0), cv::Vec3d(1000.0, 8.0, 100000.0), 1),
                        ray(cv::Vec3d(1000.0009765625, 0.0, 100000.0),
                            cv::Vec3d(1000.0009765625, 7.999999839999997, 100000.0016), 2)};
        curtain.stretches.push_back(stretch);
        std::vector<cv::Vec3d> v = {cv::Vec3d(999.0, 4.0, 100000.0), cv::Vec3d(1001.0, 4.0, 100000.0)};
        for (int order = 0; order < 2; ++order) {
            const auto hits = intersectCurtain(curtain, v);
            QCOMPARE(crossingCount(hits), std::size_t{1});
            for (const CurtainHit& h : hits) {
                if (!h.touch && !h.tangential) {
                    QVERIFY(h.transversality > 0.85);
                    QVERIFY(h.transversality < 0.86);
                }
            }
            std::reverse(v.begin(), v.end());
        }
    }

    // A sliver face (its second ray 5e-6 vx aside over 1000 vx) passes the
    // area rule; its barycentrics must not cancel to nothing, or a vertex
    // hit keeps the rounding of whichever segment found it.
    void sliverFaceBlendsCanonically()
    {
        const auto ray = [](const cv::Vec3d& start, std::size_t sample) {
            BentRay r;
            r.startSample = sample;
            r.side = 1;
            for (int k = 0; k <= 1; ++k) {
                r.points.push_back(start + cv::Vec3d(0.0, 0.0, 8.0 * k));
                r.s.push_back(8.0 * k);
                r.theta.push_back(0.0);
            }
            return r;
        };
        BentCurtain curtain;
        BentStretch stretch;
        stretch.rays = {ray(cv::Vec3d(1000.0, 0.0, 0.0), 0), ray(cv::Vec3d(1000.0, 0.000005, 1000.0), 1)};
        curtain.stretches.push_back(stretch);
        std::vector<cv::Vec3d> v = {cv::Vec3d(900.0, -0.01, 200.0), cv::Vec3d(1000.0, 0.00000125, 253.0),
                                    cv::Vec3d(1100.0, 0.02, 300.0)};
        std::vector<std::vector<uint64_t>> terms;
        for (int order = 0; order < 2; ++order) {
            const auto hits = intersectCurtain(curtain, v);
            QVERIFY(!hits.empty());
            std::vector<uint64_t> t;
            for (const CurtainHit& h : hits) {
                t.push_back(bits(h.s));
                t.push_back(bits(h.across));
                t.push_back(bits(h.along));
                for (int c = 0; c < 3; ++c) {
                    t.push_back(bits(h.point[c]));
                }
            }
            terms.push_back(t);
            std::reverse(v.begin(), v.end());
        }
        QCOMPARE(terms[0], terms[1]);
    }

    // Probes off the surface are judged by their orthogonal foot on each
    // incident face (specialist S2 / final pass 3): the reviewer's tilted
    // rays and a V grazing the strip at ray B's start, whose probes land
    // on different faces when dropped along an axis instead - a touch,
    // not a crossing, in both storage orders.
    void offPlaneProbesUseTheOrthogonalFoot()
    {
        const auto ray = [](const cv::Vec3d& p0, const cv::Vec3d& p1, std::size_t sample) {
            BentRay r;
            r.startSample = sample;
            r.side = 1;
            r.points = {p0, p1};
            r.s = {0.0, std::sqrt(14.0)};
            r.theta = {0.0, 0.0};
            return r;
        };
        BentCurtain curtain;
        BentStretch stretch;
        stretch.rays = {ray(cv::Vec3d(0.0, 1000.0, 2.0), cv::Vec3d(-2.0, 999.0, 5.0), 0),
                        ray(cv::Vec3d(-3.0, 1001.0, -4.0), cv::Vec3d(0.0, 1000.0, -2.0), 1)};
        curtain.stretches.push_back(stretch);
        std::vector<cv::Vec3d> v = {cv::Vec3d(0.0, 1000.5, -0.2), cv::Vec3d(0.0, 1000.0, 0.0),
                                    cv::Vec3d(-0.5, 999.6, 0.0)};
        for (int order = 0; order < 2; ++order) {
            const auto hits = intersectCurtain(curtain, v);
            QCOMPARE(crossingCount(hits), std::size_t{0});
            for (const CurtainHit& h : hits) {
                QVERIFY(h.touch || h.tangential);
            }
            std::reverse(v.begin(), v.end());
        }
    }

    // The crosser running exactly along the seed edge: one contact, owned
    // by the outward side, not one per face it grazes.
    void fullSeedEdgeOverlapIsOneContact()
    {
        const auto ray = [](const cv::Vec3d& start, std::size_t sample, int side) {
            BentRay r;
            r.startSample = sample;
            r.side = side;
            for (int k = 0; k <= 1; ++k) {
                r.points.push_back(start + cv::Vec3d(0.0, 0.0, 8.0 * k * side));
                r.s.push_back(8.0 * k);
                r.theta.push_back(0.0);
            }
            return r;
        };
        BentCurtain curtain;
        BentStretch stretch;
        stretch.rays = {ray(cv::Vec3d(1000.0, -4.0, 0.0), 0, 1), ray(cv::Vec3d(1000.0, -4.0, 0.0), 0, -1),
                        ray(cv::Vec3d(1000.0, 4.0, 0.0), 1, 1), ray(cv::Vec3d(1000.0, 4.0, 0.0), 1, -1)};
        curtain.stretches.push_back(stretch);
        std::vector<cv::Vec3d> v = {cv::Vec3d(1000.0, -4.0, 0.0), cv::Vec3d(1000.0, 4.0, 0.0)};
        for (int order = 0; order < 2; ++order) {
            const auto hits = intersectCurtain(curtain, v);
            QCOMPARE(hits.size(), std::size_t{1});
            QVERIFY(hits[0].tangential);
            QCOMPARE(hits[0].side, 1);
            std::reverse(v.begin(), v.end());
        }
    }

    // A segment overlapping one row of a strip, leaving it, and crossing the
    // next row's tilted face further along: the crease contact at the
    // overlap's end is a contact, the later crossing stands; with and
    // without a sample between them, both polyline and ray orders.
    void overlapThenLaterCrossingKeepsTheCrossing()
    {
        BentRay a;
        a.side = 1;
        a.startSample = 0;
        a.points = {cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(5.0, 0.0, 0.0), cv::Vec3d(8.0, 0.0, 4.0)};
        a.s = {0.0, 5.0, 10.0};
        a.theta = {0.0, 0.0, 0.0};
        BentRay b;
        b.side = 1;
        b.startSample = 1;
        b.points = {cv::Vec3d(0.0, 5.0, 0.0), cv::Vec3d(5.0, 5.0, 0.0), cv::Vec3d(9.0, 5.0, -3.0)};
        b.s = {0.0, 5.0, 10.0};
        b.theta = {0.0, 0.0, 0.0};
        BentRay none;
        none.side = -1;
        const auto curtainWith = [&](const BentRay& first, const BentRay& second) {
            BentCurtain curtain;
            BentStretch stretch;
            stretch.rays = {first, none, second, none};
            curtain.stretches.push_back(stretch);
            return curtain;
        };
        for (const bool subdivided : {false, true}) {
            std::vector<cv::Vec3d> v = {cv::Vec3d(0.5, 3.75, 0.0), cv::Vec3d(8.5, 3.75, 0.0)};
            if (subdivided) {
                v = {cv::Vec3d(0.5, 3.75, 0.0), cv::Vec3d(6.0, 3.75, 0.0), cv::Vec3d(8.5, 3.75, 0.0)};
            }
            for (int order = 0; order < 2; ++order) {
                for (const auto& curtain : {curtainWith(a, b), curtainWith(b, a)}) {
                    const auto hits = intersectCurtain(curtain, v);
                    std::size_t crossings = 0;
                    for (const CurtainHit& hit : hits) {
                        if (hit.touch || hit.tangential) {
                            continue;
                        }
                        ++crossings;
                        QVERIFY(std::abs(hit.point[0] - 85.0 / 12.0) < 1e-6);
                        QVERIFY(std::abs(hit.transversality - 3.0 / std::sqrt(50.0)) < 1e-6);
                    }
                    QCOMPARE(crossings, std::size_t{1});
                }
                std::reverse(v.begin(), v.end());
            }
        }
    }

    void polylineOrderIndependence()
    {
        const auto h = shelfH();
        ShelfField shelf(450.0, 550.0, 50.0);
        const BentCurtain curtain = curtainOf(shelf, h, testParams());
        const BentStretch& stretch = curtain.stretches[0];
        std::vector<cv::Vec3d> v;
        for (std::size_t i = stretch.firstSample + 5, k = 0; i + 5 <= stretch.lastSample; i += 9, ++k) {
            auto c = radialCrosser(h, i, 10.0 + 5.0 * k);
            if (k % 2 == 1) {
                std::reverse(c.begin(), c.end());
            }
            v.insert(v.end(), c.begin(), c.end());
        }
        const auto forward = intersectCurtain(curtain, v);
        std::reverse(v.begin(), v.end());
        const auto backward = intersectCurtain(curtain, v);
        QCOMPARE(forward.size(), backward.size());
        QVERIFY(forward.size() >= 3);
        // Bit for bit: the intersection runs on each segment in canonical
        // direction, so the hit's geometry does not carry the storage order
        // even in its rounding (the classification orders tied readings by
        // these values).
        for (std::size_t i = 0; i < forward.size(); ++i) {
            QCOMPARE(forward[i].side, backward[i].side);
            QCOMPARE(forward[i].rayA, backward[i].rayA);
            QCOMPARE(bits(forward[i].s), bits(backward[i].s));
            QCOMPARE(bits(forward[i].theta), bits(backward[i].theta));
            QCOMPARE(bits(forward[i].across), bits(backward[i].across));
            QCOMPARE(bits(forward[i].along), bits(backward[i].along));
            for (int c = 0; c < 3; ++c) {
                QCOMPARE(bits(forward[i].point[c]), bits(backward[i].point[c]));
            }
            QCOMPARE(forward[i].touch, backward[i].touch);
            QCOMPARE(forward[i].tangential, backward[i].tangential);
            QCOMPARE(bits(forward[i].transversality), bits(backward[i].transversality));
        }
    }

    // The same surface from rays in opposite order (the owner visiting A
    // then B, and B then A a turn on; or the owner stored the other way):
    // the faces are built on the rays in canonical order, so the hits of
    // the two strips are bit for bit alike in everything the classification
    // orders by (s, transversality, point), whichever way the crosser is
    // stored.
    void oppositeDirectionStripsReadAlike()
    {
        const auto ray = [](const cv::Vec3d& start, std::size_t sample) {
            BentRay r;
            r.startSample = sample;
            r.side = 1;
            for (int k = 0; k <= 3; ++k) {
                r.points.push_back(start + cv::Vec3d(0.0, 0.0, 8.0 * k));
                r.s.push_back(8.0 * k);
                r.theta.push_back(0.0);
            }
            return r;
        };
        const cv::Vec3d a(1000.0, -4.0, 0.0);
        const cv::Vec3d b(1000.0, 4.0, 0.0);
        std::vector<cv::Vec3d> v = {cv::Vec3d(999.0, -3.9, 4.2), cv::Vec3d(1001.0, -3.9, 4.2)};
        for (int order = 0; order < 2; ++order) {
            BentCurtain curtain;
            BentStretch ab;
            ab.rays = {ray(a, 0), ray(b, 1)};
            BentStretch ba;
            ba.rays = {ray(b, 2), ray(a, 3)};
            curtain.stretches = {ab, ba};
            const auto hits = intersectCurtain(curtain, v);
            QCOMPARE(crossingCount(hits), std::size_t{2});
            QCOMPARE(hits.size(), std::size_t{2});
            const CurtainHit& first = hits[0];
            const CurtainHit& second = hits[1];
            QVERIFY(first.stretch != second.stretch);
            QCOMPARE(bits(first.s), bits(second.s));
            QCOMPARE(bits(first.transversality), bits(second.transversality));
            QCOMPARE(bits(first.theta), bits(second.theta));
            QCOMPARE(bits(first.along), bits(second.along));
            for (int c = 0; c < 3; ++c) {
                QCOMPARE(bits(first.point[c]), bits(second.point[c]));
            }
            QVERIFY(std::abs(first.s - 4.2) < 1e-9);
            // Across is in each strip's own frame: 0 on its first ray.
            QVERIFY(std::abs(first.across + second.across - 1.0) < 1e-9);
            std::reverse(v.begin(), v.end());
        }
    }

    void fiberOrderIndependence()
    {
        auto h = shelfH();
        const BentRayParams params = testParams();
        ShelfField shelf(450.0, 550.0, 50.0);
        std::vector<cv::Vec3d> v;
        for (std::size_t i = 60, k = 0; i <= 140; i += 9, ++k) {
            auto c = radialCrosser(h, i, 10.0 + 5.0 * k);
            if (k % 2 == 1) {
                std::reverse(c.begin(), c.end());
            }
            v.insert(v.end(), c.begin(), c.end());
        }
        const auto forward = intersectCurtain(curtainOf(shelf, h, params), v);
        std::reverse(h.begin(), h.end());
        const auto backward = intersectCurtain(curtainOf(shelf, h, params), v);
        QCOMPARE(forward.size(), backward.size());
        QVERIFY(forward.size() >= 3);
        // The same records in the same order.
        for (std::size_t i = 0; i < forward.size(); ++i) {
            const cv::Vec3d d = forward[i].point - backward[i].point;
            QVERIFY(std::sqrt(d.dot(d)) < 1e-6);
            QVERIFY(std::abs(forward[i].s - backward[i].s) < 1e-6);
        }
        for (const CurtainHit& a : forward) {
            bool found = false;
            for (const CurtainHit& b : backward) {
                const cv::Vec3d d = a.point - b.point;
                if (a.side == b.side && std::sqrt(d.dot(d)) < 1e-6 && std::abs(a.s - b.s) < 1e-6 &&
                    std::abs(a.transversality - b.transversality) < 1e-9) {
                    found = true;
                }
            }
            if (!found) {
                qWarning("forward hit: side %d s %.4f tr %.4f point (%.4f,%.4f,%.4f) across %.3f quad %zu along %.3f touch %d tang %d",
                         a.side, a.s, a.transversality, a.point[0], a.point[1], a.point[2], a.across, a.quad, a.along, a.touch, a.tangential);
                for (const CurtainHit& b : backward) {
                    const cv::Vec3d d = a.point - b.point;
                    if (std::sqrt(d.dot(d)) < 5.0) {
                        qWarning("  near backward: side %d s %.4f tr %.4f point (%.4f,%.4f,%.4f) across %.3f quad %zu along %.3f touch %d tang %d",
                                 b.side, b.s, b.transversality, b.point[0], b.point[1], b.point[2], b.across, b.quad, b.along, b.touch, b.tangential);
                    }
                }
            }
            QVERIFY(found);
        }
    }
};

QTEST_APPLESS_MAIN(TestFiberMapBentRays)
#include "test_fiber_map_bent_rays.moc"
