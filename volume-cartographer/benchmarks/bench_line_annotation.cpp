// The line annotation window on real fibers: for each fiber, the views VC3D's
// Line Annotation builds (vc::lasagna::buildLineViewSurfaces: the bottom strip,
// the side strip, and the current/side cut planes) shown in four annotation
// viewers, then stepped along the fiber the way scrolling the line position
// does: the cuts follow LineAnnotationDialog::updatePlaneSurface /
// updateSidePlaneSurface and the strip cameras recentre on the position.
// Fibers are VC3D fiber JSON files (line_points in level-0 voxels), the longest
// `count` of the case's fiber folder.

#include "bench_session.hpp"

#include "CState.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace vcbench;

namespace {

cv::Vec3f normalized(const cv::Vec3f& v)
{
    const float n = float(cv::norm(v));
    return n > 1e-6f ? v / n : cv::Vec3f(NAN, NAN, NAN);
}

bool finite(const cv::Vec3f& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

struct Views {
    vc::lasagna::LineViewSurfaces surfaces;
    vc::lasagna::LineModel line;
    std::shared_ptr<PlaneSurface> cut = std::make_shared<PlaneSurface>();
    std::shared_ptr<PlaneSurface> sideCut = std::make_shared<PlaneSurface>();
};

// The cut planes and strip cameras at fractional line position `position`.
void moveTo(Views& views, double position, std::vector<Viewer*>& viewers)
{
    const auto& pts = views.line.points;
    const auto last = pts.size() - 1;
    const auto i0 = std::min(last, std::size_t(std::floor(position)));
    const auto i1 = std::min(last, i0 + 1);
    const double t = position - double(i0);
    const cv::Vec3f origin(pts[i0].position * (1 - t) + pts[i1].position * t);
    const auto a = std::min(i0, last - 1);
    const cv::Vec3f tangent = normalized(cv::Vec3f(pts[a + 1].position - pts[a].position));
    // The cut's up: the line's display frame normal, else its transported up vector.
    cv::Vec3f up(pts[i0].sampledNormal.normal);
    if (!finite(normalized(up)) && i0 < views.surfaces.lineUpVectors.size())
        up = views.surfaces.lineUpVectors[i0];
    if (finite(origin) && finite(tangent) && finite(normalized(up))) {
        views.cut->setFromNormalAndUp(origin, tangent, up);
        const auto sideNormal = normalized(up.cross(tangent));
        if (finite(sideNormal))
            views.sideCut->setFromNormalAndUp(origin, sideNormal, tangent);
    }
    const double column = views.surfaces.stripPositionMap.originalPositionToStripGridColumn(position);
    for (auto* strip : {views.surfaces.lineSurface.get(), views.surfaces.lineSideSlice.get()}) {
        if (!strip || !std::isfinite(column))
            continue;
        const auto uv = strip->gridToSurface({column, double(strip->rawPointsPtr()->rows) / 2});
        for (auto* viewer : viewers)
            if (viewer->get().currentSurface() == strip)
                viewer->get().centerOnSurfacePoint(cv::Vec2f(float(uv[0]), float(uv[1])));
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        Fixture fixture(argc, argv, "line_annotation");
        if (fixture.listOnly())
            return 0;
        Session session(argc, argv, fixture);
        const auto fibers = fixture.loadFibers(fixture.integer("count", 4));
        const int steps = fixture.integer("steps", 24);
        const int passes = fixture.integer("passes", 2);
        const int stripW = session.viewSize(fixture.integer("strip_width", 1600));
        const int stripH = session.viewSize(fixture.integer("strip_height", 320));
        const int cutSize = session.viewSize(fixture.integer("cut_size", 640));
        const double timeoutMs = fixture.num("timeout_ms", 120000);
        const auto& dump = fixture.dumpDir();
        fixture.rejectUnused();
        auto report = fixture.makeReport();

        const std::string stripName = "line strip", sideName = "line side strip", cutName = "line cut",
                          sideCutName = "line side cut";
        std::vector<std::unique_ptr<Viewer>> owned;
        owned.push_back(std::make_unique<Viewer>(session, stripName, true, stripW, stripH));
        owned.push_back(std::make_unique<Viewer>(session, sideName, true, stripW, stripH));
        owned.push_back(std::make_unique<Viewer>(session, cutName, true, cutSize, cutSize));
        owned.push_back(std::make_unique<Viewer>(session, sideCutName, true, cutSize, cutSize));
        std::vector<Viewer*> viewers;
        for (auto& v : owned)
            viewers.push_back(v.get());
        session.showVolume();
        waitSettled(session, viewers, Clock::now(), timeoutMs);

        std::string names;
        for (int pass = 0; pass < passes; ++pass) {
            const std::string tag = pass == 0 ? "first " : "repeat ";
            for (const auto& fiber : fibers) {
                if (pass == 0)
                    names += (names.empty() ? "" : ",") + fiber.name + "(" + std::to_string(fiber.points.size()) + ")";
                // Open: build the views, register them, show the middle of the fiber.
                auto start = Clock::now();
                Views views;
                views.line = vc::lasagna::lineModelForInspection(fiber.points);
                vc::lasagna::LineViewConfig config;
                config.buildLineZSlices = false;
                views.surfaces = vc::lasagna::buildLineViewSurfaces(views.line, config);
                report.add(tag + "build line views", "ms", msSince(start));
                session.state().setSurface(stripName, views.surfaces.lineSurface);
                session.state().setSurface(sideName, views.surfaces.lineSideSlice);
                const double middle = double(fiber.points.size() - 1) / 2;
                moveTo(views, middle, viewers);
                session.state().setSurface(cutName, views.cut);
                session.state().setSurface(sideCutName, views.sideCut);
                record(report, tag + "open fiber (4 viewers)", waitSettled(session, viewers, start, timeoutMs));
                if (!dump.empty() && pass == 0) {
                    std::filesystem::create_directories(dump);
                    for (auto* v : viewers)
                        v->save(std::filesystem::path(dump) /
                                (session.pipelineName() + "-" + fiber.name + "-" + v->surfaceName() + ".png"));
                }

                // Scroll along the whole fiber.
                for (int s = 0; s < steps; ++s) {
                    const double position = double(fiber.points.size() - 1) * s / std::max(1, steps - 1);
                    start = Clock::now();
                    moveTo(views, position, viewers);
                    session.state().setSurface(cutName, views.cut, false, true);
                    session.state().setSurface(sideCutName, views.sideCut, false, true);
                    record(report, tag + "step along fiber (4 viewers)",
                           waitSettled(session, viewers, start, timeoutMs));
                }
            }
        }

        session.note("fibers", names);
        session.noteCaches();
        report.setConfig(session.config());
        report.finish(fixture.jsonPath());
        owned.clear();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_line_annotation: %s\n", e.what());
        return 1;
    }
}
