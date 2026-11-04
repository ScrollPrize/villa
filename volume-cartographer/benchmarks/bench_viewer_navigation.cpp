// Volume viewers and navigation: VC3D's standard layout (XY/XZ/YZ slice planes
// through the focus point and the flattened segmentation), opened on a real
// volume and segment, then driven by the inputs a user gives: arrow-key pans,
// zoom steps, slice scrolling, focus jumps and a held-key pan burst. Each step
// is timed from the input to the first presented frame and to every viewer
// settling (final frame shown, no reads left).

#include "bench_session.hpp"

#include "CState.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QThread>

#include <cstdio>

using namespace vcbench;

namespace {

struct Layout {
    std::shared_ptr<QuadSurface> segment;
    std::vector<std::pair<std::string, std::shared_ptr<PlaneSurface>>> planes;
    std::vector<std::unique_ptr<Viewer>> viewers;  // the planes', then the segmentation's
    cv::Vec3f focus;

    std::vector<Viewer*> all() const
    {
        std::vector<Viewer*> out;
        for (const auto& v : viewers)
            out.push_back(v.get());
        return out;
    }
};

// Moves the slice planes to `point` and re-registers them, as the focus POI does.
void setPlanes(Session& session, Layout& layout, const cv::Vec3f& point)
{
    for (auto& [name, plane] : layout.planes) {
        plane->setOrigin(point);
        session.state().setSurface(name, plane, false, true);
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        Fixture fixture(argc, argv, "viewer_navigation");
        if (fixture.listOnly())
            return 0;
        Session session(argc, argv, fixture);
        const int width = session.viewSize(fixture.integer("width", 960));
        const int height = session.viewSize(fixture.integer("height", 540));
        const int passes = fixture.integer("passes", 2);
        // Pixels per level-0 voxel the viewers are zoomed to after opening (VC3D opens
        // slice views on the whole scroll).
        const float viewScale = float(fixture.num("view_scale", 1.0));
        const int panSteps = fixture.integer("pan_steps", 8);
        const int zoomSteps = fixture.integer("zoom_steps", 4);
        const int sliceSteps = fixture.integer("slice_steps", 8);
        const int jumps = fixture.integer("jumps", 8);
        const int burst = fixture.integer("burst", 60);
        const double timeoutMs = fixture.num("timeout_ms", 120000);
        const auto& dump = fixture.dumpDir();

        Layout layout;
        layout.segment = fixture.loadSegment();
        fixture.rejectUnused();
        layout.focus = surfaceCenter(*layout.segment);
        const auto jumpPoints = surfaceSamplePoints(*layout.segment, jumps);
        layout.planes = {{"xy plane", std::make_shared<PlaneSurface>(layout.focus, cv::Vec3f(0, 0, 1))},
                         {"xz plane", std::make_shared<PlaneSurface>(layout.focus, cv::Vec3f(0, 1, 0))},
                         {"yz plane", std::make_shared<PlaneSurface>(layout.focus, cv::Vec3f(1, 0, 0))}};

        auto report = fixture.makeReport();

        // Open: viewers exist, then the surfaces and the volume arrive (cold caches).
        for (const auto& [name, plane] : layout.planes)
            layout.viewers.push_back(std::make_unique<Viewer>(session, name, false, width, height));
        layout.viewers.push_back(std::make_unique<Viewer>(session, "segmentation", false, width, height));
        auto start = Clock::now();
        for (const auto& [name, plane] : layout.planes)
            session.state().setSurface(name, plane);
        session.state().setSurface("segmentation", layout.segment);
        session.showVolume();
        record(report, "open (4 viewers)", waitSettled(session, layout.all(), start, timeoutMs));

        start = Clock::now();
        for (const auto& v : layout.viewers)
            v->get().adjustZoomByFactor(viewScale / v->get().getCurrentScale());
        record(report, "zoom to view scale (4 viewers)", waitSettled(session, layout.all(), start, timeoutMs));
        session.note("view_scale", std::to_string(viewScale));
        if (!dump.empty()) {
            std::filesystem::create_directories(dump);
            for (const auto& v : layout.viewers)
                v->save(std::filesystem::path(dump) / (session.pipelineName() + "-" + v->surfaceName() + ".png"));
        }

        std::vector<float> initialScale;
        for (const auto& v : layout.viewers)
            initialScale.push_back(v->get().getCurrentScale());

        for (int pass = 0; pass < passes; ++pass) {
            const std::string tag = pass == 0 ? "first " : "repeat ";
            // Back to the starting view.
            start = Clock::now();
            setPlanes(session, layout, layout.focus);
            for (std::size_t i = 0; i < layout.viewers.size(); ++i) {
                auto& viewer = layout.viewers[i]->get();
                viewer.adjustZoomByFactor(initialScale[i] / viewer.getCurrentScale());
            }
            layout.viewers.back()->get().centerOnVolumePoint(layout.focus);
            waitSettled(session, layout.all(), start, timeoutMs);

            for (std::size_t i = 0; i < layout.viewers.size(); ++i) {
                auto* viewer = layout.viewers[i].get();
                const bool isSegment = i + 1 == layout.viewers.size();
                const std::vector<Viewer*> one{viewer};
                const auto label = tag + viewer->surfaceName();

                for (int s = 0; s < 2 * panSteps; ++s) {
                    start = Clock::now();
                    viewer->get().onKeyPress(s < panSteps ? Qt::Key_Right : Qt::Key_Left, Qt::NoModifier);
                    record(report, label + " pan", waitSettled(session, one, start, timeoutMs));
                }
                for (int s = 0; s < 2 * zoomSteps; ++s) {
                    start = Clock::now();
                    viewer->get().adjustZoomByFactor(s < zoomSteps ? 1.25f : 1.0f / 1.25f);
                    record(report, label + " zoom", waitSettled(session, one, start, timeoutMs));
                }
                for (int s = 0; s < 2 * sliceSteps; ++s) {
                    const float delta = s < sliceSteps ? 1.0f : -1.0f;
                    start = Clock::now();
                    if (isSegment) {
                        viewer->get().adjustSurfaceOffset(delta);
                    } else {
                        auto& plane = layout.planes[i].second;
                        plane->setOrigin(plane->origin() + plane->normal({0, 0, 0}) * delta);
                        session.state().setSurface(layout.planes[i].first, plane, false, true);
                    }
                    record(report, label + " slice", waitSettled(session, one, start, timeoutMs));
                }
            }

            // Focus jumps move every slice plane and recentre the segmentation view.
            for (const auto& point : jumpPoints) {
                start = Clock::now();
                setPlanes(session, layout, point);
                layout.viewers.back()->get().centerOnVolumePoint(point);
                record(report, tag + "focus jump (4 viewers)", waitSettled(session, layout.all(), start, timeoutMs));
            }
            setPlanes(session, layout, layout.focus);
            waitSettled(session, layout.all(), Clock::now(), timeoutMs);

            // A held arrow key on the XY view: inputs at 60 Hz, frames as they come.
            auto* xy = layout.viewers.front().get();
            const std::vector<Viewer*> one{xy};
            const auto framesBefore = xy->framesCompleted();
            xy->takeWorkerMs();
            start = Clock::now();
            for (int s = 0; s < burst; ++s) {
                const auto due = start + std::chrono::microseconds(16667LL * s);
                while (Clock::now() < due) {
                    session.processEvents();
                    QThread::usleep(200);
                }
                xy->get().onKeyPress(Qt::Key_Right, Qt::NoModifier);
            }
            const auto inputEnd = Clock::now();
            const auto settle = waitSettled(session, one, inputEnd, timeoutMs);
            const double seconds = msSince(start) / 1000.0;
            const auto frames = xy->framesCompleted() - framesBefore;
            report.add(tag + "xy pan burst (60 Hz input)", "presented_fps", double(frames) / seconds);
            report.add(tag + "xy pan burst (60 Hz input)", "settled_ms", settle.settledMs);
            for (const double ms : settle.workerMs)
                report.add(tag + "xy pan burst (60 Hz input)", "worker_ms", ms);
        }

        session.noteCaches();
        // Prefetch plan passes fetch what they lacked; a timed run the cache did not
        // fully serve records nothing (verifyCache).
        const int code = session.verifyCache();
        if (code == 0 && !fixture.prefetching()) {
            report.setConfig(session.config());
            report.finish();
        }
        layout.viewers.clear();
        return code;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_viewer_navigation: %s\n", e.what());
        return 1;
    }
}
