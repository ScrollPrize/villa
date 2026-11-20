// Built only with VC_TESTING (VC3D --test-ink-labels). Drives the ink label
// controller through the segmentation viewer's own mouse slots, the path a
// pointer drag (or the agent bridge's canvas.drag) takes.
//
// Optional real data: VC3D_INKLABELS_VOLUME (zarr), VC3D_INKLABELS_SURFACE
// (tifxyz), VC3D_INKLABELS_PREDICTION (an image of the surface's render) and
// VC3D_INKLABELS_OUTPUT (a directory for captures and the saved label pair);
// VC3D_INKLABELS_FOCUS ("x,y" label px), VC3D_INKLABELS_ZOOM and
// VC3D_INKLABELS_PREDICTION_COLORMAP adjust the captures.
#include "CState.hpp"
#include "SurfaceCatalog.hpp"
#include "ToolOutput.hpp"
#include "ViewerManager.hpp"
#include "vc/core/types/Volume.hpp"
#include "inklabels/InkLabelController.hpp"
#include "overlays/InkDetectionOverlayController.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "test/support/TestVolume.hpp"

#include <QApplication>
#include <QCryptographicHash>
#include <QElapsedTimer>
#include <QFile>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QThread>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vc_object_store/Uuid.hpp>

#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

QByteArray fileHash(const std::filesystem::path& path)
{
    QFile file(QString::fromStdString(path.string()));
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    return QCryptographicHash::hash(file.readAll(), QCryptographicHash::Sha256).toHex();
}

cv::Vec3f bilinear(const cv::Mat_<cv::Vec3f>& points, double u, double v)
{
    const int c0 = std::clamp(static_cast<int>(std::floor(u)), 0, points.cols - 2);
    const int r0 = std::clamp(static_cast<int>(std::floor(v)), 0, points.rows - 2);
    const float fu = static_cast<float>(u - c0), fv = static_cast<float>(v - r0);
    return points(r0, c0) * (1 - fu) * (1 - fv) + points(r0, c0 + 1) * fu * (1 - fv) +
           points(r0 + 1, c0) * (1 - fu) * fv + points(r0 + 1, c0 + 1) * fu * fv;
}

} // namespace

int runInkLabelControllerTests()
{
    using vc3d::inklabels::Label;
    auto require = [](bool ok, const std::string& message) {
        if (!ok) throw std::runtime_error(message);
    };
    auto pump = [](int ms) {
        QElapsedTimer timer;
        timer.start();
        while (timer.elapsed() < ms) {
            QApplication::processEvents();
            QThread::msleep(5);
        }
    };
    try {
        QTemporaryDir temporary;
        qputenv("VC3D_CONFIG_DIR", temporary.path().toUtf8());
        const QString volumePath = qEnvironmentVariable("VC3D_INKLABELS_VOLUME");
        const QString surfacePath = qEnvironmentVariable("VC3D_INKLABELS_SURFACE");
        const QString predictionPath = qEnvironmentVariable("VC3D_INKLABELS_PREDICTION");
        const QString outputDir = qEnvironmentVariable("VC3D_INKLABELS_OUTPUT");
        const bool real = !volumePath.isEmpty() && !surfacePath.isEmpty();
        const std::filesystem::path out = outputDir.isEmpty() ? std::filesystem::path(temporary.path().toStdString())
                                                              : std::filesystem::path(outputDir.toStdString());
        std::filesystem::create_directories(out);

        std::shared_ptr<Volume> volume;
        std::shared_ptr<QuadSurface> surface;
        if (real) {
            volume = Volume::New(volumePath.toStdString());
            surface = std::make_shared<QuadSurface>(surfacePath.toStdString());
        } else {
            vc3d::test::VolumeSpec spec;
            spec.shapeZYX = {64, 64, 64};
            spec.chunkShapeZYX = {32, 32, 32};
            volume = vc3d::test::createVolume(std::filesystem::path(temporary.path().toStdString()) / "ct.zarr", spec);
            cv::Mat_<cv::Vec3f> points(24, 30);
            for (int r = 0; r < points.rows; ++r)
                for (int c = 0; c < points.cols; ++c)
                    points(r, c) = {4.0f + 2.0f * c, 6.0f + 2.0f * r, 32.0f};
            surface = std::make_shared<QuadSurface>(points, cv::Vec2f(0.05f, 0.05f));
        }
        surface->id = "ink-label-test-surface";
        const cv::Mat_<cv::Vec3f> points = surface->rawPoints();
        const cv::Size canvas = vc3d::inklabels::canvasSize(points.rows, points.cols, surface->scale());

        CState state;
        state.setCurrentVolume(volume);
        state.setSurface("segmentation", surface);
        ViewerManager viewers(&state, state.pointCollection());
        InkDetectionOverlayController prediction(&state);
        viewers.setInkDetectionOverlay(&prediction);
        InkLabelController labels(&state);
        labels.bindViewerManager(&viewers);
        QWidget host;
        host.resize(1280, 720);
        auto* viewer = viewers.createViewerInWidget("segmentation", &host);
        auto* chunked = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject());
        require(chunked != nullptr, "segmentation viewer is a chunked viewer");
        if (auto* widget = qobject_cast<QWidget*>(viewer->asQObject())) {
            widget->resize(host.size());
        }
        host.show();
        pump(500);
        require(labels.hasSurface(), "the controller follows the segmentation surface");
        require(labels.layer().size() == canvas, "labels use the surface's render canvas");
        require(cv::countNonZero(labels.layer().labels()) == 0, "a new surface starts unlabelled");

        // Look at a region of the segment (the middle, or a chosen one for real data).
        const cv::Vec3f centre = surface->center();
        cv::Point2f focus(canvas.width * 0.5f, canvas.height * 0.5f);
        if (const QString f = qEnvironmentVariable("VC3D_INKLABELS_FOCUS"); !f.isEmpty()) {
            const auto parts = f.split(',');
            focus = {parts.value(0).toFloat(), parts.value(1).toFloat()};
        }
        viewer->centerOnSurfacePoint({focus.x - centre[0], focus.y - centre[1]});
        if (const QString z = qEnvironmentVariable("VC3D_INKLABELS_ZOOM"); !z.isEmpty()) {
            viewer->adjustZoomByFactor(z.toFloat());
        }
        QElapsedTimer settle;
        settle.start();
        while (!chunked->hasDisplayedRenderFrame() && settle.elapsed() < 30000) pump(50);
        pump(real ? 4000 : 300);

        auto sceneAt = [&](cv::Point2f px) { return viewer->surfaceCoordsToScene(px.x - centre[0], px.y - centre[1]); };
        auto stroke = [&](std::vector<cv::Point2f> path, Qt::KeyboardModifiers modifiers = Qt::NoModifier) {
            chunked->onMousePress(sceneAt(path.front()), Qt::LeftButton, modifiers);
            for (std::size_t i = 1; i < path.size(); ++i) {
                const cv::Point2f a = path[i - 1], b = path[i];
                for (int s = 1; s <= 8; ++s)
                    chunked->onMouseMove(sceneAt(a + (b - a) * (s / 8.0f)), Qt::LeftButton, modifiers);
            }
            chunked->onMouseRelease(sceneAt(path.back()), Qt::LeftButton, modifiers);
            QApplication::processEvents();
        };
        auto capture = [&](const std::string& name) {
            if (outputDir.isEmpty()) return;
            pump(300);
            require(host.grab().save(QString::fromStdString((out / name).string())), "cannot save " + name);
        };
        capture("01_segment_unlabelled.png");

        // Painting needs labelling switched on.
        stroke({focus});
        require(cv::countNonZero(labels.layer().labels()) == 0, "drags do nothing while labelling is off");
        labels.setLabellingEnabled(true);

        // One ink dot: it lands on the label pixels under the pointer.
        labels.setBrushRadius(5.0f);
        const cv::Point2f dot = focus + cv::Point2f(-120.0f, -40.0f);
        stroke({dot});
        cv::Mat_<uint8_t> inkMask = labels.layer().labels() == static_cast<uint8_t>(Label::Ink);
        require(cv::countNonZero(inkMask) > 0, "an ink click paints");
        std::vector<cv::Point> inkPixels;
        cv::findNonZero(inkMask, inkPixels);
        cv::Point2f centroid(0.5f, 0.5f);
        for (const auto& p : inkPixels) centroid += cv::Point2f(p) / static_cast<float>(inkPixels.size());
        const float offset2d = static_cast<float>(cv::norm(centroid - dot));
        std::cout << "ink dot at label px (" << dot.x << ", " << dot.y << "), painted centroid ("
                  << centroid.x << ", " << centroid.y << "), offset " << offset2d << " px\n";
        require(offset2d < 0.75f, "the painted dot sits under the pointer");

        // P6: the painted pixels map through the tifxyz to the volume point the viewer reports.
        const double u = centroid.x * points.cols / static_cast<double>(canvas.width);
        const double v = centroid.y * points.rows / static_cast<double>(canvas.height);
        const cv::Vec3f fromLabel = bilinear(points, u, v);
        const cv::Vec3f fromViewer = chunked->sceneToVolume(sceneAt(dot));
        const double offset3d = cv::norm(fromLabel - fromViewer);
        std::cout << "volume point from label " << fromLabel << ", from viewer " << fromViewer
                  << ", distance " << offset3d << " voxels\n";
        require(offset3d < 1.5, "label pixels map to the viewer's volume point");

        // Strokes of each class; Alt paints the other class.
        labels.setBrushRadius(8.0f);
        stroke({focus + cv::Point2f(-200, 30), focus + cv::Point2f(200, 30)}, Qt::AltModifier);
        stroke({focus + cv::Point2f(-150, -10), focus + cv::Point2f(-60, 60), focus + cv::Point2f(40, -10)});
        labels.setTool(InkLabelController::Tool::NoInk);
        stroke({focus + cv::Point2f(-200, 90), focus + cv::Point2f(200, 90)});
        labels.setTool(InkLabelController::Tool::Ink);
        labels.setBrushRadius(4.0f);
        stroke({focus + cv::Point2f(60, -30), focus + cv::Point2f(110, 40), focus + cv::Point2f(160, -30)});
        auto counts = labels.layer().counts();
        require(counts[1] > 0 && counts[2] > 0, "ink and no-ink both painted");
        capture("02_labels_on_segment.png");

        // Prediction overlay under the labels at two opacities.
        if (!predictionPath.isEmpty()) {
            prediction.addLocalDetection(predictionPath.toStdString());
            require(prediction.hasLoadedSelection(), "the prediction image loads");
            if (const QString colormap = qEnvironmentVariable("VC3D_INKLABELS_PREDICTION_COLORMAP"); !colormap.isEmpty()) {
                prediction.setColormapId(colormap.toStdString());
            }
            prediction.setOpacity(30);
            labels.setOpacity(45);
            capture("03_prediction_30.png");
            prediction.setOpacity(85);
            capture("04_prediction_85.png");
            labels.setLabelsVisible(false);
            capture("05_prediction_85_labels_hidden.png");
            labels.setLabelsVisible(true);
            prediction.setOpacity(0);
        }

        // Undo: 25 more strokes, each undone in turn back to the state before them.
        labels.setTool(InkLabelController::Tool::Ink);
        labels.setBrushRadius(3.0f);
        const cv::Mat_<uint8_t> beforeBatch = labels.layer().labels().clone();
        std::vector<cv::Mat_<uint8_t>> states{beforeBatch};
        const std::size_t depthBefore = labels.layer().undoDepth();
        for (int i = 0; i < 25; ++i) {
            // Every fifth stroke erases the one before it, partly.
            const int column = i % 5 == 4 ? i - 1 : i;
            const cv::Point2f a = focus + cv::Point2f(-240.0f + 18.0f * column, 130.0f);
            const cv::Point2f b = a + cv::Point2f(9.0f, 40.0f + (column % 3) * 10.0f);
            if (i % 5 == 4) labels.setTool(InkLabelController::Tool::Erase);
            stroke({a, b + (i % 5 == 4 ? cv::Point2f(0, -20) : cv::Point2f())},
                   i % 5 == 2 ? Qt::AltModifier : Qt::NoModifier);
            labels.setTool(InkLabelController::Tool::Ink);
            states.push_back(labels.layer().labels().clone());
        }
        require(labels.layer().undoDepth() - depthBefore == 25, "each of the 25 strokes is one undo step");
        for (int k = 0; k < 25; ++k) {
            labels.undo();
            require(cv::countNonZero(labels.layer().labels() != states[states.size() - 2 - k]) == 0,
                    "undo " + std::to_string(k + 1) + " restores the earlier labels");
        }
        for (int k = 0; k < 25; ++k) labels.redo();
        require(cv::countNonZero(labels.layer().labels() != states.back()) == 0, "redo reapplies the strokes");
        capture("06_after_undo_redo.png");

        // Save, load into a fresh controller, save again: identical files.
        const auto first = vc3d::inklabels::trainingLabelFiles(out, "w_spike");
        require(labels.saveTo(out, "w_spike"), "saving the label pair");
        const QByteArray inkHash = fileHash(first.inklabels), supervisionHash = fileHash(first.supervisionMask);
        InkLabelController reloaded(&state);
        reloaded.bindViewerManager(&viewers);
        require(reloaded.load(first), "loading the saved pair");
        require(cv::countNonZero(reloaded.layer().labels() != labels.layer().labels()) == 0,
                "loaded labels equal the painted ones");
        const auto second = vc3d::inklabels::trainingLabelFiles(out / "resaved", "w_spike");
        require(reloaded.saveTo(out / "resaved", "w_spike"), "saving the reloaded labels");
        require(fileHash(second.inklabels) == inkHash && fileHash(second.supervisionMask) == supervisionHash,
                "re-saving loaded labels reproduces the files byte for byte");
        cv::imwrite((out / "w_spike_states.png").string(), labels.layer().labels());
        counts = labels.layer().counts();
        std::cout << "canvas " << canvas.width << " x " << canvas.height << " px, ink " << counts[2]
                  << " px, no-ink " << counts[1] << " px, undo depth " << labels.layer().undoDepth()
                  << ", sha256 ink " << inkHash.constData() << ", supervision " << supervisionHash.constData() << "\n";

        // Labels saved for a store surface survive a segment edit that resizes its grid: VC3D must not
        // paint over them on the new grid or autosave a different-size pair over them.
        if (!real) {
            namespace vo = vc::objects;
            labels.setLabellingEnabled(false);
            const auto storePath = std::filesystem::path(temporary.path().toStdString()) / "labels.vcobj";
            vo::OpenOptions storeOptions;
            storeOptions.author = "ink-label-test";
            const std::string uuid = vo::util::newUuid();
            {
                auto db = vo::ObjectStore::create(storePath, storeOptions);
                auto txn = db.write({{uuid, std::nullopt}});
                vo::ObjectRecord record;
                record.uuid = uuid;
                record.kind = "surface";
                record.name = "store-segment";
                record.meta = nlohmann::json{
                    {"coordinates", {{"volume_uuid", volume->id()}, {"scale_factor", 1}}},
                    {"catalog", {{"sample", "sample"}, {"segment", "segment"}, {"files", nlohmann::json::array()}}}}.dump();
                db.upsertObject(txn, record);
                txn.commit();
            }
            state.objectStoreSession()->setGeometryFetcher("catalog",
                [](vo::ObjectStore&, const std::string&, const vo::ObjectStoreSession::FetchWrite&) {
                    throw std::runtime_error("offline");
                });
            const auto attachment = state.objectStoreSession()->attach(storePath, storeOptions);
            SurfaceCatalog catalog(state.objectStoreSession());
            catalog.setSource(attachment, volume, {volume->id()});
            QElapsedTimer listed;
            listed.start();
            while ((catalog.loading() || catalog.rows().empty()) && listed.elapsed() < 10000) pump(10);
            require(catalog.rows().size() == 1, "the test store lists its surface");

            // Store surfaces keep labels in VC3D's export folder (application data): use Qt's test location.
            QStandardPaths::setTestModeEnabled(true);
            const std::filesystem::path exportFolder = vc3d::surfaceExportDirectory(QString::fromStdString(uuid)).toStdString();
            struct RemoveAfter {
                std::filesystem::path path;
                ~RemoveAfter() { std::error_code ec; std::filesystem::remove_all(path, ec); }
            } removeExportFolder{exportFolder};

            InkLabelController stored(&state);
            stored.setSurfaceCatalog(&catalog);
            stored.bindViewerManager(&viewers);
            auto withColumns = [&](int cols) {
                cv::Mat_<cv::Vec3f> grid(points.rows, cols);
                for (int r = 0; r < grid.rows; ++r)
                    for (int c = 0; c < grid.cols; ++c)
                        grid(r, c) = {4.0f + 2.0f * c, 6.0f + 2.0f * r, 32.0f};
                auto s = std::make_shared<QuadSurface>(grid, surface->scale());
                s->id = uuid;
                return s;
            };
            state.setSurface("segmentation", withColumns(points.cols));
            pump(300);
            require(stored.labelDirectory() == exportFolder, "a store surface keeps its labels in its export folder");
            stored.setLabellingEnabled(true);
            stored.setBrushRadius(6.0f);
            stroke({focus + cv::Point2f(-60, 0), focus + cv::Point2f(60, 0)});
            const auto painted = stored.layer().counts();
            require(painted[2] > 0, "painting on the store surface");
            const auto files = stored.defaultFiles();
            state.setSurface("segmentation", nullptr);  // switching surface saves the labels
            pump(100);
            require(std::filesystem::exists(files.inklabels) && std::filesystem::exists(files.supervisionMask),
                    "switching surface autosaves the labels to the export folder");
            state.setSurface("segmentation", withColumns(points.cols));
            pump(300);
            require(stored.layer().counts() == painted && !stored.layer().modified(),
                    "reopening the surface loads its labels from the export folder");
            const QByteArray storedInk = fileHash(files.inklabels), storedSupervision = fileHash(files.supervisionMask);

            // The same surface after an edit that added a grid column (Manual Add, growth).
            state.setSurface("segmentation", withColumns(points.cols + 1));
            pump(300);
            require(stored.savedLabelsBlocked(), "labels saved for another grid size are reported");
            stroke({focus + cv::Point2f(-60, 40), focus + cv::Point2f(60, 40)});
            require(stored.layer().counts()[1] + stored.layer().counts()[2] == 0, "no painting over a resized grid");
            require(!stored.save(), "Save does not write over labels saved for another grid size");
            state.setSurface("segmentation", nullptr);  // the autosave on switching surface
            pump(100);
            require(fileHash(files.inklabels) == storedInk && fileHash(files.supervisionMask) == storedSupervision,
                    "labels saved for a resized surface are not overwritten");
            // Back on the grid they were painted on, they load again.
            state.setSurface("segmentation", withColumns(points.cols));
            pump(300);
            require(!stored.savedLabelsBlocked() && stored.layer().counts()[2] > 0, "labels load again on their own grid");
            stored.setLabellingEnabled(false);
            std::cout << "store surface resized: saved labels unchanged\n";
        }
        std::cout << "Ink labels: painting, frame, undo/redo, save/load and re-save passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Ink label test: " << error.what() << std::endl;
        return 1;
    }
}
