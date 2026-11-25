// Read-only live-store regression/benchmark. Run through --test-object-store;
// setup and the already-open source fiber are outside the measured nearby query.
if (const char* endpoint = std::getenv("VC3D_NEARBY_BENCH_ENDPOINT")) {
    const char* volumePath = std::getenv("VC3D_NEARBY_BENCH_VOLUME");
    const char* cachePath = std::getenv("VC3D_NEARBY_BENCH_CACHE");
    require(volumePath && cachePath, "nearby benchmark requires volume and isolated cache paths");
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::REMOTE_CACHE_DIR, QString::fromUtf8(cachePath));
    settings.setValue(vc3d::settings::perf::REMOTE_CACHE_MIN_FREE_GIB, 0);
    settings.sync();
    QTemporaryDir directory;
    const auto localPath = fs::path(directory.path().toStdString()) / "project.vcobj";
    OpenOptions localOptions; localOptions.author = "nearby-read-only-benchmark";
    { auto store = ObjectStore::create(localPath, localOptions); }
    CState state;
    openStoreAsProject(state, QString::fromStdString(localPath.string()), "nearby-read-only-benchmark");
    state.setCurrentVolume(Volume::New(volumePath));
    const auto attachment = state.attachHostedStore(endpoint, "nearby-read-only-benchmark");
    state.setStoreFor("fiber", attachment);
    Controller controller(&state, nullptr, nullptr, nullptr);
    controller._errorDialogsSuppressed = true;
    std::string seedUuid;
    const auto connection = state.objectStoreSession();
    connection->readStore(attachment, [&](const StoreReader& reader, const auto&) {
        ObjectSelection selection; selection.kind = "fiber"; selection.volume = state.annotationVolume()->volume;
        selection.targetScale = state.annotationVolume()->scale; selection.limit = 1;
        seedUuid = reader.selectObjects(selection, std::vector<std::string>{}).at(0).uuid;
    });
    const auto seed = controller.readStoredFibersCached({seedUuid}, true).at(0);
    require(seed.linePoints.size() > 2, "real source fiber has sampled geometry");
    const size_t middle = seed.linePoints.size() / 2;
    const auto center = seed.linePoints[middle];
    const auto normal = normalizedOrZero(seed.linePoints[middle + 1] - seed.linePoints[middle - 1]);
    const auto axis = std::abs(normal[2]) < .9 ? cv::Vec3d{0, 0, 1} : cv::Vec3d{0, 1, 0};
    const auto u = normalizedOrZero(normal.cross(axis)) * 500.;
    const auto v = normalizedOrZero(normal.cross(u)) * 500.;
    cv::Mat_<cv::Vec3f> points(2, 2);
    for (int r = 0; r < 2; ++r) for (int c = 0; c < 2; ++c)
        points(r, c) = cv::Vec3f(center + (2 * r - 1) * u + (2 * c - 1) * v);
    const std::string surfaceName = "nearby-benchmark";
    state.setSurface(surfaceName, std::make_shared<QuadSurface>(points, cv::Vec2f{1, 1}));
    auto dialog = std::make_unique<LineAnnotationDialog>(nullptr);
    auto open = std::make_shared<Controller::LineAnnotationSession>();
    open->surfaceName = surfaceName; open->fiberId = 0; open->fiberUuid = seed.uuid;
    open->suppressErrorDialogs = true; open->suppressFiberSave = true;
    Controller::PaneRecord pane; pane.surfaceName = surfaceName; pane.session = open; pane.dialog = dialog.get();
    controller._panes.push_back(pane);
    state.pollObjectStoreChanges(); QApplication::processEvents();
    const bool synchronous = qEnvironmentVariable("VC3D_NEARBY_BENCH_MODE") == "sync";
    const int repetitions = std::max(1, qEnvironmentVariableIntValue("VC3D_NEARBY_BENCH_REPETITIONS"));
    const auto stats = [&](const char* operation) {
        if (const char* proxy = std::getenv("VC3D_NEARBY_BENCH_STATS")) {
            vcr::http::Session http;
            return nlohmann::json::parse(http.request("GET", std::string(proxy) + operation).body);
        }
        return nlohmann::json::object();
    };
    for (int iteration = 0; iteration < repetitions; ++iteration) {
        (void)stats("/benchmark/reset");
        QElapsedTimer timer; timer.start();
        double lastBeat = 0, maxDelay = 0;
        int heartbeats = 0;
        QTimer heartbeat; heartbeat.setInterval(5);
        QObject::connect(&heartbeat, &QTimer::timeout, [&] {
            const double now = double(timer.nsecsElapsed()) / 1e6;
            maxDelay = std::max(maxDelay, now - lastBeat); lastBeat = now; ++heartbeats;
        });
        heartbeat.start();
        std::vector<Controller::SideStripMarker> markers;
        if (synchronous) {
            std::array<double, 6> bounds{INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY};
            for (const auto& point : points) for (int axis = 0; axis < 3; ++axis) {
                bounds[axis] = std::min(bounds[axis], double(point[axis]) - 1.0e-6);
                bounds[axis + 3] = std::max(bounds[axis + 3], double(point[axis]) + 1.0e-6);
            }
            std::vector<std::string> selected;
            Controller::SideStripIntersectionRequest request;
            connection->readStore(attachment, [&](const StoreReader& reader, const auto&) {
                ObjectSelection selection; selection.kind = "fiber"; selection.volume = state.annotationVolume()->volume;
                selection.targetScale = state.annotationVolume()->scale; selection.bbox = bounds;
                for (const auto& object : reader.selectObjects(selection, std::vector<std::string>{})) selected.push_back(object.uuid);
                request.unboundFibers = reader.unboundObjectCount("fiber", selection.volume.value());
            });
            std::sort(selected.begin(), selected.end()); selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
            const auto stored = controller.readStoredFibersCached(selected, true);
            request.fibers = controller.fiberSnapshotsForSideStripQuery(stored);
            request.stripPoints = points;
            // Measure the old synchronous preparation stall separately from the
            // shared numerical intersection implementation, which was already async.
            maxDelay = double(timer.nsecsElapsed()) / 1e6;
            auto result = Controller::runSideStripIntersectionQuery(request);
            require(result.ok, "reference intersection succeeds");
            markers = std::move(result.markers);
        } else {
            controller.dispatchSideStripIntersectionQuery(surfaceName);
            while (controller._sideStripIntersectionRunning && timer.elapsed() < 120000) {
                QApplication::processEvents(); QThread::msleep(1);
            }
            require(!controller._sideStripIntersectionRunning && controller._sideStripReuseCache.contains(surfaceName),
                    "background selection, geometry and intersections finish");
            markers = controller._sideStripReuseCache.at(surfaceName).markers;
        }
        const double elapsed = double(timer.nsecsElapsed()) / 1e6;
        heartbeat.stop();
        nlohmann::json geometry = nlohmann::json::object(), markerValues = nlohmann::json::array();
        for (const auto& [id, entry] : controller._sideStripStoredSnapshotCache) {
            auto& line = geometry[controller.fiberUuidForId(id)] = nlohmann::json::array();
            for (const auto& point : entry.snapshot.polyline->points) line.push_back({point[0], point[1], point[2]});
        }
        for (const auto& marker : markers) markerValues.push_back({controller.fiberUuidForId(marker.fiberId),
            {marker.point[0], marker.point[1], marker.point[2]}, marker.segmentIndex, marker.arclength, marker.distance,
            marker.projectedBranchLink, marker.pendingBranchLink});
        const auto hash = [](const nlohmann::json& value) { const auto text = value.dump(); return payloadHash({text.begin(), text.end()}); };
        const auto network = stats("/benchmark/stats");
        std::cout << "NEARBY_BENCH " << nlohmann::json{{"mode", synchronous ? "sync" : "async"}, {"iteration", iteration},
            {"elapsed_ms", elapsed}, {"max_event_gap_ms", maxDelay}, {"heartbeats", heartbeats}, {"source_uuid", seedUuid},
            {"candidates", geometry.size()}, {"markers", markers.size()}, {"geometry_sha256", hash(geometry)},
            {"markers_sha256", hash(markerValues)}, {"network", network}}.dump() << std::endl;
    }
    if (!synchronous && qEnvironmentVariableIsSet("VC3D_NEARBY_BENCH_STRESS")) {
        const auto drain = [&] {
            QElapsedTimer deadline; deadline.start();
            while (controller._sideStripIntersectionRunning && deadline.elapsed() < 120000) {
                QApplication::processEvents(); QThread::msleep(1);
            }
            require(!controller._sideStripIntersectionRunning && controller._pendingSideStripIntersectionRequests.empty(),
                    "nearby scheduler drains running and latest pending requests");
        };
        auto secondDialog = std::make_unique<LineAnnotationDialog>(nullptr);
        auto secondSession = std::make_shared<Controller::LineAnnotationSession>();
        secondSession->surfaceName = "nearby-second"; secondSession->suppressFiberSave = true;
        state.setSurface(secondSession->surfaceName, std::make_shared<QuadSurface>(points, cv::Vec2f{1, 1}));
        Controller::PaneRecord second;
        second.surfaceName = secondSession->surfaceName; second.session = secondSession; second.dialog = secondDialog.get();
        controller._panes.push_back(second);
        controller._sideStripReuseCache.clear();
        controller.dispatchSideStripIntersectionQuery(surfaceName);
        require(controller._sideStripIntersectionRunning, "loading counts as a running request");
        ++open->lineRevision;
        controller.dispatchSideStripIntersectionQuery(surfaceName);
        controller.dispatchSideStripIntersectionQuery(second.surfaceName);
        controller.dispatchSideStripIntersectionQuery(second.surfaceName);
        require(controller._pendingSideStripIntersectionRequests.size() == 2, "latest request retained once per surface");
        drain();
        require(controller._sideStripReuseCache.contains(surfaceName) && controller._sideStripReuseCache.contains(second.surfaceName),
                "both surfaces publish after overlapping loads");
        controller._sideStripReuseCache.clear();
        controller.dispatchSideStripIntersectionQuery(surfaceName);
        controller.invalidateSideStripQueries();
        controller.dispatchSideStripIntersectionQuery(surfaceName);
        drain();
        require(controller._lastPublishedSideStripToken > controller._sideStripInvalidationWatermark,
                "invalidated loading cannot swallow a replacement request");
        {
            auto transient = std::make_unique<Controller>(&state, nullptr, nullptr, nullptr);
            transient->_panes.push_back(pane);
            transient->dispatchSideStripIntersectionQuery(surfaceName);
            require(transient->_sideStripIntersectionRunning, "destruction test has an active load");
            QElapsedTimer deadline; deadline.start();
            transient.reset();
            require(deadline.elapsed() < 1000, "controller destruction does not wait for hosted annotation reads");
        }
        require(QThreadPool::globalInstance()->waitForDone(10000), "destroyed controller workers cancel and drain");
        controller._panes.pop_back();
        std::cout << "NEARBY_STRESS passed" << std::endl;
    }
    controller._panes.clear();
    state.closeProject();
    return 0;
}
