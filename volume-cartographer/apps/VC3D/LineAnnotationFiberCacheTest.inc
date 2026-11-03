// Included inside runObjectStoreControllerTests(). Always clone the supplied
// real store; neither the source nor a user's working project is edited.
if (const char* source = std::getenv("VC3D_FIBER_CACHE_TEST_STORE")) {
    QTemporaryDir directory;
    const auto copy = fs::path(directory.path().toStdString()) / "fibers.vcobj";
    OpenOptions options; options.author = "fiber-cache-regression";
    cloneHub(source, copy, options);
    detachHub(copy, options.author);
    CState state;
    Controller controller(&state, nullptr, nullptr, nullptr);
    controller._errorDialogsSuppressed = true;
    openStoreAsProject(state, QString::fromStdString(copy.string()), options.author.c_str());
    const auto all = controller.readStoredFibers(std::nullopt, true);
    const auto selected = std::find_if(all.begin(), all.end(), [](const auto& fiber) {
        return fiber.uuid == "6af54546-cad8-4696-9f4a-763334378476";
    });
    require(selected != all.end() && selected->branches.size() >= 2, "PHerc1451 linked workload exists");
    const auto original = *selected;
    {
        // Exercise the actual Spiral transport/export paths on real data. The
        // files must contain the exporter's exact bytes, including linked peers.
        const auto paths = controller.fiberTransportPaths({original.id});
        std::map<std::string, std::string> documents;
        const auto attachment = *state.storeFor("fiber");
        state.objectStoreSession()->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            ObjectSelection selection; selection.kind = "fiber"; selection.includeRetired = true;
            documents = fiberNetworkJson(db, txn, db.currentRevisions(txn, selection), {original.uuid});
        });
        require(paths.size() == documents.size() && paths.size() > 1, "transport includes the real linked component");
        for (const auto& [uuid, path] : paths) {
            std::ifstream input(path, std::ios::binary);
            const std::string bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
            require(bytes == documents.at(uuid), "atomic artifact publication preserves exact JSON bytes");
        }
        require(controller.fiberTransportPaths({original.id}) == paths, "unchanged artifacts reuse their paths");
        const auto exportedPath = fs::path(directory.path().toStdString()) / "export.json";
        QString error;
        int exported = 0;
        require(controller.exportFibersToPath(exportedPath, &error, &exported), "real fiber bundle exports inside the sandbox");
        std::ifstream input(exportedPath);
        const auto exportedJson = nlohmann::json::parse(input);
        input.close();
        require(exported > 0 && exportedJson.at("point_collections").size() == size_t(exported),
                "the published bundle is complete");
        require(controller.exportFibersToPath(exportedPath, &error), "atomic export replaces an existing bundle");
    }
    auto live = std::make_shared<Controller::LineAnnotationSession>();
    live->surfaceName = "cache-regression";
    live->fiberId = original.id; live->fiberUuid = original.uuid;
    live->fiberFileName = original.fileName; live->fiberUsername = original.username;
    live->fiberStartedAt = original.startedAt; live->fiberSequence = original.sequence;
    live->storePin = original.storePin; live->coordinateIdentity = original.coordinateIdentity;
    live->frame = original.frame; live->fiberOptimizationMode = original.optimizationMode;
    live->fiberTags = original.tags; live->fiberManualHvTag = original.manualHvTag;
    live->fiberWidth = original.width; live->fiberWidthGapFraction = original.widthGapFraction;
    live->modelOnly = original.modelOnly; live->branches = original.branches;
    live->suppressGeneratedViews = true; live->suppressErrorDialogs = true;
    for (size_t i = 0; i < original.controlPoints.size(); ++i) {
        const auto& cp = original.controlPoints[i];
        vc3d::line_annotation::LineControlPoint control(double(i), cp, i == 0, int(i));
        control.storeId = cp.storeId; control.segmentToNext = cp.segmentToNext;
        control.tags = cp.tags; control.direction = cp.direction;
        control.displayNormal = cp.displayNormal; control.displayNormalSource = cp.displayNormalSource;
        live->controlPoints.push_back(control);
    }
    for (const auto& position : original.linePoints) {
        vc::lasagna::LinePoint point; point.position = position;
        live->optimizedLine.points.push_back(point);
    }
    Controller::PaneRecord pane; pane.surfaceName = live->surfaceName; pane.session = live;
    controller._panes.push_back(pane);
    const auto attachment = *state.storeFor("fiber");
    const auto connection = state.objectStoreSession();
    const auto journal = [&] {
        int64_t position = 0;
        connection->read(attachment, [&](const auto& db, const auto& txn, const auto&) { position = db.journalPosition(txn); });
        return position;
    };
    const auto read = [&](const std::string& uuid) {
        return controller.readStoredFibers(std::vector<std::string>{uuid}, true).at(0);
    };
    controller.saveSessionAsFiber(*live, false);
    require(controller._fiberSaveFailureCount == 0, "initial canonical save succeeds");
    state.pollObjectStoreChanges();
    for (int i = 0; i < 5; ++i) {
        std::reverse(live->branches.begin(), live->branches.end());
        const auto position = journal();
        const auto pin = live->storePin;
        controller.saveSessionAsFiber(*live, false);
        require(controller._fiberSaveFailureCount == 0 && live->storePin == pin && journal() == position,
                "reordered unchanged branches produce no commit or journal writes");
        live->fiberWidth += .001;
        controller.saveSessionAsFiber(*live, false);
        require(controller._fiberSaveFailureCount == 0 && live->storePin != pin &&
                read(original.uuid).width == live->fiberWidth, "width changes publish synchronously");
        state.pollObjectStoreChanges();
    }
    require(read(original.uuid).controlPoints == original.controlPoints &&
            read(original.uuid).linePoints == original.linePoints, "width saves preserve exact geometry and annotations");
    auto pin = live->storePin;
    live->modelOnly = !live->modelOnly;
    controller.saveSessionAsFiber(*live, false);
    require(live->storePin != pin && read(original.uuid).modelOnly == live->modelOnly, "source metadata changes are saved");
    pin = live->storePin;
    live->controlPoints[1].volumePoint[0] = std::nextafter(live->controlPoints[1].volumePoint[0], INFINITY);
    controller.saveSessionAsFiber(*live, false);
    require(live->storePin != pin && read(original.uuid).controlPoints[1][0] == live->controlPoints[1].volumePoint[0],
            "a one-ULP coordinate edit is not swallowed by the no-op check");
    pin = live->storePin;
    live->branches.front().pending = !live->branches.front().pending;
    controller.saveSessionAsFiber(*live, false);
    require(live->storePin != pin, "relationship approval changes publish");
    state.pollObjectStoreChanges();
    // A stale editor must still hit the checked write, even with unchanged inputs.
    connection->editRecords(attachment, {original.uuid}, [](auto& record) { record.name += "-external"; });
    const auto failures = controller._fiberSaveFailureCount;
    controller.saveSessionAsFiber(*live, false);
    require(controller._fiberSaveFailureCount == failures + 1 &&
            read(original.uuid).fileName.ends_with("-external.json"), "external revisions cannot be silently overwritten");
    controller._panes.clear();
    state.pollObjectStoreChanges();

    std::vector<std::string> ids;
    for (const auto& fiber : all) ids.push_back(fiber.uuid);
    const auto warm = [&] {
        const auto fibers = controller.readStoredFibersCached(ids, true);
        (void)controller.fiberSnapshotsForSideStripQuery(fibers);
    };
    const auto checkFresh = [&] {
        const auto cached = controller.readStoredFibersCached(ids, true);
        const auto fresh = controller.readStoredFibers(ids, true);
        require(cached.size() == fresh.size(), "cache excludes deleted and retired fibers");
        for (size_t i = 0; i < fresh.size(); ++i) {
            const auto& a = cached[i]; const auto& b = fresh[i];
            require(a.uuid == b.uuid && a.storePin == b.storePin && a.fileName == b.fileName &&
                    a.width == b.width && a.tags == b.tags && a.controlPoints == b.controlPoints && a.linePoints == b.linePoints &&
                    a.branches.size() == b.branches.size(), "cached fibers match independent fresh reads");
            for (const auto& branch : a.branches) {
                const auto other = std::find_if(b.branches.begin(), b.branches.end(), [&](const auto& item) {
                    return item.edgeUuid == branch.edgeUuid;
                });
                require(other != b.branches.end() && other->branchFiberId == branch.branchFiberId &&
                        other->branchFileName == branch.branchFileName && other->pending == branch.pending &&
                        other->adjacent == branch.adjacent && other->autoLink == branch.autoLink &&
                        other->controlPointIndex == branch.controlPointIndex && other->branchControlPointIndex == branch.branchControlPointIndex &&
                        other->controlPointPosition == branch.controlPointPosition && other->branchControlPointPosition == branch.branchControlPointPosition &&
                        other->controlPointDirection == branch.controlPointDirection && other->branchControlPointDirection == branch.branchControlPointDirection,
                        "cached relationship endpoints and exact directions match fresh reads");
            }
        }
    };
    warm();
    auto edited = read(original.uuid);
    const auto unrelated = std::find_if(all.begin(), all.end(), [&](const auto& fiber) {
        try {
            vc3d::line_annotation::validateFiberSaveSnapshots(
                std::vector<Controller::FiberSaveSnapshot>{controller.makeFiberSaveSnapshot(fiber)});
        } catch (const std::exception&) { return false; }
        return fiber.uuid != original.uuid && std::none_of(edited.branches.begin(), edited.branches.end(),
            [&](const auto& branch) { return branch.branchFiberId == fiber.id; });
    });
    require(unrelated != all.end(), "unrelated real fiber exists");
    const auto* decodedPoints = controller._storedFiberDecodeCache.at(unrelated->uuid).linePoints.data();
    const auto sidePolyline = controller._sideStripStoredSnapshotCache.at(unrelated->id).snapshot.polyline;
    edited.width += .001;
    controller.commitFibers({&edited});
    checkFresh(); // before polling: peer generation checks must already work
    state.pollObjectStoreChanges();
    require(controller._storedFiberDecodeCache.contains(unrelated->uuid) &&
            controller._storedFiberDecodeCache.at(unrelated->uuid).linePoints.data() == decodedPoints &&
            controller._sideStripStoredSnapshotCache.at(unrelated->id).snapshot.polyline == sidePolyline,
            "a local edit preserves unrelated decoded fibers and side-strip snapshots");
    checkFresh();
    warm();
    edited = read(original.uuid);
    const auto peerUuid = controller.fiberUuidForId(edited.branches.front().branchFiberId);
    connection->editRecords(attachment, {peerUuid}, [](auto& record) { record.name += "-peer"; });
    checkFresh(); // branch display names depend on the peer even without a geometry change
    state.pollObjectStoreChanges();
    checkFresh();
    warm();
    edited = read(original.uuid);
    auto removedBranch = edited.branches.front();
    edited.branches.erase(edited.branches.begin());
    controller.commitFibers({&edited});
    state.pollObjectStoreChanges();
    require(!controller._storedFiberDecodeCache.contains(peerUuid), "deleted edge invalidates its old peer");
    checkFresh();
    warm();
    edited = read(original.uuid);
    auto newTarget = read(unrelated->uuid);
    removedBranch.branchFiberId = newTarget.id;
    removedBranch.branchControlPointIndex = 0;
    removedBranch.branchControlPointPosition = newTarget.controlPoints.front();
    removedBranch.branchControlPointDirection = endpointTangentFromLinePoints(
        newTarget.linePoints, newTarget.controlPoints.front(), {1., 0., 0.});
    edited.branches.push_back(removedBranch);
    controller.commitFibers({&edited});
    state.pollObjectStoreChanges();
    require(!controller._storedFiberDecodeCache.contains(newTarget.uuid), "new edge invalidates its new peer");
    checkFresh();
    warm();
    newTarget = read(newTarget.uuid);
    for (auto& point : newTarget.controlPoints) point[0] += .001;
    for (auto& point : newTarget.linePoints) point[0] += .001;
    controller.commitFibers({&newTarget});
    checkFresh();
    state.pollObjectStoreChanges();
    checkFresh();
    warm();
    newTarget = read(newTarget.uuid);
    std::vector<Controller::FiberSaveSnapshot> empty;
    controller.commitFiberStoreAction(empty, {newTarget});
    state.pollObjectStoreChanges();
    require(!controller._storedFiberDecodeCache.contains(newTarget.uuid), "retirement invalidates the retired fiber");
    checkFresh();
    for (int step = 0; step < 2; ++step) {
        warm();
        if (step == 0)
            (void)registerVolumeTransform(*connection, attachment, "cache-source", "cache-target",
                                          VolumeAffine{}, "cache regression");
        else {
            std::vector<std::string> route;
            connection->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                route = db.volumeTransforms(txn).resolve("cache-source", "cache-target").resolved().path;
            });
            (void)chooseVolumeTransform(*connection, attachment, "cache-source", "cache-target", route);
        }
        (void)controller.readStoredFibersCached({original.uuid}, true);
        require(controller._storedFiberDecodeCache.size() == 1 && controller._sideStripStoredSnapshotCache.empty(),
                "registration changes invalidate all coordinate-dependent caches before queued events arrive");
        checkFresh();
        state.pollObjectStoreChanges();
        require(controller._storedFiberDecodeCache.empty() && controller._storedFiberDependencies.empty() &&
                controller._sideStripStoredSnapshotCache.empty(), "registration events clear all coordinate-dependent caches");
        checkFresh();
    }
    controller._panes.clear();
    state.closeProject();
    std::cout << "PHerc1451 fiber save/cache regressions passed (" << all.size() << " live fibers)" << std::endl;
    return 0;
}
