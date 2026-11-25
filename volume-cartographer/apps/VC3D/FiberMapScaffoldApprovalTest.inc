// The scaffold approval dock's rules against a real store: a measured
// crossing may only be approved while both pieces it lies on are approved
// geometry, the approval drops out again when that geometry approval is cleared,
// and an anchor's covering span is resolved from the fiber's own controls.
// Included in runObjectStoreControllerTests (VC_TESTING only).
{
    namespace la = vc3d::line_annotation;
    using vc::fiber_graph::ScaffoldConstraintKind;
    using vc::fiber_graph::ScaffoldStoredResidual;
    QTemporaryDir directory;
    const auto storePath = fs::path(directory.path().toStdString()) / "scaffold-approval.vcobj";
    OpenOptions options; options.author = "scaffold-approval-test";
    { auto db = ObjectStore::create(storePath, options); }
    CState state;
    Controller controller(&state, nullptr, nullptr, nullptr);
    controller._errorDialogsSuppressed = true;
    openStoreAsProject(state, QString::fromStdString(storePath.string()), options.author.c_str());

    // Two crossing fibers, eight controls every 10 vx, one line point per vx.
    const auto makeFiber = [&](const char* name, bool vertical) {
        Controller::StoredFiber fiber;
        fiber.id = controller.nextFiberId(); fiber.fileName = name;
        fiber.username = "test"; fiber.startedAt = "20260101"; fiber.sequence = 1;
        for (int i = 0; i < 8; ++i) {
            const cv::Vec3d point = vertical ? cv::Vec3d{35.0, 10.0 * i - 35.0, 1} : cv::Vec3d{10.0 * i, 0, 1};
            la::StoredControlPoint control{point};
            control.direction = vertical ? cv::Vec3d{0, 1, 0} : cv::Vec3d{1, 0, 0};
            if (i < 7) { la::FiberTraceSegmentMetadata span; control.segmentToNext = span; }
            fiber.controlPoints.push_back(control);
        }
        for (int step = 0; step <= 70; ++step)
            fiber.linePoints.emplace_back(vertical ? cv::Vec3d{35.0, double(step) - 35.0, 1}
                                                   : cv::Vec3d{double(step), 0, 1});
        std::vector<Controller::FiberSaveSnapshot> batch{controller.makeFiberSaveSnapshot(fiber)};
        controller.commitFiberStoreAction(batch);
        auto stored = controller.readStoredFibers(std::vector<std::string>{batch.front().fiber.uuid}, true);
        if (stored.empty()) throw std::runtime_error("scaffold approval test: fiber missing");
        return stored.front();
    };
    const auto horizontal = makeFiber("scaffold-h.json", false);
    const auto vertical = makeFiber("scaffold-v.json", true);
    state.setStoreFor("fiber", horizontal.storePin->attachment);

    // One saved run with one violated perpendicular crossing at arc 35 on
    // both fibers, which is control 3's span on each.
    const auto runUuid = vc::objects::util::newUuid();
    {
        auto request = state.objectStoreSession()->newRequest(horizontal.storePin->attachment);
        for (const auto role : {vc::objects::FiberGraphTable::Runs, vc::objects::FiberGraphTable::Residuals})
            request.operations.push_back(
                vc::objects::CreateExtensionTableOp{vc::objects::fiberGraphSolveTable(role)});
        auto run = vc::objects::fiberGraphRunRow(runUuid, "winding", "lemon-scaffold", {});
        auto residual = vc::objects::fiberGraphSolveRow(
            vc::objects::FiberGraphTable::Residuals,
            {{"run_uuid", vc::objects::extensionUuid(runUuid)},
             {"residual_index", std::int64_t(0)},
             {"kind", std::int64_t(ScaffoldConstraintKind::Perpendicular)},
             {"fiber_a", vc::objects::extensionUuid(horizontal.uuid)},
             {"fiber_b", vc::objects::extensionUuid(vertical.uuid)},
             {"target", std::int64_t(1)},
             {"weight", 1.0},
             {"residual", 0.5},
             {"arc_a", 35.0},
             {"arc_b", 35.0}});
        request.operations.push_back(vc::objects::AppendExtensionRowsOp{
            vc::objects::fiberGraphSolveTable(vc::objects::FiberGraphTable::Runs).name, {run.values}});
        request.operations.push_back(vc::objects::AppendExtensionRowsOp{
            vc::objects::fiberGraphSolveTable(vc::objects::FiberGraphTable::Residuals).name, {residual.values}});
        state.objectStoreSession()->submitDurably(horizontal.storePin->attachment, request);
    }

    vc3d::fiber_map::ScaffoldSolveController solves(&state);
    solves.reloadRuns();
    require(solves.runsLoaded(), "the dock reads the store's runs");
    require(solves.runs().size() == 1 && solves.runs().front().runUuid == runUuid, "the saved run is listed");
    solves.selectRun(runUuid);
    const auto* view = solves.selected();
    require(view && view->residuals.size() == 1, "the run's residual is read back");
    const auto residual = view->residuals.front();
    solves.loadApprovals();

    // The covering span of each anchor comes from the fiber's own controls,
    // and starts out unapproved.
    const auto spanA = solves.anchorSpan(residual.fiberA, residual.arcA);
    const auto spanB = solves.anchorSpan(residual.fiberB, residual.arcB);
    require(spanA.known && spanB.known, "both anchors resolve to a covering span");
    require(spanA.controlFrom == 3 && spanA.controlTo == 4, "arc 35 lies between controls 3 and 4");
    require(!spanA.geometryApproved && !spanB.geometryApproved, "nothing is approved yet");

    // The dock's stretches end on control points: the accused stretch lies in
    // one that covers it, and opening it hands the annotation the line points
    // of those controls.
    {
        ScaffoldStoredResidual accusing = residual;
        vc::fiber_graph::ScaffoldStoredPiece stretch;
        stretch.fiberUuid = horizontal.uuid;
        stretch.arcFrom = 12.4;
        stretch.arcTo = 27.6;
        accusing.cycleHops = 2;
        accusing.pieces = {stretch};
        const auto segments = vc3d::fiber_map::scaffoldApprovalSegments({accusing}, solves.approvals());
        const auto found = std::find_if(segments.begin(), segments.end(), [&](const auto& segment) {
            return segment.fiberUuid == horizontal.uuid && segment.cost > 0;
        });
        require(found != segments.end(), "the accused stretch lies in one of the fiber's stretches");
        require(found->arcFrom <= 12.4 && found->arcTo >= 27.6, "that stretch covers the whole accusation");
        require(found->controlFrom == std::round(found->controlFrom) && found->controlTo == std::round(found->controlTo),
                "its ends are control points");
        const auto onControl = controller.storedLineIndexAtArc(horizontal.id, found->arcFrom);
        require(onControl && *onControl == std::int64_t(std::llround(found->arcFrom)),
                "a stretch's end is an existing control's line point");
        const auto controls = controller.storedControlSummaries(horizontal.id);
        require(controls.size() == 8, "the fiber's controls are untouched");
    }

    // Approval is refused while the pieces are not approved, and nothing is
    // written: the dock's button rule and the store write agree.
    const auto refusal = solves.setConstraintApproval(residual, true);
    require(!refusal.isEmpty(), "a crossing over unapproved geometry is refused");
    require(!solves.constraintApproved(residual), "the refusal stored no approval");

    // The geometry between controls 3 and 4 of both fibers gets a state in
    // their approval layer, as the approval window writes it.
    const auto writeLayer = [&](const Controller::StoredFiber& fiber, vc::objects::FiberStretchState stretchState) {
        auto request = state.objectStoreSession()->newRequest(fiber.storePin->attachment);
        vc::objects::FiberApprovalLayer layer;
        state.objectStoreSession()->readStore(fiber.storePin->attachment, [&](const StoreReader& reader, const auto&) {
            const auto revisions = reader.objectsAtVersions({fiber.storePin->reference()});
            const auto content = vc::objects::fiberPieceContent(reader, *revisions.front());
            const auto stored = vc::objects::readFiberApproval(reader, fiber.uuid);
            layer.fiberUuid = fiber.uuid;
            layer.generation = (stored ? stored->generation : 0) + 1;
            layer.fiberVersion = revisions.front()->version;
            layer.author = request.author;
            layer.savedAt = "2026-11-18T12:00:00Z";
            layer.host = "test-host";
            vc::objects::setFiberStretchState(layer.stretches, vc::objects::FiberApprovalAspect::Geometry,
                                              double(content.controlLineIndex[3]), double(content.controlLineIndex[4]),
                                              stretchState, content.linePoints);
        });
        for (const auto& table : vc::objects::fiberApprovalTables())
            request.operations.push_back(vc::objects::CreateExtensionTableOp{table});
        auto rows = vc::objects::fiberApprovalRows(layer);
        request.operations.push_back(vc::objects::AppendExtensionRowsOp{vc::objects::kFiberApprovalTable, {rows.header}});
        if (!rows.stretches.empty())
            request.operations.push_back(
                vc::objects::AppendExtensionRowsOp{vc::objects::kFiberApprovalStretchTable, rows.stretches});
        state.objectStoreSession()->submitDurably(fiber.storePin->attachment, request);
    };
    const auto approvePiece = [&](const Controller::StoredFiber& fiber) {
        writeLayer(fiber, vc::objects::FiberStretchState::Approved);
    };
    approvePiece(horizontal);
    approvePiece(vertical);
    solves.loadApprovals();
    require(solves.anchorSpan(residual.fiberA, residual.arcA).geometryApproved &&
                solves.anchorSpan(residual.fiberB, residual.arcB).geometryApproved,
            "both covering pieces are approved now");

    const auto approved = solves.setConstraintApproval(residual, true);
    require(approved.isEmpty(), "the crossing is approved once its pieces are");
    require(solves.constraintApproved(residual), "the approval is read back");

    // Back to undefined: the next generation of the layer decides.
    writeLayer(horizontal, vc::objects::FiberStretchState::Undefined);
    solves.loadApprovals();
    require(!solves.anchorSpan(residual.fiberA, residual.arcA).geometryApproved,
            "the revoked piece is unapproved again");
    std::cout << "scaffold approval dock rules ok" << std::endl;
}
