// The approval window of a stored fiber: a scaffold solve's normals as the
// baseline, normal points and stretch states stored as the fiber's approval
// layer (docs/fiber_approval_workflow.md). Included in
// runObjectStoreControllerTests (VC_TESTING only).
{
    namespace la = vc3d::line_annotation;
    QTemporaryDir directory;
    const auto storePath = fs::path(directory.path().toStdString()) / "approval.vcobj";
    OpenOptions options; options.author = "approval-test";
    { auto db = ObjectStore::create(storePath, options); }
    CState state;
    Controller controller(&state, nullptr, nullptr, nullptr);
    controller._errorDialogsSuppressed = true;
    openStoreAsProject(state, QString::fromStdString(storePath.string()), options.author.c_str());
    auto resolve = [&](const std::string& uuid) {
        auto fibers = controller.readStoredFibers(std::vector<std::string>{uuid}, true);
        if (fibers.empty()) throw std::runtime_error("approval test: fiber missing");
        return fibers.front();
    };
    // Eight controls every 10 vx, one line point per vx, a traced span descriptor on each span.
    Controller::StoredFiber fiber;
    fiber.id = controller.nextFiberId(); fiber.fileName = "approval.json";
    fiber.username = "test"; fiber.startedAt = "20260101"; fiber.sequence = 1;
    for (int i = 0; i < 8; ++i) {
        la::StoredControlPoint control{cv::Vec3d{10.0 * i, 0, 1}};
        control.direction = cv::Vec3d{1, 0, 0};
        if (i < 7) { la::FiberTraceSegmentMetadata span; control.segmentToNext = span; }
        fiber.controlPoints.push_back(control);
    }
    for (int x = 0; x <= 70; ++x) fiber.linePoints.emplace_back(double(x), 0, 1);
    std::vector<Controller::FiberSaveSnapshot> batch{controller.makeFiberSaveSnapshot(fiber)};
    controller.commitFiberStoreAction(batch);
    fiber = resolve(batch.front().fiber.uuid);
    require(fiber.controlPoints.size() == 8 && fiber.controlPoints[3].storeId >= 0, "approval test fiber stored with ids");

    // A live session of the stored fiber, as the panes build one.
    const auto makeSession = [&](const Controller::StoredFiber& stored, const char* name) {
        auto live = std::make_shared<Controller::LineAnnotationSession>();
        live->surfaceName = name; live->fiberId = stored.id; live->fiberUuid = stored.uuid;
        live->fiberFileName = stored.fileName; live->fiberUsername = stored.username;
        live->fiberStartedAt = stored.startedAt; live->fiberSequence = stored.sequence;
        live->storePin = stored.storePin; live->graphCommit = stored.graphCommit;
        live->coordinateIdentity = stored.coordinateIdentity; live->frame = stored.frame;
        live->fiberOptimizationMode = stored.optimizationMode;
        live->suppressGeneratedViews = true; live->suppressErrorDialogs = true;
        for (size_t i = 0; i < stored.controlPoints.size(); ++i) {
            const auto& cp = stored.controlPoints[i];
            la::LineControlPoint control(cp[0], cp, i == 0, int(cp[0]));
            control.storeId = cp.storeId; control.segmentToNext = cp.segmentToNext;
            control.direction = cp.direction; control.tags = cp.tags;
            control.displayNormal = cp.displayNormal; control.displayNormalSource = cp.displayNormalSource;
            live->controlPoints.push_back(control);
        }
        for (const auto& point : stored.linePoints) { vc::lasagna::LinePoint line; line.position = point; live->optimizedLine.points.push_back(line); }
        live->optimizationState = Controller::SessionOptimizationState::Optimized;
        Controller::PaneRecord pane; pane.surfaceName = live->surfaceName; pane.session = live;
        controller._panes.push_back(pane);
        return live;
    };

    // A fiber under review against a scaffold solve takes the solve's normals
    // as its baseline. The session's line carries extrapolated tails the
    // solve never read: they must not decide whether this is the solve's line.
    {
        controller._panes.clear();
        fiber = resolve(fiber.uuid);
        auto live = makeSession(fiber, "solve-review");
        auto solver = std::make_shared<Controller::SolverLineNormals>();
        for (const auto& point : live->optimizedLine.points) {
            solver->points.emplace_back(float(point.position[0]), float(point.position[1]), float(point.position[2]));
            solver->normals.emplace_back(0.f, 0.f, 1.f);
        }
        controller.setSolverLineNormals(live->fiberId, solver);
        const auto annotated = live->optimizedLine.points.size();
        require(controller.orientedLineNormalsForSession(*live).size() == annotated, "the solve's normals are the baseline of its own line");
        // Tails of 300 points to either side, running 300 voxels and more away from the line.
        std::vector<vc::lasagna::LinePoint> withTails;
        for (int k = 300; k >= 1; --k) { vc::lasagna::LinePoint tail; tail.position = cv::Vec3d(-double(k), 300.0 + k, 1); withTails.push_back(tail); }
        for (const auto& point : live->optimizedLine.points) withTails.push_back(point);
        for (int k = 1; k <= 300; ++k) { vc::lasagna::LinePoint tail; tail.position = cv::Vec3d(70.0 + k, 300.0 + k, 1); withTails.push_back(tail); }
        live->optimizedLine.points = withTails;
        for (auto& control : live->controlPoints) control.linePosition += 300.0;
        const auto normals = controller.orientedLineNormalsForSession(*live);
        require(normals.size() == annotated + 600 && normals[300 + 35] == cv::Vec3f(0.f, 0.f, 1.f) && normals.front() == cv::Vec3f(0.f, 0.f, 1.f),
                "tails past the end controls do not make the solve's normals unusable");
        // The annotated part itself somewhere else is still refused.
        for (auto& point : live->optimizedLine.points) point.position[1] += 500.0;
        bool refused = false;
        try { (void)controller.orientedLineNormalsForSession(*live); } catch (const std::runtime_error&) { refused = true; }
        require(refused, "a line that is not the solve's is refused");
        controller.setSolverLineNormals(live->fiberId, nullptr);
    }
    // The approval window: normal points and stretch states on a fiber that
    // is never changed, stored as the fiber's approval layer.
    {
        controller._panes.clear();
        fiber = resolve(fiber.uuid);
        const auto versionBefore = fiber.storePin->reference().version;
        auto live = makeSession(fiber, "approval");
        auto solver = std::make_shared<Controller::SolverLineNormals>();
        for (const auto& point : live->optimizedLine.points) {
            solver->points.emplace_back(float(point.position[0]), float(point.position[1]), float(point.position[2]));
            solver->normals.emplace_back(0.f, 0.f, 1.f);
        }
        controller.setSolverLineNormals(live->fiberId, solver);
        live->readOnly = true;
        live->suppressFiberSave = true;
        Controller::ApprovalOpen open;
        open.runUuid = "run-under-test";
        QString why;
        require(controller.loadApprovalLayer(*live, open, &why), "the approval layer of a fiber without one opens");
        require(live->approvalMode && live->approvalSelection.has_value(), "the annotated line is the first stretch");
        require(controller.refreshApprovalViews(*live), "the solve's normals are shown");
        const auto controlsBefore = live->controlPoints.size();
        const auto lineBefore = live->optimizedLine.points;
        const auto readLayer = [&]() {
            std::optional<vc::objects::FiberApprovalLayer> layer;
            state.objectStoreSession()->readStore(fiber.storePin->attachment, [&](const StoreReader& reader, const auto&) {
                layer = vc::objects::readFiberApproval(reader, fiber.uuid);
            });
            return layer;
        };
        // A normal point 30 degrees about the line, between two control points of the fiber.
        const double position = live->controlPoints[3].linePosition + 4.0;
        const cv::Vec3d wanted(0.0, std::sin(std::acos(-1.0) / 6.0), std::cos(std::acos(-1.0) / 6.0));
        require(controller.setFiberApprovalNormalPoint(fiber.id, position, wanted), "a normal point is stored");
        require(live->controlPoints.size() == controlsBefore + 1, "the normal point is a control point of the window only");
        const auto shown = vc::fiber_tracer::displayVectorAt(live->shownLineNormals, position);
        require(std::abs(cv::normalize(shown).dot(wanted)) > 0.999, "the window shows the wanted normal at the normal point");
        auto layer = readLayer();
        require(layer && layer->generation == 1 && layer->normalPoints.size() == 1 && layer->stretches.empty() &&
                    layer->baselineRun == "run-under-test" && layer->lineNormals.size() == lineBefore.size(),
                "the layer holds the normal point and the shown normals");
        require(std::abs(layer->normalPoints[0].position - position) < 1e-9 &&
                    std::abs(layer->normalPoints[0].normal[1] - wanted[1]) < 1e-6, "the normal point is stored as set");
        // The correction is interpolated between normal points only: the
        // fiber's own control points do not end it, so one normal point turns
        // the whole line and a second one, holding the solve's normal, ends it.
        require(std::abs(cv::normalize(vc::fiber_tracer::displayVectorAt(live->shownLineNormals, position + 20.0)).dot(wanted)) > 0.999,
                "one normal point corrects the whole line");
        require(controller.setFiberApprovalNormalPoint(fiber.id, position + 20.0, cv::Vec3d(0, 0, 1)), "a second normal point");
        require(std::abs(vc::fiber_tracer::displayVectorAt(live->shownLineNormals, position + 20.0)[2]) > 0.999 &&
                    std::abs(cv::normalize(vc::fiber_tracer::displayVectorAt(live->shownLineNormals, position)).dot(wanted)) > 0.999,
                "the second normal point ends the correction");
        require(controller.setFiberApprovalNormalPoint(fiber.id, position + 20.0, std::nullopt, true), "and is deleted again");
        // A normal point may lie on a control point of the fiber, which stays as it is.
        const auto onControl = live->controlPoints[2].linePosition;
        require(controller.setFiberApprovalNormalPoint(fiber.id, onControl, wanted) &&
                    controller.setFiberApprovalNormalPoint(fiber.id, onControl, std::nullopt, true),
                "a normal point on a control point of the fiber is set and deleted");
        // Normals are worked on inside the selected stretch only, its ends included.
        {
            const auto whole = live->approvalSelection;
            const auto generation = live->approvalGeneration;
            live->approvalSelection = std::pair{position - 10.0, position + 10.0};
            require(!controller.setFiberApprovalNormalPoint(fiber.id, position + 15.0, wanted) &&
                        live->approvalGeneration == generation, "a normal point outside the selected stretch is refused");
            require(controller.setFiberApprovalNormalPoint(fiber.id, position + 10.0, wanted), "one on its end is set");
            live->approvalSelection = std::pair{0.0, position};
            require(!controller.setFiberApprovalNormalPoint(fiber.id, position + 10.0, std::nullopt, true),
                    "a normal point outside the selected stretch is not deleted");
            live->approvalSelection = whole;
            require(controller.setFiberApprovalNormalPoint(fiber.id, position + 10.0, std::nullopt, true), "inside it is");
        }
        // States, once per aspect, over arbitrary positions.
        require(controller.setFiberApprovalState(fiber.id, position - 2.5, position + 3.25, 1, 1), "normals approved");
        require(controller.setFiberApprovalState(fiber.id, 5.5, 12.0, 0, 2), "geometry broken");
        layer = readLayer();
        require(layer && layer->generation == 9 && layer->stretches.size() == 2, "both states are stored");
        const auto status = vc::objects::fiberApprovalStatus(*layer, live->approvalLine);
        require(status.staleStretches == 0 && status.staleNormalPoints == 0 && status.lineNormalsUsable, "the layer is current");
        const auto approved = vc::objects::approvedFiberLineNormals(*layer, status, live->approvalLine.size());
        const auto at = size_t(std::llround(position));
        require(approved.size() == lineBefore.size() && std::abs(approved[at][1] - wanted[1]) < 0.02 &&
                    approved[at + 10][2] == 0, "the solver gets the shown normals on the approved stretch only");
        require(controller.fiberHasApproval(fiber.uuid), "the fiber is known to carry approvals");
        // Undefined clears; deleting the normal point removes it.
        require(controller.setFiberApprovalState(fiber.id, 0.0, double(lineBefore.size() - 1), 0, 0), "geometry undefined again");
        require(controller.setFiberApprovalNormalPoint(fiber.id, position + 0.2, std::nullopt, true), "the normal point is deleted");
        layer = readLayer();
        require(layer && layer->generation == 11 && layer->normalPoints.empty() && layer->stretches.size() == 1 &&
                    live->controlPoints.size() == controlsBefore, "the newest generation is the layer");
        // Nothing of the fiber changed.
        const auto after = resolve(fiber.uuid);
        require(after.storePin->reference().version == versionBefore && after.controlPoints.size() == fiber.controlPoints.size() &&
                    after.linePoints.size() == fiber.linePoints.size(), "the fiber itself is untouched");
        require(live->optimizedLine.points.size() == lineBefore.size() && live->approvalChanged, "the line is as it was");
        // Reopened, the layer comes back.
        controller._panes.clear();
        auto again = makeSession(after, "approval-again");
        controller.setSolverLineNormals(again->fiberId, solver);
        require(controller.loadApprovalLayer(*again, open, &why) && again->approvalStretches.size() == 1 &&
                    again->approvalGeneration == 11, "a reopened window shows the stored layer");
        controller.setSolverLineNormals(again->fiberId, nullptr);
        controller._panes.clear();
    }
    state.closeProject();
    std::cout << "approval window PASS" << std::endl;
}
