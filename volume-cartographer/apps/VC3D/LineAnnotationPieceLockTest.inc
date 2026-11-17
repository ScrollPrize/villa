// Piece approval of a stored fiber: lock controls, the lock set, the save
// guard and crop, the approval rows (LineAnnotationPieceLock.hpp). Included
// in runObjectStoreControllerTests (VC_TESTING only).
{
    namespace la = vc3d::line_annotation;
    using Aspect = la::FiberPieceAspect;
    QTemporaryDir directory;
    const auto storePath = fs::path(directory.path().toStdString()) / "pieces.vcobj";
    OpenOptions options; options.author = "piece-test";
    { auto db = ObjectStore::create(storePath, options); }
    CState state;
    Controller controller(&state, nullptr, nullptr, nullptr);
    controller._errorDialogsSuppressed = true;
    openStoreAsProject(state, QString::fromStdString(storePath.string()), options.author.c_str());
    auto resolve = [&](const std::string& uuid) {
        auto fibers = controller.readStoredFibers(std::vector<std::string>{uuid}, true);
        if (fibers.empty()) throw std::runtime_error("piece test: fiber missing");
        return fibers.front();
    };
    // Eight controls every 10 vx, one line point per vx, a traced span descriptor on each span.
    Controller::StoredFiber fiber;
    fiber.id = controller.nextFiberId(); fiber.fileName = "piece.json";
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
    require(fiber.controlPoints.size() == 8 && fiber.controlPoints[3].storeId >= 0, "piece test fiber stored with ids");

    // The session-side content fingerprints exactly as the store does.
    {
        const auto sessionContent = la::pieceContentOfStoredFiber(fiber);
        std::optional<vc::objects::FiberPieceContent> storeContent;
        state.objectStoreSession()->readStore(fiber.storePin->attachment, [&](const StoreReader& reader, const auto&) {
            const auto revisions = reader.objectsAtVersions({fiber.storePin->reference()});
            storeContent = vc::objects::fiberPieceContent(reader, *revisions.front());
        });
        require(storeContent && storeContent->controlIds == sessionContent.controlIds &&
                storeContent->controlLineIndex == sessionContent.controlLineIndex, "stored and session content agree");
        const auto range = *vc::objects::fiberPieceRange(sessionContent, fiber.controlPoints[2].storeId,
                                                         fiber.controlPoints[5].storeId, fiber.controlPoints[3].storeId);
        const auto a = vc::objects::fingerprintFiberPiece(sessionContent, range, Aspect::Geometry);
        const auto b = vc::objects::fingerprintFiberPiece(*storeContent, range, Aspect::Geometry);
        require(a.spans == b.spans && a.controls == b.controls, "session fingerprints equal the store's");
    }

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
        controller.refreshPieceLockState(*live, true);
        return live;
    };
    const auto spansLocked = [](const la::PieceLockState& lock) {
        std::vector<size_t> locked;
        for (size_t s = 0; s < lock.lockedSpans.size(); ++s) if (lock.lockedSpans[s]) locked.push_back(s);
        return locked;
    };
    const auto storeVersion = [&](const std::shared_ptr<Controller::LineAnnotationSession>& live) {
        return live->storePin->reference().version;
    };

    // Piece A planned on two line points inside spans: lock controls appear in
    // the session only, nothing is saved until the first edit.
    auto sessionA = makeSession(fiber, "piece-a");
    require(spansLocked(sessionA->pieceLock).empty(), "an unapproved fiber locks nothing");
    QString planError;
    la::PiecePlan planA;
    planA.a.lineIndex = 25; planA.b.lineIndex = 55;
    require(controller.applyPiecePlan(*sessionA, planA, &planError), planError.toStdString().c_str());
    require(sessionA->controlPoints.size() == 10, "two lock controls inserted");
    require(sessionA->pieceLock.openedControls && sessionA->pieceLock.openedControls->first == 3 &&
            sessionA->pieceLock.openedControls->second == 7, "the piece spans the inserted locks");
    require(sessionA->pieceLock.pendingFirstEdit && sessionA->pieceLock.savesHeld(), "saves are held before the first edit");
    require(la::isLockControl(sessionA->controlPoints[3]) && sessionA->controlPoints[3].direction &&
            sessionA->controlPoints[3].optimizedIndex == 25 && sessionA->controlPoints[7].optimizedIndex == 55,
            "lock controls sit on their line points with a direction");
    require(spansLocked(sessionA->pieceLock) == std::vector<size_t>{0, 1, 2, 7, 8}, "everything outside the piece is locked");
    require(la::dimmedLineRanges(sessionA->pieceLock, sessionA->controlPoints, 71) ==
            std::vector<std::pair<double, double>>{{0.0, 25.0}, {55.0, 70.0}}, "the outside ranges are dimmed");
    {
        const auto request = controller.makeFiberModeOptimizationRequest(*sessionA, true);
        require(request.lockedSpans == std::vector<size_t>{0, 1, 2, 7, 8} && request.retainLeftTail && request.retainRightTail,
                "the solver request fixes the outside spans and both tails");
    }
    const auto versionBefore = storeVersion(sessionA);
    controller.saveSessionAsFiber(*sessionA, false);
    require(storeVersion(sessionA) == versionBefore && resolve(fiber.uuid).controlPoints.size() == 8,
            "opening a piece without an edit writes nothing");
    // Refusals outside, acceptance inside; the first accepted edit releases the save hold.
    {
        const std::array<size_t, 1> outside{1}, inside{4}, lock{3};
        require(controller.refuseLockedEdit(*sessionA, outside, {}), "an outside span is refused");
        require(controller.refuseLockedEdit(*sessionA, {}, lock), "a lock control is refused");
        require(sessionA->pieceLock.savesHeld(), "a refused edit keeps the hold");
        require(!controller.refuseLockedEdit(*sessionA, inside, {}), "an inside span is allowed");
        require(!sessionA->pieceLock.savesHeld(), "the first edit releases the hold");
    }
    // The edit: move a line point and a control inside the piece, then save.
    sessionA->controlPoints[5].volumePoint[1] += 1.0;
    sessionA->optimizedLine.points[40].position[1] += 1.0;
    const auto failuresBefore = controller._fiberSaveFailureCount;
    controller.saveSessionAsFiber(*sessionA, false);
    require(controller._fiberSaveFailureCount == failuresBefore && storeVersion(sessionA) > versionBefore, "the first edit saves");
    fiber = resolve(fiber.uuid);
    require(fiber.controlPoints.size() == 10, "lock controls are stored");
    require(std::count_if(fiber.controlPoints.begin(), fiber.controlPoints.end(), [](const auto& cp) {
        return std::find(cp.tags.begin(), cp.tags.end(), vc::objects::kFiberLockControlPointTag) != cp.tags.end(); }) == 2,
            "both lock tags are stored");
    require(!sessionA->pieceLock.pendingFirstEdit && sessionA->controlPoints[3].storeId >= 0 && sessionA->controlPoints[7].storeId >= 0,
            "the session reconciles the lock controls' ids after the commit");
    const int64_t lockA = sessionA->controlPoints[3].storeId, lockB = sessionA->controlPoints[7].storeId;
    // Approve piece A.
    {
        const auto error = controller.setFiberPieceApproval(fiber.id, Aspect::Geometry, true);
        require(!error, error ? error->toStdString().c_str() : "approved");
        const auto boxes = la::pieceApprovalBoxState(sessionA->pieceLock);
        require(boxes.pieceOpen && boxes.approved[0] && !boxes.approved[1], "the geometry box is checked, normals not");
        require(sessionA->pieceLock.states.live.size() == 1 && sessionA->pieceLock.states.live[0].lockCpA == lockA &&
                sessionA->pieceLock.states.live[0].lockCpB == lockB, "one live row covers the piece between its locks");
    }
    const std::string approver = sessionA->pieceLock.states.live[0].author;
    require(!approver.empty(), "the approval carries its author");
    controller._panes.clear();

    // The fiber opened normally: the approved spans are locked, the rest is free.
    fiber = resolve(fiber.uuid);
    auto plain = makeSession(fiber, "plain");
    require(spansLocked(plain->pieceLock) == std::vector<size_t>{3, 4, 5, 6}, "a normal open locks the approved spans");
    require(plain->pieceLock.lockedControls[3] && plain->pieceLock.lockedControls[7] && !plain->pieceLock.lockedControls[4],
            "lock controls are locked, interior controls are not");
    {
        const std::array<size_t, 1> approved{4}, free{1};
        require(controller.refuseLockedEdit(*plain, approved, {}), "an approved span is refused");
        require(!controller.refuseLockedEdit(*plain, free, {}), "a free span is allowed");
        const auto request = controller.makeFiberModeOptimizationRequest(*plain, true);
        require(request.lockedSpans == std::vector<size_t>{3, 4, 5, 6} && !request.retainLeftTail && !request.retainRightTail,
                "a normal open fixes the approved spans and leaves the tails");
    }
    // A change outside the approved piece saves; a change inside (bypassing the predicate) is refused by the commit.
    plain->optimizedLine.points[5].position[1] += 1.0;
    {
        const auto failures = controller._fiberSaveFailureCount;
        const auto version = storeVersion(plain);
        controller.saveSessionAsFiber(*plain, false);
        require(controller._fiberSaveFailureCount == failures && storeVersion(plain) > version, "a change outside the approved piece saves");
    }
    plain->optimizedLine.points[45].position[1] += 1.0;
    {
        const auto failures = controller._fiberSaveFailureCount;
        const auto version = storeVersion(plain);
        controller.saveSessionAsFiber(*plain, false);
        require(controller._fiberSaveFailureCount > failures && storeVersion(plain) == version, "a change inside the approved piece is refused by the commit");
        plain->optimizedLine.points[45].position[1] -= 1.0;
    }
    controller._panes.clear();

    // Piece B overlaps A's right end: its first edit crops A, B's approval then covers the overlap.
    fiber = resolve(fiber.uuid);
    auto sessionB = makeSession(fiber, "piece-b");
    la::PiecePlan planB;
    planB.a.lineIndex = 45; planB.b.fiberEnd = true;
    require(controller.applyPiecePlan(*sessionB, planB, &planError), planError.toStdString().c_str());
    require(sessionB->controlPoints.size() == 11 && sessionB->pieceLock.openedControls->second == 10, "B runs from its new lock to the fiber end");
    {
        const auto request = controller.makeFiberModeOptimizationRequest(*sessionB, true);
        require(request.retainLeftTail && !request.retainRightTail, "an end piece keeps only the locked side's tail");
    }
    {
        const std::array<size_t, 1> inside{7};
        require(!controller.refuseLockedEdit(*sessionB, inside, {}), "B edits inside its range");
    }
    sessionB->controlPoints[9].volumePoint[1] += 1.0;
    sessionB->optimizedLine.points[60].position[1] += 1.0;
    {
        const auto failures = controller._fiberSaveFailureCount;
        controller.saveSessionAsFiber(*sessionB, false);
        require(controller._fiberSaveFailureCount == failures, "B's first edit saves");
    }
    fiber = resolve(fiber.uuid);
    const int64_t lockMid = sessionB->controlPoints[6].storeId;
    require(lockMid >= 0 && sessionB->pieceLock.states.live.size() == 1 && sessionB->pieceLock.states.live[0].lockCpA == lockA &&
            sessionB->pieceLock.states.live[0].lockCpB == lockMid && sessionB->pieceLock.states.live[0].author == approver,
            "A's approval is cropped to B's new lock and keeps its stamp");
    require(sessionB->pieceLock.states.approvedSpans[0] == std::vector<size_t>{3, 4, 5}, "only A's remaining spans stay approved");
    require(sessionB->pieceLock.states.releasableLockCpIds == std::set<int64_t>{lockB}, "A's old right lock is releasable");
    require(!la::isLockControl(sessionB->controlPoints[8]), "the released lock control lost its tag in the session");
    {
        const auto error = controller.setFiberPieceApproval(fiber.id, Aspect::Geometry, true);
        require(!error, error ? error->toStdString().c_str() : "B approved");
        require(sessionB->pieceLock.states.approvedSpans[0] == std::vector<size_t>{3, 4, 5, 6, 7, 8, 9}, "B's approval covers the overlap and the end");
        const auto normals = controller.setFiberPieceApproval(fiber.id, Aspect::Normals, true);
        require(!normals && la::pieceApprovalBoxState(sessionB->pieceLock).approved[1], "normals approve separately");
        const auto revoke = controller.setFiberPieceApproval(fiber.id, Aspect::Geometry, false);
        require(!revoke && sessionB->pieceLock.states.approvedSpans[0] == std::vector<size_t>{3, 4, 5}, "revoking B uncovers its spans");
        require(la::pieceApprovalBoxState(sessionB->pieceLock).approved[1], "the normals approval survives the geometry revocation");
    }
    controller._panes.clear();
    // Growth past an end piece stays uncovered until re-approval.
    fiber = resolve(fiber.uuid);
    require(fiber.controlPoints.size() == 11, "the released lock tag is dropped at the save");
    require(std::count_if(fiber.controlPoints.begin(), fiber.controlPoints.end(), [](const auto& cp) {
        return std::find(cp.tags.begin(), cp.tags.end(), vc::objects::kFiberLockControlPointTag) != cp.tags.end(); }) == 2,
            "two lock controls remain tagged");
    state.closeProject();
    std::cout << "piece lock PASS" << std::endl;
}
