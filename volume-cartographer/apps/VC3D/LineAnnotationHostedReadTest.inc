// Included inside runObjectStoreControllerTests(). Fiber read validation:
// the commit-sequence fast path, numbered-version validation, dependency
// provenance and batched decodes, on a local store and through a hosted
// vc_object_server (VC3D_OBJECT_SERVER_BIN, default: next to this binary).
{
    namespace la = vc3d::line_annotation;
    QTemporaryDir directory;
    const auto storePath = fs::path(directory.path().toStdString()) / "reads.vcobj";
    OpenOptions readOptions; readOptions.author = "fiber-read-test";
    { auto db = ObjectStore::create(storePath, readOptions); }
    // Three fibers A, B, C; A–B linked (approved), C unlinked.
    auto makeFibers = [&](Controller& controller) {
        Controller::StoredFiber a;
        a.id = controller.nextFiberId(); a.fileName = "read_a.json";
        a.username = "test"; a.startedAt = "20260101"; a.sequence = 1;
        for (int x : {0, 10, 20}) { a.controlPoints.emplace_back(cv::Vec3d{x * 1., 0, 1}); a.linePoints.emplace_back(x * 1., 0, 1); }
        auto b = a; b.uuid = vc::objects::util::newUuid(); b.id = controller.nextFiberId(); b.fileName = "read_b.json";
        for (auto& p : b.controlPoints) p[1] = 5;
        for (auto& p : b.linePoints) p[1] = 5;
        auto c = a; c.uuid = vc::objects::util::newUuid(); c.id = controller.nextFiberId(); c.fileName = "read_c.json";
        for (auto& p : c.controlPoints) p[1] = 9;
        for (auto& p : c.linePoints) p[1] = 9;
        std::vector<Controller::FiberSaveSnapshot> batch{controller.makeFiberSaveSnapshot(a), controller.makeFiberSaveSnapshot(b),
                                                         controller.makeFiberSaveSnapshot(c)};
        controller.commitFiberStoreAction(batch);
        a = batch[0].fiber; b = batch[1].fiber; c = batch[2].fiber;
        Controller::FiberBranchRef link;
        link.controlPointIndex = 1; link.branchControlPointIndex = 1; link.branchFiberId = b.id; link.branchFileName = b.fileName;
        link.controlPointPosition = a.controlPoints[1]; link.branchControlPointPosition = b.controlPoints[1];
        link.controlPointDirection = {1, 0, 0}; link.branchControlPointDirection = {1, 0, 0};
        // Created pending, approved by a second save (an approval is of a stored link).
        link.pending = true;
        a.branches.push_back(link);
        std::swap(link.controlPointPosition, link.branchControlPointPosition);
        link.branchFiberId = a.id; link.branchFileName = a.fileName;
        b.branches.push_back(link);
        batch = {controller.makeFiberSaveSnapshot(a), controller.makeFiberSaveSnapshot(b)};
        controller.commitFiberStoreAction(batch);
        a = batch[0].fiber; b = batch[1].fiber;
        a.branches.front().pending = false; b.branches.front().pending = false;
        batch = {controller.makeFiberSaveSnapshot(a), controller.makeFiberSaveSnapshot(b)};
        controller.commitFiberStoreAction(batch);
        return std::array<std::string, 3>{a.uuid, b.uuid, c.uuid};
    };
    std::array<std::string, 3> uuids;
    {
        CState state;
        Controller controller(&state, nullptr, nullptr, nullptr);
        controller._errorDialogsSuppressed = true;
        openStoreAsProject(state, QString::fromStdString(storePath.string()), "fiber-read-test");
        uuids = makeFibers(controller);
        // The hosted part serves a consistent copy of this initial state.
        vc::objects::copyProject(*state.objectStoreSession(), *state.storeFor("fiber"),
                                 fs::path(directory.path().toStdString()) / "served.vcobj", "fiber read test", readOptions);
        const auto& [aUuid, bUuid, cUuid] = uuids;
        const auto read = [&](const std::string& uuid) {
            auto fibers = controller.storedFibersShared(std::vector<std::string>{uuid}, true);
            return fibers.empty() ? Controller::StoredFiberPtr{} : fibers.front();
        };
        // Fast path: an unchanged store answers from the cache with one status.
        (void)read(aUuid);
        auto fast = controller._storedFiberFastValidations, slow = controller._storedFiberSlowValidations;
        const auto again = read(aUuid);
        require(again && controller._storedFiberFastValidations == fast + 1 && controller._storedFiberSlowValidations == slow,
                "an unchanged store position answers from the cache");
        require(again->branches.size() == 1 && !again->branches.front().pending, "A is linked to B (approved)");
        // A relationship-only change by another writer (no event delivered):
        // A's record generation is unchanged, its numbered version is not.
        const auto edgeUuid = again->branches.front().edgeUuid;
        // A second writer on the same file, expecting both endpoints' heads.
        const auto checkedWrite = [&](const std::function<void(ObjectStore&, WriteTxn&)>& edit) {
            auto db = ObjectStore::open(storePath, readOptions);
            RecordSelection expected;
            {
                auto txn = db.read();
                for (const auto& uuid : {aUuid, bUuid})
                    expected[uuid] = db.currentObjectRevision(txn, uuid)->reference();
            }
            auto txn = db.write(expected);
            edit(db, txn);
            txn.commit();
        };
        checkedWrite([&](ObjectStore& db, WriteTxn& txn) {
            db.retainRelationship(txn, edgeUuid);
            db.setApproval(txn, edgeUuid, "pending");
        });
        const auto relinked = read(aUuid);
        require(relinked && relinked->branches.size() == 1 && relinked->branches.front().pending,
                "a relationship-only change is seen through A's numbered version");
        // Duplicates, aliases and missing UUIDs keep the slow path's filter and order.
        auto upper = aUuid; std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        const auto mixed = controller.storedFibersShared(
            std::vector<std::string>{cUuid, aUuid, vc::objects::util::newUuid(), upper, cUuid}, false);
        require(mixed.size() == 2 && mixed[0]->uuid == cUuid && mixed[1]->uuid == aUuid,
                "requested first-occurrence order, canonical de-duplication, missing skipped");
        // The same order and de-duplication from the fast path (no missing
        // UUID: missing objects are not cached, so they always list).
        fast = controller._storedFiberFastValidations; slow = controller._storedFiberSlowValidations;
        const auto mixedAgain = controller.storedFibersShared(std::vector<std::string>{cUuid, aUuid, upper, cUuid}, false);
        require(controller._storedFiberFastValidations == fast + 1 && controller._storedFiberSlowValidations == slow,
                "aliases and duplicates of cached fibers take the fast path");
        require(mixedAgain.size() == 2 && mixedAgain[0]->uuid == cUuid && mixedAgain[1]->uuid == aUuid,
                "the fast path returns the slow path's order and filter");
        // Retiring B while keeping its recorded relationship: A's own head may
        // stay unchanged; its pinned link must still be re-evaluated.
        (void)read(aUuid);
        checkedWrite([&](ObjectStore& db, WriteTxn& txn) {
            db.retireObject(txn, bUuid);
            db.keepRecordedRelationshipsOf(txn, bUuid);
        });
        const auto afterRetire = read(aUuid);
        const auto uncached = controller.readStoredFibers(std::vector<std::string>{aUuid}, true);
        require(afterRetire && uncached.size() == 1 && afterRetire->branches.size() == uncached.front().branches.size(),
                "a retired peer's kept relationship decodes like an uncached read");
        const auto afterRetireAgain = read(aUuid);
        require(afterRetireAgain && afterRetireAgain->branches.size() == uncached.front().branches.size(),
                "and stays so on the fast path");
        require(!read(bUuid) && !read(bUuid), "a retired fiber is absent on both reads");
        // Batched decodes equal single decodes (same IDs: mapping already assigned).
        controller._storedFiberDecodeCache.clear(); controller._storedFiberDependencies.clear();
        const auto batched = controller.storedFibersShared(std::nullopt, true);
        for (const auto& fiber : batched) {
            controller._storedFiberDecodeCache.clear(); controller._storedFiberDependencies.clear();
            const auto single = read(fiber->uuid);
            require(single && single->id == fiber->id && single->controlPoints == fiber->controlPoints &&
                    single->linePoints == fiber->linePoints && single->branches.size() == fiber->branches.size() &&
                    single->storePin == fiber->storePin && single->tags == fiber->tags,
                    "batched and single decodes agree");
        }
        state.closeProject();
    }
    // Real-store equivalence of batch shapes (env-gated).
    if (const char* source = std::getenv("VC3D_FIBER_DECODE_TEST_STORE")) {
        QTemporaryDir copyDirectory;
        const auto copy = fs::path(copyDirectory.path().toStdString()) / "decode.vcobj";
        fs::copy_file(source, copy);
        CState state;
        Controller controller(&state, nullptr, nullptr, nullptr);
        controller._errorDialogsSuppressed = true;
        openStoreAsProject(state, QString::fromStdString(copy.string()), "decode-test", true);
        const auto all = controller.readStoredFibers(std::nullopt, true);  // assigns every runtime ID
        require(!all.empty(), "real store has fibers");
        std::mt19937 random(7);
        std::vector<std::string> order;
        for (const auto& fiber : all) order.push_back(fiber.uuid);
        const auto compare = [&](const Controller::StoredFiber& expected, const Controller::StoredFiber& actual) {
            bool same = expected.id == actual.id && expected.controlPoints == actual.controlPoints &&
                expected.linePoints == actual.linePoints && expected.storePin == actual.storePin &&
                expected.tags == actual.tags && expected.readOnly == actual.readOnly &&
                expected.editorError == actual.editorError && expected.branches.size() == actual.branches.size();
            for (size_t i = 0; same && i < expected.branches.size(); ++i) {
                const auto& x = expected.branches[i]; const auto& y = actual.branches[i];
                same = x.edgeUuid == y.edgeUuid && x.branchFiberId == y.branchFiberId &&
                       x.controlPointIndex == y.controlPointIndex && x.branchControlPointIndex == y.branchControlPointIndex &&
                       x.pending == y.pending && x.adjacent == y.adjacent && x.autoLink == y.autoLink;
            }
            return same;
        };
        std::map<std::string, const Controller::StoredFiber*> byUuid;
        for (const auto& fiber : all) byUuid[fiber.uuid] = &fiber;
        for (int round = 0; round < 6; ++round) {
            std::shuffle(order.begin(), order.end(), random);
            const size_t count = round == 0 ? 1 : std::min(order.size(), size_t(1) << round);
            const std::vector<std::string> sample(order.begin(), order.begin() + count);
            controller._storedFiberDecodeCache.clear(); controller._storedFiberDependencies.clear();
            for (const auto& fiber : controller.storedFibersShared(sample, true))
                require(compare(*byUuid.at(fiber->uuid), *fiber), "random batches decode like the all-fiber read");
        }
        state.closeProject();
    }
    // Hosted: a vc_object_server on a copy of the store, the controller's
    // fiber store attached through it; writes through a second hosted client
    // are interleaved at the read's stages.
    const auto serverBinary = [] {
        if (const char* path = std::getenv("VC3D_OBJECT_SERVER_BIN")) return fs::path(path);
        return fs::path(QCoreApplication::applicationDirPath().toStdString()) / "vc_object_server";
    }();
    if (fs::exists(serverBinary)) {
        const auto served = fs::path(directory.path().toStdString()) / "served.vcobj";
        // Serves `store` on a free port; returns the endpoint once it is ready.
        const auto startServer = [&](QProcess& server, const fs::path& store) {
            server.setProcessChannelMode(QProcess::MergedChannels);
            server.start(QString::fromStdString(serverBinary.string()),
                         {QString::fromStdString(store.string()), "--port", "0"});
            require(server.waitForStarted(10000), "object server starts");
            QByteArray output;
            QElapsedTimer started; started.start();
            const QRegularExpression ready("serving .* on 127\\.0\\.0\\.1:(\\d+)");
            while (started.elapsed() < 15000) {
                if (server.state() != QProcess::Running) break;
                server.waitForReadyRead(200);
                output += server.readAll();
                if (const auto match = ready.match(QString::fromUtf8(output)); match.hasMatch())
                    return "http://127.0.0.1:" + match.captured(1).toStdString();
            }
            throw std::runtime_error("object server did not become ready: " + output.toStdString());
        };
        QProcess server;
        const auto stopServer = qScopeGuard([&] { server.terminate(); server.waitForFinished(10000); });
        const auto endpoint = startServer(server, served);
        const auto local = fs::path(directory.path().toStdString()) / "project-local.vcobj";
        { auto db = ObjectStore::create(local, readOptions); }
        CState state;
        Controller controller(&state, nullptr, nullptr, nullptr);
        controller._errorDialogsSuppressed = true;
        openStoreAsProject(state, QString::fromStdString(local.string()), "fiber-read-test");
        const auto hosted = state.attachHostedStore(endpoint, "fiber-read-test");
        state.setStoreFor("fiber", hosted);
        ObjectStoreSession writer;
        HostedSessionOptions writerOptions; writerOptions.author = "fiber-read-test-writer";
        const auto other = writer.attachHosted(endpoint, writerOptions);
        const auto& [aUuid, bUuid, cUuid] = uuids;
        const auto read = [&](const std::string& uuid) {
            auto fibers = controller.storedFibersShared(std::vector<std::string>{uuid}, true);
            return fibers.empty() ? Controller::StoredFiberPtr{} : fibers.front();
        };
        auto submitOther = [&](std::vector<TransactionOperation> operations, const std::string& expectUuid) {
            auto request = writer.newRequest(other);
            writer.readStore(other, [&](const StoreReader& reader, const auto&) {
                request.expectedObjects[expectUuid] = reader.currentObject(expectUuid)->reference();
            });
            request.operations = std::move(operations);
            writer.submit(other, request);
        };
        auto first = read(aUuid);
        require(first && first->branches.size() == 1, "hosted decode of A with its link");
        auto fast = controller._storedFiberFastValidations;
        require(read(aUuid) && controller._storedFiberFastValidations == fast + 1, "hosted fast path");
        // Another client's commit while A is being decoded: that attempt is
        // discarded (its graph would straddle two store positions) and the
        // read is made again, at the new position.
        controller._storedFiberDecodeCache.clear(); controller._storedFiberDependencies.clear();
        bool changed = false;
        int64_t sequenceAfterChange = -1;
        controller._fiberReadTestStage = [&](const char* stage) {
            if (changed || std::string_view(stage) != "decode-heads") return;
            changed = true;
            submitOther({SetMetaOp{bUuid, "provenance_test", "true"}}, bUuid);
            writer.readStore(other, [&](const StoreReader& reader, const auto&) { sequenceAfterChange = reader.status().commitSequence; });
        };
        const auto afterChange = read(aUuid);
        controller._fiberReadTestStage = {};
        require(changed && afterChange && afterChange->graphCommit == sequenceAfterChange,
                "a read interleaved with another client's commit is made again at the new position");
        // A store that keeps committing: the read gives up.
        {
            controller._storedFiberDecodeCache.clear(); controller._storedFiberDependencies.clear();
            bool refused = false;
            controller._fiberReadTestStage = [&](const char* stage) {
                if (std::string_view(stage) == "decode-heads") submitOther({SetMetaOp{bUuid, "provenance_test", "again"}}, bUuid);
            };
            try { (void)read(aUuid); } catch (const std::exception&) { refused = true; }
            controller._fiberReadTestStage = {};
            require(refused, "a read that every attempt interleaves with a commit is refused");
            (void)read(aUuid);
        }
        const auto peer = controller.storedFibersShared(std::vector<std::string>{bUuid}, false);
        require(afterChange && afterChange->branches.size() == 1 && !peer.empty(), "A and B decode at the new position");
        // Any commit ends every cached decode (each carries the position its
        // graph was loaded at): C is decoded again after a change to A.
        {
            const auto cBefore = read(cUuid);
            fast = controller._storedFiberFastValidations;
            require(read(cUuid) == cBefore && controller._storedFiberFastValidations == fast + 1, "an unchanged position answers from the cache");
            submitOther({SetMetaOp{aUuid, "move_position", "true"}}, aUuid);
            const auto cAfter = read(cUuid);
            require(cAfter && cAfter != cBefore && cAfter->graphCommit && cBefore->graphCommit &&
                    *cAfter->graphCommit == *cBefore->graphCommit + 1, "a moved position decodes again, at that position");
        }

        // Bounded stale validation: inside one HostedStatusScope the fast path
        // answers from the scope's first status; a read in a nested event loop
        // validates with a status of its own, and the memo ends there.
        (void)read(cUuid);
        {
            const Controller::HostedStatusScope scope(controller);
            const auto cached = read(cUuid);
            submitOther({SetMetaOp{cUuid, "memo_test", "true"}}, cUuid);
            require(read(cUuid) == cached, "within one scope the memo answers as of its first status");
            Controller::StoredFiberPtr nested;
            QTimer::singleShot(0, [&] { nested = read(cUuid); });
            QEventLoop loop;
            QTimer::singleShot(0, &loop, &QEventLoop::quit);
            loop.exec();
            require(nested && nested != cached, "a read inside a nested event loop sees the other writer's commit");
            require(read(cUuid) == nested, "and the scope's memo does not come back after the nested loop");
        }
        // A validation's status becomes the memo only if nothing else the read
        // observed contradicted it: here the decode of A sees a commit made
        // after the validation's status, so a later cached read of C in the same
        // scope must not be answered from that older status.
        (void)read(aUuid);
        (void)read(cUuid);
        {
            const Controller::HostedStatusScope scope(controller);
            controller._storedFiberDecodeCache.erase(aUuid);
            controller._storedFiberDependencies.erase(aUuid);
            bool committed = false;
            controller._fiberReadTestStage = [&](const char* stage) {
                if (committed || std::string_view(stage) != "validate-status") return;
                committed = true;
                submitOther({SetMetaOp{cUuid, "candidate_test", "true"}}, cUuid);
            };
            const auto cBefore = controller._storedFiberDecodeCache.at(cUuid);
            (void)read(aUuid);
            controller._fiberReadTestStage = {};
            require(committed, "the commit ran between the validation status and the decode");
            const auto cAfter = read(cUuid);
            require(cAfter && cAfter != cBefore, "the decode's newer observation dropped the memo candidate");
        }
        // Attaching and detaching a store the controller does not read leaves
        // it alone; changing its fiber route resets it.
        {
            const auto unrelatedPath = fs::path(directory.path().toStdString()) / "unrelated.vcobj";
            { auto db = ObjectStore::create(unrelatedPath, readOptions); }
            const auto before = controller._packageGeneration;
            const auto unrelated = state.attachStore(unrelatedPath, "fiber-read-test", false);
            state.pollObjectStoreChanges();
            QCoreApplication::processEvents();
            require(controller._packageGeneration == before, "an unrelated attachment does not reset the controller");
            state.setStoreFor("fiber", unrelated);
            state.pollObjectStoreChanges();
            QCoreApplication::processEvents();
            require(controller._packageGeneration != before, "a fiber route change does");
            state.setStoreFor("fiber", hosted);
            state.pollObjectStoreChanges();
            QCoreApplication::processEvents();
        }
        // Route preparation is told which kinds change, so editors that read
        // other kinds are left alone (CWindow prepares line annotation only for
        // fiber/umbilicus).
        {
            std::vector<std::set<std::string>> prepared;
            state.prepareStoreKindsSwitch = [&](const std::set<std::string>& kinds) { prepared.push_back(kinds); return true; };
            const auto data = state.vpkg()->document().data();
            const nlohmann::json previous = data.at("selections").value("surface", nlohmann::json(nullptr));
            std::string someId;
            for (const auto& entry : data.at("stores"))
                if (entry.at("id") != previous) { someId = entry.at("id").get<std::string>(); break; }
            require(!someId.empty(), "the project references a second store");
            {
                state.setStoreSelection("surface", someId);
                state.setStoreSelection("surface", previous.is_string() ? previous.get<std::string>() : std::string());
                require(!prepared.empty() && prepared.front() == std::set<std::string>{"surface"},
                        "a surface route change prepares only for surfaces");
            }
            state.prepareStoreKindsSwitch = {};
        }
        // The store view: cached reads validate against the position the
        // controller's delivered events (and its own commits) brought it to,
        // without asking the store.
        {
            Controller viewing(&state, nullptr, nullptr, nullptr);
            viewing._errorDialogsSuppressed = true;
            viewing._fiberStoreViewEnabled = true;
            auto* session = state.objectStoreSession().get();
            const auto deliver = [&] { session->refresh(hosted); state.pollObjectStoreChanges(); };
            const auto view = [&](const std::string& uuid) {
                auto fibers = viewing.storedFibersShared(std::vector<std::string>{uuid}, true);
                return fibers.empty() ? Controller::StoredFiberPtr{} : fibers.front();
            };
            const auto sequence = [&] { return viewing._fiberStoreView ? viewing._fiberStoreView->sequence : int64_t(-1); };
            deliver();
            const auto a1 = view(aUuid);
            const auto c1 = view(cUuid);
            require(a1 && c1 && viewing._fiberStoreView, "the view starts at the attachment's last check");
            auto fast = viewing._storedFiberFastValidations;
            auto statusReads = viewing._storedFiberStatusReads;
            require(view(aUuid) == a1 && viewing._storedFiberFastValidations == fast + 1 &&
                    viewing._storedFiberStatusReads == statusReads, "a cached read answers from the view");
            // Another client's commit: seen when its event is delivered.
            submitOther({SetMetaOp{aUuid, "view_test", "true"}}, aUuid);
            require(view(aUuid) == a1, "another client's commit is not seen before its event");
            const auto before = sequence();
            deliver();
            const auto a2 = view(aUuid);
            require(sequence() == before + 1 && a2 && a2 != a1, "and is decoded again after it");
            const auto c1b = view(cUuid);
            require(c1b && c1b != c1 && c1b->graphCommit == sequence(),
                    "every cached decode ends with the commit: C is decoded again at the new position");
            // This controller's own commit directly after the view: the view
            // moves with it, what it changed is decoded once, and its event
            // finds nothing to do.
            {
                Controller::StoredFiber own = *view(cUuid);
                own.tags.push_back("view-own");
                const auto at = sequence();
                viewing.commitFibers({&own});
                require(sequence() == at + 1, "an own commit directly after the view moves it");
                const auto c2 = view(cUuid);
                require(c2 && c2 != c1b && c2->storePin == own.storePin && c2->graphCommit == at + 1 &&
                        std::find(c2->tags.begin(), c2->tags.end(), "view-own") != c2->tags.end(),
                        "the saved fiber is decoded at its new version and position");
                const auto a3 = view(aUuid);
                fast = viewing._storedFiberFastValidations;
                require(view(cUuid) == c2 && view(aUuid) == a3 && viewing._storedFiberFastValidations == fast + 2,
                        "and reads at that position answer from the cache");
                deliver();
                fast = viewing._storedFiberFastValidations;
                require(sequence() == at + 1 && view(cUuid) == c2 && view(aUuid) == a3 &&
                        viewing._storedFiberFastValidations == fast + 2, "its own event changes nothing");
                // Another client's commit this controller has not seen: a save
                // from the older position is refused; after its event, the
                // fiber read again saves.
                submitOther({SetMetaOp{aUuid, "view_between", "true"}}, aUuid);
                Controller::StoredFiber again = *view(cUuid);
                again.tags.push_back("view-own-2");
                bool refusedSave = false;
                try { viewing.commitFibers({&again}); } catch (const std::exception&) { refusedSave = true; }
                require(refusedSave, "a save that has not seen another client's commit is refused");
                deliver();
                again = *view(cUuid);
                again.tags.push_back("view-own-2");
                viewing.commitFibers({&again});
                const auto c3 = view(cUuid);
                require(sequence() == at + 3 && c3 && c3->storePin == again.storePin, "and succeeds from the current position");
            }
        }
        // Background publication: a placement's save runs on a worker; saves
        // made meanwhile queue behind it (one per session), a synchronous save
        // first takes its result, and tag edits queued behind it still run.
        {
            Controller publishing(&state, nullptr, nullptr, nullptr);
            publishing._errorDialogsSuppressed = true;
            publishing._backgroundPublication = true;
            auto* session = state.objectStoreSession().get();
            session->refresh(hosted); state.pollObjectStoreChanges();
            const auto stored = [&] {
                return publishing.readStoredFibers(std::vector<std::string>{cUuid}, true).at(0);
            };
            const auto openSession = [&](const Controller::StoredFiber& original, const std::string& name) {
            auto live = std::make_shared<Controller::LineAnnotationSession>();
            live->surfaceName = name;
            live->fiberId = original.id; live->fiberUuid = original.uuid;
            live->fiberFileName = original.fileName; live->fiberUsername = original.username;
            live->fiberStartedAt = original.startedAt; live->fiberSequence = original.sequence;
            live->storePin = original.storePin; live->coordinateIdentity = original.coordinateIdentity;
            live->graphCommit = original.graphCommit;
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
            publishing._panes.push_back(pane);
            return live;
            };
            const auto original = stored();
            const auto live = openSession(original, "publication-test");
            // A second open fiber (A, not linked to C): this controller's
            // saves of C must not leave its session behind the store.
            const auto other = openSession(publishing.readStoredFibers(std::vector<std::string>{aUuid}, true).at(0), "publication-other");
            const auto settle = [&](const char* what) {
                QElapsedTimer waited; waited.start();
                while ((publishing._publication || !publishing._afterPublication.empty() ||
                        publishing._publicationDrainQueued || !publishing._deferredControlEdits.empty()) &&
                       waited.elapsed() < 20000)
                    QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
                require(!publishing._publication && publishing._afterPublication.empty() &&
                        publishing._deferredControlEdits.empty(), what);
            };
            const auto failures = publishing._fiberSaveFailureCount;
            // One runs; two more saves of the same session queue as one.
            auto pin = live->storePin;
            live->fiberWidth += .25;
            publishing.saveSessionAsFiber(*live, false, true);
            require(publishing._publication && live->storePin == pin, "a background save returns before it is stored");
            live->fiberWidth += .25;
            publishing.saveSessionAsFiber(*live, false, true);
            live->fiberWidth += .25;
            publishing.saveSessionAsFiber(*live, false, true);
            require(publishing._afterPublication.size() == 1, "saves behind a publication queue once per session");
            settle("the publication and the queued save finish");
            {
                const auto before = publishing._fiberSaveFailureCount;
                other->fiberWidth += .25;
                publishing.saveSessionAsFiber(*other, false);
                require(publishing._fiberSaveFailureCount == before &&
                        publishing.readStoredFibers(std::vector<std::string>{aUuid}, true).at(0).width == other->fiberWidth,
                        "another open fiber still saves after this controller's saves of C");
                settle("nothing queued after the other fiber's save");
            }
            require(publishing._fiberSaveFailureCount == failures && live->storePin != pin &&
                    stored().width == live->fiberWidth && stored().storePin == live->storePin,
                    "the queued save stored the session as it was when it ran, at the pin the session holds");
            // A synchronous save while one runs takes its pin first (the
            // closing save's case): no version conflict.
            pin = live->storePin;
            live->fiberWidth += .25;
            publishing.saveSessionAsFiber(*live, false, true);
            require(bool(publishing._publication), "a second background save starts");
            live->fiberWidth += .25;
            publishing.saveSessionAsFiber(*live, false);
            require(!publishing._publication && publishing._fiberSaveFailureCount == failures &&
                    stored().width == live->fiberWidth && stored().storePin == live->storePin && live->storePin != pin,
                    "a synchronous save joins the running publication and saves on top of it");
            settle("nothing is left queued after the join");
            // A tag edit queued behind a publication that a synchronous save
            // then joins is still applied.
            {
                live->fiberWidth += .25;
                publishing.saveSessionAsFiber(*live, false, true);
                require(bool(publishing._publication), "a third background save starts");
                int replayed = 0;
                std::vector<size_t> at;
                require(publishing.deferControlEdit(*live, {size_t(1)}, [&](const std::vector<size_t>& indices) {
                            ++replayed; at = indices;
                        }), "a tag edit during a publication is queued");
                publishing.joinPublication();
                require(!publishing._publication && replayed == 0, "the join does not run it inside the caller");
                settle("the queued tag edit drains from the event loop");
                require(replayed == 1 && at == std::vector<size_t>{1}, "and is applied once to the control it named");
            }
            // A control-point tag edit (a break) is saved like a placement:
            // off the GUI thread, and stored when that save ends.
            {
                const auto before = publishing._fiberSaveFailureCount;
                const auto pinBefore = live->storePin;
                const auto tagged = live->controlPoints.at(1).volumePoint;
                publishing.handleGeneratedControlPointSetBreak(live->surfaceName, 1, true);
                require(bool(publishing._publication) && live->storePin == pinBefore,
                        "a tag edit's save runs in the background");
                settle("the tag edit is published");
                const auto after = stored();
                const auto control = std::find_if(after.controlPoints.begin(), after.controlPoints.end(),
                                                  [&](const auto& candidate) { return candidate == tagged; });
                require(publishing._fiberSaveFailureCount == before && control != after.controlPoints.end() &&
                        vc3d::line_annotation::hasControlPointTag(control->tags, vc3d::line_annotation::kBreakTag) &&
                        live->storePin != pinBefore && stored().storePin == live->storePin,
                        "and the tag is stored at the pin the session holds");
                publishing.handleGeneratedControlPointSetBreak(live->surfaceName, 1, false);
                settle("the tag removal is published");
                require(publishing._fiberSaveFailureCount == before, "the tag is removed again");
                // A gap tags both of its ends in one save: the GUI thread
                // never waits for the first end's save to store the second.
                const auto first = live->controlPoints.at(0).volumePoint, second = live->controlPoints.at(1).volumePoint;
                const auto commitBeforeGap = stored().graphCommit;
                require(bool(commitBeforeGap), "the stored fiber names its commit before the gap");
                publishing.handleGeneratedSpanSetGap(live->surfaceName, 0, 1, true);
                require(bool(publishing._publication) && !publishing._heldControlEdits,
                        "a gap's save runs in the background");
                settle("the gap is published");
                require(stored().graphCommit && *stored().graphCommit == *commitBeforeGap + 1,
                        "both ends of a gap are stored by one save");
                const auto gapped = stored();
                const auto isBreak = [&](const cv::Vec3d& where) {
                    const auto found = std::find_if(gapped.controlPoints.begin(), gapped.controlPoints.end(),
                                                    [&](const auto& candidate) { return candidate == where; });
                    return found != gapped.controlPoints.end() &&
                        vc3d::line_annotation::hasControlPointTag(found->tags, vc3d::line_annotation::kBreakTag);
                };
                require(publishing._fiberSaveFailureCount == before && isBreak(first) && isBreak(second) &&
                        gapped.storePin == live->storePin, "both ends of a gap are stored");
                publishing.handleGeneratedSpanSetGap(live->surfaceName, 0, 1, false);
                settle("the gap removal is published");
                const auto ungapped = stored();
                require(publishing._fiberSaveFailureCount == before &&
                        std::none_of(ungapped.controlPoints.begin(), ungapped.controlPoints.end(), [](const auto& control) {
                            return vc3d::line_annotation::hasControlPointTag(control.tags, vc3d::line_annotation::kBreakTag);
                        }) && ungapped.storePin == live->storePin, "the gap is removed again");
            }
            // The install gives a published control its id by position when a
            // control placed meanwhile shifted the order.
            {
                require(live->controlPoints.size() >= 2 && live->optimizedLine.points.size() >= 2, "C has a line");
                const auto last = live->controlPoints.back();
                const int lineEnd = int(live->optimizedLine.points.size());
                const cv::Vec3d step = last.volumePoint - live->controlPoints[live->controlPoints.size() - 2].volumePoint;
                const auto addControl = [&](double linePosition, const cv::Vec3d& where, int optimizedIndex) {
                    vc3d::line_annotation::LineControlPoint control(linePosition, where, false, optimizedIndex);
                    control.storeId = -1;
                    live->controlPoints.push_back(control);
                };
                vc::lasagna::LinePoint p1; p1.position = last.volumePoint + step * .5;
                vc::lasagna::LinePoint p2; p2.position = last.volumePoint + step;
                live->optimizedLine.points.push_back(p1);
                live->optimizedLine.points.push_back(p2);
                addControl(last.linePosition + 2, p2.position, lineEnd + 1);
                publishing.saveSessionAsFiber(*live, false, true);
                require(bool(publishing._publication), "the placement's background save starts");
                addControl(last.linePosition + 1, p1.position, lineEnd);  // placed before it, while it publishes
                settle("the placement is published");
                const auto idOf = [&](const cv::Vec3d& where) -> int64_t {
                    for (const auto& control : live->controlPoints) if (control.volumePoint == where) return control.storeId;
                    return int64_t(-2);
                };
                const int64_t published = idOf(p2.position);
                require(published >= 0 && idOf(p1.position) < 0,
                        "the published control has its id; the one placed since has none yet");
                publishing.saveSessionAsFiber(*live, false);
                const auto after = stored();
                const auto kept = std::find_if(after.controlPoints.begin(), after.controlPoints.end(),
                                               [&](const auto& control) { return control == p2.position; });
                require(publishing._fiberSaveFailureCount == failures && kept != after.controlPoints.end() &&
                        kept->storeId == published && idOf(p1.position) >= 0 && idOf(p1.position) != published,
                        "the next save keeps the published control's identity and gives the new one its own");
            }
            // A window closed while a save runs: the pane's cleanup and its
            // close save wait behind that save instead of blocking, and the
            // close save itself runs in the background.
            {
                const auto before = publishing._fiberSaveFailureCount;
                const auto readA = [&] { return publishing.readStoredFibers(std::vector<std::string>{aUuid}, true).at(0); };
                const auto closing = openSession(readA(), "publication-closing");
                closing->taskState = Controller::LineAnnotationSession::TaskState::Succeeded;
                closing->fiberWidth += .5;
                const auto widthToStore = closing->fiberWidth;
                live->fiberWidth += .25;
                publishing.saveSessionAsFiber(*live, false, true);
                const auto running = publishing._publication;
                require(bool(running), "a save of another fiber is running");
                publishing.closeWindowlessPane("publication-closing");
                require(publishing._publication == running && publishing.paneForSurface("publication-closing") &&
                        publishing._closingSurfaces.size() == 1,
                        "closing a window during a save returns at once and keeps the pane until that save ends");
                settle("the save, the queued close and its save finish");
                require(!publishing.paneForSurface("publication-closing") && publishing._closingSurfaces.empty() &&
                        publishing._fiberSaveFailureCount == before && readA().width == widthToStore,
                        "the closed pane is gone and its session was stored");
                // Nothing running: the close save still leaves the GUI thread.
                const auto again = openSession(readA(), "publication-closing-2");
                again->taskState = Controller::LineAnnotationSession::TaskState::Succeeded;
                again->fiberWidth += .5;
                const auto second = again->fiberWidth;
                publishing.closeWindowlessPane("publication-closing-2");
                require(bool(publishing._publication) && !publishing.paneForSurface("publication-closing-2"),
                        "a close with nothing running saves in the background");
                // An open of a fiber meanwhile waits for that save: it must
                // read what the closed window stored.
                publishing.settleClosedPanes();
                require(!publishing._publication && !publishing._closedPanePublication &&
                        publishing._fiberSaveFailureCount == before && readA().width == second,
                        "an open waits for a closed window's save and reads what it stored");
                settle("nothing is left queued");
                // The same for a close still queued behind a running save.
                const auto third = openSession(readA(), "publication-closing-3");
                third->taskState = Controller::LineAnnotationSession::TaskState::Succeeded;
                third->fiberWidth += .5;
                const auto thirdWidth = third->fiberWidth;
                live->fiberWidth += .25;
                publishing.saveSessionAsFiber(*live, false, true);
                publishing.closeWindowlessPane("publication-closing-3");
                require(bool(publishing._publication) && publishing._closingSurfaces.size() == 1, "a close is queued");
                publishing.settleClosedPanes();
                require(!publishing._publication && publishing._closingSurfaces.empty() &&
                        !publishing.paneForSurface("publication-closing-3") &&
                        publishing._fiberSaveFailureCount == before && readA().width == thirdWidth,
                        "an open makes a queued close save before it reads");
                settle("the queued close finds its pane gone");
            }
            // A tag edit whose background save is refused (a commit by
            // another client that this controller has not seen) is reported
            // and stays in the session, like a placement whose save failed.
            {
                const auto before = publishing._fiberSaveFailureCount;
                const auto where = live->controlPoints.at(1).volumePoint;
                const auto hasBreak = [&] {
                    const auto control = std::find_if(live->controlPoints.begin(), live->controlPoints.end(),
                                                      [&](const auto& candidate) { return candidate.volumePoint == where; });
                    return control != live->controlPoints.end() &&
                        vc3d::line_annotation::hasControlPointTag(control->tags, vc3d::line_annotation::kBreakTag);
                };
                submitOther({SetMetaOp{aUuid, "tag_retain_test", "true"}}, aUuid);
                publishing.handleGeneratedControlPointSetBreak(live->surfaceName, 1, true);
                require(bool(publishing._publication) && hasBreak(), "the tag shows at once while its save runs");
                settle("the refused tag save ends");
                for (int i = 0; i < 5; ++i) QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
                require(publishing._fiberSaveFailureCount == before + 1 && hasBreak() &&
                        std::any_of(publishing._panes.begin(), publishing._panes.end(),
                                    [&](const auto& open) { return open.session == live; }),
                        "a tag whose save failed is reported and kept in the session");
            }
            // A publication that fails (another client changed the fiber): it
            // is counted, and what was queued behind it does not stay queued.
            {
                submitOther({SetMetaOp{cUuid, "publication_test", "true"}}, cUuid);
                live->fiberWidth += .25;
                publishing.saveSessionAsFiber(*live, false, true);
                require(bool(publishing._publication), "the conflicting background save starts");
                live->fiberWidth += .25;
                publishing.saveSessionAsFiber(*live, false, true);
                settle("a failed publication ends and its queue drains");
                require(publishing._fiberSaveFailureCount > failures, "the failed publication is counted");
                QCoreApplication::processEvents();
            }
            publishing._panes.clear();
            session->refresh(hosted); state.pollObjectStoreChanges();
            QCoreApplication::processEvents();
        }
        writer.detach(other);
        state.closeProject();

        // Transforms: a fiber stored in a source volume's coordinates, read in
        // the current volume through a recorded registration. Another client
        // edits the registration and no event is delivered: the next read must
        // not decode through (and certify) the attachment's stale transforms.
        QTemporaryDir volumeDirectory;
        vc3d::test::VolumeSpec spec;
        spec.shapeZYX = {32, 32, 32}; spec.chunkShapeZYX = {32, 32, 32};
        const auto volume = vc3d::test::createVolume(fs::path(volumeDirectory.path().toStdString()) / "ct.zarr", spec);
        const auto transformed = fs::path(directory.path().toStdString()) / "transformed.vcobj";
        { auto db = ObjectStore::create(transformed, readOptions); }
        std::string boundUuid, partnerUuid;
        {
            CState creator;
            Controller creating(&creator, nullptr, nullptr, nullptr);
            creating._errorDialogsSuppressed = true;
            openStoreAsProject(creator, QString::fromStdString(transformed.string()), "fiber-read-test");
            creator.setCurrentVolume(volume);
            Controller::StoredFiber fiber;
            fiber.id = creating.nextFiberId(); fiber.fileName = "bound.json";
            fiber.username = "test"; fiber.startedAt = "20260101"; fiber.sequence = 1;
            for (int x : {2, 6, 10}) { fiber.controlPoints.emplace_back(cv::Vec3d{x * 1., 4, 4}); fiber.linePoints.emplace_back(x * 1., 4, 4); }
            // A partner linked to it (control 1 <-> partner control 2),
            // approved and adjacent, so the map snapshot has links to compare.
            Controller::StoredFiber partner = fiber;
            partner.uuid = vc::objects::util::newUuid(); partner.id = creating.nextFiberId(); partner.fileName = "partner.json";
            for (auto& p : partner.controlPoints) p[1] = 9;
            for (auto& p : partner.linePoints) p[1] = 9;
            std::vector<Controller::FiberSaveSnapshot> batch{creating.makeFiberSaveSnapshot(fiber), creating.makeFiberSaveSnapshot(partner)};
            creating.commitFiberStoreAction(batch);
            fiber = batch[0].fiber; partner = batch[1].fiber;
            Controller::FiberBranchRef link;
            link.controlPointIndex = 1; link.branchControlPointIndex = 2; link.branchFiberId = partner.id;
            link.branchFileName = partner.fileName; link.pending = true; link.adjacent = true;
            link.controlPointPosition = fiber.controlPoints[1]; link.branchControlPointPosition = partner.controlPoints[2];
            link.controlPointDirection = {1, 0, 0}; link.branchControlPointDirection = {1, 0, 0};
            fiber.branches.push_back(link);
            std::swap(link.controlPointIndex, link.branchControlPointIndex);
            std::swap(link.controlPointPosition, link.branchControlPointPosition);
            link.branchFiberId = fiber.id; link.branchFileName = fiber.fileName;
            partner.branches.push_back(link);
            batch = {creating.makeFiberSaveSnapshot(fiber), creating.makeFiberSaveSnapshot(partner)};
            creating.commitFiberStoreAction(batch);
            fiber = batch[0].fiber; partner = batch[1].fiber;
            fiber.branches.front().pending = false; partner.branches.front().pending = false;
            batch = {creating.makeFiberSaveSnapshot(fiber), creating.makeFiberSaveSnapshot(partner)};
            creating.commitFiberStoreAction(batch);
            boundUuid = batch.front().fiber.uuid;
            partnerUuid = batch.back().fiber.uuid;
            creator.closeProject();
        }
        const std::string source = "source-" + vc::objects::util::newUuid();
        ObjectRecord registration; registration.uuid = vc::objects::util::newUuid();
        registration = volumeTransformRecord(registration, source, volume->id(), VolumeAffine::identity(), "test registration");
        {
            auto db = ObjectStore::open(transformed, readOptions);
            RecordSelection expected;
            {
                auto txn = db.read();
                expected[boundUuid] = db.currentObjectRevision(txn, boundUuid)->reference();
                expected[partnerUuid] = db.currentObjectRevision(txn, partnerUuid)->reference();
            }
            expected[registration.uuid] = std::nullopt;  // created here
            auto txn = db.write(expected);
            db.setMeta(txn, boundUuid, "coordinates", vc::objects::volumeCoordinates(source, 1).dump());
            db.setMeta(txn, partnerUuid, "coordinates", vc::objects::volumeCoordinates(source, 1).dump());
            db.upsertObject(txn, registration);
            txn.commit();
        }
        QProcess transformServer;
        const auto stopTransformServer = qScopeGuard([&] { transformServer.terminate(); transformServer.waitForFinished(10000); });
        const auto transformEndpoint = startServer(transformServer, transformed);
        const auto transformLocal = fs::path(directory.path().toStdString()) / "transform-local.vcobj";
        { auto db = ObjectStore::create(transformLocal, readOptions); }
        // No background poll: only the controller's own refresh can bring the
        // attachment's transforms up to date before the read.
        qputenv("VC3D_HOSTED_POLL_MS", "3600000");
        const auto restorePoll = qScopeGuard([] { qunsetenv("VC3D_HOSTED_POLL_MS"); });
        CState viewing;
        Controller reading(&viewing, nullptr, nullptr, nullptr);
        reading._errorDialogsSuppressed = true;
        openStoreAsProject(viewing, QString::fromStdString(transformLocal.string()), "fiber-read-test");
        const auto viewed = viewing.attachHostedStore(transformEndpoint, "fiber-read-test");
        viewing.setStoreFor("fiber", viewed);
        viewing.setCurrentVolume(volume);
        const auto readBound = [&] {
            auto fibers = reading.storedFibersShared(std::vector<std::string>{boundUuid}, true);
            require(fibers.size() == 1 && fibers.front()->frame, "the bound fiber is read in the current volume");
            return fibers.front()->controlPoints.front()[0];
        };
        const double before = readBound();
        ObjectStoreSession editor;
        HostedSessionOptions editorOptions; editorOptions.author = "fiber-read-test-writer";
        const auto editing = editor.attachHosted(transformEndpoint, editorOptions);
        auto shifted = VolumeAffine::identity(); shifted.values[3] = 12;
        registration = volumeTransformRecord(registration, source, volume->id(), shifted, "test registration");
        {
            auto request = editor.newRequest(editing);
            editor.readStore(editing, [&](const StoreReader& reader, const auto&) {
                request.expectedObjects[registration.uuid] = reader.currentObject(registration.uuid)->reference();
            });
            request.operations = {UpdateObjectOp{objectRecordJson(registration)}};
            editor.submit(editing, request);
        }
        require(readBound() == before + 12, "a registration edit is decoded through current transforms");
        require(readBound() == before + 12, "and the cached decode keeps the current mapping");
        // The attachment's last check already saw the position this read
        // validates at: no refresh, and the decode is cached.
        {
            reading._storedFiberDecodeCache.clear(); reading._storedFiberDependencies.clear();
            bool refreshed = false;
            reading._fiberReadTestStage = [&](const char* stage) {
                if (std::string_view(stage) == "transforms-refresh") refreshed = true;
            };
            require(readBound() == before + 12, "a decode at the checked position");
            reading._fiberReadTestStage = {};
            require(!refreshed && reading._storedFiberDecodeCache.contains(boundUuid),
                    "skips the transform refresh and is cached");
        }
        // The Fiber Map snapshot with its store reads on another thread equals
        // the synchronous one: same runtime ids (fresh controllers, so every
        // uuid is new to both), labels, geometry and links.
        {
            Controller synchronous(&viewing, nullptr, nullptr, nullptr);
            Controller threaded(&viewing, nullptr, nullptr, nullptr);
            synchronous._errorDialogsSuppressed = threaded._errorDialogsSuppressed = true;
            const auto expectedSnapshot = synchronous.fiberMapSnapshot();
            auto decode = threaded.beginFiberMapSnapshot();
            std::thread worker([&] { Controller::decodeFiberMapSnapshot(*decode); });
            worker.join();
            const auto snapshot = threaded.finishFiberMapSnapshot(*decode);
            require(!expectedSnapshot.fibers.empty() && snapshot.fibers.size() == expectedSnapshot.fibers.size(),
                    "the threaded map snapshot has the same fibers");
            for (size_t i = 0; i < snapshot.fibers.size(); ++i) {
                const auto& a = snapshot.fibers[i];
                const auto& b = expectedSnapshot.fibers[i];
                bool sameLinks = a.links.size() == b.links.size();
                for (size_t k = 0; sameLinks && k < a.links.size(); ++k)
                    sameLinks = a.links[k].controlPointIndex == b.links[k].controlPointIndex &&
                                a.links[k].branchFiberId == b.links[k].branchFiberId &&
                                a.links[k].branchControlPointIndex == b.links[k].branchControlPointIndex &&
                                a.links[k].pending == b.links[k].pending && a.links[k].adjacent == b.links[k].adjacent &&
                                a.links[k].autoLink == b.links[k].autoLink;
                require(a.id == b.id && a.label == b.label && a.fileName == b.fileName && a.hvTag == b.hvTag &&
                        a.controlPoints == b.controlPoints && a.linePoints == b.linePoints && sameLinks &&
                        a.tracedSegments == b.tracedSegments && a.breaks == b.breaks &&
                        a.kollesisTerminations == b.kollesisTerminations && a.gapSegments == b.gapSegments &&
                        a.damagedSegments == b.damagedSegments,
                        "the threaded map snapshot matches fiber by fiber");
            }
            require(threaded._storedFiberDecodeCache.contains(boundUuid), "the worker's decodes join the cache");
            {
                const auto byFile = [&](const std::string& file) -> const Controller::FiberMapFiber& {
                    for (const auto& fiber : snapshot.fibers) if (fiber.fileName == file) return fiber;
                    throw std::runtime_error("map fiber " + file + " missing");
                };
                const auto& boundEntry = byFile("bound.json");
                const auto& partnerEntry = byFile("partner.json");
                require(boundEntry.links.size() == 1 && boundEntry.links[0].controlPointIndex == 1 &&
                        boundEntry.links[0].branchFiberId == partnerEntry.id && boundEntry.links[0].branchControlPointIndex == 2 &&
                        !boundEntry.links[0].pending && boundEntry.links[0].adjacent,
                        "the bound fiber's link: indices, peer id and flags as stored");
                require(partnerEntry.links.size() == 1 && partnerEntry.links[0].controlPointIndex == 2 &&
                        partnerEntry.links[0].branchFiberId == boundEntry.id && partnerEntry.links[0].branchControlPointIndex == 1 &&
                        !partnerEntry.links[0].pending && partnerEntry.links[0].adjacent,
                        "the partner's reciprocal link");
            }
            // The GUI allocates a runtime id for a uuid the worker met as new
            // while the worker runs: the finished snapshot uses that id.
            {
                Controller racing(&viewing, nullptr, nullptr, nullptr);
                racing._errorDialogsSuppressed = true;
                auto raced = racing.beginFiberMapSnapshot();
                Controller::decodeFiberMapSnapshot(*raced);
                (void)racing.fiberIdForUuid(vc::objects::util::newUuid());  // an unrelated allocation first
                const auto partnerId = racing.fiberIdForUuid(partnerUuid);
                const auto guiId = racing.fiberIdForUuid(boundUuid);
                const auto racedSnapshot = racing.finishFiberMapSnapshot(*raced);
                bool linkRemapped = false;
                for (const auto& fiber : racedSnapshot.fibers)
                    if (fiber.fileName == "bound.json")
                        linkRemapped = fiber.id == guiId && fiber.links.size() == 1 && fiber.links[0].branchFiberId == partnerId;
                const auto cached = racing._storedFiberDecodeCache.at(boundUuid);
                require(racedSnapshot.fibers.size() == 2 && linkRemapped && cached->id == guiId &&
                        cached->branches.size() == 1 && cached->branches[0].branchFiberId == partnerId,
                        "GUI allocations made while the worker ran are the ids the snapshot, its links and the cache use");
            }
            // A project change between begin and finish refuses the result.
            auto stale = threaded.beginFiberMapSnapshot();
            Controller::decodeFiberMapSnapshot(*stale);
            ++threaded._packageGeneration;
            bool refused = false;
            try { (void)threaded.finishFiberMapSnapshot(*stale); } catch (const std::exception&) { refused = true; }
            require(refused, "a snapshot read across a project change is refused");
        }
        editor.detach(editing);
        viewing.closeProject();
    } else {
        std::cout << "fiber read test: no vc_object_server at " << serverBinary << ", hosted part skipped" << std::endl;
    }
}
