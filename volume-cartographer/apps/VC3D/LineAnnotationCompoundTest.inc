// Optional real-volume integration regression, using the same private session
// entry points as the GUI. Only the temporary project/settings are modified.
if (const char* source = std::getenv("VC3D_COMPOUND_TEST_VOLUME")) {
    QTemporaryDir temporary;
    const auto store = fs::path(temporary.path().toStdString()) / "compound.vcobj";
    OpenOptions options; options.author = "compound-normal-test";
    { auto db = ObjectStore::create(store, options); }
    CState state;
    Controller controller(&state, nullptr, nullptr, nullptr);
    controller._errorDialogsSuppressed = true;
    openStoreAsProject(state, QString::fromStdString(store.string()), options.author.c_str());
    const auto dataset = vc::lasagna::LasagnaDataset::openLocation(source);
    std::vector<VolumePkg::PreparedVolumeAttachment> volumes;
    for (auto& prepared : vc::lasagna::prepareLasagnaProjectVolumes(dataset, source))
        volumes.push_back({std::move(prepared.location), std::move(prepared.tags), std::move(prepared.volume)});
    state.vpkg()->attachPreparedLasagnaDataset(source, {}, true, volumes);
    require(state.vpkg()->selectedLasagnaDataset().empty(), "no Lasagna selected");
    require(state.vpkg()->lasagnaDatasetEntries().empty(), "no Lasagna attached");

    require(state.vpkg()->annotationNormalSource() == "fiber", "attachment initializes project normal source");
    const cv::Vec3d seed{6160.2939453125, 8019.69580078125, 26137.80859375};
    const cv::Vec3d direction{0.2243, 0.0238, -0.9740};
    // Exercise the same attachment + progressive scalar display as VC3D. There
    // is no physical level zero in this compound, and presence is packed CZYX.
    OpenedVolumes openedVolumes(64 * 1024 * 1024);
    const auto viewer = vc3d::openViewerLevels(openedVolumes, *volumes.front().volume);
    // This OME pyramid's display coordinate frame is its finest stored grid.
    const double displayScale = double(dataset.manifest().baseShapeZYX.value()[0]) / volumes.front().volume->shape()[0];
    vcr::grid::Frame frame;
    frame.plane = vcr::grid::Plane{{float(seed[0] / displayScale), float(seed[1] / displayScale), float(seed[2] / displayScale)},
        {1,0,0}, {0,1,0}, {0,0,1}, 1, 1};
    auto request = viewer.request(2, 0);
    require(request.levels.size() == 1 && request.levels[0].channel == 0, "viewer selects packed presence");
    vcr::grid::Pending pending;
    vcr::grid::sampleCached(frame, request, &pending);
    vcr::grid::load(pending);
    const auto rendered = vcr::grid::sampleCached(frame, request, nullptr);
    require(rendered.level[0] == 0 && rendered.values[0] > 0, "progressive display reads real packed presence");
    Controller::LineAnnotationSession session;
    session.suppressErrorDialogs = true;
    require(controller.ensureDatasetForSession(session), "GUI base normals resolve without Lasagna");
    require(session.usingFiberNormals, "saved project choice uses predicted fiber normals");
    require(controller.ensureFiberInferenceDatasetForSession(session), "GUI trace inputs resolve");
    require(session.traceNormalSampler && session.fiberPredictionField, "native tracing backend ready");
    const auto baseNormal = session.normalSampler->sampleNormal(seed);
    const auto traceNormal = session.traceNormalSampler->sampleNormal(seed / session.fiberTraceToBaseScale);
    require(baseNormal.valid && traceNormal.valid, "actual compound normals are valid at the seed");
    require(cv::norm(baseNormal.normal - traceNormal.normal) < 1.e-6, "base/trace frames sample the same normal");
    const auto* nx = session.dataset->manifest().groupForChannel("nx");
    require(nx && nx->channelIndex("nx") == dataset.manifest().groupForChannel("sheet_nx")->channelIndex("sheet_nx"),
            "normal sampler uses sheet_nx, not the fiber tangent nx");
    QString error;
    auto inputs = controller.resolveFiberOptimizationInputs({}, {}, &error);
    require(inputs.has_value(), "headless GUI optimization also needs no Lasagna");
    require(controller.resolveAlignmentMetricsDataset() != nullptr, "alignment metrics need no Lasagna either");
    require(cv::norm(inputs->baseNormalSampler->sampleNormal(seed).normal - baseNormal.normal) < 1.e-6,
            "interactive and headless normal selection agree");

    vc::fiber_tracer::FiberTraceConfig config;
    config.parallelThreads = 1;
    const auto trace = vc::fiber_tracer::traceFiberExtrapolation(
        *session.fiberPredictionField, seed / session.fiberTraceToBaseScale, direction,
        64.0 / session.fiberTraceToBaseScale, config, session.traceNormalSampler.get());
    require(trace.points.size() > 1, "native tracing advances on real compound data without Lasagna");
    std::cout << "compound_gui volume=" << source << " trace_points=" << trace.points.size()
              << " trace_to_base=" << session.fiberTraceToBaseScale
              << " normal=" << baseNormal.normal << std::endl;

    state.vpkg()->setAnnotationNormalSource("lasagna");
    Controller::LineAnnotationSession external;
    external.suppressErrorDialogs = true;
    require(!controller.ensureDatasetForSession(external), "explicit Lasagna still requires a Lasagna dataset");
    require(!controller.resolveFiberOptimizationInputs({}, {}, &error), "headless respects explicit Lasagna selection");
    state.vpkg()->setAnnotationNormalSource("fiber");
    require(controller.ensureDatasetForSession(external) && external.usingFiberNormals,
            "explicit predicted normal selection works");
    require(controller.resolveFiberOptimizationInputs({}, {}, &error).has_value(), "headless explicit fiber normals work");
    require(state.vpkg()->selectedLasagnaDataset().empty(), "no hidden Lasagna attachment was introduced");

    state.setCurrentVolume(volumes.front().volume);
    const auto displaySeed = seed / displayScale;
    Controller::LineAnnotationSession displayed;
    displayed.suppressErrorDialogs = true;
    require(controller.ensureDatasetForSession(displayed) &&
            controller.ensureFiberInferenceDatasetForSession(displayed), "tracing from the compound display is ready");
    require(cv::norm(displayed.normalSampler->sampleNormal(displaySeed).normal - baseNormal.normal) < 1.e-6,
            "compound display coordinates map to the same physical normal");
    require(cv::norm(displaySeed / displayed.fiberTraceToBaseScale - seed / session.fiberTraceToBaseScale) < 1.e-6,
            "compound display seed maps to the same physical trace seed");
    const auto displayedInputs = controller.resolveFiberOptimizationInputs({}, {}, &error);
    require(displayedInputs && cv::norm(displayedInputs->baseNormalSampler->sampleNormal(displaySeed).normal - baseNormal.normal) < 1.e-6,
            "headless compound-display mapping agrees");

    // An explicitly selected external normal manifest must override the compound
    // default. Alias this test's normal channels in a separate normal dataset.
    const auto normalPath = fs::path(temporary.path().toStdString()) / "normal.lasagna.json";
    const auto* normalGroup = dataset.manifest().groupForChannel("sheet_nx");
    auto channels = normalGroup->channels;
    for (auto& channel : channels) {
        if (channel == "sheet_nx") channel = "nx";
        else if (channel == "sheet_ny") channel = "ny";
        else if (channel == "sheet_grad_mag") channel = "grad_mag";
        else channel = "unused_" + channel;
    }
    nlohmann::json normalJson = {
        {"version", 3}, {"base_shape_zyx", dataset.manifest().baseShapeZYX.value()},
        {"grad_mag_encode_scale", 1.0}, {"grad_mag_factor", 1.0},
        {"groups", {{"normal", {{"zarr", normalGroup->zarrPath.string()},
            {"scaledown", normalGroup->scaledown}, {"channels", channels}}}}}};
    { std::ofstream file(normalPath); file << normalJson; }
    state.vpkg()->setSelectedLasagnaDataset(normalPath.string());
    Controller::LineAnnotationSession explicitSource;
    explicitSource.suppressErrorDialogs = true;
    require(controller.ensureDatasetForSession(explicitSource) && explicitSource.usingFiberNormals,
            "selecting an external dataset does not override the saved fiber normal choice");
    state.vpkg()->setAnnotationNormalSource("lasagna");
    require(controller.ensureDatasetForSession(explicitSource) && !explicitSource.usingFiberNormals,
            "explicit Lasagna overrides predicted normals");
    require(explicitSource.normalSampler->sampleNormal(displaySeed).valid, "external normal sampler is usable");
    const auto externalInputs = controller.resolveFiberOptimizationInputs({}, {}, &error);
    require(externalInputs && externalInputs->normalManifestLocation != externalInputs->fiberManifestLocation,
            "headless explicit Lasagna uses its separate source");

    // Two distinct manifests over real data: exchange the predicted normal's
    // x/y channels so stale samples are observable, not merely stale pointers.
    const auto secondFiber = fs::path(temporary.path().toStdString()) / "fiber-second.json";
    auto fiberJson = normalJson;
    fiberJson["source_to_base"] = dataset.manifest().sourceToBase;
    auto fiberChannels = normalGroup->channels;
    std::iter_swap(std::find(fiberChannels.begin(), fiberChannels.end(), "sheet_nx"),
                   std::find(fiberChannels.begin(), fiberChannels.end(), "sheet_ny"));
    fiberJson["groups"]["normal"]["channels"] = fiberChannels;
    { std::ofstream file(secondFiber); file << fiberJson; }
    const auto secondNormal = fs::path(temporary.path().toStdString()) / "normal-second.json";
    { std::ofstream file(secondNormal); file << normalJson; }
    // Project-local locations are persisted and offered by the GUI relatively.
    const auto secondFiberLocation = secondFiber.filename().string();
    const auto secondNormalLocation = secondNormal.filename().string();
    const auto firstNormalLocation = normalPath.filename().string();
    controller.handleNormalSourceSelectionChanged("fiber");
    auto live = std::make_shared<Controller::LineAnnotationSession>();
    live->surfaceName = "compound-change-test";
    live->fiberId = 101;
    live->suppressErrorDialogs = true;
    live->suppressFiberSave = true;
    require(controller.ensureFiberInferenceDatasetForSession(*live), "initial session has both fields");
    const std::vector<cv::Vec3d> points{displaySeed, displaySeed + direction * 4.0};
    live->optimizedLine = Controller::lineModelFromPoints(points, live->normalSampler.get());
    live->controlPoints.resize(2);
    live->controlPoints[0].volumePoint = points[0];
    live->controlPoints[1].volumePoint = points[1];
    live->controlPoints[1].linePosition = 1.0;
    live->optimizationState = Controller::SessionOptimizationState::Optimized;
    controller._panes.push_back({1, Controller::SourceKind::Plane, live->surfaceName, nullptr, live});
    auto modelOnly = std::make_shared<Controller::LineAnnotationSession>();
    modelOnly->modelOnly = true;
    modelOnly->normalSampler = live->normalSampler;
    controller._panes.push_back({2, Controller::SourceKind::Plane, "model-only", nullptr, modelOnly});
    const auto oldNormal = live->normalSampler;
    const auto oldGeneration = controller._fiberMetricsGeneration;
    const auto oldOrientation = controller._orientationEpoch;
    controller._fiberAlignmentMetrics[101] = {};
    controller._pendingFiberAlignmentMetrics.insert(202); // no open pane
    controller._pendingFiberAlignmentMetricTokens[202] = 7;
    controller._fiberMetricsPending = true;
    Controller::FiberMetricsTaskResult oldResult;
    oldResult.ok = true;
    oldResult.generation = oldGeneration;
    oldResult.requestedFiberIds = {202};
    oldResult.requestTokens[202] = 7;
    oldResult.metrics[202] = {};
    QFutureWatcher<Controller::FiberMetricsTaskResult> oldWatcher;
    oldWatcher.setFuture(QtConcurrent::run([oldResult] { return oldResult; }));
    oldWatcher.waitForFinished();

    controller.handleFiberInferenceDatasetSelectionChanged(secondFiberLocation);
    require(!live->normalSampler && !live->fiberPredictionField && !live->traceNormalSampler,
            "compound switch invalidates both fields and their trace-scale adapter");
    require(live->selectedLasagnaManifestIdentity.empty() && live->selectedFiberManifestIdentity.empty(),
            "compound switch clears stale provenance before resolving the new fields");
    require(live->normalSamplesStale && live->optimizationState == Controller::SessionOptimizationState::Unoptimized,
            "compound switch marks sampled normals and solve status stale");
    require(controller._fiberMetricsGeneration == oldGeneration + 1 &&
            controller._fiberAlignmentMetrics.empty() && controller._pendingFiberAlignmentMetrics.empty() &&
            controller._pendingFiberAlignmentMetricTokens.empty() && !controller._fiberMetricsPending,
            "compound switch invalidates all metrics, including fibers without open panes");
    require(controller._orientationEpoch == oldOrientation + 1 && controller._staleViewRefreshQueued,
            "normal changes queue a coalesced rebuild of open strips");
    require(modelOnly->normalSampler == oldNormal && !modelOnly->normalSamplesStale,
            "neural model-only sessions are unaffected by native prediction selection");

    // A late old worker must neither publish metrics nor erase a new request.
    controller._pendingFiberAlignmentMetrics.insert(202);
    controller._pendingFiberAlignmentMetricTokens[202] = 8;
    controller._fiberMetricsPending = true;
    controller.finishFiberAlignmentMetrics(&oldWatcher);
    require(!controller._fiberAlignmentMetrics.contains(202) &&
            controller.isAlignmentPendingForFiber(202, 8), "late old-source metrics cannot publish or clear new requests");
    require(controller.ensureFiberInferenceDatasetForSession(*live), "new compound inputs resolve");
    require(!live->normalSamplesStale && live->lineRevision == 1 &&
            live->optimizedLine.points[0].position == points[0] &&
            live->optimizedLine.points[1].position == points[1], "refresh resamples normals without retracing geometry");
    require(cv::norm(live->optimizedLine.points[0].sampledNormal.normal - baseNormal.normal) > 0.01,
            "line model actually samples changed normal channels");
    require(live->selectedFiberManifestIdentity == secondFiberLocation &&
            live->selectedLasagnaManifestIdentity == secondFiberLocation, "new provenance uses the compound for both fields");

    const auto compoundSampler = live->normalSampler;
    const auto compoundPredictions = live->fiberPredictionField;
    auto generation = controller._fiberMetricsGeneration;
    controller.handleLasagnaDatasetSelectionChanged(secondNormalLocation);
    require(controller._fiberMetricsGeneration == generation && live->normalSampler == compoundSampler,
            "changing inactive Lasagna selection does not invalidate predicted normals");
    controller.handleNormalSourceSelectionChanged("lasagna");
    require(controller._fiberMetricsGeneration == generation + 1 && !live->normalSampler &&
            live->fiberPredictionField == compoundPredictions, "normal-source switch preserves predictions but invalidates normals");
    require(controller.ensureFiberInferenceDatasetForSession(*live), "separate Lasagna source resolves");
    const auto lasagnaSampler = live->normalSampler;
    generation = controller._fiberMetricsGeneration;
    controller.handleFiberInferenceDatasetSelectionChanged(source);
    require(controller._fiberMetricsGeneration == generation && live->normalSampler == lasagnaSampler &&
            !live->fiberPredictionField && !live->traceNormalSampler && !live->normalSamplesStale,
            "prediction-only switch retains base normals/metrics but resets trace-scale normals");
    require(controller.ensureFiberInferenceDatasetForSession(*live), "trace-scale adapter is rebuilt after prediction change");
    const auto predictions = live->fiberPredictionField;
    controller.handleLasagnaDatasetSelectionChanged(firstNormalLocation);
    require(controller._fiberMetricsGeneration == generation + 1 && !live->normalSampler &&
            live->fiberPredictionField == predictions && live->normalSamplesStale,
            "separate Lasagna switch uses the same normal invalidation as a compound switch");
    generation = controller._fiberMetricsGeneration;
    controller.handleLasagnaDatasetSelectionChanged(firstNormalLocation);
    controller.handleFiberInferenceDatasetSelectionChanged(source);
    controller.handleNormalSourceSelectionChanged("lasagna");
    require(controller._fiberMetricsGeneration == generation, "reselecting the same inputs is a no-op");
    live->taskState = Controller::LineAnnotationSession::TaskState::Running;
    controller.handleNormalSourceSelectionChanged("fiber");
    controller.handleFiberInferenceDatasetSelectionChanged(secondFiberLocation);
    controller.handleLasagnaDatasetSelectionChanged(secondNormalLocation);
    require(controller._fiberMetricsGeneration == generation && state.vpkg()->annotationNormalSource() == "lasagna" &&
            state.vpkg()->selectedFiberInferenceDataset() == source && state.vpkg()->selectedLasagnaDataset() == firstNormalLocation,
            "all input selectors refuse changes during a solve");
    live->taskState = Controller::LineAnnotationSession::TaskState::Idle;
    controller._panes.clear();
    controller.handleNormalSourceSelectionChanged("fiber");
    controller._fiberAlignmentMetrics[303] = {};
    generation = controller._fiberMetricsGeneration;
    controller.handleFiberInferenceDatasetSelectionChanged(secondFiberLocation);
    require(controller._fiberMetricsGeneration == generation + 1 && controller._fiberAlignmentMetrics.empty(),
            "normal invalidation does not depend on any session being open");
    std::cout << "compound_gui PASS: unified dataset invalidation, late metrics, sampled normals, source identity, independent fields" << std::endl;
    std::cout << "compound_gui PASS: project choice, predicted, explicit Lasagna requirement, interactive/headless, native trace" << std::endl;
    return 0;
}
