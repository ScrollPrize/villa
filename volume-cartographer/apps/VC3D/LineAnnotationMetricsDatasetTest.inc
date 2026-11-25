// Alignment metrics pick the same dataset as an annotation pane, under both
// normal sources, but build no normal sampler on the GUI thread: their worker
// builds its own, and constructing one costs ~0.4 s on a real Lasagna dataset.
{
    QTemporaryDir temporary;
    const auto root = fs::path(temporary.path().toStdString());
    const auto store = root / "metrics.vcobj";
    OpenOptions options; options.author = "metrics-dataset-test";
    { auto db = ObjectStore::create(store, options); }
    CState state;
    Controller controller(&state, nullptr, nullptr, nullptr);
    controller._errorDialogsSuppressed = true;
    openStoreAsProject(state, QString::fromStdString(store.string()), options.author.c_str());

    const auto writeText = [](const fs::path& path, const std::string& text) { std::ofstream(path) << text; };
    // Lasagna normals: one packed grad_mag/nx/ny group.
    const auto lasagnaDir = root / "lasagna";
    fs::create_directories(lasagnaDir);
    {
        auto writer = vcr::zarr::Writer::create(lasagnaDir / "pred.zarr",
            {.shape = {3, 2, 2, 2}, .chunks = {3, 2, 2, 2}, .compressor = ""});
        std::vector<uint8_t> payload(3 * 2 * 2 * 2, 128);
        std::fill(payload.begin(), payload.begin() + 16, 255);
        writer.writeChunk({0, 0, 0, 0}, payload.data(), payload.size());
    }
    const auto lasagnaManifest = lasagnaDir / "dataset.lasagna.json";
    writeText(lasagnaManifest, R"({"version": 2, "packed_channel_view": true, "source_to_base": 1.0,
        "grad_mag_encode_scale": 255.0, "grad_mag_factor": 1.0,
        "groups": {"pred": {"zarr": "pred.zarr", "scaledown": 0, "channels": ["grad_mag", "nx", "ny"]}}})");
    // Predicted fiber normals: a compound prediction with sheet_* channels.
    const auto fiberDir = root / "fiber";
    {
        auto writer = vcr::zarr::Writer::create(fiberDir / "sd2/2",
            {.shape = {7, 2, 2, 2}, .chunks = {7, 2, 2, 2}, .compressor = "", .fillValue = 0.});
        const std::vector<uint8_t> payload(7 * 2 * 2 * 2, 200);
        writer.writeChunk({0, 0, 0, 0}, payload.data(), payload.size());
    }
    writeText(fiberDir / ".zattrs", R"({"lasagna": {"version": 3, "base_shape_zyx": [7, 7, 7],
        "source_to_base": 1.0, "validity_channel": "presence", "groups": {"sd2": {"scaledown": 2, "zarr": "sd2/2",
        "channels": ["presence", "nx", "ny", "width", "sheet_nx", "sheet_ny", "sheet_grad_mag"]}}}})");

    state.vpkg()->setSelectedLasagnaDataset(lasagnaManifest.string());
    state.vpkg()->setSelectedFiberInferenceDataset(fiberDir.string());
    for (const std::string source : {"lasagna", "fiber"}) {
        state.vpkg()->setAnnotationNormalSource(source);
        const auto builds = controller._sessionNormalSamplerBuilds;
        const auto metrics = controller.resolveAlignmentMetricsDataset();
        require(metrics != nullptr, "alignment metrics resolve a dataset");
        require(controller._sessionNormalSamplerBuilds == builds, "alignment metrics build no GUI-thread normal sampler");

        Controller::LineAnnotationSession pane;
        pane.suppressErrorDialogs = true;
        require(controller.ensureDatasetForSession(pane) && pane.normalSampler, "an annotation pane gets a sampler");
        require(controller._sessionNormalSamplerBuilds == builds + 1, "the pane built exactly one sampler");
        require(pane.usingFiberNormals == (source == "fiber"), "the pane follows the project normal source");
        require(metrics->manifest().manifestPath == pane.dataset->manifest().manifestPath &&
                metrics->manifest().workingToBaseScale == pane.dataset->manifest().workingToBaseScale &&
                metrics->manifest().groupForChannel("nx")->channelIndex("nx") ==
                    pane.dataset->manifest().groupForChannel("nx")->channelIndex("nx"),
                "metrics and pane resolve the same normal dataset");
        // The worker's own sampler from the metrics dataset samples what the pane samples.
        vc::lasagna::LasagnaNormalSampler worker(*metrics);
        const cv::Vec3d point = source == "fiber" ? cv::Vec3d{2, 2, 2} : cv::Vec3d{0.5, 0.5, 0.5};
        const auto a = worker.sampleNormal(point), b = pane.normalSampler->sampleNormal(point);
        require(a.valid == b.valid && (!a.valid || cv::norm(a.normal - b.normal) < 1.e-9),
                "the worker's sampler matches the pane's");
    }
    std::cout << "metrics_dataset PASS: no GUI-thread sampler, same dataset as panes (lasagna + fiber normals)" << std::endl;

    // Picking the dataset that is already the default saves it; inputs changed
    // outside the menus (an attachment) invalidate as a menu selection does.
    state.vpkg()->clearSelectedFiberInferenceDataset();
    state.vpkg()->attachPreparedLasagnaDataset(fiberDir.string(), {}, true, {});
    const auto fiberEntries = state.vpkg()->fiberInferenceDatasetEntries();
    require(fiberEntries.size() == 1 && state.vpkg()->selectedFiberInferenceDataset() == fiberEntries.front().location &&
            state.vpkg()->document().data()["view"].value("selected_fiber_inference_dataset", std::string{}).empty(),
            "the attached fiber dataset is the unsaved default");
    controller.syncAnnotationInputsWithProject();
    const auto generation = controller._fiberMetricsGeneration;
    controller.handleFiberInferenceDatasetSelectionChanged(state.vpkg()->selectedFiberInferenceDataset());
    require(controller._fiberMetricsGeneration == generation, "picking the current dataset invalidates nothing");
    require(!state.vpkg()->document().data()["view"].value("selected_fiber_inference_dataset", std::string{}).empty(),
            "picking the current default saves it as the user's choice");
    controller.syncAnnotationInputsWithProject();
    require(controller._fiberMetricsGeneration == generation, "unchanged inputs invalidate nothing");
    state.vpkg()->setSelectedFiberInferenceDataset(lasagnaManifest.string());
    // Not while a solve runs: its result would land on the invalidated session.
    auto running = std::make_shared<Controller::LineAnnotationSession>();
    running->taskState = Controller::LineAnnotationSession::TaskState::Running;
    controller._panes.push_back({1, Controller::SourceKind::Plane, "running", nullptr, running});
    controller.syncAnnotationInputsWithProject();
    require(controller._fiberMetricsGeneration == generation && controller._annotationInputsSyncDeferred,
            "a change during a solve waits for it");
    running->taskState = Controller::LineAnnotationSession::TaskState::Idle;
    controller.syncAnnotationInputsWithProject();
    controller._panes.clear();
    require(controller._fiberMetricsGeneration == generation + 1 && !controller._annotationInputsSyncDeferred,
            "a changed fiber dataset or normal source invalidates once no solve runs");
    // An unattached fiber dataset without predicted normals falls back to
    // Lasagna, so changing the Lasagna dataset must invalidate the normals.
    const auto unattached = (root / "unattached-fiber.lasagna.json").string();
    controller.handleFiberInferenceDatasetSelectionChanged(unattached);
    require(state.vpkg()->annotationNormalSource() == "fiber" && !state.vpkg()->hasPredictedFiberNormals(unattached),
            "an unattached fiber dataset is checked from its manifest when used");
    const auto beforeLasagna = controller._fiberMetricsGeneration;
    controller.handleLasagnaDatasetSelectionChanged(fiberDir.string());
    require(controller._fiberMetricsGeneration == beforeLasagna + 1,
            "a Lasagna change invalidates when Lasagna may supply the normals");
    std::cout << "annotation_inputs PASS: explicit picks persist, external changes invalidate" << std::endl;
    state.closeProject();
}
