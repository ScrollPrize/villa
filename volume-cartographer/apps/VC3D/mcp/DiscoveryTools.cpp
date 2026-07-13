#include "DiscoveryTools.hpp"

#include <fastmcpp/tools/tool.hpp>

namespace vc::mcp
{
namespace
{
Json objectSchema(Json properties, Json required)
{
    return {{"type", "object"}, {"properties", std::move(properties)}, {"required", std::move(required)}, {"additionalProperties", false}};
}
Json baseProperties()
{
    return {{"client_request_id", {{"type", "string"}, {"minLength", 1}, {"maxLength", 128}}}, {"profile", {{"type", "string"}, {"minLength", 1}, {"maxLength", 128}}}};
}
Json submissionSchema()
{
    return objectSchema(
        {{"job_id", {{"type", "string"}}},
         {"state", {{"type", "string"}}},
         {"operation", {{"type", "string"}}},
         {"job_resource", {{"type", "string"}}},
         {"log_resource", {{"type", "string"}}},
         {"submitted_at", {{"type", "string"}}}},
        Json::array({"job_id", "state", "operation", "job_resource", "log_resource", "submitted_at"}));
}
fastmcpp::tools::Tool makeTool(const std::string& name, const std::string& description, Json schema, const std::shared_ptr<JobStore>& store, bool app = false)
{
    fastmcpp::tools::Tool tool{name, std::move(schema), submissionSchema(), [store, name](const Json& input) {
                                   return store->submitDiscovery(name, input);
                               }};
    tool.set_description(description)
        .set_annotations({{"readOnlyHint", false}, {"destructiveHint", false}, {"idempotentHint", true}, {"openWorldHint", false}})
        .set_validate_args(true);
    if (app) {
        fastmcpp::AppConfig config;
        config.resource_uri = "ui://vc/inspector.html";
        config.visibility = {"tool_result"};
        tool.set_app(std::move(config));
    }
    return tool;
}
Json withBase(Json extra)
{
    Json properties = baseProperties();
    for (auto& [key, value] : extra.items())
        properties[key] = value;
    return properties;
}
}  // namespace

void registerDiscoveryTools(fastmcpp::tools::ToolManager& tools, const std::shared_ptr<JobStore>& store, const std::shared_ptr<CpuDiscovery>& discovery)
{
    const Json path = {{"type", "string"}, {"minLength", 1}};
    if (discovery && discovery->surfaceBundleAvailable()) {
        const Json artifactReference = objectSchema(
            {{"job_id", {{"type", "string"}, {"minLength", 1}}}, {"artifact_id", {{"type", "string"}, {"const", "surface"}}}},
            Json::array({"job_id", "artifact_id"}));
        const Json xyzMetadata = {{"type", "array"}, {"minItems", 3}, {"maxItems", 3}, {"items", {{"type", "number"}}}};
        Json volume = objectSchema(
            {{"kind", {{"type", "string"}, {"enum", Json::array({"local_zarr", "remote_zarr"})}}},
             {"path", path},
             {"uri", path},
             {"array_path", {{"type", "string"}, {"minLength", 1}, {"maxLength", 128}}},
             {"scale", {{"type", "integer"}, {"minimum", 0}, {"maximum", 16}}},
             {"voxel_spacing", xyzMetadata},
             {"voxel_spacing_unit", {{"type", "string"}, {"const", "um"}}},
             {"origin_xyz", xyzMetadata}},
            Json::array({"kind"}));
        volume["oneOf"] = Json::array({Json{{"required", Json::array({"path"})}}, Json{{"required", Json::array({"uri"})}}});
        const Json uvRegion = objectSchema(
            {{"u", {{"type", "integer"}, {"minimum", 0}}},
             {"v", {{"type", "integer"}, {"minimum", 0}}},
             {"width", {{"type", "integer"}, {"minimum", 1}, {"maximum", 8192}}},
             {"height", {{"type", "integer"}, {"minimum", 1}, {"maximum", 8192}}}},
            Json::array({"u", "v", "width", "height"}));
        tools.register_tool(makeTool(
            "surface_render_registered_roi",
            "Convert a completed VC TIFXYZ artifact into a canonical surface Zarr bundle and register bounded CT intensity",
            objectSchema(
                withBase(
                    {{"surface", artifactReference},
                     {"volume", volume},
                     {"coordinate_space", {{"type", "string"}, {"enum", Json::array({"ct_l0_xyz", "ct_l2_xyz"})}}},
                     {"uv_region", uvRegion},
                     {"normal_padding_voxels", {{"type", "integer"}, {"minimum", 0}, {"maximum", 64}}}}),
                Json::array({"surface", "volume", "coordinate_space", "client_request_id"})),
            store,
            true));
        const Json registeredReference = objectSchema(
            {{"job_id", {{"type", "string"}, {"minLength", 1}}}, {"artifact_id", {{"type", "string"}, {"const", "registered-surface"}}}},
            Json::array({"job_id", "artifact_id"}));
        tools.register_tool(makeTool(
            "surface_validate_geometry",
            "Measure bounded regular-grid stretch, normal discontinuity, local folds, degeneracy, and boundaries without claiming surface "
            "correctness",
            objectSchema(withBase({{"surface", registeredReference}}), Json::array({"surface", "client_request_id"})),
            store,
            true));
        tools.register_tool(makeTool(
            "surface_measure_volume_alignment",
            "Measure bounded trilinear CT intensity gradients along registered surface normals without claiming layer correctness",
            objectSchema(
                withBase({{"surface", registeredReference}, {"maximum_offset_voxels", {{"type", "integer"}, {"minimum", 1}, {"maximum", 16}}}}),
                Json::array({"surface", "client_request_id"})),
            store,
            true));
        tools.register_tool(makeTool(
            "surface_render_normal_stack",
            "Render a bounded HxWx26 or HxWx62 uint8 CT stack for a supported ink-model input contract along registered surface normals",
            objectSchema(
                withBase(
                    {{"surface", registeredReference},
                     {"model_profile", {{"type", "string"}, {"enum", Json::array({"timesformer-26", "resnet152-3d-decoder-62"})}}},
                     {"reverse_layers", {{"type", "boolean"}}},
                     {"layer_step_voxels", {{"type", "number"}, {"exclusiveMinimum", 0}, {"maximum", 4}}}}),
                Json::array({"surface", "model_profile", "client_request_id"})),
            store,
            true));
    }
    if (discovery && discovery->structuralEvidenceAvailable()) {
        const Json registeredReference = objectSchema(
            {{"job_id", {{"type", "string"}, {"minLength", 1}}}, {"artifact_id", {{"type", "string"}, {"const", "registered-surface"}}}},
            Json::array({"job_id", "artifact_id"}));
        const Json gridReference = objectSchema(
            {{"job_id", {{"type", "string"}, {"minLength", 1}}}, {"artifact_id", {{"type", "string"}, {"const", "grid-coherence"}}}},
            Json::array({"job_id", "artifact_id"}));
        const Json polarity = {{"type", "string"}, {"enum", Json::array({"bright", "dark"})}};
        const Json period = {{"type", "number"}, {"exclusiveMinimum", 0}, {"maximum", 200}};
        const Json nullTrials = {{"type", "integer"}, {"minimum", 4}, {"maximum", 64}};
        const Json nullSeed = {{"type", "integer"}, {"minimum", 0}, {"maximum", 2147483647}};
        Json gridParameters =
            {{"polarity", polarity},
             {"letter_period_mm", period},
             {"line_period_mm", period},
             {"column_period_mm", period},
             {"window_width_mm", {{"type", "number"}, {"exclusiveMinimum", 0}, {"maximum", 200}}},
             {"window_height_mm", {{"type", "number"}, {"exclusiveMinimum", 0}, {"maximum", 200}}},
             {"step_mm", {{"type", "number"}, {"exclusiveMinimum", 0}, {"maximum", 200}}},
             {"minimum_cycles", {{"type", "number"}, {"minimum", 2}, {"maximum", 10}}},
             {"null_trials", nullTrials},
             {"null_seed", nullSeed}};
        Json gridProperties = gridParameters;
        gridProperties["surface"] = registeredReference;
        tools.register_tool(makeTool(
            "text_measure_grid_coherence",
            "Measure cycle-gated physical writing-grid periodicity with deterministic null trials; never an ink or text truth claim",
            objectSchema(withBase(std::move(gridProperties)), Json::array({"surface", "client_request_id"})),
            store,
            true));
        Json compareProperties = gridParameters;
        compareProperties["surface_a"] = registeredReference;
        compareProperties["surface_b"] = registeredReference;
        tools.register_tool(makeTool(
            "ink_compare_registered_predictions",
            "Compare two identically registered signal surfaces and generate agreement, divergence, and review-priority maps",
            objectSchema(withBase(std::move(compareProperties)), Json::array({"surface_a", "surface_b", "client_request_id"})),
            store,
            true));
        tools.register_tool(makeTool(
            "text_epoch_fold_structure",
            "Search and fold a detected line period with a look-elsewhere-corrected permutation null; cannot transcribe text",
            objectSchema(
                withBase(
                    {{"surface", registeredReference},
                     {"grid", gridReference},
                     {"polarity", polarity},
                     {"period_tolerance", {{"type", "number"}, {"exclusiveMinimum", 0}, {"maximum", 0.25}}},
                     {"period_steps", {{"type", "integer"}, {"minimum", 9}, {"maximum", 101}}},
                     {"phase_bins", {{"type", "integer"}, {"minimum", 16}, {"maximum", 256}}},
                     {"null_trials", nullTrials},
                     {"null_seed", nullSeed}}),
                Json::array({"surface", "grid", "client_request_id"})),
            store,
            true));
    }
    if (discovery && discovery->evidenceFusionAvailable()) {
        auto artifactReference = [](const std::string& artifactId) {
            return objectSchema(
                {{"job_id", {{"type", "string"}, {"minLength", 1}}}, {"artifact_id", {{"type", "string"}, {"const", artifactId}}}},
                Json::array({"job_id", "artifact_id"}));
        };
        const Json registeredReference = artifactReference("registered-surface");
        tools.register_tool(makeTool(
            "surface_test_stability",
            "Measure geometry, normal, validity, and registered-signal robustness across supplied perturbation runs",
            objectSchema(
                withBase(
                    {{"baseline", registeredReference},
                     {"variants", {{"type", "array"}, {"minItems", 1}, {"maxItems", 7}, {"items", registeredReference}}},
                     {"displacement_scale_mm", {{"type", "number"}, {"exclusiveMinimum", 0}, {"maximum", 1000}}},
                     {"normal_angle_scale_degrees", {{"type", "number"}, {"exclusiveMinimum", 0}, {"maximum", 1000}}},
                     {"signal_scale", {{"type", "number"}, {"exclusiveMinimum", 0}, {"maximum", 1000}}}}),
                Json::array({"baseline", "variants", "client_request_id"})),
            store,
            true));
        const Json candidate = objectSchema(
            {{"id", {{"type", "string"}, {"minLength", 1}, {"maxLength", 64}, {"pattern", "^[A-Za-z0-9._-]+$"}}},
             {"geometry", artifactReference("surface-geometry")},
             {"alignment", artifactReference("surface-ct-alignment")},
             {"grid", artifactReference("grid-coherence")},
             {"stability", artifactReference("surface-stability")}},
            Json::array({"id", "geometry", "alignment", "grid"}));
        const Json weights = objectSchema(
            {{"geometry", {{"type", "number"}, {"minimum", 0}, {"maximum", 100}}},
             {"alignment", {{"type", "number"}, {"minimum", 0}, {"maximum", 100}}},
             {"grid", {{"type", "number"}, {"minimum", 0}, {"maximum", 100}}},
             {"stability", {{"type", "number"}, {"minimum", 0}, {"maximum", 100}}}},
            Json::array());
        tools.register_tool(makeTool(
            "ink_fuse_registered_scores",
            "Fuse an uncalibrated ResNet152 ink-model score with DinoVol UV similarity while preserving every component and optional stability map; the result is review priority, not ink probability",
            objectSchema(
                withBase(
                    {{"ink_model", artifactReference("ink-prediction")},
                     {"dinovol", artifactReference("dinovol-exemplar")},
                     {"stability", artifactReference("surface-stability")},
                     {"weights", objectSchema(
                         {{"ink_model", {{"type", "number"}, {"minimum", 0}, {"maximum", 100}}},
                          {"dinovol", {{"type", "number"}, {"minimum", 0}, {"maximum", 100}}},
                          {"stability", {{"type", "number"}, {"minimum", 0}, {"maximum", 100}}}},
                         Json::array())}}),
                Json::array({"ink_model", "dinovol", "client_request_id"})),
            store,
            true));
        tools.register_tool(makeTool(
            "surface_rank_evidence",
            "Rank bounded candidates with an explicit weighted geometric mean while retaining every component and formula",
            objectSchema(
                withBase({{"candidates", {{"type", "array"}, {"minItems", 1}, {"maxItems", 16}, {"items", candidate}}}, {"weights", weights}}),
                Json::array({"candidates", "client_request_id"})),
            store,
            true));
    }
    if (discovery && discovery->reviewAvailable()) {
        auto artifactReference = [](const std::string& artifactId) {
            return objectSchema(
                {{"job_id", {{"type", "string"}, {"minLength", 1}}}, {"artifact_id", {{"type", "string"}, {"const", artifactId}}}},
                Json::array({"job_id", "artifact_id"}));
        };
        tools.register_tool(makeTool(
            "review_create_queue",
            "Create an immutable prioritized review queue from evidence ranking and optional structural divergence regions",
            objectSchema(
                withBase(
                    {{"ranking", artifactReference("evidence-ranking")},
                     {"comparison", artifactReference("structural-comparison")},
                     {"max_items", {{"type", "integer"}, {"minimum", 1}, {"maximum", 100}}},
                     {"divergence_percentile", {{"type", "number"}, {"minimum", 50}, {"maximum", 99.9}}}}),
                Json::array({"ranking", "client_request_id"})),
            store,
            true));
        const Json assessment = objectSchema(
            {{"item_id", {{"type", "string"}, {"minLength", 1}, {"maxLength", 128}}},
             {"decision", {{"type", "string"}, {"enum", Json::array({"accept", "reject", "uncertain", "defer"})}}},
             {"confidence", {{"type", "number"}, {"minimum", 0}, {"maximum", 1}}},
             {"reason_codes",
              {{"type", "array"},
               {"maxItems", 8},
               {"items", {{"type", "string"}, {"minLength", 1}, {"maxLength", 64}, {"pattern", "^[A-Za-z0-9._-]+$"}}}}},
             {"notes", {{"type", "string"}, {"maxLength", 1000}}}},
            Json::array({"item_id", "decision"}));
        tools.register_tool(makeTool(
            "review_record_assessment",
            "Create a new immutable reviewer-assessment artifact without mutating its source queue",
            objectSchema(
                withBase(
                    {{"queue", artifactReference("review-queue")},
                     {"reviewer_id", {{"type", "string"}, {"minLength", 1}, {"maxLength", 64}, {"pattern", "^[A-Za-z0-9._-]+$"}}},
                     {"assessments", {{"type", "array"}, {"minItems", 1}, {"maxItems", 100}, {"items", assessment}}}}),
                Json::array({"queue", "reviewer_id", "assessments", "client_request_id"})),
            store,
            true));
        tools.register_tool(makeTool(
            "metric_evaluate_labels",
            "Evaluate queue priorities against supplied reviewer labels with coverage, ranking metrics, calibration, and agreement",
            objectSchema(
                withBase({{"assessments", {{"type", "array"}, {"minItems", 1}, {"maxItems", 16}, {"items", artifactReference("review-assessment")}}}}),
                Json::array({"assessments", "client_request_id"})),
            store,
            true));
    }
    if (discovery && discovery->nnunetAvailable()) {
        const Json xyzMetadata = {{"type", "array"}, {"minItems", 3}, {"maxItems", 3}, {"items", {{"type", "number"}}}};
        Json source = objectSchema(
            {{"kind", {{"type", "string"}, {"enum", Json::array({"local_zarr", "remote_zarr"})}}},
             {"path", path},
             {"uri", path},
             {"array_path", {{"type", "string"}, {"minLength", 1}, {"maxLength", 128}}},
             {"scale", {{"type", "integer"}, {"minimum", 0}, {"maximum", 16}}},
             {"voxel_spacing", xyzMetadata},
             {"voxel_spacing_unit", {{"type", "string"}, {"const", "um"}}},
             {"origin_xyz", xyzMetadata}},
            Json::array({"kind"}));
        source["oneOf"] = Json::array({Json{{"required", Json::array({"path"})}}, Json{{"required", Json::array({"uri"})}}});
        const Json region = objectSchema(
            {{"x", {{"type", "integer"}, {"minimum", 0}}},
             {"y", {{"type", "integer"}, {"minimum", 0}}},
             {"z", {{"type", "integer"}, {"minimum", 0}}},
             {"width", {{"type", "integer"}, {"minimum", 1}, {"maximum", 256}}},
             {"height", {{"type", "integer"}, {"minimum", 1}, {"maximum", 256}}},
             {"depth", {{"type", "integer"}, {"minimum", 1}, {"maximum", 256}}},
             {"space", {{"type", "string"}, {"enum", Json::array({"ct_l0_xyz", "ct_l2_xyz"})}}}},
            Json::array({"x", "y", "z", "width", "height", "depth", "space"}));
        Json schema = objectSchema(
            withBase(
                {{"volume_path", path},
                 {"source", source},
                 {"region", region},
                 {"model", {{"type", "string"}, {"const", "vc-surface-nnunet-058"}}},
                 {"device", {{"type", "string"}, {"enum", Json::array({"cpu", "mps"})}}},
                 {"tile_size", {{"type", "integer"}, {"enum", Json::array({64, 96, 128})}}},
                 {"overlap", {{"type", "number"}, {"minimum", 0}, {"maximum", 0.75}}},
                 {"threshold", {{"type", "number"}, {"minimum", 0}, {"maximum", 1}}},
                 {"cpu_threads", {{"type", "integer"}, {"minimum", 1}, {"maximum", 16}}},
                 {"checkpoint_sha256",
                  {{"type", "string"}, {"const", "8b90543a3b8063d1158467364fcf825527fb18edc3af852ffcb91906f0e3e763"}}}}),
            Json::array({"model", "device", "checkpoint_sha256", "client_request_id"}));
        schema["oneOf"] =
            Json::array({Json{{"required", Json::array({"volume_path"})}}, Json{{"required", Json::array({"source", "region"})}}});
        tools.register_tool(
            makeTool("volume_run_segmentation", "Run Dataset 058 segmentation from bounded local files or staged local/remote Zarr ROIs", std::move(schema), store, true));
    }
    tools.register_tool(makeTool(
        "vc_render_surface_diagnostics",
        "Compute bounded CPU diagnostics from an existing surface-volume TIFF stack",
        objectSchema(withBase({{"surface_volume_path", path}}), Json::array({"surface_volume_path", "client_request_id"})),
        store));
    tools.register_tool(makeTool(
        "ink_compute_classical_features",
        "Compute deterministic classical CPU feature maps (not ink probabilities)",
        objectSchema(withBase({{"diagnostics_path", path}}), Json::array({"diagnostics_path", "client_request_id"})),
        store));
    tools.register_tool(makeTool(
        "ink_find_candidate_regions",
        "Rank connected candidate regions from a heuristic score image",
        objectSchema(
            withBase(
                {{"score_path", path},
                 {"threshold", {{"type", "number"}, {"minimum", -1}, {"maximum", 255}}},
                 {"min_area", {{"type", "integer"}, {"minimum", 1}, {"maximum", 1000000}}},
                 {"max_candidates", {{"type", "integer"}, {"minimum", 1}, {"maximum", 500}}}}),
            Json::array({"score_path", "client_request_id"})),
        store));
    tools.register_tool(makeTool(
        "ink_render_candidate_report",
        "Render bounded context crops for ranked candidate regions",
        objectSchema(
            withBase(
                {{"candidate_set_path", path},
                 {"max_candidates", {{"type", "integer"}, {"minimum", 1}, {"maximum", 100}}},
                 {"context_pixels", {{"type", "integer"}, {"minimum", 0}, {"maximum", 512}}}}),
            Json::array({"candidate_set_path", "client_request_id"})),
        store));
    tools.register_tool(makeTool(
        "text_analyze_layout",
        "Compute deterministic component, skeleton, and line-layout evidence without OCR",
        objectSchema(
            withBase({{"mask_path", path}, {"threshold", {{"type", "integer"}, {"minimum", 0}, {"maximum", 255}}}}),
            Json::array({"mask_path", "client_request_id"})),
        store));
    if (discovery && discovery->dinov3Available()) {
        const Json point = objectSchema({{"x", {{"type", "number"}}}, {"y", {{"type", "number"}}}}, Json::array({"x", "y"}));
        const Json bbox = objectSchema(
            {{"x", {{"type", "integer"}, {"minimum", 0}}},
             {"y", {{"type", "integer"}, {"minimum", 0}}},
             {"width", {{"type", "integer"}, {"minimum", 16}, {"maximum", 2048}}},
             {"height", {{"type", "integer"}, {"minimum", 16}, {"maximum", 2048}}}},
            Json::array({"x", "y", "width", "height"}));
        tools.register_tool(makeTool(
            "dinov3_exemplar_search",
            "Rerank a bounded 2D diagnostic image using pinned DINOv3 ViT-S dense features on CPU",
            objectSchema(
                withBase(
                    {{"image_path", path},
                     {"repository_path", path},
                     {"repository_commit", {{"type", "string"}, {"pattern", "^[0-9a-f]{40}$"}}},
                     {"weights_path", path},
                     {"weights_sha256", {{"type", "string"}, {"pattern", "^[0-9a-f]{64}$"}}},
                     {"model", {{"type", "string"}, {"enum", Json::array({"dinov3_vits16", "dinov3_vits16plus"})}}},
                     {"search_bbox", bbox},
                     {"positive_examples", {{"type", "array"}, {"minItems", 1}, {"maxItems", 32}, {"items", point}}},
                     {"negative_examples", {{"type", "array"}, {"maxItems", 32}, {"items", point}}},
                     {"top_k", {{"type", "integer"}, {"minimum", 1}, {"maximum", 500}}},
                     {"cpu_threads", {{"type", "integer"}, {"minimum", 1}, {"maximum", 64}}}}),
                Json::array(
                    {"image_path",
                     "repository_path",
                     "repository_commit",
                     "weights_path",
                     "weights_sha256",
                     "positive_examples",
                     "client_request_id"})),
            store));
    }
    if (discovery && discovery->inkModelAvailable()) {
        const Json surfaceVolume = objectSchema(
            {{"job_id", {{"type", "string"}, {"minLength", 1}}}, {"artifact_id", {{"type", "string"}, {"const", "surface-volume"}}}},
            Json::array({"job_id", "artifact_id"}));
        tools.register_tool(makeTool(
            "ink_run_resnet152_inference",
            "Produce an uncalibrated ink-model score with the pinned ResNet152/3D-decoder checkpoint on a validated HxWx62 surface volume",
            objectSchema(
                withBase(
                    {{"surface_volume", surfaceVolume},
                     {"model_profile", {{"type", "string"}, {"const", "resnet152-3d-decoder-62"}}},
                     {"device", {{"type", "string"}, {"enum", Json::array({"cpu", "mps", "cuda"})}}},
                     {"tile_size", {{"type", "integer"}, {"enum", Json::array({64, 128, 256})}}},
                     {"stride", {{"type", "integer"}, {"minimum", 1}, {"maximum", 256}}},
                     {"reverse_layers", {{"type", "boolean"}}}}),
                Json::array({"surface_volume", "model_profile", "client_request_id"})),
            store,
            true));
    }
    if (discovery && discovery->dinovolAvailable()) {
        const Json point = objectSchema(
            {{"u", {{"type", "number"}}}, {"v", {{"type", "number"}}}, {"offset", {{"type", "number"}}}}, Json::array({"u", "v", "offset"}));
        const Json registeredSurface = objectSchema(
            {{"job_id", {{"type", "string"}, {"minLength", 1}}}, {"artifact_id", {{"type", "string"}, {"const", "registered-surface"}}}},
            Json::array({"job_id", "artifact_id"}));
        tools.register_tool(makeTool(
            "dinovol_exemplar_search",
            "Run pinned Dinovol teacher-backbone exemplar similarity on bounded raw CT with projection to a registered TIFXYZ chart",
            objectSchema(
                withBase(
                    {{"surface", registeredSurface},
                     {"device", {{"type", "string"}, {"const", "mps"}}},
                     {"positive_examples", {{"type", "array"}, {"minItems", 1}, {"maxItems", 32}, {"items", point}}},
                     {"negative_examples", {{"type", "array"}, {"maxItems", 32}, {"items", point}}}}),
                Json::array({"surface", "device", "positive_examples", "client_request_id"})),
            store,
            true));
    }
}
}  // namespace vc::mcp
