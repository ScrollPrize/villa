#include <iostream>
#include "utils/Json.hpp"

#include "vc/core/types/VcDataset.hpp"

#include <opencv2/core.hpp>

#include "vc/core/util/ArgParse.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"

#include <filesystem>
#include <fstream>

#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/StreamOperators.hpp"
#include "vc/tracer/Tracer.hpp"


using shape = std::vector<size_t>;


using Json = utils::Json;

static void add_target_context(utils::Json& meta, const std::filesystem::path& volume_path)
{
    std::filesystem::path normalized_volume_path = volume_path.lexically_normal();
    if (normalized_volume_path.filename().empty()) {
        normalized_volume_path = normalized_volume_path.parent_path();
    }

    const std::string volume_name = normalized_volume_path.filename().string();
    if (!volume_name.empty() && !meta.contains("target_volume")) {
        meta["target_volume"] = volume_name;
    }

    const std::filesystem::path volumes_dir = normalized_volume_path.parent_path();
    if (volumes_dir.filename() != "volumes") {
        return;
    }

    const std::filesystem::path volpkg_root = volumes_dir.parent_path();
    std::string scroll_name = volpkg_root.filename().string();

    const std::filesystem::path config_path = volpkg_root / "config.json";
    std::error_code ec;
    if (std::filesystem::is_regular_file(config_path, ec)) {
        try {
            auto cfg = utils::Json::parse_file(config_path);
            if (cfg.contains("name") && cfg["name"].is_string()) {
                scroll_name = cfg["name"].get_string();
            }
        } catch (...) {
            // Keep folder-based fallback if config.json cannot be parsed.
        }
    }

    if (!scroll_name.empty() && !meta.contains("scroll_source")) {
        meta["scroll_source"] = scroll_name;
    }
}




int main(int argc, char *argv[])
{
    vc::cli::ArgParser parser;
    parser.add_option("volume", {"v"}, true, "Path to zarr volume");
    parser.add_option("src-dir", {"s"}, true, "Source directory containing segments");
    parser.add_option("target-dir", {"t"}, true, "Target directory for output");
    parser.add_option("params", {"p"}, true, "Path to JSON parameters file");
    parser.add_option("src-segment", {}, false, "Source segment path");
    parser.add_option("resume", {"r"}, false, "Resume from a previous output surface (becomes source segment)");

    std::string error;
    auto args = parser.parse(argc, argv, &error);
    if (!error.empty()) {
        std::cerr << error << "\n\n" << parser.help_text("Usage: " + std::string(argv[0]));
        return EXIT_FAILURE;
    }

    std::filesystem::path vol_path = args.value("volume");
    std::filesystem::path src_dir = args.value("src-dir");
    std::filesystem::path tgt_dir = args.value("target-dir");
    std::filesystem::path params_path = args.value("params");

    // Handle src-segment vs resume options
    std::filesystem::path src_path;
    bool resume_mode = args.has("resume");

    if (resume_mode && args.has("src-segment")) {
        std::cerr << "Error: Cannot specify both --src-segment and --resume\n";
        return EXIT_FAILURE;
    }

    if (resume_mode) {
        src_path = args.value("resume");
    } else if (args.has("src-segment")) {
        src_path = args.value("src-segment");
    } else {
        std::cerr << "Error: Must specify either --src-segment or --resume\n\n"
                  << parser.help_text("Usage: " + std::string(argv[0]));
        return EXIT_FAILURE;
    }

    while (src_path.filename().empty())
        src_path = src_path.parent_path();

    std::ifstream params_f(params_path);
    Json params = Json::parse_file(params_path);
    // Honor optional CUDA toggle from params (default true)
    if (params.contains("use_cuda")) {
        set_space_tracing_use_cuda(params.value("use_cuda", true));
    } else {
        set_space_tracing_use_cuda(true);
    }
    params["tgt_dir"] = tgt_dir;
    if (resume_mode) {
        params["resume"] = true;
    }

    std::unique_ptr<vc::VcDataset> ds = std::make_unique<vc::VcDataset>(vol_path / "0");

    std::cout << "zarr dataset size for scale group 0 " << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->defaultChunkShape() << std::endl;

    float voxelsize = Json::parse_file(vol_path/"meta.json")["voxelsize"].get_float();

    std::vector<QuadSurface*> surfaces;

    std::filesystem::path meta_fn = src_path / "meta.json";
    if (!std::filesystem::exists(meta_fn)) {
        std::cerr << "Error: meta.json not found at " << meta_fn << std::endl;
        return EXIT_FAILURE;
    }

    utils::Json meta = utils::Json::parse_file(meta_fn);
    QuadSurface *src = new QuadSurface(src_path, meta);
    src->readOverlappingJson();

    for (const auto& entry : std::filesystem::directory_iterator(src_dir))
        if (std::filesystem::is_directory(entry)) {
            std::filesystem::path meta_fn = entry.path() / "meta.json";
            if (!std::filesystem::exists(meta_fn))
                continue;

            utils::Json meta = utils::Json::parse_file(meta_fn);

            if (!meta.count("bbox"))
                continue;

            if (meta.value("format", std::string{"NONE"}) != "tifxyz")
                continue;

            QuadSurface *sm;
            if (entry.path().filename() == src->id)
                sm = src;
            else {
                sm = new QuadSurface(entry.path(), meta);
                sm->readOverlappingJson();
            }

            surfaces.push_back(sm);
        }

    QuadSurface *surf = grow_surf_from_surfs(src, surfaces, params, voxelsize);

    if (!surf)
        return EXIT_SUCCESS;

    surf->meta["source"] = "vc_grow_seg_from_segments";
    surf->meta["vc_grow_seg_from_segments_params"] = utils::Json::parse(params.dump());
    add_target_context(surf->meta, vol_path);
    std::string uuid = "auto_trace_" + get_surface_time_str();;
    std::filesystem::path seg_dir = tgt_dir / uuid;
    surf->save(seg_dir, uuid);

    delete surf;
    for(auto sm : surfaces) {
        delete sm;
    }

    return EXIT_SUCCESS;
}
