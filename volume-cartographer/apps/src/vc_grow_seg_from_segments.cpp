#include <iostream>
#include "utils/Json.hpp"

#include "vc/core/types/VcDataset.hpp"

#include <boost/program_options.hpp>
#include <opencv2/core.hpp>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/StreamOperators.hpp"
#include "vc/tracer/Tracer.hpp"


using shape = std::vector<size_t>;


using Json = utils::Json;
namespace po = boost::program_options;

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

static void add_internal_volume_shape(utils::Json& params, const std::vector<size_t>& shape)
{
    if (shape.size() != 3) {
        throw std::runtime_error("Expected zarr dataset shape [z, y, x]");
    }

    auto volume_shape = utils::Json::array();
    volume_shape.push_back(static_cast<int>(shape[0]));
    volume_shape.push_back(static_cast<int>(shape[1]));
    volume_shape.push_back(static_cast<int>(shape[2]));
    params["_volume_shape"] = std::move(volume_shape);
}




int main(int argc, char *argv[])
{
    std::filesystem::path vol_path, src_dir, tgt_dir, params_path, src_path;

    bool use_old_args = argc == 6 && argv[1][0] != '-' && argv[2][0] != '-' &&
        argv[3][0] != '-' && argv[4][0] != '-' && argv[5][0] != '-';

    if (use_old_args) {
        vol_path = argv[1];
        src_dir = argv[2];
        tgt_dir = argv[3];
        params_path = argv[4];
        src_path = argv[5];
    } else {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("volume,v", po::value<std::string>()->required(), "OME-Zarr volume path")
            ("src-dir,s", po::value<std::string>()->required(), "Directory containing source segments")
            ("target-dir,t", po::value<std::string>()->required(), "Target directory for output traces")
            ("params,p", po::value<std::string>()->required(), "JSON parameters file")
            ("src-segment", po::value<std::string>()->required(), "Source segment path to grow from");

        po::variables_map vm;
        try {
            po::store(po::parse_command_line(argc, argv, desc), vm);

            if (vm.count("help")) {
                std::cout << desc << std::endl;
                return EXIT_SUCCESS;
            }

            po::notify(vm);
        } catch (const po::error& e) {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << "usage: " << argv[0]
                      << " --volume <zarr-volume>"
                      << " --src-dir <src-dir>"
                      << " --target-dir <tgt-dir>"
                      << " --params <json-params>"
                      << " --src-segment <src-segment>" << std::endl << std::endl;
            std::cerr << desc << std::endl;
            return EXIT_FAILURE;
        }

        vol_path = vm["volume"].as<std::string>();
        src_dir = vm["src-dir"].as<std::string>();
        tgt_dir = vm["target-dir"].as<std::string>();
        params_path = vm["params"].as<std::string>();
        src_path = vm["src-segment"].as<std::string>();
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

    std::unique_ptr<vc::VcDataset> ds = std::make_unique<vc::VcDataset>(vol_path / "0");

    std::cout << "zarr dataset size for scale group 0 " << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->defaultChunkShape() << std::endl;
    add_internal_volume_shape(params, ds->shape());

    float voxelsize = Json::parse_file(vol_path/"meta.json")["voxelsize"].get_float();

    std::string name_prefix = "auto_grown_";
    std::vector<QuadSurface*> surfaces;

    std::filesystem::path meta_fn = src_path / "meta.json";
    if (!std::filesystem::exists(meta_fn)) {
        std::cerr << "Error: meta.json not found at " << meta_fn << std::endl;
        return EXIT_FAILURE;
    }

    utils::Json meta = utils::Json::parse_file(meta_fn);
    QuadSurface *src = new QuadSurface(src_path, meta);

    for (const auto& entry : std::filesystem::directory_iterator(src_dir))
        if (std::filesystem::is_directory(entry)) {
            std::string name = entry.path().filename();
            if (name.compare(0, name_prefix.size(), name_prefix))
                continue;

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
