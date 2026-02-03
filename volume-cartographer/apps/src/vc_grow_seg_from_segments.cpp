#include <nlohmann/json.hpp>

#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

#include <opencv2/core.hpp>

#include "vc/core/util/ArgParse.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"

#include <filesystem>

#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/StreamOperators.hpp"
#include "vc/tracer/Tracer.hpp"


using shape = z5::types::ShapeType;
using namespace xt::placeholders;


using json = nlohmann::json;




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
    json params = json::parse(params_f);
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

    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, "0", json::parse(std::ifstream(vol_path/"0/.zarray")).value<std::string>("dimension_separator","."));
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    std::cout << "zarr dataset size for scale group 0 " << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;

    float voxelsize = json::parse(std::ifstream(vol_path/"meta.json"))["voxelsize"];

    std::vector<QuadSurface*> surfaces;

    std::filesystem::path meta_fn = src_path / "meta.json";
    if (!std::filesystem::exists(meta_fn)) {
        std::cerr << "Error: meta.json not found at " << meta_fn << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream meta_f(meta_fn);
    if (!meta_f.is_open() || !meta_f.good()) {
        std::cerr << "Error: Could not open " << meta_fn << std::endl;
        return EXIT_FAILURE;
    }

    json meta = json::parse(meta_f);
    QuadSurface *src = new QuadSurface(src_path, meta);
    src->readOverlappingJson();

    for (const auto& entry : std::filesystem::directory_iterator(src_dir))
        if (std::filesystem::is_directory(entry)) {
            std::filesystem::path meta_fn = entry.path() / "meta.json";
            if (!std::filesystem::exists(meta_fn))
                continue;

            std::ifstream meta_f(meta_fn);
            json meta = json::parse(meta_f);

            if (!meta.count("bbox"))
                continue;

            if (meta.value("format","NONE") != "tifxyz")
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

    (*surf->meta)["source"] = "vc_grow_seg_from_segments";
    (*surf->meta)["vc_grow_seg_from_segments_params"] = params;
    std::string uuid = "auto_trace_" + get_surface_time_str();;
    std::filesystem::path seg_dir = tgt_dir / uuid;
    surf->save(seg_dir, uuid);

    delete surf;
    for(auto sm : surfaces) {
        delete sm;
    }

    return EXIT_SUCCESS;
}
