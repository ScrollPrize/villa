#include <nlohmann/json.hpp>

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(io, xio.hpp)
#include XTENSORINCLUDE(views, xview.hpp)

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

#include <opencv2/core.hpp>

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
    if (argc != 6) {
        std::cout << "usage: " << argv[0] << " <zarr-volume> <src-dir> <tgt-dir> <json-params> <src-segment>" << std::endl;
        return EXIT_SUCCESS;
    }

    std::filesystem::path vol_path = argv[1];
    std::filesystem::path src_dir = argv[2];
    std::filesystem::path tgt_dir = argv[3];
    std::filesystem::path params_path = argv[4];
    std::filesystem::path src_path = argv[5];
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

    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, "0", json::parse(std::ifstream(vol_path/"0/.zarray")).value<std::string>("dimension_separator","."));
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    std::cout << "zarr dataset size for scale group 0 " << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;

    float voxelsize = json::parse(std::ifstream(vol_path/"meta.json"))["voxelsize"];

    std::string name_prefix = "auto_grown_";
    std::vector<SurfaceMeta*> surfaces;

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
    SurfaceMeta *src = new SurfaceMeta(src_path, meta);
    src->readOverlapping();

    for (const auto& entry : std::filesystem::directory_iterator(src_dir))
        if (std::filesystem::is_directory(entry)) {
            std::string name = entry.path().filename();
            if (name.compare(0, name_prefix.size(), name_prefix))
                continue;

            std::filesystem::path meta_fn = entry.path() / "meta.json";
            if (!std::filesystem::exists(meta_fn))
                continue;

            std::ifstream meta_f(meta_fn);
            json meta = json::parse(meta_f);

            if (!meta.count("bbox"))
                continue;

            if (meta.value("format","NONE") != "tifxyz")
                continue;

            SurfaceMeta *sm;
            if (entry.path().filename() == src->name())
                sm = src;
            else {
                sm = new SurfaceMeta(entry.path(), meta);
                sm->readOverlapping();
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
