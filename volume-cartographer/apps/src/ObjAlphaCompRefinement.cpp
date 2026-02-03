#include <nlohmann/json.hpp>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/StreamOperators.hpp"
#include "vc/core/util/Surface.hpp"

#include <filesystem>
#include <fstream>
#include <optional>
#include <vector>
#include <iostream>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <cstdint>

enum class SurfaceInputType {
    Obj,
    Tifxyz
};

struct RefinementConfig {
    bool refine = true;
    float start = -2.0f;
    float stop = 30.0f;
    float step = 2.0f;
    float low = 118.0f / 255.0f;
    float high = 165.0f / 255.0f;
    float border_off = 1.0f;
    int   r = 3;
    bool  gen_vertexcolor = false;
    bool  overwrite = true;
    float reader_scale = 0.5f;
    std::string dataset_group = "1";
    std::size_t cache_bytes = static_cast<std::size_t>(10e9);
};

static RefinementConfig parse_config(const nlohmann::json& params)
{
    RefinementConfig cfg;
    auto try_set = [&params](const char* key, auto& out) {
        if (params.contains(key)) {
            out = params.at(key).get<std::decay_t<decltype(out)>>();
        }
    };

    try_set("refine", cfg.refine);
    try_set("start", cfg.start);
    try_set("stop", cfg.stop);
    try_set("step", cfg.step);
    try_set("border_off", cfg.border_off);
    try_set("r", cfg.r);
    try_set("gen_vertexcolor", cfg.gen_vertexcolor);
    try_set("overwrite", cfg.overwrite);
    try_set("reader_scale", cfg.reader_scale);
    if (params.contains("cache_bytes")) {
        const auto bytes = params.at("cache_bytes").get<std::uint64_t>();
        cfg.cache_bytes = static_cast<std::size_t>(bytes);
    } else if (params.contains("cache_mb")) {
        const auto mb = params.at("cache_mb").get<double>();
        cfg.cache_bytes = static_cast<std::size_t>(std::max(1.0, mb) * 1024.0 * 1024.0);
    } else if (params.contains("cache_gb")) {
        const auto gb = params.at("cache_gb").get<double>();
        cfg.cache_bytes = static_cast<std::size_t>(std::max(0.001, gb) * 1024.0 * 1024.0 * 1024.0);
    }

    if (params.contains("low")) {
        double v = params.at("low").get<double>();
        if (v > 1.0) v /= 255.0;
        cfg.low = static_cast<float>(std::clamp(v, 0.0, 1.0));
    }
    if (params.contains("high")) {
        double v = params.at("high").get<double>();
        if (v > 1.0) v /= 255.0;
        cfg.high = static_cast<float>(std::clamp(v, 0.0, 1.0));
    }

    if (params.contains("scale_group")) {
        const auto& v = params.at("scale_group");
        cfg.dataset_group = v.is_string() ? v.get<std::string>()
                                          : std::to_string(v.get<int>());
    } else if (params.contains("scale_level")) {
        const auto& v = params.at("scale_level");
        cfg.dataset_group = v.is_string() ? v.get<std::string>()
                                          : std::to_string(v.get<int>());
    }

    return cfg;
}

static SurfaceInputType detect_surface_type(const std::filesystem::path& src)
{
    if (src.extension() == ".obj" && std::filesystem::is_regular_file(src)) {
        return SurfaceInputType::Obj;
    }

    if (std::filesystem::is_directory(src)) {
        const bool hasX = std::filesystem::exists(src / "x.tif");
        const bool hasY = std::filesystem::exists(src / "y.tif");
        const bool hasZ = std::filesystem::exists(src / "z.tif");
        if (hasX && hasY && hasZ) {
            return SurfaceInputType::Tifxyz;
        }
    }

    throw std::runtime_error("unsupported surface input: " + src.string());
}

cv::Vec3f parse_vec3f(std::string line, std::string type = "")
{
    cv::Vec3f v;
    std::istringstream iss(line);
    std::string t;
    if (!(iss >> t >> v[0] >> v[1] >> v[2]) || (type.size() && t != type)) {
        std::cout << t << v << type << line << std::endl;
        throw std::runtime_error("error in parse_vec3f()");
    }
    return v;
}

bool istype(const std::string &line, const std::string &type)
{
    return line.rfind(type+" ", 0) == 0;
}


struct DSReader {
    z5::Dataset* ds;
    float scale;
    ChunkCache<uint8_t>* cache;
    std::mutex read_mutex;
};

float alphacomp_offset(DSReader &reader, cv::Vec3f point, cv::Vec3f normal, float start, float stop, float step, float low, float high, int r)
{
    int d = 2*r+1;
    cv::Size size = {d,d};
    cv::Point2i c = {r,r};

    float transparent = 1;
    cv::Mat_<float> blur(size, 0);
    float integ_z = 0;

    cv::Mat_<cv::Vec3f> coords;
    PlaneSurface plane(point, normal);
    plane.gen(&coords, nullptr, size, cv::Vec3f(0,0,0), reader.scale, {0,0,0});

    coords *= reader.scale;
    float s = copysignf(1.0,step);

    for(double off=start;off*s<=stop*s;off+=step) {
        cv::Mat_<uint8_t> slice;
        //I hate opencv
        cv::Mat_<cv::Vec3f> offmat(size, normal*off*reader.scale);
        {
            std::lock_guard<std::mutex> lock(reader.read_mutex);
            readInterpolated3D(slice, reader.ds, coords+offmat, reader.cache);
        }

        cv::Mat floatslice;
        slice.convertTo(floatslice, CV_32F, 1/255.0);

        cv::GaussianBlur(floatslice, blur, {d,d}, 0);
        cv::Mat_<float> opaq_slice = blur;

        opaq_slice = (opaq_slice-low)/(high-low);
        opaq_slice = cv::min(opaq_slice,1);
        opaq_slice = cv::max(opaq_slice,0);

        float joint = transparent*opaq_slice(c);
        integ_z += joint * off;
        transparent = transparent-joint;
    }

    integ_z /= (1-transparent+1e-5);

    return integ_z;
}

static inline bool normal_is_valid(const cv::Vec3f& n)
{
    return std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2]) &&
           !(std::abs(n[0]) < 1e-6f && std::abs(n[1]) < 1e-6f && std::abs(n[2]) < 1e-6f);
}

static void refine_vertices(std::vector<cv::Vec3f>& vertices,
                            const std::vector<cv::Vec3f>& normals,
                            const RefinementConfig& cfg,
                            DSReader& reader)
{
    if (!cfg.refine) {
        return;
    }

#pragma omp parallel for
    for (int j = 0; j < static_cast<int>(vertices.size()); ++j) {
        const auto& normal = normals[j];
        if (!normal_is_valid(normal)) {
            continue;
        }
        const float off = alphacomp_offset(reader,
                                           vertices[j],
                                           normal,
                                           cfg.start,
                                           cfg.stop,
                                           cfg.step,
                                           cfg.low,
                                           cfg.high,
                                           cfg.r);
        vertices[j] += normal * (off + cfg.border_off);
    }
}

int process_obj(const std::string& src,
                const std::string& tgt,
                DSReader& reader,
                const RefinementConfig& cfg)
{
    std::ifstream obj(src);
    std::ofstream out(tgt);
    std::string line;
    std::string last_line;
    int v_count = 0;
    int vn_count = 0;
    std::vector<cv::Vec3f> vs;
    std::vector<cv::Vec3f> vns;

    while (std::getline(obj, line))
    {
        if (istype(line, "v")) {
            if (vs.size() != vns.size())
                throw std::runtime_error("sorry our taste in obj is quite peculiar ...");
            vs.push_back(parse_vec3f(line));
        }
        if (istype(line, "vn")) {
            if (vs.size()-1 != vns.size())
                throw std::runtime_error("sorry our taste in obj is quite peculiar ...");
                cv::Vec3f normal = parse_vec3f(line);
                normalize(normal, normal);
                vns.push_back(normal);
        }
        // if (vs.size() % 10000 == 0)
            // std::cout << vs.size() << std::endl;
    }

    if (vs.size() != vns.size())
        throw std::runtime_error("sorry our taste in obj is quite peculiar ...");

    refine_vertices(vs, vns, cfg, reader);

    cv::Mat_<uint8_t> slice;
    bool vertexcolor = cfg.gen_vertexcolor;
    if (vertexcolor) {
        std::lock_guard<std::mutex> lock(reader.read_mutex);
        cv::Mat_<cv::Vec3f> vs_mat(static_cast<int>(vs.size()), 1, vs.data());
        readInterpolated3D(slice, reader.ds, vs_mat*reader.scale, reader.cache);
    }

    obj.clear();
    obj.seekg(0);
    int v_counter = 0;
    while (std::getline(obj, line))
    {
        if (istype(line, "v")) {
            cv::Vec3f v = vs[v_counter];
            out << "v " << v[0] << " " << v[1] << " " << v[2];
            if (vertexcolor) {
                float col = int(slice(v_counter, 0)*1000/255.0)/1000.0;
                out << " " << col << " " << col << " " << col;
            }
            out << "\n";
            v_counter++;
        }
        else
            out << line << "\n";
    }

    return EXIT_SUCCESS;
}

int process_tifxyz(const std::filesystem::path& src,
                   const std::filesystem::path& dst,
                   DSReader& reader,
                   const RefinementConfig& cfg)
{
    std::unique_ptr<QuadSurface> surf;
    try {
        surf = load_quad_from_tifxyz(src.string());
    } catch (const std::exception& e) {
        std::cerr << "failed to load tifxyz: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat_<cv::Vec3f> original = surf->rawPoints().clone();
    if (original.empty()) {
        std::cerr << "tifxyz surface has no points\n";
        return EXIT_FAILURE;
    }

    cv::Mat_<cv::Vec3f> refined = original.clone();

    if (cfg.refine) {
        const int rows = original.rows;
        const int cols = original.cols;

#pragma omp parallel for collapse(2)
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (!loc_valid(original, cv::Vec2d(y, x))) {
                    continue;
                }
                const cv::Vec3f normal = surf->gridNormal(y, x);
                if (!normal_is_valid(normal)) {
                    continue;
                }
                const cv::Vec3f& point = original(y, x);
                const float off = alphacomp_offset(reader,
                                                   point,
                                                   normal,
                                                   cfg.start,
                                                   cfg.stop,
                                                   cfg.step,
                                                   cfg.low,
                                                   cfg.high,
                                                   cfg.r);
                refined(y, x) = point + normal * (off + cfg.border_off);
            }
        }
    }

    QuadSurface outSurf(refined, surf->scale());
    std::string uuid = !surf->id.empty() ? surf->id : dst.filename().string();
    if (uuid.empty()) {
        uuid = "refined_surface";
    }
    outSurf.id = uuid;

    if (surf->meta) {
        outSurf.meta = std::make_unique<nlohmann::json>(*surf->meta);
    }

    try {
        outSurf.save(dst.string(), uuid, cfg.overwrite);
    } catch (const std::exception& e) {
        std::cerr << "failed to save tifxyz: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if (cfg.gen_vertexcolor) {
        std::cerr << "warning: gen_vertexcolor not supported for tifxyz output; ignoring flag\n";
    }

    return EXIT_SUCCESS;
}

class MeasureLife
{
public:
    MeasureLife(std::string msg)
    {
        std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    ~MeasureLife()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

int main(int argc, char *argv[])
{
    if (argc != 5) {
        std::cout << "usage: " << argv[0]
                  << " <zarr-volume> <src-surface> <out-surface> <json-params>" << std::endl;
        return EXIT_SUCCESS;
    }

    const std::filesystem::path vol_path    = argv[1];
    const std::filesystem::path surface_src = argv[2];
    const std::filesystem::path surface_dst = argv[3];
    const std::filesystem::path params_path = argv[4];

    std::ifstream params_f(params_path);
    if (!params_f) {
        std::cerr << "failed to open params json: " << params_path << std::endl;
        return EXIT_FAILURE;
    }

    const nlohmann::json params = nlohmann::json::parse(params_f);
    const RefinementConfig cfg = parse_config(params);

    z5::filesystem::handle::Group group(vol_path.string(), z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, cfg.dataset_group, "/");
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    std::cout << "zarr dataset size for scale group " << cfg.dataset_group << " " << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;
    std::cout << "chunk cache size (bytes) " << cfg.cache_bytes << std::endl;
    ChunkCache<uint8_t> chunk_cache(cfg.cache_bytes);

    DSReader reader = {ds.get(), cfg.reader_scale, &chunk_cache};

    MeasureLife timer("processing surface ...\n");

    int ret = EXIT_FAILURE;
    const SurfaceInputType inputType = detect_surface_type(surface_src);

    switch (inputType) {
        case SurfaceInputType::Obj:
            ret = process_obj(surface_src.string(), surface_dst.string(), reader, cfg);
            break;
        case SurfaceInputType::Tifxyz:
            ret = process_tifxyz(surface_src, surface_dst, reader, cfg);
            break;
    }

    return ret;
}
