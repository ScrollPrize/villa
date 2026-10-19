#include "utils/Json.hpp"
#include "vc_objrefine_offset_report.hpp"


#include "vc/core/types/Volume.hpp"
#include "vc/core/render/ChunkCache.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/StreamOperators.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Tiff.hpp"

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
#include <limits>

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
    // tifxyz only: also save each vertex's offset and opacity as a report
    // (see docs/objrefine.md). It never changes the surface written.
    bool  offset_report = false;
    float offset_threshold = 5.0f;
    float offset_min_opacity = 0.5f;
};

static RefinementConfig parse_config(const utils::Json& params)
{
    RefinementConfig cfg;
    if (params.contains("refine")) cfg.refine = params.at("refine").get_bool();
    if (params.contains("start")) cfg.start = params.at("start").get_float();
    if (params.contains("stop")) cfg.stop = params.at("stop").get_float();
    if (params.contains("step")) cfg.step = params.at("step").get_float();
    if (params.contains("border_off")) cfg.border_off = params.at("border_off").get_float();
    if (params.contains("r")) cfg.r = params.at("r").get_int();
    if (params.contains("gen_vertexcolor")) cfg.gen_vertexcolor = params.at("gen_vertexcolor").get_bool();
    if (params.contains("overwrite")) cfg.overwrite = params.at("overwrite").get_bool();
    if (params.contains("reader_scale")) cfg.reader_scale = params.at("reader_scale").get_float();
    if (params.contains("offset_report")) cfg.offset_report = params.at("offset_report").get_bool();
    if (params.contains("offset_threshold")) cfg.offset_threshold = params.at("offset_threshold").get_float();
    if (params.contains("offset_min_opacity")) cfg.offset_min_opacity = params.at("offset_min_opacity").get_float();
    if (params.contains("cache_bytes")) {
        const auto bytes = params.at("cache_bytes").get_uint64();
        cfg.cache_bytes = static_cast<std::size_t>(bytes);
    } else if (params.contains("cache_mb")) {
        const auto mb = params.at("cache_mb").get_double();
        cfg.cache_bytes = static_cast<std::size_t>(std::max(1.0, mb) * 1024.0 * 1024.0);
    } else if (params.contains("cache_gb")) {
        const auto gb = params.at("cache_gb").get_double();
        cfg.cache_bytes = static_cast<std::size_t>(std::max(0.001, gb) * 1024.0 * 1024.0 * 1024.0);
    }

    if (params.contains("low")) {
        double v = params.at("low").get_double();
        if (v > 1.0) v /= 255.0;
        cfg.low = static_cast<float>(std::clamp(v, 0.0, 1.0));
    }
    if (params.contains("high")) {
        double v = params.at("high").get_double();
        if (v > 1.0) v /= 255.0;
        cfg.high = static_cast<float>(std::clamp(v, 0.0, 1.0));
    }

    if (params.contains("scale_group")) {
        const auto& v = params.at("scale_group");
        cfg.dataset_group = v.is_string() ? v.get_string()
                                          : std::to_string(v.get_int());
    } else if (params.contains("scale_level")) {
        const auto& v = params.at("scale_level");
        cfg.dataset_group = v.is_string() ? v.get_string()
                                          : std::to_string(v.get_int());
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
    Volume* volume;
    float scale;
    int level;
    std::mutex read_mutex;
};

// Opacity-weighted offset along `normal` of the first papyrus the scan meets.
// `opacity`, when given, receives the accumulated opacity in [0, 1]; near 0
// means no papyrus was found between start and stop.
float alphacomp_offset(DSReader &reader, cv::Vec3f point, cv::Vec3f normal, float start, float stop, float step, float low, float high, int r,
                       float* opacity = nullptr)
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

    float s = copysignf(1.0,step);

    for(double off=start;off*s<=stop*s;off+=step) {
        cv::Mat_<uint8_t> slice;
        cv::Mat_<cv::Vec3f> offmat(size, normal*off);
        {
            std::lock_guard<std::mutex> lock(reader.read_mutex);
            vc::SampleParams sp;
            sp.level = reader.level;
            reader.volume->sample(slice, coords+offmat, sp);
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
    if (opacity) {
        *opacity = 1 - transparent;
    }

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

    if (cfg.offset_report) {
        std::cerr << "warning: offset_report needs tifxyz input (it writes maps on the segment grid); ignoring\n";
    }

    refine_vertices(vs, vns, cfg, reader);

    cv::Mat_<uint8_t> slice;
    bool vertexcolor = cfg.gen_vertexcolor;
    if (vertexcolor) {
        std::lock_guard<std::mutex> lock(reader.read_mutex);
        cv::Mat_<cv::Vec3f> vs_mat(static_cast<int>(vs.size()), 1, vs.data());
        vc::SampleParams sp;
        sp.level = reader.level;
        reader.volume->sample(slice, vs_mat, sp);
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

namespace {

namespace fs = std::filesystem;

fs::path normalizedPath(const fs::path& p)
{
    std::error_code ec;
    fs::path c = fs::weakly_canonical(p, ec);
    if (ec || c.empty()) {
        c = fs::absolute(p).lexically_normal();
    }
    if (!c.has_filename() && c.has_parent_path() && c != c.root_path()) {
        c = c.parent_path();
    }
    return c;
}

bool isReportFile(const fs::path& name)
{
    return name == "offset.tif" || name == "opacity.tif" || name == "report.json";
}

// A report-only run writes a plain folder with the three report files and no
// surface. Refuse any output that is, is inside, or holds a segment, so the
// run can never rewrite or delete a segment's files. Empty when allowed.
std::string reportOnlyRefusal(const fs::path& src, const fs::path& dst, bool overwrite)
{
    const fs::path s = normalizedPath(src);
    const fs::path d = normalizedPath(dst);
    const fs::path rel = d.lexically_relative(s);
    if (!rel.empty() && *rel.begin() != "..") {
        return "the output must not be the source segment or a folder inside it";
    }
    std::error_code ec;
    if (!fs::exists(d, ec)) {
        return {};
    }
    if (!fs::is_directory(d, ec)) {
        return d.string() + " exists and is not a folder";
    }
    bool hasReport = false;
    fs::directory_iterator it(d, ec);
    for (; !ec && it != fs::directory_iterator(); it.increment(ec)) {
        if (!isReportFile(it->path().filename())) {
            return d.string() + " holds " + it->path().filename().string() +
                   "; write the report to a new or empty folder, not into a segment";
        }
        hasReport = true;
    }
    if (ec) {
        return "cannot read " + d.string() + ": " + ec.message();
    }
    if (hasReport && !overwrite) {
        return d.string() + " already holds a report and overwrite is false";
    }
    return {};
}

// Write the maps and report.json into a hidden sibling folder, then swap it
// in, so `dir` ends up with a whole report or none. `dir` holds only report
// files (reportOnlyRefusal) or is the tool's own alphacomp_offset/ folder.
void writeOffsetReport(const fs::path& dir,
                       const cv::Mat_<float>& offsetMap,
                       const cv::Mat_<float>& opacityMap,
                       const utils::Json& report,
                       float dpi)
{
    const fs::path tmp = dir.parent_path() / ("." + dir.filename().string() + ".tmp");
    std::error_code ec;
    fs::remove_all(tmp, ec);
    try {
        fs::create_directories(tmp);
        writeTiff(tmp / "offset.tif", offsetMap, -1, 0, 0, -1.0f, COMPRESSION_NONE, dpi);
        writeTiff(tmp / "opacity.tif", opacityMap, -1, 0, 0, -1.0f, COMPRESSION_NONE, dpi);
        {
            std::ofstream out(tmp / "report.json");
            out << report.dump(4) << std::endl;
            if (!out) {
                throw std::runtime_error("cannot write " + (tmp / "report.json").string());
            }
        }
        fs::remove_all(dir);
        fs::rename(tmp, dir);
    } catch (...) {
        fs::remove_all(tmp, ec);
        throw;
    }
}

}  // namespace

int process_tifxyz(const std::filesystem::path& src,
                   const std::filesystem::path& dst,
                   DSReader& reader,
                   const RefinementConfig& cfg)
{
    // Report-only: measure the source and write only the report, no surface.
    const bool reportOnly = cfg.offset_report && !cfg.refine;
    const std::filesystem::path reportDir =
        reportOnly ? normalizedPath(dst) : dst / "alphacomp_offset";
    if (reportOnly) {
        const std::string why = reportOnlyRefusal(src, dst, cfg.overwrite);
        if (!why.empty()) {
            std::cerr << "offset report: " << why << std::endl;
            return EXIT_FAILURE;
        }
    }

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

    // NaN marks vertices where no ray was cast: the same ones refine leaves
    // in place (loc_valid fails, which includes the last row and column, or
    // gridNormal has no valid normal there).
    cv::Mat_<float> offsetMap;
    cv::Mat_<float> opacityMap;
    if (cfg.offset_report) {
        offsetMap = cv::Mat_<float>(original.size(), std::numeric_limits<float>::quiet_NaN());
        opacityMap = cv::Mat_<float>(original.size(), std::numeric_limits<float>::quiet_NaN());
    }

    if (cfg.refine || cfg.offset_report) {
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
                float opacity = 0.0f;
                const float off = alphacomp_offset(reader,
                                                   point,
                                                   normal,
                                                   cfg.start,
                                                   cfg.stop,
                                                   cfg.step,
                                                   cfg.low,
                                                   cfg.high,
                                                   cfg.r,
                                                   &opacity);
                if (cfg.refine) {
                    refined(y, x) = point + normal * (off + cfg.border_off);
                }
                if (cfg.offset_report) {
                    offsetMap(y, x) = off;
                    opacityMap(y, x) = opacity;
                }
            }
        }
    }

    if (!reportOnly) {
        QuadSurface outSurf(refined, surf->scale());
        outSurf.id = dst.filename().string();

        if (!surf->meta.is_null()) {
            outSurf.meta = surf->meta;
        }

        try {
            outSurf.saveEdited(src, dst, cfg.overwrite);
        } catch (const std::exception& e) {
            std::cerr << "failed to save tifxyz: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }

        if (!cfg.offset_report) {
            // An earlier report described the geometry just replaced. The
            // save already drops subfolders on Linux and macOS, not on Windows.
            std::error_code ec;
            std::filesystem::remove_all(reportDir, ec);
        }

        if (cfg.gen_vertexcolor) {
            std::cerr << "warning: gen_vertexcolor not supported for tifxyz output; ignoring flag\n";
        }
    }

    if (cfg.offset_report) {
        utils::Json report = utils::Json::object();
        report["tool"] = "vc_objrefine";
        report["schema_version"] = 1;
        report["offset_map"] = "offset.tif";
        report["opacity_map"] = "opacity.tif";
        report["units"] = "source surface coordinates (full-resolution voxels) along the grid normal; positive is the normal direction";
        utils::Json inputs = utils::Json::object();
        inputs["volume"] = normalizedPath(reader.volume->path()).string();
        inputs["source_surface"] = normalizedPath(src).string();
        if (surf->meta.is_object() && surf->meta.contains("uuid") &&
            surf->meta["uuid"].is_string()) {
            inputs["source_uuid"] = surf->meta["uuid"].get_string();
        } else {
            inputs["source_uuid"] = nullptr;
        }
        report["inputs"] = std::move(inputs);
        utils::Json params = utils::Json::object();
        params["refine"] = cfg.refine;
        params["start"] = static_cast<double>(cfg.start);
        params["stop"] = static_cast<double>(cfg.stop);
        params["step"] = static_cast<double>(cfg.step);
        params["low"] = static_cast<double>(cfg.low);
        params["high"] = static_cast<double>(cfg.high);
        params["r"] = cfg.r;
        params["border_off"] = static_cast<double>(cfg.border_off);
        params["reader_scale"] = static_cast<double>(cfg.reader_scale);
        params["scale_group"] = cfg.dataset_group;
        report["parameters"] = std::move(params);
        report["summary"] = vc_objrefine::summarizeOffsets(
            offsetMap, opacityMap, surf->validMask(),
            cfg.offset_threshold, cfg.offset_min_opacity);

        try {
            writeOffsetReport(reportDir, offsetMap, opacityMap, report, surf->dpi());
        } catch (const std::exception& e) {
            if (!reportOnly) {
                std::cerr << "the surface was saved to " << dst << ", but ";
            }
            std::cerr << "failed to write the offset report: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "offset report: " << reportDir << std::endl;
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

    const utils::Json params = utils::Json::parse_file(params_path);
    const RefinementConfig cfg = parse_config(params);
    if (cfg.offset_report &&
        !(std::isfinite(cfg.offset_threshold) && cfg.offset_threshold >= 0.0f &&
          std::isfinite(cfg.offset_min_opacity) && cfg.offset_min_opacity >= 0.0f &&
          cfg.offset_min_opacity <= 1.0f)) {
        std::cerr << "offset_threshold must be >= 0 and offset_min_opacity in [0, 1]" << std::endl;
        return EXIT_FAILURE;
    }

    const int level = std::stoi(cfg.dataset_group);
    Volume volume(vol_path);
    if (!volume.hasScaleLevel(level)) {
        // Sparse pyramids keep absent levels as {0,0,0} placeholders; reading
        // through one silently yields fill value instead of failing.
        std::cerr << "Error: dataset group " << cfg.dataset_group
                  << " is not present in this volume; present levels:";
        for (int presentLevel : volume.presentScaleLevels())
            std::cerr << " " << presentLevel;
        std::cerr << std::endl;
        return EXIT_FAILURE;
    }
    vc::render::processChunkCacheService()->configureDecodedByteCapacity(
        cfg.cache_bytes);
    auto* chunk_cache = volume.chunkedCache();

    const auto shape = chunk_cache->shape(level);
    const auto chunk_shape = chunk_cache->chunkShape(level);
    std::cout << "zarr dataset size for scale group " << cfg.dataset_group
              << " [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
    std::cout << "chunk shape shape "
              << "[" << chunk_shape[0] << ", " << chunk_shape[1] << ", " << chunk_shape[2] << "]" << std::endl;
    std::cout << "chunk cache size (bytes) " << cfg.cache_bytes << std::endl;

    DSReader reader = {&volume, cfg.reader_scale, level};

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
