// vc_rgb2tifxyz.cpp
// Convert an RGB(A) PNG (R=X, G=Y, B=Z normalized to [0,1]) back to a tifxyz quadmesh.
//
// Usage:
//   vc_rgb2tifxyz <rgb.png> <bounds.json> <out_dir> [name] [--scale S] [--mask mask.png] [--invalid-black]
//
// If [name] is omitted, <out_dir> is treated as the final segment directory
// (meta.json will be written directly there). With [name], output goes to <out_dir>/<name>.

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <cmath>
#include <vector>
#include <filesystem>
#include <chrono>
#include <system_error>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <nlohmann/json.hpp>
#include "vc/core/util/Surface.hpp"  // QuadSurface

namespace fs = std::filesystem;
using json = nlohmann::json;

// ---------- utils ----------
static std::string time_str() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&tt);
    std::ostringstream oss;
    oss << std::put_time(&bt, "%Y%m%d%H%M%S") << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

// Avoid clashes with std::filesystem free functions
static bool path_is_regular_file(const fs::path& p) {
    std::error_code ec;
    return fs::exists(p, ec) && fs::is_regular_file(p, ec);
}
static bool rename_or_copy_move(const fs::path& src, const fs::path& dst, std::string* err=nullptr) {
    std::error_code ec;
    fs::rename(src, dst, ec);
    if (!ec) return true;
    try {
        fs::create_directories(dst.parent_path());
        fs::copy(src, dst, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        fs::remove_all(src);
        return true;
    } catch (const std::exception& ex) {
        if (err) *err = std::string("move failed: ") + ex.what();
        return false;
    }
}

// ---------- bounds & scale ----------
struct Bounds {
    float minx, miny, minz;
    float maxx, maxy, maxz;
};

// Extracts "scale" from json (number or array). Returns true if present.
// - scale_json_out: original json value (preserved verbatim for meta.json)
// - scale_for_save: a float to pass into QuadSurface (number or first array element)
static bool extract_scale_from_json(const json& j, json& scale_json_out, float& scale_for_save) {
    if (!j.contains("scale")) return false;
    const auto& s = j["scale"];
    if (s.is_number()) {
        scale_json_out = s;
        scale_for_save = s.get<float>();
        return true;
    }
    if (s.is_array() && !s.empty() && s[0].is_number()) {
        scale_json_out = s;                 // preserve full array in meta
        scale_for_save = s[0].get<float>(); // QuadSurface takes a single float
        return true;
    }
    return false;
}

// If user overrides with --scale S, rewrite the json scale value (number or array)
static void apply_scale_override(json& scale_json, float S) {
    if (scale_json.is_null()) { scale_json = S; return; }
    if (scale_json.is_number()) { scale_json = S; return; }
    if (scale_json.is_array()) {
        for (auto& v : scale_json) {
            if (v.is_number()) v = S;
        }
        return;
    }
    // unknown type -> just set number
    scale_json = S;
}

static bool load_bounds_and_scale(const fs::path& p, Bounds& b,
                                  json& scale_json_out, float& scale_for_save_out,
                                  std::string* why=nullptr)
{
    if (!path_is_regular_file(p)) {
        if (why) *why = "bounds path is not a regular file (did you pass a directory?)";
        return false;
    }
    std::ifstream f(p);
    if (!f) { if (why) *why = "cannot open file"; return false; }

    json j;
    try { f >> j; }
    catch (const std::exception& ex) { if (why) *why = std::string("json parse error: ") + ex.what(); return false; }

    auto arr_ok = [&](const char* key) {
        return j.contains(key) && j[key].is_array() && j[key].size() == 3;
    };

    // 1) bbox: [[minx,miny,minz],[maxx,maxy,maxz]]
    if (j.contains("bbox") && j["bbox"].is_array() && j["bbox"].size() == 2 &&
        j["bbox"][0].is_array() && j["bbox"][0].size() == 3 &&
        j["bbox"][1].is_array() && j["bbox"][1].size() == 3) {

        const auto& a = j["bbox"][0];
        const auto& c = j["bbox"][1];

        float x0 = a[0].get<float>(), y0 = a[1].get<float>(), z0 = a[2].get<float>();
        float x1 = c[0].get<float>(), y1 = c[1].get<float>(), z1 = c[2].get<float>();

        b.minx = std::min(x0, x1);  b.maxx = std::max(x0, x1);
        b.miny = std::min(y0, y1);  b.maxy = std::max(y0, y1);
        b.minz = std::min(z0, z1);  b.maxz = std::max(z0, z1);
    }
    // 2) { "min":[...], "max":[...] }
    else if (arr_ok("min") && arr_ok("max")) {
        b.minx = j["min"][0].get<float>();
        b.miny = j["min"][1].get<float>();
        b.minz = j["min"][2].get<float>();
        b.maxx = j["max"][0].get<float>();
        b.maxy = j["max"][1].get<float>();
        b.maxz = j["max"][2].get<float>();
        if (b.minx > b.maxx) std::swap(b.minx, b.maxx);
        if (b.miny > b.maxy) std::swap(b.miny, b.maxy);
        if (b.minz > b.maxz) std::swap(b.minz, b.maxz);
    }
    // 3) flat keys
    else if (j.contains("min_x") && j.contains("min_y") && j.contains("min_z") &&
             j.contains("max_x") && j.contains("max_y") && j.contains("max_z")) {
        b.minx = j["min_x"].get<float>();
        b.miny = j["min_y"].get<float>();
        b.minz = j["min_z"].get<float>();
        b.maxx = j["max_x"].get<float>();
        b.maxy = j["max_y"].get<float>();
        b.maxz = j["max_z"].get<float>();
        if (b.minx > b.maxx) std::swap(b.minx, b.maxx);
        if (b.miny > b.maxy) std::swap(b.miny, b.maxy);
        if (b.minz > b.maxz) std::swap(b.minz, b.maxz);
    } else {
        if (why) *why = "unrecognized bounds schema";
        return false;
    }

    // Extract scale if present
    scale_json_out = nullptr;
    scale_for_save_out = 1.0f;
    extract_scale_from_json(j, scale_json_out, scale_for_save_out);

    return true;
}

// ---------- metadata patch ----------
static bool patch_target_meta(const fs::path& target_dir,
                              const std::string& new_uuid,
                              const Bounds& B,
                              const json& scale_json,   // preserved (or overridden) json value
                              std::string source_png_basename,
                              std::string* err = nullptr)
{
    const fs::path meta_path = target_dir / "meta.json";
    json j;

    try {
        std::ifstream in(meta_path);
        if (!in) { if (err) *err = "cannot open " + meta_path.string(); return false; }
        in >> j;
    } catch (const std::exception& ex) {
        if (err) *err = std::string("parse error in ") + meta_path.string() + ": " + ex.what();
        return false;
    }

    j["uuid"] = new_uuid;
    j["bbox"] = json::array({
        json::array({B.minx, B.miny, B.minz}),
        json::array({B.maxx, B.maxy, B.maxz})
    });

    // Preserve exact shape/type of scale if provided
    if (!scale_json.is_null()) {
        j["scale"] = scale_json;
    }

    j["source_png"]   = source_png_basename;
    j["converted_by"] = "vc_rgb2tifxyz";

    try {
        std::ofstream out(meta_path);
        if (!out) { if (err) *err = "cannot write " + meta_path.string(); return false; }
        out << std::setw(2) << j << "\n";
    } catch (const std::exception& ex) {
        if (err) *err = std::string("write error for ") + meta_path.string() + ": " + ex.what();
        return false;
    }
    return true;
}

// Copy/patch other JSONs from src_dir -> dst_dir (skips meta.json)
static bool copy_and_patch_jsons(const fs::path& src_dir,
                                 const fs::path& dst_dir,
                                 const std::string& new_uuid,
                                 const Bounds& B,
                                 const json& scale_json, // preserved (or overridden)
                                 const std::string& source_png_basename,
                                 std::string* err = nullptr)
{
    std::error_code ec;
    if (!fs::exists(src_dir, ec) || !fs::is_directory(src_dir, ec)) {
        if (err) *err = "source json dir not found: " + src_dir.string();
        return false;
    }

    for (auto& e : fs::directory_iterator(src_dir)) {
        if (!e.is_regular_file()) continue;
        if (e.path().extension() != ".json") continue;
        if (e.path().filename() == "meta.json") continue; // don't clobber our meta.json

        json j;
        try {
            std::ifstream in(e.path());
            if (!in) { if (err) *err = "cannot open " + e.path().string(); return false; }
            in >> j;
        } catch (const std::exception& ex) {
            if (err) *err = std::string("parse error in ") + e.path().string() + ": " + ex.what();
            return false;
        }

        if (j.contains("uuid")) j["uuid"] = new_uuid;

        if (j.contains("bbox") && j["bbox"].is_array() && j["bbox"].size() == 2) {
            j["bbox"] = {
                {B.minx, B.miny, B.minz},
                {B.maxx, B.maxy, B.maxz}
            };
        }

        if (!scale_json.is_null() && j.contains("scale")) {
            j["scale"] = scale_json; // preserve exact structure
        }

        j["source_png"]   = source_png_basename;
        j["converted_by"] = "vc_rgb2tifxyz";

        fs::path out_json = dst_dir / e.path().filename();
        try {
            std::ofstream out(out_json);
            if (!out) { if (err) *err = "cannot write " + out_json.string(); return false; }
            out << std::setw(2) << j << "\n";
        } catch (const std::exception& ex) {
            if (err) *err = std::string("write error for ") + out_json.string() + ": " + ex.what();
            return false;
        }
    }
    return true;
}

// ---------- main ----------
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr <<
          "Usage:\n  " << argv[0]
          << " <rgb.png> <bounds.json> <out_dir> [name] [--scale S] [--mask mask.png] [--invalid-black]\n";
        return 1;
    }

    const fs::path rgb_path    = argv[1];
    const fs::path bounds_path = argv[2];
    fs::path out_dir           = argv[3];
    std::string name_arg       = (argc >= 5 && argv[4][0] != '-') ? std::string(argv[4]) : std::string();

    // --- Options
    bool   scale_overridden = false;
    float  scale_override   = 1.0f;
    fs::path mask_path;
    bool invalid_black = false;

    int opt_start = 4;
    if (!name_arg.empty()) opt_start = 5;

    for (int i = opt_start; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--scale" && i+1 < argc) {
            try { scale_override = std::stof(argv[++i]); scale_overridden = true; }
            catch (...) { std::cerr << "Error: invalid --scale value.\n"; return 1; }
        } else if (s == "--mask" && i+1 < argc) {
            mask_path = fs::path(argv[++i]);
        } else if (s == "--invalid-black") {
            invalid_black = true;
        } else if (!s.empty() && s[0] == '-') {
            std::cerr << "Warning: unknown option: " << s << "\n";
        }
    }

    // --- Bounds & (original) scale
    Bounds B;
    json   scale_json = nullptr;
    float  scale_for_save = 1.0f;

    {
        std::string why;
        if (!load_bounds_and_scale(bounds_path, B, scale_json, scale_for_save, &why)) {
            std::cerr << "Error: failed to parse bounds from " << bounds_path << " - " << why << "\n";
            return 2;
        }
    }

    // If user overrode scale, replace json and scale_for_save
    if (scale_overridden) {
        apply_scale_override(scale_json, scale_override);
        scale_for_save = scale_override;
    }

    const float eps = 1e-12f;
    const float rx = std::max(B.maxx - B.minx, eps);
    const float ry = std::max(B.maxy - B.miny, eps);
    const float rz = std::max(B.maxz - B.minz, eps);

    // --- Load RGB(A)
    cv::Mat img = cv::imread(rgb_path.string(), cv::IMREAD_UNCHANGED);
    if (img.empty()) { std::cerr << "Error: could not read image " << rgb_path << "\n"; return 3; }
    if (!(img.type() == CV_8UC3 || img.type() == CV_8UC4)) {
        std::cerr << "Error: expected 8-bit 3- or 4-channel image. Got type=" << img.type() << "\n";
        return 4;
    }
    const bool has_alpha = (img.type() == CV_8UC4);

    // --- Optional mask
    cv::Mat mask;
    if (!mask_path.empty()) {
        cv::Mat m = cv::imread(mask_path.string(), cv::IMREAD_GRAYSCALE);
        if (m.empty() || m.size() != img.size()) {
            std::cerr << "Error: mask missing or size mismatch.\n";
            return 5;
        }
        mask = m;
    }

    const int rows = img.rows, cols = img.cols;

    // --- Reconstruct points (world/voxel coords according to bounds)
    cv::Mat_<cv::Vec3f> points(rows, cols, cv::Vec3f(-1.f, -1.f, -1.f));
    for (int j = 0; j < rows; ++j) {
        const auto* p4   = has_alpha ? img.ptr<cv::Vec4b>(j) : nullptr;
        const auto* p3   = has_alpha ? nullptr : img.ptr<cv::Vec3b>(j);
        const uint8_t* mrow = mask.empty() ? nullptr : mask.ptr<uint8_t>(j);

        for (int i = 0; i < cols; ++i) {
            bool invalid = false;

            if (has_alpha && p4[i][3] == 0) invalid = true;
            if (!invalid && mrow && mrow[i] == 0) invalid = true;
            if (!invalid && invalid_black && !has_alpha && !mrow) {
                if (p3[i][0] == 0 && p3[i][1] == 0 && p3[i][2] == 0) invalid = true;
            }

            if (invalid) {
                points(j,i) = cv::Vec3f(-1.f, -1.f, -1.f);
                continue;
            }

            float nx, ny, nz;
            if (has_alpha) { nz = p4[i][0] / 255.0f; ny = p4[i][1] / 255.0f; nx = p4[i][2] / 255.0f; }
            else           { nz = p3[i][0] / 255.0f; ny = p3[i][1] / 255.0f; nx = p3[i][2] / 255.0f; }

            const float X = B.minx + nx * rx;
            const float Y = B.miny + ny * ry;
            const float Z = B.minz + nz * rz;
            points(j,i) = cv::Vec3f(X, Y, Z);
        }
    }

    // --- Decide final output directory
    fs::path final_dir = name_arg.empty() ? out_dir : (out_dir / name_arg);
    const std::string final_name = final_dir.filename().string();

    // Ensure parent dir exists
    std::error_code ec;
    fs::create_directories(final_dir.parent_path(), ec);

    // Stage to a temp dir to normalize QuadSurface::save behavior
    fs::path staging_base = final_dir.parent_path() / (".tmp_rgb2tifxyz_" + time_str());

    // Save via QuadSurface using the *preserved* scale (or override)
    std::unique_ptr<QuadSurface> surf(new QuadSurface(points, scale_for_save));
    std::cout << "Saving (staging) dir=\"" << staging_base.string() << "\" name=" << final_name
              << "\" (scale=" << scale_for_save << ")\n";
    surf->save(staging_base, final_name);

    // Discover where meta.json landed
    fs::path produced_dir;
    if (path_is_regular_file(staging_base / "meta.json")) {
        produced_dir = staging_base;
    } else if (path_is_regular_file(staging_base / final_name / "meta.json")) {
        produced_dir = staging_base / final_name;
    } else {
        std::cerr << "Error: could not find meta.json in staging output.\n";
        fs::remove_all(staging_base, ec);
        return 6;
    }

    // If final_dir already has a meta.json, write to a unique suffixed dir
    fs::path chosen_final = final_dir;
    if (path_is_regular_file(final_dir / "meta.json")) {
        chosen_final = final_dir.parent_path() / (final_dir.filename().string() + "_" + time_str());
        std::cerr << "Warning: final dir already contains a meta.json; writing to: " << chosen_final << "\n";
    }

    // Move produced_dir -> chosen_final
    std::string move_err;
    if (!rename_or_copy_move(produced_dir, chosen_final, &move_err)) {
        std::cerr << "Error: failed to move output to final dir: " << move_err << "\n";
        fs::remove_all(staging_base, ec);
        return 7;
    }
    fs::remove_all(staging_base, ec);

    // Patch freshly written meta.json (uuid, bbox, exact scale json preserved)
    std::string meta_err;
    if (!patch_target_meta(chosen_final, chosen_final.filename().string(), B,
                           scale_json, rgb_path.filename().string(), &meta_err)) {
        std::cerr << "Warning: failed to patch meta.json: " << meta_err << "\n";
    }

    // Copy+patch other JSONs from the bounds file folder (keep exact scale json)
    const fs::path src_json_dir = bounds_path.parent_path();
    std::string err;
    if (!copy_and_patch_jsons(src_json_dir, chosen_final,
                              /*new_uuid*/ chosen_final.filename().string(),
                              /*bounds*/ B,
                              /*scale_json*/ scale_json,
                              /*source_png*/ rgb_path.filename().string(),
                              &err))
    {
        std::cerr << "Warning: failed to copy/patch JSONs: " << err << "\n";
    } else {
        std::cout << "Copied & patched JSONs from \"" << src_json_dir.string() << "\"\n";
    }

    // Final info
    // If scale_json is array, print first component; if number, print it.
    float scale_print = scale_for_save;
    std::cout << "Done.\n"
              << "  segment dir: " << chosen_final << "\n"
              << "  size: " << cols << " x " << rows << "\n"
              << "  scale: " << scale_print << "\n";
    return 0;
}
