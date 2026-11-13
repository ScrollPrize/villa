// helirec-reconstruct: slices of an extended-FOV helical scan pair from the raw projection folders
// (helirec::Reconstruction, helirec/reconstruction.hpp).
//
// Two modes:
//
// * fractions (independent single slices, TIFF; PNG previews opt in with --previews):
//     helirec-reconstruct PROJ -n 8 --tag baseline
//     helirec-reconstruct PROJ --fractions 0.4 0.5 --set "filters=paganin(delta_beta=500)+unsharp(sigma=1.2,coeff=4)" --tag db500
//
// * range (contiguous slices, reconstructed in chunks of --chunk slices; OME-Zarr by default):
//     helirec-reconstruct PROJ --z Z0 Z1 --roi X0 X1 Y0 Y1 --uint8 --tag block
//     helirec-reconstruct PROJ --heights H0 H1 --format tiff --tag block_tif
//
//   --z uses volume slice indices (z = view height - first doable height, z = 0 at the top); --heights uses helirec
//   view heights; both are [start, end). --roi X0 X1 Y0 Y1 crops each slice ([start, end)). Output: volume.zarr
//   (float32 OME-Zarr, raw chunks), volume_uint8.zarr with --uint8 (window --u8_window, lossless vc-delta3d chunks,
//   0 outside the reconstruction disk); empty chunks are not written; or float32/ (and uint8/) TIFF stacks with
//   --format tiff; manifest.json. --uint8_only writes the uint8 version alone.
//
// Units: attenuation in cm^-1 by default; --set mode=legacy_reference gives the reference workflow's units (half of
// it) and conventions. The default uint8 window is the reference one in the run's units.
// Calibration is computed once and cached (--cache_dir): the geometry calibration (geometry.json; --set
// geometry=PATH names one made before) and the intensity calibration (calibration_arrays.bin, calibration.json).
// --config FILE: a JSON object of options (keys = option names without the dashes); the command line wins. Each run
// saves its config.json, which reruns it.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>

#include "helirec/kernels.hpp"
#include "helirec/progress.hpp"
#include "helirec/reconstruction.hpp"
#include "helirec/reference.hpp"
#include "helirec/zarr.hpp"
#include "vc_reader/image_write.hpp"
#include "io/command_line.hpp"
#include "io/json.hpp"

using namespace helirec;
namespace fs = std::filesystem;

namespace {

using namespace helirec::command_line;

// slices and previews as files (vc_reader's writers): TIFF uncompressed, as tifffile.imwrite writes them
void write_tiff(const std::string& path, const float* data, int ny, int nx) { vcr::sources::writeTiff(path, data, "<f4", ny, nx); }
void write_tiff(const std::string& path, const std::uint8_t* data, int ny, int nx) { vcr::sources::writeTiff(path, data, "|u1", ny, nx); }
void write_png(const std::string& path, const std::uint8_t* data, int ny, int nx) { vcr::sources::writeImage(path, vcr::sources::ImageView{data, "|u1", ny, nx}); }

const Spec& option_spec() {
    static const Spec spec{{
                               {"config", 1, 's'},         {"n_slices", 1, 'i'},       {"range", 2, 'f'},           {"object_extent", 2, 'f'},
                               {"fractions", -1, 'f'},     {"z", 2, 'i'},              {"heights", 2, 'i'},         {"roi", 4, 'i'},
                               {"format", 1, 's'},         {"uint8", 0, 'b'},          {"uint8_only", 0, 'b'},      {"u8_window", 2, 'f'},
                               {"bbox", 4, 'i'},           {"reference_volume", 1, 's'}, {"reference_scale", 1, 'i'}, {"reference_cache", 1, 's'},
                               {"chunk", 1, 'i'},          {"zarr_chunks", 3, 'i'},    {"levels", 1, 'i'},          {"compression", 1, 's'},
                               {"u8_compression", 1, 's'}, {"u8_delta3d_preset", 1, 's'}, {"tag", 1, 's'},          {"out_root", 1, 's'},
                               {"previews", 0, 'b'},       {"no_preview", 0, 'b'},     {"overwrite", 0, 'b'},       {"set", -2, 's'},
                               {"device", 1, 's'},         {"cache_dir", 1, 's'},      {"projections", 1, 's'},
                           },
                           {{"-n", "n_slices"}},
                           "projections",
                           false};
    return spec;
}

int usage() {
    std::fprintf(stderr,
                 "usage: helirec-reconstruct [PROJECTIONS] [--config FILE] [-n N] [--range R0 R1] [--object_extent A B] [--fractions F ...]\n"
                 "                           [--z Z0 Z1 | --heights H0 H1] [--roi X0 X1 Y0 Y1] [--bbox X0 X1 Y0 Y1] [--format zarr|tiff]\n"
                 "                           [--uint8] [--uint8_only] [--u8_window LO HI] [--reference_volume ZARR] [--reference_scale LEVEL]\n"
                 "                           [--reference_cache DIR] [--chunk N] [--zarr_chunks CZ CY CX] [--levels N] [--compression none|zlib]\n"
                 "                           [--u8_compression vc-delta3d|none|zlib] [--u8_delta3d_preset fast|medium|slow|slower|delta]\n"
                 "                           [--tag TAG] [--out_root DIR] [--previews] [--overwrite] [--set KEY=VALUE ...] [--device cuda|cpu]\n"
                 "                           [--cache_dir DIR]\n"
                 "(see the head of tools/helirec_reconstruct.cpp and README.md)\n");
    return 2;
}

std::string now_iso() {
    const std::time_t t = std::time(nullptr);
    char b[32];
    std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return b;
}

std::string now_tag() {
    const std::time_t t = std::time(nullptr);
    char b[32];
    std::strftime(b, sizeof b, "%Y%m%d_%H%M%S", std::localtime(&t));
    return b;
}

double since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

std::string fmt(const char* f, double v) {
    char b[64];
    std::snprintf(b, sizeof b, f, v);
    return b;
}

void say(const std::string& line) {
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
}

std::vector<std::uint8_t> to_u8(const float* img, int ny, int nx, const std::array<double, 2>& window) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(ny) * nx);
    Uint8Window w;
    w.lo = window[0], w.hi = window[1];
    cpu::to_uint8(img, static_cast<std::size_t>(nx), out.data(), ny, nx, w);
    return out;
}

// the image's mean over blocks of f x f pixels, f = max(1, rows // size)
std::vector<float> downsample(const float* img, int ny, int nx, int size, int& oy, int& ox) {
    const int f = std::max(1, ny / size);
    oy = ny / f, ox = nx / f;
    std::vector<float> out(static_cast<std::size_t>(oy) * ox);
    for (int y = 0; y < oy; ++y)
        for (int x = 0; x < ox; ++x) {
            double s = 0;
            for (int a = 0; a < f; ++a)
                for (int b = 0; b < f; ++b) s += img[static_cast<std::size_t>(y * f + a) * nx + x * f + b];
            out[static_cast<std::size_t>(y) * ox + x] = static_cast<float>(s / (f * f));
        }
    return out;
}

struct Preview {
    std::vector<std::uint8_t> pixels;
    int ny = 0, nx = 0;
};

void contact_sheet(const std::string& path, const std::vector<Preview>& previews) {
    const int n = static_cast<int>(previews.size());
    const int cols = std::min(5, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n)))));
    const int rows = (n + cols - 1) / cols;
    int s = 0, w = 0;
    for (const auto& p : previews) s = std::max(s, p.ny), w = std::max(w, p.nx);
    std::vector<std::uint8_t> sheet(static_cast<std::size_t>(rows) * s * cols * w, 0);
    for (int k = 0; k < n; ++k) {
        const Preview& p = previews[static_cast<std::size_t>(k)];
        for (int y = 0; y < p.ny; ++y)
            std::copy(p.pixels.begin() + static_cast<std::ptrdiff_t>(y) * p.nx, p.pixels.begin() + static_cast<std::ptrdiff_t>(y + 1) * p.nx,
                      sheet.begin() + static_cast<std::ptrdiff_t>((k / cols) * s + y) * cols * w + (k % cols) * w);
    }
    write_png(path, sheet.data(), rows * s, cols * w);
}

Json reference_description(const ReferenceMask& ref) {
    const auto f = ref.factors();
    const auto l = ref.level_shape();
    return Json::dict()
        .set("location", Json::of(ref.location()))
        .set("scale", Json::whole_number(ref.scale()))
        .set("factors", Json::list().add(Json::whole_number(f[0])).add(Json::whole_number(f[1])).add(Json::whole_number(f[2])))
        .set("level_shape", Json::list().add(Json::whole_number(l[0])).add(Json::whole_number(l[1])).add(Json::whole_number(l[2])))
        .set("slice_ratio", Json::of(ref.slice_ratio()));
}

Json ints_json(const std::vector<int>& v) {
    Json j = Json::list();
    for (int x : v) j.add(Json::whole_number(x));
    return j;
}

struct Run {
    Args a;
    Reconstruction* r = nullptr;
    std::string out;
    Json manifest;
    std::array<double, 2> window{};
    std::unique_ptr<ReferenceMask> reference;
    std::optional<SliceRoi> roi, bbox;
    int N = 0, lo = 0, hi = 0;

    void save_manifest() const { write_text(out + "/manifest.json", manifest.text()); }
    Json& slices() {
        for (auto& kv : manifest.object)
            if (kv.first == "slices") return kv.second;
        fail("manifest without slices");
    }
    void put(const std::string& k, Json v) {
        for (auto& kv : manifest.object)
            if (kv.first == k) return void(kv.second = std::move(v));
        manifest.object.emplace_back(k, std::move(v));
    }

    void fractions_mode() {
        std::vector<double> fractions = a.numbers("fractions");
        if (fractions.empty()) {
            const auto ext = a.numbers("object_extent"), range = a.numbers("range");
            const double A = ext[0], B = ext[1], f0 = A + (B - A) * range[0], f1 = A + (B - A) * range[1];
            const int n = static_cast<int>(a.number("n_slices"));
            if (n == 1) fractions = {0.5 * (f0 + f1)};
            else
                for (int i = 0; i < n; ++i) fractions.push_back(f0 + (f1 - f0) * i / (n - 1));
        }
        const bool previews = a.flag("previews");
        const bool uint8 = a.flag("uint8"), uint8_only = a.flag("uint8_only");
        if (previews) fs::create_directories(out + "/preview");
        std::vector<Preview> sheet;
        auto rec = r->reconstructor();
        const SliceRoi area = roi ? *roi : SliceRoi{0, N, 0, N};
        const int ny = area.y1 - area.y0, nx = area.x1 - area.x0;
        ProgressBar bar(fractions.size(), "slices", "slice");
        std::vector<float> img(static_cast<std::size_t>(ny) * nx);
        std::vector<std::uint8_t> u8(img.size());
        for (double f : fractions) {
            const SliceInfo si = r->slice_info(f);
            const auto t0 = std::chrono::steady_clock::now();
            const int z = si.view_height - lo;
            SliceOutput o;
            o.f32 = img.data();
            o.u8 = u8.data();
            o.u8_lo = window[0], o.u8_hi = window[1];
            o.u8_zero_outside = false;  // (single slices keep their values outside the reconstruction disk)
            if (reference) o.u8_mask = reference->read(z, z + 1, area);
            o.region = bbox;
            rec->reconstruct(si.view_height, si.view_height + 1, o, nullptr, &area, 0, 0, &bar);
            const double dt = since(t0);
            bar.set_postfix("write");
            char name[64];
            std::snprintf(name, sizeof name, "slice_f%.4f", f);
            Json entry = Json::dict()
                             .set("name", Json::of(std::string(name)))
                             .set("fraction", Json::of(f))
                             .set("view_height", Json::whole_number(si.view_height))
                             .set("z", Json::whole_number(z))
                             .set("nabu_index", Json::whole_number(si.nabu_index))
                             .set("height_mm", Json::of(si.height_mm))
                             .set("seconds", Json::of(std::round(dt * 100) / 100));
            std::string path;
            if (!uint8_only) {
                path = out + "/" + name + ".tiff";
                entry.set("tiff", Json::of(path));
                write_tiff(path, img.data(), ny, nx);
            }
            if (uint8) {
                const std::string p8 = out + "/" + name + "_uint8.tif";
                entry.set("tiff_uint8", Json::of(p8));
                write_tiff(p8, u8.data(), ny, nx);
                if (path.empty()) path = p8;
            }
            if (previews) {
                const std::string pp = out + "/preview/" + name + ".png";
                entry.set("preview", Json::of(pp));
                write_png(pp, u8.data(), ny, nx);
                Preview p;
                const auto small = downsample(img.data(), ny, nx, 512, p.ny, p.nx);
                p.pixels = to_u8(small.data(), p.ny, p.nx, window);
                sheet.push_back(std::move(p));
            }
            slices().add(entry);
            bar.write("f=" + fmt("%.4f", f) + " h=" + std::to_string(si.view_height) + " z=" + std::to_string(z) + " (" + fmt("%.2f", si.height_mm) + " mm) " +
                      fmt("%.1f", dt) + "s -> " + path);
            save_manifest();
        }
        bar.close();
        if (!sheet.empty()) contact_sheet(out + "/contact_sheet.png", sheet);
    }

    void range_mode() {
        int h0, h1;
        if (a.get("z")) {
            const auto z = a.ints("z");
            h0 = z[0] + lo, h1 = z[1] + lo;
        } else {
            const auto h = a.ints("heights");
            h0 = h[0], h1 = h[1];
        }
        if (!(lo <= h0 && h0 < h1 && h1 <= hi))
            fail("range outside the doable view heights [" + std::to_string(lo) + ", " + std::to_string(hi) + ") (z [0, " + std::to_string(hi - lo) + "))");
        const SliceRoi area = roi ? *roi : SliceRoi{0, N, 0, N};
        const int n = h1 - h0, ny = area.y1 - area.y0, nx = area.x1 - area.x0, z0 = h0 - lo;
        const int chunk = static_cast<int>(a.number("chunk"));
        const std::string format = a.text("format");
        const bool uint8 = a.flag("uint8"), uint8_only = a.flag("uint8_only");
        put("view_heights", ints_json({h0, h1}));
        put("z", ints_json({z0, z0 + n}));
        put("roi_xyxy", ints_json({area.x0, area.x1, area.y0, area.y1}));
        put("shape_zyx", ints_json({n, ny, nx}));
        put("chunk", Json::whole_number(chunk));
        put("format", Json::of(format));
        if (reference) put("reference_volume", reference_description(*reference));
        if (bbox) put("bbox_xyxy", ints_json({bbox->x0, bbox->x1, bbox->y0, bbox->y1}));
        Json helirec_attrs = Json::dict();
        for (const char* k : {"created", "projections", "mode", "params", "provenance", "view_heights", "z", "roi_xyxy", "chunk"}) helirec_attrs.set(k, *manifest.find(k));
        const Json provenance = Json::dict().set("helirec", helirec_attrs).set("helirec_calibration", *manifest.find("calibration"));
        Json u8_attrs = provenance;
        u8_attrs.set("helirec_uint8_window", Json::list().add(Json::of(window[0])).add(Json::of(window[1])));
        if (reference) u8_attrs.set("helirec_reference_volume", reference_description(*reference));
        std::vector<std::pair<std::string, std::unique_ptr<OmeZarrWriter>>> writers;
        Json outputs = Json::dict();
        const double vox = r->main().pixel_size_m * 1e6;
        const std::string tag = a.text("tag");
        if (format == "zarr") {
            ZarrIndex chunks{128, 128, 128};  // 32-aligned for vc-delta3d presets
            if (a.get("zarr_chunks")) {
                const auto c = a.ints("zarr_chunks");
                chunks = {c[0], c[1], c[2]};
            }
            const auto options = [&](const char* dtype, const std::string& compression, const std::string& name, const Json& attrs) {
                OmeZarrOptions o;
                o.shape = {n, ny, nx};
                o.dtype = dtype;
                o.chunks = chunks;
                o.voxel_size_um = vox;
                o.origin = {z0, area.y0, area.x0};
                o.levels = static_cast<int>(a.number("levels"));
                o.compression = compression;
                o.name = name;
                o.attrs_json = attrs.text();
                o.overwrite = a.flag("overwrite");
                return o;
            };
            if (!uint8_only)
                writers.emplace_back("float32", std::make_unique<OmeZarrWriter>(out + "/volume.zarr", options("<f4", a.text("compression"), tag + " float32", provenance)));
            if (uint8) {
                OmeZarrOptions o = options("|u1", a.text("u8_compression"), tag + " uint8", u8_attrs);
                o.delta3d_preset = a.text("u8_delta3d_preset");
                writers.emplace_back("uint8", std::make_unique<OmeZarrWriter>(out + "/volume_uint8.zarr", o));
            }
            for (const auto& [k, w] : writers) outputs.set(k, Json::of(w->path()));
        } else {
            if (!uint8_only) outputs.set("float32", Json::of(out + "/float32"));
            if (uint8) outputs.set("uint8", Json::of(out + "/uint8"));
            for (const auto& kv : outputs.object) fs::create_directories(kv.second.string);
        }
        put("outputs", outputs);
        save_manifest();
        const auto t_all = std::chrono::steady_clock::now();
        auto rec = r->reconstructor();
        std::vector<std::pair<int, int>> chunks;
        for (int c0 = h0; c0 < h1; c0 += chunk) chunks.emplace_back(c0, std::min(h1, c0 + chunk));
        // the reference volume's mask of a chunk is read while the chunk before it reconstructs (two buffers in turn)
        const auto read_mask = [&](std::size_t i) {
            return std::async(std::launch::async, [&, i] { return reference->read(chunks[i].first - lo, chunks[i].second - lo, area, static_cast<int>(i % 2)); });
        };
        std::future<OutputMask> pending;
        if (reference) pending = read_mask(0);
        OmeZarrWriter* f32_to = nullptr;
        OmeZarrWriter* u8_to = nullptr;
        for (auto& [k, w] : writers) (k == "float32" ? f32_to : u8_to) = w.get();
        std::vector<float> vol;
        std::vector<std::uint8_t> vol8;
        Preview mid;
        const int h_mid = (h0 + h1) / 2;
        {
            ProgressBar bar(static_cast<std::uint64_t>(n), "reconstruct", "slice");
            for (std::size_t i = 0; i < chunks.size(); ++i) {
                const auto [c0, c1] = chunks[i];
                SliceOutput o;
                if (reference) {
                    o.u8_mask = pending.get();
                    if (i + 1 < chunks.size()) pending = read_mask(i + 1);
                }
                const std::size_t px = static_cast<std::size_t>(c1 - c0) * ny * nx;
                if (!uint8_only) {
                    if (f32_to) o.f32_sink = f32_to;
                    else vol.resize(px), o.f32 = vol.data();
                }
                if (uint8) {
                    if (u8_to) o.u8_sink = u8_to;
                    else vol8.resize(px), o.u8 = vol8.data();
                    o.u8_lo = window[0], o.u8_hi = window[1];
                    const Json* clip = Json::parse(r->parameters()).find("clip_outer_circle");
                    o.u8_zero_outside = clip && clip->is(Json::Bool) && clip->boolean;
                }
                o.sink_z = c0 - h0;
                o.region = bbox;
                const int next_lo = i + 1 < chunks.size() ? chunks[i + 1].first : 0, next_hi = i + 1 < chunks.size() ? chunks[i + 1].second : 0;
                rec->reconstruct(c0, c1, o, nullptr, &area, next_lo, next_hi, &bar);
                bar.set_postfix("write");
                if (format != "zarr") {
                    for (int k = 0; k < c1 - c0; ++k) {
                        char name[32];
                        const int zz = c0 + k - lo;
                        if (!uint8_only) {
                            std::snprintf(name, sizeof name, "/float32/z%05d.tiff", zz);
                            write_tiff(out + name, vol.data() + static_cast<std::size_t>(k) * ny * nx, ny, nx);
                        }
                        if (uint8) {
                            std::snprintf(name, sizeof name, "/uint8/z%05d.tif", zz);
                            write_tiff(out + name, vol8.data() + static_cast<std::size_t>(k) * ny * nx, ny, nx);
                        }
                    }
                    if (c0 <= h_mid && h_mid < c1 && a.flag("previews")) {
                        const std::size_t at = static_cast<std::size_t>(h_mid - c0) * ny * nx;
                        mid.ny = ny, mid.nx = nx;
                        mid.pixels = uint8_only ? std::vector<std::uint8_t>(vol8.begin() + static_cast<std::ptrdiff_t>(at), vol8.begin() + static_cast<std::ptrdiff_t>(at) + ny * nx)
                                                : to_u8(vol.data() + at, ny, nx, window);
                    }
                }
            }
        }
        Json written = Json::dict();
        for (auto& [kind, w] : writers) {
            std::unique_ptr<ProgressBar> level_bar;  // one bar per pyramid level
            const std::string k = kind;
            w->close([&](int l, std::int64_t slabs) -> std::function<void()> {
                level_bar = std::make_unique<ProgressBar>(static_cast<std::uint64_t>(slabs), k + " level " + std::to_string(l), "slab");
                ProgressBar* b = level_bar.get();
                return [b] { b->update(); };
            });
            level_bar.reset();
            written.set(kind, Json::dict().set("written", Json::whole_number(w->n_written())).set("skipped_empty", Json::whole_number(w->n_skipped())));
        }
        if (!writers.empty()) put("chunks_written", written);
        if (!writers.empty() && a.flag("previews")) {  // the middle slice, as stored
            const int k = h_mid - h0;
            mid.ny = ny, mid.nx = nx;
            OmeZarrWriter* w = u8_to ? u8_to : f32_to;
            if (u8_to) {
                mid.pixels.resize(static_cast<std::size_t>(ny) * nx);
                w->read(0, k, k + 1, mid.pixels.data());
            } else {
                std::vector<float> s(static_cast<std::size_t>(ny) * nx);
                w->read(0, k, k + 1, s.data());
                mid.pixels = to_u8(s.data(), ny, nx, window);
            }
        }
        if (!mid.pixels.empty()) write_png(out + "/preview_mid.png", mid.pixels.data(), mid.ny, mid.nx);
        put("seconds", Json::of(std::round(since(t_all) * 10) / 10));
        save_manifest();
    }
};

int run(int argc, char** argv) {
    Json given = parse(argc, argv, option_spec());
    if (const Json* np = given.find("no_preview"))  // (--no_preview: the default)
        if (np->boolean) put(given, "previews", Json::of_bool(false));
    // the defaults, then the config file, then the command line
    Json all = in_effect(Json::dict()
                             .set("n_slices", Json::whole_number(5))
                             .set("range", Json::list().add(Json::of(1.0 / 6)).add(Json::of(5.0 / 6)))
                             .set("object_extent", Json::list().add(Json::of(0.0)).add(Json::of(1.0)))
                             .set("reference_scale", Json::whole_number(0))
                             .set("chunk", Json::whole_number(32))
                             .set("levels", Json::whole_number(1))
                             .set("compression", Json::of("none"))
                             .set("u8_compression", Json::of("vc-delta3d"))
                             .set("u8_delta3d_preset", Json::of("fast"))
                             .set("out_root", Json::of(fs::absolute("helirec_slices").string()))
                             .set("set", Json::list())
                             .set("device", Json::of("cuda")),
                         given, option_spec(), [](const std::string& k, const Json& v, Json& clean) {
                             if (k != "no_preview") return false;
                             put(clean, "previews", Json::of_bool(!(v.is(Json::Bool) && v.boolean)));
                             return true;
                         });
    const auto merge = [&](const Json& from) {
        for (const auto& [k, v] : from.object) put(all, k, v);
    };
    Args a{all};
    if (a.text("projections").empty()) fail("projections folder missing (positional argument or 'projections' in --config)");

    const bool range_mode = a.get("z") || a.get("heights");
    if (a.get("z") && a.get("heights")) fail("use either --z or --heights");
    if (range_mode && a.get("fractions")) fail("--fractions cannot be combined with --z/--heights");
    if (!a.get("format")) merge(Json::dict().set("format", Json::of(range_mode ? "zarr" : "tiff")));
    a.all = all;
    const std::string format = a.text("format");
    if (format != "zarr" && format != "tiff") fail("--format is zarr or tiff");
    if (!range_mode && format == "zarr") fail("OME-Zarr output needs a contiguous range (--z or --heights)");
    if (a.number("chunk") < 1) fail("--chunk must be >= 1");
    if (a.flag("uint8_only")) merge(Json::dict().set("uint8", Json::of_bool(true)));
    a.all = all;
    if (a.get("reference_volume") && !a.flag("uint8")) fail("--reference_volume masks the uint8 output: add --uint8 or --uint8_only");
    const std::string device = a.text("device");
    if (device != "cuda" && device != "cpu") fail("--device is cuda or cpu");

    if (!a.get("tag")) merge(Json::dict().set("tag", Json::of(now_tag())));
    merge(Json::dict()
              .set("projections", Json::of(fs::absolute(a.text("projections")).lexically_normal().string()))
              .set("out_root", Json::of(fs::absolute(a.text("out_root")).lexically_normal().string())));
    a.all = all;
    const std::string outdir = a.text("out_root") + "/" + a.text("tag");
    if (fs::exists(outdir) && !a.flag("overwrite")) fail(outdir + " exists (use another --tag or --overwrite)");
    fs::create_directories(outdir);
    write_text(outdir + "/config.json", all.text());  // rerun with: helirec-reconstruct --config config.json
    const std::string cache = a.get("cache_dir") ? a.text("cache_dir") : a.text("out_root") + "/_calibration";
    std::vector<std::string> sets;
    for (const Json& s : a.get("set")->array) sets.push_back(s.string);

    const auto t0 = std::chrono::steady_clock::now();
    Reconstruction::Options opt;
    opt.cache_dir = cache;
    opt.device = device == "cuda" ? Device::CUDA : Device::CPU;
    opt.log = say;
    opt.progress = true;
    Reconstruction r(a.text("projections"), parse_settings(sets), opt);
    Run run;
    run.a = a;
    run.r = &r;
    run.out = outdir;
    std::tie(run.lo, run.hi) = r.doable();
    run.N = r.reconstructor()->grid_size();
    run.window = a.get("u8_window") ? std::array<double, 2>{a.numbers("u8_window")[0], a.numbers("u8_window")[1]} : display_window(r.mode());
    const Json info = Json::parse(r.info());
    say("setup " + fmt("%.1f", since(t0)) + "s: doable view heights [" + std::to_string(run.lo) + ", " + std::to_string(run.hi) + ") (z 0.." +
        std::to_string(run.hi - run.lo - 1) + "), rotation axis " + fmt("%.3f", r.rot_center()) + ", grid " + std::to_string(run.N) + ", margin " +
        std::to_string(static_cast<int>(info.find("margin")->number)));
    if (a.get("reference_volume")) {
        try {
            run.reference = std::make_unique<ReferenceMask>(a.text("reference_volume"), static_cast<int>(a.number("reference_scale")),
                                                            std::array<std::int64_t, 3>{run.hi - run.lo, run.N, run.N}, a.text("reference_cache"));
        } catch (const std::exception& e) {
            fail(std::string("--reference_volume: ") + e.what());
        }
        const auto f = run.reference->factors();
        const auto l = run.reference->level_shape();
        say("reference volume " + a.text("reference_volume") + ": uint8 output masked where its level " + std::to_string(run.reference->scale()) + " (" +
            std::to_string(l[0]) + "x" + std::to_string(l[1]) + "x" + std::to_string(l[2]) + ", " + std::to_string(f[0]) + "x" + std::to_string(f[1]) + "x" +
            std::to_string(f[2]) + " voxels each) is 0");
        if (run.reference->slice_ratio() != 1)
            say("  the reference has " + fmt("%.5f", run.reference->slice_ratio()) + " slices per slice of this reconstruction (another helical pitch calibration): its "
                "slices are spread evenly over the " + std::to_string(run.hi - run.lo) + " of the reconstruction; the mask is placed to within a few slices");
    }
    const auto box = [&](const char* k, std::optional<SliceRoi>& dst) {
        if (!a.get(k)) return;
        const auto v = a.ints(k);
        if (!(0 <= v[0] && v[0] < v[1] && v[1] <= run.N && 0 <= v[2] && v[2] < v[3] && v[3] <= run.N))
            fail(std::string("--") + k + " must lie within [0, " + std::to_string(run.N) + "]");
        dst = SliceRoi{v[0], v[1], v[2], v[3]};
    };
    box("roi", run.roi);
    box("bbox", run.bbox);
    run.manifest = Json::dict()
                       .set("created", Json::of(now_iso()))
                       .set("projections", Json::of(a.text("projections")))
                       .set("device", Json::of(device))
                       .set("mode", Json::of(r.mode()))
                       .set("params", Json::parse(r.parameters()))
                       .set("provenance", Json::parse(r.provenance()))
                       .set("calibration", Json::parse(r.intensity().json()))
                       .set("uint8_window", Json::list().add(Json::of(run.window[0])).add(Json::of(run.window[1])))
                       .set("slices", Json::list());
    if (range_mode) run.range_mode();
    else run.fractions_mode();
    say("done in " + fmt("%.0f", since(t0)) + "s -> " + outdir);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const Exit& e) {
        if (e.message == "help") {
            usage();
            return 0;
        }
        std::fprintf(stderr, "helirec-reconstruct: %s\n", e.message.c_str());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "helirec-reconstruct: %s\n", e.what());
        return 1;
    }
}
