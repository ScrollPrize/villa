// helirec-composite: layer composites (max, mean) of a tifxyz surface, plus xy slices (helirec/surface_composite.hpp).
//
// Two sources:
//   * a reconstructed OME-Zarr block (helirec-reconstruct output):
//         helirec-composite VOLUME.zarr TIFXYZ_DIR --layers 15 --start -4 --tag seg
//   * the raw projections directly (band reconstruction: the exact backprojector at the pixels the layers need, no
//     volume), optionally sweeping filter parameters:
//         helirec-composite --projections PROJ --cache_dir CAL TIFXYZ_DIR --z Z0 Z1 --layers 15 --start -4
//         helirec-composite --projections PROJ ... --set "filters=paganin(delta_beta=1000)" --sweep paganin.delta_beta=250,500,1000,2000
//
// Layer k (k = 0 .. layers-1) is at offset direction (start + k step) voxels along the surface's unit normal
// (vc_reader's: dP/du x dP/dv, u increasing columns, v increasing rows); --direction auto steps toward the side with
// the higher mean intensity over offsets 1..5 (the material). Sampling and composites are vc_reader's.
//
// Output (one folder; per parameter set of a sweep, one subfolder each): max.tif, mean.tif (float32, NaN outside),
// max_u8.{tif,png}, mean_u8.{tif,png}, mask.png, composite.json, layers/NN_offset+O.tif, xy_slices/ (--xy_slices N),
// surface_volume.zarr (--surface_zarr). Pixels are full-resolution UV pixels of the surface (--out_shape renders the
// same extent at another size); composite.json gives the crop's origin on that canvas.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <set>

#include "helirec/surface_composite.hpp"
#include "io/command_line.hpp"
#include "io/json.hpp"

using namespace helirec;
using namespace helirec::command_line;
using namespace helirec::surface;
namespace fs = std::filesystem;

namespace {

const Spec& option_spec() {
    static const Spec spec{{
                               {"config", 1, 's'},     {"volume", 1, 's'},       {"surface", 1, 's'},       {"projections", 1, 's'},
                               {"cache_dir", 1, 's'},  {"z", 2, 'i'},            {"set", -2, 's'},          {"sweep", -2, 's'},
                               {"chunk", 1, 'i'},      {"no_prefetch", 0, 'b'},  {"device", 1, 's'},        {"level", 1, 'i'},
                               {"layers", 1, 'i'},     {"start", 1, 'f'},        {"step", 1, 'f'},          {"direction", 1, 's'},
                               {"interpolation", 1, 's'}, {"out_shape", 2, 'i'}, {"xy_slices", 1, 'i'},     {"slice_margin", 1, 'i'},
                               {"u8_window", 2, 'f'},  {"surface_zarr", 0, 'b'}, {"save_layers", 0, 'b'},   {"tag", 1, 's'},
                               {"out_root", 1, 's'},   {"overwrite", 0, 'b'},    {"inputs", -1, 's'},
                           },
                           {},
                           "inputs",
                           true};
    return spec;
}

int usage() {
    std::fprintf(stderr,
                 "usage: helirec-composite [VOLUME.zarr] TIFXYZ_DIR [--projections PROJ --cache_dir CAL] [--z Z0 Z1] [--set KEY=VALUE ...]\n"
                 "                         [--sweep KEY=V1,V2,...] [--chunk N] [--no_prefetch] [--device cuda|cpu] [--level L] [--layers N]\n"
                 "                         [--start S] [--step S] [--direction auto|+1|-1] [--interpolation bicubic|linear] [--out_shape H W]\n"
                 "                         [--xy_slices N] [--slice_margin M] [--u8_window LO HI] [--surface_zarr] [--tag TAG] [--out_root DIR]\n"
                 "                         [--overwrite] [--config FILE]\n"
                 "(see the head of tools/helirec_composite.cpp and README.md)\n");
    return 2;
}

void say(const std::string& line) {
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
}

std::string now(const char* format) {
    const std::time_t t = std::time(nullptr);
    char b[32];
    std::strftime(b, sizeof b, format, std::localtime(&t));
    return b;
}

std::string absolute(const std::string& p) { return fs::absolute(p).lexically_normal().string(); }

Json numbers(const std::vector<double>& v) {
    Json j = Json::list();
    for (double x : v) j.add(Json::of(x));
    return j;
}
Json ints(std::initializer_list<long long> v) {
    Json j = Json::list();
    for (long long x : v) j.add(Json::whole_number(x));
    return j;
}

// the units mode of an OME-Zarr volume: the helirec mode in its provenance; volumes without one (written by the
// reference workflow) are in legacy reference units
std::string volume_mode(const std::string& path) {
    try {
        const auto group = vcr::zarr::Group::open(path, vcr::zarr::OpenOptions{});
        const auto& a = group.attributes();
        if (a.contains("helirec") && a["helirec"].contains("mode")) return a["helirec"]["mode"].get<std::string>();
    } catch (const std::exception&) {
    }
    return "legacy_reference";
}

// the window's rows and columns where some pixel is good: (first row, rows, first column, columns)
std::array<int, 4> crop_of(const std::vector<std::uint8_t>& good, int H, int W) {
    int r0 = H, r1 = -1, c0 = W, c1 = -1;
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            if (good[static_cast<std::size_t>(r) * W + c]) r0 = std::min(r0, r), r1 = std::max(r1, r), c0 = std::min(c0, c), c1 = std::max(c1, c);
    if (r1 < 0) return {0, 0, 0, 0};
    return {r0, r1 - r0 + 1, c0, c1 - c0 + 1};
}

std::vector<float> crop_layers(const std::vector<float>& layers, int L, int H, int W, const std::array<int, 4>& c) {
    std::vector<float> out(static_cast<std::size_t>(L) * c[1] * c[3]);
    for (int l = 0; l < L; ++l)
        for (int r = 0; r < c[1]; ++r)
            for (int k = 0; k < c[3]; ++k)
                out[(static_cast<std::size_t>(l) * c[1] + r) * c[3] + k] = layers[(static_cast<std::size_t>(l) * H + c[0] + r) * W + c[2] + k];
    return out;
}

std::vector<std::uint8_t> crop_mask(const std::vector<std::uint8_t>& m, int W, const std::array<int, 4>& c) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(c[1]) * c[3]);
    for (int r = 0; r < c[1]; ++r)
        for (int k = 0; k < c[3]; ++k) out[static_cast<std::size_t>(r) * c[3] + k] = m[static_cast<std::size_t>(c[0] + r) * W + c[2] + k];
    return out;
}

std::vector<std::uint8_t> all_finite(const std::vector<float>& layers, int L, std::size_t px) {
    std::vector<std::uint8_t> good(px, 1);
    for (int l = 0; l < L; ++l)
        for (std::size_t p = 0; p < px; ++p) good[p] = good[p] && std::isfinite(layers[static_cast<std::size_t>(l) * px + p]);
    return good;
}

int run(int argc, char** argv) {
    const Json given = parse(argc, argv, option_spec());
    Args a{in_effect(Json::dict()
                         .set("set", Json::list())
                         .set("sweep", Json::list())
                         .set("device", Json::of("cuda"))
                         .set("level", Json::whole_number(0))
                         .set("layers", Json::whole_number(5))
                         .set("start", Json::of(0.0))
                         .set("step", Json::of(1.0))
                         .set("direction", Json::of("auto"))
                         .set("interpolation", Json::of("bicubic"))
                         .set("xy_slices", Json::whole_number(0))
                         .set("slice_margin", Json::whole_number(64))
                         .set("out_root", Json::of(absolute("helirec_composites"))),
                     given, option_spec())};
    const auto texts = a.texts("inputs");
    const bool from_projections = a.get("projections");
    std::string volume = a.text("volume"), surface_path = a.text("surface");
    if (surface_path.empty() && !texts.empty()) surface_path = texts.back();
    if (!from_projections) {
        if (volume.empty() && texts.size() >= 2) volume = texts.front();
        if (volume.empty()) fail("give a volume (OME-Zarr) or --projections");
    }
    if (surface_path.empty()) fail("surface (tifxyz) missing");
    const int n_layers = static_cast<int>(a.number("layers"));
    if (n_layers < 1) fail("--layers must be >= 1");
    const std::string direction_source = a.text("direction");
    if (direction_source != "auto" && direction_source != "+1" && direction_source != "-1") fail("--direction is auto, +1 or -1");
    const Json sweep = Json::parse(sweep_sets(a.texts("sweep")));
    if (!sweep.array.empty() && !from_projections) fail("--sweep needs --projections");
    const vcr::grid::Interpolation interpolation = vcr::grid::interpolationNamed(a.text("interpolation"));

    const auto t0 = std::chrono::steady_clock::now();
    const auto seconds = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    const std::string tag = a.get("tag") ? a.text("tag") : now("%Y%m%d_%H%M%S");
    const std::string outdir = absolute(a.text("out_root")) + "/" + tag;
    if (fs::exists(outdir) && !a.flag("overwrite")) fail(outdir + " exists (use another --tag or --overwrite)");
    fs::create_directories(outdir);
    {
        Json eff = Json::dict();
        for (const auto& [k, v] : a.all.object) {
            if (k == "config" || k == "inputs" || k == "save_layers" || v.is(Json::Null)) continue;
            eff.set(k, (k == "volume" || k == "surface" || k == "projections" || k == "cache_dir" || k == "out_root") && v.is(Json::String) ? Json::of(absolute(v.string)) : v);
        }
        if (!volume.empty()) put(eff, "volume", Json::of(absolute(volume)));
        put(eff, "surface", Json::of(absolute(surface_path)));
        put(eff, "tag", Json::of(tag));
        write_text(outdir + "/config.json", eff.text());
    }
    const std::string settings = from_projections ? parse_settings(a.texts("set")) : "{}";
    std::array<double, 2> window8{};
    if (a.get("u8_window")) window8 = {a.numbers("u8_window")[0], a.numbers("u8_window")[1]};
    else {
        const Json* mode = Json::parse(settings).find("mode");
        window8 = display_window(from_projections ? (mode ? mode->string : "corrected") : volume_mode(volume));
    }

    const vcr::tifxyz::Surface surf = vcr::tifxyz::read(surface_path);
    const auto full = surf.fullShape();
    std::array<double, 2> ps{1.0, 1.0};
    if (a.get("out_shape")) ps = {a.numbers("out_shape")[0] / full[0], a.numbers("out_shape")[1] / full[1]};
    const double start = a.number("start"), step = a.number("step");
    double reach = 0;
    for (double o : layer_offsets(1, start, step, n_layers)) reach = std::max(reach, std::fabs(o));
    const Json common = Json::dict()
                            .set("created", Json::of(now("%Y-%m-%dT%H:%M:%S")))
                            .set("surface", Json::of(absolute(surface_path)))
                            .set("surface_uuid", Json::of(surf.metadata().uuid))
                            .set("surface_full_shape", ints({full[0], full[1]}))
                            .set("surface_scale_rows_cols", numbers({surf.metadata().scale[0], surf.metadata().scale[1]}))
                            .set("interpolation", Json::of(a.text("interpolation")))
                            .set("normal", Json::of(std::string("dP/du x dP/dv (u = columns, v = rows), vc_reader / VC3D")))
                            .set("canvas_shape", ints({std::llround(full[0] * ps[0]), std::llround(full[1] * ps[1])}))
                            .set("pixel_scale_vs_full_res", numbers({ps[0], ps[1]}))
                            .set("u8_window", numbers({window8[0], window8[1]}));
    const auto meta_of = [&](const Json& extra) {
        Json m = common;
        for (const auto& [k, v] : extra.object) put(m, k, v);
        return m;
    };

    if (!from_projections) {
        // ============================================================ from a reconstructed volume
        Volume vol(volume, static_cast<int>(a.number("level")));
        const auto b = vol.bounds0();
        const Window window = uv_window(surf, b[0], b[1], std::max(reach, 15.0), ps);
        int direction = direction_source == "-1" ? -1 : 1;
        Json side;
        if (direction_source == "auto") {
            const Side s = material_side(surf, vol, window, ps, interpolation);
            direction = s.direction, side = Json::parse(s.evidence);
        }
        const auto offsets = layer_offsets(direction, start, step, n_layers);
        const int H = window.r1 - window.r0, W = window.c1 - window.c0;
        const auto layers = sample_layers(surf, vol, window, ps, offsets, interpolation);
        const auto good_all = all_finite(layers, n_layers, static_cast<std::size_t>(H) * W);
        const auto c = crop_of(good_all, H, W);
        if (!c[1]) fail("no surface pixel has all its layers inside the volume");
        const auto good = crop_mask(good_all, W, c);
        std::vector<Slice> slices;
        for (int z : spread(0, static_cast<int>(vol.shape()[0]), static_cast<int>(a.number("xy_slices"))))
            slices.push_back(Slice{z * vol.factor() + vol.origin0()[0], vol.slice(z), static_cast<int>(vol.shape()[1]), static_cast<int>(vol.shape()[2])});
        std::size_t valid = 0;
        for (auto g : good) valid += g;
        const int r0 = window.r0 + c[0], c0 = window.c0 + c[2];
        const Json meta = meta_of(Json::dict()
                                      .set("source", Json::of(std::string("volume")))
                                      .set("volume", Json::of(vol.path()))
                                      .set("level", Json::whole_number(vol.level()))
                                      .set("volume_origin_zyx_level0", ints({vol.origin0()[0], vol.origin0()[1], vol.origin0()[2]}))
                                      .set("volume_shape", ints({vol.shape()[0], vol.shape()[1], vol.shape()[2]}))
                                      .set("voxel_um", Json::of(vol.voxel_um()))
                                      .set("direction", Json::whole_number(direction))
                                      .set("direction_source", Json::of(direction_source))
                                      .set("side_test", side)
                                      .set("layer_offsets_voxels", numbers(offsets))
                                      .set("crop", Json::dict().set("row0", Json::whole_number(r0)).set("col0", Json::whole_number(c0)).set("rows", Json::whole_number(c[1])).set("cols", Json::whole_number(c[3])))
                                      .set("valid_pixels", Json::whole_number(static_cast<long long>(valid)))
                                      .set("xy_slice_frame", Json::of(std::string("slices of the volume block: zarr (y, x) of the stored crop"))));
        write_set(outdir, crop_layers(layers, n_layers, H, W, c), c[1], c[3], offsets, good, meta.text(), window8, slices, a.flag("surface_zarr"));
        char line[256];
        std::snprintf(line, sizeof line, "%d x %d px (%zu valid), canvas origin (row %d, col %d), direction %+d, offsets %+g..%+g -> %s (%.0fs)", c[1], c[3], valid, r0, c0,
                      direction, offsets.front(), offsets.back(), outdir.c_str(), seconds());
        say(line);
        return 0;
    }

    // ================================================================ from the projections (band reconstruction)
    Reconstruction::Options opt;
    opt.cache_dir = a.text("cache_dir");
    opt.device = a.text("device") == "cpu" ? Device::CPU : Device::CUDA;
    Reconstruction r(a.text("projections"), settings, opt);
    const int N = r.reconstructor()->grid_size();
    const auto [lo, hi] = r.doable();
    int z0 = 0, z1 = hi - lo;
    if (a.get("z")) z0 = a.ints("z")[0], z1 = a.ints("z")[1];
    z0 = std::max(0, z0), z1 = std::min(hi - lo, z1);
    const int h0 = z0 + lo, h1 = z1 + lo;
    const Window window = uv_window(surf, {double(z0), 0, 0}, {double(z1 - 1), double(N - 1), double(N - 1)}, std::max(reach, 15.0), ps);
    const vcr::grid::Raster raster = raster_of(surf, window, ps);
    const Points P = points(surf, raster, interpolation);
    // offsets to render: the layers for the given direction, or both directions and the side test's -5..5
    std::vector<double> render;
    if (direction_source == "auto") {
        std::set<double> s;
        for (int d : {1, -1})
            for (double o : layer_offsets(d, start, step, n_layers)) s.insert(o);
        for (int d = -5; d <= 5; ++d) s.insert(d);
        render.assign(s.begin(), s.end());
    } else {
        render = layer_offsets(direction_source == "-1" ? -1 : 1, start, step, n_layers);
    }
    const int L = static_cast<int>(render.size()), H = P.height, W = P.width;
    const std::size_t px = static_cast<std::size_t>(H) * W;
    std::vector<float> pts(static_cast<std::size_t>(L) * px * 3);
    for (int l = 0; l < L; ++l)
        for (std::size_t p = 0; p < px; ++p)
            for (int k = 0; k < 3; ++k) pts[(l * px + p) * 3 + k] = static_cast<float>(P.xyz[3 * p + k] + render[static_cast<std::size_t>(l)] * P.normals[3 * p + k]);
    // the sweep's filter chains, each the reconstructor's chain with the set's changes
    const std::string base_filters = Json::parse(r.parameters()).find("filters")->text();
    std::vector<std::string> chains;
    for (const Json& set : sweep.array) {
        Json changes = Json::dict();
        for (const auto& [k, v] : set.object) changes.set(k, v);
        chains.push_back(override_filter_chain(base_filters, changes.text()));
    }
    const int K = std::max<int>(1, static_cast<int>(chains.size()));
    BandComposite bc(r, pts.data(), L, H, W, P.ok.data(), std::array<int, 2>{h0, h1}, K);
    pts.clear(), pts.shrink_to_fit();
    const int chunk = a.get("chunk") ? static_cast<int>(a.number("chunk")) : (K > 1 ? 40 : 64);
    say("surface " + surf.metadata().uuid + ": window rows (" + std::to_string(window.r0) + ", " + std::to_string(window.r1) + ") cols (" + std::to_string(window.c0) + ", " +
        std::to_string(window.c1) + "), " + std::to_string(bc.n_points()) + " points, z " + std::to_string(z0) + ".." + std::to_string(z1 - 1) + " (h " +
        std::to_string(h0) + ".." + std::to_string(h1 - 1) + "), " + std::to_string(K) + " parameter set(s), chunk " + std::to_string(chunk));
    const Json sec = Json::parse(bc.run(chunk, false, chains, !a.flag("no_prefetch"), [](int d, int n, double el) {
        say("  " + std::to_string(d) + "/" + std::to_string(n) + " slices, " + std::to_string(static_cast<long>(std::lround(el))) + "s");
    }));
    const double wall = sec.find("wall")->number;
    char line[256];
    std::snprintf(line, sizeof line, "band reconstruction %.1fs (%.1f ms/slice)", wall, 1e3 * wall / (h1 - h0));
    say(line);
    const auto c = crop_of(bc.valid(), H, W);
    if (!c[1]) fail("no surface pixel has all its layers inside the z range");
    const int r0 = window.r0 + c[0], c0 = window.c0 + c[2];
    const auto [bx0, bx1, by0, by1] = bc.box();
    const int m = static_cast<int>(a.number("slice_margin"));
    const SliceRoi slice_roi = m < 0 ? SliceRoi{0, N, 0, N} : SliceRoi{std::max(0, bx0 - m), std::min(N, bx1 + m), std::max(0, by0 - m), std::min(N, by1 + m)};
    const auto zs = spread(z0, z1, static_cast<int>(a.number("xy_slices")));
    const auto good = crop_mask(bc.valid(), W, c);
    std::size_t valid = 0;
    for (auto g : good) valid += g;
    std::vector<FilterChain> sweep_chains;
    for (const auto& ch : chains) sweep_chains.push_back(filter_chain(ch));
    for (int k = 0; k < K; ++k) {
        const auto lay_all = crop_layers(bc.layers(k), L, H, W, c);
        int direction = direction_source == "-1" ? -1 : 1;
        Json side;
        if (direction_source == "auto") {
            const Side s = side_from_layers(lay_all, c[1], c[3], render);
            direction = s.direction, side = Json::parse(s.evidence);
        }
        const auto offsets = layer_offsets(direction, start, step, n_layers);
        std::vector<float> layers;
        const std::size_t cpx = static_cast<std::size_t>(c[1]) * c[3];
        for (double o : offsets) {
            const std::size_t i = static_cast<std::size_t>(std::find(render.begin(), render.end(), o) - render.begin());
            layers.insert(layers.end(), lay_all.begin() + static_cast<std::ptrdiff_t>(i * cpx), lay_all.begin() + static_cast<std::ptrdiff_t>((i + 1) * cpx));
        }
        if (!chains.empty()) r.set_parameters("{\"filters\": " + chains[static_cast<std::size_t>(k)] + "}");
        std::vector<Slice> slices;
        const int sy = slice_roi.y1 - slice_roi.y0, sx = slice_roi.x1 - slice_roi.x0;
        for (int z : zs) {
            Slice s{z, std::vector<float>(static_cast<std::size_t>(sy) * sx), sy, sx};
            r.reconstructor()->reconstruct(z + lo, z + lo + 1, s.values.data(), nullptr, &slice_roi);
            slices.push_back(std::move(s));
        }
        const std::string sub = chains.empty() ? std::string() : set_name(sweep.array[static_cast<std::size_t>(k)].text());
        const std::string out_k = sub.empty() ? outdir : outdir + "/" + sub;
        fs::create_directories(out_k);
        const Json params = Json::parse(r.parameters());
        const Json meta = meta_of(Json::dict()
                                      .set("source", Json::of(std::string("projections")))
                                      .set("projections", Json::of(absolute(a.text("projections"))))
                                      .set("params", params)
                                      .set("provenance", Json::parse(r.provenance()))
                                      .set("sweep_set", chains.empty() ? Json() : sweep.array[static_cast<std::size_t>(k)])
                                      .set("filters", *params.find("filters"))
                                      .set("sweep_margin_rows", chains.empty() ? Json() : Json::whole_number(r.reconstructor()->sweep_margin(sweep_chains)))
                                      .set("z", ints({z0, z1}))
                                      .set("view_heights", ints({h0, h1}))
                                      .set("backprojector", *params.find("backprojector"))
                                      .set("composite_path", Json::of(std::string(bc.band() ? "band (exact backprojector at the layer pixels)" : "ROI slices + trilinear sampling")))
                                      .set("direction", Json::whole_number(direction))
                                      .set("direction_source", Json::of(direction_source))
                                      .set("side_test", side)
                                      .set("layer_offsets_voxels", numbers(offsets))
                                      .set("crop", Json::dict().set("row0", Json::whole_number(r0)).set("col0", Json::whole_number(c0)).set("rows", Json::whole_number(c[1])).set("cols", Json::whole_number(c[3])))
                                      .set("valid_pixels", Json::whole_number(static_cast<long long>(valid)))
                                      .set("band_box_x0x1y0y1", ints({bx0, bx1, by0, by1}))
                                      .set("xy_slice_roi_x0x1y0y1", ints({slice_roi.x0, slice_roi.x1, slice_roi.y0, slice_roi.y1}))
                                      .set("seconds_band", Json::of(std::round(wall * 100) / 100)));
        write_set(out_k, layers, c[1], c[3], offsets, good, meta.text(), window8, slices, a.flag("surface_zarr"));
        std::snprintf(line, sizeof line, "%d x %d px (%zu valid), canvas origin (row %d, col %d), direction %+d, %zu xy slices -> %s", c[1], c[3], valid, r0, c0, direction,
                      slices.size(), out_k.c_str());
        say((sub.empty() ? std::string() : "set " + sub + ": ") + line);
    }
    if (!chains.empty()) r.set_parameters("{\"filters\": " + base_filters + "}");
    std::snprintf(line, sizeof line, "done in %.0fs", seconds());
    say(line);
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
        std::fprintf(stderr, "helirec-composite: %s\n", e.message.c_str());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "helirec-composite: %s\n", e.what());
        return 1;
    }
}
