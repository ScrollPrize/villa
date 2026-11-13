// helirec/surface_composite.hpp: layers along a surface from a volume block, the material side, and the outputs.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <set>

#include <map>
#include "helirec/kernels.hpp"
#include "helirec/surface_composite.hpp"
#include "helirec/zarr.hpp"
#include "io/json.hpp"
#include "vc_reader/array.hpp"
#include "vc_reader/composite.hpp"
#include "vc_reader/image_write.hpp"

namespace helirec::surface {

namespace {

constexpr float kNaNf = std::numeric_limits<float>::quiet_NaN();

// Python's round(): half to even
long round_half_even(double v) { return static_cast<long>(std::nearbyint(v)); }

// the q-th percentile of the finite values (linear between ranks)
double percentile(std::vector<float> v, double q) {
    v.erase(std::remove_if(v.begin(), v.end(), [](float x) { return !std::isfinite(x); }), v.end());
    if (v.empty()) return std::nan("");
    std::sort(v.begin(), v.end());
    const double rank = q / 100.0 * static_cast<double>(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(rank), hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (rank - static_cast<double>(lo)) * (static_cast<double>(v[hi]) - v[lo]);
}

void write_text(const std::string& path, const std::string& text) {
    std::ofstream o(path);
    o << text << "\n";
    if (!o) throw Error("cannot write " + path);
}

void png(const std::string& path, const std::vector<std::uint8_t>& img, int ny, int nx) {
    vcr::sources::writeImage(path, vcr::sources::ImageView{img.data(), "|u1", ny, nx});
}

void tiff(const std::string& path, const void* data, const char* dtype, int ny, int nx, bool deflate) {
    vcr::sources::TiffOptions o;
    o.compression = deflate ? "deflate" : "none";
    vcr::sources::writeTiff(path, data, dtype, ny, nx, 1, o);
}

// rint(clip((img - lo) / (hi - lo), 0, 1) 255) as uint8 in float32, 0 where the value is not finite
std::vector<std::uint8_t> to_u8(const float* img, int ny, int nx, const std::array<double, 2>& window) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(ny) * nx);
    Uint8Window w;
    w.lo = window[0], w.hi = window[1];
    cpu::to_uint8(img, static_cast<std::size_t>(nx), out.data(), ny, nx, w);
    for (std::size_t i = 0; i < out.size(); ++i)
        if (!std::isfinite(img[i])) out[i] = 0;
    return out;
}

std::string format_offset(double o) {  // Python f"{o:+05.1f}"
    char b[32];
    std::snprintf(b, sizeof b, "%+05.1f", o);
    return b;
}

}  // namespace

std::vector<double> layer_offsets(int direction, double start, double step, int n) {
    std::vector<double> out;
    for (int k = 0; k < n; ++k) out.push_back(direction * (start + k * step) + 0.0);
    return out;
}

std::vector<int> spread(int z0, int z1, int n) {
    std::set<int> s;
    for (int k = 0; k < n; ++k) s.insert(static_cast<int>(z0 + (k + 0.5) * (z1 - z0) / n));
    return std::vector<int>(s.begin(), s.end());
}

// ---------------------------------------------------------------- the volume block

Volume::Volume(const std::string& path_in, int level) : level_(level) {
    path_ = std::filesystem::absolute(path_in).lexically_normal().string();
    vcr::zarr::Group group = vcr::zarr::Group::open(path_, vcr::zarr::OpenOptions{});
    const vcr::zarr::Json ms = group.multiscales();
    if (!ms.is_array() || ms.empty()) throw Error(path_ + ": no OME multiscales");
    const auto& datasets = ms[0].at("datasets");
    if (level < 0 || static_cast<std::size_t>(level) >= datasets.size()) throw Error(path_ + ": no level " + std::to_string(level));
    const std::string key = datasets[static_cast<std::size_t>(level)].at("path").get<std::string>();
    const vcr::zarr::LevelPlacement declared = vcr::zarr::declaredPlacement(group.attributes(), key);  // ZYX scale and translation
    s_ = 1 << level;
    voxel_um_ = declared.scale[0] / s_;
    // level-l index i covers level-0 voxels [o + s i, o + s i + s - 1], centre o + s i + (s - 1) / 2
    for (int k = 0; k < 3; ++k) origin0_[k] = round_half_even(declared.offset[k] / voxel_um_ - (s_ - 1) / 2.0);
    array_ = group.array(key);
    const auto& shape = array_.meta.shape;
    if (shape.size() != 3) throw Error(path_ + ": level " + key + " is not a 3-D array");
    for (int k = 0; k < 3; ++k) shape_[k] = shape[k];
}

std::array<double, 3> Volume::index(const double* xyz) const {
    std::array<double, 3> out;
    for (int k = 0; k < 3; ++k) out[k] = (xyz[2 - k] - origin0_[k] - (s_ - 1) / 2.0) / s_;
    return out;
}

std::array<std::array<double, 3>, 2> Volume::bounds0() const {
    std::array<std::array<double, 3>, 2> b;
    for (int k = 0; k < 3; ++k) {
        b[0][k] = static_cast<double>(origin0_[k]) + (s_ - 1) / 2.0;
        b[1][k] = b[0][k] + s_ * (static_cast<double>(shape_[k]) - 1);
    }
    return b;
}

vcr::grid::Level Volume::grid_level() {
    std::array<double, 12> m{};  // row-major 3 x 4: output axis k (Z, Y, X) reads input component 2 - k
    for (int k = 0; k < 3; ++k) {
        m[static_cast<std::size_t>(4 * k + 2 - k)] = 1.0 / s_;
        m[static_cast<std::size_t>(4 * k + 3)] = -(origin0_[k] + (s_ - 1) / 2.0) / s_;
    }
    return vcr::grid::levelFromMap(*array_.array, array_.meta.type, m);
}

std::vector<float> Volume::slice(std::int64_t z) {
    std::vector<float> out(static_cast<std::size_t>(shape_[1] * shape_[2]));
    array_.array->read_many_float({vcr::Index{z, 0, 0}}, {vcr::Index{1, shape_[1], shape_[2]}}, {reinterpret_cast<std::uint8_t*>(out.data())}, array_.meta.type,
                                  false);
    return out;
}

// ---------------------------------------------------------------- the surface over the block

Window uv_window(const vcr::tifxyz::Surface& surf, const std::array<double, 3>& lo, const std::array<double, 3>& hi, double reach,
                 const std::array<double, 2>& ps) {
    const std::int64_t R = surf.rows(), C = surf.cols();
    const auto& xyz = surf.xyz();
    const auto& valid = surf.valid();
    std::int64_t rmin = -1, rmax = -1, cmin = -1, cmax = -1;
    for (std::int64_t r = 0; r < R; ++r)
        for (std::int64_t c = 0; c < C; ++c) {
            const std::size_t i = static_cast<std::size_t>(r * C + c);
            if (!valid[i]) continue;
            bool near = true;
            for (int k = 0; k < 3; ++k) {  // ZYX of the vertex against the box
                const double v = xyz[3 * i + 2 - k];
                near = near && v >= lo[k] - reach - 2 && v <= hi[k] + reach + 2;
            }
            if (!near) continue;
            rmin = rmin < 0 ? r : std::min(rmin, r), rmax = std::max(rmax, r);
            cmin = cmin < 0 ? c : std::min(cmin, c), cmax = std::max(cmax, c);
        }
    if (rmin < 0) throw Error("the surface does not reach the volume block");
    const double sy = surf.metadata().scale[0], sx = surf.metadata().scale[1];
    const auto full = surf.fullShape();
    Window w;
    w.H = static_cast<int>(round_half_even(full[0] * ps[0]));
    w.W = static_cast<int>(round_half_even(full[1] * ps[1]));
    w.r0 = std::max(0, static_cast<int>(std::floor((rmin - 1) / sy * ps[0])));
    w.r1 = std::min(w.H, static_cast<int>(std::ceil((rmax + 1) / sy * ps[0])) + 1);
    w.c0 = std::max(0, static_cast<int>(std::floor((cmin - 1) / sx * ps[1])));
    w.c1 = std::min(w.W, static_cast<int>(std::ceil((cmax + 1) / sx * ps[1])) + 1);
    return w;
}

vcr::grid::Raster raster_of(const vcr::tifxyz::Surface& surf, const Window& w, const std::array<double, 2>& ps) {
    const double sy = surf.metadata().scale[0], sx = surf.metadata().scale[1];
    vcr::grid::Raster r;
    r.x0 = w.c0 * sx / ps[1];
    r.y0 = w.r0 * sy / ps[0];
    r.sx = sx / ps[1];
    r.sy = sy / ps[0];
    r.width = w.c1 - w.c0;
    r.height = w.r1 - w.r0;
    return r;
}

Points points(const vcr::tifxyz::Surface& surf, const vcr::grid::Raster& raster, vcr::grid::Interpolation interpolation) {
    Points p;
    p.width = raster.width, p.height = raster.height;
    const std::size_t n = static_cast<std::size_t>(raster.width) * static_cast<std::size_t>(raster.height);
    std::vector<float> pos(3 * n), nrm(3 * n);
    surf.resample(raster, interpolation, pos.data(), nrm.data());
    p.xyz.assign(3 * n, 0.0), p.normals.assign(3 * n, 0.0), p.ok.assign(n, 0);
    for (std::size_t k = 0; k < n; ++k) {
        bool ok = true;
        for (int a = 0; a < 3; ++a) ok = ok && std::isfinite(pos[3 * k + a]) && std::isfinite(nrm[3 * k + a]);
        if (!ok) continue;
        p.ok[k] = 1;
        for (int a = 0; a < 3; ++a) p.xyz[3 * k + a] = pos[3 * k + a], p.normals[3 * k + a] = nrm[3 * k + a];
    }
    return p;
}

std::vector<float> sample_layers(const vcr::tifxyz::Surface& surf, Volume& vol, const Window& w, const std::array<double, 2>& ps, const std::vector<double>& offsets,
                                 vcr::grid::Interpolation interpolation, int tile_rows) {
    const int L = static_cast<int>(offsets.size()), H = w.r1 - w.r0, W = w.c1 - w.c0;
    std::vector<float> out(static_cast<std::size_t>(L) * H * W);
    vcr::grid::Request request;  // (vc_reader's defaults: linear, NaN where there is no value)
    request.levels = {vol.grid_level()};
    request.offsets.assign(offsets.begin(), offsets.end());
    const vcr::grid::Surface& g = surf.interpolator(interpolation);
    for (int a = w.r0; a < w.r1; a += tile_rows) {
        Window t = w;
        t.r0 = a, t.r1 = std::min(w.r1, a + tile_rows);
        const vcr::grid::Raster raster = raster_of(surf, t, ps);
        const vcr::grid::Result res = vcr::grid::sample(g, raster, request);
        const std::size_t n = static_cast<std::size_t>(raster.width) * raster.height;
        for (int l = 0; l < L; ++l)
            std::copy(res.values.begin() + static_cast<std::ptrdiff_t>(l * n), res.values.begin() + static_cast<std::ptrdiff_t>((l + 1) * n),
                      out.begin() + static_cast<std::ptrdiff_t>((static_cast<std::size_t>(l) * H + (a - w.r0)) * W));
    }
    return out;
}

Side side_from_layers(const std::vector<float>& layers, int H, int W, const std::vector<double>& offsets, int depth) {
    const std::size_t px = static_cast<std::size_t>(H) * W;
    const auto at = [&](double o) -> int {
        for (std::size_t i = 0; i < offsets.size(); ++i)
            if (offsets[i] == o) return static_cast<int>(i);
        return -1;
    };
    for (int d = -depth; d <= depth; ++d)
        if (d && at(d) < 0) throw Error("side test needs offsets -" + std::to_string(depth) + ".." + std::to_string(depth));
    std::vector<std::uint8_t> good(px, 1);
    std::size_t n_good = 0;
    for (std::size_t p = 0; p < px; ++p) {
        for (int d = -depth; d <= depth && good[p]; ++d)
            if (d) good[p] = std::isfinite(layers[static_cast<std::size_t>(at(d)) * px + p]);
        n_good += good[p];
    }
    if (n_good < 100) throw Error("too few surface pixels inside the volume to decide the material side");
    // the mean of every offset's layer over the good pixels
    std::vector<double> sorted(offsets);
    std::sort(sorted.begin(), sorted.end());
    std::map<double, double> prof;
    for (double o : sorted) {
        const std::size_t i = static_cast<std::size_t>(at(o));
        double sum = 0;
        for (std::size_t p = 0; p < px; ++p)
            if (good[p]) sum += layers[i * px + p];
        prof[o] = sum / static_cast<double>(n_good);
    }
    double plus = 0, minus = 0;
    for (int d = 1; d <= depth; ++d) plus += prof[d] / depth, minus += prof[-d] / depth;
    std::size_t brighter = 0;
    for (std::size_t p = 0; p < px; ++p) {
        if (!good[p]) continue;
        double diff = 0;
        for (int d = 1; d <= depth; ++d) diff += layers[static_cast<std::size_t>(at(d)) * px + p] - layers[static_cast<std::size_t>(at(-d)) * px + p];
        brighter += diff > 0;
    }
    Side s;
    s.direction = plus > minus ? 1 : -1;
    Json offs = Json::list(), means = Json::list();
    for (const auto& [o, m] : prof) offs.add(Json::of(o)), means.add(Json::of(m));
    s.evidence = Json::dict()
                     .set("offsets", offs)
                     .set("mean", means)
                     .set("plus_side_mean", Json::of(plus))
                     .set("minus_side_mean", Json::of(minus))
                     .set("frac_pixels_plus_brighter", Json::of(static_cast<double>(brighter) / static_cast<double>(n_good)))
                     .set("n_pixels", Json::whole_number(static_cast<long long>(n_good)))
                     .text();
    return s;
}

Side material_side(const vcr::tifxyz::Surface& surf, Volume& vol, const Window& w, const std::array<double, 2>& ps, vcr::grid::Interpolation interpolation, int depth,
                   int sub) {
    std::vector<double> offs;
    for (int d = -3 * depth; d <= 3 * depth; ++d) offs.push_back(d);
    // every sub-th output pixel is enough for a mean profile
    Window s = w;
    s.r0 = w.r0 / sub, s.r1 = (w.r1 + sub - 1) / sub, s.c0 = w.c0 / sub, s.c1 = (w.c1 + sub - 1) / sub;
    const auto layers = sample_layers(surf, vol, s, {ps[0] / sub, ps[1] / sub}, offs, interpolation);
    return side_from_layers(layers, s.r1 - s.r0, s.c1 - s.c0, offs, depth);
}

// ---------------------------------------------------------------- the outputs

void write_set(const std::string& out, const std::vector<float>& layers, int H, int W, const std::vector<double>& offsets, const std::vector<std::uint8_t>& good,
               std::string meta_text, const std::array<double, 2>& u8_window, const std::vector<Slice>& slices, bool surface_zarr) {
    namespace fs = std::filesystem;
    fs::create_directories(fs::path(out) / "layers");
    const std::size_t px = static_cast<std::size_t>(H) * W, L = offsets.size();
    // max and mean of the layers where every layer is there (vc_reader's reduction; no cutoff)
    std::vector<float> filled(L * px);
    for (std::size_t l = 0; l < L; ++l)
        for (std::size_t p = 0; p < px; ++p) filled[l * px + p] = good[p] ? layers[l * px + p] : 0.0f;
    std::map<std::string, std::vector<float>> comp;
    for (const auto& [name, method] : {std::pair<const char*, vcr::composite::Method>{"max", vcr::composite::Method::Max}, {"mean", vcr::composite::Method::Mean}}) {
        vcr::composite::Options o;
        o.method = method;
        o.isoCutoff = -std::numeric_limits<float>::infinity();
        std::vector<float> v(px);
        vcr::composite::reduce(filled.data(), L, px, o, v.data());
        for (std::size_t p = 0; p < px; ++p)
            if (!good[p]) v[p] = kNaNf;
        comp[name] = std::move(v);
    }
    for (const auto& [k, img] : comp) {
        tiff(out + "/" + k + ".tif", img.data(), "<f4", H, W, true);
        const auto u8 = to_u8(img.data(), H, W, u8_window);
        tiff(out + "/" + k + "_u8.tif", u8.data(), "|u1", H, W, true);
        png(out + "/" + k + "_u8.png", u8, H, W);
    }
    std::vector<std::uint8_t> mask(px);
    for (std::size_t p = 0; p < px; ++p) mask[p] = good[p] ? 255 : 0;
    png(out + "/mask.png", mask, H, W);
    std::vector<float> layer(px);
    for (std::size_t l = 0; l < L; ++l) {
        for (std::size_t p = 0; p < px; ++p) layer[p] = good[p] ? layers[l * px + p] : kNaNf;
        char name[32];
        std::snprintf(name, sizeof name, "%02zu", l);
        tiff(out + "/layers/" + name + "_offset" + format_offset(offsets[l]) + ".tif", layer.data(), "<f4", H, W, true);
    }
    Json meta = Json::parse(meta_text);
    const auto put = [&](const std::string& key, Json v) {
        for (auto& kv : meta.object)
            if (kv.first == key) return void(kv.second = std::move(v));
        meta.object.emplace_back(key, std::move(v));
    };
    if (!slices.empty()) {
        fs::create_directories(fs::path(out) / "xy_slices");
        Json zs = Json::list();
        for (const Slice& s : slices) {
            char name[32];
            std::snprintf(name, sizeof name, "/xy_slices/z%05lld", static_cast<long long>(s.z));
            tiff(out + name + ".tif", s.values.data(), "<f4", s.ny, s.nx, false);
            png(out + name + ".png", to_u8(s.values.data(), s.ny, s.nx, u8_window), s.ny, s.nx);
            zs.add(Json::whole_number(s.z));
        }
        put("xy_slices", zs);
    }
    if (surface_zarr) {
        // the layer stack as a uint8 OME-Zarr surface volume (layers, H, W): depth first, the surface at the centre layer
        std::vector<std::uint8_t> u8(L * px, 0);
        for (std::size_t l = 0; l < L; ++l) {
            const auto v = to_u8(layers.data() + l * px, H, W, u8_window);
            for (std::size_t p = 0; p < px; ++p) u8[l * px + p] = good[p] ? v[p] : 0;
        }
        const Json* crop = meta.find("crop");
        const Json* vox = meta.find("voxel_um");
        OmeZarrOptions o;
        o.shape = {static_cast<std::int64_t>(L), H, W};
        o.dtype = "|u1";
        o.chunks = {static_cast<std::int64_t>(L), 128, 128};
        o.voxel_size_um = vox && vox->is(Json::Number) ? vox->number : 8.64;
        o.origin = {0, static_cast<std::int64_t>(crop->find("row0")->number), static_cast<std::int64_t>(crop->find("col0")->number)};
        o.levels = 3;
        o.compression = "none";
        o.overwrite = true;
        o.name = "surface volume";
        o.attrs_json = Json::dict()
                           .set("helirec_surface_volume", Json::dict().set("u8_window", Json::list().add(Json::of(u8_window[0])).add(Json::of(u8_window[1]))))
                           .text();
        OmeZarrWriter writer(out + "/surface_volume.zarr", o);
        writer.write(0, static_cast<std::int64_t>(L), u8.data());
        writer.close();
        Json offs = Json::list();
        for (double x : offsets) offs.add(Json::of(x));
        put("surface_volume", Json::dict()
                                  .set("path", Json::of(std::string("surface_volume.zarr")))
                                  .set("layers", Json::whole_number(static_cast<long long>(L)))
                                  .set("offsets", offs)
                                  .set("centre_layer", Json::whole_number(static_cast<long long>(L / 2)))
                                  .set("dtype", Json::of(std::string("uint8")))
                                  .set("window", Json::list().add(Json::of(u8_window[0])).add(Json::of(u8_window[1]))));
    }
    Json stats = Json::dict();
    for (const auto& [k, v] : comp)
        stats.set(k, Json::dict().set("p1", Json::of(percentile(v, 1))).set("p50", Json::of(percentile(v, 50))).set("p99", Json::of(percentile(v, 99))));
    put("stats", stats);
    write_text(out + "/composite.json", meta.text());
}

// ---------------------------------------------------------------- sweeps

std::string sweep_sets(const std::vector<std::string>& items) {
    std::vector<std::pair<std::string, std::vector<Json>>> cols;
    std::size_t n = 0;
    for (const std::string& item : items) {
        const std::size_t eq = item.find('=');
        if (eq == std::string::npos) throw Error("bad --sweep '" + item + "', expected KEY=V1,V2,...");
        const std::string key = item.substr(0, eq), vals = item.substr(eq + 1);
        std::vector<Json> v;
        const char sep = key == "filters" ? ';' : ',';
        std::size_t i = 0;
        while (i <= vals.size()) {
            std::size_t j = vals.find(sep, i);
            if (j == std::string::npos) j = vals.size();
            const std::string t = vals.substr(i, j - i);
            v.push_back(key == "filters" ? Json::of(t) : Json::parse(parse_value(t)));
            i = j + 1;
        }
        n = std::max(n, v.size());
        cols.emplace_back(key, std::move(v));
    }
    for (const auto& [k, v] : cols)
        if (v.size() != 1 && v.size() != n) throw Error("--sweep lists must have one length (or one value)");
    Json out = Json::list();
    for (std::size_t i = 0; i < n; ++i) {
        Json set = Json::dict();
        for (const auto& [k, v] : cols) set.set(k, v.size() > 1 ? v[i] : v[0]);
        out.add(set);
    }
    return out.text();
}

std::string set_name(const std::string& changes) {
    const auto shorten = [](std::string k) {
        const auto replace = [&](const std::string& a, const std::string& b) {
            for (std::size_t i = k.find(a); i != std::string::npos; i = k.find(a, i + b.size())) k.replace(i, a.size(), b);
        };
        replace("mtf_paganin.", "mp_");
        replace("paganin.delta_beta", "db");
        replace("unsharp.", "us_");
        replace(".", "_");
        return k;
    };
    std::string out;
    for (const auto& [k, v] : Json::parse(changes).object) {
        char b[64];
        std::string value;
        if (v.is(Json::Number)) std::snprintf(b, sizeof b, "%g", v.number), value = b;
        else value = v.is(Json::String) ? v.string : v.text();
        out += (out.empty() ? "" : "_") + shorten(k) + value;
    }
    return out;
}

}  // namespace helirec::surface
