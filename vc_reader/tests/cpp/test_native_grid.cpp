// Surfaces stored as vertex grids: positions and normals at raster pixels, and
// volumes sampled along surfaces and planes with layers, fine-to-coarse levels,
// cache-only reads, pixel masks and composites.
#include <vc_reader/array.hpp>
#include <vc_reader/grid.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

void require(bool value, const std::string& message)
{
    if (!value)
        throw std::runtime_error(message);
}

bool near(float a, float b, float tolerance = 1e-4f) { return std::abs(a - b) <= tolerance; }

// rows x cols vertices on the plane z = 10: vertex (r, c) at (2c, 3r, 10).
std::vector<float> flatGrid(int rows, int cols)
{
    std::vector<float> xyz(std::size_t(rows) * std::size_t(cols) * 3);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            float* p = &xyz[(std::size_t(r) * std::size_t(cols) + std::size_t(c)) * 3];
            p[0] = 2.0f * float(c);
            p[1] = 3.0f * float(r);
            p[2] = 10.0f;
        }
    return xyz;
}

void testDefaultInterpolation()
{
    // Interior Hermite patches reproduce z = x^2 and its normal exactly. A
    // bilinear patch at x = 1.25 instead has z = 1.75 and slope 3.
    std::vector<float> xyz(5 * 5 * 3);
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 5; ++c) {
            float* p = &xyz[(r * 5 + c) * 3];
            p[0] = float(c);
            p[1] = float(r);
            p[2] = float(c * c);
        }
    const vcr::grid::Vertices vertices{xyz.data(), 5, 5, 15};
    vcr::grid::Surface surface(vertices);
    float p[3], n[3];
    surface.resample({1.25, 1.5, 1, 1, 1, 1}, p, n);
    const float length = std::sqrt(7.25f);
    require(near(p[0], 1.25f) && near(p[1], 1.5f) && near(p[2], 1.5625f), "default bicubic position");
    require(near(n[0], -2.5f / length) && near(n[1], 0) && near(n[2], 1 / length), "default bicubic normal");
    vcr::grid::Surface linear(vertices, {vcr::grid::Interpolation::Linear});
    linear.resample({1.25, 1.5, 1, 1, 1, 1}, p, n);
    require(near(p[2], 1.75f) && near(n[0], -3 / std::sqrt(10.0f)), "explicit bilinear remains available");
}

void testResample()
{
    const int rows = 6, cols = 8;
    auto xyz = flatGrid(rows, cols);
    const vcr::grid::Vertices v{xyz.data(), rows, cols, std::size_t(cols) * 3};
    const vcr::grid::Raster raster{1.25, 0.5, 0.5, 0.25, 9, 15};
    for (const auto mode : {vcr::grid::Interpolation::Linear, vcr::grid::Interpolation::Bicubic}) {
        vcr::grid::Surface surface(v, {mode});
        std::vector<float> p(std::size_t(raster.width) * std::size_t(raster.height) * 3), n(p.size());
        surface.resample(raster, p.data(), n.data());
        for (int j = 0; j < raster.height; ++j)
            for (int i = 0; i < raster.width; ++i) {
                const float* q = &p[(std::size_t(j) * std::size_t(raster.width) + std::size_t(i)) * 3];
                const float* m = &n[(std::size_t(j) * std::size_t(raster.width) + std::size_t(i)) * 3];
                const double gx = raster.x0 + i * raster.sx, gy = raster.y0 + j * raster.sy;
                require(near(q[0], float(2 * gx)) && near(q[1], float(3 * gy)) && near(q[2], 10.0f),
                        "flat surface position");
                // The patch normal: column direction x row direction = +z here.
                require(near(m[2], 1.0f) && near(m[0], 0.0f) && near(m[1], 0.0f), "flat surface normal");
            }
    }

    // Outside the grid, and on the cells of an invalid vertex, there is no point. A
    // vertex is invalid when its flag says so (its coordinates do not matter).
    std::vector<std::uint8_t> valid(std::size_t(rows) * std::size_t(cols), 1);
    valid[std::size_t(2) * std::size_t(cols) + std::size_t(3)] = 0;
    const vcr::grid::Vertices hv{xyz.data(), rows, cols, std::size_t(cols) * 3, nullptr, valid.data()};
    const vcr::grid::Raster probe{3.0, 2.0, 0.25, 0.25, 5, 1};  // grid x 3.0 .. 4.0 on row 2
    float out[15];
    vcr::grid::Surface nearest(hv);
    nearest.resample(probe, out, nullptr);
    for (int i = 0; i < 4; ++i)  // x 3.0 .. 3.75: the cell (3..4, 2..3) holds the missing vertex
        require(std::isnan(out[3 * i]), "a cell with an invalid vertex has no point");
    require(near(out[12], 8.0f) && near(out[13], 6.0f), "the next cell is a patch");
    for (const auto mode : {vcr::grid::Interpolation::Linear, vcr::grid::Interpolation::Bicubic}) {
        vcr::grid::Surface s(hv, {mode});
        const vcr::grid::Raster cell{1.0, 0.0, 0.1, 0.1, 41, 41};  // the cells around the hole
        std::vector<float> q(41 * 41 * 3);
        s.resample(cell, q.data(), nullptr);
        std::size_t points = 0;
        for (std::size_t i = 0; i < q.size(); i += 3) {
            points += !std::isnan(q[i]);
            require(std::isnan(q[i]) || (q[i] >= 1.999f && q[i] <= 10.001f && q[i + 1] >= -0.001f && near(q[i + 2], 10.0f)),
                    "positions are made of valid vertices only");
        }
        require(points > 0, "cells away from the hole are patches");
    }
    const vcr::grid::Raster outside{-3.0, 1.0, 1.0, 1.0, 2, 1};
    nearest.resample(outside, out, nullptr);
    require(std::isnan(out[0]) && std::isnan(out[3]), "outside the grid");
    {
        std::vector<float> nanHole = xyz;
        nanHole[(std::size_t(2) * std::size_t(cols) + std::size_t(3)) * 3 + 1] = NAN;
        vcr::grid::Surface unflagged({nanHole.data(), rows, cols, std::size_t(cols) * 3});
        unflagged.resample(probe, out, nullptr);
        require(std::isnan(out[0]) && near(out[12], 8.0f), "without flags, a NaN vertex is invalid");
    }
    // Flipped normals: the patch normal is negated, a vertex normal is not.
    {
        vcr::grid::SurfaceOptions flipped;
        flipped.flipNormals = true;
        float p[3], n[3];
        vcr::grid::Surface s(v, flipped);
        s.resample({1.5, 1.5, 1, 1, 1, 1}, p, n);
        require(near(n[2], -1.0f), "flipped patch normal");
    }

    // Anisotropic spacing: the sheet z = x (in vertex units) with voxels twice as long
    // in z is z = 2x physically, so its unit physical normal is (-2, 0, 1) / sqrt(5), and
    // one physical unit along it moves (-2, 0, 0.5) / sqrt(5) in vertex units.
    {
        std::vector<float> tilted(5 * 5 * 3);
        for (int r = 0; r < 5; ++r)
            for (int c = 0; c < 5; ++c) {
                float* q = &tilted[(std::size_t(r) * 5 + std::size_t(c)) * 3];
                q[0] = float(c);
                q[1] = float(r);
                q[2] = float(c);
            }
        vcr::grid::SurfaceOptions options;
        options.spacing = {1, 1, 2};
        vcr::grid::Surface sheet({tilted.data(), 5, 5, 15}, options);
        float p[3], n[3];
        sheet.resample({1.5, 2.5, 1, 1, 1, 1}, p, n);
        const float k = 1.0f / std::sqrt(5.0f);
        require(near(p[0], 1.5f) && near(p[1], 2.5f) && near(p[2], 1.5f), "anisotropic position");
        require(near(n[0], -2 * k) && near(n[1], 0) && near(n[2], 0.5f * k), "anisotropic normal step");
    }

    // A sheared raster: pixel (i, j) at grid column x0 + i sx + j kx, row y0 + i ky + j sy.
    {
        vcr::grid::Surface surface(v);
        const vcr::grid::Raster sheared{1.5, 0.5, 0.75, 0.5, 4, 5, 0.25, 0.125};
        std::vector<float> q(4 * 5 * 3);
        surface.resample(sheared, q.data(), nullptr);
        for (int j = 0; j < 5; ++j)
            for (int i = 0; i < 4; ++i) {
                const float* at = &q[std::size_t(j * 4 + i) * 3];
                const double gx = 1.5 + i * 0.75 + j * 0.25, gy = 0.5 + i * 0.125 + j * 0.5;
                require(near(at[0], float(2 * gx)) && near(at[1], float(3 * gy)), "sheared raster position");
            }
    }

    // Vertex normals replace the patch's, also from padded rows.
    {
        const std::size_t stride = std::size_t(cols) * 3 + 3;
        std::vector<float> padded(std::size_t(rows) * stride), normals(padded.size());
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                std::copy_n(&xyz[(std::size_t(r) * cols + c) * 3], 3, &padded[std::size_t(r) * stride + std::size_t(c) * 3]);
                float* n = &normals[std::size_t(r) * stride + std::size_t(c) * 3];
                n[0] = 0.6f;
                n[1] = 0.0f;
                n[2] = 0.8f;
            }
        vcr::grid::Surface given({padded.data(), rows, cols, stride, normals.data()});
        float p[3], n[3];
        given.resample({2.5, 1.5, 1, 1, 1, 1}, p, n);
        require(near(p[0], 5.0f) && near(p[1], 4.5f) && near(n[0], 0.6f) && near(n[1], 0.0f) && near(n[2], 0.8f),
                "vertex normals");
    }

    // Two pieces: no point between them, positions inside each from that piece.
    for (const auto mode : {vcr::grid::Interpolation::Linear, vcr::grid::Interpolation::Bicubic}) {
        vcr::grid::Surface pieces(v, {mode, {{0, 4}, {4, 8}}});
        const vcr::grid::Raster across{2.5, 1.0, 1.0, 1.0, 3, 1};  // grid x 2.5, 3.5 (the seam cell), 4.5
        pieces.resample(across, out, nullptr);
        require(near(out[0], 5.0f) && std::isnan(out[3]) && near(out[6], 9.0f), "pieces are not interpolated across");
    }

    // A cylinder of radius 40 about the y axis, a vertex every 0.05 rad: bicubic normals
    // follow the analytic normal (sin a, 0, cos a) far closer than the linear patches'
    // and change continuously across cell edges, where the linear ones step.
    {
        const int rows = 6, cols = 24;
        std::vector<float> cylinder(std::size_t(rows) * cols * 3);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                float* q = &cylinder[(std::size_t(r) * cols + std::size_t(c)) * 3];
                q[0] = 40.0f * std::sin(0.05f * float(c));
                q[1] = 3.0f * float(r);
                q[2] = 40.0f * std::cos(0.05f * float(c));
            }
        const vcr::grid::Raster line{2.0, 2.5, 0.01, 1.0, 1801, 1};  // grid x 2 .. 20 on one row
        float error[2]{}, step[2]{};
        for (const int k : {0, 1}) {
            vcr::grid::Surface surface({cylinder.data(), rows, cols, std::size_t(cols) * 3},
                                       {k ? vcr::grid::Interpolation::Bicubic : vcr::grid::Interpolation::Linear});
            std::vector<float> p(1801 * 3), n(p.size());
            surface.resample(line, p.data(), n.data());
            for (int i = 0; i < 1801; ++i) {
                const float a = 0.05f * float(2.0 + 0.01 * i);
                const float* m = &n[std::size_t(i) * 3];
                const float dot = std::abs(m[0] * std::sin(a) + m[2] * std::cos(a));
                error[k] = std::max(error[k], std::acos(std::min(1.0f, dot)));
                if (i)
                    step[k] = std::max(step[k], std::abs(m[0] - n[std::size_t(i - 1) * 3]));
            }
        }
        require(error[1] < 0.1f * error[0] && error[1] < 1e-3f, "bicubic normals follow a cylinder");
        require(step[1] < 0.1f * step[0], "bicubic normals are continuous across cells");

        // The border cells and the cells next to a hole: their one sided derivatives must
        // keep the bicubic normals within the interior error, not fall back to the linear one.
        std::vector<std::uint8_t> valid(std::size_t(rows) * cols, 1);
        valid[std::size_t(2) * cols + 12] = 0;
        vcr::grid::Surface holed({cylinder.data(), rows, cols, std::size_t(cols) * 3, nullptr, valid.data()},
                                 {vcr::grid::Interpolation::Bicubic});
        const auto worst = [&](double x0, int n, double row) {
            std::vector<float> p(std::size_t(n) * 3), m(p.size());
            holed.resample({x0, row, 0.01, 1.0, n, 1}, p.data(), m.data());
            float e = 0;
            for (int i = 0; i < n; ++i) {
                const float a = 0.05f * float(x0 + 0.01 * i), *q = &m[std::size_t(i) * 3];
                if (std::isfinite(q[0]))
                    e = std::max(e, std::acos(std::min(1.0f, std::abs(q[0] * std::sin(a) + q[2] * std::cos(a)))));
            }
            return e;
        };
        const float interior = worst(4.0, 601, 4.5), first = worst(0.0, 101, 4.5), last = worst(22.0, 101, 4.5);
        const float nearHole = worst(10.0, 401, 2.5);  // cells 10 and 13 beside the hole's cells
        require(first < 2 * interior && last < 2 * interior, "bicubic normals in border cells");
        require(nearHole < 2 * interior, "bicubic normals in cells next to a hole");
    }
}

}  // namespace

int main()
{
    namespace fs = std::filesystem;
    const auto path = fs::temp_directory_path() / ("vcr-native-grid-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); } } cleanup{path};
    try {
        testDefaultInterpolation();
        testResample();

        // A 16^3 volume with value = x + 2y + 3z (chunks 8^3) and its 8^3 half level
        // with value 200 everywhere, both local stores.
        const auto fine = path / "0", coarse = path / "1";
        fs::create_directories(fine);
        fs::create_directories(coarse);
        for (int cz = 0; cz < 2; ++cz)
            for (int cy = 0; cy < 2; ++cy)
                for (int cx = 0; cx < 2; ++cx) {
                    std::vector<std::uint8_t> chunk(512);
                    for (int z = 0; z < 8; ++z)
                        for (int y = 0; y < 8; ++y)
                            for (int x = 0; x < 8; ++x)
                                chunk[std::size_t((z * 8 + y) * 8 + x)] =
                                    std::uint8_t((cx * 8 + x) + 2 * (cy * 8 + y) + 3 * (cz * 8 + z));
                    std::ofstream o(fine / (std::to_string(cz) + "." + std::to_string(cy) + "." + std::to_string(cx)),
                                    std::ios::binary);
                    o.write(reinterpret_cast<const char*>(chunk.data()), 512);
                }
        {
            std::vector<std::uint8_t> chunk(512, 200);
            std::ofstream o(coarse / "0.0.0", std::ios::binary);
            o.write(reinterpret_cast<const char*>(chunk.data()), 512);
        }
        const auto open = [](const fs::path& root, std::int64_t n) {
            vcr::ArraySpec spec;
            spec.shape = {n, n, n};
            spec.chunks = {8, 8, 8};
            spec.fill = {0};
            spec.identity = root.string();
            spec.layers.push_back({"array", root.string(), "", "", false});
            return std::make_unique<vcr::Array>(spec, std::make_shared<vcr::MemoryCache>(1 << 20));
        };
        auto a0 = open(fine, 16), a1 = open(coarse, 8);
        const vcr::ElementType u8{'u', 1, false};
        // Scalar views into CZYX chunks, including a nonzero channel chunk.
        // Progressive loads must retain the channel axis in missing-chunk keys.
        {
            const auto packed = path / "packed";
            fs::create_directories(packed);
            for (int c = 0; c < 2; ++c) {
                std::vector<std::uint8_t> bytes(2 * 512);
                for (int channel = 0; channel < 2; ++channel)
                    for (int z = 0; z < 8; ++z)
                        for (int y = 0; y < 8; ++y)
                            for (int x = 0; x < 8; ++x)
                                bytes[channel * 512 + (z * 8 + y) * 8 + x] =
                                    std::uint8_t(50 * (c * 2 + channel) + x + 2 * y + 3 * z);
                std::ofstream out(packed / (std::to_string(c) + ".0.0.0"), std::ios::binary);
                out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            }
            vcr::ArraySpec spec;
            spec.shape = {4, 8, 8, 8}; spec.chunks = {2, 8, 8, 8};
            spec.fill = {0}; spec.identity = packed.string();
            spec.layers.push_back({"array", packed.string(), "", "", false});
            vcr::Array array(spec, std::make_shared<vcr::MemoryCache>(1 << 20));
            vcr::grid::Level level{&array, u8}; level.channel = 3;
            vcr::grid::Frame frame;
            frame.plane = vcr::grid::Plane{{2.5f, 3, 4}, {1,0,0}, {0,1,0}, {0,0,1}, 1, 1};
            vcr::grid::Request request; request.levels = {level};
            vcr::grid::Pending pending;
            require(vcr::grid::sampleCached(frame, request, &pending).level[0] == vcr::grid::Result::kMissing,
                    "packed channel initially missing");
            vcr::grid::load(pending);
            require(near(vcr::grid::sampleCached(frame, request, nullptr).values[0], 170.5f),
                    "packed nonzero channel after progressive load");
            request.levels[0].channel = 0;
            require(near(vcr::grid::sample(frame, request).values[0], 20.5f), "packed first channel");
            require(vcr::grid::fallbackLevels({level}, 0, 10, 10, 1) == 0, "packed fallback extent");
            request.levels[0].channel = 4;
            bool rejected = false;
            try { vcr::grid::sample(frame, request); } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "invalid channel rejected");
        }
        const vcr::grid::Level L0{a0.get(), u8, {1, 1, 1}, {0, 0, 0}};
        // Level-1 voxel i is centred on level-0 voxel 2i + 0.5.
        const vcr::grid::Level L1{a1.get(), u8, {0.5, 0.5, 0.5}, {-0.25, -0.25, -0.25}};
        const auto value = [](float x, float y, float z) { return x + 2 * y + 3 * z; };

        // A z plane at z = 4, x 2..9, y 3..6; layers at z = 4 and z = 5 (normal +z).
        vcr::grid::Plane plane{{2, 3, 4}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 8, 4};
        vcr::grid::Request request;
        request.levels = {L0};
        request.offsets = {0.0f, 1.0f};
        auto r = vcr::grid::sample(plane, request);
        require(r.width == 8 && r.height == 4 && r.layers == 2 && r.values.size() == 64 && r.level.size() == 32,
                "plane result shape");
        for (int k = 0; k < 2; ++k)  // values layer-major, levels per pixel
            for (int j = 0; j < 4; ++j)
                for (int i = 0; i < 8; ++i) {
                    const std::size_t q = std::size_t(j * 8 + i);
                    require(r.level[q] == 0 && near(r.values[std::size_t(k) * 32 + q], value(2.0f + i, 3.0f + j, 4.0f + k)),
                            "plane layer value");
                }

        // Composite: max over the two layers is the upper one.
        request.composite = vcr::composite::Options{vcr::composite::Method::Max};
        r = vcr::grid::sample(plane, request);
        require(r.values.size() == 32 && near(r.values[9], value(3, 4, 5)), "plane max composite");
        request.composite.reset();

        // Cache-only on a fresh array: missing at level 0, the coarse level supplies
        // nothing either (also not in memory) -> kMissing; after a blocking read of the
        // coarse level, the coarse level supplies, flagged as level 1.
        auto fresh0 = open(fine, 16), fresh1 = open(coarse, 8);
        vcr::grid::Request cached;
        cached.levels = {{fresh0.get(), u8, L0.scale, L0.offset}, {fresh1.get(), u8, L1.scale, L1.offset}};
        cached.cachedOnly = true;
        r = vcr::grid::sample(plane, cached);
        for (std::size_t q = 0; q < 32; ++q)
            require(r.level[q] == vcr::grid::Result::kMissing && std::isnan(r.values[q]), "nothing cached: missing");
        vcr::grid::Request load = cached;
        load.cachedOnly = false;
        load.levels = {cached.levels[1]};
        vcr::grid::sample(plane, load);
        r = vcr::grid::sample(plane, cached);
        for (std::size_t q = 0; q < 32; ++q)
            require(r.level[q] == 1 && near(r.values[q], 200.0f), "coarse level fills while the fine one is missing");
        load.levels = {cached.levels[0]};
        std::vector<std::uint8_t> only(32, 0);
        only[5] = 1;
        load.pixels = only.data();
        vcr::grid::sample(plane, load);  // loads the chunk holding pixel 5 (all of this plane's level-0 chunk)
        r = vcr::grid::sample(plane, cached);
        require(r.level[5] == 0 && near(r.values[5], value(7, 3, 4)), "the loaded fine chunk supplies");

        // Three levels, only the last in memory: points fall through two levels to it,
        // and a masked read over two levels resolves from the second.
        auto first = open(fine, 16), second = open(fine, 16), third = open(coarse, 8);
        third->sample_nearest(std::vector<double>{0, 0, 0}.data(), 1, std::vector<std::uint8_t>(1).data());
        vcr::grid::Request three = cached;
        three.levels = {{first.get(), u8, L0.scale, L0.offset},
                        {second.get(), u8, L0.scale, L0.offset},
                        {third.get(), u8, L1.scale, L1.offset}};
        r = vcr::grid::sample(plane, three);
        for (std::size_t q = 0; q < 32; ++q)
            require(r.level[q] == 2 && near(r.values[q], 200.0f), "third level fills both missing levels");
        three.levels.erase(three.levels.begin());
        three.pixels = only.data();
        r = vcr::grid::sample(plane, three);
        require(r.level[5] == 1 && r.level[4] == vcr::grid::Result::kNone, "masked read across two levels");

        // Pixel mask: other pixels are not sampled.
        vcr::grid::Request masked = request;
        masked.pixels = only.data();
        r = vcr::grid::sample(plane, masked);
        require(r.level[5] == 0 && r.level[4] == vcr::grid::Result::kNone, "pixel mask");

        // A plane leaving the volume: points outside read as fill, as everywhere.
        vcr::grid::Plane edge{{14, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 4, 1};
        request.offsets = {0.0f};
        r = vcr::grid::sample(edge, request);
        require(r.level[1] == 0 && near(r.values[1], value(15, 0, 0)) && r.level[3] == 0 && r.values[3] == 0.0f,
                "outside the volume reads as fill");
        // Linear sampling there matches Array::sample_linear, fill neighbours included.
        request.order = 1;
        vcr::grid::Plane between{{14.5f, 3.25f, 4.5f}, {0.5f, 0, 0}, {0, 0.5f, 0}, {0, 0, 1}, 6, 3};
        r = vcr::grid::sample(between, request);
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 6; ++i) {
                const double at[3]{4.5, 3.25 + 0.5 * j, 14.5 + 0.5 * i};
                float expected = 0;
                a0->sample_linear(at, 1, u8, false, &expected);
                require(near(r.values[std::size_t(j * 6 + i)], expected, 1e-3f), "linear matches Array::sample_linear");
            }
        // A general level map (as from a rotated registration):
        // index (z, y, x) = (p.x + 0.25, 0.5 p.y + 0.5 p.z + 1, p.z - 0.5).
        {
            vcr::grid::Request general;
            general.levels = {{a0.get(), u8, {}, {0.25, 1, -0.5}, std::array<double, 9>{1, 0, 0, 0, 0.5, 0.5, 0, 0, 1}}};
            general.offsets = {0.0f, 1.5f};
            r = vcr::grid::sample(between, general);
            for (int k = 0; k < 2; ++k)
                for (int j = 0; j < 3; ++j)
                    for (int i = 0; i < 6; ++i) {
                        const double x = 14.5 + 0.5 * i, y = 3.25 + 0.5 * j, z = 4.5 + 1.5 * k;
                        const double at[3]{x + 0.25, 0.5 * y + 0.5 * z + 1, z - 0.5};
                        float expected = 0;
                        a0->sample_linear(at, 1, u8, false, &expected);
                        require(near(r.values[std::size_t(k * 18 + j * 6 + i)], expected, 1e-3f),
                                "a general level map matches Array::sample_linear");
                    }
        }
        // A pixel whose layers come from two levels reports the second; a missing layer wins.
        auto partial0 = open(fine, 16), partial1 = open(coarse, 8);
        partial0->chunk({0, 0, 0});  // z 0..7 at level 0 in memory
        partial1->chunk({0, 0, 0});
        vcr::grid::Request mixed;
        mixed.levels = {{partial0.get(), u8, L0.scale, L0.offset}, {partial1.get(), u8, L1.scale, L1.offset}};
        mixed.cachedOnly = true;
        mixed.offsets = {0.0f, 6.0f};  // z 2 and z 8: level 0, then only level 1 holds it
        vcr::grid::Plane corner{{1, 1, 2}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 2, 1};
        r = vcr::grid::sample(corner, mixed);
        require(r.level[0] == 1 && near(r.values[0], value(1, 1, 2)) && near(r.values[2], 200.0f), "levels mix per layer");
        mixed.levels.pop_back();
        r = vcr::grid::sample(corner, mixed);
        require(r.level[0] == vcr::grid::Result::kMissing && near(r.values[0], value(1, 1, 2)), "a missing layer wins");

        // Along a surface: the flat grid lies at z = 10 with x = 2c, y = 3r; sampled
        // linearly between voxels, and one layer down the normal.
        auto xyz = flatGrid(4, 6);
        vcr::grid::Surface surface({xyz.data(), 4, 6, 18}, {vcr::grid::Interpolation::Bicubic});
        const vcr::grid::Raster raster{0.5, 0.5, 0.25, 0.25, 8, 8};
        request.offsets = {0.0f, 1.0f};
        r = vcr::grid::sample(surface, raster, request);
        std::vector<float> pts(64 * 3), nrm(64 * 3);
        surface.resample(raster, pts.data(), nrm.data());
        for (std::size_t q = 0; q < 64; ++q) {
            const double at[3]{pts[q * 3 + 2], pts[q * 3 + 1], pts[q * 3]};
            float expected = 0;
            a0->sample_linear(at, 1, u8, false, &expected);
            require(r.level[q] == 0 && near(r.values[q], expected, 1e-3f), "surface value");
            const double below[3]{pts[q * 3 + 2] + nrm[q * 3 + 2], pts[q * 3 + 1] + nrm[q * 3 + 1],
                                  pts[q * 3] + nrm[q * 3]};
            a0->sample_linear(below, 1, u8, false, &expected);
            require(near(r.values[64 + q], expected, 1e-3f), "surface layer along the normal");
        }

        // Views: level choice by zoom, magnification over a surface raster.
        {
            const auto e1 = vcr::grid::voxelExtent(L1);
            require(e1 == std::array<double, 3>{2, 2, 2}, "level 1 voxels span two working voxels");
            const vcr::grid::Level sheared{a0.get(), u8, {}, {0, 0, 0}, std::array<double, 9>{0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0}};
            require(vcr::grid::voxelExtent(sheared) == std::array<double, 3>{2, 2, 2}, "extent of a general level map");
            const std::vector<vcr::grid::Level> both{L0, L1}, onlyCoarse{vcr::grid::Level{}, L1};
            require(vcr::grid::levelForView(both, 1.0f) == 0 && vcr::grid::levelForView(both, 0.25f) == 1 &&
                        vcr::grid::levelForView(both, 8.0f) == 0 && vcr::grid::levelForView(both, NAN) == 0,
                    "level for a view's zoom");
            require(vcr::grid::levelForView(onlyCoarse, 1.0f) == 1, "absent levels are skipped");
            // A 16-pixel view at 1 px/voxel spans 16 voxels: only a level-1 chunk (16) covers it.
            require(vcr::grid::fallbackLevels(both, 0, 16, 16, 1.0f) == 1 &&
                        vcr::grid::fallbackLevels(both, 0, 16, 16, 4.0f) == 0 &&
                        vcr::grid::fallbackLevels(both, 0, 16, 16, 1.0f, 0) == 0,
                    "fallback levels up to one spanning the view");
            // Vertices every 2 voxels in x and 3 in y; 4 pixels per vertex: x is the most
            // magnified direction, 2 pixels per voxel.
            auto grid = flatGrid(4, 6);
            const vcr::grid::Surface flat({grid.data(), 4, 6, 18});
            require(near(vcr::grid::pixelsPerVoxel(flat, {0.5, 0.5, 0.25, 0.25, 8, 8}), 2.0f), "pixels per voxel");
            require(vcr::grid::pixelsPerVoxel(flat, {50, 50, 0.25, 0.25, 8, 8}) == 0.0f, "no surface under the view");
        }

        // Progressive frames: drawn from the caches alone; the reads they ask for complete them.
        {
            auto p0 = open(fine, 16), p1 = open(coarse, 8);
            vcr::grid::Frame frame;
            frame.plane = plane;
            vcr::grid::Request view;
            view.levels = {{p0.get(), u8, L0.scale, L0.offset}, {p1.get(), u8, L1.scale, L1.offset}};
            vcr::grid::Pending pending;
            r = vcr::grid::sampleCached(frame, view, &pending);
            for (std::size_t q = 0; q < 32; ++q)
                require(r.level[q] == vcr::grid::Result::kMissing, "an uncached frame is missing everywhere");
            require(p0->stats().disk_hits == 0 && pending.coarse.size() == 1 && pending.target.size() == 1,
                    "the frame read nothing and asks for a preview and its level");
            require(r.missingChunks.size() == 2 && !r.missingChunks[0].empty() &&
                        pending.target[0].chunks == r.missingChunks[0] && pending.coarse[0].chunks == r.missingChunks[1],
                    "a CPU frame names the chunks its reads fetch");
            // Cancelled: no chunk read starts, in either part or in a blocking sample.
            const std::atomic<bool> cancelled{true};
            int parts = 0;
            vcr::grid::load(pending, [&] { ++parts; return true; }, &cancelled);
            require(parts == 1 && p0->stats().disk_hits == 0 && p1->stats().disk_hits == 0, "a cancelled load reads nothing");
            vcr::grid::Request blocking = view;
            blocking.cancel = &cancelled;
            bool threw = false;
            try { vcr::grid::sample(frame, blocking); } catch (const vcr::Cancelled&) { threw = true; }
            require(threw && p0->stats().disk_hits == 0, "a cancelled blocking sample throws Cancelled");
            parts = 0;
            require(vcr::grid::load(pending, [&] { ++parts; return false; }).empty() && parts == 1, "load stops when told");
            vcr::grid::Pending rest;
            r = vcr::grid::sampleCached(frame, view, &rest);
            for (std::size_t q = 0; q < 32; ++q)
                require(r.level[q] == 1 && near(r.values[q], 200.0f), "the preview fills the frame");
            require(rest.coarse.empty() && rest.target.size() == 1, "only the view's level is still asked for");
            const auto readsBefore = p0->stats().disk_hits.load();
            vcr::grid::load(rest);
            require(p0->stats().disk_hits.load() - readsBefore == rest.target[0].chunks.size(), "each lacking chunk is read once");
            vcr::grid::Pending none;
            r = vcr::grid::sampleCached(frame, view, &none);
            require(none.empty() && r.level[9] == 0 && near(r.values[9], value(3, 4, 4)), "the loaded frame is complete");
            // A surface frame keeps its surface alive.
            auto xyz = flatGrid(4, 6);
            vcr::grid::Frame along;
            along.surface = vcr::grid::Surface::snapshot({xyz.data(), 4, 6, 18});
            along.raster = {0.5, 0.5, 0.25, 0.25, 8, 8};
            xyz.assign(xyz.size(), NAN);
            vcr::grid::Pending more;
            vcr::grid::sampleCached(along, view, &more);
            vcr::grid::load(more);
            r = vcr::grid::sampleCached(along, view, nullptr);
            require(r.level[0] == 0 && near(r.values[0], value(1, 1.5f, 10), 1e-3f), "a surface frame completes too");
        }

        // Placement maps: scale form when axis aligned, otherwise general.
        const auto scaled = vcr::grid::levelFromMap(*a0, u8, {0, 0, 2, 1, 0, 3, 0, 2, 4, 0, 0, 3});
        require(!scaled.linear && scaled.scale == std::array<double, 3>{2, 3, 4} &&
                    scaled.offset == std::array<double, 3>{1, 2, 3},
                "axis-aligned level map");
        const auto general = vcr::grid::levelFromMap(*a0, u8, {1, 0, 2, 1, 0, 3, 0, 2, 4, 0, 0, 3});
        require(general.linear && (*general.linear)[0] == 1 && (*general.linear)[8] == 0, "general level map");
        bool rejected = false;
        try {
            vcr::grid::levelFromMap(*a0, u8, {NAN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "nonfinite level map");
        std::cout << "Native grid surfaces and sampling passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
