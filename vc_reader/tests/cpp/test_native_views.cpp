#include <vc_reader/views.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>
#ifdef VC_READER_CUDA
#include <cuda_runtime.h>
#endif
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(#x) + " at line " + std::to_string(__LINE__)); } while(false)
using namespace vcr::views;
template<class F> void rejects(F f) { bool caught = false; try { f(); } catch (const std::invalid_argument&) { caught = true; } CHECK(caught); }

#ifdef VC_READER_CUDA
template<class T> struct DeviceVector {
    T* ptr = nullptr; std::size_t n;
    explicit DeviceVector(std::size_t count) : n(count) { CHECK(cudaMalloc(&ptr, std::max<std::size_t>(1, n) * sizeof(T)) == cudaSuccess); }
    DeviceVector(const std::vector<T>& v) : DeviceVector(v.size()) { CHECK(cudaMemcpy(ptr, v.data(), n * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess); }
    ~DeviceVector() { cudaFree(ptr); }
    std::vector<T> host() const { std::vector<T> v(n); CHECK(cudaDeviceSynchronize() == cudaSuccess); CHECK(cudaMemcpy(v.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess); return v; }
};
#endif

// Oriented crop coordinates by explicit transpose/flip of a coordinate array.
std::vector<Index3> orientedCrop(const Index3& origin, const Index3& n, int id)
{
    const auto o = orientation(id);
    const Index3 m{n[o.perm[0]], n[o.perm[1]], n[o.perm[2]]};
    std::vector<Index3> out(std::size_t(m[0] * m[1] * m[2]));
    for (std::int64_t i0 = 0; i0 < m[0]; ++i0) for (std::int64_t i1 = 0; i1 < m[1]; ++i1) for (std::int64_t i2 = 0; i2 < m[2]; ++i2) {
        const std::int64_t i[3]{i0, i1, i2};
        Index3 u{};
        for (int k = 0; k < 3; ++k) u[o.perm[k]] = o.flips[k] ? n[o.perm[k]] - 1 - i[k] : i[k];
        out[std::size_t((i0 * m[1] + i1) * m[2] + i2)] = {origin[0] + u[0], origin[1] + u[1], origin[2] + u[2]};
    }
    return out;
}

void verify(bool gpu)
{
    std::mt19937_64 random(91);
    for (int id = 0; id < ORIENTATION_COUNT; ++id) CHECK(orientationId(orientation(id)) == id);
    rejects([] { orientation(48); });
    rejects([] { orientationId({{0, 0, 1}, {}}); });
    {
        std::vector<std::uint64_t> seeds(4800);
        for (std::size_t i = 0; i < seeds.size(); ++i) seeds[i] = i * 7919;
        std::vector<std::int32_t> ids(seeds.size()), again(1);
        drawOrientations(seeds.data(), seeds.size(), ids.data());
        std::vector<int> hits(48);
        for (auto id : ids) { CHECK(id >= 0 && id < 48); ++hits[std::size_t(id)]; }
        for (auto h : hits) CHECK(h > 50 && h < 160);
        drawOrientations(&seeds[17], 1, again.data());
        CHECK(again[0] == ids[17]);
    }
    // Token support.
    {
        const Index3 shape{6, 4, 8}, patch{3, 2, 4};
        std::vector<std::uint8_t> valid(2 * 6 * 4 * 8), support(2 * 2 * 2 * 2), expected(support.size(), 1);
        for (auto& v : valid) v = random() % 23 ? 1 : 0;
        for (std::size_t b = 0; b < 2; ++b) for (int z = 0; z < 6; ++z) for (int y = 0; y < 4; ++y) for (int x = 0; x < 8; ++x)
            if (!valid[((b * 6 + z) * 4 + y) * 8 + x]) expected[((b * 2 + z / 3) * 2 + y / 2) * 2 + x / 4] = 0;
        tokenSupport(valid.data(), 2, shape, patch, support.data());
        CHECK(support == expected);
        rejects([&] { tokenSupport(valid.data(), 2, {6, 4, 7}, patch, support.data()); });
#ifdef VC_READER_CUDA
        if (gpu) {
            DeviceVector<std::uint8_t> v(valid), s(support.size());
            cuda::tokenSupport(v.ptr, 2, shape, patch, s.ptr);
            CHECK(s.host() == expected);
        }
#endif
    }
    // Correspondence against explicit oriented coordinate crops.
    {
        const Index3 grid{7, 6, 8}, ns{3, 4, 2}, nt{5, 3, 4};
        std::vector<std::uint8_t> support(2 * 7 * 6 * 8);
        for (auto& v : support) v = random() % 5 ? 1 : 0;
        const std::size_t count = 400;
        std::vector<std::int64_t> index(count), so(3 * count), to(3 * count), sr(count), tr(count);
        for (std::size_t p = 0; p < count; ++p) {
            index[p] = std::int64_t(random() % 2);
            sr[p] = std::int64_t(random() % 48); tr[p] = std::int64_t(random() % 48);
            for (int a = 0; a < 3; ++a) {
                so[3 * p + a] = std::int64_t(random() % std::uint64_t(grid[a] - ns[a] + 3)) - 1;
                to[3 * p + a] = std::int64_t(random() % std::uint64_t(grid[a] - nt[a] + 1));
            }
        }
        const Supports grids{support.data(), 2, grid};
        const Pairs pairs{count, ns, nt, index.data(), so.data(), sr.data(), to.data(), tr.data()};
        const std::size_t tokens = 24;
        std::vector<std::int32_t> ti(count * tokens), lat(3 * count * tokens);
        std::vector<std::uint8_t> sv(count * tokens);
        correspondence(grids, pairs, ti.data(), sv.data(), lat.data());
        for (std::size_t p = 0; p < count; ++p) {
            const auto student = orientedCrop({so[3 * p], so[3 * p + 1], so[3 * p + 2]}, ns, int(sr[p]));
            const auto teacher = orientedCrop({to[3 * p], to[3 * p + 1], to[3 * p + 2]}, nt, int(tr[p]));
            for (std::size_t i = 0; i < tokens; ++i) {
                const auto g = student[i];
                bool inGrid = true;
                for (int a = 0; a < 3; ++a) inGrid = inGrid && g[a] >= 0 && g[a] < grid[a];
                const bool valid = inGrid && support[std::size_t(index[p] * 336 + (g[0] * 6 + g[1]) * 8 + g[2])];
                const auto hit = std::find(teacher.begin(), teacher.end(), g);
                const std::int32_t expected = valid && hit != teacher.end() ? std::int32_t(hit - teacher.begin()) : -1;
                CHECK(ti[p * tokens + i] == expected);
                CHECK(sv[p * tokens + i] == valid);
                for (int a = 0; a < 3; ++a) CHECK(lat[3 * (p * tokens + i) + a] == g[a]);
            }
        }
#ifdef VC_READER_CUDA
        if (gpu) {
            DeviceVector<std::uint8_t> s(support), v(count * tokens);
            DeviceVector<std::int32_t> t(count * tokens), l(3 * count * tokens);
            cuda::correspondence({s.ptr, 2, grid}, pairs, t.ptr, v.ptr, l.ptr);
            CHECK(t.host() == ti); CHECK(v.host() == sv); CHECK(l.host() == lat);
        }
#endif
        std::vector<std::int64_t> bad(sr);
        bad[3] = 48;
        rejects([&] { auto q = pairs; q.studentOrientations = bad.data(); correspondence(grids, q, ti.data(), sv.data()); });
        bad = index; bad[5] = 2;
        rejects([&] { auto q = pairs; q.supportIndex = bad.data(); correspondence(grids, q, ti.data(), sv.data()); });
    }
    // Placement.
    {
        const Index3 tokens{12, 10, 14}, teacher{8, 8, 8};
        const std::vector<Index3> students{{8, 8, 8}, {4, 4, 4}, {4, 4, 4}};
        std::vector<std::uint8_t> support(std::size_t(12 * 10 * 14));
        for (auto& v : support) v = random() % 4 ? 1 : 0;
        const PlanOptions options{64, 0.4, 5, 32, 16, 1234};
        const auto a = planViewSets(support.data(), tokens, teacher, students, options);
        const auto b = planViewSets(support.data(), tokens, teacher, students, options);
        CHECK(a.ok && a.studentOrigins == b.studentOrigins && a.teacherOrigins == b.teacherOrigins && a.rejections == b.rejections);
        for (std::size_t s = 0; s < 64; ++s) for (std::size_t k = 0; k < 3; ++k) {
            const auto* o = &a.studentOrigins[3 * (s * 3 + k)];
            const auto* t = &a.teacherOrigins[3 * s];
            std::int64_t overlap = 0;
            for (std::int64_t z = 0; z < 12; ++z) for (std::int64_t y = 0; y < 10; ++y) for (std::int64_t x = 0; x < 14; ++x) {
                const std::int64_t c[3]{z, y, x};
                bool in = true;
                for (int d = 0; d < 3; ++d) in = in && c[d] >= o[d] && c[d] < o[d] + students[k][d] && c[d] >= t[d] && c[d] < t[d] + 8;
                overlap += in && support[std::size_t((z * 10 + y) * 14 + x)];
            }
            CHECK(overlap == a.studentOverlap[s * 3 + k]);
            CHECK(overlap >= 5 && double(overlap) >= 0.4 * double(students[k][0] * students[k][1] * students[k][2]));
            for (int d = 0; d < 3; ++d) CHECK(o[d] >= 0 && o[d] + students[k][d] <= tokens[d]);
        }
        std::vector<std::uint8_t> empty(support.size(), 0);
        const auto none = planViewSets(empty.data(), tokens, teacher, students, options);
        CHECK(!none.ok && none.teacherAttempts[0] == 16 && none.rejections[0] == 16 * 32);
        rejects([&] { planViewSets(support.data(), tokens, {13, 8, 8}, students, options); });
        rejects([&] { planViewSets(support.data(), tokens, teacher, {{0, 4, 4}}, options); });
    }
    // Masks.
    {
        const Index3 shape{6, 7, 8};
        std::vector<std::uint64_t> seeds(40);
        for (std::size_t i = 0; i < seeds.size(); ++i) seeds[i] = random();
        for (auto mode : {MaskMode::Random, MaskMode::Multiblock}) {
            MaskOptions o; o.mode = mode; o.probability = 0.7; o.fraction = 0.45; o.maxBlockTokens = 168;
            std::vector<std::uint8_t> masks(40 * 336), one(336);
            bool applied[40]; std::int64_t counts[40];
            patchMasks(shape, seeds.data(), 40, o, masks.data(), applied, counts);
            for (std::size_t b = 0; b < 40; ++b) {
                std::int64_t sum = 0;
                for (std::size_t i = 0; i < 336; ++i) sum += masks[b * 336 + i];
                CHECK(sum == counts[b] && counts[b] == (applied[b] ? 151 : 0));
                patchMasks(shape, &seeds[b], 1, o, one.data());
                CHECK(std::equal(one.begin(), one.end(), masks.begin() + std::ptrdiff_t(b * 336)));
            }
        }
    }
    // Masked cells.
    {
        const Index3 shape{4, 4, 6}, patch{2, 2, 3};
        const std::size_t batch = 2, channels = 3, n = batch * channels * 96;
        std::vector<float> images(n), out(n);
        for (auto& v : images) v = float(random() % 1000) / 7.f;
        std::vector<std::uint8_t> masks(batch * 8);
        for (auto& v : masks) v = random() % 2;
        maskPatchCells({images.data(), out.data(), Real::Float32, batch, channels, shape, patch}, masks.data(), -2.5f);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t v = i % 96, b = i / 96 / channels, z = v / 24, y = v / 6 % 4, x = v % 6;
            const bool masked = masks[b * 8 + (z / 2 * 2 + y / 2) * 2 + x / 3];
            CHECK(out[i] == (masked ? -2.5f : images[i]));
        }
        auto inplace = images;
        maskPatchCells({inplace.data(), inplace.data(), Real::Float32, batch, channels, shape, patch}, masks.data(), -2.5f);
        CHECK(inplace == out);
        std::vector<std::uint16_t> half(n, 7), halfOut(n);
        maskPatchCells({half.data(), halfOut.data(), Real::Float16, batch, channels, shape, patch}, masks.data(), 0.1f);
        maskPatchCells({half.data(), half.data(), Real::BFloat16, batch, channels, shape, patch}, masks.data(), 0.1f);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t v = i % 96, b = i / 96 / channels, z = v / 24, y = v / 6 % 4, x = v % 6;
            const bool masked = masks[b * 8 + (z / 2 * 2 + y / 2) * 2 + x / 3];
            CHECK(halfOut[i] == (masked ? 0x2e66 : 7));
            CHECK(half[i] == (masked ? 0x3dcd : 7));
        }
#ifdef VC_READER_CUDA
        if (gpu) {
            DeviceVector<float> in(images), o(n);
            DeviceVector<std::uint8_t> m(masks);
            cuda::maskPatchCells({in.ptr, o.ptr, Real::Float32, batch, channels, shape, patch}, m.ptr, -2.5f);
            CHECK(o.host() == out);
            cuda::maskPatchCells({in.ptr, in.ptr, Real::Float32, batch, channels, shape, patch}, m.ptr, -2.5f);
            CHECK(in.host() == out);
        }
#endif
    }
}

int main(int argc, char**)
{
    bool gpu = argc > 1;
#ifdef VC_READER_CUDA
    if (gpu) { int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77; }
#else
    if (gpu) return 77;
#endif
    verify(gpu);
    std::cout << "views passed\n";
}
