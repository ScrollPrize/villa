#include "../../src/cpp/sheet_normal/normal.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

// The CPU estimator must give the same bits as its plain formulation below: one
// separable filter channel at a time, every tap, and every weight computed per
// site. Both are compiled in this file with the same flags, so this holds for
// any floating-point contraction the build uses.
namespace reference {
using namespace vcr::sheet_normal;

void prepare(const uint8_t* src, std::vector<float>& g, float& floor)
{
    Kernels k;
    std::vector<float> z(4 * zcount), y(6 * ycount), mags(count);
    g.assign(6 * count, 0.f);
    for (int c = 0; c < 4; ++c) for (int zz = 0; zz < sites; ++zz) {
        float* dst = z.data() + c * zcount + zz * width * width;
        std::fill_n(dst, width * width, 0.f);
        for (int j = 0; j < 13; ++j) {
            const uint8_t* plane = src + (3 * zz + j) * width * width;
            const float w = k.taps[c][j];
            for (int i = 0; i < width * width; ++i) dst[i] += w * (float(plane[i]) * (1.f / 255.f));
        }
    }
    for (int c = 0; c < 6; ++c) for (int zz = 0; zz < sites; ++zz) for (int yy = 0; yy < sites; ++yy) {
        int scale = c / 3, axis = c % 3, zc = 2 * scale + (axis == 2), yc = 2 * scale + (axis == 1);
        float* dst = y.data() + c * ycount + (zz * sites + yy) * width;
        std::fill_n(dst, width, 0.f);
        for (int j = 0; j < 13; ++j) {
            const float* row = z.data() + zc * zcount + (zz * width + 3 * yy + j) * width;
            for (int xx = 0; xx < width; ++xx) dst[xx] += k.taps[yc][j] * row[xx];
        }
    }
    for (int c = 0; c < 6; ++c) for (int i = 0; i < count; ++i) {
        int xc = 2 * (c / 3) + (c % 3 == 0), x = i % sites, row = i / sites;
        float value = 0;
        for (int j = 0; j < 13; ++j) value += k.taps[xc][j] * y[c * ycount + row * width + 3 * x + j];
        g[c * count + i] = value;
    }
    for (int i = 0; i < count; ++i) mags[i] = magnitude(g[3 * count + i], g[4 * count + i], g[5 * count + i]);
    std::nth_element(mags.begin(), mags.begin() + count / 2, mags.end());
    floor = std::max(mags[count / 2], 1e-8f);
}

void evaluate(const std::vector<float>& g, float floor, const float* center, float* output)
{
    std::fill_n(output, 4, 0.f);
    if (!center[3]) return;
    double normal[3]{}, gap = 0;
    bool valid = true;
    for (int pass = 0; pass < 4 && valid; ++pass) {
        double sums[7]{};
        for (int i = 0; i < count; ++i)
            contribution(i, g.data() + (pass ? 3 * count : 0), center, normal, floor, pass != 0, sums);
        valid = principal(sums, normal, gap);
    }
    if (valid) { for (int j = 0; j < 3; ++j) output[j] = float(normal[j]); output[3] = float(gap); }
}
} // namespace reference

int main()
{
    using namespace vcr::sheet_normal;
    const std::size_t n = std::size_t(width) * width * width;
    std::vector<uint8_t> image(n);
    const float centers[][4] = {{48, 48, 48, 1}, {48.25f, 47.5f, 49, 1}, {50.9f, 48.1f, 49.7f, 1},
                                {49, 48, 47, 0}, {48, 48, 48, 1}};
    CpuWorkspace workspace;  // reused across images, as callers do
    std::uint32_t lcg = 12345;
    int checked = 0;
    for (int kind = 0; kind < 5; ++kind) {
        for (int z = 0; z < width; ++z)
            for (int y = 0; y < width; ++y)
                for (int x = 0; x < width; ++x) {
                    uint8_t& v = image[(std::size_t(z) * width + y) * width + x];
                    lcg = lcg * 1664525u + 1013904223u;
                    switch (kind) {
                    case 0: v = static_cast<uint8_t>(120 + 70 * std::sin(.3 * (x + .5 * y + .2 * z))); break;
                    case 1: v = static_cast<uint8_t>(lcg >> 24); break;  // noise
                    case 2: v = 43; break;                               // flat: no normal
                    case 3: v = (std::abs(z - 54 + (x - 54) / 4) < 5) ? 220 : 0; break;  // one sheet, zero background
                    default: v = static_cast<uint8_t>(std::max(0.0, 255 * std::sin(.21 * (z + .3 * x)) + (lcg >> 28)));
                    }
                }
        std::vector<float> g;
        float floor = 0;
        reference::prepare(image.data(), g, floor);
        CpuPatch patch;
        workspace.prepare(image.data(), patch);
        if (std::memcmp(patch.gradients.get(), g.data(), 6 * count * sizeof(float)) != 0 ||
            std::memcmp(&patch.floor, &floor, sizeof floor) != 0) {
            std::cerr << "image " << kind << ": filtered gradients differ from the reference\n";
            return 1;
        }
        for (const auto& center : centers) {
            float got[4], want[4];
            patch.evaluate(center, got);
            reference::evaluate(g, floor, center, want);
            if (std::memcmp(got, want, sizeof got) != 0) {
                std::cerr << "image " << kind << ": normal differs from the reference at center "
                          << center[0] << ' ' << center[1] << ' ' << center[2] << '\n';
                return 2;
            }
            ++checked;
        }
    }
    std::cout << checked << " normals equal to the reference, bit for bit\n";
    return 0;
}
