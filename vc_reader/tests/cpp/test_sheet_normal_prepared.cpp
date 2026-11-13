#include "../../src/cpp/sheet_normal/normal.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

// The four passes as the CUDA kernel runs them, one contribution() per site.
static void reference(const vcr::sheet_normal::CpuPatch& patch, const float* center, float* out)
{
    using namespace vcr::sheet_normal;
    std::fill_n(out, 4, 0.f);
    if (!center[3]) return;
    double normal[3]{}, gap = 0;
    bool valid = true;
    for (int pass = 0; pass < 4 && valid; ++pass) {
        double sums[7]{};
        for (int i = 0; i < count; ++i)
            contribution(i, patch.gradients.get() + (pass ? 3 * count : 0), center, normal,
                         patch.floor, pass != 0, sums);
        valid = principal(sums, normal, gap);
    }
    if (valid) { for (int j = 0; j < 3; ++j) out[j] = float(normal[j]); out[3] = float(gap); }
}

// Reusing one prepared patch must not retain a previous query's position or
// validity, and replacing the image must replace all gradients and the floor.
// The CPU evaluation must equal the per-site contribution() loop bit for bit.
int main()
{
    using namespace vcr::sheet_normal;
    std::vector<uint8_t> image(width * width * width);
    CpuWorkspace workspace;
    CpuPatch patch;
    // Consecutive centers differ in one coordinate only (z, y, then x), and repeat.
    const float centers[][4] = {{48,48,48,1}, {48.25f,47.5f,49,1}, {48.25f,47.5f,49.5f,1},
                                {48.25f,48,49.5f,1}, {48.75f,48,49.5f,1}, {49,48,47,0},
                                {47,49.25f,48.5f,1}, {48,48,48,1}, {48,48,48,1}};
    // Textured, flat, then textured along another direction with a dark half.
    for (int pattern = 0; pattern < 3; ++pattern) {
        const bool flat = pattern == 1;
        for (int z = 0; z < width; ++z)
            for (int y = 0; y < width; ++y)
                for (int x = 0; x < width; ++x)
                    image[(z * width + y) * width + x] = flat ? 43 : pattern && x < 60 ? 0 :
                        static_cast<uint8_t>(120 + 70 * std::sin(pattern ? .2 * (.3*x + y - .4*z)
                                                                         : .3 * (x + .5*y + .2*z)));
        workspace.prepare(image.data(), patch);
        for (const auto& center : centers) {
            float prepared[4], direct[4];
            patch.evaluate(center, prepared);
            // cpu() only accepts centers in [48,51); a fresh patch covers the rest.
            CpuWorkspace freshWorkspace;
            CpuPatch fresh;
            freshWorkspace.prepare(image.data(), fresh);
            fresh.evaluate(center, direct);
            if (std::memcmp(prepared, direct, sizeof direct) != 0) {
                std::cerr << "Prepared patch differs from direct evaluation\n";
                return 1;
            }
            float expected[4];
            reference(patch, center, expected);
            if (std::memcmp(prepared, expected, sizeof expected) != 0) {
                std::cerr << "CPU evaluation differs from contribution()\n";
                return 4;
            }
            if ((flat || !center[3]) && prepared[3] != 0) return 2;
            if (!flat && center[3] && prepared[3] <= 0) return 3;
        }
    }
}
