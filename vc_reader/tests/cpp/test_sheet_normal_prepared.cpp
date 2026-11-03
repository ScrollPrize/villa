#include "../../src/cpp/sheet_normal/normal.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

// Reusing one prepared patch must not retain a previous query's position or
// validity, and replacing the image must replace all gradients and the floor.
int main()
{
    using namespace vcr::sheet_normal;
    std::vector<uint8_t> image(width * width * width);
    CpuWorkspace workspace;
    CpuPatch patch;
    const float centers[][4] = {{48,48,48,1}, {48.25f,47.5f,49,1},
                                {49,48,47,0}, {47,49.25f,48.5f,1}, {48,48,48,1}};
    for (int flat = 0; flat < 2; ++flat) {
        for (int z = 0; z < width; ++z)
            for (int y = 0; y < width; ++y)
                for (int x = 0; x < width; ++x)
                    image[(z * width + y) * width + x] = flat ? 43 :
                        static_cast<uint8_t>(120 + 70 * std::sin(.3 * (x + .5*y + .2*z)));
        workspace.prepare(image.data(), patch);
        for (const auto& center : centers) {
            float prepared[4], direct[4];
            patch.evaluate(center, prepared);
            cpu(image.data(), center, direct, 1);
            if (std::memcmp(prepared, direct, sizeof direct) != 0) {
                std::cerr << "Prepared patch differs from direct evaluation\n";
                return 1;
            }
            if ((flat || !center[3]) && prepared[3] != 0) return 2;
            if (!flat && center[3] && prepared[3] <= 0) return 3;
        }
    }
}
