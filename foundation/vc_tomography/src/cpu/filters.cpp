#include <algorithm>
#include <vector>

#include "helirec/filters.hpp"

namespace helirec::cpu {

// (2r+1)^2 median per frame, coordinates clamped at the edges
void median_frames(float* frames, int n_frames, int H, int W, int r) {
    const std::size_t plane = static_cast<std::size_t>(H) * W;
#pragma omp parallel
    {
        std::vector<float> win((2 * r + 1) * (2 * r + 1)), src(plane);
#pragma omp for schedule(static)
        for (int f = 0; f < n_frames; ++f) {
            float* fr = frames + f * plane;
            std::copy(fr, fr + plane, src.begin());
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    int k = 0;
                    for (int dy = -r; dy <= r; ++dy) {
                        const int yy = std::clamp(y + dy, 0, H - 1);
                        for (int dx = -r; dx <= r; ++dx) win[k++] = src[static_cast<std::size_t>(yy) * W + std::clamp(x + dx, 0, W - 1)];
                    }
                    std::nth_element(win.begin(), win.begin() + k / 2, win.begin() + k);
                    fr[static_cast<std::size_t>(y) * W + x] = win[k / 2];
                }
        }
    }
}

}  // namespace helirec::cpu
