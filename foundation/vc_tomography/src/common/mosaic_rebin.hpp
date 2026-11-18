#pragma once
#include "interp_core.hpp"
#include "vc_tomography/mosaic.hpp"

namespace vc_tomography::elementwise {

struct PlaceExposure {
    const float *signal, *weight;
    const std::int32_t* first;
    const float* quality;
    float level;
    int n, stride;
    float *sums, *support;
    int width;
    HR_HD void operator()(std::size_t i) const {
        const int r = i / n, c = i % n, x = first[r] + c;
        if (x < 0 || x >= width) return;
        const std::size_t src = std::size_t(r) * stride + c, dst = std::size_t(r) * width + x;
        const float w = weight[src] > 0.f ? weight[src] : 0.f;
        sums[dst] += quality[r] * (signal[src] + level * w);
        support[dst] += quality[r] * w;
    }
};

// Initialize the ordinary FBP with its complementary weight (focus < 0), then
// accumulate one bounded regional FBP at a time without another full-size image.
struct BlendRegion {
    const CanvasDisc* discs;
    int count, focus;
    SliceRoi region, output;
    const float* image;
    float* result;
    HR_HD static float taper(const CanvasDisc& d, int x, int y) {
        const double dx = x - d.x, dy = y - d.y, r2 = dx * dx + dy * dy;
        if (r2 >= d.radius * d.radius) return 0.f;
        const double inner = d.radius - d.fade;
        if (r2 <= inner * inner) return 1.f;
        return float(0.5 - 0.5 * cos(3.141592653589793 * (d.radius - sqrt(r2)) / d.fade));
    }
    HR_HD void operator()(std::size_t i) const {
        const int width = region.x1 - region.x0;
        const int x = region.x0 + int(i % width), y = region.y0 + int(i / width);
        const float w = focus < 0 ? 1.f : taper(discs[focus], x, y);
        if (w == 0.f) return;
        float sum = 0.f;
        for (int k = 0; k < count; ++k) sum += taper(discs[k], x, y);
        const std::size_t dst = std::size_t(y - output.y0) * (output.x1 - output.x0) + x - output.x0;
        if (focus < 0) result[dst] *= sum < 1.f ? 1.f - sum : 0.f;
        else result[dst] += image[i] * w / (sum > 1.f ? sum : 1.f);
    }
};

// At fixed tile-local column c the parallel ray direction is a - fan[c]
// (the sign of BackprojectParams::fan). Gather the two measured views either
// side of the requested a = target + fan[c]. Interpolate counts of weight
// and weighted line integrals separately, including across the turn boundary.
struct RebinTileRows {
    CanvasRebin grid;
    const float *s, *w, *gain;
    int rows, n, stride;
    float *out_s, *out_w;

    HR_HD void operator()(std::size_t i) const {
        const int r = static_cast<int>(i / n), c = static_cast<int>(i % n);
        double target = grid.measured[r] + grid.fan[c];
        if (target >= rows) target -= rows;
        int k = grid.lower[static_cast<int>(target)];
        while (grid.views[k + 1].bin <= target) ++k;
        const CanvasView lo = grid.views[k], hi = grid.views[k + 1];
        const float t = static_cast<float>((target - lo.bin) / (hi.bin - lo.bin));
        const std::size_t a = static_cast<std::size_t>(lo.row) * n + c;
        const std::size_t b = static_cast<std::size_t>(hi.row) * n + c;
        const float wa = hd::mul(1.f - t, w[a]), wb = hd::mul(t, w[b]);
        const std::size_t out = static_cast<std::size_t>(r) * stride + c;
        out_w[out] = hd::mul(gain[c], hd::add(wa, wb));
        const float sa = wa > 0.f ? hd::mul(wa, s[a]) : 0.f;
        const float sb = wb > 0.f ? hd::mul(wb, s[b]) : 0.f;
        out_s[out] = hd::mul(gain[c], hd::add(sa, sb));
    }
};
// Both source rows have already been delayed onto integer canvas columns at
// their own view angles. Compare the same canvas column in both of them.
struct RebinPlacedRows {
    const CanvasBracket* brackets;
    const float *s, *w;
    int n, stride;
    float *out_s, *out_w;

    HR_HD void operator()(std::size_t i) const {
        const int r = static_cast<int>(i / n), c = static_cast<int>(i % n);
        const CanvasBracket b = brackets[r];
        const int x0 = c + b.dx_lo, x1 = c + b.dx_hi;
        const bool have0 = x0 >= 0 && x0 < n, have1 = x1 >= 0 && x1 < n;
        const std::size_t a = static_cast<std::size_t>(b.lo) * stride + x0;
        const std::size_t z = static_cast<std::size_t>(b.hi) * stride + x1;
        const std::size_t o = static_cast<std::size_t>(r) * stride + c;
        out_s[o] = hd::add(hd::mul(1.f - b.fraction, have0 ? s[a] : 0.f), hd::mul(b.fraction, have1 ? s[z] : 0.f));
        out_w[o] = hd::add(hd::mul(1.f - b.fraction, have0 ? w[a] : 0.f), hd::mul(b.fraction, have1 ? w[z] : 0.f));
    }
};
// Native rows have already received their ordinary Fourier detector shift.
// Invert the additional detector displacement due to the column fan at a
// fixed canvas offset, then gather the resulting rays once in angle.
struct GatherRows {
    CanvasGather grid;
    const float *s, *w;
    int rows, n, stride;
    float *out_s, *out_w;

    struct Sample { double distance; float value, weight; };

    HR_HD double fan_bins(double x) const {
        if (!grid.fan) return 0;
        x = fmin(fmax(x, 0.), static_cast<double>(n - 1));
        const int j = static_cast<int>(x);
        return j == n - 1 ? grid.fan[j].bins :
            grid.fan[j].bins + (x - j) * (grid.fan[j + 1].bins - grid.fan[j].bins);
    }
    HR_HD double displacement(const CanvasPose& pose, int j) const {
        const CanvasFan f = grid.fan[j];
        return pose.projection * f.cos_minus_one - pose.tangent * f.sin_angle;
    }
    // For native view a, detector coordinate j lands at j + delta(j)
    // relative to its ordinary origin. The calibrated map is monotone.
    HR_HD double detector(const CanvasPose& pose, double u) const {
        if (!grid.fan) return u;
        int lo = static_cast<int>(fmin(fmax(floor(u), 0.), static_cast<double>(n - 2)));
        double left = lo + displacement(pose, lo), right = lo + 1 + displacement(pose, lo + 1);
        if (u < left || u > right) {
            int hi = n - 1;
            lo = 0;
            left = displacement(pose, lo);
            right = hi + displacement(pose, hi);
            if (u <= left) return u - left;
            if (u >= right) return hi + u - right;
            while (hi - lo > 1) {
                const int mid = (lo + hi) / 2;
                const double v = mid + displacement(pose, mid);
                if (v <= u) { lo = mid; left = v; }
                else { hi = mid; right = v; }
            }
        }
        return lo + (u - left) / (right - left);
    }
    HR_HD Sample sample(int index, int column, double target, double target_fan) const {
        double period = 0;
        if (index < 1) { index += rows; period = -rows; }
        if (index > rows) { index -= rows; period = rows; }
        const CanvasView view = grid.views[index];
        const CanvasPose pose = grid.poses[view.row];
        const int local = column - pose.first;
        const double x = detector(pose, local - pose.fraction);
        const double shifted = grid.fan ? x + pose.fraction : static_cast<double>(local);
        Sample v{(view.bin + period - target) + (target_fan - fan_bins(x)), 0, 0};
        if (shifted <= -1 || shifted >= n) return v;
        const int j = static_cast<int>(floor(shifted));
        const float t = static_cast<float>(shifted - j);
        const auto row = static_cast<std::size_t>(view.row) * stride;
        if (j >= 0) {
            v.value = (1 - t) * s[row + j];
            v.weight = (1 - t) * w[row + j];
        }
        if (t > 0 && j + 1 < n) {
            v.value += t * s[row + j + 1];
            v.weight += t * w[row + j + 1];
        }
        if (!(v.weight > 0)) { v.value = 0; v.weight = 0; }
        return v;
    }

    HR_HD void operator()(std::size_t i) const {
        const int r = static_cast<int>(i / n), c = static_cast<int>(i % n);
        const CanvasTarget dst = grid.targets[r];
        const int column = dst.first + c;
        const double fan = fan_bins(c - dst.fraction);
        double target = dst.bin + fan;
        if (target < 0 || target >= rows) {
            target = fmod(target, static_cast<double>(rows));
            if (target < 0) target += rows;
            if (target >= rows) target = 0;
        }
        int k = grid.lower[static_cast<int>(target)];
        while (grid.views[k + 1].bin <= target) ++k;
        const Sample lo = sample(k, column, target, fan), hi = sample(k + 1, column, target, fan);
        const double gap = hi.distance - lo.distance;
        const float t = gap > 0 ? static_cast<float>(fmin(1., fmax(0., -lo.distance / gap))) : 0.f;
        const auto out = static_cast<std::size_t>(r) * stride + c;
        out_w[out] = (1 - t) * lo.weight + t * hi.weight;
        out_s[out] = (1 - t) * lo.value + t * hi.value;
        // Use the local physical-angle fraction in acquisition-index space.
        // All six taps are cardinal at measured views, including irregular
        // ones. Trust the signed kernel only on contiguous, measured support.
        // A gap >1.5 nominal bins signals missing angular support. A weight
        // ratio below 1/4 makes division by the weaker taps amplify noise.
        if (!grid.lanczos || rows < 6 || t == 0 || t == 1 || gap > 1.5 || lo.weight <= 0 || hi.weight <= 0) return;
        float sum = 0, norm = 0, least = fminf(lo.weight, hi.weight), most = fmaxf(lo.weight, hi.weight);
        double previous = 0;
        for (int tap = -2; tap <= 3; ++tap) {
            const Sample v = tap == 0 ? lo : tap == 1 ? hi : sample(k + tap, column, target, fan);
            if (v.weight <= 0 || (tap > -2 && (v.distance <= previous || v.distance - previous > 1.5))) return;
            previous = v.distance;
            least = fminf(least, v.weight);
            most = fmaxf(most, v.weight);
            const float a = interp_core::lanczos3(tap - t);
            sum += a * (v.value / v.weight);
            norm += a;
        }
        if (least >= 0.25f * most) out_s[out] = out_w[out] * (sum / norm);
    }
};
// Keep the stronger recording at its measured angle. When the opposite
// support b exceeds the native support a, it supplies (b-a)/(b+a) of this
// direction. For antipodal views their average equals the weight-balanced fold.
struct CompleteViews {
    const CanvasOpposite* plan;
    const float *s, *w;
    int width;
    float *out, *support;
    HR_HD void operator()(std::size_t i) const {
        const CanvasOpposite& p = plan[i / width];
        const int mirror = width - 1 - static_cast<int>(i % width);
        const auto lo = static_cast<std::size_t>(p.row[2]) * width + mirror;
        const auto hi = static_cast<std::size_t>(p.row[3]) * width + mirror;
        const float a = w[i], t = p.fraction;
        const float b = (1 - t) * w[lo] + t * w[hi];
        support[i] = a + b;
        const float own = a > 0 ? s[i] / a : 0;
        if (b <= a) { out[i] = own; return; }
        float donor = ((1 - t) * (w[lo] > 0 ? s[lo] : 0) + t * (w[hi] > 0 ? s[hi] : 0)) / b;
        if (p.lanczos) {
            float sum = 0, least = b, most = b;
            for (int j = 0; j < 6; ++j) {
                const auto k = static_cast<std::size_t>(p.row[j]) * width + mirror;
                least = fminf(least, w[k]);
                most = fmaxf(most, w[k]);
                if (least < 0.25f * most) break;
                sum += p.coefficient[j] * (s[k] / w[k]);
            }
            if (least >= 0.25f * most) donor = sum;
        }
        out[i] = (2 * (a > 0 ? s[i] : 0) + (b - a) * donor) / (a + b);
    }
};

// The completed projection spans a symmetric canvas. Its own end values
// supply the ramp's two constant extensions at the physical view angle.
struct PadMeasured {
    const CanvasOpposite* plan;
    const float* sino;
    int width, padded, split;
    float* out;
    HR_HD void operator()(std::size_t i) const {
        const auto r = i / padded;
        const int x = static_cast<int>(i % padded);
        const int c = x < width ? x : (x < split ? width - 1 : 0);
        out[i] = plan[r].quadrature * sino[r * width + c];
    }
};
}  // namespace vc_tomography::elementwise
