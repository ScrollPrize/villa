// helirec/calibration/intensity.hpp: the reduction of darks and flats, the weight map and its apodisation widths, and detector
// images at the reduced resolutions of frames stored as JPEG 2000.
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "helirec/calibration/intensity.hpp"
#include "helirec/raw_reader.hpp"

namespace helirec::calibration {

std::vector<float> mean_frames(RawReader& reader, const int32_t* file_index, const int32_t* frame_index, std::size_t n, const float* dark,
                               const double* control_A, double nominal_A, const std::function<void(std::size_t)>& progress) {
    if (!n) throw Error("mean_frames: no frames");
    if (reader.dtype() != "uint16") throw Error("mean_frames: frames of 16-bit counts");
    if ((dark == nullptr) != (control_A == nullptr)) throw Error("mean_frames: a dark and the ring current of every frame go together");
    const std::size_t px = static_cast<std::size_t>(reader.rows()) * static_cast<std::size_t>(reader.cols());
    const std::size_t batch = 40;
    std::vector<uint16_t> buf(std::min(batch, n) * px);
    std::vector<float> acc(px);
    const std::ptrdiff_t P = static_cast<std::ptrdiff_t>(px);
    for (std::size_t s = 0; s < n; s += batch) {
        const std::size_t e = std::min(n, s + batch);
        reader.read_frames(file_index + s, frame_index + s, e - s, buf.data());
        for (std::size_t k = s; k < e; ++k) {
            const uint16_t* F = buf.data() + (k - s) * px;
            const bool first = k == 0;
            // every pixel's sum runs over the frames in order (float32, as numpy.mean of the stack does not: this is
            // the reference's arithmetic); the pixels are independent
            if (dark) {
                const float scale = static_cast<float>(nominal_A) / static_cast<float>(control_A[k]);
#pragma omp parallel for schedule(static)
                for (std::ptrdiff_t i = 0; i < P; ++i) {
                    const float f = dark[i] + (static_cast<float>(F[i]) - dark[i]) * scale;
                    acc[static_cast<std::size_t>(i)] = first ? f : acc[static_cast<std::size_t>(i)] + f;
                }
            } else {
#pragma omp parallel for schedule(static)
                for (std::ptrdiff_t i = 0; i < P; ++i) {
                    const float f = static_cast<float>(F[i]);
                    acc[static_cast<std::size_t>(i)] = first ? f : acc[static_cast<std::size_t>(i)] + f;
                }
            }
        }
        if (progress) progress(e - s);
    }
    const float count = static_cast<float>(n);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < P; ++i) acc[static_cast<std::size_t>(i)] /= count;
    return acc;
}

std::vector<float> weight_map(const std::vector<const float*>& flats, int rows, int cols, double transition_width_vertical, double sigma, int crop_top,
                              int crop_bottom) {
    if (flats.empty()) throw Error("weight_map: no flats");
    const std::size_t px = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    std::vector<float> prof(flats[0], flats[0] + px);
    for (std::size_t k = 1; k < flats.size(); ++k)
        for (std::size_t i = 0; i < px; ++i) prof[i] += flats[k][i];
    const float count = static_cast<float>(flats.size());
    float top = 0;
    for (std::size_t i = 0; i < px; ++i) {
        prof[i] /= count;
        top = i ? std::max(top, prof[i]) : prof[i];
    }
    for (float& v : prof) v /= top;
    std::vector<float> a(px), g(px);
    gaussian_filter1d<float>(prof.data(), a.data(), 1, static_cast<std::size_t>(rows), static_cast<std::size_t>(cols), 1, sigma, "reflect");
    gaussian_filter1d<float>(a.data(), g.data(), 1, static_cast<std::size_t>(rows), static_cast<std::size_t>(cols), 2, sigma, "reflect");
    const std::vector<float> border = weight_border_v(rows, transition_width_vertical, crop_top, crop_bottom);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) g[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(c)] *= border[static_cast<std::size_t>(r)];
    return g;
}

ApodisationWidths apodisation_widths(int n_cols, double est_cor_main_offset, double companion_dx, double transition_width_horizontal,
                                     double transition_width_borders) {
    const double r = (n_cols - 1) / 2.0 + est_cor_main_offset;
    if (r < (n_cols - 1) / 2.0) throw Error("rotation axis in the left half of the main scan is not supported yet");
    const double pente = std::min(2 * transition_width_horizontal, n_cols - r);
    if (!(companion_dx < 0)) throw Error("companion on the right of the main scan is not supported yet");
    ApodisationWidths w;  // companion on the left of the main scan
    w.main_left = transition_width_borders, w.main_right = pente;
    w.companion_left = 0.0, w.companion_right = transition_width_borders;
    return w;
}

namespace {

// the low-pass of JPEG 2000's irreversible (9/7) wavelet, taps 0, +-1, .. +-4; unit gain
constexpr double kLow97[5] = {0.6029490182363579, 0.2668641184428723, -0.07822326652898785, -0.01686411844287495, 0.02674875741080976};

// n samples at x[0], x[stride], ..: sample q of the result is the low-pass about x[2 q] (whole-sample symmetric at
// both ends); (n + 1) / 2 samples at out[0], out[out_stride], ..
void low_band(const double* x, std::size_t n, std::ptrdiff_t stride, double* out, std::ptrdiff_t out_stride) {
    const long last = static_cast<long>(n) - 1;
    const auto at = [&](long i) {
        if (last == 0) return x[0];
        i = std::labs(i) % (2 * last);
        if (i > last) i = 2 * last - i;
        return x[i * stride];
    };
    for (long q = 0, i = 0; i <= last; ++q, i += 2) {
        double s = kLow97[0] * x[i * stride];
        for (int m = 1; m < 5; ++m) s = s + kLow97[m] * (at(i - m) + at(i + m));
        out[q * out_stride] = s;
    }
}

}  // namespace

std::vector<float> at_resolution(const double* images, std::size_t n, int rows, int cols, int reduce, int chunk_rows, int* out_rows, int* out_cols) {
    if (reduce < 0 || rows < 1 || cols < 1 || chunk_rows < 1) throw Error("at_resolution: bad sizes");
    if (reduce && (chunk_rows >> (reduce - 1)) < 1) throw Error("at_resolution: the chunks have fewer rows than 2^(reduce - 1)");
    const int keep_rows = rows >> reduce;
    int final_cols = cols;
    for (int l = 0; l < reduce; ++l) final_cols = (final_cols + 1) / 2;
    if (out_rows) *out_rows = keep_rows;
    if (out_cols) *out_cols = final_cols;
    std::vector<float> out(n * static_cast<std::size_t>(keep_rows) * static_cast<std::size_t>(final_cols));
    const int padded = ((rows + chunk_rows - 1) / chunk_rows) * chunk_rows;  // zeros beyond the frame in the last chunk
    const std::ptrdiff_t N = static_cast<std::ptrdiff_t>(n);
#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t im = 0; im < N; ++im) {
        std::size_t R = static_cast<std::size_t>(padded), C = static_cast<std::size_t>(cols);
        std::vector<double> x(R * C, 0.0), y;
        std::copy(images + static_cast<std::size_t>(im) * static_cast<std::size_t>(rows) * C,
                  images + (static_cast<std::size_t>(im) + 1) * static_cast<std::size_t>(rows) * C, x.begin());
        for (int level = 0; level < reduce; ++level) {
            const std::size_t c = static_cast<std::size_t>(chunk_rows >> level);
            // down the rows, each chunk of rows alone
            std::size_t R2 = 0;
            for (std::size_t a = 0; a < R; a += c) R2 += (std::min(c, R - a) + 1) / 2;
            y.assign(R2 * C, 0.0);
            std::size_t at_row = 0;
            for (std::size_t a = 0; a < R; a += c) {
                const std::size_t len = std::min(c, R - a);
                for (std::size_t k = 0; k < C; ++k)
                    low_band(x.data() + a * C + k, len, static_cast<std::ptrdiff_t>(C), y.data() + at_row * C + k, static_cast<std::ptrdiff_t>(C));
                at_row += (len + 1) / 2;
            }
            // along the columns
            const std::size_t C2 = (C + 1) / 2;
            x.assign(R2 * C2, 0.0);
            for (std::size_t r = 0; r < R2; ++r) low_band(y.data() + r * C, C, 1, x.data() + r * C2, 1);
            R = R2, C = C2;
        }
        float* o = out.data() + static_cast<std::size_t>(im) * static_cast<std::size_t>(keep_rows) * static_cast<std::size_t>(final_cols);
        for (std::size_t r = 0; r < static_cast<std::size_t>(keep_rows); ++r)
            for (std::size_t k = 0; k < C; ++k) o[r * C + k] = static_cast<float>(x[r * C + k]);
    }
    return out;
}

}  // namespace helirec::calibration
