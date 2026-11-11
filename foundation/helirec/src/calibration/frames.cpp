#include "helirec/calibration/frames.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <map>

#include "helirec/raw_reader.hpp"

namespace helirec::calibration {

Frames::Frames(const Record& rec, int n_threads) : rec_(&rec), n_threads_(n_threads) {
    std::vector<std::string> files;
    std::map<std::string, std::int32_t> index;
    std::string dataset;
    for (const Stretch& s : rec.stretches) {
        if (!rows_) rows_ = s.rows, cols_ = s.cols;
        if (s.rows != rows_ || s.cols != cols_) throw Error("stretch " + s.entry + ": frame size differs from the other stretches");
        std::vector<std::int32_t> of_piece;
        for (const FramePiece& p : s.pieces) {
            if (dataset.empty()) dataset = p.dataset;
            if (p.dataset != dataset) throw Error("stretch " + s.entry + ": frame files use different dataset names");
            auto it = index.find(p.file);
            if (it == index.end()) {
                it = index.emplace(p.file, static_cast<std::int32_t>(files.size())).first;
                files.push_back(p.file);
            }
            of_piece.push_back(it->second);
        }
        piece_file_.push_back(std::move(of_piece));
    }
    reader_ = std::make_unique<RawReader>(std::move(files), dataset, rows_, cols_);
}

Frames::~Frames() = default;

void Frames::locate(int stretch, std::int64_t frame, std::int32_t& file, std::int32_t& idx) const {
    const Stretch& s = rec_->stretches.at(stretch);
    if (frame < 0 || frame >= s.n_frames) throw Error("stretch " + s.entry + ": frame " + std::to_string(frame) + " does not exist");
    // the piece holding `frame`: the last one starting at or before it
    auto it = std::upper_bound(s.pieces.begin(), s.pieces.end(), frame, [](std::int64_t f, const FramePiece& p) { return f < p.first; });
    if (it == s.pieces.begin()) throw Error("stretch " + s.entry + ": frame " + std::to_string(frame) + " is in no file");
    --it;
    if (frame >= it->first + it->count) throw Error("stretch " + s.entry + ": frame " + std::to_string(frame) + " is in no file");
    file = piece_file_[stretch][it - s.pieces.begin()];
    idx = static_cast<std::int32_t>(it->source_first + (frame - it->first));
}

void Frames::read_rows(int stretch, const std::int64_t* frame, const std::int32_t* row0, std::size_t n, int nrows, std::uint16_t* out) {
    std::vector<std::int32_t> file(n), idx(n);
    for (std::size_t i = 0; i < n; ++i) locate(stretch, frame[i], file[i], idx[i]);
    reader_->read_rows(file.data(), idx.data(), row0, n, nrows, out, n_threads_);
}

void Frames::read(int stretch, std::int64_t first, std::int64_t count, std::uint16_t* out) {
    std::vector<std::int64_t> frame(count);
    std::vector<std::int32_t> row0(count, 0);
    for (std::int64_t i = 0; i < count; ++i) frame[i] = first + i;
    read_rows(stretch, frame.data(), row0.data(), static_cast<std::size_t>(count), rows_, out);
}

void Frames::visit(int stretch, std::int64_t first, std::int64_t count,
                   const std::function<void(std::int64_t, const std::uint16_t*)>& visit) {
    std::vector<std::int32_t> file(count), idx(count);
    for (std::int64_t i = 0; i < count; ++i) locate(stretch, first + i, file[i], idx[i]);
    reader_->visit_frames(file.data(), idx.data(), static_cast<std::size_t>(count), rows_,
                          [&](std::size_t i, int, int, const std::uint16_t* data) { visit(first + static_cast<std::int64_t>(i), data); },
                          n_threads_);
}

SeriesStats series_stats(Frames& frames, int stretch) {
    const Stretch& s = frames.record().stretches.at(stretch);
    const std::size_t np = static_cast<std::size_t>(frames.rows()) * frames.cols();
    SeriesStats st;
    st.n = s.n_frames;
    st.frame_mean.assign(static_cast<std::size_t>(s.n_frames), 0.0);
    // integer sums: exact in double for 16-bit values over any series that fits on disk
    std::vector<double> sum(np, 0.0), sumsq(np, 0.0);
    const std::int64_t chunk = 16;
    std::vector<std::uint16_t> buf(static_cast<std::size_t>(chunk) * np);
    for (std::int64_t f0 = 0; f0 < s.n_frames; f0 += chunk) {
        const std::int64_t n = std::min(chunk, s.n_frames - f0);
        frames.read(stretch, f0, n, buf.data());
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(np); ++p) {
            double a = 0, b = 0;
            for (std::int64_t k = 0; k < n; ++k) {
                const double v = buf[static_cast<std::size_t>(k) * np + p];
                a += v;
                b += v * v;
            }
            sum[p] += a;
            sumsq[p] += b;
        }
#pragma omp parallel for schedule(static)
        for (std::int64_t k = 0; k < n; ++k) {
            double a = 0;
            const std::uint16_t* frame = buf.data() + static_cast<std::size_t>(k) * np;
            for (std::size_t p = 0; p < np; ++p) a += frame[p];
            st.frame_mean[static_cast<std::size_t>(f0 + k)] = a / static_cast<double>(np);
        }
    }
    st.mean.resize(np);
    st.variance.resize(np);
    const double n = static_cast<double>(s.n_frames);
    for (std::size_t p = 0; p < np; ++p) {
        st.mean[p] = sum[p] / n;
        st.variance[p] = n > 1 ? std::max(0.0, (sumsq[p] - sum[p] * sum[p] / n) / (n - 1)) : 0.0;
    }
    return st;
}

FlatField make_flat_field(Frames& frames) {
    const Record& rec = frames.record();
    FlatField ff;
    ff.rows = frames.rows();
    ff.cols = frames.cols();
    const std::size_t np = static_cast<std::size_t>(ff.rows) * ff.cols;
    std::vector<double> dark(np, 0.0), flat(np, 0.0);
    double n_dark = 0, n_flat = 0;
    for (std::size_t i = 0; i < rec.stretches.size(); ++i) {
        const Role role = rec.stretches[i].role;
        if (role != Role::Dark && role != Role::Flat) continue;
        SeriesStats st = series_stats(frames, static_cast<int>(i));
        std::vector<double>& acc = role == Role::Dark ? dark : flat;
        (role == Role::Dark ? n_dark : n_flat) += static_cast<double>(st.n);
        for (std::size_t p = 0; p < np; ++p) acc[p] += st.mean[p] * static_cast<double>(st.n);
        ff.series.push_back(std::move(st));
        ff.series_stretch.push_back(static_cast<int>(i));
    }
    if (n_dark == 0 || n_flat == 0) throw Error("the record holds no dark or no flat frames");
    ff.dark.resize(np);
    ff.gain.resize(np);
    for (std::size_t p = 0; p < np; ++p) {
        ff.dark[p] = static_cast<float>(dark[p] / n_dark);
        ff.gain[p] = static_cast<float>(std::max(1.0, flat[p] / n_flat - dark[p] / n_dark));
    }
    return ff;
}

void line_integrals(const FlatField& ff, const std::uint16_t* raw, const std::int32_t* row0, std::size_t n, int nrows, int col0,
                    int ncols, bool flip_cols, float* out) {
    if (col0 < 0 || col0 + ncols > ff.cols) throw Error("line_integrals: columns outside the detector");
#pragma omp parallel for collapse(2) schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        for (int r = 0; r < nrows; ++r) {
            float* dst = out + (static_cast<std::size_t>(i) * nrows + r) * ncols;
            const int row = row0[i] + r;
            if (row < 0 || row >= ff.rows) {
                std::fill(dst, dst + ncols, 0.0f);
                continue;
            }
            const std::uint16_t* src = raw + (static_cast<std::size_t>(i) * nrows + r) * ff.cols + col0;
            const float* dark = ff.dark.data() + static_cast<std::size_t>(row) * ff.cols + col0;
            const float* gain = ff.gain.data() + static_cast<std::size_t>(row) * ff.cols + col0;
            for (int c = 0; c < ncols; ++c) {
                const float t = std::max(1e-6f, (static_cast<float>(src[c]) - dark[c]) / gain[c]);
                dst[flip_cols ? ncols - 1 - c : c] = -std::log(t);
            }
        }
    }
}

void DetectorMap::displacement(double c, double v, double& du, double& dv) const {
    const double x = (c - c0) / cs, y = (v - v0) / vs;
    du = dv = 0;
    for (std::size_t k = 0; k < i.size(); ++k) {
        const double m = std::pow(x, i[k]) * std::pow(y, j[k]);
        du += eu[k] * m;
        dv += ev[k] * m;
    }
}

void DetectorMap::source(double x, double y, double& c, double& v) const {
    c = x, v = y;
    for (int pass = 0; pass < 3; ++pass) {  // the map changes by 1e-3 per pixel: three passes leave nothing
        double du, dv;
        displacement(c, v, du, dv);
        c = x - du, v = y - dv;
    }
}

namespace {

// weights of the four samples around a position at fraction f past the second of them (Keys, a = -1/2)
inline void cubic_weights(float f, float w[4]) {
    w[0] = ((-0.5f * f + 1.0f) * f - 0.5f) * f;
    w[1] = (1.5f * f - 2.5f) * f * f + 1.0f;
    w[2] = ((-1.5f * f + 2.0f) * f + 0.5f) * f;
    w[3] = (0.5f * f - 0.5f) * f * f;
}

}  // namespace

void line_integrals_mapped(Frames& frames, const FlatField& ff, const DetectorMap& map, int stretch, const std::int64_t* frame,
                           int row0, std::size_t n, int nrows, int col0, int ncols, bool flip_cols, float* out) {
    const int rows = frames.rows(), cols = frames.cols();
    static std::vector<std::uint16_t> raw;  // kept between blocks: their pages are touched once
    static std::vector<float> plain;
    if (map.empty()) {
        std::vector<std::int32_t> r0(n, row0);
        raw.resize(n * static_cast<std::size_t>(nrows) * cols);
        frames.read_rows(stretch, frame, r0.data(), n, nrows, raw.data());
        line_integrals(ff, raw.data(), r0.data(), n, nrows, col0, ncols, flip_cols, out);
        return;
    }
    // where every sample of the block is read from, the same in every frame
    struct Tap {
        std::int32_t r, c;  // the second of the four rows and of the four columns, in the detector
        float wr[4], wc[4];
    };
    std::vector<Tap> taps(static_cast<std::size_t>(nrows) * ncols);
    int r_min = rows, r_max = -1, c_min = cols, c_max = -1;
#pragma omp parallel for schedule(static) reduction(min : r_min, c_min) reduction(max : r_max, c_max)
    for (int r = 0; r < nrows; ++r)
        for (int c = 0; c < ncols; ++c) {
            double sc, sv;
            map.source(flip_cols ? col0 + ncols - 1 - c : col0 + c, row0 + r, sc, sv);
            sc = std::clamp(sc, 0.0, cols - 1.0);
            sv = std::clamp(sv, 0.0, rows - 1.0);
            Tap& t = taps[static_cast<std::size_t>(r) * ncols + c];
            t.r = static_cast<std::int32_t>(std::floor(sv));
            t.c = static_cast<std::int32_t>(std::floor(sc));
            cubic_weights(static_cast<float>(sv - t.r), t.wr);
            cubic_weights(static_cast<float>(sc - t.c), t.wc);
            r_min = std::min(r_min, t.r - 1), r_max = std::max(r_max, t.r + 2);
            c_min = std::min(c_min, t.c - 1), c_max = std::max(c_max, t.c + 2);
        }
    r_min = std::max(r_min, 0), r_max = std::min(r_max, rows - 1);
    c_min = std::max(c_min, 0), c_max = std::min(c_max, cols - 1);
    const int nr = r_max - r_min + 1, nc = c_max - c_min + 1;
    std::vector<std::int32_t> r0(n, r_min);
    raw.resize(n * static_cast<std::size_t>(nr) * cols);
    plain.resize(n * static_cast<std::size_t>(nr) * nc);
    frames.read_rows(stretch, frame, r0.data(), n, nr, raw.data());
    line_integrals(ff, raw.data(), r0.data(), n, nr, c_min, nc, false, plain.data());
#pragma omp parallel for collapse(2) schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
        for (int r = 0; r < nrows; ++r) {
            const float* src = plain.data() + static_cast<std::size_t>(i) * nr * nc;
            float* dst = out + (static_cast<std::size_t>(i) * nrows + r) * ncols;
            const Tap* tp = taps.data() + static_cast<std::size_t>(r) * ncols;
            for (int c = 0; c < ncols; ++c) {
                const Tap& t = tp[c];
                float sum = 0;
                for (int a = 0; a < 4; ++a) {
                    const float* line = src + static_cast<std::size_t>(std::clamp(t.r - 1 + a, r_min, r_max) - r_min) * nc;
                    float s = 0;
                    for (int b = 0; b < 4; ++b) s += t.wc[b] * line[std::clamp(t.c - 1 + b, c_min, c_max) - c_min];
                    sum += t.wr[a] * s;
                }
                dst[c] = sum;
            }
        }
}

void shear_frames(float* block, std::size_t n, int nrows, int ncols, double shear) {
    if (shear == 0.0 || n < 2) return;
    static std::vector<float> copy;
    const std::size_t plane = static_cast<std::size_t>(nrows) * ncols;
    copy.assign(block, block + n * plane);
    const std::ptrdiff_t last = static_cast<std::ptrdiff_t>(n) - 1;
#pragma omp parallel for collapse(2) schedule(static)
    for (std::ptrdiff_t q = 0; q < static_cast<std::ptrdiff_t>(n); ++q)
        for (int r = 0; r < nrows; ++r) {
            float* dst = block + static_cast<std::size_t>(q) * plane + static_cast<std::size_t>(r) * ncols;
            for (int c = 0; c < ncols; ++c) {
                const double at = std::clamp(static_cast<double>(q) + shear * (c - 0.5 * (ncols - 1)), 0.0, static_cast<double>(last));
                const std::ptrdiff_t k = static_cast<std::ptrdiff_t>(std::floor(at));
                float w[4];
                cubic_weights(static_cast<float>(at - static_cast<double>(k)), w);
                float sum = 0;
                for (int a = 0; a < 4; ++a)
                    sum += w[a] * copy[static_cast<std::size_t>(std::clamp<std::ptrdiff_t>(k - 1 + a, 0, last)) * plane + static_cast<std::size_t>(r) * ncols + c];
                dst[c] = sum;
            }
        }
}

}  // namespace helirec::calibration
