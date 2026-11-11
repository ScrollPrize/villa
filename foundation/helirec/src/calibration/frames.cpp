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

}  // namespace helirec::calibration
