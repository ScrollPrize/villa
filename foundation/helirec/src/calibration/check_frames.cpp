// The record against every frame on disk: the inventory pass, the dark and flat series, and the beam's level.
#include <omp.h>

#include <cmath>
#include <fstream>

#include "calibration/util.hpp"
#include "calibration/ceres_support.hpp"
#include "helirec/calibration/record_checks.hpp"

namespace helirec::calibration {

namespace {

std::string join(const std::vector<std::string>& v, const char* sep = "; ") {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? sep : "") + v[i];
    return s;
}

int n_strips(const Inventory& inv, int cols) { return (cols + inv.strip_cols - 1) / inv.strip_cols; }

// the rectangle of the recorded statistics; without one, the whole frame
struct Rect {
    int x = 0, y = 0, w = 0, h = 0;
    bool recorded = false;
};
Rect stat_rect(const Stretch& s, int rows, int cols) {
    Rect r;
    r.recorded = s.stat_sum.size() == static_cast<std::size_t>(s.n_frames) && s.stat_w > 0 && s.stat_h > 0;
    if (r.recorded) r.x = s.stat_x, r.y = s.stat_y, r.w = s.stat_w, r.h = s.stat_h;
    else r.w = cols, r.h = rows;
    return r;
}

}  // namespace

Inventory check_inventory(const Record& rec, Frames& frames, Report& rep, int step) {
    if (step < 0) throw Error("inventory step must not be negative");
    const std::string scan = rec.label;
    const int rows = frames.rows(), cols = frames.cols();
    Inventory inv;
    inv.stretches.resize(rec.stretches.size());
    const int ns = n_strips(inv, cols);
    const int max_threads = std::max(64, omp_get_max_threads());

    std::int64_t frames_read = 0, compared = 0, mismatched = 0, absent = 0, located = 0;
    std::vector<std::string> first_bad, unrecorded, lossy, sparse;
    for (std::size_t si = 0; si < rec.stretches.size(); ++si) {
        const Stretch& s = rec.stretches[si];
        Inventory::PerStretch& out = inv.stretches[si];
        const std::size_t n = static_cast<std::size_t>(s.n_frames);
        const Rect rect = stat_rect(s, rows, cols);
        const bool have = rect.recorded;
        const int x0 = rect.x, y0 = rect.y, w = rect.w, h = rect.h;
        if (x0 < 0 || y0 < 0 || x0 + w > cols || y0 + h > rows) throw Error("stretch " + s.entry + ": statistics rectangle outside the frame");
        out.sum.assign(n, 0);
        out.min.assign(n, 0);
        out.max.assign(n, 0);
        out.frame_max.assign(n, 0);
        out.strips.assign(n * ns, 0);
        out.histogram.assign(65536, 0);
        std::vector<std::vector<std::int64_t>> hist(max_threads);
        const auto on_disk = s.frames_on_disk();
        absent += s.n_frames - s.n_on_disk();
        if (!have && s.n_on_disk() > 0) unrecorded.push_back(s.entry);
        std::int64_t every = step > 0 ? step : 1;
        if (s.n_on_disk() > 0) {
            const Frames::Stored st = frames.stored(static_cast<int>(si));
            if (step == 0 && st.compressed && !have) every = default_inventory_sparse_step;
            if (every > 1) located += frames.locate_all(static_cast<int>(si));
            if (st.compressed)
                lossy.push_back(fmt("%s: %s, %s, %d bits, chunks of %d rows, %.0f bytes per frame = %.2f bits per value", s.entry.c_str(), st.codec.c_str(),
                                    st.reversible ? "reversible (exact)" : "irreversible (9/7 wavelet: the stored values are not the detector's)", st.bit_depth, st.chunk_rows,
                                    st.bytes_per_frame, st.bytes_per_frame * 8 / (static_cast<double>(rows) * cols)));
        }
        const auto each = [&](std::int64_t f, const std::uint16_t* data) {
            std::vector<std::int64_t>& my = hist[omp_get_thread_num()];
            if (my.empty()) my.assign(65536, 0);
            std::uint16_t top = 0;
            for (std::size_t p = 0; p < static_cast<std::size_t>(rows) * cols; ++p) {
                ++my[data[p]];
                top = std::max(top, data[p]);
            }
            std::uint64_t sum = 0;
            std::uint16_t lo = 65535, hi = 0;
            std::vector<std::uint64_t> strip(ns, 0);
            for (int r = y0; r < y0 + h; ++r) {
                const std::uint16_t* row = data + static_cast<std::size_t>(r) * cols;
                for (int c = x0; c < x0 + w; ++c) {
                    lo = std::min(lo, row[c]);
                    hi = std::max(hi, row[c]);
                }
                for (int k = 0; k < ns; ++k) {
                    const int a = std::max(x0, k * inv.strip_cols), b = std::min(x0 + w, (k + 1) * inv.strip_cols);
                    std::uint64_t t = 0;
                    for (int c = a; c < b; ++c) t += row[c];
                    strip[k] += t;
                }
            }
            for (int k = 0; k < ns; ++k) {
                const int a = std::max(x0, k * inv.strip_cols), b = std::min(x0 + w, (k + 1) * inv.strip_cols);
                sum += strip[k];
                out.strips[static_cast<std::size_t>(f) * ns + k] = b > a ? static_cast<float>(static_cast<double>(strip[k]) / (static_cast<double>(b - a) * h)) : 0.0f;
            }
            out.sum[f] = static_cast<double>(sum);
            out.min[f] = lo;
            out.max[f] = hi;
            out.frame_max[f] = top;
        };
        std::int64_t read_here = 0;
        if (every == 1) {
            for (const auto& [a, b] : on_disk) frames.visit(static_cast<int>(si), a, b - a, each), read_here += b - a;
        } else {
            // one frame in `every`, in runs of single frames (the reader takes whole frames in parallel)
            std::vector<std::int64_t> picked;
            for (const auto& [a, b] : on_disk)
                for (std::int64_t f = a; f < b; f += every) picked.push_back(f);
            const std::size_t np = static_cast<std::size_t>(rows) * cols;
            const std::size_t batch = 96;
            std::vector<std::uint16_t> buf(std::min(batch, picked.size()) * np);
            std::vector<std::int32_t> zero(batch, 0);
            for (std::size_t i0 = 0; i0 < picked.size(); i0 += batch) {
                const std::size_t m = std::min(batch, picked.size() - i0);
                frames.read_rows(static_cast<int>(si), picked.data() + i0, zero.data(), m, rows, buf.data());
#pragma omp parallel for schedule(dynamic, 1)
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(m); ++i) each(picked[i0 + i], buf.data() + static_cast<std::size_t>(i) * np);
            }
            read_here = static_cast<std::int64_t>(picked.size());
            sparse.push_back(fmt("%s: one frame in %lld (%lld of %lld on disk)", s.entry.c_str(), static_cast<long long>(every), static_cast<long long>(read_here),
                                 static_cast<long long>(s.n_on_disk())));
        }
        for (const auto& my : hist)
            if (!my.empty())
                for (int v = 0; v < 65536; ++v) out.histogram[v] += my[v];
        frames_read += read_here;
        if (!have) continue;
        for (std::size_t f = 0; f < n; ++f) {
            if (!s.on_disk(static_cast<std::int64_t>(f)) || (every > 1 && !out.was_read(f))) continue;
            ++compared;
            const bool same = out.sum[f] == s.stat_sum[f] && (s.stat_min.size() != n || out.min[f] == s.stat_min[f]) &&
                              (s.stat_max.size() != n || out.max[f] == s.stat_max[f]) &&
                              (s.stat_avg.size() != n || std::fabs(out.sum[f] / (static_cast<double>(w) * h) - s.stat_avg[f]) <= 1e-9 * s.stat_avg[f]);
            if (!same) {
                ++mismatched;
                if (first_bad.size() < 8) first_bad.push_back(fmt("%s frame %zu (sum %.0f, recorded %.0f)", s.entry.c_str(), f, out.sum[f], s.stat_sum[f]));
            }
        }
    }
    {
        Finding& f = rep.add("4.1", scan, "every frame on disk against the statistics recorded for it at acquisition");
        f.stated = "sum, min, max and mean of each frame over the statistics rectangle";
        f.found = fmt("%lld frames read, %lld compared, %lld differ", static_cast<long long>(frames_read), static_cast<long long>(compared), static_cast<long long>(mismatched));
        if (!first_bad.empty()) f.found += ": " + join(first_bad);
        f.verdict = compared == 0 ? "unverified" : mismatched == 0 ? "verified" : "fails";
        f.note = "the sums are integers and are compared exactly: frame k of the record is the frame the detector software saw as k, unaltered. "
                 "This does not show that the camera delivered a frame for every trigger";
        f.numbers = {{"frames", static_cast<double>(frames_read)}, {"compared", static_cast<double>(compared)}, {"mismatched", static_cast<double>(mismatched)}};
        if (absent) {
            f.found += fmt("; %lld frames of the record are not on disk and were not read", static_cast<long long>(absent));
            f.numbers.push_back({"absent", static_cast<double>(absent)});
        }
        if (!unrecorded.empty()) {
            f.found += "; no statistics recorded for " + join(unrecorded, ", ") + ": their frames were read and decoded, and compared with nothing";
            if (f.verdict == "verified") f.verdict = "unverified";
        }
        if (!sparse.empty()) {
            f.found += "; not every frame was read: " + join(sparse) + fmt("; every chunk of the %lld frames of those stretches on disk was located in its file", static_cast<long long>(located));
            if (f.verdict == "verified") f.verdict = "unverified";
            f.numbers.push_back({"located", static_cast<double>(located)});
        }
    }
    // 4.8: frames that are not stored as the detector delivered them
    if (!lossy.empty()) {
        Finding& f = rep.add("4.8 8.1", scan, "how the frames are stored");
        f.stated = "the dataset's filter and the header of the first chunk of each stretch";
        f.found = join(lossy);
        bool exact = true;
        for (const std::string& l : lossy) exact &= l.find("irreversible") == std::string::npos;
        f.verdict = exact ? "note" : "unverified";
        f.note = exact ? "compressed without loss"
                       : "what an irreversible codestream decodes to depends on the decoder to about a count (here OpenJPEG). The noise of these frames is not the detector's: "
                         "a noise model, and anything read off single values, cannot be taken from them";
    }
    // 8.1.4, 8.1.7: the raw values
    {
        Finding& f = rep.add("8.1.4 8.1.7", scan, "range of the raw values, whole frames");
        std::vector<std::string> parts;
        for (std::size_t si = 0; si < rec.stretches.size(); ++si) {
            const Stretch& s = rec.stretches[si];
            const auto& hgram = inv.stretches[si].histogram;
            std::int64_t total = 0;
            for (std::int64_t v : hgram) total += v;
            if (!total) {
                parts.push_back(fmt("%s %s: no frame on disk", s.entry.c_str(), role_name(s.role)));
                continue;
            }
            int lowest = 0, highest = 65535;
            while (lowest < 65535 && !hgram[lowest]) ++lowest;
            while (highest > 0 && !hgram[highest]) --highest;
            // the continuous distribution: from the most frequent value outwards until 64 values in a row are empty
            int mode = lowest;
            for (int v = lowest; v <= highest; ++v)
                if (hgram[v] > hgram[mode]) mode = v;
            int bulk = mode, bulk_low = mode, empty = 0;
            for (int v = mode; v <= highest; ++v) {
                if (hgram[v]) bulk = v, empty = 0;
                else if (++empty >= 64) break;
            }
            empty = 0;
            for (int v = mode; v >= lowest; --v) {
                if (hgram[v]) bulk_low = v, empty = 0;
                else if (++empty >= 64) break;
            }
            std::int64_t above = 0, below = 0;
            for (int v = bulk + 1; v <= 65535; ++v) above += hgram[v];
            for (int v = 0; v < bulk_low; ++v) below += hgram[v];
            std::int64_t frames_top = 0, first = -1;
            for (std::size_t k = 0; k < inv.stretches[si].frame_max.size(); ++k)
                if (inv.stretches[si].frame_max[k] == 65535) {
                    ++frames_top;
                    if (first < 0) first = static_cast<std::int64_t>(k);
                }
            std::string part = fmt("%s %s: %d to %d; continuous from %d to %d, %lld values below and %lld above; %lld values at 65535", s.entry.c_str(), role_name(s.role),
                                   lowest, highest, bulk_low, bulk, static_cast<long long>(below), static_cast<long long>(above), static_cast<long long>(hgram[65535]));
            if (frames_top) part += fmt(" in %lld frames (first: %lld)", static_cast<long long>(frames_top), static_cast<long long>(first));
            parts.push_back(part);
            f.numbers.push_back({"max_" + s.entry, static_cast<double>(highest)});
            f.numbers.push_back({"bulk_max_" + s.entry, static_cast<double>(bulk)});
            f.numbers.push_back({"above_bulk_" + s.entry, static_cast<double>(above)});
            f.numbers.push_back({"bulk_min_" + s.entry, static_cast<double>(bulk_low)});
            f.numbers.push_back({"below_bulk_" + s.entry, static_cast<double>(below)});
            (void)total;
        }
        f.stated = "16-bit values; the saturation flag of the detector software is off";
        f.found = join(parts);
        f.note = "values outside the continuous distribution are isolated outliers (8.1.7). A value is the mean of binned sensor pixels: one saturated sensor pixel does not reach 65535";
    }
    return inv;
}

void save_inventory(const Inventory& inv, const Record& rec, const std::string& path) {
    std::ofstream bin(path + ".bin", std::ios::binary);
    std::ofstream idx(path + ".json");
    if (!bin || !idx) throw Error("cannot write " + path);
    idx << "{\"strip_cols\": " << inv.strip_cols << ", \"layout\": \"per stretch: sum f64[n], min f64[n], max f64[n], frame_max u16[n], strips f32[n, n_strips], histogram i64[65536]\", \"stretches\": [";
    for (std::size_t i = 0; i < inv.stretches.size(); ++i) {
        const auto& s = inv.stretches[i];
        const std::size_t n = s.sum.size();
        idx << (i ? ", " : "") << "{\"entry\": \"" << rec.stretches[i].entry << "\", \"n\": " << n << ", \"n_strips\": " << (n ? s.strips.size() / n : 0)
            << ", \"offset\": " << bin.tellp() << "}";
        bin.write(reinterpret_cast<const char*>(s.sum.data()), n * 8);
        bin.write(reinterpret_cast<const char*>(s.min.data()), n * 8);
        bin.write(reinterpret_cast<const char*>(s.max.data()), n * 8);
        bin.write(reinterpret_cast<const char*>(s.frame_max.data()), n * 2);
        bin.write(reinterpret_cast<const char*>(s.strips.data()), s.strips.size() * 4);
        bin.write(reinterpret_cast<const char*>(s.histogram.data()), s.histogram.size() * 8);
    }
    idx << "]}\n";
}

Inventory load_inventory(const Record& rec, int cols, const std::string& path) {
    std::ifstream bin(path + ".bin", std::ios::binary);
    if (!bin) throw Error("cannot read " + path + ".bin");
    Inventory inv;
    const int ns = n_strips(inv, cols);
    inv.stretches.resize(rec.stretches.size());
    for (std::size_t i = 0; i < rec.stretches.size(); ++i) {
        auto& s = inv.stretches[i];
        const std::size_t n = static_cast<std::size_t>(rec.stretches[i].n_frames);
        s.sum.resize(n), s.min.resize(n), s.max.resize(n), s.frame_max.resize(n), s.strips.resize(n * ns), s.histogram.resize(65536);
        bin.read(reinterpret_cast<char*>(s.sum.data()), n * 8);
        bin.read(reinterpret_cast<char*>(s.min.data()), n * 8);
        bin.read(reinterpret_cast<char*>(s.max.data()), n * 8);
        bin.read(reinterpret_cast<char*>(s.frame_max.data()), n * 2);
        bin.read(reinterpret_cast<char*>(s.strips.data()), s.strips.size() * 4);
        bin.read(reinterpret_cast<char*>(s.histogram.data()), s.histogram.size() * 8);
        if (!bin) throw Error(path + ".bin does not match the record");
    }
    return inv;
}

// ------------------------------------------------------------------------------------------------ series

namespace {

// mean of `v` (rows, cols) over blocks of `size` x `size` pixels
std::vector<double> block_means(const std::vector<double>& v, int rows, int cols, int size, int& br, int& bc) {
    br = rows / size, bc = cols / size;
    std::vector<double> out(static_cast<std::size_t>(br) * bc, 0.0);
    for (int i = 0; i < br; ++i)
        for (int j = 0; j < bc; ++j) {
            double s = 0;
            for (int r = i * size; r < (i + 1) * size; ++r)
                for (int c = j * size; c < (j + 1) * size; ++c) s += v[static_cast<std::size_t>(r) * cols + c];
            out[static_cast<std::size_t>(i) * bc + j] = s / (static_cast<double>(size) * size);
        }
    return out;
}

}  // namespace

void check_dark_pair(const Record& rec, const FlatField& ff, const Record& other_rec, const FlatField& other_flat, Report& rep) {
    const Record* const other = &other_rec;
    const FlatField* const other_ff = &other_flat;
    const std::size_t np = static_cast<std::size_t>(ff.rows) * ff.cols;
    const SeriesStats* dark = nullptr;
    for (std::size_t k = 0; k < ff.series.size(); ++k)
        if (rec.stretches[ff.series_stretch[k]].role == Role::Dark && !dark) dark = &ff.series[k];
    if (dark && other && other_ff) {
        const SeriesStats* dark2 = nullptr;
        for (std::size_t k = 0; k < other_ff->series.size(); ++k)
            if (other->stretches[other_ff->series_stretch[k]].role == Role::Dark && !dark2) dark2 = &other_ff->series[k];
        if (dark2) {
            std::vector<double> diff(np);
            long double expect = 0;
            for (std::size_t p = 0; p < np; ++p) {
                diff[p] = dark2->mean[p] - dark->mean[p];
                expect += dark->variance[p] / dark->n + dark2->variance[p] / dark2->n;
            }
            const Spread d = spread(diff);
            const double sigma = std::sqrt(static_cast<double>(expect / np));
            std::int64_t moved = 0;
            for (std::size_t p = 0; p < np; ++p)
                if (std::fabs(diff[p] - d.mean) > 6 * std::sqrt(dark->variance[p] / dark->n + dark2->variance[p] / dark2->n) + 1e-9) ++moved;
            Finding& f = rep.add("8.1.2", rec.label + "+" + other->label, "dark level per pixel in the two scans");
            f.stated = "a dark series at the start of each scan, none at the end";
            f.found = fmt("second minus first: %+.3f counts on average; %.3f counts rms from pixel to pixel where the noise of the two means gives %.3f; %lld pixels changed by more than six times their noise",
                          d.mean, d.sd, sigma, static_cast<long long>(moved));
            f.verdict = "unverified";
            f.note = "The mean shift, excess pixel scatter and changed pixels must be explained by an offset/noise model. Agreement within one count or twice the expected noise does not establish stability; nothing measures the dark offset during a scan";
            f.numbers = {{"mean_difference", d.mean}, {"rms_difference", d.sd}, {"noise_expected", sigma}};
        }
    }
}

void check_series(const Record& rec, Frames& frames, const FlatField& ff, Report& rep, const Record* other, Frames* other_frames, const FlatField* other_ff) {
    const std::string scan = rec.label;
    const int rows = ff.rows, cols = ff.cols;
    const std::size_t np = static_cast<std::size_t>(rows) * cols;
    const int size = 32;
    int br = 0, bc = 0;

    // ---- darks: 4.2, 8.1.1
    const SeriesStats* dark = nullptr;
    std::vector<const SeriesStats*> flats;
    std::vector<std::string> flat_entries;
    for (std::size_t k = 0; k < ff.series.size(); ++k) {
        const Stretch& s = rec.stretches[ff.series_stretch[k]];
        if (s.role == Role::Dark && !dark) dark = &ff.series[k];
        if (s.role == Role::Flat) flats.push_back(&ff.series[k]), flat_entries.push_back(s.entry);
    }
    if (dark) {
        const Spread level = spread(dark->mean);
        long double var = 0;
        for (double v : dark->variance) var += v;
        const double noise = std::sqrt(static_cast<double>(var / np));
        const std::size_t n = dark->frame_mean.size();
        const Spread fm = spread(dark->frame_mean), head = spread(dark->frame_mean, 0, std::min<std::size_t>(50, n)), tail = spread(dark->frame_mean, n - std::min<std::size_t>(50, n), n);
        std::int64_t odd = 0;
        for (double v : dark->mean)
            if (std::fabs(v - level.mean) > 10 * noise) ++odd;
        Finding& f = rep.add("4.2 8.1.1", scan, "dark frames");
        f.stated = "the dark stretch";
        f.found = fmt("%lld frames; level %.3f counts (pixel means %.1f to %.1f, spread %.2f rms); noise of one pixel from frame to frame %.2f counts rms; "
                      "frame means %.3f to %.3f, first 50 frames %.3f, last 50 %.3f; %lld pixels differ from the level by more than ten times the noise",
                      static_cast<long long>(dark->n), level.mean, level.min, level.max, level.sd, noise, fm.min, fm.max, head.mean, tail.mean, static_cast<long long>(odd));
        f.verdict = "unverified";
        f.note = "Low levels support the dark-frame role. The measured temporal and spatial variations still need a detector offset/noise model; no dark was acquired during the projections";
        f.numbers = {{"level", level.mean}, {"noise_rms", noise}, {"odd_pixels", static_cast<double>(odd)}};
    }
    // ---- flats: 4.2, 8.2.2, 8.2.5
    if (dark && !flats.empty()) {
        std::vector<std::string> parts;
        // a projection for comparison: transmission of the first projection frame against the mean flat
        std::vector<double> t(np, 1.0);
        const auto proj = rec.stretches_with(Role::Projection);
        std::string proj_text;
        int shown = -1;
        std::int64_t shown_frame = 0;
        for (int i : proj) {
            const auto held = rec.stretches[i].frames_on_disk();
            if (held.empty()) continue;
            shown = i, shown_frame = held.front().first;
            break;
        }
        if (shown >= 0) {
            std::vector<std::uint16_t> raw(np);
            frames.read(shown, shown_frame, 1, raw.data());
            for (std::size_t p = 0; p < np; ++p) t[p] = (raw[p] - ff.dark[p]) / ff.gain[p];
            const auto bm = block_means(t, rows, cols, size, br, bc);
            const Spread sp = spread(bm);
            std::int64_t low = 0;
            for (double v : bm)
                if (v < 0.9) ++low;
            proj_text = fmt("for comparison, projection %lld of %s over the mean flat: %.3f to %.3f, %.0f %% of the blocks below 0.9", static_cast<long long>(shown_frame),
                            rec.stretches[shown].entry.c_str(), sp.min, sp.max, 100.0 * low / bm.size());
        }
        for (std::size_t k = 0; k < flats.size(); ++k) {
            std::vector<double> ratio(np);
            for (std::size_t p = 0; p < np; ++p) ratio[p] = (flats[k]->mean[p] - dark->mean[p]) / ff.gain[p];
            const auto bm = block_means(ratio, rows, cols, size, br, bc);
            const Spread sp = spread(bm);
            const std::size_t n = flats[k]->frame_mean.size();
            const Spread head = spread(flats[k]->frame_mean, 0, std::min<std::size_t>(20, n)), tail = spread(flats[k]->frame_mean, n - std::min<std::size_t>(20, n), n);
            long double level = 0;
            for (std::size_t p = 0; p < np; ++p) level += flats[k]->mean[p] - dark->mean[p];
            parts.push_back(fmt("%s: mean %.1f counts above dark; over the mean of all flats, in %dx%d blocks: %.4f to %.4f (%.4f rms about %.4f); frame mean changes by %+.2f %% through the series",
                                flat_entries[k].c_str(), static_cast<double>(level / np), size, size, sp.min, sp.max, sp.sd, sp.mean, (tail.mean - head.mean) / (head.mean - spread(dark->mean).mean) * 100));
        }
        Finding& f = rep.add("4.2 8.2.2 8.2.5", scan, "flat frames");
        f.stated = "the flat stretches, taken with the sample moved out sideways";
        f.found = join(parts) + (proj_text.empty() ? "" : "; " + proj_text);
        f.verdict = "unverified";
        f.note = "These are 32x32 block averages, not bounds on every pixel. Differences between and within the series require a time-dependent beam/flat model and residual noise validation; a 10 percent agreement is not verification. Something present in every flat is not excluded";
    }
    // ---- 8.2.3 from the frames, where the record holds no statistics of them
    if (dark && !flats.empty()) {
        std::vector<std::string> parts;
        std::vector<double> per_ma;
        const double dark_level = spread(dark->mean).mean;
        bool recorded = false;
        for (std::size_t k = 0; k < ff.series.size(); ++k) {
            const Stretch& s = rec.stretches[ff.series_stretch[k]];
            if (s.role != Role::Flat) continue;
            recorded |= !s.stat_avg.empty();
            const auto& fm = ff.series[k].frame_mean;
            const std::size_t n = fm.size();
            if (n < 40 || s.monitor.current.size() != n) continue;
            const double level = spread(fm).mean - dark_level, current = spread(s.monitor.current).mean;
            const double l0 = spread(fm, 0, 20).mean - dark_level, l1 = spread(fm, n - 20, n).mean - dark_level;
            const double c0 = spread(s.monitor.current, 0, 20).mean, c1 = spread(s.monitor.current, n - 20, n).mean;
            per_ma.push_back(level / current);
            parts.push_back(fmt("%s: %.1f counts above dark at %.3f mA = %.2f per mA; within the series the level changes by %+.2f %% and the current by %+.2f %% over %.0f s",
                                s.entry.c_str(), level, current, level / current, (l1 / l0 - 1) * 100, (c1 / c0 - 1) * 100, s.epoch.size() == n ? s.epoch.back() - s.epoch.front() : NAN));
        }
        if (!recorded && !parts.empty()) {
            Finding& f = rep.add("8.2.3", scan, "flat level against the ring current (whole frames)");
            f.stated = "ring current read with every flat";
            f.found = join(parts);
            f.verdict = "unverified";
            if (per_ma.size() > 1) {
                const Spread p = spread(per_ma);
                f.found += fmt("; level per mA differs by %.2f %% between the series", (p.max / p.min - 1) * 100);
                f.numbers = {{"per_mA_spread_percent", (p.max / p.min - 1) * 100}};
            }
            for (std::size_t k = 0; k < per_ma.size(); ++k) f.numbers.push_back({"per_mA_" + std::to_string(k), per_ma[k]});
            f.note = "if the ring current were proportional to the beam at the detector, the level per mA would be one number";
        }
    }
    // ---- 8.1.2: the dark of the other scan
    if (other && other_ff) check_dark_pair(rec, ff, *other, *other_ff, rep);
    (void)other_frames;
}

// ------------------------------------------------------------------------------------------------ beam

namespace {

// linear interpolation of the monitor's reading at time t; NaN readings are skipped
double monitor_at(const Monitor& m, double t, double* derivative = nullptr) {
    if (derivative) *derivative = 0;
    const std::size_t n = std::min(m.epoch.size(), m.current.size());
    if (!n) return NAN;
    std::size_t hi = std::lower_bound(m.epoch.begin(), m.epoch.begin() + n, t) - m.epoch.begin();
    std::size_t lo = hi ? hi - 1 : 0;
    while (hi < n && !std::isfinite(m.current[hi])) ++hi;
    while (lo > 0 && !std::isfinite(m.current[lo])) --lo;
    if (hi >= n) return m.current[lo];
    if (!std::isfinite(m.current[lo]) || hi == lo || m.epoch[hi] == m.epoch[lo]) return m.current[hi];
    const double w = std::clamp((t - m.epoch[lo]) / (m.epoch[hi] - m.epoch[lo]), 0.0, 1.0);
    if (derivative && t >= m.epoch[lo] && t <= m.epoch[hi])
        *derivative = (m.current[hi]-m.current[lo])/(m.epoch[hi]-m.epoch[lo]);
    return m.current[lo] + w * (m.current[hi] - m.current[lo]);
}

struct MonitorResidual final : ceres::SizedCostFunction<1,3> {
    const Monitor& monitor;
    double epoch, level, current_center, level_center;
    MonitorResidual(const Monitor& m, double t, double y, double mx, double my)
        : monitor(m), epoch(t), level(y), current_center(mx), level_center(my) {}
    bool Evaluate(double const* const* parameters, double* residual, double** jacobian) const override {
        const double* p = parameters[0];  // delay, slope, centred offset
        double derivative;
        const double current = monitor_at(monitor,epoch+p[0],&derivative)-current_center;
        residual[0] = (level-level_center)-(p[1]*current+p[2]);
        if (jacobian && jacobian[0]) {
            jacobian[0][0] = -p[1]*derivative;
            jacobian[0][1] = -current;
            jacobian[0][2] = -1;
        }
        return std::isfinite(residual[0]);
    }
};

}  // namespace

void check_beam(const Record& rec, const Inventory& inv, const FlatField& ff, Report& rep) {
    const std::string scan = rec.label;
    const int cols = ff.cols;
    const int ns = n_strips(inv, cols);
    // mean ring current over the flat frames: the reference the mean flat belongs to
    double flat_current = 0, flat_frames = 0;
    for (int i : rec.stretches_with(Role::Flat)) {
        const Stretch& s = rec.stretches[i];
        for (double c : s.monitor.current)
            if (std::isfinite(c)) flat_current += c, flat_frames += 1;
    }
    flat_current = flat_frames ? flat_current / flat_frames : NAN;

    for (int si : rec.stretches_with(Role::Projection)) {
        const Stretch& s = rec.stretches[si];
        const auto& data = inv.stretches.at(si);
        const Rect rect = stat_rect(s, ff.rows, cols);
        // the frames that were read: every frame of the stretch, or what a partial copy or a sparse inventory holds.
        // Below, frame i is the i-th of them, frame number at[i] of the stretch.
        if (data.strips.size() != static_cast<std::size_t>(s.n_frames) * ns) continue;
        std::vector<std::size_t> at;
        for (std::size_t fr = 0; fr < static_cast<std::size_t>(s.n_frames); ++fr)
            if (data.was_read(fr)) at.push_back(fr);
        const std::size_t n = at.size();
        if (n < 8) continue;
        const bool sparse = n != static_cast<std::size_t>(s.n_frames);
        std::vector<float> packed;
        const float* strips = data.strips.data();
        if (sparse) {
            packed.resize(n * ns);
            for (std::size_t i = 0; i < n; ++i) std::copy_n(data.strips.data() + at[i] * ns, ns, packed.data() + i * ns);
            strips = packed.data();
        }
        const bool timed = s.epoch_trig.size() == static_cast<std::size_t>(s.n_frames);
        std::vector<double> times;
        if (timed && sparse)
            for (std::size_t fr : at) times.push_back(s.epoch_trig[fr]);
        const double* const epoch = !timed ? nullptr : sparse ? times.data() : s.epoch_trig.data();
        // dark and gain of each strip over the statistics rectangle
        std::vector<double> dark(ns, 0), gain(ns, 0);
        for (int k = 0; k < ns; ++k) {
            const int a = std::max(rect.x, k * inv.strip_cols), b = std::min(rect.x + rect.w, (k + 1) * inv.strip_cols);
            double d = 0, g = 0;
            for (int r = rect.y; r < rect.y + rect.h; ++r)
                for (int c = a; c < b; ++c) d += ff.dark[static_cast<std::size_t>(r) * cols + c], g += ff.gain[static_cast<std::size_t>(r) * cols + c];
            const double count = static_cast<double>(std::max(1, b - a)) * rect.h;
            dark[k] = d / count, gain[k] = g / count;
        }
        // strips that stay in air through the whole stretch: in every frame within 5 % of the frame's brightest
        // strip (the beam's level may change through the stretch; the sample's edge absorbs far more than that)
        std::vector<double> brightest(n, 0.0);
        for (std::size_t f = 0; f < n; ++f)
            for (int k = 0; k < ns; ++k) brightest[f] = std::max(brightest[f], (strips[f * ns + k] - dark[k]) / gain[k]);
        std::vector<int> air;
        for (int k = 0; k < ns; ++k) {
            bool in_air = true;
            for (std::size_t f = 0; f < n && in_air; ++f) in_air = (strips[f * ns + k] - dark[k]) / gain[k] >= 0.95 * brightest[f];
            if (in_air) air.push_back(k);
        }
        Finding& f = rep.add("4.7 8.2.1 8.2.3 8.2.4 8.2.5", scan, "level of the air beside the sample, stretch " + s.entry);
        f.stated = fmt("flats before and after the scan (mean ring current %.3f mA) and the ring current through the stretch", flat_current);
        if (sparse) f.stated += fmt("; %zu frames between %zu and %zu, the ones that were read", n, at.front(), at.back());
        const Spread bright = spread(brightest);
        if (air.empty() || bright.mean < 0.75) {
            // a strip in air reads about 1 over the flat. Where even the brightest strip stays far below it,
            // every ray passes through the sample: there is no air to take the beam's level from.
            f.found = air.empty() ? "no column strip stays in air through the stretch" : std::string("no column is air");
            f.found += fmt("; the brightest strip of a frame reads %.3f of the flat on average (%.3f to %.3f)", bright.mean, bright.min, bright.max);
            if (bright.mean < 0.75) f.found += ": every ray passes through the sample, and the level of a frame cannot be taken from the frame";
            f.verdict = "unverified";
            f.numbers = {{"brightest_mean", bright.mean}, {"brightest_min", bright.min}, {"brightest_max", bright.max}};
            continue;
        }
        std::vector<double> level(n), current(n), ratio(n), index(n);
        for (std::size_t fr = 0; fr < n; ++fr) {
            double t = 0;
            for (int k : air) t += (strips[fr * ns + k] - dark[k]) / gain[k];
            level[fr] = t / air.size();
            current[fr] = timed ? monitor_at(s.monitor, epoch[fr]) : NAN;
            ratio[fr] = level[fr] / (current[fr] / flat_current);
            index[fr] = static_cast<double>(fr);
        }
        const Spread lv = spread(level), rt = spread(ratio);
        // frame-to-frame scatter: rms of the first difference / sqrt(2)
        long double dd = 0;
        for (std::size_t fr = 1; fr < n; ++fr) dd += (level[fr] - level[fr - 1]) * (level[fr] - level[fr - 1]);
        const double jitter = std::sqrt(static_cast<double>(dd / (n - 1)) / 2);
        const std::size_t m = std::min<std::size_t>(200, n / 4);
        const double l0 = spread(level, 0, m).mean, l1 = spread(level, n - m, n).mean, c0 = spread(current, 0, m).mean, c1 = spread(current, n - m, n).mean;
        f.found = fmt("air = columns %d..%d of the statistics rows (%zu strips of %d columns that stay within 5 %% of the brightest strip in every frame); level over the mean flat %.4f on average, %.4f to %.4f; "
                      "from frame to frame %.4f rms; first %zu frames %.4f, last %.4f (%+.2f %%) while the ring current goes %.3f -> %.3f mA (%+.2f %%); "
                      "air level over (current / current of the flats): %.4f on average, %.4f to %.4f",
                      air.front() * inv.strip_cols, std::min(cols, (air.back() + 1) * inv.strip_cols) - 1, air.size(), inv.strip_cols, lv.mean, lv.min, lv.max, jitter, m, l0, l1,
                      (l1 / l0 - 1) * 100, c0, c1, (c1 / c0 - 1) * 100, rt.mean, rt.min, rt.max);
        f.numbers = {{"air_mean", lv.mean}, {"air_min", lv.min}, {"air_max", lv.max}, {"air_over_current_mean", rt.mean}, {"air_over_current_min", rt.min},
                     {"air_over_current_max", rt.max}, {"air_strips", static_cast<double>(air.size())}};
        // a refill: where the ring current rises by more than 1 mA, the frames in which the air level steps up,
        // and the delay of the monitor that explains the step best
        std::size_t rise0 = n, rise1 = 0;
        if (timed) {
            const Monitor& mon = s.monitor;
            double low = INFINITY, t0 = NAN, t1 = NAN;
            for (std::size_t k = 0; k < mon.current.size() && k < mon.epoch.size(); ++k) {
                if (!std::isfinite(mon.current[k])) continue;
                if (mon.current[k] > low + 1.0) {
                    if (!std::isfinite(t0)) t0 = mon.epoch[k];
                    t1 = mon.epoch[k];
                } else if (!std::isfinite(t0)) {
                    low = std::min(low, mon.current[k]);
                }
            }
            if (std::isfinite(t0)) {
                // from 30 s before the current first exceeds its minimum by 1 mA to 30 s after it stops rising
                double last_rise = t0, prev = NAN;
                for (std::size_t k = 0; k < mon.current.size() && k < mon.epoch.size(); ++k) {
                    if (!std::isfinite(mon.current[k])) continue;
                    if (mon.epoch[k] >= t0 && std::isfinite(prev) && mon.current[k] > prev + 0.05) last_rise = mon.epoch[k];
                    prev = mon.current[k];
                }
                rise0 = static_cast<std::size_t>(std::lower_bound(epoch, epoch + n, t0 - 30.0) - epoch);
                rise1 = std::min(n - 1, static_cast<std::size_t>(std::lower_bound(epoch, epoch + n, last_rise + 30.0) - epoch));
                (void)t1;
            }
        }
        if (rise0 < rise1 && timed) {
            // narrow to where the level actually changes: first and last frame at 10 % and 90 % of the step
            const double low = spread(level, rise0 > 100 ? rise0 - 100 : 0, rise0 + 1).mean, top = spread(level, rise1, std::min(n, rise1 + 100)).mean;
            std::size_t a = rise0, b = rise1;
            while (a < rise1 && level[a] < low + 0.10 * (top - low)) ++a;
            while (b > a && level[b] > low + 0.90 * (top - low)) --b;
            // delay: the level is modelled as alpha * current(t - delay) + beta over the frames around the rise
            const std::size_t w0 = a > 300 ? a - 300 : 0, w1 = std::min(n, b + 300);
            double best_delay = NAN, best_rms = 1e30, best_alpha = NAN;
            FitSummary best_solver;
            const double mx = spread(current,w0,w1).mean, my = spread(level,w0,w1).mean;
            // Multiple physical starting delays; continuous fitting is entirely Ceres.
            for (double delay : {-10.,-5.,0.,5.,10.}) {
                std::vector<double> x(w1-w0), y(w1-w0);
                for (std::size_t fr = w0; fr < w1; ++fr) x[fr-w0] = monitor_at(s.monitor,epoch[fr]+delay), y[fr-w0] = level[fr];
                const auto initial = fit_line(x,y,0,x.size());
                if (!initial.usable) continue;
                double p[3] = {delay,initial.slope,initial.intercept+initial.slope*mx-my};
                ceres::Problem problem;
                for (std::size_t fr = w0; fr < w1; ++fr)
                    problem.AddResidualBlock(new MonitorResidual(s.monitor,epoch[fr],level[fr],mx,my),nullptr,p);
                problem.SetParameterLowerBound(p,0,-10);
                problem.SetParameterUpperBound(p,0,10);
                ceres::Solver::Summary summary;
                ceres::Solve(fitting::options(3),&problem,&summary);
                const double rms = std::sqrt(2*summary.final_cost/(w1-w0));
                if (summary.IsSolutionUsable() && rms < best_rms) {
                    best_rms = rms; best_delay = p[0]; best_alpha = p[1]; best_solver = fitting::summary(summary);
                }
            }
            if (!best_solver.usable) {
                f.found += "; monitor-delay fit unavailable (insufficient variation or unusable fit)";
                continue;
            }
            f.found += fmt("; around the refill the air level rises by %+.1f %% between projections %zu and %zu (10 %% and 90 %% of the step); modelled as a multiple of the ring current read %.2f s %s the frame "
                           "(residual %.4f rms, %.2f %% of the level per %% of current)",
                           (top / low - 1) * 100, at[a], at[b], std::fabs(best_delay), best_delay >= 0 ? "after" : "before", best_rms, best_alpha * spread(current, w0, w1).mean / spread(level, w0, w1).mean);
            f.numbers.push_back({"rise_first_frame", static_cast<double>(at[a])});
            f.numbers.push_back({"rise_last_frame", static_cast<double>(at[b])});
            f.numbers.push_back({"monitor_delay_s", best_delay});
            f.found += "; Ceres " + best_solver.solver_version + " " + best_solver.termination;
        }
        f.note = "air is found from the frames themselves; nothing here assumes where the sample is. The flat used is the mean of all the scan's flats";
    }
}

}  // namespace helirec::calibration
