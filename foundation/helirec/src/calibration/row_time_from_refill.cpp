// helirec/calibration/geometry_calibration.hpp: the time of a detector row from the beam dips of a refill.
//
// Each injection shot dips the beam for a few milliseconds, the shots sit on a raster, and which rows of which
// frame record a shot depends on the row time through the frame period and the gate only. Here the stretch
// with a refill is found, the columns in air around it, and every row's level in the frames around it; the
// dips are fitted by calibration::fit_refill_shots (helirec/calibration/refill.hpp).
#include <algorithm>
#include <cmath>
#include <limits>

#include "calibration/util.hpp"
#include "helirec/calibration/geometry_calibration.hpp"
#include "helirec/calibration/refill.hpp"
#include "io/json.hpp"

namespace helirec::calibration::geometry_calibration {

namespace {

// a stretch that may hold a refill: its scan, the record's stretch, its name, and the rows kept out of it
struct Candidate {
    int scan = 0, stretch = 0;
    std::string name;
    const Blocked* blocked = nullptr;
};

std::string row_time_of(Session& session, int n_rows, int n_cols, const std::vector<Candidate>& candidates, const Log& log) {
    const std::size_t rows = static_cast<std::size_t>(n_rows), cols = static_cast<std::size_t>(n_cols);
    for (const Candidate& candidate : candidates) {
        {
            const int scan = candidate.scan, st = candidate.stretch;
            const Stretch& stretch = session.record(scan).stretches[static_cast<std::size_t>(st)];
            std::vector<double> cur, when;
            for (std::size_t i = 0; i < std::min(stretch.monitor.current.size(), stretch.monitor.epoch.size()); ++i)
                if (std::isfinite(stretch.monitor.current[i]) && std::isfinite(stretch.monitor.epoch[i])) cur.push_back(stretch.monitor.current[i]), when.push_back(stretch.monitor.epoch[i]);
            if (cur.size() < 8 || *std::max_element(cur.begin(), cur.end()) - cur[0] < 0.02 * cur[0]) continue;
            const std::vector<double>& epoch = stretch.epoch_trig;
            const std::size_t lo_i = static_cast<std::size_t>(std::min_element(cur.begin(), cur.end()) - cur.begin());
            const std::size_t hi_i = static_cast<std::size_t>(std::max_element(cur.begin() + static_cast<std::ptrdiff_t>(lo_i), cur.end()) - cur.begin());
            const double t_lo = when[lo_i], t_hi = when[hi_i];
            const long f_lo = std::max(20L, static_cast<long>(std::lower_bound(epoch.begin(), epoch.end(), t_lo - 8.0) - epoch.begin()));
            const long f_hi = std::min(static_cast<long>(epoch.size()) - 20, static_cast<long>(std::lower_bound(epoch.begin(), epoch.end(), t_hi + 8.0) - epoch.begin()));
            if (f_hi - f_lo < 100) continue;
            const double T = median_of(stretch.timer_period), E = median_of(stretch.timer_delta);
            if (!stretch.on_disk(std::max(20L, f_lo - 400)) || !stretch.on_disk(f_hi + 15)) continue;  // a partial copy of the acquisition
            // columns that are air in every frame around the refill
            std::vector<std::uint8_t> free_row(rows, 1);
            if (candidate.blocked)
                for (std::size_t i = 0; i < candidate.blocked->frames.size(); ++i)
                    for (std::size_t r = 0; r < rows; ++r)
                        if (candidate.blocked->mask[i * rows + r]) free_row[r] = 0;
            const int r_lo = n_rows / 4, r_hi = 3 * n_rows / 4;
            int n_free = 0;
            for (int r = r_lo; r < r_hi; ++r) n_free += free_row[static_cast<std::size_t>(r)];
            if (!n_free) continue;
            std::vector<double> peak(cols, 0.0);
            Frames& frames = session.frames(scan);
            const FlatField& ff = session.flat_field(scan);
            for (const double fd : linspace(static_cast<double>(f_lo), static_cast<double>(f_hi), 24)) {
                const std::int64_t f = static_cast<std::int64_t>(fd);
                const std::int32_t r0 = r_lo;
                std::vector<std::uint16_t> raw(static_cast<std::size_t>(r_hi - r_lo) * cols);
                frames.read_rows(st, &f, &r0, 1, r_hi - r_lo, raw.data());
                std::vector<float> p(static_cast<std::size_t>(r_hi - r_lo) * cols);
                line_integrals(ff, raw.data(), &r0, 1, r_hi - r_lo, 0, n_cols, false, p.data());
                for (std::size_t c = 0; c < cols; ++c) {
                    float s = 0;
                    for (int r = r_lo; r < r_hi; ++r)
                        if (free_row[static_cast<std::size_t>(r)]) s += p[static_cast<std::size_t>(r - r_lo) * cols + c];
                    peak[c] = std::max(peak[c], static_cast<double>(s / static_cast<float>(n_free)));
                }
            }
            std::vector<double> sorted = peak;
            std::sort(sorted.begin(), sorted.end());
            const double pos = 0.02 * static_cast<double>(cols - 1);
            const std::size_t k = static_cast<std::size_t>(std::floor(pos));
            const double low = sorted[k] + (pos - static_cast<double>(k)) * (sorted[std::min(k + 1, cols - 1)] - sorted[k]);
            // the widest run of columns in air
            long c0 = -1, c1 = -1;
            for (std::size_t c = 0; c < cols;) {
                if (!(peak[c] < low + 0.03)) {
                    ++c;
                    continue;
                }
                std::size_t e = c;
                while (e < cols && peak[e] < low + 0.03) ++e;
                if (c0 < 0 || static_cast<long>(e - c) >= c1 - c0) c0 = static_cast<long>(c), c1 = static_cast<long>(e);  // of equal runs the later one
                c = e;
            }
            if (c0 < 0) continue;
            c0 += 8, c1 -= 8;
            if (c1 - c0 < 16) continue;
            std::vector<double> dark(rows);
            for (std::size_t r = 0; r < rows; ++r) {
                float s = 0;
                for (long c = c0; c < c1; ++c) s += ff.dark[r * cols + static_cast<std::size_t>(c)];
                dark[r] = static_cast<double>(s / static_cast<float>(c1 - c0));
            }
            const int pre = static_cast<int>(std::min(400L, f_lo - 20));  // frames before the refill: what a row band scatters by without a shot
            const long first = f_lo - pre, n_frames = f_hi + 16 - first;
            std::vector<double> x(static_cast<std::size_t>(n_frames) * rows);
            std::vector<std::uint16_t> raw(64 * rows * cols);
            for (long i0 = 0; i0 < n_frames; i0 += 64) {
                const std::size_t n = static_cast<std::size_t>(std::min(64L, n_frames - i0));
                std::vector<std::int64_t> fr(n);
                std::vector<std::int32_t> r0(n, 0);
                for (std::size_t i = 0; i < n; ++i) fr[i] = first + i0 + static_cast<long>(i);
                frames.read_rows(st, fr.data(), r0.data(), n, n_rows, raw.data());
                const std::ptrdiff_t todo = static_cast<std::ptrdiff_t>(n * rows);
#pragma omp parallel for schedule(static)
                for (std::ptrdiff_t q = 0; q < todo; ++q) {
                    const std::uint16_t* line = raw.data() + static_cast<std::size_t>(q) * cols;
                    double s = 0;
                    for (long c = c0; c < c1; ++c) s += line[c];
                    x[static_cast<std::size_t>(i0) * rows + static_cast<std::size_t>(q)] = std::log(std::max(s / static_cast<double>(c1 - c0) - dark[static_cast<std::size_t>(q) % rows], 1.0));
                }
            }
            const RefillSolution got = fit_refill_shots(x.data(), static_cast<std::size_t>(n_frames), n_rows, T, E, pre);
            if (!got.usable) continue;
            if (log)
                log(fmt("  refill in scan %d %s, frames %ld..%ld: %zu shots on a raster of %.3f ms, air columns %ld..%ld; row time %+.3f +- %.3f us per stored row", scan, candidate.name.c_str(), f_lo,
                        f_hi, got.shots.size(), got.x[3] * 1e3, c0, c1, got.x[0] * 1e6, got.row_time_se * 1e6));
            Json d = Json::dict();
            Json solver = Json::dict();
            solver.set("solver_version", Json::of(got.solver_version)).set("termination", Json::of(got.termination)).set("message", Json::of(got.message));
            solver.set("cost", Json::of(got.cost)).set("iterations", Json::whole_number(got.iterations));
            d.set("seconds_per_row", Json::of(got.x[0])).set("se", got.uncertainty_available ? Json::of(got.row_time_se) : Json()).set("solver", solver);
            d.set("scan", Json::whole_number(scan)).set("stretch", Json::of(candidate.name));
            d.set("frames", Json::list().add(Json::whole_number(f_lo)).add(Json::whole_number(f_hi))).set("shots", Json::whole_number(static_cast<long long>(got.shots.size())));
            d.set("raster_s", Json::of(got.x[3])).set("air_columns", Json::list().add(Json::whole_number(c0)).add(Json::whole_number(c1)));
            d.set("period_s", Json::of(T)).set("gate_s", Json::of(E));
            return d.text();
        }
    }
    return "null";
}

}  // namespace

std::string row_time_from_refill(Session& session, const Acquisition& acq, const Log& log) {
    std::vector<Candidate> candidates;
    for (int scan = 0; scan < static_cast<int>(acq.scans.size()); ++scan)
        for (const bool helix : {true, false})  // the helix first
            for (const StretchLayout& st : acq.scans[static_cast<std::size_t>(scan)].stretches)
                if (st.helix == helix) candidates.push_back({scan, st.index, st.name, &st.blocked});
    return row_time_of(session, acq.rows, acq.cols, candidates, log);
}

}  // namespace helirec::calibration::geometry_calibration
