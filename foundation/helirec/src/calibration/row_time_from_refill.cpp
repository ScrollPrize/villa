// helirec/calibration/geometry_calibration.hpp: the time of a detector row from the beam dips of a refill.
//
// Each injection shot dips the beam for a few milliseconds, the shots sit on a raster, and which rows of which
// frame record a shot depends on the row time through the frame period and the gate only.
#include <algorithm>
#include <cmath>
#include <limits>

#include "calibration/util.hpp"
#include "helirec/calibration/geometry_calibration.hpp"
#include "helirec/calibration/solve.hpp"
#include "io/json.hpp"

namespace helirec::calibration::geometry_calibration {

namespace {

// Least squares of a model that is not a line in its parameters, by damped Gauss-Newton steps (Levenberg and
// Marquardt) in parameters scaled by `scale`, each step a linear least squares through the steps' solver.
// residual(p, r) fills r (always the same length). Returns the sum of squares at the end; p is updated.
template <class Residual>
double minimise(std::vector<double>& p, const std::vector<double>& scale, std::size_t m, const Residual& residual, std::vector<double>* jacobian = nullptr) {
    const std::size_t n = p.size();
    std::vector<double> r(m), r2(m), J(m * n), trial(n);
    const auto sum_sq = [](const std::vector<double>& v) {
        double s = 0;
        for (double x : v) s += x * x;
        return s;
    };
    const auto derivative = [&]() {  // forward differences, the step a fixed fraction of the parameter's size or of 1
        const double rel = std::sqrt(std::numeric_limits<double>::epsilon());
        for (std::size_t k = 0; k < n; ++k) {
            const double h = rel * (p[k] >= 0 ? 1.0 : -1.0) * std::max(1.0, std::fabs(p[k]));
            trial = p;
            trial[k] += h;
            residual(trial, r2);
            for (std::size_t i = 0; i < m; ++i) J[i * n + k] = (r2[i] - r[i]) / h;
        }
    };
    residual(p, r);
    double cost = sum_sq(r), lambda = 1e-3;
    for (int it = 0; it < 200; ++it) {
        derivative();
        bool stepped = false;
        double gain = 0, moved = 0;
        for (int attempt = 0; attempt < 12 && !stepped; ++attempt) {
            // [J S; sqrt(lambda) D] d = [-r; 0], D the columns' lengths (the damping does not depend on the units)
            LinearProblem lp;
            lp.unknowns = n;
            lp.y.assign(m + n, 0.0);
            std::vector<double> len(n, 0.0);
            for (std::size_t i = 0; i < m; ++i) {
                lp.y[i] = -r[i];
                for (std::size_t k = 0; k < n; ++k) {
                    const double v = J[i * n + k] * scale[k];
                    if (v == 0) continue;
                    lp.row.push_back(static_cast<std::int64_t>(i)), lp.col.push_back(static_cast<std::int64_t>(k)), lp.value.push_back(v);
                    len[k] += v * v;
                }
            }
            for (std::size_t k = 0; k < n; ++k) {
                const double d = std::sqrt(lambda * len[k]);
                if (d == 0) continue;
                lp.row.push_back(static_cast<std::int64_t>(m + k)), lp.col.push_back(static_cast<std::int64_t>(k)), lp.value.push_back(d);
            }
            const LinearSolution s = solve_linear(lp);
            moved = 0;
            for (std::size_t k = 0; k < n; ++k) trial[k] = p[k] + s.x[k] * scale[k], moved = std::max(moved, std::fabs(s.x[k]));
            residual(trial, r2);
            const double c2 = sum_sq(r2);
            if (c2 < cost) {
                gain = (cost - c2) / cost;
                p = trial, r = r2, cost = c2;
                lambda = std::max(lambda / 5, 1e-12);
                stepped = true;
            } else {
                lambda *= 4;
            }
        }
        if (!stepped || (gain < 1e-12 && moved < 1e-6)) break;
    }
    if (jacobian) {
        derivative();
        *jacobian = J;
    }
    return cost;
}

}  // namespace

std::optional<RefillFit> fit_refill_shots(const std::vector<double>& x, std::size_t frames, int n_rows, double T, double E, int pre) {
    const std::size_t R = static_cast<std::size_t>(n_rows);
    if (x.size() != frames * R) throw Error("fit_refill_shots: the levels are not (frames, rows)");
    const int B = 20, nb = n_rows / B;
    std::vector<double> vc(static_cast<std::size_t>(nb)), u(static_cast<std::size_t>(nb));
    for (int b = 0; b < nb; ++b) {
        vc[static_cast<std::size_t>(b)] = (b + 0.5) * B - 0.5;
        u[static_cast<std::size_t>(b)] = (vc[static_cast<std::size_t>(b)] - (n_rows - 1) / 2.0) / (n_rows / 2.0);
    }
    const long len = static_cast<long>(frames);
    // a frame's rows against the same rows some frames before and after
    const auto dip = [&](long n) {
        std::vector<double> d(R);
        for (std::size_t r = 0; r < R; ++r) {
            double before = 0, after = 0;
            for (long k = n - 7; k < n - 2; ++k) before += x[static_cast<std::size_t>(k) * R + r];
            for (long k = n + 4; k < n + 9; ++k) after += x[static_cast<std::size_t>(k) * R + r];
            before /= 5, after /= 5;
            d[r] = x[static_cast<std::size_t>(n) * R + r] - (before + (after - before) * (4.5 / 10.5));
        }
        return d;
    };
    const auto level = [&](long n) { return mean_of(dip(n)); };
    std::vector<int> shots;
    for (long n = std::max(16L, static_cast<long>(pre) - 8); n < len - 16; ++n) {
        const double d = level(n);
        if (!(d < -0.003)) continue;
        if (!shots.empty() && n - shots.back() <= 2) {
            if (d < level(shots.back())) shots.back() = static_cast<int>(n);
        } else {
            shots.push_back(static_cast<int>(n));
        }
    }
    const std::size_t S = shots.size();
    if (S < 12) return std::nullopt;
    const auto band = [&](long n) {
        const std::vector<double> d = dip(n);
        std::vector<double> out(static_cast<std::size_t>(nb));
        for (int b = 0; b < nb; ++b) {
            double s = 0;
            for (int k = 0; k < B; ++k) s += d[static_cast<std::size_t>(b * B + k)];
            out[static_cast<std::size_t>(b)] = s / B;
        }
        return out;
    };
    const std::size_t NB = static_cast<std::size_t>(nb);
    std::vector<double> data(S * 3 * NB);
    for (std::size_t s = 0; s < S; ++s)
        for (int o = -1; o <= 1; ++o) {
            const std::vector<double> v = band(shots[s] + o);
            std::copy(v.begin(), v.end(), data.begin() + static_cast<std::ptrdiff_t>((s * 3 + static_cast<std::size_t>(o + 1)) * NB));
        }
    // what a band of rows scatters by without a shot
    std::vector<double> quiet;
    const auto add_quiet = [&](long n) {
        const std::vector<double> v = band(n);
        quiet.insert(quiet.end(), v.begin(), v.end());
    };
    if (pre > 80) {
        for (long n = 16; n < pre - 16; ++n) add_quiet(n);
    } else {
        for (long n = 16; n < len - 16; ++n) {
            long nearest = std::numeric_limits<long>::max();
            for (const int s : shots) nearest = std::min(nearest, std::labs(static_cast<long>(s) - n));
            if (nearest > 3) add_quiet(n);
        }
    }
    const double noise = sd_of(quiet);
    // the raster: the gaps between neighbouring shots are whole steps of it (a shot may be missing)
    std::vector<double> t(S), j(S, 0.0);
    for (std::size_t s = 0; s < S; ++s) t[s] = (shots[s] - shots[0]) * T;
    double least = std::numeric_limits<double>::max();
    for (std::size_t s = 1; s < S; ++s) least = std::min(least, t[s] - t[s - 1]);
    double raster = 0;
    {
        double sum = 0;
        int count = 0;
        for (std::size_t s = 1; s < S; ++s)
            if (t[s] - t[s - 1] < 1.5 * least) sum += t[s] - t[s - 1], ++count;
        raster = sum / count;
    }
    for (std::size_t s = 1; s < S; ++s) j[s] = j[s - 1] + std::nearbyint((t[s] - t[s - 1]) / raster);
    raster = fit_line(j, t, 0, S).slope;

    // the dips: a shot at t0 + j raster takes the beam down for its decay theta; row v of a frame is open from its
    // start + tau v for the gate E
    const std::size_t M = S * 3 * NB;
    const auto model = [&](const std::vector<double>& p, std::vector<double>& out) {
        const double tau = p[0], theta = std::fabs(p[1]), t0 = p[2], ras = p[3];
        const auto P = [&](double z) { return z > 0 ? 1 - std::exp(-z / theta) : 0.0; };
        for (std::size_t s = 0; s < S; ++s)
            for (int o = -1; o <= 1; ++o)
                for (std::size_t b = 0; b < NB; ++b) {
                    const double alpha = 1 + p[4] * u[b] + p[5] * u[b] * u[b] + p[6] * u[b] * u[b] * u[b] + p[7] * u[b] * u[b] * u[b] * u[b];
                    const double z = (shots[s] - shots[0] + o) * T + tau * vc[b] - (t0 + j[s] * ras);
                    out[(s * 3 + static_cast<std::size_t>(o + 1)) * NB + b] = -p[8 + s] * alpha * (P(z + E) - P(z));
                }
    };
    const auto residual = [&](const std::vector<double>& p, std::vector<double>& r) {
        model(p, r);
        for (std::size_t i = 0; i < M; ++i) r[i] = (r[i] - data[i]) / noise;
    };
    std::vector<double> scale = {1e-6, 1e-3, 1e-3, 1e-5, .1, .1, .1, .1};
    scale.resize(8 + S, 1e-3);
    std::optional<RefillFit> best;
    std::vector<double> r(M);
    for (const double tau0 : {-0.6 * T / n_rows, 0.6 * T / n_rows}) {
        // where the first shot lies in its frame is not known: the start that fits best of a ladder of them
        std::vector<double> start;
        double start_cost = 0;
        const int steps = static_cast<int>(std::ceil((2.6 * T) / (T / 16) - 1e-9));
        for (int i = 0; i < steps; ++i) {
            std::vector<double> p = {tau0, 0.003, -1.3 * T + i * (T / 16), raster, 0, 0, 0, 0};
            p.resize(8 + S, 0.009);
            model(p, r);
            double c = 0;
            for (std::size_t q = 0; q < M; ++q) c += (r[q] - data[q]) * (r[q] - data[q]);
            if (start.empty() || c < start_cost) start = p, start_cost = c;
        }
        std::vector<double> J;
        const double cost = minimise(start, scale, M, residual, &J);
        if (best && !(cost < best->cost)) continue;
        // the row time's standard error: the first diagonal element of (J^T J)^-1 times the residuals' variance
        const std::size_t n = start.size();
        LinearProblem lp;
        lp.unknowns = n;
        lp.y.assign(n, 0.0);
        lp.y[0] = 1.0;
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t b = 0; b < n; ++b) {
                double s = 0;
                for (std::size_t i = 0; i < M; ++i) s += J[i * n + a] * J[i * n + b];
                if (s != 0) lp.row.push_back(static_cast<std::int64_t>(a)), lp.col.push_back(static_cast<std::int64_t>(b)), lp.value.push_back(s);
            }
        const LinearSolution inv = solve_linear(lp);
        RefillFit f;
        f.p = start, f.cost = cost, f.noise = noise, f.shots = shots;
        f.se = std::sqrt(std::max(0.0, inv.x[0]) * cost / static_cast<double>(M - n));
        best = f;
    }
    return best;
}

std::string row_time_from_refill(Session& session, const Layout& lay, const Log& log) {
    const std::size_t rows = static_cast<std::size_t>(lay.rows), cols = static_cast<std::size_t>(lay.cols);
    for (int scan = 0; scan < 2; ++scan)
        for (const int a : {Helix, Rest1, Rest2}) {
            const int st = lay.stretch[scan][a];
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
            // columns that are air in every frame around the refill
            std::vector<std::uint8_t> free_row(rows, 1);
            if (const auto it = lay.blocked.find({scan, a}); it != lay.blocked.end())
                for (std::size_t i = 0; i < it->second.frames.size(); ++i)
                    for (std::size_t r = 0; r < rows; ++r)
                        if (it->second.mask[i * rows + r]) free_row[r] = 0;
            const int r_lo = lay.rows / 4, r_hi = 3 * lay.rows / 4;
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
                line_integrals(ff, raw.data(), &r0, 1, r_hi - r_lo, 0, lay.cols, false, p.data());
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
                frames.read_rows(st, fr.data(), r0.data(), n, lay.rows, raw.data());
                const std::ptrdiff_t todo = static_cast<std::ptrdiff_t>(n * rows);
#pragma omp parallel for schedule(static)
                for (std::ptrdiff_t q = 0; q < todo; ++q) {
                    const std::uint16_t* line = raw.data() + static_cast<std::size_t>(q) * cols;
                    double s = 0;
                    for (long c = c0; c < c1; ++c) s += line[c];
                    x[static_cast<std::size_t>(i0) * rows + static_cast<std::size_t>(q)] = std::log(std::max(s / static_cast<double>(c1 - c0) - dark[static_cast<std::size_t>(q) % rows], 1.0));
                }
            }
            const std::optional<RefillFit> got = fit_refill_shots(x, static_cast<std::size_t>(n_frames), lay.rows, T, E, pre);
            if (!got) continue;
            if (log)
                log(fmt("  refill in scan %d %s, frames %ld..%ld: %zu shots on a raster of %.3f ms, air columns %ld..%ld; row time %+.3f +- %.3f us per stored row", scan, kNames[a], f_lo,
                        f_hi, got->shots.size(), got->p[3] * 1e3, c0, c1, got->p[0] * 1e6, got->se * 1e6));
            Json d = Json::dict();
            d.set("seconds_per_row", Json::of(got->p[0])).set("se", Json::of(got->se)).set("scan", Json::whole_number(scan)).set("stretch", Json::of(std::string(kNames[a])));
            d.set("frames", Json::list().add(Json::whole_number(f_lo)).add(Json::whole_number(f_hi))).set("shots", Json::whole_number(static_cast<long long>(got->shots.size())));
            d.set("raster_s", Json::of(got->p[3])).set("air_columns", Json::list().add(Json::whole_number(c0)).add(Json::whole_number(c1)));
            d.set("period_s", Json::of(T)).set("gate_s", Json::of(E));
            return d.text();
        }
    return "null";
}

}  // namespace helirec::calibration::geometry_calibration
