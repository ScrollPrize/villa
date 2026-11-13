// One pair of blocks cut, converted and registered (compare_blocks.hpp), and comparisons placed by the caller
// (Session::measure, session.hpp).
#include "calibration/compare_blocks.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <thread>

#include "calibration/util.hpp"

namespace helirec::calibration {

namespace {

bool is_good(int m) {
    if (m < 2 || m % 2) return false;
    for (int p : {2, 3, 5, 7})
        while (m % p == 0) m /= p;
    return m == 1;
}

double seconds_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

}  // namespace

Size pick_size(int max_total, int max_inner, int min_margin) {
    for (int n = std::min(max_inner, max_total - 2 * min_margin); n >= 16; --n) {
        if (!is_good(n)) continue;
        for (int m = min_margin; m <= min_margin + 24 && n + 2 * m <= max_total; ++m)
            if (is_good(n + 2 * m)) return {n, m};
    }
    return {};
}

Size diagnostic_columns(const Size& full, const ViewOptions& opt) {
    if (!opt.columns || opt.columns >= full.n) return full;
    const Size small = pick_size(full.total(), opt.columns, full.m);
    if (!small.n) throw Error("view columns leave no usable registration window");
    return small;
}

Trajectory trajectory(const Record& rec, const Stretch& s, double row_sign) {
    Trajectory t;
    t.angle_deg = stated_angle(s);
    const auto z = stated_height(rec, s);
    const double px_mm = pixel_um(rec) * 1e-3;
    t.rows.resize(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) t.rows[i] = row_sign * z[i] / px_mm;
    return t;
}

bool place_rows(int rows, int N, const Size& size, int& row_a) {
    const int lo = std::max(0, size.m - N), hi = std::min(rows - size.n, rows - size.n - size.m - N);
    if (hi < lo) return false;
    row_a = (lo + hi) / 2;
    return true;
}

Registrar& Registrars::get(const int n[3], const int margin[3], const RegistrationOptions& opt) {
    for (std::size_t i = 0; i < items.size(); ++i)
        if (items[i]->fits(n, margin, device)) {
            if (i) std::rotate(items.begin(), items.begin() + static_cast<std::ptrdiff_t>(i), items.begin() + static_cast<std::ptrdiff_t>(i) + 1);
            return *items.front();
        }
    if (items.size() >= 3) items.pop_back();  // each holds about a gigabyte of work memory
    items.insert(items.begin(), std::make_unique<Registrar>(n, margin, opt, device));
    return *items.front();
}
Registrars& registrars() {
    static Registrars r;
    return r;
}

ViewTiming& view_timing() {
    static ViewTiming t;
    return t;
}

void keep_curvature(const Registration& r, Measured& m) {
    const double c[6] = {r.curvature[0][0], r.curvature[0][1], r.curvature[0][2], r.curvature[1][1], r.curvature[1][2], r.curvature[2][2]};
    std::copy(c, c + 6, m.curvature);
}

bool measure(const Pairing& P, const BlockPlan& first_plan, Cut cut, const RegistrationOptions& ropt, const ViewOptions& opt, Measured& m) {
    // Crop corresponding rays, keeping the mirror centre unchanged for opposite views.
    // This changes only the measurement support, never the raw images or trajectory.
    if (opt.columns && opt.columns < cut.cols.n) {
        const Size small = diagnostic_columns(cut.cols, opt);
        // FFT-friendly padding can grow for a narrower window. Inset both possible
        // endpoints so its padded extent remains inside the original readable region.
        crop_column_origins(cut.col_a, cut.col_b, cut.cols.n, small.n, opt.column_fraction, P.mirror,
                            std::max(0, small.m - cut.cols.m));
        cut.cols = small;
    }
    const int rows = P.A.frames->rows(), cols = P.A.frames->cols();
    const std::int64_t nb = P.B.rec->stretches[P.sb].n_frames;
    BlockPlan plan = first_plan;
    m.plan = plan;
    m.cut = cut;
    m.ok = false;
    const int n[3] = {plan.n, cut.rows.n, cut.cols.n}, mg[3] = {plan.margin, cut.rows.m, cut.cols.m};
    const int L[3] = {n[0] + 2 * mg[0], n[1] + 2 * mg[1], n[2] + 2 * mg[2]};
    if (cut.row_a < 0 || cut.row_a + n[1] > rows || cut.col_a < 0 || cut.col_a + n[2] > cols) return false;
    const DetectorMap* const map_b = P.map_b ? P.map_b : P.map;
    const bool modelled_a = P.map && !P.map->empty(), modelled_b = map_b && !map_b->empty();
    static std::vector<std::uint16_t> raw;  // kept between blocks: its pages are touched once
    if (!modelled_a || !modelled_b) raw.resize(static_cast<std::size_t>(L[0]) * L[1] * cols);
    // the blocks are written where the registration's device takes them fastest
    Registrar& registrar = registrars().get(n, mg, ropt);
    float* a = registrar.staging_a();
    float* b = registrar.staging_b();
    std::vector<std::int64_t> frame(L[0]);
    std::vector<std::int32_t> row0(L[0]);
    for (int q = 0; q < n[0]; ++q) frame[q] = plan.p0 + q, row0[q] = cut.row_a;
    auto t0 = std::chrono::steady_clock::now();
    if (modelled_a) {
        line_integrals_mapped(*P.A.frames, *P.A.ff, *P.map, P.sa, frame.data(), cut.row_a, n[0], n[1], cut.col_a, n[2], false, a);
        view_timing().integrals += seconds_since(t0);
    } else {
        P.A.frames->read_rows(P.sa, frame.data(), row0.data(), n[0], n[1], raw.data());
        view_timing().read += seconds_since(t0);
        t0 = std::chrono::steady_clock::now();
        line_integrals(*P.A.ff, raw.data(), row0.data(), n[0], n[1], cut.col_a, n[2], false, a);
        view_timing().integrals += seconds_since(t0);
    }
    ++view_timing().blocks;
    int extra[3] = {0, 0, 0};
    const std::int64_t J0 = plan.j0 - plan.p0;
    for (int pass = 0; pass < 8; ++pass) {
        if (extra[0] && !plan_block_at(*P.ta, *P.tb, plan.p0, n[0], mg[0], J0 + extra[0], P.angle_offset, plan)) return false;
        if (plan.max_row_shift + 3 > mg[1]) return false;  // the margin of rows cannot take the pre-alignment
        if (plan.j0 - mg[0] < 0 || plan.j0 + n[0] + mg[0] > nb) return false;
        // a sway of B's content along the columns: whole columns go into the cut, the rest is resampled per frame
        std::vector<double> sway, sway_per_row;
        int sway_whole = 0, pad = 0;
        const int row_b = cut.row_a + plan.row_offset + extra[1] - mg[1];
        const double row_ref = 0.5 * (rows - 1);  // the row the frame's angle is taken to belong to
        if (P.sway_px != 0.0) {
            sway.resize(static_cast<std::size_t>(L[0]));
            sway_per_row.resize(static_cast<std::size_t>(L[0]));
            for (int q = 0; q < L[0]; ++q) {
                const double a = (P.tb->angle_deg[static_cast<std::size_t>(plan.j0 - mg[0] + q)] - P.sway_phase_deg) * (M_PI / 180.0);
                sway[q] = P.sway_px * std::cos(a);
                sway_per_row[q] = -P.sway_px * std::sin(a) * P.sway_row_deg * (M_PI / 180.0);
            }
            sway_whole = static_cast<int>(std::lround(sway[static_cast<std::size_t>(mg[0] + n[0] / 2)]));
            const double far = std::max(std::fabs(row_b - row_ref), std::fabs(row_b + L[1] - 1 - row_ref));
            double largest = 0;
            for (int q = 0; q < L[0]; ++q) {
                sway[q] -= sway_whole;
                largest = std::max(largest, std::fabs(sway[q]) + std::fabs(sway_per_row[q]) * far);
            }
            pad = static_cast<int>(std::ceil(largest)) + 3;
        }
        const int col_b = (P.mirror ? cut.col_b - extra[2] : cut.col_b + extra[2]) - mg[2] + sway_whole;
        if (col_b - pad < 0 || col_b + L[2] + pad > cols || row_b < 0 || row_b + L[1] > rows) return false;
        for (int q = 0; q < L[0]; ++q) frame[q] = plan.j0 - mg[0] + q, row0[q] = row_b;
        t0 = std::chrono::steady_clock::now();
        if (pad) {
            static std::vector<float> wide;  // kept between blocks
            const int nw = L[2] + 2 * pad;
            wide.resize(static_cast<std::size_t>(L[0]) * L[1] * nw);
            if (modelled_b) {
                line_integrals_mapped(*P.B.frames, *P.B.ff, *map_b, P.sb, frame.data(), row_b, L[0], L[1], col_b - pad, nw, false, wide.data());
            } else {
                raw.resize(static_cast<std::size_t>(L[0]) * L[1] * cols);
                P.B.frames->read_rows(P.sb, frame.data(), row0.data(), L[0], L[1], raw.data());
                view_timing().read += seconds_since(t0);
                t0 = std::chrono::steady_clock::now();
                line_integrals(*P.B.ff, raw.data(), row0.data(), L[0], L[1], col_b - pad, nw, false, wide.data());
            }
            shift_columns(wide.data(), L[0], L[1], L[2], pad, sway.data(), P.mirror, b, P.sway_row_deg != 0.0 ? sway_per_row.data() : nullptr, row_b - row_ref);
        } else if (modelled_b) {
            line_integrals_mapped(*P.B.frames, *P.B.ff, *map_b, P.sb, frame.data(), row_b, L[0], L[1], col_b, L[2], P.mirror, b);
        } else {
            P.B.frames->read_rows(P.sb, frame.data(), row0.data(), L[0], L[1], raw.data());
            view_timing().read += seconds_since(t0);
            t0 = std::chrono::steady_clock::now();
            line_integrals(*P.B.ff, raw.data(), row0.data(), L[0], L[1], col_b, L[2], P.mirror, b);
        }
        shear_frames(b, L[0], L[1], L[2], P.frame_shear);
        view_timing().integrals += seconds_since(t0);
        t0 = std::chrono::steady_clock::now();
        const Registration r = registrar.run(a, b, plan.row_shift.data());
        view_timing().registration += seconds_since(t0);
        ++view_timing().registrations;
        m.plan = plan;
        m.correlation = r.correlation;
        m.evaluations += r.evaluations;
        m.d[0] = r.shift[0];
        m.d[1] = extra[1] + r.shift[1];
        m.d[2] = extra[2] + r.shift[2];
        keep_curvature(r, m);
        if (r.whole[0] || r.whole[1] || r.whole[2]) {
            for (int j = 0; j < 3; ++j) extra[j] += r.whole[j];
            continue;
        }
        m.ok = r.converged && !r.at_edge && r.correlation > 0.15;
        return m.ok;
    }
    return false;
}

// ---------------------------------------------------------------- a session: scans kept open for many measurements

struct Session::Impl {
    int threads = 0;
    std::vector<std::unique_ptr<Record>> records;
    std::vector<std::unique_ptr<Frames>> frames;
    std::vector<std::unique_ptr<FlatField>> flats;
    std::map<std::pair<int, int>, Trajectory> trajectories;  // (scan, stretch), for trajectory_sign
    double trajectory_sign = 0;
    std::vector<DetectorMap> maps;  // per scan; empty: its blocks are cut from the frames as stored
    int read_ahead = std::max(2, static_cast<int>(std::thread::hardware_concurrency()) / 3);  // workers that decode ahead, per scan
    bool read_ahead_set = false;

    // what every measurement starts with: the sense of the rows, the device of the registrars, the readers' workers
    void begin(Session& s, double row_sign, Device device) {
        if (row_sign != 1.0 && row_sign != -1.0) throw Error("calibration session: row_sign is +1 or -1");
        if (trajectory_sign != row_sign) {
            trajectories.clear();
            trajectory_sign = row_sign;
        }
        if (registrars().device != device) registrars().items.clear();
        registrars().device = device;
        if (!read_ahead_set) {
            for (std::size_t k = 0; k < s.size(); ++k) s.frames(static_cast<int>(k)).set_prefetch_threads(read_ahead);
            read_ahead_set = true;
        }
    }
    const Trajectory& trajectory_of(const Session& s, int scan, int stretch) {
        const auto key = std::make_pair(scan, stretch);
        auto it = trajectories.find(key);
        if (it == trajectories.end()) {
            const Record& rec = s.record(scan);
            const Stretch& st = rec.stretches.at(static_cast<std::size_t>(stretch));
            if (!st.latched()) throw Error("stretch " + st.entry + " has no latched angles");
            it = trajectories.emplace(key, trajectory(rec, st, trajectory_sign)).first;
        }
        return it->second;
    }
};

namespace {

// a measurement as the caller gets it (the columns of views.tsv)
void fill(ViewMeasurement& res, const Measured& m, const Trajectory& ta) {
    res.frame_a = m.plan.p0;
    res.frame_b = m.plan.j0;
    res.frames = m.plan.n;
    res.row_a = m.cut.row_a;
    res.rows = m.cut.rows.n;
    res.row_offset = m.plan.row_offset;
    res.col_a = m.cut.col_a;
    res.col_b = m.cut.col_b;
    res.cols = m.cut.cols.n;
    res.stated_dq = m.plan.stated_dq;
    res.dq = m.d[0];
    res.stated_dv = m.plan.stated_dv();
    res.dv = m.d[1];
    res.stated_D = m.plan.stated_D;
    res.D = measured_D(m.plan, m.d[0], m.d[1]);
    res.du = m.d[2];
    res.correlation = m.correlation;
    res.evaluations = m.evaluations;
    res.ok = m.ok;
    res.slope_a = m.plan.slope_a;
    res.slope_b = m.plan.slope_b;
    res.step_deg = m.plan.step_deg;
    res.angle_a_deg = ta.angle_deg[static_cast<std::size_t>(m.plan.p0 + m.plan.n / 2)];
}

}  // namespace

Session::Session(const std::vector<std::string>& scans, int threads) : impl_(std::make_unique<Impl>()) {
    if (scans.empty()) throw Error("calibration session: no scan");
    impl_->threads = threads;
    for (const std::string& path : scans) {
        impl_->records.push_back(std::make_unique<Record>(read_record(path)));
        impl_->frames.push_back(std::make_unique<Frames>(*impl_->records.back(), threads));
        impl_->flats.emplace_back();
    }
    impl_->maps.resize(impl_->records.size());
}
Session::~Session() = default;

std::size_t Session::size() const { return impl_->records.size(); }
const Record& Session::record(int scan) const { return *impl_->records.at(static_cast<std::size_t>(scan)); }
Frames& Session::frames(int scan) { return *impl_->frames.at(static_cast<std::size_t>(scan)); }
const FlatField& Session::flat_field(int scan) {
    auto& ff = impl_->flats.at(static_cast<std::size_t>(scan));
    if (!ff) ff = std::make_unique<FlatField>(make_flat_field(frames(scan)));
    return *ff;
}

void Session::set_detector_map(const DetectorMap& map, int scan) {
    if (map.j.size() != map.i.size() || map.eu.size() != map.i.size() || map.ev.size() != map.i.size()) throw Error("calibration session: a detector map has one power pair and two coefficients per term");
    if (map.cs == 0 || map.vs == 0) throw Error("calibration session: a detector map's scales are not zero");
    if (scan < 0) std::fill(impl_->maps.begin(), impl_->maps.end(), map);
    else impl_->maps.at(static_cast<std::size_t>(scan)) = map;
}
const DetectorMap& Session::detector_map(int scan) const { return impl_->maps.at(static_cast<std::size_t>(scan)); }

void Session::set_read_ahead(int threads) {
    impl_->read_ahead = std::max(0, threads);
    impl_->read_ahead_set = false;
}

std::vector<ViewMeasurement> Session::measure(const std::vector<ViewRequest>& requests, double row_sign, Device device) {
    impl_->begin(*this, row_sign, device);
    const auto traj = [&](int scan, int stretch) -> const Trajectory& { return impl_->trajectory_of(*this, scan, stretch); };
    const RegistrationOptions ropt;
    const ViewOptions vopt;  // no diagnostic crop: the caller places the columns
    // every request placed first (what the record says, no frame read), so that the frames of the ones that
    // follow can be decoded while one is measured
    struct Placed {
        bool planned = false;
        BlockPlan plan;
        Cut cut;
    };
    std::vector<Placed> placed(requests.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const ViewRequest& q = requests[i];
        const Trajectory& ta = traj(q.scan_a, q.stretch_a);
        const Trajectory& tb = traj(q.scan_b, q.stretch_b);
        (void)flat_field(q.scan_a);
        (void)flat_field(q.scan_b);
        const int rows = frames(q.scan_a).rows();
        Placed& pl = placed[i];
        const Size fsize = pick_size(q.frames + 24, q.frames, 8);
        if (!fsize.n || !plan_block(ta, tb, q.frame_a, fsize.n, fsize.m, q.frame_hint, q.angle_offset_deg, pl.plan)) continue;
        const int row_margin = static_cast<int>(std::ceil(pl.plan.max_row_shift)) + 8;
        pl.cut.rows = pick_size(std::min(rows - std::abs(pl.plan.row_offset) - 4, q.rows + 2 * row_margin + 48), q.rows, row_margin);
        pl.cut.cols = pick_size(q.cols + 64, q.cols, 8);
        pl.cut.col_a = q.col_a;
        pl.cut.col_b = q.col_b;
        if (!pl.cut.rows.n || !pl.cut.cols.n) continue;
        if (q.row_a >= 0) pl.cut.row_a = q.row_a;
        else if (!place_rows(rows, pl.plan.row_offset, pl.cut.rows, pl.cut.row_a)) continue;
        pl.planned = true;
    }
    const auto announce = [&](std::size_t i) {
        if (i >= requests.size() || !placed[i].planned || !impl_->read_ahead) return;
        const ViewRequest& q = requests[i];
        const Placed& pl = placed[i];
        const int pad = 8;  // rows a detector map may reach beyond the block
        frames(q.scan_a).prefetch_rows(q.stretch_a, pl.plan.p0, pl.plan.n, pl.cut.row_a - pad, pl.cut.rows.n + 2 * pad);
        frames(q.scan_b).prefetch_rows(q.stretch_b, pl.plan.j0 - pl.plan.margin, pl.plan.n + 2 * pl.plan.margin, pl.cut.row_a + pl.plan.row_offset - pl.cut.rows.m - pad,
                                       pl.cut.rows.total() + 2 * pad);
    };
    std::vector<ViewMeasurement> out;
    out.reserve(requests.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const ViewRequest& q = requests[i];
        ViewMeasurement r;
        r.label = q.label;
        out.push_back(r);
        ViewMeasurement& res = out.back();
        announce(i + 1);
        announce(i + 2);
        if (!placed[i].planned) continue;
        const Trajectory& ta = traj(q.scan_a, q.stretch_a);
        const Trajectory& tb = traj(q.scan_b, q.stretch_b);
        const Pairing P{{&record(q.scan_a), &frames(q.scan_a), &flat_field(q.scan_a)},
                        {&record(q.scan_b), &frames(q.scan_b), &flat_field(q.scan_b)},
                        q.stretch_a, q.stretch_b, &ta, &tb, q.angle_offset_deg, q.mirror, &impl_->maps[static_cast<std::size_t>(q.scan_a)],
                        &impl_->maps[static_cast<std::size_t>(q.scan_b)], q.frame_shear, q.sway_px, q.sway_phase_deg, q.sway_row_deg};
        res.planned = true;
        Measured m;
        calibration::measure(P, placed[i].plan, placed[i].cut, ropt, vopt, m);
        fill(res, m, ta);
    }
    return out;
}

std::vector<PatchMeasurement> Session::measure_patches(const std::vector<ViewRequest>& requests, int patch_rows, int patch_cols, double row_sign, Device device, bool from_block) {
    if (patch_rows < 2 || patch_cols < 2) throw Error("calibration session: a patch has at least two rows and two columns");
    impl_->begin(*this, row_sign, device);
    const auto traj = [&](int scan, int stretch) -> const Trajectory& { return impl_->trajectory_of(*this, scan, stretch); };
    const RegistrationOptions ropt;
    // every request placed first, as in measure(): the plan of its block, and its patches
    struct Placed {
        bool planned = false;
        BlockPlan plan;
        std::vector<Cut> cuts;
        std::vector<std::pair<int, int>> place;  // (column index, row index) of each
        Cut whole;                               // the block as one cut, as measure() takes it
        bool has_whole = false;
        int row_lo = 0, row_n = 0;               // A's rows of all patches
    };
    std::vector<Placed> placed(requests.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const ViewRequest& q = requests[i];
        const Trajectory& ta = traj(q.scan_a, q.stretch_a);
        const Trajectory& tb = traj(q.scan_b, q.stretch_b);
        (void)flat_field(q.scan_a);
        (void)flat_field(q.scan_b);
        const int rows = frames(q.scan_a).rows();
        Placed& pl = placed[i];
        const Size fsize = pick_size(q.frames + 24, q.frames, 8);
        if (!fsize.n || !plan_block(ta, tb, q.frame_a, fsize.n, fsize.m, q.frame_hint, q.angle_offset_deg, pl.plan)) continue;
        const int pr = std::min(patch_rows, q.rows), pc = std::min(patch_cols, q.cols);
        const int row_margin = static_cast<int>(std::ceil(pl.plan.max_row_shift)) + 8;
        const Size sr = pick_size(std::min(rows - std::abs(pl.plan.row_offset) - 4, pr + 2 * row_margin + 48), pr, row_margin);
        const Size sc = pick_size(pc + 64, pc, 8);
        if (!sr.n || !sc.n) continue;
        const int nj = std::max(1, q.rows / pr), ni = std::max(1, q.cols / pc);
        int row0 = q.row_a;
        if (row0 < 0 && !place_rows(rows, pl.plan.row_offset, Size{(nj - 1) * pr + sr.n, sr.m}, row0)) continue;
        for (int j = 0; j < nj; ++j)
            for (int k = 0; k < ni; ++k) {
                Cut c;
                c.rows = sr;
                c.cols = sc;
                c.row_a = row0 + j * pr;
                c.col_a = q.col_a + k * pc;
                c.col_b = q.mirror ? q.col_b + (q.cols - pc) - k * pc : q.col_b + k * pc;  // a mirrored B's patches run the other way
                pl.cuts.push_back(c);
                pl.place.emplace_back(k, j);
            }
        pl.row_lo = row0;
        pl.row_n = (nj - 1) * pr + sr.n;
        pl.planned = true;
        if (from_block && pl.cuts.size() > 1) {
            pl.whole.rows = pick_size(std::min(rows - std::abs(pl.plan.row_offset) - 4, q.rows + 2 * row_margin + 48), q.rows, row_margin);
            pl.whole.cols = pick_size(q.cols + 64, q.cols, 8);
            pl.whole.row_a = row0, pl.whole.col_a = q.col_a, pl.whole.col_b = q.col_b;
            pl.has_whole = pl.whole.rows.n && pl.whole.cols.n;
        }
    }
    const auto announce = [&](std::size_t i) {
        if (i >= requests.size() || !placed[i].planned || !impl_->read_ahead) return;
        const ViewRequest& q = requests[i];
        const Placed& pl = placed[i];
        const int pad = 8, m = pl.cuts.front().rows.m;
        frames(q.scan_a).prefetch_rows(q.stretch_a, pl.plan.p0, pl.plan.n, pl.row_lo - pad, pl.row_n + 2 * pad);
        frames(q.scan_b).prefetch_rows(q.stretch_b, pl.plan.j0 - pl.plan.margin, pl.plan.n + 2 * pl.plan.margin, pl.row_lo + pl.plan.row_offset - m - pad, pl.row_n + 2 * m + 2 * pad);
    };
    std::vector<PatchMeasurement> out;
    std::vector<Measured> measured;
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const ViewRequest& q = requests[i];
        announce(i + 1);
        announce(i + 2);
        if (!placed[i].planned) {
            PatchMeasurement r;
            r.request = static_cast<int>(i);
            r.m.label = q.label;
            out.push_back(r);
            continue;
        }
        const Trajectory& ta = traj(q.scan_a, q.stretch_a);
        const Trajectory& tb = traj(q.scan_b, q.stretch_b);
        const Pairing P{{&record(q.scan_a), &frames(q.scan_a), &flat_field(q.scan_a)},
                        {&record(q.scan_b), &frames(q.scan_b), &flat_field(q.scan_b)},
                        q.stretch_a, q.stretch_b, &ta, &tb, q.angle_offset_deg, q.mirror, &impl_->maps[static_cast<std::size_t>(q.scan_a)],
                        &impl_->maps[static_cast<std::size_t>(q.scan_b)], q.frame_shear, q.sway_px, q.sway_phase_deg, q.sway_row_deg};
        calibration::measure_patches(P, placed[i].plan, placed[i].cuts, ropt, device, measured, placed[i].has_whole ? &placed[i].whole : nullptr);
        for (std::size_t k = 0; k < measured.size(); ++k) {
            PatchMeasurement r;
            r.request = static_cast<int>(i);
            r.i = placed[i].place[k].first;
            r.j = placed[i].place[k].second;
            r.m.label = q.label;
            r.m.planned = true;
            fill(r.m, measured[k], ta);
            std::copy(measured[k].curvature, measured[k].curvature + 6, r.curvature);
            out.push_back(r);
        }
    }
    return out;
}

MeasureTiming measure_timing(bool reset) {
    ViewTiming& t = view_timing();
    const MeasureTiming out{t.read, t.integrals, t.registration, t.blocks, t.registrations};
    if (reset) t = ViewTiming{};
    return out;
}

}  // namespace helirec::calibration
