// Many small blocks (patches) of one pair of views measured from one reading of the frames (view_measure.hpp,
// measure_patches). Every step is measure()'s own, on the part of the block a patch occupies: the values a patch
// is registered on are those measure() would read for it, so the two agree to the rounding of the transforms.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>

#include "calibration/view_measure.hpp"

namespace helirec::calibration {

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

// Registrars of the patch sizes in use, the few most recent kept between blocks like registrars(). Where several
// patches are registered side by side there is one per thread, each transform on one thread (the patches are what
// runs in parallel), on CUDA kDeviceRegistrars of them; a block that is one patch gets one registrar with the
// threads measure() gives its own, so that it returns the same numbers.
struct PatchRegistrars {
    struct Set {
        int n[3], m[3];
        Device device;
        bool single;
        std::vector<std::unique_ptr<Registrar>> items;
    };
    std::vector<Set> sets;  // most recent first

    std::vector<std::unique_ptr<Registrar>>& get(const int n[3], const int m[3], Device dev, int count, const RegistrationOptions& opt) {
        const bool single = count == 1;
        std::size_t at = sets.size();
        for (std::size_t s = 0; s < sets.size() && at == sets.size(); ++s) {
            bool same = sets[s].device == dev && sets[s].single == single;
            for (int j = 0; j < 3; ++j) same = same && sets[s].n[j] == n[j] && sets[s].m[j] == m[j];
            if (same) at = s;
        }
        if (at == sets.size()) {
            if (sets.size() >= 3) sets.pop_back();  // a block's registrar holds about a gigabyte of work memory
            Set s;
            for (int j = 0; j < 3; ++j) s.n[j] = n[j], s.m[j] = m[j];
            s.device = dev, s.single = single;
            sets.insert(sets.begin(), std::move(s));
        } else if (at) {
            std::rotate(sets.begin(), sets.begin() + static_cast<std::ptrdiff_t>(at), sets.begin() + static_cast<std::ptrdiff_t>(at) + 1);
        }
        Set& s = sets.front();
        RegistrationOptions o = opt;
        if (!single) o.threads = 1;
        while (static_cast<int>(s.items.size()) < count) s.items.push_back(std::make_unique<Registrar>(n, m, o, dev));
        return s.items;
    }
};
PatchRegistrars& patch_registrars() {
    static PatchRegistrars r;
    return r;
}

// Frames, rows and columns read beyond what the patches need as they stand: a patch that the registration moves
// by whole samples is cut again from the same reading. Past that, B is read once more where the patches then are.
constexpr int kSlack[3] = {1, 4, 4};
constexpr int kReadings = 3;
// Registrars at work on one device at a time (each from a thread of its own): a patch's registration is many small
// steps with the host in between, so one registrar leaves the device idle most of the time.
// (PHerc0343P, 20 256 patches of 32 x 128 x 128 on an RTX 5090: 69 s of registration with one, 58 s with four, 62 s
// with eight.)
constexpr int kDeviceRegistrars = 4;

// B's content displaced along the columns per frame of a block (measure()): the part of a column, the line in the
// row, and the whole columns that go into the cut
struct Sway {
    std::vector<double> part, per_row;
    int whole = 0;
};

// one patch on its way through measure()'s loop
struct State {
    BlockPlan plan;
    int extra[3] = {0, 0, 0};
    int pass = 0;
    bool open = false;  // not finished
    bool wants = false; // its next cut lies outside what was read
};

}  // namespace

void measure_patches(const Pairing& P, const BlockPlan& plan, const std::vector<Cut>& patches, const RegistrationOptions& ropt, Device device, std::vector<Measured>& result,
                     const Cut* whole) {
    result.assign(patches.size(), Measured{});
    if (patches.empty()) return;
    const Size sr = patches[0].rows, sc = patches[0].cols;
    for (const Cut& c : patches)
        if (c.rows.n != sr.n || c.rows.m != sr.m || c.cols.n != sc.n || c.cols.m != sc.m) throw Error("measure_patches: the patches of a block have one size");
    // The block as one cut goes last, beyond the patches: it is registered first, from the same readings, and the
    // patches start where it ends (a patch holds a hundredth of the block's content and, started from the plan
    // alone, can settle on a false maximum some frames or rows away; started from the block it has a sample to go).
    const std::size_t K = patches.size();
    const bool seeded = whole && K > 1;
    std::vector<Cut> cuts = patches;
    if (seeded) cuts.push_back(*whole);
    std::vector<Measured> out(cuts.size());
    const int rows = P.A.frames->rows(), cols = P.A.frames->cols();
    const std::int64_t nb = P.B.rec->stretches[P.sb].n_frames;
    struct Shape {
        int n[3], mg[3], L[3];
    };
    const auto shape_of = [&](const Cut& c) {
        Shape s{{plan.n, c.rows.n, c.cols.n}, {plan.margin, c.rows.m, c.cols.m}, {0, 0, 0}};
        for (int j = 0; j < 3; ++j) s.L[j] = s.n[j] + 2 * s.mg[j];
        return s;
    };
    const Shape shapes[2] = {shape_of(cuts.front()), shape_of(cuts.back())};  // of the patches, of the block
    const auto shape = [&](std::size_t k) -> const Shape& { return shapes[k < K ? 0 : 1]; };
    const int n0 = plan.n, m0 = plan.margin, L0 = n0 + 2 * m0;  // the frames are the plan's for every cut
    const double row_ref = 0.5 * (rows - 1);  // the row the frame's angle is taken to belong to
    const std::int64_t J0 = plan.j0 - plan.p0;
    const bool swaying = P.sway_px != 0.0;

    // ---- the sway, as measure() takes it: of one frame of B, of a plan's block, and the columns it needs beside
    const auto sway_at = [&](std::int64_t frame_b, double& part, double& per_row) {
        const double a = (P.tb->angle_deg[static_cast<std::size_t>(frame_b)] - P.sway_phase_deg) * (M_PI / 180.0);
        part = P.sway_px * std::cos(a);
        per_row = -P.sway_px * std::sin(a) * P.sway_row_deg * (M_PI / 180.0);
    };
    const auto sway_of = [&](const BlockPlan& pl) {
        Sway s;
        if (!swaying) return s;
        s.part.resize(static_cast<std::size_t>(L0));
        s.per_row.resize(static_cast<std::size_t>(L0));
        for (int q = 0; q < L0; ++q) sway_at(pl.j0 - m0 + q, s.part[q], s.per_row[q]);
        s.whole = static_cast<int>(std::lround(s.part[static_cast<std::size_t>(m0 + n0 / 2)]));
        for (double& v : s.part) v -= s.whole;
        return s;
    };
    const auto pad_of = [&](const Sway& s, int row_lo, int nrows) {
        if (!swaying) return 0;
        const double far = std::max(std::fabs(row_lo - row_ref), std::fabs(row_lo + nrows - 1 - row_ref));
        double largest = 0;
        for (std::size_t q = 0; q < s.part.size(); ++q) largest = std::max(largest, std::fabs(s.part[q]) + std::fabs(s.per_row[q]) * far);
        return static_cast<int>(std::ceil(largest)) + 3;
    };
    const Sway base = sway_of(plan);

    // ---- B's block of a patch as measure() would cut it in the patch's state: its first frame and row, and its
    // first column less the sway's whole columns. false: measure() stops there.
    std::vector<State> state(cuts.size());
    const auto b_block_of = [&](std::size_t k, std::int64_t& frame_b, int& row_b, int& col) {
        const State& s = state[k];
        const Cut& c = cuts[k];
        const int* const mg = shape(k).mg;
        const int* const L = shape(k).L;
        if (s.plan.max_row_shift + 3 > mg[1]) return false;  // the margin of rows cannot take the pre-alignment
        if (s.plan.j0 - m0 < 0 || s.plan.j0 + n0 + m0 > nb) return false;
        frame_b = s.plan.j0 - m0;
        row_b = c.row_a + s.plan.row_offset + s.extra[1] - mg[1];
        col = (P.mirror ? c.col_b - s.extra[2] : c.col_b + s.extra[2]) - mg[2];
        if (row_b < 0 || row_b + L[1] > rows) return false;
        const Sway own = s.extra[0] ? sway_of(s.plan) : Sway{};
        const Sway& sw = s.extra[0] ? own : base;
        const int pad = pad_of(sw, row_b, L[1]), col_b = col + sw.whole;
        return !(col_b - pad < 0 || col_b + L[2] + pad > cols);
    };

    // ---- A: the patches the detector holds, read once
    int ra_lo = rows, ra_hi = 0, ca_lo = cols, ca_hi = 0;
    std::size_t n_open = 0;
    for (std::size_t k = 0; k < cuts.size(); ++k) {
        const Cut& c = cuts[k];
        const int* const n = shape(k).n;
        out[k].plan = plan, out[k].cut = c;
        state[k].plan = plan;
        if (c.row_a < 0 || c.row_a + n[1] > rows || c.col_a < 0 || c.col_a + n[2] > cols) continue;
        std::int64_t frame_b;
        int row_b, col;
        if (!b_block_of(k, frame_b, row_b, col)) continue;
        state[k].open = true;
        ++n_open;
        ra_lo = std::min(ra_lo, c.row_a), ra_hi = std::max(ra_hi, c.row_a + n[1]);
        ca_lo = std::min(ca_lo, c.col_a), ca_hi = std::max(ca_hi, c.col_a + n[2]);
    }
    const auto finish = [&] { result.assign(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(K)); };
    if (!n_open) return finish();
    const DetectorMap* const map_b = P.map_b ? P.map_b : P.map;
    const bool modelled_a = P.map && !P.map->empty(), modelled_b = map_b && !map_b->empty();
    static std::vector<std::uint16_t> raw;  // kept between blocks, as in measure()
    static std::vector<float> a_block, b_block, wide;
    std::vector<std::int64_t> frame;
    std::vector<std::int32_t> row0;
    const int RA = ra_hi - ra_lo, CA = ca_hi - ca_lo;
    a_block.resize(static_cast<std::size_t>(n0) * RA * CA);
    frame.resize(static_cast<std::size_t>(n0));
    row0.assign(static_cast<std::size_t>(n0), ra_lo);
    for (int q = 0; q < n0; ++q) frame[q] = plan.p0 + q;
    auto t0 = std::chrono::steady_clock::now();
    if (modelled_a) {
        line_integrals_mapped(*P.A.frames, *P.A.ff, *P.map, P.sa, frame.data(), ra_lo, n0, RA, ca_lo, CA, false, a_block.data());
        view_timing().integrals += seconds_since(t0);
    } else {
        raw.resize(static_cast<std::size_t>(n0) * RA * cols);
        P.A.frames->read_rows(P.sa, frame.data(), row0.data(), n0, RA, raw.data());
        view_timing().read += seconds_since(t0);
        t0 = std::chrono::steady_clock::now();
        line_integrals(*P.A.ff, raw.data(), row0.data(), n0, RA, ca_lo, CA, false, a_block.data());
        view_timing().integrals += seconds_since(t0);
    }
    ++view_timing().blocks;

    // ---- B, for the cuts first to last (all of one shape): read where the open ones are, each taken as far as that
    // reading serves it; read again where the rest then are. What a reading before holds is used first: the patches
    // of a block that is registered itself begin in the block's last reading.
    std::int64_t fb_lo = nb, fb_hi = 0;
    int rb_lo = rows, rb_hi = 0, cb_lo = cols, cb_hi = 0, FB = 0, RB = 0, CB = 0;
    const auto run = [&](std::size_t first, std::size_t last) {
    const int* const n = shape(first).n;
    const int* const mg = shape(first).mg;
    const int* const L = shape(first).L;
    const std::size_t count = last - first;
    // the block's reading holds its patches too, whose margins may be the wider ones
    const int wider[3] = {0, first >= K ? std::max(0, shapes[0].mg[1] - mg[1]) : 0, first >= K ? std::max(0, shapes[0].mg[2] - mg[2]) : 0};
    std::size_t n_open = 0;
    for (std::size_t k = first; k < last; ++k) n_open += state[k].open ? 1 : 0;
    if (!n_open) return;
    const int T = count == 1 ? 1 : std::max(2, std::min<int>(device == Device::CUDA ? std::min(omp_get_max_threads(), kDeviceRegistrars) : omp_get_max_threads(), static_cast<int>(count)));
    std::vector<std::unique_ptr<Registrar>>& regs = patch_registrars().get(n, mg, device, T, ropt);
    std::string failure;
    std::mutex failure_mutex;
    for (int reading = CB > 0 ? -1 : 0; reading < kReadings && n_open; ++reading) {
        std::vector<std::size_t> todo;
        if (reading >= 0) fb_lo = nb, fb_hi = 0, rb_lo = rows, rb_hi = 0, cb_lo = cols, cb_hi = 0;
        for (std::size_t k = first; k < last; ++k) {
            if (!state[k].open) continue;
            std::int64_t frame_b;
            int row_b, col;
            if (!b_block_of(k, frame_b, row_b, col)) {  // measure() stops: the patch stays as it is
                state[k].open = false;
                continue;
            }
            state[k].wants = false;
            todo.push_back(k);
            if (reading < 0) continue;
            fb_lo = std::min(fb_lo, frame_b), fb_hi = std::max<std::int64_t>(fb_hi, frame_b + L[0]);
            rb_lo = std::min(rb_lo, row_b), rb_hi = std::max(rb_hi, row_b + L[1]);
            cb_lo = std::min(cb_lo, col + base.whole), cb_hi = std::max(cb_hi, col + base.whole + L[2]);
        }
        if (todo.empty()) break;
        auto t0 = std::chrono::steady_clock::now();
        if (reading >= 0) {
        fb_lo = std::max<std::int64_t>(0, fb_lo - kSlack[0]), fb_hi = std::min(nb, fb_hi + kSlack[0]);
        rb_lo = std::max(0, rb_lo - kSlack[1] - wider[1]), rb_hi = std::min(rows, rb_hi + kSlack[1] + wider[1]);
        FB = static_cast<int>(fb_hi - fb_lo), RB = rb_hi - rb_lo;
        // every frame read is resampled by its own sway, less the base plan's whole columns: a patch on another
        // plan finds its content at the same place
        Sway window;
        if (swaying) {
            window.part.resize(static_cast<std::size_t>(FB));
            window.per_row.resize(static_cast<std::size_t>(FB));
            for (int q = 0; q < FB; ++q) {
                sway_at(fb_lo + q, window.part[q], window.per_row[q]);
                window.part[q] -= base.whole;
            }
        }
        const int pad = pad_of(window, rb_lo, RB);
        cb_lo = std::max(pad, cb_lo - kSlack[2] - wider[2]), cb_hi = std::min(cols - pad, cb_hi + kSlack[2] + wider[2]);
        CB = std::max(0, cb_hi - cb_lo);
        if (CB > 0) {
            b_block.resize(static_cast<std::size_t>(FB) * RB * CB);
            frame.resize(static_cast<std::size_t>(FB));
            row0.assign(static_cast<std::size_t>(FB), rb_lo);
            for (int q = 0; q < FB; ++q) frame[q] = fb_lo + q;
            t0 = std::chrono::steady_clock::now();
            const int nw = CB + 2 * pad;
            if (pad) wide.resize(static_cast<std::size_t>(FB) * RB * nw);
            float* const first = pad ? wide.data() : b_block.data();
            if (modelled_b) {
                line_integrals_mapped(*P.B.frames, *P.B.ff, *map_b, P.sb, frame.data(), rb_lo, FB, RB, cb_lo - pad, nw, pad ? false : P.mirror, first);
            } else {
                raw.resize(static_cast<std::size_t>(FB) * RB * cols);
                P.B.frames->read_rows(P.sb, frame.data(), row0.data(), FB, RB, raw.data());
                view_timing().read += seconds_since(t0);
                t0 = std::chrono::steady_clock::now();
                line_integrals(*P.B.ff, raw.data(), row0.data(), FB, RB, cb_lo - pad, nw, pad ? false : P.mirror, first);
            }
            if (pad) shift_columns(wide.data(), FB, RB, CB, pad, window.part.data(), P.mirror, b_block.data(), P.sway_row_deg != 0.0 ? window.per_row.data() : nullptr, rb_lo - row_ref);
            view_timing().integrals += seconds_since(t0);
        }
        }

        int registrations = 0;
        t0 = std::chrono::steady_clock::now();
        // one patch as far as this reading serves it; the number of registrations it took
        const auto advance = [&](std::size_t k, Registrar& registrar) {
            int registrations = 0;
            try {
                const Cut& c = cuts[k];
                State& s = state[k];
                Measured& m = out[k];
                float* a = registrar.staging_a();
                float* b = registrar.staging_b();
                // (a block that is one patch is a hundred megabytes: its frames are copied side by side)
#pragma omp parallel for schedule(static) if (T == 1)
                for (int q = 0; q < n[0]; ++q)
                    for (int r = 0; r < n[1]; ++r)
                        std::memcpy(a + (static_cast<std::size_t>(q) * n[1] + r) * n[2],
                                    a_block.data() + (static_cast<std::size_t>(q) * RA + (c.row_a - ra_lo + r)) * CA + (c.col_a - ca_lo), static_cast<std::size_t>(n[2]) * sizeof(float));
                while (s.open) {
                    if (s.pass >= 8) {
                        s.open = false;
                        break;
                    }
                    std::int64_t frame_b;
                    int row_b, col;
                    if (!b_block_of(k, frame_b, row_b, col)) {
                        s.open = false;
                        break;
                    }
                    // where that cut lies in what was read (its columns reversed for a mirrored B)
                    const std::int64_t f0 = frame_b - fb_lo;
                    const int y0 = row_b - rb_lo, x0 = P.mirror ? cb_hi - (col + base.whole + L[2]) : col + base.whole - cb_lo;
                    if (f0 < 0 || f0 + L[0] > FB || y0 < 0 || y0 + L[1] > RB || x0 < 0 || x0 + L[2] > CB) {
                        s.wants = true;
                        break;
                    }
#pragma omp parallel for schedule(static) if (T == 1)
                    for (int q = 0; q < L[0]; ++q)
                        for (int r = 0; r < L[1]; ++r)
                            std::memcpy(b + (static_cast<std::size_t>(q) * L[1] + r) * L[2],
                                        b_block.data() + (static_cast<std::size_t>(f0 + q) * RB + (y0 + r)) * CB + x0, static_cast<std::size_t>(L[2]) * sizeof(float));
                    shear_frames(b, L[0], L[1], L[2], P.frame_shear);
                    const Registration r = registrar.run(a, b, s.plan.row_shift.data());
                    ++registrations;
                    ++s.pass;
                    m.plan = s.plan;
                    m.correlation = r.correlation;
                    m.evaluations += r.evaluations;
                    m.d[0] = r.shift[0];
                    m.d[1] = s.extra[1] + r.shift[1];
                    m.d[2] = s.extra[2] + r.shift[2];
                    keep_registration(r, m);
                    if (r.whole[0] || r.whole[1] || r.whole[2]) {
                        for (int j = 0; j < 3; ++j) s.extra[j] += r.whole[j];
                        // other frames of B: planned again, as measure() does before it cuts
                        if (s.extra[0] && !plan_block_at(*P.ta, *P.tb, plan.p0, n[0], mg[0], J0 + s.extra[0], P.angle_offset, s.plan)) s.open = false;
                        continue;
                    }
                    m.ok = r.converged && !r.at_edge && r.correlation > 0.15;
                    s.open = false;
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (failure.empty()) failure = e.what();
            }
            return registrations;
        };
        if (T == 1) {  // no team: the registrar's own loops and transforms keep their threads, as in measure()
            for (std::size_t k : todo) registrations += advance(k, *regs[0]);
        } else {
#pragma omp parallel for num_threads(T) schedule(dynamic) reduction(+ : registrations)
            for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(todo.size()); ++t)
                registrations += advance(todo[static_cast<std::size_t>(t)], *regs[static_cast<std::size_t>(omp_get_thread_num())]);
        }
        view_timing().registration += seconds_since(t0);
        view_timing().registrations += registrations;
        if (!failure.empty()) throw Error(failure);
        n_open = 0;
        for (std::size_t k : todo) n_open += state[k].open ? 1 : 0;
    }
    };
    if (seeded) {
        run(K, K + 1);
        if (out[K].ok)
            for (std::size_t k = 0; k < K; ++k) {
                if (!state[k].open) continue;
                const State before = state[k];
                state[k].plan = state[K].plan;
                std::copy(state[K].extra, state[K].extra + 3, state[k].extra);
                std::int64_t frame_b;
                int row_b, col;
                if (!b_block_of(k, frame_b, row_b, col)) state[k] = before;  // where the block ended the patch cannot be cut: from the plan
            }
    }
    run(0, K);

    // what three readings did not serve is measured by measure() itself, from the start
    const ViewOptions vopt;
    for (std::size_t k = 0; k < K; ++k)
        if (state[k].open) {
            if (registrars().device != device) registrars().items.clear();
            registrars().device = device;
            out[k] = Measured{};
            measure(P, plan, cuts[k], ropt, vopt, out[k]);
        }
    finish();
}

}  // namespace helirec::calibration
