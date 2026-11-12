// The record against itself: no frame is read here.
#include <sys/stat.h>

#include <cmath>
#include <map>
#include <set>

#include "calibration/util.hpp"
#include "helirec/calibration/verify.hpp"

namespace helirec::calibration {

namespace {

bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string join(const std::vector<std::string>& v, const char* sep = "; ") {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? sep : "") + v[i];
    return s;
}

double mean_finite(const std::vector<double>& v) {
    long double s = 0;
    std::size_t n = 0;
    for (double x : v)
        if (std::isfinite(x)) s += x, ++n;
    return n ? static_cast<double>(s / n) : std::numeric_limits<double>::quiet_NaN();
}

// mean of v over the readings whose time lies in [t0, t1]; NaN readings are skipped
double mean_between(const std::vector<double>& t, const std::vector<double>& v, double t0, double t1) {
    long double s = 0;
    std::size_t n = 0;
    for (std::size_t i = 0; i < t.size() && i < v.size(); ++i)
        if (t[i] >= t0 && t[i] <= t1 && std::isfinite(v[i])) s += v[i], ++n;
    return n ? static_cast<double>(s / n) : std::numeric_limits<double>::quiet_NaN();
}

// the mean level of the dark frames as the detector software recorded it (statistics rectangle)
double recorded_dark_level(const Record& rec) {
    std::vector<double> all;
    for (int i : rec.stretches_with(Role::Dark)) all.insert(all.end(), rec.stretches[i].stat_avg.begin(), rec.stretches[i].stat_avg.end());
    return mean_finite(all);
}

}  // namespace

void check_record(const Record& rec, Report& rep, double eps, double radius_px) {
    const std::string scan = rec.label;
    const Tolerances tol = tolerances(rec, eps, radius_px);
    const double px_mm = pixel_um(rec) * 1e-3;
    const Values& T = rec.technique;
    const std::string det = "detector/" + rec.detector + "/";
    const auto proj = rec.stretches_with(Role::Projection);

    // ------------------------------------------------------------------ 4.1 frame counts
    {
        Finding& f = rep.add("4.1", scan, "frame counts of the stretches");
        f.stated = fmt("sequence: %g darks, %g flats per series, %g projections per %g deg, nb_turns %g",
                       T.number("scan/dark_n"), T.number("scan/flat_n"), T.number("scan/tomo_n"), T.number("scan/scan_range"),
                       T.number("scan/nb_turns", NAN));
        std::vector<std::string> parts;
        bool ok = true;
        std::int64_t total = 0, n_proj = 0;
        for (const Stretch& s : rec.stretches) {
            parts.push_back(fmt("%s %s %lld", s.entry.c_str(), role_name(s.role), static_cast<long long>(s.n_frames)));
            total += s.n_frames;
            if (s.role == Role::Dark) ok &= s.n_frames == static_cast<std::int64_t>(T.number("scan/dark_n"));
            if (s.role == Role::Flat) ok &= s.n_frames == static_cast<std::int64_t>(T.number("scan/flat_n"));
            if (s.role == Role::Projection) {
                n_proj += s.n_frames;
                ok &= s.scan_parameters.has_number("npoints") && s.n_frames == static_cast<std::int64_t>(s.scan_parameters.number("npoints"));
                ok &= s.n_frames % static_cast<std::int64_t>(T.number("scan/tomo_n")) == 0;
            }
            if (s.role == Role::Unknown) ok = false;
            if (s.files.header.has_number("acquisition/nb_frames"))
                ok &= s.n_frames == static_cast<std::int64_t>(s.files.header.number("acquisition/nb_frames"));
            ok &= s.end_reason == "SUCCESS";
        }
        f.found = join(parts) + fmt("; %lld frames in all, every stretch ended with SUCCESS: %s", static_cast<long long>(total), ok ? "yes" : "NO");
        f.verdict = ok ? "verified" : "fails";
        const double turns = static_cast<double>(n_proj) / T.number("scan/tomo_n");
        if (T.has_number("scan/nb_turns") && turns != T.number("scan/nb_turns")) {
            f.note = fmt("the projection stretches hold %g turns of frames where the sequence states nb_turns %g; the acquisition software's definition must be established", turns, T.number("scan/nb_turns"));
            if (ok) f.verdict = "unverified";
        }
        f.numbers = {{"frames", static_cast<double>(total)}, {"projection_frames", static_cast<double>(n_proj)}};
    }
    // ------------------------------------------------------------------ 4.1 frame -> file map
    {
        Finding& f = rep.add("4.1", scan, "map from frame to file");
        std::int64_t files = 0, missing = 0, gaps = 0, on_disk = 0, mapped_on_disk = 0;
        bool have_counts = true;
        std::vector<std::string> held;
        for (const Stretch& s : rec.stretches) {
            std::int64_t next = 0;
            std::set<std::string> seen;
            for (const FramePiece& p : s.pieces) {
                if (p.first != next) ++gaps;
                next = p.first + p.count;
                if (seen.insert(p.file).second) {
                    ++files;
                    if (!file_exists(p.file)) ++missing;
                }
            }
            if (next != s.n_frames) ++gaps;
            mapped_on_disk += s.n_on_disk();
            if (s.files.frames_in_file.empty() && s.n_on_disk() > 0) have_counts = false;
            for (std::int64_t n : s.files.frames_in_file) on_disk += n;
            std::vector<std::string> ranges;
            for (const auto& [a, b] : s.frames_on_disk()) ranges.push_back(fmt("%lld-%lld", static_cast<long long>(a), static_cast<long long>(b - 1)));
            held.push_back(s.entry + ": " + (ranges.empty() ? std::string("none") : join(ranges, ", ")));
        }
        std::int64_t total = 0;
        for (const Stretch& s : rec.stretches) total += s.n_frames;
        f.stated = fmt("%lld frames mapped onto %lld files", static_cast<long long>(total), static_cast<long long>(files));
        f.found = fmt("%lld files missing, %lld breaks in the map", static_cast<long long>(missing), static_cast<long long>(gaps));
        if (have_counts) f.found += fmt("; the files hold %lld frames", static_cast<long long>(on_disk));
        f.verdict = (missing == 0 && gaps == 0 && (!have_counts || on_disk == total)) ? "verified" : "fails";
        if (missing > 0) {
            // a partial copy: what is not on disk cannot be checked, and is never read (the virtual dataset
            // of the master file would return zeros for it)
            f.found += fmt("; %lld of the %lld frames are on disk (%s)", static_cast<long long>(mapped_on_disk), static_cast<long long>(total), join(held).c_str());
            f.verdict = gaps == 0 && (!have_counts || on_disk == mapped_on_disk) ? "unverified" : "fails";
            f.note = "this copy of the acquisition is partial. Every check on frames covers the frames on disk only; no frame that is absent is read";
            f.numbers = {{"frames", static_cast<double>(total)}, {"frames_on_disk", static_cast<double>(mapped_on_disk)}, {"files_missing", static_cast<double>(missing)}};
        }
    }
    // ------------------------------------------------------------------ 4.1, 4.5, 4.6 the hardware timer
    {
        // counts of the timer between the edges of the gates, in ns. The counter is 32 bits wide and may wrap
        // within a stretch (differences are taken modulo 2^32); its tick is 1 us unless the latched times say
        // otherwise (the stretch's own timer_delta against its first gate, to the nearest power of ten).
        std::map<std::int64_t, std::int64_t> periods, gates;
        std::int64_t n = 0;
        double coarsest_tick_us = 0;
        std::vector<std::string> ticks;
        for (int i : proj) {
            const Stretch& s = rec.stretches[i];
            if (s.timer_raw.size() != 2 * static_cast<std::size_t>(s.n_frames) + 1) continue;
            const auto counts = [&](std::size_t a, std::size_t b) {
                return static_cast<std::int64_t>(static_cast<std::uint32_t>(s.timer_raw[a]) - static_cast<std::uint32_t>(s.timer_raw[b]));
            };
            double tick_us = 1;
            if (!s.timer_delta.empty() && s.n_frames > 0 && counts(1, 0) > 0) {
                const double t = s.timer_delta[0] * 1e6 / static_cast<double>(counts(1, 0));
                if (t > 0) tick_us = std::pow(10.0, std::round(std::log10(t)));
            }
            if (tick_us != 1) ticks.push_back(fmt("%s counts every %g us", s.entry.c_str(), tick_us));
            coarsest_tick_us = std::max(coarsest_tick_us, tick_us);
            for (std::int64_t k = 0; k < s.n_frames; ++k) {
                ++periods[std::llround(static_cast<double>(counts(2 * k + 2, 2 * k)) * tick_us * 1e3)];
                ++gates[std::llround(static_cast<double>(counts(2 * k + 1, 2 * k)) * tick_us * 1e3)];
                ++n;
            }
        }
        const auto text = [](const std::map<std::int64_t, std::int64_t>& m) {
            std::vector<std::string> parts;
            for (const auto& [ns, count] : m) parts.push_back(fmt("%.10g us x %lld", static_cast<double>(ns) * 1e-3, static_cast<long long>(count)));
            return join(parts, ", ");
        };
        Finding& f = rep.add("4.1 4.6", scan, "hardware timer: trigger period and gate length of every projection");
        f.stated = fmt("exposure %g ms, latency %g ms", T.number("scan/exposure_time"), T.number("scan/latency_time"));
        f.found = fmt("%lld projections; periods: %s; gates: %s", static_cast<long long>(n), text(periods).c_str(), text(gates).c_str());
        const double gate_us = T.number("scan/exposure_time") * 1e3, period_us = gate_us + T.number("scan/latency_time") * 1e3;
        bool ok = periods.size() == 1 && gates.size() == 1 && n > 0 && std::fabs(gates.begin()->first * 1e-3 - gate_us) < 1e-6 &&
                  std::fabs(periods.begin()->first * 1e-3 - period_us) < 1e-6;
        if (!ok && n > 0) {
            // two questions kept apart: is there one period and one gate (to a count of the timer), and are they the stated ones
            const auto commonest = [](const std::map<std::int64_t, std::int64_t>& m) {
                std::int64_t best = 0, most = -1;
                for (const auto& [ns, count] : m)
                    if (count > most) most = count, best = ns;
                return best * 1e-3;
            };
            const double period = commonest(periods), gate = commonest(gates);
            bool uniform = true;
            for (const auto& [ns, count] : periods) uniform &= std::fabs(ns * 1e-3 - period) <= coarsest_tick_us * (1 + 1e-9);
            for (const auto& [ns, count] : gates) uniform &= std::fabs(ns * 1e-3 - gate) <= coarsest_tick_us * (1 + 1e-9);
            const bool as_stated = std::fabs(period - period_us) <= coarsest_tick_us * (1 + 1e-9) && std::fabs(gate - gate_us) <= coarsest_tick_us * (1 + 1e-9);
            ok = uniform && as_stated;
            f.found += uniform ? fmt("; every period lies within one count of the timer (%g us) of %.10g us and every gate of %.10g us", coarsest_tick_us, period, gate)
                               : fmt("; periods or gates differ by more than one count of the timer (%g us) from the commonest, %.10g us and %.10g us", coarsest_tick_us, period, gate);
            if (!as_stated)
                f.found += fmt("; the stated exposure plus latency is %.10g us: the period is %+.10g us from it, the gate %+.10g us from the stated exposure", period_us, period - period_us,
                               gate - gate_us);
        }
        if (!ticks.empty()) f.found += "; " + join(ticks);
        f.verdict = ok ? "verified" : "fails";
        f.note = "one period for every trigger means the timer latched no gap; it does not show that the camera delivered a frame for every trigger";
    }
    // ------------------------------------------------------------------ 4.1 arrival times in the frame files
    {
        Finding& f = rep.add("4.1", scan, "arrival time of each frame at the detector software (frame files)");
        std::vector<std::string> parts;
        bool have = false, ok = true;
        double worst = 0;
        for (const Stretch& s : rec.stretches) {
            const auto& t = s.files.time_of_frame;
            const bool whole = t.size() == static_cast<std::size_t>(s.n_frames);
            if ((!whole && t.size() != static_cast<std::size_t>(s.n_on_disk())) || t.size() < 2) continue;
            bool increasing = true;
            for (std::size_t i = 1; i < t.size() && increasing; ++i) increasing = t[i] > t[i - 1];
            if (!increasing) {
                // this detector software stores the time since the previous frame, not a time of arrival
                have = true;
                double lo = 1e30, hi = -1e30, sum = 0;
                for (std::size_t i = 1; i < t.size(); ++i) lo = std::min(lo, t[i]), hi = std::max(hi, t[i]), sum += t[i];
                parts.push_back(fmt("%s: stored as the time since the previous frame%s: mean %.4f ms, %.2f to %.2f ms", s.entry.c_str(),
                                    whole ? "" : " (frames on disk)", sum / (t.size() - 1) * 1e3, lo * 1e3, hi * 1e3));
                if (s.role == Role::Projection && !s.timer_period.empty()) {
                    const double period = s.timer_period[0];
                    worst = std::max(worst, hi / period);
                    ok &= hi < 1.5 * period && lo > 0;
                }
                continue;
            }
            if (!whole) continue;
            have = true;
            double lo = 1e30, hi = -1e30;
            for (std::size_t i = 1; i < t.size(); ++i) lo = std::min(lo, t[i] - t[i - 1]), hi = std::max(hi, t[i] - t[i - 1]);
            const double mean = (t.back() - t.front()) / (t.size() - 1);
            parts.push_back(fmt("%s: mean %.4f ms, %.2f to %.2f ms", s.entry.c_str(), mean * 1e3, lo * 1e3, hi * 1e3));
            if (s.role == Role::Projection && !s.timer_period.empty()) {
                const double period = s.timer_period[0];
                worst = std::max(worst, hi / period);
                ok &= hi < 1.5 * period && lo > 0;
            }
        }
        f.stated = "not part of the master file: time_of_frame in each frame file, a software time";
        f.found = have ? join(parts) : std::string("frame files not read");
        f.verdict = !have || ok ? "unverified" : "fails";
        f.note = have ? fmt("longest projection arrival gap %.2f periods. These are software arrival times; even regular arrivals do not establish one exposure per trigger. Camera frame identifiers or an independent exposure correspondence remain required", worst)
                      : "No complete frame arrival series was checked";
    }
    // ------------------------------------------------------------------ 4.2 roles
    {
        Finding& f = rep.add("4.2", scan, "role of each stretch");
        std::vector<std::string> parts;
        bool ok = true, absent = false;
        for (const Stretch& s : rec.stretches) {
            const std::string lima = s.files.header.text("header/image_key");
            parts.push_back(fmt("%s '%s': sequence %s, image_key %s, frame files image_key %s", s.entry.c_str(), s.title.c_str(),
                                role_name(s.role), s.has_image_key ? std::to_string(s.image_key).c_str() : "absent",
                                lima.empty() ? "?" : lima.c_str()));
            const int expect = s.role == Role::Dark ? 2 : s.role == Role::Flat ? 1 : s.role == Role::Projection ? 0 : -1;
            if (s.has_image_key) ok &= s.image_key == expect;
            else absent = true;
            if (!lima.empty()) ok &= std::atoi(lima.c_str()) == expect;
        }
        f.stated = "image_key 2 dark, 1 flat, 0 projection; a title and a sequence type per stretch";
        f.found = join(parts);
        f.verdict = ok ? "verified" : "fails";
        if (absent) f.note = "a stretch without image_key in the master file has its role only from the sequence's list, its title and its frame files";
    }
    // ------------------------------------------------------------------ 4.4 the stretches
    {
        Finding& f = rep.add("4.4", scan, "stretches of projections and their extent");
        std::vector<std::string> parts;
        for (int i : proj) {
            const Stretch& s = rec.stretches[i];
            if (!s.latched()) continue;
            const auto a = stated_angle(s);
            const std::string z = rec.translation_z[1];
            std::string part = fmt("%s: %lld frames, angle %.4f to %.4f deg, %s %.4f to %.4f mm", s.entry.c_str(),
                                   static_cast<long long>(s.n_frames), a.front(), a.back(), z.c_str(), positioner(s, z), positioner(s, z, true));
            if (!s.z_center.empty()) part += fmt(", latched %.4f to %.4f mm", s.z_center.front(), s.z_center.back());
            else part += ", no height latched (stage at rest)";
            parts.push_back(part);
        }
        f.stated = fmt("sequence '%s'", T.text("scan/sequence").c_str());
        f.found = join(parts);
    }
    // ------------------------------------------------------------------ 4.5 the clocks
    {
        Finding& f = rep.add("4.5", scan, "Unix time against the hardware timer at each trigger");
        std::vector<std::string> parts;
        bool ok = !proj.empty(), monotonic = true;
        double last = -1;
        for (int i : proj) {
            const Stretch& s = rec.stretches[i];
            if (s.epoch_trig.size() != s.timer_trig.size() || s.epoch_trig.empty()) { ok = false; continue; }
            double lo = 1e300, hi = -1e300;
            for (std::size_t k = 0; k < s.epoch_trig.size(); ++k) {
                const double d = (s.epoch_trig[k] - s.epoch_trig[0]) - (s.timer_trig[k] - s.timer_trig[0]);
                lo = std::min(lo, d), hi = std::max(hi, d);
                if (s.epoch_trig[k] <= last) monotonic = false;
                last = s.epoch_trig[k];
            }
            parts.push_back(fmt("%s: within %.2f us over %.0f s", s.entry.c_str(), (hi - lo) * 1e6, s.timer_trig.back() - s.timer_trig.front()));
            ok &= (hi - lo) < 2e-6;
        }
        f.stated = "epoch_trig and timer_trig per projection";
        f.found = join(parts) + (monotonic ? "; times increase through all stretches" : "; times do NOT increase monotonically");
        f.verdict = ok && monotonic ? "verified" : "fails";
        f.note = "a double holds a Unix time of 1.7e9 s to 0.24 us; the agreement is at that limit";
    }
    // ------------------------------------------------------------------ 4.7 monitors
    {
        Finding& f = rep.add("4.7", scan, "beam monitor readings");
        std::vector<std::string> parts;
        for (const Stretch& s : rec.stretches) {
            const Monitor& m = s.monitor;
            if (m.current.size() < 3 || m.epoch.size() != m.current.size()) continue;
            std::int64_t changes = 0, nan = 0;
            double smallest = 1e30;
            for (std::size_t k = 0; k < m.current.size(); ++k) {
                if (!std::isfinite(m.current[k])) { ++nan; continue; }
                if (k && std::isfinite(m.current[k - 1]) && m.current[k] != m.current[k - 1]) {
                    ++changes;
                    smallest = std::min(smallest, std::fabs(m.current[k] - m.current[k - 1]));
                }
            }
            const double span = m.epoch.back() - m.epoch.front();
            parts.push_back(fmt("%s (from %s): %zu readings every %.3f s, value changes every %.2f s, smallest step %.3g mA, %lld NaN",
                                s.entry.c_str(), m.entry.c_str(), m.current.size(), span / (m.current.size() - 1),
                                changes ? span / changes : NAN, changes ? smallest : NAN, static_cast<long long>(nan)));
        }
        f.stated = "ring current, lifetime, refill countdown, single-bunch current with a software time per reading";
        f.found = join(parts);
        f.note = "the delay of a reading against the beam is not recorded";
    }
    // ------------------------------------------------------------------ 4.8 detector configuration
    {
        Finding& f = rep.add("4.8", scan, "detector configuration: master file against the frame files");
        const auto& bin = T.numbers(det + "binning");
        const auto& roi = T.numbers(det + "roi");
        const auto& flip = T.numbers(det + "flipping");
        const auto& size = T.numbers(det + "size");
        const auto& pix = T.numbers(det + "pixel_size");
        f.stated = fmt("%s, sensor pixel %.4g um, binning %gx%g, roi [%g, %g, %g, %g], flipping [%g, %g], size %gx%g, %g bytes per value",
                       T.text(det + "type").c_str(), pix.size() ? pix[0] * 1e6 : NAN, bin.size() > 1 ? bin[0] : NAN, bin.size() > 1 ? bin[1] : NAN,
                       roi.size() > 3 ? roi[0] : NAN, roi.size() > 3 ? roi[1] : NAN, roi.size() > 3 ? roi[2] : NAN, roi.size() > 3 ? roi[3] : NAN,
                       flip.size() > 1 ? flip[0] : NAN, flip.size() > 1 ? flip[1] : NAN, size.size() > 1 ? size[0] : NAN, size.size() > 1 ? size[1] : NAN,
                       T.number(det + "depth", NAN));
        bool ok = true, have = false;
        std::string found;
        for (const Stretch& s : rec.stretches) {
            const Values& h = s.files.header;
            if (!h.has_number("image_operation/binning/x")) continue;
            const bool same = bin.size() == 2 && roi.size() == 4 && flip.size() == 2 && size.size() == 2 &&
                              h.number("image_operation/binning/x") == bin[0] && h.number("image_operation/binning/y") == bin[1] &&
                              h.number("image_operation/region_of_interest/xstart") == roi[0] && h.number("image_operation/region_of_interest/ystart") == roi[1] &&
                              h.number("image_operation/region_of_interest/xsize") == roi[2] && h.number("image_operation/region_of_interest/ysize") == roi[3] &&
                              h.number("image_operation/flipping/x") == flip[0] && h.number("image_operation/flipping/y") == flip[1] &&
                              h.number("image_operation/dimension/xsize") == size[0] && h.number("image_operation/dimension/ysize") == size[1] &&
                              s.cols == static_cast<int>(size[0]) && s.rows == static_cast<int>(size[1]);
            ok &= same;
            if (!have)
                found = fmt("frame files: sensor %gx%g, bin_mode %s, rotation %s, value type %s, trigger %s; ", h.number("detector_information/max_image_size/xsize", NAN),
                            h.number("detector_information/max_image_size/ysize", NAN), h.text("image_operation/bin_mode").c_str(),
                            h.text("image_operation/rotation").c_str(), h.text("detector_information/image_lima_type").c_str(), h.text("acquisition/trigger_mode").c_str());
            have = true;
        }
        f.found = have ? found + (ok ? "binning, region, flips and size agree with the master file in every stretch" : "the frame files DISAGREE with the master file")
                       : std::string("frame files not read");
        f.verdict = !have ? "unverified" : ok ? "verified" : "fails";
        f.note = "the frame files state the sensor's full size and how the binning is computed (bin_mode); the order of region and flip, and the camera's exposure mode (4.9), are stated nowhere";
    }
    // ------------------------------------------------------------------ 5.1.1, 5.1.2, 5.2.1 pixel size and distances
    {
        Finding& f = rep.add("5.1.1 5.1.2 5.2.1", scan, "pixel size and distances");
        const double sample = T.number("scan/sample_pixel_size"), optics = T.number("optic/optics_pixel_size", NAN);
        const double ss = T.number("scan/source_sample_distance"), sd = T.number("scan/sample_detector_distance");
        const auto& pix = T.numbers(det + "pixel_size");
        const auto& bin = T.numbers(det + "binning");
        const double sensor = pix.empty() || bin.empty() ? NAN : pix[0] * 1e6 * bin[0];
        f.stated = fmt("sample pixel %.6g um, optics pixel %.9g um, source to sample %.6g mm, sample to detector %.8g mm, effective propagation %.8g mm, energy %g keV",
                       sample, optics, ss, sd, T.number("scan/effective_propagation_distance", NAN), T.number("scan/energy", NAN));
        const double mag = (ss + sd) / ss;
        f.found = fmt("optics pixel / sample pixel = %.9f; (source to detector) / (source to sample) = %.9f; binned sensor pixel %.6g um, which is %.6f of the optics pixel; sd / magnification = %.6f mm",
                      optics / sample, mag, sensor, sensor / optics, sd / mag);
        f.verdict = std::fabs(optics / sample - mag) < 1e-9 ? "verified" : "fails";
        f.note = "the optics pixel is the sample pixel times the beam's magnification: the record holds one statement of the pixel size, not two";
        f.numbers = {{"sample_pixel_um", sample}, {"magnification", mag}};
    }
    // ------------------------------------------------------------------ 5.1.6, 5.1.7 senses
    {
        Finding& f = rep.add("5.1.6 5.1.7", scan, "senses of the detector axes and of the rotation");
        const auto& axes = rec.instrument.texts(rec.detector + "_optic/data_axes");
        const auto& flip = T.numbers(det + "flipping");
        f.stated = fmt("data_axes [%s], flipping [%g, %g], rotation_is_clockwise %g", join(axes, ", ").c_str(), flip.size() > 1 ? flip[0] : NAN,
                       flip.size() > 1 ? flip[1] : NAN, T.number("tomoconfig/rotation_is_clockwise", NAN));
        std::vector<std::string> parts;
        for (int i : proj) {
            const Stretch& s = rec.stretches[i];
            if (s.rot_raw.size() < 3 || s.rot_trig.size() < 2) continue;
            parts.push_back(fmt("%s: angle %s, encoder counts %s", s.entry.c_str(), s.rot_trig.back() > s.rot_trig.front() ? "increases" : "decreases",
                                s.rot_raw.back() > s.rot_raw.front() ? "increase" : "decrease"));
        }
        f.found = join(parts);
        f.note = "the conventions behind these flags are not in the record; the handedness cannot be found from the data";
    }
    // ------------------------------------------------------------------ 5.4.1 angles
    for (int i : proj) {
        const Stretch& s = rec.stretches[i];
        if (!s.latched() || s.rot_raw.size() != 2 * s.rot_trig.size() + 1) continue;
        const std::size_t n = s.rot_trig.size();
        std::vector<double> raw0(n), raw_arc(n), index(n);
        for (std::size_t k = 0; k < n; ++k) {
            raw0[k] = s.rot_raw[2 * k];
            raw_arc[k] = static_cast<double>(s.rot_raw[2 * k + 1]) - s.rot_raw[2 * k];
            index[k] = static_cast<double>(k);
        }
        const Line counts = fit_line(raw0, s.rot_trig, 0, n);  // deg per count
        double arc_err = 0, center_err = 0;
        for (std::size_t k = 0; k < n; ++k) {
            arc_err = std::max(arc_err, std::fabs(s.rot_delta[k] - raw_arc[k] * counts.slope));
            if (!s.rot_center.empty()) {
                double d = s.rot_center[k] - (s.rot_trig[k] + 0.5 * s.rot_delta[k]);
                d -= 360.0 * std::round(d / 360.0);
                center_err = std::max(center_err, std::fabs(d));
            }
        }
        const auto angle = stated_angle(s);
        const Line steps = fit_line(index, angle, 0, n);
        const double nominal = s.scan_parameters.number("step_size", tol.step_deg);
        const double end_error = (n - 1) * (nominal - steps.slope);
        Finding& f = rep.add("5.4.1", scan, "rotation angle of each projection, stretch " + s.entry);
        f.stated = fmt("%s_trig, %s, %s_delta per projection; commanded step %.6g deg", rec.rotation[0].c_str(), s.rot_center_name.c_str(), rec.rotation[0].c_str(), nominal);
        f.found = fmt("derived angles reproduce the raw counts to %.1e deg (trig), %.1e deg (arc), %.1e deg (center); %.6g deg per count = %.1f counts per turn; "
                      "step %.7f deg per frame, departure from a uniform step at most %.2e deg (rms %.2e); counting frames at %.6g deg is off by %.4f deg at the end of the stretch (%.2f px at R)",
                      counts.max_abs, arc_err, center_err, std::fabs(counts.slope), 360.0 / std::fabs(counts.slope), steps.slope, steps.max_abs, steps.rms,
                      nominal, end_error, std::fabs(end_error) / 57.29577951308232 * tol.R);
        f.verdict = counts.max_abs < 1e-6 && arc_err < 1e-6 && center_err < 1e-6 ? "verified" : "fails";
        f.note = fmt("verdict: the derived angles are the raw encoder counts. Diagnostic target on the angle itself: %.6g deg", tol.angle_deg());
        f.numbers = {{"step_deg", steps.slope}, {"deg_per_count", counts.slope}, {"nonuniform_max_deg", steps.max_abs}, {"count_error_end_deg", end_error},
                     {"first_deg", angle.front()}, {"last_deg", angle.back()}};
    }
    // ------------------------------------------------------------------ 5.4.2 arc per exposure; 4.6 from the encoder
    {
        Finding& f = rep.add("5.4.2 4.6", scan, "arc turned during each exposure");
        std::vector<std::string> parts;
        for (int i : proj) {
            const Stretch& s = rec.stretches[i];
            if (!s.latched()) continue;
            const Spread d = spread(s.rot_delta), p = spread(s.rot_period, 0, s.rot_period.size());
            parts.push_back(fmt("%s: %.5f deg (%.5f to %.5f); arc / period = %.5f, times the %.1f ms period = %.3f ms", s.entry.c_str(), d.mean, d.min, d.max,
                                d.mean / p.mean, s.timer_period[0] * 1e3, d.mean / p.mean * s.timer_period[0] * 1e3));
        }
        f.stated = fmt("%s_delta per projection; gate %g ms", rec.rotation[0].c_str(), T.number("scan/exposure_time"));
        f.found = join(parts);
    }
    // ------------------------------------------------------------------ 5.4.3 - 5.4.5, 5.4.10 height along the axis
    for (int i : proj) {
        const Stretch& s = rec.stretches[i];
        if (s.z_center.empty() || s.z_raw.size() != 2 * s.z_trig.size() + 1) continue;
        const std::size_t n = s.z_trig.size();
        std::vector<double> raw0(n);
        for (std::size_t k = 0; k < n; ++k) raw0[k] = s.z_raw[2 * k];
        const Line counts = fit_line(raw0, s.z_trig, 0, n);  // mm per count
        const auto angle = stated_angle(s);
        std::vector<double> turns(n);
        for (std::size_t k = 0; k < n; ++k) turns[k] = (angle[k] - angle[0]) / 360.0;
        const std::size_t settle = std::min<std::size_t>(n / 4, 1000);
        const Line travel = fit_line(turns, s.z_center, settle, n);  // mm per turn, after the start
        std::vector<double> res(n);
        for (std::size_t k = 0; k < n; ++k) res[k] = (s.z_center[k] - (travel.intercept + travel.slope * turns[k])) / px_mm;
        const Spread r_all = spread(res, settle, n), r0 = spread(res, 0, std::min<std::size_t>(100, n)),
                     r1 = spread(res, std::min<std::size_t>(200, n), std::min<std::size_t>(300, n)), r_end = spread(res, n - std::min<std::size_t>(100, n), n);
        // the part of the scatter that repeats once per turn
        long double c = 0, sn = 0;
        for (std::size_t k = settle; k < n; ++k) c += res[k] * std::cos(2 * M_PI * turns[k]), sn += res[k] * std::sin(2 * M_PI * turns[k]);
        const double once = std::sqrt(static_cast<double>(c * c + sn * sn)) * 2 / (n - settle) / std::sqrt(2.0);
        const double commanded = T.number("scan/z_step", NAN);
        const double slave = s.scan_parameters.number("slave_step_size", NAN) * 360.0 / s.scan_parameters.number("step_size", NAN);
        const Spread dz = spread(s.z_delta);
        {
            Finding& f = rep.add("5.4.3 5.4.4", scan, "position along the axis of each projection, stretch " + s.entry);
            f.stated = fmt("%s_trig, %s_center, %s_delta per projection [mm]", rec.translation_z[0].c_str(), rec.translation_z[0].c_str(), rec.translation_z[0].c_str());
            f.found = fmt("derived heights reproduce the raw counts to %.1e mm; %.6g mm per count = %.4f rows at %.4g um; travel during an exposure %.3f um = %.4f rows",
                          counts.max_abs, std::fabs(counts.slope), std::fabs(counts.slope) / px_mm, px_mm * 1e3, dz.mean * 1e3, dz.mean / px_mm);
            f.verdict = counts.max_abs < 1e-9 ? "verified" : "fails";
            f.note = "verdict: the derived heights are the raw encoder counts. Rows follow only through the stated pixel size";
        }
        {
            Finding& f = rep.add("5.4.5", scan, "travel along the axis per turn, stretch " + s.entry);
            f.stated = fmt("commanded: z_step %.6g mm per turn (scan), %.6g mm per turn (slave step x frames per turn)", commanded, slave);
            f.found = fmt("encoder against encoder: %.5f mm per turn = %.3f rows at %.4g um (commanded: %.3f rows); difference %.3f rows per turn against a tolerance of %.4f",
                          travel.slope, travel.slope / px_mm, px_mm * 1e3, commanded / px_mm, (travel.slope - commanded) / px_mm,
                          tol.eps * std::fabs(travel.slope / px_mm) / tol.H);
            assess_residuals(f, {(travel.slope - commanded) / px_mm}, tol.eps * std::fabs(travel.slope / px_mm) / tol.H, "rows/turn");
            f.note = "verdict: commanded against recorded travel. Whether the recorded travel is the sample's, in rows, is for the frames to show";
            f.numbers = {{"mm_per_turn", travel.slope}, {"rows_per_turn", travel.slope / px_mm}, {"commanded_mm_per_turn", commanded}};
        }
        {
            Finding& f = rep.add("5.4.10", scan, "steadiness of the travel along the axis, stretch " + s.entry);
            f.stated = "the latched heights";
            f.found = fmt("against a constant travel per turn (fitted from frame %zu on): frames 0-99 %+.3f rows, frames 200-299 %+.3f rows, last 100 frames %+.3f rows; "
                          "scatter %.3f rows rms (%.3f to %+.3f), of which %.3f rows rms repeat once per turn",
                          settle, r0.mean, r1.mean, r_end.mean, r_all.sd, r_all.min, r_all.max, once);
            assess_residuals(f, res, tol.rows(), "rows");
            f.note = "Residuals test a constant-pitch model, not the validity of the per-frame encoders. Startup transients, periodic structure and scatter all require an explicit trajectory and uncertainty model";
        }
    }
    // ------------------------------------------------------------------ 5.4.6 boundaries between stretches
    for (std::size_t k = 0; k + 1 < proj.size(); ++k) {
        const Stretch& a = rec.stretches[proj[k]];
        const Stretch& b = rec.stretches[proj[k + 1]];
        if (!a.latched() || !b.latched()) continue;
        const auto aa = stated_angle(a), ab = stated_angle(b);
        const auto za = stated_height(rec, a), zb = stated_height(rec, b);
        const double step = (aa.back() - aa.front()) / (aa.size() - 1);
        const double gap = ab.front() - aa.back();
        const double dz = zb.front() - za.back();
        // travel per frame of whichever stretch moves along the axis
        const auto& zm = !a.z_center.empty() ? za : zb;
        const double per_frame = (zm.back() - zm.front()) / (zm.size() - 1);
        Finding& f = rep.add("5.4.6", scan, "boundary between stretches " + a.entry + " and " + b.entry);
        f.stated = fmt("angle %.4f -> %.4f deg; height %.4f -> %.4f mm (%s)", aa.back(), ab.front(), za.back(), zb.front(),
                       a.z_center.empty() || b.z_center.empty() ? "one side from the stage's position at rest, the other latched" : "both latched");
        f.found = fmt("angle gap %.4f deg where one step is %.4f deg: %.4f deg more than a step (%.2f px at R); height step %.2f um where one frame travels %.2f um: %+.2f rows off a continuous series",
                      gap, step, gap - step, std::fabs(gap - step) / 57.29577951308232 * tol.R, dz * 1e3, per_frame * 1e3, (dz - per_frame) / px_mm);
        f.numbers = {{"angle_gap_deg", gap}, {"step_deg", step}, {"height_step_mm", dz}};
        f.note = "these are the offsets a reconstruction makes if it counts the frames of the stretches as one series";
    }
    // ------------------------------------------------------------------ 5.3.1 nominal axis column
    {
        Finding& f = rep.add("5.3.1", scan, "axis column from the lateral stage (nominal)");
        const Stretch& s = rec.stretches[proj.empty() ? 0 : proj[0]];
        const double y = positioner(s, rec.translation_y[1]);
        f.stated = fmt("%s = %.4f mm at the start of %s", rec.translation_y[1].c_str(), y, s.entry.c_str());
        const double middle = T.numbers(det + "size").at(0) / 2;
        f.found = fmt("%.2f px at %.4g um; with the assumption that %s = 0 puts the axis on the middle column (%.1f): column %.2f", y / px_mm, px_mm * 1e3,
                      rec.translation_y[1].c_str(), middle, middle + y / px_mm);
        f.note = "the assumption is stated nowhere in the record, nor is the sign";
        bool moved = false;
        for (int i : proj) moved |= positioner(rec.stretches[i], rec.translation_y[1]) != y || positioner(rec.stretches[i], rec.translation_y[1], true) != y;
        if (moved) f.note += "; the stage's position CHANGES between the projection stretches";
        f.numbers = {{"yrot_mm", y}, {"nominal_axis_column", middle + y / px_mm}};
    }
    // ------------------------------------------------------------------ 8.1 recorded levels
    {
        Finding& f = rep.add("8.1.1 8.1.4 8.1.7", scan, "recorded frame statistics (middle rows only)");
        std::vector<std::string> parts;
        for (const Stretch& s : rec.stretches) {
            if (s.stat_avg.empty()) continue;
            const Spread avg = spread(s.stat_avg), mx = spread(s.stat_max), mn = spread(s.stat_min);
            std::int64_t top = 0, first = -1;
            for (std::size_t k = 0; k < s.stat_max.size(); ++k)
                if (s.stat_max[k] >= 65535) {
                    ++top;
                    if (first < 0) first = static_cast<std::int64_t>(k);
                }
            std::string part = fmt("%s %s: mean %.2f (%.2f to %.2f), lowest value %.0f, highest %.0f", s.entry.c_str(), role_name(s.role), avg.mean, avg.min, avg.max, mn.min, mx.max);
            if (top) part += fmt(", %lld frames reach 65535 (first: frame %lld)", static_cast<long long>(top), static_cast<long long>(first));
            parts.push_back(part);
        }
        const Stretch& s0 = rec.stretches[0];
        f.stated = fmt("avg, min, max, std, sum per frame over columns %d..%d, rows %d..%d", s0.stat_x, s0.stat_x + s0.stat_w - 1, s0.stat_y, s0.stat_y + s0.stat_h - 1);
        f.found = join(parts);
        f.note = "the statistics cover the middle of the frame only; the frames themselves are checked by the inventory";
        if (parts.empty()) {
            f.stated = "nothing: the record holds no per-frame statistics of the detector software";
            f.verdict = "unverified";
            f.note = "without them no frame on disk can be shown to be the frame the detector software saw (4.1), and the levels come from the frames alone";
        }
    }
    // ------------------------------------------------------------------ 8.2.3 flats against the ring current
    {
        Finding& f = rep.add("8.2.3", scan, "flat level against the ring current (recorded statistics)");
        const double dark = recorded_dark_level(rec);
        std::vector<std::string> parts;
        std::vector<double> per_ma;
        for (int i : rec.stretches_with(Role::Flat)) {
            const Stretch& s = rec.stretches[i];
            if (s.stat_avg.size() < 40 || s.monitor.current.size() != s.stat_avg.size()) continue;
            const std::size_t n = s.stat_avg.size();
            const double level = mean_finite(s.stat_avg) - dark, current = mean_finite(s.monitor.current);
            const double l0 = spread(s.stat_avg, 0, 20).mean - dark, l1 = spread(s.stat_avg, n - 20, n).mean - dark;
            const double c0 = spread(s.monitor.current, 0, 20).mean, c1 = spread(s.monitor.current, n - 20, n).mean;
            per_ma.push_back(level / current);
            parts.push_back(fmt("%s: %.1f counts above dark at %.3f mA = %.2f per mA; within the series the level changes by %+.2f %% and the current by %+.2f %% over %.0f s",
                                s.entry.c_str(), level, current, level / current, (l1 / l0 - 1) * 100, (c1 / c0 - 1) * 100,
                                s.epoch.size() == n ? s.epoch.back() - s.epoch.front() : NAN));
        }
        f.stated = "ring current read with every flat; per-frame mean level of the flats";
        f.found = join(parts);
        if (per_ma.size() > 1) {
            const Spread p = spread(per_ma);
            f.found += fmt("; level per mA differs by %.2f %% between the series", (p.max / p.min - 1) * 100);
            f.numbers = {{"per_mA_spread_percent", (p.max / p.min - 1) * 100}};
        }
        f.note = "if the ring current were proportional to the beam at the detector, the level per mA would be one number";
        if (parts.empty()) {
            f.found = "no flat series has recorded statistics: see the flat frames themselves (series)";
            f.verdict = "unverified";
        }
    }
    // ------------------------------------------------------------------ 8.2.4 refills
    {
        Finding& f = rep.add("8.2.4", scan, "rises of the ring current during the projections");
        std::vector<std::string> parts;
        for (int i : proj) {
            const Stretch& s = rec.stretches[i];
            const Monitor& m = s.monitor;
            if (m.current.size() < 2 || s.epoch_trig.empty()) continue;
            double t0 = NAN, t1 = NAN, c0 = NAN, c1 = NAN, prev = NAN, prev_t = NAN;
            for (std::size_t k = 0; k < m.current.size(); ++k) {
                if (!std::isfinite(m.current[k])) continue;
                if (std::isfinite(prev) && m.current[k] > prev + 0.05) {
                    if (!std::isfinite(t0)) t0 = prev_t, c0 = prev;
                    t1 = m.epoch[k], c1 = m.current[k];
                }
                prev = m.current[k], prev_t = m.epoch[k];
            }
            const double first = mean_between(m.epoch, m.current, m.epoch.front(), m.epoch.front() + 2), last = mean_between(m.epoch, m.current, m.epoch.back() - 2, m.epoch.back());
            std::string part = fmt("%s: %.2f mA -> %.2f mA", s.entry.c_str(), first, last);
            if (std::isfinite(t0)) {
                const auto frame_at = [&](double t) { return static_cast<long long>(std::lower_bound(s.epoch_trig.begin(), s.epoch_trig.end(), t) - s.epoch_trig.begin()); };
                part += fmt(", rising from %.2f to %.2f mA between projections %lld and %lld (%.1f s)", c0, c1, frame_at(t0), frame_at(t1), t1 - t0);
                f.numbers.push_back({"rise_first_frame_" + s.entry, static_cast<double>(frame_at(t0))});
                f.numbers.push_back({"rise_last_frame_" + s.entry, static_cast<double>(frame_at(t1))});
            } else {
                part += ", no rise";
            }
            // the largest steps of the recorded frame mean
            if (s.stat_avg.size() > 2) {
                std::size_t at = 1;
                double biggest = 0;
                for (std::size_t k = 1; k < s.stat_avg.size(); ++k) {
                    const double d = s.stat_avg[k] / s.stat_avg[k - 1] - 1;
                    if (std::fabs(d) > std::fabs(biggest)) biggest = d, at = k;
                }
                part += fmt("; largest step of the frame mean %+.2f %% at projection %zu", biggest * 100, at);
            }
            parts.push_back(part);
        }
        f.stated = "ring current every 0.5 s through the projections";
        f.found = join(parts);
        f.note = "frame numbers are placed by the readings' software time against epoch_trig; the monitor's delay is unknown";
    }
}

void check_record_pair(const Record& a, const Record& b, Report& rep, double eps) {
    const std::string scan = a.label + "+" + b.label;
    const Tolerances tol = tolerances(a, eps);
    const double px_mm = pixel_um(a) * 1e-3;
    const auto pa = a.stretches_with(Role::Projection), pb = b.stretches_with(Role::Projection);

    // ------------------------------------------------------------------ 4.3 which is which
    {
        Finding& f = rep.add("4.3", scan, "the scans of the acquisition and what differs between them");
        // every motor position that differs at the start of the first projection stretch
        std::vector<std::string> diff;
        if (!pa.empty() && !pb.empty()) {
            const auto& ma = a.stretches[pa[0]].positioners_start;
            const auto& mb = b.stretches[pb[0]].positioners_start;
            for (const auto& [name, value] : ma) {
                const auto it = mb.find(name);
                if (it != mb.end() && std::fabs(it->second - value) > 1e-6) diff.push_back(fmt("%s %.4f -> %.4f", name.c_str(), value, it->second));
            }
        }
        f.stated = fmt("'%s' (%s) and '%s' (%s)", a.sample_name.c_str(), a.start_time.c_str(), b.sample_name.c_str(), b.start_time.c_str());
        f.found = "motor positions that differ at the start of the first projection stretch: " + (diff.empty() ? std::string("none") : join(diff, ", "));
        const bool same_pixel = pixel_um(a) == pixel_um(b) && a.stretches.size() == b.stretches.size();
        f.verdict = same_pixel ? "note" : "fails";
    }
    // ------------------------------------------------------------------ 6.1.1 lateral move
    {
        Finding& f = rep.add("6.1.1", scan, "lateral move between the scans (stage reading)");
        const std::string y = a.translation_y[1];
        const double ya = positioner(a.stretches[pa.at(0)], y), yb = positioner(b.stretches[pb.at(0)], y);
        bool constant = true;
        for (int i : pa) constant &= positioner(a.stretches[i], y) == ya && positioner(a.stretches[i], y, true) == ya;
        for (int i : pb) constant &= positioner(b.stretches[i], y) == yb && positioner(b.stretches[i], y, true) == yb;
        f.stated = fmt("%s %.4f mm and %.4f mm", y.c_str(), ya, yb);
        f.found = fmt("move %.4f mm = %.3f px at %.4g um; the reading is %s at the start and end of every projection stretch; %.4g px is %.1e of the move", yb - ya,
                      (yb - ya) / px_mm, px_mm * 1e3, constant ? "the same" : "NOT the same", tol.eps, tol.eps / std::fabs((yb - ya) / px_mm));
        f.numbers = {{"move_mm", yb - ya}, {"move_px", (yb - ya) / px_mm}};
    }
    // ------------------------------------------------------------------ per stretch: 6.1.2, 6.2.1 - 6.2.4
    for (std::size_t k = 0; k < pa.size() && k < pb.size(); ++k) {
        const Stretch& sa = a.stretches[pa[k]];
        const Stretch& sb = b.stretches[pb[k]];
        if (!sa.latched() || !sb.latched() || sa.n_frames != sb.n_frames) continue;
        const auto aa = stated_angle(sa), ab = stated_angle(sb);
        const auto za = stated_height(a, sa), zb = stated_height(b, sb);
        std::vector<double> da(aa.size()), dz(aa.size()), dt(aa.size());
        for (std::size_t i = 0; i < aa.size(); ++i) da[i] = ab[i] - aa[i], dz[i] = zb[i] - za[i], dt[i] = sb.epoch_trig[i] - sa.epoch_trig[i];
        const Spread sda = spread(da), sdz = spread(dz), sdt = spread(dt);
        {
            Finding& f = rep.add("6.2.1", scan, "angle of the two scans at equal projection index, stretch " + sa.entry);
            f.stated = "the encoder angles of both scans";
            const double worst = std::max(std::fabs(sda.min), std::fabs(sda.max));
            f.found = fmt("difference %+.2e deg on average, at most %.2e deg (%.3f px at R); target %.6g deg", sda.mean, worst, worst / 57.29577951308232 * tol.R, tol.angle_deg());
            assess_residuals(f, da, tol.angle_deg(), "deg");
            f.note = "a statement about the stage's encoder. The sample's angle against the beam in each scan (6.1.6) is for the frames to show";
            f.numbers = {{"mean_deg", sda.mean}, {"max_abs_deg", worst}};
        }
        {
            Finding& f = rep.add("6.1.2 6.2.2 6.2.3", scan, "height of the two scans at equal projection index, stretch " + sa.entry);
            f.stated = sa.z_center.empty() ? "the stage's position at rest in both scans" : "the latched heights of both scans";
            f.found = fmt("second minus first: %+.2f um = %+.3f rows (rms %.2f um about that, %+.2f to %+.2f um); first frames at %.4f and %.4f mm", sdz.mean * 1e3,
                          sdz.mean / px_mm, sdz.sd * 1e3, sdz.min * 1e3, sdz.max * 1e3, za.front(), zb.front());
            f.numbers = {{"mean_mm", sdz.mean}, {"rms_mm", sdz.sd}};
        }
        {
            Finding& f = rep.add("6.2.4", scan, "time between corresponding projections, stretch " + sa.entry);
            f.stated = "epoch_trig of both scans";
            f.found = fmt("%.1f s (%.4f to %.4f)", sdt.mean, sdt.min, sdt.max);
        }
    }
    // ------------------------------------------------------------------ 8.1.2 dark level of the two scans
    {
        Finding& f = rep.add("8.1.2", scan, "recorded dark level of the two scans");
        const double da = recorded_dark_level(a), db = recorded_dark_level(b);
        f.stated = "per-frame mean of the dark frames (middle rows)";
        f.found = fmt("%.3f and %.3f counts: %+.3f", da, db, db - da);
        f.numbers = {{"first", da}, {"second", db}};
    }
}

bool lateral_stage_differs(const Record& a, const Record& b) {
    const auto pa = a.stretches_with(Role::Projection), pb = b.stretches_with(Role::Projection);
    if (pa.empty() || pb.empty()) return false;
    return std::fabs(positioner(a.stretches[pa[0]], a.translation_y[1]) - positioner(b.stretches[pb[0]], b.translation_y[1])) > 1e-6;
}

namespace {

// the first and second name of a motor of the tomo configuration ("sample_u" -> {"mrsx", "sx"}); "" if absent
std::string config_motor(const Record& rec, const std::string& key) {
    const auto& v = rec.technique.texts("tomoconfig/" + key);
    return v.empty() ? std::string() : v.back();
}

struct Placement {
    bool have = false;
    double u = 0, v = 0;  // mm, the stages on the rotation
    bool constant = true;
};

Placement placement_of(const Record& rec) {
    Placement p;
    const std::string mu = config_motor(rec, "sample_u"), mv = config_motor(rec, "sample_v");
    const auto proj = rec.stretches_with(Role::Projection);
    if (mu.empty() || mv.empty() || proj.empty()) return p;
    const Stretch& s0 = rec.stretches[proj[0]];
    if (!s0.positioners_start.count(mu) || !s0.positioners_start.count(mv)) return p;
    p.have = true;
    p.u = positioner(s0, mu), p.v = positioner(s0, mv);
    for (int i : proj)
        for (bool end : {false, true}) p.constant &= positioner(rec.stretches[i], mu, end) == p.u && positioner(rec.stretches[i], mv, end) == p.v;
    return p;
}

}  // namespace

// Scans that differ in where the sample sits on the rotation stage (a mosaic), or any number of scans above two.
// A sample displaced by d on the rotation stage is seen at angle t and column c along the sample's line
// s = c - axis - n(t).d: two tiles record the same lines at the same angle on columns n(t).(d_j - d_k) apart.
void check_record_tiles(const std::vector<Record>& recs, Report& rep, double eps, double radius_px) {
    if (recs.size() < 2) return;
    const Record& first = recs[0];
    const std::string all = first.label + "+" + std::to_string(recs.size() - 1);
    const double px_mm = pixel_um(first) * 1e-3;
    const auto& size = first.technique.numbers("detector/" + first.detector + "/size");
    const double cols = size.at(0);
    std::vector<std::vector<int>> proj;
    for (const Record& r : recs) proj.push_back(r.stretches_with(Role::Projection));

    // ------------------------------------------------------------------ 4.3 which is which
    {
        Finding& f = rep.add("4.3", all, "the scans of the acquisition and what differs between them");
        std::map<std::string, std::vector<double>> values;
        bool comparable = true;
        std::vector<std::string> names;
        for (std::size_t k = 0; k < recs.size(); ++k) {
            names.push_back(fmt("'%s' (%s)", recs[k].sample_name.c_str(), recs[k].start_time.c_str()));
            comparable &= pixel_um(recs[k]) == pixel_um(first) && recs[k].stretches.size() == first.stretches.size() && !proj[k].empty();
        }
        if (comparable) {
            const auto& m0 = first.stretches[proj[0][0]].positioners_start;
            for (const auto& [name, value] : m0) {
                bool differs = false;
                std::vector<double> v;
                for (std::size_t k = 0; k < recs.size(); ++k) {
                    const auto& mk = recs[k].stretches[proj[k][0]].positioners_start;
                    const auto it = mk.find(name);
                    v.push_back(it == mk.end() ? NAN : it->second);
                    differs |= it != mk.end() && std::fabs(it->second - value) > 1e-6;
                }
                if (differs) values[name] = v;
            }
        }
        std::vector<std::string> diff;
        for (const auto& [name, v] : values) {
            std::vector<std::string> each;
            for (double x : v) each.push_back(fmt("%.4f", x));
            diff.push_back(name + " " + join(each, ", "));
        }
        f.stated = join(names, ", ");
        f.found = "motor positions that differ at the start of the first projection stretch: " + (diff.empty() ? std::string("none") : join(diff));
        const std::string mu = config_motor(first, "sample_u"), mv = config_motor(first, "sample_v");
        const bool lateral = values.count(first.translation_y[1]) > 0, sample = values.count(mu) > 0 || values.count(mv) > 0;
        f.found += lateral ? "; the lateral stage of the rotation axis differs: scans at different places of the axis against the detector"
                 : sample ? "; the sample's stages on the rotation differ and the axis stays: tiles of a mosaic"
                          : "; neither the lateral stage nor the sample's stages differ";
        f.verdict = comparable ? "note" : "fails";
        f.numbers = {{"scans", static_cast<double>(recs.size())}, {"mosaic", sample && !lateral ? 1.0 : 0.0}};
    }

    // ------------------------------------------------------------------ 6.1.1 where the sample sits in each scan
    std::vector<Placement> place;
    for (const Record& r : recs) place.push_back(placement_of(r));
    bool placed = true;
    for (const Placement& p : place) placed &= p.have;
    // the nominal axis column (the assumption of 5.3.1) and the farthest a tile sees from its axis
    const double y_mm = proj[0].empty() ? 0.0 : positioner(first.stretches[proj[0][0]], first.translation_y[1]);
    const double axis = cols / 2 + y_mm / px_mm;
    const double reach = std::max(axis, cols - 1 - axis);
    double lever = reach;
    if (placed) {
        Finding& f = rep.add("6.1.1", all, "place of the sample on the rotation stage in each scan (stage readings)");
        const std::string mu = config_motor(first, "sample_u"), mv = config_motor(first, "sample_v");
        std::vector<std::string> parts;
        bool constant = true;
        for (std::size_t k = 0; k < recs.size(); ++k) {
            const double du = (place[k].u - place[0].u) / px_mm, dv = (place[k].v - place[0].v) / px_mm;
            parts.push_back(fmt("%s: %.4f, %.4f mm = (%+.2f, %+.2f) px from %s", recs[k].label.c_str(), place[k].u, place[k].v, du, dv, first.label.c_str()));
            constant &= place[k].constant;
            f.numbers.push_back({"u_px_" + recs[k].label, du});
            f.numbers.push_back({"v_px_" + recs[k].label, dv});
            for (std::size_t j = 0; j < recs.size(); ++j)
                lever = std::max(lever, std::hypot(place[k].u - place[j].u, place[k].v - place[j].v) / px_mm + reach);
        }
        f.stated = fmt("%s and %s per scan", mu.c_str(), mv.c_str());
        f.found = join(parts) + fmt("; the readings are %s at the start and end of every projection stretch; %.4g px is %.1e of the distance between neighbours",
                                    constant ? "the same" : "NOT the same", eps, eps / std::max(1.0, lever - reach));
        // the stage's own statement of the sample's place in the laboratory: the pseudo-motors, which turn with the rotation
        const std::string mx = config_motor(first, "sample_x"), my = config_motor(first, "sample_y"), rot = first.rotation[1];
        double worst[2] = {0, 0};
        bool have_lab = !mx.empty() && !my.empty();
        for (std::size_t k = 0; k < recs.size() && have_lab; ++k)
            for (int i : proj[k]) {
                const Stretch& s = recs[k].stretches[i];
                if (!s.positioners_start.count(mx) || !s.positioners_start.count(my) || !s.positioners_start.count(rot)) { have_lab = false; break; }
                const double a = positioner(s, rot) * M_PI / 180, x = positioner(s, mx), y = positioner(s, my);
                for (int sense = 0; sense < 2; ++sense) {
                    const double sn = (sense ? -1 : 1) * std::sin(a), cs = std::cos(a);
                    worst[sense] = std::max({worst[sense], std::fabs(x - (place[k].u * cs + place[k].v * sn)), std::fabs(y - (-place[k].u * sn + place[k].v * cs))});
                }
            }
        if (have_lab) {
            const int sense = worst[1] < worst[0] ? 1 : 0;
            f.found += fmt("; %s = %s cos(%s) %s %s sin(%s) and %s = %s %s sin(%s) + %s cos(%s) at the start of every projection stretch to %.2g um", mx.c_str(), mu.c_str(), rot.c_str(),
                           sense ? "-" : "+", mv.c_str(), rot.c_str(), my.c_str(), sense ? "" : "-", mu.c_str(), rot.c_str(), mv.c_str(), rot.c_str(), worst[sense] * 1e3);
            f.numbers.push_back({"lab_sense", sense ? -1.0 : 1.0});
            f.numbers.push_back({"lab_worst_mm", worst[sense]});
        }
        f.note = "stage readings. Where the sample was is for the frames to show: two scans record the same lines at equal angle on columns n(angle).(d_j - d_k) apart";
    }
    Tolerances tol = tolerances(first, eps, radius_px > 0 ? radius_px : lever);

    // ------------------------------------------------------------------ 6.3.1 which lines each scan records
    if (placed) {
        Finding& f = rep.add("6.3.1 5.3.1", all, "which lines the scans record, from the stage readings and the nominal axis column");
        const int window = 1024, steps = 720;
        std::vector<std::string> parts;
        int pairs = 0;
        for (std::size_t k = 0; k < recs.size(); ++k)
            for (std::size_t j = k + 1; j < recs.size(); ++j) {
                const double du = (place[j].u - place[k].u) / px_mm, dv = (place[j].v - place[k].v) / px_mm, D = std::hypot(du, dv);
                if (D >= 2 * reach) continue;
                int equal = 0, opposite = 0;
                double narrowest = 1e30;
                for (int t = 0; t < steps; ++t) {
                    const double a = 2 * M_PI * t / steps, delta = du * std::cos(a) + dv * std::sin(a);
                    const double w_equal = cols - std::fabs(delta);
                    // one scan at this angle against the other half a turn on, either way round
                    const auto opp = [&](double d) { return std::min(cols - 1 - axis, d + axis) - std::max(-axis, d - (cols - 1 - axis)); };
                    const double w_opp = std::max(opp(delta), opp(-delta));
                    equal += w_equal >= window;
                    opposite += w_opp >= window;
                    narrowest = std::min(narrowest, std::max(w_equal, w_opp));
                }
                ++pairs;
                parts.push_back(fmt("%s-%s %.1f px apart: %.0f %% of the angles at equal angle, %.0f %% in opposite views, narrowest %.0f columns", recs[k].label.c_str(),
                                    recs[j].label.c_str(), D, 100.0 * equal / steps, 100.0 * opposite / steps, narrowest));
            }
        // how many scans record a line (either view), over the lines between the outermost any scan records
        std::int64_t count[3] = {0, 0, 0}, total = 0;
        double hull_min = 1e30, hull_max = 0;
        for (int t = 0; t < steps; ++t) {
            const double a = 2 * M_PI * t / steps;
            std::vector<double> centre;
            for (const Placement& p : place) centre.push_back((p.u * std::cos(a) + p.v * std::sin(a)) / px_mm);
            const double lo = *std::min_element(centre.begin(), centre.end()) - reach, hi = *std::max_element(centre.begin(), centre.end()) + reach;
            hull_min = std::min(hull_min, (hi - lo) / 2), hull_max = std::max(hull_max, (hi - lo) / 2);
            for (double sline = lo + 4; sline < hi; sline += 8) {
                int n = 0;
                for (double c : centre) n += std::fabs(sline - c) <= reach;
                ++count[std::min(n, 2)];
                ++total;
            }
        }
        f.stated = fmt("axis on column %.1f of %.0f (nominal): a scan records the lines within %.1f px of its axis, from one side or from both", axis, cols, reach);
        f.found = fmt("%d pairs of scans closer than twice that; lines both record, in windows of at least %d columns: %s; of the lines between the outermost any scan records "
                      "(%.0f to %.0f px wide, half): %.1f %% by no scan, %.1f %% by one, %.1f %% by two or more; the farthest a scan's rays pass from its own axis inside that: %.0f px",
                      pairs, window, join(parts).c_str(), hull_min, hull_max, 100.0 * count[0] / total, 100.0 * count[1] / total, 100.0 * count[2] / total, lever);
        f.note = "nominal: the stage readings, the stated pixel size and the assumed axis column. The lever arm of every angle is the last number, not the detector's width";
        f.numbers = {{"pairs", static_cast<double>(pairs)}, {"reach_px", reach}, {"lever_px", lever}, {"uncovered_percent", 100.0 * count[0] / total}};
    }

    // ------------------------------------------------------------------ per stretch: 6.1.2, 6.2.1 - 6.2.4
    for (std::size_t k = 0; k < proj[0].size(); ++k) {
        const Stretch& sa = first.stretches[proj[0][k]];
        if (!sa.latched()) continue;
        const auto aa = stated_angle(sa);
        const auto za = stated_height(first, sa);
        std::vector<double> all_da;
        std::vector<std::string> angle_parts, height_parts, time_parts;
        double worst_angle = 0;
        std::vector<std::pair<std::string, double>> numbers;
        for (std::size_t r = 1; r < recs.size(); ++r) {
            if (k >= proj[r].size()) continue;
            const Stretch& sb = recs[r].stretches[proj[r][k]];
            if (!sb.latched() || sb.n_frames != sa.n_frames) continue;
            const auto ab = stated_angle(sb);
            const auto zb = stated_height(recs[r], sb);
            std::vector<double> da(aa.size()), dz(aa.size());
            for (std::size_t i = 0; i < aa.size(); ++i) da[i] = ab[i] - aa[i], dz[i] = zb[i] - za[i];
            const Spread sda = spread(da), sdz = spread(dz);
            const double worst = std::max(std::fabs(sda.min), std::fabs(sda.max));
            worst_angle = std::max(worst_angle, worst);
            all_da.insert(all_da.end(), da.begin(), da.end());
            angle_parts.push_back(fmt("%s %+.2e (at most %.2e)", recs[r].label.c_str(), sda.mean, worst));
            height_parts.push_back(fmt("%s %+.2f um = %+.3f rows (rms %.2f um, %+.2f to %+.2f)", recs[r].label.c_str(), sdz.mean * 1e3, sdz.mean / px_mm, sdz.sd * 1e3, sdz.min * 1e3, sdz.max * 1e3));
            time_parts.push_back(fmt("%s %.1f s", recs[r].label.c_str(), sb.epoch_trig.empty() || sa.epoch_trig.empty() ? NAN : sb.epoch_trig[0] - sa.epoch_trig[0]));
            numbers.push_back({"height_mean_mm_" + recs[r].label, sdz.mean});
            numbers.push_back({"angle_mean_deg_" + recs[r].label, sda.mean});
        }
        if (angle_parts.empty()) continue;
        {
            Finding& f = rep.add("6.2.1", all, "angle of the scans at equal projection index, stretch " + sa.entry);
            f.stated = "the encoder angles of every scan";
            f.found = fmt("against %s [deg]: %s; at most %.2e deg = %.3f px at the lever arm of %.0f px; target %.6g deg", first.label.c_str(), join(angle_parts).c_str(), worst_angle,
                          worst_angle / 57.29577951308232 * tol.R, tol.R, tol.angle_deg());
            assess_residuals(f, all_da, tol.angle_deg(), "deg");
            f.note = "a statement about the stage's encoder. The sample's angle against the beam in each scan (6.1.6) is for the frames to show";
            f.numbers = numbers;
        }
        {
            Finding& f = rep.add("6.1.2 6.2.2 6.2.3", all, "height of the scans at equal projection index, stretch " + sa.entry);
            f.stated = sa.z_center.empty() ? "the stage's position at rest in every scan" : "the latched heights of every scan";
            f.found = fmt("minus %s: %s", first.label.c_str(), join(height_parts).c_str());
        }
        {
            Finding& f = rep.add("6.2.4", all, "time between corresponding projections, stretch " + sa.entry);
            f.stated = "epoch_trig of every scan";
            f.found = fmt("after %s: %s", first.label.c_str(), join(time_parts).c_str());
        }
    }

    // ------------------------------------------------------------------ 6.3.5 the beam through the scans
    {
        Finding& f = rep.add("6.3.5 8.2.3", all, "ring current through the scans");
        std::vector<std::string> parts;
        for (const Record& r : recs) {
            double lo = 1e30, hi = -1e30, flats = 0, n_flats = 0, start = NAN, end = NAN;
            for (const Stretch& s : r.stretches) {
                for (double c : s.monitor.current) {
                    if (!std::isfinite(c)) continue;
                    if (s.role == Role::Flat) flats += c, n_flats += 1;
                    if (s.role != Role::Projection) continue;
                    if (!std::isfinite(start)) start = c;
                    end = c;
                    lo = std::min(lo, c), hi = std::max(hi, c);
                }
            }
            parts.push_back(fmt("%s: %.2f -> %.2f mA through the projections (%.2f to %.2f), flats at %.2f mA", r.label.c_str(), start, end, lo, hi, n_flats ? flats / n_flats : NAN));
        }
        f.stated = "ring current read through every stretch; each scan has its own darks and flats";
        f.found = join(parts);
        f.note = "the level of one scan against another where they record the same lines is for the frames to show";
    }
}

}  // namespace helirec::calibration
