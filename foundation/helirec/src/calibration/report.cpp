#include <cmath>
#include <sstream>

#include "calibration/util.hpp"
#include "helirec/calibration/record_checks.hpp"

namespace helirec::calibration {

void assess_residuals(Finding& f, const std::vector<double>& residuals, double target, const std::string& unit) {
    if (!std::isfinite(target) || target <= 0) throw Error("residual target must be finite and positive");
    ResidualCheck r;
    r.unit = unit;
    r.target = target;
    std::vector<double> finite;
    finite.reserve(residuals.size());
    long double squares = 0;
    for (double x : residuals) {
        if (!std::isfinite(x)) { ++r.invalid; continue; }
        finite.push_back(x);
        squares += static_cast<long double>(x) * x;
        r.nonzero += x != 0;
        r.above_target += std::fabs(x) > target;
    }
    r.samples = finite.size();
    if (r.samples) {
        const Spread s = spread(finite);
        r.mean = s.mean; r.min = s.min; r.max = s.max;
        r.max_abs = std::max(std::fabs(s.min), std::fabs(s.max));
        r.rms = std::sqrt(static_cast<double>(squares / r.samples));
        r.block_sem = s.sem();
    } else {
        r.mean = r.min = r.max = r.max_abs = r.rms = r.block_sem = NAN;
    }
    f.residuals = r;
    f.verdict = r.above_target ? "fails" : "unverified";
}

Finding& Report::add(std::string requirement, std::string scan, std::string quantity) {
    Finding f;
    f.requirement = std::move(requirement);
    f.scan = std::move(scan);
    f.quantity = std::move(quantity);
    f.verdict = "note";
    findings.push_back(std::move(f));
    return findings.back();
}

namespace {

std::string quoted(const std::string& s) {
    std::string o = "\"";
    for (const char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) o += fmt("\\u%04x", c);
                else o += c;
        }
    }
    return o + "\"";
}

std::string number(double v) { return std::isfinite(v) ? fmt("%.12g", v) : std::string("null"); }

}  // namespace

std::string Report::json() const {
    std::ostringstream o;
    o << "{\"assessment_policy\": \"per_sample_target_no_uncertainty_discount\", \"position_target_px\": "
      << number(position_target_px) << ", \"findings\": [\n";
    for (std::size_t i = 0; i < findings.size(); ++i) {
        const Finding& f = findings[i];
        o << "  {\"requirement\": " << quoted(f.requirement) << ", \"scan\": " << quoted(f.scan)
          << ", \"quantity\": " << quoted(f.quantity) << ",\n   \"stated\": " << quoted(f.stated)
          << ",\n   \"found\": " << quoted(f.found) << ",\n   \"verdict\": " << quoted(f.verdict)
          << ", \"requires_follow_up\": " << (f.verdict == "verified" ? "false" : "true")
          << ", \"note\": " << quoted(f.note);
        if (f.residuals) {
            const auto& r = *f.residuals;
            o << ",\n   \"residuals\": {\"unit\": " << quoted(r.unit) << ", \"target\": " << number(r.target)
              << ", \"samples\": " << r.samples << ", \"invalid\": " << r.invalid
              << ", \"nonzero\": " << r.nonzero << ", \"above_target\": " << r.above_target
              << ", \"mean\": " << number(r.mean) << ", \"min\": " << number(r.min) << ", \"max\": " << number(r.max)
              << ", \"rms\": " << number(r.rms) << ", \"max_abs\": " << number(r.max_abs)
              << ", \"block_sem\": " << number(r.block_sem) << "}";
        }
        o << ",\n   \"numbers\": {";
        for (std::size_t k = 0; k < f.numbers.size(); ++k)
            o << (k ? ", " : "") << quoted(f.numbers[k].first) << ": " << number(f.numbers[k].second);
        o << "}}" << (i + 1 < findings.size() ? "," : "") << "\n";
    }
    o << "]}\n";
    return o.str();
}

std::string Report::text() const {
    std::ostringstream o;
    for (const Finding& f : findings) {
        o << "[" << f.verdict << "] " << f.requirement << " " << f.scan << " - " << f.quantity << "\n";
        if (!f.stated.empty()) o << "    stated: " << f.stated << "\n";
        if (!f.found.empty()) o << "    found:  " << f.found << "\n";
        if (!f.note.empty()) o << "    note:   " << f.note << "\n";
        if (f.residuals) {
            const auto& r = *f.residuals;
            o << "    residuals: " << r.samples << " finite samples, " << r.invalid << " invalid; " << r.nonzero
              << " nonzero, " << r.above_target << " above target " << number(r.target) << " " << r.unit
              << "; min " << number(r.min) << ", max " << number(r.max) << ", rms " << number(r.rms)
              << ", max absolute " << number(r.max_abs) << "; descriptive block SEM " << number(r.block_sem) << "\n"
              << "    follow-up: account for residuals and measurement uncertainty; validate coverage and remaining error after correction. "
                 "Below target does not mean accounted for; block SEM excludes systematic error and correlation between blocks.\n";
        }
    }
    return o.str();
}

Tolerances tolerances(const Record& rec, double eps, double radius_px) {
    if (!std::isfinite(eps) || eps <= 0) throw Error("position target must be finite and positive");
    Tolerances t;
    t.eps = eps;
    const auto& size = rec.technique.numbers("detector/" + rec.detector + "/size");
    if (size.size() != 2) throw Error("the record does not state the detector's size");
    // the largest radius: a second scan with its axis at the edge of the field spans the detector's full width
    t.R = radius_px > 0 ? radius_px : size[0];
    t.H = size[1];
    t.step_deg = rec.technique.number("scan/scan_range") / rec.technique.number("scan/tomo_n");
    return t;
}

}  // namespace helirec::calibration
