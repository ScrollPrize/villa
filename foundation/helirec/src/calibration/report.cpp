#include <cmath>
#include <sstream>

#include "calibration/util.hpp"
#include "helirec/calibration/verify.hpp"

namespace helirec::calibration {

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
    o << "{\"findings\": [\n";
    for (std::size_t i = 0; i < findings.size(); ++i) {
        const Finding& f = findings[i];
        o << "  {\"requirement\": " << quoted(f.requirement) << ", \"scan\": " << quoted(f.scan)
          << ", \"quantity\": " << quoted(f.quantity) << ",\n   \"stated\": " << quoted(f.stated)
          << ",\n   \"found\": " << quoted(f.found) << ",\n   \"verdict\": " << quoted(f.verdict)
          << ", \"note\": " << quoted(f.note) << ",\n   \"numbers\": {";
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
    }
    return o.str();
}

Tolerances tolerances(const Record& rec, double eps) {
    Tolerances t;
    t.eps = eps;
    const auto& size = rec.technique.numbers("detector/" + rec.detector + "/size");
    if (size.size() != 2) throw Error("the record does not state the detector's size");
    // the largest radius: a second scan with its axis at the edge of the field spans the detector's full width
    t.R = size[0];
    t.H = size[1];
    t.step_deg = rec.technique.number("scan/scan_range") / rec.technique.number("scan/tomo_n");
    return t;
}

}  // namespace helirec::calibration
