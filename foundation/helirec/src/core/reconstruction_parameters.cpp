// helirec/reconstruction.hpp: the reconstruction's parameters (defaults, mode legacy_reference, --set, filter chains,
// the scintillator model, the backprojector).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>

#include "helirec/reconstruction.hpp"
#include "helirec/sino.hpp"
#include "io/json.hpp"

namespace helirec {

namespace {

Json num(double v) { return Json::of(v); }
Json whole(long long v) { return Json::whole_number(v); }
Json text(const std::string& v) { return Json::of(v); }

const Json* member(const Json& j, const std::string& key) { return j.find(key); }

// set `key` of an object, in place if it is there, else at its end
void put(Json& j, const std::string& key, Json v) {
    for (auto& [k, x] : j.object)
        if (k == key) {
            x = std::move(v);
            return;
        }
    j.object.emplace_back(key, std::move(v));
}

void drop(Json& j, const std::string& key) {
    j.object.erase(std::remove_if(j.object.begin(), j.object.end(), [&](const auto& kv) { return kv.first == key; }), j.object.end());
}

Json reference_chain() {
    return Json::list()
        .add(Json::dict().set("type", text("paganin")).set("delta_beta", num(1000.0)))
        .add(Json::dict().set("type", text("unsharp")).set("sigma", num(1.2)).set("coeff", num(4.0)));
}

// the parameters every mode shares the values of, in order (mode, filters and the calibration keys aside)
Json shared(const char* backprojector, const char* x_interp, const char* z_interp, const char* angle_interp, const char* sharpness_weighting) {
    return Json::dict()
        .set("margin", whole(-1))
        .set("n_grid", whole(0))
        .set("angular_tolerance_steps", num(5.0))
        .set("luminosity_repartition", num(100.0))
        .set("mend_threshold", num(1e-4))
        .set("log_clip_min", num(1e-6))
        .set("log_clip_max", num(10.0))
        .set("block_angles", whole(500))
        .set("backprojector", text(backprojector))
        .set("grid_size", whole(0))
        .set("ghbp_legs", whole(8))
        .set("ghbp_reduction_steps", whole(2))
        .set("clip_outer_circle", Json::of_bool(true))
        .set("axis_source_meters", num(1e9))
        .set("sld", text(""))
        .set("x_interp", text(x_interp))
        .set("z_interp", text(z_interp))
        .set("angle_interp", text(angle_interp))
        .set("exposure_arc", num(0.0))
        .set("angle_subset", text(""))
        .set("only_scan", text(""))
        .set("sharpness_weighting", text(sharpness_weighting))
        .set("sharp_f", num(0.35))
        .set("sharp_band", text("0.15:0.30"));
}

Json defaults_json() {
    Json d = Json::dict().set("mode", text("corrected")).set("filters", Json::list());
    for (auto& kv : shared("fbp", "sinc", "lanczos8", "views", "auto").object) d.object.push_back(kv);
    return d.set("geometry", text("")).set("row_time_us", text(""));
}

Json legacy_json() {
    Json d = Json::dict().set("filters", reference_chain());
    for (auto& kv : shared("ghbp", "linear", "linear", "linear", "").object) d.object.push_back(kv);
    return d;
}

const std::vector<std::string> kRemoved = {"z_translation_scale", "companion_registration", "stationary_registration", "axis_refinement", "rolling_shutter"};

std::string strip(const std::string& s) {
    const std::size_t a = s.find_first_not_of(" \t\n\r");
    if (a == std::string::npos) return std::string();
    return s.substr(a, s.find_last_not_of(" \t\n\r") - a + 1);
}

// Python float(): the whole text a real number (also inf and nan), else nothing
bool real_of(const std::string& t, double& v) {
    const std::string s = strip(t);
    if (s.empty() || s.find_first_of("xX") != std::string::npos) return false;
    char* end = nullptr;
    v = std::strtod(s.c_str(), &end);
    return end == s.c_str() + s.size();
}

Json value_of(const std::string& t) {
    double v = 0;
    if (!real_of(t, v)) return text(t);
    static const std::regex integer(R"([+-]?\d+)");
    if (std::regex_match(strip(t), integer)) return whole(static_cast<long long>(v));
    return num(v);
}

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

Json chain_of(const std::string& text_in) {
    const std::string t = strip(text_in);
    if (t.empty() || t == "none" || t == "[]") return Json::list();
    if (t[0] == '[') return Json::parse(t);
    Json chain = Json::list();
    static const std::regex item_re(R"(\s*([\w-]+)\s*(?:\((.*)\))?\s*)");
    std::size_t i = 0;
    while (i <= t.size()) {
        std::size_t j = t.find('+', i);
        if (j == std::string::npos) j = t.size();
        const std::string item = t.substr(i, j - i);
        std::smatch m;
        if (!std::regex_match(item, m, item_re)) throw Error("cannot parse filter '" + item + "'");
        Json f = Json::dict().set("type", text(m[1].str()));
        const std::string args = m[2].matched ? m[2].str() : std::string();
        std::size_t a = 0;
        while (a <= args.size()) {
            std::size_t b = args.find(',', a);
            if (b == std::string::npos) b = args.size();
            const std::string kv = args.substr(a, b - a);
            if (!kv.empty()) {
                const std::size_t eq = kv.find('=');
                if (eq == std::string::npos) throw Error("cannot parse filter parameter '" + kv + "' (expected name=value)");
                put(f, strip(kv.substr(0, eq)), value_of(strip(kv.substr(eq + 1))));
            }
            a = b + 1;
        }
        chain.add(f);
        i = j + 1;
    }
    return chain;
}

// a chain given as text or as a list
Json as_chain(const Json& v) {
    if (v.is(Json::String)) return chain_of(v.string);
    if (v.is(Json::Array)) return v;
    if (v.is(Json::Null)) return Json::list();
    throw Error("filters: a chain is text or a list of filters");
}

std::string short_number(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%g", v);
    return b;
}

std::string chain_text(const Json& chain) {
    std::string out;
    for (const Json& f : chain.array) {
        if (!out.empty()) out += "+";
        const Json* type = member(f, "type");
        out += type ? type->string : "?";
        std::string args;
        for (const auto& [k, v] : f.object) {
            if (k == "type") continue;
            args += (args.empty() ? "" : ",") + k + "=" + (v.is(Json::Number) ? short_number(v.number) : v.is(Json::String) ? v.string : v.text());
        }
        if (!args.empty()) out += "(" + args + ")";
    }
    return out;
}

// the filter that "TYPE" or "TYPE[k]" names
Json& locate(Json& chain, const std::string& ref) {
    static const std::regex re(R"(([\w-]+)(?:\[(\d+)\])?)");
    std::smatch m;
    if (!std::regex_match(ref, m, re)) throw Error("bad filter reference '" + ref + "'");
    const std::string type = m[1].str();
    const int k = m[2].matched ? std::stoi(m[2].str()) : 0;
    int seen = 0;
    for (Json& f : chain.array) {
        const Json* t = member(f, "type");
        if (t && t->is(Json::String) && t->string == type && seen++ == k) return f;
    }
    throw Error("no filter '" + ref + "' in chain " + chain_text(chain));
}

Json override_chain(Json chain, const std::vector<std::pair<std::string, Json>>& changes) {
    for (const auto& [key, v] : changes) {
        if (key == "filters") {
            chain = as_chain(v);
            continue;
        }
        const std::size_t dot = key.rfind('.');
        if (dot == std::string::npos || dot == 0) throw Error("'" + key + "' is not TYPE.param");
        put(locate(chain, key.substr(0, dot)), key.substr(dot + 1), v);
    }
    return chain;
}

// Python's float() of a value: numbers, booleans, and text that reads as one
bool number_of(const Json& v, double& out) {
    if (v.is(Json::Number)) return out = v.number, true;
    if (v.is(Json::Bool)) return out = v.boolean ? 1.0 : 0.0, true;
    if (v.is(Json::String)) return real_of(v.string, out);
    return false;
}

bool truth(const Json& v) {
    switch (v.type) {
        case Json::Bool: return v.boolean;
        case Json::Number: return v.number != 0;
        case Json::String: return !v.string.empty();
        case Json::Array: return !v.array.empty();
        case Json::Object: return !v.object.empty();
        default: return false;
    }
}

bool json_equal(const Json& a, const Json& b) {
    if (a.type != b.type) return false;
    switch (a.type) {
        case Json::Null: return true;
        case Json::Bool: return a.boolean == b.boolean;
        case Json::Number: return a.number == b.number;
        case Json::String: return a.string == b.string;
        case Json::Array:
            if (a.array.size() != b.array.size()) return false;
            for (std::size_t i = 0; i < a.array.size(); ++i)
                if (!json_equal(a.array[i], b.array[i])) return false;
            return true;
        case Json::Object:
            if (a.object.size() != b.object.size()) return false;
            for (const auto& [k, v] : a.object) {
                const Json* w = b.find(k);
                if (!w || !json_equal(v, *w)) return false;
            }
            return true;
    }
    return false;
}

// a chain with its numbers (and the text that reads as one) as numbers: 1000 == 1000.0 == "1000"
Json chain_norm(const Json& chain) {
    Json out = Json::list();
    for (const Json& f : as_chain(chain).array) {
        Json g = Json::dict();
        for (const auto& [k, v] : f.object) {
            double x = 0;
            g.set(k, k != "type" && number_of(v, x) ? num(x) : v);
        }
        out.add(g);
    }
    return out;
}

// recon.py _same: is the value given the reference's value
bool same(const std::string& key, const Json& a, const Json& b) {
    if (key == "filters") return json_equal(chain_norm(a), chain_norm(b));
    if (a.is(Json::Bool) || b.is(Json::Bool)) return truth(a) == truth(b);
    if (b.is(Json::Number)) {
        double x = 0;
        return number_of(a, x) && x == b.number;
    }
    return json_equal(a, b);
}

std::vector<double> terms_of(const Json& v, const std::string& name, bool positive) {
    std::vector<double> out;
    const auto add = [&](double x) {
        if (!std::isfinite(x) || (positive ? x <= 0 : x < 0))
            throw Error("sld: " + name + " must contain finite " + (positive ? "positive" : "non-negative") + " values");
        out.push_back(x);
    };
    if (v.is(Json::String)) {
        std::string s = v.string;
        std::replace(s.begin(), s.end(), ';', ':');
        std::size_t i = 0;
        while (i <= s.size()) {
            std::size_t j = s.find(':', i);
            if (j == std::string::npos) j = s.size();
            double x = 0;
            if (!real_of(s.substr(i, j - i), x)) throw Error("sld: " + name + " must contain finite values");
            add(x);
            i = j + 1;
        }
    } else if (v.is(Json::Array)) {
        for (const Json& e : v.array) {
            double x = 0;
            if (!number_of(e, x)) throw Error("sld: " + name + " must contain finite values");
            add(x);
        }
    } else {
        double x = 0;
        if (!number_of(v, x)) throw Error("sld: " + name + " must contain finite values");
        add(x);
    }
    if (out.empty()) throw Error("sld: " + name + " must contain finite values");
    return out;
}

int positive_int(const Json& v, const std::string& name, int minimum) {
    double x = 0;
    if (!number_of(v, x) || !std::isfinite(x) || x != std::floor(x) || x < minimum)
        throw Error("sld: " + name + " must be an integer >= " + std::to_string(minimum));
    return static_cast<int>(x);
}

std::string read_text(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw Error("cannot read " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string expand_user(const std::string& p) {
    if (p.size() >= 1 && p[0] == '~' && (p.size() == 1 || p[1] == '/')) {
        const char* home = std::getenv("HOME");
        if (home) return std::string(home) + p.substr(1);
    }
    return p;
}

}  // namespace

// ---------------------------------------------------------------- public

std::string default_parameters() { return defaults_json().text(); }
std::string legacy_reference_parameters() { return legacy_json().text(); }

const std::vector<std::string>& calibration_keys() {
    static const std::vector<std::string> keys = {"geometry", "row_time_us"};
    return keys;
}

std::array<double, 2> display_window(const std::string& mode) {
    const double k = mode == "legacy_reference" ? 1.0 : 2.0;
    return {k * -0.03, k * 0.145};
}

std::string mode_conventions(const std::string& mode) {
    if (mode == "legacy_reference")
        return Json::dict()
            .set("output_units", text("reference_units (attenuation / 2)"))
            .set("ramp_angular_weight", text("pi/n_angles"))
            .set("rebalance", text("nabuxx (endpoint quirks)"))
            .set("ramp_padding_reflection", text("lround(2*rot_center)"))
            .set("mending_domain", text("main-scan first/last column as left/right widths"))
            .set("paganin_kernel", text("nabuxx pag_version 2 (truncated per chunk, DC not renormalised)"))
            .set("column_offset", text("two-column linear splat"))
            .set("luminosity_window", text("clipped to the chunk"))
            .set("row_position", text("linear"))
            .set("angular_gridding", text("two-bin splat"))
            .set("exposure_arc", text("none"))
            .text();
    if (mode != "corrected") throw Error("mode must be one of ('corrected', 'legacy_reference'), not '" + mode + "'");
    return Json::dict()
        .set("output_units", text("attenuation_cm-1"))
        .set("ramp_angular_weight", text("2pi/n_angles"))
        .set("rebalance", text("periodic_opposite_view"))
        .set("ramp_padding_reflection", text("2*rot_center (interpolated)"))
        .set("mending_domain", text("merged columns covered by the scans' weight maps"))
        .set("paganin_kernel", text("fixed, rows [-m, m], unit DC"))
        .set("column_offset", text("x_interp (sinc by default: rows shifted exactly; lanczos3 for measured-only gridding); linear splat next to unmeasured columns"))
        .set("luminosity_window", text("full 25 rows (12 context rows read either side of a chunk)"))
        .set("row_position", text("z_interp kernel (lanczos8 by default); linear next to unmeasured rows"))
        .set("angular_gridding", text("angle_interp (views by default: a sinogram row per projection, each at its measured angle; nearest: one bin of one turn, the "
                                      "bin's mean measured angle; ghbp and linear: two-bin splat)"))
        .set("exposure_arc", text("as recorded by default; exposure_arc=auto deconvolves the scan's recorded arc along the angle (disabled with angle_interp=views)"))
        .text();
}

std::string parse_filter_chain(const std::string& t) { return chain_of(t).text(); }

std::string override_filter_chain(const std::string& chain, const std::string& changes) {
    const Json c = Json::parse(changes);
    std::vector<std::pair<std::string, Json>> list(c.object.begin(), c.object.end());
    return override_chain(as_chain(Json::parse(chain)), list).text();
}

std::string filter_chain_text(const std::string& chain) { return chain_text(as_chain(Json::parse(chain))); }

bool is_filter_key(const std::string& key) { return key == "filters" || key.find('.') != std::string::npos; }

std::string parse_value(const std::string& t) { return value_of(t).text(); }

std::string parse_settings(const std::vector<std::string>& items) {
    const Json defaults = defaults_json();
    Json out = Json::dict();
    for (const std::string& it : items) {
        const std::size_t eq = it.find('=');
        if (eq == std::string::npos) throw Error("bad --set '" + it + "', expected KEY=VALUE");
        const std::string k = strip(it.substr(0, eq)), v = strip(it.substr(eq + 1));
        if (is_filter_key(k)) {
            put(out, k, k == "filters" ? chain_of(v) : value_of(v));
            continue;
        }
        const Json* d = member(defaults, k);
        if (!d) {
            std::string known;
            for (const auto& kv : defaults.object) known += (known.empty() ? "" : ", ") + kv.first;
            throw Error("unknown parameter '" + k + "'; known: " + known);
        }
        double x = 0;
        if (d->is(Json::Bool)) {
            const std::string l = lower(v);
            put(out, k, Json::of_bool(l == "1" || l == "true" || l == "yes" || l == "on"));
        } else if (d->is(Json::Number) && real_of(v, x)) {
            if (d->whole && x != std::floor(x)) throw Error("--set " + k + ": a whole number, not '" + v + "'");
            put(out, k, d->whole ? whole(static_cast<long long>(x)) : num(x));
        } else {
            put(out, k, text(v));  // (exposure_arc=auto and the text parameters)
        }
    }
    return out.text();
}

std::string update_parameters(const std::string& params_text, const std::string& changes_text) {
    Json params = Json::parse(params_text);
    Json kw = Json::parse(changes_text);
    if (!params.is(Json::Object) || !kw.is(Json::Object)) throw Error("parameters are a JSON object");
    if (const Json* smear = member(kw, "detector_smear")) {  // (the name of sharpness_weighting until 2026-11-11)
        const Json v = *smear;
        drop(kw, "detector_smear");
        if (!member(kw, "sharpness_weighting")) put(kw, "sharpness_weighting", v);
    }
    std::vector<std::string> gone;
    for (const auto& kv : kw.object)
        if (std::find(kRemoved.begin(), kRemoved.end(), kv.first) != kRemoved.end()) gone.push_back(kv.first);
    if (!gone.empty()) {
        std::sort(gone.begin(), gone.end());
        std::string list;
        for (const auto& g : gone) list += (list.empty() ? "" : ", ") + g;
        throw Error("[" + list + "]: removed with the calibration they belonged to; the geometry calibration (the parameter geometry, helirec-calibration "
                    "geometry) holds what they stood for");
    }
    const Json defaults = defaults_json(), legacy = legacy_json();
    Json regular = Json::dict();
    std::vector<std::pair<std::string, Json>> filter_changes;
    std::vector<std::string> unknown, filter_keys;
    for (const auto& [k, v] : kw.object) {
        if (!is_filter_key(k) || k == "filters") {
            regular.set(k, v);
            if (!member(defaults, k)) unknown.push_back(k);
        } else {
            filter_changes.emplace_back(k, v);
            filter_keys.push_back(k);
        }
    }
    if (!unknown.empty()) {
        std::sort(unknown.begin(), unknown.end());
        std::string list, known;
        for (const auto& u : unknown) list += (list.empty() ? "" : ", ") + u;
        for (const auto& kv : defaults.object) known += (known.empty() ? "" : ", ") + kv.first;
        throw Error("unknown parameters [" + list + "]; known: " + known + " and TYPE.param filter keys");
    }
    const Json* mode_given = member(regular, "mode");
    const Json* mode_now = member(params, "mode");
    const std::string mode = mode_given ? mode_given->string : mode_now ? mode_now->string : "corrected";
    if (mode != "corrected" && mode != "legacy_reference") throw Error("mode must be one of ('corrected', 'legacy_reference'), not '" + mode + "'");
    Json named = Json::dict();  // (the calibration in use: kept by every mode)
    for (const std::string& k : calibration_keys()) {
        const Json* v = member(regular, k);
        if (!v) v = member(params, k);
        named.set(k, v ? *v : *member(defaults, k));
    }
    if (mode == "legacy_reference") {
        std::vector<std::string> bad;
        for (const auto& [k, v] : regular.object) {
            if (k == "mode" || std::find(calibration_keys().begin(), calibration_keys().end(), k) != calibration_keys().end()) continue;
            if (!same(k, v, *member(legacy, k))) bad.push_back(k);
        }
        std::sort(bad.begin(), bad.end());
        std::sort(filter_keys.begin(), filter_keys.end());
        bad.insert(bad.end(), filter_keys.begin(), filter_keys.end());
        if (!bad.empty()) {
            std::string list, pinned;
            for (const auto& b : bad) {
                list += (list.empty() ? "" : ", ") + b;
                if (const Json* l = member(legacy, b)) pinned += (pinned.empty() ? "" : ", ") + b + "=" + (l->is(Json::String) ? l->string : l->text());
            }
            throw Error("mode legacy_reference pins [" + list + "] to the reference values (" + pinned + "); vary them in mode 'corrected' instead");
        }
        Json out = Json::dict().set("mode", text(mode));
        for (const auto& kv : legacy.object) out.object.push_back(kv);
        for (const auto& kv : named.object) out.object.push_back(kv);
        return out.text();
    }
    if (mode_now && mode_now->string == "legacy_reference") {  // leaving the legacy mode: start again from the defaults
        params = defaults;
        for (const auto& [k, v] : named.object) put(params, k, v);
    }
    for (const auto& [k, v] : regular.object) put(params, k, k == "filters" ? as_chain(v) : v);
    if (!filter_changes.empty()) {
        const Json* f = member(params, "filters");
        put(params, "filters", override_chain(f ? as_chain(*f) : Json::list(), filter_changes));
    }
    return params.text();
}

SldParams sld_params(const std::string& spec_text, std::string* resolved) {
    SldParams p;
    if (resolved) *resolved = "{}";
    // the spec: a JSON object, "key=value,...", or a path
    Json spec;
    const std::string t = strip(spec_text);
    if (t.empty() || t == "{}") return p;
    if (t[0] == '{') {
        spec = Json::parse(t);
    } else if (t.find('=') == std::string::npos) {
        spec = Json::dict().set("json", text(t));
    } else {
        spec = Json::dict();
        std::size_t i = 0;
        while (i <= t.size()) {
            std::size_t j = t.find(',', i);
            if (j == std::string::npos) j = t.size();
            const std::string kv = t.substr(i, j - i);
            if (!strip(kv).empty()) {
                const std::size_t eq = kv.find('=');
                if (eq == std::string::npos) throw Error("sld: expected key=value entries");
                spec.set(strip(kv.substr(0, eq)), text(strip(kv.substr(eq + 1))));
            }
            i = j + 1;
        }
    }
    bool loaded = false;
    if (const Json* path = member(spec, "json")) {
        const Json payload = Json::parse(read_text(expand_user(path->is(Json::String) ? path->string : path->text())));
        if (!payload.is(Json::Object) || !member(payload, "fraction") || !member(payload, "scale_l"))
            throw Error("sld: Night Rail JSON must contain fraction and scale_l");
        // only the model's fields: the fitter's ROI and source-image metadata are not processing options
        static const std::set<std::string> names = {"fraction", "scale_l", "replica_shift_x", "replica_shift_y", "replica_factor", "save_binning"};
        Json d = Json::dict();
        for (const auto& [k, v] : payload.object)
            if (names.count(k)) d.set(k, v);
        for (const auto& [k, v] : spec.object)
            if (k != "json") put(d, k, v);
        spec = d;
        loaded = true;
    }
    static const std::set<std::string> known = {"fraction", "length", "scale_l", "bin", "stride", "profile", "border_x", "border_y", "outside",
                                                "replica_shift_x", "replica_shift_y", "replica_factor", "save_binning"};
    bool unknown = !member(spec, "fraction");
    for (const auto& kv : spec.object) unknown = unknown || !known.count(kv.first);
    if (unknown) {
        std::vector<std::string> keys;
        for (const auto& kv : spec.object) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        std::string list;
        for (const auto& k : keys) list += (list.empty() ? "" : ", ") + k;
        throw Error("sld: expected fraction and known model options, got [" + list + "]");
    }
    if (member(spec, "length") && member(spec, "scale_l")) throw Error("sld: supply length or scale_l, not both");
    const std::vector<double> fractions = terms_of(*member(spec, "fraction"), "fraction", false);
    const Json* length = member(spec, "length");
    if (!length) length = member(spec, "scale_l");
    const std::vector<double> lengths = length ? terms_of(*length, "length", true) : std::vector<double>{100.0};
    if (fractions.size() != lengths.size()) throw Error("sld: fraction and length lists must have equal lengths");
    const bool inferred = loaded || member(spec, "scale_l") || fractions.size() > 1 || member(spec, "border_x") || member(spec, "border_y");
    const Json* profile_v = member(spec, "profile");
    const std::string profile = profile_v ? (profile_v->is(Json::String) ? profile_v->string : profile_v->text()) : inferred ? "bm18" : "legacy";
    if (profile != "legacy" && profile != "bm18") throw Error("sld: profile must be legacy or bm18");
    if (positive_int(member(spec, "save_binning") ? *member(spec, "save_binning") : whole(1), "save_binning", 1) != 1)
        throw Error("sld: calibration was fitted on binned images; convert scale_l to full detector pixels first");
    for (const char* name : {"replica_shift_x", "replica_shift_y", "replica_factor"}) {
        double x = 0;
        if (const Json* v = member(spec, name); v && (!number_of(*v, x) || !std::isfinite(x) || x != 0))
            throw Error("sld: shifted-replica calibrations are not supported");
    }
    const bool bm18 = profile == "bm18";
    const Json* outside_v = member(spec, "outside");
    const std::string outside = outside_v ? (outside_v->is(Json::String) ? outside_v->string : outside_v->text()) : "dark";
    if (outside != "dark" && outside != "lit") throw Error("sld: outside must be dark or lit");
    if (bm18 && outside_v) throw Error("sld: outside belongs to the binned profile; the BM18 profile takes border_x and border_y");
    const int binning = positive_int(member(spec, "bin") ? *member(spec, "bin") : whole(bm18 ? 1 : 8), "bin", 1);
    const int stride = positive_int(member(spec, "stride") ? *member(spec, "stride") : whole(bm18 ? 1 : 10), "stride", 1);
    const auto numbers = [](const std::vector<double>& v) {
        Json j = Json::list();
        for (double x : v) j.add(num(x));
        return j;
    };
    if (bm18) {
        if (binning != 1) throw Error("sld: BM18 full-resolution correction requires bin=1; stride may be increased explicitly");
        p.bm18 = true;
        p.fractions = fractions;
        p.lengths = lengths;
        p.bin = 1;
        p.stride = stride;
        p.border_x = positive_int(member(spec, "border_x") ? *member(spec, "border_x") : whole(100), "border_x", 0);
        p.border_y = positive_int(member(spec, "border_y") ? *member(spec, "border_y") : whole(100), "border_y", 0);
        if (resolved)
            *resolved = Json::dict()
                            .set("sld_bm18", Json::of_bool(true))
                            .set("sld_fractions", numbers(fractions))
                            .set("sld_lengths", numbers(lengths))
                            .set("sld_bin", whole(1))
                            .set("sld_stride", whole(stride))
                            .set("sld_border_x", whole(p.border_x))
                            .set("sld_border_y", whole(p.border_y))
                            .text();
        return p;
    }
    if (inferred) throw Error("sld: calibration JSON, multiple tails and borders require profile=bm18");
    if (binning > 64 || fractions[0] >= 1 || lengths[0] < 1) throw Error("sld: legacy profile requires fraction < 1, length >= 1 and bin <= 64");
    if (!fractions[0]) return p;
    p.fraction = fractions[0];
    p.length = lengths[0];
    p.bin = binning;
    p.stride = stride;
    p.lit_outside = outside == "lit";
    if (resolved) {
        Json r = Json::dict().set("sld_fraction", num(p.fraction)).set("sld_length", num(p.length)).set("sld_bin", whole(binning)).set("sld_stride", whole(stride));
        if (p.lit_outside) r.set("sld_lit_outside", Json::of_bool(true));
        *resolved = r.text();
    }
    return p;
}

std::pair<int, int> parse_angle_subset(const std::string& spec) {
    const std::string t = strip(spec);
    if (t.empty()) return {0, 1};
    const std::size_t slash = t.find('/');
    double i = 0, n = 0;
    if (slash == std::string::npos || !real_of(t.substr(0, slash), i) || !real_of(t.substr(slash + 1), n) || i != std::floor(i) || n != std::floor(n) ||
        t.find('.') != std::string::npos)
        throw Error("angle_subset must be 'I/N' (subset I of N), not '" + spec + "'");
    if (!(0 <= i && i < n)) throw Error("angle_subset '" + spec + "': I must be in [0, N)");
    return {static_cast<int>(i), static_cast<int>(n)};
}

std::string backprojector_config(const std::string& spec_text) {
    const std::string t = strip(spec_text);
    Json s;
    if (!t.empty() && t[0] == '{') s = Json::parse(t);
    else {
        const Json chain = chain_of(t);
        if (chain.array.empty()) throw Error("no backprojector given");
        s = chain.array[0];
    }
    const Json* type_v = member(s, "type");
    if (!type_v || !type_v->is(Json::String)) throw Error("a backprojector spec names its type");
    const std::string type = type_v->string;
    drop(s, "type");
    // (as Python's str() of the value)
    const auto value_text = [](const Json& v) -> std::string {
        if (v.is(Json::String)) return v.string;
        if (v.is(Json::Bool)) return v.boolean ? "True" : "False";
        if (v.is(Json::Number) && !v.whole) return python_repr(v.number);
        return v.text();
    };
    Json out = Json::dict();
    if (type == "ghbp" || type == "fbp") {
        out.set("backprojector", text(type));
        if (type == "fbp") {
            const Json* interp = member(s, "interp");
            const Json* over = member(s, "oversample");
            double o = 2;
            if (over && !number_of(*over, o)) throw Error("backprojector fbp: oversample is a number");
            out.set("bp_interp", text(interp ? value_text(*interp) : "bspline3")).set("bp_oversample", whole(static_cast<long long>(o)));
            drop(s, "interp");
            drop(s, "oversample");
        }
        for (const char* k : {"filter_name", "filter_par"})  // the ramp filter (checked where it is made)
            if (const Json* v = member(s, k)) {
                out.set(std::string("ramp_") + k, text(value_text(*v)));
                drop(s, k);
            }
    } else if (type == "sirt" || type == "mlem" || type == "gridrec") {
        Json params = Json::dict();
        for (const auto& [k, v] : s.object) params.set(k, text(value_text(v)));
        out.set("backprojector", text(type)).set("tomo_params", params);
        s.object.clear();
    } else if (type == "python") {
        out.set("backprojector", text("external")).set("external_bp", s);
        s.object.clear();
    } else {
        throw Error("unknown backprojector '" + type + "' (ghbp, fbp, sirt, mlem, gridrec, python)");
    }
    if (!s.object.empty()) {
        std::vector<std::string> keys;
        for (const auto& kv : s.object) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        std::string list;
        for (const auto& k : keys) list += (list.empty() ? "" : ", ") + k;
        throw Error("backprojector " + type + ": unknown parameters [" + list + "]");
    }
    return out.text();
}

void apply_backprojector(const std::string& spec, ReconConfig& cfg, const ExternalBackprojectorFactory& external) {
    const Json c = Json::parse(backprojector_config(spec));
    cfg.backprojector = member(c, "backprojector")->string;
    if (const Json* v = member(c, "bp_interp")) cfg.bp_interp = v->string;
    if (const Json* v = member(c, "bp_oversample")) cfg.bp_oversample = static_cast<int>(v->number);
    if (const Json* v = member(c, "ramp_filter_name")) cfg.ramp_filter_name = v->string;
    if (const Json* v = member(c, "ramp_filter_par")) cfg.ramp_filter_par = ramp_filter_par(v->string);
    if (const Json* v = member(c, "tomo_params"))
        for (const auto& [k, x] : v->object) cfg.tomo_params.values[k] = x.string;
    if (const Json* v = member(c, "external_bp")) {
        if (!external) throw Error("backprojector python(...) is a function of the Python session (helirec.backprojectors): not available here");
        FilterParams fp;
        for (const auto& [k, x] : v->object) fp.values[k] = x.is(Json::String) ? x.string : x.whole ? x.text() : python_repr(x.number);
        cfg.external = external(fp);
    }
}

}  // namespace helirec
