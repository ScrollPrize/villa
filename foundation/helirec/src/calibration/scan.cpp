// A BLISS tomo scan as the reconstruction reads it (helirec/calibration/scan.hpp), derived as nxtomomill h52nx and
// nx2etf derive it. The arithmetic follows numpy's where the reference uses it (numpy.linspace in long double for
// time stamps, numpy.interp, numpy.unwrap, numpy.median), so that the values agree bit for bit.
#include "helirec/calibration/scan.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <regex>
#include <set>

#include "calibration/h5_read.hpp"
#include "helirec/common.hpp"

namespace helirec::calibration {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ------------------------------------------------------------------ paths (os.path's, lexically)

std::string normpath(const std::string& p) {
    if (p.empty()) return ".";
    const bool absolute = p[0] == '/';
    int initial = absolute ? 1 : 0;
    if (absolute && p.size() > 1 && p[1] == '/' && (p.size() < 3 || p[2] != '/')) initial = 2;  // POSIX "//"
    std::vector<std::string> parts;
    std::size_t i = 0;
    while (i <= p.size()) {
        std::size_t j = p.find('/', i);
        if (j == std::string::npos) j = p.size();
        const std::string c = p.substr(i, j - i);
        if (c.empty() || c == ".") {
        } else if (c != "..") {
            parts.push_back(c);
        } else if ((!absolute && parts.empty()) || (!parts.empty() && parts.back() == "..")) {
            parts.push_back(c);
        } else if (!parts.empty()) {
            parts.pop_back();
        }
        i = j + 1;
    }
    std::string out(static_cast<std::size_t>(initial), '/');
    for (std::size_t k = 0; k < parts.size(); ++k) out += (k ? "/" : "") + parts[k];
    return out.empty() ? "." : out;
}

std::string join(const std::string& a, const std::string& b) {
    if (!b.empty() && b[0] == '/') return b;
    if (a.empty() || a.back() == '/') return a + b;
    return a + "/" + b;
}

std::string abspath(const std::string& p) {
    if (!p.empty() && p[0] == '/') return normpath(p);
    return normpath(join(std::filesystem::current_path().string(), p));
}

std::string basename(const std::string& p) {
    const std::size_t i = p.find_last_of('/');
    return i == std::string::npos ? p : p.substr(i + 1);
}

std::string dirname(const std::string& p) {
    const std::size_t i = p.find_last_of('/');
    if (i == std::string::npos) return std::string();
    std::string head = p.substr(0, i + 1);
    if (head.find_first_not_of('/') != std::string::npos)
        while (!head.empty() && head.back() == '/') head.pop_back();
    return head;
}

bool is_file(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool is_dir(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::vector<std::string> list_dir(const std::string& d) {
    std::vector<std::string> names;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(d, ec), end; !ec && it != end; it.increment(ec))
        names.push_back(it->path().filename().string());
    if (ec) return {};
    std::sort(names.begin(), names.end());
    return names;
}

// ------------------------------------------------------------------ time stamps

std::int64_t days_from_civil(std::int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

// datetime.fromisoformat(s), in UTC when it carries an offset, as numpy datetime64[ms]: milliseconds since the
// epoch (the microseconds rounded down)
std::int64_t iso_ms(const std::string& s) {
    const auto bad = [&]() { return Error("invalid isoformat string: '" + s + "'"); };
    std::size_t i = 0;
    const auto digits = [&](int n) {
        if (i + n > s.size()) throw bad();
        int v = 0;
        for (int k = 0; k < n; ++k) {
            if (!std::isdigit(static_cast<unsigned char>(s[i]))) throw bad();
            v = v * 10 + (s[i++] - '0');
        }
        return v;
    };
    const auto expect = [&](char c) {
        if (i >= s.size() || s[i] != c) throw bad();
        ++i;
    };
    const int year = digits(4);
    expect('-');
    const int month = digits(2);
    expect('-');
    const int day = digits(2);
    int hour = 0, minute = 0, second = 0;
    std::int64_t micro = 0, offset_us = 0;
    if (i < s.size()) {
        ++i;  // the separator (any one character)
        hour = digits(2);
        if (i < s.size() && s[i] == ':') {
            ++i;
            minute = digits(2);
            if (i < s.size() && s[i] == ':') {
                ++i;
                second = digits(2);
                if (i < s.size() && (s[i] == '.' || s[i] == ',')) {
                    ++i;
                    int n = 0;
                    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
                        if (n < 6) micro = micro * 10 + (s[i] - '0');
                        ++n, ++i;
                    }
                    if (n == 0) throw bad();
                    for (; n < 6; ++n) micro *= 10;
                }
            }
        }
        if (i < s.size()) {
            const char c = s[i++];
            if (c == 'Z' && i == s.size()) {
            } else if (c == '+' || c == '-') {
                const int oh = digits(2);
                int om = 0, os = 0;
                std::int64_t ou = 0;
                if (i < s.size() && s[i] == ':') ++i;
                if (i < s.size()) om = digits(2);
                if (i < s.size() && s[i] == ':') ++i;
                if (i < s.size()) os = digits(2);
                if (i < s.size() && s[i] == '.') {
                    ++i;
                    ou = digits(6);
                }
                if (i != s.size()) throw bad();
                offset_us = (((oh * 60LL + om) * 60 + os) * 1000000 + ou) * (c == '-' ? -1 : 1);
            } else {
                throw bad();
            }
        }
    }
    if (month < 1 || month > 12 || day < 1 || day > 31 || hour > 23 || minute > 59 || second > 59) throw bad();
    const std::int64_t us = ((days_from_civil(year, month, day) * 24 + hour) * 60 + minute) * 60000000LL +
                            second * 1000000LL + micro - offset_us;
    return us >= 0 ? us / 1000 : -((-us + 999) / 1000);
}

// numpy.linspace(float128(t0), float128(t1), n, dtype='<M8[ms]'): the times spread in long double, truncated
std::vector<std::int64_t> linspace_ms(std::int64_t t0, std::int64_t t1, std::size_t n) {
    std::vector<std::int64_t> out(n);
    const long double start = static_cast<long double>(t0), stop = static_cast<long double>(t1);
    const long double delta = stop - start;
    if (n == 1) {
        out[0] = static_cast<std::int64_t>(0.0L * delta + start);
        return out;
    }
    const long double div = static_cast<long double>(n - 1);
    const long double step = delta / div;
    for (std::size_t i = 0; i < n; ++i) {
        long double y = static_cast<long double>(i);
        if (step == 0) {
            y /= div;
            y *= delta;
        } else {
            y *= step;
        }
        y += start;
        out[i] = static_cast<std::int64_t>(y);
    }
    out[n - 1] = static_cast<std::int64_t>(stop);
    return out;
}

// ------------------------------------------------------------------ numpy's arithmetic

// numpy.interp(x, xp, fp) for one x (xp increasing)
double interp(double x, const std::vector<double>& xp, const std::vector<double>& fp) {
    const std::size_t n = xp.size();
    if (n == 1) return fp[0];
    if (std::isnan(x)) return x;
    if (x < xp[0]) return fp[0];
    if (x > xp[n - 1]) return fp[n - 1];
    const std::size_t j = static_cast<std::size_t>(std::upper_bound(xp.begin(), xp.end(), x) - xp.begin()) - 1;
    if (j == n - 1 || xp[j] == x) return fp[j];
    const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    double r = slope * (x - xp[j]) + fp[j];
    if (std::isnan(r)) {
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if (std::isnan(r) && fp[j] == fp[j + 1]) r = fp[j];
    }
    return r;
}

// numpy.remainder for doubles
double py_mod(double a, double b) {
    double mod = std::fmod(a, b);
    if (b == 0) return mod;
    if (mod != 0) {
        if ((b < 0) != (mod < 0)) mod += b;
    } else {
        mod = std::copysign(0.0, b);
    }
    return mod;
}

// numpy.unwrap(p, period=360)
std::vector<double> unwrap_360(const std::vector<double>& p) {
    std::vector<double> up(p);
    double sum = 0;
    for (std::size_t i = 1; i < p.size(); ++i) {
        const double dd = p[i] - p[i - 1];
        double ddmod = py_mod(dd - (-180.0), 360.0) + (-180.0);
        if (ddmod == -180.0 && dd > 0) ddmod = 180.0;
        double correct = ddmod - dd;
        if (std::fabs(dd) < 180.0) correct = 0;
        sum += correct;
        up[i] = p[i] + sum;
    }
    return up;
}

double median(std::vector<double> v) {
    if (v.empty()) return kNaN;
    for (double x : v)
        if (std::isnan(x)) return kNaN;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

// numpy.isclose(a, b, rtol, atol)
bool is_close(double a, double b, double rtol, double atol) {
    if (std::isnan(a) || std::isnan(b)) return false;
    if (std::isinf(a) || std::isinf(b)) return a == b;
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

// ------------------------------------------------------------------ units (pint's factors to SI)

const std::map<std::string, double>& length_to_m() {
    static const std::map<std::string, double> t = {
        {"m", 1.0}, {"meter", 1.0}, {"mm", 1e-3}, {"millimeter", 1e-3}, {"um", 1e-6}, {"\xc2\xb5m", 1e-6},
        {"\xce\xbcm", 1e-6}, {"micrometer", 1e-6}, {"micron", 1e-6}, {"nm", 1e-9}, {"cm", 1e-2}};
    return t;
}

const std::map<std::string, double>& current_to_a() {
    static const std::map<std::string, double> t = {{"a", 1.0}, {"ampere", 1.0}, {"ma", 1e-3}, {"milliampere", 1e-3}};
    return t;
}

double si_factor(const std::string& unit, const std::map<std::string, double>& table, const std::string& fallback) {
    const std::string u = unit.empty() ? fallback : unit;
    auto it = table.find(u);
    if (it == table.end()) {
        std::string lower = u;
        for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        it = table.find(lower);
    }
    if (it == table.end()) throw Error("unknown unit '" + unit + "'");
    return it->second;
}

void to_si(std::vector<double>& v, const std::string& unit, const std::map<std::string, double>& table, const std::string& fallback) {
    const double f = si_factor(unit, table, fallback);
    if (f != 1.0)
        for (double& x : v) x *= f;
}

double to_si(double v, const std::string& unit, const std::map<std::string, double>& table, const std::string& fallback) {
    const double f = si_factor(unit, table, fallback);
    return f != 1.0 ? v * f : v;
}

// ------------------------------------------------------------------ reading the master

bool is_group(hid_t f, const std::string& path) {
    if (!h5::exists(f, path)) return false;
    H5O_info2_t info;
    if (H5Oget_info_by_name3(f, path.c_str(), &info, H5O_INFO_BASIC, H5P_DEFAULT) < 0) return false;
    return info.type == H5O_TYPE_GROUP;
}

std::string strip(const std::string& s) {
    const std::size_t a = s.find_first_not_of(" \t\n\r\f\v");
    if (a == std::string::npos) return std::string();
    return s.substr(a, s.find_last_not_of(" \t\n\r\f\v") - a + 1);
}

std::string units(hid_t f, const std::string& path) { return strip(h5::attribute_text(f, path, "units")); }

std::string text(hid_t f, const std::string& path) {
    const h5::Shape sh = h5::shape(f, path);
    if (sh.type_class == H5T_STRING) return h5::read_string(f, path);
    const auto v = h5::read_doubles(f, path);
    if (v.empty()) throw Error(path + " is empty");
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", v[0]);
    return buf;
}

std::string optional_text(hid_t f, const std::string& path) { return h5::exists(f, path) ? text(f, path) : std::string(); }

const char* const kInitTitles[] = {"tomo:", "zseries", "multitomo", "backandforth", "back_and_forth"};
const char* const kDarkTitles[] = {"dark images", "dark"};
const char* const kFlatTitles[] = {"flat", "reference images", "ref", "refend"};
const char* const kProjTitles[] = {"projections", "ascan rot 0", "ascan diffrz 0 180 1600 0.1"};
const char* const kAlignTitles[] = {"static images", "ascan diffrz 180 0 4 0.1"};

std::string title_role(const std::string& title) {
    const auto starts = [&](const auto& prefixes) {
        for (const char* p : prefixes)
            if (title.rfind(p, 0) == 0) return true;
        return false;
    };
    if (starts(kInitTitles)) return "init";
    if (starts(kDarkTitles)) return "dark";
    if (starts(kFlatTitles)) return "flat";
    if (starts(kProjTitles)) return "proj";
    if (starts(kAlignTitles)) return "align";
    return "";
}

// nxtomomill's ScanTypeFinder: technique/image_key, then technique/scan_category, then the title ("" for none)
std::string entry_role(hid_t f, const std::string& e) {
    const std::string tech = e + "/technique";
    const bool has_tech = h5::exists(f, tech);
    if (has_tech && h5::exists(f, tech + "/image_key")) {
        bool read = false;
        double k = 0;
        try {
            const auto v = h5::read_doubles(f, tech + "/image_key");
            if (!v.empty() && std::isfinite(v[0])) read = true, k = v[0];
        } catch (const Error&) {
        }
        if (read) {
            const long key = static_cast<long>(k);
            if (key == 0) return "proj";
            if (key == 1) return "flat";
            if (key == 2) return "dark";
            if (key == -1) return "align";
            return "";
        }
    }
    if (has_tech && h5::exists(f, tech + "/scan_category")) return "init";
    if (!h5::exists(f, e + "/title")) return "";
    return title_role(text(f, e + "/title"));
}

// the tomo detector: technique/tomoconfig/detector (else technique/detector) of the first entry that names it,
// else the first NXdetector of an entry's instrument with a 3-D image
std::string find_detector(hid_t f, const std::vector<std::string>& entries) {
    for (const std::string& e : entries) {
        for (const char* p : {"/technique/tomoconfig/detector", "/technique/detector"}) {
            const std::string path = e + p;
            if (!h5::exists(f, path)) continue;
            if (h5::is_dataset(f, path)) {
                const auto v = h5::read_strings(f, path);
                if (v.empty()) throw Error(path + " names no detector");
                return v[0];
            }
            if (is_group(f, path)) {
                const auto c = h5::children(f, path);
                if (c.size() == 1) return c[0];
            }
        }
    }
    for (const std::string& e : entries) {
        const std::string inst = e + "/instrument";
        if (!h5::exists(f, inst)) continue;
        for (const std::string& k : h5::children(f, inst)) {
            const std::string g = inst + "/" + k;
            if (!is_group(f, g) || h5::attribute_text(f, g, "NX_class") != "NXdetector") continue;
            const std::string d = h5::exists(f, g + "/image") ? g + "/image" : g + "/data";
            if (h5::is_dataset(f, d) && h5::shape(f, d).dims.size() == 3) return k;
        }
    }
    throw Error("no tomography detector found");
}

// the entry's frames of the detector: instrument/<det>/image (else data), 3-D with at least one frame; "" for none
std::string frame_dataset_path(hid_t f, const std::string& e, const std::string& det) {
    for (const char* k : {"image", "data"}) {
        const std::string path = e + "/instrument/" + det + "/" + k;
        if (!h5::is_dataset(f, path)) continue;
        const h5::Shape sh = h5::shape(f, path);
        if (sh.dims.size() == 3 && sh.dims[0] > 0) return path;
    }
    return std::string();
}

std::string dtype_name(hid_t f, const std::string& path) {
    h5::Id d(H5Dopen2(f, path.c_str(), H5P_DEFAULT), H5Dclose);
    h5::Id t(H5Dget_type(d.id), H5Tclose);
    const std::size_t bits = H5Tget_size(t.id) * 8;
    switch (H5Tget_class(t.id)) {
        case H5T_INTEGER: return (H5Tget_sign(t.id) == H5T_SGN_NONE ? "uint" : "int") + std::to_string(bits);
        case H5T_FLOAT: return "float" + std::to_string(bits);
        default: throw Error(path + ": frames of a type that is neither integer nor float");
    }
}

// Lima's naming of its files, for frames not mapped by a virtual dataset: scanNNNN/<prefix>XXXX.h5, a given number
// of frames in each
FrameList named_frames(hid_t f, const std::string& e, const std::string& det, const std::string& master_dir, std::int64_t n) {
    const std::string acq = e + "/instrument/" + det + "/acq_parameters/";
    const std::string ctrl = e + "/instrument/" + det + "/ctrl_parameters/";
    const std::string prefix = text(f, acq + "saving_prefix");
    const std::int64_t per_file = static_cast<std::int64_t>(h5::read_double(f, ctrl + "saving_frame_per_file"));
    std::string sub = text(f, acq + "saving_directory");
    while (!sub.empty() && sub.back() == '/') sub.pop_back();
    sub = basename(sub);
    if (per_file <= 0) throw Error(e + ": no frames per file");
    FrameList out;
    for (std::int64_t i = 0; i < (n + per_file - 1) / per_file; ++i) {
        char num[32];
        std::snprintf(num, sizeof num, "%04lld", static_cast<long long>(i));
        out.files.push_back(join(join(master_dir, sub), prefix + num + ".h5"));
    }
    out.dataset = "/entry_0000/instrument/" + det + "/data";
    for (std::int64_t k = 0; k < n; ++k) {
        out.file_index.push_back(static_cast<std::int32_t>(k / per_file));
        out.frame_index.push_back(static_cast<std::int32_t>(k % per_file));
    }
    return out;
}

// every frame of an entry's image dataset in the detector's files, through its virtual mapping
FrameList frames_of(hid_t f, const std::string& master, const std::string& master_dir, const std::string& e, const std::string& det,
                    const std::string& path) {
    const h5::FrameDataset fd = h5::frame_dataset(f, path);
    if (!fd.is_virtual) return named_frames(f, e, det, master_dir, fd.frames);
    FrameList out;
    out.file_index.assign(static_cast<std::size_t>(fd.frames), -1);
    out.frame_index.assign(static_cast<std::size_t>(fd.frames), -1);
    std::map<std::string, std::int32_t> pos;
    std::set<std::string> datasets;
    for (const h5::VirtualPiece& p : fd.pieces) {
        std::string fn = p.file_name == "." ? master : join(master_dir, p.file_name);
        fn = normpath(fn);
        datasets.insert(p.dataset);
        if (!p.whole_frames) throw Error("the virtual mapping of " + path + " is not made of whole frames");
        if (p.count != p.source_count) throw Error("irregular virtual mapping in " + path);
        auto it = pos.find(fn);
        if (it == pos.end()) {
            it = pos.emplace(fn, static_cast<std::int32_t>(out.files.size())).first;
            out.files.push_back(fn);
        }
        for (std::int64_t k = 0; k < p.count; ++k) {
            const std::size_t at = static_cast<std::size_t>(p.first + k);
            if (at >= out.file_index.size()) throw Error("the virtual mapping of " + path + " reaches beyond its frames");
            out.file_index[at] = it->second;
            out.frame_index[at] = static_cast<std::int32_t>(p.source_first + k);
        }
    }
    const auto uncovered = std::count(out.file_index.begin(), out.file_index.end(), -1);
    if (uncovered) throw Error("the virtual dataset " + path + " does not cover " + std::to_string(uncovered) + " frames");
    if (datasets.size() != 1) throw Error("the virtual dataset " + path + " takes frames from several datasets");
    out.dataset = *datasets.begin();
    if (out.dataset.empty() || out.dataset[0] != '/') out.dataset = "/" + out.dataset;
    return out;
}

// a motor's value at each of the entry's n frames, and its unit: measurement/<name> (n values) for the first name
// that has one, else the entry's scalar instrument/positioners/<name>
std::pair<std::vector<double>, std::string> per_frame(hid_t f, const std::string& e, const std::vector<std::string>& names, std::int64_t n,
                                                      const std::string& kind) {
    for (const std::string& nm : names) {
        const std::string path = e + "/measurement/" + nm;
        if (!h5::is_dataset(f, path)) continue;
        const h5::Shape sh = h5::shape(f, path);
        if (sh.dims.size() == 1 && static_cast<std::int64_t>(sh.dims[0]) == n) return {h5::read_doubles(f, path), units(f, path)};
    }
    for (const std::string& nm : names) {
        const std::string path = e + "/instrument/positioners/" + nm;
        if (!h5::exists(f, path)) continue;
        const h5::Shape sh = h5::shape(f, path);
        if (sh.type_class == H5T_STRING) continue;
        return {std::vector<double>(static_cast<std::size_t>(n), h5::read_double(f, path)), units(f, path)};
    }
    std::string list;
    for (const std::string& nm : names) list += (list.empty() ? "" : ", ") + nm;
    throw Error("no " + kind + " motor among [" + list + "] in /" + e);
}

std::vector<std::string> config_names(hid_t f, const std::string& init, const std::string& key, std::vector<std::string> fallback) {
    for (const char* grp : {"/technique/tomoconfig/", "/instrument/mrtomo_config/"}) {
        const std::string path = init + grp + key;
        if (h5::exists(f, path)) return h5::read_strings(f, path);
    }
    return fallback;
}

struct Quantity {
    double value;
    std::string unit;
};
std::optional<Quantity> scalar(hid_t f, const std::string& group, const std::string& name) {
    if (group.empty() || !h5::exists(f, group + "/" + name)) return std::nullopt;
    const std::string path = group + "/" + name;
    return Quantity{h5::read_double(f, path), units(f, path)};
}

// ------------------------------------------------------------------ the ring current (nxtomomill deduce_machine_current)

struct NoCurrent : Error {
    using Error::Error;
};

// the entry's "current" (in the entry, its measurement group, then instrument/machine) [A]
std::optional<std::vector<double>> entry_currents(hid_t f, const std::string& e) {
    for (const char* grp : {"", "/measurement", "/instrument/machine"}) {
        const std::string path = e + grp + "/current";
        if (!h5::is_dataset(f, path)) continue;
        if (h5::shape(f, path).type_class == H5T_STRING) continue;
        std::vector<double> v = h5::read_doubles(f, path);
        to_si(v, units(f, path), current_to_a(), "mA");
        return v;
    }
    return std::nullopt;
}

struct SequenceEntry {
    std::string name;
    std::int64_t n_frames = 0;
};

// the current at every frame of the sequence's entries, in order
std::vector<double> machine_current(hid_t f, const std::vector<SequenceEntry>& sequence) {
    std::map<std::int64_t, double> known;
    std::vector<std::int64_t> frame_t;
    for (const SequenceEntry& s : sequence) {
        std::optional<std::int64_t> st, en;
        if (h5::exists(f, s.name + "/start_time")) st = iso_ms(text(f, s.name + "/start_time"));
        if (h5::exists(f, s.name + "/end_time")) en = iso_ms(text(f, s.name + "/end_time"));
        if (s.n_frames) {
            if (!st || !en) {
                const auto t = st ? st : en;
                if (!t) throw NoCurrent("no start/end time in " + s.name);
                frame_t.insert(frame_t.end(), static_cast<std::size_t>(s.n_frames), *t);
            } else {
                const auto t = linspace_ms(*st, *en, static_cast<std::size_t>(s.n_frames));
                frame_t.insert(frame_t.end(), t.begin(), t.end());
            }
        }
        const auto cur = entry_currents(f, s.name);
        if (!cur || cur->empty()) continue;
        std::map<std::int64_t, double> fresh;
        if (!st || !en) {
            const auto t = st ? st : en;
            if (t) fresh[*t] = (*cur)[0];
        } else if (cur->size() == 1) {
            fresh[*st] = (*cur)[0];
        } else {
            const auto t = linspace_ms(*st, *en, cur->size());
            for (std::size_t i = 0; i < t.size(); ++i) fresh[t[i]] = (*cur)[i];
        }
        for (const auto& [t, c] : fresh)
            if (!std::isnan(c)) known[t] = c;
    }
    if (known.empty()) throw NoCurrent("no machine current found");
    std::vector<double> xp, fp;
    for (const auto& [t, c] : known) {
        xp.push_back(static_cast<double>(t * 1000000));  // datetime64[ns] as float64
        fp.push_back(c);
    }
    std::vector<double> out(frame_t.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = interp(static_cast<double>(frame_t[i] * 1000000), xp, fp);
    return out;
}

// ------------------------------------------------------------------ the master's location

std::string resolve_master(const std::string& given) {
    const std::string path = abspath(given);
    if (is_file(path)) return path;
    if (!is_dir(path)) throw Error("no such file or directory: " + path);
    std::string trimmed = path;
    while (trimmed.size() > 1 && trimmed.back() == '/') trimmed.pop_back();
    const std::string candidate = join(path, basename(trimmed) + ".h5");
    if (is_file(candidate)) return candidate;  // the scan directory <name>_0001/<name>_0001.h5
    std::vector<std::string> subs;
    for (const std::string& d : list_dir(path))
        if (is_file(join(join(path, d), d + ".h5"))) subs.push_back(d);
    if (subs.size() == 1) return join(join(path, subs[0]), subs[0] + ".h5");  // its parent directory
    if (subs.size() > 1) throw Error(path + " holds several scans; open one of them");
    throw Error("no BLISS master <dir>/<dir>.h5 found in " + path);
}

std::pair<long, long> entry_key(const std::string& name) {
    const std::size_t dot = name.find('.');
    const auto number = [](const std::string& s, long& v) {
        if (s.empty()) return false;
        char* end = nullptr;
        v = std::strtol(s.c_str(), &end, 10);
        return end && *end == '\0';
    };
    long a = 0, b = 0;
    if (dot == std::string::npos || name.find('.', dot + 1) != std::string::npos || !number(name.substr(0, dot), a) ||
        !number(name.substr(dot + 1), b))
        return {1L << 30, 0};
    return {a, b};
}

bool is_tomo_master(const std::string& path) {
    try {
        h5::Id f = h5::open_file(path);
        for (const std::string& k : h5::children(f.id, "/"))
            if (is_group(f.id, k) && entry_role(f.id, k) == "init") return true;
    } catch (const Error&) {
    }
    return false;
}

bool compatible(const Scan& a, const Scan& b) {
    const double ra = a.unwrapped_angles_deg.back() - a.unwrapped_angles_deg.front();
    const double rb = b.unwrapped_angles_deg.back() - b.unwrapped_angles_deg.front();
    return a.rows == b.rows && a.cols == b.cols && a.projections.size() == b.projections.size() &&
           is_close(a.pixel_size_m, b.pixel_size_m, 1e-6, 1e-8) && is_close(a.energy_kev, b.energy_kev, 1e-6, 1e-8) &&
           std::fabs(ra - rb) < 1.0;
}

}  // namespace

// ------------------------------------------------------------------ FrameList

FrameList FrameList::subset(const std::vector<std::int64_t>& idx) const {
    FrameList out;
    out.files = files;
    out.dataset = dataset;
    out.file_index.reserve(idx.size());
    out.frame_index.reserve(idx.size());
    for (std::int64_t i : idx) {
        out.file_index.push_back(file_index.at(static_cast<std::size_t>(i)));
        out.frame_index.push_back(frame_index.at(static_cast<std::size_t>(i)));
    }
    return out;
}

FrameList FrameList::concatenate(const std::vector<FrameList>& parts) {
    FrameList out;
    if (parts.empty()) return out;
    std::set<std::string> datasets;
    for (const FrameList& p : parts)
        if (p.size()) datasets.insert(p.dataset);
    if (datasets.size() > 1) throw Error("cannot concatenate frames of different datasets");
    out.dataset = datasets.empty() ? parts[0].dataset : *datasets.begin();
    std::map<std::string, std::int32_t> pos;
    for (const FrameList& p : parts) {
        std::vector<std::int32_t> remap(p.files.size());
        for (std::size_t k = 0; k < p.files.size(); ++k) {
            auto it = pos.find(p.files[k]);
            if (it == pos.end()) {
                it = pos.emplace(p.files[k], static_cast<std::int32_t>(out.files.size())).first;
                out.files.push_back(p.files[k]);
            }
            remap[k] = it->second;
        }
        for (std::size_t i = 0; i < p.size(); ++i) {
            out.file_index.push_back(remap.at(static_cast<std::size_t>(p.file_index[i])));
            out.frame_index.push_back(p.frame_index[i]);
        }
    }
    return out;
}

// ------------------------------------------------------------------ Scan

std::vector<std::string> Scan::referenced_files() const {
    std::vector<std::string> out;
    std::set<std::string> seen;
    std::vector<const FrameList*> lists = {&projections, &darks};
    for (const FrameList& f : flats) lists.push_back(&f);
    for (const FrameList* r : lists) {
        const std::set<std::int32_t> used(r->file_index.begin(), r->file_index.end());
        for (std::int32_t k : used) {
            const std::string& fn = r->files.at(static_cast<std::size_t>(k));
            if (seen.insert(fn).second) out.push_back(fn);
        }
    }
    return out;
}

void Scan::check_files() const {
    std::vector<std::string> missing;
    for (const std::string& fn : referenced_files())
        if (!is_file(fn)) missing.push_back(fn);
    if (missing.empty()) return;
    std::string list;
    for (std::size_t i = 0; i < missing.size() && i < 5; ++i) list += (i ? ", " : "") + missing[i];
    throw Error(std::to_string(missing.size()) + " detector files missing for " + name + ": " + list + (missing.size() > 5 ? " ..." : ""));
}

Scan open_scan(const std::string& path) {
    Scan scan;
    scan.master = resolve_master(path);
    scan.directory = dirname(scan.master);
    {
        const std::string b = basename(scan.master);
        const std::size_t dot = b.find_last_of('.');
        scan.name = dot == std::string::npos || dot == 0 ? b : b.substr(0, dot);
    }
    const std::string& master = scan.master;
    h5::Id file = h5::open_file(master);
    const hid_t f = file.id;

    // the entries in nxtomomill's order: by start time (to the millisecond), ties in the file's order
    std::vector<std::string> keys;
    for (const std::string& k : h5::children(f, "/"))
        if (is_group(f, k)) keys.push_back(k);
    {
        std::vector<std::pair<std::int64_t, std::string>> order;
        for (const std::string& k : keys) {
            std::int64_t t = 0;
            if (h5::exists(f, k + "/start_time")) t = iso_ms(text(f, k + "/start_time"));
            else if (entry_key(k).first < (1L << 30)) t = entry_key(k).first;
            order.emplace_back(t, k);
        }
        std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (std::size_t i = 0; i < keys.size(); ++i) keys[i] = order[i].second;
    }
    std::map<std::string, std::string> roles;
    std::vector<std::string> inits;
    for (const std::string& k : keys) {
        roles[k] = entry_role(f, k);
        if (roles[k] == "init") inits.push_back(k);
    }
    if (inits.empty()) throw Error(master + ": no tomo initialisation entry (not a BLISS tomo scan?)");
    if (inits.size() > 1) throw Error(master + ": several tomo sequences (z-series?) are not supported");
    const std::string init = inits[0];
    std::vector<std::string> steps;
    for (auto it = std::find(keys.begin(), keys.end(), init) + 1; it != keys.end(); ++it) {
        const std::string& r = roles[*it];
        if (r == "dark" || r == "flat" || r == "proj" || r == "align") steps.push_back(*it);
    }
    {
        std::vector<std::string> both = {init};
        both.insert(both.end(), steps.begin(), steps.end());
        scan.detector = find_detector(f, both);
    }
    const std::string& det = scan.detector;
    const auto rot_names = config_names(f, init, "rotation", {"mrsrot", "srot"});
    const auto z_names = config_names(f, init, "translation_z", {"mrsz", "sz"});
    const auto y_names = config_names(f, init, "translation_y", {"mryrot", "yrot"});

    std::vector<SequenceEntry> sequence;  // every step, with its frames (none for the entries of monitors)
    std::map<std::string, std::vector<FrameList>> refs;
    std::vector<char> kinds;  // per frame of the sequence: 'd', 'f', 'p' or 'a'
    std::vector<FrameList> flat_chunks;
    std::vector<double> arc_delta, arc_period;
    std::vector<std::int64_t> proj_index;  // the place in the sequence of each projection
    for (const std::string& e : steps) {
        const std::string ds = frame_dataset_path(f, e, det);
        const std::string& role = roles[e];
        std::int64_t nfr = 0;
        if (!ds.empty()) nfr = static_cast<std::int64_t>(h5::shape(f, ds).dims[0]);
        sequence.push_back({e, nfr});
        scan.entries.push_back({e, role, optional_text(f, e + "/title"), nfr});
        if (!nfr) continue;
        const h5::Shape sh = h5::shape(f, ds);
        const std::string dtype = dtype_name(f, ds);
        if (scan.rows == 0) {
            scan.rows = static_cast<int>(sh.dims[1]);
            scan.cols = static_cast<int>(sh.dims[2]);
            scan.dtype = dtype;
        } else if (static_cast<int>(sh.dims[1]) != scan.rows || static_cast<int>(sh.dims[2]) != scan.cols || dtype != scan.dtype) {
            throw Error(e + ": frames of another shape or type than the first entry's");
        }
        FrameList r = frames_of(f, master, scan.directory, e, det, ds);
        if (role == "flat") flat_chunks.push_back(r);
        refs[role].push_back(std::move(r));
        kinds.insert(kinds.end(), static_cast<std::size_t>(nfr), role[0]);
        if (role == "proj") {
            auto [a, au] = per_frame(f, e, rot_names, nfr, "rotation");
            auto [z, zu] = per_frame(f, e, z_names, nfr, "translation_z");
            if (!(au.empty() || au == "deg" || au == "degree" || au == "degrees")) throw Error("unexpected rotation unit '" + au + "'");
            to_si(z, zu, length_to_m(), "mm");
            scan.angles_deg.insert(scan.angles_deg.end(), a.begin(), a.end());
            scan.z_translation_m.insert(scan.z_translation_m.end(), z.begin(), z.end());
            const std::int64_t first = static_cast<std::int64_t>(kinds.size()) - nfr;
            for (std::int64_t k = 0; k < nfr; ++k) proj_index.push_back(first + k);
            for (const std::string& nm : rot_names) {  // the arc swept during and between exposures, as recorded
                const std::string d = e + "/measurement/" + nm + "_delta", p = e + "/measurement/" + nm + "_period";
                if (h5::exists(f, d) && h5::exists(f, p)) {
                    for (double v : h5::read_doubles(f, d)) arc_delta.push_back(std::fabs(v));
                    for (double v : h5::read_doubles(f, p)) arc_period.push_back(std::fabs(v));
                    break;
                }
            }
        }
    }
    if (refs["proj"].empty()) throw Error(master + ": no projection frames");

    // the ring current at every frame of the sequence
    std::vector<double> current;
    try {
        current = machine_current(f, sequence);
    } catch (const NoCurrent&) {
        current.assign(kinds.size(), kNaN);
    }
    scan.unwrapped_angles_deg = unwrap_360(scan.angles_deg);
    scan.projections = FrameList::concatenate(refs["proj"]);
    for (std::size_t i = 0; i < kinds.size(); ++i) {
        if (kinds[i] == 'p') scan.control_A.push_back(current[i]);
        if (kinds[i] == 'd') scan.darks_control_A.push_back(current[i]);
    }

    // the flat series: uninterrupted runs of flat frames in the sequence
    {
        const FrameList all = FrameList::concatenate(flat_chunks);
        std::vector<double> proj_x(proj_index.begin(), proj_index.end());
        std::int64_t n_flat = 0;
        std::size_t i = 0;
        while (i < kinds.size()) {
            if (kinds[i] != 'f') {
                ++i;
                continue;
            }
            std::size_t j = i;
            std::vector<std::int64_t> pos;
            std::vector<double> ctl;
            while (j < kinds.size() && kinds[j] == 'f') pos.push_back(n_flat++), ctl.push_back(current[j]), ++j;
            scan.flats.push_back(all.subset(pos));
            scan.flats_control_A.push_back(std::move(ctl));
            const double mid = 0.5 * static_cast<double>(static_cast<std::int64_t>(i) + static_cast<std::int64_t>(j - 1));
            scan.flats_unwrapped_angles_deg.push_back(interp(mid, proj_x, scan.unwrapped_angles_deg));
            i = j;
        }
    }
    scan.darks = FrameList::concatenate(refs["dark"]);

    // the scalars
    const std::string tech = init + "/technique";
    const std::string sc = is_group(f, tech + "/scan") ? tech + "/scan" : std::string();
    const std::string optic = is_group(f, tech + "/optic") ? tech + "/optic" : std::string();
    auto px = scalar(f, optic, "sample_pixel_size");
    if (!px) px = scalar(f, sc, "sample_pixel_size");
    if (!px) throw Error(master + ": no sample pixel size");
    scan.pixel_size_m = to_si(px->value, px->unit, length_to_m(), "um");
    if (const auto d = scalar(f, optic, "optics_pixel_size")) scan.detector_pixel_size_m = to_si(d->value, d->unit, length_to_m(), "um");
    const auto en = scalar(f, sc, "energy");
    scan.energy_kev = en ? en->value : kNaN;
    if (const auto prop = scalar(f, sc, "effective_propagation_distance")) {
        scan.distance_m = to_si(prop->value, prop->unit, length_to_m(), "mm");
    } else {  // nxtomomill: (-z1 z2) / (-z1 + z2), z1 = -(source to sample)
        const auto z2 = scalar(f, sc, "sample_detector_distance");
        const auto z1 = scalar(f, sc, "source_sample_distance");
        if (!z2) {
            scan.distance_m = kNaN;
        } else if (!z1) {
            scan.distance_m = to_si(z2->value, z2->unit, length_to_m(), "mm");
        } else {
            const double a = -to_si(z1->value, z1->unit, length_to_m(), "mm");
            const double b = to_si(z2->value, z2->unit, length_to_m(), "mm");
            scan.distance_m = (-a * b) / (-a + b);
        }
    }
    const auto et = scalar(f, sc, "exposure_time");
    if (et) scan.count_time_s = et->value * 1e-3;
    const bool continuous = !sc.empty() && h5::exists(f, sc + "/scan_type") && [&] {
        std::string t = text(f, sc + "/scan_type");
        for (char& c : t) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return t == "CONTINUOUS";
    }();
    if (continuous && !arc_delta.empty()) {
        const double period = median(arc_period);
        if (period > 0) scan.exposure_arc = median(arc_delta) / period;
    }
    if (continuous && !scan.exposure_arc && et && et->value != 0) {
        const auto lt = scalar(f, sc, "latency_time");
        scan.exposure_arc = et->value / (et->value + (lt ? lt->value : 0.0));
    }
    if (scan.exposure_arc && !(0.0 < *scan.exposure_arc && *scan.exposure_arc <= 1.0)) scan.exposure_arc.reset();
    // the lateral motor at the first frame of the sequence
    {
        auto [y, yu] = per_frame(f, steps[0], y_names, 1, "translation_y");
        scan.y_translation_m = to_si(y[0], yu, length_to_m(), "mm");
    }
    scan.estimated_cor_from_motor_pix = scan.y_translation_m / scan.pixel_size_m;
    scan.sample_name = optional_text(f, init + "/sample/name");
    scan.start_time = optional_text(f, init + "/start_time");
    scan.end_time = optional_text(f, init + "/end_time");
    return scan;
}

std::vector<Scan> discover_scans(const std::string& root_given, int max_depth, bool require_files) {
    const std::string root = abspath(root_given);
    std::vector<std::string> found;
    const std::function<void(const std::string&, int)> walk = [&](const std::string& d, int depth) {
        const auto names = list_dir(d);
        const std::string base = basename(d);
        const std::string master = join(d, base + ".h5");
        if (std::find(names.begin(), names.end(), base + ".h5") != names.end() && is_file(master)) {
            if (is_tomo_master(master)) found.push_back(master);
            return;
        }
        if (depth >= max_depth) return;
        for (const std::string& n : names) {
            const std::string p = join(d, n);
            if (!n.empty() && n[0] != '.' && is_dir(p)) walk(p, depth + 1);
        }
    };
    walk(root, 0);
    std::stable_sort(found.begin(), found.end(), [](const std::string& a, const std::string& b) {
        return std::make_pair(basename(a), a) < std::make_pair(basename(b), b);
    });
    std::vector<Scan> scans;
    for (const std::string& m : found) {
        try {
            Scan s = open_scan(m);
            if (require_files) s.check_files();
            scans.push_back(std::move(s));
        } catch (const std::exception&) {
        }
    }
    return scans;
}

std::optional<TetraName> parse_tetra_name(const std::string& name) {
    static const std::regex re(R"(^(.+?)(?:_(HA|TA[2-9]?))?((?:_(?:fc|cs)_)*?)_(\d{4})(?:_(\d{4}))?$)");
    static const std::vector<std::string> kinds = {"HA", "TA", "TA2", "TA3", "TA4", "TA5", "TA6", "TA7", "TA8", "TA9"};
    std::smatch m;
    if (!std::regex_match(name, m, re) || !m[2].matched) return std::nullopt;
    TetraName t;
    t.base = m[1].str();
    t.kind = m[2].str();
    t.rank = static_cast<int>(std::find(kinds.begin(), kinds.end(), t.kind) - kinds.begin());
    t.suffix = "_" + m[4].str() + (m[5].matched ? "_" + m[5].str() : std::string());
    t.crash = m[3].length() > 0;
    return t;
}

std::vector<std::vector<int>> group_companions(const std::vector<Scan>& scans) {
    std::map<std::pair<std::string, std::string>, std::vector<std::pair<int, int>>> families;  // (base, suffix) -> (rank, scan)
    std::vector<int> untagged;
    for (int i = 0; i < static_cast<int>(scans.size()); ++i) {
        const auto t = parse_tetra_name(scans[i].name);
        if (!t) untagged.push_back(i);
        else if (!t->crash) families[{t->base, t->suffix}].push_back({t->rank, i});
    }
    std::vector<std::vector<int>> out;
    for (auto& [key, members] : families) {
        std::stable_sort(members.begin(), members.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (std::size_t k = 1; k < members.size(); ++k)
            if (members[k].first == members[k - 1].first) throw Error("duplicate HA/TA kinds for " + key.first + key.second);
        if (members.size() >= 2) {
            std::vector<int> g;
            for (const auto& m : members) g.push_back(m.second);
            out.push_back(std::move(g));
        }
    }
    // scans without a kind in their name: grouped by geometry
    std::set<std::size_t> used;
    for (std::size_t i = 0; i < untagged.size(); ++i) {
        if (used.count(i)) continue;
        const Scan& a = scans[static_cast<std::size_t>(untagged[i])];
        std::vector<int> g = {untagged[i]};
        for (std::size_t j = i + 1; j < untagged.size(); ++j) {
            const Scan& b = scans[static_cast<std::size_t>(untagged[j])];
            if (used.count(j) || !compatible(a, b) || is_close(a.y_translation_m, b.y_translation_m, 1e-5, 1e-6)) continue;
            bool distinct = true;
            for (int c : g) distinct = distinct && !is_close(scans[static_cast<std::size_t>(c)].y_translation_m, b.y_translation_m, 1e-5, 1e-6);
            if (!distinct) continue;
            g.push_back(untagged[j]);
            used.insert(j);
        }
        if (g.size() >= 2) {
            std::stable_sort(g.begin(), g.end(), [&](int x, int y) {
                return std::fabs(scans[static_cast<std::size_t>(x)].estimated_cor_from_motor_pix) <
                       std::fabs(scans[static_cast<std::size_t>(y)].estimated_cor_from_motor_pix);
            });
            out.push_back(std::move(g));
        }
    }
    return out;
}

std::vector<int> pair_companions(const std::vector<Scan>& scans) {
    const auto groups = group_companions(scans);
    if (groups.size() != 1) {
        std::string list;
        for (const auto& g : groups) {
            list += list.empty() ? "[" : ", [";
            for (std::size_t k = 0; k < g.size(); ++k) list += (k ? ", " : "") + scans[static_cast<std::size_t>(g[k])].name;
            list += "]";
        }
        throw Error("expected one HA/TA family, found " + std::to_string(groups.size()) + (list.empty() ? "" : ": " + list));
    }
    const auto& g = groups[0];
    for (std::size_t k = 1; k < g.size(); ++k)
        if (!compatible(scans[static_cast<std::size_t>(g[0])], scans[static_cast<std::size_t>(g[k])]))
            throw Error(scans[static_cast<std::size_t>(g[0])].name + " and " + scans[static_cast<std::size_t>(g[k])].name + " are not compatible companions");
    return g;
}

}  // namespace helirec::calibration
