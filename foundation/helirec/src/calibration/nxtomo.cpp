// A NeXus scan file with an NXtomo subentry (GDA, Diamond I12) as a Record and as a Scan: calibration/nxtomo.hpp
// states the layout read and what is derived.
#include "calibration/nxtomo.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>

#include "calibration/h5_read.hpp"
#include "calibration/iso_time.hpp"
#include "calibration/unwrap.hpp"
#include "io/json.hpp"

namespace helirec::calibration::nxtomo {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

bool is_file(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool is_group(hid_t f, const std::string& path) {
    if (!h5::exists(f, path)) return false;
    H5O_info2_t info;
    return H5Oget_info_by_name3(f, path.c_str(), &info, H5O_INFO_BASIC, H5P_DEFAULT) >= 0 && info.type == H5O_TYPE_GROUP;
}

std::string text_if(hid_t f, const std::string& path) {
    if (!h5::is_dataset(f, path) || h5::shape(f, path).type_class != H5T_STRING) return std::string();
    return h5::read_string(f, path);
}

bool is_number(hid_t f, const std::string& path, hsize_t* count = nullptr) {
    if (!h5::is_dataset(f, path)) return false;
    const h5::Shape sh = h5::shape(f, path);
    if (count) *count = sh.count();
    return sh.type_class == H5T_INTEGER || sh.type_class == H5T_FLOAT;
}

// the entry of the file that holds an NXtomo subentry; "" if none
std::string tomo_entry_of(hid_t f) {
    for (const std::string& e : h5::children(f, "/"))
        if (text_if(f, e + "/tomo_entry/definition") == "NXtomo") return e;
    return std::string();
}

double median(std::vector<double> v) {
    if (v.empty()) return kNaN;
    const std::size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m), v.end());
    if (v.size() % 2) return v[m];
    return (*std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m)) + v[m]) / 2.0;
}

// metres per unit of a length; 0: not a length this reader knows
double metres_per(const std::string& unit) {
    static const std::map<std::string, double> table = {{"m", 1.0}, {"mm", 1e-3}, {"um", 1e-6}, {"micron", 1e-6}, {"nm", 1e-9}};
    const auto it = table.find(unit);
    return it == table.end() ? 0.0 : it->second;
}

// a length the NXtomo detector group states, in metres; NaN where it holds none it gives a unit for
double stated_length_m(hid_t f, const std::string& path) {
    hsize_t n = 0;
    if (!is_number(f, path, &n) || n != 1) return kNaN;
    const double v = h5::read_double(f, path), per = metres_per(h5::attribute_text(f, path, "units"));
    return std::isfinite(v) && v > 0 && per > 0 ? v * per : kNaN;
}

// "name(a=1, b=x, ..)" -> {a: "1", b: "x"}; a part without "=" belongs to the value before it
std::map<std::string, std::string> command_arguments(const std::string& command) {
    std::map<std::string, std::string> out;
    const std::size_t open = command.find('('), close = command.rfind(')');
    if (open == std::string::npos || close == std::string::npos || close <= open) return out;
    const std::string body = command.substr(open + 1, close - open - 1);
    std::string last;
    std::size_t i = 0;
    while (i <= body.size()) {
        std::size_t j = body.find(',', i);
        if (j == std::string::npos) j = body.size();
        std::string part = body.substr(i, j - i);
        const std::size_t a = part.find_first_not_of(' ');
        part = a == std::string::npos ? std::string() : part.substr(a);
        const std::size_t eq = part.find('=');
        if (eq != std::string::npos && eq > 0) last = part.substr(0, eq), out[last] = part.substr(eq + 1);
        else if (!last.empty()) out[last] += "," + body.substr(i, j - i);
        i = j + 1;
    }
    return out;
}

// the number a command's argument states; NaN if it has none or it is no number
double argument(const std::map<std::string, std::string>& args, const std::string& name) {
    const auto it = args.find(name);
    if (it == args.end() || it->second.empty()) return kNaN;
    char* end = nullptr;
    const double v = std::strtod(it->second.c_str(), &end);
    return end && *end == '\0' ? v : kNaN;
}

struct Run {
    int key = 0;
    std::int64_t first = 0, count = 0;
};

// What both readings take from the file.
struct Master {
    std::string path, directory, entry;
    std::string title, start_time, end_time, detector = "detector";
    std::vector<int> image_key;      // per frame the record describes
    std::vector<double> angle_deg;   // per frame, as stored
    std::vector<double> current_A;   // per frame (the scan's one value repeated)
    std::vector<Run> runs;           // of equal image_key, in order
    double count_time_s = kNaN, period_s = kNaN;  // of the projections (medians); NaN: not recorded
    double step_deg = kNaN;                       // median angular step of the projections
    std::string frame_file, frame_dataset, dtype;
    bool frames_present = false;
    std::int64_t frames_in_file = 0;
    int rows = 0, cols = 0;
    std::map<std::string, double> positions;      // every motor before the scan
    std::string command;
    std::map<std::string, std::string> arguments; // of the command
    std::string rotation = "rotation_angle", translation, height_stage, on_rotation[2];
    double pixel_um = kNaN, energy_kev = kNaN, propagation_m = kNaN, source_m = kNaN;

    // the translation stage's position during the frames of a role: the command's, else the snapshot's
    double across_beam(Role role) const {
        const double stated = argument(arguments, role == Role::Flat ? "outOfBeamPosition" : "inBeamPosition");
        if (std::isfinite(stated) && role != Role::Dark) return stated;
        const auto it = positions.find(translation);
        return it == positions.end() ? kNaN : it->second;
    }
    double height() const {
        const auto it = positions.find(height_stage);
        return it == positions.end() ? 0.0 : it->second;
    }
    std::map<std::string, double> positions_of(Role role) const {
        std::map<std::string, double> out = positions;
        const double y = across_beam(role);
        if (!translation.empty() && std::isfinite(y)) out[translation] = y;
        if (!out.count(height_stage)) out[height_stage] = 0.0;
        return out;
    }
};

Role role_of(int key) { return key == 2 ? Role::Dark : key == 1 ? Role::Flat : key == 0 ? Role::Projection : Role::Unknown; }

// the name of the nth run of a role (from 1): "darks", "flats", "projections", then "flats2", ..
std::string run_name(Role role, int nth) {
    const std::string kind = role == Role::Dark ? "darks" : role == Role::Flat ? "flats" : "projections";
    return nth == 1 ? kind : kind + std::to_string(nth);
}

// where the frames are: the target of an external link, or the dataset in the master itself
void locate_frames(hid_t f, Master& m, const std::string& path) {
    H5L_info2_t info;
    if (H5Lget_info2(f, path.c_str(), &info, H5P_DEFAULT) < 0) throw Error(m.path + ": no frames at " + path);
    if (info.type == H5L_TYPE_EXTERNAL) {
        std::vector<char> value(info.u.val_size);
        const char* file = nullptr;
        const char* object = nullptr;
        if (H5Lget_val(f, path.c_str(), value.data(), value.size(), H5P_DEFAULT) < 0 ||
            H5Lunpack_elink_val(value.data(), value.size(), nullptr, &file, &object) < 0)
            throw Error(m.path + ": unreadable link at " + path);
        const std::string name = file;
        m.frame_file = !name.empty() && name[0] == '/' ? name : m.directory + "/" + name;
        m.frame_dataset = object;
    } else {
        m.frame_file = m.path;
        m.frame_dataset = "/" + path;
    }
    m.frames_present = is_file(m.frame_file);
    if (!m.frames_present) return;
    h5::Id file = h5::open_file(m.frame_file);
    const h5::FrameDataset fd = h5::frame_dataset(file.id, m.frame_dataset);
    if (fd.is_virtual) throw Error(m.frame_file + ": the frames of an NXtomo master are read from one dataset, not a virtual one");
    m.frames_in_file = fd.frames, m.rows = fd.rows, m.cols = fd.cols;
    h5::Id d(H5Dopen2(file.id, m.frame_dataset.c_str(), H5P_DEFAULT), H5Dclose);
    h5::Id t(H5Dget_type(d.id), H5Tclose);
    const H5T_class_t cls = H5Tget_class(t.id);
    const std::string bits = std::to_string(8 * H5Tget_size(t.id));
    m.dtype = cls == H5T_FLOAT ? "float" + bits : cls == H5T_INTEGER ? (H5Tget_sign(t.id) == H5T_SGN_NONE ? "uint" : "int") + bits : "unknown";
}

void read_supplement(Master& m) {
    const std::string path = m.directory + "/" + kSupplement;
    std::vector<std::string> on_rotation;
    if (is_file(path)) {
        std::ifstream in(path, std::ios::binary);
        std::stringstream text;
        text << in.rdbuf();
        Json doc;
        try {
            doc = Json::parse(text.str());
        } catch (const std::exception& e) {
            throw Error(path + ": " + e.what());
        }
        if (!doc.is(Json::Object)) throw Error(path + ": not a JSON object");
        const auto number = [&](const char* key, double& out) {
            const Json* v = doc.find(key);
            if (!v || v->is(Json::Null)) return;
            if (!v->is(Json::Number)) throw Error(path + ": '" + key + "' is not a number");
            out = v->number;
        };
        number("pixel_size_um", m.pixel_um);
        number("energy_kev", m.energy_kev);
        number("propagation_distance_m", m.propagation_m);
        number("source_sample_distance_m", m.source_m);
        if (const Json* v = doc.find("height_stage"); v && v->is(Json::String)) m.height_stage = v->string;
        if (const Json* v = doc.find("stages_on_rotation"); v && v->is(Json::Array))
            for (const Json& name : v->array)
                if (name.is(Json::String)) on_rotation.push_back(name.string);
        if (!on_rotation.empty() && on_rotation.size() != 2) throw Error(path + ": 'stages_on_rotation' names two motors");
    }
    // the stages named after the translation stage's, where the file holds such motors
    const std::size_t cut = m.translation.find_last_of('_');
    const std::string prefix = cut == std::string::npos ? std::string() : m.translation.substr(0, cut);
    if (m.height_stage.empty() && !prefix.empty())
        for (const char* suffix : {"_y1", "_y"})
            if (m.height_stage.empty() && m.positions.count(prefix + suffix)) m.height_stage = prefix + suffix;
    if (m.height_stage.empty()) m.height_stage = "height";  // not known: every scan at the height 0
    if (on_rotation.empty() && !prefix.empty() && m.positions.count(prefix + "_tx") && m.positions.count(prefix + "_tz")) on_rotation = {prefix + "_tx", prefix + "_tz"};
    if (on_rotation.size() == 2) m.on_rotation[0] = on_rotation[0], m.on_rotation[1] = on_rotation[1];
}

Master read_master(const std::string& path) {
    Master m;
    m.path = std::filesystem::absolute(path).lexically_normal().string();
    m.directory = std::filesystem::path(m.path).parent_path().string();
    h5::Id file = h5::open_file(m.path);
    const hid_t f = file.id;
    m.entry = tomo_entry_of(f);
    if (m.entry.empty()) throw Error(m.path + ": no entry holds an NXtomo subentry");
    const std::string e = m.entry, tomo = e + "/tomo_entry", det = tomo + "/instrument/detector";
    m.title = text_if(f, e + "/title");
    m.start_time = text_if(f, e + "/start_time");
    m.end_time = text_if(f, e + "/end_time");

    locate_frames(f, m, det + "/data");

    if (!is_number(f, det + "/image_key")) throw Error(m.path + ": no " + det + "/image_key");
    const std::vector<double> key = h5::read_doubles(f, det + "/image_key");
    std::string angle_path;
    for (const std::string& p : {tomo + "/sample/rotation_angle", tomo + "/data/rotation_angle"})
        if (angle_path.empty() && is_number(f, p)) angle_path = p;
    if (angle_path.empty()) throw Error(m.path + ": no rotation_angle in " + tomo);
    const std::vector<double> angle = h5::read_doubles(f, angle_path);
    // the frames the record describes (it may be shorter than the frame file by a frame)
    std::size_t n = std::min(key.size(), angle.size());
    if (m.frames_present) n = std::min<std::size_t>(n, static_cast<std::size_t>(m.frames_in_file));
    if (n == 0) throw Error(m.path + ": the record describes no frame");
    m.image_key.resize(n);
    for (std::size_t i = 0; i < n; ++i) m.image_key[i] = static_cast<int>(std::lround(key[i]));
    m.angle_deg.assign(angle.begin(), angle.begin() + static_cast<std::ptrdiff_t>(n));
    for (std::size_t i = 0; i < n;) {
        std::size_t j = i;
        while (j < n && m.image_key[j] == m.image_key[i]) ++j;
        m.runs.push_back({m.image_key[i], static_cast<std::int64_t>(i), static_cast<std::int64_t>(j - i)});
        i = j;
    }

    // the ring current: per frame, or the scan's one value for every frame
    {
        double per_A = 1e-3;  // GDA states mA
        const std::string p = tomo + "/control/data";
        std::vector<double> c;
        if (is_number(f, p)) {
            c = h5::read_doubles(f, p);
            const std::string unit = h5::attribute_text(f, p, "units");
            if (unit == "A") per_A = 1.0;
            else if (unit == "uA") per_A = 1e-6;
        }
        m.current_A.assign(n, kNaN);
        for (std::size_t i = 0; i < n; ++i)
            if (c.size() >= n) m.current_A[i] = c[i] * per_A;
            else if (!c.empty()) m.current_A[i] = c[0] * per_A;
    }

    // exposure and frame period of the projections
    std::vector<std::size_t> proj;
    for (std::size_t i = 0; i < n; ++i)
        if (m.image_key[i] == 0) proj.push_back(i);
    {
        std::vector<double> step;
        for (std::size_t k = 1; k < proj.size(); ++k)
            if (proj[k] == proj[k - 1] + 1) step.push_back(m.angle_deg[proj[k]] - m.angle_deg[proj[k - 1]]);
        m.step_deg = median(step);
    }
    std::vector<double> capture;
    for (const std::string& g : h5::children(f, e + "/instrument")) {
        const std::string group = e + "/instrument/" + g;
        if (!is_group(f, group)) continue;
        hsize_t count = 0;
        if (is_number(f, group + "/count_time", &count) && count > 0) {
            m.detector = g;
            const std::vector<double> t = h5::read_doubles(f, group + "/count_time");
            std::vector<double> of_projections;
            for (const std::size_t i : proj)
                if (i < t.size()) of_projections.push_back(t[i]);
            m.count_time_s = of_projections.empty() ? t[0] : median(of_projections);
        }
        if (capture.empty() && is_number(f, group + "/CaptureTime", &count) && count >= n) capture = h5::read_doubles(f, group + "/CaptureTime");
    }
    if (!capture.empty() && proj.size() > 1 && !m.start_time.empty() && !m.end_time.empty()) {
        std::vector<double> dt;
        for (std::size_t k = 1; k < proj.size(); ++k)
            if (proj[k] == proj[k - 1] + 1) dt.push_back(capture[proj[k]] - capture[proj[k - 1]]);
        const double period = median(dt);
        double lasted = kNaN;  // the entry, seconds
        try {
            lasted = static_cast<double>(iso_ms(m.end_time) - iso_ms(m.start_time)) * 1e-3;
        } catch (const Error&) {
        }
        // the capture's unit is not stated: seconds if the projections then fit into the entry, else milliseconds
        if (period > 0 && std::isfinite(lasted)) {
            const double span = period * static_cast<double>(proj.size());
            if (span <= lasted) m.period_s = period;
            else if (span * 1e-3 <= lasted) m.period_s = period * 1e-3;
        }
    }

    // every motor before the scan, the command and the stages it names
    const std::string before = e + "/before_scan";
    for (const std::string& g : h5::children(f, before)) {
        const std::string group = before + "/" + g;
        hsize_t count = 0;
        if (is_number(f, group, &count)) {
            if (count == 1) m.positions[g] = h5::read_double(f, group);
            continue;
        }
        if (!is_group(f, group)) continue;
        for (const std::string& name : h5::children(f, group))
            if (is_number(f, group + "/" + name, &count) && count == 1) m.positions[name] = h5::read_double(f, group + "/" + name);
    }
    m.command = text_if(f, before + "/flyscan_cfg/tomography_cmd");
    m.arguments = command_arguments(m.command);
    m.translation = text_if(f, before + "/flyscan_cfg/tomography_translation");
    if (const std::string theta = text_if(f, before + "/flyscan_cfg/tomography_flyscan_theta"); !theta.empty()) m.rotation = theta;

    m.pixel_um = stated_length_m(f, det + "/x_pixel_size") * 1e6;
    m.propagation_m = stated_length_m(f, det + "/distance");
    read_supplement(m);
    if (!(std::isfinite(m.pixel_um) && m.pixel_um > 0))
        throw Error(m.path + ": the record states no pixel size; give it in " + m.directory + "/" + kSupplement + " as {\"pixel_size_um\": ..}");
    return m;
}

Value numbers(std::vector<double> v) {
    Value out;
    out.num = std::move(v);
    return out;
}
Value texts(std::vector<std::string> v) {
    Value out;
    out.str = std::move(v);
    return out;
}

// the frames of a run as a frame list
FrameList frames_of(const Master& m, const std::vector<const Run*>& runs) {
    FrameList out;
    out.files = {m.frame_file};
    out.dataset = m.frame_dataset;
    for (const Run* r : runs)
        for (std::int64_t i = 0; i < r->count; ++i) out.file_index.push_back(0), out.frame_index.push_back(static_cast<std::int32_t>(r->first + i));
    return out;
}

}  // namespace

bool is_master(const std::string& path) {
    try {
        h5::Id f = h5::open_file(path);
        return !tomo_entry_of(f.id).empty();
    } catch (const Error&) {
        return false;
    }
}

Record read_record(const std::string& path, bool frame_files) {
    const Master m = read_master(path);
    Record rec;
    rec.path = m.path;
    rec.directory = m.directory;
    rec.sample_name = m.title;
    {
        const std::size_t i = m.title.find_last_of('_');
        rec.label = i == std::string::npos ? m.title : m.title.substr(i + 1);
        if (rec.label.empty()) rec.label = std::filesystem::path(m.path).stem().string();
    }
    rec.detector = m.detector;
    rec.rotation[0] = rec.rotation[1] = m.rotation;
    rec.translation_y[0] = rec.translation_y[1] = m.translation.empty() ? std::string("translation") : m.translation;
    rec.translation_z[0] = rec.translation_z[1] = m.height_stage;
    rec.start_time = m.start_time;
    rec.end_time = m.end_time;

    // what the checks and the calibration ask of a record's technique, under the names a BLISS record has them
    const double per_turn = std::isfinite(m.step_deg) && m.step_deg != 0 ? std::nearbyint(360.0 / std::fabs(m.step_deg)) : kNaN;
    auto& tech = rec.technique.items;
    tech["scan/sample_pixel_size"] = numbers({m.pixel_um});
    if (std::isfinite(per_turn)) tech["scan/tomo_n"] = numbers({per_turn}), tech["scan/scan_range"] = numbers({per_turn * m.step_deg});
    if (std::isfinite(m.energy_kev)) tech["scan/energy"] = numbers({m.energy_kev});
    if (std::isfinite(m.source_m)) tech["scan/source_sample_distance"] = numbers({m.source_m * 1e3});
    if (std::isfinite(m.propagation_m)) tech["scan/sample_detector_distance"] = numbers({m.propagation_m * 1e3});
    if (std::isfinite(m.count_time_s)) tech["scan/exposure_time"] = numbers({m.count_time_s * 1e3});
    if (m.rows > 0) tech["detector/" + m.detector + "/size"] = numbers({static_cast<double>(m.cols), static_cast<double>(m.rows)});
    tech["tomoconfig/detector"] = texts({m.detector});
    tech["tomoconfig/rotation"] = texts({m.rotation});
    tech["tomoconfig/translation_y"] = texts({rec.translation_y[1]});
    tech["tomoconfig/translation_z"] = texts({m.height_stage});
    if (!m.on_rotation[0].empty()) tech["tomoconfig/sample_u"] = texts({m.on_rotation[0]}), tech["tomoconfig/sample_v"] = texts({m.on_rotation[1]});
    auto& inst = rec.instrument.items;
    if (!m.command.empty()) inst["command"] = texts({m.command});
    for (const auto& [name, value] : m.arguments) inst["command/" + name] = texts({value});

    const double arc = std::isfinite(m.period_s) && std::isfinite(m.count_time_s) && m.period_s > 0 ? m.step_deg * m.count_time_s / m.period_s : 0.0;
    int n_of[3] = {0, 0, 0};
    for (const Run& run : m.runs) {
        const Role role = role_of(run.key);
        if (role == Role::Unknown) continue;  // frames marked invalid
        Stretch s;
        s.entry = run_name(role, ++n_of[run.key]);
        s.title = m.title;
        s.start_time = m.start_time, s.end_time = m.end_time;
        s.has_image_key = true, s.image_key = run.key, s.role = role;
        s.n_frames = run.count, s.rows = m.rows, s.cols = m.cols;
        s.pieces.push_back({m.frame_file, m.frame_dataset, 0, run.count, run.first, m.frames_present});
        s.positioners_start = s.positioners_end = m.positions_of(role);
        if (std::isfinite(m.current_A[static_cast<std::size_t>(run.first)])) s.machine.items["current"] = numbers({m.current_A[static_cast<std::size_t>(run.first)] * 1e3});
        if (role == Role::Projection) {
            const std::size_t a = static_cast<std::size_t>(run.first), c = static_cast<std::size_t>(run.count);
            s.rot_trig.assign(m.angle_deg.begin() + static_cast<std::ptrdiff_t>(a), m.angle_deg.begin() + static_cast<std::ptrdiff_t>(a + c));
            s.rot_delta.assign(c, arc);
            s.rot_period.assign(c, m.step_deg);
            s.rot_center.resize(c);
            for (std::size_t i = 0; i < c; ++i) s.rot_center[i] = s.rot_trig[i] + 0.5 * arc;
            s.rot_center_name = m.rotation;
            if (std::isfinite(m.period_s)) s.timer_period.assign(c, m.period_s);
            if (std::isfinite(m.count_time_s)) s.timer_delta.assign(c, m.count_time_s);
        }
        if (frame_files && m.frames_present) s.files.frames_in_file = {m.frames_in_file};
        rec.stretches.push_back(std::move(s));
    }
    if (rec.stretches.empty()) throw Error(m.path + ": no frame is a dark, a flat or a projection");
    return rec;
}

Scan open_scan(const std::string& master) {
    const Master m = read_master(master);
    Scan scan;
    scan.master = m.path;
    scan.directory = m.directory;
    scan.name = std::filesystem::path(m.path).stem().string();
    scan.rows = m.rows, scan.cols = m.cols;
    scan.dtype = m.dtype;
    scan.detector = m.detector;
    scan.sample_name = m.title;
    scan.start_time = m.start_time, scan.end_time = m.end_time;

    std::vector<const Run*> projections, darks;
    std::vector<std::size_t> flat_after;  // per flat series: how many projection frames precede it
    std::size_t n_proj = 0;
    int n_of[3] = {0, 0, 0};
    for (const Run& run : m.runs) {
        const Role role = role_of(run.key);
        if (role == Role::Unknown) continue;
        scan.entries.push_back({run_name(role, ++n_of[run.key]), role == Role::Dark ? "dark" : role == Role::Flat ? "flat" : "proj", m.title, run.count});
        const auto a = static_cast<std::ptrdiff_t>(run.first), b = static_cast<std::ptrdiff_t>(run.first + run.count);
        if (role == Role::Projection) {
            projections.push_back(&run);
            scan.angles_deg.insert(scan.angles_deg.end(), m.angle_deg.begin() + a, m.angle_deg.begin() + b);
            scan.control_A.insert(scan.control_A.end(), m.current_A.begin() + a, m.current_A.begin() + b);
            n_proj += static_cast<std::size_t>(run.count);
        } else if (role == Role::Dark) {
            darks.push_back(&run);
            scan.darks_control_A.insert(scan.darks_control_A.end(), m.current_A.begin() + a, m.current_A.begin() + b);
        } else {
            scan.flats.push_back(frames_of(m, {&run}));
            scan.flats_control_A.emplace_back(m.current_A.begin() + a, m.current_A.begin() + b);
            flat_after.push_back(n_proj);
        }
    }
    if (projections.empty()) throw Error(m.path + ": the record holds no projection");
    if (darks.empty() || scan.flats.empty()) throw Error(m.path + ": the record holds no dark or no flat frames");
    scan.projections = frames_of(m, projections);
    scan.darks = frames_of(m, darks);

    // the arc swept during an exposure, and the angle at its middle
    if (std::isfinite(m.period_s) && std::isfinite(m.count_time_s) && m.period_s > 0) {
        const double arc = m.count_time_s / m.period_s;
        if (arc > 0 && arc <= 1) scan.exposure_arc = arc;
    }
    const double half_arc = scan.exposure_arc ? 0.5 * *scan.exposure_arc * m.step_deg : 0.0;
    for (double& a : scan.angles_deg) a += half_arc;
    scan.unwrapped_angles_deg = unwrap_360(scan.angles_deg);
    // a flat series belongs to the projection recorded next to it: the one after it, or the last
    for (const std::size_t after : flat_after) scan.flats_unwrapped_angles_deg.push_back(scan.unwrapped_angles_deg[std::min(after, n_proj - 1)]);

    scan.pixel_size_m = m.pixel_um * 1e-6;
    scan.z_translation_m.assign(n_proj, m.height() * 1e-3);
    const double y = m.across_beam(Role::Projection);
    scan.y_translation_m = std::isfinite(y) ? y * 1e-3 : 0.0;
    scan.estimated_cor_from_motor_pix = scan.y_translation_m / scan.pixel_size_m;
    scan.energy_kev = m.energy_kev;
    scan.distance_m = m.propagation_m;
    if (std::isfinite(m.count_time_s)) scan.count_time_s = m.count_time_s;
    return scan;
}

}  // namespace helirec::calibration::nxtomo
