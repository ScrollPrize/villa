#include "helirec/calibration/record.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>

#include "calibration/h5_read.hpp"

namespace helirec::calibration {

const char* role_name(Role r) {
    switch (r) {
        case Role::Dark: return "dark";
        case Role::Flat: return "flat";
        case Role::Projection: return "projection";
        default: return "unknown";
    }
}

bool Values::has_number(const std::string& key) const {
    const auto it = items.find(key);
    return it != items.end() && !it->second.num.empty();
}
double Values::number(const std::string& key) const {
    const auto it = items.find(key);
    if (it == items.end() || it->second.num.empty()) throw Error("the record holds no number '" + key + "'");
    return it->second.num[0];
}
double Values::number(const std::string& key, double fallback) const {
    const auto it = items.find(key);
    return it == items.end() || it->second.num.empty() ? fallback : it->second.num[0];
}
std::string Values::text(const std::string& key) const {
    const auto it = items.find(key);
    return it == items.end() || it->second.str.empty() ? std::string() : it->second.str[0];
}
const std::vector<double>& Values::numbers(const std::string& key) const {
    static const std::vector<double> none;
    const auto it = items.find(key);
    return it == items.end() ? none : it->second.num;
}
const std::vector<std::string>& Values::texts(const std::string& key) const {
    static const std::vector<std::string> none;
    const auto it = items.find(key);
    return it == items.end() ? none : it->second.str;
}

std::vector<int> Record::stretches_with(Role role) const {
    std::vector<int> out;
    for (std::size_t i = 0; i < stretches.size(); ++i)
        if (stretches[i].role == role) out.push_back(static_cast<int>(i));
    return out;
}
std::vector<std::pair<std::int64_t, std::int64_t>> Stretch::frames_on_disk() const {
    std::vector<std::pair<std::int64_t, std::int64_t>> out;
    for (const FramePiece& p : pieces) {
        if (!p.present || p.count <= 0) continue;
        if (!out.empty() && out.back().second == p.first) out.back().second = p.first + p.count;
        else out.push_back({p.first, p.first + p.count});
    }
    return out;
}
std::int64_t Stretch::n_on_disk() const {
    std::int64_t n = 0;
    for (const FramePiece& p : pieces)
        if (p.present) n += p.count;
    return n;
}
bool Stretch::on_disk(std::int64_t frame) const {
    auto it = std::upper_bound(pieces.begin(), pieces.end(), frame, [](std::int64_t f, const FramePiece& p) { return f < p.first; });
    if (it == pieces.begin()) return false;
    --it;
    return it->present && frame < it->first + it->count;
}

const Stretch& Record::stretch(const std::string& entry) const {
    for (const Stretch& s : stretches)
        if (s.entry == entry) return s;
    throw Error("the record has no stretch " + entry);
}

namespace {

constexpr hsize_t kSmall = 16;  // largest array kept as a "value"

// every scalar / small array below `path`, recursively; `skip` names child groups that are not entered
void collect(hid_t loc, const std::string& path, const std::string& prefix, Values& out,
             const std::vector<std::string>& skip = {}) {
    for (const std::string& name : h5::children(loc, path)) {
        if (std::find(skip.begin(), skip.end(), name) != skip.end()) continue;
        const std::string child = path + "/" + name;
        const std::string key = prefix.empty() ? name : prefix + "/" + name;
        if (!h5::exists(loc, child)) continue;  // dangling link
        if (h5::is_dataset(loc, child)) {
            const h5::Shape sh = h5::shape(loc, child);
            if (sh.count() == 0 || sh.count() > kSmall) continue;
            Value v;
            if (sh.type_class == H5T_STRING) v.str = h5::read_strings(loc, child);
            else if (sh.type_class == H5T_INTEGER || sh.type_class == H5T_FLOAT || sh.type_class == H5T_ENUM)
                v.num = h5::read_doubles(loc, child);
            else continue;
            out.items[key] = std::move(v);
        } else {
            collect(loc, child, key, out, skip);
        }
    }
}

std::vector<double> doubles_if(hid_t loc, const std::string& path) {
    return h5::is_dataset(loc, path) ? h5::read_doubles(loc, path) : std::vector<double>();
}
std::vector<std::int32_t> int32_if(hid_t loc, const std::string& path) {
    return h5::is_dataset(loc, path) ? h5::read_int32(loc, path) : std::vector<std::int32_t>();
}
std::string string_if(hid_t loc, const std::string& path) {
    return h5::is_dataset(loc, path) ? h5::read_string(loc, path) : std::string();
}

std::map<std::string, double> positioners(hid_t loc, const std::string& path) {
    std::map<std::string, double> out;
    for (const std::string& name : h5::children(loc, path)) {
        const std::string child = path + "/" + name;
        if (!h5::is_dataset(loc, child)) continue;
        const h5::Shape sh = h5::shape(loc, child);
        if (sh.count() != 1 || (sh.type_class != H5T_INTEGER && sh.type_class != H5T_FLOAT)) continue;
        out[name] = h5::read_double(loc, child);
    }
    return out;
}

std::string directory_of(const std::string& path) {
    const std::size_t i = path.find_last_of('/');
    return i == std::string::npos ? std::string(".") : path.substr(0, i);
}

std::string virtual_name(hid_t dcpl, std::size_t i, ssize_t (*get)(hid_t, size_t, char*, size_t)) {
    const ssize_t len = get(dcpl, i, nullptr, 0);
    if (len < 0) throw Error("cannot read a virtual dataset mapping");
    std::string s(static_cast<std::size_t>(len) + 1, '\0');
    get(dcpl, i, s.data(), s.size());
    s.resize(static_cast<std::size_t>(len));
    return s;
}

// where the frames of `dataset` are stored: the pieces of a virtual dataset, or the dataset itself
void frame_pieces(hid_t file, const std::string& file_path, const std::string& dataset, Stretch& s) {
    h5::Id d(H5Dopen2(file, dataset.c_str(), H5P_DEFAULT), H5Dclose);
    if (!d) throw Error("no dataset " + dataset);
    h5::Id space(H5Dget_space(d.id), H5Sclose);
    hsize_t dims[3] = {0, 0, 0};
    if (H5Sget_simple_extent_ndims(space.id) != 3) throw Error(dataset + " is not (frames, rows, columns)");
    H5Sget_simple_extent_dims(space.id, dims, nullptr);
    s.n_frames = static_cast<std::int64_t>(dims[0]);
    s.rows = static_cast<int>(dims[1]);
    s.cols = static_cast<int>(dims[2]);
    h5::Id dcpl(H5Dget_create_plist(d.id), H5Pclose);
    if (H5Pget_layout(dcpl.id) != H5D_VIRTUAL) {
        s.pieces.push_back({file_path, dataset, 0, s.n_frames, 0, true});
        return;
    }
    std::size_t n = 0;
    H5Pget_virtual_count(dcpl.id, &n);
    const std::string dir = directory_of(file_path);
    for (std::size_t i = 0; i < n; ++i) {
        FramePiece p;
        const std::string name = virtual_name(dcpl.id, i, H5Pget_virtual_filename);
        p.file = name == "." ? file_path : (!name.empty() && name[0] == '/' ? name : dir + "/" + name);
        p.dataset = virtual_name(dcpl.id, i, H5Pget_virtual_dsetname);
        h5::Id vs(H5Pget_virtual_vspace(dcpl.id, i), H5Sclose);
        hsize_t a[3], b[3];
        if (H5Sget_select_bounds(vs.id, a, b) < 0) throw Error(dataset + ": unreadable virtual mapping");
        if (a[1] != 0 || a[2] != 0 || b[1] + 1 != dims[1] || b[2] + 1 != dims[2])
            throw Error(dataset + ": a virtual piece does not hold whole frames");
        p.first = static_cast<std::int64_t>(a[0]);
        p.count = static_cast<std::int64_t>(b[0] - a[0] + 1);
        h5::Id ss(H5Pget_virtual_srcspace(dcpl.id, i), H5Sclose);
        hsize_t sa[3] = {0, 0, 0}, sb[3] = {0, 0, 0};
        if (H5Sget_simple_extent_ndims(ss.id) == 3 && H5Sget_select_bounds(ss.id, sa, sb) >= 0)
            p.source_first = static_cast<std::int64_t>(sa[0]);
        struct stat st;
        p.present = ::stat(p.file.c_str(), &st) == 0 && S_ISREG(st.st_mode);
        s.pieces.push_back(std::move(p));
    }
    std::sort(s.pieces.begin(), s.pieces.end(), [](const FramePiece& x, const FramePiece& y) { return x.first < y.first; });
}

void read_monitor(hid_t file, const std::string& entry, Monitor& m) {
    const std::string i = entry + "/instrument/";
    if (!h5::is_dataset(file, i + "current/signal")) return;
    m.entry = entry;
    m.current = doubles_if(file, i + "current/signal");
    m.lifetime = doubles_if(file, i + "lifetime/signal");
    m.refill = doubles_if(file, i + "refill/signal");
    m.sbcurr = doubles_if(file, i + "sbcurr/signal");
    m.epoch = doubles_if(file, i + "epoch/value");
    m.elapsed = doubles_if(file, i + "elapsed_time/value");
}

void read_frame_files(const std::string& detector, Stretch& s) {
    std::string last;
    for (const FramePiece& p : s.pieces) {
        if (p.file == last || !p.present) continue;
        last = p.file;
        h5::Id f = h5::open_file(p.file);
        // the detector's group is the parent of the frame dataset
        const std::size_t cut = p.dataset.find_last_of('/');
        const std::string group = cut == std::string::npos ? std::string("/") : p.dataset.substr(0, cut);
        if (s.files.frames_in_file.empty()) collect(f.id, group, "", s.files.header);
        const h5::Shape sh = h5::shape(f.id, p.dataset);
        s.files.frames_in_file.push_back(sh.dims.empty() ? 0 : static_cast<std::int64_t>(sh.dims[0]));
        const auto t = doubles_if(f.id, group + "/time_of_frame");
        s.files.time_of_frame.insert(s.files.time_of_frame.end(), t.begin(), t.end());
    }
    (void)detector;
}

std::pair<int, int> entry_number(const std::string& name) {
    char* end = nullptr;
    const long a = std::strtol(name.c_str(), &end, 10);
    const long b = (end && *end == '.') ? std::strtol(end + 1, nullptr, 10) : 0;
    return {static_cast<int>(a), static_cast<int>(b)};
}

}  // namespace

Record read_record(const std::string& h5_path, bool frame_files) {
    h5::Id f = h5::open_file(h5_path);
    Record rec;
    rec.path = h5_path;
    rec.directory = directory_of(h5_path);

    std::vector<std::string> entries = h5::children(f.id, "/");
    std::sort(entries.begin(), entries.end(),
              [](const std::string& a, const std::string& b) { return entry_number(a) < entry_number(b); });
    if (entries.empty()) throw Error(h5_path + ": no entries");
    const std::string head = entries.front();
    if (!h5::exists(f.id, head + "/technique")) throw Error(h5_path + ": entry " + head + " describes no tomo sequence");

    collect(f.id, head + "/technique", "", rec.technique);
    collect(f.id, head + "/instrument", "", rec.instrument, {"positioners", "positioners_start", "positioners_end"});
    rec.start_time = string_if(f.id, head + "/start_time");
    rec.end_time = string_if(f.id, head + "/end_time");
    rec.end_reason = string_if(f.id, head + "/end_reason");
    rec.sample_name = string_if(f.id, head + "/sample/name");
    {
        const std::size_t i = rec.sample_name.find_last_of('_');
        rec.label = i == std::string::npos ? rec.sample_name : rec.sample_name.substr(i + 1);
    }
    const auto first_of = [&](const std::string& key) {
        const auto& v = rec.technique.texts(key);
        if (v.empty()) throw Error(h5_path + ": the record does not name '" + key + "'");
        return v;
    };
    rec.detector = first_of("tomoconfig/detector")[0];
    const auto pair = [&](const std::string& key, std::string out[2]) {
        const auto& v = first_of(key);
        out[0] = v[0];
        out[1] = v.size() > 1 ? v[1] : v[0];
    };
    pair("tomoconfig/rotation", rec.rotation);
    pair("tomoconfig/translation_z", rec.translation_z);
    pair("tomoconfig/translation_y", rec.translation_y);

    for (const std::string& entry : entries) {
        const std::string image = entry + "/instrument/" + rec.detector + "/image";
        if (!h5::is_dataset(f.id, image)) continue;
        Stretch s;
        s.entry = entry;
        s.title = string_if(f.id, entry + "/title");
        s.start_time = string_if(f.id, entry + "/start_time");
        s.end_time = string_if(f.id, entry + "/end_time");
        s.end_reason = string_if(f.id, entry + "/end_reason");
        collect(f.id, entry + "/technique", "", s.technique);
        if (s.technique.has_number("image_key")) {
            s.has_image_key = true;
            s.image_key = static_cast<int>(s.technique.number("image_key"));
        }
        // the role the sequence itself gives the stretch: technique/subscans/scan<k>/type of entry 1.1, in order
        {
            const std::string type = rec.technique.text("subscans/scan" + std::to_string(rec.stretches.size()) + "/type");
            if (type == "tomo:dark") s.role = Role::Dark;
            else if (type == "tomo:flat") s.role = Role::Flat;
            else if (type.rfind("tomo:", 0) == 0) s.role = Role::Projection;
        }
        frame_pieces(f.id, h5_path, image, s);

        const std::string i = entry + "/instrument/";
        s.timer_raw = int32_if(f.id, i + "timer_raw/data");
        s.epoch_trig = doubles_if(f.id, i + "epoch_trig/data");
        s.timer_trig = doubles_if(f.id, i + "timer_trig/data");
        s.timer_delta = doubles_if(f.id, i + "timer_delta/data");
        s.timer_period = doubles_if(f.id, i + "timer_period/data");
        const std::string& rot = rec.rotation[0];
        s.rot_raw = int32_if(f.id, i + rot + "_raw/data");
        s.rot_trig = doubles_if(f.id, i + rot + "_trig/data");
        s.rot_delta = doubles_if(f.id, i + rot + "_delta/data");
        s.rot_period = doubles_if(f.id, i + rot + "_period/data");
        for (const char* suffix : {"_center", "_cen360"}) {
            if (!h5::is_dataset(f.id, i + rot + suffix + "/data")) continue;
            s.rot_center_name = rot + suffix;
            s.rot_center = h5::read_doubles(f.id, i + rot + suffix + "/data");
            break;
        }
        const std::string& z = rec.translation_z[0];
        s.z_raw = int32_if(f.id, i + z + "_raw/data");
        s.z_trig = doubles_if(f.id, i + z + "_trig/data");
        s.z_center = doubles_if(f.id, i + z + "_center/data");
        s.z_delta = doubles_if(f.id, i + z + "_delta/data");
        s.z_period = doubles_if(f.id, i + z + "_period/data");
        s.epoch = doubles_if(f.id, i + "epoch/value");
        s.elapsed = doubles_if(f.id, i + "elapsed_time/value");

        const std::string stat = i + rec.detector + "_full/";
        s.stat_avg = doubles_if(f.id, stat + "avg");
        s.stat_min = doubles_if(f.id, stat + "min");
        s.stat_max = doubles_if(f.id, stat + "max");
        s.stat_std = doubles_if(f.id, stat + "std");
        s.stat_sum = doubles_if(f.id, stat + "sum");
        if (h5::is_dataset(f.id, stat + "selection/x")) {
            s.stat_x = static_cast<int>(h5::read_double(f.id, stat + "selection/x"));
            s.stat_y = static_cast<int>(h5::read_double(f.id, stat + "selection/y"));
            s.stat_w = static_cast<int>(h5::read_double(f.id, stat + "selection/width"));
            s.stat_h = static_cast<int>(h5::read_double(f.id, stat + "selection/height"));
        }
        s.positioners_start = positioners(f.id, i + "positioners_start");
        s.positioners_end = positioners(f.id, i + "positioners_end");
        collect(f.id, i + "fscan_parameters", "", s.scan_parameters);
        collect(f.id, i + "machine", "", s.machine);

        // the monitors of a projection stretch N.1 are in N.2; darks and flats carry their own
        const auto [major, minor] = entry_number(entry);
        const std::string companion = std::to_string(major) + "." + std::to_string(minor + 1);
        if (h5::exists(f.id, companion)) read_monitor(f.id, companion, s.monitor);
        if (s.monitor.entry.empty()) read_monitor(f.id, entry, s.monitor);

        if (frame_files) read_frame_files(rec.detector, s);
        rec.stretches.push_back(std::move(s));
    }
    if (rec.stretches.empty()) throw Error(h5_path + ": no stretch holds frames of detector " + rec.detector);
    return rec;
}

}  // namespace helirec::calibration
