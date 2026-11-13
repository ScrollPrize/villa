#pragma once
// Reading values out of an HDF5 file with the C library: what the acquisition record needs (record.cpp).
// libhdf5 (serial build) is not thread safe: everything here is called from one thread.
#include <hdf5.h>
#include <sys/types.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "helirec/common.hpp"

namespace helirec::calibration::h5 {

struct Id {
    hid_t id = -1;
    herr_t (*close)(hid_t) = nullptr;
    Id() = default;
    Id(hid_t i, herr_t (*c)(hid_t)) : id(i), close(c) {}
    Id(Id&& o) noexcept : id(o.id), close(o.close) { o.id = -1; }
    Id& operator=(Id&& o) noexcept {
        if (this != &o) {
            reset();
            id = o.id;
            close = o.close;
            o.id = -1;
        }
        return *this;
    }
    Id(const Id&) = delete;
    Id& operator=(const Id&) = delete;
    ~Id() { reset(); }
    void reset() {
        if (id >= 0 && close) close(id);
        id = -1;
    }
    explicit operator bool() const { return id >= 0; }
};

inline void quiet() {
    static std::once_flag once;
    std::call_once(once, [] { H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); });
}

inline Id open_file(const std::string& path) {
    quiet();
    Id f(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (!f) throw Error("cannot open HDF5 file " + path);
    return f;
}

// true when `path` (relative to `loc`) names an object; every component is tested, so a missing parent is "false"
inline bool exists(hid_t loc, const std::string& path) {
    std::size_t i = 0;
    while (i < path.size()) {
        std::size_t j = path.find('/', i);
        if (j == std::string::npos) j = path.size();
        if (j > i) {
            const std::string part = path.substr(0, j);
            if (H5Lexists(loc, part.c_str(), H5P_DEFAULT) <= 0) return false;
        }
        i = j + 1;
    }
    return H5Oexists_by_name(loc, path.c_str(), H5P_DEFAULT) > 0;
}

inline bool is_dataset(hid_t loc, const std::string& path) {
    if (!exists(loc, path)) return false;
    H5O_info2_t info;
    if (H5Oget_info_by_name3(loc, path.c_str(), &info, H5O_INFO_BASIC, H5P_DEFAULT) < 0) return false;
    return info.type == H5O_TYPE_DATASET;
}

inline std::vector<std::string> children(hid_t loc, const std::string& path) {
    std::vector<std::string> names;
    Id g(H5Gopen2(loc, path.c_str(), H5P_DEFAULT), H5Gclose);
    if (!g) return names;
    auto cb = [](hid_t, const char* name, const H5L_info2_t*, void* data) -> herr_t {
        static_cast<std::vector<std::string>*>(data)->push_back(name);
        return 0;
    };
    H5Literate2(g.id, H5_INDEX_NAME, H5_ITER_INC, nullptr, cb, &names);
    return names;
}

struct Shape {
    std::vector<hsize_t> dims;
    H5T_class_t type_class = H5T_NO_CLASS;
    std::size_t type_size = 0;
    hsize_t count() const {
        hsize_t n = 1;
        for (hsize_t d : dims) n *= d;
        return n;
    }
};

inline Shape shape(hid_t loc, const std::string& path) {
    Id d(H5Dopen2(loc, path.c_str(), H5P_DEFAULT), H5Dclose);
    if (!d) throw Error("no dataset " + path);
    Id s(H5Dget_space(d.id), H5Sclose);
    Id t(H5Dget_type(d.id), H5Tclose);
    Shape out;
    const int nd = H5Sget_simple_extent_ndims(s.id);
    if (nd > 0) {
        out.dims.resize(nd);
        H5Sget_simple_extent_dims(s.id, out.dims.data(), nullptr);
    }
    if (H5Sget_simple_extent_type(s.id) == H5S_NULL) out.dims.assign(1, 0);
    out.type_class = H5Tget_class(t.id);
    out.type_size = H5Tget_size(t.id);
    return out;
}

// integer, float and enum (h5py bool) datasets of any shape, flattened, as double
inline std::vector<double> read_doubles(hid_t loc, const std::string& path) {
    Id d(H5Dopen2(loc, path.c_str(), H5P_DEFAULT), H5Dclose);
    if (!d) throw Error("no dataset " + path);
    Id s(H5Dget_space(d.id), H5Sclose);
    Id t(H5Dget_type(d.id), H5Tclose);
    const hssize_t n = H5Sget_simple_extent_npoints(s.id);
    std::vector<double> out(n > 0 ? static_cast<std::size_t>(n) : 0);
    if (out.empty()) return out;
    const H5T_class_t cls = H5Tget_class(t.id);
    if (cls == H5T_INTEGER || cls == H5T_FLOAT) {
        if (H5Dread(d.id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0)
            throw Error("cannot read " + path);
    } else if (cls == H5T_ENUM) {
        // h5py stores bool as an enum over a one-byte integer: read the stored integers
        Id native(H5Tget_native_type(t.id, H5T_DIR_ASCEND), H5Tclose);
        const std::size_t size = H5Tget_size(native.id);
        std::vector<unsigned char> raw(out.size() * size);
        if (H5Dread(d.id, native.id, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw.data()) < 0) throw Error("cannot read " + path);
        for (std::size_t i = 0; i < out.size(); ++i) {
            const unsigned char* p = raw.data() + i * size;
            std::int64_t v = 0;
            if (size == 1) v = static_cast<std::int8_t>(p[0]);
            else if (size == 2) { std::int16_t x; std::memcpy(&x, p, 2); v = x; }
            else if (size == 4) { std::int32_t x; std::memcpy(&x, p, 4); v = x; }
            else if (size == 8) { std::memcpy(&v, p, 8); }
            else throw Error("dataset " + path + ": unsupported enum size");
            out[i] = static_cast<double>(v);
        }
    } else {
        throw Error("dataset " + path + " is not numeric");
    }
    return out;
}

inline std::vector<std::int32_t> read_int32(hid_t loc, const std::string& path) {
    Id d(H5Dopen2(loc, path.c_str(), H5P_DEFAULT), H5Dclose);
    if (!d) throw Error("no dataset " + path);
    Id s(H5Dget_space(d.id), H5Sclose);
    Id t(H5Dget_type(d.id), H5Tclose);
    if (H5Tget_class(t.id) != H5T_INTEGER || H5Tget_size(t.id) > 4)
        throw Error("dataset " + path + " is not a 32-bit integer");
    const hssize_t n = H5Sget_simple_extent_npoints(s.id);
    std::vector<std::int32_t> out(n > 0 ? static_cast<std::size_t>(n) : 0);
    if (!out.empty() && H5Dread(d.id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0)
        throw Error("cannot read " + path);
    return out;
}

// string datasets (variable or fixed length) of any shape, flattened
inline std::vector<std::string> read_strings(hid_t loc, const std::string& path) {
    Id d(H5Dopen2(loc, path.c_str(), H5P_DEFAULT), H5Dclose);
    if (!d) throw Error("no dataset " + path);
    Id s(H5Dget_space(d.id), H5Sclose);
    Id t(H5Dget_type(d.id), H5Tclose);
    if (H5Tget_class(t.id) != H5T_STRING) throw Error("dataset " + path + " is not a string");
    const hssize_t n = H5Sget_simple_extent_npoints(s.id);
    std::vector<std::string> out(n > 0 ? static_cast<std::size_t>(n) : 0);
    if (out.empty()) return out;
    if (H5Tis_variable_str(t.id) > 0) {
        Id mem(H5Tcopy(H5T_C_S1), H5Tclose);
        H5Tset_size(mem.id, H5T_VARIABLE);
        H5Tset_cset(mem.id, H5Tget_cset(t.id));
        std::vector<char*> ptr(out.size(), nullptr);
        if (H5Dread(d.id, mem.id, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr.data()) < 0) throw Error("cannot read " + path);
        for (std::size_t i = 0; i < out.size(); ++i)
            if (ptr[i]) out[i] = ptr[i];
        H5Treclaim(mem.id, s.id, H5P_DEFAULT, ptr.data());
    } else {
        const std::size_t size = H5Tget_size(t.id);
        std::vector<char> raw(out.size() * size);
        if (H5Dread(d.id, t.id, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw.data()) < 0) throw Error("cannot read " + path);
        for (std::size_t i = 0; i < out.size(); ++i) {
            const char* p = raw.data() + i * size;
            out[i].assign(p, strnlen(p, size));
        }
    }
    return out;
}

inline std::string read_string(hid_t loc, const std::string& path) {
    const auto v = read_strings(loc, path);
    return v.empty() ? std::string() : v[0];
}

inline double read_double(hid_t loc, const std::string& path) {
    const auto v = read_doubles(loc, path);
    if (v.empty()) throw Error("dataset " + path + " is empty");
    return v[0];
}

// the first value of the string attribute `name` of the object at `path`; "" when there is none or it is not text
inline std::string attribute_text(hid_t loc, const std::string& path, const std::string& name) {
    if (H5Aexists_by_name(loc, path.c_str(), name.c_str(), H5P_DEFAULT) <= 0) return std::string();
    Id a(H5Aopen_by_name(loc, path.c_str(), name.c_str(), H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    if (!a) return std::string();
    Id t(H5Aget_type(a.id), H5Tclose);
    if (H5Tget_class(t.id) != H5T_STRING) return std::string();
    Id s(H5Aget_space(a.id), H5Sclose);
    const hssize_t n = H5Sget_simple_extent_npoints(s.id);
    if (n <= 0) return std::string();
    std::string out;
    if (H5Tis_variable_str(t.id) > 0) {
        Id mem(H5Tcopy(H5T_C_S1), H5Tclose);
        H5Tset_size(mem.id, H5T_VARIABLE);
        H5Tset_cset(mem.id, H5Tget_cset(t.id));
        std::vector<char*> ptr(static_cast<std::size_t>(n), nullptr);
        if (H5Aread(a.id, mem.id, ptr.data()) < 0) return std::string();
        if (ptr[0]) out = ptr[0];
        H5Treclaim(mem.id, s.id, H5P_DEFAULT, ptr.data());
    } else {
        const std::size_t size = H5Tget_size(t.id);
        std::vector<char> raw(static_cast<std::size_t>(n) * size);
        if (H5Aread(a.id, t.id, raw.data()) < 0) return std::string();
        out.assign(raw.data(), strnlen(raw.data(), size));
    }
    return out;
}

// A dataset of frames (frames, rows, columns) and, when it is virtual, its pieces in the order of its mapping:
// frames [first, first + count) of the dataset are frames [source_first, source_first + source_count) of
// `dataset` in the file `file_name` (as the mapping names it: ".", relative to the master's directory, or
// absolute). spans_frames: the bounds of the piece in the dataset span whole frames; whole_frames: also, both
// sides select every pixel of their frames (no strides or gaps).
struct VirtualPiece {
    std::string file_name, dataset;
    std::int64_t first = 0, count = 0, source_first = 0, source_count = 0;
    bool spans_frames = true, whole_frames = true;
};
struct FrameDataset {
    std::int64_t frames = 0;
    int rows = 0, cols = 0;
    bool is_virtual = false;
    std::vector<VirtualPiece> pieces;
};

inline std::string virtual_name(hid_t dcpl, std::size_t i, ssize_t (*get)(hid_t, size_t, char*, size_t)) {
    const ssize_t len = get(dcpl, i, nullptr, 0);
    if (len < 0) throw Error("cannot read a virtual dataset mapping");
    std::string s(static_cast<std::size_t>(len) + 1, '\0');
    get(dcpl, i, s.data(), s.size());
    s.resize(static_cast<std::size_t>(len));
    return s;
}

inline FrameDataset frame_dataset(hid_t loc, const std::string& dataset) {
    Id d(H5Dopen2(loc, dataset.c_str(), H5P_DEFAULT), H5Dclose);
    if (!d) throw Error("no dataset " + dataset);
    Id space(H5Dget_space(d.id), H5Sclose);
    hsize_t dims[3] = {0, 0, 0};
    if (H5Sget_simple_extent_ndims(space.id) != 3) throw Error(dataset + " is not (frames, rows, columns)");
    H5Sget_simple_extent_dims(space.id, dims, nullptr);
    FrameDataset out;
    out.frames = static_cast<std::int64_t>(dims[0]);
    out.rows = static_cast<int>(dims[1]);
    out.cols = static_cast<int>(dims[2]);
    Id dcpl(H5Dget_create_plist(d.id), H5Pclose);
    if (H5Pget_layout(dcpl.id) != H5D_VIRTUAL) return out;
    out.is_virtual = true;
    std::size_t n = 0;
    H5Pget_virtual_count(dcpl.id, &n);
    for (std::size_t i = 0; i < n; ++i) {
        VirtualPiece p;
        p.file_name = virtual_name(dcpl.id, i, H5Pget_virtual_filename);
        p.dataset = virtual_name(dcpl.id, i, H5Pget_virtual_dsetname);
        Id vs(H5Pget_virtual_vspace(dcpl.id, i), H5Sclose);
        hsize_t a[3], b[3];
        if (H5Sget_select_bounds(vs.id, a, b) < 0) throw Error(dataset + ": unreadable virtual mapping");
        p.spans_frames = a[1] == 0 && a[2] == 0 && b[1] + 1 == dims[1] && b[2] + 1 == dims[2];
        p.whole_frames = p.spans_frames &&
                         static_cast<hsize_t>(H5Sget_select_npoints(vs.id)) == (b[0] - a[0] + 1) * dims[1] * dims[2];
        p.first = static_cast<std::int64_t>(a[0]);
        p.count = static_cast<std::int64_t>(b[0] - a[0] + 1);
        Id ss(H5Pget_virtual_srcspace(dcpl.id, i), H5Sclose);
        hsize_t sa[3] = {0, 0, 0}, sb[3] = {0, 0, 0};
        if (H5Sget_simple_extent_ndims(ss.id) == 3 && H5Sget_select_bounds(ss.id, sa, sb) >= 0) {
            p.source_first = static_cast<std::int64_t>(sa[0]);
            p.source_count = static_cast<std::int64_t>(sb[0] - sa[0] + 1);
            hsize_t sd[3] = {0, 0, 0};
            H5Sget_simple_extent_dims(ss.id, sd, nullptr);
            p.whole_frames = p.whole_frames && sa[1] == 0 && sa[2] == 0 &&
                             static_cast<hsize_t>(H5Sget_select_npoints(ss.id)) == static_cast<hsize_t>(p.source_count) * dims[1] * dims[2];
        } else {
            p.source_count = p.count;
        }
        out.pieces.push_back(std::move(p));
    }
    return out;
}

}  // namespace helirec::calibration::h5
