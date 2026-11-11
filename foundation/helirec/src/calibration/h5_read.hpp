#pragma once
// Reading values out of an HDF5 file with the C library: what the acquisition record needs (record.cpp).
// libhdf5 (serial build) is not thread safe: everything here is called from one thread.
#include <hdf5.h>

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

}  // namespace helirec::calibration::h5
