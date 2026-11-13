#pragma once
// A file of float32 arrays, the form of the calibrations' caches. Its first line is JSON,
//     {"arrays": [{"name": "dark", "shape": [1200, 2528], "offset": 0}, ...], ...the writer's own members...}
// and the arrays' values follow it, float32 little-endian, each at its offset in bytes from the end of that line.
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "helirec/common.hpp"
#include "io/json.hpp"

namespace helirec {

struct NamedArray {
    std::string name;
    std::vector<std::size_t> shape;
    std::vector<float> values;
};

struct ArrayFile {
    Json header;  // the first line, "arrays" included
    std::vector<NamedArray> arrays;
    const NamedArray* find(const std::string& name) const {
        for (const NamedArray& a : arrays)
            if (a.name == name) return &a;
        return nullptr;
    }
};

namespace array_file_detail {
inline bool little_endian() {
    const std::uint32_t one = 1;
    unsigned char b;
    std::memcpy(&b, &one, 1);
    return b == 1;
}
inline void swap_bytes(std::vector<float>& v) {
    for (float& x : v) {
        unsigned char b[4];
        std::memcpy(b, &x, 4);
        std::swap(b[0], b[3]), std::swap(b[1], b[2]);
        std::memcpy(&x, b, 4);
    }
}
}  // namespace array_file_detail

// `header`: an object of the writer's members ("arrays" is added). Written to a temporary file and renamed into
// place, so that no reader sees a part of it.
inline void write_array_file(const std::string& path, Json header, const std::vector<NamedArray>& arrays) {
    Json list = Json::list();
    std::uint64_t offset = 0;
    for (const NamedArray& a : arrays) {
        std::size_t n = 1;
        Json shape = Json::list();
        for (std::size_t d : a.shape) n *= d, shape.add(Json::whole_number(static_cast<long long>(d)));
        if (n != a.values.size()) throw Error("array " + a.name + ": its values do not fill its shape");
        list.add(Json::dict().set("name", Json::of(a.name)).set("shape", shape).set("offset", Json::whole_number(static_cast<long long>(offset))));
        offset += 4 * n;
    }
    header.set("arrays", list);
    const std::string tmp = path + ".tmp" + std::to_string(static_cast<long>(::getpid()));
    {
        std::ofstream o(tmp, std::ios::binary);
        if (!o) throw Error("cannot write " + tmp);
        o << header.text() << "\n";
        for (const NamedArray& a : arrays) {
            if (array_file_detail::little_endian()) {
                o.write(reinterpret_cast<const char*>(a.values.data()), static_cast<std::streamsize>(4 * a.values.size()));
            } else {
                std::vector<float> v = a.values;
                array_file_detail::swap_bytes(v);
                o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(4 * v.size()));
            }
        }
        if (!o) {
            std::remove(tmp.c_str());
            throw Error("cannot write " + tmp);
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw Error("cannot write " + path);
    }
}

// values: false reads the first line alone (the arrays come back without values)
inline ArrayFile read_array_file(const std::string& path, bool values = true) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error("cannot read " + path);
    std::string line;
    if (!std::getline(in, line)) throw Error(path + ": no header");
    ArrayFile out;
    out.header = Json::parse(line);
    const Json* list = out.header.find("arrays");
    if (!list || !list->is(Json::Array)) throw Error(path + ": the header lists no arrays");
    const std::streamoff start = in.tellg();
    for (const Json& e : list->array) {
        const Json* name = e.find("name");
        const Json* shape = e.find("shape");
        const Json* offset = e.find("offset");
        if (!name || !shape || !offset || !name->is(Json::String) || !shape->is(Json::Array) || !offset->is(Json::Number))
            throw Error(path + ": an array without name, shape or offset");
        NamedArray a;
        a.name = name->string;
        std::size_t n = 1;
        for (const Json& d : shape->array) a.shape.push_back(static_cast<std::size_t>(d.number)), n *= a.shape.back();
        if (values) {
            a.values.resize(n);
            in.seekg(start + static_cast<std::streamoff>(offset->number));
            in.read(reinterpret_cast<char*>(a.values.data()), static_cast<std::streamsize>(4 * n));
            if (!in) throw Error(path + ": array " + a.name + " is cut short");
            if (!array_file_detail::little_endian()) array_file_detail::swap_bytes(a.values);
        }
        out.arrays.push_back(std::move(a));
    }
    return out;
}

}  // namespace helirec
