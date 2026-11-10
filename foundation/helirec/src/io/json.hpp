#pragma once
// A JSON value and its parser: what reading Zarr metadata (.zarray, .zattrs) needs.
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "helirec/common.hpp"

namespace helirec {

struct Json {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;
    bool boolean = false;
    double number = 0;
    std::string string;
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;

    // the member `key` of an object (null: absent, or not an object)
    const Json* find(const std::string& key) const {
        if (type == Object)
            for (const auto& [k, v] : object)
                if (k == key) return &v;
        return nullptr;
    }
    bool is(Type t) const { return type == t; }

    static Json parse(const std::string& text) {
        std::size_t i = 0;
        Json v = value(text, i);
        space(text, i);
        if (i != text.size()) throw Error("JSON: text after the value");
        return v;
    }

  private:
    static void space(const std::string& t, std::size_t& i) {
        while (i < t.size() && (t[i] == ' ' || t[i] == '\t' || t[i] == '\n' || t[i] == '\r')) ++i;
    }
    static std::string text_of(const std::string& t, std::size_t& i) {
        std::string s;
        for (++i; i < t.size() && t[i] != '"'; ++i) {
            if (t[i] != '\\') { s += t[i]; continue; }
            if (++i >= t.size()) break;
            switch (t[i]) {
                case 'n': s += '\n'; break;
                case 't': s += '\t'; break;
                case 'r': s += '\r'; break;
                case 'b': s += '\b'; break;
                case 'f': s += '\f'; break;
                case 'u': {   // (code points beyond Latin-1 as UTF-8; surrogate pairs are not joined)
                    if (i + 4 >= t.size()) throw Error("JSON: bad \\u escape");
                    const unsigned c = static_cast<unsigned>(std::strtoul(t.substr(i + 1, 4).c_str(), nullptr, 16));
                    i += 4;
                    if (c < 0x80) s += static_cast<char>(c);
                    else if (c < 0x800) { s += static_cast<char>(0xC0 | (c >> 6)); s += static_cast<char>(0x80 | (c & 0x3F)); }
                    else { s += static_cast<char>(0xE0 | (c >> 12)); s += static_cast<char>(0x80 | ((c >> 6) & 0x3F)); s += static_cast<char>(0x80 | (c & 0x3F)); }
                    break;
                }
                default: s += t[i];
            }
        }
        if (i >= t.size()) throw Error("JSON: unterminated string");
        ++i;
        return s;
    }
    static Json value(const std::string& t, std::size_t& i) {
        space(t, i);
        if (i >= t.size()) throw Error("JSON: unexpected end");
        Json v;
        const char c = t[i];
        if (c == '{') {
            v.type = Object;
            ++i;
            space(t, i);
            if (i < t.size() && t[i] == '}') { ++i; return v; }
            for (;;) {
                space(t, i);
                if (i >= t.size() || t[i] != '"') throw Error("JSON: object key expected");
                std::string key = text_of(t, i);
                space(t, i);
                if (i >= t.size() || t[i] != ':') throw Error("JSON: ':' expected");
                ++i;
                v.object.emplace_back(std::move(key), value(t, i));
                space(t, i);
                if (i < t.size() && t[i] == ',') { ++i; continue; }
                if (i < t.size() && t[i] == '}') { ++i; return v; }
                throw Error("JSON: ',' or '}' expected");
            }
        }
        if (c == '[') {
            v.type = Array;
            ++i;
            space(t, i);
            if (i < t.size() && t[i] == ']') { ++i; return v; }
            for (;;) {
                v.array.push_back(value(t, i));
                space(t, i);
                if (i < t.size() && t[i] == ',') { ++i; continue; }
                if (i < t.size() && t[i] == ']') { ++i; return v; }
                throw Error("JSON: ',' or ']' expected");
            }
        }
        if (c == '"') {
            v.type = String;
            v.string = text_of(t, i);
            return v;
        }
        if (t.compare(i, 4, "null") == 0) { i += 4; return v; }
        if (t.compare(i, 4, "true") == 0) { i += 4; v.type = Bool; v.boolean = true; return v; }
        if (t.compare(i, 5, "false") == 0) { i += 5; v.type = Bool; return v; }
        char* end = nullptr;
        v.number = std::strtod(t.c_str() + i, &end);   // (also NaN / Infinity, as NumPy-written metadata may hold)
        if (end == t.c_str() + i) throw Error("JSON: unexpected character");
        v.type = Number;
        i = static_cast<std::size_t>(end - t.c_str());
        return v;
    }
};

}  // namespace helirec
