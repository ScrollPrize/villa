#pragma once
// The command lines of helirec's tools: options named as Python's argparse names them (--name, dashes or underscores),
// typed values, a positional argument, and a --config JSON file of options that the command line overrides.
#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "io/json.hpp"

namespace helirec::command_line {

// a reason to stop with a message ("help": print the usage)
struct Exit {
    std::string message;
};
[[noreturn]] inline void fail(const std::string& m) { throw Exit{m}; }

// name, number of values (0: a flag; -1: one or more; -2: repeatable, one each), kind ('i' whole, 'f' real, 's' text, 'b' flag)
struct Option {
    const char* name;
    int nargs;
    char kind;
};
struct Spec {
    std::vector<Option> options;
    std::map<std::string, std::string> short_names;  // "-n" -> "n_slices"
    std::string positional;                          // the name the positional arguments go to
    bool positional_many = false;                    // a list of them, else one
    const Option* find(const std::string& name) const {
        for (const auto& o : options)
            if (name == o.name) return &o;
        return nullptr;
    }
};

inline void put(Json& j, const std::string& key, Json v) {
    for (auto& kv : j.object)
        if (kv.first == key) return void(kv.second = std::move(v));
    j.object.emplace_back(key, std::move(v));
}

inline Json value_of(char kind, const std::string& t, const std::string& name) {
    try {
        std::size_t used = 0;
        if (kind == 'i') {
            const long long v = std::stoll(t, &used);
            if (used != t.size()) throw std::invalid_argument(t);
            return Json::whole_number(v);
        }
        if (kind == 'f') {
            const double v = std::stod(t, &used);
            if (used != t.size()) throw std::invalid_argument(t);
            return Json::of(v);
        }
    } catch (const std::exception&) {
        fail("--" + name + ": invalid value '" + t + "'");
    }
    return Json::of(t);
}

// the options given on the command line, as JSON values of their kinds
inline Json parse(int argc, char** argv, const Spec& spec) {
    Json a = Json::dict();
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") throw Exit{"help"};
        if (arg.empty() || arg[0] != '-') {
            if (spec.positional_many) {
                Json list = a.find(spec.positional) ? *a.find(spec.positional) : Json::list();
                put(a, spec.positional, list.add(Json::of(arg)));
            } else {
                if (a.find(spec.positional)) fail("unexpected argument '" + arg + "'");
                put(a, spec.positional, Json::of(arg));
            }
            continue;
        }
        std::string name, inline_value;
        bool has_inline = false;
        if (spec.short_names.count(arg)) name = spec.short_names.at(arg);
        else if (arg.rfind("--", 0) == 0) {
            name = arg.substr(2);
            const std::size_t eq = name.find('=');
            if (eq != std::string::npos) inline_value = name.substr(eq + 1), name = name.substr(0, eq), has_inline = true;
            std::replace(name.begin(), name.end(), '-', '_');
        } else fail("unknown option " + arg);
        const Option* s = spec.find(name);
        if (!s || name == spec.positional) fail("unknown option " + arg);
        if (s->nargs == 0) {
            put(a, name, Json::of_bool(true));
            continue;
        }
        std::vector<std::string> vals;
        if (has_inline) vals.push_back(inline_value);
        const int want = s->nargs > 0 ? s->nargs : 1;
        while (static_cast<int>(vals.size()) < want) {
            if (i + 1 >= argc) fail("option --" + name + " needs " + std::to_string(want) + " value(s)");
            vals.push_back(argv[++i]);
        }
        if (s->nargs == -1)  // one or more: everything up to the next option
            while (i + 1 < argc && (argv[i + 1][0] != '-' || std::isdigit(static_cast<unsigned char>(argv[i + 1][1])) || argv[i + 1][1] == '.'))
                vals.push_back(argv[++i]);
        if (s->nargs == 1) {
            put(a, name, value_of(s->kind, vals[0], name));
        } else if (s->nargs == -2) {
            Json list = a.find(name) ? *a.find(name) : Json::list();
            put(a, name, list.add(Json::of(vals[0])));
        } else {
            Json list = Json::list();
            for (const auto& v : vals) list.add(value_of(s->kind, v, name));
            put(a, name, list);
        }
    }
    return a;
}

inline std::string read_text(const std::string& path) {
    std::ifstream in(path);
    if (!in) fail("cannot read " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

inline void write_text(const std::string& path, const std::string& text) {
    std::ofstream o(path);
    o << text << "\n";
    if (!o) fail("cannot write " + path);
}

// The options in effect: `defaults`, then the --config file's (keys: option names; "_..." ignored; `adapt` may
// rename or reinterpret a key and returns false for one it does not know), then the command line's
inline Json in_effect(Json defaults, const Json& given, const Spec& spec, const std::function<bool(const std::string&, const Json&, Json&)>& adapt = nullptr) {
    const auto merge = [&](const Json& from) {
        for (const auto& [k, v] : from.object)
            if (k != "config") put(defaults, k, v);
    };
    if (const Json* config = given.find("config")) {
        const Json file = Json::parse(read_text(config->string));
        Json clean = Json::dict();
        std::vector<std::string> unknown;
        for (const auto& [k0, v] : file.object) {
            if (!k0.empty() && k0[0] == '_') continue;
            std::string k = k0;
            std::replace(k.begin(), k.end(), '-', '_');
            if (adapt && adapt(k, v, clean)) continue;
            if (!spec.find(k)) unknown.push_back(k);
            else put(clean, k, v);
        }
        if (!unknown.empty()) {
            std::sort(unknown.begin(), unknown.end());
            std::string list;
            for (const auto& u : unknown) list += (list.empty() ? "" : ", ") + u;
            fail(config->string + ": unknown keys [" + list + "]");
        }
        merge(clean);
    }
    merge(given);
    return defaults;
}

// typed access to the options in effect
struct Args {
    Json all;
    const Json* get(const char* k) const {
        const Json* v = all.find(k);
        return v && !v->is(Json::Null) ? v : nullptr;
    }
    bool flag(const char* k) const { return get(k) && get(k)->is(Json::Bool) && get(k)->boolean; }
    std::string text(const char* k) const { return get(k) ? get(k)->string : std::string(); }
    double number(const char* k) const { return get(k)->number; }
    std::vector<double> numbers(const char* k) const {
        std::vector<double> v;
        if (get(k))
            for (const Json& x : get(k)->array) v.push_back(x.number);
        return v;
    }
    std::vector<int> ints(const char* k) const {
        std::vector<int> v;
        for (double x : numbers(k)) v.push_back(static_cast<int>(x));
        return v;
    }
    std::vector<std::string> texts(const char* k) const {
        std::vector<std::string> v;
        if (get(k))
            for (const Json& x : get(k)->array) v.push_back(x.string);
        return v;
    }
};

}  // namespace helirec::command_line
