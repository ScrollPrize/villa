// JsonSafe.hpp - safe JSON access helpers for VC
#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <cmath>

namespace vc::json_safe {

// Returns a number if present (float/int or string convertible), else def.
inline double number_or(const nlohmann::json* m, const char* key, double def) {
    if (!m || !m->is_object()) return def;
    auto it = m->find(key);
    if (it == m->end()) return def;
    if (it->is_number_float())   return it->get<double>();
    if (it->is_number_integer()) return static_cast<double>(it->get<int64_t>());
    if (it->is_string()) {
        try { return std::stod(it->get<std::string>()); } catch (...) { return def; }
    }
    return def;
}

// Returns a string if present and of string type, else def.
inline std::string string_or(const nlohmann::json* m, const char* key, const std::string& def) {
    if (!m || !m->is_object()) return def;
    auto it = m->find(key);
    if (it != m->end() && it->is_string()) return it->get<std::string>();
    return def;
}

// Returns tags object if present & object, else {} (by value).
inline nlohmann::json tags_or_empty(const nlohmann::json* m) {
    if (!m || !m->is_object()) return nlohmann::json::object();
    auto it = m->find("tags");
    if (it != m->end() && it->is_object()) return *it;
    return nlohmann::json::object();
}

inline bool has_tag(const nlohmann::json* m, const char* tag) {
    auto t = tags_or_empty(m);
    return t.contains(tag);
}

// Ensure *p is non-null and an object {}.
inline void ensure_object(nlohmann::json*& p) {
    if (!p) { p = new nlohmann::json(nlohmann::json::object()); return; }
    if (!p->is_object()) *p = nlohmann::json::object();
}

// Ensure *p is an object and (*p)["tags"] is an object; returns a ref.
inline nlohmann::json& ensure_tags(nlohmann::json*& p) {
    ensure_object(p);
    nlohmann::json& root = *p;
    auto it = root.find("tags");
    if (it == root.end() || !it->is_object()) {
        root["tags"] = nlohmann::json::object();
    }
    return root["tags"];
}

} // namespace vc::json_safe