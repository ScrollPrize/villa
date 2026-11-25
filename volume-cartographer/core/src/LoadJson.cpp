#include "vc/core/util/LoadJson.hpp"

#include <fstream>

namespace vc::json {


void require_fields(
    const nlohmann::json& json,
    std::initializer_list<const char*> fields,
    const std::string& context)
{
    for (const char* field : fields) {
        if (!json.contains(field)) {
            throw std::runtime_error(context + " missing required field: " + field);
        }
    }
}

void require_type(
    const nlohmann::json& json,
    const char* field,
    const std::string& expected,
    const std::string& context)
{
    if (!json.contains(field)) {
        throw std::runtime_error(context + " missing required field: " + field);
    }
    std::string actual = json[field].get<std::string>();
    if (actual != expected) {
        throw std::runtime_error(context + " has type '" + actual + "', expected '" + expected + "'");
    }
}

double number_or(const nlohmann::json& m, const char* key, double def) {
    if (m.is_null() || !m.is_object()) return def;
    if (!m.contains(key)) return def;
    const auto& v = m[key];
    if (v.is_number()) return v.get<double>();
    if (v.is_string()) {
        try { return std::stod(v.get<std::string>()); } catch (...) { return def; }
    }
    return def;
}

std::string string_or(const nlohmann::json& m, const char* key, const std::string& def) {
    if (m.is_null() || !m.is_object()) return def;
    if (m.contains(key) && m[key].is_string()) return m[key].get<std::string>();
    return def;
}

} // namespace vc::json
