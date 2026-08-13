#include "vc/core/util/LoadJson.hpp"

#include <fstream>

namespace vc::json {

utils::Json load_json_file(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("JSON file not found: " + path.string());
    }
    return utils::Json::parse_file(path);
}

void require_fields(
    const utils::Json& json,
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
    const utils::Json& json,
    const char* field,
    const std::string& expected,
    const std::string& context)
{
    if (!json.contains(field)) {
        throw std::runtime_error(context + " missing required field: " + field);
    }
    std::string actual = json[field].get_string();
    if (actual != expected) {
        throw std::runtime_error(context + " has type '" + actual + "', expected '" + expected + "'");
    }
}

double number_or(const utils::Json& m, const char* key, double def) {
    if (m.is_null() || !m.is_object()) return def;
    if (!m.contains(key)) return def;
    const auto& v = m[key];
    if (v.is_number_float())   return v.get_double();
    if (v.is_number_integer()) return static_cast<double>(v.get_int64());
    if (v.is_string()) {
        try { return std::stod(v.get_string()); } catch (...) { return def; }
    }
    return def;
}

std::string string_or(const utils::Json& m, const char* key, const std::string& def) {
    if (m.is_null() || !m.is_object()) return def;
    if (m.contains(key) && m[key].is_string()) return m[key].get_string();
    return def;
}

utils::Json tags_or_empty(const utils::Json& m) {
    if (m.is_null() || !m.is_object()) return utils::Json::object();
    if (m.contains("tags") && m["tags"].is_object()) return m["tags"];
    return utils::Json::object();
}

bool has_tag(const utils::Json& m, const char* tag) {
    auto t = tags_or_empty(m);
    return t.contains(tag);
}

void ensure_object(utils::Json& m) {
    if (m.is_null() || !m.is_object()) m = utils::Json::object();
}

utils::Json& ensure_tags(utils::Json& m) {
    ensure_object(m);
    if (!m.contains("tags") || !m["tags"].is_object()) {
        m["tags"] = utils::Json::object();
    }
    return m["tags"];
}

} // namespace vc::json
