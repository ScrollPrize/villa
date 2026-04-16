// LoadJson.hpp - JSON loading and validation utilities for VC
#pragma once

#include <filesystem>
#include <string>
#include <cmath>
#include <initializer_list>

#include "utils/Json.hpp"

namespace vc::json {

utils::Json load_json_file(const std::filesystem::path& path);

// ============ VALIDATION ============

void require_fields(
    const utils::Json& json,
    std::initializer_list<const char*> fields,
    const std::string& context);

void require_type(
    const utils::Json& json,
    const char* field,
    const std::string& expected,
    const std::string& context);

// ============ SAFE ACCESS HELPERS (utils::Json overloads) ============

// Returns a number if present (float/int or string convertible), else def.
double number_or(const utils::Json& m, const char* key, double def);

// Returns a string if present and of string type, else def.
std::string string_or(const utils::Json& m, const char* key, const std::string& def);

// Returns tags object if present & object, else {} (by value).
utils::Json tags_or_empty(const utils::Json& m);

bool has_tag(const utils::Json& m, const char* tag);

// Ensure meta is non-null and an object {}.
void ensure_object(utils::Json& m);

// Ensure meta is an object and meta["tags"] is an object; returns a ref.
utils::Json& ensure_tags(utils::Json& m);

} // namespace vc::json
