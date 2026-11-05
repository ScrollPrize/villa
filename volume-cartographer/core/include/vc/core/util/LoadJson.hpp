// LoadJson.hpp - JSON loading and validation utilities for VC
#pragma once

#include <filesystem>
#include <string>
#include <cmath>
#include <initializer_list>

#include <nlohmann/json.hpp>

namespace vc::json {


// ============ VALIDATION ============

void require_fields(
    const nlohmann::json& json,
    std::initializer_list<const char*> fields,
    const std::string& context);

void require_type(
    const nlohmann::json& json,
    const char* field,
    const std::string& expected,
    const std::string& context);

// ============ SAFE ACCESS HELPERS ============

// Returns a number if present (float/int or string convertible), else def.
double number_or(const nlohmann::json& m, const char* key, double def);

// Returns a string if present and of string type, else def.
std::string string_or(const nlohmann::json& m, const char* key, const std::string& def);

} // namespace vc::json
