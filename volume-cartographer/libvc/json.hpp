#pragma once
// Thin pimpl wrapper around nlohmann::json.
// Only json.cpp includes <nlohmann/json.hpp>. Everything else uses this header.
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace vc {

class Json {
    struct Impl;
    std::unique_ptr<Impl> p_;

public:
    Json();
    ~Json();
    Json(const Json&);
    Json(Json&&) noexcept;
    Json& operator=(const Json&);
    Json& operator=(Json&&) noexcept;

    // Construct from values
    Json(bool v);
    Json(int v);
    Json(int64_t v);
    Json(uint64_t v);
    Json(double v);
    Json(const char* v);
    Json(std::string_view v);

    // Factories
    static Json object();
    static Json array();
    static Json parse(std::string_view text);
    static Json parse_file(const std::filesystem::path& path);

    // Serialize
    std::string dump(int indent = -1) const;
    void dump_to_file(const std::filesystem::path& path, int indent = 2) const;

    // Type checks
    bool is_null() const;
    bool is_object() const;
    bool is_array() const;
    bool is_string() const;
    bool is_number() const;

    // Size
    size_t size() const;
    bool empty() const;

    // Read — returns copy (no reference semantics, no lifetime issues)
    Json operator[](const char* key) const;
    Json operator[](size_t index) const;
    bool contains(const char* key) const;

    // Typed getters
    std::string str() const;
    int i32() const;
    int64_t i64() const;
    uint64_t u64() const;
    double f64() const;
    float f32() const;
    bool boolean() const;

    // Get with default
    std::string str_or(const char* key, const char* def) const;
    int i32_or(const char* key, int def) const;
    int64_t i64_or(const char* key, int64_t def) const;
    double f64_or(const char* key, double def) const;
    bool bool_or(const char* key, bool def) const;

    // Array getters
    std::vector<std::string> str_array() const;
    std::vector<double> f64_array() const;
    std::vector<int> i32_array() const;

    // Write — mutate in place
    void set(const char* key, Json val);
    void push(Json val);
    void erase(const char* key);

    // Iterate object keys — returns pairs of (key, value copy)
    std::vector<std::pair<std::string, Json>> items() const;

    // Iterate array — returns copies of each element
    std::vector<Json> elements() const;

    bool operator==(const Json& o) const;
};

} // namespace vc
