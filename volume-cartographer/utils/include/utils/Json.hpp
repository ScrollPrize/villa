#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace utils {

// Thin non-template wrapper around nlohmann::json.
// Only Json.cpp includes <nlohmann/json.hpp>; everything else uses this header.
class Json {
public:
    // -- Constructors / destructors --
    Json();                                     // null
    ~Json();
    Json(const Json& o);
    Json(Json&& o) noexcept;
    Json& operator=(const Json& o);
    Json& operator=(Json&& o) noexcept;

    // Implicit conversions from common types
    Json(std::nullptr_t);
    Json(bool v);
    Json(int v);
    Json(int64_t v);
    Json(uint64_t v);
#if defined(__APPLE__)
    Json(size_t v);
#endif
    Json(double v);
    Json(const char* v);
    Json(const std::string& v);
    Json(std::string_view v);

    // Construct from initializer list (for objects: {{"key", val}, ...})
    Json(std::initializer_list<std::pair<const std::string, Json>> pairs);

    // -- Named constructors --
    static Json object();
    static Json array();
    static Json parse(std::string_view text);
    static Json parse_file(const std::filesystem::path& path);

    // -- Serialization --
    [[nodiscard]] std::string dump(int indent = -1) const;

    // -- Type checks --
    [[nodiscard]] bool is_null() const;
    [[nodiscard]] bool is_object() const;
    [[nodiscard]] bool is_array() const;
    [[nodiscard]] bool is_string() const;
    [[nodiscard]] bool is_number() const;
    [[nodiscard]] bool is_number_float() const;
    [[nodiscard]] bool is_number_integer() const;
    [[nodiscard]] bool is_boolean() const;

    // -- Size / emptiness --
    [[nodiscard]] size_t size() const;
    [[nodiscard]] bool empty() const;

    // -- Object access --
    [[nodiscard]] bool contains(const std::string& key) const;
    [[nodiscard]] size_t count(const std::string& key) const;
    Json& operator[](const std::string& key);
    const Json& operator[](const std::string& key) const;
    Json& at(const std::string& key);
    const Json& at(const std::string& key) const;

    // -- Array access --
    Json& operator[](size_t index);
    const Json& operator[](size_t index) const;
    Json& at(size_t index);
    const Json& at(size_t index) const;
    void push_back(const Json& val);
    void push_back(Json&& val);
    void emplace_back(Json&& val);

    // -- Get typed values (non-template explicit overloads) --
    [[nodiscard]] std::string get_string() const;
    [[nodiscard]] int get_int() const;
    [[nodiscard]] int64_t get_int64() const;
    [[nodiscard]] uint64_t get_uint64() const;
    [[nodiscard]] double get_double() const;
    [[nodiscard]] float get_float() const;
    [[nodiscard]] size_t get_size_t() const;
    [[nodiscard]] bool get_bool() const;

    // Get with default (like nlohmann .value())
    [[nodiscard]] std::string value(const std::string& key, const std::string& def) const;
    [[nodiscard]] std::string value(const std::string& key, const char* def) const;
    [[nodiscard]] int value(const std::string& key, int def) const;
    [[nodiscard]] int64_t value(const std::string& key, int64_t def) const;
    [[nodiscard]] float value(const std::string& key, float def) const;
    [[nodiscard]] double value(const std::string& key, double def) const;
    [[nodiscard]] bool value(const std::string& key, bool def) const;
    [[nodiscard]] uint64_t value(const std::string& key, uint64_t def) const;

    // Get vectors
    [[nodiscard]] std::vector<std::string> get_string_array() const;
    [[nodiscard]] std::vector<int> get_int_array() const;
    [[nodiscard]] std::vector<double> get_double_array() const;
    [[nodiscard]] std::vector<size_t> get_size_t_array() const;

    // -- Mutation --
    void update(const Json& other);   // merge object keys
    void merge_patch(const Json& patch);
    void erase(const std::string& key);  // remove key from object

    // -- Assignment from common types --
    Json& operator=(std::nullptr_t);
    Json& operator=(bool v);
    Json& operator=(int v);
    Json& operator=(int64_t v);
    Json& operator=(uint64_t v);
#if defined(__APPLE__)
    Json& operator=(size_t v);
#endif
    Json& operator=(double v);
    Json& operator=(const char* v);
    Json& operator=(const std::string& v);

    // -- Comparison --
    bool operator==(const Json& o) const;
    bool operator!=(const Json& o) const;

    // -- Iterator (for range-for) --
    class Iterator {
    public:
        Iterator();
        ~Iterator();
        Iterator(const Iterator&);
        Iterator& operator=(const Iterator&);
        Iterator& operator++();
        bool operator!=(const Iterator& o) const;
        Json& operator*();
        const Json& operator*() const;
        // For object iteration: key()
        [[nodiscard]] std::string key() const;
    private:
        friend class Json;
        struct IterImpl;
        std::unique_ptr<IterImpl> impl_;
    };

    Iterator begin();
    Iterator end();

    class ConstIterator {
    public:
        ConstIterator();
        ~ConstIterator();
        ConstIterator(const ConstIterator&);
        ConstIterator& operator=(const ConstIterator&);
        ConstIterator& operator++();
        bool operator!=(const ConstIterator& o) const;
        const Json& operator*() const;
        [[nodiscard]] std::string key() const;
    private:
        friend class Json;
        struct IterImpl;
        std::unique_ptr<IterImpl> impl_;
    };

    ConstIterator begin() const;
    ConstIterator end() const;

    // -- Low-level: access the underlying nlohmann::json (only for transition code) --
    // These are intentionally ugly names to discourage casual use.
    void* _raw_ptr();
    const void* _raw_ptr() const;

    // Construct wrapping an existing nlohmann::json (takes ownership via move)
    struct from_raw_t {};
    Json(from_raw_t, void* nlohmann_json_ptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Stream output
std::ostream& operator<<(std::ostream& os, const Json& j);

} // namespace utils
