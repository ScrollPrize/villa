// THE ONLY FILE THAT INCLUDES <nlohmann/json.hpp>
#include "json.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

namespace vc {

using nj = nlohmann::json;

struct Json::Impl {
    nj j;
    Impl() = default;
    Impl(nj v) : j(std::move(v)) {}
};

Json::Json() : p_(std::make_unique<Impl>()) {}
Json::~Json() = default;
Json::Json(const Json& o) : p_(std::make_unique<Impl>(o.p_->j)) {}
Json::Json(Json&& o) noexcept = default;
Json& Json::operator=(const Json& o) { p_->j = o.p_->j; return *this; }
Json& Json::operator=(Json&& o) noexcept = default;

Json::Json(bool v) : p_(std::make_unique<Impl>(nj(v))) {}
Json::Json(int v) : p_(std::make_unique<Impl>(nj(v))) {}
Json::Json(int64_t v) : p_(std::make_unique<Impl>(nj(v))) {}
Json::Json(uint64_t v) : p_(std::make_unique<Impl>(nj(v))) {}
Json::Json(double v) : p_(std::make_unique<Impl>(nj(v))) {}
Json::Json(const char* v) : p_(std::make_unique<Impl>(nj(std::string(v)))) {}
Json::Json(std::string_view v) : p_(std::make_unique<Impl>(nj(std::string(v)))) {}

Json Json::object() { Json j; j.p_->j = nj::object(); return j; }
Json Json::array() { Json j; j.p_->j = nj::array(); return j; }
Json Json::parse(std::string_view text) { Json j; j.p_->j = nj::parse(text); return j; }
Json Json::parse_file(const std::filesystem::path& path) {
    std::ifstream f(path);
    Json j;
    j.p_->j = nj::parse(f);
    return j;
}

std::string Json::dump(int indent) const { return p_->j.dump(indent); }
void Json::dump_to_file(const std::filesystem::path& path, int indent) const {
    std::ofstream f(path);
    f << p_->j.dump(indent);
}

bool Json::is_null() const { return p_->j.is_null(); }
bool Json::is_object() const { return p_->j.is_object(); }
bool Json::is_array() const { return p_->j.is_array(); }
bool Json::is_string() const { return p_->j.is_string(); }
bool Json::is_number() const { return p_->j.is_number(); }

size_t Json::size() const { return p_->j.size(); }
bool Json::empty() const { return p_->j.empty(); }

Json Json::operator[](const char* key) const {
    Json r;
    if (p_->j.contains(key)) r.p_->j = p_->j[key];
    return r;
}
Json Json::operator[](size_t index) const {
    Json r;
    r.p_->j = p_->j.at(index);
    return r;
}
bool Json::contains(const char* key) const { return p_->j.contains(key); }

std::string Json::str() const { return p_->j.get<std::string>(); }
int Json::i32() const { return p_->j.get<int>(); }
int64_t Json::i64() const { return p_->j.get<int64_t>(); }
uint64_t Json::u64() const { return p_->j.get<uint64_t>(); }
double Json::f64() const { return p_->j.get<double>(); }
float Json::f32() const { return p_->j.get<float>(); }
bool Json::boolean() const { return p_->j.get<bool>(); }

std::string Json::str_or(const char* key, const char* def) const {
    return p_->j.value(key, std::string(def));
}
int Json::i32_or(const char* key, int def) const { return p_->j.value(key, def); }
int64_t Json::i64_or(const char* key, int64_t def) const { return p_->j.value(key, def); }
double Json::f64_or(const char* key, double def) const { return p_->j.value(key, def); }
bool Json::bool_or(const char* key, bool def) const { return p_->j.value(key, def); }

std::vector<std::string> Json::str_array() const { return p_->j.get<std::vector<std::string>>(); }
std::vector<double> Json::f64_array() const { return p_->j.get<std::vector<double>>(); }
std::vector<int> Json::i32_array() const { return p_->j.get<std::vector<int>>(); }

void Json::set(const char* key, Json val) { p_->j[key] = std::move(val.p_->j); }
void Json::push(Json val) { p_->j.push_back(std::move(val.p_->j)); }
void Json::erase(const char* key) { p_->j.erase(key); }

std::vector<std::pair<std::string, Json>> Json::items() const {
    std::vector<std::pair<std::string, Json>> out;
    for (auto& [k, v] : p_->j.items()) {
        Json jv; jv.p_->j = v;
        out.emplace_back(k, std::move(jv));
    }
    return out;
}

std::vector<Json> Json::elements() const {
    std::vector<Json> out;
    for (auto& v : p_->j) {
        Json jv; jv.p_->j = v;
        out.push_back(std::move(jv));
    }
    return out;
}

bool Json::operator==(const Json& o) const { return p_->j == o.p_->j; }

} // namespace vc
