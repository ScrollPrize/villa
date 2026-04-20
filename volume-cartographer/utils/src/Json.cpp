// ============================================================================
// THE ONLY FILE THAT INCLUDES <nlohmann/json.hpp>
// ============================================================================
#include "utils/Json.hpp"
#include <nlohmann/json.hpp>
#include <deque>
#include <fstream>
#include <ostream>

namespace utils {

using njson = nlohmann::json;

// Impl: either owns a njson (heap-allocated) or references one in a parent.
// Copy always makes an owning deep copy. Move transfers ownership.
struct Json::Impl {
    njson* ptr;
    bool owned;

    Impl() : ptr(new njson()), owned(true) {}
    explicit Impl(njson&& v) : ptr(new njson(std::move(v))), owned(true) {}
    explicit Impl(const njson& v) : ptr(new njson(v)), owned(true) {}
    Impl(njson* p, bool own) : ptr(p), owned(own) {}
    ~Impl() { if (owned) delete ptr; }

    // No copy — use clone()
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    njson& j() { return *ptr; }
    const njson& j() const { return *ptr; }
};

// ---- Constructors ----
Json::Json() : impl_(std::make_unique<Impl>()) {}
Json::~Json() = default;

Json::Json(const Json& o) : impl_(std::make_unique<Impl>(o.impl_->j())) {}

Json::Json(Json&& o) noexcept : impl_(std::move(o.impl_)) {
    if (!impl_) impl_ = std::make_unique<Impl>(); // moved-from is null
}

Json& Json::operator=(const Json& o) {
    if (this != &o) impl_->j() = o.impl_->j();
    return *this;
}

Json& Json::operator=(Json&& o) noexcept {
    if (this != &o) {
        // If we're non-owning (ref into parent), copy the value
        if (!impl_->owned) {
            impl_->j() = std::move(o.impl_->j());
        } else {
            impl_ = std::move(o.impl_);
            if (!impl_) impl_ = std::make_unique<Impl>();
        }
    }
    return *this;
}

Json::Json(std::nullptr_t) : impl_(std::make_unique<Impl>(njson(nullptr))) {}
Json::Json(bool v) : impl_(std::make_unique<Impl>(njson(v))) {}
Json::Json(int v) : impl_(std::make_unique<Impl>(njson(v))) {}
Json::Json(int64_t v) : impl_(std::make_unique<Impl>(njson(v))) {}
Json::Json(uint64_t v) : impl_(std::make_unique<Impl>(njson(v))) {}
Json::Json(double v) : impl_(std::make_unique<Impl>(njson(v))) {}
Json::Json(const char* v) : impl_(std::make_unique<Impl>(njson(std::string(v)))) {}
Json::Json(const std::string& v) : impl_(std::make_unique<Impl>(njson(v))) {}
Json::Json(std::string_view v) : impl_(std::make_unique<Impl>(njson(std::string(v)))) {}

Json::Json(std::initializer_list<std::pair<const std::string, Json>> pairs)
    : impl_(std::make_unique<Impl>(njson::object()))
{
    for (auto& [k, v] : pairs)
        impl_->j()[k] = v.impl_->j();
}

// ---- Named constructors ----
Json Json::object() { Json j; j.impl_->j() = njson::object(); return j; }
Json Json::array() { Json j; j.impl_->j() = njson::array(); return j; }

Json Json::parse(std::string_view text) {
    Json j;
    j.impl_->j() = njson::parse(text);
    return j;
}

Json Json::parse_file(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path.string());
    Json j;
    j.impl_->j() = njson::parse(f);
    return j;
}

// ---- Serialization ----
std::string Json::dump(int indent) const { return impl_->j().dump(indent); }

// ---- Type checks ----
bool Json::is_null() const { return impl_->j().is_null(); }
bool Json::is_object() const { return impl_->j().is_object(); }
bool Json::is_array() const { return impl_->j().is_array(); }
bool Json::is_string() const { return impl_->j().is_string(); }
bool Json::is_number() const { return impl_->j().is_number(); }
bool Json::is_number_float() const { return impl_->j().is_number_float(); }
bool Json::is_number_integer() const { return impl_->j().is_number_integer(); }
bool Json::is_boolean() const { return impl_->j().is_boolean(); }

// ---- Size ----
size_t Json::size() const { return impl_->j().size(); }
bool Json::empty() const { return impl_->j().empty(); }

// ---- Object access ----
bool Json::contains(const std::string& key) const { return impl_->j().contains(key); }
size_t Json::count(const std::string& key) const { return impl_->j().count(key); }

// operator[] returns a NON-OWNING Json referencing into parent's njson.
// This is safe as long as parent outlives the reference (same as raw njson).
Json& Json::operator[](const std::string& key) {
    // We need stable storage for the returned Json. Use thread_local cache.
    // Max depth of chained [] is small (typically 1-2).
    thread_local std::deque<Json> cache;
    cache.emplace_back();
    auto& child = cache.back();
    child.impl_ = std::make_unique<Impl>(&(impl_->j()[key]), false);
    // Trim cache if it grows too large (shouldn't happen in practice)
    while (cache.size() > 64) cache.pop_front();
    return child;
}

const Json& Json::operator[](const std::string& key) const {
    thread_local std::deque<Json> cache;
    cache.emplace_back();
    auto& child = cache.back();
    // const_cast is safe: the const version only exposes const& of the child
    child.impl_ = std::make_unique<Impl>(
        const_cast<njson*>(&(impl_->j().at(key))), false);
    while (cache.size() > 64) cache.pop_front();
    return child;
}

Json& Json::at(const std::string& key) {
    thread_local std::deque<Json> cache;
    cache.emplace_back();
    auto& child = cache.back();
    child.impl_ = std::make_unique<Impl>(&(impl_->j().at(key)), false);
    while (cache.size() > 64) cache.pop_front();
    return child;
}

const Json& Json::at(const std::string& key) const {
    thread_local std::deque<Json> cache;
    cache.emplace_back();
    auto& child = cache.back();
    child.impl_ = std::make_unique<Impl>(
        const_cast<njson*>(&(impl_->j().at(key))), false);
    while (cache.size() > 64) cache.pop_front();
    return child;
}

// ---- Array access ----
Json& Json::operator[](size_t index) {
    thread_local std::deque<Json> cache;
    cache.emplace_back();
    auto& child = cache.back();
    child.impl_ = std::make_unique<Impl>(&(impl_->j()[index]), false);
    while (cache.size() > 64) cache.pop_front();
    return child;
}

const Json& Json::operator[](size_t index) const {
    thread_local std::deque<Json> cache;
    cache.emplace_back();
    auto& child = cache.back();
    child.impl_ = std::make_unique<Impl>(
        const_cast<njson*>(&(impl_->j().at(index))), false);
    while (cache.size() > 64) cache.pop_front();
    return child;
}

Json& Json::at(size_t index) {
    thread_local std::deque<Json> cache;
    cache.emplace_back();
    auto& child = cache.back();
    child.impl_ = std::make_unique<Impl>(&(impl_->j().at(index)), false);
    while (cache.size() > 64) cache.pop_front();
    return child;
}

const Json& Json::at(size_t index) const {
    thread_local std::deque<Json> cache;
    cache.emplace_back();
    auto& child = cache.back();
    child.impl_ = std::make_unique<Impl>(
        const_cast<njson*>(&(impl_->j().at(index))), false);
    while (cache.size() > 64) cache.pop_front();
    return child;
}

void Json::push_back(const Json& val) { impl_->j().push_back(val.impl_->j()); }
void Json::push_back(Json&& val) { impl_->j().push_back(std::move(val.impl_->j())); }
void Json::emplace_back(Json&& val) { impl_->j().emplace_back(std::move(val.impl_->j())); }

// ---- Get typed values ----
std::string Json::get_string() const { return impl_->j().get<std::string>(); }
int Json::get_int() const { return impl_->j().get<int>(); }
int64_t Json::get_int64() const { return impl_->j().get<int64_t>(); }
uint64_t Json::get_uint64() const { return impl_->j().get<uint64_t>(); }
double Json::get_double() const { return impl_->j().get<double>(); }
float Json::get_float() const { return impl_->j().get<float>(); }
size_t Json::get_size_t() const { return impl_->j().get<size_t>(); }
bool Json::get_bool() const { return impl_->j().get<bool>(); }

std::string Json::value(const std::string& key, const std::string& def) const {
    return impl_->j().value(key, def);
}
std::string Json::value(const std::string& key, const char* def) const {
    return impl_->j().value(key, std::string(def));
}
int Json::value(const std::string& key, int def) const {
    return impl_->j().value(key, def);
}
int64_t Json::value(const std::string& key, int64_t def) const {
    return impl_->j().value(key, def);
}
float Json::value(const std::string& key, float def) const {
    return impl_->j().value(key, def);
}
double Json::value(const std::string& key, double def) const {
    return impl_->j().value(key, def);
}
bool Json::value(const std::string& key, bool def) const {
    return impl_->j().value(key, def);
}
uint64_t Json::value(const std::string& key, uint64_t def) const {
    return impl_->j().value(key, def);
}

std::vector<std::string> Json::get_string_array() const {
    return impl_->j().get<std::vector<std::string>>();
}
std::vector<int> Json::get_int_array() const {
    return impl_->j().get<std::vector<int>>();
}
std::vector<double> Json::get_double_array() const {
    return impl_->j().get<std::vector<double>>();
}
std::vector<size_t> Json::get_size_t_array() const {
    return impl_->j().get<std::vector<size_t>>();
}

// ---- Mutation ----
void Json::update(const Json& other) { impl_->j().update(other.impl_->j()); }
void Json::merge_patch(const Json& patch) { impl_->j().merge_patch(patch.impl_->j()); }
void Json::erase(const std::string& key) { impl_->j().erase(key); }

// ---- Assignment from types ----
Json& Json::operator=(std::nullptr_t) { impl_->j() = nullptr; return *this; }
Json& Json::operator=(bool v) { impl_->j() = v; return *this; }
Json& Json::operator=(int v) { impl_->j() = v; return *this; }
Json& Json::operator=(int64_t v) { impl_->j() = v; return *this; }
Json& Json::operator=(uint64_t v) { impl_->j() = v; return *this; }
Json& Json::operator=(double v) { impl_->j() = v; return *this; }
Json& Json::operator=(const char* v) { impl_->j() = std::string(v); return *this; }
Json& Json::operator=(const std::string& v) { impl_->j() = v; return *this; }

// ---- Comparison ----
bool Json::operator==(const Json& o) const { return impl_->j() == o.impl_->j(); }
bool Json::operator!=(const Json& o) const { return impl_->j() != o.impl_->j(); }

// ---- Iterators ----
struct Json::Iterator::IterImpl {
    njson::iterator it;
    Json current; // wraps the current element
};

Json::Iterator::Iterator() : impl_(std::make_unique<IterImpl>()) {}
Json::Iterator::~Iterator() = default;
Json::Iterator::Iterator(const Iterator& o) : impl_(std::make_unique<IterImpl>(*o.impl_)) {}
Json::Iterator& Json::Iterator::operator=(const Iterator& o) {
    *impl_ = *o.impl_;
    return *this;
}
Json::Iterator& Json::Iterator::operator++() { ++impl_->it; return *this; }
bool Json::Iterator::operator!=(const Iterator& o) const { return impl_->it != o.impl_->it; }
Json& Json::Iterator::operator*() {
    impl_->current.impl_ = std::make_unique<Impl>(&(*impl_->it), false);
    return impl_->current;
}
const Json& Json::Iterator::operator*() const {
    impl_->current.impl_ = std::make_unique<Impl>(
        const_cast<njson*>(&(*impl_->it)), false);
    return impl_->current;
}
std::string Json::Iterator::key() const { return impl_->it.key(); }

Json::Iterator Json::begin() {
    Iterator it;
    it.impl_->it = impl_->j().begin();
    return it;
}
Json::Iterator Json::end() {
    Iterator it;
    it.impl_->it = impl_->j().end();
    return it;
}

struct Json::ConstIterator::IterImpl {
    njson::const_iterator it;
    mutable Json current;
};

Json::ConstIterator::ConstIterator() : impl_(std::make_unique<IterImpl>()) {}
Json::ConstIterator::~ConstIterator() = default;
Json::ConstIterator::ConstIterator(const ConstIterator& o) : impl_(std::make_unique<IterImpl>(*o.impl_)) {}
Json::ConstIterator& Json::ConstIterator::operator=(const ConstIterator& o) {
    *impl_ = *o.impl_;
    return *this;
}
Json::ConstIterator& Json::ConstIterator::operator++() { ++impl_->it; return *this; }
bool Json::ConstIterator::operator!=(const ConstIterator& o) const { return impl_->it != o.impl_->it; }
const Json& Json::ConstIterator::operator*() const {
    impl_->current.impl_ = std::make_unique<Impl>(
        const_cast<njson*>(&(*impl_->it)), false);
    return impl_->current;
}
std::string Json::ConstIterator::key() const { return impl_->it.key(); }

Json::ConstIterator Json::begin() const {
    ConstIterator it;
    it.impl_->it = impl_->j().begin();
    return it;
}
Json::ConstIterator Json::end() const {
    ConstIterator it;
    it.impl_->it = impl_->j().end();
    return it;
}

// ---- Raw access (escape hatch) ----
void* Json::_raw_ptr() { return impl_->ptr; }
const void* Json::_raw_ptr() const { return impl_->ptr; }

Json::Json(from_raw_t, void* p) : impl_(std::make_unique<Impl>(
    std::move(*static_cast<njson*>(p)))) { delete static_cast<njson*>(p); }

// ---- Stream ----
std::ostream& operator<<(std::ostream& os, const Json& j) {
    os << j.dump();
    return os;
}

} // namespace utils
