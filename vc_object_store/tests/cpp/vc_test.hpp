// Minimal native runner: checks remain active in Release builds.
#pragma once
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace doctest {
struct Contains {
    std::string needle;
    bool match(const std::string& text) const { return text.find(needle) != std::string::npos; }
};
namespace detail {
struct RequireFailed {};
struct Case { const char* name; void (*run)(); };
inline std::vector<Case>& cases() { static std::vector<Case> value; return value; }
inline int failures = 0;
struct Registrar {
    Registrar(const char* name, void (*run)()) { cases().push_back({name, run}); }
};
inline void check(bool passed, const char* expression, const char* file, int line, bool fatal) {
    if (passed) return;
    ++failures;
    std::cerr << file << ':' << line << ": FAILED: " << expression << '\n';
    if (fatal) throw RequireFailed{};
}
template<class F> bool succeeds(F&& expression) {
    try { return static_cast<bool>(expression()); }
    catch (...) { return false; }
}
template<class F> bool throws(F&& expression) {
    try { expression(); } catch (...) { return true; }
    return false;
}
template<class E, class F> bool throwsWith(F&& expression, const Contains& matcher) {
    try { expression(); }
    catch (const E& error) { return matcher.match(error.what()); }
    catch (...) {}
    return false;
}
inline int runAll() {
    for (const auto& test : cases()) {
        try { test.run(); }
        catch (const RequireFailed&) {}
        catch (const std::exception& error) {
            ++failures;
            std::cerr << test.name << ": " << error.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << test.name << ": unknown exception\n";
        }
    }
    std::cout << cases().size() << " test case(s), " << failures << " failure(s)\n";
    return failures ? 1 : 0;
}
}
}
#define VC_JOIN_IMPL(a, b) a##b
#define VC_JOIN(a, b) VC_JOIN_IMPL(a, b)
#define TEST_CASE(name) \
    static void VC_JOIN(vc_test_, __LINE__)(); \
    static doctest::detail::Registrar VC_JOIN(vc_register_, __LINE__)(name, VC_JOIN(vc_test_, __LINE__)); \
    static void VC_JOIN(vc_test_, __LINE__)()
#define VC_CHECK(fatal, ...) \
    doctest::detail::check(doctest::detail::succeeds([&] { return (__VA_ARGS__); }), #__VA_ARGS__, __FILE__, __LINE__, fatal)
#define CHECK(...) VC_CHECK(false, __VA_ARGS__)
#define REQUIRE(...) VC_CHECK(true, __VA_ARGS__)
#define CHECK_FALSE(...) CHECK(!(__VA_ARGS__))
#define CHECK_THROWS(...) CHECK(doctest::detail::throws([&] { (void)(__VA_ARGS__); }))
#define CHECK_THROWS_WITH_AS(expression, matcher, type) \
    CHECK(doctest::detail::throwsWith<type>([&] { (void)(expression); }, matcher))
#ifdef VC_TEST_MAIN
int main() { return doctest::detail::runAll(); }
#endif
