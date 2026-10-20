// Minimal test runner. Define VC_TEST_MAIN in one translation unit per executable.
#pragma once

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace doctest {

// --- Approx: matches doctest's comparison semantics exactly ---
class Approx {
public:
    explicit Approx(double value) : m_value(value) {}

    Approx& epsilon(double eps) { m_epsilon = eps; return *this; }

    friend bool operator==(double lhs, const Approx& rhs) {
        return std::fabs(lhs - rhs.m_value) <
               rhs.m_epsilon * (rhs.m_scale + std::max(std::fabs(lhs), std::fabs(rhs.m_value)));
    }
    friend bool operator==(const Approx& lhs, double rhs) { return operator==(rhs, lhs); }
    friend bool operator!=(double lhs, const Approx& rhs) { return !operator==(lhs, rhs); }
    friend bool operator!=(const Approx& lhs, double rhs) { return !operator==(rhs, lhs); }

private:
    double m_epsilon = std::numeric_limits<float>::epsilon() * 100;
    double m_scale = 1.0;
    double m_value;
};

// --- Contains: substring matcher used by CHECK_THROWS_WITH_AS ---
struct Contains {
    explicit Contains(std::string n) : needle(std::move(n)) {}
    bool match(const std::string& haystack) const {
        return haystack.find(needle) != std::string::npos;
    }
    std::string needle;
};

namespace detail {

// Thrown by the REQUIRE family to abort the current test case body.
struct RequireFailed {};

struct Runner {
    struct Case { std::string name; std::function<void()> fn; };
    std::vector<Case> cases;
    int failures = 0;

    static Runner& get() { static Runner r; return r; }
};

inline void fail(const char* file, int line, const std::string& expr) {
    Runner::get().failures++;
    std::cerr << file << ":" << line << ": FAILED: " << expr << "\n";
}

inline void runCase(const Runner::Case& c) {
    auto& r = Runner::get();
    try {
        c.fn();
    } catch (const RequireFailed&) {
        // Failure already counted by REQUIRE.
    } catch (const std::exception& e) {
        r.failures++;
        std::cerr << "uncaught exception in '" << c.name << "': " << e.what() << "\n";
    } catch (...) {
        r.failures++;
        std::cerr << "uncaught unknown exception in '" << c.name << "'\n";
    }
}

inline int runAll() {
    auto& r = Runner::get();
    for (auto& c : r.cases) {
        runCase(c);
    }
    if (r.failures) {
        std::cerr << r.failures << " check(s) failed\n";
        return 1;
    }
    std::cout << r.cases.size() << " test case(s) passed\n";
    return 0;
}

struct Registrar {
    Registrar(std::string name, std::function<void()> fn) {
        Runner::get().cases.push_back({std::move(name), std::move(fn)});
    }
};

}  // namespace detail
}  // namespace doctest

// ---- Test registration ----
#define DOCTEST_CONCAT2(a, b) a##b
#define DOCTEST_CONCAT(a, b) DOCTEST_CONCAT2(a, b)
#define DOCTEST_ANON(prefix) DOCTEST_CONCAT(prefix, __LINE__)

#define TEST_CASE(name)                                                        \
    static void DOCTEST_ANON(vc_test_fn_)();                                   \
    static ::doctest::detail::Registrar DOCTEST_ANON(vc_test_reg_)(            \
        name, &DOCTEST_ANON(vc_test_fn_));                                     \
    static void DOCTEST_ANON(vc_test_fn_)()

#define DOCTEST_CHECK_IMPL(cond, fatal, exprstr)                               \
    do {                                                                       \
        bool _vc_ok = false;                                                   \
        try { _vc_ok = static_cast<bool>(cond); }                              \
        catch (...) { _vc_ok = false; }                                        \
        if (!_vc_ok) {                                                         \
            ::doctest::detail::fail(__FILE__, __LINE__, exprstr);              \
            if (fatal) throw ::doctest::detail::RequireFailed{};               \
        }                                                                      \
    } while (0)

// Variadic so expressions containing top-level commas (e.g. brace-init like
// std::vector{1, 2, 3} or multi-arg calls) parse as one macro argument, the
// same way doctest's CHECK(...) does.
#define CHECK(...)         DOCTEST_CHECK_IMPL((__VA_ARGS__), false, "CHECK(" #__VA_ARGS__ ")")
#define CHECK_FALSE(...)   DOCTEST_CHECK_IMPL(!(__VA_ARGS__), false, "CHECK_FALSE(" #__VA_ARGS__ ")")
#define REQUIRE(...)       DOCTEST_CHECK_IMPL((__VA_ARGS__), true,  "REQUIRE(" #__VA_ARGS__ ")")

#define DOCTEST_THROWS_IMPL(expr, fatal)                                       \
    do {                                                                       \
        bool _vc_threw = false;                                                \
        try { (void)(expr); } catch (...) { _vc_threw = true; }                \
        if (!_vc_threw) {                                                      \
            ::doctest::detail::fail(__FILE__, __LINE__, "CHECK_THROWS(" #expr ")"); \
            if (fatal) throw ::doctest::detail::RequireFailed{};               \
        }                                                                      \
    } while (0)
#define CHECK_THROWS(expr)   DOCTEST_THROWS_IMPL((expr), false)

#define DOCTEST_THROWS_AS_IMPL(expr, ex, fatal)                                \
    do {                                                                       \
        bool _vc_ok = false;                                                   \
        try { (void)(expr); }                                                  \
        catch (const ex&) { _vc_ok = true; }                                   \
        catch (...) {}                                                         \
        if (!_vc_ok) {                                                         \
            ::doctest::detail::fail(__FILE__, __LINE__,                        \
                "CHECK_THROWS_AS(" #expr ", " #ex ")");                        \
            if (fatal) throw ::doctest::detail::RequireFailed{};               \
        }                                                                      \
    } while (0)
#define CHECK_THROWS_AS(expr, ex)   DOCTEST_THROWS_AS_IMPL((expr), ex, false)

// CHECK_THROWS_WITH_AS(expr, matcher, ExceptionType): the thrown exception
// must be ExceptionType and its what() must satisfy the matcher (Contains).
#define DOCTEST_THROWS_WITH_AS_IMPL(expr, matcher, ex, fatal)                  \
    do {                                                                       \
        bool _vc_ok = false;                                                   \
        try { (void)(expr); }                                                  \
        catch (const ex& _vc_e) { _vc_ok = (matcher).match(_vc_e.what()); }    \
        catch (...) {}                                                         \
        if (!_vc_ok) {                                                         \
            ::doctest::detail::fail(__FILE__, __LINE__,                        \
                "CHECK_THROWS_WITH_AS(" #expr ", " #ex ")");                   \
            if (fatal) throw ::doctest::detail::RequireFailed{};               \
        }                                                                      \
    } while (0)
#define CHECK_THROWS_WITH_AS(expr, matcher, ex)   DOCTEST_THROWS_WITH_AS_IMPL((expr), matcher, ex, false)

#define DOCTEST_NOTHROW_IMPL(expr, fatal)                                      \
    do {                                                                       \
        try { (void)(expr); }                                                  \
        catch (...) {                                                          \
            ::doctest::detail::fail(__FILE__, __LINE__, "NOTHROW(" #expr ")"); \
            if (fatal) throw ::doctest::detail::RequireFailed{};               \
        }                                                                      \
    } while (0)
#define CHECK_NOTHROW(expr)   DOCTEST_NOTHROW_IMPL((expr), false)

// ---- main() ----
#ifdef VC_TEST_MAIN
int main() { return ::doctest::detail::runAll(); }
#endif
