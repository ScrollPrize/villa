// vc_test — a minimal, doctest-compatible test framework.
//
// Drop-in for the subset of <doctest/doctest.h> this repo's tests use:
// TEST_CASE / SUBCASE, CHECK[_FALSE] / REQUIRE[_FALSE], CHECK_THROWS[_AS] /
// CHECK_THROWS_WITH_AS / CHECK_NOTHROW, REQUIRE_NOTHROW / REQUIRE_MESSAGE,
// FAIL / WARN / INFO / MESSAGE, doctest::Approx (+.epsilon), doctest::Contains.
//
// Why: doctest's single header is the heaviest include in the test build
// (~350s of parse time across the suite). This header is tiny, so that cost
// disappears. Tests are validated by exit code (non-zero on any failure);
// we don't reproduce doctest's report format.
//
// Define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN in exactly one TU per test exe
// (the tests already do) to emit main().
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace doctest {

// --- Approx: matches doctest's comparison semantics exactly ---
class Approx {
public:
    explicit Approx(double value) : m_value(value) {}

    Approx& epsilon(double eps) { m_epsilon = eps; return *this; }
    Approx& scale(double s) { m_scale = s; return *this; }

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

    // --- subcase re-execution state (one TEST_CASE pass at a time) ---
    std::vector<std::string> path;     // current nesting path this pass
    std::set<std::string> done;        // subcase paths fully executed (this case)
    std::string enteredThisPass;       // the one leaf-entered subcase, if any
    bool sawUnvisited = false;         // a runnable subcase was skipped -> need another pass

    static Runner& get() { static Runner r; return r; }
};

inline void fail(const char* file, int line, const std::string& expr) {
    Runner::get().failures++;
    std::cerr << file << ":" << line << ": FAILED: " << expr << "\n";
}

// RAII gate for a SUBCASE. doctest model: each pass enters at most one new
// subcase path; others are deferred to later passes.
struct Subcase {
    bool run = false;
    explicit Subcase(const char* name) {
        auto& r = Runner::get();
        const std::string p = (r.path.empty() ? "" : r.path.back() + "/") + name;
        if (r.done.count(p)) {
            return;  // already executed in a prior pass
        }
        if (!r.enteredThisPass.empty()) {
            // a sibling/other subcase already runs this pass; revisit later
            r.sawUnvisited = true;
            return;
        }
        r.enteredThisPass = p;
        r.path.push_back(p);
        run = true;
    }
    ~Subcase() {
        if (run) {
            auto& r = Runner::get();
            r.done.insert(r.path.back());
            r.path.pop_back();
        }
    }
    explicit operator bool() const { return run; }
};

inline void runCase(const Runner::Case& c) {
    auto& r = Runner::get();
    r.done.clear();
    do {
        r.path.clear();
        r.enteredThisPass.clear();
        r.sawUnvisited = false;
        try {
            c.fn();
        } catch (const RequireFailed&) {
            // failure already counted; this pass aborts
        } catch (const std::exception& e) {
            r.failures++;
            std::cerr << "uncaught exception in '" << c.name << "': " << e.what() << "\n";
        } catch (...) {
            r.failures++;
            std::cerr << "uncaught unknown exception in '" << c.name << "'\n";
        }
        // Another pass is needed only if this pass deferred a runnable subcase.
    } while (r.sawUnvisited);
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

#define SUBCASE(name)                                                          \
    if (::doctest::detail::Subcase DOCTEST_ANON(vc_sub_){name};                \
        DOCTEST_ANON(vc_sub_))

// ---- Assertions ----
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
#define REQUIRE_FALSE(...) DOCTEST_CHECK_IMPL(!(__VA_ARGS__), true, "REQUIRE_FALSE(" #__VA_ARGS__ ")")
#define WARN(...)          DOCTEST_CHECK_IMPL((__VA_ARGS__), false, "WARN(" #__VA_ARGS__ ")")

// REQUIRE_MESSAGE(cond, msg<<stream): fatal, with a streamed message.
#define DOCTEST_MSG_IMPL(cond, fatal, kind, msg)                               \
    do {                                                                       \
        bool _vc_ok = false;                                                   \
        try { _vc_ok = static_cast<bool>(cond); } catch (...) { _vc_ok = false; } \
        if (!_vc_ok) {                                                         \
            std::ostringstream _vc_os; _vc_os << kind ": " << msg;             \
            ::doctest::detail::fail(__FILE__, __LINE__, _vc_os.str());         \
            if (fatal) throw ::doctest::detail::RequireFailed{};               \
        }                                                                      \
    } while (0)

#define REQUIRE_MESSAGE(cond, msg) DOCTEST_MSG_IMPL((cond), true,  "REQUIRE", msg)
#define CHECK_MESSAGE(cond, msg)   DOCTEST_MSG_IMPL((cond), false, "CHECK",   msg)

// Informational — never fail. Tests write these as a stream expression,
// e.g. MESSAGE("x: " << e.what()); the macro prepends an ostream so the
// whole thing streams. (A rare comma form like INFO("a: ", x) streams only
// the first piece via the comma operator — fine, it's informational.)
#define MESSAGE(...) do { std::cerr << "[MESSAGE] " << __VA_ARGS__ << "\n"; } while (0)
#define INFO(...)    do { std::ostringstream _vc; _vc << __VA_ARGS__; (void)_vc; } while (0)
// CAPTURE(x): doctest logs x's value as failure context. We don't track
// context, so just evaluate it (catching the common case where the captured
// expr has side-effect-free value) and discard.
#define CAPTURE(...) do { std::ostringstream _vc; _vc << #__VA_ARGS__ " = " << (__VA_ARGS__); (void)_vc; } while (0)
#define FAIL(...)                                                              \
    do {                                                                       \
        std::ostringstream _vc_os; _vc_os << "FAIL: " << __VA_ARGS__;          \
        ::doctest::detail::fail(__FILE__, __LINE__, _vc_os.str());             \
        throw ::doctest::detail::RequireFailed{};                              \
    } while (0)

// ---- Exception assertions ----
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
#define REQUIRE_THROWS(expr) DOCTEST_THROWS_IMPL((expr), true)

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
#define REQUIRE_THROWS_AS(expr, ex) DOCTEST_THROWS_AS_IMPL((expr), ex, true)

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
#define REQUIRE_THROWS_WITH_AS(expr, matcher, ex) DOCTEST_THROWS_WITH_AS_IMPL((expr), matcher, ex, true)

#define DOCTEST_NOTHROW_IMPL(expr, fatal)                                      \
    do {                                                                       \
        try { (void)(expr); }                                                  \
        catch (...) {                                                          \
            ::doctest::detail::fail(__FILE__, __LINE__, "NOTHROW(" #expr ")"); \
            if (fatal) throw ::doctest::detail::RequireFailed{};               \
        }                                                                      \
    } while (0)
#define CHECK_NOTHROW(expr)   DOCTEST_NOTHROW_IMPL((expr), false)
#define REQUIRE_NOTHROW(expr) DOCTEST_NOTHROW_IMPL((expr), true)

// ---- main() ----
#ifdef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
int main() { return ::doctest::detail::runAll(); }
#endif
