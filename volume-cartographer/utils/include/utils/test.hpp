#pragma once
#include <vector>
#include <string>
#include <string_view>
#include <functional>
#include <source_location>
#include <format>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <concepts>
#include <type_traits>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <atomic>

namespace utils::test {

// ---------------------------------------------------------------------------
// Portable println (std::print/println requires libstdc++ 14+)
// ---------------------------------------------------------------------------

template<typename... Args>
void println(std::FILE* f, std::format_string<Args...> fmt, Args&&... args) {
    auto s = std::format(fmt, std::forward<Args>(args)...);
    std::fwrite(s.data(), 1, s.size(), f);
    std::fputc('\n', f);
}

template<typename... Args>
void println(std::format_string<Args...> fmt, Args&&... args) {
    println(stdout, fmt, std::forward<Args>(args)...);
}

// ---------------------------------------------------------------------------
// Concepts
// ---------------------------------------------------------------------------

template<typename T>
concept Printable = std::formattable<T, char>;

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

namespace color {
    inline constexpr const char* green  = "\033[32m";
    inline constexpr const char* red    = "\033[31m";
    inline constexpr const char* yellow = "\033[33m";
    inline constexpr const char* reset  = "\033[0m";
    inline constexpr const char* bold   = "\033[1m";
} // namespace color

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

struct TestCase {
    std::string_view name;
    std::string_view file;
    int line;
    std::function<void()> func;
};

struct TestFailure {};

struct Context {
    std::atomic<int> passed{0};
    std::atomic<int> failed{0};
    std::atomic<int> checks{0};
    bool current_failed  = false;
    bool use_color       = true;
    bool verbose         = false;
    std::string filter;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

inline Context& ctx() {
    static Context c;
    return c;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

inline const char* col(const char* code) {
    return ctx().use_color ? code : "";
}

template<typename T>
std::string to_string_val(const T& v) {
    if constexpr (Printable<T>) {
        return std::format("{}", v);
    } else {
        return "<non-printable>";
    }
}

// ---------------------------------------------------------------------------
// Assertion reporting
// ---------------------------------------------------------------------------

inline void fail_assert(const char* expr,
                        std::string_view lhs,
                        std::string_view rhs,
                        std::source_location loc) {
    ctx().current_failed = true;
    println(stderr, "    {}{}:{}{}: {}REQUIRE/CHECK({}) failed{}",
                 col(color::bold), loc.file_name(), col(color::reset),
                 loc.line(),
                 col(color::red), expr, col(color::reset));
    if (!lhs.empty() || !rhs.empty()) {
        println(stderr, "      lhs = {}", lhs);
        println(stderr, "      rhs = {}", rhs);
    }
}

template<typename A, typename B>
void fail_cmp(const char* expr, const A& a, const B& b,
              std::source_location loc) {
    fail_assert(expr, to_string_val(a), to_string_val(b), loc);
}

inline void pass_assert(const char* expr, std::source_location loc) {
    ctx().checks++;
    if (ctx().verbose) {
        println("    {}PASS{}: {} ({}:{})",
                     col(color::green), col(color::reset),
                     expr, loc.file_name(), loc.line());
    }
}

// ---------------------------------------------------------------------------
// Auto-registration
// ---------------------------------------------------------------------------

struct AutoRegister {
    AutoRegister(std::string_view name, std::function<void()> func,
                 std::source_location loc = std::source_location::current()) {
        registry().push_back({name, loc.file_name(), static_cast<int>(loc.line()), std::move(func)});
    }
};

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

inline int run_all(int argc = 0, const char** argv = nullptr) {
    auto& c = ctx();
    c.passed = 0;
    c.failed = 0;
    c.checks = 0;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg.starts_with("--filter=")) {
            c.filter = std::string(arg.substr(9));
        } else if (arg == "--no-color") {
            c.use_color = false;
        } else if (arg == "--verbose") {
            c.verbose = true;
        } else if (arg == "--list") {
            for (auto& tc : registry())
                println("{}", tc.name);
            return 0;
        }
    }

    // Collect matching tests
    std::vector<TestCase*> to_run;
    for (auto& tc : registry()) {
        if (c.filter.empty() ||
            std::string_view(tc.name).find(c.filter) != std::string_view::npos) {
            to_run.push_back(&tc);
        }
    }

    println("{}[==========]{} Running {} test{}",
                 col(color::bold), col(color::reset),
                 to_run.size(), to_run.size() == 1 ? "" : "s");

    auto wall_start = std::chrono::high_resolution_clock::now();

    for (auto* tc : to_run) {
        println("{}[ RUN      ]{} {}", col(color::green), col(color::reset), tc->name);
        c.current_failed = false;
        auto t0 = std::chrono::high_resolution_clock::now();

        try {
            tc->func();
        } catch (const TestFailure&) {
            // Fatal assertion -- already recorded
        } catch (const std::exception& e) {
            c.current_failed = true;
            println(stderr, "    {}Unhandled exception{}: {}", col(color::red), col(color::reset), e.what());
        } catch (...) {
            c.current_failed = true;
            println(stderr, "    {}Unhandled unknown exception{}", col(color::red), col(color::reset));
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (c.current_failed) {
            c.failed++;
            println("{}[  FAILED  ]{} {} ({:.1f}ms)", col(color::red), col(color::reset), tc->name, ms);
        } else {
            c.passed++;
            println("{}[       OK ]{} {} ({:.1f}ms)", col(color::green), col(color::reset), tc->name, ms);
        }
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    const int p = c.passed.load();
    const int f = c.failed.load();
    println("{}[==========]{} {}{} passed{}, {}{} failed{} ({:.1f}ms total)",
                 col(color::bold), col(color::reset),
                 col(color::green), p, col(color::reset),
                 f ? col(color::red) : col(color::green), f, col(color::reset),
                 total_ms);

    return f > 0 ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Static assertion support
// ---------------------------------------------------------------------------

#define STATIC_REQUIRE(expr) static_assert(expr, "STATIC_REQUIRE(" #expr ") failed")

} // namespace utils::test

// ===========================================================================
// Macros  (must be outside namespace)
// ===========================================================================

// Unique identifier helpers
#define UTILS_TEST_CAT2(a, b) a##b
#define UTILS_TEST_CAT(a, b) UTILS_TEST_CAT2(a, b)

#ifdef __COUNTER__
#define UTILS_TEST_UID(prefix) UTILS_TEST_CAT(prefix, __COUNTER__)
#else
#define UTILS_TEST_UID(prefix) UTILS_TEST_CAT(prefix, __LINE__)
#endif

// ---------------------------------------------------------------------------
// TEST_CASE
// ---------------------------------------------------------------------------

#define TEST_CASE(tname)                                                       \
    static void UTILS_TEST_UID(utils_test_func_)();                          \
    static ::utils::test::AutoRegister UTILS_TEST_UID(utils_test_reg_)(     \
        tname, UTILS_TEST_UID(utils_test_func_));                            \
    static void UTILS_TEST_UID(utils_test_func_)()

// Note: The above relies on __COUNTER__ incrementing identically across the
// three expansions within a single macro invocation.  On all major compilers
// (GCC, Clang, MSVC) __COUNTER__ is expanded once per *top-level* macro
// invocation when it appears in the replacement list of a nested macro, so
// UTILS_TEST_UID yields the same value for all three uses.
//
// However, if a compiler expands __COUNTER__ per-occurrence inside the nested
// helper, the identifiers would mismatch.  The alternative below avoids that
// entirely by using __LINE__ (safe as long as you don't put two TEST_CASEs on
// the same line):

#undef TEST_CASE
#define TEST_CASE(tname)                                                       \
    static void UTILS_TEST_CAT(utils_test_func_, __LINE__)();                \
    static ::utils::test::AutoRegister                                        \
        UTILS_TEST_CAT(utils_test_reg_, __LINE__)(                           \
            tname, UTILS_TEST_CAT(utils_test_func_, __LINE__));              \
    static void UTILS_TEST_CAT(utils_test_func_, __LINE__)()

// ---------------------------------------------------------------------------
// SECTION
// ---------------------------------------------------------------------------

#define SECTION(sname)                                                         \
    if (::utils::test::ctx().verbose)                                         \
        ::utils::test::println("  {}-- {}{}", ::utils::test::col(::utils::test::color::yellow), sname, \
                     ::utils::test::col(::utils::test::color::reset));       \
    if (true)

// ---------------------------------------------------------------------------
// REQUIRE / CHECK  (boolean)
// ---------------------------------------------------------------------------

#define REQUIRE(expr)                                                          \
    do {                                                                        \
        if (!(expr)) {                                                          \
            ::utils::test::fail_assert(#expr, "", "",                          \
                std::source_location::current());                               \
            throw ::utils::test::TestFailure{};                                \
        }                                                                       \
        ::utils::test::pass_assert(#expr, std::source_location::current());    \
    } while (0)

#define CHECK(expr)                                                            \
    do {                                                                        \
        if (!(expr)) {                                                          \
            ::utils::test::fail_assert(#expr, "", "",                          \
                std::source_location::current());                               \
        } else {                                                                \
            ::utils::test::pass_assert(#expr, std::source_location::current());\
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Comparison macros  (generic helper)
// ---------------------------------------------------------------------------

#define UTILS_CMP_ASSERT(a, b, op, fatal)                                     \
    do {                                                                        \
        const auto _utils_a = (a);                                             \
        const auto _utils_b = (b);                                            \
        if (!(_utils_a op _utils_b)) {                                        \
            ::utils::test::fail_cmp(#a " " #op " " #b,                        \
                _utils_a, _utils_b, std::source_location::current());         \
            if constexpr (fatal) throw ::utils::test::TestFailure{};           \
        } else {                                                                \
            ::utils::test::pass_assert(#a " " #op " " #b,                     \
                std::source_location::current());                               \
        }                                                                       \
    } while (0)

#define REQUIRE_EQ(a, b) UTILS_CMP_ASSERT(a, b, ==, true)
#define CHECK_EQ(a, b)   UTILS_CMP_ASSERT(a, b, ==, false)
#define REQUIRE_NE(a, b) UTILS_CMP_ASSERT(a, b, !=, true)
#define CHECK_NE(a, b)   UTILS_CMP_ASSERT(a, b, !=, false)
#define REQUIRE_LT(a, b) UTILS_CMP_ASSERT(a, b, <,  true)
#define CHECK_LT(a, b)   UTILS_CMP_ASSERT(a, b, <,  false)
#define REQUIRE_GT(a, b) UTILS_CMP_ASSERT(a, b, >,  true)
#define CHECK_GT(a, b)   UTILS_CMP_ASSERT(a, b, >,  false)
#define REQUIRE_LE(a, b) UTILS_CMP_ASSERT(a, b, <=, true)
#define CHECK_LE(a, b)   UTILS_CMP_ASSERT(a, b, <=, false)
#define REQUIRE_GE(a, b) UTILS_CMP_ASSERT(a, b, >=, true)
#define CHECK_GE(a, b)   UTILS_CMP_ASSERT(a, b, >=, false)

// ---------------------------------------------------------------------------
// REQUIRE_NEAR / CHECK_NEAR  (floating point)
// ---------------------------------------------------------------------------

#define UTILS_NEAR_ASSERT(a, b, eps, fatal)                                   \
    do {                                                                        \
        const auto _utils_a = static_cast<double>(a);                          \
        const auto _utils_b = static_cast<double>(b);                          \
        const auto _utils2_e = static_cast<double>(eps);                        \
        if (std::fabs(_utils_a - _utils_b) > _utils2_e) {                    \
            ::utils::test::fail_assert(                                        \
                #a " ~= " #b " (eps=" #eps ")",                                \
                ::utils::test::to_string_val(_utils_a),                       \
                ::utils::test::to_string_val(_utils_b),                       \
                std::source_location::current());                               \
            if constexpr (fatal) throw ::utils::test::TestFailure{};           \
        } else {                                                                \
            ::utils::test::pass_assert(#a " ~= " #b,                          \
                std::source_location::current());                               \
        }                                                                       \
    } while (0)

#define REQUIRE_NEAR(a, b, eps) UTILS_NEAR_ASSERT(a, b, eps, true)
#define CHECK_NEAR(a, b, eps)   UTILS_NEAR_ASSERT(a, b, eps, false)

// ---------------------------------------------------------------------------
// REQUIRE_THROWS / CHECK_THROWS
// ---------------------------------------------------------------------------

#define REQUIRE_THROWS(expr)                                                   \
    do {                                                                        \
        bool _utils2_threw = false;                                             \
        try { (void)(expr); } catch (...) { _utils2_threw = true; }            \
        if (!_utils2_threw) {                                                   \
            ::utils::test::fail_assert(#expr " throws", "", "",               \
                std::source_location::current());                               \
            throw ::utils::test::TestFailure{};                                \
        }                                                                       \
        ::utils::test::pass_assert(#expr " throws",                           \
            std::source_location::current());                                   \
    } while (0)

#define CHECK_THROWS(expr)                                                     \
    do {                                                                        \
        bool _utils2_threw = false;                                             \
        try { (void)(expr); } catch (...) { _utils2_threw = true; }            \
        if (!_utils2_threw) {                                                   \
            ::utils::test::fail_assert(#expr " throws", "", "",               \
                std::source_location::current());                               \
        } else {                                                                \
            ::utils::test::pass_assert(#expr " throws",                       \
                std::source_location::current());                               \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// BENCHMARK
// ---------------------------------------------------------------------------

#define BENCHMARK(bname, iterations)                                           \
    {                                                                           \
        auto _utils_bstart = std::chrono::high_resolution_clock::now();        \
        for (long _utils2_i = 0; _utils2_i < static_cast<long>(iterations);    \
             ++_utils2_i)

// Usage:  BENCHMARK("name", N) { body; } BENCHMARK_END;
// Or the self-closing form via a helper:
#define BENCHMARK_END                                                          \
        auto _utils_bend = std::chrono::high_resolution_clock::now();          \
        double _utils2_ns = std::chrono::duration<double, std::nano>(           \
            _utils_bend - _utils_bstart).count();                             \
        ::utils::test::println("    {}BENCH{} {}: {:.1f} ns/op",               \
            ::utils::test::col(::utils::test::color::yellow),                 \
            ::utils::test::col(::utils::test::color::reset),                  \
            bname, _utils2_ns / static_cast<double>(iterations));               \
    }

// Convenience: single-expression benchmark
#undef BENCHMARK
#define BENCHMARK(bname, iterations)                                           \
    do {                                                                        \
        auto _utils_bstart = std::chrono::high_resolution_clock::now();        \
        for (long _utils2_i = 0; _utils2_i < static_cast<long>(iterations);    \
             ++_utils2_i) {

#undef BENCHMARK_END
#define BENCHMARK_END                                                          \
        }                                                                       \
        auto _utils_bend = std::chrono::high_resolution_clock::now();          \
        double _utils2_ns = std::chrono::duration<double, std::nano>(           \
            _utils_bend - _utils_bstart).count();                             \
        ::utils::test::println("    {}BENCH{}: {:.1f} ns/op",                  \
            ::utils::test::col(::utils::test::color::yellow),                 \
            ::utils::test::col(::utils::test::color::reset),                  \
            _utils2_ns / static_cast<double>(_utils2_i));                       \
    } while (0)

// ---------------------------------------------------------------------------
// UTILS_TEST_MAIN
// ---------------------------------------------------------------------------

#define UTILS_TEST_MAIN()                                                     \
    int main(int argc, const char** argv) {                                    \
        return ::utils::test::run_all(argc, argv);                            \
    }
