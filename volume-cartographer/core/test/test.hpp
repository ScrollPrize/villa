#pragma once

//
// Minimal GTest-compatible test header.
// Supports: TEST(suite, name), EXPECT_TRUE/FALSE, EXPECT_EQ/NE,
//           EXPECT_FLOAT_EQ, EXPECT_NEAR, ASSERT_* variants.
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace vc_test {

struct Test {
    const char* suite;
    const char* name;
    void (*fn)();
};

inline std::vector<Test>& tests()
{
    static std::vector<Test> t;
    return t;
}

inline int& fail_count()
{
    static int n = 0;
    return n;
}

struct Register {
    Register(const char* suite, const char* name, void (*fn)())
    {
        tests().push_back({suite, name, fn});
    }
};

} // namespace vc_test

// ---- test registration ------------------------------------------------------

#define TEST(suite, name)                                                    \
    static void vc_test_##suite##_##name();                                  \
    static ::vc_test::Register vc_reg_##suite##_##name(                      \
        #suite, #name, vc_test_##suite##_##name);                            \
    static void vc_test_##suite##_##name()

// ---- expect (non-fatal) -----------------------------------------------------

#define EXPECT_TRUE(expr)                                                    \
    do {                                                                     \
        if (!(expr)) {                                                       \
            std::fprintf(stderr, "  FAIL %s:%d: EXPECT_TRUE(%s)\n",         \
                         __FILE__, __LINE__, #expr);                         \
            ++::vc_test::fail_count();                                       \
        }                                                                    \
    } while (0)

#define EXPECT_FALSE(expr) EXPECT_TRUE(!(expr))

#define EXPECT_EQ(a, b)                                                      \
    do {                                                                     \
        if (!((a) == (b))) {                                                 \
            std::fprintf(stderr, "  FAIL %s:%d: EXPECT_EQ(%s, %s)\n",      \
                         __FILE__, __LINE__, #a, #b);                        \
            ++::vc_test::fail_count();                                       \
        }                                                                    \
    } while (0)

#define EXPECT_NE(a, b)                                                      \
    do {                                                                     \
        if (!((a) != (b))) {                                                 \
            std::fprintf(stderr, "  FAIL %s:%d: EXPECT_NE(%s, %s)\n",      \
                         __FILE__, __LINE__, #a, #b);                        \
            ++::vc_test::fail_count();                                       \
        }                                                                    \
    } while (0)

#define EXPECT_LT(a, b)                                                      \
    do {                                                                     \
        if (!((a) < (b))) {                                                  \
            std::fprintf(stderr, "  FAIL %s:%d: EXPECT_LT(%s, %s)\n",      \
                         __FILE__, __LINE__, #a, #b);                        \
            ++::vc_test::fail_count();                                       \
        }                                                                    \
    } while (0)

#define EXPECT_GT(a, b)                                                      \
    do {                                                                     \
        if (!((a) > (b))) {                                                  \
            std::fprintf(stderr, "  FAIL %s:%d: EXPECT_GT(%s, %s)\n",      \
                         __FILE__, __LINE__, #a, #b);                        \
            ++::vc_test::fail_count();                                       \
        }                                                                    \
    } while (0)

#define EXPECT_LE(a, b)                                                      \
    do {                                                                     \
        if (!((a) <= (b))) {                                                 \
            std::fprintf(stderr, "  FAIL %s:%d: EXPECT_LE(%s, %s)\n",      \
                         __FILE__, __LINE__, #a, #b);                        \
            ++::vc_test::fail_count();                                       \
        }                                                                    \
    } while (0)

#define EXPECT_GE(a, b)                                                      \
    do {                                                                     \
        if (!((a) >= (b))) {                                                 \
            std::fprintf(stderr, "  FAIL %s:%d: EXPECT_GE(%s, %s)\n",      \
                         __FILE__, __LINE__, #a, #b);                        \
            ++::vc_test::fail_count();                                       \
        }                                                                    \
    } while (0)

#define EXPECT_NEAR(a, b, eps)                                               \
    do {                                                                     \
        auto va_ = (a); auto vb_ = (b);                                     \
        if (std::fabs(double(va_) - double(vb_)) > double(eps)) {           \
            std::fprintf(stderr,                                             \
                "  FAIL %s:%d: EXPECT_NEAR(%s, %s, %s) got %g vs %g\n",    \
                __FILE__, __LINE__, #a, #b, #eps,                            \
                double(va_), double(vb_));                                   \
            ++::vc_test::fail_count();                                       \
        }                                                                    \
    } while (0)

#define EXPECT_FLOAT_EQ(a, b) EXPECT_NEAR(a, b, 1e-6)

// ---- assert (fatal — aborts current test) -----------------------------------

#define ASSERT_TRUE(expr)                                                    \
    do {                                                                     \
        if (!(expr)) {                                                       \
            std::fprintf(stderr, "  FAIL %s:%d: ASSERT_TRUE(%s)\n",         \
                         __FILE__, __LINE__, #expr);                         \
            ++::vc_test::fail_count();                                       \
            return;                                                          \
        }                                                                    \
    } while (0)

#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))
#define ASSERT_EQ(a, b) do { if (!((a) == (b))) { \
    std::fprintf(stderr, "  FAIL %s:%d: ASSERT_EQ(%s, %s)\n", __FILE__, __LINE__, #a, #b); \
    ++::vc_test::fail_count(); return; } } while (0)
#define ASSERT_NE(a, b) do { if (!((a) != (b))) { \
    std::fprintf(stderr, "  FAIL %s:%d: ASSERT_NE(%s, %s)\n", __FILE__, __LINE__, #a, #b); \
    ++::vc_test::fail_count(); return; } } while (0)

// ---- runner -----------------------------------------------------------------

inline int vc_test_main()
{
    int passed = 0, failed = 0;
    for (auto& t : ::vc_test::tests()) {
        ::vc_test::fail_count() = 0;
        t.fn();
        if (::vc_test::fail_count() == 0) {
            std::printf("  PASS  %s.%s\n", t.suite, t.name);
            ++passed;
        } else {
            std::printf("  FAIL  %s.%s\n", t.suite, t.name);
            ++failed;
        }
    }
    std::printf("\n%d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}

int main() { return vc_test_main(); }
