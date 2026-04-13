#include <utils/test.hpp>
#include <utils/hash.hpp>
#include <string>
#include <string_view>
#include <unordered_set>

using namespace utils;

// ============================================================================
// Compile-time tests
// ============================================================================

// Compile-time literal (uses its own consteval implementation)
static_assert("test"_hash != 0);
static_assert("a"_hash != "b"_hash);
static_assert("hello"_hash == "hello"_hash, "deterministic");
static_assert("hello"_hash != "world"_hash, "different inputs differ");

// hash_combine determinism
static_assert(hash_combine(0, 42) == hash_combine(0, 42));
static_assert(hash_combine(0, 42) != hash_combine(0, 43));
static_assert(hash_combine(0, 42) != hash_combine(1, 42), "seed matters");

// fnv1a_value with integers
static_assert(fnv1a_value(42) == fnv1a_value(42));
static_assert(fnv1a_value(42) != fnv1a_value(43));

// ============================================================================
// Runtime tests
// ============================================================================

TEST_CASE("FNV-1a: basic consistency") {
    auto h1 = fnv1a("hello");
    auto h2 = fnv1a("hello");
    REQUIRE_EQ(h1, h2);

    auto h3 = fnv1a("world");
    REQUIRE_NE(h1, h3);
}

TEST_CASE("FNV-1a: known value for empty string") {
    REQUIRE_EQ(fnv1a(""), fnv_offset_basis);
}

TEST_CASE("FNV-1a: raw bytes interface") {
    const char data[] = "test";
    auto h1 = fnv1a(data, 4);
    auto h2 = fnv1a(std::string_view("test"));
    REQUIRE_EQ(h1, h2);
}

TEST_CASE("FNV-1a: distribution - no collisions in small set") {
    // Hash a set of short strings and verify no collisions
    std::unordered_set<std::uint64_t> hashes;
    const char* words[] = {
        "alpha", "beta", "gamma", "delta", "epsilon",
        "zeta", "eta", "theta", "iota", "kappa",
        "lambda", "mu", "nu", "xi", "omicron",
        "pi", "rho", "sigma", "tau", "upsilon"
    };
    for (auto w : words) {
        auto [it, inserted] = hashes.insert(fnv1a(w));
        REQUIRE(inserted);
    }
    REQUIRE_EQ(hashes.size(), std::size_t{20});
}

TEST_CASE("FNV-1a: single character variations") {
    // Verify that single-character differences produce different hashes
    auto ha = fnv1a("a");
    auto hb = fnv1a("b");
    auto hc = fnv1a("c");
    REQUIRE_NE(ha, hb);
    REQUIRE_NE(hb, hc);
    REQUIRE_NE(ha, hc);
}

TEST_CASE("fnv1a_value: trivially copyable types") {
    REQUIRE_EQ(fnv1a_value(100), fnv1a_value(100));
    REQUIRE_NE(fnv1a_value(100), fnv1a_value(101));

    REQUIRE_EQ(fnv1a_value(3.14), fnv1a_value(3.14));
    REQUIRE_NE(fnv1a_value(3.14), fnv1a_value(2.71));
}

TEST_CASE("hash_combine: determinism and sensitivity") {
    auto a = hash_combine(0, 1);
    auto b = hash_combine(0, 2);
    REQUIRE_NE(a, b);

    // Same call gives same result
    REQUIRE_EQ(hash_combine(123, 456), hash_combine(123, 456));

    // Order matters
    REQUIRE_NE(hash_combine(1, 2), hash_combine(2, 1));
}

TEST_CASE("hash_combine_values: variadic") {
    auto h1 = hash_combine_values(1, 2, 3);
    auto h2 = hash_combine_values(1, 2, 3);
    REQUIRE_EQ(h1, h2);

    auto h3 = hash_combine_values(1, 2, 4);
    REQUIRE_NE(h1, h3);

    // Different argument count
    auto h4 = hash_combine_values(1, 2);
    REQUIRE_NE(h1, h4);
}

TEST_CASE("Hasher: std-hashable types") {
    Hasher h;
    // int
    auto hi1 = h(42);
    auto hi2 = h(42);
    REQUIRE_EQ(hi1, hi2);
    REQUIRE_NE(h(42), h(43));

    // std::string
    auto hs1 = h(std::string("hello"));
    auto hs2 = h(std::string("hello"));
    REQUIRE_EQ(hs1, hs2);
    REQUIRE_NE(h(std::string("hello")), h(std::string("world")));
}

TEST_CASE("Hasher: pair hashing") {
    Hasher h;
    auto hp1 = h(std::pair{1, 2});
    auto hp2 = h(std::pair{1, 2});
    REQUIRE_EQ(hp1, hp2);

    REQUIRE_NE(h(std::pair{1, 2}), h(std::pair{2, 1}));
    REQUIRE_NE(h(std::pair{1, 2}), h(std::pair{1, 3}));
}

TEST_CASE("Hasher: tuple hashing") {
    Hasher h;
    auto ht1 = h(std::tuple{1, 2.0, 3});
    auto ht2 = h(std::tuple{1, 2.0, 3});
    REQUIRE_EQ(ht1, ht2);

    REQUIRE_NE(h(std::tuple{1, 2.0, 3}), h(std::tuple{1, 2.0, 4}));
}

TEST_CASE("Hasher: array hashing") {
    Hasher h;
    std::array<int, 3> a{1, 2, 3};
    std::array<int, 3> b{1, 2, 3};
    std::array<int, 3> c{1, 2, 4};

    REQUIRE_EQ(h(a), h(b));
    REQUIRE_NE(h(a), h(c));
}

TEST_CASE("Compile-time _hash literal matches runtime") {
    constexpr auto ct = "foobar"_hash;
    auto rt = fnv1a("foobar");
    REQUIRE_EQ(ct, rt);
}

TEST_CASE("_hash literal: various strings") {
    // Verify several compile-time hashes are distinct
    constexpr auto h1 = "one"_hash;
    constexpr auto h2 = "two"_hash;
    constexpr auto h3 = "three"_hash;
    REQUIRE_NE(h1, h2);
    REQUIRE_NE(h2, h3);
    REQUIRE_NE(h1, h3);
}

TEST_CASE("Hasher: transparent hashing") {
    // Hasher is marked is_transparent -- verify string_view and string hash equally
    Hasher h;
    std::string s = "hello";
    std::string_view sv = s;
    REQUIRE_EQ(h(s), h(sv));
}

UTILS_TEST_MAIN()
