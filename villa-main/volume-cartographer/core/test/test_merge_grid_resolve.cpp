// Unit tests for the merge.json grid parser
// (vc::merge::gmResolveGrid / gmCheckConnected).
//
// Each case stages an empty <tmp>/paths/<name>/ directory tree so the
// parser's is_directory() check passes, writes a minimal merge.json,
// and asserts the resulting surface/edge sets.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc_merge_tifxyz_grid.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using vc::merge::GMSurfaceSpec;
using vc::merge::GMEdgeSpec;

namespace {

struct TmpPaths {
    fs::path root;
    fs::path pathsDir;

    explicit TmpPaths(const std::string& tag)
    {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        root = fs::temp_directory_path() /
               ("vc_merge_grid_" + tag + "_" + std::to_string(rng()));
        pathsDir = root / "paths";
        fs::create_directories(pathsDir);
    }
    ~TmpPaths()
    {
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    void mkSurface(const std::string& name)
    {
        fs::create_directories(pathsDir / name);
    }

    fs::path writeMerge(const std::string& body)
    {
        const fs::path p = root / "merge.json";
        std::ofstream f(p);
        f << body;
        return p;
    }
};

bool hasEdge(const std::vector<GMEdgeSpec>& edges,
             const std::string& a,
             const std::string& b)
{
    auto canon = [](std::string x, std::string y) {
        if (x > y) std::swap(x, y);
        return std::pair{x, y};
    };
    const auto target = canon(a, b);
    return std::any_of(edges.begin(), edges.end(),
        [&](const GMEdgeSpec& e) { return canon(e.a, e.b) == target; });
}

}

TEST_CASE("2x2 valid grid yields 4 surfaces and 4 (2H+2V) edges")
{
    TmpPaths t{"twoxtwo"};
    for (auto n : {"a", "b", "c", "d"}) t.mkSurface(n);
    const auto mj = t.writeMerge(R"({"rows":["a b","c d"]})");

    std::vector<GMSurfaceSpec> surfaces;
    std::vector<GMEdgeSpec> edges;
    REQUIRE_NOTHROW(vc::merge::gmResolveGrid(mj, t.pathsDir, surfaces, edges));

    CHECK(surfaces.size() == 4);
    CHECK(edges.size() == 4);
    CHECK(hasEdge(edges, "a", "b"));
    CHECK(hasEdge(edges, "c", "d"));
    CHECK(hasEdge(edges, "a", "c"));
    CHECK(hasEdge(edges, "b", "d"));
    CHECK_FALSE(hasEdge(edges, "a", "d")); // no diagonals
}

TEST_CASE("Ragged rows: vertical edges only over min(C0,C1) columns")
{
    TmpPaths t{"ragged"};
    for (auto n : {"a", "b", "c", "d", "e"}) t.mkSurface(n);
    // row0 has 3 cells (a,b,c), row1 has 2 (d,e). Vertical edges are
    // a-d and b-e; c has no vertical neighbour.
    const auto mj = t.writeMerge(R"({"rows":["a b c","d e"]})");

    std::vector<GMSurfaceSpec> surfaces;
    std::vector<GMEdgeSpec> edges;
    REQUIRE_NOTHROW(vc::merge::gmResolveGrid(mj, t.pathsDir, surfaces, edges));

    CHECK(surfaces.size() == 5);
    CHECK(hasEdge(edges, "a", "b"));
    CHECK(hasEdge(edges, "b", "c"));
    CHECK(hasEdge(edges, "d", "e"));
    CHECK(hasEdge(edges, "a", "d"));
    CHECK(hasEdge(edges, "b", "e"));
    CHECK_FALSE(hasEdge(edges, "c", "e"));
    CHECK(edges.size() == 5);
}

TEST_CASE("Empty cells (null and \"\") drop their edges")
{
    TmpPaths t{"empty"};
    for (auto n : {"a", "b", "c"}) t.mkSurface(n);
    // null in row0 col1, "" in row1 col1: both forms must skip edges
    // touching that slot.
    const auto mj = t.writeMerge(
        R"({"rows":[["a", null, "b"],["c", "", "b"]]})");

    std::vector<GMSurfaceSpec> surfaces;
    std::vector<GMEdgeSpec> edges;
    REQUIRE_NOTHROW(vc::merge::gmResolveGrid(mj, t.pathsDir, surfaces, edges));

    CHECK(surfaces.size() == 3); // a, b, c
    // Column-0 vertical: a-c. Row-1 horizontal: c-(empty)-b -> c not
    // adjacent to b through the empty cell. b is in both rows col2 so
    // self-edge is suppressed.
    CHECK(hasEdge(edges, "a", "c"));
    CHECK_FALSE(hasEdge(edges, "a", "b")); // separated by empty cell
    CHECK_FALSE(hasEdge(edges, "c", "b")); // separated by empty cell
    CHECK_FALSE(hasEdge(edges, "b", "b")); // no self-edge
}

TEST_CASE("Duplicate cell name dedups surface and avoids self-edge")
{
    TmpPaths t{"dup"};
    for (auto n : {"a", "b", "c"}) t.mkSurface(n);
    // 'b' appears in (0,1) and (0,2) -- adjacent cells with the SAME
    // name produce no self-edge but the rest of the grid still wires
    // up through it.
    const auto mj = t.writeMerge(R"({"rows":["a b b","a b b"]})");

    std::vector<GMSurfaceSpec> surfaces;
    std::vector<GMEdgeSpec> edges;
    REQUIRE_NOTHROW(vc::merge::gmResolveGrid(mj, t.pathsDir, surfaces, edges));

    CHECK(surfaces.size() == 2); // a and b
    CHECK(hasEdge(edges, "a", "b"));
    // The self-edge b-b in (0,1)-(0,2) and (1,1)-(1,2) and the vertical
    // b-b along col1/col2 must not appear.
    for (const auto& e : edges) CHECK(e.a != e.b);
}

TEST_CASE("Disconnected graph throws from gmCheckConnected")
{
    TmpPaths t{"disc"};
    for (auto n : {"a", "b", "c", "d"}) t.mkSurface(n);
    // Two disjoint 2-cell rows separated by a row of empties.
    const auto mj = t.writeMerge(
        R"({"rows":["a b",["",""], "c d"]})");

    std::vector<GMSurfaceSpec> surfaces;
    std::vector<GMEdgeSpec> edges;
    REQUIRE_NOTHROW(vc::merge::gmResolveGrid(mj, t.pathsDir, surfaces, edges));
    CHECK(surfaces.size() == 4);
    CHECK_THROWS_WITH_AS(vc::merge::gmCheckConnected(surfaces, edges),
                         doctest::Contains("disconnected"),
                         std::runtime_error);
}

TEST_CASE("Fewer than 2 distinct surfaces throws")
{
    TmpPaths t{"small"};
    t.mkSurface("only");
    const auto mj = t.writeMerge(R"({"rows":["only only","only only"]})");

    std::vector<GMSurfaceSpec> surfaces;
    std::vector<GMEdgeSpec> edges;
    CHECK_THROWS_WITH_AS(vc::merge::gmResolveGrid(mj, t.pathsDir, surfaces, edges),
                         doctest::Contains("at least 2 distinct"),
                         std::runtime_error);
}

TEST_CASE("Missing tifxyz directory throws with the offending path")
{
    TmpPaths t{"missing"};
    t.mkSurface("a"); // 'b' deliberately not created
    const auto mj = t.writeMerge(R"({"rows":["a b"]})");

    std::vector<GMSurfaceSpec> surfaces;
    std::vector<GMEdgeSpec> edges;
    CHECK_THROWS_WITH_AS(vc::merge::gmResolveGrid(mj, t.pathsDir, surfaces, edges),
                         doctest::Contains("'b'"),
                         std::runtime_error);
}
