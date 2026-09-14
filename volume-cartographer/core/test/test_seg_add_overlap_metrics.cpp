#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "../../apps/src/vc_seg_add_overlap_metrics.hpp"

#include <cmath>

using vc_seg_overlap::SourceMatches;
using vc_seg_overlap::SurfaceInfo;

TEST_CASE("pair report is directional, sparse, and deterministic")
{
    std::vector<SurfaceInfo> targets{
        {"/z", "z", "z"},
        {"/b", "b", "b"},
        {"/a", "a", "a"},
    };
    std::vector<SourceMatches> sources{
        {{"/b", "b", "b"}, 6, 4, {{"/a", 3}, {"/z", 1}}},
        {{"/a", "a", "a"}, 2, 2, {{"/b", 2}}},
    };

    const utils::Json report = vc_seg_overlap::buildReport(
        std::move(sources), std::move(targets), 2, 2.0f);

    CHECK(report["schema_version"].get_int() == 1);
    CHECK(report["parameters"]["point_stride"].get_int() == 2);
    CHECK(report["source_count"].get_int() == 2);
    CHECK(report["target_count"].get_int() == 3);
    CHECK(report["directed_overlap_pair_count"].get_int() == 3);
    CHECK(report["self_pairs_excluded"].get_bool());
    CHECK(report["zero_match_pairs_omitted"].get_bool());

    const utils::Json& targetRows = report["targets"];
    REQUIRE(targetRows.size() == 3);
    CHECK(targetRows[size_t(0)]["target_path"].get_string() == "a");
    CHECK(targetRows[size_t(1)]["target_path"].get_string() == "b");
    CHECK(targetRows[size_t(2)]["target_path"].get_string() == "z");

    const utils::Json& sourceRows = report["sources"];
    REQUIRE(sourceRows.size() == 2);

    const utils::Json& a = sourceRows[size_t(0)];
    CHECK(a["source_id"].get_string() == "a");
    CHECK(a["source_path"].get_string() == "a");
    CHECK(a["valid_source_points"].get_int() == 2);
    CHECK(a["queried_source_points"].get_int() == 2);
    REQUIRE(a["hits"].size() == 1);
    CHECK(a["hits"][size_t(0)]["target_id"].get_string() == "b");
    CHECK(a["hits"][size_t(0)]["target_path"].get_string() == "b");
    CHECK(a["hits"][size_t(0)]["matched_source_points"].get_int() == 2);
    CHECK(std::fabs(a["hits"][size_t(0)]["source_coverage_fraction"].get_float() - 1.0) < 1e-12);

    const utils::Json& b = sourceRows[size_t(1)];
    CHECK(b["source_id"].get_string() == "b");
    CHECK(b["valid_source_points"].get_int() == 6);
    CHECK(b["queried_source_points"].get_int() == 4);
    REQUIRE(b["hits"].size() == 2);
    CHECK(b["hits"][size_t(0)]["target_id"].get_string() == "a");
    CHECK(std::fabs(b["hits"][size_t(0)]["source_coverage_fraction"].get_float() - 0.75) < 1e-12);
    CHECK(b["hits"][size_t(1)]["target_id"].get_string() == "z");
    CHECK(std::fabs(b["hits"][size_t(1)]["source_coverage_fraction"].get_float() - 0.25) < 1e-12);
}

TEST_CASE("an empty source produces no positive hits")
{
    const utils::Json report = vc_seg_overlap::buildReport(
        {{{"/source", "source", "."}, 0, 0, {}}},
        {{"/target", "target", "."}},
        1,
        2.0f);

    const utils::Json& source = report["sources"][size_t(0)];
    CHECK(source["valid_source_points"].get_int() == 0);
    CHECK(source["queried_source_points"].get_int() == 0);
    CHECK(source["hits"].size() == 0);
}

TEST_CASE("pair report rejects impossible point counts")
{
    CHECK_THROWS_AS(
        vc_seg_overlap::buildReport(
            {{{"/source", "source", "."}, 1, 2, {}}},
            {{"/target", "target", "."}},
            1,
            2.0f),
        std::invalid_argument);

    CHECK_THROWS_AS(
        vc_seg_overlap::buildReport(
            {{{"/source", "source", "."}, 2, 2, {{"/target", 3}}}},
            {{"/target", "target", "."}},
            1,
            2.0f),
        std::invalid_argument);
}
