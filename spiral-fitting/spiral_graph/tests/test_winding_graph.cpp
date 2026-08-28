#include <spiral_graph/fiber_layout.hpp>
#include <spiral_graph/registration.hpp>
#include <spiral_graph/theta_topology.hpp>
#include <spiral_graph/track_io.hpp>
#include <spiral_graph/track_spatial_index.hpp>
#include <spiral_graph/winding_graph.hpp>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// Keep the test checks active in Release builds, where the standard assert
// macro would otherwise compile the test body away.
#undef assert
#define assert(condition)                                                        \
    do {                                                                         \
        if (!(condition)) {                                                      \
            throw std::runtime_error("test assertion failed: " #condition);     \
        }                                                                        \
    } while (false)

using spiral::winding::Constraint;
using spiral::winding::PatchThetaTopology;
using spiral::winding::Provenance;
using spiral::winding::WindingGraph;

namespace {

void test_graph_and_rollback()
{
    WindingGraph graph;
    graph.ensure_patch("A");
    graph.ensure_patch("B");
    graph.ensure_patch("C");
    assert(graph.add_constraint("A", "B", 2, 2).committed);
    assert(graph.add_constraint("B", "C", 3, 3).committed);
    const auto initial = graph.lifted_relative_winding("A", "C");
    assert(initial->representative == 5);
    assert(initial->period == 0);

    assert(graph.add_anchor("A", 10, 10).committed);
    assert(graph.add_anchor("C", 15, 15).committed);
    const std::size_t anchored_before = graph.constraints().size();
    const auto anchor_conflict = graph.add_anchor("C", 16, 16);
    assert(!anchor_conflict.committed);
    assert(anchor_conflict.conflict->residual == -1);
    assert(graph.constraints().size() == anchored_before);

    std::vector<Constraint> transaction{
        {graph.patch_node("A"), graph.patch_node("C"), 4, 4,
         Provenance{"test", "batch", "sheet transition", ""}, false},
        {0, graph.patch_node("C"), 16, 0,
         Provenance{"test", "batch", "bad anchor", ""}, true},
    };
    const auto rejected = graph.add_constraints(transaction);
    assert(!rejected.committed);
    assert(graph.constraints().size() == anchored_before);
    assert(graph.holonomy_count() == 0);
}

void test_persistence()
{
    WindingGraph graph;
    graph.ensure_patch("left");
    graph.ensure_patch("right");
    assert(graph.add_constraint(
        "left", "right", -7, -7,
        {"test", "cache", "edge", "round trip"}).committed);

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path directory
        = std::filesystem::temp_directory_path()
        / ("spiral-winding-test-" + std::to_string(stamp));
    std::filesystem::create_directories(directory);
    const auto cache = directory / "graph.bin";
    graph.save(cache);
    WindingGraph loaded = WindingGraph::open(cache);
    assert(loaded.stats().patch_count == 2);
    assert(loaded.stats().constraint_count == 1);
    const auto relation = loaded.lifted_relative_winding("left", "right");
    assert(relation->representative == -7);
    assert(relation->period == 0);
    assert(loaded.constraints().front().provenance.detail == "round trip");
    std::filesystem::remove_all(directory);
}

void test_lifted_holonomy()
{
    WindingGraph graph;
    graph.ensure_patch("A");
    graph.ensure_patch("B");
    graph.ensure_patch("C");
    assert(graph.add_constraint("A", "B", 2, 2).committed);
    assert(graph.add_constraint("B", "C", 3, 3).committed);

    const auto closing = graph.add_constraint(
        "A", "C", 4, 5,
        {"test", "cycle", "one turn", "lifted sheet"});
    assert(closing.committed);
    assert(closing.holonomies_added == 1);
    assert(!closing.conflict.has_value());
    assert(graph.stats().constraint_count == 3);
    assert(graph.stats().holonomy_count == 1);

    const auto lifted = graph.lifted_relative_winding("A", "C");
    assert(lifted.has_value());
    assert(lifted->representative == 5);
    assert(lifted->period == 1);

    const auto cycle = graph.holonomy(0);
    assert(cycle.reported_holonomy == 1);
    assert(cycle.geometric_holonomy == 0);
    assert(cycle.inconsistency == 1);
    assert(cycle.cycle.size() == 3);
    const std::int64_t cycle_sum = std::accumulate(
        cycle.cycle.begin(), cycle.cycle.end(), std::int64_t{0},
        [](std::int64_t value, const auto& edge) {
            return value + edge.delta;
        });
    assert(cycle_sum == 1);
    const std::int64_t geometric_cycle_sum = std::accumulate(
        cycle.cycle.begin(), cycle.cycle.end(), std::int64_t{0},
        [](std::int64_t value, const auto& edge) {
            return value + edge.geometric_delta;
        });
    assert(geometric_cycle_sum == 0);

    // A later hard failure still rolls back retained holonomies from the
    // entire transaction.
    const std::size_t before = graph.constraints().size();
    std::vector<Constraint> transaction{
        {graph.patch_node("A"), graph.patch_node("C"), 3, 3,
         Provenance{"test", "batch", "second sheet", ""}, false},
        {0, graph.patch_node("A"), 10, 0,
         Provenance{"test", "batch", "first anchor", ""}, true},
        {0, graph.patch_node("C"), 20, 0,
         Provenance{"test", "batch", "bad anchor", ""}, true},
    };
    const auto rejected = graph.add_constraints(transaction);
    assert(!rejected.committed);
    assert(graph.constraints().size() == before);
    assert(graph.holonomy_count() == 1);

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path directory
        = std::filesystem::temp_directory_path()
        / ("spiral-lifted-test-" + std::to_string(stamp));
    std::filesystem::create_directories(directory);
    const auto cache = directory / "graph.bin";
    graph.save(cache);
    WindingGraph loaded = WindingGraph::open(cache);
    assert(loaded.stats().holonomy_count == 1);
    assert(loaded.lifted_relative_winding("A", "C")->period == 1);
    std::filesystem::remove_all(directory);
}

void test_theta_topology()
{
    const std::vector<std::uint8_t> mask{1, 1, 1};
    auto topology = PatchThetaTopology::from_mask(1, 3, mask);
    const std::vector<float> theta{6.0f, 0.1f, 0.2f};
    assert(!topology.assign_theta(theta).has_value());
    assert(topology.potential(0) == 0);
    assert(topology.potential(1) == -1);
    assert(topology.potential(2) == -1);
    assert(topology.potential_at(0.25, 1.25, 6.2f) == 0);

    const std::array<std::uint8_t, 4> boundary_mask{1, 0, 0, 0};
    auto boundary = PatchThetaTopology::from_mask(2, 2, boundary_mask, true);
    const std::array<float, 1> boundary_theta{0.25f};
    assert(!boundary.assign_theta(boundary_theta));
    // A closest-point solver may return the shared edge as exactly column 1;
    // use the valid quad on the other side instead of flooring into the mask.
    assert(boundary.potential_at(0.5, 1.0, 0.25f) == 0);

    // Largest-component retention is deterministic and drops the singleton.
    const std::vector<std::uint8_t> split{
        1, 1, 0,
        0, 0, 0,
        0, 0, 1,
    };
    auto largest = PatchThetaTopology::from_mask(3, 3, split);
    assert(largest.node_count() == 2);
    assert(largest.node_at(0, 0).has_value());
    assert(!largest.node_at(2, 2).has_value());

    auto inconsistent = PatchThetaTopology::from_mask(
        2, 2, std::vector<std::uint8_t>{1, 1, 1, 1});
    const auto theta_conflict = inconsistent.assign_theta(
        std::vector<float>{0.0f, 2.0f, 6.0f, 4.0f});
    assert(theta_conflict.has_value());
    assert(theta_conflict->residual == -1);
    const std::int64_t theta_cycle_sum = std::accumulate(
        theta_conflict->cycle.begin(), theta_conflict->cycle.end(),
        std::int64_t{0}, [](std::int64_t value, const auto& edge) {
            return value + edge.delta;
        });
    assert(theta_cycle_sum == theta_conflict->residual);
}

template <typename T>
void write_values(const std::filesystem::path& path, const std::vector<T>& values)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    assert(stream.good());
}

void test_track_spatial_index()
{
    struct Header {
        char magic[8];
        std::uint32_t version;
        std::uint32_t header_size;
        std::uint64_t track_count;
        std::uint64_t point_count;
        std::uint64_t reserved[4];
    };
    static_assert(sizeof(Header) == 64);
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path()
        / ("spiral-track-index-test-" + std::to_string(stamp));
    const auto store_path = directory / "tiny.vctracks";
    const auto index_path = directory / "tiny.winding-index";
    std::filesystem::create_directories(store_path);
    Header header{{'V', 'C', 'T', 'R', 'K', '0', '1', '\0'}, 1, 64, 2, 5, {}};
    write_values(store_path / "header.bin", std::vector<Header>{header});
    write_values<std::int32_t>(store_path / "coordinates.i32", {
        1, 1, 1,
        3, 2, 1,
        8, 8, 8,
        -1, -1, -1,
        9, 8, 8,
    });
    write_values<std::int64_t>(store_path / "offsets.i64", {0, 3, 5});
    write_values<std::uint64_t>(store_path / "source_ids.u64", {10, 11});
    write_values<std::int8_t>(store_path / "family_codes.i8", {0, 1});
    write_values<double>(store_path / "arclengths.f64", {1.0, 2.0});
    write_values<double>(store_path / "tortuosities.f64", {1.0, 1.0});
    write_values<std::int32_t>(store_path / "z_bounds.i32", {1, 8, -1, 9});

    spiral::trackio::PackedTrackStore tracks(store_path);
    const auto built = spiral::trackio::TrackSpatialIndex::build(
        tracks, index_path, 4, sizeof(std::uint64_t) * 8);
    assert(!built.already_present);
    assert(built.point_count == 5);
    spiral::trackio::TrackSpatialIndex index;
    index.open(index_path);
    std::vector<std::uint64_t> points;
    index.query({0, 0, 0, 3.9f, 3.9f, 3.9f}, points);
    std::sort(points.begin(), points.end());
    assert((points == std::vector<std::uint64_t>{0, 1}));
    points.clear();
    index.query({-2, -2, -2, -0.1f, -0.1f, -0.1f}, points);
    assert((points == std::vector<std::uint64_t>{3}));
    const auto reused = spiral::trackio::TrackSpatialIndex::build(
        tracks, index_path, 4, sizeof(std::uint64_t) * 8);
    assert(reused.already_present);
    std::filesystem::remove_all(directory);
}

void test_fiber_first_layout()
{
    using json = nlohmann::json;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path directory
        = std::filesystem::temp_directory_path()
        / ("spiral-fiber-layout-test-" + std::to_string(stamp));
    const auto fibers = directory / "fibers";
    const auto cache = directory / "cache";
    std::filesystem::create_directories(fibers);
    std::filesystem::create_directories(cache);
    const auto write_fiber = [&](const std::string& name, const std::string& axis,
                                 const json& controls, const json& line,
                                 const json& branches) {
        std::ofstream stream(fibers / name);
        stream << json{
            {"type", "vc3d_fiber"},
            {"control_points", controls},
            {"line_points", line},
            {"hv_classification", {
                {"manual_tag", ""}, {"automatic_tag", axis}}},
            {"branches", branches},
        }.dump(2);
    };
    write_fiber(
        "h.json", "H", json::array({
            json::array({0, 0, 10}), json::array({10, 0, 10}),
            json::array({20, 0, 10})}),
        json::array({
            json::array({-10, 0, 10}), json::array({0, 0, 10}),
            json::array({5, 0, 10}), json::array({10, 0, 10}),
            json::array({15, 0, 10}), json::array({20, 0, 10}),
            json::array({30, 0, 10})}),
        json::array({json{
            {"control_point_index", 1}, {"branch_file", "v.json"},
            {"branch_control_point_index", 1}, {"pending", false}}}));
    write_fiber(
        "v.json", "V", json::array({
            json::array({10, 0, -10}), json::array({10, 0, 10}),
            json::array({10, 0, 30})}),
        json::array({
            json::array({10, 0, -10}), json::array({10, 0, 0}),
            json::array({10, 0, 5}), json::array({10, 0, 10}),
            json::array({10, 0, 15}), json::array({10, 0, 20}),
            json::array({10, 0, 30})}),
        json::array({json{
            {"control_point_index", 1}, {"branch_file", "h.json"},
            {"branch_control_point_index", 1}, {"pending", false}}}));
    // A pending link cannot pull this longer fiber into the selected component.
    write_fiber(
        "z-isolated.json", "H", json::array({
            json::array({0, 5, 50}), json::array({100, 5, 50})}),
        json::array({
            json::array({0, 5, 50}), json::array({100, 5, 50})}),
        json::array({json{
            {"control_point_index", 0}, {"branch_file", "h.json"},
            {"branch_control_point_index", 0}, {"pending", true}}}));
    {
        std::ofstream stream(cache / "manifest.json");
        stream << json{
            {"schema", "spiral-winding-graph"}, {"version", 2},
            {"options", {{"fiber_coordinate_scale", 1.0}}},
            {"patches", json::array()},
            {"sources", json::array({json{
                {"kind", 1}, {"paths", json::array({fibers.string()})},
                {"coordinate_scale", 1.0},
                {"invalid_items", json::array()}}})},
        }.dump(2);
    }
    const auto result = spiral::layout::layout_largest_fiber_component(
        cache,
        [](std::span<const spiral::winding::Zyx> points) {
            std::vector<float> theta;
            theta.reserve(points.size());
            for (const auto& point : points) {
                assert(point.z >= 0.0f && point.z < 21.0f);
                theta.push_back(static_cast<float>(std::fmod(
                    point.x * 0.05f + point.z, 2.0 * std::numbers::pi)));
            }
            return theta;
        }, {}, std::pair{0.0f, 21.0f});
    assert(result.fibers.size() == 2);
    assert(result.root_fiber == "h.json");
    assert(result.crossings.size() == 1);
    assert(std::abs(result.crossings[0].u_residual) < 1e-6);
    assert(std::abs(result.crossings[0].v_residual) < 1e-6);
    const auto horizontal = std::find_if(
        result.fibers.begin(), result.fibers.end(),
        [](const auto& fiber) { return fiber.axis == 'H'; });
    const auto vertical = std::find_if(
        result.fibers.begin(), result.fibers.end(),
        [](const auto& fiber) { return fiber.axis == 'V'; });
    assert(horizontal != result.fibers.end());
    assert(vertical != result.fibers.end());
    assert(horizontal->points.size() == 5);
    assert(vertical->points.size() == 7);
    assert(!vertical->points.front().theta_valid);
    assert(vertical->points[1].theta_valid);
    assert(!vertical->points.back().theta_valid);
    assert(result.theta_covered_points == 10);
    assert(result.theta_uncovered_points == 4);
    assert(horizontal->points.front().u < horizontal->points.back().u);
    assert(vertical->points.front().z < vertical->points.back().z);
    assert(vertical->points.front().v < vertical->points.back().v);
    const auto vertical_winding = vertical->points[1].winding;
    assert(std::all_of(
        vertical->points.begin() + 1, vertical->points.end() - 1,
        [&](const auto& point) { return point.winding == vertical_winding; }));
    assert(result.excluded_fibers == std::vector<std::string>{"z-isolated.json"});
    std::filesystem::remove_all(directory);
}

void test_native_registration()
{
    std::vector<spiral::registration::Correspondence2d> values;
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            const double u = column * 4.0;
            const double v = row * 4.0;
            // Reflection across local v followed by a quarter turn.
            values.push_back({u, v, v + 30.0, u - 7.0});
        }
    }
    values.push_back({100, 100, -500, 700});
    spiral::registration::Options options;
    options.min_inliers = 16;
    const auto fit = spiral::registration::fit_rigid_2d(values, options);
    assert(fit.accepted);
    assert(fit.reflected);
    assert(fit.inliers == 16);
    assert(fit.rms < 1e-8);
    values.pop_back();
    values.pop_back();
    const auto boundary = spiral::registration::fit_rigid_2d(values, options);
    assert(!boundary.accepted);
    assert(boundary.rejection == "too_few_contacts");

    values.clear();
    for (int row = 0; row < 100; ++row) {
        for (int column = 0; column < 100; ++column) {
            values.push_back({
                static_cast<double>(column), static_cast<double>(row),
                static_cast<double>(column + 12), static_cast<double>(row - 9),
            });
        }
    }
    options.max_hypotheses = 64;
    const auto dense = spiral::registration::fit_rigid_2d(values, options);
    assert(dense.accepted);
    assert(dense.inliers == values.size());
    assert(dense.rms < 1e-8);
}

void test_pose_graph_refinement()
{
    std::vector<spiral::registration::Pose2d> poses(3);
    poses[1].translation_u = 11.0;
    poses[2].translation_u = 23.0;
    std::vector<spiral::registration::AbsolutePoseConstraint> absolute;
    std::vector<spiral::registration::RelativePoseConstraint> relative;
    for (int index = 0; index < 20; ++index) {
        const double v = index * 2.0;
        absolute.push_back({0, 0.0, v, 0.0, v});
        relative.push_back({0, 10.0, v, 1, 0.0, v});
        relative.push_back({1, 10.0, v, 2, 0.0, v});
        relative.push_back({0, 20.0, v, 2, 0.0, v});
    }
    const auto result = spiral::registration::refine_pose_graph(
        poses, absolute, relative);
    assert(result.usable);
    assert(result.final_cost < result.initial_cost);
    assert(std::abs(result.poses[1].translation_u - 10.0) < 1e-6);
    assert(std::abs(result.poses[2].translation_u - 20.0) < 1e-6);
}

} // namespace

int main()
{
    test_graph_and_rollback();
    test_persistence();
    test_lifted_holonomy();
    test_theta_topology();
    test_track_spatial_index();
    test_fiber_first_layout();
    test_native_registration();
    test_pose_graph_refinement();
    return 0;
}
