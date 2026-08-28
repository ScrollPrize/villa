#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <spiral_graph/winding_graph.hpp>

namespace spiral::winding {

struct Zyx {
    float z = 0.0f;
    float y = 0.0f;
    float x = 0.0f;
};

using ThetaProvider = std::function<std::vector<float>(std::span<const Zyx>)>;

struct ThetaProviders {
    ThetaProvider reported;
    ThetaProvider geometric;
};

enum class InputRole : std::uint8_t {
    absolute,
    relative,
    same_winding,
};

struct GraphOptions {
    float contact_tolerance = 2.5f;
    std::size_t theta_batch_size = 1'048'576;
    int workers = 0;
    float fiber_coordinate_scale = 0.25f;
    std::size_t surface_sampling_stride = 1;
    float z_min = -std::numeric_limits<float>::infinity();
    float z_max = std::numeric_limits<float>::infinity();
};

struct TrackInfo {
    std::uint64_t tracks = 0;
    std::uint64_t points = 0;
    std::uint64_t crossings = 0;
};

struct TrackIndexInfo {
    std::uint64_t points = 0;
    std::uint64_t cells = 0;
    std::uint32_t cell_size = 0;
    bool already_present = false;
};

struct ContactHit {
    std::string patch_id;
    float distance = 0.0f;
    float row = 0.0f;
    float column = 0.0f;
};

// Streamable, cache-backed samples for laying one patch out in unwrapped
// winding/z coordinates. Arrays are packed row-major over valid quad centres.
struct PatchLayoutData {
    std::string patch_id;
    std::string source_path;
    std::size_t quad_rows = 0;
    std::size_t quad_columns = 0;
    std::vector<std::uint32_t> quad_ij;
    std::vector<float> zyx;
    std::vector<float> reported_local_turn;
    std::vector<float> geometric_local_turn;
    // Original source-grid vertices incident to retained valid quads. Their
    // continuous turns are reconstructed from the neighboring lifted quad
    // centres, including one-sided boundary extrapolation.
    std::size_t vertex_rows = 0;
    std::size_t vertex_columns = 0;
    std::vector<std::uint32_t> vertex_ij;
    std::vector<float> vertex_zyx;
    std::vector<float> reported_vertex_turn;
};

// Owns the patch/source data needed to turn file inputs into root-frame graph
// constraints. The graph algebra itself remains independent and usable from
// C++ without Python or Torch.
class InputGraph {
public:
    explicit InputGraph(GraphOptions options = {});
    ~InputGraph();
    InputGraph(InputGraph&&) noexcept;
    InputGraph& operator=(InputGraph&&) noexcept;
    InputGraph(const InputGraph&) = delete;
    InputGraph& operator=(const InputGraph&) = delete;

    WindingGraph& graph() noexcept;
    const WindingGraph& graph() const noexcept;
    const GraphOptions& options() const noexcept;
    const std::string& theta_provider_key() const noexcept;
    void set_theta_provider_key(std::string key);

    AddResult add_patches(
        const std::vector<std::filesystem::path>& paths,
        const ThetaProviders& theta_providers);
    // Enable or disable an already cached patch as a contact target. Validity
    // may only change before any dependent source/constraint is committed.
    bool set_patch_valid(const std::string& patch_id, bool valid);
    bool patch_valid(const std::string& patch_id) const;
    std::vector<std::vector<ContactHit>> inspect_contacts(
        std::span<const Zyx> points,
        std::optional<float> tolerance = std::nullopt) const;
    PatchLayoutData patch_layout(const std::string& patch_id) const;
    AddResult add_point_collections(
        const std::vector<std::filesystem::path>& paths,
        InputRole role,
        const ThetaProviders& theta_providers);
    // Derive the constraints represented by point collections without
    // mutating the graph or registering the source. This is intended for
    // holdout evaluation against an existing graph.
    std::vector<Constraint> inspect_point_collections(
        const std::vector<std::filesystem::path>& paths,
        InputRole role,
        const ThetaProviders& theta_providers) const;
    AddResult add_fibers(
        const std::filesystem::path& directory,
        float coordinate_scale,
        std::vector<std::string> invalid_fibers = {});
    AddResult add_tracks(
        const std::filesystem::path& tracks,
        const std::filesystem::path& crossings,
        const std::filesystem::path& spatial_index,
        const ThetaProviders& theta_providers);

    TrackIndexInfo prepare_track_index(
        const std::filesystem::path& tracks,
        const std::filesystem::path& output,
        std::uint32_t cell_size = 32,
        std::size_t memory_budget_bytes = 512ull << 20) const;

    TrackInfo inspect_tracks(
        const std::filesystem::path& tracks,
        const std::filesystem::path& crossings = {}) const;

    void save(const std::filesystem::path& cache_directory) const;
    static InputGraph open(
        const std::filesystem::path& cache_directory,
        GraphOptions options = {},
        const ThetaProvider& geometric_provider = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char* input_role_name(InputRole role) noexcept;

} // namespace spiral::winding
