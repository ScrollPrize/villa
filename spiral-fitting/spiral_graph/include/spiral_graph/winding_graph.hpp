#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace spiral::winding {

using NodeId = std::uint32_t;

enum class ConflictKind : std::uint8_t {
    absolute_anchor,
    patch_theta,
    source_theta,
};

struct Provenance {
    std::string source_type;
    std::string source;
    std::string item;
    std::string detail;
};

// The directed equation carried by an edge is
//     value(to) - value(from) == delta.
struct Constraint {
    NodeId from = 0;
    NodeId to = 0;
    std::int64_t delta = 0;
    // Independent transport of raw polar angle around the umbilicus.
    std::int64_t geometric_delta = 0;
    Provenance provenance;
    bool absolute = false;
};

struct CycleEdge {
    NodeId from = 0;
    NodeId to = 0;
    std::int64_t delta = 0;
    std::int64_t geometric_delta = 0;
    std::size_t constraint_index = 0;
    Provenance provenance;
    bool closing = false;
};

struct Conflict {
    ConflictKind kind = ConflictKind::absolute_anchor;
    std::int64_t residual = 0;
    Constraint closing_constraint;
    std::vector<CycleEdge> cycle;
};

struct Holonomy {
    std::int64_t reported_holonomy = 0;
    std::int64_t geometric_holonomy = 0;
    std::int64_t inconsistency = 0;
    Constraint closing_constraint;
    std::vector<CycleEdge> cycle;
};

struct HolonomyAudit {
    std::int64_t reported_holonomy = 0;
    std::int64_t geometric_holonomy = 0;
    std::int64_t inconsistency = 0;
    std::size_t constraint_index = 0;
};

struct AddResult {
    bool committed = false;
    bool already_present = false;
    std::size_t nodes_added = 0;
    std::size_t constraints_added = 0;
    std::size_t anchors_added = 0;
    std::size_t holonomies_added = 0;
    std::optional<Conflict> conflict;
};

struct GraphStats {
    std::size_t patch_count = 0;
    std::size_t constraint_count = 0;
    std::size_t component_count = 0;
    std::size_t anchored_component_count = 0;
    std::size_t holonomy_count = 0;
};

struct LiftedWinding {
    // One deterministic representative, selected by the graph's spanning
    // forest. Other path transports differ by multiples of period.
    std::int64_t representative = 0;
    // Zero means unique. A positive value h means representative + k*h for
    // any integer k is reachable by winding around retained cycles.
    std::int64_t period = 0;
};

// Append-only integer lift of wrapped-theta transport. Nonzero relative cycles
// are retained as sheet-changing holonomy. Mutating batches remain atomic for
// hard failures such as contradictory absolute anchors.
class WindingGraph {
public:
    WindingGraph();

    NodeId ensure_patch(std::string id);
    std::size_t node_count() const noexcept { return names_.size(); }
    // Used by a higher-level add transaction after its constraint batch was
    // rejected. Only unconstrained trailing nodes may be discarded.
    void discard_trailing_patches(std::size_t node_marker);
    bool has_patch(std::string_view id) const;
    NodeId patch_node(std::string_view id) const;
    const std::string& node_name(NodeId node) const;

    AddResult add_constraints(const std::vector<Constraint>& constraints);
    AddResult add_constraint(
        std::string_view from_patch,
        std::string_view to_patch,
        std::int64_t root_delta,
        std::int64_t geometric_delta,
        Provenance provenance = {});
    AddResult add_anchor(
        std::string_view patch,
        std::int64_t root_winding,
        std::int64_t geometric_root_winding,
        Provenance provenance = {});

    std::optional<LiftedWinding> lifted_relative_winding(
        std::string_view from_patch,
        std::string_view to_patch) const;
    std::size_t holonomy_count() const noexcept { return holonomies_.size(); }
    std::vector<HolonomyAudit> holonomy_audits() const;
    Holonomy holonomy(std::size_t index) const;
    GraphStats stats() const;
    const std::vector<Constraint>& constraints() const noexcept
    {
        return constraints_;
    }

    void save(const std::filesystem::path& path) const;
    static WindingGraph open(const std::filesystem::path& path);

private:
    struct RootPotential {
        NodeId root = 0;
        std::int64_t potential = 0; // value(node) - value(root)
        std::int64_t geometric_potential = 0;
    };
    struct UnionChange {
        NodeId child = 0;
        NodeId parent = 0;
        std::uint32_t parent_size = 0;
        std::int64_t parent_holonomy_period = 0;
    };
    struct HolonomyRecord {
        std::size_t constraint_index = 0;
        std::int64_t reported_holonomy = 0;
        std::int64_t geometric_holonomy = 0;
    };
    struct HolonomyChange {
        NodeId root = 0;
        std::int64_t previous_period = 0;
    };

    RootPotential find(NodeId node) const;
    bool apply_constraint(const Constraint& constraint, Conflict& conflict);
    void rollback(
        std::size_t union_marker,
        std::size_t edge_marker,
        std::size_t holonomy_marker,
        std::size_t node_marker);
    std::vector<CycleEdge> witness_path(NodeId from, NodeId to) const;
    void append_loaded_constraint(const Constraint& constraint);

    std::vector<std::string> names_;
    std::unordered_map<std::string, NodeId> name_to_node_;
    std::vector<NodeId> parent_;
    std::vector<std::uint32_t> size_;
    std::vector<std::int64_t> parent_delta_;
    std::vector<std::int64_t> parent_geometric_delta_;
    std::vector<std::int64_t> root_holonomy_period_;
    std::vector<Constraint> constraints_;
    std::vector<std::vector<std::size_t>> forest_edges_;
    std::vector<UnionChange> union_log_;
    std::vector<HolonomyRecord> holonomies_;
    std::vector<HolonomyChange> holonomy_changes_;
};

const char* conflict_kind_name(ConflictKind kind) noexcept;

} // namespace spiral::winding
