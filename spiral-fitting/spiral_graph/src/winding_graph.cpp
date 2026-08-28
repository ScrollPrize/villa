#include <spiral_graph/winding_graph.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace spiral::winding {
namespace {

constexpr std::array<char, 8> cache_magic{'S', 'P', 'W', 'G', 'R', 'P', 'H', '\0'};
constexpr std::uint32_t cache_version = 2;
constexpr std::string_view ground_name = "<absolute-ground>";

template <typename T>
void write_scalar(std::ostream& stream, const T& value)
{
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!stream) throw std::runtime_error("failed writing winding graph cache");
}

template <typename T>
T read_scalar(std::istream& stream)
{
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!stream) throw std::runtime_error("truncated winding graph cache");
    return value;
}

void write_string(std::ostream& stream, const std::string& value)
{
    if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("winding graph string exceeds UINT32_MAX");
    }
    write_scalar(stream, static_cast<std::uint32_t>(value.size()));
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!stream) throw std::runtime_error("failed writing winding graph cache");
}

std::string read_string(std::istream& stream)
{
    const auto size = read_scalar<std::uint32_t>(stream);
    std::string value(size, '\0');
    stream.read(value.data(), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("truncated winding graph string");
    return value;
}

void write_provenance(std::ostream& stream, const Provenance& provenance)
{
    write_string(stream, provenance.source_type);
    write_string(stream, provenance.source);
    write_string(stream, provenance.item);
    write_string(stream, provenance.detail);
}

Provenance read_provenance(std::istream& stream)
{
    return {
        read_string(stream), read_string(stream),
        read_string(stream), read_string(stream),
    };
}

std::int64_t checked_add(std::int64_t a, std::int64_t b)
{
    std::int64_t result = 0;
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_add_overflow(a, b, &result)) {
        throw std::overflow_error("winding potential overflow");
    }
#else
    if ((b > 0 && a > std::numeric_limits<std::int64_t>::max() - b)
        || (b < 0 && a < std::numeric_limits<std::int64_t>::min() - b)) {
        throw std::overflow_error("winding potential overflow");
    }
    result = a + b;
#endif
    return result;
}

std::int64_t checked_sub(std::int64_t a, std::int64_t b)
{
    if (b == std::numeric_limits<std::int64_t>::min()) {
        return checked_add(checked_add(a, std::numeric_limits<std::int64_t>::max()), 1);
    }
    return checked_add(a, -b);
}

std::int64_t checked_magnitude(std::int64_t value)
{
    if (value == std::numeric_limits<std::int64_t>::min()) {
        throw std::overflow_error("holonomy period exceeds INT64_MAX");
    }
    return value < 0 ? -value : value;
}

} // namespace

const char* conflict_kind_name(ConflictKind kind) noexcept
{
    switch (kind) {
    case ConflictKind::absolute_anchor: return "absolute_anchor";
    case ConflictKind::patch_theta: return "patch_theta";
    case ConflictKind::source_theta: return "source_theta";
    }
    return "unknown";
}

WindingGraph::WindingGraph()
{
    names_.emplace_back(ground_name);
    name_to_node_.emplace(names_.front(), 0);
    parent_.push_back(0);
    size_.push_back(1);
    parent_delta_.push_back(0);
    parent_geometric_delta_.push_back(0);
    root_holonomy_period_.push_back(0);
    forest_edges_.emplace_back();
}

NodeId WindingGraph::ensure_patch(std::string id)
{
    if (id.empty()) throw std::invalid_argument("patch id must not be empty");
    if (id == ground_name) throw std::invalid_argument("patch id is reserved");
    if (const auto found = name_to_node_.find(id); found != name_to_node_.end()) {
        return found->second;
    }
    if (names_.size() > std::numeric_limits<NodeId>::max()) {
        throw std::overflow_error("winding graph node count exceeds UINT32_MAX");
    }
    const NodeId node = static_cast<NodeId>(names_.size());
    name_to_node_.emplace(id, node);
    names_.push_back(std::move(id));
    parent_.push_back(node);
    size_.push_back(1);
    parent_delta_.push_back(0);
    parent_geometric_delta_.push_back(0);
    root_holonomy_period_.push_back(0);
    forest_edges_.emplace_back();
    return node;
}

void WindingGraph::discard_trailing_patches(std::size_t node_marker)
{
    if (node_marker < 1 || node_marker > names_.size()) {
        throw std::invalid_argument("invalid winding graph node marker");
    }
    for (const Constraint& edge : constraints_) {
        if (edge.from >= node_marker || edge.to >= node_marker) {
            throw std::logic_error("cannot discard a constrained graph node");
        }
    }
    rollback(
        union_log_.size(), constraints_.size(), holonomies_.size(), node_marker);
}

bool WindingGraph::has_patch(std::string_view id) const
{
    return name_to_node_.contains(std::string(id));
}

NodeId WindingGraph::patch_node(std::string_view id) const
{
    const auto found = name_to_node_.find(std::string(id));
    if (found == name_to_node_.end() || found->second == 0) {
        throw std::out_of_range("unknown patch id: " + std::string(id));
    }
    return found->second;
}

const std::string& WindingGraph::node_name(NodeId node) const
{
    if (node >= names_.size()) throw std::out_of_range("graph node is out of range");
    return names_[node];
}

WindingGraph::RootPotential WindingGraph::find(NodeId node) const
{
    if (node >= parent_.size()) throw std::out_of_range("graph node is out of range");
    std::int64_t potential = 0;
    std::int64_t geometric_potential = 0;
    NodeId cursor = node;
    while (parent_[cursor] != cursor) {
        potential = checked_add(potential, parent_delta_[cursor]);
        geometric_potential = checked_add(
            geometric_potential, parent_geometric_delta_[cursor]);
        cursor = parent_[cursor];
    }
    return {cursor, potential, geometric_potential};
}

std::vector<CycleEdge> WindingGraph::witness_path(NodeId from, NodeId to) const
{
    if (from == to) return {};
    const NodeId missing = std::numeric_limits<NodeId>::max();
    const std::size_t no_edge = std::numeric_limits<std::size_t>::max();
    std::vector<NodeId> previous(names_.size(), missing);
    std::vector<std::size_t> previous_edge(names_.size(), no_edge);
    std::deque<NodeId> queue;
    previous[from] = from;
    queue.push_back(from);
    while (!queue.empty() && previous[to] == missing) {
        const NodeId node = queue.front();
        queue.pop_front();
        for (const std::size_t edge_index : forest_edges_[node]) {
            const Constraint& edge = constraints_[edge_index];
            const NodeId next = edge.from == node ? edge.to : edge.from;
            if (previous[next] != missing) continue;
            previous[next] = node;
            previous_edge[next] = edge_index;
            queue.push_back(next);
        }
    }
    if (previous[to] == missing) {
        throw std::logic_error("weighted union and witness forest disagree");
    }

    std::vector<CycleEdge> reverse;
    for (NodeId node = to; node != from; node = previous[node]) {
        const NodeId prior = previous[node];
        const std::size_t edge_index = previous_edge[node];
        const Constraint& edge = constraints_[edge_index];
        const bool forward = edge.from == prior && edge.to == node;
        reverse.push_back({
            prior,
            node,
            forward ? edge.delta : checked_sub(0, edge.delta),
            forward ? edge.geometric_delta
                    : checked_sub(0, edge.geometric_delta),
            edge_index,
            edge.provenance,
            false,
        });
    }
    std::reverse(reverse.begin(), reverse.end());
    return reverse;
}

bool WindingGraph::apply_constraint(
    const Constraint& constraint, Conflict& conflict)
{
    if (constraint.from >= names_.size() || constraint.to >= names_.size()) {
        throw std::out_of_range("constraint endpoint is out of range");
    }
    const RootPotential a = find(constraint.from);
    const RootPotential b = find(constraint.to);
    if (a.root == b.root) {
        const std::int64_t implied = checked_sub(b.potential, a.potential);
        const std::int64_t geometric_implied = checked_sub(
            b.geometric_potential, a.geometric_potential);
        const std::int64_t reported_holonomy = checked_sub(
            implied, constraint.delta);
        const std::int64_t geometric_holonomy = checked_sub(
            geometric_implied, constraint.geometric_delta);
        if (constraint.absolute && reported_holonomy == 0) {
            constraints_.push_back(constraint);
            return true;
        }
        if (!constraint.absolute) {
            const std::size_t edge_index = constraints_.size();
            constraints_.push_back(constraint);
            holonomies_.push_back({
                edge_index, reported_holonomy, geometric_holonomy});
            holonomy_changes_.push_back({a.root, root_holonomy_period_[a.root]});
            root_holonomy_period_[a.root] = std::gcd(
                root_holonomy_period_[a.root],
                checked_magnitude(reported_holonomy));
            return true;
        }
        conflict.kind = ConflictKind::absolute_anchor;
        conflict.residual = reported_holonomy;
        conflict.closing_constraint = constraint;
        conflict.cycle = witness_path(constraint.from, constraint.to);
        conflict.cycle.push_back({
            constraint.to,
            constraint.from,
            checked_sub(0, constraint.delta),
            checked_sub(0, constraint.geometric_delta),
            constraints_.size(),
            constraint.provenance,
            true,
        });
        return false;
    }

    const std::size_t edge_index = constraints_.size();
    constraints_.push_back(constraint);
    if (size_[a.root] >= size_[b.root]) {
        union_log_.push_back({
            b.root, a.root, size_[a.root], root_holonomy_period_[a.root]});
        parent_[b.root] = a.root;
        // value(root_b) - value(root_a) = d + pot(a) - pot(b)
        parent_delta_[b.root] = checked_add(
            constraint.delta, checked_sub(a.potential, b.potential));
        parent_geometric_delta_[b.root] = checked_add(
            constraint.geometric_delta,
            checked_sub(a.geometric_potential, b.geometric_potential));
        size_[a.root] += size_[b.root];
        root_holonomy_period_[a.root] = std::gcd(
            root_holonomy_period_[a.root], root_holonomy_period_[b.root]);
    } else {
        union_log_.push_back({
            a.root, b.root, size_[b.root], root_holonomy_period_[b.root]});
        parent_[a.root] = b.root;
        // value(root_a) - value(root_b) = -d - pot(a) + pot(b)
        parent_delta_[a.root] = checked_sub(
            checked_sub(b.potential, a.potential), constraint.delta);
        parent_geometric_delta_[a.root] = checked_sub(
            checked_sub(b.geometric_potential, a.geometric_potential),
            constraint.geometric_delta);
        size_[b.root] += size_[a.root];
        root_holonomy_period_[b.root] = std::gcd(
            root_holonomy_period_[b.root], root_holonomy_period_[a.root]);
    }
    forest_edges_[constraint.from].push_back(edge_index);
    forest_edges_[constraint.to].push_back(edge_index);
    return true;
}

void WindingGraph::rollback(
    std::size_t union_marker,
    std::size_t edge_marker,
    std::size_t holonomy_marker,
    std::size_t node_marker)
{
    for (std::size_t edge_index = constraints_.size(); edge_index > edge_marker;) {
        --edge_index;
        const Constraint& edge = constraints_[edge_index];
        if (!forest_edges_[edge.from].empty()
            && forest_edges_[edge.from].back() == edge_index) {
            forest_edges_[edge.from].pop_back();
        }
        if (!forest_edges_[edge.to].empty()
            && forest_edges_[edge.to].back() == edge_index) {
            forest_edges_[edge.to].pop_back();
        }
    }
    constraints_.resize(edge_marker);
    holonomies_.resize(holonomy_marker);
    while (holonomy_changes_.size() > holonomy_marker) {
        const HolonomyChange change = holonomy_changes_.back();
        holonomy_changes_.pop_back();
        root_holonomy_period_[change.root] = change.previous_period;
    }
    while (union_log_.size() > union_marker) {
        const UnionChange change = union_log_.back();
        union_log_.pop_back();
        parent_[change.child] = change.child;
        parent_delta_[change.child] = 0;
        parent_geometric_delta_[change.child] = 0;
        size_[change.parent] = change.parent_size;
        root_holonomy_period_[change.parent]
            = change.parent_holonomy_period;
    }
    while (names_.size() > node_marker) {
        name_to_node_.erase(names_.back());
        names_.pop_back();
        parent_.pop_back();
        size_.pop_back();
        parent_delta_.pop_back();
        parent_geometric_delta_.pop_back();
        root_holonomy_period_.pop_back();
        forest_edges_.pop_back();
    }
}

AddResult WindingGraph::add_constraints(
    const std::vector<Constraint>& constraints)
{
    AddResult result;
    if (constraints.empty()) {
        result.committed = true;
        return result;
    }
    const std::size_t union_marker = union_log_.size();
    const std::size_t edge_marker = constraints_.size();
    const std::size_t holonomy_marker = holonomies_.size();
    const std::size_t node_marker = names_.size();
    std::size_t anchors = 0;
    for (const Constraint& constraint : constraints) {
        Conflict conflict;
        if (!apply_constraint(constraint, conflict)) {
            rollback(
                union_marker, edge_marker, holonomy_marker, node_marker);
            result.conflict = std::move(conflict);
            return result;
        }
        anchors += constraint.absolute ? 1 : 0;
    }
    result.committed = true;
    result.constraints_added = constraints_.size() - edge_marker;
    result.anchors_added = anchors;
    result.holonomies_added = holonomies_.size() - holonomy_marker;
    return result;
}

AddResult WindingGraph::add_constraint(
    std::string_view from_patch,
    std::string_view to_patch,
    std::int64_t root_delta,
    std::int64_t geometric_delta,
    Provenance provenance)
{
    return add_constraints({Constraint{
        patch_node(from_patch), patch_node(to_patch), root_delta, geometric_delta,
        std::move(provenance), false,
    }});
}

AddResult WindingGraph::add_anchor(
    std::string_view patch,
    std::int64_t root_winding,
    std::int64_t geometric_root_winding,
    Provenance provenance)
{
    return add_constraints({Constraint{
        0, patch_node(patch), root_winding, geometric_root_winding,
        std::move(provenance), true,
    }});
}

std::optional<LiftedWinding> WindingGraph::lifted_relative_winding(
    std::string_view from_patch,
    std::string_view to_patch) const
{
    const RootPotential a = find(patch_node(from_patch));
    const RootPotential b = find(patch_node(to_patch));
    if (a.root != b.root) return std::nullopt;
    return LiftedWinding{
        checked_sub(b.potential, a.potential),
        root_holonomy_period_[a.root],
    };
}

Holonomy WindingGraph::holonomy(std::size_t index) const
{
    const HolonomyRecord& record = holonomies_.at(index);
    const Constraint& constraint = constraints_.at(record.constraint_index);
    Holonomy output;
    output.reported_holonomy = record.reported_holonomy;
    output.geometric_holonomy = record.geometric_holonomy;
    output.inconsistency = checked_sub(
        record.reported_holonomy, record.geometric_holonomy);
    output.closing_constraint = constraint;
    output.cycle = witness_path(constraint.from, constraint.to);
    output.cycle.push_back({
        constraint.to,
        constraint.from,
        checked_sub(0, constraint.delta),
        checked_sub(0, constraint.geometric_delta),
        record.constraint_index,
        constraint.provenance,
        true,
    });
    return output;
}

std::vector<HolonomyAudit> WindingGraph::holonomy_audits() const
{
    std::vector<HolonomyAudit> output;
    output.reserve(holonomies_.size());
    for (const HolonomyRecord& record : holonomies_) {
        output.push_back({
            record.reported_holonomy,
            record.geometric_holonomy,
            checked_sub(record.reported_holonomy, record.geometric_holonomy),
            record.constraint_index,
        });
    }
    return output;
}

GraphStats WindingGraph::stats() const
{
    GraphStats output;
    output.patch_count = names_.size() - 1;
    output.constraint_count = constraints_.size();
    std::unordered_map<NodeId, bool> roots;
    roots.reserve(names_.size());
    const NodeId ground_root = find(0).root;
    for (NodeId node = 1; node < names_.size(); ++node) {
        roots.emplace(find(node).root, false);
    }
    output.component_count = roots.size();
    output.anchored_component_count = roots.contains(ground_root) ? 1 : 0;
    output.holonomy_count = holonomies_.size();
    return output;
}

void WindingGraph::save(const std::filesystem::path& path) const
{
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    const std::filesystem::path temporary = path.string() + ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) throw std::runtime_error("cannot create " + temporary.string());
        stream.write(cache_magic.data(), cache_magic.size());
        write_scalar(stream, cache_version);
        write_scalar(stream, static_cast<std::uint32_t>(0));
        write_scalar(stream, static_cast<std::uint64_t>(names_.size() - 1));
        write_scalar(stream, static_cast<std::uint64_t>(constraints_.size()));
        for (std::size_t node = 1; node < names_.size(); ++node) {
            write_string(stream, names_[node]);
        }
        for (const Constraint& edge : constraints_) {
            write_scalar(stream, edge.from);
            write_scalar(stream, edge.to);
            write_scalar(stream, edge.delta);
            write_scalar(stream, edge.geometric_delta);
            write_scalar(stream, static_cast<std::uint8_t>(edge.absolute));
            write_provenance(stream, edge.provenance);
        }
        stream.flush();
        if (!stream) throw std::runtime_error("failed flushing " + temporary.string());
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        // std::filesystem::rename cannot replace an existing file on Windows;
        // the supported Unix targets do, but keep a portable fallback.
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
    }
    if (error) {
        throw std::system_error(error, "cannot publish winding graph cache");
    }
}

void WindingGraph::append_loaded_constraint(const Constraint& constraint)
{
    Conflict conflict;
    if (!apply_constraint(constraint, conflict)) {
        throw std::runtime_error(
            "cached graph contains a contradictory absolute constraint with residual "
            + std::to_string(conflict.residual));
    }
}

WindingGraph WindingGraph::open(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    std::array<char, 8> magic{};
    stream.read(magic.data(), magic.size());
    if (magic != cache_magic) throw std::runtime_error("not a winding graph cache");
    const std::uint32_t version = read_scalar<std::uint32_t>(stream);
    static_cast<void>(read_scalar<std::uint32_t>(stream));
    if (version != 1 && version != cache_version) {
        throw std::runtime_error("unsupported winding graph cache version");
    }
    const std::uint64_t node_count = read_scalar<std::uint64_t>(stream);
    const std::uint64_t edge_count = read_scalar<std::uint64_t>(stream);
    if (node_count > std::numeric_limits<NodeId>::max()
        || edge_count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("winding graph cache is too large for this host");
    }
    if (version == 1 && edge_count != 0) {
        throw std::runtime_error(
            "cached constraints predate geometric holonomy; rebuild sources");
    }
    WindingGraph graph;
    for (std::uint64_t node = 0; node < node_count; ++node) {
        graph.ensure_patch(read_string(stream));
    }
    graph.constraints_.reserve(static_cast<std::size_t>(edge_count));
    graph.union_log_.reserve(static_cast<std::size_t>(node_count));
    for (std::uint64_t edge = 0; edge < edge_count; ++edge) {
        Constraint constraint;
        constraint.from = read_scalar<NodeId>(stream);
        constraint.to = read_scalar<NodeId>(stream);
        constraint.delta = read_scalar<std::int64_t>(stream);
        if (version >= 2) {
            constraint.geometric_delta = read_scalar<std::int64_t>(stream);
        }
        constraint.absolute = read_scalar<std::uint8_t>(stream) != 0;
        constraint.provenance = read_provenance(stream);
        graph.append_loaded_constraint(constraint);
    }
    return graph;
}

} // namespace spiral::winding
