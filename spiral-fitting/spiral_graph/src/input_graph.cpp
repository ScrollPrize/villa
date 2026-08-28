#include <spiral_graph/input_graph.hpp>

#include <spiral_graph/surface_index.hpp>
#include <spiral_graph/theta_topology.hpp>
#include <spiral_graph/track_io.hpp>
#include <spiral_graph/track_spatial_index.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <tiffio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace spiral::winding {
namespace {

using json = nlohmann::json;

struct SourcePoint {
    std::int64_t id = 0;
    Zyx zyx{};
    std::optional<std::int64_t> annotation;
    float theta = 0.0f;
    float geometric_theta = 0.0f;
    std::int32_t source_potential = 0;
    std::int32_t geometric_source_potential = 0;
};

struct Attachment {
    std::size_t point = 0;
    std::size_t patch = 0;
    std::int32_t patch_potential = 0;
    std::int32_t geometric_patch_potential = 0;
    std::int64_t source_potential = 0;
    std::int64_t geometric_source_potential = 0;
    float distance = 0.0f;
    float row = 0.0f;
    float column = 0.0f;
};

struct SourceSpec {
    enum class Kind : std::uint8_t { point_collection, fibers, tracks };
    Kind kind = Kind::point_collection;
    std::vector<std::filesystem::path> paths;
    InputRole role = InputRole::same_winding;
    float coordinate_scale = 1.0f;
    std::vector<std::string> invalid_items;
};

std::string canonical_text(const std::filesystem::path& path)
{
    std::error_code error;
    const auto canonical = std::filesystem::weakly_canonical(path, error);
    return (error ? std::filesystem::absolute(path) : canonical).string();
}

std::string path_fingerprint(const std::filesystem::path& path)
{
    const auto stamp = [](const std::filesystem::path& file) {
        std::error_code error;
        const auto size = std::filesystem::file_size(file, error);
        if (error) throw std::runtime_error("cannot stat " + file.string());
        const auto modified = std::filesystem::last_write_time(file, error);
        if (error) throw std::runtime_error("cannot stat " + file.string());
        return canonical_text(file) + "@" + std::to_string(size) + "@"
            + std::to_string(modified.time_since_epoch().count());
    };
    if (std::filesystem::is_regular_file(path)) return stamp(path);
    if (!std::filesystem::is_directory(path)) {
        throw std::runtime_error("input path does not exist: " + path.string());
    }
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file()) files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    std::string output = canonical_text(path);
    for (const auto& file : files) output += "|" + stamp(file);
    return output;
}

std::string source_identity(const SourceSpec& spec)
{
    std::string output = std::to_string(static_cast<int>(spec.kind)) + ":"
        + std::to_string(static_cast<int>(spec.role)) + ":"
        + std::to_string(spec.coordinate_scale);
    for (const auto& path : spec.paths) output += ":" + canonical_text(path);
    for (const auto& item : spec.invalid_items) output += ":invalid=" + item;
    return output;
}

std::string source_key(const SourceSpec& spec)
{
    std::string output = source_identity(spec);
    for (const auto& path : spec.paths) output += ":" + path_fingerprint(path);
    return output;
}

std::uint64_t constraint_signature(const Constraint& constraint)
{
    std::uint64_t hash = 1469598103934665603ull;
    const auto add_bytes = [&hash](const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for (std::size_t index = 0; index < size; ++index) {
            hash ^= bytes[index];
            hash *= 1099511628211ull;
        }
    };
    const auto add_string = [&add_bytes](const std::string& value) {
        const std::uint64_t size = value.size();
        add_bytes(&size, sizeof(size));
        add_bytes(value.data(), value.size());
    };
    add_bytes(&constraint.from, sizeof(constraint.from));
    add_bytes(&constraint.to, sizeof(constraint.to));
    add_bytes(&constraint.delta, sizeof(constraint.delta));
    add_bytes(&constraint.geometric_delta, sizeof(constraint.geometric_delta));
    add_bytes(&constraint.absolute, sizeof(constraint.absolute));
    add_string(constraint.provenance.source_type);
    add_string(constraint.provenance.source);
    add_string(constraint.provenance.item);
    add_string(constraint.provenance.detail);
    return hash;
}

bool same_constraint(const Constraint& a, const Constraint& b)
{
    return a.from == b.from && a.to == b.to && a.delta == b.delta
        && a.geometric_delta == b.geometric_delta
        && a.absolute == b.absolute
        && a.provenance.source_type == b.provenance.source_type
        && a.provenance.source == b.provenance.source
        && a.provenance.item == b.provenance.item
        && a.provenance.detail == b.provenance.detail;
}

void discard_existing_constraints(
    const WindingGraph& graph,
    std::vector<Constraint>& generated)
{
    if (generated.empty() || graph.constraints().empty()) return;
    std::vector<std::pair<std::uint64_t, std::size_t>> existing;
    existing.reserve(graph.constraints().size());
    for (std::size_t index = 0; index < graph.constraints().size(); ++index) {
        existing.emplace_back(
            constraint_signature(graph.constraints()[index]), index);
    }
    std::sort(existing.begin(), existing.end());
    std::vector<Constraint> novel;
    novel.reserve(generated.size());
    for (auto& constraint : generated) {
        const std::uint64_t signature = constraint_signature(constraint);
        const auto range = std::equal_range(
            existing.begin(), existing.end(),
            std::pair{signature, std::size_t{0}},
            [](const auto& a, const auto& b) { return a.first < b.first; });
        bool found = false;
        for (auto iterator = range.first; iterator != range.second; ++iterator) {
            if (same_constraint(
                    constraint, graph.constraints()[iterator->second])) {
                found = true;
                break;
            }
        }
        if (!found) novel.push_back(std::move(constraint));
    }
    generated = std::move(novel);
}

std::string patch_fingerprint(const std::filesystem::path& path)
{
    std::string output = path_fingerprint(path / "meta.json") + "|"
        + path_fingerprint(path / "x.tif") + "|"
        + path_fingerprint(path / "y.tif") + "|"
        + path_fingerprint(path / "z.tif");
    if (std::filesystem::is_regular_file(path / "mask.tif")) {
        output += "|" + path_fingerprint(path / "mask.tif");
    }
    return output;
}

std::pair<std::size_t, std::size_t> tiff_dimensions(
    const std::filesystem::path& path)
{
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    if (!tif) throw std::runtime_error("cannot open TIFF " + path.string());
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFClose(tif);
    return {height, width};
}

std::vector<std::filesystem::path> discover_patch_paths(
    const std::vector<std::filesystem::path>& requested)
{
    std::vector<std::filesystem::path> output;
    for (const auto& path : requested) {
        if (std::filesystem::is_regular_file(path / "meta.json")
            && std::filesystem::is_regular_file(path / "x.tif")
            && std::filesystem::is_regular_file(path / "y.tif")
            && std::filesystem::is_regular_file(path / "z.tif")) {
            output.push_back(path);
            continue;
        }
        if (!std::filesystem::is_directory(path)) {
            throw std::invalid_argument("patch path is not a directory: " + path.string());
        }
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (!entry.is_directory()) continue;
            const auto child = entry.path();
            if (std::filesystem::is_regular_file(child / "meta.json")
                && std::filesystem::is_regular_file(child / "x.tif")
                && std::filesystem::is_regular_file(child / "y.tif")
                && std::filesystem::is_regular_file(child / "z.tif")) {
                output.push_back(child);
            }
        }
    }
    std::sort(output.begin(), output.end(), [](const auto& a, const auto& b) {
        return canonical_text(a) < canonical_text(b);
    });
    output.erase(std::unique(output.begin(), output.end(), [](const auto& a, const auto& b) {
        return canonical_text(a) == canonical_text(b);
    }), output.end());
    return output;
}

std::vector<float> evaluate_theta(
    const ThetaProvider& provider,
    std::span<const Zyx> points,
    std::size_t batch_size)
{
    if (!provider) throw std::invalid_argument("a theta provider is required");
    if (batch_size == 0) throw std::invalid_argument("theta batch size must be positive");
    std::vector<float> output;
    output.reserve(points.size());
    for (std::size_t begin = 0; begin < points.size(); begin += batch_size) {
        const std::size_t count = std::min(batch_size, points.size() - begin);
        std::vector<float> batch = provider(points.subspan(begin, count));
        if (batch.size() != count) {
            throw std::runtime_error("theta provider returned the wrong number of values");
        }
        for (const float value : batch) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("theta provider returned a non-finite value");
            }
        }
        output.insert(output.end(), batch.begin(), batch.end());
    }
    return output;
}

std::int64_t integer_annotation(const json& value, const std::string& context)
{
    if (!value.is_number()) throw std::runtime_error(context + " is not numeric");
    const double number = value.get<double>();
    if (!std::isfinite(number) || std::abs(number - std::round(number)) > 1e-6) {
        throw std::runtime_error(context + " must be an integer");
    }
    if (number < static_cast<double>(std::numeric_limits<std::int64_t>::min())
        || number > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error(context + " is outside int64 range");
    }
    return static_cast<std::int64_t>(std::llround(number));
}

Zyx xyz_json(const json& value, float scale = 1.0f)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error("point position must be an [x, y, z] array");
    }
    const float x = value[0].get<float>() * scale;
    const float y = value[1].get<float>() * scale;
    const float z = value[2].get<float>() * scale;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        throw std::runtime_error("point position must be finite");
    }
    return {z, y, x};
}

struct DsuCycleEdge {
    std::uint32_t from = 0;
    std::uint32_t to = 0;
    std::int64_t delta = 0;
    std::size_t edge_index = 0;
    bool closing = false;
};

struct DsuConflict {
    std::int64_t residual = 0;
    std::uint32_t closing_from = 0;
    std::uint32_t closing_to = 0;
    std::int64_t closing_delta = 0;
    std::vector<DsuCycleEdge> cycle;
};

class SourceHolonomy final : public std::runtime_error {
public:
    explicit SourceHolonomy(Conflict conflict)
        : std::runtime_error(
            "source topology has holonomy " + std::to_string(conflict.residual)),
          conflict(std::move(conflict)) {}
    Conflict conflict;
};

class WeightedDsu {
public:
    explicit WeightedDsu(std::size_t count)
    {
        if (count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::overflow_error("source topology exceeds INT32_MAX nodes");
        }
        parent_.resize(count);
        size_.assign(count, 1);
        delta_.assign(count, 0);
        head_.assign(count, -1);
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    std::pair<std::uint32_t, std::int64_t> find(std::size_t node) const
    {
        std::int64_t value = 0;
        std::uint32_t cursor = static_cast<std::uint32_t>(node);
        while (parent_[cursor] != cursor) {
            value += delta_[cursor];
            cursor = parent_[cursor];
        }
        return {cursor, value};
    }

    std::optional<DsuConflict> add(
        std::size_t from, std::size_t to, std::int64_t required)
    {
        const auto [a, pa] = find(from);
        const auto [b, pb] = find(to);
        if (a == b) {
            const std::int64_t implied = pb - pa;
            if (implied == required) return std::nullopt;
            DsuConflict conflict;
            conflict.residual = implied - required;
            conflict.closing_from = static_cast<std::uint32_t>(from);
            conflict.closing_to = static_cast<std::uint32_t>(to);
            conflict.closing_delta = required;
            conflict.cycle = witness(
                static_cast<std::uint32_t>(from),
                static_cast<std::uint32_t>(to));
            conflict.cycle.push_back({
                static_cast<std::uint32_t>(to),
                static_cast<std::uint32_t>(from),
                -required,
                edges_.size(),
                true,
            });
            return conflict;
        }
        if (size_[a] >= size_[b]) {
            parent_[b] = a;
            delta_[b] = required + pa - pb;
            size_[a] += size_[b];
        } else {
            parent_[a] = b;
            delta_[a] = pb - pa - required;
            size_[b] += size_[a];
        }
        const std::size_t edge_index = edges_.size();
        constexpr std::size_t max_arc
            = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
        if (arcs_.size() > max_arc - 1) {
            throw std::overflow_error("source spanning forest exceeds INT32_MAX arcs");
        }
        edges_.push_back({
            static_cast<std::uint32_t>(from),
            static_cast<std::uint32_t>(to),
            required,
        });
        add_arc(static_cast<std::uint32_t>(from), static_cast<std::uint32_t>(to), edge_index);
        add_arc(static_cast<std::uint32_t>(to), static_cast<std::uint32_t>(from), edge_index);
        return std::nullopt;
    }

private:
    struct Edge {
        std::uint32_t from;
        std::uint32_t to;
        std::int64_t delta;
    };
    struct Arc {
        std::uint32_t to;
        std::uint32_t edge;
        std::int32_t next;
    };

    void add_arc(std::uint32_t from, std::uint32_t to, std::size_t edge)
    {
        arcs_.push_back({to, static_cast<std::uint32_t>(edge), head_[from]});
        head_[from] = static_cast<std::int32_t>(arcs_.size() - 1);
    }

    std::vector<DsuCycleEdge> witness(std::uint32_t from, std::uint32_t to) const
    {
        std::vector<std::int32_t> previous(parent_.size(), -1);
        std::vector<std::int32_t> previous_edge(parent_.size(), -1);
        std::deque<std::uint32_t> queue;
        previous[from] = static_cast<std::int32_t>(from);
        queue.push_back(from);
        while (!queue.empty() && previous[to] < 0) {
            const std::uint32_t node = queue.front();
            queue.pop_front();
            for (std::int32_t arc_index = head_[node]; arc_index >= 0;
                 arc_index = arcs_[static_cast<std::size_t>(arc_index)].next) {
                const Arc& arc = arcs_[static_cast<std::size_t>(arc_index)];
                if (previous[arc.to] >= 0) continue;
                previous[arc.to] = static_cast<std::int32_t>(node);
                previous_edge[arc.to] = static_cast<std::int32_t>(arc.edge);
                queue.push_back(arc.to);
            }
        }
        if (previous[to] < 0) throw std::logic_error("source DSU forest is disconnected");
        std::vector<DsuCycleEdge> reverse;
        for (std::uint32_t node = to; node != from;
             node = static_cast<std::uint32_t>(previous[node])) {
            const auto prior = static_cast<std::uint32_t>(previous[node]);
            const auto edge_index = static_cast<std::size_t>(previous_edge[node]);
            const Edge& edge = edges_[edge_index];
            const bool forward = edge.from == prior && edge.to == node;
            reverse.push_back({
                prior, node, forward ? edge.delta : -edge.delta,
                edge_index, false,
            });
        }
        std::reverse(reverse.begin(), reverse.end());
        return reverse;
    }

    std::vector<std::uint32_t> parent_;
    std::vector<std::uint32_t> size_;
    std::vector<std::int64_t> delta_;
    std::vector<std::int32_t> head_;
    std::vector<Edge> edges_;
    std::vector<Arc> arcs_;
};

class ScratchInt32 {
public:
    ScratchInt32(std::size_t count, const std::filesystem::path& directory)
        : count_(count)
    {
        if (count_ == 0) return;
        std::filesystem::create_directories(directory);
        path_ = directory / (
            ".track-potentials-" + std::to_string(::getpid()) + "-"
            + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".tmp");
        descriptor_ = ::open(
            path_.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (descriptor_ < 0) {
            throw std::system_error(
                errno, std::generic_category(), "cannot create " + path_.string());
        }
        if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t)) {
            close();
            throw std::overflow_error("track-potential scratch size overflows size_t");
        }
        const std::size_t bytes = count_ * sizeof(std::int32_t);
        if (::ftruncate(descriptor_, static_cast<off_t>(bytes)) != 0) {
            const int error = errno;
            close();
            throw std::system_error(
                error, std::generic_category(), "cannot size " + path_.string());
        }
        void* mapping = ::mmap(
            nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor_, 0);
        if (mapping == MAP_FAILED) {
            const int error = errno;
            close();
            throw std::system_error(
                error, std::generic_category(), "cannot map " + path_.string());
        }
        values_ = static_cast<std::int32_t*>(mapping);
    }

    ScratchInt32(const ScratchInt32&) = delete;
    ScratchInt32& operator=(const ScratchInt32&) = delete;
    ~ScratchInt32() { close(); }

    std::int32_t& operator[](std::size_t index) { return values_[index]; }
    const std::int32_t& operator[](std::size_t index) const { return values_[index]; }
    void release() noexcept { close(); }

private:
    void close() noexcept
    {
        if (values_) {
            ::munmap(values_, count_ * sizeof(std::int32_t));
            values_ = nullptr;
        }
        if (descriptor_ >= 0) {
            ::close(descriptor_);
            descriptor_ = -1;
        }
        if (!path_.empty()) {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }
    }

    std::filesystem::path path_;
    int descriptor_ = -1;
    std::int32_t* values_ = nullptr;
    std::size_t count_ = 0;
};

SourceHolonomy source_holonomy(
    const DsuConflict& source,
    std::string source_type,
    std::string path,
    std::string closing_item)
{
    Conflict conflict;
    conflict.kind = ConflictKind::source_theta;
    conflict.residual = source.residual;
    conflict.closing_constraint = {
        source.closing_from,
        source.closing_to,
        source.closing_delta,
        0,
        {source_type, path, std::move(closing_item), "source topology closing edge"},
        false,
    };
    conflict.cycle.reserve(source.cycle.size());
    for (const DsuCycleEdge& edge : source.cycle) {
        conflict.cycle.push_back({
            edge.from,
            edge.to,
            edge.delta,
            0,
            edge.edge_index,
            {source_type, path,
             std::to_string(edge.from) + "->" + std::to_string(edge.to),
             "source spanning-forest edge"},
            edge.closing,
        });
    }
    return SourceHolonomy(std::move(conflict));
}

} // namespace

const char* input_role_name(InputRole role) noexcept
{
    switch (role) {
    case InputRole::absolute: return "absolute";
    case InputRole::relative: return "relative";
    case InputRole::same_winding: return "same_winding";
    }
    return "unknown";
}

struct InputGraph::Impl {
    struct PatchRecord {
        std::filesystem::path path;
        std::shared_ptr<surfcore::SurfaceData> surface;
        PatchThetaTopology topology;
        PatchThetaTopology geometric_topology;
        trackio::SpatialBounds bounds;
        double area = 0.0;
        std::string fingerprint;
        bool valid = true;
        bool geometric_ready = false;
    };

    explicit Impl(GraphOptions value) : options(std::move(value))
    {
        if (!(options.contact_tolerance >= 0.0f)
            || !std::isfinite(options.contact_tolerance)) {
            throw std::invalid_argument("contact tolerance must be finite and nonnegative");
        }
        if (options.theta_batch_size == 0) {
            throw std::invalid_argument("theta batch size must be positive");
        }
        if (options.workers < 0) {
            throw std::invalid_argument("workers must be nonnegative");
        }
        if (!(options.fiber_coordinate_scale > 0.0f)
            || !std::isfinite(options.fiber_coordinate_scale)) {
            throw std::invalid_argument("fiber scale must be finite and positive");
        }
        if (options.surface_sampling_stride == 0
            || options.surface_sampling_stride
                > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("surface sampling stride is out of range");
        }
        if (std::isnan(options.z_min) || std::isnan(options.z_max)
            || !(options.z_min < options.z_max)) {
            throw std::invalid_argument("z range must be ordered and non-NaN");
        }
    }

    GraphOptions options;
    std::string theta_provider_key;
    WindingGraph graph;
    std::vector<PatchRecord> patches;
    surfcore::SurfacePatchIndex surface_index;
    std::vector<std::size_t> indexed_patches;
    std::unordered_map<std::string, std::size_t> patch_by_id;
    std::vector<SourceSpec> sources;
    std::unordered_set<std::string> source_keys;
    std::unordered_set<std::string> source_identities;

    std::optional<PatchRecord> load_patch(const std::filesystem::path& path) const
    {
        std::ifstream metadata_stream(path / "meta.json");
        if (!metadata_stream) throw std::runtime_error("missing " + (path / "meta.json").string());
        json metadata;
        metadata_stream >> metadata;
        const auto [rows, columns] = tiff_dimensions(path / "x.tif");
        if (tiff_dimensions(path / "y.tif") != std::pair{rows, columns}
            || tiff_dimensions(path / "z.tif") != std::pair{rows, columns}) {
            throw std::runtime_error("TIFXYZ coordinate plane dimensions differ");
        }
        if (rows < 2 || columns < 2) return std::nullopt;
        auto surface = std::make_shared<surfcore::SurfaceData>();
        surface->id = metadata.value("uuid", path.filename().string());
        if (surface->id.empty()) surface->id = path.filename().string();
        surface->rows = rows;
        surface->cols = columns;
        if (metadata.contains("scale") && metadata["scale"].is_array()
            && metadata["scale"].size() == 2) {
            surface->scale_i = metadata["scale"][0].get<float>();
            surface->scale_j = metadata["scale"][1].get<float>();
        }
        if (!(surface->scale_i > 0.0f) || !(surface->scale_j > 0.0f)) {
            throw std::runtime_error("TIFXYZ scale must be positive");
        }
        surface->point_source = surfcore::open_mapped_tifxyz_point_source(
            path, rows, columns);
        const std::vector<std::uint8_t> declared_mask
            = std::filesystem::is_regular_file(path / "mask.tif")
            ? surfcore::read_tifxyz_mask(path / "mask.tif", rows, columns)
            : std::vector<std::uint8_t>{};
        std::vector<std::uint8_t> valid_vertices(rows * columns, 0);
        trackio::SpatialBounds bounds{
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
        };
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                if (!declared_mask.empty()
                    && !declared_mask[row * columns + column]) continue;
                if (!surface->valid_point(row, column)) continue;
                valid_vertices[row * columns + column] = 1;
                const auto point = surface->at(row, column);
                bounds.z_min = std::min(bounds.z_min, point.z);
                bounds.y_min = std::min(bounds.y_min, point.y);
                bounds.x_min = std::min(bounds.x_min, point.x);
                bounds.z_max = std::max(bounds.z_max, point.z);
                bounds.y_max = std::max(bounds.y_max, point.y);
                bounds.x_max = std::max(bounds.x_max, point.x);
            }
        }
        std::vector<std::uint8_t> valid((rows - 1) * (columns - 1), 0);
        std::size_t valid_count = 0;
        for (std::size_t row = 0; row + 1 < rows; ++row) {
            for (std::size_t column = 0; column + 1 < columns; ++column) {
                valid[row * (columns - 1) + column]
                    = valid_vertices[row * columns + column]
                    && valid_vertices[row * columns + column + 1]
                    && valid_vertices[(row + 1) * columns + column]
                    && valid_vertices[(row + 1) * columns + column + 1];
                if (valid[row * (columns - 1) + column]) {
                    const float quad_z = 0.25f * (
                        surface->at(row, column).z
                        + surface->at(row, column + 1).z
                        + surface->at(row + 1, column).z
                        + surface->at(row + 1, column + 1).z);
                    valid[row * (columns - 1) + column]
                        = quad_z >= options.z_min && quad_z < options.z_max;
                }
                valid_count += valid[row * (columns - 1) + column] ? 1 : 0;
            }
        }
        if (valid_count == 0) return std::nullopt;
        PatchThetaTopology topology = PatchThetaTopology::from_mask(
            rows - 1, columns - 1, valid, true);
        PatchThetaTopology geometric_topology = PatchThetaTopology::from_mask(
            rows - 1, columns - 1, valid, true);
        surface->valid_quads = topology.valid_mask();
        const double area = static_cast<double>(topology.node_count())
            * surface->scale_i * surface->scale_j;
        return PatchRecord{
            path, std::move(surface), std::move(topology),
            std::move(geometric_topology), bounds, area,
            patch_fingerprint(path), true, false,
        };
    }

    void assign_patch_theta(
        std::vector<PatchRecord>& staged,
        const ThetaProviders& providers) const
    {
        std::vector<Zyx> coordinates;
        std::vector<std::pair<std::size_t, std::size_t>> owners;
        std::size_t total = 0;
        for (const auto& patch : staged) total += patch.topology.node_count();
        coordinates.reserve(total);
        owners.reserve(total);
        for (std::size_t patch_index = 0; patch_index < staged.size(); ++patch_index) {
            auto& patch = staged[patch_index];
            for (std::size_t node = 0; node < patch.topology.node_count(); ++node) {
                const auto [row, column] = patch.topology.cell(node);
                const auto a = patch.surface->at(row, column);
                const auto b = patch.surface->at(row, column + 1);
                const auto c = patch.surface->at(row + 1, column);
                const auto d = patch.surface->at(row + 1, column + 1);
                coordinates.push_back({
                    0.25f * (a.z + b.z + c.z + d.z),
                    0.25f * (a.y + b.y + c.y + d.y),
                    0.25f * (a.x + b.x + c.x + d.x),
                });
                owners.emplace_back(patch_index, node);
            }
        }
        const std::vector<float> theta = evaluate_theta(
            providers.reported, coordinates, options.theta_batch_size);
        const std::vector<float> geometric_theta = evaluate_theta(
            providers.geometric, coordinates, options.theta_batch_size);
        std::size_t offset = 0;
        for (auto& patch : staged) {
            const std::size_t count = patch.topology.node_count();
            const auto conflict = patch.topology.assign_theta(
                std::span(theta).subspan(offset, count));
            if (conflict) {
                Conflict graph_conflict;
                graph_conflict.kind = ConflictKind::patch_theta;
                graph_conflict.residual = conflict->residual;
                graph_conflict.closing_constraint = {
                    static_cast<NodeId>(conflict->from_node),
                    static_cast<NodeId>(conflict->to_node),
                    conflict->expected_delta,
                    0,
                    {"patch_theta", patch.path.string(), patch.surface->id,
                     "non-tree theta neighbor"},
                    false,
                };
                for (std::size_t edge = 0; edge < conflict->cycle.size(); ++edge) {
                    const auto& theta_edge = conflict->cycle[edge];
                    graph_conflict.cycle.push_back({
                        static_cast<NodeId>(theta_edge.from_node),
                        static_cast<NodeId>(theta_edge.to_node),
                        theta_edge.delta,
                        0,
                        edge,
                        {"patch_theta", patch.path.string(), patch.surface->id,
                         theta_edge.closing ? "closing neighbor" : "topology tree"},
                        theta_edge.closing,
                    });
                }
                throw SourceHolonomy(std::move(graph_conflict));
            }
            if (const auto conflict = patch.geometric_topology.assign_theta(
                    std::span(geometric_theta).subspan(offset, count))) {
                Conflict graph_conflict;
                graph_conflict.kind = ConflictKind::patch_theta;
                graph_conflict.residual = conflict->residual;
                graph_conflict.closing_constraint = {
                    static_cast<NodeId>(conflict->from_node),
                    static_cast<NodeId>(conflict->to_node),
                    conflict->expected_delta, 0,
                    {"geometric_theta", patch.path.string(), patch.surface->id,
                     "non-tree polar-theta neighbor"}, false,
                };
                throw SourceHolonomy(std::move(graph_conflict));
            }
            patch.geometric_ready = true;
            offset += count;
        }
    }

    void assign_geometric_patch_theta(
        std::vector<PatchRecord>& staged,
        const ThetaProvider& provider) const
    {
        if (!provider) {
            throw std::runtime_error(
                "legacy patch cache requires a theta provider with geometric_theta");
        }
        std::vector<Zyx> coordinates;
        std::size_t total = 0;
        for (const auto& patch : staged) total += patch.topology.node_count();
        coordinates.reserve(total);
        for (const auto& patch : staged) {
            for (std::size_t node = 0; node < patch.topology.node_count(); ++node) {
                const auto [row, column] = patch.topology.cell(node);
                const auto a = patch.surface->at(row, column);
                const auto b = patch.surface->at(row, column + 1);
                const auto c = patch.surface->at(row + 1, column);
                const auto d = patch.surface->at(row + 1, column + 1);
                coordinates.push_back({
                    0.25f * (a.z + b.z + c.z + d.z),
                    0.25f * (a.y + b.y + c.y + d.y),
                    0.25f * (a.x + b.x + c.x + d.x),
                });
            }
        }
        const std::vector<float> theta = evaluate_theta(
            provider, coordinates, options.theta_batch_size);
        std::size_t offset = 0;
        for (auto& patch : staged) {
            const std::size_t count = patch.geometric_topology.node_count();
            if (const auto conflict = patch.geometric_topology.assign_theta(
                    std::span(theta).subspan(offset, count))) {
                throw std::runtime_error(
                    "geometric patch theta is not liftable for "
                    + patch.surface->id + " (residual "
                    + std::to_string(conflict->residual) + ")");
            }
            patch.geometric_ready = true;
            offset += count;
        }
    }

    void rebuild_surface_index()
    {
        std::vector<std::shared_ptr<surfcore::SurfaceData>> surfaces;
        surfaces.reserve(patches.size());
        indexed_patches.clear();
        indexed_patches.reserve(patches.size());
        patch_by_id.clear();
        patch_by_id.reserve(patches.size());
        for (std::size_t index = 0; index < patches.size(); ++index) {
            patch_by_id.emplace(patches[index].surface->id, index);
            if (!patches[index].valid) continue;
            surfaces.push_back(patches[index].surface);
            indexed_patches.push_back(index);
        }
        surface_index.rebuild(
            std::move(surfaces), options.contact_tolerance,
            static_cast<int>(options.surface_sampling_stride));
    }

    std::optional<Attachment> attach(
        std::size_t point_index,
        const SourcePoint& point,
        std::int64_t source_potential,
        std::int64_t geometric_source_potential,
        surfcore::QueryScratch& scratch) const
    {
        std::vector<surfcore::SurfaceHit> hits;
        surface_index.query_point(
            {point.zyx.x, point.zyx.y, point.zyx.z},
            options.contact_tolerance, hits, scratch);
        if (hits.empty()) return std::nullopt;
        const surfcore::SurfaceHit* best = &hits.front();
        for (const auto& hit : hits) {
            const std::size_t patch = indexed_patches.at(
                static_cast<std::size_t>(hit.surface));
            const std::size_t best_patch = indexed_patches.at(
                static_cast<std::size_t>(best->surface));
            const double area = patches[patch].area;
            const double best_area = patches[best_patch].area;
            // Distance is the only reliable discriminator when candidate
            // patches represent neighboring windings.  Area-first selection
            // can attach a fiber to a substantially farther surface merely
            // because that patch is larger.  Retain area as a deterministic
            // tie-break for genuinely coincident/overlapping surfaces.
            if (hit.distance < best->distance
                || (hit.distance == best->distance && area > best_area)
                || (hit.distance == best->distance && area == best_area
                    && hit.surface < best->surface)) {
                best = &hit;
            }
        }
        const std::size_t patch = indexed_patches.at(
            static_cast<std::size_t>(best->surface));
        return Attachment{
            point_index,
            patch,
            patches[patch].topology.potential_at(best->j, best->i, point.theta),
            patches[patch].geometric_topology.potential_at(
                best->j, best->i, point.geometric_theta),
            source_potential,
            geometric_source_potential,
            best->distance,
            best->j,
            best->i,
        };
    }

    Constraint relation_constraint(
        const Attachment& a,
        const Attachment& b,
        std::int64_t source_delta,
        std::int64_t geometric_source_delta,
        Provenance provenance) const
    {
        // ThetaCrossingMap potentials are shifted-radius corrections. Physical
        // winding transport has the opposite sign, so for a source-reported
        // physical delta d:
        //   x_Q - x_P = d - p(P,a) + p(Q,b).
        const std::int64_t root_delta = source_delta
            - static_cast<std::int64_t>(a.patch_potential)
            + static_cast<std::int64_t>(b.patch_potential);
        const std::int64_t geometric_root_delta = geometric_source_delta
            - static_cast<std::int64_t>(a.geometric_patch_potential)
            + static_cast<std::int64_t>(b.geometric_patch_potential);
        return {
            graph.patch_node(patches[a.patch].surface->id),
            graph.patch_node(patches[b.patch].surface->id),
            root_delta,
            geometric_root_delta,
            std::move(provenance),
            false,
        };
    }

    std::vector<Constraint> point_collection_constraints(
        const SourceSpec& spec,
        const ThetaProviders& providers) const
    {
        std::vector<Constraint> constraints;
        for (const auto& path : spec.paths) {
            std::ifstream stream(path);
            if (!stream) throw std::runtime_error("cannot open " + path.string());
            json document;
            stream >> document;
            if (document.value("vc_pointcollections_json_version", "") != "1") {
                throw std::runtime_error("unsupported point-collection version in " + path.string());
            }
            const json& collections = document.at("collections");
            std::vector<std::pair<std::int64_t, const json*>> ordered_collections;
            for (auto item = collections.begin(); item != collections.end(); ++item) {
                ordered_collections.emplace_back(std::stoll(item.key()), &item.value());
            }
            std::sort(ordered_collections.begin(), ordered_collections.end());
            for (const auto& [collection_id, collection_ptr] : ordered_collections) {
                const json& collection = *collection_ptr;
                std::vector<SourcePoint> points;
                for (auto item = collection.at("points").begin();
                     item != collection.at("points").end(); ++item) {
                    SourcePoint point;
                    point.id = std::stoll(item.key());
                    point.zyx = xyz_json(item.value().at("p"));
                    if (item.value().contains("wind_a")
                        && !item.value().at("wind_a").is_null()) {
                        point.annotation = integer_annotation(
                            item.value().at("wind_a"), "wind_a");
                    }
                    points.push_back(point);
                }
                std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) {
                    return a.id < b.id;
                });
                points.erase(std::remove_if(
                    points.begin(), points.end(), [this](const auto& point) {
                        return point.zyx.z < options.z_min
                            || point.zyx.z >= options.z_max;
                    }), points.end());
                if (spec.role == InputRole::relative) {
                    points.erase(std::remove_if(points.begin(), points.end(), [](const auto& point) {
                        return !point.annotation.has_value();
                    }), points.end());
                }
                if (points.empty()) continue;
                std::vector<Zyx> zyx;
                zyx.reserve(points.size());
                for (const auto& point : points) zyx.push_back(point.zyx);
                const std::vector<float> theta = evaluate_theta(
                    providers.reported, zyx, options.theta_batch_size);
                const std::vector<float> geometric_theta = evaluate_theta(
                    providers.geometric, zyx, options.theta_batch_size);
                points.front().source_potential = 0;
                points.front().geometric_source_potential = 0;
                for (std::size_t index = 0; index < points.size(); ++index) {
                    points[index].theta = theta[index];
                    points[index].geometric_theta = geometric_theta[index];
                    if (index) {
                        points[index].source_potential
                            = points[index - 1].source_potential
                            + PatchThetaTopology::crossing_step(
                                theta[index] - theta[index - 1]);
                        points[index].geometric_source_potential
                            = points[index - 1].geometric_source_potential
                            + PatchThetaTopology::crossing_step(
                                geometric_theta[index]
                                - geometric_theta[index - 1]);
                    }
                }
                std::vector<Attachment> attached;
                surfcore::QueryScratch scratch;
                for (std::size_t index = 0; index < points.size(); ++index) {
                    if (auto hit = attach(
                            index, points[index], points[index].source_potential,
                            points[index].geometric_source_potential, scratch)) {
                        attached.push_back(*hit);
                    }
                }
                const std::string collection_name
                    = collection.value("name", std::to_string(collection_id));
                if (spec.role == InputRole::absolute) {
                    for (const Attachment& hit : attached) {
                        const SourcePoint& point = points[hit.point];
                        if (!point.annotation || *point.annotation <= 0) {
                            throw std::runtime_error(
                                "absolute point " + std::to_string(point.id)
                                + " has no positive integer wind_a");
                        }
                        constraints.push_back({
                            0,
                            graph.patch_node(patches[hit.patch].surface->id),
                            *point.annotation + hit.patch_potential,
                            *point.annotation + hit.geometric_patch_potential,
                            {"point_collection", path.string(),
                             collection_name + ":" + std::to_string(point.id),
                             "absolute attachment; raw="
                                + std::to_string(*point.annotation)
                                + "; patch_theta="
                                + std::to_string(hit.patch_potential)
                                + "; sampled_theta="
                                + std::to_string(point.theta)
                                + "; distance="
                                + std::to_string(hit.distance)
                                + "; ij=(" + std::to_string(hit.row)
                                + "," + std::to_string(hit.column) + ")"},
                            true,
                        });
                    }
                    continue;
                }
                for (std::size_t index = 1; index < attached.size(); ++index) {
                    const Attachment& a = attached[index - 1];
                    const Attachment& b = attached[index];
                    const SourcePoint& pa = points[a.point];
                    const SourcePoint& pb = points[b.point];
                    const std::int64_t raw = spec.role == InputRole::relative
                        ? *pb.annotation - *pa.annotation : 0;
                    const std::int64_t branch
                        = pa.source_potential - pb.source_potential;
                    const std::int64_t geometric_branch
                        = pa.geometric_source_potential
                        - pb.geometric_source_potential;
                    constraints.push_back(relation_constraint(
                        a, b, raw + branch, geometric_branch,
                        {"point_collection", path.string(),
                         collection_name + ":" + std::to_string(pa.id)
                            + "->" + std::to_string(pb.id),
                         std::string(input_role_name(spec.role))
                            + "; raw=" + std::to_string(raw)
                            + "; theta_branch=" + std::to_string(branch)
                            + "; geometric_theta_branch="
                            + std::to_string(geometric_branch)}));
                }
            }
        }
        return constraints;
    }

    std::vector<Constraint> track_constraints(
        const SourceSpec& spec,
        const ThetaProviders& providers) const
    {
        trackio::PackedTrackStore tracks(spec.paths[0]);
        trackio::CrossingStore crossings(spec.paths[1], tracks);
        const std::size_t point_count = static_cast<std::size_t>(tracks.point_count);
        const std::size_t track_count = static_cast<std::size_t>(tracks.track_count);
        std::vector<std::uint8_t> retained_tracks(track_count, 0);
        for (std::size_t track = 0; track < track_count; ++track) {
            retained_tracks[track]
                = static_cast<float>(tracks.z_bounds[2 * track]) >= options.z_min
                && static_cast<float>(tracks.z_bounds[2 * track + 1]) < options.z_max;
        }
        const bool indexed_candidates = spec.paths.size() >= 3 && !spec.paths[2].empty();
        std::vector<std::uint64_t> candidate_bits;
        if (indexed_candidates) candidate_bits.assign((point_count + 63) / 64, 0);
        if (spec.paths.size() >= 3 && !spec.paths[2].empty()) {
            trackio::TrackSpatialIndex spatial_index;
            spatial_index.open(spec.paths[2]);
            spatial_index.validate_source(tracks);
            std::vector<std::uint64_t> patch_candidates;
            for (const auto& patch : patches) {
                auto bounds = patch.bounds;
                bounds.z_min -= options.contact_tolerance;
                bounds.y_min -= options.contact_tolerance;
                bounds.x_min -= options.contact_tolerance;
                bounds.z_max += options.contact_tolerance;
                bounds.y_max += options.contact_tolerance;
                bounds.x_max += options.contact_tolerance;
                patch_candidates.clear();
                spatial_index.query(bounds, patch_candidates);
                for (const std::uint64_t point : patch_candidates) {
                    if (point >= point_count) {
                        throw std::runtime_error("track spatial index has an invalid point id");
                    }
                    candidate_bits[static_cast<std::size_t>(point / 64)]
                        |= std::uint64_t{1} << (point % 64);
                }
            }
        }
        const std::filesystem::path scratch_directory = indexed_candidates
            ? spec.paths[2] : std::filesystem::temp_directory_path();
        ScratchInt32 local_potential(point_count, scratch_directory);
        ScratchInt32 geometric_local_potential(point_count, scratch_directory);
        struct RawContact {
            Attachment attachment;
            std::uint32_t track = 0;
            std::uint32_t local = 0;
            std::uint32_t root = 0;
            std::int64_t potential = 0;
            std::int64_t geometric_potential = 0;
        };
        std::vector<RawContact> raw_contacts;
        std::size_t track_cursor = 0;
        float preceding_theta = 0.0f;
        float preceding_geometric_theta = 0.0f;
        bool have_preceding_theta = false;
        std::size_t preceding_point = 0;
        for (std::size_t chunk_begin = 0; chunk_begin < point_count;
             chunk_begin += options.theta_batch_size) {
            const std::size_t chunk_end = std::min(
                point_count, chunk_begin + options.theta_batch_size);
            std::vector<Zyx> zyx;
            std::vector<std::size_t> global_points;
            std::vector<std::size_t> point_tracks;
            std::vector<std::size_t> point_locals;
            zyx.reserve(chunk_end - chunk_begin);
            global_points.reserve(chunk_end - chunk_begin);
            point_tracks.reserve(chunk_end - chunk_begin);
            point_locals.reserve(chunk_end - chunk_begin);
            for (std::size_t point = chunk_begin; point < chunk_end; ++point) {
                while (track_cursor + 1 < track_count
                       && point >= tracks.point_end(track_cursor)) {
                    ++track_cursor;
                }
                if (!retained_tracks[track_cursor]) continue;
                zyx.push_back({
                    static_cast<float>(tracks.coords[3 * point]),
                    static_cast<float>(tracks.coords[3 * point + 1]),
                    static_cast<float>(tracks.coords[3 * point + 2]),
                });
                global_points.push_back(point);
                point_tracks.push_back(track_cursor);
                point_locals.push_back(point - tracks.point_begin(track_cursor));
            }
            const std::vector<float> theta = evaluate_theta(
                providers.reported, zyx, options.theta_batch_size);
            const std::vector<float> geometric_theta = evaluate_theta(
                providers.geometric, zyx, options.theta_batch_size);
            for (std::size_t offset = 0; offset < theta.size(); ++offset) {
                const std::size_t point = global_points[offset];
                const std::size_t begin = tracks.point_begin(point_tracks[offset]);
                if (point == begin) {
                    local_potential[point] = 0;
                    geometric_local_potential[point] = 0;
                } else {
                    const bool previous_is_in_batch
                        = offset != 0 && global_points[offset - 1] + 1 == point;
                    if (!previous_is_in_batch
                        && (!have_preceding_theta || preceding_point + 1 != point)) {
                        throw std::logic_error("missing theta at track chunk boundary");
                    }
                    const float previous_theta
                        = previous_is_in_batch ? theta[offset - 1] : preceding_theta;
                    local_potential[point] = local_potential[point - 1]
                        + PatchThetaTopology::crossing_step(
                            theta[offset] - previous_theta);
                    const float previous_geometric_theta = previous_is_in_batch
                        ? geometric_theta[offset - 1]
                        : preceding_geometric_theta;
                    geometric_local_potential[point]
                        = geometric_local_potential[point - 1]
                        + PatchThetaTopology::crossing_step(
                            geometric_theta[offset] - previous_geometric_theta);
                }
            }
            if (!theta.empty()) {
                preceding_theta = theta.back();
                preceding_geometric_theta = geometric_theta.back();
                have_preceding_theta = true;
                preceding_point = global_points.back();
            }
            std::vector<std::size_t> candidate_offsets;
            if (!indexed_candidates) {
                candidate_offsets.resize(zyx.size());
                std::iota(candidate_offsets.begin(), candidate_offsets.end(), 0);
            } else {
                for (std::size_t offset = 0; offset < global_points.size(); ++offset) {
                    const std::size_t point = global_points[offset];
                    if (candidate_bits[point / 64] & (std::uint64_t{1} << (point % 64))) {
                        candidate_offsets.push_back(offset);
                    }
                }
            }
            std::vector<std::optional<Attachment>> batch_hits(candidate_offsets.size());
#ifdef _OPENMP
            const int worker_count = options.workers > 0
                ? options.workers : omp_get_max_threads();
#pragma omp parallel num_threads(worker_count)
#else
#pragma omp parallel
#endif
            {
                surfcore::QueryScratch scratch;
#pragma omp for schedule(static)
                for (std::int64_t signed_candidate = 0;
                     signed_candidate < static_cast<std::int64_t>(candidate_offsets.size());
                     ++signed_candidate) {
                    const std::size_t candidate = static_cast<std::size_t>(signed_candidate);
                    const std::size_t offset = candidate_offsets[candidate];
                    SourcePoint point{
                        static_cast<std::int64_t>(point_locals[offset]),
                        zyx[offset], {}, theta[offset],
                        geometric_theta[offset],
                        local_potential[global_points[offset]],
                        geometric_local_potential[global_points[offset]],
                    };
                    batch_hits[candidate] = attach(
                        offset, point, local_potential[global_points[offset]],
                        geometric_local_potential[global_points[offset]], scratch);
                }
            }
            for (std::size_t candidate = 0; candidate < candidate_offsets.size(); ++candidate) {
                if (!batch_hits[candidate]) continue;
                const std::size_t offset = candidate_offsets[candidate];
                RawContact contact{
                    *batch_hits[candidate],
                    static_cast<std::uint32_t>(point_tracks[offset]),
                    static_cast<std::uint32_t>(point_locals[offset]),
                    0,
                    0,
                    0,
                };
                // Dense tracks can contribute hundreds of consecutive points
                // with the exact same patch equation. Retain run boundaries
                // and every changed equation, which is sufficient for both
                // connectivity and holonomy detection.
                const auto equation = [](const RawContact& value) {
                    return std::pair{
                        static_cast<std::int64_t>(
                        value.attachment.patch_potential)
                            - value.attachment.source_potential,
                        static_cast<std::int64_t>(
                            value.attachment.geometric_patch_potential)
                            - value.attachment.geometric_source_potential};
                };
                if (!raw_contacts.empty()
                    && raw_contacts.back().track == contact.track
                    && raw_contacts.back().attachment.patch
                        == contact.attachment.patch
                    && equation(raw_contacts.back()) == equation(contact)) {
                    continue;
                }
                raw_contacts.push_back(std::move(contact));
            }
        }

        WeightedDsu components(track_count);
        WeightedDsu geometric_components(track_count);
        for (std::size_t track = 0; track < track_count; ++track) {
            if (!retained_tracks[track]) continue;
            for (std::size_t record = crossings.begin(track);
                 record < crossings.end(track); ++record) {
                const std::int32_t partner_signed = crossings.partners[record];
                if (partner_signed < 0) continue;
                const std::size_t partner = static_cast<std::size_t>(partner_signed);
                if (partner >= track_count || partner < track) continue;
                if (!retained_tracks[partner]) continue;
                if (partner == track
                    && crossings.partner_local[record] <= crossings.self_local[record]) continue;
                const auto self_local = static_cast<std::size_t>(crossings.self_local[record]);
                const auto partner_local = static_cast<std::size_t>(crossings.partner_local[record]);
                const std::size_t self_point = tracks.point_begin(track) + self_local;
                const std::size_t partner_point = tracks.point_begin(partner) + partner_local;
                if (self_point >= tracks.point_end(track)
                    || partner_point >= tracks.point_end(partner)) {
                    throw std::runtime_error("track crossing local index is out of range");
                }
                const std::int64_t required
                    = static_cast<std::int64_t>(local_potential[partner_point])
                    - static_cast<std::int64_t>(local_potential[self_point]);
                if (const auto conflict = components.add(track, partner, required)) {
                    throw source_holonomy(
                        *conflict, "vctracks", spec.paths[0].string(),
                        std::to_string(track) + ":" + std::to_string(self_local)
                            + "->" + std::to_string(partner) + ":"
                            + std::to_string(partner_local));
                }
                const std::int64_t geometric_required
                    = static_cast<std::int64_t>(
                        geometric_local_potential[partner_point])
                    - static_cast<std::int64_t>(
                        geometric_local_potential[self_point]);
                if (const auto conflict = geometric_components.add(
                        track, partner, geometric_required)) {
                    throw source_holonomy(
                        *conflict, "geometric_vctracks", spec.paths[0].string(),
                        std::to_string(track) + ":" + std::to_string(self_local)
                            + "->" + std::to_string(partner) + ":"
                            + std::to_string(partner_local));
                }
            }
        }

        for (RawContact& raw : raw_contacts) {
            const auto [root, frame] = components.find(raw.track);
            const auto [geometric_root, geometric_frame]
                = geometric_components.find(raw.track);
            if (geometric_root != root) {
                throw std::logic_error("reported and geometric track components differ");
            }
            raw.root = root;
            raw.potential = frame - raw.attachment.source_potential;
            raw.geometric_potential
                = geometric_frame - raw.attachment.geometric_source_potential;
        }
        local_potential.release();
        geometric_local_potential.release();

        // Group in-place instead of duplicating every contact into a tree of
        // vectors. Within one source component, one equation per patch is
        // enough. A duplicate equation with a different value is retained as
        // a self-constraint so the graph transaction reports the holonomy.
        std::sort(raw_contacts.begin(), raw_contacts.end(), [](const auto& a, const auto& b) {
            if (a.root != b.root) return a.root < b.root;
            if (a.attachment.patch != b.attachment.patch) {
                return a.attachment.patch < b.attachment.patch;
            }
            if (a.track != b.track) return a.track < b.track;
            return a.local < b.local;
        });
        std::vector<Constraint> constraints;
        const auto item = [](const RawContact& contact) {
            return std::to_string(contact.track) + ":"
                + std::to_string(contact.local);
        };
        for (std::size_t root_begin = 0; root_begin < raw_contacts.size();) {
            std::size_t root_end = root_begin + 1;
            while (root_end < raw_contacts.size()
                   && raw_contacts[root_end].root == raw_contacts[root_begin].root) {
                ++root_end;
            }
            std::size_t representative = root_begin;
            for (std::size_t patch_begin = root_begin;
                 patch_begin < root_end;) {
                std::size_t patch_end = patch_begin + 1;
                while (patch_end < root_end
                       && raw_contacts[patch_end].attachment.patch
                           == raw_contacts[patch_begin].attachment.patch) {
                    ++patch_end;
                }
                const RawContact& patch_contact = raw_contacts[patch_begin];
                if (patch_begin != root_begin) {
                    const RawContact& component_contact = raw_contacts[representative];
                    constraints.push_back(relation_constraint(
                        component_contact.attachment,
                        patch_contact.attachment,
                        patch_contact.potential - component_contact.potential,
                        patch_contact.geometric_potential
                            - component_contact.geometric_potential,
                        {"vctracks", spec.paths[0].string(),
                         item(component_contact) + "->" + item(patch_contact),
                         "same_winding; crossing-connected track component"}));
                }
                const std::int64_t expected
                    = patch_contact.potential
                    + patch_contact.attachment.patch_potential;
                const std::int64_t geometric_expected
                    = patch_contact.geometric_potential
                    + patch_contact.attachment.geometric_patch_potential;
                for (std::size_t duplicate = patch_begin + 1;
                     duplicate < patch_end; ++duplicate) {
                    const RawContact& next = raw_contacts[duplicate];
                    const std::int64_t observed
                        = next.potential + next.attachment.patch_potential;
                    const std::int64_t geometric_observed
                        = next.geometric_potential
                        + next.attachment.geometric_patch_potential;
                    if (observed == expected
                        && geometric_observed == geometric_expected) continue;
                    constraints.push_back(relation_constraint(
                        patch_contact.attachment,
                        next.attachment,
                        next.potential - patch_contact.potential,
                        next.geometric_potential
                            - patch_contact.geometric_potential,
                        {"vctracks", spec.paths[0].string(),
                         item(patch_contact) + "->" + item(next),
                         "same patch reached with inconsistent transported winding"}));
                }
                patch_begin = patch_end;
            }
            root_begin = root_end;
        }
        return constraints;
    }

    std::vector<Constraint> constraints_for(
        const SourceSpec& source,
        const ThetaProviders& providers) const
    {
        switch (source.kind) {
        case SourceSpec::Kind::point_collection:
            return point_collection_constraints(source, providers);
        case SourceSpec::Kind::tracks:
            return track_constraints(source, providers);
        }
        throw std::logic_error("unknown source kind");
    }
};

InputGraph::InputGraph(GraphOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
InputGraph::~InputGraph() = default;
InputGraph::InputGraph(InputGraph&&) noexcept = default;
InputGraph& InputGraph::operator=(InputGraph&&) noexcept = default;
WindingGraph& InputGraph::graph() noexcept { return impl_->graph; }
const WindingGraph& InputGraph::graph() const noexcept { return impl_->graph; }
const GraphOptions& InputGraph::options() const noexcept { return impl_->options; }
const std::string& InputGraph::theta_provider_key() const noexcept
{
    return impl_->theta_provider_key;
}

void InputGraph::set_theta_provider_key(std::string key)
{
    if (!impl_->theta_provider_key.empty()) {
        if (key.empty()) {
            throw std::runtime_error(
                "cached graph requires a theta provider with a cache_key");
        }
        if (impl_->theta_provider_key != key) {
            throw std::runtime_error(
                "theta provider does not match the graph cache");
        }
        return;
    }
    impl_->theta_provider_key = std::move(key);
}

AddResult InputGraph::add_patches(
    const std::vector<std::filesystem::path>& paths,
    const ThetaProviders& theta_providers)
{
    const auto discovered = discover_patch_paths(paths);
    std::vector<Impl::PatchRecord> staged;
    for (const auto& path : discovered) {
        auto loaded = impl_->load_patch(path);
        if (!loaded) continue;
        auto& patch = *loaded;
        if (const auto found = impl_->patch_by_id.find(patch.surface->id);
            found != impl_->patch_by_id.end()) {
            const auto& existing = impl_->patches[found->second];
            if (canonical_text(existing.path) != canonical_text(patch.path)
                || existing.fingerprint != patch.fingerprint) {
                throw std::runtime_error(
                    "patch id already exists with different content: "
                    + patch.surface->id);
            }
            continue;
        }
        staged.push_back(std::move(patch));
    }
    if (staged.empty()) {
        AddResult result;
        result.committed = true;
        result.already_present = !discovered.empty();
        return result;
    }
    try {
        impl_->assign_patch_theta(staged, theta_providers);
    } catch (const SourceHolonomy& error) {
        AddResult result;
        result.conflict = error.conflict;
        return result;
    }
    const std::size_t patch_marker = impl_->patches.size();
    const std::size_t node_marker = impl_->graph.node_count();
    try {
        for (auto& patch : staged) {
            impl_->graph.ensure_patch(patch.surface->id);
            impl_->patches.push_back(std::move(patch));
        }
        impl_->rebuild_surface_index();
        std::vector<Constraint> constraints;
        for (const auto& source : impl_->sources) {
            auto added = impl_->constraints_for(source, theta_providers);
            constraints.insert(constraints.end(),
                               std::make_move_iterator(added.begin()),
                               std::make_move_iterator(added.end()));
        }
        // Replaying registered sources is necessary so a late patch can find
        // contacts through their persistent indexes. Preserve append-only
        // semantics by adding only assertions not already in the graph.
        discard_existing_constraints(impl_->graph, constraints);
        AddResult result = impl_->graph.add_constraints(constraints);
        if (!result.committed) {
            impl_->patches.resize(patch_marker);
            impl_->graph.discard_trailing_patches(node_marker);
            impl_->rebuild_surface_index();
            return result;
        }
        result.nodes_added = impl_->patches.size() - patch_marker;
        return result;
    } catch (const SourceHolonomy& error) {
        impl_->patches.resize(patch_marker);
        impl_->graph.discard_trailing_patches(node_marker);
        impl_->rebuild_surface_index();
        AddResult result;
        result.conflict = error.conflict;
        return result;
    } catch (...) {
        impl_->patches.resize(patch_marker);
        impl_->graph.discard_trailing_patches(node_marker);
        impl_->rebuild_surface_index();
        throw;
    }
}

bool InputGraph::set_patch_valid(const std::string& patch_id, bool valid)
{
    const auto found = impl_->patch_by_id.find(patch_id);
    if (found == impl_->patch_by_id.end()) {
        throw std::out_of_range("unknown patch id: " + patch_id);
    }
    auto& patch = impl_->patches[found->second];
    if (patch.valid == valid) return false;
    if (!impl_->sources.empty() || impl_->graph.stats().constraint_count != 0) {
        throw std::runtime_error(
            "patch validity can only change before dependent sources or "
            "constraints are committed; rebuild or reopen a patch-only cache");
    }
    patch.valid = valid;
    impl_->rebuild_surface_index();
    return true;
}

bool InputGraph::patch_valid(const std::string& patch_id) const
{
    const auto found = impl_->patch_by_id.find(patch_id);
    if (found == impl_->patch_by_id.end()) {
        throw std::out_of_range("unknown patch id: " + patch_id);
    }
    return impl_->patches[found->second].valid;
}

std::vector<std::vector<ContactHit>> InputGraph::inspect_contacts(
    std::span<const Zyx> points,
    std::optional<float> tolerance) const
{
    const float query_tolerance = tolerance.value_or(impl_->options.contact_tolerance);
    if (!(query_tolerance >= 0.0f) || !std::isfinite(query_tolerance)) {
        throw std::invalid_argument("contact tolerance must be finite and nonnegative");
    }
    std::vector<std::vector<ContactHit>> output(points.size());
    std::vector<surfcore::SurfaceHit> hits;
    surfcore::QueryScratch scratch;
    for (std::size_t point = 0; point < points.size(); ++point) {
        hits.clear();
        impl_->surface_index.query_point(
            {points[point].x, points[point].y, points[point].z},
            query_tolerance, hits, scratch);
        auto& contacts = output[point];
        contacts.reserve(hits.size());
        for (const auto& hit : hits) {
            const std::size_t patch = impl_->indexed_patches.at(
                static_cast<std::size_t>(hit.surface));
            contacts.push_back({
                impl_->patches[patch].surface->id,
                hit.distance,
                hit.j,
                hit.i,
            });
        }
        std::sort(contacts.begin(), contacts.end(), [](const auto& a, const auto& b) {
            if (a.distance != b.distance) return a.distance < b.distance;
            return a.patch_id < b.patch_id;
        });
    }
    return output;
}

PatchLayoutData InputGraph::patch_layout(const std::string& patch_id) const
{
    const auto found = impl_->patch_by_id.find(patch_id);
    if (found == impl_->patch_by_id.end()) {
        throw std::out_of_range("unknown patch id: " + patch_id);
    }
    const auto& patch = impl_->patches[found->second];
    if (!patch.geometric_ready) {
        throw std::runtime_error(
            "geometric patch theta is unavailable for " + patch_id);
    }
    const std::size_t count = patch.topology.node_count();
    PatchLayoutData output;
    output.patch_id = patch.surface->id;
    output.source_path = canonical_text(patch.path);
    output.quad_rows = patch.topology.rows();
    output.quad_columns = patch.topology.columns();
    output.quad_ij.reserve(count * 2);
    output.zyx.reserve(count * 3);
    output.reported_local_turn.reserve(count);
    output.geometric_local_turn.reserve(count);
    constexpr double turns_per_radian
        = 1.0 / (2.0 * 3.141592653589793238462643383279502884);
    for (std::size_t node = 0; node < count; ++node) {
        const auto [row, column] = patch.topology.cell(node);
        if (row > std::numeric_limits<std::uint32_t>::max()
            || column > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("patch layout index exceeds uint32");
        }
        const auto a = patch.surface->at(row, column);
        const auto b = patch.surface->at(row, column + 1);
        const auto c = patch.surface->at(row + 1, column);
        const auto d = patch.surface->at(row + 1, column + 1);
        output.quad_ij.push_back(static_cast<std::uint32_t>(row));
        output.quad_ij.push_back(static_cast<std::uint32_t>(column));
        output.zyx.push_back(0.25f * (a.z + b.z + c.z + d.z));
        output.zyx.push_back(0.25f * (a.y + b.y + c.y + d.y));
        output.zyx.push_back(0.25f * (a.x + b.x + c.x + d.x));
        output.reported_local_turn.push_back(static_cast<float>(
            -static_cast<double>(patch.topology.potential(node))
            + static_cast<double>(patch.topology.theta(node))
                * turns_per_radian));
        output.geometric_local_turn.push_back(static_cast<float>(
            -static_cast<double>(patch.geometric_topology.potential(node))
            + static_cast<double>(patch.geometric_topology.theta(node))
                * turns_per_radian));
    }
    output.vertex_rows = output.quad_rows + 1;
    output.vertex_columns = output.quad_columns + 1;
    const std::size_t vertex_count = output.vertex_rows * output.vertex_columns;
    std::vector<double> vertex_turn_sum(vertex_count, 0.0);
    std::vector<std::uint32_t> vertex_turn_count(vertex_count, 0);
    const auto local_turn = [&](std::size_t node) {
        return static_cast<double>(output.reported_local_turn[node]);
    };
    const auto derivative = [&](std::size_t node, int row_delta, int column_delta) {
        const auto [row, column] = patch.topology.cell(node);
        const auto before_row = static_cast<std::int64_t>(row) - row_delta;
        const auto before_column = static_cast<std::int64_t>(column) - column_delta;
        const auto after_row = static_cast<std::int64_t>(row) + row_delta;
        const auto after_column = static_cast<std::int64_t>(column) + column_delta;
        std::optional<std::size_t> before;
        std::optional<std::size_t> after;
        if (before_row >= 0 && before_column >= 0) {
            before = patch.topology.node_at(
                static_cast<std::size_t>(before_row),
                static_cast<std::size_t>(before_column));
        }
        if (after_row >= 0 && after_column >= 0) {
            after = patch.topology.node_at(
                static_cast<std::size_t>(after_row),
                static_cast<std::size_t>(after_column));
        }
        if (before && after) {
            return 0.5 * (local_turn(*after) - local_turn(*before));
        }
        if (after) return local_turn(*after) - local_turn(node);
        if (before) return local_turn(node) - local_turn(*before);
        return 0.0;
    };
    for (std::size_t node = 0; node < count; ++node) {
        const auto [row, column] = patch.topology.cell(node);
        const double row_derivative = derivative(node, 1, 0);
        const double column_derivative = derivative(node, 0, 1);
        for (std::size_t dr = 0; dr <= 1; ++dr) {
            for (std::size_t dc = 0; dc <= 1; ++dc) {
                const std::size_t vertex = (row + dr) * output.vertex_columns
                    + column + dc;
                vertex_turn_sum[vertex] += local_turn(node)
                    + (static_cast<double>(dr) - 0.5) * row_derivative
                    + (static_cast<double>(dc) - 0.5) * column_derivative;
                ++vertex_turn_count[vertex];
            }
        }
    }
    output.vertex_ij.reserve(vertex_count * 2);
    output.vertex_zyx.reserve(vertex_count * 3);
    output.reported_vertex_turn.reserve(vertex_count);
    for (std::size_t row = 0; row < output.vertex_rows; ++row) {
        for (std::size_t column = 0; column < output.vertex_columns; ++column) {
            const std::size_t vertex = row * output.vertex_columns + column;
            if (vertex_turn_count[vertex] == 0) continue;
            if (row > std::numeric_limits<std::uint32_t>::max()
                || column > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("patch vertex layout index exceeds uint32");
            }
            const auto point = patch.surface->at(row, column);
            output.vertex_ij.push_back(static_cast<std::uint32_t>(row));
            output.vertex_ij.push_back(static_cast<std::uint32_t>(column));
            output.vertex_zyx.push_back(point.z);
            output.vertex_zyx.push_back(point.y);
            output.vertex_zyx.push_back(point.x);
            output.reported_vertex_turn.push_back(static_cast<float>(
                vertex_turn_sum[vertex] / vertex_turn_count[vertex]));
        }
    }
    return output;
}

AddResult InputGraph::add_point_collections(
    const std::vector<std::filesystem::path>& paths,
    InputRole role,
    const ThetaProviders& theta_providers)
{
    SourceSpec source;
    source.kind = SourceSpec::Kind::point_collection;
    source.paths = paths;
    std::sort(source.paths.begin(), source.paths.end());
    source.role = role;
    const std::string identity = source_identity(source);
    const std::string key = source_key(source);
    if (impl_->source_identities.contains(identity)) {
        if (!impl_->source_keys.contains(key)) {
            throw std::runtime_error(
                "source content changed; append-only graphs require a rebuild: "
                + identity);
        }
        AddResult result;
        result.committed = true;
        result.already_present = true;
        return result;
    }
    std::vector<Constraint> constraints;
    try {
        constraints = impl_->constraints_for(source, theta_providers);
    } catch (const SourceHolonomy& error) {
        AddResult result;
        result.conflict = error.conflict;
        return result;
    }
    AddResult result = impl_->graph.add_constraints(constraints);
    if (result.committed) {
        impl_->source_keys.insert(key);
        impl_->source_identities.insert(identity);
        impl_->sources.push_back(std::move(source));
    }
    return result;
}

std::vector<Constraint> InputGraph::inspect_point_collections(
    const std::vector<std::filesystem::path>& paths,
    InputRole role,
    const ThetaProviders& theta_providers) const
{
    SourceSpec source;
    source.kind = SourceSpec::Kind::point_collection;
    source.paths = paths;
    std::sort(source.paths.begin(), source.paths.end());
    source.role = role;
    return impl_->constraints_for(source, theta_providers);
}

AddResult InputGraph::add_fibers(
    const std::filesystem::path& directory,
    float coordinate_scale,
    std::vector<std::string> invalid_fibers)
{
    if (!(coordinate_scale > 0.0f) || !std::isfinite(coordinate_scale)) {
        throw std::invalid_argument("fiber scale must be finite and positive");
    }
    SourceSpec source;
    source.kind = SourceSpec::Kind::fibers;
    source.paths = {directory};
    source.role = InputRole::same_winding;
    source.coordinate_scale = coordinate_scale;
    std::sort(invalid_fibers.begin(), invalid_fibers.end());
    invalid_fibers.erase(
        std::unique(invalid_fibers.begin(), invalid_fibers.end()),
        invalid_fibers.end());
    source.invalid_items = std::move(invalid_fibers);
    const std::string identity = source_identity(source);
    const std::string key = source_key(source);
    if (impl_->source_identities.contains(identity)) {
        if (!impl_->source_keys.contains(key)) {
            throw std::runtime_error(
                "fiber content changed; append-only graphs require a rebuild");
        }
        AddResult result;
        result.committed = true;
        result.already_present = true;
        return result;
    }
    AddResult result;
    result.committed = true;
    impl_->source_keys.insert(key);
    impl_->source_identities.insert(identity);
    impl_->sources.push_back(std::move(source));
    return result;
}

AddResult InputGraph::add_tracks(
    const std::filesystem::path& tracks,
    const std::filesystem::path& crossings,
    const std::filesystem::path& spatial_index,
    const ThetaProviders& theta_providers)
{
    SourceSpec source;
    source.kind = SourceSpec::Kind::tracks;
    source.paths = {tracks, crossings, spatial_index};
    source.role = InputRole::same_winding;
    const std::string identity = source_identity(source);
    const std::string key = source_key(source);
    if (impl_->source_identities.contains(identity)) {
        if (!impl_->source_keys.contains(key)) {
            throw std::runtime_error(
                "track source changed; append-only graphs require a rebuild");
        }
        AddResult result;
        result.committed = true;
        result.already_present = true;
        return result;
    }
    std::vector<Constraint> constraints;
    try {
        constraints = impl_->constraints_for(source, theta_providers);
    } catch (const SourceHolonomy& error) {
        AddResult result;
        result.conflict = error.conflict;
        return result;
    }
    AddResult result = impl_->graph.add_constraints(constraints);
    if (result.committed) {
        impl_->source_keys.insert(key);
        impl_->source_identities.insert(identity);
        impl_->sources.push_back(std::move(source));
    }
    return result;
}

TrackIndexInfo InputGraph::prepare_track_index(
    const std::filesystem::path& tracks_path,
    const std::filesystem::path& output,
    std::uint32_t cell_size,
    std::size_t memory_budget_bytes) const
{
    trackio::PackedTrackStore tracks(tracks_path);
    const auto info = trackio::TrackSpatialIndex::build(
        tracks, output, cell_size, memory_budget_bytes);
    return {info.point_count, info.cell_count, info.cell_size, info.already_present};
}

TrackInfo InputGraph::inspect_tracks(
    const std::filesystem::path& tracks_path,
    const std::filesystem::path& crossings_path) const
{
    trackio::PackedTrackStore tracks(tracks_path);
    TrackInfo info{
        static_cast<std::uint64_t>(tracks.track_count),
        static_cast<std::uint64_t>(tracks.point_count),
        0,
    };
    if (!crossings_path.empty()) {
        trackio::CrossingStore crossings(crossings_path, tracks);
        info.crossings = crossings.records;
    }
    return info;
}

void InputGraph::save(const std::filesystem::path& cache_directory) const
{
    for (const auto& patch : impl_->patches) {
        if (!patch.geometric_ready) {
            throw std::runtime_error(
                "cannot save graph before geometric patch theta is available");
        }
    }
    std::filesystem::create_directories(cache_directory);
    impl_->graph.save(cache_directory / "graph.bin");
    json manifest{
        {"schema", "spiral-winding-graph"},
        {"version", 2},
        {"theta_provider_key", impl_->theta_provider_key},
        {"options", {
            {"contact_tolerance", impl_->options.contact_tolerance},
            {"theta_batch_size", impl_->options.theta_batch_size},
            {"fiber_coordinate_scale", impl_->options.fiber_coordinate_scale},
            {"surface_sampling_stride", impl_->options.surface_sampling_stride},
            {"z_min", std::isfinite(impl_->options.z_min)
                ? json(impl_->options.z_min) : json(nullptr)},
            {"z_max", std::isfinite(impl_->options.z_max)
                ? json(impl_->options.z_max) : json(nullptr)},
        }},
        {"patches", json::array()},
        {"sources", json::array()},
    };
    std::ofstream theta_stream(
        cache_directory / "patch_theta.f32.tmp", std::ios::binary | std::ios::trunc);
    std::ofstream geometric_theta_stream(
        cache_directory / "patch_geometric_theta.f32.tmp",
        std::ios::binary | std::ios::trunc);
    if (!theta_stream) throw std::runtime_error("cannot create patch theta cache");
    if (!geometric_theta_stream) {
        throw std::runtime_error("cannot create geometric patch theta cache");
    }
    std::uint64_t theta_offset = 0;
    for (const auto& patch : impl_->patches) {
        const auto& theta = patch.topology.theta_values();
        manifest["patches"].push_back({
            {"id", patch.surface->id},
            {"path", canonical_text(patch.path)},
            {"theta_offset", theta_offset},
            {"theta_count", theta.size()},
            {"fingerprint", patch.fingerprint},
            {"valid", patch.valid},
        });
        theta_stream.write(
            reinterpret_cast<const char*>(theta.data()),
            static_cast<std::streamsize>(theta.size() * sizeof(float)));
        const auto& geometric_theta = patch.geometric_topology.theta_values();
        geometric_theta_stream.write(
            reinterpret_cast<const char*>(geometric_theta.data()),
            static_cast<std::streamsize>(geometric_theta.size() * sizeof(float)));
        theta_offset += theta.size();
    }
    theta_stream.close();
    geometric_theta_stream.close();
    if (!theta_stream) throw std::runtime_error("failed writing patch theta cache");
    if (!geometric_theta_stream) {
        throw std::runtime_error("failed writing geometric patch theta cache");
    }
    std::filesystem::rename(
        cache_directory / "patch_theta.f32.tmp",
        cache_directory / "patch_theta.f32");
    std::filesystem::rename(
        cache_directory / "patch_geometric_theta.f32.tmp",
        cache_directory / "patch_geometric_theta.f32");
    for (const auto& source : impl_->sources) {
        json paths = json::array();
        for (const auto& path : source.paths) paths.push_back(canonical_text(path));
        manifest["sources"].push_back({
            {"kind", static_cast<int>(source.kind)},
            {"paths", std::move(paths)},
            {"role", static_cast<int>(source.role)},
            {"coordinate_scale", source.coordinate_scale},
            {"invalid_items", source.invalid_items},
            {"identity", source_identity(source)},
            {"fingerprint", source_key(source)},
        });
    }
    const auto temporary = cache_directory / "manifest.json.tmp";
    {
        std::ofstream stream(temporary);
        stream << manifest.dump(2) << '\n';
        if (!stream) throw std::runtime_error("failed writing graph manifest");
    }
    std::filesystem::rename(temporary, cache_directory / "manifest.json");
}

InputGraph InputGraph::open(
    const std::filesystem::path& cache_directory,
    GraphOptions options,
    const ThetaProvider& geometric_provider)
{
    std::ifstream stream(cache_directory / "manifest.json");
    if (!stream) throw std::runtime_error("cannot open graph manifest");
    json manifest;
    stream >> manifest;
    const int version = manifest.value("version", 0);
    if (manifest.value("schema", "") != "spiral-winding-graph"
        || (version != 1 && version != 2)) {
        throw std::runtime_error("unsupported graph manifest");
    }
    const json& cached_options = manifest.value("options", json::object());
    options.contact_tolerance = cached_options.value(
        "contact_tolerance", options.contact_tolerance);
    options.fiber_coordinate_scale = cached_options.value(
        "fiber_coordinate_scale", options.fiber_coordinate_scale);
    options.surface_sampling_stride = cached_options.value(
        "surface_sampling_stride", options.surface_sampling_stride);
    if (cached_options.contains("z_min") && !cached_options["z_min"].is_null()) {
        options.z_min = cached_options["z_min"].get<float>();
    }
    if (cached_options.contains("z_max") && !cached_options["z_max"].is_null()) {
        options.z_max = cached_options["z_max"].get<float>();
    }
    InputGraph result(options);
    result.impl_->theta_provider_key
        = manifest.value("theta_provider_key", std::string{});
    result.impl_->graph = WindingGraph::open(cache_directory / "graph.bin");
    std::ifstream theta_stream(cache_directory / "patch_theta.f32", std::ios::binary);
    if (!theta_stream) throw std::runtime_error("cannot open patch theta cache");
    std::ifstream geometric_theta_stream;
    if (version >= 2) {
        geometric_theta_stream.open(
            cache_directory / "patch_geometric_theta.f32", std::ios::binary);
        if (!geometric_theta_stream) {
            throw std::runtime_error("cannot open geometric patch theta cache");
        }
    }
    for (const auto& entry : manifest.at("patches")) {
        const std::filesystem::path patch_path = entry.at("path").get<std::string>();
        if (patch_fingerprint(patch_path) != entry.at("fingerprint").get<std::string>()) {
            throw std::runtime_error(
                "cached patch content changed; rebuild the graph: "
                + patch_path.string());
        }
        auto loaded = result.impl_->load_patch(patch_path);
        if (!loaded) {
            throw std::runtime_error(
                "cached patch no longer has a valid quad: " + patch_path.string());
        }
        auto patch = std::move(*loaded);
        patch.valid = entry.value("valid", true);
        if (patch.surface->id != entry.at("id").get<std::string>()) {
            throw std::runtime_error("cached patch identity changed: " + patch.path.string());
        }
        const std::size_t count = entry.at("theta_count").get<std::size_t>();
        if (count != patch.topology.node_count()) {
            throw std::runtime_error("cached patch topology changed: " + patch.path.string());
        }
        const std::uint64_t offset = entry.at("theta_offset").get<std::uint64_t>();
        theta_stream.seekg(static_cast<std::streamoff>(offset * sizeof(float)));
        std::vector<float> theta(count);
        theta_stream.read(
            reinterpret_cast<char*>(theta.data()),
            static_cast<std::streamsize>(theta.size() * sizeof(float)));
        if (!theta_stream) throw std::runtime_error("truncated patch theta cache");
        if (const auto conflict = patch.topology.assign_theta(theta)) {
            throw std::runtime_error(
                "cached patch theta is no longer liftable (residual "
                + std::to_string(conflict->residual) + ")");
        }
        if (version >= 2) {
            geometric_theta_stream.seekg(
                static_cast<std::streamoff>(offset * sizeof(float)));
            std::vector<float> geometric_theta(count);
            geometric_theta_stream.read(
                reinterpret_cast<char*>(geometric_theta.data()),
                static_cast<std::streamsize>(
                    geometric_theta.size() * sizeof(float)));
            if (!geometric_theta_stream) {
                throw std::runtime_error("truncated geometric patch theta cache");
            }
            if (const auto conflict
                = patch.geometric_topology.assign_theta(geometric_theta)) {
                throw std::runtime_error(
                    "cached geometric patch theta is no longer liftable (residual "
                    + std::to_string(conflict->residual) + ")");
            }
            patch.geometric_ready = true;
        }
        result.impl_->patches.push_back(std::move(patch));
    }
    if (version == 1) {
        result.impl_->assign_geometric_patch_theta(
            result.impl_->patches, geometric_provider);
    }
    result.impl_->rebuild_surface_index();
    for (const auto& entry : manifest.at("sources")) {
        SourceSpec source;
        source.kind = static_cast<SourceSpec::Kind>(entry.at("kind").get<int>());
        source.role = static_cast<InputRole>(entry.at("role").get<int>());
        source.coordinate_scale = entry.value("coordinate_scale", 1.0f);
        source.invalid_items = entry.value(
            "invalid_items", std::vector<std::string>{});
        for (const auto& path : entry.at("paths")) {
            source.paths.emplace_back(path.get<std::string>());
        }
        const std::string identity = source_identity(source);
        const std::string fingerprint = source_key(source);
        if (identity != entry.at("identity").get<std::string>()
            || fingerprint != entry.at("fingerprint").get<std::string>()) {
            throw std::runtime_error(
                "cached source content changed; rebuild the graph: " + identity);
        }
        result.impl_->source_keys.insert(fingerprint);
        result.impl_->source_identities.insert(identity);
        result.impl_->sources.push_back(std::move(source));
    }
    return result;
}

} // namespace spiral::winding
