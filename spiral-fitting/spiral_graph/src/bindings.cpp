#include <spiral_graph/input_graph.hpp>
#include <spiral_graph/fiber_layout.hpp>
#include <spiral_graph/registration.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
namespace winding = spiral::winding;
namespace layout = spiral::layout;
namespace registration = spiral::registration;

namespace {

using ThetaArray = nb::ndarray<
    nb::numpy, const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> own_1d(std::vector<T>&& values)
{
    auto* held = new std::vector<T>(std::move(values));
    nb::capsule owner(held, [](void* pointer) noexcept {
        delete static_cast<std::vector<T>*>(pointer);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(
        held->data(), {held->size()}, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> own_2d(
    std::vector<T>&& values, std::size_t rows, std::size_t columns)
{
    if (rows != 0 && columns > values.size() / rows) {
        throw std::invalid_argument("owned array shape exceeds storage");
    }
    if (rows * columns != values.size()) {
        throw std::invalid_argument("owned array shape does not match storage");
    }
    auto* held = new std::vector<T>(std::move(values));
    nb::capsule owner(held, [](void* pointer) noexcept {
        delete static_cast<std::vector<T>*>(pointer);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(
        held->data(), {rows, columns}, owner);
}

winding::ThetaProvider theta_provider(nb::object provider)
{
    if (provider.is_none()) {
        throw std::invalid_argument("checkpoint theta provider is required");
    }
    return [provider = std::move(provider)](std::span<const winding::Zyx> points) {
        nb::ndarray<nb::numpy, const float, nb::shape<-1, 3>, nb::c_contig> input(
            reinterpret_cast<const float*>(points.data()),
            {points.size(), std::size_t{3}});
        ThetaArray values = nb::cast<ThetaArray>(provider(input));
        if (values.shape(0) != points.size()) {
            throw std::runtime_error("theta provider returned the wrong shape");
        }
        return std::vector<float>(values.data(), values.data() + values.shape(0));
    };
}

class PyWindingGraph {
public:
    using ZyxBatch = nb::ndarray<
        nb::numpy, const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
    using LayoutRaster = nb::ndarray<
        nb::numpy, float, nb::shape<-1, -1, 3>, nb::c_contig, nb::device::cpu>;
    using LayoutLabels = nb::ndarray<
        nb::numpy, std::int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
    using LayoutPixelIndices = nb::ndarray<
        nb::numpy, const std::uint64_t, nb::ndim<1>, nb::c_contig,
        nb::device::cpu>;
    using LayoutContributions = nb::ndarray<
        nb::numpy, const float, nb::shape<-1, 3>, nb::c_contig,
        nb::device::cpu>;
    using LayoutWeights = nb::ndarray<
        nb::numpy, const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using LayoutPatchIndices = nb::ndarray<
        nb::numpy, const std::int32_t, nb::ndim<1>, nb::c_contig,
        nb::device::cpu>;

    PyWindingGraph(
        winding::GraphOptions options,
        nb::object theta_provider,
        std::optional<std::filesystem::path> cache_directory = std::nullopt)
        : input_(options_for_provider(std::move(options), theta_provider)),
          theta_provider_(std::move(theta_provider)),
          cache_directory_(std::move(cache_directory))
    {
        apply_provider_key();
    }

    static PyWindingGraph* create(
        const std::filesystem::path& cache_directory,
        nb::object theta_provider,
        winding::GraphOptions options)
    {
        if (std::filesystem::exists(cache_directory / "manifest.json")) {
            throw std::runtime_error(
                "graph cache already exists; use WindingGraph.open()");
        }
        return new PyWindingGraph(
            std::move(options), std::move(theta_provider), cache_directory);
    }

    static PyWindingGraph* open(
        const std::filesystem::path& cache_directory,
        nb::object theta_provider,
        winding::GraphOptions options)
    {
        auto output = std::make_unique<PyWindingGraph>(
            options, std::move(theta_provider), cache_directory);
        winding::ThetaProvider geometric;
        if (!output->theta_provider_.is_none()) {
            geometric = output->geometric_provider();
        }
        output->input_ = winding::InputGraph::open(
            cache_directory, std::move(options), geometric);
        output->apply_provider_key();
        return output.release();
    }

    void set_theta_provider(nb::object provider)
    {
        theta_provider_ = std::move(provider);
        apply_provider_key();
    }

    winding::NodeId add_patch_id(const std::string& id)
    {
        return input_.graph().ensure_patch(id);
    }

    winding::AddResult add_patches(
        const std::vector<std::filesystem::path>& paths)
    {
        return input_.add_patches(paths, providers());
    }

    bool set_patch_valid(const std::string& patch_id, bool valid)
    {
        return input_.set_patch_valid(patch_id, valid);
    }

    bool patch_valid(const std::string& patch_id) const
    {
        return input_.patch_valid(patch_id);
    }

    std::vector<std::vector<winding::ContactHit>> inspect_contacts(
        ZyxBatch points, std::optional<float> tolerance) const
    {
        std::vector<winding::Zyx> zyx;
        zyx.reserve(points.shape(0));
        for (std::size_t index = 0; index < points.shape(0); ++index) {
            zyx.push_back({points(index, 0), points(index, 1), points(index, 2)});
        }
        return input_.inspect_contacts(zyx, tolerance);
    }

    nb::dict patch_layout(const std::string& patch_id) const
    {
        winding::PatchLayoutData layout = input_.patch_layout(patch_id);
        const std::size_t count = layout.reported_local_turn.size();
        nb::dict output;
        output["patch_id"] = layout.patch_id;
        output["source_path"] = layout.source_path;
        output["quad_shape"] = nb::make_tuple(
            layout.quad_rows, layout.quad_columns);
        output["quad_ij"] = own_2d(
            std::move(layout.quad_ij), count, std::size_t{2});
        output["zyx"] = own_2d(
            std::move(layout.zyx), count, std::size_t{3});
        output["reported_local_turn"] = own_1d(
            std::move(layout.reported_local_turn));
        output["geometric_local_turn"] = own_1d(
            std::move(layout.geometric_local_turn));
        const std::size_t vertex_count = layout.reported_vertex_turn.size();
        output["vertex_shape"] = nb::make_tuple(
            layout.vertex_rows, layout.vertex_columns);
        output["vertex_ij"] = own_2d(
            std::move(layout.vertex_ij), vertex_count, std::size_t{2});
        output["vertex_zyx"] = own_2d(
            std::move(layout.vertex_zyx), vertex_count, std::size_t{3});
        output["reported_vertex_turn"] = own_1d(
            std::move(layout.reported_vertex_turn));
        return output;
    }

    std::size_t rasterize_patch_layout(
        const std::string& patch_id,
        std::int64_t root_winding,
        double turn_min,
        double z_min,
        double turn_pixels,
        double z_spacing,
        std::int32_t patch_index,
        LayoutRaster raster,
        LayoutLabels labels) const
    {
        if (raster.shape(0) != labels.shape(0)
            || raster.shape(1) != labels.shape(1)) {
            throw std::invalid_argument("raster and label shapes do not match");
        }
        if (patch_index < 0) {
            throw std::invalid_argument("patch index must be nonnegative");
        }
        if (!(std::isfinite(turn_min) && std::isfinite(z_min)
              && std::isfinite(turn_pixels) && turn_pixels > 0
              && std::isfinite(z_spacing) && z_spacing > 0)) {
            throw std::invalid_argument("layout transform must be finite and positive");
        }

        winding::PatchLayoutData layout = input_.patch_layout(patch_id);
        const std::size_t count = layout.reported_local_turn.size();
        if (layout.quad_ij.size() != count * 2
            || layout.zyx.size() != count * 3
            || layout.vertex_ij.size() != layout.reported_vertex_turn.size() * 2
            || layout.vertex_zyx.size() != layout.reported_vertex_turn.size() * 3) {
            throw std::runtime_error("patch layout arrays have inconsistent sizes");
        }
        if (layout.quad_rows != 0
            && layout.quad_columns
                > std::numeric_limits<std::size_t>::max() / layout.quad_rows) {
            throw std::overflow_error("patch layout grid is too large");
        }
        const std::size_t grid_size = layout.quad_rows * layout.quad_columns;
        const std::size_t missing = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> grid(grid_size, missing);
        for (std::size_t index = 0; index < count; ++index) {
            const std::size_t row = layout.quad_ij[index * 2];
            const std::size_t column = layout.quad_ij[index * 2 + 1];
            if (row >= layout.quad_rows || column >= layout.quad_columns) {
                throw std::runtime_error("patch layout coordinate is out of bounds");
            }
            grid[row * layout.quad_columns + column] = index;
        }
        if (layout.vertex_rows != layout.quad_rows + 1
            || layout.vertex_columns != layout.quad_columns + 1
            || (layout.vertex_rows != 0
                && layout.vertex_columns
                    > std::numeric_limits<std::size_t>::max()
                        / layout.vertex_rows)) {
            throw std::runtime_error("patch vertex layout shape is inconsistent");
        }
        const std::size_t vertex_grid_size
            = layout.vertex_rows * layout.vertex_columns;
        std::vector<std::size_t> vertex_grid(vertex_grid_size, missing);
        for (std::size_t index = 0;
             index < layout.reported_vertex_turn.size(); ++index) {
            const std::size_t row = layout.vertex_ij[index * 2];
            const std::size_t column = layout.vertex_ij[index * 2 + 1];
            if (row >= layout.vertex_rows || column >= layout.vertex_columns) {
                throw std::runtime_error("patch vertex coordinate is out of bounds");
            }
            vertex_grid[row * layout.vertex_columns + column] = index;
        }

        struct Vertex {
            double x;
            double y;
            float zyx[3];
        };
        auto vertex = [&](std::size_t index) {
            return Vertex{
                (static_cast<double>(layout.reported_vertex_turn[index])
                 + static_cast<double>(root_winding) - turn_min)
                    * turn_pixels,
                (static_cast<double>(layout.vertex_zyx[index * 3]) - z_min)
                    / z_spacing,
                {layout.vertex_zyx[index * 3],
                 layout.vertex_zyx[index * 3 + 1],
                 layout.vertex_zyx[index * 3 + 2]},
            };
        };

        const std::int64_t height = static_cast<std::int64_t>(raster.shape(0));
        const std::int64_t width = static_cast<std::int64_t>(raster.shape(1));
        std::size_t written = 0;
        auto triangle = [&](const Vertex& a, const Vertex& b, const Vertex& c) {
            const double denominator =
                (b.y - c.y) * (a.x - c.x)
                + (c.x - b.x) * (a.y - c.y);
            if (!std::isfinite(denominator) || std::abs(denominator) < 1e-10) {
                return;
            }
            const std::int64_t x_begin = std::max<std::int64_t>(
                0, static_cast<std::int64_t>(std::ceil(
                       std::min({a.x, b.x, c.x}) - 1e-6)));
            const std::int64_t x_end = std::min<std::int64_t>(
                width - 1, static_cast<std::int64_t>(std::floor(
                               std::max({a.x, b.x, c.x}) + 1e-6)));
            const std::int64_t y_begin = std::max<std::int64_t>(
                0, static_cast<std::int64_t>(std::ceil(
                       std::min({a.y, b.y, c.y}) - 1e-6)));
            const std::int64_t y_end = std::min<std::int64_t>(
                height - 1, static_cast<std::int64_t>(std::floor(
                                std::max({a.y, b.y, c.y}) + 1e-6)));
            if (x_begin > x_end || y_begin > y_end) return;
            for (std::int64_t y = y_begin; y <= y_end; ++y) {
                for (std::int64_t x = x_begin; x <= x_end; ++x) {
                    if (labels(y, x) >= 0) continue;
                    const double w0 =
                        ((b.y - c.y) * (static_cast<double>(x) - c.x)
                         + (c.x - b.x) * (static_cast<double>(y) - c.y))
                        / denominator;
                    const double w1 =
                        ((c.y - a.y) * (static_cast<double>(x) - c.x)
                         + (a.x - c.x) * (static_cast<double>(y) - c.y))
                        / denominator;
                    const double w2 = 1.0 - w0 - w1;
                    constexpr double tolerance = -1e-6;
                    if (w0 < tolerance || w1 < tolerance || w2 < tolerance) {
                        continue;
                    }
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        raster(y, x, axis) = static_cast<float>(
                            w0 * a.zyx[axis] + w1 * b.zyx[axis]
                            + w2 * c.zyx[axis]);
                    }
                    labels(y, x) = patch_index;
                    ++written;
                }
            }
        };

        nb::gil_scoped_release release;
        for (std::size_t row = 0; row < layout.quad_rows; ++row) {
            for (std::size_t column = 0; column < layout.quad_columns;
                 ++column) {
                if (grid[row * layout.quad_columns + column] == missing) continue;
                const std::size_t top_left =
                    vertex_grid[row * layout.vertex_columns + column];
                const std::size_t top_right =
                    vertex_grid[row * layout.vertex_columns + column + 1];
                const std::size_t bottom_left =
                    vertex_grid[(row + 1) * layout.vertex_columns + column];
                const std::size_t bottom_right =
                    vertex_grid[(row + 1) * layout.vertex_columns + column + 1];
                if (top_left == missing || top_right == missing
                    || bottom_left == missing || bottom_right == missing) {
                    throw std::logic_error(
                        "valid source quad is missing a reconstructed vertex");
                }
                const Vertex tl = vertex(top_left);
                const Vertex tr = vertex(top_right);
                const Vertex bl = vertex(bottom_left);
                const Vertex br = vertex(bottom_right);
                triangle(bl, tl, tr);
                triangle(bl, tr, br);
            }
        }
        return written;
    }

    nb::dict fuse_layout_contributions(
        LayoutPixelIndices pixels,
        LayoutContributions contributions,
        LayoutWeights weights,
        LayoutPatchIndices patches,
        double agreement_distance,
        double continuity_distance,
        LayoutRaster raster,
        LayoutLabels labels) const
    {
        const std::size_t count = pixels.shape(0);
        if (contributions.shape(0) != count || weights.shape(0) != count
            || patches.shape(0) != count) {
            throw std::invalid_argument("layout contribution arrays differ in length");
        }
        if (raster.shape(0) != labels.shape(0)
            || raster.shape(1) != labels.shape(1)) {
            throw std::invalid_argument("raster and label shapes do not match");
        }
        if (!(agreement_distance > 0.0) || !std::isfinite(agreement_distance)) {
            throw std::invalid_argument(
                "layout agreement distance must be finite and positive");
        }
        if (!(continuity_distance > 0.0) || !std::isfinite(continuity_distance)) {
            throw std::invalid_argument(
                "layout continuity distance must be finite and positive");
        }
        const std::size_t height = raster.shape(0);
        const std::size_t width = raster.shape(1);
        if (height != 0 && width > std::numeric_limits<std::size_t>::max() / height) {
            throw std::overflow_error("layout raster is too large");
        }
        const std::size_t pixel_count = height * width;
        std::vector<std::size_t> order(count);
        for (std::size_t index = 0; index < count; ++index) {
            if (pixels(index) >= pixel_count) {
                throw std::out_of_range("layout contribution pixel is out of bounds");
            }
            if (patches(index) < 0 || !(weights(index) >= 0.0f)
                || !std::isfinite(weights(index))) {
                throw std::invalid_argument("invalid layout contribution metadata");
            }
            for (std::size_t axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(contributions(index, axis))) {
                    throw std::invalid_argument(
                        "layout contribution contains a non-finite coordinate");
                }
            }
            order[index] = index;
        }
        std::size_t occupied = 0;
        std::size_t blended = 0;
        std::size_t conflicting_pixels = 0;
        std::size_t conflicting_contributions = 0;
        std::size_t repaired_edges = 0;
        const double threshold_squared = agreement_distance * agreement_distance;
        const double continuity_squared = continuity_distance * continuity_distance;
        {
            nb::gil_scoped_release release;
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                if (pixels(a) != pixels(b)) return pixels(a) < pixels(b);
                if (patches(a) != patches(b)) return patches(a) < patches(b);
                return a < b;
            });
            std::vector<std::vector<std::size_t>> clusters;
            const std::size_t missing = std::numeric_limits<std::size_t>::max();
            std::vector<std::size_t> range_begin(pixel_count, missing);
            std::vector<std::size_t> range_end(pixel_count, missing);
            std::vector<float> selected_interior_weight(pixel_count, 0.0f);
            for (std::size_t begin = 0; begin < count;) {
                std::size_t end = begin + 1;
                while (end < count
                       && pixels(order[end]) == pixels(order[begin])) ++end;
                clusters.clear();
                for (std::size_t position = begin; position < end; ++position) {
                    const std::size_t candidate = order[position];
                    bool inserted = false;
                    for (auto& cluster : clusters) {
                        bool consistent = true;
                        for (const std::size_t other : cluster) {
                            double squared = 0.0;
                            for (std::size_t axis = 0; axis < 3; ++axis) {
                                const double delta = contributions(candidate, axis)
                                    - contributions(other, axis);
                                squared += delta * delta;
                            }
                            if (squared > threshold_squared) {
                                consistent = false;
                                break;
                            }
                        }
                        if (consistent) {
                            cluster.push_back(candidate);
                            inserted = true;
                            break;
                        }
                    }
                    if (!inserted) clusters.push_back({candidate});
                }
                std::size_t best = 0;
                double best_weight = -1.0;
                float best_interior_weight = -1.0f;
                for (std::size_t cluster = 0; cluster < clusters.size(); ++cluster) {
                    double total = 0.0;
                    float interior_weight = 0.0f;
                    for (const std::size_t contributor : clusters[cluster]) {
                        total += std::max(1e-12, static_cast<double>(weights(contributor)));
                        interior_weight = std::max(
                            interior_weight, weights(contributor));
                    }
                    if (interior_weight > best_interior_weight
                        || (interior_weight == best_interior_weight
                            && total > best_weight)) {
                        best = cluster;
                        best_weight = total;
                        best_interior_weight = interior_weight;
                    }
                }
                std::array<double, 3> sum{};
                std::size_t dominant = clusters[best].front();
                float dominant_weight = weights(dominant);
                for (const std::size_t contributor : clusters[best]) {
                    const double weight = std::max(
                        1e-12, static_cast<double>(weights(contributor)));
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        sum[axis] += weight * contributions(contributor, axis);
                    }
                    if (weights(contributor) > dominant_weight) {
                        dominant = contributor;
                        dominant_weight = weights(contributor);
                    }
                }
                const std::size_t pixel = pixels(order[begin]);
                range_begin[pixel] = begin;
                range_end[pixel] = end;
                const std::size_t row = pixel / width;
                const std::size_t column = pixel % width;
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    raster(row, column, axis)
                        = static_cast<float>(sum[axis] / best_weight);
                }
                labels(row, column) = patches(dominant);
                selected_interior_weight[pixel] = best_interior_weight;
                ++occupied;
                blended += clusters[best].size() - 1;
                if (clusters.size() > 1) {
                    ++conflicting_pixels;
                    conflicting_contributions += end - begin - clusters[best].size();
                }
                begin = end;
            }
            const auto output_distance_squared = [&](std::size_t a, std::size_t b) {
                const std::size_t a_row = a / width;
                const std::size_t a_column = a % width;
                const std::size_t b_row = b / width;
                const std::size_t b_column = b % width;
                double squared = 0.0;
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    const double delta = raster(a_row, a_column, axis)
                        - raster(b_row, b_column, axis);
                    squared += delta * delta;
                }
                return squared;
            };
            const auto repair_edge = [&](std::size_t a, std::size_t b) {
                if (range_begin[a] == missing || range_begin[b] == missing
                    || output_distance_squared(a, b) <= continuity_squared) {
                    return false;
                }
                std::size_t left = range_begin[a];
                std::size_t right = range_begin[b];
                std::size_t best_a = missing;
                std::size_t best_b = missing;
                bool change_a = false;
                float best_fixed_support = -1.0f;
                while (left < range_end[a] && right < range_end[b]) {
                    const std::size_t contributor_a = order[left];
                    const std::size_t contributor_b = order[right];
                    if (patches(contributor_a) < patches(contributor_b)) {
                        ++left;
                        continue;
                    }
                    if (patches(contributor_b) < patches(contributor_a)) {
                        ++right;
                        continue;
                    }
                    double squared = 0.0;
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        const double delta = contributions(contributor_a, axis)
                            - contributions(contributor_b, axis);
                        squared += delta * delta;
                    }
                    if (squared <= continuity_squared) {
                        const std::size_t a_row = a / width;
                        const std::size_t a_column = a % width;
                        const std::size_t b_row = b / width;
                        const std::size_t b_column = b % width;
                        double agrees_a = 0.0;
                        double agrees_b = 0.0;
                        for (std::size_t axis = 0; axis < 3; ++axis) {
                            const double delta_a = contributions(contributor_a, axis)
                                - raster(a_row, a_column, axis);
                            const double delta_b = contributions(contributor_b, axis)
                                - raster(b_row, b_column, axis);
                            agrees_a += delta_a * delta_a;
                            agrees_b += delta_b * delta_b;
                        }
                        // Extend an already selected sheet through a source
                        // patch that agrees at the fixed endpoint. Never
                        // replace both endpoints with an unrelated third sheet.
                        if (agrees_a <= threshold_squared
                            && selected_interior_weight[a] > best_fixed_support) {
                            best_a = contributor_a;
                            best_b = contributor_b;
                            change_a = false;
                            best_fixed_support = selected_interior_weight[a];
                        }
                        if (agrees_b <= threshold_squared
                            && selected_interior_weight[b] > best_fixed_support) {
                            best_a = contributor_a;
                            best_b = contributor_b;
                            change_a = true;
                            best_fixed_support = selected_interior_weight[b];
                        }
                    }
                    ++left;
                    ++right;
                }
                if (best_a == missing) return false;
                const std::size_t a_row = a / width;
                const std::size_t a_column = a % width;
                const std::size_t b_row = b / width;
                const std::size_t b_column = b % width;
                if (change_a) {
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        raster(a_row, a_column, axis) = contributions(best_a, axis);
                    }
                    labels(a_row, a_column) = patches(best_a);
                    selected_interior_weight[a] = weights(best_a);
                } else {
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        raster(b_row, b_column, axis) = contributions(best_b, axis);
                    }
                    labels(b_row, b_column) = patches(best_b);
                    selected_interior_weight[b] = weights(best_b);
                }
                return true;
            };
            // Enforce the topology already present in the source patches.
            // Alternating scan directions avoids giving either canvas axis a
            // permanent priority when several supported edges share a vertex.
            for (std::size_t pass = 0; pass < 1; ++pass) {
                std::size_t changed = 0;
                if (pass % 2 == 0) {
                    for (std::size_t row = 0; row < height; ++row) {
                        for (std::size_t column = 0; column + 1 < width; ++column) {
                            changed += repair_edge(
                                row * width + column, row * width + column + 1);
                        }
                    }
                    for (std::size_t row = 0; row + 1 < height; ++row) {
                        for (std::size_t column = 0; column < width; ++column) {
                            changed += repair_edge(
                                row * width + column, (row + 1) * width + column);
                        }
                    }
                } else {
                    for (std::size_t row = height; row-- > 1;) {
                        for (std::size_t column = width; column-- > 0;) {
                            changed += repair_edge(
                                row * width + column,
                                (row - 1) * width + column);
                        }
                    }
                    for (std::size_t row = height; row-- > 0;) {
                        for (std::size_t column = width; column-- > 1;) {
                            changed += repair_edge(
                                row * width + column,
                                row * width + column - 1);
                        }
                    }
                }
                repaired_edges += changed;
                if (changed == 0) break;
            }
        }
        nb::dict result;
        result["contributions"] = count;
        result["occupied_pixels"] = occupied;
        result["blended_contributions"] = blended;
        result["conflicting_pixels"] = conflicting_pixels;
        result["discarded_conflicting_contributions"]
            = conflicting_contributions;
        result["source_topology_edge_repairs"] = repaired_edges;
        return result;
    }

    winding::AddResult add_point_collections(
        const std::vector<std::filesystem::path>& paths,
        winding::InputRole role)
    {
        return input_.add_point_collections(paths, role, providers());
    }

    std::vector<winding::Constraint> inspect_point_collections(
        const std::vector<std::filesystem::path>& paths,
        winding::InputRole role)
    {
        return input_.inspect_point_collections(paths, role, providers());
    }

    winding::AddResult add_fibers(
        const std::filesystem::path& directory,
        std::optional<float> coordinate_scale,
        std::vector<std::string> invalid_fibers)
    {
        const auto& options = input_.options();
        return input_.add_fibers(
            directory,
            coordinate_scale.value_or(options.fiber_coordinate_scale),
            std::move(invalid_fibers));
    }

    winding::AddResult add_tracks(
        const std::filesystem::path& tracks,
        std::optional<std::filesystem::path> crossings,
        std::optional<std::filesystem::path> spatial_index)
    {
        if (!crossings) {
            std::string value = tracks.string();
            constexpr std::string_view suffix = ".vctracks";
            if (value.ends_with(suffix)) value.resize(value.size() - suffix.size());
            crossings = value + ".crossings.npz";
        }
        if (!spatial_index) {
            std::string value = tracks.string();
            constexpr std::string_view suffix = ".vctracks";
            if (value.ends_with(suffix)) value.resize(value.size() - suffix.size());
            spatial_index = value + ".winding-index";
        }
        if (!std::filesystem::is_regular_file(*spatial_index / "header.bin")) {
            input_.prepare_track_index(tracks, *spatial_index);
        }
        return input_.add_tracks(
            tracks, *crossings, *spatial_index, providers());
    }

    winding::AddResult add_constraint(
        const std::string& from_patch,
        const std::string& to_patch,
        std::int64_t delta,
        std::int64_t geometric_delta,
        winding::Provenance provenance)
    {
        return input_.graph().add_constraint(
            from_patch, to_patch, delta, geometric_delta, std::move(provenance));
    }

    winding::AddResult add_anchor(
        const std::string& patch,
        std::int64_t winding,
        std::int64_t geometric_winding,
        winding::Provenance provenance)
    {
        return input_.graph().add_anchor(
            patch, winding, geometric_winding, std::move(provenance));
    }

    nb::object lifted_relative_winding(
        const std::string& from_patch,
        const std::string& to_patch) const
    {
        const auto value = input_.graph().lifted_relative_winding(
            from_patch, to_patch);
        return value ? nb::cast(*value) : nb::none();
    }

    winding::Holonomy holonomy(std::size_t index) const
    {
        return input_.graph().holonomy(index);
    }

    std::vector<winding::HolonomyAudit> holonomy_audits() const
    {
        return input_.graph().holonomy_audits();
    }

    winding::GraphStats stats() const { return input_.graph().stats(); }

    bool has_patch(const std::string& patch) const
    {
        return input_.graph().has_patch(patch);
    }

    winding::NodeId patch_node(const std::string& patch) const
    {
        return input_.graph().patch_node(patch);
    }

    std::string node_name(winding::NodeId node) const
    {
        return input_.graph().node_name(node);
    }

    std::vector<winding::Constraint> constraints() const
    {
        return input_.graph().constraints();
    }

    winding::TrackInfo inspect_tracks(
        const std::filesystem::path& tracks,
        std::optional<std::filesystem::path> crossings) const
    {
        return input_.inspect_tracks(tracks, crossings.value_or(std::filesystem::path{}));
    }

    winding::TrackIndexInfo prepare_track_index(
        const std::filesystem::path& tracks,
        std::optional<std::filesystem::path> output,
        std::uint32_t cell_size,
        std::size_t memory_budget_bytes) const
    {
        if (!output) {
            std::string value = tracks.string();
            constexpr std::string_view suffix = ".vctracks";
            if (value.ends_with(suffix)) value.resize(value.size() - suffix.size());
            output = value + ".winding-index";
        }
        return input_.prepare_track_index(
            tracks, *output, cell_size, memory_budget_bytes);
    }

    void save(std::optional<std::filesystem::path> directory = std::nullopt)
    {
        if (directory) cache_directory_ = *directory;
        if (!cache_directory_) {
            throw std::runtime_error("save requires a cache directory");
        }
        input_.save(*cache_directory_);
    }

private:
    static winding::GraphOptions options_for_provider(
        winding::GraphOptions options,
        const nb::object& provider)
    {
        if (provider.is_none()) return options;
        if (nb::hasattr(provider, "z_begin")
            && nb::hasattr(provider, "z_end")) {
            if (!std::isfinite(options.z_min)) {
                options.z_min = nb::cast<float>(provider.attr("z_begin"));
            }
            if (!std::isfinite(options.z_max)) {
                options.z_max = nb::cast<float>(provider.attr("z_end"));
            }
        }
        return options;
    }

    void apply_provider_key()
    {
        if (theta_provider_.is_none()) return;
        std::string key;
        if (nb::hasattr(theta_provider_, "cache_key")) {
            key = nb::cast<std::string>(theta_provider_.attr("cache_key"));
        }
        input_.set_theta_provider_key(std::move(key));
    }

    winding::ThetaProvider reported_provider()
    {
        if (theta_provider_.is_none()) {
            throw std::runtime_error(
                "this operation requires a theta provider; call set_theta_provider()");
        }
        return [this](std::span<const winding::Zyx> points) {
            // Zyx is exactly three tightly packed floats. The view is borrowed
            // only for the duration of the synchronous provider call.
            nb::ndarray<nb::numpy, const float, nb::shape<-1, 3>, nb::c_contig> input(
                reinterpret_cast<const float*>(points.data()),
                {points.size(), std::size_t{3}});
            nb::object returned = theta_provider_(input);
            ThetaArray values = nb::cast<ThetaArray>(returned);
            if (values.shape(0) != points.size()) {
                throw std::runtime_error("theta provider returned the wrong shape");
            }
            return std::vector<float>(values.data(), values.data() + values.shape(0));
        };
    }

    winding::ThetaProvider geometric_provider()
    {
        if (theta_provider_.is_none()) {
            throw std::runtime_error(
                "this operation requires a theta provider; call set_theta_provider()");
        }
        if (!nb::hasattr(theta_provider_, "geometric_theta")) {
            throw std::runtime_error(
                "theta provider must define geometric_theta(zyx)");
        }
        return [this](std::span<const winding::Zyx> points) {
            nb::ndarray<nb::numpy, const float, nb::shape<-1, 3>, nb::c_contig> input(
                reinterpret_cast<const float*>(points.data()),
                {points.size(), std::size_t{3}});
            nb::object returned = theta_provider_.attr("geometric_theta")(input);
            ThetaArray values = nb::cast<ThetaArray>(returned);
            if (values.shape(0) != points.size()) {
                throw std::runtime_error(
                    "geometric theta provider returned the wrong shape");
            }
            return std::vector<float>(values.data(), values.data() + values.shape(0));
        };
    }

    winding::ThetaProviders providers()
    {
        return {reported_provider(), geometric_provider()};
    }

    winding::InputGraph input_;
    nb::object theta_provider_;
    std::optional<std::filesystem::path> cache_directory_;
};

} // namespace

NB_MODULE(_spiral_graph, module)
{
    module.doc() = "High-throughput incremental winding-constraint graph.";

    nb::enum_<winding::ConflictKind>(module, "ConflictKind")
        .value("ABSOLUTE_ANCHOR", winding::ConflictKind::absolute_anchor)
        .value("PATCH_THETA", winding::ConflictKind::patch_theta)
        .value("SOURCE_THETA", winding::ConflictKind::source_theta);

    nb::enum_<winding::InputRole>(module, "InputRole")
        .value("ABSOLUTE", winding::InputRole::absolute)
        .value("RELATIVE", winding::InputRole::relative)
        .value("SAME_WINDING", winding::InputRole::same_winding);

    nb::class_<winding::Provenance>(module, "Provenance")
        .def(nb::init<>())
        .def_rw("source_type", &winding::Provenance::source_type)
        .def_rw("source", &winding::Provenance::source)
        .def_rw("item", &winding::Provenance::item)
        .def_rw("detail", &winding::Provenance::detail);

    nb::class_<winding::Constraint>(module, "Constraint")
        .def_ro("from_node", &winding::Constraint::from)
        .def_ro("to_node", &winding::Constraint::to)
        .def_ro("delta", &winding::Constraint::delta)
        .def_ro("geometric_delta", &winding::Constraint::geometric_delta)
        .def_ro("provenance", &winding::Constraint::provenance)
        .def_ro("absolute", &winding::Constraint::absolute);

    nb::class_<winding::CycleEdge>(module, "CycleEdge")
        .def_ro("from_node", &winding::CycleEdge::from)
        .def_ro("to_node", &winding::CycleEdge::to)
        .def_ro("delta", &winding::CycleEdge::delta)
        .def_ro("geometric_delta", &winding::CycleEdge::geometric_delta)
        .def_ro("constraint_index", &winding::CycleEdge::constraint_index)
        .def_ro("provenance", &winding::CycleEdge::provenance)
        .def_ro("closing", &winding::CycleEdge::closing);

    nb::class_<winding::Conflict>(module, "Conflict")
        .def_ro("kind", &winding::Conflict::kind)
        .def_ro("residual", &winding::Conflict::residual)
        .def_ro("closing_constraint", &winding::Conflict::closing_constraint)
        .def_ro("cycle", &winding::Conflict::cycle);

    nb::class_<winding::Holonomy>(module, "Holonomy")
        .def_ro("reported_holonomy", &winding::Holonomy::reported_holonomy)
        .def_ro("geometric_holonomy", &winding::Holonomy::geometric_holonomy)
        .def_ro("inconsistency", &winding::Holonomy::inconsistency)
        .def_ro("closing_constraint", &winding::Holonomy::closing_constraint)
        .def_ro("cycle", &winding::Holonomy::cycle);

    nb::class_<winding::HolonomyAudit>(module, "HolonomyAudit")
        .def_ro("reported_holonomy", &winding::HolonomyAudit::reported_holonomy)
        .def_ro("geometric_holonomy", &winding::HolonomyAudit::geometric_holonomy)
        .def_ro("inconsistency", &winding::HolonomyAudit::inconsistency)
        .def_ro("constraint_index", &winding::HolonomyAudit::constraint_index);

    nb::class_<winding::AddResult>(module, "AddResult")
        .def_ro("committed", &winding::AddResult::committed)
        .def_ro("already_present", &winding::AddResult::already_present)
        .def_ro("nodes_added", &winding::AddResult::nodes_added)
        .def_ro("constraints_added", &winding::AddResult::constraints_added)
        .def_ro("anchors_added", &winding::AddResult::anchors_added)
        .def_ro("holonomies_added", &winding::AddResult::holonomies_added)
        .def_ro("conflict", &winding::AddResult::conflict);

    nb::class_<winding::LiftedWinding>(module, "LiftedWinding")
        .def_ro("representative", &winding::LiftedWinding::representative)
        .def_ro("period", &winding::LiftedWinding::period);

    nb::class_<winding::GraphStats>(module, "GraphStats")
        .def_ro("patch_count", &winding::GraphStats::patch_count)
        .def_ro("constraint_count", &winding::GraphStats::constraint_count)
        .def_ro("component_count", &winding::GraphStats::component_count)
        .def_ro("anchored_component_count", &winding::GraphStats::anchored_component_count)
        .def_ro("holonomy_count", &winding::GraphStats::holonomy_count);

    nb::class_<winding::TrackInfo>(module, "TrackInfo")
        .def_ro("tracks", &winding::TrackInfo::tracks)
        .def_ro("points", &winding::TrackInfo::points)
        .def_ro("crossings", &winding::TrackInfo::crossings);

    nb::class_<winding::TrackIndexInfo>(module, "TrackIndexInfo")
        .def_ro("points", &winding::TrackIndexInfo::points)
        .def_ro("cells", &winding::TrackIndexInfo::cells)
        .def_ro("cell_size", &winding::TrackIndexInfo::cell_size)
        .def_ro("already_present", &winding::TrackIndexInfo::already_present);

    nb::class_<winding::ContactHit>(module, "ContactHit")
        .def_ro("patch_id", &winding::ContactHit::patch_id)
        .def_ro("distance", &winding::ContactHit::distance)
        .def_ro("row", &winding::ContactHit::row)
        .def_ro("column", &winding::ContactHit::column);

    nb::class_<winding::GraphOptions>(module, "GraphOptions")
        .def(nb::init<>())
        .def_rw("contact_tolerance", &winding::GraphOptions::contact_tolerance)
        .def_rw("theta_batch_size", &winding::GraphOptions::theta_batch_size)
        .def_rw("workers", &winding::GraphOptions::workers)
        .def_rw("fiber_coordinate_scale", &winding::GraphOptions::fiber_coordinate_scale)
        .def_rw("surface_sampling_stride", &winding::GraphOptions::surface_sampling_stride)
        .def_rw("z_min", &winding::GraphOptions::z_min)
        .def_rw("z_max", &winding::GraphOptions::z_max);

    nb::class_<layout::LayoutOptions>(module, "LayoutOptions")
        .def(nb::init<>())
        .def_rw("contact_tolerance", &layout::LayoutOptions::contact_tolerance)
        .def_rw("min_inliers", &layout::LayoutOptions::min_inliers)
        .def_rw("uv_ransac_tolerance", &layout::LayoutOptions::uv_ransac_tolerance)
        .def_rw("max_refit_rms", &layout::LayoutOptions::max_refit_rms)
        .def_rw("ransac_hypotheses", &layout::LayoutOptions::ransac_hypotheses)
        .def_rw("max_raster_samples", &layout::LayoutOptions::max_raster_samples)
        .def_rw("theta_batch_size", &layout::LayoutOptions::theta_batch_size)
        .def_rw("workers", &layout::LayoutOptions::workers);

    nb::class_<layout::FiberPointLayout>(module, "FiberPointLayout")
        .def_ro("z", &layout::FiberPointLayout::z)
        .def_ro("y", &layout::FiberPointLayout::y)
        .def_ro("x", &layout::FiberPointLayout::x)
        .def_ro("u", &layout::FiberPointLayout::u)
        .def_ro("v", &layout::FiberPointLayout::v)
        .def_ro("winding", &layout::FiberPointLayout::winding)
        .def_ro("fractional_winding", &layout::FiberPointLayout::fractional_winding)
        .def_ro("theta_valid", &layout::FiberPointLayout::theta_valid);

    nb::class_<layout::FiberLayout>(module, "FiberLayout")
        .def_ro("id", &layout::FiberLayout::id)
        .def_prop_ro("axis", [](const layout::FiberLayout& value) {
            return std::string(1, value.axis);
        })
        .def_ro("logical_track", &layout::FiberLayout::logical_track)
        .def_ro("reversed", &layout::FiberLayout::reversed)
        .def_ro("arclength", &layout::FiberLayout::arclength)
        .def_ro("winding_offset", &layout::FiberLayout::winding_offset)
        .def_ro("points", &layout::FiberLayout::points);

    nb::class_<layout::CrossingKnot>(module, "CrossingKnot")
        .def_ro("first_fiber", &layout::CrossingKnot::first_fiber)
        .def_ro("first_point", &layout::CrossingKnot::first_point)
        .def_ro("second_fiber", &layout::CrossingKnot::second_fiber)
        .def_ro("second_point", &layout::CrossingKnot::second_point)
        .def_ro("u_residual", &layout::CrossingKnot::u_residual)
        .def_ro("v_residual", &layout::CrossingKnot::v_residual);

    nb::class_<layout::LayoutResult>(module, "LayoutResult")
        .def_ro("fibers", &layout::LayoutResult::fibers)
        .def_ro("crossings", &layout::LayoutResult::crossings)
        .def_ro("excluded_fibers", &layout::LayoutResult::excluded_fibers)
        .def_ro("root_fiber", &layout::LayoutResult::root_fiber)
        .def_ro("total_arclength", &layout::LayoutResult::total_arclength)
        .def_ro("initial_cost", &layout::LayoutResult::initial_cost)
        .def_ro("final_cost", &layout::LayoutResult::final_cost)
        .def_ro("solver_iterations", &layout::LayoutResult::solver_iterations)
        .def_ro("theta_covered_points", &layout::LayoutResult::theta_covered_points)
        .def_ro("theta_uncovered_points", &layout::LayoutResult::theta_uncovered_points);

    nb::class_<registration::Result>(module, "RegistrationResult")
        .def_ro("accepted", &registration::Result::accepted)
        .def_ro("reflected", &registration::Result::reflected)
        .def_ro("r00", &registration::Result::r00)
        .def_ro("r01", &registration::Result::r01)
        .def_ro("r10", &registration::Result::r10)
        .def_ro("r11", &registration::Result::r11)
        .def_ro("translation_u", &registration::Result::translation_u)
        .def_ro("translation_v", &registration::Result::translation_v)
        .def_ro("rms", &registration::Result::rms)
        .def_ro("inliers", &registration::Result::inliers)
        .def_ro("rejection", &registration::Result::rejection);

    nb::class_<registration::Pose2d>(module, "Pose2d")
        .def_ro("r00", &registration::Pose2d::r00)
        .def_ro("r01", &registration::Pose2d::r01)
        .def_ro("r10", &registration::Pose2d::r10)
        .def_ro("r11", &registration::Pose2d::r11)
        .def_ro("translation_u", &registration::Pose2d::translation_u)
        .def_ro("translation_v", &registration::Pose2d::translation_v);

    nb::class_<registration::PoseGraphResult>(module, "PoseGraphResult")
        .def_ro("usable", &registration::PoseGraphResult::usable)
        .def_ro("poses", &registration::PoseGraphResult::poses)
        .def_ro("initial_cost", &registration::PoseGraphResult::initial_cost)
        .def_ro("final_cost", &registration::PoseGraphResult::final_cost)
        .def_ro("iterations", &registration::PoseGraphResult::iterations);

    module.def(
        "fit_rigid_registration",
        [](nb::ndarray<nb::numpy, const double, nb::shape<-1, 2>,
                      nb::c_contig, nb::device::cpu> source,
           nb::ndarray<nb::numpy, const double, nb::shape<-1, 2>,
                      nb::c_contig, nb::device::cpu> target,
           const layout::LayoutOptions& options) {
            if (source.shape(0) != target.shape(0)) {
                throw std::invalid_argument(
                    "registration source and target counts differ");
            }
            std::vector<registration::Correspondence2d> values;
            values.reserve(source.shape(0));
            for (std::size_t index = 0; index < source.shape(0); ++index) {
                values.push_back({
                    source(index, 0), source(index, 1),
                    target(index, 0), target(index, 1),
                });
            }
            registration::Options fit_options;
            fit_options.min_inliers = options.min_inliers;
            fit_options.inlier_tolerance = options.uv_ransac_tolerance;
            fit_options.max_refit_rms = options.max_refit_rms;
            fit_options.max_hypotheses = options.ransac_hypotheses;
            return registration::fit_rigid_2d(values, fit_options);
        },
        nb::arg("source_uv"), nb::arg("target_uv"),
        nb::arg("options") = layout::LayoutOptions{});

    module.def(
        "refine_patch_pose_graph",
        [](nb::ndarray<nb::numpy, const double, nb::shape<-1, 2, 3>,
                      nb::c_contig, nb::device::cpu> poses,
           nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<1>,
                      nb::c_contig, nb::device::cpu> absolute_patch,
           nb::ndarray<nb::numpy, const double, nb::shape<-1, 2>,
                      nb::c_contig, nb::device::cpu> absolute_local,
           nb::ndarray<nb::numpy, const double, nb::shape<-1, 2>,
                      nb::c_contig, nb::device::cpu> absolute_target,
           nb::ndarray<nb::numpy, const std::int64_t, nb::shape<-1, 2>,
                      nb::c_contig, nb::device::cpu> relative_patch,
           nb::ndarray<nb::numpy, const double, nb::shape<-1, 2>,
                      nb::c_contig, nb::device::cpu> relative_first,
           nb::ndarray<nb::numpy, const double, nb::shape<-1, 2>,
                      nb::c_contig, nb::device::cpu> relative_second,
           const layout::LayoutOptions& options) {
            if (absolute_patch.shape(0) != absolute_local.shape(0)
                || absolute_patch.shape(0) != absolute_target.shape(0)
                || relative_patch.shape(0) != relative_first.shape(0)
                || relative_patch.shape(0) != relative_second.shape(0)) {
                throw std::invalid_argument("pose graph array counts differ");
            }
            std::vector<registration::Pose2d> initial;
            initial.reserve(poses.shape(0));
            for (std::size_t index = 0; index < poses.shape(0); ++index) {
                initial.push_back({
                    poses(index, 0, 0), poses(index, 0, 1),
                    poses(index, 1, 0), poses(index, 1, 1),
                    poses(index, 0, 2), poses(index, 1, 2),
                });
            }
            std::vector<registration::AbsolutePoseConstraint> absolute;
            absolute.reserve(absolute_patch.shape(0));
            for (std::size_t index = 0; index < absolute_patch.shape(0); ++index) {
                if (absolute_patch(index) < 0) {
                    throw std::invalid_argument("negative absolute patch index");
                }
                absolute.push_back({
                    static_cast<std::size_t>(absolute_patch(index)),
                    absolute_local(index, 0), absolute_local(index, 1),
                    absolute_target(index, 0), absolute_target(index, 1),
                });
            }
            std::vector<registration::RelativePoseConstraint> relative;
            relative.reserve(relative_patch.shape(0));
            for (std::size_t index = 0; index < relative_patch.shape(0); ++index) {
                if (relative_patch(index, 0) < 0 || relative_patch(index, 1) < 0) {
                    throw std::invalid_argument("negative relative patch index");
                }
                relative.push_back({
                    static_cast<std::size_t>(relative_patch(index, 0)),
                    relative_first(index, 0), relative_first(index, 1),
                    static_cast<std::size_t>(relative_patch(index, 1)),
                    relative_second(index, 0), relative_second(index, 1),
                });
            }
            return registration::refine_pose_graph(
                initial, absolute, relative,
                options.uv_ransac_tolerance, 200,
                options.workers > 0 ? options.workers : 1);
        },
        nb::arg("poses"), nb::arg("absolute_patch"),
        nb::arg("absolute_local"), nb::arg("absolute_target"),
        nb::arg("relative_patch"), nb::arg("relative_first"),
        nb::arg("relative_second"),
        nb::arg("options") = layout::LayoutOptions{});

    module.def(
        "layout_largest_fiber_component",
        [](const std::filesystem::path& cache, nb::object provider,
           const layout::LayoutOptions& options) {
            std::optional<std::pair<float, float>> z_range;
            if (nb::hasattr(provider, "z_begin") && nb::hasattr(provider, "z_end")) {
                z_range = std::pair{
                    nb::cast<float>(provider.attr("z_begin")),
                    nb::cast<float>(provider.attr("z_end")),
                };
            }
            return layout::layout_largest_fiber_component(
                cache, theta_provider(provider), options, z_range);
        },
        nb::arg("cache"), nb::arg("theta_provider"),
        nb::arg("options") = layout::LayoutOptions{});

    nb::class_<PyWindingGraph>(module, "WindingGraph")
        .def(nb::init<winding::GraphOptions, nb::object, std::optional<std::filesystem::path>>(),
             nb::arg("options") = winding::GraphOptions{},
             nb::arg("theta_provider") = nb::none(),
             nb::arg("cache_dir") = nb::none())
        .def_static("create", &PyWindingGraph::create,
                    nb::arg("cache_dir"),
                    nb::arg("theta_provider") = nb::none(),
                    nb::arg("options") = winding::GraphOptions{},
                    nb::rv_policy::take_ownership)
        .def_static("open", &PyWindingGraph::open,
                    nb::arg("cache_dir"),
                    nb::arg("theta_provider") = nb::none(),
                    nb::arg("options") = winding::GraphOptions{},
                    nb::rv_policy::take_ownership)
        .def("set_theta_provider", &PyWindingGraph::set_theta_provider,
             nb::arg("provider"))
        .def("add_patch_id", &PyWindingGraph::add_patch_id, nb::arg("patch_id"))
        .def("add_patches", &PyWindingGraph::add_patches, nb::arg("paths"))
        .def("set_patch_valid", &PyWindingGraph::set_patch_valid,
             nb::arg("patch_id"), nb::arg("valid"))
        .def("patch_valid", &PyWindingGraph::patch_valid, nb::arg("patch_id"))
        .def("inspect_contacts", &PyWindingGraph::inspect_contacts,
             nb::arg("zyx"), nb::arg("tolerance") = nb::none())
        .def("patch_layout", &PyWindingGraph::patch_layout,
             nb::arg("patch_id"))
        .def("rasterize_patch_layout", &PyWindingGraph::rasterize_patch_layout,
             nb::arg("patch_id"), nb::arg("root_winding"),
             nb::arg("turn_min"), nb::arg("z_min"), nb::arg("turn_pixels"),
             nb::arg("z_spacing"), nb::arg("patch_index"), nb::arg("raster"),
             nb::arg("labels"))
        .def("fuse_layout_contributions",
             &PyWindingGraph::fuse_layout_contributions,
             nb::arg("pixels"), nb::arg("zyx"), nb::arg("weights"),
             nb::arg("patches"), nb::arg("agreement_distance"),
             nb::arg("continuity_distance"),
             nb::arg("raster"), nb::arg("labels"))
        .def("add_point_collections", &PyWindingGraph::add_point_collections,
             nb::arg("paths"), nb::arg("role"))
        .def("inspect_point_collections", &PyWindingGraph::inspect_point_collections,
             nb::arg("paths"), nb::arg("role"))
        .def("add_fibers", &PyWindingGraph::add_fibers,
             nb::arg("directory"),
             nb::arg("coordinate_scale") = nb::none(),
             nb::arg("invalid_fibers") = std::vector<std::string>{})
        .def("add_tracks", &PyWindingGraph::add_tracks,
             nb::arg("tracks"), nb::arg("crossings") = nb::none(),
             nb::arg("index") = nb::none())
        .def("add_constraint", &PyWindingGraph::add_constraint,
             nb::arg("from_patch"), nb::arg("to_patch"), nb::arg("root_delta"),
             nb::arg("geometric_delta"),
             nb::arg("provenance") = winding::Provenance{})
        .def("add_anchor", &PyWindingGraph::add_anchor,
             nb::arg("patch"), nb::arg("root_winding"),
             nb::arg("geometric_root_winding"),
             nb::arg("provenance") = winding::Provenance{})
        .def("lifted_relative_winding", &PyWindingGraph::lifted_relative_winding,
             nb::arg("from_patch"), nb::arg("to_patch"))
        .def("holonomy", &PyWindingGraph::holonomy, nb::arg("index"))
        .def("holonomy_audits", &PyWindingGraph::holonomy_audits)
        .def("stats", &PyWindingGraph::stats)
        .def("has_patch", &PyWindingGraph::has_patch, nb::arg("patch"))
        .def("patch_node", &PyWindingGraph::patch_node, nb::arg("patch"))
        .def("node_name", &PyWindingGraph::node_name, nb::arg("node"))
        .def("constraints", &PyWindingGraph::constraints)
        .def("inspect_tracks", &PyWindingGraph::inspect_tracks,
             nb::arg("tracks"), nb::arg("crossings") = nb::none())
        .def("prepare_track_index", &PyWindingGraph::prepare_track_index,
             nb::arg("tracks"), nb::arg("output") = nb::none(),
             nb::arg("cell_size") = 32,
             nb::arg("memory_budget_bytes") = std::size_t{512} << 20)
        .def("save", &PyWindingGraph::save, nb::arg("cache_dir") = nb::none());
}
