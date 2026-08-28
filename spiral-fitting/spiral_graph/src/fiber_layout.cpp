#include <spiral_graph/fiber_layout.hpp>

#include <spiral_graph/fiber_io.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <ceres/ceres.h>
#include <nlohmann/json.hpp>

namespace spiral::layout {
namespace {

using json = nlohmann::json;
constexpr double pi = 3.141592653589793238462643383279502884;

class Dsu {
public:
    explicit Dsu(std::size_t count) : parent_(count), size_(count, 1)
    {
        std::iota(parent_.begin(), parent_.end(), 0);
    }
    std::size_t find(std::size_t node)
    {
        while (parent_[node] != node) {
            parent_[node] = parent_[parent_[node]];
            node = parent_[node];
        }
        return node;
    }
    void merge(std::size_t a, std::size_t b)
    {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (size_[a] < size_[b]) std::swap(a, b);
        parent_[b] = a;
        size_[a] += size_[b];
    }
private:
    std::vector<std::size_t> parent_;
    std::vector<std::size_t> size_;
};

struct Link {
    std::size_t a = 0;
    std::size_t pa = 0;
    std::size_t b = 0;
    std::size_t pb = 0;
};

struct Prepared {
    fiber::Fiber source;
    bool reversed = false;
    std::vector<double> arclength;
    std::vector<float> theta;
    std::vector<float> fraction;
    std::vector<std::int32_t> local_winding;
};

double median(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const double high = values[middle];
    if (values.size() % 2) return high;
    std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
    return 0.5 * (values[middle - 1] + high);
}

double wrapped_delta(double value)
{
    while (value > pi) value -= 2.0 * pi;
    while (value <= -pi) value += 2.0 * pi;
    return value;
}

std::vector<float> evaluate(
    const winding::ThetaProvider& provider,
    const std::vector<winding::Zyx>& points,
    std::size_t batch_size)
{
    if (!provider) throw std::invalid_argument("checkpoint theta is required");
    if (batch_size == 0) throw std::invalid_argument("theta batch size must be positive");
    std::vector<float> values;
    values.reserve(points.size());
    for (std::size_t begin = 0; begin < points.size(); begin += batch_size) {
        const std::size_t count = std::min(batch_size, points.size() - begin);
        std::vector<float> batch = provider(
            std::span<const winding::Zyx>(points).subspan(begin, count));
        if (batch.size() != count) {
            throw std::runtime_error("checkpoint theta returned the wrong batch size");
        }
        for (float value : batch) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("checkpoint theta returned a non-finite value");
            }
        }
        values.insert(values.end(), batch.begin(), batch.end());
    }
    return values;
}

std::vector<double> point_arclength(const fiber::Fiber& value, bool reverse)
{
    std::vector<double> output(value.points.size(), 0.0);
    for (std::size_t index = 1; index < value.points.size(); ++index) {
        const std::size_t a = reverse ? value.points.size() - index : index - 1;
        const std::size_t b = reverse ? value.points.size() - index - 1 : index;
        const double dz = value.points[b].z - value.points[a].z;
        const double dy = value.points[b].y - value.points[a].y;
        const double dx = value.points[b].x - value.points[a].x;
        output[index] = output[index - 1] + std::sqrt(dz * dz + dy * dy + dx * dx);
    }
    return output;
}

std::size_t oriented_index(const Prepared& fiber, std::size_t original)
{
    return fiber.reversed ? fiber.source.points.size() - original - 1 : original;
}

std::size_t original_index(const Prepared& fiber, std::size_t oriented)
{
    return fiber.reversed ? fiber.source.points.size() - oriented - 1 : oriented;
}

bool horizontal_should_reverse(
    const fiber::Fiber& value, const std::vector<float>& theta)
{
    std::vector<double> slopes;
    for (std::size_t index = 1; index < value.points.size(); ++index) {
        if (!std::isfinite(theta[index - 1]) || !std::isfinite(theta[index])) continue;
        const double dz = value.points[index].z - value.points[index - 1].z;
        const double dy = value.points[index].y - value.points[index - 1].y;
        const double dx = value.points[index].x - value.points[index - 1].x;
        const double distance = std::sqrt(dz * dz + dy * dy + dx * dx);
        if (distance > 1e-8) {
            slopes.push_back(wrapped_delta(theta[index] - theta[index - 1]) / distance);
        }
    }
    double slope = median(std::move(slopes));
    if (std::abs(slope) < 1e-12 && theta.size() > 1) {
        double accumulated = 0.0;
        for (std::size_t index = 1; index < theta.size(); ++index) {
            if (!std::isfinite(theta[index - 1]) || !std::isfinite(theta[index])) continue;
            accumulated += wrapped_delta(theta[index] - theta[index - 1]);
        }
        slope = accumulated;
    }
    if (std::abs(slope) < 1e-12) {
        throw std::runtime_error(
            "checkpoint theta cannot orient horizontal fiber " + value.id);
    }
    return slope < 0.0;
}

bool vertical_should_reverse(const fiber::Fiber& value)
{
    std::vector<double> slopes;
    for (std::size_t index = 1; index < value.points.size(); ++index) {
        const double dz = value.points[index].z - value.points[index - 1].z;
        const double dy = value.points[index].y - value.points[index - 1].y;
        const double dx = value.points[index].x - value.points[index - 1].x;
        const double distance = std::sqrt(dz * dz + dy * dy + dx * dx);
        if (distance > 1e-8) slopes.push_back(dz / distance);
    }
    double slope = median(std::move(slopes));
    if (std::abs(slope) < 1e-12 && value.points.size() > 1) {
        slope = value.points.back().z - value.points.front().z;
    }
    if (std::abs(slope) < 1e-12) {
        throw std::runtime_error("physical z cannot orient vertical fiber " + value.id);
    }
    return slope < 0.0;
}

std::vector<Link> resolve_links(const std::vector<Prepared>& fibers)
{
    std::unordered_map<std::string, std::size_t> by_id;
    for (std::size_t index = 0; index < fibers.size(); ++index) {
        if (!by_id.emplace(fibers[index].source.id, index).second) {
            throw std::runtime_error("duplicate fiber ID " + fibers[index].source.id);
        }
    }
    std::set<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>> seen;
    std::vector<Link> links;
    for (std::size_t a = 0; a < fibers.size(); ++a) {
        for (const fiber::Branch& branch : fibers[a].source.branches) {
            if (branch.pending) continue;
            const auto found = by_id.find(branch.other_file);
            if (found == by_id.end()) continue;
            const std::size_t b = found->second;
            if (fibers[a].source.points.empty() || fibers[b].source.points.empty()
                || branch.other_control >= fibers[b].source.control_line.size()
                || fibers[b].source.control_line[branch.other_control]
                    == std::numeric_limits<std::size_t>::max()) continue;
            const std::size_t pa_original = fiber::retained_control_point(
                fibers[a].source, branch.local_control);
            const std::size_t pb_original = fiber::retained_control_point(
                fibers[b].source, branch.other_control);
            const std::size_t pa = oriented_index(fibers[a], pa_original);
            const std::size_t pb = oriented_index(fibers[b], pb_original);
            auto first = std::pair{a, pa};
            auto second = std::pair{b, pb};
            if (second < first) std::swap(first, second);
            if (!seen.emplace(first.first, first.second, second.first, second.second).second) {
                continue;
            }
            links.push_back({first.first, first.second, second.first, second.second});
        }
    }
    std::sort(links.begin(), links.end(), [](const Link& a, const Link& b) {
        return std::tie(a.a, a.pa, a.b, a.pb) < std::tie(b.a, b.pa, b.b, b.pb);
    });
    return links;
}

struct DifferenceResidual {
    DifferenceResidual(double a, double b) : a(a), b(b) {}
    template <typename T> bool operator()(const T* x, const T* y, T* residual) const
    {
        residual[0] = (x[0] + T(a)) - (y[0] + T(b));
        return true;
    }
    double a;
    double b;
};

struct CoordinateResidual {
    explicit CoordinateResidual(double value) : value(value) {}
    template <typename T> bool operator()(const T* offset, const T* line, T* residual) const
    {
        residual[0] = offset[0] + T(value) - line[0];
        return true;
    }
    double value;
};

std::int32_t crossing_step(float before, float after)
{
    const double delta = static_cast<double>(after) - before;
    if (delta > pi) return -1;
    if (delta < -pi) return 1;
    return 0;
}

} // namespace

LayoutResult layout_largest_fiber_component(
    const std::filesystem::path& cache_directory,
    const winding::ThetaProvider& checkpoint_theta,
    const LayoutOptions& options,
    std::optional<std::pair<float, float>> checkpoint_z_range)
{
    if (!(options.contact_tolerance >= 0.0)
        || !(options.uv_ransac_tolerance > 0.0)
        || !(options.max_refit_rms > 0.0)
        || options.min_inliers == 0 || options.ransac_hypotheses == 0
        || options.max_raster_samples == 0) {
        throw std::invalid_argument("invalid layout options");
    }
    if (checkpoint_z_range
        && (!(checkpoint_z_range->first < checkpoint_z_range->second)
            || !std::isfinite(checkpoint_z_range->first)
            || !std::isfinite(checkpoint_z_range->second))) {
        throw std::invalid_argument("checkpoint z range must be finite and ordered");
    }
    std::ifstream manifest_stream(cache_directory / "manifest.json");
    if (!manifest_stream) throw std::runtime_error("cannot open graph manifest");
    json manifest;
    manifest_stream >> manifest;
    if (manifest.value("schema", std::string{}) != "spiral-winding-graph") {
        throw std::runtime_error("unsupported graph manifest");
    }
    const json cached_options = manifest.value("options", json::object());
    std::vector<fiber::Fiber> loaded;
    for (const auto& source : manifest.at("sources")) {
        if (source.at("kind").get<int>() != 1) continue;
        fiber::LoadOptions load;
        load.coordinate_scale = source.value(
            "coordinate_scale", cached_options.value("fiber_coordinate_scale", 0.25f));
        for (const std::string& id : source.value(
                 "invalid_items", std::vector<std::string>{})) {
            load.invalid_fibers.insert(id);
        }
        for (const auto& path : source.at("paths")) {
            auto next = fiber::load_vc3d_fiber_directory(
                path.get<std::string>(), load);
            loaded.insert(
                loaded.end(), std::make_move_iterator(next.begin()),
                std::make_move_iterator(next.end()));
        }
    }
    if (loaded.empty()) throw std::runtime_error("cache contains no approved fibers");
    std::sort(loaded.begin(), loaded.end(), [](const auto& a, const auto& b) {
        return a.id < b.id;
    });

    std::vector<winding::Zyx> all_points;
    for (const auto& value : loaded) {
        for (const auto& point : value.points) {
            all_points.push_back({point.z, point.y, point.x});
        }
    }
    std::vector<winding::Zyx> covered_points;
    std::vector<std::size_t> covered_indices;
    covered_points.reserve(all_points.size());
    covered_indices.reserve(all_points.size());
    for (std::size_t index = 0; index < all_points.size(); ++index) {
        const bool covered = !checkpoint_z_range
            || (all_points[index].z >= checkpoint_z_range->first
                && all_points[index].z < checkpoint_z_range->second);
        if (covered) {
            covered_points.push_back(all_points[index]);
            covered_indices.push_back(index);
        }
    }
    const std::vector<float> covered_theta = evaluate(
        checkpoint_theta, covered_points, options.theta_batch_size);
    std::vector<float> all_theta(
        all_points.size(), std::numeric_limits<float>::quiet_NaN());
    for (std::size_t index = 0; index < covered_indices.size(); ++index) {
        all_theta[covered_indices[index]] = covered_theta[index];
    }
    std::vector<Prepared> prepared;
    prepared.reserve(loaded.size());
    std::vector<std::string> without_checkpoint_theta;
    std::size_t theta_offset = 0;
    for (auto& value : loaded) {
        std::vector<float> theta(
            all_theta.begin() + static_cast<std::ptrdiff_t>(theta_offset),
            all_theta.begin() + static_cast<std::ptrdiff_t>(
                theta_offset + value.points.size()));
        theta_offset += value.points.size();
        if (std::none_of(theta.begin(), theta.end(), [](float item) {
                return std::isfinite(item);
            })) {
            without_checkpoint_theta.push_back(value.id);
            continue;
        }
        Prepared next;
        next.source = std::move(value);
        next.reversed = next.source.axis == fiber::Axis::horizontal
            ? horizontal_should_reverse(next.source, theta)
            : vertical_should_reverse(next.source);
        if (next.reversed) std::reverse(theta.begin(), theta.end());
        next.theta = std::move(theta);
        next.arclength = point_arclength(next.source, next.reversed);
        next.fraction.assign(
            next.theta.size(), std::numeric_limits<float>::quiet_NaN());
        next.local_winding.assign(next.theta.size(), 0);
        bool previous_valid = false;
        bool completed_valid_run = false;
        for (std::size_t point = 0; point < next.theta.size(); ++point) {
            if (!std::isfinite(next.theta[point])) {
                if (previous_valid) completed_valid_run = true;
                previous_valid = false;
                continue;
            }
            if (completed_valid_run) {
                throw std::runtime_error(
                    "checkpoint coverage is disconnected along fiber "
                    + next.source.id);
            }
            double fraction = std::fmod(
                static_cast<double>(next.theta[point]) / (2.0 * pi), 1.0);
            if (fraction < 0.0) fraction += 1.0;
            if (fraction >= 1.0) fraction = 0.0;
            next.fraction[point] = static_cast<float>(fraction);
            // Only H fibers advance around theta.  V fibers transport the
            // already-established sheet gauge between H tracks; a noisy seam
            // crossing along z must never manufacture another winding.
            if (previous_valid
                && next.source.axis == fiber::Axis::horizontal) {
                next.local_winding[point] = next.local_winding[point - 1]
                    + crossing_step(next.theta[point - 1], next.theta[point]);
            } else if (previous_valid) {
                next.local_winding[point] = next.local_winding[point - 1];
            }
            previous_valid = true;
        }
        prepared.push_back(std::move(next));
    }
    if (prepared.empty()) {
        throw std::runtime_error("no fiber points lie inside the checkpoint z range");
    }
    const std::vector<Link> links = resolve_links(prepared);

    Dsu components(prepared.size());
    for (const Link& link : links) components.merge(link.a, link.b);
    struct ComponentRank {
        std::vector<std::size_t> members;
        double arclength = 0.0;
        std::string lexical;
    };
    std::map<std::size_t, ComponentRank> groups;
    for (std::size_t index = 0; index < prepared.size(); ++index) {
        auto& group = groups[components.find(index)];
        group.members.push_back(index);
        group.arclength += prepared[index].arclength.empty()
            ? 0.0 : prepared[index].arclength.back();
        if (group.lexical.empty() || prepared[index].source.id < group.lexical) {
            group.lexical = prepared[index].source.id;
        }
    }
    auto selected = std::max_element(groups.begin(), groups.end(), [](const auto& a, const auto& b) {
        const auto& x = a.second;
        const auto& y = b.second;
        if (x.members.size() != y.members.size()) return x.members.size() < y.members.size();
        if (x.arclength != y.arclength) return x.arclength < y.arclength;
        return x.lexical > y.lexical;
    });
    if (selected == groups.end()) throw std::runtime_error("no fiber component exists");
    std::vector<bool> member(prepared.size(), false);
    for (std::size_t index : selected->second.members) member[index] = true;

    Dsu tracks(prepared.size());
    std::vector<Link> selected_links;
    for (const Link& link : links) {
        if (!member[link.a] || !member[link.b]) continue;
        selected_links.push_back(link);
        if (prepared[link.a].source.axis == prepared[link.b].source.axis) {
            tracks.merge(link.a, link.b);
        }
    }
    std::map<std::size_t, std::size_t> track_numbers;
    for (std::size_t index : selected->second.members) {
        const std::size_t root = tracks.find(index);
        if (!track_numbers.contains(root)) {
            track_numbers.emplace(root, track_numbers.size());
        }
    }

    std::vector<double> offsets(prepared.size(), 0.0);
    std::vector<double> lines(track_numbers.size(), 0.0);
    ceres::Problem problem;
    auto* loss = new ceres::HuberLoss(20.0);
    for (const Link& link : selected_links) {
        const bool same_axis = prepared[link.a].source.axis
            == prepared[link.b].source.axis;
        if (same_axis) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<DifferenceResidual, 1, 1, 1>(
                    new DifferenceResidual(
                        prepared[link.a].arclength[link.pa],
                        prepared[link.b].arclength[link.pb])),
                loss, &offsets[link.a], &offsets[link.b]);
            continue;
        }
        const std::size_t h = prepared[link.a].source.axis == fiber::Axis::horizontal
            ? link.a : link.b;
        const std::size_t ph = h == link.a ? link.pa : link.pb;
        const std::size_t v = h == link.a ? link.b : link.a;
        const std::size_t pv = h == link.a ? link.pb : link.pa;
        const std::size_t h_track = track_numbers.at(tracks.find(h));
        const std::size_t v_track = track_numbers.at(tracks.find(v));
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CoordinateResidual, 1, 1, 1>(
                new CoordinateResidual(prepared[h].arclength[ph])),
            loss, &offsets[h], &lines[v_track]);
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CoordinateResidual, 1, 1, 1>(
                new CoordinateResidual(prepared[v].arclength[pv])),
            loss, &offsets[v], &lines[h_track]);
    }
    std::optional<std::size_t> anchor_h;
    std::optional<std::size_t> anchor_v;
    for (std::size_t index : selected->second.members) {
        if (prepared[index].source.axis == fiber::Axis::horizontal && !anchor_h) anchor_h = index;
        if (prepared[index].source.axis == fiber::Axis::vertical && !anchor_v) anchor_v = index;
    }
    if (!anchor_h || !anchor_v) {
        throw std::runtime_error("largest fiber component must contain H and V fibers");
    }
    problem.SetParameterBlockConstant(
        &lines[track_numbers.at(tracks.find(*anchor_h))]);
    problem.SetParameterBlockConstant(
        &lines[track_numbers.at(tracks.find(*anchor_v))]);
    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = 200;
    solver_options.num_threads = options.workers > 0 ? options.workers : 1;
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
        throw std::runtime_error("fiber layout solve failed: " + summary.BriefReport());
    }
    for (std::size_t index : selected->second.members) {
        if (!std::isfinite(offsets[index])
            || !std::isfinite(lines[track_numbers.at(tracks.find(index))])) {
            throw std::runtime_error("fiber layout contains a non-finite coordinate");
        }
    }
    for (const Link& link : selected_links) {
        if (prepared[link.a].source.axis != prepared[link.b].source.axis) continue;
        const double residual
            = offsets[link.a] + prepared[link.a].arclength[link.pa]
            - offsets[link.b] - prepared[link.b].arclength[link.pb];
        if (std::abs(residual) > 1e-4) {
            throw std::runtime_error(
                "unresolved same-axis continuation cycle at "
                + prepared[link.a].source.id + " -> "
                + prepared[link.b].source.id);
        }
    }
    std::map<std::size_t, std::vector<std::pair<double, double>>> knot_orders;
    for (const Link& link : selected_links) {
        if (prepared[link.a].source.axis == prepared[link.b].source.axis) continue;
        const std::size_t h = prepared[link.a].source.axis == fiber::Axis::horizontal
            ? link.a : link.b;
        const std::size_t ph = h == link.a ? link.pa : link.pb;
        const std::size_t v = h == link.a ? link.b : link.a;
        const std::size_t pv = h == link.a ? link.pb : link.pa;
        knot_orders[tracks.find(h)].push_back({
            offsets[h] + prepared[h].arclength[ph],
            lines[track_numbers.at(tracks.find(v))],
        });
        knot_orders[tracks.find(v)].push_back({
            offsets[v] + prepared[v].arclength[pv],
            lines[track_numbers.at(tracks.find(h))],
        });
    }
    for (auto& [track, knots] : knot_orders) {
        static_cast<void>(track);
        std::sort(knots.begin(), knots.end());
        for (std::size_t index = 1; index < knots.size(); ++index) {
            if (knots[index].first > knots[index - 1].first + 1e-6
                && knots[index].second <= knots[index - 1].second) {
                throw std::runtime_error("fiber crossing solve inverted knot order");
            }
        }
    }

    // The Ceres variables define the common coordinate of every crossing
    // track.  Raw arclength is only the metric observation used by the solve;
    // emitting offset + arclength here would put the two sides of a robust
    // crossing residual at different UV positions.  Interpolate through the
    // optimized knots so every H/V link is one exact UV point, and retain
    // unit slope only beyond the observed knot interval.
    const auto mapped_track_coordinate = [&](std::size_t track, double value) {
        const auto iterator = knot_orders.find(track);
        if (iterator == knot_orders.end() || iterator->second.empty()) {
            throw std::runtime_error("logical fiber track has no crossing knot");
        }
        const auto& knots = iterator->second;
        if (knots.size() == 1 || value <= knots.front().first) {
            return knots.front().second + value - knots.front().first;
        }
        if (value >= knots.back().first) {
            return knots.back().second + value - knots.back().first;
        }
        const auto upper = std::upper_bound(
            knots.begin(), knots.end(), value,
            [](double coordinate, const auto& knot) {
                return coordinate < knot.first;
            });
        const auto lower = std::prev(upper);
        const double span = upper->first - lower->first;
        if (!(span > 1e-8)) {
            throw std::runtime_error("fiber crossing solve has duplicate knot coordinates");
        }
        const double alpha = (value - lower->first) / span;
        return lower->second + alpha * (upper->second - lower->second);
    };

    // Propagate exact integer winding offsets and reject contradictory cycles.
    std::vector<std::vector<std::pair<std::size_t, std::int32_t>>> winding_graph(
        prepared.size());
    for (const Link& link : selected_links) {
        if (!std::isfinite(prepared[link.a].fraction[link.pa])
            || !std::isfinite(prepared[link.b].fraction[link.pb])) {
            continue;
        }
        const double turn_a = prepared[link.a].local_winding[link.pa]
            + prepared[link.a].fraction[link.pa];
        const double turn_b = prepared[link.b].local_winding[link.pb]
            + prepared[link.b].fraction[link.pb];
        const auto delta = static_cast<std::int32_t>(std::llround(turn_a - turn_b));
        winding_graph[link.a].push_back({link.b, delta});
        winding_graph[link.b].push_back({link.a, -delta});
    }
    const auto root_iterator = std::min_element(
        selected->second.members.begin(), selected->second.members.end(),
        [&](std::size_t a, std::size_t b) {
            const bool a_horizontal
                = prepared[a].source.axis == fiber::Axis::horizontal;
            const bool b_horizontal
                = prepared[b].source.axis == fiber::Axis::horizontal;
            if (a_horizontal != b_horizontal) return a_horizontal;
            return prepared[a].source.id < prepared[b].source.id;
        });
    const std::size_t root = *root_iterator;
    const auto first_root_theta = std::find_if(
        prepared[root].theta.begin(), prepared[root].theta.end(), [](float value) {
            return std::isfinite(value);
        });
    if (first_root_theta == prepared[root].theta.end()) {
        throw std::logic_error("selected root fiber has no checkpoint theta");
    }
    const std::size_t root_point = static_cast<std::size_t>(
        first_root_theta - prepared[root].theta.begin());
    constexpr std::int32_t unset = std::numeric_limits<std::int32_t>::min();
    std::vector<std::int32_t> winding_offset(prepared.size(), unset);
    winding_offset[root] = -prepared[root].local_winding[root_point];
    std::deque<std::size_t> queue{root};
    while (!queue.empty()) {
        const std::size_t a = queue.front();
        queue.pop_front();
        for (const auto [b, delta] : winding_graph[a]) {
            const std::int32_t expected = winding_offset[a] + delta;
            if (winding_offset[b] == unset) {
                winding_offset[b] = expected;
                queue.push_back(b);
            } else if (winding_offset[b] != expected) {
                throw std::runtime_error(
                    "conflicting checkpoint winding cycle at "
                    + prepared[a].source.id + " -> " + prepared[b].source.id);
            }
        }
    }
    for (const std::size_t index : selected->second.members) {
        if (winding_offset[index] == unset) {
            throw std::runtime_error(
                "checkpoint-covered fiber has no global winding path: "
                + prepared[index].source.id);
        }
    }

    LayoutResult result;
    result.root_fiber = prepared[root].source.id;
    result.total_arclength = selected->second.arclength;
    result.initial_cost = summary.initial_cost;
    result.final_cost = summary.final_cost;
    result.solver_iterations = summary.iterations.size();
    result.theta_covered_points = covered_indices.size();
    result.theta_uncovered_points = all_points.size() - covered_indices.size();
    result.excluded_fibers = std::move(without_checkpoint_theta);
    for (std::size_t index = 0; index < prepared.size(); ++index) {
        if (!member[index]) result.excluded_fibers.push_back(prepared[index].source.id);
    }
    std::sort(result.excluded_fibers.begin(), result.excluded_fibers.end());
    for (std::size_t index : selected->second.members) {
        const Prepared& source = prepared[index];
        FiberLayout output;
        output.id = source.source.id;
        output.axis = source.source.axis == fiber::Axis::horizontal ? 'H' : 'V';
        output.logical_track = track_numbers.at(tracks.find(index));
        output.reversed = source.reversed;
        output.arclength = source.arclength.empty() ? 0.0 : source.arclength.back();
        output.winding_offset = winding_offset[index];
        const double line = lines[output.logical_track];
        for (std::size_t point = 0; point < source.source.points.size(); ++point) {
            const std::size_t original = original_index(source, point);
            const fiber::Point xyz = source.source.points[original];
            const double track_arclength
                = offsets[index] + source.arclength[point];
            const double along = mapped_track_coordinate(
                tracks.find(index), track_arclength);
            const bool theta_valid = std::isfinite(source.fraction[point]);
            output.points.push_back({
                xyz.z, xyz.y, xyz.x,
                source.source.axis == fiber::Axis::horizontal ? along : line,
                source.source.axis == fiber::Axis::horizontal ? line : along,
                theta_valid ? static_cast<std::int32_t>(
                    winding_offset[index] + source.local_winding[point]) : 0,
                theta_valid ? source.fraction[point] : 0.0f,
                theta_valid,
            });
        }
        result.fibers.push_back(std::move(output));
    }
    std::sort(result.fibers.begin(), result.fibers.end(), [](const auto& a, const auto& b) {
        return a.id < b.id;
    });
    for (const Link& link : selected_links) {
        if (prepared[link.a].source.axis == prepared[link.b].source.axis) continue;
        const std::size_t h = prepared[link.a].source.axis == fiber::Axis::horizontal
            ? link.a : link.b;
        const std::size_t ph = h == link.a ? link.pa : link.pb;
        const std::size_t v = h == link.a ? link.b : link.a;
        const std::size_t pv = h == link.a ? link.pb : link.pa;
        const double h_u = mapped_track_coordinate(
            tracks.find(h), offsets[h] + prepared[h].arclength[ph]);
        const double h_v = lines[track_numbers.at(tracks.find(h))];
        const double v_u = lines[track_numbers.at(tracks.find(v))];
        const double v_v = mapped_track_coordinate(
            tracks.find(v), offsets[v] + prepared[v].arclength[pv]);
        result.crossings.push_back({
            prepared[h].source.id, ph, prepared[v].source.id, pv,
            h_u - v_u, h_v - v_v,
        });
    }
    return result;
}

} // namespace spiral::layout
