// Shared line->patch geometry for fibers and tracks: a port of
// vc::constraints::autoLinkFiber + makeFastFiberNode from the removed
// constraint-graph workspace (see ConstraintGraph.md in git history at
// 679effcbe^). Differences from the original, by design here:
//  * the fiber strip is reduced: stripRowsEachSide rows (default 1)
//    at stripSpacing instead of the original 3 rows each side;
//  * strip normals come from CT (1/24/4 anisotropic tensor), without pred_dt snapping;
//  * geometry is converted into explicit working voxel coordinates, and
//    results are committed as pending graph edges via commitCandidates
//    instead of review decisions.

#include "GraphLinksLinePatch.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <thread>
#include <unordered_map>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/Uuid.hpp"
#include "vc/core/util/CtSheetNormal.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

namespace vc::graph_links::detail {
namespace {

// Both sources feed exactly the same strip marching and patch-link gates.
// Profiling includes cached queries, and does not alter either source's cache.
class SheetNormalSource {
public:
    SheetNormalSource(const std::filesystem::path& ctVolume, const LinePatchParams& params) {
        if (params.normalsManifest.empty()) {
            ct_ = std::make_unique<vc::CtSheetNormalSampler>(ctVolume, params.ctLevel,
                params.workingToBaseScale / params.ctToBaseScale);
        } else {
            vc::lasagna::LasagnaDatasetOpenOptions options;
            options.workingToBaseScale = params.workingToBaseScale;
            dataset_ = std::make_unique<vc::lasagna::LasagnaDataset>(
                vc::lasagna::LasagnaDataset::openLocation(params.normalsManifest, options));
            lasagna_ = std::make_unique<vc::lasagna::LasagnaNormalSampler>(*dataset_);
        }
    }
    ~SheetNormalSource() {
        if (profile_) printProfile("total");
    }
    const char* method() const {
        return ct_ ? vc::CtSheetNormalSampler::method : "lasagna-normal-v1";
    }
    vc::CtSheetNormal sampleNormal(const cv::Vec3d& point, const std::atomic_bool* cancel) const {
        if (cancel && cancel->load()) throw std::runtime_error("graph links: line-patch linking cancelled");
        const auto start = profile_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        vc::CtSheetNormal result;
        if (ct_) result = ct_->sampleNormal(point, cancel);
        else {
            const auto sample = lasagna_->sampleNormal(point);
            result = {sample.normal, sample.valid, 0, sample.reason};
        }
        if (profile_) {
            seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            ++queries_;
            invalid_ += !result.valid;
            if (queries_ % 100 == 0) printProfile("progress");
        }
        return result;
    }
private:
    void printProfile(const char* phase) const {
        std::fprintf(stderr, "FIBER_PATCH_PROFILE %s queries %zu normal_seconds %.9f invalid %zu\n",
                     phase, queries_, seconds_, invalid_);
    }
    std::unique_ptr<vc::CtSheetNormalSampler> ct_;
    std::unique_ptr<vc::lasagna::LasagnaDataset> dataset_;
    std::unique_ptr<vc::lasagna::LasagnaNormalSampler> lasagna_;
    bool profile_ = std::getenv("VC_FIBER_PATCH_PROFILE") != nullptr;
    mutable size_t queries_ = 0, invalid_ = 0;
    mutable double seconds_ = 0;
};

cv::Vec3d unit(const cv::Vec3d& v)
{
    const double n = cv::norm(v);
    return n > 1e-12 ? v / n : cv::Vec3d{0, 0, 0};
}

void checkCancelled(const std::atomic_bool* cancel)
{
    if (cancel && cancel->load())
        throw std::runtime_error("graph links: line-patch linking cancelled");
}

// Interpolate entries flagged bad from nearest good neighbours.
void interpolateBad(std::vector<cv::Vec3d>& values, std::vector<char>& bad, bool unitize)
{
    const int n = int(values.size());
    int firstGood = -1;
    for (int i = 0; i < n; ++i)
        if (!bad[i]) { firstGood = i; break; }
    if (firstGood < 0) return;
    for (int i = 0; i < firstGood; ++i) values[i] = values[firstGood];
    int lastGood = firstGood;
    for (int i = firstGood + 1; i < n; ++i) {
        if (bad[i]) continue;
        for (int j = lastGood + 1; j < i; ++j) {
            const double f = double(j - lastGood) / (i - lastGood);
            values[j] = values[lastGood] * (1 - f) + values[i] * f;
            if (unitize) values[j] = unit(values[j]);
        }
        lastGood = i;
    }
    for (int i = lastGood + 1; i < n; ++i) values[i] = values[lastGood];
    std::fill(bad.begin(), bad.end(), 0);
}

std::vector<cv::Vec3d> gaussianAlong(const std::vector<cv::Vec3d>& values, double sigma)
{
    const int n = int(values.size());
    std::vector<cv::Vec3d> out(n);
    const int radius = int(std::ceil(3 * sigma));
    for (int i = 0; i < n; ++i) {
        cv::Vec3d sum{0, 0, 0};
        double weight = 0;
        for (int k = -radius; k <= radius; ++k) {
            const int j = i + k;
            if (j < 0 || j >= n) continue;
            const double w = std::exp(-0.5 * k * k / (sigma * sigma));
            sum += values[j] * w;
            weight += w;
        }
        out[i] = sum / weight;
    }
    return out;
}

// The reduced fast strip: centerline resampled every stripSpacing,
// marched stripRowsEachSide rows out along the in-sheet cross direction
// while following normals estimated directly from CT intensities.
std::shared_ptr<QuadSurface> buildStrip(const LineData& fiber, const SheetNormalSource& sampler,
                                        const LinePatchParams& params, const std::atomic_bool* cancel)
{
    const double spacing = params.stripSpacing;
    const int rowsEachSide = std::max(0, params.stripRowsEachSide);
    const int rows = 2 * rowsEachSide + 1;
    const int middle = rowsEachSide;
    const auto& arc = fiber.arclength;

    std::vector<double> columns;
    for (double a = arc[fiber.firstLine]; a < arc[fiber.lastLine]; a += spacing) columns.push_back(a);
    if (columns.size() < 2 || arc[fiber.lastLine] - columns.back() > spacing * 0.25)
        columns.push_back(arc[fiber.lastLine]);
    else
        columns.back() = arc[fiber.lastLine];
    const int cols = int(columns.size());
    auto atArc = [&](double a) {
        auto it = std::lower_bound(arc.begin(), arc.end(), a);
        if (it == arc.begin()) return fiber.line.front();
        if (it == arc.end()) return fiber.line.back();
        const size_t hi = size_t(it - arc.begin());
        const double span = arc[hi] - arc[hi - 1];
        const double t = span > 0 ? (a - arc[hi - 1]) / span : 0;
        return fiber.line[hi - 1] * (1 - t) + fiber.line[hi] * t;
    };

    std::vector<cv::Vec3d> center(cols), tangent(cols), normal(cols);
    std::vector<char> badNormal(cols, 0);
    for (int c = 0; c < cols; ++c) center[c] = atArc(columns[c]);
    for (int c = 0; c < cols; ++c)
        tangent[c] = unit(center[std::min(c + 1, cols - 1)] - center[std::max(c - 1, 0)]);
    bool anyNormal = false;
    for (int c = 0; c < cols; ++c) {
        checkCancelled(cancel);
        const auto sample = sampler.sampleNormal(center[c], cancel);
        badNormal[c] = !sample.valid || cv::norm(sample.normal) < 1e-6;
        normal[c] = badNormal[c] ? cv::Vec3d{0, 0, 0} : unit(sample.normal);
        anyNormal |= !badNormal[c];
    }
    if (!anyNormal)
        throw std::runtime_error("no valid sheet normals along fiber " + fiber.uuid);
    int previous = -1;
    for (int c = 0; c < cols; ++c) {
        if (badNormal[c]) continue;
        if (previous >= 0 && normal[c].dot(normal[previous]) < 0) normal[c] = -normal[c];
        previous = c;
    }
    interpolateBad(normal, badNormal, true);
    {
        auto smoothed = gaussianAlong(normal, 2.0);
        for (auto& n : smoothed) n = unit(n);
        for (int c = 0; c < cols; ++c)
            if (normal[c].dot(smoothed[c]) < std::cos(35.0 * CV_PI / 180.0)) badNormal[c] = 1;
        interpolateBad(normal, badNormal, true);
        for (int c = 0; c < cols; ++c)
            if (std::abs(normal[c].dot(tangent[c])) > std::cos(20.0 * CV_PI / 180.0)) badNormal[c] = 1;
        interpolateBad(normal, badNormal, true);
    }
    std::vector<cv::Vec3d> cross(cols);
    std::vector<char> badCross(cols, 0);
    for (int c = 0; c < cols; ++c) {
        cross[c] = unit(normal[c].cross(tangent[c]));
        badCross[c] = cv::norm(cross[c]) < 0.5;
    }
    if (std::all_of(badCross.begin(), badCross.end(), [](char b) { return b; }))
        throw std::runtime_error("no usable in-sheet cross direction for fiber " + fiber.uuid);
    interpolateBad(cross, badCross, true);

    const int stepsPerRow = std::max(1, int(std::lround(spacing / params.stripMarchStep)));
    const double step = spacing / stepsPerRow;
    std::vector<cv::Vec3d> xyz(size_t(rows) * cols);
    for (int c = 0; c < cols; ++c) xyz[size_t(middle) * cols + c] = center[c];
    // Columns march independently. Start where the centerline pass finished
    // so its most recent CT gradient tiles can be reused before eviction.
    for (int c = cols; c-- > 0;) {
        checkCancelled(cancel);
        for (int side : {-1, 1}) {
            if (rowsEachSide == 0) break;
            cv::Vec3d p = center[c], direction = cross[c] * double(side), n = normal[c];
            for (int r = 1; r <= rowsEachSide; ++r) {
                for (int k = 0; k < stepsPerRow; ++k) {
                    const auto sample = sampler.sampleNormal(p, cancel);
                    if (sample.valid && cv::norm(sample.normal) > 1e-6) {
                        auto local = unit(sample.normal);
                        if (local.dot(n) < 0) local = -local;
                        if (local.dot(n) > std::cos(35.0 * CV_PI / 180.0)) n = local;
                    }
                    auto d = unit(direction - n * direction.dot(n));
                    if (cv::norm(d) < 0.5 || d.dot(direction) < 0.5) d = direction;
                    direction = d;
                    p += direction * step;

                }
                xyz[size_t(middle + side * r) * cols + c] = p;
            }
        }
    }
    // Contract inner-bend rows before they can fold across the centerline.
    std::vector<double> radius(cols, std::numeric_limits<double>::infinity());
    std::vector<cv::Vec3d> curvatureDir(cols, cv::Vec3d{0, 0, 0});
    for (int c = 1; c + 1 < cols; ++c) {
        const auto dt = tangent[c + 1] - tangent[c - 1];
        const double angle = std::asin(std::clamp(cv::norm(tangent[c + 1].cross(tangent[c - 1])), 0.0, 1.0));
        if (angle > 1e-6) {
            radius[c] = (columns[c + 1] - columns[c - 1]) / angle;
            curvatureDir[c] = unit(dt);
        }
    }
    // Repair offsets that swing, run backwards, or collapse. Smooth more
    // widely near folds, retaining the contraction on the inside of bends.
    for (int r = 0; r < rows; ++r) {
        if (r == middle) continue;
        std::vector<cv::Vec3d> offset(cols);
        std::vector<char> bad(cols, 0);
        for (int c = 0; c < cols; ++c) offset[c] = xyz[size_t(r) * cols + c] - center[c];
        const double length = std::abs(r - middle) * spacing;
        std::vector<double> cap(cols, length);
        std::vector<cv::Vec3d> capValues(cols);
        for (int c = 0; c < cols; ++c) {
            if (curvatureDir[c].dot(unit(offset[c])) > 0.3 && std::isfinite(radius[c]))
                cap[c] = std::clamp(0.8 * radius[c], 0.35 * length, length);
            capValues[c] = {cap[c], 0, 0};
        }
        capValues = gaussianAlong(capValues, 2.0);
        for (int c = 0; c < cols; ++c) cap[c] = std::min(cap[c], capValues[c][0]);
        for (int c = 0; c < cols; ++c) {
            const auto o = unit(offset[c]);
            const auto prev = unit(offset[std::max(c - 1, 0)]);
            const auto next = unit(offset[std::min(c + 1, cols - 1)]);
            const bool swings = (c > 0 && o.dot(prev) < std::cos(45.0 * CV_PI / 180.0)) &&
                (c + 1 < cols && o.dot(next) < std::cos(45.0 * CV_PI / 180.0));
            const bool backwards = c + 1 < cols &&
                (xyz[size_t(r) * cols + c + 1] - xyz[size_t(r) * cols + c]).dot(tangent[c]) <= 0;
            bad[c] = swings || backwards || cv::norm(offset[c]) < 0.5 * length;
        }
        std::vector<char> folded = bad;
        if (std::count(bad.begin(), bad.end(), char(1)) == cols)
            for (int c = 0; c < cols; ++c) {
                offset[c] = cross[c] * double(r - middle) * spacing;
                bad[c] = folded[c] = 0;
            }
        interpolateBad(offset, bad, false);
        const auto narrow = gaussianAlong(offset, 1.0), wide = gaussianAlong(offset, 3.0);
        std::vector<char> nearFold(cols, 0);
        for (int c = 0; c < cols; ++c)
            if (folded[c])
                for (int k = std::max(0, c - 4); k <= std::min(cols - 1, c + 4); ++k) nearFold[k] = 1;
        for (int c = 0; c < cols; ++c) {
            const auto& smoothed = nearFold[c] ? wide[c] : narrow[c];
            const double target = cap[c] < length - 1e-9 ? std::clamp(cv::norm(smoothed), 0.35 * length, cap[c]) : length;
            xyz[size_t(r) * cols + c] = center[c] + unit(smoothed) * target;
        }
    }

    cv::Mat_<cv::Vec3f> points(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            points(r, c) = cv::Vec3f(xyz[size_t(r) * cols + c]);
    return std::make_shared<QuadSurface>(points, cv::Vec2f(1.0f, 1.0f));
}

// Indexed surface and area-weighted vertex normals, using the same
// diagonal as SurfacePatchIndex: (00,10,01) and (10,11,01).
struct Surface {
    std::shared_ptr<QuadSurface> quad;
    SurfacePatchIndex index;
    std::vector<cv::Vec3d> vertices, normals;
    cv::Vec3d low = cv::Vec3d::all(std::numeric_limits<double>::infinity());
    cv::Vec3d high = -low;

    explicit Surface(std::shared_ptr<QuadSurface> q) : quad(std::move(q))
    {
        quad->ensureLoaded();
        const auto& points = *quad->rawPointsPtr();
        if (points.rows < 2 || points.cols < 2)
            throw std::runtime_error("surface needs at least two rows and columns");
        std::vector<cv::Vec3d> gridNormals(points.total(), cv::Vec3d{0, 0, 0});
        for (int r = 0; r + 1 < points.rows; ++r)
            for (int c = 0; c + 1 < points.cols; ++c) {
                if (!validQuad(r, c)) continue;
                const size_t a = size_t(r) * points.cols + c, b = a + 1;
                const size_t d = a + points.cols, e = d + 1;
                const cv::Vec3d n0 = (cv::Vec3d(points(r, c + 1)) - cv::Vec3d(points(r, c))).cross(
                    cv::Vec3d(points(r + 1, c)) - cv::Vec3d(points(r, c)));
                const cv::Vec3d n1 = (cv::Vec3d(points(r + 1, c + 1)) - cv::Vec3d(points(r, c + 1))).cross(
                    cv::Vec3d(points(r + 1, c)) - cv::Vec3d(points(r, c + 1)));
                for (size_t i : {a, b, d}) gridNormals[i] += n0;
                for (size_t i : {b, e, d}) gridNormals[i] += n1;
            }
        for (int r = 0; r < points.rows; ++r)
            for (int c = 0; c < points.cols; ++c) {
                if (!valid(points(r, c))) continue;
                const cv::Vec3d p(points(r, c));
                vertices.push_back(p);
                normals.push_back(unit(gridNormals[size_t(r) * points.cols + c]));
                for (int d = 0; d < 3; ++d) {
                    low[d] = std::min(low[d], p[d]);
                    high[d] = std::max(high[d], p[d]);
                }
            }
        if (vertices.empty() || std::none_of(normals.begin(), normals.end(),
                                            [](const auto& n) { return cv::norm(n) > 0.5; }))
            throw std::runtime_error("surface has no usable triangles");
        index.setSamplingStride(1);
        index.rebuild({quad});
    }

    static bool valid(const cv::Vec3f& p)
    {
        return p[0] != -1 && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
    }
    bool validQuad(int r, int c) const
    {
        const auto& p = *quad->rawPointsPtr();
        return valid(p(r, c)) && valid(p(r, c + 1)) && valid(p(r + 1, c)) && valid(p(r + 1, c + 1));
    }
    bool inBox(const cv::Vec3d& p, double margin) const
    {
        for (int d = 0; d < 3; ++d)
            if (p[d] < low[d] - margin || p[d] > high[d] + margin) return false;
        return true;
    }
    struct Hit { double distance, normalDot; cv::Vec2d grid; };
    std::optional<Hit> nearest(const cv::Vec3d& p, const cv::Vec3d& normal, double tolerance) const
    {
        if (!inBox(p, tolerance)) return {};
        SurfacePatchIndex::PointQuery query;
        query.worldPoint = cv::Vec3f(p);
        query.tolerance = float(tolerance);
        const auto hit = index.locate(query);
        if (!hit) return {};
        const cv::Vec2d grid(quad->ptrToGrid(hit->ptr));
        const auto& points = *quad->rawPointsPtr();
        const int col = std::clamp(int(std::floor(grid[0])), 0, points.cols - 2);
        const int row = std::clamp(int(std::floor(grid[1])), 0, points.rows - 2);
        // A hit on an edge of a hole can round into the empty cell. Try
        // neighbouring cells that also contain that boundary point.
        for (int r = row; r >= std::max(0, row - 1); --r)
            for (int c = col; c >= std::max(0, col - 1); --c) {
                const double u = grid[0] - c, v = grid[1] - r;
                if (u < -1e-5 || v < -1e-5 || u > 1 + 1e-5 || v > 1 + 1e-5 || !validQuad(r, c)) continue;
                const cv::Vec3d a(points(r, c)), b(points(r, c + 1));
                const cv::Vec3d d(points(r + 1, c)), e(points(r + 1, c + 1));
                const auto n = unit(u + v <= 1 ? (b - a).cross(d - a) : (e - b).cross(d - b));
                return Hit{hit->distance, std::abs(n.dot(normal)), grid};
            }
        return {};
    }
};

struct Patch {
    std::filesystem::path path;
    std::string uuid;
    std::unique_ptr<Surface> surface;
};

struct Proposal {
    size_t patch = 0;
    std::vector<size_t> contacts;
    double contact = 0, stripSame = 0, stripAdjacent = 0, linkedSame = 0, linkedAdjacent = 0;
    bool anchor = false;
    int round = -1;
};

// Median sample spacing gives isolated contacts their own sample length.
double contactLength(const LineData& fiber, const std::vector<size_t>& contacts, double gap)
{
    std::vector<double> steps;
    for (size_t i = fiber.firstLine + 1; i <= fiber.lastLine; ++i)
        steps.push_back(fiber.arclength[i] - fiber.arclength[i - 1]);
    std::nth_element(steps.begin(), steps.begin() + steps.size() / 2, steps.end());
    const double step = steps[steps.size() / 2];
    double total = 0;
    size_t start = 0;
    for (size_t i = 1; i <= contacts.size(); ++i)
        if (i == contacts.size() || fiber.arclength[contacts[i]] - fiber.arclength[contacts[i - 1]] > gap) {
            total += fiber.arclength[contacts[i - 1]] - fiber.arclength[contacts[start]] + step;
            start = i;
        }
    return total;
}

bool sameSheet(double distance, double dot, const LinePatchParams& p)
{
    return distance <= p.sameSheet && dot > p.sameNormal;
}
bool adjacentSheet(double distance, double dot, const LinePatchParams& p)
{
    return distance >= p.adjacentLow && distance <= p.adjacentHigh && dot > p.adjacentNormal;
}

std::pair<double, double> stripAgreement(const Surface& strip, const Surface& patch,
                                        const LinePatchParams& p, const std::atomic_bool* cancel)
{
    const int rows = strip.quad->rawPointsPtr()->rows;
    double weight = 0, same = 0, adjacent = 0;
    size_t selected = 0;
    for (size_t i = 0; i < patch.vertices.size(); ++i) {
        if ((i & 1023) == 0) checkCancelled(cancel);
        if (!strip.inBox(patch.vertices[i], p.stripMargin)) continue;
        ++selected;
        const auto hit = strip.nearest(patch.vertices[i], patch.normals[i], p.adjacentHigh);
        const int rowDistance = hit ? std::abs(std::clamp(int(std::lround(hit->grid[1])), 0, rows - 1) - rows / 2)
                                    : int(p.stripWeights.size());
        const double w = rowDistance < int(p.stripWeights.size()) ? p.stripWeights[rowDistance] : 0.05;
        weight += w;
        if (!hit) continue;
        if (sameSheet(hit->distance, hit->normalDot, p)) same += w;
        else if (adjacentSheet(hit->distance, hit->normalDot, p)) adjacent += w;
    }
    if (selected < 5 || weight <= 0) return {0, 0};
    return {same / weight, adjacent / weight};
}

struct Relation {
    std::vector<double> distance, normalDot;
    explicit Relation(size_t n) : distance(n, std::numeric_limits<double>::infinity()), normalDot(n, 0) {}
    void update(const Surface& self, const Surface& other, const LinePatchParams& p,
                const std::atomic_bool* cancel)
    {
        for (int d = 0; d < 3; ++d)
            if (self.high[d] + p.stripMargin < other.low[d] || other.high[d] + p.stripMargin < self.low[d]) return;
        for (size_t i = 0; i < self.vertices.size(); ++i) {
            if ((i & 1023) == 0) checkCancelled(cancel);
            if (!other.inBox(self.vertices[i], p.stripMargin)) continue;
            const auto hit = other.nearest(self.vertices[i], self.normals[i], p.adjacentHigh);
            if (hit && hit->distance < distance[i]) {
                distance[i] = hit->distance;
                normalDot[i] = hit->normalDot;
            }
        }
    }
    std::pair<double, double> fractions(const LinePatchParams& p) const
    {
        size_t same = 0, adjacent = 0;
        for (size_t i = 0; i < distance.size(); ++i) {
            if (sameSheet(distance[i], normalDot[i], p)) ++same;
            else if (adjacentSheet(distance[i], normalDot[i], p)) ++adjacent;
        }
        if (distance.empty()) return {0, 0};
        return {double(same) / distance.size(), double(adjacent) / distance.size()};
    }
};

void autoLink(const LineData& fiber, std::vector<Proposal>& proposals, const std::vector<Patch>& patches,
              const SheetNormalSource& sampler, const LinePatchParams& p, const std::atomic_bool* cancel)
{
    if (proposals.empty()) return;
    const Surface strip(buildStrip(fiber, sampler, p, cancel));
    std::vector<size_t> fresh;
    std::vector<std::unique_ptr<Relation>> relations(proposals.size());
    for (size_t i = 0; i < proposals.size(); ++i) {
        auto& c = proposals[i];
        const auto& patch = *patches[c.patch].surface;
        c.contact = contactLength(fiber, c.contacts, p.contactGap);
        if (c.contact >= p.anchorContact) {
            std::tie(c.stripSame, c.stripAdjacent) = stripAgreement(strip, patch, p, cancel);
            c.anchor = c.stripSame > 0 && c.stripSame > p.stripGateRatio * c.stripAdjacent + p.stripGateMargin;
        }
        if (c.anchor) {
            c.round = 0;
            fresh.push_back(i);
        } else relations[i] = std::make_unique<Relation>(patch.vertices.size());
    }
    for (int round = 1; round <= p.maxRounds && !fresh.empty(); ++round) {
        checkCancelled(cancel);
        std::vector<size_t> pending;
        for (size_t i = 0; i < proposals.size(); ++i)
            if (proposals[i].round < 0) pending.push_back(i);
        const size_t width = std::min<size_t>(8, std::max(1u, std::thread::hardware_concurrency()));
        for (size_t first = 0; first < pending.size(); first += width) {
            std::vector<std::future<void>> jobs;
            for (size_t j = first; j < std::min(first + width, pending.size()); ++j)
                jobs.push_back(std::async(std::launch::async, [&, i = pending[j]] {
                    for (size_t f : fresh)
                        relations[i]->update(*patches[proposals[i].patch].surface,
                                             *patches[proposals[f].patch].surface, p, cancel);
                }));
            for (auto& job : jobs) job.get();
        }
        fresh.clear();
        for (size_t i : pending) {
            auto& c = proposals[i];
            std::tie(c.linkedSame, c.linkedAdjacent) = relations[i]->fractions(p);
            if (c.linkedSame >= p.growSame && c.linkedSame > p.growRatio * c.linkedAdjacent) {
                c.round = round;
                fresh.push_back(i);
            }
        }
    }
}

nlohmann::json settingsJson(const LinePatchParams& p)
{
    return {{"patch_scale", p.patchScale},
            {"working_to_base_scale", p.workingToBaseScale},
            {"ct_to_base_scale", p.ctToBaseScale}, {"ct_level", p.ctLevel},
            {"normals_manifest", p.normalsManifest},
            {"proximity", p.proximity}, {"min_contacts", p.minContacts},
            {"strip_spacing", p.stripSpacing}, {"strip_rows_each_side", p.stripRowsEachSide},
            {"strip_march_step", p.stripMarchStep}, {"contact_gap", p.contactGap}, {"anchor_contact", p.anchorContact},
            {"same_sheet", p.sameSheet}, {"adjacent_low", p.adjacentLow}, {"adjacent_high", p.adjacentHigh},
            {"same_normal", p.sameNormal}, {"adjacent_normal", p.adjacentNormal},
            {"grow_same", p.growSame}, {"grow_ratio", p.growRatio}, {"strip_weights", p.stripWeights},
            {"strip_gate_ratio", p.stripGateRatio}, {"strip_gate_margin", p.stripGateMargin},
            {"strip_margin", p.stripMargin}, {"max_rounds", p.maxRounds}};
}

void validateParams(const LinePatchParams& p)
{
    const auto settings = settingsJson(p);
    for (const auto& [name, value] : settings.items())
        if (value.is_number() && (!std::isfinite(value.get<double>()) || value.get<double>() < 0))
            throw std::invalid_argument("graph links: invalid " + name);
    if (p.patchScale <= 0 || p.workingToBaseScale <= 0 || p.ctToBaseScale <= 0 || p.ctLevel < 0 ||
        !std::isfinite(p.workingToBaseScale / p.ctToBaseScale) ||
        p.workingToBaseScale / p.ctToBaseScale <= 0 ||
        p.proximity <= 0 || p.minContacts < 2 || p.stripSpacing <= 0 || p.stripMarchStep <= 0 ||
        p.stripRowsEachSide < 1 || p.stripRowsEachSide > (std::numeric_limits<int>::max() - 1) / 2 ||
        p.stripSpacing / p.stripMarchStep > std::numeric_limits<int>::max() ||
        p.sameSheet <= 0 || p.adjacentLow <= p.sameSheet || p.adjacentHigh <= p.adjacentLow ||
        p.sameNormal > 1 || p.adjacentNormal > 1 || p.growSame > 1 || p.stripGateRatio <= 0 ||
        p.maxRounds < 1 || p.edgeKind.empty())
        throw std::invalid_argument("graph links: invalid fiber-patch parameters");
    for (double w : p.stripWeights)
        if (!std::isfinite(w) || w < 0) throw std::invalid_argument("graph links: invalid strip weight");
}

}  // namespace

void validateLinePatchParams(const LinePatchParams& params, double sourceScale)
{
    validateParams(params);
    if (!std::isfinite(sourceScale) || sourceScale <= 0)
        throw std::invalid_argument("graph links: invalid source coordinate scale");
}

DetectorReport linkLinesToPatches(graph::GraphDb& db, size_t lineCount, const LineLoader& load,
    const std::filesystem::path& patchesDir, const std::filesystem::path& ctVolume,
    const LinePatchParams& params, double sourceScale, const char* detector, int version,
    const std::atomic_bool* cancel)
{
    validateLinePatchParams(params, sourceScale);
    const bool tracks = std::string(detector) == kTrackPatchDetector;
    DetectorReport report;
    try {
        checkCancelled(cancel);
        const SheetNormalSource sampler(ctVolume, params);
        std::vector<std::filesystem::path> dirs;
        for (const auto& e : std::filesystem::directory_iterator(patchesDir))
            if (e.is_directory() && std::filesystem::exists(e.path() / "meta.json")) dirs.push_back(e.path());
        std::sort(dirs.begin(), dirs.end());
        std::set<std::string> identities;
        auto unique = [&](const std::string& uuid) {
            if (!identities.insert(uuid).second)
                throw std::runtime_error("graph links: duplicate input UUID " + uuid);
        };
        std::vector<Patch> patches;
        for (const auto& dir : dirs) {
            checkCancelled(cancel);
            std::shared_ptr<QuadSurface> quad;
            std::unique_ptr<Surface> surface;
            std::string uuid;
            try {
                quad = std::make_shared<QuadSurface>(dir);
                if (!vc::util::isUuid(quad->objectUuid())) throw std::runtime_error("missing/invalid UUID");
                uuid = quad->objectUuid();
                const auto scale = quad->scale();
                if (!(scale[0] > 0) || !(scale[1] > 0) || !std::isfinite(scale[0]) || !std::isfinite(scale[1]))
                    throw std::runtime_error("invalid patch scale");
                if (params.patchScale != 1.0) {
                    quad->ensureLoaded();
                    auto points = quad->rawPointsPtr()->clone();
                    for (auto& p : points) {
                        if (!Surface::valid(p)) continue; // preserve holes, including the -1 sentinel
                        const cv::Vec3d scaled = cv::Vec3d(p) * params.patchScale;
                        p = cv::Vec3f(scaled);
                        if (!Surface::valid(p)) throw std::runtime_error("scaled patch coordinate is invalid");
                    }
                    // Preserve grid coordinates/UV scale. Only world XYZ changes;
                    // use a detached surface so mapped source TIFFs stay untouched.
                    quad = std::make_shared<QuadSurface>(points, scale);
                }
                surface = std::make_unique<Surface>(quad);
            } catch (const std::exception& e) {
                report.problems.push_back(dir.filename().string() + ": " + e.what() + ", skipped");
                continue;
            }
            unique(uuid);
            patches.push_back({dir, uuid, std::move(surface)});
        }
        report.objectsConsidered = int64_t(patches.size());
        std::vector<graph::ObjectRecord> registrations;
        // One discovery index finds every patch touched by each dense line
        // point. locateAll returns one nearest hit per surface.
        SurfacePatchIndex discovery;
        discovery.setSamplingStride(1);
        std::vector<SurfacePatchIndex::SurfacePtr> surfaces;
        std::unordered_map<QuadSurface*, size_t> patchIndex;
        for (size_t i = 0; i < patches.size(); ++i) {
            surfaces.push_back(patches[i].surface->quad);
            patchIndex.emplace(surfaces.back().get(), i);
        }
        discovery.rebuild(surfaces);
        std::vector<LinkCandidate> candidates;
        StaleEdgeScope scope{{params.edgeKind}, {}};
        for (const auto& patch : patches) scope.objectUuids.push_back(patch.uuid);
        auto settings = settingsJson(params);
        settings[tracks ? "track_scale" : "fiber_scale"] = sourceScale;
        for (size_t input = 0; input < lineCount; ++input) {
            checkCancelled(cancel);
            auto loaded = load(input, report);
            if (!loaded) continue;
            const auto& fiber = *loaded;
            unique(fiber.uuid);
            ++report.objectsConsidered;
            registrations.push_back(fiber.object);
            std::map<size_t, std::vector<size_t>> contacts;
            for (size_t i = fiber.firstLine; i <= fiber.lastLine; ++i) {
                if ((i & 255) == 0) checkCancelled(cancel);
                SurfacePatchIndex::PointQuery q;
                q.worldPoint = cv::Vec3f(fiber.line[i]);
                q.tolerance = float(params.proximity);
                for (const auto& hit : discovery.locateAll(q))
                    contacts[patchIndex.at(hit.surface.get())].push_back(i);
            }
            std::vector<Proposal> proposals;
            for (auto& [patch, lines] : contacts)
                if (lines.size() >= params.minContacts) {
                    Proposal proposal;
                    proposal.patch = patch;
                    proposal.contacts = std::move(lines);
                    proposals.push_back(std::move(proposal));
                }
            try {
                autoLink(fiber, proposals, patches, sampler, params, cancel);
            } catch (const std::exception& e) {
                checkCancelled(cancel);
                report.problems.push_back(fiber.path.filename().string() + ": " + e.what() + ", skipped");
                continue;
            }
            if (std::getenv("VC_FIBER_PATCH_PROFILE")) {
                size_t eligible = 0, anchors = 0, linked = 0;
                double maxContact = 0, maxSame = 0, maxAdjacent = 0;
                for (const auto& proposal : proposals) {
                    eligible += proposal.contact >= params.anchorContact;
                    anchors += proposal.anchor;
                    linked += proposal.round >= 0;
                    maxContact = std::max(maxContact, proposal.contact);
                    maxSame = std::max(maxSame, proposal.stripSame);
                    maxAdjacent = std::max(maxAdjacent, proposal.stripAdjacent);
                }
                const nlohmann::json profile{{"fiber", fiber.path.filename().string()},
                    {"proposals", proposals.size()}, {"eligible", eligible}, {"anchors", anchors},
                    {"linked", linked}, {"max_contact", maxContact},
                    {"max_strip_same", maxSame}, {"max_strip_adjacent", maxAdjacent}};
                std::fprintf(stderr, "FIBER_PATCH_FIBER %s\n", profile.dump().c_str());
            }
            scope.objectUuids.push_back(fiber.uuid);
            for (const auto& p : proposals) {
                if (p.round < 0) continue;
                const auto& patch = patches[p.patch];
                const size_t median = p.contacts[p.contacts.size() / 2];
                const auto& contact = fiber.line[median];
                LinkCandidate c;
                c.aUuid = fiber.uuid;
                c.bUuid = patch.uuid;
                c.kind = params.edgeKind;
                c.aAnchor.index = int64_t(median);
                c.aAnchor.position = std::array<double, 3>{contact[0] / sourceScale,
                    contact[1] / sourceScale, contact[2] / sourceScale};
                SurfacePatchIndex::PointQuery q;
                q.worldPoint = cv::Vec3f(contact);
                q.tolerance = float(params.proximity);
                const auto hit = patch.surface->index.locate(q);
                nlohmann::json evidence{{"contacts", p.contacts.size()}, {"contact_length", p.contact},
                    {"anchor", p.anchor}, {"round", p.round}, {"strip_same", p.stripSame},
                    {"strip_adjacent", p.stripAdjacent}, {"linked_same", p.linkedSame},
                    {"linked_adjacent", p.linkedAdjacent}, {"median_line_index", median}};
                if (hit) {
                    const auto grid = patch.surface->quad->ptrToGrid(hit->ptr);
                    evidence["patch_grid"] = {grid[0], grid[1]};
                }
                c.score = p.anchor ? p.stripSame : p.linkedSame;
                auto metadata = nlohmann::json{{tracks ? "track_patch" : "fiber_patch", evidence}, {"settings", settings},
                    {"normal_source", params.normalsManifest.empty() ? ctVolume.string() : params.normalsManifest},
                    {"normal_method", sampler.method()},
                    {"pred_dt_snap", false},
                    {"anchor_index_space", "line_points"}, {"anchor_position_space", "source"}, {"measurement_space", "working_voxels"}};
                if (tracks) {
                    metadata["track_dataset_ref"] = fiber.object.track->datasetRef;
                    metadata["track_source_id"] = fiber.object.track->externalKey;
                }
                if (params.normalsManifest.empty()) metadata["ct_volume"] = ctVolume.string();
                else metadata["normals_manifest"] = params.normalsManifest;
                c.meta = metadata.dump();
                candidates.push_back(std::move(c));
            }
        }
        checkCancelled(cancel);
        // Geometry work is complete before any registration writes. Preserve
        // existing metadata and approval; UUID kind conflicts abort atomically.
        auto txn = db.write();
        auto registerObject = [&](graph::ObjectRecord rec) {
            if (const auto existing = db.objectByUuid(txn, rec.uuid)) {
                if (existing->kind != rec.kind) throw std::runtime_error("graph links: UUID kind conflict " + rec.uuid);
                if (rec.track) db.setTrackLength(txn, rec.uuid, rec.track->length, rec.track->family);
                if (rec.fiber) db.setFiberLength(txn, rec.uuid, rec.fiber->length);
                return;
            }
            db.upsertObject(txn, rec);
        };
        for (const auto& rec : registrations) registerObject(rec);
        for (const auto& patch : patches) {
            graph::ObjectRecord rec;
            rec.uuid = patch.uuid;
            rec.kind = "patch";
            graph::PatchDetail detail;
            const auto& s = *patch.surface;
            detail.gridWidth = s.quad->rawPointsPtr()->cols;
            detail.gridHeight = s.quad->rawPointsPtr()->rows;
            detail.scale = s.quad->scale()[0];
            detail.bbox = std::array<double, 6>{s.low[0] / params.patchScale, s.low[1] / params.patchScale,
                s.low[2] / params.patchScale, s.high[0] / params.patchScale,
                s.high[1] / params.patchScale, s.high[2] / params.patchScale};
            rec.patch = detail;
            rec.name = patch.path.filename().string();
            rec.storageKind = "file";
            rec.storageRef = std::filesystem::absolute(patch.path).lexically_relative(
                std::filesystem::absolute(db.path().parent_path())).string();
            rec.dataGeneration = 1;
            registerObject(rec);
        }
        checkCancelled(cancel);
        const auto committed = commitCandidates(db, txn, detector, version, candidates, params.replaceStalePending, &scope);
        checkCancelled(cancel);
        txn.commit();
        report.candidatesFound = committed.candidatesFound;
        report.edgesCreated = committed.edgesCreated;
        report.edgesSkippedExisting = committed.edgesSkippedExisting;
        report.staleRemoved = committed.staleRemoved;
    } catch (const std::exception&) {
        if (!cancel || !cancel->load()) throw;
        report.problems.push_back(std::string(detector) + ": cancelled; no objects or edges committed");
    }
    return report;
}

}  // namespace vc::graph_links::detail
