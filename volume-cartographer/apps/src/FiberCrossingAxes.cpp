#include "FiberCrossingAxes.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include <vc_object_store/objects/FiberConstraintGraph.hpp>
#include <opencv2/core/utility.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace {
constexpr char kMagic[8] = {'V', 'C', 'X', 'A', 'X', 'E', 'S', '1'};
constexpr double kTangentHalfSpan = 16.;   // base voxels either side of the event arc
constexpr double kMinimumCrossingSine = .5;  // keep crossings of at least 30 degrees

struct PointTangent { cv::Vec3d point, tangent; };
PointTangent at(const vc::fiber_tracer::PolylineArcGeometry& g, double arc)
{
    const double length = g.length();
    const cv::Vec3d a = vc::fiber_tracer::samplePolylineArc(g, std::max(0., arc - kTangentHalfSpan)).point;
    const cv::Vec3d b = vc::fiber_tracer::samplePolylineArc(g, std::min(length, arc + kTangentHalfSpan)).point;
    cv::Vec3d t = b - a;
    const double n = cv::norm(t);
    return {vc::fiber_tracer::samplePolylineArc(g, std::clamp(arc, 0., length)).point, n > 0. ? t / n : t};
}
}  // namespace

void writeFiberCrossingAxes(const std::filesystem::path& output, const std::string& crossingStore,
    const std::vector<std::string>& ids, const std::vector<std::vector<cv::Vec3d>>& lines, size_t workers)
{
    if (ids.size() != lines.size()) throw std::invalid_argument("crossing axes: ids and lines differ in size");
    vc::objects::OpenOptions open;
    open.readOnly = true;
    auto store = vc::objects::ObjectCollection::open(crossingStore, open);
    const auto txn = store.read();
    const auto columns = vc::objects::readFiberGraphColumns(store, txn, workers);
    std::unordered_map<std::string, size_t> index;
    for (size_t i = 0; i < ids.size(); ++i) index.emplace(ids[i], i);
    std::vector<int64_t> fiber(columns.nodes, -1);
    for (size_t v = 0; v < columns.nodes; ++v) {
        const auto found = index.find(vc::objects::fiberGraphUuid(columns.fiberUuids[v]));
        if (found != index.end()) fiber[v] = int64_t(found->second);
    }
    std::vector<vc::fiber_tracer::PolylineArcGeometry> geometry(lines.size());
    cv::parallel_for_(cv::Range(0, int(lines.size())), [&](const cv::Range& r) {
        for (int i = r.start; i < r.end; ++i)
            if (lines[size_t(i)].size() >= 2) geometry[size_t(i)] = vc::fiber_tracer::makePolylineArcGeometry(lines[size_t(i)]);
    }, int(workers));
    // Both sides of every perpendicular event, grouped by own fiber.
    std::vector<std::array<size_t, 2>> sides;  // (event, side)
    std::vector<size_t> count(lines.size() + 1, 0);
    for (size_t e = 0; e < columns.events; ++e) {
        if (columns.parallel[e]) continue;
        const auto [a, b] = columns.endpoints[e];
        if (fiber[size_t(a)] < 0 || fiber[size_t(b)] < 0)
            throw std::runtime_error("crossing axes: crossing store references a fiber that was not loaded");
        ++count[size_t(fiber[size_t(a)]) + 1];
        ++count[size_t(fiber[size_t(b)]) + 1];
    }
    for (size_t i = 0; i < lines.size(); ++i) count[i + 1] += count[i];
    sides.resize(count.back());
    {
        std::vector<size_t> fill(count.begin(), count.end() - 1);
        for (size_t e = 0; e < columns.events; ++e) {
            if (columns.parallel[e]) continue;
            const auto [a, b] = columns.endpoints[e];
            sides[fill[size_t(fiber[size_t(a)])]++] = {e, 0};
            sides[fill[size_t(fiber[size_t(b)])]++] = {e, 1};
        }
    }
    std::vector<std::vector<FiberCrossingAxis>> axes(lines.size());
    cv::parallel_for_(cv::Range(0, int(lines.size())), [&](const cv::Range& r) {
        for (int fi = r.start; fi < r.end; ++fi) {
            const size_t f = size_t(fi);
            auto& out = axes[f];
            for (size_t k = count[f]; k < count[f + 1]; ++k) {
                const auto [e, side] = sides[k];
                const auto ends = columns.endpoints[e];
                const size_t partner = size_t(fiber[size_t(ends[1 - side])]);
                const double ownArc = columns.arcs[e][side], partnerArc = columns.arcs[e][1 - side];
                if (geometry[f].points.empty() || geometry[partner].points.empty()) continue;
                const auto own = at(geometry[f], ownArc), other = at(geometry[partner], partnerArc);
                cv::Vec3d x = own.tangent.cross(other.tangent);
                const double s = cv::norm(x);
                if (!(s > kMinimumCrossingSine)) continue;
                FiberCrossingAxis c;
                c.arc = ownArc; c.axis = x / s; c.offset = other.point - own.point; c.gap = cv::norm(c.offset);
                c.ownTangent = own.tangent;
                out.push_back(c);
            }
            std::sort(out.begin(), out.end(), [](const auto& p, const auto& q) { return p.arc < q.arc; });
        }
    }, int(workers));
    std::ofstream file(output, std::ios::binary);
    if (!file) throw std::runtime_error("crossing axes: cannot write " + output.string());
    const auto put = [&](const void* data, size_t bytes) { file.write(static_cast<const char*>(data), std::streamsize(bytes)); };
    put(kMagic, sizeof kMagic);
    uint64_t listed = 0;
    for (const auto& list : axes) listed += !list.empty();
    put(&listed, sizeof listed);
    for (size_t f = 0; f < lines.size(); ++f) {
        if (axes[f].empty()) continue;
        const auto uuid = vc::objects::fiberGraphUuid(ids[f]);
        put(uuid.data(), uuid.size());
        const uint64_t n = axes[f].size();
        put(&n, sizeof n);
        for (const auto& c : axes[f]) {
            const std::array<double, 11> v{c.arc, c.axis[0], c.axis[1], c.axis[2], c.gap, c.offset[0], c.offset[1],
                                           c.offset[2], c.ownTangent[0], c.ownTangent[1], c.ownTangent[2]};
            put(v.data(), sizeof v);
        }
    }
    if (!file) throw std::runtime_error("crossing axes: write failed " + output.string());
}

std::vector<std::vector<FiberCrossingAxis>> readFiberCrossingAxes(const std::filesystem::path& path,
    const std::vector<std::string>& ids, size_t& unknownFibers)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("crossing axes: cannot read " + path.string());
    const auto get = [&](void* data, size_t bytes) {
        if (!file.read(static_cast<char*>(data), std::streamsize(bytes))) throw std::runtime_error("crossing axes: truncated " + path.string());
    };
    char magic[8];
    get(magic, sizeof magic);
    if (std::memcmp(magic, kMagic, sizeof magic) != 0) throw std::runtime_error("crossing axes: not a crossing axes file " + path.string());
    std::map<vc::objects::FiberGraphUuid, size_t> index;
    for (size_t i = 0; i < ids.size(); ++i) index.emplace(vc::objects::fiberGraphUuid(ids[i]), i);
    std::vector<std::vector<FiberCrossingAxis>> axes(ids.size());
    uint64_t listed = 0;
    get(&listed, sizeof listed);
    unknownFibers = 0;
    std::vector<std::array<double, 11>> buffer;
    for (uint64_t k = 0; k < listed; ++k) {
        vc::objects::FiberGraphUuid uuid;
        get(uuid.data(), uuid.size());
        uint64_t n = 0;
        get(&n, sizeof n);
        buffer.resize(n);
        get(buffer.data(), n * sizeof(buffer[0]));
        const auto found = index.find(uuid);
        if (found == index.end()) { ++unknownFibers; continue; }
        auto& out = axes[found->second];
        out.resize(n);
        for (size_t j = 0; j < n; ++j) {
            const auto& v = buffer[j];
            out[j] = {v[0], {v[1], v[2], v[3]}, v[4], {v[5], v[6], v[7]}, {v[8], v[9], v[10]}};
        }
    }
    return axes;
}
