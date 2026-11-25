#include "vc_object_store/objects/FiberApprovalExtension.hpp"

#include "vc_object_store/objects/StoreReader.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace vc::objects {
namespace {
using Type = ExtensionType;
ExtensionColumn required(const char* name, Type type) { return {name, type, false}; }

struct Hasher {
    uint64_t state = UINT64_C(14695981039346656037);
    void byte(uint8_t b) { state ^= b; state *= UINT64_C(1099511628211); }
    void f64(double v)
    {
        const auto bits = std::bit_cast<uint64_t>(v == 0.0 ? 0.0 : v); // -0 == +0
        for (int i = 0; i < 8; ++i) byte(uint8_t(bits >> (8 * i)));
    }
};

int64_t integer(const ExtensionValue& value, const char* what)
{
    if (const auto* v = std::get_if<int64_t>(&value)) return *v;
    throw std::runtime_error(std::string("fiber approval: column is not an integer: ") + what);
}
double real(const ExtensionValue& value, const char* what)
{
    if (const auto* v = std::get_if<double>(&value)) return *v;
    if (const auto* v = std::get_if<int64_t>(&value)) return double(*v);
    throw std::runtime_error(std::string("fiber approval: column is not a number: ") + what);
}
std::string text(const ExtensionValue& value, const char* what)
{
    if (const auto* v = std::get_if<std::string>(&value)) return *v;
    throw std::runtime_error(std::string("fiber approval: column is not text: ") + what);
}

bool usable(const geometry::Point& normal)
{
    const double length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    return std::isfinite(length) && length > 1e-9;
}

template <typename Visit>
bool scan(const StoreReader& reader, const char* table, std::span<const ExtensionValue> prefix, Visit&& visit)
{
    if (!reader.status().extensionsPresent) return false;
    try {
        reader.scanExtensionRows(table, [&](const ExtensionRow& row) { visit(row); return true; }, prefix);
    } catch (const std::invalid_argument&) {
        return false; // no such table in this store: no layers
    }
    return true;
}
} // namespace

std::vector<ExtensionTable> fiberApprovalTables()
{
    ExtensionTable header;
    header.name = kFiberApprovalTable;
    header.extension = kFiberApprovalExtension;
    header.formatVersion = kFiberApprovalFormat;
    header.provisioning =
        R"({"line_normals":"3 float32 little-endian per stored line point","positions":"stored line point index"})";
    header.keySpace = "objects";
    header.columns = {required("fiber_uuid", Type::Blob), required("generation", Type::Integer),
                      required("fiber_version", Type::Integer), required("baseline_run", Type::Text),
                      required("author", Type::Text), required("saved_at", Type::Text), required("host", Type::Text),
                      required("line_normals", Type::Blob)};
    header.primaryKey = {"fiber_uuid", "generation"};

    ExtensionTable stretch;
    stretch.name = kFiberApprovalStretchTable;
    stretch.extension = kFiberApprovalExtension;
    stretch.formatVersion = kFiberApprovalFormat;
    stretch.provisioning =
        R"({"aspects":{"geometry":0,"normals":1},"states":{"approved":1,"broken":2},"fingerprint":"fnv1a64-v1"})";
    stretch.keySpace = "objects";
    stretch.columns = {required("fiber_uuid", Type::Blob), required("generation", Type::Integer),
                       required("item", Type::Integer), required("aspect", Type::Integer),
                       required("state", Type::Integer), required("from_position", Type::Real),
                       required("to_position", Type::Real), required("fingerprint", Type::Integer)};
    stretch.primaryKey = {"fiber_uuid", "generation", "item"};

    ExtensionTable point;
    point.name = kFiberApprovalNormalPointTable;
    point.extension = kFiberApprovalExtension;
    point.formatVersion = kFiberApprovalFormat;
    point.provisioning = R"({"normal":"axis, facing not stated","fingerprint":"fnv1a64-v1"})";
    point.keySpace = "objects";
    point.columns = {required("fiber_uuid", Type::Blob), required("generation", Type::Integer),
                     required("item", Type::Integer), required("position", Type::Real),
                     required("normal_x", Type::Real), required("normal_y", Type::Real),
                     required("normal_z", Type::Real), required("fingerprint", Type::Integer)};
    point.primaryKey = {"fiber_uuid", "generation", "item"};
    return {header, stretch, point};
}

FiberApprovalRows fiberApprovalRows(const FiberApprovalLayer& layer)
{
    FiberApprovalRows rows;
    std::vector<uint8_t> normals(layer.lineNormals.size() * 3 * sizeof(float));
    for (size_t i = 0; i < layer.lineNormals.size(); ++i)
        for (size_t k = 0; k < 3; ++k) {
            const float value = float(layer.lineNormals[i][k]);
            std::memcpy(normals.data() + (3 * i + k) * sizeof(float), &value, sizeof(float));
        }
    const auto fiber = extensionUuid(layer.fiberUuid);
    rows.header = {fiber, layer.generation, layer.fiberVersion, layer.baselineRun, layer.author, layer.savedAt,
                   layer.host, std::move(normals)};
    for (size_t i = 0; i < layer.stretches.size(); ++i) {
        const auto& s = layer.stretches[i];
        rows.stretches.push_back({fiber, layer.generation, int64_t(i), int64_t(s.aspect), int64_t(s.state), s.from,
                                  s.to, std::bit_cast<int64_t>(s.fingerprint)});
    }
    for (size_t i = 0; i < layer.normalPoints.size(); ++i) {
        const auto& p = layer.normalPoints[i];
        rows.normalPoints.push_back({fiber, layer.generation, int64_t(i), p.position, p.normal[0], p.normal[1],
                                     p.normal[2], std::bit_cast<int64_t>(p.fingerprint)});
    }
    return rows;
}

uint64_t fiberLineFingerprint(std::span<const geometry::Point> line, double from, double to)
{
    Hasher hash;
    if (line.empty() || !(to >= from)) return hash.state;
    const double last = double(line.size() - 1);
    const auto first = size_t(std::clamp(std::floor(from), 0.0, last));
    const auto end = size_t(std::clamp(std::ceil(to), 0.0, last));
    for (size_t i = first; i <= end; ++i)
        for (const double c : line[i]) hash.f64(c);
    return hash.state;
}

void setFiberStretchState(std::vector<FiberStretch>& stretches, FiberApprovalAspect aspect, double from, double to,
                          FiberStretchState state, std::span<const geometry::Point> line)
{
    if (!std::isfinite(from) || !std::isfinite(to) || !(to > from))
        throw std::invalid_argument("fiber approval: a stretch needs a start before its end");
    if (line.size() < 2) throw std::invalid_argument("fiber approval: the fiber has no line");
    from = std::max(from, 0.0);
    to = std::min(to, double(line.size() - 1));
    if (!(to > from)) throw std::invalid_argument("fiber approval: the stretch lies outside the line");
    std::vector<FiberStretch> out;
    const auto keep = [&](FiberStretch part) {
        if (!(part.to > part.from)) return;
        part.fingerprint = fiberLineFingerprint(line, part.from, part.to);
        out.push_back(part);
    };
    for (const auto& old : stretches) {
        if (old.aspect != aspect || old.to <= from || old.from >= to) { out.push_back(old); continue; }
        auto left = old, right = old;
        left.to = from;
        right.from = to;
        if (old.from < from) keep(left);
        if (old.to > to) keep(right);
    }
    if (state != FiberStretchState::Undefined) {
        // One stretch per run of a state: it joins its neighbours of that state.
        FiberStretch added{aspect, state, from, to, 0};
        for (auto it = out.begin(); it != out.end();) {
            const bool touches = it->aspect == aspect && it->state == state &&
                                 (std::abs(it->to - added.from) < 1e-9 || std::abs(it->from - added.to) < 1e-9);
            if (!touches) { ++it; continue; }
            added.from = std::min(added.from, it->from);
            added.to = std::max(added.to, it->to);
            it = out.erase(it);
        }
        keep(added);
    }
    std::sort(out.begin(), out.end(), [](const FiberStretch& a, const FiberStretch& b) {
        return a.aspect != b.aspect ? a.aspect < b.aspect : a.from < b.from;
    });
    stretches = std::move(out);
}

FiberStretchState fiberStretchStateAt(std::span<const FiberStretch> stretches, FiberApprovalAspect aspect,
                                      double position)
{
    for (const auto& s : stretches)
        if (s.aspect == aspect && position >= s.from && position <= s.to) return s.state;
    return FiberStretchState::Undefined;
}

size_t setFiberNormalPoint(std::vector<FiberNormalPoint>& points, double position, const geometry::Point& normal,
                           std::span<const geometry::Point> line, double tolerance)
{
    if (line.size() < 2) throw std::invalid_argument("fiber approval: the fiber has no line");
    if (!usable(normal)) throw std::invalid_argument("fiber approval: a normal point needs a normal");
    position = std::clamp(position, 0.0, double(line.size() - 1));
    FiberNormalPoint point{position, normal, fiberLineFingerprint(line, position, position)};
    const double length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    for (double& c : point.normal) c /= length;
    for (auto& old : points)
        if (std::abs(old.position - position) <= tolerance) {
            // At the old point's place, fingerprinted over the line as it is now.
            point.position = old.position;
            point.fingerprint = fiberLineFingerprint(line, point.position, point.position);
            old = point;
            return size_t(&old - points.data());
        }
    const auto at = std::lower_bound(points.begin(), points.end(), position,
                                     [](const FiberNormalPoint& p, double v) { return p.position < v; });
    const auto index = size_t(at - points.begin());
    points.insert(at, point);
    return index;
}

FiberApprovalStatus fiberApprovalStatus(const FiberApprovalLayer& layer, std::span<const geometry::Point> line)
{
    FiberApprovalStatus status;
    const double last = line.empty() ? -1.0 : double(line.size() - 1);
    for (const auto& s : layer.stretches) {
        const bool current = s.to <= last && fiberLineFingerprint(line, s.from, s.to) == s.fingerprint;
        status.stretchCurrent.push_back(current);
        if (!current) ++status.staleStretches;
    }
    for (const auto& p : layer.normalPoints) {
        const bool current = p.position <= last && fiberLineFingerprint(line, p.position, p.position) == p.fingerprint;
        status.normalPointCurrent.push_back(current);
        if (!current) ++status.staleNormalPoints;
    }
    status.lineNormalsUsable = !line.empty() && layer.lineNormals.size() == line.size();
    return status;
}

std::vector<geometry::Point> approvedFiberLineNormals(const FiberApprovalLayer& layer,
                                                      const FiberApprovalStatus& status, size_t linePoints)
{
    if (!status.lineNormalsUsable || layer.lineNormals.size() != linePoints || linePoints == 0) return {};
    std::vector<geometry::Point> out(linePoints, geometry::Point{0, 0, 0});
    bool any = false;
    const auto take = [&](size_t index) {
        if (!usable(layer.lineNormals[index])) return;
        out[index] = layer.lineNormals[index];
        any = true;
    };
    for (size_t i = 0; i < layer.stretches.size(); ++i) {
        const auto& s = layer.stretches[i];
        if (!status.stretchCurrent[i] || s.aspect != FiberApprovalAspect::Normals ||
            s.state != FiberStretchState::Approved) continue;
        const auto first = size_t(std::ceil(s.from - 1e-9)), end = size_t(std::floor(s.to + 1e-9));
        for (size_t j = first; j <= end && j < linePoints; ++j) take(j);
    }
    for (size_t i = 0; i < layer.normalPoints.size(); ++i) {
        if (!status.normalPointCurrent[i]) continue;
        // The wanted normal itself, at the line point nearest the point.
        const auto index = std::min(size_t(std::llround(layer.normalPoints[i].position)), linePoints - 1);
        bool broken = false;
        for (size_t k = 0; k < layer.stretches.size(); ++k) {
            const auto& stretch = layer.stretches[k];
            broken = broken || (status.stretchCurrent[k] && stretch.aspect == FiberApprovalAspect::Normals &&
                                stretch.state == FiberStretchState::Broken &&
                                layer.normalPoints[i].position >= stretch.from &&
                                layer.normalPoints[i].position <= stretch.to);
        }
        if (broken) continue;
        out[index] = layer.normalPoints[i].normal;
        any = true;
    }
    if (!any) out.clear();
    return out;
}

std::vector<std::pair<double, double>> fiberStretchRanges(const FiberApprovalLayer& layer,
                                                          const FiberApprovalStatus& status,
                                                          FiberApprovalAspect aspect, FiberStretchState state)
{
    std::vector<std::pair<double, double>> ranges;
    for (size_t i = 0; i < layer.stretches.size(); ++i)
        if (status.stretchCurrent[i] && layer.stretches[i].aspect == aspect && layer.stretches[i].state == state)
            ranges.emplace_back(layer.stretches[i].from, layer.stretches[i].to);
    return ranges;
}

std::optional<FiberApprovalLayer> readFiberApproval(const StoreReader& reader, const std::string& fiberUuid)
{
    const std::array<ExtensionValue, 1> fiber{extensionUuid(fiberUuid)};
    std::optional<FiberApprovalLayer> layer;
    // Primary-key order: the last header row is the newest generation.
    if (!scan(reader, kFiberApprovalTable, fiber, [&](const ExtensionRow& row) {
            if (row.size() != 8) throw std::runtime_error("fiber approval: header row has the wrong width");
            FiberApprovalLayer read;
            read.fiberUuid = fiberUuid;
            read.generation = integer(row[1], "generation");
            if (layer && read.generation <= layer->generation) return;
            read.fiberVersion = integer(row[2], "fiber_version");
            read.baselineRun = text(row[3], "baseline_run");
            read.author = text(row[4], "author");
            read.savedAt = text(row[5], "saved_at");
            read.host = text(row[6], "host");
            const auto* blob = std::get_if<std::vector<uint8_t>>(&row[7]);
            if (!blob || blob->size() % (3 * sizeof(float)) != 0)
                throw std::runtime_error("fiber approval: line normals blob has the wrong size");
            read.lineNormals.resize(blob->size() / (3 * sizeof(float)));
            for (size_t i = 0; i < read.lineNormals.size(); ++i)
                for (size_t k = 0; k < 3; ++k) {
                    float value = 0;
                    std::memcpy(&value, blob->data() + (3 * i + k) * sizeof(float), sizeof(float));
                    read.lineNormals[i][k] = double(value);
                }
            layer = std::move(read);
        }) || !layer)
        return std::nullopt;
    const std::array<ExtensionValue, 2> generation{extensionUuid(fiberUuid), layer->generation};
    scan(reader, kFiberApprovalStretchTable, generation, [&](const ExtensionRow& row) {
        if (row.size() != 8) throw std::runtime_error("fiber approval: stretch row has the wrong width");
        if (integer(row[1], "generation") != layer->generation) return;
        layer->stretches.push_back({FiberApprovalAspect(integer(row[3], "aspect")),
                                    FiberStretchState(integer(row[4], "state")), real(row[5], "from_position"),
                                    real(row[6], "to_position"),
                                    std::bit_cast<uint64_t>(integer(row[7], "fingerprint"))});
    });
    scan(reader, kFiberApprovalNormalPointTable, generation, [&](const ExtensionRow& row) {
        if (row.size() != 8) throw std::runtime_error("fiber approval: normal point row has the wrong width");
        if (integer(row[1], "generation") != layer->generation) return;
        layer->normalPoints.push_back({real(row[3], "position"),
                                       {real(row[4], "normal_x"), real(row[5], "normal_y"), real(row[6], "normal_z")},
                                       std::bit_cast<uint64_t>(integer(row[7], "fingerprint"))});
    });
    return layer;
}

std::map<std::string, int64_t> fiberApprovalGenerations(const StoreReader& reader)
{
    std::map<std::string, int64_t> newest;
    scan(reader, kFiberApprovalTable, {}, [&](const ExtensionRow& row) {
        if (row.size() != 8) throw std::runtime_error("fiber approval: header row has the wrong width");
        auto& generation = newest[extensionUuid(row[0])];
        generation = std::max(generation, integer(row[1], "generation"));
    });
    return newest;
}
} // namespace vc::objects
