#pragma once
#include "vc_object_store/objects/Extensions.hpp"
#include <nlohmann/json_fwd.hpp>

namespace vc::objects {
inline constexpr const char* kFiberGraphSolveExtension="ext_fiber_graph_solve";
inline constexpr const char* kFiberSourceTable="ext_fiber_graph_solve_sources";
inline constexpr const char* kFiberConstraintTable="ext_fiber_graph_solve_constraints";
inline constexpr int kFiberConstraintFormat=4;
inline constexpr double kMaximumCompactWindingDistance=254./16.;
std::vector<uint8_t> encodeFiberArc(double);
double decodeFiberArc(std::span<const uint8_t>);
std::vector<uint8_t> encodeFiberDirection(const std::array<double,3>&);
std::array<double,3> decodeFiberDirection(std::span<const uint8_t>);
// nullopt means the constraint must be skipped, not saturated to the maximum.
std::optional<uint8_t> encodeFiberWindingDistance(double);
double decodeFiberWindingDistance(std::span<const uint8_t>);
// Deliberately compact typed provisioning. Do not accept full source metadata,
// paths, reports or ID arrays and then parse/filter/copy them into the store.
struct FiberSourceInfo {
    std::string uuid; // empty when the source supplies no identity
    std::optional<std::string> sha256;
    std::optional<int64_t> fiberCount, pointCount;
};
struct FiberConstraintSourceInfo {
    // The AFC format supplies no independent file UUID. This identifies its
    // input fibers only; sha256, when known, identifies the constraint file.
    std::string fiberSourceUuid;
    std::optional<std::string> sha256;
    std::optional<int64_t> constraintCount;
    // Arc lengths and directions use this SAMPLE/VOLUME base-voxel frame.
    std::optional<std::string> coordinateSpace;
};
// Ordinary optional extension, not part of the core fiber/edge schema. Source
// AFV/AFC identity has no vcobj commit: record only the supplied identity and
// counts, not the source report. Referenced vcobj input pins may be supplied.
ExtensionTable fiberSourceExtension(const FiberSourceInfo& source);
ExtensionTable fiberConstraintExtension(const FiberConstraintSourceInfo& source,
    std::span<const ExtensionSource> basedOn = {});
InitialExtensionRow fiberSourceRow(const std::string& uuid,int64_t sourceId,const std::string& annotation);

// Native streaming sources already have typed measurements. Do not turn them
// into JSON only to parse them back while preparing a bulk import.
struct FiberConstraintNormal {
    std::optional<int64_t> valid;
    std::array<std::optional<double>,3> xyz;
};
struct FiberConstraintData {
    std::array<std::optional<double>,7> values; // arcs, scores, distances, evidence
    std::array<FiberConstraintNormal,2> normals;
    std::optional<std::string> extra;
    std::optional<std::array<std::array<double,3>,2>> measuredPoints;
};
InitialExtensionRow fiberConstraintRow(const std::string& edgeUuid,int64_t sourceId,
    int64_t sourceFiberA,int64_t sourceFiberB,bool sourceReversed,FiberConstraintData data);

// The input measurement's endpoint order is retained. source_reversed says it
// is opposite to the core edge's canonical order. Signed scalars are NEVER
// guessed/flipped merely because canonical UUID order changed.
InitialExtensionRow fiberConstraintRow(const std::string& edgeUuid,int64_t sourceId,
    int64_t sourceFiberA,int64_t sourceFiberB,bool sourceReversed,
    const std::array<std::optional<double>,7>& values, // arcA/B, scores, distances, evidence
    const nlohmann::json& normalA,const nlohmann::json& normalB,
    const nlohmann::json& measurement,
    const std::optional<std::array<std::array<double,3>,2>>& measuredPoints = {});
} // namespace vc::objects
