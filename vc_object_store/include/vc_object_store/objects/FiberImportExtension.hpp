#pragma once
#include "vc_object_store/objects/Extensions.hpp"
#include <nlohmann/json_fwd.hpp>

namespace vc::objects {
inline constexpr const char* kFiberGraphSolveExtension="ext_fiber_graph_solve";
inline constexpr const char* kFiberSourceTable="ext_fiber_graph_solve_sources";
inline constexpr const char* kFiberConstraintTable="ext_fiber_graph_solve_constraints";
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
};
// Ordinary optional extension, not part of the core fiber/edge schema. Source
// AFV/AFC identity has no vcobj commit: record only the supplied identity and
// counts, not the source report. Linked vcobj input pins may be supplied.
ExtensionTable fiberSourceExtension(const FiberSourceInfo& source);
ExtensionTable fiberConstraintExtension(const FiberConstraintSourceInfo& source,
    std::span<const ExtensionSource> basedOn = {});
InitialExtensionRow fiberSourceRow(const std::string& uuid,int64_t sourceId,const std::string& annotation);

// The input measurement's endpoint order is retained. source_reversed says it
// is opposite to the core edge's canonical order. Signed scalars are NEVER
// guessed/flipped merely because canonical UUID order changed.
InitialExtensionRow fiberConstraintRow(const std::string& edgeUuid,int64_t sourceId,
    int64_t sourceFiberA,int64_t sourceFiberB,bool sourceReversed,
    const std::array<std::optional<double>,7>& values, // arcA/B, scores, distances, evidence
    const nlohmann::json& normalA,const nlohmann::json& normalB,
    const nlohmann::json& measurement);
} // namespace vc::objects
