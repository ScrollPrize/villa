#include "vc_object_store/objects/RelationshipInputs.hpp"
#include <array>
#include <algorithm>

namespace vc::objects {
RelationshipInputDecision fittingRelationshipInput(const EdgeRecord& edge, bool applicable,
    const std::map<std::string, std::string>& kinds, bool approvedOnly)
{
    RelationshipInputDecision result;
    const auto a = kinds.find(edge.aUuid), b = kinds.find(edge.bUuid);
    if (a == kinds.end() || b == kinds.end()) result.exclusion = "endpoint outside selection";
    else if (!applicable) result.exclusion = "inapplicable";
    else {
        constexpr std::array order{"surface", "point_collection", "fiber", "track"};
        const auto rank = [&](const auto& kind) { return std::find(order.begin(), order.end(), kind) - order.begin(); };
        result.reversed = std::pair(rank(b->second), edge.bUuid) < std::pair(rank(a->second), edge.aUuid);
        const auto& first = result.reversed ? b->second : a->second;
        const auto& second = result.reversed ? a->second : b->second;
        struct Rule { const char* kind; const char* a; const char* b; const char* family; };
        constexpr Rule rules[] = {
            {"point-surface", "surface", "point_collection", "pcl_patch"},
            {"same-surface", "surface", "fiber", "fiber_patch"},
            {"same-surface", "surface", "surface", "patch_overlap"},
            {"fiber_branch", "fiber", "fiber", "fiber_branch"},
            {"crossing", "track", "track", "track_crossing"},
        };
        for (const auto& rule : rules)
            if (edge.kind == rule.kind && first == rule.a && second == rule.b) result.family = rule.family;
        if (result.family.empty()) result.exclusion = "unused kind " + edge.kind;
        else if (edge.approvalState == "rejected") result.exclusion = "rejected";
        else if (approvedOnly && edge.approvalState != "approved") result.exclusion = edge.approvalState + " (approved-only)";
    }
    return result;
}
}
