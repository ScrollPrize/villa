#pragma once
#include "ObjectStore.hpp"

namespace vc::objects {
// Shared geometry-fitting input policy. This is consumer eligibility, not
// identity/routing: callers retain the selected relationship's full reference.
struct RelationshipInputDecision {
    std::string family, exclusion;
    bool reversed = false;
};
RelationshipInputDecision fittingRelationshipInput(const EdgeRecord&, bool applicable,
    const std::map<std::string, std::string>& endpointKinds, bool approvedOnly);
}
