#pragma once
#include "vc_object_store/objects/FiberJson.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include <array>

namespace vc::objects {
struct FiberEditControl {
    std::array<double, 3> position{};
    int64_t storeId = -1;
    double operator[](size_t i) const { return position.at(i); }
};
struct FiberEditBranch {
    std::string edgeUuid, targetUuid;
    int controlPointIndex = -1, branchControlPointIndex = -1;
    std::array<double, 3> controlPointDirection{}, branchControlPointDirection{};
    bool pending = false, adjacent = false;
};
// Application-independent edit payload. Coordinates remain in the stored frame.
// Pins cover all edited/retired records and every referenced endpoint.
struct FiberEdit {
    std::string uuid, name;
    std::vector<std::string> tags;
    std::optional<PinnedObject> storePin;
    std::vector<FiberEditControl> controlPoints;
    std::vector<std::array<double, 3>> linePoints;
    std::vector<FiberEditBranch> branches;
    ObjectRecord record;
    std::string sourceAnnotations = "{}";
    std::string coordinateIdentity = "{}";
};
// Publishes geometry, metadata, links and retirement in one checked action.
// Updates edit pins, control IDs and edge IDs only after a successful commit.
RevisionSelection publishFiberEdits(ObjectStoreSession&, const AttachmentIdentity&,
    std::vector<FiberEdit>& edits, const std::vector<FiberEdit>& retired,
    const std::vector<FiberEdit>& endpoints);
}
