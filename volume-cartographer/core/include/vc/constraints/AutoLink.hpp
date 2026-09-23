#pragma once
#include "vc/constraints/ConstraintGraph.hpp"
#include <functional>
#include <set>

namespace vc::constraints {
// Automatic fiber/patch linking from geometry alone. Validated on the
// fiber_linking review (2026-09): zero cross-validated false links at 84% of
// the human-accepted associations. Two mechanisms carry the result:
//  * Anchors: patches the fiber lies on for at least anchorContact voxels of
//    merged contact runs, whose vertices agree with the fiber's normal-field
//    strip more than they sit one sheet away from it (centerline-weighted).
//  * Growth: patches overlapping the linked set on the same sheet (near, with
//    parallel normals) clearly more than on an adjacent sheet, repeated until
//    stable. A wrong-sheet cluster is internally consistent but never touches
//    the linked set, so it stays unlinked and is left for human review.
struct AutoLinkSettings {
    double contactGap = 40;        // arclength gap that splits contact runs (2 x spacing)
    double anchorContact = 75;     // minimum merged contact run length of an anchor, in graph voxels
    double sameSheet = 3;          // vertex-to-surface distance counted as the same sheet
    double adjacentLow = 4.5, adjacentHigh = 16; // distance band counted as an adjacent sheet
    double sameNormal = .7, adjacentNormal = .85; // |cos| between surface normals
    double growSame = .1;          // minimum same-sheet vertex fraction against the linked set
    double growRatio = 2;          // same-sheet fraction must exceed growRatio x adjacent fraction
    std::array<double,4> stripWeights{1,.7,.4,.15}; // trust by strip rows from the centerline
    double stripGateRatio = 1.5;   // anchor needs strip same-sheet fraction > stripGateRatio x adjacent fraction + stripGateMargin
    double stripGateMargin = 0;
    double stripMargin = 20;       // bounding-box margin for strip and patch overlap queries
    int maxRounds = 8;
};
struct AutoLinkCandidate {
    std::string id;
    double contact = 0;            // merged contact run length along the fiber
    double stripSame = 0, stripAdjacent = 0; // centerline-weighted strip agreement fractions
    bool anchor = false, linked = false;
    int round = -1;                // 0 for anchors, growth round otherwise, -1 unlinked
    double linkedSame = 0, linkedAdjacent = 0; // final relation to the other linked patches
};
struct AutoLinkResult {
    std::vector<AutoLinkCandidate> candidates; // proposal order
    size_t anchors = 0, linked = 0;
};
using PatchLoader = std::function<std::shared_ptr<const Node>(const std::string&)>;
// fiber: the review fiber node (strip). proposals: fiber-review candidates
// with dense contacts within the discovery radius. Deterministic; the loader
// may return null for unavailable patches, which stay unlinked.
AutoLinkResult autoLinkFiber(const Node& fiber, const std::vector<Candidate>& proposals, const PatchLoader&,
                             const AutoLinkSettings& = {}, const std::atomic_bool* cancel = nullptr, Progress = {});
nlohmann::json autoLinkSettingsToJson(const AutoLinkSettings&);
AutoLinkSettings autoLinkSettingsFromJson(const nlohmann::json&);
} // namespace vc::constraints
