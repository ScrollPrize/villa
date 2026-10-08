#pragma once

#include "FiberMapBentRays.hpp"
#include "FiberWindingSolver.hpp"

#include <algorithm>

namespace vc3d::fiber_map
{

// A curtain hit as the pair shard stores it: the owner's seeds and the
// strip's ray starts (its identity), the ray length and accumulated angle,
// the hit point and its position on the other polyline, the contact flags,
// and the earliest arclength at which the owner's own line crosses ANY
// contributing strip side (`selfCrossingS`, infinity when none does).
// One conversion for the layout and for tests that drive hand-made
// curtains through the classification.
inline winding::BentHit bentHitFromCurtain(const bent::BentCurtain& curtain, const bent::CurtainHit& hit,
                                           bool ownerIsV, const bent::CurtainSelfCrossings& selfCrossings)
{
    winding::BentHit record;
    record.ownerIsV = ownerIsV;
    record.stretch = hit.stretch;
    record.side = hit.side;
    record.seedA = hit.seedA;
    record.seedB = hit.seedB;
    record.across = hit.across;
    const bent::BentStretch& stretch = curtain.stretches[hit.stretch];
    for (const auto& [a, b] : hit.contributingStrips) {
        const auto seeds = std::minmax(stretch.rays[a].startSample, stretch.rays[b].startSample);
        record.contributingStrips.emplace_back(seeds.first, seeds.second);
    }
    std::sort(record.contributingStrips.begin(), record.contributingStrips.end());
    record.contributingStrips.erase(
        std::unique(record.contributingStrips.begin(), record.contributingStrips.end()),
        record.contributingStrips.end());
    const cv::Vec3d& startA = stretch.rays[hit.rayA].points.front();
    const cv::Vec3d& startB = stretch.rays[hit.rayB].points.front();
    record.startAX = startA[0];
    record.startAY = startA[1];
    record.startAZ = startA[2];
    record.startBX = startB[0];
    record.startBY = startB[1];
    record.startBZ = startB[2];
    record.s = hit.s;
    record.theta = hit.theta;
    record.hitX = hit.point[0];
    record.hitY = hit.point[1];
    record.hitZ = hit.point[2];
    record.segment = hit.segment;
    record.t = hit.t;
    record.transversality = hit.transversality;
    record.touch = hit.touch;
    record.tangential = hit.tangential;
    record.selfCrossingS = bent::firstSelfCrossing(hit, selfCrossings);
    record.turnDeg = hit.turnDeg;
    record.minConditioning = hit.minConditioning;
    return record;
}

} // namespace vc3d::fiber_map
