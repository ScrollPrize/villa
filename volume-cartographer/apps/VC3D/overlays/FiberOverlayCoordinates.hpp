#pragma once
#include <cmath>
#include <vector>

namespace vc3d {
// One transform per object. Unbound objects never enter rendering or hit tests.
template<class Chain, class ResolveScale>
std::vector<Chain> boundFiberChains(const std::vector<Chain>& chains, ResolveScale resolveScale)
{
    std::vector<Chain> result;
    result.reserve(chains.size());
    for (const auto& chain : chains) {
        const auto scale = resolveScale(chain.id);
        if (!scale || !(*scale > 0.0) || !std::isfinite(*scale)) continue;
        auto& transformed = result.emplace_back(chain);
        for (auto& point : transformed.points) point *= static_cast<float>(*scale);
    }
    return result;
}
}
