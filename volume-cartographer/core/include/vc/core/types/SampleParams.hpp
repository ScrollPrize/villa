#pragma once

#include <optional>

#include "vc/core/types/Sampling.hpp"
#include "vc/core/util/Compositing.hpp"

namespace vc {

// Unified sampling parameters for Volume methods.
// Default-constructed gives level-0 trilinear with no composite.
struct SampleParams {
    int level = 0;
    Sampling method = Sampling::Trilinear;

    // Compositing (optional). When set, caller must provide normals.
    std::optional<CompositeParams> composite;
    int zStart = 0;
    int zEnd = 0;
};

}  // namespace vc
