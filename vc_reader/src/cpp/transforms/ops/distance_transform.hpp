#pragma once
#include "options.hpp"

namespace vcr::transforms {
enum class DistanceMode { Signed, Inside, Outside };
// vesuvius aux targets: inside = EDT(foreground), outside = EDT(background),
// signed = outside - inside; foreground is value > threshold of one channel.
struct DistanceTransform {
    std::vector<std::string> on; std::size_t channel = 0; double threshold = 0;
    DistanceMode mode = DistanceMode::Signed; std::string suffix = "_sdt";
    std::string normals; std::string components = "zyx";  // normals: field suffix, empty for none
    double p = 1;
};
} // namespace vcr::transforms
