#pragma once
#include <vc_reader/registrations.hpp>
#include <map>
#include <memory>
#include <optional>
namespace vc::objects {
using VolumeAffine = vcr::catalog::Affine;
struct VolumeTransforms {
    vcr::catalog::RegistrationGraph graph;
    std::map<std::pair<std::string,std::string>,std::vector<std::string>> choices;
    vcr::catalog::TransformResolution resolve(const std::string& source,const std::string& target) const {
        const auto it=choices.find({source,target});
        return graph.resolve(source,target,it==choices.end()?std::nullopt:std::optional(it->second));
    }
    std::optional<VolumeAffine> toVolume(const nlohmann::json& metadata,const std::string& target,double targetScale=1) const;
};
// Conservative transformed AABB, with outward rounding for floating arithmetic.
std::array<double,6> transformBounds(const std::array<double,6>&,const VolumeAffine&);
}
