#pragma once
#include <vc_reader/registrations.hpp>
#include <map>
#include <memory>
#include <optional>
namespace vc::objects {
class ObjectStore;
class ObjectStoreSession;
struct AttachmentIdentity;
struct ObjectRecord;
struct Txn;
using VolumeAffine = vcr::catalog::Affine;
struct VolumeTransforms {
    vcr::catalog::RegistrationGraph graph;
    std::map<std::pair<std::string,std::string>,std::vector<std::string>> choices;
    vcr::catalog::TransformResolution resolve(const std::string& source,const std::string& target) const {
        const auto it=choices.find({source,target});
        return graph.resolve(source,target,it==choices.end()?std::nullopt:std::optional(it->second));
    }
    std::optional<VolumeAffine> toVolume(const nlohmann::json& metadata,const std::string& target,double targetScale=1) const;
    // A volume with a route into a target volume.
    struct Related {
        std::string volume;
        // Resolved with the stored choice applied.
        vcr::catalog::TransformStatus status=vcr::catalog::TransformStatus::Unavailable;
        // Every distinct route, ignoring the choice.
        std::vector<vcr::catalog::TransformCandidate> routes;
        // The stored choice, when there is one.
        std::optional<std::vector<std::string>> chosen;
    };
    // The volumes with a route into `target`, by id.
    std::vector<Related> related(const std::string& target) const;
};
// The stored route choice from `source` to `target`, when there is one.
std::optional<ObjectRecord> volumeTransformChoice(const ObjectStore&,const Txn&,const std::string& source,const std::string& target);
// Through a session's attachment: store `matrix` (source voxels to target
// voxels) as a new registration named `name`, returning its UUID; make `path`
// (one of the pair's routes) the route from `source` to `target`, replacing a
// stored choice, returning the choice's UUID.
std::string registerVolumeTransform(ObjectStoreSession&,const AttachmentIdentity&,const std::string& source,
                                    const std::string& target,const VolumeAffine& matrix,const std::string& name);
std::string chooseVolumeTransform(ObjectStoreSession&,const AttachmentIdentity&,const std::string& source,
                                  const std::string& target,const std::vector<std::string>& path);
// Conservative transformed AABB, with outward rounding for floating arithmetic.
std::array<double,6> transformBounds(const std::array<double,6>&,const VolumeAffine&);
}
