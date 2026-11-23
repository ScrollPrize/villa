#pragma once
#include <vc_reader/registrations.hpp>
#include <map>
#include <memory>
#include <optional>
namespace vc::objects {
class StoreReader;
class ObjectStore;
class ObjectStoreSession;
struct AttachmentIdentity;
struct ObjectRecord;
struct Txn;
struct TransactionRequest;
using VolumeAffine = vcr::catalog::Affine;
struct VolumeTransforms {
    vcr::catalog::RegistrationGraph graph;
    std::map<std::pair<std::string,std::string>,std::vector<std::string>> choices;
    // The route choice in force from `source` to `target`: the stored choice
    // for that pair, else the stored choice for the reverse pair walked
    // backwards, so one choice settles both directions of a volume pair.
    std::optional<std::vector<std::string>> chosen(const std::string& source,const std::string& target) const;
    vcr::catalog::TransformResolution resolve(const std::string& source,const std::string& target) const {
        return graph.resolve(source,target,chosen(source,target));
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
// The transform context of recorded registrations (each `volume_transform`
// record with its head revision UUID, or its ordinary lineage/version when
// history is disabled, as the registration's revision evidence)
// and route choices (`volume_transform_choice` records), plus `catalog`
// registrations the store has not recorded. Local and hosted stores build it
// the same way.
VolumeTransforms volumeTransformsFrom(const std::vector<std::pair<ObjectRecord,std::string>>& registrations,
                                      const std::vector<ObjectRecord>& choices,
                                      const vcr::catalog::RegistrationGraph* catalog);
// Throws TransactionConflict when the object's mapping is unavailable,
// ambiguous or differs from the one used to compute an edit.
void checkVolumeMapping(const ObjectStore&, const Txn&, const std::string& objectUuid,
                        const std::string& target, double targetScale, const VolumeAffine& expected);
// Stable object UUID of a recorded catalog registration (by catalog id).
std::string catalogTransformUuid(const std::string& registrationId);
// The stored route choice from `source` to `target`, when there is one.
std::optional<ObjectRecord> volumeTransformChoice(const ObjectStore&,const Txn&,const std::string& source,const std::string& target);
std::optional<ObjectRecord> volumeTransformChoice(const StoreReader&,const std::string& source,const std::string& target);
// `base` (a new record with its UUID, or the existing one) as the
// registration of `matrix` from `source` to `target` named `name`; validates it.
ObjectRecord volumeTransformRecord(ObjectRecord base,const std::string& source,const std::string& target,
                                   const VolumeAffine& matrix,const std::string& name);
// `base` as the choice of route `path` from `source` to `target` (not validated).
ObjectRecord volumeTransformChoiceRecord(ObjectRecord base,const std::string& source,const std::string& target,
                                         const std::vector<std::string>& path);
// Through a session's attachment (local or hosted), as checked requests: store
// `matrix` (source voxels to target voxels) as a new registration named `name`,
// returning its UUID; make `path` (one of the pair's routes, resolved against
// the recorded transforms, which the request depends on) the route from
// `source` to `target`, replacing a stored choice, returning the choice's UUID.
std::string registerVolumeTransform(ObjectStoreSession&,const AttachmentIdentity&,const std::string& source,
                                    const std::string& target,const VolumeAffine& matrix,const std::string& name);
std::string chooseVolumeTransform(ObjectStoreSession&,const AttachmentIdentity&,const std::string& source,
                                  const std::string& target,const std::vector<std::string>& path);
// Prepare against one selected graph, including absence of a previous choice.
std::string chooseVolumeTransformRequest(const StoreReader&, TransactionRequest&, const std::string& source,
                                         const std::string& target, const std::vector<std::string>& path);
// Conservative transformed AABB, with outward rounding for floating arithmetic.
std::array<double,6> transformBounds(const std::array<double,6>&,const VolumeAffine&);
}
