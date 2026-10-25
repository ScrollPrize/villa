#pragma once
// Volume registrations as VC3D handles them: which volumes the store relates to
// the open one, by which routes, and the two things VC3D does to them, register
// a new affine and choose a route. Every mapping is the store's own.
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/objects/VolumeTransforms.hpp>

#include <optional>
#include <string>
#include <vector>

namespace vc3d::registrations {

// A volume the store maps into the open one.
struct Related {
    std::string volume;
    // As the store resolves it, the stored choice applied.
    vcr::catalog::TransformStatus status = vcr::catalog::TransformStatus::Unavailable;
    // Every distinct route, ignoring the choice.
    std::vector<vcr::catalog::TransformCandidate> routes;
    // The stored choice, when there is one.
    std::optional<std::vector<std::string>> chosen;
};

// The volumes `transforms` relates to `volume`, by id.
std::vector<Related> related(const vc::objects::VolumeTransforms& transforms, const std::string& volume);

// A route's steps by registration name (or id), inverse steps marked.
std::string describeRoute(const vcr::catalog::RegistrationGraph& graph, const std::vector<std::string>& path);

// An affine typed as three or four rows of four numbers (any separators, so a
// pasted JSON matrix works). Throws when it is not one or is singular.
vc::objects::VolumeAffine parseAffine(const std::string& text);

// Store `matrix` (source voxels to target voxels) as a new registration;
// returns its UUID.
std::string registerAffine(vc::objects::ObjectStoreSession& session, const vc::objects::AttachmentIdentity& attachment,
                           const std::string& source, const std::string& target,
                           const vc::objects::VolumeAffine& matrix, const std::string& name);

// Make `path` (one of the pair's routes) the store's route from `source` to `target`.
void chooseRoute(vc::objects::ObjectStoreSession& session, const vc::objects::AttachmentIdentity& attachment,
                 const std::string& source, const std::string& target, const std::vector<std::string>& path);

} // namespace vc3d::registrations
