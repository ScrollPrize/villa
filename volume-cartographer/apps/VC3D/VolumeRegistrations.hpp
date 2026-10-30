#pragma once
// The affine a user types when registering a volume (the store relates,
// resolves and records registrations: vc::objects::VolumeTransforms).
#include <vc_object_store/objects/VolumeTransforms.hpp>

#include <string>

namespace vc3d::registrations {

// An affine typed as three or four rows of four numbers (any separators, so a
// pasted JSON matrix works). Throws when it is not one or is singular.
vc::objects::VolumeAffine parseAffine(const std::string& text);

} // namespace vc3d::registrations
