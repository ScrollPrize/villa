#pragma once
#include "vc/core/types/Volume.hpp"
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/objects/VolumeTransforms.hpp>

namespace vc3d {
// The frame an object was read in: `volume`'s voxels at `scale` (an Open Data
// pyramid level), `mapping` from its stored coordinates. Edits map back through it.
struct VolumeView {
    std::string volume;
    double scale = 1;
    vc::objects::VolumeAffine mapping;
};
}
