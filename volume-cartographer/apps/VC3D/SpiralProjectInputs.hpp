#pragma once

#include <vc_object_store/objects/ObjectStoreSession.hpp>

#include <QJsonObject>

#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <string>

class VolumePkg;

// What a Spiral service reads of the VC3D project (spiral-fitting's
// fit_session.SpiralProject). The project file need not be readable on the
// service's host, so every session request carries this block instead.
namespace vc3d::spiral {

// What the project knows exactly of one store volume; unset where it does not.
struct ProjectVolumeFacts {
    std::optional<double> voxelSizeUm;
    std::optional<std::array<std::size_t, 3>> baseShapeZYX;
    std::optional<bool> zTopToBottom;
    std::optional<bool> leftHandedCoordinates;
};

// The facts of `pkg`'s loaded volumes by the store volume their annotations
// are kept in (vc::project::annotationVolume): every working level of a
// catalog volume is one entry. A catalog volume gives its orientation and
// full-resolution shape; any other volume its shape only when it is open at
// its own full resolution. The service picks the entry of the volume its
// inputs name, so entries no fit uses (a Lasagna channel) are harmless.
std::map<std::string, ProjectVolumeFacts> projectVolumeFacts(const VolumePkg& pkg);

// The request's "project" block: `name`, and per volume its facts and, when
// `umbilicusStore` is given, the umbilicus that store holds for it (points
// ZYX in the volume's own voxels). Throws when an umbilicus cannot be read.
QJsonObject projectBlock(const std::string& name,
                         const std::map<std::string, ProjectVolumeFacts>& volumes,
                         vc::objects::ObjectStoreSession* session,
                         const std::optional<vc::objects::AttachmentIdentity>& umbilicusStore);

} // namespace vc3d::spiral
