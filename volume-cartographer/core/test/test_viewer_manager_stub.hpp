#pragma once
// ViewerManager's stand-in for overlay tests: locate() reads the drafts of
// the session given here, in `volume`'s voxels (as VC3D reads its store).
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <memory>
#include <string>

namespace vc3d_test {
void setLocateSession(std::shared_ptr<vc::objects::ObjectStoreSession> session, std::string volume);
}
