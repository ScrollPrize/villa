#pragma once

// Committed Spiral fits: surfaces tagged spiral-fit that a Spiral service
// wrote to the object store (spiral-fitting/service_commits.py). Each holds
// the flattened preview of one model state, in voxels of its fit volume at
// scale 1, with an int32 winding_id channel and its winding bounds in meta.

#include "SpiralPreviewWindings.hpp"

#include <vc_object_store/objects/ObjectStoreSession.hpp>

#include <opencv2/core/mat.hpp>

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

class QuadSurface;

namespace vc3d::spiral {

inline constexpr char kSpiralFitTag[] = "spiral-fit";

// Whether `record` is a committed Spiral fit (a surface tagged spiral-fit),
// retired or not. A fit never takes one in as an input.
bool isSpiralFit(const vc::objects::ObjectRecord& record);

// The committed Spiral fits among `uuids` in the attachment (retired ones
// included); a UUID it no longer holds is not one. Throws when the store
// cannot be read.
std::set<std::string> spiralFitsAmong(vc::objects::ObjectStoreSession&,
                                      const vc::objects::AttachmentIdentity&,
                                      const std::vector<std::string>& uuids);

// One committed fit as a listing shows it.
struct StoredFitEntry {
    std::string uuid;
    std::string name;
    std::string createdAt;
    std::string createdBy;
    std::optional<int64_t> sourceIteration;
};

// The active committed fits in the attachment, newest first.
std::vector<StoredFitEntry> listStoredFits(vc::objects::ObjectStoreSession&,
                                           const vc::objects::AttachmentIdentity&);

// One committed fit, read whole: its surface in its fit volume's voxels, and
// its winding-ID map checked against the bounds it declares.
struct StoredFit {
    std::string uuid;
    std::string name;
    std::string geometryRevision;
    std::string volume;  // the fit volume it is bound to
    std::shared_ptr<QuadSurface> surface;
    std::vector<PreviewComponent> components;
    cv::Mat_<int32_t> windingIds;
};

// Read the committed fit `uuid` (its current geometry, or `geometryRevision`)
// from the attached store, which holds every commit as soon as it is made
// (a file shared in place, or a served store). Throws std::runtime_error
// with the reason when it cannot be read.
StoredFit readStoredFit(const std::shared_ptr<vc::objects::ObjectStoreSession>&,
                        const vc::objects::AttachmentIdentity&, const std::string& uuid,
                        const std::optional<std::string>& geometryRevision = std::nullopt);

}  // namespace vc3d::spiral
