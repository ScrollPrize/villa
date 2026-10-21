#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <type_traits>

namespace vc::objects::detail {
template<class Record, class Visit>
void commonFields(Record& r, Visit visit)
{
    visit("generation", r.generation);
    visit("created_by", r.createdBy); visit("created_at", r.createdAt); visit("created_host", r.createdHost);
    visit("changed_by", r.changedBy); visit("changed_at", r.changedAt); visit("changed_host", r.changedHost);
    visit("approval_state", r.approvalState);
    visit("approved_by", r.approvedBy); visit("approved_at", r.approvedAt); visit("approved_host", r.approvedHost);
    visit("approved_generation", r.approvedGeneration);
    visit("tags", r.tags);
}

template<class Record, class Visit>
void objectFields(Record& r, Visit visit)
{
    visit("name", r.name);
    visit("format_version", r.formatVersion);
    visit("storage_kind", r.storageKind); visit("storage_ref", r.storageRef);
    visit("data_generation", r.dataGeneration);
}

template<class Detail, class Visit>
void detailFields(Detail& d, Visit visit)
{
    using T = std::remove_cv_t<Detail>;
    if constexpr (std::is_same_v<T, FiberDetail>) {
        visit("family", d.family); visit("manual_family", d.manualFamily);
        visit("control_point_count", d.controlPointCount); visit("line_point_count", d.linePointCount);
        visit("length", d.length); visit("width", d.width);
        visit("width_gap_fraction", d.widthGapFraction); visit("optimization_mode", d.optimizationMode);
    } else if constexpr (std::is_same_v<T, PatchDetail>) {
        visit("grid_width", d.gridWidth); visit("grid_height", d.gridHeight);
        visit("scale", d.scale);
    } else {
        static_assert(std::is_same_v<T, TrackDetail>);
        visit("dataset_ref", d.datasetRef); visit("external_key", d.externalKey);
        visit("family", d.family); visit("point_count", d.pointCount); visit("length", d.length);
    }
    visit("bbox", d.bbox);
}

template<class Record, class Visit>
void objectDetails(Record& r, Visit visit)
{
    visit("fiber", r.fiber); visit("patch", r.patch); visit("track", r.track);
}
}
