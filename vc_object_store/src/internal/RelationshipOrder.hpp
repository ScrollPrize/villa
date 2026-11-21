#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <algorithm>

namespace vc::objects::detail {
// One order for native, wire and reopened graph-metadata cache results.
inline void orderRelationships(std::vector<SelectedRelationship>& entries)
{
    std::sort(entries.begin(),entries.end(),[](const auto& a,const auto& b){
        if(a.relationship.record.uuid!=b.relationship.record.uuid)return a.relationship.record.uuid<b.relationship.record.uuid;
        return a.relationship.reference()<b.relationship.reference();
    });
}
}
