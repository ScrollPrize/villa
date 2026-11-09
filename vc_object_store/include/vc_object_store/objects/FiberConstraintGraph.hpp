#pragma once
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/FiberImportExtension.hpp"

namespace vc::objects {
// A measured constraint in SOURCE endpoint order. The native edge remains
// canonical, but source_reversed is applied to endpoints, not signed evidence.
struct FiberConstraintRecord {
    EdgeRecord edge;
    InitialExtensionRow measurement;
    int64_t sourceId=0;
    std::string a,b;
    std::array<double,3> pointA{},pointB{},normalA{},normalB{};
    bool validA=false,validB=false;
    double arcA=0,arcB=0,parallelScore=0,perpendicularScore=0;
    double windingDistance=0,parallelWindingDistance=0,evidenceWeight=0;
};
// Bulk scans each core edge kind once, then joins typed measurements in memory.
// No SQLite file assumptions, per-endpoint database queries or JSON parsing.
std::vector<FiberConstraintRecord> readFiberConstraintGraph(
    const ObjectCollection&,const CollectionTxn&);
} // namespace vc::objects
