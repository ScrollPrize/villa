#pragma once
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/FiberImportExtension.hpp"

namespace vc::objects {
using FiberGraphUuid = std::array<unsigned char,16>;
struct CompactFiberConstraint {
    FiberGraphUuid uuid{},a{},b{};
    size_t runtimeIndex=0;
    std::array<double,3> normalA{},normalB{},connector{};
    double arcA=0,arcB=0,windingDistance=0;
    bool parallel=false,validA=false,validB=false,validConnector=false;
    size_t measurementPartition=0;
};
// Numeric projection only. Output is UUID ordered and gets dense runtime indices;
// source indices/endpoint XYZ are not part of the compact persisted format.
std::vector<CompactFiberConstraint> readCompactFiberConstraintGraph(
    const ObjectCollection&,const CollectionTxn&,size_t workers=32,
    const std::function<void(const char*,size_t,size_t)>& progress={});
// Stream original rows only when a caller needs to publish modified measurements.
// Visitor calls are concurrent, batches bounded, and winning overlay rows only.
using FiberMeasurementBatch = std::vector<std::pair<size_t,ExtensionRow>>;
std::function<void()> prepareFiberConstraintMeasurementScan(const ObjectCollection&,const CollectionTxn&,
    std::span<const CompactFiberConstraint>,size_t workers,
    const std::function<void(FiberMeasurementBatch&&)>&);
FiberGraphUuid fiberGraphUuid(const std::string&);
std::string fiberGraphUuid(const FiberGraphUuid&);
// A measured constraint in canonical core-edge endpoint order.
struct FiberConstraintRecord {
    EdgeRecord edge;
    InitialExtensionRow measurement;
    size_t runtimeIndex=0;
    std::string a,b;
    std::array<double,3> normalA{},normalB{},connector{};
    bool parallel=false,validA=false,validB=false,validConnector=false;
    double arcA=0,arcB=0,windingDistance=0;
};
// Bulk scans each core edge kind once, then joins typed measurements in memory.
// No SQLite file assumptions, per-endpoint database queries or JSON parsing.
std::vector<FiberConstraintRecord> readFiberConstraintGraph(
    const ObjectCollection&,const CollectionTxn&);
} // namespace vc::objects
