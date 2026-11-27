#pragma once
#include "vc_object_store/objects/FiberConstraintGraph.hpp"
#include "vc_object_store/objects/FiberGraphSolveExtension.hpp"
#include "vc_object_store/objects/InitialImport.hpp"

namespace vc::objects {
struct FixedFiberUvColumns {
    std::span<const FiberGraphUuid> uuids;
    std::span<const int64_t> family,sampleOwner;
    std::span<const double> winding,direction,sampleDs,q,x,v,widths;
};
struct FiberResultColumn {
    std::string name;
    std::variant<std::span<const int64_t>,std::span<const double>> values;
};
InitialImportResult exportFiberResultColumns(const std::filesystem::path& destination,
    const std::filesystem::path& dependency,const ExtensionSource& expectedSource,
    const std::string& runUuid,const std::optional<std::string>& inputRunUuid,
    const std::string& stage,const std::string& method,
    const std::map<std::string,FiberGraphSetting>& settings,std::span<const FiberGraphUuid> uuids,
    std::span<const FiberResultColumn> columns,std::span<const ExtensionRow> metrics={},size_t workers=32);

// Native column-to-extension export through the same atomic initial import
// API as other graph producers. Samples are grouped by their fiber owner.
// No copied input objects or per-object routing metadata are introduced.
InitialImportResult exportFixedFiberUvColumns(const std::filesystem::path& destination,
    const std::filesystem::path& dependency,const ExtensionSource& expectedSource,
    const std::string& runUuid,const std::optional<std::string>& inputRunUuid,const std::string& method,
    const std::map<std::string,FiberGraphSetting>& settings,const FixedFiberUvColumns&,
    size_t workers=32);
} // namespace vc::objects
