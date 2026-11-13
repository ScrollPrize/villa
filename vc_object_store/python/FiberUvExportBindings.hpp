#pragma once
#include "vc_object_store/objects/FiberUvExport.hpp"
#include "vc_object_store/ParallelFor.hpp"
#include "vc_object_store/Uuid.hpp"
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

namespace {
using UvExportInt=nb::ndarray<const int64_t,nb::ndim<1>,nb::c_contig,nb::device::cpu>;
using UvExportReal=nb::ndarray<const double,nb::ndim<1>,nb::c_contig,nb::device::cpu>;
using UvExportUuid=nb::ndarray<const uint8_t,nb::ndim<2>,nb::c_contig,nb::device::cpu>;

void bindFiberUvExport(nb::module_& module)
{
    using namespace nb::literals;
    module.def("format_uuid_ascii",[](UvExportUuid binary,size_t workers){
        if(binary.shape(1)!=16)throw std::invalid_argument("binary UUID column must have 16 bytes per row");
        std::unique_ptr<uint8_t[]> text;
        {nb::gil_scoped_release release;text=vo::fiberGraphUuidAscii(
            std::span(reinterpret_cast<const vo::FiberGraphUuid*>(binary.data()),binary.shape(0)),workers);}
        auto* raw=text.release();nb::capsule owner(raw,[](void* p)noexcept{delete[] static_cast<uint8_t*>(p);});
        return nb::ndarray<const uint8_t,nb::numpy,nb::ndim<2>>(raw,{binary.shape(0),size_t(36)},owner);
    },"binary"_a,"workers"_a=32,"Format selected binary UUIDs as an owned ASCII column in parallel.");
    module.def("canonical_uuid_ascii",[](UvExportUuid ascii,size_t workers){
        if(ascii.shape(1)!=36)throw std::invalid_argument("UUID ASCII column must have 36 bytes per row");
        auto values=std::make_unique_for_overwrite<uint8_t[]>(ascii.shape(0)*36);
        {nb::gil_scoped_release release;
            vc::objects::parallelFor((ascii.shape(0)+4095)/4096,workers,[&](size_t task){
                for(size_t i=task*4096;i<std::min(ascii.shape(0),(task+1)*4096);++i){
                    const auto text=vo::util::canonicalUuid(std::string(reinterpret_cast<const char*>(ascii.data()+i*36),36));
                    std::copy(text.begin(),text.end(),values.get()+i*36);
                }
            });}
        auto* raw=values.release();nb::capsule owner(raw,[](void* p)noexcept{delete[] static_cast<uint8_t*>(p);});
        return nb::ndarray<const uint8_t,nb::numpy,nb::ndim<2>>(raw,{ascii.shape(0),size_t(36)},owner);
    },"ascii"_a,"workers"_a=32,"Canonicalize a UUID ASCII column using the shared UUID validator in parallel.");
    module.def("fiber_uuid_bytes",[](UvExportUuid ascii,size_t workers){
        if(ascii.shape(1)!=36)throw std::invalid_argument("UUID ASCII column must have 36 bytes per row");
        auto values=std::make_unique_for_overwrite<vo::FiberGraphUuid[]>(ascii.shape(0));
        {nb::gil_scoped_release release;
            vc::objects::parallelFor((ascii.shape(0)+4095)/4096,workers,[&](size_t task){
                for(size_t i=task*4096;i<std::min(ascii.shape(0),(task+1)*4096);++i)
                    values[i]=vo::fiberGraphUuid(std::string(reinterpret_cast<const char*>(ascii.data()+i*36),36));
            });}
        auto* raw=values.release();nb::capsule owner(raw,[](void* p)noexcept{delete[] static_cast<vo::FiberGraphUuid*>(p);});
        return nb::ndarray<const uint8_t,nb::numpy,nb::ndim<2>>(reinterpret_cast<uint8_t*>(raw),{ascii.shape(0),size_t(16)},owner);
    },"ascii"_a,"workers"_a=32,"Convert a UUID text column to owned binary UUIDs in parallel.");
    module.def("export_fiber_result_columns",[](const std::filesystem::path& destination,
        const std::filesystem::path& dependency,const nb::dict& expectedSource,
        const std::string& runUuid,const std::optional<std::string>& inputRun,const std::string& stage,
        const std::string& method,const nb::dict& settings,UvExportUuid uuids,
        const nb::dict& inputColumns,const nb::sequence& inputMetrics,size_t workers){
        if(uuids.shape(1)!=16)throw std::invalid_argument("fiber UUID column must be binary [N,16]");
        const auto expected=vo::extensionSourcesFromJson("["+jsonDumps(expectedSource)+"]").at(0);
        const auto scalarSettings=fiberGraphSettingsFromPy(settings);
        const auto schema=vo::fiberGraphSolveTable(vo::FiberGraphTable::FiberResults);
        std::vector<UvExportInt> integerOwners;std::vector<UvExportReal> realOwners;
        std::vector<vo::FiberResultColumn> columns;columns.reserve(nb::len(inputColumns));
        for(const auto [key,value]:inputColumns){
            const auto name=nb::cast<std::string>(key);
            const auto field=std::find_if(schema.columns.begin(),schema.columns.end(),[&](const auto& c){return c.name==name;});
            if(field==schema.columns.end())throw std::invalid_argument("unknown fiber result field: "+name);
            const auto array=nb::cast<nb::ndarray<>>(value);
            if(field->type==vo::ExtensionType::Integer&&array.dtype()==nb::dtype<int64_t>()){
                auto typed=nb::cast<UvExportInt>(value);columns.push_back({name,std::span(typed.data(),typed.shape(0))});integerOwners.push_back(typed);
            }else if(field->type==vo::ExtensionType::Real&&array.dtype()==nb::dtype<double>()){
                auto typed=nb::cast<UvExportReal>(value);columns.push_back({name,std::span(typed.data(),typed.shape(0))});realOwners.push_back(typed);
            }else throw std::invalid_argument("fiber result field requires its declared int64/float64 column type: "+name);
        }
        // Metrics are small run-level summaries, not corpus rows. Decode them
        // with the existing named-row builder and its cross-field rules.
        std::vector<vo::ExtensionRow> metrics;
        for(const auto value:inputMetrics){
            const auto input=nb::cast<nb::dict>(value);
            std::map<std::string,vo::ExtensionValue> fields{{"run_uuid",vo::extensionUuid(runUuid)}};
            for(const auto [key,item]:input)fields.emplace(nb::cast<std::string>(key),extensionValueFromPy(item));
            metrics.push_back(vo::fiberGraphSolveRow(vo::FiberGraphTable::Metrics,fields).values);
        }
        vo::InitialImportResult result;
        {nb::gil_scoped_release release;result=vo::exportFiberResultColumns(destination,dependency,expected,runUuid,inputRun,
            stage,method,scalarSettings,std::span(reinterpret_cast<const vo::FiberGraphUuid*>(uuids.data()),uuids.shape(0)),columns,metrics,workers);}
        nb::dict out;out["run_uuid"]=runUuid;out["path"]=result.path.string();
        out["ingestion_seconds"]=result.ingestionSeconds;out["persistence_seconds"]=result.persistenceSeconds;
        out["total_seconds"]=result.totalSeconds;return out;
    },"destination"_a,"dependency"_a,"expected_source"_a,"run_uuid"_a,"input_run_uuid"_a,
      "stage"_a,"method"_a,"settings"_a,"fiber_uuid"_a,"columns"_a,"metrics"_a=nb::tuple(),"workers"_a=32,
      "Write numeric fiber result columns through shared bounded parallel initial-import batches.");
    module.def("export_fixed_uv_columns",[](const std::filesystem::path& destination,
        const std::filesystem::path& dependency,const nb::dict& expectedSource,
        const std::string& runUuid,const std::optional<std::string>& inputRun,const std::string& method,const nb::dict& settings,
        UvExportUuid uuids,UvExportInt family,UvExportReal winding,UvExportReal direction,
        UvExportInt owner,UvExportReal sampleDs,UvExportReal q,UvExportReal x,UvExportReal v,UvExportReal widths,size_t workers){
        if(uuids.shape(1)!=16)throw std::invalid_argument("UV UUID column must be binary [N,16]");
        const auto expected=vo::extensionSourcesFromJson("["+jsonDumps(expectedSource)+"]").at(0);
        const auto scalarSettings=fiberGraphSettingsFromPy(settings);
        const auto reals=[](UvExportReal a){return std::span(a.data(),a.shape(0));};
        const vo::FixedFiberUvColumns columns{std::span(reinterpret_cast<const vo::FiberGraphUuid*>(uuids.data()),uuids.shape(0)),
            std::span(family.data(),family.shape(0)),std::span(owner.data(),owner.shape(0)),
            reals(winding),reals(direction),reals(sampleDs),reals(q),reals(x),reals(v),reals(widths)};
        vo::InitialImportResult result;
        {nb::gil_scoped_release release;
            result=vo::exportFixedFiberUvColumns(destination,dependency,expected,runUuid,inputRun,method,scalarSettings,columns,workers);}
        nb::dict out;out["run_uuid"]=runUuid;out["path"]=result.path.string();
        out["ingestion_seconds"]=result.ingestionSeconds;out["persistence_seconds"]=result.persistenceSeconds;
        out["total_seconds"]=result.totalSeconds;return out;
    },"destination"_a,"dependency"_a,"expected_source"_a,"run_uuid"_a,"input_run_uuid"_a,"method"_a,"settings"_a,
      "fiber_uuid"_a,"family"_a,"winding"_a,"direction"_a,"sample_owner"_a,"sample_ds"_a,"q"_a,"x"_a,"v"_a,"widths"_a,
      "workers"_a=32,"Write grouped UV result columns through bounded parallel initial-import batches.");
}
}
