#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include "vc_reader/catalog.hpp"
#include "catalog_io.hpp"
namespace nb=nanobind;
using namespace vcr::catalog;
namespace {
nb::object pyJson(const nlohmann::json& j){return nb::module_::import_("json").attr("loads")(j.dump());}
nlohmann::json json(const nb::object& v){return nlohmann::json::parse(nb::cast<std::string>(nb::module_::import_("json").attr("dumps")(v)));}
template<class T> void metadata(nb::class_<T>& cls) {
    cls.def_prop_ro("raw",[](const T& v){return pyJson(v.raw);});
}
}
NB_MODULE(_catalog,m) {
    auto access=nb::class_<OpenDataAccessRoot>(m,"AccessRoot");
    access.def_ro("type",&OpenDataAccessRoot::type).def_ro("url",&OpenDataAccessRoot::url).def_ro("usage",&OpenDataAccessRoot::usage);metadata(access);
    auto origin=nb::class_<OpenDataOrigin>(m,"Origin");
    origin.def_ro("path",&OpenDataOrigin::path).def_ro("access_roots",&OpenDataOrigin::accessRoots);metadata(origin);
    auto artifact=nb::class_<OpenDataArtifact>(m,"Artifact");
    artifact.def_ro("type",&OpenDataArtifact::type).def_ro("origins",&OpenDataArtifact::origins)
        .def_ro("level",&OpenDataArtifact::sourceCoordinateLevel).def_ro("level_present",&OpenDataArtifact::levelParameterPresent)
        .def_ro("target_volume_id",&OpenDataArtifact::targetVolumeId).def_ro("model_id",&OpenDataArtifact::modelId)
        .def_ro("url",&OpenDataArtifact::resolvedUrl).def_ro("source_path",&OpenDataArtifact::sourcePath)
        .def_prop_ro("parameters",[](const OpenDataArtifact& a){return pyJson(a.parameters);})
        .def_prop_ro("properties",[](const OpenDataArtifact& a){return pyJson(a.properties);})
        .def_prop_ro("creation_info",[](const OpenDataArtifact& a){return pyJson(a.creationInfo);});metadata(artifact);
    auto scan=nb::class_<OpenDataScan>(m,"Scan");
    scan.def_ro("id",&OpenDataScan::id).def_ro("pixel_size_um",&OpenDataScan::pixelSizeUm).def_ro("artifacts",&OpenDataScan::artifacts);metadata(scan);
    auto volume=nb::class_<OpenDataVolume>(m,"Volume");
    volume.def_ro("id",&OpenDataVolume::id).def_ro("scan_id",&OpenDataVolume::scanId)
        .def_ro("pixel_size_um",&OpenDataVolume::pixelSizeUm).def_ro("energy_kev",&OpenDataVolume::energyKeV)
        .def_ro("detector_distance_mm",&OpenDataVolume::detectorDistanceMm).def_ro("shape_zyx",&OpenDataVolume::shapeZYX)
        .def_ro("dtype",&OpenDataVolume::dataFormat).def_ro("artifacts",&OpenDataVolume::artifacts)
        .def_prop_ro("properties",[](const OpenDataVolume& v){return pyJson(v.properties);});metadata(volume);
    auto segment=nb::class_<OpenDataSegment>(m,"Segment");
    segment.def_ro("id",&OpenDataSegment::id).def_ro("original_volume_id",&OpenDataSegment::originalVolumeId)
        .def_ro("width",&OpenDataSegment::width).def_ro("height",&OpenDataSegment::height).def_ro("artifacts",&OpenDataSegment::artifacts)
        .def_prop_ro("properties",[](const OpenDataSegment& v){return pyJson(v.properties);});metadata(segment);
    auto sample=nb::class_<OpenDataSample>(m,"Sample");
    sample.def_ro("id",&OpenDataSample::id).def_ro("type",&OpenDataSample::type).def_ro("description",&OpenDataSample::description)
        .def_ro("scans",&OpenDataSample::scans).def_ro("volumes",&OpenDataSample::volumes).def_ro("segments",&OpenDataSample::segments)
        .def_ro("artifacts",&OpenDataSample::artifacts).def_prop_ro("properties",[](const OpenDataSample& v){return pyJson(v.properties);});metadata(sample);
    auto model=nb::class_<OpenDataModel>(m,"Model");model.def_ro("id",&OpenDataModel::id);metadata(model);
    auto catalog=nb::class_<OpenDataManifest>(m,"Catalog");
    catalog.def_ro("samples",&OpenDataManifest::samples).def_ro("models",&OpenDataManifest::models)
        .def("transform",[](const OpenDataManifest& c,const std::string& sample,const std::string& source,const std::string& target)->nb::object {
            const auto* s=c.findSample(sample);if(!s)return nb::none();
            RegistrationGraph graph;graph.addSample(*s);const auto resolution=graph.resolve(source,target);
            if(resolution.status==TransformStatus::Unavailable)return nb::none();
            const auto& candidate=resolution.resolved();
            std::string derivation;
            for(const auto& g:s->volumeTransforms)if(g.fromVolumeId==source)for(const auto& t:g.transforms)if(t.toVolumeId==target)derivation=t.derivationPath;
            return pyJson({{"matrix",candidate.forward.json()},{"derivation_path",derivation.empty()?nlohmann::json(nullptr):nlohmann::json(derivation)}});
        });metadata(catalog);
    m.def("parse",[](const std::string& text){return parseOpenDataManifest(text);});
    m.def("parse_artifacts",[](const nb::object& entries){return parseCatalogArtifacts(json(entries));});
    m.def("read_file",[](const std::string& path){return pyJson(nlohmann::json::parse(readCatalogFile(path)));});
    m.def("load",[](const std::string& source,const std::string& cache,bool refresh,double age){
        CatalogLoadResult result;
        {nb::gil_scoped_release release;result=loadCatalog(source,{cache,refresh,age});}
        if(!result.warning.empty())nb::module_::import_("warnings").attr("warn")(result.warning);
        return result.manifest;
    },nb::arg("source"),nb::arg("cache")="",nb::arg("refresh")=false,nb::arg("max_age")=3600.);
    m.def("join_url",&joinOpenDataUrl);
    m.def("to_https",[](const std::string& url)->std::optional<std::string>{
        if(url.starts_with("http://")||url.starts_with("https://")||url.starts_with("s3://")||url.starts_with("s3+"))return resolveOpenDataUrl(url);
        return std::nullopt;
    });
    m.def("affine_inverse",[](const nb::object& rows){return pyJson(Affine::fromJson(json(rows)).inverse().json());});
    m.def("affine_compose",[](const nb::object& a,const nb::object& b){return pyJson((Affine::fromJson(json(a))*Affine::fromJson(json(b))).json());});
    m.def("affine_apply",[](const nb::object& rows,const std::array<double,3>& point){return Affine::fromJson(json(rows)).apply(point);});
}
