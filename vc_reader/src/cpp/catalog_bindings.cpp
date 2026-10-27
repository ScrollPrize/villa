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
// A surface with its artifact copied, so it outlives the catalog.
struct SurfaceCopy {OpenDataArtifact artifact;bool authored;bool canonical;std::string volumeId;double unitsPerVoxel;std::string id;};
template<class T> void metadata(nb::class_<T>& cls) {
    cls.def_prop_ro("raw",[](const T& v){return pyJson(v.raw);});
}
}
NB_MODULE(_catalog,m) {
    nb::class_<OpenDataReference>(m,"Reference").def_ro("type",&OpenDataReference::type).def_ro("id",&OpenDataReference::id);
    nb::class_<OpenDataCreation>(m,"Creation").def_ro("date",&OpenDataCreation::date)
        .def_ro("process",&OpenDataCreation::process).def_ro("derived_from",&OpenDataCreation::derivedFrom);
    nb::class_<OpenDataBounds>(m,"Bounds").def_ro("low",&OpenDataBounds::low).def_ro("high",&OpenDataBounds::high);
    nb::class_<OpenDataVolumeCoverage>(m,"VolumeCoverage").def_ro("volume_id",&OpenDataVolumeCoverage::volumeId)
        .def_ro("bbox",&OpenDataVolumeCoverage::bbox).def_ro("overlap_ratio",&OpenDataVolumeCoverage::overlapRatio)
        .def_ro("computed_at",&OpenDataVolumeCoverage::computedAt);
    nb::class_<OpenDataTifxyzInfo>(m,"TifxyzInfo").def_ro("scale",&OpenDataTifxyzInfo::scale)
        .def_ro("bbox",&OpenDataTifxyzInfo::bbox).def_ro("tiff_dimensions",&OpenDataTifxyzInfo::tiffDimensions);
    nb::class_<OpenDataSupersession>(m,"Supersession").def_ro("volume_id",&OpenDataSupersession::volumeId)
        .def_ro("date",&OpenDataSupersession::date).def_ro("reason",&OpenDataSupersession::reason);
    nb::class_<OpenDataLicense>(m,"License").def_ro("name",&OpenDataLicense::name).def_ro("url",&OpenDataLicense::url);
    auto access=nb::class_<OpenDataAccessRoot>(m,"AccessRoot");
    access.def_ro("type",&OpenDataAccessRoot::type).def_ro("url",&OpenDataAccessRoot::url).def_ro("usage",&OpenDataAccessRoot::usage);metadata(access);
    auto origin=nb::class_<OpenDataOrigin>(m,"Origin");
    origin.def_ro("path",&OpenDataOrigin::path).def_ro("access_roots",&OpenDataOrigin::accessRoots);metadata(origin);
    auto artifact=nb::class_<OpenDataArtifact>(m,"Artifact");
    artifact.def_ro("type",&OpenDataArtifact::type).def_ro("origins",&OpenDataArtifact::origins)
        .def_ro("level",&OpenDataArtifact::sourceCoordinateLevel).def_ro("level_present",&OpenDataArtifact::levelParameterPresent)
        .def_ro("target_volume_id",&OpenDataArtifact::targetVolumeId).def_ro("model_id",&OpenDataArtifact::modelId)
        .def_ro("render_level",&OpenDataArtifact::renderLevel).def_ro("downscale",&OpenDataArtifact::downscale)
        .def_ro("format",&OpenDataArtifact::format).def_ro("composite_mode",&OpenDataArtifact::compositeMode)
        .def_ro("label_set",&OpenDataArtifact::labelSet).def_ro("width",&OpenDataArtifact::width)
        .def_ro("height",&OpenDataArtifact::height).def_ro("arrays",&OpenDataArtifact::arrays)
        .def_ro("supersedes",&OpenDataArtifact::supersedes).def_ro("registration",&OpenDataArtifact::registration)
        .def_ro("annotated_on",&OpenDataArtifact::annotatedOn).def_ro("mapping_coverage",&OpenDataArtifact::mappingCoverage)
        .def_ro("creation",&OpenDataArtifact::creation).def_ro("publication_tier",&OpenDataArtifact::publicationTier)
        .def_ro("url",&OpenDataArtifact::resolvedUrl).def_ro("source_path",&OpenDataArtifact::sourcePath)
        .def_prop_ro("parameters",[](const OpenDataArtifact& a){return pyJson(a.parameters);})
        .def_prop_ro("properties",[](const OpenDataArtifact& a){return pyJson(a.properties);})
        .def_prop_ro("creation_info",[](const OpenDataArtifact& a){return pyJson(a.creationInfo);});metadata(artifact);
    auto scan=nb::class_<OpenDataScan>(m,"Scan");
    scan.def_ro("id",&OpenDataScan::id).def_ro("long_id",&OpenDataScan::longId).def_ro("creation",&OpenDataScan::creation)
        .def_ro("pixel_size_um",&OpenDataScan::pixelSizeUm).def_ro("energy_kev",&OpenDataScan::energyKeV)
        .def_ro("detector_distance_mm",&OpenDataScan::detectorDistanceMm).def_ro("location",&OpenDataScan::location)
        .def_ro("beamline",&OpenDataScan::beamline).def_ro("name",&OpenDataScan::name)
        .def_ro("artifacts",&OpenDataScan::artifacts);metadata(scan);
    auto volume=nb::class_<OpenDataVolume>(m,"Volume");
    volume.def_ro("id",&OpenDataVolume::id).def_ro("long_id",&OpenDataVolume::longId).def_ro("scan_id",&OpenDataVolume::scanId)
        .def_ro("suffix",&OpenDataVolume::suffix).def_ro("creation",&OpenDataVolume::creation)
        .def_ro("pixel_size_um",&OpenDataVolume::pixelSizeUm).def_ro("energy_kev",&OpenDataVolume::energyKeV)
        .def_ro("detector_distance_mm",&OpenDataVolume::detectorDistanceMm).def_ro("shape_zyx",&OpenDataVolume::shapeZYX)
        .def_ro("dtype",&OpenDataVolume::dataFormat).def_ro("artifacts",&OpenDataVolume::artifacts)
        .def_ro("z_direction_is_top_to_bottom",&OpenDataVolume::zDirectionIsTopToBottom)
        .def_ro("left_handed_coordinates",&OpenDataVolume::leftHandedCoordinates)
        .def_ro("text_side",&OpenDataVolume::textSide).def_ro("superseded_by",&OpenDataVolume::supersededBy)
        .def_ro("license",&OpenDataVolume::license).def_ro("render_level",&OpenDataVolume::renderLevel)
        .def_ro("render_segments",&OpenDataVolume::renderSegments)
        .def_ro("window_u16_min",&OpenDataVolume::windowU16Min).def_ro("window_u16_max",&OpenDataVolume::windowU16Max)
        .def_ro("target_window_f32_min",&OpenDataVolume::targetWindowF32Min)
        .def_ro("target_window_f32_max",&OpenDataVolume::targetWindowF32Max)
        .def_prop_ro("properties",[](const OpenDataVolume& v){return pyJson(v.properties);});metadata(volume);
    nb::class_<SurfaceCopy>(m,"Surface")
        .def_ro("artifact",&SurfaceCopy::artifact)
        .def_ro("authored",&SurfaceCopy::authored).def_ro("canonical",&SurfaceCopy::canonical)
        .def_ro("volume_id",&SurfaceCopy::volumeId).def_ro("units_per_voxel",&SurfaceCopy::unitsPerVoxel)
        .def_ro("id",&SurfaceCopy::id);
    auto segment=nb::class_<OpenDataSegment>(m,"Segment");
    segment.def_ro("id",&OpenDataSegment::id).def_ro("long_id",&OpenDataSegment::longId).def_ro("suffix",&OpenDataSegment::suffix)
        .def_ro("original_volume_id",&OpenDataSegment::originalVolumeId).def_ro("creation",&OpenDataSegment::creation)
        .def_ro("width",&OpenDataSegment::width).def_ro("height",&OpenDataSegment::height).def_ro("artifacts",&OpenDataSegment::artifacts)
        .def_ro("original_volume_downscale",&OpenDataSegment::originalVolumeDownscale)
        .def_ro("volume_coverage",&OpenDataSegment::volumeCoverage)
        .def_ro("ink_models_allow",&OpenDataSegment::inkModelsAllow).def_ro("ink_models_deny",&OpenDataSegment::inkModelsDeny)
        .def_ro("spiral_dataset_revision",&OpenDataSegment::spiralDatasetRevision)
        .def_ro("original_tifxyz",&OpenDataSegment::originalTifxyz)
        .def_prop_ro("render",[](const OpenDataSegment& v){return pyJson(v.render);})
        .def("source_volume_id",&OpenDataSegment::sourceVolumeId).def("stable_id",&OpenDataSegment::stableId)
        .def_prop_ro("properties",[](const OpenDataSegment& v){return pyJson(v.properties);});metadata(segment);
    auto sample=nb::class_<OpenDataSample>(m,"Sample");
    sample.def_ro("id",&OpenDataSample::id).def_ro("type",&OpenDataSample::type).def_ro("description",&OpenDataSample::description)
        .def_ro("canonical_volume_id",&OpenDataSample::canonicalVolumeId)
        .def_ro("scans",&OpenDataSample::scans).def_ro("volumes",&OpenDataSample::volumes).def_ro("segments",&OpenDataSample::segments)
        .def_ro("artifacts",&OpenDataSample::artifacts).def("surfaces",[](const OpenDataSample& sample,const OpenDataSegment& segment){
            std::vector<SurfaceCopy> copies;
            for(const auto& s:sample.surfaces(segment))copies.push_back({*s.artifact,s.authored,s.canonical,s.volumeId,s.unitsPerVoxel,s.id});
            return copies;
        })
        .def_prop_ro("properties",[](const OpenDataSample& v){return pyJson(v.properties);});metadata(sample);
    auto model=nb::class_<OpenDataModel>(m,"Model");
    model.def_ro("id",&OpenDataModel::id).def_ro("long_id",&OpenDataModel::longId).def_ro("suffix",&OpenDataModel::suffix)
        .def_ro("creation",&OpenDataModel::creation).def_ro("task",&OpenDataModel::task)
        .def_ro("architecture",&OpenDataModel::architecture).def_ro("compatible_samples",&OpenDataModel::compatibleSamples)
        .def_ro("excluded_samples",&OpenDataModel::excludedSamples)
        .def_prop_ro("properties",[](const OpenDataModel& v){return pyJson(v.properties);});metadata(model);
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
    m.def("load",[](const std::string& source,const std::string& cache,bool refresh,double age,double timeout){
        CatalogLoadResult result;
        {nb::gil_scoped_release release;result=loadCatalog(source,{cache,refresh,age,timeout});}
        if(!result.warning.empty())nb::module_::import_("warnings").attr("warn")(result.warning);
        return result.manifest;
    },nb::arg("source"),nb::arg("cache")="",nb::arg("refresh")=false,nb::arg("max_age")=3600.,nb::arg("timeout")=60.);
    m.def("join_url",&joinOpenDataUrl);
    m.def("to_https",[](const std::string& url)->std::optional<std::string>{
        if(url.starts_with("http://")||url.starts_with("https://")||url.starts_with("s3://")||url.starts_with("s3+"))return resolveOpenDataUrl(url);
        return std::nullopt;
    });
    m.def("affine_inverse",[](const nb::object& rows){return pyJson(Affine::fromJson(json(rows)).inverse().json());});
    m.def("affine_compose",[](const nb::object& a,const nb::object& b){return pyJson((Affine::fromJson(json(a))*Affine::fromJson(json(b))).json());});
    m.def("affine_apply_many",[](const nb::object& rows,std::vector<std::array<double,3>> points){
        const auto matrix=Affine::fromJson(json(rows));
        {nb::gil_scoped_release release;for(auto& point:points)point=matrix.apply(point);}
        return points;
    });
    m.def("affine_apply",[](const nb::object& rows,const std::array<double,3>& point){return Affine::fromJson(json(rows)).apply(point);});
}
