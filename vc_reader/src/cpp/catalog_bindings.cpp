#include <cmath>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/filesystem.h>
#include "vc_reader/browsing.hpp"
#include "vc_reader/catalog.hpp"
#include "vc_reader/registrations.hpp"
#include "vc_reader/sources.hpp"
#include "catalog_io.hpp"
namespace nb = nanobind;
using namespace vcr::catalog;
using namespace vcr::sources;
namespace {
nb::object pyJson(const nlohmann::json& j) { return nb::module_::import_("json").attr("loads")(j.dump()); }

std::string typeName(nb::handle value) { return nb::cast<std::string>(nb::str(value.type().attr("__name__"))); }

template <class... T> bool encodeBound(nb::handle value, nlohmann::json& out)
{
    return ((nb::isinstance<T>(value) ? (out = encode(nb::cast<const T&>(value)), true) : false) || ...);
}

// A Python value as the JSON a descriptor decodes. `tagged`: mappings in schema 1's tagged
// form, as constructor arguments need; a schema-1 document is already tagged.
nlohmann::json toJson(nb::handle value, bool tagged = true)
{
    if (value.is_none()) return nullptr;
    if (PyBool_Check(value.ptr())) return nb::cast<bool>(value);
    if (PyUnicode_Check(value.ptr())) return nb::cast<std::string>(value);
    if (PyFloat_Check(value.ptr())) {
        const double number = PyFloat_AsDouble(value.ptr());
        if (!std::isfinite(number)) throw std::invalid_argument("nonfinite declaration");
        return number;
    }
    if (PyIndex_Check(value.ptr())) {
        nb::object number = nb::steal(PyNumber_Index(value.ptr()));
        int overflow = 0;
        const long long small = PyLong_AsLongLongAndOverflow(number.ptr(), &overflow);
        if (!overflow) return std::int64_t(small);
        const unsigned long long large = PyLong_AsUnsignedLongLong(number.ptr());
        if (PyErr_Occurred()) {
            PyErr_Clear();
            if (!tagged) return PyLong_AsDouble(number.ptr());  // as parsing the document's text would
            throw std::invalid_argument("declared integer does not fit in 64 bits");
        }
        return std::uint64_t(large);
    }
    nlohmann::json bound;
    if (encodeBound<VolumeTransform, OpenDataVolumeCoverage, Frame, LevelRequest, Selection, Storage, Plane, VolumeRegistration,
                    ValueFrame, Encoding, Surface, SurfaceRegistration, Companion, QueryContext, Source>(value, bound))
        return bound;
    if (PyList_Check(value.ptr()) || PyTuple_Check(value.ptr())) {
        nlohmann::json out = nlohmann::json::array();
        for (nb::handle item : value) out.push_back(toJson(item, tagged));
        return out;
    }
    if (PyMapping_Check(value.ptr())) {
        nlohmann::json out = nlohmann::json::object();
        nb::object items = nb::steal(PyMapping_Items(value.ptr()));
        for (nb::handle item : items) {
            nb::handle key = item[0];
            if (!PyUnicode_Check(key.ptr())) throw DeclarationTypeError("metadata keys must be strings");
            out[nb::cast<std::string>(key)] = toJson(item[1], tagged);
        }
        if (!tagged) return out;
        nlohmann::json pairs = nlohmann::json::array();
        for (const auto& item : out.items()) pairs.push_back(nlohmann::json::array({item.key(), item.value()}));
        return {{"type", "mapping"}, {"items", std::move(pairs)}};
    }
    if (nb::hasattr(value, "tolist")) return toJson(value.attr("tolist")(), tagged);
    throw DeclarationTypeError("unsupported declaration value: " + typeName(value));
}

template <class T> nb::tuple tupleOf(const std::vector<T>& values)
{
    nb::list out;
    for (const auto& value : values) out.append(nb::cast(value));
    return nb::tuple(out);
}
template <class T> nb::object optionalTuple(const std::optional<std::vector<T>>& values)
{
    return values ? nb::object(tupleOf(*values)) : nb::none();
}
template <std::size_t N> nb::tuple vecTuple(const std::array<double, N>& value)
{
    nb::list out;
    for (const double v : value) out.append(v);
    return nb::tuple(out);
}
nb::object optionalVec(const std::optional<Vec3>& value) { return value ? nb::object(vecTuple(*value)) : nb::none(); }
nb::object pixelAffineTuple(const std::optional<PixelAffine>& value)
{
    if (!value) return nb::none();
    return nb::make_tuple(vecTuple((*value)[0]), vecTuple((*value)[1]));
}
nb::object wide(const WideInt& value)
{
    if (!value.negative) return nb::cast(value.magnitude);
    return nb::cast(-static_cast<std::int64_t>(value.magnitude - 1) - 1);
}
nb::tuple matrixOf(const Affine& a)
{
    nb::list rows;
    for (int r = 0; r < 3; ++r) rows.append(nb::make_tuple(a(r, 0), a(r, 1), a(r, 2), a(r, 3)));
    return nb::tuple(rows);
}
std::optional<std::string> noneIfEmpty(const std::string& value) { return value.empty() ? std::nullopt : std::optional(value); }
std::string fspath(nb::handle value) { return nb::cast<std::string>(nb::module_::import_("os").attr("fspath")(value)); }
// The id of an entity, or a plain id.
std::optional<std::string> idOfAny(nb::handle value)
{
    if (value.is_none()) return std::nullopt;
    return nb::cast<std::string>(nb::hasattr(value, "id") ? nb::object(value.attr("id")) : nb::borrow(value));
}

template <class T>
T fromPython(const std::vector<const char*>& fields, const nb::args& args, const nb::kwargs& kwargs)
{
    if (args.size() > fields.size())
        throw DeclarationTypeError("too many positional arguments: " + std::to_string(args.size()));
    nlohmann::json values = nlohmann::json::object();
    for (std::size_t i = 0; i < args.size(); ++i) values[fields[i]] = toJson(args[i]);
    for (auto [key, value] : kwargs) {
        const auto name = nb::cast<std::string>(key);
        if (values.contains(name)) throw DeclarationTypeError("got multiple values for " + name);
        values[name] = toJson(value);
    }
    return decodeFields<T>(values);
}

// Immutable value types: constructed by field name through the C++ decoder, so Python and
// C++ share one validation; compared and hashed by value; pickled through their fields.
template <class T>
void bindValue(nb::class_<T>& cls, std::vector<const char*> fields)
{
    cls.def("__init__", [fields](T* self, nb::args args, nb::kwargs kwargs) { new (self) T(fromPython<T>(fields, args, kwargs)); });
    const auto replace = [](const T& value, nb::kwargs changes) {
        auto values = encode(value).at("fields");
        for (auto [key, change] : changes) values[nb::cast<std::string>(key)] = toJson(change);
        return decodeFields<T>(values);
    };
    cls.def("replace", replace).def("__replace__", replace)
        .def("__eq__", [](const T& a, nb::handle b) { return nb::isinstance<T>(b) && a == nb::cast<const T&>(b); })
        .def("__ne__", [](const T& a, nb::handle b) { return !(nb::isinstance<T>(b) && a == nb::cast<const T&>(b)); })
        .def("__hash__", [](const T& a) { return std::hash<std::string>{}(encode(a).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace)); })
        .def("as_dict", [](const T& a) { return pyJson(plainFields(encode(a))); }, "The fields as plain nested values.")
        .def("__reduce__", [fields](nb::handle self) {
            nb::list values;
            for (const char* field : fields) values.append(self.attr(field));
            return nb::make_tuple(self.type(), nb::tuple(values));
        })
        .def("__repr__", [fields](nb::handle self) {
            std::string text = typeName(self) + "(";
            for (std::size_t i = 0; i < fields.size(); ++i)
                text += std::string(i ? ", " : "") + fields[i] + "=" + nb::cast<std::string>(nb::repr(self.attr(fields[i])));
            return text + ")";
        });
}

template <class T> void bindContext(nb::class_<T>& cls)
{
    cls.def_prop_ro("artifacts", &T::artifacts).def_prop_ro("labels", &T::labels).def_prop_ro("surface_volumes", &T::surfaceVolumes)
        .def_prop_ro("representations", &T::representations).def_prop_ro("geometry", &T::geometry);
}

// A descriptor opens through the Python reader, which adds NumPy indexing and pickling.
nb::object sourceReader(const Source& source, const nb::kwargs& options)
{
    return nb::module_::import_("vc_reader.reader.source_reader").attr("SourceReader")(source, **options);
}
OpenedGrid openedGrid(nb::handle reader)
{
    return nb::cast<OpenedGrid>(reader.attr("_engine")().attr("opened_grid")());
}

template <class T>
auto criteriaPredicate(const nb::dict& criteria)
{
    return [criteria](const T& value) {
        nb::object object = nb::cast(value);
        for (auto [key, wanted] : criteria) {
            nb::object have = nb::getattr(object, nb::cast<std::string>(key).c_str());
            const auto id = [](nb::handle v) { return nb::hasattr(v, "id") ? nb::object(v.attr("id")) : nb::borrow(v); };
            if (!id(have).equal(id(wanted))) return false;
        }
        return true;
    };
}
std::optional<bool> takeIncludeSuperseded(nb::dict& criteria)
{
    if (!criteria.contains("include_superseded")) return std::nullopt;
    nb::object value = criteria["include_superseded"];
    PyDict_DelItemString(criteria.ptr(), "include_superseded");
    if (value.is_none()) return std::nullopt;
    if (!PyBool_Check(value.ptr())) throw DeclarationTypeError("include_superseded must be boolean");
    return nb::cast<bool>(value);
}

template <class T>
nb::class_<Collection<T>> bindCollection(nb::module_& m, const char* name)
{
    using C = Collection<T>;
    auto cls = nb::class_<C>(m, name);
    cls.def("__iter__", [](const C& c) { return nb::iter(nb::cast(c.visible())); })
        .def("__len__", &C::size)
        .def("__bool__", [](const C& c) { return !c.empty(); })
        .def("__getitem__", [](const C& c, const std::string& id) { return c.at(id); })
        .def_prop_ro("include_superseded", &C::includeSuperseded)
        .def("filter", [](const C& c, nb::kwargs kwargs) {
            nb::dict criteria = nb::steal<nb::dict>(PyDict_Copy(kwargs.ptr()));
            const auto include = takeIncludeSuperseded(criteria);
            return c.filter(criteriaPredicate<T>(criteria), include);
        })
        .def("get", [](const C& c, nb::kwargs kwargs) {
            nb::dict criteria = nb::steal<nb::dict>(PyDict_Copy(kwargs.ptr()));
            const auto description = nb::cast<std::string>(nb::repr(kwargs));
            const auto include = takeIncludeSuperseded(criteria);
            return c.get(criteriaPredicate<T>(criteria), include, description);
        }, "The item matching all criteria, requiring a unique match. NotFoundError for no match, AmbiguousMatchError for several.")
        .def("__repr__", [](const C& c) { return "Collection(" + std::to_string(c.size()) + " entities)"; });
    return cls;
}

// A surface with its artifact copied, so it outlives the catalog.
struct SurfaceCopy { OpenDataArtifact artifact; bool authored; bool canonical; std::string volumeId; std::string id; };
template <class T> void metadata(nb::class_<T>& cls) { cls.def_prop_ro("raw", [](const T& v) { return pyJson(v.raw); }); }
template <class T> nb::dict byId(const std::vector<T>& values)
{
    nb::dict out;
    for (const auto& value : values) out[value.id.c_str()] = nb::cast(value);
    return out;
}
Catalog loadCatalogFrom(nb::handle source, nb::handle cacheDir, bool useCache, double maxAge, bool refresh, double timeout)
{
    const auto location = fspath(source);
    CatalogLoadOptions options;
    if (useCache) options.cacheFile = catalogCacheFile(location, cacheDir.is_none() ? std::filesystem::path{} : std::filesystem::path(fspath(cacheDir)));
    options.refresh = refresh;
    options.maxAgeSeconds = maxAge;
    options.timeoutSeconds = timeout;
    std::string warning;
    std::optional<Catalog> catalog;
    {
        nb::gil_scoped_release release;
        catalog = Catalog::load(location, options, &warning);
    }
    if (!warning.empty()) nb::module_::import_("warnings").attr("warn")(warning);
    return *catalog;
}
Catalog catalogFrom(nb::handle data, nb::handle source)
{
    return Catalog(parseOpenDataManifest(toJson(data, false).dump()), source.is_none() ? "" : fspath(source));
}
} // namespace

NB_MODULE(_catalog, m) {
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const DeclarationTypeError& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
        } catch (const NotFoundError& e) {
            nb::object error = nb::module_::import_("vc_reader.errors").attr("NotFoundError");
            PyErr_SetString(error.ptr(), e.what());
        } catch (const AmbiguousMatchError& e) {
            nb::object error = nb::module_::import_("vc_reader.errors").attr("AmbiguousMatchError")(e.what(), e.candidates);
            PyErr_SetObject(error.type().ptr(), error.ptr());
        }
    });

    // ---- records ----
    nb::class_<OpenDataReference>(m, "Reference").def_ro("type", &OpenDataReference::type).def_ro("id", &OpenDataReference::id);
    nb::class_<OpenDataCreation>(m, "Creation").def_ro("date", &OpenDataCreation::date)
        .def_ro("process", &OpenDataCreation::process).def_ro("derived_from", &OpenDataCreation::derivedFrom);
    nb::class_<OpenDataBounds>(m, "Bounds").def_ro("low", &OpenDataBounds::low).def_ro("high", &OpenDataBounds::high);
    auto coverage = nb::class_<OpenDataVolumeCoverage>(m, "VolumeCoverage");
    bindValue(coverage, {"volume_id", "bbox_min", "bbox_max", "overlap_ratio", "computed_at"});
    coverage.def_ro("volume_id", &OpenDataVolumeCoverage::volumeId)
        .def_prop_ro("bbox_min", [](const OpenDataVolumeCoverage& c) { return vecTuple(c.bbox.low); })
        .def_prop_ro("bbox_max", [](const OpenDataVolumeCoverage& c) { return vecTuple(c.bbox.high); })
        .def_ro("overlap_ratio", &OpenDataVolumeCoverage::overlapRatio)
        .def_prop_ro("computed_at", [](const OpenDataVolumeCoverage& c) { return noneIfEmpty(c.computedAt); });
    nb::class_<OpenDataSupersession>(m, "Supersession").def_ro("volume_id", &OpenDataSupersession::volumeId)
        .def_ro("date", &OpenDataSupersession::date).def_ro("reason", &OpenDataSupersession::reason);
    nb::class_<OpenDataLicense>(m, "License").def_ro("name", &OpenDataLicense::name).def_ro("url", &OpenDataLicense::url);
    auto access = nb::class_<OpenDataAccessRoot>(m, "AccessRoot");
    access.def_ro("type", &OpenDataAccessRoot::type).def_ro("url", &OpenDataAccessRoot::url).def_ro("usage", &OpenDataAccessRoot::usage);
    metadata(access);
    auto origin = nb::class_<OpenDataOrigin>(m, "Origin");
    origin.def_ro("path", &OpenDataOrigin::path)
        .def_prop_ro("access_roots", [](const OpenDataOrigin& o) { return tupleOf(o.accessRoots); })
        .def_prop_ro("copy_info", [](const OpenDataOrigin& o) { return o.raw.contains("copy_info") ? pyJson(o.raw["copy_info"]) : nb::none(); })
        .def("urls", [](const OpenDataOrigin& o) { return tupleOf(o.urls()); });
    metadata(origin);
    auto artifact = nb::class_<OpenDataArtifact>(m, "Artifact");
    artifact.def_ro("type", &OpenDataArtifact::type).def_ro("id", &OpenDataArtifact::id)
        .def_prop_ro("origins", [](const OpenDataArtifact& a) { return tupleOf(a.origins); })
        .def_ro("level", &OpenDataArtifact::sourceCoordinateLevel, "parameters.level: the source pyramid level a volume prediction ran on.")
        .def_prop_ro("grid_level", &OpenDataArtifact::gridLevel, "Source-volume pyramid level that this artifact's own level 0 lies on.")
        .def_ro("target_volume_id", &OpenDataArtifact::targetVolumeId).def_ro("model_id", &OpenDataArtifact::modelId)
        .def_prop_ro("render_level", [](const OpenDataArtifact& a) { return a.renderLevel.value_or(0); }, "Pyramid level of target_volume a render was computed from (0 when absent).")
        .def_ro("downscale", &OpenDataArtifact::downscale).def_ro("format", &OpenDataArtifact::format)
        .def_prop_ro("label_set", [](const OpenDataArtifact& a) { return noneIfEmpty(a.labelSet); })
        .def_ro("width", &OpenDataArtifact::width).def_ro("height", &OpenDataArtifact::height)
        .def_prop_ro("arrays", [](const OpenDataArtifact& a) { return tupleOf(a.arrays); })
        .def_prop_ro("supersedes", [](const OpenDataArtifact& a) { return tupleOf(a.supersedes); })
        .def_ro("registration", &OpenDataArtifact::registration).def_ro("annotated_on", &OpenDataArtifact::annotatedOn)
        .def_ro("mapping_coverage", &OpenDataArtifact::mappingCoverage).def_ro("creation", &OpenDataArtifact::creation)
        .def("url", [](const OpenDataArtifact& a, std::optional<std::string> scheme) { return a.location(scheme.value_or("")); }, nb::arg("scheme") = "https",
             "The preferred location: scheme 'https' (default; anonymous HTTP), 's3', or None to keep its own.")
        .def("urls", [](const OpenDataArtifact& a) { return tupleOf(a.urls()); }, "Every location as listed, in metadata order.")
        .def_prop_ro("path", [](const OpenDataArtifact& a) { return a.origins.empty() ? std::optional<std::string>{} : std::optional(a.origins[0].path); }, "Path of the first origin (relative to its access root).")
        .def_prop_ro("usage", [](const OpenDataArtifact& a) { return tupleOf(a.usage()); }, "Distinct access classes of the artifact's roots.")
        .def_prop_ro("parameters", [](const OpenDataArtifact& a) { return pyJson(a.parameters); })
        .def_prop_ro("properties", [](const OpenDataArtifact& a) { return pyJson(a.properties); })
        .def_prop_ro("creation_info", [](const OpenDataArtifact& a) { return a.creationInfo.empty() ? nb::none() : pyJson(a.creationInfo); })
        .def_static("from_json", [](nb::handle entry) { return parseCatalogArtifacts(nlohmann::json::array({toJson(entry, false)})).at(0); })
        .def("__repr__", [](const OpenDataArtifact& a) {
            std::string text = "Artifact('" + a.type + "'";
            for (const auto& item : a.parameters.items()) text += ", " + item.key() + "=" + item.value().dump();
            return text + ", path=" + (a.origins.empty() ? "None" : "'" + a.origins[0].path + "'") + ")";
        });
    metadata(artifact);
    auto scan = nb::class_<OpenDataScan>(m, "Scan");
    scan.def_ro("id", &OpenDataScan::id).def_prop_ro("long_id", [](const OpenDataScan& s) { return noneIfEmpty(s.longId); })
        .def_ro("creation", &OpenDataScan::creation)
        .def_ro("pixel_size_um", &OpenDataScan::pixelSizeUm).def_ro("energy_kev", &OpenDataScan::energyKeV)
        .def_ro("detector_distance_mm", &OpenDataScan::detectorDistanceMm)
        .def_prop_ro("location", [](const OpenDataScan& s) { return noneIfEmpty(s.location); })
        .def_prop_ro("name", [](const OpenDataScan& s) { return noneIfEmpty(s.name); }, "The facility's scan name (ESRF scanRadix, DLS scan_name).")
        .def_prop_ro("date", [](const OpenDataScan& s) { return noneIfEmpty(s.creation.date); })
        .def_prop_ro("artifacts", [](const OpenDataScan& s) { return tupleOf(s.artifacts); })
        .def("__repr__", [](const OpenDataScan& s) { return "Scan('" + (s.longId.empty() ? s.id : s.longId) + "')"; });
    metadata(scan);
    auto volume = nb::class_<OpenDataVolume>(m, "VolumeRecord");
    volume.def_ro("id", &OpenDataVolume::id).def_prop_ro("long_id", [](const OpenDataVolume& v) { return noneIfEmpty(v.longId); })
        .def_prop_ro("scan_id", [](const OpenDataVolume& v) { return noneIfEmpty(v.scanId); })
        .def_prop_ro("suffix", [](const OpenDataVolume& v) { return noneIfEmpty(v.suffix); }).def_ro("creation", &OpenDataVolume::creation)
        .def_ro("pixel_size_um", &OpenDataVolume::pixelSizeUm).def_ro("energy_kev", &OpenDataVolume::energyKeV)
        .def_ro("detector_distance_mm", &OpenDataVolume::detectorDistanceMm)
        .def_prop_ro("shape_zyx", [](const OpenDataVolume& v) -> nb::object {
            if (!v.shapeZYX) return nb::none();
            return nb::make_tuple((*v.shapeZYX)[0], (*v.shapeZYX)[1], (*v.shapeZYX)[2]);
        })
        .def_prop_ro("dtype", [](const OpenDataVolume& v) { return noneIfEmpty(v.dataFormat); })
        .def_prop_ro("artifacts", [](const OpenDataVolume& v) { return tupleOf(v.artifacts); })
        .def_prop_ro("z_direction_is_top_to_bottom", [](const OpenDataVolume& v) { return v.orientation.zTopToBottom; })
        .def_prop_ro("left_handed_coordinates", [](const OpenDataVolume& v) { return v.orientation.leftHandedCoordinates; })
        .def_prop_ro("text_side", [](const OpenDataVolume& v) { return noneIfEmpty(v.textSide); })
        .def_ro("superseded_by", &OpenDataVolume::supersededBy)
        .def_prop_ro("superseded_by_id", [](const OpenDataVolume& v) { return v.supersededBy ? std::optional(v.supersededBy->volumeId) : std::nullopt; })
        .def_prop_ro("is_superseded", [](const OpenDataVolume& v) { return v.supersededBy.has_value(); })
        .def_ro("license", &OpenDataVolume::license).def_ro("render_level", &OpenDataVolume::renderLevel)
        .def_ro("render_segments", &OpenDataVolume::renderSegments)
        .def_ro("window_u16_min", &OpenDataVolume::windowU16Min).def_ro("window_u16_max", &OpenDataVolume::windowU16Max)
        .def_prop_ro("properties", [](const OpenDataVolume& v) { return pyJson(v.properties); });
    metadata(volume);
    nb::class_<SurfaceCopy>(m, "CatalogSurface")
        .def_ro("artifact", &SurfaceCopy::artifact).def_ro("authored", &SurfaceCopy::authored).def_ro("canonical", &SurfaceCopy::canonical)
        .def_ro("volume_id", &SurfaceCopy::volumeId).def_ro("id", &SurfaceCopy::id);
    auto segment = nb::class_<OpenDataSegment>(m, "SegmentRecord");
    segment.def_ro("id", &OpenDataSegment::id).def_prop_ro("long_id", [](const OpenDataSegment& s) { return noneIfEmpty(s.longId); })
        .def_prop_ro("suffix", [](const OpenDataSegment& s) { return noneIfEmpty(s.suffix); })
        .def_ro("original_volume_id", &OpenDataSegment::originalVolumeId).def_ro("creation", &OpenDataSegment::creation)
        .def_ro("width", &OpenDataSegment::width).def_ro("height", &OpenDataSegment::height)
        .def_prop_ro("artifacts", [](const OpenDataSegment& s) { return tupleOf(s.artifacts); })
        .def("artifacts_of", [](const OpenDataSegment& s, nb::args types, nb::kwargs kwargs) {
            std::vector<std::string> wanted;
            for (nb::handle type : types) wanted.push_back(nb::cast<std::string>(type));
            nb::list out;
            for (const auto* a : s.artifactsOf(wanted, kwargs.contains("target_volume") ? idOfAny(kwargs["target_volume"]) : std::nullopt)) out.append(nb::cast(*a));
            return out;
        })
        .def_prop_ro("original_volume_downscale", [](const OpenDataSegment& s) { return std::int64_t(s.originalVolumeDownscale); })
        .def_prop_ro("coverage", [](const OpenDataSegment& s) {
            nb::dict out;
            for (const auto& c : s.volumeCoverage) out[c.volumeId.c_str()] = nb::cast(c);
            return out;
        })
        .def_prop_ro("render", [](const OpenDataSegment& s) { return pyJson(s.render); })
        .def_prop_ro("properties", [](const OpenDataSegment& s) { return pyJson(s.properties); });
    metadata(segment);
    auto sample = nb::class_<OpenDataSample>(m, "SampleRecord");
    sample.def_ro("id", &OpenDataSample::id)
        .def_prop_ro("type", [](const OpenDataSample& s) { return noneIfEmpty(s.type); })
        .def_prop_ro("description", [](const OpenDataSample& s) { return noneIfEmpty(s.description); })
        .def_prop_ro("display_name", [](const OpenDataSample& s) { return s.properties.contains("description") ? pyJson(s.properties["description"]) : nb::none(); })
        .def_prop_ro("aliases", [](const OpenDataSample& s) { return tupleOf(communityAliases(s.id)); })
        .def_prop_ro("scans", [](const OpenDataSample& s) { return byId(s.scans); })
        .def_prop_ro("volumes", [](const OpenDataSample& s) { return byId(s.volumes); })
        .def_prop_ro("segments", [](const OpenDataSample& s) { return byId(s.segments); })
        .def_prop_ro("artifacts", [](const OpenDataSample& s) { return tupleOf(s.artifacts); })
        .def("surfaces", [](const OpenDataSample& s, const OpenDataSegment& seg) {
            std::vector<SurfaceCopy> copies;
            for (const auto& surface : s.surfaces(seg)) copies.push_back({*surface.artifact, surface.authored, surface.canonical, surface.volumeId, surface.id});
            return copies;
        })
        .def_prop_ro("properties", [](const OpenDataSample& s) { return pyJson(s.properties); });
    metadata(sample);
    auto model = nb::class_<OpenDataModel>(m, "Model");
    model.def_ro("id", &OpenDataModel::id).def_prop_ro("long_id", [](const OpenDataModel& v) { return noneIfEmpty(v.longId); })
        .def_prop_ro("name", [](const OpenDataModel& v) { return noneIfEmpty(v.suffix); }).def_ro("creation", &OpenDataModel::creation)
        .def_prop_ro("task", [](const OpenDataModel& v) { return noneIfEmpty(v.task); })
        .def_prop_ro("architecture", [](const OpenDataModel& v) { return noneIfEmpty(v.architecture); })
        .def_prop_ro("properties", [](const OpenDataModel& v) { return pyJson(v.properties); })
        .def("__repr__", [](const OpenDataModel& v) { return "Model('" + v.id + "', name='" + v.suffix + "', task='" + v.task + "')"; });
    metadata(model);

    // ---- affines ----
    auto affine = nb::class_<VolumeTransform>(m, "Affine", "p' = R @ p + t on column vectors [x, y, z, 1]. matrix holds the three rows of [R | t]. "
                                                            "Compose with @: (a @ b).apply(p) == a.apply(b.apply(p)).");
    bindValue(affine, {"matrix", "derivation_path"});
    affine.def_static("from_rows", [](nb::handle rows, std::optional<std::string> derivation) {
            return VolumeTransform{Affine::fromJson(toJson(rows)), derivation.value_or("")};
        }, nb::arg("rows"), nb::arg("derivation_path") = nb::none())
        .def_static("identity", [] { return VolumeTransform{}; })
        .def_static("scale", [](double s) { return VolumeTransform{Affine::scale(s), {}}; }, "Uniform scale about the origin. Pyramid levels use block_reduced.")
        .def_static("block_reduced", [](double s) { return VolumeTransform{Affine::blockReduced(s), {}}; },
                    "Voxels of a pyramid level s times coarser than a reference, block-reduced from it: voxel i is centred at s * i + (s - 1) / 2 reference voxels. "
                    "Affine.block_reduced(2**level) maps level voxels to level 0.")
        .def_prop_ro("matrix", [](const VolumeTransform& a) { return matrixOf(a.forward); })
        .def_prop_ro("derivation_path", [](const VolumeTransform& a) { return noneIfEmpty(a.derivationPath); })
        .def_prop_ro("linear", [](const VolumeTransform& a) {
            const auto& f = a.forward;
            return nb::make_tuple(nb::make_tuple(f(0, 0), f(0, 1), f(0, 2)), nb::make_tuple(f(1, 0), f(1, 1), f(1, 2)), nb::make_tuple(f(2, 0), f(2, 1), f(2, 2)));
        })
        .def_prop_ro("translation", [](const VolumeTransform& a) { return nb::make_tuple(a.forward(0, 3), a.forward(1, 3), a.forward(2, 3)); })
        .def("apply", [](const VolumeTransform& a, const std::array<double, 3>& point) { return vecTuple(a.forward.apply(point)); })
        .def("__matmul__", [](const VolumeTransform& a, const VolumeTransform& b) { return VolumeTransform{a.forward * b.forward, {}}; })
        .def("inverse", [](const VolumeTransform& a) { return VolumeTransform{a.forward.inverse(), {}}; });

    // ---- descriptors ----
    auto frame = nb::class_<Frame>(m, "Frame");
    bindValue(frame, {"sample_id", "volume_id", "level"});
    frame.def_ro("sample_id", &Frame::sampleId).def_ro("volume_id", &Frame::volumeId).def_ro("level", &Frame::level);
    auto levelRequest = nb::class_<LevelRequest>(m, "LevelRequest", "Requested reference resolution; a negative sublevel means unavailable.");
    bindValue(levelRequest, {"reference", "sublevel"});
    levelRequest.def_ro("reference", &LevelRequest::reference).def_ro("sublevel", &LevelRequest::sublevel);
    auto selection = nb::class_<Selection>(m, "Selection");
    bindValue(selection, {"array_path", "level", "series", "channels", "time", "corresponding"});
    selection.def_ro("array_path", &Selection::arrayPath).def_ro("level", &Selection::level).def_ro("series", &Selection::series)
        .def_prop_ro("channels", [](const Selection& s) { return optionalTuple(s.channels); })
        .def_ro("time", &Selection::time).def_ro("corresponding", &Selection::corresponding)
        .def("spatial", &Selection::spatial, "The same grid whatever channels and time it shows.");
    auto storage = nb::class_<Storage>(m, "Storage");
    bindValue(storage, {"location", "format", "locations", "alternatives", "selection", "pyramid", "grid_level"});
    storage.def_ro("location", &Storage::location).def_ro("format", &Storage::format)
        .def_prop_ro("locations", [](const Storage& s) { return tupleOf(s.locations); })
        .def_prop_ro("alternatives", [](const Storage& s) { return tupleOf(s.alternatives); })
        .def_ro("selection", &Storage::selection).def_ro("pyramid", &Storage::pyramid).def_ro("grid_level", &Storage::gridLevel);
    auto plane = nb::class_<Plane>(m, "Plane", "XY pixel centers and one-voxel normal footprint in volume level-0 XYZ.");
    bindValue(plane, {"origin", "basis_x", "basis_y", "thickness"});
    plane.def_prop_ro("origin", [](const Plane& p) { return vecTuple(p.origin); })
        .def_prop_ro("basis_x", [](const Plane& p) { return vecTuple(p.basisX); })
        .def_prop_ro("basis_y", [](const Plane& p) { return vecTuple(p.basisY); })
        .def_prop_ro("thickness", [](const Plane& p) { return vecTuple(p.thickness); });
    auto volumeRegistration = nb::class_<VolumeRegistration>(m, "VolumeRegistration");
    bindValue(volumeRegistration, {"sample_id", "volume_id", "affine", "plane", "mode", "storage_frame", "declared_grid", "origin", "storage_units", "volume_spacing_um"});
    volumeRegistration.def_ro("sample_id", &VolumeRegistration::sampleId).def_ro("volume_id", &VolumeRegistration::volumeId)
        .def_ro("affine", &VolumeRegistration::affine).def_ro("plane", &VolumeRegistration::plane).def_ro("mode", &VolumeRegistration::mode)
        .def_ro("storage_frame", &VolumeRegistration::storageFrame).def_ro("declared_grid", &VolumeRegistration::declaredGrid)
        .def_ro("origin", &VolumeRegistration::origin).def_ro("storage_units", &VolumeRegistration::storageUnits)
        .def_prop_ro("volume_spacing_um", [](const VolumeRegistration& r) { return optionalVec(r.volumeSpacingUm); });
    auto valueFrame = nb::class_<ValueFrame>(m, "ValueFrame");
    bindValue(valueFrame, {"kind", "sample_id", "volume_id", "affine", "level", "name", "units"});
    valueFrame.def_ro("kind", &ValueFrame::kind).def_ro("sample_id", &ValueFrame::sampleId).def_ro("volume_id", &ValueFrame::volumeId)
        .def_ro("affine", &ValueFrame::affine).def_ro("level", &ValueFrame::level).def_ro("name", &ValueFrame::name).def_ro("units", &ValueFrame::units);
    auto encoding = nb::class_<Encoding>(m, "Encoding");
    bindValue(encoding, {"kind", "pairs", "dtype", "fallback"});
    encoding.def_ro("kind", &Encoding::kind).def_ro("dtype", &Encoding::dtype)
        .def_prop_ro("pairs", [](const Encoding& e) {
            nb::list out;
            for (const auto& pair : e.pairs) {
                nb::object key;
                if (e.kind == "integer") {
                    key = wide(pair.key.at(0));
                } else {
                    nb::list components;
                    for (const auto& c : pair.key) components.append(wide(c));
                    key = nb::tuple(components);
                }
                out.append(nb::make_tuple(key, wide(pair.value)));
            }
            return nb::tuple(out);
        })
        .def_prop_ro("fallback", [](const Encoding& e) { return e.fallback ? wide(*e.fallback) : nb::none(); });
    auto surface = nb::class_<Surface>(m, "Surface");
    bindValue(surface, {"id", "geometry", "sample_id", "volume_id", "axes", "component_order", "xyz_to_volume", "pixel_to_uv", "spacing_um", "mask", "normals", "tifxyz_mask"});
    const auto sharedOrNone = [](const std::optional<Shared<Source>>& value) { return value ? nb::cast(**value) : nb::none(); };
    surface.def_ro("id", &Surface::id).def_ro("geometry", &Surface::geometry).def_ro("sample_id", &Surface::sampleId).def_ro("volume_id", &Surface::volumeId)
        .def_prop_ro("axes", [](const Surface& s) { return optionalTuple(s.axes); })
        .def_prop_ro("component_order", [](const Surface& s) { return optionalTuple(s.componentOrder); })
        .def_ro("xyz_to_volume", &Surface::xyzToVolume)
        .def_prop_ro("pixel_to_uv", [](const Surface& s) { return pixelAffineTuple(s.pixelToUv); })
        .def_prop_ro("spacing_um", [](const Surface& s) { return optionalVec(s.spacingUm); })
        .def_prop_ro("mask", [sharedOrNone](const Surface& s) { return sharedOrNone(s.mask); })
        .def_prop_ro("normals", [sharedOrNone](const Surface& s) { return sharedOrNone(s.normals); })
        .def_ro("tifxyz_mask", &Surface::tifxyzMask);
    auto surfaceRegistration = nb::class_<SurfaceRegistration>(m, "SurfaceRegistration");
    bindValue(surfaceRegistration, {"surface", "pixel_to_uv", "origin", "declared_grid"});
    surfaceRegistration.def_ro("surface", &SurfaceRegistration::surface)
        .def_prop_ro("pixel_to_uv", [](const SurfaceRegistration& r) { return pixelAffineTuple(r.pixelToUv); })
        .def_ro("origin", &SurfaceRegistration::origin).def_ro("declared_grid", &SurfaceRegistration::declaredGrid)
        .def_prop_ro("sample_id", [](const SurfaceRegistration& r) { return r.surface.sampleId; })
        .def_prop_ro("volume_id", [](const SurfaceRegistration& r) { return r.surface.volumeId; });
    auto companion = nb::class_<Companion>(m, "Companion");
    bindValue(companion, {"name", "source", "same_grid"});
    companion.def_ro("name", &Companion::name).def_prop_ro("source", [](const Companion& c) { return *c.source; }).def_ro("same_grid", &Companion::sameGrid);
    auto queryContext = nb::class_<QueryContext>(m, "QueryContext");
    bindValue(queryContext, {"reference", "relation", "transform", "coverage"});
    queryContext.def_ro("reference", &QueryContext::reference).def_ro("relation", &QueryContext::relation)
        .def_ro("transform", &QueryContext::transform).def_ro("coverage", &QueryContext::coverage);
    auto source = nb::class_<Source>(m, "Source");
    bindValue(source, {"id", "storage", "role", "axes", "registration", "component_order", "value_frame", "normalize", "encoding", "companions", "provenance",
                       "artifact_id", "artifact_kind", "sample_id", "owner_id", "model_id", "label_set", "supersedes", "coverage", "derived", "shape", "dtype", "fill_value", "query"});
    source.def_ro("id", &Source::id).def_ro("storage", &Source::storage).def_ro("role", &Source::role)
        .def_prop_ro("axes", [](const Source& s) { return optionalTuple(s.axes); })
        .def_ro("registration", &Source::registration)
        .def_prop_ro("component_order", [](const Source& s) { return optionalTuple(s.componentOrder); })
        .def_ro("value_frame", &Source::valueFrame).def_ro("normalize", &Source::normalize).def_ro("encoding", &Source::encoding)
        .def_prop_ro("companions", [](const Source& s) { return tupleOf(s.companions); })
        .def_prop_ro("provenance", [](const Source& s) { return pyJson(*s.provenance); })
        .def_ro("artifact_id", &Source::artifactId).def_ro("artifact_kind", &Source::artifactKind).def_ro("sample_id", &Source::sampleId)
        .def_ro("owner_id", &Source::ownerId).def_ro("model_id", &Source::modelId).def_ro("label_set", &Source::labelSet)
        .def_prop_ro("supersedes", [](const Source& s) { return tupleOf(s.supersedes); })
        .def_prop_ro("coverage", [](const Source& s) { return tupleOf(s.coverage); })
        .def_ro("derived", &Source::derived)
        .def_prop_ro("shape", [](const Source& s) { return optionalTuple(s.shape); })
        .def_ro("dtype", &Source::dtype)
        .def_prop_ro("fill_value", [](const Source& s) { return s.fillValue.is_null() ? nb::none() : pyJson(s.fillValue); })
        .def_ro("query", &Source::query)
        .def_prop_ro("unresolved_fields", [](const Source& s) { return tupleOf(s.unresolvedFields()); },
                     "What a reader still needs and nobody declared; no I/O.")
        .def("with_selection", [](const Source& s, nb::kwargs kwargs) { return s.withSelection(fromPython<Selection>({}, nb::borrow<nb::args>(nb::tuple().ptr()), kwargs)); },
             "Replace storage selection; the declaration's original grid stays fixed.")
        .def("to_dict", [](const Source& s) { return pyJson(s.toDict()); })
        .def_static("from_dict", [](nb::handle value) { return Source::fromDict(toJson(value, false)); })
        .def("open", [](const Source& s, nb::kwargs options) { return sourceReader(s, options); }, "Open a source for spatial crops.");

    // ---- browsing ----
    nb::class_<Pairing>(m, "Pairing").def_ro("label_id", &Pairing::labelId).def_ro("image_id", &Pairing::imageId)
        .def_prop_ro("pixel_affine", [](const Pairing& p) { return pixelAffineTuple(p.pixelAffine); }).def_ro("evidence", &Pairing::evidence)
        .def("__getstate__", [](const Pairing& p) { return nb::make_tuple(p.labelId, p.imageId, p.pixelAffine, p.evidence); })
        .def("__setstate__", [](Pairing& self, const std::tuple<std::string, std::string, std::optional<PixelAffine>, std::string>& state) {
            new (&self) Pairing{std::get<0>(state), std::get<1>(state), std::get<2>(state), std::get<3>(state)};
        });
    auto scrolls = bindCollection<ScrollSource>(m, "ScrollCollection");
    auto volumes = bindCollection<Volume>(m, "VolumeCollection");
    bindCollection<Segment>(m, "SegmentCollection");
    auto assets = bindCollection<Asset>(m, "AssetCollection");
    bindCollection<VolumeMapping>(m, "VolumeMappingCollection");
    bindCollection<ResolutionCandidate>(m, "ResolutionCollection");
    scrolls.def_prop_ro("volumes", &volumesOf, "Aggregated; retains each owning scroll even when volume IDs coincide.");
    volumes.def("at_resolution", [](const Collection<Volume>& c, double um, double relTol) { return atResolution(c, um, relTol); },
                nb::arg("um"), nb::kw_only(), nb::arg("rel_tol") = kDefaultResolutionRelTol);
    assets.def("ink", [](const Collection<Asset>& c, nb::kwargs kwargs) {
            nb::dict criteria = nb::steal<nb::dict>(PyDict_Copy(kwargs.ptr()));
            const auto description = nb::cast<std::string>(nb::repr(kwargs));
            const auto include = takeIncludeSuperseded(criteria);
            return c.filter(&isInkLabel).get(criteriaPredicate<Asset>(criteria), include, description);
        }, "The matching ink label, raising for missing or ambiguous matches. filter(artifact_kind='ink-labels') browses several.")
        .def("zarr", [](const Collection<Asset>& c, nb::kwargs kwargs) {
            nb::dict criteria = nb::steal<nb::dict>(PyDict_Copy(kwargs.ptr()));
            const auto description = nb::cast<std::string>(nb::repr(kwargs));
            const auto include = takeIncludeSuperseded(criteria);
            return c.filter(&isZarrAsset).get(criteriaPredicate<Asset>(criteria), include, description);
        }, "The matching Zarr asset, raising for missing or ambiguous matches. filter(format='zarr') browses several.");

    auto scroll = nb::class_<ScrollSource>(m, "ScrollSource", "A sample with optional direct-volume context; navigation requests no arrays.");
    bindContext(scroll);
    auto asset = nb::class_<Asset>(m, "Asset");
    auto volumeView = nb::class_<Volume>(m, "Volume");
    bindContext(volumeView);
    auto segmentView = nb::class_<Segment>(m, "Segment");
    bindContext(segmentView);
    auto mapping = nb::class_<VolumeMapping>(m, "VolumeMapping");
    auto level = nb::class_<VolumeLevel>(m, "VolumeLevel");
    auto candidate = nb::class_<ResolutionCandidate>(m, "ResolutionCandidate");
    auto catalogClass = nb::class_<Catalog>(m, "Catalog");

    scroll.def("__init__", [](ScrollSource* self, const std::string& name, nb::handle volume, nb::handle catalog) {
            const Catalog owner = catalog.is_none() ? loadCatalogFrom(nb::str(std::string(kDefaultManifestUrl).c_str()), nb::none(), true, 3600., false, 60.) : nb::cast<Catalog>(catalog);
            new (self) ScrollSource(owner.scroll(name, idOfAny(volume)));
        }, nb::arg("scroll"), nb::arg("volume") = nb::none(), nb::kw_only(), nb::arg("catalog") = nb::none())
        .def_prop_ro("id", &ScrollSource::id)
        .def_prop_ro("catalog", [](const ScrollSource& s) { return Catalog(s.index()->published() ? s.index()->published() : s.index()); })
        .def_prop_ro("images", &ScrollSource::images)
        .def_prop_ro("record", [](const ScrollSource& s) { return s.record(); })
        .def_prop_ro("raw", [](const ScrollSource& s) { return pyJson(s.record().raw); })
        .def_prop_ro("properties", [](const ScrollSource& s) { return pyJson(s.record().properties); })
        .def_prop_ro("description", [](const ScrollSource& s) { return noneIfEmpty(s.record().description); })
        .def_prop_ro("type", [](const ScrollSource& s) { return noneIfEmpty(s.record().type); })
        .def_prop_ro("display_name", [](const ScrollSource& s) { const auto& p = s.record().properties; return p.contains("description") ? pyJson(p["description"]) : nb::none(); })
        .def_prop_ro("aliases", [](const ScrollSource& s) { return tupleOf(communityAliases(s.id())); })
        .def_prop_ro("scans", [](const ScrollSource& s) { return byId(s.record().scans); })
        .def_prop_ro("volume", &ScrollSource::volume).def_prop_ro("volumes", &ScrollSource::volumes).def_prop_ro("segments", &ScrollSource::segments)
        .def("on_volume", [](const ScrollSource& s, nb::handle volume) { return s.onVolume(*idOfAny(volume)); })
        .def("transform", [](const ScrollSource& s, nb::handle from, nb::handle to) { return s.transform(*idOfAny(from), *idOfAny(to)); },
             "Level-0 [x, y, z] of one volume -> level-0 of another; None when they are not registered to each other.")
        .def("with_source", [](const ScrollSource& s, const Source& source, const std::string& kind, nb::handle volume, nb::handle segment, nb::handle image, nb::handle label,
                               bool sameGrid, nb::handle pixelAffine) {
            SourceAttachment attachment;
            attachment.kind = kind;
            attachment.volume = idOfAny(volume);
            attachment.segment = idOfAny(segment);
            attachment.image = idOfAny(image);
            attachment.label = idOfAny(label);
            attachment.sameGrid = sameGrid;
            if (!pixelAffine.is_none()) {
                nlohmann::json rows = toJson(pixelAffine);
                if (!rows.is_array() || rows.size() != 2 || rows[0].size() != 3 || rows[1].size() != 3) throw std::invalid_argument("pixel-to-UV transform must have shape 2x3");
                attachment.pixelAffine = rows.get<PixelAffine>();
            }
            return s.withSource(source, attachment);
        }, nb::arg("source"), nb::kw_only(), nb::arg("kind"), nb::arg("volume") = nb::none(), nb::arg("segment") = nb::none(), nb::arg("image") = nb::none(),
           nb::arg("label") = nb::none(), nb::arg("same_grid") = false, nb::arg("pixel_affine") = nb::none(),
           "Attach a descriptor to a new view without mutating published metadata. kind is volume/image/label/representation/geometry/artifact. "
           "Pairing requires image= (for labels) or label= (for images) and exactly one of same_grid=True or a 2x3 label-XY-to-image-XY pixel_affine.")
        .def("to_dict", [](const ScrollSource& s) { return pyJson(s.toDict()); })
        .def_static("from_dict", [](nb::handle value) { return ScrollSource::fromDict(toJson(value, false)); })
        .def("__getstate__", [](const ScrollSource& s) { return pyJson(s.toDict()); })
        .def("__setstate__", [](ScrollSource& self, nb::handle state) { new (&self) ScrollSource(ScrollSource::fromDict(toJson(state, false))); })
        .def("__repr__", [](const ScrollSource& s) {
            return "ScrollSource('" + s.id() + "', volume=" + (s.volumeContext() ? "'" + *s.volumeContext() + "'" : std::string("None")) + ")";
        });

    volumeView.def_prop_ro("id", &Volume::id).def_prop_ro("sample_id", &Volume::sampleId).def_prop_ro("scroll", &Volume::scroll)
        .def_prop_ro("record", [](const Volume& v) { const auto* r = v.record(); return r ? nb::cast(*r) : nb::none(); })
        .def_prop_ro("images", &Volume::volumeImages)
        .def_prop_ro("source", &Volume::source)
        .def_prop_ro("segments", &Volume::segments)
        .def_prop_ro("pixel_size_um", &Volume::pixelSizeUm)
        .def_prop_ro("is_superseded", &Volume::isSuperseded)
        .def_prop_ro("shape_zyx", [](const Volume& v) -> nb::object {
            const auto* r = v.record();
            return r && r->shapeZYX ? nb::object(nb::make_tuple((*r->shapeZYX)[0], (*r->shapeZYX)[1], (*r->shapeZYX)[2])) : nb::none();
        })
        .def_prop_ro("long_id", [](const Volume& v) { const auto* r = v.record(); return r ? noneIfEmpty(r->longId) : std::nullopt; })
        .def_prop_ro("energy_kev", [](const Volume& v) { const auto* r = v.record(); return r ? r->energyKeV : std::nullopt; })
        .def_prop_ro("scan_id", [](const Volume& v) { const auto* r = v.record(); return r ? noneIfEmpty(r->scanId) : std::nullopt; })
        .def("__getattr__", [](nb::handle self, const std::string& name) -> nb::object {
            nb::object record = self.attr("record");
            if (name.starts_with("_") || record.is_none()) throw nb::attribute_error(name.c_str());
            return nb::getattr(record, name.c_str());
        })
        .def("level", [](const Volume& v, std::int64_t l) { return VolumeLevel(v, l); })
        .def("open", [](const Volume& v, std::int64_t l, nb::kwargs options) { return sourceReader(VolumeLevel(v, l).source(), options); }, nb::kw_only(), nb::arg("level") = 0, nb::arg("options"))
        .def("open_aligned", [](const Volume& v, nb::handle targets, std::int64_t l, nb::handle backend, nb::kwargs options) {
            nb::dict sources;
            if (!targets.is_none()) {
                nb::object items = targets.attr("items")();
                for (nb::handle item : items) sources[item[0]] = nb::isinstance<Asset>(item[1]) ? nb::object(item[1].attr("source")) : nb::borrow(item[1]);
            }
            return nb::module_::import_("vc_reader.sampling").attr("open_sample")(VolumeLevel(v, l), sources, nb::arg("backend") = backend, **options);
        }, nb::arg("targets") = nb::none(), nb::kw_only(), nb::arg("level") = 0, nb::arg("backend") = "cpu", nb::arg("options"))
        .def("__getstate__", [](const Volume& v) { return nb::make_tuple(v.scroll(), v.sampleId(), v.id()); })
        .def("__setstate__", [](Volume& self, const std::tuple<ScrollSource, std::string, std::string>& state) {
            new (&self) Volume(std::get<0>(state).index(), std::get<1>(state), std::get<2>(state));
        })
        .def("__repr__", [](const Volume& v) { return "Volume('" + v.sampleId() + "', '" + v.id() + "')"; });

    segmentView.def_prop_ro("id", &Segment::id).def_prop_ro("sample_id", &Segment::sampleId).def_prop_ro("images", &Segment::images)
        .def_prop_ro("record", [](const Segment& s) { const auto* r = s.record(); return r ? nb::cast(*r) : nb::none(); })
        .def_prop_ro("scroll", &Segment::scroll).def_prop_ro("volumes", &Segment::volumes)
        .def("on_volume", [](const Segment& s, nb::handle volume) { return s.onVolume(*idOfAny(volume)); })
        .def("__getstate__", [](const Segment& s) { return nb::make_tuple(s.scroll(), s.sampleId(), s.id(), s.volumeContext()); })
        .def("__setstate__", [](Segment& self, const std::tuple<ScrollSource, std::string, std::string, std::optional<std::string>>& state) {
            new (&self) Segment(std::get<0>(state).index(), std::get<1>(state), std::get<2>(state), std::get<3>(state));
        })
        .def("__repr__", [](const Segment& s) {
            return "Segment('" + s.id() + "', volume=" + (s.volumeContext() ? "'" + *s.volumeContext() + "'" : std::string("None")) + ")";
        });

    asset.def_prop_ro("id", &Asset::id).def_prop_ro("source", [](const Asset& a) { return a.source(); })
        .def("__getattr__", [](nb::handle self, const std::string& name) -> nb::object {
            if (name.starts_with("_")) throw nb::attribute_error(name.c_str());
            return nb::getattr(nb::object(self.attr("source")), name.c_str());
        })
        .def_prop_ro("kind", &Asset::kind)
        .def_prop_ro("format", [](const Asset& a) { return a.source().storage.format; })
        .def_prop_ro("is_superseded", &Asset::isSuperseded)
        .def_prop_ro("volume", &Asset::volume).def_prop_ro("segment", &Asset::segment).def_prop_ro("scroll", &Asset::scroll)
        .def_prop_ro("images", &Asset::images).def_prop_ro("labels", &Asset::labels)
        .def_prop_ro("surface_volumes", &Asset::surfaceVolumes).def_prop_ro("masks", &Asset::masks)
        .def_prop_ro("depth_semantics", &Asset::depthSemantics).def_prop_ro("mappable_volumes", &Asset::mappableVolumes)
        .def("summary", [](const Asset& a) { return pyJson(a.summary()); },
             "JSON-ready source ownership, volume mappings and unresolved fields. No storage headers or image arrays are requested.")
        .def("open", [](const Asset& a, std::optional<std::int64_t> l, nb::kwargs options) {
            return sourceReader(l ? selectLevel(a.source(), *l) : a.source(), options);
        }, nb::kw_only(), nb::arg("level") = nb::none(), nb::arg("options"))
        .def("open_paired", [](const Asset& a, std::optional<Asset> image, std::int64_t l, nb::handle layers, nb::handle backend, nb::kwargs options) {
            const auto pairing = a.pairingWith(image);
            const auto chosen = image ? *image : Asset(a.index(), pairing.imageId);
            return nb::module_::import_("vc_reader.paired").attr("PairedReader")(a.source(), chosen.source(), pairing, nb::arg("level") = l, nb::arg("layers") = layers,
                                                                                  nb::arg("backend") = backend, **options);
        }, nb::kw_only(), nb::arg("image") = nb::none(), nb::arg("level") = 0, nb::arg("layers") = nb::slice(nb::none(), nb::none(), nb::none()), nb::arg("backend") = "cpu", nb::arg("options"))
        .def("__getstate__", [](const Asset& a) {
            if (a.kind() == "mask") throw nb::type_error("a mask is pickled through its source");
            return nb::make_tuple(a.scroll(), a.id());
        })
        .def("__setstate__", [](Asset& self, const std::tuple<ScrollSource, std::string>& state) {
            new (&self) Asset(std::get<0>(state).index(), std::get<1>(state));
        })
        .def("__repr__", [](const Asset& a) {
            std::string text = "Asset('" + a.id() + "', kind='" + a.kind() + "'";
            if (const auto v = a.volume()) text += ", source_volume='" + v->id() + "'";
            if (a.kind() == "label") {
                text += ", mappable_volumes=[";
                bool first = true;
                for (const auto& m : a.mappableVolumes().visible()) {
                    text += (first ? "'" : ", '") + m.id() + "'";
                    first = false;
                }
                text += "]";
            }
            return text + ")";
        });

    mapping.def_ro("volume", &VolumeMapping::volume).def_ro("transform", &VolumeMapping::transform).def_ro("coverage", &VolumeMapping::coverage)
        .def_prop_ro("required_operations", [](const VolumeMapping& v) { return tupleOf(v.requiredOperations); })
        .def_prop_ro("id", &VolumeMapping::id).def_prop_ro("is_superseded", [](const VolumeMapping& v) { return v.volume.isSuperseded(); });

    level.def_ro("volume", &VolumeLevel::volume).def_ro("level", &VolumeLevel::level)
        .def_prop_ro("source", &VolumeLevel::source)
        .def("__getstate__", [](const VolumeLevel& l) { return nb::make_tuple(l.volume, l.level); })
        .def("__setstate__", [](VolumeLevel& self, const std::tuple<Volume, std::int64_t>& state) { new (&self) VolumeLevel(std::get<0>(state), std::get<1>(state)); })
        .def("open", [](const VolumeLevel& l, nb::kwargs options) { return sourceReader(l.source(), options); });

    nb::class_<OpenedGrid>(m, "OpenedGrid");
    const auto verifiedBy = [](const ResolutionCandidate& c, nb::handle reader) { return c.verifiedAgainst(openedGrid(reader)); };
    candidate.def_ro("volume", &ResolutionCandidate::volume).def_ro("source", &ResolutionCandidate::source).def_ro("level", &ResolutionCandidate::level)
        .def_ro("estimated_resolution_um", &ResolutionCandidate::estimatedResolutionUm).def_ro("verified", &ResolutionCandidate::verified)
        .def_ro("selection", &ResolutionCandidate::selection)
        .def_prop_ro("spacing_um", [](const ResolutionCandidate& c) { return optionalVec(c.spacingUm); })
        .def_prop_ro("shape", [](const ResolutionCandidate& c) -> nb::object {
            return c.shape ? nb::object(nb::make_tuple((*c.shape)[0], (*c.shape)[1], (*c.shape)[2])) : nb::none();
        })
        .def_prop_ro("id", &ResolutionCandidate::id)
        .def_prop_ro("scroll", [](const ResolutionCandidate& c) { return c.volume.scroll(); })
        .def("resolve", [verifiedBy](const ResolutionCandidate& c, nb::kwargs options) {
            nb::object reader = sourceReader(selectLevel(c.source, c.level), options);
            try {
                auto out = verifiedBy(c, reader);
                reader.attr("close")();
                return out;
            } catch (...) {
                reader.attr("close")();
                throw;
            }
        })
        .def("open", [verifiedBy](const ResolutionCandidate& c, nb::kwargs options) {
            nb::object reader = sourceReader(selectLevel(c.source, c.level), options);
            try {
                (void)verifiedBy(c, reader);
                return reader;
            } catch (...) {
                reader.attr("close")();
                throw;
            }
        });

    catalogClass.def("__init__", [](Catalog* self, nb::handle data, nb::handle source) { new (self) Catalog(catalogFrom(data, source)); }, nb::arg("data"), nb::arg("source") = nb::none())
        .def_static("load", &loadCatalogFrom, nb::arg("source") = std::string(kDefaultManifestUrl), nb::kw_only(), nb::arg("cache_dir") = nb::none(), nb::arg("use_cache") = true,
                    nb::arg("max_age") = 3600., nb::arg("refresh") = false, nb::arg("timeout") = 60.,
                    "Load from an http(s) URL (cached on disk) or a local file (plain or gzipped JSON).")
        .def_static("cache_file", [](nb::handle source, nb::handle cacheDir) {
            return catalogCacheFile(fspath(source), cacheDir.is_none() ? std::filesystem::path{} : std::filesystem::path(fspath(cacheDir)));
        }, nb::arg("source") = std::string(kDefaultManifestUrl), nb::arg("cache_dir") = nb::none(), "Where load caches a manifest.")
        .def_prop_ro("raw", [](const Catalog& c) { return pyJson(c.manifest().raw); })
        .def_prop_ro("source", [](const Catalog& c) { return noneIfEmpty(c.source()); })
        .def_prop_ro("models", [](const Catalog& c) { return byId(c.manifest().models); })
        .def_prop_ro("samples", &Catalog::samples).def_prop_ro("scrolls", &Catalog::scrolls)
        .def("__reduce__", [](nb::handle self) { return nb::make_tuple(self.type(), nb::make_tuple(self.attr("raw"), self.attr("source"))); })
        .def("__repr__", [](const Catalog& c) {
            return "Catalog(samples=" + std::to_string(c.manifest().samples.size()) + ", models=" + std::to_string(c.manifest().models.size()) +
                   ", source=" + (c.source().empty() ? std::string("None") : "'" + c.source() + "'") + ")";
        });

    struct ArtifactTypes {};
    nb::class_<ArtifactTypes>(m, "ArtifactType", "Artifact type strings documented by metadata.schema.json. New types appear over time; unknown types are kept as plain artifacts.")
        .def_prop_ro_static("PHOTO", [](nb::handle) { return artifact_type::photo; })
        .def_prop_ro_static("PHOTO_MASK", [](nb::handle) { return artifact_type::photoMask; })
        .def_prop_ro_static("OME_ZARR", [](nb::handle) { return artifact_type::omeZarr; })
        .def_prop_ro_static("SURFACE_PREDICTION_ZARR", [](nb::handle) { return artifact_type::surfacePredictionZarr; })
        .def_prop_ro_static("NORMAL_GRIDS", [](nb::handle) { return artifact_type::normalGrids; })
        .def_prop_ro_static("LASAGNA", [](nb::handle) { return artifact_type::lasagna; })
        .def_prop_ro_static("FIBERLETS", [](nb::handle) { return artifact_type::fiberlets; })
        .def_prop_ro_static("INK_DETECTION_3D_ZARR", [](nb::handle) { return artifact_type::inkDetection3dZarr; })
        .def_prop_ro_static("UMBILICUS", [](nb::handle) { return artifact_type::umbilicus; })
        .def_prop_ro_static("TIFXYZ", [](nb::handle) { return artifact_type::tifxyz; })
        .def_prop_ro_static("TIFXYZ_FLATTENED", [](nb::handle) { return artifact_type::tifxyzFlattened; })
        .def_prop_ro_static("TIFXYZ_NORMALIZED", [](nb::handle) { return artifact_type::tifxyzNormalized; })
        .def_prop_ro_static("TIFXYZ_TRANSFORMED", [](nb::handle) { return artifact_type::tifxyzTransformed; })
        .def_prop_ro_static("OBJ", [](nb::handle) { return artifact_type::obj; })
        .def_prop_ro_static("OBJ_LEGACY_UNITS", [](nb::handle) { return artifact_type::objLegacyUnits; })
        .def_prop_ro_static("OBJ_FLATTENED", [](nb::handle) { return artifact_type::objFlattened; })
        .def_prop_ro_static("OBJ_NORMALIZED", [](nb::handle) { return artifact_type::objNormalized; })
        .def_prop_ro_static("LAYERS_ZARR", [](nb::handle) { return artifact_type::layersZarr; })
        .def_prop_ro_static("LAYERS_TIF", [](nb::handle) { return artifact_type::layersTif; })
        .def_prop_ro_static("ALPHA_RENDER", [](nb::handle) { return artifact_type::alphaRender; })
        .def_prop_ro_static("ALPHA_RENDER_DOWNSAMPLED", [](nb::handle) { return artifact_type::alphaRenderDownsampled; })
        .def_prop_ro_static("INK_DETECTION", [](nb::handle) { return artifact_type::inkDetection; })
        .def_prop_ro_static("INK_DETECTION_DOWNSAMPLED", [](nb::handle) { return artifact_type::inkDetectionDownsampled; })
        .def_prop_ro_static("INK_3D_PREDICTION", [](nb::handle) { return artifact_type::ink3dPrediction; })
        .def_prop_ro_static("INK_LABELS", [](nb::handle) { return artifact_type::inkLabels; });
    m.attr("DEFAULT_METADATA_URL") = std::string(kDefaultManifestUrl);
    m.attr("DEFAULT_RESOLUTION_REL_TOL") = kDefaultResolutionRelTol;
    m.attr("PYRAMID_LEVELS") = kPyramidLevels;
    m.def("parse_artifacts", [](nb::handle entries) { return tupleOf(parseCatalogArtifacts(toJson(entries, false))); });
    m.def("select_level", &selectLevel, nb::arg("source"), nb::arg("level"), "The source at a pyramid level; a single-resolution source has only level 0.");
    m.def("default_cache_dir", [] { return defaultCatalogCacheDir(); }, "Where manifests are cached (see vcr::catalog::defaultCatalogCacheDir).");
    m.def("join_url", &joinOpenDataUrl);
    m.def("to_s3", &openDataS3Url, "s3:// form of url, or None if it is not in a public channel.");
    m.def("to_https", &openDataHttpsUrl, "Public HTTPS location resolved by the shared native catalog.");
}
