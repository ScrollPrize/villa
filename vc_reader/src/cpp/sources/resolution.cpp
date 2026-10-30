// Selected storage grids, without changing the declarations they came from. Spatial storage
// transforms use XYZ column vectors and micrometers for dimensional frames.
#include "vc_reader/source_reader.hpp"

#include <cmath>
#include <set>

namespace vcr::sources {
using nlohmann::json;
namespace {
[[noreturn]] void bad(const std::string& message) { throw std::invalid_argument(message); }
const VolumeRegistration* volumeOf(const std::optional<SourceRegistration>& registration)
{
    return registration ? std::get_if<VolumeRegistration>(&*registration) : nullptr;
}
const char* spatialAxis(std::string_view axis) { return axis == "x" ? "x" : axis == "y" ? "y" : axis == "z" ? "z" : nullptr; }
int xyzIndex(std::string_view axis) { return axis == "x" ? 0 : axis == "y" ? 1 : 2; }
} // namespace

std::optional<double> micrometersPer(std::string_view unit)
{
    if (unit == "meter" || unit == "metre" || unit == "m") return 1e6;
    if (unit == "millimeter" || unit == "mm") return 1e3;
    if (unit == "micrometer" || unit == "micrometre" || unit == "um" || unit == "µm") return 1.;
    if (unit == "nanometer" || unit == "nm") return .001;
    return std::nullopt;
}

VolumeTransform pixelAffine(const PixelAffine& rows)
{
    Affine out;
    out(0, 0) = rows[0][0], out(0, 1) = rows[0][1], out(0, 3) = rows[0][2];
    out(1, 0) = rows[1][0], out(1, 1) = rows[1][1], out(1, 3) = rows[1][2];
    (void)out.inverse();
    return {out, {}};
}

json blockReducedTransforms(const json& transforms, const json& finest)
{
    json out = transforms.is_array() ? transforms : json::array();
    const auto named = [](const json& list, const char* type) -> const json* {
        if (list.is_array())
            for (const auto& item : list)
                if (item.is_object() && item.value("type", "") == type) return &item;
        return nullptr;
    };
    if (named(out, "translation")) return out;
    const auto* scale = named(out, "scale");
    const auto* base = named(finest, "scale");
    if (!scale || !base || !scale->contains("scale") || !base->contains("scale")) return out;
    const auto& s = (*scale)["scale"];
    const auto& b = (*base)["scale"];
    if (!s.is_array() || !b.is_array() || s.size() != b.size()) return out;
    const auto* translation = named(finest, "translation");
    json shifted = json::array();
    for (std::size_t i = 0; i < s.size(); ++i) {
        const double origin = translation ? (*translation)["translation"].at(i).get<double>() : 0.;
        shifted.push_back(origin + (s[i].get<double>() - b[i].get<double>()) / 2);
    }
    out.push_back({{"type", "translation"}, {"translation", shifted}});
    return out;
}

StorageGrid storageTransform(const std::vector<std::string>& axes, const std::vector<std::optional<std::string>>& units, const std::vector<json>& groups)
{
    std::vector<json> transforms;
    for (const auto& group : groups)
        if (group.is_array())
            for (const auto& transform : group) transforms.push_back(transform);
    if (transforms.empty()) return {};
    Affine result;
    for (const auto& transform : transforms) {
        const auto kind = transform.is_object() ? transform.value("type", "") : std::string();
        if (kind != "scale" && kind != "translation" && kind != "identity")
            bad("unsupported storage transformation '" + kind + "'; declare a supported grid");
        if (kind == "identity") continue;
        const auto values = transform.value(kind, json::array());
        if (!values.is_array() || values.size() != axes.size()) bad("storage transformation rank/nonfinite values contradict storage axes");
        Affine step;
        for (std::size_t i = 0; i < axes.size(); ++i) {
            if (!values[i].is_number() || !std::isfinite(values[i].get<double>())) bad("storage transformation rank/nonfinite values contradict storage axes");
            const double value = values[i].get<double>();
            if (!spatialAxis(axes[i])) continue;
            const int axis = xyzIndex(axes[i]);
            if (kind == "scale") {
                if (value <= 0) bad("storage scale must be positive");
                step(axis, axis) = value;
            } else {
                step(axis, 3) = value;
            }
        }
        result = step * result;
    }
    std::set<std::optional<std::string>> spatialUnits;
    for (std::size_t i = 0; i < axes.size(); ++i)
        if (spatialAxis(axes[i])) spatialUnits.insert(i < units.size() ? units[i] : std::nullopt);
    const bool dimensional = !(spatialUnits.size() == 1 && !spatialUnits.begin()->has_value());
    if (dimensional) {
        std::array<double, 3> factor{1, 1, 1};
        for (const auto& unit : spatialUnits)
            if (!unit || !micrometersPer(*unit)) bad("unknown or mixed dimensional/unitless storage units");
        for (std::size_t i = 0; i < axes.size(); ++i)
            if (spatialAxis(axes[i])) factor[xyzIndex(axes[i])] = *micrometersPer(*units[i]);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c) result(r, c) *= factor[r];
    }
    (void)result.inverse();
    return {VolumeTransform{result, {}}, dimensional ? std::optional<std::string>("micrometer") : std::nullopt};
}

SourceRegistration regrid(const SourceRegistration& registration, const VolumeTransform& originalFromSelected, const Selection& selection)
{
    const auto& r = originalFromSelected.forward;
    if (const auto* surface = std::get_if<SurfaceRegistration>(&registration)) {
        auto out = *surface;
        out.declaredGrid = selection;
        if (surface->pixelToUv) {
            PixelAffine mapped;
            for (int i = 0; i < 2; ++i) {
                const auto& row = (*surface->pixelToUv)[i];
                mapped[i] = {row[0] * r(0, 0) + row[1] * r(1, 0), row[0] * r(0, 1) + row[1] * r(1, 1), row[0] * r(0, 3) + row[1] * r(1, 3) + row[2]};
            }
            out.pixelToUv = mapped;
        }
        return out;
    }
    auto out = std::get<VolumeRegistration>(registration);
    out.declaredGrid = selection;
    if (out.plane) {
        Plane plane = *out.plane;
        for (int i = 0; i < 3; ++i) {
            const double x = out.plane->basisX[i], y = out.plane->basisY[i];
            plane.origin[i] = out.plane->origin[i] + r(0, 3) * x + r(1, 3) * y;
            plane.basisX[i] = r(0, 0) * x + r(1, 0) * y;
            plane.basisY[i] = r(0, 1) * x + r(1, 1) * y;
        }
        validate(plane);
        out.plane = plane;
    } else if (out.affine) {
        out.affine = VolumeTransform{out.affine->forward * r, {}};
    }
    return out;
}

std::optional<SourceRegistration> resolveRegistration(const Source& source, const Selection& selection, std::optional<VolumeTransform> selected,
                                                      const std::optional<std::string>& units, const std::optional<VolumeTransform>& declared,
                                                      const std::optional<std::string>& declaredUnits, std::optional<VolumeTransform> relative)
{
    if (!source.registration) return std::nullopt;
    const auto* volume = std::get_if<VolumeRegistration>(&*source.registration);
    if (volume && volume->mode == "storage") {
        if (!volume->affine || !selected) bad("storage placement requires both frame placement and array-to-storage metadata");
        auto transform = selected->forward;
        if (units) {
            const auto factor = volume->storageUnits ? micrometersPer(*volume->storageUnits) : std::nullopt;
            if (!factor) bad("dimensional storage placement requires explicit known storage_units");
            transform = Affine::scale(1 / *factor) * transform;
        } else if (volume->storageUnits) {
            bad("unitless storage metadata cannot resolve dimensional storage placement");
        }
        auto out = *volume;
        out.affine = VolumeTransform{volume->affine->forward * transform, {}};
        out.mode = "array";
        out.storageFrame.reset();
        out.storageUnits.reset();
        out.declaredGrid = selection;
        return out;
    }
    if (volume && volume->origin == "catalog-fallback" && selected) {
        auto out = *volume;
        if (units) {
            if (!volume->volumeSpacingUm) bad("physical storage metadata needs volume_spacing_um; cannot use a catalog estimate");
            Affine rows = selected->forward;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c) rows(r, c) /= (*volume->volumeSpacingUm)[r];
            (void)rows.inverse();
            out.affine = VolumeTransform{rows, {}};
        } else {
            // Unitless metadata only establishes relative pyramid grids. Anchor it at the
            // explicitly known base grid, not an invented global frame.
            if (!declared || !volume->affine) bad("storage metadata has no connection to the declared volume grid");
            out.affine = VolumeTransform{volume->affine->forward * declared->forward.inverse() * selected->forward, {}};
        }
        out.declaredGrid = selection;
        out.origin = "storage-metadata";
        return out;
    }
    const auto& declaredGrid = volume ? volume->declaredGrid : std::get<SurfaceRegistration>(*source.registration).declaredGrid;
    if (declaredGrid.value_or(Selection{}).spatial() == selection.spatial()) {
        auto out = *source.registration;
        std::visit([&](auto& r) { r.declaredGrid = selection; }, out);
        return out;
    }
    if (!relative && selected && declared) {
        if (units != declaredUnits) bad("selected and declared grids have incompatible units");
        relative = VolumeTransform{declared->forward.inverse() * selected->forward, {}};
    }
    if (!relative) bad("selected grid has no known relationship to declared_grid; supply explicit placement");
    return regrid(*source.registration, *relative, selection);
}

bool matchesReference(const Source& source, const std::optional<SourceRegistration>& registration, const Frame& reference, const std::optional<std::vector<std::string>>& axes)
{
    const auto* volume = volumeOf(registration);
    if (!volume) return false;
    std::optional<Affine> transform;
    if (volume->affine) transform = volume->affine->forward;
    if (volume->plane) {
        const auto& plane = *volume->plane;
        Affine rows;
        for (int r = 0; r < 3; ++r) {
            rows(r, 0) = plane.basisX[r], rows(r, 1) = plane.basisY[r], rows(r, 2) = 0, rows(r, 3) = plane.origin[r];
        }
        transform = rows;
    }
    if (!transform) return false;
    if (reference.sampleId != volume->sampleId || reference.volumeId != volume->volumeId) {
        if (!source.query || source.query->reference != reference) bad("mapped corresponding level needs its published query transform");
        transform = source.query->transform.forward * *transform;
    }
    // Gram matrices accept exact reflections/permutations, but not shears.
    const auto names = axes ? *axes : source.axes.value_or(std::vector<std::string>{"z", "y", "x"});
    std::vector<int> dimensions;
    for (const auto& axis : names)
        if (spatialAxis(axis)) dimensions.push_back(xyzIndex(axis));
    const double expected = std::ldexp(1.0, int(2 * reference.level));
    for (const int i : dimensions)
        for (const int j : dimensions) {
            double sum = 0;
            for (int k = 0; k < 3; ++k) sum += (*transform)(k, i) * (*transform)(k, j);
            if (sum != (i == j ? expected : 0.)) return false;
        }
    return true;
}

VolumeTransform arrayPlacement(const std::optional<SourceRegistration>& registration)
{
    const auto* volume = volumeOf(registration);
    if (!volume) bad("volume registration is required; surface targets require surface projection");
    std::optional<VolumeTransform> affine = volume->affine;
    if (volume->plane) {
        const auto& p = *volume->plane;
        Affine rows;
        for (int r = 0; r < 3; ++r) {
            rows(r, 0) = p.basisX[r], rows(r, 1) = p.basisY[r], rows(r, 2) = p.thickness[r], rows(r, 3) = p.origin[r];
        }
        affine = VolumeTransform{rows, {}};
    }
    if (!affine || volume->mode != "array") bad("complete resolved array placement is required");
    const auto inverse = affine->forward.inverse();  // also rejects nonfinite values
    (void)inverse;
    for (const double v : affine->forward.values)
        if (!std::isfinite(v)) bad("registration and its inverse must be finite");
    return *affine;
}
} // namespace vcr::sources
