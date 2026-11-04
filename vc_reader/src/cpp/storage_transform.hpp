// Axis-aligned storage transforms; independent of array readers and registrations.
#pragma once
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>
#include <nlohmann/json.hpp>

namespace vcr::storage {
struct AxisTransform {
    double scale = 1, offset = 0;
};
struct Transform {
    std::vector<AxisTransform> axes;
    bool scaled = false, translated = false;
};

// Selected axes may omit a dimension (e.g. Z in a 2-D image). Supplying rank
// additionally validates every stored component, including nonspatial axes.
inline Transform parse(const nlohmann::json& transforms,
                       const std::vector<std::optional<std::size_t>>& axes,
                       std::optional<std::size_t> rank = std::nullopt)
{
    Transform out{std::vector<AxisTransform>(axes.size())};
    for (const auto& t : transforms) {
        const auto type = t.is_object() ? t.value("type", std::string()) : std::string();
        if (type == "identity") continue;
        if (type != "scale" && type != "translation")
            throw std::invalid_argument("unsupported storage transformation '" + type + "'");
        if (!t.contains(type)) throw std::invalid_argument("storage transformation is missing its values");
        const auto& values = t.at(type);
        if (!values.is_array() || (rank && values.size() != *rank))
            throw std::invalid_argument("storage transformation rank contradicts storage axes");
        const auto value = [&](std::size_t i) {
            if (i >= values.size() || !values[i].is_number() || !std::isfinite(values[i].get<double>()))
                throw std::invalid_argument("storage transformation must cover its axes with finite numbers");
            return values[i].get<double>();
        };
        if (rank) for (std::size_t i = 0; i < *rank; ++i) (void)value(i);
        for (std::size_t i = 0; i < axes.size(); ++i) {
            if (!axes[i]) continue;
            const double v = value(*axes[i]);
            auto& a = out.axes[i];
            if (type == "scale") {
                if (!(v > 0)) throw std::invalid_argument("storage scale must be positive");
                a.scale *= v;
                a.offset *= v;
            } else a.offset += v;
        }
        (type == "scale" ? out.scaled : out.translated) = true;
    }
    return out;
}

// A block-reduced voxel is centred in the block of finer voxels it represents.
inline double reducedOffset(const AxisTransform& selected, const AxisTransform& finest)
{
    return finest.offset + (selected.scale - finest.scale) / 2;
}
} // namespace vcr::storage
