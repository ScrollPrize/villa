#include "vc/tracer/DirectionField.hpp"
#include "vc_reader/slice.hpp"

#include <cmath>
#include <stdexcept>

DirectionField::DirectionField(std::string dir, const std::filesystem::path& root,
                               int level, float fieldWeight,
                               const std::optional<std::filesystem::path>& confidenceRoot)
    : direction(std::move(dir)), scale(std::pow(2.0f, -float(level))), weight(fieldWeight),
      native(std::make_unique<Native>())
{
    native->service = vcr::chunks::processService();
    vcr::zarr::OpenOptions options;
    options.threads = 1;
    const std::array<const char*, 3> axes{"x", "y", "z"};
    for (std::size_t c = 0; c < native->arrays.size(); ++c) {
        if (c == 3 && !confidenceRoot) break;
        const auto path = c < 3 ? root / axes[c] / std::to_string(level)
                               : *confidenceRoot / std::to_string(level);
        auto opened = std::make_shared<vcr::zarr::OpenedArray>(vcr::zarr::openArray(path.string(), options));
        if (opened->meta.shape.size() != 3 || opened->meta.type.kind != 'u' || opened->meta.itemsize() != 1)
            throw std::runtime_error("Direction fields require 3-D uint8 arrays: " + path.string());
        if (c > 0 && c < 3 && opened->meta.shape != native->arrays[0]->meta.shape)
            throw std::runtime_error("Direction field XYZ shapes differ: " + path.string());
        native->arrays[c] = std::move(opened);
    }
    for (std::size_t c = 0; c < native->arrays.size(); ++c) {
        const auto& opened = native->arrays[c];
        if (!opened) break;
        native->sources[c] = native->service->addSource(
            {std::shared_ptr<vcr::Array>(opened, opened->array.get())});
    }
}

DirectionField::Native::~Native()
{
    if (service)
        for (const auto source : sources)
            if (source) service->removeSource(source);
}

DirectionField::~DirectionField() = default;
DirectionField::DirectionField(DirectionField&&) noexcept = default;
DirectionField& DirectionField::operator=(DirectionField&&) noexcept = default;

DirectionConstraintSample sampleDirectionConstraint(const DirectionField& field, cv::Vec3d xyz)
{
    const float point[] = {float(xyz[0] * field.scale), float(xyz[1] * field.scale), float(xyz[2] * field.scale)};
    float values[4] = {128, 128, 128, 255};
    for (std::size_t c = 0; c < field.native->arrays.size(); ++c) {
        const auto& opened = field.native->arrays[c];
        if (!opened) break;
        vcr::slice::Cached source{*opened->array, *field.native->service, field.native->sources[c]};
        vcr::slice::points(source, opened->meta.type, point, 1, vcr::slice::Order::Nearest, &values[c]);
    }
    return {(cv::Vec3f{values[2], values[1], values[0]} - cv::Vec3f{128, 128, 128}) / 127.f,
            values[3] / 255.f};
}
