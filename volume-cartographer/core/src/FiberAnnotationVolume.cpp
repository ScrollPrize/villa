#include "vc/core/util/FiberAnnotationVolume.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace vc {
namespace {

std::string stableIdentity(const FiberAnnotationInput& fiber)
{
    if (!fiber.identity.empty())
        return fiber.identity;
    if (!fiber.sourcePath.empty())
        return fiber.sourcePath.generic_string();
    throw std::invalid_argument("fiber identity and sourcePath may not both be empty");
}

bool coordinateCompatible(const FiberAnnotationInput& fiber,
                          const std::string& targetCoordinateSpace)
{
    return !fiber.coordinateSpace || fiber.coordinateSpace->empty() ||
           targetCoordinateSpace.empty() || *fiber.coordinateSpace == targetCoordinateSpace;
}

std::vector<std::array<double, 3>> scaled(
    const std::vector<std::array<double, 3>>& points,
    double scale)
{
    std::vector<std::array<double, 3>> result = points;
    for (auto& point : result) {
        for (double& value : point)
            value *= scale;
    }
    return result;
}

} // namespace

FiberAnnotationBatches makeFiberAnnotationBatches(
    const std::vector<FiberAnnotationInput>& fibers,
    const std::string& targetCoordinateSpace,
    double coordinateScale)
{
    if (!std::isfinite(coordinateScale) || coordinateScale <= 0.0)
        throw std::invalid_argument("fiber coordinateScale must be finite and positive");

    std::map<std::string, std::vector<const FiberAnnotationInput*>> compatible;
    for (const auto& fiber : fibers) {
        if (coordinateCompatible(fiber, targetCoordinateSpace))
            compatible[stableIdentity(fiber)].push_back(&fiber);
    }
    if (compatible.size() > std::numeric_limits<std::uint16_t>::max())
        throw std::overflow_error("more than 65535 compatible fibers cannot be labeled");

    FiberAnnotationBatches result;
    result.batches.reserve(compatible.size() * 2);
    result.labels.reserve(compatible.size());
    std::uint32_t nextLabel = 1;
    for (const auto& [identity, entries] : compatible) {
        const auto label = static_cast<std::uint16_t>(nextLabel++);
        result.labels.push_back({label, identity, entries.front()->sourcePath});
        for (const auto* fiber : entries) {
            const auto& centerline = fiber->linePointsXYZ.empty()
                ? fiber->controlPointsXYZ
                : fiber->linePointsXYZ;
            if (!centerline.empty()) {
                result.batches.push_back(AnnotationPointBatch{
                    label,
                    scaled(centerline, coordinateScale),
                    AnnotationCoordinateOrder::XYZ,
                    AnnotationGeometryMode::OrderedPolyline,
                    1.0,
                });
            }
            if (!fiber->controlPointsXYZ.empty()) {
                result.batches.push_back(AnnotationPointBatch{
                    label,
                    scaled(fiber->controlPointsXYZ, coordinateScale),
                    AnnotationCoordinateOrder::XYZ,
                    AnnotationGeometryMode::Points,
                    3.0,
                });
            }
        }
    }
    return result;
}

utils::Json fiberAnnotationAttributes(const FiberAnnotationBatches& adapter,
                                      const std::string& coordinateSpace)
{
    utils::Json attrs = utils::Json::object();
    utils::Json labels = utils::Json::object();
    for (const auto& mapping : adapter.labels) {
        labels[std::to_string(mapping.label)] = utils::Json{
            {"identity", mapping.identity},
            {"path", mapping.sourcePath.generic_string()},
        };
    }
    attrs["fiber_labels"] = std::move(labels);
    if (!coordinateSpace.empty())
        attrs["vc_open_data_coordinate_space"] = coordinateSpace;
    return attrs;
}

} // namespace vc
