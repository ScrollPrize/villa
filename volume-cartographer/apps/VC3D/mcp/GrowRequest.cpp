#include "VolumeCartographer.hpp"

#include <fastmcpp/exceptions.hpp>

#include <cmath>
#include <filesystem>
#include <string_view>
#include <system_error>

namespace vc::mcp
{
namespace
{

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw fastmcpp::ValidationError(message);
}

}  // namespace

Json validateAndNormalizeLocalGrowRequest(const Json& request)
{
    require(request.is_object(), "request must be an object");
    for (const auto* field : {"seed", "prediction_space", "profile", "client_request_id"})
        require(request.contains(field), std::string("missing required field: ") + field);
    const bool hasPath = request.contains("prediction_path");
    const bool hasUri = request.contains("prediction_uri");
    require(hasPath != hasUri, "provide exactly one of prediction_path or prediction_uri");

    std::string predictionSource;
    if (hasPath) {
        require(request.at("prediction_path").is_string(), "prediction_path must be a string");
        std::filesystem::path prediction(request.at("prediction_path").get<std::string>());
        require(prediction.is_absolute(), "prediction_path must be absolute");
        std::error_code error;
        prediction = std::filesystem::weakly_canonical(prediction, error);
        require(!error && std::filesystem::is_directory(prediction), "prediction_path must be an existing directory");
        predictionSource = prediction.string();
    } else {
        require(request.at("prediction_uri").is_string(), "prediction_uri must be a string");
        predictionSource = request.at("prediction_uri").get<std::string>();
        constexpr std::string_view ash2txtPrefix = "https://dl.ash2txt.org/other/dev/scrolls/5/volumes/";
        constexpr std::string_view vesuviusPrefix = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/";
        const auto matchedPrefix = predictionSource.starts_with(ash2txtPrefix)    ? ash2txtPrefix
                                   : predictionSource.starts_with(vesuviusPrefix) ? vesuviusPrefix
                                                                                  : std::string_view{};
        require(!matchedPrefix.empty(), "prediction_uri is outside the local server allowlist");
        require(
            predictionSource.size() > matchedPrefix.size() && predictionSource.find("..") == std::string::npos &&
                predictionSource.find('?') == std::string::npos && predictionSource.find('#') == std::string::npos,
            "prediction_uri contains a disallowed path or query");
    }

    require(request.at("profile") == "scroll3-conservative-v1", "unsupported profile");
    const auto& seed = request.at("seed");
    require(seed.is_object(), "seed must be an object");
    for (const auto* field : {"x", "y", "z", "space"})
        require(seed.contains(field), std::string("missing seed field: ") + field);
    require(seed.at("x").is_number() && seed.at("y").is_number() && seed.at("z").is_number(), "seed coordinates must be numbers");
    for (const auto* axis : {"x", "y", "z"}) {
        const double value = seed.at(axis).get<double>();
        require(std::isfinite(value) && value > 0.0, std::string("seed.") + axis + " must be finite and greater than zero");
    }
    require(seed.at("space") == "ct_l0_xyz" || seed.at("space") == "ct_l2_xyz", "unsupported coordinate space");
    require(request.at("prediction_space") == "ct_l0_xyz" || request.at("prediction_space") == "ct_l2_xyz", "unsupported prediction_space");
    require(request.at("client_request_id").is_string(), "client_request_id must be a string");
    const auto requestId = request.at("client_request_id").get<std::string>();
    require(!requestId.empty() && requestId.size() <= 128, "client_request_id must contain 1 to 128 characters");

    int generations = 256;
    double minAreaCm2 = 0.3;
    if (request.contains("limits")) {
        require(request.at("limits").is_object(), "limits must be an object");
        generations = request.at("limits").value("max_generations", generations);
        minAreaCm2 = request.at("limits").value("min_area_cm2", minAreaCm2);
    }
    require(generations >= 1 && generations <= 10000, "limits.max_generations must be from 1 to 10000");
    require(std::isfinite(minAreaCm2) && minAreaCm2 >= 0.0 && minAreaCm2 <= 100.0, "limits.min_area_cm2 must be from 0 to 100");
    if (hasUri)
        require(request.contains("voxel_size_um"), "remote prediction_uri requires voxel_size_um");
    if (request.contains("voxel_size_um")) {
        require(request.at("voxel_size_um").is_number(), "voxel_size_um must be a number");
        const double voxelSize = request.at("voxel_size_um").get<double>();
        require(
            std::isfinite(voxelSize) && voxelSize > 0.0 && voxelSize <= 10000.0, "voxel_size_um must be greater than 0 and at most 10000");
    }

    Json normalized = request;
    normalized["prediction_source"] = predictionSource;
    const double submittedScale = seed.at("space") == "ct_l2_xyz" ? 4.0 : 1.0;
    const double predictionScale = request.at("prediction_space") == "ct_l2_xyz" ? 4.0 : 1.0;
    const auto coordinate = [&](const char* axis) { return seed.at(axis).get<double>() * submittedScale / predictionScale; };
    const auto l0Coordinate = [&](const char* axis) { return seed.at(axis).get<double>() * submittedScale; };
    normalized["coordinates"] =
        {{"submitted", seed},
         {"ct_l0", {{"x", l0Coordinate("x")}, {"y", l0Coordinate("y")}, {"z", l0Coordinate("z")}, {"space", "ct_l0_xyz"}}},
         {"vc_input", {{"x", coordinate("x")}, {"y", coordinate("y")}, {"z", coordinate("z")}, {"space", request.at("prediction_space")}}},
         {"transform",
          {{"submitted_to_l0_scale", submittedScale}, {"l0_to_prediction_scale", 1.0 / predictionScale}, {"permutation", Json::array({0, 1, 2})}}}};
    normalized["limits"] = {{"max_generations", generations}, {"min_area_cm2", minAreaCm2}};
    return normalized;
}

}  // namespace vc::mcp
