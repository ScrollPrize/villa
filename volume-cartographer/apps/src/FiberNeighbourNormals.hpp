#pragma once
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <filesystem>
#include <string>
#include <vector>

// Solved strip normals sampled at a regular arc spacing along every fiber
// (unsigned: the whole-fiber sign comes from the sign stage).
struct FiberStripSample {
    double arc = 0.;
    cv::Vec3d point{0., 0., 0.};
    cv::Vec3d normal{0., 0., 0.};
};

void writeFiberStripSamples(const std::filesystem::path& output, const std::vector<std::string>& ids,
    const std::vector<std::vector<FiberStripSample>>& samples);

// Samples per fiber of `ids` (empty for fibers the file does not list).
std::vector<std::vector<FiberStripSample>> readFiberStripSamples(const std::filesystem::path& path,
    const std::vector<std::string>& ids, size_t& unknownFibers);

// Normal and sign at every strip sample of the query fibers, inferred only from
// the signed strip samples of OTHER fibers within a radius: the principal axis
// of the distance-weighted outer products (perpendicular to the own tangent),
// signed by the weighted majority. Config keys: strip_samples, fiber_signs
// (JSON uuid -> sign), neighbour_queries (JSON uuid -> uuids to exclude),
// neighbour_variants ([{falloff, sigma, radius}]), neighbour_normals_output,
// workers; optionally neighbour_samples_output (+ neighbour_samples_variant):
// the signed inferred normals of one variant as strip samples. Returns the run summary.
nlohmann::json inferFiberNeighbourNormals(const nlohmann::json& cfg);
