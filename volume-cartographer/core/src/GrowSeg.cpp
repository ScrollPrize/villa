#include "vc/tracer/GrowSeg.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>

#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/tracer/Tracer.hpp"
#include "z5/factory.hxx"

namespace {

std::string read_dimension_separator(const std::filesystem::path& zarray_path) {
    std::ifstream zarray_stream(zarray_path);
    if (!zarray_stream) {
        throw std::runtime_error("Unable to open scale `0` zarray metadata at " + zarray_path.string());
    }

    nlohmann::json zarray = nlohmann::json::parse(zarray_stream);
    return zarray.value("dimension_separator", std::string("."));
}

std::unique_ptr<z5::Dataset> open_scale0_dataset(const std::filesystem::path& volume_path) {
    if (!std::filesystem::exists(volume_path)) {
        throw std::runtime_error("Volume path does not exist: " + volume_path.string());
    }

    z5::filesystem::handle::Group group(volume_path, z5::FileMode::FileMode::r);
    const auto zarray_path = volume_path / "0" / ".zarray";
    const auto separator = read_dimension_separator(zarray_path);

    z5::filesystem::handle::Dataset ds_handle(group, "0", separator);
    return z5::filesystem::openDataset(ds_handle);
}

float read_voxel_size(const std::filesystem::path& volume_path) {
    const auto meta_path = volume_path / "meta.json";
    std::ifstream meta_stream(meta_path);
    if (!meta_stream) {
        std::cerr << "WARNING: Missing meta.json at " << meta_path << "; assuming voxelsize = 1.0\n";
        return 1.0f;
    }

    try {
        const auto meta = nlohmann::json::parse(meta_stream);
        if (!meta.contains("voxelsize")) {
            std::cerr << "WARNING: meta.json missing voxelsize; assuming voxelsize = 1.0\n";
            return 1.0f;
        }
        return meta["voxelsize"].get<float>();
    } catch (const std::exception& ex) {
        std::cerr << "WARNING: Failed to parse voxelsize: " << ex.what() << "; assuming voxelsize = 1.0\n";
        return 1.0f;
    }
}

std::filesystem::path resolve_cache_root(const GrowSegRequest& request) {
    if (request.cache_root_override) {
        return *request.cache_root_override;
    }

    if (request.params.contains("cache_root")) {
        const auto& cache_root = request.params["cache_root"];
        if (cache_root.is_string()) {
            return std::filesystem::path(cache_root.get<std::string>());
        }
        if (!cache_root.is_null()) {
            std::cerr << "WARNING: cache_root must be a string; ignoring parameter\n";
        }
    }

    return {};
}

std::vector<DirectionField> make_direction_fields(const nlohmann::json& params,
                                                  ChunkCache* chunk_cache,
                                                  const std::filesystem::path& cache_root) {
    std::vector<DirectionField> direction_fields;
    if (!params.contains("direction_fields")) {
        return direction_fields;
    }

    if (!params["direction_fields"].is_array()) {
        std::cerr << "WARNING: direction_fields must be an array; ignoring\n";
        return direction_fields;
    }

    const std::string cache_root_str = cache_root.empty() ? std::string{} : cache_root.string();

    for (const auto& direction_field : params["direction_fields"]) {
        if (!direction_field.contains("zarr") || !direction_field.contains("dir") || !direction_field.contains("scale")) {
            std::cerr << "WARNING: direction_field entries must contain zarr, dir, and scale keys; skipping entry\n";
            continue;
        }

        const auto direction = direction_field["dir"].get<std::string>();
        if (!std::ranges::contains(std::array{"horizontal", "vertical", "normal"}, direction)) {
            std::cerr << "WARNING: invalid direction '" << direction << "'; skipping direction_field entry\n";
            continue;
        }

        const auto ome_scale = direction_field["scale"].get<int>();
        const auto zarr_path = direction_field["zarr"].get<std::string>();
        const float scale_factor = std::pow(2.0f, static_cast<float>(-ome_scale));

        try {
            z5::filesystem::handle::Group dirs_group(zarr_path, z5::FileMode::FileMode::r);
            std::vector<std::unique_ptr<z5::Dataset>> direction_datasets;
            for (const char axis : std::string("xyz")) {
                z5::filesystem::handle::Group axis_group(dirs_group, std::string(1, axis));
                z5::filesystem::handle::Dataset dataset_handle(axis_group, std::to_string(ome_scale), ".");
                direction_datasets.push_back(z5::filesystem::openDataset(dataset_handle));
            }

            std::unique_ptr<z5::Dataset> weight_dataset;
            if (direction_field.contains("weight_zarr")) {
                const auto weight_zarr_path = direction_field["weight_zarr"].get<std::string>();
                z5::filesystem::handle::Group weight_group(weight_zarr_path);
                z5::filesystem::handle::Dataset weight_ds_handle(weight_group, std::to_string(ome_scale), ".");
                weight_dataset = z5::filesystem::openDataset(weight_ds_handle);
            }

            float weight = 1.0f;
            if (direction_field.contains("weight")) {
                try {
                    weight = std::clamp(direction_field["weight"].get<float>(), 0.0f, 10.0f);
                } catch (const std::exception& ex) {
                    std::cerr << "WARNING: invalid weight in direction_field " << zarr_path << ": " << ex.what() << std::endl;
                }
            }

            const auto unique_id = std::to_string(std::hash<std::string>{}(dirs_group.path().string() + std::to_string(ome_scale)));
            direction_fields.emplace_back(
                direction,
                std::make_unique<Chunked3dVec3fFromUint8>(std::move(direction_datasets), scale_factor, chunk_cache, cache_root_str, unique_id),
                weight_dataset ? std::make_unique<Chunked3dFloatFromUint8>(std::move(weight_dataset), scale_factor, chunk_cache, cache_root_str, unique_id + "_conf") : nullptr,
                weight);
        } catch (const std::exception& ex) {
            std::cerr << "WARNING: failed to load direction_field datasets from " << zarr_path << ": " << ex.what() << std::endl;
        }
    }

    return direction_fields;
}

const VCCollection& corrections_or_default(const GrowSegRequest& request) {
    if (request.corrections) {
        return *request.corrections;
    }

    static VCCollection empty_collection;
    return empty_collection;
}

} // namespace

GrowSegResult run_grow_seg_from_seed(const GrowSegRequest& request) {
    GrowSegResult result;

    std::shared_ptr<z5::Dataset> dataset;
    if (request.dataset) {
        dataset = request.dataset;
    } else {
        auto dataset_unique = open_scale0_dataset(request.volume_path);
        dataset = std::shared_ptr<z5::Dataset>(std::move(dataset_unique));
    }

    const auto cache_root = resolve_cache_root(request);
    ChunkCache chunk_cache(request.params.value("cache_size", 1e9));
    auto direction_fields = make_direction_fields(request.params, &chunk_cache, cache_root);

    if (request.params.contains("use_cuda")) {
        set_space_tracing_use_cuda(request.params.value("use_cuda", false));
    } else {
        set_space_tracing_use_cuda(false);
    }

    float voxelsize = 1.0f;
    if (request.voxel_size_override) {
        voxelsize = *request.voxel_size_override;
    } else {
        voxelsize = read_voxel_size(request.volume_path);
    }
    result.voxel_size = voxelsize;

    std::unique_ptr<QuadSurface> resume_surface;
    if (request.resume_surface_path && !request.resume_surface_path->empty()) {
        const bool has_corrections = request.corrections && !request.corrections->getAllCollections().empty();
        const int load_flags = has_corrections ? SURF_LOAD_IGNORE_MASK : 0;
        resume_surface.reset(load_quad_from_tifxyz(request.resume_surface_path->string(), load_flags));
    }

    cv::Vec3f origin_f{
        static_cast<float>(request.origin[0]),
        static_cast<float>(request.origin[1]),
        static_cast<float>(request.origin[2])
    };

    QuadSurface* raw_surface = tracer(
        dataset.get(),
        1.0f,
        &chunk_cache,
        origin_f,
        request.params,
        cache_root.string(),
        voxelsize,
        direction_fields,
        resume_surface.get(),
        request.output_dir,
        request.meta,
        corrections_or_default(request));

    if (!raw_surface) {
        throw std::runtime_error("Tracer returned null surface");
    }

    result.surface.reset(raw_surface);

    if (result.surface->meta && result.surface->meta->contains("area_cm2")) {
        try {
            result.area_cm2 = (*result.surface->meta)["area_cm2"].get<double>();
        } catch (const std::exception&) {
            result.area_cm2 = 0.0;
        }
    }

    return result;
}
