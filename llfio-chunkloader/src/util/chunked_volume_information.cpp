//
// Created by Lukas Karafiat on 11.02.24.
//
#include "util/chunked_volume_information.h"

#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;

ChunkedVolumeInformation
ChunkedVolumeInformation::read_from_file(std::filesystem::path const& file_path)
{
	std::ifstream file(file_path);
	if (!file.is_open())
	{
		throw std::filesystem::filesystem_error("could not open volume information", file_path, std::error_code(errno, std::system_category()));
	}

	json json_structure = json::parse(file);

	ChunkedVolumeInformation info = {
		.name = json_structure["name"],
		.uuid = json_structure["uuid"],
		.type = json_structure["type"],

		.height = json_structure["height"],
		.width  = json_structure["width"],
		.slices = json_structure["slices"],

		.chunk_size = json_structure["chunk_size"],

		.max = json_structure["max"],
		.min = json_structure["min"],

		.voxel_size = json_structure["voxelsize"],
	};

	return info;
}

void
ChunkedVolumeInformation::write_to_file(ChunkedVolumeInformation const& info, const std::filesystem::path& file)
{
	json json_structure;

	json_structure["name"] = info.name;
	json_structure["uuid"] = info.uuid;
	json_structure["type"] = info.type;

	json_structure["height"] = info.height;
	json_structure["width"] = info.width;
	json_structure["slices"] = info.slices;

	json_structure["chunk_size"] = info.chunk_size;

	json_structure["max"] = info.max;
	json_structure["min"] = info.min;

	json_structure["voxelsize"] = info.voxel_size;

	std::ofstream stream(file);
	stream << json_structure;
}

ChunkedVolumeInformation
ChunkedVolumeInformation::from_volume_information(const VolumeInformation& info, uint64_t chunk_size)
{
	ChunkedVolumeInformation new_info = {
		.name = info.name,
		.uuid = info.uuid,
		.type = info.type,

		.height = info.height,
		.width  = info.width,
		.slices = info.slices,

		.chunk_size = chunk_size,

		.max = info.max,
		.min = info.min,

		.voxel_size = info.voxel_size,
	};
	return new_info;
}
