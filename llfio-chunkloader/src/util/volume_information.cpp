//
// Created by Lukas Karafiat on 11.02.24.
//
#include "util/volume_information.h"

#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;

VolumeInformation
VolumeInformation::read_from_file(std::filesystem::path const& file_path)
{
	std::ifstream file(file_path);
	if (!file.is_open())
	{
		throw std::filesystem::filesystem_error("could not open volume information", file_path, std::error_code(errno, std::system_category()));
	}

	json json_structure = json::parse(file);

	VolumeInformation info = {
		.name = json_structure["name"],
		.uuid = json_structure["uuid"],
		.type = json_structure["type"],

		.height = json_structure["height"],
		.width  = json_structure["width"],
		.slices = json_structure["slices"],

		.max = json_structure["max"],
		.min = json_structure["min"],

		.voxel_size = json_structure["voxelsize"],
	};

	return info;
}

void
VolumeInformation::write_to_file(VolumeInformation const& info, const std::filesystem::path& file)
{
	json json_structure;

	json_structure["name"] = info.name;
	json_structure["uuid"] = info.uuid;
	json_structure["type"] = info.type;

	json_structure["height"] = info.height;
	json_structure["width"] = info.width;
	json_structure["slices"] = info.slices;

	json_structure["max"] = info.max;
	json_structure["min"] = info.min;


	json_structure["voxelsize"] = info.voxel_size;

	std::ofstream stream(file);
	stream << json_structure;
}
