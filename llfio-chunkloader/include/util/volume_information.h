//
// Created by Lukas Karafiat on 11.02.24.
//

#ifndef UTIL_VOLUME_INFORMATION_H
#define UTIL_VOLUME_INFORMATION_H

#include <string>
#include <cstdint>
#include <filesystem>

struct VolumeInformation
{
	std::string name;
	std::string uuid;
	std::string type = "vol";
	uint64_t    height;
	uint64_t    width;
	uint64_t    slices;
	double      max  = 65535.0;
	double      min  = 0.0;
	double      voxel_size;

	static
	VolumeInformation
	read_from_file(std::filesystem::path const& file);

	static
	void
	write_to_file(VolumeInformation const& info, std::filesystem::path const& file);
};

#endif //UTIL_VOLUME_INFORMATION_H
