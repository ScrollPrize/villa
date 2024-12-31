//
// Created by Lukas Karafiat on 11.02.24.
//

#ifndef UTIL_CHUNKED_VOLUME_INFORMATION_H
#define UTIL_CHUNKED_VOLUME_INFORMATION_H

#include "util/volume_information.h"

#include <string>
#include <cstdint>
#include <filesystem>

struct ChunkedVolumeInformation
{
	std::string name;
	std::string uuid;
	std::string type = "vol";
	uint64_t    height;
	uint64_t    width;
	uint64_t    slices;
	uint64_t    chunk_size;
	double      max  = 65535.0;
	double      min  = 0.0;
	double      voxel_size;

	static
	ChunkedVolumeInformation
	read_from_file(std::filesystem::path const& file);

	static
	void
	write_to_file(ChunkedVolumeInformation const& info, std::filesystem::path const& file);

	static
	ChunkedVolumeInformation
	from_volume_information(VolumeInformation const& info, uint64_t chunk_size);
};

#endif //UTIL_CHUNKED_VOLUME_INFORMATION_H
