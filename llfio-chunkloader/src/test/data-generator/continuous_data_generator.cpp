//
// Created by Lukas Karafiat on 21.02.24.
//

#include "test/data-generator/continuous_data_generator.h"

#include "test/data.h"
#include "util/chunked_volume_information.h"

namespace fs = std::filesystem;

TestData
create_continuous_walkthrough_data(fs::path const& volume_path)
{
	TestData test_data;

	ChunkedVolumeInformation info = ChunkedVolumeInformation::read_from_file(volume_path / "meta.json");

	uint64_t max_x = info.width / info.chunk_size;
	uint64_t max_y = info.height / info.chunk_size;
	uint64_t max_z = info.slices / info.chunk_size;

	test_data.volume_path = volume_path;

	for (int64_t z = 0; z < max_z; z++)
	{
		ChunkCoordinates coordinates;

		for (int64_t y = 0; y < max_y; y++)
		{
			for (int64_t x = 0; x < max_x; x++)
			{ coordinates.emplace_back(x, y, z); }
		}

		test_data.chunks_to_preload.push_back(coordinates);
		test_data.chunks_to_load.push_back(coordinates);
		test_data.chunks_to_drop.push_back(coordinates);
	}

	return test_data;
}
