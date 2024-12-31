//
// Created by Lukas Karafiat on 21.02.24.
//

#ifndef VWTCHUNKLOADINGTESTINGANDBENCHING_TEST_DATA_H
#define VWTCHUNKLOADINGTESTINGANDBENCHING_TEST_DATA_H

#include <filesystem>
#include <vector>

#include "chunkloader/chunk_loader.h"

using ChunkCoordinates = std::vector<ChunkCoordinate>;

struct TestData
{
	std::filesystem::path volume_path;
	std::vector<ChunkCoordinates> chunks_to_preload;
	std::vector<ChunkCoordinates> chunks_to_load;
	std::vector<ChunkCoordinates> chunks_to_drop;
};

#endif //VWTCHUNKLOADINGTESTINGANDBENCHING_TEST_DATA_H
