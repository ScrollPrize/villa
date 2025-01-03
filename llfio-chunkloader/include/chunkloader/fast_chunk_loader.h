//
// Created by Christofer Held on 18.01.2024.
//
#ifndef CHUNKLOADER_FASTCHUNKLOADER_H
#define CHUNKLOADER_FASTCHUNKLOADER_H

#include <string>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <stack>

#include "chunk_loader.h"
#include "util/chunked_volume_information.h"

class FastChunkLoader : public ChunkLoader
{
private:
	struct ChunkCount
	{
		uint64_t x, y, z;
	};

	struct ChunkDimension
	{
		uint64_t width, height, depth;
	};

	ChunkedVolumeInformation info;

	std::vector<uint16_t> memory_pool;

	std::vector<std::span<uint16_t>>               free_list;
	std::map<ChunkCoordinate, std::span<uint16_t>> occupied_list;

	ChunkCount     chunk_count;
	ChunkDimension chunk_dimension;

	void
	preload_chunk();

	void
	read_chunk(ChunkCoordinate coord, std::span<uint16_t>& memory);

	void
	write_chunk(ChunkCoordinate coord, std::span<uint16_t> memory);

public:
	explicit FastChunkLoader(std::filesystem::path const& chunked_volume_directory, uint64_t maximum_memory_consumption);

	~FastChunkLoader() override = default;

	/**
	 * Blocking call that gets a list of Coordinates and returns the chunks asap.
	 * @param coordinate of the chunk
	 * @return shared pointers of the loaded chunks
	 */
	std::vector<std::shared_ptr<Chunk>>
	load(const std::vector<ChunkCoordinate>& coordinate) override;

	/**
	 * Non Blocking call. This tells the ChunkLoader that we expect to need soon
	 * @param coordinates is a list of chunks i will properly load soon.
	 */
	void
	preload(std::vector<ChunkCoordinate> coordinates) override;

	/**
	 * Blocking call. Returns which Chunks are prepared. I need that to clean up. Must contain all chunk cached.
	 * @return a list of ChunksCoordinates. That are currently stored in memory
	 */
	std::vector<ChunkCoordinate>
	prepared() override;

	/**
	 * Non Blocking call tells the ChunkLoader which chunks are not needed anymore.
	 * @param coordinates the list of coordinates
	 */
	void
	drop(std::vector<ChunkCoordinate> coordinates) override;

	/**
	 * Blocking call. Writes all given chunks back to the persistent volume.
	 * @param chunks is a list of chunks that are written back to the persistent volume
	 */
	void
	write(std::vector<Chunk> chunks) override;
};

#endif // CHUNKLOADER_FASTCHUNKLOADER_H
