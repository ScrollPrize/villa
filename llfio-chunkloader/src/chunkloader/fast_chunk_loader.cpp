//
// Created by Christofer Held on 18.01.2024.
//
#include "chunkloader/fast_chunk_loader.h"
#include <filesystem>
#include <algorithm>

#include <nlohmann/json.hpp>
#include <fstream>
#include <ranges>

#include <llfio/llfio.hpp>

namespace fs = std::filesystem;
namespace llfio = LLFIO_V2_NAMESPACE;

static llfio::mapped_file_handle mapped_file_handle;

// Function to convert uint16_t from big endian
static
uint16_t
from_big_endian(uint16_t value)
{
	if constexpr (std::endian::native == std::endian::little)
	{
		return (value >> 8) | (value << 8);
	}
	else
	{
		return value; // already big-endian
	}
}

// Function to convert uint16_t to big endian
static
uint16_t
to_big_endian(uint16_t value)
{
	if constexpr (std::endian::native == std::endian::little)
	{
		return (value >> 8) | (value << 8);
	}
	else
	{
		return value; // already big-endian
	}
}

FastChunkLoader::FastChunkLoader(fs::path const& chunked_volume_path, uint64_t maximum_memory_consumption)
{
	info = ChunkedVolumeInformation::read_from_file(chunked_volume_path / "meta.json");

	uint64_t chunk_pixel_count = info.chunk_size * info.chunk_size * info.chunk_size;
	uint64_t chunk_memory_size = chunk_pixel_count * sizeof(uint16_t);

	uint64_t chunk_count_for_memory = maximum_memory_consumption / chunk_memory_size;

	memory_pool = std::vector<uint16_t>(chunk_count_for_memory * chunk_pixel_count);

	free_list = std::vector<std::span<uint16_t>>(chunk_count_for_memory);

	for (int i = 0; i < chunk_count_for_memory; i++)
	{ free_list[i] = std::span<uint16_t>(memory_pool.data() + i * chunk_pixel_count, chunk_pixel_count); }

	extern llfio::mapped_file_handle mapped_file_handle;

	mapped_file_handle = llfio::mapped_file({}, chunked_volume_path
	                                            / "volume.chunk_pkg", llfio::file_handle::mode::write).value();

	auto length = mapped_file_handle.maximum_extent().value();

	chunk_dimension.width  = info.chunk_size;
	chunk_dimension.height = info.chunk_size;
	chunk_dimension.depth  = info.chunk_size;

	chunk_count.x = info.width / chunk_dimension.width;
	chunk_count.y = info.height / chunk_dimension.height;
	chunk_count.z = info.slices / chunk_dimension.depth;
}

std::vector<std::shared_ptr<Chunk>>
FastChunkLoader::load(std::vector<ChunkCoordinate> const& coordinates)
{
	// prepare return values
	std::vector<std::shared_ptr<Chunk>> return_values(coordinates.size());

	uint64_t index = 0;

	std::vector<ChunkCoordinate> not_found_coordinates;
	not_found_coordinates.reserve(coordinates.size());

	// search already stored chunks + add to list
	for (ChunkCoordinate const& coordinate: coordinates)
	{
		if (occupied_list.find(coordinate) == occupied_list.end())
		{ not_found_coordinates.push_back(coordinate); }
		else
		{ return_values[index++] = std::make_shared<Chunk>(coordinate, occupied_list[coordinate]); }
	}

	if (free_list.size() < not_found_coordinates.size())
	{
		throw std::exception();
	}

	// fill chunks
#pragma omp parallel for
	for (uint64_t i = 0; i < not_found_coordinates.size(); i++)
	{ read_chunk(not_found_coordinates[i], free_list[free_list.size() - 1 - i]); }

	// take chunks
	for (ChunkCoordinate const& coordinate: not_found_coordinates)
	{
		occupied_list
			.insert(std::pair<ChunkCoordinate, std::span<uint16_t>>(
				ChunkCoordinate(coordinate), std::span<uint16_t>(free_list.back())));
		return_values[index++] = std::make_shared<Chunk>(coordinate, free_list.back());
		free_list.pop_back();
	}

	return return_values;
}

void
FastChunkLoader::read_chunk(ChunkCoordinate coord, std::span<uint16_t>& memory)
{
	uint64_t step_size = chunk_dimension.width * chunk_dimension.height * chunk_dimension.depth;

	uint64_t first_index = coord.y * chunk_count.y * chunk_count.x * step_size
	                       + coord.y * chunk_count.x * step_size
	                       + coord.x * step_size;

	uint64_t last_index = first_index + chunk_dimension.width * chunk_dimension.height * chunk_dimension.depth;

	uint16_t *file_memory = reinterpret_cast<uint16_t *>(mapped_file_handle.address());

	std::copy(file_memory + first_index, file_memory + last_index, memory.begin());

	// convert from big-endian to native format
	std::transform(memory.begin(), memory.end(), memory.begin(), from_big_endian);
}

void
FastChunkLoader::preload_chunk()
{}

void
FastChunkLoader::preload(std::vector<ChunkCoordinate> coordinates)
{}

std::vector<ChunkCoordinate>
FastChunkLoader::prepared()
{
	std::ranges::elements_view   chunk_coordinates = std::views::keys(occupied_list);
	std::vector<ChunkCoordinate> prepared{chunk_coordinates.begin(), chunk_coordinates.end()};

	return prepared;
}

void
FastChunkLoader::drop(std::vector<ChunkCoordinate> coordinates)
{
	for (ChunkCoordinate const& coordinate: coordinates)
	{
		if (occupied_list.find(coordinate) != occupied_list.end())
		{
			free_list.push_back(occupied_list.at(coordinate));
			occupied_list.erase(coordinate);
		}
	}
}

void
FastChunkLoader::write_chunk(ChunkCoordinate coord, std::span<uint16_t> memory)
{
	uint64_t step_size = chunk_dimension.width * chunk_dimension.height * chunk_dimension.depth;

	uint64_t first_index = coord.y * chunk_count.y * chunk_count.x * step_size
	                       + coord.y * chunk_count.x * step_size
	                       + coord.x * step_size;

	uint64_t last_index = first_index + chunk_dimension.width * chunk_dimension.height * chunk_dimension.depth;

	uint16_t *file_memory = reinterpret_cast<uint16_t *>(mapped_file_handle.address());

	std::copy(memory.begin(), memory.end(), file_memory + first_index);

	// convert from native format to big-endian
	std::transform(file_memory + first_index, file_memory + last_index, file_memory + first_index, to_big_endian);
}

void
FastChunkLoader::write(std::vector<Chunk> chunks)
{
#pragma omp parallel for
	for (Chunk const& chunk: chunks)
	{ write_chunk(chunk.coordinate, chunk.data); }
}