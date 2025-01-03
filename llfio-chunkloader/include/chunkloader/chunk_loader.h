//
// Created by Christofer Held on 18.01.2024.
//

#ifndef CHUNKLOADER_H
#define CHUNKLOADER_H

#include <utility>
#include <vector>
#include <memory>
#include <span>

struct ChunkCoordinate
{
	int64_t x, y, z;

	int64_t
	operator<(ChunkCoordinate other) const
	{
		if (x != other.x)
		{
			return x < other.x;
		}
		if (y != other.y)
		{
			return y < other.y;
		}
		return z < other.z;
	};

};

struct Chunk
{
	ChunkCoordinate     coordinate;
	std::span<uint16_t> data;

	Chunk(ChunkCoordinate coordinate, std::span<uint16_t> data) : coordinate(coordinate), data(data)
	{}
};

class ChunkLoader
{
public:
	/**
	 * Blocking call that gets a list of Coordinates and returns the chunks asap.
	 * @param coordinates of the chunk
	 * @return shared ptrs of the loaded chunks
	 */
	virtual std::vector<std::shared_ptr<Chunk>>
	load(const std::vector<ChunkCoordinate>& coordinates) = 0;

	/**
	 * Non Blocking call. This tells the ChunkLoader what we expect to need soon
	 * @param coordinates is a list of chunks I will properly load soon.
	 */
	virtual void
	preload(std::vector<ChunkCoordinate> coordinates) = 0;

	/**
	 * Blocking call. Returns which chunks are prepared. I need that to clean up. Must contain all cached chunks.
	 * @return a list of ChunksCoordinates. That are currently stored in memory
	 */
	virtual std::vector<ChunkCoordinate>
	prepared() = 0;

	/**
	 * Non Blocking call Tells the Chunkloader which chunks are not needed anymore.
	 * @param coordinates the list of coordinates
	 */
	virtual void
	drop(std::vector<ChunkCoordinate> coordinates) = 0;

	/**
	 * Blocking call. Writes all given chunks back to the persistent volume.
	 * @param chunks is a list of chunks that are written back to the persistent volume
	 */
	virtual void
	write(std::vector<Chunk> chunks) = 0;

	/**
	 * Virtual Destructor
	 */
	virtual ~ChunkLoader() = default;
};

#endif // CHUNKLOADER_H
