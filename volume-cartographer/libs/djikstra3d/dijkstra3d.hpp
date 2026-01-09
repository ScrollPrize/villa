/*
 * An implementation of a Edgar Dijkstra's Shortest Path Algorithm.
 * An absolute classic.
 * 
 * E. W. Dijkstra.
 * "A Note on Two Problems in Connexion with Graphs"
 * Numerische Mathematik 1. pp. 269-271. (1959)
 *
 * Of course, I use a priority queue.
 *
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton University
 * Date: August 2018
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <vector>

#include "./hedly.h"
#include "./libdivide.h"

#ifndef DIJKSTRA3D_HPP
#define DIJKSTRA3D_HPP

#define NHOOD_SIZE 26

namespace dijkstra {

#define sq(x) ((x) * (x))
static size_t _dummy_max_loc;

// helper function to compute 2D anisotropy ("_s" = "square")
inline float _s(const float wa, const float wb) {
  return std::sqrt(wa * wa + wb * wb);
}

// helper function to compute 3D anisotropy ("_c" = "cube")
inline float _c(const float wa, const float wb, const float wc) {
  return std::sqrt(wa * wa + wb * wb + wc * wc);
}

// helper function to compute cross product
inline float _cross(const float sx, const float sy, const float sz, 
const float tx, const float ty, const float tz) {
  return std::sqrt((sy*tz - sz*ty) * (sy*tz - sz*ty) + (sz*tx - sx*tz) * (sz*tx - sx*tz) + (sx*ty - sy*tx) * (sx*ty - sy*tx) / 
          std::sqrt(sx*sx + sy*sy + sz*sz) / std::sqrt(tx*tx + ty*ty + tz*tz));
}

inline float* fill(float *arr, const float value, const size_t size) {
  for (size_t i = 0; i < size; i++) {
    arr[i] = value;
  }
  return arr;
}

void connectivity_check(int connectivity) {
  if (connectivity != 6 && connectivity != 18 && connectivity != 26) {
    throw std::runtime_error("Only 6, 18, and 26 connectivities are supported.");
  }
}

template <typename OUT = uint32_t>
inline std::vector<OUT> query_shortest_path(const OUT* parents, const OUT target) {
  std::vector<OUT> path;
  OUT loc = target;
  while (parents[loc]) {
    path.push_back(loc);
    loc = parents[loc] - 1; // offset by 1 to disambiguate the 0th index
  }
  path.push_back(loc);

  return path;
}

void compute_neighborhood_helper_6(
  int *neighborhood, 
  const int x, const int y, const int z,
  const uint64_t sx, const uint64_t sy, const uint64_t sz
) {

  const int sxy = sx * sy;

  // 6-hood
  neighborhood[0] = -1 * (x > 0); // -x
  neighborhood[1] = (x < (static_cast<int>(sx) - 1)); // +x
  neighborhood[2] = -static_cast<int>(sx) * (y > 0); // -y
  neighborhood[3] = static_cast<int>(sx) * (y < static_cast<int>(sy) - 1); // +y
  neighborhood[4] = -sxy * static_cast<int>(z > 0); // -z
  neighborhood[5] = sxy * (z < static_cast<int>(sz) - 1); // +z
}

void compute_neighborhood_helper_18(
  int *neighborhood, 
  const int x, const int y, const int z,
  const uint64_t sx, const uint64_t sy, const uint64_t sz
) {
  // 6-hood
  compute_neighborhood_helper_6(neighborhood, x,y,z, sx,sy,sz);

  // 18-hood

  // xy diagonals
  neighborhood[6] = (neighborhood[0] + neighborhood[2]) * (neighborhood[0] && neighborhood[2]); // up-left
  neighborhood[7] = (neighborhood[0] + neighborhood[3]) * (neighborhood[0] && neighborhood[3]); // up-right
  neighborhood[8] = (neighborhood[1] + neighborhood[2]) * (neighborhood[1] && neighborhood[2]); // down-left
  neighborhood[9] = (neighborhood[1] + neighborhood[3]) * (neighborhood[1] && neighborhood[3]); // down-right

  // yz diagonals
  neighborhood[10] = (neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]); // up-left
  neighborhood[11] = (neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]); // up-right
  neighborhood[12] = (neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]); // down-left
  neighborhood[13] = (neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]); // down-right

  // xz diagonals
  neighborhood[14] = (neighborhood[0] + neighborhood[4]) * (neighborhood[0] && neighborhood[4]); // up-left
  neighborhood[15] = (neighborhood[0] + neighborhood[5]) * (neighborhood[0] && neighborhood[5]); // up-right
  neighborhood[16] = (neighborhood[1] + neighborhood[4]) * (neighborhood[1] && neighborhood[4]); // down-left
  neighborhood[17] = (neighborhood[1] + neighborhood[5]) * (neighborhood[1] && neighborhood[5]); // down-right
}

void compute_neighborhood_helper_26(
  int *neighborhood, 
  const int x, const int y, const int z,
  const uint64_t sx, const uint64_t sy, const uint64_t sz
) {
  compute_neighborhood_helper_18(neighborhood, x,y,z, sx,sy,sz);
  
  // 26-hood

  // Now the eight corners of the cube
  neighborhood[18] = (neighborhood[0] + neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]);
  neighborhood[19] = (neighborhood[1] + neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]);
  neighborhood[20] = (neighborhood[0] + neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]);
  neighborhood[21] = (neighborhood[0] + neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]);
  neighborhood[22] = (neighborhood[1] + neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]);
  neighborhood[23] = (neighborhood[1] + neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]);
  neighborhood[24] = (neighborhood[0] + neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]);
  neighborhood[25] = (neighborhood[1] + neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]);
}

inline void compute_neighborhood(
  int *neighborhood, 
  const int x, const int y, const int z,
  const uint64_t sx, const uint64_t sy, const uint64_t sz,
  const int connectivity = 26, const uint32_t* voxel_connectivity_graph = NULL) {

  if (connectivity == 26) {
    compute_neighborhood_helper_26(neighborhood, x, y, z, sx, sy, sz);
  }
  else if (connectivity == 18) {
    compute_neighborhood_helper_18(neighborhood, x, y, z, sx, sy, sz);
  }
  else {
    compute_neighborhood_helper_6(neighborhood, x, y, z, sx, sy, sz);
  }

  if (voxel_connectivity_graph == NULL) {
    return;
  }

  uint64_t loc = x + sx * (y + sy * z);
  uint32_t graph = voxel_connectivity_graph[loc];

  // graph conventions are defined here:
  // https://github.com/seung-lab/connected-components-3d/blob/3.2.0/cc3d_graphs.hpp#L73-L92

  // 6-hood
  neighborhood[0] *= ((graph & 0b000010) > 0); // -x
  neighborhood[1] *= ((graph & 0b000001) > 0); // +x
  neighborhood[2] *= ((graph & 0b001000) > 0); // -y
  neighborhood[3] *= ((graph & 0b000100) > 0); // +y
  neighborhood[4] *= ((graph & 0b100000) > 0); // -z
  neighborhood[5] *= ((graph & 0b010000) > 0); // +z

  // 18-hood

  // xy diagonals
  neighborhood[6] *= ((graph & 0b1000000000) > 0); // up-left -x,-y
  neighborhood[7] *= ((graph & 0b0010000000) > 0); // up-right -x,+y
  neighborhood[8] *= ((graph & 0b0100000000) > 0); // down-left +x,-y
  neighborhood[9] *= ((graph & 0b0001000000) > 0); // down-right +x,+y

  // yz diagonals
  neighborhood[10] *= ((graph & 0b100000000000000000) > 0); // up-left -y,-z
  neighborhood[11] *= ((graph & 0b000010000000000000) > 0); // up-right -y,+z
  neighborhood[12] *= ((graph & 0b010000000000000000) > 0); // down-left +y,-z
  neighborhood[13] *= ((graph & 0b000001000000000000) > 0); // down-right +y,+z

  // xz diagonals
  neighborhood[14] *= ((graph & 0b001000000000000000) > 0); // up-left, -x,-z
  neighborhood[15] *= ((graph & 0b000000100000000000) > 0); // up-right, -x,+z
  neighborhood[16] *= ((graph & 0b000100000000000000) > 0); // down-left +x,-z
  neighborhood[17] *= ((graph & 0b000000010000000000) > 0); // down-right +x,+z

  // 26-hood

  // Now the eight corners of the cube
  neighborhood[18] *= ((graph & 0b10000000000000000000000000) > 0); // -x,-y,-z
  neighborhood[19] *= ((graph & 0b01000000000000000000000000) > 0); // +x,-y,-z
  neighborhood[20] *= ((graph & 0b00100000000000000000000000) > 0); // -x,+y,-z
  neighborhood[21] *= ((graph & 0b00001000000000000000000000) > 0); // -x,-y,+z
  neighborhood[22] *= ((graph & 0b00010000000000000000000000) > 0); // +x,+y,-z
  neighborhood[23] *= ((graph & 0b00000100000000000000000000) > 0); // +x,-y,+z
  neighborhood[24] *= ((graph & 0b00000010000000000000000000) > 0); // -x,+y,+z
  neighborhood[25] *= ((graph & 0b00000001000000000000000000) > 0); // +x,+y,+z
}


template <typename T = uint32_t>
class HeapNode {
public:
  float key; 
  T value;

  HeapNode() {
    key = 0;
    value = 0;
  }

  HeapNode (float k, T val) {
    key = k;
    value = val;
  }

  HeapNode (const HeapNode<T> &h) {
    key = h.key;
    value = h.value;
  }
};

template <typename T = uint32_t>
struct HeapNodeCompare {
  bool operator()(const HeapNode<T> &t1, const HeapNode<T> &t2) const {
    return t1.key >= t2.key;
  }
};

#define DIJKSTRA_3D_PREFETCH_26WAY(field, loc) \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) + sxy - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - sxy - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) + sxy + sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) + sxy - sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - sxy + sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - sxy - sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) + sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - sx - 1]), 0, 1);

/* Perform dijkstra's shortest path algorithm
 * on a 3D image grid. Vertices are voxels and
 * edges are the 26 nearest neighbors (except
 * for the edges of the image where the number
 * of edges is reduced).
 *
 * For given input voxels A and B, the edge
 * weight from A to B is B and from B to A is
 * A. All weights must be non-negative (incl. 
 * negative zero).
 *
 * I take advantage of negative weights to mean
 * "visited".
 *
 * Parameters:
 *  T* field: Input weights. T can be be a floating or 
 *     signed integer type, but not an unsigned int.
 *  sx, sy, sz: size of the volume along x,y,z axes in voxels.
 *  source: 1D index of starting voxel
 *  target: 1D index of target voxel
 *
 * Returns: vector containing 1D indices of the path from
 *   source to target including source and target.
 */
template <typename T, typename OUT = uint32_t>
std::vector<OUT> dijkstra3d(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, const size_t target,
    const int connectivity = 26, 
    const uint32_t* voxel_connectivity_graph = NULL
  ) {

  connectivity_check(connectivity);

  if (source == target) {
    return std::vector<OUT>{ static_cast<OUT>(source) };
  }

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;
  
  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  std::unique_ptr<float[]> dist(new float[voxels]());
  std::unique_ptr<OUT[]> parents(new OUT[voxels]());
  fill(dist.get(), +INFINITY, voxels);
  dist[source] = -0;

  int neighborhood[NHOOD_SIZE] = {};

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue;
  queue.emplace(0.0, source);

  size_t loc;
  float delta;
  size_t neighboridx;

  int x, y, z;
  bool target_reached = false;

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();
    
    if (std::signbit(dist[loc])) {
      continue;
    }

    // As early as possible, start fetching the
    // data from RAM b/c the annotated lines below
    // have 30-50% cache miss.
    DIJKSTRA_3D_PREFETCH_26WAY(field, loc)
    DIJKSTRA_3D_PREFETCH_26WAY(dist, loc)

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity, voxel_connectivity_graph);

    for (int i = 0; i < connectivity; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = static_cast<float>(field[neighboridx]); // high cache miss

      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (dist[loc] + delta < dist[neighboridx]) { // high cache miss
        dist[neighboridx] = dist[loc] + delta;
        parents[neighboridx] = loc + 1; // +1 to avoid 0 ambiguity

        // Dijkstra, Edgar. "Go To Statement Considered Harmful".
        // Communications of the ACM. Vol. 11. No. 3 March 1968. pp. 147-148
        if (neighboridx == target) {
          target_reached = true;
          goto OUTSIDE;
        }

        queue.emplace(dist[neighboridx], neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  OUTSIDE:
  dist.reset();

  std::vector<OUT> path;
  // if voxel graph supplied, it's possible 
  // to never reach target.
  if (target_reached) { 
    path = query_shortest_path<OUT>(parents.get(), target);
  }

  return path;
}

template <typename T, typename OUT = uint32_t>
std::vector<OUT> dijkstra3d_anisotropy(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, const size_t target,
    const int connectivity = 26, 
    const float wx = 1.0, const float wy = 1.0, const float wz = 1.0,
    const uint32_t* voxel_connectivity_graph = NULL
  ) {

  connectivity_check(connectivity);

  if (source == target) {
    return std::vector<OUT>{ static_cast<OUT>(source) };
  }

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;
  
  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  std::unique_ptr<float[]> dist(new float[voxels]());
  std::unique_ptr<OUT[]> parents(new OUT[voxels]());
  fill(dist.get(), +INFINITY, voxels);
  dist[source] = -0;

  int neighborhood[NHOOD_SIZE] = {};

  float neighbor_multiplier[NHOOD_SIZE] = { 
    wx, wx, wy, wy, wz, wz, // axial directions (6)
    
    // square diagonals (12)
    _s(wx, wy), _s(wx, wy), _s(wx, wy), _s(wx, wy),  
    _s(wy, wz), _s(wy, wz), _s(wy, wz), _s(wy, wz),
    _s(wx, wz), _s(wx, wz), _s(wx, wz), _s(wx, wz),

    // cube diagonals (8)
    _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), 
    _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz)
  };

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue;
  queue.emplace(0.0, source);

  size_t loc;
  float delta;
  size_t neighboridx;

  int x, y, z;
  bool target_reached = false;

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();
    
    if (std::signbit(dist[loc])) {
      continue;
    }

    // As early as possible, start fetching the
    // data from RAM b/c the annotated lines below
    // have 30-50% cache miss.
    DIJKSTRA_3D_PREFETCH_26WAY(field, loc)
    DIJKSTRA_3D_PREFETCH_26WAY(dist, loc)

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity, voxel_connectivity_graph);

    for (int i = 0; i < connectivity; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = static_cast<float>(field[neighboridx]) * neighbor_multiplier[i]; // high cache miss
      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (dist[loc] + delta < dist[neighboridx]) { // high cache miss
        dist[neighboridx] = dist[loc] + delta;
        parents[neighboridx] = loc + 1; // +1 to avoid 0 ambiguity

        // Dijkstra, Edgar. "Go To Statement Considered Harmful".
        // Communications of the ACM. Vol. 11. No. 3 March 1968. pp. 147-148
        if (neighboridx == target) {
          target_reached = true;
          goto OUTSIDE;
        }

        queue.emplace(dist[neighboridx], neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  OUTSIDE:
  dist.reset();

  std::vector<OUT> path;
  // if voxel graph supplied, it's possible 
  // to never reach target.
  if (target_reached) { 
    path = query_shortest_path<OUT>(parents.get(), target);
  }

  return path;
}

template <typename OUT = uint32_t>
std::vector<OUT> binary_dijkstra3d(
  uint8_t* field, 
  const size_t sx, const size_t sy, const size_t sz, 
  const size_t source, const size_t target,
  const int connectivity = 26, 
  const float wx = 1.0, const float wy = 1.0, const float wz = 1.0,
  const bool euclidean_distance = true,
  const uint32_t* voxel_connectivity_graph = NULL,
  const uint8_t background_color = 0
) {

  connectivity_check(connectivity);

  if (field[source] == background_color) {
    return std::vector<OUT>();
  }
  else if (source == target) {
    return std::vector<OUT>{ static_cast<OUT>(source) };
  }

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;
  
  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  std::unique_ptr<float[]> dist(new float[voxels]());
  std::unique_ptr<OUT[]> parents(new OUT[voxels]());
  fill(dist.get(), +INFINITY, voxels);
  dist[source] = 0.0;

  int neighborhood[NHOOD_SIZE] = {};

  float neighbor_multiplier[NHOOD_SIZE] = { 
    wx, wx, wy, wy, wz, wz, // axial directions (6)
    
    // square diagonals (12)
    _s(wx, wy), _s(wx, wy), _s(wx, wy), _s(wx, wy),  
    _s(wy, wz), _s(wy, wz), _s(wy, wz), _s(wy, wz),
    _s(wx, wz), _s(wx, wz), _s(wx, wz), _s(wx, wz),

    // cube diagonals (8)
    _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), 
    _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz)
  };
  if (!euclidean_distance) {
    fill(neighbor_multiplier, 1.0, NHOOD_SIZE);
  }

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue;
  queue.emplace(0.0, source);

  size_t loc;
  size_t neighboridx;
  float new_dist = 0;

  int x, y, z;
  bool target_reached = false;

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();

    if (std::signbit(dist[loc])) {
      continue;
    }

    // As early as possible, start fetching the
    // data from RAM b/c the annotated lines below
    // have 30-50% cache miss.
    DIJKSTRA_3D_PREFETCH_26WAY(field, loc)
    DIJKSTRA_3D_PREFETCH_26WAY(dist, loc)

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity, voxel_connectivity_graph);

    for (int i = 0; i < connectivity; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      
      if (field[neighboridx] == background_color) {
        continue;
      }

      new_dist = dist[loc] + neighbor_multiplier[i];

      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (new_dist < dist[neighboridx]) { // high cache miss
        dist[neighboridx] = new_dist;
        parents[neighboridx] = loc + 1; // +1 to avoid 0 ambiguity

        // Dijkstra, Edgar. "Go To Statement Considered Harmful".
        // Communications of the ACM. Vol. 11. No. 3 March 1968. pp. 147-148
        if (neighboridx == target) {
          target_reached = true;
          goto OUTSIDE;
        }

        queue.emplace(dist[neighboridx], neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  OUTSIDE:
  dist.reset();

  std::vector<OUT> path;
  // it's possible to never reach target.
  if (target_reached) {
    path = query_shortest_path<OUT>(parents.get(), target);
  }
  
  return path;
}

/* Perform dijkstra's shortest path algorithm
 * on a 3D image grid where the source is specified
 * but the target is a value on the grid (typically 0)
 * rather than a point.
 * 
 * Vertices are voxels and edges are the 26 nearest 
 * neighbors (except for the edges of the image 
 * where the number of edges is reduced).
 *
 * For given input voxels A and B, the edge
 * weight from A to B is B and from B to A is
 * A. All weights must be non-negative (incl. 
 * negative zero).
 *
 * I take advantage of negative weights to mean
 * "visited".
 *
 * Parameters:
 *  T* field: Input weights. T can be be a floating or 
 *     signed integer type, but not an unsigned int.
 *  sx, sy, sz: size of the volume along x,y,z axes in voxels.
 *  source: 1D index of starting voxel
 *  target: 1D index of target voxel
 *
 * Returns: vector containing 1D indices of the path from
 *   source to target including source and target.
 */
template <typename T, typename OUT = uint32_t>
std::vector<OUT> value_target_dijkstra3d(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, const T target,
    const int connectivity = 26, 
    const uint32_t* voxel_connectivity_graph = NULL
  ) {

  connectivity_check(connectivity);

  if (field[source] == target) {
    return std::vector<OUT>{ static_cast<OUT>(source) };
  }

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;
  
  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  std::unique_ptr<float[]> dist(new float[voxels]());
  std::unique_ptr<OUT[]> parents(new OUT[voxels]());
  fill(dist.get(), +INFINITY, voxels);
  dist[source] = -0;

  int neighborhood[NHOOD_SIZE] = {};

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue;
  queue.emplace(0.0, source);

  size_t loc;
  float delta;
  size_t neighboridx;

  int x, y, z;
  size_t target_loc = voxels;

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();
    
    if (std::signbit(dist[loc])) {
      continue;
    }

    // As early as possible, start fetching the
    // data from RAM b/c the annotated lines below
    // have 30-50% cache miss.
    DIJKSTRA_3D_PREFETCH_26WAY(field, loc)
    DIJKSTRA_3D_PREFETCH_26WAY(dist, loc)

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity, voxel_connectivity_graph);

    for (int i = 0; i < connectivity; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = static_cast<float>(field[neighboridx]); // high cache miss

      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (dist[loc] + delta < dist[neighboridx]) { // high cache miss
        dist[neighboridx] = dist[loc] + delta;
        parents[neighboridx] = loc + 1; // +1 to avoid 0 ambiguity

        // Dijkstra, Edgar. "Go To Statement Considered Harmful".
        // Communications of the ACM. Vol. 11. No. 3 March 1968. pp. 147-148
        if (delta == target) {
          target_loc = neighboridx;
          goto OUTSIDE;
        }

        queue.emplace(dist[neighboridx], neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  OUTSIDE:
  dist.reset();

  std::vector<OUT> path;
  // if voxel graph supplied, it's possible 
  // to never reach target.
  if (target_loc < voxels) { // voxels is an impossible target
    path = query_shortest_path<OUT>(parents.get(), target_loc);
  }

  return path;
}

// helper function for bidirectional_dijkstra
template <typename T, typename OUT>
inline void bidirectional_core(
    const size_t loc, 
    T* field, float *dist, OUT* parents, 
    int *neighborhood, int connectivity,
    std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> &queue
  ) {
  
  float delta;
  size_t neighboridx;

  for (int i = 0; i < connectivity; i++) {
    if (neighborhood[i] == 0) {
      continue;
    }

    neighboridx = loc + neighborhood[i];
    delta = static_cast<float>(field[neighboridx]);

    // Visited nodes are negative and thus the current node
    // will always be less than as field is filled with non-negative
    // integers.
    if (dist[loc] + delta < dist[neighboridx]) { 
      dist[neighboridx] = dist[loc] + delta;
      parents[neighboridx] = loc + 1; // +1 to avoid 0 ambiguity
      queue.emplace(dist[neighboridx], neighboridx);
    }
  }

  dist[loc] *= -1;
}

template <typename T, typename OUT = uint32_t>
std::vector<OUT> bidirectional_dijkstra3d(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, const size_t target,
    const int connectivity = 26,
    const uint32_t* voxel_connectivity_graph = NULL
  ) {

  connectivity_check(connectivity);

  if (source == target) {
    return std::vector<OUT>{ static_cast<OUT>(source) };
  }

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;
  
  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  float *dist_fwd = new float[voxels]();
  OUT *parents_fwd = new OUT[voxels]();

  float *dist_rev = new float[voxels]();
  OUT *parents_rev = new OUT[voxels]();

  fill(dist_fwd, +INFINITY, voxels);
  fill(dist_rev, +INFINITY, voxels);
  dist_fwd[source] = 0;
  dist_rev[target] = 0;

  int neighborhood[NHOOD_SIZE] = {};

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue_fwd;
  queue_fwd.emplace(dist_fwd[source], source);

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue_rev;
  queue_rev.emplace(dist_rev[target], target);

  size_t loc = source;
  size_t final_loc = source;
  int x, y, z;

  bool forward = false;
  float cost = INFINITY;

  std::function<float(size_t)> costfn = [field, target, dist_fwd, dist_rev](size_t loc) { 
    return abs(dist_rev[loc]) + abs(dist_fwd[loc]) + static_cast<float>(field[target]) - static_cast<float>(field[loc]);
  };

  while (!queue_fwd.empty() && !queue_rev.empty()) {
    forward = !forward;

    if (forward) {
      loc = queue_fwd.top().value;
      queue_fwd.pop();

      if (dist_rev[loc] < INFINITY) {
        if (costfn(loc) < cost) {
          cost = costfn(loc);
          final_loc = loc;
        }
        else {
          break;
        }
      }
      else if (std::signbit(dist_fwd[loc])) {
        continue;
      }
    }
    else {
      loc = queue_rev.top().value;
      queue_rev.pop();

      if (dist_fwd[loc] < INFINITY) {
        if (costfn(loc) < cost) {
          cost = costfn(loc);
          final_loc = loc;
        }
        else {
          break;
        }
      }
      else if (std::signbit(dist_rev[loc])) {
        continue;
      }
    }

    DIJKSTRA_3D_PREFETCH_26WAY(field, loc)

    if (forward) {
      DIJKSTRA_3D_PREFETCH_26WAY(dist_fwd, loc)
    }
    else {
      DIJKSTRA_3D_PREFETCH_26WAY(dist_rev, loc)
    }

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity, voxel_connectivity_graph);

    if (forward) {
      bidirectional_core<T,OUT>(
        loc, field, dist_fwd, parents_fwd, 
        neighborhood, connectivity, queue_fwd
      );
    }
    else {
      bidirectional_core<T,OUT>(
        loc, field, dist_rev, parents_rev, 
        neighborhood, connectivity, queue_rev
      );
    }
  }

  bool target_reached = (dist_fwd[final_loc] < INFINITY) && (dist_rev[final_loc] < INFINITY);

  delete [] dist_fwd;
  delete [] dist_rev;

  // We will always have a "meet in middle" victory condition
  // because we set it up so if fwd finds target or rev finds
  // source, that will count as "meeting in the middle" because
  // those points were initialized.

  std::vector<OUT> path_fwd, path_rev;

  if (target_reached) {
    path_rev = query_shortest_path<OUT>(parents_rev, final_loc);
    delete [] parents_rev;
    path_fwd = query_shortest_path<OUT>(parents_fwd, final_loc);
    delete [] parents_fwd;
    std::reverse(path_fwd.begin(), path_fwd.end());
    path_fwd.insert(path_fwd.end(), path_rev.begin() + 1, path_rev.end());
  }
  else {
    delete [] parents_rev;
    delete [] parents_fwd;
  }

  return path_fwd;
}

// A* using distance to target as a heuristic
// setting normalizer to a postive value allows you
// to manipulate the A* priority. By default it 
// is normalized to the field minimum which guarantees
// correctness.
template <typename T, typename OUT = uint32_t>
std::vector<OUT> compass_guided_dijkstra3d(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, const size_t target,
    const int connectivity = 26, float normalizer = -1,
    const uint32_t* voxel_connectivity_graph = NULL
  ) {

  connectivity_check(connectivity);

  if (source == target) {
    return std::vector<OUT>{ static_cast<OUT>(source) };
  }

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;
  
  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  std::unique_ptr<float[]> dist(new float[voxels]());
  std::unique_ptr<OUT[]> parents(new OUT[voxels]());
  fill(dist.get(), +INFINITY, voxels);
  dist[source] = 0;

  // Normalizer value must be positive. 
  // If negative, use min field value.
  if (normalizer < 0) {
    normalizer = static_cast<float>(field[0]);
    for (size_t i = 0; i < voxels; i++) {
      if (normalizer > static_cast<float>(field[i])) {
        normalizer = static_cast<float>(field[i]);
      }
    }
  }

  int neighborhood[NHOOD_SIZE] = {};

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue;
  queue.emplace(0.0, source);

  size_t loc;
  float delta;
  size_t neighboridx;

  int x, y, z;
  int tx, ty, tz;
  float heuristic_cost;
  bool target_reached = false;

  tz = target / fast_sxy;
  ty = (target - (tz * sxy)) / fast_sx;
  tx = target - sx * (ty + tz * sy);

  std::function<void(size_t)> xyzfn = [&x,&y,&z,power_of_two,xshift,yshift,sx,sy,sxy,fast_sx,fast_sxy](size_t l){
    if (power_of_two) {
      z = l >> (xshift + yshift);
      y = (l - (z << (xshift + yshift))) >> xshift;
      x = l - ((y + (z << yshift)) << xshift);
    }
    else {
      z = l / fast_sxy;
      y = (l - (z * sxy)) / fast_sx;
      x = l - sx * (y + z * sy);
    }
  };

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();
    
    if (std::signbit(dist[loc])) {
      continue;
    }

    DIJKSTRA_3D_PREFETCH_26WAY(field, loc)
    DIJKSTRA_3D_PREFETCH_26WAY(dist, loc)

    xyzfn(loc);

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity, voxel_connectivity_graph);

    for (int i = 0; i < connectivity; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = static_cast<float>(field[neighboridx]);

      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (dist[loc] + delta < dist[neighboridx]) { 
        dist[neighboridx] = dist[loc] + delta;
        parents[neighboridx] = loc + 1; // +1 to avoid 0 ambiguity

        if (neighboridx == target) {
          target_reached = true;
          goto OUTSIDE;
        }

        xyzfn(neighboridx);

        if (connectivity == 6) { // manhattan (L_1)
          heuristic_cost = static_cast<float>(
            abs(tx - x) + abs(ty - y) + abs(tz - z)
          );
        }
        else if (connectivity == 18) {
          // The faces + edges case is weird... 
          // It's 8 connected on each plane, but you can't
          // move diagonally in 3D.
          heuristic_cost = static_cast<float>(
            std::min(
              std::min(
                std::max(abs(tx - x), abs(ty - y)) + abs(tz - z),
                std::max(abs(tx - x), abs(tz - z)) + abs(ty - y)
              ),
              std::max(abs(ty - y), abs(tz - z)) + abs(tx - x)
            )
          );
        }
        else { // chebychev (L_inf)
          heuristic_cost = static_cast<float>(
            std::max(std::max(abs(tx - x), abs(ty - y)), abs(tz - z)) 
          );
        }

        queue.emplace(dist[neighboridx] + normalizer * heuristic_cost, neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  OUTSIDE:
  dist.reset();

  std::vector<OUT> path;
  if (target_reached) {
    path = query_shortest_path<OUT>(parents.get(), target);
  }

  return path;
}

template <typename T, typename OUT = uint32_t>
std::vector<OUT> compass_guided_dijkstra3d_anisotropy_line_preference(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, const size_t target,
    const int connectivity = 26, float normalizer = -1,
    float line_preference_weight = 1.0,
    const float wx = 1.0, const float wy = 1.0, const float wz = 1.0,
    const uint32_t* voxel_connectivity_graph = NULL
  ) {

  connectivity_check(connectivity);

  if (source == target) {
    return std::vector<OUT>{ static_cast<OUT>(source) };
  }

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;
  
  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  std::unique_ptr<float[]> dist(new float[voxels]());
  std::unique_ptr<OUT[]> parents(new OUT[voxels]());
  fill(dist.get(), +INFINITY, voxels);
  dist[source] = 0;

  // Normalizer value must be positive. 
  // If negative, use min field value.
  if (normalizer < 0) {
    normalizer = static_cast<float>(field[0]);
    for (size_t i = 0; i < voxels; i++) {
      if (normalizer > static_cast<float>(field[i])) {
        normalizer = static_cast<float>(field[i]);
      }
    }
  }

  int neighborhood[NHOOD_SIZE] = {};

  float neighbor_multiplier[NHOOD_SIZE] = { 
    wx, wx, wy, wy, wz, wz, // axial directions (6)
    
    // square diagonals (12)
    _s(wx, wy), _s(wx, wy), _s(wx, wy), _s(wx, wy),  
    _s(wy, wz), _s(wy, wz), _s(wy, wz), _s(wy, wz),
    _s(wx, wz), _s(wx, wz), _s(wx, wz), _s(wx, wz),

    // cube diagonals (8)
    _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), 
    _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz)
  };

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue;
  queue.emplace(0.0, source);

  size_t loc;
  float delta;
  size_t neighboridx;

  int x, y, z;
  int srcx, srcy, srcz;
  int tx, ty, tz;
  float heuristic_cost;
  bool target_reached = false;

  tz = target / fast_sxy;
  ty = (target - (tz * sxy)) / fast_sx;
  tx = target - sx * (ty + tz * sy);

  srcz = source / fast_sxy;
  srcy = (source - (srcz * sxy)) / fast_sx;
  srcx = source - sx * (srcy + srcz * sy);

  std::function<void(size_t)> xyzfn = [&x,&y,&z,power_of_two,xshift,yshift,sx,sy,sxy,fast_sx,fast_sxy](size_t l){
    if (power_of_two) {
      z = l >> (xshift + yshift);
      y = (l - (z << (xshift + yshift))) >> xshift;
      x = l - ((y + (z << yshift)) << xshift);
    }
    else {
      z = l / fast_sxy;
      y = (l - (z * sxy)) / fast_sx;
      x = l - sx * (y + z * sy);
    }
  };

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();
    
    if (std::signbit(dist[loc])) {
      continue;
    }

    DIJKSTRA_3D_PREFETCH_26WAY(field, loc)
    DIJKSTRA_3D_PREFETCH_26WAY(dist, loc)

    xyzfn(loc);

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity, voxel_connectivity_graph);

    for (int i = 0; i < connectivity; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = static_cast<float>(field[neighboridx]) * neighbor_multiplier[i];

      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (dist[loc] + delta < dist[neighboridx]) { 
        dist[neighboridx] = dist[loc] + delta;
        parents[neighboridx] = loc + 1; // +1 to avoid 0 ambiguity

        if (neighboridx == target) {
          target_reached = true;
          goto OUTSIDE;
        }

        xyzfn(neighboridx);

        if (connectivity == 6) { // manhattan (L_1)
          heuristic_cost = static_cast<float>(
            abs(tx - x) * wx + abs(ty - y) * wy + abs(tz - z) * wz
          );
        }
        else if (connectivity == 18) {
          // The faces + edges case is weird... 
          // It's 8 connected on each plane, but you can't
          // move diagonally in 3D.
          heuristic_cost = static_cast<float>(
            std::min(
              std::min(
                std::max(abs(tx - x) * wx, abs(ty - y) * wy) + abs(tz - z) * wz,
                std::max(abs(tx - x) * wx, abs(tz - z) * wz) + abs(ty - y) * wy
              ),
              std::max(abs(ty - y) * wy, abs(tz - z) * wz) + abs(tx - x) * wx
            )
          );
        }
        else { // chebychev (L_inf)
          heuristic_cost = static_cast<float>(
            std::max(std::max(abs(tx - x) * wx, abs(ty - y) * wy), abs(tz - z) * wz)
          );
        }

        // weight the heuristic_cost with cross product
        heuristic_cost += _cross((tx - srcx) * wx, (ty - srcy) * wy, (tz - srcz) * wz, 
                                 (tx - x) * wx, (ty - y) * wy, (tz - z) * wz) * line_preference_weight;
        queue.emplace(dist[neighboridx] + normalizer * heuristic_cost, neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  OUTSIDE:
  dist.reset();

  std::vector<OUT> path;
  if (target_reached) {
    path = query_shortest_path<OUT>(parents.get(), target);
  }

  return path;
}



template <typename T, typename OUT = uint32_t>
std::vector<OUT> dijkstra2d(
    T* field, 
    const size_t sx, const size_t sy, 
    const size_t source, const size_t target,
    const int connectivity = 8
  ) {

  int adj_connectivity = connectivity;

  if (connectivity == 4) {
    adj_connectivity = 6; // 3d equivalent
  }
  else if (connectivity == 8) {
    adj_connectivity = 18; // 3d equivalent, 26 works too but 18 might be faster
  }

  return dijkstra3d<T,OUT>(field, sx, sy, 1, source, target, adj_connectivity);
}

template <typename T, typename OUT = uint32_t>
OUT* parental_field3d(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, OUT* parents = NULL,
    const int connectivity = 26,
    const uint32_t* voxel_connectivity_graph = NULL
  ) {

  connectivity_check(connectivity);

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;

  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  std::unique_ptr<float[]> dist(new float[voxels]());
  
  if (parents == NULL) {
    parents = new OUT[voxels]();
  }

  fill(dist.get(), +INFINITY, voxels);
  dist[source] = -0;

  int neighborhood[NHOOD_SIZE] = {};

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue;
  queue.emplace(0.0, source);

  size_t loc;
  float delta;
  size_t neighboridx;

  size_t x, y, z;

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();

    if (std::signbit(dist[loc])) {
      continue;
    }

    DIJKSTRA_3D_PREFETCH_26WAY(field, loc)
    DIJKSTRA_3D_PREFETCH_26WAY(dist, loc)

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity, voxel_connectivity_graph);

    for (int i = 0; i < connectivity; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = static_cast<float>(field[neighboridx]);

      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (dist[loc] + delta < dist[neighboridx]) { 
        dist[neighboridx] = dist[loc] + delta;
        parents[neighboridx] = loc + 1; // +1 to avoid 0 ambiguity
        queue.emplace(dist[neighboridx], neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  return parents;
}

template <typename T>
float* distance_field3d(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const std::vector<size_t> &sources, const size_t connectivity=26,
    const uint32_t* voxel_connectivity_graph = NULL,
    size_t &max_loc = _dummy_max_loc
  ) {

  connectivity_check(connectivity);

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;

  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  float *dist = new float[voxels]();
  fill(dist, +INFINITY, voxels);

  int neighborhood[NHOOD_SIZE] = {};

  std::priority_queue<HeapNode<size_t>, std::vector<HeapNode<size_t>>, HeapNodeCompare<size_t>> queue;

  float max_dist = -1;
  for (size_t source : sources) {
    max_loc = source;
    dist[source] = -0;
    queue.emplace(0.0, source);
  }

  size_t loc, next_loc;
  float delta;
  size_t neighboridx;

  size_t x, y, z;

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();

    if (max_dist < std::abs(dist[loc])) {
      max_dist = std::abs(dist[loc]);
      max_loc = loc;
    }

    if (std::signbit(dist[loc])) {
      continue;
    }

    if (!queue.empty()) {
      next_loc = queue.top().value;
      if (!std::signbit(dist[next_loc])) {

        // As early as possible, start fetching the
        // data from RAM b/c the annotated lines below
        // have 30-50% cache miss.
        DIJKSTRA_3D_PREFETCH_26WAY(field, next_loc)
        DIJKSTRA_3D_PREFETCH_26WAY(dist, next_loc)
      }
    }

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, 26, voxel_connectivity_graph);

    for (size_t i = 0; i < connectivity; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = static_cast<float>(field[neighboridx]);

      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (dist[loc] + delta < dist[neighboridx]) { 
        dist[neighboridx] = dist[loc] + delta;
        queue.emplace(dist[neighboridx], neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  for (size_t i = 0; i < voxels; i++) {
    dist[i] = std::fabs(dist[i]);
  }

  return dist;
}

template <typename T>
float* distance_field3d(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, const size_t connectivity=26,
    const uint32_t* voxel_connectivity_graph = NULL,
    size_t &max_loc = _dummy_max_loc
  ) {

  const std::vector<size_t> sources = { source };
  return distance_field3d<T>(
    field, 
    sx, sy, sz, 
    sources, connectivity, 
    voxel_connectivity_graph, max_loc
  );
}

/*  For the euclidean distance field calculation, it is 
    1-2 orders of magnitude faster to compute the distance field
    using an equation instead of dijkstra's method, but the eqn
    only works in free space (straight line of sight from the source).

    Therefore, when we have outside information that some distance 
    around the source point is free space, we can use a hybrid approach
    and use the equation to label the "safe" region and use dijkstra
    for the rest. It may be possible to extend the use of the equation
    with further development of this hybrid approach.

    Here's an unoptimized version of the equation that works for 
    <1,1,1> isotropic volumes only, but is easier to understand.
    The equation in actual use below is adapted for anisotropic 
    volumes.

    It works by starting with the manhattan distance between the
    source and the target point and then subtracts 3D diagonals,
    then 2D diagonals and adds their contribution back in. This works
    on the principle that the shortest distance between the source and 
    target requires that the maximum number of corner directions be 
    taken. Note that this equation sees distance the way the dijkstra
    approach would, it is quite different from a simple euclidean distance.

        float corners = std::min(std::min(dx, dy), dz);
        float edge_xy = std::min(dx, dy) - corners;
        float edge_yz = std::min(dy, dz) - corners;
        float edge_xz = std::min(dx, dz) - corners;
        float edges = edge_xy + edge_yz + edge_xz;

        float faces = (dx + dy + dz) - 2 * edges - 3 * corners;

        dist[loc] = corners * sqrt(3) + edges * sqrt(2) + faces;

    In order to integrate with the following dijkstra method, the 
    fringe of the free space is pushed onto the dijkstra priority 
    queue. The non-fringe free-space is negated (x -1) in order to
    mark it as visited. Dijkstra will then find the minimum of the 
    fringe and proceed normally.

    For reference, in a 512^3, dijkstra's method appears to work at 
    about 1 MVx/sec while this method is about 85 MVx/sec on the same
    hardware. Tests of a simpler version of the equation (no radius testing,
    less support for anisotropy) were working at 210 MVx/sec so some room
    for optimization is likely.

    In a field test in Kimimaro on a cell body starting from the point of
    maximum distance from the boundary, the timing improved from 20 sec to
    10 sec, so a 2x overall performance increase for a single use of the 
    free space eqn.

    Returns: a maximum value location computed (there may be ties)
 */
size_t edf_free_space(
    uint8_t* field, float* dist,
    std::priority_queue<HeapNode<size_t>, std::vector<HeapNode<size_t>>, HeapNodeCompare<size_t>> &queue,
    const int64_t source, const float safe_radius,
    const int64_t sx, const int64_t sy, const int64_t sz, 
    const float wx, const float wy, const float wz
  ) {

  const int64_t sxy = sx * sy;

  int64_t src_z = source / sxy;
  int64_t src_y = (source - (src_z * sxy)) / sx;
  int64_t src_x = source - sx * (src_y + src_z * sy);

  int64_t loc = 0;
  float radius = 0;
  float corner_increment = std::sqrt(sq(wx) + sq(wy) + sq(wz));

  const int64_t ZERO = 0; // set type of zero in a cross compiler friendly way
  int64_t xstart = std::max(ZERO, src_x - static_cast<int64_t>(safe_radius) - 1);
  int64_t ystart = std::max(ZERO, src_y - static_cast<int64_t>(safe_radius) - 1);
  int64_t zstart = std::max(ZERO, src_z - static_cast<int64_t>(safe_radius) - 1);

  int64_t xend = std::min(sx, src_x + static_cast<int64_t>(safe_radius) + 2);
  int64_t yend = std::min(sy, src_y + static_cast<int64_t>(safe_radius) + 2);
  int64_t zend = std::min(sz, src_z + static_cast<int64_t>(safe_radius) + 2);

  float max_dist = -1;
  size_t max_loc = sx * sy * sz + 1;

  for (int64_t z = zstart; z < zend; z++) {
    for (int64_t y = ystart; y < yend; y++) {
      for (int64_t x = xstart; x < xend; x++) {
        loc = x + sx * (y + sy * z);

        if (field[loc] == 0) {
          continue;
        }

        radius = std::sqrt(sq(wx * (x - src_x)) + sq(wy * (y - src_y)) + sq(wz * (z - src_z)));
        if (radius > safe_radius) {
          continue;
        }

        float dx = std::abs(static_cast<float>(x - src_x));
        float dy = std::abs(static_cast<float>(y - src_y));
        float dz = std::abs(static_cast<float>(z - src_z));

        float dxyz = std::min(std::min(dx, dy), dz);
        float dxy = std::min(dx, dy);
        float dyz = std::min(dy, dz);
        float dxz = std::min(dx, dz);

        dist[loc] = (dxyz * std::sqrt(wx * wx + wy * wy + wz * wz) 
           + wx * (dx - dxyz) + wy * (dy - dxyz) + wz * (dz - dxyz)
           + (dxy - dxyz) * (std::sqrt(wx * wx + wy * wy) - wx - wy)
           + (dxz - dxyz) * (std::sqrt(wx * wx + wz * wz) - wx - wz)
           + (dyz - dxyz) * (std::sqrt(wy * wy + wz * wz) - wy - wz)
        );

        if (dist[loc] > max_dist) {
          max_dist = dist[loc];
          max_loc = loc;
        }

        if (radius + corner_increment > safe_radius) {
          queue.emplace(dist[loc], loc);  
        }
        else {
          dist[loc] *= -1;
        }
      }
    }
  }

  return max_loc;
}

// really a chamfer distance
float* euclidean_distance_field3d(
    uint8_t* field, // really a boolean field
    const size_t sx, const size_t sy, const size_t sz, 
    const float wx, const float wy, const float wz, 
    const std::vector<size_t> &sources, 
    const float free_space_radius = 0,
    float* dist = NULL,
    const uint32_t* voxel_connectivity_graph = NULL,
    size_t &max_loc = _dummy_max_loc
  ) {

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;

  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  if (dist == NULL) {
    dist = new float[voxels]();
  }

  fill(dist, +INFINITY, voxels);

  int neighborhood[NHOOD_SIZE] = {};

  float neighbor_multiplier[NHOOD_SIZE] = { 
    wx, wx, wy, wy, wz, wz, // axial directions (6)
    
    // square diagonals (12)
    _s(wx, wy), _s(wx, wy), _s(wx, wy), _s(wx, wy),  
    _s(wy, wz), _s(wy, wz), _s(wy, wz), _s(wy, wz),
    _s(wx, wz), _s(wx, wz), _s(wx, wz), _s(wx, wz),

    // cube diagonals (8)
    _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), 
    _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz)
  };

  std::priority_queue<HeapNode<size_t>, std::vector<HeapNode<size_t>>, HeapNodeCompare<size_t>> queue;

  for (size_t source : sources) {
    dist[source] = -0;
    queue.emplace(0.0, source);
  }


  float max_dist = -1;
  max_loc = voxels + 1;

  if (free_space_radius > 0 && sources.size() == 1) {
    max_loc = edf_free_space(
      field, dist, queue, 
      sources[0], free_space_radius,
      sx, sy, sz,
      wx, wy, wz
    );
    max_dist = std::abs(dist[max_loc]);
  }

  size_t loc, next_loc;
  float new_dist;
  size_t neighboridx;

  size_t x, y, z;

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();

    if (max_dist < std::abs(dist[loc])) {
      max_dist = std::abs(dist[loc]);
      max_loc = loc;
    }

    if (std::signbit(dist[loc])) {
      continue;
    }

    if (!queue.empty()) {
      next_loc = queue.top().value;
      if (!std::signbit(dist[next_loc])) {

        // As early as possible, start fetching the
        // data from RAM b/c the annotated lines below
        // have 30-50% cache miss.
        DIJKSTRA_3D_PREFETCH_26WAY(field, next_loc)
        DIJKSTRA_3D_PREFETCH_26WAY(dist, next_loc)
      }
    }

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, 26, voxel_connectivity_graph);

    for (int i = 0; i < NHOOD_SIZE; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      if (field[neighboridx] == 0) {
        continue;
      }

      new_dist = dist[loc] + neighbor_multiplier[i];
      
      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (new_dist < dist[neighboridx]) { 
        dist[neighboridx] = new_dist;
        queue.emplace(new_dist, neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  for (size_t i = 0; i < voxels; i++) {
    dist[i] = std::fabs(dist[i]);
  }

  return dist;
}

float* euclidean_distance_field3d(
    uint8_t* field, // really a boolean field
    const size_t sx, const size_t sy, const size_t sz, 
    const float wx, const float wy, const float wz, 
    const size_t source, 
    const float free_space_radius = 0,
    float* dist = NULL,
    const uint32_t* voxel_connectivity_graph = NULL,
    size_t &max_loc = _dummy_max_loc
  ) {

  const std::vector<size_t> sources = { source };
  return euclidean_distance_field3d(
    field, 
    sx, sy, sz,
    wx, wy, wz,
    sources, free_space_radius, 
    dist, voxel_connectivity_graph,
    max_loc
  );
}


class HeapFeatureNode {
public:
  float key;
  uint64_t value;
  uint64_t source;

  HeapFeatureNode() {
    key = 0;
    value = 0;
  }

  HeapFeatureNode (float k, uint64_t val, uint64_t src) {
    key = k;
    value = val;
    source = src;
  }

  HeapFeatureNode (const HeapFeatureNode &h) {
    key = h.key;
    value = h.value;
    source = h.source;
  }
};

struct HeapFeatureNodeCompare {
  bool operator()(const HeapFeatureNode &t1, const HeapFeatureNode &t2) const {
    return t1.key >= t2.key;
  }
};

// returns a map of the nearest source vertex
template <typename OUT = uint64_t>
OUT* edf_with_feature_map(
  uint8_t* field, // really a boolean field
  const uint64_t sx, const uint64_t sy, const uint64_t sz, 
  const float wx, const float wy, const float wz, 
  const std::vector<uint64_t> &sources,
  float* dist = NULL, OUT* feature_map = NULL,
  size_t &max_loc = _dummy_max_loc
) {

  const uint64_t voxels = sx * sy * sz;
  const uint64_t sxy = sx * sy;

  const libdivide::divider<uint64_t> fast_sx(sx); 
  const libdivide::divider<uint64_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  bool clear_dist = false;
  if (dist == NULL) {
    dist = new float[voxels]();
    clear_dist = true;
  }
  if (feature_map == NULL) {
    feature_map = new OUT[voxels]();
  }

  fill(dist, +INFINITY, voxels);

  int neighborhood[NHOOD_SIZE] = {};

  float neighbor_multiplier[NHOOD_SIZE] = { 
    wx, wx, wy, wy, wz, wz, // axial directions (6)
    
    // square diagonals (12)
    _s(wx, wy), _s(wx, wy), _s(wx, wy), _s(wx, wy),  
    _s(wy, wz), _s(wy, wz), _s(wy, wz), _s(wy, wz),
    _s(wx, wz), _s(wx, wz), _s(wx, wz), _s(wx, wz),

    // cube diagonals (8)
    _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), 
    _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz)
  };

  std::priority_queue<
    HeapFeatureNode, std::vector<HeapFeatureNode>, HeapFeatureNodeCompare
  > queue;

  uint64_t src = 1;
  for (uint64_t source : sources) {
    dist[source] = -0;
    feature_map[source] = src;
    queue.emplace(0.0, source, src);
    src++;
  }

  uint64_t loc, next_loc;
  float new_dist;
  uint64_t neighboridx;

  uint64_t x, y, z;

  float max_dist = -1;
  max_loc = voxels + 1;

  while (!queue.empty()) {
    src = queue.top().source;
    loc = queue.top().value;
    queue.pop();

    if (max_dist < std::abs(dist[loc])) {
      max_dist = std::abs(dist[loc]);
      max_loc = loc;
    }

    if (std::signbit(dist[loc])) {
      continue;
    }

    if (!queue.empty()) {
      next_loc = queue.top().value;
      if (!std::signbit(dist[next_loc])) {

        // As early as possible, start fetching the
        // data from RAM b/c the annotated lines below
        // have 30-50% cache miss.
        DIJKSTRA_3D_PREFETCH_26WAY(field, next_loc)
        DIJKSTRA_3D_PREFETCH_26WAY(dist, next_loc)
      }
    }

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, NHOOD_SIZE, NULL);

    for (int i = 0; i < NHOOD_SIZE; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      if (field[neighboridx] == 0) {
        continue;
      }

      new_dist = dist[loc] + neighbor_multiplier[i];
      
      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (new_dist < dist[neighboridx]) { 
        dist[neighboridx] = new_dist;
        feature_map[neighboridx] = src;
        queue.emplace(new_dist, neighboridx, src);
      }
      else if (new_dist == dist[neighboridx] && src > feature_map[neighboridx]) {
        feature_map[neighboridx] = src;
      }
    }

    dist[loc] = -dist[loc];
  }

  if (clear_dist) {
    delete[] dist;
  }
  else {
    for (size_t i = 0; i < voxels; i++) {
      dist[i] = std::fabs(dist[i]);
    }
  }

  return feature_map;
}

#undef sq
#undef NHOOD_SIZE
#undef DIJKSTRA_3D_PREFETCH_26WAY

}; // namespace dijkstra3d

#endif