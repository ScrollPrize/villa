#pragma once
// Lossless vc-delta3d (ScrollPrize/vc-delta3d) chunk codec for uint8 volumes, as used by
// vc_reader / VC3D (Zarr compressor {"id": "vc-delta3d", "quant": 1}).
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace helirec {

bool delta3d_available();
// one (z, y, x) uint8 chunk -> encoded payload (lossless, default preset)
// preset: "fast" (default of vc_reader, preset frames, needs vc-delta3d >= the pinned revision to read),
// "medium"/"slow"/"slower", or "delta" (legacy D3D1 frames, readable by older vc_reader builds)
std::vector<uint8_t> delta3d_compress_u8(const uint8_t* data, int z, int y, int x, const char* preset = "fast");
// payload -> out (out_size bytes); throws on malformed input or size mismatch
void delta3d_decompress(const uint8_t* src, std::size_t n, uint8_t* out, std::size_t out_size);

}  // namespace helirec
