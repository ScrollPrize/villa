#pragma once

#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <nlohmann/json.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vc::fiber_tracer::detail {
inline constexpr std::size_t kMaxFiberPackedBytes=1024ULL*1024*1024;
// Shared chunk/route payload storage. Known frame size and exact frame extent
// are required; damaged files cannot trigger an unbounded allocation.
inline nlohmann::json readFiberPacked(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);
    if(!input) throw std::runtime_error("cannot read chunk paths: "+path.string());
    if(path.extension()!=".zst") return nlohmann::json::parse(input);
    const std::string compressed((std::istreambuf_iterator<char>(input)),{});
    try {
        return nlohmann::json::from_msgpack(vcr::zarr::decompressZstdFrame(
            {reinterpret_cast<const std::uint8_t*>(compressed.data()),compressed.size()},kMaxFiberPackedBytes));
    } catch(const nlohmann::json::exception&) {
        throw;
    } catch(const std::exception& error) {
        throw std::runtime_error("cannot decompress chunk paths: "+path.string()+": "+error.what());
    }
}
inline void writeFiberPacked(const std::filesystem::path& path,const nlohmann::json& document) {
    const auto bytes=nlohmann::json::to_msgpack(document);
    if(bytes.size()>kMaxFiberPackedBytes)throw std::runtime_error("chunk cache exceeds 1 GiB decoded limit");
    const auto compressed=vcr::zarr::compressBytes({{"id","zstd"},{"level",3}},bytes);
    vcr::zarr::atomicWrite(path,compressed.data(),compressed.size(),vcr::zarr::Durability::Fsync);
}
} // namespace vc::fiber_tracer::detail
