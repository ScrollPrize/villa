#pragma once

#include "vc/core/util/AtomicFile.hpp"
#include <nlohmann/json.hpp>
#include <zstd.h>
#include <filesystem>
#include <fstream>
#include <vector>

namespace vc::fiber_tracer::detail {
// Shared chunk/route payload storage. Known frame size and exact frame extent
// are required; damaged files cannot trigger an unbounded allocation.
inline nlohmann::json readFiberPacked(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);
    if(!input) throw std::runtime_error("cannot read chunk paths: "+path.string());
    if(path.extension()!=".zst") return nlohmann::json::parse(input);
    const std::string compressed((std::istreambuf_iterator<char>(input)),{});
    const auto size=ZSTD_getFrameContentSize(compressed.data(),compressed.size());
    if(size==ZSTD_CONTENTSIZE_ERROR || size==ZSTD_CONTENTSIZE_UNKNOWN || size>1024ULL*1024*1024)
        throw std::runtime_error("invalid or oversized chunk cache frame: "+path.string());
    if(ZSTD_findFrameCompressedSize(compressed.data(),compressed.size())!=compressed.size())
        throw std::runtime_error("invalid chunk cache frame extent: "+path.string());
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    const auto decoded=ZSTD_decompress(bytes.data(),bytes.size(),compressed.data(),compressed.size());
    if(ZSTD_isError(decoded) || decoded!=bytes.size())
        throw std::runtime_error("cannot decompress chunk paths: "+path.string());
    return nlohmann::json::from_msgpack(bytes);
}
inline void writeFiberPacked(const std::filesystem::path& path,const nlohmann::json& document) {
    const auto bytes=nlohmann::json::to_msgpack(document);
    if(bytes.size()>1024ULL*1024*1024)throw std::runtime_error("chunk cache exceeds 1 GiB decoded limit");
    std::vector<std::byte> compressed(ZSTD_compressBound(bytes.size()));
    const auto size=ZSTD_compress(compressed.data(),compressed.size(),bytes.data(),bytes.size(),3);
    if(ZSTD_isError(size))throw std::runtime_error(std::string("chunk compression failed: ")+ZSTD_getErrorName(size));
    compressed.resize(size);
    vc::core::util::atomicWriteBytes(path,compressed);
}
} // namespace vc::fiber_tracer::detail
