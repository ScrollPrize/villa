#pragma once
// Zarr v2/v3 array metadata.

#include "vc_reader/native/reader_types.hpp"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace vcr::zarr {
using Json = nlohmann::json;

struct ArrayMeta {
    Index shape, chunks;
    std::string dtype;               // numpy typestr, e.g. "|u1", "<u2", "<f4"
    ElementType type;                // how stored elements read as numbers
    std::vector<std::uint8_t> fill;  // one element, in stored byte order
    std::string separator = ".";
    Json compressor;                 // v2 compressor object, or null
    Json filters;                    // v2 filters, or null
    std::string order = "C";
    struct V3 {
        std::string prefix;          // chunk key prefix ("c/" or "c.")
        Index shards;                // chunks per shard along each axis; empty when unsharded
        bool indexEnd = true, indexCrc = false;
        Json metadata;               // the zarr.json as stored
    };
    std::optional<V3> v3;
    Json raw;                        // the metadata as a v2 .zarray

    std::size_t itemsize() const { return type.itemsize; }
    // The reader's codec id ("" uncompressed); throws when the chunks cannot be decoded.
    std::string codec() const;
    std::size_t chunkBytes() const;
    // .zarray of the same array stored uncompressed (the decoded-cache layout).
    Json decodedZarray() const;
    // .zarray of the decoded chunks stored with lossless Delta3D (as VC3D writes them).
    Json delta3dZarray() const;
};

// Parses a .zarray or v3 zarr.json. Throws for unsupported layouts and dtypes;
// whether the chunks can be decoded is checked by codec().
ArrayMeta parseArrayMeta(const Json& metadata);

}  // namespace vcr::zarr
