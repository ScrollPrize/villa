#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "utils/zarr.hpp"

#include <string>
#include <vector>

TEST_CASE("parse_zarray: minimal v2 metadata")
{
    const std::string j = R"({
        "zarr_format": 2,
        "shape": [32, 64, 128],
        "chunks": [16, 16, 16],
        "dtype": "<u1",
        "compressor": null,
        "fill_value": 0,
        "dimension_separator": "."
    })";
    auto meta = utils::detail::parse_zarray(j);
    CHECK(meta.version == utils::ZarrVersion::v2);
    CHECK(meta.shape == std::vector<std::size_t>{32, 64, 128});
    CHECK(meta.chunks == std::vector<std::size_t>{16, 16, 16});
    CHECK(meta.dtype == utils::ZarrDtype::uint8);
    CHECK(meta.byte_order == '<');
    CHECK(meta.compressor_id.empty());
    REQUIRE(meta.fill_value.has_value());
    CHECK(*meta.fill_value == doctest::Approx(0.0));
    CHECK(meta.dimension_separator == ".");
}

TEST_CASE("parse_zarray: blosc compressor with level")
{
    const std::string j = R"({
        "zarr_format": 2,
        "shape": [16, 16, 16],
        "chunks": [16, 16, 16],
        "dtype": "<u2",
        "compressor": {"id": "blosc", "clevel": 5, "cname": "zstd"},
        "fill_value": null
    })";
    auto meta = utils::detail::parse_zarray(j);
    CHECK(meta.dtype == utils::ZarrDtype::uint16);
    CHECK(meta.compressor_id == "blosc");
    CHECK(meta.compression_level == 5);
    CHECK_FALSE(meta.fill_value.has_value());
}

TEST_CASE("parse_zarray: rejects non-object root")
{
    CHECK_THROWS([&] { (void)utils::detail::parse_zarray("[1,2,3]"); }());
}

TEST_CASE("parse_zarray: rejects truly unknown dtype")
{
    const std::string j = R"({"shape":[1],"chunks":[1],"dtype":"<S10"})";
    CHECK_THROWS([&] { (void)utils::detail::parse_zarray(j); }());
}

TEST_CASE("parse_zarr_json: minimal v3 metadata")
{
    const std::string j = R"({
        "zarr_format": 3,
        "node_type": "array",
        "shape": [16, 16, 16],
        "data_type": "uint8",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [16, 16, 16]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
    })";
    auto meta = utils::detail::parse_zarr_json(j);
    CHECK(meta.version == utils::ZarrVersion::v3);
    CHECK(meta.node_type == "array");
    CHECK(meta.shape == std::vector<std::size_t>{16, 16, 16});
    CHECK(meta.dtype == utils::ZarrDtype::uint8);
    CHECK(meta.chunks == std::vector<std::size_t>{16, 16, 16});
    CHECK(meta.dimension_separator == "/");
    CHECK_FALSE(meta.shard_config.has_value());
    REQUIRE(meta.codecs.size() == 1);
    CHECK(meta.codecs[0].name == "bytes");
}

TEST_CASE("parse_zarr_json: sharded v3 metadata")
{
    const std::string j = R"({
        "zarr_format": 3,
        "node_type": "array",
        "shape": [128, 128, 128],
        "data_type": "uint16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [64, 64, 64]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 7,
        "codecs": [{
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": [16, 16, 16],
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                                 {"name": "crc32c"}]
            }
        }]
    })";
    auto meta = utils::detail::parse_zarr_json(j);
    CHECK(meta.dtype == utils::ZarrDtype::uint16);
    CHECK(meta.chunks == std::vector<std::size_t>{64, 64, 64});
    REQUIRE(meta.shard_config.has_value());
    CHECK(meta.shard_config->sub_chunks == std::vector<std::size_t>{16, 16, 16});
    CHECK(meta.shard_config->sub_codecs.size() == 1);
    CHECK(meta.shard_config->index_codecs.size() == 2);
    REQUIRE(meta.fill_value.has_value());
    CHECK(*meta.fill_value == doctest::Approx(7.0));
}

TEST_CASE("parse_zarr_json: rejects non-object root")
{
    CHECK_THROWS([&] { (void)utils::detail::parse_zarr_json("[]"); }());
}

TEST_CASE("parse_zarr_json: rejects truly unknown v3 data_type")
{
    const std::string j = R"({
        "zarr_format": 3,
        "node_type": "array",
        "shape": [1],
        "data_type": "string",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1]}},
        "chunk_key_encoding": {"name": "default"},
        "codecs": []
    })";
    CHECK_THROWS([&] { (void)utils::detail::parse_zarr_json(j); }());
}

TEST_CASE("parse_codec_config: name + nested configuration")
{
    auto json = utils::Json::parse(R"({"name":"blosc","configuration":{"cname":"zstd","clevel":5}})");
    auto cc = utils::detail::parse_codec_config(json);
    CHECK(cc.name == "blosc");
    REQUIRE(cc.configuration);
    CHECK(cc.configuration->is_object());
    CHECK((*cc.configuration)["cname"].get_string() == "zstd");
    CHECK((*cc.configuration)["clevel"].get_int() == 5);
}

TEST_CASE("serialize_zarr_json + parse_zarr_json round-trip")
{
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.node_type = "array";
    meta.shape = {32, 32, 32};
    meta.chunks = {16, 16, 16};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.dimension_separator = "/";
    meta.fill_value = 0.0;

    auto serialized = utils::detail::serialize_zarr_json(meta);
    auto parsed = utils::detail::parse_zarr_json(serialized);

    CHECK(parsed.shape == meta.shape);
    CHECK(parsed.chunks == meta.chunks);
    CHECK(parsed.dtype == meta.dtype);
    REQUIRE(parsed.fill_value.has_value());
    CHECK(*parsed.fill_value == doctest::Approx(*meta.fill_value));
}

TEST_CASE("dtype_size: returns correct widths")
{
    CHECK(utils::dtype_size(utils::ZarrDtype::uint8) == 1);
    CHECK(utils::dtype_size(utils::ZarrDtype::uint16) == 2);
    CHECK(utils::dtype_size(utils::ZarrDtype::int32) == 4);
}

TEST_CASE("dtype_string returns parseable strings")
{
    CHECK(utils::parse_dtype(std::string(utils::dtype_string(utils::ZarrDtype::uint8))) ==
          utils::ZarrDtype::uint8);
    CHECK(utils::parse_dtype(std::string(utils::dtype_string(utils::ZarrDtype::uint16))) ==
          utils::ZarrDtype::uint16);
}

TEST_CASE("parse_dtype: handles sized prefixes")
{
    CHECK(utils::parse_dtype("<u1") == utils::ZarrDtype::uint8);
    CHECK(utils::parse_dtype(">u2") == utils::ZarrDtype::uint16);
    CHECK(utils::parse_dtype("|u1") == utils::ZarrDtype::uint8);
}

TEST_CASE("parse_dtype: returns nullopt for truly unknown")
{
    CHECK_FALSE(utils::parse_dtype("<S10").has_value());
    CHECK_FALSE(utils::parse_dtype("garbage").has_value());
}

TEST_CASE("parse_dtype_v3: handles canonical v3 names")
{
    CHECK(utils::parse_dtype_v3("uint8") == utils::ZarrDtype::uint8);
    CHECK(utils::parse_dtype_v3("uint16") == utils::ZarrDtype::uint16);
    CHECK_FALSE(utils::parse_dtype_v3("string").has_value());
    CHECK_FALSE(utils::parse_dtype_v3("not_a_type").has_value());
}

TEST_CASE("is_canonical_c3d: requires v3 + sharded with c3d sub-codec")
{
    utils::ZarrMetadata bare;
    CHECK_FALSE(utils::is_canonical_c3d(bare));

    utils::ZarrMetadata v2;
    v2.version = utils::ZarrVersion::v2;
    v2.compressor_id = "c3d";
    CHECK_FALSE(utils::is_canonical_c3d(v2));
}
