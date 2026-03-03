#include <utils/test.hpp>
#include <utils/zarr.hpp>
#include <filesystem>
#include <vector>
#include <cstring>

namespace fs = std::filesystem;

static fs::path make_temp_dir(const char* suffix) {
    auto p = fs::temp_directory_path() / ("utils2_test_zarr_" + std::string(suffix));
    fs::remove_all(p);
    return p;
}

TEST_CASE("ZarrDtype size") {
    REQUIRE_EQ(utils::dtype_size(utils::ZarrDtype::uint8), std::size_t(1));
    REQUIRE_EQ(utils::dtype_size(utils::ZarrDtype::uint16), std::size_t(2));
    REQUIRE_EQ(utils::dtype_size(utils::ZarrDtype::uint32), std::size_t(4));
    REQUIRE_EQ(utils::dtype_size(utils::ZarrDtype::uint64), std::size_t(8));
    REQUIRE_EQ(utils::dtype_size(utils::ZarrDtype::float32), std::size_t(4));
    REQUIRE_EQ(utils::dtype_size(utils::ZarrDtype::float64), std::size_t(8));
}

TEST_CASE("ZarrDtype parse_dtype") {
    auto d1 = utils::parse_dtype("<u2");
    REQUIRE(d1.has_value());
    REQUIRE_EQ(*d1, utils::ZarrDtype::uint16);

    auto d2 = utils::parse_dtype(">f4");
    REQUIRE(d2.has_value());
    REQUIRE_EQ(*d2, utils::ZarrDtype::float32);

    auto d3 = utils::parse_dtype("|u1");
    REQUIRE(d3.has_value());
    REQUIRE_EQ(*d3, utils::ZarrDtype::uint8);

    auto d4 = utils::parse_dtype("i4");
    REQUIRE(d4.has_value());
    REQUIRE_EQ(*d4, utils::ZarrDtype::int32);

    auto bad = utils::parse_dtype("xyz");
    REQUIRE(!bad.has_value());
}

TEST_CASE("ZarrDtype dtype_string") {
    REQUIRE_EQ(utils::dtype_string(utils::ZarrDtype::uint16), std::string_view("u2"));
    REQUIRE_EQ(utils::dtype_string(utils::ZarrDtype::float64), std::string_view("f8"));
}

TEST_CASE("ZarrMetadata helpers") {
    utils::ZarrMetadata meta;
    meta.shape = {100, 200, 300};
    meta.chunks = {10, 20, 30};
    meta.dtype = utils::ZarrDtype::uint16;

    REQUIRE_EQ(meta.ndim(), std::size_t(3));
    REQUIRE_EQ(meta.num_chunks_along(0), std::size_t(10));
    REQUIRE_EQ(meta.num_chunks_along(1), std::size_t(10));
    REQUIRE_EQ(meta.num_chunks_along(2), std::size_t(10));
    REQUIRE_EQ(meta.chunk_byte_size(), std::size_t(10 * 20 * 30 * 2));
}

TEST_CASE("ZarrArray create and read back") {
    auto dir = make_temp_dir("create");

    utils::ZarrMetadata meta;
    meta.shape = {64, 64};
    meta.chunks = {32, 32};
    meta.dtype = utils::ZarrDtype::uint16;
    meta.byte_order = '<';

    auto arr = utils::ZarrArray::create(dir / "test.zarr", meta);

    REQUIRE_EQ(arr.metadata().shape.size(), std::size_t(2));
    REQUIRE_EQ(arr.metadata().shape[0], std::size_t(64));
    REQUIRE_EQ(arr.metadata().chunks[0], std::size_t(32));
    REQUIRE_EQ(arr.metadata().dtype, utils::ZarrDtype::uint16);

    // Re-open and verify metadata persisted.
    auto arr2 = utils::ZarrArray::open(dir / "test.zarr");
    REQUIRE_EQ(arr2.metadata().shape[0], std::size_t(64));
    REQUIRE_EQ(arr2.metadata().dtype, utils::ZarrDtype::uint16);

    fs::remove_all(dir);
}

TEST_CASE("ZarrArray chunk write/read round trip") {
    auto dir = make_temp_dir("chunk_rw");

    utils::ZarrMetadata meta;
    meta.shape = {16, 16};
    meta.chunks = {8, 8};
    meta.dtype = utils::ZarrDtype::float32;

    auto arr = utils::ZarrArray::create(dir / "arr.zarr", meta);

    // Write a chunk of float32 data (8*8*4 = 256 bytes).
    std::vector<float> chunk_data(64, 3.14f);
    std::vector<std::byte> bytes(chunk_data.size() * sizeof(float));
    std::memcpy(bytes.data(), chunk_data.data(), bytes.size());

    std::array<std::size_t, 2> idx = {0, 0};
    arr.write_chunk(idx, bytes);

    REQUIRE(arr.chunk_exists(idx));

    auto read_back = arr.read_chunk(idx);
    REQUIRE(read_back.has_value());
    REQUIRE_EQ(read_back->size(), bytes.size());

    std::vector<float> got(64);
    std::memcpy(got.data(), read_back->data(), got.size() * sizeof(float));
    REQUIRE_NEAR(got[0], 3.14f, 1e-5);
    REQUIRE_NEAR(got[63], 3.14f, 1e-5);

    fs::remove_all(dir);
}

TEST_CASE("ZarrArray chunk not found returns nullopt") {
    auto dir = make_temp_dir("chunk_missing");

    utils::ZarrMetadata meta;
    meta.shape = {16};
    meta.chunks = {8};
    meta.dtype = utils::ZarrDtype::uint8;

    auto arr = utils::ZarrArray::create(dir / "arr.zarr", meta);

    std::array<std::size_t, 1> idx = {0};
    REQUIRE(!arr.chunk_exists(idx));
    auto result = arr.read_chunk(idx);
    REQUIRE(!result.has_value());

    fs::remove_all(dir);
}

TEST_CASE("Zarr pyramid level counting") {
    auto dir = make_temp_dir("pyramid");

    // Create 3 pyramid levels.
    for (int i = 0; i < 3; ++i) {
        utils::ZarrMetadata meta;
        meta.shape = {64u >> i, 64u >> i};
        meta.chunks = {32, 32};
        meta.dtype = utils::ZarrDtype::uint16;
        utils::ZarrArray::create(dir / std::to_string(i), meta);
    }

    REQUIRE_EQ(utils::count_pyramid_levels(dir), std::size_t(3));

    auto levels = utils::open_pyramid(dir);
    REQUIRE_EQ(levels.size(), std::size_t(3));
    REQUIRE_EQ(levels[0].metadata().shape[0], std::size_t(64));
    REQUIRE_EQ(levels[2].metadata().shape[0], std::size_t(16));

    fs::remove_all(dir);
}

TEST_CASE("Zarr .zarray JSON round trip") {
    utils::ZarrMetadata meta;
    meta.shape = {128, 256};
    meta.chunks = {64, 64};
    meta.dtype = utils::ZarrDtype::float64;
    meta.byte_order = '<';
    meta.compressor_id = "blosc";
    meta.compression_level = 3;
    meta.fill_value = 0.0;
    meta.dimension_separator = ".";

    auto json = utils::detail::serialize_zarray(meta);
    auto parsed = utils::detail::parse_zarray(json);

    REQUIRE_EQ(parsed.shape.size(), std::size_t(2));
    REQUIRE_EQ(parsed.shape[0], std::size_t(128));
    REQUIRE_EQ(parsed.shape[1], std::size_t(256));
    REQUIRE_EQ(parsed.chunks[0], std::size_t(64));
    REQUIRE_EQ(parsed.dtype, utils::ZarrDtype::float64);
    REQUIRE_EQ(parsed.byte_order, '<');
    REQUIRE_EQ(parsed.compressor_id, std::string("blosc"));
    REQUIRE_EQ(parsed.compression_level, 3);
    REQUIRE(parsed.fill_value.has_value());
    REQUIRE_NEAR(*parsed.fill_value, 0.0, 1e-12);
}

TEST_CASE("ZarrDtype dtype_string_v3") {
    REQUIRE_EQ(utils::dtype_string_v3(utils::ZarrDtype::uint16), std::string_view("uint16"));
    REQUIRE_EQ(utils::dtype_string_v3(utils::ZarrDtype::float32), std::string_view("float32"));
    REQUIRE_EQ(utils::dtype_string_v3(utils::ZarrDtype::int64), std::string_view("int64"));
    REQUIRE_EQ(utils::dtype_string_v3(utils::ZarrDtype::bool_), std::string_view("bool"));
    REQUIRE_EQ(utils::dtype_string_v3(utils::ZarrDtype::complex128), std::string_view("complex128"));
}

TEST_CASE("ZarrDtype parse_dtype_v3") {
    auto d1 = utils::parse_dtype_v3("uint16");
    REQUIRE(d1.has_value());
    REQUIRE_EQ(*d1, utils::ZarrDtype::uint16);

    auto d2 = utils::parse_dtype_v3("float64");
    REQUIRE(d2.has_value());
    REQUIRE_EQ(*d2, utils::ZarrDtype::float64);

    auto d3 = utils::parse_dtype_v3("bool");
    REQUIRE(d3.has_value());
    REQUIRE_EQ(*d3, utils::ZarrDtype::bool_);

    auto bad = utils::parse_dtype_v3("invalid");
    REQUIRE(!bad.has_value());
}

TEST_CASE("ZarrDtype all sizes") {
    REQUIRE_EQ(utils::dtype_size(utils::ZarrDtype::bool_), std::size_t(1));
    REQUIRE_EQ(utils::dtype_size(utils::ZarrDtype::int8), std::size_t(1));
    REQUIRE_EQ(utils::dtype_size(utils::ZarrDtype::int16), std::size_t(2));
    REQUIRE_EQ(utils::dtype_size(utils::ZarrDtype::int32), std::size_t(4));
    REQUIRE_EQ(utils::dtype_size(utils::ZarrDtype::int64), std::size_t(8));
    REQUIRE_EQ(utils::dtype_size(utils::ZarrDtype::float16), std::size_t(2));
    REQUIRE_EQ(utils::dtype_size(utils::ZarrDtype::complex64), std::size_t(8));
    REQUIRE_EQ(utils::dtype_size(utils::ZarrDtype::complex128), std::size_t(16));
}

TEST_CASE("ZarrDtype parse_dtype all v2 types") {
    REQUIRE_EQ(*utils::parse_dtype("b1"), utils::ZarrDtype::bool_);
    REQUIRE_EQ(*utils::parse_dtype("i1"), utils::ZarrDtype::int8);
    REQUIRE_EQ(*utils::parse_dtype("i2"), utils::ZarrDtype::int16);
    REQUIRE_EQ(*utils::parse_dtype("i8"), utils::ZarrDtype::int64);
    REQUIRE_EQ(*utils::parse_dtype("f2"), utils::ZarrDtype::float16);
    REQUIRE_EQ(*utils::parse_dtype("c8"), utils::ZarrDtype::complex64);
    REQUIRE_EQ(*utils::parse_dtype("c16"), utils::ZarrDtype::complex128);
}

TEST_CASE("ZarrDtype dtype_string_v2 all types") {
    REQUIRE_EQ(utils::dtype_string_v2(utils::ZarrDtype::bool_), std::string_view("b1"));
    REQUIRE_EQ(utils::dtype_string_v2(utils::ZarrDtype::uint8), std::string_view("u1"));
    REQUIRE_EQ(utils::dtype_string_v2(utils::ZarrDtype::int8), std::string_view("i1"));
    REQUIRE_EQ(utils::dtype_string_v2(utils::ZarrDtype::float16), std::string_view("f2"));
    REQUIRE_EQ(utils::dtype_string_v2(utils::ZarrDtype::complex64), std::string_view("c8"));
}

TEST_CASE("ZarrMetadata num_chunks_along non-divisible") {
    utils::ZarrMetadata meta;
    meta.shape = {100, 200};
    meta.chunks = {30, 60};
    meta.dtype = utils::ZarrDtype::uint8;

    // ceil(100/30) = 4, ceil(200/60) = 4
    REQUIRE_EQ(meta.num_chunks_along(0), std::size_t(4));
    REQUIRE_EQ(meta.num_chunks_along(1), std::size_t(4));
}

TEST_CASE("ZarrMetadata chunk_byte_size 1D") {
    utils::ZarrMetadata meta;
    meta.shape = {1000};
    meta.chunks = {256};
    meta.dtype = utils::ZarrDtype::float32;

    REQUIRE_EQ(meta.chunk_byte_size(), std::size_t(256 * 4));
}

TEST_CASE("FileSystemStore operations") {
    auto dir = make_temp_dir("fsstore");
    utils::FileSystemStore store(dir);

    REQUIRE_EQ(store.root(), dir);

    // Key does not exist yet
    REQUIRE(!store.exists("test_key"));

    // Set a value
    std::string data = "hello zarr store";
    std::vector<std::byte> bytes(data.size());
    std::memcpy(bytes.data(), data.data(), data.size());
    store.set("test_key", bytes);

    REQUIRE(store.exists("test_key"));

    // Get it back
    auto got = store.get("test_key");
    REQUIRE_EQ(got.size(), bytes.size());
    std::string got_str(reinterpret_cast<const char*>(got.data()), got.size());
    REQUIRE_EQ(got_str, data);

    // get_if_exists
    auto opt = store.get_if_exists("test_key");
    REQUIRE(opt.has_value());
    REQUIRE_EQ(opt->size(), bytes.size());

    auto missing = store.get_if_exists("nonexistent");
    REQUIRE(!missing.has_value());

    fs::remove_all(dir);
}

TEST_CASE("FileSystemStore get_partial") {
    auto dir = make_temp_dir("fsstore_partial");
    utils::FileSystemStore store(dir);

    std::string data = "abcdefghij";
    std::vector<std::byte> bytes(data.size());
    std::memcpy(bytes.data(), data.data(), data.size());
    store.set("partial_key", bytes);

    // Read middle portion
    auto partial = store.get_partial("partial_key", 3, 4);
    REQUIRE(partial.has_value());
    REQUIRE_EQ(partial->size(), std::size_t(4));
    std::string got(reinterpret_cast<const char*>(partial->data()), partial->size());
    REQUIRE_EQ(got, std::string("defg"));

    // Read past end -> clamped
    auto past_end = store.get_partial("partial_key", 8, 100);
    REQUIRE(past_end.has_value());
    REQUIRE_EQ(past_end->size(), std::size_t(2)); // only "ij"

    // Nonexistent key
    auto none = store.get_partial("nonexistent", 0, 10);
    REQUIRE(!none.has_value());

    fs::remove_all(dir);
}

TEST_CASE("ZarrArray multiple chunks") {
    auto dir = make_temp_dir("multi_chunks");

    utils::ZarrMetadata meta;
    meta.shape = {16};
    meta.chunks = {4};
    meta.dtype = utils::ZarrDtype::uint8;

    auto arr = utils::ZarrArray::create(dir / "arr.zarr", meta);

    // Write all 4 chunks (16/4 = 4)
    for (std::size_t i = 0; i < 4; ++i) {
        std::vector<std::byte> chunk_data(4, static_cast<std::byte>(i * 10 + 1));
        std::array<std::size_t, 1> idx = {i};
        arr.write_chunk(idx, chunk_data);
    }

    // Read back and verify each chunk
    for (std::size_t i = 0; i < 4; ++i) {
        std::array<std::size_t, 1> idx = {i};
        REQUIRE(arr.chunk_exists(idx));
        auto data = arr.read_chunk(idx);
        REQUIRE(data.has_value());
        REQUIRE_EQ(data->size(), std::size_t(4));
        REQUIRE_EQ(static_cast<std::uint8_t>((*data)[0]), static_cast<std::uint8_t>(i * 10 + 1));
    }

    // Non-existent chunk index
    std::array<std::size_t, 1> bad_idx = {10};
    REQUIRE(!arr.chunk_exists(bad_idx));

    fs::remove_all(dir);
}

TEST_CASE("OME-NGFF MultiscaleMetadata parse and serialize round trip") {
    // Build a MultiscaleMetadata manually
    utils::MultiscaleMetadata meta;
    meta.version = "0.4";
    meta.name = "test_volume";
    meta.axes = {
        {.name = "z", .type = utils::AxisType::space, .unit = "micrometer"},
        {.name = "y", .type = utils::AxisType::space, .unit = "micrometer"},
        {.name = "x", .type = utils::AxisType::space, .unit = "micrometer"},
    };

    utils::MultiscaleDataset ds0;
    ds0.path = "0";
    ds0.transforms.push_back(utils::ScaleTransform{.scale = {1.0, 0.5, 0.5}});
    meta.datasets.push_back(ds0);

    utils::MultiscaleDataset ds1;
    ds1.path = "1";
    ds1.transforms.push_back(utils::ScaleTransform{.scale = {1.0, 1.0, 1.0}});
    meta.datasets.push_back(ds1);

    REQUIRE_EQ(meta.num_levels(), std::size_t(2));
    REQUIRE_EQ(meta.ndim(), std::size_t(3));

    // axis_index
    auto xi = meta.axis_index("x");
    REQUIRE(xi.has_value());
    REQUIRE_EQ(*xi, std::size_t(2));

    auto yi = meta.axis_index("y");
    REQUIRE(yi.has_value());
    REQUIRE_EQ(*yi, std::size_t(1));

    auto zi = meta.axis_index("z");
    REQUIRE(zi.has_value());
    REQUIRE_EQ(*zi, std::size_t(0));

    auto missing = meta.axis_index("t");
    REQUIRE(!missing.has_value());

    // voxel_size for level 0
    auto vs0 = meta.voxel_size(0);
    REQUIRE_NEAR(vs0[0], 0.5, 1e-12);  // x
    REQUIRE_NEAR(vs0[1], 0.5, 1e-12);  // y
    REQUIRE_NEAR(vs0[2], 1.0, 1e-12);  // z

    // Serialize and re-parse
    auto jv = utils::serialize_ome_metadata(meta);
    auto parsed = utils::parse_ome_metadata(jv);

    REQUIRE_EQ(parsed.name, std::string("test_volume"));
    REQUIRE_EQ(parsed.axes.size(), std::size_t(3));
    REQUIRE_EQ(parsed.axes[0].name, std::string("z"));
    REQUIRE_EQ(parsed.axes[2].name, std::string("x"));
    REQUIRE_EQ(parsed.datasets.size(), std::size_t(2));
    REQUIRE_EQ(parsed.datasets[0].path, std::string("0"));

    auto vs_parsed = parsed.voxel_size(0);
    REQUIRE_NEAR(vs_parsed[0], 0.5, 1e-12);
}

TEST_CASE("Zarr .zarray round trip with dimension_separator /") {
    utils::ZarrMetadata meta;
    meta.shape = {32, 32};
    meta.chunks = {16, 16};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.dimension_separator = "/";

    auto json = utils::detail::serialize_zarray(meta);
    auto parsed = utils::detail::parse_zarray(json);

    REQUIRE_EQ(parsed.dimension_separator, std::string("/"));
}

TEST_CASE("Zarr .zarray round trip with no compressor") {
    utils::ZarrMetadata meta;
    meta.shape = {64};
    meta.chunks = {64};
    meta.dtype = utils::ZarrDtype::int16;
    meta.compressor_id = "";

    auto json = utils::detail::serialize_zarray(meta);
    auto parsed = utils::detail::parse_zarray(json);

    REQUIRE_EQ(parsed.dtype, utils::ZarrDtype::int16);
}

// ===========================================================================
// NEW TESTS -- targeting uncovered lines
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. Zarr v3 metadata JSON round-trip (zarr.json parse/serialize)
// ---------------------------------------------------------------------------

TEST_CASE("Zarr v3 zarr.json round trip basic") {
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {256, 256, 128};
    meta.chunks = {64, 64, 64};
    meta.dtype = utils::ZarrDtype::float32;
    meta.fill_value = 0.0;
    meta.node_type = "array";
    meta.chunk_key_encoding = "default";

    auto json = utils::detail::serialize_zarr_json(meta);
    auto parsed = utils::detail::parse_zarr_json(json);

    REQUIRE_EQ(parsed.version, utils::ZarrVersion::v3);
    REQUIRE_EQ(parsed.node_type, std::string("array"));
    REQUIRE_EQ(parsed.shape.size(), std::size_t(3));
    REQUIRE_EQ(parsed.shape[0], std::size_t(256));
    REQUIRE_EQ(parsed.shape[1], std::size_t(256));
    REQUIRE_EQ(parsed.shape[2], std::size_t(128));
    REQUIRE_EQ(parsed.chunks.size(), std::size_t(3));
    REQUIRE_EQ(parsed.chunks[0], std::size_t(64));
    REQUIRE_EQ(parsed.dtype, utils::ZarrDtype::float32);
    REQUIRE(parsed.fill_value.has_value());
    REQUIRE_NEAR(*parsed.fill_value, 0.0, 1e-12);
    // default encoding -> separator should be "/"
    REQUIRE_EQ(parsed.chunk_key_encoding, std::string("default"));
    REQUIRE_EQ(parsed.dimension_separator, std::string("/"));
}

TEST_CASE("Zarr v3 zarr.json round trip v2 encoding") {
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {100, 100};
    meta.chunks = {50, 50};
    meta.dtype = utils::ZarrDtype::uint16;
    meta.chunk_key_encoding = "v2";
    meta.dimension_separator = ".";

    auto json = utils::detail::serialize_zarr_json(meta);
    auto parsed = utils::detail::parse_zarr_json(json);

    REQUIRE_EQ(parsed.chunk_key_encoding, std::string("v2"));
    REQUIRE_EQ(parsed.dimension_separator, std::string("."));
}

TEST_CASE("Zarr v3 zarr.json null fill_value") {
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {32};
    meta.chunks = {32};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.fill_value = std::nullopt;

    auto json = utils::detail::serialize_zarr_json(meta);
    auto parsed = utils::detail::parse_zarr_json(json);

    REQUIRE(!parsed.fill_value.has_value());
}

TEST_CASE("Zarr v3 zarr.json all dtypes round trip") {
    auto test_dtype = [](utils::ZarrDtype dt) {
        utils::ZarrMetadata meta;
        meta.version = utils::ZarrVersion::v3;
        meta.shape = {16};
        meta.chunks = {16};
        meta.dtype = dt;

        auto json = utils::detail::serialize_zarr_json(meta);
        auto parsed = utils::detail::parse_zarr_json(json);
        REQUIRE_EQ(parsed.dtype, dt);
    };

    test_dtype(utils::ZarrDtype::bool_);
    test_dtype(utils::ZarrDtype::uint8);
    test_dtype(utils::ZarrDtype::uint16);
    test_dtype(utils::ZarrDtype::uint32);
    test_dtype(utils::ZarrDtype::uint64);
    test_dtype(utils::ZarrDtype::int8);
    test_dtype(utils::ZarrDtype::int16);
    test_dtype(utils::ZarrDtype::int32);
    test_dtype(utils::ZarrDtype::int64);
    test_dtype(utils::ZarrDtype::float16);
    test_dtype(utils::ZarrDtype::float32);
    test_dtype(utils::ZarrDtype::float64);
    test_dtype(utils::ZarrDtype::complex64);
    test_dtype(utils::ZarrDtype::complex128);
}

// ---------------------------------------------------------------------------
// 2. Codec pipeline configuration parsing
// ---------------------------------------------------------------------------

TEST_CASE("Zarr v3 codec pipeline parsing") {
    std::string json_str = R"({
        "zarr_format": 3,
        "node_type": "array",
        "shape": [64, 64],
        "data_type": "float32",
        "chunk_grid": {
            "name": "regular",
            "configuration": { "chunk_shape": [32, 32] }
        },
        "chunk_key_encoding": { "name": "default" },
        "fill_value": 0,
        "codecs": [
            {
                "name": "bytes",
                "configuration": { "endian": "little" }
            },
            {
                "name": "blosc",
                "configuration": { "cname": "lz4", "clevel": 5 }
            }
        ]
    })";

    auto meta = utils::detail::parse_zarr_json(json_str);
    REQUIRE_EQ(meta.codecs.size(), std::size_t(2));
    REQUIRE_EQ(meta.codecs[0].name, std::string("bytes"));
    REQUIRE(meta.codecs[0].configuration.is_object());
    REQUIRE_EQ(meta.codecs[1].name, std::string("blosc"));
    REQUIRE(meta.codecs[1].configuration.is_object());

    // No shard config should be detected
    REQUIRE(!meta.shard_config.has_value());
}

TEST_CASE("Zarr v3 codecs serialize round trip") {
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {32};
    meta.chunks = {32};
    meta.dtype = utils::ZarrDtype::uint16;

    // Add explicit codecs
    utils::ZarrCodecConfig bytes_codec;
    bytes_codec.name = "bytes";
    utils::JsonObject bytes_cfg;
    bytes_cfg["endian"] = utils::JsonValue("little");
    bytes_codec.configuration = utils::JsonValue(std::move(bytes_cfg));
    meta.codecs.push_back(bytes_codec);

    utils::ZarrCodecConfig zstd_codec;
    zstd_codec.name = "zstd";
    utils::JsonObject zstd_cfg;
    zstd_cfg["level"] = utils::JsonValue(3);
    zstd_codec.configuration = utils::JsonValue(std::move(zstd_cfg));
    meta.codecs.push_back(zstd_codec);

    auto json = utils::detail::serialize_zarr_json(meta);
    auto parsed = utils::detail::parse_zarr_json(json);

    REQUIRE_EQ(parsed.codecs.size(), std::size_t(2));
    REQUIRE_EQ(parsed.codecs[0].name, std::string("bytes"));
    REQUIRE_EQ(parsed.codecs[1].name, std::string("zstd"));
}

TEST_CASE("Zarr v3 default bytes codec when no codecs specified") {
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {16};
    meta.chunks = {16};
    meta.dtype = utils::ZarrDtype::uint8;
    // No codecs set - should get default bytes codec in serialized form

    auto json = utils::detail::serialize_zarr_json(meta);
    auto parsed = utils::detail::parse_zarr_json(json);

    REQUIRE_GE(parsed.codecs.size(), std::size_t(1));
    REQUIRE_EQ(parsed.codecs[0].name, std::string("bytes"));
}

// ---------------------------------------------------------------------------
// 3. Shard index encoding/decoding
// ---------------------------------------------------------------------------

TEST_CASE("ShardIndex serialize and deserialize round trip") {
    utils::detail::ShardIndex index;
    index.entries.resize(4);
    index.entries[0] = {0, 100};
    index.entries[1] = {100, 200};
    index.entries[2] = {~std::uint64_t(0), ~std::uint64_t(0)}; // missing
    index.entries[3] = {300, 50};

    auto bytes = index.serialize();
    REQUIRE_EQ(bytes.size(), std::size_t(4 * 16)); // 4 entries * 16 bytes each

    auto deserialized = utils::detail::ShardIndex::deserialize(bytes, 4);
    REQUIRE_EQ(deserialized.entries.size(), std::size_t(4));
    REQUIRE_EQ(deserialized.entries[0].offset, std::uint64_t(0));
    REQUIRE_EQ(deserialized.entries[0].nbytes, std::uint64_t(100));
    REQUIRE_EQ(deserialized.entries[1].offset, std::uint64_t(100));
    REQUIRE_EQ(deserialized.entries[1].nbytes, std::uint64_t(200));
    REQUIRE(deserialized.entries[2].is_missing());
    REQUIRE_EQ(deserialized.entries[3].offset, std::uint64_t(300));
    REQUIRE_EQ(deserialized.entries[3].nbytes, std::uint64_t(50));
}

TEST_CASE("ShardIndexEntry is_missing") {
    utils::detail::ShardIndexEntry present = {0, 100};
    REQUIRE(!present.is_missing());

    utils::detail::ShardIndexEntry missing = {~std::uint64_t(0), ~std::uint64_t(0)};
    REQUIRE(missing.is_missing());

    // Default-constructed should be missing
    utils::detail::ShardIndexEntry default_entry;
    REQUIRE(default_entry.is_missing());
}

TEST_CASE("ShardIndex too small throws") {
    std::vector<std::byte> too_small(10); // need 4*16=64 bytes for 4 entries
    bool threw = false;
    try {
        utils::detail::ShardIndex::deserialize(too_small, 4);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE("LE64 read/write round trip") {
    std::byte buf[8]{};
    utils::detail::write_le64(buf, 0x0102030405060708ULL);
    auto val = utils::detail::read_le64(buf);
    REQUIRE_EQ(val, std::uint64_t(0x0102030405060708ULL));

    // Test with zero
    utils::detail::write_le64(buf, 0);
    REQUIRE_EQ(utils::detail::read_le64(buf), std::uint64_t(0));

    // Test with max
    utils::detail::write_le64(buf, ~std::uint64_t(0));
    REQUIRE_EQ(utils::detail::read_le64(buf), ~std::uint64_t(0));
}

// ---------------------------------------------------------------------------
// 4. ZarrMetadata shard-related helpers
// ---------------------------------------------------------------------------

TEST_CASE("ZarrMetadata shard helpers") {
    utils::ZarrMetadata meta;
    meta.shape = {256, 256};
    meta.chunks = {64, 64};
    meta.dtype = utils::ZarrDtype::uint16;
    meta.version = utils::ZarrVersion::v3;

    utils::ShardConfig sc;
    sc.sub_chunks = {16, 16};
    sc.index_at_end = true;
    meta.shard_config = sc;

    // sub_chunk_byte_size: 16 * 16 * 2 = 512
    REQUIRE_EQ(meta.sub_chunk_byte_size(), std::size_t(16 * 16 * 2));

    // sub_chunks_per_shard: 64/16 = 4 per dim
    REQUIRE_EQ(meta.sub_chunks_per_shard(0), std::size_t(4));
    REQUIRE_EQ(meta.sub_chunks_per_shard(1), std::size_t(4));

    // total_sub_chunks_per_shard: 4 * 4 = 16
    REQUIRE_EQ(meta.total_sub_chunks_per_shard(), std::size_t(16));

    // Without shard config, sub_chunk_byte_size == chunk_byte_size
    utils::ZarrMetadata meta2;
    meta2.shape = {32};
    meta2.chunks = {32};
    meta2.dtype = utils::ZarrDtype::float32;
    REQUIRE_EQ(meta2.sub_chunk_byte_size(), meta2.chunk_byte_size());
    REQUIRE_EQ(meta2.total_sub_chunks_per_shard(), std::size_t(1));
    REQUIRE_EQ(meta2.sub_chunks_per_shard(0), std::size_t(1));
}

TEST_CASE("ZarrMetadata v3_separator") {
    utils::ZarrMetadata meta;
    meta.chunk_key_encoding = "default";
    REQUIRE_EQ(meta.v3_separator(), std::string("/"));

    meta.chunk_key_encoding = "v2";
    REQUIRE_EQ(meta.v3_separator(), std::string("."));
}

TEST_CASE("ZarrMetadata num_chunks_along edge cases") {
    utils::ZarrMetadata meta;
    meta.shape = {10};
    meta.chunks = {10};
    meta.dtype = utils::ZarrDtype::uint8;

    // Exact divisor
    REQUIRE_EQ(meta.num_chunks_along(0), std::size_t(1));

    // Out of range dimension
    REQUIRE_EQ(meta.num_chunks_along(5), std::size_t(0));
}

// ---------------------------------------------------------------------------
// 5. ZarrArray v3 creation on disk with write/read
// ---------------------------------------------------------------------------

TEST_CASE("ZarrArray v3 create and read back") {
    auto dir = make_temp_dir("v3_create");

    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {32, 32};
    meta.chunks = {16, 16};
    meta.dtype = utils::ZarrDtype::uint16;
    meta.fill_value = 0.0;
    meta.node_type = "array";

    auto arr = utils::ZarrArray::create(dir / "test_v3.zarr", meta);

    REQUIRE_EQ(arr.version(), utils::ZarrVersion::v3);
    REQUIRE_EQ(arr.metadata().shape[0], std::size_t(32));
    REQUIRE_EQ(arr.metadata().chunks[0], std::size_t(16));
    REQUIRE(!arr.is_sharded());

    // zarr.json should exist
    REQUIRE(fs::exists(dir / "test_v3.zarr" / "zarr.json"));

    // Re-open and verify
    auto arr2 = utils::ZarrArray::open(dir / "test_v3.zarr");
    REQUIRE_EQ(arr2.version(), utils::ZarrVersion::v3);
    REQUIRE_EQ(arr2.metadata().dtype, utils::ZarrDtype::uint16);
    REQUIRE_EQ(arr2.metadata().shape[0], std::size_t(32));

    fs::remove_all(dir);
}

TEST_CASE("ZarrArray v3 chunk write/read round trip") {
    auto dir = make_temp_dir("v3_chunk_rw");

    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {16, 16};
    meta.chunks = {8, 8};
    meta.dtype = utils::ZarrDtype::float32;

    auto arr = utils::ZarrArray::create(dir / "arr.zarr", meta);

    // Write chunk at (0,0)
    std::vector<float> chunk_data(64, 2.718f);
    std::vector<std::byte> bytes(chunk_data.size() * sizeof(float));
    std::memcpy(bytes.data(), chunk_data.data(), bytes.size());

    std::array<std::size_t, 2> idx = {0, 0};
    arr.write_chunk(idx, bytes);
    REQUIRE(arr.chunk_exists(idx));

    auto read_back = arr.read_chunk(idx);
    REQUIRE(read_back.has_value());
    REQUIRE_EQ(read_back->size(), bytes.size());

    std::vector<float> got(64);
    std::memcpy(got.data(), read_back->data(), got.size() * sizeof(float));
    REQUIRE_NEAR(got[0], 2.718f, 1e-5);

    // Write another chunk at (1, 1)
    std::array<std::size_t, 2> idx2 = {1, 1};
    std::vector<float> chunk_data2(64, -1.5f);
    std::vector<std::byte> bytes2(chunk_data2.size() * sizeof(float));
    std::memcpy(bytes2.data(), chunk_data2.data(), bytes2.size());
    arr.write_chunk(idx2, bytes2);

    auto read_back2 = arr.read_chunk(idx2);
    REQUIRE(read_back2.has_value());
    std::vector<float> got2(64);
    std::memcpy(got2.data(), read_back2->data(), got2.size() * sizeof(float));
    REQUIRE_NEAR(got2[0], -1.5f, 1e-5);

    fs::remove_all(dir);
}

TEST_CASE("ZarrArray v3 chunk key format") {
    auto dir = make_temp_dir("v3_chunk_key");

    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {32, 32};
    meta.chunks = {16, 16};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.chunk_key_encoding = "default";

    auto arr = utils::ZarrArray::create(dir / "arr.zarr", meta);

    std::array<std::size_t, 2> idx = {1, 2};
    auto key = arr.chunk_key(idx);
    REQUIRE_EQ(key, std::string("c/1/2"));

    // v2 encoding
    utils::ZarrMetadata meta2;
    meta2.version = utils::ZarrVersion::v3;
    meta2.shape = {32, 32};
    meta2.chunks = {16, 16};
    meta2.dtype = utils::ZarrDtype::uint8;
    meta2.chunk_key_encoding = "v2";
    meta2.dimension_separator = ".";

    auto arr2 = utils::ZarrArray::create(dir / "arr2.zarr", meta2);
    auto key2 = arr2.chunk_key(idx);
    REQUIRE_EQ(key2, std::string("c.1.2"));

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// 6. ZarrArray v3 with sharding -- write and read back
// ---------------------------------------------------------------------------

TEST_CASE("Zarr v3 sharding metadata parse round trip") {
    std::string json_str = R"({
        "zarr_format": 3,
        "node_type": "array",
        "shape": [128, 128],
        "data_type": "uint16",
        "chunk_grid": {
            "name": "regular",
            "configuration": { "chunk_shape": [64, 64] }
        },
        "chunk_key_encoding": { "name": "default" },
        "fill_value": 0,
        "codecs": [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [16, 16],
                    "index_location": "end",
                    "index_codecs": [
                        { "name": "bytes", "configuration": { "endian": "little" } }
                    ],
                    "codecs": [
                        { "name": "bytes", "configuration": { "endian": "little" } }
                    ]
                }
            }
        ]
    })";

    auto meta = utils::detail::parse_zarr_json(json_str);
    REQUIRE(meta.shard_config.has_value());
    REQUIRE_EQ(meta.shard_config->sub_chunks.size(), std::size_t(2));
    REQUIRE_EQ(meta.shard_config->sub_chunks[0], std::size_t(16));
    REQUIRE_EQ(meta.shard_config->sub_chunks[1], std::size_t(16));
    REQUIRE(meta.shard_config->index_at_end);
    REQUIRE_EQ(meta.shard_config->index_codecs.size(), std::size_t(1));
    REQUIRE_EQ(meta.shard_config->index_codecs[0].name, std::string("bytes"));
    REQUIRE_EQ(meta.shard_config->sub_codecs.size(), std::size_t(1));
    REQUIRE_EQ(meta.shard_config->sub_codecs[0].name, std::string("bytes"));
}

TEST_CASE("Zarr v3 sharding metadata serialize with shard_config") {
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {64, 64};
    meta.chunks = {32, 32};
    meta.dtype = utils::ZarrDtype::float32;

    utils::ShardConfig sc;
    sc.sub_chunks = {8, 8};
    sc.index_at_end = true;
    meta.shard_config = sc;
    // Leave codecs empty so serialize_zarr_json builds a sharding_indexed codec

    auto json = utils::detail::serialize_zarr_json(meta);
    auto parsed = utils::detail::parse_zarr_json(json);

    REQUIRE(parsed.shard_config.has_value());
    REQUIRE_EQ(parsed.shard_config->sub_chunks[0], std::size_t(8));
    REQUIRE_EQ(parsed.shard_config->sub_chunks[1], std::size_t(8));
    REQUIRE(parsed.shard_config->index_at_end);
}

TEST_CASE("Zarr v3 sharding index_location start") {
    std::string json_str = R"({
        "zarr_format": 3,
        "node_type": "array",
        "shape": [32, 32],
        "data_type": "uint8",
        "chunk_grid": {
            "name": "regular",
            "configuration": { "chunk_shape": [32, 32] }
        },
        "chunk_key_encoding": { "name": "default" },
        "fill_value": 0,
        "codecs": [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [16, 16],
                    "index_location": "start",
                    "index_codecs": [
                        { "name": "bytes", "configuration": { "endian": "little" } }
                    ],
                    "codecs": [
                        { "name": "bytes", "configuration": { "endian": "little" } }
                    ]
                }
            }
        ]
    })";

    auto meta = utils::detail::parse_zarr_json(json_str);
    REQUIRE(meta.shard_config.has_value());
    REQUIRE(!meta.shard_config->index_at_end);
}

TEST_CASE("ZarrArray v3 sharded write_shard and read_inner_chunk") {
    auto dir = make_temp_dir("v3_shard_rw");

    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {32, 32};
    meta.chunks = {32, 32};  // one shard covers the whole array
    meta.dtype = utils::ZarrDtype::uint16;

    utils::ShardConfig sc;
    sc.sub_chunks = {16, 16};
    sc.index_at_end = true;
    meta.shard_config = sc;

    auto arr = utils::ZarrArray::create(dir / "sharded.zarr", meta);
    REQUIRE(arr.is_sharded());

    // Shard has 2x2 = 4 inner chunks of 16x16 uint16 = 512 bytes each
    const std::size_t inner_size = 16 * 16 * 2;

    std::vector<std::optional<std::vector<std::byte>>> inner_chunks(4);

    // Fill inner chunk 0 with value 10
    inner_chunks[0] = std::vector<std::byte>(inner_size, static_cast<std::byte>(10));
    // Fill inner chunk 1 with value 20
    inner_chunks[1] = std::vector<std::byte>(inner_size, static_cast<std::byte>(20));
    // Inner chunk 2: missing (nullopt)
    // Fill inner chunk 3 with value 30
    inner_chunks[3] = std::vector<std::byte>(inner_size, static_cast<std::byte>(30));

    std::array<std::size_t, 2> shard_idx = {0, 0};
    arr.write_shard(shard_idx, inner_chunks);

    // Read inner chunks back
    std::array<std::size_t, 2> inner0 = {0, 0};
    auto data0 = arr.read_inner_chunk(shard_idx, inner0);
    REQUIRE(data0.has_value());
    REQUIRE_EQ(data0->size(), inner_size);
    REQUIRE_EQ(static_cast<std::uint8_t>((*data0)[0]), std::uint8_t(10));

    std::array<std::size_t, 2> inner1 = {0, 1};
    auto data1 = arr.read_inner_chunk(shard_idx, inner1);
    REQUIRE(data1.has_value());
    REQUIRE_EQ(static_cast<std::uint8_t>((*data1)[0]), std::uint8_t(20));

    // Inner chunk 2 is missing
    std::array<std::size_t, 2> inner2 = {1, 0};
    auto data2 = arr.read_inner_chunk(shard_idx, inner2);
    REQUIRE(!data2.has_value());

    std::array<std::size_t, 2> inner3 = {1, 1};
    auto data3 = arr.read_inner_chunk(shard_idx, inner3);
    REQUIRE(data3.has_value());
    REQUIRE_EQ(static_cast<std::uint8_t>((*data3)[0]), std::uint8_t(30));

    fs::remove_all(dir);
}

TEST_CASE("ZarrArray v3 sharded write_shard index at start") {
    auto dir = make_temp_dir("v3_shard_start");

    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {16, 16};
    meta.chunks = {16, 16};
    meta.dtype = utils::ZarrDtype::uint8;

    utils::ShardConfig sc;
    sc.sub_chunks = {8, 8};
    sc.index_at_end = false;  // index at start
    meta.shard_config = sc;

    auto arr = utils::ZarrArray::create(dir / "sharded.zarr", meta);

    const std::size_t inner_size = 8 * 8 * 1;
    std::vector<std::optional<std::vector<std::byte>>> inner_chunks(4);
    inner_chunks[0] = std::vector<std::byte>(inner_size, static_cast<std::byte>(42));
    inner_chunks[1] = std::vector<std::byte>(inner_size, static_cast<std::byte>(99));
    // 2 and 3 are missing

    std::array<std::size_t, 2> shard_idx = {0, 0};
    arr.write_shard(shard_idx, inner_chunks);

    // Read back
    std::array<std::size_t, 2> inner0 = {0, 0};
    auto data0 = arr.read_inner_chunk(shard_idx, inner0);
    REQUIRE(data0.has_value());
    REQUIRE_EQ(static_cast<std::uint8_t>((*data0)[0]), std::uint8_t(42));

    std::array<std::size_t, 2> inner1 = {0, 1};
    auto data1 = arr.read_inner_chunk(shard_idx, inner1);
    REQUIRE(data1.has_value());
    REQUIRE_EQ(static_cast<std::uint8_t>((*data1)[0]), std::uint8_t(99));

    // Missing chunks
    std::array<std::size_t, 2> inner2 = {1, 0};
    auto data2 = arr.read_inner_chunk(shard_idx, inner2);
    REQUIRE(!data2.has_value());

    fs::remove_all(dir);
}

TEST_CASE("ZarrArray v3 read_chunk dispatches to sharding") {
    auto dir = make_temp_dir("v3_shard_dispatch");

    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {32, 32};
    meta.chunks = {32, 32};
    meta.dtype = utils::ZarrDtype::uint8;

    utils::ShardConfig sc;
    sc.sub_chunks = {16, 16};
    sc.index_at_end = true;
    meta.shard_config = sc;

    auto arr = utils::ZarrArray::create(dir / "sharded.zarr", meta);

    const std::size_t inner_size = 16 * 16 * 1;
    std::vector<std::optional<std::vector<std::byte>>> inner_chunks(4);
    inner_chunks[0] = std::vector<std::byte>(inner_size, static_cast<std::byte>(77));
    inner_chunks[1] = std::vector<std::byte>(inner_size, static_cast<std::byte>(88));
    inner_chunks[2] = std::vector<std::byte>(inner_size, static_cast<std::byte>(99));
    inner_chunks[3] = std::vector<std::byte>(inner_size, static_cast<std::byte>(55));

    std::array<std::size_t, 2> shard_idx = {0, 0};
    arr.write_shard(shard_idx, inner_chunks);

    // read_chunk with inner chunk indices should dispatch through read_inner_chunk_from_shard
    // Inner chunk (0,0) -> shard (0,0) inner (0,0)
    std::array<std::size_t, 2> chunk0 = {0, 0};
    auto d0 = arr.read_chunk(chunk0);
    REQUIRE(d0.has_value());
    REQUIRE_EQ(static_cast<std::uint8_t>((*d0)[0]), std::uint8_t(77));

    // Inner chunk (0,1)
    std::array<std::size_t, 2> chunk1 = {0, 1};
    auto d1 = arr.read_chunk(chunk1);
    REQUIRE(d1.has_value());
    REQUIRE_EQ(static_cast<std::uint8_t>((*d1)[0]), std::uint8_t(88));

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// 7. FileSystemStore erase and get_string/set_string
// ---------------------------------------------------------------------------

TEST_CASE("FileSystemStore erase and string ops") {
    auto dir = make_temp_dir("fsstore_erase");
    utils::FileSystemStore store(dir);

    store.set_string("hello", "world");
    REQUIRE(store.exists("hello"));
    REQUIRE_EQ(store.get_string("hello"), std::string("world"));

    store.erase("hello");
    REQUIRE(!store.exists("hello"));

    // get on non-existent key should throw
    bool threw = false;
    try {
        (void)store.get("nonexistent");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);

    fs::remove_all(dir);
}

TEST_CASE("FileSystemStore nested key paths") {
    auto dir = make_temp_dir("fsstore_nested");
    utils::FileSystemStore store(dir);

    store.set_string("a/b/c/data.json", "{\"x\": 1}");
    REQUIRE(store.exists("a/b/c/data.json"));
    REQUIRE_EQ(store.get_string("a/b/c/data.json"), std::string("{\"x\": 1}"));

    // get_partial on offset beyond file returns empty
    auto empty = store.get_partial("a/b/c/data.json", 1000, 10);
    REQUIRE(empty.has_value());
    REQUIRE_EQ(empty->size(), std::size_t(0));

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// 8. ZarrArray open from Store
// ---------------------------------------------------------------------------

TEST_CASE("ZarrArray open from FileSystemStore") {
    auto dir = make_temp_dir("store_open");

    // First, create an array on disk
    utils::ZarrMetadata meta;
    meta.shape = {32, 32};
    meta.chunks = {16, 16};
    meta.dtype = utils::ZarrDtype::float32;

    utils::ZarrArray::create(dir / "myarray", meta);

    // Now open through a Store
    auto store = std::make_shared<utils::FileSystemStore>(dir);
    auto arr = utils::ZarrArray::open(store, "myarray");

    REQUIRE_EQ(arr.metadata().shape[0], std::size_t(32));
    REQUIRE_EQ(arr.metadata().dtype, utils::ZarrDtype::float32);

    fs::remove_all(dir);
}

TEST_CASE("ZarrArray open from Store v3") {
    auto dir = make_temp_dir("store_open_v3");

    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {64};
    meta.chunks = {32};
    meta.dtype = utils::ZarrDtype::int32;

    utils::ZarrArray::create(dir / "v3arr", meta);

    auto store = std::make_shared<utils::FileSystemStore>(dir);
    auto arr = utils::ZarrArray::open(store, "v3arr");

    REQUIRE_EQ(arr.version(), utils::ZarrVersion::v3);
    REQUIRE_EQ(arr.metadata().dtype, utils::ZarrDtype::int32);

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// 9. V2 filter encode/decode round trips
// ---------------------------------------------------------------------------

TEST_CASE("ZarrFilter delta encode/decode round trip") {
    utils::ZarrFilter f;
    f.id = utils::ZarrFilterId::delta;
    f.dtype = utils::ZarrDtype::int32;

    // Create data: [10, 20, 30, 40]
    std::vector<std::int32_t> data = {10, 20, 30, 40};
    std::vector<std::byte> bytes(data.size() * sizeof(std::int32_t));
    std::memcpy(bytes.data(), data.data(), bytes.size());

    auto encoded = f.encode(bytes);
    auto decoded = f.decode(encoded);

    REQUIRE_EQ(decoded.size(), bytes.size());
    std::vector<std::int32_t> result(4);
    std::memcpy(result.data(), decoded.data(), decoded.size());
    REQUIRE_EQ(result[0], 10);
    REQUIRE_EQ(result[1], 20);
    REQUIRE_EQ(result[2], 30);
    REQUIRE_EQ(result[3], 40);
}

TEST_CASE("ZarrFilter fixedscaleoffset encode/decode round trip float64") {
    utils::ZarrFilter f;
    f.id = utils::ZarrFilterId::fixedscaleoffset;
    f.dtype = utils::ZarrDtype::float64;
    f.offset = 100.0;
    f.scale = 2.0;

    std::vector<double> data = {105.0, 110.0, 200.0};
    std::vector<std::byte> bytes(data.size() * sizeof(double));
    std::memcpy(bytes.data(), data.data(), bytes.size());

    auto encoded = f.encode(bytes);
    auto decoded = f.decode(encoded);

    REQUIRE_EQ(decoded.size(), bytes.size());
    std::vector<double> result(3);
    std::memcpy(result.data(), decoded.data(), decoded.size());
    REQUIRE_NEAR(result[0], 105.0, 1e-10);
    REQUIRE_NEAR(result[1], 110.0, 1e-10);
    REQUIRE_NEAR(result[2], 200.0, 1e-10);
}

TEST_CASE("ZarrFilter fixedscaleoffset float32") {
    utils::ZarrFilter f;
    f.id = utils::ZarrFilterId::fixedscaleoffset;
    f.dtype = utils::ZarrDtype::float32;
    f.offset = 10.0;
    f.scale = 3.0;

    std::vector<float> data = {15.0f, 20.0f};
    std::vector<std::byte> bytes(data.size() * sizeof(float));
    std::memcpy(bytes.data(), data.data(), bytes.size());

    auto encoded = f.encode(bytes);
    auto decoded = f.decode(encoded);

    std::vector<float> result(2);
    std::memcpy(result.data(), decoded.data(), decoded.size());
    REQUIRE_NEAR(result[0], 15.0f, 1e-4);
    REQUIRE_NEAR(result[1], 20.0f, 1e-4);
}

TEST_CASE("ZarrFilter fixedscaleoffset non-float passthrough") {
    utils::ZarrFilter f;
    f.id = utils::ZarrFilterId::fixedscaleoffset;
    f.dtype = utils::ZarrDtype::uint16;  // not float -> passthrough

    std::vector<std::byte> bytes = {std::byte(1), std::byte(2), std::byte(3), std::byte(4)};
    auto encoded = f.encode(bytes);
    REQUIRE_EQ(encoded.size(), bytes.size());
    REQUIRE_EQ(static_cast<std::uint8_t>(encoded[0]), std::uint8_t(1));
    REQUIRE_EQ(static_cast<std::uint8_t>(encoded[3]), std::uint8_t(4));
}

TEST_CASE("ZarrFilter quantize encode float64") {
    utils::ZarrFilter f;
    f.id = utils::ZarrFilterId::quantize;
    f.dtype = utils::ZarrDtype::float64;
    f.digits = 2;

    std::vector<double> data = {3.14159265, 2.71828182};
    std::vector<std::byte> bytes(data.size() * sizeof(double));
    std::memcpy(bytes.data(), data.data(), bytes.size());

    auto encoded = f.encode(bytes);
    REQUIRE_EQ(encoded.size(), bytes.size());

    std::vector<double> result(2);
    std::memcpy(result.data(), encoded.data(), encoded.size());
    REQUIRE_NEAR(result[0], 3.14, 1e-10);
    REQUIRE_NEAR(result[1], 2.72, 1e-10);

    // Quantize decode is identity
    auto decoded = f.decode(encoded);
    std::vector<double> result2(2);
    std::memcpy(result2.data(), decoded.data(), decoded.size());
    REQUIRE_NEAR(result2[0], 3.14, 1e-10);
}

TEST_CASE("ZarrFilter quantize encode float32") {
    utils::ZarrFilter f;
    f.id = utils::ZarrFilterId::quantize;
    f.dtype = utils::ZarrDtype::float32;
    f.digits = 1;

    std::vector<float> data = {3.14159f, 2.71828f};
    std::vector<std::byte> bytes(data.size() * sizeof(float));
    std::memcpy(bytes.data(), data.data(), bytes.size());

    auto encoded = f.encode(bytes);
    std::vector<float> result(2);
    std::memcpy(result.data(), encoded.data(), encoded.size());
    REQUIRE_NEAR(result[0], 3.1f, 0.05f);
    REQUIRE_NEAR(result[1], 2.7f, 0.05f);
}

TEST_CASE("ZarrFilter quantize non-float passthrough") {
    utils::ZarrFilter f;
    f.id = utils::ZarrFilterId::quantize;
    f.dtype = utils::ZarrDtype::int32;

    std::vector<std::byte> bytes = {std::byte(1), std::byte(2), std::byte(3), std::byte(4)};
    auto encoded = f.encode(bytes);
    REQUIRE_EQ(encoded.size(), bytes.size());
    // Should be passthrough
    for (std::size_t i = 0; i < bytes.size(); ++i)
        REQUIRE_EQ(static_cast<std::uint8_t>(encoded[i]), static_cast<std::uint8_t>(bytes[i]));
}

TEST_CASE("ZarrFilter bad elem size passthrough") {
    utils::ZarrFilter f;
    f.id = utils::ZarrFilterId::delta;
    f.dtype = utils::ZarrDtype::int32;

    // 5 bytes is not divisible by 4 -> passthrough
    std::vector<std::byte> bytes = {std::byte(1), std::byte(2), std::byte(3),
                                     std::byte(4), std::byte(5)};
    auto encoded = f.encode(bytes);
    REQUIRE_EQ(encoded.size(), bytes.size());
}

// ---------------------------------------------------------------------------
// 10. V2 .zarray round trip with filters
// ---------------------------------------------------------------------------

TEST_CASE("Zarr .zarray round trip with delta filter") {
    utils::ZarrMetadata meta;
    meta.shape = {100};
    meta.chunks = {50};
    meta.dtype = utils::ZarrDtype::int32;
    meta.compressor_id = "";

    utils::ZarrFilter f;
    f.id = utils::ZarrFilterId::delta;
    f.dtype = utils::ZarrDtype::int32;
    f.astype = utils::ZarrDtype::int32;
    meta.filters.push_back(f);

    auto json = utils::detail::serialize_zarray(meta);
    auto parsed = utils::detail::parse_zarray(json);

    REQUIRE_EQ(parsed.filters.size(), std::size_t(1));
    REQUIRE_EQ(parsed.filters[0].id, utils::ZarrFilterId::delta);
    REQUIRE_EQ(parsed.filters[0].dtype, utils::ZarrDtype::int32);
    REQUIRE_EQ(parsed.filters[0].astype, utils::ZarrDtype::int32);
}

TEST_CASE("Zarr .zarray round trip with fixedscaleoffset filter") {
    utils::ZarrMetadata meta;
    meta.shape = {64};
    meta.chunks = {64};
    meta.dtype = utils::ZarrDtype::float64;

    utils::ZarrFilter f;
    f.id = utils::ZarrFilterId::fixedscaleoffset;
    f.offset = 5.0;
    f.scale = 2.0;
    meta.filters.push_back(f);

    auto json = utils::detail::serialize_zarray(meta);
    auto parsed = utils::detail::parse_zarray(json);

    REQUIRE_EQ(parsed.filters.size(), std::size_t(1));
    REQUIRE_EQ(parsed.filters[0].id, utils::ZarrFilterId::fixedscaleoffset);
    REQUIRE_NEAR(parsed.filters[0].offset, 5.0, 1e-10);
    REQUIRE_NEAR(parsed.filters[0].scale, 2.0, 1e-10);
}

TEST_CASE("Zarr .zarray round trip with quantize filter") {
    utils::ZarrMetadata meta;
    meta.shape = {64};
    meta.chunks = {64};
    meta.dtype = utils::ZarrDtype::float32;

    utils::ZarrFilter f;
    f.id = utils::ZarrFilterId::quantize;
    f.digits = 3;
    meta.filters.push_back(f);

    auto json = utils::detail::serialize_zarray(meta);
    auto parsed = utils::detail::parse_zarray(json);

    REQUIRE_EQ(parsed.filters.size(), std::size_t(1));
    REQUIRE_EQ(parsed.filters[0].id, utils::ZarrFilterId::quantize);
    REQUIRE_EQ(parsed.filters[0].digits, 3);
}

TEST_CASE("ZarrArray v2 with delta filter write/read round trip") {
    auto dir = make_temp_dir("v2_filter_rw");

    utils::ZarrMetadata meta;
    meta.shape = {16};
    meta.chunks = {16};
    meta.dtype = utils::ZarrDtype::int32;
    meta.compressor_id = "";

    utils::ZarrFilter f;
    f.id = utils::ZarrFilterId::delta;
    f.dtype = utils::ZarrDtype::int32;
    meta.filters.push_back(f);

    auto arr = utils::ZarrArray::create(dir / "filtered.zarr", meta);

    std::vector<std::int32_t> data = {10, 20, 30, 40, 50, 60, 70, 80,
                                       90, 100, 110, 120, 130, 140, 150, 160};
    std::vector<std::byte> bytes(data.size() * sizeof(std::int32_t));
    std::memcpy(bytes.data(), data.data(), bytes.size());

    std::array<std::size_t, 1> idx = {0};
    arr.write_chunk(idx, bytes);

    auto read_back = arr.read_chunk(idx);
    REQUIRE(read_back.has_value());
    REQUIRE_EQ(read_back->size(), bytes.size());

    std::vector<std::int32_t> result(16);
    std::memcpy(result.data(), read_back->data(), read_back->size());
    for (int i = 0; i < 16; ++i) {
        REQUIRE_EQ(result[i], data[i]);
    }

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// 11. OmeZarrWriter and OmeZarrReader
// ---------------------------------------------------------------------------

TEST_CASE("OmeZarrWriter creates pyramid and OmeZarrReader reads it") {
    auto dir = make_temp_dir("ome_writer_reader");

    utils::MultiscaleMetadata ms;
    ms.version = "0.4";
    ms.name = "test_vol";
    ms.type = "gaussian";
    ms.axes = {
        {.name = "y", .type = utils::AxisType::space, .unit = "micrometer"},
        {.name = "x", .type = utils::AxisType::space, .unit = "micrometer"},
    };

    {
        utils::MultiscaleDataset ds0;
        ds0.path = "0";
        ds0.transforms.push_back(utils::ScaleTransform{{1.0, 1.0}});
        ms.datasets.push_back(ds0);
    }
    {
        utils::MultiscaleDataset ds1;
        ds1.path = "1";
        ds1.transforms.push_back(utils::ScaleTransform{{2.0, 2.0}});
        ms.datasets.push_back(ds1);
    }

    utils::OmeZarrWriter::Config cfg;
    cfg.root = dir / "volume.ome.zarr";
    cfg.multiscale = ms;
    cfg.dtype = utils::ZarrDtype::uint16;
    cfg.chunk_shape = {32, 32};

    utils::OmeZarrWriter writer(cfg);

    // Add 2 levels (don't hold references across add_level calls -- vector may reallocate)
    (void)writer.add_level({64, 64});
    (void)writer.add_level({32, 32});

    // Write a chunk to level 0
    std::vector<std::uint16_t> chunk_data(32 * 32, 42);
    std::vector<std::byte> chunk_bytes(chunk_data.size() * sizeof(std::uint16_t));
    std::memcpy(chunk_bytes.data(), chunk_data.data(), chunk_bytes.size());
    std::array<std::size_t, 2> idx0 = {0, 0};
    writer.write_chunk(0, idx0, chunk_bytes);

    // Write chunk to level 1
    std::vector<std::uint16_t> chunk_data1(32 * 32, 21);
    std::vector<std::byte> chunk_bytes1(chunk_data1.size() * sizeof(std::uint16_t));
    std::memcpy(chunk_bytes1.data(), chunk_data1.data(), chunk_bytes1.size());
    writer.write_chunk(1, idx0, chunk_bytes1);

    writer.finalize();

    // Verify .zattrs and .zgroup exist
    REQUIRE(fs::exists(dir / "volume.ome.zarr" / ".zattrs"));
    REQUIRE(fs::exists(dir / "volume.ome.zarr" / ".zgroup"));

    // Read back with OmeZarrReader
    utils::OmeZarrReader reader(dir / "volume.ome.zarr");

    REQUIRE_EQ(reader.num_levels(), std::size_t(2));
    REQUIRE_EQ(reader.multiscale().name, std::string("test_vol"));
    REQUIRE_EQ(reader.axes().size(), std::size_t(2));

    auto shape0 = reader.shape(0);
    REQUIRE_EQ(shape0.size(), std::size_t(2));
    REQUIRE_EQ(shape0[0], std::size_t(64));

    auto shape1 = reader.shape(1);
    REQUIRE_EQ(shape1[0], std::size_t(32));

    // Read chunk from level 0
    auto chunk = reader.read_chunk(0, idx0);
    REQUIRE(chunk.has_value());
    REQUIRE_EQ(chunk->size(), chunk_bytes.size());

    std::vector<std::uint16_t> result(32 * 32);
    std::memcpy(result.data(), chunk->data(), chunk->size());
    REQUIRE_EQ(result[0], std::uint16_t(42));

    // Voxel size
    auto vs = reader.voxel_size(0);
    REQUIRE_NEAR(vs[0], 1.0, 1e-12);
    REQUIRE_NEAR(vs[1], 1.0, 1e-12);

    auto vs1 = reader.voxel_size(1);
    REQUIRE_NEAR(vs1[0], 2.0, 1e-12);

    // No labels
    REQUIRE(!reader.has_labels());
    auto label_names = reader.label_names();
    REQUIRE(label_names.empty());

    // Not a plate
    REQUIRE(!reader.is_plate());
    auto pm = reader.plate_metadata();
    REQUIRE(!pm.has_value());

    fs::remove_all(dir);
}

TEST_CASE("OmeZarrWriter too many levels throws") {
    auto dir = make_temp_dir("ome_writer_excess");

    auto ms = utils::make_standard_multiscale(
        "small", {16, 16}, 1,
        {
            {.name = "y", .type = utils::AxisType::space},
            {.name = "x", .type = utils::AxisType::space},
        });

    utils::OmeZarrWriter::Config cfg;
    cfg.root = dir / "vol.zarr";
    cfg.multiscale = ms;
    cfg.dtype = utils::ZarrDtype::uint8;
    cfg.chunk_shape = {16, 16};

    utils::OmeZarrWriter writer(cfg);
    (void)writer.add_level({16, 16}); // ok, 1 level

    bool threw = false;
    try {
        (void)writer.add_level({8, 8}); // too many
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// 12. Label metadata serialization/parsing
// ---------------------------------------------------------------------------

TEST_CASE("LabelMetadata parse round trip") {
    // Build label metadata JSON manually
    auto json = utils::json_object({
        {"image-label", utils::json_object({
            {"version", utils::JsonValue("0.4")},
            {"colors", utils::JsonValue(utils::JsonArray{
                utils::json_object({
                    {"label-value", utils::JsonValue(1)},
                    {"rgba", utils::JsonValue(utils::JsonArray{
                        utils::JsonValue(255), utils::JsonValue(0),
                        utils::JsonValue(0), utils::JsonValue(255)
                    })}
                }),
                utils::json_object({
                    {"label-value", utils::JsonValue(2)},
                    {"rgba", utils::JsonValue(utils::JsonArray{
                        utils::JsonValue(0), utils::JsonValue(255),
                        utils::JsonValue(0), utils::JsonValue(128)
                    })}
                })
            })},
            {"properties", utils::JsonValue(utils::JsonArray{
                utils::JsonValue("area"), utils::JsonValue("centroid")
            })}
        })}
    });

    auto meta = utils::parse_label_metadata(json);

    REQUIRE_EQ(meta.version, std::string("0.4"));
    REQUIRE_EQ(meta.colors.size(), std::size_t(2));
    REQUIRE_EQ(meta.colors[0].label_value, std::uint32_t(1));
    REQUIRE_EQ(meta.colors[0].rgba[0], std::uint8_t(255));
    REQUIRE_EQ(meta.colors[0].rgba[1], std::uint8_t(0));
    REQUIRE_EQ(meta.colors[0].rgba[2], std::uint8_t(0));
    REQUIRE_EQ(meta.colors[0].rgba[3], std::uint8_t(255));
    REQUIRE_EQ(meta.colors[1].label_value, std::uint32_t(2));
    REQUIRE_EQ(meta.colors[1].rgba[1], std::uint8_t(255));
    REQUIRE_EQ(meta.properties.size(), std::size_t(2));
    REQUIRE_EQ(meta.properties[0], std::string("area"));
    REQUIRE_EQ(meta.properties[1], std::string("centroid"));
}

// ---------------------------------------------------------------------------
// 13. Plate metadata serialization/parsing
// ---------------------------------------------------------------------------

TEST_CASE("PlateMetadata parse round trip") {
    auto json = utils::json_object({
        {"plate", utils::json_object({
            {"version", utils::JsonValue("0.4")},
            {"name", utils::JsonValue("my_plate")},
            {"field_count", utils::JsonValue(2)},
            {"columns", utils::JsonValue(utils::JsonArray{
                utils::json_object({{"name", utils::JsonValue("1")}}),
                utils::json_object({{"name", utils::JsonValue("2")}}),
                utils::json_object({{"name", utils::JsonValue("3")}})
            })},
            {"rows", utils::JsonValue(utils::JsonArray{
                utils::json_object({{"name", utils::JsonValue("A")}}),
                utils::json_object({{"name", utils::JsonValue("B")}})
            })},
            {"wells", utils::JsonValue(utils::JsonArray{
                utils::json_object({
                    {"path", utils::JsonValue("A/1")},
                    {"rowIndex", utils::JsonValue(0)},
                    {"columnIndex", utils::JsonValue(0)}
                }),
                utils::json_object({
                    {"path", utils::JsonValue("B/2")},
                    {"rowIndex", utils::JsonValue(1)},
                    {"columnIndex", utils::JsonValue(1)}
                })
            })}
        })}
    });

    auto meta = utils::parse_plate_metadata(json);

    REQUIRE_EQ(meta.version, std::string("0.4"));
    REQUIRE_EQ(meta.name, std::string("my_plate"));
    REQUIRE_EQ(meta.field_count, std::size_t(2));
    REQUIRE_EQ(meta.columns.size(), std::size_t(3));
    REQUIRE_EQ(meta.columns[0], std::string("1"));
    REQUIRE_EQ(meta.columns[2], std::string("3"));
    REQUIRE_EQ(meta.rows.size(), std::size_t(2));
    REQUIRE_EQ(meta.rows[0], std::string("A"));
    REQUIRE_EQ(meta.rows[1], std::string("B"));
    REQUIRE_EQ(meta.wells.size(), std::size_t(2));
    REQUIRE_EQ(meta.wells[0].path, std::string("A/1"));
    REQUIRE_EQ(meta.wells[0].row, std::size_t(0));
    REQUIRE_EQ(meta.wells[0].col, std::size_t(0));
    REQUIRE_EQ(meta.wells[1].path, std::string("B/2"));
    REQUIRE_EQ(meta.wells[1].row, std::size_t(1));
    REQUIRE_EQ(meta.wells[1].col, std::size_t(1));
}

TEST_CASE("PlateMetadata parse missing plate throws") {
    auto json = utils::json_object({{"not_plate", utils::JsonValue("x")}});
    bool threw = false;
    try {
        (void)utils::parse_plate_metadata(json);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
}

// ---------------------------------------------------------------------------
// 14. Coordinate transforms parsing
// ---------------------------------------------------------------------------

TEST_CASE("CoordinateTransform parse scale and translation") {
    auto json = utils::JsonValue(utils::JsonArray{
        utils::json_object({
            {"type", utils::JsonValue("scale")},
            {"scale", utils::JsonValue(utils::JsonArray{
                utils::JsonValue(1.0), utils::JsonValue(2.0), utils::JsonValue(3.0)
            })}
        }),
        utils::json_object({
            {"type", utils::JsonValue("translation")},
            {"translation", utils::JsonValue(utils::JsonArray{
                utils::JsonValue(10.0), utils::JsonValue(20.0), utils::JsonValue(30.0)
            })}
        })
    });

    auto transforms = utils::ome_detail::parse_transforms(json);
    REQUIRE_EQ(transforms.size(), std::size_t(2));

    auto* st = std::get_if<utils::ScaleTransform>(&transforms[0]);
    REQUIRE(st != nullptr);
    REQUIRE_EQ(st->scale.size(), std::size_t(3));
    REQUIRE_NEAR(st->scale[0], 1.0, 1e-12);
    REQUIRE_NEAR(st->scale[1], 2.0, 1e-12);
    REQUIRE_NEAR(st->scale[2], 3.0, 1e-12);

    auto* tt = std::get_if<utils::TranslationTransform>(&transforms[1]);
    REQUIRE(tt != nullptr);
    REQUIRE_EQ(tt->translation.size(), std::size_t(3));
    REQUIRE_NEAR(tt->translation[0], 10.0, 1e-12);
    REQUIRE_NEAR(tt->translation[1], 20.0, 1e-12);
    REQUIRE_NEAR(tt->translation[2], 30.0, 1e-12);
}

TEST_CASE("CoordinateTransform serialize round trip") {
    std::vector<utils::CoordinateTransform> transforms;
    transforms.emplace_back(utils::ScaleTransform{{1.5, 2.5, 3.5}});
    transforms.emplace_back(utils::TranslationTransform{{100.0, 200.0, 300.0}});

    auto json = utils::ome_detail::serialize_transforms(transforms);
    auto parsed = utils::ome_detail::parse_transforms(json);

    REQUIRE_EQ(parsed.size(), std::size_t(2));

    auto* st = std::get_if<utils::ScaleTransform>(&parsed[0]);
    REQUIRE(st != nullptr);
    REQUIRE_NEAR(st->scale[0], 1.5, 1e-12);

    auto* tt = std::get_if<utils::TranslationTransform>(&parsed[1]);
    REQUIRE(tt != nullptr);
    REQUIRE_NEAR(tt->translation[0], 100.0, 1e-12);
}

TEST_CASE("CoordinateTransform parse non-array returns empty") {
    auto json = utils::JsonValue("not an array");
    auto transforms = utils::ome_detail::parse_transforms(json);
    REQUIRE(transforms.empty());
}

// ---------------------------------------------------------------------------
// 15. AxisType parsing
// ---------------------------------------------------------------------------

TEST_CASE("AxisType parse and string round trip") {
    REQUIRE_EQ(utils::ome_detail::parse_axis_type("space"), utils::AxisType::space);
    REQUIRE_EQ(utils::ome_detail::parse_axis_type("time"), utils::AxisType::time);
    REQUIRE_EQ(utils::ome_detail::parse_axis_type("channel"), utils::AxisType::channel);
    REQUIRE_EQ(utils::ome_detail::parse_axis_type("unknown"), utils::AxisType::custom);

    REQUIRE_EQ(utils::ome_detail::axis_type_string(utils::AxisType::space), std::string_view("space"));
    REQUIRE_EQ(utils::ome_detail::axis_type_string(utils::AxisType::time), std::string_view("time"));
    REQUIRE_EQ(utils::ome_detail::axis_type_string(utils::AxisType::channel), std::string_view("channel"));
    REQUIRE_EQ(utils::ome_detail::axis_type_string(utils::AxisType::custom), std::string_view("custom"));
}

// ---------------------------------------------------------------------------
// 16. make_standard_multiscale
// ---------------------------------------------------------------------------

TEST_CASE("make_standard_multiscale") {
    auto ms = utils::make_standard_multiscale(
        "test_vol",
        {256, 256, 128},
        4,
        {
            {.name = "z", .type = utils::AxisType::space, .unit = "um"},
            {.name = "y", .type = utils::AxisType::space, .unit = "um"},
            {.name = "x", .type = utils::AxisType::space, .unit = "um"},
        },
        {0.5, 0.5, 1.0}
    );

    REQUIRE_EQ(ms.version, std::string("0.4"));
    REQUIRE_EQ(ms.name, std::string("test_vol"));
    REQUIRE_EQ(ms.type, std::string("gaussian"));
    REQUIRE_EQ(ms.num_levels(), std::size_t(4));
    REQUIRE_EQ(ms.ndim(), std::size_t(3));

    // Level 0: factor=1 -> scale = voxel_size * 1
    auto vs0 = ms.voxel_size(0);
    REQUIRE_NEAR(vs0[0], 0.5, 1e-12);   // x
    REQUIRE_NEAR(vs0[1], 0.5, 1e-12);   // y
    REQUIRE_NEAR(vs0[2], 1.0, 1e-12);   // z

    // Level 1: factor=2
    auto vs1 = ms.voxel_size(1);
    REQUIRE_NEAR(vs1[0], 1.0, 1e-12);
    REQUIRE_NEAR(vs1[1], 1.0, 1e-12);
    REQUIRE_NEAR(vs1[2], 2.0, 1e-12);

    // Level 3: factor=8
    auto vs3 = ms.voxel_size(3);
    REQUIRE_NEAR(vs3[0], 4.0, 1e-12);
    REQUIRE_NEAR(vs3[1], 4.0, 1e-12);
    REQUIRE_NEAR(vs3[2], 8.0, 1e-12);

    // Dataset paths
    REQUIRE_EQ(ms.datasets[0].path, std::string("0"));
    REQUIRE_EQ(ms.datasets[3].path, std::string("3"));
}

// ---------------------------------------------------------------------------
// 17. compute_downsample_factors
// ---------------------------------------------------------------------------

TEST_CASE("compute_downsample_factors") {
    auto ms = utils::make_standard_multiscale(
        "test", {64, 64, 64}, 3,
        {
            {.name = "z", .type = utils::AxisType::space},
            {.name = "y", .type = utils::AxisType::space},
            {.name = "x", .type = utils::AxisType::space},
        },
        {1.0, 1.0, 1.0}
    );

    auto factors = utils::compute_downsample_factors(ms, 0, 1);
    REQUIRE_EQ(factors.size(), std::size_t(3));
    REQUIRE_NEAR(factors[0], 2.0, 1e-12);
    REQUIRE_NEAR(factors[1], 2.0, 1e-12);
    REQUIRE_NEAR(factors[2], 2.0, 1e-12);

    auto factors02 = utils::compute_downsample_factors(ms, 0, 2);
    REQUIRE_NEAR(factors02[0], 4.0, 1e-12);
}

TEST_CASE("compute_downsample_factors out of range throws") {
    auto ms = utils::make_standard_multiscale(
        "test", {32, 32}, 2,
        {
            {.name = "y", .type = utils::AxisType::space},
            {.name = "x", .type = utils::AxisType::space},
        });

    bool threw = false;
    try {
        (void)utils::compute_downsample_factors(ms, 0, 5);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
}

// ---------------------------------------------------------------------------
// 18. MultiscaleMetadata with translation transforms
// ---------------------------------------------------------------------------

TEST_CASE("MultiscaleMetadata with translation transform round trip") {
    utils::MultiscaleMetadata meta;
    meta.version = "0.4";
    meta.name = "translated";
    meta.axes = {
        {.name = "z", .type = utils::AxisType::space, .unit = "um"},
        {.name = "y", .type = utils::AxisType::space, .unit = "um"},
        {.name = "x", .type = utils::AxisType::space, .unit = "um"},
    };

    utils::MultiscaleDataset ds;
    ds.path = "0";
    ds.transforms.push_back(utils::ScaleTransform{{1.0, 1.0, 1.0}});
    ds.transforms.push_back(utils::TranslationTransform{{10.0, 20.0, 30.0}});
    meta.datasets.push_back(ds);

    auto jv = utils::serialize_ome_metadata(meta);
    auto parsed = utils::parse_ome_metadata(jv);

    REQUIRE_EQ(parsed.datasets[0].transforms.size(), std::size_t(2));

    auto* st = std::get_if<utils::ScaleTransform>(&parsed.datasets[0].transforms[0]);
    REQUIRE(st != nullptr);

    auto* tt = std::get_if<utils::TranslationTransform>(&parsed.datasets[0].transforms[1]);
    REQUIRE(tt != nullptr);
    REQUIRE_NEAR(tt->translation[0], 10.0, 1e-12);
    REQUIRE_NEAR(tt->translation[1], 20.0, 1e-12);
    REQUIRE_NEAR(tt->translation[2], 30.0, 1e-12);
}

// ---------------------------------------------------------------------------
// 19. MultiscaleMetadata with time/channel axes
// ---------------------------------------------------------------------------

TEST_CASE("MultiscaleMetadata with time and channel axes") {
    utils::MultiscaleMetadata meta;
    meta.version = "0.4";
    meta.axes = {
        {.name = "t", .type = utils::AxisType::time, .unit = "second"},
        {.name = "c", .type = utils::AxisType::channel},
        {.name = "z", .type = utils::AxisType::space, .unit = "um"},
        {.name = "y", .type = utils::AxisType::space, .unit = "um"},
        {.name = "x", .type = utils::AxisType::space, .unit = "um"},
    };

    utils::MultiscaleDataset ds;
    ds.path = "0";
    ds.transforms.push_back(utils::ScaleTransform{{1.0, 1.0, 0.5, 0.5, 0.5}});
    meta.datasets.push_back(ds);

    REQUIRE_EQ(meta.ndim(), std::size_t(5));
    auto ti = meta.axis_index("t");
    REQUIRE(ti.has_value());
    REQUIRE_EQ(*ti, std::size_t(0));

    auto ci = meta.axis_index("c");
    REQUIRE(ci.has_value());
    REQUIRE_EQ(*ci, std::size_t(1));

    auto jv = utils::serialize_ome_metadata(meta);
    auto parsed = utils::parse_ome_metadata(jv);

    REQUIRE_EQ(parsed.axes.size(), std::size_t(5));
    REQUIRE_EQ(parsed.axes[0].type, utils::AxisType::time);
    REQUIRE_EQ(parsed.axes[0].unit, std::string("second"));
    REQUIRE_EQ(parsed.axes[1].type, utils::AxisType::channel);
    REQUIRE_EQ(parsed.axes[2].type, utils::AxisType::space);
}

// ---------------------------------------------------------------------------
// 20. MultiscaleMetadata voxel_size out of range throws
// ---------------------------------------------------------------------------

TEST_CASE("MultiscaleMetadata voxel_size out of range throws") {
    utils::MultiscaleMetadata meta;
    meta.axes = {{.name = "x", .type = utils::AxisType::space}};
    meta.datasets.push_back(utils::MultiscaleDataset{.path = "0"});

    bool threw = false;
    try {
        (void)meta.voxel_size(5);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
}

// ---------------------------------------------------------------------------
// 21. ConsolidatedMetadata parsing
// ---------------------------------------------------------------------------

TEST_CASE("ConsolidatedMetadata parse") {
    std::string json_str = R"({
        "metadata": {
            "0/.zarray": {
                "shape": [64, 64],
                "chunks": [32, 32],
                "dtype": "<u2",
                "compressor": null,
                "fill_value": 0,
                "order": "C",
                "filters": null,
                "dimension_separator": "."
            },
            "1/.zarray": {
                "shape": [32, 32],
                "chunks": [16, 16],
                "dtype": "<f4",
                "compressor": null,
                "fill_value": 0,
                "order": "C",
                "filters": null,
                "dimension_separator": "."
            },
            "mygroup/.zattrs": {
                "key": "value"
            }
        }
    })";

    auto cm = utils::detail::ConsolidatedMetadata::parse(json_str);

    REQUIRE_EQ(cm.arrays.size(), std::size_t(2));
    REQUIRE(cm.arrays.count("0") > 0);
    REQUIRE(cm.arrays.count("1") > 0);
    REQUIRE_EQ(cm.arrays["0"].shape[0], std::size_t(64));
    REQUIRE_EQ(cm.arrays["0"].dtype, utils::ZarrDtype::uint16);
    REQUIRE_EQ(cm.arrays["1"].shape[0], std::size_t(32));
    REQUIRE_EQ(cm.arrays["1"].dtype, utils::ZarrDtype::float32);

    REQUIRE(cm.attrs.count("mygroup") > 0);
}

// ---------------------------------------------------------------------------
// 22. detect_version
// ---------------------------------------------------------------------------

TEST_CASE("detect_version v2 vs v3") {
    auto dir = make_temp_dir("detect_version");

    // v2: write .zarray
    auto v2_path = dir / "v2arr";
    fs::create_directories(v2_path);
    utils::detail::write_file(v2_path / ".zarray", "{}");
    REQUIRE_EQ(utils::detail::detect_version(v2_path), utils::ZarrVersion::v2);

    // v3: write zarr.json
    auto v3_path = dir / "v3arr";
    fs::create_directories(v3_path);
    utils::detail::write_file(v3_path / "zarr.json", "{}");
    REQUIRE_EQ(utils::detail::detect_version(v3_path), utils::ZarrVersion::v3);

    // Neither: should throw
    auto bad_path = dir / "badarr";
    fs::create_directories(bad_path);
    bool threw = false;
    try {
        utils::detail::detect_version(bad_path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// 23. ZarrArray attributes read/write
// ---------------------------------------------------------------------------

TEST_CASE("ZarrArray v2 attributes read/write") {
    auto dir = make_temp_dir("v2_attrs");

    utils::ZarrMetadata meta;
    meta.shape = {16};
    meta.chunks = {16};
    meta.dtype = utils::ZarrDtype::uint8;

    auto arr = utils::ZarrArray::create(dir / "arr.zarr", meta);

    // Initially no .zattrs -> should return "{}"
    auto attrs = arr.read_attrs();
    REQUIRE_EQ(attrs, std::string("{}"));

    // Write attributes
    arr.write_attrs("{\"description\": \"test array\"}");
    auto attrs2 = arr.read_attrs();
    REQUIRE(attrs2.find("test array") != std::string::npos);

    fs::remove_all(dir);
}

TEST_CASE("ZarrArray v3 attributes read/write") {
    auto dir = make_temp_dir("v3_attrs");

    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {16};
    meta.chunks = {16};
    meta.dtype = utils::ZarrDtype::uint8;

    auto arr = utils::ZarrArray::create(dir / "arr.zarr", meta);

    // Write attributes (merges into zarr.json)
    arr.write_attrs("{\"info\": \"v3 test\"}");
    auto attrs = arr.read_attrs();
    REQUIRE(attrs.find("v3 test") != std::string::npos);

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// 24. ZarrArray v2 chunk key format
// ---------------------------------------------------------------------------

TEST_CASE("ZarrArray v2 chunk key format") {
    auto dir = make_temp_dir("v2_chunk_key");

    utils::ZarrMetadata meta;
    meta.shape = {32, 32, 32};
    meta.chunks = {16, 16, 16};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.dimension_separator = ".";

    auto arr = utils::ZarrArray::create(dir / "arr.zarr", meta);

    std::array<std::size_t, 3> idx = {1, 2, 3};
    auto key = arr.chunk_key(idx);
    REQUIRE_EQ(key, std::string("1.2.3"));

    // With "/" separator
    utils::ZarrMetadata meta2;
    meta2.shape = {32, 32};
    meta2.chunks = {16, 16};
    meta2.dtype = utils::ZarrDtype::uint8;
    meta2.dimension_separator = "/";

    auto arr2 = utils::ZarrArray::create(dir / "arr2.zarr", meta2);
    std::array<std::size_t, 2> idx2 = {0, 1};
    auto key2 = arr2.chunk_key(idx2);
    REQUIRE_EQ(key2, std::string("0/1"));

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// 25. 3D ZarrArray multi-dimensional chunk indexing
// ---------------------------------------------------------------------------

TEST_CASE("ZarrArray 3D chunk write/read") {
    auto dir = make_temp_dir("3d_chunks");

    utils::ZarrMetadata meta;
    meta.shape = {16, 16, 16};
    meta.chunks = {8, 8, 8};
    meta.dtype = utils::ZarrDtype::uint8;

    auto arr = utils::ZarrArray::create(dir / "arr.zarr", meta);

    // Write 2x2x2 = 8 chunks
    for (std::size_t z = 0; z < 2; ++z) {
        for (std::size_t y = 0; y < 2; ++y) {
            for (std::size_t x = 0; x < 2; ++x) {
                auto val = static_cast<std::byte>(z * 4 + y * 2 + x + 1);
                std::vector<std::byte> chunk_data(8 * 8 * 8, val);
                std::array<std::size_t, 3> idx = {z, y, x};
                arr.write_chunk(idx, chunk_data);
            }
        }
    }

    // Verify each chunk
    for (std::size_t z = 0; z < 2; ++z) {
        for (std::size_t y = 0; y < 2; ++y) {
            for (std::size_t x = 0; x < 2; ++x) {
                std::array<std::size_t, 3> idx = {z, y, x};
                REQUIRE(arr.chunk_exists(idx));
                auto data = arr.read_chunk(idx);
                REQUIRE(data.has_value());
                REQUIRE_EQ(data->size(), std::size_t(512));
                auto expected = static_cast<std::uint8_t>(z * 4 + y * 2 + x + 1);
                REQUIRE_EQ(static_cast<std::uint8_t>((*data)[0]), expected);
                REQUIRE_EQ(static_cast<std::uint8_t>((*data)[511]), expected);
            }
        }
    }

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// 26. .zarray fill_value with non-zero value
// ---------------------------------------------------------------------------

TEST_CASE("Zarr .zarray round trip with non-zero fill_value") {
    utils::ZarrMetadata meta;
    meta.shape = {32};
    meta.chunks = {32};
    meta.dtype = utils::ZarrDtype::float64;
    meta.fill_value = 999.5;

    auto json = utils::detail::serialize_zarray(meta);
    auto parsed = utils::detail::parse_zarray(json);

    REQUIRE(parsed.fill_value.has_value());
    REQUIRE_NEAR(*parsed.fill_value, 999.5, 1e-10);
}

TEST_CASE("Zarr .zarray round trip with null fill_value") {
    utils::ZarrMetadata meta;
    meta.shape = {32};
    meta.chunks = {32};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.fill_value = std::nullopt;

    auto json = utils::detail::serialize_zarray(meta);
    auto parsed = utils::detail::parse_zarray(json);

    REQUIRE(!parsed.fill_value.has_value());
}

// ---------------------------------------------------------------------------
// 27. Byteswap helper
// ---------------------------------------------------------------------------

TEST_CASE("byteswap_inplace") {
    // 4-byte element: [0x01, 0x02, 0x03, 0x04] -> [0x04, 0x03, 0x02, 0x01]
    std::vector<std::byte> data = {std::byte(0x01), std::byte(0x02),
                                    std::byte(0x03), std::byte(0x04)};
    utils::detail::byteswap_inplace(data, 4);
    REQUIRE_EQ(static_cast<std::uint8_t>(data[0]), std::uint8_t(0x04));
    REQUIRE_EQ(static_cast<std::uint8_t>(data[1]), std::uint8_t(0x03));
    REQUIRE_EQ(static_cast<std::uint8_t>(data[2]), std::uint8_t(0x02));
    REQUIRE_EQ(static_cast<std::uint8_t>(data[3]), std::uint8_t(0x01));

    // elem_size = 1 -> no-op
    std::vector<std::byte> data2 = {std::byte(0xAA)};
    utils::detail::byteswap_inplace(data2, 1);
    REQUIRE_EQ(static_cast<std::uint8_t>(data2[0]), std::uint8_t(0xAA));

    // 2-byte elements: [0x01, 0x02, 0x03, 0x04] -> [0x02, 0x01, 0x04, 0x03]
    std::vector<std::byte> data3 = {std::byte(0x01), std::byte(0x02),
                                     std::byte(0x03), std::byte(0x04)};
    utils::detail::byteswap_inplace(data3, 2);
    REQUIRE_EQ(static_cast<std::uint8_t>(data3[0]), std::uint8_t(0x02));
    REQUIRE_EQ(static_cast<std::uint8_t>(data3[1]), std::uint8_t(0x01));
    REQUIRE_EQ(static_cast<std::uint8_t>(data3[2]), std::uint8_t(0x04));
    REQUIRE_EQ(static_cast<std::uint8_t>(data3[3]), std::uint8_t(0x03));
}

// ---------------------------------------------------------------------------
// 28. detail file I/O helpers
// ---------------------------------------------------------------------------

TEST_CASE("detail read_file and write_file round trip") {
    auto dir = make_temp_dir("detail_io");
    fs::create_directories(dir);

    utils::detail::write_file(dir / "test.txt", "hello world");
    auto content = utils::detail::read_file(dir / "test.txt");
    REQUIRE_EQ(content, std::string("hello world"));

    // Bytes version
    std::vector<std::byte> bytes = {std::byte(0xDE), std::byte(0xAD)};
    utils::detail::write_file_bytes(dir / "test.bin", bytes);
    auto bytes_back = utils::detail::read_file_bytes(dir / "test.bin");
    REQUIRE_EQ(bytes_back.size(), std::size_t(2));
    REQUIRE_EQ(static_cast<std::uint8_t>(bytes_back[0]), std::uint8_t(0xDE));
    REQUIRE_EQ(static_cast<std::uint8_t>(bytes_back[1]), std::uint8_t(0xAD));

    // read_file on non-existent should throw
    bool threw = false;
    try {
        utils::detail::read_file(dir / "nonexistent.txt");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// 29. OmeZarrReader with labels
// ---------------------------------------------------------------------------

TEST_CASE("OmeZarrReader labels directory") {
    auto dir = make_temp_dir("ome_labels");

    // Create a minimal OME-Zarr with labels
    auto ms = utils::make_standard_multiscale(
        "img", {16, 16}, 1,
        {
            {.name = "y", .type = utils::AxisType::space},
            {.name = "x", .type = utils::AxisType::space},
        });

    utils::OmeZarrWriter::Config cfg;
    cfg.root = dir / "img.ome.zarr";
    cfg.multiscale = ms;
    cfg.dtype = utils::ZarrDtype::uint16;
    cfg.chunk_shape = {16, 16};

    utils::OmeZarrWriter writer(cfg);
    (void)writer.add_level({16, 16});
    writer.finalize();

    // Create labels directory with .zattrs listing the label names
    auto labels_dir = dir / "img.ome.zarr" / "labels";
    fs::create_directories(labels_dir);
    utils::detail::write_file(labels_dir / ".zattrs",
        "{\"labels\": [\"segmentation\"]}\n");

    // Create the label as a full OME-Zarr
    auto label_ms = utils::make_standard_multiscale(
        "segmentation", {16, 16}, 1,
        {
            {.name = "y", .type = utils::AxisType::space},
            {.name = "x", .type = utils::AxisType::space},
        });

    utils::OmeZarrWriter::Config label_cfg;
    label_cfg.root = labels_dir / "segmentation";
    label_cfg.multiscale = label_ms;
    label_cfg.dtype = utils::ZarrDtype::uint8;
    label_cfg.chunk_shape = {16, 16};

    utils::OmeZarrWriter label_writer(label_cfg);
    (void)label_writer.add_level({16, 16});
    label_writer.finalize();

    // Now read
    utils::OmeZarrReader reader(dir / "img.ome.zarr");
    REQUIRE(reader.has_labels());

    auto names = reader.label_names();
    REQUIRE_EQ(names.size(), std::size_t(1));
    REQUIRE_EQ(names[0], std::string("segmentation"));

    auto label_reader = reader.open_label("segmentation");
    REQUIRE_EQ(label_reader.num_levels(), std::size_t(1));

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// 30. MultiscaleMetadata with type field
// ---------------------------------------------------------------------------

TEST_CASE("MultiscaleMetadata type field preserved") {
    utils::MultiscaleMetadata meta;
    meta.version = "0.4";
    meta.type = "gaussian";
    meta.axes = {
        {.name = "y", .type = utils::AxisType::space},
        {.name = "x", .type = utils::AxisType::space},
    };
    meta.datasets.push_back(utils::MultiscaleDataset{
        .path = "0",
        .transforms = {utils::ScaleTransform{{1.0, 1.0}}}
    });

    auto jv = utils::serialize_ome_metadata(meta);
    auto parsed = utils::parse_ome_metadata(jv);

    REQUIRE_EQ(parsed.type, std::string("gaussian"));
}

// ---------------------------------------------------------------------------
// 31. Zarr .zarray with multiple filters
// ---------------------------------------------------------------------------

TEST_CASE("Zarr .zarray with multiple filters round trip") {
    utils::ZarrMetadata meta;
    meta.shape = {64};
    meta.chunks = {64};
    meta.dtype = utils::ZarrDtype::float64;

    utils::ZarrFilter f1;
    f1.id = utils::ZarrFilterId::delta;
    f1.dtype = utils::ZarrDtype::float64;
    f1.astype = utils::ZarrDtype::float64;
    meta.filters.push_back(f1);

    utils::ZarrFilter f2;
    f2.id = utils::ZarrFilterId::quantize;
    f2.digits = 5;
    meta.filters.push_back(f2);

    auto json = utils::detail::serialize_zarray(meta);
    auto parsed = utils::detail::parse_zarray(json);

    REQUIRE_EQ(parsed.filters.size(), std::size_t(2));
    REQUIRE_EQ(parsed.filters[0].id, utils::ZarrFilterId::delta);
    REQUIRE_EQ(parsed.filters[1].id, utils::ZarrFilterId::quantize);
    REQUIRE_EQ(parsed.filters[1].digits, 5);
}

// ---------------------------------------------------------------------------
// 32. ZarrArray create with CodecRegistry
// ---------------------------------------------------------------------------

TEST_CASE("ZarrArray create and open with CodecRegistry") {
    auto dir = make_temp_dir("codec_registry");

    // Create a v2 array with a custom codec name
    utils::ZarrMetadata meta;
    meta.shape = {16};
    meta.chunks = {16};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.compressor_id = "mycodec";

    // Define a trivial "codec" that is just identity
    utils::ZarrArray::CodecRegistry registry;
    registry["mycodec"] = utils::ZarrArray::Codec{
        .compress = [](std::span<const std::byte> data) {
            return std::vector<std::byte>(data.begin(), data.end());
        },
        .decompress = [](std::span<const std::byte> data, std::size_t) {
            return std::vector<std::byte>(data.begin(), data.end());
        }
    };

    auto arr = utils::ZarrArray::create(dir / "arr.zarr", meta, registry);

    std::vector<std::byte> chunk(16, std::byte(42));
    std::array<std::size_t, 1> idx = {0};
    arr.write_chunk(idx, chunk);

    auto read_back = arr.read_chunk(idx);
    REQUIRE(read_back.has_value());
    REQUIRE_EQ(static_cast<std::uint8_t>((*read_back)[0]), std::uint8_t(42));

    // Re-open with registry
    auto arr2 = utils::ZarrArray::open(dir / "arr.zarr", registry);
    auto read_back2 = arr2.read_chunk(idx);
    REQUIRE(read_back2.has_value());
    REQUIRE_EQ(static_cast<std::uint8_t>((*read_back2)[0]), std::uint8_t(42));

    fs::remove_all(dir);
}

TEST_CASE("ZarrArray v3 create with CodecRegistry") {
    auto dir = make_temp_dir("v3_codec_registry");

    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {16};
    meta.chunks = {16};
    meta.dtype = utils::ZarrDtype::uint8;

    utils::ZarrCodecConfig bytes_codec;
    bytes_codec.name = "bytes";
    meta.codecs.push_back(bytes_codec);

    utils::ZarrCodecConfig custom_codec;
    custom_codec.name = "mycompressor";
    meta.codecs.push_back(custom_codec);

    utils::ZarrArray::CodecRegistry registry;
    registry["mycompressor"] = utils::ZarrArray::Codec{
        .compress = [](std::span<const std::byte> data) {
            return std::vector<std::byte>(data.begin(), data.end());
        },
        .decompress = [](std::span<const std::byte> data, std::size_t) {
            return std::vector<std::byte>(data.begin(), data.end());
        }
    };

    auto arr = utils::ZarrArray::create(dir / "arr.zarr", meta, registry);

    std::vector<std::byte> chunk(16, std::byte(99));
    std::array<std::size_t, 1> idx = {0};
    arr.write_chunk(idx, chunk);

    auto read_back = arr.read_chunk(idx);
    REQUIRE(read_back.has_value());
    REQUIRE_EQ(static_cast<std::uint8_t>((*read_back)[0]), std::uint8_t(99));

    // Re-open with registry
    auto arr2 = utils::ZarrArray::open(dir / "arr.zarr", registry);
    auto read_back2 = arr2.read_chunk(idx);
    REQUIRE(read_back2.has_value());
    REQUIRE_EQ(static_cast<std::uint8_t>((*read_back2)[0]), std::uint8_t(99));

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// 33. Endianness helper
// ---------------------------------------------------------------------------

TEST_CASE("is_little_endian") {
    // We just verify it returns a bool and doesn't crash.
    // On x86/ARM it should be true.
    bool le = utils::detail::is_little_endian();
    REQUIRE(le); // almost all modern CPUs
}

UTILS_TEST_MAIN()
