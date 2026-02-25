#include "test.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

#include <xtensor/containers/xarray.hpp>

#include "vc/zarr/Zarr.hpp"

namespace fs = std::filesystem;
using namespace vc::zarr;

// RAII temp directory
struct TmpDir {
    fs::path path;
    TmpDir()
    {
        path = fs::temp_directory_path() / ("test_zarr_" + std::to_string(getpid()));
        fs::create_directories(path);
    }
    ~TmpDir() { fs::remove_all(path); }
};

// Helper: check that expr throws std::runtime_error
#define EXPECT_THROW(expr)                                                   \
    do {                                                                      \
        bool caught = false;                                                  \
        try { expr; } catch (const std::runtime_error&) { caught = true; }   \
        if (!caught) {                                                        \
            std::fprintf(stderr, "  FAIL %s:%d: expected throw from %s\n",   \
                         __FILE__, __LINE__, #expr);                          \
            ++::vc_test::fail_count();                                        \
        }                                                                     \
    } while (0)

// Helper: create a blosc-compressed uint8 dataset in a temp group
static std::unique_ptr<Dataset> makeDataset(
    const Group& grp, const std::string& name,
    const std::string& dtype, const ShapeType& shape, const ShapeType& chunks,
    const std::string& dimSep = ".")
{
    nlohmann::json opts = {{"cname", "zstd"}, {"clevel", 1}, {"shuffle", 0}};
    return Dataset::create(grp, name, dtype, shape, chunks, "blosc", opts, 0, dimSep);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Datatype suite
// ─────────────────────────────────────────────────────────────────────────────

TEST(Datatype, FromZarrRoundTrip)
{
    EXPECT_EQ(dtypeFromZarr("|u1"), Datatype::uint8);
    EXPECT_EQ(dtypeFromZarr("<u2"), Datatype::uint16);
    EXPECT_EQ(dtypeFromZarr("<f4"), Datatype::float32);

    EXPECT_EQ(dtypeToZarr(Datatype::uint8), "|u1");
    EXPECT_EQ(dtypeToZarr(Datatype::uint16), "<u2");
    EXPECT_EQ(dtypeToZarr(Datatype::float32), "<f4");
}

TEST(Datatype, FromNameRoundTrip)
{
    EXPECT_EQ(dtypeFromName("uint8"), Datatype::uint8);
    EXPECT_EQ(dtypeFromName("uint16"), Datatype::uint16);
    EXPECT_EQ(dtypeFromName("float32"), Datatype::float32);

    EXPECT_EQ(dtypeToName(Datatype::uint8), "uint8");
    EXPECT_EQ(dtypeToName(Datatype::uint16), "uint16");
    EXPECT_EQ(dtypeToName(Datatype::float32), "float32");
}

TEST(Datatype, Size)
{
    EXPECT_EQ(dtypeSize(Datatype::uint8), 1u);
    EXPECT_EQ(dtypeSize(Datatype::uint16), 2u);
    EXPECT_EQ(dtypeSize(Datatype::float32), 4u);
}

TEST(Datatype, FromZarrInvalid)
{
    EXPECT_THROW(dtypeFromZarr("<f8"));
    EXPECT_THROW(dtypeFromZarr(""));
}

TEST(Datatype, FromNameInvalid)
{
    EXPECT_THROW(dtypeFromName("int32"));
    EXPECT_THROW(dtypeFromName(""));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Group suite
// ─────────────────────────────────────────────────────────────────────────────

TEST(Group, Create)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");
    EXPECT_TRUE(grp.exists());
    EXPECT_TRUE(fs::exists(grp.path() / ".zgroup"));
}

TEST(Group, CreateChild)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");
    auto child = grp.createGroup("sub");
    EXPECT_TRUE(child.exists());
    EXPECT_TRUE(fs::exists(child.path() / ".zgroup"));
    EXPECT_EQ(child.path(), grp.path() / "sub");
}

TEST(Group, Keys)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");
    grp.createGroup("aaa");
    grp.createGroup("bbb");
    // Create a regular file — should not appear in keys()
    std::ofstream(grp.path() / "somefile.txt") << "hi";

    auto k = grp.keys();
    std::sort(k.begin(), k.end());
    ASSERT_EQ(k.size(), 2u);
    EXPECT_EQ(k[0], "aaa");
    EXPECT_EQ(k[1], "bbb");
}

TEST(Group, ReadWriteAttrs)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");

    // Initially empty
    auto a1 = grp.readAttrs();
    EXPECT_TRUE(a1.empty());

    // Write and read back
    grp.writeAttrs({{"foo", 42}, {"bar", "hello"}});
    auto a2 = grp.readAttrs();
    EXPECT_EQ(a2["foo"], 42);
    EXPECT_EQ(a2["bar"], "hello");

    // Merge behavior: writing again merges, doesn't overwrite unrelated keys
    grp.writeAttrs({{"baz", true}});
    auto a3 = grp.readAttrs();
    EXPECT_EQ(a3["foo"], 42);
    EXPECT_EQ(a3["baz"], true);
}

TEST(Group, NonExistent)
{
    TmpDir tmp;
    Group grp(tmp.path / "does_not_exist");
    EXPECT_FALSE(grp.exists());
    auto k = grp.keys();
    EXPECT_TRUE(k.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
//  DatasetMeta suite
// ─────────────────────────────────────────────────────────────────────────────

TEST(DatasetMeta, RoundTrip)
{
    DatasetMeta m;
    m.shape = {100, 200, 300};
    m.chunks = {32, 32, 32};
    m.dtype = Datatype::uint16;
    m.dimSeparator = "/";
    m.bloscCodec = "zstd";
    m.bloscLevel = 3;
    m.bloscShuffle = 1;
    m.fillValue = 42;

    auto j = m.toJson();
    DatasetMeta m2;
    m2.fromJson(j);

    EXPECT_EQ(m2.shape, m.shape);
    EXPECT_EQ(m2.chunks, m.chunks);
    EXPECT_EQ(m2.dtype, m.dtype);
    EXPECT_EQ(m2.dimSeparator, m.dimSeparator);
    EXPECT_EQ(m2.bloscCodec, m.bloscCodec);
    EXPECT_EQ(m2.bloscLevel, m.bloscLevel);
    EXPECT_EQ(m2.bloscShuffle, m.bloscShuffle);
    EXPECT_FLOAT_EQ(m2.fillValue, m.fillValue);
}

TEST(DatasetMeta, FillValueVariants)
{
    // null fill_value → 0
    {
        nlohmann::json j = {
            {"zarr_format", 2}, {"dtype", "|u1"}, {"shape", {10}},
            {"chunks", {10}}, {"fill_value", nullptr}, {"order", "C"},
            {"filters", nullptr}, {"compressor", nullptr}};
        DatasetMeta m;
        m.fromJson(j);
        EXPECT_FLOAT_EQ(m.fillValue, 0.0);
    }
    // numeric fill_value
    {
        nlohmann::json j = {
            {"zarr_format", 2}, {"dtype", "<f4"}, {"shape", {10}},
            {"chunks", {10}}, {"fill_value", 3.14}, {"order", "C"},
            {"filters", nullptr}, {"compressor", nullptr}};
        DatasetMeta m;
        m.fromJson(j);
        EXPECT_NEAR(m.fillValue, 3.14, 1e-6);
    }
    // NaN string
    {
        nlohmann::json j = {
            {"zarr_format", 2}, {"dtype", "<f4"}, {"shape", {10}},
            {"chunks", {10}}, {"fill_value", "NaN"}, {"order", "C"},
            {"filters", nullptr}, {"compressor", nullptr}};
        DatasetMeta m;
        m.fromJson(j);
        EXPECT_TRUE(std::isnan(m.fillValue));
    }
    // Infinity string
    {
        nlohmann::json j = {
            {"zarr_format", 2}, {"dtype", "<f4"}, {"shape", {10}},
            {"chunks", {10}}, {"fill_value", "Infinity"}, {"order", "C"},
            {"filters", nullptr}, {"compressor", nullptr}};
        DatasetMeta m;
        m.fromJson(j);
        EXPECT_TRUE(std::isinf(m.fillValue));
        EXPECT_TRUE(m.fillValue > 0);
    }
}

TEST(DatasetMeta, UnsupportedCompressor)
{
    nlohmann::json j = {
        {"zarr_format", 2}, {"dtype", "|u1"}, {"shape", {10}},
        {"chunks", {10}}, {"fill_value", 0}, {"order", "C"},
        {"filters", nullptr},
        {"compressor", {{"id", "lz4"}}}};
    DatasetMeta m;
    EXPECT_THROW(m.fromJson(j));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Dataset suite
// ─────────────────────────────────────────────────────────────────────────────

TEST(Dataset, CreateAndOpen)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");
    auto ds = makeDataset(grp, "data", "uint16", {64, 64, 64}, {32, 32, 32});

    EXPECT_EQ(ds->shape(), (ShapeType{64, 64, 64}));
    EXPECT_EQ(ds->defaultChunkShape(), (ShapeType{32, 32, 32}));
    EXPECT_EQ(ds->defaultChunkSize(), 32u * 32u * 32u);
    EXPECT_EQ(ds->getDtype(), Datatype::uint16);
    EXPECT_EQ(ds->dimension(), 3u);

    // Re-open by path
    auto ds2 = Dataset::open(ds->path());
    EXPECT_EQ(ds2->shape(), ds->shape());
    EXPECT_EQ(ds2->getDtype(), ds->getDtype());

    // Re-open via parent group + name
    auto ds3 = Dataset::open(grp, "data");
    EXPECT_EQ(ds3->shape(), ds->shape());
}

TEST(Dataset, ChunkPathDot)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");
    auto ds = makeDataset(grp, "data", "uint8", {8, 8, 8}, {4, 4, 4}, ".");

    fs::path p;
    ds->chunkPath({0, 1, 2}, p);
    EXPECT_EQ(p.filename().string(), "0.1.2");
}

TEST(Dataset, ChunkPathSlash)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");
    auto ds = makeDataset(grp, "data", "uint8", {8, 8, 8}, {4, 4, 4}, "/");

    fs::path p;
    ds->chunkPath({0, 1, 2}, p);
    // With "/" separator the path ends in .../0/1/2
    EXPECT_EQ(p.filename().string(), "2");
    EXPECT_EQ(p.parent_path().filename().string(), "1");
    EXPECT_EQ(p.parent_path().parent_path().filename().string(), "0");
}

TEST(Dataset, WriteReadChunk)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");
    auto ds = makeDataset(grp, "data", "uint8", {16, 16}, {8, 8});

    const std::size_t n = ds->defaultChunkSize();
    std::vector<uint8_t> writeData(n);
    for (std::size_t i = 0; i < n; ++i)
        writeData[i] = static_cast<uint8_t>(i & 0xFF);

    ShapeType chunkId = {0, 0};
    EXPECT_FALSE(ds->chunkExists(chunkId));
    ds->writeChunk(chunkId, writeData.data());
    EXPECT_TRUE(ds->chunkExists(chunkId));

    std::vector<uint8_t> readData(n, 0);
    ds->readChunk(chunkId, readData.data());
    EXPECT_EQ(readData, writeData);
}

TEST(Dataset, CompressDecompress)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");
    auto ds = makeDataset(grp, "data", "uint16", {32}, {32});

    std::vector<uint16_t> original(32);
    for (int i = 0; i < 32; ++i)
        original[i] = static_cast<uint16_t>(i * 100);

    std::vector<char> compressed;
    ds->compress(original.data(), compressed, original.size());
    EXPECT_TRUE(!compressed.empty());

    std::vector<uint16_t> decompressed(32, 0);
    ds->decompress(compressed, decompressed.data(), decompressed.size());
    EXPECT_EQ(decompressed, original);
}

TEST(Dataset, CheckRequestShape)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");
    auto ds = makeDataset(grp, "data", "uint8", {64, 64}, {32, 32});

    // Valid request — should not throw
    ds->checkRequestShape({0, 0}, {64, 64});
    ds->checkRequestShape({32, 32}, {32, 32});

    // Out of bounds
    EXPECT_THROW(ds->checkRequestShape({0, 0}, {65, 64}));
    EXPECT_THROW(ds->checkRequestShape({33, 0}, {32, 64}));

    // Wrong dimension count
    EXPECT_THROW(ds->checkRequestShape({0}, {64}));

    // Zero shape
    EXPECT_THROW(ds->checkRequestShape({0, 0}, {0, 64}));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Subarray suite
// ─────────────────────────────────────────────────────────────────────────────

TEST(Subarray, WriteReadSingleChunk3D)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");
    // Single chunk: exercises 3D subarray template without multi-chunk stride bug
    auto ds = makeDataset(grp, "data", "uint8", {4, 4, 4}, {4, 4, 4});

    xt::xarray<uint8_t> writeArr = xt::xarray<uint8_t>::from_shape({4, 4, 4});
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            for (std::size_t k = 0; k < 4; ++k)
                writeArr(i, j, k) = static_cast<uint8_t>(i * 16 + j * 4 + k);

    ShapeType offset = {0, 0, 0};
    writeSubarray<uint8_t>(*ds, writeArr, offset.begin());

    xt::xarray<uint8_t> readArr = xt::xarray<uint8_t>::from_shape({4, 4, 4});
    readSubarray<uint8_t>(*ds, readArr, offset.begin());
    EXPECT_EQ(readArr, writeArr);
}

TEST(Subarray, WriteReadFullRoundTrip2D)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");
    auto ds = makeDataset(grp, "data", "uint16", {16, 16}, {8, 8});

    // Write full array spanning multiple chunks in one call
    xt::xarray<uint16_t> writeArr = xt::xarray<uint16_t>::from_shape({16, 16});
    for (std::size_t i = 0; i < 16; ++i)
        for (std::size_t j = 0; j < 16; ++j)
            writeArr(i, j) = static_cast<uint16_t>(i * 16 + j);

    ShapeType offset = {0, 0};
    writeSubarray<uint16_t>(*ds, writeArr, offset.begin());

    // Read full array back in one call
    xt::xarray<uint16_t> readArr = xt::xarray<uint16_t>::from_shape({16, 16});
    readSubarray<uint16_t>(*ds, readArr, offset.begin());

    EXPECT_EQ(readArr, writeArr);
}

TEST(Subarray, WriteReadFullRoundTrip3D)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");
    auto ds = makeDataset(grp, "data", "uint8", {8, 8, 8}, {4, 4, 4});

    xt::xarray<uint8_t> writeArr = xt::xarray<uint8_t>::from_shape({8, 8, 8});
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            for (std::size_t k = 0; k < 8; ++k)
                writeArr(i, j, k) = static_cast<uint8_t>((i * 64 + j * 8 + k) & 0xFF);

    ShapeType offset = {0, 0, 0};
    writeSubarray<uint8_t>(*ds, writeArr, offset.begin());

    xt::xarray<uint8_t> readArr = xt::xarray<uint8_t>::from_shape({8, 8, 8});
    readSubarray<uint8_t>(*ds, readArr, offset.begin());

    EXPECT_EQ(readArr, writeArr);
}

TEST(Subarray, PartialRead)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");
    auto ds = makeDataset(grp, "data", "uint16", {16, 16}, {8, 8});

    // Fill entire dataset with pattern
    xt::xarray<uint16_t> full = xt::xarray<uint16_t>::from_shape({16, 16});
    for (std::size_t i = 0; i < 16; ++i)
        for (std::size_t j = 0; j < 16; ++j)
            full(i, j) = static_cast<uint16_t>(i * 16 + j);

    ShapeType fullOff = {0, 0};
    writeSubarray<uint16_t>(*ds, full, fullOff.begin());

    // Read a 4x4 sub-region starting at (2, 3)
    xt::xarray<uint16_t> sub = xt::xarray<uint16_t>::from_shape({4, 4});
    ShapeType subOff = {2, 3};
    readSubarray<uint16_t>(*ds, sub, subOff.begin());

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_EQ(sub(i, j), full(i + 2, j + 3));
}

TEST(Subarray, EdgeChunks)
{
    TmpDir tmp;
    auto grp = Group::create(tmp.path / "root");
    // Shape 10 not divisible by chunk size 4 → edge chunks
    auto ds = makeDataset(grp, "data", "float32", {10, 10}, {4, 4});

    xt::xarray<float> writeArr = xt::xarray<float>::from_shape({10, 10});
    for (std::size_t i = 0; i < 10; ++i)
        for (std::size_t j = 0; j < 10; ++j)
            writeArr(i, j) = static_cast<float>(i * 10 + j);

    ShapeType offset = {0, 0};
    writeSubarray<float>(*ds, writeArr, offset.begin());

    // Read the edge region at the bottom-right: rows 8-9, cols 8-9
    xt::xarray<float> edge = xt::xarray<float>::from_shape({2, 2});
    ShapeType edgeOff = {8, 8};
    readSubarray<float>(*ds, edge, edgeOff.begin());

    EXPECT_FLOAT_EQ(edge(0, 0), 88.0f);
    EXPECT_FLOAT_EQ(edge(0, 1), 89.0f);
    EXPECT_FLOAT_EQ(edge(1, 0), 98.0f);
    EXPECT_FLOAT_EQ(edge(1, 1), 99.0f);
}
