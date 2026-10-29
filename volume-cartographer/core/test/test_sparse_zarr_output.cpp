// CLI workflow: rasterize a surface, publish sparse chunks, and build the label pyramid.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"
#include <vc_reader/zarr.hpp>

#include <QProcess>
#include <QTemporaryDir>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>

namespace fs = std::filesystem;
using vcr::zarr::Json;

namespace {
struct Fixture {
    QTemporaryDir temp;
    fs::path root = temp.path().toStdString();
    fs::path surface = root / "surface";
    Fixture() {
        REQUIRE(temp.isValid());
        cv::Mat_<cv::Vec3f> points(2, 2);
        points(0, 0) = {3, 4, 7};
        points(0, 1) = {22, 4, 7};
        points(1, 0) = {3, 20, 7};
        points(1, 1) = {22, 20, 7};
        QuadSurface(points, {1, 1}).save(surface, true);
    }
    int run(const fs::path& output, QStringList extra = {}) {
        const auto* binary = std::getenv("VC_TIFXYZ2ZARR_SPARSE_BIN");
        REQUIRE(binary);
        QProcess process;
        QStringList args{QString::fromStdString(surface.string()), QString::fromStdString(output.string()),
                         "--chunk-size", "8", "--threads", "3"};
        args.append(extra);
        process.start(QString::fromLocal8Bit(binary), args);
        REQUIRE(process.waitForStarted());
        REQUIRE(process.waitForFinished(60000));
        INFO(process.readAllStandardError().toStdString());
        REQUIRE(process.exitStatus() == QProcess::NormalExit);
        return process.exitCode();
    }
};

std::vector<uint8_t> pixels(vcr::zarr::OpenedArray& opened) {
    const auto& shape = opened.meta.shape;
    std::vector<uint8_t> values(std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>()));
    opened.array->read_many({{0, 0, 0}}, {shape}, {values.data()});
    return values;
}

void checkPyramid(const fs::path& output) {
    auto shape = vcr::Index{19, 21, 23};
    std::vector<uint8_t> expected(19 * 21 * 23, 0);
    for (int y = 4; y <= 20; ++y)
        for (int x = 3; x <= 22; ++x) expected[(7 * 21 + y) * 23 + x] = 1;
    for (int level = 0; level <= 5; ++level) {
        auto opened = vcr::zarr::openArray((output / std::to_string(level)).string(), {});
        CHECK(opened.meta.shape == shape);
        CHECK(opened.meta.dtype == "|u1");
        CHECK(pixels(opened) == expected);
        auto writer = vcr::zarr::Writer::open(output / std::to_string(level));
        if (level == 0) {
            CHECK_FALSE(writer.chunkExists({1, 0, 0}));
            CHECK(writer.chunkExists({0, 2, 2}));  // partial Y/X edge
        }
        vcr::Index nextShape{(shape[0] + 1) / 2, (shape[1] + 1) / 2, (shape[2] + 1) / 2};
        std::vector<uint8_t> next(nextShape[0] * nextShape[1] * nextShape[2], 0);
        for (int64_t z = 0; z < shape[0]; ++z)
            for (int64_t y = 0; y < shape[1]; ++y)
                for (int64_t x = 0; x < shape[2]; ++x)
                    if (expected[(z * shape[1] + y) * shape[2] + x])
                        next[((z / 2) * nextShape[1] + y / 2) * nextShape[2] + x / 2] = 255;
        shape = nextShape;
        expected = std::move(next);
    }
    const auto attrs = vcr::zarr::readAttributes(output);
    CHECK(attrs["multiscales"][0]["datasets"].size() == 6);
    std::ifstream in(output / "meta.json");
    const auto meta = Json::parse(in);
    CHECK(meta["uuid"] == output.filename().string());
    CHECK(meta["label_volume"] == "rasterized");
}
}

TEST_CASE("sparse rasterizer publishes identical explicit and reference shapes") {
    Fixture f;
    const auto explicitOutput = f.root / "explicit";
    REQUIRE(f.run(explicitOutput, {"--shape-z", "19", "--shape-y", "21", "--shape-x", "23"}) == 0);
    checkPyramid(explicitOutput);
    vcr::zarr::CreateOptions options;
    options.shape = {19, 21, 23}; options.chunks = {8, 8, 8};
    const auto reference = f.root / "reference";
    vcr::zarr::Writer::create(reference / "0", options);
    for (const auto& path : {reference, reference / "0"}) {
        const auto output = f.root / "from-reference";
        REQUIRE(f.run(output, {"--reference-zarr", QString::fromStdString(path.string()), "--overwrite"}) == 0);
        checkPyramid(output);
    }
    // Clipping away the surface must publish metadata without materializing empty chunks.
    const auto empty = f.root / "empty";
    REQUIRE(f.run(empty, {"--reference-zarr", QString::fromStdString(reference.string()), "--z-min", "8"}) == 0);
    for (int level = 0; level <= 5; ++level) {
        auto opened = vcr::zarr::openArray((empty / std::to_string(level)).string(), {});
        const auto values = pixels(opened);
        CHECK(std::all_of(values.begin(), values.end(), [](auto v) { return v == 0; }));
        for (const auto& entry : fs::directory_iterator(empty / std::to_string(level)))
            CHECK(entry.path().filename().string().front() == '.');
    }
    for (const auto& entry : fs::directory_iterator(f.root))
        CHECK(entry.path().filename().string().find(".partial-") == std::string::npos);
}

TEST_CASE("sparse rasterizer accepts v3 references and rejects non-volume ranks") {
    Fixture f;
    const auto reference = f.root / "reference";
    fs::create_directories(reference / "0");
    Json metadata{{"zarr_format", 3}, {"node_type", "array"}, {"data_type", "uint8"},
                  {"shape", {19, 21, 23}}, {"fill_value", 0},
                  {"chunk_grid", {{"name", "regular"}, {"configuration", {{"chunk_shape", {8, 8, 8}}}}}},
                  {"chunk_key_encoding", {{"name", "default"}, {"configuration", {{"separator", "/"}}}}},
                  {"codecs", Json::array({Json{{"name", "bytes"}}})}};
    std::ofstream(reference / "0" / "zarr.json") << metadata;
    const auto output = f.root / "v3";
    REQUIRE(f.run(output, {"--reference-zarr", QString::fromStdString(reference.string())}) == 0);
    checkPyramid(output);
    metadata["shape"] = {1, 19, 21, 23};
    metadata["chunk_grid"]["configuration"]["chunk_shape"] = {1, 8, 8, 8};
    std::ofstream(reference / "0" / "zarr.json") << metadata;
    CHECK(f.run(f.root / "invalid", {"--reference-zarr", QString::fromStdString(reference.string())}) != 0);
    CHECK_FALSE(fs::exists(f.root / "invalid"));
}
