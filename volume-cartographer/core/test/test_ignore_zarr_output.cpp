// Exercise ignore-label editing, sparse publication, reuse and pyramid reads via the CLI.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vc_reader/zarr.hpp>
#include <QProcess>
#include <QTemporaryDir>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

namespace fs = std::filesystem;
namespace {
QString path(const fs::path& p) { return QString::fromStdString(p.string()); }

int run(const QStringList& args) {
    const auto* binary = std::getenv("VC_ADD_IGNORE_LABEL_BIN");
    REQUIRE(binary);
    QProcess process;
    process.start(QString::fromLocal8Bit(binary), args);
    REQUIRE(process.waitForStarted());
    REQUIRE(process.waitForFinished(60000));
    const auto error = process.readAllStandardError().toStdString();
    if (process.exitCode() != 0) std::cerr << error;
    REQUIRE(process.exitStatus() == QProcess::NormalExit);
    return process.exitCode();
}

std::size_t elements(const vcr::Index& shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>());
}

std::vector<uint8_t> pixels(const fs::path& p) {
    auto opened = vcr::zarr::openArray(p.string(), {.threads = 1});
    std::vector<uint8_t> data(elements(opened.meta.shape));
    opened.array->read_many({{0, 0, 0}}, {opened.meta.shape}, {data.data()});
    return data;
}

struct Fixture {
    QTemporaryDir temp;
    fs::path root, input;
    std::vector<uint8_t> foreground;

    explicit Fixture(const std::string& compressor) {
        REQUIRE(temp.isValid());
        root = temp.path().toStdString(); input = root / "input";
        vcr::Index shape{9, 33, 35};
        std::vector<uint8_t> data(elements(shape), 0);
        for (int z = 1; z < 8; ++z)
            for (int y = 0; y < 33; ++y)
                for (int x = 0; x < 35; ++x) {
                    const int r2 = (y - 16) * (y - 16) + (x - 17) * (x - 17);
                    if (r2 >= 64 && r2 <= 100) data[(z * 33 + y) * 35 + x] = 1;
                }
        foreground = data;
        for (int level = 0; level < 4; ++level) {
            vcr::zarr::CreateOptions options;
            options.shape = shape; options.chunks = {4, 16, 16}; options.compressor = compressor;
            auto w = vcr::zarr::Writer::create(input / std::to_string(level), options);
            w.writeRegion({0, 0, 0}, shape, data.data());
            if (level == 0) w.removeChunk({2, 2, 2});  // absent partial edge chunk
            vcr::zarr::writeAttributes(w.path(), {{"channel", "labels"}});
            vcr::Index nextShape{(shape[0]+1)/2, (shape[1]+1)/2, (shape[2]+1)/2};
            std::vector<uint8_t> next(elements(nextShape), 0);
            for (int64_t z = 0; z < shape[0]; ++z)
                for (int64_t y = 0; y < shape[1]; ++y)
                    for (int64_t x = 0; x < shape[2]; ++x)
                        if (data[(z*shape[1]+y)*shape[2]+x])
                            next[((z/2)*nextShape[1]+y/2)*nextShape[2]+x/2] = 1;
            shape = nextShape; data = std::move(next);
        }
        std::ofstream(input / "meta.json") << R"({"label_volume":"rasterized","voxelsize":7.91})";
        vcr::zarr::writeAttributes(input, {{"source", "ignore-label-test"}});
    }

    QStringList args(const fs::path& output) const {
        return {"--input", path(input), "--output", path(output), "--compute-level", "0",
                "--workers", "3", "--z-min", "1", "--z-max", "8"};
    }

    std::vector<std::vector<uint8_t>> check(const fs::path& output) const {
        std::vector<std::vector<uint8_t>> result;
        for (int level = 0; level < 4; ++level) {
            const auto node = output / std::to_string(level);
            auto opened = vcr::zarr::openArray(node.string(), {.threads = 1});
            CHECK(opened.meta.fill == std::vector<uint8_t>{2});
            CHECK(vcr::zarr::readAttributes(node)["channel"] == "labels");
            result.push_back(pixels(node));
        }
        REQUIRE(result[0].size() == foreground.size());
        for (std::size_t i = 0; i < foreground.size(); ++i)
            if (foreground[i]) CHECK(result[0][i] == 1);
        CHECK(std::count(result[0].begin(), result[0].end(), uint8_t(2)) > 0);
        CHECK(vcr::zarr::readAttributes(output)["source"] == "ignore-label-test");
        auto w = vcr::zarr::Writer::open(output / "0");
        CHECK_FALSE(w.chunkExists({2, 2, 2}));
        return result;
    }
};
}

TEST_CASE("ignore labels preserve output across streaming, preloading, slabs and reuse") {
    for (const auto* codec : {"none", "gzip"}) {
        Fixture f(codec);
        std::vector<std::vector<uint8_t>> reference;
        for (const auto* mode : {"stream", "preload", "slab"}) {
            const auto output = f.root / mode;
            auto args = f.args(output);
            args.append({"--mode", "legacy-slice", "--algo-mode", "hull", "--map-mode", "exact",
                         "--cache-mode", std::string(mode) == "preload" ? "preload" : "stream"});
            if (std::string(mode) == "slab") args.append("--chunk-z-slab");
            REQUIRE(run(args) == 0);
            const auto result = f.check(output);
            if (reference.empty()) reference = result;
            else CHECK(result == reference);
            REQUIRE(run(args + QStringList{"--reuse-output-tree"}) == 0);
            CHECK(f.check(output) == result);
        }
        const auto output = f.root / "stream", vis = f.root / "vis";
        auto args = f.args(output);
        args.append({"--mode", "legacy-slice", "--visualize", "4", "--visualize-dir", path(vis)});
        REQUIRE(run(args) == 0);
        const auto image = cv::imread((vis / "slice_4_output_ignore.png").string(), cv::IMREAD_GRAYSCALE);
        REQUIRE(image.rows == 33);
        REQUIRE(image.cols == 35);
        CHECK(std::equal(image.ptr<uint8_t>(), image.ptr<uint8_t>() + image.total(), reference[0].begin() + 4*33*35));

        // Reuse must reject incompatible geometry before modifying any data.
        vcr::zarr::CreateOptions wrong;
        wrong.shape = {1, 2, 3}; wrong.chunks = {4, 16, 16}; wrong.compressor = codec;
        vcr::zarr::Writer::create(output / "1", wrong);
        args = f.args(output);
        args.append({"--mode", "legacy-slice", "--reuse-output-tree"});
        CHECK(run(args) != 0);
        CHECK(pixels(output / "0") == reference[0]);
    }
}

TEST_CASE("ignore-label built-in geometry and sparse-edit regressions") {
    REQUIRE(run({"--self-test"}) == 0);
}

TEST_CASE("chunk alpha wrap publishes and overwrites a sparse label pyramid") {
    Fixture f("gzip");
    const auto output = f.root / "wrapped";
    auto args = f.args(output);
    args.append({"--mode", "chunk-alpha-wrap", "--chunk-alpha", "2", "--skip-inner"});
    REQUIRE(run(args) == 0);
    const auto result = f.check(output);
    REQUIRE(run(args + QStringList{"--overwrite"}) == 0);
    CHECK(f.check(output) == result);
}
