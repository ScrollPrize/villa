// Exercise normal-field Zarr publication and both visualization consumers via the CLI.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <vc_reader/zarr.hpp>

#include <QProcess>
#include <QProcessEnvironment>
#include <QTemporaryDir>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace {
void run(QStringList args) {
    const auto* binary = std::getenv("VC_NGRIDS_BIN");
    REQUIRE(binary);
    QProcess process;
    auto env = QProcessEnvironment::systemEnvironment();
    env.insert("OMP_NUM_THREADS", "2");
    process.setProcessEnvironment(env);
    process.start(QString::fromLocal8Bit(binary), args);
    REQUIRE(process.waitForStarted());
    REQUIRE(process.waitForFinished(60000));
    INFO(process.readAllStandardError().toStdString());
    REQUIRE(process.exitStatus() == QProcess::NormalExit);
    REQUIRE(process.exitCode() == 0);
}
QString path(const fs::path& p) { return QString::fromStdString(p.string()); }
std::string text(const fs::path& p) {
    std::ostringstream out;
    out << std::ifstream(p).rdbuf();
    return out.str();
}
}

TEST_CASE("normal-field visualization reads separators, fill and physical coordinates") {
    QTemporaryDir temp;
    REQUIRE(temp.isValid());
    const fs::path root = temp.path().toStdString();
    std::string reference;
    for (const auto* separator : {".", "/"}) {
        const auto field = root / (std::string(separator) == "." ? "dot" : "slash");
        vcr::zarr::CreateOptions options;
        options.shape = {3, 4, 5}; options.chunks = {2, 2, 2};
        options.fillValue = 128; options.separator = separator;
        options.compressor = "gzip";
        for (const auto* axis : {"x", "y", "z"}) {
            auto writer = vcr::zarr::Writer::create(field / axis / "0", options);
            std::vector<uint8_t> chunk(8, axis[0] == 'x' ? 255 : 128);
            writer.writeChunk({0, 0, 0}, chunk.data(), chunk.size());
        }
        vcr::zarr::writeAttributes(field, {{"grid_origin_xyz", {10, 20, 30}}, {"sample_step", 2}});
        const auto ply = field / "normals.ply";
        run({"--input", path(field), "--vis-ply", path(ply)});
        const auto result = text(ply);
        CHECK(result.find("element vertex 00000000000000000016") != std::string::npos);
        CHECK(result.find("element edge 00000000000000000008") != std::string::npos);
        CHECK(result.find("10 20 30 255 255 0\n11 20 30 255 255 0") != std::string::npos);
        if (reference.empty()) reference = result;
        else CHECK(result == reference);
        const auto cropped = field / "crop.ply";
        run({"--input", path(field), "--vis-ply", path(cropped), "--crop", "10", "20", "30", "11", "21", "31"});
        CHECK(text(cropped).find("element edge 00000000000000000001") != std::string::npos);

        // Every interpolation footprint lies inside the one populated 2^3 block.
        cv::Mat_<cv::Vec3f> points(2, 2);
        points(0, 0) = {10.25f, 20.25f, 30.5f};
        points(0, 1) = {11.25f, 20.25f, 30.5f};
        points(1, 0) = {10.25f, 21.25f, 30.5f};
        points(1, 1) = {11.25f, 21.25f, 30.5f};
        const auto surface = field / "surface";
        QuadSurface(points, {1, 1}).save(surface, true);
        const auto onSurface = field / "on-surface.ply";
        run({"--input", path(field), "--vis-ply", path(onSurface), "--surf", path(surface)});
        CHECK(text(onSurface).find("element edge 00000000000000000000") == std::string::npos);
    }
}

TEST_CASE("fitted normals publish seven arrays with distinct fill values") {
    QTemporaryDir temp;
    REQUIRE(temp.isValid());
    const fs::path root = temp.path().toStdString(), grids = root / "grids", output = root / "normals";
    fs::create_directories(grids);
    std::ofstream(grids / "metadata.json") << R"({"sparse-volume":8})";
    for (const auto* plane : {"xy", "xz", "yz"}) {
        fs::create_directories(grids / plane);
        for (int slice = 0; slice < 65; slice += 8) {
            vc::core::util::GridStore grid(cv::Rect(0, 0, 65, 65), 8);
            if (std::string(plane) != "yz") {
                std::vector<cv::Point> line;
                for (int i = 0; i < 65; ++i) line.emplace_back(32, i);
                grid.add(line);
            }
            char filename[32];
            std::snprintf(filename, sizeof(filename), "%06d.grid", slice);
            grid.save((grids / plane / filename).string());
        }
    }
    run({"--input", path(grids), "--fit-normals", "--output-zarr", path(output),
         "--step", "16", "--crop", "16", "16", "16", "49", "49", "49"});
    for (const auto* name : {"x", "y", "z", "fit_rms", "fit_frac_short_paths", "fit_used_radius", "fit_segment_count"}) {
        const bool normal = std::string(name).size() == 1;
        auto opened = vcr::zarr::openArray((output / name / "0").string(), {});
        CHECK(opened.meta.shape == vcr::Index({5, 5, 5}));
        CHECK(opened.meta.separator == "/");
        CHECK(opened.meta.fill == std::vector<uint8_t>({uint8_t(normal ? 128 : 0)}));
        std::vector<uint8_t> values(125);
        opened.array->read_many({{0, 0, 0}}, {{5, 5, 5}}, {values.data()});
        CHECK(values.front() == (normal ? 128 : 0));
        CHECK(values.back() == (normal ? 128 : 0));
        if (std::string(name) == "x")
            CHECK(std::count(values.begin(), values.end(), uint8_t(128)) < 125);
    }
    const auto attrs = vcr::zarr::readAttributes(output);
    CHECK(attrs["sample_step"] == 16);
    CHECK(attrs["encoding"]["fill_value"] == 128);
    CHECK(attrs["fit_rms_group"] == "fit_rms/0");
    CHECK_FALSE(fs::exists(output / ".vc_ngrids_fit_spool"));
}
