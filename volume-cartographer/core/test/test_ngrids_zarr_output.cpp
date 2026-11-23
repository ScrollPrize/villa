// Exercise normal-field Zarr publication and both visualization consumers via the CLI.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <vc_reader/array.hpp>

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
