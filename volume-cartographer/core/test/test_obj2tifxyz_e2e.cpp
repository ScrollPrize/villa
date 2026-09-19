#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

constexpr int kGrid = 41;
constexpr double kSpacing = 10.0;

fs::path locateBinary(const fs::path& candidate)
{
    if (fs::exists(candidate) && fs::is_regular_file(candidate)) return candidate;
    return {};
}

fs::path findVcObj2Tifxyz()
{
    if (const char* env = std::getenv("VC_OBJ2TIFXYZ_BIN")) {
        if (auto p = locateBinary(env); !p.empty()) return p;
    }
    for (const fs::path& base : {fs::path("build/bin"),
                                 fs::path("build-macos/bin"),
                                 fs::path("build-macos-rel/bin")}) {
        if (auto p = locateBinary(base / "vc_obj2tifxyz"); !p.empty()) return p;
    }
    if (const char* path = std::getenv("PATH")) {
        std::string s = path;
        std::string::size_type from = 0;
        while (from <= s.size()) {
            auto next = s.find(':', from);
            std::string seg = s.substr(from, next == std::string::npos ? std::string::npos
                                                                       : next - from);
            if (!seg.empty()) {
                if (auto p = locateBinary(fs::path(seg) / "vc_obj2tifxyz"); !p.empty()) return p;
            }
            if (next == std::string::npos) break;
            from = next + 1;
        }
    }
    return {};
}

void writePlaneObj(const fs::path& obj, double uvStep)
{
    std::ofstream f(obj);
    f.precision(9);
    for (int r = 0; r < kGrid; ++r) {
        for (int c = 0; c < kGrid; ++c) {
            f << "v " << c * kSpacing << ' ' << r * kSpacing << " 100\n";
        }
    }
    for (int r = 0; r < kGrid; ++r) {
        for (int c = 0; c < kGrid; ++c) {
            f << "vt " << c * uvStep << ' ' << r * uvStep << '\n';
        }
    }
    auto idx = [](int r, int c) { return r * kGrid + c + 1; };
    for (int r = 0; r + 1 < kGrid; ++r) {
        for (int c = 0; c + 1 < kGrid; ++c) {
            const int a = idx(r, c), b = idx(r, c + 1), d = idx(r + 1, c), e = idx(r + 1, c + 1);
            f << "f " << a << '/' << a << ' ' << b << '/' << b << ' ' << e << '/' << e << '\n';
            f << "f " << a << '/' << a << ' ' << e << '/' << e << ' ' << d << '/' << d << '\n';
        }
    }
}

void writeDiamondObj(const fs::path& obj)
{
    std::ofstream f(obj);
    f << "v 50 0 100\nv 100 50 100\nv 50 100 100\nv 0 50 100\nv 50 50 100\n"
      << "vt 0.5 0\nvt 1 0.5\nvt 0.5 1\nvt 0 0.5\nvt 0.5 0.5\n"
      << "f 1/1 2/2 5/5\nf 2/2 3/3 5/5\nf 3/3 4/4 5/5\nf 4/4 1/1 5/5\n";
}

void writeSourceTifxyz(const fs::path& dir)
{
    cv::Mat_<cv::Vec3f> pts(kGrid, kGrid);
    for (int r = 0; r < kGrid; ++r) {
        for (int c = 0; c < kGrid; ++c) {
            pts(r, c) = cv::Vec3f(static_cast<float>(c * kSpacing),
                                  static_cast<float>(r * kSpacing), 100.f);
        }
    }
    const float s = static_cast<float>(1.0 / kSpacing);
    QuadSurface surf(pts, cv::Vec2f(s, s));
    surf.path = dir;
    surf.id = dir.filename().string();
    surf.save(dir.string(), surf.id, false);
}

int run(const fs::path& bin, const std::string& args, const fs::path& log)
{
    const std::string cmd = bin.string() + " " + args + " > " + log.string() + " 2>&1";
    return std::system(cmd.c_str());
}

nlohmann::json readMeta(const fs::path& dir)
{
    std::ifstream f(dir / "meta.json");
    REQUIRE_MESSAGE(f.good(), "meta.json missing under " << dir.string());
    nlohmann::json j;
    f >> j;
    return j;
}

void checkScale(const fs::path& dir, double expected)
{
    const auto meta = readMeta(dir);
    REQUIRE(meta.contains("scale"));
    CHECK(meta["scale"][0].get<double>() == doctest::Approx(expected).epsilon(0.02));
    CHECK(meta["scale"][1].get<double>() == doctest::Approx(expected).epsilon(0.02));
}

}

TEST_CASE("vc_obj2tifxyz writes a scale that describes the emitted grid")
{
    const char* runFlag = std::getenv("VC_RUN_E2E");
    if (!runFlag || std::string(runFlag) != "1") {
        MESSAGE("VC_RUN_E2E != 1; skipping (set VC_RUN_E2E=1 to enable)");
        return;
    }

    const fs::path bin = findVcObj2Tifxyz();
    REQUIRE_MESSAGE(!bin.empty(),
                    "vc_obj2tifxyz binary not found; build first or set VC_OBJ2TIFXYZ_BIN");

    std::random_device rd;
    std::mt19937_64 rng(rd());
    const fs::path root = fs::temp_directory_path() / ("vc_obj2tifxyz_e2e_" + std::to_string(rng()));
    fs::create_directories(root);
    const fs::path normalised = root / "normalised.obj";
    writePlaneObj(normalised, 1.0 / (kGrid - 1));
    const fs::path metric = root / "metric.obj";
    writePlaneObj(metric, kSpacing);
    const fs::path source = root / "source";
    writeSourceTifxyz(source);
    const std::string stretch = std::to_string(kGrid - 1);

    SUBCASE("default stretch factor rasterizes nothing and must fail")
    {
        const fs::path diamond = root / "diamond.obj";
        writeDiamondObj(diamond);
        const fs::path out = root / "default";
        const fs::path log = root / "default.log";
        const int rc = run(bin, diamond.string() + " " + out.string(), log);
        INFO("log: ", log.string());
        CHECK(rc != 0);
        CHECK_FALSE(fs::exists(out / "meta.json"));
    }

    SUBCASE("a 2 x 2 output grid is refused even when its corners rasterize")
    {
        const fs::path out = root / "degenerate";
        const fs::path log = root / "degenerate.log";
        INFO("log: ", log.string());
        CHECK(run(bin, normalised.string() + " " + out.string(), log) != 0);
        CHECK_FALSE(fs::exists(out / "meta.json"));
    }

    SUBCASE("normalised UVs: scale is the reciprocal of the measured spacing")
    {
        const fs::path out = root / "stretched";
        const fs::path log = root / "stretched.log";
        INFO("log: ", log.string());
        REQUIRE(run(bin, normalised.string() + " " + out.string() + " " + stretch, log) == 0);
        checkScale(out, 1.0 / kSpacing);
    }

    SUBCASE("grid size and scale follow stretch_factor")
    {
        for (int s : {20, 80}) {
            const fs::path out = root / ("stretch-" + std::to_string(s));
            const fs::path log = root / ("stretch-" + std::to_string(s) + ".log");
            INFO("stretch ", s, ", log: ", log.string());
            REQUIRE(run(bin, normalised.string() + " " + out.string() + " " + std::to_string(s), log) == 0);
            const cv::Mat x = cv::imread((out / "x.tif").string(), cv::IMREAD_UNCHANGED);
            REQUIRE_FALSE(x.empty());
            CHECK(x.cols == s + 1);
            CHECK(x.rows == s + 1);
            checkScale(out, s / ((kGrid - 1) * kSpacing));
        }
    }

    SUBCASE("decimation halves the scale")
    {
        const fs::path out = root / "decimated";
        const fs::path log = root / "decimated.log";
        INFO("log: ", log.string());
        REQUIRE(run(bin, normalised.string() + " " + out.string() + " " + stretch + " --uv-downsample=2", log) == 0);
        checkScale(out, 1.0 / (2 * kSpacing));
    }

    SUBCASE("mesh_units does not change the scale")
    {
        const fs::path out = root / "mesh-units";
        const fs::path log = root / "mesh-units.log";
        INFO("log: ", log.string());
        REQUIRE(run(bin, normalised.string() + " " + out.string() + " " + stretch + " 7.91", log) == 0);
        checkScale(out, 1.0 / kSpacing);
    }

    SUBCASE("source-scale mode keeps the source scale")
    {
        const fs::path out = root / "source-scale";
        const fs::path log = root / "source-scale.log";
        INFO("log: ", log.string());
        REQUIRE(run(bin, metric.string() + " " + out.string() + " --tifxyz-source=" + source.string(), log) == 0);
        checkScale(out, 1.0 / kSpacing);
    }

    SUBCASE("source-scale mode reduces the scale by the decimation applied")
    {
        const fs::path out = root / "source-scale-decimated";
        const fs::path log = root / "source-scale-decimated.log";
        INFO("log: ", log.string());
        REQUIRE(run(bin, metric.string() + " " + out.string() + " --tifxyz-source=" + source.string() + " --uv-downsample=4", log) == 0);
        checkScale(out, 1.0 / (4 * kSpacing));
    }

    fs::remove_all(root);
}
