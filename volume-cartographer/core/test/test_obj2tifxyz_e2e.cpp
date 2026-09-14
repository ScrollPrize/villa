// End-to-end test for vc_obj2tifxyz on a synthetic planar mesh with UVs
// normalised to [0,1], the parametrisation of the published segment OBJs.
//
// Covers two regressions:
//   - #1320: with the default stretch factor the 2x2 grid covers only the UV
//     bounding-box corners; a mesh that does not reach them (a diamond, like
//     the published meshes whose corners are cut) rasterizes nothing and the
//     tool must exit non-zero instead of writing an empty tifxyz.
//   - #1319: meta.json "scale" must be grid cells per OBJ unit measured from
//     the grid actually written (a 10-unit spacing stores 0.1), not the UV
//     step, and decimation must be reflected in it.
//
// Opt-in: only runs when VC_RUN_E2E is set to "1".

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

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

// kGrid x kGrid planar mesh, vertices kSpacing apart at z=100, UVs in [0,1].
void writeNormalisedPlaneObj(const fs::path& obj)
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
            f << "vt " << static_cast<double>(c) / (kGrid - 1) << ' '
              << static_cast<double>(r) / (kGrid - 1) << '\n';
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

// Four triangles forming a diamond whose UV bounding box is [0,1]^2 but whose
// corners (0,0), (1,0), (0,1), (1,1) lie outside the mesh.
void writeDiamondObj(const fs::path& obj)
{
    std::ofstream f(obj);
    f << "v 50 0 100\nv 100 50 100\nv 50 100 100\nv 0 50 100\nv 50 50 100\n"
      << "vt 0.5 0\nvt 1 0.5\nvt 0.5 1\nvt 0 0.5\nvt 0.5 0.5\n"
      << "f 1/1 2/2 5/5\nf 2/2 3/3 5/5\nf 3/3 4/4 5/5\nf 4/4 1/1 5/5\n";
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

}

TEST_CASE("vc_obj2tifxyz on a normalised-UV mesh")
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
    const fs::path obj = root / "plane.obj";
    writeNormalisedPlaneObj(obj);

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

    SUBCASE("scale is grid cells per OBJ unit measured from the written grid")
    {
        const fs::path out = root / "stretched";
        const fs::path log = root / "stretched.log";
        const int rc = run(bin, obj.string() + " " + out.string() + " " + std::to_string(kGrid - 1), log);
        INFO("log: ", log.string());
        REQUIRE(rc == 0);
        const auto meta = readMeta(out);
        REQUIRE(meta.contains("scale"));
        CHECK(meta["scale"][0].get<double>() == doctest::Approx(1.0 / kSpacing).epsilon(0.02));
        CHECK(meta["scale"][1].get<double>() == doctest::Approx(1.0 / kSpacing).epsilon(0.02));
    }

    SUBCASE("decimation halves the scale")
    {
        const fs::path out = root / "decimated";
        const fs::path log = root / "decimated.log";
        const int rc = run(bin, obj.string() + " " + out.string() + " " + std::to_string(kGrid - 1) +
                                " --uv-downsample=2", log);
        INFO("log: ", log.string());
        REQUIRE(rc == 0);
        const auto meta = readMeta(out);
        REQUIRE(meta.contains("scale"));
        CHECK(meta["scale"][0].get<double>() == doctest::Approx(1.0 / (2 * kSpacing)).epsilon(0.02));
        CHECK(meta["scale"][1].get<double>() == doctest::Approx(1.0 / (2 * kSpacing)).epsilon(0.02));
    }

    fs::remove_all(root);
}
