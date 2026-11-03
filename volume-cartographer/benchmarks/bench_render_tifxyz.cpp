// vc_render_tifxyz end to end: runs this tree's tool on a real segment and
// volume with identical arguments, `--iterations` times, recording wall time,
// CPU time and peak RSS of each run, and a summary of the rendered TIFFs (so
// two trees' outputs can be compared). Process start-up is part of the time:
// it is part of the workflow.

#include "bench_session.hpp"

#include <opencv2/imgcodecs.hpp>

#include <spawn.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/wait.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

extern char** environ;

using namespace vcbench;

namespace {

struct Run {
    double wallMs = 0, userMs = 0, sysMs = 0, maxRssMb = 0;
    int status = 0;
};

Run spawn(const std::vector<std::string>& argv, const std::vector<std::string>& extraEnv)
{
    std::vector<char*> args;
    for (const auto& a : argv)
        args.push_back(const_cast<char*>(a.c_str()));
    args.push_back(nullptr);
    std::vector<std::string> envStrings;
    for (char** e = environ; *e; ++e)
        envStrings.emplace_back(*e);
    envStrings.insert(envStrings.end(), extraEnv.begin(), extraEnv.end());
    std::vector<char*> env;
    for (auto& e : envStrings)
        env.push_back(e.data());
    env.push_back(nullptr);

    Run run;
    const auto start = Clock::now();
    pid_t pid = 0;
    if (posix_spawn(&pid, args[0], nullptr, nullptr, args.data(), env.data()) != 0)
        throw std::runtime_error("cannot start " + argv[0]);
    rusage usage{};
    if (wait4(pid, &run.status, 0, &usage) != pid)
        throw std::runtime_error("wait4 failed");
    run.wallMs = msSince(start);
    run.userMs = usage.ru_utime.tv_sec * 1e3 + usage.ru_utime.tv_usec / 1e3;
    run.sysMs = usage.ru_stime.tv_sec * 1e3 + usage.ru_stime.tv_usec / 1e3;
    run.maxRssMb = double(usage.ru_maxrss) / 1024.0;
    return run;
}

struct Output {
    std::size_t images = 0, pixels = 0, nonzero = 0;
    double mean = 0;
};

Output summarizeTiffs(const std::filesystem::path& folder)
{
    Output out;
    double sum = 0;
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(folder))
        if (entry.path().extension() == ".tif" || entry.path().extension() == ".tiff")
            files.push_back(entry.path());
    for (const auto& file : files) {
        std::vector<cv::Mat> pages;
        cv::imreadmulti(file.string(), pages, cv::IMREAD_UNCHANGED);
        for (const auto& page : pages) {
            ++out.images;
            out.pixels += page.total();
            out.nonzero += std::size_t(cv::countNonZero(page.reshape(1)));
            sum += cv::sum(page)[0];
        }
    }
    out.mean = out.pixels ? sum / double(out.pixels) : 0;
    return out;
}

std::string number(double value)
{
    std::ostringstream out;
    out << value;
    return out.str();
}

} // namespace

int main(int argc, char** argv)
{
    try {
        Fixture fixture(argc, argv, "render_tifxyz");
        if (fixture.listOnly())
            return 0;
        const auto tool = fixture.str("tool", VC_BENCH_RENDER_TIFXYZ);
        const auto& segment = fixture.segment();
        const int iterations = fixture.integer("iterations", 3);
        const auto workDir = std::filesystem::path(
            fixture.str("work_dir", (std::filesystem::temp_directory_path() / "vc_bench_render_tifxyz").string()));
        const bool prefetch = fixture.prefetching();
        std::vector<std::string> common{
            "--group-idx", fixture.str("group", "2"),
            "--scale", fixture.str("scale", "1"),
            "--scale-segmentation", number(segment.scale),
            "--num-slices", fixture.str("slices", "21"),
            "--surface-interpolation", fixture.str("surface_interpolation", "linear"),
            "--cache-gb", fixture.str("cache_gb", "16"),
            "--segmentation", segment.path.string(),
        };
        if (const auto crop = fixture.numbers("crop"); !crop.empty()) {
            if (crop.size() != 4)
                throw std::invalid_argument("crop: [x, y, width, height]");
            const char* keys[] = {"--crop-x", "--crop-y", "--crop-width", "--crop-height"};
            for (int i = 0; i < 4; ++i) {
                common.push_back(keys[i]);
                common.push_back(std::to_string(int(crop[i])));
            }
        }
        const auto& pipeline = fixture.pipelineName();
#ifdef VC_BENCH_VC_READER
        const auto gpuCacheGb = fixture.str("gpu_cache_gb", "8");
        if (fixture.backend() == "cuda") {
            common.push_back("--cuda");
            common.push_back("--cuda-cache-gb");
            common.push_back(gpuCacheGb);
        }
#endif

        std::filesystem::create_directories(workDir);
        std::vector<std::string> extraEnv;
        // cold, partial and --prefetch read the volume's URL through the cache root the
        // fixture names in this config folder's cache.json.
        const auto vc3dConfig = workDir / ("vc3d-config-" + std::to_string(::getpid()));
        const auto source = fixture.volumeSource(vc3dConfig, false);
        if (source.remote)
            extraEnv.push_back("VC3D_CONFIG_DIR=" + vc3dConfig.string());
        if (!source.offline.empty()) {
            // Cache state cached: the tool reads the cache only, and fails on a chunk it lacks.
            std::ofstream ini(vc3dConfig / "VC3D.ini");
            ini << "[perf]\nremote_offline=" << source.offline << "\n";
            if (!ini)
                throw std::runtime_error("cannot write " + (vc3dConfig / "VC3D.ini").string());
        }
        const auto& volumeArg = source.location;
        fixture.rejectUnused();

        auto config = fixture.config();
        config["tool"] = tool;
        // The tool's options: "--key value", or a bare "--flag".
        for (std::size_t i = 0; i < common.size(); ++i) {
            const bool hasValue = i + 1 < common.size() && common[i + 1].rfind("--", 0) != 0;
            if (common[i] != "--segmentation")
                config[common[i].substr(2)] = hasValue ? common[i + 1] : "1";
            if (hasValue)
                ++i;
        }
        auto report = fixture.makeReport();
        report.setConfig(config);

        for (int i = 0; i < (prefetch ? 1 : iterations); ++i) {
            const auto out = workDir / ("render_tifxyz-" + pipeline + "-" + std::to_string(i));
            std::filesystem::remove_all(out);
            std::vector<std::string> argvFull{tool, "--volume", volumeArg, "--tif-output", out.string()};
            argvFull.insert(argvFull.end(), common.begin(), common.end());
            if (i == 0) {
                std::string line;
                for (const auto& a : argvFull)
                    line += a + " ";
                std::fprintf(stderr, "running: %s\n", line.c_str());
            }
            const auto run = spawn(argvFull, extraEnv);
            if (!WIFEXITED(run.status) || WEXITSTATUS(run.status) != 0)
                throw std::runtime_error(
                    "vc_render_tifxyz failed (status " + std::to_string(run.status) + ")" +
                    (source.offline.empty() ? std::string()
                                            : std::string("; it reads the cache offline, so a chunk the cache lacks "
                                                          "fails it: run benchmarks/prefetch.sh render_tifxyz")));
            const std::string step = i == 0 ? "first run" : "repeat run";
            report.add(step, "wall_ms", run.wallMs);
            report.add(step, "cpu_ms", run.userMs + run.sysMs);
            report.add(step, "max_rss_mb", run.maxRssMb);
            const auto summary = summarizeTiffs(out);
            report.add(step, "out_mean", summary.mean);
            report.add(step, "out_nonzero_pct",
                       summary.pixels ? 100.0 * double(summary.nonzero) / double(summary.pixels) : 0);
            report.add(step, "out_images", double(summary.images));
        }
        report.finish();
        std::filesystem::remove_all(vc3dConfig);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_render_tifxyz: %s\n", e.what());
        return 1;
    }
}
