// vc_render_tifxyz end to end: runs this tree's tool on a real segment and
// volume with identical arguments, `--iterations` times, recording wall time,
// CPU time and peak RSS of each run, and a summary of the rendered TIFFs (so
// two trees' outputs can be compared). Process start-up is part of the time:
// it is part of the workflow.

#include "bench_common.hpp"

#include <opencv2/imgcodecs.hpp>

#include <spawn.h>
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

} // namespace

int main(int argc, char** argv)
{
    try {
        Args args(argc, argv);
        const auto tool = args.str("tool", VC_BENCH_RENDER_TIFXYZ);
        const auto volume = args.str("volume");
        const auto segment = args.str("segment");
        if (volume.empty() || segment.empty())
            throw std::invalid_argument("--volume <OME-Zarr folder> and --segment <tifxyz folder> are required");
        const auto backend = args.str("backend", "cpu");
        const int iterations = int(args.num("iterations", 3));
        const auto workDir = std::filesystem::path(args.str("work-dir", "/mnt/raid_nvme/dev_builds/bench_out"));
        const bool prefetch = args.flag("prefetch");
        const auto json = args.str("json");
        std::vector<std::string> common{
            "--group-idx", args.str("group", "2"),
            "--scale", args.str("scale", "1"),
            "--scale-segmentation", args.str("segment-scale", "1"),
            "--num-slices", args.str("slices", "21"),
            "--surface-interpolation", args.str("surface-interpolation", "linear"),
            "--cache-gb", args.str("cache-gb", "16"),
            "--segmentation", segment,
        };
        if (const auto crop = args.str("crop"); !crop.empty()) {
            int x, y, w, h;
            if (std::sscanf(crop.c_str(), "%d,%d,%d,%d", &x, &y, &w, &h) != 4)
                throw std::invalid_argument("--crop x,y,width,height");
            for (const auto& [k, v] : std::vector<std::pair<std::string, int>>{
                     {"--crop-x", x}, {"--crop-y", y}, {"--crop-width", w}, {"--crop-height", h}}) {
                common.push_back(k);
                common.push_back(std::to_string(v));
            }
        }
#ifdef VC_BENCH_VC_READER
        const std::string pipeline = "vc_reader-" + backend;
        if (backend == "cuda") {
            common.push_back("--cuda");
            common.push_back("--cuda-cache-gb");
            common.push_back(args.str("gpu-cache-gb", "8"));
        } else if (backend != "cpu") {
            throw std::invalid_argument("--backend must be cpu or cuda");
        }
#else
        const std::string pipeline = "chunkcache";
        if (backend != "cpu")
            throw std::invalid_argument("this tree's vc_render_tifxyz has no --backend but cpu");
        if (prefetch)
            throw std::invalid_argument("--prefetch runs on a vc_reader build");
#endif
        args.rejectUnused();

        std::filesystem::create_directories(workDir);
        std::vector<std::string> extraEnv;
        std::string volumeArg = volume;
        if (prefetch) {
            // As bench_pipeline_vcr's --prefetch: the s3:// source with the cache the
            // local folder lives in as VC3D's cache root, so the run fills it.
            const auto path = std::filesystem::weakly_canonical(volume).generic_string();
            const auto at = path.find("/s3/");
            if (at == std::string::npos)
                throw std::invalid_argument("--prefetch needs a --volume inside a vc_reader cache");
            const auto config = workDir / "prefetch-config";
            std::filesystem::create_directories(config);
            std::ofstream(config / "cache.json") << "{\"version\": 1, \"root\": \"" << path.substr(0, at) << "\"}\n";
            extraEnv.push_back("VC3D_CONFIG_DIR=" + config.string());
            volumeArg = "s3://" + path.substr(at + 4);
        }

        std::map<std::string, std::string> config{{"tool", tool}, {"volume", volumeArg}, {"segment", segment}};
        for (std::size_t i = 0; i + 1 < common.size(); i += 2)
            if (common[i] != "--segmentation")
                config[common[i].substr(2)] = common[i + 1];
        Report report("render_tifxyz", pipeline, config);

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
                throw std::runtime_error("vc_render_tifxyz failed (status " + std::to_string(run.status) + ")");
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
        report.finish(json);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_render_tifxyz: %s\n", e.what());
        return 1;
    }
}
