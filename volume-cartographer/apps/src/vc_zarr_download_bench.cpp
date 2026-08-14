#include "vc/core/render/ZarrDownloadBenchmark.hpp"
#include "vc/core/util/RemoteAuth.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include <boost/program_options.hpp>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

constexpr double kMiB = 1024.0 * 1024.0;

struct TemporaryOutput {
    fs::path path;
    bool removeOnExit = false;

    ~TemporaryOutput()
    {
        if (!removeOnExit || path.empty())
            return;
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

vc::render::ZarrDownloadSchedule parseSchedule(const std::string& value)
{
    if (value == "auto")
        return vc::render::ZarrDownloadSchedule::Adaptive;
    if (value == "fixed")
        return vc::render::ZarrDownloadSchedule::Fixed;
    throw std::invalid_argument("--mode must be 'auto' or 'fixed'");
}

} // namespace

int main(int argc, char** argv)
{
    std::string source;
    std::string mode = "auto";
    std::string sink = "discard";
    std::string tempDirectory;
    int level = 0;
    std::size_t chunks = 256;
    std::size_t workers = 64;
    std::size_t minimumWorkers = 2;
    std::size_t samplesPerWorker = 4;
    double minimumEpochSeconds = 2.0;
    double maximumEpochSeconds = 5.0;
    double unstableProbeSeconds = 60.0;
    double stableProbeSeconds = 300.0;
    double minimumStabilitySeconds = 300.0;
    std::size_t initialProbeMultiplier = 4;
    std::size_t continuousSearchTurns = 5;
    std::uint64_t seed = 0;
    bool keepTemp = false;
    bool anonymous = false;

    po::options_description options("vc_zarr_download_bench options");
    options.add_options()
        ("help,h", "Show help")
        ("source", po::value<std::string>(&source)->required(),
         "S3/HTTP OME-Zarr root or concrete array URL")
        ("level,l", po::value<int>(&level)->default_value(0),
         "Logical pyramid level to benchmark")
        ("chunks,n", po::value<std::size_t>(&chunks)->default_value(256),
         "Number of unique chunks to download")
        ("mode", po::value<std::string>(&mode)->default_value("auto"),
         "Download admission mode: auto or fixed")
        ("workers,j", po::value<std::size_t>(&workers)->default_value(64),
         "Maximum workers in auto mode; exact workers in fixed mode")
        ("min-workers", po::value<std::size_t>(&minimumWorkers)->default_value(2),
         "Initial/minimum workers in auto mode")
        ("samples-per-worker", po::value<std::size_t>(&samplesPerWorker)->default_value(4),
         "Successful chunks per admitted worker in the displayed bandwidth window")
        ("epoch-min-seconds", po::value<double>(&minimumEpochSeconds)->default_value(2.0),
         "Minimum auto-probe measurement duration")
        ("epoch-max-seconds", po::value<double>(&maximumEpochSeconds)->default_value(5.0),
         "Maximum auto-probe measurement duration")
        ("unstable-probe-seconds", po::value<double>(&unstableProbeSeconds)->default_value(60.0),
         "Exploration interval after a 2x bandwidth change")
        ("stable-probe-seconds", po::value<double>(&stableProbeSeconds)->default_value(300.0),
         "Exploration interval at stable bandwidth")
        ("stability-min-seconds", po::value<double>(&minimumStabilitySeconds)->default_value(300.0),
         "Required saturated observation time before bandwidth can be stable")
        ("initial-probe-multiplier", po::value<std::size_t>(&initialProbeMultiplier)->default_value(4),
         "Initial outward concurrency probe multiplier")
        ("search-turns", po::value<std::size_t>(&continuousSearchTurns)->default_value(5),
         "Direction reversals/center confirmations before search settles")
        ("seed", po::value<std::uint64_t>(&seed)->default_value(0),
         "Deterministic distributed chunk-selection seed")
        ("sink", po::value<std::string>(&sink)->default_value("discard"),
         "Encoded payload sink: discard or temp")
        ("temp-dir", po::value<std::string>(&tempDirectory),
         "Directory for --sink=temp (generated under the system temp dir by default)")
        ("keep-temp", po::bool_switch(&keepTemp),
         "Keep a generated temporary output directory")
        ("anonymous", po::bool_switch(&anonymous),
         "Do not load AWS credentials; use unsigned requests");

    po::positional_options_description positional;
    positional.add("source", 1);
    po::variables_map parsed;
    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(options)
                      .positional(positional)
                      .run(),
                  parsed);
        if (parsed.count("help")) {
            std::cout << "Usage: vc_zarr_download_bench [options] <s3-or-http-zarr>\n\n"
                      << options << '\n';
            return 0;
        }
        po::notify(parsed);

        const auto schedule = parseSchedule(mode);
        if (workers == 0 || minimumWorkers == 0 || samplesPerWorker == 0)
            throw std::invalid_argument("worker and sample counts must be positive");
        if (schedule == vc::render::ZarrDownloadSchedule::Adaptive &&
            minimumWorkers > workers)
            throw std::invalid_argument("--min-workers cannot exceed --workers");
        if (minimumEpochSeconds < 0.0 || maximumEpochSeconds < minimumEpochSeconds)
            throw std::invalid_argument(
                "epoch durations must satisfy 0 <= minimum <= maximum");
        if (unstableProbeSeconds < 0.0 || stableProbeSeconds < unstableProbeSeconds)
            throw std::invalid_argument(
                "probe intervals must satisfy 0 <= unstable <= stable");
        if (minimumStabilitySeconds < 0.0)
            throw std::invalid_argument(
                "--stability-min-seconds cannot be negative");
        if (initialProbeMultiplier < 2 || continuousSearchTurns == 0)
            throw std::invalid_argument(
                "initial probe multiplier must be >= 2 and search turns positive");
        if (sink != "discard" && sink != "temp")
            throw std::invalid_argument("--sink must be 'discard' or 'temp'");
        if (sink == "discard" && !tempDirectory.empty())
            throw std::invalid_argument("--temp-dir requires --sink=temp");
        if (sink == "discard" && keepTemp)
            throw std::invalid_argument("--keep-temp requires --sink=temp");

        vc::render::RemoteZarrOpenOptions openOptions;
        openOptions.discoverAwsCredentials = !anonymous;
        std::cout << "Opening "
                  << vc::parseRemoteVolumeSpec(source).portableLocator << "\n";
        auto remoteOpen = vc::render::openRemoteZarrPyramid(
            source, std::move(openOptions));
        const auto& opened = remoteOpen.opened;
        if (level < 0 || static_cast<std::size_t>(level) >= opened.fetchers.size() ||
            !opened.fetchers[static_cast<std::size_t>(level)]) {
            throw std::out_of_range("requested --level is not present in the Zarr pyramid");
        }

        TemporaryOutput temporary;
        vc::render::ZarrDownloadBenchmarkOptions benchmark;
        benchmark.level = level;
        benchmark.chunkCount = chunks;
        benchmark.seed = seed;
        benchmark.workers = workers;
        benchmark.schedule = schedule;
        benchmark.adaptive.minimum = minimumWorkers;
        benchmark.adaptive.maximum = workers;
        benchmark.adaptive.successfulSamplesPerWorker = samplesPerWorker;
        benchmark.adaptive.minimumEpochSeconds = minimumEpochSeconds;
        benchmark.adaptive.maximumEpochSeconds = maximumEpochSeconds;
        benchmark.adaptive.unstableProbeIntervalSeconds = unstableProbeSeconds;
        benchmark.adaptive.stableProbeIntervalSeconds = stableProbeSeconds;
        benchmark.adaptive.minimumStabilityObservationSeconds =
            minimumStabilitySeconds;
        benchmark.adaptive.initialProbeMultiplier = initialProbeMultiplier;
        benchmark.adaptive.continuousSearchTurns = continuousSearchTurns;
        benchmark.progressInterval = std::chrono::seconds(1);
        benchmark.progressCallback = [samplesPerWorker](
                                         const vc::render::ZarrDownloadProgress& progress) {
            std::cout << "progress elapsed=" << std::fixed << std::setprecision(1)
                      << progress.elapsedSeconds << "s"
                      << " queued=" << progress.queuedChunks
                      << " downloading=" << progress.downloadingChunks
                      << " bandwidth=" << std::setprecision(2)
                      << progress.transferStats.bytesPerSecond / kMiB << "MiB/s"
                      << " admission=" << progress.transferStats.admissionLimit
                      << " samples=" << progress.transferStats.sampleCount << '/'
                      << progress.transferStats.admissionLimit * samplesPerWorker
                      << "\n";
        };
        if (sink == "temp") {
            if (!tempDirectory.empty()) {
                temporary.path = tempDirectory;
            } else {
                const auto nonce = std::chrono::high_resolution_clock::now()
                                       .time_since_epoch().count();
                temporary.path = fs::temp_directory_path() /
                    ("vc_zarr_download_bench_" + std::to_string(nonce));
                temporary.removeOnExit = !keepTemp;
            }
            benchmark.outputDirectory = temporary.path;
        }

        const auto& shape = opened.shapes.at(static_cast<std::size_t>(level));
        const auto& chunkShape = opened.chunkShapes.at(static_cast<std::size_t>(level));
        std::cout << "level logical=" << level;
        if (static_cast<std::size_t>(level) < opened.levelNumbers.size() &&
            opened.levelNumbers[static_cast<std::size_t>(level)] >= 0) {
            std::cout << " physical="
                      << opened.levelNumbers[static_cast<std::size_t>(level)];
        }
        std::cout << " shape=" << shape[0] << 'x' << shape[1] << 'x' << shape[2]
                  << " chunk=" << chunkShape[0] << 'x' << chunkShape[1] << 'x'
                  << chunkShape[2] << " requests=" << chunks
                  << " mode=" << mode << " workers=" << workers
                  << " sink=" << sink << "\n";
        if (benchmark.outputDirectory)
            std::cout << "temporary_output=" << benchmark.outputDirectory->string() << "\n";

        const auto result = vc::render::runZarrDownloadBenchmark(opened, benchmark);
        std::cout << "result requested=" << result.requestedChunks
                  << " found=" << result.foundChunks
                  << " missing=" << result.missingChunks
                  << " http_errors=" << result.httpErrors
                  << " io_errors=" << result.ioErrors
                  << " decode_errors=" << result.decodeErrors
                  << " sink_errors=" << result.sinkErrors
                  << " bytes=" << result.encodedBytes << "\n";
        std::cout << "bandwidth=" << std::fixed << std::setprecision(2)
                  << result.finalTransferStats.bytesPerSecond / kMiB << "MiB/s"
                  << " wall_s=" << result.wallSeconds
                  << " bytes=" << result.encodedBytes << "\n";
        std::cout << "latency_ms mean=" << result.latencyMeanMilliseconds
                  << " p50=" << result.latencyP50Milliseconds
                  << " p95=" << result.latencyP95Milliseconds
                  << " min=" << result.latencyMinimumMilliseconds
                  << " max=" << result.latencyMaximumMilliseconds
                  << " peak_active=" << result.peakActive
                  << " final_admission=" << result.finalTransferStats.admissionLimit
                  << " samples=" << result.finalTransferStats.sampleCount << "\n";
        std::cout << "admission";
        for (const auto& sample : result.concurrencySamples) {
            std::cout << " [chunks=" << sample.completedChunks
                      << " MiB=" << std::setprecision(1)
                      << static_cast<double>(sample.encodedBytes) / kMiB
                      << " workers=" << sample.admissionLimit << "]";
        }
        std::cout << "\n";
        if (!result.firstError.empty())
            std::cout << "first_error=" << result.firstError << "\n";
        if (benchmark.outputDirectory && !temporary.removeOnExit)
            std::cout << "kept_output=" << benchmark.outputDirectory->string() << "\n";

        const auto failures = result.httpErrors + result.ioErrors +
            result.decodeErrors + result.sinkErrors;
        if (result.foundChunks == 0 || failures != 0)
            return 2;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "vc_zarr_download_bench error: " << e.what() << "\n";
        return 1;
    }
}
