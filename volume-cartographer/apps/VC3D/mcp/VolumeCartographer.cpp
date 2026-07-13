#include "VolumeCartographer.hpp"

#include <QCryptographicHash>
#include <QFile>
#include <QImage>
#include <QProcess>
#include <QProcessEnvironment>
#include <QStringList>

#include <fstream>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace vc::mcp
{
namespace
{

std::string coordinateArgument(const Json& value)
{
    if (value.is_number_integer())
        return std::to_string(value.get<std::int64_t>());
    return std::to_string(value.get<double>());
}

std::string sha256(const std::filesystem::path& path)
{
    QFile file(QString::fromStdString(path.string()));
    if (!file.open(QIODevice::ReadOnly))
        throw std::runtime_error("cannot hash artifact " + path.string());
    QCryptographicHash hash(QCryptographicHash::Sha256);
    hash.addData(&file);
    return hash.result().toHex().toStdString();
}

void emitOutput(QProcess& process, const VolumeCartographer::LogCallback& log, std::string& pending)
{
    pending += process.readAll().toStdString();
    std::size_t newline = 0;
    while ((newline = pending.find('\n')) != std::string::npos) {
        std::string line = pending.substr(0, newline);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        log(std::move(line));
        pending.erase(0, newline + 1);
    }
}

}  // namespace

LocalVolumeCartographer::LocalVolumeCartographer(LocalWorkerConfig config) : config_(std::move(config))
{
    std::error_code error;
    config_.growExecutable = std::filesystem::weakly_canonical(config_.growExecutable, error);
    if (error || !std::filesystem::is_regular_file(config_.growExecutable))
        throw std::runtime_error("VC growth executable does not exist: " + config_.growExecutable.string());
    config_.workRoot = std::filesystem::absolute(config_.workRoot).lexically_normal();
    std::filesystem::create_directories(config_.workRoot);
}

Json LocalVolumeCartographer::validateAndNormalizeGrow(const Json& request) const
{
    return validateAndNormalizeLocalGrowRequest(request);
}

WorkerResult LocalVolumeCartographer::growSurface(const std::string& jobId, const Json& normalized, const std::atomic<bool>& cancelRequested, LogCallback log) const
{
    const auto workdir = config_.workRoot / jobId;
    const auto output = workdir / "surface";
    std::filesystem::create_directories(output);
    const auto profilePath = workdir / "profile.json";
    Json profile =
        {{"mode", "seed"},
         {"step_size", 20},
         {"generations", normalized.at("limits").at("max_generations")},
         {"min_area_cm", normalized.at("limits").at("min_area_cm2")},
         {"use_cuda", false}};
    if (normalized.contains("voxel_size_um"))
        profile["voxelsize"] = normalized.at("voxel_size_um");
    {
        std::ofstream stream(profilePath);
        if (!stream)
            throw std::runtime_error("cannot write generated VC profile");
        stream << profile.dump(2) << '\n';
    }

    const auto& seed = normalized.at("coordinates").at("vc_input");
    const std::vector<std::string> arguments =
        {"--volume",
         normalized.at("prediction_source").get<std::string>(),
         "--target-dir",
         output.string(),
         "--params",
         profilePath.string(),
         "--seed",
         coordinateArgument(seed.at("x")),
         coordinateArgument(seed.at("y")),
         coordinateArgument(seed.at("z")),
         "--segment-name",
         "surface"};

    QStringList qtArguments;
    for (const auto& argument : arguments)
        qtArguments.push_back(QString::fromStdString(argument));

    QProcess process;
    process.setProgram(QString::fromStdString(config_.growExecutable.string()));
    process.setArguments(qtArguments);
    process.setWorkingDirectory(QString::fromStdString(workdir.string()));
    process.setProcessChannelMode(QProcess::MergedChannels);
    const auto parentEnvironment = QProcessEnvironment::systemEnvironment();
    QProcessEnvironment environment;
    for (const auto* name :
         {"PATH", "HOME", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "OMP_NUM_THREADS"}) {
        const auto key = QString::fromLatin1(name);
        if (parentEnvironment.contains(key))
            environment.insert(key, parentEnvironment.value(key));
    }
    process.setProcessEnvironment(environment);
    process.start();
    if (!process.waitForStarted(10000))
        throw std::runtime_error("failed to start VC growth executable: " + process.errorString().toStdString());

    std::string pending;
    const auto deadline = std::chrono::steady_clock::now() + config_.timeout;
    bool terminated = false;
    while (process.state() != QProcess::NotRunning) {
        process.waitForReadyRead(100);
        emitOutput(process, log, pending);
        if (cancelRequested.load() || std::chrono::steady_clock::now() >= deadline) {
            terminated = true;
            process.terminate();
            if (!process.waitForFinished(3000)) {
                process.kill();
                process.waitForFinished(3000);
            }
            break;
        }
    }
    process.waitForFinished(1000);
    emitOutput(process, log, pending);
    if (!pending.empty())
        log(std::move(pending));
    if (terminated) {
        if (cancelRequested.load())
            throw std::runtime_error("VC process cancelled");
        throw std::runtime_error("VC process exceeded its timeout");
    }
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0)
        throw std::runtime_error("VC process exited with code " + std::to_string(process.exitCode()));
    if (!std::filesystem::is_regular_file(output / "meta.json"))
        throw std::runtime_error("VC process exited successfully but produced no surface meta.json");

    Json manifest =
        {{"executable", config_.growExecutable.string()},
         {"arguments", arguments},
         {"profile", profile},
         {"working_directory", workdir.string()},
         {"output_directory", output.string()},
         {"exit_code", process.exitCode()}};
    std::ofstream(workdir / "command-manifest.json") << manifest.dump(2) << '\n';

    Json files = Json::array();
    for (const auto& entry : std::filesystem::recursive_directory_iterator(output))
        if (entry.is_regular_file())
            files.push_back(
                {{"name", std::filesystem::relative(entry.path(), output).string()},
                 {"size_bytes", entry.file_size()},
                 {"sha256", sha256(entry.path())}});
    Json surfaceMeta = Json::parse(std::ifstream(output / "meta.json"));
    Json artifacts = Json::array(
        {{{"artifact_id", "surface"},
          {"path", output.string()},
          {"media_type", "application/vnd.volume-cartographer.tifxyz"},
          {"metadata", {{"surface", surfaceMeta}, {"files", files}}}}});

    QImage generations(QString::fromStdString((output / "generations.tif").string()));
    if (!generations.isNull()) {
        const auto previewPath = workdir / "generation-preview.png";
        if (generations.save(QString::fromStdString(previewPath.string()), "PNG"))
            artifacts.push_back(
                {{"artifact_id", "generation-preview"},
                 {"path", previewPath.string()},
                 {"media_type", "image/png"},
                 {"size_bytes", std::filesystem::file_size(previewPath)},
                 {"sha256", sha256(previewPath)}});
    }
    return {process.exitCode(), std::move(manifest), std::move(artifacts)};
}

}  // namespace vc::mcp
