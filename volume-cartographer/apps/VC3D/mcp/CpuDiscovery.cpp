#include "CpuDiscovery.hpp"

#include <fastmcpp/exceptions.hpp>

#include <QProcess>
#include <QProcessEnvironment>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <numeric>
#include <set>

namespace vc::mcp
{
namespace
{
constexpr int kMaxSide = 8192;
constexpr std::size_t kMaxPixels = 64ull * 1024ull * 1024ull;
constexpr std::size_t kMaxSlices = 129;

void require(bool ok, const std::string& message)
{
    if (!ok)
        throw fastmcpp::ValidationError(message);
}

std::filesystem::path existingAbsolute(const Json& request, const char* key, bool directory = false)
{
    require(request.contains(key) && request.at(key).is_string(), std::string(key) + " is required");
    std::filesystem::path path = request.at(key).get<std::string>();
    require(path.is_absolute(), std::string(key) + " must be an absolute local path");
    std::error_code error;
    path = std::filesystem::weakly_canonical(path, error);
    require(!error && std::filesystem::exists(path), std::string(key) + " does not exist");
    require(!directory || std::filesystem::is_directory(path), std::string(key) + " must be a directory");
    return path;
}

bool allowedRemoteVolume(const std::string& uri)
{
    return (uri.starts_with("s3://vesuvius-challenge-open-data/") ||
            uri.starts_with("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/") || uri.starts_with("https://dl.ash2txt.org/")) &&
           uri.find("..") == std::string::npos && uri.find('?') == std::string::npos && uri.find('#') == std::string::npos;
}

void validateRequestId(const Json& request)
{
    require(request.contains("client_request_id") && request.at("client_request_id").is_string(), "client_request_id is required");
    const auto id = request.at("client_request_id").get<std::string>();
    require(!id.empty() && id.size() <= 128, "client_request_id must contain 1 to 128 characters");
}

std::vector<std::filesystem::path> imageFiles(const std::filesystem::path& input)
{
    if (std::filesystem::is_regular_file(input))
        return {input};
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(input)) {
        if (!entry.is_regular_file())
            continue;
        auto extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (extension == ".tif" || extension == ".tiff" || extension == ".png")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<cv::Mat> readStack(const std::filesystem::path& input)
{
    std::vector<cv::Mat> stack;
    const auto files = imageFiles(input);
    require(!files.empty(), "surface volume contains no TIFF/PNG slices");
    if (files.size() == 1 && (files.front().extension() == ".tif" || files.front().extension() == ".tiff"))
        cv::imreadmulti(files.front().string(), stack, cv::IMREAD_GRAYSCALE);
    if (stack.empty()) {
        require(files.size() <= kMaxSlices, "surface volume exceeds the 129-slice CPU limit");
        for (const auto& file : files) {
            cv::Mat image = cv::imread(file.string(), cv::IMREAD_GRAYSCALE);
            require(!image.empty(), "failed to decode " + file.string());
            stack.push_back(std::move(image));
        }
    }
    require(!stack.empty() && stack.size() <= kMaxSlices, "surface volume exceeds the 129-slice CPU limit");
    const auto size = stack.front().size();
    require(
        size.width > 0 && size.height > 0 && size.width <= kMaxSide && size.height <= kMaxSide, "surface slices exceed dimension limits");
    require(
        std::size_t(size.width) * std::size_t(size.height) * stack.size() <= kMaxPixels, "surface stack exceeds the 64-megapixel CPU limit");
    for (auto& image : stack) {
        require(image.size() == size && image.type() == CV_8UC1, "surface slices must have matching 8-bit grayscale dimensions");
    }
    return stack;
}

cv::Mat normalizeU8(const cv::Mat& source)
{
    cv::Mat output;
    double minimum = 0, maximum = 0;
    cv::minMaxLoc(source, &minimum, &maximum);
    if (maximum <= minimum)
        return cv::Mat::zeros(source.size(), CV_8U);
    source.convertTo(output, CV_8U, 255.0 / (maximum - minimum), -minimum * 255.0 / (maximum - minimum));
    return output;
}

void writeImage(const std::filesystem::path& path, const cv::Mat& image)
{
    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("failed to write " + path.string());
}

Json artifact(const std::string& id, const std::filesystem::path& path, const std::string& mediaType)
{
    return {{"artifact_id", id}, {"path", path.string()}, {"media_type", mediaType}};
}

WorkerResult diagnostics(const std::filesystem::path& output, const Json& input, const std::atomic<bool>& cancelled, const CpuDiscovery::LogCallback& log)
{
    auto stack = readStack(input.at("surface_volume_path").get<std::string>());
    if (cancelled.load())
        throw std::runtime_error("CPU discovery cancelled");
    log("Loaded " + std::to_string(stack.size()) + " surface slices");
    const int rows = stack[0].rows, cols = stack[0].cols;
    cv::Mat mean = cv::Mat::zeros(rows, cols, CV_32F), minimum, maximum, deviation = cv::Mat::zeros(rows, cols, CV_32F);
    cv::Mat median(rows, cols, CV_8U);
    cv::Mat gradient = cv::Mat::zeros(rows, cols, CV_32F), depth = cv::Mat::zeros(rows, cols, CV_16U),
            persistence = cv::Mat::zeros(rows, cols, CV_32F);
    stack[0].convertTo(minimum, CV_32F);
    minimum.copyTo(maximum);
    for (const auto& slice : stack) {
        cv::Mat value;
        slice.convertTo(value, CV_32F);
        mean += value;
        cv::min(minimum, value, minimum);
        cv::max(maximum, value, maximum);
    }
    mean /= float(stack.size());
    std::vector<uchar> column(stack.size());
    for (int y = 0; y < rows; ++y) {
        auto* outputRow = median.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x) {
            for (std::size_t z = 0; z < stack.size(); ++z)
                column[z] = stack[z].at<uchar>(y, x);
            auto middle = column.begin() + column.size() / 2;
            std::nth_element(column.begin(), middle, column.end());
            outputRow[x] = *middle;
        }
    }
    for (std::size_t z = 0; z < stack.size(); ++z) {
        cv::Mat value, delta;
        stack[z].convertTo(value, CV_32F);
        cv::absdiff(value, mean, delta);
        deviation += delta;
        cv::Mat greater;
        cv::compare(value, mean + 8.0f, greater, cv::CMP_GT);
        greater.convertTo(greater, CV_32F, 1.0 / 255.0);
        persistence += greater;
        for (int y = 0; y < rows; ++y) {
            const auto* p = stack[z].ptr<uchar>(y);
            auto* best = maximum.ptr<float>(y);
            auto* d = depth.ptr<std::uint16_t>(y);
            for (int x = 0; x < cols; ++x)
                if (float(p[x]) >= best[x]) {
                    best[x] = p[x];
                    d[x] = std::uint16_t(z);
                }
        }
        if (z > 0) {
            cv::Mat diff;
            cv::absdiff(stack[z], stack[z - 1], diff);
            cv::Mat f;
            diff.convertTo(f, CV_32F);
            cv::max(gradient, f, gradient);
        }
    }
    deviation /= float(stack.size());
    persistence /= float(stack.size());
    std::filesystem::create_directories(output);
    writeImage(output / "mean.png", normalizeU8(mean));
    writeImage(output / "median.png", median);
    writeImage(output / "min.png", normalizeU8(minimum));
    writeImage(output / "max.png", normalizeU8(maximum));
    writeImage(output / "deviation-from-mean.png", normalizeU8(deviation));
    writeImage(output / "normal-gradient.png", normalizeU8(gradient));
    writeImage(output / "depth-of-max.png", normalizeU8(depth));
    writeImage(output / "persistence.png", normalizeU8(persistence));
    const Json manifest =
        {{"kind", "cpu_surface_diagnostics_v1"},
         {"backend", "cpu"},
         {"source", input.at("surface_volume_path")},
         {"shape_offset_v_u", {stack.size(), rows, cols}},
         {"outputs",
          {"mean.png",
           "median.png",
           "min.png",
           "max.png",
           "deviation-from-mean.png",
           "normal-gradient.png",
           "depth-of-max.png",
           "persistence.png"}}};
    std::ofstream(output / "manifest.json") << manifest.dump(2) << '\n';
    return {
        0,
        {{"operation", "vc_render_surface_diagnostics"}, {"backend", "cpu"}, {"input", input}},
        Json::array({artifact("diagnostics", output, "application/vnd.vc.cpu-diagnostics")})};
}

WorkerResult features(const std::filesystem::path& output, const Json& input, const std::atomic<bool>& cancelled, const CpuDiscovery::LogCallback&)
{
    const auto root = std::filesystem::path(input.at("diagnostics_path").get<std::string>());
    cv::Mat image = cv::imread((root / "mean.png").string(), cv::IMREAD_GRAYSCALE);
    require(!image.empty(), "diagnostics_path has no mean.png");
    if (cancelled.load())
        throw std::runtime_error("CPU discovery cancelled");
    std::filesystem::create_directories(output);
    cv::Mat clahe, blackhat, tophat, dog, lap, gaborMax = cv::Mat::zeros(image.size(), CV_32F);
    cv::createCLAHE(2.0, {8, 8})->apply(image, clahe);
    auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {17, 17});
    cv::morphologyEx(image, blackhat, cv::MORPH_BLACKHAT, kernel);
    cv::morphologyEx(image, tophat, cv::MORPH_TOPHAT, kernel);
    cv::Mat g1, g2;
    cv::GaussianBlur(image, g1, {0, 0}, 1.2);
    cv::GaussianBlur(image, g2, {0, 0}, 4.0);
    cv::subtract(g1, g2, dog, cv::noArray(), CV_32F);
    cv::Laplacian(g2, lap, CV_32F, 3);
    cv::absdiff(lap, cv::Scalar(0), lap);
    cv::Mat source;
    image.convertTo(source, CV_32F, 1.0 / 255.0);
    for (double theta : {0.0, CV_PI / 4, CV_PI / 2, 3 * CV_PI / 4}) {
        cv::Mat response;
        cv::filter2D(source, response, CV_32F, cv::getGaborKernel({21, 21}, 3.5, theta, 9.0, 0.6));
        cv::absdiff(response, cv::Scalar(0), response);
        cv::max(gaborMax, response, gaborMax);
    }
    cv::Mat combined = normalizeU8(blackhat);
    cv::addWeighted(combined, .35, normalizeU8(dog), .25, 0, combined);
    cv::addWeighted(combined, .7, normalizeU8(lap), .15, 0, combined);
    cv::addWeighted(combined, .85, normalizeU8(gaborMax), .15, 0, combined);
    writeImage(output / "clahe.png", clahe);
    writeImage(output / "black-hat.png", normalizeU8(blackhat));
    writeImage(output / "top-hat.png", normalizeU8(tophat));
    writeImage(output / "difference-of-gaussians.png", normalizeU8(dog));
    writeImage(output / "laplacian-of-gaussian.png", normalizeU8(lap));
    writeImage(output / "oriented-gabor.png", normalizeU8(gaborMax));
    writeImage(output / "candidate-score.png", combined);
    Json manifest =
        {{"kind", "classical_ink_features_v1"}, {"backend", "cpu"}, {"score_semantics", "heuristic_candidate_score_not_probability"}, {"source", root.string()}};
    std::ofstream(output / "manifest.json") << manifest.dump(2) << '\n';
    return {
        0,
        {{"operation", "ink_compute_classical_features"}, {"backend", "cpu"}, {"input", input}},
        Json::array({artifact("classical-features", output, "application/vnd.vc.classical-features")})};
}

WorkerResult candidates(const std::filesystem::path& output, const Json& input, const std::atomic<bool>& cancelled, const CpuDiscovery::LogCallback&)
{
    cv::Mat score = cv::imread(input.at("score_path").get<std::string>(), cv::IMREAD_GRAYSCALE);
    require(!score.empty(), "failed to decode score_path");
    cv::Mat mask;
    const double threshold = input.value("threshold", -1.0);
    cv::threshold(score, mask, threshold >= 0 ? threshold : 0, 255, threshold >= 0 ? cv::THRESH_BINARY : cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    cv::Mat labels, stats, centers;
    int count = cv::connectedComponentsWithStats(mask, labels, stats, centers, 8, CV_32S);
    struct Item {
        int label;
        double score;
        int area;
    };
    std::vector<Item> ranked;
    for (int i = 1; i < count; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < input.value("min_area", 8))
            continue;
        cv::Mat component = labels == i;
        ranked.push_back({i, cv::mean(score, component)[0], area});
    }
    std::sort(ranked.begin(), ranked.end(), [](const Item& a, const Item& b) {
        return a.score > b.score || (a.score == b.score && a.label < b.label);
    });
    ranked.resize(std::min<std::size_t>(ranked.size(), input.value("max_candidates", 500)));
    Json list = Json::array();
    int index = 0;
    for (const auto& item : ranked) {
        if (cancelled.load())
            throw std::runtime_error("CPU discovery cancelled");
        int i = item.label;
        list.push_back(
            {{"id", "candidate-" + std::to_string(++index)},
             {"bbox_uv",
              {{"x", stats.at<int>(i, cv::CC_STAT_LEFT)},
               {"y", stats.at<int>(i, cv::CC_STAT_TOP)},
               {"width", stats.at<int>(i, cv::CC_STAT_WIDTH)},
               {"height", stats.at<int>(i, cv::CC_STAT_HEIGHT)}}},
             {"area_pixels", item.area},
             {"heuristic_score", item.score / 255.0}});
    }
    std::filesystem::create_directories(output);
    writeImage(output / "candidate-mask.png", mask);
    writeImage(output / "candidate-score.png", score);
    Json document = {{"kind", "candidate_regions_v1"}, {"score_semantics", "heuristic_not_probability"}, {"source_score_path", input.at("score_path")}, {"candidates", list}};
    std::ofstream(output / "candidate-set.json") << document.dump(2) << '\n';
    return {
        0,
        {{"operation", "ink_find_candidate_regions"}, {"backend", "cpu"}, {"input", input}},
        Json::array({artifact("candidate-set", output, "application/vnd.vc.candidate-set")})};
}

WorkerResult report(const std::filesystem::path& output, const Json& input, const std::atomic<bool>&, const CpuDiscovery::LogCallback&)
{
    const auto setPath = std::filesystem::path(input.at("candidate_set_path").get<std::string>());
    Json set = Json::parse(std::ifstream(setPath / "candidate-set.json"));
    cv::Mat image = cv::imread((setPath / "candidate-score.png").string(), cv::IMREAD_GRAYSCALE);
    require(!image.empty(), "candidate set has no candidate-score.png");
    std::filesystem::create_directories(output);
    Json index = Json::array();
    int limit = std::min<int>(input.value("max_candidates", 100), set.at("candidates").size());
    for (int i = 0; i < limit; ++i) {
        const auto& c = set.at("candidates").at(i);
        const auto& b = c.at("bbox_uv");
        int pad = input.value("context_pixels", 32);
        cv::Rect r(b.at("x").get<int>() - pad, b.at("y").get<int>() - pad, b.at("width").get<int>() + 2 * pad, b.at("height").get<int>() + 2 * pad);
        r &= cv::Rect(0, 0, image.cols, image.rows);
        auto name = c.at("id").get<std::string>() + ".png";
        writeImage(output / name, image(r));
        index.push_back({{"id", c.at("id")}, {"preview", name}, {"bbox_uv", b}});
    }
    std::ofstream(output / "report.json") << Json{{"kind", "candidate_report_v1"}, {"candidates", index}}.dump(2) << '\n';
    return {
        0,
        {{"operation", "ink_render_candidate_report"}, {"backend", "cpu"}, {"input", input}},
        Json::array({artifact("candidate-report", output, "application/vnd.vc.candidate-report")})};
}

WorkerResult layout(const std::filesystem::path& output, const Json& input, const std::atomic<bool>&, const CpuDiscovery::LogCallback&)
{
    cv::Mat mask = cv::imread(input.at("mask_path").get<std::string>(), cv::IMREAD_GRAYSCALE);
    require(!mask.empty(), "failed to decode mask_path");
    cv::threshold(mask, mask, input.value("threshold", 127), 255, cv::THRESH_BINARY);
    cv::Mat skeleton = cv::Mat::zeros(mask.size(), CV_8U), current = mask.clone(), eroded, opened, temp;
    auto element = cv::getStructuringElement(cv::MORPH_CROSS, {3, 3});
    while (cv::countNonZero(current) > 0) {
        cv::erode(current, eroded, element);
        cv::dilate(eroded, opened, element);
        cv::subtract(current, opened, temp);
        cv::bitwise_or(skeleton, temp, skeleton);
        eroded.copyTo(current);
    }
    cv::Mat labels, stats, centers;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centers, 8, CV_32S);
    Json components = Json::array();
    std::vector<double> ys;
    for (int i = 1; i < n; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA), w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        if (area < 4)
            continue;
        double cy = centers.at<double>(i, 1);
        ys.push_back(cy);
        components.push_back(
            {{"bbox_uv", {{"x", stats.at<int>(i, cv::CC_STAT_LEFT)}, {"y", stats.at<int>(i, cv::CC_STAT_TOP)}, {"width", w}, {"height", h}}},
             {"area", area},
             {"elongation", double(std::max(w, h)) / std::max(1, std::min(w, h))}});
    }
    std::sort(ys.begin(), ys.end());
    int lines = 0;
    double previous = -1e9;
    for (double y : ys)
        if (y - previous > 12) {
            ++lines;
            previous = y;
        } else
            previous = (previous + y) / 2;
    double score = std::min(1.0, components.size() / 20.0) * std::min(1.0, lines / 3.0);
    std::filesystem::create_directories(output);
    writeImage(output / "mask.png", mask);
    writeImage(output / "skeleton.png", skeleton);
    Json result = {{"kind", "text_layout_v1"}, {"text_like_score", score}, {"score_semantics", "heuristic_not_transcription"}, {"line_hypothesis_count", lines}, {"components", components}};
    std::ofstream(output / "layout-analysis.json") << result.dump(2) << '\n';
    return {
        0,
        {{"operation", "text_analyze_layout"}, {"backend", "cpu"}, {"input", input}},
        Json::array({artifact("text-layout", output, "application/vnd.vc.text-layout")})};
}

void runFixedProcess(
    const std::filesystem::path& program,
    const std::vector<std::string>& arguments,
    std::chrono::seconds timeout,
    const std::atomic<bool>& cancelled,
    const CpuDiscovery::LogCallback& log,
    const std::string& label)
{
    QProcess process;
    process.setProgram(QString::fromStdString(program.string()));
    QStringList qtArguments;
    for (const auto& argument : arguments)
        qtArguments.push_back(QString::fromStdString(argument));
    process.setArguments(qtArguments);
    process.setProcessChannelMode(QProcess::MergedChannels);
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    for (const auto& key : environment.keys()) {
        if (key.startsWith("AWS_") || key.startsWith("GOOGLE_") || key.startsWith("AZURE_") || key.startsWith("S3_"))
            environment.remove(key);
    }
    process.setProcessEnvironment(environment);
    process.start();
    if (!process.waitForStarted(10000))
        throw std::runtime_error("failed to start " + label);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (process.state() != QProcess::NotRunning) {
        process.waitForReadyRead(100);
        const auto text = process.readAll().toStdString();
        if (!text.empty())
            log(text);
        if (cancelled.load() || std::chrono::steady_clock::now() >= deadline) {
            process.kill();
            process.waitForFinished();
            throw std::runtime_error(cancelled.load() ? label + " cancelled" : label + " timed out");
        }
    }
    const auto trailing = process.readAll().toStdString();
    if (!trailing.empty())
        log(trailing);
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0)
        throw std::runtime_error(label + " failed with exit code " + std::to_string(process.exitCode()));
}

WorkerResult registeredSurface(
    const CpuDiscoveryConfig& config, const std::filesystem::path& output, const Json& input, const std::atomic<bool>& cancelled, const CpuDiscovery::LogCallback& log)
{
    require(config.analysisPython && config.surfaceBundleAdapter && config.volumeStager, "registered surface service is not configured");
    std::filesystem::create_directories(output);
    const auto importRequest = output / "import-request.json";
    Json importInput = input;
    importInput.erase("volume");  // Only the controlled stager receives the local/remote volume location.
    importInput.erase("surface_path");
    std::ofstream(importRequest) << importInput.dump(2) << '\n';
    runFixedProcess(
        *config.analysisPython,
        {config.surfaceBundleAdapter->string(),
         "import",
         "--request",
         importRequest.string(),
         "--surface",
         input.at("surface_path").get<std::string>(),
         "--output",
         output.string()},
        config.timeout,
        cancelled,
        log,
        "surface bundle import");
    const Json importManifest = Json::parse(std::ifstream(output / "import-manifest.json"));

    auto stageRegion = importManifest.at("required_volume_region_xyz");
    stageRegion["space"] = input.at("coordinate_space");
    const Json stageRequestDocument = {{"source", input.at("volume")}, {"region", stageRegion}};
    const auto stageDirectory = output / "staging";
    std::filesystem::create_directories(stageDirectory);
    const auto stageRequest = stageDirectory / "request.json";
    std::ofstream(stageRequest) << stageRequestDocument.dump(2) << '\n';
    runFixedProcess(
        *config.analysisPython,
        {config.volumeStager->string(), "--request", stageRequest.string(), "--output", stageDirectory.string()},
        config.timeout,
        cancelled,
        log,
        "registered surface volume staging");
    const Json stageManifest = Json::parse(std::ifstream(stageDirectory / "stage-manifest.json"));

    const Json renderRequestDocument =
        {{"surface_bundle", importManifest.at("surface_bundle")},
         {"staged_volume", stageManifest.at("staged_path")},
         {"staged_region_xyz", stageManifest.at("submitted_region_xyz")},
         {"volume_source", stageManifest.at("source")},
         {"array_path", stageManifest.at("array_path")},
         {"scale", stageManifest.at("scale")},
         {"voxel_spacing", stageManifest.at("voxel_spacing")},
         {"voxel_spacing_unit", stageManifest.at("voxel_spacing_unit")},
         {"voxel_spacing_explicit", stageManifest.at("voxel_spacing_explicit")},
         {"origin_xyz", stageManifest.at("origin_xyz")}};
    const auto renderRequest = output / "render-request.json";
    std::ofstream(renderRequest) << renderRequestDocument.dump(2) << '\n';
    runFixedProcess(
        *config.analysisPython,
        {config.surfaceBundleAdapter->string(), "render", "--request", renderRequest.string(), "--output", output.string()},
        config.timeout,
        cancelled,
        log,
        "registered surface rendering");
    require(std::filesystem::is_regular_file(output / "manifest.json"), "registered surface renderer produced no manifest");
    const Json command =
        {{"operation", "surface_render_registered_roi"},
         {"backend", "cpu"},
         {"adapter", config.surfaceBundleAdapter->string()},
         {"surface_artifact", input.at("surface")},
         {"volume_source_kind", input.at("volume").at("kind")}};
    return {0, command, Json::array({artifact("registered-surface", output, "application/vnd.vc.registered-surface+zarr")})};
}

WorkerResult resnet152Inference(
    const CpuDiscoveryConfig& config, const std::filesystem::path& output, const Json& input, const std::atomic<bool>& cancelled, const CpuDiscovery::LogCallback& log)
{
    require(
        config.inkModelPython && config.inkModelAdapter && config.inkModelRepository && config.inkModelCheckpoint && !config.inkModelRepositoryCommit.empty(),
        "ResNet152 ink-model inference is not configured");
    std::filesystem::create_directories(output);
    Json adapterInput = input;
    adapterInput.erase("surface_volume_path");
    adapterInput.erase("surface_volume_media_type");
    adapterInput["repository_commit"] = config.inkModelRepositoryCommit;
    const auto requestPath = output / "request.json";
    std::ofstream(requestPath) << adapterInput.dump(2) << '\n';
    runFixedProcess(
        *config.inkModelPython,
        {config.inkModelAdapter->string(),
         "--request",
         requestPath.string(),
         "--artifact",
         input.at("surface_volume_path").get<std::string>(),
         "--checkpoint",
         config.inkModelCheckpoint->string(),
         "--repository",
         config.inkModelRepository->string(),
         "--output",
         output.string()},
        config.timeout,
        cancelled,
        log,
        "ResNet152 ink-model inference");
    require(std::filesystem::is_regular_file(output / "manifest.json"), "ResNet152 ink-model inference produced no manifest");
    return {
        0,
        {{"operation", "ink_run_resnet152_inference"},
         {"backend", "pinned-local"},
         {"checkpoint_sha256", "36dd0de84b7b7aa6590184192c7415466cd8a1ba7c1e59f42c6373846373c3e0"},
         {"repository_commit", config.inkModelRepositoryCommit}},
        Json::array({artifact("ink-prediction", output, "application/vnd.vc.ink-model-score+zarr")})};
}

WorkerResult surfaceEvidence(
    const std::string& operation,
    const CpuDiscoveryConfig& config,
    const std::filesystem::path& output,
    const Json& input,
    const std::atomic<bool>& cancelled,
    const CpuDiscovery::LogCallback& log)
{
    require(config.analysisPython && config.surfaceBundleAdapter, "surface evidence service is not configured");
    std::filesystem::create_directories(output);
    Json adapterInput = input;
    adapterInput.erase("surface_path");
    adapterInput.erase("surface_media_type");
    const auto requestPath = output / "request.json";
    std::ofstream(requestPath) << adapterInput.dump(2) << '\n';
    const bool geometry = operation == "surface_validate_geometry";
    const bool normalStack = operation == "surface_render_normal_stack";
    const std::string command = geometry ? "geometry" : normalStack ? "normal-stack" : "alignment";
    runFixedProcess(
        *config.analysisPython,
        {config.surfaceBundleAdapter->string(),
         command,
         "--request",
         requestPath.string(),
         "--artifact",
         input.at("surface_path").get<std::string>(),
         "--output",
         output.string()},
        config.timeout,
        cancelled,
        log,
        geometry      ? "surface geometry diagnostics"
        : normalStack ? "ink-model normal-stack rendering"
                      : "surface CT alignment");
    require(std::filesystem::is_regular_file(output / "manifest.json"), operation + " produced no manifest");
    const std::string artifactId = geometry ? "surface-geometry" : normalStack ? "surface-volume" : "surface-ct-alignment";
    const std::string mediaType = geometry      ? "application/vnd.vc.surface-geometry+zarr"
                                  : normalStack ? "application/vnd.vc.surface-volume+zarr"
                                                : "application/vnd.vc.surface-ct-alignment+zarr";
    return {0, {{"operation", operation}, {"backend", "cpu"}, {"adapter", config.surfaceBundleAdapter->string()}, {"source_artifact", input.at("surface")}}, Json::array({artifact(artifactId, output, mediaType)})};
}

WorkerResult structuralEvidence(
    const std::string& operation,
    const CpuDiscoveryConfig& config,
    const std::filesystem::path& output,
    const Json& input,
    const std::atomic<bool>& cancelled,
    const CpuDiscovery::LogCallback& log)
{
    require(config.analysisPython && config.structuralEvidenceAdapter, "structural evidence service is not configured");
    std::filesystem::create_directories(output);
    Json adapterInput = input;
    for (const auto* key :
         {"surface_path",
          "surface_media_type",
          "surface_a_path",
          "surface_a_media_type",
          "surface_b_path",
          "surface_b_media_type",
          "grid_path",
          "grid_media_type"})
        adapterInput.erase(key);
    const auto requestPath = output / "request.json";
    std::ofstream(requestPath) << adapterInput.dump(2) << '\n';
    std::vector<std::string> arguments = {config.structuralEvidenceAdapter->string()};
    std::string command;
    std::string artifactId;
    std::string mediaType;
    if (operation == "text_measure_grid_coherence") {
        command = "grid";
        artifactId = "grid-coherence";
        mediaType = "application/vnd.vc.grid-coherence+zarr";
        arguments.insert(
            arguments.end(),
            {command, "--request", requestPath.string(), "--artifact", input.at("surface_path").get<std::string>(), "--output", output.string()});
    } else if (operation == "ink_compare_registered_predictions") {
        command = "compare";
        artifactId = "structural-comparison";
        mediaType = "application/vnd.vc.structural-comparison+zarr";
        arguments.insert(
            arguments.end(),
            {command,
             "--request",
             requestPath.string(),
             "--artifact-a",
             input.at("surface_a_path").get<std::string>(),
             "--artifact-b",
             input.at("surface_b_path").get<std::string>(),
             "--output",
             output.string()});
    } else {
        command = "fold";
        artifactId = "epoch-fold";
        mediaType = "application/vnd.vc.epoch-fold+json";
        arguments.insert(
            arguments.end(),
            {command,
             "--request",
             requestPath.string(),
             "--artifact",
             input.at("surface_path").get<std::string>(),
             "--grid-artifact",
             input.at("grid_path").get<std::string>(),
             "--output",
             output.string()});
    }
    runFixedProcess(*config.analysisPython, arguments, config.timeout, cancelled, log, operation);
    require(std::filesystem::is_regular_file(output / "manifest.json"), operation + " produced no manifest");
    return {0, {{"operation", operation}, {"backend", "cpu"}, {"adapter", config.structuralEvidenceAdapter->string()}}, Json::array({artifact(artifactId, output, mediaType)})};
}

WorkerResult fusionEvidence(
    const std::string& operation,
    const CpuDiscoveryConfig& config,
    const std::filesystem::path& output,
    const Json& input,
    const std::atomic<bool>& cancelled,
    const CpuDiscovery::LogCallback& log)
{
    require(config.analysisPython && config.evidenceFusionAdapter, "evidence fusion service is not configured");
    std::filesystem::create_directories(output);
    const auto requestPath = output / "request.json";
    std::vector<std::string> arguments = {config.evidenceFusionAdapter->string()};
    std::string artifactId;
    std::string mediaType;
    if (operation == "surface_test_stability") {
        Json adapterInput = input;
        adapterInput.erase("baseline_path");
        adapterInput.erase("baseline_media_type");
        adapterInput.erase("resolved_variants");
        std::ofstream(requestPath) << adapterInput.dump(2) << '\n';
        const auto variantsPath = output / "resolved-variants.json";
        Json paths = Json::array();
        for (const auto& variant : input.at("resolved_variants"))
            paths.push_back(variant.at("path"));
        std::ofstream(variantsPath) << Json{{"paths", paths}}.dump(2) << '\n';
        arguments.insert(
            arguments.end(),
            {"stability",
             "--request",
             requestPath.string(),
             "--baseline",
             input.at("baseline_path").get<std::string>(),
             "--variants",
             variantsPath.string(),
             "--output",
             output.string()});
        artifactId = "surface-stability";
        mediaType = "application/vnd.vc.surface-stability+zarr";
    } else if (operation == "ink_fuse_registered_scores") {
        std::ofstream(requestPath) << input.dump(2) << '\n';
        arguments.insert(arguments.end(), {"ink-fuse", "--request", requestPath.string(),
                                           "--output", output.string()});
        artifactId = "ink-fusion";
        mediaType = "application/vnd.vc.ink-fusion+zarr";
    } else {
        std::ofstream(requestPath) << input.dump(2) << '\n';
        arguments.insert(arguments.end(), {"rank", "--request", requestPath.string(), "--output", output.string()});
        artifactId = "evidence-ranking";
        mediaType = "application/vnd.vc.evidence-ranking+json";
    }
    runFixedProcess(*config.analysisPython, arguments, config.timeout, cancelled, log, operation);
    require(std::filesystem::is_regular_file(output / "manifest.json"), operation + " produced no manifest");
    return {0, {{"operation", operation}, {"backend", "cpu"}, {"adapter", config.evidenceFusionAdapter->string()}}, Json::array({artifact(artifactId, output, mediaType)})};
}

WorkerResult reviewEvidence(
    const std::string& operation,
    const CpuDiscoveryConfig& config,
    const std::filesystem::path& output,
    const Json& input,
    const std::atomic<bool>& cancelled,
    const CpuDiscovery::LogCallback& log)
{
    require(config.analysisPython && config.reviewAdapter, "review artifact service is not configured");
    std::filesystem::create_directories(output);
    Json adapterInput = input;
    for (const auto* key :
         {"ranking_path",
          "ranking_media_type",
          "comparison_path",
          "comparison_media_type",
          "queue_path",
          "queue_media_type",
          "resolved_assessments"})
        adapterInput.erase(key);
    const auto requestPath = output / "request.json";
    std::ofstream(requestPath) << adapterInput.dump(2) << '\n';
    std::vector<std::string> arguments = {config.reviewAdapter->string()};
    std::string artifactId, mediaType;
    if (operation == "review_create_queue") {
        arguments.insert(arguments.end(), {"create", "--request", requestPath.string(), "--ranking", input.at("ranking_path").get<std::string>()});
        if (input.contains("comparison_path"))
            arguments.insert(arguments.end(), {"--comparison", input.at("comparison_path").get<std::string>()});
        arguments.insert(arguments.end(), {"--output", output.string()});
        artifactId = "review-queue";
        mediaType = "application/vnd.vc.review-queue+json";
    } else if (operation == "review_record_assessment") {
        arguments.insert(
            arguments.end(),
            {"assess", "--request", requestPath.string(), "--queue", input.at("queue_path").get<std::string>(), "--output", output.string()});
        artifactId = "review-assessment";
        mediaType = "application/vnd.vc.review-assessment+json";
    } else {
        const auto assessmentsPath = output / "resolved-assessments.json";
        Json paths = Json::array();
        for (const auto& assessment : input.at("resolved_assessments"))
            paths.push_back(assessment.at("path"));
        std::ofstream(assessmentsPath) << Json{{"paths", paths}}.dump(2) << '\n';
        arguments.insert(
            arguments.end(), {"evaluate", "--request", requestPath.string(), "--assessments", assessmentsPath.string(), "--output", output.string()});
        artifactId = "label-evaluation";
        mediaType = "application/vnd.vc.label-evaluation+json";
    }
    runFixedProcess(*config.analysisPython, arguments, config.timeout, cancelled, log, operation);
    require(std::filesystem::is_regular_file(output / "manifest.json"), operation + " produced no manifest");
    return {0, {{"operation", operation}, {"backend", "cpu"}, {"adapter", config.reviewAdapter->string()}}, Json::array({artifact(artifactId, output, mediaType)})};
}

WorkerResult runNnunet(const CpuDiscoveryConfig& config, const std::filesystem::path& output, const Json& input, const std::atomic<bool>& cancelled, const CpuDiscovery::LogCallback& log)
{
    require(
        config.nnunetPython && config.nnunetAdapter && config.volumeStager && config.nnunetModelDir, "nnU-Net service is not configured");
    std::filesystem::create_directories(output);
    Json request = input;
    std::optional<Json> stagingManifest;
    if (input.contains("source")) {
        const auto stageDir = output / "staging";
        std::filesystem::create_directories(stageDir);
        const auto stageRequest = stageDir / "request.json";
        std::ofstream(stageRequest) << input.dump(2) << '\n';
        QProcess stage;
        stage.setProgram(QString::fromStdString(config.nnunetPython->string()));
        stage.setArguments(
            {QString::fromStdString(config.volumeStager->string()),
             "--request",
             QString::fromStdString(stageRequest.string()),
             "--output",
             QString::fromStdString(stageDir.string())});
        stage.setProcessChannelMode(QProcess::MergedChannels);
        stage.start();
        require(stage.waitForStarted(10000), "failed to start controlled volume stager");
        const auto stageDeadline = std::chrono::steady_clock::now() + config.timeout;
        while (stage.state() != QProcess::NotRunning) {
            stage.waitForReadyRead(100);
            const auto text = stage.readAll().toStdString();
            if (!text.empty())
                log(text);
            if (cancelled.load() || std::chrono::steady_clock::now() >= stageDeadline) {
                stage.kill();
                stage.waitForFinished();
                throw std::runtime_error(cancelled.load() ? "volume staging cancelled" : "volume staging timed out");
            }
        }
        require(stage.exitStatus() == QProcess::NormalExit && stage.exitCode() == 0, "controlled volume staging failed");
        stagingManifest = Json::parse(std::ifstream(stageDir / "stage-manifest.json"));
        request.erase("source");
        request.erase("region");
        request["volume_path"] = stagingManifest->at("staged_path");
    }
    request["model_dir"] = config.nnunetModelDir->string();
    const auto requestPath = output / "request.json";
    std::ofstream(requestPath) << request.dump(2) << '\n';
    QProcess process;
    process.setProgram(QString::fromStdString(config.nnunetPython->string()));
    process.setArguments(
        {QString::fromStdString(config.nnunetAdapter->string()),
         "--request",
         QString::fromStdString(requestPath.string()),
         "--output",
         QString::fromStdString(output.string()),
         "--device",
         QString::fromStdString(input.value("device", "cpu"))});
    process.setProcessChannelMode(QProcess::MergedChannels);
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    // The inference subprocess receives only a local staged path. Do not leak
    // ambient object-storage credentials even when the parent has them.
    for (const auto& key : environment.keys()) {
        if (key.startsWith("AWS_") || key.startsWith("GOOGLE_") || key.startsWith("AZURE_") || key.startsWith("S3_"))
            environment.remove(key);
    }
    environment.insert("PYTORCH_ENABLE_MPS_FALLBACK", "0");
    environment.insert("nnUNet_results", QString::fromStdString(config.nnunetModelDir->parent_path().parent_path().string()));
    process.setProcessEnvironment(environment);
    process.start();
    require(process.waitForStarted(10000), "failed to start nnU-Net adapter");
    const auto deadline = std::chrono::steady_clock::now() + config.timeout;
    while (process.state() != QProcess::NotRunning) {
        process.waitForReadyRead(100);
        const auto text = process.readAll().toStdString();
        if (!text.empty())
            log(text);
        if (cancelled.load() || std::chrono::steady_clock::now() >= deadline) {
            process.kill();
            process.waitForFinished();
            throw std::runtime_error(cancelled.load() ? "CPU discovery cancelled" : "nnU-Net adapter timed out");
        }
    }
    require(process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0, "nnU-Net adapter failed");
    require(std::filesystem::exists(output / "manifest.json"), "nnU-Net adapter produced no manifest.json");
    if (stagingManifest) {
        Json manifest = Json::parse(std::ifstream(output / "manifest.json"));
        manifest["staging"] = *stagingManifest;
        manifest["submitted_region_xyz"] = stagingManifest->at("submitted_region_xyz");
        manifest["array_slices_zyx"] = stagingManifest->at("array_slices_zyx");
        manifest["scale"] = stagingManifest->at("scale");
        manifest["array_path"] = stagingManifest->at("array_path");
        manifest["voxel_spacing"] = stagingManifest->at("voxel_spacing");
        manifest["origin_xyz"] = stagingManifest->at("origin_xyz");
        const Json spatialMetadata =
            {{"axes", Json::array({"z", "y", "x"})},
             {"outputs", Json::array({"surface-probability.npy", "surface-probability.tif", "surface-mask.npy", "surface-mask.tif"})},
             {"submitted_region_xyz", stagingManifest->at("submitted_region_xyz")},
             {"array_slices_zyx", stagingManifest->at("array_slices_zyx")},
             {"scale", stagingManifest->at("scale")},
             {"array_path", stagingManifest->at("array_path")},
             {"voxel_spacing", stagingManifest->at("voxel_spacing")},
             {"origin_xyz", stagingManifest->at("origin_xyz")}};
        std::ofstream(output / "spatial-metadata.json") << spatialMetadata.dump(2) << '\n';
        manifest["spatial_metadata"] = "spatial-metadata.json";
        std::ofstream(output / "manifest.json") << manifest.dump(2) << '\n';
    }
    return {
        0,
        {{"operation", "volume_run_segmentation"}, {"backend", input.value("device", "cpu")}, {"adapter", config.nnunetAdapter->string()}, {"input", input}},
        Json::array({artifact("volume-segmentation", output, "application/vnd.vc.volume-segmentation")})};
}

WorkerResult runExternalExemplar(
    const std::string& operation,
    const std::string& artifactId,
    const std::filesystem::path& executable,
    std::chrono::seconds timeout,
    const std::filesystem::path& output,
    const Json& input,
    const std::atomic<bool>& cancelled,
    const CpuDiscovery::LogCallback& log)
{
    std::filesystem::create_directories(output);
    const auto requestPath = output / "request.json";
    std::ofstream(requestPath) << input.dump(2) << '\n';
    QProcess process;
    process.setProgram(QString::fromStdString(executable.string()));
    process.setArguments(
        {"--request", QString::fromStdString(requestPath.string()), "--output", QString::fromStdString(output.string()), "--device", "cpu"});
    process.setProcessChannelMode(QProcess::MergedChannels);
    process.start();
    require(process.waitForStarted(10000), "failed to start pinned " + operation + " adapter");
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (process.state() != QProcess::NotRunning) {
        process.waitForReadyRead(100);
        const auto text = process.readAll().toStdString();
        if (!text.empty())
            log(text);
        if (cancelled.load() || std::chrono::steady_clock::now() >= deadline) {
            process.kill();
            process.waitForFinished();
            throw std::runtime_error(cancelled.load() ? "CPU discovery cancelled" : operation + " adapter timed out");
        }
    }
    require(process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0, operation + " adapter failed");
    require(std::filesystem::exists(output / "manifest.json"), operation + " adapter produced no manifest.json");
    return {
        0,
        {{"operation", operation}, {"backend", "cpu"}, {"executable", executable.string()}, {"input", input}},
        Json::array({artifact(artifactId, output, "application/vnd.vc." + artifactId)})};
}

WorkerResult dinov3(const CpuDiscoveryConfig& config, const std::filesystem::path& output, const Json& input, const std::atomic<bool>& cancelled, const CpuDiscovery::LogCallback& log)
{
    require(config.dinov3Executable.has_value(), "DINOv3 CPU adapter is not configured");
    return runExternalExemplar("dinov3_exemplar_search", "dinov3-exemplar", *config.dinov3Executable, config.timeout, output, input, cancelled, log);
}

WorkerResult dinovol(const CpuDiscoveryConfig& config, const std::filesystem::path& output, const Json& input, const std::atomic<bool>& cancelled, const CpuDiscovery::LogCallback& log)
{
    require(
        config.dinovolPython && config.dinovolAdapter && config.dinovolRepository && config.dinovolCheckpoint &&
            !config.dinovolRepositoryCommit.empty(),
        "Dinovol MPS adapter is not configured");
    std::filesystem::create_directories(output);
    Json adapterInput = input;
    adapterInput.erase("surface_path");
    adapterInput.erase("surface_media_type");
    adapterInput["repository_commit"] = config.dinovolRepositoryCommit;
    const auto requestPath = output / "request.json";
    std::ofstream(requestPath) << adapterInput.dump(2) << '\n';
    runFixedProcess(
        *config.dinovolPython,
        {config.dinovolAdapter->string(),
         "--request",
         requestPath.string(),
         "--surface",
         input.at("surface_path").get<std::string>(),
         "--checkpoint",
         config.dinovolCheckpoint->string(),
         "--repository",
         config.dinovolRepository->string(),
         "--output",
         output.string(),
         "--device",
         "mps"},
        config.timeout,
        cancelled,
        log,
        "Dinovol MPS exemplar search");
    require(std::filesystem::is_regular_file(output / "manifest.json"), "Dinovol adapter produced no manifest");
    return {
        0,
        {{"operation", "dinovol_exemplar_search"},
         {"backend", "mps"},
         {"checkpoint_sha256", "e041ca870dd2570f8a44d1dd26db1197b3f74121f62023bc774fbc9d40e51a59"},
         {"repository_commit", config.dinovolRepositoryCommit}},
        Json::array({artifact("dinovol-exemplar", output, "application/vnd.vc.dinovol-exemplar")})};
}
}  // namespace

CpuDiscovery::CpuDiscovery(CpuDiscoveryConfig config) : config_(std::move(config))
{
    config_.workRoot = std::filesystem::absolute(config_.workRoot).lexically_normal();
    std::filesystem::create_directories(config_.workRoot);
    auto canonicalFile = [](std::optional<std::filesystem::path>& value) {
        if (!value)
            return;
        std::error_code error;
        auto path = std::filesystem::weakly_canonical(*value, error);
        if (error || !std::filesystem::is_regular_file(path))
            value.reset();
        else
            value = path;
    };
    if (config_.nnunetPython) {
        auto path = std::filesystem::absolute(*config_.nnunetPython).lexically_normal();
        if (!std::filesystem::is_regular_file(path))
            config_.nnunetPython.reset();
        else
            config_.nnunetPython = path;
    }
    canonicalFile(config_.nnunetAdapter);
    canonicalFile(config_.volumeStager);
    if (config_.analysisPython) {
        auto path = std::filesystem::absolute(*config_.analysisPython).lexically_normal();
        if (!std::filesystem::is_regular_file(path))
            config_.analysisPython.reset();
        else
            config_.analysisPython = path;
    }
    canonicalFile(config_.surfaceBundleAdapter);
    canonicalFile(config_.structuralEvidenceAdapter);
    canonicalFile(config_.evidenceFusionAdapter);
    canonicalFile(config_.reviewAdapter);
    if (config_.nnunetModelDir) {
        std::error_code error;
        auto path = std::filesystem::weakly_canonical(*config_.nnunetModelDir, error);
        if (error || !std::filesystem::is_directory(path) || !std::filesystem::is_regular_file(path / "fold_0" / "checkpoint_best.pth"))
            config_.nnunetModelDir.reset();
        else
            config_.nnunetModelDir = path;
    }
    if (config_.dinov3Executable) {
        std::error_code ec;
        auto p = std::filesystem::weakly_canonical(*config_.dinov3Executable, ec);
        if (ec || !std::filesystem::is_regular_file(p))
            config_.dinov3Executable.reset();
        else
            config_.dinov3Executable = p;
    }
    if (config_.dinovolExecutable) {
        std::error_code ec;
        auto p = std::filesystem::weakly_canonical(*config_.dinovolExecutable, ec);
        if (ec || !std::filesystem::is_regular_file(p))
            config_.dinovolExecutable.reset();
        else
            config_.dinovolExecutable = p;
    }
    if (config_.dinovolPython) {
        auto path = std::filesystem::absolute(*config_.dinovolPython).lexically_normal();
        if (!std::filesystem::is_regular_file(path))
            config_.dinovolPython.reset();
        else
            config_.dinovolPython = path;  // Preserve a virtualenv symlink and its sys.prefix.
    }
    canonicalFile(config_.dinovolAdapter);
    canonicalFile(config_.dinovolCheckpoint);
    if (config_.dinovolRepository) {
        std::error_code ec;
        auto p = std::filesystem::weakly_canonical(*config_.dinovolRepository, ec);
        if (ec || !std::filesystem::is_directory(p / ".git"))
            config_.dinovolRepository.reset();
        else
            config_.dinovolRepository = p;
    }
}

Json CpuDiscovery::validate(const std::string& operation, const Json& request) const
{
    validateRequestId(request);
    Json out = request;
    out["profile"] = request.value("profile", operation + "-cpu-v1");
    if (operation == "volume_run_segmentation") {
        require(nnunetAvailable(), "nnU-Net segmentation service is not configured");
        const bool hasPath = request.contains("volume_path");
        const bool hasSource = request.contains("source");
        require(hasPath != hasSource, "provide exactly one of volume_path or source");
        if (hasPath) {
            out["volume_path"] = existingAbsolute(request, "volume_path").string();
        } else {
            const auto& source = request.at("source");
            require(source.is_object() && source.contains("kind"), "source.kind is required");
            const auto kind = source.at("kind").get<std::string>();
            if (kind == "local_zarr") {
                out["source"]["path"] = existingAbsolute(source, "path", true).string();
            } else if (kind == "remote_zarr") {
                require(
                    source.contains("uri") && source.at("uri").is_string() && allowedRemoteVolume(source.at("uri").get<std::string>()),
                    "remote Zarr URI is outside the public allowlist");
            } else {
                throw fastmcpp::ValidationError("source.kind must be local_zarr or remote_zarr");
            }
            const auto arrayPath = source.value("array_path", "0");
            require(
                !arrayPath.empty() && arrayPath.size() <= 128 && arrayPath.front() != '/' && arrayPath.find("..") == std::string::npos,
                "source.array_path must be a relative Zarr key without traversal");
            require(request.contains("region") && request.at("region").is_object(), "a bounded XYZ region is required for Zarr input");
            const auto& region = request.at("region");
            for (const auto* key : {"x", "y", "z", "width", "height", "depth"})
                require(region.contains(key) && region.at(key).is_number_integer(), std::string("region.") + key + " is required");
            const auto x = region.at("x").get<long long>();
            const auto y = region.at("y").get<long long>();
            const auto z = region.at("z").get<long long>();
            const auto width = region.at("width").get<long long>();
            const auto height = region.at("height").get<long long>();
            const auto depth = region.at("depth").get<long long>();
            require(x >= 0 && y >= 0 && z >= 0, "Zarr region origin must be non-negative");
            require(
                width > 0 && height > 0 && depth > 0 && width <= 256 && height <= 256 && depth <= 256,
                "Zarr region exceeds 256 voxels per axis");
            require(width * height * depth <= 16777216, "Zarr region exceeds 16M voxels");
            const auto space = region.value("space", "");
            require(space == "ct_l0_xyz" || space == "ct_l2_xyz", "region.space must be ct_l0_xyz or ct_l2_xyz");
        }
        require(request.value("model", "vc-surface-nnunet-058") == "vc-surface-nnunet-058", "unsupported segmentation model");
        const auto device = request.value("device", "cpu");
        require(device == "cpu" || device == "mps", "device must be cpu or mps");
    } else if (operation == "surface_render_registered_roi") {
        require(surfaceBundleAvailable(), "registered surface service is not configured");
        require(request.contains("surface") && request.at("surface").is_object(), "surface artifact reference is required");
        require(
            request.value("surface_media_type", "") == "application/vnd.volume-cartographer.tifxyz",
            "surface artifact must be a VC TIFXYZ surface");
        out["surface_path"] = existingAbsolute(request, "surface_path", true).string();
        require(request.contains("volume") && request.at("volume").is_object(), "volume source is required");
        const auto& volume = request.at("volume");
        require(volume.contains("kind") && volume.at("kind").is_string(), "volume.kind is required");
        const auto kind = volume.at("kind").get<std::string>();
        if (kind == "local_zarr")
            out["volume"]["path"] = existingAbsolute(volume, "path", true).string();
        else if (kind == "remote_zarr")
            require(
                volume.contains("uri") && volume.at("uri").is_string() && allowedRemoteVolume(volume.at("uri").get<std::string>()),
                "remote Zarr URI is outside the public allowlist");
        else
            throw fastmcpp::ValidationError("volume.kind must be local_zarr or remote_zarr");
        const auto arrayPath = volume.value("array_path", "0");
        require(
            !arrayPath.empty() && arrayPath.size() <= 128 && arrayPath.front() != '/' && arrayPath.find("..") == std::string::npos,
            "volume.array_path must be a relative Zarr key without traversal");
        const auto space = request.value("coordinate_space", "");
        require(space == "ct_l0_xyz" || space == "ct_l2_xyz", "coordinate_space must be ct_l0_xyz or ct_l2_xyz");
        if (request.contains("uv_region")) {
            const auto& region = request.at("uv_region");
            for (const auto* key : {"u", "v", "width", "height"})
                require(region.contains(key) && region.at(key).is_number_integer(), std::string("uv_region.") + key + " is required");
            const auto width = region.at("width").get<long long>();
            const auto height = region.at("height").get<long long>();
            require(
                region.at("u").get<long long>() >= 0 && region.at("v").get<long long>() >= 0 && width > 0 && height > 0 && width <= 8192 &&
                    height <= 8192 && width * height <= 16777216,
                "uv_region is invalid or exceeds 16M pixels");
        }
    } else if (
        operation == "surface_validate_geometry" || operation == "surface_measure_volume_alignment" ||
        operation == "surface_render_normal_stack") {
        require(surfaceBundleAvailable(), "surface evidence service is not configured");
        require(request.contains("surface") && request.at("surface").is_object(), "registered surface artifact reference is required");
        require(
            request.value("surface_media_type", "") == "application/vnd.vc.registered-surface+zarr",
            "surface artifact must be a registered surface bundle");
        out["surface_path"] = existingAbsolute(request, "surface_path", true).string();
        require(
            std::filesystem::is_regular_file(std::filesystem::path(out.at("surface_path").get<std::string>()) / "manifest.json"),
            "registered surface artifact has no manifest.json");
        if (operation == "surface_measure_volume_alignment") {
            const int maximumOffset = request.value("maximum_offset_voxels", 2);
            require(maximumOffset >= 1 && maximumOffset <= 16, "maximum_offset_voxels must be from 1 to 16");
        }
        if (operation == "surface_render_normal_stack") {
            const auto profile = request.value("model_profile", "");
            require(
                profile == "timesformer-26" || profile == "resnet152-3d-decoder-62",
                "model_profile must be timesformer-26 or resnet152-3d-decoder-62");
            const double step = request.value("layer_step_voxels", 1.0);
            require(std::isfinite(step) && step > 0 && step <= 4, "layer_step_voxels must be finite, positive, and at most 4");
        }
    } else if (
        operation == "text_measure_grid_coherence" || operation == "ink_compare_registered_predictions" ||
        operation == "text_epoch_fold_structure") {
        require(structuralEvidenceAvailable(), "structural evidence service is not configured");
        auto registeredArtifact = [&](const char* key) {
            const std::string pathKey = std::string(key) + "_path";
            const std::string mediaKey = std::string(key) + "_media_type";
            require(request.contains(key) && request.at(key).is_object(), std::string(key) + " artifact reference is required");
            require(
                request.value(mediaKey, "") == "application/vnd.vc.registered-surface+zarr",
                std::string(key) + " must reference a registered surface artifact");
            out[pathKey] = existingAbsolute(request, pathKey.c_str(), true).string();
        };
        if (operation == "ink_compare_registered_predictions") {
            registeredArtifact("surface_a");
            registeredArtifact("surface_b");
        } else {
            registeredArtifact("surface");
        }
        const auto polarity = request.value("polarity", "bright");
        require(polarity == "bright" || polarity == "dark", "polarity must be bright or dark");
        if (operation == "text_epoch_fold_structure") {
            require(request.contains("grid") && request.at("grid").is_object(), "grid artifact reference is required");
            require(
                request.value("grid_media_type", "") == "application/vnd.vc.grid-coherence+zarr",
                "grid must reference a grid-coherence artifact");
            out["grid_path"] = existingAbsolute(request, "grid_path", true).string();
            const double tolerance = request.value("period_tolerance", 0.10);
            const int steps = request.value("period_steps", 41);
            const int bins = request.value("phase_bins", 64);
            require(
                std::isfinite(tolerance) && tolerance > 0 && tolerance <= 0.25, "period_tolerance must be greater than 0 and at most 0.25");
            require(steps >= 9 && steps <= 101, "period_steps must be from 9 to 101");
            require(bins >= 16 && bins <= 256, "phase_bins must be from 16 to 256");
        } else {
            for (const auto* key : {"letter_period_mm", "line_period_mm", "column_period_mm"})
                if (request.contains(key)) {
                    const auto value = request.at(key).get<double>();
                    require(std::isfinite(value) && value > 0 && value <= 200, std::string(key) + " must be greater than 0 and at most 200");
                }
            for (const auto* key : {"window_width_mm", "window_height_mm", "step_mm", "minimum_cycles"})
                if (request.contains(key)) {
                    const auto value = request.at(key).get<double>();
                    require(std::isfinite(value) && value > 0 && value <= 200, std::string(key) + " must be finite and positive");
                }
        }
        const int trials = request.value("null_trials", 16);
        require(trials >= 4 && trials <= 64, "null_trials must be from 4 to 64");
        const auto seed = request.value("null_seed", 0LL);
        require(seed >= 0 && seed <= 2147483647LL, "null_seed must be from 0 to 2147483647");
    } else if (operation == "surface_test_stability" || operation == "surface_rank_evidence" ||
               operation == "ink_fuse_registered_scores") {
        require(evidenceFusionAvailable(), "evidence fusion service is not configured");
        auto canonicalArtifactPath = [](const Json& value, const std::string& label) {
            require(value.is_string(), label + " path is missing");
            std::filesystem::path path = value.get<std::string>();
            require(path.is_absolute(), label + " path must be absolute");
            std::error_code error;
            path = std::filesystem::weakly_canonical(path, error);
            require(!error && std::filesystem::is_directory(path), label + " artifact no longer exists");
            return path.string();
        };
        if (operation == "surface_test_stability") {
            require(request.contains("baseline") && request.at("baseline").is_object(), "baseline artifact reference is required");
            require(
                request.value("baseline_media_type", "") == "application/vnd.vc.registered-surface+zarr",
                "baseline must reference a registered surface artifact");
            out["baseline_path"] = canonicalArtifactPath(request.at("baseline_path"), "baseline");
            require(
                request.contains("resolved_variants") && request.at("resolved_variants").is_array() &&
                    !request.at("resolved_variants").empty() && request.at("resolved_variants").size() <= 7,
                "stability requires from 1 to 7 registered variants");
            for (std::size_t index = 0; index < out.at("resolved_variants").size(); ++index) {
                auto& variant = out["resolved_variants"][index];
                require(
                    variant.value("media_type", "") == "application/vnd.vc.registered-surface+zarr",
                    "every stability variant must be a registered surface artifact");
                variant["path"] = canonicalArtifactPath(variant.at("path"), "variant");
            }
            for (const auto* key : {"displacement_scale_mm", "normal_angle_scale_degrees", "signal_scale"})
                if (request.contains(key)) {
                    const double value = request.at(key).get<double>();
                    require(std::isfinite(value) && value > 0 && value <= 1000, std::string(key) + " must be finite and positive");
                }
        } else if (operation == "ink_fuse_registered_scores") {
            require(request.contains("ink_model") && request.at("ink_model").is_object(),
                    "ink_model artifact reference is required");
            require(request.contains("dinovol") && request.at("dinovol").is_object(),
                    "dinovol artifact reference is required");
            require(request.value("ink_model_media_type", "") ==
                        "application/vnd.vc.ink-model-score+zarr",
                    "ink_model must reference an ink-prediction artifact");
            require(request.value("dinovol_media_type", "") ==
                        "application/vnd.vc.dinovol-exemplar",
                    "dinovol must reference a dinovol-exemplar artifact");
            out["ink_model_path"] = canonicalArtifactPath(request.at("ink_model_path"), "ink_model");
            out["dinovol_path"] = canonicalArtifactPath(request.at("dinovol_path"), "dinovol");
            if (request.contains("stability")) {
                require(request.value("stability_media_type", "") ==
                            "application/vnd.vc.surface-stability+zarr",
                        "stability must reference a surface-stability artifact");
                out["stability_path"] = canonicalArtifactPath(request.at("stability_path"), "stability");
            }
            if (request.contains("weights")) {
                require(request.at("weights").is_object(), "fusion weights must be an object");
                for (const auto& [key, value] : request.at("weights").items()) {
                    require(key == "ink_model" || key == "dinovol" || key == "stability",
                            "unsupported fusion weight: " + key);
                    const double weight = value.get<double>();
                    require(std::isfinite(weight) && weight >= 0 && weight <= 100,
                            "fusion weights must be from 0 to 100");
                    require(key != "stability" || request.contains("stability") || weight == 0,
                            "a non-zero stability weight requires a stability artifact");
                }
            }
        } else {
            require(
                request.contains("resolved_candidates") && request.at("resolved_candidates").is_array() &&
                    !request.at("resolved_candidates").empty() && request.at("resolved_candidates").size() <= 16,
                "evidence ranking requires from 1 to 16 candidates");
            std::set<std::string> ids;
            const std::map<std::string, std::string> expected =
                {{"geometry", "application/vnd.vc.surface-geometry+zarr"},
                 {"alignment", "application/vnd.vc.surface-ct-alignment+zarr"},
                 {"grid", "application/vnd.vc.grid-coherence+zarr"},
                 {"stability", "application/vnd.vc.surface-stability+zarr"}};
            for (auto& candidate : out["resolved_candidates"]) {
                const auto id = candidate.at("id").get<std::string>();
                require(
                    !id.empty() && id.size() <= 64 &&
                        std::all_of(id.begin(), id.end(), [](unsigned char c) { return std::isalnum(c) || c == '.' || c == '_' || c == '-'; }) &&
                        ids.insert(id).second,
                    "candidate ids must be unique, bounded, and contain only letters, digits, dot, underscore, or hyphen");
                auto& resolved = candidate.at("resolved");
                for (const auto* requiredKey : {"geometry", "alignment", "grid"})
                    require(resolved.contains(requiredKey), std::string("candidate is missing ") + requiredKey + " evidence");
                for (const auto& [key, mediaType] : expected) {
                    if (!resolved.contains(key))
                        continue;
                    require(resolved.value(key + "_media_type", "") == mediaType, "candidate " + key + " artifact has the wrong media type");
                    resolved[key] = canonicalArtifactPath(resolved.at(key), "candidate " + key);
                }
            }
            if (request.contains("weights"))
                for (const auto& [key, value] : request.at("weights").items()) {
                    require(expected.contains(key), "unsupported evidence weight: " + key);
                    const double weight = value.get<double>();
                    require(std::isfinite(weight) && weight >= 0 && weight <= 100, "evidence weights must be from 0 to 100");
                }
        }
    } else if (operation == "review_create_queue" || operation == "review_record_assessment" || operation == "metric_evaluate_labels") {
        require(reviewAvailable(), "review artifact service is not configured");
        auto canonicalReviewPath = [](const Json& value, const std::string& label) {
            require(value.is_string(), label + " path is missing");
            std::filesystem::path path = value.get<std::string>();
            require(path.is_absolute(), label + " path must be absolute");
            std::error_code error;
            path = std::filesystem::weakly_canonical(path, error);
            require(!error && std::filesystem::is_directory(path), label + " artifact no longer exists");
            return path.string();
        };
        if (operation == "review_create_queue") {
            require(
                request.value("ranking_media_type", "") == "application/vnd.vc.evidence-ranking+json",
                "ranking must reference an evidence-ranking artifact");
            out["ranking_path"] = canonicalReviewPath(request.at("ranking_path"), "ranking");
            if (request.contains("comparison")) {
                require(
                    request.value("comparison_media_type", "") == "application/vnd.vc.structural-comparison+zarr",
                    "comparison must reference a structural-comparison artifact");
                out["comparison_path"] = canonicalReviewPath(request.at("comparison_path"), "comparison");
            }
            const int maximum = request.value("max_items", 50);
            require(maximum >= 1 && maximum <= 100, "max_items must be from 1 to 100");
            const double percentile = request.value("divergence_percentile", 90.0);
            require(std::isfinite(percentile) && percentile >= 50 && percentile <= 99.9, "divergence_percentile must be from 50 to 99.9");
        } else if (operation == "review_record_assessment") {
            require(
                request.value("queue_media_type", "") == "application/vnd.vc.review-queue+json",
                "queue must reference a review-queue artifact");
            out["queue_path"] = canonicalReviewPath(request.at("queue_path"), "queue");
            require(request.contains("reviewer_id") && request.at("reviewer_id").is_string(), "reviewer_id is required");
            const auto reviewer = request.at("reviewer_id").get<std::string>();
            require(
                !reviewer.empty() && reviewer.size() <= 64 &&
                    std::all_of(
                        reviewer.begin(), reviewer.end(), [](unsigned char c) { return std::isalnum(c) || c == '.' || c == '_' || c == '-'; }),
                "reviewer_id must be bounded and contain only letters, digits, dot, underscore, or hyphen");
            require(
                request.contains("assessments") && request.at("assessments").is_array() && !request.at("assessments").empty() &&
                    request.at("assessments").size() <= 100,
                "assessments must contain from 1 to 100 records");
        } else {
            require(
                request.contains("resolved_assessments") && request.at("resolved_assessments").is_array() &&
                    !request.at("resolved_assessments").empty() && request.at("resolved_assessments").size() <= 16,
                "evaluation requires from 1 to 16 assessment artifacts");
            for (auto& assessment : out["resolved_assessments"]) {
                require(
                    assessment.value("media_type", "") == "application/vnd.vc.review-assessment+json",
                    "every evaluation input must be a review-assessment artifact");
                assessment["path"] = canonicalReviewPath(assessment.at("path"), "assessment");
            }
        }
    } else if (operation == "vc_render_surface_diagnostics")
        out["surface_volume_path"] = existingAbsolute(request, "surface_volume_path").string();
    else if (operation == "ink_compute_classical_features")
        out["diagnostics_path"] = existingAbsolute(request, "diagnostics_path", true).string();
    else if (operation == "ink_find_candidate_regions")
        out["score_path"] = existingAbsolute(request, "score_path").string();
    else if (operation == "ink_render_candidate_report")
        out["candidate_set_path"] = existingAbsolute(request, "candidate_set_path", true).string();
    else if (operation == "text_analyze_layout")
        out["mask_path"] = existingAbsolute(request, "mask_path").string();
    else if (operation == "dinov3_exemplar_search") {
        require(dinov3Available(), "DINOv3 CPU adapter is not configured");
        out["image_path"] = existingAbsolute(request, "image_path").string();
        out["repository_path"] = existingAbsolute(request, "repository_path", true).string();
        out["weights_path"] = existingAbsolute(request, "weights_path").string();
        require(
            request.contains("repository_commit") && request.at("repository_commit").is_string() &&
                request.at("repository_commit").get<std::string>().size() == 40,
            "repository_commit must be a 40-character pinned Git commit");
        require(
            request.contains("weights_sha256") && request.at("weights_sha256").is_string() &&
                request.at("weights_sha256").get<std::string>().size() == 64,
            "weights_sha256 must be a 64-character lowercase SHA-256 digest");
        require(
            request.contains("positive_examples") && request.at("positive_examples").is_array() && !request.at("positive_examples").empty(),
            "positive_examples is required");
    } else if (operation == "ink_run_resnet152_inference") {
        require(inkModelAvailable(), "ResNet152 ink-model inference is not configured");
        require(
            request.contains("surface_volume") && request.at("surface_volume").is_object(), "surface_volume artifact reference is required");
        require(
            request.value("surface_volume_media_type", "") == "application/vnd.vc.surface-volume+zarr",
            "surface_volume must reference a supported surface-volume artifact");
        out["surface_volume_path"] = existingAbsolute(request, "surface_volume_path", true).string();
        require(request.value("model_profile", "") == "resnet152-3d-decoder-62", "only resnet152-3d-decoder-62 is supported");
        const auto device = request.value("device", "mps");
        require(device == "cpu" || device == "mps" || device == "cuda", "device must be cpu, mps, or cuda");
        const int tile = request.value("tile_size", 64);
        const int stride = request.value("stride", std::max(1, tile / 2));
        require((tile == 64 || tile == 128 || tile == 256) && stride >= 1 && stride <= tile, "invalid ink-model tile_size or stride");
    } else if (operation == "dinovol_exemplar_search") {
        require(dinovolAvailable(), "Dinovol MPS adapter is not configured");
        require(request.contains("surface") && request.at("surface").is_object(), "registered surface artifact reference is required");
        require(
            request.value("surface_media_type", "") == "application/vnd.vc.registered-surface+zarr",
            "surface must reference a registered-surface artifact");
        out["surface_path"] = existingAbsolute(request, "surface_path", true).string();
        require(
            request.contains("positive_examples") && request.at("positive_examples").is_array() && !request.at("positive_examples").empty(),
            "positive_examples is required");
    } else
        throw fastmcpp::ValidationError("unknown CPU discovery operation");
    return out;
}

WorkerResult CpuDiscovery::run(const std::string& operation, const std::string& jobId, const Json& normalized, const std::atomic<bool>& cancelled, LogCallback log) const
{
    auto output = config_.workRoot / jobId / operation;
    if (operation == "volume_run_segmentation")
        return runNnunet(config_, output, normalized, cancelled, log);
    if (operation == "surface_render_registered_roi")
        return registeredSurface(config_, output, normalized, cancelled, log);
    if (operation == "surface_validate_geometry" || operation == "surface_measure_volume_alignment" ||
        operation == "surface_render_normal_stack")
        return surfaceEvidence(operation, config_, output, normalized, cancelled, log);
    if (operation == "text_measure_grid_coherence" || operation == "ink_compare_registered_predictions" ||
        operation == "text_epoch_fold_structure")
        return structuralEvidence(operation, config_, output, normalized, cancelled, log);
    if (operation == "surface_test_stability" || operation == "surface_rank_evidence" ||
        operation == "ink_fuse_registered_scores")
        return fusionEvidence(operation, config_, output, normalized, cancelled, log);
    if (operation == "review_create_queue" || operation == "review_record_assessment" || operation == "metric_evaluate_labels")
        return reviewEvidence(operation, config_, output, normalized, cancelled, log);
    if (operation == "vc_render_surface_diagnostics")
        return diagnostics(output, normalized, cancelled, log);
    if (operation == "ink_compute_classical_features")
        return features(output, normalized, cancelled, log);
    if (operation == "ink_find_candidate_regions")
        return candidates(output, normalized, cancelled, log);
    if (operation == "ink_render_candidate_report")
        return report(output, normalized, cancelled, log);
    if (operation == "text_analyze_layout")
        return layout(output, normalized, cancelled, log);
    if (operation == "dinov3_exemplar_search")
        return dinov3(config_, output, normalized, cancelled, log);
    if (operation == "ink_run_resnet152_inference")
        return resnet152Inference(config_, output, normalized, cancelled, log);
    if (operation == "dinovol_exemplar_search")
        return dinovol(config_, output, normalized, cancelled, log);
    throw std::runtime_error("unknown CPU discovery operation");
}
}  // namespace vc::mcp
