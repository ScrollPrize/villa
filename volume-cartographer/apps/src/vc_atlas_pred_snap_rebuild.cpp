#include "vc/atlas/Atlas.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kEpsilon = 1.0e-12;
constexpr double kOutwardWindingLimit = 0.5;
constexpr double kInwardWindingLimit = 0.25;
constexpr double kInwardFirstHitWeight = 4.0;
constexpr int kMaxTraceSteps = 1024;

struct Options {
    fs::path atlasDir;
    fs::path projectVolpkgJson;
    fs::path volpkgRoot;
    fs::path lasagnaManifest;
    fs::path fiberFilter;
    fs::path debugImagesDir;
    bool dryRun = false;
    bool overwriteManual = false;
    bool verbose = false;
    bool traceSamples = false;
};

struct TraceStats {
    int samples = 0;
    int missingPredDt = 0;
    double lastWinding = 0.0;
    std::optional<double> firstInsideWinding;
    std::optional<double> firstInsidePredDt;
    std::optional<cv::Vec3d> firstInsidePoint;
    std::optional<double> maxInsidePredDt;
    std::optional<double> maxInsideWinding;
    std::optional<cv::Vec3d> maxInsidePoint;
    std::optional<cv::Vec3d> lastPoint;
};

void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0
        << " <atlas_dir> <project.volpkg.json>"
        << " [--lasagna-manifest=manifest.lasagna.json]"
        << " [--fiber=fibers/name.json]"
        << " [--debug-images-dir=dir]"
        << " [--dry-run] [--overwrite-manual] [--verbose] [--trace-samples]\n\n"
        << "Regenerates atlas pred-snap attachments and prints per-control search\n"
        << "diagnostics. Existing manual pred-snap records are preserved unless\n"
        << "--overwrite-manual is passed. Auto/null records are regenerated.\n"
        << "The second positional argument must be a *.volpkg.json file, not a\n"
        << "volume package directory. The selected Lasagna dataset from that file is\n"
        << "used unless --lasagna-manifest is supplied.\n";
}

Options parseArgs(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if (arg == "--dry-run") {
            options.dryRun = true;
            continue;
        }
        if (arg == "--overwrite-manual") {
            options.overwriteManual = true;
            continue;
        }
        if (arg == "--verbose") {
            options.verbose = true;
            continue;
        }
        if (arg == "--trace-samples") {
            options.traceSamples = true;
            options.verbose = true;
            continue;
        }
        if (arg.rfind("--lasagna-manifest=", 0) == 0) {
            options.lasagnaManifest = arg.substr(std::string("--lasagna-manifest=").size());
            continue;
        }
        if (arg.rfind("--fiber=", 0) == 0) {
            options.fiberFilter = fs::path(arg.substr(std::string("--fiber=").size()))
                                      .lexically_normal();
            continue;
        }
        if (arg.rfind("--debug-images-dir=", 0) == 0) {
            options.debugImagesDir = arg.substr(std::string("--debug-images-dir=").size());
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            throw std::invalid_argument("unknown option: " + arg);
        }
        if (options.atlasDir.empty()) {
            options.atlasDir = arg;
        } else if (options.projectVolpkgJson.empty()) {
            options.projectVolpkgJson = arg;
        } else {
            throw std::invalid_argument("too many positional arguments");
        }
    }
    if (options.atlasDir.empty()) {
        throw std::invalid_argument("missing atlas_dir");
    }
    if (options.projectVolpkgJson.empty()) {
        throw std::invalid_argument("missing project.volpkg.json");
    }
    if (!fs::is_regular_file(options.projectVolpkgJson) ||
        options.projectVolpkgJson.filename().string().find(".volpkg.json") == std::string::npos) {
        throw std::invalid_argument("second positional argument must be a *.volpkg.json file");
    }
    return options;
}

void resolveProjectContext(Options& options)
{
    const auto pkg = VolumePkg::load(options.projectVolpkgJson);
    if (!pkg) {
        throw std::runtime_error("failed to load project file: " +
                                 options.projectVolpkgJson.string());
    }
    options.volpkgRoot = fs::path(pkg->getVolpkgDirectory());
    if (options.volpkgRoot.empty()) {
        throw std::runtime_error("failed to resolve volpkg root from " +
                                 options.projectVolpkgJson.string());
    }
    if (options.lasagnaManifest.empty()) {
        options.lasagnaManifest = pkg->selectedLasagnaDatasetPath();
    }
    if (options.lasagnaManifest.empty()) {
        throw std::runtime_error(
            "missing Lasagna manifest; set selected Lasagna dataset in project or pass --lasagna-manifest=...");
    }
}

double norm(const cv::Vec3d& value)
{
    return std::sqrt(value.dot(value));
}

bool finitePoint(const cv::Vec3d& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

cv::Vec3d normalizedOrZero(const cv::Vec3d& value)
{
    if (!finitePoint(value)) {
        return {0.0, 0.0, 0.0};
    }
    const double n = norm(value);
    if (!(n > kEpsilon) || !std::isfinite(n)) {
        return {0.0, 0.0, 0.0};
    }
    return value * (1.0 / n);
}

bool validNormal(const cv::Vec3d& value)
{
    return norm(normalizedOrZero(value)) > 0.0;
}

std::string pointString(const cv::Vec3d& point)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(3)
        << '[' << point[0] << ',' << point[1] << ',' << point[2] << ']';
    return out.str();
}

std::string optionalDoubleString(std::optional<double> value)
{
    if (!value) {
        return "missing";
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(3) << *value;
    return out.str();
}

std::string optionalDoubleString(double value)
{
    if (!std::isfinite(value)) {
        return "missing";
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

vc::atlas::FiberInput fiberInputFromMapping(const vc::atlas::FiberMapping& mapping)
{
    vc::atlas::FiberInput input;
    input.fiberPath = mapping.fiberPath;
    input.controlPoints.reserve(mapping.controlAnchors.size());
    input.controlLineIndices.reserve(mapping.controlAnchors.size());
    for (const auto& anchor : mapping.controlAnchors) {
        input.controlPoints.push_back(anchor.world);
        input.controlLineIndices.push_back(anchor.sourceIndex);
    }
    return input;
}

const vc::atlas::AtlasAnchor* findControlAnchor(const vc::atlas::FiberMapping& mapping,
                                                int controlSourceIndex)
{
    const auto it = std::find_if(
        mapping.controlAnchors.begin(),
        mapping.controlAnchors.end(),
        [controlSourceIndex](const vc::atlas::AtlasAnchor& anchor) {
            return anchor.sourceIndex == controlSourceIndex;
        });
    return it == mapping.controlAnchors.end() ? nullptr : &*it;
}

TraceStats traceDirection(const cv::Vec3d& start,
                          const cv::Vec3d& unitDirection,
                          double stepVx,
                          double maxWinding,
                          const vc::lasagna::LasagnaNormalSampler& sampler,
                          bool printSamples,
                          const char* label)
{
    TraceStats stats;
    cv::Vec3d previous = start;
    double accumulated = 0.0;
    for (int step = 1; step <= kMaxTraceSteps && accumulated <= maxWinding + 1.0e-9; ++step) {
        const cv::Vec3d current =
            start + unitDirection * (stepVx * static_cast<double>(step));
        const double segmentWinding =
            sampler.windingDistance(previous, current, stepVx);
        if (!std::isfinite(segmentWinding) || segmentWinding < 0.0) {
            if (printSamples) {
                std::cout << "      " << label << " stopped: non-finite winding segment\n";
            }
            break;
        }
        accumulated += segmentWinding;
        if (accumulated > maxWinding + 1.0e-9) {
            break;
        }
        const auto predDt = sampler.samplePredDt(current);
        ++stats.samples;
        stats.lastWinding = accumulated;
        stats.lastPoint = current;
        if (!predDt) {
            ++stats.missingPredDt;
        }
        if (printSamples) {
            std::cout << "      " << label << " step=" << step
                      << " w=" << optionalDoubleString(accumulated)
                      << " pred_dt=" << optionalDoubleString(predDt)
                      << " p=" << pointString(current) << '\n';
        }
        if (predDt && vc::atlas::atlasPredDtIsInside(*predDt)) {
            if (!stats.firstInsideWinding) {
                stats.firstInsideWinding = accumulated;
                stats.firstInsidePredDt = predDt;
                stats.firstInsidePoint = current;
            }
            if (!stats.maxInsidePredDt || *predDt > *stats.maxInsidePredDt) {
                stats.maxInsidePredDt = predDt;
                stats.maxInsideWinding = accumulated;
                stats.maxInsidePoint = current;
            }
        }
        previous = current;
        if (segmentWinding <= kEpsilon && step > 16) {
            if (printSamples) {
                std::cout << "      " << label << " stopped: zero winding progress\n";
            }
            break;
        }
    }
    return stats;
}

void printTraceSummary(const char* label, const TraceStats& stats, double weight = 1.0)
{
    std::cout << "    " << label
              << ": samples=" << stats.samples
              << " missing_pred_dt=" << stats.missingPredDt
              << " last_w=" << optionalDoubleString(stats.lastWinding);
    if (stats.firstInsideWinding) {
        std::cout << " first_inside_w=" << optionalDoubleString(stats.firstInsideWinding)
                  << " weighted_first_w=" << optionalDoubleString(*stats.firstInsideWinding * weight)
                  << " first_pred_dt=" << optionalDoubleString(stats.firstInsidePredDt)
                  << " first_p=" << pointString(*stats.firstInsidePoint);
    } else {
        std::cout << " first_inside=none";
    }
    if (stats.maxInsidePredDt) {
        std::cout << " max_inside_pred_dt=" << optionalDoubleString(stats.maxInsidePredDt)
                  << " max_w=" << optionalDoubleString(stats.maxInsideWinding)
                  << " max_p=" << pointString(*stats.maxInsidePoint);
    }
    std::cout << '\n';
}

const vc::atlas::AtlasPredSnapPoint* generatedPointForControl(
    const vc::atlas::AtlasPredSnapSet& set,
    const cv::Vec3d& controlPoint)
{
    const std::string key = vc::atlas::atlasPredSnapControlPointKey(controlPoint);
    const auto it = std::find_if(
        set.points.begin(),
        set.points.end(),
        [&key](const vc::atlas::AtlasPredSnapPoint& point) {
            return vc::atlas::atlasPredSnapControlPointKey(point.controlPoint) == key;
        });
    return it == set.points.end() ? nullptr : &*it;
}

std::string safeDebugName(const fs::path& path)
{
    std::string name = path.generic_string();
    if (name.empty()) {
        name = "fiber";
    }
    for (char& ch : name) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_')) {
            ch = '_';
        }
    }
    return name;
}

cv::Point projectToPanel(const cv::Vec3d& point,
                         const cv::Vec3d& center,
                         const cv::Vec3d& normalAxis,
                         const cv::Vec3d& tangentAxis,
                         int panelOffsetX,
                         int panelSize)
{
    const cv::Vec3d delta = point - center;
    const double u = delta.dot(normalAxis);
    const double v = delta.dot(tangentAxis);
    const int x = panelOffsetX + panelSize / 2 + static_cast<int>(std::lround(u));
    const int y = panelSize / 2 - static_cast<int>(std::lround(v));
    return {x, y};
}

void drawDebugPanel(cv::Mat& image,
                    int panelOffsetX,
                    const cv::Vec3d& controlPoint,
                    const cv::Vec3d& normalAxis,
                    const cv::Vec3d& tangentAxis,
                    const vc::lasagna::LasagnaNormalSampler& sampler)
{
    constexpr int kPanelSize = 256;
    for (int y = 0; y < kPanelSize; ++y) {
        for (int x = 0; x < kPanelSize; ++x) {
            const double u = static_cast<double>(x - kPanelSize / 2);
            const double v = static_cast<double>(kPanelSize / 2 - y);
            const cv::Vec3d samplePoint = controlPoint + normalAxis * u + tangentAxis * v;
            const auto predDt = sampler.samplePredDt(samplePoint);
            cv::Vec3b color{24, 0, 24};
            if (predDt) {
                const double clamped = std::clamp(*predDt, 0.0, 175.0);
                const auto gray = static_cast<uchar>(std::lround(clamped * 255.0 / 175.0));
                color = cv::Vec3b{gray, gray, gray};
                if (vc::atlas::atlasPredDtIsInside(*predDt)) {
                    color = cv::Vec3b{
                        static_cast<uchar>(std::min(255, static_cast<int>(gray) + 35)),
                        static_cast<uchar>(std::min(255, static_cast<int>(gray) + 20)),
                        gray};
                }
            }
            image.at<cv::Vec3b>(y, panelOffsetX + x) = color;
        }
    }
}

void saveControlDebugImage(const fs::path& outputPath,
                           const cv::Vec3d& controlPoint,
                           const cv::Vec3d& alignedNormal,
                           const TraceStats& outward,
                           const TraceStats& inward,
                           const vc::atlas::AtlasPredSnapPoint* result,
                           const vc::lasagna::LasagnaNormalSampler& sampler)
{
    constexpr int kPanelSize = 256;
    cv::Vec3d normalAxis = normalizedOrZero(alignedNormal);
    if (!validNormal(normalAxis)) {
        return;
    }
    cv::Vec3d seed{0.0, 0.0, 1.0};
    if (std::abs(normalAxis.dot(seed)) > 0.9) {
        seed = {0.0, 1.0, 0.0};
    }
    const cv::Vec3d tangentA = normalizedOrZero(normalAxis.cross(seed));
    const cv::Vec3d tangentB = normalizedOrZero(normalAxis.cross(tangentA));
    if (!validNormal(tangentA) || !validNormal(tangentB)) {
        return;
    }

    cv::Mat image(kPanelSize, kPanelSize * 2, CV_8UC3, cv::Scalar(0, 0, 0));
    drawDebugPanel(image, 0, controlPoint, normalAxis, tangentA, sampler);
    drawDebugPanel(image, kPanelSize, controlPoint, normalAxis, tangentB, sampler);

    auto drawOverlays = [&](int offsetX, const cv::Vec3d& tangentAxis) {
        const cv::Point center{kPanelSize / 2 + offsetX, kPanelSize / 2};
        if (outward.lastPoint) {
            cv::line(image,
                     center,
                     projectToPanel(*outward.lastPoint, controlPoint, normalAxis, tangentAxis, offsetX, kPanelSize),
                     cv::Scalar(0, 0, 255),
                     1,
                     cv::LINE_AA);
        }
        if (inward.lastPoint) {
            cv::line(image,
                     center,
                     projectToPanel(*inward.lastPoint, controlPoint, normalAxis, tangentAxis, offsetX, kPanelSize),
                     cv::Scalar(255, 255, 0),
                     1,
                     cv::LINE_AA);
        }
        cv::circle(image, center, 4, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
        if (result && result->predSnapPoint) {
            cv::circle(image,
                       projectToPanel(*result->predSnapPoint, controlPoint, normalAxis, tangentAxis, offsetX, kPanelSize),
                       4,
                       cv::Scalar(0, 165, 255),
                       -1,
                       cv::LINE_AA);
        }
    };
    drawOverlays(0, tangentA);
    drawOverlays(kPanelSize, tangentB);
    cv::line(image, {kPanelSize, 0}, {kPanelSize, kPanelSize - 1}, cv::Scalar(32, 32, 32), 1);

    fs::create_directories(outputPath.parent_path());
    if (!cv::imwrite(outputPath.string(), image)) {
        throw std::runtime_error("failed to write debug image: " + outputPath.string());
    }
}

void printControlDebug(const vc::atlas::FiberInput& input,
                       const vc::atlas::FiberMapping& mapping,
                       const QuadSurface& baseSurface,
                       const vc::lasagna::LasagnaNormalSampler& sampler,
                       const vc::atlas::AtlasPredSnapSet& generated,
                       double predDtStepVx,
                       bool traceSamples)
{
    for (size_t controlIndex = 0; controlIndex < input.controlPoints.size(); ++controlIndex) {
        const cv::Vec3d& controlPoint = input.controlPoints[controlIndex];
        const int sourceIndex = controlIndex < input.controlLineIndices.size()
            ? input.controlLineIndices[controlIndex]
            : static_cast<int>(controlIndex);
        std::cout << "  control[" << controlIndex << "] source_line_index=" << sourceIndex
                  << " key=" << vc::atlas::atlasPredSnapControlPointKey(controlPoint)
                  << " p=" << pointString(controlPoint) << '\n';

        const vc::atlas::AtlasAnchor* anchor = findControlAnchor(mapping, sourceIndex);
        if (!anchor) {
            std::cout << "    fail: no control anchor with source_index=" << sourceIndex
                      << " (mapping stores line_points indices, not control ordinals)\n";
            continue;
        }

        const auto basePoint = vc::atlas::atlasAnchorBasePoint(*anchor, mapping, baseSurface);
        const auto baseNormal = vc::atlas::atlasAnchorBaseNormal(*anchor, mapping, baseSurface);
        std::cout << "    anchor: atlas_u=" << anchor->atlasU
                  << " atlas_v=" << anchor->atlasV
                  << " base_p=" << (basePoint ? pointString(*basePoint) : std::string("missing"))
                  << " base_normal=" << (baseNormal ? pointString(*baseNormal) : std::string("missing"))
                  << '\n';
        if (!baseNormal) {
            std::cout << "    fail: could not sample atlas base normal at anchor\n";
            continue;
        }

        const auto normalSample = sampler.sampleNormal(controlPoint);
        std::cout << "    lasagna_normal: valid=" << (normalSample.valid ? "yes" : "no")
                  << " raw=" << pointString(normalSample.normal);
        if (!normalSample.reason.empty()) {
            std::cout << " reason=\"" << normalSample.reason << '"';
        }
        std::cout << '\n';
        if (!normalSample.valid || !validNormal(normalSample.normal)) {
            std::cout << "    fail: invalid Lasagna normal at control point\n";
            continue;
        }

        cv::Vec3d alignedNormal = normalizedOrZero(normalSample.normal);
        const double rawDotBase = alignedNormal.dot(*baseNormal);
        if (rawDotBase < 0.0) {
            alignedNormal *= -1.0;
        }
        const auto startPredDt = sampler.samplePredDt(controlPoint);
        std::cout << "    search_normal=" << pointString(alignedNormal)
                  << " raw_dot_base=" << rawDotBase
                  << " aligned_dot_base=" << alignedNormal.dot(*baseNormal)
                  << " start_pred_dt=" << optionalDoubleString(startPredDt)
                  << " start_inside="
                  << (startPredDt && vc::atlas::atlasPredDtIsInside(*startPredDt) ? "yes" : "no")
                  << '\n';

        const TraceStats outward = traceDirection(controlPoint,
                                                  alignedNormal,
                                                  predDtStepVx,
                                                  kOutwardWindingLimit,
                                                  sampler,
                                                  traceSamples,
                                                  "outward");
        const TraceStats inward = traceDirection(controlPoint,
                                                 -alignedNormal,
                                                 predDtStepVx,
                                                 kInwardWindingLimit,
                                                 sampler,
                                                 traceSamples,
                                                 "inward");
        printTraceSummary("outward", outward);
        printTraceSummary("inward", inward, kInwardFirstHitWeight);

        const auto* result = generatedPointForControl(generated, controlPoint);
        if (!result) {
            std::cout << "    result: missing generated record\n";
        } else if (!result->predSnapPoint) {
            std::cout << "    result: null pred_snap_point\n";
        } else {
            std::cout << "    result: snap=" << pointString(*result->predSnapPoint)
                      << " pred_dt=" << optionalDoubleString(result->predDtValue)
                      << " direction="
                      << (result->direction == vc::atlas::AtlasPredSnapDirection::Inside
                              ? "inside"
                              : "outside")
                      << " weighted_first_hit_w="
                      << optionalDoubleString(result->weightedFirstHitWindingDistance)
                      << '\n';
        }
    }
}

void saveFiberDebugImages(const fs::path& debugImagesDir,
                          const vc::atlas::FiberInput& input,
                          const vc::atlas::FiberMapping& mapping,
                          const QuadSurface& baseSurface,
                          const vc::lasagna::LasagnaNormalSampler& sampler,
                          const vc::atlas::AtlasPredSnapSet& generated,
                          double predDtStepVx)
{
    const fs::path fiberDir = debugImagesDir / safeDebugName(mapping.fiberPath);
    for (size_t controlIndex = 0; controlIndex < input.controlPoints.size(); ++controlIndex) {
        const cv::Vec3d& controlPoint = input.controlPoints[controlIndex];
        const int sourceIndex = controlIndex < input.controlLineIndices.size()
            ? input.controlLineIndices[controlIndex]
            : static_cast<int>(controlIndex);
        const vc::atlas::AtlasAnchor* anchor = findControlAnchor(mapping, sourceIndex);
        if (!anchor) {
            continue;
        }
        const auto baseNormal = vc::atlas::atlasAnchorBaseNormal(*anchor, mapping, baseSurface);
        if (!baseNormal) {
            continue;
        }
        const auto normalSample = sampler.sampleNormal(controlPoint);
        if (!normalSample.valid || !validNormal(normalSample.normal)) {
            continue;
        }

        cv::Vec3d alignedNormal = normalizedOrZero(normalSample.normal);
        if (alignedNormal.dot(*baseNormal) < 0.0) {
            alignedNormal *= -1.0;
        }
        const TraceStats outward = traceDirection(controlPoint,
                                                  alignedNormal,
                                                  predDtStepVx,
                                                  kOutwardWindingLimit,
                                                  sampler,
                                                  false,
                                                  "outward");
        const TraceStats inward = traceDirection(controlPoint,
                                                 -alignedNormal,
                                                 predDtStepVx,
                                                 kInwardWindingLimit,
                                                 sampler,
                                                 false,
                                                 "inward");

        std::ostringstream filename;
        filename << "cp_" << std::setw(6) << std::setfill('0') << controlIndex << ".png";
        saveControlDebugImage(fiberDir / filename.str(),
                              controlPoint,
                              alignedNormal,
                              outward,
                              inward,
                              generatedPointForControl(generated, controlPoint),
                              sampler);
    }
}

bool pathMatchesFilter(const fs::path& fiberPath, const fs::path& filter)
{
    if (filter.empty()) {
        return true;
    }
    const fs::path normalizedFiber = fiberPath.lexically_normal();
    return normalizedFiber == filter ||
           normalizedFiber.filename() == filter.filename() ||
           normalizedFiber.generic_string().find(filter.generic_string()) != std::string::npos;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::cout.imbue(std::locale::classic());
        Options options = parseArgs(argc, argv);
        resolveProjectContext(options);

        const auto dataset = vc::lasagna::LasagnaDataset::open(options.lasagnaManifest);
        vc::lasagna::LasagnaNormalSampler sampler(dataset);
        if (!sampler.hasPredDtChannel()) {
            throw std::runtime_error("Lasagna dataset has no pred_dt channel: " +
                                     options.lasagnaManifest.string());
        }
        const auto predDtSpacing = sampler.predDtSpacing();
        if (!predDtSpacing || !std::isfinite(*predDtSpacing) || *predDtSpacing <= 0.0) {
            throw std::runtime_error("Lasagna pred_dt channel has no valid spacing");
        }
        const double predDtStepVx = 0.5 * *predDtSpacing;

        vc::atlas::Atlas atlas =
            vc::atlas::Atlas::load(options.atlasDir, options.volpkgRoot);
        if (atlas.metadata.baseMeshPath.empty()) {
            throw std::runtime_error("atlas metadata is missing base_mesh_path");
        }
        const fs::path basePath = (options.atlasDir / atlas.metadata.baseMeshPath)
                                      .lexically_normal();
        QuadSurface baseSurface(basePath);

        if (options.verbose) {
            std::cout << "atlas=" << options.atlasDir << '\n'
                      << "project=" << options.projectVolpkgJson << '\n'
                      << "volpkg_root=" << options.volpkgRoot << '\n'
                      << "lasagna_manifest=" << options.lasagnaManifest << '\n'
                      << "pred_dt_spacing=" << optionalDoubleString(predDtSpacing) << '\n'
                      << "pred_dt_search_step=" << optionalDoubleString(predDtStepVx) << '\n'
                      << "base_mesh=" << basePath << '\n'
                      << "fibers=" << atlas.fibers.size()
                      << " dry_run=" << (options.dryRun ? "yes" : "no")
                      << " overwrite_manual=" << (options.overwriteManual ? "yes" : "no")
                      << '\n';
        }

        size_t fibersProcessed = 0;
        size_t controlsProcessed = 0;
        size_t snapsFound = 0;
        size_t snapsNull = 0;
        size_t attachmentsWritten = 0;

        for (const auto& mapping : atlas.fibers) {
            if (!pathMatchesFilter(mapping.fiberPath, options.fiberFilter)) {
                continue;
            }

            ++fibersProcessed;
            const vc::atlas::FiberInput input = fiberInputFromMapping(mapping);
            const fs::path attachmentPath =
                vc::atlas::atlasPredSnapAttachmentPath(options.atlasDir,
                                                       mapping.fiberPath);
            if (options.verbose) {
                std::cout << "\nfiber=" << mapping.fiberPath
                          << " controls=" << input.controlPoints.size()
                          << " control_anchors=" << mapping.controlAnchors.size()
                          << " attachment=" << attachmentPath << '\n';
            }

            vc::atlas::AtlasPredSnapSet generated =
                vc::atlas::generateAtlasPredSnapSet(input, mapping, baseSurface, sampler);
            if (!options.debugImagesDir.empty()) {
                saveFiberDebugImages(options.debugImagesDir,
                                     input,
                                     mapping,
                                     baseSurface,
                                     sampler,
                                     generated,
                                     predDtStepVx);
            }
            if (options.verbose) {
                printControlDebug(input,
                                  mapping,
                                  baseSurface,
                                  sampler,
                                  generated,
                                  predDtStepVx,
                                  options.traceSamples);
            }

            size_t fiberFound = 0;
            size_t fiberNull = 0;
            controlsProcessed += generated.points.size();
            for (const auto& point : generated.points) {
                if (point.predSnapPoint) {
                    ++snapsFound;
                    ++fiberFound;
                } else {
                    ++snapsNull;
                    ++fiberNull;
                }
            }
            std::cout << "fiber=" << mapping.fiberPath
                      << " controls=" << generated.points.size()
                      << " snaps_found=" << fiberFound
                      << " snaps_null=" << fiberNull << '\n';

            vc::atlas::AtlasPredSnapSet finalSet;
            if (options.overwriteManual) {
                finalSet = std::move(generated);
            } else {
                vc::atlas::AtlasPredSnapSet existing =
                    vc::atlas::loadAtlasPredSnapSet(attachmentPath);
                finalSet = vc::atlas::mergeAtlasPredSnapSetByControlPoint(
                    std::move(existing),
                    generated);
            }

            if (!options.dryRun) {
                vc::atlas::saveAtlasPredSnapSet(attachmentPath, finalSet);
                ++attachmentsWritten;
            }
        }

        std::cout << "summary: fibers_processed=" << fibersProcessed
                  << " controls=" << controlsProcessed
                  << " snaps_found=" << snapsFound
                  << " snaps_null=" << snapsNull
                  << " attachments_written=" << attachmentsWritten << '\n';
        if (fibersProcessed == 0) {
            std::cout << "warning: no fibers matched the requested filter\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        printUsage(argv[0]);
        return 1;
    }
}
