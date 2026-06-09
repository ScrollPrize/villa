#include "vc/atlas/Atlas.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path atlasDir;
    fs::path volpkgJson;
};

struct ProjectContext {
    fs::path volpkgRoot;
    fs::path lasagnaManifestPath;
};

struct ControlVector {
    std::string objectId;
    int sourceIndex = 0;
    double atlasU = 0.0;
    double atlasV = 0.0;
    double actualAtlasU = 0.0;
    int windingOffset = 0;
    cv::Vec3d controlPoint{0.0, 0.0, 0.0};
    cv::Vec3d basePoint{0.0, 0.0, 0.0};
    cv::Vec3d vector{0.0, 0.0, 0.0};
    double length = 0.0;
    bool valid = false;
    double remappedLength = std::numeric_limits<double>::quiet_NaN();
    bool remappedValid = false;
};

using ControlLengthBySourceIndex = std::unordered_map<int, double>;
using ControlLengthByObjectId = std::unordered_map<std::string, ControlLengthBySourceIndex>;

struct RemapReport {
    ControlLengthByObjectId lengths;
    std::unordered_map<std::string, vc::atlas::FiberMapping> mappings;
    int remappedControls = 0;
    int failedFibers = 0;
};

void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " <atlas_dir> [project.volpkg.json]\n"
        << "Loads a native VC3D atlas and reports control-anchor vectors from\n"
        << "the current base mesh point at each control atlas coordinate to the\n"
        << "stored control point. Line anchors are not reported.\n"
        << "Also writes old/new control-connection OBJ files in the cwd.\n";
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
        if (!arg.empty() && arg[0] == '-') {
            throw std::invalid_argument("unknown option: " + arg);
        }
        if (options.atlasDir.empty()) {
            options.atlasDir = arg;
        } else if (options.volpkgJson.empty()) {
            options.volpkgJson = arg;
        } else {
            throw std::invalid_argument("too many positional arguments");
        }
    }
    if (options.atlasDir.empty()) {
        throw std::invalid_argument("missing atlas_dir");
    }
    return options;
}

bool endsWith(const std::string& value, const std::string& suffix)
{
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

ProjectContext loadProjectContext(const fs::path& path)
{
    ProjectContext context;
    if (path.empty()) {
        return context;
    }
    if (!fs::is_regular_file(path)) {
        throw std::invalid_argument("project.volpkg.json is not a file: " + path.string());
    }
    if (!endsWith(path.filename().string(), ".volpkg.json")) {
        throw std::invalid_argument("package context must be a *.volpkg.json file: " +
                                    path.string());
    }
    const auto pkg = VolumePkg::load(path);
    context.volpkgRoot = pkg ? fs::path(pkg->getVolpkgDirectory()) : fs::path{};
    if (context.volpkgRoot.empty()) {
        throw std::runtime_error("failed to resolve volpkg directory from " + path.string());
    }
    context.lasagnaManifestPath = pkg->selectedLasagnaDatasetPath();
    return context;
}

double norm(const cv::Vec3d& v)
{
    return std::sqrt(v.dot(v));
}

cv::Vec3d pointFromJson(const nlohmann::json& value, const fs::path& path, const char* name)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(std::string(name) + " point must be [x,y,z]: " +
                                 path.string());
    }
    cv::Vec3d point{
        value.at(0).get<double>(),
        value.at(1).get<double>(),
        value.at(2).get<double>(),
    };
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) {
        throw std::runtime_error(std::string(name) + " point contains non-finite value: " +
                                 path.string());
    }
    return point;
}

std::vector<cv::Vec3d> pointArrayFromJson(const nlohmann::json& root,
                                          const char* key,
                                          const fs::path& path)
{
    const auto it = root.find(key);
    if (it == root.end() || !it->is_array()) {
        throw std::runtime_error("fiber JSON is missing array " + std::string(key) + ": " +
                                 path.string());
    }
    std::vector<cv::Vec3d> points;
    points.reserve(it->size());
    for (const auto& point : *it) {
        points.push_back(pointFromJson(point, path, key));
    }
    return points;
}

vc::atlas::FiberInput loadFiberInput(const vc::atlas::LasagnaAtlasObject& object)
{
    std::ifstream in(object.fiberPath);
    if (!in) {
        throw std::runtime_error("failed to open fiber JSON: " +
                                 object.fiberPath.string());
    }
    const nlohmann::json root = nlohmann::json::parse(in);
    vc::atlas::FiberInput input;
    input.fiberPath = object.fiberRelativePath;
    input.controlPoints = pointArrayFromJson(root, "control_points", object.fiberPath);
    input.linePoints = pointArrayFromJson(root, "line_points", object.fiberPath);
    vc::atlas::validateFiberInputControlPoints(input);
    return input;
}

RemapReport remapFibers(const vc::atlas::LasagnaAtlasExport& exportData,
                        const QuadSurface& baseSurface,
                        const vc::lasagna::NormalSampler& sampler)
{
    auto baseSurfacePtr = std::shared_ptr<QuadSurface>(
        const_cast<QuadSurface*>(&baseSurface),
        [](QuadSurface*) {});
    SurfacePatchIndex baseIndex;
    baseIndex.rebuild({baseSurfacePtr});

    RemapReport report;
    for (const auto& object : exportData.objects) {
        try {
            const vc::atlas::FiberInput input = loadFiberInput(object);
            vc::atlas::FiberMapping remapped =
                vc::atlas::mapFiberToBaseSurface(input, baseSurface, baseIndex, sampler);
            auto& lengths = report.lengths[object.id];
            for (const auto& anchor : remapped.controlAnchors) {
                const auto basePoint =
                    vc::atlas::atlasAnchorBasePoint(anchor, remapped, baseSurface);
                if (!basePoint) {
                    continue;
                }
                const double length = norm(anchor.world - *basePoint);
                if (std::isfinite(length)) {
                    lengths[anchor.sourceIndex] = length;
                    ++report.remappedControls;
                }
            }
            report.mappings.emplace(object.id, std::move(remapped));
        } catch (const std::exception& ex) {
            ++report.failedFibers;
            std::cerr << "vc_atlas_inspect: remap failed for " << object.id
                      << ": " << ex.what() << '\n';
        }
    }
    return report;
}

std::vector<ControlVector> collectControlVectors(const vc::atlas::Atlas& atlas,
                                                 const QuadSurface& baseSurface,
                                                 const ControlLengthByObjectId* remappedLengths)
{
    const int periodColumns = vc::atlas::atlasHorizontalPeriodColumns(baseSurface);
    std::vector<ControlVector> out;
    for (const auto& fiber : atlas.fibers) {
        for (const auto& anchor : fiber.controlAnchors) {
            ControlVector row;
            row.objectId = fiber.fiberPath.generic_string();
            row.sourceIndex = anchor.sourceIndex;
            row.atlasU = anchor.atlasU;
            row.atlasV = anchor.atlasV;
            row.actualAtlasU = vc::atlas::actualAtlasU(anchor, fiber, periodColumns);
            row.windingOffset = fiber.windingOffset;
            row.controlPoint = anchor.world;
            if (const auto basePoint = vc::atlas::atlasAnchorBasePoint(anchor, fiber, baseSurface)) {
                row.basePoint = *basePoint;
                row.vector = row.controlPoint - row.basePoint;
                row.length = norm(row.vector);
                row.valid = std::isfinite(row.length);
            }
            if (remappedLengths) {
                const auto objectIt = remappedLengths->find(row.objectId);
                if (objectIt != remappedLengths->end()) {
                    const auto lengthIt = objectIt->second.find(row.sourceIndex);
                    if (lengthIt != objectIt->second.end()) {
                        row.remappedLength = lengthIt->second;
                        row.remappedValid = std::isfinite(row.remappedLength);
                    }
                }
            }
            out.push_back(row);
        }
    }
    return out;
}

void printHuman(const vc::atlas::LasagnaAtlasExport& exportData,
                const QuadSurface& baseSurface,
                const std::vector<ControlVector>& rows)
{
    constexpr int kIntWidth = 5;
    constexpr int kAtlasWidth = 9;
    constexpr int kPointWidth = 11;
    constexpr int kVecWidth = 10;
    constexpr int kLengthWidth = 9;

    const int periodColumns = vc::atlas::atlasHorizontalPeriodColumns(baseSurface);
    int validCount = 0;
    double minLength = std::numeric_limits<double>::infinity();
    double maxLength = 0.0;
    double sumLength = 0.0;
    for (const auto& row : rows) {
        if (!row.valid) {
            continue;
        }
        ++validCount;
        minLength = std::min(minLength, row.length);
        maxLength = std::max(maxLength, row.length);
        sumLength += row.length;
    }

    std::cout.imbue(std::locale::classic());
    std::cout << "Atlas control inspection\n"
              << "  atlas_dir=" << exportData.atlasDir.generic_string() << '\n'
              << "  base_mesh=" << exportData.basePath.generic_string() << '\n'
              << "  period_columns=" << periodColumns << '\n'
              << "  fibers=" << exportData.atlas.fibers.size()
              << " control_anchors=" << rows.size()
              << " valid_control_vectors=" << validCount << '\n';
    if (validCount > 0) {
        std::cout << std::fixed << std::setprecision(2)
                  << "  control_length_min=" << minLength
                  << " mean=" << (sumLength / static_cast<double>(validCount))
                  << " max=" << maxLength << '\n';
    }

    auto printHeader = [](const std::string& objectId) {
        std::cout << "\n" << objectId << '\n'
                  << std::right
                  << std::setw(kIntWidth) << "idx"
                  << std::setw(kIntWidth) << "wnd"
                  << std::setw(kAtlasWidth) << "au"
                  << std::setw(kAtlasWidth) << "av"
                  << std::setw(kAtlasWidth) << "u"
                  << std::setw(kPointWidth) << "base_x"
                  << std::setw(kPointWidth) << "base_y"
                  << std::setw(kPointWidth) << "base_z"
                  << std::setw(kPointWidth) << "ctrl_x"
                  << std::setw(kPointWidth) << "ctrl_y"
                  << std::setw(kPointWidth) << "ctrl_z"
                  << std::setw(kVecWidth) << "dx"
                  << std::setw(kVecWidth) << "dy"
                  << std::setw(kVecWidth) << "dz"
                  << std::setw(kLengthWidth) << "len_old"
                  << std::setw(kLengthWidth) << "len_new"
                  << '\n';
    };

    auto printNumber = [](double value, int width) {
        if (std::isfinite(value)) {
            std::cout << std::setw(width) << value;
        } else {
            std::cout << std::setw(width) << "nan";
        }
    };

    std::string currentObjectId;
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& row : rows) {
        if (row.objectId != currentObjectId) {
            currentObjectId = row.objectId;
            printHeader(currentObjectId);
        }
        std::cout << std::right
                  << std::setw(kIntWidth) << row.sourceIndex
                  << std::setw(kIntWidth) << row.windingOffset;
        printNumber(row.atlasU, kAtlasWidth);
        printNumber(row.atlasV, kAtlasWidth);
        printNumber(row.actualAtlasU, kAtlasWidth);
        printNumber(row.valid ? row.basePoint[0] : std::numeric_limits<double>::quiet_NaN(),
                    kPointWidth);
        printNumber(row.valid ? row.basePoint[1] : std::numeric_limits<double>::quiet_NaN(),
                    kPointWidth);
        printNumber(row.valid ? row.basePoint[2] : std::numeric_limits<double>::quiet_NaN(),
                    kPointWidth);
        printNumber(row.controlPoint[0], kPointWidth);
        printNumber(row.controlPoint[1], kPointWidth);
        printNumber(row.controlPoint[2], kPointWidth);
        printNumber(row.valid ? row.vector[0] : std::numeric_limits<double>::quiet_NaN(),
                    kVecWidth);
        printNumber(row.valid ? row.vector[1] : std::numeric_limits<double>::quiet_NaN(),
                    kVecWidth);
        printNumber(row.valid ? row.vector[2] : std::numeric_limits<double>::quiet_NaN(),
                    kVecWidth);
        printNumber(row.valid ? row.length : std::numeric_limits<double>::quiet_NaN(),
                    kLengthWidth);
        printNumber(row.remappedValid ? row.remappedLength
                                      : std::numeric_limits<double>::quiet_NaN(),
                    kLengthWidth);
        std::cout << '\n';
    }
}

std::string sanitizeObjName(const std::string& value)
{
    std::string out;
    out.reserve(value.size());
    for (const unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.') {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('_');
        }
    }
    if (out.empty()) {
        return "fiber";
    }
    return out;
}

int writeMappingObj(const fs::path& path,
                    const std::string& label,
                    const vc::atlas::FiberMapping& mapping,
                    const QuadSurface& baseSurface)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write OBJ: " + path.string());
    }
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(6)
        << "# vc_atlas_inspect " << label << " control connections\n"
        << "# fiber " << mapping.fiberPath.generic_string() << '\n'
        << "o " << label << '_' << sanitizeObjName(mapping.fiberPath.generic_string()) << '\n';

    int vertexIndex = 1;
    int segments = 0;
    for (const auto& anchor : mapping.controlAnchors) {
        const auto basePoint = vc::atlas::atlasAnchorBasePoint(anchor, mapping, baseSurface);
        if (!basePoint) {
            continue;
        }
        const cv::Vec3d vector = anchor.world - *basePoint;
        const double length = norm(vector);
        if (!std::isfinite(length)) {
            continue;
        }
        out << "# source_index " << anchor.sourceIndex
            << " length " << length << '\n'
            << "v " << (*basePoint)[0] << ' ' << (*basePoint)[1] << ' '
            << (*basePoint)[2] << '\n'
            << "v " << anchor.world[0] << ' ' << anchor.world[1] << ' '
            << anchor.world[2] << '\n'
            << "l " << vertexIndex << ' ' << (vertexIndex + 1) << '\n';
        vertexIndex += 2;
        ++segments;
    }
    return segments;
}

void writeMissingMappingObj(const fs::path& path,
                            const std::string& label,
                            const vc::atlas::FiberMapping& mapping)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write OBJ: " + path.string());
    }
    out << "# vc_atlas_inspect " << label << " control connections\n"
        << "# fiber " << mapping.fiberPath.generic_string() << '\n'
        << "# no regenerated mapping available\n"
        << "o " << label << '_' << sanitizeObjName(mapping.fiberPath.generic_string()) << '\n';
}

int writeDebugObjs(const vc::atlas::LasagnaAtlasExport& exportData,
                   const QuadSurface& baseSurface,
                   const RemapReport* remapReport)
{
    int written = 0;
    for (const auto& mapping : exportData.atlas.fibers) {
        const std::string name = sanitizeObjName(mapping.fiberPath.generic_string());
        const fs::path oldPath = fs::current_path() / ("atlas_inspect_old_" + name + ".obj");
        writeMappingObj(oldPath, "old", mapping, baseSurface);
        ++written;

        if (!remapReport) {
            continue;
        }
        const fs::path newPath = fs::current_path() / ("atlas_inspect_new_" + name + ".obj");
        const auto it = remapReport->mappings.find(mapping.fiberPath.generic_string());
        if (it == remapReport->mappings.end()) {
            writeMissingMappingObj(newPath, "new", mapping);
            ++written;
            continue;
        }
        writeMappingObj(newPath, "new", it->second, baseSurface);
        ++written;
    }
    return written;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parseArgs(argc, argv);
        const ProjectContext project = loadProjectContext(options.volpkgJson);
        const auto exportData = vc::atlas::loadLasagnaAtlasExport(options.atlasDir,
                                                                  project.volpkgRoot);
        const QuadSurface baseSurface(exportData.basePath);

        std::optional<RemapReport> remapReport;
        if (!project.lasagnaManifestPath.empty()) {
            const vc::lasagna::LasagnaDataset dataset =
                vc::lasagna::LasagnaDataset::open(project.lasagnaManifestPath);
            const vc::lasagna::LasagnaNormalSampler sampler(dataset);
            remapReport = remapFibers(exportData, baseSurface, sampler);
        } else {
            std::cerr << "vc_atlas_inspect: no selected Lasagna dataset; "
                      << "len_new and new OBJ output will be missing. "
                      << "Pass project.volpkg.json for remap comparison.\n";
        }

        const auto rows = collectControlVectors(exportData.atlas,
                                                baseSurface,
                                                remapReport ? &remapReport->lengths : nullptr);
        printHuman(exportData, baseSurface, rows);
        const int objCount = writeDebugObjs(exportData,
                                            baseSurface,
                                            remapReport ? &*remapReport : nullptr);
        if (remapReport) {
            std::cout << "\nremap_control_vectors=" << remapReport->remappedControls
                      << " remap_failed_fibers=" << remapReport->failedFibers << '\n';
        }
        std::cout << "debug_objs_written=" << objCount
                  << " dir=" << fs::current_path().generic_string() << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "vc_atlas_inspect: " << ex.what() << '\n';
        printUsage(argv[0]);
        return 1;
    }
}
