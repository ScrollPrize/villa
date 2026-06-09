#pragma once

#include <filesystem>
#include <exception>
#include <memory>
#include <optional>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>

class QuadSurface;
class SurfacePatchIndex;

namespace vc::lasagna {
struct LasagnaDatasetManifest;
class NormalSampler;
}

namespace vc::atlas {

struct AtlasMetadata {
    std::string type = "vc3d_atlas";
    int version = 4;
    std::string name;
    std::filesystem::path baseMeshPath;
    std::filesystem::path sourceBaseMeshPath;
    int zeroWindingColumn = 0;
    int seedLineIndex = 0;
    double seedAtlasU = 0.0;
    double seedAtlasV = 0.0;
};

struct AtlasAnchor {
    int sourceIndex = 0;
    cv::Vec3d world{0.0, 0.0, 0.0};
    double atlasU = 0.0;
    double atlasV = 0.0;
    double distance = 0.0;
};

struct FiberMapping {
    std::filesystem::path fiberPath;
    int windingOffset = 0;
    std::vector<AtlasAnchor> lineAnchors;
    std::vector<AtlasAnchor> controlAnchors;
};

struct AtlasLinkEndpoint {
    std::filesystem::path fiberPath;
    int sourceIndex = 0;
    double arclength = 0.0;
    double atlasU = 0.0;
    double atlasV = 0.0;
};

struct AtlasLink {
    AtlasLinkEndpoint first;
    AtlasLinkEndpoint second;
    int desiredWindingDelta = 0;
};

struct Atlas {
    AtlasMetadata metadata;
    std::vector<AtlasLink> links;
    std::vector<FiberMapping> fibers;

    void save(const std::filesystem::path& atlasDir) const;
    static Atlas load(const std::filesystem::path& atlasDir);
};

struct AtlasCoveredSize {
    double width = 0.0;
    double height = 0.0;
    bool valid = false;
};

struct AtlasDisplayRange {
    int baseColumns = 0;
    int leftmostWinding = 0;
    int rightmostWinding = 0;
    int unwrapCount = 1;
    double atlasUOffset = 0.0;
    bool hasMappedObjects = false;
};

struct FiberInput {
    std::filesystem::path fiberPath;
    std::vector<cv::Vec3d> controlPoints;
    std::vector<cv::Vec3d> linePoints;
    std::vector<int> controlLineIndices;
};

struct SurfaceCandidate {
    std::string name;
    std::filesystem::path path;
    std::shared_ptr<QuadSurface> surface;
};

struct BaseSelection {
    int surfaceIndex = -1;
    std::string surfaceName;
    cv::Vec3d seedPoint{0.0, 0.0, 0.0};
    int seedLineIndex = 0;
    cv::Vec3d world{0.0, 0.0, 0.0};
    double atlasU = 0.0;
    double atlasV = 0.0;
    double distance = 0.0;
};

struct LineMappingOptions {
    double rayHalfLength = 96.0;
    double mismatchRatio = 10.0;
};

struct ProjectionHit {
    std::shared_ptr<QuadSurface> surface;
    int surfaceIndex = -1;
    std::string surfaceName;
    cv::Vec3d world{0.0, 0.0, 0.0};
    double atlasU = 0.0;
    double atlasV = 0.0;
    double distance = 0.0;
};

struct FiberRuntimeIdentityMap {
    std::vector<std::filesystem::path> canonicalPaths;
    std::unordered_map<std::string, uint64_t> idByPathKey;
    std::unordered_map<uint64_t, std::filesystem::path> pathById;

    [[nodiscard]] uint64_t idForPath(const std::filesystem::path& path) const;
    [[nodiscard]] std::filesystem::path pathForId(uint64_t id) const;
};

struct AtlasFiberSearchSets {
    std::vector<uint64_t> sourceFiberIds;
    std::vector<uint64_t> targetFiberIds;
    std::vector<std::filesystem::path> sourceFiberPaths;
    std::vector<std::filesystem::path> targetFiberPaths;
};

struct AtlasDirectoryInfo {
    std::filesystem::path path;
    std::string name;
};

struct LasagnaAtlasObject {
    std::string id;
    std::filesystem::path fiberPath;
    std::filesystem::path mappingPath;
    std::filesystem::path fiberRelativePath;
    std::filesystem::path mappingRelativePath;
    int windingOffset = 0;
};

struct LasagnaAtlasExport {
    Atlas atlas;
    std::filesystem::path atlasDir;
    std::filesystem::path volpkgRoot;
    std::filesystem::path basePath;
    std::filesystem::path baseRelativePath;
    std::vector<LasagnaAtlasObject> objects;
    nlohmann::json compactJson;
};

std::string sanitizeAtlasName(std::string name);
std::string atlasFiberPathKey(const std::filesystem::path& path);
std::vector<std::string> atlasMappedFiberPathKeys(const Atlas& atlas);
FiberRuntimeIdentityMap makeFiberRuntimeIdentityMap(
    const std::vector<std::filesystem::path>& orderedCanonicalFiberPaths);
AtlasFiberSearchSets atlasFiberSearchSets(
    const Atlas& atlas,
    const FiberRuntimeIdentityMap& runtimeIds);
std::vector<AtlasDirectoryInfo> discoverAtlasDirectories(
    const std::filesystem::path& volpkgRoot);
LasagnaAtlasExport loadLasagnaAtlasExport(
    const std::filesystem::path& atlasDir,
    const std::filesystem::path& volpkgRoot = {});
std::filesystem::path uniqueAtlasDirectory(const std::filesystem::path& volpkgRoot,
                                           const std::string& baseName);
std::filesystem::path initShellDirectoryFromManifest(
    const vc::lasagna::LasagnaDatasetManifest& manifest);
std::vector<SurfaceCandidate> loadInitShellCandidates(
    const std::filesystem::path& initShellDir);

std::vector<ProjectionHit> projectPointAlongNormalToSurfaces(
    const cv::Vec3d& linePoint,
    const cv::Vec3d& normal,
    const std::vector<SurfaceCandidate>& surfaces,
    const SurfacePatchIndex& index,
    double rayHalfLength);

BaseSelection selectBaseSurfaceBySeedRay(const FiberInput& fiber,
                                         const std::vector<SurfaceCandidate>& surfaces,
                                         const SurfacePatchIndex& index,
                                         const vc::lasagna::NormalSampler& normalSampler,
                                         const LineMappingOptions& options = {});

int computeZeroWindingColumn(const QuadSurface& surface);
void saveAtlasBaseMeshCopy(const QuadSurface& surface,
                           const std::filesystem::path& targetDir);
AtlasCoveredSize mappedObjectCoveredAtlasSize(
    const Atlas& atlas,
    cv::Vec2f atlasScale = cv::Vec2f(1.0f, 1.0f),
    int periodColumns = 0);
int atlasHorizontalPeriodColumns(const QuadSurface& surface);
int atlasWindingForColumn(double atlasU, int periodColumns, int zeroWindingColumn);
double actualAtlasU(const AtlasAnchor& anchor,
                    const FiberMapping& fiber,
                    int periodColumns);
std::optional<cv::Vec3d> atlasBasePointAt(double atlasU,
                                          double atlasV,
                                          const QuadSurface& baseSurface);
std::optional<cv::Vec3d> atlasAnchorBasePoint(const AtlasAnchor& anchor,
                                              const FiberMapping& fiber,
                                              const QuadSurface& baseSurface);
AtlasDisplayRange atlasDisplayRange(const Atlas& atlas, int baseColumns);
void layoutAtlasObjects(Atlas& atlas, int periodColumns);
cv::Vec2f atlasGridToSurfaceCoords(double atlasU,
                                   double atlasV,
                                   const QuadSurface& displaySurface,
                                   double atlasUOffset = 0.0);
std::shared_ptr<QuadSurface> repeatedAtlasDisplaySurface(const QuadSurface& baseSurface,
                                                        int unwrapCount,
                                                        int startColumn = 0);

void validateFiberInputControlPoints(FiberInput& fiber);
bool atlasLoadErrorRequiresRebuild(const std::exception& ex);
Atlas rebuildAtlasFromSourceFibers(const std::filesystem::path& atlasDir,
                                   const std::filesystem::path& volpkgRoot,
                                   const vc::lasagna::NormalSampler& normalSampler,
                                   const LineMappingOptions& options = {});
Atlas rebuildAtlasFromSourceFibers(const std::filesystem::path& atlasDir,
                                   const std::filesystem::path& volpkgRoot,
                                   const QuadSurface& baseSurface,
                                   SurfacePatchIndex& baseIndex,
                                   const vc::lasagna::NormalSampler& normalSampler,
                                   const LineMappingOptions& options = {});

FiberMapping mapFiberToBaseSurface(const FiberInput& fiber,
                                   const QuadSurface& baseSurface,
                                   SurfacePatchIndex& baseIndex,
                                   const vc::lasagna::NormalSampler& normalSampler,
                                   const LineMappingOptions& options = {});

Atlas createSingleFiberAtlas(const std::filesystem::path& volpkgRoot,
                             const std::string& atlasName,
                             const FiberInput& fiber,
                             const SurfaceCandidate& baseSurface,
                             int zeroWindingColumn,
                             FiberMapping mapping);

} // namespace vc::atlas
