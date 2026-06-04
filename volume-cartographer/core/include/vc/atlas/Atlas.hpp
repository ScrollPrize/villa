#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
    int version = 1;
    std::string name;
    std::filesystem::path baseMeshPath;
    std::filesystem::path sourceBaseMeshPath;
    int idxRotationColumns = 0;
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
    std::vector<AtlasAnchor> lineAnchors;
    std::vector<AtlasAnchor> controlAnchors;
};

struct Atlas {
    AtlasMetadata metadata;
    std::vector<std::string> links;
    std::vector<FiberMapping> fibers;

    void save(const std::filesystem::path& atlasDir) const;
    static Atlas load(const std::filesystem::path& atlasDir);
};

struct AtlasCoveredSize {
    int width = 0;
    int height = 0;
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
    double mismatchRatio = 1.5;
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

std::string sanitizeAtlasName(std::string name);
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

int computeIdxRotationColumns(const QuadSurface& surface);
std::shared_ptr<QuadSurface> idxRotatedSurface(const QuadSurface& surface,
                                               int rotationColumns);
void saveIdxRotatedBaseMesh(const QuadSurface& surface,
                            int rotationColumns,
                            const std::filesystem::path& targetDir);
AtlasCoveredSize mappedObjectCoveredAtlasSize(const Atlas& atlas);
AtlasDisplayRange atlasDisplayRange(const Atlas& atlas, int baseColumns);
cv::Vec2f atlasGridToSurfaceCoords(double atlasU,
                                   double atlasV,
                                   const QuadSurface& displaySurface,
                                   double atlasUOffset = 0.0);
std::shared_ptr<QuadSurface> repeatedAtlasDisplaySurface(const QuadSurface& baseSurface,
                                                        int unwrapCount);

FiberMapping mapFiberToBaseSurface(const FiberInput& fiber,
                                   const QuadSurface& baseSurface,
                                   SurfacePatchIndex& baseIndex,
                                   const vc::lasagna::NormalSampler& normalSampler,
                                   const LineMappingOptions& options = {});

Atlas createSingleFiberAtlas(const std::filesystem::path& volpkgRoot,
                             const std::string& atlasName,
                             const FiberInput& fiber,
                             const SurfaceCandidate& baseSurface,
                             int rotationColumns,
                             FiberMapping mapping);

} // namespace vc::atlas
