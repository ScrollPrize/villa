#pragma once

#include <filesystem>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>

#include "vc/lasagna/LineModel.hpp"

class QuadSurface;
class SurfacePatchIndex;

namespace vc::lasagna {
struct LasagnaDatasetManifest;
struct NormalSample;
class NormalSampler;
class LasagnaNormalSampler;
}

namespace vc::atlas {

struct AtlasMetadata {
    std::string type = "vc3d_atlas";
    int version = 5;
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
    static Atlas load(const std::filesystem::path& atlasDir,
                      const std::filesystem::path& volpkgRoot);
    static Atlas load(const std::filesystem::path& atlasDir,
                      const std::filesystem::path& volpkgRoot,
                      const std::filesystem::path& fiberPathRoot);
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

struct AtlasLayoutConflict {
    std::filesystem::path fiberPath;
    int existingOffset = 0;
    int candidateOffset = 0;
};

struct AtlasBaseMappingContext {
    std::shared_ptr<QuadSurface> baseSurface;
    std::shared_ptr<SurfacePatchIndex> baseIndex;
};

struct AtlasSignedWindingDisplay {
    double signedWindingDistance = 0.0;
    bool sourceFiberIsH = true;
    int hAnchorSourceIndex = -1;
    double hToVOutwardProjection = 0.0;
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

enum class AtlasPredSnapSource {
    Auto,
    Optimized,
    Manual,
};

enum class AtlasPredSnapDirection {
    Inside,
    Outside,
};

struct AtlasPredSnapCandidate {
    cv::Vec3d point{0.0, 0.0, 0.0};
    std::optional<double> predDtValue;
    std::optional<AtlasPredSnapDirection> direction;
    std::optional<double> windingDistance;
};

struct AtlasPredSnapPoint {
    std::filesystem::path fiberPath;
    std::optional<int> sourceIndex;
    cv::Vec3d controlPoint{0.0, 0.0, 0.0};
    std::optional<cv::Vec3d> predSnapPoint;
    std::vector<AtlasPredSnapCandidate> candidates;
    std::optional<int> selectedCandidateIndex;
    AtlasPredSnapSource source = AtlasPredSnapSource::Auto;
    std::string status;
    std::string statusReason;
    std::optional<double> predDtValue;
    std::optional<AtlasPredSnapDirection> direction;
    std::optional<double> weightedFirstHitWindingDistance;
    std::optional<cv::Vec3d> searchNormal;
    std::string generatedAtUtc;
};

struct AtlasPredSnapSet {
    std::filesystem::path fiberPath;
    std::vector<AtlasPredSnapPoint> points;
};

struct AtlasPredSnapSampling {
    std::function<vc::lasagna::NormalSample(const cv::Vec3d&)> sampleNormal;
    std::function<std::optional<double>(const cv::Vec3d&)> samplePredDt;
    std::function<double(const cv::Vec3d&, const cv::Vec3d&, double)> windingDistance;
    double predDtThreshold = 110.0;
    double predDtStepVx = 0.05;
    double outwardWindingLimit = 1.0;
    double inwardWindingLimit = 1.0;
};

struct AtlasPredSnapAttachmentReport {
    size_t fibersChecked = 0;
    size_t attachmentsCreated = 0;
};

struct AtlasSnapCandidateSet {
    std::string id;
    std::filesystem::path fiberPath;
    int sourceIndex = 0;
    cv::Vec3d controlPoint{0.0, 0.0, 0.0};
    std::vector<cv::Vec3d> candidates;
    bool fixed = false;
    bool manual = false;
    bool eligible = true;
    std::string status;
    std::string statusReason;
};

struct AtlasSnapPairTerm {
    std::string id;
    size_t firstControl = 0;
    size_t secondControl = 0;
};

struct AtlasSnapPairMatrix {
    std::string id;
    std::vector<std::vector<double>> rawValues;
    std::vector<std::vector<double>> normalizedValues;
    nlohmann::json metadata;
};

struct AtlasSnapOptimizationProblem {
    std::vector<AtlasSnapCandidateSet> controls;
    std::vector<AtlasSnapPairTerm> terms;
};

struct AtlasSnapOptimizationResult {
    std::vector<size_t> selectedCandidateIndices;
    double objective = 0.0;
};

struct AtlasSnapOptimizeOptions {
    nlohmann::json rankOptions = nlohmann::json::object();
    int predDtThreshold = 110;
    size_t exhaustiveAssignmentLimit = 1000000;
};

struct AtlasSnapOptimizeReport {
    size_t controls = 0;
    size_t fixedControls = 0;
    size_t manualControls = 0;
    size_t singletonControls = 0;
    size_t links = 0;
    size_t variableControls = 0;
    size_t unscoredVariableControls = 0;
    size_t pairTerms = 0;
    size_t skippedPairTerms = 0;
    size_t cacheHits = 0;
    size_t rankJobsRequested = 0;
    size_t successfulPairTerms = 0;
    size_t zeroContributionTerms = 0;
    double objective = 0.0;
};

using AtlasSnapPairRankProgress =
    std::function<void(size_t resultIndex, const nlohmann::json& result)>;
using AtlasSnapPairRanker = std::function<nlohmann::json(
    const nlohmann::json& request,
    const AtlasSnapPairRankProgress& onResult)>;

struct LasagnaAtlasObject {
    std::string id;
    std::filesystem::path fiberPath;
    std::filesystem::path mappingPath;
    std::filesystem::path predSnapAttachmentPath;
    std::filesystem::path fiberRelativePath;
    std::filesystem::path mappingRelativePath;
    std::filesystem::path predSnapAttachmentRelativePath;
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
std::string atlasPredSnapControlPointKey(const cv::Vec3d& point);
std::filesystem::path atlasPredSnapAttachmentPath(
    const std::filesystem::path& atlasDir,
    const std::filesystem::path& fiberPath);
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
    const std::filesystem::path& volpkgRoot = {},
    const std::filesystem::path& fiberPathRoot = {});
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
std::optional<cv::Vec3d> atlasAnchorBaseNormal(const AtlasAnchor& anchor,
                                               const FiberMapping& fiber,
                                               const QuadSurface& baseSurface);
AtlasSignedWindingDisplay signedAtlasSearchWindingDisplay(
    double windingDistance,
    bool sourceFiberDisplaysAsH,
    double sourceLinePosition,
    double targetLinePosition,
    const cv::Vec3d& sourcePoint,
    const cv::Vec3d& targetPoint,
    const FiberMapping& sourceMapping,
    const FiberMapping& targetMapping,
    const QuadSurface& baseSurface);
AtlasDisplayRange atlasDisplayRange(const Atlas& atlas, int baseColumns);
int atlasLinkWindingOffsetDelta(const AtlasLink& link,
                                int periodColumns,
                                int zeroWindingColumn);
std::vector<AtlasLayoutConflict> layoutAtlasObjects(Atlas& atlas, int periodColumns);
cv::Vec2f atlasGridToSurfaceCoords(double atlasU,
                                   double atlasV,
                                   const QuadSurface& displaySurface,
                                   double atlasUOffset = 0.0);
std::shared_ptr<QuadSurface> repeatedAtlasDisplaySurface(const QuadSurface& baseSurface,
                                                        int unwrapCount,
                                                        int startColumn = 0);

bool atlasPredDtIsInside(double predDtValue, double threshold = 110.0);
std::vector<AtlasPredSnapCandidate> findAtlasPredSnapCandidates(
    const cv::Vec3d& controlPoint,
    const cv::Vec3d& alignedNormal,
    const AtlasPredSnapSampling& sampling);
AtlasPredSnapSet generateAtlasPredSnapSet(
    const FiberInput& fiber,
    const FiberMapping& mapping,
    const QuadSurface& baseSurface,
    const AtlasPredSnapSampling& sampling);
AtlasPredSnapSet generateAtlasPredSnapSet(
    const FiberInput& fiber,
    const FiberMapping& mapping,
    const QuadSurface& baseSurface,
    const vc::lasagna::LasagnaNormalSampler& sampler);
AtlasPredSnapSet loadAtlasPredSnapSet(
    const std::filesystem::path& attachmentPath);
void saveAtlasPredSnapSet(const std::filesystem::path& attachmentPath,
                          const AtlasPredSnapSet& set);
AtlasPredSnapSet mergeAtlasPredSnapSetByControlPoint(
    AtlasPredSnapSet existing,
    const AtlasPredSnapSet& generated);
AtlasPredSnapSet ensureAtlasPredSnapSet(
    const std::filesystem::path& atlasDir,
    const FiberInput& fiber,
    const FiberMapping& mapping,
    const QuadSurface& baseSurface,
    const AtlasPredSnapSampling& sampling);
AtlasPredSnapSet ensureAtlasPredSnapSet(
    const std::filesystem::path& atlasDir,
    const FiberInput& fiber,
    const FiberMapping& mapping,
    const QuadSurface& baseSurface,
    const vc::lasagna::LasagnaNormalSampler& sampler);
AtlasPredSnapSet setManualAtlasPredSnapPoint(
    const std::filesystem::path& atlasDir,
    const std::filesystem::path& fiberPath,
    const cv::Vec3d& controlPoint,
    const cv::Vec3d& predSnapPoint,
    std::optional<double> predDtValue = std::nullopt);
AtlasPredSnapAttachmentReport ensureAtlasPredSnapAttachments(
    const std::filesystem::path& atlasDir,
    const std::filesystem::path& volpkgRoot,
    const vc::lasagna::LasagnaNormalSampler& sampler);
std::filesystem::path atlasPredSnapRankCachePath(
    const std::filesystem::path& atlasDir);
std::string atlasSnapRankTermCacheKey(
    const std::filesystem::path& manifestPath,
    const nlohmann::json& rankOptions,
    const std::vector<cv::Vec3d>& sideA,
    const std::vector<cv::Vec3d>& sideB);
AtlasSnapOptimizationProblem buildAtlasSnapOptimizationProblem(
    const Atlas& atlas,
    const std::unordered_map<std::string, AtlasPredSnapSet>& predSnapSets);
AtlasSnapPairMatrix atlasSnapPairMatrixFromRankResult(
    const AtlasSnapPairTerm& term,
    size_t sideACount,
    size_t sideBCount,
    const nlohmann::json& result);
AtlasSnapOptimizationResult optimizeAtlasSnapCandidates(
    const AtlasSnapOptimizationProblem& problem,
    const std::vector<AtlasSnapPairMatrix>& matrices,
    const AtlasSnapOptimizeOptions& options = {});
AtlasSnapOptimizeReport optimizeAtlasPredSnapCandidates(
    const std::filesystem::path& atlasDir,
    const std::filesystem::path& volpkgRoot,
    const std::filesystem::path& manifestPath,
    const vc::lasagna::LasagnaNormalSampler& sampler,
    const AtlasSnapPairRanker& ranker,
    const AtlasSnapOptimizeOptions& options = {});

void validateFiberInputControlPoints(FiberInput& fiber);
bool atlasLoadErrorRequiresRebuild(const std::exception& ex);
AtlasBaseMappingContext atlasBaseMappingContextFromSurface(
    std::shared_ptr<QuadSurface> baseSurface);
AtlasBaseMappingContext loadAtlasBaseMappingContext(const std::filesystem::path& atlasDir,
                                                    const Atlas& atlas);
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
