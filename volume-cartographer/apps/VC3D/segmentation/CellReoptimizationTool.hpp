#pragma once

#include <QObject>
#include <opencv2/core.hpp>
#include <set>
#include <utility>
#include <vector>

class QuadSurface;
class SegmentationModule;
class SegmentationEditManager;
class SegmentationOverlayController;
class VCCollection;

/**
 * Tool for cell-based reoptimization using approval mask boundaries.
 *
 * This tool allows users to click on an unapproved region in the flattened view,
 * flood fill to identify the connected unapproved area bounded by the approval mask,
 * extract the boundary, and automatically place correction points around that boundary
 * for subsequent reoptimization.
 */
class CellReoptimizationTool : public QObject
{
    Q_OBJECT

public:
    struct Config
    {
        int maxFloodSteps = 500;         // Maximum cells to include in flood fill
        int maxCorrectionPoints = 50;    // Maximum number of correction points to place
        float minBoundarySpacing = 5.0f; // Minimum spacing between points (grid steps)
        float perimeterOffset = 0.0f;    // Offset to apply to boundary (positive=expand, negative=shrink)
    };

    struct FloodResult
    {
        std::vector<std::pair<int, int>> floodedCells;  // All cells in flooded region
        std::vector<std::pair<int, int>> boundaryCells; // Ordered boundary cells
        bool reachedMaxSteps = false;                   // True if flood was limited by maxFloodSteps
    };

    CellReoptimizationTool(SegmentationModule& module,
                           SegmentationEditManager* editManager,
                           SegmentationOverlayController* overlay,
                           VCCollection* pointCollection,
                           QObject* parent = nullptr);

    void setConfig(const Config& config);
    [[nodiscard]] Config config() const { return _config; }

    void setSurface(QuadSurface* surface);

    /**
     * Execute cell reoptimization at the given grid position.
     * Performs flood fill, extracts boundary, and places correction points.
     * @param row Grid row of click position
     * @param col Grid column of click position
     * @return Number of correction points placed, or 0 if operation failed
     */
    int executeAtGridPosition(int row, int col);

    /**
     * Perform flood fill from the given start position.
     * Stops at approval boundaries or when maxFloodSteps is reached.
     */
    FloodResult performFloodFill(int startRow, int startCol);

    /**
     * Extract world coordinates for boundary points from flood result.
     * Samples the boundary at roughly equal intervals.
     */
    std::vector<cv::Vec3f> extractBoundaryWorldPoints(const FloodResult& result);

signals:
    void statusMessage(const QString& message, int timeoutMs);
    void correctionPointsPlaced(int count);
    void collectionCreated(uint64_t collectionId);

private:
    /**
     * BFS flood fill implementation.
     * Uses 4-connectivity for flood fill (prevents diagonal leakage).
     */
    FloodResult floodFillBFS(int startRow, int startCol);

    /**
     * Extract ordered boundary cells using contour following.
     * Uses 8-connectivity for smoother boundary.
     */
    std::vector<std::pair<int, int>> extractOrderedBoundary(
        const std::set<std::pair<int, int>>& floodedSet);

    /**
     * Sample boundary points at roughly equal arc-length intervals.
     */
    std::vector<std::pair<int, int>> sampleBoundaryPoints(
        const std::vector<std::pair<int, int>>& orderedBoundary,
        int maxPoints,
        float minSpacing);

    /**
     * Convert grid positions to world coordinates.
     */
    std::vector<cv::Vec3f> gridToWorldCoordinates(
        const std::vector<std::pair<int, int>>& gridPositions);

    /**
     * Convert floating-point grid positions to world coordinates (for offset points).
     */
    std::vector<cv::Vec3f> gridToWorldCoordinatesFloat(
        const std::vector<std::pair<float, float>>& gridPositions);

    /**
     * Apply perimeter offset to boundary points.
     * Computes center of mass and moves each point along the radial direction.
     * Positive offset expands (repels from center), negative shrinks (attracts to center).
     */
    std::vector<std::pair<float, float>> applyPerimeterOffset(
        const std::vector<std::pair<int, int>>& boundaryPoints,
        float offset);

    /**
     * Check if a cell is an approval boundary (stops flood fill).
     * Returns true if queryApprovalStatus > 0.
     */
    bool isApprovalBoundary(int row, int col) const;

    /**
     * Check if grid position is valid (within surface bounds).
     */
    bool isValidGridPosition(int row, int col) const;

    /**
     * Get grid dimensions from the current surface.
     */
    std::pair<int, int> gridDimensions() const;

    SegmentationModule& _module;
    SegmentationEditManager* _editManager{nullptr};
    SegmentationOverlayController* _overlay{nullptr};
    VCCollection* _pointCollection{nullptr};
    QuadSurface* _surface{nullptr};
    Config _config;
};
