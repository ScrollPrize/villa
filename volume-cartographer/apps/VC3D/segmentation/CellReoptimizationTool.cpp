#include "CellReoptimizationTool.hpp"

#include <queue>
#include <cmath>

#include "SegmentationModule.hpp"
#include "SegmentationEditManager.hpp"
#include "../overlays/SegmentationOverlayController.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/util/QuadSurface.hpp"

CellReoptimizationTool::CellReoptimizationTool(
    SegmentationModule& module,
    SegmentationEditManager* editManager,
    SegmentationOverlayController* overlay,
    VCCollection* pointCollection,
    QObject* parent)
    : QObject(parent)
    , _module(module)
    , _editManager(editManager)
    , _overlay(overlay)
    , _pointCollection(pointCollection)
{
}

void CellReoptimizationTool::setConfig(const Config& config)
{
    _config = config;
}

void CellReoptimizationTool::setSurface(QuadSurface* surface)
{
    _surface = surface;
}

int CellReoptimizationTool::executeAtGridPosition(int row, int col)
{
    if (!_surface || !_overlay || !_pointCollection || !_editManager) {
        emit statusMessage(tr("Cell reoptimization: missing dependencies"), 2000);
        return 0;
    }

    // Check if start position is valid
    if (!isValidGridPosition(row, col)) {
        emit statusMessage(tr("Cell reoptimization: invalid grid position"), 2000);
        return 0;
    }

    // Check if start position is already approved (can't flood from approved area)
    if (isApprovalBoundary(row, col)) {
        emit statusMessage(tr("Cell reoptimization: cannot start from approved area"), 2000);
        return 0;
    }

    // Perform flood fill
    FloodResult result = performFloodFill(row, col);

    if (result.floodedCells.empty()) {
        emit statusMessage(tr("Cell reoptimization: flood fill found no cells"), 2000);
        return 0;
    }

    // Extract boundary world points
    std::vector<cv::Vec3f> boundaryPoints = extractBoundaryWorldPoints(result);

    if (boundaryPoints.empty()) {
        emit statusMessage(tr("Cell reoptimization: no boundary points found"), 2000);
        return 0;
    }

    // Create correction point collection
    std::string collectionName = _pointCollection->generateNewCollectionName("cell_reopt");
    _pointCollection->addPoints(collectionName, boundaryPoints);

    // Get the collection ID so we can register it with the corrections system
    uint64_t collectionId = _pointCollection->getCollectionId(collectionName);

    int numPoints = static_cast<int>(boundaryPoints.size());

    QString message = result.reachedMaxSteps
        ? tr("Placed %1 correction points (flood limited to %2 cells)")
              .arg(numPoints).arg(_config.maxFloodSteps)
        : tr("Placed %1 correction points around %2 cells")
              .arg(numPoints).arg(result.floodedCells.size());

    emit statusMessage(message, 3000);
    emit correctionPointsPlaced(numPoints);
    emit collectionCreated(collectionId);

    return numPoints;
}

CellReoptimizationTool::FloodResult CellReoptimizationTool::performFloodFill(int startRow, int startCol)
{
    return floodFillBFS(startRow, startCol);
}

std::vector<cv::Vec3f> CellReoptimizationTool::extractBoundaryWorldPoints(const FloodResult& result)
{
    if (result.boundaryCells.empty()) {
        return {};
    }

    // Sample boundary at equal intervals
    std::vector<std::pair<int, int>> sampledBoundary = sampleBoundaryPoints(
        result.boundaryCells,
        _config.maxCorrectionPoints,
        _config.minBoundarySpacing);

    // Apply perimeter offset if configured
    if (std::abs(_config.perimeterOffset) > 0.001f) {
        auto offsetBoundary = applyPerimeterOffset(sampledBoundary, _config.perimeterOffset);
        return gridToWorldCoordinatesFloat(offsetBoundary);
    }

    // Convert to world coordinates
    return gridToWorldCoordinates(sampledBoundary);
}

CellReoptimizationTool::FloodResult CellReoptimizationTool::floodFillBFS(int startRow, int startCol)
{
    FloodResult result;

    if (!isValidGridPosition(startRow, startCol) || isApprovalBoundary(startRow, startCol)) {
        return result;
    }

    std::queue<std::pair<int, int>> queue;
    std::set<std::pair<int, int>> visited;

    queue.push({startRow, startCol});
    visited.insert({startRow, startCol});

    // 4-connected neighbors for flood fill (prevents diagonal leakage)
    const int dr[] = {-1, 1, 0, 0};
    const int dc[] = {0, 0, -1, 1};

    while (!queue.empty() && static_cast<int>(result.floodedCells.size()) < _config.maxFloodSteps) {
        auto [row, col] = queue.front();
        queue.pop();

        result.floodedCells.push_back({row, col});

        for (int i = 0; i < 4; ++i) {
            int nr = row + dr[i];
            int nc = col + dc[i];

            if (!isValidGridPosition(nr, nc)) {
                continue;
            }

            if (isApprovalBoundary(nr, nc)) {
                continue;
            }

            if (visited.count({nr, nc}) == 0) {
                visited.insert({nr, nc});
                queue.push({nr, nc});
            }
        }
    }

    result.reachedMaxSteps = !queue.empty();

    // Build the set for boundary extraction
    std::set<std::pair<int, int>> floodedSet(result.floodedCells.begin(), result.floodedCells.end());

    // Extract ordered boundary
    result.boundaryCells = extractOrderedBoundary(floodedSet);

    return result;
}

std::vector<std::pair<int, int>> CellReoptimizationTool::extractOrderedBoundary(
    const std::set<std::pair<int, int>>& floodedSet)
{
    if (floodedSet.empty()) {
        return {};
    }

    // 8-connected neighbors for boundary detection
    const int dr8[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dc8[] = {-1, 0, 1, -1, 1, -1, 0, 1};

    // Find cells that are on the boundary (have at least one non-flooded neighbor)
    std::set<std::pair<int, int>> boundarySet;
    std::pair<int, int> startCell = {-1, -1};

    for (const auto& cell : floodedSet) {
        bool isBoundary = false;
        for (int i = 0; i < 8; ++i) {
            int nr = cell.first + dr8[i];
            int nc = cell.second + dc8[i];

            // A cell is on boundary if it has a neighbor outside the flooded set
            // This includes: approval boundary cells, invalid cells, or just outside flood
            if (floodedSet.count({nr, nc}) == 0) {
                isBoundary = true;
                break;
            }
        }

        if (isBoundary) {
            boundarySet.insert(cell);
            if (startCell.first < 0) {
                startCell = cell;
            }
        }
    }

    if (boundarySet.empty()) {
        return {};
    }

    // Contour following: walk around the boundary in order
    std::vector<std::pair<int, int>> orderedBoundary;
    std::set<std::pair<int, int>> visitedBoundary;

    orderedBoundary.push_back(startCell);
    visitedBoundary.insert(startCell);

    auto current = startCell;

    // Keep walking until we can't find any more unvisited boundary neighbors
    while (true) {
        bool foundNext = false;

        // Try to find the next boundary cell that's 8-connected to current
        for (int i = 0; i < 8; ++i) {
            int nr = current.first + dr8[i];
            int nc = current.second + dc8[i];

            if (boundarySet.count({nr, nc}) > 0 && visitedBoundary.count({nr, nc}) == 0) {
                orderedBoundary.push_back({nr, nc});
                visitedBoundary.insert({nr, nc});
                current = {nr, nc};
                foundNext = true;
                break;
            }
        }

        if (!foundNext) {
            break;
        }
    }

    return orderedBoundary;
}

std::vector<std::pair<int, int>> CellReoptimizationTool::sampleBoundaryPoints(
    const std::vector<std::pair<int, int>>& orderedBoundary,
    int maxPoints,
    float minSpacing)
{
    if (orderedBoundary.empty()) {
        return {};
    }

    if (static_cast<int>(orderedBoundary.size()) <= maxPoints) {
        // If we have fewer points than max, check if spacing is adequate
        // For now, just return all if the boundary is small
        if (orderedBoundary.size() <= 3) {
            return orderedBoundary;
        }
    }

    // Calculate total boundary length
    float totalLength = 0.0f;
    for (size_t i = 1; i < orderedBoundary.size(); ++i) {
        float dr = static_cast<float>(orderedBoundary[i].first - orderedBoundary[i - 1].first);
        float dc = static_cast<float>(orderedBoundary[i].second - orderedBoundary[i - 1].second);
        totalLength += std::sqrt(dr * dr + dc * dc);
    }

    if (totalLength <= 0.0f) {
        return {orderedBoundary.front()};
    }

    // Calculate ideal spacing
    int effectiveMaxPoints = std::min(maxPoints, static_cast<int>(orderedBoundary.size()));
    float idealSpacing = totalLength / static_cast<float>(std::max(1, effectiveMaxPoints - 1));
    float spacing = std::max(idealSpacing, minSpacing);

    // Sample at intervals
    std::vector<std::pair<int, int>> sampled;
    sampled.push_back(orderedBoundary.front());

    float accumulated = 0.0f;
    for (size_t i = 1; i < orderedBoundary.size(); ++i) {
        float dr = static_cast<float>(orderedBoundary[i].first - orderedBoundary[i - 1].first);
        float dc = static_cast<float>(orderedBoundary[i].second - orderedBoundary[i - 1].second);
        accumulated += std::sqrt(dr * dr + dc * dc);

        if (accumulated >= spacing) {
            sampled.push_back(orderedBoundary[i]);
            accumulated = 0.0f;

            if (static_cast<int>(sampled.size()) >= maxPoints) {
                break;
            }
        }
    }

    // Ensure we don't exceed maxPoints
    if (static_cast<int>(sampled.size()) > maxPoints) {
        sampled.resize(maxPoints);
    }

    return sampled;
}

std::vector<cv::Vec3f> CellReoptimizationTool::gridToWorldCoordinates(
    const std::vector<std::pair<int, int>>& gridPositions)
{
    std::vector<cv::Vec3f> worldCoords;
    worldCoords.reserve(gridPositions.size());

    for (const auto& [row, col] : gridPositions) {
        auto worldPos = _editManager->vertexWorldPosition(row, col);
        if (worldPos.has_value()) {
            worldCoords.push_back(*worldPos);
        }
    }

    return worldCoords;
}

bool CellReoptimizationTool::isApprovalBoundary(int row, int col) const
{
    if (!_overlay) {
        return false;
    }

    // queryApprovalStatus returns:
    // 0 = not approved, 1 = saved approved, 2 = pending approved, 3 = pending unapproved
    // We stop at anything approved (status 1 or 2)
    int status = _overlay->queryApprovalStatus(row, col);
    return status == 1 || status == 2;
}

bool CellReoptimizationTool::isValidGridPosition(int row, int col) const
{
    auto [rows, cols] = gridDimensions();
    return row >= 0 && row < rows && col >= 0 && col < cols;
}

std::pair<int, int> CellReoptimizationTool::gridDimensions() const
{
    if (!_surface) {
        return {0, 0};
    }

    const auto* points = _surface->rawPointsPtr();
    if (!points) {
        return {0, 0};
    }

    return {points->rows, points->cols};
}

std::vector<std::pair<float, float>> CellReoptimizationTool::applyPerimeterOffset(
    const std::vector<std::pair<int, int>>& boundaryPoints,
    float offset)
{
    if (boundaryPoints.empty()) {
        return {};
    }

    // Compute center of mass
    float centerRow = 0.0f;
    float centerCol = 0.0f;
    for (const auto& [row, col] : boundaryPoints) {
        centerRow += static_cast<float>(row);
        centerCol += static_cast<float>(col);
    }
    centerRow /= static_cast<float>(boundaryPoints.size());
    centerCol /= static_cast<float>(boundaryPoints.size());

    // Apply offset along radial direction from center
    std::vector<std::pair<float, float>> offsetPoints;
    offsetPoints.reserve(boundaryPoints.size());

    for (const auto& [row, col] : boundaryPoints) {
        float dr = static_cast<float>(row) - centerRow;
        float dc = static_cast<float>(col) - centerCol;
        float dist = std::sqrt(dr * dr + dc * dc);

        float newRow, newCol;
        if (dist > 0.001f) {
            // Normalize direction and apply offset
            float dirRow = dr / dist;
            float dirCol = dc / dist;
            newRow = static_cast<float>(row) + dirRow * offset;
            newCol = static_cast<float>(col) + dirCol * offset;
        } else {
            // Point is at center, no direction to offset
            newRow = static_cast<float>(row);
            newCol = static_cast<float>(col);
        }

        offsetPoints.emplace_back(newRow, newCol);
    }

    return offsetPoints;
}

std::vector<cv::Vec3f> CellReoptimizationTool::gridToWorldCoordinatesFloat(
    const std::vector<std::pair<float, float>>& gridPositions)
{
    std::vector<cv::Vec3f> worldCoords;
    worldCoords.reserve(gridPositions.size());

    for (const auto& [row, col] : gridPositions) {
        // Round to nearest integer for lookup, but use interpolation ideally
        // For now, use nearest neighbor
        int irow = static_cast<int>(std::round(row));
        int icol = static_cast<int>(std::round(col));

        // Clamp to valid range
        auto [rows, cols] = gridDimensions();
        irow = std::clamp(irow, 0, rows - 1);
        icol = std::clamp(icol, 0, cols - 1);

        auto worldPos = _editManager->vertexWorldPosition(irow, icol);
        if (worldPos.has_value()) {
            worldCoords.push_back(*worldPos);
        }
    }

    return worldCoords;
}
