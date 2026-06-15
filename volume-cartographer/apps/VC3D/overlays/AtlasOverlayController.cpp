#include "AtlasOverlayController.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <QRectF>

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
constexpr const char* kOverlayGroup = "atlas_objects";

ViewerOverlayControllerBase::OverlayStyle atlasLineStyle()
{
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(220, 60, 50);
    style.brushColor = Qt::transparent;
    style.penWidth = 2.0;
    style.z = 50.0;
    return style;
}

ViewerOverlayControllerBase::OverlayStyle atlasAnchorStyle()
{
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(255, 230, 70);
    style.brushColor = QColor(255, 230, 70);
    style.penWidth = 0.0;
    style.z = 60.0;
    return style;
}

ViewerOverlayControllerBase::OverlayStyle selectedAtlasLineStyle()
{
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(80, 210, 255);
    style.brushColor = Qt::transparent;
    style.penWidth = 3.0;
    style.z = 66.0;
    return style;
}

ViewerOverlayControllerBase::OverlayStyle selectedAtlasAnchorStyle()
{
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(80, 210, 255);
    style.brushColor = QColor(80, 210, 255);
    style.penWidth = 1.0;
    style.z = 70.0;
    return style;
}

ViewerOverlayControllerBase::OverlayStyle selectedAtlasControlStyle()
{
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(255, 70, 80);
    style.brushColor = QColor(255, 70, 80, 230);
    style.penWidth = 1.5;
    style.z = 74.0;
    return style;
}

ViewerOverlayControllerBase::OverlayStyle searchCrossStyle(bool emphasized)
{
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = emphasized ? QColor(80, 210, 255) : QColor(255, 255, 255);
    style.brushColor = Qt::transparent;
    style.penWidth = emphasized ? 2.2 : 1.4;
    style.z = emphasized ? 86.0 : 84.0;
    return style;
}

ViewerOverlayControllerBase::OverlayStyle searchPreviewLineStyle(bool emphasized)
{
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = emphasized ? QColor(80, 210, 255) : QColor(255, 255, 255);
    style.brushColor = Qt::transparent;
    style.penWidth = emphasized ? 2.4 : 1.6;
    style.penStyle = Qt::DashLine;
    style.z = emphasized ? 82.0 : 80.0;
    return style;
}

std::optional<std::pair<int, int>> controlAnchorSourceRange(
    const vc::atlas::FiberMapping& fiber)
{
    if (fiber.lineAnchors.empty()) {
        return std::nullopt;
    }

    int first = std::numeric_limits<int>::max();
    int last = std::numeric_limits<int>::min();
    for (const auto& anchor : fiber.controlAnchors) {
        first = std::min(first, anchor.sourceIndex);
        last = std::max(last, anchor.sourceIndex);
    }
    if (fiber.controlAnchors.size() < 2 || first > last) {
        return std::nullopt;
    }
    return std::make_pair(first, last);
}

bool anchorInControlRange(const vc::atlas::AtlasAnchor& anchor,
                          const std::optional<std::pair<int, int>>& range)
{
    if (!range) {
        return true;
    }
    return anchor.sourceIndex >= range->first && anchor.sourceIndex <= range->second;
}
}

AtlasOverlayController::AtlasOverlayController(QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroup, parent)
{
}

void AtlasOverlayController::setAtlas(vc::atlas::Atlas atlas,
                                      std::shared_ptr<const QuadSurface> displaySurface,
                                      vc::atlas::AtlasDisplayRange displayRange)
{
    _atlas = std::move(atlas);
    _displaySurface = std::move(displaySurface);
    _displayRange = displayRange;
    clearSearchPreviews();
    refreshAll();
}

void AtlasOverlayController::clearAtlas()
{
    _atlas.reset();
    _displaySurface.reset();
    _displayRange = {};
    _searchPreviewCandidates.clear();
    _searchPreviewFibers.clear();
    _hoverSearchResult.reset();
    _selectedSearchResults.clear();
    _selectedFiberPathKeys.clear();
    _selectedControlPoint.reset();
    refreshAll();
}

void AtlasOverlayController::clearSearchPreviews()
{
    _searchPreviewCandidates.clear();
    _searchPreviewFibers.clear();
    _hoverSearchResult.reset();
    _selectedSearchResults.clear();
    _selectedFiberPathKeys.clear();
    _selectedControlPoint.reset();
    refreshAll();
}

void AtlasOverlayController::setSearchPreviewCandidates(std::vector<SearchPreviewCandidate> candidates)
{
    _searchPreviewCandidates = std::move(candidates);
    _searchPreviewFibers.clear();
    _hoverSearchResult.reset();
    _selectedSearchResults.clear();
    refreshAll();
}

void AtlasOverlayController::setSearchPreviewHover(std::optional<int> resultIndex)
{
    if (_hoverSearchResult == resultIndex) {
        return;
    }
    _hoverSearchResult = resultIndex;
    refreshAll();
}

void AtlasOverlayController::setSearchPreviewSelection(std::set<int> resultIndices)
{
    if (_selectedSearchResults == resultIndices) {
        return;
    }
    _selectedSearchResults = std::move(resultIndices);
    refreshAll();
}

void AtlasOverlayController::setSearchPreviewFiber(SearchPreviewFiber fiber)
{
    if (fiber.resultIndex < 0) {
        return;
    }
    _searchPreviewFibers[fiber.resultIndex] = std::move(fiber.mapping);
    refreshAll();
}

void AtlasOverlayController::setSelectedFiberPaths(std::set<std::string> fiberPathKeys)
{
    if (_selectedFiberPathKeys == fiberPathKeys) {
        return;
    }
    _selectedFiberPathKeys = std::move(fiberPathKeys);
    refreshAll();
}

void AtlasOverlayController::setSelectedControlPoint(
    std::optional<std::pair<std::string, int>> controlPoint)
{
    if (_selectedControlPoint == controlPoint) {
        return;
    }
    _selectedControlPoint = std::move(controlPoint);
    refreshAll();
}

std::optional<cv::Vec2f> AtlasOverlayController::atlasAnchorToSurface(
    const vc::atlas::AtlasAnchor& anchor,
    const vc::atlas::FiberMapping& fiber) const
{
    if (!_displaySurface || !std::isfinite(anchor.atlasU) || !std::isfinite(anchor.atlasV)) {
        return std::nullopt;
    }
    const double atlasU = vc::atlas::actualAtlasU(anchor, fiber, _displayRange.baseColumns);
    const cv::Vec2f surfaceCoord =
        vc::atlas::atlasGridToSurfaceCoords(atlasU,
                                            anchor.atlasV,
                                            *_displaySurface,
                                            _displayRange.atlasUOffset);
    if (!std::isfinite(surfaceCoord[0]) || !std::isfinite(surfaceCoord[1])) {
        return std::nullopt;
    }
    return surfaceCoord;
}

std::optional<QRectF> AtlasOverlayController::surfaceBounds() const
{
    if (!_atlas) {
        return std::nullopt;
    }

    bool havePoint = false;
    float minX = std::numeric_limits<float>::infinity();
    float minY = std::numeric_limits<float>::infinity();
    float maxX = -std::numeric_limits<float>::infinity();
    float maxY = -std::numeric_limits<float>::infinity();
    for (const auto& fiber : _atlas->fibers) {
        const auto sourceRange = controlAnchorSourceRange(fiber);
        for (const auto& anchor : fiber.lineAnchors) {
            if (!anchorInControlRange(anchor, sourceRange)) {
                continue;
            }
            const auto surfaceCoord = atlasAnchorToSurface(anchor, fiber);
            if (!surfaceCoord) {
                continue;
            }
            havePoint = true;
            minX = std::min(minX, (*surfaceCoord)[0]);
            minY = std::min(minY, (*surfaceCoord)[1]);
            maxX = std::max(maxX, (*surfaceCoord)[0]);
            maxY = std::max(maxY, (*surfaceCoord)[1]);
        }
    }
    if (!havePoint) {
        return std::nullopt;
    }
    return QRectF(QPointF(minX, minY), QPointF(maxX, maxY)).normalized();
}

bool AtlasOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return viewer && _atlas.has_value() && _displaySurface != nullptr;
}

void AtlasOverlayController::collectPrimitives(VolumeViewerBase* viewer,
                                               OverlayBuilder& builder)
{
    if (!isOverlayEnabledFor(viewer)) {
        return;
    }

    const auto lineStyle = atlasLineStyle();
    const auto anchorStyle = atlasAnchorStyle();
    const auto selectedLineStyle = selectedAtlasLineStyle();
    const auto selectedAnchorStyle = selectedAtlasAnchorStyle();
    const auto selectedControlStyle = selectedAtlasControlStyle();
    for (const auto& fiber : _atlas->fibers) {
        const std::string fiberPathKey = vc::atlas::atlasFiberPathKey(fiber.fiberPath);
        const bool selectedFiber =
            _selectedFiberPathKeys.find(fiberPathKey) != _selectedFiberPathKeys.end();
        const auto sourceRange = controlAnchorSourceRange(fiber);
        std::vector<cv::Vec2f> linePoints;
        linePoints.reserve(fiber.lineAnchors.size());
        for (const auto& anchor : fiber.lineAnchors) {
            if (!anchorInControlRange(anchor, sourceRange)) {
                continue;
            }
            const auto surfaceCoord = atlasAnchorToSurface(anchor, fiber);
            if (!surfaceCoord) {
                continue;
            }
            linePoints.push_back(*surfaceCoord);
        }
        if (linePoints.size() >= 2) {
            builder.addSurfaceLineStrip(linePoints,
                                        false,
                                        selectedFiber ? selectedLineStyle : lineStyle);
        }
        for (const auto& anchor : fiber.controlAnchors) {
            const auto surfaceCoord = atlasAnchorToSurface(anchor, fiber);
            if (!surfaceCoord) {
                continue;
            }
            const bool selectedControl =
                _selectedControlPoint &&
                _selectedControlPoint->first == fiberPathKey &&
                _selectedControlPoint->second == anchor.sourceIndex;
            builder.addSurfacePoint(*surfaceCoord,
                                    selectedControl ? 6.5 : (selectedFiber ? 5.0 : 4.0),
                                    selectedControl
                                        ? selectedControlStyle
                                        : (selectedFiber ? selectedAnchorStyle : anchorStyle));
        }
    }

    auto isEmphasized = [this](int resultIndex) {
        return _hoverSearchResult == resultIndex ||
               _selectedSearchResults.find(resultIndex) != _selectedSearchResults.end();
    };

    constexpr float crossHalfSize = 4.0f;
    for (const auto& candidate : _searchPreviewCandidates) {
        if (candidate.resultIndex < 0 ||
            !std::isfinite(candidate.surfaceCoord[0]) ||
            !std::isfinite(candidate.surfaceCoord[1])) {
            continue;
        }
        const auto style = searchCrossStyle(isEmphasized(candidate.resultIndex));
        builder.addSurfaceLineStrip({
            cv::Vec2f(candidate.surfaceCoord[0] - crossHalfSize, candidate.surfaceCoord[1]),
            cv::Vec2f(candidate.surfaceCoord[0] + crossHalfSize, candidate.surfaceCoord[1]),
        }, false, style);
        builder.addSurfaceLineStrip({
            cv::Vec2f(candidate.surfaceCoord[0], candidate.surfaceCoord[1] - crossHalfSize),
            cv::Vec2f(candidate.surfaceCoord[0], candidate.surfaceCoord[1] + crossHalfSize),
        }, false, style);
    }

    std::set<int> visiblePreviewResults = _selectedSearchResults;
    if (_hoverSearchResult) {
        visiblePreviewResults.insert(*_hoverSearchResult);
    }
    for (const int resultIndex : visiblePreviewResults) {
        const auto it = _searchPreviewFibers.find(resultIndex);
        if (it == _searchPreviewFibers.end()) {
            continue;
        }
        const auto& mapping = it->second;
        std::vector<cv::Vec2f> linePoints;
        linePoints.reserve(mapping.lineAnchors.size());
        for (const auto& anchor : mapping.lineAnchors) {
            const auto surfaceCoord = atlasAnchorToSurface(anchor, mapping);
            if (!surfaceCoord) {
                continue;
            }
            linePoints.push_back(*surfaceCoord);
        }
        if (linePoints.size() >= 2) {
            builder.addSurfaceLineStrip(linePoints,
                                        false,
                                        searchPreviewLineStyle(isEmphasized(resultIndex)));
        }
    }
}
