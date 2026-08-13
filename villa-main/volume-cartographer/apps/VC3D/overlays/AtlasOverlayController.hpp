#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include "vc/atlas/Atlas.hpp"

#include <QRectF>

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

class QuadSurface;

class AtlasOverlayController : public ViewerOverlayControllerBase
{
public:
    struct SearchPreviewCandidate {
        int resultIndex = -1;
        cv::Vec2f surfaceCoord{0.0f, 0.0f};
    };

    struct SearchPreviewFiber {
        int resultIndex = -1;
        vc::atlas::FiberMapping mapping;
    };

    explicit AtlasOverlayController(QObject* parent = nullptr);

    void setAtlas(vc::atlas::Atlas atlas,
                  std::shared_ptr<const QuadSurface> displaySurface,
                  vc::atlas::AtlasDisplayRange displayRange);
    void clearAtlas();
    void clearSearchPreviews();
    void setSearchPreviewCandidates(std::vector<SearchPreviewCandidate> candidates);
    void setSearchPreviewHover(std::optional<int> resultIndex);
    void setSearchPreviewSelection(std::set<int> resultIndices);
    void setSearchPreviewFiber(SearchPreviewFiber fiber);
    void setSelectedFiberPaths(std::set<std::string> fiberPathKeys);
    void setSelectedControlPoint(std::optional<std::pair<std::string, int>> controlPoint);

    [[nodiscard]] std::optional<QRectF> surfaceBounds() const;
    [[nodiscard]] std::optional<cv::Vec2f> atlasAnchorToSurface(
        const vc::atlas::AtlasAnchor& anchor,
        const vc::atlas::FiberMapping& fiber) const;

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    std::optional<vc::atlas::Atlas> _atlas;
    std::shared_ptr<const QuadSurface> _displaySurface;
    vc::atlas::AtlasDisplayRange _displayRange;
    std::vector<SearchPreviewCandidate> _searchPreviewCandidates;
    std::map<int, vc::atlas::FiberMapping> _searchPreviewFibers;
    std::optional<int> _hoverSearchResult;
    std::set<int> _selectedSearchResults;
    std::set<std::string> _selectedFiberPathKeys;
    std::optional<std::pair<std::string, int>> _selectedControlPoint;
};
