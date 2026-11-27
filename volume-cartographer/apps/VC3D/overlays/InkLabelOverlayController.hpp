#pragma once

// Draws the open ink label session on the segment view, above the ink
// prediction. Labels are shown as display tiles at a level of detail picked
// from the zoom: level k pools 2^k x 2^k label pixels into one image pixel,
// so a whole segment stays cheap to draw. Images are Indexed8 over 16 display
// codes; a style change only swaps their color table.

#include "ViewerOverlayControllerBase.hpp"

#include <QColor>
#include <QHash>
#include <QImage>
#include <QPixmap>
#include <QRect>
#include <QTransform>

#include <opencv2/core/types.hpp>

#include <array>
#include <cstdint>
#include <list>
#include <set>
#include <vector>

namespace vc3d::ink_labels {
struct SuggestionSet;
}

class InkLabelController;
class VolumeViewerBase;

class InkLabelOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    static constexpr int kLevels = 9;
    // The display cache holds at least this many tiles, and at least twice
    // the tiles all viewers showed in their last frame, so a 4K view zoomed
    // in does not rebuild its tiles on every repaint.
    static constexpr int kMinCachedTiles = 384;
    static constexpr qreal kZ = 19.0;

    explicit InkLabelOverlayController(const InkLabelController& labels, QObject* parent = nullptr);

    // Label pixels changed: display tiles covering them are rebuilt on next use.
    void invalidate(const QRect& labelRect);
    void setStyle(bool ink, bool supervision, bool validation, double opacity);
    void setPolygonPreview(std::vector<cv::Point2d> labelPx, QColor color);
    // Hatched preview of threshold suggestions: picked components in the ink
    // color, the rest in cyan. Null clears it.
    void setSuggestionPreview(const vc3d::ink_labels::SuggestionSet* suggestions, const std::set<int>& selected);
    // The preview's code image (0 none, 1 and 2 unpicked stripes, 3 and 4
    // picked stripes) and its color table.
    [[nodiscard]] static QList<QRgb> suggestionColorTable();
    // The session (or its canvas) changed: drop every cached tile.
    void resetForSession();

    // Display level for a zoom of `devicePxPerLabelPx`: floor(log2(1/d)),
    // clamped to [0, 8].
    [[nodiscard]] static int lodFor(double devicePxPerLabelPx);
    // The 16-entry color table for a style.
    [[nodiscard]] static QList<QRgb> colorTableFor(bool ink, bool supervision, bool validation, double opacity);
    // A scene point moved to the nearest device pixel corner. Display tiles
    // that share an edge in label space then share it to the device pixel, so
    // no seam opens between them at fractional zooms.
    [[nodiscard]] static QPointF snapToDevicePixel(const QPointF& scenePos, const QTransform& toDevice);
    // The cache capacity for this many tiles visible across all viewers.
    [[nodiscard]] static int cacheCapacityFor(int visibleTiles);

#ifdef VC_TESTING
    // The code image of display tile (level, i, j), built or cached.
    [[nodiscard]] QImage displayTileForTesting(int level, int i, int j);
    [[nodiscard]] uint32_t revisionForTesting(int level, int i, int j) const { return revision(level, i, j); }
    [[nodiscard]] int buildsForTesting() const { return _builds; }
    [[nodiscard]] int cacheCapacityForTesting() const { return cacheCapacity(); }
    [[nodiscard]] const QImage& suggestionImageForTesting() const { return _suggestionImage; }
#endif

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    struct CacheEntry {
        uint32_t revision = 0;
        QImage image;
        QPixmap pixmap;
        std::list<quint64>::iterator lru;
    };

    [[nodiscard]] static quint64 keyFor(int level, int i, int j);
    [[nodiscard]] uint32_t revision(int level, int i, int j) const;
    CacheEntry& ensureTile(int level, int i, int j);
    [[nodiscard]] QImage buildTile(int level, int i, int j) const;

    const InkLabelController& _labels;
    std::array<std::vector<uint32_t>, kLevels> _revisions;
    std::array<cv::Size, kLevels> _levelGrid;
    [[nodiscard]] int cacheCapacity() const;
    QHash<quint64, CacheEntry> _cache;
    // Tiles each viewer drew in its last frame.
    QHash<const VolumeViewerBase*, int> _visibleTiles;
    std::list<quint64> _lru;
    QList<QRgb> _colorTable;
    std::vector<cv::Point2d> _polygon;
    QColor _polygonColor;
    QImage _suggestionImage;
    QPixmap _suggestionPixmap;
    cv::Point _suggestionOrigin;
    int _builds = 0;
};
