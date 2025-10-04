#pragma once

#include <optional>
#include <unordered_set>
#include <vector>

#include <QVector>
#include <QString>

#include <opencv2/core.hpp>

#include "SegmentationGrowth.hpp"

class SegmentationModule;
class SegmentationWidget;
class VCCollection;

namespace segmentation
{
class CorrectionsState
{
public:
    CorrectionsState(SegmentationModule& module,
                     SegmentationWidget* widget,
                     VCCollection* collection);

    void setWidget(SegmentationWidget* widget);
    void setCollection(VCCollection* collection);

    bool setAnnotateMode(bool enabled, bool userInitiated, bool editingEnabled);
    void setActiveCollection(uint64_t collectionId, bool userInitiated);
    uint64_t createCollection(bool announce);
    void handlePointAdded(const cv::Vec3f& worldPos);
    void handlePointRemoved(const cv::Vec3f& worldPos);

    void onZRangeChanged(bool enabled, int zMin, int zMax);

    void setGrowthInProgress(bool running);
    void clearAll(bool editingEnabled);
    void refreshWidget();
    void pruneMissing();

    [[nodiscard]] bool annotateMode() const { return _annotateMode; }
    [[nodiscard]] bool growthInProgress() const { return _growthInProgress; }
    [[nodiscard]] uint64_t activeCollection() const { return _activeCollectionId; }
    [[nodiscard]] std::optional<std::pair<int, int>> zRange() const;
    [[nodiscard]] SegmentationCorrectionsPayload buildPayload() const;

    void onCollectionRemoved(uint64_t id);
    void onCollectionChanged(uint64_t id);

private:
    void emitStatus(const QString& message, int timeoutMs);

    SegmentationModule& _module;
    SegmentationWidget* _widget{nullptr};
    VCCollection* _collection{nullptr};

    bool _annotateMode{false};
    uint64_t _activeCollectionId{0};
    std::vector<uint64_t> _pendingCollectionIds;
    std::unordered_set<uint64_t> _managedCollectionIds;
    bool _growthInProgress{false};
    bool _zRangeEnabled{false};
    int _zMin{0};
    int _zMax{0};
    std::optional<std::pair<int, int>> _zRange;
};

} // namespace segmentation

