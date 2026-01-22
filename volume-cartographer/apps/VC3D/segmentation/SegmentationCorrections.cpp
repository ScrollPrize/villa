#include "SegmentationCorrections.hpp"

#include "SegmentationModule.hpp"
#include "SegmentationWidget.hpp"

#include "vc/ui/VCCollection.hpp"

#include <QCoreApplication>

#include <algorithm>
#include <limits>

namespace segmentation
{
CorrectionsState::CorrectionsState(SegmentationModule& module,
                                   SegmentationWidget* widget,
                                   VCCollection* collection)
    : _module(module)
    , _widget(widget)
    , _collection(collection)
{
    if (_collection) {
        const auto& collections = _collection->getAllCollections();
        _pendingCollectionIds.reserve(collections.size());
        for (const auto& entry : collections) {
            _pendingCollectionIds.push_back(entry.first);
        }
    }
    refreshWidget();
}

void CorrectionsState::setWidget(SegmentationWidget* widget)
{
    _widget = widget;
    refreshWidget();
}

void CorrectionsState::setCollection(VCCollection* collection)
{
    _collection = collection;
    _pendingCollectionIds.clear();
    _managedCollectionIds.clear();
    _activeCollectionId = 0;

    if (_collection) {
        const auto& collections = _collection->getAllCollections();
        _pendingCollectionIds.reserve(collections.size());
        for (const auto& entry : collections) {
            _pendingCollectionIds.push_back(entry.first);
        }
    }

    refreshWidget();
}

bool CorrectionsState::setAnnotateMode(bool enabled, bool userInitiated, bool editingEnabled)
{
    if (!_collection || _growthInProgress || !editingEnabled) {
        enabled = false;
    }

    if (enabled && _activeCollectionId == 0) {
        if (createCollection(false) == 0) {
            enabled = false;
        }
    }

    if (_annotateMode == enabled) {
        refreshWidget();
        return _annotateMode;
    }

    _annotateMode = enabled;

    if (_widget) {
        _widget->setCorrectionsAnnotateChecked(enabled);
    }

    if (userInitiated) {
        const QString message = enabled ? QObject::tr("Correction annotation enabled")
                                        : QObject::tr("Correction annotation disabled");
        emitStatus(message, kStatusShort);
    }

    refreshWidget();
    return _annotateMode;
}

void CorrectionsState::setActiveCollection(uint64_t collectionId, bool userInitiated)
{
    if (!_collection) {
        return;
    }

    if (collectionId == 0) {
        _activeCollectionId = 0;
        setAnnotateMode(false, false, _module.editingEnabled());
        refreshWidget();
        return;
    }

    const auto& collections = _collection->getAllCollections();
    if (collections.find(collectionId) == collections.end()) {
        pruneMissing();
        emitStatus(QObject::tr("Selected correction set no longer exists."), kStatusShort);
        refreshWidget();
        return;
    }

    if (std::find(_pendingCollectionIds.begin(), _pendingCollectionIds.end(), collectionId) ==
        _pendingCollectionIds.end()) {
        _pendingCollectionIds.push_back(collectionId);
    }

    _activeCollectionId = collectionId;

    if (userInitiated) {
        emitStatus(QObject::tr("Active correction set changed."), kStatusShort);
    }

    refreshWidget();
}

uint64_t CorrectionsState::createCollection(bool announce)
{
    if (!_collection) {
        return 0;
    }

    const std::string newName = _collection->generateNewCollectionName("correction");
    const uint64_t newId = _collection->addCollection(newName);
    if (newId == 0) {
        return 0;
    }

    if (std::find(_pendingCollectionIds.begin(), _pendingCollectionIds.end(), newId) ==
        _pendingCollectionIds.end()) {
        _pendingCollectionIds.push_back(newId);
    }

    _managedCollectionIds.insert(newId);
    _activeCollectionId = newId;

    if (announce) {
        emitStatus(QObject::tr("Created correction set '%1'.").arg(QString::fromStdString(newName)), kStatusShort);
    }

    refreshWidget();
    return newId;
}

void CorrectionsState::handlePointAdded(const cv::Vec3f& worldPos)
{
    if (!_collection || _activeCollectionId == 0) {
        return;
    }

    const auto& collections = _collection->getAllCollections();
    auto it = collections.find(_activeCollectionId);
    if (it == collections.end()) {
        pruneMissing();
        refreshWidget();
        return;
    }

    _collection->addPoint(it->second.name, worldPos);
}

void CorrectionsState::handlePointRemoved(const cv::Vec3f& worldPos)
{
    if (!_collection || _activeCollectionId == 0) {
        return;
    }

    const auto& collections = _collection->getAllCollections();
    auto it = collections.find(_activeCollectionId);
    if (it == collections.end()) {
        pruneMissing();
        refreshWidget();
        return;
    }

    const auto& points = it->second.points;
    if (points.empty()) {
        return;
    }

    uint64_t closestId = 0;
    float closestDistance = std::numeric_limits<float>::max();
    for (const auto& entry : points) {
        const float dist = cv::norm(entry.second.p - worldPos);
        if (dist < closestDistance) {
            closestDistance = dist;
            closestId = entry.second.id;
        }
    }

    if (closestId != 0) {
        _collection->removePoint(closestId);
    }
}

void CorrectionsState::onZRangeChanged(bool enabled, int zMin, int zMax)
{
    _zRangeEnabled = enabled;
    if (zMin > zMax) {
        std::swap(zMin, zMax);
    }
    _zMin = zMin;
    _zMax = zMax;
    if (enabled) {
        _zRange = std::make_pair(_zMin, _zMax);
    } else {
        _zRange.reset();
    }
}

bool CorrectionsState::hasCorrections() const
{
    if (!_collection) {
        return false;
    }

    const auto& collections = _collection->getAllCollections();
    for (const auto& entry : collections) {
        if (!entry.second.points.empty()) {
            return true;
        }
    }
    return false;
}

void CorrectionsState::setGrowthInProgress(bool running)
{
    _growthInProgress = running;
    if (_widget) {
        _widget->setCorrectionsEnabled(!_growthInProgress && _collection != nullptr);
    }
    if (_growthInProgress) {
        setAnnotateMode(false, false, _module.editingEnabled());
    }
    refreshWidget();
}

void CorrectionsState::clearAll(bool editingEnabled)
{
    setAnnotateMode(false, false, editingEnabled);

    if (_collection) {
        for (uint64_t id : _pendingCollectionIds) {
            if (_managedCollectionIds.count(id) > 0) {
                _collection->clearCollection(id);
            }
        }
    }

    _pendingCollectionIds.clear();
    _managedCollectionIds.clear();
    _activeCollectionId = 0;

    // Repopulate _pendingCollectionIds with surviving persistent collections
    // (those with anchor2d set that were not cleared above)
    if (_collection) {
        const auto& collections = _collection->getAllCollections();
        for (const auto& entry : collections) {
            if (entry.second.anchor2d.has_value()) {
                _pendingCollectionIds.push_back(entry.first);
            }
        }
    }

    refreshWidget();
}

void CorrectionsState::refreshWidget()
{
    if (!_widget) {
        return;
    }

    pruneMissing();

    const bool correctionsAvailable = (_collection != nullptr) && !_growthInProgress;
    QVector<QPair<uint64_t, QString>> entries;
    if (_collection) {
        const auto& collections = _collection->getAllCollections();
        entries.reserve(static_cast<int>(_pendingCollectionIds.size()));
        for (uint64_t id : _pendingCollectionIds) {
            auto it = collections.find(id);
            if (it != collections.end()) {
                entries.append({id, QString::fromStdString(it->second.name)});
            }
        }
    }

    std::optional<uint64_t> active;
    if (_activeCollectionId != 0) {
        active = _activeCollectionId;
    }

    _widget->setCorrectionCollections(entries, active);
    _widget->setCorrectionsEnabled(correctionsAvailable);
    _widget->setCorrectionsAnnotateChecked(_annotateMode && correctionsAvailable);
}

void CorrectionsState::pruneMissing()
{
    if (!_collection) {
        _pendingCollectionIds.clear();
        _managedCollectionIds.clear();
        _activeCollectionId = 0;
        return;
    }

    const auto& collections = _collection->getAllCollections();

    auto pendingErase = std::remove_if(_pendingCollectionIds.begin(),
                                       _pendingCollectionIds.end(),
                                       [&collections](uint64_t id) {
                                           return collections.find(id) == collections.end();
                                       });
    if (pendingErase != _pendingCollectionIds.end()) {
        _pendingCollectionIds.erase(pendingErase, _pendingCollectionIds.end());
    }

    for (auto it = _managedCollectionIds.begin(); it != _managedCollectionIds.end();) {
        if (collections.find(*it) == collections.end()) {
            it = _managedCollectionIds.erase(it);
        } else {
            ++it;
        }
    }

    if (_activeCollectionId != 0 && collections.find(_activeCollectionId) == collections.end()) {
        _activeCollectionId = 0;
    }
}

std::optional<std::pair<int, int>> CorrectionsState::zRange() const
{
    if (!_zRangeEnabled) {
        return std::nullopt;
    }
    return _zRange;
}

SegmentationCorrectionsPayload CorrectionsState::buildPayload(bool onlyActiveCollection) const
{
    SegmentationCorrectionsPayload payload;
    if (!_collection) {
        return payload;
    }

    const auto& collections = _collection->getAllCollections();

    // Determine which collection IDs to include
    std::vector<uint64_t> idsToInclude;
    if (onlyActiveCollection) {
        // Only include the active collection (if valid)
        if (_activeCollectionId != 0 && collections.find(_activeCollectionId) != collections.end()) {
            idsToInclude.push_back(_activeCollectionId);
        }
    } else {
        // Include all pending collections
        idsToInclude = _pendingCollectionIds;
    }

    for (uint64_t id : idsToInclude) {
        auto it = collections.find(id);
        if (it == collections.end()) {
            continue;
        }

        SegmentationCorrectionsPayload::Collection entry;
        entry.id = it->second.id;
        entry.name = it->second.name;
        entry.metadata = it->second.metadata;
        entry.color = it->second.color;
        entry.anchor2d = it->second.anchor2d;

        std::vector<ColPoint> points;
        points.reserve(it->second.points.size());
        for (const auto& pair : it->second.points) {
            points.push_back(pair.second);
        }
        std::sort(points.begin(), points.end(), [](const ColPoint& a, const ColPoint& b) {
            return a.id < b.id;
        });
        if (points.empty()) {
            continue;
        }

        entry.points = std::move(points);
        payload.collections.push_back(std::move(entry));
    }

    return payload;
}

SegmentationCorrectionsPayload CorrectionsState::buildPayloadForCollection(uint64_t collectionId) const
{
    SegmentationCorrectionsPayload payload;
    if (!_collection || collectionId == 0) {
        return payload;
    }

    const auto& collections = _collection->getAllCollections();
    auto it = collections.find(collectionId);
    if (it == collections.end()) {
        return payload;
    }

    SegmentationCorrectionsPayload::Collection entry;
    entry.id = it->second.id;
    entry.name = it->second.name;
    entry.metadata = it->second.metadata;
    entry.color = it->second.color;
    entry.anchor2d = it->second.anchor2d;

    std::vector<ColPoint> points;
    points.reserve(it->second.points.size());
    for (const auto& pair : it->second.points) {
        points.push_back(pair.second);
    }
    std::sort(points.begin(), points.end(), [](const ColPoint& a, const ColPoint& b) {
        return a.id < b.id;
    });
    if (points.empty()) {
        return payload;
    }

    entry.points = std::move(points);
    payload.collections.push_back(std::move(entry));

    return payload;
}

void CorrectionsState::onCollectionRemoved(uint64_t id)
{
    if (!_collection) {
        return;
    }

    const uint64_t sentinel = std::numeric_limits<uint64_t>::max();
    if (id == sentinel) {
        _pendingCollectionIds.clear();
        _managedCollectionIds.clear();
        _activeCollectionId = 0;
        setAnnotateMode(false, false, _module.editingEnabled());
        refreshWidget();
        return;
    }

    auto eraseIt = std::remove(_pendingCollectionIds.begin(), _pendingCollectionIds.end(), id);
    if (eraseIt != _pendingCollectionIds.end()) {
        _pendingCollectionIds.erase(eraseIt, _pendingCollectionIds.end());
    }

    _managedCollectionIds.erase(id);

    if (_activeCollectionId == id) {
        _activeCollectionId = 0;
        setAnnotateMode(false, false, _module.editingEnabled());
    }

    refreshWidget();
}

void CorrectionsState::onCollectionChanged(uint64_t id)
{
    const uint64_t sentinel = std::numeric_limits<uint64_t>::max();
    if (id == sentinel) {
        return;
    }

    if (std::find(_pendingCollectionIds.begin(), _pendingCollectionIds.end(), id) !=
        _pendingCollectionIds.end()) {
        refreshWidget();
    }
}

void CorrectionsState::emitStatus(const QString& message, int timeoutMs)
{
    emit _module.statusMessageRequested(message, timeoutMs);
}

} // namespace segmentation
