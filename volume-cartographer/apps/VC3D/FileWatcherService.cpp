#include "FileWatcherService.hpp"

#ifdef __linux__

#include "CState.hpp"
#include "SurfacePanelController.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "SurfaceTreeWidget.hpp"
#include "vc/core/util/Logging.hpp"
#include "VCSettings.hpp"
#include <sys/inotify.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <QTimer>
#include <QSettings>
#include <QSignalBlocker>
#include <QTreeWidgetItemIterator>

FileWatcherService::FileWatcherService(CState* state, QObject* parent)
    : QObject(parent)
    , _state(state)
{
    _processTimer = new QTimer(this);
    connect(_processTimer, &QTimer::timeout, this, &FileWatcherService::processPendingEvents);
}

FileWatcherService::~FileWatcherService()
{
    stopWatching();
}

void FileWatcherService::startWatching()
{
    if (!_state->vpkg()) {
        return;
    }

    // Check if file watching is enabled in settings
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    if (!settings.value(vc3d::settings::perf::ENABLE_FILE_WATCHING, vc3d::settings::perf::ENABLE_FILE_WATCHING_DEFAULT).toBool()) {
        Logger()->info("File watching is disabled in settings");
        return;
    }

    // Stop any existing watches
    stopWatching();

    // Initialize inotify
    _inotifyFd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    if (_inotifyFd < 0) {
        Logger()->error("Failed to initialize inotify: {}", strerror(errno));
        return;
    }

    // Watch both paths and traces directories
    auto availableDirs = _state->vpkg()->getAvailableSegmentationDirectories();
    for (const auto& dirName : availableDirs) {
        std::filesystem::path dirPath = std::filesystem::path(_state->vpkg()->getVolpkgDirectory()) / dirName;

        if (!std::filesystem::exists(dirPath)) {
            Logger()->debug("Directory {} does not exist, skipping watch", dirPath.string());
            continue;
        }

        // Watch for directory create, delete, and move events
        int wd = inotify_add_watch(_inotifyFd, dirPath.c_str(),
                                  IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO | IN_ONLYDIR);

        if (wd < 0) {
            Logger()->error("Failed to add inotify watch for {}: {}", dirPath.string(), strerror(errno));
            continue;
        }

        _watchDescriptors[wd] = dirName;
        Logger()->info("Started inotify watch for {} directory (wd={})", dirName, wd);
    }

    // Set up Qt socket notifier to integrate with event loop
    _inotifyNotifier = new QSocketNotifier(_inotifyFd, QSocketNotifier::Read, this);
    connect(_inotifyNotifier, &QSocketNotifier::activated, this, &FileWatcherService::onInotifyEvent);
}

void FileWatcherService::stopWatching()
{
    if (_processTimer) {
        _processTimer->stop();
    }

    if (_inotifyNotifier) {
        delete _inotifyNotifier;
        _inotifyNotifier = nullptr;
    }

    if (_inotifyFd >= 0) {
        // Remove all watches
        for (const auto& [wd, dirName] : _watchDescriptors) {
            inotify_rm_watch(_inotifyFd, wd);
        }
        _watchDescriptors.clear();

        ::close(_inotifyFd);
        _inotifyFd = -1;
    }

    _pendingMoves.clear();
}

void FileWatcherService::onInotifyEvent()
{
    char buffer[4096] __attribute__((aligned(__alignof__(struct inotify_event))));
    ssize_t length = read(_inotifyFd, buffer, sizeof(buffer));

    if (length < 0) {
        if (errno != EAGAIN) {
            std::cerr << "Error reading inotify events: " << strerror(errno) << std::endl;
        }
        return;
    }

    ssize_t i = 0;
    while (i < length) {
        struct inotify_event* event = reinterpret_cast<struct inotify_event*>(&buffer[i]);

        // IN_Q_OVERFLOW events have len == 0, so check before the len > 0 block
        if (event->mask & IN_Q_OVERFLOW) {
            std::cerr << "Inotify queue overflow - some events may have been lost" << std::endl;
            // Could trigger a full reload here if needed
        }

        if (event->len > 0) {
            std::string fileName(event->name);

            // Skip hidden files and temporary files
            if (fileName.empty() || fileName[0] == '.' || fileName.find("~") != std::string::npos) {
                i += sizeof(struct inotify_event) + event->len;
                continue;
            }

            // Find the directory name for this watch descriptor
            auto it = _watchDescriptors.find(event->wd);
            if (it != _watchDescriptors.end()) {
                std::string dirName = it->second;

                // Handle different event types
                if (event->mask & IN_CREATE) {
                    // New segment created
                    InotifyEvent evt;
                    evt.type = InotifyEvent::Addition;
                    evt.dirName = dirName;
                    evt.segmentId = fileName;
                    _pendingEvents.push_back(evt);

                } else if (event->mask & IN_DELETE) {
                    // Segment deleted
                    InotifyEvent evt;
                    evt.type = InotifyEvent::Removal;
                    evt.dirName = dirName;
                    evt.segmentId = fileName;
                    _pendingEvents.push_back(evt);

                } else if (event->mask & IN_MOVED_FROM) {
                    // First part of move/rename - store with cookie
                    // Store both the filename and directory for orphaned move cleanup
                    _pendingMoves[event->cookie] = dirName + "/" + fileName;

                } else if (event->mask & IN_MOVED_TO) {
                    // Second part of move/rename
                    auto moveIt = _pendingMoves.find(event->cookie);
                    if (moveIt != _pendingMoves.end()) {
                        // This is a rename within watched directories
                        // Extract the old filename from the stored path
                        std::string oldPath = moveIt->second;
                        size_t lastSlash = oldPath.rfind('/');
                        std::string oldName = (lastSlash != std::string::npos) ?
                                               oldPath.substr(lastSlash + 1) : oldPath;
                        _pendingMoves.erase(moveIt);

                        InotifyEvent evt;
                        evt.type = InotifyEvent::Rename;
                        evt.dirName = dirName;
                        evt.segmentId = oldName;  // old segment ID
                        evt.newId = fileName;      // new segment ID
                        _pendingEvents.push_back(evt);

                    } else {
                        // File moved from outside watched directory - treat as new addition
                        InotifyEvent evt;
                        evt.type = InotifyEvent::Addition;
                        evt.dirName = dirName;
                        evt.segmentId = fileName;
                        _pendingEvents.push_back(evt);
                    }

                }
            }
        }

        i += sizeof(struct inotify_event) + event->len;
    }

    // Clean up old pending moves that never got their MOVED_TO pair
    if (!_pendingMoves.empty()) {
        static auto lastCleanup = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();

        // Clean up orphaned moves every 5 seconds
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastCleanup).count() > 5) {
            for (auto it = _pendingMoves.begin(); it != _pendingMoves.end(); ) {
                // Extract directory and filename from stored path
                std::string fullPath = it->second;
                size_t lastSlash = fullPath.rfind('/');
                if (lastSlash != std::string::npos) {
                    std::string dirName = fullPath.substr(0, lastSlash);
                    std::string fileName = fullPath.substr(lastSlash + 1);

                    // Treat orphaned MOVED_FROM as deletions
                    InotifyEvent evt;
                    evt.type = InotifyEvent::Removal;
                    evt.dirName = dirName;
                    evt.segmentId = fileName;
                    _pendingEvents.push_back(evt);
                }

                it = _pendingMoves.erase(it);
            }
            lastCleanup = now;
        }
    }

    scheduleProcessing();
}

void FileWatcherService::processSegmentUpdate(const std::string& dirName, const std::string& segmentName)
{
    if (!_state->vpkg()) return;

    Logger()->info("Processing update of {} in {}", segmentName, dirName);

    bool isCurrentDir = (dirName == _state->vpkg()->getSegmentationDirectory());
    if (!isCurrentDir) {
        Logger()->debug("Update in non-current directory {}, skipping UI update", dirName);
        return;
    }

    std::string segmentId = segmentName; // UUID = directory name

    // Check if the segment exists
    auto seg = _state->vpkg()->segmentation(segmentId);
    if (!seg) {
        Logger()->warn("Segment {} not found for update, treating as addition", segmentId);
        processSegmentAddition(dirName, segmentName);
        return;
    }

    // Skip reloads triggered right after editing sessions end
    if (shouldSkipForSegment(segmentId, "reload")) {
        return;
    }

    bool wasSelected = (_state->activeSurfaceId() == segmentId);

    // Reload the segmentation
    if (_state->vpkg()->reloadSingleSegmentation(segmentId)) {
        try {
            auto reloadedSurf = _state->vpkg()->loadSurface(segmentId);
            if (reloadedSurf) {
                if (_state) {
                    _state->setSurface(segmentId, reloadedSurf, false, false);
                }

                if (_surfacePanel) {
                    _surfacePanel->refreshSurfaceMetrics(segmentId);
                }

                emit statusMessage(tr("Updated: %1").arg(QString::fromStdString(segmentName)), 2000);

                if (wasSelected) {
                    _state->setActiveSurface(segmentId, reloadedSurf);

                    if (_state) {
                        _state->setSurface("segmentation", reloadedSurf, false, false);
                    }

                    if (_surfacePanel) {
                        _surfacePanel->syncSelectionUi(segmentId, reloadedSurf.get());
                    }
                }
            }
        } catch (const std::exception& e) {
            Logger()->error("Failed to reload segment {}: {}", segmentId, e.what());
        }
    }
}

void FileWatcherService::processSegmentRename(const std::string& dirName,
                                              const std::string& oldDirName,
                                              const std::string& newDirName)
{
    if (!_state->vpkg()) return;

    Logger()->info("Processing rename in {}: {} -> {}", dirName, oldDirName, newDirName);

    bool isCurrentDir = (dirName == _state->vpkg()->getSegmentationDirectory());

    // The old UUID would have been the old directory name
    std::string oldId = oldDirName;
    std::string newId = newDirName;

    // Check if the old segment exists
    if (!_state->vpkg()->segmentation(oldId)) {
        Logger()->warn("Old segment {} not found, treating as new addition", oldId);
        processSegmentAddition(dirName, newDirName);
        return;
    }

    // Remove the old entry
    bool wasSelected = isCurrentDir && (_state->activeSurfaceId() == oldId);
    _state->vpkg()->removeSingleSegmentation(oldId);

    if (isCurrentDir && _surfacePanel) {
        _surfacePanel->removeSingleSegmentation(oldId);
    }

    // Add with new name (which will read the meta.json and update the UUID)
    std::string previousDir;
    if (!isCurrentDir) {
        previousDir = _state->vpkg()->getSegmentationDirectory();
        _state->vpkg()->setSegmentationDirectory(dirName);
    }

    if (_state->vpkg()->addSingleSegmentation(newDirName)) {
        // The UUID in meta.json will be updated when the segment is saved/loaded
        try {
            auto loadedSurf = _state->vpkg()->loadSurface(newId);

            if (loadedSurf && isCurrentDir) {
                _state->setSurface(newId, loadedSurf, true);
                if (_surfacePanel) {
                    _surfacePanel->addSingleSegmentation(newId);
                }

                emit statusMessage(tr("Renamed: %1 → %2")
                                       .arg(QString::fromStdString(oldDirName),
                                            QString::fromStdString(newDirName)), 3000);

                // Reselect if it was selected
                if (wasSelected) {
                    auto surf = _state->surface(newId);
                    _state->setActiveSurface(newId, std::dynamic_pointer_cast<QuadSurface>(surf));
                    if (surf && _surfacePanel) {
                        _surfacePanel->syncSelectionUi(newId, dynamic_cast<QuadSurface*>(surf.get()));
                    }
                }

            }
        } catch (const std::exception& e) {
            Logger()->error("Failed to load renamed segment {}: {}", newId, e.what());
        }
    }

    if (!isCurrentDir && !previousDir.empty()) {
        _state->vpkg()->setSegmentationDirectory(previousDir);
    }
}

void FileWatcherService::processSegmentAddition(const std::string& dirName, const std::string& segmentName)
{
    if (!_state->vpkg()) return;

    Logger()->info("Processing addition of {} to {}", segmentName, dirName);

    bool isCurrentDir = (dirName == _state->vpkg()->getSegmentationDirectory());

    // The UUID will be the directory name (or will be updated to match)
    std::string segmentId = segmentName;

    // Skip addition if editing just stopped recently to avoid thrashing the active surface
    if (shouldSkipForSegment(segmentId, "addition")) {
        return;
    }

    // Switch directory if needed
    std::string previousDir;
    if (!isCurrentDir) {
        previousDir = _state->vpkg()->getSegmentationDirectory();
        _state->vpkg()->setSegmentationDirectory(dirName);
    }

    // Add the segment
    if (_state->vpkg()->addSingleSegmentation(segmentName)) {
        if (isCurrentDir) {
            try {
                auto loadedSurf = _state->vpkg()->loadSurface(segmentId);
                if (loadedSurf) {
                    _state->setSurface(segmentId, loadedSurf, true);
                    if (_surfacePanel) {
                        _surfacePanel->addSingleSegmentation(segmentId);
                    }
                    emit statusMessage(tr("Added: %1").arg(QString::fromStdString(segmentName)), 2000);
                }
            } catch (const std::exception& e) {
                Logger()->error("Failed to load segment {}: {}", segmentId, e.what());
            }
        }
    }

    if (!isCurrentDir && !previousDir.empty()) {
        _state->vpkg()->setSegmentationDirectory(previousDir);
    }
}

void FileWatcherService::processSegmentRemoval(const std::string& dirName, const std::string& segmentName)
{
    if (!_state->vpkg()) return;

    std::string segmentId = segmentName;

    Logger()->info("Processing removal of {} from {}", segmentId, dirName);

    // First check if this segment even exists and belongs to this directory
    auto seg = _state->vpkg()->segmentation(segmentId);
    if (!seg) {
        Logger()->debug("Segment {} not found, ignoring removal event from {}", segmentId, dirName);
        return;
    }

    // Verify the segment is actually in the directory that reported the removal
    if (seg->path().parent_path().filename() != dirName) {
        Logger()->warn("Removal event for {} from {}, but segment is actually in {}",
                      segmentId, dirName, seg->path().parent_path().filename().string());
        return;
    }

    bool isCurrentDir = (dirName == _state->vpkg()->getSegmentationDirectory());

    // Skip removal if editing just ended or is still running
    if (shouldSkipForSegment(segmentId, "removal")) {
        return;
    }

    // Remove from VolumePkg
    if (_state->vpkg()->removeSingleSegmentation(segmentId)) {
        if (isCurrentDir && _surfacePanel) {
            _surfacePanel->removeSingleSegmentation(segmentId);
            emit statusMessage(tr("Removed: %1").arg(QString::fromStdString(segmentName)), 2000);
        }
    }
}

void FileWatcherService::processPendingEvents()
{
    if (_pendingEvents.empty() && _pendingSegmentUpdates.empty()) {
        return;
    }

    // Store current selection to restore later
    std::string previousSelection = _state->activeSurfaceId();
    auto previousSurface = _state->activeSurface().lock();

    // Track if the previously selected segment gets removed
    bool previousSelectionRemoved = false;

    // Sort events to process removals first, then renames, then additions
    std::vector<InotifyEvent> removals, renames, additions, updates;
    for (const auto& evt : _pendingEvents) {
        switch (evt.type) {
            case InotifyEvent::Removal:
                removals.push_back(evt);
                break;
            case InotifyEvent::Rename:
                renames.push_back(evt);
                break;
            case InotifyEvent::Addition:
                additions.push_back(evt);
                break;
            case InotifyEvent::Update:
                updates.push_back(evt);
                break;
        }
    }

    if (!removals.empty() && !additions.empty()) {
        using EventKey = std::pair<std::string, std::string>;
        auto makeKey = [](const InotifyEvent& evt) -> EventKey {
            return {evt.dirName, evt.segmentId};
        };

        std::map<EventKey, int> additionCounts;
        for (const auto& addition : additions) {
            additionCounts[makeKey(addition)]++;
        }

        std::map<EventKey, int> pairedCounts;
        std::vector<InotifyEvent> filteredRemovals;
        filteredRemovals.reserve(removals.size());

        for (const auto& removal : removals) {
            EventKey key = makeKey(removal);
            auto availableIt = additionCounts.find(key);
            const int available = (availableIt != additionCounts.end()) ? availableIt->second : 0;
            auto existingPairIt = pairedCounts.find(key);
            const int alreadyPaired = (existingPairIt != pairedCounts.end()) ? existingPairIt->second : 0;

            if (available > alreadyPaired) {
                pairedCounts[key] = alreadyPaired + 1;

                InotifyEvent updateEvt;
                updateEvt.type = InotifyEvent::Update;
                updateEvt.dirName = removal.dirName;
                updateEvt.segmentId = removal.segmentId;
                updates.push_back(updateEvt);
            } else {
                filteredRemovals.push_back(removal);
            }
        }

        std::vector<InotifyEvent> filteredAdditions;
        filteredAdditions.reserve(additions.size());
        for (const auto& addition : additions) {
            EventKey key = makeKey(addition);
            auto pairIt = pairedCounts.find(key);
            if (pairIt != pairedCounts.end() && pairIt->second > 0) {
                pairIt->second--;
                continue;
            }
            filteredAdditions.push_back(addition);
        }

        removals.swap(filteredRemovals);
        additions.swap(filteredAdditions);
    }

    for (const auto& evt : removals) {
        if (evt.segmentId == previousSelection) {
            previousSelectionRemoved = true;
            break;
        }
    }

    // Process in order: removals, renames, additions, updates
    for (const auto& evt : removals) {
        processSegmentRemoval(evt.dirName, evt.segmentId);
    }

    for (const auto& evt : renames) {
        processSegmentRename(evt.dirName, evt.segmentId, evt.newId);
    }

    for (const auto& evt : additions) {
        processSegmentAddition(evt.dirName, evt.segmentId);
    }

    for (const auto& evt : updates) {
        processSegmentUpdate(evt.dirName, evt.segmentId);
    }

    // Process unique segment updates
    for (const auto& [dirName, segmentId] : _pendingSegmentUpdates) {
        processSegmentUpdate(dirName, segmentId);
    }

    // Clear the queues
    _pendingEvents.clear();
    _pendingSegmentUpdates.clear();

    // Restore selection if it still exists (might have been renamed or re-added)
    if (!previousSelection.empty() && previousSelectionRemoved) {
        // If editing is active for this segment, skip re-emitting the segmentation surface change
        // because autosave-triggered inotify events were intentionally skipped.
        bool skipRestoreForActiveEdit = false;
        if (_segmentationModule && _segmentationModule->editingEnabled()) {
            auto* activeBaseSurface = _segmentationModule->activeBaseSurface();
            if (activeBaseSurface && activeBaseSurface->id == previousSelection) {
                Logger()->info("Skipping selection restore of {} - currently being edited", previousSelection);
                skipRestoreForActiveEdit = true;
            }
        }

        if (!skipRestoreForActiveEdit) {
            // Check if the segment was re-added in this batch
            bool wasReAdded = false;
            for (const auto& evt : additions) {
                if (evt.segmentId == previousSelection) {
                    wasReAdded = true;
                    break;
                }
            }

            if (wasReAdded) {
                // The segment was removed and re-added - restore selection
                auto seg = _state->vpkg() ? _state->vpkg()->segmentation(previousSelection) : nullptr;
                if (seg) {
                    auto surf = _state->vpkg()->getSurface(previousSelection);
                    _state->setActiveSurface(previousSelection, surf);
                    if (surf) {

                        if (_state) {
                            _state->setSurface("segmentation", surf, false, false);
                        }

                        if (_treeWidget) {
                            QTreeWidgetItemIterator it(_treeWidget);
                            while (*it) {
                                if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() == previousSelection) {
                                    const QSignalBlocker blocker{_treeWidget};
                                    _treeWidget->clearSelection();
                                    (*it)->setSelected(true);
                                    _treeWidget->scrollToItem(*it);
                                    break;
                                }
                                ++it;
                            }
                        }

                        if (_surfacePanel) {
                            _surfacePanel->syncSelectionUi(previousSelection, surf.get());
                        }

                        // Window title update skipped - handled by caller
                    }
                }
            }
        }
    } else if (!previousSelection.empty()) {
        // Original logic for non-removed segments
        auto seg = _state->vpkg() ? _state->vpkg()->segmentation(previousSelection) : nullptr;
        if (seg) {
            _state->setActiveSurface(previousSelection, std::dynamic_pointer_cast<QuadSurface>(previousSurface));

            if (_surfacePanel) {
                auto surface = _state->surface(previousSelection);
                if (surface) {
                    _surfacePanel->syncSelectionUi(previousSelection, dynamic_cast<QuadSurface*>(surface.get()));
                }
            }
        }
    }

    // Refresh filters once at the end instead of multiple times
    if (_surfacePanel) {
        _surfacePanel->refreshFiltersOnly();
    }
}

void FileWatcherService::scheduleProcessing()
{
    if (!_processTimer) {
        return;
    }

    // Stop any existing timer
    _processTimer->stop();

    // Use single-shot timer with short delay
    _processTimer->setSingleShot(true);
    _processTimer->setInterval(THROTTLE_MS);
    _processTimer->start();
}

bool FileWatcherService::shouldSkipForSegment(const std::string& segmentId, const char* eventCategory)
{
    if (segmentId.empty()) {
        return false;
    }

    const char* category = eventCategory ? eventCategory : "inotify event";

    if (_segmentationModule && _segmentationModule->editingEnabled()) {
        if (auto* activeBaseSurface = _segmentationModule->activeBaseSurface()) {
            if (activeBaseSurface->id == segmentId) {
                Logger()->info("Skipping {} for {} - currently being edited", category, segmentId);
                return true;
            }
        }
    }

    // Also skip if approval mask editing is active for this segment
    if (_segmentationModule && _segmentationModule->isEditingApprovalMask()) {
        // Get the segment being used for approval mask (the active segmentation surface)
        if (_state) {
            auto segSurface = _state->surface("segmentation");
            if (segSurface && segSurface->id == segmentId) {
                Logger()->info("Skipping {} for {} - approval mask being edited", category, segmentId);
                return true;
            }
        }
    }

    pruneExpiredRecentlyEdited();

    if (_recentlyEditedSegments.empty()) {
        return false;
    }

    auto it = _recentlyEditedSegments.find(segmentId);
    if (it == _recentlyEditedSegments.end()) {
        return false;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - it->second;
    const auto grace = std::chrono::seconds(RECENT_EDIT_GRACE_SECONDS);

    if (elapsed < grace) {
        const double elapsedSeconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() / 1000.0;
        const double remainingSeconds = std::max(
            0.0, static_cast<double>(RECENT_EDIT_GRACE_SECONDS) - elapsedSeconds);
        Logger()->info("Skipping {} for {} - edits ended {:.1f}s ago ({}s grace remaining)",
                       category,
                       segmentId,
                       elapsedSeconds,
                       static_cast<int>(std::ceil(remainingSeconds)));
        return true;
    }

    _recentlyEditedSegments.erase(it);
    return false;
}

void FileWatcherService::markSegmentRecentlyEdited(const std::string& segmentId)
{
    if (segmentId.empty()) {
        return;
    }

    pruneExpiredRecentlyEdited();
    _recentlyEditedSegments[segmentId] = std::chrono::steady_clock::now();
}

void FileWatcherService::pruneExpiredRecentlyEdited()
{
    if (_recentlyEditedSegments.empty()) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto grace = std::chrono::seconds(RECENT_EDIT_GRACE_SECONDS);

    for (auto it = _recentlyEditedSegments.begin(); it != _recentlyEditedSegments.end(); ) {
        if (now - it->second >= grace) {
            it = _recentlyEditedSegments.erase(it);
        } else {
            ++it;
        }
    }
}

#else // !__linux__

// Stub implementations for non-Linux platforms (no inotify)
FileWatcherService::FileWatcherService(CState* state, QObject* parent)
    : QObject(parent), _state(state) {}
FileWatcherService::~FileWatcherService() = default;
void FileWatcherService::startWatching() {}
void FileWatcherService::stopWatching() {}
void FileWatcherService::markSegmentRecentlyEdited(const std::string&) {}

#endif // __linux__
