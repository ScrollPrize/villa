#pragma once

#include <QObject>
#include <QElapsedTimer>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <functional>

#ifdef __linux__
#include <QSocketNotifier>
#endif

class QTimer;
class CState;
class SurfacePanelController;
class SegmentationModule;
class SurfaceTreeWidget;

class FileWatcherService : public QObject
{
    Q_OBJECT

public:
    explicit FileWatcherService(CState* state, QObject* parent = nullptr);
    ~FileWatcherService();

    void setSurfacePanel(SurfacePanelController* panel) { _surfacePanel = panel; }
    void setSegmentationModule(SegmentationModule* mod) { _segmentationModule = mod; }
    void setTreeWidget(SurfaceTreeWidget* tree) { _treeWidget = tree; }

    void startWatching();
    void stopWatching();

    void markSegmentRecentlyEdited(const std::string& segmentId);

signals:
    void statusMessage(const QString& text, int timeout);
    void volumeCatalogChanged(const QString& preferredVolumeId);

#ifdef __linux__
private slots:
    void onInotifyEvent();
    void processPendingEvents();

private:
    void processSegmentUpdate(const std::string& dirName, const std::string& segmentName);
    void processSegmentRename(const std::string& dirName,
                              const std::string& oldDirName,
                              const std::string& newDirName);
    void processSegmentAddition(const std::string& dirName, const std::string& segmentName);
    void processSegmentRemoval(const std::string& dirName, const std::string& segmentName);
    void processVolumeAddition(const std::string& volumeName);
    void processVolumeRemoval(const std::string& volumeName);
    void processVolumeRename(const std::string& oldName, const std::string& newName);
    void scheduleProcessing();
    bool shouldSkipForSegment(const std::string& segmentId, const char* eventCategory);
    void pruneExpiredRecentlyEdited();
#endif

private:
    CState* _state{nullptr};
    SurfacePanelController* _surfacePanel{nullptr};
    SegmentationModule* _segmentationModule{nullptr};
    SurfaceTreeWidget* _treeWidget{nullptr};

#ifdef __linux__
    int _inotifyFd{-1};
    QSocketNotifier* _inotifyNotifier{nullptr};
    QTimer* _processTimer{nullptr};

    struct WatchDescriptorInfo {
        std::string dirName;
        bool isVolumeWatch{false};
    };

    struct PendingMoveInfo {
        std::string dirName;
        std::string movedName;
        bool isVolumeWatch{false};
    };

    std::map<int, WatchDescriptorInfo> _watchDescriptors;
    std::map<uint32_t, PendingMoveInfo> _pendingMoves;
    std::unordered_map<std::string, int> _pendingVolumeAddAttempts;
    std::set<std::string> _pendingVolumeAddRetries;

    struct InotifyEvent {
        enum Type { Addition, Removal, Rename, Update };
        enum Domain { Segment, Volume };
        Type type;
        Domain resourceDomain;
        std::string dirName;
        std::string segmentId;
        std::string newId;
    };

    std::vector<InotifyEvent> _pendingEvents;
    std::set<std::pair<std::string, std::string>> _pendingSegmentUpdates;

    QElapsedTimer _lastProcessTime;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> _recentlyEditedSegments;

    static constexpr int THROTTLE_MS = 100;
    static constexpr int RECENT_EDIT_GRACE_SECONDS = 30;
    static constexpr int VOLUME_ADD_MAX_ATTEMPTS = 2;
    static constexpr int VOLUME_ADD_RETRY_DELAY_MS = 700;
#endif
};
