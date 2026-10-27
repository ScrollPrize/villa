#pragma once

#include <QObject>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <unordered_map>

#ifdef __linux__
#include <QSocketNotifier>
#endif

class QTimer;
class CState;

// Watches the project's volumes/ directory and updates its volume catalog.
class FileWatcherService : public QObject
{
    Q_OBJECT

public:
    explicit FileWatcherService(CState* state, QObject* parent = nullptr);
    ~FileWatcherService();

    void startWatching();
    void stopWatching();

signals:
    void volumeCatalogChanged(const QString& preferredVolumeId);

#ifdef __linux__
private slots:
    void onInotifyEvent();
    void processPendingEvents();

private:
    void processVolumeAddition(const std::string& volumeName);
    void processVolumeRemoval(const std::string& volumeName);
    void processVolumeRename(const std::string& oldName, const std::string& newName);
    void scheduleProcessing();
#endif

private:
    CState* _state{nullptr};

#ifdef __linux__
    int _inotifyFd{-1};
    QSocketNotifier* _inotifyNotifier{nullptr};
    QTimer* _processTimer{nullptr};

    std::set<int> _watchDescriptors;
    // MOVED_FROM names by cookie, waiting for their MOVED_TO pair.
    std::map<uint32_t, std::string> _pendingMoves;
    std::unordered_map<std::string, int> _pendingVolumeAddAttempts;
    std::set<std::string> _pendingVolumeAddRetries;

    struct InotifyEvent {
        enum Type { Addition, Removal, Rename };
        Type type;
        std::string name;
        std::string newName;
    };

    std::vector<InotifyEvent> _pendingEvents;

    static constexpr int THROTTLE_MS = 100;
    static constexpr int VOLUME_ADD_MAX_ATTEMPTS = 2;
    static constexpr int VOLUME_ADD_RETRY_DELAY_MS = 700;
#endif
};
