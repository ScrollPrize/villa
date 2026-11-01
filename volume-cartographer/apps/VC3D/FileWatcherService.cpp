#include "FileWatcherService.hpp"

#ifdef __linux__

#include "CState.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Logging.hpp"
#include "VCSettings.hpp"
#include <sys/inotify.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <QTimer>
#include <QSettings>

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

    // Watch the volumes directory for live updates
    {
        std::filesystem::path volumesDir = std::filesystem::path(_state->vpkg()->directory().string()) / "volumes";
        if (std::filesystem::exists(volumesDir)) {
            int wd = inotify_add_watch(_inotifyFd, volumesDir.c_str(),
                                      IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO | IN_ONLYDIR);
            if (wd < 0) {
                Logger()->error("Failed to add inotify watch for volumes directory: {}", strerror(errno));
            } else {
                _watchDescriptors.insert(wd);
                Logger()->info("Started inotify watch for volumes directory (wd={})", wd);
            }
        } else {
            Logger()->warn("Volumes directory does not exist at {} - skipping watch", volumesDir.string());
        }
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
        for (const int wd : _watchDescriptors) {
            inotify_rm_watch(_inotifyFd, wd);
        }
        _watchDescriptors.clear();

        ::close(_inotifyFd);
        _inotifyFd = -1;
    }

    _pendingMoves.clear();
    _pendingVolumeAddAttempts.clear();
    _pendingVolumeAddRetries.clear();
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

            if (_watchDescriptors.contains(event->wd)) {
                if (event->mask & IN_CREATE) {
                    _pendingEvents.push_back({InotifyEvent::Addition, fileName, {}});
                } else if (event->mask & IN_DELETE) {
                    _pendingEvents.push_back({InotifyEvent::Removal, fileName, {}});
                } else if (event->mask & IN_MOVED_FROM) {
                    // First part of a rename: wait for its MOVED_TO pair
                    _pendingMoves[event->cookie] = fileName;
                } else if (event->mask & IN_MOVED_TO) {
                    auto moveIt = _pendingMoves.find(event->cookie);
                    if (moveIt != _pendingMoves.end()) {
                        _pendingEvents.push_back({InotifyEvent::Rename, moveIt->second, fileName});
                        _pendingMoves.erase(moveIt);
                    } else {
                        // Moved in from outside the watched directory
                        _pendingEvents.push_back({InotifyEvent::Addition, fileName, {}});
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
            // Treat orphaned MOVED_FROM as deletions
            for (const auto& [cookie, name] : _pendingMoves) {
                _pendingEvents.push_back({InotifyEvent::Removal, name, {}});
            }
            _pendingMoves.clear();
            lastCleanup = now;
        }
    }

    scheduleProcessing();
}

void FileWatcherService::processVolumeAddition(const std::string& volumeName)
{
    if (!_state->vpkg()) return;

    Logger()->info("Processing volume addition of {}", volumeName);
    if (_state->vpkg()->addSingleVolume(volumeName)) {
        _pendingVolumeAddAttempts.erase(volumeName);
        _pendingVolumeAddRetries.erase(volumeName);
        emit volumeCatalogChanged(QString::fromStdString(volumeName));
        return;
    }

    if (_pendingVolumeAddRetries.count(volumeName) > 0) {
        return;
    }

    const int currentAttempts = _pendingVolumeAddAttempts[volumeName] + 1;
    _pendingVolumeAddAttempts[volumeName] = currentAttempts;

    if (currentAttempts >= VOLUME_ADD_MAX_ATTEMPTS) {
        Logger()->warn("Giving up on volume '{}' after {} attempts", volumeName, currentAttempts);
        _pendingVolumeAddAttempts.erase(volumeName);
        _pendingVolumeAddRetries.erase(volumeName);
        return;
    }

    const QString volumeId = QString::fromStdString(volumeName);
    Logger()->warn("Volume '{}' is not yet valid. Retrying {}/{} in {}ms",
                   volumeName, currentAttempts + 1, VOLUME_ADD_MAX_ATTEMPTS, VOLUME_ADD_RETRY_DELAY_MS);
    _pendingVolumeAddRetries.insert(volumeName);
    QTimer::singleShot(VOLUME_ADD_RETRY_DELAY_MS, this, [this, volumeId]() {
        if (!_state || !_state->vpkg()) {
            _pendingVolumeAddAttempts.erase(volumeId.toStdString());
            _pendingVolumeAddRetries.erase(volumeId.toStdString());
            return;
        }
        _pendingVolumeAddRetries.erase(volumeId.toStdString());
        processVolumeAddition(volumeId.toStdString());
    });
}

void FileWatcherService::processVolumeRemoval(const std::string& volumeName)
{
    if (!_state->vpkg()) return;

    Logger()->info("Processing volume removal of {}", volumeName);
    if (_state->vpkg()->removeSingleVolume(volumeName)) {
        _pendingVolumeAddAttempts.erase(volumeName);
        emit volumeCatalogChanged(QString::fromStdString(volumeName));
    }
}

void FileWatcherService::processVolumeRename(const std::string& oldVolumeName,
                                            const std::string& newVolumeName)
{
    if (!_state->vpkg()) return;

    Logger()->info("Processing volume rename {} -> {}", oldVolumeName, newVolumeName);
    _pendingVolumeAddAttempts.erase(oldVolumeName);
    _pendingVolumeAddAttempts.erase(newVolumeName);
    _pendingVolumeAddRetries.erase(oldVolumeName);
    _pendingVolumeAddRetries.erase(newVolumeName);

    _state->vpkg()->removeSingleVolume(oldVolumeName);
    if (_state->vpkg()->addSingleVolume(newVolumeName)) {
        emit volumeCatalogChanged(QString::fromStdString(newVolumeName));
    } else if (_state->vpkg()->hasVolume(oldVolumeName) || _state->hasVpkg()) {
        emit volumeCatalogChanged(QString::fromStdString(oldVolumeName));
    }
}

void FileWatcherService::processPendingEvents()
{
    // Removals, then renames, then additions
    for (const auto type : {InotifyEvent::Removal, InotifyEvent::Rename, InotifyEvent::Addition}) {
        for (const auto& evt : _pendingEvents) {
            if (evt.type != type) continue;
            if (type == InotifyEvent::Removal) processVolumeRemoval(evt.name);
            else if (type == InotifyEvent::Rename) processVolumeRename(evt.name, evt.newName);
            else processVolumeAddition(evt.name);
        }
    }
    _pendingEvents.clear();
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

#else // !__linux__

// Stub implementations for non-Linux platforms (no inotify)
FileWatcherService::FileWatcherService(CState* state, QObject* parent)
    : QObject(parent), _state(state) {}
FileWatcherService::~FileWatcherService() = default;
void FileWatcherService::startWatching() {}
void FileWatcherService::stopWatching() {}

#endif // __linux__
