#include "SessionRecovery.hpp"

#include "VCSettings.hpp"
#include "vc/ui/VCCollection.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLockFile>
#include <QSaveFile>
#include <QSet>
#include <QSettings>
#include <QSysInfo>
#include <QTimer>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <system_error>

namespace {

const QString kSessionPrefix = QStringLiteral("session-");
const QString kLockFile = QStringLiteral("session.lock");
const QString kRecordFile = QStringLiteral("session.json");
const QString kPointsFile = QStringLiteral("points.json");
constexpr int kRecordVersion = 1;

std::string localPath(const QString& path)
{
    return QFile::encodeName(path).toStdString();
}

// Lock files this process holds through takeLock().
QSet<QString>& locksHeldHere()
{
    static QSet<QString> held;
    return held;
}

// Takes a session folder's lock, or returns null when a running VC3D holds
// it. QLockFile treats a lock whose process has died as stale and takes it
// over, which is how a crashed session shows up here. It cannot tell a dead
// process that had this process's PID (containers often reuse PIDs) from
// this one, so such a lock that this process does not hold is stale too.
std::shared_ptr<QLockFile> takeLock(const QString& directory)
{
    const QString path = QDir(directory).filePath(kLockFile);
    auto lock = std::make_unique<QLockFile>(path);
    lock->setStaleLockTime(0);
    if (!lock->tryLock(0)) {
        qint64 pid = 0;
        QString host;
        QString application;
        const bool deadTwin = !locksHeldHere().contains(path) &&
                              lock->getLockInfo(&pid, &host, &application) &&
                              pid == QCoreApplication::applicationPid() &&
                              host == QSysInfo::machineHostName();
        // Unreadable locks count as held, so nothing is offered or pruned.
        if (!deadTwin || !lock->removeStaleLockFile() || !lock->tryLock(0)) {
            return nullptr;
        }
    }
    locksHeldHere().insert(path);
    return std::shared_ptr<QLockFile>(lock.release(), [path](QLockFile* held) {
        locksHeldHere().remove(path);
        delete held;  // Unlocks
    });
}

bool heldByRunningSession(const QString& directory)
{
    return takeLock(directory) == nullptr;
}

bool isTemporaryCollection(const std::string& name)
{
    return SessionRecovery::temporaryCollections().contains(QString::fromStdString(name));
}

// The part of the collections a recovery copy keeps, or null when that has
// no points. Segment correction points (anchored collections) are left out,
// since their segment's corrections.json already keeps them, and so are the
// seeding panel's temporary previews.
std::shared_ptr<const PointCollections> recoverablePoints(const PointCollections& live,
                                                          int* collections,
                                                          int* points)
{
    auto copy = std::make_shared<PointCollections>(live);
    *collections = 0;
    *points = 0;
    for (const auto& [id, collection] : live.getAllCollections()) {
        if (collection.anchor2d || isTemporaryCollection(collection.name)) {
            copy->clearCollection(id);
        } else {
            ++*collections;
            *points += static_cast<int>(collection.points.size());
        }
    }
    return *points > 0 ? copy : nullptr;
}

// Whether loading a points file over these collections would lose anything.
bool holdsWork(const PointCollections& points)
{
    const auto& collections = points.getAllCollections();
    return std::any_of(collections.begin(), collections.end(), [](const auto& entry) {
        return !entry.second.points.empty() || entry.second.anchor2d;
    });
}

// One background write: the points copy (or its removal), then the record.
struct SessionWrite
{
    QString directory;
    std::shared_ptr<const PointCollections> points;  // Null: no points to keep
    QByteArray record;
};

bool writeSession(const SessionWrite& job)
{
    const QString pointsPath = QDir(job.directory).filePath(kPointsFile);
    if (job.points) {
        // Write beside the copy and rename over it, so a crash mid-write
        // leaves the previous copy intact.
        const QString temporary = pointsPath + QStringLiteral(".tmp");
        bool saved = job.points->saveToJSON(localPath(temporary));
        if (saved) {
            std::error_code error;
            std::filesystem::rename(localPath(temporary), localPath(pointsPath), error);
            saved = !error;
        }
        if (!saved) {
            QFile::remove(temporary);
            return false;
        }
    } else if (QFileInfo::exists(pointsPath) && !QFile::remove(pointsPath)) {
        return false;
    }
    QSaveFile file(QDir(job.directory).filePath(kRecordFile));
    return file.open(QIODevice::WriteOnly) && file.write(job.record) == job.record.size() && file.commit();
}

// Session folders under directory. Symbolic links are skipped, so pruning
// and Discard never delete what a link points to.
QFileInfoList sessionFolders(const QString& directory)
{
    return QDir(directory).entryInfoList({kSessionPrefix + QStringLiteral("*")},
                                         QDir::Dirs | QDir::NoDotAndDotDot | QDir::NoSymLinks);
}

// A folder that another VC3D may be creating right now: it has no record yet
// and changed in the last minute. begin() makes the folder before it takes
// the lock, so in between such a folder looks like a crashed session's.
// Pruning and listing leave it alone, and do not even try its lock, which
// would make that VC3D's begin() fail.
constexpr int kFreshFolderSeconds = 60;

bool mayBeStarting(const QFileInfo& folder)
{
    if (QFileInfo::exists(QDir(folder.absoluteFilePath()).filePath(kRecordFile))) {
        return false;
    }
    const QDateTime changed = folder.lastModified();
    return changed.isValid() && changed.secsTo(QDateTime::currentDateTime()) < kFreshFolderSeconds;
}

SessionRecovery::LeftoverSession readLeftover(const QString& directory)
{
    SessionRecovery::LeftoverSession session;
    session.directory = directory;
    QFile file(QDir(directory).filePath(kRecordFile));
    if (file.open(QIODevice::ReadOnly)) {
        const QJsonObject record = QJsonDocument::fromJson(file.readAll()).object();
        session.projectPath = record.value(QStringLiteral("project")).toString();
        session.segmentId = record.value(QStringLiteral("segment")).toString();
        session.started = QDateTime::fromString(record.value(QStringLiteral("started")).toString(), Qt::ISODate);
        session.updated = QDateTime::fromString(record.value(QStringLiteral("updated")).toString(), Qt::ISODate);
        session.collectionCount = record.value(QStringLiteral("collections")).toInt();
        session.pointCount = record.value(QStringLiteral("points")).toInt();
    }
    if (!session.updated.isValid()) {
        session.updated = QFileInfo(directory).lastModified();
    }
    return session;
}

} // namespace

QString SessionRecovery::LeftoverSession::pointsFile() const
{
    return QDir(directory).filePath(kPointsFile);
}

bool SessionRecovery::LeftoverSession::hasPoints() const
{
    return pointCount > 0 && QFileInfo::exists(pointsFile());
}

const QStringList& SessionRecovery::temporaryCollections()
{
    // SeedingWidget's ray, peak and radius previews: recomputed on demand.
    static const QStringList names{QStringLiteral("ray_preview"), QStringLiteral("seeding_peaks"),
                                   QStringLiteral("seeding_preview")};
    return names;
}

SessionRecovery::SessionRecovery(QString directory, QObject* parent)
    : QObject(parent)
    , _directory(std::move(directory))
    , _saveTimer(new QTimer(this))
    , _writeWatcher(new QFutureWatcher<bool>(this))
{
    _saveTimer->setSingleShot(true);
    connect(_saveTimer, &QTimer::timeout, this, &SessionRecovery::flush);
    connect(_writeWatcher, &QFutureWatcher<bool>::finished, this, [this]() {
        collectWrite();
        if (_flushPending) {
            flush();
        }
    });
}

SessionRecovery::~SessionRecovery()
{
    // Leave the folder complete, as a crash after the write would.
    if (_write.isValid()) {
        _write.waitForFinished();
    }
}

QString SessionRecovery::defaultDirectory()
{
    return QDir(QFileInfo(vc3d::settingsFilePath()).absolutePath()).filePath(QStringLiteral("recovery"));
}

bool SessionRecovery::enabledInSettings()
{
    const QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    return settings
        .value(vc3d::settings::backup::SESSION_RECOVERY, vc3d::settings::backup::SESSION_RECOVERY_DEFAULT)
        .toBool();
}

bool SessionRecovery::begin(VCCollection* collection)
{
    if (_lock) {
        return true;
    }
    const QDateTime now = QDateTime::currentDateTimeUtc();
    const QString name = kSessionPrefix + now.toString(QStringLiteral("yyyyMMdd-HHmmss-zzz-")) +
                         QString::number(QCoreApplication::applicationPid());
    const QString directory = QDir(_directory).filePath(name);
    auto lock = QDir().mkpath(directory) ? takeLock(directory) : nullptr;
    if (!lock) {
        _writeFailed = true;
        emit writeFailed(tr("Could not create the crash recovery folder %1. Crash recovery is off "
                            "until VC3D restarts.")
                             .arg(QDir::toNativeSeparators(directory)));
        return false;
    }
    _lock = std::move(lock);
    _sessionDir = directory;
    _started = now;

    _collection = collection;
    if (collection) {
        // Changes to the temporary previews alone leave the copy as it is.
        const auto changed = [this] { scheduleSave(); };
        const auto changedIn = [this](uint64_t id) {
            if (isCopied(id)) {
                scheduleSave();
            }
        };
        connect(collection, &VCCollection::collectionChanged, this, changedIn);
        connect(collection, &VCCollection::collectionsAdded, this, [this](const std::vector<uint64_t>& ids) {
            if (std::any_of(ids.begin(), ids.end(), [this](uint64_t id) { return isCopied(id); })) {
                scheduleSave();
            }
        });
        connect(collection, &VCCollection::collectionRemoved, this, changed);
        connect(collection, &VCCollection::pointAdded, this,
                [changedIn](const ColPoint& point) { changedIn(point.collectionId); });
        connect(collection, &VCCollection::pointsAdded, this, [this](const std::vector<ColPoint>& points) {
            if (std::any_of(points.begin(), points.end(),
                            [this](const ColPoint& point) { return isCopied(point.collectionId); })) {
                scheduleSave();
            }
        });
        connect(collection, &VCCollection::pointChanged, this,
                [changedIn](const ColPoint& point) { changedIn(point.collectionId); });
        connect(collection, &VCCollection::pointRemoved, this,
                [changedIn](vc::PointRef point) { changedIn(point.collectionId); });
        connect(collection, &VCCollection::pointsRemoved, this, [this](const std::vector<vc::PointRef>& points) {
            if (std::any_of(points.begin(), points.end(),
                            [this](const vc::PointRef& point) { return isCopied(point.collectionId); })) {
                scheduleSave();
            }
        });
    }
    if (auto* app = QCoreApplication::instance()) {
        connect(app, &QCoreApplication::aboutToQuit, this, &SessionRecovery::markCleanExit);
    }
    flush();
    pruneLeftovers();
    return true;
}

void SessionRecovery::setProject(const QString& path)
{
    if (path == _projectPath) {
        return;
    }
    _projectPath = path;
    _segmentId.clear();
    flush();
}

void SessionRecovery::setSegment(const QString& segmentId)
{
    if (segmentId == _segmentId) {
        return;
    }
    _segmentId = segmentId;
    flush();
}

bool SessionRecovery::isCopied(uint64_t collectionId) const
{
    if (!_collection) {
        return false;
    }
    const auto& collections = _collection->getAllCollections();
    const auto found = collections.find(collectionId);
    return found == collections.end() || !isTemporaryCollection(found->second.name);
}

void SessionRecovery::scheduleSave()
{
    // Not restarted by later changes, so a steady stream of edits is still
    // saved every few seconds.
    if (!_sessionDir.isEmpty() && !_writeFailed && !_saveTimer->isActive()) {
        _saveTimer->start(kSaveDelayMs);
    }
}

void SessionRecovery::flush()
{
    _saveTimer->stop();
    collectWrite();
    if (_sessionDir.isEmpty() || _writeFailed) {
        _flushPending = false;
        return;
    }
    if (_write.isValid()) {
        _flushPending = true;  // Written when the running write finishes
        return;
    }
    _flushPending = false;

    // Copying the collections is cheap; serializing them is not, so that
    // happens on a worker thread.
    std::shared_ptr<const PointCollections> points;
    _collectionCount = 0;
    _pointCount = 0;
    if (_collection) {
        points = recoverablePoints(*_collection, &_collectionCount, &_pointCount);
    }

    QJsonObject record;
    record[QStringLiteral("version")] = kRecordVersion;
    record[QStringLiteral("pid")] = static_cast<qint64>(QCoreApplication::applicationPid());
    record[QStringLiteral("started")] = _started.toString(Qt::ISODateWithMs);
    record[QStringLiteral("updated")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
    record[QStringLiteral("project")] = _projectPath;
    record[QStringLiteral("segment")] = _segmentId;
    record[QStringLiteral("collections")] = _collectionCount;
    record[QStringLiteral("points")] = _pointCount;

    _write = QtConcurrent::run(writeSession,
                               SessionWrite{_sessionDir, std::move(points), QJsonDocument(record).toJson()});
    _writeWatcher->setFuture(_write);
}

bool SessionRecovery::isWriting() const
{
    return _write.isValid() && !_write.isFinished();
}

bool SessionRecovery::waitForWrite()
{
    for (;;) {
        if (_write.isValid()) {
            _write.waitForFinished();
        }
        collectWrite();
        if (!_flushPending || _writeFailed) {
            return !_writeFailed;
        }
        flush();  // The write that was waiting for the one just finished
    }
}

bool SessionRecovery::writeNow()
{
    flush();
    return waitForWrite() && !_sessionDir.isEmpty() && _pointCount > 0 &&
           QFileInfo::exists(QDir(_sessionDir).filePath(kPointsFile));
}

// Takes the result of a finished write, once.
void SessionRecovery::collectWrite()
{
    if (!_write.isValid() || !_write.isFinished()) {
        return;
    }
    const bool written = _write.result();
    _write = QFuture<bool>();
    if (!written && !_writeFailed) {
        _writeFailed = true;
        _flushPending = false;
        emit writeFailed(tr("Could not write the crash recovery copy in %1. Crash recovery is off "
                            "until VC3D restarts.")
                             .arg(QDir::toNativeSeparators(_sessionDir)));
    }
}

void SessionRecovery::markCleanExit()
{
    _saveTimer->stop();
    _flushPending = false;
    if (_collection) {
        disconnect(_collection, nullptr, this, nullptr);
        _collection = nullptr;
    }
    waitForWrite();
    if (_sessionDir.isEmpty()) {
        return;
    }
    _lock.reset();  // Unlocks
    QDir(_sessionDir).removeRecursively();
    _sessionDir.clear();
}

std::vector<SessionRecovery::LeftoverSession> SessionRecovery::leftoverSessions() const
{
    std::vector<LeftoverSession> sessions;
    const QFileInfoList folders = sessionFolders(_directory);
    for (const QFileInfo& folder : folders) {
        const QString directory = folder.absoluteFilePath();
        if (directory == QFileInfo(_sessionDir).absoluteFilePath() || mayBeStarting(folder) ||
            heldByRunningSession(directory)) {
            continue;
        }
        LeftoverSession session = readLeftover(directory);
        if (session.hasPoints()) {
            sessions.push_back(std::move(session));
        }
    }
    std::sort(sessions.begin(), sessions.end(), [](const LeftoverSession& a, const LeftoverSession& b) {
        return a.updated > b.updated;
    });
    return sessions;
}

void SessionRecovery::pruneLeftovers()
{
    // Folders with nothing to recover go at once; otherwise keep the newest few.
    const QFileInfoList folders = sessionFolders(_directory);
    for (const QFileInfo& folder : folders) {
        const QString directory = folder.absoluteFilePath();
        if (directory == QFileInfo(_sessionDir).absoluteFilePath() || mayBeStarting(folder) ||
            heldByRunningSession(directory)) {
            continue;
        }
        const LeftoverSession session = readLeftover(directory);
        if (!session.hasPoints()) {
            discard(session);
        }
    }
    const auto sessions = leftoverSessions();
    for (std::size_t i = kMaxLeftoverSessions; i < sessions.size(); ++i) {
        discard(sessions[i]);
    }
}

std::shared_ptr<QLockFile> SessionRecovery::claim(const LeftoverSession& session)
{
    return session.directory.isEmpty() ? nullptr : takeLock(session.directory);
}

bool SessionRecovery::restorePoints(const LeftoverSession& session,
                                    VCCollection* collection,
                                    QString* error)
{
    QString problem;
    if (!collection || !session.hasPoints()) {
        problem = tr("no saved points");
    } else if (holdsWork(*collection)) {
        problem = tr("points or segment correction points are loaded now");
    } else if (!collection->loadFromJSON(localPath(session.pointsFile()))) {
        problem = tr("%1 could not be read").arg(QDir::toNativeSeparators(session.pointsFile()));
    }
    if (error) {
        *error = problem;
    }
    return problem.isEmpty();
}

bool SessionRecovery::discard(const LeftoverSession& session)
{
    // Never through a symbolic link: removeRecursively would empty its target.
    return !session.directory.isEmpty() && !QFileInfo(session.directory).isSymLink() &&
           QDir(session.directory).removeRecursively();
}
