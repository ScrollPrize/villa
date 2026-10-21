#pragma once

#include <QDateTime>
#include <QFuture>
#include <QObject>
#include <QPointer>
#include <QString>
#include <QStringList>

#include <cstdint>
#include <memory>
#include <vector>

class QLockFile;
class QTimer;
class VCCollection;
template <typename T> class QFutureWatcher;

// Crash recovery for VC3D sessions.
//
// Each running VC3D keeps a session folder in the recovery directory
// (~/.VC3D/recovery, next to VC3D.ini). The folder holds a lock file, a
// record of the open project and segment, and a copy of the point
// collections that is rewritten a few seconds after they change. Segment
// correction points (collections anchored to the segment grid) are left out
// of the copy, since they are already saved with their segment, and so are
// the seeding panel's temporary previews (temporaryCollections()). The copy is
// serialized on a worker thread, and the record is written after it, so the
// record never describes points that are not on disk yet. A clean exit
// deletes the folder, so a folder whose lock no running VC3D holds is left
// over from a session that ended unexpectedly; SessionRecoveryDialog offers
// it at the next start. Nothing is written outside the recovery directory.
// The backup/session_recovery setting turns all of this off.
class SessionRecovery : public QObject
{
    Q_OBJECT

public:
    struct LeftoverSession
    {
        QString directory;
        QString projectPath;
        QString segmentId;
        QDateTime started;
        QDateTime updated;
        int collectionCount{0};
        int pointCount{0};

        [[nodiscard]] QString pointsFile() const;
        [[nodiscard]] bool hasPoints() const;
    };

    static constexpr int kSaveDelayMs = 3000;
    static constexpr int kMaxLeftoverSessions = 5;

    explicit SessionRecovery(QString directory, QObject* parent = nullptr);
    ~SessionRecovery() override;

    // <settings directory>/recovery; follows VC3D_CONFIG_DIR like VC3D.ini.
    static QString defaultDirectory();
    // The backup/session_recovery setting (on unless turned off).
    static bool enabledInSettings();

    // Creates this session's folder and lock, prunes old leftovers, and
    // keeps a copy of collection from now on. A clean exit is recorded when
    // the application quits normally. When the folder or its lock cannot be
    // created, emits writeFailed and keeps nothing.
    bool begin(VCCollection* collection);
    // Both rewrite the record together with the points copy.
    void setProject(const QString& path);
    void setSegment(const QString& segmentId);
    // Starts writing the points copy and the session record now, in the
    // background. While a write runs, the newest state is written after it.
    void flush();
    // Waits until everything flushed so far is written. False once a write
    // has failed; after the first failure nothing more is written.
    bool waitForWrite();
    // Writes the points copy and the record now and waits for them. True when
    // this session's folder then holds a copy of at least one point; false
    // when recovery has not begun, a write failed or there is nothing to copy.
    bool writeNow();
    [[nodiscard]] bool isWriting() const;
    // Deletes this session's folder, after any running write. Called on a
    // normal quit.
    void markCleanExit();

    [[nodiscard]] QString sessionDirectory() const { return _sessionDir; }
    [[nodiscard]] QString segmentId() const { return _segmentId; }

    // Names of the collections VC3D fills with temporary previews, which a
    // recovery copy leaves out.
    [[nodiscard]] static const QStringList& temporaryCollections();

    // Sessions that ended without a clean exit and left points to recover,
    // newest first. Sessions of running VC3D instances are skipped, and so
    // are folders less than a minute old without a record, which another
    // VC3D may be creating.
    [[nodiscard]] std::vector<LeftoverSession> leftoverSessions() const;

    // Takes a leftover session's lock, so no other VC3D offers it while this
    // one does; destroying the result lets it go. Null when a running VC3D
    // holds it.
    [[nodiscard]] static std::shared_ptr<QLockFile> claim(const LeftoverSession& session);

    // Loads the leftover session's points into collection. Refused, with
    // nothing changed, while collection holds points or segment correction
    // collections: loading replaces them, and VC3D would then rewrite the
    // active segment's corrections.json.
    static bool restorePoints(const LeftoverSession& session,
                              VCCollection* collection,
                              QString* error = nullptr);
    // Deletes a leftover session's folder.
    static bool discard(const LeftoverSession& session);

signals:
    // The session folder could not be created or a recovery copy could not
    // be written; emitted once, after which nothing more is written.
    void writeFailed(const QString& message);

private:
    void scheduleSave();
    // Whether a change to this collection can change the copy.
    [[nodiscard]] bool isCopied(uint64_t collectionId) const;
    void collectWrite();
    void pruneLeftovers();

    QString _directory;
    QString _sessionDir;
    std::shared_ptr<QLockFile> _lock;
    QPointer<VCCollection> _collection;
    QTimer* _saveTimer{nullptr};
    QFuture<bool> _write;
    QFutureWatcher<bool>* _writeWatcher{nullptr};
    bool _flushPending{false};
    bool _writeFailed{false};
    QDateTime _started;
    QString _projectPath;
    QString _segmentId;
    int _collectionCount{0};
    int _pointCount{0};
};
