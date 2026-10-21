#include "SessionRecovery.hpp"
#include "SessionRecoveryDialog.hpp"
#include "vc/ui/VCCollection.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QDialog>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLockFile>
#include <QPointer>
#include <QProcess>
#include <QPushButton>
#include <QSaveFile>
#include <QSettings>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QtTest/QtTest>

#include "VCSettings.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <system_error>

#ifdef Q_OS_UNIX
#include <sys/stat.h>
#endif

namespace {

constexpr auto kChildEnv = "VC3D_SESSION_RECOVERY_TEST_CHILD";
constexpr int kChildExitCode = 42;
const QString kProject = QStringLiteral("/data/PHerc0139.volpkg.json");
const QString kSegment = QStringLiteral("w045");

void fillPoints(VCCollection& points)
{
    points.addPoints("corrections", {{100.5f, 200.25f, 300.0f}, {101.0f, 201.0f, 301.0f}});
    const ColPoint winding = points.addPoint("windings", {10.0f, 20.0f, 30.0f});
    ColPoint annotated = winding;
    annotated.winding_annotation = 3.0f;
    points.updatePoint(annotated);
    const auto id = points.getCollectionId("windings");
    points.setCollectionTag(*id, "source", "recovery-test");
    points.setCollectionColor(*id, {0.25f, 0.5f, 0.75f});
}

bool samePoints(const VCCollection& a, const VCCollection& b)
{
    const auto& left = a.getAllCollections();
    const auto& right = b.getAllCollections();
    if (left.size() != right.size()) {
        return false;
    }
    for (const auto& [id, collection] : left) {
        const auto other = right.find(id);
        if (other == right.end() || other->second.name != collection.name ||
            other->second.uuid != collection.uuid || other->second.tags != collection.tags ||
            other->second.color != collection.color ||
            other->second.points.size() != collection.points.size()) {
            return false;
        }
        for (const auto& [pointId, point] : collection.points) {
            const auto match = other->second.points.find(pointId);
            if (match == other->second.points.end() || match->second.p != point.p ||
                match->second.creation_time != point.creation_time ||
                (match->second.winding_annotation != point.winding_annotation &&
                 !(std::isnan(point.winding_annotation) && std::isnan(match->second.winding_annotation)))) {
                return false;
            }
        }
    }
    return true;
}

QStringList filesUnder(const QString& directory)
{
    QStringList files;
    QDirIterator it(directory, QDir::Files | QDir::Hidden, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        files << it.next();
    }
    files.sort();
    return files;
}

QJsonObject readRecord(const QString& sessionDirectory)
{
    QFile file(QDir(sessionDirectory).filePath("session.json"));
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    return QJsonDocument::fromJson(file.readAll()).object();
}

// Restores VC3D_CONFIG_DIR when a test that changed it ends.
class ConfigDirOverride
{
public:
    explicit ConfigDirOverride(const QString& directory) : _previous(qgetenv("VC3D_CONFIG_DIR"))
    {
        qputenv("VC3D_CONFIG_DIR", directory.toUtf8());
    }
    ~ConfigDirOverride()
    {
        if (_previous.isNull()) {
            qunsetenv("VC3D_CONFIG_DIR");
        } else {
            qputenv("VC3D_CONFIG_DIR", _previous);
        }
    }

private:
    QByteArray _previous;
};

// Writes a leftover session folder by hand, as a crashed VC3D would leave it.
void writeLeftover(const QString& root, const QString& name, const QDateTime& updated, bool withPoints,
                   const QString& project = kProject, const QString& segment = kSegment)
{
    const QString directory = QDir(root).filePath(name);
    QVERIFY(QDir().mkpath(directory));
    QJsonObject record;
    record["version"] = 1;
    record["updated"] = updated.toString(Qt::ISODateWithMs);
    record["project"] = project;
    record["segment"] = segment;
    record["collections"] = withPoints ? 1 : 0;
    record["points"] = withPoints ? 1 : 0;
    QSaveFile file(QDir(directory).filePath("session.json"));
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.write(QJsonDocument(record).toJson());
    QVERIFY(file.commit());
    if (withPoints) {
        VCCollection points;
        points.addPoint("corrections", {1.0f, 2.0f, 3.0f});
        QVERIFY(points.saveToJSON(QDir(directory).filePath("points.json").toStdString()));
    }
}

// A segment's correction points, as SegmentationModule loads them from its
// corrections.json into the shared collection.
void addCorrection(VCCollection& points, const std::string& name)
{
    points.addPoint(name, {7.0f, 8.0f, 9.0f});
    points.setCollectionAnchor2d(*points.getCollectionId(name), cv::Vec2f(5.0f, 6.0f));
}

// Records what the recovery dialog asks of the main window. Opening a
// project clears the collections, as CWindow::CloseVolume does.
struct FakeWindow
{
    VCCollection* points{nullptr};
    QString project;
    bool openSucceeds{true};
    QStringList calls;
    // When set, selecting a segment also records what is on disk at that
    // moment: the running session's copy and whether oldCopy still exists.
    SessionRecovery* recovery{nullptr};
    QString oldCopy;

    SessionRecoveryDialog::Actions actions()
    {
        return {
            [this] { return project; },
            [this](const QString& path, QString* error) {
                calls << "open " + path;
                if (!openSucceeds) {
                    *error = QStringLiteral("not found");
                    return false;
                }
                points->clearAll();
                project = path;
                return true;
            },
            [this](const QString& id, QString*) {
                QString call =
                    QStringLiteral("segment %1 with %2 points").arg(id).arg(points->getPoints("corrections").size());
                if (recovery) {
                    VCCollection copy;
                    const bool read = copy.loadFromJSON(
                        QDir(recovery->sessionDirectory()).filePath("points.json").toStdString());
                    call += QStringLiteral("; copy %1 points for segment %2; old copy %3")
                                .arg(read ? copy.getPoints("corrections").size() : 0)
                                .arg(readRecord(recovery->sessionDirectory()).value("segment").toString(),
                                     QFileInfo::exists(oldCopy) ? QStringLiteral("kept") : QStringLiteral("gone"));
                }
                calls << call;
                addCorrection(*points, "segment corrections");
                return true;
            },
            [this](const QString& text) { calls << "status " + text; }};
    }
};

// Child process: run a session with points, then die without any cleanup,
// leaving its lock file behind like a crash or a kill would.
int runCrashingChild(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    const QStringList dirs = QString::fromLocal8Bit(qgetenv(kChildEnv)).split(QLatin1Char('|'));
    VCCollection points;
    SessionRecovery recovery(dirs.value(0));
    if (!recovery.begin(&points)) {
        return 1;
    }
    recovery.setProject(kProject);
    recovery.setSegment(kSegment);
    fillPoints(points);
    recovery.flush();
    if (!recovery.waitForWrite() || !points.saveToJSON(QDir(dirs.value(1)).filePath("expected.json").toStdString())) {
        return 2;
    }
    std::_Exit(kChildExitCode);
}

} // namespace

class SessionRecoveryTest : public QObject
{
    Q_OBJECT

private slots:
    void defaultDirectoryFollowsConfigDir()
    {
        QTemporaryDir config;
        const QByteArray previous = qgetenv("VC3D_CONFIG_DIR");
        qputenv("VC3D_CONFIG_DIR", config.path().toUtf8());
        QCOMPARE(QDir::cleanPath(SessionRecovery::defaultDirectory()),
                 QDir::cleanPath(QDir(config.path()).filePath("recovery")));
        if (previous.isNull()) {
            qunsetenv("VC3D_CONFIG_DIR");
        } else {
            qputenv("VC3D_CONFIG_DIR", previous);
        }
    }

    void cleanExitLeavesNothingToRecover()
    {
        QTemporaryDir root;
        VCCollection points;
        {
            SessionRecovery session(root.path());
            QVERIFY(session.begin(&points));
            fillPoints(points);
            session.flush();
            QVERIFY(session.waitForWrite());
            QVERIFY(QFileInfo::exists(QDir(session.sessionDirectory()).filePath("points.json")));
            session.markCleanExit();
            QVERIFY(session.sessionDirectory().isEmpty());
        }
        SessionRecovery next(root.path());
        QVERIFY(next.leftoverSessions().empty());
        QVERIFY(QDir(root.path()).entryList(QDir::Dirs | QDir::NoDotAndDotDot).isEmpty());
    }

    void crashedSessionIsOfferedWithIdenticalPoints()
    {
        QTemporaryDir root;
        QTemporaryDir reference;
        QProcess child;
        QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
        environment.insert(kChildEnv, root.path() + "|" + reference.path());
        child.setProcessEnvironment(environment);
        child.start(QCoreApplication::applicationFilePath(), {});
        QVERIFY(child.waitForFinished(30000));
        QCOMPARE(child.exitCode(), kChildExitCode);

        // The crashed session left its folder and lock behind.
        const QStringList folders = QDir(root.path()).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        QCOMPARE(folders.size(), 1);
        QVERIFY(QFileInfo::exists(QDir(root.path()).filePath(folders.front() + "/session.lock")));

        VCCollection live;
        SessionRecovery next(root.path());
        QVERIFY(next.begin(&live));
        const auto leftovers = next.leftoverSessions();
        QCOMPARE(leftovers.size(), std::size_t{1});
        const auto& leftover = leftovers.front();
        QCOMPARE(leftover.projectPath, kProject);
        QCOMPARE(leftover.segmentId, kSegment);
        QCOMPARE(leftover.pointCount, 3);
        QCOMPARE(leftover.collectionCount, 2);
        QVERIFY(leftover.hasPoints());

        // Never applied on its own: the live collection is still empty.
        QVERIFY(live.getAllCollections().empty());

        QString error;
        QVERIFY2(SessionRecovery::restorePoints(leftover, &live, &error), qPrintable(error));
        VCCollection expected;
        QVERIFY(expected.loadFromJSON(QDir(reference.path()).filePath("expected.json").toStdString()));
        QVERIFY(samePoints(live, expected));

        QVERIFY(SessionRecovery::discard(leftover));
        QVERIFY(next.leftoverSessions().empty());
        next.markCleanExit();
    }

    void sessionDroppedWithoutCleanExitIsOffered()
    {
        QTemporaryDir root;
        VCCollection points;
        auto session = std::make_unique<SessionRecovery>(root.path());
        QVERIFY(session->begin(&points));
        session->setProject(kProject);
        session->setSegment(kSegment);
        fillPoints(points);
        session->flush();
        QVERIFY(session->waitForWrite());
        session.reset();  // Lock released, folder kept: same as a crash.

        SessionRecovery next(root.path());
        const auto leftovers = next.leftoverSessions();
        QCOMPARE(leftovers.size(), std::size_t{1});
        QCOMPARE(leftovers.front().pointCount, 3);
        QVERIFY(SessionRecovery::discard(leftovers.front()));

        // A session that placed no points is neither offered nor kept, even
        // with a project and a segment open: an exit without cleanup (Ctrl+C,
        // kill) is common and must not prompt on every start.
        VCCollection empty;
        session = std::make_unique<SessionRecovery>(root.path());
        QVERIFY(session->begin(&empty));
        session->setProject(kProject);
        session->setSegment(kSegment);
        QVERIFY(session->waitForWrite());
        session.reset();
        SessionRecovery third(root.path());
        QVERIFY(third.leftoverSessions().empty());
        QVERIFY(third.begin(nullptr));
        QCOMPARE(QDir(root.path()).entryList(QDir::Dirs | QDir::NoDotAndDotDot).size(), 1);
        third.markCleanExit();
    }

    void copyLeavesOutSegmentCorrections()
    {
        QTemporaryDir root;
        VCCollection points;
        auto session = std::make_unique<SessionRecovery>(root.path());
        QVERIFY(session->begin(&points));
        session->setProject(kProject);
        addCorrection(points, "segment corrections");
        session->flush();
        QVERIFY(session->waitForWrite());
        // Correction points alone leave nothing to recover.
        QVERIFY(!QFileInfo::exists(QDir(session->sessionDirectory()).filePath("points.json")));

        points.addPoint("my points", {1.0f, 2.0f, 3.0f});
        session->flush();
        QVERIFY(session->waitForWrite());
        session.reset();

        SessionRecovery next(root.path());
        const auto leftovers = next.leftoverSessions();
        QCOMPARE(leftovers.size(), std::size_t{1});
        QCOMPARE(leftovers.front().pointCount, 1);
        QCOMPARE(leftovers.front().collectionCount, 1);
        VCCollection restored;
        QVERIFY(SessionRecovery::restorePoints(leftovers.front(), &restored));
        QCOMPARE(restored.getAllCollections().size(), std::size_t{1});
        QCOMPARE(restored.getPoints("my points").size(), std::size_t{1});
    }

    void restoreNeverReplacesLoadedWork()
    {
        QTemporaryDir root;
        QTemporaryDir segment;
        writeLeftover(root.path(), "session-crashed", QDateTime::currentDateTimeUtc(), true);
        SessionRecovery session(root.path());
        const auto leftovers = session.leftoverSessions();
        QCOMPARE(leftovers.size(), std::size_t{1});

        // Loading the copy would drop the active segment's correction points,
        // and VC3D's corrections autosave would then rewrite or delete its
        // corrections.json.
        VCCollection live;
        addCorrection(live, "segment corrections");
        QVERIFY(live.saveToSegmentPath(segment.path().toStdString()));
        QString error;
        QVERIFY(!SessionRecovery::restorePoints(leftovers.front(), &live, &error));
        QVERIFY(error.contains("correction"));
        QVERIFY(live.saveToSegmentPath(segment.path().toStdString()));  // the autosave
        VCCollection onDisk;
        QVERIFY(onDisk.loadFromSegmentPath(segment.path().toStdString()));
        QCOMPARE(onDisk.getPoints("segment corrections").size(), std::size_t{1});

        // Nor does it replace points placed in this session.
        VCCollection work;
        work.addPoint("new work", {5.0f, 5.0f, 5.0f});
        QVERIFY(!SessionRecovery::restorePoints(leftovers.front(), &work));
        QCOMPARE(work.getAllCollections().size(), std::size_t{1});
        QCOMPARE(session.leftoverSessions().size(), std::size_t{1});
    }

    void claimedSessionIsNotOfferedTwice()
    {
        // Two VC3D instances started together after a crash: only one offers.
        QTemporaryDir root;
        writeLeftover(root.path(), "session-crashed", QDateTime::currentDateTimeUtc(), true);
        SessionRecovery first(root.path());
        SessionRecovery second(root.path());
        QPointer<SessionRecoveryDialog> dialog = SessionRecoveryDialog::offer(nullptr, first, nullptr, {});
        QVERIFY(dialog);
        QVERIFY(second.leftoverSessions().empty());
        QVERIFY(SessionRecoveryDialog::offer(nullptr, second, nullptr, {}) == nullptr);

        dialog->findChild<QPushButton*>("laterSessionButton")->click();
        QCOMPARE(second.leftoverSessions().size(), std::size_t{1});
        QPointer<SessionRecoveryDialog> again = SessionRecoveryDialog::offer(nullptr, second, nullptr, {});
        QVERIFY(again);
        again->close();
    }

    void lockOfADeadProcessWithThisPidIsStale()
    {
        // Containers often start VC3D with the same PID every time. QLockFile
        // then takes a crashed session's lock for this process's own.
        QTemporaryDir root;
        writeLeftover(root.path(), "session-crashed", QDateTime::currentDateTimeUtc(), true);
        QByteArray contents;
        {
            QLockFile twin(QDir(root.path()).filePath("twin.lock"));
            QVERIFY(twin.tryLock(0));
            QFile file(QDir(root.path()).filePath("twin.lock"));
            QVERIFY(file.open(QIODevice::ReadOnly));
            contents = file.readAll();
        }
        const QString lockPath = QDir(root.path()).filePath("session-crashed/session.lock");
        QFile lock(lockPath);
        QVERIFY(lock.open(QIODevice::WriteOnly));
        QCOMPARE(lock.write(contents), contents.size());
        lock.close();
        {
            QLockFile probe(lockPath);
            probe.setStaleLockTime(0);
            QVERIFY(!probe.tryLock(0));  // QLockFile alone takes it as held
        }

        SessionRecovery session(root.path());
        QCOMPARE(session.leftoverSessions().size(), std::size_t{1});
    }

    void failedBeginIsReported()
    {
        QTemporaryDir root;
        const QString notADirectory = QDir(root.path()).filePath("recovery");
        QFile file(notADirectory);
        QVERIFY(file.open(QIODevice::WriteOnly));
        file.close();
        SessionRecovery session(notADirectory);
        QSignalSpy failures(&session, &SessionRecovery::writeFailed);
        VCCollection points;
        QVERIFY(!session.begin(&points));
        QCOMPARE(failures.count(), 1);
        points.addPoint("corrections", {1.0f, 2.0f, 3.0f});
        session.flush();
        QVERIFY(!session.isWriting());
        QCOMPARE(failures.count(), 1);
    }

    void runningSessionIsNotOffered()
    {
        QTemporaryDir root;
        VCCollection points;
        SessionRecovery running(root.path());
        QVERIFY(running.begin(&points));
        running.setProject(kProject);
        running.setSegment(kSegment);
        fillPoints(points);
        running.flush();
        QVERIFY(running.waitForWrite());

        SessionRecovery other(root.path());
        QVERIFY(other.leftoverSessions().empty());
        QVERIFY(other.begin(nullptr));
        QVERIFY(other.leftoverSessions().empty());
        QVERIFY(QFileInfo::exists(running.sessionDirectory()));
        other.markCleanExit();
        running.markCleanExit();
    }

    void copyFollowsEditsAfterADelay()
    {
        QTemporaryDir root;
        VCCollection points;
        SessionRecovery session(root.path());
        QVERIFY(session.begin(&points));
        const QString copy = QDir(session.sessionDirectory()).filePath("points.json");
        QVERIFY(!QFileInfo::exists(copy));

        points.addPoint("corrections", {1.0f, 2.0f, 3.0f});
        QTRY_VERIFY_WITH_TIMEOUT(QFileInfo::exists(copy), SessionRecovery::kSaveDelayMs + 3000);

        // Removing every point removes the copy, so nothing stale is offered.
        points.clearAll();
        QTRY_VERIFY_WITH_TIMEOUT(!QFileInfo::exists(copy), SessionRecovery::kSaveDelayMs + 3000);
        session.markCleanExit();
    }

    void writesOnlyInsideTheRecoveryDirectory()
    {
        QTemporaryDir root;
        QTemporaryDir project;
        const QString segmentDir = QDir(project.path()).filePath("paths/w045");
        QVERIFY(QDir().mkpath(segmentDir));
        const QStringList before = filesUnder(project.path());

        VCCollection points;
        auto session = std::make_unique<SessionRecovery>(root.path());
        QVERIFY(session->begin(&points));
        session->setProject(QDir(project.path()).filePath("scroll.volpkg.json"));
        session->setSegment("w045");
        fillPoints(points);
        session->flush();
        QVERIFY(session->waitForWrite());
        session.reset();

        SessionRecovery next(root.path());
        QVERIFY(next.begin(nullptr));
        const auto leftovers = next.leftoverSessions();
        QCOMPARE(leftovers.size(), std::size_t{1});
        VCCollection restored;
        QVERIFY(SessionRecovery::restorePoints(leftovers.front(), &restored));
        next.markCleanExit();

        QCOMPARE(filesUnder(project.path()), before);
        for (const QString& file : filesUnder(root.path())) {
            QVERIFY(file.startsWith(QDir(root.path()).absolutePath()));
        }
    }

    void prunesEmptyAndOldLeftovers()
    {
        QTemporaryDir root;
        const QDateTime now = QDateTime::currentDateTimeUtc();
        for (int i = 0; i < 7; ++i) {
            writeLeftover(root.path(), QStringLiteral("session-old-%1").arg(i), now.addSecs(-60 * (i + 1)), true);
        }
        // A project and a segment but no points: nothing to recover.
        writeLeftover(root.path(), "session-empty", now, false);

        SessionRecovery session(root.path());
        QVERIFY(session.begin(nullptr));
        const auto leftovers = session.leftoverSessions();
        QCOMPARE(leftovers.size(), std::size_t{SessionRecovery::kMaxLeftoverSessions});
        QVERIFY(leftovers.front().directory.endsWith("session-old-0"));
        QVERIFY(leftovers.back().directory.endsWith("session-old-4"));
        QVERIFY(!QFileInfo::exists(QDir(root.path()).filePath("session-empty")));
        QVERIFY(!QFileInfo::exists(QDir(root.path()).filePath("session-old-6")));
        session.markCleanExit();
    }

    void settingTurnsRecoveryOff()
    {
        QTemporaryDir config;
        ConfigDirOverride override(config.path());
        QVERIFY(SessionRecovery::enabledInSettings());
        {
            QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
            settings.setValue(vc3d::settings::backup::SESSION_RECOVERY, false);
        }
        QVERIFY(!SessionRecovery::enabledInSettings());
    }

    void flushWritesInTheBackground()
    {
#ifndef Q_OS_UNIX
        QSKIP("Needs a named pipe to hold the write open");
#else
        // A named pipe in place of the temporary copy holds the write open
        // until the test reads it, so this does not depend on timing.
        QTemporaryDir root;
        VCCollection points;
        SessionRecovery session(root.path());
        QVERIFY(session.begin(&points));
        QVERIFY(session.waitForWrite());
        session.setProject(kProject);
        QVERIFY(session.waitForWrite());
        fillPoints(points);
        const QString copy = QDir(session.sessionDirectory()).filePath("points.json");
        QCOMPARE(mkfifo(QFile::encodeName(copy + ".tmp").constData(), 0600), 0);

        session.flush();
        QVERIFY(session.isWriting());  // flush() returned while the write waits

        // Changes made meanwhile wait for the running write, and the record
        // still describes what is on disk.
        session.setProject(QStringLiteral("/data/other.volpkg.json"));
        points.addPoint("corrections", {7.0f, 8.0f, 9.0f});
        QCOMPARE(readRecord(session.sessionDirectory()).value("project").toString(), kProject);
        QCOMPARE(readRecord(session.sessionDirectory()).value("points").toInt(), 0);

        QFile pipe(copy + ".tmp");
        QVERIFY(pipe.open(QIODevice::ReadOnly));
        QVERIFY(!pipe.readAll().isEmpty());
        pipe.close();

        // The first write ends, then the newest state is written.
        QTRY_COMPARE_WITH_TIMEOUT(readRecord(session.sessionDirectory()).value("points").toInt(), 4, 10000);
        QVERIFY(session.waitForWrite());
        QCOMPARE(readRecord(session.sessionDirectory()).value("project").toString(),
                 QStringLiteral("/data/other.volpkg.json"));
        VCCollection restored;
        QVERIFY(restored.loadFromJSON(QFile::encodeName(copy).toStdString()));
        QVERIFY(samePoints(restored, points));
        session.markCleanExit();
#endif
    }

    void cleanExitWaitsForTheRunningWrite()
    {
        QTemporaryDir root;
        VCCollection points;
        SessionRecovery session(root.path());
        QVERIFY(session.begin(&points));
        for (int i = 0; i < 2000; ++i) {
            points.addPoint("corrections", {static_cast<float>(i), 1.0f, 2.0f});
        }
        session.flush();
        session.markCleanExit();
        QVERIFY(QDir(root.path()).entryList(QDir::Dirs | QDir::NoDotAndDotDot).isEmpty());
    }

    void failedWriteIsReportedOnceAndStopsRetrying()
    {
        QTemporaryDir root;
        VCCollection points;
        SessionRecovery session(root.path());
        QVERIFY(session.begin(&points));
        QVERIFY(session.waitForWrite());
        // A folder where the temporary copy goes makes every write fail.
        QVERIFY(QDir(session.sessionDirectory()).mkdir("points.json.tmp"));
        QSignalSpy failures(&session, &SessionRecovery::writeFailed);

        fillPoints(points);
        session.flush();
        QVERIFY(!session.waitForWrite());
        QCOMPARE(failures.count(), 1);

        points.addPoint("corrections", {4.0f, 5.0f, 6.0f});
        session.flush();
        QVERIFY(!session.isWriting());
        QVERIFY(!session.waitForWrite());
        QTest::qWait(SessionRecovery::kSaveDelayMs / 10);
        QCOMPARE(failures.count(), 1);
        session.markCleanExit();
    }

    void dialogRestoresIntoTheRecordedProjectThenSelectsTheSegment()
    {
        QTemporaryDir root;
        writeLeftover(root.path(), "session-crashed", QDateTime::currentDateTimeUtc(), true);
        SessionRecovery session(root.path());
        VCCollection live;
        QVERIFY(session.begin(&live));
        FakeWindow window{&live};
        window.recovery = &session;
        window.oldCopy = QDir(root.path()).filePath("session-crashed");

        QPointer<SessionRecoveryDialog> dialog =
            SessionRecoveryDialog::offer(nullptr, session, &live, window.actions());
        QVERIFY(dialog);
        QVERIFY(dialog->isModal());
        auto* segment = dialog->findChild<QCheckBox*>("selectSegmentCheck");
        QVERIFY(segment && segment->isChecked() && segment->text().contains(kSegment));
        // Nothing happens until Recover is pressed.
        QVERIFY(window.calls.isEmpty());
        QVERIFY(live.getAllCollections().empty());

        dialog->findChild<QPushButton*>("recoverSessionButton")->click();
        // The project opens first (clearing the collections), the points
        // follow, and the segment, whose correction points load with it, goes
        // last, so they neither block nor replace the restore. By the time the
        // segment loads, which can take long enough to crash in, the points
        // are in this session's copy, which names the segment, and only then
        // is the old copy gone.
        QCOMPARE(window.calls.size(), 3);
        QCOMPARE(window.calls.at(0), "open " + kProject);
        QCOMPARE(window.calls.at(1),
                 QStringLiteral("segment %1 with 1 points; copy 1 points for segment %1; old copy gone").arg(kSegment));
        QVERIFY(window.calls.at(2).contains("Recovered 1 point"));
        // The points are only in memory now; the message says how to keep them.
        QVERIFY(window.calls.at(2).contains("Save in the Point Collections dock"));
        QCOMPARE(live.getPoints("corrections").size(), std::size_t{1});
        QCOMPARE(live.getPoints("segment corrections").size(), std::size_t{1});
        QVERIFY(session.leftoverSessions().empty());
        QVERIFY(!QFileInfo::exists(QDir(root.path()).filePath("session-crashed")));
        session.markCleanExit();
    }

    void dialogWithoutASegmentStillReopensTheProject()
    {
        QTemporaryDir root;
        writeLeftover(root.path(), "session-crashed", QDateTime::currentDateTimeUtc(), true, kProject, QString());
        SessionRecovery session(root.path());
        VCCollection live;
        QVERIFY(session.begin(&live));
        FakeWindow window{&live};

        QPointer<SessionRecoveryDialog> dialog =
            SessionRecoveryDialog::offer(nullptr, session, &live, window.actions());
        QVERIFY(dialog);
        QVERIFY(dialog->findChild<QCheckBox*>("selectSegmentCheck") == nullptr);
        dialog->findChild<QPushButton*>("recoverSessionButton")->click();
        QCOMPARE(window.calls.size(), 2);
        QCOMPARE(window.calls.at(0), "open " + kProject);
        QCOMPARE(live.getPoints("corrections").size(), std::size_t{1});
        QVERIFY(session.leftoverSessions().empty());
        session.markCleanExit();
    }

    void dialogDoesNotReopenAProjectThatIsOpen()
    {
        QTemporaryDir root;
        writeLeftover(root.path(), "session-crashed", QDateTime::currentDateTimeUtc(), true);
        SessionRecovery session(root.path());
        VCCollection live;
        QVERIFY(session.begin(&live));
        FakeWindow window{&live, kProject};

        QPointer<SessionRecoveryDialog> dialog =
            SessionRecoveryDialog::offer(nullptr, session, &live, window.actions());
        QVERIFY(dialog);
        dialog->findChild<QCheckBox*>("selectSegmentCheck")->setChecked(false);
        dialog->findChild<QPushButton*>("recoverSessionButton")->click();
        QCOMPARE(window.calls.size(), 1);
        QVERIFY(window.calls.front().startsWith("status Recovered"));
        QCOMPARE(live.getPoints("corrections").size(), std::size_t{1});
        session.markCleanExit();
    }

    void dialogKeepsTheCopyWhenThePointsCannotGoBack()
    {
        QTemporaryDir root;
        writeLeftover(root.path(), "session-crashed", QDateTime::currentDateTimeUtc(), true);
        SessionRecovery session(root.path());
        VCCollection live;
        QVERIFY(session.begin(&live));

        // The project does not open: nothing goes into whatever is open now.
        FakeWindow failing{&live};
        failing.openSucceeds = false;
        QPointer<SessionRecoveryDialog> dialog =
            SessionRecoveryDialog::offer(nullptr, session, &live, failing.actions());
        QVERIFY(dialog);
        dialog->findChild<QPushButton*>("recoverSessionButton")->click();
        QVERIFY(live.getAllCollections().empty());
        QCOMPARE(failing.calls.size(), 2);
        QVERIFY(failing.calls.at(1).contains("not found"));
        QVERIFY(failing.calls.at(1).contains("kept"));
        QCOMPARE(session.leftoverSessions().size(), std::size_t{1});

        // The recorded project is open, but points are loaded already.
        FakeWindow busy{&live, kProject};
        live.addPoint("new work", {5.0f, 5.0f, 5.0f});
        dialog = SessionRecoveryDialog::offer(nullptr, session, &live, busy.actions());
        QVERIFY(dialog);
        dialog->findChild<QPushButton*>("recoverSessionButton")->click();
        QCOMPARE(live.getAllCollections().size(), std::size_t{1});
        QCOMPARE(busy.calls.size(), 1);
        QVERIFY(busy.calls.front().contains("kept"));
        QCOMPARE(session.leftoverSessions().size(), std::size_t{1});
        session.markCleanExit();
    }

    void dialogKeepsTheOldCopyWhenTheNewOneCannotBeWritten()
    {
        QTemporaryDir root;
        writeLeftover(root.path(), "session-crashed", QDateTime::currentDateTimeUtc(), true);
        SessionRecovery session(root.path());
        VCCollection live;
        QVERIFY(session.begin(&live));
        QVERIFY(session.waitForWrite());
        // A folder where the temporary copy goes makes the new copy fail.
        QVERIFY(QDir(session.sessionDirectory()).mkdir("points.json.tmp"));
        FakeWindow window{&live};

        QPointer<SessionRecoveryDialog> dialog =
            SessionRecoveryDialog::offer(nullptr, session, &live, window.actions());
        QVERIFY(dialog);
        dialog->findChild<QPushButton*>("recoverSessionButton")->click();
        // The points are back and the segment is selected, but the old copy
        // stays, since nothing else holds the points on disk.
        QCOMPARE(live.getPoints("corrections").size(), std::size_t{1});
        QCOMPARE(window.calls.size(), 3);
        QVERIFY(window.calls.at(1).startsWith("segment"));
        QVERIFY(window.calls.at(2).contains("kept"));
        QVERIFY(QFileInfo::exists(QDir(root.path()).filePath("session-crashed/points.json")));
        QCOMPARE(session.leftoverSessions().size(), std::size_t{1});
        session.markCleanExit();
    }

    void copyLeavesOutTemporaryPreviews()
    {
        QTemporaryDir root;
        VCCollection points;
        auto session = std::make_unique<SessionRecovery>(root.path());
        QVERIFY(session->begin(&points));
        QVERIFY(session->waitForWrite());
        const QString copy = QDir(session->sessionDirectory()).filePath("points.json");
        const QString updated = readRecord(session->sessionDirectory()).value("updated").toString();
        QVERIFY(!updated.isEmpty());

        // The seeding panel's previews leave nothing to recover, and changing
        // them does not even schedule a copy.
        for (const QString& name : SessionRecovery::temporaryCollections()) {
            points.addPoints(name.toStdString(), {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
        }
        QTest::qWait(SessionRecovery::kSaveDelayMs + 500);
        QCOMPARE(readRecord(session->sessionDirectory()).value("updated").toString(), updated);
        session->flush();
        QVERIFY(session->waitForWrite());
        QVERIFY(!QFileInfo::exists(copy));
        QCOMPARE(readRecord(session->sessionDirectory()).value("points").toInt(), 0);

        points.addPoint("my points", {1.0f, 2.0f, 3.0f});
        session->flush();
        QVERIFY(session->waitForWrite());
        session.reset();

        SessionRecovery next(root.path());
        const auto leftovers = next.leftoverSessions();
        QCOMPARE(leftovers.size(), std::size_t{1});
        QCOMPARE(leftovers.front().pointCount, 1);
        VCCollection restored;
        QVERIFY(SessionRecovery::restorePoints(leftovers.front(), &restored));
        QCOMPARE(restored.getAllCollections().size(), std::size_t{1});
        QCOMPARE(restored.getPoints("my points").size(), std::size_t{1});
    }

    void folderAnotherVC3DIsCreatingIsLeftAlone()
    {
        // begin() makes its folder before it takes the lock. A second VC3D
        // starting at that moment must not prune the folder, or the first one
        // runs without recovery.
        QTemporaryDir root;
        const QString starting = QDir(root.path()).filePath("session-starting");
        QVERIFY(QDir().mkpath(starting));
        // The same, but from a start long ago that never got further.
        const QString stale = QDir(root.path()).filePath("session-stale");
        QVERIFY(QDir().mkpath(stale));
        std::error_code error;
        std::filesystem::last_write_time(QFile::encodeName(stale).toStdString(),
                                         std::filesystem::file_time_type::clock::now() - std::chrono::minutes(5),
                                         error);
        QVERIFY(!error);

        SessionRecovery session(root.path());
        QVERIFY(session.begin(nullptr));
        QVERIFY(session.leftoverSessions().empty());
        QVERIFY(QFileInfo::exists(starting));
        QVERIFY(!QFileInfo::exists(QDir(starting).filePath("session.lock")));
        QVERIFY(!QFileInfo::exists(stale));
        session.markCleanExit();

        // The starting VC3D can still take its lock.
        SessionRecovery::LeftoverSession other;
        other.directory = starting;
        QVERIFY(SessionRecovery::claim(other));
    }

    void pruningAndDiscardNeverFollowALink()
    {
#ifndef Q_OS_UNIX
        QSKIP("Needs a symbolic link to a folder");
#else
        QTemporaryDir root;
        QTemporaryDir elsewhere;
        QFile file(QDir(elsewhere.path()).filePath("keep.txt"));
        QVERIFY(file.open(QIODevice::WriteOnly));
        file.write("data");
        file.close();
        // Named like a leftover with nothing to recover, which pruning deletes.
        const QString link = QDir(root.path()).filePath("session-link");
        QVERIFY(QFile::link(elsewhere.path(), link));

        SessionRecovery session(root.path());
        QVERIFY(session.begin(nullptr));
        QVERIFY(session.leftoverSessions().empty());
        SessionRecovery::LeftoverSession linked;
        linked.directory = link;
        QVERIFY(!SessionRecovery::discard(linked));
        QCOMPARE(QDir(elsewhere.path()).entryList(QDir::Files | QDir::Hidden), QStringList{"keep.txt"});
        session.markCleanExit();
#endif
    }

    void dialogLaterKeepsAndDiscardDeletes()
    {
        QTemporaryDir root;
        writeLeftover(root.path(), "session-crashed", QDateTime::currentDateTimeUtc(), true);
        SessionRecovery session(root.path());
        QVERIFY(session.begin(nullptr));

        QPointer<SessionRecoveryDialog> later = SessionRecoveryDialog::offer(nullptr, session, nullptr, {});
        QVERIFY(later);
        later->findChild<QPushButton*>("laterSessionButton")->click();
        QCOMPARE(session.leftoverSessions().size(), std::size_t{1});

        QPointer<SessionRecoveryDialog> discard = SessionRecoveryDialog::offer(nullptr, session, nullptr, {});
        QVERIFY(discard);
        discard->findChild<QPushButton*>("discardSessionButton")->click();
        QVERIFY(session.leftoverSessions().empty());
        QVERIFY(SessionRecoveryDialog::offer(nullptr, session, nullptr, {}) == nullptr);
        session.markCleanExit();
    }
};

int main(int argc, char** argv)
{
    if (qEnvironmentVariableIsSet(kChildEnv)) {
        return runCrashingChild(argc, argv);
    }
    QApplication app(argc, argv);
    SessionRecoveryTest test;
    return QTest::qExec(&test, argc, argv);
}

#include "test_session_recovery.moc"
