#include "ProjectStoreCounts.hpp"

#include <QtTest/QtTest>
#include <QScopeGuard>
#include <QSemaphore>
#include <QThreadPool>
#include <QTimer>

class ProjectStoreCountsTest : public QObject
{
    Q_OBJECT
private slots:
    void slowReadKeepsEventLoopResponsiveAndCoalescesRefreshes()
    {
        const ProjectStoreCounts::Identity id{"store", 1};
        int updates = 0;
        ProjectStoreCounts counts(nullptr, [&] { ++updates; });
        QSemaphore entered, resume;
        const auto unblock = qScopeGuard([&] { resume.release(); QThreadPool::globalInstance()->waitForDone(); });
        auto* guiThread = QThread::currentThread();
        std::atomic_bool workerThread{false};
        counts.request(id, [&] {
            workerThread = QThread::currentThread() != guiThread;
            entered.release();
            resume.acquire();
            return ProjectStoreCounts::Counts{{"surface", 1}};
        });
        QVERIFY(entered.tryAcquire(1, 5000));
        bool heartbeat = false;
        QTimer::singleShot(0, this, [&] { heartbeat = true; });
        QTRY_VERIFY(heartbeat);
        QVERIFY(workerThread);
        QCOMPARE(updates, 0);
        QVERIFY(counts.snapshot(id)->values.empty());
        std::atomic_int superseded{0};
        counts.request(id, [&] { ++superseded; return ProjectStoreCounts::Counts{{"surface", 2}}; });
        counts.request(id, [] { return ProjectStoreCounts::Counts{{"surface", 3}}; });
        resume.release();
        QTRY_COMPARE(updates, 1);
        QCOMPARE(superseded.load(), 0);
        QCOMPARE(counts.snapshot(id)->values.at("surface"), 3);
    }

    void failureRetainsLastCountsAndRecoveryClearsError()
    {
        const ProjectStoreCounts::Identity id{"store", 1};
        int updates = 0;
        ProjectStoreCounts counts(nullptr, [&] { ++updates; });
        counts.request(id, [] { return ProjectStoreCounts::Counts{{"fiber", 7}}; });
        QTRY_COMPARE(updates, 1);
        counts.request(id, []() -> ProjectStoreCounts::Counts { throw std::runtime_error("offline"); });
        QCOMPARE(counts.snapshot(id)->values.at("fiber"), 7);
        QTRY_COMPARE(updates, 2);
        QCOMPARE(counts.snapshot(id)->values.at("fiber"), 7);
        QCOMPARE(counts.snapshot(id)->error, QString("offline"));
        counts.request(id, [] { return ProjectStoreCounts::Counts{{"fiber", 8}}; });
        QTRY_COMPARE(updates, 3);
        QCOMPARE(counts.snapshot(id)->values.at("fiber"), 8);
        QVERIFY(counts.snapshot(id)->error.isEmpty());
    }

    void detachedResultsAndDestroyedOwnersAreIgnored()
    {
        const ProjectStoreCounts::Identity oldId{"store", 1}, newId{"store", 2};
        int updates = 0;
        auto counts = std::make_unique<ProjectStoreCounts>(nullptr, [&] { ++updates; });
        QSemaphore entered, resume;
        const auto unblock = qScopeGuard([&] { resume.release(); QThreadPool::globalInstance()->waitForDone(); });
        auto slow = [&] {
            entered.release(); resume.acquire();
            return ProjectStoreCounts::Counts{{"surface", 1}};
        };
        counts->request(oldId, slow);
        QVERIFY(entered.tryAcquire(1, 5000));
        counts->retain({newId});
        counts->request(newId, [] { return ProjectStoreCounts::Counts{{"surface", 9}}; });
        resume.release();
        QThreadPool::globalInstance()->waitForDone();
        QTRY_COMPARE(updates, 1);
        QVERIFY(!counts->snapshot(oldId));
        QCOMPARE(counts->snapshot(newId)->values.at("surface"), 9);
        counts->request(newId, slow);
        QVERIFY(entered.tryAcquire(1, 5000));
        counts.reset(); // destruction must not wait for the blocked request
        resume.release();
        QThreadPool::globalInstance()->waitForDone();
        QCoreApplication::processEvents();
        QCOMPARE(updates, 1);
    }
};

QTEST_GUILESS_MAIN(ProjectStoreCountsTest)
#include "test_project_store_counts.moc"
