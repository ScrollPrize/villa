#include <QtTest/QtTest>

#include "FiberSaveBatchTracker.hpp"

class TestFiberSaveBatchTracker : public QObject {
    Q_OBJECT

private slots:
    void emptyBatchCompletes()
    {
        int calls = 0;
        FiberSaveBatchTracker batch([&](bool success, const QString& error) {
            ++calls;
            QVERIFY(success);
            QVERIFY(error.isEmpty());
        });

        batch.finishScheduling();
        QCOMPARE(calls, 1);
        batch.finishScheduling();
        QCOMPARE(calls, 1);
    }

    void waitsForEveryJob()
    {
        int calls = 0;
        bool success = true;
        QString error;
        FiberSaveBatchTracker batch([&](bool ok, const QString& detail) {
            ++calls;
            success = ok;
            error = detail;
        });

        batch.addJob();
        batch.addJob();
        batch.finishScheduling();
        batch.finishJob();
        QCOMPARE(calls, 0);
        batch.finishJob(QStringLiteral("write failed"));
        QCOMPARE(calls, 1);
        QVERIFY(!success);
        QCOMPARE(error, QStringLiteral("write failed"));
    }

    void combinesSchedulingAndWorkerErrors()
    {
        int calls = 0;
        QString error;
        FiberSaveBatchTracker batch([&](bool success, const QString& detail) {
            ++calls;
            QVERIFY(!success);
            error = detail;
        });

        batch.addError(QStringLiteral("validation failed"));
        batch.addJob();
        batch.finishJob(QStringLiteral("write failed"));
        QCOMPARE(calls, 0);
        batch.finishScheduling();
        QCOMPARE(calls, 1);
        QCOMPARE(error, QStringLiteral("validation failed\nwrite failed"));
    }
};

QTEST_APPLESS_MAIN(TestFiberSaveBatchTracker)
#include "test_fiber_save_batch_tracker.moc"
