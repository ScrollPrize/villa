#include <QtTest/QtTest>

#include <filesystem>
#include <future>
#include <memory>
#include <vector>

#include <QTemporaryDir>

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

        batch.finish();
        QCOMPARE(calls, 1);
        batch.finish();
        QCOMPARE(calls, 1);
    }

    void combinesPublicationErrors()
    {
        int calls = 0;
        QString error;
        FiberSaveBatchTracker batch([&](bool success, const QString& detail) {
            ++calls;
            QVERIFY(!success);
            error = detail;
        });

        batch.addError(QStringLiteral("validation failed"));
        batch.addError(QStringLiteral("write failed"));
        QCOMPARE(calls, 0);
        batch.finish();
        QCOMPARE(calls, 1);
        QCOMPARE(error, QStringLiteral("validation failed\nwrite failed"));
    }


};

QTEST_APPLESS_MAIN(TestFiberSaveBatchTracker)
#include "test_fiber_save_batch_tracker.moc"
