#include <QtTest/QtTest>

#include <filesystem>
#include <future>
#include <memory>
#include <vector>

#include <QTemporaryDir>

#include "FiberSaveBatchTracker.hpp"
#include "LineAnnotationFiberSaveJob.hpp"

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

    void waitersCompleteAfterAsynchronousDiskWrite()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const auto path = std::filesystem::path(dir.path().toStdString()) / "fiber.json";

        int calls = 0;
        auto makeWaiter = [&]() {
            auto waiter = std::make_shared<FiberSaveBatchTracker>(
                [&](bool success, const QString& error) {
                    QVERIFY(success);
                    QVERIFY(error.isEmpty());
                    QVERIFY(std::filesystem::exists(path));
                    ++calls;
                });
            waiter->addJob();
            waiter->finishScheduling();
            return waiter;
        };
        std::vector<std::shared_ptr<FiberSaveBatchTracker>> waiters{
            makeWaiter(), makeWaiter()};

        vc3d::line_annotation::FiberSavePayload payload;
        payload.fiberId = 7;
        payload.generation = 3;
        payload.path = path;
        payload.json = {{"type", "vc3d_fiber"}, {"generation", 3}};
        auto worker = std::async(
            std::launch::async,
            [payload = std::move(payload)]() mutable {
                return vc3d::line_annotation::runFiberSaveJob(
                    42, {std::move(payload)});
            });

        QCOMPARE(calls, 0);
        const auto result = worker.get();
        QVERIFY(result.ok);
        for (const auto& waiter : waiters) {
            waiter->finishJob();
        }
        QCOMPARE(calls, 2);
    }
};

QTEST_APPLESS_MAIN(TestFiberSaveBatchTracker)
#include "test_fiber_save_batch_tracker.moc"
