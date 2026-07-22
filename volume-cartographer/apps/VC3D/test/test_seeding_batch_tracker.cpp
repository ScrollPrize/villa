// Unit tests for SeedingBatchTracker — the honest run/expand batch outcome
// aggregation extracted from SeedingWidget (SPEC §1).
//
// Deterministic and process-free: the tracker takes an opaque per-child key
// (a const void*, the QProcess identity in production) plus pre-extracted
// outcome fields, so we exercise every terminal shape (clean exit, nonzero
// exit, crash, failed-to-start, cancel, duplicate terminal) without spawning
// any child. No .volpkg data required.
//
// SCOPE NOTE: the "a child that fails to start does not HANG the batch"
// guarantee has two halves. This test covers the tracker LOGIC half — a
// failed-to-start terminal still advances completion so isComplete()/finalize()
// fire (see failedToStartFinalizesFailed). The signal-WIRING half — routing
// QProcess::errorOccurred(FailedToStart) into the completion path inside
// SeedingWidget::launchBatchProcess — needs a real QProcess and is covered by
// the offscreen/manual path, not this unit test.

#include <QtTest/QtTest>

#include "SeedingBatchTracker.hpp"

// Distinct opaque keys standing in for QProcess* identities.
static const void* keyFor(int i)
{
    static char keySlots[64];
    return &keySlots[i];
}

class TestSeedingBatchTracker : public QObject {
    Q_OBJECT

private slots:
    // Every child exits 0 -> success, "finished: N/N".
    void allCleanSucceeds()
    {
        SeedingBatchTracker t;
        t.begin(QStringLiteral("run"), 3);
        for (int i = 0; i < 3; ++i) {
            QVERIFY(t.recordTerminal(keyFor(i), false, false, 0, QString()));
        }
        QVERIFY(t.isComplete());
        const auto r = t.finalize();
        QVERIFY(r.success);
        QVERIFY(!r.canceled);
        QCOMPARE(r.completed, 3);
        QCOMPARE(r.total, 3);
        QCOMPARE(r.message, QStringLiteral("Seeding run finished: 3/3"));
    }

    // One nonzero child -> failed, failures == 1, message names the count.
    void oneNonzeroFails()
    {
        SeedingBatchTracker t;
        t.begin(QStringLiteral("run"), 3);
        QVERIFY(t.recordTerminal(keyFor(0), false, false, 0, QString()));
        QVERIFY(t.recordTerminal(keyFor(1), false, false, 7, QString()));
        QVERIFY(t.recordTerminal(keyFor(2), false, false, 0, QString()));
        QVERIFY(t.isComplete());
        QCOMPARE(t.failures(), 1);
        const auto r = t.finalize();
        QVERIFY(!r.success);
        QVERIFY(!r.canceled);
        QCOMPARE(r.completed, 3);
        QVERIFY(r.message.startsWith(QStringLiteral("Seeding run failed: 1 of 3 child processes failed")));
    }

    // Every child nonzero -> failed, failures == N.
    void allNonzeroFails()
    {
        SeedingBatchTracker t;
        t.begin(QStringLiteral("expand"), 4);
        for (int i = 0; i < 4; ++i) {
            QVERIFY(t.recordTerminal(keyFor(i), false, false, 1, QString()));
        }
        QCOMPARE(t.failures(), 4);
        const auto r = t.finalize();
        QVERIFY(!r.success);
        QCOMPARE(r.completed, 4);
        QVERIFY(r.message.startsWith(QStringLiteral("Seeding expand failed: 4 of 4 child processes failed")));
    }

    // A failed-to-start child still advances completion (proves no hang) and
    // finalizes failed.
    void failedToStartFinalizesFailed()
    {
        SeedingBatchTracker t;
        t.begin(QStringLiteral("run"), 2);
        QVERIFY(t.recordTerminal(keyFor(0), false, false, 0, QString()));
        QVERIFY(!t.isComplete());
        // failedToStart == true, exitCode irrelevant.
        QVERIFY(t.recordTerminal(keyFor(1), true, true, -1, QString()));
        QVERIFY(t.isComplete());  // advanced to total despite the failed start
        QCOMPARE(t.failures(), 1);
        const auto r = t.finalize();
        QVERIFY(!r.success);
        QVERIFY(!r.canceled);
        QCOMPARE(r.completed, 2);
    }

    // A crashed (CrashExit) child -> failed.
    void crashedFails()
    {
        SeedingBatchTracker t;
        t.begin(QStringLiteral("run"), 2);
        QVERIFY(t.recordTerminal(keyFor(0), false, false, 0, QString()));
        QVERIFY(t.recordTerminal(keyFor(1), false, /*crashed=*/true, 0, QString()));
        QCOMPARE(t.failures(), 1);
        const auto r = t.finalize();
        QVERIFY(!r.success);
    }

    // Cancel mid-batch then remaining terminal -> canceled == true, wording,
    // and finalize() is idempotent (second call == first, no re-mutation).
    void cancelThenFinalizeIdempotent()
    {
        SeedingBatchTracker t;
        t.begin(QStringLiteral("run"), 4);
        QVERIFY(t.recordTerminal(keyFor(0), false, false, 0, QString()));
        t.requestCancel();
        QVERIFY(t.cancelRequested());
        // A child still terminating after the cancel is latched.
        QVERIFY(t.recordTerminal(keyFor(1), false, false, 0, QString()));

        const auto r1 = t.finalize();
        QVERIFY(!r1.success);
        QVERIFY(r1.canceled);
        QCOMPARE(r1.completed, 2);
        QCOMPARE(r1.total, 4);
        QCOMPARE(r1.message, QStringLiteral("Seeding run canceled after 2/4"));

        QVERIFY(t.finalized());
        const auto r2 = t.finalize();  // idempotent
        QCOMPARE(r2.success, r1.success);
        QCOMPARE(r2.canceled, r1.canceled);
        QCOMPARE(r2.completed, r1.completed);
        QCOMPARE(r2.total, r1.total);
        QCOMPARE(r2.message, r1.message);
    }

    // Dedup: recording the same key twice counts once (completion does not
    // double-advance). finished() and errorOccurred() firing for one child.
    void duplicateTerminalCountsOnce()
    {
        SeedingBatchTracker t;
        t.begin(QStringLiteral("run"), 2);
        QVERIFY(t.recordTerminal(keyFor(0), false, false, 0, QString()));
        QCOMPARE(t.completed(), 1);
        // Same key again -> no-op, returns false.
        QVERIFY(!t.recordTerminal(keyFor(0), false, false, 5, QString()));
        QCOMPARE(t.completed(), 1);
        QCOMPARE(t.failures(), 0);
        QVERIFY(!t.isComplete());
        // The genuine second child completes the batch.
        QVERIFY(t.recordTerminal(keyFor(1), false, false, 0, QString()));
        QVERIFY(t.isComplete());
        const auto r = t.finalize();
        QVERIFY(r.success);
        QCOMPARE(r.completed, 2);
    }

    // Mixed batch: some zero, some nonzero, one failed-to-start ->
    // completed == total and the failure count is exact.
    void mixedBatchCounts()
    {
        SeedingBatchTracker t;
        t.begin(QStringLiteral("run"), 5);
        QVERIFY(t.recordTerminal(keyFor(0), false, false, 0, QString()));   // ok
        QVERIFY(t.recordTerminal(keyFor(1), false, false, 3, QString()));   // nonzero
        QVERIFY(t.recordTerminal(keyFor(2), false, false, 0, QString()));   // ok
        QVERIFY(t.recordTerminal(keyFor(3), true, true, -1, QString()));    // failed to start
        QVERIFY(t.recordTerminal(keyFor(4), false, true, 0, QString()));    // crashed
        QVERIFY(t.isComplete());
        QCOMPARE(t.completed(), 5);
        QCOMPARE(t.failures(), 3);
        const auto r = t.finalize();
        QVERIFY(!r.success);
        QVERIFY(!r.canceled);
        QCOMPARE(r.completed, 5);
        QCOMPARE(r.total, 5);
        QVERIFY(r.message.startsWith(QStringLiteral("Seeding run failed: 3 of 5 child processes failed")));
    }

    // The optional diagnostic label + output tail flow into the failure message.
    void failureDiagnosticCarriesLabelAndTail()
    {
        SeedingBatchTracker t;
        t.begin(QStringLiteral("run"), 1);
        QVERIFY(t.recordTerminal(keyFor(0), false, false, 9,
                                 QStringLiteral("  boom on line 42  "),
                                 QStringLiteral("Segmentation for point 0")));
        const auto r = t.finalize();
        QVERIFY(r.message.contains(QStringLiteral("Segmentation for point 0: exit code 9")));
        QVERIFY(r.message.contains(QStringLiteral("[boom on line 42]")));  // trimmed tail
    }
};

QTEST_APPLESS_MAIN(TestSeedingBatchTracker)
#include "test_seeding_batch_tracker.moc"
