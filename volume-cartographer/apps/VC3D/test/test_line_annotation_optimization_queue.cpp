// Coverage for apps/VC3D/LineAnnotationOptimizationQueue.hpp - the line
// annotation solve lifecycle as a pure decision object. Asserted here:
// single-flight, epoch-refused publication after a session mutation, dirty
// span coalescing and dedup, full-line stickiness, index remapping across
// control-point renumbering (insert, collapse, lost tracking), exactly-once
// pending dispatch, and shutdown.

#include <QtTest/QtTest>

#include "LineAnnotationOptimizationQueue.hpp"

using vc3d::line_annotation::OptimizationCoalescingQueue;
using State = OptimizationCoalescingQueue::State;

class TestLineAnnotationOptimizationQueue : public QObject
{
    Q_OBJECT

private slots:
    void beginSolveIsSingleFlight()
    {
        OptimizationCoalescingQueue queue;
        std::uint64_t epoch = 99;
        QVERIFY(queue.beginSolve(epoch));
        QCOMPARE(queue.state(), State::Running);
        QCOMPARE(epoch, queue.epoch());
        std::uint64_t second = 0;
        QVERIFY(!queue.beginSolve(second));
    }

    void mutationRefusesInFlightPublication()
    {
        OptimizationCoalescingQueue queue;
        std::uint64_t solveEpoch = 0;
        QVERIFY(queue.beginSolve(solveEpoch));
        QVERIFY(queue.mayPublish(solveEpoch));
        queue.noteSessionMutated();
        QVERIFY(!queue.mayPublish(solveEpoch));
        // The next solve starts in the new epoch and publishes fine.
        (void)queue.finishSolve();
        std::uint64_t next = 0;
        QVERIFY(queue.beginSolve(next));
        QVERIFY(queue.mayPublish(next));
    }

    void pendingUnionsAndDedupsDirtySpans()
    {
        OptimizationCoalescingQueue queue;
        queue.addPending({1, 2}, false);
        queue.addPending({2, 4}, false);
        const auto pending = queue.takePending();
        QVERIFY(pending.requested);
        QVERIFY(!pending.fullLine);
        QCOMPARE(pending.dirtySegments,
                 (std::vector<std::size_t>{1, 2, 4}));
        // Exactly once: the slot is consumed.
        QVERIFY(!queue.hasPending());
        QVERIFY(!queue.takePending().requested);
    }

    void fullLineWinsAndSticks()
    {
        OptimizationCoalescingQueue queue;
        queue.addPending({3}, false);
        queue.addPending({}, true);
        queue.addPending({5}, false);
        const auto pending = queue.takePending();
        QVERIFY(pending.requested);
        QVERIFY(pending.fullLine);
        QVERIFY(pending.dirtySegments.empty());
    }

    void remapCarriesSpansAcrossAnInsertion()
    {
        // 4 controls (spans 0..2); pending span 2. A control inserted at new
        // index 1 shifts controls 1..3 to 2..4: old span 2 (controls 2->3)
        // becomes new span 3 (controls 3->4).
        OptimizationCoalescingQueue queue;
        queue.addPending({2}, false);
        queue.remapPendingDirty({0, 2, 3, 4}, 4);
        const auto pending = queue.takePending();
        QCOMPARE(pending.dirtySegments, (std::vector<std::size_t>{3}));
    }

    void remapSplitsASpanTheInsertLandedIn()
    {
        // Pending span 0 (controls 0->1); a control inserted between them
        // maps control 1 to 2, so old span 0 covers new spans 0 and 1.
        OptimizationCoalescingQueue queue;
        queue.addPending({0}, false);
        queue.remapPendingDirty({0, 2}, 2);
        const auto pending = queue.takePending();
        QCOMPARE(pending.dirtySegments, (std::vector<std::size_t>{0, 1}));
    }

    void remapDropsASpanWhoseEndpointsCollapsed()
    {
        // Controls 1 and 2 collapsed into replacement index 1: old span 1
        // (controls 1->2) vanished; the edit dirties the replacement's own
        // spans itself, so nothing survives the remap and the request clears.
        OptimizationCoalescingQueue queue;
        queue.addPending({1}, false);
        queue.remapPendingDirty({0, 1, 1, 2}, 2);
        QVERIFY(!queue.hasPending());
    }

    void remapFallsBackToFullLineWhenTrackingIsLost()
    {
        // Pending span 3 but the map only covers controls 0..3 (span 3's
        // right endpoint is unmapped): dirty tracking is lost, so the
        // pending request escalates to a full pass instead of silently
        // dropping the span.
        OptimizationCoalescingQueue queue;
        queue.addPending({3}, false);
        queue.remapPendingDirty({0, 1, 2, 3}, 3);
        const auto pending = queue.takePending();
        QVERIFY(pending.requested);
        QVERIFY(pending.fullLine);
    }

    void finishSolveSurrendersPendingExactlyOnce()
    {
        OptimizationCoalescingQueue queue;
        std::uint64_t epoch = 0;
        QVERIFY(queue.beginSolve(epoch));
        queue.noteSessionMutated();
        queue.addPending({0}, false);
        const auto pending = queue.finishSolve();
        QCOMPARE(queue.state(), State::Idle);
        QVERIFY(pending.requested);
        QCOMPARE(pending.dirtySegments, (std::vector<std::size_t>{0}));
        QVERIFY(!queue.hasPending());
        QVERIFY(!queue.finishSolve().requested);
    }

    void remapDirtySpansHelperMirrorsPendingSemantics()
    {
        // The static helper is what the controller uses to carry the
        // IN-FLIGHT solve's dirty set across a renumbering, so a superseded
        // solve can fold its spans back into the pending union on discard.
        std::vector<std::size_t> spans{2};
        QVERIFY(OptimizationCoalescingQueue::remapDirtySpans(
            spans, {0, 2, 3, 4}, 4));
        QCOMPARE(spans, (std::vector<std::size_t>{3}));

        spans = {0};
        QVERIFY(OptimizationCoalescingQueue::remapDirtySpans(spans, {0, 2}, 2));
        QCOMPARE(spans, (std::vector<std::size_t>{0, 1}));

        spans = {1};
        QVERIFY(OptimizationCoalescingQueue::remapDirtySpans(
            spans, {0, 1, 1, 2}, 2));
        QVERIFY(spans.empty());

        spans = {3};
        QVERIFY(!OptimizationCoalescingQueue::remapDirtySpans(
            spans, {0, 1, 2, 3}, 3));
    }

    void shutdownRefusesEverything()
    {
        OptimizationCoalescingQueue queue;
        std::uint64_t solveEpoch = 0;
        QVERIFY(queue.beginSolve(solveEpoch));
        queue.addPending({1}, false);
        queue.shutdown();
        QVERIFY(!queue.mayPublish(solveEpoch));
        QVERIFY(!queue.hasPending());
        QVERIFY(!queue.finishSolve().requested);
        std::uint64_t next = 0;
        QVERIFY(!queue.beginSolve(next));
        queue.addPending({2}, false);
        QVERIFY(!queue.hasPending());
        QCOMPARE(queue.state(), State::ShuttingDown);
    }
};

QTEST_APPLESS_MAIN(TestLineAnnotationOptimizationQueue)
#include "test_line_annotation_optimization_queue.moc"
