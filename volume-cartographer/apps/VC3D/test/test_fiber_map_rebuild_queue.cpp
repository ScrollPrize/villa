// Coverage for apps/VC3D/FiberMapRebuildQueue.hpp - the Fiber Map's
// asynchronous-rebuild lifecycle as a pure decision object. Every transition
// the workspace glue depends on is asserted here: single-flight, coalescing
// priority, pending survival across every terminal outcome, exactly-once
// dispatch, epoch invalidation, and shutdown.

#include <QtTest/QtTest>

#include "FiberMapRebuildQueue.hpp"

using vc3d::fiber_map::FiberMapRebuildQueue;
using Pending = FiberMapRebuildQueue::Pending;
using Request = FiberMapRebuildQueue::Request;
using State = FiberMapRebuildQueue::State;

class TestFiberMapRebuildQueue : public QObject
{
    Q_OBJECT

private slots:
    void idleRequestStartsExactlyOne()
    {
        FiberMapRebuildQueue queue;
        QCOMPARE(queue.request(false), Request::Start);
        QCOMPARE(queue.state(), State::Running);
        // A second request while running coalesces, never double-starts.
        QCOMPARE(queue.request(false), Request::Coalesced);
        QCOMPARE(queue.pending(), Pending::Update);
    }

    void fullWinsCoalescingInEitherOrder()
    {
        FiberMapRebuildQueue queue;
        (void)queue.request(false);
        QCOMPARE(queue.request(false), Request::Coalesced);
        QCOMPARE(queue.request(true), Request::Coalesced);
        QCOMPARE(queue.pending(), Pending::Full);
        // ...and an update arriving after a full must not downgrade it.
        QCOMPARE(queue.request(false), Request::Coalesced);
        QCOMPARE(queue.pending(), Pending::Full);
    }

    void pendingSurvivesEveryOutcomeAndDispatchesOnce()
    {
        // The queue cannot know the outcome (success, discard, error) - the
        // holder's epilogue is identical in all three - so one sequence
        // covers them: pending set while running must come back from
        // finishApply exactly once.
        FiberMapRebuildQueue queue;
        (void)queue.request(true);
        (void)queue.request(false);
        queue.beginApply();
        QCOMPARE(queue.state(), State::Applying);
        QCOMPARE(queue.finishApply(), Pending::Update);
        QCOMPARE(queue.state(), State::Idle);
        // Exactly once: the slot is consumed.
        QCOMPARE(queue.pending(), Pending::None);
        QCOMPARE(queue.finishApply(), Pending::None);
    }

    void requestsDuringApplyCoalesceToo()
    {
        FiberMapRebuildQueue queue;
        (void)queue.request(false);
        queue.beginApply();
        // Publication runs scene/tree code that can re-enter gates; a
        // request raised there must coalesce, not start concurrently.
        QCOMPARE(queue.request(true), Request::Coalesced);
        QCOMPARE(queue.finishApply(), Pending::Full);
    }

    void epochRefusesInvalidatedPublication()
    {
        FiberMapRebuildQueue queue;
        (void)queue.request(false);
        const std::uint64_t buildEpoch = queue.epoch();
        QVERIFY(queue.mayPublish(buildEpoch));
        queue.invalidate();  // package switch / explicit clear mid-flight
        QVERIFY(!queue.mayPublish(buildEpoch));
        // The next build starts in the new epoch and may publish.
        queue.beginApply();
        (void)queue.finishApply();
        QCOMPARE(queue.request(false), Request::Start);
        QVERIFY(queue.mayPublish(queue.epoch()));
    }

    void publicationRequiresRunningState()
    {
        FiberMapRebuildQueue queue;
        QVERIFY(!queue.mayPublish(queue.epoch()));  // Idle: nothing expected
        (void)queue.request(false);
        queue.beginApply();
        // Once applying, a second (stray) completion may not publish.
        QVERIFY(!queue.mayPublish(queue.epoch()));
    }

    void shutdownDropsPendingRefusesStartsAndInvalidates()
    {
        FiberMapRebuildQueue queue;
        (void)queue.request(false);
        (void)queue.request(true);
        const std::uint64_t buildEpoch = queue.epoch();
        queue.shutdown();
        QVERIFY(!queue.mayPublish(buildEpoch));
        QCOMPARE(queue.request(false), Request::Refused);
        queue.beginApply();
        QCOMPARE(queue.finishApply(), Pending::None);
        QCOMPARE(queue.state(), State::ShuttingDown);
    }
};

QTEST_APPLESS_MAIN(TestFiberMapRebuildQueue)
#include "test_fiber_map_rebuild_queue.moc"
