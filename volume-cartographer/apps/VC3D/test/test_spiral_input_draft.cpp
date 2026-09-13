#include "SpiralInputDraft.hpp"

#include <QtTest/QtTest>

using namespace vc3d::spiral;

class SpiralInputDraftTest : public QObject
{
    Q_OBJECT

private slots:
    void oldAcknowledgementPreservesNewerEdit()
    {
        InputDraft draft("collection", {{{"points", 2}}, false}, 1);
        draft.edit({{{"points", 3}}, false});
        const auto submitted = draft.snapshot();
        draft.edit({{{"points", 4}}, false});
        draft.acknowledgeAccepted(submitted, 2);
        draft.acknowledgeApplied(2);
        draft.acknowledgePersisted(2);
        QVERIFY(draft.dirty());
        QCOMPARE(draft.snapshot().content.manifest.value("points").toInt(), 4);
        QCOMPARE(submitted.content.manifest.value("points").toInt(), 3);
        QCOMPARE(draft.accepted(), quint64(2));
    }

    void invalidDraftRemainsDirtyAndBlocksWholeSelection()
    {
        InputDraft valid("patch", {{{"geometry", "first"}}, false});
        InputDraft invalid("pcl", {{{"points", 2}}, false}, 1);
        invalid.edit({{{"points", 1}}, false}, "Needs two points");
        InputDraftSubmission coordinator;
        const auto refused = coordinator.begin("one", {&valid, &invalid});
        QVERIFY(!refused.batch);
        QVERIFY(!coordinator.active());
        QVERIFY(invalid.dirty());
        QCOMPARE(refused.errors.value("pcl"), QString("Needs two points"));
        // Excluding the bad row submits the good one and retains the bad draft.
        const auto accepted = coordinator.begin("two", {&valid});
        QCOMPARE(accepted.batch->entries.size(), qsizetype(1));
        QVERIFY(invalid.dirty());
    }

    void deletionAndRestoreRemainDraftsAfterApplication()
    {
        InputDraft draft("fiber", {{{"points", 2}}, false}, 1);
        draft.edit({{{"points", 3}}, false});
        draft.remove();
        const auto deleted = draft.snapshot();
        draft.acknowledgeAccepted(deleted, 2);
        draft.acknowledgeApplied(2);
        QVERIFY(draft.deleted());
        QVERIFY(draft.canRestore());
        QVERIFY(draft.needsCommit());
        draft.restore();
        QVERIFY(!draft.deleted());
        QVERIFY(draft.dirty());
        QCOMPARE(draft.snapshot().content.manifest.value("points").toInt(), 3);
    }

    void restoreRetainsInvalidity()
    {
        InputDraft draft("fiber", {}, 1);
        draft.edit({}, "Empty fiber");
        draft.remove();
        QVERIFY(draft.valid());
        draft.restore();
        QVERIFY(!draft.valid());
        QVERIFY(draft.dirty());
    }

    void committedDeletionHasNoRestoreAction()
    {
        InputDraft draft("fiber", {{{"points", 2}}, false}, 1);
        draft.remove();
        draft.acknowledgeAccepted(draft.snapshot(), 2);
        draft.acknowledgeApplied(2);
        draft.acknowledgePersisted(2);
        QVERIFY(!draft.canRestore());
        draft.restore();
        QVERIFY(draft.deleted());
    }

    void retryAndRepeatedClicksAttachToCapturedBatch()
    {
        InputDraft draft("patch", {{{"geometry", "first"}}, false});
        InputDraftSubmission coordinator;
        const auto first = coordinator.begin("one", {&draft});
        draft.edit({{{"geometry", "second"}}, false});
        coordinator.transportInterrupted();
        const auto retry = coordinator.begin("two", {&draft});
        QCOMPARE(retry.batch->commandId, QString("one"));
        QVERIFY(retry.batch->outcomeUnknown);
        QCOMPARE(retry.batch->entries.front().content.manifest.value("geometry").toString(),
                 QString("first"));
        coordinator.reconciled("another-command");
        QVERIFY(coordinator.active());
        draft.acknowledgeAccepted(first.batch->entries.front(), 1);
        coordinator.reconciled("one");
        QVERIFY(!coordinator.active());
        QVERIFY(draft.dirty());
        QCOMPARE(coordinator.begin("three", {&draft}).batch->entries.front().expectedAccepted,
                 quint64(1));
    }

    void discardLocalKeepsAcceptedButUncommittedRevision()
    {
        InputDraft draft("pcl", {{{"points", 2}}, false}, 1);
        draft.edit({{{"points", 3}}, false});
        const auto submitted = draft.snapshot();
        draft.acknowledgeAccepted(submitted, 2);
        draft.acknowledgeApplied(2);
        draft.edit({{{"points", 4}}, false});
        draft.discardLocalChanges();
        QVERIFY(!draft.dirty());
        QVERIFY(draft.needsCommit());
        QCOMPARE(draft.snapshot().content.manifest.value("points").toInt(), 3);
    }
};

QTEST_APPLESS_MAIN(SpiralInputDraftTest)
#include "test_spiral_input_draft.moc"
