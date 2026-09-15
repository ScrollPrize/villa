// Coverage for LineAnnotationFiberDeletion.hpp: deleting stored fibers across
// the save drain in LineAnnotationController::deleteFibers. The drain runs a
// nested event loop in which the list can change under the delete, so the file
// name captured before the wait - not the id - decides what is deleted
// afterwards, and a package change during the wait deletes nothing. The
// fixture's reload deliberately renumbers every fiber from 1: that is the
// regression the first review comment found (runtime ids are stable across
// in-package reloads since), and the delete must stay correct even under it.

#include <QtTest/QtTest>

#include <cstdint>
#include <string>
#include <vector>

#include "LineAnnotationFiberDeletion.hpp"

using vc3d::line_annotation::FiberDeleteCapture;
using vc3d::line_annotation::FiberDeletePackageIdentity;
using vc3d::line_annotation::FiberDeleteResolution;
using vc3d::line_annotation::captureFiberDeleteTargets;
using vc3d::line_annotation::resolveFiberDeleteTargets;
using vc3d::line_annotation::resolveFiberDeletionAcrossWait;
using vc3d::line_annotation::sessionBelongsToDeletedFiber;
using vc3d::line_annotation::branchRefersToDeletedFiber;
using vc3d::line_annotation::sameFiberIdentity;
using vc3d::line_annotation::FiberDeleteOutcome;
using vc3d::line_annotation::FiberDeleteTarget;

namespace
{

struct Fiber {
    uint64_t id = 0;
    std::string fileName;
};

// A reload as the loader did it before runtime ids were made stable within
// a package: the list rebuilt from disk in its own order and the ids handed
// out again from 1. The delete has to stay correct even under that.
std::vector<Fiber> reloaded(const std::vector<std::string>& fileNames)
{
    std::vector<Fiber> fibers;
    uint64_t id = 1;
    for (const std::string& name : fileNames) {
        fibers.push_back(Fiber{id++, name});
    }
    return fibers;
}

const FiberDeletePackageIdentity kPackage{7, "/vol/fibers/proj"};

} // namespace

class TestLineAnnotationFiberDeletion : public QObject
{
    Q_OBJECT

private slots:
    void captureKeepsFileNamesAndReportsTheRest()
    {
        const std::vector<Fiber> fibers = {{1, "a.json"}, {2, "b.json"}, {3, ""}};
        const FiberDeleteCapture capture = captureFiberDeleteTargets({2, 3, 9}, fibers);
        QCOMPARE(capture.targets.size(), std::size_t{1});
        QCOMPARE(capture.targets[0].requestedId, uint64_t{2});
        QCOMPARE(capture.targets[0].fileName, std::string("b.json"));
        QCOMPARE(capture.unnamed, std::vector<uint64_t>{3});
        QCOMPARE(capture.notLoaded, std::vector<uint64_t>{9});
    }

    // The scenario of the review comment: the wait reloads the list, a new
    // fiber sorts first, every id moves by one. The requested id 2 named
    // b.json; after the reload id 2 is a.json and b.json is id 3.
    void reloadDuringTheWaitResolvesByFileName()
    {
        std::vector<Fiber> fibers = reloaded({"a.json", "b.json", "c.json"});
        int waits = 0;
        const FiberDeleteResolution resolution = resolveFiberDeletionAcrossWait(
            std::vector<uint64_t>{2},
            [&fibers]() -> const std::vector<Fiber>& { return fibers; },
            []() { return kPackage; },
            [&fibers, &waits]() {
                ++waits;
                fibers = reloaded({"0-new.json", "a.json", "b.json", "c.json"});
            });
        QCOMPARE(waits, 1);
        QVERIFY(!resolution.aborted);
        QCOMPARE(resolution.resolvedIds, std::vector<uint64_t>{3});
        QVERIFY(resolution.missing.empty());
        QCOMPARE(fibers[1].id, uint64_t{2});
        QCOMPARE(fibers[1].fileName, std::string("a.json"));
    }

    // Ids can come back in a different order than they were requested; the
    // caller binary-searches the result, so it is sorted.
    void resolvedIdsAreSortedAfterAnOrderReversal()
    {
        std::vector<Fiber> fibers = reloaded({"z.json", "a.json"});
        const FiberDeleteResolution resolution = resolveFiberDeletionAcrossWait(
            std::vector<uint64_t>{1, 2},
            [&fibers]() -> const std::vector<Fiber>& { return fibers; },
            []() { return kPackage; },
            [&fibers]() { fibers = reloaded({"a.json", "z.json"}); });
        QVERIFY(!resolution.aborted);
        QCOMPARE(resolution.resolvedIds, (std::vector<uint64_t>{1, 2}));
    }

    void fileVanishedDuringTheWaitIsMissingNotDeleted()
    {
        std::vector<Fiber> fibers = reloaded({"a.json", "b.json"});
        const FiberDeleteResolution resolution = resolveFiberDeletionAcrossWait(
            std::vector<uint64_t>{2},
            [&fibers]() -> const std::vector<Fiber>& { return fibers; },
            []() { return kPackage; },
            [&fibers]() { fibers = reloaded({"a.json"}); });
        QVERIFY(!resolution.aborted);
        QVERIFY(resolution.resolvedIds.empty());
        QCOMPARE(resolution.missing.size(), std::size_t{1});
        QCOMPARE(resolution.missing[0].fileName, std::string("b.json"));
    }

    // A project switch bumps the package generation; the same file name in
    // the new project is a different fiber.
    void packageGenerationChangeAbortsEverything()
    {
        std::vector<Fiber> fibers = reloaded({"a.json", "b.json"});
        FiberDeletePackageIdentity identity = kPackage;
        const FiberDeleteResolution resolution = resolveFiberDeletionAcrossWait(
            std::vector<uint64_t>{1, 2},
            [&fibers]() -> const std::vector<Fiber>& { return fibers; },
            [&identity]() { return identity; },
            [&identity]() { identity.packageGeneration += 1; });
        QVERIFY(resolution.aborted);
        QVERIFY(resolution.resolvedIds.empty());
        QVERIFY(resolution.missing.empty());
    }

    void fibersDirectoryChangeAbortsEverything()
    {
        const std::vector<Fiber> fibers = reloaded({"a.json"});
        const FiberDeleteCapture capture = captureFiberDeleteTargets({1}, fibers);
        FiberDeletePackageIdentity after = kPackage;
        after.fibersDir = "/vol/fibers/other";
        const FiberDeleteResolution resolution =
            resolveFiberDeleteTargets(capture, fibers, kPackage, after);
        QVERIFY(resolution.aborted);
        QVERIFY(resolution.resolvedIds.empty());
    }

    // No fibers directory means no package to delete from, whichever side of
    // the wait it is missing on.
    void emptyFibersDirectoryAborts()
    {
        const std::vector<Fiber> fibers = reloaded({"a.json"});
        const FiberDeleteCapture capture = captureFiberDeleteTargets({1}, fibers);
        FiberDeletePackageIdentity none = kPackage;
        none.fibersDir.clear();
        QVERIFY(resolveFiberDeleteTargets(capture, fibers, none, kPackage).aborted);
        QVERIFY(resolveFiberDeleteTargets(capture, fibers, kPackage, none).aborted);
        QVERIFY(!resolveFiberDeleteTargets(capture, fibers, kPackage, kPackage).aborted);
    }

    void unchangedListResolvesTheSameIds()
    {
        std::vector<Fiber> fibers = reloaded({"a.json", "b.json", "c.json"});
        const FiberDeleteResolution resolution = resolveFiberDeletionAcrossWait(
            std::vector<uint64_t>{3, 1},
            [&fibers]() -> const std::vector<Fiber>& { return fibers; },
            []() { return kPackage; },
            []() {});
        QVERIFY(!resolution.aborted);
        QCOMPARE(resolution.resolvedIds, (std::vector<uint64_t>{1, 3}));
    }

    // Two loaded fibers under one file name cannot be told apart; neither is
    // deleted.
    void duplicateFileNamesAreAmbiguous()
    {
        const std::vector<Fiber> before = reloaded({"a.json", "b.json"});
        const FiberDeleteCapture capture = captureFiberDeleteTargets({2}, before);
        const std::vector<Fiber> after = {{1, "a.json"}, {2, "b.json"}, {3, "b.json"}};
        const FiberDeleteResolution resolution =
            resolveFiberDeleteTargets(capture, after, kPackage, kPackage);
        QVERIFY(!resolution.aborted);
        QVERIFY(resolution.resolvedIds.empty());
        QCOMPARE(resolution.ambiguous.size(), std::size_t{1});
    }

    void mixedOutcomeIsReportedPerTarget()
    {
        std::vector<Fiber> fibers = reloaded({"a.json", "b.json", "c.json", ""});
        const FiberDeleteResolution resolution = resolveFiberDeletionAcrossWait(
            std::vector<uint64_t>{1, 2, 4, 8},
            [&fibers]() -> const std::vector<Fiber>& { return fibers; },
            []() { return kPackage; },
            // b.json is gone, a.json moved to id 2, c.json is new id 1.
            [&fibers]() { fibers = reloaded({"c.json", "a.json"}); });
        QVERIFY(!resolution.aborted);
        QCOMPARE(resolution.resolvedIds, std::vector<uint64_t>{2});
        QCOMPARE(resolution.missing.size(), std::size_t{1});
        QCOMPARE(resolution.missing[0].fileName, std::string("b.json"));
        QCOMPARE(resolution.unnamed, std::vector<uint64_t>{4});
        QCOMPARE(resolution.notLoaded, std::vector<uint64_t>{8});
    }

    // The same id requested twice still deletes one fiber once.
    void duplicateRequestedIdsResolveOnce()
    {
        std::vector<Fiber> fibers = reloaded({"a.json", "b.json", "c.json"});
        const FiberDeleteResolution resolution = resolveFiberDeletionAcrossWait(
            std::vector<uint64_t>{2, 2},
            [&fibers]() -> const std::vector<Fiber>& { return fibers; },
            []() { return kPackage; },
            [&fibers]() { fibers = reloaded({"0-new.json", "a.json", "b.json", "c.json"}); });
        QVERIFY(!resolution.aborted);
        QCOMPARE(resolution.resolvedIds, std::vector<uint64_t>{3});
    }

    // Save suppression after the delete: a named session is matched by its
    // file name only, so the session of an unrelated fiber that happens to
    // hold a deleted fiber's NEW id after a reload keeps saving; a session
    // without a name falls back to its id.
    void sessionSuppressionMatchesNamedSessionsByFileNameOnly()
    {
        // Initially a=1, b=2, c=3 with c's session open (id 3). b is deleted
        // after a reload made it new=1, a=2, b=3, c=4: deletedIds = {3}.
        const std::vector<uint64_t> deletedIds = {3};
        const std::vector<std::string> deletedNames = {"b.json"};
        QVERIFY(!sessionBelongsToDeletedFiber(3, "c.json", deletedIds, deletedNames));
        QVERIFY(sessionBelongsToDeletedFiber(2, "b.json", deletedIds, deletedNames));
        QVERIFY(sessionBelongsToDeletedFiber(3, "", deletedIds, deletedNames));
        QVERIFY(!sessionBelongsToDeletedFiber(4, "", deletedIds, deletedNames));
        QVERIFY(!sessionBelongsToDeletedFiber(3, "a.json", deletedIds, deletedNames));
    }

    // Link cleanup after the delete: a branch ref that names its target is
    // matched by that name only, so an open session's link to c.json is not
    // torn out just because a reload gave the deleted b.json the id the
    // link still records for c.
    void branchCleanupMatchesNamedReferencesByFileNameOnly()
    {
        // Initially a=1, b=2, c=3; A's session links to C as (3, "c.json").
        // After the reload new=1, a=2, b=3, c=4, and b (now id 3) is deleted.
        QVERIFY(!branchRefersToDeletedFiber(3, "c.json", 3, "b.json"));
        QVERIFY(branchRefersToDeletedFiber(2, "b.json", 3, "b.json"));
        // A ref without a name has only its id.
        QVERIFY(branchRefersToDeletedFiber(3, "", 3, "b.json"));
        QVERIFY(!branchRefersToDeletedFiber(4, "", 3, "b.json"));
        // A deleted fiber without a name matches by id as well.
        QVERIFY(branchRefersToDeletedFiber(3, "c.json", 3, ""));
        QVERIFY(!branchRefersToDeletedFiber(3, "c.json", 0, ""));
    }

    // The scheduler upserts a session's snapshot into the stored list; after
    // a reload the session's id can be another stored fiber's, so a named
    // snapshot lands on the stored fiber with its name, never on the id.
    void storedFiberIdentityPrefersFileNames()
    {
        // A's session (opened as id 1, "a.json") after a reload that made
        // new=1, a=2: the snapshot must match stored (2, "a.json"), not (1, "new.json").
        QVERIFY(!sameFiberIdentity(1, "a.json", 1, "new.json"));
        QVERIFY(sameFiberIdentity(1, "a.json", 2, "a.json"));
        QVERIFY(sameFiberIdentity(5, "", 5, "x.json"));
        QVERIFY(!sameFiberIdentity(5, "", 6, "x.json"));
        QVERIFY(!sameFiberIdentity(0, "", 0, ""));
    }

    // A caller reporting per requested id (the agent bridge) reads the
    // outcome through the captured names, never through ids that a reload
    // may have handed to other fibers.
    void outcomeAnswersPerRequestedIdByCapturedName()
    {
        FiberDeleteOutcome outcome;
        outcome.requested = {FiberDeleteTarget{2, "b.json"}, FiberDeleteTarget{3, "c.json"}};
        outcome.deletedFileNames = {"b.json"};
        QVERIFY(outcome.deletedRequested(2));
        QVERIFY(!outcome.deletedRequested(3));
        QVERIFY(!outcome.deletedRequested(9));
        // The resolution carries the capture for the caller to build this.
        std::vector<Fiber> fibers = reloaded({"a.json", "b.json"});
        const FiberDeleteResolution resolution = resolveFiberDeletionAcrossWait(
            std::vector<uint64_t>{2},
            [&fibers]() -> const std::vector<Fiber>& { return fibers; },
            []() { return kPackage; },
            []() {});
        QCOMPARE(resolution.targets.size(), std::size_t{1});
        QCOMPARE(resolution.targets[0].requestedId, uint64_t{2});
        QCOMPARE(resolution.targets[0].fileName, std::string("b.json"));
    }

    // Nothing to delete: the wait is skipped, and the report still names
    // what was asked for.
    void noTargetsSkipsTheWait()
    {
        std::vector<Fiber> fibers = reloaded({"a.json"});
        int waits = 0;
        const FiberDeleteResolution resolution = resolveFiberDeletionAcrossWait(
            std::vector<uint64_t>{5},
            [&fibers]() -> const std::vector<Fiber>& { return fibers; },
            []() { return kPackage; },
            [&waits]() { ++waits; });
        QCOMPARE(waits, 0);
        QVERIFY(!resolution.aborted);
        QVERIFY(resolution.resolvedIds.empty());
        QCOMPARE(resolution.notLoaded, std::vector<uint64_t>{5});
    }
};

QTEST_APPLESS_MAIN(TestLineAnnotationFiberDeletion)

#include "test_line_annotation_fiber_deletion.moc"
