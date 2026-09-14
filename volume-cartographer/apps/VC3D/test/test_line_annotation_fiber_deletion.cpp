// Coverage for LineAnnotationFiberDeletion.hpp: deleting stored fibers across
// the save drain in LineAnnotationController::deleteFibers. The drain runs a
// nested event loop, and a reload during it reassigns runtime ids from 1, so
// the file name captured before the wait - not the id - decides what is
// deleted afterwards, and a package change during the wait deletes nothing.

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

namespace
{

struct Fiber {
    uint64_t id = 0;
    std::string fileName;
};

// A reload as loadFibersForCurrentPackage does it: the list is rebuilt from
// disk in its own order and the ids are handed out again from 1.
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
