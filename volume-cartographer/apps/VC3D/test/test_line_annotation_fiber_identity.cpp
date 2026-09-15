// Coverage for LineAnnotationFiberIdentity.hpp: runtime fiber ids that stay
// stable across reloads of one package's fiber list, and the names-first
// identity rule used wherever a record carries both an id and a file name.

#include <QtTest/QtTest>

#include <cstdint>
#include <string>
#include <vector>

#include "LineAnnotationFiberIdentity.hpp"

using vc3d::line_annotation::RuntimeFiberIdSpace;
using vc3d::line_annotation::allocateRuntimeFiberId;
using vc3d::line_annotation::assignStableRuntimeIds;
using vc3d::line_annotation::sameFiberIdentity;
using vc3d::line_annotation::bindRuntimeFiberIdentity;
using vc3d::line_annotation::rebindRuntimeFiberIdentity;
using vc3d::line_annotation::retireRuntimeFiberIdentity;

namespace
{

struct Fiber {
    uint64_t id = 0;
    std::string fileName;
};

// A (re)load as the controller does it: the list is rebuilt from disk in its
// own sorted order, ids not yet assigned.
std::vector<Fiber> loaded(const std::vector<std::string>& fileNames)
{
    std::vector<Fiber> fibers;
    for (const std::string& name : fileNames) {
        fibers.push_back(Fiber{0, name});
    }
    return fibers;
}

std::vector<uint64_t> idsOf(const std::vector<Fiber>& fibers)
{
    std::vector<uint64_t> ids;
    for (const Fiber& fiber : fibers) {
        ids.push_back(fiber.id);
    }
    return ids;
}

} // namespace

class TestLineAnnotationFiberIdentity : public QObject
{
    Q_OBJECT

private slots:
    void firstLoadNumbersFromOne()
    {
        RuntimeFiberIdSpace space;
        std::vector<Fiber> fibers = loaded({"a.json", "b.json", "c.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{1, 2, 3}));
        QCOMPARE(space.nextId, uint64_t{4});
    }

    // The review comment's scenario: a=1, b=2, c=3 are loaded and an
    // annotation session is open on a (id 1). An import adds 0-new.json,
    // which sorts first. Before, the reload renumbered everything from 1 and
    // the session's id 1 then meant 0-new.json; now a, b, c keep 1, 2, 3 and
    // the new file gets 4.
    void reloadKeepsKnownIdsAndGivesNewFilesFreshOnes()
    {
        RuntimeFiberIdSpace space;
        std::vector<Fiber> fibers = loaded({"a.json", "b.json", "c.json"});
        assignStableRuntimeIds(fibers, space, {});
        const uint64_t openSessionId = fibers[0].id;

        fibers = loaded({"0-new.json", "a.json", "b.json", "c.json"});
        assignStableRuntimeIds(fibers, space, {openSessionId});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{4, 1, 2, 3}));
        QCOMPARE(fibers[1].fileName, std::string("a.json"));
        QCOMPARE(fibers[1].id, openSessionId);
    }

    void sortOrderDoesNotMatter()
    {
        RuntimeFiberIdSpace space;
        std::vector<Fiber> fibers = loaded({"z.json", "a.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{1, 2}));
        fibers = loaded({"a.json", "z.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{2, 1}));
    }

    // A fiber deleted through the app has its name retired: a later file
    // under that name is a new fiber with a fresh id, and the old id is never
    // handed out again, so anything still holding it can only miss.
    void deletedNamesAreRetiredAndIdsNeverRecycled()
    {
        RuntimeFiberIdSpace space;
        std::vector<Fiber> fibers = loaded({"a.json", "b.json", "c.json"});
        assignStableRuntimeIds(fibers, space, {});
        retireRuntimeFiberIdentity(space, "c.json");
        fibers = loaded({"a.json", "b.json", "d.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{1, 2, 4}));
        // An import under the retired name is a different fiber.
        fibers = loaded({"a.json", "b.json", "c.json", "d.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{1, 2, 5, 4}));
    }

    // A fiber the app creates (a saved session, a merge, a split half) is
    // bound when it is persisted, so the first reload keeps the id it was
    // allocated instead of treating the new file as unknown.
    void createdFibersKeepTheirAllocatedIdAcrossTheFirstReload()
    {
        RuntimeFiberIdSpace space;
        std::vector<Fiber> fibers = loaded({"a.json"});
        assignStableRuntimeIds(fibers, space, {});
        const uint64_t created = allocateRuntimeFiberId(space, {});
        QCOMPARE(created, uint64_t{2});
        bindRuntimeFiberIdentity(space, "created.json", created);
        fibers = loaded({"a.json", "created.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{1, 2}));
        // Binding never lowers the mark, and binding an unnamed or id-less
        // record is a no-op.
        bindRuntimeFiberIdentity(space, "", 9);
        bindRuntimeFiberIdentity(space, "x.json", 0);
        QCOMPARE(space.idByFileName.size(), std::size_t{2});
        QCOMPARE(allocateRuntimeFiberId(space, {}), uint64_t{3});
    }

    // Binding a name that another id holds moves the name: the new record
    // owns it from now on (the caller has established that the file under
    // that name is this fiber, e.g. an accepted save).
    void bindingMovesANameToItsNewOwner()
    {
        RuntimeFiberIdSpace space;
        std::vector<Fiber> fibers = loaded({"a.json"});
        assignStableRuntimeIds(fibers, space, {});
        bindRuntimeFiberIdentity(space, "a.json", 5);
        QCOMPARE(space.idByFileName.at("a.json"), uint64_t{5});
        QCOMPARE(allocateRuntimeFiberId(space, {}), uint64_t{6});
        fibers = loaded({"a.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(fibers[0].id, uint64_t{5});
    }

    // A rename keeps the fiber's id under the new name and frees the old
    // name: a later import under the old name is a different fiber.
    void renameMovesTheBindingWithTheId()
    {
        RuntimeFiberIdSpace space;
        std::vector<Fiber> fibers = loaded({"a.json", "b.json"});
        assignStableRuntimeIds(fibers, space, {});
        rebindRuntimeFiberIdentity(space, "a.json", "renamed.json", 1);
        fibers = loaded({"b.json", "renamed.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{2, 1}));
        fibers = loaded({"a.json", "b.json", "renamed.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{3, 2, 1}));
    }

    // Open sessions hold ids outside the stored list (a fiber created but not
    // yet saved, or whose file vanished); a new file must not collide with them.
    void reservedSessionIdsLiftTheMark()
    {
        RuntimeFiberIdSpace space;
        std::vector<Fiber> fibers = loaded({"a.json"});
        assignStableRuntimeIds(fibers, space, {});
        fibers = loaded({"a.json", "b.json"});
        assignStableRuntimeIds(fibers, space, {7});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{1, 8}));
    }

    void newSessionsAllocateAboveEverything()
    {
        RuntimeFiberIdSpace space;
        std::vector<Fiber> fibers = loaded({"a.json", "b.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(allocateRuntimeFiberId(space, {}), uint64_t{3});
        QCOMPARE(allocateRuntimeFiberId(space, {}), uint64_t{4});
        // A live id the space did not hand out (e.g. one chosen as
        // max(parent + 1)) still lifts the mark.
        QCOMPARE(allocateRuntimeFiberId(space, {10}), uint64_t{11});
        // And a later reload does not hand 3, 4 or 11 to a new file.
        fibers = loaded({"a.json", "b.json", "c.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(fibers[2].id, uint64_t{12});
    }

    // Two files that load to one file name cannot share an id; an empty file
    // name has no identity at all. Each such fiber gets a fresh id.
    void duplicateAndEmptyNamesGetFreshIds()
    {
        RuntimeFiberIdSpace space;
        std::vector<Fiber> fibers = loaded({"a.json", "a.json", ""});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{1, 2, 3}));
        QCOMPARE(space.idByFileName.at("a.json"), uint64_t{1});
        // Reloaded, the first a.json keeps 1 and the duplicate is fresh again.
        fibers = loaded({"a.json", "a.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{1, 4}));
    }

    // A package switch starts from an empty space: the same file name in a
    // different package is a different fiber and numbering restarts at 1.
    void freshSpaceRestartsNumbering()
    {
        RuntimeFiberIdSpace old;
        std::vector<Fiber> fibers = loaded({"a.json", "b.json"});
        assignStableRuntimeIds(fibers, old, {});
        RuntimeFiberIdSpace space;
        fibers = loaded({"b.json", "a.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{1, 2}));
    }

    void reassigningTheSameListChangesNothing()
    {
        RuntimeFiberIdSpace space;
        std::vector<Fiber> fibers = loaded({"a.json", "b.json", "c.json"});
        assignStableRuntimeIds(fibers, space, {});
        const std::vector<uint64_t> before = idsOf(fibers);
        const uint64_t mark = space.nextId;
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(idsOf(fibers), before);
        QCOMPARE(space.nextId, mark);
    }

    // The space belongs to the package, not to the list: a load that finds
    // nothing (a strict pass that removed everything, a repair retry that
    // started from an emptied list) must not forget the bindings, and a
    // fiber absent from one load returns with its id on the next.
    void spaceSurvivesAnEmptyLoadAndATemporaryAbsence()
    {
        RuntimeFiberIdSpace space;
        std::vector<Fiber> fibers = loaded({"a.json", "b.json", "c.json"});
        assignStableRuntimeIds(fibers, space, {});
        std::vector<Fiber> none;
        assignStableRuntimeIds(none, space, {});
        fibers = loaded({"a.json", "c.json"});   // b failed to load this time
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{1, 3}));
        fibers = loaded({"a.json", "b.json", "c.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(idsOf(fibers), (std::vector<uint64_t>{1, 2, 3}));
    }

    // Within one load the list may be assigned more than once (the loader's
    // own pass, then the strict fallback's): a name discovered in the first
    // pass is not given a second id by the next.
    void secondPassInOneLoadAssignsNothingNew()
    {
        RuntimeFiberIdSpace space;
        std::vector<Fiber> fibers = loaded({"a.json"});
        assignStableRuntimeIds(fibers, space, {});
        std::vector<Fiber> again = loaded({"a.json", "new.json"});
        assignStableRuntimeIds(again, space, {});
        QCOMPARE(idsOf(again), (std::vector<uint64_t>{1, 2}));
        std::vector<Fiber> strict = loaded({"a.json", "new.json"});
        assignStableRuntimeIds(strict, space, {});
        QCOMPARE(idsOf(strict), (std::vector<uint64_t>{1, 2}));
        QCOMPARE(space.nextId, uint64_t{3});
    }

    // A split makes two fibers; both ids come from the allocator, and
    // deleting both before the next allocation does not hand either out again.
    void splitHalvesAreBothReservedAndNeverRecycled()
    {
        RuntimeFiberIdSpace space;
        std::vector<Fiber> fibers = loaded({"parent.json"});
        assignStableRuntimeIds(fibers, space, {});
        const uint64_t parentId = fibers[0].id;
        const uint64_t prefix = allocateRuntimeFiberId(space, {parentId}, parentId + 1);
        const uint64_t suffix = allocateRuntimeFiberId(space, {parentId, prefix}, prefix + 1);
        QCOMPARE(prefix, uint64_t{2});
        QCOMPARE(suffix, uint64_t{3});
        // Both halves and the parent are gone; a new fiber is 4, not 2 or 3.
        QCOMPARE(allocateRuntimeFiberId(space, {}), uint64_t{4});
        // And a reload with a brand-new file also skips them.
        fibers = loaded({"other.json"});
        assignStableRuntimeIds(fibers, space, {});
        QCOMPARE(fibers[0].id, uint64_t{5});
    }

    // The names-first rule the synchronisers use. The comment's example: an
    // open session A still holds id 1 while, after a renumbering reload, a
    // link in fiber D recorded as (1, new.json) points at another fiber; the
    // link must not be taken for a reference to A.
    void namesDecideWhenBothAreKnown()
    {
        QVERIFY(!sameFiberIdentity(1, "a.json", 1, "new.json"));
        QVERIFY(sameFiberIdentity(1, "a.json", 2, "a.json"));
        QVERIFY(sameFiberIdentity(5, "", 5, "x.json"));
        QVERIFY(sameFiberIdentity(5, "x.json", 5, ""));
        QVERIFY(!sameFiberIdentity(5, "", 6, "x.json"));
        QVERIFY(!sameFiberIdentity(0, "", 0, ""));
        QVERIFY(!sameFiberIdentity(0, "", 0, "x.json"));
    }
};

QTEST_APPLESS_MAIN(TestLineAnnotationFiberIdentity)

#include "test_line_annotation_fiber_identity.moc"
