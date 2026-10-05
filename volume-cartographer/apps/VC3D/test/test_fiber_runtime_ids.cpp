// Coverage for FiberRuntimeIds (apps/VC3D/FiberRuntimeIds.hpp): runtime ids
// keyed by (source root, file name), never reused for the controller's
// lifetime, following the fiber through rename and delete rather than
// staying with the name.

#include <QtTest/QtTest>

#include <filesystem>
#include <string>

#include "FiberRuntimeIds.hpp"

using vc3d::FiberRuntimeIds;

namespace
{
const std::filesystem::path kSource{"/pkg/fibers"};
const std::filesystem::path kOtherSource{"/spiral/paths/fibers"};
} // namespace

class TestFiberRuntimeIds : public QObject
{
    Q_OBJECT

private slots:
    // The reported scenario: load a.json (id 1), rename it to b.json (still
    // 1), import a different fiber as a.json. The newcomer must not be 1.
    void renameFreesTheOldName()
    {
        FiberRuntimeIds ids;
        const uint64_t a = ids.forFile(kSource, "a.json");
        QCOMPARE(a, uint64_t{1});
        ids.rename(kSource, "a.json", "b.json", a);
        QCOMPARE(ids.forFile(kSource, "b.json"), a);
        const uint64_t newcomer = ids.forFile(kSource, "a.json");
        QVERIFY(newcomer != a);
        QCOMPARE(newcomer, uint64_t{2});
        // And the renamed fiber still answers under its new name only.
        QCOMPARE(ids.forFile(kSource, "b.json"), a);
        QCOMPARE(ids.forFile(kSource, "a.json"), newcomer);
    }

    // A rename keeps the id across the reload that follows it (the loader
    // remembers what it finds on disk), and renaming to the same name is a
    // no-op.
    void renameKeepsTheIdThroughReload()
    {
        FiberRuntimeIds ids;
        const uint64_t a = ids.forFile(kSource, "a.json");
        ids.rename(kSource, "a.json", "b.json", a);
        ids.remember(kSource, "b.json", a);
        QCOMPARE(ids.forFile(kSource, "b.json"), a);
        ids.rename(kSource, "b.json", "b.json", a);
        QCOMPARE(ids.forFile(kSource, "b.json"), a);
    }

    // A deleted fiber's name is free for a different fiber, which gets a
    // fresh id; the deleted id is never handed out again.
    void forgetFreesTheNameAndRetiresTheId()
    {
        FiberRuntimeIds ids;
        const uint64_t a = ids.forFile(kSource, "a.json");
        const uint64_t b = ids.forFile(kSource, "b.json");
        ids.forget(kSource, "a.json");
        const uint64_t again = ids.forFile(kSource, "a.json");
        QVERIFY(again != a);
        QVERIFY(again > b);
        QCOMPARE(ids.forFile(kSource, "b.json"), b);
        QVERIFY(ids.allocate() > again);
    }

    // Bindings are per source: the same file name in another source is
    // another fiber, and a rename or delete in one source leaves the other
    // source's binding alone.
    void sourcesAreDistinct()
    {
        FiberRuntimeIds ids;
        const uint64_t here = ids.forFile(kSource, "a.json");
        const uint64_t there = ids.forFile(kOtherSource, "a.json");
        QVERIFY(here != there);
        ids.rename(kSource, "a.json", "b.json", here);
        QCOMPARE(ids.forFile(kOtherSource, "a.json"), there);
        ids.forget(kOtherSource, "a.json");
        QCOMPARE(ids.forFile(kSource, "b.json"), here);
        QVERIFY(ids.forFile(kOtherSource, "a.json") != there);
    }

    // The controller's reload pattern for a deleted fiber whose editor is
    // still open: the file is forgotten, the session's id is re-remembered
    // under an EMPTY name (reserved, not bound), a different fiber imported
    // under the old name gets a fresh id, and repeating the re-remember never
    // hands the old name back to the deleted id.
    void deletedSessionReservesItsIdWithoutItsName()
    {
        FiberRuntimeIds ids;
        const uint64_t deleted = ids.forFile(kSource, "a.json");
        ids.forget(kSource, "a.json");
        ids.remember(kSource, "", deleted);
        const uint64_t newcomer = ids.forFile(kSource, "a.json");
        QVERIFY(newcomer != deleted);
        ids.remember(kSource, "", deleted);
        ids.remember(kSource, "a.json", newcomer);
        QCOMPARE(ids.forFile(kSource, "a.json"), newcomer);
        QVERIFY(ids.allocate() > newcomer);
    }

    // Remembering an id only ever raises the next id; unknown names are
    // harmless to forget or rename from.
    void nextIdOnlyGrows()
    {
        FiberRuntimeIds ids;
        ids.remember(kSource, "x.json", 10);
        QCOMPARE(ids.allocate(), uint64_t{11});
        ids.remember(kSource, "y.json", 3);
        QCOMPARE(ids.allocate(), uint64_t{12});
        ids.forget(kSource, "never.json");
        ids.rename(kSource, "never.json", "z.json", 5);
        QCOMPARE(ids.forFile(kSource, "z.json"), uint64_t{5});
        QCOMPARE(ids.allocate(), uint64_t{13});
        // Empty names bind nothing.
        ids.remember(kSource, "", 40);
        QCOMPARE(ids.forFile(kSource, "w.json"), uint64_t{41});
    }
};

QTEST_APPLESS_MAIN(TestFiberRuntimeIds)
#include "test_fiber_runtime_ids.moc"
