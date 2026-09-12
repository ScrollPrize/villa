#include "FiberSourceDedupe.hpp"
#include "FiberRuntimeIds.hpp"

#include <QtTest/QtTest>

namespace {

vc3d::FiberDedupeEntry entry(const char* source, const char* fileName,
                             const char* content, bool canonical)
{
    return {std::filesystem::path(source), fileName, content, canonical};
}

}  // namespace

class FiberSourceDedupeTest final : public QObject
{
    Q_OBJECT

private slots:
    void RuntimeIdsSurviveSourceChanges()
    {
        vc3d::FiberRuntimeIds ids;
        const auto existing = ids.forFile("/z/package", "fiber.json");
        const auto branch = ids.forFile("/z/package", "branch.json");
        const auto added = ids.forFile("/a/spiral", "fiber.json");
        QVERIFY(added != existing);
        QCOMPARE(ids.forFile("/z/package", "fiber.json"), existing);
        QCOMPARE(ids.forFile("/z/package", "branch.json"), branch);
        // Unregistering a source does not release its IDs. Re-registration
        // and strict reloads must recover the same identities.
        const auto later = ids.forFile("/b/spiral", "new.json");
        QVERIFY(later > added);
        QCOMPARE(ids.forFile("/a/spiral", "fiber.json"), added);
        ids.remember("/draft", "unsaved.json", 100);
        QVERIFY(ids.allocate() > 100);
        QCOMPARE(ids.forFile("/draft", "unsaved.json"), uint64_t{100});
    }

    void DistinctFibersAreAllKept()
    {
        const auto result = vc3d::dedupeFiberSources(
            {entry("/vpkg/fibers", "a.json", "A", true),
             entry("/vpkg/fibers", "b.json", "B", true),
             entry("/spiral/fibers", "c.json", "C", true)},
            {"/vpkg/fibers", "/spiral/fibers"});
        QCOMPARE(result.kept, (std::vector<std::size_t>{0, 1, 2}));
        QVERIFY(result.linkAliases.empty());
    }

    void SameFileNameAcrossSourcesKeepsThePrimarySource()
    {
        // The service committed a copy of the volpkg fiber into paths.fibers.
        const auto result = vc3d::dedupeFiberSources(
            {entry("/spiral/fibers", "sean_20260901T120000000_000003.json", "A", true),
             entry("/vpkg/fibers", "sean_20260901T120000000_000003.json", "A2", true)},
            {"/vpkg/fibers", "/spiral/fibers"});
        QCOMPARE(result.kept, (std::vector<std::size_t>{1}));
        QCOMPARE(result.linkAliases.size(), std::size_t{1});
        QCOMPARE(result.linkAliases.at(vc3d::fiberSourceFileKey(
                     "/spiral/fibers", "sean_20260901T120000000_000003.json")),
                 vc3d::fiberSourceFileKey(
                     "/vpkg/fibers", "sean_20260901T120000000_000003.json"));
    }

    void SameContentUnderNumericNameKeepsTheCanonicalName()
    {
        // A stale "<runtime id>.json" copy sorts before the original.
        const auto result = vc3d::dedupeFiberSources(
            {entry("/spiral/fibers", "12.json", "A", false),
             entry("/spiral/fibers", "sean_20260901T120000000_000003.json", "A", true)},
            {"/spiral/fibers"});
        QCOMPARE(result.kept, (std::vector<std::size_t>{1}));
        QCOMPARE(result.linkAliases.at(
                     vc3d::fiberSourceFileKey("/spiral/fibers", "12.json")),
                 vc3d::fiberSourceFileKey(
                     "/spiral/fibers", "sean_20260901T120000000_000003.json"));
    }

    void CanonicalNameOutranksSourcePreference()
    {
        const auto result = vc3d::dedupeFiberSources(
            {entry("/vpkg/fibers", "7.json", "A", false),
             entry("/spiral/fibers", "sean_20260901T120000000_000003.json", "A", true)},
            {"/vpkg/fibers", "/spiral/fibers"});
        QCOMPARE(result.kept, (std::vector<std::size_t>{1}));
    }

    void GroupsAreTransitiveAcrossNameAndContent()
    {
        const auto result = vc3d::dedupeFiberSources(
            {entry("/vpkg/fibers", "x.json", "A", true),
             entry("/spiral/fibers", "x.json", "B", true),
             entry("/spiral/fibers", "9.json", "B", false)},
            {"/vpkg/fibers", "/spiral/fibers"});
        QCOMPARE(result.kept, (std::vector<std::size_t>{0}));
        QCOMPARE(result.linkAliases.size(), std::size_t{2});
    }

    void EmptyContentKeyNeverMergesByContent()
    {
        const auto result = vc3d::dedupeFiberSources(
            {entry("/vpkg/fibers", "a.json", "", true),
             entry("/vpkg/fibers", "b.json", "", true)},
            {"/vpkg/fibers"});
        QCOMPARE(result.kept, (std::vector<std::size_t>{0, 1}));
    }

    void UnlistedSourceRanksLast()
    {
        const auto result = vc3d::dedupeFiberSources(
            {entry("/elsewhere", "a.json", "A", true),
             entry("/spiral/fibers", "a.json", "A", true)},
            {"/spiral/fibers"});
        QCOMPARE(result.kept, (std::vector<std::size_t>{1}));
    }
};

QTEST_GUILESS_MAIN(FiberSourceDedupeTest)
#include "test_fiber_source_dedupe.moc"
