#include "FiberSourceDedupe.hpp"

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
