#include <QtTest/QtTest>

#include "LineAnnotationAdjacentLinks.hpp"

#include <array>
#include <cstdint>
#include <filesystem>

namespace {

struct Branch {
    int controlPointIndex = 0;
    uint64_t branchFiberId = 2;
    int branchControlPointIndex = 1;
    std::string branchFileName = "b.json";
    std::array<double, 3> controlPointPosition{12867.714596623457, 10041.881788022982,
                                               16661.818795840572};
    std::array<double, 3> branchControlPointPosition{12871.478552834642, 10018.210630149675,
                                                     16662.399929928943};
    std::array<double, 3> controlPointDirection{0.9963752036835241, 0.058142649700040565,
                                                0.06209577900690697};
    std::array<double, 3> branchControlPointDirection{-0.024412605581528416,
                                                      0.13769928147742536,
                                                      -0.9901731831196608};
    bool pending = true;
    bool adjacent = true;
};

struct Fiber {
    uint64_t id = 1;
    std::string fileName = "a.json";
    std::filesystem::path sourceRoot = "/fibers";
    std::vector<Branch> branches;
    bool adjacentBranchesPresent = true;
    bool needsSave = false;
    bool adjacentHealed = false;
};

bool matches(const Fiber& fiber, const Branch& branch, const Branch& candidate)
{
    const auto exact = [](const auto& a, const auto& b) { return a == b; };
    return vc3d::line_annotation::reciprocalBranchRefMatches(
        fiber, branch, candidate, exact, exact);
}

void heal(std::vector<Fiber>& fibers)
{
    vc3d::line_annotation::restoreMissingAdjacentBranchRefs(
        fibers,
        [](const Fiber& fiber, const std::string& name) {
            return (fiber.sourceRoot / name).lexically_normal().string();
        },
        matches);
}

std::vector<Fiber> legacyPair()
{
    Fiber a;
    a.branches.push_back(Branch{});
    Fiber b;
    b.id = 2;
    b.fileName = "b.json";
    b.adjacentBranchesPresent = false;
    return {a, b};
}

} // namespace

class TestLineAnnotationAdjacentLinks : public QObject
{
    Q_OBJECT

private slots:
    void missingArrayRestoresExactReciprocal()
    {
        auto fibers = legacyPair();
        heal(fibers);
        QCOMPARE(fibers[1].branches.size(), std::size_t{1});
        const auto& restored = fibers[1].branches.front();
        QVERIFY(matches(fibers[0], fibers[0].branches.front(), restored));
        QVERIFY(restored.adjacent);
        QVERIFY(restored.pending);
        QCOMPARE(restored.branchFiberId, fibers[0].id);
        QVERIFY(fibers[1].needsSave && fibers[1].adjacentHealed);
        QVERIFY(!fibers[0].needsSave);
        heal(fibers);
        QCOMPARE(fibers[1].branches.size(), std::size_t{1});
    }

    void presentEmptyArrayIsUntouched()
    {
        auto fibers = legacyPair();
        fibers[1].adjacentBranchesPresent = true;
        heal(fibers);
        QVERIFY(fibers[1].branches.empty());
        QVERIFY(!fibers[1].needsSave && !fibers[1].adjacentHealed);
    }

    void reciprocalMustHaveTheSameKind()
    {
        auto fibers = legacyPair();
        heal(fibers);
        auto ordinary = fibers[1].branches.front();
        ordinary.adjacent = false;
        QVERIFY(!matches(fibers[0], fibers[0].branches.front(), ordinary));
        fibers[1].branches = {ordinary};
        fibers[1].needsSave = false;
        fibers[1].adjacentBranchesPresent = true;
        heal(fibers);
        QCOMPARE(fibers[1].branches.size(), std::size_t{1});
        QVERIFY(!fibers[1].branches.front().adjacent);
        QVERIFY(!fibers[1].needsSave);
    }

    void severalPeersRestoreIntoOneLegacyFile()
    {
        auto fibers = legacyPair();
        Fiber c = fibers[0];
        c.id = 3;
        c.fileName = "c.json";
        fibers.push_back(c);
        heal(fibers);
        QCOMPARE(fibers[1].branches.size(), std::size_t{2});
        QVERIFY(matches(fibers[0], fibers[0].branches.front(), fibers[1].branches[0]));
        QVERIFY(matches(fibers[2], fibers[2].branches.front(), fibers[1].branches[1]));
    }

    void sourceRootsAreIsolated()
    {
        auto fibers = legacyPair();
        Fiber otherCopy = fibers[1];
        otherCopy.sourceRoot = "/another-copy";
        fibers.push_back(otherCopy);
        heal(fibers);
        QCOMPARE(fibers[1].branches.size(), std::size_t{1});
        QVERIFY(fibers[2].branches.empty());
        QVERIFY(!fibers[2].needsSave);
    }

    void missingPeerAndOrdinaryLinksCannotHeal()
    {
        auto fibers = legacyPair();
        fibers[0].branches.front().adjacent = false;
        heal(fibers);
        QVERIFY(fibers[1].branches.empty());
        fibers[0].branches.front().adjacent = true;
        fibers[0].branches.front().branchFileName = "missing.json";
        heal(fibers);
        QVERIFY(fibers[1].branches.empty());
    }

    void selfLinkDoesNotInvalidateTraversal()
    {
        auto fibers = legacyPair();
        fibers.resize(1);
        fibers[0].adjacentBranchesPresent = false;
        fibers[0].branches.front().branchFileName = "a.json";
        heal(fibers);
        QCOMPARE(fibers[0].branches.size(), std::size_t{2});
        QVERIFY(matches(fibers[0], fibers[0].branches[0], fibers[0].branches[1]));
    }
};

QTEST_APPLESS_MAIN(TestLineAnnotationAdjacentLinks)
#include "test_line_annotation_adjacent_links.moc"
