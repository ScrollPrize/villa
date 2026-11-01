#include <QtTest/QtTest>

#include "MergeTifxyzInputs.hpp"

namespace mt = vc3d::merge_tifxyz;

class MergeTifxyzInputsTest : public QObject
{
    Q_OBJECT

private slots:
    void solidGridUsesPaddedStringRows();
    void gridWithHolesUsesArrayRows();
    void validateRejectsBadGrids();
    void distinctSurfacesKeepsFirstAppearance();
};

void MergeTifxyzInputsTest::solidGridUsesPaddedStringRows()
{
    const mt::Grid rows{{"a", "bbb"}, {"cc", "d"}};
    QCOMPARE(mt::gridJsonText(rows),
             QStringLiteral("{\n  \"rows\": [\n"
                            "    \"a  bbb\",\n"
                            "    \"cc d\"\n"
                            "  ]\n}\n"));
}

void MergeTifxyzInputsTest::gridWithHolesUsesArrayRows()
{
    const mt::Grid rows{{"a", ""}, {" b ", "c"}};
    QCOMPARE(mt::gridJsonText(rows),
             QStringLiteral("{\n  \"rows\": [\n"
                            "    [\"a\", null],\n"
                            "    [\"b\", \"c\"]\n"
                            "  ]\n}\n"));
}

void MergeTifxyzInputsTest::validateRejectsBadGrids()
{
    const QStringList available{"a", "b", "c d"};
    QVERIFY(mt::validate({{"a", "b"}}, available, {}, 5.0, 10.0).isEmpty());
    QVERIFY(mt::validate({{"a", "b"}}, available, "b", 5.0, 10.0).isEmpty());
    QVERIFY(mt::validate({{"a", "a"}}, available, {}, 5.0, 10.0)
                .contains(QStringLiteral("at least 2")));
    QVERIFY(mt::validate({{"a", "x"}}, available, {}, 5.0, 10.0)
                .contains(QStringLiteral("x")));
    QVERIFY(mt::validate({{"a", "c d"}}, available, {}, 5.0, 10.0)
                .contains(QStringLiteral("whitespace")));
    QVERIFY(mt::validate({{"a", "b"}}, available, {}, 10.0, 10.0)
                .contains(QStringLiteral("RANSAC")));
    QVERIFY(mt::validate({{"a", "b"}}, available, "c d", 5.0, 10.0)
                .contains(QStringLiteral("Reference")));
}

void MergeTifxyzInputsTest::distinctSurfacesKeepsFirstAppearance()
{
    const mt::Grid rows{{"b", "", "a"}, {" a ", "c", "b"}};
    QCOMPARE(mt::distinctSurfaces(rows), (QStringList{"b", "a", "c"}));
}

QTEST_GUILESS_MAIN(MergeTifxyzInputsTest)
#include "test_merge_tifxyz_inputs.moc"
