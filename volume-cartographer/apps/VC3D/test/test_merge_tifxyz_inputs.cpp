#include <QtTest/QtTest>

#include "MergeTifxyzInputs.hpp"

#include <QDir>
#include <QFile>
#include <QTemporaryDir>

namespace mt = vc3d::merge_tifxyz;

class MergeTifxyzInputsTest : public QObject
{
    Q_OBJECT

private slots:
    void solidGridUsesPaddedStringRows();
    void gridWithHolesUsesArrayRows();
    void validateRejectsBadGrids();
    void writeMergeJsonPicksFreeOutputName();
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
    QCOMPARE(mt::alphaFirst({"", "zeta", "alpha", "beta"}), QStringLiteral("alpha"));
}

void MergeTifxyzInputsTest::validateRejectsBadGrids()
{
    const QStringList available{"a", "b", "c d"};
    QVERIFY(mt::validate({{"a", "b"}}, available, "/p", {}, 5.0, 10.0).isEmpty());
    QVERIFY(mt::validate({{"a", "b"}}, available, "/p", "b", 5.0, 10.0).isEmpty());
    QVERIFY(mt::validate({{"a", "a"}}, available, "/p", {}, 5.0, 10.0)
                .contains(QStringLiteral("at least 2")));
    QVERIFY(mt::validate({{"a", "x"}}, available, "/p", {}, 5.0, 10.0)
                .contains(QStringLiteral("x")));
    QVERIFY(mt::validate({{"a", "c d"}}, available, "/p", {}, 5.0, 10.0)
                .contains(QStringLiteral("whitespace")));
    QVERIFY(mt::validate({{"a", "b"}}, available, "/p", {}, 10.0, 10.0)
                .contains(QStringLiteral("RANSAC")));
    QVERIFY(mt::validate({{"a", "b"}}, available, "/p", "c d", 5.0, 10.0)
                .contains(QStringLiteral("Reference")));
}

void MergeTifxyzInputsTest::writeMergeJsonPicksFreeOutputName()
{
    QTemporaryDir root;
    QVERIFY(root.isValid());
    const mt::Grid rows{{"b", "a"}};

    QString error;
    const QString first = mt::writeMergeJson(root.path(), rows, &error);
    QCOMPARE(first, QDir(root.path()).filePath(QStringLiteral("a_merged.merge.json")));
    QFile written(first);
    QVERIFY(written.open(QIODevice::ReadOnly));
    QCOMPARE(QString::fromUtf8(written.readAll()), mt::gridJsonText(rows));

    // An existing output directory moves the name to the next version.
    QVERIFY(QDir(root.path()).mkpath(QStringLiteral("paths/a_merged")));
    const QString second = mt::writeMergeJson(root.path(), rows, &error);
    QCOMPARE(second, QDir(root.path()).filePath(QStringLiteral("a_merged_v2.merge.json")));

    QVERIFY(mt::writeMergeJson(QString(), rows, &error).isEmpty());
    QVERIFY(!error.isEmpty());
}

QTEST_GUILESS_MAIN(MergeTifxyzInputsTest)
#include "test_merge_tifxyz_inputs.moc"
