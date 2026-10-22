#include <QtTest/QtTest>

#include "ProjectLocations.hpp"

#include <QTemporaryDir>

#include <filesystem>

namespace fs = std::filesystem;
namespace pl = vc3d::project_locations;

class ProjectLocationsTest : public QObject
{
    Q_OBJECT

private slots:
    void remoteSpellingsAreNeverLocal();
    void localSpellingsResolveAgainstTheProject();
    void findEntryMatchesAnySpellingOfAStoredEntry();
};

void ProjectLocationsTest::remoteSpellingsAreNeverLocal()
{
    QVERIFY(pl::looksRemote(QStringLiteral("s3://bucket/grid")));
    QVERIFY(pl::looksRemote(QStringLiteral("  HTTPS://host/m.lasagna.json")));
    QVERIFY(pl::looksRemote(QStringLiteral("S3://bucket/grid")));
    QVERIFY(!pl::looksRemote(QStringLiteral("file:///data/grid")));
    QVERIFY(!pl::looksRemote(QStringLiteral("/data/grid")));
    QVERIFY(!pl::looksRemote(QStringLiteral("grids/g1")));
}

void ProjectLocationsTest::localSpellingsResolveAgainstTheProject()
{
    const fs::path project("/data/project");
    QCOMPARE(pl::storedLocalLocation(QStringLiteral("grids/g1"), project),
             QStringLiteral("/data/project/grids/g1"));
    QCOMPARE(pl::storedLocalLocation(QStringLiteral("/data/grids/g1/"), project),
             QStringLiteral("/data/grids/g1"));
    QCOMPARE(pl::storedLocalLocation(QStringLiteral(" /data/x/../grids//g1 "), project),
             QStringLiteral("/data/grids/g1"));
    QCOMPARE(pl::storedLocalLocation(QStringLiteral("file:///data/grids/g1"), project),
             QStringLiteral("/data/grids/g1"));
    QCOMPARE(pl::storedLocalLocation(QStringLiteral("/"), project), QStringLiteral("/"));
}

void ProjectLocationsTest::findEntryMatchesAnySpellingOfAStoredEntry()
{
    QTemporaryDir root;
    QVERIFY(root.isValid());
    const fs::path project(root.path().toStdString());
    fs::create_directories(project / "grids" / "g1");
    fs::create_directories(project / "grids" / "g2");

    // An older project may store a relative entry; a retry spells it
    // absolute, with a trailing separator, or through "..".
    const std::vector<vc::project::Entry> entries{{"grids/g1", {}}, {"s3://bucket/v.zarr", {}}};
    const QString absolute = QString::fromStdString((project / "grids" / "g1").string());
    QCOMPARE(pl::findEntry(entries, absolute, project), QStringLiteral("grids/g1"));
    QCOMPARE(pl::findEntry(entries, absolute + QLatin1Char('/'), project),
             QStringLiteral("grids/g1"));
    QCOMPARE(pl::findEntry(entries, QStringLiteral("grids/g2/../g1"), project),
             QStringLiteral("grids/g1"));
    QVERIFY(pl::findEntry(entries, QStringLiteral("grids/g2"), project).isEmpty());
    QCOMPARE(pl::findEntry(entries, QStringLiteral("s3://bucket/v.zarr"), project),
             QStringLiteral("s3://bucket/v.zarr"));
}

QTEST_GUILESS_MAIN(ProjectLocationsTest)
#include "test_project_locations.moc"
