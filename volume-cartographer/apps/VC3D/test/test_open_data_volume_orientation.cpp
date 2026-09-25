#include <QtTest>

#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include "OpenDataVolumeOrientation.hpp"

using vc3d::opendata::CatalogVolumeOrientationLookup;
using vc3d::opendata::VolumeOrientation;

namespace
{

VolumeOrientation orientation(std::optional<bool> topToBottom, std::optional<bool> leftHanded)
{
    VolumeOrientation value;
    value.zTopToBottom = topToBottom;
    value.leftHandedCoordinates = leftHanded;
    return value;
}

// A manifest with one sample and one volume carrying the given properties
// object (JSON text).
std::string manifestText(const std::string& properties)
{
    return R"({"samples": {"PHerc0139": {"volumes": {"20260102150214": {"properties": )" +
           properties + R"(}}}}})";
}

void writeFile(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream out(path, std::ios::trunc);
    out << text;
}

} // namespace

class TestOpenDataVolumeOrientation : public QObject
{
    Q_OBJECT

private slots:
    void coordinateSpaceNamesSampleAndVolume()
    {
        const auto ids = vc3d::opendata::sampleAndVolumeOfCoordinateSpace(
            "PHerc0139/20260102150214@L0");
        QVERIFY(ids.has_value());
        QCOMPARE(ids->first, std::string("PHerc0139"));
        QCOMPARE(ids->second, std::string("20260102150214"));

        // No level suffix is also a name.
        const auto bare = vc3d::opendata::sampleAndVolumeOfCoordinateSpace(
            "PHerc0139/20260102150214");
        QVERIFY(bare.has_value());
        QCOMPARE(bare->second, std::string("20260102150214"));

        QVERIFY(!vc3d::opendata::sampleAndVolumeOfCoordinateSpace("").has_value());
        QVERIFY(!vc3d::opendata::sampleAndVolumeOfCoordinateSpace("PHerc0139").has_value());
        QVERIFY(!vc3d::opendata::sampleAndVolumeOfCoordinateSpace("PHerc0139/").has_value());
        QVERIFY(!vc3d::opendata::sampleAndVolumeOfCoordinateSpace("/20260102150214").has_value());
        QVERIFY(!vc3d::opendata::sampleAndVolumeOfCoordinateSpace("a/b/c").has_value());

        // The catalog volume drops the pyramid level: two levels of one
        // volume are one catalog entry.
        QCOMPARE(vc3d::opendata::catalogVolumeOfCoordinateSpace("PHerc0139/20260102150214@L0"),
                 std::string("PHerc0139/20260102150214"));
        QCOMPARE(vc3d::opendata::catalogVolumeOfCoordinateSpace("PHerc0139/20260102150214@L2"),
                 std::string("PHerc0139/20260102150214"));
        QCOMPARE(vc3d::opendata::catalogVolumeOfCoordinateSpace(""), std::string());
        QCOMPARE(vc3d::opendata::catalogVolumeOfCoordinateSpace("PHerc0139"), std::string());
    }

    // The sense is -1 exactly when one of the two properties holds, and
    // unknown while either is unset.
    void windingSenseFollowsTheTwoProperties()
    {
        QCOMPARE(vc3d::opendata::windingChiralityOf(orientation(true, false)), std::optional<int>(-1));
        QCOMPARE(vc3d::opendata::windingChiralityOf(orientation(false, true)), std::optional<int>(-1));
        QCOMPARE(vc3d::opendata::windingChiralityOf(orientation(false, false)), std::optional<int>(1));
        QCOMPARE(vc3d::opendata::windingChiralityOf(orientation(true, true)), std::optional<int>(1));
        QVERIFY(!vc3d::opendata::windingChiralityOf(orientation(std::nullopt, false)).has_value());
        QVERIFY(!vc3d::opendata::windingChiralityOf(orientation(true, std::nullopt)).has_value());
        QVERIFY(!vc3d::opendata::windingChiralityOf(orientation(std::nullopt, std::nullopt)).has_value());
    }

    void propertiesAreReadAsBooleansOrTheirStrings()
    {
        const auto manifest = vc3d::opendata::parseOpenDataManifest(manifestText(
            R"({"z_direction_is_top_to_bottom": true, "left_handed_coordinates": "false"})"));
        const auto found =
            vc3d::opendata::findVolumeOrientation(manifest, "PHerc0139", "20260102150214");
        QVERIFY(found.has_value());
        QCOMPARE(found->zTopToBottom, std::optional<bool>(true));
        QCOMPARE(found->leftHandedCoordinates, std::optional<bool>(false));

        // Absent, null, or anything else is unset - never a default.
        const auto unset = vc3d::opendata::parseOpenDataManifest(manifestText(
            R"({"left_handed_coordinates": null, "pixel_size_um": 2.399})"));
        const auto partial =
            vc3d::opendata::findVolumeOrientation(unset, "PHerc0139", "20260102150214");
        QVERIFY(partial.has_value());
        QVERIFY(!partial->zTopToBottom.has_value());
        QVERIFY(!partial->leftHandedCoordinates.has_value());

        QVERIFY(!vc3d::opendata::findVolumeOrientation(manifest, "PHerc0139", "20250728140407")
                     .has_value());
        QVERIFY(!vc3d::opendata::findVolumeOrientation(manifest, "PHerc1451", "20260102150214")
                     .has_value());
    }

    // The lookup answers from the cached manifest file, parses it once per
    // version, and notices when the file changes or disappears.
    void lookupFollowsTheManifestFile()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const std::filesystem::path path =
            std::filesystem::path(dir.path().toStdString()) / "metadata.json";

        CatalogVolumeOrientationLookup lookup(path);
        // No file yet.
        QVERIFY(!lookup.lookup("PHerc0139/20260102150214@L0").has_value());

        writeFile(path, manifestText(
                            R"({"z_direction_is_top_to_bottom": true, "left_handed_coordinates": false})"));
        auto found = lookup.lookup("PHerc0139/20260102150214@L0");
        QVERIFY(found.has_value());
        QCOMPARE(vc3d::opendata::windingChiralityOf(*found), std::optional<int>(-1));
        // A volume the manifest lacks, and a name that is no coordinate space.
        QVERIFY(!lookup.lookup("PHerc0139/20250728140407@L0").has_value());
        QVERIFY(!lookup.lookup("PHerc0139").has_value());

        // Rewritten with the property withdrawn: the file's size moves, so
        // the answer follows without any other prompt.
        writeFile(path, manifestText(R"({"left_handed_coordinates": false})"));
        found = lookup.lookup("PHerc0139/20260102150214@L0");
        QVERIFY(found.has_value());
        QVERIFY(!found->zTopToBottom.has_value());
        QVERIFY(!vc3d::opendata::windingChiralityOf(*found).has_value());

        // Unparseable: no answer, and no exception.
        writeFile(path, "{ not json");
        QVERIFY(!lookup.lookup("PHerc0139/20260102150214@L0").has_value());

        // Removed: no answer.
        QVERIFY(std::filesystem::remove(path));
        QVERIFY(!lookup.lookup("PHerc0139/20260102150214@L0").has_value());
    }

    // An answer comes with the manifest version it was read from, the same
    // token a stat alone reports, so a rebuild's answer can be checked
    // against the file later without a parse; nothing at all for a volume
    // the catalog is not asked about.
    void answersAreBoundToTheManifestVersion()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const std::filesystem::path path =
            std::filesystem::path(dir.path().toStdString()) / "metadata.json";
        CatalogVolumeOrientationLookup lookup(path);
        const std::string space = "PHerc0139/20260102150214@L0";
        QCOMPARE(lookup.manifestToken(""), std::string());
        QVERIFY(!lookup.resolve("").orientation.has_value());
        QCOMPARE(lookup.resolve("").manifestToken, std::string());
        QCOMPARE(lookup.manifestToken(space), std::string("absent"));
        QCOMPARE(lookup.resolve(space).manifestToken, std::string("absent"));

        writeFile(path, manifestText(
                            R"({"z_direction_is_top_to_bottom": true, "left_handed_coordinates": false})"));
        const auto first = lookup.resolve(space);
        QVERIFY(first.orientation.has_value());
        QCOMPARE(vc3d::opendata::windingChiralityOf(*first.orientation), std::optional<int>(-1));
        QVERIFY(first.manifestToken != "absent");
        QCOMPARE(lookup.manifestToken(space), first.manifestToken);
        // A volume the manifest lacks, and a name that is no coordinate
        // space, still report the version they were checked against.
        QCOMPARE(lookup.resolve("PHerc0139/20250728140407@L0").manifestToken, first.manifestToken);
        QCOMPARE(lookup.resolve("PHerc0139").manifestToken, first.manifestToken);

        // Rewritten: another version, another answer, and the stat sees the
        // change without a parse.
        writeFile(path, manifestText(R"({"left_handed_coordinates": false})"));
        QVERIFY(lookup.manifestToken(space) != first.manifestToken);
        const auto second = lookup.resolve(space);
        QVERIFY(second.manifestToken != first.manifestToken);
        QCOMPARE(lookup.manifestToken(space), second.manifestToken);
        QVERIFY(second.orientation.has_value());
        QVERIFY(!vc3d::opendata::windingChiralityOf(*second.orientation).has_value());
    }

    // A manifest replaced under every read answers nothing under a version
    // no stat reports, so a build on that answer can never pass for a build
    // on the settled file; once the file settles, the next read answers it.
    void aManifestReplacedUnderEveryReadIsUnstable()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const std::filesystem::path path =
            std::filesystem::path(dir.path().toStdString()) / "metadata.json";
        const std::string space = "PHerc0139/20260102150214@L0";
        // Distinct sizes, so every rewrite is a new version to the stat.
        const std::string oriented = manifestText(
            R"({"z_direction_is_top_to_bottom": true, "left_handed_coordinates": false})");
        writeFile(path, oriented);
        CatalogVolumeOrientationLookup lookup(path);
        int rewrites = 0;
        lookup.setAfterParseHookForTesting([&]() {
            ++rewrites;
            writeFile(path, oriented + std::string(rewrites, ' '));
        });
        const auto unstable = lookup.resolve(space);
        QCOMPARE(rewrites, 3);
        QVERIFY(!unstable.orientation.has_value());
        QCOMPARE(unstable.manifestToken, std::string("unstable"));
        QVERIFY(lookup.manifestToken(space) != unstable.manifestToken);
        // Nothing was memoized: the settled file is read afresh.
        lookup.setAfterParseHookForTesting({});
        const auto settled = lookup.resolve(space);
        QVERIFY(settled.orientation.has_value());
        QCOMPARE(vc3d::opendata::windingChiralityOf(*settled.orientation), std::optional<int>(-1));
        QCOMPARE(settled.manifestToken, lookup.manifestToken(space));
        // One replacement mid-read is recovered from within the attempts.
        int once = 0;
        lookup.setAfterParseHookForTesting([&]() {
            if (once++ == 0) {
                writeFile(path, manifestText(R"({"left_handed_coordinates": false})"));
            }
        });
        const auto recovered = lookup.resolve("PHerc0139/20260102150214@L2");
        QCOMPARE(once, 2);
        QVERIFY(recovered.orientation.has_value());
        QVERIFY(!vc3d::opendata::windingChiralityOf(*recovered.orientation).has_value());
        QCOMPARE(recovered.manifestToken, lookup.manifestToken(space));
    }
};

QTEST_GUILESS_MAIN(TestOpenDataVolumeOrientation)
#include "test_open_data_volume_orientation.moc"
