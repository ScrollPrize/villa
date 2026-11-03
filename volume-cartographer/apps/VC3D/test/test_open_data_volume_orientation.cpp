#include <QtTest>

#include "OpenDataVolumeOrientation.hpp"

using vcr::catalog::OpenDataOrientation;

namespace
{

OpenDataOrientation orientation(std::optional<bool> topToBottom, std::optional<bool> leftHanded)
{
    return {topToBottom, leftHanded};
}

// A manifest with one sample and one volume carrying the given properties
// object (JSON text).
std::string manifestText(const std::string& properties)
{
    return R"({"samples": {"PHerc0139": {"volumes": {"20260102150214": {"properties": )" +
           properties + R"(}}}}})";
}

} // namespace

class TestOpenDataVolumeOrientation : public QObject
{
    Q_OBJECT

private slots:
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

    void propertiesAreReadAsBooleans()
    {
        const auto manifest = vcr::catalog::parseOpenDataManifest(manifestText(
            R"({"z_direction_is_top_to_bottom": true, "left_handed_coordinates": false})"));
        const auto found = manifest.findSample("PHerc0139")->findVolume("20260102150214")->orientation;
        QCOMPARE(found.zTopToBottom, std::optional<bool>(true));
        QCOMPARE(found.leftHandedCoordinates, std::optional<bool>(false));

        // Absent, null, or anything else is unset - never a default.
        const auto unset = vcr::catalog::parseOpenDataManifest(manifestText(
            R"({"left_handed_coordinates": null, "z_direction_is_top_to_bottom": "true", "pixel_size_um": 2.399})"));
        const auto partial = unset.findSample("PHerc0139")->findVolume("20260102150214")->orientation;
        QVERIFY(!partial.zTopToBottom.has_value());
        QVERIFY(!partial.leftHandedCoordinates.has_value());
        QVERIFY(vc3d::opendata::orientationText(found) != vc3d::opendata::orientationText(partial));
    }
};

QTEST_GUILESS_MAIN(TestOpenDataVolumeOrientation)
#include "test_open_data_volume_orientation.moc"
