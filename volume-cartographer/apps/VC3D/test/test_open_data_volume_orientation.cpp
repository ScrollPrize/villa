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
    // The store's umbilicus is the first authority, the catalog the second,
    // and a sense nobody states is refused - never decided from the geometry.
    void theSenseIsStatedOrRefused()
    {
        using vc3d::opendata::CatalogVolumeSense;
        using vc3d::opendata::statedWindingSense;
        const std::optional<CatalogVolumeSense> oriented = CatalogVolumeSense{"s/v", orientation(false, false)};
        const std::optional<CatalogVolumeSense> unoriented =
            CatalogVolumeSense{"s/v", orientation(std::nullopt, false)};

        const auto store = statedWindingSense(-1, oriented);
        QCOMPARE(store.sense, -1); // the catalog alone would say +1
        QVERIFY(!store.fromCatalog);
        const auto catalog = statedWindingSense(std::nullopt, oriented);
        QCOMPARE(catalog.sense, 1);
        QVERIFY(catalog.fromCatalog);

        const auto refused = [](std::optional<int> umbilicus, const std::optional<CatalogVolumeSense>& catalog) {
            try {
                (void)statedWindingSense(umbilicus, catalog);
            } catch (const std::runtime_error&) {
                return true;
            }
            return false;
        };
        QVERIFY(refused(std::nullopt, unoriented));
        QVERIFY(refused(std::nullopt, std::nullopt));
        QVERIFY(refused(0, oriented));
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

    void propertiesAreReadAsBooleans()
    {
        const auto manifest = vcr::catalog::parseOpenDataManifest(manifestText(
            R"({"z_direction_is_top_to_bottom": true, "left_handed_coordinates": false})"));
        const auto found = manifest.findSample("PHerc0139")->findVolume("20260102150214")->orientation;
        QCOMPARE(found.zTopToBottom, std::optional<bool>(true));
        QCOMPARE(found.leftHandedCoordinates, std::optional<bool>(false));
        QVERIFY(vc3d::opendata::orientationText(found) != vc3d::opendata::orientationText(orientation(std::nullopt, std::nullopt)));
    }
};

QTEST_GUILESS_MAIN(TestOpenDataVolumeOrientation)
#include "test_open_data_volume_orientation.moc"
