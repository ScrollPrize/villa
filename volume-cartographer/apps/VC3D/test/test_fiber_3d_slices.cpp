#include "fibers3d/Fiber3DSlices.hpp"
#include "vc/core/types/Volume.hpp"
#include "../../../core/test/support/TestZarrVolume.hpp"
#include <QTemporaryDir>
#include <QTest>
#include <cmath>

using namespace vc3d::fibers3d;
namespace
{
cv::Vec3d vec(const Point& p)
{
    return {p[0], p[1], p[2]};
}
}  // namespace
class Fiber3DSlicesTest : public QObject
{
    Q_OBJECT
private slots:
    void ctSamplesFollowWorldPlanesAndMaskVolumeEdges()
    {
        QTemporaryDir directory;
        vc::test::ZarrVolumeSpec options;
        options.shapeZYX = {32, 32, 32};
        options.chunkShapeZYX = {16, 16, 16};
        options.numLevels = 2;
        auto volume = vc::test::createZarrVolume(directory.path().toStdString(), options);
        Array3D<uint8_t> data({32, 32, 32});
        for (int z = 0; z < 32; ++z)
            for (int y = 0; y < 32; ++y)
                for (int x = 0; x < 32; ++x)
                    data(z, y, x) = uint8_t(x + 2 * y + 3 * z);
        vc::test::writeZYX(*volume, data, {0, 0, 0});
        for (const auto& p : std::vector<
                 slices::Plane>{{0, {8, 8, 8}, {8, 0, 0}, {0, 8, 0}, 9, 9}, {1, {8, 8, 8}, {8, 0, 0}, {0, 0, 8}, 9, 9}, {2, {8, 8, 8}, {0, 8, 0}, {0, 0, 8}, 9, 9}, {0, {8, 8, 8}, {8, 0, 4}, {0, 8, 0}, 9, 9}}) {
            auto image = slices::sample(p, *volume);
            QCOMPARE(image->level, 0);
            QCOMPARE(image->tile.left.front(), p.origin);
            for (int r = 0; r < 9; ++r)
                for (int c = 0; c < 9; ++c) {
                    const auto pos = vec(p.origin) + vec(p.right) * (c / 8.) + vec(p.down) * (r / 8.);
                    const auto value = image->tile.pixels[(r * 9 + c) * 2];
                    QVERIFY(std::abs(double(value) - (pos[0] + 2 * pos[1] + 3 * pos[2])) <= 1.);
                    QCOMPARE(image->tile.pixels[(r * 9 + c) * 2 + 1], uint8_t(255));
                }
        }
        slices::Plane edge{0, {-2, 8, 8}, {8, 0, 0}, {0, 8, 0}, 9, 9};
        auto image = slices::sample(edge, *volume);
        QCOMPARE(image->tile.pixels[1], uint8_t(0));
        QCOMPARE(image->tile.pixels[5], uint8_t(255));
        const slices::Plane overview{0, {4, 4, 8}, {24, 0, 0}, {0, 24, 0}, 25, 25};
        auto preview = slices::sample(overview, *volume, nullptr, 7);
        QVERIFY(preview->preview);
        QCOMPARE(preview->level, 1);  // choose the available finer level when L2 is absent
        QCOMPARE(preview->tile.width, 7);
        auto detail = slices::sample(overview, *volume);
        QVERIFY(!detail->preview);
        QCOMPARE(detail->level, 0);
        QCOMPARE(preview->tile.left.front(), detail->tile.left.front());
        QCOMPARE(preview->tile.right.back(), detail->tile.right.back());
        std::atomic_bool cancelled{true};
        QVERIFY_THROWS_EXCEPTION(std::runtime_error, slices::sample(edge, *volume, &cancelled));
        edge.down = edge.right;
        QVERIFY_THROWS_EXCEPTION(std::invalid_argument, slices::sample(edge, *volume));
    }
    void volumetricSamplesKeepThicknessAndReverseDirection()
    {
        QTemporaryDir directory;
        vc::test::ZarrVolumeSpec options;
        options.shapeZYX = {32, 32, 32};
        options.chunkShapeZYX = {16, 16, 16};
        auto volume = vc::test::createZarrVolume(directory.path().toStdString(), options);
        Array3D<uint8_t> data({32, 32, 32});
        for (int z = 0; z < 32; ++z)
            for (int y = 0; y < 32; ++y)
                for (int x = 0; x < 32; ++x)
                    data(z, y, x) = uint8_t(x + 2 * y + 3 * z);
        vc::test::writeZYX(*volume, data, {0, 0, 0});
        slices::Plane p{0, {8, 8, 12}, {8, 0, 0}, {0, 8, 0}, 9, 9, true, {0, 0, 1}, 3, 2};
        auto image = slices::sample(p, *volume);
        QVERIFY(image->slab.has_value());
        QVERIFY(!image->preview);
        const auto& slab = *image->slab;
        QCOMPARE(slab.depth, 6);
        QCOMPARE(slab.through, (Point{0, 0, 6}));
        QCOMPARE(slab.origin, (Point{7.5, 7.5, 9.5}));
        for (int z = 0; z < 6; ++z)
            for (int y = 0; y < 9; ++y)
                for (int x = 0; x < 9; ++x) {
                    const size_t i = ((z * 9 + y) * 9 + x) * 2;
                    QCOMPARE(slab.pixels[i], uint8_t((8 + x) + 2 * (8 + y) + 3 * (10 + z)));
                    QCOMPARE(slab.pixels[i + 1], uint8_t(255));
                }
        p.normal = {0, 0, -1};
        auto reverse = slices::sample(p, *volume);
        QCOMPARE(reverse->slab->origin, (Point{7.5, 7.5, 14.5}));
        QCOMPARE(reverse->slab->pixels[0], uint8_t(8 + 16 + 42));
        p.front = p.behind = 0;
        auto single = slices::sample(p, *volume);
        QCOMPARE(single->slab->depth, 1);
        QCOMPARE(single->slab->through, (Point{0, 0, -1}));
        p.front = 3;
        p.behind = 2;
        auto preview = slices::sample(p, *volume, nullptr, 5);
        QVERIFY(preview->preview);
        QCOMPARE(preview->slab->through, reverse->slab->through);
        QCOMPARE(preview->slab->origin[2], reverse->slab->origin[2]);
        std::atomic_bool cancelled{true};
        QVERIFY_THROWS_EXCEPTION(std::runtime_error, slices::sample(p, *volume, &cancelled));
        p.normal = {0, 0, 0};
        QVERIFY_THROWS_EXCEPTION(std::invalid_argument, slices::sample(p, *volume));
    }
    void intersectingPlanesKeepAreaUvsAndCorrectAlphaOrder()
    {
        slices::Images images;
        for (const auto& p :
             std::vector<slices::Plane>{{0, {-10, -10, 0}, {20, 0, 0}, {0, 20, 0}, 21, 21}, {1, {-10, 0, -10}, {20, 0, 0}, {0, 0, 20}, 21, 21}, {2, {0, -10, -10}, {0, 20, 0}, {0, 0, 20}, 21, 21}}) {
            auto image = std::make_shared<slices::Image>();
            image->plane = p;
            images.push_back(image);
        }
        const auto tree = slices::partition(images);
        const auto polygons = slices::backToFront(tree.get(), {20, 30, 40});
        QCOMPARE(polygons.size(), size_t(7));  // 1 XY + 2 XZ + 4 YZ
        std::array<double, 3> areas{};
        for (const auto* polygon : polygons) {
            const auto& p = images[polygon->image]->plane;
            for (const auto& v : polygon->vertices) {
                QVERIFY(cv::norm(vec(v.position) - (vec(p.origin) + vec(p.right) * v.u + vec(p.down) * v.v)) < 1e-10);
                QVERIFY(v.u >= 0 && v.u <= 1 && v.v >= 0 && v.v <= 1);
            }
            const auto a = vec(polygon->vertices[0].position);
            for (size_t i = 1; i + 1 < polygon->vertices.size(); ++i)
                areas[polygon->image] += .5 * cv::norm((vec(polygon->vertices[i].position) - a).cross(vec(polygon->vertices[i + 1].position) - a));
        }
        for (auto area : areas)
            QVERIFY(std::abs(area - 400.) < 1e-10);
        for (const Point& eye : std::vector<Point>{{20, 30, 40}, {-20, 30, 40}, {20, -30, 40}, {20, 30, -40}, {-20, -30, -40}}) {
            const auto ray = cv::Vec3d(-1, 2, 3) - vec(eye);
            double previous = INFINITY;
            int hits = 0;
            for (const auto* polygon : slices::backToFront(tree.get(), eye)) {
                const auto a = vec(polygon->vertices[0].position);
                auto normal = (vec(polygon->vertices[1].position) - a).cross(vec(polygon->vertices[2].position) - a);
                const double t = (a - vec(eye)).dot(normal) / ray.dot(normal);
                const auto hit = vec(eye) + ray * t;
                bool inside = t > 0;
                for (size_t i = 0; i < polygon->vertices.size(); ++i) {
                    const auto v = vec(polygon->vertices[i].position), w = vec(polygon->vertices[(i + 1) % polygon->vertices.size()].position);
                    if ((w - v).cross(hit - v).dot(normal) < -1e-7)
                        inside = false;
                }
                if (inside) {
                    QVERIFY(t <= previous + 1e-10);
                    previous = t;
                    ++hits;
                }
            }
            QCOMPARE(hits, 3);
        }
    }
};
QTEST_GUILESS_MAIN(Fiber3DSlicesTest)
#include "test_fiber_3d_slices.moc"
