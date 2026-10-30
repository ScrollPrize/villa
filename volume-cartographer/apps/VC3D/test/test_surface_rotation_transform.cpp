#include "overlays/SurfaceRotationTransform.hpp"

#include <QTest>

#include <string>
#include <vector>


namespace
{
struct RecordingSurface {
    std::vector<std::string> operations;

    void rotate(float) { operations.emplace_back("rotate"); }
    void flipV() { operations.emplace_back("flipV"); }
};
}  // namespace

class SurfaceRotationTransformTest : public QObject
{
    Q_OBJECT

private slots:
    void rotatesThenFlipsHorizontally()
    {
        RecordingSurface surface;
        vc3d::surface_rotation::Transform{37.0f, true}.applyInMemory(surface);
        const std::vector<std::string> expected{"rotate", "flipV"};
        QVERIFY(surface.operations == expected);
    }

    void rotatesWithoutFlip()
    {
        RecordingSurface surface;
        vc3d::surface_rotation::Transform{37.0f, false}.applyInMemory(surface);
        const std::vector<std::string> expected{"rotate"};
        QVERIFY(surface.operations == expected);
    }

    void supportsHorizontalFlipWithoutRotation()
    {
        RecordingSurface surface;
        vc3d::surface_rotation::Transform{0.0f, true}.applyInMemory(surface);
        const std::vector<std::string> expected{"flipV"};
        QVERIFY(surface.operations == expected);
    }

    void skipsNoOpWithoutTouchingTheSurface()
    {
        RecordingSurface surface;
        const vc3d::surface_rotation::Transform transform{0.0f, false};
        QVERIFY(transform.isNoOp());
        transform.applyInMemory(surface);
        QVERIFY(surface.operations.empty());
    }

    void blocksPersistenceForDisconnectedComponents()
    {
        QVERIFY(vc3d::surface_rotation::TransformPersistenceCompatibility{}.allowed());
        QVERIFY(!vc3d::surface_rotation::TransformPersistenceCompatibility{true}.allowed());
    }
};

QTEST_APPLESS_MAIN(SurfaceRotationTransformTest)

#include "test_surface_rotation_transform.moc"
