#include "overlays/SurfaceRotationTransform.hpp"

#include <QTemporaryDir>
#include <QTest>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct RecordingSurface {
    std::filesystem::path path{"source-segment"};
    std::vector<std::string> operations;
    bool rotateSawBackingPath{false};
    bool flipSawBackingPath{false};
    bool failFlip{false};

    void rotate(float)
    {
        operations.emplace_back("rotate");
        rotateSawBackingPath = rotateSawBackingPath || !path.empty();
    }

    void flipV()
    {
        operations.emplace_back("flipV");
        flipSawBackingPath = flipSawBackingPath || !path.empty();
        if (failFlip) {
            throw std::runtime_error("flip failed");
        }
    }
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
        QVERIFY(surface.rotateSawBackingPath);
        QVERIFY(!surface.flipSawBackingPath);
        QVERIFY(surface.path == std::filesystem::path{"source-segment"});
    }

    void keepsBackingPathVisibleForRotationOnly()
    {
        RecordingSurface surface;

        vc3d::surface_rotation::Transform{37.0f, false}.applyInMemory(surface);

        const std::vector<std::string> expected{"rotate"};
        QVERIFY(surface.operations == expected);
        QVERIFY(surface.rotateSawBackingPath);
        QVERIFY(!surface.flipSawBackingPath);
        QVERIFY(surface.path == std::filesystem::path{"source-segment"});
    }

    void supportsHorizontalFlipWithoutRotation()
    {
        RecordingSurface surface;

        vc3d::surface_rotation::Transform{0.0f, true}.applyInMemory(surface);

        const std::vector<std::string> expected{"flipV"};
        QVERIFY(surface.operations == expected);
        QVERIFY(!surface.flipSawBackingPath);
        QVERIFY(surface.path == std::filesystem::path{"source-segment"});
    }

    void skipsNoOpWithoutTouchingTheSurface()
    {
        RecordingSurface surface;

        const vc3d::surface_rotation::Transform transform{0.0f, false};
        QVERIFY(transform.isNoOp());
        transform.applyInMemory(surface);

        QVERIFY(surface.operations.empty());
        QVERIFY(!surface.flipSawBackingPath);
        QVERIFY(surface.path == std::filesystem::path{"source-segment"});
    }

    void restoresBackingPathWhenTransformThrows()
    {
        RecordingSurface surface;
        surface.failFlip = true;

        bool threw = false;
        try {
            vc3d::surface_rotation::Transform{0.0f, true}.applyInMemory(surface);
        } catch (const std::runtime_error&) {
            threw = true;
        }

        QVERIFY(threw);
        QVERIFY(!surface.flipSawBackingPath);
        QVERIFY(surface.path == std::filesystem::path{"source-segment"});
    }

    void blocksTransformPersistenceForUnsupportedSurfaceFeatures()
    {
        const vc3d::surface_rotation::TransformPersistenceCompatibility supported;
        const vc3d::surface_rotation::TransformPersistenceCompatibility mask{true, false};
        const vc3d::surface_rotation::TransformPersistenceCompatibility components{false, true};
        const vc3d::surface_rotation::TransformPersistenceCompatibility both{true, true};

        QVERIFY(supported.allowed());
        QVERIFY(!mask.allowed());
        QVERIFY(!components.allowed());
        QVERIFY(!both.allowed());
    }

    void detectsPersistenceCompatibilityFromSurfaceFiles()
    {
        QTemporaryDir temporaryDirectory;
        QVERIFY(temporaryDirectory.isValid());
        const std::filesystem::path root{temporaryDirectory.path().toStdString()};

        const auto supported = vc3d::surface_rotation::transformPersistenceCompatibility(root, false);
        QVERIFY(supported.allowed());

        std::ofstream(root / "multilayer_mask.tif").put('\0');
        const auto mask = vc3d::surface_rotation::transformPersistenceCompatibility(root, false);
        QVERIFY(mask.hasMultipageMask);
        QVERIFY(!mask.allowed());

        const auto components = vc3d::surface_rotation::transformPersistenceCompatibility(std::filesystem::path{}, true);
        QVERIFY(components.hasDisconnectedComponents);
        QVERIFY(!components.allowed());
    }

    void reconcilesHorizontalFlipSelectionWithCompatibility()
    {
        const vc3d::surface_rotation::TransformPersistenceCompatibility supported;
        const vc3d::surface_rotation::TransformPersistenceCompatibility unsupported{true, false};

        const auto retained = vc3d::surface_rotation::reconcileHorizontalFlipSelection(true, supported);
        QVERIFY(retained.selected);
        QVERIFY(!retained.selectionCleared);

        const auto cleared = vc3d::surface_rotation::reconcileHorizontalFlipSelection(true, unsupported);
        QVERIFY(!cleared.selected);
        QVERIFY(cleared.selectionCleared);

        const auto alreadyClear = vc3d::surface_rotation::reconcileHorizontalFlipSelection(false, unsupported);
        QVERIFY(!alreadyClear.selected);
        QVERIFY(!alreadyClear.selectionCleared);
    }
};

QTEST_APPLESS_MAIN(SurfaceRotationTransformTest)

#include "test_surface_rotation_transform.moc"
