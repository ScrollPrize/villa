#include "overlays/SurfaceRotationTransform.hpp"

#include <QTemporaryDir>
#include <QTest>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tiffio.h>

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

bool writeTestTiff(const std::filesystem::path& path, int directoryCount)
{
    TIFF* tiff = TIFFOpen(path.string().c_str(), "w");
    if (!tiff) {
        return false;
    }

    bool success = true;
    for (int directory = 0; directory < directoryCount && success; ++directory) {
        TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, 2U);
        TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, 2U);
        TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, 2U);
        TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE);

        std::array<std::uint8_t, 2> row{
            static_cast<std::uint8_t>(directory + 1),
            static_cast<std::uint8_t>(directory + 2)};
        for (std::uint32_t y = 0; y < 2; ++y) {
            if (TIFFWriteScanline(tiff, row.data(), y, 0) < 0) {
                success = false;
                break;
            }
        }
        if (success && TIFFWriteDirectory(tiff) != 1) {
            success = false;
        }
    }

    TIFFClose(tiff);
    return success;
}
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
        const vc3d::surface_rotation::TransformPersistenceCompatibility mask{true, false, false};
        const vc3d::surface_rotation::TransformPersistenceCompatibility inspectionFailure{false, true, false};
        const vc3d::surface_rotation::TransformPersistenceCompatibility components{false, false, true};
        const vc3d::surface_rotation::TransformPersistenceCompatibility all{true, true, true};

        QVERIFY(supported.allowed());
        QVERIFY(!mask.allowed());
        QVERIFY(!inspectionFailure.allowed());
        QVERIFY(!components.allowed());
        QVERIFY(!all.allowed());
    }

    void detectsActualMaskPageCount()
    {
        QTemporaryDir temporaryDirectory;
        QVERIFY(temporaryDirectory.isValid());
        const std::filesystem::path root{temporaryDirectory.path().toStdString()};
        const auto maskPath = root / "mask.tif";

        const auto noMask = vc3d::surface_rotation::transformPersistenceCompatibility(root, false);
        QVERIFY(noMask.allowed());

        QVERIFY(writeTestTiff(maskPath, 1));
        const auto singlePage = vc3d::surface_rotation::transformPersistenceCompatibility(root, false);
        QVERIFY(!singlePage.hasMultipageMask);
        QVERIFY(!singlePage.maskInspectionFailed);
        QVERIFY(singlePage.allowed());

        QVERIFY(writeTestTiff(maskPath, 2));
        const auto multipage = vc3d::surface_rotation::transformPersistenceCompatibility(root, false);
        QVERIFY(multipage.hasMultipageMask);
        QVERIFY(!multipage.maskInspectionFailed);
        QVERIFY(!multipage.allowed());

        const auto components = vc3d::surface_rotation::transformPersistenceCompatibility(std::filesystem::path{}, true);
        QVERIFY(components.hasDisconnectedComponents);
        QVERIFY(!components.allowed());
    }

    void detectsLegacyMaskByPageCountRatherThanFilename()
    {
        QTemporaryDir temporaryDirectory;
        QVERIFY(temporaryDirectory.isValid());
        const std::filesystem::path root{temporaryDirectory.path().toStdString()};
        const auto maskPath = root / "multilayer_mask.tif";

        QVERIFY(writeTestTiff(maskPath, 1));
        const auto singlePage = vc3d::surface_rotation::transformPersistenceCompatibility(root, false);
        QVERIFY(singlePage.allowed());

        QVERIFY(writeTestTiff(maskPath, 2));
        const auto multipage = vc3d::surface_rotation::transformPersistenceCompatibility(root, false);
        QVERIFY(multipage.hasMultipageMask);
        QVERIFY(!multipage.allowed());
    }

    void blocksPersistenceWhenMaskInspectionFails()
    {
        QTemporaryDir temporaryDirectory;
        QVERIFY(temporaryDirectory.isValid());
        const std::filesystem::path root{temporaryDirectory.path().toStdString()};

        std::ofstream(root / "mask.tif") << "not a TIFF";
        const auto unreadable = vc3d::surface_rotation::transformPersistenceCompatibility(root, false);
        QVERIFY(!unreadable.hasMultipageMask);
        QVERIFY(unreadable.maskInspectionFailed);
        QVERIFY(!unreadable.allowed());
    }
};

QTEST_APPLESS_MAIN(SurfaceRotationTransformTest)

#include "test_surface_rotation_transform.moc"
