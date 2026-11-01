#include "fibers3d/Fiber3DCanvas.hpp"
#include "fibers3d/Fiber3DRibbon.hpp"
#include "scene3d/VolumePreview.hpp"

#include <QApplication>
#include <QSurfaceFormat>
#include <QTest>
#include <QThreadPool>

using namespace vc3d::fibers3d;

class Fiber3DCanvasTest : public QObject
{
    Q_OBJECT

    static std::shared_ptr<Scene> line(const QString& key)
    {
        auto scene = std::make_shared<Scene>();
        Fiber fiber;
        fiber.key = key;
        fiber.runs = {Run{{{-20, 0, 0}, {20, 0, 0}}}};
        scene->fibers.push_back(std::move(fiber));
        return scene;
    }

    static int grayPixels(const QImage& image)
    {
        int count = 0;
        const double ratio = image.devicePixelRatio();
        // Exclude labels: only the CT pixels in the scene count.
        for (int y = int(60 * ratio); y < image.height() - int(30 * ratio); ++y)
            for (int x = 0; x < image.width(); ++x) {
                const auto c = image.pixelColor(x, y);
                // Translucent CT blends with the slightly blue background.
                if (c.red() > 100 && std::abs(c.red()-c.green()) <= 3 && std::abs(c.green()-c.blue()) <= 3)
                    ++count;
            }
        return count;
    }

private slots:
    void volumeShowsInteriorWithoutSurfaceOrSlices()
    {
        vc3d::scene3d::Scene3DCanvas canvas;
        canvas.resize(480,360);
        canvas.show();
        QVERIFY(QTest::qWaitForWindowExposed(&canvas));
        QVERIFY(canvas.isValid());
        canvas.setSurfaceOpacity(1);
        canvas.setCTVolumeDensity(4);
        auto volume = [&](bool interior) {
            auto data = std::make_shared<vc3d::scene3d::VolumeData>();
            data->bounds = {{-16,-16,-16},{16,16,16}};
            auto image = std::make_shared<slices::Image>();
            auto& slab = image->slab.emplace();
            slab.origin = {-16,-16,-16};
            slab.right={32,0,0}; slab.down={0,32,0}; slab.through={0,0,32};
            slab.width=slab.height=slab.depth=16;
            slab.pixels.resize(16*16*16*2);
            for (int z=0; z<16; ++z)
                for (int y=0; y<16; ++y)
                    for (int x=0; x<16; ++x) {
                        const bool core = x>=4 && x<12 && y>=4 && y<12 && z>=4 && z<12;
                        const int i=2*((z*16+y)*16+x);
                        slab.pixels[i]=interior && core ? 230 : 35;
                        slab.pixels[i+1]=255;
                    }
            data->images.push_back(image);
            return data;
        };
        const auto outerOnly = volume(false), withInterior = volume(true);
        canvas.setCTVolume(outerOnly);
        QTRY_VERIFY(canvas.isCTVolumeReady());
        const auto outer = canvas.grabFramebuffer();
        canvas.setCTVolume(withInterior);
        QTRY_VERIFY(canvas.isCTVolumeReady());
        const auto internal = canvas.grabFramebuffer();
        QVERIFY(internal != outer); // same contour and boundary voxels
        QVERIFY(canvas.ctSlices().empty());
        QVERIFY(!canvas.surfaceMesh());
        canvas.setCTVolumeDensity(0);
        const auto transparent = canvas.grabFramebuffer();
        canvas.setCTVolume({});
        QCOMPARE(canvas.grabFramebuffer(), transparent);
        canvas.setCTVolume(withInterior);
        canvas.setCTVolumeDensity(4);
        canvas.setPerspective(true);
        QTRY_VERIFY(canvas.isCTVolumeReady());
        QVERIFY(canvas.grabFramebuffer() != transparent);
        canvas.clearContent();
        QVERIFY(!canvas.ctVolume());
    }
    void ctSlicesRenderWithoutARibbon_data()
    {
        QTest::addColumn<bool>("volumetric");
        QTest::addColumn<bool>("withFibers");
        QTest::newRow("single-slice") << false << true;
        QTest::newRow("ct-only-slice") << false << false;
        QTest::newRow("volumetric-slab") << true << true;
        QTest::newRow("ct-only-volume") << true << false;
    }
    void ctSlicesRenderWithoutARibbon()
    {
        QFETCH(bool, volumetric);
        QFETCH(bool, withFibers);
        QWidget container;
        container.resize(480, 360);
        std::unique_ptr<vc3d::scene3d::Scene3DCanvas> view;
        if (withFibers) view = std::make_unique<Fiber3DCanvas>();
        else view = std::make_unique<vc3d::scene3d::Scene3DCanvas>();
        auto& canvas = *view;
        canvas.resize(480, 360);
        canvas.show();
        QVERIFY(QTest::qWaitForWindowExposed(&canvas));
        QVERIFY(canvas.isValid());
        if (withFibers) {
            auto& fibers = static_cast<Fiber3DCanvas&>(canvas);
            auto scene = line("fiber");
            fibers.setScene(scene);
            QTRY_VERIFY(fibers.scene() == scene && !fibers.isPreparing());
            fibers.fitAll();
        }
        const int without = grayPixels(canvas.grabFramebuffer());
        auto image = std::make_shared<slices::Image>();
        image->plane = {0, {-20,-12,0}, {40,0,0}, {0,24,0}, 2, 2};
        image->tile.width = image->tile.height = 2;
        image->tile.pixels = {180,255,180,255,180,255,180,255};
        if (volumetric) {
            auto& slab = image->slab.emplace();
            slab.origin = {-20,-12,-4};
            slab.right = {40,0,0}; slab.down = {0,24,0}; slab.through = {0,0,8};
            slab.width = slab.height = 2; slab.depth = 8;
            for (int i=0; i<32; ++i) {
                slab.pixels.push_back(180); slab.pixels.push_back(255);
            }
        }
        canvas.setCTSlices({image});
        QTRY_VERIFY(grayPixels(canvas.grabFramebuffer()) > without + 1000);
        canvas.setCTSlices({});
        QTRY_COMPARE(grayPixels(canvas.grabFramebuffer()), without);
        canvas.setCTSlices({image});
        // Moving a widget between top-level windows recreates its GL context.
        // Both base and fiber resources must be released and uploaded again.
        canvas.setParent(&container);
        container.show();
        canvas.show();
        QVERIFY(QTest::qWaitForWindowExposed(&container));
        QTRY_VERIFY(grayPixels(canvas.grabFramebuffer()) > without + 1000);
    }
    void surfaceWithoutFibers()
    {
        vc3d::scene3d::Scene3DCanvas canvas;
        canvas.resize(480, 360);
        canvas.show();
        QVERIFY(QTest::qWaitForWindowExposed(&canvas));
        QVERIFY(canvas.isValid());
        const auto empty = canvas.grabFramebuffer();
        auto mesh = std::make_shared<surface::Mesh>();
        mesh->bounds = {{-20,-20,0}, {20,20,0}};
        mesh->vertices = {{{-20,-20,0},{0,0,1}}, {{20,-20,0},{0,0,1}}, {{0,20,0},{0,0,1}}};
        mesh->indices = {0,1,2};
        canvas.setSurfaceMesh(mesh);
        canvas.fitAll();
        QTRY_VERIFY(canvas.isSurfaceMeshReady());
        QVERIFY(canvas.grabFramebuffer() != empty);
        canvas.setSurfaceVisible(false);
        QCOMPARE(canvas.grabFramebuffer(), empty);
        canvas.setSurfaceVisible(true);
        QTRY_VERIFY(canvas.grabFramebuffer() != empty);
        const auto shaded = canvas.grabFramebuffer();
        auto textured = std::make_shared<surface::Mesh>(*mesh);
        auto texture = std::make_shared<surface::Texture>();
        texture->width = texture->height = 2;
        texture->pixels = {200,255,200,255,200,255,200,255};
        textured->texture = texture;
        canvas.setSurfaceMesh(textured);
        QTRY_VERIFY(canvas.isSurfaceMeshReady());
        QTRY_VERIFY(grayPixels(canvas.grabFramebuffer()) > grayPixels(shaded) + 1000);
        canvas.setSurfaceTextured(false);
        QCOMPARE(canvas.grabFramebuffer(), shaded);
        canvas.setSurfaceTextured(true);
        canvas.setCTWindow(220, 255);
        QCOMPARE(grayPixels(canvas.grabFramebuffer()), 0);
        canvas.setCTWindow(0, 255);
        QVERIFY(grayPixels(canvas.grabFramebuffer()) > 1000);
        canvas.setSurfaceMesh({});
        QCOMPARE(canvas.grabFramebuffer(), empty);
    }
    void pinnedFiberKeepsPaletteAndDisplaysRibbon()
    {
        Fiber3DCanvas canvas;
        canvas.resize(480, 360);
        canvas.show();
        QVERIFY(QTest::qWaitForWindowExposed(&canvas));
        if (!canvas.isValid())
            QSKIP("OpenGL 3.3 widget unavailable");
        DisplaySettings settings;
        settings.colorMode = ColorMode::Custom;
        settings.custom = QColor(240, 40, 40);
        settings.highlightSelection = false;
        // Test the palette on fully covered interior pixels. A one-pixel line
        // can consist only of antialias samples blended with the gray ribbon
        // at DPR 1, while the same logical line has a solid core at DPR 2.
        settings.lineWidth = 4.f;
        canvas.setDisplaySettings(settings);
        auto scene = line("annotation");
        canvas.setScene(scene);
        canvas.setSelectedKey("annotation");
        QTRY_VERIFY(canvas.scene() == scene && !canvas.isPreparing());
        canvas.fitSelected();
        const auto without = canvas.grabFramebuffer();
        auto image = std::make_shared<ribbon::Image>();
        image->key = "annotation";
        ct::Tile tile;
        tile.width = tile.height = 2;
        tile.left = {{-20, -6, 0}, {20, -6, 0}};
        tile.right = {{-20, 6, 0}, {20, 6, 0}};
        tile.pixels = {180, 255, 180, 255, 180, 255, 180, 255};
        image->tiles.push_back(std::move(tile));
        canvas.setCTRibbon(image);
        const auto with = canvas.grabFramebuffer();
        QVERIFY(grayPixels(with) > grayPixels(without) + 1000);
        int red = 0, yellow = 0;
        const double ratio = with.devicePixelRatio();
        for (int y = int(60 * ratio); y < with.height() - int(30 * ratio); ++y)
            for (int x = 0; x < with.width(); ++x) {
                auto c = with.pixelColor(x, y);
                red += c.red() > 120 && c.green() < 80 && c.blue() < 80;
                yellow += c.red() > 200 && c.green() > 200 && c.blue() < 160;
            }
        QVERIFY2(red > 20, qPrintable(QStringLiteral("Only %1 fully red pixels at DPR %2").arg(red).arg(ratio)));
        QCOMPARE(yellow, 0);
        canvas.setSelectedKey({});
        QCOMPARE(grayPixels(canvas.grabFramebuffer()), grayPixels(without));
        canvas.clear();
        QVERIFY(!canvas.ctRibbon());
    }

    void completedWorkerIsConsumedBeforeReplacement()
    {
        Fiber3DCanvas canvas;
        canvas.resize(320, 240);
        canvas.show();
        QVERIFY(QTest::qWaitForWindowExposed(&canvas));
        if (!canvas.isValid())
            QSKIP("OpenGL 3.3 widget unavailable");
        for (int i = 0; i < 10; ++i) {
            canvas.setScene(line("superseded"));
            // Finish CPU work without dispatching the watcher's queued signals.
            QVERIFY(QThreadPool::globalInstance()->waitForDone(5000));
            auto latest = line(QString::number(i));
            canvas.setScene(latest);
            QTRY_VERIFY(canvas.scene() == latest && !canvas.isPreparing());
        }
        canvas.setScene(line("cleared"));
        QVERIFY(QThreadPool::globalInstance()->waitForDone(5000));
        canvas.clear();
        QTRY_VERIFY(!canvas.isPreparing());
        QVERIFY(!canvas.scene());
    }
};

int main(int argc, char** argv)
{
    QSurfaceFormat format;
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setSamples(4);
    QSurfaceFormat::setDefaultFormat(format);
    QApplication app(argc, argv);
    Fiber3DCanvasTest test;
    return QTest::qExec(&test, argc, argv);
}
#include "test_fiber_3d_canvas.moc"
