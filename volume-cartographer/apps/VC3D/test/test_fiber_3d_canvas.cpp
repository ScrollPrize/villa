#include "fibers3d/Fiber3DCanvas.hpp"
#include "fibers3d/Fiber3DRibbon.hpp"

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
                if (c.red() > 100 && c.red() == c.green() && c.green() == c.blue())
                    ++count;
            }
        return count;
    }

private slots:
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
