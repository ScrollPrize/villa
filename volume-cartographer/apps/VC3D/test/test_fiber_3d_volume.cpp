#include "fibers3d/Fiber3DVolumeRenderer.hpp"
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QTest>
#include <cmath>

using namespace vc3d::fibers3d;
class Fiber3DVolumeTest : public QObject
{
    Q_OBJECT
    QOpenGLContext context;
    QOffscreenSurface surface;
private slots:
    void initTestCase()
    {
        QSurfaceFormat format;
        format.setVersion(3, 3);
        format.setProfile(QSurfaceFormat::CoreProfile);
        format.setDepthBufferSize(24);
        format.setStencilBufferSize(8);
        context.setFormat(format);
        if (!context.create())
            QSKIP("OpenGL 3.3 context unavailable");
        surface.setFormat(context.format());
        surface.create();
        if (!context.makeCurrent(&surface))
            QSKIP("Offscreen OpenGL surface unavailable");
    }
    void thicknessOverlapDepthAndStepSize()
    {
        QOpenGLFunctions_3_3_Core gl;
        QVERIFY(gl.initializeOpenGLFunctions());
        QOpenGLFramebufferObjectFormat format;
        format.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);
        format.setSamples(4);
        QOpenGLFramebufferObject fbo(128, 128, format);
        QVERIFY(fbo.isValid());
        Fiber3DVolumeRenderer renderer;
        auto image = std::make_shared<slices::Image>();
        image->plane.axis = 0;
        auto& slab = image->slab.emplace();
        slab.origin = {-1, -1, -.5};
        slab.right = {2, 0, 0};
        slab.down = {0, 2, 0};
        slab.through = {0, 0, 1};
        slab.width = slab.height = 2;
        slab.depth = 1;
        for (int i = 0; i < 4; ++i) {
            slab.pixels.push_back(128);
            slab.pixels.push_back(255);
        }
        std::array<CompositeParams, 3> transfer;
        for (auto& p : transfer) {
            p.alphaOpacity = .25;
            p.tfGamma = 1;
        }
        QMatrix4x4 matrix;
        matrix.ortho(-2.f, 2.f, -2.f, 2.f, -10.f, 10.f);
        auto draw = [&](const slices::Images& images, float depth, bool interactive) {
            for (int i = 0; i < 100; ++i) {
                fbo.bind();
                gl.glViewport(0, 0, 128, 128);
                gl.glDisable(GL_SCISSOR_TEST);
                gl.glDepthMask(GL_TRUE);
                gl.glClearColor(0, 0, 0, 1);
                gl.glClearDepth(depth);
                gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                if (!renderer.draw(images, transfer, matrix, {}, fbo.handle(), 128, 128, 0, 255, 1, interactive)) {
                    if (!renderer.error().isEmpty())
                        return QImage{};
                    gl.glFinish();
                    return fbo.toImage();
                }
            }
            return QImage{};
        };
        auto one = draw({image}, 1, false);
        QVERIFY2(!one.isNull(), qPrintable(renderer.error()));
        const int value = one.pixelColor(64, 64).red();
        // gray=.502, unit alpha=.25*.502: one voxel emits 16/255.
        QVERIFY(std::abs(value - 16) <= 2);
        QCOMPARE(one.pixelColor(5, 5).red(), 0);
        auto duplicate = std::make_shared<slices::Image>(*image);
        duplicate->plane.axis = 1;
        auto overlap = draw({image, duplicate}, 1, false);
        QVERIFY2(!overlap.isNull(), qPrintable(renderer.error()));
        QVERIFY(std::abs(overlap.pixelColor(64, 64).red() - value) <= 1);
        auto interactive = draw({image}, 1, true);
        QVERIFY(!interactive.isNull());
        QVERIFY(std::abs(interactive.pixelColor(64, 64).red() - value) <= 1);
        auto hidden = draw({image}, .25, false);
        QVERIFY(!hidden.isNull());
        QCOMPARE(hidden.pixelColor(64, 64).red(), 0);
        auto thick = std::make_shared<slices::Image>(*image);
        thick->slab->origin = {-1, -1, -1};
        thick->slab->through = {0, 0, 2};
        auto two = draw({thick}, 1, false);
        QVERIFY(!two.isNull());
        QVERIFY(std::abs(two.pixelColor(64, 64).red() - 30) <= 2);
    }
};
QTEST_MAIN(Fiber3DVolumeTest)
#include "test_fiber_3d_volume.moc"
