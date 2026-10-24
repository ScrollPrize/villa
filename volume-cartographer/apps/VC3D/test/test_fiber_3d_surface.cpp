#include "fibers3d/Fiber3DSurfaceMesh.hpp"
#include "fibers3d/Fiber3DSurfaceRenderer.hpp"
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QTest>
#include <stdexcept>

using namespace vc3d::fibers3d;
class Fiber3DSurfaceTest : public QObject {
    Q_OBJECT
    static cv::Mat_<cv::Vec3f> grid(int n) {
        cv::Mat_<cv::Vec3f> p(n,n);
        for(int y=0;y<n;++y) for(int x=0;x<n;++x) p(y,x)={float(100+x),float(200+y),300.f};
        return p;
    }
private slots:
    void holesCoordinatesAndBudget() {
        std::atomic_bool cancelled{false};
        auto p=grid(5); const auto original=p.clone();
        auto full=surface::build(p,1<<20,cancelled);
        QCOMPARE(full->indices.size()/3,size_t(32));
        QCOMPARE(full->bounds.lower, (Point{100,200,300}));
        QCOMPARE(full->bounds.upper, (Point{104,204,300}));
        QCOMPARE(cv::norm(p-original),0.);
        p(2,2)={-1,-1,-1};
        auto hole=surface::build(p,1<<20,cancelled);
        QCOMPARE(hole->indices.size()/3,size_t(24));
        for(size_t i=0;i<hole->indices.size();i+=3) {
            float minX=INFINITY,maxX=-INFINITY,minY=INFINITY,maxY=-INFINITY;
            for(int j=0;j<3;++j) {
                auto& v=hole->vertices[hole->indices[i+j]];
                minX=std::min(minX,v.position[0]); maxX=std::max(maxX,v.position[0]);
                minY=std::min(minY,v.position[1]); maxY=std::max(maxY,v.position[1]);
                QCOMPARE(v.normal[0],0.f); QCOMPARE(v.normal[1],0.f); QVERIFY(v.normal[2]>0);
            }
            QVERIFY(maxX-minX<=1 && maxY-minY<=1);
            QVERIFY(!(minX<3 && maxX>1 && minY<3 && maxY>1));
        }
        p(2,2)={102,202,NAN};
        QCOMPARE(surface::build(p,1<<20,cancelled)->indices.size(),hole->indices.size());
        QVERIFY_THROWS_EXCEPTION(std::runtime_error,surface::build(p,32,cancelled));
        cancelled=true; QVERIFY(!surface::build(p,1<<20,cancelled));
        cancelled=false; p.setTo(cv::Vec3f(100,200,300));
        QVERIFY(surface::build(p,1<<20,cancelled)->indices.empty());
    }
    void nativeRendering() {
        QOpenGLContext context; QOffscreenSurface offscreen;
        QSurfaceFormat format; format.setVersion(3,3); format.setProfile(QSurfaceFormat::CoreProfile);
        format.setDepthBufferSize(24); context.setFormat(format);
        if(!context.create()) QSKIP("OpenGL 3.3 context unavailable");
        offscreen.setFormat(context.format()); offscreen.create();
        if(!context.makeCurrent(&offscreen)) QSKIP("Offscreen context unavailable");
        QOpenGLFunctions_3_3_Core gl; QVERIFY(gl.initializeOpenGLFunctions());
        std::atomic_bool cancelled{false}; auto p=grid(7); p(3,3)={-1,-1,-1};
        auto mesh=surface::build(p,1<<20,cancelled);
        QMatrix4x4 matrix; matrix.ortho(-1.f,7.f,-1.f,7.f,-10.f,10.f);
        for(int samples : {0,4}) {
            QOpenGLFramebufferObjectFormat fmt; fmt.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil); fmt.setSamples(samples);
            QOpenGLFramebufferObject fbo(128,128,fmt); QVERIFY(fbo.isValid());
            Fiber3DSurfaceRenderer renderer;
            auto draw=[&](std::shared_ptr<const surface::Mesh> m,float opacity,float depth=1) {
                fbo.bind(); gl.glViewport(0,0,128,128); gl.glDisable(GL_SCISSOR_TEST); gl.glDepthMask(GL_TRUE);
                gl.glClearColor(0,0,0,1); gl.glClearDepth(depth); gl.glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
                renderer.draw(m,matrix,mesh->origin,{},opacity); gl.glFinish(); return fbo.toImage();
            };
            auto opaque=draw(mesh,1); QVERIFY2(renderer.error().isEmpty(),qPrintable(renderer.error()));
            QVERIFY(opaque.pixelColor(32,32).green()>100);
            QCOMPARE(opaque.pixelColor(64,64).green(),0); // real hole survives rasterization
            auto half=draw(mesh,.5);
            int sum=0,full=0;
            for(int y=28;y<36;++y) for(int x=28;x<36;++x) {sum+=half.pixelColor(x,y).green(); full+=opaque.pixelColor(x,y).green();}
            QVERIFY(sum>full*.35 && sum<full*.65);
            QCOMPARE(draw(mesh,0).pixelColor(32,32).green(),0);
            QCOMPARE(draw(mesh,1,.25).pixelColor(32,32).green(),0);
            QCOMPARE(draw({},1).pixelColor(32,32).green(),0); // unload removes old GPU geometry
            QVERIFY(draw(mesh,1).pixelColor(32,32).green()>100);
            QCOMPARE(gl.glGetError(),GLenum(GL_NO_ERROR));
        }
    }
};
QTEST_MAIN(Fiber3DSurfaceTest)
#include "test_fiber_3d_surface.moc"
