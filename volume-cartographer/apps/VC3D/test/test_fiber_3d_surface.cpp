#include "fibers3d/Fiber3DSurfaceMesh.hpp"
#include "fibers3d/Fiber3DSurfaceRenderer.hpp"
#include "fibers3d/Fiber3DSurfaceProximity.hpp"
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
    void indexMatchesExhaustiveTriangleSearch() {
        auto p=grid(20);p(5,5)={-1,-1,-1};p(6,5)={-1,-1,-1};
        std::atomic_bool cancelled{false};auto mesh=surface::build(p,1<<20,cancelled);auto index=surface::index(mesh,cancelled);
        QVERIFY(index->nodes.size()>1);
        auto scene=std::make_shared<Scene>();
        for(int i=0;i<40;++i){double x=95+(i*7)%32,y=195+(i*11)%32,z=296+i%10;Fiber f;f.runs={Run{{{x,y,z},{x+3,y+7,z+5}}}};scene->fibers.push_back(f);}
        const auto geometry=render::build(scene);const auto filtered=surface::nearby(*geometry,*index,2.25,{},cancelled);
        auto vertex=[&](uint32_t id){const auto&p=mesh->vertices[id].position;return Point{p[0]+mesh->origin[0],p[1]+mesh->origin[1],p[2]+mesh->origin[2]};};
        for(size_t fi=0;fi<scene->fibers.size();++fi){const auto&line=scene->fibers[fi].runs[0].points;bool keep=false;
            for(size_t i=0;i<mesh->indices.size();i+=3)if(surface::segmentTriangleDistance(line[0],line[1],vertex(mesh->indices[i]),vertex(mesh->indices[i+1]),vertex(mesh->indices[i+2]))<=2.25){keep=true;break;}
            QCOMPARE(filtered.includes(fi),keep);
        }
    }
    void proximityUsesBothSidesAndTriangleInteriors() {
        using surface::segmentTriangleDistance;
        const Point a{0,0,0},b{10,0,0},c{0,10,0};
        QCOMPARE(segmentTriangleDistance({2,2,-10},{2,2,10},a,b,c),0.);
        QCOMPARE(segmentTriangleDistance({2,2,2},{3,3,2},a,b,c),2.);
        QCOMPARE(segmentTriangleDistance({2,2,-2},{3,3,-2},a,b,c),2.);
        QCOMPARE(segmentTriangleDistance({-10,2,2},{20,2,2},a,b,c),2.);
        QCOMPARE(segmentTriangleDistance({-1,-1,0},{-1,-1,0},a,b,c),std::sqrt(2.));
        std::atomic_bool cancelled{false};auto p=grid(5);
        for(int y=0;y<5;++y)for(int x=0;x<5;++x)p(y,x)={float(100+10*x),float(200+10*y),300};
        p(2,2)={-1,-1,-1};auto mesh=surface::build(p,1<<20,cancelled);auto index=surface::index(mesh,cancelled);
        auto scene=std::make_shared<Scene>();
        auto add=[&](std::vector<Run> runs){Fiber f;f.runs=std::move(runs);scene->fibers.push_back(std::move(f));};
        add({Run{{{105,205,302},{115,205,302}}}});
        add({Run{{{105,205,298},{115,205,298}}}});
        add({Run{{{105,205,303},{115,205,303}}}});
        add({Run{{{105,205,290},{105,205,310}}}});
        add({Run{{{120,220,200},{120,220,400}}}}); // crosses a hole, not a triangle
        add({Run{{{50,205,302},{180,205,302}}}}); // both ends far outside the surface
        add({Run{{{90,190,302},{95,195,302}}}});
        add({Run{{{105,205,280},{105,205,290}}},Run{{{105,205,310},{105,205,320}}}}); // gap
        auto geometry=render::build(scene);
        auto near=surface::nearby(*geometry,*index,2,{},cancelled);
        QCOMPARE(near.fibers,(std::vector<uint8_t>{1,1,0,1,0,1,0,0}));
        auto zero=surface::nearby(*geometry,*index,0,{},cancelled);
        QCOMPARE(zero.fibers,(std::vector<uint8_t>{0,0,0,1,0,0,0,0}));
        auto mask=render::visibility(*geometry,{0,1,1,1,1,1,1,1});
        QCOMPARE(surface::nearby(*geometry,*index,2,mask,cancelled).fibers,(std::vector<uint8_t>{0,1,0,1,0,1,0,0}));
        // Clipping cannot keep a fibre because of a hidden part of its path.
        scene->clipBounds=Bounds{{0,0,310},{1000,1000,400}};geometry=render::build(scene);
        QCOMPARE(surface::nearby(*geometry,*index,2,{},cancelled).count,size_t(0));
        cancelled=true;QVERIFY_THROWS_EXCEPTION(std::runtime_error,surface::index(mesh,cancelled));
        QVERIFY_THROWS_EXCEPTION(std::runtime_error,surface::nearby(*geometry,*index,2,{},cancelled));
    }
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
