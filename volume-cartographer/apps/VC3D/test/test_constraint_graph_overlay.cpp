#include <QtTest>
#include <QGraphicsScene>
#include <QGraphicsPathItem>
#include <QPainter>
#include <QPointer>
#include "../overlays/OverlayBatchItem.hpp"
#include "../ConstraintGraphSelectionBar.hpp"
#include "../ConstraintGraphPatchColors.hpp"
#include "../volume_viewers/SurfaceComparison.hpp"
#include <QVBoxLayout>
#include <set>

namespace {
QImage render(QGraphicsScene& scene) {
    QImage image(120,60,QImage::Format_ARGB32_Premultiplied);image.fill(Qt::black);
    QPainter painter(&image);scene.render(&painter,QRectF(0,0,120,60),QRectF(0,0,120,60));
    return image;
}
QPen dotPen() {QPen pen(QColor("#ffba45"));pen.setWidthF(4);pen.setCapStyle(Qt::RoundCap);return pen;}
}
class ConstraintGraphOverlayTest:public QObject {
    Q_OBJECT
private slots:
    void committedTintIsSubtleAndDoesNotTintProposalOrHoles(){
        QImage fiber(3,1,QImage::Format_RGB32);fiber.fill(QColor(80,80,80));
        QImage patch(3,1,QImage::Format_RGB32);patch.fill(QColor(128,128,128));
        const auto original=patch.copy();
        cv::Mat_<uint8_t> coverage(1,3,uint8_t(1));coverage(0,2)=0;
        ConstraintGraphPatchColors colors;const auto tint=colors.contextTint("accepted");
        QCOMPARE(tint.rgb(),colors.color("accepted").rgb());QCOMPARE(tint.alpha(),10);
        QCOMPARE(colors.contextTint("accepted",25).alpha(),64);
        QCOMPARE(vc3d::blendSurfaceImages(fiber,patch,coverage,1,colors.contextTint("accepted",0)),
                 vc3d::blendSurfaceImages(fiber,patch,coverage,1));
        const auto fullTint=vc3d::blendSurfaceImages(fiber,patch,coverage,1,colors.contextTint("accepted",100));
        QCOMPARE(fullTint.pixel(0,0),colors.color("accepted").rgb());
        QCOMPARE(fullTint.pixel(2,0),fiber.pixel(2,0));
        auto tinted=vc3d::blendSurfaceImages(fiber,patch,coverage,1,tint);
        QVERIFY(tinted.pixel(0,0)!=patch.pixel(0,0));
        for(int channel:{16,8,0})QVERIFY(std::abs(int((tinted.pixel(0,0)>>channel)&255)-128)<=10);
        QCOMPARE(tinted.pixel(2,0),fiber.pixel(2,0));
        QCOMPARE(vc3d::blendSurfaceImages(fiber,patch,coverage,0,tint),fiber);
        cv::Mat_<uint8_t> proposalCoverage(1,3,uint8_t(0));proposalCoverage(0,0)=1;
        std::vector<vc3d::SurfaceImageLayer> layers{{patch,coverage,1,tint},{patch,proposalCoverage,1,{}}};
        const auto image=vc3d::blendSurfaceLayers(fiber,layers);
        QCOMPARE(image.pixel(0,0),patch.pixel(0,0)); // Active proposal remains grayscale.
        QCOMPARE(image.pixel(1,0),tinted.pixel(1,0));
        QCOMPARE(image.pixel(2,0),fiber.pixel(2,0));
        // Reopening a committed layer must remove its tint completely.
        layers[0].tint=QColor{};QCOMPARE(vc3d::blendSurfaceLayers(fiber,layers),vc3d::blendSurfaceImages(fiber,patch,coverage,1));
        QCOMPARE(patch,original);
    }
    void manyReviewLayersMatchSequentialBlendingExactly(){
        QImage base(17,11,QImage::Format_ARGB32);base.fill(qRgba(27,113,208,61));
        const auto original=base.copy();
        std::vector<QImage> images;std::vector<cv::Mat_<uint8_t>> masks;
        for(int i=0;i<33;++i){
            QImage image(base.size(),QImage::Format_RGB32);
            cv::Mat_<uint8_t> storage(13,21,uint8_t(0));auto mask=storage(cv::Rect(2,1,17,11));
            for(int y=0;y<11;++y)for(int x=0;x<17;++x){
                image.setPixel(x,y,qRgb((i*31+x)%256,(y*17+i)%256,(x*y+i*7)%256));
                mask(y,x)=uint8_t((x+y+i)%3!=0);
            }
            images.push_back(image);masks.push_back(mask);
        }
        std::vector<vc3d::SurfaceImageLayer> layers;
        auto expected=base.convertToFormat(QImage::Format_RGB32);
        for(size_t i=0;i<images.size();++i){
            const float opacity=(i%4)*.25f;
            layers.push_back({images[i],masks[i],opacity});
            for(int y=0;y<11;++y)for(int x=0;x<17;++x)if(masks[i](y,x))
                expected.setPixel(x,y,vc3d::alphaBlendArgb(expected.pixel(x,y),images[i].pixel(x,y),opacity));
        }
        QCOMPARE(vc3d::blendSurfaceLayers(base,layers),expected);
        QCOMPARE(base,original);
        QCOMPARE(vc3d::blendSurfaceImages(base,images.front(),masks.front(),0),base.convertToFormat(QImage::Format_RGB32));
    }
    void reviewLayersHaveIndependentOpacityAndStableOrder(){
        QImage fiber(3,1,QImage::Format_RGB32);fiber.fill(Qt::white);
        QImage committed(3,1,QImage::Format_RGB32);committed.fill(Qt::black);
        QImage proposal(3,1,QImage::Format_RGB32);proposal.fill(Qt::red);
        cv::Mat_<uint8_t> oldCoverage(1,3,uint8_t(1)),newCoverage(1,3,uint8_t(1));
        oldCoverage(0,2)=0;newCoverage(0,1)=0;
        std::vector<vc3d::SurfaceImageLayer> layers{{committed,oldCoverage,.2f},{proposal,newCoverage,.5f}};
        const auto image=vc3d::blendSurfaceLayers(fiber,layers);
        QCOMPARE(image.pixelColor(0,0),QColor(230,102,102));
        QCOMPARE(image.pixelColor(1,0),QColor(204,204,204));
        QCOMPARE(image.pixelColor(2,0),QColor(255,128,128));
        layers[1].opacity=0;
        QCOMPARE(vc3d::blendSurfaceLayers(fiber,layers),vc3d::blendSurfaceImages(fiber,committed,oldCoverage,.2f));
        layers[0].opacity=0;QCOMPARE(vc3d::blendSurfaceLayers(fiber,layers),fiber);
        layers[1].opacity=1;QCOMPARE(vc3d::blendSurfaceLayers(fiber,layers).pixelColor(0,0),QColor(Qt::red));
    }
    void regionLimitedBlendMatchesWholeImageBlend(){
        QImage base(9,7,QImage::Format_RGB32);
        for(int y=0;y<7;++y)for(int x=0;x<9;++x)base.setPixel(x,y,qRgb(x*20,y*30,(x+y)*10));
        QImage patch(9,7,QImage::Format_RGB32);patch.fill(QColor(200,40,90));
        cv::Mat_<uint8_t> coverage(7,9,uint8_t(0));
        for(int y=2;y<5;++y)for(int x=3;x<7;++x)coverage(y,x)=1;
        const auto whole=vc3d::blendSurfaceImages(base,patch,coverage,.35f,QColor(10,200,30,40));
        auto limited=base;const cv::Rect region(3,2,4,3);
        vc3d::blendSurfaceImageInPlace(limited,patch,coverage,.35f,QColor(10,200,30,40),&region);
        QCOMPARE(limited,whole);
        // A region beyond the image is clipped; an empty region blends nothing.
        auto clipped=base;const cv::Rect large(-5,-5,100,100);
        vc3d::blendSurfaceImageInPlace(clipped,patch,coverage,.35f,QColor(10,200,30,40),&large);
        QCOMPARE(clipped,whole);
        auto none=base;const cv::Rect empty;
        vc3d::blendSurfaceImageInPlace(none,patch,coverage,.35f,{},&empty);
        QCOMPARE(none,base);
    }
    void comparisonOpacityPreservesHolesAndBlackPixels(){
        QImage base(3,1,QImage::Format_RGB32);base.fill(Qt::white);
        QImage patch(3,1,QImage::Format_RGB32);patch.fill(Qt::black);
        patch.setPixelColor(2,0,Qt::red);
        cv::Mat_<uint8_t> coverage(1,3,uint8_t(1));coverage(0,1)=0;
        QCOMPARE(vc3d::blendSurfaceImages(base,patch,coverage,0),base);
        auto mixed=vc3d::blendSurfaceImages(base,patch,coverage,.5f);
        QCOMPARE(mixed.pixelColor(0,0),QColor(128,128,128));
        QCOMPARE(mixed.pixelColor(1,0),QColor(Qt::white));
        auto opaque=vc3d::blendSurfaceImages(base,patch,coverage,1);
        QCOMPARE(opaque.pixelColor(0,0),QColor(Qt::black));
        QCOMPARE(opaque.pixelColor(1,0),QColor(Qt::white));
        QCOMPARE(opaque.pixelColor(2,0),QColor(Qt::red));
        QCOMPARE(vc3d::blendSurfaceImages(base,patch,{},.5f),base);
    }
    void patchColorsRemainDistinctAndStable(){
        ConstraintGraphPatchColors colors;auto first=colors.color("patch:a");
        std::set<QRgb> seen;
        for(int i=0;i<100;++i)QVERIFY(seen.insert(colors.color("patch:"+std::to_string(i)).rgb()).second);
        QCOMPARE(colors.color("patch:a"),first);
        QVERIFY(colors.color("patch:b")!=first);
    }
    void selectionBarCanBePlacedAndStaysInViewport(){
        QWidget viewport;viewport.resize(600,400);
        ConstraintGraphSelectionBar bar(&viewport);auto* layout=new QVBoxLayout(&bar);auto* grip=bar.makeGrip();layout->addWidget(grip);
        viewport.show();bar.show();bar.placeNear({40,50});QCOMPARE(bar.pos(),QPoint(40,50));
        QTest::mousePress(grip,Qt::LeftButton,Qt::NoModifier,QPoint(10,10));
        QTest::mouseMove(grip,QPoint(90,60));QTest::mouseRelease(grip,Qt::LeftButton,Qt::NoModifier,QPoint(10,10));
        const auto placed=bar.pos();QVERIFY(placed!=QPoint(40,50));
        bar.placeNear({200,200});QCOMPARE(bar.pos(),placed);
        viewport.resize(bar.width()+10,bar.height()+10);QCoreApplication::processEvents();
        QVERIFY(viewport.rect().contains(bar.geometry()));
    }

    void retainedGeometryAndConstantSizeDots(){
        QGraphicsScene scene;
        QPointer<OverlayBatchItem> root=new OverlayBatchItem;root->setFlag(QGraphicsItem::ItemHasNoContents);scene.addItem(root);
        QPainterPath path;path.moveTo(0,0);path.lineTo(20,0);
        auto* line=new QGraphicsPathItem(path,root);QPen green(Qt::green);green.setCosmetic(true);line->setPen(green);
        root->setTransform(QTransform(2,0,0,2,20,20));
        QPointer<OverlayBatchItem> dots=new OverlayBatchItem;scene.addItem(dots);dots->setZValue(25);
        dots->setUniformPoints({{20,20},{60,20}},dotPen());
        auto image=render(scene);
        QCOMPARE(image.pixelColor(20,20),dotPen().color());QCOMPARE(image.pixelColor(35,20),QColor(Qt::green));
        QVERIFY(image.pixelColor(23,20)!=dotPen().color());
        root->setTransform(QTransform(4,0,0,4,20,20));dots->setUniformPoints({{20,20},{100,20}},dotPen());
        image=render(scene);
        QCOMPARE(image.pixelColor(100,20),dotPen().color());QCOMPARE(image.pixelColor(80,20),QColor(Qt::green));
        QVERIFY(image.pixelColor(103,20)!=dotPen().color());
        // Clearing a surface deletes its scene; cache handles must become null
        // instead of leaving dangling pointers to retained geometry or dots.
        scene.clear();QVERIFY(!root);QVERIFY(!dots);
    }
    void replacingBatchModeClearsPreviousMarks(){
        QGraphicsScene scene;auto* item=new OverlayBatchItem;scene.addItem(item);
        item->setUniformPoints({{20,20}},dotPen());
        item->setPointCommands({{{50,20},2,QPen(Qt::NoPen),QBrush(Qt::cyan),QRectF(47,17,6,6)}});
        auto image=render(scene);QCOMPARE(image.pixelColor(20,20),QColor(Qt::black));QCOMPARE(image.pixelColor(50,20),QColor(Qt::cyan));
        item->setUniformPoints({{80,20}},dotPen());
        image=render(scene);QCOMPARE(image.pixelColor(50,20),QColor(Qt::black));QCOMPARE(image.pixelColor(80,20),dotPen().color());
        item->setLineCommands({});QVERIFY(item->boundingRect().isEmpty());
        QCOMPARE(render(scene).pixelColor(80,20),QColor(Qt::black));
    }
};
QTEST_MAIN(ConstraintGraphOverlayTest)
#include "test_constraint_graph_overlay.moc"
