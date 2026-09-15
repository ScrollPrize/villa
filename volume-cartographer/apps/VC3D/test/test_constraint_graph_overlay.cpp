#include <QtTest>
#include <QGraphicsScene>
#include <QGraphicsPathItem>
#include <QPainter>
#include <QPointer>
#include "../overlays/OverlayBatchItem.hpp"
#include "../ConstraintGraphSelectionBar.hpp"
#include "../ConstraintGraphPatchColors.hpp"
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
