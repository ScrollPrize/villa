#include "volume_viewers/IntersectionLayerItem.hpp"

#include <QGraphicsPathItem>
#include <QGraphicsRectItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QImage>
#include <QtTest/QtTest>

#include <cmath>
#include <random>

namespace {

constexpr int kViewSize = 320;

// Disjoint short segments, one subpath each, as renderIntersections builds.
QPainterPath makeSegments(int count, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    QPainterPath path;
    double x = 40.0;
    double y = 40.0;
    double angle = 0.0;
    for (int i = 0; i < count; ++i) {
        if (i % 60 == 0) {
            x = 20.0 + u(rng) * (kViewSize - 40.0);
            y = 20.0 + u(rng) * (kViewSize - 40.0);
            angle = u(rng) * 6.28318;
        }
        angle += (u(rng) - 0.5) * 0.4;
        const double nx = x + std::cos(angle) * 3.0;
        const double ny = y + std::sin(angle) * 3.0;
        path.moveTo(x, y);
        path.lineTo(nx, ny);
        x = nx;
        y = ny;
    }
    return path;
}

QPen cosmeticPen(QColor color, qreal width)
{
    QPen pen(color);
    pen.setWidthF(width);
    pen.setCapStyle(Qt::FlatCap);
    pen.setJoinStyle(Qt::RoundJoin);
    pen.setCosmetic(true);
    return pen;
}

// Mirrors the chunked viewer's view configuration.
void configureView(QGraphicsView& view)
{
    view.setRenderHint(QPainter::Antialiasing, false);
    view.setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    view.setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    view.setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    view.setFrameShape(QFrame::NoFrame);
    view.setFixedSize(kViewSize, kViewSize);
    view.setSceneRect(0, 0, kViewSize, kViewSize);
}

int maxChannelDifference(const QImage& a, const QImage& b)
{
    const QImage ia = a.convertToFormat(QImage::Format_ARGB32);
    const QImage ib = b.convertToFormat(QImage::Format_ARGB32);
    if (ia.size() != ib.size()) {
        return 256;
    }
    int worst = 0;
    for (int y = 0; y < ia.height(); ++y) {
        const auto* ra = reinterpret_cast<const QRgb*>(ia.constScanLine(y));
        const auto* rb = reinterpret_cast<const QRgb*>(ib.constScanLine(y));
        for (int x = 0; x < ia.width(); ++x) {
            worst = std::max({worst,
                              std::abs(qRed(ra[x]) - qRed(rb[x])),
                              std::abs(qGreen(ra[x]) - qGreen(rb[x])),
                              std::abs(qBlue(ra[x]) - qBlue(rb[x])),
                              std::abs(qAlpha(ra[x]) - qAlpha(rb[x]))});
        }
    }
    return worst;
}

}  // namespace

class IntersectionLayerItemTest : public QObject
{
    Q_OBJECT

private slots:
    void cachedPaintMatchesDirectPaint();
    void cacheIsReusedUntilInputsChange();
    void smallGroupsAreDrawnDirectly();
    void shapeDoesNotCaptureItemAt();
};

void IntersectionLayerItemTest::cachedPaintMatchesDirectPaint()
{
    const QPainterPath red = makeSegments(6000, 1);
    const QPainterPath blue = makeSegments(6000, 2);
    const QPen redPen = cosmeticPen(QColor(255, 40, 40, 122), 2.2);
    const QPen bluePen = cosmeticPen(QColor(40, 90, 255, 122), 2.2);
    // Non-identity item transform, as updateIntersectionPreviewTransform sets.
    const QTransform preview(1.25, 0, 0, 0, 1.25, 0, -30, 12, 1);

    auto addBackground = [](QGraphicsScene& scene) {
        auto* bg = scene.addRect(0, 0, kViewSize, kViewSize, Qt::NoPen, QBrush(QColor(90, 90, 90)));
        bg->setZValue(0);
    };

    QGraphicsScene directScene;
    directScene.setItemIndexMethod(QGraphicsScene::NoIndex);
    addBackground(directScene);
    for (const auto& [path, pen] : {std::pair{red, redPen}, std::pair{blue, bluePen}}) {
        auto* item = directScene.addPath(path, pen);
        item->setZValue(100);
        item->setTransform(preview);
    }
    QGraphicsView directView(&directScene);
    configureView(directView);

    QGraphicsScene cachedScene;
    cachedScene.setItemIndexMethod(QGraphicsScene::NoIndex);
    addBackground(cachedScene);
    auto* layer = new IntersectionLayerItem();
    layer->setEntries({{red, redPen}, {blue, bluePen}});
    layer->setZValue(100);
    layer->setTransform(preview);
    cachedScene.addItem(layer);
    QGraphicsView cachedView(&cachedScene);
    configureView(cachedView);

    const QImage direct = directView.grab().toImage();
    const QImage cached = cachedView.grab().toImage();
    QVERIFY(layer->hasCachedRaster());
    QCOMPARE(layer->rasterizationCount(), 1);
    int strokedPixels = 0;
    for (int y = 0; y < direct.height(); ++y) {
        for (int x = 0; x < direct.width(); ++x) {
            strokedPixels += direct.pixelColor(x, y) != QColor(90, 90, 90);
        }
    }
    QVERIFY2(strokedPixels > 5000, "reference image has no strokes to compare");
    // Compositing the pre-blended layer can differ from sequential blending
    // by 8-bit rounding only.
    QVERIFY2(maxChannelDifference(direct, cached) <= 2,
             qPrintable(QStringLiteral("max channel difference %1")
                            .arg(maxChannelDifference(direct, cached))));

    // A cache hit must produce the same pixels as the first paint.
    const QImage cachedAgain = cachedView.grab().toImage();
    QCOMPARE(layer->rasterizationCount(), 1);
    QCOMPARE(maxChannelDifference(cached, cachedAgain), 0);
}

void IntersectionLayerItemTest::cacheIsReusedUntilInputsChange()
{
    QGraphicsScene scene;
    scene.setItemIndexMethod(QGraphicsScene::NoIndex);
    auto* layer = new IntersectionLayerItem();
    layer->setEntries({{makeSegments(5000, 3), cosmeticPen(Qt::green, 2.2)}});
    scene.addItem(layer);
    auto* crosshair = scene.addEllipse(-6, -6, 12, 12, QPen(Qt::cyan));
    crosshair->setZValue(120);
    QGraphicsView view(&scene);
    configureView(view);

    (void)view.grab();
    QCOMPARE(layer->rasterizationCount(), 1);

    // Crosshair moves repaint the whole viewport but must not re-stroke.
    for (int i = 0; i < 5; ++i) {
        crosshair->setPos(40 + i * 10, 60);
        (void)view.grab();
    }
    QCOMPARE(layer->rasterizationCount(), 1);

    layer->setTransform(QTransform::fromTranslate(5, 0));
    (void)view.grab();
    QCOMPARE(layer->rasterizationCount(), 2);

    layer->setEntries({{makeSegments(5000, 4), cosmeticPen(Qt::green, 2.2)}});
    (void)view.grab();
    QCOMPARE(layer->rasterizationCount(), 3);

    view.setFixedSize(kViewSize / 2, kViewSize / 2);
    (void)view.grab();
    QCOMPARE(layer->rasterizationCount(), 4);
}

void IntersectionLayerItemTest::smallGroupsAreDrawnDirectly()
{
    QGraphicsScene scene;
    auto* layer = new IntersectionLayerItem();
    layer->setEntries({{makeSegments(10, 5), cosmeticPen(Qt::yellow, 2.2)}});
    scene.addItem(layer);
    QGraphicsView view(&scene);
    configureView(view);

    const QImage image = view.grab().toImage();
    QVERIFY(!layer->hasCachedRaster());
    QCOMPARE(layer->rasterizationCount(), 0);
    bool anyYellow = false;
    for (int y = 0; y < image.height() && !anyYellow; ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const QColor c = image.pixelColor(x, y);
            if (c.red() > 200 && c.green() > 200 && c.blue() < 50) {
                anyYellow = true;
                break;
            }
        }
    }
    QVERIFY(anyYellow);
}

void IntersectionLayerItemTest::shapeDoesNotCaptureItemAt()
{
    QGraphicsScene scene;
    scene.setItemIndexMethod(QGraphicsScene::NoIndex);
    auto* below = scene.addRect(0, 0, kViewSize, kViewSize, Qt::NoPen, QBrush(Qt::black));
    below->setZValue(0);
    QPainterPath line;
    line.moveTo(0, kViewSize / 2.0);
    line.lineTo(kViewSize, kViewSize / 2.0);
    auto* layer = new IntersectionLayerItem();
    layer->setEntries({{line, cosmeticPen(Qt::red, 4.0)}});
    layer->setZValue(100);
    scene.addItem(layer);
    QGraphicsView view(&scene);
    configureView(view);

    QCOMPARE(view.itemAt(QPoint(kViewSize / 2, kViewSize / 2)), below);
}

QTEST_MAIN(IntersectionLayerItemTest)
#include "test_intersection_layer_item.moc"
