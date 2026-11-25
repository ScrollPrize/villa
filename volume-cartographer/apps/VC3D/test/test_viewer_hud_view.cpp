// The measurement overlays inside a real CVolumeViewerView: which clicks the line
// profile may take, what it reads from a frame with loading gaps, when a frame is not
// measured, and painting every overlay in panes down to a few pixels.

#include <QtTest/QtTest>

#include <QApplication>
#include <QGraphicsProxyWidget>
#include <QGraphicsScene>
#include <QMouseEvent>
#include <QPushButton>
#include <QSignalSpy>

#include "volume_viewers/CVolumeViewerView.hpp"

#include <cmath>
#include <memory>

namespace {

// A live frame the size of the viewport: every column holds its x coordinate,
// column `gapColumn` was not sampled (a loading gap drawn as a gray placeholder),
// and frame x/y are volume x/y on the slice z = 40.
struct FakeFrame {
    int gapColumn = 20;
    bool live = true;
    double voxelSizeUm = 0.0;
    std::uint64_t serial = 1;
    QString composite;
    // A flattened view: no plane map, and column `holeColumn` maps to the (-1, -1, -1)
    // a hole in the surface reports.
    bool flattened = false;
    int holeColumn = -1;
    bool samples = true;

    CVolumeViewerView::FrameResult operator()(const QSize& size) const
    {
        CVolumeViewerView::FrameResult out;
        if (!live) {
            out.unavailable = QStringLiteral("updating");
            return out;
        }
        CVolumeViewerView::MeasurementFrame f;
        f.serial = serial;
        f.values = cv::Mat_<std::uint8_t>(size.height(), size.width());
        f.coverage = cv::Mat_<std::uint8_t>(size.height(), size.width(), std::uint8_t(1));
        for (int y = 0; y < size.height(); ++y) {
            for (int x = 0; x < size.width(); ++x) {
                f.values(y, x) = static_cast<std::uint8_t>(x % 256);
            }
            if (gapColumn >= 0 && gapColumn < size.width()) {
                f.values(y, gapColumn) = 64;
                f.coverage(y, gapColumn) = 0;
            }
        }
        vc3d::hud::AffineMap map;
        map.origin = {0.0, 0.0, 40.0};
        map.dx = {1.0, 0.0, 0.0};
        map.dy = {0.0, 1.0, 0.0};
        if (!flattened) {
            f.planeMap = map;
        }
        f.toVolume = [hole = holeColumn](const QPointF& p) {
            if (hole >= 0 && std::abs(p.x() - hole) < 0.5) {
                return cv::Vec3f(-1.0f, -1.0f, -1.0f);
            }
            return cv::Vec3f(float(p.x()), float(p.y()), 40.0f);
        };
        f.voxelSizeUm = voxelSizeUm;
        f.composite = composite;
        f.samples = samples;
        out.frame = std::move(f);
        return out;
    }
};

void sendMove(QWidget* w, QPointF at, Qt::MouseButtons buttons)
{
    QMouseEvent ev(QEvent::MouseMove, at, w->mapToGlobal(at), Qt::NoButton, buttons, Qt::NoModifier);
    QApplication::sendEvent(w, &ev);
}

}  // namespace

class TestViewerHudView : public QObject
{
    Q_OBJECT

    std::unique_ptr<QGraphicsScene> _scene;
    std::unique_ptr<CVolumeViewerView> _view;
    QImage _fb;
    FakeFrame _frame;

    QWidget* vp() const { return _view->viewport(); }

    void showAt(int w, int h)
    {
        _view->resize(w, h);
        _scene->setSceneRect(0, 0, vp()->width(), vp()->height());
        _fb = QImage(std::max(1, vp()->width()), std::max(1, vp()->height()), QImage::Format_RGB32);
        _fb.fill(QColor(90, 90, 90));
        _view->setDirectFramebuffer(&_fb);
    }

private slots:
    void init()
    {
        _scene = std::make_unique<QGraphicsScene>();
        _view = std::make_unique<CVolumeViewerView>();
        _view->setScene(_scene.get());
        _view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        _view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        _view->setAlignment(Qt::AlignLeft | Qt::AlignTop);
        _view->setFrameShape(QFrame::NoFrame);  // the viewport is the whole pane
        _frame = FakeFrame{};
        _view->setMeasurementFrameSource([this] { return _frame(vp()->size()); });
        _view->show();
        QVERIFY(QTest::qWaitForWindowExposed(_view.get()));
        showAt(400, 300);
        vc3d::hud::Options options;
        options.lineProfile = true;
        _view->setHudOptions(options);
    }

    void cleanup()
    {
        _view.reset();
        _scene.reset();
    }

    void profileModeLeavesPlainClicksToTheTools()
    {
        QSignalSpy press(_view.get(), &CVolumeViewerView::sendMousePress);
        QSignalSpy click(_view.get(), &CVolumeViewerView::sendVolumeClicked);
        QSignalSpy release(_view.get(), &CVolumeViewerView::sendMouseRelease);
        QTest::mousePress(vp(), Qt::LeftButton, Qt::NoModifier, QPoint(100, 100));
        sendMove(vp(), QPointF(160, 100), Qt::LeftButton);
        QTest::mouseRelease(vp(), Qt::LeftButton, Qt::NoModifier, QPoint(160, 100));
        QCOMPARE(press.count(), 1);
        QCOMPARE(click.count(), 1);
        QCOMPARE(release.count(), 1);
        QVERIFY(!_view->profileLine());
    }

    void anArmedViewTakesOneDragForTheLine()
    {
        QSignalSpy press(_view.get(), &CVolumeViewerView::sendMousePress);
        QSignalSpy started(_view.get(), &CVolumeViewerView::profileLineDrawStarted);
        _view->armProfileLine(true);
        QVERIFY(_view->profileLineArmed());
        QTest::mousePress(vp(), Qt::LeftButton, Qt::NoModifier, QPoint(50, 60));
        sendMove(vp(), QPointF(150, 60), Qt::LeftButton);
        QTest::mouseRelease(vp(), Qt::LeftButton, Qt::NoModifier, QPoint(150, 60));
        QCOMPARE(started.count(), 1);
        QCOMPARE(press.count(), 0);
        QVERIFY(_view->profileLine());
        QCOMPARE(_view->profileLine()->p2(), QPointF(150, 60));
        QVERIFY(!_view->profileLineArmed());
        // The next click is the tools' again.
        QTest::mouseClick(vp(), Qt::LeftButton, Qt::NoModifier, QPoint(70, 70));
        QCOMPARE(press.count(), 1);
        QVERIFY(_view->profileLine());
    }

    void armingNeedsTheProfileOverlay()
    {
        _view->setHudOptions(vc3d::hud::Options{});
        _view->armProfileLine(true);
        QVERIFY(!_view->profileLineArmed());
    }

    void escapeDisarms()
    {
        _view->armProfileLine(true);
        QTest::keyClick(_view.get(), Qt::Key_Escape);
        QVERIFY(!_view->profileLineArmed());
        QSignalSpy press(_view.get(), &CVolumeViewerView::sendMousePress);
        QTest::mouseClick(vp(), Qt::LeftButton, Qt::NoModifier, QPoint(70, 70));
        QCOMPARE(press.count(), 1);
    }

    void escapeDuringADragKeepsTheWholeGestureFromTheTools()
    {
        QSignalSpy press(_view.get(), &CVolumeViewerView::sendMousePress);
        QSignalSpy click(_view.get(), &CVolumeViewerView::sendVolumeClicked);
        QSignalSpy release(_view.get(), &CVolumeViewerView::sendMouseRelease);
        QSignalSpy keys(_view.get(), &CVolumeViewerView::sendKeyPress);
        QSignalSpy keyReleases(_view.get(), &CVolumeViewerView::sendKeyRelease);
        QSignalSpy cancelled(_view.get(), &CVolumeViewerView::profileLineCancelled);
        _view->armProfileLine(true);
        QTest::mousePress(vp(), Qt::LeftButton, Qt::NoModifier, QPoint(50, 60));
        sendMove(vp(), QPointF(120, 60), Qt::LeftButton);
        // Escape, held long enough to repeat, then let go.
        QKeyEvent esc(QEvent::KeyPress, Qt::Key_Escape, Qt::NoModifier);
        QApplication::sendEvent(_view.get(), &esc);
        QKeyEvent repeatUp(QEvent::KeyRelease, Qt::Key_Escape, Qt::NoModifier, QString(), true);
        QApplication::sendEvent(_view.get(), &repeatUp);
        QKeyEvent repeatDown(QEvent::KeyPress, Qt::Key_Escape, Qt::NoModifier, QString(), true);
        QApplication::sendEvent(_view.get(), &repeatDown);
        QKeyEvent up(QEvent::KeyRelease, Qt::Key_Escape, Qt::NoModifier);
        QApplication::sendEvent(_view.get(), &up);
        QCOMPARE(cancelled.count(), 1);
        QVERIFY(!_view->profileLine());
        // The rest of the drag stays with the cancelled profile.
        sendMove(vp(), QPointF(160, 60), Qt::LeftButton);
        QTest::mouseRelease(vp(), Qt::LeftButton, Qt::NoModifier, QPoint(160, 60));
        QCOMPARE(press.count(), 0);
        QCOMPARE(click.count(), 0);
        QCOMPARE(release.count(), 0);
        // Measure and the other tools never saw the Escape, its repeats or release.
        QCOMPARE(keys.count(), 0);
        QCOMPARE(keyReleases.count(), 0);
        // Afterwards clicks and Escape are the tools' again.
        QTest::mouseClick(vp(), Qt::LeftButton, Qt::NoModifier, QPoint(70, 70));
        QCOMPARE(press.count(), 1);
        QCOMPARE(click.count(), 1);
        QTest::keyClick(_view.get(), Qt::Key_Escape);
        QCOMPARE(keys.count(), 1);
    }

    void escapeWhileArmedStaysWithTheProfile()
    {
        QSignalSpy keys(_view.get(), &CVolumeViewerView::sendKeyPress);
        QSignalSpy cancelled(_view.get(), &CVolumeViewerView::profileLineCancelled);
        _view->setProfileLine(QLineF(QPointF(10, 50), QPointF(30, 50)));
        _view->armProfileLine(true);
        QTest::keyClick(_view.get(), Qt::Key_Escape);
        QCOMPARE(cancelled.count(), 1);
        QCOMPARE(keys.count(), 0);
        QVERIFY(!_view->profileLineArmed());
        QVERIFY(_view->profileLine());  // only arming is cancelled; the drawn line stays
    }

    void embeddedControlsKeepTheirClicksWhileArmed()
    {
        auto* button = new QPushButton(QStringLiteral("Apply"));
        QSignalSpy clicked(button, &QPushButton::clicked);
        QGraphicsProxyWidget* proxy = _scene->addWidget(button);
        proxy->setPos(200, 200);
        proxy->resize(80, 30);
        _view->armProfileLine(true);
        const QPoint at = _view->mapFromScene(proxy->sceneBoundingRect().center());
        QTest::mouseClick(vp(), Qt::LeftButton, Qt::NoModifier, at);
        QCOMPARE(clicked.count(), 1);
        QVERIFY(!_view->profileLine());
        QVERIFY(_view->profileLineArmed());
    }

    void profileReadsOnlySampledPixels()
    {
        _view->setProfileLine(QLineF(QPointF(10, 50), QPointF(30, 50)));
        const auto r = _view->profileReadout();
        QVERIFY(r.available);
        QCOMPARE(int(r.values.size()), 21);
        QCOMPARE(r.values[0], 10.0f);
        QVERIFY(std::isnan(r.values[10]));  // column 20 is the gap; its 64 never shows
        QCOMPARE(r.values[11], 21.0f);
        // No stated voxel size: lengths are voxels, along the plane.
        QVERIFY(!r.physical);
        QVERIFY(!r.alongSurface);
        QVERIFY(r.length);
        QCOMPARE(*r.length, 20.0);
        QVERIFY(r.start && r.end);
        QCOMPARE((*r.start)[0], 10.0);
        QCOMPARE((*r.end)[2], 40.0);
    }

    void lengthsAreMicronsOnlyWithAStatedVoxelSize()
    {
        _frame.voxelSizeUm = 7.91;
        _view->setProfileLine(QLineF(QPointF(10, 50), QPointF(30, 50)));
        const auto r = _view->profileReadout();
        QVERIFY(r.physical);
        QVERIFY(std::abs(*r.length - 20.0 * 7.91) < 1e-9);
    }

    void aSurfaceHoleHasNoPositionOrLength()
    {
        _frame.flattened = true;
        _frame.holeColumn = 25;
        _view->setProfileLine(QLineF(QPointF(10, 50), QPointF(30, 50)));
        auto r = _view->profileReadout();
        QVERIFY(r.available);
        QVERIFY(r.alongSurface);
        QVERIFY(!r.length);              // the path crosses the hole
        QVERIFY(std::isnan(r.arc[15]));  // from the hole on
        QCOMPARE(r.arc[14], 14.0);
        QVERIFY(r.start && r.end);
        // A line that ends in the hole has no end position.
        _view->setProfileLine(QLineF(QPointF(10, 50), QPointF(25, 50)));
        r = _view->profileReadout();
        QVERIFY(r.start);
        QVERIFY(!r.end);
    }

    void missingSamplesAreCounted()
    {
        _view->setProfileLine(QLineF(QPointF(10, 50), QPointF(30, 50)));
        const auto r = _view->profileReadout();
        QCOMPARE(r.missing, 1);  // the gap column
        QVERIFY(r.complete);
    }

    void aSlabCompositeIsNotProfiled()
    {
        _frame.composite = QStringLiteral("max of 9 slices");
        _view->setProfileLine(QLineF(QPointF(10, 50), QPointF(30, 50)));
        const auto r = _view->profileReadout();
        QVERIFY(!r.available);
        QVERIFY(r.unavailable.contains(QStringLiteral("max of 9 slices")));
        QVERIFY(r.values.empty());
    }

    void aFrameWithoutItsMeasurementMaskIsNotMeasured()
    {
        // Rendered before an overlay that reads samples was on: positions only.
        _frame.samples = false;
        _view->setProfileLine(QLineF(QPointF(10, 50), QPointF(30, 50)));
        const auto r = _view->profileReadout();
        QVERIFY(!r.available);
        QCOMPARE(r.unavailable, QStringLiteral("updating"));
    }

    void aFrameThatIsNotLiveIsNotMeasured()
    {
        _view->setProfileLine(QLineF(QPointF(10, 50), QPointF(30, 50)));
        _frame.live = false;
        auto r = _view->profileReadout();
        QVERIFY(!r.available);
        QCOMPARE(r.unavailable, QStringLiteral("updating"));
        // A retained frame slid to follow a pan is not the live view either.
        _frame.live = true;
        _view->setDirectFramebufferMapping(1.0, QPointF(12.0, 0.0));
        r = _view->profileReadout();
        QVERIFY(!r.available);
        _view->setDirectFramebufferMapping(1.0, QPointF(0.0, 0.0));
        QVERIFY(_view->profileReadout().available);
    }

    void everyOverlayPaintsInSmallPanes()
    {
        vc3d::hud::Options options;
        options.histogram = true;
        options.lineProfile = true;
        options.grid = true;
        options.loupe = true;
        _view->setHudOptions(options);
        const QSize panes[] = {{400, 300}, {500, 300}, {284, 124}, {80, 80}, {44, 2}, {2, 44}, {600, 90}};
        for (const QSize& pane : panes) {
            for (double step : {0.0, 0.5, 2.5, 25.0}) {
                options.gridStep = step;
                _view->setHudOptions(options);
                showAt(pane.width(), pane.height());
                _view->setProfileLine(QLineF(QPointF(1, 1), QPointF(vp()->width() - 1, vp()->height() - 1)));
                sendMove(vp(), QPointF(vp()->width() / 2.0, vp()->height() / 2.0), Qt::NoButton);
                QCOMPARE(vp()->size(), pane);
                const QImage shot = vp()->grab().toImage();
                QVERIFY(!shot.isNull());
            }
        }
    }
};

QTEST_MAIN(TestViewerHudView)

#include "test_viewer_hud_view.moc"
