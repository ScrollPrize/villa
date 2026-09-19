#include "SpiralBrushPatch.hpp"
#include "SpiralPatchMode.hpp"
#include "SpiralPatchCells.hpp"
#include "SpiralPatchProjection.hpp"
#include "SpiralInputRows.hpp"
#include "SpiralInputFilter.hpp"
#include <QTest>
#include <QDir>
#include <QImage>
#include <opencv2/imgcodecs.hpp>
#include <cmath>

class SpiralBrushPatchTest : public QObject
{
    Q_OBJECT
private slots:
    void previewProjectionChecksWindingAndPreservesOriginal()
    {
        auto makeSurface = [](float z, float winding) {
            auto* points = new cv::Mat_<cv::Vec3f>(5, 5);
            for (int row = 0; row < 5; ++row)
                for (int col = 0; col < 5; ++col)
                    (*points)(row, col) = {100.0f + col * 10, 100.0f + row * 10, z};
            auto surface = std::make_shared<QuadSurface>(points, cv::Vec2f(0.1f, 0.1f));
            surface->setChannel("d", cv::Mat1f(5, 5, winding));
            return surface;
        };
        const auto source = makeSurface(100, 7);
        const auto target = makeSurface(102, 7);
        SurfacePatchIndex index;
        index.rebuild({source, target});
        const auto a = source->gridToSurface({1, 1});
        const auto b = source->gridToSurface({3, 3});
        QPainterPath original;
        original.addRect(QRectF(QPointF(a[0], a[1]), QPointF(b[0], b[1])).normalized());
        const auto saved = original;
        const auto mapped = vc3d::spiral::projectPatchShape(original, source, target, &index, 10);
        QVERIFY(mapped.has_value());
        QVERIFY(!mapped->isEmpty());
        target->setChannel("d", cv::Mat1f(5, 5, 8.0f));
        QVERIFY(!vc3d::spiral::projectPatchShape(original, source, target, &index, 10));
        QCOMPARE(original, saved);
    }
    void localRowsRetainIdentityColorErrorsAndRemoval()
    {
        const QJsonArray service{QJsonObject{{"id", "patch"}, {"kind", "patch"},
            {"state", "error"}, {"error", "invalid quad"}, {"removable", true}}};
        const QJsonArray local{QJsonObject{{"id", "patch"}, {"kind", "patch"},
            {"color", "#00ffff"}, {"local", false}},
            QJsonObject{{"id", "tiny"}, {"kind", "patch"}, {"state", "error"},
                {"error", "too small"}, {"removable", true}, {"local", true}}};
        const auto rows = vc3d::spiral::mergePatchDraftRows(service, local);
        QCOMPARE(rows.size(), 2);
        QCOMPARE(rows[0].toObject().value("error").toString(), QString("invalid quad"));
        QCOMPARE(rows[0].toObject().value("color").toString(), QString("#00ffff"));
        QVERIFY(rows[1].toObject().value("removable").toBool());
        QVERIFY(rows[1].toObject().value("dirty").toBool());
        QVERIFY(!rows[1].toObject().value("committed").toBool());
        QVERIFY(vc3d::spiral::inputVisible(rows[1].toObject(), "tiny", {}, false));
        const auto fresh = vc3d::spiral::mergePatchDraftRows({}, {
            QJsonObject{{"id", "new"}, {"kind", "patch"}, {"local", true}}});
        QVERIFY(vc3d::spiral::inputVisible(fresh[0].toObject(), "new", {}, false));
        QCOMPARE(vc3d::spiral::mergePatchDraftRows(service, local), rows);
    }
    void realScrollSelection()
    {
        const QString source = qEnvironmentVariable("SPIRAL_PATCH_REAL_INPUT");
        if (source.isEmpty()) QSKIP("Set SPIRAL_PATCH_REAL_INPUT to validate a real scroll patch");
        const cv::Mat1f x = cv::imread((source + "/x.tif").toStdString(), cv::IMREAD_UNCHANGED);
        const cv::Mat1f y = cv::imread((source + "/y.tif").toStdString(), cv::IMREAD_UNCHANGED);
        const cv::Mat1f z = cv::imread((source + "/z.tif").toStdString(), cv::IMREAD_UNCHANGED);
        QVERIFY(!x.empty());
        QCOMPARE(x.size(), y.size());
        QCOMPARE(x.size(), z.size());
        cv::Mat1b selected(x.rows, x.cols, uchar{0});
        const int split = x.rows * 2 / 3;
        for (int row = 0; row < x.rows; ++row)
            for (int col = 0; col < x.cols; ++col)
                if (std::isfinite(x(row, col)) && x(row, col) != -1
                    && std::isfinite(y(row, col)) && y(row, col) != -1
                    && std::isfinite(z(row, col)) && z(row, col) != -1
                    && (row < split || row > split + 3)) selected(row, col) = 1;
        const auto retained = vc3d::spiral::largestPatchQuadComponent(selected);
        QVERIFY(cv::countNonZero(retained) > 0);
        QVERIFY(cv::countNonZero(retained) < cv::countNonZero(selected));
        QCOMPARE(cv::countNonZero(retained != vc3d::spiral::largestPatchQuadComponent(retained)), 0);
        const QString output = qEnvironmentVariable("SPIRAL_PATCH_EVIDENCE_DIR");
        if (output.isEmpty()) return;
        QVERIFY(QDir().mkpath(output));
        for (const bool after : {false, true}) {
            QImage image(x.cols, x.rows, QImage::Format_RGB32);
            for (int row = 0; row < x.rows; ++row) {
                for (int col = 0; col < x.cols; ++col) {
                    QColor color = x(row, col) == -1 ? QColor(20, 20, 20) : QColor(95, 95, 95);
                    if ((after ? retained : selected)(row, col)) color = QColor(35, 220, 175);
                    image.setPixelColor(col, row, color);
                }
            }
            QVERIFY(image.scaledToHeight(732).save(output + (after ? "/selection-after.png" : "/selection-before.png")));
        }
        qInfo() << "Real patch" << source << "selected vertices" << cv::countNonZero(selected)
                << "retained vertices" << cv::countNonZero(retained);
    }
    void projectedCellsKeepLargestCompleteQuadComponent()
    {
        cv::Mat1b selected(10, 12, uchar{0});
        selected(cv::Rect(1, 1, 4, 4)).setTo(1);
        selected(cv::Rect(8, 7, 2, 2)).setTo(1);
        selected(0, 10) = 1;
        auto retained = vc3d::spiral::largestPatchQuadComponent(selected);
        QCOMPARE(cv::countNonZero(retained), 16);
        QCOMPARE(retained(2, 2), uchar{1});
        QCOMPARE(retained(7, 8), uchar{0});
    }
    void componentTiesAndTinySelectionsAreDeterministic()
    {
        cv::Mat1b selected(5, 5, uchar{0});
        selected(1, 1) = 1;
        QCOMPARE(cv::countNonZero(vc3d::spiral::largestPatchQuadComponent(selected)), 0);
        selected(cv::Rect(0, 0, 2, 2)).setTo(1);
        selected(cv::Rect(3, 3, 2, 2)).setTo(1);
        auto retained = vc3d::spiral::largestPatchQuadComponent(selected);
        QCOMPARE(cv::countNonZero(retained), 4);
        QCOMPARE(retained(0, 0), uchar{1});
        QCOMPARE(retained(3, 3), uchar{0});
    }
    void ctrlTapAndChords()
    {
        SpiralPatchMode mode;
        QKeyEvent down(QEvent::KeyPress, Qt::Key_Control, Qt::ControlModifier);
        QKeyEvent up(QEvent::KeyRelease, Qt::Key_Control, Qt::NoModifier);
        QKeyEvent repeat(QEvent::KeyPress, Qt::Key_Control, Qt::ControlModifier, {}, true);
        QVERIFY(!mode.observe(down));
        QVERIFY(!mode.active());
        mode.observe(repeat);
        QVERIFY(mode.observe(up));
        QVERIFY(mode.active());
        for (auto type : {QEvent::Wheel, QEvent::MouseButtonPress, QEvent::MouseButtonDblClick}) {
            mode.observe(down);
            QEvent chord(type);
            mode.observe(chord);
            QVERIFY(!mode.observe(up));
            QVERIFY(mode.active());
        }
        mode.observe(down);
        QKeyEvent chord(QEvent::KeyPress, Qt::Key_C, Qt::ControlModifier);
        mode.observe(chord);
        QVERIFY(!mode.observe(up));
        QVERIFY(mode.active());
        QEvent release(QEvent::MouseButtonRelease);
        mode.observe(release);
        mode.observe(down);
        QVERIFY(mode.observe(up));
        QVERIFY(!mode.active());
        mode.observe(down);
        QVERIFY(mode.observe(up));
        QVERIFY(mode.active());
        QKeyEvent escape(QEvent::KeyPress, Qt::Key_Escape, Qt::NoModifier);
        QVERIFY(mode.observe(escape));
        QVERIFY(!mode.active());
    }
    void interruptedTapDoesNotToggle()
    {
        SpiralPatchMode mode;
        QKeyEvent down(QEvent::KeyPress, Qt::Key_Control, Qt::ControlModifier);
        QKeyEvent up(QEvent::KeyRelease, Qt::Key_Control, Qt::NoModifier);
        for (auto type : {QEvent::FocusOut, QEvent::Leave, QEvent::WindowDeactivate}) {
            mode.observe(down);
            QEvent interruption(type);
            mode.observe(interruption);
            QVERIFY(!mode.observe(up));
            QVERIFY(!mode.active());
        }
    }
    void staleUploadPreservesEditAndColor()
    {
        SpiralBrushPatch patch;
        patch.id = "brush_test";
        patch.color = Qt::cyan;
        patch.shape.addRect(0, 0, 10, 10);
        patch.submitted();
        QVERIFY(!patch.removableLocally());
        patch.shape.addRect(10, 0, 10, 10);
        patch.changed();
        patch.staged = true;
        patch.accepted();
        QCOMPARE(patch.state, SpiralGestureState::Painted);
        QVERIFY(patch.shape.contains(QPointF(15, 5)));
        QVERIFY(!patch.acceptedShape.contains(QPointF(15, 5)));
        QCOMPARE(patch.color, QColor(Qt::cyan));
        QCOMPARE(patch.id, QString("brush_test"));
    }
    void staleFailureDoesNotMarkNewerEditReady()
    {
        SpiralBrushPatch patch;
        patch.shape.addRect(0, 0, 10, 10);
        patch.submitted();
        patch.changed();
        patch.failed("older save failed");
        QCOMPARE(patch.state, SpiralGestureState::Painted);
        QVERIFY(patch.error.isEmpty());
        QVERIFY(!patch.uploadInFlight);
        patch.submitted();
        patch.failed("current save failed");
        QCOMPARE(patch.state, SpiralGestureState::Ready);
        QCOMPARE(patch.error, QString("current save failed"));
    }
    void localRowsMatchAliasesWithoutReplacingServiceIdentity()
    {
        const QJsonArray service{QJsonObject{{"id", "service-id"}, {"alias", "brush-id"},
            {"kind", "patch"}, {"error", "apply failed"}}};
        const QJsonArray local{QJsonObject{{"id", "brush-id"}, {"kind", "patch"},
            {"color", "#00ffff"}, {"local", false}}};
        const auto rows = vc3d::spiral::mergePatchDraftRows(service, local);
        QCOMPARE(rows.size(), 1);
        QCOMPARE(rows[0].toObject().value("id").toString(), QString("service-id"));
        QCOMPARE(rows[0].toObject().value("error").toString(), QString("apply failed"));
        QCOMPARE(rows[0].toObject().value("color").toString(), QString("#00ffff"));
    }
    void restoreErasedPatchRecoversSelection()
    {
        SpiralBrushPatch patch;
        patch.shape.addRect(0, 0, 10, 10);
        patch.submitted();
        patch.accepted();
        patch.shape = {};
        patch.changed();
        patch.submitted();
        patch.setRemoved(true);
        patch.accepted();
        QVERIFY(!patch.visible());
        patch.setRemoved(false);
        QVERIFY(patch.visible());
        QVERIFY(patch.shape.contains(QPointF(5, 5)));
    }
    void completeErasureDistinguishesLocalFromAdded()
    {
        SpiralBrushPatch patch;
        patch.shape.addRect(0, 0, 10, 10);
        patch.shape = patch.shape.subtracted(patch.shape);
        QVERIFY(patch.emptyLocal());
        patch.staged = true;
        QVERIFY(!patch.emptyLocal());
        patch.changed();
        QCOMPARE(patch.state, SpiralGestureState::Painted);
    }
};
QTEST_MAIN(SpiralBrushPatchTest)
#include "test_spiral_brush_patch.moc"
