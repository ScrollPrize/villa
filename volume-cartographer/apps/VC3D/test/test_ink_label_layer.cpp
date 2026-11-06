#include "inklabels/InkLabelIO.hpp"
#include "inklabels/InkLabelLayer.hpp"

#include <QTemporaryDir>
#include <QTest>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <vector>

using namespace vc3d::inklabels;
namespace fs = std::filesystem;

namespace {

bool sameLabels(const cv::Mat_<uint8_t>& a, const cv::Mat_<uint8_t>& b)
{
    return a.size() == b.size() && cv::countNonZero(a != b) == 0;
}

// Pixels whose centres lie within `radius` of `centre`, counted independently
// of the layer.
int discPixelCount(cv::Point2f centre, float radius, cv::Size size)
{
    int count = 0;
    for (int y = 0; y < size.height; ++y) {
        for (int x = 0; x < size.width; ++x) {
            const float dx = x + 0.5f - centre.x;
            const float dy = y + 0.5f - centre.y;
            if (dx * dx + dy * dy <= radius * radius) {
                ++count;
            }
        }
    }
    return count;
}

bool stroke(InkLabelLayer& layer, Label value, float radius, std::vector<cv::Point2f> points)
{
    layer.beginStroke(value, radius);
    for (const auto& p : points) {
        layer.strokeTo(p);
    }
    return layer.endStroke();
}

// A deterministic pattern with all three states, painted through strokes.
InkLabelLayer roundTripPattern()
{
    InkLabelLayer layer(cv::Size(1500, 1200));
    // Two broad no-ink regions (negatives around letters).
    for (int y = 120; y <= 520; y += 40) {
        stroke(layer, Label::NoInk, 30.0f, {{100.0f, float(y)}, {1350.0f, float(y)}});
    }
    stroke(layer, Label::NoInk, 60.0f, {{200.0f, 800.0f}, {700.0f, 1050.0f}, {1200.0f, 800.0f}});
    // Ink strokes on top, partly outside the negatives.
    stroke(layer, Label::Ink, 6.0f, {{150.0f, 150.0f}, {600.0f, 480.0f}});
    stroke(layer, Label::Ink, 3.5f, {{700.0f, 200.0f}, {760.0f, 420.0f}, {820.0f, 200.0f}});
    stroke(layer, Label::Ink, 9.0f, {{1100.0f, 300.0f}, {1450.0f, 700.0f}});
    stroke(layer, Label::Ink, 2.0f, {{400.0f, 900.0f}, {1000.0f, 900.0f}});
    // Erase a gap back to unlabelled.
    stroke(layer, Label::Unlabelled, 12.0f, {{300.0f, 100.0f}, {300.0f, 560.0f}});
    return layer;
}

} // namespace

class InkLabelLayerTest : public QObject
{
    Q_OBJECT
private slots:
    void newLayerIsUnlabelled()
    {
        InkLabelLayer layer(cv::Size(64, 48));
        QVERIFY(layer.size() == cv::Size(64, 48));
        QCOMPARE(cv::countNonZero(layer.labels()), 0);
        QVERIFY(!layer.canUndo());
        QVERIFY(!layer.modified());
    }

    void inkStampCoversTheDisc()
    {
        for (const float radius : {0.75f, 3.0f, 7.5f}) {
            InkLabelLayer layer(cv::Size(40, 40));
            const cv::Point2f centre(20.3f, 18.7f);
            stroke(layer, Label::Ink, radius, {centre});
            QCOMPARE(cv::countNonZero(layer.labels() == 2), discPixelCount(centre, radius, layer.size()));
            QCOMPARE(cv::countNonZero(layer.labels() == 1), 0);
        }
        // Clipped at the image edge without touching anything outside.
        InkLabelLayer edge(cv::Size(10, 10));
        stroke(edge, Label::Ink, 4.0f, {{0.0f, 0.0f}});
        QCOMPARE(cv::countNonZero(edge.labels() == 2), discPixelCount({0.0f, 0.0f}, 4.0f, edge.size()));
    }

    void noInkAndEraseSetTheirValues()
    {
        InkLabelLayer layer(cv::Size(50, 50));
        stroke(layer, Label::NoInk, 10.0f, {{25.0f, 25.0f}});
        const int negatives = cv::countNonZero(layer.labels() == 1);
        QVERIFY(negatives > 0);
        stroke(layer, Label::Ink, 3.0f, {{25.0f, 25.0f}});
        QCOMPARE(cv::countNonZero(layer.labels() == 2), discPixelCount({25.0f, 25.0f}, 3.0f, layer.size()));
        QCOMPARE(cv::countNonZero(layer.labels() == 1),
                 negatives - discPixelCount({25.0f, 25.0f}, 3.0f, layer.size()));
        stroke(layer, Label::Unlabelled, 20.0f, {{25.0f, 25.0f}});
        QCOMPARE(cv::countNonZero(layer.labels()), 0);
        const auto counts = layer.counts();
        QCOMPARE(counts[0], std::size_t(2500));
    }

    void strokesSweepWithoutGaps()
    {
        InkLabelLayer layer(cv::Size(200, 40));
        stroke(layer, Label::Ink, 2.0f, {{10.0f, 20.0f}, {190.0f, 20.0f}});
        // Every column the line crosses is painted.
        for (int x = 10; x < 190; ++x) {
            QVERIFY2(layer.labels()(20, x) == 2, qPrintable(QString("gap at x=%1").arg(x)));
        }
    }

    void undoAndRedoRestoreEveryStroke()
    {
        InkLabelLayer layer(cv::Size(300, 260));
        std::vector<cv::Mat_<uint8_t>> states{layer.labels().clone()};
        for (int i = 0; i < 25; ++i) {
            const Label value = static_cast<Label>(i % 3 == 2 ? 0 : 1 + (i % 2));
            const float radius = 2.0f + float(i % 7) * 3.0f;
            const cv::Point2f a(10.0f + 11.0f * i, 20.0f + 7.0f * (i % 5));
            const cv::Point2f b(290.0f - 9.0f * i, 240.0f - 13.0f * (i % 4));
            if (stroke(layer, value, radius, {a, b})) {
                states.push_back(layer.labels().clone());
            }
        }
        const std::size_t undoable = layer.undoDepth();
        QCOMPARE(undoable, states.size() - 1);
        QVERIFY(undoable >= InkLabelLayer::kMinUndoStrokes);
        for (std::size_t k = 0; k < undoable; ++k) {
            layer.undo();
            QVERIFY2(sameLabels(layer.labels(), states[states.size() - 2 - k]),
                     qPrintable(QString("undo %1").arg(k + 1)));
        }
        QVERIFY(!layer.canUndo());
        for (std::size_t k = 0; k < undoable; ++k) {
            layer.redo();
            QVERIFY2(sameLabels(layer.labels(), states[states.size() - undoable + k]),
                     qPrintable(QString("redo %1").arg(k + 1)));
        }
        // A new stroke after an undo drops the redo history.
        layer.undo();
        stroke(layer, Label::Ink, 5.0f, {{150.0f, 130.0f}});
        QVERIFY(!layer.canRedo());
    }

    void minimumUndoDepthSurvivesTheMemoryBudget()
    {
        InkLabelLayer layer(cv::Size(512, 512));
        layer.setUndoMemoryBudget(1);
        for (int i = 0; i < 30; ++i) {
            stroke(layer, i % 2 ? Label::Ink : Label::NoInk, 40.0f, {{256.0f, 256.0f}});
        }
        QCOMPARE(layer.undoDepth(), InkLabelLayer::kMinUndoStrokes);
    }

    void cancelRevertsAnUnfinishedStroke()
    {
        InkLabelLayer layer(cv::Size(80, 80));
        stroke(layer, Label::NoInk, 20.0f, {{40.0f, 40.0f}});
        const cv::Mat_<uint8_t> before = layer.labels().clone();
        layer.beginStroke(Label::Ink, 5.0f);
        layer.strokeTo({10.0f, 10.0f});
        layer.strokeTo({70.0f, 70.0f});
        layer.cancelStroke();
        QVERIFY(sameLabels(layer.labels(), before));
        QCOMPARE(layer.undoDepth(), std::size_t(1));
    }

    void savedPairLoadsBackExactly()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const InkLabelLayer layer = roundTripPattern();
        const auto files = trainingLabelFiles(dir.path().toStdString(), "w047", 2);
        QCOMPARE(QString::fromStdString(files.inklabels.filename().string()), QString("w047_inklabels_v2.tif"));
        QCOMPARE(QString::fromStdString(files.supervisionMask.filename().string()),
                 QString("w047_supervision_mask_v2.tif"));
        writeTrainingLabels(files, layer.labels());
        QVERIFY(!fs::exists(files.inklabels.string() + ".tmp"));
        const auto loaded = readTrainingLabels(files);
        QVERIFY(sameLabels(loaded.labels, layer.labels()));
        QCOMPARE(loaded.inkOutsideSupervision, std::size_t(0));
        const auto counts = layer.counts();
        QVERIFY(counts[0] > 0 && counts[1] > 0 && counts[2] > 0);

        // The files hold exactly 0 and 255.
        const cv::Mat ink = cv::imread(files.inklabels.string(), cv::IMREAD_UNCHANGED);
        QCOMPARE(ink.type(), CV_8UC1);
        QCOMPARE(cv::countNonZero(ink == 255), static_cast<int>(counts[2]));
        QCOMPARE(cv::countNonZero(ink != 0 & ink != 255), 0);

        // Ink outside supervision (e.g. hand-edited files) loads as unlabelled and is counted.
        cv::Mat_<uint8_t> stray = ink.clone();
        cv::Mat_<uint8_t> supervision = cv::imread(files.supervisionMask.string(), cv::IMREAD_UNCHANGED);
        stray(5, 5) = 255;
        supervision(5, 5) = 0;
        const auto strayFiles = trainingLabelFiles(dir.path().toStdString(), "stray");
        QVERIFY(cv::imwrite(strayFiles.inklabels.string(), stray));
        QVERIFY(cv::imwrite(strayFiles.supervisionMask.string(), supervision));
        const auto strayLoaded = readTrainingLabels(strayFiles);
        QCOMPARE(strayLoaded.inkOutsideSupervision, std::size_t(1));
        QCOMPARE(int(strayLoaded.labels(5, 5)), 0);
    }

    void canvasMatchesTheRenderer()
    {
        // Tracer meshes store scale 0.05 as float32; the renderer still gives 20 px per cell.
        QCOMPARE(fullResolutionExtent(152, 0.05f), 3040);
        QVERIFY(canvasSize(152, 300, cv::Vec2f(0.05f, 0.05f)) == cv::Size(6000, 3040));
        // Exact halves round away from zero (7 x 5 at scale 2 renders as 4 x 3).
        QVERIFY(canvasSize(5, 7, cv::Vec2f(2.0f, 2.0f)) == cv::Size(4, 3));
        // A scale whose reciprocal is not an integer.
        QCOMPARE(fullResolutionExtent(100, 0.3f), 333);
        QCOMPARE(fullResolutionExtent(10, 0.0f), 10);
    }

    // Writes the pattern for the trainer round trip (tools outside this repo)
    // when VC_INK_LABELS_ROUNDTRIP_DIR names a directory.
    void writeRoundTripFixture()
    {
        const char* target = std::getenv("VC_INK_LABELS_ROUNDTRIP_DIR");
        if (!target || !*target) {
            QSKIP("VC_INK_LABELS_ROUNDTRIP_DIR not set");
        }
        const InkLabelLayer layer = roundTripPattern();
        const fs::path dir(target);
        writeTrainingLabels(trainingLabelFiles(dir, "roundtrip"), layer.labels());
        QVERIFY(cv::imwrite((dir / "roundtrip_states.png").string(), layer.labels()));
    }
};

QTEST_MAIN(InkLabelLayerTest)
#include "test_ink_label_layer.moc"
