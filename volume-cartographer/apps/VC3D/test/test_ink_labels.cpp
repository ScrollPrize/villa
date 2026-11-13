// Ink labels: raster, history, files, controller, overlay and keys.

#include "Keybinds.hpp"
#include "CState.hpp"
#include "SurfaceCatalog.hpp"
#include "StatusDockPanelHost.hpp"
#include "../../../core/test/support/TestProject.hpp"
#include "SpiralBrushCursorWidget.hpp"
#include "ink_labels/InkLabelAssist.hpp"
#include "ink_labels/InkLabelImport.hpp"
#include "ink_labels/InkLabelController.hpp"
#include "ink_labels/InkLabelRaster.hpp"
#include "ink_labels/InkLabelStore.hpp"
#include "overlays/InkLabelOverlayController.hpp"
#include "viewer_controls/panels/InkLabelPanel.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/RenderCanvas.hpp"
#include <vc_reader/zarr_write.hpp>

#include <QApplication>
#include <QDockWidget>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QCheckBox>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QKeyEvent>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QShortcut>
#include <QSlider>
#include <QAction>
#include <QTemporaryDir>
#include <QThreadPool>
#include <QTimer>
#include <QtConcurrent/QtConcurrent>
#include <QtTest/QtTest>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <tiffio.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>

namespace fs = std::filesystem;
using namespace vc3d::ink_labels;

namespace {

fs::path toPath(const QTemporaryDir& dir)
{
    return fs::path(dir.path().toStdString());
}

uint64_t commitTo(TileRaster& raster, History& history, const Shape& shape, PaintOp op, const char* label = "paint")
{
    EditTxn txn(raster, label, Actor::User);
    const uint64_t changed = txn.apply(shape, op);
    HistoryEntry entry = txn.commit(history.nextSerial());
    if (!entry.changes.empty())
        history.push(std::move(entry));
    return changed;
}

Shape disc(double x, double y, double r)
{
    return {Shape::Kind::Disc, {{x, y}}, r};
}

Shape capsule(cv::Point2d a, cv::Point2d b, double r)
{
    return {Shape::Kind::Capsule, {a, b}, r};
}

Shape polygon(std::vector<cv::Point2d> pts)
{
    return {Shape::Kind::Polygon, std::move(pts), 0};
}

cv::Mat_<uint8_t> fullImage(const TileRaster& raster)
{
    cv::Mat_<uint8_t> out;
    raster.readRect(cv::Rect(cv::Point(0, 0), raster.size()), out);
    return out;
}

std::array<uint64_t, 8> bruteCounts(const TileRaster& raster)
{
    std::array<uint64_t, 8> counts{};
    const cv::Mat_<uint8_t> img = fullImage(raster);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            ++counts[std::size_t(img(y, x) & 7)];
    return counts;
}

bool sameImage(const cv::Mat& a, const cv::Mat& b)
{
    if (a.size() != b.size() || a.type() != b.type())
        return false;
    cv::Mat diff;
    cv::compare(a, b, diff, cv::CMP_NE);
    return cv::countNonZero(diff.reshape(1)) == 0;
}

cv::Mat_<uint8_t> readGray(const fs::path& path)
{
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    return cv::Mat_<uint8_t>(img);
}

struct TiffTags {
    uint32_t width = 0, height = 0, tileW = 0, tileH = 0;
    uint16_t spp = 0, bps = 0, compression = 0;
};

TiffTags readTags(const fs::path& path)
{
    TiffTags t;
    TIFF* tf = TIFFOpen(path.string().c_str(), "r");
    if (!tf)
        return t;
    TIFFGetField(tf, TIFFTAG_IMAGEWIDTH, &t.width);
    TIFFGetField(tf, TIFFTAG_IMAGELENGTH, &t.height);
    TIFFGetField(tf, TIFFTAG_TILEWIDTH, &t.tileW);
    TIFFGetField(tf, TIFFTAG_TILELENGTH, &t.tileH);
    TIFFGetFieldDefaulted(tf, TIFFTAG_SAMPLESPERPIXEL, &t.spp);
    TIFFGetFieldDefaulted(tf, TIFFTAG_BITSPERSAMPLE, &t.bps);
    TIFFGetFieldDefaulted(tf, TIFFTAG_COMPRESSION, &t.compression);
    TIFFClose(tf);
    return t;
}

CanvasKey keyFor(cv::Size canvas)
{
    // A grid of canvas/20 at scale 0.05 renders exactly `canvas`.
    return canvasKeyFor(vc::RenderCanvas::make({canvas.width / 20, canvas.height / 20}, {0.05f, 0.05f}, 1.0));
}

SaveRequest saveRequest(const TileRaster& raster, const StorePaths& paths, const CanvasKey& key, Target target, uint64_t serial)
{
    SaveRequest req;
    req.snapshot = raster.snapshot();
    req.paths = paths;
    req.sidecar.canvas = key;
    req.sidecar.segmentName = paths.prefix;
    req.sidecar.stateSerial = serial;
    req.sidecar.savedBy = "test";
    req.target = target;
    return req;
}

// The controller with stand-ins for the main window: a shortcut host that
// records the key presses it receives, and recorded status messages.
class KeyProbe : public QWidget
{
public:
    QList<QKeySequence> seen;

protected:
    void keyPressEvent(QKeyEvent* event) override
    {
        seen.append(QKeySequence(event->keyCombination()));
        QWidget::keyPressEvent(event);
    }
};

struct Labeler {
    KeyProbe host;
    QStringList messages;
    bool otherTool = false;
    bool editing = false;
    bool workspace = true;
    QString modeLine;
    int approvalStops = 0;
    std::unique_ptr<InkLabelController> labels;

    explicit Labeler(CState* state = nullptr)
    {
        InkLabelController::Deps deps;
        deps.state = state;
        deps.shortcutHost = &host;
        deps.status = [this](const QString& text, int) { messages << text; };
        deps.otherSurfaceToolActive = [this] { return otherTool || editing; };
        deps.segmentationEditing = [this] { return editing; };
        deps.stopSurfacePainting = [this] { ++approvalStops; };
        deps.workspaceActive = [this] { return workspace; };
        deps.modeLine = [this](const QString& text) { modeLine = text; };
        labels = std::make_unique<InkLabelController>(deps);
    }
    // Closing checkpoints unsaved edits on a worker; let it land before the
    // test's temporary folder is removed.
    ~Labeler()
    {
        labels.reset();
        QThreadPool::globalInstance()->waitForDone();
    }
};

vc::RenderCanvas testCanvas(cv::Size canvas)
{
    return vc::RenderCanvas::make({canvas.width / 20, canvas.height / 20}, {0.05f, 0.05f}, 1.0);
}

// Waits until the store's write has finished and its watcher has run.
void settleWrites(InkLabelController& labels, const StorePaths& paths)
{
    const QString key = QString::fromStdString(paths.dir.string());
    QFuture<SaveOutcome> future = labels.inFlightForTesting(key);
    if (future.isValid())
        future.waitForFinished();
    QTRY_VERIFY(!labels.inFlightForTesting(key).isValid());
}

}  // namespace

class TestInkLabels : public QObject
{
    Q_OBJECT

private slots:
    void paintOpTable();
    void summaryCodesPool();
    void stampsPaintAndSeamsMatch();
    void polygonsFillAndCollapse();
    void uniformCollapseIgnoresOutOfCanvas();
    void snapshotIsolation();
    void historyUndoRedoAndBudget();
    void dirtyTrackingAcrossSaves();
    void countsMatchBruteForce();
    void storePathsAndSuffixes();
    void tiffRoundTrip();
    void loadRefusesWrongShapesAndWarns();
    void sidecarCanvasRoundTrip();
    void saveProtocolDetectsOutsideEdits();
    void autosaveTargetLeavesTrainerFiles();
    void workerSaveReadsSnapshot();
    void secondOpenIsReadOnly();
    void keysCollideWithNothing();
    void controllerSaveAndReopen();
    void undoKeyBeatsPanelFocus();
    void shortcutsFollowLabelMode();
    void keysWorkOutsideTheWorkspaceWidget();
    void polygonKeysAreShortcutsOnlyWhileOpen();
    void autosaveBeforeUndoIsNotRestored();
    void nestedOpenRefusedDuringRestorePrompt();
    void openWritesNothingUntilFirstEdit();
    void unusableAutosaveIsMovedAside();
    void labelRootLeavesTheRemoteCache();
    void segmentSwitchKeepsAutosave();
    void recoverPolicies();
    void closeGuardSavesThenCloses();
    void otherToolTakesThePress();
    void displayTilesPoolExactly();
    void displayTilesMeetOnDevicePixels();
    void panelDrivesController();
    void panelSelectionFeedbackTracksController();
    void panelOpacityReadoutTracksController();
    void namedPanelControls();
    void statusHostRoutesInkLabels();
    void panelAgentUndoPreservesHumanEntry();
    void viewportStrokeRefusesMissingViewer();
    void panelRetainsFailedActivationRefusal();
    void panelRetainsToolRefusal();
    void predictionGuideMapsWithFlips();
    void floodConnectivityAndBounds();
    void floodFollowsEqualLabels();
    void suggestionsFilterAndAccept();
    void cleanupsMatchFixtures();
    void escapeDropsRunningAssist();
    void cursorRingDefaultUnchanged();
    void fillToolAndAssistKeys();
    void importPngAndRgbTiff();
    void importLabelZarr65();
    void importZarr2DVersion3();
    void importResamplesHalfAndQuarter();
    void importLegacyPngPair();
    void numberedVersionsWriteAllKinds();
    void datasetEntryTargetsTheSegment();
    void agentEditsReadBackAndUndo();
    void agentWritesBackValues();
    void agentAssistRunsNowAndReports();
    void shapeCanvasOverlap();
};

void TestInkLabels::paintOpTable()
{
    QCOMPARE(paintOp(PaintClass::Ink, false, false), (PaintOp{3, 0}));
    QCOMPARE(paintOp(PaintClass::Ink, true, false), (PaintOp{0, 3}));
    QCOMPARE(paintOp(PaintClass::Background, false, false), (PaintOp{2, 1}));
    QCOMPARE(paintOp(PaintClass::Background, false, true), (PaintOp{2, 0}));
    QCOMPARE(paintOp(PaintClass::Background, true, false), (PaintOp{0, 3}));
    QCOMPARE(paintOp(PaintClass::Validation, false, false), (PaintOp{4, 0}));
    QCOMPARE(paintOp(PaintClass::Validation, true, true), (PaintOp{0, 4}));

    // Background over ink makes background; keeping ink leaves ink supervised.
    QCOMPARE(int(paintOp(PaintClass::Background, false, false).apply(3)), 2);
    QCOMPARE(int(paintOp(PaintClass::Background, false, true).apply(3)), 3);
    // The eraser returns to unknown and keeps validation.
    QCOMPARE(int(paintOp(PaintClass::Ink, true, false).apply(7)), 4);
    QCOMPARE(int(paintOp(PaintClass::Background, true, false).apply(2)), 0);
    // Erasing unknown pixels changes nothing.
    QCOMPARE(int(paintOp(PaintClass::Ink, true, false).apply(0)), 0);
    QCOMPARE(int(paintOp(PaintClass::Validation, true, false).apply(3)), 3);
}

void TestInkLabels::summaryCodesPool()
{
    QCOMPARE(int(summaryCode(0)), 0);
    QCOMPARE(int(summaryCode(kInk | kSupervision)), 0b1011);
    QCOMPARE(int(summaryCode(kSupervision)), 0b1010);
    QCOMPARE(int(summaryCode(kValidation)), 0b0100);
    // Any-bits OR, the all-supervised bit ANDs.
    QCOMPARE(int(poolCodes(summaryCode(3), summaryCode(0))), 0b1001);
    QCOMPARE(int(poolCodes(summaryCode(2), summaryCode(3))), 0b1011);
    QCOMPARE(int(poolCodes(summaryCode(4), summaryCode(2))), 0b1100);
    // The neutral code is the identity.
    for (int v = 0; v < 8; ++v)
        QCOMPARE(int(poolCodes(summaryCode(uint8_t(v)), kNeutralCode)), int(summaryCode(uint8_t(v))));
}

void TestInkLabels::stampsPaintAndSeamsMatch()
{
    const PaintOp ink = paintOp(PaintClass::Ink, false, false);

    // The smallest click still paints its pixel.
    {
        TileRaster raster({600, 600});
        History history;
        QVERIFY(commitTo(raster, history, disc(10.5, 10.5, 0.5), ink) >= 1);
        QCOMPARE(int(raster.at(10, 10)), 3);
        QVERIFY(commitTo(raster, history, disc(300.1, 20.9, 0.5), ink) >= 1);
        QCOMPARE(int(raster.at(300, 20)), 3);
    }

    // A fast drag leaves no gap along the segment.
    {
        TileRaster raster({600, 600});
        History history;
        const cv::Point2d a(40.3, 51.7), b(340.9, 260.2);
        commitTo(raster, history, capsule(a, b, 3.0), ink);
        for (int i = 0; i <= 2000; ++i) {
            const double t = i / 2000.0;
            const cv::Point2d p = a + t * (b - a);
            QCOMPARE(int(raster.at(int(std::floor(p.x)), int(std::floor(p.y)))), 3);
        }
    }

    // A disc across four tiles equals the same disc drawn in one window.
    {
        TileRaster raster({600, 600});
        History history;
        const cv::Point2d c(256.3, 255.7);
        const double r = 40.0;
        commitTo(raster, history, disc(c.x, c.y, r), ink);
        cv::Mat_<uint8_t> reference(600, 600, uint8_t(0));
        cv::circle(reference,
                   cv::Point(int(std::lround((c.x - 0.5) * 256)), int(std::lround((c.y - 0.5) * 256))),
                   int(std::lround(r * 256)),
                   cv::Scalar(255),
                   cv::FILLED,
                   cv::LINE_8,
                   8);
        const cv::Mat_<uint8_t> painted = fullImage(raster);
        QCOMPARE(cv::countNonZero((painted == 3) != (reference == 255)), 0);
        QVERIFY(cv::countNonZero(reference) > 4000);
        // It really touched all four tiles.
        for (int index : {0, 1, 3, 4})
            QVERIFY(raster.tile(index).isDense());
    }

    // A point far off the canvas cannot wrap around in the fixed-point
    // rasterizer and paint the wrong side: such a shape changes nothing.
    {
        TileRaster raster({600, 600});
        History history;
        QCOMPARE(commitTo(raster, history, capsule({-1.0e7, 10.5}, {100.5, 10.5}, 3.0), ink), uint64_t(0));
        QCOMPARE(commitTo(raster, history, polygon({{-1.0e9, 0}, {100, 0}, {100, 100}}), ink), uint64_t(0));
        QCOMPARE(bruteCounts(raster)[0], uint64_t(600) * 600);
        // Inside the range a long stroke paints exactly its on-canvas part.
        QVERIFY(commitTo(raster, history, capsule({-1.0e6, 10.5}, {100.5, 10.5}, 1.0), ink) > 0);
        QCOMPARE(int(raster.at(50, 10)), 3);
        QCOMPARE(int(raster.at(0, 10)), 3);
        QCOMPARE(int(raster.at(599, 10)), 0);
        QCOMPARE(int(raster.at(150, 10)), 0);
    }
}

void TestInkLabels::polygonsFillAndCollapse()
{
    const PaintOp ink = paintOp(PaintClass::Ink, false, false);

    // A concave L: the filled area is within one perimeter of the true area.
    {
        TileRaster raster({400, 400});
        History history;
        commitTo(raster, history, polygon({{20, 20}, {220, 20}, {220, 80}, {80, 80}, {80, 260}, {20, 260}}), ink);
        const double area = 200.0 * 60.0 + 60.0 * 180.0;
        const double perimeter = 2.0 * (200.0 + 240.0);
        const auto counts = raster.counts();
        QVERIFY(std::abs(double(counts[3]) - area) <= perimeter);
        QCOMPARE(int(raster.at(150, 150)), 0);  // the notch stays empty
        QCOMPARE(int(raster.at(50, 200)), 3);
    }

    // A figure eight fills both lobes.
    {
        TileRaster raster({300, 300});
        History history;
        commitTo(raster, history, polygon({{20, 20}, {120, 120}, {220, 20}, {220, 220}, {120, 120}, {20, 220}}), ink);
        QCOMPARE(int(raster.at(40, 120)), 3);
        QCOMPARE(int(raster.at(200, 120)), 3);
    }

    // A huge polygon leaves its interior tiles Uniform.
    {
        TileRaster raster({7000, 7000});
        History history;
        commitTo(raster, history, polygon({{100, 100}, {6100, 100}, {6100, 6100}, {100, 6100}}), ink);
        const int center = raster.tileIndex(12, 12);
        QVERIFY(!raster.tile(center).isDense());
        QCOMPARE(int(raster.tile(center).uniform), 3);
        // Only tiles the edges cross stay dense.
        QVERIFY(raster.denseTileCount() <= 4 * 25);
        // OpenCV's scanline fill includes pixels on the outline.
        QVERIFY(std::abs(double(raster.counts()[3]) - 6000.0 * 6000.0) <= 4.0 * 6000.0);
    }
}

void TestInkLabels::uniformCollapseIgnoresOutOfCanvas()
{
    TileRaster raster({300, 300});
    History history;
    commitTo(raster, history, polygon({{-10, -10}, {310, -10}, {310, 310}, {-10, 310}}), paintOp(PaintClass::Background, false, false));
    QCOMPARE(raster.denseTileCount(), std::size_t(0));
    for (int i = 0; i < raster.tileCount(); ++i)
        QCOMPARE(int(raster.tile(i).uniform), 2);
    QCOMPARE(raster.counts()[2], uint64_t(300) * 300);

    // An edge tile painted fully inside the canvas collapses too.
    commitTo(raster, history, polygon({{256, 256}, {300, 256}, {300, 300}, {256, 300}}), paintOp(PaintClass::Ink, false, false));
    QVERIFY(!raster.tile(3).isDense());
    QCOMPARE(int(raster.tile(3).uniform), 3);
}

void TestInkLabels::snapshotIsolation()
{
    TileRaster raster({1000, 600});
    History history;
    commitTo(raster, history, disc(100, 100, 30), paintOp(PaintClass::Ink, false, false));
    commitTo(raster, history, disc(700, 400, 30), paintOp(PaintClass::Ink, false, false));
    const TileRaster snap = raster.snapshot();
    const cv::Mat_<uint8_t> before = fullImage(snap);

    commitTo(raster, history, disc(110, 100, 30), paintOp(PaintClass::Background, false, false));
    QVERIFY(sameImage(fullImage(snap), before));
    QVERIFY(!sameImage(fullImage(raster), before));
    // Tiles the edit never touched are still shared.
    const int untouched = raster.tileIndex(2, 1);
    QVERIFY(raster.tile(untouched).isDense());
    QCOMPARE(snap.tile(untouched).dense.get(), raster.tile(untouched).dense.get());
    QVERIFY(snap.tile(0).dense.get() != raster.tile(0).dense.get());
}

void TestInkLabels::historyUndoRedoAndBudget()
{
    TileRaster raster({800, 800});
    History history;
    const PaintOp ink = paintOp(PaintClass::Ink, false, false);
    const cv::Mat_<uint8_t> empty = fullImage(raster);
    commitTo(raster, history, disc(200, 200, 50), ink);
    const cv::Mat_<uint8_t> one = fullImage(raster);
    const auto oneCounts = raster.counts();
    commitTo(raster, history, capsule({100, 600}, {700, 600}, 20), ink);
    const cv::Mat_<uint8_t> two = fullImage(raster);
    QCOMPARE(history.undoDepth(), std::size_t(2));

    QVERIFY(history.undo(raster));
    QVERIFY(sameImage(fullImage(raster), one));
    QCOMPARE(raster.counts(), oneCounts);
    QVERIFY(history.undo(raster));
    QVERIFY(sameImage(fullImage(raster), empty));
    QVERIFY(!history.undo(raster));
    QVERIFY(history.redo(raster));
    QVERIFY(history.redo(raster));
    QVERIFY(sameImage(fullImage(raster), two));
    QVERIFY(!history.redo(raster));

    // A new edit clears redo.
    QVERIFY(history.undo(raster));
    QCOMPARE(history.redoDepth(), std::size_t(1));
    commitTo(raster, history, disc(600, 200, 10), ink);
    QCOMPARE(history.redoDepth(), std::size_t(0));

    // A tight budget drops the oldest entries and keeps the newest.
    TileRaster small({1024, 256});
    const std::size_t budget = 2 * std::size_t(kTilePixels) + 200;
    History tight(budget);
    std::vector<uint64_t> serials;
    for (int i = 0; i < 4; ++i) {
        EditTxn txn(small, "stroke", Actor::User);
        txn.apply(disc(128 + 256 * i, 128, 20), ink);
        HistoryEntry e = txn.commit(tight.nextSerial());
        serials.push_back(e.serial);
        tight.push(std::move(e));
    }
    QVERIFY(tight.bytes() <= budget);
    QCOMPARE(tight.undoDepth(), std::size_t(2));
    QCOMPARE(tight.top()->serial, serials.back());
    // Undo everything that is left: the base is the newest dropped entry.
    while (tight.undo(small)) {
    }
    QCOMPARE(tight.stateSerial(), serials[1]);
    QCOMPARE(int(small.at(128, 128)), 3);  // the dropped strokes stay
    QCOMPARE(int(small.at(384, 128)), 3);
    QCOMPARE(int(small.at(640, 128)), 0);

    // A single entry over the budget is still kept.
    History tiny(1);
    TileRaster r2({300, 300});
    commitTo(r2, tiny, disc(150, 150, 100), ink);
    QCOMPARE(tiny.undoDepth(), std::size_t(1));
}

void TestInkLabels::dirtyTrackingAcrossSaves()
{
    TileRaster raster({512, 512});
    History history;
    const PaintOp ink = paintOp(PaintClass::Ink, false, false);
    QVERIFY(!history.dirty());
    commitTo(raster, history, disc(100, 100, 10), ink);
    QVERIFY(history.dirty());

    // A save of this state starts, then the user paints again.
    const uint64_t savedSerial = history.stateSerial();
    commitTo(raster, history, disc(300, 300, 10), ink);
    history.markClean(savedSerial);
    QVERIFY(history.dirty());
    QVERIFY(history.undo(raster));
    QVERIFY(!history.dirty());
    QVERIFY(history.redo(raster));
    QVERIFY(history.dirty());
    history.markClean(history.stateSerial());
    QVERIFY(!history.dirty());
    QVERIFY(history.undo(raster));
    QVERIFY(history.dirty());

    history.reset(41);
    QVERIFY(!history.dirty());
    QCOMPARE(history.stateSerial(), uint64_t(41));
    QCOMPARE(history.nextSerial(), uint64_t(42));
}

void TestInkLabels::countsMatchBruteForce()
{
    TileRaster raster({1100, 700});
    History history;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> ux(-20, 1120), uy(-20, 720), ur(0.5, 60);
    std::uniform_int_distribution<int> kind(0, 3), cls(0, 2), coin(0, 1);
    for (int i = 0; i < 500; ++i) {
        const PaintOp op = paintOp(PaintClass(cls(rng)), coin(rng) == 1, coin(rng) == 1);
        switch (kind(rng)) {
            case 0:
                commitTo(raster, history, disc(ux(rng), uy(rng), ur(rng)), op);
                break;
            case 1:
                commitTo(raster, history, capsule({ux(rng), uy(rng)}, {ux(rng), uy(rng)}, ur(rng)), op);
                break;
            case 2:
                commitTo(raster, history, polygon({{ux(rng), uy(rng)}, {ux(rng), uy(rng)}, {ux(rng), uy(rng)}, {ux(rng), uy(rng)}}), op);
                break;
            default: {
                if (coin(rng) == 1 && history.undo(raster))
                    break;
                cv::Mat_<uint8_t> mask(37, 53, uint8_t(0));
                cv::circle(mask, {26, 18}, 15, cv::Scalar(1), cv::FILLED);
                EditTxn txn(raster, "mask", Actor::Agent);
                txn.applyMask({int(ux(rng)), int(uy(rng))}, mask, op);
                HistoryEntry e = txn.commit(history.nextSerial());
                if (!e.changes.empty())
                    history.push(std::move(e));
            }
        }
    }
    QCOMPARE(raster.counts(), bruteCounts(raster));
    uint64_t total = 0;
    for (auto c : raster.counts())
        total += c;
    QCOMPARE(total, uint64_t(1100) * 700);
}

void TestInkLabels::storePathsAndSuffixes()
{
    const StorePaths paths = storePathsFor("/data/project", "seg_a");
    QCOMPARE(QString::fromStdString(paths.dir.string()), QStringLiteral("/data/project/ink_labels/seg_a"));
    QCOMPARE(QString::fromStdString(paths.file(LabelKind::Ink).filename().string()), QStringLiteral("seg_a_inklabels.tif"));
    QCOMPARE(QString::fromStdString(paths.file(LabelKind::Supervision).filename().string()), QStringLiteral("seg_a_supervision_mask.tif"));
    QCOMPARE(QString::fromStdString(paths.file(LabelKind::Validation).filename().string()), QStringLiteral("seg_a_validation_mask.tif"));
    StorePaths v2 = paths;
    v2.version = 2;
    QCOMPARE(QString::fromStdString(v2.file(LabelKind::Ink).filename().string()), QStringLiteral("seg_a_inklabels_v2.tif"));

    // The trainer reads .tif/.tiff/.zarr and create_label_zarrs .tif/.tiff/.png:
    // every auxiliary file must use another extension.
    const std::vector<fs::path> auxiliary{paths.sidecar(),
                                          paths.sidecar(Target::Autosave),
                                          paths.file(LabelKind::Ink, Target::Autosave),
                                          paths.file(LabelKind::Validation, Target::Autosave),
                                          paths.previous(LabelKind::Supervision),
                                          paths.lock(),
                                          fs::path(paths.file(LabelKind::Ink).string() + ".tmp-12-3")};
    for (const auto& p : auxiliary) {
        const std::string ext = p.extension().string();
        QVERIFY2(ext != ".tif" && ext != ".tiff" && ext != ".png" && ext != ".zarr", p.string().c_str());
    }
}

void TestInkLabels::tiffRoundTrip()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const cv::Size canvas(4097, 2049);
    const CanvasKey key = canvasKeyFor(vc::RenderCanvas::make({4097, 2049}, {1.f, 1.f}, 1.0));
    QCOMPARE(key.width, 4097);
    const StorePaths paths = storePathsFor(toPath(dir), "seg");
    TileRaster raster(canvas);
    History history;
    commitTo(raster, history, polygon({{0, 0}, {4097, 0}, {4097, 2049}, {0, 2049}}), paintOp(PaintClass::Background, false, false));
    commitTo(raster, history, capsule({30, 40}, {4000, 2000}, 25), paintOp(PaintClass::Ink, false, false));
    commitTo(raster, history, disc(4090, 2040, 30), paintOp(PaintClass::Ink, false, false));

    SaveRequest req = saveRequest(raster, paths, key, Target::Trainer, history.stateSerial());
    const SaveOutcome out = saveLabels(req);
    QVERIFY2(out.ok, qPrintable(out.error));
    QVERIFY(fs::exists(paths.file(LabelKind::Ink)));
    QVERIFY(fs::exists(paths.file(LabelKind::Supervision)));
    QVERIFY(!fs::exists(paths.file(LabelKind::Validation)));
    for (const LabelKind kind : {LabelKind::Ink, LabelKind::Supervision}) {
        const TiffTags t = readTags(paths.file(kind));
        QCOMPARE(t.width, uint32_t(4097));
        QCOMPARE(t.height, uint32_t(2049));
        QCOMPARE(t.tileW, uint32_t(1024));
        QCOMPARE(t.tileH, uint32_t(1024));
        QCOMPARE(t.spp, uint16_t(1));
        QCOMPARE(t.bps, uint16_t(8));
        QCOMPARE(t.compression, uint16_t(COMPRESSION_LZW));
        const cv::Mat_<uint8_t> img = readGray(paths.file(kind));
        QCOMPARE(img.size(), canvas);
        const int nonBinary = cv::countNonZero((img != 0) & (img != 255));
        QCOMPARE(nonBinary, 0);
    }
    const cv::Mat_<uint8_t> inkImage = readGray(paths.file(LabelKind::Ink));
    QCOMPARE(uint64_t(cv::countNonZero(inkImage)), raster.counts()[3]);

    const LoadOutcome back = loadLabels(paths, key, Target::Trainer);
    QVERIFY2(back.ok, qPrintable(back.error));
    QVERIFY(back.warnings.isEmpty());
    QVERIFY(sameImage(fullImage(back.raster), fullImage(raster)));
    QCOMPARE(back.raster.counts(), raster.counts());
    QVERIFY(!back.validationPresent);

    // Validation is written when asked (a V bit is set or a file exists).
    commitTo(raster, history, disc(1000, 1000, 100), paintOp(PaintClass::Validation, false, false));
    SaveRequest req2 = saveRequest(raster, paths, key, Target::Trainer, history.stateSerial());
    req2.writeValidation = true;
    req2.expectedSha256 = out.sha256;
    const SaveOutcome out2 = saveLabels(req2);
    QVERIFY2(out2.ok, qPrintable(out2.error));
    QVERIFY(fs::exists(paths.file(LabelKind::Validation)));
    const LoadOutcome back2 = loadLabels(paths, key, Target::Trainer);
    QVERIFY(back2.ok);
    QVERIFY(back2.validationPresent);
    QVERIFY(sameImage(fullImage(back2.raster), fullImage(raster)));
}

void TestInkLabels::loadRefusesWrongShapesAndWarns()
{
    QTemporaryDir dir;
    const StorePaths paths = storePathsFor(toPath(dir), "seg");
    fs::create_directories(paths.dir);
    const CanvasKey key = keyFor({400, 200});

    cv::Mat_<uint8_t> wrong(100, 300, uint8_t(255));
    QVERIFY(cv::imwrite(paths.file(LabelKind::Ink).string(), wrong));
    LoadOutcome out = loadLabels(paths, key, Target::Trainer);
    QVERIFY(!out.ok);
    QVERIFY(out.error.contains(QStringLiteral("300 x 100")));
    QVERIFY(out.error.contains(QStringLiteral("400 x 200")));

    cv::Mat_<uint8_t> rotated(400, 200, uint8_t(255));
    QVERIFY(cv::imwrite(paths.file(LabelKind::Ink).string(), rotated));
    out = loadLabels(paths, key, Target::Trainer);
    QVERIFY(!out.ok);
    QVERIFY(out.error.contains(QStringLiteral("rotated")));

    // RGB uses channel 0 with a warning; anti-aliased values warn too.
    // imwrite stores BGR as RGB, so the red channel becomes sample 0.
    cv::Mat rgb(200, 400, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(rgb, cv::Rect(10, 10, 50, 50), cv::Scalar(0, 0, 255), cv::FILLED);
    cv::rectangle(rgb, cv::Rect(200, 10, 50, 50), cv::Scalar(255, 255, 0), cv::FILLED);
    QVERIFY(cv::imwrite(paths.file(LabelKind::Ink).string(), rgb));
    cv::Mat_<uint8_t> soft(200, 400, uint8_t(0));
    soft(cv::Rect(100, 100, 20, 20)) = 255;
    soft(cv::Rect(120, 100, 5, 20)) = 128;
    soft(cv::Rect(125, 100, 2, 20)) = 64;
    QVERIFY(cv::imwrite(paths.file(LabelKind::Supervision).string(), soft));
    out = loadLabels(paths, key, Target::Trainer);
    QVERIFY2(out.ok, qPrintable(out.error));
    const QString warnings = out.warnings.join('\n');
    QVERIFY(warnings.contains(QStringLiteral("more than one sample")));
    QVERIFY(warnings.contains(QStringLiteral("128 (100 px)")));
    QVERIFY(warnings.contains(QStringLiteral("64 (40 px)")));
    QCOMPARE(int(out.raster.at(120, 110) & kSupervision), int(kSupervision));  // 128 counts as labeled
    const auto counts = out.raster.counts();
    QCOMPARE(counts[1] + counts[3] + counts[5] + counts[7], uint64_t(50 * 50));
}

void TestInkLabels::sidecarCanvasRoundTrip()
{
    QTemporaryDir dir;
    for (const float scale : {0.05f, 0.3f}) {
        const auto canvas = vc::RenderCanvas::make({152, 76}, {scale, scale}, 1.0);
        Sidecar s;
        s.canvas = canvasKeyFor(canvas);
        s.segmentName = "seg_a";
        s.segmentUuid = "abc";
        s.sha256[LabelKind::Ink] = "00ff";
        s.fileNames[LabelKind::Ink] = "seg_a_inklabels.tif";
        s.stateSerial = 412;
        s.predictionShown = std::string("/tmp/pred.png");
        s.edits = QJsonObject{{"user", QJsonObject{{"brush", 3}}}};
        s.provenance.append(QJsonObject{{"source", "test"}});
        const fs::path path = toPath(dir) / "s.json";
        QString error;
        QVERIFY2(writeSidecar(path, s, &error), qPrintable(error));
        const auto back = readSidecar(path, &error);
        QVERIFY2(back.has_value(), qPrintable(error));
        QVERIFY(back->canvas == s.canvas);
        QCOMPARE(back->canvas.scaleX(), scale);
        QCOMPARE(back->stateSerial, uint64_t(412));
        QCOMPARE(QString::fromStdString(back->sha256.at(LabelKind::Ink)), QStringLiteral("00ff"));
        QVERIFY(!back->sha256.count(LabelKind::Validation));
        QCOMPARE(QString::fromStdString(*back->predictionShown), QStringLiteral("/tmp/pred.png"));
        QCOMPARE(back->edits.value("user").toObject().value("brush").toInt(), 3);
        QCOMPARE(back->provenance.size(), 1);

        // The file names the scale both ways.
        QFile f(QString::fromStdString(path.string()));
        QVERIFY(f.open(QIODevice::ReadOnly));
        const QJsonObject root = QJsonDocument::fromJson(f.readAll()).object();
        QCOMPARE(root.value("schema").toString(), QStringLiteral("vc3d.ink_labels/1"));
        QCOMPARE(root.value("canvas").toObject().value("scale_f32_hex").toArray().size(), 2);
        QVERIFY(root.value("files").toObject().value("validation_mask").isNull());
    }

    // A canvas mismatch refuses to load and names both canvases.
    const StorePaths paths = storePathsFor(toPath(dir), "seg");
    TileRaster raster({400, 200});
    const CanvasKey key = keyFor({400, 200});
    QVERIFY(saveLabels(saveRequest(raster, paths, key, Target::Trainer, 1)).ok);
    const CanvasKey other = keyFor({420, 200});
    const LoadOutcome out = loadLabels(paths, other, Target::Trainer);
    QVERIFY(!out.ok);
    QVERIFY(out.canvasMismatch);
    QVERIFY(out.error.contains(QStringLiteral("400 x 200 px")));
    QVERIFY(out.error.contains(QStringLiteral("420 x 200 px")));
}

void TestInkLabels::saveProtocolDetectsOutsideEdits()
{
    QTemporaryDir dir;
    const StorePaths paths = storePathsFor(toPath(dir), "seg");
    const CanvasKey key = keyFor({600, 400});
    TileRaster raster({600, 400});
    History history;
    commitTo(raster, history, disc(100, 100, 40), paintOp(PaintClass::Ink, false, false));
    const SaveOutcome first = saveLabels(saveRequest(raster, paths, key, Target::Trainer, history.stateSerial()));
    QVERIFY2(first.ok, qPrintable(first.error));

    // The sidecar records the hashes of the files in place.
    QString error;
    const auto sidecar = readSidecar(paths.sidecar(), &error);
    QVERIFY(sidecar.has_value());
    for (const auto& [kind, path] : first.files)
        QCOMPARE(sidecar->sha256.at(kind), fileSha256(path));

    // An outside edit blocks the save.
    const std::string firstInk = fileSha256(paths.file(LabelKind::Ink));
    cv::Mat_<uint8_t> edited = readGray(paths.file(LabelKind::Ink));
    edited(cv::Rect(300, 300, 10, 10)) = 255;
    QVERIFY(cv::imwrite(paths.file(LabelKind::Ink).string(), edited));
    commitTo(raster, history, disc(400, 200, 40), paintOp(PaintClass::Ink, false, false));
    SaveRequest second = saveRequest(raster, paths, key, Target::Trainer, history.stateSerial());
    second.expectedSha256 = first.sha256;
    const SaveOutcome blocked = saveLabels(second);
    QVERIFY(!blocked.ok);
    QCOMPARE(blocked.externallyEdited, QStringList{QStringLiteral("seg_inklabels.tif")});
    // Nothing was written; no temp file is left.
    for (const auto& entry : fs::directory_iterator(paths.dir))
        QVERIFY2(entry.path().string().find(".tmp-") == std::string::npos, entry.path().string().c_str());

    // Overwriting keeps the edited file as the previous generation.
    const std::string editedSha = fileSha256(paths.file(LabelKind::Ink));
    second.overwriteExternalEdits = true;
    const SaveOutcome forced = saveLabels(second);
    QVERIFY2(forced.ok, qPrintable(forced.error));
    QCOMPARE(fileSha256(paths.previous(LabelKind::Ink)), editedSha);
    QVERIFY(firstInk != editedSha);

    // A file swapped in from another save is reported on load.
    fs::copy_file(paths.previous(LabelKind::Ink), paths.file(LabelKind::Ink), fs::copy_options::overwrite_existing);
    const LoadOutcome loaded = loadLabels(paths, key, Target::Trainer);
    QVERIFY(loaded.ok);
    QVERIFY(loaded.warnings.join('\n').contains(QStringLiteral("seg_inklabels.tif")));
    QVERIFY(loaded.warnings.join('\n').contains(QStringLiteral(".prev")));
}

void TestInkLabels::autosaveTargetLeavesTrainerFiles()
{
    QTemporaryDir dir;
    const StorePaths paths = storePathsFor(toPath(dir), "seg");
    const CanvasKey key = keyFor({600, 400});
    TileRaster raster({600, 400});
    History history;
    commitTo(raster, history, disc(100, 100, 40), paintOp(PaintClass::Ink, false, false));
    const SaveOutcome trainer = saveLabels(saveRequest(raster, paths, key, Target::Trainer, history.stateSerial()));
    QVERIFY(trainer.ok);
    const std::string inkSha = fileSha256(paths.file(LabelKind::Ink));
    const std::string sidecarSha = fileSha256(paths.sidecar());

    commitTo(raster, history, disc(300, 200, 60), paintOp(PaintClass::Background, false, false));
    SaveRequest checkpoint = saveRequest(raster, paths, key, Target::Autosave, history.stateSerial());
    checkpoint.sidecar.writeSeq = 7;
    const SaveOutcome autosave = saveLabels(checkpoint);
    QVERIFY2(autosave.ok, qPrintable(autosave.error));
    QCOMPARE(fileSha256(paths.file(LabelKind::Ink)), inkSha);
    QCOMPARE(fileSha256(paths.sidecar()), sidecarSha);
    QVERIFY(fs::exists(paths.file(LabelKind::Ink, Target::Autosave)));
    QVERIFY(fs::exists(paths.sidecar(Target::Autosave)));
    QVERIFY(!fs::exists(paths.previous(LabelKind::Ink)));
    QString error;
    const auto autosaveSidecar = readSidecar(paths.sidecar(Target::Autosave), &error);
    QVERIFY(autosaveSidecar && autosaveSidecar->autosave);
    QCOMPARE(autosaveSidecar->writeSeq, uint64_t(7));
    QVERIFY(QString::fromStdString(autosaveSidecar->fileNames.at(LabelKind::Ink)).endsWith(QStringLiteral(".autosave")));

    const LoadOutcome restored = loadLabels(paths, key, Target::Autosave);
    QVERIFY2(restored.ok, qPrintable(restored.error));
    QVERIFY(sameImage(fullImage(restored.raster), fullImage(raster)));

    // A trainer save of a newer state removes the checkpoint.
    SaveRequest save = saveRequest(raster, paths, key, Target::Trainer, history.stateSerial());
    save.expectedSha256 = trainer.sha256;
    const SaveOutcome second = saveLabels(save);
    QVERIFY(second.ok);
    QVERIFY(!targetExists(paths, Target::Autosave));

    // So does a save of an older serial: after an undo the saved state can
    // have a smaller serial than a checkpoint written before the undo.
    QVERIFY(saveLabels(saveRequest(raster, paths, key, Target::Autosave, history.stateSerial() + 5)).ok);
    QVERIFY(targetExists(paths, Target::Autosave));
    SaveRequest afterUndo = saveRequest(raster, paths, key, Target::Trainer, 1);
    afterUndo.expectedSha256 = second.sha256;
    QVERIFY(saveLabels(afterUndo).ok);
    QVERIFY(!targetExists(paths, Target::Autosave));

    // A checkpoint that cannot be restored is moved aside, not deleted.
    QVERIFY(saveLabels(saveRequest(raster, paths, key, Target::Autosave, 9)).ok);
    QVERIFY(moveTargetAside(paths, Target::Autosave));
    QVERIFY(!targetExists(paths, Target::Autosave));
    fs::path aside = paths.sidecar(Target::Autosave);
    aside += ".unusable";
    QVERIFY(fs::exists(aside));
    aside = paths.file(LabelKind::Ink, Target::Autosave);
    aside += ".unusable";
    QVERIFY(fs::exists(aside));
}

void TestInkLabels::workerSaveReadsSnapshot()
{
    QTemporaryDir dir;
    const StorePaths paths = storePathsFor(toPath(dir), "seg");
    const CanvasKey key = keyFor({2000, 1400});
    TileRaster raster({2000, 1400});
    History history;
    commitTo(raster, history, polygon({{0, 0}, {2000, 0}, {2000, 1400}, {0, 1400}}), paintOp(PaintClass::Background, false, false));
    commitTo(raster, history, capsule({50, 50}, {1900, 1300}, 30), paintOp(PaintClass::Ink, false, false));
    const SaveRequest req = saveRequest(raster, paths, key, Target::Trainer, history.stateSerial());
    const cv::Mat_<uint8_t> expected = fullImage(req.snapshot);

    QFuture<SaveOutcome> future = QtConcurrent::run([req] { return saveLabels(req); });
    // Keep painting the live raster while the worker writes.
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> ux(0, 2000), uy(0, 1400);
    for (int i = 0; i < 200; ++i)
        commitTo(raster, history, disc(ux(rng), uy(rng), 25), paintOp(PaintClass::Ink, i % 2 == 0, false));
    future.waitForFinished();
    const SaveOutcome out = future.result();
    QVERIFY2(out.ok, qPrintable(out.error));
    const LoadOutcome back = loadLabels(paths, key, Target::Trainer);
    QVERIFY(back.ok);
    QVERIFY(sameImage(fullImage(back.raster), expected));
}

void TestInkLabels::secondOpenIsReadOnly()
{
    QTemporaryDir dir;
    Labeler first, second;
    const auto canvas = testCanvas({600, 400});
    QVERIFY(first.labels->openForTesting(canvas, toPath(dir), "seg").ok);
    QVERIFY(!first.labels->session()->readOnly);

    // The first edit creates the folder and takes the QLockFile.
    QVERIFY(first.labels->applyShape(disc(300, 200, 20), Actor::User) > 0);
    QVERIFY(fs::exists(first.labels->session()->paths.lock()));

    // The QLockFile is held: a second VC3D opens read-only and cannot paint.
    const auto out = second.labels->openForTesting(canvas, toPath(dir), "seg");
    QVERIFY(out.ok);
    QVERIFY(out.readOnly);
    QVERIFY(second.labels->session()->lockReason.contains(QStringLiteral("another VC3D")));
    QCOMPARE(second.labels->applyShape(disc(100, 100, 20), Actor::User), uint64_t(0));
    QVERIFY(!second.labels->beginStroke({50, 50}, false, Actor::User));
    QVERIFY(second.labels->save().result().error.contains(QStringLiteral("another VC3D")));

    // Refusing read-only reports the holder instead.
    Labeler third;
    InkLabelController::OpenOptions strict;
    strict.readOnlyIfLocked = false;
    const auto refused = third.labels->openForTesting(canvas, toPath(dir), "seg", strict);
    QVERIFY(!refused.ok);
    QCOMPARE(refused.error, InkLabelController::OpenError::LockedByOther);

    // Two VC3Ds that opened a store before either painted: the first edit
    // wins the lock and the other turns read-only at its first edit.
    Labeler a, b;
    QVERIFY(a.labels->openForTesting(canvas, toPath(dir), "fresh").ok);
    QVERIFY(b.labels->openForTesting(canvas, toPath(dir), "fresh").ok);
    QVERIFY(!b.labels->session()->readOnly);
    QVERIFY(a.labels->applyShape(disc(100, 100, 20), Actor::User) > 0);
    QCOMPARE(b.labels->applyShape(disc(100, 100, 20), Actor::User), uint64_t(0));
    QVERIFY(b.labels->session()->readOnly);
    QVERIFY(b.labels->session()->lockReason.contains(QStringLiteral("another VC3D")));
}

void TestInkLabels::keysCollideWithNothing()
{
    namespace kb = vc3d::keybinds;
    const std::vector<const kb::ShortcutDef*> ink{&kb::shortcuts::InkLabelMode,
                                                  &kb::shortcuts::InkLabelInk,
                                                  &kb::shortcuts::InkLabelBackground,
                                                  &kb::shortcuts::InkLabelValidation,
                                                  &kb::shortcuts::InkLabelEraser,
                                                  &kb::shortcuts::InkLabelTool,
                                                  &kb::shortcuts::InkLabelSuggest,
                                                  &kb::shortcuts::InkLabelPickAll,
                                                  &kb::shortcuts::InkLabelShow,
                                                  &kb::shortcuts::InkLabelUndo,
                                                  &kb::shortcuts::InkLabelRedo,
                                                  &kb::shortcuts::InkLabelSave};
    const std::vector<const kb::ShortcutDef*> shortcuts{&kb::shortcuts::OpenVolpkg,
                                                        &kb::shortcuts::AxisAlignedSlices,
                                                        &kb::shortcuts::SurfaceNormals,
                                                        &kb::shortcuts::CompositeView,
                                                        &kb::shortcuts::RawPointsOverlay,
                                                        &kb::shortcuts::ZoomIn,
                                                        &kb::shortcuts::ZoomOut,
                                                        &kb::shortcuts::ResetView,
                                                        &kb::shortcuts::ZoomToFit,
                                                        &kb::shortcuts::WorldOffsetZPos,
                                                        &kb::shortcuts::WorldOffsetZNeg,
                                                        &kb::shortcuts::CycleNextSegment,
                                                        &kb::shortcuts::CyclePrevSegment,
                                                        &kb::shortcuts::ApplyApprovedTag,
                                                        &kb::shortcuts::ApplyDefectiveTag,
                                                        &kb::shortcuts::FocusedView,
                                                        &kb::shortcuts::CycleViewers};
    const std::vector<const kb::KeyPressDef*> keypresses{
        &kb::keypress::ToggleVolumeOverlay, &kb::keypress::CenterFocusOnCursor, &kb::keypress::RecenterFocus,
        &kb::keypress::SliceStepDecrease,   &kb::keypress::SliceStepIncrease,   &kb::keypress::SegmentationUndo,
        &kb::keypress::ManualAddToggle,     &kb::keypress::GrowSegmentation,
        &kb::keypress::CancelOperation,     &kb::keypress::PushPullIn,          &kb::keypress::PushPullOut,
        &kb::keypress::PushPullInAlpha,     &kb::keypress::PushPullOutAlpha,    &kb::keypress::PushPullRadiusDown,
        &kb::keypress::PushPullRadiusUp,    &kb::keypress::EnableEditing,       &kb::keypress::ToggleAnnotation,
        &kb::keypress::GrowthLeft,          &kb::keypress::GrowthUp,            &kb::keypress::GrowthDown,
        &kb::keypress::GrowthRight,         &kb::keypress::GrowthAll,           &kb::keypress::GrowthStepAll,
        &kb::keypress::GrowthFill,          &kb::keypress::DeletePoint,         &kb::range_slider::StepDownLeft,
        &kb::range_slider::StepDownDown,    &kb::range_slider::StepUpRight,     &kb::range_slider::StepUpUp,
        &kb::range_slider::PageDown,        &kb::range_slider::PageUp};

    // Keys bound outside Keybinds.hpp: Lasagna (Ctrl+L, Ctrl+Shift+L), fiber
    // annotation (J), same-wrap annotation (Shift+E) and the "Hold G" help.
    std::vector<std::pair<QKeySequence, QString>> taken{{QKeySequence(QStringLiteral("Ctrl+L")), QStringLiteral("lasagna")},
                                                        {QKeySequence(QStringLiteral("Ctrl+Shift+L")), QStringLiteral("lasagna repeat")},
                                                        {QKeySequence(QStringLiteral("J")), QStringLiteral("fiber annotation")},
                                                        {QKeySequence(QStringLiteral("Shift+E")), QStringLiteral("same wrap")},
                                                        {QKeySequence(QStringLiteral("G")), QStringLiteral("hold G")}};
    for (const auto* def : shortcuts)
        for (const QKeySequence& seq : kb::sequencesFor(*def))
            taken.emplace_back(seq, QString::fromUtf8(def->id));
    for (const auto* def : keypresses)
        taken.emplace_back(kb::sequenceFor(*def), QString::fromUtf8(def->id));

    QSet<QString> mine;
    for (const auto* def : ink) {
        const QList<QKeySequence> seqs = kb::sequencesFor(*def);
        QVERIFY2(!seqs.isEmpty() && !seqs.front().isEmpty(), def->id);
        for (const QKeySequence& seq : seqs) {
            QVERIFY2(!mine.contains(seq.toString()), qPrintable(seq.toString()));
            mine.insert(seq.toString());
            for (const auto& [other, owner] : taken) {
                // The one documented override: Ctrl+Z undoes labels in label
                // mode, where segmentation editing is always off.
                if (def == &kb::shortcuts::InkLabelUndo && owner == QStringLiteral("segmentation_undo"))
                    continue;
                QVERIFY2(seq != other, qPrintable(QStringLiteral("%1 (%2) collides with %3").arg(seq.toString(), def->id, owner)));
            }
        }
    }
    QCOMPARE(kb::sequencesFor(kb::shortcuts::InkLabelUndo).front(), kb::sequenceFor(kb::keypress::SegmentationUndo));
    QVERIFY(kb::buildKeybindsHelpText().contains(QStringLiteral("Ink Labels")));
}

void TestInkLabels::controllerSaveAndReopen()
{
    QTemporaryDir dir;
    const auto canvas = testCanvas({800, 600});
    StorePaths paths;
    std::array<uint64_t, 8> counts{};
    std::map<LabelKind, std::string> sha;
    {
        Labeler l;
        const auto out = l.labels->openForTesting(canvas, toPath(dir), "seg_a");
        QVERIFY2(out.ok, qPrintable(out.message));
        paths = l.labels->session()->paths;
        QCOMPARE(QString::fromStdString(paths.dir.string()), QString::fromStdString((toPath(dir) / "ink_labels" / "seg_a").string()));
        l.labels->setRadius(15);
        l.labels->setPaintClass(PaintClass::Background);
        QVERIFY(l.labels->applyShape(polygon({{100, 100}, {700, 100}, {700, 500}, {100, 500}}), Actor::User) > 0);
        l.labels->setPaintClass(PaintClass::Ink);
        QVERIFY(l.labels->beginStroke({150, 150}, false, Actor::User));
        l.labels->extendStroke({400, 300});
        l.labels->extendStroke({650, 200});
        l.labels->endStroke();
        QCOMPARE(int(l.labels->history()->undoDepth()), 2);
        QVERIFY(l.labels->session()->dirty);

        const SaveOutcome saved = l.labels->save().result();
        QVERIFY2(saved.ok, qPrintable(saved.error));
        QTRY_VERIFY(!l.labels->session()->dirty);
        QVERIFY(l.labels->lastSave().has_value());
        counts = l.labels->session()->counts;
        sha = saved.sha256;
        QCOMPARE(sha.size(), std::size_t(2));  // no validation painted
        l.labels->closeSession();
        settleWrites(*l.labels, paths);
    }
    QVERIFY(!targetExists(paths, Target::Autosave));

    Labeler again;
    const auto reopened = again.labels->openForTesting(canvas, toPath(dir), "seg_a");
    QVERIFY2(reopened.ok, qPrintable(reopened.message));
    QVERIFY(!reopened.readOnly);
    QVERIFY(reopened.warnings.isEmpty());
    QCOMPARE(again.labels->session()->counts, counts);
    QVERIFY(!again.labels->session()->dirty);
    QString error;
    const auto sidecar = readSidecar(paths.sidecar(), &error);
    QVERIFY(sidecar.has_value());
    for (const auto& [kind, hash] : sha) {
        QCOMPARE(sidecar->sha256.at(kind), hash);
        QCOMPARE(fileSha256(paths.file(kind)), hash);
    }
    QCOMPARE(sidecar->edits.value("user").toObject().value("polygon").toInt(), 1);
    QCOMPARE(sidecar->edits.value("user").toObject().value("brush").toInt(), 1);
}

void TestInkLabels::undoKeyBeatsPanelFocus()
{
    QTemporaryDir dir;
    Labeler l;
    auto* button = new QPushButton(QStringLiteral("dock control"), &l.host);
    l.host.resize(300, 200);
    l.host.show();
    QVERIFY(QTest::qWaitForWindowActive(&l.host));
    QVERIFY(l.labels->openForTesting(testCanvas({600, 400}), toPath(dir), "seg").ok);
    l.labels->setLabelModeEnabled(true);
    QVERIFY(l.labels->labelModeEnabled());
    QVERIFY(l.labels->applyShape(disc(200, 200, 30), Actor::User) > 0);
    QCOMPARE(int(l.labels->history()->undoDepth()), 1);

    button->setFocus();
    QTRY_VERIFY(button->hasFocus());
    QTest::keyClick(button, Qt::Key_Z, Qt::ControlModifier);
    QCOMPARE(int(l.labels->history()->undoDepth()), 0);
    QCOMPARE(int(l.labels->history()->redoDepth()), 1);
    QVERIFY(!l.host.seen.contains(QKeySequence(Qt::ControlModifier | Qt::Key_Z)));
    QCOMPARE(l.labels->session()->counts[0], uint64_t(600) * 400);
}

void TestInkLabels::shortcutsFollowLabelMode()
{
    QTemporaryDir dir;
    Labeler l;
    l.host.resize(300, 200);
    // The host stands in for the main window; the keys are window shortcuts.
    l.host.setFocusPolicy(Qt::StrongFocus);
    l.host.show();
    QVERIFY(QTest::qWaitForWindowActive(&l.host));
    l.host.setFocus();
    QTRY_VERIFY(l.host.hasFocus());
    QVERIFY(l.labels->openForTesting(testCanvas({600, 400}), toPath(dir), "seg").ok);
    l.labels->setPaintClass(PaintClass::Background);
    QVERIFY(!l.labels->labelModeEnabled());

    // Label mode off: W is not ours.
    QTest::keyClick(&l.host, Qt::Key_W);
    QCOMPARE(l.labels->paintClass(), PaintClass::Background);
    QVERIFY(l.host.seen.contains(QKeySequence(Qt::Key_W)));

    // I turns label mode on (and approval painting off); W then picks ink.
    QTest::keyClick(&l.host, Qt::Key_I);
    QVERIFY(l.labels->labelModeEnabled());
    QCOMPARE(l.approvalStops, 1);
    // The first time, a hint lists the keys; the status bar keeps a mode line.
    QVERIFY(l.messages.last().contains(QStringLiteral("W ink")));
    QVERIFY(l.modeLine.startsWith(QStringLiteral("Ink labels: background")));
    QVERIFY(l.labels->labelModeAction()->isChecked());
    QTest::keyClick(&l.host, Qt::Key_W);
    QCOMPARE(l.labels->paintClass(), PaintClass::Ink);
    QTest::keyClick(&l.host, Qt::Key_Z);
    QVERIFY(l.labels->erase());
    QTest::keyClick(&l.host, Qt::Key_V);
    QCOMPARE(l.labels->paintClass(), PaintClass::Validation);
    QVERIFY(!l.labels->erase());
    QTest::keyClick(&l.host, Qt::Key_L);
    QCOMPARE(l.labels->tool(), InkLabelController::Tool::Polygon);
    QTest::keyClick(&l.host, Qt::Key_H);
    QVERIFY(!l.labels->showLabels());
    QTest::keyClick(&l.host, Qt::Key_I);
    QVERIFY(!l.labels->labelModeEnabled());
    QVERIFY(l.modeLine.isEmpty());
    QVERIFY(!l.labels->labelModeAction()->isChecked());
    // H still works outside label mode while a session is open.
    QTest::keyClick(&l.host, Qt::Key_H);
    QVERIFY(l.labels->showLabels());
    for (QShortcut* shortcut : l.labels->shortcutsForTesting()) {
        QVERIFY(!shortcut->autoRepeat());
        QCOMPARE(shortcut->context(), Qt::WindowShortcut);
    }

    // Another workspace is current: none of the keys are ours, so its own
    // bindings of the same letters keep working.
    l.workspace = false;
    l.labels->updateShortcutStates();
    QTest::keyClick(&l.host, Qt::Key_I);
    QVERIFY(!l.labels->labelModeEnabled());
    QVERIFY(l.host.seen.contains(QKeySequence(Qt::Key_I)));
    l.workspace = true;
    l.labels->updateShortcutStates();

    // The menu action toggles label mode too, and shows a refusal.
    l.labels->labelModeAction()->trigger();
    QVERIFY(l.labels->labelModeEnabled());
    l.labels->labelModeAction()->trigger();
    QVERIFY(!l.labels->labelModeEnabled());
    l.editing = true;
    const int stops = l.approvalStops;
    l.labels->labelModeAction()->trigger();
    QVERIFY(!l.labels->labelModeEnabled());
    QVERIFY(!l.labels->labelModeAction()->isChecked());
    // Refused before touching approval painting.
    QCOMPARE(l.approvalStops, stops);
}

void TestInkLabels::keysWorkOutsideTheWorkspaceWidget()
{
    // In VC3D the label keys live on the main window; the dock bar and its
    // popup and the status bar are siblings of the segment workspace. Focus
    // in any of them must leave the keys working.
    QTemporaryDir dir;
    Labeler l;
    auto* workspace = new QWidget(&l.host);
    workspace->setObjectName(QStringLiteral("segmentWorkspace"));
    auto* statusBarButton = new QPushButton(QStringLiteral("dock bar"), &l.host);
    statusBarButton->move(0, 150);
    l.host.resize(300, 200);
    l.host.show();
    QVERIFY(QTest::qWaitForWindowActive(&l.host));
    QVERIFY(l.labels->openForTesting(testCanvas({600, 400}), toPath(dir), "seg").ok);
    statusBarButton->setFocus();
    QTRY_VERIFY(statusBarButton->hasFocus());
    QTest::keyClick(statusBarButton, Qt::Key_I);
    QVERIFY(l.labels->labelModeEnabled());
    QTest::keyClick(statusBarButton, Qt::Key_F);
    QCOMPARE(l.labels->paintClass(), PaintClass::Background);
    QTest::keyClick(statusBarButton, Qt::Key_W);
    QCOMPARE(l.labels->paintClass(), PaintClass::Ink);
    QVERIFY(l.labels->applyShape(disc(200, 200, 30), Actor::User) > 0);
    QTest::keyClick(statusBarButton, Qt::Key_Z, Qt::ControlModifier);
    QCOMPARE(int(l.labels->history()->undoDepth()), 0);
}

void TestInkLabels::polygonKeysAreShortcutsOnlyWhileOpen()
{
    QTemporaryDir dir;
    Labeler l;
    auto* button = new QPushButton(QStringLiteral("dock control"), &l.host);
    l.host.resize(300, 200);
    l.host.show();
    QVERIFY(QTest::qWaitForWindowActive(&l.host));
    QVERIFY(l.labels->openForTesting(testCanvas({600, 400}), toPath(dir), "seg").ok);
    l.labels->setLabelModeEnabled(true);
    l.labels->setTool(InkLabelController::Tool::Polygon);
    button->setFocus();
    QTRY_VERIFY(button->hasFocus());

    // No polygon: Enter and Esc are not taken from the focused widget.
    QTest::keyClick(button, Qt::Key_Escape);
    QVERIFY(l.host.seen.contains(QKeySequence(Qt::Key_Escape)));

    for (const cv::Point2d p : {cv::Point2d(100, 100), cv::Point2d(300, 100), cv::Point2d(300, 300), cv::Point2d(50, 50)}) {
        QVERIFY(l.labels->pressForTesting(p));
        l.labels->releaseForTesting();
    }
    QCOMPARE(int(l.labels->openPolygon().size()), 4);
    // With focus in a dock control, Backspace drops the last point and Enter
    // closes the polygon.
    QTest::keyClick(button, Qt::Key_Backspace);
    QCOMPARE(int(l.labels->openPolygon().size()), 3);
    QTest::keyClick(button, Qt::Key_Return);
    QVERIFY(l.labels->openPolygon().empty());
    QCOMPARE(int(l.labels->history()->undoDepth()), 1);
    QCOMPARE(int(l.labels->raster()->at(250, 150)), 3);

    // Esc cancels an open polygon.
    QVERIFY(l.labels->pressForTesting({10, 10}));
    l.labels->releaseForTesting();
    QTest::keyClick(button, Qt::Key_Escape);
    QVERIFY(l.labels->openPolygon().empty());
    QCOMPARE(int(l.labels->history()->undoDepth()), 1);
}

void TestInkLabels::segmentSwitchKeepsAutosave()
{
    QTemporaryDir dir;
    Labeler l;
    const auto canvas = testCanvas({1200, 900});
    QVERIFY(l.labels->openForTesting(canvas, toPath(dir), "a").ok);
    const StorePaths pathsA = l.labels->session()->paths;
    QVERIFY(l.labels->applyShape(capsule({100, 100}, {1100, 800}, 40), Actor::User) > 0);
    const auto counts = l.labels->session()->counts;

    // Switching segment checkpoints the unsaved edits.
    QVERIFY(l.labels->openForTesting(canvas, toPath(dir), "b").ok);
    QVERIFY(l.messages.join('\n').contains(QStringLiteral("kept as autosave")));
    // Switching straight back waits for that write, then restores it
    // without asking (Fail would otherwise refuse).
    InkLabelController::OpenOptions options;
    options.recover = RecoverPolicy::Fail;
    const auto back = l.labels->openForTesting(canvas, toPath(dir), "a", options);
    QVERIFY2(back.ok, qPrintable(back.message));
    QVERIFY(back.restoredAutosave);
    QVERIFY(!back.readOnly);
    QCOMPARE(l.labels->session()->counts, counts);
    QVERIFY(l.labels->session()->dirty);
    QVERIFY(!fs::exists(pathsA.file(LabelKind::Ink)));
    QVERIFY(fs::exists(pathsA.file(LabelKind::Ink, Target::Autosave)));
}

void TestInkLabels::recoverPolicies()
{
    QTemporaryDir dir;
    const auto canvas = testCanvas({600, 400});
    StorePaths paths;
    std::array<uint64_t, 8> saved{}, edited{};
    {
        Labeler l;
        QVERIFY(l.labels->openForTesting(canvas, toPath(dir), "seg").ok);
        paths = l.labels->session()->paths;
        QVERIFY(l.labels->applyShape(disc(100, 100, 30), Actor::User) > 0);
        QVERIFY(l.labels->save().result().ok);
        QTRY_VERIFY(!l.labels->session()->dirty);
        saved = l.labels->session()->counts;
        QVERIFY(l.labels->applyShape(disc(400, 300, 50), Actor::User) > 0);
        edited = l.labels->session()->counts;
        // A periodic checkpoint, as after a crash.
        l.labels->runAutosaveTickForTesting(true);
        settleWrites(*l.labels, paths);
        QVERIFY(targetExists(paths, Target::Autosave));
        QVERIFY(!l.labels->lastAutosave().isNull());
    }
    // The destructor checkpointed nothing new; the earlier checkpoint stays.
    QTRY_VERIFY(targetExists(paths, Target::Autosave));

    InkLabelController::OpenOptions options;
    {
        Labeler l;
        options.recover = RecoverPolicy::Fail;
        const auto failed = l.labels->openForTesting(canvas, toPath(dir), "seg", options);
        QVERIFY(!failed.ok);
        QCOMPARE(failed.error, InkLabelController::OpenError::AutosavePending);
        QVERIFY(!failed.autosaveAtUtc.isEmpty());
        QVERIFY(l.labels->session() == nullptr);
    }
    {
        Labeler l;
        options.recover = RecoverPolicy::Restore;
        const auto restored = l.labels->openForTesting(canvas, toPath(dir), "seg", options);
        QVERIFY2(restored.ok, qPrintable(restored.message));
        QVERIFY(restored.restoredAutosave);
        QCOMPARE(l.labels->session()->counts, edited);
        QVERIFY(l.labels->session()->dirty);
        // Close without saving: the checkpoint is kept for next time.
        l.labels->closeSession();
        settleWrites(*l.labels, paths);
    }
    {
        Labeler l;
        options.recover = RecoverPolicy::Discard;
        const auto discarded = l.labels->openForTesting(canvas, toPath(dir), "seg", options);
        QVERIFY2(discarded.ok, qPrintable(discarded.message));
        QVERIFY(!discarded.restoredAutosave);
        QCOMPARE(l.labels->session()->counts, saved);
        QVERIFY(!targetExists(paths, Target::Autosave));
    }
}

void TestInkLabels::autosaveBeforeUndoIsNotRestored()
{
    // Paint three strokes, checkpoint, undo two and save: the checkpoint
    // holds the undone strokes and has the larger state serial, but the save
    // supersedes it, and a later open must not bring the strokes back.
    QTemporaryDir dir;
    const auto canvas = testCanvas({600, 400});
    Labeler l;
    QVERIFY(l.labels->openForTesting(canvas, toPath(dir), "seg").ok);
    const StorePaths paths = l.labels->session()->paths;
    QVERIFY(l.labels->applyShape(disc(100, 100, 20), Actor::User) > 0);
    const auto afterOne = l.labels->session()->counts;
    QVERIFY(l.labels->applyShape(disc(300, 200, 20), Actor::User) > 0);
    QVERIFY(l.labels->applyShape(disc(500, 300, 20), Actor::User) > 0);
    l.labels->runAutosaveTickForTesting(true);
    settleWrites(*l.labels, paths);
    QVERIFY(targetExists(paths, Target::Autosave));
    QVERIFY(l.labels->undo(Actor::User));
    QVERIFY(l.labels->undo(Actor::User));
    QCOMPARE(l.labels->session()->counts, afterOne);
    QVERIFY(l.labels->save().result().ok);
    settleWrites(*l.labels, paths);
    QVERIFY(!targetExists(paths, Target::Autosave));

    // Switch away and back, in the same run and in a new one.
    QVERIFY(l.labels->openForTesting(canvas, toPath(dir), "other").ok);
    InkLabelController::OpenOptions fail;
    fail.recover = RecoverPolicy::Fail;
    const auto back = l.labels->openForTesting(canvas, toPath(dir), "seg", fail);
    QVERIFY2(back.ok, qPrintable(back.message));
    QVERIFY(!back.restoredAutosave);
    QVERIFY(!l.labels->session()->dirty);
    QCOMPARE(l.labels->session()->counts, afterOne);
    l.labels->closeSession();
    settleWrites(*l.labels, paths);

    Labeler fresh;
    const auto reopened = fresh.labels->openForTesting(canvas, toPath(dir), "seg", fail);
    QVERIFY2(reopened.ok, qPrintable(reopened.message));
    QVERIFY(!reopened.restoredAutosave);
    QCOMPARE(fresh.labels->session()->counts, afterOne);

    // An undo back to the saved state leaves a checkpoint stale: closing the
    // clean session deletes it, so it cannot come back either.
    QVERIFY(fresh.labels->applyShape(disc(400, 100, 20), Actor::User) > 0);
    fresh.labels->runAutosaveTickForTesting(true);
    settleWrites(*fresh.labels, paths);
    QVERIFY(targetExists(paths, Target::Autosave));
    QVERIFY(fresh.labels->undo(Actor::User));
    QVERIFY(!fresh.labels->session()->dirty);
    fresh.labels->closeSession();
    QVERIFY(!targetExists(paths, Target::Autosave));
}

void TestInkLabels::nestedOpenRefusedDuringRestorePrompt()
{
    QTemporaryDir dir;
    const auto canvas = testCanvas({600, 400});
    StorePaths paths;
    {
        Labeler l;
        QVERIFY(l.labels->openForTesting(canvas, toPath(dir), "seg").ok);
        paths = l.labels->session()->paths;
        QVERIFY(l.labels->applyShape(disc(100, 100, 30), Actor::User) > 0);
        l.labels->runAutosaveTickForTesting(true);
        settleWrites(*l.labels, paths);
    }
    QVERIFY(targetExists(paths, Target::Autosave));

    // The restore question runs a nested event loop. An open that arrives
    // meanwhile (a bridge call, a key) is refused instead of being replaced
    // silently when the question is answered.
    Labeler l;
    l.host.show();
    InkLabelController::OpenOutcome nested;
    bool answered = false;
    QTimer poll;
    poll.setInterval(20);
    QObject::connect(&poll, &QTimer::timeout, [&] {
        auto* box = l.host.findChild<QMessageBox*>(QStringLiteral("inkLabelRecoverPrompt"));
        if (!box || !box->isVisible() || answered)
            return;
        nested = l.labels->openForTesting(canvas, toPath(dir), "other");
        for (QAbstractButton* button : box->buttons())
            if (button->text() == QStringLiteral("Restore"))
                button->click();
        answered = true;
    });
    poll.start();
    const auto outer = l.labels->openForTesting(canvas, toPath(dir), "seg");  // RecoverPolicy::Ask
    poll.stop();
    QVERIFY(answered);
    QCOMPARE(nested.error, InkLabelController::OpenError::Busy);
    QVERIFY2(outer.ok, qPrintable(outer.message));
    QVERIFY(outer.restoredAutosave);
    QCOMPARE(l.labels->session()->segment, std::string("seg"));
}

void TestInkLabels::openWritesNothingUntilFirstEdit()
{
    QTemporaryDir dir;
    Labeler l;
    QVERIFY(l.labels->openForTesting(testCanvas({600, 400}), toPath(dir), "seg").ok);
    const StorePaths paths = l.labels->session()->paths;
    l.labels->setLabelModeEnabled(true);
    // Opening and turning label mode on create no folder and no lock file.
    QVERIFY(!fs::exists(toPath(dir) / "ink_labels"));
    QVERIFY(l.labels->applyShape(disc(100, 100, 20), Actor::User) > 0);
    QVERIFY(fs::is_directory(paths.dir));
    QVERIFY(fs::exists(paths.lock()));
}

void TestInkLabels::unusableAutosaveIsMovedAside()
{
    QTemporaryDir dir;
    const StorePaths paths = storePathsFor(toPath(dir), "seg");
    // A checkpoint written for another canvas (a re-render before the first
    // save), and one whose sidecar is unreadable (a crash between renames).
    TileRaster other({800, 600});
    History history;
    commitTo(other, history, disc(100, 100, 30), paintOp(PaintClass::Ink, false, false));
    QVERIFY(saveLabels(saveRequest(other, paths, keyFor({800, 600}), Target::Autosave, 5)).ok);
    {
        Labeler l;
        const auto out = l.labels->openForTesting(testCanvas({600, 400}), toPath(dir), "seg");
        QVERIFY2(out.ok, qPrintable(out.message));
        QVERIFY(!out.restoredAutosave);
        QVERIFY(out.warnings.join(' ').contains(QStringLiteral("cannot be restored")));
        QVERIFY(!targetExists(paths, Target::Autosave));
        fs::path aside = paths.file(LabelKind::Ink, Target::Autosave);
        aside += ".unusable";
        QVERIFY(fs::exists(aside));
        QCOMPARE(l.labels->session()->counts[0], uint64_t(600) * 400);
    }
    QVERIFY(saveLabels(saveRequest(other, paths, keyFor({800, 600}), Target::Autosave, 6)).ok);
    {
        std::ofstream broken(paths.sidecar(Target::Autosave));
        broken << "{ not json";
    }
    Labeler l;
    const auto out = l.labels->openForTesting(testCanvas({600, 400}), toPath(dir), "seg");
    QVERIFY2(out.ok, qPrintable(out.message));
    QVERIFY(out.warnings.join(' ').contains(QStringLiteral("cannot be restored")));
    fs::path aside = paths.sidecar(Target::Autosave);
    aside += ".unusable";
    QVERIFY(fs::exists(aside));
}

void TestInkLabels::labelRootLeavesTheRemoteCache()
{
    QCOMPARE(labelRootFor("/data/proj/p.volpkg.json", "/cache", "/appdata"), fs::path("/data/proj"));
    QCOMPARE(labelRootFor("/cache/open_data/projects/scroll_a.volpkg.json", "/cache", "/appdata"), fs::path("/appdata/scroll_a"));
    QCOMPARE(labelRootFor("/cache/open_data/projects/x.volpkg.json", "/cache/", "/appdata"), fs::path("/appdata/x"));
    QCOMPARE(labelRootFor("/cachefoo/p.volpkg.json", "/cache", "/appdata"), fs::path("/cachefoo"));
    QCOMPARE(labelRootFor("/data/p.volpkg.json", "", "/appdata"), fs::path("/data"));
}

void TestInkLabels::closeGuardSavesThenCloses()
{
    QTemporaryDir dir;
    Labeler l;
    l.host.show();
    QVERIFY(l.labels->openForTesting(testCanvas({600, 400}), toPath(dir), "seg").ok);
    bool retried = false;
    QVERIFY(l.labels->readyToClose([&retried] { retried = true; }));  // clean: closes at once
    QVERIFY(l.labels->applyShape(disc(300, 200, 40), Actor::User) > 0);

    QVERIFY(!l.labels->readyToClose([&retried] { retried = true; }));
    auto* box = l.host.findChild<QMessageBox*>(QStringLiteral("inkLabelCloseGuard"));
    QVERIFY(box);
    QTRY_VERIFY(box->isVisible());
    box->button(QMessageBox::Save)->click();
    QTRY_VERIFY(retried);
    QVERIFY(!l.labels->session()->dirty);
    QVERIFY(fs::exists(l.labels->session()->paths.file(LabelKind::Ink)));
    // The callback's close() comes back here and passes.
    QVERIFY(l.labels->readyToClose([] {}));

    // Discard lets exactly one close through: if another guard cancels that
    // close, new edits are asked about again.
    QVERIFY(l.labels->applyShape(disc(100, 100, 40), Actor::User) > 0);
    retried = false;
    QVERIFY(!l.labels->readyToClose([&retried] { retried = true; }));
    box = l.host.findChild<QMessageBox*>(QStringLiteral("inkLabelCloseGuard"));
    QVERIFY(box);
    QTRY_VERIFY(box->isVisible());
    box->button(QMessageBox::Discard)->click();
    QTRY_VERIFY(retried);
    QVERIFY(l.labels->readyToClose([] {}));
    QVERIFY(l.labels->applyShape(disc(500, 300, 40), Actor::User) > 0);
    QVERIFY(!l.labels->readyToClose([] {}));
    QTRY_VERIFY(l.host.findChild<QMessageBox*>(QStringLiteral("inkLabelCloseGuard")));
    for (auto* open : l.host.findChildren<QMessageBox*>(QStringLiteral("inkLabelCloseGuard")))
        open->reject();
}

void TestInkLabels::otherToolTakesThePress()
{
    QTemporaryDir dir;
    Labeler l;
    QVERIFY(l.labels->openForTesting(testCanvas({600, 400}), toPath(dir), "seg").ok);
    l.labels->setLabelModeEnabled(true);
    QVERIFY(l.labels->pressForTesting({100, 100}));
    l.labels->releaseForTesting();
    QCOMPARE(int(l.labels->history()->undoDepth()), 1);

    // Editing or approval painting was turned on elsewhere: the press is
    // passed on and label mode turns off.
    l.otherTool = true;
    QVERIFY(!l.labels->pressForTesting({200, 200}));
    QVERIFY(!l.labels->labelModeEnabled());
    QCOMPARE(int(l.labels->history()->undoDepth()), 1);
    // Approval painting is turned off when label mode turns on.
    const int stops = l.approvalStops;
    l.otherTool = false;
    l.labels->setLabelModeEnabled(true);
    QVERIFY(l.labels->labelModeEnabled());
    QCOMPARE(l.approvalStops, stops + 1);
    l.labels->setLabelModeEnabled(false);
    // Segmentation editing refuses label mode before anything else changes.
    l.editing = true;
    l.labels->setLabelModeEnabled(true);
    QVERIFY(!l.labels->labelModeEnabled());
    QCOMPARE(l.approvalStops, stops + 1);
    QVERIFY(l.messages.join('\n').contains(QStringLiteral("Turn off segmentation editing")));
}

void TestInkLabels::displayTilesPoolExactly()
{
    // The cache keeps at least twice the visible tiles (a 4K view needs
    // about 600 at level 0).
    QCOMPARE(InkLabelOverlayController::cacheCapacityFor(0), InkLabelOverlayController::kMinCachedTiles);
    QCOMPARE(InkLabelOverlayController::cacheCapacityFor(100), InkLabelOverlayController::kMinCachedTiles);
    QCOMPARE(InkLabelOverlayController::cacheCapacityFor(600), 1200);

    QCOMPARE(InkLabelOverlayController::lodFor(4.0), 0);
    QCOMPARE(InkLabelOverlayController::lodFor(1.0), 0);
    QCOMPARE(InkLabelOverlayController::lodFor(0.51), 0);
    QCOMPARE(InkLabelOverlayController::lodFor(0.5), 1);
    QCOMPARE(InkLabelOverlayController::lodFor(0.26), 1);
    QCOMPARE(InkLabelOverlayController::lodFor(0.125), 3);
    QCOMPARE(InkLabelOverlayController::lodFor(1e-6), 8);
    QCOMPARE(InkLabelOverlayController::lodFor(0.0), 0);

    QTemporaryDir dir;
    Labeler l;
    const cv::Size size(700, 620);
    QVERIFY(l.labels->openForTesting(testCanvas(size), toPath(dir), "seg").ok);
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> ux(0, size.width), uy(0, size.height), ur(1, 40);
    l.labels->setPaintClass(PaintClass::Background);
    l.labels->applyShape(polygon({{50, 30}, {690, 60}, {600, 600}, {20, 500}}), Actor::User);
    for (int i = 0; i < 60; ++i) {
        l.labels->setPaintClass(PaintClass(i % 3));
        l.labels->setErase(i % 7 == 0);
        l.labels->applyShape(disc(ux(rng), uy(rng), ur(rng)), Actor::User);
    }
    const cv::Mat_<uint8_t> values = fullImage(*l.labels->raster());
    InkLabelOverlayController& overlay = *l.labels->overlay();
    for (int level = 0; level <= 3; ++level) {
        const int span = kTileSize << level, step = 1 << level;
        for (int j = 0; j * span < size.height; ++j)
            for (int i = 0; i * span < size.width; ++i) {
                const QImage image = overlay.displayTileForTesting(level, i, j);
                QCOMPARE(image.format(), QImage::Format_Indexed8);
                QCOMPARE(image.width(), (std::min(span, size.width - i * span) + step - 1) / step);
                QCOMPARE(image.height(), (std::min(span, size.height - j * span) + step - 1) / step);
                for (int y = 0; y < image.height(); ++y)
                    for (int x = 0; x < image.width(); ++x) {
                        uint8_t code = kNeutralCode;
                        for (int py = 0; py < step; ++py)
                            for (int px = 0; px < step; ++px) {
                                const int lx = i * span + x * step + px, ly = j * span + y * step + py;
                                if (lx < size.width && ly < size.height)
                                    code = poolCodes(code, summaryCode(values(ly, lx)));
                            }
                        if (image.pixelIndex(x, y) != code)
                            QFAIL(qPrintable(QStringLiteral("level %1 tile (%2, %3) pixel (%4, %5): %6 != %7")
                                                 .arg(level)
                                                 .arg(i)
                                                 .arg(j)
                                                 .arg(x)
                                                 .arg(y)
                                                 .arg(image.pixelIndex(x, y))
                                                 .arg(code)));
                    }
            }
    }

    // A style change swaps the palette only; a paint bumps every level.
    const int builds = overlay.buildsForTesting();
    const uint32_t rev = overlay.revisionForTesting(2, 0, 0);
    l.labels->setDisplay(true, false, true, true, 0.3);
    const QImage restyled = overlay.displayTileForTesting(2, 0, 0);
    QCOMPARE(overlay.buildsForTesting(), builds);
    QCOMPARE(overlay.revisionForTesting(2, 0, 0), rev);
    QCOMPARE(restyled.colorTable(), InkLabelOverlayController::colorTableFor(false, true, true, 0.3));
    QCOMPARE(qAlpha(restyled.colorTable()[0b1011]), 36);  // ink hidden: supervised blue at 120 * 0.3
    std::array<uint32_t, InkLabelOverlayController::kLevels> before{};
    for (int level = 0; level < InkLabelOverlayController::kLevels; ++level)
        before[std::size_t(level)] = overlay.revisionForTesting(level, 0, 0);
    // Toggle validation at (10, 10) so the paint surely changes pixels.
    l.labels->setPaintClass(PaintClass::Validation);
    l.labels->setErase((values(10, 10) & kValidation) != 0);
    QVERIFY(l.labels->applyShape(disc(10, 10, 3), Actor::User) > 0);
    for (int level = 0; level < InkLabelOverlayController::kLevels; ++level)
        QVERIFY(overlay.revisionForTesting(level, 0, 0) > before[std::size_t(level)]);
    (void)overlay.displayTileForTesting(2, 0, 0);
    QCOMPARE(overlay.buildsForTesting(), builds + 1);
}

void TestInkLabels::displayTilesMeetOnDevicePixels()
{
    // A fractional zoom and offset, as at scale 0.22: neighboring tiles must
    // end and start on the same whole device pixel, or a 1 px seam shows.
    const QTransform toDevice = QTransform::fromTranslate(13.37, -7.25).scale(0.2213, 0.2213);
    const qreal tileScene = 1024.0 * 1.0007;
    const int pixels = 256;
    for (int j = 0; j < 40; ++j) {
        const QPointF top(-3001.3, 977.7 + j * tileScene);
        const QPointF bottom(-3001.3, 977.7 + (j + 1) * tileScene);
        const QPointF o = InkLabelOverlayController::snapToDevicePixel(top, toDevice);
        const QPointF far = InkLabelOverlayController::snapToDevicePixel(bottom, toDevice);
        const qreal scaleY = (far.y() - o.y()) / pixels;
        // Where this tile's last image row ends, and where the next one begins.
        const qreal end = toDevice.map(QPointF(o.x(), o.y() + pixels * scaleY)).y();
        const qreal nextStart = toDevice.map(InkLabelOverlayController::snapToDevicePixel(bottom, toDevice)).y();
        QVERIFY(std::abs(end - nextStart) < 1e-6);
        QVERIFY(std::abs(nextStart - std::round(nextStart)) < 1e-6);
        // Snapping moves a corner by at most half a device pixel.
        QVERIFY(std::abs(toDevice.map(o).y() - toDevice.map(top).y()) <= 0.5 + 1e-9);
    }
    // A singular transform leaves the point alone.
    QCOMPARE(InkLabelOverlayController::snapToDevicePixel({3.25, 4.5}, QTransform(0, 0, 0, 0, 0, 0)), QPointF(3.25, 4.5));
}

void TestInkLabels::statusHostRoutesInkLabels()
{
    Labeler l;
    QWidget window;
    window.resize(1200, 1000);
    auto* layout = new QVBoxLayout(&window);
    auto* host = new StatusDockPanelHost(new QWidget, &window);
    layout->addWidget(host);
    auto* dock = new QDockWidget(QStringLiteral("Ink Detection"), &window);
    dock->setObjectName(QStringLiteral("dockWidgetInkDetection"));
    auto* scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    auto* panel = new InkLabelPanel(l.labels.get());
    scroll->setWidget(panel);
    dock->setWidget(scroll);
    host->addDock(dock); // Same reparenting step as ordinary CWindow startup.
    QVERIFY(!dock->widget());
    QVERIFY(!dock->findChild<InkLabelPanel*>());
    QCOMPARE(host->contentForDock(dock), static_cast<QWidget*>(scroll));
    QCOMPARE(host->contentForDock(dock)->findChild<InkLabelPanel*>(), panel);
    QVERIFY(!host->panelWidgetForDock(dock));
    window.show();
    host->showDock(dock);
    auto* attached = host->panelWidgetForDock(dock);
    QVERIFY(attached && attached != dock && attached->isAncestorOf(panel));
    QTRY_VERIFY(attached->isVisible());
    QVERIFY(!attached->grab().isNull());
    for (const QString& name : {QStringLiteral("inkLabelGroup"), QStringLiteral("inkLabelAssistGroup"),
                               QStringLiteral("inkLabelFilesGroup")}) {
        auto* group = panel->findChild<CollapsibleSettingsGroup*>(name);
        QVERIFY(group);
        group->setExpanded(true);
        QVERIFY(group->isExpanded());
    }
    QVERIFY(host->setDockDetached(dock, true));
    QVERIFY(host->isDockDetached(dock));
    QCOMPARE(host->contentForDock(dock), static_cast<QWidget*>(scroll));
    QCOMPARE(host->panelWidgetForDock(dock), static_cast<QWidget*>(dock));
    QCOMPARE(dock->widget(), static_cast<QWidget*>(scroll));
    QVERIFY(dock->isFloating());
    QTRY_VERIFY(dock->isVisible());
    QVERIFY(!host->panelWidgetForDock(dock)->grab().isNull());
    QVERIFY(host->setDockDetached(dock, false));
    QVERIFY(!host->isDockDetached(dock));
    QVERIFY(!dock->widget());
    QCOMPARE(host->panelWidgetForDock(dock), attached);
    QCOMPARE(host->contentForDock(dock)->findChild<InkLabelPanel*>(), panel);
    QVERIFY(host->setDockDetached(dock, true));
    dock->hide(); // Existing close/visibility path must attach content again.
    QTRY_VERIFY(!host->isDockDetached(dock));
    QVERIFY(!host->panelWidgetForDock(dock));
    host->showDock(dock);
    QCOMPARE(host->panelWidgetForDock(dock), attached);

    auto* other = new QDockWidget(QStringLiteral("Other"), &window);
    other->setObjectName(QStringLiteral("otherPanelForTest"));
    other->setWidget(new QWidget);
    host->addDock(other);
    host->showDock(other);
    QVERIFY(!host->panelWidgetForDock(dock)); // Never capture another active page.
    QVERIFY(host->panelWidgetForDock(other));
    QDockWidget unknown;
    QVERIFY(!host->contentForDock(&unknown));
    QVERIFY(!host->setDockDetached(&unknown, true));
    QVERIFY(!host->panelWidgetForDock(&unknown));
}

void TestInkLabels::panelAgentUndoPreservesHumanEntry()
{
    QTemporaryDir dir;
    Labeler l;
    InkLabelPanel panel(l.labels.get());
    QVERIFY(l.labels->openForTesting(testCanvas({80, 80}), toPath(dir), "undo-guard").ok);
    auto* undo = panel.findChild<QPushButton*>(QStringLiteral("inkLabelControl_undo"));
    QVERIFY(undo);
    QSignalSpy clicked(undo, &QPushButton::clicked);
    QVERIFY(l.labels->applyShape(disc(20, 20, 3), Actor::User) > 0);
    const auto human = fullImage(*l.labels->raster());
    const auto serial = l.labels->session()->stateSerial;
    const auto depth = l.labels->history()->undoDepth();
    QVERIFY(undo->isEnabled());
    QVERIFY(!panel.clickUndoForAgent());
    QCOMPARE(clicked.count(), 0);
    QCOMPARE(l.labels->session()->stateSerial, serial);
    QCOMPARE(l.labels->history()->undoDepth(), depth);
    QCOMPARE(l.labels->history()->redoDepth(), std::size_t(0));
    QCOMPARE(l.labels->history()->top()->actor, Actor::User);
    QVERIFY(sameImage(human, fullImage(*l.labels->raster())));

    QVERIFY(l.labels->applyShape(disc(40, 40, 3), Actor::Agent) > 0);
    QVERIFY(panel.clickUndoForAgent());
    QCOMPARE(clicked.count(), 1); // Positive path still emits the widget signal.
    QVERIFY(sameImage(human, fullImage(*l.labels->raster())));
    QCOMPARE(l.labels->history()->undoDepth(), depth);
    QCOMPARE(l.labels->history()->top()->actor, Actor::User);
    QVERIFY(!panel.clickUndoForAgent());
    QCOMPARE(clicked.count(), 1);
    QVERIFY(sameImage(human, fullImage(*l.labels->raster())));
    QVERIFY(l.labels->undo(Actor::Agent, true)); // Explicit force remains separate.
}

void TestInkLabels::namedPanelControls()
{
    Labeler l;
    InkLabelPanel panel(l.labels.get());
    for (const QString& name : {QStringLiteral("ink"), QStringLiteral("background"),
                               QStringLiteral("validation"), QStringLiteral("erase"),
                               QStringLiteral("brush"), QStringLiteral("polygon"), QStringLiteral("fill"),
                               QStringLiteral("show"), QStringLiteral("show_ink"),
                               QStringLiteral("show_supervision"), QStringLiteral("show_validation"),
                               QStringLiteral("keep_ink"), QStringLiteral("undo"), QStringLiteral("redo")})
        QVERIFY2(panel.findChild<QAbstractButton*>(QStringLiteral("inkLabelControl_") + name), qPrintable(name));
    QVERIFY(panel.findChild<QSlider*>(QStringLiteral("inkLabelControl_opacity")));
    QVERIFY(panel.findChild<QSlider*>(QStringLiteral("inkLabelRadius")));
    QVERIFY(!panel.findChild<QWidget*>(QStringLiteral("inkLabelControl_import")));
}

void TestInkLabels::viewportStrokeRefusesMissingViewer()
{
    QTemporaryDir dir;
    Labeler l;
    const auto result = l.labels->dispatchViewportStroke(nullptr, {QPointF(10, 10)});
    QVERIFY(!result.ok);
    QVERIFY(!result.error.isEmpty());
    QCOMPARE(result.delivered, 0);
    QCOMPARE(result.handled, 0);
    QVERIFY(!l.labels->session());
    QVERIFY(!l.labels->strokeOpen());
    QVERIFY(l.labels->openForTesting(testCanvas({80, 80}), toPath(dir), "viewport-refusal").ok);
    l.labels->setLabelModeEnabled(true);
    const auto before = fullImage(*l.labels->raster());
    const auto serial = l.labels->session()->stateSerial;
    const auto missing = l.labels->dispatchViewportStroke(nullptr, {QPointF(10, 10)});
    QVERIFY(!missing.ok);
    QCOMPARE(missing.delivered, 0);
    QVERIFY(sameImage(before, fullImage(*l.labels->raster())));
    QCOMPARE(l.labels->session()->stateSerial, serial);
    l.labels->setTool(InkLabelController::Tool::Polygon);
    const auto wrongTool = l.labels->dispatchViewportStroke(nullptr, {QPointF(10, 10)});
    QVERIFY(!wrongTool.ok);
    QCOMPARE(wrongTool.delivered, 0);
    QVERIFY(sameImage(before, fullImage(*l.labels->raster())));
}

void TestInkLabels::panelDrivesController()
{
    QTemporaryDir dir;
    Labeler l;
    InkLabelPanel panel(l.labels.get());
    for (QWidget* w : panel.findChildren<QWidget*>()) {
        if (qobject_cast<QPushButton*>(w) || qobject_cast<QCheckBox*>(w) || qobject_cast<QSlider*>(w))
            QVERIFY2(w->focusPolicy() == Qt::NoFocus, qPrintable(w->objectName()));
    }
    QVERIFY(panel.findChild<CollapsibleSettingsGroup*>(QStringLiteral("inkLabelGroup")));
    auto* assistGroup = panel.findChild<CollapsibleSettingsGroup*>(QStringLiteral("inkLabelAssistGroup"));
    QVERIFY(assistGroup && !assistGroup->isExpanded());  // folded until needed
    QVERIFY(l.labels->openForTesting(testCanvas({600, 400}), toPath(dir), "seg").ok);
    auto* folder = panel.findChild<QLabel*>(QStringLiteral("inkLabelFolder"));
    QVERIFY(folder->toolTip().endsWith(QStringLiteral("seg")));
    QCOMPARE(folder->text(), folder->toolTip());
    QVERIFY(folder->wordWrap());
    QVERIFY(folder->textInteractionFlags().testFlag(Qt::TextSelectableByMouse));
    const auto buttons = panel.findChildren<QPushButton*>();
    const auto byText = [&buttons](const QString& prefix) -> QPushButton* {
        for (auto* b : buttons)
            if (b->text().startsWith(prefix))
                return b;
        return nullptr;
    };
    for (const QString& cls : {QStringLiteral("Ink"), QStringLiteral("Background"), QStringLiteral("Validation"), QStringLiteral("Eraser")})
        QVERIFY2(!byText(cls)->icon().isNull(), qPrintable(cls));  // color swatches
    byText(QStringLiteral("Background"))->click();
    QCOMPARE(l.labels->paintClass(), PaintClass::Background);
    byText(QStringLiteral("Eraser"))->click();
    QVERIFY(l.labels->erase());
    byText(QStringLiteral("Polygon"))->click();
    QCOMPARE(l.labels->tool(), InkLabelController::Tool::Polygon);
    auto* radius = panel.findChild<QSlider*>(QStringLiteral("inkLabelRadius"));
    radius->setValue(radius->maximum());
    QCOMPARE(l.labels->radius(), kMaxRadius);
    l.labels->setRadius(16);
    QCOMPARE(radius->value(), 500);  // 16 px, sqrt(0.5 * 512), sits mid-way on the log scale

    l.labels->setErase(false);
    l.labels->setPaintClass(PaintClass::Ink);
    l.labels->applyShape(disc(100, 100, 10), Actor::User);
    QTRY_VERIFY(panel.findChild<QLabel*>(QStringLiteral("inkLabelCounts"))->text().startsWith(QStringLiteral("ink 3")));
    QCOMPARE(byText(QStringLiteral("Undo"))->text(), QStringLiteral("Undo (1)"));
    byText(QStringLiteral("Undo"))->click();
    QCOMPARE(int(l.labels->history()->undoDepth()), 0);
    QVERIFY(panel.findChild<QLabel*>(QStringLiteral("inkLabelStatus"))->text().contains(QStringLiteral("no unsaved changes")));
}

void TestInkLabels::panelSelectionFeedbackTracksController()
{
    QTemporaryDir dir;
    Labeler l;
    QVERIFY(l.labels->openForTesting(testCanvas({200, 160}), toPath(dir), "seg").ok);
    InkLabelPanel panel(l.labels.get());
    const auto button = [&panel](const QString& name) {
        return panel.findChild<QPushButton*>(name == QLatin1String("mode") ? QStringLiteral("inkLabelModeButton")
                                                                        : QStringLiteral("inkLabelControl_") + name);
    };
    const auto matches = [&button](const QString& name, bool selected) {
        const auto* b = button(name);
        return b && b->isChecked() == selected && b->text().contains(QChar(0x2713)) == selected &&
               !b->accessibleName().isEmpty() &&
               b->accessibleDescription() == (selected ? QStringLiteral("Selected") : QStringLiteral("Not selected"));
    };
    QVERIFY(matches(QStringLiteral("mode"), false));
    QVERIFY(matches(QStringLiteral("ink"), true));
    QVERIFY(matches(QStringLiteral("brush"), true));
    const QStringList classes{QStringLiteral("ink"), QStringLiteral("background"), QStringLiteral("validation")};
    for (const auto& name : classes) {
        QVERIFY(button(name));
        button(name)->click();
        for (const auto& other : classes)
            QVERIFY(matches(other, other == name));
    }
    // Controller/keyboard changes refresh under QSignalBlocker too.
    l.labels->setPaintClass(PaintClass::Ink);
    QVERIFY(matches(QStringLiteral("ink"), true));
    QVERIFY(matches(QStringLiteral("validation"), false));
    const QStringList tools{QStringLiteral("brush"), QStringLiteral("polygon"), QStringLiteral("fill")};
    for (const auto& name : tools) {
        QVERIFY(button(name));
        button(name)->click();
        for (const auto& other : tools)
            QVERIFY(matches(other, other == name));
    }
    l.labels->setTool(InkLabelController::Tool::Brush);
    QVERIFY(matches(QStringLiteral("brush"), true));
    QVERIFY(matches(QStringLiteral("fill"), false));
    QVERIFY(button(QStringLiteral("erase")));
    button(QStringLiteral("erase"))->click();
    QVERIFY(matches(QStringLiteral("erase"), true));
    l.labels->setErase(false);
    QVERIFY(matches(QStringLiteral("erase"), false));
    button(QStringLiteral("mode"))->click();
    QVERIFY(l.labels->labelModeEnabled());
    QVERIFY(matches(QStringLiteral("mode"), true));
    l.labels->setLabelModeEnabled(false);
    QVERIFY(matches(QStringLiteral("mode"), false));
    l.editing = true;
    button(QStringLiteral("mode"))->click();
    QVERIFY(!l.labels->labelModeEnabled());
    QVERIFY(matches(QStringLiteral("mode"), false)); // Refusal must not leave a selected marker.
    QCOMPARE(int(l.labels->history()->undoDepth()), 0);
    QVERIFY(!l.labels->session()->dirty);
}

void TestInkLabels::panelOpacityReadoutTracksController()
{
    QTemporaryDir dir;
    Labeler l;
    QVERIFY(l.labels->openForTesting(testCanvas({200, 160}), toPath(dir), "seg").ok);
    l.labels->setDisplay(true, true, true, true, 0.70);
    InkLabelPanel panel(l.labels.get());
    auto* slider = panel.findChild<QSlider*>(QStringLiteral("inkLabelControl_opacity"));
    auto* value = panel.findChild<QLabel*>(QStringLiteral("inkLabelOpacityValue"));
    QVERIFY(slider && value);
    QCOMPARE(value->text(), QStringLiteral("70%"));
    QVERIFY(!slider->accessibleName().isEmpty());
    QSignalSpy changed(l.labels.get(), &InkLabelController::displayChanged);
    slider->setValue(37);
    QCOMPARE(l.labels->opacity(), 0.37);
    QCOMPARE(value->text(), QStringLiteral("37%"));
    QCOMPARE(changed.size(), 1); // The readout must not create a display feedback loop.
    // External display updates also synchronize the value without valueChanged.
    l.labels->setDisplay(true, true, true, true, 0.0);
    QCOMPARE(slider->value(), 0);
    QCOMPARE(value->text(), QStringLiteral("0%"));
    slider->setValue(100);
    QCOMPARE(value->text(), QStringLiteral("100%"));
    QCOMPARE(l.labels->opacity(), 1.0);
    InkLabelPanel reopened(l.labels.get());
    QCOMPARE(reopened.findChild<QLabel*>(QStringLiteral("inkLabelOpacityValue"))->text(), QStringLiteral("100%"));
    QCOMPARE(int(l.labels->history()->undoDepth()), 0);
    QVERIFY(!l.labels->session()->dirty);
}

// A real catalog frame failure leaves the same null state consumed by the
// label controller after CWindow rejects activation. No CT or remote catalog.
void TestInkLabels::panelRetainsFailedActivationRefusal()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const auto project = vc::test::createTestProject(toPath(dir) / "fixture.vcobj");
    CState state;
    state.setVpkg(project.open());
    const auto attachment = *state.storeFor("surface");
    const auto session = state.objectStoreSession();
    const std::string uuid = vc::objects::util::newUuid();
    cv::Mat_<cv::Vec3f> points(3, 3);
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            points(row, col) = {float(col), float(row), 2.f};
    QuadSurface grid(points, {1.f, 1.f});
    auto request = session->newRequest(attachment);
    request.expectedObjects = {{uuid, std::nullopt}};
    request.operations = vc::objects::createSurfaceOperations(
        uuid, "fixture", {}, vc::objects::volumeCoordinates("bound-volume", 1),
        {"label-refusal-test"}, grid.meta, quad_surface_geometry(grid));
    session->submit(attachment, request);

    QString loadFailure;
    try {
        (void)SurfaceCatalog::open(session, attachment, {"different-volume", 1}, uuid);
    } catch (const std::exception& error) {
        loadFailure = QString::fromUtf8(error.what());
        state.clearActiveSurface();
        state.setSurface("segmentation", nullptr);
    }
    QVERIFY(loadFailure.contains(QStringLiteral("no unambiguous transform")));
    QVERIFY(state.activeSurface().expired());

    Labeler l(&state);
    InkLabelPanel panel(l.labels.get());
    auto* button = panel.findChild<QPushButton*>(QStringLiteral("inkLabelModeButton"));
    auto* banner = panel.findChild<QLabel*>(QStringLiteral("inkLabelBanner"));
    QVERIFY(button && banner);
    button->click();
    const QString expected = QStringLiteral("Select a segment to label ink.");
    QCOMPARE(l.messages.last(), expected);
    QCOMPARE(l.labels->labelModeRefusal(), expected);
    QVERIFY(!l.labels->labelModeEnabled());
    QVERIFY(!button->isChecked());
    QVERIFY(!l.labels->session());
    QCOMPARE(banner->text(), expected);
    QVERIFY(!banner->isHidden());
    QVERIFY(!fs::exists(toPath(dir) / "ink_labels"));

    // Recreating the panel must not lose a refusal when there is no session.
    InkLabelPanel reopened(l.labels.get());
    auto* reopenedBanner = reopened.findChild<QLabel*>(QStringLiteral("inkLabelBanner"));
    QCOMPARE(reopenedBanner->text(), expected);
    QVERIFY(!reopenedBanner->isHidden());

    const auto opened = SurfaceCatalog::open(session, attachment, {"bound-volume", 1}, uuid);
    state.setActiveSurface(uuid, opened.surface);
    state.setSurface("segmentation", opened.surface);
    button->click();
    QVERIFY(l.labels->labelModeEnabled());
    QVERIFY(button->isChecked());
    QVERIFY(l.labels->session());
    QVERIFY(l.labels->labelModeRefusal().isEmpty());
    QVERIFY(banner->isHidden());
    QVERIFY(reopenedBanner->isHidden());
    QCOMPARE(QString::fromStdString(l.labels->session()->paths.dir.string()),
             QString::fromStdString((fs::canonical(toPath(dir)) / "ink_labels" / uuid).string()));
    // Opening labels alone still creates no files or changes to geometry.
    QVERIFY(!fs::exists(toPath(dir) / "ink_labels"));
    QCOMPARE(opened.surface->rawPointsPtr()->size(), cv::Size(3, 3));
}

void TestInkLabels::panelRetainsToolRefusal()
{
    QTemporaryDir dir;
    Labeler l;
    QVERIFY(l.labels->openForTesting(testCanvas({200, 160}), toPath(dir), "seg").ok);
    InkLabelPanel panel(l.labels.get());
    auto* banner = panel.findChild<QLabel*>(QStringLiteral("inkLabelBanner"));
    auto* button = panel.findChild<QPushButton*>(QStringLiteral("inkLabelModeButton"));
    QVERIFY(banner && button);
    l.editing = true;
    l.labels->labelModeAction()->trigger();
    const QString expected = QStringLiteral("Turn off segmentation editing to label ink; editing changes the grid labels are attached to.");
    QCOMPARE(banner->text(), expected);
    QCOMPARE(l.labels->labelModeRefusal(), expected);
    QVERIFY(!banner->isHidden());
    QVERIFY(!button->isChecked());
    QVERIFY(!l.labels->labelModeEnabled());
    QCOMPARE(l.approvalStops, 0);
    // A successful retry clears the exact refusal in an already-open session.
    l.editing = false;
    button->click();
    QVERIFY(l.labels->labelModeEnabled());
    QVERIFY(l.labels->labelModeRefusal().isEmpty());
    QVERIFY(banner->isHidden());
}

// ---------------------------------------------------------------------------
// Assisted labeling
// ---------------------------------------------------------------------------

namespace {

// Paints exactly the pixels of `rect` with `op`, as one history entry.
void paintRect(TileRaster& raster, History& history, const cv::Rect& rect, PaintOp op)
{
    EditTxn txn(raster, "rect", Actor::User);
    txn.applyMask(rect.tl(), cv::Mat_<uint8_t>(rect.size(), uint8_t(255)), op);
    HistoryEntry entry = txn.commit(history.nextSerial());
    if (!entry.changes.empty())
        history.push(std::move(entry));
}

// A 10 x 8 grid at scale 0.1: a 100 x 80 px canvas.
vc::RenderCanvas assistCanvas()
{
    return vc::RenderCanvas::make({10, 8}, {0.1f, 0.1f}, 1.0);
}

PredictionGuide numberedGuide(int cols, int rows)
{
    PredictionGuide guide;
    guide.scalar.create(rows, cols);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            guide.scalar(y, x) = uint8_t(y * cols + x);
    guide.source = "numbered";
    return guide;
}

// A guide as large as the canvas grid stretch, 255 inside `ink` rects.
PredictionGuide guideWith(cv::Size canvas, const std::vector<cv::Rect>& ink)
{
    PredictionGuide guide;
    guide.scalar = cv::Mat_<uint8_t>::zeros(canvas);
    for (const auto& r : ink)
        guide.scalar(r & cv::Rect(cv::Point(0, 0), canvas)) = 255;
    guide.source = "fixture";
    return guide;
}

}  // namespace

void TestInkLabels::predictionGuideMapsWithFlips()
{
    const auto canvas = assistCanvas();
    QCOMPARE(canvas.size, cv::Size(100, 80));
    // A 5 x 4 prediction stretched over the 10 x 8 grid: each prediction
    // pixel spans 2 grid steps, so 20 x 20 label px.
    for (const bool hflip : {false, true})
        for (const bool vflip : {false, true}) {
            PredictionGuide guide = numberedGuide(5, 4);
            guide.hflip = hflip;
            guide.vflip = vflip;
            for (int y = 0; y < 80; y += 7)
                for (int x = 0; x < 100; x += 9) {
                    const cv::Vec2d g = canvas.canvasToGrid({x + 0.5, y + 0.5});
                    int i = std::clamp(int(std::floor(g[0] * 5 / 10)), 0, 4);
                    int j = std::clamp(int(std::floor(g[1] * 4 / 8)), 0, 3);
                    if (hflip)
                        i = 4 - i;
                    if (vflip)
                        j = 3 - j;
                    QCOMPARE(int(predictionAt(guide, canvas, {x + 0.5, y + 0.5})), j * 5 + i);
                }
            // The sampled rect agrees with the point lookup everywhere.
            const cv::Rect rect(13, 7, 61, 49);
            const cv::Mat_<uint8_t> sampled = sampleGuide(guide, canvas, rect);
            for (int y = 0; y < rect.height; ++y)
                for (int x = 0; x < rect.width; ++x)
                    QCOMPARE(sampled(y, x), predictionAt(guide, canvas, {rect.x + x + 0.5, rect.y + y + 0.5}));
        }
    // Grid (0, 0) is label (0, 0) here (canvas 100 = 10 / 0.1), so prediction
    // pixel (0, 0) covers label pixels 0..19 exactly.
    const PredictionGuide guide = numberedGuide(5, 4);
    QCOMPARE(int(predictionAt(guide, canvas, {19.5, 19.5})), 0);
    QCOMPARE(int(predictionAt(guide, canvas, {20.5, 0.5})), 1);
    QCOMPARE(int(predictionAt(guide, canvas, {0.5, 20.5})), 5);
}

void TestInkLabels::floodConnectivityAndBounds()
{
    // Guide pixels equal label pixels on a grid of 1 at scale 1.
    const auto canvas = vc::RenderCanvas::make({64, 64}, {1.f, 1.f}, 1.0);
    PredictionGuide guide;
    guide.scalar = cv::Mat_<uint8_t>::zeros(64, 64);
    // Two blocks touching only at a corner.
    guide.scalar(cv::Rect(10, 10, 5, 5)) = 200;
    guide.scalar(cv::Rect(15, 15, 5, 5)) = 200;
    TileRaster raster(canvas.size);
    FloodOptions options;
    options.threshold = 128;
    auto four = floodRegion(raster, &guide, GuideKind::Prediction, canvas, {11, 11}, {}, options);
    QVERIFY(four);
    QCOMPARE(four->count(), uint64_t(25));
    QVERIFY(!four->reachedBound);
    options.connectivity = 8;
    auto eight = floodRegion(raster, &guide, GuideKind::Prediction, canvas, {11, 11}, {}, options);
    QVERIFY(eight);
    QCOMPARE(eight->count(), uint64_t(50));
    // Below the threshold from a dark seed: everything else.
    options.above = false;
    auto dark = floodRegion(raster, &guide, GuideKind::Prediction, canvas, {0, 0}, {}, options);
    QVERIFY(dark);
    QCOMPARE(dark->count(), uint64_t(64 * 64 - 50));
    // A seed that does not pass gives nothing.
    QVERIFY(!floodRegion(raster, &guide, GuideKind::Prediction, canvas, {11, 11}, {}, options));
    // "Unlabeled only" stops at labeled pixels.
    options.above = true;
    options.connectivity = 4;
    options.unlabeledOnly = true;
    History labeled;
    paintRect(raster, labeled, cv::Rect(10, 12, 5, 1), paintOp(PaintClass::Background, false, false));
    auto unlabeled = floodRegion(raster, &guide, GuideKind::Prediction, canvas, {11, 11}, {}, options);
    QVERIFY(unlabeled);
    QCOMPARE(unlabeled->count(), uint64_t(10));
    // An explicit bound clips the region.
    options.unlabeledOnly = false;
    auto clipped = floodRegion(raster, &guide, GuideKind::Prediction, canvas, {11, 11}, cv::Rect(0, 0, 13, 64), options);
    QVERIFY(clipped);
    QCOMPARE(clipped->count(), uint64_t(3 * 5));
    QCOMPARE(clipped->mask.size(), cv::Size(13, 64));
    // It stopped at the bound, not at the canvas edge: the caller says so.
    QVERIFY(clipped->reachedBound);

    // The window cap: a 5000 px wide passable band fills only 4096 px of it.
    const auto wide = vc::RenderCanvas::make({5000, 40}, {1.f, 1.f}, 1.0);
    PredictionGuide band;
    band.scalar = cv::Mat_<uint8_t>(40, 5000, uint8_t(255));
    TileRaster wideRaster(wide.size);
    auto capped = floodRegion(wideRaster, &band, GuideKind::Prediction, wide, {2500, 20}, {}, FloodOptions{});
    QVERIFY(capped);
    QCOMPARE(capped->mask.cols, kMaxAssistWindow);
    QCOMPARE(capped->origin.x, 2500 - kMaxAssistWindow / 2);
    QCOMPARE(capped->count(), uint64_t(kMaxAssistWindow) * 40);
}

void TestInkLabels::floodFollowsEqualLabels()
{
    const auto canvas = vc::RenderCanvas::make({60, 40}, {1.f, 1.f}, 1.0);
    TileRaster raster(canvas.size);
    History history;
    commitTo(raster, history, polygon({{0, 0}, {30, 0}, {30, 40}, {0, 40}}), paintOp(PaintClass::Background, false, false));
    commitTo(raster, history, disc(15, 20, 6), paintOp(PaintClass::Ink, false, false));
    const cv::Mat_<uint8_t> values = fullImage(raster);
    FloodOptions options;
    // Seeded in the background half: the ring around the ink disc, not the
    // disc and not the unlabeled half.
    auto region = floodRegion(raster, nullptr, GuideKind::Labels, canvas, {2, 2}, {}, options);
    QVERIFY(region);
    uint64_t background = 0;
    for (int y = 0; y < 40; ++y)
        for (int x = 0; x < 60; ++x)
            background += values(y, x) == kSupervision;
    QCOMPARE(region->count(), background);
    for (int y = 0; y < 40; ++y)
        for (int x = 0; x < 60; ++x)
            QCOMPARE(region->mask(y, x) != 0, values(y, x) == kSupervision);
    // The prediction guide without a prediction refuses.
    QVERIFY(!floodRegion(raster, nullptr, GuideKind::Prediction, canvas, {2, 2}, {}, options));
}

void TestInkLabels::suggestionsFilterAndAccept()
{
    const auto canvas = vc::RenderCanvas::make({200, 120}, {1.f, 1.f}, 1.0);
    // A big blob, a small blob, a 1 px line (removed by the opening) and salt
    // noise (removed by the median).
    PredictionGuide guide = guideWith(canvas.size, {cv::Rect(20, 20, 40, 30), cv::Rect(120, 70, 6, 6)});
    guide.scalar(cv::Rect(80, 10, 1, 100)) = 255;
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> ux(0, 199), uy(0, 119);
    for (int i = 0; i < 300; ++i)
        guide.scalar(uy(rng), ux(rng)) = 255;
    guide.scalar(cv::Rect(20, 20, 40, 30)) = 255;

    SuggestionSet set = suggestComponents(guide, canvas, {}, 128, true, 3, 20);
    QCOMPARE(int(set.items.size()), 2);
    QCOMPARE(set.idAt({30, 30}), 1);
    QCOMPARE(set.idAt({122, 72}), 2);
    QCOMPARE(set.idAt({80, 50}), 0);
    // A larger minimum keeps the big blob only.
    SuggestionSet big = suggestComponents(guide, canvas, {}, 128, true, 3, 100);
    QCOMPARE(int(big.items.size()), 1);
    QVERIFY(std::abs(big.items.front().area - 40 * 30) <= 2 * (40 + 30));
    // Pixels already labeled ink or background are left out, so a
    // suggestion over background can never turn it into ink.
    {
        TileRaster labels(canvas.size);
        History history;
        paintRect(labels, history, cv::Rect(20, 20, 20, 30), paintOp(PaintClass::Background, false, false));
        const SuggestionSet trimmed = suggestComponents(guide, canvas, {}, 128, true, 3, 20, &labels);
        QCOMPARE(trimmed.idAt({25, 30}), 0);
        const int id = trimmed.idAt({50, 30});
        QVERIFY(id > 0);
        MaskResult mask = trimmed.maskFor({id});
        QCOMPARE(int(mask.mask(30 - mask.origin.y, 25 - mask.origin.x)), 0);
        // And dropSupervised clears a mask made before the labels changed.
        MaskResult stale = set.maskFor({1});
        QVERIFY(stale.mask(30 - stale.origin.y, 25 - stale.origin.x) != 0);
        dropSupervised(labels, stale);
        QCOMPARE(int(stale.mask(30 - stale.origin.y, 25 - stale.origin.x)), 0);
    }

    // Through the controller: pick one, accept as one ink entry, and the
    // sidecar records it.
    QTemporaryDir dir;
    Labeler l;
    QVERIFY(l.labels->openForTesting(canvas, toPath(dir), "seg").ok);
    l.labels->setPredictionGuideForTesting(guide);
    InkLabelController::AssistSettings a;
    a.threshold = 128;
    a.median = 3;
    a.minArea = 20;
    l.labels->setAssistSettings(a);
    QVERIFY(l.labels->suggest(cv::Rect(0, 0, 200, 120)));
    QTRY_VERIFY(!l.labels->assistRunning());
    QVERIFY(l.labels->suggestions());
    QCOMPARE(int(l.labels->suggestions()->items.size()), 2);
    QVERIFY(!l.labels->overlay()->suggestionImageForTesting().isNull());
    // Enter before picking keeps the suggestions and says how to pick.
    QCOMPARE(l.labels->acceptSuggestions(), uint64_t(0));
    QVERIFY(l.labels->suggestions());
    QVERIFY(l.messages.last().contains(QStringLiteral("Nothing picked")));
    // While suggestions show, a press picks instead of painting.
    l.labels->setLabelModeEnabled(true);
    QVERIFY(l.labels->pressForTesting({30, 30}));
    QCOMPARE(l.labels->selectedSuggestions(), std::set<int>{1});
    QCOMPARE(int(l.labels->history()->undoDepth()), 0);
    const uint64_t changed = l.labels->acceptSuggestions();
    QCOMPARE(changed, uint64_t(set.items.front().area));
    QCOMPARE(int(l.labels->history()->undoDepth()), 1);
    QVERIFY(!l.labels->suggestions());
    QVERIFY(l.labels->overlay()->suggestionImageForTesting().isNull());
    QCOMPARE(LabelCounts::from(l.labels->session()->counts).ink, changed);
    QVERIFY(l.labels->save().result().ok);
    QString error;
    const auto sidecar = readSidecar(l.labels->session()->paths.sidecar(), &error);
    QVERIFY(sidecar);
    QCOMPARE(sidecar->provenance.size(), qsizetype(1));
    const QJsonObject entry = sidecar->provenance.first().toObject();
    QCOMPARE(entry.value("kind").toString(), QStringLiteral("suggestion"));
    QCOMPARE(entry.value("source").toString(), QStringLiteral("fixture"));
    QCOMPARE(entry.value("threshold").toInt(), 128);
    QCOMPARE(entry.value("median").toInt(), 3);
    QCOMPARE(entry.value("min_area").toInt(), 20);
    QCOMPARE(uint64_t(entry.value("pixels").toDouble()), changed);
    QCOMPARE(sidecar->edits.value("user").toObject().value("suggestion").toInt(), 1);

    // Again: the big blob is ink now, so only the small one is suggested.
    // Pick all, then paint background over part of it before accepting (as
    // an agent could): accepting keeps that background.
    QVERIFY(l.labels->suggest(cv::Rect(0, 0, 200, 120)));
    QTRY_VERIFY(!l.labels->assistRunning());
    QCOMPARE(int(l.labels->suggestions()->items.size()), 1);
    l.labels->pickAllSuggestions();
    QCOMPARE(l.labels->selectedSuggestions().size(), std::size_t(1));
    l.labels->setPaintClass(PaintClass::Background);
    QVERIFY(l.labels->applyShape(polygon({{118, 68}, {124, 68}, {124, 80}, {118, 80}}), Actor::User) > 0);
    QVERIFY(l.labels->acceptSuggestions() > 0);
    QCOMPARE(int(l.labels->raster()->at(121, 72)), int(kSupervision));
    QCOMPARE(int(l.labels->raster()->at(125, 73)), int(kInk | kSupervision));
}

void TestInkLabels::cleanupsMatchFixtures()
{
    const cv::Size size(80, 60);
    const auto build = [&](const std::function<void(TileRaster&, History&)>& paint) {
        TileRaster raster(size);
        History history;
        paint(raster, history);
        return raster;
    };
    const PaintOp ink = paintOp(PaintClass::Ink, false, false);
    const PaintOp background = paintOp(PaintClass::Background, false, false);

    // Remove specks: a 2 x 2 speck goes, a 10 x 10 square stays; the speck
    // stays supervised (background).
    {
        TileRaster r = build([&](TileRaster& t, History& h) {
            paintRect(t, h, cv::Rect(10, 10, 10, 10), ink);
            paintRect(t, h, cv::Rect(40, 40, 2, 2), ink);
        });
        const CleanupResult c = cleanup(r, {}, Cleanup::RemoveSpecks, 10);
        QCOMPARE(c.count(), uint64_t(4));
        QVERIFY(c.mask(40, 40) && c.mask(41, 41) && !c.mask(15, 15));
        QCOMPARE(c.op, (PaintOp{0, kInk}));
        QCOMPARE(int(c.op.apply(kInk | kSupervision)), int(kSupervision));
    }
    // Fill holes: a 2 x 2 hole fills; the outside (touching the bound) does not.
    {
        TileRaster r = build([&](TileRaster& t, History& h) {
            paintRect(t, h, cv::Rect(10, 10, 20, 20), ink);
            paintRect(t, h, cv::Rect(19, 19, 2, 2), paintOp(PaintClass::Ink, true, false));
        });
        const CleanupResult c = cleanup(r, {}, Cleanup::FillHoles, 10);
        QCOMPARE(c.count(), uint64_t(4));
        QVERIFY(c.mask(19, 19) && c.mask(20, 20) && !c.mask(0, 0));
        QCOMPARE(c.op, (PaintOp{uint8_t(kInk | kSupervision), 0}));
    }
    // Grow and shrink by 1 px against a brute force with the 3 x 3 ellipse
    // (a cross).
    {
        TileRaster r = build([&](TileRaster& t, History& h) { paintRect(t, h, cv::Rect(20, 20, 20, 15), ink); });
        const cv::Mat_<uint8_t> v = fullImage(r);
        const auto isInk = [&](int x, int y) { return x >= 0 && y >= 0 && x < size.width && y < size.height && (v(y, x) & kInk); };
        const CleanupResult grow = cleanup(r, {}, Cleanup::Grow, 1);
        const CleanupResult shrink = cleanup(r, {}, Cleanup::Shrink, 1);
        for (int y = 0; y < size.height; ++y)
            for (int x = 0; x < size.width; ++x) {
                const bool cross = isInk(x - 1, y) || isInk(x + 1, y) || isInk(x, y - 1) || isInk(x, y + 1);
                QCOMPARE(grow.mask(y, x) != 0, !isInk(x, y) && cross);
                const bool inside = isInk(x - 1, y) && isInk(x + 1, y) && isInk(x, y - 1) && isInk(x, y + 1);
                QCOMPARE(shrink.mask(y, x) != 0, isInk(x, y) && !inside);
            }
        QCOMPARE(shrink.op, (PaintOp{0, kInk}));
    }
    // Background ring: only unlabeled pixels next to ink become background.
    {
        TileRaster r = build([&](TileRaster& t, History& h) {
            paintRect(t, h, cv::Rect(20, 20, 10, 10), ink);
            paintRect(t, h, cv::Rect(30, 20, 3, 10), paintOp(PaintClass::Validation, false, false));
            paintRect(t, h, cv::Rect(17, 20, 3, 10), background);
        });
        const cv::Mat_<uint8_t> v = fullImage(r);
        const CleanupResult c = cleanup(r, {}, Cleanup::BackgroundRing, 2);
        QCOMPARE(c.op, (PaintOp{kSupervision, 0}));
        for (int y = 0; y < size.height; ++y)
            for (int x = 0; x < size.width; ++x)
                if (c.mask(y, x))
                    QCOMPARE(int(v(y, x) & (kInk | kSupervision)), 0);
        QVERIFY(c.mask(25, 31));   // right of the ink: validation only, so unlabeled
        QVERIFY(!c.mask(25, 18));  // left: already background
        QVERIFY(c.mask(18, 25));   // above the ink
        QVERIFY(!c.mask(25, 25));  // the ink itself
    }
}

void TestInkLabels::escapeDropsRunningAssist()
{
    QTemporaryDir dir;
    Labeler l;
    // A large canvas keeps the cleanup worker busy for a moment.
    const auto canvas = testCanvas({4000, 3000});
    QVERIFY(l.labels->openForTesting(canvas, toPath(dir), "seg").ok);
    QVERIFY(l.labels->applyShape(polygon({{100, 100}, {3900, 100}, {3900, 2900}, {100, 2900}}), Actor::User) > 0);
    InkLabelController::AssistSettings a;
    a.cleanupSize = 6;
    l.labels->setAssistSettings(a);
    QVERIFY(l.labels->runCleanup(Cleanup::BackgroundRing));
    QVERIFY(l.labels->assistRunning());
    // A second assist is refused while one runs.
    QVERIFY(!l.labels->runCleanup(Cleanup::Grow));
    QVERIFY(l.labels->keyForTesting(Qt::Key_Escape));
    QVERIFY(!l.labels->assistRunning());
    QFuture<void> future = l.labels->assistFutureForTesting();
    future.waitForFinished();
    QTest::qWait(50);
    QCoreApplication::processEvents();
    QCOMPARE(int(l.labels->history()->undoDepth()), 1);
    // Without Esc the same cleanup lands as one entry.
    QVERIFY(l.labels->runCleanup(Cleanup::BackgroundRing));
    QTRY_VERIFY_WITH_TIMEOUT(!l.labels->assistRunning(), 30000);
    QCOMPARE(int(l.labels->history()->undoDepth()), 2);
    QCOMPARE(l.labels->history()->top()->label, std::string("background ring"));
}

void TestInkLabels::cursorRingDefaultUnchanged()
{
    SpiralBrushCursorWidget cursor;
    cursor.setCursorState(QPointF(10, 10), 20, true, false);
    QCOMPARE(cursor.ringColor(), QColor(255, 255, 255, 220));
    cursor.setCursorState(QPointF(10, 10), 20, true, false, QColor(50, 255, 215), QColor(255, 64, 64));
    QCOMPARE(cursor.ringColor(), QColor(255, 64, 64));

    QTemporaryDir dir;
    Labeler l;
    l.labels->setPaintClass(PaintClass::Background);
    QCOMPARE(l.labels->classColor(), QColor(64, 128, 255));
    l.labels->setErase(true);
    QCOMPARE(l.labels->classColor(), QColor(200, 200, 200));
}

void TestInkLabels::fillToolAndAssistKeys()
{
    QTemporaryDir dir;
    Labeler l;
    const auto canvas = vc::RenderCanvas::make({100, 80}, {1.f, 1.f}, 1.0);
    QVERIFY(l.labels->openForTesting(canvas, toPath(dir), "seg").ok);
    l.labels->setLabelModeEnabled(true);
    // L cycles brush, polygon, fill.
    l.labels->setTool(InkLabelController::Tool::Fill);
    QCOMPARE(l.labels->toolName(), QStringLiteral("fill"));
    // No prediction: the prediction fill refuses with a message.
    QVERIFY(l.labels->pressForTesting({10, 10}));
    QVERIFY(l.messages.join('\n').contains(QStringLiteral("single-channel ink prediction")));
    QCOMPARE(int(l.labels->history()->undoDepth()), 0);
    // With one, a click fills the region as one entry with the class op.
    l.labels->setPredictionGuideForTesting(guideWith(canvas.size, {cv::Rect(5, 5, 20, 10), cv::Rect(50, 40, 20, 20)}));
    l.labels->setPaintClass(PaintClass::Background);
    QVERIFY(l.labels->pressForTesting({10, 10}));
    QTRY_VERIFY(!l.labels->assistRunning());
    QCOMPARE(int(l.labels->history()->undoDepth()), 1);
    QCOMPARE(LabelCounts::from(l.labels->session()->counts).background, uint64_t(200));
    QCOMPARE(l.labels->history()->top()->label, std::string("fill"));
    // Esc and Enter only matter while an assist is pending.
    QVERIFY(!l.labels->keyForTesting(Qt::Key_Escape));
    QVERIFY(l.labels->suggest(cv::Rect(0, 0, 100, 80)));
    QTRY_VERIFY(!l.labels->assistRunning());
    QVERIFY(l.labels->suggestions());
    // The filled block is background now, so only the other block is
    // suggested.
    QCOMPARE(int(l.labels->suggestions()->items.size()), 1);
    QVERIFY(l.labels->suggestions()->items.front().bbox.x >= 50);
    const uint64_t area = uint64_t(l.labels->suggestions()->items.front().area);
    const auto shortcutNamed = [&l](const QString& name) -> QShortcut* {
        for (QShortcut* shortcut : l.labels->shortcutsForTesting())
            if (shortcut->objectName() == name)
                return shortcut;
        return nullptr;
    };
    QVERIFY(shortcutNamed(QStringLiteral("ink_label_pick_all"))->isEnabled());
    l.labels->setSuggestionSelected(1, true);
    QVERIFY(l.labels->keyForTesting(Qt::Key_Return));
    QVERIFY(!l.labels->suggestions());
    QVERIFY(!shortcutNamed(QStringLiteral("ink_label_pick_all"))->isEnabled());
    QCOMPARE(int(l.labels->history()->undoDepth()), 2);
    // The accepted component (the opened region) is ink now.
    QCOMPARE(LabelCounts::from(l.labels->session()->counts).ink, area);
}

// ---------------------------------------------------------------------------
// Imports, versions and the trainer handoff
// ---------------------------------------------------------------------------

namespace {

// Imports into a fresh raster of the canvas and returns the outcome.
ImportOutcome importInto(TileRaster& raster, const fs::path& path, const vc::RenderCanvas& canvas, const ImportOptions& options)
{
    EditTxn txn(raster, "import", Actor::User);
    ImportOutcome out = importLabel(path, canvas, options, txn);
    if (out.ok)
        (void)txn.commit(1);
    return out;
}

uint64_t countBit(const TileRaster& raster, uint8_t bit)
{
    const cv::Mat_<uint8_t> v = fullImage(raster);
    uint64_t n = 0;
    for (int y = 0; y < v.rows; ++y)
        for (int x = 0; x < v.cols; ++x)
            n += (v(y, x) & bit) != 0;
    return n;
}

}  // namespace

void TestInkLabels::importPngAndRgbTiff()
{
    QTemporaryDir dir;
    const auto canvas = testCanvas({200, 140});
    // An anti-aliased PNG: every pixel above 0 counts, with a warning.
    cv::Mat_<uint8_t> aa = cv::Mat_<uint8_t>::zeros(140, 200);
    cv::circle(aa, {100, 70}, 30, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
    const fs::path png = toPath(dir) / "seg_a_inklabels.png";
    QVERIFY(cv::imwrite(png.string(), aa));
    QCOMPARE(detectFormat(png), ImportFormat::Png);
    TileRaster raster(canvas.size);
    ImportOutcome out = importInto(raster, png, canvas, {});
    QVERIFY2(out.ok, qPrintable(out.error));
    QCOMPARE(out.setPixels, uint64_t(cv::countNonZero(aa)));
    QCOMPARE(countBit(raster, kInk), out.setPixels);
    QVERIFY(out.nonBinary > 0);
    QVERIFY(out.warnings.join('\n').contains(QStringLiteral("other than 0")));

    // An RGB TIFF named as a supervision mask: the first sample (red) is used.
    cv::Mat bgr(140, 200, CV_8UC3, cv::Scalar(0, 0, 0));
    bgr(cv::Rect(10, 10, 50, 40)).setTo(cv::Scalar(0, 0, 255));  // red square
    bgr(cv::Rect(100, 80, 30, 30)).setTo(cv::Scalar(255, 0, 0)); // blue square, ignored
    const fs::path tif = toPath(dir) / "seg_a_supervision_mask.tif";
    QVERIFY(cv::imwrite(tif.string(), bgr));
    QCOMPARE(kindForFileName(tif), LabelKind::Supervision);
    ImportOptions options;
    options.kind = kindForFileName(tif);
    out = importInto(raster, tif, canvas, options);
    QVERIFY2(out.ok, qPrintable(out.error));
    QCOMPARE(out.setPixels, uint64_t(50 * 40));
    QCOMPARE(countBit(raster, kSupervision), uint64_t(50 * 40));
    QCOMPARE(countBit(raster, kInk), uint64_t(cv::countNonZero(aa)));  // other bits untouched
    QVERIFY(out.warnings.join('\n').contains(QStringLiteral("more than one sample")));

    // Through the controller: one undo entry, provenance recorded.
    Labeler l;
    QVERIFY(l.labels->openForTesting(canvas, toPath(dir), "seg_a").ok);
    int finished = 0;
    ImportOutcome last;
    QObject::connect(l.labels.get(), &InkLabelController::importFinished, l.labels.get(),
                     [&](const QString&, const ImportOutcome& outcome, const ImportOptions&) {
                         ++finished;
                         last = outcome;
                     });
    QVERIFY(l.labels->importFile(png, {}));
    QTRY_COMPARE(finished, 1);
    QCOMPARE(int(l.labels->history()->undoDepth()), 1);
    QCOMPARE(LabelCounts::from(l.labels->session()->counts).ink, uint64_t(cv::countNonZero(aa)));
    QVERIFY(l.labels->undo(Actor::User));
    QCOMPARE(LabelCounts::from(l.labels->session()->counts).ink, uint64_t(0));
    QVERIFY(l.labels->redo(Actor::User));
    QVERIFY(l.labels->save().result().ok);
    QString error;
    const auto sidecar = readSidecar(l.labels->session()->paths.sidecar(), &error);
    QVERIFY(sidecar);
    QCOMPARE(sidecar->provenance.first().toObject().value("kind").toString(), QStringLiteral("import"));
    QCOMPARE(sidecar->provenance.first().toObject().value("format").toString(), QStringLiteral("png"));
    // A wrong-size file is refused through the controller too, with nothing applied.
    const fs::path small = toPath(dir) / "small_inklabels.png";
    QVERIFY(cv::imwrite(small.string(), cv::Mat_<uint8_t>::zeros(17, 23)));
    QVERIFY(l.labels->importFile(small, {}));
    QTRY_COMPARE(finished, 2);
    QVERIFY(!last.ok);
    QVERIFY(last.error.contains(QStringLiteral("23 x 17")));
    QCOMPARE(int(l.labels->history()->undoDepth()), 1);
}

void TestInkLabels::importLabelZarr65()
{
    QTemporaryDir dir;
    const auto canvas = testCanvas({300, 200});
    const fs::path root = toPath(dir) / "seg_a_supervision_mask.zarr";
    fs::create_directories(root);
    vcr::zarr::CreateOptions create;
    create.shape = {65, 200, 300};
    create.chunks = {65, 64, 64};
    create.separator = "/";
    auto ds = std::make_unique<vcr::zarr::Writer>(vcr::zarr::Writer::create(root / "0", create));
    cv::Mat_<uint8_t> plane = cv::Mat_<uint8_t>::zeros(200, 300);
    plane(cv::Rect(40, 30, 100, 50)) = 255;
    ds->writeRegion({32, 0, 0}, {1, 200, 300}, plane.data);
    // Other planes are ignored.
    cv::Mat_<uint8_t> other(200, 300, uint8_t(255));
    ds->writeRegion({0, 0, 0}, {1, 200, 300}, other.data);
    ds.reset();
    QCOMPARE(detectFormat(root), ImportFormat::LabelZarr65);
    TileRaster raster(canvas.size);
    ImportOptions options;
    options.kind = kindForFileName(root);
    QCOMPARE(options.kind, LabelKind::Supervision);
    const ImportOutcome out = importInto(raster, root, canvas, options);
    QVERIFY2(out.ok, qPrintable(out.error));
    QCOMPARE(out.format, ImportFormat::LabelZarr65);
    QCOMPARE(countBit(raster, kSupervision), uint64_t(100 * 50));
    QCOMPARE(countBit(raster, kInk), uint64_t(0));
    // The level directory works as well as the root.
    TileRaster again(canvas.size);
    QVERIFY(importInto(again, root / "0", canvas, options).ok);
    QVERIFY(sameImage(fullImage(again), fullImage(raster)));
}

void TestInkLabels::importZarr2DVersion3()
{
    // A 2D zarr v3 array at <x>.zarr/0 with the plain bytes codec, written by
    // hand: 150 x 100 px in 64 x 64 chunks.
    QTemporaryDir dir;
    const auto canvas = testCanvas({100, 150});
    QCOMPARE(canvas.size, cv::Size(100, 140));
    const cv::Size size = canvas.size;
    const fs::path array = toPath(dir) / "seg_a_inklabels.zarr" / "0";
    fs::create_directories(array);
    {
        std::ofstream meta(array / "zarr.json");
        meta << R"({"zarr_format": 3, "node_type": "array", "shape": [)" << size.height << ", " << size.width
             << R"(], "data_type": "uint8", "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [64, 64]}},)"
             << R"( "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}}, "fill_value": 0,)"
             << R"( "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]})";
    }
    cv::Mat_<uint8_t> image = cv::Mat_<uint8_t>::zeros(size);
    image(cv::Rect(20, 30, 70, 90)) = 255;
    for (int cy = 0; cy * 64 < size.height; ++cy)
        for (int cx = 0; cx * 64 < size.width; ++cx) {
            cv::Mat_<uint8_t> chunk = cv::Mat_<uint8_t>::zeros(64, 64);
            const cv::Rect r = cv::Rect(cx * 64, cy * 64, 64, 64) & cv::Rect(cv::Point(0, 0), size);
            image(r).copyTo(chunk(cv::Rect(0, 0, r.width, r.height)));
            const fs::path file = array / "c" / std::to_string(cy) / std::to_string(cx);
            fs::create_directories(file.parent_path());
            std::ofstream out(file, std::ios::binary);
            out.write(reinterpret_cast<const char*>(chunk.data), 64 * 64);
        }
    // The root is a v3 group, as in an OME-Zarr pyramid.
    {
        std::ofstream group(array.parent_path() / "zarr.json");
        group << R"({"zarr_format": 3, "node_type": "group", "attributes": {}})";
    }
    QCOMPARE(detectFormat(array.parent_path()), ImportFormat::Zarr2D);
    QCOMPARE(detectFormat(array), ImportFormat::Zarr2D);
    TileRaster raster(size);
    const ImportOutcome out = importInto(raster, array.parent_path(), canvas, {});
    QVERIFY2(out.ok, qPrintable(out.error));
    QCOMPARE(out.format, ImportFormat::Zarr2D);
    QCOMPARE(countBit(raster, kInk), uint64_t(70 * 90));
    const cv::Mat_<uint8_t> v = fullImage(raster);
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x)
            QCOMPARE((v(y, x) & kInk) != 0, image(y, x) != 0);
}

void TestInkLabels::importResamplesHalfAndQuarter()
{
    QTemporaryDir dir;
    // A grid of 3041 x 9 at scale 1: W = 3041, the half render is 1521 wide
    // (3041 / 2 rounds up), the quarter 760.
    const auto canvas = vc::RenderCanvas::make({3041, 9}, {1.f, 1.f}, 1.0);
    QCOMPARE(canvas.size, cv::Size(3041, 9));
    for (const double rs : {0.5, 0.25}) {
        const cv::Size source = vc::renderCanvasSize(canvas.grid, canvas.gridScale, rs);
        QCOMPARE(source.width, rs == 0.5 ? 1521 : 760);
        cv::Mat_<uint8_t> image(source);
        for (int y = 0; y < source.height; ++y)
            for (int x = 0; x < source.width; ++x)
                image(y, x) = ((x * 7 + y * 3) % 5 == 0) ? 255 : 0;
        const fs::path tif = toPath(dir) / ("half_inklabels_" + std::to_string(int(rs * 100)) + ".tif");
        QVERIFY(cv::imwrite(tif.string(), image));

        TileRaster raster(canvas.size);
        ImportOutcome out = importInto(raster, tif, canvas, {});
        QVERIFY(!out.ok);
        QVERIFY(out.needsDecision);
        QCOMPARE(out.resampledFrom, rs);
        QCOMPARE(countBit(raster, kInk), uint64_t(0));

        ImportOptions options;
        options.allowResample = true;
        out = importInto(raster, tif, canvas, options);
        QVERIFY2(out.ok, qPrintable(out.error));
        QCOMPARE(out.resampledFrom, rs);
        const cv::Mat_<uint8_t> v = fullImage(raster);
        for (int y = 0; y < canvas.size.height; ++y)
            for (int x = 0; x < canvas.size.width; ++x) {
                const int sx = std::clamp(int(std::floor(source.width / 2.0 + rs * (x + 0.5 - canvas.size.width / 2.0))), 0, source.width - 1);
                const int sy = std::clamp(int(std::floor(source.height / 2.0 + rs * (y + 0.5 - canvas.size.height / 2.0))), 0, source.height - 1);
                if (((v(y, x) & kInk) != 0) != (image(sy, sx) != 0))
                    QFAIL(qPrintable(QStringLiteral("rs %1 pixel (%2, %3)").arg(rs).arg(x).arg(y)));
            }
    }
    // A rotated render and an unrelated size are refused, saying why.
    TileRaster raster(canvas.size);
    const fs::path rotated = toPath(dir) / "rot_inklabels.tif";
    QVERIFY(cv::imwrite(rotated.string(), cv::Mat_<uint8_t>::zeros(3041, 9)));
    ImportOutcome out = importInto(raster, rotated, canvas, {});
    QVERIFY(!out.ok && !out.needsDecision);
    QVERIFY(out.error.contains(QStringLiteral("rotated render")));
    const fs::path odd = toPath(dir) / "odd_inklabels.tif";
    QVERIFY(cv::imwrite(odd.string(), cv::Mat_<uint8_t>::zeros(10, 3000)));
    out = importInto(raster, odd, canvas, {});
    QVERIFY(!out.ok && !out.needsDecision);
    QVERIFY(out.error.contains(QStringLiteral("tifxyz_label_transfer")));
}

void TestInkLabels::importLegacyPngPair()
{
    QTemporaryDir dir;
    const auto canvas = testCanvas({200, 140});
    cv::Mat_<uint8_t> ink = cv::Mat_<uint8_t>::zeros(140, 200);
    ink(cv::Rect(50, 50, 20, 20)) = 255;
    cv::Mat_<uint8_t> mask = cv::Mat_<uint8_t>::zeros(140, 200);
    mask(cv::Rect(10, 10, 180, 120)) = 255;
    QVERIFY(cv::imwrite((toPath(dir) / "seg_a_inklabels.png").string(), ink));
    QVERIFY(cv::imwrite((toPath(dir) / "seg_a_mask.png").string(), mask));
    const fs::path path = toPath(dir) / "seg_a_inklabels.png";
    QCOMPARE(detectFormat(path), ImportFormat::LegacyPngPair);

    TileRaster raster(canvas.size);
    ImportOutcome out = importInto(raster, path, canvas, {});
    QVERIFY(!out.ok && out.needsDecision);
    QVERIFY(out.error.contains(QStringLiteral("seg_a_mask.png")));
    ImportOptions options;
    options.legacyMaskAs = LegacyMaskAs::Supervision;
    out = importInto(raster, path, canvas, options);
    QVERIFY2(out.ok, qPrintable(out.error));
    QCOMPARE(countBit(raster, kInk), uint64_t(400));
    QCOMPARE(countBit(raster, kSupervision), uint64_t(180 * 120));

    TileRaster inkOnly(canvas.size);
    options.legacyMaskAs = LegacyMaskAs::Ignore;
    QVERIFY(importInto(inkOnly, path, canvas, options).ok);
    QCOMPARE(countBit(inkOnly, kInk), uint64_t(400));
    QCOMPARE(countBit(inkOnly, kSupervision), uint64_t(0));
}

void TestInkLabels::numberedVersionsWriteAllKinds()
{
    QTemporaryDir dir;
    const auto canvas = testCanvas({400, 300});
    StorePaths paths;
    {
        Labeler l;
        QVERIFY(l.labels->openForTesting(canvas, toPath(dir), "seg").ok);
        paths = l.labels->session()->paths;
        QCOMPARE(paths.version, 1);
        QVERIFY(l.labels->applyShape(disc(100, 100, 30), Actor::User) > 0);
        QVERIFY(l.labels->save().result().ok);
        QTRY_VERIFY(!l.labels->session()->dirty);
        QVERIFY(!fs::exists(paths.file(LabelKind::Validation)));

        QVERIFY(l.labels->applyShape(disc(300, 200, 30), Actor::User) > 0);
        const SaveOutcome v2 = l.labels->saveAsNewVersion().result();
        QVERIFY2(v2.ok, qPrintable(v2.error));
        QTRY_COMPARE(l.labels->session()->paths.version, 2);
        QTRY_VERIFY(!l.labels->session()->dirty);
        StorePaths p2 = paths;
        p2.version = 2;
        for (const LabelKind kind : kAllKinds)
            QVERIFY2(fs::exists(p2.file(kind)), p2.file(kind).string().c_str());
        QCOMPARE(p2.file(LabelKind::Validation).filename().string(), std::string("seg_validation_mask_v2.tif"));
        QCOMPARE(cv::countNonZero(readGray(p2.file(LabelKind::Validation))), 0);
        QVERIFY(fs::exists(p2.sidecar()));
        QCOMPARE(listVersions(paths.dir, "seg"), (std::vector<int>{1, 2}));
        QCOMPARE(l.labels->versions(), (std::vector<int>{1, 2}));
        // v1 still holds the first state.
        QCOMPARE(cv::countNonZero(readGray(paths.file(LabelKind::Ink))) < cv::countNonZero(readGray(p2.file(LabelKind::Ink))), true);
        l.labels->closeSession();
        settleWrites(*l.labels, paths);
    }
    Labeler l;
    // The newest version opens by default; others on request.
    QVERIFY(l.labels->openForTesting(canvas, toPath(dir), "seg").ok);
    QCOMPARE(l.labels->session()->paths.version, 2);
    const auto twoDiscs = LabelCounts::from(l.labels->session()->counts).ink;
    InkLabelController::OpenOptions one;
    one.version = 1;
    l.labels->closeSession();
    QVERIFY(l.labels->openForTesting(canvas, toPath(dir), "seg", one).ok);
    QCOMPARE(l.labels->session()->paths.version, 1);
    QVERIFY(LabelCounts::from(l.labels->session()->counts).ink < twoDiscs);
    InkLabelController::OpenOptions missing;
    missing.version = 5;
    l.labels->closeSession();
    const auto refused = l.labels->openForTesting(canvas, toPath(dir), "seg", missing);
    QVERIFY(!refused.ok);
    QCOMPARE(refused.error, InkLabelController::OpenError::NoVersion);
    QVERIFY(refused.message.contains(QStringLiteral("1, 2")));
}

void TestInkLabels::datasetEntryTargetsTheSegment()
{
    const StorePaths paths = storePathsFor("/data/proj", "seg_a");
    QJsonObject entry = datasetEntry(paths, "/data/proj", {1});
    QCOMPARE(entry.value("segments_path").toString(), QStringLiteral("/data/proj/ink_labels"));
    QCOMPARE(entry.value("segment_names").toArray(), QJsonArray{QStringLiteral("seg_a")});
    QCOMPARE(entry.value("surface_volume_paths").toObject().value("seg_a").toString(),
             QStringLiteral("/data/proj/ink_labels/seg_a/seg_a.zarr"));
    QVERIFY(!entry.contains("label_version"));  // one version: the trainer's default
    // With several versions, the saved one is named, also v1 (unnumbered),
    // since the trainer otherwise takes the newest.
    QCOMPARE(datasetEntry(paths, "/data/proj", {1, 2}).value("label_version").toString(), QStringLiteral("v1"));
    StorePaths v3 = paths;
    v3.version = 3;
    QCOMPARE(datasetEntry(v3, "/data/proj", {1, 3}).value("label_version").toString(), QStringLiteral("v3"));
    // Built as JSON, so quotes and backslashes in paths stay valid.
    const StorePaths odd = storePathsFor("C:\\data\\my \"proj\"", "seg_a");
    const QByteArray text = QJsonDocument(datasetEntry(odd, "C:\\data\\my \"proj\"", {1})).toJson();
    QJsonParseError error;
    QVERIFY(QJsonDocument::fromJson(text, &error).isObject());
    QCOMPARE(error.error, QJsonParseError::NoError);

    // Version names: _v1, _v0, junk and numbers too large for an int are
    // skipped instead of crashing the open.
    QTemporaryDir dir;
    const fs::path store = toPath(dir) / "ink_labels" / "seg";
    fs::create_directories(store);
    for (const char* name : {"seg_inklabels.tif", "seg_inklabels_v2.tif", "seg_supervision_mask_v7.tif",
                             "seg_inklabels_v1.tif", "seg_inklabels_v0.tif", "seg_inklabels_v99999999999.tif",
                             "seg_inklabels_vx.tif", "seg_inklabels_v3.tif.prev"})
        std::ofstream(store / name) << "x";
    QCOMPARE(listVersions(store, "seg"), (std::vector<int>{1, 2, 7}));
}

// ---------------------------------------------------------------------------
// Agent bridge support
// ---------------------------------------------------------------------------

void TestInkLabels::agentEditsReadBackAndUndo()
{
    QTemporaryDir dir;
    Labeler l;
    QVERIFY(l.labels->openForTesting(testCanvas({300, 200}), toPath(dir), "seg").ok);
    // A user stroke, then an agent stroke through three points as one entry.
    QVERIFY(l.labels->applyShape(disc(40, 40, 10), Actor::User) > 0);
    const uint64_t userInk = LabelCounts::from(l.labels->session()->counts).ink;
    const std::vector<Shape> stroke{capsule({100, 100}, {150, 100}, 4), capsule({150, 100}, {150, 150}, 4)};
    const uint64_t changed = l.labels->applyShapes(stroke, paintOp(PaintClass::Ink, false, false), Actor::Agent, "agent ink stroke");
    QVERIFY(changed > 0);
    QCOMPARE(int(l.labels->history()->undoDepth()), 2);
    QCOMPARE(l.labels->history()->top()->actor, Actor::Agent);

    // Read back: values and one bit, clipped to the canvas, with counts.
    const fs::path values = toPath(dir) / "values.png";
    auto out = l.labels->readRect(cv::Rect(90, 90, 80, 80), InkLabelController::ReadFormat::Values, values);
    QVERIFY2(out.ok, qPrintable(out.error));
    QCOMPARE(out.counts.ink, changed);
    cv::Mat_<uint8_t> expected;
    l.labels->raster()->readRect(cv::Rect(90, 90, 80, 80), expected);
    QVERIFY(sameImage(readGray(values), expected));
    const fs::path ink = toPath(dir) / "ink.tif";
    out = l.labels->readRect(cv::Rect(280, 180, 50, 50), InkLabelController::ReadFormat::Ink, ink);
    QVERIFY(out.ok);
    QCOMPARE(out.rect, cv::Rect(280, 180, 20, 20));
    QCOMPARE(readGray(ink).size(), cv::Size(20, 20));
    QVERIFY(!l.labels->readRect(cv::Rect(400, 400, 10, 10), InkLabelController::ReadFormat::Ink, ink).ok);
    QVERIFY(!l.labels->readRect(cv::Rect(0, 0, 5000, 10), InkLabelController::ReadFormat::Ink, ink).ok);

    // An agent undo takes back its own entry, then stops at the user's
    // unless forced.
    QVERIFY(l.labels->undo(Actor::Agent));
    QCOMPARE(int(l.labels->history()->undoDepth()), 1);
    QVERIFY(!l.labels->undo(Actor::Agent));
    QCOMPARE(int(l.labels->history()->undoDepth()), 1);
    QVERIFY(l.labels->undo(Actor::Agent, true));
    QCOMPARE(int(l.labels->history()->undoDepth()), 0);
    QVERIFY(l.labels->redo(Actor::Agent));
    QVERIFY(l.labels->redo(Actor::Agent));
    QCOMPARE(LabelCounts::from(l.labels->session()->counts).ink, userInk + changed);

    // A trainer save is visible while it runs; a checkpoint is not one.
    QVERIFY(!l.labels->trainerSaveInFlight());
    QFuture<SaveOutcome> save = l.labels->save();
    QVERIFY(l.labels->trainerSaveInFlight() || save.isFinished());
    save.waitForFinished();
    QTRY_VERIFY(!l.labels->trainerSaveInFlight());
}

void TestInkLabels::agentWritesBackValues()
{
    // What ink_labels.read writes as "values" can be painted back exactly,
    // as one undo entry, whatever was there before.
    QTemporaryDir dir;
    Labeler l;
    QVERIFY(l.labels->openForTesting(testCanvas({300, 200}), toPath(dir), "seg").ok);
    QVERIFY(l.labels->applyShape(disc(60, 60, 30), Actor::User) > 0);
    cv::Mat_<uint8_t> values(40, 50, uint8_t(0));
    values(cv::Rect(0, 0, 25, 40)) = uint8_t(kInk | kSupervision);
    values(cv::Rect(25, 0, 25, 20)) = uint8_t(kSupervision);
    values(cv::Rect(10, 30, 30, 10)) = uint8_t(kValidation);
    const uint64_t changed = l.labels->applyValues({40, 40}, values, Actor::Agent, "agent values");
    QVERIFY(changed > 0);
    QCOMPARE(int(l.labels->history()->undoDepth()), 2);
    QCOMPARE(l.labels->history()->top()->actor, Actor::Agent);
    cv::Mat_<uint8_t> back;
    l.labels->raster()->readRect(cv::Rect(40, 40, 50, 40), back);
    QVERIFY(sameImage(back, values));
    // Outside the rectangle nothing changed.
    QCOMPARE(int(l.labels->raster()->at(35, 60)), int(kInk | kSupervision));
    QVERIFY(l.labels->undo(Actor::Agent));
    l.labels->raster()->readRect(cv::Rect(40, 40, 50, 40), back);
    QCOMPARE(int(back(10, 10)), int(kInk | kSupervision));  // the user's disc again
}

void TestInkLabels::agentAssistRunsNowAndReports()
{
    QTemporaryDir dir;
    Labeler l;
    const auto canvas = vc::RenderCanvas::make({120, 80}, {1.f, 1.f}, 1.0);
    QVERIFY(l.labels->openForTesting(canvas, toPath(dir), "seg").ok);
    // No prediction: the refusal comes back as the report's error.
    auto report = l.labels->runAssistNow([&] { return l.labels->suggest(cv::Rect(0, 0, 120, 80)); });
    QVERIFY(report.error.contains(QStringLiteral("single-channel ink prediction")));

    l.labels->setPredictionGuideForTesting(guideWith(canvas.size, {cv::Rect(10, 10, 30, 20), cv::Rect(70, 40, 30, 30)}));
    // A fill with an explicit op lands at once, recorded as the agent's.
    report = l.labels->runAssistNow(
        [&] { return l.labels->fillAt({15, 15}, cv::Rect(0, 0, 120, 80), paintOp(PaintClass::Background, false, false)); });
    QVERIFY2(report.error.isEmpty(), qPrintable(report.error));
    QCOMPARE(report.changedPixels, uint64_t(30 * 20));
    QVERIFY(!report.reachedBound);
    QVERIFY(!l.labels->assistRunning());
    QCOMPARE(l.labels->history()->top()->actor, Actor::Agent);
    QCOMPARE(LabelCounts::from(l.labels->session()->counts).background, uint64_t(30 * 20));
    // A fill cut by its bound says so.
    report = l.labels->runAssistNow(
        [&] { return l.labels->fillAt({75, 45}, cv::Rect(60, 30, 20, 50), paintOp(PaintClass::Validation, false, false)); });
    QVERIFY(report.reachedBound);

    // Suggestions are there at once; accepting them as the agent records it.
    report = l.labels->runAssistNow([&] { return l.labels->suggest(cv::Rect(0, 0, 120, 80)); });
    QVERIFY(report.error.isEmpty());
    QVERIFY(l.labels->suggestions());
    QCOMPARE(int(l.labels->suggestions()->items.size()), 1);  // the background block is left out
    l.labels->pickAllSuggestions();
    QVERIFY(l.labels->acceptSuggestions(Actor::Agent) > 0);
    QCOMPARE(l.labels->history()->top()->actor, Actor::Agent);

    // A cleanup too.
    report = l.labels->runAssistNow([&] { return l.labels->runCleanup(Cleanup::Shrink, cv::Rect(0, 0, 120, 80)); });
    QVERIFY(report.error.isEmpty());
    QVERIFY(report.changedPixels > 0);
}

void TestInkLabels::shapeCanvasOverlap()
{
    const cv::Size size(100, 60);
    const auto polygon = [](std::vector<cv::Point2d> pts) { return Shape{Shape::Kind::Polygon, std::move(pts), 0.0}; };
    // Polygons have no radius: a zero-width extent must still count.
    QCOMPARE(canvasOverlap({polygon({{10, 10}, {40, 10}, {40, 30}})}, size), CanvasOverlap::Inside);
    QCOMPARE(canvasOverlap({polygon({{90, 10}, {140, 10}, {140, 30}})}, size), CanvasOverlap::Clipped);
    QCOMPARE(canvasOverlap({polygon({{-50, 10}, {-10, 10}, {-10, 30}})}, size), CanvasOverlap::Outside);
    // Discs and capsules reach out by their stamp radius.
    QCOMPARE(canvasOverlap({disc(-2, 30, 3)}, size), CanvasOverlap::Clipped);
    QCOMPARE(canvasOverlap({disc(-5, 30, 3)}, size), CanvasOverlap::Outside);
    QCOMPARE(canvasOverlap({disc(50, 30, 3)}, size), CanvasOverlap::Inside);
    QCOMPARE(canvasOverlap({capsule({-20, 30}, {120, 30}, 2)}, size), CanvasOverlap::Clipped);
    // Several shapes are judged together; no shape covers nothing.
    QCOMPARE(canvasOverlap({disc(-50, 30, 3), disc(50, 30, 3)}, size), CanvasOverlap::Clipped);
    QCOMPARE(canvasOverlap({}, size), CanvasOverlap::Outside);
}

QTEST_MAIN(TestInkLabels)
#include "test_ink_labels.moc"
