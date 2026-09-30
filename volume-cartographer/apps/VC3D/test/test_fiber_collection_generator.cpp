// Command line, progress events, zone and blocks of the Automated Fiber Volume
// generator. Process-free: no Python is started.

#include "FiberCollectionGenerator.hpp"

#include <QDir>
#include <QFile>
#include <QTemporaryDir>
#include <QtTest/QtTest>

using namespace vc3d::fibergen;

class FiberCollectionGeneratorTest : public QObject
{
    Q_OBJECT

private slots:
    void argumentsDescribeTheWholeRequest()
    {
        Request request;
        request.volume = "https://example.org/scroll.zarr";
        request.level = 1;
        request.origin = {10, 20, 30};
        request.size = {512, 256, 128};
        request.output = "/data/fibers.afv";
        request.coordinateSpace = "PHerc0813/20250821151723";
        request.nativeScale = 2;
        request.voxelSizeUm = 9.362;
        request.sourcePath = "s3://bucket/scroll.zarr/";
        request.thresholdPercent = 55;
        request.mirror = false;
        request.blockSize = 256;
        request.previewDirectory = "/tmp/previews";
        const QStringList expected{
            "-m", "vesuvius.afv_spline_generator", "--volume", "https://example.org/scroll.zarr", "--level", "1",
            "--origin", "10", "20", "30", "--size", "512", "256", "128", "--output", "/data/fibers.afv",
            "--coordinate-space", "PHerc0813/20250821151723", "--native-scale", "2", "--voxel-size", "9.362",
            "--source-path", "s3://bucket/scroll.zarr/", "--model", "Qualzz20/afv_fiber_9um", "--threshold", "55",
            "--no-mirror", "--block-size", "256", "--preview-dir", "/tmp/previews", "--progress", "json"};
        QCOMPARE(arguments(request), expected);
    }

    void unknownValuesAreLeftToTheGenerator()
    {
        Request request;
        request.volume = "/data/scroll.zarr";
        request.output = "/data/fibers.afv";
        request.coordinateSpace = "space";
        const auto args = arguments(request);
        QVERIFY(!args.contains("--voxel-size"));
        QVERIFY(!args.contains("--source-path"));
        QVERIFY(!args.contains("--no-mirror"));
        QVERIFY(!args.contains("--preview-dir"));
        QCOMPARE(args.mid(args.indexOf("--threshold"), 2), (QStringList{"--threshold", "60"}));
        QCOMPARE(args.mid(args.indexOf("--block-size"), 2), (QStringList{"--block-size", "512"}));
    }

    void parsesProgressEvents()
    {
        const auto progress = parseEvent(R"({"event": "progress", "stage": "predict", "message": "Predicting", "fraction": 0.25})");
        QVERIFY(progress);
        QCOMPARE(progress->kind, Event::Kind::Progress);
        QCOMPARE(progress->message, QString("Predicting"));
        QCOMPARE(progress->fraction, 0.25);
        QCOMPARE(parseEvent(R"({"event": "progress", "message": "", "fraction": 7})")->fraction, 1.0);

        const auto done = parseEvent(R"({"event": "done", "fraction": 1.0, "output": "/data/fibers.afv", "fibers": 4783, "uuid": "u"})" "\n");
        QVERIFY(done);
        QCOMPARE(done->kind, Event::Kind::Done);
        QCOMPARE(done->output, QString("/data/fibers.afv"));
        QCOMPARE(done->fibers, qint64(4783));

        QCOMPARE(parseEvent(R"({"event": "error", "message": "The volume has no array 1"})")->message, QString("The volume has no array 1"));
        QCOMPARE(parseEvent(R"({"event": "warning", "message": "slow"})")->kind, Event::Kind::Warning);
    }

    void parsesBlockEvents()
    {
        const auto plan = parseEvent(
            R"({"event": "plan", "blocks": [{"origin": [0, 0, 0], "size": [512, 512, 512]}, {"origin": [512, 0, 0], "size": [88, 512, 512]}], "block_size": 512})");
        QVERIFY(plan);
        QCOMPARE(plan->kind, Event::Kind::Plan);
        QCOMPARE(plan->blocks.size(), size_t(2));
        QVERIFY((plan->blocks[1] == Block{{512, 0, 0}, {88, 512, 512}}));
        QVERIFY(!parseEvent(R"({"event": "plan", "blocks": [{"origin": [0, 0], "size": [1, 1, 1]}]})"));

        const auto block = parseEvent(R"({"event": "block", "index": 1, "state": "predicting"})");
        QVERIFY(block);
        QCOMPARE(block->kind, Event::Kind::Block);
        QCOMPARE(block->index, 1);
        QCOMPARE(block->state, QString("predicting"));
        QVERIFY(!parseEvent(R"({"event": "block", "state": "done"})"));

        const auto preview = parseEvent(R"({"event": "preview", "path": "/tmp/previews/preview-0001.afv", "fibers": 12})");
        QVERIFY(preview);
        QCOMPARE(preview->kind, Event::Kind::Preview);
        QCOMPARE(preview->path, QString("/tmp/previews/preview-0001.afv"));
        QVERIFY(!parseEvent(R"({"event": "preview"})"));
    }

    void zoneIsCutIntoBlocks()
    {
        const auto blocks = zoneBlocks({100, 200, 300}, {600, 512, 512}, 512);
        QCOMPARE(blocks.size(), size_t(2));
        QVERIFY((blocks[0] == Block{{100, 200, 300}, {512, 512, 512}}));
        QVERIFY((blocks[1] == Block{{612, 200, 300}, {88, 512, 512}}));
        const auto grid = zoneBlocks({0, 0, 0}, {1024, 1024, 512}, 512);
        QCOMPARE(grid.size(), size_t(4));
        QVERIFY((grid[1] == Block{{512, 0, 0}, {512, 512, 512}}));
        QVERIFY((grid[2] == Block{{0, 512, 0}, {512, 512, 512}}));
        QVERIFY(zoneBlocks({0, 0, 0}, {0, 10, 10}, 512).empty());
    }

    void ignoresOtherOutput()
    {
        QVERIFY(!parseEvent("Loading checkpoint: /models/checkpoint_final.pth"));
        QVERIFY(!parseEvent("[1, 2]"));
        QVERIFY(!parseEvent(R"({"event": "other"})"));
        QVERIFY(!parseEvent(R"({"event": "done", "fibers": 3})"));
    }

    void zoneStaysInsideTheVolume()
    {
        using Zone = std::array<int, 3>;
        QCOMPARE(zoneOrigin({500, 500, 500}, {512, 512, 512}, {2000, 1500, 1000}), (Zone{244, 244, 244}));
        QCOMPARE(zoneOrigin({10, 1490, 999}, {512, 512, 512}, {2000, 1500, 1000}), (Zone{0, 988, 488}));
        QCOMPARE(zoneOrigin({150, 900, 900}, {512, 64, 64}, {300, 2000, 2000}), (Zone{0, 868, 868}));
    }

    void sourceTreeGoesFirstOnPythonPath()
    {
        QTemporaryDir root;
        QVERIFY(root.isValid());
        const QString source = root.filePath("villa/vesuvius/src");
        QVERIFY(QDir().mkpath(source + "/vesuvius/afv_spline_generator"));
        QFile module(source + "/vesuvius/afv_spline_generator/__main__.py");
        QVERIFY(module.open(QIODevice::WriteOnly));
        module.close();
        const QString app = root.filePath("villa/volume-cartographer/build/bin");
        QVERIFY(QDir().mkpath(app));
        QCOMPARE(vesuviusSourceDirectory(app), QDir::cleanPath(source));
        QVERIFY(vesuviusSourceDirectory(root.path()).isEmpty());

        QProcessEnvironment inherited;
        inherited.insert("PYTHONPATH", "/other");
        QCOMPARE(environment(inherited, source).value("PYTHONPATH"), source + QDir::listSeparator() + "/other");
        QCOMPARE(environment(QProcessEnvironment(), source).value("PYTHONPATH"), source);
        QCOMPARE(environment(inherited, QString()).value("PYTHONPATH"), QString("/other"));
    }
};

QTEST_APPLESS_MAIN(FiberCollectionGeneratorTest)
#include "test_fiber_collection_generator.moc"
