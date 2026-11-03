#include <csignal>

#include <QApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QScopeGuard>
#include <QTemporaryFile>
#include <QThread>
#include <QtTest/QtTest>

#include "CommandLineToolRunner.hpp"

Q_DECLARE_METATYPE(CommandLineToolRunner::Tool)

namespace {

constexpr auto kChildArgument = "--command-runner-child";

QString helperProgram()
{
    return QCoreApplication::applicationFilePath();
}

QStringList helperArguments(const QString& mode, const QString& outputPath = {})
{
    QStringList arguments{kChildArgument, mode};
    if (!outputPath.isEmpty()) {
        arguments.push_back(outputPath);
    }
    return arguments;
}

bool startHelper(CommandLineToolRunner& runner,
                 const QString& mode,
                 CommandLineToolRunner::ExecutionOptions options =
                     CommandLineToolRunner::ExecutionOptions::silent(),
                 const QString& outputPath = {})
{
    return runner.executeCustomCommand(
        helperProgram(), helperArguments(mode, outputPath), mode, options);
}

void expectSingleCompletion(QSignalSpy& finished)
{
    QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 1, 3000);
    QTest::qWait(100);
    QCOMPARE(finished.count(), 1);
}

bool completionSucceeded(const QSignalSpy& finished, int index = 0)
{
    return finished.at(index).at(1).toBool();
}

QString completionMessage(const QSignalSpy& finished, int index = 0)
{
    return finished.at(index).at(2).toString();
}

int runChild(const QStringList& arguments)
{
    const int marker = arguments.indexOf(kChildArgument);
    if (marker < 0 || marker + 1 >= arguments.size()) {
        return -1;
    }

    const QString mode = arguments.at(marker + 1);
    if (mode == "success") {
        return 0;
    }
    if (mode == "delayed-success") {
        QThread::msleep(150);
        return 0;
    }
    if (mode == "wait") {
        QThread::sleep(30);
        return 0;
    }
    if (mode == "crash") {
        std::raise(SIGKILL);
        return 1;
    }
    if (mode == "report") {
        // A report line split across writes, and one without a final newline.
        QFile out;
        out.open(stdout, QIODevice::WriteOnly);
        out.write("progress\nsurface_uu");
        out.flush();
        QThread::msleep(50);
        out.write("id: 11111111-1111-1111-1111-111111111111\nsurface_uuid: 22222222-2222-2222-2222-222222222222");
        out.flush();
        return 0;
    }
    if (mode == "write-omp" && marker + 2 < arguments.size()) {
        QFile output(arguments.at(marker + 2));
        if (!output.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            return 2;
        }
        output.write(qgetenv("OMP_NUM_THREADS"));
        return 0;
    }
    return 3;
}

} // namespace

class TestCommandLineToolRunner : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase()
    {
        qRegisterMetaType<CommandLineToolRunner::Tool>();
        qRegisterMetaType<QProcess::ProcessError>();
    }

    void normalExitFinishesOnce()
    {
        CommandLineToolRunner runner(nullptr, {});
        QSignalSpy finished(&runner, &CommandLineToolRunner::toolFinished);

        QVERIFY(startHelper(runner, "success"));
        expectSingleCompletion(finished);

        QVERIFY(completionSucceeded(finished));
        QVERIFY(!runner.isRunning());
    }

    void crashFinishesOnce()
    {
        CommandLineToolRunner runner(nullptr, {});
        QSignalSpy finished(&runner, &CommandLineToolRunner::toolFinished);

        QVERIFY(startHelper(runner, "crash"));
        expectSingleCompletion(finished);

        QVERIFY(!completionSucceeded(finished));
        QVERIFY(completionMessage(finished).contains("crash", Qt::CaseInsensitive));
        QVERIFY(!runner.isRunning());
    }

    void failedStartFinishesOnce()
    {
        QTemporaryFile invalidProgram;
        QVERIFY(invalidProgram.open());
        invalidProgram.write("#!/definitely/missing/command-runner-interpreter\n");
        invalidProgram.close();
        QVERIFY(QFile::setPermissions(
            invalidProgram.fileName(),
            QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner));

        CommandLineToolRunner runner(nullptr, {});
        QSignalSpy finished(&runner, &CommandLineToolRunner::toolFinished);

        QVERIFY(runner.executeCustomCommand(
            invalidProgram.fileName(), {}, "failed start",
            CommandLineToolRunner::ExecutionOptions::silent()));
        expectSingleCompletion(finished);

        QVERIFY(!completionSucceeded(finished));
        QVERIFY(completionMessage(finished).contains(
            "could not be started", Qt::CaseInsensitive));
        QVERIFY(!runner.isRunning());
    }

    void cancelFinishesOnceAndRunnerIsReusable()
    {
        CommandLineToolRunner runner(nullptr, {});
        QSignalSpy finished(&runner, &CommandLineToolRunner::toolFinished);

        QVERIFY(startHelper(runner, "wait"));
        QTRY_VERIFY_WITH_TIMEOUT(runner.isRunning(), 1000);
        runner.cancel();
        expectSingleCompletion(finished);

        QVERIFY(!completionSucceeded(finished));
        QVERIFY(!runner.isRunning());

        QVERIFY(startHelper(runner, "success"));
        QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 2, 3000);
        QVERIFY(completionSucceeded(finished, 1));
    }

    void completionObserverCanStartNextRun()
    {
        CommandLineToolRunner runner(nullptr, {});
        QSignalSpy finished(&runner, &CommandLineToolRunner::toolFinished);
        QList<bool> silentDuringDelivery;
        bool secondStarted = false;

        connect(
            &runner, &CommandLineToolRunner::toolFinished, &runner,
            [&](CommandLineToolRunner::Tool, bool, const QString&, const QString&, bool) {
                silentDuringDelivery.push_back(runner.currentExecutionIsSilent());
                if (silentDuringDelivery.size() == 1) {
                    secondStarted = startHelper(
                        runner, "success",
                        {CommandLineToolRunner::Presentation::Interactive, false});
                }
            });

        QVERIFY(startHelper(runner, "delayed-success"));
        QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 2, 3000);
        QTest::qWait(100);

        QCOMPARE(finished.count(), 2);
        QVERIFY(secondStarted);
        QVERIFY(completionSucceeded(finished, 0));
        QVERIFY(completionSucceeded(finished, 1));
        QCOMPARE(silentDuringDelivery, QList<bool>({true, false}));
        QVERIFY(!runner.isRunning());
    }

    void ompThreadsApplyToOneAttempt()
    {
        const bool hadOmpThreads = qEnvironmentVariableIsSet("OMP_NUM_THREADS");
        const QByteArray previousOmpThreads = qgetenv("OMP_NUM_THREADS");
        const auto restoreOmpThreads = qScopeGuard([=] {
            if (hadOmpThreads) {
                qputenv("OMP_NUM_THREADS", previousOmpThreads);
            } else {
                qunsetenv("OMP_NUM_THREADS");
            }
        });
        qputenv("OMP_NUM_THREADS", "17");

        QTemporaryFile firstOutput;
        QTemporaryFile secondOutput;
        QVERIFY(firstOutput.open());
        QVERIFY(secondOutput.open());
        const QString firstPath = firstOutput.fileName();
        const QString secondPath = secondOutput.fileName();
        firstOutput.close();
        secondOutput.close();

        CommandLineToolRunner runner(nullptr, {});
        QSignalSpy finished(&runner, &CommandLineToolRunner::toolFinished);

        runner.setNextOmpThreads(739);
        QVERIFY(startHelper(runner, "write-omp",
                            CommandLineToolRunner::ExecutionOptions::silent(), firstPath));
        QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 1, 3000);
        QVERIFY(completionSucceeded(finished, 0));

        QVERIFY(startHelper(runner, "write-omp",
                            CommandLineToolRunner::ExecutionOptions::silent(), secondPath));
        QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 2, 3000);
        QVERIFY(completionSucceeded(finished, 1));

        QFile first(firstPath);
        QFile second(secondPath);
        QVERIFY(first.open(QIODevice::ReadOnly));
        QVERIFY(second.open(QIODevice::ReadOnly));
        QCOMPARE(first.readAll(), QByteArray("739"));
        QCOMPARE(second.readAll(), QByteArray("17"));
    }

    void zarrRenderCanRequestSiblingTiffSlices()
    {
        CommandLineToolRunner runner(nullptr, {});
        runner.setVolumePath("volume.zarr");
        runner.setStore("objects.vcobj", 1.0);
        runner.setSurface("surface-uuid");
        const QString zarrPath = QDir::temp().filePath("render.zarr");
        runner.setOutputPattern(zarrPath);
        runner.setRenderParams(1.0f, 0, 4);
        runner.setRenderOutputFormat(CommandLineToolRunner::RenderOutputFormat::Zarr);
        runner.setIncludeTifs(true);

        const QStringList arguments =
            runner.argumentsForTesting(CommandLineToolRunner::Tool::RenderTifXYZ);
        QCOMPARE(arguments.count("--zarr-output"), 1);
        QCOMPARE(arguments.count("--tif-output"), 1);

        const int tifOption = arguments.indexOf("--tif-output");
        QVERIFY(tifOption >= 0 && tifOption + 1 < arguments.size());
        const QFileInfo zarrInfo(zarrPath);
        QCOMPARE(arguments.at(tifOption + 1),
                 zarrInfo.dir().filePath(zarrInfo.completeBaseName() + "_layers"));
    }

    void tiffRenderDoesNotAddASecondTiffOutput()
    {
        CommandLineToolRunner runner(nullptr, {});
        runner.setVolumePath("volume.zarr");
        runner.setStore("objects.vcobj", 1.0);
        runner.setSurface("surface-uuid");
        runner.setOutputPattern(QDir::temp().filePath("render_layers"));
        runner.setRenderParams(1.0f, 0, 4);
        runner.setRenderOutputFormat(CommandLineToolRunner::RenderOutputFormat::TifStack);
        runner.setIncludeTifs(true);

        const QStringList arguments =
            runner.argumentsForTesting(CommandLineToolRunner::Tool::RenderTifXYZ);
        QCOMPARE(arguments.count("--tif-output"), 1);
        QCOMPARE(arguments.count("--zarr-output"), 0);
    }

    void renderReadsTheStoredSurface()
    {
        CommandLineToolRunner runner(nullptr, {});
        runner.setVolumePath("volume.zarr");
        runner.setStore("objects.vcobj", 1.0);
        runner.setSurface("surface-uuid");
        runner.setOutputPattern(QDir::temp().filePath("render_layers"));
        const QStringList arguments =
            runner.argumentsForTesting(CommandLineToolRunner::Tool::RenderTifXYZ);
        const int store = arguments.indexOf("--store");
        const int surface = arguments.indexOf("--surface");
        QVERIFY(store >= 0 && surface >= 0);
        QCOMPARE(arguments.at(store + 1), QString("objects.vcobj"));
        QCOMPARE(arguments.at(surface + 1), QString("surface-uuid"));
        QVERIFY(!arguments.contains("--segmentation"));
    }

    // Every store tool takes the one store contract: --store, --volume-uuid,
    // its inputs as --surface, a new surface's --name, --tag.
    void storeToolsTakeOneContract()
    {
        const auto value = [](const QStringList& args, const QString& option) {
            const int at = args.indexOf(option);
            return at >= 0 && at + 1 < args.size() ? args.at(at + 1) : QString();
        };
        CommandLineToolRunner runner(nullptr, {});
        runner.setStore("objects.vcobj", 1.0);

        runner.setTraceParams("volume.zarr", "vol", "source", "trace", "params.json", "src-uuid");
        auto args = runner.argumentsForTesting(CommandLineToolRunner::Tool::GrowSegFromSegment);
        QCOMPARE(value(args, "--store"), QString("objects.vcobj"));
        QCOMPARE(value(args, "--surface"), QString("src-uuid"));
        QCOMPARE(value(args, "--volume-uuid"), QString("vol"));
        QVERIFY2(!args.contains("--voxel-scale"), "the volume's own voxels need no scale");
        QCOMPARE(value(args, "--tag"), QString("trace"));

        runner.setNeighborCopyParams("volume.zarr", "params.json", "resume-uuid", "vol", "copy", "");
        args = runner.argumentsForTesting(CommandLineToolRunner::Tool::NeighborCopy);
        QCOMPARE(value(args, "--surface"), QString("resume-uuid"));
        QCOMPARE(value(args, "--tag"), QString("copy"));

        runner.setObjRefineParams("volume.zarr", "src-uuid", "refined", "params.json", "vol");
        args = runner.argumentsForTesting(CommandLineToolRunner::Tool::AlphaCompRefine);
        QCOMPARE(value(args, "--surface"), QString("src-uuid"));
        QCOMPARE(value(args, "--name"), QString("refined"));
        QCOMPARE(args.mid(args.indexOf("volume.zarr"), 2), QStringList({"volume.zarr", "params.json"}));
        runner.setObjRefineParams("volume.zarr", "src-uuid", "", "params.json", "vol");
        args = runner.argumentsForTesting(CommandLineToolRunner::Tool::AlphaCompRefine);
        QVERIFY2(!args.contains("--name"), "no name refines in place");

        runner.setMergePatchParams("parent-uuid", "child-uuid", true, 16, 6, 4, 3000, 5, 10, 3, 0, 0);
        args = runner.argumentsForTesting(CommandLineToolRunner::Tool::MergePatch);
        QCOMPARE(args.count("--surface"), 2);
        QCOMPARE(args.at(args.indexOf("--surface") + 1), QString("parent-uuid"));
        QCOMPARE(args.at(args.lastIndexOf("--surface") + 1), QString("child-uuid"));
        QVERIFY(args.contains("--explicit-roles"));
        QVERIFY(!args.contains("--parent") && !args.contains("--child"));

        // At a working level the tools work in that level's voxels of the volume.
        runner.setStore("objects.vcobj", 0.25);
        runner.setTraceParams("volume.zarr", "vol", "source", "trace", "params.json", "src-uuid");
        args = runner.argumentsForTesting(CommandLineToolRunner::Tool::GrowSegFromSegment);
        QCOMPARE(value(args, "--volume-uuid"), QString("vol"));
        QCOMPARE(value(args, "--voxel-scale"), QString("0.25"));
    }

    void createdSurfacesComeFromReportLines()
    {
        CommandLineToolRunner runner(nullptr, {});
        QSignalSpy finished(&runner, &CommandLineToolRunner::toolFinished);
        QVERIFY(startHelper(runner, "report"));
        expectSingleCompletion(finished);
        QVERIFY(completionSucceeded(finished));
        QCOMPARE(runner.createdSurfaces(),
                 QStringList({"11111111-1111-1111-1111-111111111111",
                              "22222222-2222-2222-2222-222222222222"}));
    }
};

int main(int argc, char** argv)
{
    const QStringList arguments = [&] {
        QStringList values;
        values.reserve(argc);
        for (int i = 0; i < argc; ++i) {
            values.push_back(QString::fromLocal8Bit(argv[i]));
        }
        return values;
    }();

    if (arguments.contains(kChildArgument)) {
        return runChild(arguments);
    }

    QApplication application(argc, argv);
    TestCommandLineToolRunner test;
    return QTest::qExec(&test, argc, argv);
}

#include "test_command_line_tool_runner.moc"
