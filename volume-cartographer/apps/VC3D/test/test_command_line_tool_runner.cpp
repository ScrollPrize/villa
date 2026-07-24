#include <csignal>

#include <QApplication>
#include <QFile>
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
            "failed to start", Qt::CaseInsensitive));
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

    void pendingProcessErrorOverridesCleanExit()
    {
        CommandLineToolRunner runner(nullptr, {});
        QSignalSpy finished(&runner, &CommandLineToolRunner::toolFinished);

        QVERIFY(startHelper(runner, "delayed-success"));
        QTRY_VERIFY_WITH_TIMEOUT(runner.isRunning(), 1000);
        QVERIFY(QMetaObject::invokeMethod(
            &runner,
            "onProcessError",
            Qt::DirectConnection,
            Q_ARG(QProcess::ProcessError, QProcess::ReadError)));
        expectSingleCompletion(finished);

        QVERIFY(!completionSucceeded(finished));
        QVERIFY(completionMessage(finished).contains(
            "read error", Qt::CaseInsensitive));
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
