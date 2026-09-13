#include <QtTest>
#include <QJsonDocument>
#include <QJsonArray>
#include <QFile>
#include <QDir>
#include <QTemporaryDir>
#include <QProcess>
#include "SpiralServiceManager.hpp"

class SpiralInputWorkflowTests : public QObject {
    Q_OBJECT
    static void write(const QString& path, const QByteArray& data) {
        QFile file(path);
        QVERIFY(file.open(QIODevice::WriteOnly));
        QCOMPARE(file.write(data), data.size());
    }
    static QJsonObject document(const QString& path) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) return {};
        return QJsonDocument::fromJson(file.readAll()).object();
    }
    static int successCount(const QSignalSpy& results) {
        int count = 0;
        for (const auto& result : results) if (result[0].toString().isEmpty()) ++count;
        return count;
    }
    static QJsonObject fiberRow(const SpiralServiceManager& client) {
        for (const auto& value : client.inputDraftStatus())
            if (value.toObject().value(QStringLiteral("kind")).toString() == QStringLiteral("fiber")) return value.toObject();
        return {};
    }
private slots:
    void remoteSaveAsNewCapturesAcceptedContent() {
        const auto python = qEnvironmentVariable("SPIRAL_TEST_PYTHON");
        if (python.isEmpty()) QSKIP("Set SPIRAL_TEST_PYTHON to the existing Spiral Python environment");
        QTemporaryDir root;
        QVERIFY(root.isValid());
        QProcess service;
        service.setProcessChannelMode(QProcess::ForwardedErrorChannel);
        auto environment = QProcessEnvironment::systemEnvironment();
        environment.remove(QStringLiteral("SPIRAL_REVISION_CLIENT_LIVE"));
        environment.remove(QStringLiteral("SPIRAL_REVISION_DROP_REPLIES"));
        environment.remove(QStringLiteral("SPIRAL_REVISION_FAIL_PUBLICATION"));
        environment.insert(QStringLiteral("SPIRAL_REVISION_REMOTE_CATALOG"), QStringLiteral("1"));
        service.setProcessEnvironment(environment);
        const auto start = [&]() {
            service.start(python, {QStringLiteral(SPIRAL_CLIENT_SERVICE_FIXTURE), root.path()});
            if (!service.waitForStarted() || !service.waitForReadyRead(30000)) return 0;
            return service.readLine().trimmed().toInt();
        };
        int port = start();
        QVERIFY(port > 0);
        SpiralServiceProfile profile;
        profile.id = QStringLiteral("remote-restart-test");
        profile.baseUrl = QUrl(QStringLiteral("http://127.0.0.1:%1").arg(port));
        profile.apiKey = QStringLiteral("test-key");
        SpiralServiceManager client;
        connect(&client, &SpiralServiceManager::errorOccurred, &client,
                [](const QString& error) { qWarning().noquote() << error; });
        client.connectToService(profile);
        QTRY_VERIFY_WITH_TIMEOUT(client.ownsInputWorkspace(), 10000);
        QCOMPARE(client.inputDraftStatus().size(), qsizetype(1));
        const auto firstWorkspace = client.inputWorkspaceId();
        const auto firstId = fiberRow(client).value(QStringLiteral("id")).toString();
        QVERIFY(!QFile::exists(fiberRow(client).value(QStringLiteral("path")).toString()));
        const auto baseline = root.filePath(QStringLiteral("dataset/fibers/baseline.json"));
        const auto original = document(baseline);
        QSignalSpy completed(&client, &SpiralServiceManager::inputBatchFinished);
        client.removeInputDraft(firstId);
        client.restoreInputDraft(firstId);
        client.applyInputDrafts();
        QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 1, 10000);
        auto external = original;
        external[QStringLiteral("name")] = QStringLiteral("external change");
        write(baseline, QJsonDocument(external).toJson());
        QSignalSpy conflicts(&client, &SpiralServiceManager::inputConflict);
        client.applyInputDrafts(true);
        QTRY_VERIFY_WITH_TIMEOUT(!conflicts.isEmpty(), 10000);
        client.resolveInputConflict(conflicts.last()[0].toJsonObject(), QStringLiteral("save_as_new"));
        QTRY_COMPARE_WITH_TIMEOUT(client.inputDraftStatus().size(), qsizetype(2), 10000);
        QString newId;
        for (const auto& value : client.inputDraftStatus()) {
            const auto row = value.toObject();
            if (row.value(QStringLiteral("id")).toString() == firstId) continue;
            newId = row.value(QStringLiteral("id")).toString();
            QCOMPARE(document(row.value(QStringLiteral("path")).toString()), original);
        }
        QVERIFY(!newId.isEmpty());
        client.applyInputDrafts(true);
        QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 2, 10000);
        QCOMPARE(document(root.filePath(QStringLiteral("dataset/fibers/%1.json").arg(newId))), original);
        QCOMPARE(document(baseline), external);
        client.disconnectFromService();
        service.terminate();
        QVERIFY(service.waitForFinished(5000));
    }
    void remoteRestoreAndRestartReplaceCatalog() {
        const auto python = qEnvironmentVariable("SPIRAL_TEST_PYTHON");
        if (python.isEmpty()) QSKIP("Set SPIRAL_TEST_PYTHON to the existing Spiral Python environment");
        QTemporaryDir root;
        QVERIFY(root.isValid());
        QProcess service;
        service.setProcessChannelMode(QProcess::ForwardedErrorChannel);
        auto environment = QProcessEnvironment::systemEnvironment();
        environment.remove(QStringLiteral("SPIRAL_REVISION_CLIENT_LIVE"));
        environment.remove(QStringLiteral("SPIRAL_REVISION_DROP_REPLIES"));
        environment.remove(QStringLiteral("SPIRAL_REVISION_FAIL_PUBLICATION"));
        environment.insert(QStringLiteral("SPIRAL_REVISION_REMOTE_CATALOG"), QStringLiteral("1"));
        service.setProcessEnvironment(environment);
        const auto start = [&]() {
            service.start(python, {QStringLiteral(SPIRAL_CLIENT_SERVICE_FIXTURE), root.path()});
            if (!service.waitForStarted() || !service.waitForReadyRead(30000)) return 0;
            return service.readLine().trimmed().toInt();
        };
        int port = start();
        QVERIFY(port > 0);
        SpiralServiceProfile profile;
        profile.id = QStringLiteral("remote-restart-test");
        profile.baseUrl = QUrl(QStringLiteral("http://127.0.0.1:%1").arg(port));
        profile.apiKey = QStringLiteral("test-key");
        SpiralServiceManager client;
        connect(&client, &SpiralServiceManager::errorOccurred, &client,
                [](const QString& error) { qWarning().noquote() << error; });
        client.connectToService(profile);
        QTRY_VERIFY_WITH_TIMEOUT(client.ownsInputWorkspace(), 10000);
        QCOMPARE(client.inputDraftStatus().size(), qsizetype(1));
        const auto firstWorkspace = client.inputWorkspaceId();
        const auto firstId = fiberRow(client).value(QStringLiteral("id")).toString();
        QVERIFY(!QFile::exists(fiberRow(client).value(QStringLiteral("path")).toString()));
        const auto baseline = root.filePath(QStringLiteral("dataset/fibers/baseline.json"));
        const auto original = document(baseline);
        QSignalSpy completed(&client, &SpiralServiceManager::inputBatchFinished);
        client.removeInputDraft(firstId);
        client.applyInputDrafts();
        QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 1, 10000);
        {
            SpiralServiceManager observer;
            observer.connectToService(profile);
            QTRY_COMPARE_WITH_TIMEOUT(observer.inputDraftStatus().size(), qsizetype(1), 10000);
            QVERIFY(fiberRow(observer).value(QStringLiteral("can_restore")).toBool());
            observer.restoreInputDraft(firstId);
            QVERIFY(!fiberRow(observer).value(QStringLiteral("deleted")).toBool());
            QVERIFY(fiberRow(observer).value(QStringLiteral("dirty")).toBool());
            observer.disconnectFromService();
        }
        client.restoreInputDraft(firstId);
        client.applyInputDrafts(true);
        QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 2, 10000);
        QCOMPARE(document(baseline), original);
        QVERIFY(!client.hasInputDrafts());

        // Establish an alias and an explicit selection in the old workspace.
        const auto local = root.filePath(QStringLiteral("local.json"));
        auto edited = original;
        edited[QStringLiteral("name")] = QStringLiteral("before restart");
        write(local, QJsonDocument(edited).toJson());
        client.stageJsonInput(QStringLiteral("fiber"), local, QStringLiteral("baseline"));
        client.setInputSelection({firstId});
        client.applyInputDrafts(true);
        QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 3, 10000);
        QVERIFY(!client.hasInputDrafts());
        client.disconnectFromService();
        service.terminate();
        QVERIFY(service.waitForFinished(5000));
        port = start();
        QVERIFY(port > 0);
        profile.baseUrl = QUrl(QStringLiteral("http://127.0.0.1:%1").arg(port));
        client.connectToService(profile);
        QTRY_VERIFY_WITH_TIMEOUT(client.ownsInputWorkspace(), 10000);
        QVERIFY(client.inputWorkspaceId() != firstWorkspace);
        QCOMPARE(client.inputDraftStatus().size(), qsizetype(1));
        const auto secondId = fiberRow(client).value(QStringLiteral("id")).toString();
        QVERIFY(secondId != firstId);
        edited[QStringLiteral("name")] = QStringLiteral("after restart");
        write(local, QJsonDocument(edited).toJson());
        client.stageJsonInput(QStringLiteral("fiber"), local, QStringLiteral("baseline"));
        QCOMPARE(client.inputDraftStatus().size(), qsizetype(1));
        QCOMPARE(fiberRow(client).value(QStringLiteral("id")).toString(), secondId);
        client.applyInputDrafts(true);
        QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 4, 10000);
        QCOMPARE(document(baseline), edited);
        QVERIFY(!client.hasInputDrafts());
        client.disconnectFromService();
        service.terminate();
        QVERIFY(service.waitForFinished(5000));
    }
    void editDuringApplyCommitAndReconnect() {
        const auto python = qEnvironmentVariable("SPIRAL_TEST_PYTHON");
        if (python.isEmpty()) QSKIP("Set SPIRAL_TEST_PYTHON to the existing Spiral Python environment");
        QTemporaryDir root;
        QVERIFY(root.isValid());
        QProcess service;
        service.setProcessChannelMode(QProcess::ForwardedErrorChannel);
        service.start(python, {QStringLiteral(SPIRAL_CLIENT_SERVICE_FIXTURE), root.path()});
        QVERIFY(service.waitForStarted());
        QVERIFY(service.waitForReadyRead(180000));
        const int port = service.readLine().trimmed().toInt();
        QVERIFY2(port > 0, service.readAllStandardError().constData());
        SpiralServiceProfile profile;
        profile.id = QStringLiteral("revision-test");
        profile.baseUrl = QUrl(QStringLiteral("http://127.0.0.1:%1").arg(port));
        profile.apiKey = QStringLiteral("test-key");
        {
            SpiralServiceManager client;
            QSignalSpy errors(&client, &SpiralServiceManager::errorOccurred);
            connect(&client, &SpiralServiceManager::errorOccurred, &client, [](const QString& error) { qWarning().noquote() << error; });
            client.connectToService(profile);
            QTRY_VERIFY_WITH_TIMEOUT(client.ownsInputWorkspace(), 10000);
            const auto workspace = client.inputWorkspaceId();
            const QString source = root.filePath(QStringLiteral("draft.json"));
            auto templateFiber = document(root.filePath(QStringLiteral("fiber-template.json")));
            templateFiber[QStringLiteral("name")] = QStringLiteral("first");
            const auto first = QJsonDocument(templateFiber).toJson();
            templateFiber[QStringLiteral("name")] = QStringLiteral("second");
            const auto second = QJsonDocument(templateFiber).toJson();
            write(source, first);
            client.stageJsonInput(QStringLiteral("fiber"), source, QStringLiteral("fiber"));
            QVERIFY(client.hasInputDrafts());
            QVERIFY(!fiberRow(client).isEmpty());
            client.stageJsonInput(QStringLiteral("pcl"), root.filePath(QStringLiteral("pcl-template.json")),
                QStringLiteral("pcl"), QStringLiteral("same_winding"));
            if (QDir(root.filePath(QStringLiteral("replacement"))).exists())
                client.stagePatch(root.filePath(QStringLiteral("replacement")), QStringLiteral("baseline"));
            const QString id = fiberRow(client).value(QStringLiteral("id")).toString();
            write(root.filePath(QStringLiteral("hold-apply")), "hold");
            QSignalSpy completed(&client, &SpiralServiceManager::inputBatchFinished);
            if (qEnvironmentVariableIsSet("SPIRAL_REVISION_DROP_REPLIES") || qEnvironmentVariableIsSet("SPIRAL_REVISION_FAIL_PUBLICATION"))
                connect(&client, &SpiralServiceManager::inputBatchFinished, &client, [&client](const QString& error) {
                    if (error.contains(QStringLiteral("unreachable")) || error.contains(QStringLiteral("recovery")))
                        QTimer::singleShot(0, &client, [&client]() { client.applyInputDrafts(); });
                });
            client.applyInputDrafts(true);
            QTRY_VERIFY_WITH_TIMEOUT(QFile::exists(root.filePath(QStringLiteral("applying"))), 10000);
            // A save during the worker boundary creates a newer local revision.
            write(source, second);
            client.stageJsonInput(QStringLiteral("fiber"), source, QStringLiteral("fiber"));
            QFile::remove(root.filePath(QStringLiteral("hold-apply")));
            QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 1, 10000);
            QVERIFY2(completed.last()[0].toString().isEmpty(), qPrintable(completed.last()[0].toString()));
            QCOMPARE(document(root.filePath(QStringLiteral("dataset/fibers/fiber.json"))).value(QStringLiteral("name")).toString(), QStringLiteral("first"));
            QVERIFY(fiberRow(client).value(QStringLiteral("dirty")).toBool());
            // Managed editor/autosave destinations are copies, including peers
            // needed by linked save batches. Mutating one cannot publish bytes.
            QString copyError;
            const auto fiberDirectory = root.filePath(QStringLiteral("dataset/fibers"));
            const auto working = client.workingCopy(fiberDirectory, &copyError);
            QVERIFY2(!working.isEmpty(), qPrintable(copyError));
            write(QDir(working).filePath(QStringLiteral("fiber.json")), second);
            QCOMPARE(document(root.filePath(QStringLiteral("dataset/fibers/fiber.json"))).value(QStringLiteral("name")).toString(), QStringLiteral("first"));
            QCOMPARE(client.workingCopy(fiberDirectory), working);
            client.reconnect();
            QTRY_VERIFY_WITH_TIMEOUT(client.isReady() && client.ownsInputWorkspace(), 10000);
            QTest::qWait(200);
            QCOMPARE(client.inputWorkspaceId(), workspace);
            QVERIFY(client.hasInputDrafts());
            client.applyInputDrafts(true);
            QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 2, 10000);
            QVERIFY2(completed.last()[0].toString().isEmpty(), qPrintable(completed.last()[0].toString()));
            QCOMPARE(document(root.filePath(QStringLiteral("dataset/fibers/fiber.json"))).value(QStringLiteral("name")).toString(), QStringLiteral("second"));
            QVERIFY(!client.hasInputDrafts());
            client.removeInputDraft(id);
            client.applyInputDrafts();
            QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 3, 10000);
            QVERIFY(QFile::exists(root.filePath(QStringLiteral("dataset/fibers/fiber.json"))));
            QVERIFY(fiberRow(client).value(QStringLiteral("can_restore")).toBool());
            client.restoreInputDraft(id);
            client.applyInputDrafts(true);
            QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 4, 10000);
            QVERIFY2(completed.last()[0].toString().isEmpty(), qPrintable(completed.last()[0].toString()));
            client.removeInputDraft(id);
            client.applyInputDrafts(true);
            QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 5, 10000);
            QVERIFY(!QFile::exists(root.filePath(QStringLiteral("dataset/fibers/fiber.json"))));
            QVERIFY(!fiberRow(client).value(QStringLiteral("can_restore")).toBool());
            if (qEnvironmentVariableIsSet("SPIRAL_REVISION_DROP_REPLIES")) {
                QFile drops(root.filePath(QStringLiteral("dropped.json")));
                QVERIFY(drops.open(QIODevice::ReadOnly));
                QCOMPARE(QJsonDocument::fromJson(drops.readAll()).array().size(), 5);
            }
        }
        service.terminate();
        QVERIFY(service.waitForFinished(5000));
    }
};
QTEST_MAIN(SpiralInputWorkflowTests)
#include "test_spiral_input_workflow.moc"
