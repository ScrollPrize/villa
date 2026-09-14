#include <QtTest>
#include <QJsonDocument>
#include <QJsonArray>
#include <QFile>
#include <QDir>
#include <QDirIterator>
#include <QTemporaryDir>
#include <QProcess>
#include <QSemaphore>
#include <QScopeGuard>
#include <QtConcurrent/QtConcurrent>
#include "SpiralServiceManager.hpp"
#include "SpiralActivityWidget.hpp"

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
    void pclEditsResolveSourceDocument() {
        SpiralServiceManager client;
        const QJsonObject first{{"id", "a"}, {"kind", "pcl"}, {"role", "same_winding"},
            {"source", "/dataset/first.json"}, {"collection_id", 0},
            {"accepted_revision", 1}, {"applied_revision", 1}, {"persisted_revision", 1}};
        auto second = first;
        second["id"] = "b";
        second["source"] = "/dataset/second.json";
        client.installInputCatalog({first, second});
        QTemporaryDir root;
        const auto path = root.filePath("replacement.json");
        write(path, R"({"collections":{"0":{"points":{}}}})");
        client.stagePclReplacement(vc3d::spiral::PclRole::SameWinding, path,
            "overlay", "replace_collection", "0", "/dataset/second.json");
        QVERIFY(!client._inputDrafts["a"]->dirty());
        QVERIFY(client._inputDrafts["b"]->dirty());
        client.stagePclReplacement(vc3d::spiral::PclRole::SameWinding, {},
            "overlay", "delete_collection", "0", "/dataset/second.json");
        QVERIFY(client._inputDrafts["b"]->deleted());
        QVERIFY(!client._inputDrafts["a"]->deleted());
        // A missing source must never select an arbitrary document.
        QSignalSpy errors(&client, &SpiralServiceManager::errorOccurred);
        client.stagePclReplacement(vc3d::spiral::PclRole::SameWinding, {},
            "unknown-overlay", "delete_collection", "0");
        QCOMPARE(errors.size(), 1);
        QCOMPARE(client._inputDrafts.size(), 2);
        QVERIFY(!client._inputDrafts["a"]->dirty());
        // Catalog editors carry the UUID directly.
        client.stagePclReplacement(vc3d::spiral::PclRole::SameWinding, {},
            "a", "delete_collection", "0");
        QVERIFY(client._inputDrafts["a"]->deleted());
    }

    void revisionOnlyRetry_data() {
        QTest::addColumn<int>("persisted");
        QTest::addColumn<bool>("commit");
        QTest::newRow("external-apply") << 2 << false;
        QTest::newRow("external-commit") << 2 << true;
        QTest::newRow("submitted-apply") << 1 << false;
    }

    void revisionOnlyRetry() {
        QFETCH(int, persisted);
        QFETCH(bool, commit);
        SpiralServiceManager client;
        QJsonObject row{{"id", "input"}, {"kind", "fiber"}, {"accepted_revision", 2},
            {"applied_revision", 1}, {"persisted_revision", persisted}};
        client.installInputCatalog({row});
        client._inputOwner = true;
        client._inputErrors["input"] = "previous application failure";
        QSignalSpy completed(&client, &SpiralServiceManager::inputBatchFinished);
        client.applyInputDrafts(commit);
        QVERIFY(client._inputCommand);
        QVERIFY(client._inputCommand->batch.entries.isEmpty());
        QCOMPARE(client._inputCommand->revisions,
            QJsonArray({QJsonObject{{"id", "input"}, {"revision", 2}}}));
        // Complete the selected revision as a successful service response would.
        row["applied_revision"] = 2;
        if (commit) row["persisted_revision"] = 2;
        client.installInputCatalog({row});
        client.finishInputCommand();
        QCOMPARE(successCount(completed), 1);
        QVERIFY(client._inputErrors.isEmpty());
        QVERIFY(client.inputDraftStatus().first().toObject().value("error").toString().isEmpty());
        QThreadPool::globalInstance()->waitForDone();
        QCoreApplication::processEvents();
    }

    void preparationAndCopyStayVisibleInPanel() {
        SpiralActivityWidget activity;
        auto* label = activity.findChild<QLabel*>(QStringLiteral("spiralInputActivityText"));
        QVERIFY(label);
        QVERIFY(activity.isHidden());
        activity.setPreparation(QStringLiteral("Preparing dataset snapshots"));
        QVERIFY(!activity.isHidden());
        QVERIFY(label->text().contains(QStringLiteral("Preparing dataset snapshots")));
        QTRY_VERIFY_WITH_TIMEOUT(label->text().contains(QStringLiteral("Elapsed: 0m 1s")), 3000);
        activity.setCopy(1, QStringLiteral("Copying inputs: 10 files, 20 MiB"));
        QVERIFY(label->text().contains(QStringLiteral("Preparing dataset snapshots")));
        QVERIFY(label->text().contains(QStringLiteral("10 files, 20 MiB")));
        activity.setPreparation({});
        QVERIFY(!activity.isHidden());
        QVERIFY(!label->text().contains(QStringLiteral("Preparing dataset snapshots")));
        activity.setCopy(0, {});
        QVERIFY(activity.isHidden());
        activity.setPreparation(QStringLiteral("Loading input list"));
        activity.reset();
        QVERIFY(activity.isHidden());
        QVERIFY(label->text().isEmpty());
    }

    void asynchronousWorkingCopy() {
        const auto source = qEnvironmentVariable("SPIRAL_TEST_COPY_SOURCE");
        if (source.isEmpty()) QSKIP("Set SPIRAL_TEST_COPY_SOURCE to a real input directory");
        QVERIFY(QFileInfo(source).isDir());
        SpiralServiceManager client;
        QSignalSpy progress(&client, &SpiralServiceManager::inputCopyProgress);
        QString working, error, second;
        int completions = 0;
        int ticks = 0;
        QTimer heartbeat;
        heartbeat.setInterval(1);
        connect(&heartbeat, &QTimer::timeout, this, [&]() { ++ticks; });
        heartbeat.start();
        QElapsedTimer elapsed;
        elapsed.start();
        client.workingCopyAsync(source, [&](const QString& path, const QString& message) {
            working = path; error = message; ++completions;
            QCOMPARE(QThread::currentThread(), client.thread());
        });
        const auto launchMs = elapsed.elapsed();
        client.workingCopyAsync(source, [&](const QString& path, const QString&) {
            second = path; ++completions;
        });
        QCOMPARE(completions, 0);
        QTRY_COMPARE_WITH_TIMEOUT(completions, 2, 120000);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QVERIFY(!working.isEmpty());
        QCOMPARE(second, working);
        QVERIFY(ticks > 0);
        QVERIFY(!progress.isEmpty());
        QCOMPARE(progress.last()[0].toInt(), 0);
        qInfo() << "Copy launch ms:" << launchMs << "completion ms:" << elapsed.elapsed()
                << "UI heartbeat ticks:" << ticks;
        QDirIterator files(source, QDir::Files | QDir::Hidden, QDirIterator::Subdirectories);
        int checked = 0;
        while (files.hasNext()) {
            const auto original = files.next();
            QFile before(original), after(QDir(working).filePath(QDir(source).relativeFilePath(original)));
            QVERIFY(before.open(QIODevice::ReadOnly));
            QVERIFY(after.open(QIODevice::ReadOnly));
            QCOMPARE(before.size(), after.size());
            while (!before.atEnd()) QCOMPARE(before.read(1024 * 1024), after.read(1024 * 1024));
            ++checked;
        }
        QVERIFY(checked > 0);
        qInfo() << "Verified identical files:" << checked;
        bool reused = false;
        client.workingCopyAsync(source, [&](const QString& path, const QString& message) {
            QCOMPARE(path, working); QVERIFY(message.isEmpty()); reused = true;
        });
        QVERIFY(reused);
    }

    void cancelledCopyDoesNotOpenEditor() {
        const auto source = qEnvironmentVariable("SPIRAL_TEST_COPY_SOURCE");
        if (source.isEmpty()) QSKIP("Set SPIRAL_TEST_COPY_SOURCE to a real input directory");
        SpiralServiceManager client;
        bool called = false;
        client.workingCopyAsync(source, [&](const QString&, const QString&) { called = true; });
        client.disconnectFromService();
        QTest::qWait(200);
        QVERIFY(!called);
        bool completed = false;
        client.workingCopyAsync(source, [&](const QString& path, const QString& error) {
            QVERIFY2(error.isEmpty(), qPrintable(error));
            QVERIFY(!path.isEmpty()); completed = true;
        });
        QTRY_VERIFY_WITH_TIMEOUT(completed, 120000);
        QVERIFY(!called);
    }

    void workingCopyFailure() {
        QTemporaryDir root;
        SpiralServiceManager client;
        bool completed = false;
        client.workingCopyAsync(root.filePath("missing"), [&](const QString& path, const QString& error) {
            QVERIFY(path.isEmpty()); QVERIFY(!error.isEmpty()); completed = true;
        });
        QTRY_VERIFY(completed);
    }

    void pclEditorsReopenAcceptedRevision() {
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
        environment.remove(QStringLiteral("SPIRAL_REVISION_REMOTE_CATALOG"));
        service.setProcessEnvironment(environment);
        service.start(python, {QStringLiteral(SPIRAL_CLIENT_SERVICE_FIXTURE), root.path()});
        QVERIFY(service.waitForStarted());
        QVERIFY(service.waitForReadyRead(30000));
        const int port = service.readLine().trimmed().toInt();
        QVERIFY(port > 0);
        const auto stopService = qScopeGuard([&]() {
            service.terminate();
            service.waitForFinished(5000);
        });
        auto original = document(qEnvironmentVariable("SPIRAL_TEST_PCL_SOURCE",
            root.filePath(QStringLiteral("pcl-template.json"))));
        const auto collections = original.value(QStringLiteral("collections")).toObject();
        QVERIFY(!collections.isEmpty());
        // Keep the real collection's geometry and metadata, assigning a key
        // different from the catalog's eventual collection ID.
        original[QStringLiteral("collections")] = QJsonObject{{QStringLiteral("17"), collections.begin().value()}};
        const auto renamed = [](QJsonObject value, const QString& name) {
            auto collection = value.value(QStringLiteral("collections")).toObject().value(QStringLiteral("17")).toObject();
            collection[QStringLiteral("name")] = name;
            value[QStringLiteral("collections")] = QJsonObject{{QStringLiteral("17"), collection}};
            return value;
        };
        SpiralServiceProfile profile;
        profile.id = QStringLiteral("pcl-reopen-test");
        profile.baseUrl = QUrl(QStringLiteral("http://127.0.0.1:%1").arg(port));
        profile.apiKey = QStringLiteral("test-key");
        SpiralServiceManager client;
        client.connectToService(profile);
        QTRY_VERIFY_WITH_TIMEOUT(client.ownsInputWorkspace(), 10000);
        QSignalSpy completed(&client, &SpiralServiceManager::inputBatchFinished);
        QSignalSpy editors(&client, &SpiralServiceManager::inputEditorRequested);
        const auto source = root.filePath(QStringLiteral("local-pcl.json"));
        write(source, QJsonDocument(original).toJson());
        client.stageJsonInput(QStringLiteral("pcl"), source, QStringLiteral("pcl"), QStringLiteral("same_winding"));
        const auto id = client.inputDraftStatus().first().toObject().value(QStringLiteral("id")).toString();
        client.applyInputDrafts(true);
        QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 1, 10000);
        const auto accepted = renamed(original, QStringLiteral("accepted edit"));
        write(source, QJsonDocument(accepted).toJson());
        client.stageJsonInput(QStringLiteral("pcl"), source, id, QStringLiteral("same_winding"));
        client.applyInputDrafts();
        QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 2, 10000);
        const auto published = document(root.filePath(QStringLiteral("dataset/same_windings.json")));
        QVERIFY(!published.isEmpty());
        const auto local = renamed(original, QStringLiteral("discard me"));
        write(source, QJsonDocument(local).toJson());
        client.stageJsonInput(QStringLiteral("pcl"), source, id, QStringLiteral("same_winding"));
        client.editInputDraft(id);
        QTRY_COMPARE_WITH_TIMEOUT(editors.size(), 1, 10000);
        QCOMPARE(document(editors.last()[1].toString()), local);
        client.discardInputDraft(id);
        client.editInputDraft(id);
        QTRY_COMPARE_WITH_TIMEOUT(editors.size(), 2, 10000);
        QCOMPARE(document(editors.last()[1].toString()), accepted);
        QCOMPARE(editors.last()[0].toJsonObject().value(QStringLiteral("alias")).toString(), id);
        QCOMPARE(document(root.filePath(QStringLiteral("dataset/same_windings.json"))), published);

        // A new accepted input must open even though it has no dataset artifact.
        write(source, QJsonDocument(original).toJson());
        client.stageJsonInput(QStringLiteral("pcl"), source, QStringLiteral("new-pcl"), QStringLiteral("same_winding"));
        const auto newId = client.inputDraftStatus().last().toObject().value(QStringLiteral("id")).toString();
        QVERIFY(newId != id);
        client.applyInputDrafts();
        QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 3, 10000);
        write(source, QJsonDocument(local).toJson());
        client.stageJsonInput(QStringLiteral("pcl"), source, newId, QStringLiteral("same_winding"));
        client.discardInputDraft(newId);
        client.editInputDraft(newId);
        QTRY_COMPARE_WITH_TIMEOUT(editors.size(), 3, 10000);
        QCOMPARE(document(editors.last()[1].toString()), original);
        QCOMPARE(document(root.filePath(QStringLiteral("dataset/same_windings.json"))), published);
        client.disconnectFromService();
    }

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
        // A mutable editor copy must never be reused as the saved snapshot.
        QSignalSpy editors(&client, &SpiralServiceManager::inputEditorRequested);
        client.editInputDraft(firstId);
        QTRY_COMPARE_WITH_TIMEOUT(editors.size(), 1, 10000);
        const auto editorPath = editors.last()[1].toString();
        auto edited = original;
        edited[QStringLiteral("name")] = QStringLiteral("unsaved editor change");
        write(editorPath, QJsonDocument(edited).toJson());
        QSignalSpy copies(&client, &SpiralServiceManager::inputCopyProgress);
        client.resolveInputConflict(conflicts.last()[0].toJsonObject(), QStringLiteral("save_as_new"));
        QTRY_COMPARE_WITH_TIMEOUT(client.inputDraftStatus().size(), qsizetype(2), 10000);
        QVERIFY(!copies.isEmpty());
        QCOMPARE(copies.last()[0].toInt(), 0);
        QString newId;
        for (const auto& value : client.inputDraftStatus()) {
            const auto row = value.toObject();
            if (row.value(QStringLiteral("id")).toString() == firstId) continue;
            newId = row.value(QStringLiteral("id")).toString();
            const auto savedPath = row.value(QStringLiteral("path")).toString();
            QVERIFY(savedPath != editorPath);
            QCOMPARE(document(savedPath), original);
            edited[QStringLiteral("name")] = QStringLiteral("later editor change");
            write(editorPath, QJsonDocument(edited).toJson());
            QCOMPARE(document(savedPath), original);
        }
        QVERIFY(!newId.isEmpty());
        client.applyInputDrafts(true);
        QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 2, 10000);
        QCOMPARE(document(root.filePath(QStringLiteral("dataset/fibers/%1.json").arg(newId))), original);
        QCOMPARE(document(baseline), external);
        // Conflict resolution accepted the external revision. Open and mutate it,
        // then discard without changing that accepted revision again.
        client.editInputDraft(firstId);
        QTRY_COMPARE_WITH_TIMEOUT(editors.size(), 2, 10000);
        const auto acceptedEditorPath = editors.last()[1].toString();
        QCOMPARE(document(acceptedEditorPath), external);
        write(acceptedEditorPath, QJsonDocument(edited).toJson());
        client.discardInputDraft(firstId);
        QTRY_COMPARE_WITH_TIMEOUT(editors.size(), 3, 10000);
        const auto restoredPath = editors.last()[1].toString();
        QVERIFY(restoredPath != acceptedEditorPath);
        QCOMPARE(document(restoredPath), external);
        client.editInputDraft(firstId);
        QTRY_COMPARE_WITH_TIMEOUT(editors.size(), 4, 10000);
        QCOMPARE(editors.last()[1].toString(), restoredPath);
        QCOMPARE(document(editors.last()[1].toString()), external);
        // Also discard after staging and opening a separate local snapshot.
        write(restoredPath, QJsonDocument(edited).toJson());
        client.stageJsonInput(QStringLiteral("fiber"), restoredPath, firstId);
        client.editInputDraft(firstId);
        QTRY_COMPARE_WITH_TIMEOUT(editors.size(), 5, 10000);
        QCOMPARE(document(editors.last()[1].toString()), edited);
        client.discardInputDraft(firstId);
        QTRY_COMPARE_WITH_TIMEOUT(editors.size(), 6, 10000);
        QCOMPARE(document(editors.last()[1].toString()), external);
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
        QSignalSpy preparation(&client, &SpiralServiceManager::inputPreparationProgress);
        client.connectToService(profile);
        QTRY_VERIFY_WITH_TIMEOUT(client.ownsInputWorkspace(), 10000);
        QVERIFY(preparation.size() >= 4);
        QVERIFY(preparation.first()[0].toString().contains(QStringLiteral("catalog")));
        QVERIFY(preparation.last()[0].toString().isEmpty());
        QCOMPARE(client.inputDraftStatus().size(), qsizetype(1));
        const auto firstWorkspace = client.inputWorkspaceId();
        const auto firstId = fiberRow(client).value(QStringLiteral("id")).toString();
        QVERIFY(!fiberRow(client).value(QStringLiteral("session_changed")).toBool());
        QVERIFY(!QFile::exists(fiberRow(client).value(QStringLiteral("path")).toString()));
        const auto baseline = root.filePath(QStringLiteral("dataset/fibers/baseline.json"));
        const auto original = document(baseline);
        QSignalSpy completed(&client, &SpiralServiceManager::inputBatchFinished);
        client.removeInputDraft(firstId);
        QVERIFY(fiberRow(client).value(QStringLiteral("session_changed")).toBool());
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
        QVERIFY(fiberRow(client).value(QStringLiteral("session_changed")).toBool());

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
        QVERIFY(!fiberRow(client).value(QStringLiteral("session_changed")).toBool());
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

        // Exit discards accepted drafts without restoring/applying the dataset.
        const auto workspacePath = root.filePath(QStringLiteral("output/editing-workspaces/%1")
                                                .arg(client.inputWorkspaceId()));
        QVERIFY(QFileInfo(workspacePath).isDir());
        auto disposable = edited;
        disposable[QStringLiteral("name")] = QStringLiteral("discard on exit");
        write(local, QJsonDocument(disposable).toJson());
        client.stageJsonInput(QStringLiteral("fiber"), local, QStringLiteral("baseline"));
        client.applyInputDrafts();
        QTRY_COMPARE_WITH_TIMEOUT(successCount(completed), 5, 10000);
        bool released = false;
        client.releaseInputWorkspace([&]() { released = true; });
        QTRY_VERIFY_WITH_TIMEOUT(released, 10000);
        QVERIFY(!client.ownsInputWorkspace());
        QVERIFY(client.inputDraftStatus().isEmpty());
        QVERIFY(!QFileInfo::exists(workspacePath));
        QCOMPARE(document(baseline), edited);
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
            // Hold the pool so preparation cannot start. Apply must return
            // without sending anything, and another edit must remain separate.
            auto* pool = QThreadPool::globalInstance();
            const int previousThreads = pool->maxThreadCount();
            pool->waitForDone();
            pool->setMaxThreadCount(1);
            QSemaphore started, release;
            auto blocker = QtConcurrent::run([&]() { started.release(); release.acquire(); });
            const auto restorePool = qScopeGuard([&]() {
                release.release();
                blocker.waitForFinished();
                pool->setMaxThreadCount(previousThreads);
            });
            QVERIFY(started.tryAcquire(1, 10000));
            client.applyInputDrafts(true);
            QTest::qWait(100);
            QVERIFY(!QFile::exists(root.filePath(QStringLiteral("applying"))));
            write(source, second);
            client.stageJsonInput(QStringLiteral("fiber"), source, QStringLiteral("fiber"));
            // Repeated Apply while preparing must not start an early transfer.
            client.applyInputDrafts();
            release.release();
            QTRY_VERIFY_WITH_TIMEOUT(QFile::exists(root.filePath(QStringLiteral("applying"))), 10000);
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
