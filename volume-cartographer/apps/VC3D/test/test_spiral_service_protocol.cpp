// VC3D's Spiral client against the real store-backed service (its resident
// runtime faked by spiral-fitting/tests/protocol_client_service.py): the
// session lease, the store binding, service-owned configuration, the input
// manifest artifact and pinned reads, frozen inputs and the reload rebuild.
// Needs SPIRAL_TEST_PYTHON (a Spiral Python environment with vc_object_store
// on PYTHONPATH); skipped otherwise.
#include "SpiralServiceManager.hpp"
#include "SpiralStoreInputs.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/objects/Hub.hpp>

#include <QElapsedTimer>
#include <QFile>
#include <QProcess>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QtTest/QtTest>

#include <cstring>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
// One 2x2 surface in volume "vol", written through the session.
std::string addSurface(ObjectStoreSession& session, const AttachmentIdentity& attachment)
{
    const auto surface = util::newUuid();
    session.act(attachment, {{surface, std::nullopt}}, [&](auto& db, auto& txn) {
        ObjectRecord record; record.uuid = surface; record.kind = "surface"; record.name = "near";
        record.meta = Json{{"coordinates", volumeCoordinates("vol", 1)}}.dump();
        db.upsertObject(txn, record);
        std::vector<float> points;
        for (int row = 0; row < 2; ++row)
            for (int col = 0; col < 2; ++col) points.insert(points.end(), {float(col), float(row), 5.f});
        GeometryArray array{"points", "<f4", {2, 2, 3}, std::vector<uint8_t>(points.size() * 4)};
        std::memcpy(array.bytes.data(), points.data(), array.bytes.size());
        db.putGeometry(txn, surface, GeometryData{"surface", 1, "source_xyz", R"({"scale":[1.0,1.0]})", {array}});
    });
    return surface;
}

struct ServiceProcess {
    QProcess process;
    int port = 0;
    ~ServiceProcess() { process.kill(); process.waitForFinished(5000); }
};

bool startService(ServiceProcess& service, const QString& python, const QTemporaryDir& directory,
                  const QString& store, const QString& hub = {})
{
    service.process.setProcessChannelMode(QProcess::ForwardedChannels);
    const QString portFile = directory.filePath(QStringLiteral("port"));
    QStringList arguments{QStringLiteral(SPIRAL_PROTOCOL_SERVICE_FIXTURE), QStringLiteral("--store"), store,
                          QStringLiteral("--root"), directory.filePath(QStringLiteral("service")),
                          QStringLiteral("--port-file"), portFile};
    if (!hub.isEmpty()) arguments << QStringLiteral("--hub") << hub;
    service.process.start(python, arguments);
    if (!service.process.waitForStarted()) return false;
    QElapsedTimer timer; timer.start();
    while (!QFile::exists(portFile) && timer.elapsed() < 60000) QTest::qWait(50);
    QFile port(portFile);
    if (!port.open(QIODevice::ReadOnly)) return false;
    service.port = port.readAll().trimmed().toInt();
    return service.port > 0;
}

SpiralServiceProfile protocolProfile(int port)
{
    SpiralServiceProfile profile;
    profile.id = QStringLiteral("protocol");
    profile.baseUrl = QUrl(QStringLiteral("http://127.0.0.1:%1").arg(port));
    profile.apiKey = QStringLiteral("key");
    return profile;
}
}

class SpiralServiceProtocolTest : public QObject
{
    Q_OBJECT

private slots:
    // VC3D edits its replica of the service's hub; Update Fit Inputs pushes
    // them, the service links and pins a position including them (the first
    // initializes the fit, later ones update its inputs in place), and VC3D
    // pulls to it.
    void updateFitInputsAtAHubPosition()
    {
        try { updateFitInputsAtAHubPositionBody(); }
        catch (const std::exception& error) { QFAIL(error.what()); }
    }

    void storeBackedSessionLifecycle()
    {
        const auto python = qEnvironmentVariable("SPIRAL_TEST_PYTHON");
        if (python.isEmpty()) QSKIP("Set SPIRAL_TEST_PYTHON to a Spiral Python environment with vc_object_store");
        QTemporaryDir directory;
        const auto storePath = std::filesystem::path(directory.filePath(QStringLiteral("inputs.vcobj")).toStdString());
        OpenOptions options; options.author = "protocol-test";
        { auto created = ObjectStore::create(storePath, options); }
        auto session = std::make_shared<ObjectStoreSession>();
        const auto attachment = session->attach(storePath, options);
        const auto surface = util::newUuid();
        session->act(attachment, {{surface, std::nullopt}}, [&](auto& db, auto& txn) {
            ObjectRecord record; record.uuid = surface; record.kind = "surface"; record.name = "near";
            record.meta = Json{{"coordinates", volumeCoordinates("vol", 1)}}.dump();
            db.upsertObject(txn, record);
            std::vector<float> points;
            for (int row = 0; row < 2; ++row)
                for (int col = 0; col < 2; ++col) points.insert(points.end(), {float(col), float(row), 5.f});
            GeometryArray array{"points", "<f4", {2, 2, 3}, std::vector<uint8_t>(points.size() * 4)};
            std::memcpy(array.bytes.data(), points.data(), array.bytes.size());
            db.putGeometry(txn, surface, GeometryData{"surface", 1, "source_xyz", R"({"scale":[1.0,1.0]})", {array}});
        });

        QProcess service;
        service.setProcessChannelMode(QProcess::ForwardedChannels);
        const QString portFile = directory.filePath(QStringLiteral("port"));
        service.start(python, {QStringLiteral(SPIRAL_PROTOCOL_SERVICE_FIXTURE),
                               QStringLiteral("--store"), QString::fromStdString(storePath.string()),
                               QStringLiteral("--root"), directory.filePath(QStringLiteral("service")),
                               QStringLiteral("--port-file"), portFile});
        QVERIFY(service.waitForStarted());
        QTRY_VERIFY_WITH_TIMEOUT(QFile::exists(portFile), 60000);
        QFile port(portFile);
        QVERIFY(port.open(QIODevice::ReadOnly));
        const int number = port.readAll().trimmed().toInt();
        QVERIFY(number > 0);
        const auto cleanup = qScopeGuard([&]() { service.kill(); service.waitForFinished(5000); });

        SpiralServiceManager manager;
        manager.setObjectStore(session, attachment);
        QSignalSpy manifests(&manager, &SpiralServiceManager::inputManifestAvailable);
        QSignalSpy errors(&manager, &SpiralServiceManager::errorOccurred);
        SpiralServiceProfile profile;
        profile.id = QStringLiteral("protocol");
        profile.baseUrl = QUrl(QStringLiteral("http://127.0.0.1:%1").arg(number));
        profile.apiKey = QStringLiteral("key");
        manager.connectToService(profile);
        QTRY_VERIFY_WITH_TIMEOUT(manager.isReady() && manager.ownsSessionLease(), 30000);
        QVERIFY(!manager.sessionLeaseId().isEmpty());
        QTRY_VERIFY_WITH_TIMEOUT(manager.storeBinding().matches(), 10000);
        QTRY_VERIFY_WITH_TIMEOUT(manager.serviceOwnedKeys().contains(QStringLiteral("object_store_path")), 10000);

        // A saved profile naming another store is not a choice the service
        // lets a client make; the request drops it.
        const QJsonObject config{{"dense_spacing_mode", "grad_mag"}, {"loss_weight_dense_spacing", 0},
                                 {"loss_weight_dense_normals", 0}, {"loss_weight_shell_outer", 0},
                                 {"object_store_path", "/elsewhere.vcobj"}};
        const QJsonObject request{{"run", QJsonObject{{"z_begin", 0}, {"z_end", 10}, {"config", config}}}};
        manager.initializeSession(request);
        QTRY_VERIFY_WITH_TIMEOUT(manager.hasActiveSession(), 30000);

        // The fitted inputs' manifest arrives, and the attached store holds
        // its inputs as fitted (geometry, metadata values and links).
        QTRY_COMPARE_WITH_TIMEOUT(manifests.count(), 1, 30000);
        QFile manifestFile(manifests.at(0).at(0).toString());
        QVERIFY(manifestFile.open(QIODevice::ReadOnly));
        const auto document = Json::parse(manifestFile.readAll().toStdString());
        QCOMPARE(QString::fromStdString(document.at("digest").get<std::string>()), manifests.at(0).at(1).toString());
        vc3d::spiral::FitInputs inputs;
        session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            inputs = vc3d::spiral::readFitInputs(db, txn, document);
        });
        QVERIFY(inputs.unchanged());
        QCOMPARE(inputs.surfaces.size(), std::size_t{1});
        QCOMPARE(inputs.surfaces.front().state, vc3d::spiral::InputState::Fitted);

        // The reload rebuild is accepted and rebuilds the session.
        errors.clear();
        manager.rebuildSession(request, true);
        QTest::qWait(1000);
        QTRY_VERIFY_WITH_TIMEOUT(manager.hasActiveSession(), 30000);
        QVERIFY2(errors.isEmpty(), qPrintable(errors.isEmpty() ? QString() : errors.at(0).at(0).toString()));

        bool released = false;
        manager.releaseSessionLease([&]() { released = true; });
        QTRY_VERIFY_WITH_TIMEOUT(released, 30000);
        QVERIFY(!manager.ownsSessionLease());
    }

private:
    void updateFitInputsAtAHubPositionBody()
    {
        const auto python = qEnvironmentVariable("SPIRAL_TEST_PYTHON");
        if (python.isEmpty()) QSKIP("Set SPIRAL_TEST_PYTHON to a Spiral Python environment with vc_object_store");
        qRegisterMetaType<SpiralServiceManager::InputUpdate>();
        QTemporaryDir directory;
        const auto hubPath = std::filesystem::path(directory.filePath(QStringLiteral("hub.vcobj")).toStdString());
        const auto replicaPath = std::filesystem::path(directory.filePath(QStringLiteral("vc3d.vcobj")).toStdString());
        OpenOptions options; options.author = "vc3d-test";
        std::string surface;
        {
            auto origin = std::make_shared<ObjectStoreSession>();
            { auto created = ObjectStore::create(hubPath, options); }
            const auto attachment = origin->attach(hubPath, options);
            surface = addSurface(*origin, attachment);
            origin->detach(attachment);
        }
        initHub(hubPath, "curator");
        cloneHub(hubPath, replicaPath, options);
        auto session = std::make_shared<ObjectStoreSession>();
        HubSessionOptions fast; fast.syncInterval = std::chrono::milliseconds(200); fast.pushDelay = std::chrono::milliseconds(20);
        const auto attachment = session->attach(replicaPath, options, fast);

        ServiceProcess service;
        QVERIFY(startService(service, python, directory, directory.filePath(QStringLiteral("service-replica.vcobj")),
                             QString::fromStdString(hubPath.string())));
        SpiralServiceManager manager;
        manager.setObjectStore(session, attachment);
        QSignalSpy manifests(&manager, &SpiralServiceManager::inputManifestAvailable);
        QSignalSpy updates(&manager, &SpiralServiceManager::inputUpdateFinished);
        QSignalSpy sessions(&manager, &SpiralServiceManager::sessionSynchronized);
        manager.connectToService(protocolProfile(service.port));
        QTRY_VERIFY_WITH_TIMEOUT(manager.isReady() && manager.ownsSessionLease(), 30000);
        QTRY_VERIFY_WITH_TIMEOUT(manager.storeBinding().hosted() && manager.storeBinding().matches(), 10000);
        QTRY_VERIFY_WITH_TIMEOUT(manager.serviceOwnedKeys().contains(QStringLiteral("object_store_input_position")), 10000);

        auto edit = [&](int cells) {
            auto hold = session->holdLeases(attachment, {surface});
            RevisionSelection pins;
            session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                pins[surface] = db.currentObjectRevision(txn, surface)->uuid;
            });
            session->act(attachment, pins, [&](auto& db, auto& txn) {
                db.setMeta(txn, surface, "source", Json{{"spiral_patch_erode_cells", cells}}.dump());
            });
        };
        auto fittedMetadata = [&](int index) {
            QFile file(manifests.at(index).at(0).toString());
            if (!file.open(QIODevice::ReadOnly)) return Json();
            return Json::parse(file.readAll().toStdString());
        };
        const QJsonObject config{{"dense_spacing_mode", "grad_mag"}, {"loss_weight_dense_spacing", 0},
                                 {"loss_weight_dense_normals", 0}, {"loss_weight_shell_outer", 0}};
        const QJsonObject request{{"run", QJsonObject{{"z_begin", 0}, {"z_end", 10}, {"config", config}}}};

        // The first update initializes the fit at a position including the edit.
        edit(2);
        manager.updateFitInputs(request);
        QTRY_COMPARE_WITH_TIMEOUT(updates.count(), 1, 60000);
        auto update = updates.at(0).at(0).value<SpiralServiceManager::InputUpdate>();
        QVERIFY2(update.error.isEmpty(), qPrintable(update.error + update.problems.join(QStringLiteral("; "))));
        QVERIFY(update.requested > 0 && update.head >= update.requested && update.fitted >= update.head);
        QTRY_COMPARE_WITH_TIMEOUT(manifests.count(), 1, 30000);
        auto fitted = fittedMetadata(0);
        QCOMPARE(fitted.at("manifest").at("position").at("sequence").get<qint64>(), update.fitted);
        QCOMPARE(fitted.at("manifest").at("objects").at(0).at("metadata").at("spiral_patch_erode_cells").get<int>(), 2);
        QVERIFY(session->hubStatus(attachment)->position >= update.fitted);

        // The next one updates the resident fit in place: no new session.
        QTRY_COMPARE_WITH_TIMEOUT(sessions.count(), 1, 10000);
        edit(3);
        manager.updateFitInputs(request);
        QTRY_COMPARE_WITH_TIMEOUT(updates.count(), 2, 60000);
        update = updates.at(1).at(0).value<SpiralServiceManager::InputUpdate>();
        QVERIFY2(update.error.isEmpty(), qPrintable(update.error + update.problems.join(QStringLiteral("; "))));
        QTRY_COMPARE_WITH_TIMEOUT(manifests.count(), 2, 30000);
        fitted = fittedMetadata(1);
        QCOMPARE(fitted.at("manifest").at("position").at("sequence").get<qint64>(), update.fitted);
        QCOMPARE(fitted.at("manifest").at("objects").at(0).at("metadata").at("spiral_patch_erode_cells").get<int>(), 3);
        QCOMPARE(sessions.count(), 1);

        // A conflict blocks the update and leaves the fit as it was: another
        // replica changes a link this replica changed too, before it pushed.
        const auto other = util::newUuid();
        std::string link;
        {
            auto hold = session->holdLeases(attachment, {surface});
            RevisionSelection pins{{other, std::nullopt}};
            session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                pins[surface] = db.currentObjectRevision(txn, surface)->uuid;
            });
            session->act(attachment, pins, [&](auto& db, auto& txn) {
                ObjectRecord record; record.uuid = other; record.kind = "surface"; record.name = "far";
                record.meta = Json{{"coordinates", volumeCoordinates("vol", 1)}}.dump();
                db.upsertObject(txn, record);
                EdgeRecord edge; edge.kind = "related"; edge.aUuid = surface; edge.bUuid = other;
                edge.aAnchor.wholeGeometry = edge.bAnchor.wholeGeometry = true;
                link = db.addEdge(txn, edge);
            });
        }
        QVERIFY(session->syncHubAndWait(attachment, std::chrono::seconds(30))->pending.empty());
        const auto bobPath = std::filesystem::path(directory.filePath(QStringLiteral("bob.vcobj")).toStdString());
        OpenOptions bobOptions; bobOptions.author = "bob";
        cloneHub(hubPath, bobPath, bobOptions);
        auto held = session->holdPushes(attachment);
        RevisionSelection ends;
        session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            for (const auto& uuid : {surface, other}) ends[uuid] = db.currentObjectRevision(txn, uuid)->uuid;
        });
        session->act(attachment, ends, [&](auto& db, auto& txn) { db.setMeta(txn, link, "note", "\"vc3d\""); });
        {
            auto bob = ObjectStore::open(bobPath, bobOptions);
            auto txn = bob.write();
            bob.setMeta(txn, link, "note", "\"bob\"");
            txn.commit();
            QVERIFY(hubSync(bob).conflicts.empty());
        }
        const auto fittedBefore = update.fitted;
        manager.updateFitInputs(request);
        QTRY_COMPARE_WITH_TIMEOUT(updates.count(), 3, 60000);
        update = updates.at(2).at(0).value<SpiralServiceManager::InputUpdate>();
        QVERIFY(update.conflicts && !update.error.isEmpty());
        QCOMPARE(update.problems, QStringList{QString::fromStdString(link)});
        QTRY_VERIFY_WITH_TIMEOUT(!manager.inputUpdateActive(), 10000);
        QCOMPARE(manifests.count(), 2);  // the fit was not rebuilt
        held.reset();
        // Resolved to the hub's side, the next update goes through.
        session->resolveHubConflicts(attachment, "hub");
        QTRY_VERIFY2_WITH_TIMEOUT(session->hubStatus(attachment)->conflicts.empty(),
                                  session->hubStatus(attachment)->error.c_str(), 30000);
        manager.updateFitInputs(request);
        QTRY_COMPARE_WITH_TIMEOUT(updates.count(), 4, 60000);
        update = updates.at(3).at(0).value<SpiralServiceManager::InputUpdate>();
        QVERIFY2(update.error.isEmpty(), qPrintable(update.error + update.problems.join(QStringLiteral("; "))));
        QVERIFY(update.fitted > fittedBefore);

        // A connection change during an update ends it here (its replies
        // are dropped) and resumes this replica's pushes.
        QSignalSpy progress(&manager, &SpiralServiceManager::inputUpdateProgress);
        edit(4);
        manager.updateFitInputs(request);
        QTRY_VERIFY_WITH_TIMEOUT(!progress.isEmpty() && progress.last().at(0).toString() == QStringLiteral("linking"),
                                 60000);
        manager.disconnectFromService();
        QCOMPARE(updates.count(), 5);
        update = updates.at(4).at(0).value<SpiralServiceManager::InputUpdate>();
        QVERIFY(update.error.contains(QStringLiteral("outcome is unknown")));
        QVERIFY(!manager.inputUpdateActive());
        edit(5);
        QTRY_VERIFY_WITH_TIMEOUT(session->hubStatus(attachment)->pending.empty(), 30000);
        manager.connectToService(protocolProfile(service.port));
        QTRY_VERIFY_WITH_TIMEOUT(manager.isReady() && manager.ownsSessionLease(), 30000);
        QTest::qWait(500);
        QCOMPARE(updates.count(), 5);  // the abandoned update's late replies are ignored

        // Detaching the store ends an update too: its push hold went with it.
        progress.clear();
        manager.updateFitInputs(request);
        QTRY_VERIFY_WITH_TIMEOUT(!progress.isEmpty() && progress.last().at(0).toString() == QStringLiteral("linking"),
                                 60000);
        manager.setObjectStore(nullptr, std::nullopt);
        QCOMPARE(updates.count(), 6);
        update = updates.at(5).at(0).value<SpiralServiceManager::InputUpdate>();
        QVERIFY(update.error.contains(QStringLiteral("attached object store changed")));
        QVERIFY(!manager.inputUpdateActive() && !session->hubStatus(attachment)->pushesHeld);
        manager.setObjectStore(session, attachment);
        QTest::qWait(500);
        QCOMPARE(updates.count(), 6);

        bool released = false;
        manager.releaseSessionLease([&]() { released = true; });
        QTRY_VERIFY_WITH_TIMEOUT(released, 30000);
        session->detach(attachment);
    }
};

QTEST_MAIN(SpiralServiceProtocolTest)

#include "test_spiral_service_protocol.moc"
