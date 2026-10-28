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

#include <QFile>
#include <QProcess>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QtTest/QtTest>

#include <cstring>

using namespace vc::objects;
using Json = nlohmann::json;

class SpiralServiceProtocolTest : public QObject
{
    Q_OBJECT

private slots:
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
};

QTEST_MAIN(SpiralServiceProtocolTest)

#include "test_spiral_service_protocol.moc"
