// VC3D's Spiral client against the real store-backed service (its resident
// runtime faked by spiral-fitting/tests/protocol_client_service.py): the
// session lease, the store binding, service-owned configuration, the input
// manifest artifact and pinned reads, frozen inputs and the reload rebuild,
// and committing a published preview to the store.
// Needs SPIRAL_TEST_PYTHON (a Spiral Python environment with vc_object_store
// on PYTHONPATH); skipped otherwise.
#include "SpiralServiceManager.hpp"
#include "SpiralStoreInputs.hpp"
#include "SpiralStoredFits.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/objects/SurfaceCreate.hpp>

#include <QElapsedTimer>
#include <QFile>
#include <QJsonArray>
#include <QProcess>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QDir>
#include <QScopeGuard>
#include <QtTest/QtTest>

#include <cstring>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
// One 2x2 surface in volume "vol", written through the session.
std::string addSurface(ObjectStoreSession& session, const AttachmentIdentity& attachment)
{
    const auto surface = util::newUuid();
    std::vector<float> points;
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 2; ++col) points.insert(points.end(), {float(col), float(row), 5.f});
    GeometryArray array{"points", "<f4", {2, 2, 3}, std::vector<uint8_t>(points.size() * 4)};
    std::memcpy(array.bytes.data(), points.data(), array.bytes.size());
    auto request = session.newRequest(attachment);
    request.expectedObjects = {{surface, std::nullopt}};
    request.operations = createSurfaceOperations(surface, "near", {}, volumeCoordinates("vol", 1),
                                                 SurfaceProvenance{"protocol-test"},
                                                 Json{{"source", {{"spiral_patch_erode_cells", 1}}}},
                                                 GeometryData{"surface", 1, "source_xyz", R"({"scale":[1.0,1.0]})", {array}});
    session.submit(attachment, request);
    return surface;
}

struct ServiceProcess {
    QProcess process;
    int port = 0;
    ~ServiceProcess() { process.kill(); process.waitForFinished(5000); }
};

bool startService(ServiceProcess& service, const QString& python, const QTemporaryDir& directory,
                  const QString& store)
{
    service.process.setProcessChannelMode(QProcess::ForwardedChannels);
    const QString portFile = directory.filePath(QStringLiteral("port"));
    QStringList arguments{QStringLiteral(SPIRAL_PROTOCOL_SERVICE_FIXTURE), QStringLiteral("--store"), store,
                          QStringLiteral("--root"), directory.filePath(QStringLiteral("service")),
                          QStringLiteral("--port-file"), portFile};
    service.process.start(python, arguments);
    if (!service.process.waitForStarted()) return false;
    QElapsedTimer timer; timer.start();
    while (!QFile::exists(portFile) && timer.elapsed() < 60000) QTest::qWait(50);
    QFile port(portFile);
    if (!port.open(QIODevice::ReadOnly)) return false;
    service.port = port.readAll().trimmed().toInt();
    return service.port > 0;
}

// What VC3D sends of a project of the objects' volume "vol": its voxel size,
// a catalog orientation (an outward sense of CW) and its umbilicus.
QJsonObject protocolProject(const std::optional<vc3d::spiral::ProjectLasagna>& = std::nullopt)
{
    const QJsonObject umbilicus{
        {QStringLiteral("uuid"), QStringLiteral("6c1f0e4e-6d7a-4d55-9d0c-2b8f9a0c1e11")},
        {QStringLiteral("version"), 1},
        {QStringLiteral("points_zyx"), QJsonArray{QJsonArray{0, 32, 32}, QJsonArray{64, 32, 32}}}};
    return {{QStringLiteral("name"), QStringLiteral("protocol")},
            {QStringLiteral("volumes"), QJsonObject{{QStringLiteral("vol"),
                QJsonObject{{QStringLiteral("voxel_size_um"), 9.6},
                            {QStringLiteral("z_direction_is_top_to_bottom"), false},
                            {QStringLiteral("left_handed_coordinates"), false},
                            {QStringLiteral("umbilicus"), umbilicus}}}}}};
}

SpiralServiceProfile protocolProfile(int port)
{
    SpiralServiceProfile profile;
    profile.id = QStringLiteral("protocol");
    profile.baseUrl = QUrl(QStringLiteral("http://127.0.0.1:%1").arg(port));
    profile.apiKey = QStringLiteral("key");
    return profile;
}

// A short run without dense losses.
QJsonObject protocolRequest(QJsonObject config = {})
{
    for (const auto& [key, value] : {std::pair{"dense_spacing_mode", QJsonValue("grad_mag")},
                                     {"loss_weight_dense_spacing", 0}, {"loss_weight_dense_normals", 0},
                                     {"loss_weight_shell_outer", 0}})
        config[QLatin1String(key)] = value;
    return {{"run", QJsonObject{{"z_begin", 0}, {"z_end", 10}, {"config", config}}}};
}

void useProtocolProject(SpiralServiceManager& manager)
{
    manager.setProjectSource([]() { return std::optional<vc3d::spiral::ProjectLasagna>(); },
                             [](const auto& lasagna) { return protocolProject(lasagna); });
}

// Sets the surface's spiral_patch_erode_cells, as a checked edit.
void editSurface(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& surface,
                 int cells)
{
    session.editRecords(attachment, {surface}, [&](ObjectRecord& record) {
        auto meta = Json::parse(record.meta);
        meta["source"] = {{"spiral_patch_erode_cells", cells}};
        record.meta = meta.dump();
    });
}

Json readManifest(const QSignalSpy& manifests, int index)
{
    QFile file(manifests.at(index).at(0).toString());
    if (!file.open(QIODevice::ReadOnly)) return Json();
    return Json::parse(file.readAll().toStdString());
}

int fittedCells(const Json& manifest)
{
    return manifest.at("manifest").at("objects").at(0).at("metadata").at("spiral_patch_erode_cells").get<int>();
}

// The update reported at index, which must have succeeded.
#define SUCCEEDED_UPDATE(spy, index, update)                                                                  \
    const auto update = (spy).at(index).at(0).value<SpiralServiceManager::InputUpdate>();                     \
    QVERIFY2(update.error.isEmpty(), qPrintable(update.error + update.problems.join(QStringLiteral("; "))))
}

class SpiralServiceProtocolTest : public QObject
{
    Q_OBJECT

private slots:
    void localLaunchWaitsForProjectStore()
    {
        const auto python = qEnvironmentVariable("SPIRAL_TEST_PYTHON");
        if (python.isEmpty()) QSKIP("Set SPIRAL_TEST_PYTHON to a Spiral Python environment");
        const auto oldPython = qgetenv("SPIRAL_PYTHON");
        const auto oldService = qgetenv("SPIRAL_SERVICE_PATH");
        const auto restore = qScopeGuard([&] {
            if (oldPython.isNull()) qunsetenv("SPIRAL_PYTHON");
            else qputenv("SPIRAL_PYTHON", oldPython);
            if (oldService.isNull()) qunsetenv("SPIRAL_SERVICE_PATH");
            else qputenv("SPIRAL_SERVICE_PATH", oldService);
        });
        qputenv("SPIRAL_PYTHON", python.toUtf8());
        const auto service = QFileInfo(QStringLiteral(SPIRAL_PROTOCOL_SERVICE_FIXTURE))
                                 .dir().absoluteFilePath(QStringLiteral("../spiral_service.py"));
        qputenv("SPIRAL_SERVICE_PATH", service.toUtf8());
        QTemporaryDir directory;
        const auto storePath = directory.filePath(QStringLiteral("project.vcobj")).toStdString();
        OpenOptions options; options.author = "test";
        { auto db = ObjectStore::create(storePath, options); }
        auto session = std::make_shared<ObjectStoreSession>();
        const auto attachment = session->attach(storePath, options);
        SpiralServiceManager manager;
        QSignalSpy errors(&manager, &SpiralServiceManager::errorOccurred);
        QSignalSpy states(&manager, &SpiralServiceManager::connectionStateChanged);
        auto profile = SpiralServiceProfile::localhostProfile();
        profile.outputRoot = directory.filePath(QStringLiteral("output"));
        profile.cacheRoot = directory.filePath(QStringLiteral("cache"));
        manager.setObjectStore(session, std::nullopt, QStringLiteral("Open a project to start Spiral."));
        manager.connectToService(profile);
        QVERIFY(errors.isEmpty());
        QVERIFY(!manager.ownsProcess());
        QCOMPARE(states.last().at(1).toString(), QStringLiteral("Open a project to start Spiral."));
        // Disconnect cancels the pending launch, including when the store arrives later.
        manager.disconnectFromService();
        manager.setObjectStore(session, attachment);
        QVERIFY(!manager.ownsProcess());
        manager.setObjectStore(session, std::nullopt);
        manager.connectToService(profile);
        manager.setObjectStore(session, attachment);
        QTRY_VERIFY_WITH_TIMEOUT(manager.isReady(), 30000);
        QVERIFY(manager.ownsProcess());
        QTRY_VERIFY_WITH_TIMEOUT(manager.storeBinding().sameFile, 10000);
        QVERIFY2(errors.isEmpty(), errors.isEmpty() ? "" : qPrintable(errors.first().first().toString()));
    }

    // A service on this machine shares VC3D's store file. The
    // session holds a lease and the service owns the store binding, so a
    // request naming another store is dropped. Saves are already in the
    // store; Update Fit Inputs names its head, the service links and pins a
    // position of the store itself, and the manifest it publishes reads back
    // here as fitted. The reload rebuild is accepted.
    void updateFitInputsOfASharedStore()
    {
        const auto python = qEnvironmentVariable("SPIRAL_TEST_PYTHON");
        if (python.isEmpty()) QSKIP("Set SPIRAL_TEST_PYTHON to a Spiral Python environment with vc_object_store");
        qRegisterMetaType<SpiralServiceManager::InputUpdate>();
        QTemporaryDir directory;
        const auto storePath = std::filesystem::path(directory.filePath(QStringLiteral("inputs.vcobj")).toStdString());
        OpenOptions options; options.author = "vc3d-test";
        { auto created = ObjectStore::create(storePath, options); }
        auto session = std::make_shared<ObjectStoreSession>();
        const auto attachment = session->attach(storePath, options);
        const auto surface = addSurface(*session, attachment);

        ServiceProcess service;
        QVERIFY(startService(service, python, directory, QString::fromStdString(storePath.string())));
        SpiralServiceManager manager;
        manager.setObjectStore(session, attachment);
        useProtocolProject(manager);
        QSignalSpy manifests(&manager, &SpiralServiceManager::inputManifestAvailable);
        QSignalSpy updates(&manager, &SpiralServiceManager::inputUpdateFinished);
        QSignalSpy progress(&manager, &SpiralServiceManager::inputUpdateProgress);
        QSignalSpy errors(&manager, &SpiralServiceManager::errorOccurred);
        manager.connectToService(protocolProfile(service.port));
        QTRY_VERIFY_WITH_TIMEOUT(manager.isReady() && manager.ownsSessionLease(), 30000);
        QVERIFY(!manager.sessionLeaseId().isEmpty());
        QTRY_VERIFY_WITH_TIMEOUT(manager.storeBinding().sameFile, 10000);
        QVERIFY(manager.storeBinding().matches() && !manager.storeBinding().serviceServed
                && manager.storeBinding().canUpdate());
        QTRY_VERIFY_WITH_TIMEOUT(manager.serviceOwnedKeys().contains(QStringLiteral("object_store_location")), 10000);

        const auto request = protocolRequest({{"object_store_location", "/elsewhere.vcobj"}});
        editSurface(*session, attachment, surface, 2);
        // A save receipt belongs to the store attached when it was saved:
        // one of another store is not sent with this store's update (the
        // service would refuse a receipt its store does not hold).
        const auto otherPath = std::filesystem::path(directory.filePath(QStringLiteral("other.vcobj")).toStdString());
        { auto created = ObjectStore::create(otherPath, options); }
        const auto other = session->attach(otherPath, options);
        manager.setObjectStore(session, other);
        manager.setLastSaveRequest(QStringLiteral("00000000-0000-4000-8000-000000000000"));
        manager.setObjectStore(session, attachment);
        QTRY_VERIFY_WITH_TIMEOUT(manager.storeBinding().sameFile, 10000);
        manager.updateFitInputs(request);
        // The update's one phase, "linking", is emitted as it begins: the
        // workspace snapshots what is saved then as the fit's inputs.
        QCOMPARE(progress.count(), 1);
        QCOMPARE(progress.at(0).at(0).toString(), QStringLiteral("linking"));
        QTRY_COMPARE_WITH_TIMEOUT(updates.count(), 1, 60000);
        SUCCEEDED_UPDATE(updates, 0, update);
        QVERIFY(update.requested > 0 && update.fitted >= update.requested);
        QTRY_COMPARE_WITH_TIMEOUT(manifests.count(), 1, 30000);
        const auto document = readManifest(manifests, 0);
        QCOMPARE(fittedCells(document), 2);
        QCOMPARE(QString::fromStdString(document.at("digest").get<std::string>()), manifests.at(0).at(1).toString());
        vc3d::spiral::FitInputs inputs;
        session->readStore(attachment, [&](const StoreReader& reader, const auto&) {
            inputs = vc3d::spiral::readFitInputs(reader, document);
        });
        QVERIFY(inputs.unchanged());
        QCOMPARE(inputs.surfaces.size(), std::size_t{1});

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

    // A service bound to a store file that this VC3D has not attached (S2):
    // the update names no position and the service links its store's head,
    // which holds what was saved on its host.
    void updateFitInputsWithoutAnAttachedStore()
    {
        const auto python = qEnvironmentVariable("SPIRAL_TEST_PYTHON");
        if (python.isEmpty()) QSKIP("Set SPIRAL_TEST_PYTHON to a Spiral Python environment with vc_object_store");
        qRegisterMetaType<SpiralServiceManager::InputUpdate>();
        QTemporaryDir directory;
        const auto storePath = std::filesystem::path(directory.filePath(QStringLiteral("inputs.vcobj")).toStdString());
        OpenOptions options; options.author = "service-host";
        { auto created = ObjectStore::create(storePath, options); }
        // A writer on the service's host, standing in for its local editors.
        auto host = std::make_shared<ObjectStoreSession>();
        const auto hostAttachment = host->attach(storePath, options);
        const auto surface = addSurface(*host, hostAttachment);

        ServiceProcess service;
        QVERIFY(startService(service, python, directory, QString::fromStdString(storePath.string())));
        SpiralServiceManager manager;
        useProtocolProject(manager);
        QSignalSpy manifests(&manager, &SpiralServiceManager::inputManifestAvailable);
        QSignalSpy updates(&manager, &SpiralServiceManager::inputUpdateFinished);
        QSignalSpy progress(&manager, &SpiralServiceManager::inputUpdateProgress);
        manager.connectToService(protocolProfile(service.port));
        QTRY_VERIFY_WITH_TIMEOUT(manager.isReady() && manager.ownsSessionLease(), 30000);
        QTRY_VERIFY_WITH_TIMEOUT(manager.storeBinding().advertised(), 10000);
        const auto& binding = manager.storeBinding();
        QVERIFY(binding.localToService() && !binding.matches() && !binding.sameFile);
        QVERIFY(binding.canUpdate() && !binding.canPosition());

        editSurface(*host, hostAttachment, surface, 3);
        manager.updateFitInputs(protocolRequest({{"object_store_location", "/elsewhere.vcobj"}}));
        QCOMPARE(progress.count(), 1);
        QTRY_COMPARE_WITH_TIMEOUT(updates.count(), 1, 60000);
        SUCCEEDED_UPDATE(updates, 0, update);
        // No position was named, so none is reported as reached and no
        // commits are described beyond it.
        QCOMPARE(update.requested, qint64{-1});
        QVERIFY(update.head >= 0 && update.fitted >= update.head);
        QVERIFY(update.includedBeyondRequested.isEmpty() && update.afterFitted.isEmpty());
        QTRY_COMPARE_WITH_TIMEOUT(manifests.count(), 1, 30000);
        QCOMPARE(fittedCells(readManifest(manifests, 0)), 3);

        bool released = false;
        manager.releaseSessionLease([&]() { released = true; });
        QTRY_VERIFY_WITH_TIMEOUT(released, 30000);
    }

    // A service sharing VC3D's store file commits the displayed preview in
    // place; VC3D reads it back as a committed fit, and committing it again
    // reuses it.
    void commitAPreviewToASharedStore()
    {
        try { commitAPreviewToASharedStoreBody(); }
        catch (const std::exception& error) { QFAIL(error.what()); }
    }

private:
    static QJsonObject fitRequest()
    {
        const QJsonObject config{{"dense_spacing_mode", "grad_mag"}, {"loss_weight_dense_spacing", 0},
                                 {"loss_weight_dense_normals", 0}, {"loss_weight_shell_outer", 0}};
        return {{"run", QJsonObject{{"z_begin", 0}, {"z_end", 10}, {"config", config}}}};
    }

    // Ask for a preview and wait until it is displayed (fetched).
    static void displayPreview(SpiralServiceManager& manager)
    {
        QSignalSpy previews(&manager, &SpiralServiceManager::previewAvailable);
        manager.requestPreview();
        QTRY_VERIFY_WITH_TIMEOUT(previews.count() >= 1, 30000);
        QVERIFY(!manager.displayedPreviewArtifact().isEmpty());
    }

    static SpiralServiceManager::CommitResult commit(SpiralServiceManager& manager, const QString& artifact = {})
    {
        QSignalSpy commits(&manager, &SpiralServiceManager::commitFinished);
        manager.commitPreview(artifact);
        if (!QTest::qWaitFor([&]() { return commits.count() == 1; }, 30000))
            throw std::runtime_error("the commit did not finish");
        return commits.at(0).at(0).value<SpiralServiceManager::CommitResult>();
    }

    void commitAPreviewToASharedStoreBody()
    {
        const auto python = qEnvironmentVariable("SPIRAL_TEST_PYTHON");
        if (python.isEmpty()) QSKIP("Set SPIRAL_TEST_PYTHON to a Spiral Python environment with vc_object_store");
        qRegisterMetaType<SpiralServiceManager::CommitResult>();
        QTemporaryDir directory;
        const auto storePath = std::filesystem::path(directory.filePath(QStringLiteral("inputs.vcobj")).toStdString());
        OpenOptions options; options.author = "vc3d-test";
        { auto created = ObjectStore::create(storePath, options); }
        auto session = std::make_shared<ObjectStoreSession>();
        const auto attachment = session->attach(storePath, options);
        addSurface(*session, attachment);

        ServiceProcess service;
        QVERIFY(startService(service, python, directory, QString::fromStdString(storePath.string())));
        SpiralServiceManager manager;
        manager.setObjectStore(session, attachment);
        manager.setProjectSource([]() { return std::optional<vc3d::spiral::ProjectLasagna>(); },
                                 [](const auto& lasagna) { return protocolProject(lasagna); });
        manager.connectToService(protocolProfile(service.port));
        QTRY_VERIFY_WITH_TIMEOUT(manager.isReady() && manager.ownsSessionLease(), 30000);
        QTRY_VERIFY_WITH_TIMEOUT(manager.storeBinding().sameFile && manager.commitsAvailable(), 10000);
        manager.initializeSession(fitRequest());
        QTRY_VERIFY_WITH_TIMEOUT(manager.hasActiveSession(), 30000);
        displayPreview(manager);

        const auto first = commit(manager);
        QVERIFY2(first.state == QStringLiteral("done"), qPrintable(first.error));
        QVERIFY(first.version > 0 && !first.reused && !first.surfaceUuid.isEmpty());
        QCOMPARE(first.artifact, manager.displayedPreviewArtifact());
        const auto fit = vc3d::spiral::readStoredFit(session, attachment, first.surfaceUuid.toStdString());
        QCOMPARE(QString::fromStdString(fit.geometryRevision), first.geometryRevision);
        QCOMPARE(fit.volume, std::string("vol"));
        QCOMPARE(fit.components.size(), std::size_t{2});
        QCOMPARE(fit.components[0].winding, 4);
        QCOMPARE(fit.components[1].winding, 5);
        QCOMPARE(vc3d::spiral::listStoredFits(*session, attachment).size(), std::size_t{1});

        const auto again = commit(manager);
        QVERIFY2(again.state == QStringLiteral("done"), qPrintable(again.error));
        QVERIFY(again.reused);
        QCOMPARE(again.surfaceUuid, first.surfaceUuid);
        QCOMPARE(vc3d::spiral::listStoredFits(*session, attachment).size(), std::size_t{1});

        bool released = false;
        manager.releaseSessionLease([&]() { released = true; });
        QTRY_VERIFY_WITH_TIMEOUT(released, 30000);
    }
};

QTEST_MAIN(SpiralServiceProtocolTest)

#include "test_spiral_service_protocol.moc"
