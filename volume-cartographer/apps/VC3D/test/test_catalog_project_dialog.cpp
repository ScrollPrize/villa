#include "OpenDataCatalogWindow.hpp"
#include "OpenDataSampleProject.hpp"
#include "support/TestProject.hpp"
#include "VCSettings.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <QApplication>
#include <QDialogButtonBox>
#include <QFile>
#include <QPushButton>
#include <QScopeGuard>
#include <QTemporaryDir>
#include <QDir>
#include <QSettings>
#include <QLineEdit>
#include <QTableWidget>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTreeWidget>
#include <QTimer>
#include <QtTest>

#include <filesystem>

namespace vc3d::opendata {

namespace {

class CatalogServer : public QTcpServer
{
public:
    CatalogServer()
    {
        connect(this, &QTcpServer::newConnection, this, [this] {
            while (auto* socket = nextPendingConnection()) {
                connect(socket, &QTcpSocket::disconnected, socket, &QObject::deleteLater);
                connect(socket, &QTcpSocket::readyRead, socket, [this, socket] {
                    if (!socket->peek(socket->bytesAvailable()).contains("\r\n\r\n"))
                        return;
                    socket->readAll();
                    const QByteArray body = sample.isEmpty() ? QByteArray{} :
                        "{\"samples\":{\"" + sample + "\":{\"sample\":{\"id\":\"" + sample + "\"}}}}";
                    socket->write((sample.isEmpty() ? "HTTP/1.1 404 Not Found" : "HTTP/1.1 200 OK") +
                                  QByteArray("\r\nConnection: close\r\nContent-Length: ") +
                                  QByteArray::number(body.size()) + "\r\n\r\n" + body);
                    socket->disconnectFromHost();
                });
            }
        });
    }

    std::string url() const
    {
        return "http://127.0.0.1:" + std::to_string(serverPort()) + "/catalog.json";
    }

    QByteArray sample{"Initial"};
};

} // namespace

class CatalogProjectDialogTest : public QObject
{
    Q_OBJECT
    // VC3D's INI and the AppData projects directory, for every slot.
    QTemporaryDir _config{QDir::current().filePath("catalog-dialog-config-XXXXXX")};

private slots:
    void initTestCase()
    {
        QVERIFY(_config.isValid());
        qputenv("VC3D_CONFIG_DIR", _config.path().toUtf8());
        qputenv("XDG_DATA_HOME", _config.filePath("data").toUtf8());
        qputenv("NO_PROXY", "127.0.0.1");
    }

    void inaccessibleCacheStillLoadsLiveCatalog()
    {
        const auto cacheDir = _config.filePath("inaccessible-cache");
        QVERIFY(QDir().mkpath(cacheDir));
        const auto permissions = QFile::permissions(cacheDir);
        QVERIFY(QFile::setPermissions(cacheDir, QFileDevice::Permissions{}));
        const auto restore = qScopeGuard([&] { QFile::setPermissions(cacheDir, permissions); });
        qputenv("VC_READER_CACHE_DIR", cacheDir.toUtf8());

        CatalogServer server;
        QVERIFY(server.listen(QHostAddress::LocalHost));
        std::error_code error;
        QVERIFY(!std::filesystem::is_regular_file(vcr::catalog::catalogCacheFile(server.url()), error));
        if (error != std::errc::permission_denied)
            QSKIP("This user or filesystem can traverse the cache directory.");

        OpenDataCatalogWindow catalog(server.url());
        catalog.setOpenSampleHandler([](const auto&) { return false; });
        auto* samples = catalog.findChild<QTableWidget*>("catalogSamples");
        auto* open = catalog.findChild<QPushButton*>("openCatalogSample");
        QVERIFY(samples && open);
        QTRY_COMPARE(samples->rowCount(), 1);
        QCOMPARE(samples->item(0, 0)->text(), QStringLiteral("Initial"));
        samples->setCurrentCell(0, 0);
        QVERIFY(!open->isEnabled());
    }

    void removedCacheDisablesProjectActions_data()
    {
        QTest::addColumn<bool>("liveRefreshSucceeds");
        QTest::newRow("live-download") << true;
        QTest::newRow("fetch-failure") << false;
    }

    void removedCacheDisablesProjectActions()
    {
        QFETCH(bool, liveRefreshSucceeds);
        QTemporaryDir temp;
        QVERIFY(temp.isValid());
        const auto cacheDir = temp.filePath("cache");
        qputenv("VC_READER_CACHE_DIR", cacheDir.toUtf8());

        CatalogServer server;
        QVERIFY(server.listen(QHostAddress::LocalHost));
        OpenDataCatalogWindow catalog(server.url());
        catalog.setOpenSampleHandler([](const auto&) { return false; });
        auto* samples = catalog.findChild<QTableWidget*>("catalogSamples");
        auto* refresh = catalog.findChild<QPushButton*>("refreshCatalog");
        auto* open = catalog.findChild<QPushButton*>("openCatalogSample");
        QVERIFY(samples && refresh && open);
        QTRY_COMPARE(samples->rowCount(), 1);
        QCOMPARE(samples->item(0, 0)->text(), QStringLiteral("Initial"));
        samples->setCurrentCell(0, 0);
        QVERIFY(open->isEnabled());

        QVERIFY(QDir(cacheDir).removeRecursively());
        QFile blockedCache(cacheDir);
        QVERIFY(blockedCache.open(QIODevice::WriteOnly));
        blockedCache.close();
        server.sample = liveRefreshSucceeds ? QByteArray("Live") : QByteArray{};
        refresh->click();
        QTRY_VERIFY(refresh->isEnabled());
        QCOMPARE(samples->rowCount(), 1);
        QCOMPARE(samples->item(0, 0)->text(), liveRefreshSucceeds ? QStringLiteral("Live") : QStringLiteral("Initial"));
        samples->setCurrentCell(0, 0);
        QVERIFY(!open->isEnabled());
    }

    void attachRepresentations()
    {
        vcr::catalog::OpenDataManifest manifest;
        vcr::catalog::OpenDataSample sample;
        sample.id = "test";
        vcr::catalog::OpenDataVolume volume;
        volume.id = "scan";
        const QStringList types{"ink-detection-3d-zarr", "ink_detection_3d_zarr",
            "surface-prediction-zarr", "lasagna", "normal-grids", "surface-prediction-mesh"};
        for (const auto& type : types) {
            vcr::catalog::OpenDataArtifact artifact;
            artifact.type = type.toStdString();
            artifact.resolvedUrl = "https://example.invalid/" + artifact.type;
            artifact.levelParameterPresent = true;
            artifact.sourceCoordinateLevel = 2;
            volume.artifacts.push_back(artifact);
        }
        auto unresolved = volume.artifacts.front();
        unresolved.resolvedUrl.clear();
        volume.artifacts.push_back(unresolved);
        auto invalidNormal = volume.artifacts[4];
        invalidNormal.levelParameterPresent = true;
        invalidNormal.sourceCoordinateLevel.reset();
        volume.artifacts.push_back(invalidNormal);
        sample.volumes.push_back(volume);
        manifest.samples.push_back(sample);
        OpenDataCatalogWindow catalog(manifest);
        std::vector<vcr::catalog::OpenDataRepresentationRef> attached;
        catalog.setAttachRepresentationHandler([&](const vcr::catalog::OpenDataSample& selected,
                const vcr::catalog::OpenDataRepresentationRef& ref) {
            QCOMPARE(selected.id, sample.id);
            QCOMPARE(ref.volumeIndex, std::size_t(0));
            attached.push_back(ref);
        });
        for (auto* table : catalog.findChildren<QTableWidget*>()) {
            if (table->horizontalHeaderItem(0)->text() == "Sample ID")
                table->setCurrentCell(0, 0);
        }
        auto* table = catalog.findChild<QTableWidget*>("catalogRepresentations");
        auto* button = catalog.findChild<QPushButton*>("attachCatalogRepresentation");
        QVERIFY(table);
        QVERIFY(button);
        QCOMPARE(table->rowCount(), 8);
        QSignalSpy volumes(&catalog, &OpenDataCatalogWindow::attachVolumeRequested);
        for (int row = 0; row < table->rowCount(); ++row) {
            table->setCurrentCell(row, 0);
            QCOMPARE(button->isEnabled(), row < 5);
            button->click();
        }
        QCOMPARE(volumes.count(), 0); // Representations must not take the URL-only path.
        // The three zarr predictions, the Lasagna and the normal grids, each
        // with its own artifact; not the mesh, the unresolved artifact or the
        // levelless normal grids.
        using Kind = vcr::catalog::OpenDataRepresentationKind;
        const std::vector<std::pair<std::size_t, Kind>> expected{
            {0, Kind::Prediction}, {1, Kind::Prediction}, {2, Kind::Prediction}, {3, Kind::Lasagna},
            {4, Kind::NormalGrids}};
        QCOMPARE(attached.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            QCOMPARE(attached[i].artifactIndex, expected[i].first);
            QCOMPARE(attached[i].kind, expected[i].second);
        }
    }

    void openAndCancel()
    {
        QTemporaryDir temp(QDir::current().filePath("catalog-dialog-XXXXXX"));
        QVERIFY(temp.isValid());
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue(vc3d::settings::viewer::REMOTE_CACHE_DIR, _config.filePath("cache"));
        settings.sync();
        vcr::catalog::OpenDataManifest manifest;
        vcr::catalog::OpenDataSample sample;
        sample.id = "PHercParis4";
        vcr::catalog::OpenDataVolume volume;
        volume.id = "test-volume";
        volume.dataFormat = "zarr";
        volume.pixelSizeUm = 2.4;
        vcr::catalog::OpenDataArtifact raw, prediction;
        raw.type = "ome-zarr";
        raw.resolvedUrl = "https://example.invalid/source.zarr";
        prediction.type = "surface-prediction-zarr";
        prediction.resolvedUrl = "https://example.invalid/prediction.zarr";
        volume.artifacts = {raw, prediction};
        sample.volumes = {volume};
        vcr::catalog::OpenDataSegment segment;
        segment.id = "segment";
        segment.originalVolumeId = volume.id;
        vcr::catalog::OpenDataArtifact tifxyz;
        tifxyz.type = "tifxyz";
        tifxyz.resolvedUrl = "https://example.invalid/segment";
        segment.artifacts = {tifxyz};
        sample.segments = {segment};
        auto first = sample;
        first.id = "A";
        first.volumes.front().id = "other-sample-volume";
        auto more = volume;
        more.id = "second-volume";
        more.artifacts[0].resolvedUrl = "https://example.invalid/second-source.zarr";
        more.artifacts[1].resolvedUrl = "https://example.invalid/second-prediction.zarr";
        sample.volumes.push_back(more);
        manifest.samples = {first, sample};
        if (qEnvironmentVariableIsSet("VC_CATALOG_TEST_MANIFEST")) {
            manifest = vcr::catalog::loadOpenDataManifestFile(qgetenv("VC_CATALOG_TEST_MANIFEST").toStdString());
            for (auto& entry : manifest.samples) entry.artifacts.clear();
        }
        OpenDataCatalogWindow catalog(manifest);
        bool created = false;
        catalog.setCreateProjectHandler([&](const auto&, const auto&, const auto&) {
            created = true;
            return true;
        });
        catalog.show();
        for (auto* table : catalog.findChildren<QTableWidget*>()) {
            if (table->horizontalHeaderItem(0)->text() == "Sample ID") {
                for (int row = 0; row < table->rowCount(); ++row) {
                    if (table->item(row, 0)->text() != "PHercParis4") continue;
                    table->scrollToItem(table->item(row, 0));
                    QTest::mouseClick(table->viewport(), Qt::LeftButton, Qt::NoModifier,
                                     table->visualItemRect(table->item(row, 0)).center());
                }
            }
        }
        for (auto* edit : catalog.findChildren<QLineEdit*>()) {
            if (edit->placeholderText() == "Filter sample ID") edit->setText("Paris4");
        }
        bool opened = false;
        QTimer::singleShot(0, &catalog, [&]() {
            auto* dialog = qobject_cast<QDialog*>(QApplication::activeModalWidget());
            QVERIFY(dialog);
            opened = true;
            QVERIFY(dialog->windowTitle().contains("PHercParis4"));
            const auto trees = dialog->findChildren<QTreeWidget*>();
            QCOMPARE(trees.size(), 2);  // volumes and representations; surfaces come from the sample store
            for (auto* tree : trees) {
                QVERIFY(tree->columnCount() >= 9);
                QVERIFY(tree->topLevelItemCount() > 0);
                QCOMPARE(tree->topLevelItem(0)->checkState(0), Qt::Unchecked);
                if (tree->headerItem()->text(0) == "Volume ID" &&
                    !qEnvironmentVariableIsSet("VC_CATALOG_TEST_MANIFEST")) {
                    QCOMPARE(tree->topLevelItem(0)->childCount(), 2);
                    QCOMPARE(tree->topLevelItem(0)->child(0)->text(0), QString("test-volume"));
                    QCOMPARE(tree->topLevelItem(0)->child(1)->text(0), QString("second-volume"));
                }
            }
            dialog->reject();
        });
        QVERIFY(QMetaObject::invokeMethod(&catalog, "createSelectedProject", Qt::DirectConnection));
        QVERIFY(opened);
        QVERIFY(!created);
        if (!qEnvironmentVariableIsSet("VC_CATALOG_TEST_MANIFEST")) {
            catalog.setCreateProjectHandler([&](const auto& selectedSample, const auto& selection,
                                                const auto& project) {
                const auto verify = [&]() {
                created = true;
                QCOMPARE(selectedSample.id, std::string("PHercParis4"));
                QVERIFY(selection.rawVolumeIds.has_value());
                QCOMPARE(selection.rawVolumeIds->size(), std::size_t{1});
                QCOMPARE(selection.rawVolumeIds->front(), std::string("test-volume"));
                QVERIFY(selection.representations.has_value());
                QVERIFY(selection.representations->empty());
                QCOMPARE(project.name, std::string("my-selection"));
                QCOMPARE(project.path.string(), temp.filePath("my-selection.VCPROJ").toStdString());
                vc::project::LoadOptions options;
                options.deferResolution = true;
                auto pkg = vc::test::emptyProject(options);
                attachOpenDataSampleVolumes(*pkg, selectedSample, &selection);
                QCOMPARE(pkg->volumeEntries().size(), std::size_t{1});
                QCOMPARE(pkg->volumeEntries().front().location, raw.resolvedUrl);
                };
                verify();
                return false;
            });
            QTimer::singleShot(0, &catalog, [&]() {
                auto* dialog = qobject_cast<QDialog*>(QApplication::activeModalWidget());
                QVERIFY(dialog);
                auto* path = dialog->findChild<QLineEdit*>("catalogProjectPath");
                QVERIFY(path);
                path->setText(temp.filePath("my-selection.VCPROJ"));
                auto trees = dialog->findChildren<QTreeWidget*>();
                for (auto* tree : trees) {
                    if (tree->topLevelItem(0)->text(0) == "Source volumes")
                        tree->topLevelItem(0)->child(0)->setCheckState(0, Qt::Checked);
                }
                dialog->findChild<QDialogButtonBox*>()->button(QDialogButtonBox::Save)->click();
            });
            QVERIFY(QMetaObject::invokeMethod(&catalog, "createSelectedProject", Qt::DirectConnection));
            QVERIFY(created);
        }
    }
};

} // namespace vc3d::opendata

QTEST_MAIN(vc3d::opendata::CatalogProjectDialogTest)
#include "test_catalog_project_dialog.moc"
