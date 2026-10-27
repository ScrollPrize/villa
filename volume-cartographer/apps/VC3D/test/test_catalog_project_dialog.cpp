#include "OpenDataCatalogWindow.hpp"
#include "OpenDataSampleProject.hpp"
#include "VCSettings.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <QApplication>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QTemporaryDir>
#include <QDir>
#include <QSettings>
#include <QLineEdit>
#include <QTableWidget>
#include <QTreeWidget>
#include <QTimer>
#include <QtTest>

using namespace vc3d::opendata;

class CatalogProjectDialogTest : public QObject
{
    Q_OBJECT
private slots:
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
        const auto oldConfig = qgetenv("VC3D_CONFIG_DIR");
        qputenv("VC3D_CONFIG_DIR", temp.path().toUtf8());
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue(vc3d::settings::viewer::REMOTE_CACHE_DIR, temp.filePath("cache"));
        settings.sync();
        const auto oldRoot = VolumePkg::autosaveRoot();
        VolumePkg::setAutosaveRoot(temp.path().toStdString());
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
                QCOMPARE(project.path.string(), temp.filePath("my-selection.VOLPKG.JSON").toStdString());
                vc::project::LoadOptions options;
                options.deferResolution = true;
                auto pkg = VolumePkg::newDetached(options);
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
                path->setText(temp.filePath("my-selection.VOLPKG.JSON"));
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
        VolumePkg::setAutosaveRoot(oldRoot);
        if (oldConfig.isNull()) qunsetenv("VC3D_CONFIG_DIR");
        else qputenv("VC3D_CONFIG_DIR", oldConfig);
    }
};

QTEST_MAIN(CatalogProjectDialogTest)
#include "test_catalog_project_dialog.moc"
