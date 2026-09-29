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
        OpenDataManifest manifest;
        OpenDataSample sample;
        sample.id = "PHercParis4";
        OpenDataVolume volume;
        volume.id = "test-volume";
        volume.dataFormat = "zarr";
        volume.pixelSizeUm = 2.4;
        OpenDataArtifact raw, prediction;
        raw.type = "ome-zarr";
        raw.resolvedUrl = "https://example.invalid/source.zarr";
        prediction.type = "surface-prediction-zarr";
        prediction.resolvedUrl = "https://example.invalid/prediction.zarr";
        volume.artifacts = {raw, prediction};
        sample.volumes = {volume};
        OpenDataSegment segment;
        segment.id = "segment";
        segment.originalVolumeId = volume.id;
        OpenDataArtifact tifxyz;
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
            manifest = loadOpenDataManifestFile(qgetenv("VC_CATALOG_TEST_MANIFEST").toStdString());
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
            QCOMPARE(trees.size(), 3);
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
                QVERIFY(selection.segmentIds.has_value());
                QVERIFY(selection.segmentIds->empty());
                QCOMPARE(project.name, std::string("my-selection"));
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
                path->setText(temp.filePath("my-selection.volpkg.json"));
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
