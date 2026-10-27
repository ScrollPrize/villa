#include <QApplication>
#include <QComboBox>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QSignalSpy>
#include <QStatusBar>
#include <QStackedWidget>
#include <QTemporaryDir>
#include <QElapsedTimer>
#include <QFile>
#include <QDir>
#include <QThread>

#define private public
#include "segmentation/panels/SegmentationLasagnaPanel.hpp"
#undef private

#include "CState.hpp"
#include "LasagnaServiceManager.hpp"
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include "SurfaceCatalog.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/FiberJson.hpp>

#include <memory>
#include <filesystem>
#include <cstdlib>
#include <iostream>

extern QJsonObject g_lastLasagnaOptimizationRequest;

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        std::exit(1);
    }
}

void ensureApplication(int& argc, char** argv, std::unique_ptr<QApplication>& app)
{
    if (!QApplication::instance()) {
        app = std::make_unique<QApplication>(argc, argv);
    }
}

QJsonObject queuedJob(int position, const QString& outputName)
{
    QJsonObject job;
    job[QStringLiteral("job_id")] = QStringLiteral("job-%1").arg(position);
    job[QStringLiteral("state")] = QStringLiteral("waiting");
    job[QStringLiteral("queue_position")] = position;
    job[QStringLiteral("output_name")] = outputName;
    return job;
}

void writeFile(const QString& path, const QByteArray& bytes)
{
    QFile file(path);
    require(file.open(QIODevice::WriteOnly), "Failed to create temporary file");
    file.write(bytes);
}

QuadSurface wrappedSurface()
{
    cv::Mat_<cv::Vec3f> points(3, 4);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            const int wrappedCol = (col == points.cols - 1) ? 0 : col;
            points(row, col) = cv::Vec3f{
                static_cast<float>(wrappedCol),
                static_cast<float>(row),
                0.0f,
            };
        }
    }
    return QuadSurface(points, cv::Vec2f{1.0f, 1.0f});
}

nlohmann::json pinJson(const vc::objects::PinnedObject& pin)
{
    return {{"store_uuid", pin.object.storeUuid}, {"object_uuid", pin.object.objectUuid},
            {"object_revision", pin.objectRevision}, {"geometry_revision", pin.geometryRevision.value_or("")}};
}

} // namespace

int main(int argc, char** argv)
{
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }

    std::unique_ptr<QApplication> app;
    ensureApplication(argc, argv, app);

    QTemporaryDir tempDir;
    require(tempDir.isValid(), "Failed to create temporary settings dir");
    const QString configPath = tempDir.filePath(QStringLiteral("config.json"));
    QFile configFile(configPath);
    require(configFile.open(QIODevice::WriteOnly | QIODevice::Text),
            "Failed to create temporary Lasagna config");
    configFile.write(R"({"args":{"existing":1}})");
    configFile.close();

    const QString settingsPath = tempDir.filePath(QStringLiteral("settings.ini"));
    QSettings settings(settingsPath, QSettings::IniFormat);
    settings.setValue(QStringLiteral("lasagna_data_input_path"), QStringLiteral("  /data/input.lasagna.json  "));
    settings.setValue(QStringLiteral("lasagna_seed_point"), QStringLiteral("  10, 20, 30  "));
    settings.setValue(QStringLiteral("lasagna_output_name"), QStringLiteral("  sheet  "));
    settings.setValue(QStringLiteral("lasagna_new_model_width"), 123);
    settings.setValue(QStringLiteral("lasagna_new_model_height"), 456);
    settings.setValue(QStringLiteral("lasagna_new_model_windings"), 7);
    settings.setValue(QStringLiteral("lasagna_offset_value"), -1.25);
    const QString altConfigPath = tempDir.filePath(QStringLiteral("alt_config.json"));
    writeFile(altConfigPath, QByteArrayLiteral(R"({"args":{"alternate":1}})"));
    const QString reoptModelConfigPath = tempDir.filePath(QStringLiteral("reopt_model.json"));
    writeFile(reoptModelConfigPath, QByteArrayLiteral(R"({"args":{"model-init":"model"}})"));

    SegmentationLasagnaPanel panel(QStringLiteral("test-lasagna"));
    panel.restoreSettings(settings);

    require(panel.lasagnaDataInputPath() == QStringLiteral("  /data/input.lasagna.json  "),
            "restoreSettings should preserve the stored data input value");
    require(panel.seedPointText() == QStringLiteral("10, 20, 30"),
            "seedPointText should return trimmed seed text");
    require(panel.newModelOutputName() == QStringLiteral("sheet"),
            "newModelOutputName should return trimmed output name");
    require(panel.newModelWidth() == 123 && panel.newModelHeight() == 456 &&
                panel.newModelWindings() == 7,
            "New-model dimensions were not restored");
    require(panel.newModelWidthUnit() == QStringLiteral("voxels"),
            "New-model width unit should default to voxels");
    require(panel.offsetValue() == -1.25, "Offset value was not restored");

    settings.setValue(QStringLiteral("lasagna_new_model_config_file_path"), configPath);
    settings.setValue(QStringLiteral("lasagna_reopt_config_file_path"), configPath);
    settings.setValue(QStringLiteral("lasagna_offset_config_file_path"), configPath);
    panel.restoreSettings(settings);
    QWidget* compactView = panel.createCompactView();
    require(compactView != nullptr, "createCompactView should return a compact Lasagna widget");
    panel._lasagnaMode = SegmentationLasagnaPanel::LasagnaMode::NewModel;
    require(panel.selectedLasagnaConfigPathForMode(SegmentationLasagnaPanel::LasagnaMode::NewModel) == configPath,
            "Selected new-model config path should restore from settings");
    require(panel.lasagnaConfigPathsForMode(SegmentationLasagnaPanel::LasagnaMode::NewModel).contains(configPath),
            "Config path list should include restored combo path");
    require(panel.lasagnaConfigText().contains(QStringLiteral("existing")),
            "lasagnaConfigText should read the selected config file");
    require(panel.lasagnaConfigJson().is_object(),
            "lasagnaConfigJson should parse object configs");
    panel.setLasagnaDataInputPath(QStringLiteral("/data/other.lasagna.json"));
    require(panel.lasagnaDataInputPath() == QStringLiteral("/data/other.lasagna.json"),
            "setLasagnaDataInputPath should update the stored data input");
    panel.setLasagnaDataInputPath(QStringLiteral("  /data/input.lasagna.json  "));
    panel.setSeedFromFocus(7, 8, 9);
    require(panel.seedPointText() == QStringLiteral("7, 8, 9"),
            "setSeedFromFocus should update the seed editor");

    panel.populateConfigCombo(panel._newModelConfigCombo,
                              QFileInfo(configPath).absolutePath(),
                              QFileInfo(configPath).fileName(),
                              panel._newModelConfigFilePath);
    require(panel._compactNewModelConfigCombo->count() == panel._newModelConfigCombo->count(),
            "Compact new-model config combo should mirror the full combo");
    const int altIndex = panel._compactNewModelConfigCombo->findData(altConfigPath);
    require(altIndex >= 0, "Compact config combo should include configs from the full combo");
    panel._compactNewModelConfigCombo->setCurrentIndex(altIndex);
    require(panel.selectedLasagnaConfigPathForMode(SegmentationLasagnaPanel::LasagnaMode::NewModel) == altConfigPath,
            "Changing the compact config selector should update shared panel state");
    panel._newModelConfigCombo->setCurrentIndex(panel._newModelConfigCombo->findData(configPath));
    panel.syncCompactConfigCombos();
    require(panel._compactNewModelConfigCombo->currentData().toString() == configPath,
            "Changing the full config selector should update the compact selector");

    // The store jobs read: a volume's surfaces, one of them a Lasagna fit
    // (its model in the store, the reference surface it read in its meta).
    namespace vo = vc::objects;
    Volume::ZarrCreateOptions volumeOptions;
    volumeOptions.shapeZYX = {4, 4, 4};
    volumeOptions.chunkShapeZYX = {4, 4, 4};
    volumeOptions.compressor = "none";
    volumeOptions.uuid = "lasagna-test-volume";
    const auto volume = Volume::New(std::filesystem::path(tempDir.filePath(QStringLiteral("ct.zarr")).toStdString()),
                                    volumeOptions);
    const auto storePath = std::filesystem::path(tempDir.filePath(QStringLiteral("objects.vcobj")).toStdString());
    vo::OpenOptions storeOptions;
    storeOptions.author = "lasagna-panel-test";
    { auto created = vo::ObjectStore::create(storePath, storeOptions); }
    CState state;
    auto session = state.objectStoreSession();
    const auto attachment = session->attach(storePath, storeOptions);
    cv::Mat_<cv::Vec3f> points(2, 2);
    points.setTo(cv::Vec3f{1.0f, 1.0f, 1.0f});
    QuadSurface grid(points, cv::Vec2f{1.0f, 1.0f});
    auto addSurface = [&](const std::string& name, nlohmann::json meta) {
        const auto uuid = vo::util::newUuid();
        meta.update(nlohmann::json{{"coordinates", vc::objects::volumeCoordinates(volume->id(), 1)}});
        session->act(attachment, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
            vo::ObjectRecord record;
            record.uuid = uuid;
            record.kind = "surface";
            record.name = name;
            record.meta = meta.dump();
            db.upsertObject(txn, record);
            db.putGeometry(txn, uuid, quad_surface_geometry(grid));
        });
        return uuid;
    };
    const auto referenceUuid = addSurface("reference_surface", nlohmann::json::object());
    const auto fittedUuid = addSurface("sheet_v002", {{"provenance", {{"tool", "lasagna"},
        {"sources", nlohmann::json::array({referenceUuid})}}}});
    addSurface("sheet_off1", nlohmann::json::object());
    const auto noModelUuid = addSurface("sheet_no_model", nlohmann::json::object());
    writeFile(QString::fromStdString((session->artifactDirectory(attachment, fittedUuid) / "model.pt").string()),
              QByteArrayLiteral("model"));
    LasagnaServiceManager::instance().setObjectStore(session, attachment);
    SurfaceCatalog catalog(session);
    catalog.setSource(attachment, volume);
    QElapsedTimer catalogTimer;
    catalogTimer.start();
    while ((catalog.loading() || catalog.rows().size() < 4) && catalogTimer.elapsed() < 10000) {
        for (const auto& event : session->takeEvents()) {
            catalog.handle(event);
        }
        QCoreApplication::processEvents();
        QThread::msleep(1);
    }
    require(catalog.rows().size() == 4, "The catalog should list the volume's four surfaces");
    panel.setSurfaceCatalog(&catalog);

    emit LasagnaServiceManager::instance().jobsUpdated(QJsonArray{
        queuedJob(1, QStringLiteral("sheet_v001.tifxyz")),
        queuedJob(2, QString()),
    });
    require(!panel._progressLabel->isHidden(), "Queue label should become visible for waiting jobs");
    require(panel._progressLabel->text() == QStringLiteral("Queue: #1 sheet_v001.tifxyz, #2"),
            "Queue label should include output names and fall back to queue numbers");

    panel.syncUiState(true, false);
    require(panel._newModelBtn->isEnabled() && panel._reoptBtn->isEnabled() &&
                panel._offsetBtn->isEnabled(),
            "Lasagna action buttons should be enabled while editing is enabled");
    require(!panel._stopBtn->isEnabled(), "Stop button should be disabled while not optimizing");
    require(!panel._progressBar->isVisible(), "Progress bar should be hidden while idle");

    panel.syncUiState(false, true);
    require(!panel._newModelBtn->isEnabled() && !panel._reoptBtn->isEnabled() &&
                !panel._offsetBtn->isEnabled(),
            "Lasagna action buttons should be disabled while editing is disabled");
    require(panel._stopBtn->isEnabled(), "Stop button should be enabled while optimizing");

    QSignalSpy optimizeRequested(&panel, &SegmentationLasagnaPanel::lasagnaOptimizeRequested);
    panel._lasagnaMode = SegmentationLasagnaPanel::LasagnaMode::ReOptimize;
    panel._reoptConfigFilePath.clear();
    panel.triggerOptimization();
    require(panel._progressLabel->text().contains(QStringLiteral("No Lasagna config")),
            "triggerOptimization should report a missing config");
    panel._reoptConfigFilePath = tempDir.filePath(QStringLiteral("missing.json"));
    panel.triggerOptimization();
    require(panel._progressLabel->text().contains(QStringLiteral("config file not found")),
            "triggerOptimization should report a missing config file");
    panel._reoptConfigFilePath = configPath;
    panel.triggerOptimization();
    require(optimizeRequested.count() == 1,
            "triggerOptimization should emit optimize request for a valid internal config");
    panel.syncUiState(true, false);
    panel._compactReoptBtn->click();
    require(optimizeRequested.count() == 2,
            "Compact Re-optimize should emit the same optimize request");
    panel.repeatLastLasagnaAction();
    require(optimizeRequested.count() == 3,
            "repeatLastLasagnaAction should repeat the last selected Lasagna action");

    panel.onConnectionModeChanged(1);
    require(panel._externalWidget->isVisible() || !panel._externalWidget->isHidden(),
            "External connection mode should show external connection widgets");
    require(!panel._datasetCombo->isEnabled(),
            "External connection mode should disable dataset combo until datasets arrive");
    panel.refreshDiscoveredServices();
    require(panel._discoveryCombo->count() >= 1,
            "Refreshing discovered services should leave manual entry available");
    QJsonObject discovered;
    discovered[QStringLiteral("host")] = QStringLiteral("192.0.2.10");
    discovered[QStringLiteral("port")] = 12345;
    discovered[QStringLiteral("pid")] = 99;
    panel._discoveryCombo->addItem(
        QStringLiteral("192.0.2.10:12345"),
        QJsonDocument(discovered).toJson(QJsonDocument::Compact));
    panel._discoveryCombo->setCurrentIndex(panel._discoveryCombo->count() - 1);
    panel.onDiscoveredServiceSelected(panel._discoveryCombo->currentIndex());
    require(panel._hostEdit->text() == QStringLiteral("192.0.2.10") &&
                panel._portEdit->text() == QStringLiteral("12345"),
            "Selecting a discovered service should populate host and port");
    panel.onConnectionModeChanged(0);

    QJsonArray datasets;
    QJsonObject datasetA;
    datasetA[QStringLiteral("name")] = QStringLiteral("A");
    datasetA[QStringLiteral("path")] = QStringLiteral("/data/a.lasagna.json");
    QJsonObject datasetB;
    datasetB[QStringLiteral("name")] = QStringLiteral("B");
    datasetB[QStringLiteral("path")] = panel.lasagnaDataInputPath();
    datasets.append(datasetA);
    datasets.append(datasetB);
    emit LasagnaServiceManager::instance().datasetsReceived(datasets);
    require(panel._datasetCombo->isEnabled() && panel._dataInputStack->currentIndex() == 1,
            "Dataset signal should populate and show the dataset combo");

    emit LasagnaServiceManager::instance().serviceError(QStringLiteral("service failed"));
    require(panel._progressLabel->text().contains(QStringLiteral("service failed")),
            "Service error should update the progress label");

    QStatusBar statusBar;
    panel.startOptimizationAtSeed(
        nullptr,
        &statusBar,
        SegmentationLasagnaPanel::LasagnaMode::NewModel,
        QString(),
        1,
        2,
        3);
    require(statusBar.currentMessage().contains(QStringLiteral("No Lasagna config")),
            "startOptimizationAtSeed should reject an empty config path");
    panel.startOptimizationAtSeed(
        nullptr,
        &statusBar,
        SegmentationLasagnaPanel::LasagnaMode::NewModel,
        tempDir.filePath(QStringLiteral("also-missing.json")),
        1,
        2,
        3);
    require(statusBar.currentMessage().contains(QStringLiteral("config file not found")),
            "startOptimizationAtSeed should reject a missing config path");

    panel.syncUiState(true, false);
    panel.startOptimizationAtSeed(
        nullptr,
        &statusBar,
        SegmentationLasagnaPanel::LasagnaMode::NewModel,
        configPath,
        1,
        2,
        3);
    require(g_lastLasagnaOptimizationRequest[QStringLiteral("output_name")].toString() == QStringLiteral("sheet") &&
                g_lastLasagnaOptimizationRequest[QStringLiteral("output_name_mode")].toString() == QStringLiteral("version"),
            "New-model launch should ask the service for the next version of the root name");
    require(statusBar.currentMessage().contains(QStringLiteral("sheet")),
            "New-model launch status should name the output root");
    require(g_lastLasagnaOptimizationRequest[QStringLiteral("volume_uuid")].toString() ==
                QStringLiteral("lasagna-test-volume"),
            "Launches should name the volume their surfaces are in");

    emit LasagnaServiceManager::instance().statusMessage(QStringLiteral("preparing"));
    require(panel._progressLabel->text() == QStringLiteral("preparing"),
            "Status message signal should update the progress label");
    require(panel._compactProgressLabel->text() == QStringLiteral("preparing"),
            "Status message signal should update compact progress label");

    emit LasagnaServiceManager::instance().optimizationProgress(
        QStringLiteral("stage"), 1, 2, 0.125, 0.5, 0.75, QStringLiteral("fit"));
    require(panel._progressBar->isVisible() || !panel._progressBar->isHidden(),
            "Optimization progress should show the progress bar");
    require(panel._progressBar->value() == 750,
            "Optimization progress should map overall progress to the progress bar");
    require(panel._compactProgressBar->value() == 750,
            "Optimization progress should map overall progress to the compact progress bar");
    require(panel._progressLabel->text().contains(QStringLiteral("fit")),
            "Optimization progress should include the stage name");

    emit LasagnaServiceManager::instance().optimizationFinished(QStringList{QString::fromStdString(fittedUuid)});
    require(panel._progressLabel->text().contains(QStringLiteral("sheet_v002")),
            "Optimization finished signal should name the new surfaces");
    require(!panel._stopBtn->isEnabled(), "Stop button should be disabled after finish");

    emit LasagnaServiceManager::instance().optimizationError(QStringLiteral("failed"));
    require(panel._progressLabel->text().contains(QStringLiteral("failed")),
            "Optimization error signal should update the label");

    emit LasagnaServiceManager::instance().serviceStarted();
    require(panel._stopServiceBtn->isEnabled(),
            "Service-started signal should enable Stop Service");
    emit LasagnaServiceManager::instance().serviceStopped();
    require(!panel._stopServiceBtn->isEnabled(),
            "Service-stopped signal should disable Stop Service");

    const QString badConfigPath = tempDir.filePath(QStringLiteral("bad.json"));
    writeFile(badConfigPath, QByteArrayLiteral("[1,2,3]"));
    panel.startOptimizationAtSeed(
        nullptr,
        &statusBar,
        SegmentationLasagnaPanel::LasagnaMode::NewModel,
        badConfigPath,
        1,
        2,
        3);
    require(statusBar.currentMessage().contains(QStringLiteral("top-level")),
            "Invalid top-level config should report a config error");

    auto openSurface = [&](const std::string& uuid) {
        auto opened = std::make_shared<QuadSurface>(points, cv::Vec2f{1.0f, 1.0f});
        opened->id = uuid;
        return opened;
    };
    auto surface = openSurface(fittedUuid);

    state.setSurface("segmentation", surface, true);
    panel._lastLasagnaMode = SegmentationLasagnaPanel::LasagnaMode::NewModel;
    panel.setState(&state);
    require(panel.currentLinkedSurfaceNames() == QStringList{QStringLiteral("reference_surface")},
            "Linked surface preview should list the surfaces the selected fit read");
    panel.startOptimizationAtSeed(
        &state,
        &statusBar,
        SegmentationLasagnaPanel::LasagnaMode::NewModel,
        configPath,
        4,
        5,
        6);
    QJsonObject newModelJobSpec = g_lastLasagnaOptimizationRequest[QStringLiteral("job_spec")].toObject();
    QJsonObject newModelJobConfig = newModelJobSpec[QStringLiteral("config")].toObject();
    QJsonObject newModelJobArgs = newModelJobConfig[QStringLiteral("args")].toObject();
    require(newModelJobArgs[QStringLiteral("model-w")].toDouble() == 123.0,
            "New Model launch should preserve the saved voxel width value");
    require(newModelJobArgs[QStringLiteral("model-w-unit")].toString() == QStringLiteral("voxels"),
            "New Model launch should send the default voxel width unit");
    QJsonArray newModelExternalSurfaces =
        newModelJobConfig[QStringLiteral("external_surfaces")].toArray();
    require(newModelExternalSurfaces.size() == 1,
            "New Model launch should still send linked refs as external_surfaces");
    panel.startOptimizationAtSeed(
        &state,
        &statusBar,
        SegmentationLasagnaPanel::LasagnaMode::Offset,
        configPath,
        4,
        5,
        6);
    require(g_lastLasagnaOptimizationRequest[QStringLiteral("output_name")].toString() == QStringLiteral("sheet") &&
                g_lastLasagnaOptimizationRequest[QStringLiteral("output_name_mode")].toString() == QStringLiteral("offset"),
            "Offset launch should ask the service for the next free offset of the root name");
    require(statusBar.currentMessage().contains(QStringLiteral("offset of sheet")),
            "Offset launch status should name the output root");
    QJsonObject jobSpec = g_lastLasagnaOptimizationRequest[QStringLiteral("job_spec")].toObject();
    QJsonObject jobConfig = jobSpec[QStringLiteral("config")].toObject();
    require(!jobConfig.contains(QStringLiteral("offset_value")),
            "VC3D job config should not transport offset_value");
    QJsonArray externalSurfaces = jobConfig[QStringLiteral("external_surfaces")].toArray();
    require(externalSurfaces.size() == 1,
            "VC3D job config should include one external surface from the linked refs");
    QJsonObject externalSurface = externalSurfaces[0].toObject();
    require(externalSurface[QStringLiteral("type")].toString() == QStringLiteral("surface") &&
                externalSurface[QStringLiteral("uuid")].toString() == QString::fromStdString(referenceUuid),
            "External surface should be the stored reference surface");
    require(g_lastLasagnaOptimizationRequest[QStringLiteral("_objects")].toArray().isEmpty(),
            "Surface launches should upload nothing");
    require(externalSurface[QStringLiteral("offset")].toDouble() == panel.offsetValue(),
            "External surface should carry the offset spinner value");

    settings.setValue(QStringLiteral("lasagna_new_model_width_unit"), QStringLiteral("wraps"));
    settings.setValue(QStringLiteral("lasagna_new_model_width_wraps"), 1.5);
    panel.restoreSettings(settings);
    require(panel.newModelWidth() == 1.5 &&
                panel.newModelWidthUnit() == QStringLiteral("wraps"),
            "Wrap width settings were not restored");
    panel.startOptimizationAtSeed(
        &state,
        &statusBar,
        SegmentationLasagnaPanel::LasagnaMode::NewModel,
        configPath,
        4,
        5,
        6);
    QJsonObject wrapJobSpec = g_lastLasagnaOptimizationRequest[QStringLiteral("job_spec")].toObject();
    QJsonObject wrapJobConfig = wrapJobSpec[QStringLiteral("config")].toObject();
    QJsonObject wrapJobArgs = wrapJobConfig[QStringLiteral("args")].toObject();
    require(wrapJobArgs[QStringLiteral("model-w")].toDouble() == 1.5,
            "New Model launch should send the saved wraps width value");
    require(wrapJobArgs[QStringLiteral("model-w-unit")].toString() == QStringLiteral("wraps"),
            "New Model launch should send the wraps width unit");

    panel.startOptimizationAtSeed(
        &state,
        &statusBar,
        SegmentationLasagnaPanel::LasagnaMode::ReOptimize,
        reoptModelConfigPath,
        4,
        5,
        6);
    QJsonObject modelJobSpec = g_lastLasagnaOptimizationRequest[QStringLiteral("job_spec")].toObject();
    require(modelJobSpec[QStringLiteral("model")].toObject()[QStringLiteral("uuid")].toString() ==
                QString::fromStdString(fittedUuid),
            "Re-optimize should read the model the fit left with the selected surface");

    state.setSurface("segmentation", openSurface(noModelUuid), true);
    panel.startOptimizationAtSeed(
        &state,
        &statusBar,
        SegmentationLasagnaPanel::LasagnaMode::ReOptimize,
        reoptModelConfigPath,
        4,
        5,
        6);
    require(statusBar.currentMessage().contains(QStringLiteral("made by a Lasagna fit")),
            "Re-optimize should reject a surface without a Lasagna model");

    state.setSurface("segmentation", surface, true);

    QString headlessError;
    require(panel.startOptimizationHeadless(
                &state,
                SegmentationLasagnaPanel::LasagnaMode::NewModel,
                configPath,
                cv::Vec3i{4, 5, 6},
                &headlessError),
            "Headless New Model launch should submit");
    require(g_lastLasagnaOptimizationRequest[QStringLiteral("job_spec")].toObject()
                [QStringLiteral("config")].toObject()[QStringLiteral("args")].toObject()
                [QStringLiteral("seed")].toArray() == QJsonArray{4, 5, 6},
            "Headless New Model launch should send its seed");

    return 0;
}
