#include <QApplication>
#include <QComboBox>
#include <QJsonArray>
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
#include <QFile>
#include <QDir>

#define private public
#include "segmentation/panels/SegmentationLasagnaPanel.hpp"
#undef private

#include "CState.hpp"
#include "LasagnaServiceManager.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

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

void writeWrappedTifxyz(const QString& path)
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
    QuadSurface surface(points, cv::Vec2f{1.0f, 1.0f});
    surface.save(path.toStdString(), true);
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
    require(panel._submittedOutputNames.contains(QStringLiteral("sheet_v001.tifxyz")),
            "New-model launch should reserve the generated output name");
    require(statusBar.currentMessage().contains(QStringLiteral("sheet_v001.tifxyz")),
            "New-model launch status should include the generated output name");

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

    emit LasagnaServiceManager::instance().optimizationFinished(QStringLiteral("/tmp/out"));
    require(panel._progressLabel->text().contains(QStringLiteral("/tmp/out")),
            "Optimization finished signal should show output path");
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

    const QString segDir = tempDir.filePath(QStringLiteral("sheet_v002.tifxyz"));
    require(QDir().mkpath(segDir), "Failed to create temporary tifxyz segment");
    writeFile(segDir + QStringLiteral("/x.tif"), QByteArrayLiteral("x"));
    writeFile(segDir + QStringLiteral("/y.tif"), QByteArrayLiteral("y"));
    writeFile(segDir + QStringLiteral("/z.tif"), QByteArrayLiteral("z"));
    writeFile(segDir + QStringLiteral("/meta.json"),
              QByteArrayLiteral(R"({"lasagna_job":{"linked_surfaces":[{"type":"tifxyz_segment","name":"reference_surface.tifxyz","hash":"md5:11111111111111111111111111111111"}]}})"));
    writeFile(segDir + QStringLiteral("/approval.tif"), QByteArrayLiteral("a"));
    writeFile(segDir + QStringLiteral("/d.tif"), QByteArrayLiteral("d"));
    writeFile(segDir + QStringLiteral("/model.pt"), QByteArrayLiteral("model"));
    require(QDir().mkpath(tempDir.filePath(QStringLiteral("sheet_off1.tifxyz"))),
            "Failed to create offset collision directory");

    cv::Mat_<cv::Vec3f> points(2, 2);
    points.setTo(cv::Vec3f{0.0f, 0.0f, 0.0f});
    auto surface = std::make_shared<QuadSurface>(points, cv::Vec2f{1.0f, 1.0f});
    surface->path = std::filesystem::path(segDir.toStdString());

    CState state(0);
    state.setSurface("segmentation", surface, true);
    panel._lastLasagnaMode = SegmentationLasagnaPanel::LasagnaMode::NewModel;
    panel.setState(&state);
    require(panel.currentLinkedSurfaceNames().contains(QStringLiteral("reference_surface.tifxyz")),
            "Linked surface preview should load names from the selected segment meta.json");
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
    require(panel._submittedOutputNames.contains(QStringLiteral("sheet_off2.tifxyz")),
            "Offset launch should reserve the next collision-free offset output name");
    require(statusBar.currentMessage().contains(QStringLiteral("sheet_off2.tifxyz")),
            "Offset launch status should include the generated offset output name");
    QJsonObject jobSpec = g_lastLasagnaOptimizationRequest[QStringLiteral("job_spec")].toObject();
    QJsonObject jobConfig = jobSpec[QStringLiteral("config")].toObject();
    require(!jobConfig.contains(QStringLiteral("offset_value")),
            "VC3D job config should not transport offset_value");
    QJsonArray externalSurfaces = jobConfig[QStringLiteral("external_surfaces")].toArray();
    require(externalSurfaces.size() == 1,
            "VC3D job config should include one external surface from the linked refs");
    QJsonObject externalSurface = externalSurfaces[0].toObject();
    require(externalSurface[QStringLiteral("type")].toString() == QStringLiteral("tifxyz_segment"),
            "External surface should preserve object ref type");
    require(externalSurface[QStringLiteral("name")].toString() == QStringLiteral("reference_surface.tifxyz"),
            "External surface should preserve object ref name");
    require(externalSurface[QStringLiteral("hash")].toString() ==
                QStringLiteral("md5:11111111111111111111111111111111"),
            "External surface should preserve object ref hash");
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

    const QString volpkgRoot = tempDir.filePath(QStringLiteral("sample.volpkg"));
    require(QDir().mkpath(volpkgRoot + QStringLiteral("/atlases/fiber_atlas/base_mesh/base.tifxyz")),
            "Failed to create atlas base directory");
    require(QDir().mkpath(volpkgRoot + QStringLiteral("/atlases/fiber_atlas/mappings/fibers")),
            "Failed to create atlas mappings directory");
    require(QDir().mkpath(volpkgRoot + QStringLiteral("/fibers")),
            "Failed to create atlas fibers directory");
    writeWrappedTifxyz(volpkgRoot + QStringLiteral("/atlases/fiber_atlas/base_mesh/base.tifxyz"));
    writeFile(volpkgRoot + QStringLiteral("/fibers/fiber.json"),
              QByteArrayLiteral(R"({"type":"vc3d_fiber","version":1,"line_points":[[10,20,30]],"control_points":[]})"));
    writeFile(volpkgRoot + QStringLiteral("/atlases/fiber_atlas/mappings/fibers/fiber.json"),
              QByteArrayLiteral(R"({"type":"vc3d_atlas_fiber_mapping","version":2,"fiber_path":"fibers/fiber.json","winding_offset":2,"line_anchors":[{"source_index":0,"world":[1,1,0],"atlas":[1,1],"distance":0}]})"));
    writeFile(volpkgRoot + QStringLiteral("/atlases/fiber_atlas/metadata.json"),
              QByteArrayLiteral(R"({"type":"vc3d_atlas","version":3,"name":"fiber_atlas","base_mesh_path":"base_mesh/base.tifxyz","zero_winding_column":1})"));

    const QString projectPath = volpkgRoot + QStringLiteral("/project.volpkg");
    writeFile(projectPath, QByteArrayLiteral(R"({"name":"atlas-test","version":1})"));
    auto vpkg = VolumePkg::New(projectPath.toStdString());
    require(panel._atlasCombo->count() == 0,
            "Atlas combo should be empty before a volume package is loaded");
    state.setVpkg(vpkg);
    require(panel._atlasCombo->count() == 1,
            "Atlas combo should refresh when the state's volume package changes");
    require(panel._atlasCombo->currentData().toString().endsWith(QStringLiteral("atlases/fiber_atlas")),
            "Atlas combo should select the discovered atlas directory");
    require(panel._compactAtlasCombo->count() == panel._atlasCombo->count(),
            "Compact atlas combo should mirror the full atlas combo");

    panel._atlasConfigFilePath = configPath;
    panel.populateConfigCombo(panel._atlasConfigCombo,
                              QFileInfo(configPath).absolutePath(),
                              QFileInfo(configPath).fileName(),
                              panel._atlasConfigFilePath);
    require(panel._compactAtlasConfigCombo->count() == panel._atlasConfigCombo->count(),
            "Compact atlas config combo should mirror the full atlas config combo");
    panel.setLasagnaDataInputPath(QStringLiteral("/data/atlas_input.lasagna.json"));
    panel.startAtlasOptimization(&state, &statusBar);
    QJsonObject atlasRequest = g_lastLasagnaOptimizationRequest;
    require(atlasRequest[QStringLiteral("data_input")].toString() == QStringLiteral("/data/atlas_input.lasagna.json"),
            "Atlas launch should send the selected data_input");
    require(!atlasRequest.contains(QStringLiteral("single_segment")) &&
                !atlasRequest.contains(QStringLiteral("copy_model")) &&
                !atlasRequest.contains(QStringLiteral("model_input")),
            "Atlas launch should not send segmentation/model launch flags");
    QJsonObject atlasJobSpec = atlasRequest[QStringLiteral("job_spec")].toObject();
    require(atlasJobSpec.contains(QStringLiteral("atlas")),
            "Atlas launch should send job_spec.atlas");
    require(!atlasJobSpec.contains(QStringLiteral("model")) &&
                !atlasJobSpec.contains(QStringLiteral("linked_surfaces")),
            "Atlas launch should not send model or linked surface refs");
    QJsonArray atlasUploads = atlasRequest[QStringLiteral("_objects")].toArray();
    require(atlasUploads.size() == 4,
            "Atlas launch should upload base, line, line-map, and compact atlas objects");
    bool sawBase = false;
    bool sawLine = false;
    bool sawMap = false;
    bool sawAtlas = false;
    QJsonObject compactAtlas;
    for (const QJsonValue& value : atlasUploads) {
        QJsonObject upload = value.toObject();
        QJsonObject ref = upload[QStringLiteral("object")].toObject();
        const QString type = ref[QStringLiteral("type")].toString();
        sawBase = sawBase || type == QStringLiteral("atlas-base");
        sawLine = sawLine || type == QStringLiteral("line");
        sawMap = sawMap || type == QStringLiteral("line-map");
        if (type == QStringLiteral("atlas")) {
            sawAtlas = true;
            QByteArray data = QByteArray::fromBase64(upload[QStringLiteral("data")].toString().toLatin1());
            compactAtlas = QJsonDocument::fromJson(data).object();
        }
        require(type != QStringLiteral("tifxyz_segment") && type != QStringLiteral("lasagna_model"),
                "Atlas launch should not upload selected segmentation or model objects");
    }
    require(sawBase && sawLine && sawMap && sawAtlas,
            "Atlas launch should include the expected atlas object types");
    QJsonObject compactMetadata = compactAtlas[QStringLiteral("metadata")].toObject();
    require(compactMetadata[QStringLiteral("zero_winding_column")].toInt() == 1,
            "Compact atlas JSON should include zero_winding_column");
    require(!compactMetadata.contains(QStringLiteral("period_columns")) &&
                !compactMetadata.contains(QStringLiteral("u_offset_columns")),
            "Compact atlas JSON should omit period_columns and u_offset_columns");
    QJsonObject compactMap = compactAtlas[QStringLiteral("maps")].toArray()[0].toObject();
    require(compactMap[QStringLiteral("winding_offset")].toInt() == 2,
            "Compact atlas JSON should include each map winding_offset");

    panel._atlasDirPath = volpkgRoot + QStringLiteral("/atlases/missing_atlas");
    panel.startAtlasOptimization(&state, &statusBar);
    require(statusBar.currentMessage().contains(QStringLiteral("Atlas directory not found")),
            "Atlas launch should fail clearly for a missing atlas directory");

    const QString missingBaseAtlas = volpkgRoot + QStringLiteral("/atlases/missing_base");
    require(QDir().mkpath(missingBaseAtlas + QStringLiteral("/mappings/fibers")),
            "Failed to create missing-base atlas directory");
    writeFile(missingBaseAtlas + QStringLiteral("/metadata.json"),
              QByteArrayLiteral(R"({"type":"vc3d_atlas","version":3,"name":"missing_base","base_mesh_path":"base_mesh/missing.tifxyz","zero_winding_column":0})"));
    writeFile(missingBaseAtlas + QStringLiteral("/mappings/fibers/fiber.json"),
              QByteArrayLiteral(R"({"type":"vc3d_atlas_fiber_mapping","version":2,"fiber_path":"fibers/fiber.json","winding_offset":0,"line_anchors":[]})"));
    panel._atlasDirPath = missingBaseAtlas;
    panel.startAtlasOptimization(&state, &statusBar);
    require(statusBar.currentMessage().contains(QStringLiteral("base mesh does not exist")),
            "Atlas launch should fail clearly for a missing base mesh");

    const QString missingFiberAtlas = volpkgRoot + QStringLiteral("/atlases/missing_fiber");
    require(QDir().mkpath(missingFiberAtlas + QStringLiteral("/base_mesh/base.tifxyz")),
            "Failed to create missing-fiber base directory");
    require(QDir().mkpath(missingFiberAtlas + QStringLiteral("/mappings/fibers")),
            "Failed to create missing-fiber mappings directory");
    writeWrappedTifxyz(missingFiberAtlas + QStringLiteral("/base_mesh/base.tifxyz"));
    writeFile(missingFiberAtlas + QStringLiteral("/metadata.json"),
              QByteArrayLiteral(R"({"type":"vc3d_atlas","version":3,"name":"missing_fiber","base_mesh_path":"base_mesh/base.tifxyz","zero_winding_column":0})"));
    writeFile(missingFiberAtlas + QStringLiteral("/mappings/fibers/fiber.json"),
              QByteArrayLiteral(R"({"type":"vc3d_atlas_fiber_mapping","version":2,"fiber_path":"fibers/does_not_exist.json","winding_offset":0,"line_anchors":[]})"));
    panel._atlasDirPath = missingFiberAtlas;
    panel.startAtlasOptimization(&state, &statusBar);
    require(statusBar.currentMessage().contains(QStringLiteral("references missing fiber path")),
            "Atlas launch should fail clearly for a missing fiber JSON");

    const QString missingMapAtlas = volpkgRoot + QStringLiteral("/atlases/missing_map");
    require(QDir().mkpath(missingMapAtlas + QStringLiteral("/base_mesh/base.tifxyz")),
            "Failed to create missing-map base directory");
    writeWrappedTifxyz(missingMapAtlas + QStringLiteral("/base_mesh/base.tifxyz"));
    writeFile(missingMapAtlas + QStringLiteral("/metadata.json"),
              QByteArrayLiteral(R"({"type":"vc3d_atlas","version":3,"name":"missing_map","base_mesh_path":"base_mesh/base.tifxyz","zero_winding_column":0})"));
    panel._atlasDirPath = missingMapAtlas;
    panel.startAtlasOptimization(&state, &statusBar);
    require(statusBar.currentMessage().contains(QStringLiteral("no fiber mappings directory")),
            "Atlas launch should fail clearly for missing mapping files");

    return 0;
}
