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
#include "vc/core/util/QuadSurface.hpp"

#include <memory>
#include <filesystem>
#include <cstdlib>
#include <iostream>

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
    settings.setValue(QStringLiteral("lasagna_window_size"), 3210);
    settings.setValue(QStringLiteral("lasagna_window_overlap"), 210);
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
    require(panel.offsetValue() == -1.25, "Offset value was not restored");
    require(panel.windowSize() == 3210 && panel.windowOverlap() == 210,
            "Offset window settings were not restored");

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
    require(panel._progressLabel->text().contains(QStringLiteral("No config")),
            "triggerOptimization should report a missing config");
    panel._reoptConfigFilePath = tempDir.filePath(QStringLiteral("missing.json"));
    panel.triggerOptimization();
    require(panel._progressLabel->text().contains(QStringLiteral("Config file not found")),
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
    require(statusBar.currentMessage().contains(QStringLiteral("Config file not found")),
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
    writeFile(segDir + QStringLiteral("/meta.json"), QByteArrayLiteral("{}"));
    writeFile(segDir + QStringLiteral("/approval.tif"), QByteArrayLiteral("a"));
    writeFile(segDir + QStringLiteral("/d.tif"), QByteArrayLiteral("d"));
    writeFile(segDir + QStringLiteral("/model.pt"), QByteArrayLiteral("model"));
    require(QDir().mkpath(tempDir.filePath(QStringLiteral("sheet_off1_w000.tifxyz"))),
            "Failed to create offset collision directory");

    cv::Mat_<cv::Vec3f> points(2, 2);
    points.setTo(cv::Vec3f{0.0f, 0.0f, 0.0f});
    auto surface = std::make_shared<QuadSurface>(points, cv::Vec2f{1.0f, 1.0f});
    surface->path = std::filesystem::path(segDir.toStdString());

    CState state(0);
    state.setSurface("segmentation", surface, true);
    panel.startOptimizationAtSeed(
        &state,
        &statusBar,
        SegmentationLasagnaPanel::LasagnaMode::Offset,
        configPath,
        4,
        5,
        6);
    require(panel._submittedOutputNames.contains(QStringLiteral("sheet_off2")),
            "Offset launch should reserve the next collision-free offset output name");
    require(statusBar.currentMessage().contains(QStringLiteral("sheet_off2")),
            "Offset launch status should include the generated offset output name");

    return 0;
}
