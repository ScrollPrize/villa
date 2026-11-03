#include "segmentation/panels/SegmentationGrowthPanel.hpp"
#include "segmentation/panels/SegmentationManualAddPanel.hpp"
#include "VCSettings.hpp"

#include <QApplication>
#include <QComboBox>
#include <QPushButton>
#include <QSettings>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTest>

Q_LOGGING_CATEGORY(lcSegWidget, "vc.segmentation.widget")

class ManualGrowthPanelTest : public QObject
{
    Q_OBJECT
    QTemporaryDir _configDir;
private slots:
    void init()
    {
        QVERIFY(_configDir.isValid());
        qputenv("VC3D_CONFIG_DIR", _configDir.path().toUtf8());
    }
    void cleanup() { qunsetenv("VC3D_CONFIG_DIR"); }
    void growthVolumeSelection()
    {
        SegmentationGrowthPanel growth("test");
        growth.setAvailableVolumes({{"ct", "CT"}, {"surface", "Surface prediction"}}, "ct");
        growth.setActiveVolume("ct");
        growth.setGrowthMethod(SegmentationGrowthMethod::ManualGrow);
        growth.show();
        auto* volumes = [&]() -> QComboBox* {
            for (auto* combo : growth.findChildren<QComboBox*>()) {
                if (combo->findData(QString("ct")) >= 0) return combo;
            }
            return nullptr;
        }();
        QVERIFY(volumes);
        QVERIFY(volumes->isVisible());
        QCOMPARE(volumes->count(), 2);
        QCOMPARE(volumes->currentData().toString(), QString("ct"));
        QSignalSpy selection(&growth, &SegmentationGrowthPanel::volumeSelectionChanged);
        volumes->setCurrentIndex(volumes->findData(QString("surface")));
        QCOMPARE(selection.count(), 1);
        QCOMPARE(selection.at(0).at(0).toString(), QString("surface"));
        growth.setGrowthInProgress(true);
        QVERIFY(!volumes->isEnabled());
        growth.setGrowthInProgress(false);
        QVERIFY(volumes->isEnabled());
        growth.setGrowthMethod(SegmentationGrowthMethod::ManualAdd);
        QVERIFY(!volumes->isVisible());
        growth.setGrowthMethod(SegmentationGrowthMethod::ManualGrow);
        QVERIFY(volumes->isVisible());
        QCOMPARE(volumes->currentData().toString(), QString("surface"));
        growth.setActiveVolume("ct");
        QCOMPARE(volumes->currentData().toString(), QString("ct"));
        QCOMPARE(selection.count(), 1); // Programmatic synchronization is silent.
        growth.setAvailableVolumes({}, {});
        QVERIFY(!volumes->isEnabled());
        QCOMPARE(volumes->count(), 0);
    }

    void splitMethods()
    {
        QTemporaryDir configDir;
        QVERIFY(configDir.isValid());
        qputenv("VC3D_CONFIG_DIR", configDir.path().toUtf8());
        SegmentationGrowthPanel growth("test");
        SegmentationManualAddPanel manual("test");
        connect(&growth, &SegmentationGrowthPanel::growthMethodChanged,
                &manual, [&](SegmentationGrowthMethod method) {
                    manual.setGrowMode(method == SegmentationGrowthMethod::ManualGrow);
                });
        QComboBox* methods = nullptr;
        for (auto* combo : growth.findChildren<QComboBox*>()) {
            if (combo->findText("Manual Fill") >= 0) methods = combo;
        }
        QVERIFY(methods);
        QVERIFY(methods->findText("Manual Grow") >= 0);
        QComboBox* fill = nullptr;
        QComboBox* walk = nullptr;
        for (auto* combo : manual.findChildren<QComboBox*>()) {
            if (combo->findText("Thin-plate spline") >= 0) fill = combo;
            if (combo->findText("Streamline") >= 0) walk = combo;
        }
        QVERIFY(fill);
        QVERIFY(walk);
        QPushButton* extendUp = nullptr;
        for (auto* button : manual.findChildren<QPushButton*>()) {
            if (button->text() == "Up") extendUp = button;
        }
        QVERIFY(extendUp);
        QSignalSpy extendRequested(&manual, &SegmentationManualAddPanel::extendRequested);
        growth.syncUiState(true, false);
        manual.syncUiState(true, false);
        manual.show();
        methods->setCurrentIndex(methods->findText("Manual Fill"));
        QCOMPARE(growth.growthMethod(), SegmentationGrowthMethod::ManualAdd);
        QVERIFY(fill->isVisible());
        QVERIFY(!walk->isVisible());
        QVERIFY(!extendUp->isVisible());
        methods->setCurrentIndex(methods->findText("Manual Grow"));
        QCOMPARE(growth.growthMethod(), SegmentationGrowthMethod::ManualGrow);
        QVERIFY(!fill->isVisible());
        QVERIFY(walk->isVisible());
        QVERIFY(extendUp->isVisible());
        extendUp->click();
        QCOMPARE(extendRequested.count(), 1);
        QCOMPARE(extendRequested.at(0).at(1).toInt(), manual.extendCells());
        manual.syncUiState(false, true);
        QVERIFY(!extendUp->isEnabled());
        manual.syncUiState(true, true);
        walk->setCurrentIndex(walk->findText("Streamline"));
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.beginGroup("test");
        SegmentationGrowthPanel restored("test");
        restored.restoreSettings(settings);
        QCOMPARE(restored.growthMethod(), SegmentationGrowthMethod::ManualGrow);
        methods->setCurrentIndex(methods->findText("Manual Fill"));
        QVERIFY(fill->isVisible());
        QVERIFY(!walk->isVisible());
        methods->setCurrentIndex(methods->findText("Manual Grow"));
        QCOMPARE(walk->currentText(), QString("Streamline"));
        QCOMPARE(segmentationGrowthMethodFromInt(4), SegmentationGrowthMethod::ManualAdd);
        QCOMPARE(segmentationGrowthMethodFromInt(5), SegmentationGrowthMethod::ManualGrow);
        QVERIFY(isManualGrowthMethod(SegmentationGrowthMethod::ManualGrow));
        QVERIFY(!isManualGrowthMethod(SegmentationGrowthMethod::Tracer));
        qunsetenv("VC3D_CONFIG_DIR");
    }
};
QTEST_MAIN(ManualGrowthPanelTest)
#include "test_manual_growth_panel.moc"
