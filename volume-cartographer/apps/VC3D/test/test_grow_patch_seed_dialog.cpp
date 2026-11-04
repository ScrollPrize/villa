#include "GrowPatchSeedDialog.hpp"
#include <QApplication>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QSpinBox>
#include <QTest>
#include <QTimer>

class TestGrowPatchSeedDialog : public QObject
{
    Q_OBJECT
private slots:
    void defaultsAndModes();
    void cancelPreservesParams();
};

void TestGrowPatchSeedDialog::defaultsAndModes()
{
    for (const QString mode : {QStringLiteral("default"), QStringLiteral("projection"), QStringLiteral("streamline")}) {
        GrowPatchSeedParams params;
        QVector<VolumeSelector::VolumeOption> volumes;
        VolumeSelector::VolumeOption volume;
        volume.id = "ct";
        volume.name = "CT";
        volume.path = "/test/ct.zarr";
        volumes.push_back(volume);
        QTimer::singleShot(0, [&]() {
            auto* dialog = qobject_cast<QDialog*>(QApplication::activeModalWidget());
            QVERIFY(dialog);
            auto* method = dialog->findChild<QComboBox*>("growthMethod");
            QCOMPARE(method->currentData().toString(), QString("default"));
            auto* spacing = dialog->findChild<QDoubleSpinBox*>("meshSpacing");
            auto* step = dialog->findChild<QDoubleSpinBox*>("integrationStep");
            auto* integrator = dialog->findChild<QComboBox*>("integrator");
            auto* workers = dialog->findChild<QSpinBox*>("workers");
            QVERIFY(!spacing->isEnabled());
            QVERIFY(!integrator->isEnabled());
            method->setCurrentIndex(method->findData(mode));
            QCOMPARE(spacing->isEnabled(), mode != "default");
            QCOMPARE(step->isEnabled(), mode != "default");
            QCOMPARE(workers->isEnabled(), mode != "default");
            QCOMPARE(integrator->isEnabled(), mode == "streamline");
            if (mode != "default") {
                auto* iterations = dialog->findChild<QSpinBox*>("iterations");
                QCOMPARE(iterations->value(), 20);
                method->setCurrentIndex(0);
                QCOMPARE(iterations->value(), 200);
                method->setCurrentIndex(method->findData(mode));
                QCOMPARE(iterations->value(), 20);
                spacing->setValue(5);
                step->setValue(0.5);
                workers->setValue(4);
                integrator->setCurrentIndex(integrator->findData("rk4"));
                dialog->findChild<QSpinBox*>("iterations")->setValue(3);
            }
            dialog->findChild<QDialogButtonBox*>()->button(QDialogButtonBox::Ok)->click();
        });
        QVERIFY(selectGrowPatchSeedParams(nullptr, volumes, params));
        QCOMPARE(params.growthMethod, mode);
        QCOMPARE(params.volumeId, QString("ct"));
        QCOMPARE(params.iterations, mode == "default" ? 200 : 3);
        QCOMPARE(params.meshSpacing, mode == "default" ? 20.0 : 5.0);
        QCOMPARE(params.integrationStep, mode == "default" ? 2.0 : 0.5);
        QCOMPARE(params.workers, mode == "default" ? 1 : 4);
        QCOMPARE(params.minAreaCm, 0.002);
    }
}

void TestGrowPatchSeedDialog::cancelPreservesParams()
{
    GrowPatchSeedParams params;
    QTimer::singleShot(0, []() {
        auto* dialog = qobject_cast<QDialog*>(QApplication::activeModalWidget());
        dialog->findChild<QComboBox*>("growthMethod")->setCurrentIndex(2);
        dialog->reject();
    });
    QVERIFY(!selectGrowPatchSeedParams(nullptr, {}, params));
    QCOMPARE(params.growthMethod, QString("default"));
}

QTEST_MAIN(TestGrowPatchSeedDialog)
#include "test_grow_patch_seed_dialog.moc"
