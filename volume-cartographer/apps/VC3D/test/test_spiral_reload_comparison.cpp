#include "SpiralReloadComparison.hpp"

#include <QJsonObject>
#include <QtTest/QtTest>

class SpiralReloadComparisonTest final : public QObject
{
    Q_OBJECT

private slots:
    void runMutableConfigAndShellPathDoNotRequireFullReload()
    {
        const QJsonObject defaults{
            {"track_dt_loss_margin", 0.025},
            {"patch_erode_patches", 2},
            {"input_use_verified_patches", true},
        };
        QJsonObject loaded{
            {"paths", QJsonObject{
                {"outer_shell", "/shell/one"},
                {"verified_patches", "/patches"},
            }},
            {"run", QJsonObject{{"config", defaults}}},
        };
        QJsonObject current = loaded;
        current["paths"] = QJsonObject{
            {"outer_shell", "/shell/two"},
            {"verified_patches", "/patches"},
        };
        QJsonObject run = current["run"].toObject();
        QJsonObject config = run["config"].toObject();
        config["track_dt_loss_margin"] = 0.5;
        run["config"] = config;
        current["run"] = run;

        const QSet<QString> mutableConfig{"track_dt_loss_margin"};
        const QSet<QString> mutablePaths{"outer_shell"};
        QCOMPARE(
            vc3d::normalizedSpiralReloadRequest(
                current, defaults, mutableConfig, mutablePaths),
            vc3d::normalizedSpiralReloadRequest(
                loaded, defaults, mutableConfig, mutablePaths));
    }

    void PatchToggleStillRequiresReload()
    {
        const QJsonObject defaults{
            {"input_use_verified_patches", true},
        };
        QJsonObject loaded{
            {"paths", QJsonObject{{"verified_patches", "/patches"}}},
            {"run", QJsonObject{{"config", defaults}}},
        };
        QJsonObject current = loaded;
        QJsonObject run = current["run"].toObject();
        QJsonObject config = run["config"].toObject();
        config["input_use_verified_patches"] = false;
        run["config"] = config;
        current["run"] = run;

        QVERIFY(
            vc3d::normalizedSpiralReloadRequest(current, defaults, {})
            != vc3d::normalizedSpiralReloadRequest(loaded, defaults, {}));
    }
};

QTEST_APPLESS_MAIN(SpiralReloadComparisonTest)
#include "test_spiral_reload_comparison.moc"
