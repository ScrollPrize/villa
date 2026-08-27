#include "SpiralReloadComparison.hpp"
#include "SpiralFiberRevisionUpload.hpp"
#include "SpiralSessionSync.hpp"

#include <QJsonObject>
#include <QtTest/QtTest>

class SpiralReloadComparisonTest final : public QObject
{
    Q_OBJECT

private slots:
    void FiberCasConflictPreservesCurrentRevisionForRetry()
    {
        const QJsonObject conflict{
            {"error", "fiber revision conflict"},
            {"current_revision", "revision-2"},
        };
        const QString current =
            vc3d::spiralFiberConflictRevision(conflict);

        QCOMPARE(current, QStringLiteral("revision-2"));
        QVERIFY(vc3d::spiralFiberUploadNeedsCasRetry(
            current, conflict["error"].toString()));
        QVERIFY(!vc3d::spiralFiberUploadNeedsCasRetry(
            {}, conflict["error"].toString()));
        QVERIFY(!vc3d::spiralFiberUploadNeedsCasRetry(current, {}));
    }

    void CheckpointLoadInitializesWithoutProfileOverrides()
    {
        const QJsonObject request{
            {"paths", QJsonObject{{"verified_patches", "/patches"}}},
            {"run", QJsonObject{
                {"z_begin", 4000},
                {"config", QJsonObject{{"model_num_flow_stages", 9}}},
            }},
        };

        const QJsonObject initialized =
            vc3d::spiralCheckpointInitializationRequest(
                request, QStringLiteral("/checkpoints/resume.ckpt"));

        QCOMPARE(initialized["paths"].toObject()["checkpoint"].toString(),
                 QStringLiteral("/checkpoints/resume.ckpt"));
        QCOMPARE(initialized["paths"].toObject()["verified_patches"].toString(),
                 QStringLiteral("/patches"));
        QCOMPARE(initialized["run"].toObject()["z_begin"].toInt(), 4000);
        QVERIFY(initialized["run"].toObject()["config"].toObject().isEmpty());

        QVERIFY(vc3d::spiralCheckpointLoadAvailable(
            true, QStringLiteral("Uninitialized"), true));
        QVERIFY(vc3d::spiralCheckpointLoadAvailable(
            true, QStringLiteral("Idle"), true));
        QVERIFY(vc3d::spiralCheckpointLoadAvailable(
            true, QStringLiteral("Error"), true));
        QVERIFY(!vc3d::spiralCheckpointLoadAvailable(
            true, QStringLiteral("Loading"), true));
        QVERIFY(!vc3d::spiralCheckpointLoadAvailable(
            true, QStringLiteral("Running"), true));
        QVERIFY(!vc3d::spiralCheckpointLoadAvailable(
            true, QStringLiteral("Uninitialized"), false));
    }

    void CheckpointLoadAdoptsOnlyTheCanonicalCheckpointPath()
    {
        const QJsonObject loaded{
            {"paths", QJsonObject{{"checkpoint", "/checkpoints/old.ckpt"}}},
            {"run", QJsonObject{{"z_begin", 4000}, {"z_end", 17000}}},
        };
        QJsonObject edited = loaded;
        edited["run"] = QJsonObject{{"z_begin", 5000}, {"z_end", 17000}};
        edited["paths"] = QJsonObject{
            {"checkpoint", "/checkpoints/new.ckpt"}};

        const QJsonObject adopted =
            vc3d::spiralSessionRequestWithCheckpoint(
                loaded, QStringLiteral("/checkpoints/new.ckpt"));

        QCOMPARE(adopted["paths"].toObject()["checkpoint"].toString(),
                 QStringLiteral("/checkpoints/new.ckpt"));
        QCOMPARE(adopted["run"].toObject()["z_begin"].toInt(), 4000);
        QVERIFY(vc3d::normalizedSpiralReloadRequest(edited, {}, {})
                != vc3d::normalizedSpiralReloadRequest(adopted, {}, {}));
    }

    void RunConfigurationContainsOnlyRunBoundaryFields()
    {
        const QJsonObject editorConfig{
            {"dense_spacing_mode", "winding_model"},
            {"z_begin", 4000},
            {"loss_weight_patch_radius", 3.0},
        };
        const QSet<QString> runBoundaryKeys{
            QStringLiteral("loss_weight_patch_radius")};
        const QJsonObject expected{
            {"loss_weight_patch_radius", 3.0}};

        QCOMPARE(
            vc3d::spiralRunBoundaryConfig(editorConfig, runBoundaryKeys),
            expected);
    }

    void CompleteRunConfigurationExcludesCheckpointRunBlockFields()
    {
        const QJsonObject defaults{
            {"loss_weight_patch_radius", 8.0},
            {"optimizer_learning_rate", 3e-5},
        };
        const QJsonObject checkpointApplied{
            {"loss_weight_patch_radius", 6.0},
            {"optimizer_learning_rate", 3e-5},
            {"z_begin", 4000},
            {"z_end", 17000},
        };
        const QJsonObject runConfig{
            {"loss_weight_patch_radius", 2.0},
            {"z_begin", 5000},
        };

        const QJsonObject configuration =
            vc3d::completeSpiralRunConfiguration(
                defaults, checkpointApplied, runConfig);

        QCOMPARE(configuration.keys(), defaults.keys());
        QCOMPARE(configuration["loss_weight_patch_radius"].toDouble(), 2.0);
        QCOMPARE(configuration["optimizer_learning_rate"].toDouble(), 3e-5);
        QVERIFY(!configuration.contains("z_begin"));
        QVERIFY(!configuration.contains("z_end"));
    }

    void EffectiveAdvancedConfigurationExcludesDockOwnedZRange()
    {
        const QJsonObject request{
            {"run", QJsonObject{
                {"z_begin", 120},
                {"z_end", 840},
                {"config", QJsonObject{
                    {"z_begin", 1},
                    {"z_end", 2},
                    {"loss_weight_patch_radius", 3.0},
                }},
            }},
        };
        const QJsonObject effective = vc3d::effectiveSpiralSessionConfig(
            request,
            QJsonObject{{"z_begin", 4000}, {"z_end", 17000}},
            QJsonObject{});

        QVERIFY(!effective.contains("z_begin"));
        QVERIFY(!effective.contains("z_end"));
        QCOMPARE(effective["loss_weight_patch_radius"].toDouble(), 3.0);
    }

    void runMutableConfigAndShellPathDoNotRequireFullReload()
    {
        const QJsonObject defaults{
            {"track_dt_loss_margin", 0.025},
            {"patch_erode_patches", 2},
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

    void PatchPathChangeStillRequiresReload()
    {
        const QJsonObject defaults;
        QJsonObject loaded{
            {"paths", QJsonObject{{"verified_patches", "/patches"}}},
            {"run", QJsonObject{{"config", defaults}}},
        };
        QJsonObject current = loaded;
        current["paths"] = QJsonObject{{"verified_patches", "/other-patches"}};

        QVERIFY(
            vc3d::normalizedSpiralReloadRequest(current, defaults, {})
            != vc3d::normalizedSpiralReloadRequest(loaded, defaults, {}));
    }

    void OnlyAllowlistedConfigurationIsAModelStageRebuild()
    {
        const QSet<QString> modelStageKeys{"model_num_flow_stages"};
        const QJsonObject loaded{
            {"paths", QJsonObject{{"verified_patches", "/patches"}}},
            {"run", QJsonObject{{"z_end", 900}, {"config", QJsonObject{
                {"model_num_flow_stages", 1},
                {"loss_weight_patch_radius", 8.0},
            }}}},
        };
        // Identical requests, and requests differing only in an allowlisted
        // key, keep the loaded inputs.
        QCOMPARE(vc3d::spiralRebuildStage(loaded, loaded, modelStageKeys),
                 QStringLiteral("model"));
        auto withConfig = [&](const QString& key, const QJsonValue& value) {
            QJsonObject request = loaded;
            QJsonObject run = request["run"].toObject();
            QJsonObject config = run["config"].toObject();
            config[key] = value;
            run["config"] = config;
            request["run"] = run;
            return request;
        };
        QCOMPARE(vc3d::spiralRebuildStage(
                     withConfig("model_num_flow_stages", 3), loaded,
                     modelStageKeys),
                 QStringLiteral("model"));
        // An unaudited key, a path, or anything else in the run block is the
        // whole build.
        QCOMPARE(vc3d::spiralRebuildStage(
                     withConfig("loss_weight_patch_radius", 1.0), loaded,
                     modelStageKeys),
                 QStringLiteral("all"));
        QJsonObject otherRun = loaded;
        otherRun["run"] = QJsonObject{
            {"z_end", 1800}, {"config", loaded["run"].toObject()["config"]}};
        QCOMPARE(vc3d::spiralRebuildStage(otherRun, loaded, modelStageKeys),
                 QStringLiteral("all"));
        QJsonObject otherPaths = loaded;
        otherPaths["paths"] = QJsonObject{{"verified_patches", "/elsewhere"}};
        QCOMPARE(vc3d::spiralRebuildStage(otherPaths, loaded, modelStageKeys),
                 QStringLiteral("all"));
        // A key present on one side only counts as changed, exactly as the
        // service's own diff over the requests counts it.
        QJsonObject dropped = loaded;
        QJsonObject droppedRun = dropped["run"].toObject();
        QJsonObject droppedConfig = droppedRun["config"].toObject();
        droppedConfig.remove("loss_weight_patch_radius");
        droppedRun["config"] = droppedConfig;
        dropped["run"] = droppedRun;
        QCOMPARE(vc3d::spiralRebuildStage(dropped, loaded, modelStageKeys),
                 QStringLiteral("all"));
    }
};

QTEST_APPLESS_MAIN(SpiralReloadComparisonTest)
#include "test_spiral_reload_comparison.moc"
