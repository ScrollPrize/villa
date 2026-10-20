#include "VolumePathLabels.hpp"
#include "elements/VolumeSelector.hpp"
#include <QtTest/QtTest>

class VolumePathLabelsTest : public QObject
{
    Q_OBJECT
private slots:
    void shortestUniqueSuffixes()
    {
        QCOMPARE(vc3d::distinguishingVolumePaths({
            "/data/a/run/presence.zarr", "/data/b/run/presence.zarr",
            "/data/c/scan.zarr"}),
            QStringList({"a/run/presence.zarr", "b/run/presence.zarr", "scan.zarr"}));
    }
    void urlsAndWindowsPaths()
    {
        QCOMPARE(vc3d::distinguishingVolumePaths({
            "https://one.example/data/a.zarr", "https://two.example/data/a.zarr"}),
            QStringList({"one.example/data/a.zarr", "two.example/data/a.zarr"}));
        QCOMPARE(vc3d::distinguishingVolumePaths({"C:\\a\\x.zarr", "D:\\a\\x.zarr"}),
            QStringList({"C:/a/x.zarr", "D:/a/x.zarr"}));
    }
    void identicalPathsAndScaleSelectors()
    {
        QCOMPARE(vc3d::distinguishingVolumePaths({"/a/x.zarr/", "/a/x.zarr"}),
            QStringList({"x.zarr", "x.zarr"}));
        QCOMPARE(vc3d::volumePathForDisplay("https://user:secret@host/x.zarr?token=secret#vc-base-scale=2"),
            QString("https://user@host/x.zarr#vc-base-scale=2"));
    }
    void uniqueAndEmptyPaths()
    {
        QVERIFY(vc3d::distinguishingVolumePaths({}).isEmpty());
        QCOMPARE(vc3d::distinguishingVolumePaths({"", "/data/scan.zarr"}),
                 QStringList({"", "scan.zarr"}));
        QCOMPARE(vc3d::distinguishingVolumePaths({
            "/data/a.zarr", "/data/b.zarr", "/data/c.zarr"}),
            QStringList({"a.zarr", "b.zarr", "c.zarr"}));
    }
    void remoteCredentialsAreDisplayOnly()
    {
        const QString url =
            "sftp://user:secret@host/data/fiber.zarr?token=secret#vc-base-scale=2";
        VolumeSelector selector;
        selector.setVolumes({{"remote", "fiber.zarr", url}}, "remote");
        QCOMPARE(selector.selectedVolumePath(), url);
        QCOMPARE(selector.comboBox()->toolTip(),
                 QString("sftp://user@host/data/fiber.zarr#vc-base-scale=2"));
        QCOMPARE(selector.comboBox()->itemText(0), QString("fiber.zarr (remote)"));
    }
    void tooltipsFollowSelection()
    {
        QComboBox combo;
        combo.addItem("None");
        combo.addItem("volume");
        combo.setItemData(1, "/data/x.zarr", Qt::ToolTipRole);
        vc3d::refreshVolumeComboTooltip(&combo);
        combo.setCurrentIndex(1);
        QCOMPARE(combo.toolTip(), QString("/data/x.zarr"));
        combo.setCurrentIndex(0);
        QVERIFY(combo.toolTip().isEmpty());
        {
            const QSignalBlocker blocker(&combo);
            combo.setCurrentIndex(1);
            vc3d::refreshVolumeComboTooltip(&combo);
        }
        QCOMPARE(combo.toolTip(), QString("/data/x.zarr"));
        {
            const QSignalBlocker blocker(&combo);
            combo.setCurrentIndex(0);
        }
        QHelpEvent hover(QEvent::ToolTip, QPoint(), QPoint());
        QApplication::sendEvent(&combo, &hover);
        QVERIFY(combo.toolTip().isEmpty());
    }
    void selectorPreservesIdsAndPaths()
    {
        VolumeSelector selector;
        selector.setVolumes({{"one", "x.zarr", "/a/x.zarr"},
                             {"two", "x.zarr", "/b/x.zarr"}}, "two");
        QCOMPARE(selector.selectedVolumeId(), QString("two"));
        QCOMPARE(selector.selectedVolumePath(), QString("/b/x.zarr"));
        QCOMPARE(selector.comboBox()->itemText(0), QString("x.zarr (one) [a/x.zarr]"));
        QCOMPARE(selector.comboBox()->toolTip(), QString("/b/x.zarr"));
        QCOMPARE(selector.comboBox()->itemData(0, Qt::ToolTipRole).toString(), QString("/a/x.zarr"));
    }
};

QTEST_MAIN(VolumePathLabelsTest)
#include "test_volume_path_labels.moc"
