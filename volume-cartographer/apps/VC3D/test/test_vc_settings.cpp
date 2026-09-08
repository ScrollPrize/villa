#include "VCSettings.hpp"

#include <QDir>
#include <QSettings>
#include <QTemporaryDir>
#include <QtTest/QtTest>

class VCSettingsTest : public QObject
{
    Q_OBJECT

private slots:
    void storedRemoteCacheDirectoryIsGlobalAndRestartOnly()
    {
        QTemporaryDir configDir;
        QTemporaryDir cacheParent;
        QVERIFY(configDir.isValid());
        QVERIFY(cacheParent.isValid());
        qputenv("VC3D_CONFIG_DIR", configDir.path().toUtf8());

        const QString first = cacheParent.filePath(
            QString::fromUtf8("f\xC3\xADrst-\xE6\xBC\xA2\xE5\xAD\x97"));
        const QString second = cacheParent.filePath(QStringLiteral("second cache"));
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue(vc3d::settings::viewer::REMOTE_CACHE_DIR, first);
        settings.sync();

        QCOMPARE(QDir::cleanPath(vc3d::remoteCachePath()), QDir::cleanPath(first));
        QVERIFY(QDir(first).exists());

        settings.setValue(vc3d::settings::viewer::REMOTE_CACHE_DIR, second);
        settings.sync();
        QCOMPARE(QDir::cleanPath(vc3d::remoteCachePath()), QDir::cleanPath(first));
        QVERIFY(!QDir(second).exists());
    }
};

QTEST_APPLESS_MAIN(VCSettingsTest)

#include "test_vc_settings.moc"
