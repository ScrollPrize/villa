#include "VCSettings.hpp"

#include <QDir>
#include <QFile>
#include <QProcess>
#include <QSettings>
#include <QTemporaryDir>
#include <QtTest/QtTest>

#include <cstdio>

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

#ifndef Q_OS_WIN
    void unwritableRemoteCacheDirectoryIsRejected()
    {
        QTemporaryDir configDir;
        QTemporaryDir cacheParent;
        QVERIFY(configDir.isValid());
        QVERIFY(cacheParent.isValid());

        const QString cacheDir = cacheParent.filePath(QStringLiteral("unwritable"));
        QVERIFY(QDir().mkpath(cacheDir));
        QVERIFY(QFile::setPermissions(cacheDir,
            QFileDevice::ReadOwner | QFileDevice::ExeOwner));

        QSettings settings(configDir.filePath(QStringLiteral("VC3D.ini")),
                           QSettings::IniFormat);
        settings.setValue(vc3d::settings::viewer::REMOTE_CACHE_DIR, cacheDir);
        settings.sync();

        QProcess child;
        QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
        environment.insert(QStringLiteral("VC3D_CONFIG_DIR"), configDir.path());
        environment.insert(QStringLiteral("VC3D_TEST_RESOLVE_CACHE"), QStringLiteral("1"));
        child.setProcessEnvironment(environment);
        child.setProcessChannelMode(QProcess::MergedChannels);
        child.start(QCoreApplication::applicationFilePath());
        QVERIFY(child.waitForFinished());

        // Privileged test users can write despite mode bits, so this fixture
        // cannot exercise the rejection path in that environment.
        if (child.exitCode() == 0)
            QSKIP("test user can write to a mode-0500 directory");
        QCOMPARE(child.exitCode(), 2);
        const QByteArray diagnostic = child.readAll();
        QVERIFY2(QString::fromUtf8(diagnostic).contains(QStringLiteral("is not writable")),
                 diagnostic.constData());
    }
#endif
};

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    if (qEnvironmentVariableIsSet("VC3D_TEST_RESOLVE_CACHE")) {
        try {
            (void)vc3d::remoteCachePath();
            return 0;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "%s\n", error.what());
            return 2;
        }
    }

    VCSettingsTest test;
    return QTest::qExec(&test, argc, argv);
}

#include "test_vc_settings.moc"
