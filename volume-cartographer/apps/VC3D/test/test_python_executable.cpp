// How VC3D starts its Python services: the installed Python tools it falls
// back to when nothing else names an interpreter (opened from Finder, with no
// virtualenv or conda env active), and the environment the services get.

#include "PythonExecutable.hpp"

#include <QDir>
#include <QFile>
#include <QTemporaryDir>
#include <QtTest/QtTest>

using namespace vc3d;

namespace {

QString makeExecutable(const QString& path)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath()))
        return {};
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly))
        return {};
    file.close();
    file.setPermissions(file.permissions() | QFileDevice::ExeOwner);
    return QDir::cleanPath(path);
}

// Writes the marker the installer leaves once an environment is complete
bool markInstalled(const QString& environment)
{
    QFile marker(QDir(environment).filePath(".vc3d-installed"));
    return marker.open(QIODevice::WriteOnly);
}

}  // namespace

class PythonExecutableTest : public QObject
{
    Q_OBJECT

private slots:
    void init()
    {
        QVERIFY(_root.isValid());
        qputenv("VC3D_PYTHON_TOOLS", _root.path().toUtf8());
        qunsetenv("VIRTUAL_ENV");
        qunsetenv("CONDA_PREFIX");
    }

    void installedEnvironmentsAndSources()
    {
        QVERIFY(installedPythonEnvironment("spiral").isEmpty());
        QVERIFY(installedPythonToolsSource("lasagna/fit_service.py").isEmpty());

        const QString spiral = makeExecutable(_root.filePath("envs/spiral/bin/python"));
        QVERIFY(!spiral.isEmpty());
        // still installing: no marker yet
        QVERIFY(installedPythonEnvironment("spiral").isEmpty());
        QVERIFY(markInstalled(_root.filePath("envs/spiral")));
        QCOMPARE(installedPythonEnvironment("spiral"), spiral);
        QVERIFY(installedPythonEnvironment("lasagna").isEmpty());

        // a file without the execute bit is not an interpreter
        QVERIFY(QDir().mkpath(_root.filePath("envs/lasagna/bin")));
        QFile plain(_root.filePath("envs/lasagna/bin/python"));
        QVERIFY(plain.open(QIODevice::WriteOnly));
        plain.close();
        QVERIFY(plain.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner));
        QVERIFY(markInstalled(_root.filePath("envs/lasagna")));
        QVERIFY(installedPythonEnvironment("lasagna").isEmpty());

        const QString service = makeExecutable(_root.filePath("villa/lasagna/fit_service.py"));
        QCOMPARE(QDir::cleanPath(installedPythonToolsSource("lasagna/fit_service.py")), service);
    }

    void servicesPreferTheActiveEnvironmentThenTheInstalledOne()
    {
        const QString installed = makeExecutable(_root.filePath("envs/vesuvius/bin/python"));
        QVERIFY(markInstalled(_root.filePath("envs/vesuvius")));
        QCOMPARE(pythonExecutableForServices(), installed);

        QTemporaryDir venv;
        const QString active = makeExecutable(venv.filePath("bin/python"));
        qputenv("VIRTUAL_ENV", venv.path().toUtf8());
        QCOMPARE(pythonExecutableForServices(), active);

        QCOMPARE(pythonExecutableForServices("/opt/chosen/bin/python"), QString("/opt/chosen/bin/python"));
    }

    // what the install prompt checks before asking: only a Python the user set up counts, never the installed
    // tools or a python3 on PATH
    void chosenPythonIsOnlyOneTheUserSetUp()
    {
        makeExecutable(_root.filePath("envs/vesuvius/bin/python"));
        QVERIFY(markInstalled(_root.filePath("envs/vesuvius")));
        QVERIFY(chosenPythonForServices().isEmpty());
        QVERIFY(!pythonExecutableForServices().isEmpty());

        QTemporaryDir own;
        QVERIFY(chosenPythonForServices(own.filePath("missing/bin/python")).isEmpty());
        const QString preferred = makeExecutable(own.filePath("tools/bin/python"));
        QCOMPARE(chosenPythonForServices(preferred), preferred);

        const QString conda = makeExecutable(own.filePath("conda/bin/python"));
        qputenv("CONDA_PREFIX", own.filePath("conda").toUtf8());
        QCOMPARE(chosenPythonForServices(), conda);
        QCOMPARE(chosenPythonForServices(preferred), preferred);
    }

    void pythonServicesGetTheLaunchValuesBack()
    {
        qputenv("VC3D_TEST_THREADS", "8");
        qunsetenv("VC3D_TEST_UNSET");
        rememberLaunchEnvironmentValue("VC3D_TEST_THREADS");
        rememberLaunchEnvironmentValue("VC3D_TEST_UNSET");
        // VC3D's own caps, as setThreadPoliciesEarly() sets them
        qputenv("VC3D_TEST_THREADS", "1");
        qputenv("VC3D_TEST_UNSET", "1");
        // a second record of a name keeps the launch value
        rememberLaunchEnvironmentValue("VC3D_TEST_THREADS");

        const QProcessEnvironment environment = pythonServiceEnvironment();
        QCOMPARE(environment.value("VC3D_TEST_THREADS"), QString("8"));
        QVERIFY(!environment.contains("VC3D_TEST_UNSET"));
        QCOMPARE(qgetenv("VC3D_TEST_THREADS"), QByteArray("1"));
        QVERIFY(environment.contains("PATH"));
    }

private:
    QTemporaryDir _root;
};

QTEST_APPLESS_MAIN(PythonExecutableTest)
#include "test_python_executable.moc"
