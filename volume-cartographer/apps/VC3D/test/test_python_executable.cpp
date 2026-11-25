// The installed Python tools VC3D falls back to when nothing else names an
// interpreter (opened from Finder, with no virtualenv or conda env active).

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

private:
    QTemporaryDir _root;
};

QTEST_APPLESS_MAIN(PythonExecutableTest)
#include "test_python_executable.moc"
