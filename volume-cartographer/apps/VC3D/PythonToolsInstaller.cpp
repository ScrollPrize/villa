#include "PythonToolsInstaller.hpp"

#include "PythonExecutable.hpp"
#include "VCSettings.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QCoreApplication>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFontDatabase>
#include <QLabel>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSettings>
#include <QTimer>
#include <QVBoxLayout>

#ifdef Q_OS_UNIX
#include <csignal>
#include <unistd.h>
#endif

namespace vc3d {

namespace {

// how long a cancelled installer gets to stop before it and its downloads are killed
constexpr int kCancelGraceMs = 5000;

// GUI-thread state of requirePythonTools for this run of VC3D
bool installRunning = false;
bool promptPending = false;
bool declinedThisSession = false;

// the environment finished installing and the villa sources its services run from are in place
bool environmentReady(const QString& env)
{
    // the installer unpacks the sources after the environments, so an interrupted install can lack them
    return !installedPythonEnvironment(env).isEmpty()
        && !installedPythonToolsSource(QStringLiteral(".villa-commit")).isEmpty();
}

QString firstLine(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return {};
    }
    return QString::fromUtf8(file.readLine()).trimmed();
}

// the installed villa sources are the ones the bundled installer installs. An app update ships an installer for newer
// sources and packages, and its services must not run against the old ones
bool toolsMatchBundle()
{
    QFile script(bundledPythonToolsInstaller());
    if (!script.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return true;
    }
    static const QRegularExpression commitLine(QStringLiteral("^VILLA_COMMIT=(\\S+)$"),
                                               QRegularExpression::MultilineOption);
    const auto match = commitLine.match(QString::fromUtf8(script.readAll()));
    if (!match.hasMatch()) {
        return true;
    }
    return firstLine(installedPythonToolsSource(QStringLiteral(".villa-commit"))) == match.captured(1);
}

bool insideInstalledTools(const QString& path)
{
    const QString root = installedPythonToolsRoot();
    return !root.isEmpty() && QDir::cleanPath(path).startsWith(QDir::cleanPath(root) + QLatin1Char('/'));
}

}  // namespace

QString bundledPythonToolsInstaller()
{
    // Contents/MacOS/VC3D -> Contents/Resources/install-python-tools.sh
    const QString path = QDir(QCoreApplication::applicationDirPath())
                             .filePath(QStringLiteral("../Resources/install-python-tools.sh"));
    return QFileInfo(path).isFile() ? QDir::cleanPath(path) : QString();
}

PythonToolsInstallDialog::PythonToolsInstallDialog(const QString& script, QWidget* parent)
    : QDialog(parent), _script(script)
{
    setWindowTitle(tr("Install Python Tools"));
    setModal(true);
    resize(760, 460);

    _status = new QLabel(
        tr("Installing the Python tools. The first install downloads several GB (mostly PyTorch) and takes a few "
           "minutes; running it again only redoes what changed."),
        this);
    _status->setObjectName(QStringLiteral("pythonToolsInstallStatus"));
    _status->setWordWrap(true);

    _output = new QPlainTextEdit(this);
    _output->setReadOnly(true);
    _output->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    _output->setMaximumBlockCount(5000);

    _buttons = new QDialogButtonBox(QDialogButtonBox::Cancel, this);
    connect(_buttons, &QDialogButtonBox::rejected, this, &PythonToolsInstallDialog::reject);

    auto* layout = new QVBoxLayout(this);
    layout->addWidget(_status);
    layout->addWidget(_output, 1);
    layout->addWidget(_buttons);
}

bool PythonToolsInstallDialog::run()
{
    _process = new QProcess(this);
    _process->setProgram(QStringLiteral("/bin/bash"));
    _process->setArguments({_script});
    // the launch values of the thread variables, as the services the tools serve get them
    _process->setProcessEnvironment(pythonServiceEnvironment());
    _process->setProcessChannelMode(QProcess::MergedChannels);
#ifdef Q_OS_UNIX
    // its own process group, so Cancel also stops the downloads and installs it started
    _process->setChildProcessModifier([] { ::setpgid(0, 0); });
#endif
    connect(_process, &QProcess::readyReadStandardOutput, this, &PythonToolsInstallDialog::appendOutput);
    connect(_process, &QProcess::finished, this, [this](int exitCode, QProcess::ExitStatus status) {
        finished(exitCode, status == QProcess::CrashExit);
    });
    connect(_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (error == QProcess::FailedToStart) {
            _output->appendPlainText(_process->errorString());
            finished(-1, true);
        }
    });
    _process->start();
    exec();
    return _succeeded;
}

void PythonToolsInstallDialog::appendOutput()
{
    auto* bar = _output->verticalScrollBar();
    const bool atEnd = bar->value() == bar->maximum();
    _output->moveCursor(QTextCursor::End);
    _output->insertPlainText(QString::fromUtf8(_process->readAllStandardOutput()));
    if (atEnd) {
        bar->setValue(bar->maximum());
    }
}

void PythonToolsInstallDialog::finished(int exitCode, bool crashed)
{
    appendOutput();
    _succeeded = !_cancelled && !crashed && exitCode == 0;
    if (_succeeded) {
        _status->setText(tr("Done: the Python tools are installed. Spiral, Lasagna, sheet strips and fibre follow "
                            "use them from now on."));
    } else if (_cancelled) {
        _status->setText(tr("Install cancelled. Install again from the Help menu; finished steps are kept."));
    } else {
        _status->setText(tr("Install failed (see the last lines above). Install again from the Help menu to retry; "
                            "finished steps are kept."));
    }
    _buttons->setStandardButtons(QDialogButtonBox::Close);
}

void PythonToolsInstallDialog::reject()
{
    if (_process && _process->state() != QProcess::NotRunning) {
        cancel();
        return;
    }
    QDialog::reject();
}

void PythonToolsInstallDialog::cancel()
{
    if (_cancelled) {
        return;
    }
    _cancelled = true;
    _status->setText(tr("Cancelling..."));
    _buttons->setEnabled(false);
#ifdef Q_OS_UNIX
    const auto group = static_cast<pid_t>(_process->processId());
    if (group > 0) {
        ::kill(-group, SIGTERM);
        QTimer::singleShot(kCancelGraceMs, _process, [process = _process, group] {
            if (process && process->state() != QProcess::NotRunning) {
                ::kill(-group, SIGKILL);
            }
        });
    }
#else
    _process->kill();
#endif
    connect(_process, &QProcess::finished, _buttons, [this] { _buttons->setEnabled(true); });
}

void installPythonTools(QWidget* parent)
{
    const QString script = bundledPythonToolsInstaller();
    if (script.isEmpty() || installRunning) {
        return;
    }
    // asking from the Help menu undoes a "Not Now" for this session
    declinedThisSession = false;
    installRunning = true;
    PythonToolsInstallDialog dialog(script, parent);
    dialog.run();
    installRunning = false;
}

namespace {

// the prompt behind requirePythonTools, run from the event loop after the feature has returned
void askToInstall(const QString& feature, const QString& env, bool update, QWidget* parent,
                  const std::function<void()>& retry)
{
    QMessageBox box(parent ? parent : QApplication::activeWindow());
#if QT_VERSION >= QT_VERSION_CHECK(6, 6, 0)
    // Qt's own dialog rather than an NSAlert, so the agent bridge (ui.dialog) can see and answer it
    box.setOptions(QMessageBox::Option::DontUseNativeDialog);
#endif
    box.setIcon(QMessageBox::Question);
    box.setWindowTitle(QObject::tr("Python Tools"));
    box.setText(update ? QObject::tr("%1 needs the Python tools for this version of VC3D. Update them now?").arg(feature)
                       : QObject::tr("%1 needs the Python tools. Install them now?").arg(feature));
    box.setInformativeText(QObject::tr(
        "They are installed once for Spiral, Lasagna, sheet strips, fibre follow and the neural tracer. The install "
        "downloads several GB and takes a few minutes; it needs no password. Help > Remove Python Tools removes "
        "them again."));
    auto* install = box.addButton(update ? QObject::tr("Update") : QObject::tr("Install"), QMessageBox::AcceptRole);
    box.addButton(QObject::tr("Not Now"), QMessageBox::RejectRole);
    box.setDefaultButton(install);
    box.setCheckBox(new QCheckBox(QObject::tr("Don't ask again")));
    box.exec();
    promptPending = false;

    if (box.clickedButton() != install) {
        // only a declined prompt is remembered: after a failed or cancelled install VC3D asks again
        if (box.checkBox()->isChecked()) {
            QSettings(settingsFilePath(), QSettings::IniFormat)
                .setValue(settings::python_tools::SKIP_INSTALL_PROMPT, true);
        }
        declinedThisSession = true;
        return;
    }
    installPythonTools(parent);
    if (retry && environmentReady(env) && toolsMatchBundle()) {
        retry();
    }
}

}  // namespace

bool requirePythonTools(const QString& feature, const QString& env, const QString& chosenPython, QWidget* parent,
                        std::function<void()> retry)
{
    if (bundledPythonToolsInstaller().isEmpty()) {
        return true;
    }
    const bool installed = environmentReady(env);
    const bool update = installed && !toolsMatchBundle();
    if (installed && !update) {
        return true;
    }
    // a Python the user set up for this feature is theirs to keep: no prompt, and no second copy of PyTorch. The
    // installer points [spiral] python at its own environment, which is no reason to skip an update
    if (!chosenPython.isEmpty() && QFileInfo(chosenPython).isExecutable() && !insideInstalledTools(chosenPython)) {
        return true;
    }
    QSettings stored(settingsFilePath(), QSettings::IniFormat);
    if (stored.value(settings::python_tools::SKIP_INSTALL_PROMPT, false).toBool()) {
        return true;
    }
    // after "Not Now" to an update, the features keep using the tools already installed
    if (update && declinedThisSession) {
        return true;
    }
    if (promptPending || installRunning || declinedThisSession) {
        return false;
    }
    // asked once the caller has returned: callers include agent bridge handlers, which must not run a modal loop,
    // and features started while another is asking share the one prompt
    promptPending = true;
    QTimer::singleShot(0, qApp, [feature, env, update, owner = QPointer<QWidget>(parent), retry = std::move(retry)] {
        askToInstall(feature, env, update, owner, retry);
    });
    return false;
}

QString pythonToolsMissingMessage(const QString& feature)
{
    return QObject::tr("%1 needs the Python tools. Install them when VC3D asks, or from Help > Install or Update "
                       "Python Tools, then start it again.")
        .arg(feature);
}

void removePythonTools(QWidget* parent)
{
    const QString root = installedPythonToolsRoot();
    // only a folder the installer made: VC3D_PYTHON_TOOLS may name anything, such as a ~/.local that also holds uv.
    // The installer keeps the uv archive it unpacked bin/uv from
    const QDir dir(root);
    if (root.isEmpty() || !dir.exists() || !QFileInfo(dir.filePath(QStringLiteral("bin/uv"))).isFile()
        || !QFileInfo(dir.filePath(QStringLiteral("downloads/uv-aarch64-apple-darwin.tar.gz"))).isFile()) {
        QMessageBox::information(parent, QObject::tr("Remove Python Tools"),
                                 QObject::tr("No Python tools are installed."));
        return;
    }
    const auto answer = QMessageBox::question(
        parent, QObject::tr("Remove Python Tools"),
        QObject::tr("Remove the Python tools?\n\nThis deletes %1. Spiral, Lasagna, sheet strips, fibre follow and the "
                    "neural tracer need them again later; services running now stop working until VC3D restarts.")
            .arg(QDir::toNativeSeparators(root)),
        QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
    if (answer != QMessageBox::Yes) {
        return;
    }

    QApplication::setOverrideCursor(Qt::WaitCursor);
    const bool removed = QDir(root).removeRecursively();
    QApplication::restoreOverrideCursor();

    // the installer points [spiral] python at its environment; a removed one would hide any other Python
    QSettings stored(settingsFilePath(), QSettings::IniFormat);
    const QString spiralPython = stored.value(QStringLiteral("spiral/python")).toString();
    if (!spiralPython.isEmpty() && QDir::cleanPath(spiralPython).startsWith(QDir::cleanPath(root) + QLatin1Char('/'))) {
        stored.remove(QStringLiteral("spiral/python"));
    }
    if (removed) {
        QMessageBox::information(parent, QObject::tr("Remove Python Tools"),
                                 QObject::tr("The Python tools are removed."));
    } else {
        QMessageBox::warning(parent, QObject::tr("Remove Python Tools"),
                             QObject::tr("Some files could not be removed. Delete %1 in Finder to finish.")
                                 .arg(QDir::toNativeSeparators(root)));
    }
}

}  // namespace vc3d
