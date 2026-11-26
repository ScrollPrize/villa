#pragma once

#include <QDialog>
#include <QPointer>
#include <QString>

#include <functional>

class QDialogButtonBox;
class QLabel;
class QPlainTextEdit;
class QProcess;
class QWidget;

namespace vc3d {

// The Python tools installer shipped inside the app bundle
// (Contents/Resources/install-python-tools.sh), or empty when this build ships
// none. Release builds for macOS ship it, so the script is covered by the
// app's signature and notarization and runs without a Gatekeeper prompt of its
// own.
QString bundledPythonToolsInstaller();

// Runs the bundled installer and shows its output as it runs, with Cancel
// while it runs and a result line when it ends.
class PythonToolsInstallDialog : public QDialog
{
    Q_OBJECT

public:
    PythonToolsInstallDialog(const QString& script, QWidget* parent = nullptr);

    // Starts the installer and runs the dialog until it is closed. True when
    // the install finished.
    bool run();

    void reject() override;

private:
    void appendOutput();
    void finished(int exitCode, bool crashed);
    void cancel();

    QString _script;
    QLabel* _status{nullptr};
    QPlainTextEdit* _output{nullptr};
    QDialogButtonBox* _buttons{nullptr};
    QPointer<QProcess> _process;
    bool _cancelled{false};
    bool _succeeded{false};
};

// Runs the bundled installer in a PythonToolsInstallDialog. Does nothing when
// this build ships no installer.
void installPythonTools(QWidget* parent);

// Called before `feature` starts a service from the installed environment
// `env` (spiral, lasagna or vesuvius). `chosenPython` is the interpreter the
// user pointed the feature at (its setting, SPIRAL_PYTHON, an active conda
// env and the like), or empty; the feature's own lookup decides. True when the
// feature can go ahead: `chosenPython` is executable and outside the
// installed tools, the environment is installed from the sources the bundled
// installer installs, this build ships no installer, or the user chose "Don't
// ask again" (VC3D then looks for a Python as before). Otherwise false, and
// once the caller has returned VC3D asks whether to install the Python tools
// now, or to update them when they were installed for another version of
// VC3D; when that install finishes it calls `retry`, if given, so the feature
// can start after all. It does not ask while it is already asking or
// installing, and after "Not Now" not again until VC3D restarts or the user
// installs from the Help menu; after "Not Now" to an update, features use the
// tools already installed. `parent` defaults to the active window.
bool requirePythonTools(const QString& feature, const QString& env, const QString& chosenPython,
                        QWidget* parent = nullptr, std::function<void()> retry = {});

// What a feature reports when it cannot start because the tools are missing.
QString pythonToolsMissingMessage(const QString& feature);

// Help > Remove Python Tools: after confirmation, deletes the installed tools
// folder and VC3D's [spiral] python setting when it points into it. macOS has
// no uninstall hook, so this is how the tools leave along with the app.
void removePythonTools(QWidget* parent);

}  // namespace vc3d
