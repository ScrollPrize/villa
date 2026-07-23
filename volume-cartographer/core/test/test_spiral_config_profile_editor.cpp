#include <QApplication>
#include <QComboBox>
#include <QDialog>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QTemporaryDir>

#include "SpiralReloadComparison.hpp"
#include "elements/SpiralConfigProfileEditor.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        std::exit(1);
    }
}
}

int main(int argc, char** argv)
{
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM"))
        qputenv("QT_QPA_PLATFORM", "offscreen");

    QTemporaryDir home;
    require(home.isValid(), "Temporary HOME is unavailable");
    qputenv("VC3D_CONFIG_DIR", home.path().toUtf8());
    std::unique_ptr<QApplication> app = std::make_unique<QApplication>(argc, argv);

    {
        QFile profiles(home.filePath(QStringLiteral("spiral-advanced-profiles.json")));
        require(profiles.open(QIODevice::WriteOnly), "Could not seed saved profiles");
        profiles.write(R"({"version":1,"profiles":[{"id":"p1","name":"My profile","json":"{\n  \"saved\": 1\n}"}]})");
    }

    SpiralConfigProfileEditor editor;
    auto* combo = editor.findChild<QComboBox*>(QStringLiteral("spiralAdvancedProfileCombo"));
    auto* textEdit = editor.findChild<QPlainTextEdit*>(QStringLiteral("spiralAdvancedJsonEditor"));
    require(combo && textEdit, "Advanced profile controls were not constructed");
    if (combo->findData(QStringLiteral("default")) < 0
        || combo->findData(QStringLiteral("custom")) < 0
        || combo->findData(QStringLiteral("p1")) < 0) {
        QStringList entries;
        for (int index = 0; index < combo->count(); ++index)
            entries << combo->itemText(index) + QStringLiteral("=")
                + combo->itemData(index).toString();
        std::cerr << "Profile entries: " << entries.join('|').toStdString() << std::endl;
        require(false, "Default, Custom, and persisted profiles must share the dropdown");
    }
    require(editor.currentText() == QStringLiteral("{}"),
            "Default must not restore legacy Advanced JSON");

    const QJsonObject defaults{
        {QStringLiteral("session_value"), 7},
        {QStringLiteral("run_value"), 11},
    };
    const QSet<QString> runKeys{QStringLiteral("run_value")};
    const QJsonObject sparseDefaultRequest{
        {QStringLiteral("run"), QJsonObject{
            {QStringLiteral("config"), QJsonObject{}},
            {QStringLiteral("z_begin"), 100},
        }},
    };
    const QJsonObject expandedCustomRequest{
        {QStringLiteral("run"), QJsonObject{
            {QStringLiteral("config"), defaults},
            {QStringLiteral("z_begin"), 100},
        }},
    };
    require(
        vc3d::normalizedSpiralReloadRequest(
            sparseDefaultRequest, defaults, runKeys)
            == vc3d::normalizedSpiralReloadRequest(
                expandedCustomRequest, defaults, runKeys),
        "Sparse Default and equivalent expanded Custom must compare equally");

    QJsonObject runOnlyChange = expandedCustomRequest;
    QJsonObject runOnlyBody = runOnlyChange.value(QStringLiteral("run")).toObject();
    QJsonObject runOnlyConfig = runOnlyBody.value(QStringLiteral("config")).toObject();
    runOnlyConfig[QStringLiteral("run_value")] = 99;
    runOnlyBody[QStringLiteral("config")] = runOnlyConfig;
    runOnlyChange[QStringLiteral("run")] = runOnlyBody;
    require(
        vc3d::normalizedSpiralReloadRequest(
            sparseDefaultRequest, defaults, runKeys)
            == vc3d::normalizedSpiralReloadRequest(
                runOnlyChange, defaults, runKeys),
        "A run-mutable Advanced value must not require reload");

    QJsonObject sessionChange = expandedCustomRequest;
    QJsonObject sessionBody = sessionChange.value(QStringLiteral("run")).toObject();
    QJsonObject sessionConfig = sessionBody.value(QStringLiteral("config")).toObject();
    sessionConfig[QStringLiteral("session_value")] = 8;
    sessionBody[QStringLiteral("config")] = sessionConfig;
    sessionChange[QStringLiteral("run")] = sessionBody;
    require(
        vc3d::normalizedSpiralReloadRequest(
            sparseDefaultRequest, defaults, runKeys)
            != vc3d::normalizedSpiralReloadRequest(
                sessionChange, defaults, runKeys),
        "A changed session-scoped Advanced value must require reload");

    editor.setSessionDefault(QJsonObject{{QStringLiteral("python_default"), 7}});
    require(editor.currentText().contains(QStringLiteral("python_default")),
            "Session Default did not update the editor");

    auto* popOut = editor.findChild<QPushButton*>(QStringLiteral("spiralAdvancedPopOut"));
    auto* dialog = editor.findChild<QDialog*>(QStringLiteral("spiralAdvancedConfigDialog"));
    require(popOut && dialog && !dialog->isModal(),
            "Advanced editor pop-out must be a modeless dialog");
    int presentationTextChanges = 0;
    QObject::connect(&editor, &SpiralConfigProfileEditor::textChanged,
                     [&presentationTextChanges]() { ++presentationTextChanges; });
    const QString textBeforePopOut = editor.currentText();
    popOut->click();
    QApplication::processEvents();
    require(dialog->isVisible() && dialog->isAncestorOf(textEdit),
            "Pop Out must move the authoritative editor into the dialog");
    require(editor.currentProfileId() == QStringLiteral("default"),
            "Pop Out must not convert Default into a Custom profile");
    require(editor.currentText() == textBeforePopOut && presentationTextChanges == 0,
            "Pop Out must not report an Advanced JSON edit");
    dialog->close();
    QApplication::processEvents();
    require(!dialog->isVisible() && editor.isAncestorOf(textEdit),
            "Closing the dialog must pop the same editor back inline");
    require(editor.currentProfileId() == QStringLiteral("default")
                && presentationTextChanges == 0,
            "Pop In must not change the profile or report an edit");

    combo->setCurrentIndex(combo->findData(QStringLiteral("p1")));
    editor.setCurrentText(QStringLiteral("{\"saved\":3}"));
    auto* save = editor.findChild<QPushButton*>(QStringLiteral("spiralAdvancedProfileSave"));
    require(save && save->isEnabled(), "Dirty persisted profile must enable Save");
    save->click();
    {
        QFile profiles(home.filePath(QStringLiteral("spiral-advanced-profiles.json")));
        require(profiles.open(QIODevice::ReadOnly), "Could not read saved profiles");
        const QJsonArray stored = QJsonDocument::fromJson(profiles.readAll())
                                      .object().value(QStringLiteral("profiles")).toArray();
        require(stored.size() == 1
                    && stored[0].toObject().value(QStringLiteral("json")).toString()
                        == QStringLiteral("{\"saved\":3}"),
                "Save must explicitly overwrite the selected profile");
    }

    combo->setCurrentIndex(combo->findData(QStringLiteral("default")));
    editor.setCurrentText(QStringLiteral("{\"draft\":2}"));
    require(editor.currentProfileId() == QStringLiteral("custom"),
            "Editing Default must create a Custom draft");

    popOut->click();
    QApplication::processEvents();
    require(dialog->isVisible() && dialog->isAncestorOf(textEdit),
            "Pop Out must move the authoritative editor into the dialog");
    dialog->close();
    QApplication::processEvents();
    require(!dialog->isVisible() && editor.isAncestorOf(textEdit),
            "Closing the dialog must pop the same editor back inline");

    SpiralConfigProfileEditor restarted;
    require(restarted.currentText() == QStringLiteral("{}"),
            "Session Default and Custom draft must not persist across instances");
    auto* restartedCombo = restarted.findChild<QComboBox*>(
        QStringLiteral("spiralAdvancedProfileCombo"));
    restartedCombo->setCurrentIndex(restartedCombo->findData(QStringLiteral("p1")));
    require(restarted.currentText() == QStringLiteral("{\"saved\":3}"),
            "Saved profiles must persist across editor instances");

    return 0;
}
