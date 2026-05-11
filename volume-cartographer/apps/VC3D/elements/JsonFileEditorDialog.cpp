#include "elements/JsonFileEditorDialog.hpp"

#include "elements/JsonProfileEditor.hpp"

#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QMessageBox>
#include <QPushButton>
#include <QSaveFile>
#include <QVBoxLayout>

JsonFileEditorDialog::JsonFileEditorDialog(const QString& filePath,
                                           const QString& title,
                                           QWidget* parent)
    : QDialog(parent)
    , _filePath(filePath)
{
    setWindowTitle(title);
    resize(720, 640);

    auto* layout = new QVBoxLayout(this);

    _editor = new JsonProfileEditor(QFileInfo(filePath).fileName(), this);
    _editor->setDescription(tr("Edit and save this JSON file in the current volume package."));
    _editor->setPlaceholderText(tr("{\n  \"key\": \"value\"\n}"));
    _editor->setTextToolTip(filePath);
    _editor->setProfiles({}, QStringLiteral("custom"));

    QFile file(filePath);
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        _editor->setCustomText(QString::fromUtf8(file.readAll()));
    } else {
        _editor->setCustomText(QString());
        QMessageBox::warning(this,
                             tr("Error"),
                             tr("Could not read %1:\n%2")
                                 .arg(QDir::toNativeSeparators(filePath), file.errorString()));
    }

    layout->addWidget(_editor);

    _buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, this);
    connect(_buttons, &QDialogButtonBox::accepted, this, &JsonFileEditorDialog::accept);
    connect(_buttons, &QDialogButtonBox::rejected, this, &JsonFileEditorDialog::reject);
    connect(_editor, &JsonProfileEditor::textChanged, this, &JsonFileEditorDialog::updateOkButton);
    layout->addWidget(_buttons);

    updateOkButton();
}

void JsonFileEditorDialog::accept()
{
    QString error;
    const auto json = _editor->jsonObject(&error);
    if (!json) {
        QMessageBox::warning(this, tr("Invalid JSON"), error);
        updateOkButton();
        return;
    }

    QSaveFile file(_filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this,
                             tr("Error"),
                             tr("Could not write %1:\n%2")
                                 .arg(QDir::toNativeSeparators(_filePath), file.errorString()));
        return;
    }

    file.write(QJsonDocument(*json).toJson(QJsonDocument::Indented));
    if (!file.commit()) {
        QMessageBox::warning(this,
                             tr("Error"),
                             tr("Could not save %1:\n%2")
                                 .arg(QDir::toNativeSeparators(_filePath), file.errorString()));
        return;
    }

    QDialog::accept();
}

void JsonFileEditorDialog::updateOkButton()
{
    if (!_buttons) {
        return;
    }

    auto* button = _buttons->button(QDialogButtonBox::Save);
    if (button) {
        button->setEnabled(_editor && _editor->isValid() && !_editor->currentText().trimmed().isEmpty());
    }
}
