#include "elements/JsonProfileEditor.hpp"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSignalBlocker>

JsonProfileEditor::JsonProfileEditor(const QString& title, QWidget* parent)
    : QGroupBox(title, parent)
{
    auto* layout = new QVBoxLayout(this);

    _description = new QLabel(this);
    _description->setWordWrap(true);
    layout->addWidget(_description);

    auto* profileRow = new QHBoxLayout();
    auto* profileLabel = new QLabel(tr("Profile:"), this);
    _profileCombo = new QComboBox(this);
    _profileCombo->setToolTip(tr("Select a predefined parameter profile.\n"
                                 "- Custom: editable\n"
                                 "- Default/Robust: auto-filled and read-only"));
    profileRow->addWidget(profileLabel);
    profileRow->addWidget(_profileCombo, 1);
    layout->addLayout(profileRow);

    _textEdit = new QPlainTextEdit(this);
    _textEdit->setTabChangesFocus(true);
    layout->addWidget(_textEdit);

    _statusLabel = new QLabel(this);
    _statusLabel->setWordWrap(true);
    _statusLabel->setVisible(false);
    _statusLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
    layout->addWidget(_statusLabel);

    connect(_textEdit, &QPlainTextEdit::textChanged, this, [this]() {
        handleTextEdited();
    });
    connect(_profileCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
        if (!_profileCombo || idx < 0) {
            return;
        }
        const QString profileId = _profileCombo->itemData(idx).toString();
        applyProfile(profileId, true);
    });
}

void JsonProfileEditor::setDescription(const QString& text)
{
    if (!_description) {
        return;
    }
    _description->setText(text);
    _description->setVisible(!text.trimmed().isEmpty());
}

void JsonProfileEditor::setPlaceholderText(const QString& text)
{
    if (_textEdit) {
        _textEdit->setPlaceholderText(text);
    }
}

void JsonProfileEditor::setTextToolTip(const QString& text)
{
    if (_textEdit) {
        _textEdit->setToolTip(text);
    }
}

void JsonProfileEditor::setProfiles(const QVector<Profile>& profiles, const QString& defaultProfileId)
{
    _profiles = profiles;
    bool hasCustom = false;
    for (const auto& profile : _profiles) {
        if (profile.id == QStringLiteral("custom")) {
            hasCustom = true;
            break;
        }
    }
    if (!hasCustom) {
        Profile customProfile;
        customProfile.id = QStringLiteral("custom");
        customProfile.label = tr("Custom");
        customProfile.editable = true;
        _profiles.prepend(customProfile);
    }

    _profileCombo->clear();
    int defaultIndex = -1;
    for (int i = 0; i < _profiles.size(); ++i) {
        const auto& profile = _profiles[i];
        _profileCombo->addItem(profile.label, profile.id);
        if (defaultIndex == -1 && !defaultProfileId.isEmpty() && profile.id == defaultProfileId) {
            defaultIndex = i;
        }
    }

    if (_profileCombo->count() > 0) {
        if (defaultIndex < 0) {
            defaultIndex = _profileCombo->findData(QStringLiteral("custom"));
        }
        const int idx = defaultIndex >= 0 ? defaultIndex : 0;
        setProfile(_profileCombo->itemData(idx).toString(), false);
    }
}

void JsonProfileEditor::setProfile(const QString& profileId, bool fromUi)
{
    if (!_profileCombo) {
        return;
    }

    int idx = _profileCombo->findData(profileId);
    if (idx < 0) {
        idx = _profileCombo->findData(QStringLiteral("custom"));
    }
    if (idx < 0 && _profileCombo->count() > 0) {
        idx = 0;
    }

    if (idx >= 0) {
        const QSignalBlocker blocker(_profileCombo);
        _profileCombo->setCurrentIndex(idx);
        applyProfile(_profileCombo->itemData(idx).toString(), fromUi);
    }
}

QString JsonProfileEditor::profile() const
{
    return _profileId;
}

QString JsonProfileEditor::customText() const
{
    return _customText;
}

void JsonProfileEditor::setCustomText(const QString& text)
{
    _customText = text;
    if (_profileId != QStringLiteral("custom")) {
        return;
    }
    if (_textEdit) {
        const QSignalBlocker blocker(_textEdit);
        _textEdit->setPlainText(_customText);
    }
    validateCurrentText();
}

QString JsonProfileEditor::currentText() const
{
    return _textEdit ? _textEdit->toPlainText() : QString();
}

std::optional<QJsonObject> JsonProfileEditor::jsonObject(QString* error) const
{
    if (error) {
        error->clear();
    }

    const QString trimmed = currentText().trimmed();
    if (trimmed.isEmpty()) {
        return std::nullopt;
    }

    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(trimmed.toUtf8(), &parseError);
    if (parseError.error != QJsonParseError::NoError) {
        if (error) {
            *error = tr("Params JSON parse error (byte %1): %2")
                         .arg(static_cast<qulonglong>(parseError.offset))
                         .arg(parseError.errorString());
        }
        return std::nullopt;
    }

    if (!doc.isObject()) {
        if (error) {
            *error = tr("Params must be a JSON object.");
        }
        return std::nullopt;
    }

    return doc.object();
}

bool JsonProfileEditor::isValid() const
{
    return _errorText.isEmpty();
}

QString JsonProfileEditor::errorText() const
{
    return _errorText;
}

void JsonProfileEditor::applyProfile(const QString& profileId, bool fromUi)
{
    const Profile* profile = findProfile(profileId);
    const QString normalizedId = profile ? profile->id : QStringLiteral("custom");

    if (_profileId == normalizedId && !fromUi) {
        return;
    }

    _profileId = normalizedId;

    const bool editable = profile ? profile->editable : true;
    const QString text = editable ? _customText : (profile ? profile->jsonText : QString());

    if (_textEdit) {
        const QSignalBlocker blocker(_textEdit);
        _textEdit->setReadOnly(!editable);
        _textEdit->setPlainText(text);
    }

    validateCurrentText();
    emit profileChanged(_profileId);
}

void JsonProfileEditor::handleTextEdited()
{
    if (_updatingProgrammatically) {
        return;
    }

    if (_profileId != QStringLiteral("custom")) {
        return;
    }

    _customText = currentText();
    validateCurrentText();
    emit textChanged();
}

void JsonProfileEditor::validateCurrentText()
{
    QString error;
    jsonObject(&error);
    _errorText = error;
    updateStatusLabel();
}

const JsonProfileEditor::Profile* JsonProfileEditor::findProfile(const QString& id) const
{
    for (const auto& profile : _profiles) {
        if (profile.id == id) {
            return &profile;
        }
    }
    return nullptr;
}

void JsonProfileEditor::updateStatusLabel()
{
    if (!_statusLabel) {
        return;
    }
    if (_errorText.isEmpty()) {
        _statusLabel->clear();
        _statusLabel->setVisible(false);
        return;
    }
    _statusLabel->setText(_errorText);
    _statusLabel->setVisible(true);
}
