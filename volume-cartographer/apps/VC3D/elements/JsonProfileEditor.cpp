#include "elements/JsonProfileEditor.hpp"
#include "elements/UserProfileStore.hpp"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSignalBlocker>
#include <QInputDialog>
#include <QMessageBox>

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
                                 "- Default/Robust: auto-filled and read-only\n"
                                 "- User profiles: editable, saved across sessions"));
    profileRow->addWidget(profileLabel);
    profileRow->addWidget(_profileCombo, 1);
    layout->addLayout(profileRow);

    auto* buttonRow = new QHBoxLayout();
    _saveAsBtn = new QPushButton(tr("Save As..."), this);
    _renameBtn = new QPushButton(tr("Rename..."), this);
    _deleteBtn = new QPushButton(tr("Delete"), this);
    buttonRow->addWidget(_saveAsBtn);
    buttonRow->addWidget(_renameBtn);
    buttonRow->addWidget(_deleteBtn);
    buttonRow->addStretch();
    layout->addLayout(buttonRow);

    connect(_saveAsBtn, &QPushButton::clicked, this, &JsonProfileEditor::onSaveAs);
    connect(_renameBtn, &QPushButton::clicked, this, &JsonProfileEditor::onRename);
    connect(_deleteBtn, &QPushButton::clicked, this, &JsonProfileEditor::onDelete);

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

    updateUserProfileButtons();
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

    loadUserProfiles();

    {
        const QSignalBlocker blocker(_profileCombo);
        _profileCombo->clear();
        for (int i = 0; i < _profiles.size(); ++i) {
            const auto& profile = _profiles[i];
            _profileCombo->addItem(profile.label, profile.id);
        }
    }

    int defaultIndex = -1;
    for (int i = 0; i < _profiles.size(); ++i) {
        if (!defaultProfileId.isEmpty() && _profiles[i].id == defaultProfileId) {
            defaultIndex = i;
            break;
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

void JsonProfileEditor::loadUserProfiles()
{
    // Remove existing user profiles from _profiles
    for (int i = _profiles.size() - 1; i >= 0; --i) {
        if (vc3d::user_profiles::isUserProfileId(_profiles[i].id)) {
            _profiles.removeAt(i);
        }
    }

    // Load from store
    const auto userProfiles = vc3d::user_profiles::loadAll();

    // Insert after Custom (index 0), before built-in presets
    int insertPos = 1;
    for (const auto& up : userProfiles) {
        Profile p;
        p.id = vc3d::user_profiles::profileIdFromStorageId(up.storageId);
        p.label = up.name;
        p.jsonText = up.jsonText;
        p.editable = true;
        _profiles.insert(insertPos, p);
        ++insertPos;
    }
}

void JsonProfileEditor::applyProfile(const QString& profileId, bool fromUi)
{
    const Profile* profile = findProfile(profileId);
    const QString normalizedId = profile ? profile->id : QStringLiteral("custom");

    if (_profileId == normalizedId && !fromUi) {
        return;
    }

    _profileId = normalizedId;

    const bool isCustom = (_profileId == QStringLiteral("custom"));
    const bool isUser = vc3d::user_profiles::isUserProfileId(_profileId);
    const bool editable = isCustom || isUser;
    const QString text = isCustom ? _customText : (profile ? profile->jsonText : QString());

    if (_textEdit) {
        const QSignalBlocker blocker(_textEdit);
        _textEdit->setReadOnly(!editable);
        _textEdit->setPlainText(text);
    }

    validateCurrentText();
    updateUserProfileButtons();
    emit profileChanged(_profileId);
}

void JsonProfileEditor::handleTextEdited()
{
    if (_updatingProgrammatically) {
        return;
    }

    const bool isCustom = (_profileId == QStringLiteral("custom"));
    const bool isUser = vc3d::user_profiles::isUserProfileId(_profileId);

    if (!isCustom && !isUser) {
        return;
    }

    if (isCustom) {
        _customText = currentText();
    } else {
        // Auto-save user profile text
        const int storageId = vc3d::user_profiles::storageIdFromProfileId(_profileId);
        if (storageId > 0) {
            const Profile* p = findProfile(_profileId);
            const QString name = p ? p->label : QString();
            const QString text = currentText();
            vc3d::user_profiles::update(storageId, name, text);

            // Update in-memory profile
            for (auto& profile : _profiles) {
                if (profile.id == _profileId) {
                    profile.jsonText = text;
                    break;
                }
            }
        }
    }

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

void JsonProfileEditor::onSaveAs()
{
    bool ok = false;
    const QString name = QInputDialog::getText(
        this, tr("Save Profile As"), tr("Profile name:"),
        QLineEdit::Normal, QString(), &ok);
    if (!ok || name.trimmed().isEmpty()) {
        return;
    }

    const QString jsonText = currentText();
    const int storageId = vc3d::user_profiles::save(name.trimmed(), jsonText);
    const QString profileId = vc3d::user_profiles::profileIdFromStorageId(storageId);

    // Reload profiles into combo
    loadUserProfiles();
    {
        const QSignalBlocker blocker(_profileCombo);
        _profileCombo->clear();
        for (const auto& profile : _profiles) {
            _profileCombo->addItem(profile.label, profile.id);
        }
    }

    setProfile(profileId, false);
    emit userProfilesChanged();
}

void JsonProfileEditor::onRename()
{
    if (!isUserProfile()) {
        return;
    }

    const Profile* current = findProfile(_profileId);
    if (!current) {
        return;
    }

    bool ok = false;
    const QString name = QInputDialog::getText(
        this, tr("Rename Profile"), tr("New name:"),
        QLineEdit::Normal, current->label, &ok);
    if (!ok || name.trimmed().isEmpty()) {
        return;
    }

    const int storageId = vc3d::user_profiles::storageIdFromProfileId(_profileId);
    if (storageId <= 0) {
        return;
    }

    vc3d::user_profiles::update(storageId, name.trimmed(), current->jsonText);

    // Reload profiles into combo
    const QString selectedId = _profileId;
    loadUserProfiles();
    {
        const QSignalBlocker blocker(_profileCombo);
        _profileCombo->clear();
        for (const auto& profile : _profiles) {
            _profileCombo->addItem(profile.label, profile.id);
        }
    }

    setProfile(selectedId, false);
    emit userProfilesChanged();
}

void JsonProfileEditor::onDelete()
{
    if (!isUserProfile()) {
        return;
    }

    const Profile* current = findProfile(_profileId);
    if (!current) {
        return;
    }

    const auto result = QMessageBox::question(
        this, tr("Delete Profile"),
        tr("Delete profile \"%1\"?").arg(current->label),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (result != QMessageBox::Yes) {
        return;
    }

    const int storageId = vc3d::user_profiles::storageIdFromProfileId(_profileId);
    if (storageId <= 0) {
        return;
    }

    vc3d::user_profiles::remove(storageId);

    // Reload profiles into combo
    loadUserProfiles();
    {
        const QSignalBlocker blocker(_profileCombo);
        _profileCombo->clear();
        for (const auto& profile : _profiles) {
            _profileCombo->addItem(profile.label, profile.id);
        }
    }

    setProfile(QStringLiteral("custom"), false);
    emit userProfilesChanged();
}

void JsonProfileEditor::updateUserProfileButtons()
{
    const bool isUser = isUserProfile();
    if (_renameBtn) {
        _renameBtn->setVisible(isUser);
    }
    if (_deleteBtn) {
        _deleteBtn->setVisible(isUser);
    }
}

bool JsonProfileEditor::isUserProfile() const
{
    return vc3d::user_profiles::isUserProfileId(_profileId);
}
