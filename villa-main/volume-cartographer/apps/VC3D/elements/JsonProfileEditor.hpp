#pragma once

#include <QGroupBox>
#include <QComboBox>
#include <QLabel>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QJsonObject>
#include <QVector>

#include <optional>

class JsonProfileEditor : public QGroupBox {
    Q_OBJECT
public:
    struct Profile {
        QString id;
        QString label;
        QString jsonText;
        bool editable{false};
    };

    explicit JsonProfileEditor(const QString& title, QWidget* parent = nullptr);

    void setDescription(const QString& text);
    void setPlaceholderText(const QString& text);
    void setTextToolTip(const QString& text);
    void setProfiles(const QVector<Profile>& profiles, const QString& defaultProfileId);
    void setProfile(const QString& profileId, bool fromUi = false);
    QString profile() const;
    QString customText() const;
    void setCustomText(const QString& text);
    QString currentText() const;

    std::optional<QJsonObject> jsonObject(QString* error) const;
    bool isValid() const;
    QString errorText() const;

    void loadUserProfiles();

signals:
    void profileChanged(const QString& profileId);
    void textChanged();
    void userProfilesChanged();

private:
    void applyProfile(const QString& profileId, bool fromUi);
    void handleTextEdited();
    void validateCurrentText();
    const Profile* findProfile(const QString& id) const;
    void updateStatusLabel();

    void onSaveAs();
    void onRename();
    void onDelete();
    void updateUserProfileButtons();
    bool isUserProfile() const;

    QLabel* _description{nullptr};
    QComboBox* _profileCombo{nullptr};
    QPlainTextEdit* _textEdit{nullptr};
    QLabel* _statusLabel{nullptr};

    QPushButton* _saveAsBtn{nullptr};
    QPushButton* _renameBtn{nullptr};
    QPushButton* _deleteBtn{nullptr};

    QVector<Profile> _profiles;
    QString _profileId{QStringLiteral("custom")};
    QString _customText;
    QString _errorText;
    bool _updatingProgrammatically{false};
};
