#pragma once

#include <QJsonObject>
#include <QVector>
#include <QWidget>

class QComboBox;
class QDialog;
class QEvent;
class QLabel;
class QPlainTextEdit;
class QPushButton;

class SpiralConfigProfileEditor final : public QWidget
{
    Q_OBJECT
public:
    explicit SpiralConfigProfileEditor(QWidget* parent = nullptr);
    ~SpiralConfigProfileEditor() override;

    QPlainTextEdit* textEdit() const { return _textEdit; }
    QString currentText() const;
    QString currentProfileId() const { return _currentProfileId; }
    bool isDefaultProfile() const;
    bool isValid() const { return _errorText.isEmpty(); }
    QString errorText() const { return _errorText; }

    void setCurrentText(const QString& text);
    void setSessionDefault(const QJsonObject& config);
    void clearSessionDefault();

signals:
    void textChanged();
    void profileChanged(const QString& profileId);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    struct StoredProfile {
        QString id;
        QString name;
        QString jsonText;
    };

    void loadProfiles();
    bool writeProfiles();
    void rebuildCombo();
    void selectProfile(const QString& profileId, bool fromUi);
    void applyCurrentProfileText();
    void handleTextEdited();
    void validateCurrentText();
    void updateUi();
    bool confirmDirtyTransition();

    bool saveCurrent();
    bool saveCurrentAs();
    void renameCurrent();
    void deleteCurrent();
    bool validProfileName(const QString& name, const QString& exceptId,
                          QString* error) const;
    StoredProfile* findStored(const QString& id);
    const StoredProfile* findStored(const QString& id) const;

    void popOut();
    void popIn();

    QWidget* _editorContents = nullptr;
    QComboBox* _profileCombo = nullptr;
    QPushButton* _saveButton = nullptr;
    QPushButton* _saveAsButton = nullptr;
    QPushButton* _renameButton = nullptr;
    QPushButton* _deleteButton = nullptr;
    QPushButton* _popButton = nullptr;
    QPushButton* _inlinePopInButton = nullptr;
    QPlainTextEdit* _textEdit = nullptr;
    QLabel* _statusLabel = nullptr;
    QDialog* _dialog = nullptr;

    QVector<StoredProfile> _profiles;
    QString _currentProfileId = QStringLiteral("default");
    QString _sessionDefaultText = QStringLiteral("{}");
    QString _customText = QStringLiteral("{}");
    QString _cleanText = QStringLiteral("{}");
    QString _errorText;
    bool _dirty = false;
    bool _programmatic = false;
    bool _poppedOut = false;
};
