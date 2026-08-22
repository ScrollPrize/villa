#pragma once

#include <QJsonObject>
#include <QHash>
#include <QVector>
#include <QWidget>

class QComboBox;
class QDialog;
class QEvent;
class QLabel;
class QPlainTextEdit;
class QPushButton;
class QScrollArea;
class CollapsibleSettingsGroup;

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
    void setSessionDefaultLabel(const QString& label);
    void showSessionDefault();
    void clearSessionDefault();
    void setCatalog(const QJsonObject& catalog);
    void showWindow();

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
    void rebuildControls();
    void controlsToJson();
    void jsonToControls();
    void filterControls(const QString& text);

    struct ControlRow {
        QString key;
        QString searchableText;
        QWidget* widget = nullptr;
    };
    struct ControlGroup {
        QString prefix;
        CollapsibleSettingsGroup* widget = nullptr;
        QVector<ControlRow> rows;
    };

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
    QJsonObject _catalog;
    QWidget* _controlsPage = nullptr;
    QScrollArea* _controlsScroll = nullptr;
    QWidget* _controlsGrid = nullptr;
    class QLineEdit* _search = nullptr;
    QHash<QString, QWidget*> _fieldEditors;
    QVector<ControlGroup> _controlGroups;
    QHash<CollapsibleSettingsGroup*, bool> _preSearchExpanded;

    QVector<StoredProfile> _profiles;
    QString _currentProfileId = QStringLiteral("default");
    QString _sessionDefaultLabel;
    QString _sessionDefaultText = QStringLiteral("{}");
    QString _customText = QStringLiteral("{}");
    QString _cleanText = QStringLiteral("{}");
    QString _errorText;
    bool _dirty = false;
    bool _programmatic = false;
    bool _poppedOut = false;
};
