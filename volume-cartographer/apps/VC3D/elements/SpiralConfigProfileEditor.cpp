#include "elements/SpiralConfigProfileEditor.hpp"

#include "VCSettings.hpp"

#include <QComboBox>
#include <QDialog>
#include <QEvent>
#include <QFile>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonParseError>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSignalBlocker>
#include <QUuid>
#include <QVBoxLayout>

#include <algorithm>

namespace {
const QString kDefaultId = QStringLiteral("default");
const QString kCustomId = QStringLiteral("custom");

QString profileStorePath()
{
    return QFileInfo(vc3d::settingsFilePath()).dir().filePath(
        QStringLiteral("spiral-advanced-profiles.json"));
}

class VerticallyResizablePlainTextEdit final : public QPlainTextEdit
{
public:
    explicit VerticallyResizablePlainTextEdit(QWidget* parent = nullptr)
        : QPlainTextEdit(parent)
    {
        setMouseTracking(true);
        setMinimumHeight(72);
    }

    QSize sizeHint() const override
    {
        QSize result = QPlainTextEdit::sizeHint();
        result.setHeight(120);
        return result;
    }

protected:
    void mousePressEvent(QMouseEvent* event) override
    {
        if (event->button() == Qt::LeftButton && onResizeEdge(event->position())) {
            _resizing = true;
            _dragStartY = event->globalPosition().y();
            _dragStartHeight = height();
            event->accept();
            return;
        }
        QPlainTextEdit::mousePressEvent(event);
    }

    void mouseMoveEvent(QMouseEvent* event) override
    {
        if (_resizing) {
            const int requested = _dragStartHeight
                + qRound(event->globalPosition().y() - _dragStartY);
            setFixedHeight(qMax(72, requested));
            event->accept();
            return;
        }
        viewport()->setCursor(onResizeEdge(event->position())
                                  ? Qt::SizeVerCursor : Qt::IBeamCursor);
        QPlainTextEdit::mouseMoveEvent(event);
    }

    void mouseReleaseEvent(QMouseEvent* event) override
    {
        if (_resizing && event->button() == Qt::LeftButton) {
            _resizing = false;
            event->accept();
            return;
        }
        QPlainTextEdit::mouseReleaseEvent(event);
    }

private:
    bool onResizeEdge(const QPointF& position) const
    {
        return position.y() >= viewport()->height() - 7;
    }

    bool _resizing = false;
    qreal _dragStartY = 0;
    int _dragStartHeight = 0;
};

QString formatted(const QJsonObject& object)
{
    return QString::fromUtf8(QJsonDocument(object).toJson(QJsonDocument::Indented)).trimmed();
}
}

SpiralConfigProfileEditor::SpiralConfigProfileEditor(QWidget* parent)
    : QWidget(parent)
{
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);

    _editorContents = new QWidget(this);
    auto* editorLayout = new QVBoxLayout(_editorContents);
    editorLayout->setContentsMargins(0, 0, 0, 0);

    auto* profileRow = new QHBoxLayout;
    profileRow->addWidget(new QLabel(tr("Profile"), _editorContents));
    _profileCombo = new QComboBox(_editorContents);
    _profileCombo->setObjectName(QStringLiteral("spiralAdvancedProfileCombo"));
    _profileCombo->setToolTip(tr("Default uses the active Python/checkpoint configuration. "
                                 "Saved profiles are available to every Spiral connection."));
    profileRow->addWidget(_profileCombo, 1);
    _popButton = new QPushButton(tr("Pop Out"), _editorContents);
    _popButton->setObjectName(QStringLiteral("spiralAdvancedPopOut"));
    profileRow->addWidget(_popButton);
    editorLayout->addLayout(profileRow);

    auto* buttonRow = new QHBoxLayout;
    _saveButton = new QPushButton(tr("Save"), _editorContents);
    _saveAsButton = new QPushButton(tr("Save As…"), _editorContents);
    _renameButton = new QPushButton(tr("Rename…"), _editorContents);
    _deleteButton = new QPushButton(tr("Delete"), _editorContents);
    _saveButton->setObjectName(QStringLiteral("spiralAdvancedProfileSave"));
    _saveAsButton->setObjectName(QStringLiteral("spiralAdvancedProfileSaveAs"));
    _renameButton->setObjectName(QStringLiteral("spiralAdvancedProfileRename"));
    _deleteButton->setObjectName(QStringLiteral("spiralAdvancedProfileDelete"));
    buttonRow->addWidget(_saveButton);
    buttonRow->addWidget(_saveAsButton);
    buttonRow->addWidget(_renameButton);
    buttonRow->addWidget(_deleteButton);
    buttonRow->addStretch(1);
    editorLayout->addLayout(buttonRow);

    _textEdit = new VerticallyResizablePlainTextEdit(_editorContents);
    _textEdit->setObjectName(QStringLiteral("spiralAdvancedJsonEditor"));
    _textEdit->setPlainText(QStringLiteral("{}"));
    _textEdit->setTabChangesFocus(true);
    _textEdit->setToolTip(tr("Sampling counts, loss weights, and loss start steps apply to the "
                             "next Run. Drag the bottom edge to resize vertically."));
    editorLayout->addWidget(_textEdit);

    _statusLabel = new QLabel(_editorContents);
    _statusLabel->setWordWrap(true);
    _statusLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
    _statusLabel->hide();
    editorLayout->addWidget(_statusLabel);
    root->addWidget(_editorContents);

    _inlinePopInButton = new QPushButton(tr("Advanced JSON is open in a separate window — Pop In"), this);
    _inlinePopInButton->setObjectName(QStringLiteral("spiralAdvancedPopIn"));
    _inlinePopInButton->hide();
    root->addWidget(_inlinePopInButton);

    _dialog = new QDialog(this, Qt::Window);
    _dialog->setObjectName(QStringLiteral("spiralAdvancedConfigDialog"));
    _dialog->setWindowTitle(tr("Spiral Advanced Config JSON"));
    _dialog->setModal(false);
    _dialog->resize(720, 620);
    _dialog->setLayout(new QVBoxLayout);
    _dialog->installEventFilter(this);

    connect(_profileCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this](int index) {
                if (_programmatic || index < 0) return;
                selectProfile(_profileCombo->itemData(index).toString(), true);
            });
    connect(_textEdit, &QPlainTextEdit::textChanged,
            this, &SpiralConfigProfileEditor::handleTextEdited);
    connect(_saveButton, &QPushButton::clicked, this, [this]() { saveCurrent(); });
    connect(_saveAsButton, &QPushButton::clicked, this, [this]() { saveCurrentAs(); });
    connect(_renameButton, &QPushButton::clicked,
            this, &SpiralConfigProfileEditor::renameCurrent);
    connect(_deleteButton, &QPushButton::clicked,
            this, &SpiralConfigProfileEditor::deleteCurrent);
    connect(_popButton, &QPushButton::clicked, this, [this]() {
        _poppedOut ? popIn() : popOut();
    });
    connect(_inlinePopInButton, &QPushButton::clicked,
            this, &SpiralConfigProfileEditor::popIn);

    loadProfiles();
    rebuildCombo();
    applyCurrentProfileText();
}

SpiralConfigProfileEditor::~SpiralConfigProfileEditor()
{
    if (_dialog) _dialog->removeEventFilter(this);
}

QString SpiralConfigProfileEditor::currentText() const
{
    return _textEdit->toPlainText();
}

bool SpiralConfigProfileEditor::isDefaultProfile() const
{
    return _currentProfileId == kDefaultId;
}

void SpiralConfigProfileEditor::setCurrentText(const QString& text)
{
    if (text == currentText()) return;
    _textEdit->setPlainText(text);
}

void SpiralConfigProfileEditor::setSessionDefault(const QJsonObject& config)
{
    _sessionDefaultText = formatted(config);
    if (!isDefaultProfile()) return;
    _programmatic = true;
    _textEdit->setPlainText(_sessionDefaultText);
    _programmatic = false;
    _cleanText = _sessionDefaultText;
    _dirty = false;
    validateCurrentText();
    updateUi();
    emit textChanged();
}

void SpiralConfigProfileEditor::clearSessionDefault()
{
    _sessionDefaultText = QStringLiteral("{}");
    if (isDefaultProfile()) setSessionDefault(QJsonObject{});
}

bool SpiralConfigProfileEditor::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == _dialog && event->type() == QEvent::Close) {
        popIn();
        return true;
    }
    return QWidget::eventFilter(watched, event);
}

void SpiralConfigProfileEditor::loadProfiles()
{
    _profiles.clear();
    QFile file(profileStorePath());
    if (file.open(QIODevice::ReadOnly)) {
        const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
        for (const QJsonValue& value : document.object()
                 .value(QStringLiteral("profiles")).toArray()) {
            const QJsonObject object = value.toObject();
            StoredProfile profile{
                object.value(QStringLiteral("id")).toString(),
                object.value(QStringLiteral("name")).toString().trimmed(),
                object.value(QStringLiteral("json")).toString()};
            if (!profile.id.isEmpty() && !profile.name.isEmpty())
                _profiles.push_back(std::move(profile));
        }
    }
    std::sort(_profiles.begin(), _profiles.end(), [](const auto& left, const auto& right) {
        return QString::localeAwareCompare(left.name, right.name) < 0;
    });
}

bool SpiralConfigProfileEditor::writeProfiles()
{
    QJsonArray profiles;
    for (const StoredProfile& profile : _profiles) {
        profiles.append(QJsonObject{
            {QStringLiteral("id"), profile.id},
            {QStringLiteral("name"), profile.name},
            {QStringLiteral("json"), profile.jsonText},
        });
    }
    QFile file(profileStorePath());
    if (!file.open(QIODevice::WriteOnly)) return false;
    const QJsonObject root{
        {QStringLiteral("version"), 1},
        {QStringLiteral("profiles"), profiles},
    };
    return file.write(QJsonDocument(root).toJson(QJsonDocument::Indented)) >= 0;
}

void SpiralConfigProfileEditor::rebuildCombo()
{
    _programmatic = true;
    const QSignalBlocker blocker(_profileCombo);
    _profileCombo->clear();
    _profileCombo->addItem(tr("Default"), kDefaultId);
    _profileCombo->addItem(tr("Custom"), kCustomId);
    for (const StoredProfile& profile : _profiles)
        _profileCombo->addItem(profile.name, profile.id);
    const int index = _profileCombo->findData(_currentProfileId);
    _profileCombo->setCurrentIndex(index >= 0 ? index : 0);
    _programmatic = false;
    updateUi();
}

void SpiralConfigProfileEditor::selectProfile(const QString& profileId, bool fromUi)
{
    if (profileId == _currentProfileId) return;
    const QString previous = _currentProfileId;
    if (fromUi && !confirmDirtyTransition()) {
        const QSignalBlocker blocker(_profileCombo);
        _profileCombo->setCurrentIndex(_profileCombo->findData(previous));
        return;
    }
    _currentProfileId = profileId;
    applyCurrentProfileText();
    rebuildCombo();
    emit profileChanged(_currentProfileId);
    emit textChanged();
}

void SpiralConfigProfileEditor::applyCurrentProfileText()
{
    QString text;
    if (_currentProfileId == kDefaultId) text = _sessionDefaultText;
    else if (_currentProfileId == kCustomId) text = _customText;
    else if (const StoredProfile* profile = findStored(_currentProfileId)) text = profile->jsonText;
    else {
        _currentProfileId = kDefaultId;
        text = _sessionDefaultText;
    }
    _programmatic = true;
    _textEdit->setPlainText(text);
    _programmatic = false;
    _cleanText = text;
    _dirty = false;
    validateCurrentText();
    updateUi();
}

void SpiralConfigProfileEditor::handleTextEdited()
{
    if (_programmatic) return;
    const QString text = currentText();
    if (_currentProfileId == kDefaultId) {
        _currentProfileId = kCustomId;
        _customText = text;
        _cleanText = QStringLiteral("{}");
        rebuildCombo();
        emit profileChanged(_currentProfileId);
    } else if (_currentProfileId == kCustomId) {
        _customText = text;
    }
    _dirty = text != _cleanText;
    validateCurrentText();
    updateUi();
    emit textChanged();
}

void SpiralConfigProfileEditor::validateCurrentText()
{
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(currentText().toUtf8(), &parseError);
    if (parseError.error != QJsonParseError::NoError)
        _errorText = tr("JSON parse error at byte %1: %2")
                         .arg(parseError.offset).arg(parseError.errorString());
    else if (!document.isObject())
        _errorText = tr("Advanced config must be a JSON object.");
    else
        _errorText.clear();
    _statusLabel->setText(_errorText);
    _statusLabel->setVisible(!_errorText.isEmpty());
}

void SpiralConfigProfileEditor::updateUi()
{
    const bool stored = findStored(_currentProfileId) != nullptr;
    _saveButton->setEnabled(stored ? _dirty : isValid());
    _renameButton->setVisible(stored);
    _deleteButton->setVisible(stored);
    const int index = _profileCombo->findData(_currentProfileId);
    if (index >= 0) {
        QString label;
        if (_currentProfileId == kDefaultId) label = tr("Default");
        else if (_currentProfileId == kCustomId) label = tr("Custom");
        else if (const StoredProfile* profile = findStored(_currentProfileId)) label = profile->name;
        if (_dirty) label += QStringLiteral(" *");
        _profileCombo->setItemText(index, label);
    }
}

bool SpiralConfigProfileEditor::confirmDirtyTransition()
{
    if (!_dirty) return true;
    const QMessageBox::StandardButton answer = QMessageBox::warning(
        this, tr("Unsaved Advanced config"),
        tr("Save changes to the current Advanced JSON profile?"),
        QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
        QMessageBox::Cancel);
    if (answer == QMessageBox::Cancel) return false;
    if (answer == QMessageBox::Save) return saveCurrent();
    if (_currentProfileId == kCustomId) _customText = QStringLiteral("{}");
    return true;
}

bool SpiralConfigProfileEditor::saveCurrent()
{
    validateCurrentText();
    if (!isValid()) {
        QMessageBox::warning(this, tr("Invalid Advanced config"), _errorText);
        return false;
    }
    StoredProfile* profile = findStored(_currentProfileId);
    if (!profile) return saveCurrentAs();
    profile->jsonText = currentText();
    if (!writeProfiles()) {
        QMessageBox::warning(this, tr("Could not save profile"),
                             tr("Could not write %1").arg(profileStorePath()));
        return false;
    }
    _cleanText = profile->jsonText;
    _dirty = false;
    updateUi();
    return true;
}

bool SpiralConfigProfileEditor::saveCurrentAs()
{
    validateCurrentText();
    if (!isValid()) {
        QMessageBox::warning(this, tr("Invalid Advanced config"), _errorText);
        return false;
    }
    bool ok = false;
    const QString name = QInputDialog::getText(
        this, tr("Save Advanced JSON Profile"), tr("Profile name:"),
        QLineEdit::Normal, QString(), &ok).trimmed();
    if (!ok) return false;
    QString error;
    if (!validProfileName(name, {}, &error)) {
        QMessageBox::warning(this, tr("Invalid profile name"), error);
        return false;
    }
    StoredProfile profile{
        QUuid::createUuid().toString(QUuid::WithoutBraces), name, currentText()};
    _profiles.push_back(profile);
    if (!writeProfiles()) {
        _profiles.removeLast();
        QMessageBox::warning(this, tr("Could not save profile"),
                             tr("Could not write %1").arg(profileStorePath()));
        return false;
    }
    _currentProfileId = profile.id;
    _cleanText = profile.jsonText;
    _dirty = false;
    loadProfiles();
    rebuildCombo();
    emit profileChanged(_currentProfileId);
    return true;
}

void SpiralConfigProfileEditor::renameCurrent()
{
    StoredProfile* profile = findStored(_currentProfileId);
    if (!profile) return;
    bool ok = false;
    const QString name = QInputDialog::getText(
        this, tr("Rename Advanced JSON Profile"), tr("New name:"),
        QLineEdit::Normal, profile->name, &ok).trimmed();
    if (!ok) return;
    QString error;
    if (!validProfileName(name, profile->id, &error)) {
        QMessageBox::warning(this, tr("Invalid profile name"), error);
        return;
    }
    profile->name = name;
    if (!writeProfiles()) {
        QMessageBox::warning(this, tr("Could not rename profile"),
                             tr("Could not write %1").arg(profileStorePath()));
        loadProfiles();
        return;
    }
    loadProfiles();
    rebuildCombo();
}

void SpiralConfigProfileEditor::deleteCurrent()
{
    const StoredProfile* profile = findStored(_currentProfileId);
    if (!profile) return;
    if (QMessageBox::question(this, tr("Delete Advanced JSON Profile"),
                              tr("Delete profile \"%1\"?").arg(profile->name),
                              QMessageBox::Yes | QMessageBox::No,
                              QMessageBox::No) != QMessageBox::Yes)
        return;
    const QString removedId = _currentProfileId;
    _profiles.erase(std::remove_if(_profiles.begin(), _profiles.end(),
                                   [&removedId](const StoredProfile& item) {
                                       return item.id == removedId;
                                   }), _profiles.end());
    if (!writeProfiles()) {
        QMessageBox::warning(this, tr("Could not delete profile"),
                             tr("Could not write %1").arg(profileStorePath()));
        loadProfiles();
        return;
    }
    _currentProfileId = kDefaultId;
    loadProfiles();
    applyCurrentProfileText();
    rebuildCombo();
    emit profileChanged(_currentProfileId);
    emit textChanged();
}

bool SpiralConfigProfileEditor::validProfileName(
    const QString& name, const QString& exceptId, QString* error) const
{
    if (name.trimmed().isEmpty()) {
        if (error) *error = tr("Profile name cannot be empty.");
        return false;
    }
    for (const StoredProfile& profile : _profiles) {
        if (profile.id != exceptId
            && profile.name.compare(name.trimmed(), Qt::CaseInsensitive) == 0) {
            if (error) *error = tr("A profile named \"%1\" already exists.").arg(name.trimmed());
            return false;
        }
    }
    return true;
}

SpiralConfigProfileEditor::StoredProfile*
SpiralConfigProfileEditor::findStored(const QString& id)
{
    for (StoredProfile& profile : _profiles) if (profile.id == id) return &profile;
    return nullptr;
}

const SpiralConfigProfileEditor::StoredProfile*
SpiralConfigProfileEditor::findStored(const QString& id) const
{
    for (const StoredProfile& profile : _profiles) if (profile.id == id) return &profile;
    return nullptr;
}

void SpiralConfigProfileEditor::popOut()
{
    if (_poppedOut) return;
    // Reparenting the live controls is presentation-only.  Some Qt platform
    // styles emit editor/combo notifications while a focused widget changes
    // native windows; those must not turn Default into Custom or make the
    // containing Spiral panel compare a fictitious config edit.
    const QSignalBlocker textBlocker(_textEdit);
    const QSignalBlocker profileBlocker(_profileCombo);
    const bool wasProgrammatic = _programmatic;
    _programmatic = true;
    layout()->removeWidget(_editorContents);
    _editorContents->setParent(_dialog);
    _dialog->layout()->addWidget(_editorContents);
    _inlinePopInButton->show();
    _popButton->setText(tr("Pop In"));
    _poppedOut = true;
    _dialog->show();
    _dialog->raise();
    _dialog->activateWindow();
    _programmatic = wasProgrammatic;
}

void SpiralConfigProfileEditor::popIn()
{
    if (!_poppedOut) return;
    const QSignalBlocker textBlocker(_textEdit);
    const QSignalBlocker profileBlocker(_profileCombo);
    const bool wasProgrammatic = _programmatic;
    _programmatic = true;
    _dialog->layout()->removeWidget(_editorContents);
    _editorContents->setParent(this);
    static_cast<QVBoxLayout*>(layout())->insertWidget(0, _editorContents);
    _inlinePopInButton->hide();
    _popButton->setText(tr("Pop Out"));
    _poppedOut = false;
    _dialog->hide();
    _programmatic = wasProgrammatic;
}
