#include "SegmentationCustomParamsPanel.hpp"

#include "elements/JsonProfileEditor.hpp"
#include "elements/JsonProfilePresets.hpp"
#include "VCSettings.hpp"

#include <QByteArray>
#include <QSettings>
#include <QSignalBlocker>
#include <QVBoxLayout>
#include <QVariant>

#include <exception>

#include <nlohmann/json.hpp>

SegmentationCustomParamsPanel::SegmentationCustomParamsPanel(const QString& settingsGroup,
                                                             QWidget* parent)
    : QWidget(parent)
    , _settingsGroup(settingsGroup)
{
    auto* panelLayout = new QVBoxLayout(this);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(0);

    _customParamsEditor = new JsonProfileEditor(tr("Custom Params"), this);
    _customParamsEditor->setDescription(
        tr("Additional JSON fields merge into the tracer params. Leave empty for defaults."));
    _customParamsEditor->setPlaceholderText(QStringLiteral("{\n    \"example_param\": 1\n}"));
    _customParamsEditor->setTextToolTip(
        tr("Optional JSON that merges into tracer parameters before growth."));

    const auto profiles = vc3d::json_profiles::tracerParamProfiles(
        [this](const char* text) { return tr(text); });
    _customParamsEditor->setProfiles(profiles, QStringLiteral("custom"));

    panelLayout->addWidget(_customParamsEditor);

    // --- Signal wiring ---

    connect(_customParamsEditor, &JsonProfileEditor::textChanged, this, [this]() {
        handleCustomParamsEdited();
    });

    connect(_customParamsEditor, &JsonProfileEditor::profileChanged, this, [this](const QString& profile) {
        if (_restoringSettings) {
            return;
        }
        applyCustomParamsProfile(profile, /*persist=*/true, /*fromUi=*/true);
    });
}

void SegmentationCustomParamsPanel::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(_settingsGroup);
    settings.setValue(key, value);
    settings.endGroup();
}

QString SegmentationCustomParamsPanel::customParamsText() const
{
    return paramsTextForProfile(_customParamsProfile);
}

QString SegmentationCustomParamsPanel::paramsTextForProfile(const QString& profile) const
{
    if (profile == QStringLiteral("custom")) {
        return _customParamsText;
    }
    if (profile == QStringLiteral("default")) {
        // Empty => use GrowPatch defaults.
        return QString();
    }
    return vc3d::json_profiles::tracerParamProfileJson(profile);
}

void SegmentationCustomParamsPanel::handleCustomParamsEdited()
{
    if (!_customParamsEditor) {
        return;
    }

    // Edits only allowed in custom profile (UI should already be read-only otherwise).
    if (_customParamsProfile != QStringLiteral("custom")) {
        return;
    }

    _customParamsText = _customParamsEditor->customText();
    writeSetting(QStringLiteral("custom_params_text"), _customParamsText);
    validateCustomParamsText();
}

void SegmentationCustomParamsPanel::validateCustomParamsText()
{
    if (_customParamsEditor) {
        _customParamsError = _customParamsEditor->errorText();
        return;
    }

    QString error;
    parseCustomParams(&error);
    _customParamsError = error;
}

std::optional<nlohmann::json> SegmentationCustomParamsPanel::parseCustomParams(QString* error) const
{
    if (error) {
        error->clear();
    }

    const QString trimmed = paramsTextForProfile(_customParamsProfile).trimmed();
    if (trimmed.isEmpty()) {
        return std::nullopt;
    }

    try {
        const QByteArray utf8 = trimmed.toUtf8();
        nlohmann::json parsed = nlohmann::json::parse(utf8.constData(), utf8.constData() + utf8.size());
        if (!parsed.is_object()) {
            if (error) {
                *error = tr("Custom params must be a JSON object.");
            }
            return std::nullopt;
        }
        return parsed;
    } catch (const nlohmann::json::parse_error& ex) {
        if (error) {
            *error = tr("Custom params JSON parse error (byte %1): %2")
                         .arg(static_cast<qulonglong>(ex.byte))
                         .arg(QString::fromStdString(ex.what()));
        }
    } catch (const std::exception& ex) {
        if (error) {
            *error = tr("Custom params JSON parse error: %1")
                         .arg(QString::fromStdString(ex.what()));
        }
    } catch (...) {
        if (error) {
            *error = tr("Custom params JSON parse error: unknown error");
        }
    }

    return std::nullopt;
}

std::optional<nlohmann::json> SegmentationCustomParamsPanel::customParamsJson() const
{
    QString error;
    auto parsed = parseCustomParams(&error);
    if (!error.isEmpty()) {
        return std::nullopt;
    }
    return parsed;
}

void SegmentationCustomParamsPanel::applyCustomParamsProfile(const QString& profile, bool persist, bool fromUi)
{
    const QString normalized = vc3d::json_profiles::isValidProfileId(profile)
        ? profile
        : QStringLiteral("custom");

    _customParamsProfile = normalized;
    if (persist) {
        writeSetting(QStringLiteral("custom_params_profile"), _customParamsProfile);
    }

    if (_customParamsEditor && !fromUi) {
        const QSignalBlocker blocker(_customParamsEditor);
        _customParamsEditor->setProfile(_customParamsProfile, false);
    }

    validateCustomParamsText();
}

void SegmentationCustomParamsPanel::restoreSettings(QSettings& settings)
{
    using namespace vc3d::settings;
    _restoringSettings = true;

    _customParamsText = settings.value(segmentation::CUSTOM_PARAMS_TEXT, QString()).toString();
    _customParamsProfile = settings.value(QStringLiteral("custom_params_profile"), _customParamsProfile).toString();
    if (!vc3d::json_profiles::isValidProfileId(_customParamsProfile)) {
        _customParamsProfile = QStringLiteral("custom");
    }
    if (_customParamsEditor) {
        _customParamsEditor->setCustomText(_customParamsText);
    }

    // Apply profile behavior after restoring.
    applyCustomParamsProfile(_customParamsProfile, /*persist=*/false, /*fromUi=*/false);

    _restoringSettings = false;
}

void SegmentationCustomParamsPanel::syncUiState(bool editingEnabled)
{
    if (_customParamsEditor) {
        if (_customParamsEditor->customText() != _customParamsText) {
            _customParamsEditor->setCustomText(_customParamsText);
        }
        if (_customParamsEditor->profile() != _customParamsProfile) {
            const QSignalBlocker blocker(_customParamsEditor);
            _customParamsEditor->setProfile(_customParamsProfile, false);
        }
        _customParamsEditor->setEnabled(editingEnabled);
    }
    validateCustomParamsText();
}
