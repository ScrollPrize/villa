#pragma once

#include <QWidget>

#include <optional>

#include <nlohmann/json_fwd.hpp>

class JsonProfileEditor;
class QSettings;

class SegmentationCustomParamsPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationCustomParamsPanel(const QString& settingsGroup,
                                           QWidget* parent = nullptr);

    [[nodiscard]] QString customParamsText() const;
    [[nodiscard]] QString customParamsProfile() const { return _customParamsProfile; }
    [[nodiscard]] bool customParamsValid() const { return _customParamsError.isEmpty(); }
    [[nodiscard]] QString customParamsError() const { return _customParamsError; }
    [[nodiscard]] std::optional<nlohmann::json> customParamsJson() const;

    void restoreSettings(QSettings& settings);
    void syncUiState(bool editingEnabled);

private:
    void writeSetting(const QString& key, const QVariant& value);
    void handleCustomParamsEdited();
    void validateCustomParamsText();
    [[nodiscard]] std::optional<nlohmann::json> parseCustomParams(QString* error) const;
    [[nodiscard]] QString paramsTextForProfile(const QString& profile) const;
    void applyCustomParamsProfile(const QString& profile, bool persist, bool fromUi);

    JsonProfileEditor* _customParamsEditor{nullptr};
    QString _customParamsText;
    QString _customParamsError;
    QString _customParamsProfile{QStringLiteral("custom")};

    bool _restoringSettings{false};
    const QString _settingsGroup;
};
