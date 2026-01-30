#pragma once

#include "segmentation/SegmentationCommon.hpp"

#include <QString>
#include <QVector>
#include <QWidget>

#include <cstdint>
#include <optional>
#include <utility>

class QCheckBox;
class QComboBox;
class QGroupBox;
class QPushButton;
class QSettings;

class SegmentationCorrectionsPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationCorrectionsPanel(const QString& settingsGroup,
                                          QWidget* parent = nullptr);

    // Getters
    [[nodiscard]] bool correctionsEnabled() const { return _correctionsEnabled; }
    [[nodiscard]] bool correctionsAnnotateChecked() const { return _correctionsAnnotateChecked; }

    // Setters
    void setCorrectionsEnabled(bool enabled);
    void setCorrectionsAnnotateChecked(bool enabled);
    void setCorrectionCollections(const QVector<QPair<uint64_t, QString>>& collections,
                                  std::optional<uint64_t> activeId);

    void restoreSettings(QSettings& settings);
    void syncUiState(bool editingEnabled, bool growthInProgress);

signals:
    void correctionsCreateRequested();
    void correctionsCollectionSelected(uint64_t collectionId);
    void correctionsAnnotateToggled(bool enabled);

private:
    void writeSetting(const QString& key, const QVariant& value);

    QGroupBox* _groupCorrections{nullptr};
    QComboBox* _comboCorrections{nullptr};
    QPushButton* _btnCorrectionsNew{nullptr};
    QCheckBox* _chkCorrectionsAnnotate{nullptr};

    bool _correctionsEnabled{false};
    bool _correctionsAnnotateChecked{false};

    bool _restoringSettings{false};
    const QString _settingsGroup;
};
