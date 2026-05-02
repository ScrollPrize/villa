#pragma once

#include <QString>
#include <QWidget>

class QLabel;
class QPushButton;
class QCheckBox;
class QSpinBox;

class ViewerTransformsPanel : public QWidget
{
    Q_OBJECT

public:
    struct UiState {
        bool previewAvailable{false};
        bool sourceAvailable{false};
        bool editingEnabled{false};
        bool affineAvailable{false};
        bool scaleOnly{false};
        bool saveAvailable{false};
        QString statusText;
    };

    explicit ViewerTransformsPanel(QWidget* parent = nullptr);

    [[nodiscard]] bool previewChecked() const;
    [[nodiscard]] bool scaleOnlyChecked() const;
    [[nodiscard]] bool invertChecked() const;
    [[nodiscard]] int scaleValue() const;

    void setPreviewChecked(bool checked, bool blockSignals = true);
    void applyUiState(const UiState& state);

signals:
    void previewToggled(bool enabled);
    void stateChanged();
    void loadAffineRequested();
    void saveTransformedRequested();

private:
    QCheckBox* _preview{nullptr};
    QCheckBox* _scaleOnly{nullptr};
    QCheckBox* _invert{nullptr};
    QSpinBox* _scale{nullptr};
    QPushButton* _loadAffine{nullptr};
    QPushButton* _saveTransformed{nullptr};
    QLabel* _status{nullptr};
};
