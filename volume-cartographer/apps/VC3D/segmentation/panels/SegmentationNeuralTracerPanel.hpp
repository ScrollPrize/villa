#pragma once

#include "segmentation/SegmentationCommon.hpp"

#include <QString>
#include <QWidget>

class QCheckBox;
class QComboBox;
class QPushButton;
class QLabel;
class QLineEdit;
class QSettings;
class QDoubleSpinBox;
class QSpinBox;
class QToolButton;
class CollapsibleSettingsGroup;

class SegmentationNeuralTracerPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationNeuralTracerPanel(const QString& settingsGroup,
                                           QWidget* parent = nullptr);

    // Getters
    [[nodiscard]] bool neuralTracerEnabled() const { return _neuralTracerEnabled; }
    [[nodiscard]] QString neuralCheckpointPath() const { return _neuralCheckpointPath; }
    [[nodiscard]] QString neuralPythonPath() const { return _neuralPythonPath; }
    [[nodiscard]] QString volumeZarrPath() const { return _volumeZarrPath; }
    [[nodiscard]] int neuralVolumeScale() const { return _neuralVolumeScale; }
    [[nodiscard]] int neuralBatchSize() const { return _neuralBatchSize; }
    [[nodiscard]] NeuralTracerModelType neuralModelType() const { return _neuralModelType; }
    [[nodiscard]] TtaMode ttaMode() const { return _ttaMode; }
    [[nodiscard]] QString ttaMergeMethod() const { return _ttaMergeMethod; }
    [[nodiscard]] double ttaOutlierDropThresh() const { return _ttaOutlierDropThresh; }
    [[nodiscard]] QString copyCheckpointPath() const;

    // Setters
    void setNeuralTracerEnabled(bool enabled);
    void setNeuralCheckpointPath(const QString& path);
    void setNeuralPythonPath(const QString& path);
    void setNeuralVolumeScale(int scale);
    void setNeuralBatchSize(int size);
    void setVolumeZarrPath(const QString& path);
    void setNeuralModelType(NeuralTracerModelType type);
    void setTtaMode(TtaMode mode);
    void setTtaMergeMethod(const QString& method);
    void setTtaOutlierDropThresh(double threshold);
    void setCopyCheckpointPath(const QString& path);

    void restoreSettings(QSettings& settings);
    void syncUiState();

signals:
    void neuralTracerEnabledChanged(bool enabled);
    void neuralTracerStatusMessage(const QString& message);
    void copyWithNtRequested();

private:
    void writeSetting(const QString& key, const QVariant& value);
    void updateTtaUiState();

    enum class CopyCheckpointPreset
    {
        CopyLatest = 0,
        CustomPath = 1
    };

    CollapsibleSettingsGroup* _groupNeuralTracer{nullptr};
    QCheckBox* _chkNeuralTracerEnabled{nullptr};
    QComboBox* _comboNeuralModelType{nullptr};
    QComboBox* _comboTtaMode{nullptr};
    QComboBox* _comboTtaMergeMethod{nullptr};
    QComboBox* _comboCopyCheckpointPreset{nullptr};
    QPushButton* _btnCopyWithNt{nullptr};
    QLineEdit* _neuralCheckpointEdit{nullptr};
    QToolButton* _neuralCheckpointBrowse{nullptr};
    QLineEdit* _neuralPythonEdit{nullptr};
    QToolButton* _neuralPythonBrowse{nullptr};
    QComboBox* _comboNeuralVolumeScale{nullptr};
    QSpinBox* _spinNeuralBatchSize{nullptr};
    QDoubleSpinBox* _spinTtaOutlierDropThresh{nullptr};
    QLabel* _lblNeuralTracerStatus{nullptr};

    bool _neuralTracerEnabled{false};
    QString _neuralCheckpointPath;
    QString _neuralPythonPath;
    QString _volumeZarrPath;
    int _neuralVolumeScale{0};
    int _neuralBatchSize{4};
    NeuralTracerModelType _neuralModelType{NeuralTracerModelType::Heatmap};
    TtaMode _ttaMode{TtaMode::Mirror};
    QString _ttaMergeMethod{QStringLiteral("vector_geomedian")};
    double _ttaOutlierDropThresh{1.25};
    CopyCheckpointPreset _copyCheckpointPreset{CopyCheckpointPreset::CopyLatest};

    bool _restoringSettings{false};
    const QString _settingsGroup;
};
