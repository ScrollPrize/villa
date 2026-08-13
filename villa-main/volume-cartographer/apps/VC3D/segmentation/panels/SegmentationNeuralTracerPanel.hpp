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
    [[nodiscard]] NeuralTracerOutputMode neuralOutputMode() const { return _neuralOutputMode; }
    [[nodiscard]] DenseTtaMode denseTtaMode() const { return _denseTtaMode; }
    [[nodiscard]] QString denseTtaMergeMethod() const { return _denseTtaMergeMethod; }
    [[nodiscard]] double denseTtaOutlierDropThresh() const { return _denseTtaOutlierDropThresh; }
    [[nodiscard]] QString denseCheckpointPath() const;
    [[nodiscard]] QString copyCheckpointPath() const;

    // Setters
    void setNeuralTracerEnabled(bool enabled);
    void setNeuralCheckpointPath(const QString& path);
    void setNeuralPythonPath(const QString& path);
    void setNeuralVolumeScale(int scale);
    void setNeuralBatchSize(int size);
    void setVolumeZarrPath(const QString& path);
    void setNeuralModelType(NeuralTracerModelType type);
    void setNeuralOutputMode(NeuralTracerOutputMode mode);
    void setDenseTtaMode(DenseTtaMode mode);
    void setDenseTtaMergeMethod(const QString& method);
    void setDenseTtaOutlierDropThresh(double threshold);
    void setDenseCheckpointPath(const QString& path);
    void setCopyCheckpointPath(const QString& path);

    void restoreSettings(QSettings& settings);
    void syncUiState();

    // Still needed by SegmentationWidget for rememberGroupState
    CollapsibleSettingsGroup* neuralTracerGroup() const { return _groupNeuralTracer; }

signals:
    void neuralTracerEnabledChanged(bool enabled);
    void neuralTracerStatusMessage(const QString& message);
    void copyWithNtRequested();

private:
    void writeSetting(const QString& key, const QVariant& value);
    void updateDenseUiState();

    enum class DenseCheckpointPreset
    {
        DenseLatest = 0,
        CustomPath = 1
    };

    enum class CopyCheckpointPreset
    {
        CopyLatest = 0,
        CustomPath = 1
    };

    CollapsibleSettingsGroup* _groupNeuralTracer{nullptr};
    QCheckBox* _chkNeuralTracerEnabled{nullptr};
    QComboBox* _comboNeuralModelType{nullptr};
    QComboBox* _comboNeuralOutputMode{nullptr};
    QComboBox* _comboDenseTtaMode{nullptr};
    QComboBox* _comboDenseTtaMergeMethod{nullptr};
    QComboBox* _comboDenseCheckpointPreset{nullptr};
    QComboBox* _comboCopyCheckpointPreset{nullptr};
    QPushButton* _btnCopyWithNt{nullptr};
    QLineEdit* _neuralCheckpointEdit{nullptr};
    QToolButton* _neuralCheckpointBrowse{nullptr};
    QLineEdit* _neuralPythonEdit{nullptr};
    QToolButton* _neuralPythonBrowse{nullptr};
    QComboBox* _comboNeuralVolumeScale{nullptr};
    QSpinBox* _spinNeuralBatchSize{nullptr};
    QDoubleSpinBox* _spinDenseTtaOutlierDropThresh{nullptr};
    QLabel* _lblNeuralTracerStatus{nullptr};

    bool _neuralTracerEnabled{false};
    QString _neuralCheckpointPath;
    QString _neuralPythonPath;
    QString _volumeZarrPath;
    int _neuralVolumeScale{0};
    int _neuralBatchSize{4};
    NeuralTracerModelType _neuralModelType{NeuralTracerModelType::Heatmap};
    NeuralTracerOutputMode _neuralOutputMode{NeuralTracerOutputMode::OverwriteCurrentSegment};
    DenseTtaMode _denseTtaMode{DenseTtaMode::Mirror};
    QString _denseTtaMergeMethod{QStringLiteral("vector_geomedian")};
    double _denseTtaOutlierDropThresh{1.25};
    DenseCheckpointPreset _denseCheckpointPreset{DenseCheckpointPreset::DenseLatest};
    CopyCheckpointPreset _copyCheckpointPreset{CopyCheckpointPreset::CopyLatest};

    bool _restoringSettings{false};
    const QString _settingsGroup;
};
