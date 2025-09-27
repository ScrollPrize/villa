#pragma once

#include <QWidget>

class QPushButton;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QLabel;
class QString;
class QVariant;

// SegmentationWidget hosts controls for interactive surface editing
class SegmentationWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationWidget(QWidget* parent = nullptr);

    [[nodiscard]] bool isEditingEnabled() const { return _editingEnabled; }
    [[nodiscard]] int downsample() const { return _downsample; }
    [[nodiscard]] float radius() const { return _radius; }
    [[nodiscard]] float sigma() const { return _sigma; }

    void setPendingChanges(bool pending);

public slots:
    void setEditingEnabled(bool enabled);
    void setDownsample(int value);
    void setRadius(float value);
    void setSigma(float value);

signals:
    void editingModeChanged(bool enabled);
    void downsampleChanged(int value);
    void radiusChanged(float value);
    void sigmaChanged(float value);
    void applyRequested();
    void resetRequested();
    void stopToolsRequested();

private:
    void setupUI();
    void updateEditingUi();
    void restoreSettings();
    void writeSetting(const QString& key, const QVariant& value);

    QCheckBox* _chkEditing;
    QLabel* _editingStatus;
    QSpinBox* _spinDownsample;
    QSpinBox* _spinRadius;
    QDoubleSpinBox* _spinSigma;
    QPushButton* _btnApply;
    QPushButton* _btnReset;
    QPushButton* _btnStopTools;

    bool _editingEnabled = false;
    int _downsample = 12;
    float _radius = 1.0f;   // grid-space radius (Chebyshev distance)
    float _sigma = 1.0f;    // neighbouring pull strength multiplier
    bool _hasPendingChanges = false;
};
