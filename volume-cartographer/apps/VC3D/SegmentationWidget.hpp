#pragma once

#include <QWidget>

#include "SegmentationInfluenceMode.hpp"

class QPushButton;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QLabel;
class QString;
class QVariant;
class QComboBox;

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
    [[nodiscard]] SegmentationInfluenceMode influenceMode() const { return _influenceMode; }
    [[nodiscard]] SegmentationRowColMode rowColMode() const { return _rowColMode; }
    [[nodiscard]] int holeSearchRadius() const { return _holeSearchRadius; }
    [[nodiscard]] int holeSmoothIterations() const { return _holeSmoothIterations; }
    [[nodiscard]] bool handlesAlwaysVisible() const { return _handlesAlwaysVisible; }
    [[nodiscard]] float handleDisplayDistance() const { return _handleDisplayDistance; }

    void setPendingChanges(bool pending);

public slots:
    void setEditingEnabled(bool enabled);
    void setDownsample(int value);
    void setRadius(float value);
    void setSigma(float value);
    void setInfluenceMode(SegmentationInfluenceMode mode);
    void setRowColMode(SegmentationRowColMode mode);
    void setHoleSearchRadius(int value);
    void setHoleSmoothIterations(int value);
    void setHandlesAlwaysVisible(bool value);
    void setHandleDisplayDistance(float value);

signals:
    void editingModeChanged(bool enabled);
    void downsampleChanged(int value);
    void radiusChanged(float value);
    void sigmaChanged(float value);
    void holeSearchRadiusChanged(int value);
    void holeSmoothIterationsChanged(int value);
    void handlesAlwaysVisibleChanged(bool value);
    void handleDisplayDistanceChanged(float value);
    void influenceModeChanged(SegmentationInfluenceMode mode);
    void rowColModeChanged(SegmentationRowColMode mode);
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
    class QComboBox* _comboInfluenceMode;
    class QComboBox* _comboRowColMode;
    QSpinBox* _spinHoleRadius;
    QSpinBox* _spinHoleIterations;
    QCheckBox* _chkHandlesAlwaysVisible;
    QDoubleSpinBox* _spinHandleDisplayDistance;
    QPushButton* _btnApply;
    QPushButton* _btnReset;
    QPushButton* _btnStopTools;

    bool _editingEnabled = false;
    int _downsample = 12;
    float _radius = 1.0f;   // grid-space radius (Chebyshev distance)
    float _sigma = 1.0f;    // neighbouring pull strength multiplier
    SegmentationInfluenceMode _influenceMode = SegmentationInfluenceMode::GridChebyshev;
    SegmentationRowColMode _rowColMode = SegmentationRowColMode::Dynamic;
    int _holeSearchRadius = 6;
    int _holeSmoothIterations = 25;
    bool _handlesAlwaysVisible = true;
    float _handleDisplayDistance = 25.0f; // world-space units
    bool _hasPendingChanges = false;
};
