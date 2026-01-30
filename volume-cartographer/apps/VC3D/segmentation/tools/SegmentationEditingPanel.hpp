#pragma once

#include "segmentation/SegmentationCommon.hpp"
#include "segmentation/SegmentationPushPullConfig.hpp"

#include <QString>
#include <QWidget>

class QCheckBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSettings;
class QSpinBox;
class CollapsibleSettingsGroup;

class SegmentationEditingPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationEditingPanel(const QString& settingsGroup,
                                      QWidget* parent = nullptr);

    // Getters
    [[nodiscard]] float dragRadius() const { return _dragRadiusSteps; }
    [[nodiscard]] float dragSigma() const { return _dragSigmaSteps; }
    [[nodiscard]] float lineRadius() const { return _lineRadiusSteps; }
    [[nodiscard]] float lineSigma() const { return _lineSigmaSteps; }
    [[nodiscard]] float pushPullRadius() const { return _pushPullRadiusSteps; }
    [[nodiscard]] float pushPullSigma() const { return _pushPullSigmaSteps; }
    [[nodiscard]] float pushPullStep() const { return _pushPullStep; }
    [[nodiscard]] AlphaPushPullConfig alphaPushPullConfig() const { return _alphaPushPullConfig; }
    [[nodiscard]] float smoothingStrength() const { return _smoothStrength; }
    [[nodiscard]] int smoothingIterations() const { return _smoothIterations; }
    [[nodiscard]] bool showHoverMarker() const { return _showHoverMarker; }

    // Setters
    void setDragRadius(float value);
    void setDragSigma(float value);
    void setLineRadius(float value);
    void setLineSigma(float value);
    void setPushPullRadius(float value);
    void setPushPullSigma(float value);
    void setPushPullStep(float value);
    void setAlphaPushPullConfig(const AlphaPushPullConfig& config);
    void setSmoothingStrength(float value);
    void setSmoothingIterations(int value);
    void setShowHoverMarker(bool enabled);

    void restoreSettings(QSettings& settings);
    void syncUiState(bool editingEnabled, bool growthInProgress);

    // Group accessors (still needed for expand/collapse state persistence by coordinator)
    CollapsibleSettingsGroup* editingGroup() const { return _groupEditing; }

signals:
    void dragRadiusChanged(float value);
    void dragSigmaChanged(float value);
    void lineRadiusChanged(float value);
    void lineSigmaChanged(float value);
    void pushPullRadiusChanged(float value);
    void pushPullSigmaChanged(float value);
    void pushPullStepChanged(float value);
    void alphaPushPullConfigChanged();
    void smoothingStrengthChanged(float value);
    void smoothingIterationsChanged(int value);
    void hoverMarkerToggled(bool enabled);
    void applyRequested();
    void resetRequested();
    void stopToolsRequested();

private:
    void writeSetting(const QString& key, const QVariant& value);
    void applyAlphaPushPullConfig(const AlphaPushPullConfig& config,
                                  bool emitSignal,
                                  bool persist = true);

    CollapsibleSettingsGroup* _groupEditing{nullptr};
    CollapsibleSettingsGroup* _groupDrag{nullptr};
    CollapsibleSettingsGroup* _groupLine{nullptr};
    CollapsibleSettingsGroup* _groupPushPull{nullptr};

    QDoubleSpinBox* _spinDragRadius{nullptr};
    QDoubleSpinBox* _spinDragSigma{nullptr};
    QDoubleSpinBox* _spinLineRadius{nullptr};
    QDoubleSpinBox* _spinLineSigma{nullptr};
    QDoubleSpinBox* _spinPushPullRadius{nullptr};
    QDoubleSpinBox* _spinPushPullSigma{nullptr};
    QDoubleSpinBox* _spinPushPullStep{nullptr};
    QWidget* _alphaPushPullPanel{nullptr};
    QCheckBox* _chkAlphaPerVertex{nullptr};
    QDoubleSpinBox* _spinAlphaStart{nullptr};
    QDoubleSpinBox* _spinAlphaStop{nullptr};
    QDoubleSpinBox* _spinAlphaStep{nullptr};
    QDoubleSpinBox* _spinAlphaLow{nullptr};
    QDoubleSpinBox* _spinAlphaHigh{nullptr};
    QDoubleSpinBox* _spinAlphaBorder{nullptr};
    QSpinBox* _spinAlphaBlurRadius{nullptr};
    QDoubleSpinBox* _spinAlphaPerVertexLimit{nullptr};
    QLabel* _lblAlphaInfo{nullptr};
    QDoubleSpinBox* _spinSmoothStrength{nullptr};
    QSpinBox* _spinSmoothIterations{nullptr};
    QPushButton* _btnApply{nullptr};
    QPushButton* _btnReset{nullptr};
    QPushButton* _btnStop{nullptr};
    QCheckBox* _chkShowHoverMarker{nullptr};

    float _dragRadiusSteps{5.75f};
    float _dragSigmaSteps{2.0f};
    float _lineRadiusSteps{5.75f};
    float _lineSigmaSteps{2.0f};
    float _pushPullRadiusSteps{5.75f};
    float _pushPullSigmaSteps{2.0f};
    float _pushPullStep{4.0f};
    AlphaPushPullConfig _alphaPushPullConfig{};
    float _smoothStrength{0.4f};
    int _smoothIterations{2};
    bool _showHoverMarker{true};

    bool _restoringSettings{false};
    const QString _settingsGroup;
};
