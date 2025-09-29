#pragma once

#include <QWidget>
#include <QPair>
#include <QVector>
#include <cstdint>

#include <optional>
#include <utility>
#include <vector>

#include "SegmentationInfluenceMode.hpp"
#include "SegmentationGrowth.hpp"

class QPushButton;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QLabel;
class QString;
class QVariant;
class QComboBox;
class QGroupBox;

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
    [[nodiscard]] float sliceFadeDistance() const { return _sliceFadeDistance; }
    [[nodiscard]] SegmentationSliceDisplayMode sliceDisplayMode() const { return _sliceDisplayMode; }
    [[nodiscard]] SegmentationRowColMode rowColMode() const { return _rowColMode; }
    [[nodiscard]] float highlightDistance() const { return _highlightDistance; }
    [[nodiscard]] int holeSearchRadius() const { return _holeSearchRadius; }
    [[nodiscard]] int holeSmoothIterations() const { return _holeSmoothIterations; }
    [[nodiscard]] bool handlesAlwaysVisible() const { return _handlesAlwaysVisible; }
    [[nodiscard]] float handleDisplayDistance() const { return _handleDisplayDistance; }
    [[nodiscard]] bool fillInvalidRegions() const { return _fillInvalidRegions; }
    [[nodiscard]] SegmentationGrowthMethod growthMethod() const { return _growthMethod; }
    [[nodiscard]] std::vector<SegmentationGrowthDirection> allowedGrowthDirections() const;

    void setPendingChanges(bool pending);
    void setCorrectionsEnabled(bool enabled);
    void setCorrectionsAnnotateChecked(bool enabled);
    void setCorrectionCollections(const QVector<QPair<uint64_t, QString>>& collections,
                                   std::optional<uint64_t> activeId);
    void setHandlesLocked(bool locked);
    [[nodiscard]] bool handlesLocked() const { return _handlesLocked; }
    void setNormalGridAvailable(bool available);
    void setAvailableVolumes(const QVector<QPair<QString, QString>>& volumes,
                             const QString& activeId);
    void setActiveVolume(const QString& volumeId);
    void setNormalGridPathHint(const QString& hint);
    [[nodiscard]] std::optional<std::pair<int, int>> correctionsZRange() const
    {
        if (!_correctionsZRangeEnabled) {
            return std::nullopt;
        }
        return std::make_pair(_correctionsZMin, _correctionsZMax);
    }

public slots:
    void setEditingEnabled(bool enabled);
    void setDownsample(int value);
    void setRadius(float value);
    void setSigma(float value);
    void setInfluenceMode(SegmentationInfluenceMode mode);
    void setSliceFadeDistance(float value);
    void setSliceDisplayMode(SegmentationSliceDisplayMode mode);
    void setRowColMode(SegmentationRowColMode mode);
    void setHighlightDistance(float value);
    void setHoleSearchRadius(int value);
    void setHoleSmoothIterations(int value);
    void setHandlesAlwaysVisible(bool value);
    void setHandleDisplayDistance(float value);
    void setFillInvalidRegions(bool value);

signals:
    void editingModeChanged(bool enabled);
    void downsampleChanged(int value);
    void radiusChanged(float value);
    void sigmaChanged(float value);
    void holeSearchRadiusChanged(int value);
    void holeSmoothIterationsChanged(int value);
    void handlesAlwaysVisibleChanged(bool value);
    void handleDisplayDistanceChanged(float value);
    void fillInvalidRegionsChanged(bool value);
    void influenceModeChanged(SegmentationInfluenceMode mode);
    void sliceFadeDistanceChanged(float value);
    void sliceDisplayModeChanged(SegmentationSliceDisplayMode mode);
    void rowColModeChanged(SegmentationRowColMode mode);
    void highlightDistanceChanged(float value);
    void applyRequested();
    void resetRequested();
    void stopToolsRequested();
    void growSurfaceRequested(SegmentationGrowthMethod method,
                              SegmentationGrowthDirection direction,
                              int steps);
    void growthMethodChanged(SegmentationGrowthMethod method);
    void correctionsAnnotateToggled(bool enabled);
    void correctionsCollectionSelected(uint64_t collectionId);
    void correctionsCreateRequested();
    void correctionsZRangeChanged(bool enabled, int zMin, int zMax);
    void volumeSelectionChanged(const QString& volumeId);

private:
    void setupUI();
    void updateEditingUi();
    void restoreSettings();
    void writeSetting(const QString& key, const QVariant& value);
    void refreshCorrectionsUiState();
    void updateGrowthModeUi();
    void setGrowthDirectionMask(int mask);
    void updateGrowthDirectionMaskFromUi(QCheckBox* changedCheckbox);
    void applyGrowthDirectionMaskToUi();
    static int normalizeGrowthDirectionMask(int mask);

    QCheckBox* _chkEditing;
    QLabel* _editingStatus;
    QGroupBox* _groupGrowth;
    QGroupBox* _groupSampling;
    QSpinBox* _spinDownsample;
    QSpinBox* _spinRadius;
    QDoubleSpinBox* _spinSigma;
    class QComboBox* _comboInfluenceMode;
    QGroupBox* _groupInfluence;
    class QGroupBox* _groupSliceVisibility;
    QDoubleSpinBox* _spinSliceFadeDistance;
    class QComboBox* _comboSliceDisplayMode;
    class QComboBox* _comboRowColMode;
    QDoubleSpinBox* _spinHighlightDistance;
    QGroupBox* _groupHole;
    QSpinBox* _spinHoleRadius;
    QSpinBox* _spinHoleIterations;
    QCheckBox* _chkFillInvalidRegions;
    QGroupBox* _groupHandleDisplay;
    QCheckBox* _chkHandlesAlwaysVisible;
    QDoubleSpinBox* _spinHandleDisplayDistance;
    QPushButton* _btnApply;
    QPushButton* _btnReset;
    QPushButton* _btnStopTools;
    class QComboBox* _comboGrowthMethod;
    class QComboBox* _comboGrowthDirection;
    QSpinBox* _spinGrowthSteps;
    QPushButton* _btnGrow;
    QCheckBox* _chkGrowthDirUp;
    QCheckBox* _chkGrowthDirDown;
    QCheckBox* _chkGrowthDirLeft;
    QCheckBox* _chkGrowthDirRight;
    class QComboBox* _comboVolume;
    QGroupBox* _groupCorrections;
    class QComboBox* _comboCorrections;
    QPushButton* _btnCorrectionsNew;
    QCheckBox* _chkCorrectionsAnnotate;
    QCheckBox* _chkCorrectionsUseZRange;
    QSpinBox* _spinCorrectionsZMin;
    QSpinBox* _spinCorrectionsZMax;
    QWidget* _normalGridStatusWidget;
    QLabel* _normalGridStatusIcon;
    QLabel* _normalGridStatusText;

    bool _editingEnabled = false;
    int _downsample = 12;
    float _radius = 1.0f;   // grid-space radius (Chebyshev distance)
    float _sigma = 1.0f;    // neighbouring pull strength multiplier
    SegmentationInfluenceMode _influenceMode = SegmentationInfluenceMode::GridChebyshev;
    float _sliceFadeDistance = 10.0f;
    SegmentationSliceDisplayMode _sliceDisplayMode = SegmentationSliceDisplayMode::Fade;
    SegmentationRowColMode _rowColMode = SegmentationRowColMode::Dynamic;
    int _holeSearchRadius = 6;
    int _holeSmoothIterations = 25;
    bool _handlesAlwaysVisible = true;
    float _handleDisplayDistance = 25.0f; // world-space units
    float _highlightDistance = 15.0f;      // screen-space pixels
    bool _fillInvalidRegions = true;
    bool _hasPendingChanges = false;
    SegmentationGrowthMethod _growthMethod{SegmentationGrowthMethod::Corrections};
    SegmentationGrowthDirection _growthDirection{SegmentationGrowthDirection::All};
    int _growthSteps{5};
    static constexpr int kGrowDirUpBit = 1 << 0;
    static constexpr int kGrowDirDownBit = 1 << 1;
    static constexpr int kGrowDirLeftBit = 1 << 2;
    static constexpr int kGrowDirRightBit = 1 << 3;
    static constexpr int kGrowDirAllMask = kGrowDirUpBit | kGrowDirDownBit | kGrowDirLeftBit | kGrowDirRightBit;
    int _growthDirectionMask{kGrowDirAllMask};
    std::optional<uint64_t> _activeCorrectionId;
    bool _correctionsEnabled{false};
    bool _correctionsZRangeEnabled{false};
    int _correctionsZMin{0};
    int _correctionsZMax{0};
    bool _handlesLocked{false};
    bool _normalGridAvailable{false};
    QVector<QPair<QString, QString>> _volumeEntries;
    QString _activeVolumeId;
};
