#pragma once

#include "segmentation/SegmentationCommon.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"

#include <QString>
#include <QStringList>
#include <QVector>
#include <QWidget>

#include <optional>
#include <utility>
#include <vector>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSettings;
class QSpinBox;

class SegmentationGrowthPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationGrowthPanel(const QString& settingsGroup,
                                     QWidget* parent = nullptr);

    // Getters
    [[nodiscard]] SegmentationGrowthMethod growthMethod() const { return _growthMethod; }
    [[nodiscard]] int growthSteps() const { return _growthSteps; }
    [[nodiscard]] int extrapolationPointCount() const { return _extrapolationPointCount; }
    [[nodiscard]] ExtrapolationType extrapolationType() const { return _extrapolationType; }
    [[nodiscard]] int sdtMaxSteps() const { return _sdtMaxSteps; }
    [[nodiscard]] float sdtStepSize() const { return _sdtStepSize; }
    [[nodiscard]] float sdtConvergence() const { return _sdtConvergence; }
    [[nodiscard]] int sdtChunkSize() const { return _sdtChunkSize; }
    [[nodiscard]] int skeletonConnectivity() const { return _skeletonConnectivity; }
    [[nodiscard]] int skeletonSliceOrientation() const { return _skeletonSliceOrientation; }
    [[nodiscard]] int skeletonChunkSize() const { return _skeletonChunkSize; }
    [[nodiscard]] int skeletonSearchRadius() const { return _skeletonSearchRadius; }
    [[nodiscard]] bool growthKeybindsEnabled() const { return _growthKeybindsEnabled; }
    [[nodiscard]] QString normal3dZarrPath() const { return _normal3dSelectedPath; }
    [[nodiscard]] std::vector<SegmentationGrowthDirection> allowedGrowthDirections() const;
    [[nodiscard]] std::optional<std::pair<int, int>> correctionsZRange() const;

    // Setters
    void setGrowthMethod(SegmentationGrowthMethod method);
    void setGrowthSteps(int steps, bool persist = true);
    void setGrowthInProgress(bool running);
    void setNormalGridAvailable(bool available);
    void setNormalGridPathHint(const QString& hint);
    void setNormalGridPath(const QString& path);
    void setNormal3dZarrCandidates(const QStringList& candidates, const QString& hint);
    void setVolumePackagePath(const QString& path);
    void setAvailableVolumes(const QVector<QPair<QString, QString>>& volumes,
                             const QString& activeId);
    void setActiveVolume(const QString& volumeId);

    void restoreSettings(QSettings& settings);
    void syncUiState(bool editingEnabled, bool growthInProgress);

signals:
    void growSurfaceRequested(SegmentationGrowthMethod method,
                              SegmentationGrowthDirection direction,
                              int steps,
                              bool inpaintOnly);
    void growthMethodChanged(SegmentationGrowthMethod method);
    void volumeSelectionChanged(const QString& volumeId);
    void correctionsZRangeChanged(bool enabled, int zMin, int zMax);

private:
    void writeSetting(const QString& key, const QVariant& value);
    void applyGrowthSteps(int steps, bool persist, bool fromUi);
    void setGrowthDirectionMask(int mask);
    void updateGrowthDirectionMaskFromUi(QCheckBox* changedCheckbox);
    void applyGrowthDirectionMaskToUi();
    static int normalizeGrowthDirectionMask(int mask);
    void updateGrowthUiState();
    void updateNormal3dUi();
    void triggerGrowthRequest(SegmentationGrowthDirection direction, int steps, bool inpaintOnly);
    [[nodiscard]] QString determineDefaultVolumeId(const QVector<QPair<QString, QString>>& volumes,
                                                   const QString& requestedId) const;

    // UI widgets
    QGroupBox* _groupGrowth{nullptr};
    QSpinBox* _spinGrowthSteps{nullptr};
    QComboBox* _comboGrowthMethod{nullptr};
    QWidget* _extrapolationOptionsPanel{nullptr};
    QLabel* _lblExtrapolationPoints{nullptr};
    QSpinBox* _spinExtrapolationPoints{nullptr};
    QComboBox* _comboExtrapolationType{nullptr};
    QWidget* _sdtParamsContainer{nullptr};
    QSpinBox* _spinSDTMaxSteps{nullptr};
    QDoubleSpinBox* _spinSDTStepSize{nullptr};
    QDoubleSpinBox* _spinSDTConvergence{nullptr};
    QSpinBox* _spinSDTChunkSize{nullptr};
    QWidget* _skeletonParamsContainer{nullptr};
    QComboBox* _comboSkeletonConnectivity{nullptr};
    QComboBox* _comboSkeletonSliceOrientation{nullptr};
    QSpinBox* _spinSkeletonChunkSize{nullptr};
    QSpinBox* _spinSkeletonSearchRadius{nullptr};
    QPushButton* _btnGrow{nullptr};
    QPushButton* _btnInpaint{nullptr};
    QCheckBox* _chkGrowthDirUp{nullptr};
    QCheckBox* _chkGrowthDirDown{nullptr};
    QCheckBox* _chkGrowthDirLeft{nullptr};
    QCheckBox* _chkGrowthDirRight{nullptr};
    QCheckBox* _chkGrowthKeybindsEnabled{nullptr};
    QCheckBox* _chkCorrectionsUseZRange{nullptr};
    QSpinBox* _spinCorrectionsZMin{nullptr};
    QSpinBox* _spinCorrectionsZMax{nullptr};
    QComboBox* _comboVolumes{nullptr};
    QLabel* _lblNormalGrid{nullptr};
    QLineEdit* _editNormalGridPath{nullptr};
    QLabel* _lblNormal3d{nullptr};
    QComboBox* _comboNormal3d{nullptr};
    QLineEdit* _editNormal3dPath{nullptr};

    // State
    SegmentationGrowthMethod _growthMethod{SegmentationGrowthMethod::Corrections};
    int _growthSteps{5};
    int _tracerGrowthSteps{5};
    int _growthDirectionMask{0};
    bool _growthKeybindsEnabled{true};
    int _extrapolationPointCount{7};
    ExtrapolationType _extrapolationType{ExtrapolationType::Linear};

    int _sdtMaxSteps{5};
    float _sdtStepSize{0.8f};
    float _sdtConvergence{0.5f};
    int _sdtChunkSize{128};

    int _skeletonConnectivity{26};
    int _skeletonSliceOrientation{0};
    int _skeletonChunkSize{128};
    int _skeletonSearchRadius{5};

    bool _normalGridAvailable{false};
    QString _normalGridHint;
    QString _normalGridDisplayPath;
    QString _normalGridPath;

    QStringList _normal3dCandidates;
    QString _normal3dHint;
    QString _normal3dSelectedPath;
    QString _volumePackagePath;
    QVector<QPair<QString, QString>> _volumeEntries;
    QString _activeVolumeId;

    bool _correctionsZRangeEnabled{false};
    int _correctionsZMin{0};
    int _correctionsZMax{0};

    bool _editingEnabled{false};
    bool _growthInProgress{false};
    bool _restoringSettings{false};
    const QString _settingsGroup;
};
