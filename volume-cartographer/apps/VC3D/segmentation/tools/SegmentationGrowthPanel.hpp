#pragma once

#include <QWidget>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
class QWidget;

class SegmentationGrowthPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationGrowthPanel(bool growthKeybindsEnabled, QWidget* parent = nullptr);

    QComboBox* growthMethodCombo() const { return _comboGrowthMethod; }
    QSpinBox* growthStepsSpin() const { return _spinGrowthSteps; }
    QWidget* extrapolationOptionsPanel() const { return _extrapolationOptionsPanel; }
    QLabel* extrapolationPointsLabel() const { return _lblExtrapolationPoints; }
    QSpinBox* extrapolationPointsSpin() const { return _spinExtrapolationPoints; }
    QComboBox* extrapolationTypeCombo() const { return _comboExtrapolationType; }

    QWidget* sdtParamsContainer() const { return _sdtParamsContainer; }
    QSpinBox* sdtMaxStepsSpin() const { return _spinSDTMaxSteps; }
    QDoubleSpinBox* sdtStepSizeSpin() const { return _spinSDTStepSize; }
    QDoubleSpinBox* sdtConvergenceSpin() const { return _spinSDTConvergence; }
    QSpinBox* sdtChunkSizeSpin() const { return _spinSDTChunkSize; }

    QWidget* skeletonParamsContainer() const { return _skeletonParamsContainer; }
    QComboBox* skeletonConnectivityCombo() const { return _comboSkeletonConnectivity; }
    QComboBox* skeletonSliceOrientationCombo() const { return _comboSkeletonSliceOrientation; }
    QSpinBox* skeletonChunkSizeSpin() const { return _spinSkeletonChunkSize; }
    QSpinBox* skeletonSearchRadiusSpin() const { return _spinSkeletonSearchRadius; }

    QPushButton* growButton() const { return _btnGrow; }
    QPushButton* inpaintButton() const { return _btnInpaint; }

    QCheckBox* growthDirUpCheck() const { return _chkGrowthDirUp; }
    QCheckBox* growthDirDownCheck() const { return _chkGrowthDirDown; }
    QCheckBox* growthDirLeftCheck() const { return _chkGrowthDirLeft; }
    QCheckBox* growthDirRightCheck() const { return _chkGrowthDirRight; }
    QCheckBox* growthKeybindsCheck() const { return _chkGrowthKeybindsEnabled; }

    QCheckBox* correctionsZRangeCheck() const { return _chkCorrectionsUseZRange; }
    QSpinBox* correctionsZMinSpin() const { return _spinCorrectionsZMin; }
    QSpinBox* correctionsZMaxSpin() const { return _spinCorrectionsZMax; }

    QComboBox* volumesCombo() const { return _comboVolumes; }

    QLabel* normalGridLabel() const { return _lblNormalGrid; }
    QLineEdit* normalGridPathEdit() const { return _editNormalGridPath; }

    QLabel* normal3dLabel() const { return _lblNormal3d; }
    QComboBox* normal3dCombo() const { return _comboNormal3d; }
    QLineEdit* normal3dPathEdit() const { return _editNormal3dPath; }

private:
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
};
