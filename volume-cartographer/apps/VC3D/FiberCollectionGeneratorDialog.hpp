#pragma once

#include "FiberCollectionGenerator.hpp"

#include <QDialog>
#include <array>
#include <vector>

class QCheckBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QSpinBox;

// Options of a new Automated Fiber Volume: output file, zone of the current
// volume and its blocks, model, extension and cleanup, and Python. The caller
// adds the volume and its coordinate identity to the request. Choices are
// remembered.
class FiberCollectionGeneratorDialog : public QDialog
{
    Q_OBJECT
public:
    // XYZ voxels of the current volume; voxelSizeUm is its voxel size, 0 if unknown.
    FiberCollectionGeneratorDialog(const std::array<int, 3>& shape, const std::array<int, 3>& center, double voxelSizeUm, QWidget* parent = nullptr);
    vc3d::fibergen::Request request() const;
    // Empty to detect Python automatically.
    QString pythonExecutable() const;
    // The blocks of the chosen zone, in processing order.
    std::vector<vc3d::fibergen::Block> blocks() const;
    void accept() override;

signals:
    void zoneChanged();

private:
    std::array<int, 3> shape_;
    QLineEdit* output_{};
    std::array<QSpinBox*, 3> center_{};
    std::array<QSpinBox*, 3> size_{};
    QSpinBox* blockSize_{};
    QLabel* zone_{};
    QLineEdit* model_{};
    QCheckBox* mirror_{};
    QDoubleSpinBox* threshold_{};
    QCheckBox* extend_{};
    QDoubleSpinBox* maxJoinAngle_{};
    QCheckBox* removeShort_{};
    QDoubleSpinBox* minLength_{};
    QDoubleSpinBox* blackDistance_{};
    QLineEdit* python_{};
    std::array<int, 3> zoneOrigin() const;
    std::array<int, 3> zoneSize() const;
    void updateZone();
};
