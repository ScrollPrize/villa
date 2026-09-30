#include "FiberCollectionGeneratorDialog.hpp"
#include "VCSettings.hpp"

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QVBoxLayout>
#include <algorithm>

namespace
{
constexpr int kDefaultZoneSize = 512;
// Voxel sizes of the scans the default model was trained on: 8.64 and 9.362 µm.
constexpr double kDefaultModelMinVoxelUm = 8.0;
constexpr double kDefaultModelMaxVoxelUm = 10.0;
constexpr auto kModelKey = "fiberCollections/generator/model";
constexpr auto kMirrorKey = "fiberCollections/generator/mirror";
constexpr auto kThresholdKey = "fiberCollections/generator/thresholdPercent";
constexpr auto kPythonKey = "fiberCollections/generator/python";
constexpr auto kSizeKey = "fiberCollections/generator/zoneSize";
constexpr auto kBlockSizeKey = "fiberCollections/generator/blockSize";
constexpr auto kExtendKey = "fiberCollections/generator/extend";
constexpr auto kMaxJoinAngleKey = "fiberCollections/generator/maxJoinAngle";
constexpr auto kRemoveShortKey = "fiberCollections/generator/removeShort";
constexpr auto kMinLengthKey = "fiberCollections/generator/minLength";
constexpr auto kBlackDistanceKey = "fiberCollections/generator/blackDistance";
constexpr auto kDirectoryKey = "fiberCollections/generator/directory";
}  // namespace

FiberCollectionGeneratorDialog::FiberCollectionGeneratorDialog(
    const std::array<int, 3>& shape, const std::array<int, 3>& center, double voxelSizeUm, QWidget* parent)
    : QDialog(parent), shape_(shape)
{
    setWindowTitle(tr("New Automated Fiber Volume"));
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    auto* layout = new QVBoxLayout(this);
    auto* intro = new QLabel(
        tr("Predict fibers in a zone of the current volume block by block, stitch them into long fibers and open the result. "
           "The blocks are drawn on the CT views. Runs Python with the vesuvius package in the background."),
        this);
    intro->setWordWrap(true);
    layout->addWidget(intro);
    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

    output_ = new QLineEdit(this);
    output_->setObjectName("fiberGeneratorOutput");
    auto* browse = new QPushButton(tr("Browse…"), this);
    auto* outputRow = new QHBoxLayout;
    outputRow->addWidget(output_, 1);
    outputRow->addWidget(browse);
    form->addRow(tr("Save as"), outputRow);
    connect(browse, &QPushButton::clicked, this, [this]() {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        const auto start = output_->text().isEmpty() ? settings.value(kDirectoryKey, QDir::homePath()).toString() : output_->text();
        // Existing volumes are never replaced; accept() explains that.
        auto path = QFileDialog::getSaveFileName(this, tr("New Automated Fiber Volume"), start,
            tr("Automated Fiber Volume (*.afv)"), nullptr, QFileDialog::DontConfirmOverwrite);
        if (path.isEmpty())
            return;
        if (QFileInfo(path).suffix().compare("afv", Qt::CaseInsensitive) != 0)
            path += ".afv";
        output_->setText(path);
    });

    const int savedSize = settings.value(kSizeKey, kDefaultZoneSize).toInt();
    auto* centerRow = new QHBoxLayout;
    auto* sizeRow = new QHBoxLayout;
    for (int i = 0; i < 3; ++i) {
        center_[i] = new QSpinBox(this);
        center_[i]->setRange(0, std::max(shape_[i] - 1, 0));
        center_[i]->setValue(center[i]);
        center_[i]->setPrefix(QStringLiteral("%1 ").arg(QLatin1Char("XYZ"[i])));
        centerRow->addWidget(center_[i]);
        size_[i] = new QSpinBox(this);
        size_[i]->setRange(1, std::max(shape_[i], 1));
        size_[i]->setSingleStep(64);
        size_[i]->setValue(std::min(savedSize, shape_[i]));
        size_[i]->setPrefix(QStringLiteral("%1 ").arg(QLatin1Char("XYZ"[i])));
        sizeRow->addWidget(size_[i]);
        connect(center_[i], &QSpinBox::valueChanged, this, &FiberCollectionGeneratorDialog::updateZone);
        connect(size_[i], &QSpinBox::valueChanged, this, &FiberCollectionGeneratorDialog::updateZone);
    }
    form->addRow(tr("Zone center (voxels)"), centerRow);
    form->addRow(tr("Zone size (voxels)"), sizeRow);
    blockSize_ = new QSpinBox(this);
    blockSize_->setRange(128, 2048);
    blockSize_->setSingleStep(64);
    blockSize_->setValue(settings.value(kBlockSizeKey, vc3d::fibergen::kDefaultBlockSize).toInt());
    blockSize_->setToolTip(tr("The zone is processed in cubes of this size; the fibers of neighbouring cubes are stitched together."));
    connect(blockSize_, &QSpinBox::valueChanged, this, &FiberCollectionGeneratorDialog::updateZone);
    form->addRow(tr("Block size (voxels)"), blockSize_);
    zone_ = new QLabel(this);
    zone_->setWordWrap(true);
    form->addRow(QString(), zone_);

    model_ = new QLineEdit(settings.value(kModelKey, vc3d::fibergen::kDefaultModel).toString(), this);
    model_->setPlaceholderText(vc3d::fibergen::kDefaultModel);
    model_->setToolTip(tr("Hugging Face repository or local folder of an nnU-Net fiber model."));
    form->addRow(tr("Model"), model_);
    mirror_ = new QCheckBox(tr("Test-time mirroring: better predictions, about 8× slower"), this);
    mirror_->setChecked(settings.value(kMirrorKey, false).toBool());
    form->addRow(QString(), mirror_);
    threshold_ = new QDoubleSpinBox(this);
    threshold_->setRange(1, 100);
    threshold_->setDecimals(0);
    threshold_->setSuffix(QStringLiteral(" %"));
    threshold_->setValue(settings.value(kThresholdKey, vc3d::fibergen::kDefaultThresholdPercent).toDouble());
    threshold_->setToolTip(tr("Minimum fiber probability kept before fitting polylines."));
    form->addRow(tr("Fiber threshold"), threshold_);

    extend_ = new QCheckBox(tr("Join fibers across gaps with the gap model: longer fibers, several times slower"), this);
    extend_->setChecked(settings.value(kExtendKey, false).toBool());
    form->addRow(QString(), extend_);
    maxJoinAngle_ = new QDoubleSpinBox(this);
    maxJoinAngle_->setRange(0, 180);
    maxJoinAngle_->setDecimals(0);
    maxJoinAngle_->setSuffix(QStringLiteral("°"));
    maxJoinAngle_->setValue(settings.value(kMaxJoinAngleKey, vc3d::fibergen::kDefaultMaxJoinAngle).toDouble());
    maxJoinAngle_->setToolTip(tr("Inferred joins turning more than this, measured over 8 voxels on each side, are cut. 180° keeps every join."));
    form->addRow(tr("Max join angle"), maxJoinAngle_);
    removeShort_ = new QCheckBox(tr("Remove fibers shorter than"), this);
    removeShort_->setChecked(settings.value(kRemoveShortKey, true).toBool());
    minLength_ = new QDoubleSpinBox(this);
    minLength_->setRange(1, 100000);
    minLength_->setDecimals(0);
    minLength_->setSuffix(tr(" voxels"));
    minLength_->setValue(settings.value(kMinLengthKey, vc3d::fibergen::kDefaultMinLength).toDouble());
    minLength_->setEnabled(removeShort_->isChecked());
    connect(removeShort_, &QCheckBox::toggled, minLength_, &QWidget::setEnabled);
    auto* shortRow = new QHBoxLayout;
    shortRow->addWidget(removeShort_);
    shortRow->addWidget(minLength_, 1);
    form->addRow(QString(), shortRow);
    blackDistance_ = new QDoubleSpinBox(this);
    blackDistance_->setRange(0, 1024);
    blackDistance_->setDecimals(0);
    blackDistance_->setSuffix(tr(" voxels"));
    blackDistance_->setSpecialValueText(tr("Keep them"));
    blackDistance_->setValue(settings.value(kBlackDistanceKey, vc3d::fibergen::kDefaultBlackDistance).toDouble());
    blackDistance_->setToolTip(tr("Fibers passing this close to the black outside the papyrus (CT value 0) are removed."));
    form->addRow(tr("Remove fibers near the outside black"), blackDistance_);

    python_ = new QLineEdit(settings.value(kPythonKey).toString(), this);
    python_->setPlaceholderText(tr("Detect automatically"));
    python_->setToolTip(tr("Python with the vesuvius package and its model dependencies."));
    auto* pythonBrowse = new QPushButton(tr("Browse…"), this);
    auto* pythonRow = new QHBoxLayout;
    pythonRow->addWidget(python_, 1);
    pythonRow->addWidget(pythonBrowse);
    form->addRow(tr("Python"), pythonRow);
    connect(pythonBrowse, &QPushButton::clicked, this, [this]() {
        const auto path = QFileDialog::getOpenFileName(this, tr("Python executable"), python_->text());
        if (!path.isEmpty())
            python_->setText(path);
    });
    layout->addLayout(form);

    if (voxelSizeUm > 0 && (voxelSizeUm < kDefaultModelMinVoxelUm || voxelSizeUm > kDefaultModelMaxVoxelUm)) {
        auto* note = new QLabel(
            tr("The default model was trained on 8.64 and 9.36 µm scans; this volume has %1 µm voxels.").arg(voxelSizeUm, 0, 'g', 4),
            this);
        note->setWordWrap(true);
        layout->addWidget(note);
    }

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Cancel, this);
    buttons->addButton(tr("Create"), QDialogButtonBox::AcceptRole);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layout->addWidget(buttons);
    updateZone();
}

std::array<int, 3> FiberCollectionGeneratorDialog::zoneSize() const
{
    return {size_[0]->value(), size_[1]->value(), size_[2]->value()};
}

std::array<int, 3> FiberCollectionGeneratorDialog::zoneOrigin() const
{
    return vc3d::fibergen::zoneOrigin({center_[0]->value(), center_[1]->value(), center_[2]->value()}, zoneSize(), shape_);
}

std::vector<vc3d::fibergen::Block> FiberCollectionGeneratorDialog::blocks() const
{
    return vc3d::fibergen::zoneBlocks(zoneOrigin(), zoneSize(), blockSize_->value());
}

void FiberCollectionGeneratorDialog::updateZone()
{
    const auto from = zoneOrigin();
    const auto extent = zoneSize();
    const auto count = blocks().size();
    const auto work = count == 1 ? tr("1 block to process") : tr("%1 blocks to process").arg(int(count));
    zone_->setText(tr("X %1–%2, Y %3–%4, Z %5–%6 of the current volume · %7")
                       .arg(from[0]).arg(from[0] + extent[0] - 1)
                       .arg(from[1]).arg(from[1] + extent[1] - 1)
                       .arg(from[2]).arg(from[2] + extent[2] - 1)
                       .arg(work));
    emit zoneChanged();
}

vc3d::fibergen::Request FiberCollectionGeneratorDialog::request() const
{
    vc3d::fibergen::Request request;
    request.origin = zoneOrigin();
    request.size = zoneSize();
    request.output = output_->text().trimmed();
    request.model = model_->text().trimmed().isEmpty() ? QString(vc3d::fibergen::kDefaultModel) : model_->text().trimmed();
    request.mirror = mirror_->isChecked();
    request.thresholdPercent = threshold_->value();
    request.blockSize = blockSize_->value();
    request.extend = extend_->isChecked();
    request.maxJoinAngle = maxJoinAngle_->value();
    request.minLength = removeShort_->isChecked() ? minLength_->value() : 0.0;
    request.blackDistance = blackDistance_->value();
    return request;
}

QString FiberCollectionGeneratorDialog::pythonExecutable() const
{
    return python_->text().trimmed();
}

void FiberCollectionGeneratorDialog::accept()
{
    auto path = output_->text().trimmed();
    if (path.isEmpty()) {
        QMessageBox::warning(this, windowTitle(), tr("Choose where to save the new volume."));
        return;
    }
    if (QFileInfo(path).suffix().compare("afv", Qt::CaseInsensitive) != 0)
        path += ".afv";
    const QFileInfo file(path);
    if (!file.isAbsolute() || !file.dir().exists()) {
        QMessageBox::warning(this, windowTitle(), tr("Choose a file in an existing folder."));
        return;
    }
    if (file.exists()) {
        QMessageBox::warning(this, windowTitle(), tr("%1 already exists. Existing volumes are never replaced; choose a new name.").arg(path));
        return;
    }
    output_->setText(path);
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(kDirectoryKey, file.absolutePath());
    const auto extent = zoneSize();
    settings.setValue(kSizeKey, *std::max_element(extent.begin(), extent.end()));
    settings.setValue(kModelKey, request().model);
    settings.setValue(kMirrorKey, mirror_->isChecked());
    settings.setValue(kThresholdKey, threshold_->value());
    settings.setValue(kBlockSizeKey, blockSize_->value());
    settings.setValue(kExtendKey, extend_->isChecked());
    settings.setValue(kMaxJoinAngleKey, maxJoinAngle_->value());
    settings.setValue(kRemoveShortKey, removeShort_->isChecked());
    settings.setValue(kMinLengthKey, minLength_->value());
    settings.setValue(kBlackDistanceKey, blackDistance_->value());
    settings.setValue(kPythonKey, pythonExecutable());
    QDialog::accept();
}
