#include "GrowPatchSeedDialog.hpp"

#include <algorithm>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QSpinBox>
#include <QVBoxLayout>

bool selectGrowPatchSeedParams(QWidget* parent,
                               const QVector<VolumeSelector::VolumeOption>& volumeOptions,
                               GrowPatchSeedParams& params)
{
    QDialog dlg(parent);
    dlg.setWindowTitle(QObject::tr("Create Segment (GrowPatch)"));

    auto* layout = new QVBoxLayout(&dlg);
    auto* form = new QFormLayout();

    auto* volumeCombo = new QComboBox(&dlg);
    for (const auto& option : volumeOptions) {
        const QString label = option.name.isEmpty()
            ? option.id
            : QStringLiteral("%1 (%2)").arg(option.name, option.id);
        volumeCombo->addItem(label, option.id);
        volumeCombo->setItemData(volumeCombo->count() - 1, option.path, Qt::UserRole + 1);
    }
    if (!params.volumeId.isEmpty()) {
        const int idx = volumeCombo->findData(params.volumeId);
        if (idx >= 0) {
            volumeCombo->setCurrentIndex(idx);
        }
    }
    form->addRow(QObject::tr("Volume:"), volumeCombo);

    auto* methodCombo = new QComboBox(&dlg);
    methodCombo->setObjectName(QStringLiteral("growthMethod"));
    methodCombo->addItem(QObject::tr("GrowPatch (default)"), QStringLiteral("default"));
    methodCombo->addItem(QObject::tr("Projection"), QStringLiteral("projection"));
    methodCombo->addItem(QObject::tr("Streamline"), QStringLiteral("streamline"));
    methodCombo->setCurrentIndex(std::max(0, methodCombo->findData(params.growthMethod)));
    form->addRow(QObject::tr("Growth mode:"), methodCombo);

    auto* iterationsSpin = new QSpinBox(&dlg);
    iterationsSpin->setObjectName(QStringLiteral("iterations"));
    iterationsSpin->setRange(1, 100000);
    iterationsSpin->setValue(std::clamp(params.iterations, 1, 100000));
    form->addRow(QObject::tr("Iterations:"), iterationsSpin);

    auto* minAreaSpin = new QDoubleSpinBox(&dlg);
    minAreaSpin->setRange(0.0, 1000000.0);
    minAreaSpin->setDecimals(6);
    minAreaSpin->setSuffix(QObject::tr(" cm²"));
    minAreaSpin->setSingleStep(0.001);
    minAreaSpin->setValue(std::clamp(params.minAreaCm, 0.0, 1000000.0));
    minAreaSpin->setToolTip(QObject::tr("Writes the min_area_cm parameter for vc_grow_seg_from_seed."));
    form->addRow(QObject::tr("Min size:"), minAreaSpin);

    auto* tagEdit = new QLineEdit(params.outputTag, &dlg);
    tagEdit->setToolTip(QObject::tr("Tag for the new surface."));
    form->addRow(QObject::tr("Output tag:"), tagEdit);

    auto* spacingSpin = new QDoubleSpinBox(&dlg);
    spacingSpin->setObjectName(QStringLiteral("meshSpacing"));
    spacingSpin->setRange(0.1, 1000.0);
    spacingSpin->setDecimals(2);
    spacingSpin->setValue(params.meshSpacing);
    spacingSpin->setSuffix(QObject::tr(" voxels"));
    form->addRow(QObject::tr("Mesh spacing:"), spacingSpin);

    auto* stepSpin = new QDoubleSpinBox(&dlg);
    stepSpin->setObjectName(QStringLiteral("integrationStep"));
    stepSpin->setRange(0.0, 1000.0);
    stepSpin->setDecimals(2);
    stepSpin->setValue(params.integrationStep);
    stepSpin->setSuffix(QObject::tr(" voxels"));
    stepSpin->setSpecialValueText(QObject::tr("One step per vertex"));
    form->addRow(QObject::tr("Max integration step:"), stepSpin);

    auto* integratorCombo = new QComboBox(&dlg);
    integratorCombo->setObjectName(QStringLiteral("integrator"));
    integratorCombo->addItem(QObject::tr("Midpoint"), QStringLiteral("midpoint"));
    integratorCombo->addItem(QObject::tr("Heun"), QStringLiteral("heun"));
    integratorCombo->addItem(QObject::tr("RK4"), QStringLiteral("rk4"));
    integratorCombo->setCurrentIndex(std::max(0, integratorCombo->findData(params.integrator)));
    form->addRow(QObject::tr("Integrator:"), integratorCombo);

    auto* workersSpin = new QSpinBox(&dlg);
    workersSpin->setObjectName(QStringLiteral("workers"));
    workersSpin->setRange(1, 256);
    workersSpin->setValue(params.workers);
    form->addRow(QObject::tr("Workers:"), workersSpin);

    auto* description = new QLabel(&dlg);
    description->setWordWrap(true);
    bool previousSheetMode = params.growthMethod != QStringLiteral("default");
    int optimizerIterations = previousSheetMode ? 200 : iterationsSpin->value();
    int sheetIterations = previousSheetMode ? iterationsSpin->value() : 20;
    const auto updateMode = [&]() {
        const auto method = methodCombo->currentData().toString();
        const bool sheet = method != QStringLiteral("default");
        spacingSpin->setEnabled(sheet);
        stepSpin->setEnabled(sheet);
        workersSpin->setEnabled(sheet);
        integratorCombo->setEnabled(method == QStringLiteral("streamline"));
        if (sheet != previousSheetMode) {
            (previousSheetMode ? sheetIterations : optimizerIterations) = iterationsSpin->value();
            iterationsSpin->setMaximum(sheet ? 1024 : 100000);
            iterationsSpin->setValue(sheet ? sheetIterations : optimizerIterations);
            previousSheetMode = sheet;
        }
        iterationsSpin->setMaximum(sheet ? 1024 : 100000);
        description->setText(sheet
            ? QObject::tr("Use a raw CT volume. Iterations set the number of cells on each side of the fixed seed; "
                          "N iterations produce a (2N+1) × (2N+1) grid. Growth stops with an error if CT normals "
                          "are unavailable or the patch leaves the volume.")
            : QObject::tr("Use the existing GrowPatch optimizer and normal grid."));
    };
    QObject::connect(methodCombo, &QComboBox::currentIndexChanged, &dlg, updateMode);
    updateMode();
    layout->addLayout(form);
    layout->addWidget(description);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
    layout->addWidget(buttons);

    QObject::connect(buttons, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dlg, [&]() {
        const QString tag = tagEdit->text().trimmed();
        if (tag.isEmpty()) {
            QMessageBox::warning(&dlg, QObject::tr("Create Segment"), QObject::tr("Output tag cannot be empty."));
            return;
        }
        if (volumeCombo->currentIndex() < 0) {
            QMessageBox::warning(&dlg, QObject::tr("Create Segment"), QObject::tr("Select a volume."));
            return;
        }
        params.volumeId = volumeCombo->currentData().toString();
        params.iterations = iterationsSpin->value();
        params.minAreaCm = minAreaSpin->value();
        params.outputTag = tag;
        params.growthMethod = methodCombo->currentData().toString();
        params.meshSpacing = spacingSpin->value();
        params.integrationStep = stepSpin->value();
        params.integrator = integratorCombo->currentData().toString();
        params.workers = workersSpin->value();
        dlg.accept();
    });

    return dlg.exec() == QDialog::Accepted;
}
