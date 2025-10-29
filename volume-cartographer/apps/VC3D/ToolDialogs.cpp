#include "ToolDialogs.hpp"

#include "VCSettings.hpp"

#include <QFormLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QFileDialog>
#include <QLabel>
#include <QGroupBox>
#include <QFontMetrics>
#include <QSizePolicy>
#include <QList>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>

#include <cmath>

// ----- helper creators -----
static QWidget* pathPicker(QWidget* parent, QLineEdit*& lineOut, const QString& dialogTitle, bool dirMode) {
    auto w = new QWidget(parent);
    auto lay = new QHBoxLayout(w);
    lay->setContentsMargins(0,0,0,0);
    lineOut = new QLineEdit(w);
    auto btn = new QPushButton("…", w);
    lay->addWidget(lineOut);
    lay->addWidget(btn);
    QObject::connect(btn, &QPushButton::clicked, w, [parent, lineOut, dialogTitle, dirMode]() {
        if (dirMode) {
            const QString dir = QFileDialog::getExistingDirectory(parent, dialogTitle, lineOut->text());
            if (!dir.isEmpty()) lineOut->setText(dir);
        } else {
            const QString file = QFileDialog::getOpenFileName(parent, dialogTitle, lineOut->text());
            if (!file.isEmpty()) lineOut->setText(file);
        }
    });
    return w;
}

static void ensureDialogWidthForEdits(QDialog* dlg, const QList<QLineEdit*>& edits, int extra = 280, int maxW = 1600) {
    QFontMetrics fm(dlg->font());
    int need = 0;
    for (auto* e : edits) {
        if (!e) continue;
        e->setMinimumWidth(800); // ensure at least 800px visible for path-like text
        e->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        int w = fm.horizontalAdvance(e->text()) + 10; // small padding so text isn't tight
        need = std::max(need, w);
    }
    dlg->adjustSize();
    int target = std::min(std::max(dlg->width(), need + extra), maxW);
    dlg->resize(target, dlg->height());
}

// ================= RenderParamsDialog =================
RenderParamsDialog::RenderParamsDialog(QWidget* parent,
                                       const QString& volumePath,
                                       const QString& segmentPath,
                                       const QString& outputPattern,
                                       double scale,
                                       int groupIdx,
                                       int numSlices)
    : QDialog(parent)
{
    setWindowTitle("Render Parameters");
    auto main = new QVBoxLayout(this);

    // Basic params
    auto basicBox = new QGroupBox("Basic", this);
    auto basic = new QFormLayout(basicBox);
    basicBox->setLayout(basic);

    QWidget* volPick = pathPicker(this, edtVolume_, "Select OME-Zarr volume", true);
    edtSegment_ = new QLineEdit(this);
    QWidget* outPick = pathPicker(this, edtOutput_, "Select output (.zarr or tif pattern)", false);
    spScale_ = new QDoubleSpinBox(this); spScale_->setDecimals(3); spScale_->setRange(0.0001, 10000.0);
    spGroup_ = new QSpinBox(this); spGroup_->setRange(0, 10);
    spSlices_ = new QSpinBox(this); spSlices_->setRange(1, 1000);
    edtThreads_ = new QLineEdit(this); edtThreads_->setPlaceholderText("optional");
    edtThreads_->setValidator(new QRegularExpressionValidator(QRegularExpression("^\\s*\\d*\\s*$"), this));

    edtVolume_->setText(volumePath);
    edtSegment_->setText(segmentPath);
    edtOutput_->setText(outputPattern);
    spScale_->setValue(scale);
    spGroup_->setValue(groupIdx);
    spSlices_->setValue(numSlices);

    basic->addRow("Volume:", volPick);
    basic->addRow("Segmentation (tifxyz dir):", edtSegment_);
    chkIncludeTifs_ = new QCheckBox("Also write TIFF slices (Zarr)", this);
    chkIncludeTifs_->setChecked(false);

    basic->addRow("Output:", outPick);
    basic->addRow("", chkIncludeTifs_);
    basic->addRow("Scale (Pg):", spScale_);
    basic->addRow("Group index:", spGroup_);
    basic->addRow("Num slices:", spSlices_);
    basic->addRow("OMP threads:", edtThreads_);

    // Advanced
    auto advBox = new QGroupBox("Advanced (optional)", this);
    advBox->setCheckable(true);
    advBox->setChecked(false);
    auto adv = new QFormLayout(advBox);
    advBox->setLayout(adv);

    spCropX_ = new QSpinBox(this); spCropX_->setRange(0, 1000000);
    spCropY_ = new QSpinBox(this); spCropY_->setRange(0, 1000000);
    spCropW_ = new QSpinBox(this); spCropW_->setRange(0, 1000000); spCropW_->setValue(0);
    spCropH_ = new QSpinBox(this); spCropH_->setRange(0, 1000000); spCropH_->setValue(0);
    QWidget* affPick = pathPicker(this, edtAffine_, "Select affine JSON", false);
    chkInvert_ = new QCheckBox("Invert affine", this);
    spScaleSeg_ = new QDoubleSpinBox(this); spScaleSeg_->setDecimals(3); spScaleSeg_->setRange(0.0001, 1000.0); spScaleSeg_->setValue(1.0);
    spRotate_ = new QDoubleSpinBox(this); spRotate_->setDecimals(2); spRotate_->setRange(-360.0, 360.0); spRotate_->setValue(0.0);
    cmbFlip_ = new QComboBox(this);
    cmbFlip_->addItem("None", -1);
    cmbFlip_->addItem("Vertical", 0);
    cmbFlip_->addItem("Horizontal", 1);
    cmbFlip_->addItem("Both", 2);

    adv->addRow("Crop X:", spCropX_);
    adv->addRow("Crop Y:", spCropY_);
    adv->addRow("Crop Width:", spCropW_);
    adv->addRow("Crop Height:", spCropH_);
    adv->addRow("Affine transform:", affPick);
    adv->addRow("Invert affine:", chkInvert_);
    adv->addRow("Scale segmentation:", spScaleSeg_);
    adv->addRow("Rotate (deg):", spRotate_);
    adv->addRow("Flip:", cmbFlip_);

    // Buttons
    auto btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    auto btnReset = btns->addButton("Reset to Defaults", QDialogButtonBox::ResetRole);
    auto btnSave  = btns->addButton("Save as Default", QDialogButtonBox::ActionRole);
    connect(btns, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(btns, &QDialogButtonBox::rejected, this, &QDialog::reject);

    main->addWidget(basicBox);
    main->addWidget(advBox);
    main->addWidget(btns);

    // Enable TIFF export only when output seems to be a Zarr
    auto updateIncludeTifsEnabled = [this]() {
        const QString t = edtOutput_->text().trimmed();
        const bool isZarr = t.endsWith(".zarr", Qt::CaseInsensitive);
        chkIncludeTifs_->setEnabled(isZarr);
        if (!isZarr) chkIncludeTifs_->setChecked(false);
    };
    updateIncludeTifsEnabled();
    connect(edtOutput_, &QLineEdit::textChanged, this, [updateIncludeTifsEnabled](const QString&){ updateIncludeTifsEnabled(); });

    ensureDialogWidthForEdits(this, QList<QLineEdit*>{ edtVolume_, edtSegment_, edtOutput_, edtAffine_ });

    // Apply saved defaults to optional controls, then session overrides
    applySavedDefaults();
    applySessionDefaults();
    connect(btnReset, &QPushButton::clicked, this, [this]() {
        QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
        s.beginGroup("render/defaults");
        const bool hasAny = s.allKeys().size() > 0;
        s.endGroup();
        if (hasAny) applySavedDefaults(); else applyCodeDefaults();
    });
    connect(btnSave, &QPushButton::clicked, this, [this]() { saveDefaults(); });
    connect(btns, &QDialogButtonBox::accepted, this, [this]() { updateSessionFromUI(); });
}

QString RenderParamsDialog::volumePath() const { return edtVolume_->text(); }
QString RenderParamsDialog::segmentPath() const { return edtSegment_->text(); }
QString RenderParamsDialog::outputPattern() const { return edtOutput_->text(); }
double RenderParamsDialog::scale() const { return spScale_->value(); }
int RenderParamsDialog::groupIdx() const { return spGroup_->value(); }
int RenderParamsDialog::numSlices() const { return spSlices_->value(); }
int RenderParamsDialog::ompThreads() const {
    const QString t = edtThreads_->text().trimmed();
    if (t.isEmpty()) return -1;
    bool ok=false; int v = t.toInt(&ok); return (ok && v>0) ? v : -1;
}
int RenderParamsDialog::cropX() const { return spCropX_->value(); }
int RenderParamsDialog::cropY() const { return spCropY_->value(); }
int RenderParamsDialog::cropWidth() const { return spCropW_->value(); }
int RenderParamsDialog::cropHeight() const { return spCropH_->value(); }
QString RenderParamsDialog::affinePath() const { return edtAffine_->text(); }
bool RenderParamsDialog::invertAffine() const { return chkInvert_->isChecked(); }
double RenderParamsDialog::scaleSegmentation() const { return spScaleSeg_->value(); }
double RenderParamsDialog::rotateDegrees() const { return spRotate_->value(); }
int RenderParamsDialog::flipAxis() const { return cmbFlip_->currentData().toInt(); }
bool RenderParamsDialog::includeTifs() const { return chkIncludeTifs_->isChecked(); }

// ---- RenderParamsDialog: defaults + session helpers ----
bool RenderParamsDialog::s_haveSession = false;
bool RenderParamsDialog::s_includeTifs = false;
int  RenderParamsDialog::s_cropX = 0;
int  RenderParamsDialog::s_cropY = 0;
int  RenderParamsDialog::s_cropW = 0;
int  RenderParamsDialog::s_cropH = 0;
bool RenderParamsDialog::s_invertAffine = false;
double RenderParamsDialog::s_scaleSeg = 1.0;
double RenderParamsDialog::s_rotateDeg = 0.0;
int  RenderParamsDialog::s_flipAxis = -1;
int  RenderParamsDialog::s_ompThreads = -1;

void RenderParamsDialog::applyCodeDefaults() {
    chkIncludeTifs_->setChecked(false);
    spCropX_->setValue(0);
    spCropY_->setValue(0);
    spCropW_->setValue(0);
    spCropH_->setValue(0);
    chkInvert_->setChecked(false);
    spScaleSeg_->setValue(1.0);
    spRotate_->setValue(0.0);
    // Flip index: 0 None, 1 Vertical, 2 Horizontal, 3 Both; but we stored data -1/0/1/2
    // Set by data match
    int idx = cmbFlip_->findData(-1);
    if (idx >= 0) cmbFlip_->setCurrentIndex(idx);
    edtThreads_->setText("");
}

void RenderParamsDialog::applySavedDefaults() {
    QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
    s.beginGroup("render/defaults");
    chkIncludeTifs_->setChecked(s.value("include_tifs", chkIncludeTifs_->isChecked()).toBool());
    spCropX_->setValue(s.value("crop_x", spCropX_->value()).toInt());
    spCropY_->setValue(s.value("crop_y", spCropY_->value()).toInt());
    spCropW_->setValue(s.value("crop_w", spCropW_->value()).toInt());
    spCropH_->setValue(s.value("crop_h", spCropH_->value()).toInt());
    chkInvert_->setChecked(s.value("invert_affine", chkInvert_->isChecked()).toBool());
    spScaleSeg_->setValue(s.value("scale_segmentation", spScaleSeg_->value()).toDouble());
    spRotate_->setValue(s.value("rotate_deg", spRotate_->value()).toDouble());
    // flip axis stored as int data
    const int flip = s.value("flip_axis", cmbFlip_->currentData().toInt()).toInt();
    int idx = cmbFlip_->findData(flip);
    if (idx >= 0) cmbFlip_->setCurrentIndex(idx);
    const int th = s.value("omp_threads", -1).toInt();
    edtThreads_->setText(th > 0 ? QString::number(th) : "");
    s.endGroup();
}

void RenderParamsDialog::applySessionDefaults() {
    if (!s_haveSession) return;
    chkIncludeTifs_->setChecked(s_includeTifs);
    spCropX_->setValue(s_cropX);
    spCropY_->setValue(s_cropY);
    spCropW_->setValue(s_cropW);
    spCropH_->setValue(s_cropH);
    chkInvert_->setChecked(s_invertAffine);
    spScaleSeg_->setValue(s_scaleSeg);
    spRotate_->setValue(s_rotateDeg);
    int idx = cmbFlip_->findData(s_flipAxis);
    if (idx >= 0) cmbFlip_->setCurrentIndex(idx);
    edtThreads_->setText(s_ompThreads > 0 ? QString::number(s_ompThreads) : "");
}

void RenderParamsDialog::saveDefaults() const {
    QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
    s.beginGroup("render/defaults");
    s.setValue("include_tifs", chkIncludeTifs_->isChecked());
    s.setValue("crop_x", spCropX_->value());
    s.setValue("crop_y", spCropY_->value());
    s.setValue("crop_w", spCropW_->value());
    s.setValue("crop_h", spCropH_->value());
    s.setValue("invert_affine", chkInvert_->isChecked());
    s.setValue("scale_segmentation", spScaleSeg_->value());
    s.setValue("rotate_deg", spRotate_->value());
    s.setValue("flip_axis", cmbFlip_->currentData().toInt());
    s.setValue("omp_threads", ompThreads());
    s.endGroup();
}

void RenderParamsDialog::updateSessionFromUI() {
    s_haveSession = true;
    s_includeTifs = chkIncludeTifs_->isChecked();
    s_cropX = spCropX_->value();
    s_cropY = spCropY_->value();
    s_cropW = spCropW_->value();
    s_cropH = spCropH_->value();
    s_invertAffine = chkInvert_->isChecked();
    s_scaleSeg = spScaleSeg_->value();
    s_rotateDeg = spRotate_->value();
    s_flipAxis = cmbFlip_->currentData().toInt();
    s_ompThreads = ompThreads();
}

// ================= TraceParamsDialog =================
TraceParamsDialog::TraceParamsDialog(QWidget* parent,
                                     const QString& volumePath,
                                     const QString& srcDir,
                                     const QString& tgtDir,
                                     const QString& jsonParams,
                                     const QString& srcSegment)
    : QDialog(parent)
{
    setWindowTitle("Run Trace Parameters");
    auto main = new QVBoxLayout(this);

    // Files/paths
    auto pathsBox = new QGroupBox("Paths", this);
    auto paths = new QFormLayout(pathsBox);
    pathsBox->setLayout(paths);

    QWidget* volPick = pathPicker(this, edtVolume_, "Select OME-Zarr volume", true);
    QWidget* srcPick = pathPicker(this, edtSrcDir_, "Select source directory (paths)", true);
    QWidget* tgtPick = pathPicker(this, edtTgtDir_, "Select target directory (traces)", true);
    QWidget* jsonPick = pathPicker(this, edtJson_, "Select trace params JSON", false);
    QWidget* segPick = pathPicker(this, edtSrcSegment_, "Select source segment (tifxyz dir)", true);
    edtThreads_ = new QLineEdit(this); edtThreads_->setPlaceholderText("optional");
    edtThreads_->setValidator(new QRegularExpressionValidator(QRegularExpression("^\\s*\\d*\\s*$"), this));

    edtVolume_->setText(volumePath);
    edtSrcDir_->setText(srcDir);
    edtTgtDir_->setText(tgtDir);
    edtJson_->setText(jsonParams);
    edtSrcSegment_->setText(srcSegment);

    paths->addRow("Volume:", volPick);
    paths->addRow("Source dir:", srcPick);
    paths->addRow("Target dir:", tgtPick);
    paths->addRow("JSON params:", jsonPick);
    paths->addRow("Source segment:", segPick);
    paths->addRow("OMP threads:", edtThreads_);

    // Advanced params
    auto advBox = new QGroupBox("Tracing Parameters", this);
    auto adv = new QFormLayout(advBox);
    advBox->setLayout(adv);

    chkFlipX_ = new QCheckBox("Flip X after first gen", this);
    spGlobalStepsWin_ = new QSpinBox(this); spGlobalStepsWin_->setRange(0, 1000000); spGlobalStepsWin_->setValue(0);
    spSrcStep_ = new QDoubleSpinBox(this); spSrcStep_->setRange(0.01, 1e6); spSrcStep_->setDecimals(3); spSrcStep_->setValue(20.0);
    spStep_ = new QDoubleSpinBox(this); spStep_->setRange(0.01, 1e6); spStep_->setDecimals(3); spStep_->setValue(10.0);
    spMaxWidth_ = new QSpinBox(this); spMaxWidth_->setRange(1, 100000000); spMaxWidth_->setValue(80000);

    spLocalCostInlTh_ = new QDoubleSpinBox(this); spLocalCostInlTh_->setRange(0.0, 1000.0); spLocalCostInlTh_->setDecimals(4); spLocalCostInlTh_->setValue(0.2);
    spSameSurfaceTh_ = new QDoubleSpinBox(this); spSameSurfaceTh_->setRange(0.0, 1000.0); spSameSurfaceTh_->setDecimals(4); spSameSurfaceTh_->setValue(2.0);
    spStraightW_ = new QDoubleSpinBox(this); spStraightW_->setRange(0.0, 1000.0); spStraightW_->setDecimals(4); spStraightW_->setValue(0.7);
    spStraightW3D_ = new QDoubleSpinBox(this); spStraightW3D_->setRange(0.0, 1000.0); spStraightW3D_->setDecimals(4); spStraightW3D_->setValue(4.0);
    spSlidingWScale_ = new QDoubleSpinBox(this); spSlidingWScale_->setRange(0.0, 1000.0); spSlidingWScale_->setDecimals(3); spSlidingWScale_->setValue(1.0);
    spZLocLossW_ = new QDoubleSpinBox(this); spZLocLossW_->setRange(0.0, 1000.0); spZLocLossW_->setDecimals(4); spZLocLossW_->setValue(0.1);
    spDistLoss2DW_ = new QDoubleSpinBox(this); spDistLoss2DW_->setRange(0.0, 1000.0); spDistLoss2DW_->setDecimals(4); spDistLoss2DW_->setValue(1.0);
    spDistLoss3DW_ = new QDoubleSpinBox(this); spDistLoss3DW_->setRange(0.0, 1000.0); spDistLoss3DW_->setDecimals(4); spDistLoss3DW_->setValue(2.0);
    spStraightMinCount_ = new QDoubleSpinBox(this); spStraightMinCount_->setRange(0.0, 1000.0); spStraightMinCount_->setDecimals(3); spStraightMinCount_->setValue(1.0);
    spInlierBaseTh_ = new QSpinBox(this); spInlierBaseTh_->setRange(0, 1000000); spInlierBaseTh_->setValue(20);
    spConsensusDefaultTh_ = new QSpinBox(this); spConsensusDefaultTh_->setRange(0, 1000000); spConsensusDefaultTh_->setValue(10);

    chkZRange_ = new QCheckBox("Enforce Z range", this);
    spZMin_ = new QDoubleSpinBox(this); spZMin_->setRange(-1e9, 1e9); spZMin_->setDecimals(3);
    spZMax_ = new QDoubleSpinBox(this); spZMax_->setRange(-1e9, 1e9); spZMax_->setDecimals(3);

    adv->addRow("Flip X:", chkFlipX_);
    adv->addRow("Global steps/window:", spGlobalStepsWin_);
    adv->addRow("Source step:", spSrcStep_);
    adv->addRow("Step:", spStep_);
    adv->addRow("Max width:", spMaxWidth_);
    adv->addRow("Local cost inlier th:", spLocalCostInlTh_);
    adv->addRow("Same-surface th:", spSameSurfaceTh_);
    adv->addRow("Straight weight (2D):", spStraightW_);
    adv->addRow("Straight weight (3D):", spStraightW3D_);
    adv->addRow("Sliding window scale:", spSlidingWScale_);
    adv->addRow("Z-loc loss w:", spZLocLossW_);
    adv->addRow("Dist loss 2D w:", spDistLoss2DW_);
    adv->addRow("Dist loss 3D w:", spDistLoss3DW_);
    adv->addRow("Straight min count:", spStraightMinCount_);
    adv->addRow("Inlier base threshold:", spInlierBaseTh_);
    adv->addRow("Consensus default th:", spConsensusDefaultTh_);
    adv->addRow("Use Z range:", chkZRange_);
    adv->addRow("Z min:", spZMin_);
    adv->addRow("Z max:", spZMax_);

    // Apply saved defaults (overrides code defaults), then overlay JSON if present
    applySavedDefaults();

    // Prefill from JSON if present
    if (!jsonParams.isEmpty()) {
        QFile f(jsonParams);
        if (f.open(QIODevice::ReadOnly)) {
            const auto doc = QJsonDocument::fromJson(f.readAll());
            f.close();
            if (doc.isObject()) {
                const auto o = doc.object();
                chkFlipX_->setChecked(o.value("flip_x").toInt(0) != 0);
                spGlobalStepsWin_->setValue(o.value("global_steps_per_window").toInt(0));
                spSrcStep_->setValue(o.value("src_step").toDouble(20.0));
                spStep_->setValue(o.value("step").toDouble(10.0));
                spMaxWidth_->setValue(o.value("max_width").toInt(80000));
                spLocalCostInlTh_->setValue(o.value("local_cost_inl_th").toDouble(0.2));
                spSameSurfaceTh_->setValue(o.value("same_surface_th").toDouble(2.0));
                spStraightW_->setValue(o.value("straight_weight").toDouble(0.7));
                spStraightW3D_->setValue(o.value("straight_weight_3D").toDouble(4.0));
                spSlidingWScale_->setValue(o.value("sliding_w_scale").toDouble(1.0));
                spZLocLossW_->setValue(o.value("z_loc_loss_w").toDouble(0.1));
                spDistLoss2DW_->setValue(o.value("dist_loss_2d_w").toDouble(1.0));
                spDistLoss3DW_->setValue(o.value("dist_loss_3d_w").toDouble(2.0));
                spStraightMinCount_->setValue(o.value("straight_min_count").toDouble(1.0));
                spInlierBaseTh_->setValue(o.value("inlier_base_threshold").toInt(20));
                spConsensusDefaultTh_->setValue(o.value("consensus_default_th").toInt(spConsensusDefaultTh_->value()));
                if (o.contains("z_range") && o.value("z_range").isArray()) {
                    const auto a = o.value("z_range").toArray();
                    if (a.size() == 2) {
                        chkZRange_->setChecked(true);
                        spZMin_->setValue(a[0].toDouble());
                        spZMax_->setValue(a[1].toDouble());
                    }
                } else if (o.contains("z_min") && o.contains("z_max")) {
                    chkZRange_->setChecked(true);
                    spZMin_->setValue(o.value("z_min").toDouble());
                    spZMax_->setValue(o.value("z_max").toDouble());
                }
            }
        }
    }

    // Buttons
    auto btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    auto btnReset = btns->addButton("Reset to Defaults", QDialogButtonBox::ResetRole);
    auto btnSave  = btns->addButton("Save as Default", QDialogButtonBox::ActionRole);
    connect(btns, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(btns, &QDialogButtonBox::rejected, this, &QDialog::reject);
    // Apply session overrides after JSON and connect accept to snapshot session values
    applySessionDefaults();
    connect(btns, &QDialogButtonBox::accepted, this, [this]() { updateSessionFromUI(); });
    connect(btnReset, &QPushButton::clicked, this, [this]() {
        // Prefer saved defaults if available; otherwise code defaults
        QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
        s.beginGroup("trace/defaults");
        const bool hasAny = s.allKeys().size() > 0;
        s.endGroup();
        if (hasAny) applySavedDefaults(); else applyCodeDefaults();
    });
    connect(btnSave, &QPushButton::clicked, this, [this]() { saveDefaults(); });

    main->addWidget(pathsBox);
    main->addWidget(advBox);
    main->addWidget(btns);

    ensureDialogWidthForEdits(this, QList<QLineEdit*>{ edtVolume_, edtSrcDir_, edtTgtDir_, edtJson_, edtSrcSegment_ });
}

QString TraceParamsDialog::volumePath() const { return edtVolume_->text(); }
QString TraceParamsDialog::srcDir() const { return edtSrcDir_->text(); }
QString TraceParamsDialog::tgtDir() const { return edtTgtDir_->text(); }
QString TraceParamsDialog::jsonParams() const { return edtJson_->text(); }
QString TraceParamsDialog::srcSegment() const { return edtSrcSegment_->text(); }
int TraceParamsDialog::ompThreads() const {
    const QString t = edtThreads_->text().trimmed();
    if (t.isEmpty()) return -1;
    bool ok=false; int v = t.toInt(&ok); return (ok && v>0) ? v : -1;
}

QJsonObject TraceParamsDialog::makeParamsJson() const {
    QJsonObject o;
    o["flip_x"] = chkFlipX_->isChecked() ? 1 : 0;
    o["global_steps_per_window"] = spGlobalStepsWin_->value();
    o["src_step"] = spSrcStep_->value();
    o["step"] = spStep_->value();
    o["max_width"] = spMaxWidth_->value();

    o["local_cost_inl_th"] = spLocalCostInlTh_->value();
    o["same_surface_th"] = spSameSurfaceTh_->value();
    o["straight_weight"] = spStraightW_->value();
    o["straight_weight_3D"] = spStraightW3D_->value();
    o["sliding_w_scale"] = spSlidingWScale_->value();
    o["z_loc_loss_w"] = spZLocLossW_->value();
    o["dist_loss_2d_w"] = spDistLoss2DW_->value();
    o["dist_loss_3d_w"] = spDistLoss3DW_->value();
    o["straight_min_count"] = spStraightMinCount_->value();
    o["inlier_base_threshold"] = spInlierBaseTh_->value();
    o["consensus_default_th"] = spConsensusDefaultTh_->value();

    if (chkZRange_->isChecked()) {
        QJsonArray zr; zr.append(spZMin_->value()); zr.append(spZMax_->value());
        o["z_range"] = zr;
    }
    return o;
}

// ==== Defaults helpers ====
void TraceParamsDialog::applyCodeDefaults() {
    chkFlipX_->setChecked(false);
    spGlobalStepsWin_->setValue(0);
    spSrcStep_->setValue(20.0);
    spStep_->setValue(10.0);
    spMaxWidth_->setValue(80000);
    spLocalCostInlTh_->setValue(0.2);
    spSameSurfaceTh_->setValue(2.0);
    spStraightW_->setValue(0.7);
    spStraightW3D_->setValue(4.0);
    spSlidingWScale_->setValue(1.0);
    spZLocLossW_->setValue(0.1);
    spDistLoss2DW_->setValue(1.0);
    spDistLoss3DW_->setValue(2.0);
    spStraightMinCount_->setValue(1.0);
    spInlierBaseTh_->setValue(20);
    spConsensusDefaultTh_->setValue(10);
    chkZRange_->setChecked(false);
    spZMin_->setValue(0.0);
    spZMax_->setValue(0.0);
}

void TraceParamsDialog::applySavedDefaults() {
    QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
    s.beginGroup("trace/defaults");
    chkFlipX_->setChecked(s.value("flip_x", chkFlipX_->isChecked()).toInt() != 0);
    spGlobalStepsWin_->setValue(s.value("global_steps_per_window", spGlobalStepsWin_->value()).toInt());
    spSrcStep_->setValue(s.value("src_step", spSrcStep_->value()).toDouble());
    spStep_->setValue(s.value("step", spStep_->value()).toDouble());
    spMaxWidth_->setValue(s.value("max_width", spMaxWidth_->value()).toInt());

    spLocalCostInlTh_->setValue(s.value("local_cost_inl_th", spLocalCostInlTh_->value()).toDouble());
    spSameSurfaceTh_->setValue(s.value("same_surface_th", spSameSurfaceTh_->value()).toDouble());
    spStraightW_->setValue(s.value("straight_weight", spStraightW_->value()).toDouble());
    spStraightW3D_->setValue(s.value("straight_weight_3D", spStraightW3D_->value()).toDouble());
    spSlidingWScale_->setValue(s.value("sliding_w_scale", spSlidingWScale_->value()).toDouble());
    spZLocLossW_->setValue(s.value("z_loc_loss_w", spZLocLossW_->value()).toDouble());
    spDistLoss2DW_->setValue(s.value("dist_loss_2d_w", spDistLoss2DW_->value()).toDouble());
    spDistLoss3DW_->setValue(s.value("dist_loss_3d_w", spDistLoss3DW_->value()).toDouble());
    spStraightMinCount_->setValue(s.value("straight_min_count", spStraightMinCount_->value()).toDouble());
    spInlierBaseTh_->setValue(s.value("inlier_base_threshold", spInlierBaseTh_->value()).toInt());
    spConsensusDefaultTh_->setValue(s.value("consensus_default_th", spConsensusDefaultTh_->value()).toInt());

    const bool useZR = s.value("use_z_range", chkZRange_->isChecked()).toBool();
    chkZRange_->setChecked(useZR);
    spZMin_->setValue(s.value("z_min", spZMin_->value()).toDouble());
    spZMax_->setValue(s.value("z_max", spZMax_->value()).toDouble());
    s.endGroup();
}

// ---- TraceParamsDialog: session helpers ----
bool   TraceParamsDialog::s_haveSession = false;
bool   TraceParamsDialog::s_flipX = false;
int    TraceParamsDialog::s_globalStepsWin = 0;
double TraceParamsDialog::s_srcStep = 20.0;
double TraceParamsDialog::s_step = 10.0;
int    TraceParamsDialog::s_maxWidth = 80000;
double TraceParamsDialog::s_localCostInlTh = 0.2;
double TraceParamsDialog::s_sameSurfaceTh = 2.0;
double TraceParamsDialog::s_straightW = 0.7;
double TraceParamsDialog::s_straightW3D = 4.0;
double TraceParamsDialog::s_slidingWScale = 1.0;
double TraceParamsDialog::s_zLocLossW = 0.1;
double TraceParamsDialog::s_distLoss2DW = 1.0;
double TraceParamsDialog::s_distLoss3DW = 2.0;
double TraceParamsDialog::s_straightMinCount = 1.0;
int    TraceParamsDialog::s_inlierBaseTh = 20;
int    TraceParamsDialog::s_consensusDefaultTh = 10;
bool   TraceParamsDialog::s_useZRange = false;
double TraceParamsDialog::s_zMin = 0.0;
double TraceParamsDialog::s_zMax = 0.0;
int    TraceParamsDialog::s_ompThreads = -1;

void TraceParamsDialog::applySessionDefaults() {
    if (!s_haveSession) return;
    chkFlipX_->setChecked(s_flipX);
    spGlobalStepsWin_->setValue(s_globalStepsWin);
    spSrcStep_->setValue(s_srcStep);
    spStep_->setValue(s_step);
    spMaxWidth_->setValue(s_maxWidth);
    spLocalCostInlTh_->setValue(s_localCostInlTh);
    spSameSurfaceTh_->setValue(s_sameSurfaceTh);
    spStraightW_->setValue(s_straightW);
    spStraightW3D_->setValue(s_straightW3D);
    spSlidingWScale_->setValue(s_slidingWScale);
    spZLocLossW_->setValue(s_zLocLossW);
    spDistLoss2DW_->setValue(s_distLoss2DW);
    spDistLoss3DW_->setValue(s_distLoss3DW);
    spStraightMinCount_->setValue(s_straightMinCount);
    spInlierBaseTh_->setValue(s_inlierBaseTh);
    spConsensusDefaultTh_->setValue(s_consensusDefaultTh);
    chkZRange_->setChecked(s_useZRange);
    spZMin_->setValue(s_zMin);
    spZMax_->setValue(s_zMax);
    if (s_ompThreads > 0) edtThreads_->setText(QString::number(s_ompThreads)); else edtThreads_->setText("");
}

void TraceParamsDialog::updateSessionFromUI() {
    s_haveSession = true;
    s_flipX = chkFlipX_->isChecked();
    s_globalStepsWin = spGlobalStepsWin_->value();
    s_srcStep = spSrcStep_->value();
    s_step = spStep_->value();
    s_maxWidth = spMaxWidth_->value();
    s_localCostInlTh = spLocalCostInlTh_->value();
    s_sameSurfaceTh = spSameSurfaceTh_->value();
    s_straightW = spStraightW_->value();
    s_straightW3D = spStraightW3D_->value();
    s_slidingWScale = spSlidingWScale_->value();
    s_zLocLossW = spZLocLossW_->value();
    s_distLoss2DW = spDistLoss2DW_->value();
    s_distLoss3DW = spDistLoss3DW_->value();
    s_straightMinCount = spStraightMinCount_->value();
    s_inlierBaseTh = spInlierBaseTh_->value();
    s_consensusDefaultTh = spConsensusDefaultTh_->value();
    s_useZRange = chkZRange_->isChecked();
    s_zMin = spZMin_->value();
    s_zMax = spZMax_->value();
    const QString t = edtThreads_->text().trimmed();
    bool ok=false; const int v = t.toInt(&ok); s_ompThreads = (ok && v>0) ? v : -1;
}

void TraceParamsDialog::saveDefaults() const {
    QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
    s.beginGroup("trace/defaults");
    s.setValue("flip_x", chkFlipX_->isChecked() ? 1 : 0);
    s.setValue("global_steps_per_window", spGlobalStepsWin_->value());
    s.setValue("src_step", spSrcStep_->value());
    s.setValue("step", spStep_->value());
    s.setValue("max_width", spMaxWidth_->value());

    s.setValue("local_cost_inl_th", spLocalCostInlTh_->value());
    s.setValue("same_surface_th", spSameSurfaceTh_->value());
    s.setValue("straight_weight", spStraightW_->value());
    s.setValue("straight_weight_3D", spStraightW3D_->value());
    s.setValue("sliding_w_scale", spSlidingWScale_->value());
    s.setValue("z_loc_loss_w", spZLocLossW_->value());
    s.setValue("dist_loss_2d_w", spDistLoss2DW_->value());
    s.setValue("dist_loss_3d_w", spDistLoss3DW_->value());
    s.setValue("straight_min_count", spStraightMinCount_->value());
    s.setValue("inlier_base_threshold", spInlierBaseTh_->value());
    s.setValue("consensus_default_th", spConsensusDefaultTh_->value());

    s.setValue("use_z_range", chkZRange_->isChecked());
    s.setValue("z_min", spZMin_->value());
    s.setValue("z_max", spZMax_->value());
    s.endGroup();
}

// ================= ConvertToObjDialog =================
// static session members
bool   ConvertToObjDialog::s_haveSession = false;
bool   ConvertToObjDialog::s_normUV = false;
bool   ConvertToObjDialog::s_alignGrid = false;
int    ConvertToObjDialog::s_decimate = 0;
bool   ConvertToObjDialog::s_clean = false;
double ConvertToObjDialog::s_cleanK = 5.0;
int    ConvertToObjDialog::s_ompThreads = -1;

ConvertToObjDialog::ConvertToObjDialog(QWidget* parent,
                                       const QString& tifxyzPath,
                                       const QString& objOutPath)
    : QDialog(parent)
{
    setWindowTitle("Convert to OBJ");
    auto main = new QVBoxLayout(this);
    auto form = new QFormLayout();

    QWidget* tifPick = pathPicker(this, edtTifxyz_, "Select TIFXYZ directory", true);
    QWidget* objPick = pathPicker(this, edtObj_, "Select output OBJ file", false);
    chkNormalize_ = new QCheckBox("Normalize UV to [0,1]", this);
    chkAlign_ = new QCheckBox("Align grid (flatten Z per row)", this);
    spDecimate_ = new QSpinBox(this); spDecimate_->setRange(0, 10); spDecimate_->setValue(0);
    chkClean_ = new QCheckBox("Clean surface outliers", this);
    spCleanK_ = new QDoubleSpinBox(this); spCleanK_->setRange(0.0, 1000.0); spCleanK_->setDecimals(2); spCleanK_->setSingleStep(0.25); spCleanK_->setValue(5.0);
    spCleanK_->setEnabled(false);
    edtThreads_ = new QLineEdit(this); edtThreads_->setPlaceholderText("optional");
    edtThreads_->setValidator(new QRegularExpressionValidator(QRegularExpression("^\\s*\\d*\\s*$"), this));

    edtTifxyz_->setText(tifxyzPath);
    edtObj_->setText(objOutPath);

    form->addRow("TIFXYZ dir:", tifPick);
    form->addRow("OBJ file:", objPick);
    form->addRow("Decimate iters:", spDecimate_);
    form->addRow("", chkNormalize_);
    form->addRow("", chkAlign_);
    form->addRow("", chkClean_);
    form->addRow("Clean K (sigma):", spCleanK_);
    form->addRow("OMP threads:", edtThreads_);

    auto btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    // Defaults helpers
    auto btnReset = btns->addButton("Reset to Defaults", QDialogButtonBox::ResetRole);
    auto btnSave  = btns->addButton("Save as Default", QDialogButtonBox::ActionRole);
    connect(btns, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(btns, &QDialogButtonBox::rejected, this, &QDialog::reject);
    // Update session values on accept
    connect(btns, &QDialogButtonBox::accepted, this, [this]() { updateSessionFromUI(); });

    main->addLayout(form);
    main->addWidget(btns);

    ensureDialogWidthForEdits(this, QList<QLineEdit*>{ edtTifxyz_, edtObj_ });

    // Enable/disable K based on clean checkbox
    connect(chkClean_, &QCheckBox::toggled, spCleanK_, &QWidget::setEnabled);

    // Apply saved defaults, then session overrides
    applySavedDefaults();
    applySessionDefaults();
    // Reset / Save handlers
    connect(btnReset, &QPushButton::clicked, this, [this]() {
        // Prefer saved defaults if available; otherwise code defaults
        QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
        s.beginGroup("toobj/defaults");
        const bool hasAny = s.allKeys().size() > 0;
        s.endGroup();
        if (hasAny) applySavedDefaults(); else applyCodeDefaults();
    });
    connect(btnSave, &QPushButton::clicked, this, [this]() { saveDefaults(); });
}

QString ConvertToObjDialog::tifxyzPath() const { return edtTifxyz_->text(); }
QString ConvertToObjDialog::objPath() const { return edtObj_->text(); }
bool ConvertToObjDialog::normalizeUV() const { return chkNormalize_->isChecked(); }
bool ConvertToObjDialog::alignGrid() const { return chkAlign_->isChecked(); }
int ConvertToObjDialog::decimateIterations() const { return spDecimate_->value(); }
bool ConvertToObjDialog::cleanSurface() const { return chkClean_->isChecked(); }
double ConvertToObjDialog::cleanK() const { return spCleanK_->value(); }
int ConvertToObjDialog::ompThreads() const {
    const QString t = edtThreads_->text().trimmed();
    if (t.isEmpty()) return -1;
    bool ok=false; int v = t.toInt(&ok); return (ok && v>0) ? v : -1;
}

void ConvertToObjDialog::applyCodeDefaults() {
    chkNormalize_->setChecked(false);
    chkAlign_->setChecked(false);
    spDecimate_->setValue(0);
    chkClean_->setChecked(false);
    spCleanK_->setValue(5.0);
    spCleanK_->setEnabled(chkClean_->isChecked());
    edtThreads_->setText("");
}

void ConvertToObjDialog::applySavedDefaults() {
    QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
    s.beginGroup("toobj/defaults");
    chkNormalize_->setChecked(s.value("normalize_uv", chkNormalize_->isChecked()).toBool());
    chkAlign_->setChecked(s.value("align_grid", chkAlign_->isChecked()).toBool());
    spDecimate_->setValue(s.value("decimate_iters", spDecimate_->value()).toInt());
    chkClean_->setChecked(s.value("clean_surface", chkClean_->isChecked()).toBool());
    spCleanK_->setValue(s.value("clean_k", spCleanK_->value()).toDouble());
    spCleanK_->setEnabled(chkClean_->isChecked());
    const int th = s.value("omp_threads", -1).toInt();
    edtThreads_->setText(th > 0 ? QString::number(th) : "");
    s.endGroup();
}

void ConvertToObjDialog::applySessionDefaults() {
    if (!s_haveSession) return;
    chkNormalize_->setChecked(s_normUV);
    chkAlign_->setChecked(s_alignGrid);
    spDecimate_->setValue(s_decimate);
    chkClean_->setChecked(s_clean);
    spCleanK_->setValue(s_cleanK);
    spCleanK_->setEnabled(chkClean_->isChecked());
    edtThreads_->setText(s_ompThreads > 0 ? QString::number(s_ompThreads) : "");
}

void ConvertToObjDialog::saveDefaults() const {
    QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
    s.beginGroup("toobj/defaults");
    s.setValue("normalize_uv", chkNormalize_->isChecked());
    s.setValue("align_grid", chkAlign_->isChecked());
    s.setValue("decimate_iters", spDecimate_->value());
    s.setValue("clean_surface", chkClean_->isChecked());
    s.setValue("clean_k", spCleanK_->value());
    const int th = ompThreads();
    s.setValue("omp_threads", th);
    s.endGroup();
}

void ConvertToObjDialog::updateSessionFromUI() {
    s_haveSession = true;
    s_normUV = chkNormalize_->isChecked();
    s_alignGrid = chkAlign_->isChecked();
    s_decimate = spDecimate_->value();
    s_clean = chkClean_->isChecked();
    s_cleanK = spCleanK_->value();
    s_ompThreads = ompThreads();
}

bool AlphaCompRefineDialog::s_haveSession = false;
double AlphaCompRefineDialog::s_start = -6.0;
double AlphaCompRefineDialog::s_stop = 30.0;
double AlphaCompRefineDialog::s_step = 2.0;
double AlphaCompRefineDialog::s_low = 26.0;
double AlphaCompRefineDialog::s_high = 255.0;
double AlphaCompRefineDialog::s_borderOff = 1.0;
int AlphaCompRefineDialog::s_radius = 3;
double AlphaCompRefineDialog::s_readerScale = 0.5;
QString AlphaCompRefineDialog::s_scaleGroup = QStringLiteral("1");
bool AlphaCompRefineDialog::s_refine = true;
bool AlphaCompRefineDialog::s_vertexColor = false;
bool AlphaCompRefineDialog::s_overwrite = true;
int AlphaCompRefineDialog::s_ompThreads = -1;

AlphaCompRefineDialog::AlphaCompRefineDialog(QWidget* parent,
                                             const QString& volumePath,
                                             const QString& srcSurfacePath,
                                             const QString& dstSurfacePath)
    : QDialog(parent)
{
    setWindowTitle(tr("Alpha-Composite Refinement"));
    auto main = new QVBoxLayout(this);

    auto pathsBox = new QGroupBox(tr("Paths"), this);
    auto paths = new QFormLayout(pathsBox);
    pathsBox->setLayout(paths);

    QWidget* volPick = pathPicker(this, edtVolume_, tr("Select OME-Zarr volume"), true);
    QWidget* srcPick = pathPicker(this, edtSrc_, tr("Select source surface"), true);
    QWidget* dstPick = pathPicker(this, edtDst_, tr("Select output surface"), true);

    edtVolume_->setText(volumePath);
    edtSrc_->setText(srcSurfacePath);
    edtDst_->setText(dstSurfacePath);

    paths->addRow(tr("Volume:"), volPick);
    paths->addRow(tr("Source:"), srcPick);
    paths->addRow(tr("Output:"), dstPick);

    main->addWidget(pathsBox);

    auto paramsBox = new QGroupBox(tr("Refinement Parameters"), this);
    auto params = new QFormLayout(paramsBox);
    paramsBox->setLayout(params);

    chkRefine_ = new QCheckBox(tr("Enable geometry refinement"), this);
    chkRefine_->setChecked(true);

    spStart_ = new QDoubleSpinBox(this); spStart_->setRange(-1000.0, 1000.0); spStart_->setDecimals(3); spStart_->setValue(-6.0);
    spStop_  = new QDoubleSpinBox(this); spStop_->setRange(-1000.0, 1000.0);  spStop_->setDecimals(3);  spStop_->setValue(30.0);
    spStep_  = new QDoubleSpinBox(this); spStep_->setRange(0.001, 1000.0);    spStep_->setDecimals(3);  spStep_->setValue(2.0);
    spLow_   = new QDoubleSpinBox(this);  spLow_->setRange(0.0, 255.0);       spLow_->setDecimals(0);   spLow_->setSingleStep(1.0);   spLow_->setValue(26.0);
    spHigh_  = new QDoubleSpinBox(this);  spHigh_->setRange(0.0, 255.0);      spHigh_->setDecimals(0);  spHigh_->setSingleStep(1.0);  spHigh_->setValue(255.0);
    spBorder_= new QDoubleSpinBox(this); spBorder_->setRange(-100.0, 100.0);  spBorder_->setDecimals(3);spBorder_->setValue(1.0);
    spRadius_= new QSpinBox(this);        spRadius_->setRange(1, 100);        spRadius_->setValue(3);
    spReaderScale_ = new QDoubleSpinBox(this); spReaderScale_->setRange(0.0001, 1000.0); spReaderScale_->setDecimals(4); spReaderScale_->setValue(0.5);
    edtScaleGroup_ = new QLineEdit(this); edtScaleGroup_->setText(QStringLiteral("1"));

    chkVertexColor_ = new QCheckBox(tr("Generate vertex color (OBJ only)"), this);
    chkOverwrite_ = new QCheckBox(tr("Overwrite if output exists"), this);
    chkOverwrite_->setChecked(true);

    edtThreads_ = new QLineEdit(this);
    edtThreads_->setPlaceholderText(tr("optional"));
    edtThreads_->setValidator(new QRegularExpressionValidator(QRegularExpression("^\\s*\\d*\\s*$"), this));

    params->addRow(chkRefine_);
    params->addRow(tr("Start:"), spStart_);
    params->addRow(tr("Stop:"), spStop_);
    params->addRow(tr("Step:"), spStep_);
    params->addRow(tr("Opacity low (0-255):"), spLow_);
    params->addRow(tr("Opacity high (0-255):"), spHigh_);
    params->addRow(tr("Border offset:"), spBorder_);
    params->addRow(tr("Gaussian radius:"), spRadius_);
    params->addRow(tr("Reader scale:"), spReaderScale_);
    params->addRow(tr("Scale group:"), edtScaleGroup_);
    params->addRow(chkVertexColor_);
    params->addRow(chkOverwrite_);
    params->addRow(tr("OMP threads:"), edtThreads_);

    main->addWidget(paramsBox);

    auto buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    connect(buttons, &QDialogButtonBox::accepted, this, &AlphaCompRefineDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &AlphaCompRefineDialog::reject);
    main->addWidget(buttons);

    applySavedDefaults();
    applySessionDefaults();
}

QString AlphaCompRefineDialog::volumePath() const { return edtVolume_->text().trimmed(); }
QString AlphaCompRefineDialog::srcPath() const { return edtSrc_->text().trimmed(); }
QString AlphaCompRefineDialog::dstPath() const { return edtDst_->text().trimmed(); }

QJsonObject AlphaCompRefineDialog::paramsJson() const
{
    QJsonObject obj;
    obj["refine"] = chkRefine_->isChecked();
    obj["start"] = spStart_->value();
    obj["stop"] = spStop_->value();
    obj["step"] = spStep_->value();
    obj["low"] = static_cast<int>(std::lround(spLow_->value()));
    obj["high"] = static_cast<int>(std::lround(spHigh_->value()));
    obj["border_off"] = spBorder_->value();
    obj["r"] = spRadius_->value();
    obj["gen_vertexcolor"] = chkVertexColor_->isChecked();
    obj["overwrite"] = chkOverwrite_->isChecked();
    obj["reader_scale"] = spReaderScale_->value();

    const QString sg = edtScaleGroup_->text().trimmed();
    obj["scale_group"] = sg.isEmpty() ? QStringLiteral("1") : sg;

    return obj;
}

int AlphaCompRefineDialog::ompThreads() const
{
    const QString text = edtThreads_->text().trimmed();
    bool ok = false;
    int v = text.toInt(&ok);
    return ok ? v : -1;
}

void AlphaCompRefineDialog::accept()
{
    updateSessionFromUI();
    saveDefaults();
    QDialog::accept();
}

void AlphaCompRefineDialog::applySavedDefaults()
{
    QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
    s.beginGroup("objrefine/defaults");
    chkRefine_->setChecked(s.value("refine", chkRefine_->isChecked()).toBool());
    spStart_->setValue(s.value("start", spStart_->value()).toDouble());
    spStop_->setValue(s.value("stop", spStop_->value()).toDouble());
    spStep_->setValue(s.value("step", spStep_->value()).toDouble());
    spLow_->setValue(s.value("low", spLow_->value()).toDouble());
    spHigh_->setValue(s.value("high", spHigh_->value()).toDouble());
    spBorder_->setValue(s.value("border_off", spBorder_->value()).toDouble());
    spRadius_->setValue(s.value("radius", spRadius_->value()).toInt());
    spReaderScale_->setValue(s.value("reader_scale", spReaderScale_->value()).toDouble());
    edtScaleGroup_->setText(s.value("scale_group", edtScaleGroup_->text()).toString());
    chkVertexColor_->setChecked(s.value("vertex_color", chkVertexColor_->isChecked()).toBool());
    chkOverwrite_->setChecked(s.value("overwrite", chkOverwrite_->isChecked()).toBool());
    const int th = s.value("omp_threads", -1).toInt();
    edtThreads_->setText(th > 0 ? QString::number(th) : "");
    s.endGroup();
}

void AlphaCompRefineDialog::applySessionDefaults()
{
    if (!s_haveSession) return;
    chkRefine_->setChecked(s_refine);
    spStart_->setValue(s_start);
    spStop_->setValue(s_stop);
    spStep_->setValue(s_step);
    spLow_->setValue(s_low);
    spHigh_->setValue(s_high);
    spBorder_->setValue(s_borderOff);
    spRadius_->setValue(s_radius);
    spReaderScale_->setValue(s_readerScale);
    edtScaleGroup_->setText(s_scaleGroup);
    chkVertexColor_->setChecked(s_vertexColor);
    chkOverwrite_->setChecked(s_overwrite);
    edtThreads_->setText(s_ompThreads > 0 ? QString::number(s_ompThreads) : "");
}

void AlphaCompRefineDialog::saveDefaults() const
{
    QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
    s.beginGroup("objrefine/defaults");
    s.setValue("refine", chkRefine_->isChecked());
    s.setValue("start", spStart_->value());
    s.setValue("stop", spStop_->value());
    s.setValue("step", spStep_->value());
    s.setValue("low", static_cast<int>(std::lround(spLow_->value())));
    s.setValue("high", static_cast<int>(std::lround(spHigh_->value())));
    s.setValue("border_off", spBorder_->value());
    s.setValue("radius", spRadius_->value());
    s.setValue("reader_scale", spReaderScale_->value());
    s.setValue("scale_group", edtScaleGroup_->text().trimmed().isEmpty() ? QStringLiteral("1") : edtScaleGroup_->text().trimmed());
    s.setValue("vertex_color", chkVertexColor_->isChecked());
    s.setValue("overwrite", chkOverwrite_->isChecked());
    s.setValue("omp_threads", ompThreads());
    s.endGroup();
}

void AlphaCompRefineDialog::updateSessionFromUI()
{
    s_haveSession = true;
    s_refine = chkRefine_->isChecked();
    s_start = spStart_->value();
    s_stop = spStop_->value();
    s_step = spStep_->value();
    s_low = spLow_->value();
    s_high = spHigh_->value();
    s_borderOff = spBorder_->value();
    s_radius = spRadius_->value();
    s_readerScale = spReaderScale_->value();
    const QString sg = edtScaleGroup_->text().trimmed();
    s_scaleGroup = sg.isEmpty() ? QStringLiteral("1") : sg;
    s_vertexColor = chkVertexColor_->isChecked();
    s_overwrite = chkOverwrite_->isChecked();
    s_ompThreads = ompThreads();
}
