#pragma once

#include <QDialog>
#include <QString>
#include <QStringList>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QJsonObject>
#include <QJsonDocument>
#include <QJsonArray>
#include <QVector>
#include <QSettings>

#include <optional>

#include "AlphaCompRefineRequest.hpp"

class JsonProfileEditor;
class VolumeSelector;

class RenderParamsDialog : public QDialog {
    Q_OBJECT
public:
    RenderParamsDialog(QWidget* parent,
                       const QString& volumePath,
                       const QString& segmentPath,
                       const QString& outputPattern,
                       double scale,
                       int groupIdx,
                       int numSlices);

    QString volumePath() const;
    QString segmentPath() const;
    QString outputPattern() const;
    double scale() const;
    int groupIdx() const;
    int numSlices() const;
    int ompThreads() const; // -1 if unset

    // Advanced
    int cropX() const;
    int cropY() const;
    int cropWidth() const;
    int cropHeight() const;
    QString affinePath() const;
    bool invertAffine() const;
    double scaleSegmentation() const;
    double rotateDegrees() const;
    int flipAxis() const; // -1 none, 0 vertical, 1 horizontal, 2 both
    bool includeTifs() const; // when output is .zarr
    bool flatten() const; // ABF++ flattening
    int flattenIterations() const;
    int flattenDownsample() const;

private:
    // Session defaults (optional-only; exclude paths and output pattern)
    static bool s_haveSession;
    static bool s_includeTifs;
    static int  s_cropX, s_cropY, s_cropW, s_cropH;
    static bool s_invertAffine;
    static double s_scaleSeg;
    static double s_rotateDeg;
    static int  s_flipAxis;
    static int  s_ompThreads;
    static bool s_flatten;
    static int  s_flattenIters;
    static int  s_flattenDownsample;

    void applyCodeDefaults();
    void applySavedDefaults();
    void applySessionDefaults();
    void saveDefaults() const; // persist optional-only to VC.ini
    void updateSessionFromUI();

    QLineEdit* edtVolume_{nullptr};
    QLineEdit* edtSegment_{nullptr};
    QLineEdit* edtOutput_{nullptr};
    QDoubleSpinBox* spScale_{nullptr};
    QSpinBox* spGroup_{nullptr};
    QSpinBox* spSlices_{nullptr};
    QLineEdit* edtThreads_{nullptr};

    QSpinBox* spCropX_{nullptr};
    QSpinBox* spCropY_{nullptr};
    QSpinBox* spCropW_{nullptr};
    QSpinBox* spCropH_{nullptr};
    QLineEdit* edtAffine_{nullptr};
    QCheckBox* chkInvert_{nullptr};
    QDoubleSpinBox* spScaleSeg_{nullptr};
    QDoubleSpinBox* spRotate_{nullptr};
    QComboBox* cmbFlip_{nullptr};
    QCheckBox* chkIncludeTifs_{nullptr};
    QCheckBox* chkFlatten_{nullptr};
    QSpinBox* spFlattenIters_{nullptr};
    QSpinBox* spFlattenDownsample_{nullptr};
};

class TraceParamsDialog : public QDialog {
    Q_OBJECT
public:
    TraceParamsDialog(QWidget* parent,
                      const QString& volumePath,
                      const QString& srcDir,
                      const QString& tgtDir,
                      const QString& jsonParams,
                      const QString& srcSegment);

    QString volumePath() const;
    QString srcDir() const;
    QString tgtDir() const;
    QString jsonParams() const;
    QString srcSegment() const;
    
    // Build a params JSON object from UI controls (merged or standalone)
    QJsonObject makeParamsJson() const;
    int ompThreads() const; // -1 if unset

private:
    // Session defaults (in-memory only; exclude paths)
    static bool   s_haveSession;
    static bool   s_flipX;
    static int    s_globalStepsWin;
    static double s_srcStep;
    static double s_step;
    static int    s_maxWidth;
    static double s_localCostInlTh;
    static double s_sameSurfaceTh;
    static double s_straightW;
    static double s_straightW3D;
    static double s_slidingWScale;
    static double s_zLocLossW;
    static double s_distLoss2DW;
    static double s_distLoss3DW;
    static double s_straightMinCount;
    static int    s_inlierBaseTh;
    static int    s_consensusDefaultTh;
    static bool   s_useZRange;
    static double s_zMin;
    static double s_zMax;
    static int    s_ompThreads;

    void applySessionDefaults();
    void updateSessionFromUI();

    QLineEdit* edtVolume_{nullptr};
    QLineEdit* edtSrcDir_{nullptr};
    QLineEdit* edtTgtDir_{nullptr};
    QLineEdit* edtJson_{nullptr};
    QLineEdit* edtSrcSegment_{nullptr};
    QLineEdit* edtThreads_{nullptr};

    // Advanced tracing parameters (parsed from JSON; defaults reflect GrowSurface.cpp)
    QCheckBox* chkFlipX_{nullptr};
    QSpinBox* spGlobalStepsWin_{nullptr};
    QDoubleSpinBox* spSrcStep_{nullptr};
    QDoubleSpinBox* spStep_{nullptr};
    QSpinBox* spMaxWidth_{nullptr};
    QDoubleSpinBox* spLocalCostInlTh_{nullptr};
    QDoubleSpinBox* spSameSurfaceTh_{nullptr};
    QDoubleSpinBox* spStraightW_{nullptr};
    QDoubleSpinBox* spStraightW3D_{nullptr};
    QDoubleSpinBox* spSlidingWScale_{nullptr};
    QDoubleSpinBox* spZLocLossW_{nullptr};
    QDoubleSpinBox* spDistLoss2DW_{nullptr};
    QDoubleSpinBox* spDistLoss3DW_{nullptr};
    QDoubleSpinBox* spStraightMinCount_{nullptr};
    QSpinBox* spInlierBaseTh_{nullptr};
    QSpinBox* spConsensusDefaultTh_{nullptr};
    QCheckBox* chkZRange_{nullptr};
    QDoubleSpinBox* spZMin_{nullptr};
    QDoubleSpinBox* spZMax_{nullptr};

    // Defaults helpers
    void applyCodeDefaults();
    void applySavedDefaults();
    void saveDefaults() const;
};

class ConvertToObjDialog : public QDialog {
    Q_OBJECT
public:
    ConvertToObjDialog(QWidget* parent,
                       const QString& tifxyzPath,
                       const QString& objOutPath);

    QString tifxyzPath() const;
    QString objPath() const;
    bool normalizeUV() const;
    bool alignGrid() const;
    int decimateIterations() const;
    bool cleanSurface() const;
    double cleanK() const;
    int ompThreads() const; // -1 if unset

private:
    // Session defaults (in-memory only)
    static bool   s_haveSession;
    static bool   s_normUV;
    static bool   s_alignGrid;
    static int    s_decimate;
    static bool   s_clean;
    static double s_cleanK;
    static int    s_ompThreads; // -1 if unset

    void applyCodeDefaults();
    void applySavedDefaults();
    void applySessionDefaults();
    void saveDefaults() const; // persist optional-only to VC.ini
    void updateSessionFromUI();

    QLineEdit* edtTifxyz_{nullptr};
    QLineEdit* edtObj_{nullptr};
    QLineEdit* edtThreads_{nullptr};
    QCheckBox* chkNormalize_{nullptr};
    QCheckBox* chkAlign_{nullptr};
    QSpinBox* spDecimate_{nullptr};
    QCheckBox* chkClean_{nullptr};
    QDoubleSpinBox* spCleanK_{nullptr};
};

class QTableWidget;
class QPushButton;
class QLabel;

// MergeTifxyzDialog
//
// Edits a 2D grid of tifxyz directory names and the RANSAC tunables for
// vc_merge_tifxyz, then writes a merge.json into the volpkg root and
// returns its path. The grid layout encodes the surface adjacency that
// the merge tool RANSAC-aligns; empty cells are allowed; selection in
// the segment list is used as the seed but the user can edit / append.
class MergeTifxyzDialog : public QDialog {
    Q_OBJECT
public:
    MergeTifxyzDialog(QWidget* parent,
                      const QStringList& seedSegmentIds,
                      const QStringList& availableSegments,
                      const QString& volpkgDir,
                      const QString& pathsDir);

    QString mergeJsonPath() const;     // populated by accept() on success
    QString refSurface() const;        // empty -> auto (largest valid-cell count)
    int ransacIters() const;
    double ransacMinThresh() const;
    double ransacMaxThresh() const;
    double ransacMadK() const;
    int ransacSeed() const;
    int anchorCap() const;
    int stripCols() const;
    int ompThreads() const; // -1 if unset

protected:
    void accept() override;

private:
    QStringList collectGridNames() const;          // row-major, empties as ""
    QString buildMergeJsonText() const;            // pretty JSON for preview + write
    void rebuildPreview();
    void updateRefCombo();
    void resizeGrid(int newRows, int newCols);
    void seedGrid(const QStringList& seedSegmentIds);
    void promptAddSegments();

    void applyCodeDefaults();
    void applySessionDefaults();
    void updateSessionFromUI();

    QStringList _availableSegments;
    QString _volpkgDir;
    QString _pathsDir;
    QString _mergeJsonPath;

    QTableWidget* tblGrid_{nullptr};
    QPushButton* btnAddRow_{nullptr};
    QPushButton* btnAddCol_{nullptr};
    QPushButton* btnRemoveRow_{nullptr};
    QPushButton* btnRemoveCol_{nullptr};
    QPushButton* btnAddSegments_{nullptr};
    QComboBox* cmbRef_{nullptr};
    QLabel* lblOutName_{nullptr};
    QSpinBox* spIters_{nullptr};
    QDoubleSpinBox* spMin_{nullptr};
    QDoubleSpinBox* spMax_{nullptr};
    QDoubleSpinBox* spMadK_{nullptr};
    QSpinBox* spSeed_{nullptr};
    QSpinBox* spAnchorCap_{nullptr};
    QSpinBox* spStripCols_{nullptr};
    QLineEdit* edtThreads_{nullptr};
    QLabel* lblPreview_{nullptr};

    // Session defaults (in-memory only; persist across one VC3D run).
    static bool   s_haveSession;
    static int    s_iters;
    static double s_min;
    static double s_max;
    static double s_madK;
    static int    s_seed;
    static int    s_anchorCap;
    static int    s_stripCols;
    static int    s_lastRows;
    static int    s_lastCols;
    static int    s_ompThreads;
};

class QComboBox;
class QSpinBox;
class QLineEdit;
class QLabel;
class QCheckBox;
class VolumePkg;

// MergePatchDialog
//
// Two-input cousin of MergeTifxyzDialog: picks a parent + child tifxyz from
// the volpkg, exposes vc_merge_patch's tunables (border-cells, blend-cells,
// idw-k, RANSAC set, anchor-cap, OMP threads), and surfaces the role
// auto-detect / explicit-override the binary supports. The parent dir is
// overwritten in place by the binary; the dialog shows a banner up front so
// the user knows. Recovery is via the rotating backup ring
// (<parent_dir>/backups/<parent_name>/{0..7}/) that the binary populates via
// QuadSurface::saveOverwrite.
class MergePatchDialog : public QDialog {
    Q_OBJECT
public:
    MergePatchDialog(QWidget* parent,
                     const QStringList& seedSegmentIds,
                     const QStringList& availableSegments,
                     std::shared_ptr<VolumePkg> volpkg,
                     const QString& volpkgDir,
                     const QString& pathsDir);

    // Resolved paths (filesystem-ready). parentPath() is the dir that will
    // be overwritten; childPath() is the dir whose data gets patched in.
    QString parentPath() const;
    QString childPath() const;
    // Roles set explicitly by the user (via the Swap button or a manual
    // override) instead of auto-detected. Drives whether the runner passes
    // --parent/--child or positional args.
    bool    explicitRoles() const;

    int     borderCells() const;
    int     blendCells() const;
    int     idwK() const;
    int     ransacIters() const;
    double  ransacMinThresh() const;
    double  ransacMaxThresh() const;
    double  ransacMadK() const;
    int     ransacSeed() const;
    int     anchorCap() const;
    int     ompThreads() const; // -1 if unset

protected:
    void accept() override;

private:
    void recomputeRoleHint();
    void refreshOverwriteBanner();
    void applyCodeDefaults();
    void applySessionDefaults();
    void updateSessionFromUI();

    // Cheap valid-cell count for an already-loaded surface, via
    // QuadSurface::countValidPoints (an in-memory scan, no I/O). Returns
    // -1 if the surface isn't loaded in the volpkg cache.
    int validCellCountFor(const QString& segId) const;

    QStringList _availableSegments;
    std::shared_ptr<VolumePkg> _volpkg;
    QString _volpkgDir;
    QString _pathsDir;

    QComboBox* cmbA_{nullptr};
    QComboBox* cmbB_{nullptr};
    QPushButton* btnSwap_{nullptr};
    QLabel* lblRoleHint_{nullptr};
    QLabel* lblBanner_{nullptr};

    QSpinBox* spBorderCells_{nullptr};
    QSpinBox* spBlendCells_{nullptr};
    QSpinBox* spIdwK_{nullptr};

    QSpinBox* spIters_{nullptr};
    QDoubleSpinBox* spMin_{nullptr};
    QDoubleSpinBox* spMax_{nullptr};
    QDoubleSpinBox* spMadK_{nullptr};
    QSpinBox* spSeed_{nullptr};
    QSpinBox* spAnchorCap_{nullptr};
    QLineEdit* edtThreads_{nullptr};

    // Resolved on accept().
    QString _parentPath;
    QString _childPath;
    bool    _explicitRoles{false};

    // Auto-detect bookkeeping: true means combo A is the parent. The Swap
    // button toggles this AND sets _userSwapped so accept() emits explicit
    // --parent/--child instead of trusting the binary's auto-detect.
    bool _aIsParent{true};
    bool _userSwapped{false};

    // Session defaults (in-memory only; persist across one VC3D run).
    static bool   ms_haveSession;
    static int    ms_borderCells;
    static int    ms_blendCells;
    static int    ms_idwK;
    static int    ms_iters;
    static double ms_min;
    static double ms_max;
    static double ms_madK;
    static int    ms_seed;
    static int    ms_anchorCap;
    static int    ms_ompThreads;
};

class AlphaCompRefineDialog : public QDialog {
    Q_OBJECT
public:
    AlphaCompRefineDialog(QWidget* parent,
                          const QString& volumePath,
                          const QString& srcSurfacePath,
                          const QString& dstSurfacePath);

    AlphaCompRefineRequest request() const;

protected:
    void accept() override;

private:
    int ompThreads() const; // -1 if unset
    void applySavedDefaults();
    void applySessionDefaults();
    void saveDefaults() const;
    void updateSessionFromUI();

    static bool s_haveSession;
    static double s_start;
    static double s_stop;
    static double s_step;
    static double s_low;
    static double s_high;
    static double s_borderOff;
    static int    s_radius;
    static double s_readerScale;
    static QString s_scaleGroup;
    static bool   s_refine;
    static bool   s_vertexColor;
    static bool   s_overwrite;
    static int    s_ompThreads;

    QLineEdit* edtVolume_{nullptr};
    QLineEdit* edtSrc_{nullptr};
    QLineEdit* edtDst_{nullptr};
    QLineEdit* edtScaleGroup_{nullptr};
    QLineEdit* edtThreads_{nullptr};
    QDoubleSpinBox* spStart_{nullptr};
    QDoubleSpinBox* spStop_{nullptr};
    QDoubleSpinBox* spStep_{nullptr};
    QDoubleSpinBox* spLow_{nullptr};
    QDoubleSpinBox* spHigh_{nullptr};
    QDoubleSpinBox* spBorder_{nullptr};
    QSpinBox* spRadius_{nullptr};
    QDoubleSpinBox* spReaderScale_{nullptr};
    QCheckBox* chkRefine_{nullptr};
    QCheckBox* chkVertexColor_{nullptr};
    QCheckBox* chkOverwrite_{nullptr};
};

struct NeighborCopyVolumeOption {
    QString id;
    QString name;
    QString path;
};

class NeighborCopyDialog : public QDialog {
    Q_OBJECT
public:
    NeighborCopyDialog(QWidget* parent,
                       const QString& surfacePath,
                       const QVector<NeighborCopyVolumeOption>& volumes,
                       const QString& defaultVolumeId,
                       const QString& defaultOutputPath);

    void accept() override;

    QString surfacePath() const;
    QString selectedVolumeId() const;
    QString selectedVolumePath() const;
    QString outputPath() const;
    int resumeLocalOptStep() const;
    int resumeLocalOptRadius() const;
    int resumeLocalMaxIters() const;
    int pass2OmpThreads() const;
    bool resumeLocalDenseQr() const;
    std::optional<QJsonObject> pass2TracerParamsJson(QString* error) const;

    // First pass parameters
    int neighborMaxDistance() const;
    int neighborMinClearance() const;
    bool neighborFill() const;
    int neighborInterpWindow() const;
    int generations() const;
    int neighborSpikeWindow() const;

private:
    void populateVolumeOptions(const QVector<NeighborCopyVolumeOption>& volumes,
                               const QString& defaultVolumeId);

    QLineEdit* edtSurface_{nullptr};
    VolumeSelector* volumeSelector_{nullptr};
    QLineEdit* edtOutput_{nullptr};
    QSpinBox* spResumeStep_{nullptr};
    QSpinBox* spResumeRadius_{nullptr};
    QSpinBox* spResumeMaxIters_{nullptr};
    QSpinBox* spPass2OmpThreads_{nullptr};
    QCheckBox* chkResumeDenseQr_{nullptr};
    JsonProfileEditor* pass2TracerParams_{nullptr};

    // First pass parameter widgets
    QSpinBox* spMaxDistance_{nullptr};
    QSpinBox* spMinClearance_{nullptr};
    QCheckBox* chkNeighborFill_{nullptr};
    QSpinBox* spInterpWindow_{nullptr};
    QSpinBox* spGenerations_{nullptr};
    QSpinBox* spSpikeWindow_{nullptr};
};

class ExportChunksDialog : public QDialog {
    Q_OBJECT
public:
    ExportChunksDialog(QWidget* parent, int surfaceWidth, double scale);

    int chunkWidth() const;
    int overlapPerSide() const;
    bool overwrite() const;

private:
    QSpinBox* spChunkWidth_{nullptr};
    QSpinBox* spOverlap_{nullptr};
    QCheckBox* chkOverwrite_{nullptr};
};

class ABFFlattenDialog : public QDialog {
    Q_OBJECT
public:
    ABFFlattenDialog(QWidget* parent);

    int iterations() const;
    int downsampleFactor() const;

private:
    // Session defaults (in-memory)
    static bool s_haveSession;
    static int s_iterations;
    static int s_downsample;

    void applySessionDefaults();
    void updateSessionFromUI();

    QSpinBox* spIterations_{nullptr};
    QSpinBox* spDownsample_{nullptr};
};

class SlimFlattenDialog : public QDialog {
    Q_OBJECT
public:
    SlimFlattenDialog(QWidget* parent, const QString& defaultOutputPath);

    int maxIterations() const;
    double tolerance() const;   // 0.0 = disabled (run all iterations)
    QString energyType() const; // "symmetric_dirichlet" or "conformal"
    QString outputPath() const;
    // Target percentage of source grid points to flatten (1..100). 100 means
    // no decimation. Below 100 the SLIM step runs on a decimated mesh and
    // the resulting UVs are lifted back to the full-resolution mesh via
    // vc_obj_uv_lift. Default 1.5% picks stride~8, which is roughly what
    // 2-iter stride-3 produced in earlier versions and matches the
    // configuration that converges on 2um Paris segments.
    double keepPercent() const;
    bool inpaintHoles() const;

private:
    static bool s_haveSession;
    static int s_iterations;
    static double s_tolerance;
    static QString s_energy;
    static double s_keepPercent;
    static bool s_inpaintHoles;

    QSpinBox* spIterations_{nullptr};
    QDoubleSpinBox* spTolerance_{nullptr};
    QComboBox* cbEnergy_{nullptr};
    QDoubleSpinBox* spKeepPercent_{nullptr};
    QCheckBox* cbInpaint_{nullptr};
    QLineEdit* edtOutput_{nullptr};
    QString defaultOutput_;
};

// Runs vc_straighten on a tifxyz surface (tifxyz -> tifxyz, no OBJ round-trip).
// Exposes the four pipeline stages and the few parameters that matter; the
// rest use the tool's CLI defaults.
class StraightenDialog : public QDialog {
    Q_OBJECT
public:
    StraightenDialog(QWidget* parent, const QString& defaultOutputPath);

    QString outputPath() const;
    // CLI args (without input/output) derived from the dialog state. Always
    // emits an explicit --overlap-pairs so the tool never falls back to its
    // implicit "no flags = full pipeline" default.
    QStringList toArgs() const;

private:
    static bool s_haveSession;
    static bool s_unbend;
    static double s_unbendSmoothCols;
    static int s_overlapPasses;
    static bool s_orthogonalize;
    static bool s_trim;
    static double s_trimMaxEdge;

    QCheckBox* cbUnbend_{nullptr};
    QDoubleSpinBox* spUnbendSmooth_{nullptr};
    QSpinBox* spOverlap_{nullptr};
    QCheckBox* cbOrtho_{nullptr};
    QCheckBox* cbTrim_{nullptr};
    QDoubleSpinBox* spTrimMaxEdge_{nullptr};
    QLineEdit* edtOutput_{nullptr};
    QString defaultOutput_;
};

class VisLasagnaObjDialog : public QDialog {
    Q_OBJECT
public:
    VisLasagnaObjDialog(QWidget* parent, const QString& outputDir);

    QString outputDir() const;
    QStringList slices() const;
    QStringList channels() const;
    QStringList losses() const;
    bool includeMesh() const;
    bool includeConnections() const;

private:
    // Session defaults (in-memory)
    static bool s_haveSession;
    static bool s_xy, s_xz, s_yz;
    static bool s_cos, s_gradMag;
    static bool s_lStep, s_lSmooth, s_lWinding, s_lNormal;
    static bool s_mesh, s_conn;

    void applySessionDefaults();
    void updateSessionFromUI();

    QLineEdit* edtOutput_{nullptr};
    QCheckBox* chkXY_{nullptr};
    QCheckBox* chkXZ_{nullptr};
    QCheckBox* chkYZ_{nullptr};
    QCheckBox* chkCos_{nullptr};
    QCheckBox* chkGradMag_{nullptr};
    QCheckBox* chkLossStep_{nullptr};
    QCheckBox* chkLossSmooth_{nullptr};
    QCheckBox* chkLossWinding_{nullptr};
    QCheckBox* chkLossNormal_{nullptr};
    QCheckBox* chkMesh_{nullptr};
    QCheckBox* chkConn_{nullptr};
};
