#pragma once

#include <QDialog>
#include <QString>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QJsonObject>
#include <QJsonDocument>
#include <QJsonArray>
#include <QSettings>

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

class AlphaCompRefineDialog : public QDialog {
    Q_OBJECT
public:
    AlphaCompRefineDialog(QWidget* parent,
                          const QString& volumePath,
                          const QString& srcSurfacePath,
                          const QString& dstSurfacePath);

    QString volumePath() const;
    QString srcPath() const;
    QString dstPath() const;
    QJsonObject paramsJson() const;
    int ompThreads() const; // -1 if unset

protected:
    void accept() override;

private:
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
