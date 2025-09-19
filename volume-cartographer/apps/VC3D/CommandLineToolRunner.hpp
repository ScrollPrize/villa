#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QProcess>
#include <QDialog>
#include <QFile>

#include "CWindow.hpp"
#include "ProgressUtil.hpp"
#include "ConsoleOutputWidget.hpp"



// Forward declaration


/**
 * @brief Class to manage execution of command-line tools
 */
class CommandLineToolRunner : public QObject
{
    Q_OBJECT

public:

    explicit CommandLineToolRunner(QStatusBar* statusBar, CWindow* mainWindow, QObject* parent = nullptr);
    
    ~CommandLineToolRunner();

    enum class Tool {
        RenderTifXYZ,
        GrowSegFromSegment,
        GrowSegFromSeeds,
        SegAddOverlap,
        tifxyz2obj,
        obj2tifxyz
    };

    void setVolumePath(const QString& path);
    void setSegmentPath(const QString& path);
    void setOutputPattern(const QString& pattern);

    // tool specific params 
    void setRenderParams(float scale, int resolution, int layers);
    void setRenderAdvanced(int cropX, int cropY, int cropWidth, int cropHeight,
                           const QString& affinePath, bool invertAffine,
                           float scaleSegmentation, double rotateDegrees, int flipAxis);
    void setGrowParams(QString volumePath, QString tgtDir, QString jsonParams, int seed_x = 0, int seed_y = 0, int seed_z = 0, bool useExpandMode = false, bool useRandomSeed = false);
    void setTraceParams(QString volumePath, QString srcDir, QString tgtDir, QString jsonParams, QString srcSegment);
    void setAddOverlapParams(QString tgtDir, QString tifxyzPath);
    void setToObjParams(QString tifxyzPath, QString objPath);
    void setToObjOptions(bool normalizeUV, bool alignGrid, int decimateIterations, bool cleanSurface, float cleanK);
    void setObj2TifxyzParams(const QString& objPath, const QString& outputDir,
                             float stretchFactor = 1000.0f,
                             float meshUnits = 1.0f,
                             int stepSize = 20);
    bool execute(Tool tool);
    void cancel();
    bool isRunning() const;
    
    void showConsoleOutput();
    void hideConsoleOutput();
    void setAutoShowConsoleOutput(bool autoShow);
    void setParallelProcesses(int count);
    void setIterationCount(int count);
    void setIncludeTifs(bool include);
    void setOmpThreads(int threads);

signals:
    void toolStarted(Tool tool, const QString& message);
    void toolFinished(Tool tool, bool success, const QString& message, const QString& outputPath, bool copyToClipboard = false);
    void consoleOutputReceived(const QString& output);

private slots:
    void onProcessStarted();
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onProcessError(QProcess::ProcessError error);
    void onProcessReadyRead();

private:
    QStringList buildArguments(Tool tool);
    QString toolName(Tool tool) const;
    QString getOutputPath() const;

    CWindow* _mainWindow;
    ProgressUtil* _progressUtil;
    
    QProcess* _process;
    ConsoleOutputWidget* _consoleOutput;
    QDialog* _consoleDialog;
    bool _autoShowConsole;
    
    QString _volumePath;
    QString _segmentPath;
    QString _outputPattern;
    QString _tgtDir;
    QString _srcDir;
    QString _srcSegment;
    QString _tifxyzPath;
    QString _objPath;
    QString _jsonParams;
    
    float _scale;
    int _resolution;
    int _layers;
    int _seed_x;
    int _seed_y;
    int _seed_z;
    bool _useExpandMode;
    bool _useRandomSeed;
    int _parallelProcesses;  // processes for xargs
    int _iterationCount;     // iterations for xargs

    // Advanced render options
    int _cropX{0};
    int _cropY{0};
    int _cropWidth{0};
    int _cropHeight{0};
    QString _affinePath;
    bool _invertAffine{false};
    float _scaleSeg{1.0f};
    double _rotateDeg{0.0};
    int _flipAxis{-1};
    bool _includeTifs{false};

    // vc_tifxyz2obj options
    bool _optNormalizeUV{false};
    bool _optAlignGrid{false};
    int  _optDecimateIter{0};
    bool _optCleanSurface{false};
    float _optCleanK{5.0f};

    Tool _currentTool;

    QFile* _logFile;
    QTextStream* _logStream;

    int _ompThreads{-1};

    QString _objOutputDir;
    float _objStretchFactor = 1000.0f;
    float _objMeshUnits = 1.0f;
    int _objStepSize = 20;
};
