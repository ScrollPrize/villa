#include "SegmentRenderThread.hpp"
#include <QDir>
#include <QFileInfo>

SegmentRenderThread::SegmentRenderThread(QObject *parent)
    : QThread(parent)
    , m_scale(1.0f)
    , m_resolution(0)
    , m_layers(31)
    , m_process(nullptr)
{
}

SegmentRenderThread::~SegmentRenderThread()
{
    if (m_process) {
        if (m_process->state() != QProcess::NotRunning) {
            m_process->terminate();
            m_process->waitForFinished(3000);
        }
        delete m_process;
    }
}

void SegmentRenderThread::setParameters(
    const QString& volumePath,
    const QString& segmentPath, 
    const QString& outputPattern,
    float scale,
    int resolution,
    int layers)
{
    m_volumePath = volumePath;
    m_segmentPath = segmentPath;
    m_outputPattern = outputPattern;
    m_scale = scale;
    m_resolution = resolution;
    m_layers = layers;
}

void SegmentRenderThread::run()
{
    QFileInfo outputInfo(m_outputPattern);
    QDir outputDir = outputInfo.dir();
    
    if (!outputDir.exists()) {
        if (!outputDir.mkpath(".")) {
            emit renderingFailed("Failed to create output directory: " + outputDir.path());
            return;
        }
    }

    QStringList args;
    args << m_volumePath
         << m_outputPattern
         << m_segmentPath
         << QString::number(m_scale)
         << QString::number(m_resolution)
         << QString::number(m_layers);

    emit renderingStarted("Starting rendering process for: " + QFileInfo(m_segmentPath).fileName());
    
    m_process = new QProcess();
    m_process->setProcessChannelMode(QProcess::MergedChannels);
    
    connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            [this](int exitCode, QProcess::ExitStatus exitStatus) {
                if (exitCode == 0 && exitStatus == QProcess::NormalExit) {
                    QFileInfo outputInfo(m_outputPattern);
                    QDir outputDir = outputInfo.dir();
                    emit renderingFinished("Segment rendering completed successfully", outputDir.path());
                } else {
                    emit renderingFailed("Rendering process failed with exit code: " + QString::number(exitCode));
                }
                
                m_process->deleteLater();
                m_process = nullptr;
            });
    
    connect(m_process, &QProcess::errorOccurred, [this](QProcess::ProcessError error) {
        QString errorMessage = "Error running rendering process: ";
        switch (error) {
            case QProcess::FailedToStart: errorMessage += "failed to start"; break;
            case QProcess::Crashed: errorMessage += "crashed"; break;
            case QProcess::Timedout: errorMessage += "timed out"; break;
            case QProcess::WriteError: errorMessage += "write error"; break;
            case QProcess::ReadError: errorMessage += "read error"; break;
            default: errorMessage += "unknown error"; break;
        }
        
        emit renderingFailed(errorMessage);
        
        m_process->deleteLater();
        m_process = nullptr;
    });
    
    m_process->start("vc_render_tifxyz", args);
    
    m_process->waitForFinished(-1);
}
