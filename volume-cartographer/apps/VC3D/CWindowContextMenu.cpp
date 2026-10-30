#include "CWindow.hpp"
#include <iostream>
#include "SurfacePanelController.hpp"
#include "VCSettings.hpp"
#include "SegmentationCommandHandler.hpp"

#include <QDialogButtonBox>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QSettings>
#include <QMessageBox>
#include <QStandardPaths>
#include <QtConcurrent>
#include <QProgressDialog>

#include "CommandLineToolRunner.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "ToolDialogs.hpp"
#include "LasagnaServiceManager.hpp"
#include "SurfaceCatalog.hpp"
#include "elements/VolumeSelector.hpp"
#include "elements/JsonProfilePresets.hpp"
#include "utils/Json.hpp"
#include <vc_object_store/Uuid.hpp>


// ====================== CWindow member functions ==============================

void CWindow::onVisLasagnaObj(const std::string& segmentId)
{
    auto& mgr = LasagnaServiceManager::instance();
    if (!mgr.isRunning()) {
        QMessageBox::warning(this, tr("Error"), tr("Lasagna service is not running"));
        return;
    }
    // The model a fit left with the surface in the store.
    if (!_surfaceCatalog || !_surfaceCatalog->attachment() || !mgr.surfaceModel(segmentId)) {
        QMessageBox::warning(this, tr("Error"), tr("This surface has no Lasagna model."));
        return;
    }

    // Default output directory: the surface's artifacts, beside its model
    QString defaultOutput = QString::fromStdString(
        (_surfaceCatalog->session()->artifactDirectory(*_surfaceCatalog->attachment(), segmentId, false) / "vis_obj")
            .string());

    VisLasagnaObjDialog dlg(this, defaultOutput);
    if (dlg.exec() != QDialog::Accepted) {
        showStatusBarMessage(tr("Vis as OBJ cancelled"), 3000);
        return;
    }

    // Build request JSON
    QJsonObject request;

    QJsonObject model;
    model[QStringLiteral("type")] = QStringLiteral("surface");
    model[QStringLiteral("uuid")] = QString::fromStdString(segmentId);
    request[QStringLiteral("model")] = model;

    // data_input is extracted from the checkpoint's _fit_config_ on the Python side.
    // No need to send it explicitly.

    // output_dir is where the client unpacks the results (not sent to server)
    request[QStringLiteral("output_dir")] = dlg.outputDir();

    QJsonArray slicesArr;
    for (const auto& s : dlg.slices()) slicesArr.append(s);
    request[QStringLiteral("slices")] = slicesArr;

    QJsonArray chansArr;
    for (const auto& c : dlg.channels()) chansArr.append(c);
    request[QStringLiteral("channels")] = chansArr;

    QJsonArray lossArr;
    for (const auto& l : dlg.losses()) lossArr.append(l);
    request[QStringLiteral("losses")] = lossArr;

    request[QStringLiteral("include_mesh")] = dlg.includeMesh();
    request[QStringLiteral("include_connections")] = dlg.includeConnections();

    // Connect result signals (one-shot)
    auto connFinish = std::make_shared<QMetaObject::Connection>();
    auto connError = std::make_shared<QMetaObject::Connection>();
    *connFinish = connect(&mgr, &LasagnaServiceManager::visExportFinished,
            this, [this, connFinish, connError](const QString& outputDir) {
                disconnect(*connFinish);
                disconnect(*connError);
                showStatusBarMessage(
                    tr("Vis OBJ export finished: %1").arg(outputDir), 5000);
            });
    *connError = connect(&mgr, &LasagnaServiceManager::visExportError,
            this, [this, connFinish, connError](const QString& message) {
                disconnect(*connFinish);
                disconnect(*connError);
                QMessageBox::warning(this, tr("Export Error"), message);
            });

    mgr.exportLasagnaVis(request);
    showStatusBarMessage(
        tr("Exporting vis OBJ for %1...").arg(QString::fromStdString(segmentId)), 3000);
}

bool CWindow::initializeCommandLineRunner()
{
    if (!_cmdRunner) {
        _cmdRunner = new CommandLineToolRunner(
            statusBar(), [this] { return getCurrentVolumePath(); }, this);

        if (_segmentationCommandHandler) {
            _segmentationCommandHandler->setCmdRunner(_cmdRunner);
        }

        connect(_cmdRunner, &CommandLineToolRunner::toolStarted,
                [this](CommandLineToolRunner::Tool /*tool*/, const QString& message) {
                    showStatusBarMessage(message, 0);
                });
        connect(_cmdRunner, &CommandLineToolRunner::toolFinished,
                [this](CommandLineToolRunner::Tool tool, bool success, const QString& message,
                       const QString& outputPath, bool copyToClipboard) {
                    Q_UNUSED(outputPath);
                    const bool neighborJobActive = _segmentationCommandHandler &&
                        _segmentationCommandHandler->neighborCopyJob().has_value() &&
                        tool == CommandLineToolRunner::Tool::NeighborCopy;
                    const bool resumeLocalActive = _segmentationCommandHandler &&
                        !_segmentationCommandHandler->neighborCopyJob() &&
                        _segmentationCommandHandler->resumeLocalJob().has_value() &&
                        tool == CommandLineToolRunner::Tool::NeighborCopy;

                    const bool silentExecution =
                        _cmdRunner && _cmdRunner->currentExecutionIsSilent();

                    if (!silentExecution) {
                        if (success) {
                            QString displayMsg = message;
                            if (copyToClipboard) displayMsg += tr(" - Path copied to clipboard");
                            showStatusBarMessage(displayMsg, 5000);
                            QMessageBox::information(this, tr("Operation Complete"), displayMsg);
                        } else {
                            showStatusBarMessage(tr("Operation failed"), 5000);
                            QMessageBox::critical(this, tr("Error"), message);
                        }
                    }

                    if (neighborJobActive) {
                        _segmentationCommandHandler->handleNeighborCopyToolFinished(success);
                    }
                    if (resumeLocalActive) {
                        _segmentationCommandHandler->resumeLocalJob().reset();
                    }
                });
    }
    return true;
}

