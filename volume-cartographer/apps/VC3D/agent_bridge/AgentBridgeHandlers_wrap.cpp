#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

#include <QBuffer>
#include <QByteArray>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFutureWatcher>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLocalServer>
#include <QLocalSocket>
#include <QMdiSubWindow>
#include <QPixmap>
#include <QPointF>
#include <QTabWidget>
#include <QTimer>
#include <QVector3D>
#include <QWidget>
#include <QtConcurrent>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <set>
#include <string>
#include <unordered_set>

#include "CWindow.hpp"
#include "AxisAlignedSliceController.hpp"
#include "CState.hpp"
#include "LasagnaServiceManager.hpp"
#include "LineAnnotationController.hpp"
#include "LineAnnotationDialog.hpp"
#include "MenuActionController.hpp"
#include "OpenDataManifest.hpp"
#include "OpenDataSampleProject.hpp"
#include "OpenDataSegmentCache.hpp"
#include "SeedingWidget.hpp"
#include "SegmentationCommandHandler.hpp"
#include "SurfacePanelController.hpp"
#include "CommandLineToolRunner.hpp"
#include "ViewerManager.hpp"
#include "WrapAnnotationWidget.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/SegmentationPushPullConfig.hpp"
#include "segmentation/SegmentationWidget.hpp"
#include "segmentation/tools/SegmentationPushPullTool.hpp"
#include "segmentation/panels/SegmentationLasagnaPanel.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"
#include "segmentation/growth/SegmentationGrower.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include "vc/core/types/Segmentation.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/ui/VCCollection.hpp"

// Same-winding wrap annotation: human flow is enable "Same-wrap
// annotation mode" -> shift-click a chunked viewer to seed preview points ->
// shift+E to commit (Ctrl+Z undoes). These RPCs expose set_mode/commit/undo;
// preview is seeded via canvas.shift_click, exactly as for a human.

QJsonObject AgentBridgeServer::handleWrapAnnotationSetMode(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const bool enabled = p.value("enabled").toBool();

    WrapAnnotationWidget* widget = _window ? _window->_wrapAnnotationWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "wrap annotation widget is not available";
        throw AgentBridgeError{-32000, "Wrap annotation widget unavailable", data};
    }

    // Drives the checkbox's toggled() signal exactly as a click would; setting
    // the current state is an inert no-op.
    widget->setSameWrapAnnotationEnabled(enabled);

    QJsonObject result;
    result["enabled"] = widget->sameWrapAnnotationEnabled();
    return result;
}


QJsonObject AgentBridgeServer::handleWrapAnnotationCommit(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();

    // Precondition: same-wrap mode must be enabled, mirroring the shift+E key
    // handler's guard (CWindow::keyPressEvent). commit is otherwise a no-op.
    WrapAnnotationWidget* widget = _window ? _window->_wrapAnnotationWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "wrap annotation widget is not available";
        throw AgentBridgeError{-32000, "Wrap annotation widget unavailable", data};
    }
    if (!widget->sameWrapAnnotationEnabled())
        throw AgentBridgeError{-32002, "same-wrap annotation mode is not enabled", {}};

    bool committed = false;
    bool hadPreview = false;

    if (p.contains("viewer") && !p.value("viewer").isNull()) {
        // Resolve an explicit target and commit on that viewer, using the same
        // round-trip resolution as canvas.click.
        VolumeViewerBase* viewer = resolveViewer(p.value("viewer"));
        auto* chunked = dynamic_cast<CChunkedVolumeViewer*>(viewer);
        if (!chunked) {
            QJsonObject data;
            data["detail"] = "viewer is not a chunked volume viewer";
            throw AgentBridgeError{-32009, "Unsupported viewer for canvas operation", data};
        }
        hadPreview = chunked->hasSameWrapAnnotationPreview();
        committed = chunked->commitSameWrapAnnotationPreview();
    } else {
        // No target: iterate the base viewers like the shift+E handler and
        // commit on the first chunked viewer that reports success.
        ViewerManager* vm = _window ? _window->_viewerManager.get() : nullptr;
        if (!vm) {
            QJsonObject data;
            data["detail"] = "viewer manager is not available";
            throw AgentBridgeError{-32000, "Viewer manager unavailable", data};
        }
        vm->forEachBaseViewer([&committed, &hadPreview](VolumeViewerBase* baseViewer) {
            if (committed || !baseViewer)
                return;
            auto* viewer = qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject());
            if (!viewer)
                return;
            if (viewer->hasSameWrapAnnotationPreview())
                hadPreview = true;
            committed = viewer->commitSameWrapAnnotationPreview();
        });
    }

    QJsonObject result;
    result["committed"] = committed;
    result["hadPreview"] = hadPreview;
    return result;
}


QJsonObject AgentBridgeServer::handleWrapAnnotationUndo(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();

    bool undone = false;

    if (p.contains("viewer") && !p.value("viewer").isNull()) {
        VolumeViewerBase* viewer = resolveViewer(p.value("viewer"));
        auto* chunked = dynamic_cast<CChunkedVolumeViewer*>(viewer);
        if (!chunked) {
            QJsonObject data;
            data["detail"] = "viewer is not a chunked volume viewer";
            throw AgentBridgeError{-32009, "Unsupported viewer for canvas operation", data};
        }
        undone = chunked->undoSameWrapAnnotation();
    } else {
        ViewerManager* vm = _window ? _window->_viewerManager.get() : nullptr;
        if (!vm) {
            QJsonObject data;
            data["detail"] = "viewer manager is not available";
            throw AgentBridgeError{-32000, "Viewer manager unavailable", data};
        }
        vm->forEachBaseViewer([&undone](VolumeViewerBase* baseViewer) {
            if (undone || !baseViewer)
                return;
            auto* viewer = qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject());
            if (!viewer)
                return;
            undone = viewer->undoSameWrapAnnotation();
        });
    }

    QJsonObject result;
    result["undone"] = undone;
    return result;
}
