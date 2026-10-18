#include "agent_bridge/AgentBridgeServer.hpp"

#include <QJsonArray>
#include <QJsonDocument>
#include <QTabWidget>

#include "CState.hpp"
#include "CWindow.hpp"
#include "WrapAnnotationWidget.hpp"
#include "segmentation/SegmentationModule.hpp"

namespace {

QByteArray fingerprint(const QJsonValue& value)
{
    if (value.isObject())
        return QJsonDocument(value.toObject()).toJson(QJsonDocument::Compact);
    if (value.isArray())
        return QJsonDocument(value.toArray()).toJson(QJsonDocument::Compact);
    // Wrap scalars so null, false and "" stay distinct.
    return QJsonDocument(QJsonArray{value}).toJson(QJsonDocument::Compact);
}

}  // namespace


QHash<QString, QByteArray> AgentBridgeServer::observedStateFingerprints(
    const QJsonObject& snapshot)
{
    QHash<QString, QByteArray> fields;
    fields["project"] = fingerprint(snapshot.value("vpkg"));
    fields["volume"] = fingerprint(snapshot.value("volume"));
    fields["activeSegment"] = fingerprint(snapshot.value("activeSurface"));

    QJsonObject editing;
    for (const char* key : {"segmentationEditingEnabled", "manualAddMode",
                            "manualAddLineMode", "manualAddInterpolation",
                            "correctionsPointMode", "sameWrapAnnotation",
                            "axisAlignedSlices"}) {
        editing[QLatin1String(key)] = snapshot.value(QLatin1String(key));
    }
    fields["editing"] = fingerprint(editing);

    // Which viewers exist and where they are shown; zoom and pan are not
    // reported.
    QJsonArray viewers;
    for (const QJsonValue& value : snapshot.value("viewers").toArray()) {
        const QJsonObject viewer = value.toObject();
        QJsonObject entry;
        for (const char* key : {"viewerId", "surfName", "workspace", "visible"})
            entry[QLatin1String(key)] = viewer.value(QLatin1String(key));
        viewers.append(entry);
    }
    fields["viewers"] = fingerprint(viewers);

    // The tab index distinguishes unnamed line-annotation tabs.
    QTabWidget* tabs = _window ? _window->_workspaceTabs : nullptr;
    fields["workspace"] = fingerprint(snapshot.value("workspace").toObject().value("current")) +
                          QByteArray::number(tabs ? tabs->currentIndex() : -1);

    try {
        fields["overlay"] = fingerprint(handleViewerGetOverlay(QJsonValue()));
    } catch (const AgentBridgeError&) {
        fields["overlay"] = QByteArray();
    }

    // Running jobs plus the latest job's state, so a job that starts and
    // finishes between two refreshes is still reported.
    QByteArray jobs;
    for (const JobRecord& job : _activeJobs)
        jobs += job.id.toUtf8() + ':' + job.state.toUtf8() + ';';
    if (const JobRecord* latest = mostRecentJob())
        jobs += "latest=" + latest->id.toUtf8() + ':' + latest->state.toUtf8();
    fields["jobs"] = jobs;
    return fields;
}


void AgentBridgeServer::refreshObservedState(const QJsonObject& snapshot)
{
    if (!_stateTracker.update(observedStateFingerprints(snapshot)))
        return;
    QJsonObject params;
    params["revision"] = static_cast<double>(_stateTracker.revision());
    params["changed"] = QJsonArray::fromStringList(_stateTracker.takePending());
    broadcastNotification(QStringLiteral("state.changed"), params);
}


void AgentBridgeServer::refreshObservedStateNow()
{
    // Runs from signals, queued work and dispatch; nothing may escape into Qt.
    try {
        syncViewerManagers();
        refreshObservedState(stateSnapshot());
    } catch (const std::exception& e) {
        qWarning("agent bridge: state check failed: %s", e.what());
    } catch (...) {
        qWarning("agent bridge: state check failed");
    }
}


void AgentBridgeServer::queueStateRefresh()
{
    if (_stateRefreshQueued)
        return;
    _stateRefreshQueued = true;
    runWhenIdle([this]() {
        _stateRefreshQueued = false;
        refreshObservedStateNow();
    });
}


void AgentBridgeServer::connectStateSignals()
{
    if (!_window)
        return;
    // Fields with an application signal are checked as soon as it fires; the
    // others (manual add, corrections, axis-aligned slices, overlay, viewer
    // visibility) after the next request or in state.get.
    if (CState* state = _window->_state) {
        connect(state, &CState::vpkgChanged, this, [this]() { queueStateRefresh(); });
        connect(state, &CState::volumeChanged, this, [this]() { queueStateRefresh(); });
        connect(state, &CState::volumeClosing, this, [this]() { queueStateRefresh(); });
        // A newly active segment replaces the "segmentation" surface; edit
        // updates of the same segment are not an observed change.
        connect(state, &CState::surfaceChanged, this,
                [this](const std::string& name, const std::shared_ptr<Surface>&,
                       bool isEditUpdate) {
                    if (!isEditUpdate && name == "segmentation")
                        queueStateRefresh();
                });
    }
    if (SegmentationModule* module = _window->_segmentationModule.get()) {
        connect(module, &SegmentationModule::editingEnabledChanged,
                this, [this]() { queueStateRefresh(); });
    }
    if (WrapAnnotationWidget* wrap = _window->_wrapAnnotationWidget) {
        connect(wrap, &WrapAnnotationWidget::sameWrapAnnotationToggled,
                this, [this]() { queueStateRefresh(); });
    }
    if (QTabWidget* tabs = _window->_workspaceTabs) {
        connect(tabs, &QTabWidget::currentChanged, this, [this]() { queueStateRefresh(); });
    }
}
