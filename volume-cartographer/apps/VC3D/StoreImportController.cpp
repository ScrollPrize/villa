#include "StoreImportController.hpp"

#include <QDir>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QPointer>
#include <QtConcurrent>

#include <cmath>
#include <utility>

#include "CState.hpp"
#include "CWindow.hpp"
#include "LineAnnotationController.hpp"

#include "vc/core/types/VolumePkg.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>

namespace {
bool fail(StoreImportFailure kind, const QString& message, const QString& param, QString* errorMessage,
          StoreImportFailure* failure, QString* paramOut)
{
    if (errorMessage) *errorMessage = message;
    if (failure) *failure = kind;
    if (paramOut) *paramOut = param;
    return false;
}
}

StoreImportController::StoreImportController(CWindow* window)
    : QObject(window)
    , _window(window)
{
}

bool StoreImportController::prepare(const StoreImportParams& params, StoreImportRequest* request,
                                    QString* errorMessage, StoreImportFailure* failure, QString* param) const
{
    const auto invalid = [&](const QString& name, const QString& message) {
        return fail(StoreImportFailure::InvalidParams, message, name, errorMessage, failure, param);
    };
    if (failure) *failure = StoreImportFailure::None;
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->vpkg())
        return fail(StoreImportFailure::NoProject, QObject::tr("Open a project first."), {}, errorMessage, failure, param);
    if (params.kind != "surface" && params.kind != "fiber" && params.kind != "point_collection")
        return invalid(QStringLiteral("kind"), QObject::tr("Unsupported import kind: %1").arg(QString::fromStdString(params.kind)));
    if (params.source.empty() || !params.source.is_absolute())
        return invalid(QStringLiteral("path"), QObject::tr("The source path must be absolute."));
    std::error_code existsError;
    if (!std::filesystem::exists(params.source, existsError))
        return fail(StoreImportFailure::SourceNotFound,
                    QObject::tr("The source does not exist: %1").arg(QString::fromStdString(params.source.string())),
                    QStringLiteral("path"), errorMessage, failure, param);

    StoreImportRequest prepared;
    prepared.kind = params.kind;
    prepared.source = params.source.lexically_normal();
    if (params.tags) {
        prepared.tags = *params.tags;
    } else {
        const auto name = QFileInfo(QDir::cleanPath(QString::fromStdString(prepared.source.string()))).fileName();
        if (!name.isEmpty()) prepared.tags.push_back(name.toStdString());
    }
    if (params.bindScale) {
        const double scale = *params.bindScale;
        if (!std::isfinite(scale) || !(scale > 0))
            return invalid(QStringLiteral("bind"), QObject::tr("The coordinate scale must be positive and finite."));
        if (!state->currentVolume() || !state->annotationVolume())
            return invalid(QStringLiteral("bind"), QObject::tr("Binding to a volume needs an open volume."));
        const auto& annotation = *state->annotationVolume();
        prepared.coordinates = vc::objects::volumeCoordinates(annotation.volume, scale * annotation.scale).dump();
    }
    if (params.uuidNamespace) {
        if (!vc::objects::util::isUuid(*params.uuidNamespace))
            return invalid(QStringLiteral("uuidNamespace"), QObject::tr("The UUID namespace must be a UUID."));
        prepared.options.uuidNamespace = vc::objects::util::canonicalUuid(*params.uuidNamespace);
    }
    prepared.options.requireSourceUuids = params.requireSourceUuids;
    prepared.options.stopOnInfrastructureError = params.stopOnInfrastructureError;

    const auto target = params.storeReference ? state->storeReference(*params.storeReference)
                                              : state->storeFor(params.kind);
    if (!target) {
        return fail(StoreImportFailure::NoStore,
                    params.storeReference
                        ? QObject::tr("No attached store has reference %1.").arg(QString::fromStdString(*params.storeReference))
                        : QObject::tr("No store is selected for %1.").arg(QString::fromStdString(params.kind)),
                    QStringLiteral("store"), errorMessage, failure, param);
    }
    if (!state->objectStoreSession()->writable(*target))
        return fail(StoreImportFailure::ReadOnlyStore, QObject::tr("The store is read-only."), QStringLiteral("store"),
                    errorMessage, failure, param);
    if (params.refuseDuringFiberStructuralEdits && params.kind == "fiber" && _window->_lineAnnotationController &&
        _window->_lineAnnotationController->structuralEditInProgress())
        return fail(StoreImportFailure::Busy,
                    QObject::tr("A merge, split or link is in progress; import once it has finished."), {},
                    errorMessage, failure, param);
    prepared.target = *target;
    prepared.storeReference = state->storeReferenceId(*target).value_or(std::string{});
    if (request) *request = std::move(prepared);
    if (errorMessage) errorMessage->clear();
    return true;
}

bool StoreImportController::start(StoreImportRequest request,
                                  std::function<void(const StoreImportProgress&)> onProgress,
                                  std::function<void(const StoreImportOutcome&)> onFinished, QString* errorMessage)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->vpkg()) {
        if (errorMessage) *errorMessage = QObject::tr("Open a project first.");
        return false;
    }
    if (_inFlight) {
        if (errorMessage) *errorMessage = QObject::tr("An import is already in progress.");
        return false;
    }
    using Report = vc::objects::GeometryIngestReport;
    const auto targetPackage = state->vpkg();
    const auto session = state->objectStoreSession();
    auto cancel = std::make_shared<std::atomic_bool>(false);
    QPointer<StoreImportController> self(this);
    auto progress = [self, onProgress](const vc::objects::util::ProgressUpdate& event) {
        if (!onProgress) return;
        StoreImportProgress update{event.completed, event.total,
                                   QString::fromUtf8(event.detail.data(), qsizetype(event.detail.size()))};
        QMetaObject::invokeMethod(self, [self, onProgress, update] {
            if (self) onProgress(update);
        }, Qt::QueuedConnection);
    };
    auto* watcher = new QFutureWatcher<Report>(this);
    connect(watcher, &QFutureWatcher<Report>::finished, this,
            [this, watcher, targetPackage, storeReference = request.storeReference, onFinished = std::move(onFinished)] {
                StoreImportOutcome outcome;
                outcome.report = watcher->result();
                outcome.storeReference = storeReference;
                watcher->deleteLater();
                _inFlight = false;
                _target.reset();
                _cancel.reset();
                outcome.projectReplaced = !_window || !_window->_state || _window->_state->vpkg() != targetPackage;
                if (onFinished) onFinished(outcome);
            });
    _inFlight = true;
    _target = request.target;
    _cancel = cancel;
    watcher->setFuture(QtConcurrent::run([session, request = std::move(request), progress, cancel]() {
        try {
            return vc::objects::ingestGeometrySource(*session, request.target, request.kind, request.source, progress,
                                                     request.coordinates, request.tags, request.options, cancel.get());
        } catch (const std::exception& error) {
            Report report;
            report.fatalError = error.what();
            return report;
        }
    }));
    if (errorMessage) errorMessage->clear();
    return true;
}

bool StoreImportController::cancel()
{
    if (!_inFlight || !_cancel) return false;
    *_cancel = true;
    return true;
}
