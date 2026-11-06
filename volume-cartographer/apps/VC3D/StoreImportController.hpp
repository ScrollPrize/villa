#pragma once

#include <QObject>
#include <QString>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <vc_object_store/objects/GeometryIngest.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>

class CWindow;
class VolumePkg;

// What to import: source files of one object kind into one project store.
struct StoreImportParams {
    // surface, fiber or point_collection.
    std::string kind;
    std::filesystem::path source;
    // Tags added to every imported object; nullopt: the source's last path component.
    std::optional<std::vector<std::string>> tags;
    // Bind to the annotation volume at this many stored units per current-volume
    // voxel. Without it the sources keep the coordinates they declare.
    std::optional<double> bindScale;
    std::optional<std::string> uuidNamespace;
    // A project store reference id; nullopt: the store routed for `kind`.
    std::optional<std::string> storeReference;
    // A fiber without a UUID needs uuidNamespace instead of its file-name identity.
    bool requireSourceUuids{false};
    // A failure that is not the source's stops the import (report.fatalError).
    bool stopOnInfrastructureError{false};
    // Refuse a fiber import while a fiber merge, split or link is running.
    bool refuseDuringFiberStructuralEdits{false};
};

enum class StoreImportFailure {
    None,
    NoProject,
    InvalidParams,
    SourceNotFound,
    NoStore,
    ReadOnlyStore,
    Busy,
};

struct StoreImportRequest {
    std::string kind;
    std::filesystem::path source;
    std::vector<std::string> tags;
    // vc::objects coordinate context ({} without a binding).
    std::string coordinates{"{}"};
    vc::objects::SourceIngestOptions options;
    vc::objects::AttachmentIdentity target;
    std::string storeReference;
};

struct StoreImportProgress {
    std::uint64_t completed{0};
    std::uint64_t total{0};
    QString detail;
};

struct StoreImportOutcome {
    vc::objects::GeometryIngestReport report;
    std::string storeReference;
    // The project was replaced or closed before the import finished.
    bool projectReplaced{false};
};

// Imports source files into a project store off the GUI thread
// (vc::objects::ingestGeometrySource), one import at a time. Import copies the
// geometry into the store and never changes routing.
class StoreImportController final : public QObject
{
public:
    explicit StoreImportController(CWindow* window);

    // Validates `params` against the open project. On failure `param` names
    // the offending parameter for InvalidParams.
    bool prepare(const StoreImportParams& params, StoreImportRequest* request, QString* errorMessage = nullptr,
                 StoreImportFailure* failure = nullptr, QString* param = nullptr) const;
    // Runs the import; both callbacks run on the GUI thread, `onFinished` once.
    bool start(StoreImportRequest request, std::function<void(const StoreImportProgress&)> onProgress,
               std::function<void(const StoreImportOutcome&)> onFinished, QString* errorMessage = nullptr);
    bool inFlight() const { return _inFlight; }
    // The attachment the running import writes to.
    std::optional<vc::objects::AttachmentIdentity> target() const { return _target; }
    // Stops the running import before its next object; false when none runs.
    bool cancel();

private:
    CWindow* _window{nullptr};
    bool _inFlight{false};
    std::optional<vc::objects::AttachmentIdentity> _target;
    std::shared_ptr<std::atomic_bool> _cancel;
};
