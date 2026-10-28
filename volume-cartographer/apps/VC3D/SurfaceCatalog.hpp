#pragma once
// The surfaces of the current volume in the user's object store and in
// catalog stores (Open Data samples): one row per surface record,
// bound to the volume or not (like the fiber catalog), kept current from the
// store's commit events, and the materialization of the one surface being
// viewed or edited. Surfaces are identified by their store UUID and read in the
// open volume's voxels, through the stores' registrations (an Open Data level
// is a registered volume). A catalog surface is edited in the user's store as
// the same surface (importCatalogSurface); the session then reads it from there.
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/objects/GeometryRegion.hpp>
#include <vc_object_store/objects/SurfaceCreate.hpp>
#include <nlohmann/json.hpp>
#include <QObject>
#include <QString>
#include <atomic>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

class QuadSurface;
class Volume;

class SurfaceCatalog : public QObject
{
    Q_OBJECT

public:
    // open() of a surface whose geometry is not in its store yet (a catalog
    // surface still remote): the session's ensureGeometry fetches it.
    struct NoGeometry : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    struct Row {
        // uuid, name, tags, meta, surface detail, change stamps and approval.
        vc::objects::ObjectRecord record;
        // Stored coordinates to the open volume's voxels; nullopt when the store
        // cannot map the surface into the volume.
        std::optional<vc::objects::VolumeAffine> mapping;
        // The store it is in; catalog rows retain their sample store.
        vc::objects::AttachmentIdentity source;
        bool catalog = false;
        // A catalog surface imported into the user's store: its first revision
        // there is the catalog version (ObjectStoreSession::restoreImported).
        bool imported() const;
        // Shown and opened in the open volume (through `mapping`).
        bool bound() const { return mapping.has_value(); }
        // Input for external tools, which read stored coordinates: a user
        // surface stored in the open volume's voxels.
        bool toolInput() const { return !catalog && mapping == vc::objects::VolumeAffine::identity(); }
    };
    // A materialized surface: points in the volume's voxels, pinned at the
    // revisions it was read from.
    struct Opened {
        std::shared_ptr<QuadSurface> surface;
        vc::objects::PinnedObject pin;
        // The view it was read through; edits map back through it.
        vc::objects::ViewFrame view;
        // The surface's current revision, not a past one being viewed.
        bool current = true;
        // The current revision in a writable attachment.
        bool editable = false;
    };

    explicit SurfaceCatalog(std::shared_ptr<vc::objects::ObjectStoreSession> session, QObject* parent = nullptr);
    ~SurfaceCatalog() override;

    // List the surfaces of `volume` in `attachment` (nothing without both).
    void setSource(const std::optional<vc::objects::AttachmentIdentity>& attachment, std::shared_ptr<Volume> volume);
    // Also list these catalog stores (while a volume is open).
    void setCatalogSources(std::vector<vc::objects::AttachmentIdentity> sources);
    // The user's store (when attached) then the catalog stores: the order in
    // which the session reads a surface held by several (imported ones).
    std::vector<vc::objects::AttachmentIdentity> sources() const;
    // Session events (CState::objectStoreChanged); other attachments are ignored.
    void handle(const vc::objects::StoreSessionEvent& event);

    const std::shared_ptr<vc::objects::ObjectStoreSession>& session() const { return session_; }
    const std::optional<vc::objects::AttachmentIdentity>& attachment() const { return attachment_; }
    const std::shared_ptr<Volume>& volume() const { return volume_; }
    const std::map<std::string, Row>& rows() const { return rows_; }
    const Row* row(const std::string& uuid) const;
    // A full listing is being read; rows() holds the previous one meanwhile.
    bool loading() const { return loading_; }
    const QString& error() const { return error_; }

    // The surface's geometry in the volume's voxels: current, or at object
    // revision `revision` of its history (read-only); channels read on demand
    // from the same revision. Throws when it cannot be read.
    Opened open(const std::string& uuid, const std::optional<std::string>& revision = std::nullopt) const;
    // The same from an explicit source, for worker threads.
    static Opened open(const std::shared_ptr<vc::objects::ObjectStoreSession>& session,
                       const vc::objects::AttachmentIdentity& attachment, const std::string& volume,
                       const std::string& uuid, const std::optional<std::string>& revision = std::nullopt);

    // Where the surfaces of every source cross `slice` (voxels of `volume`),
    // with `options` (uuids, budgets) per source; complete when every source's
    // read was. Placeholders reached (geometry not local yet) are listed with
    // their attachment; fetch them (ObjectStoreSession::ensureGeometry) and
    // read again to include them.
    struct Crossings {
        std::vector<vc::objects::VisibleRun> runs;
        bool complete = true;
        std::vector<std::pair<vc::objects::AttachmentIdentity, std::string>> remote;
    };
    static Crossings region(vc::objects::ObjectStoreSession& session,
                            const std::vector<vc::objects::AttachmentIdentity>& sources, const std::string& volume,
                            const vc::objects::RegionSlice& slice, vc::objects::RegionOptions options,
                            const std::function<bool()>& cancelled = {});

    // The nearest point of each of `uuids` (drafts or objects of the sources)
    // within `tolerance` of each of `points` (voxels of `volume`): the hits per
    // point, nearest first, each with its fractional grid position. Placeholders
    // reached are queued for fetching, not waited for.
    static std::vector<std::vector<vc::objects::SurfaceProximityHit>> nearest(
        vc::objects::ObjectStoreSession& session, const std::vector<vc::objects::AttachmentIdentity>& sources,
        const std::string& volume, const std::vector<std::string>& uuids,
        const std::vector<std::array<double, 3>>& points, double tolerance);

    // Where tools launched for the open volume read and write: the user's store
    // file (a hosted replica's local file) and the volume's id. Throws without a
    // writable attachment and an open volume.
    struct ToolStore {
        std::filesystem::path store;
        std::string volume;
    };
    ToolStore toolStore() const;

    // Edit a catalog surface in the user's store: imports it there as the same
    // surface (fetching it first), its catalog version the first revision, and
    // reads its row from there. `opened` (its current version, as on screen)
    // continues as the user's: same surface object, pinned and reading
    // channels there. Throws when the user's store places it differently.
    Opened importCatalogSurface(const Opened& opened);
    // A new surface holding `surface` (in the volume's voxels), bound to the
    // volume; `provenance` records how it was made. Sets surface.id and returns
    // the UUID.
    std::string create(QuadSurface& surface, const std::string& name, const vc::objects::SurfaceProvenance& provenance,
                       const std::vector<std::string>& tags = {});

signals:
    // The row set or a row changed.
    void rowsChanged();
    // Commits changed these surfaces (record or geometry), including retired ones.
    void surfacesCommitted(const std::vector<std::string>& uuids);
    void errorChanged(const QString& error);

private:
    // Read these surfaces' rows now (synchronously), e.g. one just created,
    // rather than waiting for the commit event.
    void readNow(const std::vector<std::string>& uuids);
    // Read the rows of `uuids` (all surfaces when nullopt) and merge them.
    void read(std::optional<std::vector<std::string>> uuids);
    void schedule();
    // The sources changed: drop the rows and pending reads, list again.
    void restart();

    std::shared_ptr<vc::objects::ObjectStoreSession> session_;
    std::optional<vc::objects::AttachmentIdentity> attachment_;
    std::shared_ptr<Volume> volume_;
    std::vector<vc::objects::AttachmentIdentity> catalogSources_;
    std::map<std::string, Row> rows_;
    // Bumped by setSource(): reads of an earlier source are dropped.
    uint64_t generation_ = 0;
    std::shared_ptr<std::atomic_bool> cancelled_;
    // One read runs at a time; reads requested meanwhile wait here.
    bool busy_ = false;
    bool pendingFull_ = false;
    std::set<std::string> pending_;
    bool loading_ = false;
    QString error_;
};
