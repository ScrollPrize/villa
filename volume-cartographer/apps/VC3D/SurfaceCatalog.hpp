#pragma once
// The surfaces of the current volume in the user's object store and in
// read-only catalog stores (Open Data samples): one row per surface record,
// bound to the volume or not (like the fiber catalog), kept current from the
// store's commit events, and the materialization of the one surface being
// viewed or edited. Surfaces are identified by their store UUID. A catalog
// surface is edited in the user's store as the same surface (importCatalogSurface);
// from then on its row and crossings come from the user's store.
#include "ObjectStoreCoordinates.hpp"
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/objects/GeometryRegion.hpp>
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
    // surface still remote); fetch() gives it its geometry.
    struct NoGeometry : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    // Gives a catalog surface its geometry (blocking; runs on a worker thread).
    using Fetcher = std::function<void(const std::filesystem::path& store, const std::string& uuid)>;
    // A store the rows come from, listed by the volume its surfaces are bound
    // to: the user's store by the open volume's id; a catalog store by the
    // catalog volume the open volume shows, `scale` open-volume voxels per
    // bound-volume voxel (an Open Data pyramid level: 1/2^level).
    struct Source {
        vc::objects::AttachmentIdentity attachment;
        std::string volume;
        double scale = 1;
        // Surfaces of a catalog store that the user's store holds too
        // (imported): read from the user's store instead.
        std::set<std::string> shadowed;
        bool operator==(const Source&) const = default;
    };
    struct Row {
        // uuid, name, tags, meta, surface detail, change stamps and approval.
        vc::objects::ObjectRecord record;
        // Stored coordinates to the open volume's voxels; nullopt when the store
        // cannot map the surface into the volume.
        std::optional<vc::objects::VolumeAffine> mapping;
        // The store it is in; catalog rows are read-only.
        Source source;
        bool catalog = false;
        // A surface imported from a catalog store: the source still holding the
        // catalog version (its first revision in the user's store).
        std::optional<Source> original;
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
        vc3d::VolumeView view;
        // The surface's current revision, not a past one being viewed.
        bool current = true;
        // The current revision in a writable attachment.
        bool editable = false;
    };

    explicit SurfaceCatalog(std::shared_ptr<vc::objects::ObjectStoreSession> session, QObject* parent = nullptr);
    ~SurfaceCatalog() override;

    // List the surfaces of `volume` in `attachment` (nothing without both).
    void setSource(const std::optional<vc::objects::AttachmentIdentity>& attachment, std::shared_ptr<Volume> volume);
    // Also list these read-only catalog stores (while a volume is open).
    void setCatalogSources(std::vector<Source> sources);
    // The user's store (when attached) then the catalog stores.
    std::vector<Source> sources() const;
    void setFetcher(Fetcher fetcher) { fetcher_ = std::move(fetcher); }
    // The work that fetches a catalog row's geometry; run it on a worker thread.
    std::function<void()> fetchJob(const std::string& uuid) const;
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
                       const std::string& uuid, double scale = 1,
                       const std::optional<std::string>& revision = std::nullopt);

    // Where the surfaces of every source cross `slice` (open-volume voxels),
    // with `options` (uuids, budgets) per source; complete when every source's
    // read was. Synchronous: callers run it on a worker thread.
    struct Crossings {
        std::vector<vc::objects::VisibleRun> runs;
        bool complete = true;
    };
    static Crossings region(vc::objects::ObjectStoreSession& session, const std::vector<Source>& sources,
                            const vc::objects::RegionSlice& slice, vc::objects::RegionOptions options,
                            const std::function<bool()>& cancelled = {});

    // Surfaces (among `uuids`, all when nullopt) within `tolerance` voxels of a
    // volume point, nearest first. Synchronous.
    std::vector<vc::objects::SurfaceProximityHit> nearest(const std::array<double, 3>& point, double tolerance,
        const std::optional<std::vector<std::string>>& uuids = std::nullopt) const;
    // The same over explicit sources, for worker threads.
    static std::vector<vc::objects::SurfaceProximityHit> nearest(vc::objects::ObjectStoreSession& session,
        const std::vector<Source>& sources, const std::array<double, 3>& point, double tolerance,
        const std::optional<std::vector<std::string>>& uuids = std::nullopt);

    // Where tools launched for the open volume read and write: the user's store
    // file (a hosted replica's local file) and the volume's id. Throws without a
    // writable attachment and an open volume.
    struct ToolStore {
        std::filesystem::path store;
        std::string volume;
    };
    ToolStore toolStore() const;

    // Record edits: they commit at once and reach rows() through the commit events.
    void setTags(const std::string& uuid, const std::vector<std::string>& tags);
    // Bind the surface to the open volume: `storedUnitsPerVoxel` stored units per voxel.
    void bind(const std::string& uuid, double storedUnitsPerVoxel);
    void rename(const std::string& uuid, const std::string& name);
    void updateMeta(const std::string& uuid, const std::function<void(nlohmann::json&)>& change);
    void retire(const std::vector<std::string>& uuids);
    // Edit a catalog surface in the user's store: imports it there as the same
    // surface, its catalog version (as stored) the first revision, and reads its
    // row from there. Throws without its geometry (fetch it first).
    void importCatalogSurface(const std::string& uuid);
    // A new surface holding `surface` (in the volume's voxels), bound to the
    // volume; `provenance` records how it was made ({tool, parameters, sources}).
    // Sets surface.id and returns the UUID.
    std::string create(QuadSurface& surface, const std::string& name, const nlohmann::json& provenance,
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
    void editRecord(const std::string& uuid, const std::function<void(vc::objects::ObjectRecord&)>& change);
    // The surface's current object revision in the user's store; throws when it is gone.
    std::string head(const std::string& uuid) const;

    std::shared_ptr<vc::objects::ObjectStoreSession> session_;
    std::optional<vc::objects::AttachmentIdentity> attachment_;
    std::shared_ptr<Volume> volume_;
    std::vector<Source> catalogSources_;
    Fetcher fetcher_;
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
