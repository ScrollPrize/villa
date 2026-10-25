#pragma once
// The surfaces of the current volume in the user's object store: one row per
// surface record, bound to the volume or not (like the fiber catalog), kept current from
// the store's commit events, and the materialization of the one surface being
// viewed or edited. Surfaces are identified by their store UUID.
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <nlohmann/json.hpp>
#include <QObject>
#include <QString>
#include <atomic>
#include <functional>
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
    struct Row {
        // uuid, name, tags, meta, surface detail, change stamps and approval.
        vc::objects::ObjectRecord record;
        // Volume voxels per stored unit; nullopt for a surface not bound to the volume.
        std::optional<double> toVolume;
    };
    // A materialized surface: points in the volume's voxels, pinned at the
    // revisions it was read from.
    struct Opened {
        std::shared_ptr<QuadSurface> surface;
        vc::objects::PinnedObject pin;
        double toVolume = 1;
        // Writable attachment and stored at the volume's scale: an edit can
        // publish the vertices it changed without rescaling the others.
        bool editable = false;
    };

    explicit SurfaceCatalog(std::shared_ptr<vc::objects::ObjectStoreSession> session, QObject* parent = nullptr);
    ~SurfaceCatalog() override;

    // List the surfaces of `volume` in `attachment` (nothing without both).
    void setSource(const std::optional<vc::objects::AttachmentIdentity>& attachment, std::shared_ptr<Volume> volume);
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

    // The surface's current geometry in the volume's voxels; channels read on
    // demand from the same revision. Throws when it cannot be read.
    Opened open(const std::string& uuid) const;
    // The same from an explicit source, for worker threads.
    static Opened open(const std::shared_ptr<vc::objects::ObjectStoreSession>& session,
                       const vc::objects::AttachmentIdentity& attachment, const std::string& volume,
                       const std::string& uuid);

    // Surfaces (among `uuids`, all when nullopt) within `tolerance` voxels of a
    // volume point, nearest first. Synchronous.
    std::vector<vc::objects::SurfaceProximityHit> nearest(const std::array<double, 3>& point, double tolerance,
        const std::optional<std::vector<std::string>>& uuids = std::nullopt) const;

    // Record edits: they commit at once and reach rows() through the commit events.
    void setTags(const std::string& uuid, const std::vector<std::string>& tags);
    void rename(const std::string& uuid, const std::string& name);
    void updateMeta(const std::string& uuid, const std::function<void(nlohmann::json&)>& change);
    void retire(const std::vector<std::string>& uuids);
    // A new surface with the current record and geometry of `uuid`; returns its UUID.
    std::string copy(const std::string& uuid, const std::string& name);
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
    // Read the rows of `uuids` (all surfaces when nullopt) and merge them.
    void read(std::optional<std::vector<std::string>> uuids);
    void schedule();
    void editRecord(const std::string& uuid, const std::function<void(vc::objects::ObjectRecord&)>& change);

    std::shared_ptr<vc::objects::ObjectStoreSession> session_;
    std::optional<vc::objects::AttachmentIdentity> attachment_;
    std::shared_ptr<Volume> volume_;
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
