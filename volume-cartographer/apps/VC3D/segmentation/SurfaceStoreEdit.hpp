#pragma once
// Where edits of an opened surface go: its object in the store, pinned at the
// revisions the edit starts from and re-pinned from each commit's receipt.
// Relationships are kept as recorded. Saves may run on worker threads.
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/objects/AttachedDocument.hpp>
#include <opencv2/core/types.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

class QuadSurface;

class SurfaceStoreEdit
{
public:
    // Edits of a surface read through `view` map back to its stored frame;
    // without one the surface holds stored coordinates.
    SurfaceStoreEdit(std::shared_ptr<vc::objects::ObjectStoreSession> session, vc::objects::PinnedObject pin,
                     std::optional<vc::objects::ViewFrame> view = std::nullopt);

    // Publish `surface`: with the stored grid size, the named channels ("mask"
    // is validity) and then the vertices of `rect` that changed; with another
    // grid size (growth), a new geometry of points, validity and channels.
    // A durable save (ObjectStoreSession::submitDurably): written to the
    // session's draft directory before it is sent. `draftRevision` is the
    // revision of the surface's session draft that `surface` was copied at:
    // the draft stays (the editor keeps it) and is marked saved when it did
    // not change since. When a hosted store does not answer, the save stays
    // pending and the next save first resolves it (its receipt, or the
    // identical request again), so saves never skip one another.
    void save(QuadSurface& surface, const cv::Rect& rect, const std::vector<std::string>& channels = {},
              std::optional<uint64_t> draftRevision = std::nullopt);

    vc::objects::PinnedObject pin() const;
    // The pin the edit started from (the opened surface's).
    const vc::objects::PinnedObject& origin() const { return _origin; }
    const std::string& uuid() const { return _uuid; }
    const std::optional<vc::objects::ViewFrame>& view() const { return _view; }
    // One editor owns both the surface and its loaded attached collections.
    // All its writes share the loaded membership guard and repin together.
    void watchAttachedCollections(const vc::objects::AttachedCollectionsSnapshot&);
    void saveAttachedCollections(vc::objects::AttachedCollectionsSnapshot&,
        const std::vector<vc::objects::AttachedCollection>&, const nlohmann::json& coordinates);
    vc::objects::TransactionReceipt submitAttached(vc::objects::TransactionRequest);

private:
    std::shared_ptr<vc::objects::ObjectStoreSession> _session;
    std::string _uuid;
    mutable std::mutex _mutex;
    vc::objects::PinnedObject _pin;
    const vc::objects::PinnedObject _origin;
    std::optional<std::string> _pending; // a save whose outcome is unknown
    void settlePending();
    void repin(const vc::objects::TransactionReceipt&);
    const std::optional<vc::objects::ViewFrame> _view;
    std::optional<vc::objects::AttachedCollectionsSnapshot> _collections;
    void guardAttached(vc::objects::TransactionRequest&) const;
};
