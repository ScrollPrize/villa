#pragma once
// Where edits of an opened surface go: its object in the store, pinned at the
// revisions the edit starts from and re-pinned from each commit's receipt.
// Relationships are kept as recorded. Saves may run on worker threads.
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <opencv2/core/types.hpp>
#include <memory>
#include <mutex>
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

    // An editing session holds the object (ObjectStoreSession::holdForEditing) until
    // release(); throws when the store is read-only or the pin is stale.
    void hold();
    void release();

    // Publish `surface`: with the stored grid size, the named channels ("mask"
    // is validity) and then the vertices of `rect` that changed; with another
    // grid size (growth), a new geometry of points, validity and channels.
    void save(QuadSurface& surface, const cv::Rect& rect, const std::vector<std::string>& channels = {});

    vc::objects::PinnedObject pin() const;
    // The pin the edit started from (the opened surface's).
    const vc::objects::PinnedObject& origin() const { return _origin; }
    const std::string& uuid() const { return _uuid; }
    const std::optional<vc::objects::ViewFrame>& view() const { return _view; }

private:
    std::shared_ptr<vc::objects::ObjectStoreSession> _session;
    std::string _uuid;
    mutable std::mutex _mutex;
    vc::objects::PinnedObject _pin;
    const vc::objects::PinnedObject _origin;
    vc::objects::LeaseHold _lease;
    const std::optional<vc::objects::ViewFrame> _view;
};
