#pragma once
// Where edits of an opened surface go: its object in the store, pinned at the
// revisions the edit starts from and re-pinned from each commit's receipt.
// Relationships are kept as recorded. Saves may run on worker threads.
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
    SurfaceStoreEdit(std::shared_ptr<vc::objects::ObjectStoreSession> session, vc::objects::PinnedObject pin);

    // An editing session holds the object (vc3d::store::holdForEditing) until
    // release(); throws when the store is read-only or the pin is stale.
    void hold();
    void release();

    // Publish `surface`: with the stored grid size, the named channels ("mask"
    // is validity) and then the vertices of `rect` that changed; with another
    // grid size (growth), a new geometry of points, validity and channels.
    void save(QuadSurface& surface, const cv::Rect& rect, const std::vector<std::string>& channels = {});

    vc::objects::PinnedObject pin() const;
    const std::string& uuid() const { return _uuid; }

private:
    std::shared_ptr<vc::objects::ObjectStoreSession> _session;
    std::string _uuid;
    mutable std::mutex _mutex;
    vc::objects::PinnedObject _pin;
    vc::objects::LeaseHold _lease;
};
