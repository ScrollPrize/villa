#pragma once
#include "geometry/SharedGeometry.hpp"

namespace vc::objects::detail {
// A flat, immutable index of 128-entry metadata buckets. Structural edits
// write a new index; coordinate deltas share the entire metadata snapshot.
class MetadataCacheScope {
    bool owner_;
public:
    MetadataCacheScope();
    ~MetadataCacheScope();
    MetadataCacheScope(const MetadataCacheScope&)=delete;
    MetadataCacheScope& operator=(const MetadataCacheScope&)=delete;
};
class MetadataSnapshot {
    sqlite3* db_;
    SharedJson buckets_;
    std::map<int64_t,SharedJson> dirty_;
    SharedJson bucket(int64_t) const;
public:
    explicit MetadataSnapshot(sqlite3*, const SharedJson& root=nullptr);
    SharedJson get(int64_t) const;
    void put(int64_t,const SharedJson&);
    void entries(const std::function<void(int64_t,const SharedJson&)>&) const;
    void mergeAllocated(const MetadataSnapshot&);
    SharedJson save();
};
SharedJson identityGet(sqlite3*,const SharedJson&,int64_t);
SharedJson packAnnotation(sqlite3*,const SharedJson&);
SharedJson unpackAnnotation(sqlite3*,const SharedJson&);
bool annotationDirectional(const SharedJson&);
// Initial sequence state. Source annotations are consumed only at creation.
SharedJson initialSequences(sqlite3*, const GeometryData&, SharedJson& source);
// Verified payloads of a metadata snapshot, dependencies before what refers to them.
void metadataPayloads(sqlite3* source,const SharedJson&,const std::function<void(const std::vector<uint8_t>&)>&);
}
