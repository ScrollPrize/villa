#include "geometry/SequenceMetadata.hpp"

namespace vc::objects::detail {
namespace {
using Json=nlohmann::json;
struct Cached {Json value;size_t bytes;};
thread_local std::unique_ptr<std::map<std::pair<sqlite3*,std::string>,Cached>> cache;
thread_local size_t cacheBytes=0;
Json load(sqlite3* db,const std::string& hash) {
    GeometryPhase phase("metadataSnapshotRead");
    if(cache) {
        const auto found=cache->find({db,hash});
        if(found!=cache->end()) {requestBytes(found->second.bytes,&FiberRegionStats::cacheBytes);return found->second.value;}
    }
    const auto bytes=readSharedPayload(db,"@metadata",hash);
    auto value=Json::parse(bytes);const auto charge=bytes.size()*16+256;
    if(cache && charge<=8*1024*1024) {
        if(cacheBytes+charge>8*1024*1024) {cache->clear();cacheBytes=0;}
        cache->emplace(std::pair{db,hash},Cached{value,bytes.size()});cacheBytes+=charge;
    }
    return value;
}
Json saveJson(sqlite3* db,const Json& value) {
    const auto text=value.dump();
    return storeSharedPayload(db,"@metadata",std::span(reinterpret_cast<const uint8_t*>(text.data()),text.size()));
}
}
MetadataCacheScope::MetadataCacheScope():owner_(!cache) {if(owner_){cache=std::make_unique<std::map<std::pair<sqlite3*,std::string>,Cached>>();cacheBytes=0;}}
MetadataCacheScope::~MetadataCacheScope(){if(owner_)cache.reset();}
MetadataSnapshot::MetadataSnapshot(sqlite3* db,const SharedJson& root):db_(db),buckets_(root.is_null()?SharedJson::object():load(db,root.get<std::string>())) {
    if(!buckets_.is_object())throw std::runtime_error("invalid metadata snapshot");
}
SharedJson MetadataSnapshot::bucket(int64_t index) const {
    if(const auto it=dirty_.find(index);it!=dirty_.end())return it->second;
    const auto key=std::to_string(index);
    auto result=buckets_.contains(key)?load(db_,buckets_.at(key)):SharedJson::object();
    if(!result.is_object())throw std::runtime_error("invalid metadata bucket");
    return result;
}
SharedJson MetadataSnapshot::get(int64_t id) const {
    if(geometryWork)++geometryWork->identityRows;
    if(id<0)return nullptr;
    return bucket(id/128).value(std::to_string(id%128),SharedJson());
}
void MetadataSnapshot::put(int64_t id,const SharedJson& value) {
    if(id<0)throw std::invalid_argument("negative metadata ID");
    auto [it,inserted]=dirty_.try_emplace(id/128);
    if(inserted) {
        const auto key=std::to_string(id/128);
        it->second=buckets_.contains(key)?load(db_,buckets_.at(key)):SharedJson::object();
    }
    if(value.is_null())it->second.erase(std::to_string(id%128));else it->second[std::to_string(id%128)]=value;
}
void MetadataSnapshot::entries(const std::function<void(int64_t,const SharedJson&)>& visit) const {
    std::set<int64_t> indices;
    for(const auto& [key,value]:buckets_.items())indices.insert(std::stoll(key));
    for(const auto& [index,value]:dirty_)indices.insert(index);
    for(auto index:indices) {
        const auto values=bucket(index);
        for(const auto& [slot,value]:values.items()) {
            const auto offset=std::stoll(slot);
            if(index<0||index>INT64_MAX/128||offset<0||offset>127)throw std::runtime_error("invalid metadata key");
            visit(index*128+offset,value);
        }
    }
}
void MetadataSnapshot::mergeAllocated(const MetadataSnapshot& other) {
    other.entries([&](int64_t id,const auto& value){if(value!=true)throw std::runtime_error("invalid allocated ID");put(id,true);});
}
SharedJson MetadataSnapshot::save() {
    for(const auto& [index,value]:dirty_) {
        const auto key=std::to_string(index);
        if(value.empty())buckets_.erase(key);else buckets_[key]=saveJson(db_,value);
    }
    dirty_.clear();return buckets_.empty()?SharedJson():saveJson(db_,buckets_);
}
SharedJson identityGet(sqlite3* db,const SharedJson& root,int64_t id) {return MetadataSnapshot(db,root).get(id);}
bool annotationDirectional(const SharedJson& value) {
    if(value.is_object())for(const auto& [key,item]:value.items())
        if(key=="segment_to_next"||key.find("direction")!=std::string::npos||key.find("tangent")!=std::string::npos||annotationDirectional(item))return true;
    if(value.is_array())for(const auto& item:value)if(annotationDirectional(item))return true;
    return false;
}
SharedJson packAnnotation(sqlite3* db,const SharedJson& value) {
    const auto text=value.dump();
    if(text.size()<=1024)return SharedJson{{"inline",value}};
    auto chunks=SharedJson::array();
    for(size_t offset=0;offset<text.size();offset+=65536)
        chunks.push_back(storeSharedPayload(db,"@annotations",std::span(reinterpret_cast<const uint8_t*>(text.data()+offset),std::min<size_t>(65536,text.size()-offset))));
    return {{"annotation_chunks",chunks},{"byte_length",text.size()}};
}
SharedJson unpackAnnotation(sqlite3* db,const SharedJson& value) {
    if(value.contains("inline"))return value.at("inline");
    std::vector<uint8_t> bytes;
    for(const auto& hash:value.at("annotation_chunks")) {const auto part=readSharedPayload(db,"@annotations",hash);bytes.insert(bytes.end(),part.begin(),part.end());}
    if(bytes.size()!=value.at("byte_length").get<size_t>())throw std::runtime_error("invalid annotation length");
    return SharedJson::parse(bytes);
}
void transferMetadata(sqlite3* source,sqlite3* destination,const SharedJson& root) {
    if(root.is_null())return;
    const auto transfer=[&](const std::string& hash){const auto bytes=readSharedPayload(source,"@metadata",hash);storeSharedPayload(destination,"@metadata",bytes);};
    const auto buckets=load(source,root);
    for(const auto& hash:buckets) {
        const auto bucket=load(source,hash);
        for(const auto& value:bucket) {
            const auto& annotation=value.is_object()&&value.contains("annotation")?value.at("annotation"):value;
            if(annotation.is_object()&&annotation.contains("annotation_chunks"))for(const auto& chunk:annotation.at("annotation_chunks"))transfer(chunk);
        }
        transfer(hash);
    }
    transfer(root);
}
}
