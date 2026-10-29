#include "vc_object_store/objects/VolumeTransforms.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "internal/Database.hpp"
#include "internal/TransformedLength.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/Uuid.hpp"
#include <algorithm>
#include <limits>
#include <string_view>
namespace vc::objects {
std::optional<VolumeAffine> VolumeTransforms::toVolume(const nlohmann::json& metadata,const std::string& target,double targetScale) const {
    if(target.empty() || !std::isfinite(targetScale) || targetScale<=0)throw std::invalid_argument("invalid target volume or scale");
    const auto coordinates=geometryCoordinates(metadata.value("coordinates",nlohmann::json::object()));
    if(coordinates["volume_uuid"].is_null()||coordinates["scale_factor"].is_null())return std::nullopt;
    const auto resolution=resolve(coordinates["volume_uuid"].get<std::string>(),target);
    if(resolution.status!=vcr::catalog::TransformStatus::Resolved)return std::nullopt;
    return VolumeAffine::scale(targetScale)*resolution.resolved().forward*VolumeAffine::scale(1./coordinates["scale_factor"].get<double>());
}
std::optional<std::vector<std::string>> VolumeTransforms::chosen(const std::string& source,const std::string& target) const {
    if(const auto it=choices.find({source,target});it!=choices.end())return it->second;
    const auto reverse=choices.find({target,source});
    if(reverse==choices.end())return std::nullopt;
    // Walk the reverse pair's route backwards: each step inverted, last first.
    constexpr std::string_view inversePrefix="inverse:";
    std::vector<std::string> path;
    for(auto step=reverse->second.rbegin();step!=reverse->second.rend();++step)
        path.push_back(step->starts_with(inversePrefix)?step->substr(inversePrefix.size()):std::string(inversePrefix)+*step);
    return path;
}
std::vector<VolumeTransforms::Related> VolumeTransforms::related(const std::string& target) const {
    std::vector<Related> result;
    for(auto& routes:graph.reaching(target)) {
        Related entry;
        entry.volume=routes.source;
        entry.status=resolve(routes.source,target).status;
        entry.routes=std::move(routes.candidates);
        entry.chosen=chosen(entry.volume,target);
        result.push_back(std::move(entry));
    }
    return result;
}
std::optional<ObjectRecord> volumeTransformChoice(const ObjectStore& store,const Txn& txn,const std::string& source,const std::string& target) {
    for(const auto& existing:store.objectsByKind(txn,"volume_transform_choice")) {
        const auto c=nlohmann::json::parse(existing.meta).at("choice");
        if(c.at("source_volume")==source&&c.at("target_volume")==target)return existing;
    }
    return std::nullopt;
}
std::string registerVolumeTransform(ObjectStoreSession& session,const AttachmentIdentity& attachment,const std::string& source,
    const std::string& target,const VolumeAffine& matrix,const std::string& name) {
    const auto uuid=util::newUuid();
    session.act(attachment,{{uuid,std::nullopt}},[&](auto& store,auto& txn){store.registerVolumeTransform(txn,source,target,matrix,name,uuid);});
    return uuid;
}
std::string chooseVolumeTransform(ObjectStoreSession& session,const AttachmentIdentity& attachment,const std::string& source,
    const std::string& target,const std::vector<std::string>& path) {
    std::string uuid=util::newUuid();
    std::optional<std::string> revision;
    session.read(attachment,[&](const auto& store,const auto& txn,const auto&) {
        if(const auto existing=volumeTransformChoice(store,txn,source,target)) {
            uuid=existing->uuid;
            revision=store.currentObjectRevision(txn,uuid)->uuid;
        }
    });
    session.act(attachment,{{uuid,revision}},[&](auto& store,auto& txn){store.chooseVolumeTransform(txn,source,target,path,uuid);});
    return uuid;
}
std::array<double,6> transformBounds(const std::array<double,6>& b,const VolumeAffine& m) {
    std::array<double,6> out;
    for(int i=0;i<3;++i) {
        double low=m(i,3),high=low,magnitude=std::abs(low);
        for(int j=0;j<3;++j) {
            const auto a=m(i,j)*b[j],c=m(i,j)*b[j+3];
            low+=std::min(a,c);high+=std::max(a,c);magnitude+=std::max(std::abs(a),std::abs(c));
        }
        const auto margin=32*std::numeric_limits<double>::epsilon()*std::max(1.,magnitude);
        out[i]=low-margin;out[i+3]=high+margin;
        if(!std::isfinite(out[i])||!std::isfinite(out[i+3]))throw std::invalid_argument("transformed bounds are not finite");
    }
    return out;
}
VolumeTransforms ObjectStore::volumeTransforms(const Txn& txn) const {
    VolumeTransforms result;
    if(impl_->options.catalogTransforms)result.graph=*impl_->options.catalogTransforms;
    for(const auto& object:objectsByKind(txn,"volume_transform")) {
        const auto meta=nlohmann::json::parse(object.meta);
        const auto& value=meta.at("registration");
        const auto head=currentObjectRevision(txn,object.uuid);
        if(!head)throw std::runtime_error("commit new volume registrations before resolving views");
        result.graph.add({object.uuid,head->uuid,
            value.at("source_volume").get<std::string>(),value.at("target_volume").get<std::string>(),
            VolumeAffine::fromJson(value.at("matrix")),value.value("description",std::string{})});
    }
    for(const auto& object:objectsByKind(txn,"volume_transform_choice")) {
        const auto value=nlohmann::json::parse(object.meta).at("choice");
        const auto key=std::pair(value.at("source_volume").get<std::string>(),value.at("target_volume").get<std::string>());
        if(!result.choices.emplace(key,value.at("path").get<std::vector<std::string>>()).second)
            throw std::runtime_error("multiple active volume transform choices for "+key.first+" -> "+key.second);
    }
    return result;
}
std::vector<vcr::catalog::TransformResolution> ObjectStore::availableTransforms(const Txn& txn,const std::string& uuid) const {
    const auto object=objectByUuid(txn,uuid);
    if(!object)throw std::invalid_argument("object not found");
    const auto coordinates=geometryCoordinates(nlohmann::json::parse(object->meta).value("coordinates",nlohmann::json::object()));
    if(coordinates["volume_uuid"].is_null()||coordinates["scale_factor"].is_null())return {};
    const auto source=coordinates["volume_uuid"].get<std::string>();
    const auto context=volumeTransforms(txn);
    auto result=context.graph.available(source);
    for(auto& resolution:result)resolution=context.resolve(source,resolution.target);
    return result;
}
double ObjectStore::geometryLengthInVolume(const Txn& txn,const std::string& uuid,const std::string& target,double scale) const {
    const auto object=objectByUuid(txn,uuid);
    if(!object || (object->kind!="fiber"&&object->kind!="track"))throw std::invalid_argument("length requires fiber or track");
    const auto mapping=volumeTransforms(txn).toVolume(nlohmann::json::parse(object->meta),target,scale);
    if(!mapping)throw std::invalid_argument("volume mapping is unavailable or ambiguous");
    const auto head=currentGeometry(txn,uuid);
    if(!head)throw std::invalid_argument("object has no geometry");
    return detail::transformedLength(static_cast<sqlite3*>(connection(txn)),head->uuid,*mapping,object->kind);
}
std::string ObjectStore::registerVolumeTransform(WriteTxn& txn,const std::string& source,const std::string& target,
    const VolumeAffine& matrix,const std::string& name,const std::optional<std::string>& uuid) {
    vcr::catalog::RegistrationGraph validate;validate.add({"validate","",source,target,matrix,name});
    ObjectRecord record;
    if(uuid) {if(auto existing=objectByUuid(txn,*uuid)){record=*existing;if(record.kind!="volume_transform")throw std::invalid_argument("not a volume registration");}else record.uuid=*uuid;}
    record.kind="volume_transform";record.name=name;
    auto meta=nlohmann::json::parse(record.meta);
    meta["registration"]={{"source_volume",source},{"target_volume",target},{"matrix",matrix.json()},{"description",name}};
    record.meta=meta.dump();return upsertObject(txn,record);
}
std::string ObjectStore::chooseVolumeTransform(WriteTxn& txn,const std::string& source,const std::string& target,const std::vector<std::string>& path,
    const std::optional<std::string>& uuid) {
    auto transforms=volumeTransforms(txn);
    (void)transforms.graph.resolve(source,target,path).resolved();
    ObjectRecord record;record.kind="volume_transform_choice";record.name=source+" -> "+target;
    if(uuid)record.uuid=*uuid;
    if(auto existing=volumeTransformChoice(*this,txn,source,target))record=std::move(*existing);
    record.meta=nlohmann::json{{"choice",{{"source_volume",source},{"target_volume",target},{"path",path}}}}.dump();
    return upsertObject(txn,record);
}
}
