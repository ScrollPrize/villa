#include "vc_reader/registrations.hpp"
#include "vc_reader/catalog.hpp"
#include <map>
#include <set>
#include <deque>
#include <string_view>
namespace vcr::catalog {
const TransformCandidate& TransformResolution::resolved() const {
    if(status!=TransformStatus::Resolved)throw std::runtime_error(status==TransformStatus::Ambiguous?"ambiguous volume transform; select a registration":"no volume transform");
    return candidates.front();
}
void RegistrationGraph::add(Registration r) {
    if(r.id.empty()||r.source.empty()||r.target.empty()||r.source==r.target)throw std::invalid_argument("registration requires distinct source and target volume IDs");
    (void)r.matrix.inverse();
    for(const auto& old:registrations_)if(old.id==r.id)throw std::invalid_argument("duplicate registration ID: "+r.id);
    registrations_.push_back(std::move(r));
    std::sort(registrations_.begin(),registrations_.end(),[](const auto& a,const auto& b){return a.id<b.id;});
}
void RegistrationGraph::addSample(const OpenDataSample& s) {
    for(const auto& g:s.volumeTransforms)for(const auto& t:g.transforms) {
        if(g.fromVolumeId==t.toVolumeId)continue;
        const auto key="catalog:"+s.id+":"+g.fromVolumeId+":"+t.toVolumeId;
        add({key,t.matrix.json().dump(),g.fromVolumeId,t.toVolumeId,t.matrix,t.derivationPath});
    }
    for(const auto& v:s.volumes) {
        const auto sup=v.properties.find("superseded_by");
        if(sup!=v.properties.end()&&sup->is_object()&&sup->contains("volume_id")&&(*sup)["volume_id"].is_string()) {
            const auto target=(*sup)["volume_id"].get<std::string>();
            if(target!=v.id&&!target.empty())add({"superseded:"+s.id+":"+v.id,target,v.id,target,Affine{},"superseded volume"});
        }
    }
}
void RegistrationGraph::addCatalog(const OpenDataManifest& c){for(const auto& s:c.samples)addSample(s);}
TransformResolution RegistrationGraph::resolve(const std::string& source,const std::string& target,const std::optional<std::vector<std::string>>& choice) const {
    TransformResolution result{source,target};
    if(source.empty()||target.empty())return result;
    if(source==target){result.status=TransformStatus::Resolved;result.candidates.push_back({});return result;}
    struct Step {std::string target,key;Affine matrix;};
    std::map<std::string,std::vector<Step>> edges;
    for(const auto& r:registrations_) {
        const auto key=r.id+"@"+r.revision;
        edges[r.source].push_back({r.target,key,r.matrix});
        edges[r.target].push_back({r.source,"inverse:"+key,r.matrix.inverse()});
    }
    for(auto& [volume, steps]:edges)
        std::stable_sort(steps.begin(),steps.end(),[](const auto& a,const auto& b){
            return a.key.starts_with("inverse:") < b.key.starts_with("inverse:");
        });
    if(choice) {
        auto volume=source;Affine matrix;
        for(const auto& key:*choice) {
            const auto& candidates=edges[volume];
            const auto it=std::find_if(candidates.begin(),candidates.end(),[&](const auto& e){return e.key==key;});
            if(it==candidates.end())return result;
            matrix=it->matrix*matrix;volume=it->target;
        }
        if(volume==target){result.status=TransformStatus::Resolved;result.candidates.push_back({matrix,matrix.inverse(),*choice});}
        return result;
    }
    // Breadth-first traversal retains distinct mappings, but never walks a
    // cycle. Catalogs contain closure edges, so equivalent paths coalesce.
    struct State {std::string volume;Affine matrix;std::vector<std::string> path,visited;};
    std::deque<State> todo{{source,{}, {},{source}}};
    std::map<std::string,std::vector<Affine>> seen;
    size_t work=0;
    while(!todo.empty()) {
        if(++work>100000)throw std::runtime_error("registration graph has too many conflicting paths");
        auto current=std::move(todo.front());todo.pop_front();
        for(const auto& e:edges[current.volume]) {
            if(std::find(current.visited.begin(),current.visited.end(),e.target)!=current.visited.end())continue;
            auto matrix=current.path.empty()?e.matrix:e.matrix*current.matrix;
            auto path=current.path;path.push_back(e.key);
            auto& mappings=seen[e.target];
            if(std::any_of(mappings.begin(),mappings.end(),[&](const auto& m){return m.equivalent(matrix);}))continue;
            mappings.push_back(matrix);
            if(e.target==target) {result.candidates.push_back({matrix,matrix.inverse(),std::move(path)});continue;}
            auto visited=current.visited;visited.push_back(e.target);
            todo.push_back({e.target,matrix,std::move(path),std::move(visited)});
        }
    }
    if(!result.candidates.empty())result.status=result.candidates.size()==1?TransformStatus::Resolved:TransformStatus::Ambiguous;
    return result;
}
std::vector<TransformResolution> RegistrationGraph::available(const std::string& source) const {
    std::set<std::string> volumes;for(const auto& r:registrations_){volumes.insert(r.source);volumes.insert(r.target);}
    std::vector<TransformResolution> result;
    for(const auto& target:volumes)if(target!=source){auto r=resolve(source,target);if(r.status!=TransformStatus::Unavailable)result.push_back(std::move(r));}
    return result;
}
std::vector<TransformResolution> RegistrationGraph::reaching(const std::string& target) const {
    std::set<std::string> volumes;for(const auto& r:registrations_){volumes.insert(r.source);volumes.insert(r.target);}
    std::vector<TransformResolution> result;
    for(const auto& source:volumes)if(source!=target){auto r=resolve(source,target);if(r.status!=TransformStatus::Unavailable)result.push_back(std::move(r));}
    return result;
}
std::string RegistrationGraph::describe(const std::vector<std::string>& path) const {
    std::string text;
    for(const auto& step:path) {
        constexpr std::string_view inversePrefix="inverse:";
        const bool inverse=step.starts_with(inversePrefix);
        const auto key=inverse?step.substr(inversePrefix.size()):step;
        std::string label=key;
        for(const auto& r:registrations_)if(r.id+"@"+r.revision==key) {
            label=(r.provenance.empty()?r.id:r.provenance)+" ("+r.source+" → "+r.target+")";
            break;
        }
        if(!text.empty())text+=", then ";
        text+=inverse?"inverse of "+label:label;
    }
    return text;
}
}
