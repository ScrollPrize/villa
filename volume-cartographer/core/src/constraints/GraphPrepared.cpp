#include "GraphPrepared.hpp"

namespace vc::constraints {
namespace {
std::shared_ptr<const PreparedGraph> prepareGraphImpl(const FitResult& result,const GraphEdit* edit,const std::atomic_bool* cancel){
    detail::cancelled(cancel);
    if(!result.graph.windingLayout.is_null())throw std::runtime_error("Imported winding layouts cannot be edited");
    if(!result.valid||!result.surface)throw std::runtime_error("Only a validated graph can be prepared for attachment");
    auto prepared=std::make_shared<PreparedGraph>();prepared->accepted=result;
    const auto& graph=result.graph;
    const auto* previous=edit?edit->base.get():nullptr;
    if(edit&&!previous)throw std::runtime_error("Graph edit has no committed revision");
    if(previous){
        const auto& a=graph.settings;const auto& b=previous->accepted.graph.settings;
        if(a.spacing!=b.spacing||a.radius!=b.radius||a.stiffness!=b.stiffness||a.fiberScale!=b.fiberScale||
           a.patchScale!=b.patchScale||a.normalScale!=b.normalScale||a.normalManifest!=b.normalManifest)
            return prepareGraphImpl(result,nullptr,cancel);
    }
    if(previous){
        for(const auto& [id,p]:previous->accepted.graph.nodes)
            if(!graph.nodes.contains(id)||graph.nodes.at(id).source!=p.source)throw std::runtime_error("Prepared edit changed an accepted source");
    }

    for(const auto& [id,p]:graph.nodes){
        detail::cancelled(cancel);prepared->vertices+=p.uv.size();
        if(previous&&previous->accepted.graph.nodes.contains(id)&&p.uv.sharesStorage(previous->accepted.graph.nodes.at(id).uv))
            prepared->charts.emplace(id,previous->charts.at(id));
        else prepared->charts.emplace(id,p);
    }
    prepared->contacts.reserve(graph.links.size());
    std::vector<size_t> priorIds;
    if(previous){
        const std::set<size_t> removed(edit->removedLinks.begin(),edit->removedLinks.end());
        for(size_t i=0;i<previous->contacts.size();++i)if(!removed.contains(i))priorIds.push_back(i);
    }

    for(size_t i=0;i<graph.links.size();++i){
        detail::cancelled(cancel);const auto& link=graph.links[i];
        const bool retained=previous&&i<priorIds.size()&&detail::contactKey(link)==detail::contactKey(previous->accepted.graph.links[priorIds[i]]);
        const auto weights=retained?previous->contacts[priorIds[i]]:[&]{auto [a,b]=detail::linkWeights(graph,link);return detail::PreparedContact{a,b};}();
        prepared->contacts.push_back(weights);
        if(retained&&graph.nodes.at(link.a.node).uv.sharesStorage(previous->accepted.graph.nodes.at(link.a.node).uv)&&
           graph.nodes.at(link.b.node).uv.sharesStorage(previous->accepted.graph.nodes.at(link.b.node).uv))prepared->squaredErrors.push_back(previous->squaredErrors[priorIds[i]]);
        else{
            const auto error=detail::interpolate(graph.nodes.at(link.a.node).uv,weights.a)-detail::interpolate(graph.nodes.at(link.b.node).uv,weights.b);
            prepared->squaredErrors.push_back(error.dot(error));
        }
        prepared->incident[link.a.node].push_back(i);
        if(link.a.node!=link.b.node)prepared->incident[link.b.node].push_back(i);
        prepared->linked.insert(std::minmax(link.a.node,link.b.node));
        if(detail::horizontalVerticalLink(graph,link).first)prepared->junctionLinks.push_back(i);
    }
    bool sameJunctions=previous&&prepared->junctionLinks.size()==previous->junctionLinks.size();
    if(sameJunctions)for(size_t i=0;i<prepared->junctionLinks.size();++i)
        sameJunctions&=detail::contactKey(graph.links[prepared->junctionLinks[i]])==detail::contactKey(previous->accepted.graph.links[previous->junctionLinks[i]]);
    if(sameJunctions)prepared->junctions=previous->junctions;
    else{
        Graph junctionGraph;junctionGraph.settings=graph.settings;junctionGraph.nodes=graph.nodes;
        for(size_t i:prepared->junctionLinks)junctionGraph.links.push_back(graph.links[i]);
        prepared->junctions=std::make_shared<detail::FiberJunctions>(junctionGraph);
    }
    for(const auto& [id,p]:graph.nodes){
        detail::cancelled(cancel);auto& values=prepared->chartStretches[id];const auto& xyz=prepared->junctions->points(*p.source);
        if(previous&&previous->accepted.graph.nodes.contains(id)&&p.uv.sharesStorage(previous->accepted.graph.nodes.at(id).uv)&&
           &xyz==&previous->junctions->points(*p.source)){values=previous->chartStretches.at(id);continue;}
        for(const auto& t:p.source->triangles)for(int i=0;i<3;++i){
            const double dist=cv::norm(xyz[t[(i+1)%3]]-xyz[t[i]]),flat=cv::norm(p.uv[t[(i+1)%3]]-p.uv[t[i]]);
            if(dist>1e-8&&flat>1e-8)values.push_back(std::max(flat/dist,dist/flat));
        }
    }
    const auto& grid=*result.surface->rawPointsPtr();const double spacing=graph.settings.spacing;
    const auto lo=detail::tileAt(result.origin,spacing),hi=detail::tileAt(result.origin+cv::Vec2d(grid.cols-2,grid.rows-2)*spacing,spacing);
    for(int y=lo.second;y<=hi.second;++y)for(int x=lo.first;x<=hi.first;++x){
        detail::cancelled(cancel);const detail::TileKey key{x,y};
        const detail::ChartRegion tile{{x*64*spacing,y*64*spacing},{(x+1)*64*spacing,(y+1)*64*spacing}};
        if(previous&&result.changedSurfaceBounds&&previous->surfaceTiles.contains(key)&&
           !tile.intersects(detail::ChartRegion{(*result.changedSurfaceBounds)[0],(*result.changedSurfaceBounds)[1]}.padded(spacing)))
            prepared->surfaceTiles.emplace(key,previous->surfaceTiles.at(key));
        else prepared->surfaceTiles.emplace(key,detail::surfaceTileStats(grid,result.origin,spacing,key));
    }
    return prepared;
}
} // namespace
std::shared_ptr<const PreparedGraph> prepareGraph(const FitResult& result,const std::atomic_bool* cancel){return prepareGraphImpl(result,nullptr,cancel);}
std::shared_ptr<const PreparedGraph> prepareGraph(const FitResult& result,const GraphEdit& edit,const std::atomic_bool* cancel){return prepareGraphImpl(result,&edit,cancel);}
const FitResult& preparedResult(const PreparedGraph& graph){return graph.accepted;}
std::set<std::pair<std::string,std::string>> detail::editedPairs(const GraphEdit& edit){
    auto pairs=edit.base->linked;const auto& graph=edit.base->accepted.graph;
    const std::set<size_t> removed(edit.removedLinks.begin(),edit.removedLinks.end());
    for(size_t i:removed){
        const auto& link=graph.links.at(i);const auto pair=std::minmax(link.a.node,link.b.node);
        bool retained=false;
        for(size_t j:edit.base->incident.at(link.a.node))if(!removed.contains(j)){
            const auto& other=graph.links[j];if(std::minmax(other.a.node,other.b.node)==pair){retained=true;break;}
        }
        if(!retained)pairs.erase(pair);
    }
    for(const auto& link:edit.addedLinks)pairs.insert(std::minmax(link.a.node,link.b.node));
    return pairs;
}
Graph materialize(const GraphEdit& edit){
    if(!edit.base)throw std::runtime_error("Graph edit has no committed revision");
    Graph graph=edit.base->accepted.graph;
    for(const auto& [id,p]:edit.addedNodes){
        if(!p.source||id!=p.source->id||!graph.nodes.emplace(id,p).second)throw std::runtime_error("Graph edit replaces an accepted source");
    }
    std::set<size_t> removed(edit.removedLinks.begin(),edit.removedLinks.end());
    if(!removed.empty()){
        if(*removed.rbegin()>=graph.links.size())throw std::runtime_error("Removed contact is outside the committed revision");
        SharedVector<Link> retained;retained.reserve(graph.links.size()-removed.size()+edit.addedLinks.size());
        for(size_t i=0;i<graph.links.size();++i)if(!removed.contains(i))retained.push_back(std::as_const(graph.links)[i]);
        graph.links=std::move(retained);
    }
    if(!edit.addedLinks.empty())graph.links.insert(graph.links.cend(),edit.addedLinks.begin(),edit.addedLinks.end());
    return graph;
}
GraphEdit graphEdit(std::shared_ptr<const PreparedGraph> base,const Graph& draft){
    if(!base)throw std::runtime_error("Graph edit has no committed revision");
    GraphEdit edit;edit.base=std::move(base);const auto& accepted=edit.base->accepted.graph;
    // Settings changes and node deletion are explicit global operations.
    if(draft.seed!=accepted.seed)throw std::runtime_error("Changing the seed requires a full refit");
    const auto& a=draft.settings;const auto& b=accepted.settings;
    if(a.spacing!=b.spacing||a.radius!=b.radius||a.stiffness!=b.stiffness||a.fiberScale!=b.fiberScale||
       a.patchScale!=b.patchScale||a.normalScale!=b.normalScale||a.normalManifest!=b.normalManifest)
        throw std::runtime_error("Changing fit settings requires a full refit");
    for(const auto& [id,p]:accepted.nodes){
        auto found=draft.nodes.find(id);
        if(found==draft.nodes.end()||found->second.source!=p.source)throw std::runtime_error("Replacing an accepted source requires a full refit");
    }
    for(const auto& [id,p]:draft.nodes)if(!accepted.nodes.contains(id))edit.addedNodes.emplace(id,p);
    if(!draft.links.sharesStorage(accepted.links)){
        // This compatibility adapter is for manual snapshot edits. Ordinary
        // additions use proposeEdit and never visit the old contact list.
        std::map<Link,std::vector<size_t>,detail::ContactLess> previous;
        for(size_t i=0;i<accepted.links.size();++i)previous[accepted.links[i]].push_back(i);
        for(const auto& link:draft.links){
            auto it=previous.find(link);
            if(it==previous.end()||it->second.empty())edit.addedLinks.push_back(link);
            else it->second.pop_back();
        }
        for(const auto& [link,indices]:previous)edit.removedLinks.insert(edit.removedLinks.end(),indices.begin(),indices.end());
        std::sort(edit.removedLinks.begin(),edit.removedLinks.end());
    }
    return edit;
}
void addKeypoint(GraphEdit& edit,Link link){
    if(!edit.base)throw std::runtime_error("Graph edit has no committed revision");
    link.keypoint=true;const auto pair=std::minmax(link.a.node,link.b.node);
    if(auto it=edit.base->incident.find(link.a.node);it!=edit.base->incident.end())for(size_t i:it->second){
        const auto& old=edit.base->accepted.graph.links[i];
        if(!old.keypoint&&std::minmax(old.a.node,old.b.node)==pair)edit.removedLinks.push_back(i);
    }
    std::sort(edit.removedLinks.begin(),edit.removedLinks.end());
    edit.removedLinks.erase(std::unique(edit.removedLinks.begin(),edit.removedLinks.end()),edit.removedLinks.end());
    std::erase_if(edit.addedLinks,[&](const Link& old){return !old.keypoint&&std::minmax(old.a.node,old.b.node)==pair;});
    edit.addedLinks.push_back(std::move(link));
}
void removeKeypoint(GraphEdit& edit,const Link& link){
    if(!edit.base||!link.keypoint)throw std::runtime_error("Not a keypoint in this graph edit");
    const auto same=[&](const Link& other){return detail::contactKey(link)==detail::contactKey(other);};
    auto added=std::find_if(edit.addedLinks.begin(),edit.addedLinks.end(),same);
    if(added!=edit.addedLinks.end()){edit.addedLinks.erase(added);return;}
    if(auto it=edit.base->incident.find(link.a.node);it!=edit.base->incident.end())for(size_t i:it->second)
        if(same(edit.base->accepted.graph.links[i])&&std::find(edit.removedLinks.begin(),edit.removedLinks.end(),i)==edit.removedLinks.end()){
            edit.removedLinks.push_back(i);return;
        }
    throw std::runtime_error("Keypoint is not present in this graph edit");
}
}
