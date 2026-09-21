#pragma once

// Research export for the sequential, human-seeded review workflow. The target
// decision's status/alignment never selects its fit. Only earlier accepted
// associations can supply context; future decisions are excluded.
namespace fiber_review_evidence {
using namespace vc::constraints;
using Json = nlohmann::json;

inline Json summary(std::vector<double> values) {
    Json result;
    result["count"] = values.size();
    if(values.empty())return result;
    double square=0;for(double v:values)square+=v*v;
    result["rms"]=std::sqrt(square/values.size());
    std::sort(values.begin(),values.end());
    for(const auto& [name,q]:std::vector<std::pair<std::string,double>>{{"q50",.5},{"q90",.9},{"q99",.99},{"max",1.}})
        result[name]=values[size_t(q*(values.size()-1))];
    for(double tolerance:{.5,1.,2.,3.,5.})
        result["within_"+std::to_string(tolerance)]=double(std::count_if(values.begin(),values.end(),[&](double v){return v<tolerance;}))/values.size();
    return result;
}

inline Json measurements(const FitResult& fit,const GraphEdit& edit,const std::string& patch) {
    Json result={{"valid",fit.valid},{"message",fit.message},{"rms",fit.correspondenceRms},{"stretch_p95",fit.stretchP95},
                 {"chart_vertices",fit.localChartVertices}};
    if(!fit.valid)return result;
    std::vector<double> fiber,overlap,edgeRatios;
    for(const auto& link:edit.addedLinks)
        fiber.push_back(cv::norm(placedLocation(fit.graph,link.a)-placedLocation(fit.graph,link.b)));
    for(const auto& target:edit.fixedTargets)
        overlap.push_back(cv::norm(placedLocation(fit.graph,target.point)-target.uv));
    const auto& placed=fit.graph.nodes.at(patch);
    for(const auto& tri:placed.source->triangles)for(int i=0;i<3;++i) {
        const int a=tri[i],b=tri[(i+1)%3];
        const double original=cv::norm(placed.source->xyz[a]-placed.source->xyz[b]);
        const double chart=cv::norm(placed.uv[a]-placed.uv[b]);
        if(original>1e-8&&chart>1e-8)edgeRatios.push_back(std::max(original/chart,chart/original));
    }
    result["fiber_error"]=summary(std::move(fiber));
    result["overlap_error"]=summary(std::move(overlap));
    result["chart_edge_ratio"]=summary(std::move(edgeRatios));
    return result;
}

inline Json candidateEvidence(const Catalog& catalog,std::shared_ptr<const PreparedGraph> prepared,
    const ReviewFiber& fiber,const std::string& patch,const ReviewFiber& earlier,
    const vc::lasagna::NormalSampler& normals,Graph* acceptedGraph=nullptr) {
    const auto& base=preparedResult(*prepared).graph;
    Json attempts=Json::array();
    for(size_t placement=0;placement<fiber.proposals.size();++placement) {
        const auto& proposal=fiber.proposals[placement];if(proposal.id!=patch)continue;
        Json attempt={{"proposal_index",placement},{"fiber_contacts",proposal.contacts.size()}};
        bool acceptedFit=false;
        try {
            const auto soloEdit=proposeEdit(catalog,prepared,proposal,false,normals,nullptr,ContactPolicy::FiberReview);
            auto edit=soloEdit;
            addReviewOverlapTargets(edit,earlier);
            attempt["overlap_targets"]=edit.fixedTargets.size();
            std::set<std::pair<double,double>> points;
            for(const auto& t:edit.fixedTargets)points.emplace(t.point.grid[0],t.point.grid[1]);
            attempt["overlap_unique_points"]=points.size();
            auto fitStart=std::chrono::steady_clock::now();
            const auto solo=fitEdit(soloEdit,nullptr,{},FitPolicy::FiberReview);
            attempt["solo_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-fitStart).count();
            // Evaluate all shared measurements, including how the solo
            // fit agrees with anchors it did not optimize against.
            attempt["solo"]=measurements(solo,edit,patch);
            fitStart=std::chrono::steady_clock::now();
            const auto anchored=edit.fixedTargets.empty()?solo:fitEdit(edit,nullptr,{},FitPolicy::FiberReview);
            attempt["anchor_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-fitStart).count();
            attempt["anchor"]=measurements(anchored,edit,patch);
            for(const auto* fit:{&solo,&anchored})
                if(fit->graph.nodes.at(base.seed).uv!=base.nodes.at(base.seed).uv)
                    throw std::runtime_error("Evidence fit moved the fixed fiber");
            if(solo.valid&&anchored.valid) {
                std::vector<double> moved;
                const auto& a=solo.graph.nodes.at(patch).uv;
                const auto& b=anchored.graph.nodes.at(patch).uv;
                for(size_t i=0;i<a.size();++i)moved.push_back(cv::norm(a[i]-b[i]));
                attempt["context_shift"]=summary(std::move(moved));
            }
            acceptedFit=anchored.valid;
            if(acceptedFit&&acceptedGraph)*acceptedGraph=anchored.graph;
        } catch(const std::exception& error) { attempt["error"]=error.what(); }
        attempts.push_back(std::move(attempt));
        // Same default-first policy for all labels. Failed placements
        // remain visible in the export, never become changed labels.
        if(acceptedFit)break;
    }
    return attempts;
}

inline Catalog reviewCandidateCatalog(const FiberReview& review) {
    Catalog catalog;std::set<std::string> patches;
    for(const auto& [id,fiber]:review.fibers)for(const auto& c:fiber.proposals)patches.insert(c.id);
    for(const auto& id:patches)catalog.patches.push_back({id,review.settings.patchFolder/id.substr(6),{}, {}});
    return catalog;
}

inline int exportAnchors(int argc,char** argv) {
    if(std::filesystem::weakly_canonical(argv[2])==std::filesystem::weakly_canonical(argv[3]))
        throw std::runtime_error("Evidence output must not overwrite the review");
    const auto review=loadReview(argv[2]);
    const size_t limit=argc>4?std::stoull(argv[4]):0;
    std::ofstream output(argv[3]);if(!output)throw std::runtime_error("Cannot open evidence output");
    vc::lasagna::LasagnaDatasetOpenOptions options;options.workingToBaseScale=review.settings.normalScale;
    const auto dataset=vc::lasagna::LasagnaDataset::openLocation(review.settings.normalManifest,options);
    vc::lasagna::LasagnaNormalSampler normals(dataset);
    const auto catalog=reviewCandidateCatalog(review);
    size_t count=0;
    for(const auto& [fid,fiber]:review.fibers) {
        if(fiber.proposals.empty())continue;
        auto graph=fiber.base;graph.settings=review.settings;
        const auto base=renderPlacement(graph);
        if(!base.valid)throw std::runtime_error("Invalid saved fiber: "+base.message);
        const auto prepared=prepareGraph(base);
        std::set<std::string> visited;
        for(size_t index=0;index<fiber.proposals.size();++index) {
            const auto& candidate=fiber.proposals[index];
            if(!visited.insert(candidate.id).second)continue;
            const auto tick=std::chrono::steady_clock::now();
            const auto own=fiber.decisions.find(candidate.id);
            const auto order=own==fiber.decisions.end()?review.nextOrder:own->second.order;
            ReviewFiber earlier;Json references=Json::array();
            for(const auto& [id,d]:fiber.decisions)
                if(id!=candidate.id&&d.status==ReviewStatus::Accepted&&d.order<order) {
                    earlier.decisions.emplace(id,d);references.push_back({{"patch",id},{"order",d.order}});
                }
            Json row={{"fiber",fid},{"patch",candidate.id},{"review_revision",review.sessionRevision},
                      {"decision_order",order},{"references",references},{"attempts",Json::array()}};
            row["attempts"]=candidateEvidence(catalog,prepared,fiber,candidate.id,earlier,normals);
            row["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count();
            output<<row.dump()<<'\n';output.flush();if(!output)throw std::runtime_error("Failed writing evidence");
            std::cout<<"candidates="<<++count<<" patch="<<candidate.id<<" anchors="<<earlier.decisions.size()
                     <<" seconds="<<row["seconds"]<<std::endl;
            if(limit&&count>=limit)return 0;
        }
    }
    return 0;
}
struct OverlapGroups {
    std::map<std::string,Placement> nodes;
    std::map<std::string,std::vector<Link>> fiberLinks;
    std::map<std::pair<std::string,std::string>,std::vector<Link>> edges;

    Json gather(const Catalog& catalog,std::shared_ptr<const PreparedGraph> prepared,
                const ReviewFiber& fiber,const vc::lasagna::NormalSampler& normals) {
        nodes.clear();fiberLinks.clear();edges.clear();
        Json failures=Json::array(),pairs=Json::array();
        for(const auto& candidate:fiber.proposals)if(!nodes.contains(candidate.id)) {
            try {
                const auto edit=proposeEdit(catalog,prepared,candidate,false,normals,nullptr,ContactPolicy::FiberReview);
                auto placement=edit.addedNodes.at(candidate.id);
                // Intrinsic patch UV, independent of its proposed fiber pose.
                placement.uv=placement.source->initialUv;
                const auto matches=detail::patchOverlapContacts(*placement.source,nodes,
                    2*preparedResult(*prepared).graph.settings.radius,nullptr,true);
                for(const auto& link:matches)edges[{link.a.node,link.b.node}].push_back(link);
                nodes.emplace(candidate.id,std::move(placement));fiberLinks[candidate.id]=edit.addedLinks;
            }catch(const std::exception& e){failures.push_back({{"patch",candidate.id},{"error",e.what()}});}
        }
        for(const auto& [key,links]:edges) {
            Json samples=Json::array();
            const auto& a=nodes.at(key.first);const auto& b=nodes.at(key.second);
            for(const auto& l:links) {
                const auto uv=detail::interpolate(a.uv,detail::weights(*a.source,l.a));
                const auto x=sourceLocation(*a.source,l.a.grid),y=sourceLocation(*b.source,l.b.grid);
                samples.push_back({uv[0],uv[1],x[0],x[1],x[2],y[0],y[1],y[2]});
            }
            pairs.push_back({{"a",key.first},{"b",key.second},{"samples",std::move(samples)}});
        }
        Json ids=Json::array();for(const auto& [id,p]:nodes)ids.push_back(id);
        return {{"nodes",ids},{"edges",std::move(pairs)},{"failures",failures}};
    }

    Json fitGroup(std::shared_ptr<const PreparedGraph> prepared,const Json& request) const {
        const auto& base=preparedResult(*prepared).graph;
        Graph group;group.settings=base.settings;group.seed=request.at("seed");
        for(const auto& id:request.at("patches"))group.nodes.emplace(id.get<std::string>(),nodes.at(id.get<std::string>()));
        if(!group.nodes.contains(group.seed))throw std::runtime_error("Group seed is missing");
        for(const auto& edge:request.at("edges")) {
            const auto key=std::make_pair(edge[0].get<std::string>(),edge[1].get<std::string>());
            if(!group.nodes.contains(key.first)||!group.nodes.contains(key.second))throw std::runtime_error("Group edge is outside component");
            const auto& links=edges.at(key);
            const size_t limit=request.value("max_edge_contacts",size_t(0));
            if(limit&&links.size()>limit) {
                if(limit<2)throw std::runtime_error("Edge sampling needs at least two contacts");
                for(size_t i=0;i<limit;++i)group.links.push_back(links[i*(links.size()-1)/(limit-1)]);
            } else for(const auto& link:links)group.links.push_back(link);
        }
        auto tick=std::chrono::steady_clock::now();
        FitResult intrinsic;
        if(group.nodes.size()==1){intrinsic.graph=group;intrinsic.valid=true;intrinsic.message="Singleton intrinsic chart";}
        else intrinsic=fitSourceCharts(group,nullptr,{},request.value("intrinsic_rigid",false));
        Json result={{"intrinsic_valid",intrinsic.valid},{"intrinsic_message",intrinsic.message},
                     {"intrinsic_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count()},
                     {"patches",Json::object()}};
        if(!intrinsic.valid)return result;
        GraphEdit edit;edit.base=prepared;edit.addedNodes=intrinsic.graph.nodes;
        edit.addedLinks=group.links.read();
        for(const auto& [id,p]:group.nodes)
            edit.addedLinks.insert(edit.addedLinks.end(),fiberLinks.at(id).begin(),fiberLinks.at(id).end());
        tick=std::chrono::steady_clock::now();
        const bool rigidAttachment=request.value("attachment_rigid",false);
        FitResult attached;
        if(rigidAttachment) {
            Graph linked=base;linked.nodes.insert(edit.addedNodes.begin(),edit.addedNodes.end());
            linked.links=edit.addedLinks;
            attached=fitSourceCharts(std::move(linked),nullptr,{},true);
        } else attached=fitEdit(edit,nullptr,{},FitPolicy::FiberReview);
        result["attachment_rigid"]=rigidAttachment;
        result["fiber_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count();
        result["valid"]=attached.valid;result["message"]=attached.message;
        if(attached.graph.nodes.at(base.seed).uv!=base.nodes.at(base.seed).uv)
            throw std::runtime_error("Group attachment moved the fixed fiber");
        for(const auto& [id,p]:group.nodes) {
            GraphEdit perPatch;perPatch.addedLinks=fiberLinks.at(id);
            auto individual=attached;
            // A multi-source display collision is not a failed chart solve.
            // Validate each actual source layer as the review UI displays it.
            if(rigidAttachment||(!attached.valid&&attached.message.find("Unlinked patches collide")!=std::string::npos)) {
                Graph layer;layer.settings=base.settings;layer.seed=id;
                layer.nodes.emplace(id,attached.graph.nodes.at(id));
                const auto rendered=renderSourceLayer(layer,false);
                individual.valid=rendered.valid;individual.message=rendered.message;
            }
            auto measured=measurements(individual,perPatch,id);
            std::vector<double> before,after;
            for(const auto& link:group.links)if(link.a.node==id||link.b.node==id) {
                before.push_back(cv::norm(placedLocation(intrinsic.graph,link.a)-placedLocation(intrinsic.graph,link.b)));
                if(individual.valid)after.push_back(cv::norm(placedLocation(attached.graph,link.a)-placedLocation(attached.graph,link.b)));
            }
            measured["intrinsic_overlap_error"]=summary(std::move(before));
            measured["overlap_error"]=summary(std::move(after));
            result["patches"][id]=std::move(measured);
        }
        return result;
    }
};

// A resident, label-blind fitting worker for Python's sequential policy. Commit
// only records a generated fit in memory, never a decision in the saved review.
inline int anchorSession(const std::filesystem::path& path) {
    const auto review=loadReview(path);
    for(const auto& [id,fiber]:review.fibers)
        if(!fiber.decisions.empty()||fiber.pending)
            throw std::runtime_error("Autonomous replay requires a review stripped of decisions and pending alignments");
    vc::lasagna::LasagnaDatasetOpenOptions options;options.workingToBaseScale=review.settings.normalScale;
    const auto dataset=vc::lasagna::LasagnaDataset::openLocation(review.settings.normalManifest,options);
    vc::lasagna::LasagnaNormalSampler normals(dataset);
    const auto catalog=reviewCandidateCatalog(review);
    std::shared_ptr<const PreparedGraph> prepared;
    ReviewFiber trusted;
    std::string fiberId,lastPatch;
    Graph lastFit;
    OverlapGroups groups;
    auto reply=[](const Json& message){std::cout<<"ANCHOR_JSON "<<message.dump()<<std::endl;};
    reply({{"ready",true}});
    for(std::string line;std::getline(std::cin,line);) {
        const auto tick=std::chrono::steady_clock::now();
        Json result;
        try {
            const auto request=Json::parse(line);
            const auto command=request.at("command").get<std::string>();
            if(command=="reset") {
                prepared.reset();trusted=ReviewFiber{};lastFit=Graph{};lastPatch.clear();
                groups=OverlapGroups{};
                fiberId=request.at("fiber");
                auto graph=review.fibers.at(fiberId).base;graph.settings=review.settings;
                const auto base=renderPlacement(graph);
                if(!base.valid)throw std::runtime_error(base.message);
                prepared=prepareGraph(base);result={{"ok",true},{"fiber",fiberId}};
            } else if(command=="gather") {
                if(!prepared)throw std::runtime_error("Reset to a fiber first");
                result=groups.gather(catalog,prepared,review.fibers.at(fiberId),normals);
                result["ok"]=true;
            } else if(command=="group") {
                if(!prepared)throw std::runtime_error("Reset to a fiber first");
                result=groups.fitGroup(prepared,request);result["ok"]=true;
            } else if(command=="fit") {
                if(!prepared)throw std::runtime_error("Reset to a fiber first");
                lastFit=Graph{};lastPatch=request.at("patch");
                if(trusted.decisions.contains(lastPatch))throw std::runtime_error("Patch is already trusted");
                result={{"fiber",fiberId},{"patch",lastPatch},{"references",Json::array()}};
                for(const auto& [id,d]:trusted.decisions)result["references"].push_back(id);
                result["attempts"]=candidateEvidence(catalog,prepared,review.fibers.at(fiberId),lastPatch,trusted,normals,&lastFit);
                result["ok"]=true;
            } else if(command=="commit") {
                const auto patch=request.at("patch").get<std::string>();
                if(patch!=lastPatch||lastFit.nodes.empty())throw std::runtime_error("Commit requires the immediately preceding valid fit");
                trusted.decisions.emplace(patch,ReviewDecision{ReviewStatus::Accepted,trusted.decisions.size()+1,std::move(lastFit)});
                lastFit=Graph{};lastPatch.clear();
                result={{"ok",true},{"trusted",trusted.decisions.size()}};
            } else throw std::runtime_error("Unknown command");
        } catch(const std::exception& e) {
            lastFit=Graph{};lastPatch.clear();result={{"ok",false},{"error",e.what()}};
        }
        result["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count();
        reply(result);
    }
    return 0;
}
} // namespace fiber_review_evidence
