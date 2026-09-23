#include "vc/constraints/FiberReview.hpp"
#include "Quarantine.hpp"
#include "GraphInternal.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <tuple>

namespace vc::constraints {
namespace {
size_t removeReviewProposals(ReviewFiber& f,const std::set<std::string>& excluded) {
    const auto before=f.proposals.size();
    // Preserve traversal and the selected proposal when it still exists.
    int next=f.current;
    if(next>=0&&excluded.contains(f.proposals[next].id)) {
        next=-1;
        for(size_t step=1;step<=f.proposals.size();++step) {
            const auto i=(size_t(f.current)+step)%f.proposals.size();
            if(!excluded.contains(f.proposals[i].id)&&!f.decisions.contains(f.proposals[i].id)){next=int(i);break;}
        }
    }
    int kept=-1;std::vector<Candidate> proposals;
    for(size_t i=0;i<f.proposals.size();++i)if(!excluded.contains(f.proposals[i].id)) {
        if(int(i)==next)kept=int(proposals.size());proposals.push_back(std::move(f.proposals[i]));
    }
    if(f.pending)for(const auto& patch:excluded)if(f.pending->nodes.contains(patch)){f.pending.reset();break;}
    f.proposals=std::move(proposals);f.current=kept;
    return before-f.proposals.size();
}
}
size_t pruneUnavailableReviewProposals(ReviewFiber& f,const std::set<std::string>& available) {
    std::set<std::string> excluded;
    for(const auto& candidate:f.proposals)if(!available.contains(candidate.id)) {
        const auto decision=f.decisions.find(candidate.id);
        if(decision!=f.decisions.end()&&decision->second.status==ReviewStatus::Accepted)continue;
        if(f.pending&&f.pending->nodes.contains(candidate.id))continue;
        excluded.insert(candidate.id);
    }
    return excluded.empty()?0:removeReviewProposals(f,excluded);
}
void addReviewOverlapTargets(GraphEdit& edit,const ReviewFiber& fiber,const std::atomic_bool* cancel) {
    detail::cancelled(cancel);edit.fixedTargets.clear();
    if(!edit.base)throw std::runtime_error("Review proposal has no fixed fiber");
    const auto& base=preparedResult(*edit.base).graph;
    std::map<std::string,Placement> references;
    for(const auto& [id,decision]:fiber.decisions) {
        if(!fitted(decision)||edit.addedNodes.contains(id))continue;
        const auto& alignment=decision.alignment;
        if(alignment.seed!=base.seed||alignment.nodes.at(base.seed).uv!=base.nodes.at(base.seed).uv)
            throw std::runtime_error("Accepted patch belongs to a different fiber placement");
        references.emplace(id,alignment.nodes.at(id));
    }
    if(references.empty())return;
    for(const auto& [id,p]:edit.addedNodes)if(!p.source->fiber) {
        // Review tolerates imperfect sources: search twice the proximity
        // radius, reject crossing normals, then robustly weight chart errors.
        const auto contacts=detail::patchOverlapContacts(*p.source,references,2*base.settings.radius,cancel,true);
        for(const auto& link:contacts) {
            const auto& reference=references.at(link.a.node);
            edit.fixedTargets.push_back({link.b,detail::interpolate(reference.uv,detail::weights(*reference.source,link.a))});
        }
    }
}
size_t acceptedPatchCount(const ReviewFiber& fiber) {
    return std::count_if(fiber.decisions.begin(),fiber.decisions.end(),[](const auto& p){return p.second.status==ReviewStatus::Accepted;});
}
void orderReviewProposals(ReviewFiber& fiber) {
    if(fiber.base.nodes.empty())return;
    const auto& p=fiber.base.nodes.at(fiber.base.seed);
    const auto& n=*p.source;
    auto position=[&](const Candidate& c) {
        double first=std::numeric_limits<double>::infinity();
        for(const auto& link:c.contacts) {
            const auto& at=link.a.node==n.id?link.a:link.b;
            auto uv=placedLocation(fiber.base,at);
            // Chart Y is up; screen traversal starts at the top.
            double value=n.direction=='V'?-uv[1]:uv[0];
            first=std::min(first,fiber.reverse?-value:value);
        }
        return first;
    };
    std::stable_sort(fiber.proposals.begin(),fiber.proposals.end(),[&](const Candidate& a,const Candidate& b){
        return std::tuple{position(a),a.id}<std::tuple{position(b),b.id};
    });
    fiber.current=-1;
}
int nextReviewProposal(const ReviewFiber& fiber,int from,int delta) {
    const int size=int(fiber.proposals.size());
    if(from<0&&delta<0)from=0;
    for(int step=1;step<=size;++step) {
        const int i=((from+(delta<0?-step:step))%size+size)%size;
        if(!fiber.decisions.contains(fiber.proposals[i].id))return i;
    }
    return -1;
}
using Json=nlohmann::json;
Json reviewToJson(const FiberReview& review) {
    Json out={{"type","vc3d_fiber_review"},{"version",2},{"settings",settingsToJson(review.settings)},
        {"selected",review.selected},{"next_order",review.nextOrder},{"stride",review.stride},
        {"proposal_opacity",review.proposalOpacity},{"committed_opacity",review.committedOpacity},
        {"committed_tint",review.committedTint},
        {"fiber_line_opacity",review.fiberLineOpacity},{"fiber_line_width",review.fiberLineWidth},
        {"boundaries",review.boundaries},{"sources",Json::object()},{"fibers",Json::object()}};
    out["quarantined"]=Json::object();
    for(const auto& [id,p]:review.quarantined)out["quarantined"][id]={{"source",p.source.string()},{"destination",p.destination.string()}};
    auto graph=[&](const Graph& g)->Json {
        if(g.nodes.empty())return nullptr;
        auto j=graphToJson(g);
        for(auto& node:j["nodes"]) {
            auto id=node.at("id").get<std::string>();auto uv=node.at("uv");node.erase("uv");
            if(out["sources"].contains(id)&&out["sources"][id]!=node)throw std::runtime_error("Source changed within review: "+id);
            out["sources"][id]=node;node={{"id",id},{"uv",std::move(uv)}};
        }
        return j;
    };
    for(const auto& [id,f]:review.fibers) {
        Json j={{"path",f.path.string()},{"direction",std::string(1,f.direction)},{"base",graph(f.base)},
            {"reverse",f.reverse},{"discovered",f.discovered},{"current",f.current},{"contact_policy",f.contactPolicyVersion},
            {"decisions",Json::object()},{"proposals",Json::array()},{"pending",f.pending?graph(*f.pending):Json(nullptr)}};
        for(const auto& [patch,d]:f.decisions) {
            Json decision={{"status",d.status==ReviewStatus::Accepted?"accepted":d.status==ReviewStatus::Skipped?"skipped":"rejected"},{"order",d.order},{"alignment",graph(d.alignment)}};
            if(d.automatic)decision["automatic"]=true;
            if(!d.note.empty())decision["note"]=d.note;
            if(!d.keypoints.empty()){Json k=Json::array();for(const auto& l:d.keypoints)k.push_back({{"a",locationToJson(l.a)},{"b",locationToJson(l.b)}});decision["keypoints"]=std::move(k);}
            if(!d.measurements.is_null())decision["measurements"]=d.measurements;
            j["decisions"][patch]=std::move(decision);
        }
        for(const auto& c:f.proposals) {
            Json links=Json::array();for(const auto& l:c.contacts)links.push_back({{"a",locationToJson(l.a)},{"b",locationToJson(l.b)}});
            j["proposals"].push_back({{"id",c.id},{"distance",c.distance},{"contacts",std::move(links)}});
        }
        out["fibers"][id]=std::move(j);
    }
    return out;
}
FiberReview reviewFromJson(const Json& document) {
    const int version=document.at("version");
    if(document.at("type")!="vc3d_fiber_review"||(version!=1&&version!=2))throw std::runtime_error("Unsupported fiber review workspace");
    FiberReview review;
    review.sessionRevision=detail::workspaceRevision(document.dump());
    if(document.contains("quarantined"))for(const auto& [id,p]:document.at("quarantined").items())
        review.quarantined[id]={p.at("source").get<std::string>(),p.at("destination").get<std::string>()};
    std::map<std::string,std::shared_ptr<const Node>> sources;
    auto graph=[&](Json j) {
        if(j.is_null())return Graph{};
        for(auto& node:j["nodes"]) {auto id=node.at("id").get<std::string>();auto uv=node.at("uv");node=document.at("sources").at(id);node["uv"]=std::move(uv);}
        auto g=graphFromJson(j);
        for(auto& [id,p]:g.nodes) {auto [it,inserted]=sources.emplace(id,p.source);p.source=it->second;}
        return g;
    };
    review.settings=settingsFromJson(document.at("settings"));
    review.selected=document.at("selected");review.nextOrder=document.at("next_order");review.stride=document.at("stride");
    review.proposalOpacity=document.at("proposal_opacity");review.committedOpacity=document.at("committed_opacity");review.boundaries=document.at("boundaries");
    review.fiberLineOpacity=document.value("fiber_line_opacity",review.fiberLineOpacity);
    review.committedTint=document.value("committed_tint",review.committedTint);
    if(review.committedTint<0||review.committedTint>100)throw std::runtime_error("Invalid committed tint strength");
    review.fiberLineWidth=document.value("fiber_line_width",review.fiberLineWidth);
    if(review.fiberLineOpacity<0||review.fiberLineOpacity>100||!std::isfinite(review.fiberLineWidth)||review.fiberLineWidth<.5||review.fiberLineWidth>10)throw std::runtime_error("Invalid fiber line style");
    if(review.stride<1||review.proposalOpacity<0||review.proposalOpacity>100||review.committedOpacity<0||review.committedOpacity>100)throw std::runtime_error("Invalid review settings");
    for(const auto& [id,j]:document.at("fibers").items()) {
        ReviewFiber f;f.path=j.at("path").get<std::string>();auto direction=j.at("direction").get<std::string>();
        if(direction.size()!=1)throw std::runtime_error("Invalid review direction");f.direction=direction[0];
        f.base=graph(j.at("base"));f.reverse=j.at("reverse");f.discovered=j.at("discovered");f.current=j.at("current");
        if(!f.base.nodes.empty()&&(f.base.seed!=id||f.base.nodes.size()!=1||!f.base.nodes.at(id).source->fiber))throw std::runtime_error("Invalid review fiber base");
        for(const auto& [patch,d]:j.at("decisions").items()) {
            ReviewStatus status;
            if(version==1)status=d.at("accepted").get<bool>()?ReviewStatus::Accepted:ReviewStatus::Rejected;
            else {
                const auto value=d.at("status").get<std::string>();
                if(value=="accepted")status=ReviewStatus::Accepted;
                else if(value=="rejected")status=ReviewStatus::Rejected;
                else if(value=="skipped")status=ReviewStatus::Skipped;
                else throw std::runtime_error("Unknown fiber review decision status: "+value);
            }
            ReviewDecision decision{status,d.at("order"),graph(d.at("alignment"))};
            decision.automatic=d.value("automatic",false);decision.note=d.value("note",std::string{});
            if(d.contains("keypoints"))for(const auto& l:d.at("keypoints"))decision.keypoints.push_back({locationFromJson(l.at("a")),locationFromJson(l.at("b")),true});
            if(d.contains("measurements"))decision.measurements=d.at("measurements");
            if(status==ReviewStatus::Accepted&&(decision.alignment.seed!=id||decision.alignment.nodes.size()!=2||!decision.alignment.nodes.contains(patch)))throw std::runtime_error("Invalid accepted association");
            if(status==ReviewStatus::Accepted&&(f.base.nodes.empty()||decision.alignment.nodes.at(id).uv!=f.base.nodes.at(id).uv))throw std::runtime_error("Accepted association moved its fiber");
            f.decisions.emplace(patch,std::move(decision));
        }
        for(const auto& c:j.at("proposals")) {
            Candidate proposal;proposal.id=c.at("id");proposal.distance=c.at("distance");
            for(const auto& l:c.at("contacts")) {
                Link contact{locationFromJson(l.at("a")),locationFromJson(l.at("b"))};
                if(contact.a.node!=id||contact.b.node!=proposal.id)throw std::runtime_error("Invalid proposal association");
                if(!f.base.nodes.empty())(void)placedLocation(f.base,contact.a);
                proposal.contacts.push_back(contact);
            }
            if(proposal.contacts.empty())throw std::runtime_error("Empty review proposal");f.proposals.push_back(std::move(proposal));
        }
        if(f.current< -1||f.current>=int(f.proposals.size()))throw std::runtime_error("Invalid review position");
        if(!j.at("pending").is_null()) {
            f.pending=graph(j.at("pending"));
            if(f.current<0||f.pending->seed!=id||f.pending->nodes.size()!=2||!f.pending->nodes.contains(f.proposals[f.current].id))throw std::runtime_error("Pending alignment does not match review position");
        }
        f.contactPolicyVersion=j.value("contact_policy",0);
        if(f.contactPolicyVersion!=1) {
            // Rediscover legacy split proposals, retaining decisions and manual
            // pending work. Only cached automatic fits need replacing.
            f.discovered=false;
            if(f.pending&&std::none_of(f.pending->links.begin(),f.pending->links.end(),[](const auto& l){return l.keypoint;}))f.pending.reset();
        }
        review.fibers.emplace(id,std::move(f));
    }
    if(!review.selected.empty()&&!review.fibers.contains(review.selected))throw std::runtime_error("Selected fiber missing");
    return review;
}
namespace {
std::string statusName(ReviewStatus s){return s==ReviewStatus::Accepted?"accepted":s==ReviewStatus::Skipped?"skipped":"rejected";}
ReviewStatus statusFromName(const std::string& value) {
    if(value=="accepted")return ReviewStatus::Accepted;if(value=="rejected")return ReviewStatus::Rejected;if(value=="skipped")return ReviewStatus::Skipped;
    throw std::runtime_error("Unknown fiber review decision status: "+value);
}
std::string autoNote(const Json& e,bool linked) {
    char buffer[200];
    const double contact=e.value("contact",0.),ss=e.value("strip_same",0.),sa=e.value("strip_adjacent",0.),ls=e.value("linked_same",0.),la=e.value("linked_adjacent",0.);
    if(linked){if(e.value("anchor",false))std::snprintf(buffer,sizeof buffer,"Auto-linked anchor: %.0f vx fiber contact, strip agreement %.2f vs %.2f adjacent",contact,ss,sa);
        else std::snprintf(buffer,sizeof buffer,"Auto-linked in round %d: %.2f same-sheet vs %.2f adjacent overlap with linked patches",e.value("round",0),ls,la);}
    else if(ss>0||sa>0)std::snprintf(buffer,sizeof buffer,"Auto-rejected: %.0f vx contact but strip agreement %.2f does not clear %.2f adjacent",contact,ss,sa);
    else std::snprintf(buffer,sizeof buffer,"Auto-rejected: %.0f vx contact, %.2f same-sheet vs %.2f adjacent overlap with linked patches",contact,ls,la);
    return buffer;
}
Json keypointsToJson(const std::vector<Link>& links){Json k=Json::array();for(const auto& l:links)k.push_back({{"a",locationToJson(l.a)},{"b",locationToJson(l.b)}});return k;}
std::vector<Link> keypointsFromJson(const Json& k){std::vector<Link> out;if(k.is_array())for(const auto& l:k)out.push_back({locationFromJson(l.at("a")),locationFromJson(l.at("b")),true});return out;}
}
bool isLinksDocument(const Json& document){return document.value("type",std::string{})=="vc_fiber_patch_links";}
FiberReview reviewFromLinks(const Json& document) {
    if(!isLinksDocument(document))throw std::runtime_error("Not a vc_fiber_patch_links document");
    const int version=document.value("version",1);
    if(version!=1&&version!=2)throw std::runtime_error("Unsupported vc_fiber_patch_links version");
    FiberReview review;review.settings=settingsFromJson(document.at("settings"));review.stride=1;
    review.sessionRevision=detail::workspaceRevision(document.dump());
    review.proposalOpacity=document.value("proposal_opacity",review.proposalOpacity);review.committedOpacity=document.value("committed_opacity",review.committedOpacity);
    review.committedTint=document.value("committed_tint",review.committedTint);review.fiberLineOpacity=document.value("fiber_line_opacity",review.fiberLineOpacity);
    review.fiberLineWidth=document.value("fiber_line_width",review.fiberLineWidth);review.boundaries=document.value("boundaries",review.boundaries);
    if(document.contains("quarantined"))for(const auto& [id,p]:document.at("quarantined").items())
        review.quarantined[id]={p.at("source").get<std::string>(),p.at("destination").get<std::string>()};
    uint64_t maxOrder=0;
    for(const auto& [name,row]:document.at("fibers").items()) {
        ReviewFiber f;f.path=row.value("path",(review.settings.fiberFolder/name).string());
        const auto direction=row.value("direction",std::string("?"));f.direction=direction.empty()?'?':direction[0];
        f.contactPolicyVersion=1;
        const std::string id="fiber:"+name;
        auto readEntries=[&](const char* key,bool linked) {
            if(!row.contains(key))return;
            for(const auto& e:row.at(key)) {
                auto patch=e.at("patch").get<std::string>();if(patch.rfind("patch:",0)!=0)patch="patch:"+patch;
                ReviewDecision d;d.status=e.contains("status")?statusFromName(e.at("status").get<std::string>()):linked?ReviewStatus::Accepted:ReviewStatus::Rejected;
                // vc_fiber_auto_link (version 1) writes no flag and every entry is automatic; VC3D (version 2) writes it explicitly.
                d.automatic=e.value("automatic",version==1);d.note=e.value("note",std::string{});
                if(d.note.empty()&&d.automatic)d.note=autoNote(e,linked);
                d.order=e.value("order",uint64_t(0));if(!d.order)d.order=++maxOrder;else maxOrder=std::max(maxOrder,d.order);
                if(e.contains("keypoints"))d.keypoints=keypointsFromJson(e.at("keypoints"));
                Json m=Json::object();for(const auto& k:{"contact","anchor","round","strip_same","strip_adjacent","linked_same","linked_adjacent"})if(e.contains(k))m[k]=e.at(k);
                if(!m.empty())d.measurements=std::move(m);
                f.decisions[patch]=std::move(d);
            }
        };
        readEntries("linked",true);readEntries("unlinked",false);
        if(row.contains("pending_keypoints"))for(const auto& [patch,k]:row.at("pending_keypoints").items())f.savedKeypoints[patch.rfind("patch:",0)==0?patch:"patch:"+patch]=keypointsFromJson(k);
        review.fibers.emplace(id,std::move(f));
    }
    review.nextOrder=maxOrder+1;
    review.selected=document.value("selected",std::string{});
    if(!review.selected.empty()&&!review.fibers.contains(review.selected))review.selected.clear();
    if(review.selected.empty()&&!review.fibers.empty())review.selected=review.fibers.begin()->first;
    return review;
}
Json linksFromReview(const FiberReview& review) {
    Json out={{"type","vc_fiber_patch_links"},{"version",2},{"settings",settingsToJson(review.settings)},{"selected",review.selected},
        {"proposal_opacity",review.proposalOpacity},{"committed_opacity",review.committedOpacity},{"committed_tint",review.committedTint},
        {"fiber_line_opacity",review.fiberLineOpacity},{"fiber_line_width",review.fiberLineWidth},{"boundaries",review.boundaries},
        {"quarantined",Json::object()},{"fibers",Json::object()}};
    for(const auto& [id,p]:review.quarantined)out["quarantined"][id]={{"source",p.source.string()},{"destination",p.destination.string()}};
    for(const auto& [id,f]:review.fibers) {
        Json row={{"path",f.path.string()},{"direction",std::string(1,f.direction)},{"linked",Json::array()},{"unlinked",Json::array()}};
        // Decisions in order, so the saved lists read like the review happened.
        std::vector<std::pair<std::string,const ReviewDecision*>> ordered;for(const auto& [patch,d]:f.decisions)ordered.emplace_back(patch,&d);
        std::sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){return std::tie(a.second->order,a.first)<std::tie(b.second->order,b.first);});
        for(const auto& [patch,d]:ordered) {
            Json e={{"patch",patch.rfind("patch:",0)==0?patch.substr(6):patch},{"status",statusName(d->status)},{"order",d->order}};
            e["automatic"]=d->automatic;
            if(!d->note.empty())e["note"]=d->note;
            // Manual pairs from a fitted placement survive as keypoints.
            std::vector<Link> keypoints=d->keypoints;
            if(keypoints.empty())for(const auto& l:d->alignment.links.read())if(l.keypoint)keypoints.push_back(l);
            if(!keypoints.empty())e["keypoints"]=keypointsToJson(keypoints);
            if(!d->measurements.is_null())for(const auto& [k,v]:d->measurements.items())e[k]=v;
            row[d->status==ReviewStatus::Accepted?"linked":"unlinked"].push_back(std::move(e));
        }
        Json pending=Json::object();
        for(const auto& [patch,k]:f.savedKeypoints)if(!k.empty()&&!f.decisions.contains(patch))pending[patch.rfind("patch:",0)==0?patch.substr(6):patch]=keypointsToJson(k);
        if(f.pending&&f.current>=0&&f.current<int(f.proposals.size())) {
            const auto& patch=f.proposals[f.current].id;
            std::vector<Link> k;for(const auto& l:f.pending->links.read())if(l.keypoint)k.push_back(l);
            if(!k.empty()&&!f.decisions.contains(patch))pending[patch.rfind("patch:",0)==0?patch.substr(6):patch]=keypointsToJson(k);
        }
        if(!pending.empty())row["pending_keypoints"]=std::move(pending);
        out["fibers"][f.path.filename().string()]=std::move(row);
    }
    return out;
}
FiberReview reviewFromDocument(const Json& document) {
    return isLinksDocument(document)?reviewFromLinks(document):reviewFromJson(document);
}
std::string saveLinks(const FiberReview& review,const std::filesystem::path& path) {
    const auto document=linksFromReview(review);
    auto temporary=path;temporary+=".tmp";
    try {
        std::ofstream output(temporary,std::ios::trunc);output<<document.dump(1);output.flush();
        if(!output)throw std::runtime_error("Failed to write fiber links workspace");output.close();
        std::filesystem::rename(temporary,path);
    } catch(...) {std::error_code error;std::filesystem::remove(temporary,error);throw;}
    return detail::workspaceRevision(document.dump());
}
std::string saveReview(const FiberReview& review,const std::filesystem::path& path) {
    const auto document=reviewToJson(review);
    auto temporary=path;temporary+=".tmp";
    try {
        std::ofstream output(temporary,std::ios::trunc);output<<document;output.flush();
        if(!output)throw std::runtime_error("Failed to write fiber review workspace");output.close();
        std::filesystem::rename(temporary,path);
    } catch(...) {std::error_code error;std::filesystem::remove(temporary,error);throw;}
    return detail::workspaceRevision(document.dump());
}
FiberReview loadReview(const std::filesystem::path& path) {
    std::ifstream input(path);if(!input)throw std::runtime_error("Cannot open fiber review workspace");
    return reviewFromDocument(Json::parse(input));
}
FiberReview quarantineReviewPatch(const FiberReview& current,const std::filesystem::path& workspace,
                                 const PatchEntry& patch,const Progress& progress) {
    namespace fs=std::filesystem;
    if(patch.id.empty()||current.quarantined.contains(patch.id))throw std::runtime_error("Select a patch to quarantine");
    const auto path=fs::absolute(workspace);
    const auto original=detail::workspaceText(path);
    if(current.sessionRevision.empty()||current.sessionRevision!=detail::workspaceRevision(Json::parse(original).dump()))
        throw std::runtime_error("The workspace changed on disk; reopen it before quarantining a patch");
    const auto moves=detail::planQuarantine(current.settings.patchFolder,path,{patch.path});
    const auto& [source,destination]=*moves.begin();
    std::set<std::string> excluded{patch.id};
    auto collect=[&](const Graph& graph){for(const auto& [id,p]:graph.nodes)
        if(!p.source->fiber&&fs::weakly_canonical(p.source->path)==source)excluded.insert(id);};
    for(const auto& [id,f]:current.fibers){
        for(const auto& [patchId,d]:f.decisions)collect(d.alignment);
        if(f.pending)collect(*f.pending);
    }
    auto result=current;
    for(const auto& id:excluded)result.quarantined[id]={source,destination};
    for(auto& [id,f]:result.fibers){
        removeReviewProposals(f,excluded);
        for(const auto& patchId:excluded)f.decisions.erase(patchId);
    }
    const bool compact=isLinksDocument(Json::parse(original));
    const auto document=compact?linksFromReview(result):reviewToJson(result);const auto candidate=compact?document.dump(1):document.dump();
    // Validate all remaining associations before the filesystem transaction.
    (void)reviewFromDocument(document);
    if(progress)progress("Saving the workspace and moving source folders to "+destination.parent_path().string());
    detail::publishQuarantine(path,original,candidate,moves);
    result.sessionRevision=detail::workspaceRevision(candidate);
    return result;
}
} // namespace vc::constraints
