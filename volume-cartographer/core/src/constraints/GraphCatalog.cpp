#include "GraphInternal.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"
#include "vc/fiber_tracer/FiberGeometry.hpp"
#include "vc/atlas/FiberHvClassification.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"
#include "vc/lasagna/Manifest.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <fstream>
#include <numeric>
#include <set>
#include "utils/thread_pool.hpp"

namespace vc::constraints {
namespace bg=boost::geometry;
namespace bgi=boost::geometry::index;
using Point=bg::model::point<double,3,bg::cs::cartesian>;
using Box=bg::model::box<Point>;
using Bound=std::pair<Box,size_t>;
struct Catalog::BoundsIndex {
    bgi::rtree<Bound,bgi::quadratic<16>> tree;
    struct Contacts {
        double radius=0,scale=0;
        int stride=0;
        struct Source {std::string id;std::weak_ptr<const Node> node;};
        std::vector<Source> sources;
        Candidate candidate;
        size_t bytes=0;
        uint64_t used=0;
    };
    // Discovery is restarted after each edit, but immutable source geometry
    // usually has the same contacts. Bound the cache and never retain sources.
    mutable std::mutex mutex;
    mutable std::map<size_t,Contacts> contacts;
    mutable size_t contactBytes=0;
    mutable uint64_t clock=0;
    struct FiberBounds {
        std::weak_ptr<const Node> source;
        double radius=0;
        int stride=0;
        struct Run {size_t first,last;};
        std::map<size_t,std::vector<Run>> patches;
        size_t bytes=0;
        uint64_t used=0;
    };
    mutable std::map<std::string,std::shared_ptr<FiberBounds>> fiberBounds;
    mutable size_t boundsBytes=0;
};
namespace {
std::vector<size_t> sampledLinePoints(size_t count,int stride) {
    std::vector<size_t> indices;
    if(!count)return indices;
    indices.reserve((count-1)/size_t(stride)+2);
    for(size_t i=0;i<count;i+=size_t(stride))indices.push_back(i);
    if(indices.back()!=count-1)indices.push_back(count-1);
    return indices;
}
Point point(const cv::Vec3d& p) { return {p[0],p[1],p[2]}; }
nlohmann::json readJson(const std::filesystem::path& p) {
    std::ifstream f(p); if(!f) throw std::runtime_error("Cannot read "+p.string());
    return nlohmann::json::parse(f);
}
std::vector<std::filesystem::path> entries(const std::filesystem::path& path) {
    std::vector<std::filesystem::path> out;
    for(const auto& e:std::filesystem::directory_iterator(path)) out.push_back(e.path());
    std::sort(out.begin(),out.end()); return out;
}
cv::Vec3d vector3(const nlohmann::json& j) {
    if(!j.is_array()||j.size()!=3) throw std::runtime_error("Expected an XYZ triple");
    cv::Vec3d p{j[0].get<double>(),j[1].get<double>(),j[2].get<double>()};
    if(!std::isfinite(p[0])||!std::isfinite(p[1])||!std::isfinite(p[2])) throw std::runtime_error("Nonfinite XYZ");
    return p;
}
Location atLine(const std::string& id,size_t index) { Location p{id};p.lineIndex=static_cast<int64_t>(index);return p; }
struct PatchContact { Location location; double distance; };
std::optional<PatchContact> patchContact(const Node& node,const QuadSurface& surface,
                                       const SurfacePatchIndex& index,const cv::Vec3d& xyz,double radius) {
    SurfacePatchIndex::PointQuery query;query.worldPoint=cv::Vec3f(xyz);query.tolerance=radius;
    auto hit=index.locate(query);if(!hit)return {};
    Location location{node.id,cv::Vec2d(surface.ptrToGrid(hit->ptr))};
    try { (void)detail::weights(node,location); } catch(const std::exception&) { return {}; }
    return PatchContact{location,hit->distance};
}
void alignNewNodes(Graph& g,const std::set<std::string>& existing) {
    auto placed=existing;
    if(placed.empty()&&!g.nodes.empty()) placed.insert(g.seed);
    while(placed.size()<g.nodes.size()) {
        bool progress=false;
        for(const auto& link:g.links) {
            auto a=link.a,b=link.b;
            if(!placed.contains(a.node)) std::swap(a,b);
            if(!placed.contains(a.node)||placed.contains(b.node)) continue;
            auto& p=g.nodes.at(b.node);
            // The linked point fixes translation. The local 3-D sheet frame
            // determines initial rotation, including a patch with a single hit.
            const auto& other=g.nodes.at(a.node);
            const auto wa=detail::weights(*other.source,a), wb=detail::weights(*p.source,b);
            const auto& na=*other.source; const auto& nb=*p.source;
            cv::Vec3d xa=na.xyz[wa.indices[1]]-na.xyz[wa.indices[0]], ya=na.xyz[wa.indices[2]]-na.xyz[wa.indices[0]];
            cv::Vec3d xb=nb.xyz[wb.indices[1]]-nb.xyz[wb.indices[0]];
            cv::Vec3d yb=nb.xyz[wb.indices[2]]-nb.xyz[wb.indices[0]];
            cv::Vec2d ua=other.uv[wa.indices[1]]-other.uv[wa.indices[0]], va=other.uv[wa.indices[2]]-other.uv[wa.indices[0]];
            const auto childU=p.uv[wb.indices[1]]-p.uv[wb.indices[0]];
            const auto childV=p.uv[wb.indices[2]]-p.uv[wb.indices[0]];
            // Propagate the parent's chart handedness along the whole link
            // chain, including parents already reflected by an earlier link.
            if(xa.cross(ya).dot(xb.cross(yb))*detail::cross(ua,va)*detail::cross(childU,childV)<0)
                for(auto& uv:p.uv) uv[0]*=-1;
            cv::Vec2d ub=p.uv[wb.indices[1]]-p.uv[wb.indices[0]];
            double det=xa.dot(xa)*ya.dot(ya)-xa.dot(ya)*xa.dot(ya);
            if(det>1e-10) {
                cv::Vec2d target=ua*((xa.dot(xb)*ya.dot(ya)-ya.dot(xb)*xa.dot(ya))/det)
                               +va*((ya.dot(xb)*xa.dot(xa)-xa.dot(xb)*xa.dot(ya))/det);
                if(cv::norm(target)>1e-8&&cv::norm(ub)>1e-8) {
                    double angle=std::atan2(target[1],target[0])-std::atan2(ub[1],ub[0]);
                    double c=std::cos(angle),s=std::sin(angle);
                    for(auto& uv:p.uv) uv={c*uv[0]-s*uv[1],s*uv[0]+c*uv[1]};
                }
            }
            auto [anchorA,anchorB]=detail::linkWeights(g,Link{a,b,link.keypoint,link.pending});
            auto delta=detail::interpolate(other.uv,anchorA)-detail::interpolate(p.uv,anchorB);
            for(auto& uv:p.uv) uv+=delta;
            placed.insert(b.node); progress=true;
        }
        if(!progress) throw std::runtime_error("Addition is disconnected from the seed");
    }
}
void addFiberLinks(Graph& g,const Catalog& c) {
    for(const auto& link:c.links) {
        if(!g.nodes.contains(link.a)||!g.nodes.contains(link.b)) continue;
        auto a=atLine(link.a,c.fibers.at(link.a).controls.at(link.controlA));
        auto b=atLine(link.b,c.fibers.at(link.b).controls.at(link.controlB));
        bool found=false;
        for(const auto& old:g.links) if(!old.keypoint&&old.a.node==a.node&&old.b.node==b.node&&old.a.lineIndex==a.lineIndex&&old.b.lineIndex==b.lineIndex) {found=true;break;}
        if(!found) g.links.push_back({a,b,false,link.pending});
    }
}
}
void Settings::validate() const {
    for(double value:{spacing,radius,fiberScale,patchScale,normalScale,volumeScale,stiffness})
        if(!std::isfinite(value)||value<=0) throw std::runtime_error("Spacing, tolerance, scales and stiffness must be positive and finite");
}
Catalog loadCatalog(const Settings& settings,const std::atomic_bool* cancel,Progress progress) {
    settings.validate(); Catalog out;
    std::map<std::string,std::string> aliases;
    struct RawLink {
        std::string a,b;size_t ca,cb;bool pending;
        std::optional<cv::Vec3d> localPosition, peerPosition;
    };
    std::vector<RawLink> links;
    std::map<std::string,std::vector<cv::Vec3d>> controlPositions;
    std::vector<cv::Vec3d> centers;
    std::string centerError;
    std::optional<std::array<size_t,3>> normalShape;
    if(!settings.normalManifest.empty()) {
        // Local metadata is enough for declared coordinate domains; a remote
        // manifest is opened by the workspace before constructing this catalog.
        if(std::filesystem::is_regular_file(settings.normalManifest)) {
            const auto manifest=lasagna::LasagnaDatasetManifest::parseFile(settings.normalManifest);
            normalShape=manifest.baseShapeZYX;
            try {
                const auto name=manifest.raw.value("umbilicus_json",std::string{});
                if(!name.empty()) {
                    const auto json=readJson(manifest.baseDirectory/name);
                    for(const auto& p:json.at("control_points")) {
                        const auto xyz=vector3(nlohmann::json::array({p.at("x"),p.at("y"),p.at("z")}));
                        centers.push_back(xyz/settings.normalScale);
                    }
                }
            }catch(const std::exception& e){centers.clear();centerError=e.what();}
        }
    }
    for(const auto& path:entries(settings.fiberFolder)) {
        detail::cancelled(cancel); if(path.extension()!=".json") continue;
        try {
            auto json=readJson(path);auto parsed=fiber_tracer::parseVc3dFiberJson(json,path.string());
            auto controls=fiber_tracer::orderedControlPointLineIndices(parsed.controlPoints,parsed.linePoints);
            if(!controls||controls->size()<2) throw std::runtime_error("Fiber needs two ordered controls on its dense line");
            double scale=settings.fiberScale;
            if(json.contains("coordinate_base_shape_zyx")) {
                if(!normalShape) throw std::runtime_error("Declared fiber domain requires Lasagna base-shape metadata");
                auto shape=json.at("coordinate_base_shape_zyx").get<std::array<size_t,3>>();
                auto factor=lasagna::dyadicCoordinateScaleBetweenShapes(shape,*normalShape);
                scale=factor/settings.normalScale;
            }
            Fiber f;f.id="fiber:"+path.filename().string();f.path=path;
            f.fullyTraced=true;
            for(size_t i=0;i+1<parsed.controlPoints.size();++i)
                if(!parsed.segmentMetadata[i].is_object()||parsed.segmentMetadata[i].value("interp_mode",std::string{})!="trace") f.fullyTraced=false;
            auto& positions=controlPositions[f.id];
            for(const auto& p:parsed.controlPoints) positions.push_back(p*scale);
            f.line=fiber_tracer::linePointsBetweenOuterControlPoints(parsed.linePoints,parsed.controlPoints);
            for(auto& p:f.line) p*=scale;
            for(auto index:*controls) f.controls.push_back(index-controls->front());
            auto hv=atlas::classifyFiberHv(parsed.controlPoints);
            auto tag=hv.automaticTag;
            if(json.contains("hv_classification")) {
                auto manual=atlas::fiberHvTagFromString(json["hv_classification"].value("manual_tag",std::string{}));
                if(manual!=atlas::FiberHvTag::Unknown) tag=manual;
            }
            auto name=atlas::fiberHvTagToString(tag);f.direction=name=="V"?'V':name=="H"?'H':'?';
            aliases[path.filename().string()]=f.id;
            auto stored=json.value("filename",std::string{});if(!stored.empty()) aliases[stored]=f.id;
            for(const auto& b:json.value("branches",nlohmann::json::array())) {
                try {
                    auto peer=std::filesystem::path(b.at("branch_file").get<std::string>()).filename().string();
                    int ca=b.at("control_point_index").get<int>(),cb=b.at("branch_control_point_index").get<int>();
                    if(ca<0||cb<0) throw std::runtime_error("Negative link control index");
                    RawLink link{f.id,peer,static_cast<size_t>(ca),static_cast<size_t>(cb),b.value("pending",false),{}, {}};
                    // Snapshot metadata is optional on older annotations. A
                    // malformed snapshot warns without dropping a valid link.
                    for(auto [key,position]:{std::pair{"control_point_position",&link.localPosition},
                                            std::pair{"branch_control_point_position",&link.peerPosition}}) {
                        if(!b.contains(key))continue;
                        try{*position=vector3(b.at(key))*scale;}
                        catch(const std::exception& e){out.warnings.push_back(f.id+": "+key+": "+e.what());}
                    }
                    links.push_back(std::move(link));
                }catch(const std::exception& e){out.warnings.push_back(path.filename().string()+": link: "+e.what());}
            }
            out.fibers.emplace(f.id,std::move(f));
        } catch(const std::exception& e) {out.warnings.push_back(path.filename().string()+": "+e.what());}
    }
    std::map<std::tuple<std::string,size_t,std::string,size_t>,FiberLink> dedup;
    std::set<std::tuple<std::string,size_t,std::string,size_t>> directed;
    for(auto l:links) {
        auto target=aliases.find(l.b);
        if(target==aliases.end()||!out.fibers.contains(l.a)||l.ca>=out.fibers.at(l.a).controls.size()||l.cb>=out.fibers.at(target->second).controls.size()) {
            out.warnings.push_back("Unresolved link: "+l.a+" -> "+l.b);continue;
        }
        l.b=target->second;
        directed.emplace(l.a,l.ca,l.b,l.cb);
        const double tolerance=1e-3/settings.normalScale;
        if((l.localPosition&&cv::norm(*l.localPosition-controlPositions.at(l.a).at(l.ca))>tolerance)||
           (l.peerPosition&&cv::norm(*l.peerPosition-controlPositions.at(l.b).at(l.cb))>tolerance))
            out.warnings.push_back("Stale link position metadata: "+l.a+" [control "+std::to_string(l.ca)+"] -> "+l.b+" [control "+std::to_string(l.cb)+"]");
        if(std::pair{l.b,l.cb}<std::pair{l.a,l.ca}) {std::swap(l.a,l.b);std::swap(l.ca,l.cb);}
        auto key=std::tuple{l.a,l.ca,l.b,l.cb}; auto& link=dedup[key];
        link={l.a,l.b,l.ca,l.cb,link.pending||l.pending};
    }
    for(const auto& [a,ca,b,cb]:directed)
        if(!directed.contains({b,cb,a,ca})) out.warnings.push_back("Missing reciprocal link: "+a+" [control "+std::to_string(ca)+"] -> "+b+" [control "+std::to_string(cb)+"]");
    std::map<std::string,std::vector<std::string>> adjacent;
    for(const auto& [key,l]:dedup) {out.links.push_back(l);adjacent[l.a].push_back(l.b);adjacent[l.b].push_back(l.a);}
    std::set<std::string> visited;
    for(const auto& [id,f]:out.fibers) {
        if(!visited.insert(id).second) continue;
        std::vector<std::string> component{id};
        for(size_t i=0;i<component.size();++i) for(const auto& next:adjacent[component[i]])
            if(visited.insert(next).second) component.push_back(next);
        std::sort(component.begin(),component.end());out.components.push_back(std::move(component));
    }
    std::sort(out.components.begin(),out.components.end(),[](const auto& a,const auto& b){return a.size()!=b.size()?a.size()>b.size():a.front()<b.front();});
    if(progress)progress("Checking source fiber cycles…");
    try{out.cycleAudit=auditFiberCycles(out,centers,417.0/settings.normalScale,cancel);}
    catch(const std::exception& e){detail::cancelled(cancel);out.cycleAudit.summary="Fiber cycle check unavailable: "+std::string(e.what());}
    if(!centerError.empty())out.cycleAudit.details.push_back("Cannot read umbilicus: "+centerError);
    auto bounds=std::make_shared<Catalog::BoundsIndex>();std::vector<Bound> values;
    size_t count=0;
    for(const auto& path:settings.patchFolder.empty()?std::vector<std::filesystem::path>{}:entries(settings.patchFolder)) {
        detail::cancelled(cancel);if(!std::filesystem::is_directory(path)) continue;
        try {
            auto json=readJson(path/"meta.json");
            auto bbox=json.at("bbox");
            PatchEntry entry{"patch:"+path.filename().string(),path,vector3(bbox.at(0))*settings.patchScale,vector3(bbox.at(1))*settings.patchScale};
            for(int d=0;d<3;++d) if(entry.low[d]>entry.high[d]) throw std::runtime_error("Inverted patch bounds");
            values.push_back({Box(point(entry.low),point(entry.high)),out.patches.size()});
            out.patches.push_back(std::move(entry));
        }catch(const std::exception& e){out.warnings.push_back(path.filename().string()+": "+e.what());}
        if(progress&&++count%1000==0) progress("Indexed "+std::to_string(count)+" patch bounds");
    }
    bounds->tree=decltype(bounds->tree)(values.begin(),values.end());out.boundsIndex=bounds;
    return out;
}

std::shared_ptr<const Node> makeFiberNode(const Fiber& f,const Settings& s,const lasagna::NormalSampler& sampler) {
    lasagna::LineModel line;
    bool haveNormal=false;
    cv::Vec3d firstNormal{0,0,0};
    for(const auto& p:f.line) {
        auto normal=sampler.sampleNormal(p);
        if(normal.valid&&!haveNormal)firstNormal=normal.normal;
        haveNormal|=normal.valid;
        line.points.push_back({p,normal,true});
    }
    if(!haveNormal)throw std::runtime_error("No Lasagna normals along fiber "+f.id);
    lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels=s.spacing;config.crossSpacingBaseVoxels=s.spacing;
    config.crossSampleCount=7;config.buildLineZSlices=false;
    // Uniform arclength columns; controls remain exact fractional references.
    config.controlPointLinePositions={0.0,static_cast<double>(f.line.size()-1)};
    auto views=lasagna::buildLineViewSurfaces(line,config);
    auto node=std::make_shared<Node>();node->id=f.id;node->path=f.path;node->fiber=true;node->direction=f.direction;
    node->line=f.line;node->controls=f.controls;
    node->lineArclength=views.stripPositionMap.originalArclengths;
    node->columnArclength=views.stripPositionMap.stripGridArclengths;
    const auto& grid=*views.lineSurface->rawPointsPtr();node->rows=grid.rows;node->cols=grid.cols;
    node->xyz.resize(grid.total());node->initialUv.resize(grid.total());
    // An intrinsic straight ribbon is an injective starting chart even for
    // a fiber that winds repeatedly around the scroll. Integrating projected
    // XYZ tangents here can fold the initial chart before optimization begins.
    const auto delta=f.line.back()-f.line.front();
    cv::Vec2d along{1,0};
    if(f.direction=='V') along={0,delta[2]>=0?-1.0:1.0};
    else {
        const auto normal=detail::unit(firstNormal);
        const auto right=detail::unit(cv::Vec3d{0,0,-1}.cross(normal));
        if(delta.dot(right)<0) along={-1,0};
    }
    const cv::Vec2d across{-along[1],along[0]};
    for(int x=0;x<grid.cols;++x) for(int y=0;y<grid.rows;++y) {
        node->xyz[y*grid.cols+x]=cv::Vec3d(grid(y,x));
        node->initialUv[y*grid.cols+x]=along*node->columnArclength[x]
            +across*((y-grid.rows/2)*s.spacing);
    }
    // LineViewBuilder chooses a coherent normal sign independently of the
    // dataset. Normalize the chart orientation once, preserving its geometry.
    detail::triangulate(*node);
    if(!node->triangles.empty()) {
        auto t=node->triangles.front();
        if(detail::cross(node->initialUv[t[1]]-node->initialUv[t[0]],node->initialUv[t[2]]-node->initialUv[t[0]])<0)
            for(auto& uv:node->initialUv) uv[0]*=-1;
    }
    return node;
}
std::shared_ptr<const Node> loadPatchNode(const PatchEntry& p,const Settings& s) {
    QuadSurface surf(p.path);surf.ensureLoaded();const auto& grid=*surf.rawPointsPtr();
    auto n=std::make_shared<Node>();n->id=p.id;n->path=p.path;n->rows=grid.rows;n->cols=grid.cols;
    n->xyz.resize(grid.total());n->initialUv.resize(grid.total());
    auto scale=surf.scale();
    if(scale[0]<=0||scale[1]<=0) throw std::runtime_error("Patch has invalid grid scale");
    for(int y=0;y<grid.rows;++y) for(int x=0;x<grid.cols;++x) {
        auto i=y*grid.cols+x;
        n->xyz[i]=grid(y,x)[0]==-1?cv::Vec3d{-1,-1,-1}:cv::Vec3d(grid(y,x))*s.patchScale;
        n->initialUv[i]={x/scale[0]*s.patchScale,y/scale[1]*s.patchScale};
    }
    detail::triangulate(*n);if(n->triangles.empty()) throw std::runtime_error("Patch has no valid quads: "+p.id);
    return n;
}
Location fiberLineLocation(const Node& n,size_t index) {
    if(!n.fiber||index>=n.lineArclength.size()||n.columnArclength.size()<2) throw std::runtime_error("Invalid fiber location");
    double a=n.lineArclength[index];auto it=std::lower_bound(n.columnArclength.begin(),n.columnArclength.end(),a);
    double col=0;
    if(it==n.columnArclength.end()) col=n.cols-1;
    else if(it!=n.columnArclength.begin()) {
        size_t hi=it-n.columnArclength.begin();double span=*it-n.columnArclength[hi-1];
        col=hi-1+(span>0?(a-n.columnArclength[hi-1])/span:0);
    }
    return {n.id,{col,static_cast<double>(n.rows/2)}};
}
cv::Vec2d placedLocation(const Graph& g,const Location& l) {
    const auto& p=g.nodes.at(l.node);return detail::interpolate(p.uv,detail::weights(*p.source,l));
}
cv::Vec3d sourceLocation(const Node& n,const cv::Vec2d& p) {return detail::interpolate(n.xyz,detail::weights(n,p));}
Graph seedGraph(const Catalog& c,size_t component,const Settings& s,const lasagna::NormalSampler& sampler,const std::atomic_bool* cancel) {
    Graph g;g.settings=s;
    for(const auto& id:c.components.at(component)) {
        detail::cancelled(cancel);auto n=makeFiberNode(c.fibers.at(id),s,sampler);g.nodes.emplace(id,Placement{n,n->initialUv});
    }
    if(g.nodes.empty()) throw std::runtime_error("No eligible fibers in component");
    g.seed=g.nodes.begin()->first;addFiberLinks(g,c);alignNewNodes(g,{});return g;
}

std::vector<Candidate> discover(const Catalog& c,const Graph& g,const std::string& selected,const std::atomic_bool* cancel,Progress progress,DiscoveryBatch found,int lineStride) {
    if(lineStride<1)throw std::invalid_argument("Discovery stride must be positive");
    detail::cancelled(cancel);
    std::map<std::string,Candidate> results;
    std::vector<Candidate> out;
    // Split separated fiber contacts before publishing a usable proposal.
    auto append=[&](Candidate candidate) {
        const auto first=out.size();
        const auto onFiber=[&](const Link& l)->const Location& {return candidate.fiber?l.b:l.a;};
        std::sort(candidate.contacts.begin(),candidate.contacts.end(),[&](const auto& a,const auto& b){return std::tie(onFiber(a).node,onFiber(a).lineIndex)<std::tie(onFiber(b).node,onFiber(b).lineIndex);});
        std::vector<double> arclength;
        if(candidate.fiber) {
            const auto& line=c.fibers.at(candidate.id).line;arclength.resize(line.size());
            for(size_t i=1;i<line.size();++i)arclength[i]=arclength[i-1]+cv::norm(line[i]-line[i-1]);
        }
        Candidate cluster;cluster.id=candidate.id;cluster.fiber=candidate.fiber;cluster.distance=candidate.distance;
        for(const auto& contact:candidate.contacts) {
            bool continuous=cluster.contacts.empty();
            if(!continuous) {
                const auto& last=onFiber(cluster.contacts.back());
                const auto& next=onFiber(contact);
                if(last.node==next.node) {
                    const auto& lengths=candidate.fiber?arclength:g.nodes.at(last.node).source->lineArclength;
                    continuous=lengths.at(next.lineIndex)-lengths.at(last.lineIndex)<=2*g.settings.spacing;
                }
            }
            if(!continuous){out.push_back(std::move(cluster));cluster=Candidate{};cluster.id=candidate.id;cluster.fiber=candidate.fiber;cluster.distance=candidate.distance;}
            cluster.contacts.push_back(contact);
        }
        if(!cluster.contacts.empty())out.push_back(std::move(cluster));
        if(found&&out.size()>first)found(std::vector<Candidate>(out.begin()+first,out.end()));
    };
    struct Query {
        std::string id;
        std::shared_ptr<const Node> source;
        std::vector<Catalog::BoundsIndex::FiberBounds::Run> runs;
    };
    std::map<size_t,std::vector<Query>> queries;
    const auto radius=g.settings.radius;
    if(progress)progress("Searching nearby patch bounds");
    for(const auto& [id,placement]:g.nodes) {
        detail::cancelled(cancel);
        if(!selected.empty()&&id!=selected) continue;
        const auto& n=*placement.source;
        if(n.fiber) {
            const auto& cache=*c.boundsIndex;
            std::shared_ptr<Catalog::BoundsIndex::FiberBounds> bounds;
            {
                std::lock_guard lock(cache.mutex);
                auto old=cache.fiberBounds.find(id);
                if(old!=cache.fiberBounds.end()&&old->second->source.lock()==placement.source&&
                   old->second->radius==radius&&old->second->stride==lineStride) {
                    bounds=old->second;bounds->used=++cache.clock;
                }
            }
            if(!bounds) {
                bounds=std::make_shared<Catalog::BoundsIndex::FiberBounds>();
                bounds->source=placement.source;bounds->radius=radius;bounds->stride=lineStride;
                const auto sampled=sampledLinePoints(n.line.size(),lineStride);
                // Nearby dense points mostly traverse the same R-tree branches.
                // Query a small batch's union once, then retain precisely the same
                // point/patch box intersections in the original per-patch order.
                for(size_t first=0;first<sampled.size();first+=32) {
                    detail::cancelled(cancel);
                    const auto end=std::min(first+32,sampled.size());
                    cv::Vec3d low=cv::Vec3d::all(INFINITY),high=cv::Vec3d::all(-INFINITY);
                    for(size_t j=first;j<end;++j)for(int d=0;d<3;++d) {
                        low[d]=std::min(low[d],n.line[sampled[j]][d]-radius);
                        high[d]=std::max(high[d],n.line[sampled[j]][d]+radius);
                    }
                    std::vector<Bound> hits;
                    c.boundsIndex->tree.query(bgi::intersects(Box(point(low),point(high))),std::back_inserter(hits));
                    for(const auto& hit:hits) {
                        const auto& patch=c.patches[hit.second];
                        for(size_t j=first;j<end;++j) {
                            const auto i=sampled[j];const auto p=n.line[i];
                            bool intersects=true;
                            for(int d=0;d<3;++d)intersects&=p[d]+radius>=patch.low[d]&&p[d]-radius<=patch.high[d];
                            if(intersects) {
                                auto& runs=bounds->patches[hit.second];
                                if(!runs.empty()&&runs.back().last+size_t(lineStride)==i)runs.back().last=i;
                                else runs.push_back({i,i});
                            }
                        }
                    }
                }
                bounds->bytes=sizeof(*bounds)+id.capacity();
                for(const auto& [patch,runs]:bounds->patches)bounds->bytes+=128+runs.capacity()*sizeof(Catalog::BoundsIndex::FiberBounds::Run);
                constexpr size_t limit=64*1024*1024;
                if(bounds->bytes<=limit) {
                    std::lock_guard lock(cache.mutex);
                    if(auto old=cache.fiberBounds.find(id);old!=cache.fiberBounds.end()) {
                        cache.boundsBytes-=old->second->bytes;cache.fiberBounds.erase(old);
                    }
                    while(cache.boundsBytes+bounds->bytes>limit&&!cache.fiberBounds.empty()) {
                        auto oldest=std::min_element(cache.fiberBounds.begin(),cache.fiberBounds.end(),[](const auto& a,const auto& b){return a.second->used<b.second->used;});
                        cache.boundsBytes-=oldest->second->bytes;cache.fiberBounds.erase(oldest);
                    }
                    bounds->used=++cache.clock;cache.boundsBytes+=bounds->bytes;
                    cache.fiberBounds.emplace(id,bounds);
                }
            }
            for(const auto& [patch,runs]:bounds->patches)if(!g.nodes.contains(c.patches[patch].id)) {
                queries[patch].push_back({id,placement.source,runs});
            }
        }
    }
    if(progress)progress("Verifying nearby patch contacts");
    size_t done=0;
    auto verify=[&](const auto& queryEntry) {
        const auto& [i,points]=queryEntry;
        detail::cancelled(cancel);const auto& entry=c.patches[i];
        const auto& cache=*c.boundsIndex;
        // For this catalog, immutable source identity + radius + stride fixes
        // every queried index. Store one identity per source, not millions of
        // repeated point indices (which would evict useful contact results).
        {
            std::lock_guard lock(cache.mutex);
            auto hit=cache.contacts.find(i);
            if(hit!=cache.contacts.end()) {
                auto& old=hit->second;
                if(old.radius==radius&&old.scale==g.settings.patchScale&&old.stride==lineStride&&old.sources.size()==points.size()&&
                   std::equal(points.begin(),points.end(),old.sources.begin(),[](const auto& q,const auto& source){
                       return q.source==source.node.lock()&&q.id==source.id;
                   })) {
                    old.used=++cache.clock;return old.candidate;
                }
            }
        }
        auto node=loadPatchNode(entry,g.settings);auto surface=detail::sourceSurface(*node);
        SurfacePatchIndex index;index.setSamplingStride(1);index.rebuild({surface});
        Candidate candidate;candidate.id=entry.id;
        for(const auto& query:points)for(const auto& run:query.runs)for(size_t line=run.first;line<=run.last;line+=size_t(lineStride)) {
            detail::cancelled(cancel);
            auto hit=patchContact(*node,*surface,index,query.source->line[line],radius);if(!hit)continue;
            if(candidate.contacts.empty()||hit->distance<candidate.distance) candidate.distance=hit->distance;
            candidate.contacts.push_back({atLine(query.id,line),hit->location});
        }
        Catalog::BoundsIndex::Contacts remembered;
        remembered.radius=radius;remembered.scale=g.settings.patchScale;remembered.stride=lineStride;
        for(const auto& query:points)remembered.sources.push_back({query.id,query.source});
        remembered.candidate=candidate;
        remembered.bytes=sizeof(remembered)+remembered.sources.capacity()*sizeof(remembered.sources[0])+candidate.id.capacity()+
            remembered.candidate.contacts.capacity()*sizeof(Link);
        for(const auto& source:remembered.sources)remembered.bytes+=source.id.capacity();
        for(const auto& link:remembered.candidate.contacts)remembered.bytes+=link.a.node.capacity()+link.b.node.capacity();
        constexpr size_t limit=128*1024*1024;
        if(remembered.bytes<=limit) {
            std::lock_guard lock(cache.mutex);
            if(auto old=cache.contacts.find(i);old!=cache.contacts.end()) {
                cache.contactBytes-=old->second.bytes;cache.contacts.erase(old);
            }
            while(cache.contactBytes+remembered.bytes>limit&&!cache.contacts.empty()) {
                auto oldest=std::min_element(cache.contacts.begin(),cache.contacts.end(),[](const auto& a,const auto& b){return a.second.used<b.second.used;});
                cache.contactBytes-=oldest->second.bytes;cache.contacts.erase(oldest);
            }
            remembered.used=++cache.clock;cache.contactBytes+=remembered.bytes;
            cache.contacts.emplace(i,std::move(remembered));
        }
        return candidate;
    };
    if(!queries.empty()) {
        // Bounded batches overlap independent TIFF reads/exact queries. Publish
        // in catalog order, retaining contact order and deterministic ties.
        const size_t width=std::min<size_t>({4,queries.size(),std::max(1u,std::thread::hardware_concurrency())});
        utils::ThreadPool workers(width);
        auto next=queries.begin();
        while(next!=queries.end()) {
            detail::cancelled(cancel);
            std::vector<decltype(next)> batch;
            for(size_t i=0;i<width&&next!=queries.end();++i,++next)batch.push_back(next);
            std::vector<Candidate> verified(batch.size());
            std::vector<std::exception_ptr> errors(batch.size());
            workers.run_indexed_batch(batch.size(),[&](size_t i){
                try{verified[i]=verify(*batch[i]);}catch(...){errors[i]=std::current_exception();}
            });
            for(size_t i=0;i<verified.size();++i) {
                if(errors[i])std::rethrow_exception(errors[i]);
                auto& candidate=verified[i];
                if(!candidate.contacts.empty())append(std::move(candidate));
                detail::cancelled(cancel);
                if(progress&&++done%50==0) progress("Searched "+std::to_string(done)+" / "+std::to_string(queries.size())+" nearby patches");
            }
        }
    }
    // Publish patch attachments first, before the separate patch-to-fiber scan.
    if(progress)progress("Searching nearby fiber contacts");
    for(const auto& [id,placement]:g.nodes) {
        if(!selected.empty()&&id!=selected)continue;
        const auto& n=*placement.source;if(n.fiber)continue;
        auto surf=detail::sourceSurface(n);SurfacePatchIndex index;index.setSamplingStride(1);index.rebuild({surf});
        for(const auto& [fid,f]:c.fibers) {
            if(g.nodes.contains(fid)) continue;
            detail::cancelled(cancel);
            for(size_t i:sampledLinePoints(f.line.size(),lineStride)) {
                auto hit=patchContact(n,*surf,index,f.line[i],radius);if(!hit)continue;
                auto& candidate=results[fid];candidate.id=fid;candidate.fiber=true;
                if(candidate.contacts.empty()||hit->distance<candidate.distance) candidate.distance=hit->distance;
                candidate.contacts.push_back({hit->location,atLine(fid,i)});
            }
        }
    }
    for(auto& [id,candidate]:results)append(std::move(candidate));
    std::stable_sort(out.begin(),out.end(),[](const auto& a,const auto& b){return a.distance!=b.distance?a.distance<b.distance:a.id<b.id;});return out;
}
Graph propose(const Catalog& c,const Graph& committed,const Candidate& candidate,bool whole,const lasagna::NormalSampler& sampler,const std::atomic_bool* cancel) {
    Graph g=committed;std::set<std::string> existing;for(const auto& [id,p]:g.nodes) existing.insert(id);
    // Keep the clicked attachment first for initial placement; other contacts
    // participate in the same joint fit rather than defining separate patches.
    g.links.insert(g.links.end(),candidate.contacts.begin(),candidate.contacts.end());
    if(candidate.fiber) {
        std::vector<std::string> additions{candidate.id};
        if(whole) for(const auto& component:c.components) if(std::find(component.begin(),component.end(),candidate.id)!=component.end()) {additions=component;break;}
        for(const auto& id:additions) if(!g.nodes.contains(id)) {
            detail::cancelled(cancel);auto n=makeFiberNode(c.fibers.at(id),g.settings,sampler);g.nodes.emplace(id,Placement{n,n->initialUv});
        }
    } else {
        auto it=std::find_if(c.patches.begin(),c.patches.end(),[&](const auto& p){return p.id==candidate.id;});
        if(it==c.patches.end()) throw std::runtime_error("Candidate patch disappeared");
        auto n=loadPatchNode(*it,g.settings);g.nodes.emplace(n->id,Placement{n,n->initialUv});
        alignNewNodes(g,existing);
        auto surface=detail::sourceSurface(*n);SurfacePatchIndex index;index.setSamplingStride(1);index.rebuild({surface});
        std::set<std::pair<std::string,int64_t>> attached;
        std::set<std::string> manual;
        for(const auto& link:g.links) {
            const auto* other=link.a.node==n->id?&link.b:link.b.node==n->id?&link.a:nullptr;
            if(!other)continue;
            if(link.keypoint)manual.insert(other->node);
            if(other->lineIndex>=0)attached.emplace(other->node,other->lineIndex);
        }
        // Discovery marks are local clusters. Once a patch is chosen, verify
        // its contacts against every placed fiber, including off-stride points.
        // Bounds only prune impossible hits; the shared exact query decides.
        cv::Vec3d low=cv::Vec3d::all(INFINITY),high=cv::Vec3d::all(-INFINITY);
        for(const auto& xyz:n->xyz)if(detail::valid(xyz))for(int d=0;d<3;++d) {
            low[d]=std::min(low[d],xyz[d]-g.settings.radius);
            high[d]=std::max(high[d],xyz[d]+g.settings.radius);
        }
        for(const auto& [id,p]:committed.nodes)if(p.source->fiber&&!manual.contains(id)) {
            const auto& fiber=*p.source;
            for(size_t i=0;i<fiber.line.size();++i) {
                detail::cancelled(cancel);const auto& xyz=fiber.line[i];
                if(attached.contains({id,int64_t(i)}))continue;
                if(xyz[0]<low[0]||xyz[0]>high[0]||xyz[1]<low[1]||xyz[1]>high[1]||xyz[2]<low[2]||xyz[2]>high[2])continue;
                auto hit=patchContact(*n,*surface,index,xyz,g.settings.radius);if(!hit)continue;
                // A winding fiber can revisit this XYZ on a distant chart
                // traversal. Only extend the clicked placement locally, using
                // the same four-spacing neighborhood as incremental fitting.
                if(cv::norm(placedLocation(g,atLine(id,i))-placedLocation(g,hit->location))>4*g.settings.spacing)continue;
                g.links.push_back({atLine(id,i),hit->location});
            }
        }
        // Actual 3-D overlap couples every already accepted patch into the same fit.
        for(const auto& [id,p]:committed.nodes) if(!p.source->fiber) {
            detail::cancelled(cancel);
            for(size_t i=0;i<p.source->xyz.size();++i) {
                if(!detail::valid(p.source->xyz[i])) continue;
                auto hit=patchContact(*n,*surface,index,p.source->xyz[i],g.settings.radius);if(!hit)continue;
                Location a{id,{static_cast<double>(i%p.source->cols),static_cast<double>(i/p.source->cols)}};
                try {detail::weights(*p.source,a);}catch(...){continue;}
                g.links.push_back({a,hit->location});
            }
        }
    }
    addFiberLinks(g,c);alignNewNodes(g,existing);return g;
}
} // namespace vc::constraints
