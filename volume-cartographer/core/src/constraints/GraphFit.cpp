#include "GraphInternal.hpp"
#include "GraphFiberJunctions.hpp"
#include "GraphPrepared.hpp"
#include "GraphFitCosts.hpp"
#include "vc/core/util/SymmetricDirichlet.hpp"
#include <ceres/ceres.h>
#include <set>
#include <tuple>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <numeric>
#include <ranges>
#include <chrono>

namespace vc::constraints {
namespace {
using Region=detail::ChartRegion;
struct LocalFit {
    const FitResult& committed;
    Region region;
    std::map<std::string,std::vector<unsigned char>> movable;
    size_t count=0,total=0;
    bool expand=false;
    const PreparedGraph* prepared=nullptr;
    std::shared_ptr<const detail::FiberJunctions> geometry;
    std::map<std::string,std::vector<size_t>> chartTriangles;
    std::vector<Link> contacts;
    std::vector<detail::PreparedContact> weights;
    std::vector<size_t> contactIds;
    std::set<std::string> parameterNodes;
    const GraphEdit* edit=nullptr;
    bool fixedReference=false;
    bool moves(const std::string& id,int i)const {auto it=movable.find(id);return it!=movable.end()&&it->second[i]!=0;}
    bool added(const std::string& id)const {return !committed.graph.nodes.contains(id);}
};
double attachmentRms(const Graph& graph,const LocalFit* local){
    if(!local||!local->prepared){
        double squared=0;for(const auto& link:graph.links.read()){const auto e=detail::linkError(graph,link);squared+=e.dot(e);}
        return graph.links.empty()?0:std::sqrt(squared/graph.links.size());
    }
    const auto& base=*local->prepared;
    auto errors=base.squaredErrors;
    for(size_t i=0;i<local->contacts.size();++i){
        if(local->contactIds[i]>=errors.size())continue;
        const auto& l=local->contacts[i];const auto& w=local->weights[i];
        const auto e=detail::interpolate(graph.nodes.at(l.a.node).uv,w.a)-detail::interpolate(graph.nodes.at(l.b.node).uv,w.b);
        errors[local->contactIds[i]]=e.dot(e);
    }
    const std::set<size_t> removed(local->edit->removedLinks.begin(),local->edit->removedLinks.end());
    double squared=0;size_t count=0;
    for(size_t i=0;i<errors.size();++i)if(!removed.contains(i)){squared+=errors[i];++count;}
    for(const auto& l:local->edit->addedLinks){const auto e=detail::linkError(graph,l);squared+=e.dot(e);++count;}
    return count?std::sqrt(squared/count):0;
}
void selectActiveCharts(const Graph& graph,LocalFit& local){
    for(const auto& [id,p]:graph.nodes){
        local.total+=p.uv.size();
        if(local.fixedReference&&!local.added(id))continue;
        if(local.prepared&&!local.added(id)&&!local.prepared->charts.at(id).bounds.intersects(local.region))continue;
        auto& mask=local.movable[id];mask.resize(p.uv.size());
        for(size_t i=0;i<p.uv.size();++i){mask[i]=detail::valid(p.source->xyz[i])&&(local.added(id)||local.region.contains(p.uv[i]));local.count+=mask[i];}
        if(local.prepared){
            auto& triangles=local.chartTriangles[id];
            if(local.added(id)){triangles.resize(p.source->triangles.size());std::iota(triangles.begin(),triangles.end(),0);}
            else triangles=local.prepared->charts.at(id).query(local.region,*p.source);
            local.parameterNodes.insert(id);
        }
    }
    if(!local.prepared)return;
    const auto& base=*local.prepared;
    std::set<size_t> incident,removed(local.edit->removedLinks.begin(),local.edit->removedLinks.end());
    for(const auto& [id,mask]:local.movable)if(auto it=base.incident.find(id);it!=base.incident.end())incident.insert(it->second.begin(),it->second.end());
    auto append=[&](const Link& link,const detail::PreparedContact& weights,size_t id){
        bool moves=false;for(int i=0;i<3;++i)moves|=local.moves(link.a.node,weights.a.indices[i])||local.moves(link.b.node,weights.b.indices[i]);
        if(!moves)return;
        local.contacts.push_back(link);local.weights.push_back(weights);local.contactIds.push_back(id);
        local.parameterNodes.insert(link.a.node);local.parameterNodes.insert(link.b.node);
    };
    for(size_t i:incident)if(!removed.contains(i))append(base.accepted.graph.links[i],base.contacts[i],i);
    for(const auto& link:local.edit->addedLinks){auto [a,b]=detail::linkWeights(graph,link);append(link,{a,b},base.contacts.size());}
}
const std::vector<Link>& fitContacts(const Graph& graph,const LocalFit* local){
    return local&&local->prepared?local->contacts:graph.links.read();
}
auto fitTriangles(const Node& node,const LocalFit* local){
    const std::vector<size_t>* indices=nullptr;
    if(local&&local->prepared){
        auto it=local->chartTriangles.find(node.id);
        if(it!=local->chartTriangles.end())indices=&it->second;
    }
    return std::views::iota(size_t(0),indices?indices->size():node.triangles.size())|
        std::views::transform([&node,indices](size_t i)->const cv::Vec3i&{return node.triangles[indices?(*indices)[i]:i];});
}
// The chart's bounding rectangle is often mostly empty between ribbons.
// Keep only an integer lookup there; XYZ scratch data belongs to covered sites.
struct RasterSample {
    // target: the rasterized position the fidelity term pulls toward.
    // point: the optimized position, which may start elsewhere.
    cv::Vec3d sum{0,0,0},normal{0,0,0},point{-1,-1,-1},target{-1,-1,-1};
    double mass=0;
    int owner=-1,visited=-1;
    bool active=false;
};
class RasterSamples {
    std::vector<int> indices_;
    std::vector<RasterSample> samples_;
public:
    explicit RasterSamples(size_t count):indices_(count,-1) {}
    const RasterSample& operator[](size_t i)const {
        static const RasterSample empty;
        return indices_[i]<0?empty:samples_[indices_[i]];
    }
    RasterSample& at(size_t i) {return samples_[indices_[i]];}
    RasterSample& insert(size_t i) {
        if(indices_[i]<0){indices_[i]=int(samples_.size());samples_.emplace_back();}
        return at(i);
    }
    void orderByGrid() {
        std::vector<RasterSample> ordered;ordered.reserve(samples_.size());
        for(auto& index:indices_)if(index>=0) {
            ordered.push_back(samples_[index]);index=int(ordered.size()-1);
        }
        samples_=std::move(ordered);
    }
};
struct Direction {
    int axis;double weight;
    template<class T> bool operator()(const T* a,const T* b,T* r) const {r[0]=T(weight)*(b[axis]-a[axis]);return true;}
};
// Each fiber owns a movable row/column. A dead band preserves small local
// bends, while deviations accumulated over a long fiber cannot drift freely.
template<class T> T outsideBand(const T& offset,double halfWidth) {
    if(dirichletValue(offset)>halfWidth)return offset-T(halfWidth);
    if(dirichletValue(offset)<-halfWidth)return offset+T(halfWidth);
    return T(0);
}
struct FiberBand {
    int axis;
    double halfWidth,weight;
    template<class T> bool operator()(const T* point,const T* center,T* r) const {
        r[0]=T(weight)*outsideBand(point[axis]-center[0],halfWidth);
        return true;
    }
};
struct RigidFiberBand {
    cv::Vec2d point;
    int axis;
    double halfWidth,weight;
    template<class T> bool operator()(const T* pose,const T* center,T* r) const {
        const T c=cos(pose[0]),s=sin(pose[0]);
        const T coordinate=axis==0?c*T(point[0])-s*T(point[1])+pose[1]
                                  :s*T(point[0])+c*T(point[1])+pose[2];
        r[0]=T(weight)*outsideBand(coordinate-center[0],halfWidth);
        return true;
    }
};
double axisRotation(const Placement& p) {
    const auto& n=*p.source;
    if(!n.fiber||(n.direction!='H'&&n.direction!='V'))return 0;
    const int first=(n.rows/2)*n.cols;
    const auto delta=p.uv[first+n.cols-1]-p.uv[first];
    if(cv::norm(delta)<1e-8)return 0;
    const double target=n.direction=='H'?0:std::acos(-1.0)/2;
    return std::remainder(target-std::atan2(delta[1],delta[0]),std::acos(-1.0));
}
struct RigidDirection {
    cv::Vec2d delta;
    int axis;
    double weight;
    template<class T> bool operator()(const T* pose,T* r) const {
        const T c=cos(pose[0]),s=sin(pose[0]);
        r[0]=T(weight)*(axis==0?c*T(delta[0])-s*T(delta[1]):s*T(delta[0])+c*T(delta[1]));
        return true;
    }
};
struct RigidMatch {
    cv::Vec2d a,b;
    double weight;
    template<class T> bool operator()(const T* pa,const T* pb,T* r) const {
        const T ca=cos(pa[0]),sa=sin(pa[0]),cb=cos(pb[0]),sb=sin(pb[0]);
        r[0]=T(weight)*(ca*T(a[0])-sa*T(a[1])+pa[1]-cb*T(b[0])+sb*T(b[1])-pb[1]);
        r[1]=T(weight)*(sa*T(a[0])+ca*T(a[1])+pa[2]-sb*T(b[0])-cb*T(b[1])-pb[2]);
        return true;
    }
};
struct Fidelity {
    cv::Vec3d target;double weight;
    template<class T> bool operator()(const T* p,T* r) const {
        for(int i=0;i<3;++i) r[i]=T(weight)*(p[i]-T(target[i]));return true;
    }
};
struct HorizontalAnchor {
    cv::Vec4d weights;
    cv::Vec3d target;
    double weight;
    template<class T> bool operator()(const T* a,const T* b,const T* c,const T* d,T* r) const {
        for(int i=0;i<3;++i)
            r[i]=T(weight)*(T(weights[0])*a[i]+T(weights[1])*b[i]+T(weights[2])*c[i]+T(weights[3])*d[i]-T(target[i]));
        return true;
    }
};
struct Cancellation : ceres::IterationCallback {
    const std::atomic_bool* cancel;
    Progress progress;
    std::string phase;
    int limit;
    std::chrono::steady_clock::time_point last{};
    explicit Cancellation(const std::atomic_bool* c,Progress p={},std::string name={},int iterations=0)
        :cancel(c),progress(std::move(p)),phase(std::move(name)),limit(iterations){}
    ceres::CallbackReturnType operator()(const ceres::IterationSummary& iteration) override {
        if(cancel&&cancel->load())return ceres::SOLVER_ABORT;
        const auto now=std::chrono::steady_clock::now();
        if(progress&&(iteration.iteration==0||iteration.iteration==limit||now-last>=std::chrono::milliseconds(500))) {
            progress(phase+": iteration "+std::to_string(iteration.iteration)+" (limit "+std::to_string(limit)+", may finish earlier)");
            last=now;
        }
        return ceres::SOLVER_CONTINUE;
    }
};
// Local edits fit onto the accepted fibers instead of bending them. A patch
// has far more chart area near a contact than a seven-row ribbon, so with
// equal weights it wins and the fiber curves inside its dead band; later
// edits then inherit that curvature. In a local fit, all fibers resist
// chart deformation ten times harder (a hundredfold in cost) and an added
// patch half as hard. Source XYZ is never rewritten; only the flattened
// placement changes. The full solver keeps its uniform weights: with stiff
// fibers it would have to translate them to satisfy contacts, which a local
// fit's frozen surroundings forbid, and its refit layouts rejected proposals
// that the local weighting accepts.
double tunable(const char* name,double fallback) {
    // Experiment overrides only; the defaults are the documented behavior.
    if(const auto* value=std::getenv(name))return std::atof(value);
    return fallback;
}
const double referenceFiberWeight=tunable("VC3D_CONSTRAINT_FIT_FIBER_WEIGHT",10);
const double addedPatchWeight=tunable("VC3D_CONSTRAINT_FIT_PATCH_WEIGHT",.5);
int solverThreads() {
    // Opt-in for experiments: parallel residual/Jacobian evaluation only.
    if(const auto* value=std::getenv("VC3D_CONSTRAINT_FIT_THREADS"))return std::max(1,std::atoi(value));
    return 1;
}
ceres::Solver::Options solverOptions(Cancellation& callback,int iterations) {
    ceres::Solver::Options options;options.linear_solver_type=ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads=solverThreads();options.max_num_iterations=iterations;options.callbacks.push_back(&callback);
    options.function_tolerance=1e-7;options.gradient_tolerance=1e-8;
    return options;
}
void profileSolve(const char* stage,const ceres::Solver::Summary& summary) {
    if(std::getenv("VC3D_PROFILE_CONSTRAINT_FIT"))
        std::clog<<"Constraint fit: "<<stage<<'\n'<<summary.FullReport()<<std::endl;
}
double triangleOrientation(const Placement& p,const cv::Vec3i& t) {
    return detail::cross(p.uv[t[1]]-p.uv[t[0]],p.uv[t[2]]-p.uv[t[0]]);
}
std::optional<cv::Vec3d> barycentric(const cv::Vec2d& p,const cv::Vec2d& a,const cv::Vec2d& b,const cv::Vec2d& c) {
    auto u=b-a,v=c-a,q=p-a;double det=detail::cross(u,v);if(std::abs(det)<1e-10) return {};
    double x=detail::cross(q,v)/det,y=detail::cross(u,q)/det;
    if(x< -1e-7||y< -1e-7||x+y>1+1e-7) return {};
    return cv::Vec3d{1-x-y,x,y};
}
}
std::optional<Location> pick(const Graph& graph,const cv::Vec2d& uv,const std::string& only) {
    // Patches take precedence over the generated support around a fiber.
    for(bool fiber:{false,true}) for(const auto& [id,p]:graph.nodes) {
        if(p.source->fiber!=fiber||(!only.empty()&&id!=only)) continue;
        const auto& n=*p.source;
        for(const auto& t:n.triangles) {
            auto w=barycentric(uv,p.uv[t[0]],p.uv[t[1]],p.uv[t[2]]);if(!w) continue;
            cv::Vec2d grid{0,0};for(int i=0;i<3;++i) grid+=cv::Vec2d{double(t[i]%n.cols),double(t[i]/n.cols)}*(*w)[i];
            return Location{id,grid};
        }
    }
    return {};
}
FitResult renderProposalSources(const Graph& draft,const Graph& committed,const std::string& selected,const std::atomic_bool* cancel) {
    Graph graph;graph.settings=draft.settings;
    for(const auto& [id,p]:draft.nodes)if(!committed.nodes.contains(id)||id==selected)graph.nodes.emplace(id,p);
    for(const auto& l:draft.links)if(graph.nodes.contains(l.a.node)&&graph.nodes.contains(l.b.node))graph.links.push_back(l);
    if(graph.nodes.empty())return {};
    graph.seed=graph.nodes.begin()->first;return renderPlacement(std::move(graph),cancel);
}
FitResult renderProposalSources(const GraphEdit& edit,const std::string& selected,const std::atomic_bool* cancel){
    Graph graph;graph.settings=edit.base->accepted.graph.settings;graph.nodes=edit.addedNodes;
    if(auto it=edit.base->accepted.graph.nodes.find(selected);it!=edit.base->accepted.graph.nodes.end())graph.nodes.emplace(*it);
    for(const auto& link:edit.addedLinks)if(graph.nodes.contains(link.a.node)&&graph.nodes.contains(link.b.node))graph.links.push_back(link);
    if(graph.nodes.empty())return {};
    graph.seed=graph.nodes.begin()->first;return renderPlacement(std::move(graph),cancel);
}
FitResult renderSourceLayer(const Graph& fitted,bool fibers,const std::atomic_bool* cancel) {
    Graph graph;graph.settings=fitted.settings;
    for(const auto& [id,p]:fitted.nodes)if(p.source->fiber==fibers)graph.nodes.emplace(id,p);
    for(const auto& link:fitted.links)if(graph.nodes.contains(link.a.node)&&graph.nodes.contains(link.b.node))graph.links.push_back(link);
    if(graph.nodes.empty())return {};
    graph.seed=graph.nodes.begin()->first;
    return renderPlacement(std::move(graph),cancel);
}
namespace {
FitResult renderSurface(Graph graph,const std::atomic_bool* cancel,bool refine,LocalFit* local=nullptr,
                        const cv::Mat_<cv::Vec3f>* saved=nullptr,const cv::Vec2d* savedOrigin=nullptr,const Progress& progress={}) {
    FitResult result;result.graph=std::move(graph);auto& g=result.graph;
    if(g.nodes.empty()) {result.message="Graph is empty";return result;}
    const auto ownedGeometry=local&&local->geometry?nullptr:std::make_shared<detail::FiberJunctions>(g,!saved);
    const auto& junctions=local&&local->geometry?*local->geometry:*ownedGeometry;
    double spacing=g.settings.spacing;
    cv::Vec2d low{INFINITY,INFINITY},high{-INFINITY,-INFINITY};
    std::vector<double> stretches;
    std::map<std::string,detail::ChartIndex> changedIndexes;
    auto chartIndex=[&](const std::string& id,const Placement& p)->const detail::ChartIndex&{
        const auto before=local->committed.graph.nodes.find(id);
        if(before!=local->committed.graph.nodes.end()&&p.uv.sharesStorage(before->second.uv))return local->prepared->charts.at(id);
        return changedIndexes.try_emplace(id,p).first->second;
    };
    for(const auto& [id,p]:g.nodes) {
        if(p.uv.size()!=p.source->xyz.size()) throw std::runtime_error("Source/placement size mismatch");
        const auto& xyz=junctions.points(*p.source);
        if(local&&local->prepared&&!local->added(id)){
            const auto& before=local->committed.graph.nodes.at(id);
            if(p.uv.sharesStorage(before.uv)&&&xyz==&local->prepared->junctions->points(*p.source)){
                const auto& bounds=local->prepared->charts.at(id).bounds;
                for(int d=0;d<2;++d){low[d]=std::min(low[d],bounds.low[d]);high[d]=std::max(high[d],bounds.high[d]);}
                const auto& cached=local->prepared->chartStretches.at(id);stretches.insert(stretches.end(),cached.begin(),cached.end());
                continue;
            }
        }
        double sign=0;
        for(const auto& t:p.source->triangles) {
            auto det=triangleOrientation(p,t);
            if(!std::isfinite(det)||std::abs(det)<1e-6||(sign&&det*sign<=0)) {
                result.message="Collapsed or inverted chart: "+id;return result;
            }
            sign=det;
            for(int i=0;i<3;++i) {
                auto uv=p.uv[t[i]];
                if(!std::isfinite(uv[0])||!std::isfinite(uv[1])) {result.message="Nonfinite placement";return result;}
                for(int d=0;d<2;++d){low[d]=std::min(low[d],uv[d]);high[d]=std::max(high[d],uv[d]);}
                double dist=cv::norm(xyz[t[(i+1)%3]]-xyz[t[i]]);
                double flat=cv::norm(p.uv[t[(i+1)%3]]-uv);
                if(dist>1e-8&&flat>1e-8) stretches.push_back(std::max(flat/dist,dist/flat));
            }
        }
    }
    if(!std::isfinite(low[0])){result.message="No valid triangles";return result;}
    for(int d=0;d<2;++d) low[d]=std::floor(low[d]/spacing)*spacing-spacing;
    if(saved) {
        if(saved->rows<2||saved->cols<2||saved->total()>16000000){result.message="Invalid saved surface dimensions";return result;}
        low=*savedOrigin;high=low+cv::Vec2d{double(saved->cols-2)*spacing,double(saved->rows-2)*spacing};
    }
    const auto fullLow=low,fullHigh=high;
    if(local) {
        auto bounds=local->region.padded(2*spacing);
        for(int d=0;d<2;++d){low[d]=std::floor(bounds.low[d]/spacing)*spacing;high[d]=std::ceil(bounds.high[d]/spacing)*spacing;}
    }
    double width=std::ceil((high[0]-low[0])/spacing)+2,height=std::ceil((high[1]-low[1])/spacing)+2;
    if(width<2||height<2||width*height>16000000) {result.message="Graph bounds exceed the 16-million-sample grid limit";return result;}
    int cols=int(width),rows=int(height);size_t count=size_t(cols)*rows;
    auto movable=[&](int i){return !local||local->region.contains(low+cv::Vec2d{double(i%cols)*spacing,double(i/cols)*spacing});};
    auto previous=[&](int i)->cv::Vec3d {
        const auto uv=low+cv::Vec2d{double(i%cols)*spacing,double(i/cols)*spacing};
        const auto at=(uv-local->committed.origin)/spacing;
        const auto& grid=*local->committed.surface->rawPointsPtr();
        const int x=int(std::lround(at[0])),y=int(std::lround(at[1]));
        return x>=0&&y>=0&&x<grid.cols&&y<grid.rows?cv::Vec3d(grid(y,x)):cv::Vec3d{-1,-1,-1};
    };
    if(progress)progress("Building output surface samples");
    auto rasterProgress=std::chrono::steady_clock::now();
    RasterSamples samples(count);
    std::vector<unsigned char> coverage(count,0);
    std::vector<std::string> ids;
    std::set<std::pair<std::string,std::string>> linked;
    if(local&&local->edit)linked=detail::editedPairs(*local->edit);
    else for(const auto& l:g.links.read()) linked.insert(std::minmax(l.a.node,l.b.node));
    size_t conflicts=0;int ordinal=0;
    // Synthetic ribbons are support only; real patches replace that support.
    for(bool fiber:{true,false}) for(const auto& [id,p]:g.nodes) {
        if(p.source->fiber!=fiber) continue;
        detail::cancelled(cancel);const auto& n=*p.source;int current=ordinal++;ids.push_back(id);
        const auto& geometry=junctions.points(n);
        const auto now=std::chrono::steady_clock::now();
        if(progress&&(ordinal==1||now-rasterProgress>=std::chrono::milliseconds(500))) {
            progress("Building output surface: source "+std::to_string(ordinal)+" / "+std::to_string(g.nodes.size()));rasterProgress=now;
        }
        std::vector<size_t> selectedTriangles;
        const bool indexed=local&&local->prepared;
        if(indexed)selectedTriangles=chartIndex(id,p).query({low,low+cv::Vec2d(cols-1,rows-1)*spacing},n);
        for(size_t ti=0;ti<(indexed?selectedTriangles.size():n.triangles.size());++ti) {
            const auto& t=n.triangles[indexed?selectedTriangles[ti]:ti];
            cv::Vec2d a=(p.uv[t[0]]-low)/spacing,b=(p.uv[t[1]]-low)/spacing,c=(p.uv[t[2]]-low)/spacing;
            auto normal=detail::unit((geometry[t[1]]-geometry[t[0]]).cross(geometry[t[2]]-geometry[t[0]]));
            if(detail::cross(b-a,c-a)<0) normal*=-1;
            int x0=std::max(0,int(std::ceil(std::min({a[0],b[0],c[0]}))));
            int x1=std::min(cols-1,int(std::floor(std::max({a[0],b[0],c[0]}))));
            int y0=std::max(0,int(std::ceil(std::min({a[1],b[1],c[1]}))));
            int y1=std::min(rows-1,int(std::floor(std::max({a[1],b[1],c[1]}))));
            for(int y=std::max(0,y0-1);y<=std::min(rows-2,y1);++y) for(int x=std::max(0,x0-1);x<=std::min(cols-2,x1);++x) {
                for(int q=0;q<4;++q)if(barycentric({x+(q%2?.75:.25),y+(q/2?.75:.25)},a,b,c))coverage[size_t(y)*cols+x]|=1<<q;
            }
            for(int y=y0;y<=y1;++y) for(int x=x0;x<=x1;++x) {
                auto w=barycentric({double(x),double(y)},a,b,c);if(!w)continue;
                size_t index=size_t(y)*cols+x;
                auto& sample=samples.insert(index);if(sample.visited==current)continue;
                sample.visited=current;
                auto xyz=n.xyz[t[0]]*(*w)[0]+n.xyz[t[1]]*(*w)[1]+n.xyz[t[2]]*(*w)[2];
                double sourceColumn=0;for(int j=0;j<3;++j)sourceColumn+=(t[j]%n.cols)*(*w)[j];
                xyz+=junctions.offset(n,sourceColumn);
                double weight=1;
                if(fiber) {
                    double sourceRow=0;for(int j=0;j<3;++j) sourceRow+=(t[j]/n.cols)*(*w)[j];
                    weight=1/(1+std::abs(sourceRow-n.rows/2));
                }
                if(sample.owner>=0&&!fiber&&g.nodes.at(ids[sample.owner]).source->fiber) {sample.mass=0;sample.sum={0,0,0};sample.normal={0,0,0};}
                if(sample.mass>0&&!fiber&&cv::norm(sample.sum/sample.mass-xyz)>g.settings.radius &&
                   !linked.contains(std::minmax(id,ids[sample.owner]))) ++conflicts;
                sample.sum+=xyz*weight;sample.normal+=normal*weight;sample.mass+=weight;sample.owner=current;
            }
        }
    }
    if(conflicts) {result.message="Unlinked patches collide in the flattened view ("+std::to_string(conflicts)+" samples)";return result;}
    // A complete set of vertices does not prove that their cell has source
    // support. Remove one boundary vertex for each uncovered cell so strict
    // QuadSurface rendering cannot bridge across a gap between ribbons.
    std::set<int> unsupported;
    for(int y=0;y+1<rows;++y)for(int x=0;x+1<cols;++x) {
        int a=y*cols+x,b=a+1,c=a+cols,d=c+1;
        if(coverage[a]==15||!samples[a].mass||!samples[b].mass||!samples[c].mass||!samples[d].mass)continue;
        int victim=a;for(int i:{b,c,d})if(samples[i].mass<samples[victim].mass)victim=i;unsupported.insert(victim);
    }
    for(int i:unsupported) {
        if(local&&!movable(i)&&detail::valid(previous(i))){local->expand=true;result.message="Surface support reaches the local boundary";return result;}
        samples.at(i).mass=0;
    }
    if(local)for(size_t i=0;i<count;++i)if(!movable(int(i))) {
        const auto old=previous(int(i));
        if(detail::valid(old)&&samples[i].owner<0){local->expand=true;result.message="Surface moved across the local boundary";return result;}
        if(samples[i].owner>=0)samples.at(i).mass=detail::valid(old)?1:0;
    }
    if(saved) {
        for(size_t i=0;i<count;++i) {
            const auto p=(*saved)(int(i)/cols,int(i)%cols);
            if(p[0]!=-1&&(!detail::valid(cv::Vec3d(p))||samples[i].owner<0)){result.message="Saved surface has unsupported samples";return result;}
            if(samples[i].owner>=0)samples.at(i).mass=p[0]!=-1?1:0;
        }
        for(int y=0;y+1<rows;++y)for(int x=0;x+1<cols;++x) {
            int a=y*cols+x;
            if(coverage[a]!=15&&samples[a].mass&&samples[a+1].mass&&samples[a+cols].mass&&samples[a+cols+1].mass){result.message="Saved surface bridges a source gap";return result;}
        }
    }
    for(size_t i=0;i<count;++i)if(samples[i].mass) {
        auto& s=samples.at(i);
        s.point=saved?cv::Vec3d((*saved)(int(i)/cols,int(i)%cols)):(local&&!movable(int(i))?previous(int(i)):s.sum/s.mass);
        s.target=s.point;
        // The accepted surface around a local edit was relaxed by an earlier
        // solve and sits up to tens of voxels from raw raster positions. Fresh
        // raster samples next to that frozen ring start the relaxation from a
        // torn grid, which it repairs with large moves that can fold a patch.
        // Start free samples where the accepted surface already is; fidelity
        // still pulls them toward the new raster positions.
        if(refine&&local&&movable(int(i))) {
            const auto old=previous(int(i));
            if(detail::valid(old))s.point=old;
        }
    }
    // Preserve the dense grid's parameter address order as well as residual
    // order, so Ceres uses the same ordering in its sparse factorization.
    if(refine)samples.orderByGrid();
    // No insertions after this point: Ceres retains pointers into the samples.
    auto point=[&](int i)->cv::Vec3d& {return samples.at(i).point;};
    // Joint XYZ fit: every overlapping source contributes to the same sample.
    // Preserve its measured position while regularizing the shared grid metric.
    ceres::Problem problem;
    std::vector<cv::Vec3i> triangles;
    for(int y=0;y+1<rows;++y)for(int x=0;x+1<cols;++x) {
        int a=y*cols+x,b=a+1,c=a+cols,d=c+1;
        if(!samples[a].mass||!samples[b].mass||!samples[c].mass||!samples[d].mass) continue;
        triangles.push_back({a,b,c});triangles.push_back({d,c,b});
    }
    // Scan in index order, as in the former set (including residual order).
    for(const auto& t:triangles)for(int i:t.val)samples.at(i).active=true;
    std::vector<int> active;
    active.reserve(std::min(count,triangles.size()*3));
    for(size_t i=0;i<count;++i)if(samples[i].active)active.push_back(int(i));
    if(active.empty()){result.message="No complete quads after rasterization";return result;}
    if(refine) {
        for(int i:active)if(movable(i))problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Fidelity,3,3>(new Fidelity{samples[i].target,5.0/spacing}),nullptr,point(i).val);
        for(const auto& t:triangles) problem.AddResidualBlock(vc::SymmetricDirichletLoss::Create(spacing,g.settings.stiffness,1e-8,0),nullptr,point(t[0]).val,point(t[1]).val,point(t[2]).val);
        // Rasterization and metric regularization must not pull an explicit
        // H/V junction back between the front and back of the sheet. Anchor
        // the rendered bilinear sample itself to the horizontal position.
        std::vector<Link> junctionLinks;
        if(local&&local->edit){
            const std::set<size_t> removed(local->edit->removedLinks.begin(),local->edit->removedLinks.end());
            for(size_t i:local->prepared->junctionLinks)if(!removed.contains(i))junctionLinks.push_back(local->committed.graph.links[i]);
            junctionLinks.insert(junctionLinks.end(),local->edit->addedLinks.begin(),local->edit->addedLinks.end());
        }
        for(const auto& link:local&&local->edit?junctionLinks:g.links.read()) {
            const auto [h,v]=detail::horizontalVerticalLink(g,link);if(!h)continue;
            const auto& horizontal=*g.nodes.at(h->node).source;
            const auto at=(placedLocation(g,*h)-low)/spacing;
            const int x=int(std::floor(at[0])),y=int(std::floor(at[1]));
            if(x<0||y<0||x+1>=cols||y+1>=rows)continue;
            const int a=y*cols+x,b=a+1,c=a+cols,d=c+1;
            if(!samples[a].mass||!samples[b].mass||!samples[c].mass||!samples[d].mass)continue;
            // Measured patches retain their existing precedence over ribbons.
            bool patch=false;for(int i:{a,b,c,d})patch|=!g.nodes.at(ids[samples[i].owner]).source->fiber;
            if(patch||(!movable(a)&&!movable(b)&&!movable(c)&&!movable(d)))continue;
            const double u=at[0]-x,w=at[1]-y;
            const auto target=detail::interpolate(horizontal.xyz,detail::weights(horizontal,*h));
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<HorizontalAnchor,3,3,3,3,3>(
                new HorizontalAnchor{{(1-u)*(1-w),u*(1-w),(1-u)*w,u*w},target,1000.0/spacing}),
                nullptr,point(a).val,point(b).val,point(c).val,point(d).val);
        }
        if(local)for(int i:active)if(!movable(i))problem.SetParameterBlockConstant(point(i).val);
        Cancellation cb(cancel,progress,"Refining output surface",12);auto options=solverOptions(cb,12);ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);profileSolve("surface",summary);detail::cancelled(cancel);
        if(!summary.IsSolutionUsable()){result.message="Shared surface fit failed: "+summary.BriefReport();return result;}
    }
    if(local) {
        // The last free ring must have settled before freezing the outside.
        for(int i:active)if(movable(i)) {
            bool boundary=false;
            for(int offset:{-1,1,-cols,cols})if(i+offset>=0&&i+offset<int(count)&&!movable(i+offset)) {
                boundary=true;const auto oldA=previous(i),oldB=previous(i+offset);
                if(!detail::valid(oldA)||!detail::valid(oldB))continue;
                const double before=cv::norm(oldA-oldB),after=cv::norm(point(i)-point(i+offset));
                if(before>1e-8&&(after>before*1.02||after<before/1.02))local->expand=true;
            }
            if(boundary&&detail::valid(previous(i))&&cv::norm(point(i)-previous(i))>g.settings.radius*.25)local->expand=true;
        }
        if(local->expand){result.message="Surface adjustment reaches the local boundary";return result;}
        result.localSurfaceSamples=std::count_if(active.begin(),active.end(),movable);
        result.changedSurfaceBounds=std::array<cv::Vec2d,2>{local->region.low,local->region.high};
    }
    cv::Mat_<cv::Vec3f> grid=saved?saved->clone():cv::Mat_<cv::Vec3f>(rows,cols,cv::Vec3f{-1,-1,-1});
    for(int i:active) grid(i/cols,i%cols)=cv::Vec3f(point(i));
    // A checksum-verified saved grid has already passed fit validation. Its
    // fused/corrected normals can differ from the immutable source ribbons;
    // reopening must not prune it again against those original normals.
    if(!saved)for(const auto& t:triangles) {
        auto cross=(point(t[1])-point(t[0])).cross(point(t[2])-point(t[0]));
        auto reference=samples[t[0]].normal+samples[t[1]].normal+samples[t[2]].normal;
        if(cv::norm(cross)<1e-6||cross.dot(reference)<=0) {
            bool support=false;for(int i:t.val) support|=g.nodes.at(ids[samples[i].owner]).source->fiber;
            support|=samples[t[0]].owner!=samples[t[1]].owner||samples[t[0]].owner!=samples[t[2]].owner;
            // A patch whose rasterized samples already fold is rejected: its
            // chart or source is wrong. A fold that only the relaxation
            // introduced is a solver artifact; prune that cell like an
            // unsupported one instead of rejecting the whole fit.
            const auto rasterCross=(samples[t[1]].target-samples[t[0]].target).cross(samples[t[2]].target-samples[t[0]].target);
            const bool rasterFolded=cv::norm(rasterCross)<1e-6||rasterCross.dot(reference)<=0;
            if(!support&&rasterFolded) {
                if(std::getenv("VC3D_PROFILE_CONSTRAINT_FIT")) {
                    std::clog<<"Inverted patch quad: "<<ids[samples[t[0]].owner]<<" grid=("<<t[0]%cols<<","<<t[0]/cols<<")"
                        <<" cos="<<cross.dot(reference)/(cv::norm(cross)*std::max(1e-12,cv::norm(reference)))
                        <<" |normal|/mass="<<cv::norm(reference)/(samples[t[0]].mass+samples[t[1]].mass+samples[t[2]].mass)
                        <<" edges="<<cv::norm(point(t[1])-point(t[0]))<<","<<cv::norm(point(t[2])-point(t[0]))
                        <<" raster="<<cv::norm(samples[t[1]].sum/samples[t[1]].mass-samples[t[0]].sum/samples[t[0]].mass)<<","<<cv::norm(samples[t[2]].sum/samples[t[2]].mass-samples[t[0]].sum/samples[t[0]].mass)
                        <<" moved="<<cv::norm(point(t[0])-samples[t[0]].sum/samples[t[0]].mass)<<std::endl;
                }
                result.message="Patch geometry contains an inverted quad";return result;
            }
            int victim=t[0];for(int i:t.val)if(samples[i].mass<samples[victim].mass)victim=i;
            if(local&&!movable(victim)){local->expand=true;result.message="Quad validation reaches the local boundary";return result;}
            grid(victim/cols,victim%cols)={-1,-1,-1};unsupported.insert(victim);
        }
    }
    if(local) {
        const auto& old=*local->committed.surface->rawPointsPtr();
        cv::Vec2d mergedLow,mergedHigh;
        for(int d=0;d<2;++d) {
            mergedLow[d]=std::min(fullLow[d],local->committed.origin[d]);
            const int extent=d==0?old.cols:old.rows;
            mergedHigh[d]=std::max(std::ceil((fullHigh[d]-mergedLow[d])/spacing)*spacing+mergedLow[d]+spacing,
                                   local->committed.origin[d]+(extent-1)*spacing);
        }
        int mergedCols=int(std::lround((mergedHigh[0]-mergedLow[0])/spacing))+1;
        int mergedRows=int(std::lround((mergedHigh[1]-mergedLow[1])/spacing))+1;
        if(int64_t(mergedRows)*mergedCols>16000000){result.message="Graph bounds exceed the 16-million-sample grid limit";return result;}
        cv::Mat_<cv::Vec3f> merged(mergedRows,mergedCols,cv::Vec3f{-1,-1,-1});
        const auto oldOffset=(local->committed.origin-mergedLow)/spacing;
        old.copyTo(merged(cv::Rect(int(std::lround(oldOffset[0])),int(std::lround(oldOffset[1])),old.cols,old.rows)));
        const auto offset=(low-mergedLow)/spacing;
        for(int y=0;y<rows;++y)for(int x=0;x<cols;++x)if(movable(y*cols+x)) {
            const int ox=x+int(std::lround(offset[0])),oy=y+int(std::lround(offset[1]));
            if(ox>=0&&oy>=0&&ox<mergedCols&&oy<mergedRows)merged(oy,ox)=grid(y,x);
        }
        grid=std::move(merged);low=mergedLow;cols=mergedCols;rows=mergedRows;
    }
    size_t completeQuads=0;
    if(local&&local->prepared){
        const auto first=detail::tileAt(low,spacing),last=detail::tileAt(low+cv::Vec2d(cols-2,rows-2)*spacing,spacing);
        const auto dirty=local->region.padded(spacing);
        for(int y=first.second;y<=last.second;++y)for(int x=first.first;x<=last.first;++x){
            const detail::TileKey key{x,y};const Region tile{{x*64*spacing,y*64*spacing},{(x+1)*64*spacing,(y+1)*64*spacing}};
            const auto cached=local->prepared->surfaceTiles.find(key);
            detail::SurfaceTileStats computed;
            const auto* stats=cached==local->prepared->surfaceTiles.end()?nullptr:&cached->second;
            if(!stats||tile.intersects(dirty)){computed=detail::surfaceTileStats(grid,low,spacing,key);stats=&computed;}
            completeQuads+=stats->quads;stretches.insert(stretches.end(),stats->stretches.begin(),stats->stretches.end());
        }
    }else for(int y=0;y+1<rows;++y)for(int x=0;x+1<cols;++x) {
        if(grid(y,x)[0]==-1||grid(y,x+1)[0]==-1||grid(y+1,x)[0]==-1||grid(y+1,x+1)[0]==-1)continue;
        ++completeQuads;
        for(const auto edge:std::array<std::pair<cv::Vec3f,cv::Vec3f>,2>{{{grid(y,x),grid(y,x+1)},{grid(y,x),grid(y+1,x)}}}) {
            const double length=cv::norm(edge.second-edge.first);
            if(length>1e-8)stretches.push_back(std::max(length/spacing,spacing/length));
        }
    }
    if(!completeQuads){result.message="No complete quads remain after boundary validation";return result;}
    result.surface=std::make_shared<QuadSurface>(grid,cv::Vec2f{float(1/spacing),float(1/spacing)});
    result.surface->setStrictQuadRenderValidity(true);
    // QuadSurface defaults to a centered origin; use absolute graph UV instead.
    auto defaultOrigin=result.surface->gridToSurface({0,0});
    result.surface->shiftSurfaceOrigin(defaultOrigin-low);result.origin=low;
    if(!stretches.empty()) {auto index=stretches.begin()+static_cast<size_t>(0.95*(stretches.size()-1));std::nth_element(stretches.begin(),index,stretches.end());result.stretchP95=*index;}
    result.correspondenceRms=attachmentRms(g,local);
    result.valid=true;result.message=unsupported.empty()?"Ready":"Ready; "+std::to_string(unsupported.size())+" unsupported boundary samples omitted";return result;
}
} // namespace
FitResult detail::restoreSurface(Graph graph,const cv::Mat_<cv::Vec3f>& grid,cv::Vec2d origin) {
    return renderSurface(std::move(graph),nullptr,false,nullptr,&grid,&origin);
}
FitResult render(Graph graph,const std::atomic_bool* cancel) {return renderSurface(std::move(graph),cancel,true);}
FitResult renderPlacement(Graph graph,const std::atomic_bool* cancel) {return renderSurface(std::move(graph),cancel,false);}

namespace {
void alignRigid(Graph& graph,const std::atomic_bool* cancel,const Progress& progress,bool alignAxes,LocalFit* local) {
    const double bandHalfWidth=2*graph.settings.spacing;
    const double bandWeight=2/graph.settings.spacing;
    // Close component loops with rigid node motions before allowing vertex
    // deformation. Large translations in the full solve otherwise encounter
    // the no-flip barrier long before the branch anchors can meet.
    if(graph.nodes.size()>1) {
        if(progress)progress("Aligning linked source charts");
        std::map<std::string,cv::Vec3d> poses;
        std::map<std::string,double> centers;
        ceres::Problem rigid;
        for(const auto& [id,p]:graph.nodes) {
            const double angle=!alignAxes||id==graph.seed||(local&&!local->added(id))?0:axisRotation(p);
            poses[id]={angle,0,0};rigid.AddParameterBlock(poses[id].val,3);
            if(local&&!local->added(id)){rigid.SetParameterBlockConstant(poses[id].val);continue;}
            const auto& n=*p.source;
            if(!alignAxes||!n.fiber||(n.direction!='H'&&n.direction!='V'))continue;
            const int axis=n.direction=='V'?0:1;
            const int first=(n.rows/2)*n.cols,last=first+n.cols-1;
            rigid.AddResidualBlock(new ceres::AutoDiffCostFunction<RigidDirection,1,3>(
                new RigidDirection{p.uv[last]-p.uv[first],axis,30/graph.settings.spacing}),nullptr,poses[id].val);
            double center=0;
            for(int x=0;x<n.cols;++x) {
                auto uv=p.uv[(n.rows/2)*n.cols+x];
                center+=axis==0?std::cos(angle)*uv[0]-std::sin(angle)*uv[1]
                              :std::sin(angle)*uv[0]+std::cos(angle)*uv[1];
            }
            centers[id]=center/n.cols;
            for(int x=0;x<n.cols;++x)
                rigid.AddResidualBlock(new ceres::AutoDiffCostFunction<RigidFiberBand,1,3,1>(
                    new RigidFiberBand{p.uv[(n.rows/2)*n.cols+x],axis,bandHalfWidth,bandWeight}),nullptr,poses[id].val,&centers[id]);
        }
        for(const auto& l:fitContacts(graph,local)) {
            if(l.a.node==l.b.node||(local&&!local->added(l.a.node)&&!local->added(l.b.node)))continue;
            auto [wa,wb]=detail::linkWeights(graph,l);
            auto a=detail::interpolate(graph.nodes.at(l.a.node).uv,wa);
            auto b=detail::interpolate(graph.nodes.at(l.b.node).uv,wb);
            rigid.AddResidualBlock(new ceres::AutoDiffCostFunction<RigidMatch,2,3,3>(
                new RigidMatch{a,b,(l.keypoint?1000.0:30.0)/graph.settings.spacing}),nullptr,poses[l.a.node].val,poses[l.b.node].val);
        }
        rigid.SetParameterBlockConstant(poses.at(graph.seed).val);
        Cancellation cb(cancel,progress,"Aligning patches and fibers",100);auto options=solverOptions(cb,100);ceres::Solver::Summary summary;
        ceres::Solve(options,&rigid,&summary);profileSolve("rigid",summary);detail::cancelled(cancel);
        if(!summary.IsSolutionUsable())throw std::runtime_error("Rigid graph alignment failed: "+summary.BriefReport());
        for(auto& [id,p]:graph.nodes) {
            if(local&&!local->added(id))continue;
            const auto pose=poses.at(id);const double c=std::cos(pose[0]),s=std::sin(pose[0]);
            for(auto& uv:p.uv)uv={c*uv[0]-s*uv[1]+pose[1],s*uv[0]+c*uv[1]+pose[2]};
        }
    }
}
FitResult fitCharts(Graph graph,const std::atomic_bool* cancel,Progress progress,bool alignAxes,LocalFit* local=nullptr) {
    graph.settings.validate();detail::cancelled(cancel);
    if(graph.nodes.empty())return render(std::move(graph),cancel);
    if(!local||!local->prepared) {
    std::set<std::string> connected{graph.seed};
    for(size_t pass=0;pass<graph.nodes.size();++pass) {
        const auto before=connected.size();
        for(const auto& l:fitContacts(graph,local)) {
            if(!graph.nodes.contains(l.a.node)||!graph.nodes.contains(l.b.node))throw std::runtime_error("Link refers to a missing node");
            if(connected.contains(l.a.node))connected.insert(l.b.node);
            if(connected.contains(l.b.node))connected.insert(l.a.node);
        }
        if(connected.size()==before)break;
    }
    if(!graph.nodes.contains(graph.seed)||connected.size()!=graph.nodes.size()) {
        FitResult failed;failed.graph=std::move(graph);failed.message="Every node must remain linked to the seed";return failed;
    }
    }
    // The positional gauge must not freeze a previously tilted seed. Rotate
    // the whole existing graph together before fixing that seed's rigid pose.
    const auto rotation=alignAxes&&!local?axisRotation(graph.nodes.at(graph.seed)):0;
    if(rotation!=0) {
        const double c=std::cos(rotation),s=std::sin(rotation);
        for(auto& [id,p]:graph.nodes)for(auto& uv:p.uv)uv={c*uv[0]-s*uv[1],s*uv[0]+c*uv[1]};
    }
    const double bandHalfWidth=2*graph.settings.spacing;
    const double bandWeight=2/graph.settings.spacing;
    alignRigid(graph,cancel,progress,alignAxes,local);
    if(progress)progress("Fitting connected graph");
    const auto ownedGeometry=local&&local->geometry?nullptr:std::make_shared<detail::FiberJunctions>(graph);
    const auto& junctions=local&&local->geometry?*local->geometry:*ownedGeometry;
    ceres::Problem problem;
    std::map<std::string,double> centers;
    for(auto& [id,p]:graph.nodes) {
        if(local&&local->prepared&&!local->movable.contains(id))continue;
        const auto& n=*p.source;
        const auto& geometry=junctions.points(n);
        const auto* movable=local?&local->movable.at(id):nullptr;
        const auto moves=[&](int i){return !movable||(*movable)[i]!=0;};
        for(const auto& t:fitTriangles(n,local)) {
            if(!moves(t[0])&&!moves(t[1])&&!moves(t[2]))continue;
            auto x=geometry[t[1]]-geometry[t[0]],y=geometry[t[2]]-geometry[t[0]];
            double length=cv::norm(x),projection=x.dot(y)/length;
            double height=std::sqrt(std::max(0.0,y.dot(y)-projection*projection));
            if(length<1e-6||height<1e-6)continue;
            double det=triangleOrientation(p,t);
            if(!std::isfinite(det)||std::abs(det)<1e-8) {FitResult failed;failed.graph=std::move(graph);failed.message="Degenerate initial chart: "+id;return failed;}
            double w=std::sqrt(graph.settings.stiffness*length*height/(2*graph.settings.spacing*graph.settings.spacing));
            if(local)w*=n.fiber?referenceFiberWeight:(local->added(id)?addedPatchWeight:1);
            auto* cost=new detail::ChartMetricCost({1/length,-projection/(length*height),1/height,w,det>0?1.0:-1.0});
            problem.AddResidualBlock(cost,nullptr,p.uv[t[0]].val,p.uv[t[1]].val,p.uv[t[2]].val);
        }
        if(alignAxes&&n.fiber&&(n.direction=='H'||n.direction=='V')) {
            const int axis=n.direction=='V'?0:1;
            const int first=(n.rows/2)*n.cols,last=first+n.cols-1;
            // The band permits local bends; it must not also permit an entire
            // fiber to tilt when a patch attaches. Tie its endpoint row/column.
            if((moves(first)||moves(last))&&problem.HasParameterBlock(p.uv[first].val)&&problem.HasParameterBlock(p.uv[last].val))
                problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Direction,1,2,2>(
                    new Direction{axis,30/graph.settings.spacing}),nullptr,p.uv[first].val,p.uv[last].val);
            double center=0;
            for(int x=0;x<n.cols;++x)center+=p.uv[(n.rows/2)*n.cols+x][axis];
            centers[id]=center/n.cols;
            for(int x=0;x<n.cols;++x) {
                auto& uv=p.uv[(n.rows/2)*n.cols+x];
                if(problem.HasParameterBlock(uv.val)&&moves((n.rows/2)*n.cols+x))
                    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<FiberBand,1,2,1>(
                        new FiberBand{axis,bandHalfWidth,bandWeight}),nullptr,uv.val,&centers[id]);
            }
        }
        if(n.fiber&&(n.direction=='H'||n.direction=='V'))for(int x=0;x+1<n.cols;++x) {
            int a=(n.rows/2)*n.cols+x,b=a+1;
            if((moves(a)||moves(b))&&
               problem.HasParameterBlock(p.uv[a].val)&&problem.HasParameterBlock(p.uv[b].val))
                problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Direction,1,2,2>(new Direction{n.direction=='V'?0:1,0.2/graph.settings.spacing}),nullptr,p.uv[a].val,p.uv[b].val);
        }
    }
    size_t contactIndex=0;
    for(const auto& l:fitContacts(graph,local)) {
        std::vector<double*> blocks;std::vector<double> coefficients;
        const auto weights=local&&local->prepared?local->weights[contactIndex++]:[&]{auto [a,b]=detail::linkWeights(graph,l);return detail::PreparedContact{a,b};}();
        const auto& wa=weights.a;const auto& wb=weights.b;
        if(local) {
            bool touches=false;
            for(int i=0;i<3;++i)touches|=local->moves(l.a.node,wa.indices[i])||local->moves(l.b.node,wb.indices[i]);
            if(!touches)continue;
        }
        auto add=[&](const Location& loc,const detail::Weights& w,double sign){auto& p=graph.nodes.at(loc.node);
            for(int i=0;i<3;++i){auto ptr=p.uv[w.indices[i]].val;auto it=std::find(blocks.begin(),blocks.end(),ptr);
                if(it==blocks.end()){blocks.push_back(ptr);coefficients.push_back(sign*w.values[i]);}
                else coefficients[it-blocks.begin()]+=sign*w.values[i];}};
        add(l.a,wa,1);add(l.b,wb,-1);
        auto* cost=new detail::MatchCost(std::move(coefficients),(l.keypoint?1000.0:30.0)/graph.settings.spacing);
        problem.AddResidualBlock(cost,nullptr,blocks);
    }
    if(graph.nodes.empty()) return render(std::move(graph),cancel);
    if(local) {
        for(auto& [id,p]:graph.nodes) {
            if(local->prepared&&!local->parameterNodes.contains(id))continue;
            for(size_t i=0;i<p.uv.size();++i)if(!local->moves(id,i)&&problem.HasParameterBlock(std::as_const(p.uv)[i].val))problem.SetParameterBlockConstant(p.uv[i].val);
            if(!local->added(id)&&centers.contains(id)&&problem.HasParameterBlock(&centers[id]))problem.SetParameterBlockConstant(&centers[id]);
        }
    } else {
        auto& seed=graph.nodes.at(graph.seed);
        if(seed.source->triangles.empty())throw std::runtime_error("Seed has no valid quads");
        auto anchor=seed.source->triangles.front();
        if(!problem.HasParameterBlock(seed.uv[anchor[0]].val)||!problem.HasParameterBlock(seed.uv[anchor[1]].val)) {
            FitResult failed;failed.graph=std::move(graph);failed.message="Seed triangles are too small to fit";return failed;
        }
        problem.SetParameterBlockConstant(seed.uv[anchor[0]].val);
        // One extra scalar removes the rotation gauge without freezing source scale.
        auto edge=seed.uv[anchor[1]]-seed.uv[anchor[0]];
        problem.SetManifold(seed.uv[anchor[1]].val,new ceres::SubsetManifold(2,{std::abs(edge[0])>=std::abs(edge[1])?1:0}));
    }
    Cancellation cb(cancel,progress,"Fitting flattened layout",60);auto options=solverOptions(cb,60);ceres::Solver::Summary summary;
    // ChartMetric deliberately rejects trial steps that flip a triangle.
    // Ceres retries them with a smaller trust region; logging every rejection
    // as a warning floods stderr on constrained fits. Keep this solve quiet;
    // unusable solutions and final geometry errors are still reported below.
    options.logging_type=ceres::SILENT;
    ceres::Solve(options,&problem,&summary);profileSolve("chart",summary);detail::cancelled(cancel);
    if(!summary.IsSolutionUsable()) {FitResult failed;failed.graph=std::move(graph);failed.message=summary.BriefReport();return failed;}
    for(const auto& l:fitContacts(graph,local))if(l.keypoint&&cv::norm(placedLocation(graph,l.a)-placedLocation(graph,l.b))>0.02) {
        FitResult failed;failed.graph=std::move(graph);failed.message="Keypoint constraints conflict; edit or remove a pair";return failed;
    }
    if(!alignAxes||local) {
        FitResult aligned;aligned.graph=std::move(graph);aligned.valid=true;
        return aligned;
    }
    if(progress)progress("Building shared tifxyz surface");
    auto result=renderSurface(std::move(graph),cancel,true,nullptr,nullptr,nullptr,progress);
    if(result.valid && result.correspondenceRms>result.graph.settings.radius) {
        result.valid=false;
        result.message="Attachment error exceeds proximity tolerance; add matching keypoint pairs";
    }
    return result;
}
} // namespace

FitResult fit(Graph graph,const std::atomic_bool* cancel,Progress progress) {
    if(!graph.nodes.contains(graph.seed))return fitCharts(std::move(graph),cancel,progress,true);
    // New linked components can start with very large loop-closure errors.
    // First let their source charts relax into a connected configuration;
    // imposing rows/columns on those errors directly can trap the no-flip
    // solve. Existing fitted graphs normally skip this initialization pass.
    double squared=0;
    for(const auto& link:graph.links.read()) {
        detail::cancelled(cancel);
        const auto error=detail::linkError(graph,link);squared+=error.dot(error);
    }
    if(!graph.links.empty()&&squared>graph.links.size()*graph.settings.radius*graph.settings.radius) {
        graph.settings.validate();
        // Large placement errors often need only rigid registration. Test that
        // inexpensive result before paying for a second full vertex solve.
        alignRigid(graph,cancel,progress,false,nullptr);
        squared=0;
        for(const auto& link:graph.links.read()) {
            detail::cancelled(cancel);
            const auto error=detail::linkError(graph,link);squared+=error.dot(error);
        }
        if(squared>graph.links.size()*graph.settings.radius*graph.settings.radius) {
            if(progress)progress("Initializing linked source geometry");
            auto initialized=fitCharts(std::move(graph),cancel,progress,false);
            if(!initialized.valid)return initialized;
            graph=std::move(initialized.graph);
        }
    }
    return fitCharts(std::move(graph),cancel,progress,true);
}
namespace {
auto linkKey(const Link& l) {
    return detail::contactKey(l);
}
struct BoundaryStrain {
    double movement=0,stretch=1;
    bool outside=false;
    std::string describe()const {
        if(outside)return "added chart leaves the local region";
        std::ostringstream out;out.precision(3);
        out<<"boundary moved "<<movement<<" vx, edge stretch "<<stretch;return out.str();
    }
};
BoundaryStrain chartBoundaryStrain(const Graph& graph,LocalFit& local) {
    BoundaryStrain strain;
    for(const auto& [id,p]:graph.nodes) {
        if(local.prepared&&!local.movable.contains(id))continue;
        if(local.added(id)) {
            for(const auto& t:p.source->triangles)for(int i:t.val)if(!local.region.contains(p.uv[i]))strain.outside=true;
            continue;
        }
        const auto& before=local.committed.graph.nodes.at(id);
        for(const auto& t:fitTriangles(*p.source,&local))for(int j=0;j<3;++j) {
            int a=t[j],b=t[(j+1)%3];
            if(local.moves(id,a)==local.moves(id,b))continue;
            if(!local.moves(id,a))std::swap(a,b);
            strain.movement=std::max(strain.movement,cv::norm(p.uv[a]-before.uv[a]));
            const double oldLength=cv::norm(before.uv[a]-before.uv[b]);
            const double newLength=cv::norm(p.uv[a]-p.uv[b]);
            if(oldLength>1e-8)strain.stretch=std::max({strain.stretch,newLength/oldLength,oldLength/newLength});
        }
    }
    return strain;
}
bool chartBoundarySettled(const Graph& graph,const BoundaryStrain& strain) {
    return !strain.outside&&strain.movement<=graph.settings.radius*.25&&strain.stretch<=1.02;
}
Region changedSurfaceBounds(const Graph& graph,const Graph& committed,LocalFit* local=nullptr) {
    Region region;
    const auto ownedAfter=local&&local->geometry?nullptr:std::make_shared<detail::FiberJunctions>(graph);
    const auto ownedBefore=local&&local->prepared?nullptr:std::make_shared<detail::FiberJunctions>(committed);
    const auto& afterGeometry=local&&local->geometry?*local->geometry:*ownedAfter;
    const auto& beforeGeometry=local&&local->prepared?*local->prepared->junctions:*ownedBefore;
    for(const auto& [id,p]:graph.nodes) {
        auto before=committed.nodes.find(id);
        const auto& afterPoints=afterGeometry.points(*p.source);
        const auto* beforePoints=before==committed.nodes.end()?nullptr:&beforeGeometry.points(*before->second.source);
        if(before!=committed.nodes.end()&&p.uv.sharesStorage(before->second.uv)&&&afterPoints==beforePoints)continue;
        for(const auto& t:p.source->triangles) {
            bool changed=before==committed.nodes.end();
            if(!changed)for(int i:t.val)changed|=p.uv[i]!=before->second.uv[i]||
                afterPoints[i]!=(*beforePoints)[i];
            if(!changed)continue;
            for(int i:t.val){region.include(p.uv[i]);if(before!=committed.nodes.end())region.include(before->second.uv[i]);}
        }
    }
    return region;
}
}
namespace {
FitResult fitIncrementalImpl(Graph graph,const FitResult& committed,const std::atomic_bool* cancel,Progress progress,const GraphEdit* edit=nullptr,bool fixedReference=false) {
    graph.settings.validate();detail::cancelled(cancel);
    const auto& a=graph.settings;const auto& b=committed.graph.settings;
    std::set<std::string> deformablePatches;
    if(edit)for(const auto& link:edit->addedLinks) {
        for(const auto& [patch,fiber]:{std::pair{link.a.node,link.b.node},std::pair{link.b.node,link.a.node}})
            if(committed.graph.nodes.contains(patch)&&!graph.nodes.at(patch).source->fiber&&
               edit->addedNodes.contains(fiber)&&graph.nodes.at(fiber).source->fiber)
                deformablePatches.insert(patch);
    }
    // The full solver re-lays out every chart, so it is reserved for edits
    // that reach most of the graph or whose boundary never settles. Callers
    // that still want its verdict on a locally rejected edit run it themselves.
    auto global=[&](Graph whole,const std::string& reason=std::string{}) {
        if(edit){
            // A stale accepted neighborhood can drift even without the new
            // object. Before asking for global rearrangement, try attaching
            // only the new charts to that stable reference. This uses the same
            // residuals and all geometry/attachment checks, with old UV fixed.
            // Adding fibers to accepted patches requires those patches to
            // deform. Freezing them would silently undo that joint fit.
            if(!fixedReference&&!edit->addedNodes.empty()&&deformablePatches.empty()){
                if(progress)progress("Fitting addition against fixed accepted charts");
                return fitIncrementalImpl(std::move(whole),committed,cancel,progress,edit,true);
            }
            FitResult pending;pending.graph=std::move(whole);pending.needsRefit=true;
            pending.message=(reason.empty()?"Attachment reaches beyond the local fit":reason)+
                "; use Refit graph to fit this proposal with the accepted graph";return pending;
        }
        if(progress)progress("Fitting full graph");
        return fit(std::move(whole),cancel,progress);
    };
    if(!committed.valid||!committed.surface||graph.seed!=committed.graph.seed||a.spacing!=b.spacing||a.stiffness!=b.stiffness||
       a.radius!=b.radius||a.fiberScale!=b.fiberScale||a.patchScale!=b.patchScale||a.normalScale!=b.normalScale||a.normalManifest!=b.normalManifest)return global(std::move(graph));
    for(const auto& [id,p]:committed.graph.nodes) {
        auto current=graph.nodes.find(id);
        if(current==graph.nodes.end()||current->second.source!=p.source)return global(std::move(graph));
    }
    bool added=false;
    for(auto& [id,p]:graph.nodes) {
        if(auto previous=committed.graph.nodes.find(id);previous!=committed.graph.nodes.end()) {
            // A manual edit can start from a previous proposal fit. Always
            // anchor its unchanged surroundings to the accepted snapshot.
            p.uv=previous->second.uv;
        } else added=true;
    }
    // The full solver starts from the edit as proposed, as it always has.
    const Graph draft=graph;
    // An added chart still in its unplaced source coordinates says nothing
    // about where the edit lands; a region around it would only grow toward
    // the real location attempt by attempt. Place it first. Proposals already
    // placed near their contacts keep their placement and region unchanged.
    double misplaced=0;
    for(const auto& link:graph.links.read())
        if(!committed.graph.nodes.contains(link.a.node)||!committed.graph.nodes.contains(link.b.node))
            misplaced=std::max(misplaced,cv::norm(detail::linkError(graph,link)));
    if(added&&misplaced>4*a.spacing){LocalFit placing{committed,{}};alignRigid(graph,cancel,{},true,&placing);}
    Region affected;
    for(const auto& [id,p]:graph.nodes)
        if(!committed.graph.nodes.contains(id))for(const auto& t:p.source->triangles)for(int i:t.val)affected.include(p.uv[i]);
    auto compare=[](const Link& x,const Link& y){return linkKey(x)<linkKey(y);};
    std::multiset<Link,decltype(compare)> edited(compare);
    std::vector<Link> changed;
    std::shared_ptr<const detail::FiberJunctions> geometry;
    if(edit){
        changed=edit->addedLinks;
        for(size_t i:edit->removedLinks)changed.push_back(committed.graph.links.at(i));
        bool changedJunction=false;
        for(const auto& link:changed)changedJunction|=detail::horizontalVerticalLink(graph,link).first!=nullptr;
        geometry=edit->base->junctions;
        if(changedJunction){
            Graph junctionGraph; junctionGraph.settings=graph.settings;junctionGraph.nodes=graph.nodes;
            const std::set<size_t> removed(edit->removedLinks.begin(),edit->removedLinks.end());
            for(size_t i:edit->base->junctionLinks)if(!removed.contains(i))junctionGraph.links.push_back(committed.graph.links[i]);
            for(const auto& link:edit->addedLinks)if(detail::horizontalVerticalLink(graph,link).first)junctionGraph.links.push_back(link);
            geometry=std::make_shared<detail::FiberJunctions>(junctionGraph);
        }
    }else{
        std::multiset<Link,decltype(compare)> old(committed.graph.links.begin(),committed.graph.links.end(),compare);
        edited.insert(graph.links.cbegin(),graph.links.cend());
        std::set_symmetric_difference(old.begin(),old.end(),edited.begin(),edited.end(),std::back_inserter(changed),compare);
    }
    for(const auto& link:changed)for(const auto& location:{link.a,link.b})affected.include(placedLocation(graph,location));
    // A long new fiber can require a warp across an entire accepted patch.
    // Cutting that patch at a narrow chart-space halo locks in its old UV
    // distortion. Free the contacted patches and budget for their vertices;
    // distant charts and the existing boundary checks remain unchanged.
    size_t chartBudget=32768;
    for(const auto& id:deformablePatches) {
        const auto& p=graph.nodes.at(id);
        for(const auto& t:p.source->triangles)for(int i:t.val)affected.include(p.uv[i]);
        chartBudget=std::min(size_t(131072),chartBudget+p.uv.size());
    }
    if(!std::isfinite(affected.low[0]))return global(draft); // Explicit refit with no local edit.
    // Doubling continues until the free region reaches most of the graph.
    // Twelve doublings of the initial margin exceed any 16-million-sample grid.
    constexpr size_t maxAttempts=12;
    std::string lastFailure;
    double lastWorst=INFINITY,margin=4*a.spacing;
    for(size_t attempt=0;attempt<maxAttempts;++attempt,margin*=2) {
        detail::cancelled(cancel);
        LocalFit local{committed,affected.padded(margin)};
        if(edit){local.prepared=edit->base.get();local.edit=edit;local.geometry=geometry;local.fixedReference=fixedReference;}
        selectActiveCharts(graph,local);
        // Interactive edits have a deterministic work budget. They never
        // silently switch to a different full-graph objective.
        if(edit?(local.count>chartBudget||attempt>=4):(local.count*3>local.total*2))return global(draft);
        if(progress)progress("Fitting local charts: "+std::to_string(local.count)+" / "+std::to_string(local.total)+" vertices, attempt "+std::to_string(attempt+1));
        auto charts=fitCharts(graph,cancel,progress,true,&local);
        const auto strain=charts.valid?chartBoundaryStrain(charts.graph,local):BoundaryStrain{};
        if(!charts.valid||!chartBoundarySettled(charts.graph,strain)) {
            if(progress)progress("Expanding local charts: "+(charts.valid?"boundary remains strained ("+strain.describe()+")":charts.message));
            if(fixedReference)return global(draft,charts.valid?strain.describe():charts.message);
            continue;
        }
        double worst=0;const Link* worstLink=nullptr;
        for(const auto& link:edit?edit->addedLinks:changed) {
            // Removed links no longer constrain the solution.
            if(!edit&&!edited.contains(link))continue;
            const double error=cv::norm(detail::linkError(charts.graph,link));
            if(error>worst){worst=error;worstLink=&link;}
        }
        if(worstLink&&std::getenv("VC3D_PROFILE_CONSTRAINT_FIT"))
            std::clog<<"Worst attachment: "<<worstLink->a.node<<" grid=("<<worstLink->a.grid[0]<<","<<worstLink->a.grid[1]<<") line="<<worstLink->a.lineIndex
                <<" <-> "<<worstLink->b.node<<" grid=("<<worstLink->b.grid[0]<<","<<worstLink->b.grid[1]<<") line="<<worstLink->b.lineIndex
                <<" keypoint="<<worstLink->keypoint<<" pending="<<worstLink->pending<<" error="<<worst<<" vx"<<std::endl;
        if(worst>a.radius) {
            // Contacts that a widened region does not bring closer are
            // inconsistent with the graph; more region will not change that.
            if(worst>lastWorst*.75) {
                if(edit)return global(draft,"Attachment error exceeds proximity tolerance");
                FitResult rejected;rejected.graph=std::move(charts.graph);
                rejected.message="Attachment error exceeds proximity tolerance; add matching keypoint pairs";
                rejected.localChartVertices=local.count;rejected.localExpansions=attempt;
                return rejected;
            }
            lastWorst=worst;
            if(progress){std::ostringstream out;out.precision(3);out<<"Expanding local charts: attachment error "<<worst<<" vx exceeds proximity tolerance";progress(out.str());}
            if(fixedReference)return global(draft,"Attachment error exceeds proximity tolerance with accepted charts fixed");
            continue;
        }
        lastWorst=INFINITY;
        auto dirty=changedSurfaceBounds(charts.graph,committed.graph,&local);
        if(!std::isfinite(dirty.low[0])) {
            auto unchanged=committed;unchanged.graph=std::move(charts.graph);
            unchanged.correspondenceRms=attachmentRms(unchanged.graph,&local);
            unchanged.localChartVertices=local.count;unchanged.localSurfaceSamples=0;unchanged.localExpansions=attempt;
            return unchanged;
        }
        // Chart and XYZ relaxation have different footprints. Once the chart
        // boundary settles, a strained XYZ boundary needs a larger surface
        // region, not another chart solve. Keep the settled chart while growing
        // XYZ; the region stops growing once it covers the whole surface.
        double surfaceMargin=margin;
        std::string failure;
        for(size_t surfaceAttempt=0;surfaceAttempt<maxAttempts;++surfaceAttempt,surfaceMargin*=2) {
            detail::cancelled(cancel);
            local.region=dirty.padded(surfaceMargin);local.expand=false;
            if(edit&&(surfaceAttempt>=4||(local.region.high[0]-local.region.low[0])*(local.region.high[1]-local.region.low[1])/(a.spacing*a.spacing)>262144))return global(draft);
            if(progress)progress("Updating local tifxyz region: attempt "+std::to_string(surfaceAttempt+1));
            auto result=renderSurface(charts.graph,cancel,true,&local,nullptr,nullptr,progress);
            if(local.expand) {
                if(progress)progress("Expanding local surface: "+result.message);
                continue;
            }
            if(!result.valid||result.correspondenceRms>a.radius) {
                failure=result.valid?std::string("Attachment error exceeds proximity tolerance"):result.message;
                if(progress)progress("Expanding local charts after surface validation: "+failure);
                break;
            }
            result.localChartVertices=local.count;result.localExpansions=attempt;
            if(progress)progress("Local fit complete: "+std::to_string(local.count)+" chart vertices, "+std::to_string(result.localSurfaceSamples)+" surface samples");
            return result;
        }
        // Stop repeating the same geometry rejection after enlarging the
        // neighborhood. A full re-layout may still resolve the boundary;
        // that more expensive operation is the caller's decision.
        if(fixedReference)return global(draft,failure);
        if(!failure.empty()&&failure==lastFailure) {
            if(edit)return global(draft,failure);
            FitResult rejected;rejected.graph=std::move(charts.graph);rejected.message=failure;
            rejected.localChartVertices=local.count;rejected.localExpansions=attempt;
            return rejected;
        }
        lastFailure=failure;
    }
    return global(draft);
}
} // namespace
FitResult fitIncremental(Graph graph,const FitResult& committed,const std::atomic_bool* cancel,Progress progress){
    return fitIncrementalImpl(std::move(graph),committed,cancel,std::move(progress));
}
FitResult fitEdit(const GraphEdit& edit,const std::atomic_bool* cancel,Progress progress){
    detail::cancelled(cancel);
    if(!edit.base)throw std::runtime_error("Graph edit has no committed revision");
    const auto& committed=edit.base->accepted;
    Graph graph;graph.settings=committed.graph.settings;graph.seed=committed.graph.seed;graph.nodes=committed.graph.nodes;
    for(const auto& [id,p]:edit.addedNodes){
        if(!p.source||id!=p.source->id||!graph.nodes.emplace(id,p).second)throw std::runtime_error("Graph edit replaces an accepted source");
    }
    graph.links=edit.addedLinks;
    for(const auto& link:edit.addedLinks){(void)placedLocation(graph,link.a);(void)placedLocation(graph,link.b);}
    for(size_t i:edit.removedLinks)if(i>=committed.graph.links.size())throw std::runtime_error("Removed contact is outside the committed revision");
    // Topology validation works on object pairs, not hundreds of thousands of
    // sampled correspondence records. Retain multiplicity when removing links.
    std::set<std::string> reached{graph.seed};
    const auto pairs=detail::editedPairs(edit);
    for(size_t n=0;n<graph.nodes.size();++n){
        const auto before=reached.size();for(const auto& [a,b]:pairs){if(reached.contains(a))reached.insert(b);if(reached.contains(b))reached.insert(a);}
        if(before==reached.size())break;
    }
    FitResult result;
    if(reached.size()!=graph.nodes.size()){result.graph=std::move(graph);result.message="Every node must remain linked to the seed";}
    else if(edit.addedNodes.empty()&&edit.addedLinks.empty()&&edit.removedLinks.empty())return committed;
    else result=fitIncrementalImpl(std::move(graph),committed,cancel,std::move(progress),&edit);
    // Materialize contact storage once, after fitting; all intermediate stages
    // use the delta and prepared incident-contact lists.
    result.graph.links=materialize(edit).links;
    if(!result.valid&&result.localChartVertices){result.needsRefit=true;result.message+="; use Refit graph for a full fit of this proposal";}
    return result;
}
} // namespace vc::constraints
