#include "GraphInternal.hpp"
#include "vc/core/util/SymmetricDirichlet.hpp"
#include <ceres/ceres.h>
#include <set>
#include <tuple>

namespace vc::constraints {
namespace {
struct Region {
    cv::Vec2d low{INFINITY,INFINITY},high{-INFINITY,-INFINITY};
    void include(cv::Vec2d p) {for(int d=0;d<2;++d){low[d]=std::min(low[d],p[d]);high[d]=std::max(high[d],p[d]);}}
    Region padded(double margin)const {return {low-cv::Vec2d::all(margin),high+cv::Vec2d::all(margin)};}
    bool contains(cv::Vec2d p)const {return p[0]>=low[0]&&p[1]>=low[1]&&p[0]<=high[0]&&p[1]<=high[1];}
};
struct LocalFit {
    const FitResult& committed;
    Region region;
    std::map<std::string,std::vector<unsigned char>> movable;
    size_t count=0,total=0;
    bool expand=false;
    bool moves(const std::string& id,int i)const {return movable.at(id)[i]!=0;}
    bool added(const std::string& id)const {return !committed.graph.nodes.contains(id);}
};
struct ChartMetric {
    double inv00,inv01,inv11,weight,orientation;
    template<class T> bool operator()(const T* a,const T* b,const T* c,T* r) const {
        const T j00=(b[0]-a[0])*T(inv00), j10=(b[1]-a[1])*T(inv00);
        const T j01=(b[0]-a[0])*T(inv01)+(c[0]-a[0])*T(inv11);
        const T j11=(b[1]-a[1])*T(inv01)+(c[1]-a[1])*T(inv11);
        const T determinant=j00*j11-j01*j10;
        // Refuse flipped/degenerate trial steps, rather than regularizing an
        // inverse through zero and silently accepting a reflected element.
        if(dirichletValue(determinant)*orientation<=1e-8) return false;
        const T w=T(weight);
        r[0]=w*j00;r[1]=w*j01;r[2]=w*j10;r[3]=w*j11;
        r[4]=w*j11/determinant;r[5]=-w*j01/determinant;
        r[6]=-w*j10/determinant;r[7]=w*j00/determinant;
        return true;
    }
};
struct Match {
    std::vector<double> coefficients;
    double weight;
    template<class T> bool operator()(T const* const* p,T* r) const {
        r[0]=T(0);r[1]=T(0);
        for(size_t i=0;i<coefficients.size();++i) for(int d=0;d<2;++d) r[d]+=p[i][d]*T(coefficients[i]*weight);
        return true;
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
struct Cancellation : ceres::IterationCallback {
    const std::atomic_bool* cancel;
    explicit Cancellation(const std::atomic_bool* c):cancel(c){}
    ceres::CallbackReturnType operator()(const ceres::IterationSummary&) override {
        return cancel&&cancel->load()?ceres::SOLVER_ABORT:ceres::SOLVER_CONTINUE;
    }
};
ceres::Solver::Options solverOptions(Cancellation& callback,int iterations) {
    ceres::Solver::Options options;options.linear_solver_type=ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads=1;options.max_num_iterations=iterations;options.callbacks.push_back(&callback);
    options.function_tolerance=1e-7;options.gradient_tolerance=1e-8;
    return options;
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
    graph.seed=graph.nodes.begin()->first;return render(std::move(graph),cancel);
}
namespace {
FitResult renderSurface(Graph graph,const std::atomic_bool* cancel,bool refine,LocalFit* local=nullptr,
                        const cv::Mat_<cv::Vec3f>* saved=nullptr,const cv::Vec2d* savedOrigin=nullptr) {
    FitResult result;result.graph=std::move(graph);auto& g=result.graph;
    if(g.nodes.empty()) {result.message="Graph is empty";return result;}
    double spacing=g.settings.spacing;
    cv::Vec2d low{INFINITY,INFINITY},high{-INFINITY,-INFINITY};
    std::vector<double> stretches;
    for(const auto& [id,p]:g.nodes) {
        if(p.uv.size()!=p.source->xyz.size()) throw std::runtime_error("Source/placement size mismatch");
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
                double dist=cv::norm(p.source->xyz[t[(i+1)%3]]-p.source->xyz[t[i]]);
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
    std::vector<cv::Vec3d> sum(count,{0,0,0}),normals(count,{0,0,0});
    std::vector<double> mass(count,0);
    std::vector<unsigned char> coverage(count,0);
    std::vector<int> owner(count,-1);std::vector<std::string> ids;
    std::set<std::pair<std::string,std::string>> linked;
    for(const auto& l:g.links) linked.insert(std::minmax(l.a.node,l.b.node));
    size_t conflicts=0;int ordinal=0;
    // Synthetic ribbons are support only; real patches replace that support.
    for(bool fiber:{true,false}) for(const auto& [id,p]:g.nodes) {
        if(p.source->fiber!=fiber) continue;
        detail::cancelled(cancel);const auto& n=*p.source;int current=ordinal++;ids.push_back(id);
        std::set<size_t> visited;
        for(const auto& t:n.triangles) {
            cv::Vec2d a=(p.uv[t[0]]-low)/spacing,b=(p.uv[t[1]]-low)/spacing,c=(p.uv[t[2]]-low)/spacing;
            auto normal=detail::unit((n.xyz[t[1]]-n.xyz[t[0]]).cross(n.xyz[t[2]]-n.xyz[t[0]]));
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
                size_t index=size_t(y)*cols+x;if(!visited.insert(index).second)continue;
                auto xyz=n.xyz[t[0]]*(*w)[0]+n.xyz[t[1]]*(*w)[1]+n.xyz[t[2]]*(*w)[2];
                double weight=1;
                if(fiber) {
                    double sourceRow=0;for(int j=0;j<3;++j) sourceRow+=(t[j]/n.cols)*(*w)[j];
                    weight=1/(1+std::abs(sourceRow-n.rows/2));
                }
                if(owner[index]>=0&&!fiber&&g.nodes.at(ids[owner[index]]).source->fiber) {mass[index]=0;sum[index]={0,0,0};normals[index]={0,0,0};}
                if(mass[index]>0&&!fiber&&cv::norm(sum[index]/mass[index]-xyz)>g.settings.radius &&
                   !linked.contains(std::minmax(id,ids[owner[index]]))) ++conflicts;
                sum[index]+=xyz*weight;normals[index]+=normal*weight;mass[index]+=weight;owner[index]=current;
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
        if(coverage[a]==15||!mass[a]||!mass[b]||!mass[c]||!mass[d])continue;
        int victim=a;for(int i:{b,c,d})if(mass[i]<mass[victim])victim=i;unsupported.insert(victim);
    }
    for(int i:unsupported) {
        if(local&&!movable(i)&&detail::valid(previous(i))){local->expand=true;result.message="Surface support reaches the local boundary";return result;}
        mass[i]=0;
    }
    if(local)for(size_t i=0;i<count;++i)if(!movable(int(i))) {
        const auto old=previous(int(i));
        if(detail::valid(old)&&owner[i]<0){local->expand=true;result.message="Surface moved across the local boundary";return result;}
        mass[i]=detail::valid(old)?1:0;
    }
    if(saved) {
        for(size_t i=0;i<count;++i) {
            const auto p=(*saved)(int(i)/cols,int(i)%cols);
            if(p[0]!=-1&&(!detail::valid(cv::Vec3d(p))||owner[i]<0)){result.message="Saved surface has unsupported samples";return result;}
            mass[i]=p[0]!=-1?1:0;
        }
        for(int y=0;y+1<rows;++y)for(int x=0;x+1<cols;++x) {
            int a=y*cols+x;
            if(coverage[a]!=15&&mass[a]&&mass[a+1]&&mass[a+cols]&&mass[a+cols+1]){result.message="Saved surface bridges a source gap";return result;}
        }
    }
    std::vector<cv::Vec3d> points(count,{-1,-1,-1});
    for(size_t i=0;i<count;++i)if(mass[i])points[i]=saved?cv::Vec3d((*saved)(int(i)/cols,int(i)%cols)):(local&&!movable(int(i))?previous(int(i)):sum[i]/mass[i]);
    // Joint XYZ fit: every overlapping source contributes to the same sample.
    // Preserve its measured position while regularizing the shared grid metric.
    ceres::Problem problem;
    std::vector<cv::Vec3i> triangles;
    for(int y=0;y+1<rows;++y)for(int x=0;x+1<cols;++x) {
        int a=y*cols+x,b=a+1,c=a+cols,d=c+1;
        if(!mass[a]||!mass[b]||!mass[c]||!mass[d]) continue;
        triangles.push_back({a,b,c});triangles.push_back({d,c,b});
    }
    std::set<int> active;
    for(const auto& t:triangles)for(int i:t.val)active.insert(i);
    if(active.empty()){result.message="No complete quads after rasterization";return result;}
    if(refine) {
        for(int i:active)if(movable(i))problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Fidelity,3,3>(new Fidelity{points[i],5.0/spacing}),nullptr,points[i].val);
        for(const auto& t:triangles) problem.AddResidualBlock(vc::SymmetricDirichletLoss::Create(spacing,g.settings.stiffness,1e-8,0),nullptr,points[t[0]].val,points[t[1]].val,points[t[2]].val);
        if(local)for(int i:active)if(!movable(i))problem.SetParameterBlockConstant(points[i].val);
        Cancellation cb(cancel);auto options=solverOptions(cb,12);ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);detail::cancelled(cancel);
        if(!summary.IsSolutionUsable()){result.message="Shared surface fit failed: "+summary.BriefReport();return result;}
    }
    if(local) {
        // The last free ring must have settled before freezing the outside.
        for(int i:active)if(movable(i)) {
            bool boundary=false;
            for(int offset:{-1,1,-cols,cols})if(i+offset>=0&&i+offset<int(count)&&!movable(i+offset)) {
                boundary=true;const auto oldA=previous(i),oldB=previous(i+offset);
                if(!detail::valid(oldA)||!detail::valid(oldB))continue;
                const double before=cv::norm(oldA-oldB),after=cv::norm(points[i]-points[i+offset]);
                if(before>1e-8&&(after>before*1.02||after<before/1.02))local->expand=true;
            }
            if(boundary&&detail::valid(previous(i))&&cv::norm(points[i]-previous(i))>g.settings.radius*.25)local->expand=true;
        }
        if(local->expand){result.message="Surface adjustment reaches the local boundary";return result;}
        result.localSurfaceSamples=std::count_if(active.begin(),active.end(),movable);
    }
    cv::Mat_<cv::Vec3f> grid=saved?saved->clone():cv::Mat_<cv::Vec3f>(rows,cols,cv::Vec3f{-1,-1,-1});
    for(int i:active) grid(i/cols,i%cols)=cv::Vec3f(points[i]);
    for(const auto& t:triangles) {
        auto cross=(points[t[1]]-points[t[0]]).cross(points[t[2]]-points[t[0]]);
        auto reference=normals[t[0]]+normals[t[1]]+normals[t[2]];
        if(cv::norm(cross)<1e-6||cross.dot(reference)<=0) {
            bool support=false;for(int i:t.val) support|=g.nodes.at(ids[owner[i]]).source->fiber;
            support|=owner[t[0]]!=owner[t[1]]||owner[t[0]]!=owner[t[2]];
            if(!support){result.message="Patch geometry contains an inverted quad";return result;}
            int victim=t[0];for(int i:t.val)if(mass[i]<mass[victim])victim=i;
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
    for(int y=0;y+1<rows;++y)for(int x=0;x+1<cols;++x) {
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
    double squared=0;for(const auto& l:g.links){auto d=detail::linkError(g,l);squared+=d.dot(d);}
    result.correspondenceRms=g.links.empty()?0:std::sqrt(squared/g.links.size());
    result.valid=true;result.message=unsupported.empty()?"Ready":"Ready; "+std::to_string(unsupported.size())+" unsupported boundary samples omitted";return result;
}
} // namespace
FitResult detail::restoreSurface(Graph graph,const cv::Mat_<cv::Vec3f>& grid,cv::Vec2d origin) {
    return renderSurface(std::move(graph),nullptr,false,nullptr,&grid,&origin);
}
FitResult render(Graph graph,const std::atomic_bool* cancel) {return renderSurface(std::move(graph),cancel,true);}
FitResult renderPlacement(Graph graph,const std::atomic_bool* cancel) {return renderSurface(std::move(graph),cancel,false);}

namespace {
FitResult fitCharts(Graph graph,const std::atomic_bool* cancel,Progress progress,bool alignAxes,LocalFit* local=nullptr) {
    graph.settings.validate();detail::cancelled(cancel);
    if(graph.nodes.empty())return render(std::move(graph),cancel);
    std::set<std::string> connected{graph.seed};
    for(size_t pass=0;pass<graph.nodes.size();++pass) {
        const auto before=connected.size();
        for(const auto& l:graph.links) {
            if(!graph.nodes.contains(l.a.node)||!graph.nodes.contains(l.b.node))throw std::runtime_error("Link refers to a missing node");
            if(connected.contains(l.a.node))connected.insert(l.b.node);
            if(connected.contains(l.b.node))connected.insert(l.a.node);
        }
        if(connected.size()==before)break;
    }
    if(!graph.nodes.contains(graph.seed)||connected.size()!=graph.nodes.size()) {
        FitResult failed;failed.graph=std::move(graph);failed.message="Every node must remain linked to the seed";return failed;
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
        for(const auto& l:graph.links) {
            if(l.a.node==l.b.node||(local&&!local->added(l.a.node)&&!local->added(l.b.node)))continue;
            auto [wa,wb]=detail::linkWeights(graph,l);
            auto a=detail::interpolate(graph.nodes.at(l.a.node).uv,wa);
            auto b=detail::interpolate(graph.nodes.at(l.b.node).uv,wb);
            rigid.AddResidualBlock(new ceres::AutoDiffCostFunction<RigidMatch,2,3,3>(
                new RigidMatch{a,b,(l.keypoint?1000.0:30.0)/graph.settings.spacing}),nullptr,poses[l.a.node].val,poses[l.b.node].val);
        }
        rigid.SetParameterBlockConstant(poses.at(graph.seed).val);
        Cancellation cb(cancel);auto options=solverOptions(cb,100);ceres::Solver::Summary summary;
        ceres::Solve(options,&rigid,&summary);detail::cancelled(cancel);
        if(!summary.IsSolutionUsable())throw std::runtime_error("Rigid graph alignment failed: "+summary.BriefReport());
        for(auto& [id,p]:graph.nodes) {
            if(local&&!local->added(id))continue;
            const auto pose=poses.at(id);const double c=std::cos(pose[0]),s=std::sin(pose[0]);
            for(auto& uv:p.uv)uv={c*uv[0]-s*uv[1]+pose[1],s*uv[0]+c*uv[1]+pose[2]};
        }
    }
    if(progress)progress("Fitting connected graph");
    ceres::Problem problem;
    std::map<std::string,double> centers;
    for(auto& [id,p]:graph.nodes) {
        const auto& n=*p.source;
        for(const auto& t:n.triangles) {
            if(local&&!local->moves(id,t[0])&&!local->moves(id,t[1])&&!local->moves(id,t[2]))continue;
            auto x=n.xyz[t[1]]-n.xyz[t[0]],y=n.xyz[t[2]]-n.xyz[t[0]];
            double length=cv::norm(x),projection=x.dot(y)/length;
            double height=std::sqrt(std::max(0.0,y.dot(y)-projection*projection));
            if(length<1e-6||height<1e-6)continue;
            double det=triangleOrientation(p,t);
            if(!std::isfinite(det)||std::abs(det)<1e-8) {FitResult failed;failed.graph=std::move(graph);failed.message="Degenerate initial chart: "+id;return failed;}
            double w=std::sqrt(graph.settings.stiffness*length*height/(2*graph.settings.spacing*graph.settings.spacing));
            auto* cost=new ceres::AutoDiffCostFunction<ChartMetric,8,2,2,2>(new ChartMetric{1/length,-projection/(length*height),1/height,w,det>0?1.0:-1.0});
            problem.AddResidualBlock(cost,nullptr,p.uv[t[0]].val,p.uv[t[1]].val,p.uv[t[2]].val);
        }
        if(alignAxes&&n.fiber&&(n.direction=='H'||n.direction=='V')) {
            const int axis=n.direction=='V'?0:1;
            double center=0;
            for(int x=0;x<n.cols;++x)center+=p.uv[(n.rows/2)*n.cols+x][axis];
            centers[id]=center/n.cols;
            for(int x=0;x<n.cols;++x) {
                auto& uv=p.uv[(n.rows/2)*n.cols+x];
                if(problem.HasParameterBlock(uv.val)&&(!local||local->moves(id,(n.rows/2)*n.cols+x)))
                    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<FiberBand,1,2,1>(
                        new FiberBand{axis,bandHalfWidth,bandWeight}),nullptr,uv.val,&centers[id]);
            }
        }
        if(n.fiber&&(n.direction=='H'||n.direction=='V'))for(int x=0;x+1<n.cols;++x) {
            int a=(n.rows/2)*n.cols+x,b=a+1;
            if((!local||local->moves(id,a)||local->moves(id,b))&&
               problem.HasParameterBlock(p.uv[a].val)&&problem.HasParameterBlock(p.uv[b].val))
                problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Direction,1,2,2>(new Direction{n.direction=='V'?0:1,0.2/graph.settings.spacing}),nullptr,p.uv[a].val,p.uv[b].val);
        }
    }
    for(const auto& l:graph.links) {
        std::vector<double*> blocks;std::vector<double> coefficients;
        auto [wa,wb]=detail::linkWeights(graph,l);
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
        auto* cost=new ceres::DynamicAutoDiffCostFunction<Match,4>(new Match{coefficients,(l.keypoint?1000.0:30.0)/graph.settings.spacing});
        for(size_t i=0;i<blocks.size();++i)cost->AddParameterBlock(2);cost->SetNumResiduals(2);
        problem.AddResidualBlock(cost,nullptr,blocks);
    }
    if(graph.nodes.empty()) return render(std::move(graph),cancel);
    if(local) {
        for(auto& [id,p]:graph.nodes) {
            for(size_t i=0;i<p.uv.size();++i)if(!local->moves(id,i)&&problem.HasParameterBlock(p.uv[i].val))problem.SetParameterBlockConstant(p.uv[i].val);
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
    Cancellation cb(cancel);auto options=solverOptions(cb,60);ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);detail::cancelled(cancel);
    if(!summary.IsSolutionUsable()) {FitResult failed;failed.graph=std::move(graph);failed.message=summary.BriefReport();return failed;}
    for(const auto& l:graph.links)if(l.keypoint&&cv::norm(placedLocation(graph,l.a)-placedLocation(graph,l.b))>0.02) {
        FitResult failed;failed.graph=std::move(graph);failed.message="Keypoint constraints conflict; edit or remove a pair";return failed;
    }
    if(!alignAxes||local) {
        FitResult aligned;aligned.graph=std::move(graph);aligned.valid=true;
        return aligned;
    }
    if(progress)progress("Building shared tifxyz surface");
    auto result=render(std::move(graph),cancel);
    if(result.valid && result.correspondenceRms>result.graph.settings.radius) {
        result.valid=false;
        result.message="Attachment error exceeds proximity tolerance; add matching keypoint pairs";
    }
    return result;
}
} // namespace

FitResult fit(Graph graph,const std::atomic_bool* cancel,Progress progress) {
    // New linked components can start with very large loop-closure errors.
    // First let their source charts relax into a connected configuration;
    // imposing rows/columns on those errors directly can trap the no-flip
    // solve. Existing fitted graphs normally skip this initialization pass.
    double squared=0;
    for(const auto& link:graph.links) {
        detail::cancelled(cancel);
        const auto error=detail::linkError(graph,link);squared+=error.dot(error);
    }
    if(!graph.links.empty()&&squared>graph.links.size()*graph.settings.radius*graph.settings.radius) {
        if(progress)progress("Initializing linked source geometry");
        auto initialized=fitCharts(std::move(graph),cancel,progress,false);
        if(!initialized.valid)return initialized;
        graph=std::move(initialized.graph);
    }
    return fitCharts(std::move(graph),cancel,progress,true);
}
namespace {
auto linkKey(const Link& l) {
    return std::tuple{l.a.node,l.a.lineIndex,l.a.grid[0],l.a.grid[1],l.b.node,l.b.lineIndex,l.b.grid[0],l.b.grid[1],l.keypoint,l.pending};
}
bool chartBoundarySettled(const Graph& graph,LocalFit& local) {
    const double movementLimit=graph.settings.radius*.25;
    for(const auto& [id,p]:graph.nodes) {
        if(local.added(id)) {
            for(const auto& t:p.source->triangles)for(int i:t.val)if(!local.region.contains(p.uv[i]))return false;
            continue;
        }
        const auto& before=local.committed.graph.nodes.at(id);
        for(const auto& t:p.source->triangles)for(int j=0;j<3;++j) {
            int a=t[j],b=t[(j+1)%3];
            if(local.moves(id,a)==local.moves(id,b))continue;
            if(!local.moves(id,a))std::swap(a,b);
            if(cv::norm(p.uv[a]-before.uv[a])>movementLimit)return false;
            const double oldLength=cv::norm(before.uv[a]-before.uv[b]);
            const double newLength=cv::norm(p.uv[a]-p.uv[b]);
            if(oldLength>1e-8&&(newLength>oldLength*1.02||newLength<oldLength/1.02))return false;
        }
    }
    return true;
}
Region changedSurfaceBounds(const Graph& graph,const Graph& committed) {
    Region region;
    for(const auto& [id,p]:graph.nodes) {
        auto before=committed.nodes.find(id);
        for(const auto& t:p.source->triangles) {
            bool changed=before==committed.nodes.end();
            if(!changed)for(int i:t.val)changed|=p.uv[i]!=before->second.uv[i];
            if(!changed)continue;
            for(int i:t.val){region.include(p.uv[i]);if(before!=committed.nodes.end())region.include(before->second.uv[i]);}
        }
    }
    return region;
}
}
FitResult fitIncremental(Graph graph,const FitResult& committed,const std::atomic_bool* cancel,Progress progress) {
    graph.settings.validate();detail::cancelled(cancel);
    const auto& a=graph.settings;const auto& b=committed.graph.settings;
    auto global=[&] {
        if(progress)progress("Fitting full graph");
        return fit(std::move(graph),cancel,progress);
    };
    if(!committed.valid||!committed.surface||graph.seed!=committed.graph.seed||a.spacing!=b.spacing||a.stiffness!=b.stiffness||
       a.radius!=b.radius||a.fiberScale!=b.fiberScale||a.patchScale!=b.patchScale||a.normalScale!=b.normalScale||a.normalManifest!=b.normalManifest)return global();
    for(const auto& [id,p]:committed.graph.nodes) {
        auto current=graph.nodes.find(id);
        if(current==graph.nodes.end()||current->second.source!=p.source)return global();
    }
    Region affected;
    for(auto& [id,p]:graph.nodes) {
        if(auto previous=committed.graph.nodes.find(id);previous!=committed.graph.nodes.end()) {
            // A manual edit can start from a previous proposal fit. Always
            // anchor its unchanged surroundings to the accepted snapshot.
            p.uv=previous->second.uv;
        } else for(const auto& t:p.source->triangles)for(int i:t.val)affected.include(p.uv[i]);
    }
    auto compare=[](const Link& x,const Link& y){return linkKey(x)<linkKey(y);};
    std::multiset<Link,decltype(compare)> old(committed.graph.links.begin(),committed.graph.links.end(),compare);
    std::multiset<Link,decltype(compare)> edited(graph.links.begin(),graph.links.end(),compare);
    std::vector<Link> changed;
    std::set_symmetric_difference(old.begin(),old.end(),edited.begin(),edited.end(),std::back_inserter(changed),compare);
    for(const auto& link:changed)for(const auto& location:{link.a,link.b})affected.include(placedLocation(graph,location));
    if(!std::isfinite(affected.low[0]))return global(); // Explicit refit with no local edit.
    double margin=4*a.spacing;
    for(size_t attempt=0;attempt<4;++attempt,margin*=2) {
        detail::cancelled(cancel);
        LocalFit local{committed,affected.padded(margin)};
        for(const auto& [id,p]:graph.nodes) {
            auto& mask=local.movable[id];mask.resize(p.uv.size());
            for(size_t i=0;i<p.uv.size();++i) {
                mask[i]=detail::valid(p.source->xyz[i])&&(local.added(id)||local.region.contains(p.uv[i]));
                local.count+=mask[i];++local.total;
            }
        }
        // A broad addition should go directly to the established full solver.
        if(local.count*3>local.total*2)return global();
        if(progress)progress("Fitting local charts: "+std::to_string(local.count)+" / "+std::to_string(local.total)+" vertices, attempt "+std::to_string(attempt+1));
        auto charts=fitCharts(graph,cancel,progress,true,&local);
        if(!charts.valid||!chartBoundarySettled(charts.graph,local)) {
            if(progress)progress("Expanding local charts: "+(charts.valid?std::string("boundary remains strained"):charts.message));
            continue;
        }
        bool matches=true;
        for(const auto& link:changed) {
            // Removed links no longer constrain the solution.
            if(!edited.contains(link))continue;
            if(cv::norm(detail::linkError(charts.graph,link))>a.radius){matches=false;break;}
        }
        if(!matches)continue;
        auto dirty=changedSurfaceBounds(charts.graph,committed.graph);
        if(!std::isfinite(dirty.low[0])) {
            auto unchanged=committed;unchanged.graph=std::move(charts.graph);
            double squared=0;for(const auto& link:unchanged.graph.links){const auto e=detail::linkError(unchanged.graph,link);squared+=e.dot(e);}
            unchanged.correspondenceRms=unchanged.graph.links.empty()?0:std::sqrt(squared/unchanged.graph.links.size());
            unchanged.localChartVertices=local.count;unchanged.localSurfaceSamples=0;unchanged.localExpansions=attempt;
            return unchanged;
        }
        local.region=dirty.padded(margin);
        if(progress)progress("Updating local tifxyz region");
        auto result=renderSurface(std::move(charts.graph),cancel,true,&local);
        if(!result.valid||local.expand) {
            if(progress)progress("Expanding local surface: "+result.message);
            continue;
        }
        if(result.correspondenceRms>a.radius)continue;
        result.localChartVertices=local.count;result.localExpansions=attempt;
        if(progress)progress("Local fit complete: "+std::to_string(local.count)+" chart vertices, "+std::to_string(result.localSurfaceSamples)+" surface samples");
        return result;
    }
    return global();
}
} // namespace vc::constraints
