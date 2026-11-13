#pragma once
#include "vc_object_store/geometry/PolylineArc.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/ParallelFor.hpp"
#include <map>
#include <numbers>
#include <numeric>
#include <type_traits>
#include <unordered_map>

namespace vc::geometry {

// Half-surface from a piecewise-linear umbilicus toward a fixed XY ray.
// Transport belongs to the positive-angle side; axis intersections are invalid.
class WindingSeam {
public:
    explicit WindingSeam(std::span<const Point> axis,double angle=-std::numbers::pi/2)
        :axis_(axis.begin(),axis.end()),angle_(angle),ray_{std::cos(angle),std::sin(angle)}
    {
        if(!std::isfinite(angle))throw std::invalid_argument("nonfinite seam angle");
        std::sort(axis_.begin(),axis_.end(),[](const auto& a,const auto& b){return a[2]<b[2];});
        if(axis_.size()<2)throw std::invalid_argument("axis needs distinct increasing Z coordinates");
        for(size_t i=0;i<axis_.size();++i){
            for(const auto value:axis_[i])if(!std::isfinite(value))throw std::invalid_argument("nonfinite umbilicus axis");
            if(i&&axis_[i][2]<=axis_[i-1][2])throw std::invalid_argument("axis needs distinct increasing Z coordinates");
        }
        for(auto& value:ray_)if(std::abs(value)<1e-15)value=0;
    }
    const std::vector<Point>& axis()const{return axis_;}
    double angle()const{return angle_;}
    std::array<double,2> ray()const{return ray_;}
    std::array<double,2> relative(const Point& p)const{
        for(const auto value:p)if(!std::isfinite(value))throw std::invalid_argument("nonfinite path");
        const auto upper=std::upper_bound(axis_.begin(),axis_.end(),p[2],[](double z,const auto& point){return z<point[2];});
        if(upper==axis_.begin())return {p[0]-axis_.front()[0],p[1]-axis_.front()[1]};
        if(upper==axis_.end())return {p[0]-axis_.back()[0],p[1]-axis_.back()[1]};
        const auto& a=*(upper-1);const auto& b=*upper;
        const auto interpolate=[&](size_t k){const double slope=(b[k]-a[k])/(b[2]-a[2]);return slope*(p[2]-a[2])+a[k];};
        return {p[0]-interpolate(0),p[1]-interpolate(1)};
    }
    double phase(const Point& p)const{
        const auto r=relative(p);constexpr double tau=2*std::numbers::pi;
        double phase=std::fmod(std::atan2(r[1],r[0])-angle_,tau);
        if(phase<0)phase+=tau;
        if(phase==0)phase=0; // Python remainder has positive zero for positive tau.
        return phase/tau;
    }
    int64_t transport(const Point& a,const Point& b)const{
        const auto interval=[&](double z){return std::upper_bound(axis_.begin(),axis_.end(),z,
            [](double value,const auto& point){return value<point[2];})-axis_.begin();};
        if(interval(a[2])==interval(b[2]))return crossing(relative(a),relative(b));
        return segment(a,b);
    }
    int64_t segment(const Point& a,const Point& b)const{
        const auto at=[&](double t){Point p;for(size_t k=0;k<3;++k)p[k]=a[k]+t*(b[k]-a[k]);return relative(p);};
        auto previous=at(0.);int64_t winding=0;
        const auto step=[&](double z){
            const double cut=(z-a[2])/(b[2]-a[2]);
            if(cut>0&&cut<1){const auto next=at(cut);winding+=crossing(previous,next);previous=next;}
        };
        if(b[2]>a[2]){
            auto it=std::upper_bound(axis_.begin(),axis_.end(),a[2],[](double z,const auto& p){return z<p[2];});
            for(;it!=axis_.end()&&(*it)[2]<b[2];++it)step((*it)[2]);
        }else if(b[2]<a[2]){
            auto it=std::lower_bound(axis_.begin(),axis_.end(),a[2],[](const auto& p,double z){return p[2]<z;});
            while(it!=axis_.begin()){--it;if((*it)[2]<=b[2])break;step((*it)[2]);}
        }
        winding+=crossing(previous,at(1.));
        return winding;
    }
private:
    int64_t crossing(const std::array<double,2>& a,const std::array<double,2>& b)const{
        const double dx=b[0]-a[0],dy=b[1]-a[1];
        const double t=std::clamp(-(a[0]*dx+a[1]*dy)/std::max(dx*dx+dy*dy,1e-300),0.,1.);
        const double x=a[0]+t*dx,y=a[1]+t*dy;
        if(std::sqrt(x*x+y*y)<1e-8)throw std::invalid_argument("path intersects umbilicus; seam transport undefined");
        const double ua=a[0]*ray_[0]+a[1]*ray_[1],ub=b[0]*ray_[0]+b[1]*ray_[1];
        const double va=ray_[0]*a[1]-ray_[1]*a[0],vb=ray_[0]*b[1]-ray_[1]*b[0];
        const bool up=va<0&&vb>=0,down=va>=0&&vb<0;
        if(!up&&!down)return 0;
        const double fraction=-va/(vb-va);
        return ua+fraction*(ub-ua)>0?(up?1:-1):0;
    }
    std::vector<Point> axis_;
    double angle_;
    std::array<double,2> ray_;
};

struct FiberArcSample {Point point;int64_t turn;};
class PackedFiberGeometry {
public:
    PackedFiberGeometry(std::span<const int64_t> offsets,const double* arcs,size_t arcCount,ptrdiff_t arcStride,
                        std::span<const double> xyz,std::span<const int64_t> turns,WindingSeam seam)
        :offsets_(offsets),arcs_(arcs),arcStride_(arcStride),index_(offsets,arcs,arcCount,arcStride),
         xyz_(xyz),turns_(turns),seam_(std::move(seam))
    {
        if(arcCount>std::numeric_limits<size_t>::max()/3||xyz.size()!=arcCount*3||turns.size()!=arcCount)
            throw std::invalid_argument("packed geometry column sizes differ");
    }
    std::pair<int64_t,int64_t> pointRange(int64_t owner)const{
        if(owner<0||uint64_t(owner)>=offsets_.size()-1)throw std::invalid_argument("unknown packed fiber");
        return {offsets_[owner],offsets_[owner+1]};
    }
    Point point(int64_t i)const{return {xyz_[3*i],xyz_[3*i+1],xyz_[3*i+2]};}
    int64_t turn(int64_t i)const{return turns_[i];}
    double arcLength(int64_t owner)const{return arcs_[(pointRange(owner).second-1)*arcStride_];}
    const WindingSeam& seam()const{return seam_;}
    FiberArcSample sample(int64_t owner,double arc)const{
        const auto at=index_.locate(owner,arc);const auto a=point(at.segment),b=point(at.segment+1);
        Point result;for(size_t k=0;k<3;++k)result[k]=a[k]+(b[k]-a[k])*at.fraction;
        return {result,turns_[at.segment]+seam_.transport(a,result)};
    }
private:
    std::span<const int64_t> offsets_;
    const double* arcs_;ptrdiff_t arcStride_;
    PolylineArcIndex index_;
    std::span<const double> xyz_;
    std::span<const int64_t> turns_;
    WindingSeam seam_;
};

struct FiberGeometryPriors {
    std::vector<double> directions,phase,z,radii,sampleArc,samplePhase,sampleZ;
    std::vector<int64_t> minimumTurn,maximumTurn,sampleOwner;
};

struct PackedWindingColumns {
    std::vector<int64_t> source,turns;
    std::vector<double> arc;
};
inline PackedWindingColumns packWindingGeometry(std::span<const int64_t> offsets,
    std::span<const double> xyz,const WindingSeam& seam,int64_t firstOwner=0,size_t workers=1)
{
    if(!workers||offsets.empty()||offsets.front()!=0||xyz.size()%3||offsets.back()<0||
       uint64_t(offsets.back())!=xyz.size()/3||firstOwner<0||
       uint64_t(firstOwner)>uint64_t(std::numeric_limits<int64_t>::max())-(offsets.size()-1))
        throw std::invalid_argument("invalid winding geometry pack inputs");
    for(size_t i=1;i<offsets.size();++i)
        if(offsets[i]<0||offsets[i]<offsets[i-1]||offsets[i]-offsets[i-1]<2)
            throw std::invalid_argument("packed fibers require at least two points");
    const auto points=xyz.size()/3;PackedWindingColumns result;
    result.source.resize(points);result.turns.resize(points);result.arc.resize(points);
    const auto point=[&](int64_t j){return Point{xyz[3*j],xyz[3*j+1],xyz[3*j+2]};};
    objects::parallelFor(offsets.size()-1,workers,[&](size_t owner){
        const auto first=offsets[owner],last=offsets[owner+1];
        result.source[first]=firstOwner+owner;result.arc[first]=0;result.turns[first]=0;
        auto previous=point(first);
        for(int64_t j=first+1;j<last;++j){
            const auto current=point(j);
            const double dx=current[0]-previous[0],dy=current[1]-previous[1],dz=current[2]-previous[2];
            // NumPy norm(diff(...),axis=1) and cumsum use these same ordered
            // float64 operations. Keep accumulation sequential within a fiber.
            const double xx=dx*dx,yy=dy*dy,zz=dz*dz;
            const double length=std::sqrt((xx+yy)+zz);
            result.source[j]=firstOwner+owner;
            result.arc[j]=result.arc[j-1]+length;
            result.turns[j]=result.turns[j-1]+seam.transport(previous,current);
            previous=current;
        }
    });
    return result;
}
inline double medianInPlace(std::span<double> values){
    const size_t middle=values.size()/2;
    std::nth_element(values.begin(),values.begin()+middle,values.end());
    const double upper=values[middle];
    return values.size()%2?upper:(*std::max_element(values.begin(),values.begin()+middle)+upper)*.5;
}
inline FiberGeometryPriors fiberGeometryPriors(const PackedFiberGeometry& geometry,std::span<const int64_t> owners,
    std::span<const int64_t> family,double spacing=1000.,size_t workers=32)
{
    if(owners.size()!=family.size()||!std::isfinite(spacing)||spacing<=0||!workers)
        throw std::invalid_argument("invalid geometry prior inputs");
    const auto n=owners.size();FiberGeometryPriors result;
    result.directions.resize(n);result.phase.resize(n);result.z.resize(n);result.radii.resize(n);
    result.minimumTurn.resize(n);result.maximumTurn.resize(n);
    std::vector<int64_t> offsets(n+1);
    constexpr size_t grain=64;
    objects::parallelFor((n+grain-1)/grain,workers,[&](size_t block){
        thread_local std::vector<double> radius;
        for(size_t i=block*grain;i<std::min(n,(block+1)*grain);++i){
            if(family[i]!=0&&family[i]!=1)throw std::invalid_argument("fiber family must be H or V");
            const auto [first,end]=geometry.pointRange(owners[i]);
            const auto start=geometry.point(first),last=geometry.point(end-1);
            result.phase[i]=geometry.seam().phase(start);result.z[i]=start[2];
            const double difference=family[i]==0?geometry.turn(end-1)+geometry.seam().phase(last)-result.phase[i]:last[2]-start[2];
            if(difference==0||!std::isfinite(difference))throw std::invalid_argument("unidentifiable fiber direction");
            result.directions[i]=difference>0?1.:-1.;
            radius.resize(end-first);int64_t minimum=geometry.turn(first),maximum=minimum;
            for(int64_t j=first;j<end;++j){
                const auto relative=geometry.seam().relative(geometry.point(j));
                radius[j-first]=std::sqrt(relative[0]*relative[0]+relative[1]*relative[1]);
                minimum=std::min(minimum,geometry.turn(j));maximum=std::max(maximum,geometry.turn(j));
            }
            result.radii[i]=medianInPlace(radius);result.minimumTurn[i]=minimum;result.maximumTurn[i]=maximum;
            const double intervals=std::ceil(geometry.arcLength(owners[i])/spacing);
            if(!std::isfinite(intervals)||intervals<0||intervals>=double(std::numeric_limits<int64_t>::max()-1))
                throw std::invalid_argument("invalid fiber sample count");
            offsets[i+1]=std::max(int64_t(2),int64_t(intervals)+1);
        }
    });
    for(size_t i=1;i<offsets.size();++i){
        if(offsets[i]>std::numeric_limits<int64_t>::max()-offsets[i-1])throw std::overflow_error("fiber sample count overflow");
        offsets[i]+=offsets[i-1];
    }
    const size_t samples=offsets.back();result.sampleOwner.resize(samples);result.sampleArc.resize(samples);
    result.samplePhase.resize(samples);result.sampleZ.resize(samples);
    objects::parallelFor((n+grain-1)/grain,workers,[&](size_t block){
        for(size_t i=block*grain;i<std::min(n,(block+1)*grain);++i){
            const auto count=offsets[i+1]-offsets[i];const double length=geometry.arcLength(owners[i]);
            for(int64_t j=0;j<count;++j){
                const double arc=j==count-1?length:length*double(j)/double(count-1);
                const auto sample=geometry.sample(owners[i],arc);const size_t at=offsets[i]+j;
                result.sampleOwner[at]=i;result.sampleArc[at]=arc;
                result.samplePhase[at]=geometry.seam().phase(sample.point);result.sampleZ[at]=sample.point[2];
            }
        }
    });
    return result;
}

template<class Label>struct FixedFiberTargetColumns {
    std::vector<Label> target;
    std::vector<double> arcA,arcB;
};
template<class Label>FixedFiberTargetColumns<Label> fixedFiberTargetColumns(const PackedFiberGeometry& geometry,
    std::span<const int64_t> owners,std::span<const Label> labels,std::span<const int64_t> a,std::span<const int64_t> b,
    std::span<const double> arcA,std::span<const double> arcB,std::span<const double> directions={},size_t workers=32)
{
    if(owners.size()!=labels.size()||a.size()!=b.size()||a.size()!=arcA.size()||a.size()!=arcB.size()||
       (!directions.empty()&&directions.size()!=owners.size())||!workers)
        throw std::invalid_argument("invalid fixed target columns");
    for(const auto value:directions)if(value!=1&&value!=-1)throw std::invalid_argument("fiber direction must be a sign");
    if constexpr(std::is_floating_point_v<Label>)
        for(const auto value:labels)if(!std::isfinite(value))throw std::invalid_argument("nonfinite winding label");
    FixedFiberTargetColumns<Label> result;const auto n=a.size();result.target.resize(n);
    if(!directions.empty()){result.arcA.resize(n);result.arcB.resize(n);}
    constexpr size_t grain=4096;
    objects::parallelFor((n+grain-1)/grain,workers,[&](size_t block){
        for(size_t i=block*grain;i<std::min(n,(block+1)*grain);++i){
            if(a[i]<0||b[i]<0||uint64_t(a[i])>=owners.size()||uint64_t(b[i])>=owners.size())
                throw std::invalid_argument("fixed target endpoint outside fiber set");
            const auto start=geometry.sample(owners[a[i]],arcA[i]),end=geometry.sample(owners[b[i]],arcB[i]);
            const auto step=end.turn-start.turn-geometry.seam().transport(start.point,end.point);
            if constexpr(std::is_integral_v<Label>){
                // Winding labels and transport have a physical range vastly
                // below this guard; reject overflow rather than invoke UB.
                constexpr auto guard=std::numeric_limits<Label>::max()/4;
                if(labels[a[i]]>guard||labels[a[i]]<-guard||labels[b[i]]>guard||labels[b[i]]<-guard||step>guard||step<-guard)
                    throw std::overflow_error("fixed winding target overflow");
            }
            result.target[i]=labels[b[i]]-labels[a[i]]+step;
            if(!directions.empty()){result.arcA[i]=arcA[i]*directions[a[i]];result.arcB[i]=arcB[i]*directions[b[i]];}
        }
    });
    return result;
}

struct GroupedMedians {std::vector<int64_t> groups;std::vector<double> medians;};
inline GroupedMedians groupedMedians(std::span<const int64_t> groups,std::span<const double> values,size_t workers=32){
    if(groups.size()!=values.size()||!workers)throw std::invalid_argument("invalid grouped median columns");
    std::map<int64_t,size_t> count;for(auto group:groups)++count[group];
    GroupedMedians result;std::vector<size_t> offsets{0};std::unordered_map<int64_t,size_t> indices;
    for(const auto& [group,n]:count){indices.emplace(group,result.groups.size());result.groups.push_back(group);offsets.push_back(offsets.back()+n);}
    auto cursor=offsets;std::vector<double> grouped(values.size());result.medians.resize(count.size());
    for(size_t i=0;i<values.size();++i){
        if(!std::isfinite(values[i]))throw std::invalid_argument("nonfinite grouped median value");
        grouped[cursor[indices.at(groups[i])]++]=values[i];
    }
    objects::parallelFor(count.size(),workers,[&](size_t i){
        result.medians[i]=medianInPlace(std::span(grouped).subspan(offsets[i],offsets[i+1]-offsets[i]));
    });return result;
}
} // namespace vc::geometry
