#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace vc::geometry {
// Shared arc-weighted midpoint quadrature. The point adapter supplies the
// same vector arithmetic and norm as its caller, preserving accumulation.
template<class Points,class AxisAt,class Norm>
double fiberAxialAlignment(const Points& points,AxisAt axisAt,Norm norm,double spacing=16.)
{
    if(!std::isfinite(spacing)||spacing<=0)throw std::invalid_argument("Invalid axial sample spacing");
    double sum=0.,weight=0.;
    for(size_t i=1;i<points.size();++i){
        const auto delta=points[i]-points[i-1];const double length=norm(delta);
        if(!std::isfinite(length))throw std::invalid_argument("Nonfinite fiber geometry");
        if(length==0)continue;
        const auto tangent=delta/length;
        const int count=std::max(1,int(std::ceil(length/spacing)));
        for(int j=0;j<count;++j){
            const auto p=points[i-1]+delta*((j+.5)/count);
            const auto axis=axisAt(p);const double magnitude=norm(axis);
            if(!std::isfinite(magnitude)||magnitude<=0)throw std::invalid_argument("Invalid umbilicus tangent");
            const double dot=std::clamp(tangent.dot(axis/magnitude),-1.,1.);
            sum+=(length/count)*(2*dot*dot-1.);
        }
        weight+=length;
    }
    if(weight==0)throw std::invalid_argument("Zero-length fiber");
    return sum/weight;
}

// Existing umbilicus-center interpolation, shared by VC's dense-center
// object and the bulk axial-prior adapter. Input points are sorted by Z and
// retain their float32 arithmetic/endpoint conventions.
template<class Points>
auto umbilicusCenter(const Points& points,double z)
{
    using Point=std::decay_t<decltype(points[0])>;
    if(points.empty())throw std::logic_error("Umbilicus interpolation requested without control points");
    const auto at=[&](Point p){p[2]=static_cast<float>(z);return p;};
    if(points.size()==1||z<=points.front()[2])return at(points.front());
    if(z>=points.back()[2])return at(points.back());
    const float target=static_cast<float>(z);
    const auto upper=std::lower_bound(points.begin(),points.end(),target,[](const Point& p,float value){return p[2]<value;});
    if(upper==points.begin())return at(*upper);
    if(upper!=points.end()&&std::abs((*upper)[2]-target)<1e-5f)return at(*upper);
    const auto& right=upper==points.end()?points.back():*upper;const auto& left=*(upper-1);
    const double z0=left[2],z1=right[2];
    if(std::abs(z1-z0)<1e-5)return at(left);
    const double t=(z-z0)/(z1-z0);
    const double x=left[0]+t*(right[0]-left[0]);const double y=left[1]+t*(right[1]-left[1]);
    return Point{static_cast<float>(x),static_cast<float>(y),static_cast<float>(z)};
}
} // namespace vc::geometry
