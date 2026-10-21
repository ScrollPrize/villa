#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vc::fiber_tracer {
// Arc-weighted midpoint quadrature; invariant to either tangent's sign.
template<class AxisAt>
double fiberAxialAlignment(const std::vector<cv::Vec3d>& points, AxisAt axisAt, double spacing=16.) {
    if(!std::isfinite(spacing) || spacing<=0)throw std::invalid_argument("Invalid axial sample spacing");
    double sum=0.,weight=0.;
    for(size_t i=1;i<points.size();++i) {
        const auto delta=points[i]-points[i-1];const double length=cv::norm(delta);
        if(!std::isfinite(length))throw std::invalid_argument("Nonfinite fiber geometry");
        if(length==0)continue;
        const auto tangent=delta/length;
        const int count=std::max(1,int(std::ceil(length/spacing)));
        for(int j=0;j<count;++j) {
            const auto p=points[i-1]+delta*((j+.5)/count);
            const cv::Vec3d axis=axisAt(p);const double norm=cv::norm(axis);
            if(!std::isfinite(norm) || norm<=0)throw std::invalid_argument("Invalid umbilicus tangent");
            const double dot=std::clamp(tangent.dot(axis/norm),-1.,1.);
            sum+=(length/count)*(2*dot*dot-1.);
        }
        weight+=length;
    }
    if(weight==0)throw std::invalid_argument("Zero-length fiber");
    return sum/weight;
}
}
