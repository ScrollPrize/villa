#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <cstdint>
#include <limits>
#include <vector>

namespace vc::fiber_tracer {

// Generic extensible storage convention. Unknown attributes are preserved by
// the JSON/MessagePack container; known consumers validate their own semantics.
inline nlohmann::json fiberAttribute(const nlohmann::json& values,
                                    const char* units, const char* meaning) {
    return {{"values",values},{"units",units},{"meaning",meaning}};
}
inline void validateFiberAttributes(const nlohmann::json& piece) {
    const auto n=piece.at("points_base").size();
    if(n<2) throw std::invalid_argument("fiber needs at least two points");
    for(const auto* group:{"point_attributes","segment_attributes","fiber_attributes"}) {
        const auto& attributes=piece.at(group);
        if(!attributes.is_object()) throw std::invalid_argument("fiber attributes must be objects");
        for(const auto& [name,a]:attributes.items()) {
            (void)name;
            if(!a.is_object() || !a.at("units").is_string() || !a.at("meaning").is_string())
                throw std::invalid_argument("attribute needs units and meaning");
            const auto& values=a.at("values");
            if(std::string_view(group)!="fiber_attributes" &&
               (!values.is_array() || values.size()!=(std::string_view(group)=="point_attributes"?n:n-1)))
                throw std::invalid_argument("fiber attribute cardinality mismatch");
        }
    }
}

// Point observations and incoming-segment observations from committed tracing.
// Lengths retain source arc measure when resampling replaces a corner by a chord.
struct FiberPathAttributes {
    std::vector<double> widths;
    std::vector<cv::Vec3d> normals;
    std::vector<uint8_t> normalValid;
    std::vector<double> quality, costs, lengths;
    void check(size_t n) const {
        if(!n || widths.size()!=n || normals.size()!=n || normalValid.size()!=n ||
           quality.size()!=n-1 || costs.size()!=n-1 || lengths.size()!=n-1)
            throw std::invalid_argument("trace observation cardinality mismatch");
    }
};

// Reverse the backward half, then append the forward half without its seed.
// Normals are axes, so reversing traversal does not negate them. Segment
// observations belong to edges: reverse their order without changing signs.
inline FiberPathAttributes joinFiberAttributes(const FiberPathAttributes& bw,const FiberPathAttributes& fw) {
    bw.check(bw.widths.size());fw.check(fw.widths.size());FiberPathAttributes out;
    const auto points=[](auto& dest,const auto& back,const auto& front) {
        dest.assign(back.rbegin(),back.rend());dest.insert(dest.end(),front.begin()+1,front.end());
    };
    const auto segments=[](auto& dest,const auto& back,const auto& front) {
        dest.assign(back.rbegin(),back.rend());dest.insert(dest.end(),front.begin(),front.end());
    };
    points(out.widths,bw.widths,fw.widths);points(out.normals,bw.normals,fw.normals);
    points(out.normalValid,bw.normalValid,fw.normalValid);
    const auto seed=bw.widths.size()-1;
    if(!out.normalValid[seed] && fw.normalValid.front()) {
        out.normals[seed]=fw.normals.front();out.normalValid[seed]=true;
    }
    segments(out.quality,bw.quality,fw.quality);segments(out.costs,bw.costs,fw.costs);
    segments(out.lengths,bw.lengths,fw.lengths);return out;
}

inline std::vector<double> fiberArc(const std::vector<cv::Vec3d>& points) {
    std::vector<double> arc(points.size(),0);
    for(size_t i=1;i<points.size();++i) arc[i]=arc[i-1]+cv::norm(points[i]-points[i-1]);
    return arc;
}

// Re-bin recorded scores, never re-score the path. Costs are additive and
// apportioned by overlap; quality is a source-length-weighted segment mean.
inline FiberPathAttributes resampleFiberAttributes(const std::vector<double>& arc,
    const FiberPathAttributes& in,const std::vector<double>& at) {
    in.check(arc.size());
    if(at.empty() || !std::is_sorted(at.begin(),at.end()) || at.front()<0 || at.back()>arc.back()+1e-6)
        throw std::invalid_argument("invalid attribute resampling positions");
    FiberPathAttributes out;
    std::vector<double> cq(arc.size(),0),cc=cq,cl=cq;
    for(size_t i=1;i<arc.size();++i) {
        cq[i]=cq[i-1]+in.quality[i-1]*in.lengths[i-1];
        cc[i]=cc[i-1]+in.costs[i-1];cl[i]=cl[i-1]+in.lengths[i-1];
    }
    size_t j=0;double pq=0,pc=0,pl=0;
    for(double a:at) {
        while(j+2<arc.size() && arc[j+1]<a)++j;
        const size_t k=std::min(j+1,arc.size()-1);
        const double t=arc[k]>arc[j]?std::clamp((a-arc[j])/(arc[k]-arc[j]),0.,1.):0;
        out.widths.push_back(in.widths[j]*(1-t)+in.widths[k]*t);
        auto b=in.normals[k];if(in.normals[j].dot(b)<0)b=-b;
        auto normal=in.normals[j]*(1-t)+b*t;const double norm=cv::norm(normal);
        const bool valid=(t==0?bool(in.normalValid[j]):t==1?bool(in.normalValid[k]):
            bool(in.normalValid[j]&&in.normalValid[k])) && norm>1e-12;
        out.normals.push_back(valid?normal/norm:cv::Vec3d{});out.normalValid.push_back(valid);
        const double q=cq[j]*(1-t)+cq[k]*t,c=cc[j]*(1-t)+cc[k]*t,l=cl[j]*(1-t)+cl[k]*t;
        if(out.widths.size()>1) {
            out.lengths.push_back(l-pl);out.costs.push_back(c-pc);
            out.quality.push_back(l>pl?(q-pq)/(l-pl):0);
        }
        pq=q;pc=c;pl=l;
    }
    return out;
}

inline void attachFiberAttributes(nlohmann::json& piece,const FiberPathAttributes& data,size_t a,size_t b) {
    data.check(data.widths.size());
    if(b>data.widths.size() || b<a+2 || piece.at("points_base").size()!=b-a)
        throw std::invalid_argument("invalid fiber attribute slice");
    using Json=nlohmann::json;
    Json normals=Json::array();
    for(size_t i=a;i<b;++i) normals.push_back(data.normalValid[i]?
        Json{data.normals[i][0],data.normals[i][1],data.normals[i][2]}:Json(nullptr));
    piece["point_attributes"]={
        {"fiber_normal",fiberAttribute(normals,"unit_vector_xyz","Predicted normal axis at committed points; sign-aligned interpolation; null if invalid")},
        {"width",fiberAttribute(std::vector<double>(data.widths.begin()+a,data.widths.begin()+b),"base_voxel","Local predicted width, interpolated between committed samples")}};
    piece["segment_attributes"]={
        {"modulated_presence",fiberAttribute(std::vector<double>(data.quality.begin()+a,data.quality.begin()+b-1),"dimensionless","Actual chosen-step direction/width-modulated presence; source-arc weighted when resampled")},
        {"tracer_cost",fiberAttribute(std::vector<double>(data.costs.begin()+a,data.costs.begin()+b-1),"cost","Actual chosen-step objective including smoothness; apportioned by source-arc overlap")},
        {"traced_length",fiberAttribute(std::vector<double>(data.lengths.begin()+a,data.lengths.begin()+b-1),"base_voxel","Original traced arc represented by this saved segment")}};
    double length=0,width=0,quality=0,cost=0;
    for(size_t i=a;i+1<b;++i) {
        const auto l=data.lengths[i];length+=l;width+=l*(data.widths[i]+data.widths[i+1])/2;
        quality+=l*data.quality[i];cost+=data.costs[i];
    }
    if(!(length>0))throw std::invalid_argument("zero length fiber attributes");
    piece["fiber_attributes"]={
        {"traced_length",fiberAttribute(length,"base_voxel","Sum of represented source-arc lengths for this piece")},
        {"width_mean",fiberAttribute(width/length,"base_voxel","Piece-local length-weighted mean of local widths")},
        {"modulated_presence_mean",fiberAttribute(quality/length,"dimensionless","Piece-local length-weighted mean of recorded quality")},
        {"modulated_presence_integral",fiberAttribute(quality,"base_voxel","Integral of recorded quality along this piece")},
        {"tracer_cost_sum",fiberAttribute(cost,"cost","Sum of apportioned chosen-step costs for this piece")},
        {"tracer_cost_per_base_voxel",fiberAttribute(cost/length,"cost/base_voxel","Piece-local tracer cost divided by represented arc length")}};
    validateFiberAttributes(piece);
}

// Run-specific requirements are separate from generic additive compatibility.
inline void validateStreamFiberAttributes(const nlohmann::json& piece) {
    validateFiberAttributes(piece);
    for(const auto* key:{"fiber_normal","width"}) (void)piece.at("point_attributes").at(key);
    for(const auto* key:{"modulated_presence","tracer_cost","traced_length"}) (void)piece.at("segment_attributes").at(key);
    for(const auto* key:{"width_mean","modulated_presence_mean","modulated_presence_integral","tracer_cost_sum","tracer_cost_per_base_voxel","traced_length"})
        (void)piece.at("fiber_attributes").at(key);
    const auto numbers=[](const nlohmann::json& a,const char* units,double low,double high) {
        if(a.at("units")!=units)throw std::invalid_argument("fiber attribute units mismatch");
        const auto check=[&](const nlohmann::json& v) {
            if(!v.is_number() || !std::isfinite(v.get<double>()) || v.get<double>()<low || v.get<double>()>high)
                throw std::invalid_argument("invalid fiber attribute value");
        };
        const auto& values=a.at("values");
        if(values.is_array())for(const auto& v:values)check(v);else check(values);
    };
    const auto inf=std::numeric_limits<double>::infinity();
    numbers(piece.at("point_attributes").at("width"),"base_voxel",0,inf);
    const auto& normals=piece.at("point_attributes").at("fiber_normal");
    if(normals.at("units")!="unit_vector_xyz")throw std::invalid_argument("fiber normal units mismatch");
    for(const auto& n:normals.at("values")) {
        if(n.is_null())continue;
        if(!n.is_array() || n.size()!=3)throw std::invalid_argument("fiber normal needs XYZ");
        double norm=0;
        for(const auto& v:n){if(!v.is_number() || !std::isfinite(v.get<double>()))throw std::invalid_argument("invalid normal");norm+=v.get<double>()*v.get<double>();}
        if(std::abs(norm-1)>1e-4)throw std::invalid_argument("fiber normal is not a unit axis");
    }
    numbers(piece.at("segment_attributes").at("modulated_presence"),"dimensionless",-1e-6,1+1e-6);
    numbers(piece.at("segment_attributes").at("tracer_cost"),"cost",-1e-9,inf);
    numbers(piece.at("segment_attributes").at("traced_length"),"base_voxel",0,inf);
    numbers(piece.at("fiber_attributes").at("width_mean"),"base_voxel",0,inf);
    numbers(piece.at("fiber_attributes").at("modulated_presence_mean"),"dimensionless",-1e-6,1+1e-6);
    numbers(piece.at("fiber_attributes").at("modulated_presence_integral"),"base_voxel",-1e-9,inf);
    numbers(piece.at("fiber_attributes").at("tracer_cost_sum"),"cost",-1e-9,inf);
    numbers(piece.at("fiber_attributes").at("tracer_cost_per_base_voxel"),"cost/base_voxel",-1e-9,inf);
    numbers(piece.at("fiber_attributes").at("traced_length"),"base_voxel",0,inf);
}
} // namespace vc::fiber_tracer
