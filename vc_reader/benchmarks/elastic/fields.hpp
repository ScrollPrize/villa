#pragma once
#include "spatial.hpp"

namespace vcr::elastic_experiment {
using namespace vcr::augment;

inline std::vector<double> expand(std::vector<double> values,Shape shape,std::size_t edge,double spacing,bool cubic) {
    return elastic_detail::expand(std::move(values),std::move(shape),{edge,edge,edge},spacing,cubic);
}
using elastic_detail::controls;

inline std::vector<double> make_field(const std::string& method, std::size_t edge,
                                     double scale, std::uint64_t seed,
                                     std::span<const double> normals={}) {
    require(edge>=4 && edge<=256 && scale>=2 && scale<=64 && std::isfinite(scale), "invalid field size/scale");
    Random rng{seed}; Grid out({edge,edge,edge});
    if (method=="bspline") return elastic_detail::bspline_field(out.shape,2*scale,rng);
    const bool dense=method=="dense", cubic=method=="bspline" || method=="normal_bspline";
    require(dense || cubic || method=="coarse", "unknown elastic method");
    // Match nominal scale approximately; measured correlation is reported.
    const double spacing=cubic ? 2*scale : 4;
    const auto n=controls(edge,spacing,cubic);
    Grid lattice({n,n,n}); std::array<std::vector<double>,3> component;
    for (auto& v:component) {
        v.resize(dense ? out.size : lattice.size);
        for (auto& value:v) value=rng.normal();
    }
    if (method=="normal_bspline") {
        require(normals.size()==3*lattice.size,"normal lattice shape mismatch");
        for (std::size_t i=0;i<lattice.size;++i) {
            double norm=0,dot=0;
            for (int a=0;a<3;++a) { norm+=normals[3*i+a]*normals[3*i+a]; dot+=normals[3*i+a]*component[a][i]; }
            if (norm>.5) for (int a=0;a<3;++a)
                // n*n^T is sign invariant. Retain 10% tangent displacement.
                component[a][i]=.1*component[a][i]+.9*dot*normals[3*i+a];
        }
    }
    std::vector<double> field(3*out.size);
    for (int a=0;a<3;++a) {
        auto& v=component[a];
        if (dense) gaussian_blur(v,out,{scale,scale,scale},6,true,true);
        else {
            if (!cubic) gaussian_blur(v,lattice,{scale/spacing,scale/spacing,scale/spacing},6,true,true);
            v=expand(std::move(v),lattice.shape,edge,spacing,cubic);
        }
        const double m=mean(v);
        for (std::size_t i=0;i<out.size;++i) field[3*i+a]=v[i]-m;
    }
    return field;
}

inline double normalize(std::vector<double>& field,std::size_t edge,double rms,double bound) {
    return elastic_detail::normalize(field,{edge,edge,edge},rms,bound);
}
} // namespace vcr::elastic_experiment
