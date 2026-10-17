#pragma once
#include "core.hpp"

namespace vcr::augment::elastic_detail {
inline std::array<double,4> weights(double t,bool cubic) {
    if (!cubic) return {1-t,t,0,0};
    return {std::pow(1-t,3)/6,(3*t*t*t-6*t*t+4)/6,
            (-3*t*t*t+3*t*t+3*t+1)/6,t*t*t/6};
}
// Tensor-product linear or cardinal cubic B-spline expansion, in dimension-wise
// separable passes. Controls have a one-knot left halo for cubic interpolation.
inline std::vector<double> expand(std::vector<double> values, Shape shape,
                                  const Shape& target_shape, double spacing, bool cubic,
                                  std::size_t first_axis=0) {
    require(shape.size()==target_shape.size() && spacing>=1 && std::isfinite(spacing), "invalid lattice");
    require(values.size()==product(shape), "control count mismatch");
    for (int axis=int(shape.size())-1;axis>=int(first_axis);--axis) {
        const auto edge=target_shape[axis];
        Grid input(shape); auto target=shape; target[axis]=edge; Grid output(target);
        std::vector<double> next(output.size);
        const auto step=input.stride[axis], blocks=input.size/(shape[axis]*step);
        for (std::size_t x=0;x<edge;++x) {
            const double q=x/spacing, t=q-std::floor(q);
            const auto lo=std::size_t(q); const auto taps=cubic ? 4u : 2u;
            require(lo+taps<=shape[axis], "control lattice lacks halo");
            const auto w=weights(t,cubic);
            for (std::size_t b=0;b<blocks;++b) {
                const auto* src=values.data()+(b*shape[axis]+lo)*step;
                auto* dst=next.data()+(b*edge+x)*step;
                for (std::size_t j=0;j<step;++j) {
                    double value=0;
                    for (std::size_t tap=0;tap<taps;++tap) value+=w[tap]*src[tap*step+j];
                    dst[j]=value;
                }
            }
        }
        shape=std::move(target); values=std::move(next);
    }
    return values;
}
inline std::size_t controls(std::size_t edge,double spacing,bool cubic) {
    return std::size_t(std::floor((edge-1)/spacing))+(cubic ? 4 : 2);
}

// Component-major control draws match the inspected experimental B-spline field.
template<std::size_t D>
std::vector<double> bspline_field_impl(const Shape& shape,double spacing,Random& rng) {
    Grid g(shape); constexpr auto d=D; Shape knots;
    for (auto n:shape) knots.push_back(controls(n,spacing,true));
    std::array<std::vector<double>,D> components;
    for (auto& v:components) {
        v.resize(product(knots)); for (auto& value:v) value=rng.normal();
        // Leave the outermost interpolation for the final interleaved write.
        // This avoids allocating and copying D full-resolution component arrays.
        v=expand(std::move(v),knots,shape,spacing,true,1);
    }
    std::vector<double> field(d*g.size);
    std::array<double,D> sums{};
    const auto slab=g.stride[0];
    for (std::size_t z=0;z<shape[0];++z) {
        const double q=z/spacing; const auto lo=std::size_t(q);
        const auto w=weights(q-std::floor(q),true);
        for (std::size_t i=0;i<slab;++i) for (std::size_t c=0;c<d;++c) {
            const auto* src=components[c].data()+lo*slab+i;
            double value=0;
            for (std::size_t tap=0;tap<4;++tap) value+=w[tap]*src[tap*slab];
            field[(z*slab+i)*d+c]=value;
            sums[c]+=value; // Same per-component reduction order as mean().
        }
    }
    for (auto& sum:sums) sum/=g.size;
    for (std::size_t i=0;i<g.size;++i) for (std::size_t c=0;c<d;++c) field[d*i+c]-=sums[c];
    return field;
}
inline std::vector<double> bspline_field(const Shape& shape,double spacing,Random& rng) {
    require(shape.size()==2 || shape.size()==3,"elastic needs a 2D or 3D grid");
    require(std::isfinite(spacing) && spacing>=1,"invalid knot spacing");
    return shape.size()==3 ? bspline_field_impl<3>(shape,spacing,rng) : bspline_field_impl<2>(shape,spacing,rng);
}

// Elastic strength convention: RMS vector displacement in
// output voxels. Limit a conservative infinity-norm Lipschitz bound to <1.
// For trilinear interpolation, each derivative is a convex combination of
// grid-edge differences. The sum of their componentwise maxima bounds every
// Jacobian row, including between voxels; identity+field is then injective.
template<std::size_t D>
double normalize_impl(std::vector<double>& field,const Shape& shape,double rms,double bound) {
    require(rms>=0 && std::isfinite(rms) && bound>0 && bound<1,"invalid strength/gradient bound");
    Grid g(shape); constexpr auto d=D; require(field.size()==d*g.size,"field shape mismatch");
    // Independent accumulators allow SIMD without enabling global fast-math.
    // The reduction is deterministic, but may differ by a few rounding bits
    // from the old serial sum. Validate before using the resulting scale.
    constexpr std::size_t energy_lanes=8;
    std::array<double,energy_lanes> energies{};
    bool finite=true;
    std::size_t i=0;
    for (;i+energy_lanes<=field.size();i+=energy_lanes)
        for (std::size_t lane=0;lane<energy_lanes;++lane) {
            const double v=field[i+lane];
            finite &= std::isfinite(v);
            energies[lane]+=v*v;
        }
    double energy=std::accumulate(energies.begin(),energies.end(),0.);
    for (;i<field.size();++i) { finite &= std::isfinite(field[i]); energy+=field[i]*field[i]; }
    require(finite,"nonfinite displacement");
    double factor=energy ? rms/std::sqrt(energy/g.size) : 0;
    double maxima[3][3]{};
    // Visit all forward neighbors together, while the current row and adjacent
    // rows/planes are cached, instead of rereading the entire field per axis.
    // Maxima of finite absolute differences are independent of reduction order.
    constexpr std::size_t lanes=2;
    double lane_max[D][lanes][D]{};
    const auto nx=shape.back(), ny=shape[d-2], nz=d==3 ? shape[0] : 1;
    for (std::size_t z=0;z<nz;++z) for (std::size_t y=0;y<ny;++y) {
        const auto base=(z*ny+y)*nx;
        auto accumulate=[&](std::size_t x,std::size_t count,bool right) {
            for (std::size_t lane=0;lane<count;++lane) for (std::size_t c=0;c<d;++c) {
                const auto index=d*(base+x+lane)+c;
                const auto current=field[index];
                if constexpr (d==3) if (z+1<nz)
                    lane_max[0][lane][c]=std::max(lane_max[0][lane][c],std::abs(field[index+d*ny*nx]-current));
                if (y+1<ny)
                    lane_max[d-2][lane][c]=std::max(lane_max[d-2][lane][c],std::abs(field[index+d*nx]-current));
                if (right)
                    lane_max[d-1][lane][c]=std::max(lane_max[d-1][lane][c],std::abs(field[index+d]-current));
            }
        };
        std::size_t x=0;
        for (;x+lanes<nx;x+=lanes) accumulate(x,lanes,true);
        for (;x<nx;++x) accumulate(x,1,x+1<nx);
    }
    for (std::size_t a=0;a<d;++a) for (std::size_t lane=0;lane<lanes;++lane)
        for (std::size_t c=0;c<d;++c) maxima[c][a]=std::max(maxima[c][a],lane_max[a][lane][c]);
    double limit=0;
    for (const auto& row:maxima) limit=std::max(limit,row[0]+row[1]+row[2]);
    if (limit) factor=std::min(factor,bound/limit);
    for (auto& v:field) v*=factor;
    return factor;
}
inline double normalize(std::vector<double>& field,const Shape& shape,double rms,double bound) {
    require(shape.size()==2 || shape.size()==3,"elastic needs a 2D or 3D grid");
    return shape.size()==3 ? normalize_impl<3>(field,shape,rms,bound) : normalize_impl<2>(field,shape,rms,bound);
}
} // namespace vcr::augment::elastic_detail
