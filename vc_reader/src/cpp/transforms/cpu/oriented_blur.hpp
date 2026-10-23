#pragma once
#include "../ops/oriented_blur.hpp"
#include "../core.hpp"

namespace vcr::transforms {
inline ChannelPlan channel_parameters(const OrientedBlur& t, const Array& a, Random& r) {
    const auto dim=a.shape.size()-1;
    require(dim==2 || dim==3,"OrientedBlur requires 2D or 3D images");
    require(t.axis.empty() || t.axis.size()==dim,"axis must match spatial dimensions");
    const double along = draw(t.sigma_along, r), across = draw(t.sigma_across, r);
    require(along >= 0 && across >= 0 && along < 1000 && across < 1000, "blur sigmas must be in [0, 1000)");
    ChannelPlan p; p.values = {along, across};
    if (!t.axis.empty()) {p.values.insert(p.values.end(), t.axis.begin(), t.axis.end());if(dim==2)p.values.push_back(0);}
    else if (t.axis_from.empty()) {
        if(dim==2){const double phi=6.2831853071795864769*r.uniform();p.values.insert(p.values.end(),{std::cos(phi),std::sin(phi),0});return p;}
        const double z = 2 * r.uniform() - 1, phi = 6.2831853071795864769 * r.uniform(), rho = std::sqrt(std::max(0., 1 - z * z));
        p.values.insert(p.values.end(), {z, rho * std::cos(phi), rho * std::sin(phi)});
    }
    return p;
}
namespace oriented_detail {
// Kernel radius max(1, ceil(3 sigma)); weights normalized in double.
template<class T = double> inline std::vector<T> kernel(double sigma) {
    const int radius = std::max(1, int(std::ceil(3 * sigma)));
    std::vector<T> w(2 * radius + 1); double sum = 0;
    for (int k = -radius; k <= radius; ++k) sum += w[k + radius] = std::exp(-.5 * (k / sigma) * (k / sigma));
    for (auto& x : w) x /= std::max(sum, 1e-12);
    return w;
}
VCR_HD inline void normalize3(double* v) {
    const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), d = n > 1e-12 ? n : 1e-12;
    for (int i = 0; i < 3; ++i) v[i] /= d;
}
VCR_HD inline void cross3(const double* a, const double* b, double* out) {
    out[0] = a[1] * b[2] - a[2] * b[1]; out[1] = a[2] * b[0] - a[0] * b[2]; out[2] = a[0] * b[1] - a[1] * b[0];
}
// Rows: along, across_a, across_b, from a zyx axis.
template<int D=3> VCR_HD inline void basis(const double* axis, double* out) {
    if constexpr(D==2) {
        const double norm=std::max(std::sqrt(axis[0]*axis[0]+axis[1]*axis[1]),1e-12);
        out[0]=axis[0]/norm;out[1]=axis[1]/norm;out[2]=-out[1];out[3]=out[0];
        return;
    }
    double* along = out; double* a = out + 3; double* b = out + 6;
    for (int i = 0; i < 3; ++i) along[i] = axis[i];
    normalize3(along);
    double ref[3] = {1, 0, 0};
    if (std::fabs(along[0]) > .9) { ref[0] = 0; ref[1] = 1; }
    cross3(along, ref, a); normalize3(a);
    cross3(along, a, b); normalize3(b);
}
// Multilinear sample at a yx/zyx position, coordinates clamped into the grid.
template<class T,class Real,int D=3> VCR_HD T sample(const T* src, const std::size_t* shape, const Real* q) {
    long long lo[3]; Real f[3];
    for (int a = 0; a < D; ++a) {
        const Real x = q[a] < 0 ? 0 : q[a] > Real(shape[a] - 1) ? Real(shape[a] - 1) : q[a];
        lo[a] = (long long)std::floor(x); f[a] = x - Real(lo[a]);
    }
    T value = 0;
    for (int corner = 0; corner < (1<<D); ++corner) {
        T w = 1; std::size_t at = 0;
        for (int a = 0; a < D; ++a) {
            const int bit = (corner >> (D - 1 - a)) & 1;
            w *= bit ? f[a] : 1 - f[a];
            long long x = lo[a] + bit; if (x >= (long long)shape[a]) x = (long long)shape[a] - 1;
            at = at * shape[a] + std::size_t(x);
        }
        if (w) value += w * T(src[at]);
    }
    return value;
}
// One pass for voxel (z, y, x): sum of weighted samples along direction u.
template<class T,class Real,int D=3> VCR_HD T pass(const T* src, const std::size_t* shape, std::size_t z, std::size_t y, std::size_t x,
                                     const double* u, const T* w, int radius) {
    T sum = 0;
    for (int k = -radius; k <= radius; ++k) {
        Real q[3];
        if constexpr(D==3){q[0]=Real(z)+Real(k)*Real(u[0]);q[1]=Real(y)+Real(k)*Real(u[1]);q[2]=Real(x)+Real(k)*Real(u[2]);}
        else {q[0]=Real(y)+Real(k)*Real(u[0]);q[1]=Real(x)+Real(k)*Real(u[1]);}
        sum += w[k + radius] * sample<T,Real,D>(src, shape, q);
    }
    return sum;
}
template<int D=3> VCR_HD inline bool wide_coordinates(const std::size_t* shape,int radius) {
    const auto extent=D==3?std::max(shape[0],std::max(shape[1],shape[2])):std::max(shape[0],shape[1]);
    return (double(extent)+radius)*8*std::numeric_limits<float>::epsilon()>1e-3;
}
// The axis of one channel plan: explicit/random values, or the target's first row.
inline std::array<double, 3> axis(const OrientedBlur& t, const Sample& s, const ChannelPlan& c, const Array* vectors) {
    if (t.axis_from.empty()) return {c.values[2], c.values[3], c.values[4]};
    const auto dim=s.at(c.key).array.spatial().size();
    const auto& f = s.at(t.axis_from);
    require(f.target.role == Role::Vectors && f.array.shape.size() == 2 && f.array.shape[0] >= 1 && f.array.shape[1] == dim,
            "OrientedBlur axis_from must name a matching Vectors target with at least one row");
    std::array<double, 3> v{};
    for (std::size_t j = 0; j < dim; ++j) v[std::size_t(vcr::transforms::axis(f.target.components[j], dim))] = vectors->get(j);
    return v;
}
} // namespace oriented_detail
inline void apply(const OrientedBlur& t, Sample& s, const Parameters& p) {
    for (const auto& c : p.channels) {
        auto& a = s.at(c.key).array; const auto shape = a.spatial(); Grid g(shape);
        const auto run_dimension=[&]<int D>() {
        double frame[D*D]; const auto axis = oriented_detail::axis(t, s, c, t.axis_from.empty() ? nullptr : &s.at(t.axis_from).array);
        oriented_detail::basis<D>(axis.data(), frame);
        const auto run = [&]<class T>() {
        auto values = a.channel<T>(c.channel); std::vector<T> next(values.size());
        for (int pass = 0; pass < D; ++pass) {
            const double sigma = c.values[pass ? 1 : 0];
            if (sigma <= 0) continue;
            const auto w = oriented_detail::kernel<T>(sigma); const int radius = int(w.size() / 2);
            const auto* direction=frame+D*pass;
            int aligned=-1;
            for(int axis=0;axis<D;++axis) {
                if constexpr(D==3){if(std::abs(direction[axis])==1 && direction[(axis+1)%3]==0 && direction[(axis+2)%3]==0)aligned=axis;}
                else {if(std::abs(direction[axis])==1 && direction[1-axis]==0)aligned=axis;}
            }
            if(aligned>=0){
                // Integer axis-aligned samples need no trilinear interpolation.
                // Negative directions retain the original reversed tap order.
                convolve_axis(values,next,g,aligned,w,Boundary::Nearest,direction[aligned]<0);
            }else{
                const bool wide = sizeof(T)==8 || oriented_detail::wide_coordinates<D>(shape.data(),radius);
                for (std::size_t i = 0; i < g.size; ++i)
                    next[i] = wide ? oriented_detail::pass<T,double,D>(values.data(), shape.data(), D==3?g.coord(i,0):0, g.coord(i,D-2), g.coord(i,D-1), direction, w.data(), radius)
                                   : oriented_detail::pass<T,float,D>(values.data(), shape.data(), D==3?g.coord(i,0):0, g.coord(i,D-2), g.coord(i,D-1), direction, w.data(), radius);
            }
            values.swap(next);
        }
        a.channel(c.channel, values);
        };
        bool wide=a.bytes==8;
        if(!wide) {
            const auto* data=static_cast<const float*>(std::as_const(a).data())+c.channel*g.size;
            wide=std::any_of(data,data+g.size,[](float x){return std::abs(double(x))>std::numeric_limits<float>::max()/1024.;});
        }
        if(wide)run.template operator()<double>();else run.template operator()<float>();
        };
        if(shape.size()==3)run_dimension.template operator()<3>();else run_dimension.template operator()<2>();
    }
    restore_valid(s);
}
} // namespace vcr::transforms
