#pragma once
#include "../ops/ridges.hpp"
#include "structure_tensor.hpp"
namespace vcr::transforms {
namespace ridge_detail {
// NumPy gradient(edge_order=2): centered interior and one-sided boundaries.
template<class T> VCR_HD inline T derivative(const T* v, std::size_t i, std::size_t x, std::size_t n, std::size_t stride) {
    if (x == 0) return T(-1.5)*v[i] + T(2)*v[i+stride] - T(.5)*v[i+2*stride];
    if (x+1 == n) return T(1.5)*v[i] - T(2)*v[i-stride] + T(.5)*v[i-2*stride];
    return (v[i+stride]-v[i-stride])*T(.5);
}
VCR_HD inline double response(double* h, int d, double sigma, double gamma, double beta1, double beta2) {
    double trace=0; for(int j=0;j<d;++j)trace+=h[j*d+j];
    if(trace==0)return 0;
    for(int j=0;j<d*d;++j)h[j]*=sigma*sigma;
    double values[3], vectors[9]; vcr::math::symmetric_eigen(h,d,values,vectors);
    for(int j=0;j<d;++j)for(int k=j+1;k<d;++k)if(fabs(values[k])<fabs(values[j])) {double tmp=values[j];values[j]=values[k];values[k]=tmp;}
    if(values[d-1]>0)return 0;
    double norm=0;for(int j=0;j<d;++j)norm+=values[j]*values[j];
    const double largest=fabs(values[d-1]), second=fabs(values[d-2]);
    const double ra=second/(largest==0?1e-10:largest);
    const double denominator=d==2?largest:sqrt(second*largest);
    const double rb=fabs(values[0])/(denominator==0?1e-10:denominator);
    const double beta=d==2?beta1:beta2;
    return (1-exp(-.5*norm/(gamma*gamma)))*exp(-.5*rb*rb/(beta*beta))*(d==2?1:exp(-.5*ra*ra/(beta1*beta1)));
}
}
inline Parameters sample(const Ridges& t, const Sample& s, Random& r) {
    return sample_channels(s,t.options,r,[](const Array&, Random&){return ChannelPlan{};});
}
inline void apply(const Ridges& t, Sample& s, const Parameters& p) {
    for(const auto& c:p.channels) {
        auto& a=s.at(c.key).array; Grid g(a.spatial());const int d=int(g.shape.size());
        for(auto n:g.shape)require(n>=3,"Ridges needs at least three samples on each spatial axis");
        const auto run=[&](auto tag) {
            using T=decltype(tag);auto input=a.channel<T>(c.channel);std::vector<T> first(g.size);
            std::vector<std::vector<T>> h(d*d,std::vector<T>(g.size));
            for(int q=0;q<d;++q) {
                for(std::size_t i=0;i<g.size;++i)first[i]=ridge_detail::derivative(input.data(),i,g.coord(i,q),g.shape[q],g.stride[q]);
                for(int r=0;r<=q;++r)for(std::size_t i=0;i<g.size;++i)
                    h[r*d+q][i]=h[q*d+r][i]=ridge_detail::derivative(first.data(),i,g.coord(i,r),g.shape[r],g.stride[r]);
            }
            for(std::size_t i=0;i<g.size;++i) {
                double matrix[9];for(int j=0;j<d*d;++j)matrix[j]=h[j][i];
                input[i]=T(ridge_detail::response(matrix,d,t.sigma,t.gamma,t.beta1,t.beta2));
            }
            a.channel(c.channel,input);
        };
        if(a.bytes==4)run(float{});else run(double{});
    }
    restore_valid(s);
}
}
