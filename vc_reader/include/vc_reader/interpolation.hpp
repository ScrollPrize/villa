#pragma once
#include <vc_reader/array.hpp>

namespace vcr {
namespace interpolation_detail {
template<class Coordinates> struct Slice {
    using value_type=typename Coordinates::value_type;
    const Coordinates& source;
    std::size_t first;
    std::int64_t base(std::size_t a) const {return source.base(a);}
    const value_type* operator()(std::size_t p,value_type* out) const {return source(first+p,out);}
    void prefetch(std::size_t p) const {source.prefetch(first+p);}
};
template<class Policy> struct First {
    static constexpr bool custom=true,clamp_boundary=true;
    static constexpr std::size_t components=1;
    using real_type=typename Policy::real_type;
    const Policy& policy;
    std::vector<std::array<real_type,8>>& values;
    std::vector<std::uint8_t>& valid;
    std::int64_t offset(std::size_t,std::size_t) const {return 0;}
    template<class Cell> void prepare_cell(Cell& c) const {
        policy.prepare_cell(c,0);
        if(c.k!=3)throw std::invalid_argument("paired interpolation requires three spatial axes");
    }
    template<class Cell,class Read> auto evaluate(const Cell&,Read&& read) const {
        std::array<real_type,8> out;
        for(std::size_t k=0;k<8;++k)out[k]=read(k,0);
        return out;
    }
    void store(std::size_t p,const std::array<real_type,8>& v) const {values[p]=v;valid[p]=1;}
    void outside(std::size_t p) const {valid[p]=0;}
};
template<class Policy> struct Second {
    static constexpr bool custom=true,clamp_boundary=true;
    static constexpr std::size_t components=1;
    using real_type=typename Policy::real_type;
    const Policy& policy;
    const std::vector<std::array<real_type,8>>& values;
    const std::vector<std::uint8_t>& valid;
    std::size_t first;
    std::int64_t offset(std::size_t,std::size_t) const {return 0;}
    template<class Cell> void prepare_cell(Cell& c) const {policy.prepare_cell(c,1);}
    template<class Cell,class Read> auto evaluate(const Cell& c,Read&& read,std::size_t p) const {
        const auto pair=[&](std::size_t k,std::size_t component){return component==0?values[p][k]:read(k,0);};
        using Result=decltype(policy.evaluate(c,pair));
        return valid[p]?std::optional<Result>{policy.evaluate(c,pair)}:std::nullopt;
    }
    template<class Value> void store(std::size_t p,const Value& v) const {
        if(v)policy.store(first+p,*v);
        else policy.outside(first+p);
    }
    void outside(std::size_t p) const {policy.outside(first+p);}
};
} // namespace interpolation_detail

// Nonlinear interpolation of two encoded fields in independent arrays. Both
// sources use the ordinary chunk traversal. Only one component's corners are
// retained, in a bounded tile, rather than returning corner batches to users.
// Sources must have the same spatial grid; chunk layouts and dtypes may differ.
template<class CoordinatesA,class CoordinatesB,class Policy>
void sample_interpolated_pair_coordinates(Array& a,const CoordinatesA& ca,const ElementType& ta,
    Array& b,const CoordinatesB& cb,const ElementType& tb,std::size_t n,const Policy& policy)
{
    constexpr std::size_t tile=65536;
    struct TileScratch {
        std::vector<std::array<typename Policy::real_type,8>> values;
        std::vector<std::uint8_t> valid;
    };
    a.parallel_for(n/tile+(n%tile!=0),[&](std::size_t part) {
        const auto first=part*tile;
        const auto count=std::min(tile,n-first);
        auto lease=workerScratch().lease();auto& scratch=lease.template resource<TileScratch>();
        auto& values=scratch.values;auto& valid=scratch.valid;
        values.resize(count);valid.resize(count);
        a.sample_interpolated_coordinates(interpolation_detail::Slice<CoordinatesA>{ca,first},count,ta,
            interpolation_detail::First<Policy>{policy,values,valid});
        b.sample_interpolated_coordinates(interpolation_detail::Slice<CoordinatesB>{cb,first},count,tb,
            interpolation_detail::Second<Policy>{policy,values,valid,first});
    });
}
} // namespace vcr
