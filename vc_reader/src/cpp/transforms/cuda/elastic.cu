#include "geometry.cuh"
#include "local_field.cuh"
#include "reductions.cuh"
namespace vcr::transforms::cuda {
// The last separable pass expands only the outer axis. One thread writes all
// components of a voxel, sharing its position and four weights. The generic
// pass would decompose the same index and fetch those weights D times.
template<int D, class Index> __device__ void expand_outer(KernelIndex vcr_launch, const float* src, float* dst,
        std::size_t input_size, std::size_t output_size, std::size_t slab,
        const float* weights, const std::size_t* positions) {
    EACH_INDEX(output_size) {
        const Index z=Index(i)/Index(slab), j=Index(i)%Index(slab);
        const Index base=Index(positions[z])*Index(slab)+j;
        const float w0=weights[4*z], w1=weights[4*z+1], w2=weights[4*z+2], w3=weights[4*z+3];
        #pragma unroll
        for (int c=0;c<D;++c) {
            const float* from=src+Index(c)*Index(input_size)+base;
            float value=w0*from[0];
            value+=w1*from[slab]; value+=w2*from[2*slab]; value+=w3*from[3*slab];
            dst[D*i+c]=value;
        }
    }
}
template<int D, class Index> struct expand_outer_launch2 {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { expand_outer<D,Index>(index,args...); }
};
template<class Index, class Real> __device__ void expand_controls(KernelIndex vcr_launch, const Real* src, Real* dst, Layout input, Layout output,
                                const Real* weights, const std::size_t* positions, int axis) {
    EACH_INDEX(output.size*output.dim) {
        const Index size=Index(output.size), component=Index(i)/size, at=Index(i)%size, step=Index(output.stride[axis]);
        const Index x=(at/step)%Index(output.shape[axis]), block=at/(Index(output.shape[axis])*step), j=at%step;
        const auto* from = src+component*Index(input.size)+(block*Index(input.shape[axis])+Index(positions[x]))*step+j;
        Real value = 0;
        for (int tap = 0; tap < 4; ++tap) value += weights[std::size_t(4)*x+tap]*from[tap*step];
        dst[axis == 0 ? at*Index(output.dim)+component : Index(i)] = value;
    }
}
template<class Index> struct expand_controls_launch1 {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { expand_controls<Index>(index,args...); }
};
template<class Real> __device__ void component_sums(KernelIndex vcr_launch, const Real* field, Real* partial, Layout g) {
    __shared__ Real values[3][256];
    for (int c = 0; c < 3; ++c) values[c][threadIdx.x] = 0;
    EACH_INDEX(g.size) for (int c = 0; c < g.dim; ++c) values[c][threadIdx.x] += field[i*g.dim+c];
    block_reduce(values);
    if (!threadIdx.x) for (int c = 0; c < 3; ++c) partial[vcr_launch.block*3+c] = values[c][0];
}
struct component_sums_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { component_sums(index,args...); }
};
template<int D, class Index> __device__ void field_energy_gradient(KernelIndex vcr_launch, const float* field, const float* means, float* partial, Layout g) {
    __shared__ float values[10][256];
    for (int m = 0; m < 10; ++m) values[m][threadIdx.x] = 0;
    EACH_INDEX(g.size) {
        bool neighbor[D];
        #pragma unroll
        for (int a=0;a<D;++a)
            neighbor[a]=(Index(i)/Index(g.stride[a]))%Index(g.shape[a])+1<Index(g.shape[a]);
        #pragma unroll
        for (int c=0;c<D;++c) {
            // Round the centered values just as a stored float32 field would.
            // This fuses the centering read into statistics, without replacing
            // centered energy by the cancellation-prone E[x*x]-E[x]*E[x].
            const float mean=means[c], value=field[D*i+c]-mean;
            values[0][threadIdx.x] += value*value;
            #pragma unroll
            for (int a=0;a<D;++a) if (neighbor[a]) {
                const float next=field[D*(i+g.stride[a])+c]-mean;
                values[1+3*c+a][threadIdx.x]=fmaxf(values[1+3*c+a][threadIdx.x],fabsf(next-value));
            }
        }
    }
    block_reduce(values);
    if (!threadIdx.x) for (int m = 0; m < 10; ++m) partial[vcr_launch.block*10+m] = values[m][0];
}
template<int D, class Index> struct field_energy_gradient_launch2 {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { field_energy_gradient<D,Index>(index,args...); }
};
template<class Real> __device__ void scale_factor(KernelIndex vcr_launch, Real* stats, Layout g, double rms) {
    Real factor = stats[0] ? Real(rms)/sqrt(stats[0]/Real(g.size)) : 0, limit = 0;
    for (int c = 0; c < g.dim; ++c) limit = fmax(limit,stats[1+3*c]+stats[2+3*c]+stats[3+3*c]);
    if (limit) factor = fmin(factor,Real(.8)/limit);
    stats[0] = factor;
}
struct scale_factor_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { scale_factor(index,args...); }
};
template<int D> __device__ void field_center_scale(KernelIndex vcr_launch, float* field, const float* means, const float* factor, std::size_t n) {
    EACH_INDEX(n) {
        #pragma unroll
        for (int c=0;c<D;++c) field[D*i+c]=(field[D*i+c]-means[c])*factor[0];
    }
}
template<int D> struct field_center_scale_launch1 {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { field_center_scale<D>(index,args...); }
};
Parameters sample_elastic(const Elastic& t, const Sample& s, Random& r, bool record) {
    using Real = float;
    auto p = geometry(sample_shape(s)); const auto [sigma,magnitude] = elastic_settings(t,r);
    p.border = t.border;
    if (!magnitude) { p.exact = true; return p; }
    const double spacing = std::max(1.,2*sigma); const auto dim = p.shape.size();
    Shape knots; for (auto n : p.shape) knots.push_back(elastic_detail::controls(n,spacing,true));
    std::vector<float> controls;
    for (std::size_t c = 0; c < dim; ++c) {
        auto values = elastic_detail::draw_controls(knots,r); controls.insert(controls.end(),values.begin(),values.end());
    }
    auto data = std::make_shared<Buffer<Real>>(controls); Shape shape = knots;
    for (int axis = int(dim)-1; axis >= 0; --axis) {
        auto target = shape; target[axis] = p.shape[axis];
        std::vector<float> weights; std::vector<std::size_t> positions;
        for (std::size_t x = 0; x < target[axis]; ++x) {
            double q = x/spacing; positions.push_back(std::size_t(q));
            auto w = elastic_detail::weights(q-std::floor(q),true); weights.insert(weights.end(),w.begin(),w.end());
        }
        Buffer<Real> w(weights); Buffer<std::size_t> pos(positions);
        auto next = std::make_shared<Buffer<Real>>(product(target)*dim);
        if (axis==0) {
            const auto launch_outer = [&]<class Index>() {
                if (dim==3)
                    launch(blocks(product(target)),256,0,vcr::device::stream(), expand_outer_launch2<3,Index>{}, data->ptr,next->ptr,product(shape),product(target),Grid(target).stride[0],w.ptr,pos.ptr);
                else
                    launch(blocks(product(target)),256,0,vcr::device::stream(), expand_outer_launch2<2,Index>{}, data->ptr,next->ptr,product(shape),product(target),Grid(target).stride[0],w.ptr,pos.ptr);
            };
            if (next->size<=UINT32_MAX && data->size<=UINT32_MAX) launch_outer.template operator()<std::uint32_t>();
            else launch_outer.template operator()<std::size_t>();
        } else if (next->size<=UINT32_MAX && data->size<=UINT32_MAX)
            launch(blocks(next->size),256,0,vcr::device::stream(), expand_controls_launch1<std::uint32_t>{}, data->ptr,next->ptr,Layout{Grid(shape)},Layout{Grid(target)},w.ptr,pos.ptr,axis);
        else
            launch(blocks(next->size),256,0,vcr::device::stream(), expand_controls_launch1<std::size_t>{}, data->ptr,next->ptr,Layout{Grid(shape)},Layout{Grid(target)},w.ptr,pos.ptr,axis);
        check(cudaGetLastError()); data = std::move(next); shape = std::move(target);
    }
    // Bound partials: process 1024 rather than 8192 records in the single-block
    // final reduction for a 128^3 crop, accumulating more voxels per thread.
    Layout g{Grid(p.shape)}; const auto count = std::min(blocks(g.size),1024u);
    Buffer<Real> partial(count*10),stats(10),means(3);
    launch(count,256,0,vcr::device::stream(), component_sums_launch{}, data->ptr,partial.ptr,g);
    // Divide each component sum once rather than once per voxel.
    launch(1,256,0,vcr::device::stream(), finish_reduce_launch2<3,true>{}, partial.ptr,means.ptr,count,g.size);
    const auto collect = [&]<class Index>() {
        if (dim==3) launch(count,256,0,vcr::device::stream(), field_energy_gradient_launch2<3,Index>{}, data->ptr,means.ptr,partial.ptr,g);
        else launch(count,256,0,vcr::device::stream(), field_energy_gradient_launch2<2,Index>{}, data->ptr,means.ptr,partial.ptr,g);
    };
    if (g.size<=UINT32_MAX) collect.template operator()<std::uint32_t>();
    else collect.template operator()<std::size_t>();
    launch(1,256,0,vcr::device::stream(), finish_reduce_launch1<10>{}, partial.ptr,stats.ptr,count);
    // Normalization is invariant across the field. Compute the square root and
    // gradient cap once, rather than in every voxel/component thread.
    launch(1,1,0,vcr::device::stream(), scale_factor_launch{}, stats.ptr,g,magnitude);
    if (dim==3) launch(blocks(g.size),256,0,vcr::device::stream(), field_center_scale_launch1<3>{}, data->ptr,means.ptr,stats.ptr,g.size);
    else launch(blocks(g.size),256,0,vcr::device::stream(), field_center_scale_launch1<2>{}, data->ptr,means.ptr,stats.ptr,g.size);
    check(cudaGetLastError());
    p.device_field = data;
    if (record) { p.field.resize(data->size); data->download(p.field.data()); }
    return p;
}
void apply_cuda(const Elastic&, Sample& s, const Parameters& p, Session& session) { geometry_cuda(s,p,session); }
} // namespace vcr::transforms::cuda
