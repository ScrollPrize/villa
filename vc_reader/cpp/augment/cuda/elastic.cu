#include "geometry.cuh"
#include "local_field.cuh"
#include "reductions.cuh"
namespace vcr::augment::cuda {
template<class Index, class Real> __global__ void expand_controls(const Real* src, Real* dst, Layout input, Layout output,
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
template<class Real> __global__ void component_sums(const Real* field, Real* partial, Layout g) {
    __shared__ Real values[3][256];
    for (int c = 0; c < 3; ++c) values[c][threadIdx.x] = 0;
    EACH_INDEX(g.size) for (int c = 0; c < g.dim; ++c) values[c][threadIdx.x] += field[i*g.dim+c];
    block_reduce(values);
    if (!threadIdx.x) for (int c = 0; c < 3; ++c) partial[blockIdx.x*3+c] = values[c][0];
}
template<class Real> __global__ void center_field(Real* field, const Real* means, Layout g) {
    EACH_INDEX(g.size*g.dim) field[i] -= means[i%g.dim];
}
template<class Real> __global__ void field_energy_gradient(const Real* field, Real* partial, Layout g) {
    __shared__ Real values[10][256];
    for (int m = 0; m < 10; ++m) values[m][threadIdx.x] = 0;
    EACH_INDEX(g.size) for (int c = 0; c < g.dim; ++c) {
        const Real value = field[g.dim*i+c]; values[0][threadIdx.x] += value*value;
        for (int a = 0; a < g.dim; ++a) if (g.coord(i,a)+1 < g.shape[a])
            values[1+3*c+a][threadIdx.x] = fmax(values[1+3*c+a][threadIdx.x],fabs(field[g.dim*(i+g.stride[a])+c]-value));
    }
    block_reduce(values);
    if (!threadIdx.x) for (int m = 0; m < 10; ++m) partial[blockIdx.x*10+m] = values[m][0];
}
template<class Real> __global__ void scale_factor(Real* stats, Layout g, double rms) {
    Real factor = stats[0] ? Real(rms)/sqrt(stats[0]/Real(g.size)) : 0, limit = 0;
    for (int c = 0; c < g.dim; ++c) limit = fmax(limit,stats[1+3*c]+stats[2+3*c]+stats[3+3*c]);
    if (limit) factor = fmin(factor,Real(.8)/limit);
    stats[0] = factor;
}
__global__ void field_scale(float* field, const float* factor, std::size_t n) {
    EACH_INDEX(n) field[i] *= factor[0];
}
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
        if (next->size<=UINT32_MAX && data->size<=UINT32_MAX)
            expand_controls<std::uint32_t><<<blocks(next->size),256,0,vcr::device::stream()>>>(data->ptr,next->ptr,Layout{Grid(shape)},Layout{Grid(target)},w.ptr,pos.ptr,axis);
        else
            expand_controls<std::size_t><<<blocks(next->size),256,0,vcr::device::stream()>>>(data->ptr,next->ptr,Layout{Grid(shape)},Layout{Grid(target)},w.ptr,pos.ptr,axis);
        check(cudaGetLastError()); data = std::move(next); shape = std::move(target);
    }
    Layout g{Grid(p.shape)}; const auto count = blocks(g.size);
    Buffer<Real> partial(count*10),stats(10);
    component_sums<<<count,256,0,vcr::device::stream()>>>(data->ptr,partial.ptr,g);
    // Divide each component sum once rather than once per voxel.
    finish_reduce<3,true><<<1,256,0,vcr::device::stream()>>>(partial.ptr,stats.ptr,count,g.size);
    center_field<<<blocks(data->size),256,0,vcr::device::stream()>>>(data->ptr,stats.ptr,g);
    field_energy_gradient<<<count,256,0,vcr::device::stream()>>>(data->ptr,partial.ptr,g);
    finish_reduce<10><<<1,256,0,vcr::device::stream()>>>(partial.ptr,stats.ptr,count);
    // Normalization is invariant across the field. Compute the square root and
    // gradient cap once, rather than in every voxel/component thread.
    scale_factor<<<1,1,0,vcr::device::stream()>>>(stats.ptr,g,magnitude);
    field_scale<<<blocks(data->size),256,0,vcr::device::stream()>>>(data->ptr,stats.ptr,data->size);
    check(cudaGetLastError());
    p.device_field = data;
    if (record) { p.field.resize(data->size); data->download(p.field.data()); }
    return p;
}
void apply_cuda(const Elastic&, Sample& s, const Parameters& p, Session& session) { geometry_cuda(s,p,session); }
} // namespace vcr::augment::cuda
