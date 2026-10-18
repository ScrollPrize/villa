#include "separable_filter.cuh"
#include "../structure_tensor.hpp"
namespace vcr::augment::cuda {
namespace {
using namespace structure_tensor_detail;
using filter_detail::separable_cuda;
template<class T, class S> __global__ void scale_copy(const S* in, T* out, std::size_t n, double scale) {
    EACH_INDEX(n) out[i] = T(double(in[i])*scale);
}
template<class T> __global__ void products(T* const* gradient, T* components, std::size_t n, int d) {
    EACH_INDEX(n) for (int c = 0; c < component_count(d); ++c) {
        int a, b; component_axes(c,d,a,b); components[c*n+i] = gradient[a][i]*gradient[b][i];
    }
}
template<class T> __global__ void eigen_fields(const T* components, T* out, T* gap, std::size_t n, int d, bool normal, int c0, int c1, int c2) {
    const int order[3] = {c0,c1,c2};
    EACH_INDEX(n) {
        double tensor[6], vector[3];
        for (int c = 0; c < component_count(d); ++c) tensor[c] = double(components[c*n+i]);
        const double confidence = principal(tensor,d,normal,vector);
        for (int j = 0; j < d; ++j) out[j*n+i] = T(vector[order[j]]);
        if (gap) gap[i] = T(confidence);
    }
}
// Gaussian point weights (normalized later) times each component; block partial sums.
template<class T> __global__ void point_sums(const T* components, Layout g, int count, double cz, double cy, double cx, double rho, double* partial) {
    __shared__ double values[7][256];
    double local[7]{};
    const double center[3] = {cz,cy,cx};
    EACH_INDEX(g.size) {
        double w = 1;
        for (int a = 0; a < g.dim; ++a) { const double u = (double(g.coord(i,a))-center[a])/rho; w *= exp(-.5*u*u); }
        local[0] += w;
        for (int c = 0; c < count; ++c) local[1+c] += w*double(components[c*g.size+i]);
    }
    for (int m = 0; m < 7; ++m) values[m][threadIdx.x] = local[m];
    __syncthreads();
    for (int offset = 128; offset; offset /= 2) {
        if (threadIdx.x < offset) for (int m = 0; m < 7; ++m) values[m][threadIdx.x] += values[m][threadIdx.x+offset];
        __syncthreads();
    }
    if (!threadIdx.x) for (int m = 0; m < 7; ++m) partial[blockIdx.x*7+m] = values[m][0];
}
template<class T> __global__ void point_tensor(const double* partial, unsigned blocks_, int d, T* out) {
    double sums[7]{};
    for (unsigned b = 0; b < blocks_; ++b) for (int m = 0; m < 7; ++m) sums[m] += partial[b*7+m];
    for (int c = 0; c < component_count(d); ++c) {
        int a, b; component_axes(c,d,a,b);
        out[a*d+b] = out[b*d+a] = T(sums[1+c]/sums[0]);
    }
}
} // namespace

void apply_cuda(const StructureTensor& t, Sample& s, const Parameters&, Session& session) {
    const auto& a = session.arrays.at(t.on); Grid grid(a.spatial()); const int d = int(grid.shape.size());
    require(t.channel < a.shape[0],"StructureTensor channel out of bounds");
    const auto name = t.output.empty() ? t.on+"_st" : t.output;
    std::vector<std::pair<std::string,Field>> outputs;
    const auto run = [&](auto tag) {
        using T = decltype(tag);
        const std::size_t n = grid.size; const int count = component_count(d);
        Buffer<T> base(n), tmp(n);
        a.typed([&](auto in) { scale_copy<<<blocks(n),256,0,vcr::device::stream()>>>(in.ptr+t.channel*n,base.ptr,n,t.scale); });
        T* data = base.ptr; T* scratch = tmp.ptr;
        if (t.derivative == Derivative::Pavel && t.sigma > 0) separable_cuda(data,scratch,grid,std::vector<Taps>(d,torch_gaussian(t.sigma)),t.boundary);
        const auto taps = gradient_taps(t,d);
        std::vector<Buffer<T>> gradient; std::vector<T*> pointers;
        for (int axis_ = 0; axis_ < d; ++axis_) {
            gradient.emplace_back(n);
            check(cudaMemcpyAsync(gradient.back().ptr,data,n*sizeof(T),cudaMemcpyDeviceToDevice,vcr::device::stream()));
            Buffer<T> local(n); T* g = gradient.back().ptr; T* sc = local.ptr;
            separable_cuda(g,sc,grid,taps[axis_],t.boundary);
            if (g != gradient.back().ptr) check(cudaMemcpyAsync(gradient.back().ptr,g,n*sizeof(T),cudaMemcpyDeviceToDevice,vcr::device::stream()));
            pointers.push_back(gradient.back().ptr);
        }
        Buffer<T*> table(pointers);
        Buffer<T> components(count*n);
        products<<<blocks(n),256,0,vcr::device::stream()>>>(table.ptr,components.ptr,n,d); check(cudaGetLastError());
        if (!t.at.empty()) {
            const unsigned parts = std::min(blocks(n),1024u); Buffer<double> partial(parts*7);
            const double cz = d == 3 ? t.at[0] : 0, cy = t.at[d-2], cx = t.at[d-1];
            if (d == 3) point_sums<<<parts,256,0,vcr::device::stream()>>>(components.ptr,Layout(grid),count,cz,cy,cx,t.rho,partial.ptr);
            else point_sums<<<parts,256,0,vcr::device::stream()>>>(components.ptr,Layout(grid),count,cy,cx,0.,t.rho,partial.ptr);
            Shape shape{std::size_t(d),std::size_t(d)}; DeviceArray tensor(shape,'f',sizeof(T));
            point_tensor<<<1,1,0,vcr::device::stream()>>>(partial.ptr,parts,d,tensor.view<T>().ptr); check(cudaGetLastError());
            outputs.emplace_back(name,Field{metadata(shape,sizeof(T)),derived(Role::Vectors),t.on});
            session.arrays.emplace(name,std::move(tensor));
            return;
        }
        const auto integration = integration_taps(t,d);
        Buffer<T> component_scratch(n);
        for (int c = 0; c < count; ++c) {
            T* cd = components.ptr+c*n; T* cs = component_scratch.ptr;
            separable_cuda(cd,cs,grid,integration,t.boundary);
            if (cd != components.ptr+c*n) check(cudaMemcpyAsync(components.ptr+c*n,cd,n*sizeof(T),cudaMemcpyDeviceToDevice,vcr::device::stream()));
        }
        Shape shape{std::size_t(t.field == StructureField::Components ? count : d)}; shape.insert(shape.end(),grid.shape.begin(),grid.shape.end());
        DeviceArray out(shape,'f',sizeof(T));
        if (t.field == StructureField::Components) {
            check(cudaMemcpyAsync(out.buffer->ptr,components.ptr,count*n*sizeof(T),cudaMemcpyDeviceToDevice,vcr::device::stream()));
        } else {
            int order[3]{}; for (int j = 0; j < d; ++j) order[j] = axis(t.components[j],d);
            std::unique_ptr<DeviceArray> gap;
            Shape gshape{1}; gshape.insert(gshape.end(),grid.shape.begin(),grid.shape.end());
            if (!t.confidence.empty()) gap = std::make_unique<DeviceArray>(gshape,'f',sizeof(T));
            eigen_fields<<<blocks(n),256,0,vcr::device::stream()>>>(components.ptr,out.view<T>().ptr,gap ? gap->view<T>().ptr : nullptr,n,d,
                                            t.field == StructureField::Normal,order[0],order[1],order[2]);
            check(cudaGetLastError());
            if (gap) {
                outputs.emplace_back(t.confidence,Field{metadata(gshape,sizeof(T)),derived(Role::Scalar),t.on});
                session.arrays.emplace(t.confidence,std::move(*gap));
            }
        }
        outputs.emplace_back(name,Field{metadata(shape,sizeof(T)),derived(t.field == StructureField::Components ? Role::Scalar : Role::NormalField,t.components),t.on});
        session.arrays.emplace(name,std::move(out));
    };
    if (a.kind == 'f' && a.bytes == 4) run(float{}); else run(double{});
    for (auto& [key,field] : outputs) { require(!s.count(key),"derived field already exists: "+key); s.emplace(key,std::move(field)); }
}
} // namespace vcr::augment::cuda
