#include "resident.cuh"
#include "../structure.hpp"
namespace vcr::augment::cuda {
namespace {
using namespace structure_detail;
// Correlation along one axis with shared taps/boundary rules; accumulates in T.
template<class T> __global__ void correlate_kernel(const T* in, T* out, Layout g, int axis, const double* taps, int radius, Boundary edge) {
    EACH_INDEX(g.size) {
        const auto x = (long long)g.coord(i,axis); const auto stride = g.stride[axis];
        const auto base = i - std::size_t(x)*stride;
        T sum = 0;
        for (int k = -radius; k <= radius; ++k) {
            const auto j = neighbour(x+k,(long long)g.shape[axis],edge);
            if (j >= 0) sum += T(taps[k+radius])*in[base+std::size_t(j)*stride];
        }
        out[i] = sum;
    }
}
// In-place separable filter through one scratch buffer of the same size.
template<class T> void separable_cuda(T*& data, T*& scratch, const Grid& grid, const std::vector<Taps>& taps, Boundary edge) {
    Layout g(grid);
    for (std::size_t a = 0; a < taps.size(); ++a) {
        if (taps[a].radius == 0 && taps[a].weights[0] == 1) continue;
        Buffer<double> weights(taps[a].weights);
        correlate_kernel<<<blocks(g.size),256>>>(data,scratch,g,int(a),weights.ptr,taps[a].radius,edge);
        check(cudaGetLastError()); std::swap(data,scratch);
    }
}
template<class T, class S> __global__ void scale_copy(const S* in, T* out, std::size_t n, double scale) {
    EACH_INDEX(n) out[i] = T(double(in[i])*scale);
}
template<class S> __global__ void edt_features(const S* labels, double* f, std::size_t n, double threshold, bool inside, int* any) {
    EACH_INDEX(n) {
        const bool feature = (double(labels[i]) > threshold) != inside;
        f[i] = feature ? 0. : 1e300;
        if (feature) *any = 1;
    }
}
// One thread per line along `axis`; lines start where that coordinate is zero.
__global__ void edt_lines(double* f, Layout g, int axis, int* v, double* z, double* scratch, std::size_t lines) {
    const std::size_t n = g.shape[axis], stride = g.stride[axis];
    EACH_INDEX(lines) {
        // Enumerate line starts: all indices with coord(axis) == 0.
        const std::size_t outer = i/stride, inner = i%stride, start = outer*stride*n + inner;
        envelope(f+start,n,stride,v+i*n,z+i*(n+1),scratch+i*n);
    }
}
__global__ void edt_finish(const double* inside, const double* outside, float* out, std::size_t n, int mode, const int* any) {
    EACH_INDEX(n) {
        const double in = any[0] ? sqrt(inside[i]) : 0., out_ = any[1] ? sqrt(outside[i]) : 0.;
        out[i] = float(mode == 0 ? out_-in : mode == 1 ? in : out_);
    }
}
template<class T> __global__ void normalize_normals(T* const* gradient, T* out, std::size_t n, int d, int c0, int c1, int c2) {
    const int order[3] = {c0,c1,c2};
    EACH_INDEX(n) {
        double norm = 0; for (int a = 0; a < d; ++a) norm += double(gradient[a][i])*double(gradient[a][i]);
        norm = fmax(sqrt(norm),1e-6);
        for (int j = 0; j < d; ++j) out[j*n+i] = T(double(gradient[order[j]][i])/norm);
    }
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
Array metadata(const Shape& shape, std::size_t bytes) { Array a; a.shape = shape; a.kind = 'f'; a.bytes = bytes; return a; }
} // namespace

void apply_cuda(const DistanceTransform& t, Sample& s, const Parameters&, Session& session) {
    std::vector<std::pair<std::string,Field>> outputs;
    for (const auto& [name,f] : s) if (selected_label(name,f,t.on)) {
        const auto& a = session.arrays.at(name); Grid grid(a.spatial()); Layout g(grid); const int d = int(grid.shape.size());
        require(t.channel < a.shape[0],"DistanceTransform channel out of bounds");
        const std::size_t n = grid.size, extent = *std::max_element(grid.shape.begin(),grid.shape.end());
        Buffer<double> inside(n), outside(n); Buffer<int> any(2); check(cudaMemsetAsync(any.ptr,0,2*sizeof(int)));
        const std::size_t lines = n/std::min_element(grid.shape.begin(),grid.shape.end())[0];
        Buffer<int> v(lines*extent); Buffer<double> z(lines*(extent+1)), scratch(lines*extent);
        for (int pass = 0; pass < 2; ++pass) {
            const bool in = pass == 0; auto& field = in ? inside : outside;
            if (in && t.mode == DistanceMode::Outside) continue;
            if (!in && t.mode == DistanceMode::Inside) continue;
            a.typed([&](auto labels) { edt_features<<<blocks(n),256>>>(labels.ptr+t.channel*n,field.ptr,n,t.threshold,in,any.ptr+(in ? 0 : 1)); });
            for (int axis_ = 0; axis_ < d; ++axis_) {
                const std::size_t count = n/grid.shape[axis_];
                edt_lines<<<blocks(count),256>>>(field.ptr,g,axis_,v.ptr,z.ptr,scratch.ptr,count);
            }
            check(cudaGetLastError());
        }
        Shape shape{1}; shape.insert(shape.end(),grid.shape.begin(),grid.shape.end());
        DeviceArray result(shape,'f',4);
        edt_finish<<<blocks(n),256>>>(inside.ptr,outside.ptr,result.view<float>().ptr,n,int(t.mode),any.ptr);
        check(cudaGetLastError());
        if (!t.normals.empty()) {
            std::vector<Buffer<float>> gradient; std::vector<float*> pointers; Buffer<float> tmp(n);
            for (int axis_ = 0; axis_ < d; ++axis_) {
                gradient.emplace_back(n);
                check(cudaMemcpyAsync(gradient.back().ptr,result.view<float>().ptr,n*sizeof(float),cudaMemcpyDeviceToDevice));
                std::vector<Taps> taps(d,scharr_smooth()); taps[axis_] = scharr_derivative();
                float* data = gradient.back().ptr; float* scratch_ptr = tmp.ptr;
                separable_cuda(data,scratch_ptr,grid,taps,Boundary::Reflect);
                if (data != gradient.back().ptr) check(cudaMemcpyAsync(gradient.back().ptr,data,n*sizeof(float),cudaMemcpyDeviceToDevice));
            }
            for (auto& b : gradient) pointers.push_back(b.ptr);
            Buffer<float*> table(pointers); int order[3]{};
            for (int j = 0; j < d; ++j) order[j] = axis(t.components[j],d);
            Shape vshape{std::size_t(d)}; vshape.insert(vshape.end(),grid.shape.begin(),grid.shape.end());
            DeviceArray normals(vshape,'f',4);
            normalize_normals<<<blocks(n),256>>>(table.ptr,normals.view<float>().ptr,n,d,order[0],order[1],order[2]);
            check(cudaGetLastError());
            outputs.emplace_back(name+t.normals,Field{metadata(vshape,4),derived(Role::NormalField,t.components),name});
            session.arrays.emplace(name+t.normals,std::move(normals));
        }
        outputs.emplace_back(name+t.suffix,Field{metadata(shape,4),derived(Role::Scalar),name});
        session.arrays.emplace(name+t.suffix,std::move(result));
    }
    for (auto& [name,f] : outputs) { require(!s.count(name),"derived field already exists: "+name); s.emplace(name,std::move(f)); }
}

void apply_cuda(const StructureTensor& t, Sample& s, const Parameters&, Session& session) {
    const auto& a = session.arrays.at(t.on); Grid grid(a.spatial()); const int d = int(grid.shape.size());
    require(t.channel < a.shape[0],"StructureTensor channel out of bounds");
    const auto name = t.output.empty() ? t.on+"_st" : t.output;
    std::vector<std::pair<std::string,Field>> outputs;
    const auto run = [&](auto tag) {
        using T = decltype(tag);
        const std::size_t n = grid.size; const int count = component_count(d);
        Buffer<T> base(n), tmp(n);
        a.typed([&](auto in) { scale_copy<<<blocks(n),256>>>(in.ptr+t.channel*n,base.ptr,n,t.scale); });
        T* data = base.ptr; T* scratch = tmp.ptr;
        if (t.derivative == Derivative::Pavel && t.sigma > 0) separable_cuda(data,scratch,grid,std::vector<Taps>(d,torch_gaussian(t.sigma)),t.boundary);
        const auto taps = gradient_taps(t,d);
        std::vector<Buffer<T>> gradient; std::vector<T*> pointers;
        for (int axis_ = 0; axis_ < d; ++axis_) {
            gradient.emplace_back(n);
            check(cudaMemcpyAsync(gradient.back().ptr,data,n*sizeof(T),cudaMemcpyDeviceToDevice));
            Buffer<T> local(n); T* g = gradient.back().ptr; T* sc = local.ptr;
            separable_cuda(g,sc,grid,taps[axis_],t.boundary);
            if (g != gradient.back().ptr) check(cudaMemcpyAsync(gradient.back().ptr,g,n*sizeof(T),cudaMemcpyDeviceToDevice));
            pointers.push_back(gradient.back().ptr);
        }
        Buffer<T*> table(pointers);
        Buffer<T> components(count*n);
        products<<<blocks(n),256>>>(table.ptr,components.ptr,n,d); check(cudaGetLastError());
        if (!t.at.empty()) {
            const unsigned parts = std::min(blocks(n),1024u); Buffer<double> partial(parts*7);
            const double cz = d == 3 ? t.at[0] : 0, cy = t.at[d-2], cx = t.at[d-1];
            if (d == 3) point_sums<<<parts,256>>>(components.ptr,Layout(grid),count,cz,cy,cx,t.rho,partial.ptr);
            else point_sums<<<parts,256>>>(components.ptr,Layout(grid),count,cy,cx,0.,t.rho,partial.ptr);
            Shape shape{std::size_t(d),std::size_t(d)}; DeviceArray tensor(shape,'f',sizeof(T));
            point_tensor<<<1,1>>>(partial.ptr,parts,d,tensor.view<T>().ptr); check(cudaGetLastError());
            outputs.emplace_back(name,Field{metadata(shape,sizeof(T)),derived(Role::Vectors),t.on});
            session.arrays.emplace(name,std::move(tensor));
            return;
        }
        const auto integration = integration_taps(t,d);
        Buffer<T> component_scratch(n);
        for (int c = 0; c < count; ++c) {
            T* cd = components.ptr+c*n; T* cs = component_scratch.ptr;
            separable_cuda(cd,cs,grid,integration,t.boundary);
            if (cd != components.ptr+c*n) check(cudaMemcpyAsync(components.ptr+c*n,cd,n*sizeof(T),cudaMemcpyDeviceToDevice));
        }
        Shape shape{std::size_t(t.field == StructureField::Components ? count : d)}; shape.insert(shape.end(),grid.shape.begin(),grid.shape.end());
        DeviceArray out(shape,'f',sizeof(T));
        if (t.field == StructureField::Components) {
            check(cudaMemcpyAsync(out.buffer->ptr,components.ptr,count*n*sizeof(T),cudaMemcpyDeviceToDevice));
        } else {
            int order[3]{}; for (int j = 0; j < d; ++j) order[j] = axis(t.components[j],d);
            std::unique_ptr<DeviceArray> gap;
            Shape gshape{1}; gshape.insert(gshape.end(),grid.shape.begin(),grid.shape.end());
            if (!t.confidence.empty()) gap = std::make_unique<DeviceArray>(gshape,'f',sizeof(T));
            eigen_fields<<<blocks(n),256>>>(components.ptr,out.view<T>().ptr,gap ? gap->view<T>().ptr : nullptr,n,d,
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
