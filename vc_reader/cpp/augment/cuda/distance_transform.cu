#include "separable_filter.cuh"
#include "../distance_transform.hpp"
namespace vcr::augment::cuda {
namespace {
using namespace distance_detail;
using filter_detail::separable_cuda;
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
} // namespace

void apply_cuda(const DistanceTransform& t, Sample& s, const Parameters&, Session& session) {
    std::vector<std::pair<std::string,Field>> outputs;
    for (const auto& [name,f] : s) if (selected_label(name,f,t.on)) {
        const auto& a = session.arrays.at(name); Grid grid(a.spatial()); Layout g(grid); const int d = int(grid.shape.size());
        require(t.channel < a.shape[0],"DistanceTransform channel out of bounds");
        const std::size_t n = grid.size, extent = *std::max_element(grid.shape.begin(),grid.shape.end());
        Buffer<double> inside(n), outside(n); Buffer<int> any(2); check(cudaMemsetAsync(any.ptr,0,2*sizeof(int),vcr::device::stream()));
        const std::size_t lines = n/std::min_element(grid.shape.begin(),grid.shape.end())[0];
        Buffer<int> v(lines*extent); Buffer<double> z(lines*(extent+1)), scratch(lines*extent);
        for (int pass = 0; pass < 2; ++pass) {
            const bool in = pass == 0; auto& field = in ? inside : outside;
            if (in && t.mode == DistanceMode::Outside) continue;
            if (!in && t.mode == DistanceMode::Inside) continue;
            a.typed([&](auto labels) { edt_features<<<blocks(n),256,0,vcr::device::stream()>>>(labels.ptr+t.channel*n,field.ptr,n,t.threshold,in,any.ptr+(in ? 0 : 1)); });
            for (int axis_ = 0; axis_ < d; ++axis_) {
                const std::size_t count = n/grid.shape[axis_];
                edt_lines<<<blocks(count),256,0,vcr::device::stream()>>>(field.ptr,g,axis_,v.ptr,z.ptr,scratch.ptr,count);
            }
            check(cudaGetLastError());
        }
        Shape shape{1}; shape.insert(shape.end(),grid.shape.begin(),grid.shape.end());
        DeviceArray result(shape,'f',4);
        edt_finish<<<blocks(n),256,0,vcr::device::stream()>>>(inside.ptr,outside.ptr,result.view<float>().ptr,n,int(t.mode),any.ptr);
        check(cudaGetLastError());
        if (!t.normals.empty()) {
            std::vector<Buffer<float>> gradient; std::vector<float*> pointers; Buffer<float> tmp(n);
            for (int axis_ = 0; axis_ < d; ++axis_) {
                gradient.emplace_back(n);
                check(cudaMemcpyAsync(gradient.back().ptr,result.view<float>().ptr,n*sizeof(float),cudaMemcpyDeviceToDevice,vcr::device::stream()));
                std::vector<Taps> taps(d,scharr_smooth()); taps[axis_] = scharr_derivative();
                float* data = gradient.back().ptr; float* scratch_ptr = tmp.ptr;
                separable_cuda(data,scratch_ptr,grid,taps,Boundary::Reflect);
                if (data != gradient.back().ptr) check(cudaMemcpyAsync(gradient.back().ptr,data,n*sizeof(float),cudaMemcpyDeviceToDevice,vcr::device::stream()));
            }
            for (auto& b : gradient) pointers.push_back(b.ptr);
            Buffer<float*> table(pointers); int order[3]{};
            for (int j = 0; j < d; ++j) order[j] = axis(t.components[j],d);
            Shape vshape{std::size_t(d)}; vshape.insert(vshape.end(),grid.shape.begin(),grid.shape.end());
            DeviceArray normals(vshape,'f',4);
            normalize_normals<<<blocks(n),256,0,vcr::device::stream()>>>(table.ptr,normals.view<float>().ptr,n,d,order[0],order[1],order[2]);
            check(cudaGetLastError());
            outputs.emplace_back(name+t.normals,Field{metadata(vshape,4),derived(Role::NormalField,t.components),name});
            session.arrays.emplace(name+t.normals,std::move(normals));
        }
        outputs.emplace_back(name+t.suffix,Field{metadata(shape,4),derived(Role::Scalar),name});
        session.arrays.emplace(name+t.suffix,std::move(result));
    }
    for (auto& [name,f] : outputs) { require(!s.count(name),"derived field already exists: "+name); s.emplace(name,std::move(f)); }
}

} // namespace vcr::augment::cuda
