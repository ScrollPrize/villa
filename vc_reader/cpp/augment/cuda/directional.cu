#include "geometry.cuh"
#include <cub/device/device_scan.cuh>
// Device counterparts of spatial.hpp's backward/jacobian/forward mapping for
// directional fields and sparse targets. Failures set Session::geometry_status.
namespace vcr::augment::cuda {
namespace {
struct Transform {
    double matrix[9]{}, inverse[9]{}, offset[3]{};
    Layout grid;
    const double* field = nullptr;
    int component[3]{};  // spatial axis of each stored component
    explicit Transform(const Parameters& p, const Target& t, const double* f) : grid(Grid(p.shape)), field(f) {
        const auto d = p.shape.size(); const auto inv = vcr::augment::inverse(p.matrix,d);
        std::copy(p.matrix.begin(),p.matrix.end(),matrix); std::copy(inv.begin(),inv.end(),inverse);
        std::copy(p.offset.begin(),p.offset.end(),offset);
        for (std::size_t j = 0; j < d; ++j) component[j] = axis(t.components[j],d);
    }
};
// Linear interpolation of the interleaved displacement field, clamped to the grid.
__device__ void displacement(const Transform& t, const double* out, double* q) {
    const auto& g = t.grid; long long lo[3]{}; double fraction[3]{};
    for (int a = 0; a < g.dim; ++a) {
        const double x = clamp_value(out[a],0,double(g.shape[a]-1));
        lo[a] = static_cast<long long>(floor(x)); fraction[a] = x-lo[a];
    }
    for (int corner = 0; corner < (1<<g.dim); ++corner) {
        double weight = 1; std::size_t at = 0; bool valid = true;
        for (int a = 0; a < g.dim; ++a) {
            const int bit = (corner>>a)&1; const auto x = lo[a]+bit;
            weight *= bit ? fraction[a] : 1-fraction[a];
            if (x >= static_cast<long long>(g.shape[a])) valid = false; else at += x*g.stride[a];
        }
        if (weight && valid) for (int a = 0; a < g.dim; ++a) q[a] += weight*t.field[at*g.dim+a];
    }
}
__device__ void backward(const Transform& t, const double* out, double* q) {
    const int d = t.grid.dim;
    for (int a = 0; a < d; ++a) {
        q[a] = t.offset[a];
        for (int b = 0; b < d; ++b) q[a] += t.matrix[a*d+b]*out[b];
    }
    if (t.field) {
        double extra[3]{}; displacement(t,out,extra);
        for (int a = 0; a < d; ++a) q[a] += extra[a];
    }
}
__device__ void jacobian(const Transform& t, const double* out, double* j) {
    const int d = t.grid.dim;
    for (int i = 0; i < d*d; ++i) j[i] = t.matrix[i];
    if (!t.field) return;
    for (int a = 0; a < d; ++a) {
        double lo[3]{out[0],out[1],out[2]}, hi[3]{out[0],out[1],out[2]}, v0[3], v1[3];
        lo[a] -= .5; hi[a] += .5; backward(t,lo,v0); backward(t,hi,v1);
        for (int b = 0; b < d; ++b) j[b*d+a] = v1[b]-v0[b];
    }
}
// Gauss-Jordan with partial pivoting, as the CPU inverse; false when singular.
__device__ bool invert(double* a, double* out, int d) {
    for (int i = 0; i < d*d; ++i) out[i] = i%(d+1) == 0;
    for (int j = 0; j < d; ++j) {
        int pivot = j; for (int i = j+1; i < d; ++i) if (fabs(a[i*d+j]) > fabs(a[pivot*d+j])) pivot = i;
        if (!isfinite(a[pivot*d+j]) || fabs(a[pivot*d+j]) <= 1e-12) return false;
        for (int k = 0; k < d; ++k) {
            const double x = a[j*d+k]; a[j*d+k] = a[pivot*d+k]; a[pivot*d+k] = x;
            const double y = out[j*d+k]; out[j*d+k] = out[pivot*d+k]; out[pivot*d+k] = y;
        }
        const double scale = a[j*d+j];
        for (int k = 0; k < d; ++k) { a[j*d+k] /= scale; out[j*d+k] /= scale; }
        for (int i = 0; i < d; ++i) if (i != j) {
            const double factor = a[i*d+j];
            for (int k = 0; k < d; ++k) { a[i*d+k] -= factor*a[j*d+k]; out[i*d+k] -= factor*out[j*d+k]; }
        }
    }
    return true;
}
__device__ void apply_matrix(const double* m, double* v, int d) {
    double out[3]{};
    for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j) out[i] += m[i*d+j]*v[j];
    for (int i = 0; i < d; ++i) v[i] = out[i];
}
__device__ void unit(double* v, int d) {
    double norm = 0; for (int i = 0; i < d; ++i) norm += v[i]*v[i];
    if (norm > 0) for (int i = 0; i < d; ++i) v[i] /= sqrt(norm);
}
// Output position of an input-frame point: Newton iteration on backward().
__device__ bool forward(const Transform& t, double* point) {
    const int d = t.grid.dim; double source[3]{point[0],point[1],point[2]};
    for (int a = 0; a < d; ++a) point[a] -= t.offset[a];
    apply_matrix(t.inverse,point,d);
    if (!t.field) return true;
    for (int iteration = 0; iteration < 50; ++iteration) {
        double q[3]{}, j[9], inv[9]; backward(t,point,q); double error = 0;
        for (int a = 0; a < d; ++a) { q[a] -= source[a]; error = fmax(error,fabs(q[a])); }
        if (error < 1e-7) return true;
        jacobian(t,point,j);
        if (!invert(j,inv,d)) return false;
        apply_matrix(inv,q,d);
        for (int a = 0; a < d; ++a) point[a] -= q[a];
    }
    return false;
}
enum : int { Unconverged = 1, Outside = 2, Singular = 4 };
template<class T> __global__ void dense_kernel(T* data, Transform t, Role role, bool normalize, int* status) {
    const auto& g = t.grid; const int d = g.dim;
    EACH_INDEX(g.size) {
        double v[3]{}, pos[3]{};
        for (int j = 0; j < d; ++j) { v[t.component[j]] = double(data[j*g.size+i]); pos[j] = double(g.coord(i,j)); }
        if (role == Role::CoordinateField) { if (!forward(t,v)) atomicOr(status,Unconverged); }
        else {
            double jac[9]; jacobian(t,pos,jac);
            if (role == Role::NormalField) {
                double transpose[9]; for (int x = 0; x < d; ++x) for (int y = 0; y < d; ++y) transpose[x*d+y] = jac[y*d+x];
                apply_matrix(transpose,v,d);
            } else {
                double inv[9];
                if (!invert(jac,inv,d)) { atomicOr(status,Singular); continue; }
                apply_matrix(inv,v,d);
            }
            if (role == Role::NormalField || normalize) unit(v,d);
        }
        for (int j = 0; j < d; ++j) data[j*g.size+i] = T(v[t.component[j]]);
    }
}
template<class T> __global__ void sparse_kernel(T* data, std::size_t count, Transform t, Role role, bool normalize,
                                                int outside, unsigned* keep, int* status) {
    const auto& g = t.grid; const int d = g.dim;
    EACH_INDEX(count) {
        double v[3]{};
        for (int j = 0; j < d; ++j) v[t.component[j]] = double(data[i*d+j]);
        if (role == Role::Points) { if (!forward(t,v)) atomicOr(status,Unconverged); }
        else apply_matrix(t.inverse,v,d);
        bool inside = true; for (int j = 0; j < d; ++j) inside &= v[j] >= 0 && v[j] <= double(g.shape[j]-1);
        if (role == Role::Points && !inside && outside == 1) atomicOr(status,Outside);
        if (role == Role::Vectors && normalize) unit(v,d);
        if (keep) keep[i] = role != Role::Points || inside;
        for (int j = 0; j < d; ++j) data[i*d+j] = T(v[t.component[j]]);
    }
}
template<class T> __global__ void compact_rows(const T* data, T* out, const unsigned* keep, const unsigned* index, std::size_t count, int d) {
    EACH_INDEX(count) if (keep[i]) for (int j = 0; j < d; ++j) out[std::size_t(index[i])*d+j] = data[i*d+j];
}
} // namespace

void directional_cuda(DeviceArray& a, const Target& target, const Parameters& p, const double* field, Session& session) {
    Transform t(p,target,field);
    a.typed([&](auto view) {
        using T = typename decltype(view)::value_type;
        if constexpr (std::is_floating_point_v<T>)
            dense_kernel<<<blocks(t.grid.size),256,0,vcr::device::stream()>>>(view.ptr,t,target.role,target.normalize,session.geometry_status());
    });
    check(cudaGetLastError());
}
void sparse_cuda(DeviceArray& a, const Target& target, const Parameters& p, const double* field, Session& session) {
    require(target.role != Role::Vectors || !field, "nonlinear transforms of sparse vectors require locations; use a vector field");
    Transform t(p,target,field);
    const auto count = a.shape[0], d = a.shape[1];
    const bool drop = target.role == Role::Points && target.outside == "drop";
    const int outside = target.outside == "error";
    std::unique_ptr<Buffer<unsigned>> keep; if (drop && count) keep = std::make_unique<Buffer<unsigned>>(count);
    a.typed([&](auto view) {
        using T = typename decltype(view)::value_type;
        if constexpr (std::is_floating_point_v<T>) {
            if (count) sparse_kernel<<<blocks(count),256,0,vcr::device::stream()>>>(view.ptr,count,t,target.role,target.normalize,outside,
                                                            keep ? keep->ptr : nullptr,session.geometry_status());
            check(cudaGetLastError());
            if (!keep) return;
            // Dropping points changes the output shape: the kept count is the
            // one value read back, and only for outside="drop".
            Buffer<unsigned> index(count); std::size_t bytes = 0;
            check(cub::DeviceScan::ExclusiveSum(nullptr,bytes,keep->ptr,index.ptr,count,vcr::device::stream()));
            Buffer<std::byte> scratch(bytes);
            check(cub::DeviceScan::ExclusiveSum(scratch.ptr,bytes,keep->ptr,index.ptr,count,vcr::device::stream()));
            unsigned last[2]{};
            check(cudaMemcpyAsync(last,index.ptr+count-1,sizeof(unsigned),cudaMemcpyDeviceToHost,vcr::device::stream()));
            check(cudaMemcpyAsync(last+1,keep->ptr+count-1,sizeof(unsigned),cudaMemcpyDeviceToHost,vcr::device::stream()));
            check(cudaStreamSynchronize(vcr::device::stream()));
            const std::size_t kept = std::size_t(last[0])+last[1];
            DeviceArray out(Shape{kept,d},a.kind,a.bytes);
            compact_rows<<<blocks(count),256,0,vcr::device::stream()>>>(view.ptr,out.view<T>().ptr,keep->ptr,index.ptr,count,int(d));
            check(cudaGetLastError());
            a = std::move(out);
        }
    });
}
} // namespace vcr::augment::cuda
