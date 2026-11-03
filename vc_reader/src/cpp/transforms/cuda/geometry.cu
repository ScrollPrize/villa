#include "common.cuh"
#include "geometry.cuh"
#include <bit>
namespace vcr::transforms::cuda {
template<class Real> struct GeometryArgs {
    Real matrix[9]{}, offset[3]{};
    const long long* device_offset=nullptr;
    double fill = 0;
    bool border = false, nearest = false, validity = false, identity = false;
    int mode = 0;
};
// Exact same-shape axis reflections need only integer address arithmetic.
// Keep geometry_cuda's target routing, sparse/directional handling and validity
// restoration shared with the general resampler.
template<class T, class Index> __device__ void reflect_kernel(KernelIndex vcr_launch, const T* src, T* dst, Layout g, std::size_t count,
                                                            unsigned reflected, Index xor_mask, bool validity) {
    EACH_INDEX(count) {
        const Index index=Index(i);
        Index at=index ^ xor_mask;
        for (int a=0;a<g.dim;++a) if (reflected & (1u<<a)) {
            const Index stride=Index(g.stride[a]), extent=Index(g.shape[a]);
            const Index x=(index/stride)%extent;
            at -= x*stride;
            at += (extent-1-x)*stride;
        }
        // A reflection is an involution: only one thread touches each pair.
        // Keeping the working buffer avoids a volume allocation and the copy
        // back into the caller's contiguous batch output.
        if (src==dst && index>at) continue;
        const T value=src[at];
        if (src==dst && index<at) {
            const T other=src[index];
            dst[at]=validity ? T(other>=T(1)) : other;
        }
        // The general validity interpolation canonicalizes signed zero too.
        dst[i]=validity ? T(value>=T(1)) : value;
    }
}
template<class T, class Index> struct reflect_kernel_launch2 {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { reflect_kernel<T,Index>(index,args...); }
};
template<class T, class Index> struct reflect_pairs_launch {
    __device__ void operator()(KernelIndex vcr_launch, T* data, std::size_t count,
                               Index mask, Index pivot, bool validity) const {
        EACH_INDEX(count/2) {
            // Insert a zero at the highest reflected bit. This enumerates
            // exactly one member of every pair, with no idle half of the grid.
            const Index below=pivot-1;
            const Index first=(Index(i)&below) | ((Index(i)&~below)<<1);
            const Index second=first^mask;
            const T a=data[first], b=data[second];
            data[first]=validity ? T(b>=T(1)) : b;
            data[second]=validity ? T(a>=T(1)) : a;
        }
    }
};
static void reflect_cuda(const DeviceArray& src, DeviceArray& out, unsigned reflected, bool validity) {
    if (!reflected && !validity && src.buffer->ptr==out.buffer->ptr) return;
    Layout grid{Grid(src.spatial())};
    std::size_t xor_mask=0;
    // Power-of-two axes occupy disjoint bits of the flattened index. Reverse
    // those bits directly; other axes keep exact integer coordinate mapping.
    for (int a=0;a<grid.dim;++a) if (reflected & (1u<<a)) {
        const auto extent=grid.shape[a], stride=grid.stride[a];
        if ((extent & (extent-1))==0 && (stride & (stride-1))==0) {
            xor_mask |= (extent-1)*stride;
            reflected &= ~(1u<<a);
        }
    }
    src.typed([&](auto input) {
        using T=typename decltype(input)::value_type;
        const auto run=[&]<class Index>() {
            if (!reflected && xor_mask && src.buffer->ptr==out.buffer->ptr)
                launch(blocks(out.size()/2),256,0,vcr::device::stream(), reflect_pairs_launch<T,Index>{}, input.ptr,out.size(),Index(xor_mask),Index(std::bit_floor(xor_mask)),validity);
            else
                launch(blocks(out.size()),256,0,vcr::device::stream(), reflect_kernel_launch2<T,Index>{}, input.ptr,out.view<T>().ptr,grid,out.size(),reflected,Index(xor_mask),validity);
        };
        if (out.size()<=UINT32_MAX) run.template operator()<std::uint32_t>();
        else run.template operator()<std::size_t>();
    });
    check(cudaGetLastError());
}
static int reflection_axes(const Parameters& p, const Shape& shape, const float* field) {
    if (!p.exact || field || p.shape!=shape) return -1;
    unsigned reflected=0;
    for (std::size_t a=0;a<shape.size();++a) {
        const auto diagonal=p.matrix[a*shape.size()+a];
        if (diagonal!=1 && diagonal!=-1) return -1;
        if (p.offset[a]!=(diagonal==1 ? 0 : double(shape[a]-1))) return -1;
        for (std::size_t b=0;b<shape.size();++b) if (a!=b && p.matrix[a*shape.size()+b]!=0) return -1;
        if (diagonal==-1) reflected |= 1u<<a;
    }
    return int(reflected);
}
template<class Acc, class T, class Real> __device__ Acc linear_sample(const T* src,Layout in,const long long* lo,
                                                                    const Real* fraction,double fill,bool validity) {
    Acc value=0;
    for(int corner=0;corner<(1<<in.dim);++corner) {
        Acc weight=1;std::size_t at=0;bool valid=true;
        for(int a=0;a<in.dim;++a) {
            const int bit=(corner>>a)&1;const auto x=lo[a]+bit;
            weight*=bit?Acc(fraction[a]):Acc(1)-Acc(fraction[a]);
            if(x<0 || x>=static_cast<long long>(in.shape[a]))valid=false;else at+=x*in.stride[a];
        }
        if(weight) {
            const Acc neighbor=valid?Acc(src[at]):Acc(fill);
            if(validity){if(neighbor<Acc(1))return 0;}
            else value+=weight*neighbor;
        }
    }
    return validity?Acc(1):value;
}
// Factor the interior tensor-product stencil into seven lerps (three in 2D).
// Boundary/validity sampling still uses the shared corner evaluator. Avoid
// b-a: that difference can overflow for finite endpoints of opposite signs.
template<int D, class Real, class T, class Index>
__device__ Real interior_sample(const T* src, Index base, Layout in, const Real* f) {
    ResizePosition<Real,Index> q[D];
    for (int a=0; a<D; ++a) q[a]={0,Index(in.stride[a]),f[a]};
    return resize_sample<D,Real>(src+base,q);
}
template<class T, class Real, int D = 0, class Index = std::size_t>
__device__ void geometry_kernel(KernelIndex vcr_launch, const T* src, T* dst, Layout in, Layout out, std::size_t channels, GeometryArgs<Real> p, const float* field, T pad, int* used_pad) {
    EACH_INDEX(out.size*channels) {
        const Index voxel = Index(i)%Index(out.size), channel = Index(i)/Index(out.size);
        const int dim = D ? D : out.dim;
        Real q[3]{}, fraction[3]{}, coords[3]{}; long long lo[3]{};
        // Decompose the output index once, rather than once per matrix row.
        Index remainder=voxel;
        for (int a=0;a<dim;++a) {
            coords[a]=Real(remainder/Index(out.stride[a]));
            remainder%=Index(out.stride[a]);
        }
        bool far = false;
        for (int a = 0; a < dim; ++a) {
            if (p.identity) q[a]=coords[a];
            else for (int b = 0; b < dim; ++b) q[a] += p.matrix[a*dim+b]*coords[b];
            q[a] += p.device_offset?Real(p.device_offset[a]):p.offset[a];
            if (field) q[a] += Real(field[std::size_t(voxel)*dim+a]);
            if (p.border && !p.validity) q[a] = fmin(fmax(q[a],Real(0)),Real(in.shape[a]-1));
            if (!isfinite(q[a]) || q[a] < -1 || q[a] > Real(in.shape[a])) { far = true; continue; }
            lo[a] = static_cast<long long>(floor(q[a])); fraction[a] = q[a]-lo[a];
        }
        if (far) { dst[i] = p.validity ? T(p.fill>=1) : pad; if (used_pad) atomicExch(used_pad,1); continue; }
        if (p.nearest) {
            bool valid = true; std::size_t at = 0;
            for (int a = 0; a < dim; ++a) {
                auto x = lo[a] + (p.mode == 2 ? 0 : (fraction[a] > .5 || (fraction[a] == .5 && (p.mode == 1 || lo[a]%2 != 0))));
                if (x < 0 || x >= static_cast<long long>(in.shape[a])) valid = false; else at += x*in.stride[a];
            }
            if (valid) dst[i] = src[channel*in.size+at];
            else { dst[i] = pad; if (used_pad) atomicExch(used_pad,1); }
        } else {
            using Value = std::conditional_t<std::is_same_v<T,double>,double,Real>;
            Value value;
            bool interior=false;
            Index base=0;
            if constexpr (D != 0) {
                interior=!p.validity;
                for (int a=0;a<D;++a) {
                    interior &= lo[a]>=0 && lo[a]+1<static_cast<long long>(in.shape[a]);
                    if (lo[a]>=0) base+=Index(lo[a])*Index(in.stride[a]);
                }
                if (interior) value=interior_sample<D,Value>(src+channel*in.size,base,in,fraction);
            }
            if (!interior) value=linear_sample<Value>(src+channel*in.size,in,lo,fraction,p.fill,p.validity);
            if constexpr(std::is_same_v<Value,float>) {
                // A float32 weighted sum can overflow near FLT_MAX even when
                // the convex interpolation result is representable.
                if(!isfinite(value)) value=float(linear_sample<double>(src+channel*in.size,in,lo,fraction,p.fill,p.validity));
            }
            dst[i] = T(value);
        }
    }
}
struct geometry_kernel_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { geometry_kernel(index,args...); }
};
template<class T, class Real, int D = 0, class Index = std::size_t> struct geometry_kernel_launch4 {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { geometry_kernel<T,Real,D,Index>(index,args...); }
};
DeviceArray resample_cuda(const DeviceArray& src, const Parameters& p, bool nearest, double fill, bool validity, const float* field) {
    Shape shape{src.shape[0]}; shape.insert(shape.end(),p.shape.begin(),p.shape.end());
    DeviceArray out(shape,src.kind,src.bytes);
    Layout in{Grid(src.spatial())}, output{Grid(p.shape)};
    const int reflected=p.device_offset?-1:reflection_axes(p,src.spatial(),field);
    if (reflected>=0 && (nearest || validity)) {
        reflect_cuda(src,out,unsigned(reflected),validity);
        return out;
    }
    // Bound float32 affine roundoff to < 0.001 voxel. Large coordinates or
    // cancellation between large matrix/offset terms use float64 coordinates.
    const bool narrow = !(src.kind == 'f' && src.bytes == 8) && geometry_float32(p, src.spatial());
    const auto run = [&]<class Real>() {
        GeometryArgs<Real> args{};
        if(p.device_offset) args.device_offset=std::static_pointer_cast<Buffer<long long>>(p.device_offset)->ptr;
        std::copy(p.matrix.begin(),p.matrix.end(),args.matrix); std::copy(p.offset.begin(),p.offset.end(),args.offset);
        args.fill = fill; args.border = p.border; args.nearest = nearest; args.validity = validity; args.mode = p.nearest_mode;
        args.identity = p.matrix == identity(p.shape.size());
        src.typed([&](auto input) {
            using T = typename decltype(input)::value_type;
            T pad{}; bool fill_valid = true;
            try { pad = Array::checked_cast<T>(fill); } catch (const std::invalid_argument&) { fill_valid = false; }
            require(std::is_floating_point_v<T> || nearest || validity,"CUDA linear interpolation requires floating-point data");
            Buffer<int> used(fill_valid || validity ? 0 : 1); int used_host = 0;
            if (used.size) used.upload(&used_host);
            bool specialized=false;
            if constexpr (std::is_same_v<T,float> && std::is_same_v<Real,float>) {
                if (out.size()<=UINT32_MAX && in.size<=UINT32_MAX) {
                    if (output.dim==3)
                        launch(blocks(out.size()),256,0,vcr::device::stream(), geometry_kernel_launch4<T,Real,3,std::uint32_t>{}, input.ptr,out.view<T>().ptr,in,output,src.shape[0],args,field,pad,used.ptr);
                    else
                        launch(blocks(out.size()),256,0,vcr::device::stream(), geometry_kernel_launch4<T,Real,2,std::uint32_t>{}, input.ptr,out.view<T>().ptr,in,output,src.shape[0],args,field,pad,used.ptr);
                    specialized=true;
                }
            }
            if (!specialized) launch(blocks(out.size()),256,0,vcr::device::stream(), geometry_kernel_launch{}, input.ptr,out.view<T>().ptr,in,output,src.shape[0],args,field,pad,used.ptr);
            check(cudaGetLastError());
            if (used.size) { used.download(&used_host); require(!used_host,"value is not representable in target dtype"); }
        });
    };
    if (narrow) run.template operator()<float>(); else run.template operator()<double>();
    return out;
}
template<class T> __device__ void resize_nearest_kernel(KernelIndex vcr_launch, const T* src,T* dst,Layout input,Layout output,
                                                       std::size_t channels,bool legacy) {
    EACH_INDEX(output.size*channels) {
        const auto voxel=i%output.size;
        std::size_t at=(i/output.size)*input.size;
        for(int axis=0;axis<output.dim;++axis)
            at+=resize_nearest_index(output.coord(voxel,axis),input.shape[axis],output.shape[axis],legacy)*input.stride[axis];
        dst[i]=src[at];
    }
}
struct resize_nearest_kernel_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { resize_nearest_kernel(index,args...); }
};
DeviceArray resize_cuda(const DeviceArray& src, const Shape& shape, bool nearest, bool legacy) {
    if(nearest) {
        Shape target{src.shape[0]};target.insert(target.end(),shape.begin(),shape.end());
        DeviceArray out(target,src.kind,src.bytes);
        src.typed([&](auto input) {
            using T=typename decltype(input)::value_type;
            launch(blocks(out.size()),256,0,vcr::device::stream(), resize_nearest_kernel_launch{}, input.ptr,out.view<T>().ptr,Layout{Grid(src.spatial())},Layout{Grid(shape)},src.shape[0],legacy);
        });
        check(cudaGetLastError());return out;
    }
    auto p = geometry(shape); p.border = true; p.nearest_mode = legacy ? 2 : 1;
    for (std::size_t a = 0; a < shape.size(); ++a) {
        const double ratio = double(src.shape[a+1])/shape[a];
        p.matrix[a*shape.size()+a] = ratio; p.offset[a] = legacy ? 0 : .5*ratio-.5;
    }
    return resample_cuda(src,p,nearest,0,false,nullptr);
}
void geometry_cuda(Sample& s, const Parameters& p, Session& session) {
    auto field = field_buffer(p.field,p.device_field);
    const float* displacement = p.field.empty() && !p.device_field ? nullptr : field->ptr;
    Sample specialized;
    for (auto& [key,f] : s) {
        auto role = f.target.role;
        if (sparse(role)) {
            // Host-supplied points keep the CPU math; resident ones stay on device.
            if (!session.arrays.count(key)) { specialized.emplace(key,f); continue; }
            auto& a = session.arrays.at(key);
            sparse_cuda(a,f.target,p,displacement,session); f.array.shape = a.shape; continue;
        }
        auto& a = session.arrays.at(key);
        const int reflected=p.device_offset?-1:reflection_axes(p,a.spatial(),displacement);
        if (reflected>=0) reflect_cuda(a,a,unsigned(reflected),role == Role::Valid);
        else a = resample_cuda(a,p,role == Role::Label || (p.exact && role != Role::Valid),f.target.fill,role == Role::Valid,field->ptr);
        if (directional(role) || (role == Role::CoordinateField && !f.target.external))
            directional_cuda(a,f.target,p,displacement,session);
        f.array.shape = a.shape;
    }
    if (!specialized.empty()) {
        auto cpu = p;
        if(p.device_offset) {
            std::vector<long long> offset(p.shape.size());
            std::static_pointer_cast<Buffer<long long>>(p.device_offset)->download(offset.data());
            std::copy(offset.begin(),offset.end(),cpu.offset.begin());
        }
        if (p.device_field && cpu.field.empty()) { cpu.field.resize(field->size); field->download(cpu.field.data()); }
        apply_geometry(specialized,cpu);
        for (auto& [key,f] : specialized) {
            if (!sparse(f.target.role)) {
                DeviceArray device(f.array.shape,f.array.kind,f.array.bytes);
                device.buffer->upload(static_cast<const std::byte*>(std::as_const(f.array).data()));
                session.arrays.at(key) = std::move(device);
            }
            s.at(key) = std::move(f);
        }
    }
    session.restore_valid(s);
}
} // namespace vcr::transforms::cuda
