#include "common.cuh"
#include "geometry.cuh"
namespace vcr::augment::cuda {
struct GeometryArgs {
    double matrix[9]{}, offset[3]{}, fill = 0;
    bool border = false, nearest = false, validity = false;
    int mode = 0;
};
template<class T> __global__ void geometry_kernel(const T* src, T* dst, Layout in, Layout out, std::size_t channels, GeometryArgs p, const double* field, T pad, int* used_pad) {
    EACH_INDEX(out.size*channels) {
        const auto voxel = i%out.size, channel = i/out.size;
        double q[3]{}, fraction[3]{}; long long lo[3]{};
        bool far = false;
        for (int a = 0; a < out.dim; ++a) {
            for (int b = 0; b < out.dim; ++b) q[a] += p.matrix[a*out.dim+b]*double(out.coord(voxel,b));
            q[a] += p.offset[a];
            if (field) q[a] += field[voxel*out.dim+a];
            if (p.border && !p.validity) q[a] = clamp_value(q[a],0,double(in.shape[a]-1));
            if (!isfinite(q[a]) || q[a] < -1 || q[a] > double(in.shape[a])) { far = true; continue; }
            lo[a] = static_cast<long long>(floor(q[a])); fraction[a] = q[a]-lo[a];
        }
        if (far) { dst[i] = p.validity ? T(p.fill>=1) : pad; atomicExch(used_pad,1); continue; }
        if (p.nearest) {
            bool valid = true; std::size_t at = 0;
            for (int a = 0; a < out.dim; ++a) {
                auto x = lo[a] + (p.mode == 2 ? 0 : (fraction[a] > .5 || (fraction[a] == .5 && (p.mode == 1 || lo[a]%2 != 0))));
                if (x < 0 || x >= static_cast<long long>(in.shape[a])) valid = false; else at += x*in.stride[a];
            }
            if (valid) dst[i] = src[channel*in.size+at];
            else { dst[i] = pad; atomicExch(used_pad,1); }
        } else {
            double value = 0;
            for (int corner = 0; corner < (1<<out.dim); ++corner) {
                double weight = 1; std::size_t at = 0; bool valid = true;
                for (int a = 0; a < out.dim; ++a) {
                    int bit = (corner>>a)&1; auto x = lo[a]+bit;
                    weight *= bit ? fraction[a] : 1-fraction[a];
                    if (x < 0 || x >= static_cast<long long>(in.shape[a])) valid = false; else at += x*in.stride[a];
                }
                if (weight) value += weight*(valid ? double(src[channel*in.size+at]) : p.fill);
            }
            if (p.validity) value = double(value >= 1.);
            dst[i] = T(value);
        }
    }
}
Array resample_cuda(const Array& src, const Parameters& p, bool nearest, double fill, bool validity, std::size_t) {
    Shape shape{src.shape[0]}; shape.insert(shape.end(),p.shape.begin(),p.shape.end());
    auto out = Array::uninitialized(shape,src.kind,src.bytes);
    Layout in{Grid(src.spatial())}, output{Grid(p.shape)};
    GeometryArgs args{}; std::copy(p.matrix.begin(),p.matrix.end(),args.matrix); std::copy(p.offset.begin(),p.offset.end(),args.offset);
    args.fill = fill; args.border = p.border; args.nearest = nearest; args.validity = validity; args.mode = p.nearest_mode;
    Buffer<double> field(p.field);
    src.typed([&](const auto* host) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(host)>>;
        T pad{}; bool fill_valid = true;
        try { pad = Array::checked_cast<T>(fill); } catch (const std::invalid_argument&) { fill_valid = false; }
        // Integer linear sampling is only used for binary validity masks.
        require(std::is_floating_point_v<T> || nearest || validity, "CUDA linear interpolation requires floating-point data");
        Buffer<T> input(src.size()), result(out.size()); input.upload(host);
        Buffer<int> used(1); int used_host = 0; used.upload(&used_host);
        geometry_kernel<<<blocks(out.size()),256>>>(input.ptr,result.ptr,in,output,src.shape[0],args,field.ptr,pad,used.ptr);
        check(cudaGetLastError()); used.download(&used_host);
        require(fill_valid || !used_host || validity, "value is not representable in target dtype");
        result.download(static_cast<T*>(out.data()));
    });
    return out;
}
Array resize_cuda(const Array& src, const Shape& shape, bool nearest, bool legacy) {
    auto p = geometry(shape); p.border = true; p.nearest_mode = legacy ? 2 : 1;
    for (std::size_t a = 0; a < shape.size(); ++a) {
        const double ratio = double(src.shape[a+1])/shape[a];
        p.matrix[a*shape.size()+a] = ratio; p.offset[a] = legacy ? 0 : .5*ratio-.5;
    }
    return resample_cuda(src,p,nearest,0,false,3);
}

} // namespace vcr::augment::cuda
