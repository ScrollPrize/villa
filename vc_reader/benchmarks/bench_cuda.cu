// Optional benchmark only: no CUDA dependency or dispatch in the CPU library.
// One finite float32 channel, cubic 3D grid, size-3 median or Elastic(3,1)
// with an int16 label channel, zero fill, and ties-to-even nearest sampling.
#include "spatial.hpp"
#include "cuda/median.cuh"
#include "cuda/reductions.cuh"
using vcr::augment::cuda::block_reduce;
using vcr::augment::cuda::finish_reduce;
#include <cuda_runtime.h>

namespace {
thread_local std::string error;
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Context {
    int edge, count;
    float *image = nullptr, *output = nullptr;
    short *labels = nullptr, *output_labels = nullptr;
    double *field = nullptr, *scratch1 = nullptr, *scratch2 = nullptr;
    double *weights = nullptr, *partial = nullptr, *stats = nullptr;
    int knots;
    cudaEvent_t start = nullptr, stop = nullptr;
    explicit Context(int n) : edge(n), count(n*n*n),
        knots(int(vcr::augment::elastic_detail::controls(n, 6, true))) {}
    ~Context() {
        cudaFree(image); cudaFree(output); cudaFree(labels);
        cudaFree(output_labels); cudaFree(field);
        cudaFree(scratch1); cudaFree(scratch2); cudaFree(weights);
        cudaFree(partial); cudaFree(stats);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }
    void allocate() {
        check(cudaMalloc(&image, count*sizeof(float)));
        check(cudaMalloc(&output, count*sizeof(float)));
        check(cudaMalloc(&labels, count*sizeof(short)));
        check(cudaMalloc(&output_labels, count*sizeof(short)));
        check(cudaMalloc(&field, count*3*sizeof(double)));
        const auto scratch_count = std::max(count, knots*knots*knots);
        check(cudaMalloc(&scratch1, scratch_count*3*sizeof(double)));
        check(cudaMalloc(&scratch2, scratch_count*3*sizeof(double)));
        check(cudaMalloc(&weights, edge*4*sizeof(double)));
        check(cudaMalloc(&partial, ((count+255)/256)*10*sizeof(double)));
        check(cudaMalloc(&stats, 10*sizeof(double)));
        std::vector<double> host_weights;
        for (int x = 0; x < edge; ++x) {
            const double q = double(x)/6;
            const auto w = vcr::augment::elastic_detail::weights(q-std::floor(q), true);
            host_weights.insert(host_weights.end(), w.begin(), w.end());
        }
        check(cudaMemcpy(weights, host_weights.data(), edge*4*sizeof(double), cudaMemcpyHostToDevice));
        check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    }
};

// GPU field experiment: double precision throughout. Expansion preserves the
// CPU's tap order. Fixed reduction trees are deterministic, but intentionally
// tested for rounding differences from the CPU's serial/component reductions.
__global__ void expand(const double* src, double* dst, const double* weights,
                       int n, int k, int axis) {
    const int size = axis == 2 ? k*k*n : axis == 1 ? k*n*n : n*n*n;
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= 3*size) return;
    const int c = i/size, at = i%size;
    const int step = axis == 2 ? 1 : axis == 1 ? n : n*n;
    const int x = at/step % n, block = at/(n*step), j = at%step;
    const int input_size = axis == 2 ? k*k*k : axis == 1 ? k*k*n : k*n*n;
    const auto* input = src+c*input_size+(block*k+x/6)*step+j;
    double value = 0;
    #pragma unroll
    for (int tap = 0; tap < 4; ++tap) value += weights[4*x+tap]*input[tap*step];
    dst[axis == 0 ? at*3+c : i] = value;
}

__global__ void component_sums(const double* field, double* partial, int count) {
    __shared__ double values[3][256];
    const int t = threadIdx.x, i = blockIdx.x*256+t;
    #pragma unroll
    for (int c = 0; c < 3; ++c) values[c][t] = i < count ? field[3*i+c] : 0;
    block_reduce(values);
    if (t == 0) for (int c = 0; c < 3; ++c) partial[blockIdx.x*3+c] = values[c][0];
}
__global__ void center(double* field, const double* sums, int count) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < count*3) field[i] -= sums[i%3]/count;
}
__global__ void energy_gradient(const double* field, double* partial, int n) {
    __shared__ double values[10][256];
    const int t = threadIdx.x, i = blockIdx.x*256+t;
    for (int m = 0; m < 10; ++m) values[m][t] = 0;
    if (i < n*n*n) {
        const int pos[3] = {i/(n*n), i/n%n, i%n}, stride[3] = {n*n,n,1};
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            const double value = field[3*i+c];
            values[0][t] += value*value;
            #pragma unroll
            for (int a = 0; a < 3; ++a)
                if (pos[a]+1 < n) values[1+3*c+a][t] = fabs(field[3*(i+stride[a])+c]-value);
        }
    }
    block_reduce(values);
    if (t == 0) for (int m = 0; m < 10; ++m) partial[blockIdx.x*10+m] = values[m][0];
}
__global__ void scale_factor(double* stats, int count) {
    double factor = stats[0] ? 1/sqrt(stats[0]/count) : 0, limit = 0;
    for (int c = 0; c < 3; ++c)
        limit = fmax(limit, stats[1+3*c]+stats[2+3*c]+stats[3+3*c]);
    if (limit) factor = fmin(factor, .8/limit);
    stats[0] = factor;
}
__global__ void scale(double* field, const double* stats, int count) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < count*3) field[i] *= stats[0];
}
void gpu_field(Context& c) {
    const int k = c.knots, n = c.edge, blocks = (c.count+255)/256;
    expand<<<(3*k*k*n+255)/256,256>>>(c.scratch1,c.scratch2,c.weights,n,k,2);
    expand<<<(3*k*n*n+255)/256,256>>>(c.scratch2,c.scratch1,c.weights,n,k,1);
    expand<<<(3*c.count+255)/256,256>>>(c.scratch1,c.field,c.weights,n,k,0);
    component_sums<<<blocks,256>>>(c.field,c.partial,c.count);
    finish_reduce<3><<<1,256>>>(c.partial,c.stats,blocks);
    center<<<(3*c.count+255)/256,256>>>(c.field,c.stats,c.count);
    energy_gradient<<<blocks,256>>>(c.field,c.partial,n);
    finish_reduce<10><<<1,256>>>(c.partial,c.stats,blocks);
    scale_factor<<<1,1>>>(c.stats,c.count);
    scale<<<(3*c.count+255)/256,256>>>(c.field,c.stats,c.count);
}

__global__ void median(const float* src, float* dst, int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n*n*n) return;
    const int z = i/(n*n), y = i/n % n, x = i % n;
    float v[27];
    #pragma unroll
    for (int dz = 0; dz < 3; ++dz) {
        #pragma unroll
        for (int dy = 0; dy < 3; ++dy) {
            #pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
                const int zz = max(0, min(n-1, z+dz-1));
                const int yy = max(0, min(n-1, y+dy-1));
                const int xx = max(0, min(n-1, x+dx-1));
                v[(dz*3+dy)*3+dx] = src[(zz*n+yy)*n+xx];
            }
        }
    }
    vcr::augment::cuda::select27(v);
    dst[i] = v[vcr::augment::median_detail::network.output];
}
__global__ void elastic(const float* src, const short* labels, const double* field,
                        float* dst, short* dst_labels, int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n*n*n) return;
    const int pos[3] = {i/(n*n), i/n % n, i % n};
    const int stride[3] = {n*n, n, 1};
    int lo[3], nearest = 0;
    double fraction[3];
    bool label_valid = true;
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        const double q = double(pos[a]) + field[3*i+a];
        lo[a] = int(floor(q)); fraction[a] = q - lo[a];
        const int at = lo[a] + (fraction[a] > .5 ||
                               (fraction[a] == .5 && lo[a] % 2 != 0));
        label_valid &= at >= 0 && at < n;
        nearest += at * stride[a];
    }
    double result = 0;
    // Preserve the CPU's axis, corner, multiplication and accumulation order.
    // Compile with --fmad=false, without --use_fast_math.
    #pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        double weight = 1;
        int at = 0; bool valid = true;
        #pragma unroll
        for (int a = 0; a < 3; ++a) {
            const int bit = (corner >> a) & 1, x = lo[a] + bit;
            weight *= bit ? fraction[a] : 1 - fraction[a];
            valid &= x >= 0 && x < n; at += x * stride[a];
        }
        if (weight) result += weight * (valid ? double(src[at]) : 0.);
    }
    dst[i] = float(result);
    dst_labels[i] = label_valid ? labels[nearest] : 0;
}
} // namespace

extern "C" const char* bench_cuda_error() { return error.c_str(); }
extern "C" void* bench_cuda_create(int edge) {
    try {
        if (edge < 1 || edge > 256) throw std::invalid_argument("edge must be 1..256");
        auto ctx = std::make_unique<Context>(edge); ctx->allocate(); return ctx.release();
    } catch (const std::exception& e) { error = e.what(); return nullptr; }
}
extern "C" void bench_cuda_destroy(void* ctx) { delete static_cast<Context*>(ctx); }

// Op 1 reuses the production CPU field as a reference experiment. Op 2 draws
// only the small random control lattice on CPU; dense expansion/normalization
// run on GPU. Both reuse production RNG code and are timed by Python.
extern "C" void* bench_cuda_field(int edge, std::uint64_t seed, int op, const double** data) {
    try {
        using namespace vcr::augment;
        Field f; f.array.shape = {1, std::size_t(edge), std::size_t(edge), std::size_t(edge)};
        Sample s{{"image", std::move(f)}};
        Random r{split_seed(seed, "0")}; r.chance(1);
        auto field = std::make_unique<std::vector<double>>();
        if (op == 2) {
            const auto k = elastic_detail::controls(edge, 6, true);
            for (int c = 0; c < 3; ++c) {
                auto values = elastic_detail::draw_controls({k,k,k}, r);
                field->insert(field->end(), values.begin(), values.end());
            }
        } else *field = sample(Elastic{3., 1.}, s, r).field;
        *data = field->data(); return field.release();
    } catch (const std::exception& e) { error = e.what(); return nullptr; }
}
extern "C" void bench_cuda_free_field(void* field) {
    delete static_cast<std::vector<double>*>(field);
}
extern "C" int bench_cuda_run(void* handle, int op, const float* image,
                               const short* labels, const double* field,
                               float* output, short* output_labels, float* kernel_ms) {
    try {
        auto& c = *static_cast<Context*>(handle);
        check(cudaMemcpy(c.image, image, c.count*sizeof(float), cudaMemcpyHostToDevice));
        if (op != 0) {
            check(cudaMemcpy(c.labels, labels, c.count*sizeof(short), cudaMemcpyHostToDevice));
            if (op == 2) check(cudaMemcpy(c.scratch1, field, c.knots*c.knots*c.knots*3*sizeof(double), cudaMemcpyHostToDevice));
            else check(cudaMemcpy(c.field, field, c.count*3*sizeof(double), cudaMemcpyHostToDevice));
        }
        check(cudaEventRecord(c.start));
        if (op == 2) gpu_field(c);
        if (op != 0) elastic<<<(c.count+127)/128,128>>>(c.image, c.labels, c.field, c.output, c.output_labels, c.edge);
        else median<<<(c.count+127)/128,128>>>(c.image, c.output, c.edge);
        check(cudaGetLastError()); check(cudaEventRecord(c.stop));
        check(cudaMemcpy(output, c.output, c.count*sizeof(float), cudaMemcpyDeviceToHost));
        if (op != 0) check(cudaMemcpy(output_labels, c.output_labels, c.count*sizeof(short), cudaMemcpyDeviceToHost));
        check(cudaEventSynchronize(c.stop));
        check(cudaEventElapsedTime(kernel_ms, c.start, c.stop));
        return 0;
    } catch (const std::exception& e) { error = e.what(); return 1; }
}
extern "C" int bench_cuda_copy_field(void* handle, double* output) {
    try {
        auto& c = *static_cast<Context*>(handle);
        check(cudaMemcpy(output,c.field,c.count*3*sizeof(double),cudaMemcpyDeviceToHost));
        return 0;
    } catch (const std::exception& e) { error = e.what(); return 1; }
}
