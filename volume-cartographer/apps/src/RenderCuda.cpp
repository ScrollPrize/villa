// RenderCuda.cpp -- see RenderCuda.hpp.
#include "RenderCuda.hpp"

#include "vc/core/util/Compositing.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

namespace vc::render::cuda {
namespace {

// ============================================================
// The kernels
// ============================================================
//
// Compiled by NVRTC with --fmad=false, and every arithmetic step is an explicit round-to-nearest
// intrinsic, so nothing is fused or reordered that Slicing.cpp does not fuse: the only fused
// multiply-adds are the std::fma chain of ChunkSampler::sampleTrilinear, and the position
// b + d * off when the host compiler fuses it too (see hostContractsMulAdd).

constexpr int kMaxMedianLayers = 256;  // the median kernel sorts in a per-thread array

constexpr const char* kKernelSource = R"CUDA(
typedef unsigned char u8;
typedef unsigned short u16;

struct Vol {
    int sz, sy, sx;         // extent (z, y, x)
    int csz, csy, csx;      // chunk shape
    int ngy, ngx;           // chunk grid (y, x)
    long long chunkElems;   // csz * csy * csx
};

__device__ __forceinline__ bool finite_bits(float f)
{
    return (__float_as_uint(f) & 0x7f800000u) != 0x7f800000u;
}

// b + d * off as the host computes it: one rounding when the host compiler contracts the
// expression into a fused multiply-add, two otherwise.
__device__ __forceinline__ float pos(float b, float d, float off, int contract)
{
    return contract ? __fmaf_rn(d, off, b) : __fadd_rn(b, __fmul_rn(d, off));
}

__device__ __forceinline__ bool in_bounds(const Vol& v, float vz, float vy, float vx)
{
    return vz >= 0.f && vy >= 0.f && vx >= 0.f && vz < (float)v.sz && vy < (float)v.sy && vx < (float)v.sx;
}

// ChunkSampler::sampleInt: 0 outside the volume and in a chunk that is not stored.
template <typename T>
__device__ __forceinline__ float vox(const int* tab, const T* pool, const Vol& v, int iz, int iy, int ix)
{
    if ((unsigned)iz >= (unsigned)v.sz || (unsigned)iy >= (unsigned)v.sy || (unsigned)ix >= (unsigned)v.sx)
        return 0.0f;
    const int cz = iz / v.csz, cy = iy / v.csy, cx = ix / v.csx;
    const int slot = tab[((long long)cz * v.ngy + cy) * v.ngx + cx];
    if (slot < 0) return 0.0f;
    const int lz = iz - cz * v.csz, ly = iy - cy * v.csy, lx = ix - cx * v.csx;
    return (float)pool[(size_t)slot * (size_t)v.chunkElems + ((size_t)lz * v.csy + ly) * v.csx + lx];
}

// ChunkSampler::sampleTrilinear
template <typename T>
__device__ float trilinear(const int* tab, const T* pool, const Vol& v, float vz, float vy, float vx)
{
    const int iz = (int)vz, iy = (int)vy, ix = (int)vx;
    const float c000 = vox(tab, pool, v, iz, iy, ix);
    const float c100 = vox(tab, pool, v, iz + 1, iy, ix);
    const float c010 = vox(tab, pool, v, iz, iy + 1, ix);
    const float c110 = vox(tab, pool, v, iz + 1, iy + 1, ix);
    const float c001 = vox(tab, pool, v, iz, iy, ix + 1);
    const float c101 = vox(tab, pool, v, iz + 1, iy, ix + 1);
    const float c011 = vox(tab, pool, v, iz, iy + 1, ix + 1);
    const float c111 = vox(tab, pool, v, iz + 1, iy + 1, ix + 1);
    const float fz = __fsub_rn(vz, (float)iz), fy = __fsub_rn(vy, (float)iy), fx = __fsub_rn(vx, (float)ix);
    const float c00 = __fmaf_rn(fx, __fsub_rn(c001, c000), c000);
    const float c01 = __fmaf_rn(fx, __fsub_rn(c011, c010), c010);
    const float c10 = __fmaf_rn(fx, __fsub_rn(c101, c100), c100);
    const float c11 = __fmaf_rn(fx, __fsub_rn(c111, c110), c110);
    const float c0 = __fmaf_rn(fy, __fsub_rn(c01, c00), c00);
    const float c1 = __fmaf_rn(fy, __fsub_rn(c11, c10), c10);
    return __fmaf_rn(fz, __fsub_rn(c1, c0), c0);
}

// readMultiSliceImpl / sampleTileSlicesImpl: uint8 truncates, uint16 rounds.
template <typename T> struct Pixel;
template <> struct Pixel<u8> {
    static __device__ __forceinline__ float max() { return 255.0f; }
    static __device__ __forceinline__ u8 convert(float s) { return (u8)(int)s; }
};
template <> struct Pixel<u16> {
    static __device__ __forceinline__ float max() { return 65535.0f; }
    static __device__ __forceinline__ u16 convert(float s) { return (u16)(int)__fadd_rn(s, 0.5f); }
};

// The layers of the columns [c0, c1) of a band: out[i][r][c] is the sample at offset i.
template <typename T>
__device__ __forceinline__ void slices(const float* base, const float* dirs, int w, int h, int c0, int c1,
                                       const float* offs, int n, const int* tab, const T* pool, const Vol& v,
                                       int contract, T* out)
{
    const int c = c0 + blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= c1 || r >= h) return;
    const size_t pix = (size_t)r * w + c;
    const float bx = base[3 * pix], by = base[3 * pix + 1], bz = base[3 * pix + 2];
    const float dx = dirs[3 * pix], dy = dirs[3 * pix + 1], dz = dirs[3 * pix + 2];
    for (int i = 0; i < n; i++) {
        const float off = offs[i];
        const float px = pos(bx, dx, off, contract), py = pos(by, dy, off, contract), pz = pos(bz, dz, off, contract);
        T val = 0;
        if (in_bounds(v, pz, py, px)) {
            float s = trilinear(tab, pool, v, pz, py, px);
            if (s < 0.f) s = 0.f;
            if (s > Pixel<T>::max()) s = Pixel<T>::max();
            val = Pixel<T>::convert(s);
        }
        out[(size_t)i * h * w + pix] = val;
    }
}

extern "C" __global__ void vcr_slices_u8(const float* base, const float* dirs, int w, int h, int c0, int c1,
                                         const float* offs, int n, const int* tab, const u8* pool, Vol v,
                                         int contract, u8* out)
{
    slices<u8>(base, dirs, w, h, c0, c1, offs, n, tab, pool, v, contract, out);
}

extern "C" __global__ void vcr_slices_u16(const float* base, const float* dirs, int w, int h, int c0, int c1,
                                          const float* offs, int n, const int* tab, const u16* pool, Vol v,
                                          int contract, u16* out)
{
    slices<u16>(base, dirs, w, h, c0, c1, offs, n, tab, pool, v, contract, out);
}

// sampleOne<uint8_t, Nearest>: 0 outside the volume, else the voxel nearest to the position.
__device__ __forceinline__ float nearest_u8(const int* tab, const u8* pool, const Vol& v, float pz, float py, float px)
{
    if (!in_bounds(v, pz, py, px)) return 0.f;
    int iz = (int)__fadd_rn(pz, 0.5f), iy = (int)__fadd_rn(py, 0.5f), ix = (int)__fadd_rn(px, 0.5f);
    if (iz >= v.sz) iz = v.sz - 1;
    if (iy >= v.sy) iy = v.sy - 1;
    if (ix >= v.sx) ix = v.sx - 1;
    return vox(tab, pool, v, iz, iy, ix);
}

// readCompositeFastImpl<uint8_t, Nearest> with the max (0), min (1) and mean (2) reducers: a pixel
// without a finite base or direction, or without a layer at or above the cutoff, is left alone.
extern "C" __global__ void vcr_composite_u8(const float* base, const float* dirs, int w, int h, int c0, int c1,
                                            float zStep, int zStart, int numLayers, const int* tab,
                                            const u8* pool, Vol v, int contract, int mode, float isoCutoff,
                                            u8* out)
{
    const int c = c0 + blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= c1 || r >= h) return;
    const size_t pix = (size_t)r * w + c;
    const float bx = base[3 * pix], by = base[3 * pix + 1], bz = base[3 * pix + 2];
    const float dx = dirs[3 * pix], dy = dirs[3 * pix + 1], dz = dirs[3 * pix + 2];
    if (!finite_bits(bx) || !finite_bits(dx)) return;
    float accum = 0.f, mx = 0.f, mn = 255.f;
    int count = 0;
    for (int li = 0; li < numLayers; li++) {
        const float z = __fmul_rn((float)(zStart + li), zStep);
        const float s = nearest_u8(tab, pool, v, pos(bz, dz, z, contract), pos(by, dy, z, contract),
                                   pos(bx, dx, z, contract));
        if (s < isoCutoff) continue;
        if (mode == 0) mx = fmaxf(mx, s);
        else if (mode == 1) mn = fminf(mn, s);
        else accum = __fadd_rn(accum, s);
        count++;
    }
    if (count == 0) return;
    float val = mode == 0 ? mx : mode == 1 ? mn : __fdiv_rn(accum, (float)count);
    if (val < 0.f) val = 0.f;
    if (val > 255.f) val = 255.f;
    out[pix] = (u8)(int)val;
}

// The same with the median reducer: the (count / 2)-th smallest layer, as the host's partial_sort.
extern "C" __global__ void vcr_median_u8(const float* base, const float* dirs, int w, int h, int c0, int c1,
                                         float zStep, int zStart, int numLayers, const int* tab,
                                         const u8* pool, Vol v, int contract, float isoCutoff, u8* out)
{
    const int c = c0 + blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= c1 || r >= h) return;
    const size_t pix = (size_t)r * w + c;
    const float bx = base[3 * pix], by = base[3 * pix + 1], bz = base[3 * pix + 2];
    const float dx = dirs[3 * pix], dy = dirs[3 * pix + 1], dz = dirs[3 * pix + 2];
    if (!finite_bits(bx) || !finite_bits(dx)) return;
    float vals[VCR_MAX_MEDIAN_LAYERS];
    int count = 0;
    for (int li = 0; li < numLayers && count < VCR_MAX_MEDIAN_LAYERS; li++) {
        const float z = __fmul_rn((float)(zStart + li), zStep);
        const float s = nearest_u8(tab, pool, v, pos(bz, dz, z, contract), pos(by, dy, z, contract),
                                   pos(bx, dx, z, contract));
        if (s < isoCutoff) continue;
        int j = count;
        while (j > 0 && vals[j - 1] > s) { vals[j] = vals[j - 1]; j--; }
        vals[j] = s;
        count++;
    }
    if (count == 0) return;
    float val = vals[count / 2];
    if (val < 0.f) val = 0.f;
    if (val > 255.f) val = 255.f;
    out[pix] = (u8)(int)val;
}

// The chunks the samples of the columns [c0, c1) can touch: per pixel, the box around the segment
// from offset omin to omax, widened by the trilinear / nearest neighbourhood.
extern "C" __global__ void vcr_mark(const float* base, const float* dirs, int w, int h, int c0, int c1,
                                    float omin, float omax, Vol v, int contract, u8* mask)
{
    const int c = c0 + blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= c1 || r >= h) return;
    const size_t pix = (size_t)r * w + c;
    const float b[3] = {base[3 * pix], base[3 * pix + 1], base[3 * pix + 2]};
    const float d[3] = {dirs[3 * pix], dirs[3 * pix + 1], dirs[3 * pix + 2]};
    const int lim[3] = {v.sx, v.sy, v.sz}, cs[3] = {v.csx, v.csy, v.csz};
    int lo[3], hi[3];
    for (int k = 0; k < 3; k++) {
        const float e1 = pos(b[k], d[k], omin, contract), e2 = pos(b[k], d[k], omax, contract);
        const float mn = fminf(e1, e2), mx = fmaxf(e1, e2), top = (float)lim[k] + 4.0f;
        if (!(mx >= -4.0f) || !(mn <= top)) return;  // off the volume, or not a number
        lo[k] = max(0, (int)floorf(fmaxf(mn, -4.0f)) - 1);
        hi[k] = min(lim[k] - 1, (int)floorf(fminf(mx, top)) + 2);
        if (lo[k] > hi[k]) return;
    }
    for (int cz = lo[2] / cs[2]; cz <= hi[2] / cs[2]; cz++)
        for (int cy = lo[1] / cs[1]; cy <= hi[1] / cs[1]; cy++)
            for (int cx = lo[0] / cs[0]; cx <= hi[0] / cs[0]; cx++)
                mask[((long long)cz * v.ngy + cy) * v.ngx + cx] = 1;
}
)CUDA";

constexpr const char* kKernelNames[] = {"vcr_mark", "vcr_slices_u8", "vcr_slices_u16", "vcr_composite_u8",
                                        "vcr_median_u8"};
enum Kernel { kMark = 0, kSlicesU8 = 1, kSlicesU16 = 2, kCompositeU8 = 3, kMedianU8 = 4 };

constexpr unsigned kBlockX = 128, kBlockY = 4;

// The host side of the kernels' Vol: same members, same order, natural alignment on both sides.
struct Vol {
    int sz = 0, sy = 0, sx = 0;
    int csz = 0, csy = 0, csx = 0;
    int ngy = 0, ngx = 0;
    long long chunkElems = 0;
};

// ============================================================
// Host arithmetic probe
// ============================================================

// Slicing.cpp computes each sample position as b + d * off. GCC (-ffp-contract=fast outside strict
// ISO mode) and Clang (-ffp-contract=on) fuse that into one fma instruction when the target has
// one, which rounds once; MSVC's /fp:precise rounds twice. The kernels must round like the build
// they ship in, so the same expression is evaluated here on inputs for which the two roundings
// differ, and the kernels are told which one the host does.
bool hostContractsMulAdd()
{
    volatile float a = 1.0f + 1.0f / 4096.0f;  // 1 + 2^-12, so a * a = 1 + 2^-11 + 2^-24
    volatile float b = a;
    volatile float c = -1.0f;
    const float r = c + a * b;                 // fused: 2^-11 + 2^-24; rounded twice: 2^-11
    return r != 1.0f / 2048.0f;
}

// ============================================================
// The driver API and NVRTC, loaded at run time
// ============================================================

#ifdef _WIN32
using Lib = HMODULE;
#define VC_CUDA_API __stdcall
Lib libOpen(const std::string& path)
{
    // A bare name searches the usual places; a path loads from there, with the directory first so
    // NVRTC finds its builtins library beside itself.
    return path.find_first_of("/\\") == std::string::npos
        ? LoadLibraryA(path.c_str())
        : LoadLibraryExA(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
}
void* libSym(Lib lib, const char* name) { return reinterpret_cast<void*>(GetProcAddress(lib, name)); }
void libClose(Lib lib) { if (lib) FreeLibrary(lib); }
constexpr const char* kDriverNames[] = {"nvcuda.dll"};
constexpr const char* kNvrtcNames[] = {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll"};
constexpr char kPathSep = '\\';
#else
using Lib = void*;
#define VC_CUDA_API
Lib libOpen(const std::string& path) { return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL); }
void* libSym(Lib lib, const char* name) { return dlsym(lib, name); }
void libClose(Lib lib) { if (lib) dlclose(lib); }
constexpr const char* kDriverNames[] = {"libcuda.so.1", "libcuda.so"};
constexpr const char* kNvrtcNames[] = {"libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"};
constexpr char kPathSep = '/';
#endif

std::string exeDir()
{
    char buf[4096];
#ifdef _WIN32
    const DWORD n = GetModuleFileNameA(nullptr, buf, DWORD(sizeof buf));
    if (n == 0 || n >= sizeof buf) return {};
    const std::string p(buf, n);
#elif defined(__APPLE__)
    uint32_t n = uint32_t(sizeof buf);
    if (_NSGetExecutablePath(buf, &n) != 0) return {};
    const std::string p(buf);
#else
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n <= 0) return {};
    const std::string p(buf, std::size_t(n));
#endif
    const auto s = p.find_last_of("/\\");
    return s == std::string::npos ? std::string{} : p.substr(0, s + 1);
}

// Where NVRTC may be: VC_NVRTC_DIR, beside the executable, the CUDA toolkit, the system search path.
std::vector<std::string> nvrtcCandidates()
{
    std::vector<std::string> dirs;
    if (const char* e = std::getenv("VC_NVRTC_DIR"); e && *e) dirs.emplace_back(e);
    if (auto d = exeDir(); !d.empty()) dirs.push_back(std::move(d));
#ifdef _WIN32
    if (const char* e = std::getenv("CUDA_PATH"); e && *e) dirs.push_back(std::string(e) + "\\bin");
#else
    dirs.emplace_back("/usr/local/cuda/lib64");
#endif
    std::vector<std::string> out;
    for (auto dir : dirs) {
        if (dir.back() != '/' && dir.back() != '\\') dir += kPathSep;
        for (const char* name : kNvrtcNames) out.push_back(dir + name);
    }
    for (const char* name : kNvrtcNames) out.emplace_back(name);
    return out;
}

using CUresult = int;
using CUdevice = int;
using CUcontext = struct CUctx_st*;
using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;
using CUstream = struct CUstream_st*;
using CUdeviceptr = unsigned long long;
using nvrtcResult = int;
using nvrtcProgram = struct _nvrtcProgram*;

constexpr int kAttrComputeCapabilityMajor = 75, kAttrComputeCapabilityMinor = 76;

struct DriverApi {
    CUresult(VC_CUDA_API* Init)(unsigned) = nullptr;
    CUresult(VC_CUDA_API* DeviceGetCount)(int*) = nullptr;
    CUresult(VC_CUDA_API* DeviceGet)(CUdevice*, int) = nullptr;
    CUresult(VC_CUDA_API* DeviceGetName)(char*, int, CUdevice) = nullptr;
    CUresult(VC_CUDA_API* DeviceGetAttribute)(int*, int, CUdevice) = nullptr;
    CUresult(VC_CUDA_API* DeviceGetPCIBusId)(char*, int, CUdevice) = nullptr;
    CUresult(VC_CUDA_API* PrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult(VC_CUDA_API* PrimaryCtxRelease)(CUdevice) = nullptr;
    CUresult(VC_CUDA_API* CtxSetCurrent)(CUcontext) = nullptr;
    CUresult(VC_CUDA_API* CtxSynchronize)() = nullptr;
    CUresult(VC_CUDA_API* ModuleLoadData)(CUmodule*, const void*) = nullptr;
    CUresult(VC_CUDA_API* ModuleUnload)(CUmodule) = nullptr;
    CUresult(VC_CUDA_API* ModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult(VC_CUDA_API* MemAlloc)(CUdeviceptr*, std::size_t) = nullptr;
    CUresult(VC_CUDA_API* MemFree)(CUdeviceptr) = nullptr;
    CUresult(VC_CUDA_API* MemcpyHtoD)(CUdeviceptr, const void*, std::size_t) = nullptr;
    CUresult(VC_CUDA_API* MemcpyDtoH)(void*, CUdeviceptr, std::size_t) = nullptr;
    CUresult(VC_CUDA_API* MemsetD8)(CUdeviceptr, unsigned char, std::size_t) = nullptr;
    CUresult(VC_CUDA_API* MemGetInfo)(std::size_t*, std::size_t*) = nullptr;
    CUresult(VC_CUDA_API* LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                        unsigned, CUstream, void**, void**) = nullptr;
    CUresult(VC_CUDA_API* GetErrorString)(CUresult, const char**) = nullptr;
};

struct NvrtcApi {
    nvrtcResult (*CreateProgram)(nvrtcProgram*, const char*, const char*, int, const char* const*,
                                 const char* const*) = nullptr;
    nvrtcResult (*CompileProgram)(nvrtcProgram, int, const char* const*) = nullptr;
    nvrtcResult (*GetProgramLogSize)(nvrtcProgram, std::size_t*) = nullptr;
    nvrtcResult (*GetProgramLog)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*GetCUBINSize)(nvrtcProgram, std::size_t*) = nullptr;
    nvrtcResult (*GetCUBIN)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*GetPTXSize)(nvrtcProgram, std::size_t*) = nullptr;
    nvrtcResult (*GetPTX)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*DestroyProgram)(nvrtcProgram*) = nullptr;
    const char* (*GetErrorString)(nvrtcResult) = nullptr;
};

template <typename Fn>
bool loadSym(Lib lib, Fn& fn, const char* name, std::string& why)
{
    void* p = libSym(lib, name);
    if (!p) {
        why = std::string("missing symbol ") + name;
        return false;
    }
    static_assert(sizeof(fn) == sizeof(p), "function pointers are object-pointer sized here");
    std::memcpy(&fn, &p, sizeof p);
    return true;
}

// One device: the primary context of device 0, one module of the kernels above, and the memory
// and launch calls the sampler needs. Open failures are soft (nullptr and why); everything after
// that throws std::runtime_error with the driver's own error text.
class Device {
public:
    static std::unique_ptr<Device> open(std::string& why)
    {
        std::unique_ptr<Device> dev(new Device);
        if (!dev->init(why)) return nullptr;
        return dev;
    }

    ~Device()
    {
        if (mod_) d_.ModuleUnload(mod_);
        if (ctx_) d_.PrimaryCtxRelease(dev_);
        libClose(nvrtcLib_);
        libClose(builtinsLib_);
        libClose(driverLib_);
    }

    const std::string& name() const { return name_; }
    const std::string& nvrtcPath() const { return nvrtcPath_; }

    std::size_t freeBytes()
    {
        std::size_t free = 0, total = 0;
        check("cuMemGetInfo", d_.MemGetInfo(&free, &total));
        return free;
    }

    CUdeviceptr alloc(std::size_t bytes)
    {
        CUdeviceptr p = 0;
        check("cuMemAlloc", d_.MemAlloc(&p, bytes ? bytes : 1));
        return p;
    }

    void free(CUdeviceptr p)
    {
        if (p) d_.MemFree(p);
    }

    void upload(CUdeviceptr dst, const void* src, std::size_t bytes)
    {
        if (bytes) check("cuMemcpyHtoD", d_.MemcpyHtoD(dst, src, bytes));
    }

    void download(void* dst, CUdeviceptr src, std::size_t bytes)
    {
        if (bytes) check("cuMemcpyDtoH", d_.MemcpyDtoH(dst, src, bytes));
    }

    void memset8(CUdeviceptr dst, std::uint8_t value, std::size_t bytes)
    {
        if (bytes) check("cuMemsetD8", d_.MemsetD8(dst, value, bytes));
    }

    // Runs the kernel on a grid of (gx, gy) blocks of (bx, by) threads and waits for it; params is
    // the usual array of pointers to each argument's value.
    void launch(int kernel, unsigned gx, unsigned gy, unsigned bx, unsigned by, void** params)
    {
        if (gx == 0 || gy == 0) return;
        check("cuLaunchKernel", d_.LaunchKernel(fn_[std::size_t(kernel)], gx, gy, 1, bx, by, 1, 0, nullptr, params, nullptr));
        check(kKernelNames[kernel], d_.CtxSynchronize());
    }

private:
    Device() = default;

    void check(const char* what, CUresult rc)
    {
        if (rc == 0) return;
        const char* s = nullptr;
        if (d_.GetErrorString) d_.GetErrorString(rc, &s);
        throw std::runtime_error(std::string(what) + ": CUDA error " + std::to_string(rc) + " (" + (s ? s : "?") + ")");
    }

    bool loadDriver(std::string& why)
    {
        for (const char* name : kDriverNames)
            if ((driverLib_ = libOpen(name))) break;
        if (!driverLib_) {
            why = "the NVIDIA driver library is not loadable";
            return false;
        }
        Lib l = driverLib_;
        return loadSym(l, d_.Init, "cuInit", why) && loadSym(l, d_.DeviceGetCount, "cuDeviceGetCount", why)
            && loadSym(l, d_.DeviceGet, "cuDeviceGet", why) && loadSym(l, d_.DeviceGetName, "cuDeviceGetName", why)
            && loadSym(l, d_.DeviceGetAttribute, "cuDeviceGetAttribute", why)
            && loadSym(l, d_.DeviceGetPCIBusId, "cuDeviceGetPCIBusId", why)
            && loadSym(l, d_.PrimaryCtxRetain, "cuDevicePrimaryCtxRetain", why)
            && loadSym(l, d_.PrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2", why)
            && loadSym(l, d_.CtxSetCurrent, "cuCtxSetCurrent", why)
            && loadSym(l, d_.CtxSynchronize, "cuCtxSynchronize", why)
            && loadSym(l, d_.ModuleLoadData, "cuModuleLoadData", why)
            && loadSym(l, d_.ModuleUnload, "cuModuleUnload", why)
            && loadSym(l, d_.ModuleGetFunction, "cuModuleGetFunction", why)
            && loadSym(l, d_.MemAlloc, "cuMemAlloc_v2", why) && loadSym(l, d_.MemFree, "cuMemFree_v2", why)
            && loadSym(l, d_.MemcpyHtoD, "cuMemcpyHtoD_v2", why)
            && loadSym(l, d_.MemcpyDtoH, "cuMemcpyDtoH_v2", why) && loadSym(l, d_.MemsetD8, "cuMemsetD8_v2", why)
            && loadSym(l, d_.MemGetInfo, "cuMemGetInfo_v2", why)
            && loadSym(l, d_.LaunchKernel, "cuLaunchKernel", why)
            && loadSym(l, d_.GetErrorString, "cuGetErrorString", why);
    }

    bool loadNvrtc(std::string& why)
    {
        const auto candidates = nvrtcCandidates();
        for (const auto& path : candidates) {
#ifdef _WIN32
            // NVRTC loads its builtins library by bare name at compile time, which only finds it
            // when it is already in the process or on the default search path. For a candidate in
            // a directory, load the companion from that directory first (nvrtc64_130_0.dll ->
            // nvrtc-builtins64_130.dll); a bare-name candidate has both on the search path.
            if (const auto slash = path.find_last_of("/\\"); slash != std::string::npos) {
                std::string builtins = path.substr(slash + 1);
                if (builtins.rfind("nvrtc64_", 0) == 0 && builtins.size() > 13) {
                    builtins = "nvrtc-builtins64_" + builtins.substr(8, 3) + ".dll";
                    builtinsLib_ = libOpen(path.substr(0, slash + 1) + builtins);
                }
            }
#endif
            if ((nvrtcLib_ = libOpen(path))) {
                nvrtcPath_ = path;
                break;
            }
            libClose(builtinsLib_);
            builtinsLib_ = nullptr;
        }
        if (!nvrtcLib_) {
            why = std::string("no NVRTC library: looked for ") + kNvrtcNames[0]
                + " in VC_NVRTC_DIR, beside the executable, the CUDA toolkit and the library search path";
            return false;
        }
        Lib l = nvrtcLib_;
        return loadSym(l, n_.CreateProgram, "nvrtcCreateProgram", why)
            && loadSym(l, n_.CompileProgram, "nvrtcCompileProgram", why)
            && loadSym(l, n_.GetProgramLogSize, "nvrtcGetProgramLogSize", why)
            && loadSym(l, n_.GetProgramLog, "nvrtcGetProgramLog", why)
            && loadSym(l, n_.GetCUBINSize, "nvrtcGetCUBINSize", why) && loadSym(l, n_.GetCUBIN, "nvrtcGetCUBIN", why)
            && loadSym(l, n_.GetPTXSize, "nvrtcGetPTXSize", why) && loadSym(l, n_.GetPTX, "nvrtcGetPTX", why)
            && loadSym(l, n_.DestroyProgram, "nvrtcDestroyProgram", why)
            && loadSym(l, n_.GetErrorString, "nvrtcGetErrorString", why);
    }

    // A cubin for the device's own architecture; when this NVRTC predates the device, PTX for
    // its compute capability and the driver's JIT.
    bool compile(int major, int minor, std::string& image, std::string& why)
    {
        const std::string median = "-DVCR_MAX_MEDIAN_LAYERS=" + std::to_string(kMaxMedianLayers);
        for (int pass = 0; pass < 2; pass++) {
            const std::string arch = std::string("--gpu-architecture=") + (pass == 0 ? "sm_" : "compute_")
                + std::to_string(major) + std::to_string(minor);
            // exact IEEE single precision: no contraction, correctly rounded / and sqrt, denormals kept
            const char* opts[] = {arch.c_str(), "--fmad=false", "--prec-div=true", "--prec-sqrt=true",
                                  "--ftz=false", "--std=c++17", median.c_str()};
            nvrtcProgram prog = nullptr;
            nvrtcResult rc = n_.CreateProgram(&prog, kKernelSource, "vc_render_tifxyz_kernels.cu", 0, nullptr, nullptr);
            if (rc != 0) {
                why = std::string("nvrtcCreateProgram: ") + n_.GetErrorString(rc);
                return false;
            }
            rc = n_.CompileProgram(prog, int(std::size(opts)), opts);
            if (rc != 0) {
                std::size_t logSize = 0;
                n_.GetProgramLogSize(prog, &logSize);
                std::string log(logSize + 1, '\0');
                n_.GetProgramLog(prog, log.data());
                log.resize(std::strlen(log.c_str()));
                why = std::string("NVRTC ") + n_.GetErrorString(rc) + " (" + arch + "): " + log.substr(0, 1500);
                n_.DestroyProgram(&prog);
                continue;
            }
            std::size_t size = 0;
            rc = pass == 0 ? n_.GetCUBINSize(prog, &size) : n_.GetPTXSize(prog, &size);
            if (rc == 0 && size > 0) {
                image.assign(size, '\0');
                rc = pass == 0 ? n_.GetCUBIN(prog, image.data()) : n_.GetPTX(prog, image.data());
            }
            n_.DestroyProgram(&prog);
            if (rc == 0 && !image.empty()) return true;
            why = std::string("NVRTC: no ") + (pass == 0 ? "cubin" : "PTX") + " (" + n_.GetErrorString(rc) + ")";
            image.clear();
        }
        return false;
    }

    bool init(std::string& why)
    {
        if (!loadDriver(why) || !loadNvrtc(why)) return false;
        CUresult rc = d_.Init(0);
        if (rc != 0) {
            why = describe("cuInit", rc);
            return false;
        }
        int count = 0;
        if ((rc = d_.DeviceGetCount(&count)) != 0 || count < 1) {
            why = "no CUDA device is visible to this process (CUDA_VISIBLE_DEVICES?)";
            return false;
        }
        if ((rc = d_.DeviceGet(&dev_, 0)) != 0) {
            why = describe("cuDeviceGet", rc);
            return false;
        }
        int major = 0, minor = 0;
        d_.DeviceGetAttribute(&major, kAttrComputeCapabilityMajor, dev_);
        d_.DeviceGetAttribute(&minor, kAttrComputeCapabilityMinor, dev_);
        char devName[256] = "?", bus[64] = "?";
        d_.DeviceGetName(devName, int(sizeof devName), dev_);
        d_.DeviceGetPCIBusId(bus, int(sizeof bus), dev_);
        name_ = std::string(devName) + " (sm_" + std::to_string(major) + std::to_string(minor) + ", pci " + bus + ")";
        if ((rc = d_.PrimaryCtxRetain(&ctx_, dev_)) != 0) {
            why = describe("cuDevicePrimaryCtxRetain", rc);
            ctx_ = nullptr;
            return false;
        }
        if ((rc = d_.CtxSetCurrent(ctx_)) != 0) {
            why = describe("cuCtxSetCurrent", rc);
            return false;
        }
        std::string image;
        if (!compile(major, minor, image, why)) return false;
        if ((rc = d_.ModuleLoadData(&mod_, image.data())) != 0) {
            why = describe("cuModuleLoadData", rc);
            mod_ = nullptr;
            return false;
        }
        for (const char* kernel : kKernelNames) {
            CUfunction f = nullptr;
            if ((rc = d_.ModuleGetFunction(&f, mod_, kernel)) != 0) {
                why = describe(kernel, rc);
                return false;
            }
            fn_.push_back(f);
        }
        return true;
    }

    std::string describe(const char* what, CUresult rc)
    {
        const char* s = nullptr;
        if (d_.GetErrorString) d_.GetErrorString(rc, &s);
        return std::string(what) + ": CUDA error " + std::to_string(rc) + " (" + (s ? s : "?") + ")";
    }

    Lib driverLib_ = nullptr;
    Lib nvrtcLib_ = nullptr;
    Lib builtinsLib_ = nullptr;
    DriverApi d_;
    NvrtcApi n_;
    CUdevice dev_ = 0;
    CUcontext ctx_ = nullptr;
    CUmodule mod_ = nullptr;
    std::vector<CUfunction> fn_;
    std::string name_;
    std::string nvrtcPath_;
};

double secondsSince(const std::chrono::steady_clock::time_point& t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int compositeMode(const std::string& method)
{
    if (method == "max") return 0;
    if (method == "min") return 1;
    if (method == "mean") return 2;
    if (method == "median") return 3;
    return -1;
}

}  // namespace

// ============================================================
// The sampler
// ============================================================

struct GpuSampler::Impl {
    IChunkedArray& array;
    const int level;
    Logger log;
    std::unique_ptr<Device> dev;
    int contract = 0;
    ChunkDtype dtype = ChunkDtype::UInt8;
    Vol vol;
    std::size_t ntab = 0;        // chunks in the level's grid
    std::size_t chunkBytes = 0;
    std::size_t nSlots = 0;      // chunks the pool holds

    // Device memory: the pool, the chunk -> slot table, the per-chunk mark, and per-band scratch.
    CUdeviceptr dPool = 0, dTab = 0, dMask = 0, dBase = 0, dDirs = 0, dOffs = 0, dOut = 0;
    std::size_t capBase = 0, capDirs = 0, capOffs = 0, capOut = 0;

    // Residency: tab mirrors dTab; a chunk the cache reports as not stored is never asked for
    // again; slots are handed out in order, then the least recently used one not needed by the
    // current band is replaced.
    std::vector<int> tab;
    std::vector<std::uint8_t> absent;
    std::vector<long long> slotChunk;
    std::vector<std::uint64_t> slotUse;
    std::uint64_t clock = 0;
    std::size_t freeCursor = 0;
    std::vector<std::size_t> dirtyTab;
    bool tabWhole = true;
    std::vector<std::uint8_t> mask;
    std::vector<std::uint8_t> staging;
    Stats stats;

    Impl(IChunkedArray& a, int lvl, Logger l) : array(a), level(lvl), log(std::move(l)) {}

    ~Impl()
    {
        if (!dev) return;
        for (CUdeviceptr p : {dPool, dTab, dMask, dBase, dDirs, dOffs, dOut}) dev->free(p);
    }

    void say(const std::string& line)
    {
        if (log) log(line);
    }

    bool init(std::size_t poolBytes, std::string& why)
    {
        const auto shape = array.shape(level);
        const auto cshape = array.chunkShape(level);
        for (int k = 0; k < 3; k++) {
            if (shape[k] <= 0 || cshape[k] <= 0) {
                why = "the volume level has no extent";
                return false;
            }
        }
        dtype = array.dtype();
        vol.sz = shape[0]; vol.sy = shape[1]; vol.sx = shape[2];
        vol.csz = cshape[0]; vol.csy = cshape[1]; vol.csx = cshape[2];
        const long long ngz = (shape[0] + cshape[0] - 1) / cshape[0];
        const long long ngy = (shape[1] + cshape[1] - 1) / cshape[1];
        const long long ngx = (shape[2] + cshape[2] - 1) / cshape[2];
        vol.ngy = int(ngy);
        vol.ngx = int(ngx);
        vol.chunkElems = (long long)cshape[0] * cshape[1] * cshape[2];
        const long long grid = ngz * ngy * ngx;
        if (grid > (1LL << 30) || vol.chunkElems > (1LL << 31)) {
            why = "the chunk grid is too large for the device table";
            return false;
        }
        ntab = std::size_t(grid);
        chunkBytes = std::size_t(vol.chunkElems) * (dtype == ChunkDtype::UInt16 ? 2 : 1);

        dev = Device::open(why);
        if (!dev) return false;
        contract = hostContractsMulAdd() ? 1 : 0;

        std::size_t freeBytes = 0;
        try {
            freeBytes = dev->freeBytes();
            if (poolBytes == 0) poolBytes = freeBytes / 2;
            const std::size_t headroom = std::size_t(512) << 20;  // the band scratch and the tables
            if (freeBytes <= headroom) {
                why = "the device has no free memory";
                return false;
            }
            poolBytes = std::min(poolBytes, freeBytes - headroom);
            nSlots = std::min(poolBytes / chunkBytes, ntab);
            // A pool is useful from a few dozen chunks up; a volume with fewer chunks than that
            // is held whole.
            if (nSlots < std::min<std::size_t>(64, ntab)) {
                why = "the chunk pool would hold fewer than 64 chunks (" + std::to_string(poolBytes >> 20) + " MB for "
                    + std::to_string(chunkBytes >> 10) + " KB chunks; raise --gpu-cache-gb)";
                return false;
            }
            dPool = dev->alloc(nSlots * chunkBytes);
            dTab = dev->alloc(ntab * sizeof(int));
            dMask = dev->alloc(ntab);
        } catch (const std::exception& e) {
            why = e.what();
            return false;
        }
        tab.assign(ntab, -1);
        absent.assign(ntab, 0);
        slotChunk.assign(nSlots, -1);
        slotUse.assign(nSlots, 0);
        mask.resize(ntab);
        char line[512];
        std::snprintf(line, sizeof line,
                      "GPU: %s; NVRTC %s; chunk pool %zu x %dx%dx%d %s = %.1f GB of %.1f GB free; positions %s",
                      dev->name().c_str(), dev->nvrtcPath().c_str(), nSlots, cshape[0], cshape[1], cshape[2],
                      dtype == ChunkDtype::UInt16 ? "uint16" : "uint8", double(nSlots * chunkBytes) / (1 << 30),
                      double(freeBytes) / (1 << 30), contract ? "fused like the host" : "rounded twice like the host");
        say(line);
        return true;
    }

    void ensure(CUdeviceptr& p, std::size_t& cap, std::size_t bytes)
    {
        if (bytes <= cap) return;
        dev->free(p);
        p = 0;
        cap = 0;
        p = dev->alloc(bytes);
        cap = bytes;
    }

    // Upload a band's geometry; the Mats are x, y, z triples row by row.
    void uploadGeometry(const cv::Mat_<cv::Vec3f>& base, const cv::Mat_<cv::Vec3f>& dirs)
    {
        if (base.size() != dirs.size()) throw std::invalid_argument("GpuSampler: base and dirs differ in size");
        const std::size_t bytes = std::size_t(base.rows) * std::size_t(base.cols) * sizeof(cv::Vec3f);
        ensure(dBase, capBase, bytes);
        ensure(dDirs, capDirs, bytes);
        const cv::Mat_<cv::Vec3f> b = base.isContinuous() ? base : base.clone();
        const cv::Mat_<cv::Vec3f> d = dirs.isContinuous() ? dirs : dirs.clone();
        dev->upload(dBase, b.ptr<float>(0), bytes);
        dev->upload(dDirs, d.ptr<float>(0), bytes);
    }

    // The chunks the samples of columns [c0, c1) can touch, as table indices, less those known
    // not to be stored.
    std::vector<long long> markSpan(int w, int h, int c0, int c1, float omin, float omax)
    {
        dev->memset8(dMask, 0, ntab);
        void* params[] = {&dBase, &dDirs, &w, &h, &c0, &c1, &omin, &omax, &vol, &contract, &dMask};
        dev->launch(kMark, unsigned(c1 - c0 + int(kBlockX) - 1) / kBlockX, (unsigned(h) + kBlockY - 1) / kBlockY,
                    kBlockX, kBlockY, params);
        dev->download(mask.data(), dMask, ntab);
        std::vector<long long> needed;
        for (std::size_t i = 0; i < ntab; i++)
            if (mask[i] && !absent[i]) needed.push_back(static_cast<long long>(i));
        return needed;
    }

    int acquireSlot()
    {
        if (freeCursor < nSlots) return int(freeCursor++);
        int best = -1;
        std::uint64_t bestUse = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t s = 0; s < nSlots; s++) {
            if (slotUse[s] < clock && slotUse[s] < bestUse) {
                bestUse = slotUse[s];
                best = int(s);
            }
        }
        if (best < 0) throw std::runtime_error("GpuSampler: the chunk pool is exhausted");
        return best;
    }

    ChunkKey keyOf(long long idx) const
    {
        ChunkKey key;
        key.level = level;
        key.ix = int(idx % vol.ngx);
        key.iy = int((idx / vol.ngx) % vol.ngy);
        key.iz = int(idx / ((long long)vol.ngx * vol.ngy));
        return key;
    }

    // Fetch the needed chunks the pool does not hold through the chunk cache and upload them.
    void makeResident(const std::vector<long long>& needed)
    {
        clock++;
        std::vector<long long> missing;
        for (long long idx : needed) {
            const int s = tab[std::size_t(idx)];
            if (s >= 0) slotUse[std::size_t(s)] = clock;
            else missing.push_back(idx);
        }
        if (missing.empty()) return;

        std::vector<ChunkKey> keys;
        keys.reserve(missing.size());
        for (long long idx : missing) keys.push_back(keyOf(idx));
        auto t0 = std::chrono::steady_clock::now();
        array.prefetchChunks(keys, false);
        std::vector<ChunkResult> results(missing.size());
        std::exception_ptr failure;
        std::atomic<bool> failed{false};
        const int n = int(missing.size());
        #pragma omp parallel for schedule(dynamic, 4)
        for (int i = 0; i < n; i++) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                const ChunkKey& k = keys[std::size_t(i)];
                results[std::size_t(i)] = array.getChunkBlocking(level, k.iz, k.iy, k.ix);
            } catch (...) {
                #pragma omp critical(gpu_sampler_fetch_error)
                {
                    if (!failure) failure = std::current_exception();
                }
                failed.store(true, std::memory_order_relaxed);
            }
        }
        if (failure) std::rethrow_exception(failure);
        stats.fetchSeconds += secondsSince(t0);

        t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < missing.size(); i++) {
            const ChunkResult& r = results[i];
            const std::size_t idx = std::size_t(missing[i]);
            if (r.status == ChunkStatus::Error)
                throw std::runtime_error(r.error.empty() ? "chunk fetch failed" : r.error);
            if (r.status != ChunkStatus::Data || !r.bytes) {
                absent[idx] = 1;  // all fill value, or not stored: samples read 0, as on the CPU
                continue;
            }
            const int slot = acquireSlot();
            if (slotChunk[std::size_t(slot)] >= 0) {
                const auto old = std::size_t(slotChunk[std::size_t(slot)]);
                tab[old] = -1;
                dirtyTab.push_back(old);
                stats.evictions++;
            }
            const void* src = r.bytes->data();
            if (r.bytes->size() < chunkBytes) {  // a short chunk: the rest is fill
                staging.assign(chunkBytes, 0);
                std::memcpy(staging.data(), src, r.bytes->size());
                src = staging.data();
            }
            dev->upload(dPool + CUdeviceptr(slot) * chunkBytes, src, chunkBytes);
            slotChunk[std::size_t(slot)] = static_cast<long long>(idx);
            slotUse[std::size_t(slot)] = clock;
            tab[idx] = slot;
            dirtyTab.push_back(idx);
            stats.chunksUploaded++;
            stats.bytesUploaded += chunkBytes;
        }
        stats.uploadSeconds += secondsSince(t0);
    }

    void uploadTable()
    {
        if (tabWhole || dirtyTab.size() * 16 > ntab) {
            dev->upload(dTab, tab.data(), ntab * sizeof(int));
            tabWhole = false;
        } else {
            for (std::size_t idx : dirtyTab) dev->upload(dTab + CUdeviceptr(idx) * sizeof(int), &tab[idx], sizeof(int));
        }
        dirtyTab.clear();
    }

    // Run `launch(c0, c1)` over the band's columns with every chunk its samples can touch resident,
    // splitting the band when it touches more chunks than the pool holds.
    template <typename Launch>
    void runSpans(int w, int h, float omin, float omax, Launch&& launch)
    {
        std::vector<std::pair<int, int>> todo{{0, w}};
        while (!todo.empty()) {
            const auto [c0, c1] = todo.back();
            todo.pop_back();
            auto t0 = std::chrono::steady_clock::now();
            const std::vector<long long> needed = markSpan(w, h, c0, c1, omin, omax);
            stats.deviceSeconds += secondsSince(t0);
            if (needed.size() > nSlots) {
                if (c1 - c0 <= 1)
                    throw std::runtime_error("GpuSampler: one column of the band touches more chunks than the pool holds "
                                             "(raise --gpu-cache-gb)");
                const int mid = c0 + (c1 - c0) / 2;
                todo.emplace_back(mid, c1);
                todo.emplace_back(c0, mid);
                continue;
            }
            makeResident(needed);
            t0 = std::chrono::steady_clock::now();
            uploadTable();
            launch(c0, c1);
            stats.deviceSeconds += secondsSince(t0);
            stats.passes++;
        }
    }

    template <typename T>
    void sampleSlices(const cv::Mat_<cv::Vec3f>& base, const cv::Mat_<cv::Vec3f>& dirs,
                      const std::vector<float>& offsets, std::vector<cv::Mat_<T>>& out)
    {
        const bool u16 = std::is_same_v<T, std::uint16_t>;
        if ((dtype == ChunkDtype::UInt16) != u16)
            throw std::invalid_argument("GpuSampler: the sample type does not match the volume's");
        const int h = base.rows, w = base.cols, n = int(offsets.size());
        out.assign(std::size_t(n), cv::Mat_<T>());
        for (auto& m : out) m.create(h, w);
        if (h <= 0 || w <= 0 || n <= 0) return;
        if (n > INT_MAX / 2) throw std::invalid_argument("GpuSampler: too many offsets");

        auto t0 = std::chrono::steady_clock::now();
        uploadGeometry(base, dirs);
        ensure(dOffs, capOffs, std::size_t(n) * sizeof(float));
        dev->upload(dOffs, offsets.data(), std::size_t(n) * sizeof(float));
        const std::size_t pixels = std::size_t(h) * std::size_t(w);
        ensure(dOut, capOut, pixels * std::size_t(n) * sizeof(T));
        stats.deviceSeconds += secondsSince(t0);
        const auto [lo, hi] = std::minmax_element(offsets.begin(), offsets.end());
        const float omin = *lo, omax = *hi;
        int nn = n;
        runSpans(w, h, omin, omax, [&](int c0, int c1) {
            int cc0 = c0, cc1 = c1, ww = w, hh = h;
            void* params[] = {&dBase, &dDirs, &ww, &hh, &cc0, &cc1, &dOffs, &nn, &dTab, &dPool, &vol, &contract, &dOut};
            dev->launch(u16 ? kSlicesU16 : kSlicesU8, unsigned(c1 - c0 + int(kBlockX) - 1) / kBlockX,
                        (unsigned(h) + kBlockY - 1) / kBlockY, kBlockX, kBlockY, params);
        });
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; i++) {
            cv::Mat_<T>& m = out[std::size_t(i)];
            if (!m.isContinuous()) m = cv::Mat_<T>(h, w);
            dev->download(m.template ptr<T>(0), dOut + CUdeviceptr(std::size_t(i) * pixels * sizeof(T)), pixels * sizeof(T));
        }
        stats.deviceSeconds += secondsSince(t0);
        stats.bands++;
    }

    void composite(const cv::Mat_<cv::Vec3f>& base, const cv::Mat_<cv::Vec3f>& dirs, float zStep, int zStart,
                   int zEnd, const CompositeParams& params, cv::Mat_<std::uint8_t>& out)
    {
        std::string why;
        if (!compositeSupported(params, zEnd - zStart + 1, &why)) throw std::invalid_argument("GpuSampler: " + why);
        if (dtype != ChunkDtype::UInt8) throw std::invalid_argument("GpuSampler: composites need a uint8 volume");
        if (out.size() != base.size()) throw std::invalid_argument("GpuSampler: the composite output must be base's size");
        const int h = base.rows, w = base.cols, numLayers = zEnd - zStart + 1;
        if (h <= 0 || w <= 0 || numLayers <= 0) return;
        const int mode = compositeMode(params.method);
        float isoCutoff = float(params.isoCutoff);

        auto t0 = std::chrono::steady_clock::now();
        uploadGeometry(base, dirs);
        const std::size_t pixels = std::size_t(h) * std::size_t(w);
        ensure(dOut, capOut, pixels);
        cv::Mat_<std::uint8_t> o = out.isContinuous() ? out : out.clone();
        dev->upload(dOut, o.ptr<std::uint8_t>(0), pixels);  // pixels the kernel leaves alone keep their value
        stats.deviceSeconds += secondsSince(t0);
        const float a = float(zStart) * zStep, b = float(zEnd) * zStep;
        runSpans(w, h, std::min(a, b), std::max(a, b), [&](int c0, int c1) {
            int cc0 = c0, cc1 = c1, ww = w, hh = h, zs = zStart, nl = numLayers, md = mode;
            float step = zStep;
            const unsigned gx = unsigned(c1 - c0 + int(kBlockX) - 1) / kBlockX, gy = (unsigned(h) + kBlockY - 1) / kBlockY;
            if (mode == 3) {
                void* p[] = {&dBase, &dDirs, &ww, &hh, &cc0, &cc1, &step, &zs, &nl, &dTab, &dPool, &vol, &contract,
                             &isoCutoff, &dOut};
                dev->launch(kMedianU8, gx, gy, kBlockX, kBlockY, p);
            } else {
                void* p[] = {&dBase, &dDirs, &ww, &hh, &cc0, &cc1, &step, &zs, &nl, &dTab, &dPool, &vol, &contract,
                             &md, &isoCutoff, &dOut};
                dev->launch(kCompositeU8, gx, gy, kBlockX, kBlockY, p);
            }
        });
        t0 = std::chrono::steady_clock::now();
        dev->download(o.ptr<std::uint8_t>(0), dOut, pixels);
        if (o.data != out.data) o.copyTo(out);
        stats.deviceSeconds += secondsSince(t0);
        stats.bands++;
    }
};

GpuSampler::GpuSampler(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GpuSampler::~GpuSampler() = default;

std::unique_ptr<GpuSampler> GpuSampler::open(IChunkedArray& array, int level, std::size_t poolBytes, Logger log,
                                             std::string& why)
{
    auto impl = std::make_unique<Impl>(array, level, std::move(log));
    if (!impl->init(poolBytes, why)) return nullptr;
    return std::unique_ptr<GpuSampler>(new GpuSampler(std::move(impl)));
}

const std::string& GpuSampler::device() const { return impl_->dev->name(); }
std::size_t GpuSampler::poolChunks() const { return impl_->nSlots; }
const Stats& GpuSampler::stats() const { return impl_->stats; }

void GpuSampler::sampleSlices(const cv::Mat_<cv::Vec3f>& base, const cv::Mat_<cv::Vec3f>& dirs,
                              const std::vector<float>& offsets, std::vector<cv::Mat_<std::uint8_t>>& out)
{
    impl_->sampleSlices<std::uint8_t>(base, dirs, offsets, out);
}

void GpuSampler::sampleSlices(const cv::Mat_<cv::Vec3f>& base, const cv::Mat_<cv::Vec3f>& dirs,
                              const std::vector<float>& offsets, std::vector<cv::Mat_<std::uint16_t>>& out)
{
    impl_->sampleSlices<std::uint16_t>(base, dirs, offsets, out);
}

bool GpuSampler::compositeSupported(const CompositeParams& params, int numLayers, std::string* why)
{
    const int mode = compositeMode(params.method);
    if (mode < 0) {
        if (why) *why = "the " + params.method + " composite is not available on the GPU";
        return false;
    }
    if (mode == 3 && numLayers > kMaxMedianLayers) {
        if (why) *why = "the GPU median composite takes at most " + std::to_string(kMaxMedianLayers) + " layers";
        return false;
    }
    return true;
}

void GpuSampler::composite(const cv::Mat_<cv::Vec3f>& base, const cv::Mat_<cv::Vec3f>& dirs, float zStep, int zStart,
                           int zEnd, const CompositeParams& params, cv::Mat_<std::uint8_t>& out)
{
    impl_->composite(base, dirs, zStep, zStart, zEnd, params, out);
}

std::string GpuSampler::summary() const
{
    const Stats& s = impl_->stats;
    char line[512];
    std::snprintf(line, sizeof line,
                  "GPU: %llu bands in %llu passes; %llu chunks uploaded (%.2f GB), %llu replaced; "
                  "chunk fetch %.1fs, upload %.1fs, device %.1fs",
                  static_cast<unsigned long long>(s.bands), static_cast<unsigned long long>(s.passes),
                  static_cast<unsigned long long>(s.chunksUploaded), double(s.bytesUploaded) / (1 << 30),
                  static_cast<unsigned long long>(s.evictions), s.fetchSeconds, s.uploadSeconds, s.deviceSeconds);
    return line;
}

}  // namespace vc::render::cuda
