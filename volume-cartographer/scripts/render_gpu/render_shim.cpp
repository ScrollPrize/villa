// render_shim.so: LD_PRELOAD shim for vc_render_tifxyz, used by render_band.sh.
// Captures the per-tile base points + unit step dirs the renderer passes to
// sampleTileSlices<u8>, and pairs them with the L0 chunk index from the next
// writeChunkSkipEmpty on the same OMP thread. Sampling then runs in render_gpu.py.
//   RENDER_SHIM_DIR=dir   (required) per-thread dumps dir/t<tid>.bin + dir/offsets.bin
//   RENDER_SHIM_MODE=dump (default) skip CPU sampling (zeros -> chunks skipped as empty)
//   RENDER_SHIM_MODE=both also run the real CPU sampler (reference zarr + dump in one run)
// The interposed symbols are matched by mangled name. In processes without them
// (nice, bash, wrapper scripts) the shim does nothing; if vc_render_tifxyz's
// signatures change, the dump has no tiles and render_gpu.py fails.
#include <opencv2/core.hpp>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace vc { namespace render { class IChunkedArray; } class VcDataset; }

using SampleFn = void (*)(std::vector<cv::Mat_<uint8_t>>&, vc::render::IChunkedArray*, int,
                          const cv::Mat_<cv::Vec3f>&, const cv::Mat_<cv::Vec3f>&, const std::vector<float>&);
using WriteFn = bool (*)(vc::VcDataset*, size_t, size_t, size_t, const void*, size_t);

static const char* kSample = "_Z16sampleTileSlicesRSt6vectorIN2cv4Mat_IhEESaIS2_EEPN2vc6render13IChunkedArrayEiRKNS1_INS0_3VecIfLi3EEEEESE_RKS_IfSaIfEE";
static const char* kWrite = "_ZN2vc9VcDataset19writeChunkSkipEmptyEmmmPKvm";

static SampleFn g_realSample = nullptr;
static WriteFn g_realWrite = nullptr;
// g_dir's constructor (priority 101) must run before shimInit (102); otherwise the
// static initializer would reset the dump dir after shimInit had set it.
static std::string g_dir __attribute__((init_priority(101)));
static bool g_both = false;
static std::once_flag g_offOnce;

[[noreturn]] static void die(const std::string& msg)
{
    fprintf(stderr, "render_shim: %s\n", msg.c_str());
    abort();
}

__attribute__((constructor(102))) static void shimInit()
{
    g_realSample = (SampleFn)dlsym(RTLD_NEXT, kSample);
    g_realWrite = (WriteFn)dlsym(RTLD_NEXT, kWrite);
    const char* d = getenv("RENDER_SHIM_DIR");
    if (d)
        g_dir = d;
    const char* m = getenv("RENDER_SHIM_MODE");
    g_both = m && !strcmp(m, "both");
}

// Dumps are only read by the same user's render_gpu.py, so create them 0600.
static FILE* openDump(const std::string& path)
{
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0)
        die("cannot open " + path);
    FILE* f = fdopen(fd, "wb");
    if (!f)
        die("cannot open " + path);
    return f;
}

static void writeAll(FILE* f, const void* p, size_t size, size_t n, const std::string& path)
{
    if (fwrite(p, size, n, f) != n)
        die("write failed: " + path);
}

// One dump file per OMP thread. Files of threads that outlive main are only flushed at
// process exit; a truncated file is caught by render_gpu.py's record check.
struct ThreadDump {
    FILE* fp = nullptr;
    std::string path;
    ~ThreadDump()
    {
        if (fp && fclose(fp) != 0)
            die("close failed: " + path);
    }
};

struct Pending { bool have = false; cv::Mat_<cv::Vec3f> base, dirs; };
static thread_local Pending g_pend;
static thread_local ThreadDump g_dump;

void sampleTileSlices(std::vector<cv::Mat_<uint8_t>>& out, vc::render::IChunkedArray* cache, int level,
                      const cv::Mat_<cv::Vec3f>& base, const cv::Mat_<cv::Vec3f>& dirs,
                      const std::vector<float>& offsets)
{
    if (!g_realSample || !g_realWrite)
        die("renderer symbols not found (signature changed?)");
    if (g_dir.empty())
        die("RENDER_SHIM_DIR is not set");
    if (g_pend.have)
        die("sampleTileSlices called twice without a writeChunkSkipEmpty in between");
    std::call_once(g_offOnce, [&] {
        std::string p = g_dir + "/offsets.bin";
        FILE* f = openDump(p);
        int32_t n = int32_t(offsets.size());
        writeAll(f, &n, 4, 1, p);
        writeAll(f, offsets.data(), 4, offsets.size(), p);
        if (fclose(f) != 0)
            die("close failed: " + p);
    });
    g_pend.base = base.clone();
    g_pend.dirs = dirs.clone();
    g_pend.have = true;
    if (g_both) {
        g_realSample(out, cache, level, base, dirs, offsets);
        return;
    }
    out.resize(offsets.size());
    for (auto& m : out) {
        m.create(base.rows, base.cols);
        m.setTo(0);
    }
}

namespace vc {
class VcDataset {
public:
    bool writeChunkSkipEmpty(size_t cz, size_t cy, size_t cx, const void* data, size_t n);
};
bool VcDataset::writeChunkSkipEmpty(size_t cz, size_t cy, size_t cx, const void* data, size_t n)
{
    if (g_pend.have) {
        if (!g_dump.fp) {
            g_dump.path = g_dir + "/t" + std::to_string(long(syscall(SYS_gettid))) + ".bin";
            g_dump.fp = openDump(g_dump.path);
        }
        const auto& p = g_dump.path;
        int32_t hdr[4] = {int32_t(cy), int32_t(cx), g_pend.base.rows, g_pend.base.cols};
        writeAll(g_dump.fp, hdr, 4, 4, p);
        for (int r = 0; r < g_pend.base.rows; r++)
            writeAll(g_dump.fp, g_pend.base.ptr<cv::Vec3f>(r), 12, g_pend.base.cols, p);
        for (int r = 0; r < g_pend.dirs.rows; r++)
            writeAll(g_dump.fp, g_pend.dirs.ptr<cv::Vec3f>(r), 12, g_pend.dirs.cols, p);
        g_pend.have = false;
    }
    return g_realWrite(this, cz, cy, cx, data, n);
}
}
