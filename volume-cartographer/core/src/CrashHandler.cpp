#include "vc/core/util/CrashHandler.hpp"
#include "vc/core/Version.hpp"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <exception>

#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <ucontext.h>
#include <unistd.h>

#include <backtrace.h>
#include <backtrace-supported.h>

namespace {

constexpr int OUT_FD = STDERR_FILENO;

std::atomic<int> g_inHandler{0};
::backtrace_state* g_btState = nullptr;
char g_exePath[4096];

void writeStr(const char* s, std::size_t n)
{
    if (!s) return;
    while (n > 0) {
        const ssize_t w = ::write(OUT_FD, s, n);
        if (w < 0) { if (errno == EINTR) continue; return; }
        s += w;
        n -= static_cast<std::size_t>(w);
    }
}

void writeStr(const char* s) { if (s) writeStr(s, std::strlen(s)); }
void writeChar(char c) { writeStr(&c, 1); }

void writeUDec(unsigned long long v)
{
    char buf[32];
    int i = static_cast<int>(sizeof(buf)) - 1;
    buf[i--] = 0;
    if (v == 0) { writeStr("0"); return; }
    while (v != 0 && i >= 0) {
        buf[i--] = static_cast<char>('0' + (v % 10));
        v /= 10;
    }
    writeStr(buf + i + 1);
}

void writeDec(long long v)
{
    if (v < 0) { writeChar('-'); v = -v; }
    writeUDec(static_cast<unsigned long long>(v));
}

void writeHex(unsigned long long v, int padTo = 0)
{
    char buf[32];
    int i = static_cast<int>(sizeof(buf)) - 1;
    buf[i--] = 0;
    int len = 0;
    if (v == 0) { buf[i--] = '0'; len = 1; }
    while (v != 0 && i >= 0) {
        const unsigned d = v & 0xFu;
        buf[i--] = (d < 10) ? static_cast<char>('0' + d) : static_cast<char>('a' + d - 10);
        v >>= 4;
        ++len;
    }
    while (len < padTo && i >= 0) { buf[i--] = '0'; ++len; }
    writeStr("0x");
    writeStr(buf + i + 1);
}

void dumpFile(const char* path, std::size_t maxBytes = 1024 * 1024)
{
    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        writeStr("(could not open ");
        writeStr(path);
        writeStr(": errno=");
        writeDec(errno);
        writeStr(")\n");
        return;
    }
    char buf[4096];
    std::size_t total = 0;
    while (total < maxBytes) {
        ssize_t r = ::read(fd, buf, sizeof(buf));
        if (r <= 0) break;
        if (total + static_cast<std::size_t>(r) > maxBytes) {
            r = static_cast<ssize_t>(maxBytes - total);
        }
        writeStr(buf, static_cast<std::size_t>(r));
        total += static_cast<std::size_t>(r);
    }
    ::close(fd);
    if (total >= maxBytes) writeStr("\n[...truncated...]\n");
}

const char* signalName(int sig)
{
    switch (sig) {
        case SIGSEGV: return "SIGSEGV (Segmentation fault)";
        case SIGBUS:  return "SIGBUS (Bus error)";
        case SIGFPE:  return "SIGFPE (Floating-point exception)";
        case SIGILL:  return "SIGILL (Illegal instruction)";
        case SIGABRT: return "SIGABRT (Aborted)";
        case SIGTRAP: return "SIGTRAP (Trace/breakpoint trap)";
        case SIGSYS:  return "SIGSYS (Bad system call)";
        case SIGTERM: return "SIGTERM (Terminated)";
        case SIGINT:  return "SIGINT (Interrupt)";
        case SIGHUP:  return "SIGHUP (Hangup)";
        case SIGQUIT: return "SIGQUIT (Quit)";
        case SIGPIPE: return "SIGPIPE (Broken pipe)";
        default: return "(unknown signal)";
    }
}

const char* sigCodeStr(int sig, int code)
{
    switch (sig) {
        case SIGSEGV:
            switch (code) {
                case SEGV_MAPERR: return "SEGV_MAPERR (address not mapped)";
                case SEGV_ACCERR: return "SEGV_ACCERR (invalid permissions for object)";
#ifdef SEGV_BNDERR
                case SEGV_BNDERR: return "SEGV_BNDERR (bounds-check failure)";
#endif
#ifdef SEGV_PKUERR
                case SEGV_PKUERR: return "SEGV_PKUERR (memory protection key)";
#endif
                default: return "(unknown SEGV code)";
            }
        case SIGBUS:
            switch (code) {
                case BUS_ADRALN: return "BUS_ADRALN (invalid address alignment)";
                case BUS_ADRERR: return "BUS_ADRERR (nonexistent physical address)";
                case BUS_OBJERR: return "BUS_OBJERR (object-specific hardware error)";
#ifdef BUS_MCEERR_AR
                case BUS_MCEERR_AR: return "BUS_MCEERR_AR (hardware memory error - action required)";
#endif
#ifdef BUS_MCEERR_AO
                case BUS_MCEERR_AO: return "BUS_MCEERR_AO (hardware memory error - action optional)";
#endif
                default: return "(unknown BUS code)";
            }
        case SIGFPE:
            switch (code) {
                case FPE_INTDIV: return "FPE_INTDIV (integer divide by zero)";
                case FPE_INTOVF: return "FPE_INTOVF (integer overflow)";
                case FPE_FLTDIV: return "FPE_FLTDIV (floating-point divide by zero)";
                case FPE_FLTOVF: return "FPE_FLTOVF (floating-point overflow)";
                case FPE_FLTUND: return "FPE_FLTUND (floating-point underflow)";
                case FPE_FLTRES: return "FPE_FLTRES (floating-point inexact result)";
                case FPE_FLTINV: return "FPE_FLTINV (floating-point invalid operation)";
                case FPE_FLTSUB: return "FPE_FLTSUB (subscript out of range)";
                default: return "(unknown FPE code)";
            }
        case SIGILL:
            switch (code) {
                case ILL_ILLOPC: return "ILL_ILLOPC (illegal opcode)";
                case ILL_ILLOPN: return "ILL_ILLOPN (illegal operand)";
                case ILL_ILLADR: return "ILL_ILLADR (illegal addressing mode)";
                case ILL_ILLTRP: return "ILL_ILLTRP (illegal trap)";
                case ILL_PRVOPC: return "ILL_PRVOPC (privileged opcode)";
                case ILL_PRVREG: return "ILL_PRVREG (privileged register)";
                case ILL_COPROC: return "ILL_COPROC (coprocessor error)";
                case ILL_BADSTK: return "ILL_BADSTK (internal stack error)";
                default: return "(unknown ILL code)";
            }
        case SIGTRAP:
            switch (code) {
                case TRAP_BRKPT: return "TRAP_BRKPT (process breakpoint)";
                case TRAP_TRACE: return "TRAP_TRACE (process trace trap)";
                default: return "(unknown TRAP code)";
            }
    }
    switch (code) {
        case SI_USER:    return "SI_USER (kill/sigsend/raise)";
        case SI_KERNEL:  return "SI_KERNEL (sent by the kernel)";
        case SI_QUEUE:   return "SI_QUEUE (sigqueue)";
        case SI_TIMER:   return "SI_TIMER (timer expired)";
        case SI_TKILL:   return "SI_TKILL (tkill/tgkill)";
        case SI_MESGQ:   return "SI_MESGQ (POSIX message queue)";
        case SI_ASYNCIO: return "SI_ASYNCIO (AIO completion)";
        default: return "(unrecognized si_code)";
    }
}

void dumpRegisters(const ucontext_t* uc)
{
#if defined(__x86_64__)
    const auto& g = uc->uc_mcontext.gregs;
    writeStr("RAX="); writeHex(static_cast<unsigned long long>(g[REG_RAX]), 16);
    writeStr(" RBX="); writeHex(static_cast<unsigned long long>(g[REG_RBX]), 16);
    writeStr(" RCX="); writeHex(static_cast<unsigned long long>(g[REG_RCX]), 16);
    writeStr(" RDX="); writeHex(static_cast<unsigned long long>(g[REG_RDX]), 16);
    writeChar('\n');
    writeStr("RSI="); writeHex(static_cast<unsigned long long>(g[REG_RSI]), 16);
    writeStr(" RDI="); writeHex(static_cast<unsigned long long>(g[REG_RDI]), 16);
    writeStr(" RBP="); writeHex(static_cast<unsigned long long>(g[REG_RBP]), 16);
    writeStr(" RSP="); writeHex(static_cast<unsigned long long>(g[REG_RSP]), 16);
    writeChar('\n');
    writeStr("R8 ="); writeHex(static_cast<unsigned long long>(g[REG_R8]),  16);
    writeStr(" R9 ="); writeHex(static_cast<unsigned long long>(g[REG_R9]),  16);
    writeStr(" R10="); writeHex(static_cast<unsigned long long>(g[REG_R10]), 16);
    writeStr(" R11="); writeHex(static_cast<unsigned long long>(g[REG_R11]), 16);
    writeChar('\n');
    writeStr("R12="); writeHex(static_cast<unsigned long long>(g[REG_R12]), 16);
    writeStr(" R13="); writeHex(static_cast<unsigned long long>(g[REG_R13]), 16);
    writeStr(" R14="); writeHex(static_cast<unsigned long long>(g[REG_R14]), 16);
    writeStr(" R15="); writeHex(static_cast<unsigned long long>(g[REG_R15]), 16);
    writeChar('\n');
    writeStr("RIP="); writeHex(static_cast<unsigned long long>(g[REG_RIP]), 16);
    writeStr(" EFL="); writeHex(static_cast<unsigned long long>(g[REG_EFL]), 16);
    writeStr(" CSGSFS="); writeHex(static_cast<unsigned long long>(g[REG_CSGSFS]), 16);
    writeChar('\n');
    writeStr("ERR="); writeHex(static_cast<unsigned long long>(g[REG_ERR]), 16);
    writeStr(" TRAPNO="); writeHex(static_cast<unsigned long long>(g[REG_TRAPNO]), 16);
    writeStr(" OLDMASK="); writeHex(static_cast<unsigned long long>(g[REG_OLDMASK]), 16);
    writeStr(" CR2="); writeHex(static_cast<unsigned long long>(g[REG_CR2]), 16);
    writeChar('\n');
#elif defined(__aarch64__)
    const auto& mc = uc->uc_mcontext;
    for (int i = 0; i < 31; ++i) {
        if (i > 0 && (i % 4) == 0) writeChar('\n');
        if (i > 0) writeChar(' ');
        writeStr("X");
        if (i < 10) writeChar(' ');
        writeUDec(static_cast<unsigned long long>(i));
        writeChar('=');
        writeHex(static_cast<unsigned long long>(mc.regs[i]), 16);
    }
    writeChar('\n');
    writeStr("SP="); writeHex(static_cast<unsigned long long>(mc.sp), 16);
    writeStr(" PC="); writeHex(static_cast<unsigned long long>(mc.pc), 16);
    writeStr(" PSTATE="); writeHex(static_cast<unsigned long long>(mc.pstate), 16);
    writeStr(" FAULT_ADDR="); writeHex(static_cast<unsigned long long>(mc.fault_address), 16);
    writeChar('\n');
#else
    (void)uc;
    writeStr("(register dump unsupported on this architecture)\n");
#endif
}

struct PcInfoCtx {
    int frame;
    bool resolved;
};

int btPcinfoCallback(void* data, uintptr_t pc, const char* filename, int lineno, const char* function)
{
    auto* ctx = static_cast<PcInfoCtx*>(data);
    if (!ctx->resolved) {
        writeStr("  #");
        if (ctx->frame < 100) writeChar(' ');
        if (ctx->frame < 10) writeChar(' ');
        writeUDec(static_cast<unsigned long long>(ctx->frame));
        writeStr("  ");
        writeHex(static_cast<unsigned long long>(pc), 16);
        writeStr("  ");
        writeStr(function ? function : "??");
        if (filename) {
            writeStr("\n        at ");
            writeStr(filename);
            if (lineno > 0) {
                writeChar(':');
                writeUDec(static_cast<unsigned long long>(lineno));
            }
        }
        writeChar('\n');
        ctx->resolved = true;
    } else {
        writeStr("        inlined: ");
        writeStr(function ? function : "??");
        if (filename) {
            writeStr("  at ");
            writeStr(filename);
            if (lineno > 0) {
                writeChar(':');
                writeUDec(static_cast<unsigned long long>(lineno));
            }
        }
        writeChar('\n');
    }
    return 0;
}

void btErrorCallback(void* /*data*/, const char* msg, int errnum)
{
    writeStr("  [libbacktrace: ");
    writeStr(msg ? msg : "(null)");
    writeStr(" errno=");
    writeDec(errnum);
    writeStr("]\n");
}

void dumpFrame(int frame, void* addr)
{
    const auto pc = reinterpret_cast<uintptr_t>(addr);
    PcInfoCtx ctx{frame, false};
    if (g_btState != nullptr) {
        ::backtrace_pcinfo(g_btState, pc, btPcinfoCallback, btErrorCallback, &ctx);
    }
    if (!ctx.resolved) {
        writeStr("  #");
        if (frame < 100) writeChar(' ');
        if (frame < 10) writeChar(' ');
        writeUDec(static_cast<unsigned long long>(frame));
        writeStr("  ");
        writeHex(static_cast<unsigned long long>(pc), 16);
        writeStr("  ??\n");
    }
}

void dumpStack()
{
    void* addrs[256];
    const int n = ::backtrace(addrs, 256);

    writeStr("Backtrace (");
    writeDec(n);
    writeStr(" frames, most-recent call first):\n");

    if (n <= 0) {
        writeStr("(no frames — execinfo backtrace() returned empty; "
                 "PC may be corrupt)\n");
        return;
    }

    writeStr("--- raw addresses (in case symbol resolution crashes below) ---\n");
    for (int i = 0; i < n; ++i) {
        writeStr("  ");
        writeHex(reinterpret_cast<unsigned long long>(addrs[i]), 16);
        writeChar('\n');
    }

    writeStr("--- resolved frames ---\n");
    for (int i = 0; i < n; ++i) {
        dumpFrame(i, addrs[i]);
    }
}

void dumpManualUnwind(const ucontext_t* uc)
{
#if defined(__aarch64__)
    if (!uc) return;
    writeStr("\n--- manual aarch64 frame-pointer walk (LR + saved FP chain) ---\n");
    const auto pc = static_cast<unsigned long long>(uc->uc_mcontext.pc);
    const auto lr = static_cast<unsigned long long>(uc->uc_mcontext.regs[30]);
    auto fp = static_cast<unsigned long long>(uc->uc_mcontext.regs[29]);
    writeStr("  PC = "); writeHex(pc, 16); writeChar('\n');
    writeStr("  LR = "); writeHex(lr, 16);
    writeStr("   <- likely return address (immediate caller of faulting frame)\n");
    if (lr != 0 && lr != pc) {
        dumpFrame(0, reinterpret_cast<void*>(lr));
    }
    int frame = 1;
    for (int i = 0; i < 64 && fp != 0; ++i) {
        const auto* frameRecord = reinterpret_cast<const unsigned long long*>(fp);
        const unsigned long long savedFp = frameRecord[0];
        const unsigned long long savedLr = frameRecord[1];
        if (savedLr == 0 || savedLr == ~0ULL) break;
        dumpFrame(frame++, reinterpret_cast<void*>(savedLr));
        if (savedFp <= fp) break;
        fp = savedFp;
    }
#elif defined(__x86_64__)
    if (!uc) return;
    writeStr("\n--- manual x86_64 frame-pointer walk ---\n");
    auto rbp = static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_RBP]);
    int frame = 0;
    for (int i = 0; i < 64 && rbp != 0; ++i) {
        const auto* frameRecord = reinterpret_cast<const unsigned long long*>(rbp);
        const unsigned long long savedRbp = frameRecord[0];
        const unsigned long long retAddr  = frameRecord[1];
        if (retAddr == 0 || retAddr == ~0ULL) break;
        dumpFrame(frame++, reinterpret_cast<void*>(retAddr));
        if (savedRbp <= rbp) break;
        rbp = savedRbp;
    }
#else
    (void)uc;
#endif
}

void dumpHeader(int sig, const siginfo_t* si)
{
    writeStr("\n");
    writeStr("================================================================================\n");
    writeStr(" VC3D CRASH REPORT  --  please copy this entire block into your bug report\n");
    writeStr("================================================================================\n");

    writeStr("Version:    ");
    writeStr(ProjectInfo::NameAndVersion().c_str());
    writeStr("\nCommit:     ");
    writeStr(ProjectInfo::RepositoryHash().c_str());
    writeStr("\nBuild:      "
#ifdef NDEBUG
            "NDEBUG"
#else
            "debug"
#endif
            ", compiled "
            __DATE__ " " __TIME__
            ", "
#if defined(__clang__)
            "clang " __clang_version__
#elif defined(__GNUC__)
            "gcc " __VERSION__
#else
            "unknown compiler"
#endif
#if defined(__x86_64__)
            ", x86_64"
#elif defined(__aarch64__)
            ", aarch64"
#endif
            "\n");
    writeStr("Executable: ");
    writeStr(g_exePath[0] ? g_exePath : "(unknown)");
    writeStr("\nPID:        ");
    writeUDec(static_cast<unsigned long long>(::getpid()));
    writeStr("\nTID:        ");
    writeUDec(static_cast<unsigned long long>(::syscall(SYS_gettid)));
    char tname[32] = {};
    if (::prctl(PR_GET_NAME, reinterpret_cast<unsigned long>(tname), 0, 0, 0) == 0) {
        writeStr(" (");
        writeStr(tname);
        writeStr(")");
    }
    writeStr("\nTime:       ");
    writeUDec(static_cast<unsigned long long>(::time(nullptr)));
    writeStr(" (unix epoch)\n");

    writeStr("\n--- SIGNAL ---\n");
    writeStr("signal:   ");
    writeDec(sig);
    writeStr(" (");
    writeStr(signalName(sig));
    writeStr(")\n");
    if (si != nullptr) {
        writeStr("si_code:  ");
        writeDec(si->si_code);
        writeStr(" (");
        writeStr(sigCodeStr(sig, si->si_code));
        writeStr(")\n");
        if (sig == SIGSEGV || sig == SIGBUS || sig == SIGFPE || sig == SIGILL || sig == SIGTRAP) {
            writeStr("si_addr:  ");
            writeHex(reinterpret_cast<unsigned long long>(si->si_addr), 16);
            writeChar('\n');
        }
        writeStr("si_pid:   ");
        writeDec(si->si_pid);
        writeStr("\nsi_uid:   ");
        writeUDec(si->si_uid);
        writeStr("\nsi_errno: ");
        writeDec(si->si_errno);
        writeChar('\n');
    }
}

void dumpResourceUsage()
{
    rusage ru{};
    if (::getrusage(RUSAGE_SELF, &ru) == 0) {
        writeStr("rss_kb:        ");
        writeUDec(static_cast<unsigned long long>(ru.ru_maxrss));
        writeStr("\nuser_time_s:   ");
        writeUDec(static_cast<unsigned long long>(ru.ru_utime.tv_sec));
        writeChar('.');
        writeUDec(static_cast<unsigned long long>(ru.ru_utime.tv_usec));
        writeStr("\nsys_time_s:    ");
        writeUDec(static_cast<unsigned long long>(ru.ru_stime.tv_sec));
        writeChar('.');
        writeUDec(static_cast<unsigned long long>(ru.ru_stime.tv_usec));
        writeStr("\nminor_faults:  ");
        writeUDec(static_cast<unsigned long long>(ru.ru_minflt));
        writeStr("\nmajor_faults:  ");
        writeUDec(static_cast<unsigned long long>(ru.ru_majflt));
        writeStr("\nvol_ctx_sw:    ");
        writeUDec(static_cast<unsigned long long>(ru.ru_nvcsw));
        writeStr("\ninvol_ctx_sw:  ");
        writeUDec(static_cast<unsigned long long>(ru.ru_nivcsw));
        writeChar('\n');
    }
}

void dumpCmdline()
{
    const int fd = ::open("/proc/self/cmdline", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    char buf[4096];
    const ssize_t r = ::read(fd, buf, sizeof(buf) - 1);
    if (r > 0) {
        for (ssize_t i = 0; i < r; ++i) {
            if (buf[i] == '\0') buf[i] = ' ';
        }
        writeStr(buf, static_cast<std::size_t>(r));
        writeChar('\n');
    }
    ::close(fd);
}

void crashHandler(int sig, siginfo_t* si, void* uc_v)
{
    if (g_inHandler.fetch_add(1, std::memory_order_relaxed) > 0) {
        const char msg[] = "\n[recursive crash inside handler — aborting]\n";
        ::write(OUT_FD, msg, sizeof(msg) - 1);
        ::_exit(128 + sig);
    }

    auto* uc = static_cast<ucontext_t*>(uc_v);

    dumpHeader(sig, si);

    writeStr("\n--- REGISTERS ---\n");
    if (uc) dumpRegisters(uc);
    else writeStr("(no ucontext)\n");

    writeStr("\n--- BACKTRACE ---\n");
    dumpStack();
    dumpManualUnwind(uc);

    writeStr("\n--- COMMAND LINE ---\n");
    dumpCmdline();

    writeStr("\n--- RESOURCE USAGE ---\n");
    dumpResourceUsage();

    writeStr("\n--- /proc/self/status ---\n");
    dumpFile("/proc/self/status", 8192);

    writeStr("\n--- /proc/self/limits ---\n");
    dumpFile("/proc/self/limits", 4096);

    writeStr("\n--- /proc/meminfo (system memory at crash) ---\n");
    dumpFile("/proc/meminfo", 4096);

    writeStr("\n--- /proc/self/maps (use with addr2line on raw addresses above) ---\n");
    dumpFile("/proc/self/maps");

    writeStr("\n================================================================================\n");
    writeStr(" END VC3D CRASH REPORT\n");
    writeStr("================================================================================\n");

    struct sigaction sa{};
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    ::sigaction(sig, &sa, nullptr);
    ::raise(sig);
}

void terminateHandler()
{
    writeStr("\n--- std::terminate called ---\n");
    try {
        const std::exception_ptr ep = std::current_exception();
        if (ep) {
            std::rethrow_exception(ep);
        } else {
            writeStr("(no active exception)\n");
        }
    } catch (const std::exception& e) {
        writeStr("Exception type: std::exception subclass\nwhat(): ");
        writeStr(e.what());
        writeChar('\n');
    } catch (...) {
        writeStr("Exception type: unknown (not std::exception)\n");
    }
    ::raise(SIGABRT);
}

}

namespace vc::crash {

void install()
{
    const ssize_t n = ::readlink("/proc/self/exe", g_exePath, sizeof(g_exePath) - 1);
    if (n > 0) g_exePath[n] = '\0';
    else      g_exePath[0] = '\0';

    g_btState = ::backtrace_create_state(g_exePath[0] ? g_exePath : nullptr,
                                         /*threaded*/ 1,
                                         btErrorCallback,
                                         nullptr);

    static char altstack[16 * 1024 * 1024];
    stack_t ss{};
    ss.ss_sp = altstack;
    ss.ss_size = sizeof(altstack);
    ss.ss_flags = 0;
    ::sigaltstack(&ss, nullptr);

    struct sigaction sa{};
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sa.sa_sigaction = crashHandler;

    const int signals[] = { SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT, SIGTRAP, SIGSYS };
    for (const int s : signals) {
        ::sigaction(s, &sa, nullptr);
    }

    std::set_terminate(terminateHandler);
}

}
