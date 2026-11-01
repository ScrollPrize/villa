#include "utils/process_shutdown.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace {
// ntdll sets this flag in RtlExitUserProcess, before LdrShutdownProcess runs
// the DLL_PROCESS_DETACH destructors. Resolved at run time: the symbol is
// exported by every ntdll since XP but has no import library in some
// toolchains.
using ShutdownFn = BOOLEAN(NTAPI*)();

ShutdownFn resolve() noexcept
{
    const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (!ntdll)
        return nullptr;
    return reinterpret_cast<ShutdownFn>(
        reinterpret_cast<void*>(GetProcAddress(ntdll, "RtlDllShutdownInProgress")));
}
} // namespace

bool utils::process_shutdown_in_progress() noexcept
{
    static const ShutdownFn fn = resolve();
    return fn != nullptr && fn() != FALSE;
}
#else
bool utils::process_shutdown_in_progress() noexcept { return false; }
#endif
