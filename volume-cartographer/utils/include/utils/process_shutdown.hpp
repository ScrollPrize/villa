#pragma once

namespace utils {

// True while the process is exiting, from the point where the OS has already
// terminated every thread but the one running the exit. On Windows that is
// ExitProcess: it kills the other threads, then unloads DLLs, and the static
// destructors of a DLL run during that unload under the loader lock. A thread
// pool destroyed there must not join its workers: they are gone, and the join
// waits forever. Always false where this cannot happen (POSIX).
bool process_shutdown_in_progress() noexcept;

} // namespace utils
