// <unistd.h> for the Microsoft C runtime headers, which have none. It is on the
// include path only for compilers that use those headers (cmake/NativeTargets.cmake);
// MinGW has its own. It declares what the reader takes from <unistd.h> on Windows.
#pragma once

#include <io.h>       // close, dup, lseek, read, unlink, write
#include <process.h>  // getpid
#include <stddef.h>

typedef int pid_t;
typedef ptrdiff_t ssize_t;

// ftruncate(2): -1 with errno set on failure.
static inline int ftruncate(int fd, long long size) { return _chsize_s(fd, size) ? -1 : 0; }
