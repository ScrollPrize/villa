#pragma once

// AF_UNIX stream-socket compat layer: POSIX everywhere, afunix.h on
// Windows 10 1803+ / Windows 11. Including this header pulls in the
// platform's socket API so call sites can keep using send()/recv()
// directly; descriptors are plain ints on both platforms (Windows socket
// handles fit in 32 bits).

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <afunix.h>
#else
#  include <sys/socket.h>
#  include <sys/un.h>
#  include <unistd.h>
#endif

#include <string>

namespace vc::unixsocket {

// Creates a SOCK_STREAM AF_UNIX socket and connects it to `path`.
// Returns the descriptor, or -1 on failure (including a too-long path).
// Handles WSAStartup on Windows.
int connectStream(const std::string& path);

void closeSocket(int sock);

// SO_RCVTIMEO with the platform's expected argument type.
void setRecvTimeoutSeconds(int sock, int seconds);

}  // namespace vc::unixsocket
