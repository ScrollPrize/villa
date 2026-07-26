#pragma once

// AF_UNIX stream-socket compat layer: POSIX everywhere, afunix.h on
// Windows 10 1803+ / Windows 11. Including this header pulls in the
// platform's socket API so call sites can keep using send()/recv() directly.

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

#if defined(_WIN32)
using Socket = SOCKET;
inline constexpr Socket invalidSocket = INVALID_SOCKET;
#else
using Socket = int;
inline constexpr Socket invalidSocket = -1;
#endif

inline bool isValid(Socket sock) { return sock != invalidSocket; }

// Creates a SOCK_STREAM AF_UNIX socket and connects it to `path`.
// Returns the platform socket handle, or invalidSocket on failure.
// Handles WSAStartup on Windows.
Socket connectStream(const std::string& path);

void closeSocket(Socket sock);

// SO_RCVTIMEO with the platform's expected argument type.
void setRecvTimeoutSeconds(Socket sock, int seconds);

}  // namespace vc::unixsocket
