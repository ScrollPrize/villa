#include "vc/core/util/UnixSocket.hpp"

#include <cstring>

#if defined(_WIN32)
#  include <mutex>
#else
#  include <sys/time.h>
#endif

namespace vc::unixsocket {

Socket connectStream(const std::string& path)
{
#if defined(_WIN32)
    static std::once_flag wsaOnce;
    std::call_once(wsaOnce, [] {
        WSADATA data;
        ::WSAStartup(MAKEWORD(2, 2), &data);
    });
#endif

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        return invalidSocket;
    }
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

#if defined(_WIN32)
    SOCKET sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET) {
        return invalidSocket;
    }
    if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::closesocket(sock);
        return invalidSocket;
    }
    return sock;
#else
    int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        return invalidSocket;
    }
    if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(sock);
        return invalidSocket;
    }
    return sock;
#endif
}

void closeSocket(Socket sock)
{
    if (!isValid(sock)) {
        return;
    }
#if defined(_WIN32)
    ::closesocket(sock);
#else
    ::close(sock);
#endif
}

void setRecvTimeoutSeconds(Socket sock, int seconds)
{
#if defined(_WIN32)
    DWORD ms = static_cast<DWORD>(seconds) * 1000u;
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char*>(&ms), sizeof(ms));
#else
    struct timeval tv{};
    tv.tv_sec = seconds;
    tv.tv_usec = 0;
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}

}  // namespace vc::unixsocket
