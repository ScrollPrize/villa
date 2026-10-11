#include "SocketRequest.hpp"

#include "vc/core/util/UnixSocket.hpp"

#include <stdexcept>

namespace vc3d {

std::string socketRequestLine(const std::string& socketPath,
                              const std::string& line,
                              int receiveTimeoutSeconds,
                              const std::string& service)
{
    vc::unixsocket::Socket sock = vc::unixsocket::connectStream(socketPath);
    if (!vc::unixsocket::isValid(sock)) {
        throw std::runtime_error("Failed to connect to the " + service + " socket.");
    }

    const std::string requestPayload = line + "\n";
    const char* ptr = requestPayload.c_str();
    size_t remaining = requestPayload.size();
    while (remaining > 0) {
        auto sent = ::send(sock, ptr, remaining, 0);
        if (sent < 0) {
            vc::unixsocket::closeSocket(sock);
            throw std::runtime_error("Failed to send the " + service + " request.");
        }
        ptr += sent;
        remaining -= static_cast<size_t>(sent);
    }

    vc::unixsocket::setRecvTimeoutSeconds(sock, receiveTimeoutSeconds);

    std::string responsePayload;
    char buf[4096];
    long long n;
    while ((n = ::recv(sock, buf, sizeof(buf), 0)) > 0) {
        for (long long i = 0; i < n; ++i) {
            if (buf[i] == '\n') {
                vc::unixsocket::closeSocket(sock);
                if (responsePayload.empty()) {
                    throw std::runtime_error("The " + service + " service returned an empty response.");
                }
                return responsePayload;
            }
            responsePayload.push_back(buf[i]);
        }
    }
    vc::unixsocket::closeSocket(sock);

    if (responsePayload.empty()) {
        throw std::runtime_error("The " + service + " service returned an empty response.");
    }
    return responsePayload;
}

}  // namespace vc3d
