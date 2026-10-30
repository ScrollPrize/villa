#pragma once

#include <string>

namespace vc3d {

// One request on a fresh AF_UNIX stream connection to a local service: sends
// `line` plus a newline and returns the reply up to (not including) its first
// newline. Throws std::runtime_error naming `service` when the connection,
// the send or the reply fails, or the reply is empty. Thread-safe: every call
// owns its connection.
std::string socketRequestLine(const std::string& socketPath,
                              const std::string& line,
                              int receiveTimeoutSeconds,
                              const std::string& service);

}  // namespace vc3d
