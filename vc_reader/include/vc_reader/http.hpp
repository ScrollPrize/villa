// HTTP requests whose answers the caller interprets (service APIs), over the
// same pooled transport, retry and backoff as vc_reader's chunk reads.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vcr::http {

struct Options {
    int connections = 4;            // concurrent requests (and kept-alive connections)
    long connectTimeoutMs = 10000;
    long stallTimeoutS = 30;        // abort an attempt after this long without receiving data
    int retries = 5;
    long requestTimeoutMs = 0;      // whole-attempt limit; 0 = none
    bool pathAsIs = false;          // preserve dot segments when paths contain opaque identifiers
};

struct Response {
    long status = 0;
    std::string body, contentType;
};

using Headers = std::vector<std::pair<std::string, std::string>>;

// Returns the status and body of every final response: 2xx, 3xx and 4xx
// (except 429) are answers, not errors. Connection failures, stalls, 429 and
// 5xx are retried with exponential backoff and throw once retries run out, so
// send only requests that are safe to repeat: idempotent ones, or ones that
// carry a durable request id the server deduplicates.
class Session {
public:
    explicit Session(Options = {});
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Response request(const char* method, const std::string& url, std::string_view body = {}, const Headers& headers = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vcr::http
