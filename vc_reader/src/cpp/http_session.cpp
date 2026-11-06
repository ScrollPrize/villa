#include "vc_reader/http.hpp"
#include "http.hpp"

namespace vcr::http {

struct Session::Impl {
    explicit Impl(const Options& o)
        : client(HttpOptions{o.connections, o.connectTimeoutMs, o.stallTimeoutS, o.retries, {}, o.requestTimeoutMs, o.pathAsIs, o.bypassProxy}) {}
    HttpClient client;
};

Session::Session(Options options) : impl_(std::make_unique<Impl>(options)) {}
Session::~Session() = default;

Response Session::request(const char* method, const std::string& url, std::string_view body, const Headers& headers)
{
    Response result;
    const auto response = impl_->client.exchange(method, url, body.data(), body.size(), headers, result.body);
    result.status = response.status;
    result.contentType = response.content_type;
    return result;
}

} // namespace vcr::http
