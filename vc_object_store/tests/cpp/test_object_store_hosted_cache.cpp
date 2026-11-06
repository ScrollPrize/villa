// Delay an old HTTP response until a transform invalidation has completed.
// The late read must neither return nor cache the invalidated mapping.
#include "TestSupport.hpp"
#include "../../src/HostedSession.hpp"
#include "vc_object_store/server/Server.hpp"
#include <vc_reader/http.hpp>
#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <condition_variable>
#include <future>
#include <iostream>

using namespace vc::objects;
namespace net = boost::asio;
namespace http = boost::beast::http;
using tcp = net::ip::tcp;

namespace {
class PausingProxy {
public:
    PausingProxy(std::string upstream, std::string target)
        : upstream_(std::move(upstream)), target_(std::move(target)),
          acceptor_(io_, {net::ip::make_address("127.0.0.1"), 0}), port_(acceptor_.local_endpoint().port()),
          thread_([this] { serve(); }) {}
    ~PausingProxy()
    {
        release();
        stopping_ = true;
        // Wake the synchronous accept without cancelling sockets in another thread.
        boost::system::error_code ignored;
        tcp::socket wake(io_);
        wake.connect({net::ip::make_address("127.0.0.1"), port_}, ignored);
        thread_.join();
        for (auto& worker : workers_) worker.join();
    }
    std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(port_); }
    void wait()
    {
        std::unique_lock lock(mutex_);
        if (!ready_.wait_for(lock, std::chrono::seconds(10), [&] { return paused_; }))
            throw std::runtime_error("transform read did not reach the response gate");
    }
    void release()
    {
        std::lock_guard lock(mutex_);
        released_ = true;
        ready_.notify_all();
    }
private:
    void serve()
    {
        while (!stopping_) {
            tcp::socket socket(io_);
            acceptor_.accept(socket);
            if (stopping_) break;
            workers_.emplace_back([this, socket = std::move(socket)]() mutable {
                try {
                    boost::beast::flat_buffer buffer;
                    http::request<http::string_body> request;
                    http::read(socket, buffer, request);
                    vcr::http::Options options; options.retries = 0; options.requestTimeoutMs = 10000;
                    vcr::http::Session transport(options);
                    const auto reply = transport.request(std::string(request.method_string()).c_str(),
                        upstream_ + std::string(request.target()), request.body());
                    if (request.target() == target_) {
                        std::unique_lock lock(mutex_);
                        if (!paused_) {
                            paused_ = true;
                            ready_.notify_all();
                            ready_.wait(lock, [&] { return released_; });
                        }
                    }
                    http::response<http::string_body> response{http::status(reply.status), 11};
                    response.body() = reply.body;
                    response.keep_alive(false);
                    response.prepare_payload();
                    http::write(socket, response);
                } catch (...) { /* Closing the socket reports the failure to the client. */ }
            });
        }
    }
    std::string upstream_, target_;
    net::io_context io_;
    tcp::acceptor acceptor_;
    uint16_t port_;
    std::atomic_bool stopping_{false};
    std::mutex mutex_;
    std::condition_variable ready_;
    bool paused_ = false, released_ = false;
    std::vector<std::thread> workers_;
    std::thread thread_;
};

void invalidationDuringRead(bool poll)
{
    TemporaryDirectory directory;
    const auto path = directory.path / "store.vcobj";
    OpenOptions options; options.author = "cache-test";
    ObjectRecord registration; registration.uuid = util::newUuid();
    registration = volumeTransformRecord(registration, "source", "target", VolumeAffine::identity(), "registration");
    {
        auto db = ObjectStore::create(path, options);
        auto txn = db.write(); db.upsertObject(txn, registration); txn.commit();
    }
    server::Server server(path, {}); server.start();
    const auto endpoint = "http://127.0.0.1:" + std::to_string(server.port());
    HostedStore writer(endpoint);
    PausingProxy proxy(endpoint, "/v1/objects/" + registration.uuid);
    HostedSessionOptions hosted; hosted.client.retries = 0; hosted.client.requestTimeoutMs = 10000;
    detail::HostedAttachment attachment(proxy.endpoint(), hosted);
    auto read = std::async(std::launch::async, [&] { return attachment.volumeTransforms(); });
    try {
        proxy.wait();
        auto matrix = VolumeAffine::identity(); matrix.values[3] = 12;
        registration = volumeTransformRecord(registration, "source", "target", matrix, "registration");
        const auto status = writer.status();
        TransactionRequest request;
        request.requestUuid = util::newUuid(); request.storeUuid = status.storeUuid;
        request.generation = status.generation; request.author = "cache-test";
        request.expectedObjects = {{registration.uuid, 1}};
        request.operations = {UpdateObjectOp{objectRecordJson(registration)}};
        if (poll) {
            writer.submit(request);
            attachment.refresh();
        } else attachment.submit(request);
    } catch (...) {
        proxy.release(); read.wait(); throw;
    }
    proxy.release();
    const auto metadata = nlohmann::json{{"coordinates", volumeCoordinates("source", 1)}};
    const auto fetched = read.get().toVolume(metadata, "target");
    require(fetched && fetched->values[3] == 12, "late transform fetch retries after invalidation");
    const auto cached = attachment.volumeTransforms().toVolume(metadata, "target");
    require(cached == fetched, "cache retains the new mapping");
}
}

int main()
try {
    invalidationDuringRead(false);
    invalidationDuringRead(true);
    std::cout << "object_store_hosted_cache: ok\n";
} catch (const std::exception& error) {
    std::cerr << "object_store_hosted_cache: " << error.what() << '\n';
    return 1;
}
