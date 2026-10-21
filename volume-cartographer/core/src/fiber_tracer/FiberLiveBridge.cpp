#include "vc/fiber_tracer/FiberLiveBridge.hpp"
#include <condition_variable>
#include <atomic>
#include <map>
#include <set>
#include <mutex>
#include <thread>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <cmath>

#if defined(__unix__) || defined(__APPLE__)
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#define VC_LIVE_SOCKET_SUPPORTED 1
#endif

namespace vc::fiber_tracer {

class FiberLiveBridge::Impl {
public:
    struct Waiter { bool done = false; };
    std::mutex mutex, sendMutex;
    std::condition_variable changed;
    std::map<size_t, std::shared_ptr<Waiter>> waiting;
    using ChunkKey = std::pair<std::string, std::array<uint32_t, 3>>;
    std::set<ChunkKey> ready;
    std::string error;
    std::atomic<bool> cancelled{false};
    size_t next = 1;
    std::chrono::seconds timeout{600};
    bool heartbeatWait = false;
    std::chrono::steady_clock::time_point lastHeartbeat = std::chrono::steady_clock::now();
    int fd = -1;
    std::thread reader;

    explicit Impl(const nlohmann::json& endpoint)
    {
#ifdef VC_LIVE_SOCKET_SUPPORTED
        const int port = endpoint.at("port").get<int>();
        const auto token = endpoint.at("token").get<std::string>();
        const int seconds = endpoint.value("timeout_seconds", 600);
        if (port < 1 || port > 65535 || token.empty() || seconds < 1)
            throw std::invalid_argument("invalid live inference endpoint");
        timeout = std::chrono::seconds(seconds);
        heartbeatWait = endpoint.value("heartbeat_wait", false);
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error("cannot open live control socket");
        timeval sendTimeout{seconds, 0};
        if (::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &sendTimeout, sizeof(sendTimeout)) != 0) {
            ::close(fd);
            fd = -1;
            throw std::runtime_error("cannot set live control send timeout");
        }
#ifdef SO_NOSIGPIPE
        int on = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on));
#endif
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<uint16_t>(port));
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
            ::close(fd);
            fd = -1;
            throw std::runtime_error("cannot connect to localhost inference coordinator");
        }
        try {
            send({{"version", 1}, {"type", "hello"}, {"token", token}});
            reader = std::thread([this] { read(); });
        } catch (...) {
            ::close(fd);
            fd = -1;
            throw;
        }
#else
        (void)endpoint;
        throw std::runtime_error("live inference control transport requires Ubuntu/macOS sockets");
#endif
    }

    ~Impl()
    {
        cancel("native evaluator closed");
        if (reader.joinable()) reader.join();
#ifdef VC_LIVE_SOCKET_SUPPORTED
        if (fd >= 0) ::close(fd);
#endif
    }

    void cancel(const std::string& reason)
    {
        {
            std::lock_guard lock(mutex);
            if (error.empty()) error = reason.empty() ? "live inference cancelled" : reason;
            cancelled.store(true, std::memory_order_release);
        }
        changed.notify_all();
#ifdef VC_LIVE_SOCKET_SUPPORTED
        if (fd >= 0) ::shutdown(fd, SHUT_RDWR);
#endif
    }

    void checkActive()
    {
        if (!cancelled.load(std::memory_order_acquire)) return;
        std::lock_guard lock(mutex);
        throw std::runtime_error(error);
    }

    std::unique_lock<std::mutex> awaitLock(FiberReadinessProfile* profile)
    {
        if (!profile) return std::unique_lock<std::mutex>(mutex);
        const auto start = std::chrono::steady_clock::now();
        std::unique_lock lock(mutex);
        profile->bridgeLockSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        return lock;
    }

    void send(const nlohmann::json& message)
    {
#ifdef VC_LIVE_SOCKET_SUPPORTED
        const std::string bytes = message.dump() + "\n";
        std::lock_guard lock(sendMutex);
        size_t offset = 0;
        while (offset < bytes.size()) {
            int flags = 0;
#ifdef MSG_NOSIGNAL
            flags = MSG_NOSIGNAL;
#endif
            const auto count = ::send(fd, bytes.data() + offset, bytes.size() - offset, flags);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("live control send failed");
            offset += static_cast<size_t>(count);
        }
#else
        (void)message;
#endif
    }

    void read()
    {
#ifdef VC_LIVE_SOCKET_SUPPORTED
        try {
            std::string pending;
            char buffer[8192];
            while (true) {
                const auto count = ::recv(fd, buffer, sizeof(buffer), 0);
                if (count < 0 && errno == EINTR) continue;
                if (count <= 0) throw std::runtime_error("inference coordinator disconnected");
                pending.append(buffer, static_cast<size_t>(count));
                for (auto newline = pending.find('\n'); newline != std::string::npos; newline = pending.find('\n')) {
                    if (newline > (1u << 20)) throw std::runtime_error("oversized live control response");
                    const auto value = nlohmann::json::parse(pending.substr(0, newline));
                    pending.erase(0, newline + 1);
                    if (value.at("version") != 1) throw std::runtime_error("unsupported live response version");
                    if (value.at("type") == "error")
                        throw std::runtime_error(value.at("message").get<std::string>());
                    if (value.at("type") == "heartbeat" && heartbeatWait) {
                        std::lock_guard lock(mutex);
                        lastHeartbeat = std::chrono::steady_clock::now();
                        changed.notify_all();
                        continue; // Liveness only: no request is marked ready.
                    }
                    if (value.at("type") != "ready") throw std::runtime_error("invalid live response type");
                    std::lock_guard lock(mutex);
                    const auto found = waiting.find(value.at("request_id").get<size_t>());
                    if (found == waiting.end()) throw std::runtime_error("unknown live request acknowledgement");
                    found->second->done = true;
                    changed.notify_all();
                }
                if (pending.size() > (1u << 20)) throw std::runtime_error("oversized live control response");
            }
        } catch (const std::exception& exception) { cancel(exception.what()); }
#endif
    }

    void await(const std::string& source, size_t caseIndex, double remaining,
               const std::vector<vc::fiber_tracer::FiberPredictionChunkDependency>& chunks,
               FiberReadinessProfile* profile)
    {
        checkActive();
        if (!std::isfinite(remaining) || remaining < 0 || source.empty())
            throw std::invalid_argument("live request needs a source and nonnegative finite remaining distance");
        if (chunks.empty()) return;
        auto state = std::make_shared<Waiter>();
        size_t request;
        std::vector<ChunkKey> missing;
        {
            auto lock = awaitLock(profile);
            if (!error.empty()) throw std::runtime_error(error);
            for (const auto& chunk : chunks) {
                ChunkKey key{chunk.arrayPath.string(), chunk.indexZYX};
                if (!ready.contains(key)) missing.push_back(std::move(key));
            }
            if (missing.empty()) return;
            request = next++;
            waiting.emplace(request, state); // registration precedes send: no lost ready event
        }
        nlohmann::json dependencies = nlohmann::json::array();
        for (const auto& [path, index] : missing)
            dependencies.push_back({{"array", path}, {"index_zyx", index}});
        try {
            send({{"version", 1}, {"type", "demand"}, {"request_id", request},
                  {"source", source}, {"case", caseIndex}, {"remaining_base_voxels", remaining},
                  {"chunks", std::move(dependencies)}});
            auto lock = awaitLock(profile);
            const auto waitStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            bool expired = false;
            if (heartbeatWait) {
                while (!state->done && error.empty()) {
                    const auto deadline = lastHeartbeat + timeout;
                    changed.wait_until(lock, deadline);
                    if (!state->done && error.empty() &&
                        std::chrono::steady_clock::now() >= lastHeartbeat + timeout) {
                        expired = true;
                        break;
                    }
                }
            } else {
                expired = !changed.wait_for(lock, timeout, [&] { return state->done || !error.empty(); });
            }
            if (expired) {
                lock.unlock();
                cancel("timed out waiting for inference chunks");
                lock.lock();
            }
            if (profile) profile->readyWaitSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - waitStart).count();
            waiting.erase(request);
            if (!error.empty()) throw std::runtime_error(error);
            // Bound native bookkeeping. Dropping ready knowledge merely asks
            // the coordinator to acknowledge durable chunks again, never reruns
            // inference or changes which files are ready.
            if (ready.size() + missing.size() > 100000) ready.clear();
            if (missing.size() <= 100000) ready.insert(missing.begin(), missing.end());
        } catch (const std::exception& exception) {
            // A simultaneous cancellation may have caused send() to fail.
            // Preserve the first session error consistently for every waiter.
            cancel(exception.what());
            std::lock_guard lock(mutex);
            waiting.erase(request);
            throw std::runtime_error(error);
        }
    }
};

FiberLiveBridge::FiberLiveBridge(const nlohmann::json& endpoint) : impl_(std::make_unique<Impl>(endpoint)) {}
FiberLiveBridge::~FiberLiveBridge() = default;
void FiberLiveBridge::cancel(const std::string& reason) { impl_->cancel(reason); }
void FiberLiveBridge::await(const std::string& source, size_t caseIndex, double remainingBase,
                           const std::vector<vc::fiber_tracer::FiberPredictionChunkDependency>& chunks, FiberReadinessProfile* profile)
{ impl_->await(source, caseIndex, remainingBase, chunks, profile); }
void FiberLiveBridge::checkActive() const { impl_->checkActive(); }

} // namespace vc::fiber_tracer
