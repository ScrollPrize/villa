#pragma once

#include <QHostAddress>
#include <QTcpServer>
#include <QTcpSocket>

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// A persistent loopback origin. Requests on a poisoned connection can keep
// failing, so a test can distinguish a new TCP connection from another GET.
class LoopbackHttpServer {
public:
    struct Request {
        std::string method, path, headers;
        int connection = 0;
    };
    struct Reply {
        int status = 200;
        std::string body;
        bool s3 = true;
    };
    using Handler = std::function<Reply(const Request&)>;

    explicit LoopbackHttpServer(Handler handler)
    {
        std::promise<unsigned short> ready;
        auto port = ready.get_future();
        thread_ = std::thread([this, handler = std::move(handler), ready = std::move(ready)]() mutable {
            try {
                QTcpServer server;
                if (!server.listen(QHostAddress::LocalHost, 0))
                    throw std::runtime_error("cannot listen on loopback");
                ready.set_value(server.serverPort());
                struct Connection {
                    std::unique_ptr<QTcpSocket> socket;
                    QByteArray pending;
                    int id;
                };
                std::vector<Connection> clients;
                int nextId = 0;
                while (!stop_) {
                    server.waitForNewConnection(5);
                    while (server.hasPendingConnections())
                        clients.push_back({std::unique_ptr<QTcpSocket>(server.nextPendingConnection()), {}, nextId++});
                    for (auto& c : clients) {
                        if (!c.socket->bytesAvailable()) c.socket->waitForReadyRead(1);
                        c.pending += c.socket->readAll();
                        auto end = c.pending.indexOf("\r\n\r\n");
                        if (end < 0) continue;
                        auto headers = c.pending.left(end + 4);
                        c.pending.remove(0, end + 4);
                        const auto first = headers.split('\n').front().split(' ');
                        if (first.size() < 2) throw std::runtime_error("malformed test request");
                        Request request{first[0].toStdString(), first[1].toStdString(), headers.toStdString(), c.id};
                        {
                            std::lock_guard lock(mutex_);
                            requests_.push_back(request);
                        }
                        const auto reply = handler(request);
                        std::string out = "HTTP/1.1 " + std::to_string(reply.status) + " Test\r\n";
                        out += "Content-Length: " + std::to_string(reply.body.size()) + "\r\n";
                        if (reply.s3) out += "Server: AmazonS3\r\nx-amz-request-id: synthetic\r\n";
                        out += "Connection: keep-alive\r\n\r\n";
                        if (request.method != "HEAD") out += reply.body;
                        c.socket->write(out.data(), static_cast<qint64>(out.size()));
                        c.socket->waitForBytesWritten(1000);
                    }
                }
            } catch (...) {
                error_ = std::current_exception();
                try { ready.set_exception(error_); } catch (const std::future_error&) {}
            }
        });
        try { url_ = "http://127.0.0.1:" + std::to_string(port.get()); }
        catch (...) { stop_ = true; thread_.join(); throw; }
    }
    ~LoopbackHttpServer() { stop_ = true; thread_.join(); }
    std::string url(const std::string& path = "/key") const { return url_ + path; }
    std::vector<Request> requests() const {
        std::lock_guard lock(mutex_);
        return requests_;
    }
private:
    std::string url_;
    std::atomic<bool> stop_{false};
    std::thread thread_;
    mutable std::mutex mutex_;
    std::vector<Request> requests_;
    std::exception_ptr error_;
};
