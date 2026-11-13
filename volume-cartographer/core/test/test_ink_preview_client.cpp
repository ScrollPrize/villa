// The live ink preview's HTTP client (apps/VC3D/overlays/InkPreviewClient) against
// an in-test ink preview server (ink preview api 1, docs/ink_preview.md).
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "overlays/InkPreviewClient.hpp"
#include "support/LoopbackHttpServer.hpp"

#include <QCoreApplication>
#include <QDateTime>
#include <QElapsedTimer>
#include <QThread>

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <thread>

namespace {

using vc::test::LoopbackHttpServer;
using Response = LoopbackHttpServer::Response;

const char* kHealth =
    R"({"service": "ink-preview", "api": 1, "backend": "fake", "model": "test-model", "patch": 64, "stride": 32,)"
    R"( "depth": 5, "voxel_um": 9.36, "max_view_px": 1000000, "cache": {"tiles": 0}})";

Response json(int status, const std::string& body)
{
    Response response;
    response.status = status;
    response.body = body;
    response.headers.push_back({"Content-Type", "application/json"});
    return response;
}

// Runs the event loop until `done` holds or `ms` pass.
bool waitFor(const std::function<bool()>& done, int ms = 5000)
{
    QElapsedTimer timer;
    timer.start();
    while (!done() && timer.elapsed() < ms) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        QThread::msleep(1);
    }
    return done();
}

QUrl urlOf(const LoopbackHttpServer& server)
{
    return QUrl(QString::fromStdString(server.url()));
}

std::optional<InkPreviewClient::Reply> preview(InkPreviewClient& client, const InkPreviewClient::View& view,
                                               const InkPreviewClient::Upload* upload = nullptr)
{
    std::optional<InkPreviewClient::Reply> out;
    client.requestPreview(view, upload, [&](const InkPreviewClient::Reply& reply) { out = reply; });
    waitFor([&] { return out.has_value(); });
    return out;
}

InkPreviewClient::View testView()
{
    InkPreviewClient::View view;
    view.key = QStringLiteral("vc3d:a+b&c=d:e=1");
    view.canvas = QSize(300, 200);
    view.rect = QRect(100, 50, 12, 10);
    view.patch = 64;
    return view;
}

// A need the view's patches can produce: inside the view widened by patch - 1 = 63 px.
const char* kNeed = R"({"need": {"x": 64, "y": 32, "w": 64, "h": 40}, "depth": 5, "missing": 1})";

} // namespace

TEST_CASE("health parses the server's model and limits")
{
    LoopbackHttpServer server;
    server.setHandler([](const LoopbackHttpServer::Request& request) -> std::optional<Response> {
        if (request.path == "/health") {
            return json(200, kHealth);
        }
        return std::nullopt;
    });
    InkPreviewClient client;
    client.setServerUrl(urlOf(server));

    std::optional<InkPreviewClient::Health> health;
    QString error;
    bool done = false;
    client.checkHealth([&](const std::optional<InkPreviewClient::Health>& h, const QString& e) {
        health = h;
        error = e;
        done = true;
    });
    REQUIRE(waitFor([&] { return done; }));
    REQUIRE(health.has_value());
    CHECK(error.isEmpty());
    CHECK(health->model == QStringLiteral("test-model"));
    CHECK(health->backend == QStringLiteral("fake"));
    CHECK(health->patch == 64);
    CHECK(health->stride == 32);
    CHECK(health->depth == 5);
    REQUIRE(health->voxelUm.has_value());
    CHECK(*health->voxelUm == doctest::Approx(9.36));
    CHECK(health->maxViewPx == 1000000);
}

TEST_CASE("health rejects another service, another api and a null voxel size is unknown")
{
    std::string body;
    LoopbackHttpServer server;
    server.setHandler([&body](const LoopbackHttpServer::Request&) -> std::optional<Response> { return json(200, body); });
    InkPreviewClient client;
    client.setServerUrl(urlOf(server));
    const auto check = [&client]() {
        std::optional<InkPreviewClient::Health> health;
        QString error;
        bool done = false;
        client.checkHealth([&](const std::optional<InkPreviewClient::Health>& h, const QString& e) {
            health = h;
            error = e;
            done = true;
        });
        REQUIRE(waitFor([&] { return done; }));
        return std::make_pair(health, error);
    };

    body = R"({"service": "something-else", "api": 1})";
    auto [h1, e1] = check();
    CHECK(!h1.has_value());
    CHECK(e1.contains(QStringLiteral("is not an ink preview server")));

    body = R"({"service": "ink-preview", "api": 2, "depth": 5, "patch": 64})";
    auto [h2, e2] = check();
    CHECK(!h2.has_value());
    CHECK(e2.contains(QStringLiteral("speaks ink preview api 2")));

    body = R"({"service": "ink-preview", "api": 1, "backend": "fake", "model": "m", "patch": 64, "stride": 32,)"
           R"( "depth": 17, "voxel_um": null, "max_view_px": 4096})";
    auto [h3, e3] = check();
    REQUIRE(h3.has_value());
    CHECK(!h3->voxelUm.has_value());
    CHECK(h3->depth == 17);
}

TEST_CASE("a 409 asks for layers, the upload follows, and 200 brings the view")
{
    const InkPreviewClient::View view = testView();
    LoopbackHttpServer server;
    server.setHandler([&view](const LoopbackHttpServer::Request& request) -> std::optional<Response> {
        if (request.method != "POST" || request.path != "/preview") {
            return json(400, R"({"error": "unexpected request"})");
        }
        if (request.body.empty()) {
            return json(409, R"({"need": {"x": 64, "y": 32, "w": 64, "h": 40}, "depth": 5, "missing": 3})");
        }
        // The upload: depth * sh * sw bytes for the rectangle the 409 named.
        if (request.query.at("sx") != "64" || request.query.at("sy") != "32" || request.query.at("sw") != "64" ||
            request.query.at("sh") != "40" || request.query.at("dtype") != "uint8" ||
            request.body.size() != std::size_t(5 * 40 * 64) ||
            request.header("content-type") != "application/octet-stream") {
            return json(400, R"({"error": "bad upload"})");
        }
        Response image;
        image.body.assign(std::size_t(view.rect.width() * view.rect.height()), '\0');
        for (std::size_t i = 0; i < image.body.size(); ++i) {
            image.body[i] = char(i % 251);
        }
        image.headers = {{"X-Shape", "10,12"}, {"X-Backend", "fake"}, {"X-Tiles", "3,1"}};
        return image;
    });
    InkPreviewClient client;
    client.setServerUrl(urlOf(server));

    const auto need = preview(client, view);
    REQUIRE(need.has_value());
    REQUIRE(need->kind == InkPreviewClient::Reply::Kind::Need);
    CHECK(need->need == QRect(64, 32, 64, 40));
    CHECK(need->depth == 5);
    CHECK(need->missing == 3);

    InkPreviewClient::Upload upload;
    upload.rect = need->need;
    upload.voxels = QByteArray(5 * 40 * 64, char(7));
    const auto image = preview(client, view, &upload);
    REQUIRE(image.has_value());
    REQUIRE_MESSAGE(image->kind == InkPreviewClient::Reply::Kind::Image, image->error.toStdString());
    CHECK(image->size == QSize(12, 10));
    REQUIRE(image->pixels.size() == 120);
    CHECK(uint8_t(image->pixels[119]) == 119);
    CHECK(image->backend == QStringLiteral("fake"));
    CHECK(image->computed == 3);
    CHECK(image->cached == 1);

    const auto requests = server.requests();
    REQUIRE(requests.size() == 2);
    const auto& first = requests.front();
    // The key's reserved characters arrive as the client wrote them.
    CHECK(first.query.at("key") == "vc3d:a+b&c=d:e=1");
    CHECK(first.query.at("cw") == "300");
    CHECK(first.query.at("ch") == "200");
    CHECK(first.query.at("x") == "100");
    CHECK(first.query.at("y") == "50");
    CHECK(first.query.at("w") == "12");
    CHECK(first.query.at("h") == "10");
    CHECK(first.query.at("variant") == "f");
    CHECK(first.query.at("format") == "u8");
    CHECK(first.query.count("sx") == 0);
}

TEST_CASE("a reply of the wrong size and a need outside the canvas fail")
{
    int calls = 0;
    LoopbackHttpServer server;
    server.setHandler([&calls](const LoopbackHttpServer::Request&) -> std::optional<Response> {
        if (calls++ == 0) {
            Response image;
            image.body.assign(10, '\0');
            image.headers = {{"X-Shape", "10,12"}};
            return image;
        }
        return json(409, R"({"need": {"x": 250, "y": 0, "w": 64, "h": 64}, "depth": 5, "missing": 1})");
    });
    InkPreviewClient client;
    client.setServerUrl(urlOf(server));

    const auto shortReply = preview(client, testView());
    REQUIRE(shortReply.has_value());
    CHECK(shortReply->kind == InkPreviewClient::Reply::Kind::Failed);
    CHECK(shortReply->error.contains(QStringLiteral("expected 120 bytes")));

    const auto outside = preview(client, testView());
    REQUIRE(outside.has_value());
    CHECK(outside->kind == InkPreviewClient::Reply::Kind::Failed);
    CHECK(outside->error.contains(QStringLiteral("is not inside the view")));
}

TEST_CASE("a need may reach at most patch - 1 pixels past the view")
{
    std::string body;
    LoopbackHttpServer server;
    server.setHandler([&body](const LoopbackHttpServer::Request&) -> std::optional<Response> { return json(409, body); });
    InkPreviewClient client;
    client.setServerUrl(urlOf(server));
    auto view = testView();
    view.canvas = QSize(60000, 40000);
    view.rect = QRect(1000, 1000, 100, 100);

    // The patches of a 64 px grid that overlap the view: 63 px past it on every side.
    body = R"({"need": {"x": 937, "y": 937, "w": 226, "h": 226}, "depth": 17, "missing": 9})";
    const auto edge = preview(client, view);
    REQUIRE(edge.has_value());
    REQUIRE_MESSAGE(edge->kind == InkPreviewClient::Reply::Kind::Need, edge->error.toStdString());
    CHECK(edge->need == QRect(937, 937, 226, 226));

    // One pixel further on the left.
    body = R"({"need": {"x": 936, "y": 937, "w": 227, "h": 226}, "depth": 17, "missing": 9})";
    const auto further = preview(client, view);
    REQUIRE(further.has_value());
    CHECK(further->kind == InkPreviewClient::Reply::Kind::Failed);
    CHECK(further->error.contains(QStringLiteral("is not inside the view (1000, 1000, 100 x 100) widened by its patch size 64")));

    // The whole canvas for a 100 x 100 view: 40.8 G voxels at depth 17 are never sampled.
    body = R"({"need": {"x": 0, "y": 0, "w": 60000, "h": 40000}, "depth": 17, "missing": 1})";
    const auto whole = preview(client, view);
    REQUIRE(whole.has_value());
    CHECK(whole->kind == InkPreviewClient::Reply::Kind::Failed);

    // Without a patch size the need must lie inside the view.
    view.patch = 0;
    body = R"({"need": {"x": 937, "y": 937, "w": 226, "h": 226}, "depth": 17, "missing": 9})";
    const auto unknown = preview(client, view);
    REQUIRE(unknown.has_value());
    CHECK(unknown->kind == InkPreviewClient::Reply::Kind::Failed);
}

TEST_CASE("redirects are not followed")
{
    // A second server that would answer, and must hear nothing.
    LoopbackHttpServer other;
    other.setHandler([](const LoopbackHttpServer::Request& request) -> std::optional<Response> {
        if (request.path == "/health") {
            return json(200, kHealth);
        }
        return json(409, kNeed);
    });
    int status = 307;
    LoopbackHttpServer local;
    const std::string target = other.url();
    local.setHandler([target, &status](const LoopbackHttpServer::Request& request) -> std::optional<Response> {
        Response r;
        r.status = status;
        r.headers.push_back({"Location", target + request.target});
        return r;
    });
    InkPreviewClient client;
    client.setServerUrl(urlOf(local));

    for (const int code : {307, 308, 301, 302}) {
        CAPTURE(code);
        status = code;
        InkPreviewClient::Upload upload;
        upload.rect = QRect(64, 32, 64, 40);
        upload.voxels = QByteArray(5 * 40 * 64, 'S');
        const auto reply = preview(client, testView(), &upload);
        REQUIRE(reply.has_value());
        CHECK(reply->kind == InkPreviewClient::Reply::Kind::Failed);
        CHECK(reply->httpStatus == code);
        CHECK(reply->error.contains(QStringLiteral("VC3D does not follow redirects")));

        std::optional<QString> healthError;
        client.checkHealth([&](const std::optional<InkPreviewClient::Health>& h, const QString& e) {
            healthError = h ? QString() : e;
        });
        REQUIRE(waitFor([&] { return healthError.has_value(); }));
        CHECK(healthError->contains(QStringLiteral("VC3D does not follow redirects")));
    }
    CHECK(local.requests().size() == 8);
    CHECK(other.requests().empty());
}

TEST_CASE("the reversed view asks for variant r")
{
    LoopbackHttpServer server;
    server.setHandler([](const LoopbackHttpServer::Request&) -> std::optional<Response> { return json(409, kNeed); });
    InkPreviewClient client;
    client.setServerUrl(urlOf(server));
    auto view = testView();
    view.reversed = true;
    REQUIRE(preview(client, view).has_value());
    REQUIRE(server.requests().size() == 1);
    CHECK(server.requests().front().query.at("variant") == "r");
}

TEST_CASE("cancel and a new server address drop replies in flight")
{
    static constexpr int kHandlerMs = 300;
    std::atomic<int> answered{0};
    LoopbackHttpServer server;
    server.setHandler([&answered](const LoopbackHttpServer::Request&) -> std::optional<Response> {
        std::this_thread::sleep_for(std::chrono::milliseconds(kHandlerMs));
        ++answered;
        return json(409, kNeed);
    });
    InkPreviewClient client;
    client.setServerUrl(urlOf(server));

    int stale = 0;
    client.requestPreview(testView(), nullptr, [&stale](const InkPreviewClient::Reply&) { ++stale; });
    CHECK(client.busy());
    client.cancel();
    CHECK(!client.busy());

    // A request after the cancel is answered; the cancelled one never is.
    const auto fresh = preview(client, testView());
    REQUIRE(fresh.has_value());
    CHECK(fresh->kind == InkPreviewClient::Reply::Kind::Need);

    client.requestPreview(testView(), nullptr, [&stale](const InkPreviewClient::Reply&) { ++stale; });
    const int before = answered.load();
    client.setServerUrl(urlOf(server));
    // Wait until the server has answered the dropped request (it may also never see it,
    // since dropping the reply closes the connection), then give its answer as long
    // again to arrive. A callback that ran would show in `stale`.
    waitFor([&] { return answered.load() > before || stale > 0; }, 10 * kHandlerMs);
    waitFor([&] { return stale > 0; }, kHandlerMs);
    CHECK(stale == 0);
}

TEST_CASE("the server's refusal and error texts reach the reply")
{
    int status = 403;
    LoopbackHttpServer server;
    server.setHandler([&status](const LoopbackHttpServer::Request&) -> std::optional<Response> {
        if (status == 403) {
            return json(403, R"({"refused": "scroll PHerc0000 is not on the allow list"})");
        }
        if (status == 500) {
            return json(500, R"({"error": "backend crashed"})");
        }
        if (status == 413) {
            return json(413, R"({"error": "view of 5000000 px exceeds max_view_px 1000000"})");
        }
        Response plain;
        plain.status = 502;
        plain.body = "not json";
        return plain;
    });
    InkPreviewClient client;
    client.setServerUrl(urlOf(server));

    const auto refused = preview(client, testView());
    REQUIRE(refused.has_value());
    CHECK(refused->kind == InkPreviewClient::Reply::Kind::Failed);
    CHECK(refused->httpStatus == 403);
    CHECK(refused->error == QStringLiteral("The server refused the view: scroll PHerc0000 is not on the allow list"));

    status = 500;
    const auto failed = preview(client, testView());
    REQUIRE(failed.has_value());
    CHECK(failed->httpStatus == 500);
    CHECK(failed->error == QStringLiteral("The ink preview server reported an error (HTTP 500): backend crashed"));

    status = 413;
    const auto tooLarge = preview(client, testView());
    REQUIRE(tooLarge.has_value());
    CHECK(tooLarge->error.contains(QStringLiteral("(HTTP 413): view of 5000000 px exceeds")));

    status = 502;
    const auto other = preview(client, testView());
    REQUIRE(other.has_value());
    CHECK(other->kind == InkPreviewClient::Reply::Kind::Failed);
    CHECK(other->error.startsWith(QStringLiteral("The ink preview server answered HTTP 502")));
}

TEST_CASE("a server that does not answer in time fails the request")
{
    LoopbackHttpServer server;
    server.setHandler([](const LoopbackHttpServer::Request&) -> std::optional<Response> {
        std::this_thread::sleep_for(std::chrono::milliseconds(1200));
        return json(409, kNeed);
    });
    InkPreviewClient client;
    client.setServerUrl(urlOf(server));
    client.setTimeouts(200, 200);

    // The error text names the timeout, so no wall-clock bound is needed here.
    const auto reply = preview(client, testView());
    REQUIRE(reply.has_value());
    CHECK(reply->kind == InkPreviewClient::Reply::Kind::Failed);
    CHECK(reply->httpStatus == 0);
    CHECK(reply->error.contains(QStringLiteral("did not answer within 0.2 s")));

    std::optional<QString> healthError;
    client.checkHealth([&](const std::optional<InkPreviewClient::Health>&, const QString& e) { healthError = e; });
    REQUIRE(waitFor([&] { return healthError.has_value(); }));
    CHECK(healthError->contains(QStringLiteral("did not answer within 0.2 s")));
}

TEST_CASE("no server and a bad address are reported")
{
    // A port that was free a moment ago: nothing listens there.
    int freePort = 0;
    {
        LoopbackHttpServer probe;
        freePort = probe.port();
    }
    InkPreviewClient client;
    client.setServerUrl(QUrl(QStringLiteral("http://127.0.0.1:%1").arg(freePort)));
    const auto reply = preview(client, testView());
    REQUIRE(reply.has_value());
    CHECK(reply->kind == InkPreviewClient::Reply::Kind::Failed);
    CHECK(reply->httpStatus == 0);
    CHECK(reply->error.startsWith(QStringLiteral("No ink preview server answers at")));

    client.setServerUrl(QUrl(QStringLiteral("127.0.0.1:8765")));
    const auto bad = preview(client, testView());
    REQUIRE(bad.has_value());
    CHECK(bad->error.contains(QStringLiteral("http://HOST:PORT")));
    CHECK(InkPreviewClient::urlProblem(QUrl(QStringLiteral("http://127.0.0.1:8765"))).isEmpty());
    CHECK(!InkPreviewClient::urlProblem(QUrl(QStringLiteral("ftp://127.0.0.1:8765"))).isEmpty());
}

// The ctest test_ink_preview_client_server (--server): one round trip against the server
// INK_PREVIEW_TEST_SERVER names, for example the standalone server's fake backend. It
// exits with kSkipped when the variable is unset or no server answers there.
constexpr int kSkipped = 77;
bool serverSkipped = false;

void runningServerRoundTrip()
{
    const QString address = qEnvironmentVariable("INK_PREVIEW_TEST_SERVER");
    InkPreviewClient client;
    client.setServerUrl(QUrl(address));
    client.setTimeouts(3000, 60000);

    std::optional<InkPreviewClient::Health> health;
    QString error;
    bool done = false;
    client.checkHealth([&](const std::optional<InkPreviewClient::Health>& h, const QString& e) {
        health = h;
        error = e;
        done = true;
    });
    REQUIRE(waitFor([&] { return done; }, 5000));
    if (!health) {
        MESSAGE("skipped: no ink preview server at " << address.toStdString() << ": " << error.toStdString());
        serverSkipped = true;
        return;
    }
    MESSAGE("server " << health->model.toStdString() << " (" << health->backend.toStdString() << "), depth "
                      << health->depth << ", patch " << health->patch << ", stride " << health->stride);

    InkPreviewClient::View view;
    view.key = QStringLiteral("vc3d-test:%1").arg(QDateTime::currentMSecsSinceEpoch());
    view.canvas = QSize(300, 200);
    view.rect = QRect(100, 50, 120, 100);
    view.patch = health->patch;
    const auto first = preview(client, view);
    REQUIRE(first.has_value());
    REQUIRE_MESSAGE(first->kind == InkPreviewClient::Reply::Kind::Need, first->error.toStdString());
    CHECK(first->depth == health->depth);
    CHECK(QRect(QPoint(0, 0), view.canvas).contains(first->need));
    CHECK(first->need.contains(view.rect));

    InkPreviewClient::Upload upload;
    upload.rect = first->need;
    upload.voxels = QByteArray(qsizetype(first->depth) * first->need.width() * first->need.height(), '\0');
    for (qsizetype i = 0; i < upload.voxels.size(); ++i) {
        upload.voxels[i] = char((i * 7) % 256);
    }
    const auto second = preview(client, view, &upload);
    REQUIRE(second.has_value());
    REQUIRE_MESSAGE(second->kind == InkPreviewClient::Reply::Kind::Image, second->error.toStdString());
    CHECK(second->size == view.rect.size());
    CHECK(second->pixels.size() == qsizetype(view.rect.width()) * view.rect.height());
    MESSAGE("200: " << second->size.width() << " x " << second->size.height() << ", " << second->computed
                    << " patches computed, " << second->cached << " cached, backend "
                    << second->backend.toStdString());

    // The same view again comes from the server's cache, without an upload.
    const auto cached = preview(client, view);
    REQUIRE(cached.has_value());
    CHECK(cached->kind == InkPreviewClient::Reply::Kind::Image);
    CHECK(cached->pixels == second->pixels);
    MESSAGE("repeat: " << cached->computed << " computed, " << cached->cached << " cached");
}

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "--server") {
        if (qEnvironmentVariable("INK_PREVIEW_TEST_SERVER").isEmpty()) {
            std::cout << "skipped: set INK_PREVIEW_TEST_SERVER to a running ink preview server's address\n";
            return kSkipped;
        }
        auto& runner = ::doctest::detail::Runner::get();
        runner.cases = {{"a running ink preview server answers the round trip", &runningServerRoundTrip}};
        const int result = ::doctest::detail::runAll();
        return result == 0 && serverSkipped ? kSkipped : result;
    }
    return ::doctest::detail::runAll();
}
