#include "McpApplication.hpp"

#include <fastmcpp/server/host_origin_guard.hpp>
#include <fastmcpp/server/streamable_http_server.hpp>

#include <cassert>
#include <httplib.h>
#include <iostream>
#include <string>
#include <vector>

using vc::mcp::Json;

namespace
{

Json initializeRequest()
{
    return {
        {"jsonrpc", "2.0"},
        {"id", 1},
        {"method", "initialize"},
        {"params",
         {{"protocolVersion", "2025-03-26"},
          {"capabilities", Json::object()},
          {"clientInfo", {{"name", "vc-http-test"}, {"version", "1"}}}}}};
}

}  // namespace

int main()
{
    constexpr const char* token = "0123456789abcdef0123456789abcdef";
    vc::mcp::McpApplication app({"test", "test", "sha256:test"});
    fastmcpp::server::StreamableHttpServerWrapper
        server(app.handler(), "127.0.0.1", 0, "/mcp", token, "", {{"Cache-Control", "no-store"}, {"X-Content-Type-Options", "nosniff"}});
    fastmcpp::server::HostOriginGuardOptions guard;
    guard.mode = fastmcpp::server::HostOriginProtectionMode::Auto;
    guard.allowed_hosts = std::vector<std::string>{"127.0.0.1", "localhost"};
    server.set_host_origin_guard(std::move(guard));
    assert(server.start());
    assert(server.port() > 0);

    httplib::Client client("127.0.0.1", server.port());
    client.set_connection_timeout(5, 0);
    client.set_read_timeout(5, 0);

    const std::string body = initializeRequest().dump();
    const auto unauthorized = client.Post("/mcp", body, "application/json");
    assert(unauthorized);
    assert(unauthorized->status == 401);

    httplib::Headers headers = {{"Authorization", std::string("Bearer ") + token}, {"Accept", "application/json, text/event-stream"}};
    const auto initialized = client.Post("/mcp", headers, body, "application/json");
    assert(initialized);
    assert(initialized->status == 200);
    assert(initialized->get_header_value("Cache-Control") == "no-store");
    assert(initialized->get_header_value("X-Content-Type-Options") == "nosniff");
    const std::string sessionId = initialized->get_header_value("Mcp-Session-Id");
    assert(!sessionId.empty());
    const auto response = Json::parse(initialized->body);
    assert(response.at("result").at("serverInfo").at("name") == "volume-cartographer");

    httplib::Headers hostileOrigin = headers;
    hostileOrigin.emplace("Origin", "https://attacker.invalid");
    const auto forbidden = client.Post("/mcp", hostileOrigin, body, "application/json");
    assert(forbidden);
    assert(forbidden->status == 403);

    headers.emplace("Mcp-Session-Id", sessionId);
    const auto terminated = client.Delete("/mcp", headers);
    assert(terminated);
    assert(terminated->status == 200 || terminated->status == 204);

    server.stop();
    std::cout << "HttpTransportTest passed\n";
    return 0;
}
