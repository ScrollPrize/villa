#include "TestSupport.hpp"
#include "vc_object_store/client/HostedStore.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/server/Server.hpp"
#include <vc_reader/http.hpp>
#include <vc_reader/sftp.hpp>
#include <atomic>
#include <array>
#include <cstdlib>
#include <iostream>

using namespace vc::objects;

int main()
try {
    for (const auto* endpoint : {"ssh://host", "ssh://host:0", "ssh://host:65536",
                                "ssh://host:8750/file.vcobj", "sftp://host:8750?x=y",
                                "ssh://user:password@host:8750", "ssh://-oProxyCommand=x:8750"})
        mustFail([&] { (void)vcr::sftp::parseForwardUrl(endpoint); });
    const auto parsed = vcr::sftp::parseForwardUrl("sftp://user@[::1]:8750/");
    require(parsed.host == "::1" && parsed.user == "user" && parsed.port == 8750, "forward endpoint parsing");
    const char* host = std::getenv("VC_SSH_TEST_HOST");
    if (!host) { std::cout << "SSH parsing passed; set VC_SSH_TEST_HOST for transport checks\n"; return 77; }
    TemporaryDirectory directory;
    const auto path = directory.path / "test.vcobj";
    OpenOptions options; options.author = "ssh-test";
    if (const char* source = std::getenv("VC_SSH_TEST_STORE")) {
        std::filesystem::copy_file(source, path);
        if (std::filesystem::exists(std::string(source) + ".artifacts"))
            std::filesystem::copy(std::string(source) + ".artifacts", path.string() + ".artifacts", std::filesystem::copy_options::recursive);
    } else { (void)ObjectStore::create(path, options); }
    server::Server service(path, {});
    service.start();
    const std::string address = std::string(host) + ":" + std::to_string(service.port());
    HostedStore direct("http://127.0.0.1:" + std::to_string(service.port()));
    const auto expected = direct.status();
    std::atomic<int> confirmations{0};
    vcr::sftp::setAuthenticator([&](const vcr::sftp::Prompt& prompt) -> std::optional<std::string> {
        if (!prompt.confirmation) return std::nullopt;
        ++confirmations;
        return "yes"; // only the disposable local sshd started by run_ssh_test.py
    });
    for (const auto* scheme : {"ssh://", "sftp://"}) {
        const auto endpoint = scheme + address;
        ObjectStoreSession session;
        const auto attachment = session.attachLocation(endpoint, options);
        require(session.hosted(attachment) && session.location(attachment) == endpoint, "session preserves the pasted URL");
        require(attachment.storeUuid == expected.storeUuid, "SSH reaches the intended database");
        ObjectRecord record; record.uuid = util::newUuid(); record.kind = "track"; record.track.emplace();
        record.track->datasetRef = "ssh-test"; record.track->externalKey = record.uuid;
        auto request = session.newRequest(attachment);
        request.expectedObjects[record.uuid] = std::nullopt;
        request.operations = {CreateObjectOp{objectRecordJson(record)}};
        session.submit(attachment, request);
        require(direct.currentObjectRevision(record.uuid).has_value(), "SSH edits reach the hosted database");
        session.detach(attachment);
    }
    require(confirmations == 1, "host verification prompt is shared and known host is reused");
    {
        const auto url = vcr::sftp::parseUrl("sftp://" + std::string(host) + vcr::sftp::encodePath(path.string()));
        vcr::sftp::Session files(url);
        const auto handle = files.open(url.path, vcr::sftp::Read);
        std::array<std::byte, 16> header{};
        require(files.read(handle, 0, header) == header.size() &&
                std::string(reinterpret_cast<const char*>(header.data()), 15) == "SQLite format 3",
                "existing SFTP file reads still use the shared SSH transport");
        files.close(handle);
    }
    mustFail([&] { vcr::sftp::Forward refused("ssh://vc-store-refused:" + std::to_string(service.port())); });
    std::string local;
    {
        vcr::sftp::Forward forward("ssh://" + address);
        local = forward.endpoint();
        HostedStore forwarded(local);
        require(forwarded.status().storeUuid == expected.storeUuid, "forward readiness precedes HTTP access");
    }
    vcr::http::Options once; once.retries = 0; once.connectTimeoutMs = 200; once.requestTimeoutMs = 500;
    vcr::http::Session closed(once);
    mustFail([&] { closed.request("GET", local + "/v1/status"); });
#ifndef _WIN32
    // Only this test process: private tunnel traffic must ignore HTTP proxies.
    setenv("http_proxy", "http://127.0.0.1:1", 1);
    setenv("ALL_PROXY", "http://127.0.0.1:1", 1);
    setenv("no_proxy", "", 1);
    setenv("NO_PROXY", "", 1);
    {
        HostedStore tunneled("ssh://" + address);
        require(tunneled.status().storeUuid == expected.storeUuid, "SSH requests bypass environment proxies");
    }
#endif
    vcr::sftp::setAuthenticator({});
    std::cout << "SSH hosted store tests passed (both schemes, publication, host trust, cleanup)\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
