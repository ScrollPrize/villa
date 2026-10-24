// SFTP hub access through the production transport. The test executable doubles
// as `ssh` on PATH: it checks the argument contract, optionally demands a
// password through SSH_ASKPASS, then execs the installed OpenSSH sftp-server.
// No network, sshd or credentials are involved.
#include "TestSupport.hpp"
#include "vc_object_store/objects/Hub.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/sftp/Sftp.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <unistd.h>

using namespace vc::objects;
namespace sftp = vc::sftp;

namespace {
std::string shellQuote(const std::string& text)
{
    std::string quoted = "'";
    for (const char c : text) quoted += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return quoted + "'";
}

int fakeSsh(int argc, char** argv)
{
    const std::vector<std::string> args(argv + 1, argv + argc);
    auto has = [&](const std::string& value) { return std::find(args.begin(), args.end(), value) != args.end(); };
    const bool interactive = has("-oBatchMode=no");
    if (!has("-s") || !has("--") || args.back() != "sftp" || !has("test-host") ||
        !has(interactive ? "-oStrictHostKeyChecking=ask" : "-oStrictHostKeyChecking=yes"))
        return 42;
    if (const char* expected = std::getenv("VC_SFTP_TEST_PASSWORD")) {
        const char* askpass = std::getenv("SSH_ASKPASS");
        const char* require = std::getenv("SSH_ASKPASS_REQUIRE");
        if (!interactive || !askpass || !require || std::string(require) != "force") {
            std::cerr << "Permission denied (password).\n";
            return 255;
        }
        bool accepted = false;
        for (int attempt = 0; attempt < 3 && !accepted; ++attempt) {
            const auto command = shellQuote(askpass) + " " + shellQuote("tester@test-host's password: ");
            FILE* pipe = popen(command.c_str(), "r");
            std::string answer;
            char buffer[256];
            while (pipe && fgets(buffer, sizeof buffer, pipe)) answer += buffer;
            if (pipe && pclose(pipe) != 0) answer.clear();
            if (!answer.empty() && answer.back() == '\n') answer.pop_back();
            accepted = answer == expected;
        }
        if (!accepted) {
            std::cerr << "Permission denied (password).\n";
            return 255;
        }
    }
    const char* server = std::getenv("VC_SFTP_TEST_SERVER");
    execl(server, server, "-e", static_cast<char*>(nullptr));
    return 43;
}

OpenOptions as(const std::string& author) { OpenOptions options; options.author = author; return options; }

std::string meta(ObjectStore& store, const std::string& uuid)
{
    auto txn = store.read();
    const auto object = store.objectByUuid(txn, uuid);
    return nlohmann::json::parse(object ? object->meta : store.edgeByUuid(txn, uuid)->meta).dump();
}

void setNote(ObjectStore& store, const std::string& uuid, const std::string& value)
{
    auto txn = store.write();
    store.setMeta(txn, uuid, "note", nlohmann::json(value).dump());
    txn.commit();
}

std::vector<std::byte> bytes(size_t size, unsigned seed)
{
    std::vector<std::byte> data(size);
    for (size_t i = 0; i < size; ++i) data[i] = std::byte((i * 131 + seed) % 251);
    return data;
}

void urls()
{
    const auto url = sftp::parseUrl("SSH://Alice@Test-Host:2222/a%20b%23c.zarr");
    require(url.user == "Alice" && url.host == "test-host" && url.port == 2222 && url.path == "/a b#c.zarr", "URL parts");
    require(url.str() == "sftp://Alice@test-host:2222/a%20b%23c.zarr", "canonical URL");
    require(url.authority() == "Alice@test-host:2222", "authority");
    require(sftp::parseUrl("sftp://host").path == "/" && sftp::parseUrl("sftp://[::1]:22/x").host == "::1", "defaults and IPv6");
    for (const auto* bad : {"sftp://user:password@host/x", "sftp://host/x?token=x", "sftp://host/x#frag", "sftp://-host/x",
                            "sftp:///x", "sftp://host/a%00b", "sftp://host:0/x", "sftp://host:99999/x", "sftp://ho st/x",
                            "http://host/x"})
        mustFail([&] { sftp::parseUrl(bad); });
    require(sftp::isSftpUrl("SFTP://host/x") && !sftp::isSftpUrl("/local/path"), "scheme detection");
}

void session(const std::filesystem::path& root)
{
    sftp::Session s(sftp::parseUrl("sftp://test-host" + sftp::encodePath(root.string())));
    require(s.supports("fsync@openssh.com"), "OpenSSH advertises fsync");
    const auto file = (root / "data space.bin").string();
    const auto data = bytes(1 << 20, 7);
    auto handle = s.open(file, sftp::Read | sftp::Write | sftp::Create | sftp::Exclusive);
    s.write(handle, 0, data);  // pipelined; a later read waits for it
    std::vector<std::byte> back(data.size());
    require(s.read(handle, 0, back) == data.size() && back == data, "read after pipelined writes");
    s.flush();
    s.fsync(handle);
    require(s.fstat(handle).size == data.size(), "fstat size");
    std::vector<std::byte> tail(100);
    require(s.read(handle, data.size() - 40, tail) == 40, "short read at end of file");
    uint64_t streamed = 0;
    require(s.read(handle, 5, data.size(), [&](std::span<const std::byte> part) {
        require(std::equal(part.begin(), part.end(), data.begin() + ptrdiff_t(5 + streamed)), "streamed bytes");
        streamed += part.size();
    }) == data.size() - 5, "streamed read stops at end of file");
    s.truncate(handle, 1000);
    s.close(handle);
    require(s.stat(file).size == 1000 && !s.stat(file).directory(), "truncate");
    mustFail([&] { s.open(file, sftp::Write | sftp::Create | sftp::Exclusive); });
    s.mkdir((root / "dir").string());
    try {
        s.mkdir((root / "dir").string());
        throw std::runtime_error("mkdir of an existing directory must fail");
    } catch (const sftp::StatusError&) {}
    require(s.stat((root / "dir").string()).directory(), "directory attributes");
    s.rename(file, (root / "dir" / "moved").string());
    const auto listing = s.list((root / "dir").string());
    require(listing.size() == 1 && listing[0].name == "moved", "listing");
    try {
        s.stat(file);
        throw std::runtime_error("stat of a missing file must fail");
    } catch (const sftp::StatusError& error) {
        require(error.status == sftp::NoSuchFile, "missing file status");
    }
    s.remove((root / "dir" / "moved").string());
    s.rmdir((root / "dir").string());
    require(s.realpath(root.string() + "/.") == std::filesystem::canonical(root).string(), "realpath");
}

void passwords(const std::filesystem::path& root)
{
    const auto location = sftp::parseUrl("sftp://tester@test-host" + sftp::encodePath(root.string()));
    setenv("VC_SFTP_TEST_PASSWORD", "s3cret", 1);
    mustFail([&] { sftp::Session batch(location); });  // batch mode never prompts
    std::atomic<int> asked = 0;
    sftp::setAuthenticator([&](const sftp::Prompt& prompt) -> std::optional<std::string> {
        require(prompt.destination == "tester@test-host" && !prompt.confirmation, "prompt context");
        require(prompt.text.find("password") != std::string::npos, "ssh's own prompt text");
        return ++asked == 1 ? "wrong" : "s3cret";
    });
    { sftp::Session first(location); require(first.stat(root.string()).directory(), "authenticated session"); }
    require(asked == 2, "a repeated prompt asks the user again instead of reusing a wrong answer");
    { sftp::Session again(location); }
    require(asked == 2, "a reconnect reuses the accepted answer");
    sftp::setAuthenticator([&](const sftp::Prompt&) -> std::optional<std::string> { ++asked; return std::nullopt; });
    mustFail([&] { sftp::Session cancelled(sftp::parseUrl("sftp://other@test-host/")); });
    require(asked == 5, "cancelling answers every prompt");
    sftp::setAuthenticator({});
    unsetenv("VC_SFTP_TEST_PASSWORD");
}

void hosting(const std::filesystem::path& root)
{
    const auto share = root / "share";
    std::filesystem::create_directory(share);
    const auto hubPath = share / "hub.vcobj";
    const auto hubUrl = "sftp://test-host" + sftp::encodePath(hubPath.string());
    const auto a = util::newUuid(), b = util::newUuid(), c = util::newUuid();
    vc::geometry::Fiber fiber;
    fiber.controlPoints = {{0, 0, 0}, {10, 0, 0}, {20, 0, 0}};
    fiber.controlPointIds = {1, 2, 3};
    {
        auto origin = ObjectStore::create(hubPath, as("curator"));
        auto txn = origin.write();
        for (const auto& uuid : {a, b}) {
            ObjectRecord record; record.uuid = uuid; record.kind = "fiber";
            origin.upsertObject(txn, record);
            origin.putGeometry(txn, uuid, vc::geometry::encode(fiber));
        }
        txn.commit();
        origin.checkpoint();
    }
    initHub(hubPath, "curator");
    mustFail([&] { ObjectStore::open(hubUrl, as("curator")); });  // only hosted hubs open remotely
    mustFail([&] { cloneHub(hubUrl + ".absent", root / "absent.vcobj", as("alice")); });

    // Alice reaches the hub over SFTP, Bob through a filesystem path: both use
    // the same dot-file lock, so they serialize against each other.
    const auto alicePath = root / "alice.vcobj", bobPath = root / "bob.vcobj";
    int64_t progress = 0;
    cloneHub(hubUrl, alicePath, as("alice"), [&](const auto&) { ++progress; });
    cloneHub(hubPath, bobPath, as("bob"));
    require(progress > 0, "remote clone reports progress");
    auto alice = ObjectStore::open(alicePath, as("alice"));
    auto bob = ObjectStore::open(bobPath, as("bob"));
    require(hubStatus(alice)->hub == std::filesystem::path(hubUrl), "replica remembers the URL");
    require(meta(alice, a) == meta(bob, a), "clones match");

    acquireLeases(alice, {a});
    mustFail([&] { acquireLeases(bob, {a}); });
    setNote(alice, a, "over sftp");
    require(hubSync(alice).pushed == 1, "push over SFTP");
    require(hubPull(bob).pulled == 1 && meta(bob, a) == meta(alice, a), "a mounted replica sees it");
    {
        auto txn = bob.write();
        ObjectRecord record; record.uuid = c; record.kind = "fiber";
        bob.upsertObject(txn, record);
        bob.putGeometry(txn, c, vc::geometry::encode(fiber));
        txn.commit();
    }
    hubSync(bob);
    hubPull(alice);
    {
        auto txn = alice.read();
        require(alice.objectByUuid(txn, c) && alice.currentGeometry(txn, c), "pull over SFTP installs geometry");
    }
    {
        auto shared = as("curator"); shared.sharedFile = true;
        auto hub = ObjectStore::open(hubUrl, shared);
        mustFail([&] { setNote(hub, a, "curator"); });  // leases hold for direct remote edits too
    }

    // Concurrent syncs through both access paths serialize on the one lock.
    acquireLeases(bob, {b});
    std::exception_ptr failure;
    std::mutex failureMutex;
    auto writer = [&](ObjectStore& store, const std::string& uuid) {
        try {
            for (int i = 0; i < 8; ++i) { setNote(store, uuid, std::to_string(i)); hubSync(store); }
        } catch (...) {
            std::lock_guard lock(failureMutex);
            failure = std::current_exception();
        }
    };
    { std::jthread x(writer, std::ref(alice), a), y(writer, std::ref(bob), b); }
    if (failure) std::rethrow_exception(failure);
    hubPull(alice);
    hubPull(bob);
    require(meta(alice, a) == meta(bob, a) && meta(alice, b) == meta(bob, b), "concurrent syncs converge");

    // Stale locks are visible and removable through either address.
    require(!hubLockAge(hubUrl) && !hubLockAge(hubPath), "unlocked after syncs");
    std::filesystem::create_directory(share / "hub.vcobj.lock");
    require(hubLockAge(hubUrl).has_value() && hubLockAge(hubPath).has_value(), "lock visible");
    removeHubLock(hubUrl);
    require(!std::filesystem::exists(share / "hub.vcobj.lock"), "lock removed over SFTP");
}
}  // namespace

int main(int argc, char** argv)
try {
    if (argc > 1 && std::string(argv[1]) == "-T") return fakeSsh(argc, argv);
    urls();
    std::string server;
    for (const auto* candidate : {"/usr/lib/openssh/sftp-server", "/usr/lib/ssh/sftp-server", "/usr/libexec/sftp-server",
                                  "/usr/libexec/openssh/sftp-server"})
        if (std::filesystem::exists(candidate)) server = candidate;
    if (server.empty()) {
        std::cout << "object_store_sftp: no installed OpenSSH sftp-server; skipped\n";
        return 77;
    }
    TemporaryDirectory directory;
    const auto bin = directory.path / "bin";
    std::filesystem::create_directory(bin);
    std::filesystem::create_symlink(std::filesystem::absolute(argv[0]), bin / "ssh");
    setenv("PATH", (bin.string() + ":" + std::getenv("PATH")).c_str(), 1);
    setenv("VC_SFTP_TEST_SERVER", server.c_str(), 1);
    session(directory.path);
    passwords(directory.path);
    hosting(directory.path);
    std::cout << "object_store_sftp: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
