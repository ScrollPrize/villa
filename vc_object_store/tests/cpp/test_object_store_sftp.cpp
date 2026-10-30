// SFTP hub access through the production transport. The test executable doubles
// as `ssh` on PATH: it checks the argument contract, optionally demands a
// password through SSH_ASKPASS, then execs the installed OpenSSH sftp-server.
// No network, sshd or credentials are involved.
#include "TestSupport.hpp"
#include "HubTestSupport.hpp"
#include "sftp/SftpVfs.hpp"
#include "hub/HubWorker.hpp"
#include "vc_object_store/objects/Hub.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/ObjectCopy.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/sftp/Sftp.hpp"
#include <nlohmann/json.hpp>
#include <sqlite3.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <sys/file.h>
#include <fcntl.h>
#include <cerrno>
#include <signal.h>
#include <fstream>
#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <future>
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
    const bool locked = !has("-s");
    if (!has("--") || (locked ? args.back().find("exec timeout") == std::string::npos : args.back() != "sftp") || !has("test-host") ||
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
    if (const char* mask = std::getenv("VC_SFTP_TEST_UMASK")) umask(mode_t(std::stoi(mask, nullptr, 8)));
    const char* server = std::getenv("VC_SFTP_TEST_SERVER");
    if (const char* pidFile = std::getenv("VC_SFTP_TEST_PID_FILE"))
        std::ofstream(pidFile) << getpid();
    auto launch = [&] {
        if (locked) execl("/bin/sh", "sh", "-c", args.back().c_str(), static_cast<char*>(nullptr));
        else execl(server, server, "-e", static_cast<char*>(nullptr));
        _exit(43);
    };
    const char* latency = std::getenv("VC_SFTP_TEST_LATENCY_MS");
    const char* disconnect = std::getenv("VC_SFTP_TEST_DISCONNECT");
    const char* replyLog = std::getenv("VC_SFTP_TEST_REPLY_LOG");
    if (const char* log = std::getenv("VC_SFTP_TEST_CONNECTION_LOG"))
        if (locked) std::ofstream(log, std::ios::app) << "locked\n";
    if (latency || disconnect || replyLog) {
        // Delay replies or discard one after the server has executed a request.
        int out[2];
        if (pipe(out)) return 44;
        const auto child = fork();
        if (child == 0) {
            dup2(out[1], 1); close(out[0]); close(out[1]);
            launch();
        }
        close(out[1]);
        std::array<char, 65536> buffer;
        std::string packets;
        for (ssize_t n; (n = read(out[0], buffer.data(), buffer.size())) > 0;) {
            if (replyLog) {
                packets.append(buffer.data(), size_t(n));
                auto u32 = [&](size_t at) {
                    uint32_t value = 0;
                    for (size_t i = at; i < at + 4; ++i) value = (value << 8) | uint8_t(packets[i]);
                    return value;
                };
                while (packets.size() >= 4 && packets.size() >= 4 + u32(0)) {
                    const auto length = u32(0);
                    std::ofstream(replyLog, std::ios::app) << int(uint8_t(packets[4])) << ' '
                        << (uint8_t(packets[4]) == 103 ? u32(9) : 0) << '\n';
                    packets.erase(0, 4 + length);
                }
            }
            if (disconnect && std::filesystem::exists(disconnect)) {
                shutdown(0, SHUT_RDWR);
                close(out[0]);
                int status; waitpid(child, &status, 0);
                return 0;
            }
            if (latency) std::this_thread::sleep_for(std::chrono::milliseconds(std::stoi(latency)));
            for (ssize_t done = 0, w; done < n; done += w)
                if ((w = write(1, buffer.data() + done, size_t(n - done))) <= 0) return 0;
        }
        close(out[0]);
        int status; waitpid(child, &status, 0);
        return 0;
    }
    launch();
    return 43;
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
    const auto pidFile = root / "session.pid";
    setenv("VC_SFTP_TEST_PID_FILE", pidFile.c_str(), 1);
    sftp::Session s(sftp::parseUrl("sftp://test-host" + sftp::encodePath(root.string())));
    unsetenv("VC_SFTP_TEST_PID_FILE");
    require(!s.transportClosed(), "connected transport has no local EOF");
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
    pid_t pid = -1;
    std::ifstream(pidFile) >> pid;
    require(pid > 0 && kill(pid, SIGTERM) == 0, "terminate fixture transport");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!s.transportClosed()) {
        require(std::chrono::steady_clock::now() < deadline, "detect idle transport loss without network I/O");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    mustFail([&] { s.stat(root.string()); });
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

    mustFail([&] { cloneHub(hubUrl, root / "not-enabled.vcobj", as("alice")); });
    enableHubProcessLocking(hubUrl);
    enableHubProcessLocking(hubUrl); // idempotent; must never replace the lock inode
    mustFail([&] { cloneHub(hubPath, root / "mounted.vcobj", as("bob")); });
    // Both clients now use process-locked SSH access.
    const auto alicePath = root / "alice.vcobj", bobPath = root / "bob.vcobj";
    int64_t progress = 0;
    cloneHub(hubUrl, alicePath, as("alice"), [&](const auto&) { ++progress; });
    cloneHub(hubUrl, bobPath, as("bob"));
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
        vc::objects::detail::RemoteHubScope scope(hubUrl);
        auto hub = ObjectStore::open(hubUrl, shared);
        mustFail([&] { setNote(hub, a, "curator"); });  // leases hold for direct remote edits too
    }

    // Independent SSH sessions serialize on the same kernel lock.
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

    // Keeping local after a conflict must release the read operation's
    // process lock before starting the separate push operation.
    renewLeases(alice, {a}, 1);
    setNote(alice, a, "offline local edit");
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    acquireLeases(bob, {a});
    setNote(bob, a, "new hub edit");
    hubSync(bob);
    require(hubPull(alice).conflicts.size() == 1, "remote conflict recorded");
    releaseLeases(bob, {a});
    renewLeases(alice, {a});
    require(resolveHubConflict(alice, a, "local").pushed == 1, "keep-local opens a fresh locked operation");
    hubPull(bob);
    require(meta(alice, a) == meta(bob, a), "resolved replicas converge");

    // Repeated cache reuse and a metadata round trip on the slow connection
    // that originally exposed stranded locks, not only a standalone stat.
    setenv("VC_SFTP_TEST_LATENCY_MS", "250", 1);
    for (int i = 0; i < 2; ++i)
        require(hubSync(alice).pushed == 0, "idle cached sync at 250 ms reply latency");
    setNote(alice, a, "slow cached write");
    require(hubPush(alice).pushed == 1, "cached push at 250 ms reply latency");
    require(hubPull(bob).pulled == 1 && meta(alice, a) == meta(bob, a), "slow cached replicas converge");
    unsetenv("VC_SFTP_TEST_LATENCY_MS");

    // A pull reads hub geometry before it locks the replica, so local edits
    // during a pull over a slow link wait only for the local install.
    {
        setenv("VC_SFTP_TEST_LATENCY_MS", "20", 1);
        const auto carolPath = root / "carol.vcobj";
        cloneHub("sftp://slow@test-host" + sftp::encodePath(hubPath.string()), carolPath, as("carol"));
        const auto d = util::newUuid();
        vc::geometry::Fiber long_;
        for (int i = 0; i < 2000; ++i) {
            long_.controlPoints.push_back({double(i), 0, 0});
            long_.controlPointIds.push_back(i + 1);
        }
        {
            auto txn = bob.write();
            ObjectRecord record; record.uuid = d; record.kind = "fiber";
            bob.upsertObject(txn, record);
            bob.putGeometry(txn, d, vc::geometry::encode(long_));
            txn.commit();
        }
        hubSync(bob);
        auto carol = ObjectStore::open(carolPath, as("carol"));
        const auto e = util::newUuid();
        {
            auto txn = carol.write();
            ObjectRecord record; record.uuid = e; record.kind = "fiber";
            carol.upsertObject(txn, record);
            txn.commit();
        }
        auto impatient = as("carol"); impatient.busyTimeoutMs = 500;
        auto editor = ObjectStore::open(carolPath, impatient);
        std::atomic<bool> pulling = true;
        std::exception_ptr pullFailure;
        int edits = 0;
        {
            std::jthread pull([&] {
                try { hubPull(carol); } catch (...) { pullFailure = std::current_exception(); }
                pulling = false;
            });
            while (pulling) setNote(editor, e, std::to_string(edits++));
        }
        unsetenv("VC_SFTP_TEST_LATENCY_MS");
        if (pullFailure) std::rethrow_exception(pullFailure);
        auto txn = carol.read();
        require(edits > 1 && carol.currentGeometry(txn, d), "local edits proceed during a slow pull");
    }

    // A journal takes the hub's permission bits, whatever the server's umask, so
    // every user who may write the hub can also roll back a journal left behind.
    {
        std::filesystem::permissions(hubPath, std::filesystem::perms(0666));
        setenv("VC_SFTP_TEST_UMASK", "077", 1);
        const auto restrictive = "sftp://umask@test-host" + sftp::encodePath(hubPath.string());
        vc::objects::detail::RemoteHubScope scope(restrictive);
        sqlite3* db = nullptr;
        require(sqlite3_open_v2(restrictive.c_str(), &db, SQLITE_OPEN_READWRITE, "vc-sftp") == SQLITE_OK, "raw remote open");
        require(sqlite3_exec(db, "BEGIN IMMEDIATE; CREATE TABLE permission_probe(x);", nullptr, nullptr, nullptr) == SQLITE_OK,
                "write inside a transaction");
        struct stat journal{};
        require(::stat((hubPath.string() + "-journal").c_str(), &journal) == 0, "journal exists during the write");
        const auto mode = journal.st_mode & 0777;
        sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr);
        sqlite3_close(db);
        unsetenv("VC_SFTP_TEST_UMASK");
        require(mode == 0666, "journal mirrors the hub's permissions");
    }

    // This permanent guard is not a stale lock, and even an old client's
    // rmdir cannot remove it. Mounted clients cannot use incompatible locks.
    mustFail([&] { hubLockAge(hubUrl); });
    mustFail([&] { removeHubLock(hubUrl); });
    mustFail([&] { removeHubLock(hubPath); });
    require(std::filesystem::exists(hubPath.string() + ".lock/process-v1"), "process lock inode remains");

    // A live editor must be able to retain its lease while sync waits for
    // another client's remote process lock. The fixture log is a barrier:
    // do not rely on sleeping long enough for the worker to start syncing.
    {
        ObjectStoreSession session;
        const auto attachment = session.attach(alicePath, as("alice"));
        const auto sync = [&] {
            const auto status = session.syncHubAndWait(attachment, std::chrono::seconds(20));
            require(status && status->error.empty(), "session sync succeeds");
        };
        sync();
        auto editing = session.holdLeases(attachment, {a});
        sync();
        const auto log = root / "held-lease-connections";
        setenv("VC_SFTP_TEST_CONNECTION_LOG", log.c_str(), 1);
        auto gate = std::make_unique<vc::objects::detail::RemoteHubScope>(hubUrl);
        const auto before = std::filesystem::file_size(log);
        auto syncing = std::async(std::launch::async, sync);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::filesystem::file_size(log) == before && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const bool started = std::filesystem::file_size(log) > before;
        auto holding = std::async(std::launch::async, [&] { return session.holdLeases(attachment, {a, a}); });
        const bool retained = holding.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
        // Release before assertions/future destruction even on regression.
        gate.reset();
        auto duplicate = holding.get();
        syncing.get();
        unsetenv("VC_SFTP_TEST_CONNECTION_LOG");
        require(started && retained, "existing lease hold does not wait for blocked remote sync");
        editing.reset();
        sync();
        mustFail([&] { acquireLeases(bob, {a}); }); // duplicate hold still owns it
        duplicate.reset();
        sync();
        acquireLeases(bob, {a});
        releaseLeases(bob, {a});
    }

}
void readAhead(const std::filesystem::path& root)
{
    using namespace vc::objects::detail;
    // Exercise the actual VFS independently of SQLite's own page cache.
    const auto path = root / "read-ahead.bin";
    const auto url = "sftp://read-ahead@test-host" + sftp::encodePath(path.string());
    const auto original = bytes(1024 * 1024 + 123, 19);
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(original.data()), original.size());
    remoteEnableProcessLock(url);
    auto* vfs = sqlite3_vfs_find(sftpVfs());
    auto close = [](sqlite3_file* file) {
        if (file->pMethods) file->pMethods->xClose(file);
        sqlite3_free(file);
    };
    using Handle = std::unique_ptr<sqlite3_file, decltype(close)>;
    auto open = [&] {
        Handle file(static_cast<sqlite3_file*>(sqlite3_malloc(vfs->szOsFile)), close);
        require(bool(file), "allocate VFS handle");
        std::memset(file.get(), 0, vfs->szOsFile);
        require(vfs->xOpen(vfs, url.c_str(), file.get(), SQLITE_OPEN_READWRITE | SQLITE_OPEN_MAIN_DB, nullptr) == SQLITE_OK,
                "open read-ahead fixture");
        return file;
    };
    Handle reader(nullptr, close), writer(nullptr, close);
    auto read = [&](size_t offset, size_t size) {
        std::vector<std::byte> out(size, std::byte{0xff});
        const auto available = offset < original.size() ? std::min(size, original.size() - offset) : 0;
        require(reader->pMethods->xRead(reader.get(), out.data(), int(size), offset) ==
                (available == size ? SQLITE_OK : SQLITE_IOERR_SHORT_READ), "VFS read status");
        require(std::equal(out.begin(), out.begin() + available, original.begin() + std::min(offset, original.size())),
                "VFS read bytes");
        require(std::all_of(out.begin() + available, out.end(), [](auto b) { return b == std::byte{0}; }),
                "short VFS read zero fills the remainder");
    };
    const auto log = root / "read-ahead-replies";
    setenv("VC_SFTP_TEST_REPLY_LOG", log.c_str(), 1);
    {
        RemoteHubScope scope(url);
        reader = open(); writer = open();
        for (size_t offset = 0; offset < 128 * 1024; offset += 4096) read(offset, 4096);
        std::ifstream packets(log);
        int type = 0; size_t size = 0, dataReplies = 0;
        while (packets >> type >> size) dataReplies += type == 103;
        require(dataReplies <= 4, "32 SQLite page reads use at most four pipelined SFTP reads");
        read(65536 - 31, 4096); // unaligned and crosses a block boundary
        read(123, 200000); // a read larger than one block
        for (size_t offset = 0; offset < 1024 * 1024; offset += 65536) read(offset, 4096);
        read(0, 4096); // eviction followed by a reread
        read(original.size() - 17, 4096);
        read(original.size() + 100, 4096);

        const auto patch = bytes(4096, 27);
        require(writer->pMethods->xWrite(writer.get(), patch.data(), patch.size(), 0) == SQLITE_OK, "write through second handle");
        std::vector<std::byte> out(patch.size());
        require(reader->pMethods->xRead(reader.get(), out.data(), out.size(), 0) == SQLITE_OK && out == patch,
                "cached reader sees another handle's queued write");
        require(writer->pMethods->xWrite(writer.get(), original.data(), patch.size(), 0) == SQLITE_OK, "restore written bytes");
        read(0, 4096);
        require(writer->pMethods->xTruncate(writer.get(), 100) == SQLITE_OK, "truncate cached file");
        require(reader->pMethods->xRead(reader.get(), out.data(), out.size(), 0) == SQLITE_IOERR_SHORT_READ &&
                std::equal(out.begin(), out.begin() + 100, original.begin()) &&
                std::all_of(out.begin() + 100, out.end(), [](auto b) { return b == std::byte{0}; }), "truncate invalidates cached bytes");
        require(writer->pMethods->xWrite(writer.get(), original.data(), original.size(), 0) == SQLITE_OK, "regrow cached file");
        read(0, 4096);
        read(original.size() - 17, 4096); // short cached block must also be invalidated by growth
    }
    unsetenv("VC_SFTP_TEST_REPLY_LOG");
    std::array<std::byte, 4096> out;
    require(reader->pMethods->xRead(reader.get(), out.data(), out.size(), 0) == SQLITE_IOERR_READ,
            "cache cannot serve reads outside the hub lock");
    {
        RemoteHubScope scope(url);
        // Simulate another writer between operations, under the new lock.
        const auto patch = bytes(out.size(), 31);
        std::fstream(path, std::ios::binary | std::ios::in | std::ios::out).write(
            reinterpret_cast<const char*>(patch.data()), patch.size());
        require(reader->pMethods->xRead(reader.get(), out.data(), out.size(), 0) == SQLITE_OK &&
                std::equal(out.begin(), out.end(), patch.begin()), "new lock generation discards read-ahead");
        require(reader->pMethods->xLock(reader.get(), SQLITE_LOCK_SHARED) == SQLITE_OK, "mark unfinished transaction");
    }
    {
        RemoteHubScope scope(url);
        require(reader->pMethods->xRead(reader.get(), out.data(), out.size(), 0) == SQLITE_IOERR_READ,
                "failed lock session cannot serve cached bytes under a new scope");
        reader.reset(); writer.reset();
    }
}

// Faults run against real OpenSSH sftp-server and the production flock command.
void processLocks(const std::filesystem::path& root, const std::filesystem::path& realStore = {})
{
    using vc::objects::detail::RemoteHubScope;
    const auto path = root / "hub ' $literal ;.vcobj";
    const auto url = "sftp://test-host" + sftp::encodePath(path.string());
    if (realStore.empty()) {
        auto store = ObjectStore::create(path, as("owner")); store.checkpoint();
    } else {
        // Snapshot a supplied real store without changing it or relying on an
        // unsafe raw copy while it might still have WAL contents.
        sqlite3 *source = nullptr, *target = nullptr;
        require(sqlite3_open_v2(realStore.c_str(), &source, SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK, "open real store");
        require(sqlite3_open(path.c_str(), &target) == SQLITE_OK, "open real snapshot");
        auto* backup = sqlite3_backup_init(target, "main", source, "main");
        require(backup && sqlite3_backup_step(backup, -1) == SQLITE_DONE, "snapshot real store");
        require(sqlite3_backup_finish(backup) == SQLITE_OK, "finish real snapshot");
        sqlite3_close(target); sqlite3_close(source);
    }
    initHub(path, "owner");
    const auto directory = path.string() + ".lock";
    std::filesystem::create_directory(directory);
    mustFail([&] { enableHubProcessLocking(url); }); // never steal an old lock
    std::filesystem::remove(directory);
    enableHubProcessLocking(url);
    struct stat before{}, after{};
    const auto marker = directory + "/process-v1";
    require(::stat(marker.c_str(), &before) == 0, "process lock inode exists");
    enableHubProcessLocking(url);
    require(::stat(marker.c_str(), &after) == 0 && before.st_ino == after.st_ino, "migration preserves the lock inode");
    {
        const int probe = open(marker.c_str(), O_RDONLY);
        require(probe >= 0, "open kernel lock probe");
        {
            RemoteHubScope scope(url);
            require(flock(probe, LOCK_EX | LOCK_NB) == -1 && (errno == EWOULDBLOCK || errno == EAGAIN),
                    "the running SFTP server actually retains the kernel lock");
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (flock(probe, LOCK_EX | LOCK_NB) != 0) {
            require(std::chrono::steady_clock::now() < deadline, "normal session close releases the kernel lock");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        close(probe);
    }
    {
        sqlite3* legacy = nullptr;
        require(sqlite3_open_v2(path.c_str(), &legacy, SQLITE_OPEN_READWRITE, "unix-dotfile") == SQLITE_OK, "legacy open");
        require(sqlite3_exec(legacy, "SELECT count(*) FROM sqlite_master", nullptr, nullptr, nullptr) == SQLITE_BUSY,
                "an unmodified legacy VFS is blocked by the permanent guard");
        sqlite3_close(legacy);
    }

    auto raw = [&] {
        sqlite3* db = nullptr;
        require(sqlite3_open_v2(url.c_str(), &db, SQLITE_OPEN_READWRITE, vc::objects::detail::sftpVfs()) == SQLITE_OK,
                "open locked remote SQLite");
        return db;
    };
    auto sql = [](sqlite3* db, const char* text) {
        const auto rc = sqlite3_exec(db, text, nullptr, nullptr, nullptr);
        if (rc != SQLITE_OK) throw std::runtime_error(std::string(text) + ": " + sqlite3_errmsg(db));
    };
    // Keep SQLite alive across independently locked server processes. Its
    // cache must survive an unchanged file, but see another writer's commit.
    {
        sqlite3* cached = nullptr;
        std::shared_ptr<const void> identity;
        auto value = [&](sqlite3* db) {
            sqlite3_stmt* query = nullptr;
            require(sqlite3_prepare_v2(db, "SELECT value FROM cache_probe", -1, &query, nullptr) == SQLITE_OK,
                    "prepare cached query");
            require(sqlite3_step(query) == SQLITE_ROW, "read cached query");
            const int result = sqlite3_column_int(query, 0);
            sqlite3_finalize(query);
            return result;
        };
        {
            RemoteHubScope scope(url);
            identity = vc::objects::detail::remoteHubCacheIdentity();
            cached = raw();
            sql(cached, "CREATE TABLE cache_probe(value); INSERT INTO cache_probe VALUES(1);");
            require(value(cached) == 1, "prime SQLite page cache");
        }
        const int probe = open(marker.c_str(), O_RDONLY);
        require(probe >= 0 && flock(probe, LOCK_EX | LOCK_NB) == 0, "idle cached SQLite holds no server lock");
        close(probe);
        {
            RemoteHubScope scope(url);
            require(identity == vc::objects::detail::remoteHubCacheIdentity(), "healthy cache identity reused");
            int misses = 0, high = 0;
            sqlite3_db_status(cached, SQLITE_DBSTATUS_CACHE_MISS, &misses, &high, 1);
            require(value(cached) == 1, "cached handle reopens under new server lock");
            sqlite3_db_status(cached, SQLITE_DBSTATUS_CACHE_MISS, &misses, &high, 0);
            require(misses == 0, "unchanged database keeps its SQLite pages");
        }
        std::exception_ptr writerFailure;
        {
            std::jthread writer([&] {
                try {
                    RemoteHubScope scope(url);
                    auto* db = raw();
                    sql(db, "UPDATE cache_probe SET value=2");
                    sqlite3_close(db);
                } catch (...) { writerFailure = std::current_exception(); }
            });
        }
        if (writerFailure) std::rethrow_exception(writerFailure);
        {
            RemoteHubScope scope(url);
            require(value(cached) == 2, "cached connection observes another writer's commit");
        }
        {
            std::jthread writer([&] {
                try {
                    sqlite3* db = nullptr;
                    {
                        RemoteHubScope scope(url);
                        db = raw();
                        sql(db, "PRAGMA cache_size=1; BEGIN IMMEDIATE; UPDATE cache_probe SET value=4; "
                                "CREATE TABLE cache_spill(x); WITH RECURSIVE n(x) AS "
                                "(VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<32) "
                                "INSERT INTO cache_spill SELECT zeroblob(4096) FROM n;");
                    }
                    sqlite3_close(db); // failed scope cannot roll back through a new session
                } catch (...) { writerFailure = std::current_exception(); }
            });
        }
        if (writerFailure) std::rethrow_exception(writerFailure);
        require(std::filesystem::file_size(path.string() + "-journal") > 512, "other writer left a hot journal");
        {
            RemoteHubScope scope(url);
            require(value(cached) == 2, "healthy cached reader recovers another writer's hot journal");
            require(!std::filesystem::exists(path.string() + "-journal"), "cached recovery removes the hot journal");
            sql(cached, "BEGIN IMMEDIATE; UPDATE cache_probe SET value=3;");
            // Deliberately let a transaction outlive its process lock.
        }
        {
            RemoteHubScope scope(url);
            require(identity != vc::objects::detail::remoteHubCacheIdentity(), "unfinished transaction invalidates cache");
            require(sqlite3_exec(cached, "COMMIT", nullptr, nullptr, nullptr) != SQLITE_OK,
                    "old transaction cannot resume under a different lock");
            sqlite3_close(cached);
            auto* db = raw();
            require(value(db) == 2, "abandoned cached transaction rolled back");
            sql(db, "DROP TABLE cache_probe");
            sqlite3_close(db);
        }
    }
    auto recovered = [&] {
        RemoteHubScope scope(url);
        auto* db = raw();
        // A dead writer must not leave the table or database damage behind.
        sql(db, "BEGIN; SELECT count(*) FROM sqlite_master;");
        sqlite3_stmt* statement = nullptr;
        require(sqlite3_prepare_v2(db, "SELECT count(*) FROM sqlite_master WHERE name='crash_probe'", -1, &statement, nullptr) == SQLITE_OK,
                "prepare recovery check");
        require(sqlite3_step(statement) == SQLITE_ROW && sqlite3_column_int(statement, 0) == 0, "crashed transaction rolled back");
        sqlite3_finalize(statement);
        require(sqlite3_prepare_v2(db, "PRAGMA integrity_check", -1, &statement, nullptr) == SQLITE_OK, "prepare integrity check");
        require(sqlite3_step(statement) == SQLITE_ROW && std::string(reinterpret_cast<const char*>(sqlite3_column_text(statement, 0))) == "ok",
                "database integrity after recovery");
        sqlite3_finalize(statement);
        sql(db, "COMMIT;");
        sqlite3_close(db);
    };
    const char* spill = "PRAGMA cache_size=4; BEGIN IMMEDIATE; CREATE TABLE crash_probe(x); "
                        "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<32) "
                        "INSERT INTO crash_probe SELECT zeroblob(4096) FROM n;";

    // Kill the client mid-write. No destructor or explicit unlock can run.
    int ready[2]; require(pipe(ready) == 0, "crash readiness pipe");
    const auto child = fork();
    require(child >= 0, "fork writer");
    if (child == 0) {
        close(ready[0]);
        try {
            RemoteHubScope scope(url);
            auto* db = raw();
            sql(db, spill);
            const char done = 'Y'; if (write(ready[1], &done, 1) != 1) _exit(3);
            for (;;) pause();
        } catch (...) { _exit(2); }
    }
    close(ready[1]);
    char done = 0;
    const auto got = read(ready[0], &done, 1);
    close(ready[0]);
    kill(child, SIGKILL);
    int status; waitpid(child, &status, 0);
    require(got == 1 && done == 'Y', "writer reached spilled transaction");
    require(std::filesystem::file_size(path.string() + "-journal") > 512, "hot journal survived client crash");
    recovered();

    // A server-enforced deadline kills a live but abandoned writer too.
    setenv("VC_HUB_SESSION_SECONDS", "2", 1);
    {
        RemoteHubScope scope(url);
        auto* db = raw();
        sql(db, spill);
        std::this_thread::sleep_for(std::chrono::milliseconds(2200));
        require(sqlite3_exec(db, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK, "expired session cannot commit");
        sqlite3_close(db);
    }
    unsetenv("VC_HUB_SESSION_SECONDS");
    recovered();

    // Lose a reply after execution. The same operation must never reconnect,
    // even for rollback/close; a fresh operation owns recovery instead.
    const auto trigger = root / "disconnect", log = root / "connections";
    setenv("VC_SFTP_TEST_DISCONNECT", trigger.c_str(), 1);
    setenv("VC_SFTP_TEST_CONNECTION_LOG", log.c_str(), 1);
    {
        RemoteHubScope scope(url);
        auto* db = raw();
        sql(db, spill);
        std::ofstream(trigger).put('x');
        require(sqlite3_exec(db, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK, "lost reply fails operation");
        require(sqlite3_exec(db, "SELECT * FROM sqlite_master;", nullptr, nullptr, nullptr) != SQLITE_OK, "lost session stays unusable");
        sqlite3_close(db);
    }
    unsetenv("VC_SFTP_TEST_DISCONNECT");
    unsetenv("VC_SFTP_TEST_CONNECTION_LOG");
    std::ifstream in(log);
    const std::string connections((std::istreambuf_iterator<char>(in)), {});
    require(connections == "locked\n", "no reconnect inside failed operation");
    recovered();

    // Normal latency does not strand a lock. Each reply waits 250 ms.
    setenv("VC_SFTP_TEST_LATENCY_MS", "250", 1);
    {
        RemoteHubScope scope(url);
        require(vc::objects::detail::remoteStat(url).exists, "locked session at 250 ms reply latency");
    }
    unsetenv("VC_SFTP_TEST_LATENCY_MS");
    recovered();

    if (!realStore.empty()) {
        const auto alicePath = root / "real-alice.vcobj", bobPath = root / "real-bob.vcobj";
        cloneHub(url, alicePath, as("alice"));
        cloneHub(url, bobPath, as("bob"));
        sqlite3* db = nullptr;
        require(sqlite3_open_v2(alicePath.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK, "read real replica");
        sqlite3_stmt* first = nullptr;
        require(sqlite3_prepare_v2(db, "SELECT lower(hex(uuid)) FROM objects WHERE kind='fiber' ORDER BY uuid LIMIT 1", -1, &first, nullptr) == SQLITE_OK,
                "select real fiber");
        require(sqlite3_step(first) == SQLITE_ROW, "real store has a fiber");
        const std::string hex(reinterpret_cast<const char*>(sqlite3_column_text(first, 0)));
        const auto id = hex.substr(0, 8) + "-" + hex.substr(8, 4) + "-" + hex.substr(12, 4) + "-" + hex.substr(16, 4) + "-" + hex.substr(20);
        sqlite3_finalize(first); sqlite3_close(db);
        auto alice = ObjectStore::open(alicePath, as("alice"));
        auto bob = ObjectStore::open(bobPath, as("bob"));
        acquireLeases(alice, {id});
        setNote(alice, id, "process lock validation");
        require(hubSync(alice).pushed == 1, "push real fiber metadata");
        require(hubPull(bob).pulled == 1 && meta(alice, id) == meta(bob, id), "real replicas converge");
        releaseLeases(alice, {id});
        std::cout << "real store: crash/reply-loss/deadline recovery, integrity and two-replica sync passed\n";
    }
}

// Optional real-data benchmark: each empty replica pulls the same new fibers
// through a cold hub handle. Setup and byte-for-byte validation are not timed.
void benchmarkPull(const std::filesystem::path& root, const std::filesystem::path& sourcePath,
                   int count, int iterations, int latencyMs)
{
    require(count > 0 && iterations > 0 && latencyMs >= 0, "invalid benchmark arguments");
    auto options = as("benchmark-source"); options.readOnly = true;
    auto source = ObjectStore::open(sourcePath, options);
    std::vector<ObjectCopy> copies;
    size_t payloadBytes = 0;
    {
        auto txn = source.read();
        auto records = source.objectsByKind(txn, "fiber");
        std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) { return a.uuid < b.uuid; });
        for (const auto& record : records) {
            if (!source.currentGeometry(txn, record.uuid)) continue;
            copies.push_back(readObjectCopy(source, txn, record.uuid, record.uuid));
            for (const auto& array : copies.back().geometry->arrays) payloadBytes += array.bytes.size();
            if (int(copies.size()) == count) break;
        }
    }
    require(int(copies.size()) == count, "not enough real fibers for benchmark");
    const auto path = root / "benchmark-hub.vcobj";
    { auto store = ObjectStore::create(path, as("owner")); store.checkpoint(); }
    initHub(path, "owner");
    const auto url = "sftp://benchmark@test-host" + sftp::encodePath(path.string());
    enableHubProcessLocking(url);
    std::vector<std::filesystem::path> replicas;
    for (int i = 0; i < iterations; ++i) {
        replicas.push_back(root / ("benchmark-" + std::to_string(i) + ".vcobj"));
        cloneHub(url, replicas.back(), as("reader-" + std::to_string(i)));
    }
    const auto publisherPath = root / "publisher.vcobj";
    cloneHub(url, publisherPath, as("publisher"));
    auto publisher = ObjectStore::open(publisherPath, as("publisher"));
    {
        auto txn = publisher.write();
        for (const auto& copy : copies) writeObjectCopy(publisher, txn, copy);
        txn.commit();
    }
    require(hubPush(publisher).pushed == count, "publish benchmark geometry");
    std::cout << "fibers=" << count << " payload_bytes=" << payloadBytes << " reply_delay_ms=" << latencyMs << '\n';
    for (int i = 0; i < iterations; ++i) {
        auto replica = ObjectStore::open(replicas[i], as("reader-" + std::to_string(i)));
        const auto log = root / ("replies-" + std::to_string(i));
        setenv("VC_SFTP_TEST_REPLY_LOG", log.c_str(), 1);
        setenv("VC_SFTP_TEST_LATENCY_MS", std::to_string(latencyMs).c_str(), 1);
        const auto start = std::chrono::steady_clock::now();
        require(hubPull(replica).pulled == count, "benchmark pull installs all fibers");
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        unsetenv("VC_SFTP_TEST_LATENCY_MS");
        unsetenv("VC_SFTP_TEST_REPLY_LOG");
        auto txn = replica.read();
        for (const auto& copy : copies) {
            const auto revision = replica.currentGeometry(txn, copy.record.uuid);
            require(bool(revision), "pulled fiber has geometry");
            const auto data = replica.readGeometryData(txn, revision->uuid);
            require(data.arrays.size() == copy.geometry->arrays.size(), "pulled array count");
            for (size_t j = 0; j < data.arrays.size(); ++j) {
                const auto& expected = copy.geometry->arrays[j];
                const auto& actual = data.arrays[j];
                require(actual.name == expected.name && actual.dtype == expected.dtype &&
                        actual.shape == expected.shape && actual.bytes == expected.bytes, "pulled geometry bytes unchanged");
            }
        }
        size_t replies = 0, reads = 0, received = 0, size = 0;
        int type = 0;
        std::ifstream packets(log);
        while (packets >> type >> size) { ++replies; reads += type == 103; received += size; }
        std::cout << "pull_seconds=" << seconds << " replies=" << replies << " data_replies=" << reads
                  << " data_bytes=" << received << std::endl;
    }
}

}  // namespace

int main(int argc, char** argv)
try {
    if (argc > 1 && std::string(argv[1]) == "-T") return fakeSsh(argc, argv);
    urls();
    {
        vc::objects::detail::HubLeaseCounts holds;
        holds.counts["held"] = 1;
        HubStatus status;
        status.replica = "self";
        status.leases.push_back({"held", "self", {}, {}, 0, 100});
        require(!holds.retain({"held", "missing"}, status, 50) && holds.counts.at("held") == 1,
                "mixed lease retention is all-or-nothing");
        require(!holds.retain({"held"}, status, 100) && !holds.retain({"held"}, status, 101),
                "expiring or expired lease requires the slow path");
        require(holds.retain({"held", "held"}, status, 99) && holds.counts.at("held") == 3,
                "duplicate valid holds are counted independently");
        status.leases.front().holder = "other";
        require(!holds.retain({"held"}, status, 50), "changed lease owner is not reused");
        status.leases.clear();
        require(!holds.retain({"held"}, status, 50), "released lease is not reused");
    }
    std::string server;
    for (const auto* candidate : {"/usr/lib/openssh/sftp-server", "/usr/lib/ssh/sftp-server", "/usr/libexec/sftp-server",
                                  "/usr/libexec/openssh/sftp-server"})
        if (std::filesystem::exists(candidate)) server = candidate;
    if (server.empty()) {
        std::cout << "object_store_sftp: no installed OpenSSH sftp-server; skipped\n";
        return 77;
    }
#ifdef __linux__
    if (std::system("command -v flock >/dev/null && command -v timeout >/dev/null") != 0) {
        std::cout << "object_store_sftp: process-lock fixture needs flock and timeout; skipped\n";
        return 77;
    }
#endif
    TemporaryDirectory directory;
    const auto bin = directory.path / "bin";
    std::filesystem::create_directory(bin);
    std::filesystem::create_symlink(std::filesystem::absolute(argv[0]), bin / "ssh");
    setenv("PATH", (bin.string() + ":" + std::getenv("PATH")).c_str(), 1);
    setenv("VC_SFTP_TEST_SERVER", server.c_str(), 1);
    if (argc == 6 && std::string(argv[1]) == "--benchmark-pull") {
        benchmarkPull(directory.path, argv[2], std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]));
        return 0;
    }
    session(directory.path);
    passwords(directory.path);
#ifdef __linux__
    readAhead(directory.path);
    hosting(directory.path);
    processLocks(directory.path, argc == 3 && std::string(argv[1]) == "--real-store" ? argv[2] : std::filesystem::path{});
#else
    std::cout << "object_store_sftp: process-lock fixture requires Linux flock/timeout; skipped\n";
    return 77;
#endif
    std::cout << "object_store_sftp: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
