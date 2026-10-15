#include <doctest/doctest.h>
#include "utils/sftp_fetch.hpp"
#include "utils/zarr.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include "vc/core/util/RemoteFileCache.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include <QCoreApplication>
#include <QTemporaryDir>
#include <QFile>
#include <QDir>
#include <QUrl>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <thread>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace
{
using B = std::string;
void integer(B& b, uint32_t n)
{
    for (int i = 24; i >= 0; i -= 8)
        b += char(n >> i);
}
uint32_t readInteger(const B& b, size_t& p)
{
    if (p + 4 > b.size())
        throw std::runtime_error("short packet");
    uint32_t n = 0;
    for (int i = 0; i < 4; ++i)
        n = (n << 8) | uint8_t(b[p++]);
    return n;
}
void string(B& b, const B& s)
{
    integer(b, uint32_t(s.size()));
    b += s;
}
B string(const B& b, size_t& p)
{
    const auto n = readInteger(b, p);
    if (p + n > b.size())
        throw std::runtime_error("short string");
    const auto s = b.substr(p, n);
    p += n;
    return s;
}
void packet(const B& b)
{
    B h;
    integer(h, b.size());
    std::cout.write(h.data(), h.size());
    std::cout.write(b.data(), b.size());
    std::cout.flush();
}
B reply(int type, uint32_t id)
{
    B b(1, char(type));
    integer(b, id);
    return b;
}
void status(uint32_t id, uint32_t code)
{
    auto b = reply(101, id);
    integer(b, code);
    string(b, "fixture status");
    string(b, "");
    packet(b);
}
void attrs(B& b, bool dir = false, uint32_t size = 100000)
{
    integer(b, 5);
    integer(b, 0);
    integer(b, size);
    integer(b, dir ? 0040755 : 0100644);
}

// Test executable doubles as a fake ssh on PATH. It speaks SFTP v3, without
// network/login requirements, and checks the actual production argv contract.
int server(int argc, char** argv)
{
#ifndef _WIN32
    if (auto* executable = std::getenv("VC_SFTP_REAL_SERVER")) {
        execl(executable, executable, "-e", static_cast<char*>(nullptr));
        return 43;
    }
#endif
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    std::vector<std::string> args(argv + 1, argv + argc);
    auto has = [&](const std::string& v) { return std::find(args.begin(), args.end(), v) != args.end(); };
    if (!has("-oBatchMode=yes") || !has("-oStrictHostKeyChecking=yes") || !has("-s") || args.back() != "sftp" || !has("test-alias"))
        return 42;
    if (has("-l") != has("alice") || has("-p") != has("2222"))
        return 44;
    if (auto* path = std::getenv("VC_SFTP_TEST_STARTS")) {
        std::ofstream f(path, std::ios::app);
        f << "start\n";
    }
    bool listed = false;
    std::vector<B> delayed;
    for (;;) {
        B header(4, '\0');
        std::cin.read(header.data(), 4);
        if (!std::cin)
            return 0;
        size_t p = 0;
        auto n = readInteger(header, p);
        B b(n, '\0');
        std::cin.read(b.data(), n);
        if (!std::cin)
            return 2;
        p = 1;
        const auto type = uint8_t(b[0]);
        if (type == 1) {
            B out(1, char(2));
            integer(out, 3);
            packet(out);
            continue;
        }
        const auto id = readInteger(b, p);
        const auto path = string(b, p);
        if (type == 17) {
            if (path == "/missing") {
                status(id, 2);
                continue;
            }
            if (path == "/denied") {
                status(id, 3);
                continue;
            }
            if (path == "/broken")
                return 1;
            if (path == "/stall")
                std::this_thread::sleep_for(std::chrono::seconds(10));
            if (path == "/malformed") {
                B header;
                integer(header, 0xffffffffu);
                std::cout.write(header.data(), header.size());
                std::cout.flush();
                continue;
            }
            auto out = reply(105, id);
            if (path.find("/unknown-type/") == 0)
                integer(out, 0);
            else
                attrs(out, path.find("zarr") != std::string::npos, path == "/large" ? 2 * 1024 * 1024 : 100000);
            packet(out);
        } else if (type == 3 || type == 11) {
            listed = false;
            auto out = reply(102, id);
            string(out, path);
            packet(out);
        } else if (type == 4)
            status(id, 0);
        else if (type == 5) {
            const uint64_t hi = readInteger(b, p);
            const uint64_t offset = (hi << 32) | readInteger(b, p);
            const auto count = readInteger(b, p);
            const auto actual = path == "/short" && count > 1000 ? count / 2 : count;
            B data(actual, '\0');
            for (size_t i = 0; i < actual; ++i)
                data[i] = char((offset + i) % 251);
            auto out = reply(103, id);
            string(out, data);
            if (path == "/reverse") {
                delayed.push_back(out);
                if (delayed.size() == 4) {
                    for (auto it = delayed.rbegin(); it != delayed.rend(); ++it)
                        packet(*it);
                    delayed.clear();
                }
            } else
                packet(out);
        } else if (type == 12) {
            if (listed) {
                status(id, 1);
                continue;
            }
            listed = true;
            auto out = reply(104, id);
            integer(out, 3);
            for (const auto& name : {".", "volume space.zarr", "file#%.json"}) {
                string(out, name);
                string(out, "ignored longname");
                if (path == "/no-permissions" || path == "/unknown-type")
                    integer(out, 0);
                else if (path == "/no-type-bits") {
                    integer(out, 4);
                    integer(out, 0755);
                } else
                    attrs(out, std::string(name).find("zarr") != std::string::npos);
            }
            packet(out);
        } else
            status(id, 8);
    }
}
}  // namespace

TEST_CASE("SSH URLs canonicalize and preserve source identity")
{
    CHECK(utils::canonical_sftp_url("ssh://alice@test-alias:2222/a%20b.zarr") == "sftp://alice@test-alias:2222/a%20b.zarr");
    CHECK(utils::is_sftp_url("SFTP://host/x"));
    CHECK(vc::project::isLocationRemote("ssh://host/x.zarr"));
    const auto spec = vc::parseRemoteVolumeSpec("ssh://host/x.zarr#vc-base-scale=2");
    CHECK(spec.sourceUrl == "sftp://host/x.zarr");
    CHECK(spec.baseScaleLevel == 2);
    CHECK_FALSE(spec.useAwsSigv4);
    for (const auto* bad : {"sftp://user:password@host/x", "sftp://host/x?token=x", "sftp://-host/x", "sftp:///x", "sftp://host/a%00b"})
        CHECK_THROWS(utils::canonical_sftp_url(bad));
    CHECK(vc::core::util::remoteFileCachePath("ssh://alice@host:22/a") == vc::core::util::remoteFileCachePath("sftp://alice@host:22/a"));
    CHECK(vc::core::util::remoteFileCachePath("sftp://alice@host:22/a") != vc::core::util::remoteFileCachePath("sftp://bob@host:22/a"));
    CHECK(vc::core::util::remoteFileCachePath("sftp://host:22/a") != vc::core::util::remoteFileCachePath("sftp://host:2222/a"));
}

TEST_CASE("Persistent SFTP reads ranges listing missing and errors")
{
    QTemporaryDir tmp(QDir::currentPath() + "/sftp-test-XXXXXX");
    REQUIRE(tmp.isValid());
#ifdef _WIN32
    const auto ssh = tmp.filePath("ssh.exe");
    REQUIRE(QFile::copy(QCoreApplication::applicationFilePath(), ssh));
#else
    const auto ssh = tmp.filePath("ssh");
    REQUIRE(QFile::link(QCoreApplication::applicationFilePath(), ssh));
#endif
    const auto previous = qgetenv("PATH");
    struct Restore {
        QByteArray path;
        ~Restore()
        {
            qputenv("PATH", path);
            qunsetenv("VC_SFTP_TEST_STARTS");
        }
    } restore{previous};
    qputenv(
        "PATH",
        tmp.path().toUtf8() + QDir::listSeparator().toLatin1() + QCoreApplication::applicationDirPath().toUtf8() +
            QDir::listSeparator().toLatin1() + previous);
    const auto starts = tmp.filePath("starts");
    qputenv("VC_SFTP_TEST_STARTS", starts.toUtf8());
    utils::HttpClient::Config config;
    config.max_retries = 0;
    utils::HttpClient client(config);
    const std::string base = "sftp://test-alias";
    CHECK(client.head(base + "/data").content_length == 100000);
    size_t observed = 0;
    {
        utils::HttpClient::ScopedDownloadObserver observer([&](size_t n) { observed += n; });
        for (const auto* path : {"/data", "/reverse", "/short"}) {
            const auto r = client.get(base + path);
            REQUIRE_MESSAGE(r.ok(), r.error_message);
            REQUIRE(r.body.size() == 100000);
            bool correct = true;
            for (size_t i = 0; i < r.body.size(); ++i)
                correct &= r.body[i] == std::byte(i % 251);
            CHECK(correct);
        }
    }
    CHECK(observed == 300000);
    const auto large = client.get(base + "/large");
    REQUIRE(large.ok());
    REQUIRE(large.body.size() == 2 * 1024 * 1024);
    bool correct = true;
    for (size_t i = 0; i < large.body.size(); ++i)
        correct &= large.body[i] == std::byte(i % 251);
    CHECK(correct);
    const auto range = client.get_range(base + "/data", 73, 500);
    REQUIRE(range.ok());
    REQUIRE(range.body.size() == 500);
    CHECK(range.body[0] == std::byte(73));
    CHECK(client.get(base + "/missing").not_found());
    const auto denied = client.get(base + "/denied");
    CHECK(denied.status_code == 403);
    CHECK_FALSE(denied.not_found());
    CHECK_THROWS(vc::httpGetString(base + "/denied"));
    utils::HttpStore store(base);
    CHECK_THROWS(store.exists("denied"));
    CHECK_THROWS(store.get_if_exists("denied"));
    auto entries = utils::list_sftp_directory(base + "/");
    REQUIRE(entries.size() == 2);
    CHECK(entries[0].directory);
    CHECK(entries[0].url == base + "/volume%20space.zarr/");
    CHECK(entries[1].url == base + "/file%23%25.json");
    QFile f(starts);
    REQUIRE(f.open(QIODevice::ReadOnly));
    CHECK(f.readAll() == "start\n");
    f.close();
    for (const auto* path : {"/no-permissions", "/no-type-bits"}) {
        const auto listing = utils::list_sftp_directory(base + path);
        REQUIRE(listing.size() == 2);
        CHECK(listing[0].directory);
        CHECK(listing[0].url == base + path + "/volume%20space.zarr/");
        CHECK_FALSE(listing[1].directory);
    }
    CHECK_FALSE(client.get(base + "/broken").ok());
    CHECK(client.get(base + "/data").ok());
    REQUIRE(f.open(QIODevice::ReadOnly));
    CHECK(f.readAll() == "start\nstart\n");
    f.close();
    CHECK_THROWS_AS(utils::list_sftp_directory(base + "/unknown-type"), std::runtime_error);
    CHECK_THROWS(client.put(base + "/data", {}));
    CHECK_FALSE(client.get(base + "/malformed").ok());
    {
        std::jthread cancel([] {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            utils::HttpClient::abortAll();
        });
        const auto interrupted = client.get(base + "/stall");
        CHECK_FALSE(interrupted.ok());
        CHECK(interrupted.error_message.find("cancelled") != std::string::npos);
    }
    utils::HttpClient::resetAbort();
    const auto explicitLogin = "sftp://alice@test-alias:2222/data";
    CHECK(client.head(explicitLogin).ok());
    utils::HttpClient::abortAll();
    CHECK_FALSE(client.get(explicitLogin).ok());
    CHECK_FALSE(client.get(base + "/data").ok());
    utils::HttpClient::resetAbort();
}

#ifndef _WIN32
TEST_CASE("Installed OpenSSH SFTP server serves a real Zarr without network")
{
    QString executable;
    for (const auto* candidate : {"/usr/lib/ssh/sftp-server", "/usr/lib/openssh/sftp-server", "/usr/libexec/sftp-server"})
        if (QFile::exists(candidate)) {
            executable = candidate;
            break;
        }
    if (executable.isEmpty()) {
        MESSAGE("No installed sftp-server; protocol fixture still covers transport");
        return;
    }
    QTemporaryDir tmp(QDir::currentPath() + "/sftp-real-test-XXXXXX");
    REQUIRE(tmp.isValid());
    REQUIRE(QFile::link(QCoreApplication::applicationFilePath(), tmp.filePath("ssh")));
    const auto previous = qgetenv("PATH");
    struct Restore {
        QByteArray path;
        ~Restore()
        {
            utils::HttpClient::abortAll();
            (void)utils::HttpClient{}.get("sftp://real-server-fixture/cleanup");
            utils::HttpClient::resetAbort();
            qputenv("PATH", path);
            qunsetenv("VC_SFTP_REAL_SERVER");
        }
    } restore{previous};
    qputenv("PATH", tmp.path().toUtf8() + ":" + previous);
    qputenv("VC_SFTP_REAL_SERVER", executable.toUtf8());
    REQUIRE(QDir().mkpath(tmp.filePath("volume.zarr/0")));
    auto write = [&](const QString& name, const QByteArray& data) {
        QFile file(tmp.filePath(name));
        REQUIRE(file.open(QIODevice::WriteOnly));
        REQUIRE(file.write(data) == data.size());
    };
    write("volume.zarr/.zgroup", R"({"zarr_format":2})");
    write("volume.zarr/.zattrs", R"({"multiscales":[{"datasets":[{"path":"0"}]}]})");
    write("volume.zarr/0/.zarray", R"({"zarr_format":2,"shape":[2,2,2],"chunks":[2,2,2],"dtype":"|u1","compressor":null,"fill_value":0,"order":"C","filters":null})");
    write("volume.zarr/0/0.0.0", "12345678");
    write("data space#%.bin", QByteArray(100000, 'x'));
    QUrl root;
    root.setScheme("sftp");
    root.setHost("real-server-fixture");
    root.setPath(tmp.path());
    const auto base = root.toEncoded().toStdString();
    utils::HttpClient client;
    const auto data = client.get(base + "/data%20space%23%25.bin");
    REQUIRE_MESSAGE(data.ok(), data.error_message);
    CHECK(data.body.size() == 100000);
    const auto range = client.get_range(base + "/data%20space%23%25.bin", 789, 50000);
    REQUIRE(range.ok());
    CHECK(range.body.size() == 50000);
    CHECK(client.head(base + "/missing").not_found());
    const auto entries = utils::list_sftp_directory(base);
    CHECK(std::any_of(entries.begin(), entries.end(), [](const auto& e) { return e.directory && e.name == "volume.zarr"; }));
    auto opened = vc::render::openRemoteZarrPyramid(base + "/volume.zarr", {});
    REQUIRE(opened.opened.shapes.size() == 1);
    CHECK(opened.opened.shapes[0][0] == 2);
    const auto chunk = client.get(base + "/volume.zarr/0/0.0.0");
    CHECK(chunk.body_string() == "12345678");
}
#endif

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "-T")
        return server(argc, argv);
    QCoreApplication app(argc, argv);
    return doctest::detail::runAll();
}
