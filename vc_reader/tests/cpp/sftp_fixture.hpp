// The test executable doubles as `ssh` on PATH and execs the installed OpenSSH
// sftp-server, so sftp:// reads need no network, sshd or credentials.
#pragma once
#include "vc_reader/sftp.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace sftp_fixture {
// Call first in main(): returns the exit status when running as `ssh`, else -1.
inline int run_as_ssh(int argc, char** argv)
{
    if (std::filesystem::path(argv[0]).filename() != "ssh")
        return -1;
    const std::vector<std::string> args(argv + 1, argv + argc);
    const auto has = [&](const std::string& value) { return std::find(args.begin(), args.end(), value) != args.end(); };
    if (!has("-s") || !has("--") || args.back() != "sftp" || !has("test-host"))
        return 42;
    const char* server = std::getenv("VCR_SFTP_TEST_SERVER");
    execl(server, server, "-e", static_cast<char*>(nullptr));
    return 43;
}

// Puts this executable on PATH as `ssh`. False when no sftp-server is installed.
inline bool install(const std::filesystem::path& work)
{
    const char* server = nullptr;
    for (const auto* candidate : {"/usr/lib/openssh/sftp-server", "/usr/lib/ssh/sftp-server", "/usr/libexec/sftp-server",
                                  "/usr/libexec/openssh/sftp-server"})
        if (std::filesystem::exists(candidate))
            server = candidate;
    if (!server)
        return false;
    std::filesystem::create_directories(work / "bin");
    std::filesystem::create_symlink(std::filesystem::canonical("/proc/self/exe"), work / "bin" / "ssh");
    setenv("PATH", ((work / "bin").string() + ":" + std::getenv("PATH")).c_str(), 1);
    setenv("VCR_SFTP_TEST_SERVER", server, 1);
    return true;
}

inline std::string url(const std::filesystem::path& path)
{
    return "sftp://test-host" + vcr::sftp::encodePath(path.string());
}
}  // namespace sftp_fixture
