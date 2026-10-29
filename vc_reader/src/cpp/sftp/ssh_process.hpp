#pragma once
#include "vc_reader/sftp.hpp"

namespace vcr::sftp::detail {

// The `ssh -s HOST sftp` child process and its byte stream. With an
// authenticator (Linux/macOS), ssh's prompts arrive through SSH_ASKPASS and are
// answered on a background thread until authenticated() is called.
class SshProcess {
public:
    SshProcess(const Url& destination, const Options& options, Authenticator authenticator);
    ~SshProcess();
    SshProcess(const SshProcess&) = delete;
    SshProcess& operator=(const SshProcess&) = delete;

    // `check` runs at least every 100 ms while waiting and may throw.
    void write(std::span<const std::byte> data, const std::function<void()>& check);
    // Waits up to `wait` for data; returns 0 when none arrived. Throws when ssh
    // has exited or closed the stream.
    size_t read(std::span<std::byte> out, std::chrono::milliseconds wait);
    bool prompting() const;  // a prompt is waiting for the user's answer
    void authenticated();    // stop answering prompts; the session is established
    std::string diagnostics();  // the end of ssh's standard error

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vcr::sftp::detail
