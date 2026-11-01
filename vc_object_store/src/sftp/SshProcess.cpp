#include "sftp/SshProcess.hpp"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>
extern char** environ;
#endif

namespace vc::sftp {
namespace detail {
namespace {
constexpr size_t kDiagnosticBytes = 8192;
// One drain reads at most this much, so a chatty stderr cannot keep callers from
// their timeout checks; only the last kDiagnosticBytes are kept anyway.
constexpr size_t kDrainBytes = 4 * kDiagnosticBytes;

#ifndef _WIN32
std::string controlPath()
{
    // Keep the socket path short enough for macOS too. mkdtemp creates mode
    // 0700, so only this user can connect to the authenticated transport.
    struct Directory {
        std::string path = "/tmp/vc-ssh-XXXXXX";
        Directory() {
            if (!mkdtemp(path.data())) throw std::runtime_error("SFTP: cannot create SSH control directory");
        }
        ~Directory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
    };
    static Directory directory;
    return directory.path + "/%C";
}
#endif

std::string shellQuote(const std::string& text)
{
    std::string quoted = "'";
    for (const char c : text) quoted += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return quoted + "'";
}

std::string lockedServerCommand(const Options& options)
{
    if (options.processLifetime.count() <= 0)
        throw std::invalid_argument("SFTP process lifetime must be positive");
    // No user-supplied text is interpreted as shell syntax. All descriptors
    // for the database and its journal belong to this one locked process.
    return "lock=" + shellQuote(options.processLock) + "; "
        "test -f \"$lock\" && test ! -L \"$lock\" || { "
        "echo 'hub: enable SSH process locking with vc_object_hub enable-ssh-locking HUB first' >&2; exit 1; }; "
        "command -v flock >/dev/null && command -v timeout >/dev/null || { "
        "echo 'hub: server needs flock and timeout (Linux util-linux/coreutils)' >&2; exit 1; }; "
        "server=; for p in /usr/lib/openssh/sftp-server /usr/lib/ssh/sftp-server "
        "/usr/libexec/openssh/sftp-server /usr/libexec/sftp-server; do "
        "if test -x \"$p\"; then server=$p; break; fi; done; "
        "test -n \"$server\" || { echo 'hub: external OpenSSH sftp-server not found' >&2; exit 1; }; "
        "exec timeout --kill-after=5s " + std::to_string(options.processLifetime.count()) +
        "s flock --exclusive --timeout 60 --no-fork \"$lock\" \"$server\"";
}

std::vector<std::string> sshArguments(const Url& url, const Options& options, bool interactive)
{
    std::vector<std::string> args{"ssh", "-T", interactive ? "-oBatchMode=no" : "-oBatchMode=yes",
        interactive ? "-oStrictHostKeyChecking=ask" : "-oStrictHostKeyChecking=yes", "-oClearAllForwardings=yes",
        "-oConnectTimeout=" + std::to_string(options.connectTimeout.count())};
    if (!url.user.empty()) args.insert(args.end(), {"-l", url.user});
    if (url.port) args.insert(args.end(), {"-p", std::to_string(*url.port)});
#ifndef _WIN32
    if (options.shareConnection)
        args.insert(args.end(), {"-oControlMaster=auto", "-oControlPersist=no", "-oControlPath=" + controlPath()});
#endif
    if (options.processLock.empty()) args.insert(args.end(), {"-s", "--", url.host, "sftp"});
    else args.insert(args.end(), {"--", url.host, lockedServerCommand(options)});
    return args;
}

void keepTail(std::string& tail, const char* data, size_t size)
{
    tail.append(data, size);
    if (tail.size() > kDiagnosticBytes) tail.erase(0, tail.size() - kDiagnosticBytes);
}

#ifndef _WIN32
// Answers given for a destination's prompts, reused when a later ssh process
// asks the same question so reconnecting does not ask the user again.
std::mutex answersMutex;
std::map<std::pair<std::string, std::string>, std::string> rememberedAnswers;

// ssh runs a generated helper as SSH_ASKPASS. The helper writes the prompt kind
// and text to one FIFO and prints the answer it reads from another; both live in
// a private 0700 directory, so secrets never appear in arguments or environment.
class Askpass {
public:
    Askpass(Authenticator authenticator, std::string destination)
        : authenticator_(std::move(authenticator)), destination_(std::move(destination))
    {
        const char* temp = std::getenv("TMPDIR");
        std::string pattern = std::string(temp && *temp ? temp : "/tmp") + "/vc-sftp-XXXXXX";
        if (!mkdtemp(pattern.data())) throw std::runtime_error("SFTP askpass: cannot create a private directory");
        dir_ = pattern;
        request_ = dir_ + "/request";
        response_ = dir_ + "/response";
        script_ = dir_ + "/askpass";
        try {
            if (mkfifo(request_.c_str(), 0600) != 0 || mkfifo(response_.c_str(), 0600) != 0)
                throw std::runtime_error("SFTP askpass: cannot create FIFOs");
            const auto body = "#!/bin/sh\nd=" + shellQuote(dir_) + "\n"
                "printf '%s\\n%s\\000' \"${SSH_ASKPASS_PROMPT:-}\" \"$1\" > \"$d/request\" || exit 1\n"
                "r=$(cat \"$d/response\") || exit 1\n"
                "case $r in\n1*) printf '%s\\n' \"${r#1}\" ;;\n*) exit 1 ;;\nesac\n";
            const int fd = ::open(script_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0700);
            if (fd < 0) throw std::runtime_error("SFTP askpass: cannot write the helper");
            const bool written = ::write(fd, body.data(), body.size()) == ssize_t(body.size());
            ::close(fd);
            if (!written) throw std::runtime_error("SFTP askpass: cannot write the helper");
            // Holding both ends keeps the FIFO open between helper invocations.
            requestFd_ = ::open(request_.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
            if (requestFd_ < 0) throw std::runtime_error("SFTP askpass: cannot open the request FIFO");
        } catch (...) {
            removeFiles();
            throw;
        }
        thread_ = std::thread([this] { run(); });
    }
    ~Askpass()
    {
        stop_ = true;
        thread_.join();
        removeFiles();
    }
    const std::string& program() const { return script_; }
    bool active() const { return active_; }

private:
    void removeFiles()
    {
        if (requestFd_ >= 0) ::close(requestFd_);
        for (const auto& path : {script_, request_, response_}) if (!path.empty()) ::unlink(path.c_str());
        if (!dir_.empty()) ::rmdir(dir_.c_str());
    }
    void run()
    {
        std::string buffer;
        char chunk[4096];
        while (!stop_) {
            pollfd ready{requestFd_, POLLIN, 0};
            if (poll(&ready, 1, 100) <= 0) continue;
            const auto n = ::read(requestFd_, chunk, sizeof chunk);
            if (n <= 0) continue;
            buffer.append(chunk, size_t(n));
            for (size_t end; (end = buffer.find('\0')) != std::string::npos;) {
                const auto record = buffer.substr(0, end);
                buffer.erase(0, end + 1);
                const auto newline = record.find('\n');
                const auto kind = record.substr(0, newline);
                auto text = newline == std::string::npos ? std::string() : record.substr(newline + 1);
                while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
                respond(answer(kind, text));
            }
        }
    }
    std::optional<std::string> answer(const std::string& kind, const std::string& text)
    {
        if (kind == "none") return std::string();  // a notification, not a question
        Prompt prompt{destination_, text, kind == "confirm" || text.find("(yes/no") != std::string::npos};
        const auto key = std::pair(destination_, text);
        if (!prompt.confirmation) {
            std::lock_guard lock(answersMutex);
            // The same question twice in one process means the answer was wrong.
            if (!asked_.insert(text).second) rememberedAnswers.erase(key);
            else if (const auto found = rememberedAnswers.find(key); found != rememberedAnswers.end()) return found->second;
        }
        std::optional<std::string> reply;
        active_ = true;
        try { reply = authenticator_(prompt); } catch (...) { reply.reset(); }
        active_ = false;
        if (reply && !prompt.confirmation) {
            std::lock_guard lock(answersMutex);
            rememberedAnswers[key] = *reply;
        }
        return reply;
    }
    void respond(const std::optional<std::string>& reply)
    {
        // The helper opens the response FIFO right after writing its request.
        int fd = -1;
        for (int attempt = 0; attempt < 1000 && fd < 0 && !stop_; ++attempt) {
            fd = ::open(response_.c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC);
            if (fd < 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (fd < 0) return;
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) & ~O_NONBLOCK);
        const auto message = reply ? "1" + *reply : std::string("0");
        (void)!::write(fd, message.data(), message.size());
        ::close(fd);
    }

    Authenticator authenticator_;
    std::string destination_, dir_, request_, response_, script_;
    int requestFd_ = -1;
    std::set<std::string> asked_;
    std::atomic<bool> stop_{false}, active_{false};
    std::thread thread_;
};
#endif
}  // namespace

#ifdef _WIN32
namespace {
std::wstring widen(const std::string& text)
{
    if (text.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, text.data(), int(text.size()), nullptr, 0);
    std::wstring wide(size_t(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), int(text.size()), wide.data(), n);
    return wide;
}
// CommandLineToArgvW quoting.
std::wstring quoteArgument(const std::wstring& argument)
{
    if (!argument.empty() && argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) return argument;
    std::wstring quoted = L"\"";
    for (auto it = argument.begin();; ++it) {
        size_t slashes = 0;
        while (it != argument.end() && *it == L'\\') { ++it; ++slashes; }
        if (it == argument.end()) { quoted.append(slashes * 2, L'\\'); break; }
        if (*it == L'"') quoted.append(slashes * 2 + 1, L'\\');
        else quoted.append(slashes, L'\\');
        quoted += *it;
    }
    return quoted + L"\"";
}
}  // namespace

// ssh's standard input. Anonymous pipes cannot be written asynchronously, so a
// stalled ssh would block WriteFile with no way to run `check`; our end of this
// named pipe is overlapped instead, and the child's end is an ordinary handle.
static bool createInputPipe(HANDLE& child, HANDLE& parent, SECURITY_ATTRIBUTES& inherit)
{
    static std::atomic<unsigned> serial{0};
    const auto name = L"\\\\.\\pipe\\vc-ssh-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
        std::to_wstring(serial.fetch_add(1));
    parent = CreateNamedPipeW(name.c_str(), PIPE_ACCESS_OUTBOUND | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
                              PIPE_TYPE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1, 1 << 16, 1 << 16, 0, nullptr);
    if (parent == INVALID_HANDLE_VALUE) { parent = nullptr; return false; }
    child = CreateFileW(name.c_str(), GENERIC_READ, 0, &inherit, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (child == INVALID_HANDLE_VALUE) { child = nullptr; CloseHandle(parent); parent = nullptr; return false; }
    return true;
}

struct SshProcess::Impl {
    HANDLE process = nullptr, input = nullptr, output = nullptr, errors = nullptr;
    HANDLE written = CreateEventW(nullptr, TRUE, FALSE, nullptr);  // completes overlapped input writes
    std::string tail;
    ~Impl() { if (written) CloseHandle(written); }
    void drainErrors()
    {
        char buffer[4096];
        DWORD available = 0;
        for (size_t drained = 0; drained < kDrainBytes && errors &&
             PeekNamedPipe(errors, nullptr, 0, nullptr, &available, nullptr) && available;) {
            DWORD n = 0;
            if (!ReadFile(errors, buffer, std::min<DWORD>(available, sizeof buffer), &n, nullptr) || !n) break;
            keepTail(tail, buffer, n);
            drained += n;
        }
    }
};

SshProcess::SshProcess(const Url& destination, const Options& options, Authenticator)
    : impl_(std::make_unique<Impl>())
{
    // Windows has no SSH_ASKPASS integration here: ssh runs in batch mode.
    std::wstring command;
    for (const auto& argument : sshArguments(destination, options, false))
        command += (command.empty() ? L"" : L" ") + quoteArgument(widen(argument));
    SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE childInput = nullptr, childOutput = nullptr, childErrors = nullptr;
    if (!impl_->written || !createInputPipe(childInput, impl_->input, inherit) ||
        !CreatePipe(&impl_->output, &childOutput, &inherit, 0) || !CreatePipe(&impl_->errors, &childErrors, &inherit, 0)) {
        for (auto handle : {childInput, childOutput, childErrors, impl_->input, impl_->output, impl_->errors})
            if (handle) CloseHandle(handle);
        throw std::runtime_error("Cannot create pipes for OpenSSH ssh");
    }
    for (auto handle : {impl_->input, impl_->output, impl_->errors}) SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW startup{};
    startup.cb = sizeof startup;
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = childInput;
    startup.hStdOutput = childOutput;
    startup.hStdError = childErrors;
    PROCESS_INFORMATION info{};
    const bool started = CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr,
                                        &startup, &info);
    for (auto handle : {childInput, childOutput, childErrors}) CloseHandle(handle);
    if (!started) {
        const auto error = GetLastError();
        for (auto handle : {impl_->input, impl_->output, impl_->errors}) CloseHandle(handle);
        throw std::runtime_error("Cannot start OpenSSH ssh (error " + std::to_string(error) + ")");
    }
    CloseHandle(info.hThread);
    impl_->process = info.hProcess;
}

SshProcess::~SshProcess()
{
    CloseHandle(impl_->input);
    if (WaitForSingleObject(impl_->process, 1000) != WAIT_OBJECT_0) TerminateProcess(impl_->process, 1);
    for (auto handle : {impl_->output, impl_->errors, impl_->process}) CloseHandle(handle);
}

void SshProcess::write(std::span<const std::byte> data, const std::function<void()>& check)
{
    for (size_t done = 0; done < data.size();) {
        check();
        OVERLAPPED pending{};
        pending.hEvent = impl_->written;
        ResetEvent(impl_->written);
        DWORD n = 0;
        if (!WriteFile(impl_->input, data.data() + done, DWORD(std::min<size_t>(data.size() - done, 1 << 20)), nullptr,
                       &pending)) {
            if (GetLastError() != ERROR_IO_PENDING) throw std::runtime_error("SSH write failed: " + diagnostics());
            // Wait as the POSIX path does: `check` every 100 ms, and it may throw.
            while (WaitForSingleObject(impl_->written, 100) == WAIT_TIMEOUT) {
                try {
                    impl_->drainErrors();
                    check();
                } catch (...) {
                    // The kernel still owns `pending` and the buffer until the write ends.
                    CancelIoEx(impl_->input, &pending);
                    GetOverlappedResult(impl_->input, &pending, &n, TRUE);
                    throw;
                }
            }
        }
        if (!GetOverlappedResult(impl_->input, &pending, &n, TRUE))
            throw std::runtime_error("SSH write failed: " + diagnostics());
        done += n;
    }
}

size_t SshProcess::read(std::span<std::byte> out, std::chrono::milliseconds wait)
{
    const auto deadline = std::chrono::steady_clock::now() + wait;
    for (;;) {
        impl_->drainErrors();
        DWORD available = 0;
        if (!PeekNamedPipe(impl_->output, nullptr, 0, nullptr, &available, nullptr))
            throw std::runtime_error("SSH connection closed: " + diagnostics());
        if (available) {
            DWORD n = 0;
            if (!ReadFile(impl_->output, out.data(), DWORD(std::min<size_t>(available, out.size())), &n, nullptr))
                throw std::runtime_error("SSH connection closed: " + diagnostics());
            return n;
        }
        if (std::chrono::steady_clock::now() >= deadline) return 0;
        Sleep(5);
    }
}

bool SshProcess::prompting() const { return false; }
bool SshProcess::closed() const
{
    DWORD status = 0;
    return !GetExitCodeProcess(impl_->process, &status) || status != STILL_ACTIVE;
}
void SshProcess::authenticated() {}
std::string SshProcess::diagnostics()
{
    impl_->drainErrors();
    return impl_->tail;
}
#else
struct SshProcess::Impl {
    pid_t pid = -1;
    int stream = -1, errors = -1;
    std::string tail;
    std::unique_ptr<Askpass> askpass;
    void drainErrors()
    {
        char buffer[4096];
        for (size_t drained = 0; errors >= 0 && drained < kDrainBytes;) {
            const auto n = ::read(errors, buffer, sizeof buffer);
            if (n > 0) { keepTail(tail, buffer, size_t(n)); drained += size_t(n); continue; }
            if (n == 0) { ::close(errors); errors = -1; }
            break;
        }
    }
};

namespace {
// Child descriptors must not be 0-2 themselves, or dup2 then close would undo them.
int aboveStandard(int fd)
{
    if (fd > 2) return fd;
    const int moved = fcntl(fd, F_DUPFD_CLOEXEC, 3);
    ::close(fd);
    return moved;
}
}  // namespace

SshProcess::SshProcess(const Url& destination, const Options& options, Authenticator authenticator)
    : impl_(std::make_unique<Impl>())
{
    const bool interactive = static_cast<bool>(authenticator);
    if (interactive) impl_->askpass = std::make_unique<Askpass>(std::move(authenticator), destination.authority());
    const auto arguments = sshArguments(destination, options, interactive);
    std::vector<std::string> environment;
    for (char** entry = environ; entry && *entry; ++entry) {
        const std::string_view value(*entry);
        if (!value.starts_with("SSH_ASKPASS=") && !value.starts_with("SSH_ASKPASS_REQUIRE=")) environment.emplace_back(value);
    }
    if (interactive) {
        environment.push_back("SSH_ASKPASS=" + impl_->askpass->program());
        environment.push_back("SSH_ASKPASS_REQUIRE=force");
    }
    std::vector<char*> argv, envp;
    for (const auto& argument : arguments) argv.push_back(const_cast<char*>(argument.c_str()));
    for (const auto& entry : environment) envp.push_back(const_cast<char*>(entry.c_str()));
    argv.push_back(nullptr);
    envp.push_back(nullptr);

    int pair[2], errors[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) throw std::runtime_error("Cannot create a socket for OpenSSH ssh");
    if (pipe(errors) != 0) {
        ::close(pair[0]); ::close(pair[1]);
        throw std::runtime_error("Cannot create a pipe for OpenSSH ssh");
    }
    pair[1] = aboveStandard(pair[1]);
    errors[1] = aboveStandard(errors[1]);
    for (const int fd : {pair[0], errors[0]}) {
        fcntl(fd, F_SETFD, FD_CLOEXEC);
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
    }
#ifdef SO_NOSIGPIPE
    const int one = 1;
    setsockopt(pair[0], SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pair[1], 0);
    posix_spawn_file_actions_adddup2(&actions, pair[1], 1);
    posix_spawn_file_actions_adddup2(&actions, errors[1], 2);
    posix_spawn_file_actions_addclose(&actions, pair[1]);
    posix_spawn_file_actions_addclose(&actions, errors[1]);
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
    sigset_t none, defaults;
    sigemptyset(&none);
    sigemptyset(&defaults);
    sigaddset(&defaults, SIGPIPE);
    posix_spawnattr_setsigmask(&attributes, &none);
    posix_spawnattr_setsigdefault(&attributes, &defaults);
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);
    const int rc = posix_spawnp(&impl_->pid, "ssh", &actions, &attributes, argv.data(), envp.data());
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attributes);
    ::close(pair[1]);
    ::close(errors[1]);
    if (rc != 0) {
        ::close(pair[0]); ::close(errors[0]);
        throw std::runtime_error(std::string("Cannot start OpenSSH ssh: ") + std::strerror(rc));
    }
    impl_->stream = pair[0];
    impl_->errors = errors[0];
}

SshProcess::~SshProcess()
{
    impl_->askpass.reset();
    ::shutdown(impl_->stream, SHUT_RDWR);
    ::close(impl_->stream);
    if (impl_->errors >= 0) ::close(impl_->errors);
    int status = 0;
    for (int attempt = 0; attempt < 100; ++attempt) {
        if (waitpid(impl_->pid, &status, WNOHANG) != 0) return;
        if (attempt == 0) kill(impl_->pid, SIGTERM);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    kill(impl_->pid, SIGKILL);
    waitpid(impl_->pid, &status, 0);
}

void SshProcess::write(std::span<const std::byte> data, const std::function<void()>& check)
{
#ifdef MSG_NOSIGNAL
    constexpr int flags = MSG_NOSIGNAL;
#else
    constexpr int flags = 0;
#endif
    for (size_t done = 0; done < data.size();) {
        const auto n = ::send(impl_->stream, data.data() + done, data.size() - done, flags);
        if (n > 0) { done += size_t(n); continue; }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
            check();
            impl_->drainErrors();
            pollfd ready{impl_->stream, POLLOUT, 0};
            poll(&ready, 1, 100);
            continue;
        }
        throw std::runtime_error("SSH write failed: " + diagnostics());
    }
}

size_t SshProcess::read(std::span<std::byte> out, std::chrono::milliseconds wait)
{
    pollfd ready[2]{{impl_->stream, POLLIN, 0}, {impl_->errors, POLLIN, 0}};
    poll(ready, impl_->errors >= 0 ? 2 : 1, int(wait.count()));
    impl_->drainErrors();
    const auto n = ::recv(impl_->stream, out.data(), out.size(), 0);
    if (n > 0) return size_t(n);
    if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR))
        throw std::runtime_error("SSH connection closed: " + diagnostics());
    return 0;
}

bool SshProcess::prompting() const { return impl_->askpass && impl_->askpass->active(); }
bool SshProcess::closed() const
{
    pollfd ready{impl_->stream, POLLIN, 0};
    if (poll(&ready, 1, 0) <= 0) return false;
    if (ready.revents & (POLLHUP | POLLERR | POLLNVAL)) return true;
    char byte;
    return (ready.revents & POLLIN) && ::recv(impl_->stream, &byte, 1, MSG_PEEK | MSG_DONTWAIT) == 0;
}
void SshProcess::authenticated() { impl_->askpass.reset(); }
std::string SshProcess::diagnostics()
{
    impl_->drainErrors();
    return impl_->tail;
}
#endif
}  // namespace detail

Authenticator terminalAuthenticator()
{
#ifdef _WIN32
    return {};
#else
    const int probe = ::open("/dev/tty", O_RDWR | O_CLOEXEC);
    if (probe < 0) return {};
    ::close(probe);
    return [](const Prompt& prompt) -> std::optional<std::string> {
        static std::mutex terminal;
        std::lock_guard lock(terminal);
        const int fd = ::open("/dev/tty", O_RDWR | O_CLOEXEC);
        if (fd < 0) return std::nullopt;
        const auto text = "[" + prompt.destination + "] " + prompt.text + (prompt.text.ends_with(' ') ? "" : " ");
        (void)!::write(fd, text.data(), text.size());
        termios saved{};
        const bool hide = !prompt.confirmation && tcgetattr(fd, &saved) == 0;
        if (hide) { auto quiet = saved; quiet.c_lflag &= ~tcflag_t(ECHO); tcsetattr(fd, TCSAFLUSH, &quiet); }
        std::string line;
        char c = 0;
        bool ended = false;
        while (::read(fd, &c, 1) == 1) {
            if (c == '\n') { ended = true; break; }
            line += c;
        }
        if (hide) { tcsetattr(fd, TCSAFLUSH, &saved); (void)!::write(fd, "\n", 1); }
        ::close(fd);
        if (!ended && line.empty()) return std::nullopt;
        return line;
    };
#endif
}

}  // namespace vc::sftp
