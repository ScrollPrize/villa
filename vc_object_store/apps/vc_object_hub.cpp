#include "vc_object_store/objects/Hub.hpp"
#include "vc_object_store/TerminalProgress.hpp"
#include "vc_object_store/sftp/Sftp.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>

namespace {
using namespace vc::objects;

constexpr const char* kUsage =
    "Usage: vc_object_hub COMMAND [arguments] [--author NAME]\n"
    "  init STORE                      turn a local store into a hub (no other connections),\n"
    "                                  then copy it onto the network share\n"
    "  clone HUB REPLICA               copy a hub into a new local replica\n"
    "  sync REPLICA [--pull|--push]    pull hub changes, then push local ones\n"
    "  status REPLICA                  pending changes, conflicts and leases (no hub contact)\n"
    "  lease acquire|renew|release REPLICA UUID... [--seconds N]\n"
    "  resolve REPLICA UUID hub|local  settle a conflict by keeping one record\n"
    "  detach REPLICA                  make a replica an ordinary standalone store\n"
    "  unlock HUB [--force]            remove a stale hub lock left by a crashed client\n"
    "HUB is a path on a network filesystem or sftp://[user@]host[:port]/path (ssh:// works too);\n"
    "ssh asks for passwords, passphrases and new host keys on this terminal.\n"
    "--author defaults to $USER. Edits on a replica need a lease on objects that exist on the hub.\n";

void print(const HubSyncReport& report)
{
    std::cout << report.pulled << " pulled, " << report.pushed << " pushed, " << report.conflicts.size() << " conflicts\n";
    for (const auto& conflict : report.conflicts)
        std::cout << "  conflict " << conflict.uuid << " (hub version " << conflict.hubVersion << ")\n";
}

void print(const std::vector<HubLease>& leases)
{
    for (const auto& lease : leases)
    {
        const auto now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const auto left = lease.expiresUs / 1000000 - now;
        std::cout << lease.objectUuid << "  " << lease.author << "@" << lease.host << "  "
                  << (left > 0 ? "expires in " + std::to_string(left) + " s" : std::string("expired")) << '\n';
    }
}
}

int main(int argc, char** argv)
try {
    vc::sftp::setAuthenticator(vc::sftp::terminalAuthenticator());
    std::vector<std::string> positional;
    std::string author;
    int64_t seconds = kDefaultLeaseSeconds;
    bool pullOnly = false, pushOnly = false, force = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--author") author = next();
        else if (arg == "--seconds") seconds = std::stoll(next());
        else if (arg == "--pull") pullOnly = true;
        else if (arg == "--push") pushOnly = true;
        else if (arg == "--force") force = true;
        else if (arg == "--help" || arg == "-h") { std::cout << kUsage; return 0; }
        else if (arg.starts_with("--")) throw std::runtime_error("unknown argument: " + arg);
        else positional.push_back(arg);
    }
    if (author.empty())
        if (const char* user = std::getenv("USER")) author = user;
    if (positional.empty()) throw std::runtime_error("missing command; see --help");
    const auto command = positional[0];
    auto need = [&](size_t count) {
        if (positional.size() < count) throw std::runtime_error("missing arguments for " + command + "; see --help");
    };
    auto open = [&](const std::filesystem::path& path) {
        if (author.empty()) throw std::runtime_error("specify --author");
        OpenOptions options; options.author = author;
        return ObjectStore::open(path, options);
    };

    if (command == "init") {
        need(2);
        initHub(positional[1], author);
        std::cout << "initialized hub " << positional[1] << "; copy it onto the share before cloning\n";
    } else if (command == "clone") {
        need(3);
        if (author.empty()) throw std::runtime_error("specify --author");
        OpenOptions options; options.author = author;
        util::TerminalProgress progress("Clone hub");
        cloneHub(positional[1], positional[2], options, progress.callback());
        progress.finish();
    } else if (command == "sync") {
        need(2);
        auto replica = open(positional[1]);
        if (pullOnly && pushOnly) throw std::runtime_error("choose --pull or --push");
        const auto report = pullOnly ? hubPull(replica) : pushOnly ? hubPush(replica) : hubSync(replica);
        print(report);
        return report.conflicts.empty() ? 0 : 2;
    } else if (command == "status") {
        need(2);
        auto replica = open(positional[1]);
        const auto status = hubStatus(replica);
        if (!status) { std::cout << "not a hosted replica\n"; return 0; }
        std::cout << "hub " << status->hub.string() << "\nreplica " << status->replica << "\n"
                  << status->pending.size() << " pending, " << status->conflicts.size() << " conflicts\n";
        for (const auto& conflict : status->conflicts) std::cout << "  conflict " << conflict.uuid << '\n';
        print(status->leases);
    } else if (command == "lease") {
        need(4);
        auto replica = open(positional[2]);
        const std::vector<std::string> objects(positional.begin() + 3, positional.end());
        if (positional[1] == "acquire") print(acquireLeases(replica, objects, seconds));
        else if (positional[1] == "renew") print(renewLeases(replica, objects, seconds));
        else if (positional[1] == "release") releaseLeases(replica, objects);
        else throw std::runtime_error("lease takes acquire, renew or release");
    } else if (command == "resolve") {
        need(4);
        auto replica = open(positional[1]);
        print(resolveHubConflict(replica, positional[2], positional[3]));
    } else if (command == "detach") {
        need(2);
        detachHub(positional[1], author);
    } else if (command == "unlock") {
        need(2);
        const auto age = hubLockAge(positional[1]);
        if (!age) { std::cout << "hub is not locked\n"; return 0; }
        std::cout << "hub locked for " << *age << " s\n";
        if (!force) { std::cout << "rerun with --force once no client is syncing\n"; return 1; }
        removeHubLock(positional[1]);
        std::cout << "lock removed; the next open rolls back any interrupted write\n";
    } else throw std::runtime_error("unknown command: " + command);
    return 0;
} catch (const std::exception& error) {
    std::cerr << "vc_object_hub: " << error.what() << '\n';
    return 1;
}
