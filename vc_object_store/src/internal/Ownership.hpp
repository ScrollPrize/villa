#pragma once
// Host-local writable-access guard for one store file.
//
// Every writable handle holds a lock on "<canonical store path>-owner" for its
// lifetime: standalone handles share it (SQLite serializes their writes), and
// a hosting service holds it exclusively. Canonical paths make symlinked or
// relative aliases of the same store use one guard; each handle opens its own
// descriptor, so several handles in one process are covered too. This is not
// a distributed lock: filesystem permissions keep clients from bypassing the
// library, and stores are never opened through network mounts.
#include <filesystem>
#include <memory>

namespace vc::objects::detail {
class OwnershipGuard {
public:
    enum class Mode { Standalone, Service };
    OwnershipGuard(const std::filesystem::path& store, Mode mode);
    ~OwnershipGuard();
    OwnershipGuard(const OwnershipGuard&) = delete;
    OwnershipGuard& operator=(const OwnershipGuard&) = delete;
private:
    struct State;
    std::unique_ptr<State> state_;
};
}
