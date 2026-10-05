"""Exercise the production syncArtifacts function with deferred fake downloads.

Run: python3 volume-cartographer/apps/VC3D/test/check_spiral_download_order.py
Requires a C++17 compiler and Qt6Core/Qt6Gui available through pkg-config. This small
harness compiles the actual function from SpiralServiceManager.cpp so it tests
request scheduling and callbacks without starting the GUI or an HTTP server.
"""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

source_dir = Path(__file__).resolve().parents[1]
source = (source_dir / "SpiralServiceManager.cpp").read_text()
start = source.index("void SpiralServiceManager::syncArtifacts(")
end = source.index("\nQStringList SpiralServiceManager::pclArtifactCachePins()", start)
function = source[start:end]
harness = r'''
#include "SpiralPclRole.hpp"
#include <QJsonObject>
#include <QStringList>
#include <array>
#include <cassert>
#include <functional>
#include <map>

constexpr int kPreviewCacheKept = 3;
struct Cache {
    using Callback = std::function<void(const QString&, const QString&, bool)>;
    std::map<QString, Callback> pending;
    int prunes = 0;
    void fetchArtifact(const QString&, const QString& id, Callback callback) {
        pending[id] = callback;
    }
    void pruneSession(const QString&, int, const QStringList&) { ++prunes; }
};
struct SpiralServiceManager {
    Cache cache;
    Cache* _artifactCache = &cache;
    quint64 _connectionGeneration = 1;
    qint64 _previewSequence = 0;
    std::array<quint64, 2> _pclSequence{};
    std::array<QString, 2> _installedPclArtifact, _fetchingPclArtifact, _lastPclLocalPath;
    QString _installedPreviewArtifact, _fetchingPreviewArtifact, _installedPreviewSession;
    QString _lastPreviewLocalPath, _installedDiagnosticsArtifact;
    QString _fetchingDiagnosticsArtifact, _lastDiagnosticsLocalPath;
    int installed = 0, errors = 0;
    QJsonObject lastRef;
    void errorOccurred(const QString&) { ++errors; }
    void previewAvailable(const QString&, qint64) {}
    void previewDiagnosticsAvailable(const QString&, qint64) {}
    void pclArtifactAvailable(vc3d::spiral::PclRole, const QString&, const QJsonObject& ref) {
        ++installed; lastRef = ref;
    }
    QStringList pclArtifactCachePins() { return {}; }
    void syncArtifacts(const QJsonObject&);
};
FUNCTION
int main() {
    using namespace vc3d::spiral;
    const auto role = kEditablePclRoles[0];
    const auto otherRole = kEditablePclRoles[1];
    const auto slot = pclRoleIndex(role);
    auto status = [](auto role, const QString& id) {
        return QJsonObject{{"session_id", "session"},
            {pclRoleStatusKey(role), QJsonObject{{"id", id}, {"source_revision", id}}}};
    };
    SpiralServiceManager manager;
    manager.syncArtifacts(status(role, "old"));
    manager.syncArtifacts(status(role, "new"));
    manager.syncArtifacts(status(otherRole, "other"));
    manager.cache.pending.at("new")("/new", {}, false);
    manager.cache.pending.at("old")("/old", {}, false);
    assert(manager._installedPclArtifact[slot] == "new");
    assert(manager._lastPclLocalPath[slot] == "/new");
    assert(manager.lastRef.value("source_revision") == "new");
    assert(manager.installed == 1 && manager.cache.prunes == 1);
    manager.cache.pending.at("other")("/other", {}, false);
    assert(manager.installed == 2); // Different roles do not supersede each other.
    manager.syncArtifacts(status(role, "third"));
    manager.syncArtifacts(status(role, "fourth"));
    manager.cache.pending.at("third")({}, "stale failure", false);
    assert(manager.errors == 0);
    assert(manager._fetchingPclArtifact[slot] == "fourth");
    ++manager._connectionGeneration;
    manager.cache.pending.at("fourth")("/fourth", {}, false);
    assert(manager.installed == 2); // Reconnection still invalidates downloads.
}
'''.replace('FUNCTION', function)
with tempfile.TemporaryDirectory(prefix="spiral-download-order-") as temporary:
    cpp = Path(temporary) / "check.cpp"
    executable = Path(temporary) / "check"
    cpp.write_text(harness)
    flags = shlex.split(subprocess.check_output(
        ["pkg-config", "--cflags", "--libs", "Qt6Core", "Qt6Gui"], text=True))
    subprocess.run([*shlex.split(os.environ.get("CXX", "c++")), "-std=c++17",
                    "-fPIC", "-I", str(source_dir), str(cpp), "-o", str(executable),
                    *flags], check=True)
    subprocess.run([str(executable)], check=True)
print("PCL download ordering checks passed")
