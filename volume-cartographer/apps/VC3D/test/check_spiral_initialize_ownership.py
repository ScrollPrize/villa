"""Run the production Initialize/Rebuild button handler against an editing lease.

Run: python3 volume-cartographer/apps/VC3D/test/check_spiral_initialize_ownership.py
Requires an existing C++ compiler and Qt6Widgets development files.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

source = (Path(__file__).resolve().parents[1] / "SpiralPanel.cpp").read_text()
start = source.index("    connect(_load, &QPushButton::clicked, this, [this]() {")
start = source.index("\n", start) + 1
end = source.index("    connect(_run,", start)
body = source[start:end].rsplit("    });", 1)[0]
harness = r'''
#include <QApplication>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QTextEdit>
#include <QWidget>
#include <cassert>
#include <functional>
struct Service {
    bool owns = true;
    int initialized = 0, rebuilt = 0;
    void initializeSession(const QJsonObject& request) {
        assert(owns);
        assert(request["paths"].toObject()["checkpoint"].toString().isEmpty());
        ++initialized;
    }
    void rebuildSession(const QJsonObject&) { assert(owns); ++rebuilt; }
    void rebuildWithDefaults() { assert(owns); ++rebuilt; }
};
struct Panel : QWidget {
    Service service;
    Service* _service = &service;
    QTextEdit advanced;
    QTextEdit* _advanced = &advanced;
    QLabel warnings;
    QLabel* _warnings = &warnings;
    QString _sessionState = "Uninitialized";
    bool _reloadRequired = true;
    int _uncommittedCount = 0;
    QString pendingRebuildStage() { return "all"; }
    void persist() {}
    QJsonObject sessionRequest() { return {{"paths", QJsonObject{{"checkpoint", "old.ckpt"}}}}; }
    void guardSessionExit(std::function<void()> action) { service.owns = false; action(); }
    void click() {
BODY
    }
};
int main(int argc, char** argv) {
    QApplication app(argc, argv);
    Panel panel;
    panel.advanced.setPlainText("{}");
    panel.click();
    assert(panel.service.initialized == 1 && panel.service.owns);
    panel._sessionState = "Idle";
    panel.click();
    assert(panel.service.rebuilt == 1 && panel.service.owns);
}
'''.replace("BODY", body)
with tempfile.TemporaryDirectory(prefix="spiral-initialize-") as directory:
    cpp = Path(directory) / "check.cpp"
    exe = Path(directory) / "check"
    cpp.write_text(harness)
    flags = shlex.split(subprocess.check_output(
        ["pkg-config", "--cflags", "--libs", "Qt6Widgets"], text=True))
    subprocess.run([*shlex.split(os.environ.get("CXX", "c++")), "-std=c++17",
                    "-fPIC", str(cpp), "-o", str(exe), *flags], check=True)
    subprocess.run([str(exe)], env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}, check=True)
print("Initialize and rebuild retain editing ownership")
