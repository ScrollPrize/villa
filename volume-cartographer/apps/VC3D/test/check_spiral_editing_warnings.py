"""Check persistent editing diagnostics using the production panel method.

Run: python3 volume-cartographer/apps/VC3D/test/check_spiral_editing_warnings.py
Requires a C++17 compiler and Qt6Core through pkg-config; no service or GUI needed.
"""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

source = (Path(__file__).resolve().parents[1] / "SpiralPanel.cpp").read_text()
start = source.index("void SpiralPanel::updateWarnings(")
end = source.index("void SpiralPanel::updateStatus(", start)
harness = r'''
#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QStringList>
#include <cassert>
struct Service {
    bool owner = false;
    bool ownsInputWorkspace() const { return owner; }
};
struct Label {
    QString text;
    void setText(const QString& value) { text = value; }
};
struct SpiralPanel : QObject {
    Service service;
    Service* _service = &service;
    Label label;
    Label* _warnings = &label;
    bool _connected = true;
    QString _editingAccessError;
    void updateWarnings(const QJsonObject&);
};
FUNCTION
int main() {
    SpiralPanel panel;
    const QJsonObject healthy{{"state", "Uninitialized"}, {"phase", "waiting"}};
    panel.updateWarnings(healthy);
    assert(panel.label.text.contains("Waiting to acquire editing access"));
    panel._editingAccessError = "Another service owns dataset editing";
    for (int poll = 0; poll < 5; ++poll) {
        panel.updateWarnings(healthy);
        assert(panel.label.text.contains(panel._editingAccessError));
        assert(panel.label.text.contains("disconnect and reconnect"));
    }
    const QJsonObject errors{{"error", "fit failed"},
        {"preview_publish_error", "preview failed"},
        {"warnings", QJsonArray{"dataset warning"}}};
    panel.updateWarnings(errors);
    assert(panel.label.text.contains("Another service owns dataset editing"));
    assert(panel.label.text.contains("fit failed"));
    assert(panel.label.text.contains("preview failed"));
    assert(panel.label.text.contains("dataset warning"));
    panel.service.owner = true;
    panel.updateWarnings(healthy);
    assert(panel._editingAccessError.isEmpty());
    assert(panel.label.text.isEmpty());
    panel.service.owner = false;
    panel._connected = false;
    panel.updateWarnings(healthy);
    assert(panel.label.text.isEmpty());
}
'''.replace("FUNCTION", source[start:end])

with tempfile.TemporaryDirectory(prefix="spiral-editing-warnings-") as temporary:
    cpp = Path(temporary) / "check.cpp"
    executable = Path(temporary) / "check"
    cpp.write_text(harness)
    flags = shlex.split(subprocess.check_output(
        ["pkg-config", "--cflags", "--libs", "Qt6Core"], text=True))
    subprocess.run([*shlex.split(os.environ.get("CXX", "c++")), "-std=c++17",
                    "-fPIC", str(cpp), "-o", str(executable), *flags], check=True)
    subprocess.run([str(executable)], check=True)
