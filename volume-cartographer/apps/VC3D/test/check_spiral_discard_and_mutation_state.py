"""Validate production PCL mutations and fiber editor retirement without a viewer.

Run: python3 volume-cartographer/apps/VC3D/test/check_spiral_discard_and_mutation_state.py
Requires a C++17 compiler. Geometry is stubbed; this checks gesture and editor bookkeeping.
"""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

source = (Path(__file__).resolve().parents[1] / "SpiralBrushController.cpp").read_text()


def block(begin, end):
    start = source.index(begin)
    return source[start:source.index(end, start)]


mutations = [
    block("collection.pclEdit->appendPreviewPoint(volumePosition, surfacePosition);",
          "    invalidateEditablePclHitIndex();"),
    block("active.pclEdit->reverse();", "    clearEditablePclHover();"),
    block("active.pclEdit->setDeleted(true);", "    }"),
]
harness = r'''
#include <cassert>
enum class GestureState { Painted, Ready, Finalizing, Finalized };
struct Edit {
    void appendPreviewPoint(int, int) {}
    void reverse() {}
    void setDeleted(bool) {}
};
struct Gesture {
    Edit* pclEdit;
    GestureState state;
};
int main() {
    Edit edit;
    for (auto initial : {GestureState::Painted, GestureState::Ready,
                         GestureState::Finalizing, GestureState::Finalized}) {
        Gesture collection{&edit, initial};
        Gesture& active = collection;
        int volumePosition = 0, surfacePosition = 0;
        MUTATION
        assert(collection.state == GestureState::Painted);
    }
}
'''
with tempfile.TemporaryDirectory(prefix="spiral-pcl-state-") as temporary:
    cpp = Path(temporary) / "check.cpp"
    executable = Path(temporary) / "check"
    for mutation in mutations:
        cpp.write_text("#include <initializer_list>\n" + harness.replace("MUTATION", mutation))
        subprocess.run([*shlex.split(os.environ.get("CXX", "c++")), "-std=c++17",
                        str(cpp), "-o", str(executable)], check=True)
        subprocess.run([str(executable)], check=True)
print("PCL append, reverse and delete rearm all gesture states")

# Removing a discarded fiber working source must prevent its old editor from
# saving when the clean replacement opens. Other source sessions stay active.
controller = (Path(__file__).resolve().parents[1] / "LineAnnotationController.cpp").read_text()
start = controller.index("    std::vector<uint64_t> retiredIds;",
                         controller.index("void LineAnnotationController::unregisterExternalFiberSource("))
end = controller.index("    const auto oldSize", start)
retirement = r'''
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
struct Session {
    std::string fiberSourceRoot;
    bool suppressFiberSave = false;
    bool autoSaveScheduled = true;
    uint64_t fiberId;
};
struct Pane { Session* session; };
void closeDialogPanesForFibers(const std::vector<uint64_t>& ids) {
    assert(ids == std::vector<uint64_t>{1});
}
int main() {
    const std::string canonical = "discarded";
    Session discarded{"discarded", false, true, 1};
    Session unrelated{"other", false, true, 2};
    std::vector<Pane> _panes{{&discarded}, {&unrelated}, {nullptr}};
    RETIREMENT
    assert(discarded.suppressFiberSave && !discarded.autoSaveScheduled);
    assert(!unrelated.suppressFiberSave && unrelated.autoSaveScheduled);
}
'''.replace("RETIREMENT", controller[start:end])
with tempfile.TemporaryDirectory(prefix="spiral-fiber-retirement-") as temporary:
    cpp = Path(temporary) / "check.cpp"
    executable = Path(temporary) / "check"
    cpp.write_text(retirement)
    subprocess.run([*shlex.split(os.environ.get("CXX", "c++")), "-std=c++17",
                    str(cpp), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
print("Retired fiber editors cannot autosave discarded geometry")
