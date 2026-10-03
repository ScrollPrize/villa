"""Exercise production AFV source-selection methods with Qt (no live service)."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
def method(file, signature):
    text = (root / file).read_text()
    start = text.index(signature)
    end = text.index('\n}', start) + 2
    return text[start:end]

methods = '\n'.join(method('SpiralPanel.cpp', 'void SpiralPanel::' + name) for name in
                   ['keepCurrentFiberSourceForSingleFiber()', 'setOpenFiberVolume(const QString& path)'])
availability = method('FiberCollectionController.cpp', 'void FiberCollectionController::setSpiralFitAvailable(bool available)')
harness = r'''
#include <QApplication>
#include <QHash>
#include <QJsonObject>
#include <QLineEdit>
#include <QPushButton>
#include <QWidget>
#include <cassert>
struct Service {
 QJsonObject advertisedDataset() { return {{"resolved", QJsonObject{{"fibers", "/server/fibers"}}}}; }
};
struct SpiralPanel : QWidget {
 Service service; Service* _service=&service;
 QHash<QString,QLineEdit*> _paths;
 QString _openFiberVolume, _autoFilledAfvPath;
 bool _afvPathManual=false;
 QJsonObject _loadedSessionRequest;
 int refreshes=0;
 void refreshReloadRequired() { ++refreshes; }
 void setOpenFiberVolume(const QString& path);
 void keepCurrentFiberSourceForSingleFiber();
};
struct FiberCollectionController : QWidget {
 QPushButton add, open;
 QPushButton* addToSpiral_=&add; QPushButton* openAnnotation_=&open;
 bool spiralFitAvailable_=false, enabled_=false;
 long selected_=0;
 void setSpiralFitAvailable(bool available);
};
METHODS
AVAILABILITY
int main(int argc, char** argv) {
 QApplication app(argc,argv);
 SpiralPanel p; QLineEdit edit; p._paths["automated_fiber_volume"]=&edit;
 QLineEdit native; native.setText("/server/fibers"); p._paths["fibers"]=&native;
 p.setOpenFiberVolume("/data/scroll.afv"); assert(edit.text()=="/data/scroll.afv");
 p.setOpenFiberVolume({}); assert(edit.text().isEmpty()); assert(native.text()=="/server/fibers");
 p.setOpenFiberVolume("/data/scroll.afv");
 p._loadedSessionRequest={{"paths",QJsonObject{{"fibers","/server/fibers"},{"automated_fiber_volume",""}}}};
 p.keepCurrentFiberSourceForSingleFiber(); assert(edit.text().isEmpty()); assert(native.text()=="/server/fibers");
 p.setOpenFiberVolume("/data/other.afv"); assert(edit.text().isEmpty()); assert(native.text()=="/server/fibers");
 p._afvPathManual=true; edit.setText("/data/chosen.afv");
 p.keepCurrentFiberSourceForSingleFiber(); assert(edit.text()=="/data/chosen.afv");
 p.setOpenFiberVolume({}); assert(edit.text()=="/data/chosen.afv");
 FiberCollectionController c; c.setSpiralFitAvailable(true); assert(!c.add.isEnabled());
 c.enabled_=true; c.selected_=7; c.open.setEnabled(false);
 c.setSpiralFitAvailable(true); assert(!c.add.isEnabled());
 c.open.setEnabled(true); c.setSpiralFitAvailable(true); assert(c.add.isEnabled());
 c.setSpiralFitAvailable(false); assert(!c.add.isEnabled());
}
'''.replace('METHODS',methods).replace('AVAILABILITY',availability)
with tempfile.TemporaryDirectory(prefix='spiral-afv-selection-') as directory:
    cpp=Path(directory)/'check.cpp'; exe=Path(directory)/'check'
    cpp.write_text(harness)
    flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','Qt6Widgets'],text=True))
    subprocess.run([*shlex.split(os.environ.get('CXX','c++')),'-std=c++17','-fPIC',str(cpp),'-o',str(exe),*flags],check=True)
    subprocess.run([str(exe)],env={**os.environ,'QT_QPA_PLATFORM':'offscreen'},check=True)
print('AFV prefill, individual-fiber choice, and Spiral button availability passed')
