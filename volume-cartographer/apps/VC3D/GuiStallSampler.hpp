#pragma once

// Diagnostic-only GUI-stall stack sampler. A heartbeat timer on the GUI
// thread feeds a monitor thread; whenever the heartbeat goes silent for
// longer than the threshold, the monitor signals the GUI thread, whose
// async-signal handler writes its current backtrace to stderr (marker line
// "=== GUI STALL SAMPLE"). Repeated samples are taken while the stall lasts,
// so a long block shows whether it is one frame or a sequence.
//
// Linux-only (signals + execinfo); a no-op elsewhere. Install from the GUI
// thread once a QApplication exists.
namespace vc3d::diag {

void installGuiStallSampler();

}  // namespace vc3d::diag
