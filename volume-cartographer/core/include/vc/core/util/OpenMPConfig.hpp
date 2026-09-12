#pragma once

namespace vc::core::util
{

/**
 * Take OpenMP's dynamic-team sizing out of the picture for a batch tool.
 *
 * OpenCV's libopencv_core carries a static initializer (parallel.cpp) that
 * calls omp_set_dynamic(1) unconditionally. With dyn-var enabled, libgomp
 * chooses each team's size from gomp_dynamic_max_threads(), which is derived
 * from getloadavg(): it caps the team at the number of online CPUs, subtracts
 * the one-minute load average, and -- crucially -- returns exactly one thread
 * when that load average is greater than or equal to the cap. Because the cap
 * is itself min(online CPUs, nthreads-var), asking for FEWER threads makes the
 * collapse easier to trigger, not harder.
 *
 * The practical effect is that on a machine that is already busy -- the normal
 * state of a batch host running several of these tools at once -- every
 * parallel region silently runs on a single thread, and no documented control
 * reveals it: OMP_NUM_THREADS, omp_set_num_threads() and any thread-count
 * option all set nthreads-var, which dyn-var then overrides. Even
 * OMP_DYNAMIC=FALSE does not help, because libgomp reads that environment
 * variable during its own initialization and OpenCV's constructor runs
 * afterwards.
 *
 * Calling this from main() is therefore the only fix that works: main() runs
 * after every static initializer, so it has the last word. Afterwards a team is
 * exactly nthreads-var and the documented controls behave as documented.
 *
 * Set VC_OMP_DYNAMIC=1 to skip this and restore the previous behaviour.
 *
 * This changes only how many threads execute a parallel region, never what is
 * computed in it.
 */
void disableOpenMPDynamicTeams();

}  // namespace vc::core::util
