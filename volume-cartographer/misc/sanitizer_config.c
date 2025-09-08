#include <sanitizer/asan_interface.h>

const char *__asan_default_options() {
    return "symbolize=1:"                    // Symbolize stack traces
           "check_initialization_order=1:"   // Check static initialization order
           "strict_string_checks=1:"         // Strict checking for string functions
           "detect_stack_use_after_return=1:" // Detect stack use-after-return
           "detect_container_overflow=1:"    // Detect STL container overflows
           "detect_leaks=1:"                 // Enable LeakSanitizer
           "leak_check_at_exit=1:"           // Check for leaks on exit
           "halt_on_error=0:"                // Continue after first error
           "print_scariness=1:"              // Print scariness score of bugs
           "verbosity=0:"                    // Verbosity level (0=quiet, 3=verbose)
           "abort_on_error=0:"               // Don't abort, useful for testing
           "allocator_may_return_null=1:";    // Return NULL on OOM instead of crash
}

const char *__tsan_default_options() {
    return "symbolize=1:"                    // Symbolize stack traces
           "print_module_map=0:"             // Don't print module map
           "halt_on_error=0:"                // Continue after first race
           "atexit_sleep_ms=0:"              // Don't sleep at exit
           "verbosity=0:"                    // Quiet by default
           "flush_memory_ms=0:"              // Disable memory flush
           "flush_symbolizer_ms=5000:"       // Flush symbolizer cache
           "memory_limit_mb=0:"              // No memory limit
           "stop_on_start=0:"                // Don't stop on start
           "running_on_valgrind=0:"          // Not under valgrind
           "history_size=7:"                 // Size of per-thread history
           "io_sync=1:"                      // Synchronize IO
           "die_after_fork=0:"               // Continue after fork
           "report_signal_unsafe=1:"         // Report signal-unsafe calls
           "report_atomic_races=1:"          // Report races on atomics
           "force_seq_cst_atomics=0:"        // Don't force seq_cst
           "suppressions=tsan.supp:"         // Suppressions file
           "ignore_noninstrumented_modules=1:" // Ignore non-instrumented code
           "shared_ptr_interceptor=1:"       // Intercept shared_ptr operations
           "print_benign=0:"                 // Don't print benign races
           "exitcode=66:"                    // Exit code for races
           "log_path=tsan_log";              // Log file path
}

const char *__ubsan_default_options() {
    return "symbolize=1:"                    // Symbolize stack traces
           "print_stacktrace=1:"              // Print stack trace on error
           "halt_on_error=0:"                 // Continue after error
           "silence_unsigned_overflow=0:"     // Report unsigned overflow
           "print_summary=1:"                 // Print summary on exit
           "report_error_type=1:"             // Report specific error type
           "dedup_token_length=3:"           // Deduplication token length
           "log_path=ubsan_log:"             // Log file path
           "fast_unwind_on_check=0:"         // Full unwind for better traces
           "abort_on_error=0";                // Don't abort
}

const char *__lsan_default_options() {
    return "symbolize=1:"                    // Symbolize stack traces
           "print_module_map=0:"             // Don't print module map
           "print_stats=0:"                  // Don't print statistics
           "print_suppressions=0:"            // Don't print suppressions
           "report_objects=0:"                // Don't report individual objects
           "use_unaligned=0:"                 // Don't use unaligned loads
           "verbosity=0:"                    // Quiet by default
           "log_threads=0:"                   // Don't log thread creation
           "log_pointers=0:"                 // Don't log pointer actions
           "exitcode=23:"                    // Exit code for leaks
           "print_summary=1:"                // Print leak summary
           "check_printf=1:"                 // Check printf functions
           "leak_check_at_exit=1:"           // Check at exit
           "allocator_may_return_null=0:"    // Crash on OOM
           "print_module_map=0:"             // Module map is noisy
           "max_leaks=0:"                    // No limit on reported leaks
           "fast_unwind_on_malloc=1";        // Fast unwind for performance
}

const char* __lsan_default_suppressions() {
    return
        // Font/GTK library leaks
        "leak:libfontconfig\n"
        "leak:libpango\n"
        "leak:libgtk-3\n"
        "leak:libglib-2.0\n"
        "leak:libgobject-2.0\n"
        "leak:libharfbuzz\n"
        "leak:libqgtk3\n"

        // Font/GTK function leaks
       "leak:FcFont*\n"
        "leak:pango_*\n"
        "leak:g_type_*\n";
}


