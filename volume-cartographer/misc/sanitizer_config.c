// Default runtime options for the *San family. Linked into any target built
// with a sanitizer enabled; per-process overrides still flow through the
// matching {ASAN,UBSAN,TSAN,LSAN,TYSAN,NSAN}_OPTIONS env vars.

const char* __asan_default_options(void)
{
    return "symbolize=1:"
           "halt_on_error=1:"
           "abort_on_error=0:"
           "check_initialization_order=1:"
           "strict_string_checks=1:"
           "detect_stack_use_after_return=1:"
           "detect_container_overflow=1:"
           "detect_leaks=1:"
           "allocator_may_return_null=1";
}

const char* __ubsan_default_options(void)
{
    return "symbolize=1:"
           "halt_on_error=1:"
           "print_stacktrace=1:"
           "print_summary=1:"
           "report_error_type=1";
}

const char* __tsan_default_options(void)
{
    return "symbolize=1:"
           "halt_on_error=1:"
           "history_size=7:"
           "report_atomic_races=1:"
           "report_signal_unsafe=1";
}

const char* __lsan_default_options(void)
{
    return "symbolize=1:"
           "print_suppressions=0:"
           "leak_check_at_exit=1:"
           "max_leaks=0";
}

const char* __tysan_default_options(void)
{
    return "symbolize=1:"
           "halt_on_error=1";
}

const char* __nsan_default_options(void)
{
    return "symbolize=1:"
           "halt_on_error=0:"
           "print_stacktrace=1";
}

const char* __lsan_default_suppressions(void)
{
    return
        "leak:libfontconfig\n"
        "leak:libpango\n"
        "leak:libgtk-3\n"
        "leak:libglib-2.0\n"
        "leak:libgobject-2.0\n"
        "leak:libharfbuzz\n"
        "leak:libqgtk3\n"
        "leak:FcFont*\n"
        "leak:pango_*\n"
        "leak:g_type_*\n";
}
