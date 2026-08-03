# Task: Make Fiber 3D Test Hang Diagnostics Opt-In

Disable Fiber 3D training hang diagnostic files, watchdog timers, and manual
dump signal handlers by default. Enable them only through an explicit training
config setting or CLI flag, while preserving the existing timeout and detailed
diagnostics when requested.
