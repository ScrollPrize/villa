#pragma once

// Marks an exception type that is thrown in one shared library and caught by type in
// another (the native reader, the Python modules, VC3D).
//
// The Python modules are built with -fvisibility=hidden, which also hides the types
// declared in the headers they include. On macOS a hidden type has its own RTTI in
// every library and libc++ compares RTTI by address, so a catch clause in the module
// does not match an exception thrown by the native reader: it arrives as its std::
// base class. With default visibility the copies compare equal (Clang on arm64 macOS
// then marks the RTTI for comparison by name). libstdc++ compares names for every
// type, so Linux behaves the same with or without this.
#if !defined(_WIN32) && (defined(__GNUC__) || defined(__clang__))
#define VCR_EXCEPTION_TYPE __attribute__((visibility("default")))
#else
#define VCR_EXCEPTION_TYPE
#endif
