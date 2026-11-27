#pragma once
// POSIX setenv for the tests on Windows, whose C runtime has _putenv_s instead.
#ifdef _WIN32
#include <cstdlib>

inline int setenv(const char* name, const char* value, int overwrite)
{
    if (!overwrite && std::getenv(name))
        return 0;
    return _putenv_s(name, value);
}
#endif
