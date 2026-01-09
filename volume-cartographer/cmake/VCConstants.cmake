# The prefix under which includes will be installed
set(vc_prefix "vc")

# Header install directory. Lowercase so includes look like:
#
#     #include <vc/core/types/PerPixelMap.h>
#
set(include_install_dir "include/${vc_prefix}")

# CMake config files
set(config_install_dir "lib/cmake/${PROJECT_NAME}")

# Extra resources
set(share_install_dir "share/${PROJECT_NAME}")

# Targets export name (VCConfig)
set(targets_export_name "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

# Get Git hash
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
if(NOT GIT_SHA1)
    execute_process(
        COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse HEAD
        OUTPUT_VARIABLE GIT_SHA1
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
if(GIT_SHA1)
    string(SUBSTRING ${GIT_SHA1} 0 7 GIT_SHA1_SHORT)
else()
    set(GIT_SHA1 Untracked)
    set(GIT_SHA1_SHORT Untracked)
endif()
