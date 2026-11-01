include_guard(GLOBAL)

# Codec targets must not inherit Qt autogen from an embedding application.
# In particular, Qt otherwise injects C++ moc sources into the C-only kernels.
# A function scope preserves the application's settings for subsequent targets.
function(vc_fetch_delta3d)
    set(CMAKE_AUTOMOC OFF)
    set(CMAKE_AUTOUIC OFF)
    set(CMAKE_AUTORCC OFF)
    FetchContent_MakeAvailable(vc_delta3d)
endfunction()
