include(FindPackageHandleStandardArgs)

find_program(NASM_EXECUTABLE
  NAMES nasm nasmw
)

# macOS has an unusable nasm in /usr/bin
# Check to make sure it actually runs
if(NASM_EXECUTABLE)
  execute_process(
    COMMAND ${NASM_EXECUTABLE}
    RESULT_VARIABLE EXIT_CODE
    OUTPUT_QUIET
    ERROR_QUIET
  )

  if(EXIT_CODE EQUAL 72)
    set(NASM_EXECUTABLE NOTFOUND)
  endif()
endif()

find_package_handle_standard_args(NASM "DEFAULT_MSG" NASM_EXECUTABLE)

mark_as_advanced(NASM_EXECUTABLE)
