cmake_minimum_required(VERSION 3.20)
project(CKrylov C)

# ---------------------------------------------------------------------------
# Find Julia and juliac
# ---------------------------------------------------------------------------
if(NOT DEFINED JULIA_EXECUTABLE)
  find_program(JULIA_EXECUTABLE julia REQUIRED)
endif()

# juliac is a Julia script located in <julia-prefix>/share/julia/juliac/juliac.jl
# We invoke it as:  julia <juliac.jl> [args...]
execute_process(
  COMMAND ${JULIA_EXECUTABLE} --startup-file=no
          -e "print(joinpath(Sys.BINDIR, \"..\", \"share\", \"julia\", \"juliac\", \"juliac.jl\"))"
  OUTPUT_VARIABLE JULIAC_SCRIPT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT EXISTS "${JULIAC_SCRIPT}")
  message(FATAL_ERROR "juliac.jl not found at ${JULIAC_SCRIPT}. Requires Julia >= 1.12.")
endif()

# Convenience wrapper: invoke juliac as  ${JULIAC}  [args...]
set(JULIAC ${JULIA_EXECUTABLE} ${JULIAC_SCRIPT})

message(STATUS "Julia:  ${JULIA_EXECUTABLE}")
message(STATUS "juliac: ${JULIAC_SCRIPT}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
set(KRYLOV_ROOT    ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(CKRYLOV_SRC  ${CMAKE_CURRENT_SOURCE_DIR}/src/CKrylov.jl)
set(HEADER_SCRIPT  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_header.jl)
set(INCLUDE_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(HEADER_OUT     ${INCLUDE_DIR}/krylov.h)

# ---------------------------------------------------------------------------
# Shared library:  CKrylov.so / CKrylov.dll
# ---------------------------------------------------------------------------
# juliac compiles every @ccallable function into a native shared library.
# --compile-ccallable  : export all Base.@ccallable functions
# --experimental       : required to enable --trim
# --trim=safe          : remove unreachable code (~19 MB vs ~269 MB without)
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/libkrylov${CMAKE_SHARED_LIBRARY_SUFFIX}
  COMMAND ${CMAKE_COMMAND} -E env JULIA_PROJECT=${KRYLOV_ROOT}
          ${JULIAC}
          --compile-ccallable
          --experimental
          --trim=safe
          --output-lib ${CMAKE_CURRENT_BINARY_DIR}/libkrylov${CMAKE_SHARED_LIBRARY_SUFFIX}
          ${CKRYLOV_SRC}
  DEPENDS ${CKRYLOV_SRC}
          ${CMAKE_CURRENT_SOURCE_DIR}/src/c_enums.jl
          ${CMAKE_CURRENT_SOURCE_DIR}/src/c_operator.jl
  COMMENT "Compiling libkrylov with juliac..."
  VERBATIM
)

add_custom_target(CKrylov_target ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libkrylov${CMAKE_SHARED_LIBRARY_SUFFIX}
)

# Expose as an imported library so example targets can link against it.
add_library(krylov SHARED IMPORTED GLOBAL)
set_target_properties(krylov PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libkrylov${CMAKE_SHARED_LIBRARY_SUFFIX}
  INTERFACE_INCLUDE_DIRECTORIES ${INCLUDE_DIR}
)
add_dependencies(krylov CKrylov_target)

# ---------------------------------------------------------------------------
# Auto-generate krylov.h after the library is built
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT  ${HEADER_OUT}
  COMMAND ${JULIA_EXECUTABLE} --startup-file=no --project=${KRYLOV_ROOT} ${HEADER_SCRIPT}
  DEPENDS ${CKRYLOV_SRC} ${HEADER_SCRIPT}
  COMMENT "Generating krylov.h..."
  VERBATIM
)

add_custom_target(krylov_header ALL DEPENDS ${HEADER_OUT})
add_dependencies(krylov krylov_header)

# ---------------------------------------------------------------------------
# Example executables
# ---------------------------------------------------------------------------
add_executable(basic_cg ${CMAKE_CURRENT_SOURCE_DIR}/examples/basic_cg.c)
target_link_libraries(basic_cg PRIVATE krylov)
target_include_directories(basic_cg PRIVATE ${INCLUDE_DIR})
add_dependencies(basic_cg krylov)
