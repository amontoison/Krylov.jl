module CKrylov

using LinearAlgebra
using Krylov

include("c_enums.jl")
include("c_operator.jl")
include("c_stores.jl")   # 136 typed stores + dispatch helpers (generated)

# ---------------------------------------------------------------------------
# Function signatures exported to the generated C header.
# Each entry: (c_name, return_type, [(arg_name, c_type), ...])
# This vector is read by scripts/generate_header.jl.
# ---------------------------------------------------------------------------
const function_sigs = Tuple{String, String, Vector{Tuple{String,String}}}[]

macro export_sig(name, ret, args...)
  arg_pairs = [(string(a.args[1]), string(a.args[2])) for a in args]
  push!(function_sigs, (string(name), string(ret), arg_pairs))
  esc(:(nothing))
end

# ---------------------------------------------------------------------------
# krylov_workspace_create
#
# Creates a typed Krylov workspace and writes its opaque pointer into *ws_out.
#
#   solver  : KrylovSolverType enum value (e.g. KRYLOV_CG, KRYLOV_GMRES)
#   m, n    : operator dimensions (m rows, n columns)
#   dtype   : KrylovDataType enum value
#   device  : KrylovDeviceType enum value (currently only KRYLOV_CPU = 0)
#   ws_out  : address of a pointer that receives the workspace handle
#
# Returns 0 on success, -1 on error, -2 on unknown (solver, dtype) combination.
# ---------------------------------------------------------------------------
@export_sig krylov_workspace_create "int" (solver, "KrylovSolverType") (m, "int") (n, "int") (dtype, "KrylovDataType") (device, "KrylovDeviceType") (ws_out, "void**")

Base.@ccallable function krylov_workspace_create(
    solver :: Cint,
    m      :: Cint,
    n      :: Cint,
    dtype  :: Cint,
    device :: Cint,
    ws_out :: Ptr{Ptr{Cvoid}},
) :: Cint
  try
    _do_create!(solver, m, n, dtype, ws_out)
  catch e
    @error "krylov_workspace_create" exception=e
    Cint(-1)
  end
end

# ---------------------------------------------------------------------------
# krylov_solve
#
#   ws           : workspace handle (from krylov_workspace_create)
#   matvec_A     : C callback  void(*)(const void *x, void *y, void *ud)
#                  computes y = A*x
#   matvec_At    : C callback for y = A'*x, or NULL for CG/GMRES/MINRES/...
#                  (required for LSQR, LSMR, CGLS, CRAIG, ...)
#   matvec_M     : preconditioner callback (same signature), or NULL
#   b            : right-hand side array of length m
#   userdata     : opaque pointer forwarded to every callback
#   atol, rtol   : absolute and relative tolerances
#   itmax        : maximum iterations (0 → solver default)
#   verbose      : verbosity level (0 = silent)
#
# Returns 0 on success, nonzero on error.
# ---------------------------------------------------------------------------
@export_sig krylov_solve "int" (ws, "void*") (matvec_A, "KrylovMatvec") (matvec_At, "KrylovMatvec") (matvec_M, "KrylovMatvec") (b, "const void*") (c, "const void*") (userdata, "void*") (atol, "double") (rtol, "double") (itmax, "int") (verbose, "int")

Base.@ccallable function krylov_solve(
    ws_ptr   :: Ptr{Cvoid},
    fptr_A   :: Ptr{Cvoid},
    fptr_At  :: Ptr{Cvoid},
    fptr_M   :: Ptr{Cvoid},
    b_ptr    :: Ptr{Cvoid},
    c_ptr    :: Ptr{Cvoid},
    userdata :: Ptr{Cvoid},
    atol     :: Cdouble,
    rtol     :: Cdouble,
    itmax    :: Cint,
    verbose  :: Cint,
) :: Cint
  try
    _do_solve!(ws_ptr, fptr_A, fptr_At, fptr_M, b_ptr, c_ptr, userdata, atol, rtol, itmax, verbose)
  catch e
    @error "krylov_solve" exception=e
    Cint(-1)
  end
end

# ---------------------------------------------------------------------------
# krylov_get_x — copies the primal solution into a user-provided buffer
# ---------------------------------------------------------------------------
@export_sig krylov_get_x "int" (ws, "void*") (x, "void*") (n, "int")

Base.@ccallable function krylov_get_x(
    ws_ptr :: Ptr{Cvoid},
    x_ptr  :: Ptr{Cvoid},
    n      :: Cint,
) :: Cint
  try
    _do_get_x!(ws_ptr, x_ptr, n)
  catch e
    @error "krylov_get_x" exception=e
    Cint(-1)
  end
end

# ---------------------------------------------------------------------------
# krylov_get_y — dual solution (TriCG, TriMR, GPMR, BiLQR, TriLQR)
# Returns -2 if the solver has only one solution vector.
# ---------------------------------------------------------------------------
@export_sig krylov_get_y "int" (ws, "void*") (y, "void*") (m, "int")

Base.@ccallable function krylov_get_y(
    ws_ptr :: Ptr{Cvoid},
    y_ptr  :: Ptr{Cvoid},
    m      :: Cint,
) :: Cint
  try
    _do_get_y!(ws_ptr, y_ptr, m)
  catch e
    @error "krylov_get_y" exception=e
    Cint(-1)
  end
end

# ---------------------------------------------------------------------------
# Scalar statistics
# ---------------------------------------------------------------------------
@export_sig krylov_is_solved    "int"    (ws, "void*")
@export_sig krylov_niter        "int"    (ws, "void*")
@export_sig krylov_elapsed_time "double" (ws, "void*")

Base.@ccallable function krylov_is_solved(ws_ptr :: Ptr{Cvoid}) :: Cint
  _do_is_solved(ws_ptr)
end

Base.@ccallable function krylov_niter(ws_ptr :: Ptr{Cvoid}) :: Cint
  _do_niter(ws_ptr)
end

Base.@ccallable function krylov_elapsed_time(ws_ptr :: Ptr{Cvoid}) :: Cdouble
  _do_elapsed_time(ws_ptr)
end

# ---------------------------------------------------------------------------
# krylov_warm_start — sets the initial guess for the next krylov_solve call
# ---------------------------------------------------------------------------
@export_sig krylov_warm_start "int" (ws, "void*") (x0, "const void*") (n, "int")

Base.@ccallable function krylov_warm_start(
    ws_ptr :: Ptr{Cvoid},
    x0_ptr :: Ptr{Cvoid},
    n      :: Cint,
) :: Cint
  try
    _do_warm_start!(ws_ptr, x0_ptr, n)
  catch e
    @error "krylov_warm_start" exception=e
    Cint(-1)
  end
end

# ---------------------------------------------------------------------------
# krylov_workspace_free — releases the workspace (handle invalid after this)
# Returns 0 on success, 1 if handle was not found.
# ---------------------------------------------------------------------------
@export_sig krylov_workspace_free "int" (ws, "void*")

Base.@ccallable function krylov_workspace_free(ws_ptr :: Ptr{Cvoid}) :: Cint
  _do_free!(ws_ptr)
end

end  # module CKrylov
