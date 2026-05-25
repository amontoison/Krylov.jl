# test_ckrylov.jl — validates the C interface by calling libkrylov from Julia
# via ccall and comparing results against Krylov.jl native Julia solvers.
#
# Usage (from the Krylov.jl root):
#   julia --startup-file=no --project=. interfaces/C/test/test_ckrylov.jl
#
# Set LIBKRYLOV_PATH to override the default library location.

using Test
using Libdl
using LinearAlgebra
using Random
using Krylov

const RNG = MersenneTwister(42)

# ============================================================================
# Load libkrylov
# ============================================================================

function default_libpath()
    dir = joinpath(@__DIR__, "..", "build")
    if Sys.iswindows()
        joinpath(dir, "bin", "libkrylov.dll")
    elseif Sys.isapple()
        joinpath(dir, "lib", "libkrylov.dylib")
    else
        joinpath(dir, "lib", "libkrylov.so")
    end
end

const LIBKRYLOV = Libdl.dlopen(get(ENV, "LIBKRYLOV_PATH", default_libpath()))

# ============================================================================
# Enums (must match krylov.h)
# ============================================================================

const KRYLOV_FLOAT32   = Cint(0)
const KRYLOV_FLOAT64   = Cint(1)
const KRYLOV_COMPLEX32 = Cint(2)
const KRYLOV_COMPLEX64 = Cint(3)
const KRYLOV_CPU       = Cint(0)

dtype_enum(::Type{Float32})    = KRYLOV_FLOAT32
dtype_enum(::Type{Float64})    = KRYLOV_FLOAT64
dtype_enum(::Type{ComplexF32}) = KRYLOV_COMPLEX32
dtype_enum(::Type{ComplexF64}) = KRYLOV_COMPLEX64

# ============================================================================
# C API wrappers
# ============================================================================

function c_workspace_create(solver::String, m::Int, n::Int, dtype::Cint)
    ws = Ref{Ptr{Cvoid}}(C_NULL)
    ret = ccall(Libdl.dlsym(LIBKRYLOV, :krylov_workspace_create), Cint,
        (Cstring, Cint, Cint, Cint, Cint, Ref{Ptr{Cvoid}}),
        solver, m, n, dtype, KRYLOV_CPU, ws)
    ret == 0 || error("workspace_create($solver, dtype=$dtype) returned $ret")
    ws[]
end

function c_workspace_free(ws::Ptr{Cvoid})
    ccall(Libdl.dlsym(LIBKRYLOV, :krylov_workspace_free), Cint, (Ptr{Cvoid},), ws)
end

function c_solve(ws::Ptr{Cvoid}, cb_A::Ptr{Cvoid}, cb_At::Ptr{Cvoid},
                 b::Vector; atol=1e-8, rtol=1e-8, itmax=0, verbose=0)
    GC.@preserve b begin
        ret = ccall(Libdl.dlsym(LIBKRYLOV, :krylov_solve), Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
             Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Cdouble, Cint, Cint),
            ws, cb_A, cb_At, C_NULL,
            pointer(b), C_NULL, Float64(atol), Float64(rtol), itmax, verbose)
    end
    ret == 0 || error("krylov_solve returned $ret")
end

function c_get_x(ws::Ptr{Cvoid}, ::Type{T}, n::Int) where T
    x = Vector{T}(undef, n)
    GC.@preserve x begin
        ret = ccall(Libdl.dlsym(LIBKRYLOV, :krylov_get_x), Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Cint), ws, pointer(x), n)
        ret == 0 || error("get_x returned $ret")
    end
    x
end

function c_get_y(ws::Ptr{Cvoid}, ::Type{T}, m::Int) where T
    y = Vector{T}(undef, m)
    GC.@preserve y begin
        ret = ccall(Libdl.dlsym(LIBKRYLOV, :krylov_get_y), Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Cint), ws, pointer(y), m)
        ret == 0 || error("get_y returned $ret")
    end
    y
end

function c_warm_start(ws::Ptr{Cvoid}, x0::Vector)
    GC.@preserve x0 begin
        ret = ccall(Libdl.dlsym(LIBKRYLOV, :krylov_warm_start), Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Cint), ws, pointer(x0), length(x0))
        ret == 0 || error("warm_start returned $ret")
    end
end

c_is_solved(ws::Ptr{Cvoid}) =
    ccall(Libdl.dlsym(LIBKRYLOV, :krylov_is_solved), Cint, (Ptr{Cvoid},), ws) == 1

c_niter(ws::Ptr{Cvoid}) =
    Int(ccall(Libdl.dlsym(LIBKRYLOV, :krylov_niter), Cint, (Ptr{Cvoid},), ws))

# ============================================================================
# Matvec callbacks — one @cfunction per precision, matrix set via global refs
# ============================================================================

# Global matrix refs (one pair per precision — not thread-safe, fine for tests)
const _A_f32  = Ref{Matrix{Float32}}()
const _At_f32 = Ref{Matrix{Float32}}()
const _A_f64  = Ref{Matrix{Float64}}()
const _At_f64 = Ref{Matrix{Float64}}()
const _A_c32  = Ref{Matrix{ComplexF32}}()
const _At_c32 = Ref{Matrix{ComplexF32}}()
const _A_c64  = Ref{Matrix{ComplexF64}}()
const _At_c64 = Ref{Matrix{ComplexF64}}()

function _cb!(xp::Ptr{Cvoid}, yp::Ptr{Cvoid}, A::Matrix{T}) where T
    m, n = size(A)
    x = unsafe_wrap(Vector{T}, Ptr{T}(xp), n)
    y = unsafe_wrap(Vector{T}, Ptr{T}(yp), m)
    mul!(y, A, x)
end

_mv_A_f32(xp::Ptr{Cvoid},  yp::Ptr{Cvoid}, ::Ptr{Cvoid}) = _cb!(xp, yp, _A_f32[])
_mv_At_f32(xp::Ptr{Cvoid}, yp::Ptr{Cvoid}, ::Ptr{Cvoid}) = _cb!(xp, yp, _At_f32[])
_mv_A_f64(xp::Ptr{Cvoid},  yp::Ptr{Cvoid}, ::Ptr{Cvoid}) = _cb!(xp, yp, _A_f64[])
_mv_At_f64(xp::Ptr{Cvoid}, yp::Ptr{Cvoid}, ::Ptr{Cvoid}) = _cb!(xp, yp, _At_f64[])
_mv_A_c32(xp::Ptr{Cvoid},  yp::Ptr{Cvoid}, ::Ptr{Cvoid}) = _cb!(xp, yp, _A_c32[])
_mv_At_c32(xp::Ptr{Cvoid}, yp::Ptr{Cvoid}, ::Ptr{Cvoid}) = _cb!(xp, yp, _At_c32[])
_mv_A_c64(xp::Ptr{Cvoid},  yp::Ptr{Cvoid}, ::Ptr{Cvoid}) = _cb!(xp, yp, _A_c64[])
_mv_At_c64(xp::Ptr{Cvoid}, yp::Ptr{Cvoid}, ::Ptr{Cvoid}) = _cb!(xp, yp, _At_c64[])

const CB_A_F32  = @cfunction(_mv_A_f32,  Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
const CB_At_F32 = @cfunction(_mv_At_f32, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
const CB_A_F64  = @cfunction(_mv_A_f64,  Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
const CB_At_F64 = @cfunction(_mv_At_f64, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
const CB_A_C32  = @cfunction(_mv_A_c32,  Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
const CB_At_C32 = @cfunction(_mv_At_c32, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
const CB_A_C64  = @cfunction(_mv_A_c64,  Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
const CB_At_C64 = @cfunction(_mv_At_c64, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))

function set_matrices!(::Type{Float32},    A, At) _A_f32[]  = A; _At_f32[]  = At end
function set_matrices!(::Type{Float64},    A, At) _A_f64[]  = A; _At_f64[]  = At end
function set_matrices!(::Type{ComplexF32}, A, At) _A_c32[]  = A; _At_c32[]  = At end
function set_matrices!(::Type{ComplexF64}, A, At) _A_c64[]  = A; _At_c64[]  = At end

function get_callbacks(::Type{Float32})    (CB_A_F32, CB_At_F32) end
function get_callbacks(::Type{Float64})    (CB_A_F64, CB_At_F64) end
function get_callbacks(::Type{ComplexF32}) (CB_A_C32, CB_At_C32) end
function get_callbacks(::Type{ComplexF64}) (CB_A_C64, CB_At_C64) end

# ============================================================================
# Test problems
# ============================================================================

# Symmetric positive definite: A = tridiag(-1, 2, -1), x_true = ones
function spd_problem(::Type{T}, n=20) where T
    A = Matrix{T}(SymTridiagonal(fill(T(2), n), fill(T(-1), n-1)))
    x_true = ones(T, n)
    b = A * x_true
    A, b, x_true
end

# Same but complex (Hermitian)
function hermitian_problem(::Type{T}, n=20) where T<:Complex
    R = real(T)
    A = Matrix{T}(SymTridiagonal(fill(R(2), n), fill(R(-1), n-1)))
    x_true = ones(T, n)
    b = A * x_true
    A, b, x_true
end

# Square non-symmetric, diagonally dominant
function nonsym_problem(::Type{T}, n=20) where T
    R = real(T)
    rng = RNG  # reproducible
    A = T.(randn(rng, R, n, n)) + T(n) * I
    x_true = ones(T, n)
    b = A * x_true
    A, b, x_true
end

# Rectangular (m > n), consistent least-squares
function ls_problem(::Type{T}, m=30, n=20) where T
    R = real(T)
    rng = RNG
    A = T.(randn(rng, R, m, n))
    x_true = ones(T, n)
    b = A * x_true
    A, b, x_true
end

# ============================================================================
# Solver metadata (names must match generate_stores.jl SOLVERS table)
# ============================================================================

# Solvers that require matvec_At
const NEED_AT = Set([
    "lslq", "lsqr", "lsmr",
    "cgls", "crls", "cgne", "crmr",
    "craig", "craigmr", "lnlq",
    "bilqr", "trilqr",
    "usymlq", "usymqr", "usymlqr",
])

# Solvers that return a dual solution via krylov_get_y
const HAS_Y = Set(["tricg", "trimr", "gpmr", "bilqr", "trilqr"])

# Non-symmetric solvers (need a non-symmetric problem)
const NONSYM = Set(["bilq", "qmr", "bicgstab", "cgs", "diom", "dqgmres",
                    "fom", "gmres", "fgmres",
                    "usymlq", "usymqr", "usymlqr",
                    "bilqr", "trilqr"])

# All 34 solver names
const ALL_SOLVERS = [
    "cg", "cr", "symmlq", "minres", "minres_qlp",
    "diom", "dqgmres", "fom", "gmres", "fgmres",
    "bicgstab", "cgs", "bilq", "qmr",
    "usymlq", "usymqr",
    "tricg", "trimr", "trilqr", "bilqr",
    "lslq", "lsqr", "lsmr", "usymlqr",
    "cgls", "crls", "cgne", "crmr",
    "craig", "craigmr", "lnlq",
    "gpmr", "car", "minares",
]

function make_problem(solver::String, ::Type{T}) where T
    if solver in NEED_AT
        ls_problem(T)
    elseif solver in NONSYM
        nonsym_problem(T)
    elseif T <: Complex
        hermitian_problem(T)
    else
        spd_problem(T)
    end
end

# ============================================================================
# Core test function
# ============================================================================

function test_solver(solver::String, ::Type{T}) where T
    tol = 100 * sqrt(eps(real(T)))

    A, b, x_true = make_problem(solver, T)
    m, n = size(A)
    At = Matrix(A')

    set_matrices!(T, A, At)
    cb_A, cb_At = get_callbacks(T)
    cb_At_arg = solver in NEED_AT ? cb_At : C_NULL

    ws = c_workspace_create(solver, m, n, dtype_enum(T))
    try
        c_solve(ws, cb_A, cb_At_arg, b; atol=1e-6, rtol=1e-6)

        @test c_is_solved(ws)
        @test c_niter(ws) > 0

        x = c_get_x(ws, T, n)
        @test norm(x - x_true) / norm(x_true) < tol

        if solver in HAS_Y
            y = c_get_y(ws, T, m)
            @test all(isfinite, y)
        end
    finally
        c_workspace_free(ws)
    end
end

# ============================================================================
# warm_start test (CG only, Float64)
# ============================================================================

function test_warm_start()
    n = 20
    A, b, x_true = spd_problem(Float64, n)
    set_matrices!(Float64, A, Matrix(A'))
    cb_A, _ = get_callbacks(Float64)

    ws = c_workspace_create("cg", n, n, KRYLOV_FLOAT64)
    try
        # First solve from zero
        c_solve(ws, cb_A, C_NULL, b; atol=1e-8, rtol=1e-8)
        niter_cold = c_niter(ws)

        # Warm start with exact solution — should converge in 1 iteration
        x0 = c_get_x(ws, Float64, n)
        c_warm_start(ws, x0)
        c_solve(ws, cb_A, C_NULL, b; atol=1e-8, rtol=1e-8)
        niter_warm = c_niter(ws)

        @test niter_warm < niter_cold
    finally
        c_workspace_free(ws)
    end
end

# ============================================================================
# Run all tests
# ============================================================================

@testset "CKrylov C interface" begin
    @testset "$T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "$solver" for solver in ALL_SOLVERS
            test_solver(solver, T)
        end
    end

    @testset "warm_start" begin
        test_warm_start()
    end
end

Libdl.dlclose(LIBKRYLOV)
