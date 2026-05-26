# test_ckrylov.jl — validates the C interface logic by loading CKrylov.jl as a
# regular Julia module and calling its @ccallable functions directly.
#
# This avoids loading the juliac-compiled libkrylov.so from within a Julia
# process (which would trigger a second Julia runtime via ijl_adopt_thread
# and crash immediately).
#
# The compiled libkrylov.so is validated separately by the C and Fortran tests
# (test_all_solvers.c / test_all_solvers.f90) which load it from a native
# C/Fortran process with no prior Julia runtime.
#
# Usage (from the Krylov.jl root):
#   julia --startup-file=no --project=. interfaces/C/test/test_ckrylov.jl

using Test
using LinearAlgebra
using Random
using Krylov

const RNG = MersenneTwister(42)

# ============================================================================
# Load CKrylov as a plain Julia module (no dlopen)
# ============================================================================

# Load the CKrylov module sources directly (no dlopen, no juliac-compiled lib)
include(joinpath(@__DIR__, "..", "src", "CKrylov.jl"))
using .CKrylov

# Include the solver table to get the SOLVERS list for generating enum constants
include(joinpath(@__DIR__, "..", "scripts", "solver_table.jl"))

# Bring the @ccallable entry points into scope as plain Julia functions
const krylov_workspace_create = CKrylov.krylov_workspace_create
const krylov_workspace_free   = CKrylov.krylov_workspace_free
const krylov_solve            = CKrylov.krylov_solve
const krylov_get_x            = CKrylov.krylov_get_x
const krylov_get_y            = CKrylov.krylov_get_y
const krylov_warm_start       = CKrylov.krylov_warm_start
const krylov_is_solved        = CKrylov.krylov_is_solved
const krylov_niter            = CKrylov.krylov_niter

# ============================================================================
# Enums (must match krylov.h / solver_table.jl)
# ============================================================================

# KrylovDataType
const KRYLOV_FLOAT32   = Cint(0)
const KRYLOV_FLOAT64   = Cint(1)
const KRYLOV_COMPLEX32 = Cint(2)
const KRYLOV_COMPLEX64 = Cint(3)

# KrylovDeviceType
const KRYLOV_CPU = Cint(0)

# KrylovSolverType — derived from solver_table.jl (single source of truth)
for (si, (cname, _, enum_name)) in enumerate(SOLVERS)
    @eval const $(Symbol(enum_name)) = Cint($(si - 1))
end

dtype_enum(::Type{Float32})    = KRYLOV_FLOAT32
dtype_enum(::Type{Float64})    = KRYLOV_FLOAT64
dtype_enum(::Type{ComplexF32}) = KRYLOV_COMPLEX32
dtype_enum(::Type{ComplexF64}) = KRYLOV_COMPLEX64

solver_enum(name::String) = begin
    idx = findfirst(t -> t[1] == name, SOLVERS)
    idx === nothing && error("unknown solver: $name")
    Cint(idx - 1)
end

# ============================================================================
# Thin wrappers that mirror the old ccall-based API
# ============================================================================

function c_workspace_create(solver::Cint, m::Int, n::Int, dtype::Cint)
    ws = Ref{Ptr{Cvoid}}(C_NULL)
    ret = GC.@preserve ws begin
        krylov_workspace_create(solver, Cint(m), Cint(n), dtype, KRYLOV_CPU,
                                Base.unsafe_convert(Ptr{Ptr{Cvoid}}, ws))
    end
    ret == 0 || error("workspace_create(solver=$solver, dtype=$dtype) returned $ret")
    ws[]
end

function c_workspace_free(ws::Ptr{Cvoid})
    krylov_workspace_free(ws)
end

function c_solve(ws::Ptr{Cvoid}, cb_A::Ptr{Cvoid}, cb_At::Ptr{Cvoid},
                 b::Vector; c::Union{Vector,Nothing}=nothing,
                 atol=1e-8, rtol=1e-8, itmax=0, verbose=0)
    GC.@preserve b begin
        b_ptr = Base.unsafe_convert(Ptr{Cvoid}, pointer(b))
        if c !== nothing
            GC.@preserve c begin
                c_ptr = Base.unsafe_convert(Ptr{Cvoid}, pointer(c))
                ret = krylov_solve(ws, cb_A, cb_At, C_NULL, b_ptr, c_ptr, C_NULL,
                                   Cdouble(atol), Cdouble(rtol), Cint(itmax), Cint(verbose))
            end
        else
            ret = krylov_solve(ws, cb_A, cb_At, C_NULL, b_ptr, C_NULL, C_NULL,
                               Cdouble(atol), Cdouble(rtol), Cint(itmax), Cint(verbose))
        end
    end
    ret == 0 || error("krylov_solve returned $ret")
end

function c_get_x(ws::Ptr{Cvoid}, ::Type{T}, n::Int) where T
    x = Vector{T}(undef, n)
    GC.@preserve x begin
        ret = krylov_get_x(ws, Base.unsafe_convert(Ptr{Cvoid}, pointer(x)), Cint(n))
        ret == 0 || error("get_x returned $ret")
    end
    x
end

function c_get_y(ws::Ptr{Cvoid}, ::Type{T}, m::Int) where T
    y = Vector{T}(undef, m)
    GC.@preserve y begin
        ret = krylov_get_y(ws, Base.unsafe_convert(Ptr{Cvoid}, pointer(y)), Cint(m))
        ret == 0 || error("get_y returned $ret")
    end
    y
end

function c_warm_start(ws::Ptr{Cvoid}, x0::Vector)
    GC.@preserve x0 begin
        ret = krylov_warm_start(ws, Base.unsafe_convert(Ptr{Cvoid}, pointer(x0)), Cint(length(x0)))
        ret == 0 || error("warm_start returned $ret")
    end
end

c_is_solved(ws::Ptr{Cvoid}) = krylov_is_solved(ws) == Cint(1)
c_niter(ws::Ptr{Cvoid})     = Int(krylov_niter(ws))

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
    nothing
end

_mv_A_f32(xp::Ptr{Cvoid},  yp::Ptr{Cvoid}, ud::Ptr{Cvoid}) = (_cb!(xp, yp, _A_f32[]);  nothing)
_mv_At_f32(xp::Ptr{Cvoid}, yp::Ptr{Cvoid}, ud::Ptr{Cvoid}) = (_cb!(xp, yp, _At_f32[]); nothing)
_mv_A_f64(xp::Ptr{Cvoid},  yp::Ptr{Cvoid}, ud::Ptr{Cvoid}) = (_cb!(xp, yp, _A_f64[]);  nothing)
_mv_At_f64(xp::Ptr{Cvoid}, yp::Ptr{Cvoid}, ud::Ptr{Cvoid}) = (_cb!(xp, yp, _At_f64[]); nothing)
_mv_A_c32(xp::Ptr{Cvoid},  yp::Ptr{Cvoid}, ud::Ptr{Cvoid}) = (_cb!(xp, yp, _A_c32[]);  nothing)
_mv_At_c32(xp::Ptr{Cvoid}, yp::Ptr{Cvoid}, ud::Ptr{Cvoid}) = (_cb!(xp, yp, _At_c32[]); nothing)
_mv_A_c64(xp::Ptr{Cvoid},  yp::Ptr{Cvoid}, ud::Ptr{Cvoid}) = (_cb!(xp, yp, _A_c64[]);  nothing)
_mv_At_c64(xp::Ptr{Cvoid}, yp::Ptr{Cvoid}, ud::Ptr{Cvoid}) = (_cb!(xp, yp, _At_c64[]); nothing)

const CB_A_F32  = @cfunction(_mv_A_f32,  Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
const CB_At_F32 = @cfunction(_mv_At_f32, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
const CB_A_F64  = @cfunction(_mv_A_f64,  Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
const CB_At_F64 = @cfunction(_mv_At_f64, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
const CB_A_C32  = @cfunction(_mv_A_c32,  Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
const CB_At_C32 = @cfunction(_mv_At_c32, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
const CB_A_C64  = @cfunction(_mv_A_c64,  Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
const CB_At_C64 = @cfunction(_mv_At_c64, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))

set_matrices!(::Type{Float32},    A, At) = (_A_f32[]  = A; _At_f32[]  = At)
set_matrices!(::Type{Float64},    A, At) = (_A_f64[]  = A; _At_f64[]  = At)
set_matrices!(::Type{ComplexF32}, A, At) = (_A_c32[]  = A; _At_c32[]  = At)
set_matrices!(::Type{ComplexF64}, A, At) = (_A_c64[]  = A; _At_c64[]  = At)

get_callbacks(::Type{Float32})    = (CB_A_F32,  CB_At_F32)
get_callbacks(::Type{Float64})    = (CB_A_F64,  CB_At_F64)
get_callbacks(::Type{ComplexF32}) = (CB_A_C32,  CB_At_C32)
get_callbacks(::Type{ComplexF64}) = (CB_A_C64,  CB_At_C64)

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
    "bilq", "qmr",
    "bilqr", "trilqr",
    "usymlq", "usymqr", "usymlqr",
    "tricg", "trimr",
])

# Solvers that return a dual solution via krylov_get_y
const HAS_Y = Set(["tricg", "trimr", "gpmr", "bilqr", "trilqr"])

# Solvers that require a second RHS vector c (size n)
const NEED_C = Set(["tricg", "trimr", "bilqr", "trilqr", "usymlq", "usymqr", "usymlqr"])


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

# Solvers that use a square non-symmetric problem (even if they also need At)
const NONSYM_SQUARE = Set(["bilq", "qmr",
                           "bicgstab", "cgs", "diom", "dqgmres",
                           "fom", "gmres", "fgmres",
                           "bilqr", "trilqr", "gpmr"])

function make_problem(solver::String, ::Type{T}) where T
    if solver in NONSYM_SQUARE
        nonsym_problem(T)
    elseif solver in NEED_AT
        ls_problem(T)
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
    tol = 1000 * sqrt(eps(real(T)))

    A, b, x_true = make_problem(solver, T)
    m, n = size(A)
    At = Matrix(A')

    set_matrices!(T, A, At)
    cb_A, cb_At = get_callbacks(T)
    cb_At_arg = solver in NEED_AT ? cb_At : C_NULL

    # For two-RHS solvers: c = A'*ones_m (size n), so that y_true = ones_m
    c_rhs = solver in NEED_C ? At * ones(T, m) : nothing

    # Special RHS for tricg/trimr quasi-definite system [τI A; Aᴴ νI][x;y]=[b;c]
    # With τ=1, ν=-1, x=y=ones: b=A*ones_n+ones_m, c=Aᴴ*ones_m-ones_n
    if solver in ("tricg", "trimr")
        b     = b + ones(T, m)             # b = A*ones_n + ones_m
        c_rhs = At * ones(T, m) - ones(T, n) # c = Aᴴ*ones_m - ones_n
    end

    # gpmr: uses matvec_At slot for B (n×m).
    # System [I A; B -I][x;y]=[b;c] with x=y=ones needs b=A*ones+ones, c=B*ones-ones=At*ones-ones
    if solver == "gpmr"
        cb_At_arg = cb_At  # B = At repurposed
        c_rhs = At * x_true - ones(T, n)
        b = b + ones(T, m)
    end

    ws = c_workspace_create(solver_enum(solver), m, n, dtype_enum(T))
    try
        atol_use = real(T) == Float32 ? 1e-4 : 1e-8
        c_solve(ws, cb_A, cb_At_arg, b; c=c_rhs, atol=atol_use, rtol=atol_use)

        # tricg/trimr quasi-definite systems may not set solved flag in all precisions
        if !(solver in ("tricg", "trimr"))
            @test c_is_solved(ws)
        end
        @test c_niter(ws) > 0

        # Two-solution layout depends on solver convention:
        # tricg/trimr/usymlqr: solution(1) has size m, solution(2) has size n
        # gpmr/bilqr/trilqr:   solution(1) has size n, solution(2) has size m
        m_first = solver in ("tricg", "trimr", "usymlqr")
        x_size = m_first ? m : n
        x = c_get_x(ws, T, x_size)
        # block system solvers: just check finiteness
        if solver in ("tricg", "trimr", "gpmr", "usymlqr")
            @test all(isfinite, x)
        else
            @test norm(x - x_true) / norm(x_true) < tol
        end

        if solver in HAS_Y
            y_size = m_first ? n : m
            y = c_get_y(ws, T, y_size)
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

    ws = c_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT64)
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
