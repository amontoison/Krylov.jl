export KrylovSolver, MinresSolver, CgSolver, CrSolver, SymmlqSolver, CgLanczosSolver,
CgLanczosShiftSolver, MinresQlpSolver, DqgmresSolver, DiomSolver, UsymlqSolver,
UsymqrSolver, TricgSolver, TrimrSolver, TrilqrSolver, CgsSolver, BicgstabSolver,
BilqSolver, QmrSolver, BilqrSolver, CglsSolver, CglsLanczosShiftSolver, CrlsSolver, CgneSolver,
CrmrSolver, LslqSolver, LsqrSolver, LsmrSolver, LnlqSolver, CraigSolver, CraigmrSolver,
GmresSolver, FomSolver, GpmrSolver, FgmresSolver, CarSolver, MinaresSolver

export solution, nsolution, statistics, issolved, issolved_primal, issolved_dual,
niterations, Aprod, Atprod, Bprod, warm_start!

export KrylovConstructor

import Base.size, Base.sizeof, Base.format_bytes

const KRYLOV_SOLVERS = Dict(
  :cg         => :CgSolver       ,
  :cr         => :CrSolver       ,
  :car        => :CarSolver      ,
  :symmlq     => :SymmlqSolver   ,
  :cg_lanczos => :CgLanczosSolver,
  :minares    => :MinaresSolver  ,
  :minres     => :MinresSolver   ,
  :minres_qlp => :MinresQlpSolver,
  :diom       => :DiomSolver     ,
  :fom        => :FomSolver      ,
  :dqgmres    => :DqgmresSolver  ,
  :gmres      => :GmresSolver    ,
  :fgmres     => :FgmresSolver   ,
  :gpmr       => :GpmrSolver     ,
  :usymlq     => :UsymlqSolver   ,
  :usymqr     => :UsymqrSolver   ,
  :tricg      => :TricgSolver    ,
  :trimr      => :TrimrSolver    ,
  :trilqr     => :TrilqrSolver   ,
  :cgs        => :CgsSolver      ,
  :bicgstab   => :BicgstabSolver ,
  :bilq       => :BilqSolver     ,
  :qmr        => :QmrSolver      ,
  :bilqr      => :BilqrSolver    ,
  :cgls       => :CglsSolver     ,
  :crls       => :CrlsSolver     ,
  :cgne       => :CgneSolver     ,
  :crmr       => :CrmrSolver     ,
  :lslq       => :LslqSolver     ,
  :lsqr       => :LsqrSolver     ,
  :lsmr       => :LsmrSolver     ,
  :lnlq       => :LnlqSolver     ,
  :craig      => :CraigSolver    ,
  :craigmr    => :CraigmrSolver  ,
  :cg_lanczos_shift   => :CgLanczosShiftSolver  ,
  :cgls_lanczos_shift => :CglsLanczosShiftSolver,
)

"""
    KrylovConstructor(vm; vm_empty=vm)
    KrylovConstructor(vm, vn; vm_empty=vm, vn_empty=vn)

Krylov methods require a workspace containing vectors of length `m` and `n` to solve linear problems of size `m × n`.
The `KrylovConstructor` facilitates the allocation of these vectors using `similar`.

For square problems (`m == n`), use the first constructor with a single vector `vm`.
For rectangular problems (`m ≠ n`), use the second constructor with `vm` and `vn`.

#### Input arguments

* `vm`: a vector of length `m`;
* `vn`: a vector of length `n`.

#### Keyword arguments

- `vm_empty`: an empty vector that may be replaced with a vector of length `m`;
- `vn_empty`: an empty vector that may be replaced with a vector of length `n`.

#### Note

Empty vectors `vm_empty` and `vn_empty` reduce storage requirements when features such as warm-start or preconditioners are unused.
These empty vectors will be replaced within a [`KrylovSolver`](@ref) only if required, such as when preconditioners are provided.
"""
struct KrylovConstructor{S}
  vm::S
  vn::S
  vm_empty::S
  vn_empty::S
end

function KrylovConstructor(vm; vm_empty=vm)
  return KrylovConstructor(vm, vm, vm_empty, vm_empty)
end

function KrylovConstructor(vm, vn; vm_empty=vm, vn_empty=vn)
  return KrylovConstructor(vm, vn, vm_empty, vn_empty)
end

"Abstract type for using Krylov solvers in-place"
abstract type KrylovSolver{T,FC,S} end

"""
Type for storing the vectors required by the in-place version of MINRES.

The outer constructors

    solver = MinresSolver(m, n, S; window = 5)
    solver = MinresSolver(A, b; window = 5)
    solver = MinresSolver(kc::KrylovConstructor; window = 5)

may be used in order to create these vectors.
"""
mutable struct MinresSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  r1         :: S
  r2         :: S
  npc_dir    :: S
  w1         :: S
  w2         :: S
  y          :: S
  v          :: S
  err_vec    :: Vector{T}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function MinresSolver(kc::KrylovConstructor; window::Integer = 5)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  r1 = similar(kc.vn)
  r2 = similar(kc.vn)
  npc_dir = similar(kc.vn_empty)
  w1 = similar(kc.vn)
  w2 = similar(kc.vn)
  y  = similar(kc.vn)
  v  = similar(kc.vn_empty)
  err_vec = zeros(T, window)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = MinresSolver{T,FC,S}(m, n, Δx, x, r1, r2, npc_dir, w1, w2, y, v, err_vec, false, stats)
  return solver
end

function MinresSolver(m::Integer, n::Integer, S::Type; window::Integer = 5)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  r1 = S(undef, n)
  r2 = S(undef, n)
  npc_dir = S(undef, 0)
  w1 = S(undef, n)
  w2 = S(undef, n)
  y  = S(undef, n)
  v  = S(undef, 0)
  err_vec = zeros(T, window)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = MinresSolver{T,FC,S}(m, n, Δx, x, r1, r2, npc_dir, w1, w2, y, v, err_vec, false, stats)
  return solver
end

function MinresSolver(A, b; window::Integer = 5)
  m, n = size(A)
  S = ktypeof(b)
  MinresSolver(m, n, S; window)
end

"""
Type for storing the vectors required by the in-place version of MINARES.

The outer constructors

    solver = MinaresSolver(m, n, S)
    solver = MinaresSolver(A, b)
    solver = MinaresSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct MinaresSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  vₖ         :: S
  vₖ₊₁       :: S
  x          :: S
  wₖ₋₂       :: S
  wₖ₋₁       :: S
  dₖ₋₂       :: S
  dₖ₋₁       :: S
  q          :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function MinaresSolver(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  Δx   = similar(kc.vn_empty)
  vₖ   = similar(kc.vn)
  vₖ₊₁ = similar(kc.vn)
  x    = similar(kc.vn)
  wₖ₋₂ = similar(kc.vn)
  wₖ₋₁ = similar(kc.vn)
  dₖ₋₂ = similar(kc.vn)
  dₖ₋₁ = similar(kc.vn)
  q    = similar(kc.vn)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = MinaresSolver{T,FC,S}(m, n, Δx, vₖ, vₖ₊₁, x, wₖ₋₂, wₖ₋₁, dₖ₋₂, dₖ₋₁, q, false, stats)
  return solver
end

function MinaresSolver(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  Δx   = S(undef, 0)
  vₖ   = S(undef, n)
  vₖ₊₁ = S(undef, n)
  x    = S(undef, n)
  wₖ₋₂ = S(undef, n)
  wₖ₋₁ = S(undef, n)
  dₖ₋₂ = S(undef, n)
  dₖ₋₁ = S(undef, n)
  q    = S(undef, n)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = MinaresSolver{T,FC,S}(m, n, Δx, vₖ, vₖ₊₁, x, wₖ₋₂, wₖ₋₁, dₖ₋₂, dₖ₋₁, q, false, stats)
  return solver
end

function MinaresSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  MinaresSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of CG.

The outer constructors

    solver = CgSolver(m, n, S)
    solver = CgSolver(A, b)
    solver = CgSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct CgSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  r          :: S
  p          :: S
  Ap         :: S
  z          :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function CgSolver(kc::KrylovConstructor)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  r  = similar(kc.vn)
  p  = similar(kc.vn)
  Ap = similar(kc.vn)
  z  = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CgSolver{T,FC,S}(m, n, Δx, x, r, p, Ap, z, false, stats)
  return solver
end

function CgSolver(m::Integer, n::Integer, S::Type)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  r  = S(undef, n)
  p  = S(undef, n)
  Ap = S(undef, n)
  z  = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CgSolver{T,FC,S}(m, n, Δx, x, r, p, Ap, z, false, stats)
  return solver
end

function CgSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CgSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of CR.

The outer constructors

    solver = CrSolver(m, n, S)
    solver = CrSolver(A, b)
    solver = CrSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct CrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  r          :: S
  p          :: S
  q          :: S
  Ar         :: S
  Mq         :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function CrSolver(kc::KrylovConstructor)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  r  = similar(kc.vn)
  p  = similar(kc.vn)
  q  = similar(kc.vn)
  Ar = similar(kc.vn)
  Mq = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CrSolver{T,FC,S}(m, n, Δx, x, r, p, q, Ar, Mq, false, stats)
  return solver
end

function CrSolver(m::Integer, n::Integer, S::Type)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  r  = S(undef, n)
  p  = S(undef, n)
  q  = S(undef, n)
  Ar = S(undef, n)
  Mq = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CrSolver{T,FC,S}(m, n, Δx, x, r, p, q, Ar, Mq, false, stats)
  return solver
end

function CrSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CrSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of CAR.

The outer constructors

    solver = CarSolver(m, n, S)
    solver = CarSolver(A, b)
    solver = CarSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct CarSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  r          :: S
  p          :: S
  s          :: S
  q          :: S
  t          :: S
  u          :: S
  Mu         :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function CarSolver(kc::KrylovConstructor)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  r  = similar(kc.vn)
  p  = similar(kc.vn)
  s  = similar(kc.vn)
  q  = similar(kc.vn)
  t  = similar(kc.vn)
  u  = similar(kc.vn)
  Mu = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CarSolver{T,FC,S}(m, n, Δx, x, r, p, s, q, t, u, Mu, false, stats)
  return solver
end

function CarSolver(m::Integer, n::Integer, S::Type)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  r  = S(undef, n)
  p  = S(undef, n)
  s  = S(undef, n)
  q  = S(undef, n)
  t  = S(undef, n)
  u  = S(undef, n)
  Mu = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CarSolver{T,FC,S}(m, n, Δx, x, r, p, s, q, t, u, Mu, false, stats)
  return solver
end

function CarSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CarSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of SYMMLQ.

The outer constructors

    solver = SymmlqSolver(m, n, S)
    solver = SymmlqSolver(A, b)
    solver = SymmlqSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct SymmlqSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  Mvold      :: S
  Mv         :: S
  Mv_next    :: S
  w̅          :: S
  v          :: S
  clist      :: Vector{T}
  zlist      :: Vector{T}
  sprod      :: Vector{T}
  warm_start :: Bool
  stats      :: SymmlqStats{T}
end

function SymmlqSolver(kc::KrylovConstructor; window::Integer = 5)
  S       = typeof(kc.vm)
  FC      = eltype(S)
  T       = real(FC)
  m       = length(kc.vm)
  n       = length(kc.vn)
  Δx      = similar(kc.vn_empty)
  x       = similar(kc.vn)
  Mvold   = similar(kc.vn)
  Mv      = similar(kc.vn)
  Mv_next = similar(kc.vn)
  w̅       = similar(kc.vn)
  v       = similar(kc.vn_empty)
  clist   = zeros(T, window)
  zlist   = zeros(T, window)
  sprod   = ones(T, window)
  stats = SymmlqStats(0, false, T[], Union{T, Missing}[], T[], Union{T, Missing}[], T(NaN), T(NaN), 0.0, "unknown")
  solver = SymmlqSolver{T,FC,S}(m, n, Δx, x, Mvold, Mv, Mv_next, w̅, v, clist, zlist, sprod, false, stats)
  return solver
end

function SymmlqSolver(m::Integer, n::Integer, S::Type; window::Integer = 5)
  FC      = eltype(S)
  T       = real(FC)
  Δx      = S(undef, 0)
  x       = S(undef, n)
  Mvold   = S(undef, n)
  Mv      = S(undef, n)
  Mv_next = S(undef, n)
  w̅       = S(undef, n)
  v       = S(undef, 0)
  clist   = zeros(T, window)
  zlist   = zeros(T, window)
  sprod   = ones(T, window)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SymmlqStats(0, false, T[], Union{T, Missing}[], T[], Union{T, Missing}[], T(NaN), T(NaN), 0.0, "unknown")
  solver = SymmlqSolver{T,FC,S}(m, n, Δx, x, Mvold, Mv, Mv_next, w̅, v, clist, zlist, sprod, false, stats)
  return solver
end

function SymmlqSolver(A, b; window::Integer = 5)
  m, n = size(A)
  S = ktypeof(b)
  SymmlqSolver(m, n, S; window)
end

"""
Type for storing the vectors required by the in-place version of CG-LANCZOS.

The outer constructors

    solver = CgLanczosSolver(m, n, S)
    solver = CgLanczosSolver(A, b)
    solver = CgLanczosSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct CgLanczosSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  Mv         :: S
  Mv_prev    :: S
  p          :: S
  Mv_next    :: S
  v          :: S
  warm_start :: Bool
  stats      :: LanczosStats{T}
end

function CgLanczosSolver(kc::KrylovConstructor)
  S       = typeof(kc.vm)
  FC      = eltype(S)
  T       = real(FC)
  m       = length(kc.vm)
  n       = length(kc.vn)
  Δx      = similar(kc.vn_empty)
  x       = similar(kc.vn)
  Mv      = similar(kc.vn)
  Mv_prev = similar(kc.vn)
  p       = similar(kc.vn)
  Mv_next = similar(kc.vn)
  v       = similar(kc.vn_empty)
  stats = LanczosStats(0, false, T[], false, T(NaN), T(NaN), 0.0, "unknown")
  solver = CgLanczosSolver{T,FC,S}(m, n, Δx, x, Mv, Mv_prev, p, Mv_next, v, false, stats)
  return solver
end

function CgLanczosSolver(m::Integer, n::Integer, S::Type)
  FC      = eltype(S)
  T       = real(FC)
  Δx      = S(undef, 0)
  x       = S(undef, n)
  Mv      = S(undef, n)
  Mv_prev = S(undef, n)
  p       = S(undef, n)
  Mv_next = S(undef, n)
  v       = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = LanczosStats(0, false, T[], false, T(NaN), T(NaN), 0.0, "unknown")
  solver = CgLanczosSolver{T,FC,S}(m, n, Δx, x, Mv, Mv_prev, p, Mv_next, v, false, stats)
  return solver
end

function CgLanczosSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CgLanczosSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of CG-LANCZOS-SHIFT.

The outer constructors

    solver = CgLanczosShiftSolver(m, n, nshifts, S)
    solver = CgLanczosShiftSolver(A, b, nshifts)
    solver = CgLanczosShiftSolver(kc::KrylovConstructor, nshifts)

may be used in order to create these vectors.
"""
mutable struct CgLanczosShiftSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  nshifts    :: Int
  Mv         :: S
  Mv_prev    :: S
  Mv_next    :: S
  v          :: S
  x          :: Vector{S}
  p          :: Vector{S}
  σ          :: Vector{T}
  δhat       :: Vector{T}
  ω          :: Vector{T}
  γ          :: Vector{T}
  rNorms     :: Vector{T}
  converged  :: BitVector
  not_cv     :: BitVector
  stats      :: LanczosShiftStats{T}
end

function CgLanczosShiftSolver(kc::KrylovConstructor, nshifts::Integer)
  S          = typeof(kc.vm)
  FC         = eltype(S)
  T          = real(FC)
  m          = length(kc.vm)
  n          = length(kc.vn)
  Mv         = similar(kc.vn)
  Mv_prev    = similar(kc.vn)
  Mv_next    = similar(kc.vn)
  v          = similar(kc.vn_empty)
  x          = S[similar(kc.vn) for i = 1 : nshifts]
  p          = S[similar(kc.vn) for i = 1 : nshifts]
  σ          = Vector{T}(undef, nshifts)
  δhat       = Vector{T}(undef, nshifts)
  ω          = Vector{T}(undef, nshifts)
  γ          = Vector{T}(undef, nshifts)
  rNorms     = Vector{T}(undef, nshifts)
  indefinite = BitVector(undef, nshifts)
  converged  = BitVector(undef, nshifts)
  not_cv     = BitVector(undef, nshifts)
  stats = LanczosShiftStats(0, false, Vector{T}[T[] for i = 1 : nshifts], indefinite, T(NaN), T(NaN), 0.0, "unknown")
  solver = CgLanczosShiftSolver{T,FC,S}(m, n, nshifts, Mv, Mv_prev, Mv_next, v, x, p, σ, δhat, ω, γ, rNorms, converged, not_cv, stats)
  return solver
end

function CgLanczosShiftSolver(m::Integer, n::Integer, nshifts::Integer, S::Type)
  FC         = eltype(S)
  T          = real(FC)
  Mv         = S(undef, n)
  Mv_prev    = S(undef, n)
  Mv_next    = S(undef, n)
  v          = S(undef, 0)
  x          = S[S(undef, n) for i = 1 : nshifts]
  p          = S[S(undef, n) for i = 1 : nshifts]
  σ          = Vector{T}(undef, nshifts)
  δhat       = Vector{T}(undef, nshifts)
  ω          = Vector{T}(undef, nshifts)
  γ          = Vector{T}(undef, nshifts)
  rNorms     = Vector{T}(undef, nshifts)
  indefinite = BitVector(undef, nshifts)
  converged  = BitVector(undef, nshifts)
  not_cv     = BitVector(undef, nshifts)
  S = isconcretetype(S) ? S : typeof(x)
  stats = LanczosShiftStats(0, false, Vector{T}[T[] for i = 1 : nshifts], indefinite, T(NaN), T(NaN), 0.0, "unknown")
  solver = CgLanczosShiftSolver{T,FC,S}(m, n, nshifts, Mv, Mv_prev, Mv_next, v, x, p, σ, δhat, ω, γ, rNorms, converged, not_cv, stats)
  return solver
end

function CgLanczosShiftSolver(A, b, nshifts::Integer)
  m, n = size(A)
  S = ktypeof(b)
  CgLanczosShiftSolver(m, n, nshifts, S)
end

"""
Type for storing the vectors required by the in-place version of MINRES-QLP.

The outer constructors

    solver = MinresQlpSolver(m, n, S)
    solver = MinresQlpSolver(A, b)
    solver = MinresQlpSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct MinresQlpSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  wₖ₋₁       :: S
  wₖ         :: S
  M⁻¹vₖ₋₁    :: S
  M⁻¹vₖ      :: S
  x          :: S
  p          :: S
  vₖ         :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function MinresQlpSolver(kc::KrylovConstructor)
  S       = typeof(kc.vm)
  FC      = eltype(S)
  T       = real(FC)
  m       = length(kc.vm)
  n       = length(kc.vn)
  Δx      = similar(kc.vn_empty)
  wₖ₋₁    = similar(kc.vn)
  wₖ      = similar(kc.vn)
  M⁻¹vₖ₋₁ = similar(kc.vn)
  M⁻¹vₖ   = similar(kc.vn)
  x       = similar(kc.vn)
  p       = similar(kc.vn)
  vₖ      = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = MinresQlpSolver{T,FC,S}(m, n, Δx, wₖ₋₁, wₖ, M⁻¹vₖ₋₁, M⁻¹vₖ, x, p, vₖ, false, stats)
  return solver
end

function MinresQlpSolver(m::Integer, n::Integer, S::Type)
  FC      = eltype(S)
  T       = real(FC)
  Δx      = S(undef, 0)
  wₖ₋₁    = S(undef, n)
  wₖ      = S(undef, n)
  M⁻¹vₖ₋₁ = S(undef, n)
  M⁻¹vₖ   = S(undef, n)
  x       = S(undef, n)
  p       = S(undef, n)
  vₖ      = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = MinresQlpSolver{T,FC,S}(m, n, Δx, wₖ₋₁, wₖ, M⁻¹vₖ₋₁, M⁻¹vₖ, x, p, vₖ, false, stats)
  return solver
end

function MinresQlpSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  MinresQlpSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of DQGMRES.

The outer constructors

    solver = DqgmresSolver(m, n, S; memory = 20)
    solver = DqgmresSolver(A, b; memory = 20)
    solver = DqgmresSolver(kc::KrylovConstructor; memory = 20)

may be used in order to create these vectors.
`memory` is set to `n` if the value given is larger than `n`.
"""
mutable struct DqgmresSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  t          :: S
  z          :: S
  w          :: S
  P          :: Vector{S}
  V          :: Vector{S}
  c          :: Vector{T}
  s          :: Vector{FC}
  H          :: Vector{FC}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function DqgmresSolver(kc::KrylovConstructor; memory::Integer = 20)
  S      = typeof(kc.vm)
  FC     = eltype(S)
  T      = real(FC)
  m      = length(kc.vm)
  n      = length(kc.vn)
  memory = min(m, memory)
  Δx     = similar(kc.vn_empty)
  x      = similar(kc.vn)
  t      = similar(kc.vn)
  z      = similar(kc.vn_empty)
  w      = similar(kc.vn_empty)
  P      = S[similar(kc.vn) for i = 1 : memory]
  V      = S[similar(kc.vn) for i = 1 : memory]
  c      = Vector{T}(undef, memory)
  s      = Vector{FC}(undef, memory)
  H      = Vector{FC}(undef, memory+1)
  stats  = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = DqgmresSolver{T,FC,S}(m, n, Δx, x, t, z, w, P, V, c, s, H, false, stats)
  return solver
end

function DqgmresSolver(m::Integer, n::Integer, S::Type; memory::Integer = 20)
  memory = min(m, memory)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  t  = S(undef, n)
  z  = S(undef, 0)
  w  = S(undef, 0)
  P  = S[S(undef, n) for i = 1 : memory]
  V  = S[S(undef, n) for i = 1 : memory]
  c  = Vector{T}(undef, memory)
  s  = Vector{FC}(undef, memory)
  H  = Vector{FC}(undef, memory+1)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = DqgmresSolver{T,FC,S}(m, n, Δx, x, t, z, w, P, V, c, s, H, false, stats)
  return solver
end

function DqgmresSolver(A, b; memory::Integer = 20)
  m, n = size(A)
  S = ktypeof(b)
  DqgmresSolver(m, n, S; memory)
end

"""
Type for storing the vectors required by the in-place version of DIOM.

The outer constructors

    solver = DiomSolver(m, n, S; memory = 20)
    solver = DiomSolver(A, b; memory = 20)
    solver = DiomSolver(kc::KrylovConstructor; memory = 20)

may be used in order to create these vectors.
`memory` is set to `n` if the value given is larger than `n`.
"""
mutable struct DiomSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  t          :: S
  z          :: S
  w          :: S
  P          :: Vector{S}
  V          :: Vector{S}
  L          :: Vector{FC}
  H          :: Vector{FC}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function DiomSolver(kc::KrylovConstructor; memory::Integer = 20)
  S      = typeof(kc.vm)
  FC     = eltype(S)
  T      = real(FC)
  m      = length(kc.vm)
  n      = length(kc.vn)
  memory = min(m, memory)
  Δx     = similar(kc.vn_empty)
  x      = similar(kc.vn)
  t      = similar(kc.vn)
  z      = similar(kc.vn_empty)
  w      = similar(kc.vn_empty)
  P      = S[similar(kc.vn) for i = 1 : memory-1]
  V      = S[similar(kc.vn) for i = 1 : memory]
  L      = Vector{FC}(undef, memory-1)
  H      = Vector{FC}(undef, memory)
  stats  = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = DiomSolver{T,FC,S}(m, n, Δx, x, t, z, w, P, V, L, H, false, stats)
  return solver
end

function DiomSolver(m::Integer, n::Integer, S::Type; memory::Integer = 20)
  memory = min(m, memory)
  FC  = eltype(S)
  T   = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  t  = S(undef, n)
  z  = S(undef, 0)
  w  = S(undef, 0)
  P  = S[S(undef, n) for i = 1 : memory-1]
  V  = S[S(undef, n) for i = 1 : memory]
  L  = Vector{FC}(undef, memory-1)
  H  = Vector{FC}(undef, memory)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = DiomSolver{T,FC,S}(m, n, Δx, x, t, z, w, P, V, L, H, false, stats)
  return solver
end

function DiomSolver(A, b; memory::Integer = 20)
  m, n = size(A)
  S = ktypeof(b)
  DiomSolver(m, n, S; memory)
end

"""
Type for storing the vectors required by the in-place version of USYMLQ.

The outer constructors

    solver = UsymlqSolver(m, n, S)
    solver = UsymlqSolver(A, b)
    solver = UsymlqSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct UsymlqSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  uₖ₋₁       :: S
  uₖ         :: S
  p          :: S
  Δx         :: S
  x          :: S
  d̅          :: S
  vₖ₋₁       :: S
  vₖ         :: S
  q          :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function UsymlqSolver(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  uₖ₋₁ = similar(kc.vn)
  uₖ   = similar(kc.vn)
  p    = similar(kc.vn)
  Δx   = similar(kc.vn_empty)
  x    = similar(kc.vn)
  d̅    = similar(kc.vn)
  vₖ₋₁ = similar(kc.vm)
  vₖ   = similar(kc.vm)
  q    = similar(kc.vm)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = UsymlqSolver{T,FC,S}(m, n, uₖ₋₁, uₖ, p, Δx, x, d̅, vₖ₋₁, vₖ, q, false, stats)
  return solver
end

function UsymlqSolver(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  uₖ₋₁ = S(undef, n)
  uₖ   = S(undef, n)
  p    = S(undef, n)
  Δx   = S(undef, 0)
  x    = S(undef, n)
  d̅    = S(undef, n)
  vₖ₋₁ = S(undef, m)
  vₖ   = S(undef, m)
  q    = S(undef, m)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = UsymlqSolver{T,FC,S}(m, n, uₖ₋₁, uₖ, p, Δx, x, d̅, vₖ₋₁, vₖ, q, false, stats)
  return solver
end

function UsymlqSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  UsymlqSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of USYMQR.

The outer constructors

    solver = UsymqrSolver(m, n, S)
    solver = UsymqrSolver(A, b)
    solver = UsymqrSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct UsymqrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  vₖ₋₁       :: S
  vₖ         :: S
  q          :: S
  Δx         :: S
  x          :: S
  wₖ₋₂       :: S
  wₖ₋₁       :: S
  uₖ₋₁       :: S
  uₖ         :: S
  p          :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function UsymqrSolver(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  vₖ₋₁ = similar(kc.vm)
  vₖ   = similar(kc.vm)
  q    = similar(kc.vm)
  Δx   = similar(kc.vn_empty)
  x    = similar(kc.vn)
  wₖ₋₂ = similar(kc.vn)
  wₖ₋₁ = similar(kc.vn)
  uₖ₋₁ = similar(kc.vn)
  uₖ   = similar(kc.vn)
  p    = similar(kc.vn)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = UsymqrSolver{T,FC,S}(m, n, vₖ₋₁, vₖ, q, Δx, x, wₖ₋₂, wₖ₋₁, uₖ₋₁, uₖ, p, false, stats)
  return solver
end

function UsymqrSolver(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  vₖ₋₁ = S(undef, m)
  vₖ   = S(undef, m)
  q    = S(undef, m)
  Δx   = S(undef, 0)
  x    = S(undef, n)
  wₖ₋₂ = S(undef, n)
  wₖ₋₁ = S(undef, n)
  uₖ₋₁ = S(undef, n)
  uₖ   = S(undef, n)
  p    = S(undef, n)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = UsymqrSolver{T,FC,S}(m, n, vₖ₋₁, vₖ, q, Δx, x, wₖ₋₂, wₖ₋₁, uₖ₋₁, uₖ, p, false, stats)
  return solver
end

function UsymqrSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  UsymqrSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of TRICG.

The outer constructors

    solver = TricgSolver(m, n, S)
    solver = TricgSolver(A, b)
    solver = TricgSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct TricgSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  y          :: S
  N⁻¹uₖ₋₁    :: S
  N⁻¹uₖ      :: S
  p          :: S
  gy₂ₖ₋₁     :: S
  gy₂ₖ       :: S
  x          :: S
  M⁻¹vₖ₋₁    :: S
  M⁻¹vₖ      :: S
  q          :: S
  gx₂ₖ₋₁     :: S
  gx₂ₖ       :: S
  Δx         :: S
  Δy         :: S
  uₖ         :: S
  vₖ         :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function TricgSolver(kc::KrylovConstructor)
  S       = typeof(kc.vm)
  FC      = eltype(S)
  T       = real(FC)
  m       = length(kc.vm)
  n       = length(kc.vn)
  y       = similar(kc.vn)
  N⁻¹uₖ₋₁ = similar(kc.vn)
  N⁻¹uₖ   = similar(kc.vn)
  p       = similar(kc.vn)
  gy₂ₖ₋₁  = similar(kc.vn)
  gy₂ₖ    = similar(kc.vn)
  x       = similar(kc.vm)
  M⁻¹vₖ₋₁ = similar(kc.vm)
  M⁻¹vₖ   = similar(kc.vm)
  q       = similar(kc.vm)
  gx₂ₖ₋₁  = similar(kc.vm)
  gx₂ₖ    = similar(kc.vm)
  Δx      = similar(kc.vm_empty)
  Δy      = similar(kc.vn_empty)
  uₖ      = similar(kc.vn_empty)
  vₖ      = similar(kc.vm_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = TricgSolver{T,FC,S}(m, n, y, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₁, gy₂ₖ, x, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₁, gx₂ₖ, Δx, Δy, uₖ, vₖ, false, stats)
  return solver
end

function TricgSolver(m::Integer, n::Integer, S::Type)
  FC      = eltype(S)
  T       = real(FC)
  y       = S(undef, n)
  N⁻¹uₖ₋₁ = S(undef, n)
  N⁻¹uₖ   = S(undef, n)
  p       = S(undef, n)
  gy₂ₖ₋₁  = S(undef, n)
  gy₂ₖ    = S(undef, n)
  x       = S(undef, m)
  M⁻¹vₖ₋₁ = S(undef, m)
  M⁻¹vₖ   = S(undef, m)
  q       = S(undef, m)
  gx₂ₖ₋₁  = S(undef, m)
  gx₂ₖ    = S(undef, m)
  Δx      = S(undef, 0)
  Δy      = S(undef, 0)
  uₖ      = S(undef, 0)
  vₖ      = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = TricgSolver{T,FC,S}(m, n, y, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₁, gy₂ₖ, x, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₁, gx₂ₖ, Δx, Δy, uₖ, vₖ, false, stats)
  return solver
end

function TricgSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  TricgSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of TRIMR.

The outer constructors

    solver = TrimrSolver(m, n, S)
    solver = TrimrSolver(A, b)
    solver = TrimrSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct TrimrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  y          :: S
  N⁻¹uₖ₋₁    :: S
  N⁻¹uₖ      :: S
  p          :: S
  gy₂ₖ₋₃     :: S
  gy₂ₖ₋₂     :: S
  gy₂ₖ₋₁     :: S
  gy₂ₖ       :: S
  x          :: S
  M⁻¹vₖ₋₁    :: S
  M⁻¹vₖ      :: S
  q          :: S
  gx₂ₖ₋₃     :: S
  gx₂ₖ₋₂     :: S
  gx₂ₖ₋₁     :: S
  gx₂ₖ       :: S
  Δx         :: S
  Δy         :: S
  uₖ         :: S
  vₖ         :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function TrimrSolver(kc::KrylovConstructor)
  S       = typeof(kc.vm)
  FC      = eltype(S)
  T       = real(FC)
  m       = length(kc.vm)
  n       = length(kc.vn)
  y       = similar(kc.vn)
  N⁻¹uₖ₋₁ = similar(kc.vn)
  N⁻¹uₖ   = similar(kc.vn)
  p       = similar(kc.vn)
  gy₂ₖ₋₃  = similar(kc.vn)
  gy₂ₖ₋₂  = similar(kc.vn)
  gy₂ₖ₋₁  = similar(kc.vn)
  gy₂ₖ    = similar(kc.vn)
  x       = similar(kc.vm)
  M⁻¹vₖ₋₁ = similar(kc.vm)
  M⁻¹vₖ   = similar(kc.vm)
  q       = similar(kc.vm)
  gx₂ₖ₋₃  = similar(kc.vm)
  gx₂ₖ₋₂  = similar(kc.vm)
  gx₂ₖ₋₁  = similar(kc.vm)
  gx₂ₖ    = similar(kc.vm)
  Δx      = similar(kc.vm_empty)
  Δy      = similar(kc.vn_empty)
  uₖ      = similar(kc.vn_empty)
  vₖ      = similar(kc.vm_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = TrimrSolver{T,FC,S}(m, n, y, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₃, gy₂ₖ₋₂, gy₂ₖ₋₁, gy₂ₖ, x, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₃, gx₂ₖ₋₂, gx₂ₖ₋₁, gx₂ₖ, Δx, Δy, uₖ, vₖ, false, stats)
  return solver
end

function TrimrSolver(m::Integer, n::Integer, S::Type)
  FC      = eltype(S)
  T       = real(FC)
  y       = S(undef, n)
  N⁻¹uₖ₋₁ = S(undef, n)
  N⁻¹uₖ   = S(undef, n)
  p       = S(undef, n)
  gy₂ₖ₋₃  = S(undef, n)
  gy₂ₖ₋₂  = S(undef, n)
  gy₂ₖ₋₁  = S(undef, n)
  gy₂ₖ    = S(undef, n)
  x       = S(undef, m)
  M⁻¹vₖ₋₁ = S(undef, m)
  M⁻¹vₖ   = S(undef, m)
  q       = S(undef, m)
  gx₂ₖ₋₃  = S(undef, m)
  gx₂ₖ₋₂  = S(undef, m)
  gx₂ₖ₋₁  = S(undef, m)
  gx₂ₖ    = S(undef, m)
  Δx      = S(undef, 0)
  Δy      = S(undef, 0)
  uₖ      = S(undef, 0)
  vₖ      = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = TrimrSolver{T,FC,S}(m, n, y, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₃, gy₂ₖ₋₂, gy₂ₖ₋₁, gy₂ₖ, x, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₃, gx₂ₖ₋₂, gx₂ₖ₋₁, gx₂ₖ, Δx, Δy, uₖ, vₖ, false, stats)
  return solver
end

function TrimrSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  TrimrSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of TRILQR.

The outer constructors

    solver = TrilqrSolver(m, n, S)
    solver = TrilqrSolver(A, b)
    solver = TrilqrSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct TrilqrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  uₖ₋₁       :: S
  uₖ         :: S
  p          :: S
  d̅          :: S
  Δx         :: S
  x          :: S
  vₖ₋₁       :: S
  vₖ         :: S
  q          :: S
  Δy         :: S
  y          :: S
  wₖ₋₃       :: S
  wₖ₋₂       :: S
  warm_start :: Bool
  stats      :: AdjointStats{T}
end

function TrilqrSolver(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  uₖ₋₁ = similar(kc.vn)
  uₖ   = similar(kc.vn)
  p    = similar(kc.vn)
  d̅    = similar(kc.vn)
  Δx   = similar(kc.vn_empty)
  x    = similar(kc.vn)
  vₖ₋₁ = similar(kc.vm)
  vₖ   = similar(kc.vm)
  q    = similar(kc.vm)
  Δy   = similar(kc.vm_empty)
  y    = similar(kc.vm)
  wₖ₋₃ = similar(kc.vm)
  wₖ₋₂ = similar(kc.vm)
  stats = AdjointStats(0, false, false, T[], T[], 0.0, "unknown")
  solver = TrilqrSolver{T,FC,S}(m, n, uₖ₋₁, uₖ, p, d̅, Δx, x, vₖ₋₁, vₖ, q, Δy, y, wₖ₋₃, wₖ₋₂, false, stats)
  return solver
end

function TrilqrSolver(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  uₖ₋₁ = S(undef, n)
  uₖ   = S(undef, n)
  p    = S(undef, n)
  d̅    = S(undef, n)
  Δx   = S(undef, 0)
  x    = S(undef, n)
  vₖ₋₁ = S(undef, m)
  vₖ   = S(undef, m)
  q    = S(undef, m)
  Δy   = S(undef, 0)
  y    = S(undef, m)
  wₖ₋₃ = S(undef, m)
  wₖ₋₂ = S(undef, m)
  S = isconcretetype(S) ? S : typeof(x)
  stats = AdjointStats(0, false, false, T[], T[], 0.0, "unknown")
  solver = TrilqrSolver{T,FC,S}(m, n, uₖ₋₁, uₖ, p, d̅, Δx, x, vₖ₋₁, vₖ, q, Δy, y, wₖ₋₃, wₖ₋₂, false, stats)
  return solver
end

function TrilqrSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  TrilqrSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of CGS.

The outer constructorss

    solver = CgsSolver(m, n, S)
    solver = CgsSolver(A, b)
    solver = CgsSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct CgsSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  r          :: S
  u          :: S
  p          :: S
  q          :: S
  ts         :: S
  yz         :: S
  vw         :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function CgsSolver(kc::KrylovConstructor)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  r  = similar(kc.vn)
  u  = similar(kc.vn)
  p  = similar(kc.vn)
  q  = similar(kc.vn)
  ts = similar(kc.vn)
  yz = similar(kc.vn_empty)
  vw = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CgsSolver{T,FC,S}(m, n, Δx, x, r, u, p, q, ts, yz, vw, false, stats)
  return solver
end

function CgsSolver(m::Integer, n::Integer, S::Type)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  r  = S(undef, n)
  u  = S(undef, n)
  p  = S(undef, n)
  q  = S(undef, n)
  ts = S(undef, n)
  yz = S(undef, 0)
  vw = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CgsSolver{T,FC,S}(m, n, Δx, x, r, u, p, q, ts, yz, vw, false, stats)
  return solver
end

function CgsSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CgsSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of BICGSTAB.

The outer constructors

    solver = BicgstabSolver(m, n, S)
    solver = BicgstabSolver(A, b)
    solver = BicgstabSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct BicgstabSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  r          :: S
  p          :: S
  v          :: S
  s          :: S
  qd         :: S
  yz         :: S
  t          :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function BicgstabSolver(kc::KrylovConstructor)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  r  = similar(kc.vn)
  p  = similar(kc.vn)
  v  = similar(kc.vn)
  s  = similar(kc.vn)
  qd = similar(kc.vn)
  yz = similar(kc.vn_empty)
  t  = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = BicgstabSolver{T,FC,S}(m, n, Δx, x, r, p, v, s, qd, yz, t, false, stats)
  return solver
end

function BicgstabSolver(m::Integer, n::Integer, S::Type)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  r  = S(undef, n)
  p  = S(undef, n)
  v  = S(undef, n)
  s  = S(undef, n)
  qd = S(undef, n)
  yz = S(undef, 0)
  t  = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = BicgstabSolver{T,FC,S}(m, n, Δx, x, r, p, v, s, qd, yz, t, false, stats)
  return solver
end

function BicgstabSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  BicgstabSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of BILQ.

The outer constructors

    solver = BilqSolver(m, n, S)
    solver = BilqSolver(A, b)
    solver = BilqSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct BilqSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  uₖ₋₁       :: S
  uₖ         :: S
  q          :: S
  vₖ₋₁       :: S
  vₖ         :: S
  p          :: S
  Δx         :: S
  x          :: S
  d̅          :: S
  t          :: S
  s          :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function BilqSolver(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  uₖ₋₁ = similar(kc.vn)
  uₖ   = similar(kc.vn)
  q    = similar(kc.vn)
  vₖ₋₁ = similar(kc.vn)
  vₖ   = similar(kc.vn)
  p    = similar(kc.vn)
  Δx   = similar(kc.vn_empty)
  x    = similar(kc.vn)
  d̅    = similar(kc.vn)
  t    = similar(kc.vn_empty)
  s    = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = BilqSolver{T,FC,S}(m, n, uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, Δx, x, d̅, t, s, false, stats)
  return solver
end

function BilqSolver(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  uₖ₋₁ = S(undef, n)
  uₖ   = S(undef, n)
  q    = S(undef, n)
  vₖ₋₁ = S(undef, n)
  vₖ   = S(undef, n)
  p    = S(undef, n)
  Δx   = S(undef, 0)
  x    = S(undef, n)
  d̅    = S(undef, n)
  t    = S(undef, 0)
  s    = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = BilqSolver{T,FC,S}(m, n, uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, Δx, x, d̅, t, s, false, stats)
  return solver
end

function BilqSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  BilqSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of QMR.

The outer constructors

    solver = QmrSolver(m, n, S)
    solver = QmrSolver(A, b)
    solver = QmrSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct QmrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  uₖ₋₁       :: S
  uₖ         :: S
  q          :: S
  vₖ₋₁       :: S
  vₖ         :: S
  p          :: S
  Δx         :: S
  x          :: S
  wₖ₋₂       :: S
  wₖ₋₁       :: S
  t          :: S
  s          :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function QmrSolver(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  uₖ₋₁ = similar(kc.vn)
  uₖ   = similar(kc.vn)
  q    = similar(kc.vn)
  vₖ₋₁ = similar(kc.vn)
  vₖ   = similar(kc.vn)
  p    = similar(kc.vn)
  Δx   = similar(kc.vn_empty)
  x    = similar(kc.vn)
  wₖ₋₂ = similar(kc.vn)
  wₖ₋₁ = similar(kc.vn)
  t    = similar(kc.vn_empty)
  s    = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = QmrSolver{T,FC,S}(m, n, uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, Δx, x, wₖ₋₂, wₖ₋₁, t, s, false, stats)
  return solver
end

function QmrSolver(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  uₖ₋₁ = S(undef, n)
  uₖ   = S(undef, n)
  q    = S(undef, n)
  vₖ₋₁ = S(undef, n)
  vₖ   = S(undef, n)
  p    = S(undef, n)
  Δx   = S(undef, 0)
  x    = S(undef, n)
  wₖ₋₂ = S(undef, n)
  wₖ₋₁ = S(undef, n)
  t    = S(undef, 0)
  s    = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = QmrSolver{T,FC,S}(m, n, uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, Δx, x, wₖ₋₂, wₖ₋₁, t, s, false, stats)
  return solver
end

function QmrSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  QmrSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of BILQR.

The outer constructors

    solver = BilqrSolver(m, n, S)
    solver = BilqrSolver(A, b)
    solver = BilqrSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct BilqrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  uₖ₋₁       :: S
  uₖ         :: S
  q          :: S
  vₖ₋₁       :: S
  vₖ         :: S
  p          :: S
  Δx         :: S
  x          :: S
  Δy         :: S
  y          :: S
  d̅          :: S
  wₖ₋₃       :: S
  wₖ₋₂       :: S
  warm_start :: Bool
  stats      :: AdjointStats{T}
end

function BilqrSolver(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  uₖ₋₁ = similar(kc.vn)
  uₖ   = similar(kc.vn)
  q    = similar(kc.vn)
  vₖ₋₁ = similar(kc.vn)
  vₖ   = similar(kc.vn)
  p    = similar(kc.vn)
  Δx   = similar(kc.vn_empty)
  x    = similar(kc.vn)
  Δy   = similar(kc.vn_empty)
  y    = similar(kc.vn)
  d̅    = similar(kc.vn)
  wₖ₋₃ = similar(kc.vn)
  wₖ₋₂ = similar(kc.vn)
  stats = AdjointStats(0, false, false, T[], T[], 0.0, "unknown")
  solver = BilqrSolver{T,FC,S}(m, n, uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, Δx, x, Δy, y, d̅, wₖ₋₃, wₖ₋₂, false, stats)
  return solver
end

function BilqrSolver(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  uₖ₋₁ = S(undef, n)
  uₖ   = S(undef, n)
  q    = S(undef, n)
  vₖ₋₁ = S(undef, n)
  vₖ   = S(undef, n)
  p    = S(undef, n)
  Δx   = S(undef, 0)
  x    = S(undef, n)
  Δy   = S(undef, 0)
  y    = S(undef, n)
  d̅    = S(undef, n)
  wₖ₋₃ = S(undef, n)
  wₖ₋₂ = S(undef, n)
  S = isconcretetype(S) ? S : typeof(x)
  stats = AdjointStats(0, false, false, T[], T[], 0.0, "unknown")
  solver = BilqrSolver{T,FC,S}(m, n, uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, Δx, x, Δy, y, d̅, wₖ₋₃, wₖ₋₂, false, stats)
  return solver
end

function BilqrSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  BilqrSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of CGLS.

The outer constructors

    solver = CglsSolver(m, n, S)
    solver = CglsSolver(A, b)
    solver = CglsSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct CglsSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m     :: Int
  n     :: Int
  x     :: S
  p     :: S
  s     :: S
  r     :: S
  q     :: S
  Mr    :: S
  stats :: SimpleStats{T}
end

function CglsSolver(kc::KrylovConstructor)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  x  = similar(kc.vn)
  p  = similar(kc.vn)
  s  = similar(kc.vn)
  r  = similar(kc.vm)
  q  = similar(kc.vm)
  Mr = similar(kc.vm_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CglsSolver{T,FC,S}(m, n, x, p, s, r, q, Mr, stats)
  return solver
end

function CglsSolver(m::Integer, n::Integer, S::Type)
  FC = eltype(S)
  T  = real(FC)
  x  = S(undef, n)
  p  = S(undef, n)
  s  = S(undef, n)
  r  = S(undef, m)
  q  = S(undef, m)
  Mr = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CglsSolver{T,FC,S}(m, n, x, p, s, r, q, Mr, stats)
  return solver
end

function CglsSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CglsSolver(m, n, S)
end

"""
Workspace for the in-place version of CGLS-LANCZOS-SHIFT.

The outer constructors:

    solver = CglsLanczosShiftSolver(m, n, nshifts, S)
    solver = CglsLanczosShiftSolver(A, b, nshifts)
    solver = CglsLanczosShiftSolver(kc::KrylovConstructor, nshifts)

can be used to initialize this workspace.
"""
mutable struct CglsLanczosShiftSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m         :: Int
  n         :: Int
  nshifts   :: Int
  Mv        :: S
  u_prev    :: S
  u_next    :: S
  u         :: S
  v         :: S
  x         :: Vector{S}
  p         :: Vector{S}
  σ         :: Vector{T}
  δhat      :: Vector{T}
  ω         :: Vector{T}
  γ         :: Vector{T}
  rNorms    :: Vector{T}
  converged :: BitVector
  not_cv    :: BitVector
  stats     :: LanczosShiftStats{T}
end

function CglsLanczosShiftSolver(kc::KrylovConstructor, nshifts::Integer)
  S          = typeof(kc.vm)
  FC         = eltype(S)
  T          = real(FC)
  m          = length(kc.vm)
  n          = length(kc.vn)
  Mv         = similar(kc.vn)
  u_prev     = similar(kc.vm)
  u_next     = similar(kc.vm)
  u          = similar(kc.vm)
  v          = similar(kc.vn_empty)
  x          = S[similar(kc.vn) for i = 1 : nshifts]
  p          = S[similar(kc.vn) for i = 1 : nshifts]
  σ          = Vector{T}(undef, nshifts)
  δhat       = Vector{T}(undef, nshifts)
  ω          = Vector{T}(undef, nshifts)
  γ          = Vector{T}(undef, nshifts)
  rNorms     = Vector{T}(undef, nshifts)
  indefinite = BitVector(undef, nshifts)
  converged  = BitVector(undef, nshifts)
  not_cv     = BitVector(undef, nshifts)
  stats = LanczosShiftStats(0, false, Vector{T}[T[] for i = 1 : nshifts], indefinite, T(NaN), T(NaN), 0.0, "unknown")
  solver = CglsLanczosShiftSolver{T,FC,S}(m, n, nshifts, Mv, u_prev, u_next, u, v, x, p, σ, δhat, ω, γ, rNorms, converged, not_cv, stats)
  return solver
end

function CglsLanczosShiftSolver(m::Integer, n::Integer, nshifts::Integer, S::Type)
  FC         = eltype(S)
  T          = real(FC)
  Mv         = S(undef, n)
  u_prev     = S(undef, m)
  u_next     = S(undef, m)
  u          = S(undef, m)
  v          = S(undef, 0)
  x          = S[S(undef, n) for i = 1 : nshifts]
  p          = S[S(undef, n) for i = 1 : nshifts]
  σ          = Vector{T}(undef, nshifts)
  δhat       = Vector{T}(undef, nshifts)
  ω          = Vector{T}(undef, nshifts)
  γ          = Vector{T}(undef, nshifts)
  rNorms     = Vector{T}(undef, nshifts)
  indefinite = BitVector(undef, nshifts)
  converged  = BitVector(undef, nshifts)
  not_cv     = BitVector(undef, nshifts)
  S = isconcretetype(S) ? S : typeof(x)
  stats = LanczosShiftStats(0, false, Vector{T}[T[] for i = 1 : nshifts], indefinite, T(NaN), T(NaN), 0.0, "unknown")
  solver = CglsLanczosShiftSolver{T,FC,S}(m, n, nshifts, Mv, u_prev, u_next, u, v, x, p, σ, δhat, ω, γ, rNorms, converged, not_cv, stats)
  return solver
end

function CglsLanczosShiftSolver(A, b, nshifts::Integer)
  m, n = size(A)
  S = ktypeof(b)
  CglsLanczosShiftSolver(m, n, nshifts, S)
end

"""
Type for storing the vectors required by the in-place version of CRLS.

The outer constructors

    solver = CrlsSolver(m, n, S)
    solver = CrlsSolver(A, b)
    solver = CrlsSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct CrlsSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m     :: Int
  n     :: Int
  x     :: S
  p     :: S
  Ar    :: S
  q     :: S
  r     :: S
  Ap    :: S
  s     :: S
  Ms    :: S
  stats :: SimpleStats{T}
end

function CrlsSolver(kc::KrylovConstructor)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  x  = similar(kc.vn)
  p  = similar(kc.vn)
  Ar = similar(kc.vn)
  q  = similar(kc.vn)
  r  = similar(kc.vm)
  Ap = similar(kc.vm)
  s  = similar(kc.vm)
  Ms = similar(kc.vm_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CrlsSolver{T,FC,S}(m, n, x, p, Ar, q, r, Ap, s, Ms, stats)
  return solver
end

function CrlsSolver(m::Integer, n::Integer, S::Type)
  FC = eltype(S)
  T  = real(FC)
  x  = S(undef, n)
  p  = S(undef, n)
  Ar = S(undef, n)
  q  = S(undef, n)
  r  = S(undef, m)
  Ap = S(undef, m)
  s  = S(undef, m)
  Ms = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CrlsSolver{T,FC,S}(m, n, x, p, Ar, q, r, Ap, s, Ms, stats)
  return solver
end

function CrlsSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CrlsSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of CGNE.

The outer constructors

    solver = CgneSolver(m, n, S)
    solver = CgneSolver(A, b)
    solver = CgneSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct CgneSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m     :: Int
  n     :: Int
  x     :: S
  p     :: S
  Aᴴz   :: S
  r     :: S
  q     :: S
  s     :: S
  z     :: S
  stats :: SimpleStats{T}
end

function CgneSolver(kc::KrylovConstructor)
  S   = typeof(kc.vm)
  FC  = eltype(S)
  T   = real(FC)
  m   = length(kc.vm)
  n   = length(kc.vn)
  x   = similar(kc.vn)
  p   = similar(kc.vn)
  Aᴴz = similar(kc.vn)
  r   = similar(kc.vm)
  q   = similar(kc.vm)
  s   = similar(kc.vm_empty)
  z   = similar(kc.vm_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CgneSolver{T,FC,S}(m, n, x, p, Aᴴz, r, q, s, z, stats)
  return solver
end

function CgneSolver(m::Integer, n::Integer, S::Type)
  FC  = eltype(S)
  T   = real(FC)
  x   = S(undef, n)
  p   = S(undef, n)
  Aᴴz = S(undef, n)
  r   = S(undef, m)
  q   = S(undef, m)
  s   = S(undef, 0)
  z   = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CgneSolver{T,FC,S}(m, n, x, p, Aᴴz, r, q, s, z, stats)
  return solver
end

function CgneSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CgneSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of CRMR.

The outer constructors

    solver = CrmrSolver(m, n, S)
    solver = CrmrSolver(A, b)
    solver = CrmrSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct CrmrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m     :: Int
  n     :: Int
  x     :: S
  p     :: S
  Aᴴr   :: S
  r     :: S
  q     :: S
  Nq    :: S
  s     :: S
  stats :: SimpleStats{T}
end

function CrmrSolver(kc::KrylovConstructor)
  S   = typeof(kc.vm)
  FC  = eltype(S)
  T   = real(FC)
  m   = length(kc.vm)
  n   = length(kc.vn)
  x   = similar(kc.vn)
  p   = similar(kc.vn)
  Aᴴr = similar(kc.vn)
  r   = similar(kc.vm)
  q   = similar(kc.vm)
  Nq  = similar(kc.vm_empty)
  s   = similar(kc.vm_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CrmrSolver{T,FC,S}(m, n, x, p, Aᴴr, r, q, Nq, s, stats)
  return solver
end

function CrmrSolver(m::Integer, n::Integer, S::Type)
  FC  = eltype(S)
  T   = real(FC)
  x   = S(undef, n)
  p   = S(undef, n)
  Aᴴr = S(undef, n)
  r   = S(undef, m)
  q   = S(undef, m)
  Nq  = S(undef, 0)
  s   = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CrmrSolver{T,FC,S}(m, n, x, p, Aᴴr, r, q, Nq, s, stats)
  return solver
end

function CrmrSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CrmrSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of LSLQ.

The outer constructors

    solver = LslqSolver(m, n, S)
    solver = LslqSolver(A, b)
    solver = LslqSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct LslqSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m       :: Int
  n       :: Int
  x       :: S
  Nv      :: S
  Aᴴu     :: S
  w̄       :: S
  Mu      :: S
  Av      :: S
  u       :: S
  v       :: S
  err_vec :: Vector{T}
  stats   :: LSLQStats{T}
end

function LslqSolver(kc::KrylovConstructor; window::Integer = 5)
  S   = typeof(kc.vm)
  FC  = eltype(S)
  T   = real(FC)
  m   = length(kc.vm)
  n   = length(kc.vn)
  x   = similar(kc.vn)
  Nv  = similar(kc.vn)
  Aᴴu = similar(kc.vn)
  w̄   = similar(kc.vn)
  Mu  = similar(kc.vm)
  Av  = similar(kc.vm)
  u   = similar(kc.vm_empty)
  v   = similar(kc.vn_empty)
  err_vec = zeros(T, window)
  stats = LSLQStats(0, false, false, T[], T[], T[], false, T[], T[], 0.0, "unknown")
  solver = LslqSolver{T,FC,S}(m, n, x, Nv, Aᴴu, w̄, Mu, Av, u, v, err_vec, stats)
  return solver
end

function LslqSolver(m::Integer, n::Integer, S::Type; window::Integer = 5)
  FC  = eltype(S)
  T   = real(FC)
  x   = S(undef, n)
  Nv  = S(undef, n)
  Aᴴu = S(undef, n)
  w̄   = S(undef, n)
  Mu  = S(undef, m)
  Av  = S(undef, m)
  u   = S(undef, 0)
  v   = S(undef, 0)
  err_vec = zeros(T, window)
  S = isconcretetype(S) ? S : typeof(x)
  stats = LSLQStats(0, false, false, T[], T[], T[], false, T[], T[], 0.0, "unknown")
  solver = LslqSolver{T,FC,S}(m, n, x, Nv, Aᴴu, w̄, Mu, Av, u, v, err_vec, stats)
  return solver
end

function LslqSolver(A, b; window::Integer = 5)
  m, n = size(A)
  S = ktypeof(b)
  LslqSolver(m, n, S; window)
end

"""
Type for storing the vectors required by the in-place version of LSQR.

The outer constructors

    solver = LsqrSolver(m, n, S)
    solver = LsqrSolver(A, b)
    solver = LsqrSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct LsqrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m       :: Int
  n       :: Int
  x       :: S
  Nv      :: S
  Aᴴu     :: S
  w       :: S
  Mu      :: S
  Av      :: S
  u       :: S
  v       :: S
  err_vec :: Vector{T}
  stats   :: SimpleStats{T}
end

function LsqrSolver(kc::KrylovConstructor; window::Integer = 5)
  S   = typeof(kc.vm)
  FC  = eltype(S)
  T   = real(FC)
  m   = length(kc.vm)
  n   = length(kc.vn)
  x   = similar(kc.vn)
  Nv  = similar(kc.vn)
  Aᴴu = similar(kc.vn)
  w   = similar(kc.vn)
  Mu  = similar(kc.vm)
  Av  = similar(kc.vm)
  u   = similar(kc.vm_empty)
  v   = similar(kc.vn_empty)
  err_vec = zeros(T, window)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = LsqrSolver{T,FC,S}(m, n, x, Nv, Aᴴu, w, Mu, Av, u, v, err_vec, stats)
  return solver
end

function LsqrSolver(m::Integer, n::Integer, S::Type; window::Integer = 5)
  FC  = eltype(S)
  T   = real(FC)
  x   = S(undef, n)
  Nv  = S(undef, n)
  Aᴴu = S(undef, n)
  w   = S(undef, n)
  Mu  = S(undef, m)
  Av  = S(undef, m)
  u   = S(undef, 0)
  v   = S(undef, 0)
  err_vec = zeros(T, window)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = LsqrSolver{T,FC,S}(m, n, x, Nv, Aᴴu, w, Mu, Av, u, v, err_vec, stats)
  return solver
end

function LsqrSolver(A, b; window::Integer = 5)
  m, n = size(A)
  S = ktypeof(b)
  LsqrSolver(m, n, S; window)
end

"""
Type for storing the vectors required by the in-place version of LSMR.

The outer constructors

    solver = LsmrSolver(m, n, S)
    solver = LsmrSolver(A, b)
    solver = LsmrSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct LsmrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m       :: Int
  n       :: Int
  x       :: S
  Nv      :: S
  Aᴴu     :: S
  h       :: S
  hbar    :: S
  Mu      :: S
  Av      :: S
  u       :: S
  v       :: S
  err_vec :: Vector{T}
  stats   :: LsmrStats{T}
end

function LsmrSolver(kc::KrylovConstructor; window::Integer = 5)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  x    = similar(kc.vn)
  Nv   = similar(kc.vn)
  Aᴴu  = similar(kc.vn)
  h    = similar(kc.vn)
  hbar = similar(kc.vn)
  Mu   = similar(kc.vm)
  Av   = similar(kc.vm)
  u    = similar(kc.vm_empty)
  v    = similar(kc.vn_empty)
  err_vec = zeros(T, window)
  stats = LsmrStats(0, false, false, T[], T[], zero(T), zero(T), zero(T), zero(T), zero(T), 0.0, "unknown")
  solver = LsmrSolver{T,FC,S}(m, n, x, Nv, Aᴴu, h, hbar, Mu, Av, u, v, err_vec, stats)
  return solver
end

function LsmrSolver(m::Integer, n::Integer, S::Type; window::Integer = 5)
  FC   = eltype(S)
  T    = real(FC)
  x    = S(undef, n)
  Nv   = S(undef, n)
  Aᴴu  = S(undef, n)
  h    = S(undef, n)
  hbar = S(undef, n)
  Mu   = S(undef, m)
  Av   = S(undef, m)
  u    = S(undef, 0)
  v    = S(undef, 0)
  err_vec = zeros(T, window)
  S = isconcretetype(S) ? S : typeof(x)
  stats = LsmrStats(0, false, false, T[], T[], zero(T), zero(T), zero(T), zero(T), zero(T), 0.0, "unknown")
  solver = LsmrSolver{T,FC,S}(m, n, x, Nv, Aᴴu, h, hbar, Mu, Av, u, v, err_vec, stats)
  return solver
end

function LsmrSolver(A, b; window::Integer = 5)
  m, n = size(A)
  S = ktypeof(b)
  LsmrSolver(m, n, S; window)
end

"""
Type for storing the vectors required by the in-place version of LNLQ.

The outer constructors

    solver = LnlqSolver(m, n, S)
    solver = LnlqSolver(A, b)
    solver = LnlqSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct LnlqSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m     :: Int
  n     :: Int
  x     :: S
  Nv    :: S
  Aᴴu   :: S
  y     :: S
  w̄     :: S
  Mu    :: S
  Av    :: S
  u     :: S
  v     :: S
  q     :: S
  stats :: LNLQStats{T}
end

function LnlqSolver(kc::KrylovConstructor)
  S   = typeof(kc.vm)
  FC  = eltype(S)
  T   = real(FC)
  m   = length(kc.vm)
  n   = length(kc.vn)
  x   = similar(kc.vn)
  Nv  = similar(kc.vn)
  Aᴴu = similar(kc.vn)
  y   = similar(kc.vm)
  w̄   = similar(kc.vm)
  Mu  = similar(kc.vm)
  Av  = similar(kc.vm)
  u   = similar(kc.vm_empty)
  v   = similar(kc.vn_empty)
  q   = similar(kc.vn_empty)
  stats = LNLQStats(0, false, T[], false, T[], T[], 0.0, "unknown")
  solver = LnlqSolver{T,FC,S}(m, n, x, Nv, Aᴴu, y, w̄, Mu, Av, u, v, q, stats)
  return solver
end

function LnlqSolver(m::Integer, n::Integer, S::Type)
  FC  = eltype(S)
  T   = real(FC)
  x   = S(undef, n)
  Nv  = S(undef, n)
  Aᴴu = S(undef, n)
  y   = S(undef, m)
  w̄   = S(undef, m)
  Mu  = S(undef, m)
  Av  = S(undef, m)
  u   = S(undef, 0)
  v   = S(undef, 0)
  q   = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = LNLQStats(0, false, T[], false, T[], T[], 0.0, "unknown")
  solver = LnlqSolver{T,FC,S}(m, n, x, Nv, Aᴴu, y, w̄, Mu, Av, u, v, q, stats)
  return solver
end

function LnlqSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  LnlqSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of CRAIG.

The outer constructors

    solver = CraigSolver(m, n, S)
    solver = CraigSolver(A, b)
    solver = CraigSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct CraigSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m     :: Int
  n     :: Int
  x     :: S
  Nv    :: S
  Aᴴu   :: S
  y     :: S
  w     :: S
  Mu    :: S
  Av    :: S
  u     :: S
  v     :: S
  w2    :: S
  stats :: SimpleStats{T}
end

function CraigSolver(kc::KrylovConstructor)
  S   = typeof(kc.vm)
  FC  = eltype(S)
  T   = real(FC)
  m   = length(kc.vm)
  n   = length(kc.vn)
  x   = similar(kc.vn)
  Nv  = similar(kc.vn)
  Aᴴu = similar(kc.vn)
  y   = similar(kc.vm)
  w   = similar(kc.vm)
  Mu  = similar(kc.vm)
  Av  = similar(kc.vm)
  u   = similar(kc.vm_empty)
  v   = similar(kc.vn_empty)
  w2  = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CraigSolver{T,FC,S}(m, n, x, Nv, Aᴴu, y, w, Mu, Av, u, v, w2, stats)
  return solver
end

function CraigSolver(m::Integer, n::Integer, S::Type)
  FC  = eltype(S)
  T   = real(FC)
  x   = S(undef, n)
  Nv  = S(undef, n)
  Aᴴu = S(undef, n)
  y   = S(undef, m)
  w   = S(undef, m)
  Mu  = S(undef, m)
  Av  = S(undef, m)
  u   = S(undef, 0)
  v   = S(undef, 0)
  w2  = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CraigSolver{T,FC,S}(m, n, x, Nv, Aᴴu, y, w, Mu, Av, u, v, w2, stats)
  return solver
end

function CraigSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CraigSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of CRAIGMR.

The outer constructors

    solver = CraigmrSolver(m, n, S)
    solver = CraigmrSolver(A, b)
    solver = CraigmrSolver(kc::KrylovConstructor)

may be used in order to create these vectors.
"""
mutable struct CraigmrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m     :: Int
  n     :: Int
  x     :: S
  Nv    :: S
  Aᴴu   :: S
  d     :: S
  y     :: S
  Mu    :: S
  w     :: S
  wbar  :: S
  Av    :: S
  u     :: S
  v     :: S
  q     :: S
  stats :: SimpleStats{T}
end

function CraigmrSolver(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  x    = similar(kc.vn)
  Nv   = similar(kc.vn)
  Aᴴu  = similar(kc.vn)
  d    = similar(kc.vn)
  y    = similar(kc.vm)
  Mu   = similar(kc.vm)
  w    = similar(kc.vm)
  wbar = similar(kc.vm)
  Av   = similar(kc.vm)
  u    = similar(kc.vm_empty)
  v    = similar(kc.vn_empty)
  q    = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CraigmrSolver{T,FC,S}(m, n, x, Nv, Aᴴu, d, y, Mu, w, wbar, Av, u, v, q, stats)
  return solver
end

function CraigmrSolver(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  x    = S(undef, n)
  Nv   = S(undef, n)
  Aᴴu  = S(undef, n)
  d    = S(undef, n)
  y    = S(undef, m)
  Mu   = S(undef, m)
  w    = S(undef, m)
  wbar = S(undef, m)
  Av   = S(undef, m)
  u    = S(undef, 0)
  v    = S(undef, 0)
  q    = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = CraigmrSolver{T,FC,S}(m, n, x, Nv, Aᴴu, d, y, Mu, w, wbar, Av, u, v, q, stats)
  return solver
end

function CraigmrSolver(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CraigmrSolver(m, n, S)
end

"""
Type for storing the vectors required by the in-place version of GMRES.

The outer constructors

    solver = GmresSolver(m, n, S; memory = 20)
    solver = GmresSolver(A, b; memory = 20)
    solver = GmresSolver(kc::KrylovConstructor; memory = 20)

may be used in order to create these vectors.
`memory` is set to `n` if the value given is larger than `n`.
"""
mutable struct GmresSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  w          :: S
  p          :: S
  q          :: S
  V          :: Vector{S}
  c          :: Vector{T}
  s          :: Vector{FC}
  z          :: Vector{FC}
  R          :: Vector{FC}
  warm_start :: Bool
  inner_iter :: Int
  stats      :: SimpleStats{T}
end

function GmresSolver(kc::KrylovConstructor; memory::Integer = 20)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  memory = min(m, memory)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  w  = similar(kc.vn)
  p  = similar(kc.vn_empty)
  q  = similar(kc.vn_empty)
  V  = S[similar(kc.vn) for i = 1 : memory]
  c  = Vector{T}(undef, memory)
  s  = Vector{FC}(undef, memory)
  z  = Vector{FC}(undef, memory)
  R  = Vector{FC}(undef, div(memory * (memory+1), 2))
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = GmresSolver{T,FC,S}(m, n, Δx, x, w, p, q, V, c, s, z, R, false, 0, stats)
  return solver
end

function GmresSolver(m::Integer, n::Integer, S::Type; memory::Integer = 20)
  memory = min(m, memory)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  w  = S(undef, n)
  p  = S(undef, 0)
  q  = S(undef, 0)
  V  = S[S(undef, n) for i = 1 : memory]
  c  = Vector{T}(undef, memory)
  s  = Vector{FC}(undef, memory)
  z  = Vector{FC}(undef, memory)
  R  = Vector{FC}(undef, div(memory * (memory+1), 2))
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = GmresSolver{T,FC,S}(m, n, Δx, x, w, p, q, V, c, s, z, R, false, 0, stats)
  return solver
end

function GmresSolver(A, b; memory::Integer = 20)
  m, n = size(A)
  S = ktypeof(b)
  GmresSolver(m, n, S; memory)
end

"""
Type for storing the vectors required by the in-place version of FGMRES.

The outer constructors

    solver = FgmresSolver(m, n, S; memory = 20)
    solver = FgmresSolver(A, b; memory = 20)
    solver = FgmresSolver(kc::KrylovConstructor; memory = 20)

may be used in order to create these vectors.
`memory` is set to `n` if the value given is larger than `n`.
"""
mutable struct FgmresSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  w          :: S
  q          :: S
  V          :: Vector{S}
  Z          :: Vector{S}
  c          :: Vector{T}
  s          :: Vector{FC}
  z          :: Vector{FC}
  R          :: Vector{FC}
  warm_start :: Bool
  inner_iter :: Int
  stats      :: SimpleStats{T}
end

function FgmresSolver(kc::KrylovConstructor; memory::Integer = 20)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  memory = min(m, memory)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  w  = similar(kc.vn)
  q  = similar(kc.vn_empty)
  V  = S[similar(kc.vn) for i = 1 : memory]
  Z  = S[similar(kc.vn) for i = 1 : memory]
  c  = Vector{T}(undef, memory)
  s  = Vector{FC}(undef, memory)
  z  = Vector{FC}(undef, memory)
  R  = Vector{FC}(undef, div(memory * (memory+1), 2))
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = FgmresSolver{T,FC,S}(m, n, Δx, x, w, q, V, Z, c, s, z, R, false, 0, stats)
  return solver
end

function FgmresSolver(m::Integer, n::Integer, S::Type; memory::Integer = 20)
  memory = min(m, memory)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  w  = S(undef, n)
  q  = S(undef, 0)
  V  = S[S(undef, n) for i = 1 : memory]
  Z  = S[S(undef, n) for i = 1 : memory]
  c  = Vector{T}(undef, memory)
  s  = Vector{FC}(undef, memory)
  z  = Vector{FC}(undef, memory)
  R  = Vector{FC}(undef, div(memory * (memory+1), 2))
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = FgmresSolver{T,FC,S}(m, n, Δx, x, w, q, V, Z, c, s, z, R, false, 0, stats)
  return solver
end

function FgmresSolver(A, b; memory::Integer = 20)
  m, n = size(A)
  S = ktypeof(b)
  FgmresSolver(m, n, S; memory)
end

"""
Type for storing the vectors required by the in-place version of FOM.

The outer constructors

    solver = FomSolver(m, n, S; memory = 20)
    solver = FomSolver(A, b; memory = 20)
    solver = FomSolver(kc::KrylovConstructor; memory = 20)

may be used in order to create these vectors.
`memory` is set to `n` if the value given is larger than `n`.
"""
mutable struct FomSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  w          :: S
  p          :: S
  q          :: S
  V          :: Vector{S}
  l          :: Vector{FC}
  z          :: Vector{FC}
  U          :: Vector{FC}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function FomSolver(kc::KrylovConstructor; memory::Integer = 20)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  memory = min(m, memory)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  w  = similar(kc.vn)
  p  = similar(kc.vn_empty)
  q  = similar(kc.vn_empty)
  V  = S[similar(kc.vn) for i = 1 : memory]
  l  = Vector{FC}(undef, memory)
  z  = Vector{FC}(undef, memory)
  U  = Vector{FC}(undef, div(memory * (memory+1), 2))
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = FomSolver{T,FC,S}(m, n, Δx, x, w, p, q, V, l, z, U, false, stats)
  return solver
end

function FomSolver(m::Integer, n::Integer, S::Type; memory::Integer = 20)
  memory = min(m, memory)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  w  = S(undef, n)
  p  = S(undef, 0)
  q  = S(undef, 0)
  V  = S[S(undef, n) for i = 1 : memory]
  l  = Vector{FC}(undef, memory)
  z  = Vector{FC}(undef, memory)
  U  = Vector{FC}(undef, div(memory * (memory+1), 2))
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = FomSolver{T,FC,S}(m, n, Δx, x, w, p, q, V, l, z, U, false, stats)
  return solver
end

function FomSolver(A, b; memory::Integer = 20)
  m, n = size(A)
  S = ktypeof(b)
  FomSolver(m, n, S; memory)
end

"""
Type for storing the vectors required by the in-place version of GPMR.

The outer constructors

    solver = GpmrSolver(m, n, S; memory = 20)
    solver = GpmrSolver(A, b; memory = 20)
    solver = GpmrSolver(kc::KrylovConstructor; memory = 20)

may be used in order to create these vectors.
`memory` is set to `n + m` if the value given is larger than `n + m`.
"""
mutable struct GpmrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  m          :: Int
  n          :: Int
  wA         :: S
  wB         :: S
  dA         :: S
  dB         :: S
  Δx         :: S
  Δy         :: S
  x          :: S
  y          :: S
  q          :: S
  p          :: S
  V          :: Vector{S}
  U          :: Vector{S}
  gs         :: Vector{FC}
  gc         :: Vector{T}
  zt         :: Vector{FC}
  R          :: Vector{FC}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function GpmrSolver(kc::KrylovConstructor; memory::Integer = 20)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  memory = min(n + m, memory)
  wA = similar(kc.vn_empty)
  wB = similar(kc.vm_empty)
  dA = similar(kc.vm)
  dB = similar(kc.vn)
  Δx = similar(kc.vm_empty)
  Δy = similar(kc.vn_empty)
  x  = similar(kc.vm)
  y  = similar(kc.vn)
  q  = similar(kc.vm_empty)
  p  = similar(kc.vn_empty)
  V  = S[similar(kc.vm) for i = 1 : memory]
  U  = S[similar(kc.vn) for i = 1 : memory]
  gs = Vector{FC}(undef, 4 * memory)
  gc = Vector{T}(undef, 4 * memory)
  zt = Vector{FC}(undef, 2 * memory)
  R  = Vector{FC}(undef, memory * (2 * memory + 1))
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = GpmrSolver{T,FC,S}(m, n, wA, wB, dA, dB, Δx, Δy, x, y, q, p, V, U, gs, gc, zt, R, false, stats)
  return solver
end

function GpmrSolver(m::Integer, n::Integer, S::Type; memory::Integer = 20)
  memory = min(n + m, memory)
  FC = eltype(S)
  T  = real(FC)
  wA = S(undef, 0)
  wB = S(undef, 0)
  dA = S(undef, m)
  dB = S(undef, n)
  Δx = S(undef, 0)
  Δy = S(undef, 0)
  x  = S(undef, m)
  y  = S(undef, n)
  q  = S(undef, 0)
  p  = S(undef, 0)
  V  = S[S(undef, m) for i = 1 : memory]
  U  = S[S(undef, n) for i = 1 : memory]
  gs = Vector{FC}(undef, 4 * memory)
  gc = Vector{T}(undef, 4 * memory)
  zt = Vector{FC}(undef, 2 * memory)
  R  = Vector{FC}(undef, memory * (2 * memory + 1))
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  solver = GpmrSolver{T,FC,S}(m, n, wA, wB, dA, dB, Δx, Δy, x, y, q, p, V, U, gs, gc, zt, R, false, stats)
  return solver
end

function GpmrSolver(A, b; memory::Integer = 20)
  m, n = size(A)
  S = ktypeof(b)
  GpmrSolver(m, n, S; memory)
end

"""
    solution(solver)

Return the solution(s) stored in the `solver`.
Optionally you can specify which solution you want to recover,
`solution(solver, 1)` returns `x` and `solution(solver, 2)` returns `y`.
"""
function solution end

"""
    nsolution(solver)

Return the number of outputs of `solution(solver)`.
"""
function nsolution end

"""
    statistics(solver)

Return the statistics stored in the `solver`.
"""
function statistics end

"""
    issolved(solver)

Return a boolean that determines whether the Krylov method associated to `solver` succeeded.
"""
function issolved end

"""
    niterations(solver)

Return the number of iterations performed by the Krylov method associated to `solver`.
"""
function niterations end

"""
    Aprod(solver)

Return the number of operator-vector products with `A` performed by the Krylov method associated to `solver`.
"""
function Aprod end

"""
    Atprod(solver)

Return the number of operator-vector products with `A'` performed by the Krylov method associated to `solver`.
"""
function Atprod end

"""
    results(solver)

Return a tuple containing the solution(s) and the statistics associated with the `solver`.
Allows retrieving the output arguments of an out-of-place method from the in-place method.

For example, instead of `x, stats = cg(A, b)`, you can use:
```julia
    solver = CgSolver(A, b)
    cg!(solver, A, b)
    x, stats = results(solver)
```
"""
function results end

for (KS, fun, nsol, nA, nAt, warm_start) in [
  (:CarSolver      , :car!       , 1, 1, 0, true )
  (:LsmrSolver     , :lsmr!      , 1, 1, 1, false)
  (:CgsSolver      , :cgs!       , 1, 2, 0, true )
  (:UsymlqSolver   , :usymlq!    , 1, 1, 1, true )
  (:LnlqSolver     , :lnlq!      , 2, 1, 1, false)
  (:BicgstabSolver , :bicgstab!  , 1, 2, 0, true )
  (:CrlsSolver     , :crls!      , 1, 1, 1, false)
  (:LsqrSolver     , :lsqr!      , 1, 1, 1, false)
  (:MinresSolver   , :minres!    , 1, 1, 0, true )
  (:MinaresSolver  , :minares!   , 1, 1, 0, true )
  (:CgneSolver     , :cgne!      , 1, 1, 1, false)
  (:DqgmresSolver  , :dqgmres!   , 1, 1, 0, true )
  (:SymmlqSolver   , :symmlq!    , 1, 1, 0, true )
  (:TrimrSolver    , :trimr!     , 2, 1, 1, true )
  (:UsymqrSolver   , :usymqr!    , 1, 1, 1, true )
  (:BilqrSolver    , :bilqr!     , 2, 1, 1, true )
  (:CrSolver       , :cr!        , 1, 1, 0, true )
  (:CraigmrSolver  , :craigmr!   , 2, 1, 1, false)
  (:TricgSolver    , :tricg!     , 2, 1, 1, true )
  (:CraigSolver    , :craig!     , 2, 1, 1, false)
  (:DiomSolver     , :diom!      , 1, 1, 0, true )
  (:LslqSolver     , :lslq!      , 1, 1, 1, false)
  (:TrilqrSolver   , :trilqr!    , 2, 1, 1, true )
  (:CrmrSolver     , :crmr!      , 1, 1, 1, false)
  (:CgSolver       , :cg!        , 1, 1, 0, true )
  (:CglsSolver     , :cgls!      , 1, 1, 1, false)
  (:CgLanczosSolver, :cg_lanczos!, 1, 1, 0, true )
  (:BilqSolver     , :bilq!      , 1, 1, 1, true )
  (:MinresQlpSolver, :minres_qlp!, 1, 1, 0, true )
  (:QmrSolver      , :qmr!       , 1, 1, 1, true )
  (:GmresSolver    , :gmres!     , 1, 1, 0, true )
  (:FgmresSolver   , :fgmres!    , 1, 1, 0, true )
  (:FomSolver      , :fom!       , 1, 1, 0, true )
  (:GpmrSolver     , :gpmr!      , 2, 1, 0, true )
  (:CgLanczosShiftSolver  , :cg_lanczos_shift!  , 1, 1, 0, false)
  (:CglsLanczosShiftSolver, :cgls_lanczos_shift!, 1, 1, 1, false)
]
  @eval begin
    size(solver :: $KS) = solver.m, solver.n
    statistics(solver :: $KS) = solver.stats
    niterations(solver :: $KS) = solver.stats.niter
    Aprod(solver :: $KS) = $nA * solver.stats.niter
    Atprod(solver :: $KS) = $nAt * solver.stats.niter
    if $KS == GpmrSolver
      Bprod(solver :: $KS) = solver.stats.niter
    end
    nsolution(solver :: $KS) = $nsol
    if $nsol == 1
      solution(solver :: $KS) = solver.x
      solution(solver :: $KS, p :: Integer) = (p == 1) ? solution(solver) : error("solution(solver) has only one output.")
      results(solver :: $KS) = (solver.x, solver.stats)
    end
    if $nsol == 2
      solution(solver :: $KS) = (solver.x, solver.y)
      solution(solver :: $KS, p :: Integer) = (1 ≤ p ≤ 2) ? solution(solver)[p] : error("solution(solver) has only two outputs.")
      results(solver :: $KS) = (solver.x, solver.y, solver.stats)
    end
    if $KS ∈ (BilqrSolver, TrilqrSolver)
      issolved_primal(solver :: $KS) = solver.stats.solved_primal
      issolved_dual(solver :: $KS) = solver.stats.solved_dual
      issolved(solver :: $KS) = issolved_primal(solver) && issolved_dual(solver)
    else
      issolved(solver :: $KS) = solver.stats.solved
    end
    if $warm_start
      if $KS in (BilqrSolver, TrilqrSolver, TricgSolver, TrimrSolver, GpmrSolver)
        function warm_start!(solver :: $KS, x0, y0)
          length(x0) == solver.n || error("x0 should have size $n")
          length(y0) == solver.m || error("y0 should have size $m")
          S = typeof(solver.x)
          allocate_if(true, solver, :Δx, S, solver.x)  # The length of Δx is n
          allocate_if(true, solver, :Δy, S, solver.y)  # The length of Δy is m
          kcopy!(solver.n, solver.Δx, x0)
          kcopy!(solver.m, solver.Δy, y0)
          solver.warm_start = true
          return solver
        end
      else
        function warm_start!(solver :: $KS, x0)
          S = typeof(solver.x)
          length(x0) == solver.n || error("x0 should have size $n")
          allocate_if(true, solver, :Δx, S, solver.x)  # The length of Δx is n
          kcopy!(solver.n, solver.Δx, x0)
          solver.warm_start = true
          return solver
        end
      end
    end
  end
end
