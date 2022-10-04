# An implementation of DIOM for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad, Practical use of some krylov subspace methods for solving indefinite and nonsymmetric linear systems.
# SIAM journal on scientific and statistical computing, 5(1), pp. 203--228, 1984.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, September 2018.

export diom, diom!

"""
    (x, stats) = diom(A, b::AbstractVector{FC}; memory::Int=20,
                      M=I, N=I, atol::T=√eps(T), rtol::T=√eps(T),
                      reorthogonalization::Bool=false, itmax::Int=0,
                      verbose::Int=0, history::Bool=false,
                      ldiv::Bool=false, callback=solver->false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the consistent linear system Ax = b of size n using DIOM.

DIOM only orthogonalizes the new vectors of the Krylov basis against the `memory` most recent vectors.
If CG is well defined on `Ax = b` and `memory = 2`, DIOM is theoretically equivalent to CG.
If `k ≤ memory` where `k` is the number of iterations, DIOM is theoretically equivalent to FOM.
Otherwise, DIOM interpolates between CG and FOM and is similar to CG with partial reorthogonalization.

Partial reorthogonalization is available with the `reorthogonalization` option.

An advantage of DIOM is that nonsymmetric or symmetric indefinite or both nonsymmetric
and indefinite systems of linear equations can be handled by this single algorithm.

This implementation allows a left preconditioner M and a right preconditioner N.

DIOM can be warm-started from an initial guess `x0` with

    (x, stats) = diom(A, b, x0; kwargs...)

where `kwargs` are the same keyword arguments as above.

The callback is called as `callback(solver)` and should return `true` if the main loop should terminate,
and `false` otherwise.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension n;
* `b`: a vector of length n.

#### Output arguments

* `x`: a dense vector of length n;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### Reference

* Y. Saad, [*Practical use of some krylov subspace methods for solving indefinite and nonsymmetric linear systems*](https://doi.org/10.1137/0905015), SIAM journal on scientific and statistical computing, 5(1), pp. 203--228, 1984.
"""
function diom end

function diom(A, b :: AbstractVector{FC}, x0 :: AbstractVector; memory :: Int=20, kwargs...) where FC <: FloatOrComplex
  solver = DiomSolver(A, b, memory)
  diom!(solver, A, b, x0; kwargs...)
  return (solver.x, solver.stats)
end

function diom(A, b :: AbstractVector{FC}; memory :: Int=20, kwargs...) where FC <: FloatOrComplex
  solver = DiomSolver(A, b, memory)
  diom!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = diom!(solver::DiomSolver, A, b; kwargs...)
    solver = diom!(solver::DiomSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`diom`](@ref).

Note that the `memory` keyword argument is the only exception.
It's required to create a `DiomSolver` and can't be changed later.

See [`DiomSolver`](@ref) for more details about the `solver`.
"""
function diom! end

function diom!(solver :: DiomSolver{T,FC,S}, A, b :: AbstractVector{FC}, x0 :: AbstractVector; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}
  warm_start!(solver, x0)
  diom!(solver, A, b; kwargs...)
  return solver
end

function diom!(solver :: DiomSolver{T,FC,S}, A, b :: AbstractVector{FC};
               M=I, N=I, atol :: T=√eps(T), rtol :: T=√eps(T),
               reorthogonalization :: Bool=false, itmax :: Int=0,
               verbose :: Int=0, history :: Bool=false,
               ldiv :: Bool=false, callback = solver -> false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("DIOM: system of size %d\n", n)

  # Check M = Iₙ and N = Iₙ
  MisI = (M === I)
  NisI = (N === I)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")

  # Set up workspace.
  allocate_if(!MisI, solver, :w, S, n)
  allocate_if(!NisI, solver, :z, S, n)
  Δx, x, t, P, V = solver.Δx, solver.x, solver.t, solver.P, solver.V
  L, H, stats = solver.L, solver.H, solver.stats
  warm_start = solver.warm_start
  rNorms = stats.residuals
  reset!(stats)
  w  = MisI ? t : solver.w
  r₀ = MisI ? t : solver.w

  # Initial solution x₀ and residual r₀.
  x .= zero(FC)  # x₀
  if warm_start
    mul!(t, A, Δx)
    @kaxpby!(n, one(FC), b, -one(FC), t)
  else
    t .= b
  end
  MisI || mulorldiv!(r₀, M, t, ldiv)  # M(b - Ax₀)
  rNorm = @knrm2(n, r₀)               # β = ‖r₀‖₂
  history && push!(rNorms, rNorm)
  if rNorm == 0
    stats.niter = 0
    stats.solved, stats.inconsistent = true, false
    stats.status = "x = 0 is a zero-residual solution"
    solver.warm_start = false
    return solver
  end

  iter = 0
  itmax == 0 && (itmax = 2*n)

  ε = atol + rtol * rNorm
  (verbose > 0) && @printf("%5s  %7s\n", "k", "‖rₖ‖")
  kdisplay(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)

  mem = length(V)  # Memory
  for i = 1 : mem
    V[i] .= zero(FC)  # Orthogonal basis of Kₖ(MAN, Mr₀).
  end
  for i = 1 : mem-1
    P[i] .= zero(FC)  # Directions Pₖ = NVₖ(Uₖ)⁻¹.
  end
  H .= zero(FC)  # Last column of the band hessenberg matrix Hₖ = LₖUₖ.
  # Each column has at most mem + 1 nonzero elements.
  # hᵢ.ₖ is stored as H[k-i+1], i ≤ k. hₖ₊₁.ₖ is not stored in H.
  # k-i+1 represents the indice of the diagonal where hᵢ.ₖ is located.
  # In addition of that, the last column of Uₖ is stored in H.
  L .= zero(FC)  # Last mem-1 pivots of Lₖ.

  # Initial ξ₁ and V₁.
  ξ = rNorm
  V[1] .= r₀ ./ rNorm

  # Stopping criterion.
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"
  user_requested_exit = false

  while !(solved || tired || user_requested_exit)

    # Update iteration index.
    iter = iter + 1

    # Set position in circulars stacks.
    pos = mod(iter-1, mem) + 1     # Position corresponding to vₖ in the circular stack V.
    next_pos = mod(iter, mem) + 1  # Position corresponding to vₖ₊₁ in the circular stack V.

    # Incomplete Arnoldi procedure.
    z = NisI ? V[pos] : solver.z
    NisI || mulorldiv!(z, N, V[pos], ldiv)  # Nvₖ, forms pₖ
    mul!(t, A, z)                           # ANvₖ
    MisI || mulorldiv!(w, M, t, ldiv)       # MANvₖ, forms vₖ₊₁
    for i = max(1, iter-mem+1) : iter
      ipos = mod(i-1, mem) + 1  # Position corresponding to vᵢ in the circular stack V.
      diag = iter - i + 1
      H[diag] = @kdot(n, w, V[ipos])    # hᵢ.ₖ = ⟨MANvₖ, vᵢ⟩
      @kaxpy!(n, -H[diag], V[ipos], w)  # w ← w - hᵢ.ₖvᵢ
    end

    # Partial reorthogonalization of the Krylov basis.
    if reorthogonalization
      for i = max(1, iter-mem+1) : iter
        ipos = mod(i-1, mem) + 1
        diag = iter - i + 1
        Htmp = @kdot(n, w, V[ipos])
        H[diag] += Htmp
        @kaxpy!(n, -Htmp, V[ipos], w)
      end
    end

    # Compute hₖ₊₁.ₖ and vₖ₊₁.
    Haux = @knrm2(n, w)         # hₖ₊₁.ₖ = ‖vₖ₊₁‖₂
    if Haux ≠ 0                 # hₖ₊₁.ₖ = 0 ⇒ "lucky breakdown"
      V[next_pos] .= w ./ Haux  # vₖ₊₁ = w / hₖ₊₁.ₖ
    end

    # Update the LU factorization of Hₖ.
    # Compute the last column of Uₖ.
    if iter ≥ 2
      # u₁.ₖ ← h₁.ₖ             if iter ≤ mem
      # uₖ₋ₘₑₘ₊₁.ₖ ← hₖ₋ₘₑₘ₊₁.ₖ if iter ≥ mem + 1
      for i = max(2,iter-mem+2) : iter
        lpos = mod(i-1, mem-1) + 1  # Position corresponding to lᵢ.ᵢ₋₁ in the circular stack L.
        diag = iter - i + 1
        next_diag = diag + 1
        # uᵢ.ₖ ← hᵢ.ₖ - lᵢ.ᵢ₋₁ * uᵢ₋₁.ₖ
        H[diag] = H[diag] - L[lpos] * H[next_diag]
        if i == iter
          # Compute ξₖ the last component of zₖ = β(Lₖ)⁻¹e₁.
          # ξₖ = -lₖ.ₖ₋₁ * ξₖ₋₁
          ξ = - L[lpos] * ξ
        end
      end
    end
    # Compute next pivot lₖ₊₁.ₖ = hₖ₊₁.ₖ / uₖ.ₖ
    next_lpos = mod(iter, mem-1) + 1
    L[next_lpos] = Haux / H[1]

    ppos = mod(iter-1, mem-1) + 1 # Position corresponding to pₖ in the circular stack P.

    # Compute the direction pₖ, the last column of Pₖ = NVₖ(Uₖ)⁻¹.
    # u₁.ₖp₁ + ... + uₖ.ₖpₖ = Nvₖ             if k ≤ mem
    # uₖ₋ₘₑₘ₊₁.ₖpₖ₋ₘₑₘ₊₁ + ... + uₖ.ₖpₖ = Nvₖ if k ≥ mem + 1
    for i = max(1,iter-mem+1) : iter-1
      ipos = mod(i-1, mem-1) + 1  # Position corresponding to pᵢ in the circular stack P.
      diag = iter - i + 1
      if ipos == ppos
        # pₖ ← -uₖ₋ₘₑₘ₊₁.ₖ * pₖ₋ₘₑₘ₊₁
        @kscal!(n, -H[diag], P[ppos])
      else
        # pₖ ← pₖ - uᵢ.ₖ * pᵢ
        @kaxpy!(n, -H[diag], P[ipos], P[ppos])
      end
    end
    # pₐᵤₓ ← pₐᵤₓ + Nvₖ
    @kaxpy!(n, one(FC), z, P[ppos])
    # pₖ = pₐᵤₓ / uₖ.ₖ
    P[ppos] .= P[ppos] ./ H[1]

    # Update solution xₖ.
    # xₖ = xₖ₋₁ + ξₖ * pₖ
    @kaxpy!(n, ξ, P[ppos], x)

    # Compute residual norm.
    # ‖ M(b - Axₖ) ‖₂ = hₖ₊₁.ₖ * |ξₖ / uₖ.ₖ|
    rNorm = Haux * abs(ξ / H[1])
    history && push!(rNorms, rNorm)

    # Stopping conditions that do not depend on user input.
    # This is to guard against tolerances that are unreasonably small.
    resid_decrease_mach = (rNorm + one(T) ≤ one(T))

    # Update stopping criterion.
    user_requested_exit = callback(solver) :: Bool
    resid_decrease_lim = rNorm ≤ ε
    solved = resid_decrease_lim || resid_decrease_mach
    tired = iter ≥ itmax
    kdisplay(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)
  end
  (verbose > 0) && @printf("\n")
  tired               && (status = "maximum number of iterations exceeded")
  solved              && (status = "solution good enough given atol and rtol")
  user_requested_exit && (status = "user-requested exit")

  # Update x
  warm_start && @kaxpy!(n, one(FC), Δx, x)
  solver.warm_start = false

  # Update stats
  stats.niter = iter
  stats.solved = solved
  stats.inconsistent = false
  stats.status = status
  return solver
end
