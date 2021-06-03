# An implementation of USYMLQR for the solution of symmetric saddle-point systems.
#
# This method is described in
#
# M. A. Saunders, H. D. Simon, and E. L. Yip
# Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations.
# SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
#
# A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin
# A tridiagonalization method for symmetric saddle-point systems.
# SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montréal, June 2021.

export usymlqr, usymlqr!

"""
    (x, y, stats) = usymlqr(A, b::AbstractVector{T}, c::AbstractVector{T};
                            M=I, N=I, atol::T=√eps(T), rtol::T=√eps(T),
                            itmax::Int=0, verbose::Int=0, history::Bool=false) where T <: AbstractFloat

Solve the symmetric saddle-point system

    [ E   A ] [ x ] = [ b ]
    [ Aᵀ    ] [ y ]   [ c ]

where E = M⁻¹ by way of the Saunders-Simon-Yip tridiagonalization using USYMLQ and USYMQR methods.
The method solves the least-squares problem

    [ E   A ] [ s ] = [ b ]
    [ Aᵀ    ] [ t ]   [ 0 ]

and the least-norm problem

    [ E   A ] [ w ] = [ 0 ]
    [ Aᵀ    ] [ z ]   [ c ]

and simply adds the solutions.

    [ M   O ]
    [ 0   N ]

indicates the weighted norm in which residuals are measured.
It's the Euclidean norm when `M` and `N` are identity operators.

#### References

* M. A. Saunders, H. D. Simon, and E. L. Yip, *Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations*, SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
* A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin, *A tridiagonalization method for symmetric saddle-point and quasi-definite systems*, SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019.
* A. Montoison and D. Orban, *TriCG and TriMR: Two Iterative Methods for Symmetric Quasi-Definite Systems*, Cahier du GERAD G-2020-41, GERAD, Montréal, 2020.
"""
function usymlqr(A, b :: AbstractVector{T}, c :: AbstractVector{T}; kwargs...) where T <: AbstractFloat
  solver = UsymlqrSolver(A, b)
  usymlqr!(solver, A, b, c; kwargs...)
end

function usymlqr!(solver :: UsymlqrSolver{T,S}, A, b :: AbstractVector{T}, c :: AbstractVector{T};
                  M=I, N=I, atol :: T=√eps(T), rtol :: T=√eps(T),
                  itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, S <: DenseVector{T}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  length(c) == n || error("Inconsistent problem size")
  (verbose > 0) && @printf("USYMLQR: system of %d equations in %d variables\n", m+n, m+n)
  
  # Check M == Iₘ and N == Iₙ
  MisI = (M == I)
  NisI = (N == I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  ktypeof(c) == S || error("ktypeof(c) ≠ $S")
  MisI || (eltype(M) == T) || error("eltype(M) ≠ $T")
  NisI || (eltype(N) == T) || error("eltype(N) ≠ $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  allocate_if(!MisI, solver, :vₖ, S, m)
  allocate_if(!NisI, solver, :uₖ, S, n)
  xₖ, yₖ, M⁻¹vₖ₋₁, M⁻¹vₖ, N⁻¹uₖ₋₁, N⁻¹uₖ, vₖ, uₖ = solver.xₖ, solver.yₖ, solver.M⁻¹vₖ₋₁, solver.M⁻¹vₖ, solver.N⁻¹uₖ₋₁, solver.N⁻¹uₖ, solver.vₖ, solver.uₖ
  vₖ = MisI ? M⁻¹vₖ : solver.vₖ
  uₖ = NisI ? N⁻¹uₖ : solver.uₖ

  iter = 0
  itmax == 0 && (itmax = n+m)

  # Initial solutions x₀ and y₀.
  xₖ .= zero(T)
  yₖ .= zero(T)

  # Initialize preconditioned orthogonal tridiagonalization process.
  M⁻¹vₖ₋₁ .= zero(T)  # v₀ = 0
  N⁻¹uₖ₋₁ .= zero(T)  # u₀ = 0

  # β₁Ev₁ = b ↔ β₁v₁ = Mb
  M⁻¹vₖ .= b
  MisI || mul!(vₖ, M, M⁻¹vₖ)
  βₖ = sqrt(@kdot(m, vₖ, M⁻¹vₖ))  # β₁ = ‖v₁‖_E
  if βₖ ≠ 0
    @kscal!(m, 1 / βₖ, M⁻¹vₖ)
    MisI || @kscal!(m, 1 / βₖ, vₖ)
  end

  # γ₁Fu₁ = c ↔ γ₁u₁ = Nb
  N⁻¹uₖ .= c
  NisI || mul!(uₖ, N, N⁻¹uₖ)
  γₖ = sqrt(@kdot(n, uₖ, N⁻¹uₖ))  # γ₁ = ‖u₁‖_F
  if γₖ ≠ 0
    @kscal!(n, 1 / γₖ, N⁻¹uₖ)
    NisI || @kscal!(n, 1 / γₖ, uₖ)
  end

  (verbose > 0) && @printf("%4s %7s %7s %7s\n", "k", "αₖ", "βₖ", "γₖ")
  display(iter, verbose) && @printf("%4d %7.1e %7.1e %7.1e\n", iter, αₖ, βₖ, γₖ)

  # initialize x and z update directions
  x = kzeros(S, n)
  xNorm = zero(T)
  z = kzeros(S, n)
  wbar = vₖ / δbar
  w = kzeros(S, n)
  wold = kzeros(S, n)
  Wnorm2 = zero(T)

  # quantities related to the update of y
  etabar = γₖ / δbar
  p = kzeros(S, m)
  pbar = copy(uₖ)
  y = kzeros(S, m)
  yC = etabar  * pbar
  zC = -etabar * wbar

  yNorm2 = zero(T)
  yNorm = zero(T)

  # quantities related to the computation of ‖x‖
  # TODO

  # Residual of the least-norm problem
  rNorm_lq = 2 * ln_tol  # just so it exists at the end of the loop!
  rNorms_lq = T[]

  status = "unknown"
  transition_to_cg = false

  # stopping conditions that apply to both problems
  tired = iter ≥ itmax
  ill_cond_lim = one(T) / Acond ≤ ctol
  ill_cond_mach = one(T) + one(T) / Acond ≤ one(T)
  ill_cond = ill_cond_mach || ill_cond_lim

  # stopping conditions related to the least-squares problem
  test_LS = rNorm_qr / (one(T) + Anorm * xNorm)
  zero_resid_lim_LS = test_LS ≤ ls_zero_resid_tol
  zero_resid_mach_LS = one(T) + test_LS ≤ one(T)
  zero_resid_LS = zero_resid_mach_LS || zero_resid_lim_LS
  test_LS = ArNorm_qr / (Anorm * max(one(T), rNorm_qr))
  solved_lim_LS = test_LS ≤ ls_optimality_tol
  solved_mach_LS = one(T) + test_LS ≤ one(T)

  # TODO: check this
  solved_LS = false  # solved_mach_LS | solved_lim_LS | zero_resid_LS

  # Stopping conditions related to the least-norm problem
  test_LN = rNorm_lq  / sqrt(cnorm^2 + Anorm2 * yNorm2)
  solved_lim_LN = test_LN ≤ ln_tol
  solved_mach_LN = one(T) + test_LN ≤ one(T) # Problem no?
  
  # TODO: check this
  solved_LN = false # solved_lim_LN | solved_mach_LN

  solved = solved_LS && solved_LN

  # TODO: remove this when finished
  tests_LS = T[]
  tests_LN = T[]

  while ! (solved || tired || ill_cond)

    iter = iter + 1

    # continue tridiagonalization
    @. u_prev = uₖ
    @kaxpby!(m, one(T), q, -αₖ, uₖ)
    Atuprev = A.tprod(u_prev)
    @kaxpby!(n, -βₖ, v_prev, -αₖ, vₖ)
    @kaxpy!(n, one(T), Atuprev, vₖ)
    βₖ = @knrm2(m, uₖ)
    if βₖ > 0
      @. uₖ /= βₖ
    end
    γₖ = @knrm2(n, vₖ)
    if γₖ > 0
      @. vₖ /= γₖ
    end

    # save vectors for next iteration
    @. v_prev = vv
    @. vv = vₖ

    # Continue the QR factorization of Tₖ₊₁.ₖ = Qₖ₊₁ [ Rₖ ].
    #                                                [ Oᵀ ]

    ƛ = -cs * γₖ
    ϵ =  sn * γₖ

    # compute optimality residual of least-squares problem at x{k-1}
    # TODO: use recurrence formula for QR residual
    if !solved_LS
      ArNorm_qr_computed = rNorm_qr * sqrt(δbar^2 + ƛ^2)
      ArNorm_qr = norm(A' * (b - A * x))  # FIXME
      @debug "" ArNorm_qr_computed ArNorm_qr abs(ArNorm_qr_computed - ArNorm_qr) / ArNorm_qr
      ArNorm_qr = ArNorm_qr_computed
      push!(ArNorms_qr, ArNorm_qr)

      test_LS = ArNorm_qr / (Anorm * max(one(T), rNorm_qr))
      solved_lim_LS = test_LS ≤ ls_optimality_tol
      solved_mach_LS = one(T) + test_LS ≤ one(T)
      solved_LS = solved_mach_LS | solved_lim_LS

      # TODO: remove this when finished
      push!(tests_LS, test_LS)
    end
    display(iter, verbose) && @printf("%7.1e ", ArNorm_qr)

    # continue QR factorization
    delta = sqrt(δbar^2 + βₖ^2)
    csold = cs
    snold = sn
    cs = δbar/ delta
    sn = βₖ / delta

    # update w (used to update x and z)
    @. wold = w
    @. w = cs * wbar

    if !solved_LS
      # the optimality conditions of the LS problem were not triggerred
      # update x and see if we have a zero residual

      ϕ = cs * ϕbar
      ϕbar = sn * ϕbar
      @kaxpy!(n, ϕ, w, x)
      xNorm = norm(x)  # FIXME

      # update least-squares residual
      rNorm_qr = abs(ϕbar)
      push!(rNorms_qr, rNorm_qr)

      # stopping conditions related to the least-squares problem
      test_LS = rNorm_qr / (one(T) + Anorm * xNorm)
      zero_resid_lim_LS = test_LS ≤ ls_zero_resid_tol
      zero_resid_mach_LS = one(T) + test_LS ≤ one(T)
      zero_resid_LS = zero_resid_mach_LS | zero_resid_lim_LS
      solved_LS |= zero_resid_LS

    end

    # continue tridiagonalization
    q = A * vₖ
    @. q -= γₖ * u_prev
    αₖ = @kdot(m, uₖ, q)

    # Update norm estimates
    Anorm2 += αₖ * αₖ + βₖ * βₖ + γₖ * γₖ
    Anorm = √Anorm2

    # Estimate κ₂(A) based on the diagonal of L.
    sigma_min = min(delta, sigma_min)
    sigma_max = max(delta, sigma_max)
    Acond = sigma_max / sigma_min

    # continue QR factorization of T{k+1,k}
    λ = cs * ƛ + sn * αₖ
    δbar= sn * ƛ - cs * αₖ

    if !solved_LN

      etaold = η
      η = cs * etabar # = etak

      # compute residual of least-norm problem at y{k-1}
      # TODO: use recurrence formula for LQ residual
      rNorm_lq_computed = sqrt((delta * η)^2 + (ϵ * etaold)^2)
      rNorm_lq = norm(A' * y - c)  # FIXME
      rNorm_lq = rNorm_lq_computed
      push!(rNorms_lq, rNorm_lq)

      # stopping conditions related to the least-norm problem
      test_LN = rNorm_lq / sqrt(cnorm^2 + Anorm2 * yNorm2)
      solved_lim_LN = test_LN ≤ ln_tol
      solved_mach_LN = one(T) + test_LN ≤ one(T)
      solved_LN = solved_lim_LN || solved_mach_LN

      # TODO: remove this when finished
      push!(tests_LN, test_LN)

      @. wbar = (vₖ - λ * w - ϵ * wold) / δbar

      if !solved_LN

          # prepare to update y and z
          @. p = cs * pbar + sn * uₖ

          # update y and z
          @. y += η * p
          @. z -= η * w
          yNorm2 += η * η
          yNorm = sqrt(yNorm2)

          @. pbar = sn * pbar - cs * uₖ
          etabarold = etabar
          etabar = -(λ * η + ϵ * etaold) / δbar # = etabar{k+1}

          # see if CG iterate has smaller residual
          # TODO: use recurrence formula for CG residual
          @. yC = y + etabar * pbar
          @. zC = z - etabar * wbar
          yCNorm2 = yNorm2 + etabar* etabar
          rNorm_cg_computed = γₖ * abs(snold * etaold - csold * etabarold)
          rNorm_cg = norm(A' * yC - c)

          # if rNorm_cg < rNorm_lq
          #   # stopping conditions related to the least-norm problem
          # test_cg = rNorm_cg / sqrt(γ₁^2 + Anorm2 * yCNorm2)
          #   solved_lim_LN = test_cg ≤ ln_tol
          #   solved_mach_LN = 1.0 + test_cg ≤ 1.0
          #   solved_LN = solved_lim_LN | solved_mach_LN
          #   # transition_to_cg = solved_LN
          #   transition_to_cg = false
          # end

          if transition_to_cg
            # @. yC = y + etabar* pbar
            # @. zC = z - etabar* wbar
          end
      end
    end
    display(iter, verbose) && @printf("%7.1e\n", rNorm_lq)

    display(iter, verbose) && @printf("%4d %8.1e %7.1e %7.1e %7.1e %7.1e %7.1e ", iter, αₖ, βₖ, γₖ, Anorm, Acond, rNorm_qr)

    # Stopping conditions that apply to both problems
    ill_cond_lim = one(T) / Acond ≤ ctol
    ill_cond_mach = one(T) + one(T) / Acond ≤ one(T)
    ill_cond = ill_cond_mach || ill_cond_lim

    tired = iter ≥ itmax
    solved = solved_LS && solved_LN
  end
  (verbose > 0) && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, false, T[], T[], status)
  return (x, y, stats)
end
