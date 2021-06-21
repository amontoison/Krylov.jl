export cg

"""
    (x, stats) = cg(A, b; M, atol, rtol, itmax, radius, linesearch, verbose)
"""
function cg(A, b :: AbstractVector{T};
            M=I, atol :: T=√eps(T), rtol :: T=√eps(T),
            itmax :: Int=0, radius :: T=zero(T), linesearch :: Bool=false,
            verbose :: Bool=false) where T <: AbstractFloat

  linesearch && (radius > 0) && error("`linesearch` set to `true` but trust-region radius > 0")

  n = size(b, 1)
  (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size")
  verbose && @printf("CG: system of %d equations in %d variables\n", n, n)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  (M == I) || (eltype(M) == T) || error("eltype(M) ≠ $T")

  # Determine the storage type of b
  S = typeof(b)

  # Initial state.
  x = kzeros(S, n)
  r = copy(b)
  z = M * r
  p = copy(z)
  γ = @kdot(n, r, z)
  γ == 0 && return x, SimpleStats(true, false, [zero(T)], T[], "x = 0 is a zero-residual solution")

  iter = 0
  itmax == 0 && (itmax = 2 * n)

  pAp = zero(T)
  rNorm = sqrt(γ)
  pNorm² = γ
  rNorms = [rNorm;]
  ε = atol + rtol * rNorm
  verbose && @printf("%5s  %7s  %8s  %8s  %8s\n", "k", "‖r‖", "pAp", "α", "σ")
  verbose && @printf("%5d  %7.1e  ", iter, rNorm)

  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  inconsistent = false
  on_boundary = false
  zero_curvature = false

  status = "unknown"

  while !(solved || tired || zero_curvature)
    Ap = A * p
    pAp = @kdot(n, p, Ap)
    if (pAp ≤ eps(T) * pNorm²) && (radius == 0)
      if abs(pAp) ≤ eps(T) * pNorm²
        zero_curvature = true
        inconsistent = !linesearch
      end
      if linesearch
        iter == 0 && (x .= b)
        solved = true
      end
    end
    (zero_curvature || solved) && continue

    α = γ / pAp

    # Compute step size to boundary if applicable.
    σ = radius > 0 ? maximum(to_boundary(x, p, radius, dNorm2=pNorm²)) : α

    verbose && @printf("%8.1e  %8.1e  %8.1e\n", pAp, α, σ)

    # Move along p from x to the boundary if either
    # the next step leads outside the trust region or
    # we have nonpositive curvature.
    if (radius > 0) && ((pAp ≤ 0) || (α > σ))
      α = σ
      on_boundary = true
    end

    @kaxpy!(n,  α,  p, x)
    @kaxpy!(n, -α, Ap, r)
    z = M * r
    γ_next = @kdot(n, r, z)
    rNorm = sqrt(γ_next)
    push!(rNorms, rNorm)

    solved = (rNorm ≤ ε) || on_boundary

    if !solved
      β = γ_next / γ
      pNorm² = γ_next + β^2 * pNorm²
      γ = γ_next
      @kaxpby!(n, one(T), z, β, p)
    end

    iter = iter + 1
    tired = iter ≥ itmax
    verbose && @printf("%5d  %7.1e  ", iter, rNorm)
  end
  verbose && @printf("\n")

  solved && on_boundary && (status = "on trust-region boundary")
  solved && linesearch && (pAp ≤ 0) && (status = "nonpositive curvature detected")
  solved && (status == "unknown") && (status = "solution good enough given atol and rtol")
  zero_curvature && (status = "zero curvature detected")
  tired && (status = "maximum number of iterations exceeded")
  stats = SimpleStats(solved, inconsistent, rNorms, T[], status)
  return (x, stats)
end
