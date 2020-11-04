export cr

"""
    (x, stats) = cr(A, b; M, atol, rtol, itmax, verbose)
"""
function cr(A, b :: AbstractVector{T};
            M=opEye(), atol :: T=√eps(T), rtol :: T=√eps(T), itmax :: Int=0, verbose :: Bool=false) where T <: AbstractFloat

  n = size(b, 1)
  (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size")
  verbose && @printf("CR: system of %d equations in %d variables\n", n, n)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  isa(M, opEye) || (eltype(M) == T) || error("eltype(M) ≠ $T")

  # Determine the storage type of b
  S = typeof(b)

  # Workspace
  x = kzeros(S, n)  # x₀
  r = copy(M * b)   # r₀
  p = copy(r)       # p₀
  Ap = copy(A * p)  # Ap₀
  γ = zero(T)
  Ar = A * r

  rNorm = norm(r)
  rNorm == 0 && return (x, SimpleStats(true, false, [rNorm], T[], "x = 0 is a zero-residual solution"))

  iter = 0
  itmax == 0 && (itmax = 2 * n)

  rNorms = [rNorm;]
  ε = atol + rtol * rNorm
  verbose && @printf("%5d  %7.1e\n", iter, rNorm)

  solved = false
  tired = false

  while !(solved || tired)

    M⁻¹Ap = M * Ap
    γ = dot(r, Ar)
    α = γ / dot(Ap, M⁻¹Ap)
    @. x = x + α * p
    @. r = r - α * M⁻¹Ap
    γ_new = dot(r, Ar)
    β = γ_new / γ
    γ = γ_new
    @. p = r + β * p
    Ar = A * r
    @. Ap = Ar + β * Ap

    rNorm = norm(r)
    push!(rNorms, rNorm)
    iter = iter + 1

    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    verbose && @printf("%5d  %7.1e\n", iter, rNorm)
  end
  verbose && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, false, rNorms, T[], status)
  return (x, stats)
end
