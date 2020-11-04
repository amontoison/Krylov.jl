export minres

"""
    (x, stats) = minres(A, b; M, atol, rtol, itmax, verbose)
"""
function minres(A, b :: AbstractVector{T};
                M=opEye(), atol :: T=√eps(T), rtol :: T=√eps(T),
                itmax :: Int=0, verbose :: Bool=false) where T <: AbstractFloat

  n, m = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  verbose && @printf("MINRES: system of size %d\n", n)

 # Tests M == Iₙ
  MisI = isa(M, opEye)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")

  # Determine the storage type of b
  S = typeof(b)

  # Initial solution x₀.
  x = kzeros(S, n)

  # β₁v₁ = Mb
  M⁻¹vₖ = copy(b)
  vₖ = M * M⁻¹vₖ
  βₖ = sqrt(@kdot(n, vₖ, M⁻¹vₖ))
  if βₖ ≠ 0
    @kscal!(n, 1 / βₖ, M⁻¹vₖ)
    MisI || @kscal!(n, 1 / βₖ, vₖ)
  end

  rNorm = βₖ
  rNorm == 0 && return x, SimpleStats(true, false, [rNorm], T[], "x = 0 is a zero-residual solution")

  iter = 0
  itmax == 0 && (itmax = 2*n)

  rNorms = [rNorm;]
  ε = atol + rtol * rNorm
  ArNorms = T[]
  κ = zero(T)
  verbose && @printf("%5s  %7s  %7s\n", "k", "‖rₖ‖", "‖Arₖ₋₁‖")
  verbose && @printf("%5d  %7.1e  %7s\n", iter, rNorm, "✗ ✗ ✗ ✗")

  # Set up workspace.
  M⁻¹vₖ₋₁ = kzeros(S, n)
  cₖ₋₂ = cₖ₋₁ = cₖ = zero(T)  # Givens cosines used for the QR factorization of Tₖ₊₁.ₖ
  sₖ₋₂ = sₖ₋₁ = sₖ = zero(T)  # Givens sines used for the QR factorization of Tₖ₊₁.ₖ
  wₖ₋₂ = kzeros(S, n)         # Column k-2 of Wₖ = Vₖ(Rₖ)⁻¹
  wₖ₋₁ = kzeros(S, n)         # Column k-1 of Wₖ = Vₖ(Rₖ)⁻¹
  ζbarₖ = βₖ                  # ζbarₖ is the last component of z̅ₖ = (Qₖ)ᵀβ₁e₁

  # Use M⁻¹vₖ₋₁ to store vₖ when a preconditioner is provided
  MisI ? (vₐᵤₓ = vₖ) : (vₐᵤₓ = M⁻¹vₖ₋₁)

  # Stopping criterion.
  solved = rNorm ≤ ε
  inconsistent = false
  tired = iter ≥ itmax
  status = "unknown"

  while !(solved || tired || inconsistent)
    # Update iteration index.
    iter = iter + 1

    # Continue the preconditioned Lanczos process.
    # M(A - λI)Vₖ = Vₖ₊₁Tₖ₊₁.ₖ
    # βₖ₊₁vₖ₊₁ = M(A - λI)vₖ - αₖvₖ - βₖvₖ₋₁

    p = A * vₖ               # p ← Avₖ

    if iter ≥ 2
      @kaxpy!(n, -βₖ, M⁻¹vₖ₋₁, p) # p ← p - βₖ * M⁻¹vₖ₋₁
    end

    αₖ = @kdot(n, vₖ, p)       # αₖ = pᵀvₖ

    @kaxpy!(n, -αₖ, M⁻¹vₖ, p)  # p ← p - αₖM⁻¹vₖ

    MisI || (vₐᵤₓ .= vₖ)  # Tempory storage for vₖ
    vₖ₊₁ = M * p          # βₖ₊₁vₖ₊₁ = MAvₖ - γₖvₖ₋₁ - αₖvₖ

    βₖ₊₁ = sqrt(@kdot(m, vₖ₊₁, p))

    if βₖ₊₁ ≠ 0
      @kscal!(m, one(T) / βₖ₊₁, vₖ₊₁)
      MisI || @kscal!(m, one(T) / βₖ₊₁, p)
    end

    # Update the QR factorization of Tₖ₊₁.ₖ = Qₖ [ Rₖ ].
    #                                            [ Oᵀ ]
    #
    # [ α₁ β₂ 0  •  •  •   0  ]      [ λ₁ γ₁ ϵ₁ 0  •  •  0  ]
    # [ β₂ α₂ β₃ •         •  ]      [ 0  λ₂ γ₂ •  •     •  ]
    # [ 0  •  •  •  •      •  ]      [ •  •  λ₃ •  •  •  •  ]
    # [ •  •  •  •  •  •   •  ] = Qₖ [ •     •  •  •  •  0  ]
    # [ •     •  •  •  •   0  ]      [ •        •  •  • ϵₖ₋₂]
    # [ •        •  •  •   βₖ ]      [ •           •  • γₖ₋₁]
    # [ •           •  βₖ  αₖ ]      [ 0  •  •  •  •  0  λₖ ]
    # [ 0  •  •  •  •  0  βₖ₊₁]      [ 0  •  •  •  •  •  0  ]
    #
    # If k = 1, we don't have any previous reflexion.
    # If k = 2, we apply the last reflexion.
    # If k ≥ 3, we only apply the two previous reflexions.

    # Apply previous Givens reflections Qₖ₋₂.ₖ₋₁
    if iter ≥ 3
      # [cₖ₋₂  sₖ₋₂] [0 ] = [  ϵₖ₋₂ ]
      # [sₖ₋₂ -cₖ₋₂] [βₖ]   [γbarₖ₋₁]
      ϵₖ₋₂    =  sₖ₋₂ * βₖ
      γbarₖ₋₁ = -cₖ₋₂ * βₖ
    end
    # Apply previous Givens reflections Qₖ₋₁.ₖ
    if iter ≥ 2
      iter == 2 && (γbarₖ₋₁ = βₖ)
      # [cₖ₋₁  sₖ₋₁] [γbarₖ₋₁] = [γₖ₋₁ ]
      # [sₖ₋₁ -cₖ₋₁] [   αₖ  ]   [λbarₖ]
      γₖ₋₁  = cₖ₋₁ * γbarₖ₋₁ + sₖ₋₁ * αₖ
      λbarₖ = sₖ₋₁ * γbarₖ₋₁ - cₖ₋₁ * αₖ
    end
    iter == 1 && (λbarₖ = αₖ)

    # Compute and apply current Givens reflection Qₖ.ₖ₊₁
    # [cₖ  sₖ] [λbarₖ] = [λₖ]
    # [sₖ -cₖ] [βₖ₊₁ ]   [0 ]
    (cₖ, sₖ, λₖ) = sym_givens(λbarₖ, βₖ₊₁)

    # Update z̅ₖ₊₁ = Qₖ.ₖ₊₁ [ z̄ₖ ]
    #                      [ 0  ]
    #
    # [cₖ  sₖ] [ζbarₖ] = [   ζₖ  ]
    # [sₖ -cₖ] [  0  ]   [ζbarₖ₊₁]
    ζₖ      = cₖ * ζbarₖ
    ζbarₖ₊₁ = sₖ * ζbarₖ

    # Compute the direction wₖ, the last column of Wₖ = Vₖ(Rₖ)⁻¹ ⟷ (Rₖ)ᵀ(Wₖ)ᵀ = (Vₖ)ᵀ.
    # w₁ = v₁ / λ₁
    if iter == 1
      wₖ = wₖ₋₁
      @kaxpy!(n, one(T), vₐᵤₓ, wₖ)
      @. wₖ = wₖ / λₖ
    end
    # w₂ = (v₂ - γ₁w₁) / λ₂
    if iter == 2
      wₖ = wₖ₋₂
      @kaxpy!(n, -γₖ₋₁, wₖ₋₁, wₖ)
      @kaxpy!(n, one(T), vₐᵤₓ, wₖ)
      @. wₖ = wₖ / λₖ
    end
    # wₖ = (vₖ - γₖ₋₁wₖ₋₁ - ϵₖ₋₂wₖ₋₂) / λₖ
    if iter ≥ 3
      @kscal!(n, -ϵₖ₋₂, wₖ₋₂)
      wₖ = wₖ₋₂
      @kaxpy!(n, -γₖ₋₁, wₖ₋₁, wₖ)
      @kaxpy!(n, one(T), vₐᵤₓ, wₖ)
      @. wₖ = wₖ / λₖ
    end

    # Compute solution xₖ.
    # xₖ ← xₖ₋₁ + ζₖ * wₖ
    @kaxpy!(n, ζₖ, wₖ, x)

    # Compute ‖rₖ‖ = |ζbarₖ₊₁|.
    rNorm = abs(ζbarₖ₊₁)
    push!(rNorms, rNorm)

    # Compute ‖ Arₖ₋₁ ‖ = |ζbarₖ| * √((λbarₖ)² + (γbarₖ)²)
    ArNorm = abs(ζbarₖ) * √(λbarₖ^2 + (cₖ₋₁ * βₖ₊₁)^2)
    push!(ArNorms, ArNorm)

    # Update M⁻¹vₖ₋₁, M⁻¹vₖ and vₖ
    @. M⁻¹vₖ₋₁ = M⁻¹vₖ
    @. M⁻¹vₖ   = p
    MisI || (vₖ = vₖ₊₁)

    # Update directions for x.
    if iter ≥ 2
      @kswap(wₖ₋₂, wₖ₋₁)
    end

    # Update sₖ₋₂, cₖ₋₂, sₖ₋₁, cₖ₋₁, ζbarₖ, βₖ.
    if iter ≥ 2
      sₖ₋₂ = sₖ₋₁
      cₖ₋₂ = cₖ₋₁
    end
    sₖ₋₁  = sₖ
    cₖ₋₁  = cₖ
    ζbarₖ = ζbarₖ₊₁
    βₖ    = βₖ₊₁

    # Update stopping criterion.
    iter == 1 && (κ = atol + rtol * ArNorm / 100)
    solved = rNorm ≤ ε
    inconsistent = !solved && ArNorm ≤ κ
    tired = iter ≥ itmax
    verbose && @printf("%5d  %7.1e  %7.1e\n", iter, rNorm, ArNorm)
  end
  verbose && @printf("\n")
  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, inconsistent, rNorms, ArNorms, status)
  return (x, stats)
end
