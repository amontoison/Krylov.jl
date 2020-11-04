export symmlq

"""
    (x, stats) = symmlq(A, b; atol, rtol, transfer_to_cg, itmax, verbose)
"""
function symmlq(A, b :: AbstractVector{T};
                M=opEye(), atol :: T=√eps(T), rtol :: T=√eps(T), transfer_to_cg :: Bool=true,
                itmax :: Int=0, verbose :: Bool=false) where T <: AbstractFloat

  n, m = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  verbose && @printf("SYMMLQ: system of size %d\n", n)

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
  verbose && @printf("%5s  %7s\n", "k", "‖rₖ‖")
  verbose && @printf("%5d  %7.1e\n", iter, rNorm)

  # Set up workspace.
  M⁻¹vₖ₋₁ = kzeros(S, n)
  cₖ₋₁ = cₖ = -one(T)        # Givens cosines used for the LQ factorization of Tₖ
  sₖ₋₁ = sₖ = zero(T)        # Givens sines used for the LQ factorization of Tₖ
  d̅ = kzeros(S, n)           # Last column of D̅ₖ = Uₖ(Qₖ)ᵀ
  ζₖ₋₁ = ζbarₖ = zero(T)     # ζₖ₋₁ and ζbarₖ are the last components of z̅ₖ = (L̅ₖ)⁻¹β₁e₁
  ζₖ₋₂ = ηₖ = zero(T)        # ζₖ₋₂ and ηₖ are used to update ζₖ₋₁ and ζbarₖ
  δbarₖ₋₁ = δbarₖ = zero(T)  # Coefficients of Lₖ₋₁ and Lₖ modified over the course of two iterations

  # Use M⁻¹vₖ₋₁ to store vₖ when a preconditioner is provided
  MisI ? (vₐᵤₓ = vₖ) : (vₐᵤₓ = M⁻¹vₖ₋₁)

  # Stopping criterion.
  solved_lq = rNorm ≤ ε
  solved_cg = false
  tired     = iter ≥ itmax
  status    = "unknown"

  while !(solved_lq || solved_cg || tired)
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
    vₖ₊₁ = M * p          # βₖ₊₁vₖ₊₁ = MAvₖ - βₖvₖ₋₁ - αₖvₖ

    βₖ₊₁ = sqrt(@kdot(m, vₖ₊₁, p))

    if βₖ₊₁ ≠ 0
      @kscal!(m, one(T) / βₖ₊₁, vₖ₊₁)
      MisI || @kscal!(m, one(T) / βₖ₊₁, p)
    end

    # Update the LQ factorization of Tₖ = L̅ₖQₖ.
    # [ α₁ β₂ 0  •  •  •  0 ]   [ δ₁   0    •   •   •    •    0   ]
    # [ β₂ α₂ β₃ •        • ]   [ λ₁   δ₂   •                 •   ]
    # [ 0  •  •  •  •     • ]   [ ϵ₁   λ₂   δ₃  •             •   ]
    # [ •  •  •  •  •  •  • ] = [ 0    •    •   •   •         •   ] Qₖ
    # [ •     •  •  •  •  0 ]   [ •    •    •   •   •    •    •   ]
    # [ •        •  •  •  βₖ]   [ •         •   •   •    •    0   ]
    # [ 0  •  •  •  0  βₖ αₖ]   [ •    •    •   0  ϵₖ₋₂ λₖ₋₁ δbarₖ]

    if iter == 1
      δbarₖ = αₖ
    elseif iter == 2
      # [δbar₁ β₂] [c₂  s₂] = [δ₁   0  ]
      # [ β₂   α₂] [s₂ -c₂]   [λ₁ δbar₂]
      (cₖ, sₖ, δₖ₋₁) = sym_givens(δbarₖ₋₁, βₖ)
      λₖ₋₁  = cₖ * βₖ + sₖ * αₖ
      δbarₖ = sₖ * βₖ - cₖ * αₖ
    else
      # [0  βₖ  αₖ] [cₖ₋₁   sₖ₋₁   0] = [sₖ₋₁βₖ  -cₖ₋₁βₖ  αₖ]
      #             [sₖ₋₁  -cₖ₋₁   0]
      #             [ 0      0     1]
      #
      # [ λₖ₋₂   δbarₖ₋₁  βₖ] [1   0   0 ] = [λₖ₋₂  δₖ₋₁    0  ]
      # [sₖ₋₁βₖ  -cₖ₋₁βₖ  αₖ] [0   cₖ  sₖ]   [ϵₖ₋₂  λₖ₋₁  δbarₖ]
      #                       [0   sₖ -cₖ]
      (cₖ, sₖ, δₖ₋₁) = sym_givens(δbarₖ₋₁, βₖ)
      ϵₖ₋₂  =  sₖ₋₁ * βₖ
      λₖ₋₁  = -cₖ₋₁ * cₖ * βₖ + sₖ * αₖ
      δbarₖ = -cₖ₋₁ * sₖ * βₖ - cₖ * αₖ
    end

    # Compute ζₖ₋₁ and ζbarₖ, last components of the solution of Lₖz̅ₖ = β₁e₁
    # [δbar₁] [ζbar₁] = [β₁]
    if iter == 1
      ηₖ = βₖ
    end
    # [δ₁    0  ] [  ζ₁ ] = [β₁]
    # [λ₁  δbar₂] [ζbar₂]   [0 ]
    if iter == 2
      ηₖ₋₁ = ηₖ
      ζₖ₋₁ = ηₖ₋₁ / δₖ₋₁
      ηₖ   = -λₖ₋₁ * ζₖ₋₁
    end
    # [λₖ₋₂  δₖ₋₁    0  ] [ζₖ₋₂ ] = [0]
    # [ϵₖ₋₂  λₖ₋₁  δbarₖ] [ζₖ₋₁ ]   [0]
    #                     [ζbarₖ]
    if iter ≥ 3
      ζₖ₋₂ = ζₖ₋₁
      ηₖ₋₁ = ηₖ
      ζₖ₋₁ = ηₖ₋₁ / δₖ₋₁
      ηₖ   = -ϵₖ₋₂ * ζₖ₋₂ - λₖ₋₁ * ζₖ₋₁
    end

    # Relations for the directions dₖ₋₁ and d̅ₖ, the last two columns of D̅ₖ = Vₖ(Qₖ)ᵀ.
    # [d̅ₖ₋₁ vₖ] [cₖ  sₖ] = [dₖ₋₁ d̅ₖ] ⟷ dₖ₋₁ = cₖ * d̅ₖ₋₁ + sₖ * vₖ
    #           [sₖ -cₖ]             ⟷ d̅ₖ   = sₖ * d̅ₖ₋₁ - cₖ * vₖ
    if iter ≥ 2
      # Compute solution xₖ.
      # (xᴸ)ₖ₋₁ ← (xᴸ)ₖ₋₂ + ζₖ₋₁ * dₖ₋₁
      @kaxpy!(n, ζₖ₋₁ * cₖ,  d̅, x)
      @kaxpy!(n, ζₖ₋₁ * sₖ, vₐᵤₓ, x)
    end

    # Compute d̅ₖ.
    if iter == 1
      # d̅₁ = v₁
      @. d̅ = vₐᵤₓ
    else
      # d̅ₖ = sₖ * d̅ₖ₋₁ - cₖ * vₖ
      @kaxpby!(n, -cₖ, vₐᵤₓ, sₖ, d̅)
    end

    # Update M⁻¹vₖ₋₁, M⁻¹vₖ and vₖ
    @. M⁻¹vₖ₋₁ = M⁻¹vₖ
    @. M⁻¹vₖ   = p
    MisI || (vₖ = vₖ₊₁)

    # Compute USYMLQ residual norm
    # ‖rₖ‖ = √((μₖ)² + (ωₖ)²)
    if iter == 1
      rNorm_lq = rNorm
    else
      μₖ = βₖ * (sₖ₋₁ * ζₖ₋₂ - cₖ₋₁ * cₖ * ζₖ₋₁) + αₖ * sₖ * ζₖ₋₁
      ωₖ = βₖ₊₁ * sₖ * ζₖ₋₁
      rNorm_lq = sqrt(μₖ^2 + ωₖ^2)
    end
    push!(rNorms, rNorm_lq)

    # Compute USYMCG residual norm
    # ‖rₖ‖ = |ρₖ|
    if transfer_to_cg && (δbarₖ ≠ 0)
      ζbarₖ = ηₖ / δbarₖ
      ρₖ = βₖ₊₁ * (sₖ * ζₖ₋₁ - cₖ * ζbarₖ)
      rNorm_cg = abs(ρₖ)
    end

    # Update sₖ₋₁, cₖ₋₁, βₖ, βₖ and δbarₖ₋₁.
    sₖ₋₁    = sₖ
    cₖ₋₁    = cₖ
    βₖ      = βₖ₊₁
    βₖ      = βₖ₊₁
    δbarₖ₋₁ = δbarₖ

    # Update stopping criterion.
    solved_lq = rNorm_lq ≤ ε
    solved_cg = transfer_to_cg && (δbarₖ ≠ 0) && (rNorm_cg ≤ ε)
    tired = iter ≥ itmax
    verbose && @printf("%5d  %7.1e\n", iter, rNorm_lq)
  end
  verbose && @printf("\n")

  # Compute USYMCG point
  # (xᶜ)ₖ ← (xᴸ)ₖ₋₁ + ζbarₖ * d̅ₖ
  if solved_cg
    @kaxpy!(n, ζbarₖ, d̅, x)
  end

  tired     && (status = "maximum number of iterations exceeded")
  solved_lq && (status = "solution xᴸ good enough given atol and rtol")
  solved_cg && (status = "solution xᶜ good enough given atol and rtol")
  stats = SimpleStats(solved_lq || solved_cg, false, rNorms, T[], status)
  return (x, stats)
end
