L   = get_div_grad(32, 32, 32)
n   = size(L, 1)
m   = div(n, 2)
A   = PreallocatedLinearOperator(L) # Dimension n x n
Au  = PreallocatedLinearOperator(L[1:m,:]) # Dimension m x n
Ao  = PreallocatedLinearOperator(L[:,1:m]) # Dimension n x m
b   = Ao * ones(m) # Dimension n
c   = Au * ones(n) # Dimension m
mem = 10

shifts  = [1.0; 2.0; 3.0; 4.0; 5.0]
nshifts = 5

# UniformScaling preconditioners I should work as I
M1 = I
M2 = I
cg(L, b, M=M1) # warmup
cg(L, b, M=M2) # warmup
opEye_bytes = @allocated cg(L, b, M=M1)
UniformScaling_bytes = @allocated cg(L, b, M=M2)
@test 0.99 * UniformScaling_bytes ≤ opEye_bytes ≤ 1.01 * UniformScaling_bytes

# without preconditioner and with Ap preallocated, SYMMLQ needs 4 n-vectors: x_lq, vold, v, w̅ (= x_cg)
storage_symmlq(n) = 4 * n
storage_symmlq_bytes(n) = 8 * storage_symmlq(n)

expected_symmlq_bytes = storage_symmlq_bytes(n)
symmlq(A, b)  # warmup
actual_symmlq_bytes = @allocated symmlq(A, b)
@test actual_symmlq_bytes ≤ 1.1 * expected_symmlq_bytes

# without preconditioner and with Ap preallocated, CG needs 3 n-vectors: x, r, p
storage_cg(n) = 3 * n
storage_cg_bytes(n) = 8 * storage_cg(n)

expected_cg_bytes = storage_cg_bytes(n)
cg(A, b)  # warmup
actual_cg_bytes = @allocated cg(A, b)
@test actual_cg_bytes ≤ 1.1 * expected_cg_bytes

# without preconditioner and with Ap preallocated, MINRES needs 5 n-vectors: x, r1, r2, w1, w2
storage_minres(n) = 5 * n
storage_minres_bytes(n) = 8 * storage_minres(n)

expected_minres_bytes = storage_minres_bytes(n)
minres(A, b)  # warmup
actual_minres_bytes = @allocated minres(A, b)
@test actual_minres_bytes ≤ 1.1 * expected_minres_bytes

# without preconditioner and with Ap preallocated, CR needs 4 n-vectors: x, r, p, q
storage_cr(n) = 4 * n
storage_cr_bytes(n) = 8 * storage_cr(n)

expected_cr_bytes = storage_cr_bytes(n)
cr(A, b, rtol=1e-6)  # warmup
actual_cr_bytes = @allocated cr(A, b, rtol=1e-6)
@test actual_cr_bytes ≤ 1.1 * expected_cr_bytes

# with (Ap, Aᵀq) preallocated, LNLQ needs:
# - 2 n-vector: x, v
# - 3 m-vectors: y, w̄, u
storage_lnlq(n, m) = 2 * n + 3 * m
storage_lnlq_bytes(n, m) = 8 * storage_lnlq(n, m)

expected_lnlq_bytes = storage_lnlq_bytes(n, m)
lnlq(Au, c)  # warmup
actual_lnlq_bytes = @allocated lnlq(Au, c)
@test actual_lnlq_bytes ≤ 1.1 * expected_lnlq_bytes

# with (Ap, Aᵀq) preallocated, CRAIGMR needs:
# - 2 n-vector: x, v
# - 4 m-vectors: y, u, w, wbar
storage_craigmr(n, m) = 2 * n + 4 * m
storage_craigmr_bytes(n, m) = 8 * storage_craigmr(n, m)

expected_craigmr_bytes = storage_craigmr_bytes(n, m)
craigmr(Au, c)  # warmup
actual_craigmr_bytes = @allocated craigmr(Au, c)
@test actual_craigmr_bytes ≤ 1.1 * expected_craigmr_bytes

# with (Ap, Aᵀq) preallocated, CRAIG needs:
# - 2 n-vector: x, v
# - 3 m-vectors: y, w, u
storage_craig(n, m) = 2 * n + 3 * m
storage_craig_bytes(n, m) = 8 * storage_craig(n, m)

expected_craig_bytes = storage_craig_bytes(n, m)
craig(Au, c)  # warmup
actual_craig_bytes = @allocated craig(Au, c)
@test actual_craig_bytes ≤ 1.1 * expected_craig_bytes

# without preconditioner and with (Ap, Aᵀq) preallocated, LSLQ needs:
# - 3 m-vectors: x_lq, v, w̄ (= x_cg)
# - 1 n-vector: u
storage_lslq(n, m) = 3 * m + n
storage_lslq_bytes(n, m) = 8 * storage_lslq(n, m)

expected_lslq_bytes = storage_lslq_bytes(n, m)
(x, stats) = lslq(Ao, b)  # warmup
actual_lslq_bytes = @allocated lslq(Ao, b)
@test actual_lslq_bytes ≤ 1.1 * expected_lslq_bytes

# without preconditioner and with (Ap, Aᵀq) preallocated, LSQR needs:
# - 3 m-vectors: x, v, w
# - 1 n-vector: u
storage_lsqr(n, m) = 3 * m + n
storage_lsqr_bytes(n, m) = 8 * storage_lsqr(n, m)

expected_lsqr_bytes = storage_lsqr_bytes(n, m)
(x, stats) = lsqr(Ao, b)  # warmup
actual_lsqr_bytes = @allocated lsqr(Ao, b)
@test actual_lsqr_bytes ≤ 1.1 * expected_lsqr_bytes

# without preconditioner and with (Ap, Aᵀq) preallocated, LSMR needs:
# - 4 m-vectors: x, v, h, hbar
# - 1 n-vector: u
storage_lsmr(n, m) = 4 * m + n
storage_lsmr_bytes(n, m) = 8 * storage_lsmr(n, m)

expected_lsmr_bytes = storage_lsmr_bytes(n, m)
(x, stats) = lsmr(Ao, b)  # warmup
actual_lsmr_bytes = @allocated lsmr(Ao, b)
@test actual_lsmr_bytes ≤ 1.1 * expected_lsmr_bytes

# with Ap preallocated, MINRES-QLP needs:
# - 5 n-vectors: wₖ₋₁, wₖ, vₖ₋₁, vₖ, x
storage_minres_qlp(n) = 5 * n
storage_minres_qlp_bytes(n) = 8 * storage_minres_qlp(n)

expected_minres_qlp_bytes = storage_minres_qlp_bytes(n)
minres_qlp(A, b)  # warmup
actual_minres_qlp_bytes = @allocated minres_qlp(A, b)
@test actual_minres_qlp_bytes ≤ 1.1 * expected_minres_qlp_bytes

# with (Ap, Aᵀp) preallocated, TriCG needs:
# - 5 n-vectors: yₖ, uₖ₋₁, uₖ, gy₂ₖ₋₁, gy₂ₖ
# - 5 m-vectors: xₖ, vₖ₋₁, vₖ, gx₂ₖ₋₁, gx₂ₖ
storage_tricg(n, m) = 5 * n + 5 * m
storage_tricg_bytes(n, m) = 8 * storage_tricg(n, m)

expected_tricg_bytes = storage_tricg_bytes(n, m)
tricg(Au, c, b)  # warmup
actual_tricg_bytes = @allocated tricg(Au, c, b)
@test actual_tricg_bytes ≤ 1.1 * expected_tricg_bytes

# with (Ap, Aᵀp) preallocated, TriMR needs:
# - 7 n-vectors: yₖ, uₖ₋₁, uₖ, gy₂ₖ₋₃, gy₂ₖ₋₂, gy₂ₖ₋₁, gy₂ₖ
# - 7 m-vectors: xₖ, vₖ₋₁, vₖ, gx₂ₖ₋₃, gx₂ₖ₋₂, gx₂ₖ₋₁, gx₂ₖ
storage_trimr(n, m) = 7 * n + 7 * m
storage_trimr_bytes(n, m) = 8 * storage_trimr(n, m)

expected_trimr_bytes = storage_trimr_bytes(n, m)
trimr(Au, c, b)  # warmup
actual_trimr_bytes = @allocated trimr(Au, c, b)
@test actual_trimr_bytes ≤ 1.1 * expected_trimr_bytes
