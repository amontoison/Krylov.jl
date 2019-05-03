function test_usymlqr()
  usymlqr_tol = 1.0e-6
  sqd_tol = 1.0e-3

  # Symmetric and positive definite system.
  A, b = symmetric_definite()
  c = copy(b)
  (x, r, y, z, stats) = usymlqr(A, b, c)
  @test(norm(r + A * x - b) ≤ usymlqr_tol)
  #@test(norm(A' * r) ≤ usymlqr_tol)
  @test(norm(y + A * z) ≤ usymlqr_tol)
  @test(norm(A' * y - c) ≤ usymlqr_tol)
  s = r + y
  t = x + z
  sol = [s; t]
  rhs = [b; c]
  K = sqd(A)
  resid = norm(rhs - K * sol) / norm(rhs)
  @printf("USYMLQR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ sqd_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  c = copy(b)
  (x, r, y, z, stats) = usymlqr(A, b, c)
  @test(norm(r + A * x - b) ≤ usymlqr_tol)
  @test(norm(A' * r) ≤ usymlqr_tol)
  @test(norm(y + A * z) ≤ usymlqr_tol)
  @test(norm(A' * y - c) ≤ usymlqr_tol)
  s = r + y
  t = x + z
  sol = [s; t]
  rhs = [b; c]
  K = sqd(A)
  resid = norm(rhs - K * sol) / norm(rhs)
  @printf("USYMLQR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ sqd_tol)
  @test(stats.solved)

  # Nonsymmetric and positive definite systems.
  A, b = nonsymmetric_definite()
  c = copy(b)
  (x, r, y, z, stats) = usymlqr(A, b, c)
  @test(norm(r + A * x - b) ≤ usymlqr_tol)
  #@test(norm(A' * r) ≤ usymlqr_tol)
  @test(norm(y + A * z) ≤ usymlqr_tol)
  #@test(norm(A' * y - c) ≤ usymlqr_tol)
  s = r + y
  t = x + z
  sol = [s; t]
  rhs = [b; c]
  K = sqd(A)
  resid = norm(rhs - K * sol) / norm(rhs)
  @printf("USYMLQR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ sqd_tol)
  @test(stats.solved)

  # Nonsymmetric indefinite variant.
  A, b = nonsymmetric_indefinite()
  c = copy(b)
  (x, r, y, z, stats) = usymlqr(A, b, c)
  @test(norm(r + A * x - b) ≤ usymlqr_tol)
  #@test(norm(A' * r) ≤ usymlqr_tol)
  @test(norm(y + A * z) ≤ usymlqr_tol)
  @test(norm(A' * y - c) ≤ usymlqr_tol)
  s = r + y
  t = x + z
  sol = [s; t]
  rhs = [b; c]
  K = sqd(A)
  resid = norm(rhs - K * sol) / norm(rhs)
  @printf("USYMLQR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ sqd_tol)
  @test(stats.solved)

  # Code coverage.
  (x, r, y, z, stats) = usymlqr(Matrix(A), b, c)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  c = copy(b)
  (x, r, y, z, stats) = usymlqr(A, b, c)
  @test(norm(r + A * x - b) ≤ usymlqr_tol)
  #@test(norm(A' * r) ≤ usymlqr_tol)
  @test(norm(y + A * z) ≤ usymlqr_tol)
  #@test(norm(A' * y - c) ≤ usymlqr_tol)
  s = r + y
  t = x + z
  sol = [s; t]
  rhs = [b; c]
  K = sqd(A)
  resid = norm(rhs - K * sol) / norm(rhs)
  @printf("USYMLQR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ sqd_tol)
  @test(stats.solved)

  # Symmetric indefinite variant, almost singular.
  A, b = almost_singular()
  c = copy(b)
  (x, r, y, z, stats) = usymlqr(A, b, c)
  @test(norm(r + A * x - b) ≤ usymlqr_tol)
  #@test(norm(A' * r) ≤ usymlqr_tol)
  @test(norm(y + A * z) ≤ usymlqr_tol)
  #@test(norm(A' * y - c) ≤ usymlqr_tol)
  s = r + y
  t = x + z
  sol = [s; t]
  rhs = [b; c]
  K = sqd(A)
  resid = norm(rhs - K * sol) / norm(rhs)
  @printf("USYMLQR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ 10 * sqd_tol)
  @test(stats.solved)

  # # Test tridiagonalization in elliptic norms.
  # A, b, M, N = two_preconditioners()
  # A, b = square_int()
  # c = copy(b)
  # (x, r, y, z, stats) = usymlqr(A, b, c, M=M, N=N)
  # @test(norm(M * r + A * x - b) ≤ usymlqr_tol)
  # @test(norm(A' * r - N * x) ≤ usymlqr_tol)
  # @test(norm(M * y + A * z) ≤ usymlqr_tol)
  # @test(norm(A' * y - N * z - c) ≤ usymlqr_tol)
  # s = r + y
  # t = x + z
  # sol = [s; t]
  # rhs = [b; c]
  # K = sqd(A, Matrix(M), Matrix(N))
  # resid = norm(rhs - K * sol) / norm(rhs)
  # @printf("USYMLQR: Relative residual: %8.1e\n", resid)
  # @test(resid ≤ usymlqr_tol)
  # @test(stats.solved)
end

test_usymlqr()
