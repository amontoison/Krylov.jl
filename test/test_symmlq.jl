function test_symmlq()
  symmlq_tol = 1.0e-6

  # Symmetric and positive definite system.
  n = 10
  A, b = symmetric_definite()
  (x, stats) = symmlq(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("SYMMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ symmlq_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  (x, stats) = symmlq(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("SYMMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ symmlq_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = symmlq(Matrix(A), b)
  show(stats)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  (x, stats) = symmlq(A, b, atol=1e-12, rtol=1e-12)
  r = b - A * x
  resid = norm(r) / norm(b)
  @show stats
  @printf("SYMMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ 100 * symmlq_tol)
  @test(stats.solved)

  # System that cause a breakdown with the symmetric Lanczos process.
  A, b = symmetric_breakdown()
  (x, stats) = symmlq(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("SYMMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ symmlq_tol)
  @test(stats.solved)
  show(stats)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = symmlq(A, b)
  @test x == b
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test with Jacobi (or diagonal) preconditioner
  A, b, M = square_preconditioned()
  (x, stats) = symmlq(A, b, M=M)
  show(stats)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("SYMMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ symmlq_tol)
  @test(stats.solved)
end

test_symmlq()
