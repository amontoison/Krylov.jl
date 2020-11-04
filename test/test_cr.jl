function test_cr()
  cr_tol = 1.0e-6

  # Symmetric and positive definite system.
  A, b = symmetric_definite()
  (x, stats) = cr(A, b, verbose=true)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("CR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ cr_tol)
  @test(stats.solved)

  # Code coverage
  (x, stats) = cr(Matrix(A), b, verbose=true)
  show(stats)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = cr(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test with Jacobi (or diagonal) preconditioner
  A, b, M = square_preconditioned()
  (x, stats) = cr(A, b, M=M)
  show(stats)
  r = b - A * x
  resid = sqrt(dot(r, M * r)) / norm(b)
  @printf("CR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ cr_tol)
  @test(stats.solved)
end

test_cr()
