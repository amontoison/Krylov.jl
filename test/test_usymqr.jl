@testset "usymqr" begin
  usymqr_tol = 1.0e-6

  # Symmetric and positive definite system.
  A, b = symmetric_definite()
  c = copy(b)
  (x, stats) = usymqr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  c = copy(b)
  (x, stats) = usymqr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)
  @test(stats.solved)

  # Nonsymmetric and positive definite systems.
  A, b = nonsymmetric_definite()
  c = copy(b)
  (x, stats) = usymqr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)
  @test(stats.solved)

  # Nonsymmetric indefinite variant.
  A, b = nonsymmetric_indefinite()
  c = copy(b)
  (x, stats) = usymqr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = usymqr(Matrix(A), b)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  c = copy(b)
  (x, stats) = usymqr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)
  @test(stats.solved)

  # Symmetric indefinite variant, almost singular.
  A, b = almost_singular()
  c = copy(b)
  (x, stats) = usymqr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)
  @test(stats.solved)

  # Test b == 0
  A, b = zero_rhs()
  c = copy(b)
  (x, stats) = usymqr(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Square and consistent systems.
  A, b = square_consistent()
  c = ones(10)
  (x, stats) = usymqr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)

  # Square and inconsistent systems.
  A, b = square_inconsistent()
  c = ones(10)
  (x, stats) = usymqr(A, b)
  @test stats.inconsistent

  # Poisson equation in polar coordinates.
  A, b = polar_poisson()
  (x, stats) = usymqr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)
  @test(stats.solved)
end
