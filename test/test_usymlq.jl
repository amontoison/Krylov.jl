@testset "usymlq" begin
  usymlq_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      c = copy(b)
      (x, stats) = usymlq(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymlq_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      c = copy(b)
      (x, stats) = usymlq(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymlq_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      c = copy(b)
      (x, stats) = usymlq(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymlq_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      c = copy(b)
      (x, stats) = usymlq(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymlq_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      c = copy(b)
      (x, stats) = usymlq(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymlq_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      c = copy(b)
      (x, stats) = usymlq(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymlq_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      c = copy(b)
      (x, stats) = usymlq(A, b, c)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Underdetermined and consistent systems.
      A, b = under_consistent(FC=FC)
      c = ones(FC, 25)
      (x, stats) = usymlq(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymlq_tol)

      # Square and consistent systems.
      A, b = square_consistent(FC=FC)
      c = ones(FC, 10)
      (x, stats) = usymlq(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymlq_tol)

      # Overdetermined and consistent systems.
      A, b = over_consistent(FC=FC)
      c = ones(FC, 10)
      (x, stats) = usymlq(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymlq_tol)

      # System that cause a breakdown with the orthogonal tridiagonalization process.
      A, b, c = unsymmetric_breakdown(FC=FC)
      (x, stats) = usymlq(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymlq_tol)
      @test(stats.solved)

      # test callback function
      A, b = sparse_laplacian(FC=FC)
      c = copy(b)
      workspace = UsymlqWorkspace(A, b)
      tol = 1.0
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      usymlq!(workspace, A, b, c, atol = 0.0, rtol = 0.0, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError usymlq(A, b, c, callback = workspace -> "string", history = true)
    end
  end
end
