function test_mp()
  @printf("Tests of multi-precision methods:\n")
  n = 5
  for fn in (:cg, :cr, :lnlq, :craig, :craigmr, :lslq, :lsqr, :lsmr,
             :minres, :symmlq, :minres_qlp, :tricg, :trimr)
    @printf("%s ", string(fn))
    for T in (Float16, Float32, Float64, BigFloat)
      A = spdiagm(-1 => -ones(T,n-1), 0 => 3*ones(T,n), 1 => -ones(T,n-1))
      b = ones(T, n)
      c = - ones(T, n)
      λ = zero(T)
      if fn in (:tricg, :trimr)
        x, y = @eval $fn($A, $b, $c)[1:2]
      else
        x = @eval $fn($A, $b)[1]
      end
      atol = √eps(T)
      rtol = √eps(T)
      Κ = (T == Float16 ? 10 : 1)
      if fn in (:tricg, :trimr)
        @test norm(x + A * y - b) ≤ Κ * (atol + norm([b; c]) * rtol)
        @test norm(A' * x - y - c) ≤ Κ * (atol + norm([b; c]) * rtol)
        @test eltype(y) == T
      else
        @test norm(A * x - b) ≤ Κ * (atol + norm(b) * rtol)
      end
      @test eltype(x) == T
    end
    @printf("✔\n")
  end
  @printf("\n")
end

test_mp()
