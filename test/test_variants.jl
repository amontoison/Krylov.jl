# Tests of variants.jl
function test_variants()
  @printf("Tests of variants:\n")
  for fn in (:cg, :cr, :lnlq, :craig, :craigmr, :lslq, :lsqr, :lsmr,
             :minres, :symmlq, :minres_qlp, :tricg, :trimr)
    @printf("%s ", string(fn))
    for T in (Float32, Float64, BigFloat)
      for S in (Int32, Int64)
        A_dense = Matrix{T}(I, 5, 5)
        A_sparse = convert(SparseMatrixCSC{T,S}, A_dense)
        b_dense = ones(T, 5)
        b_sparse = convert(SparseVector{T,S}, b_dense)
        for A in (A_dense, A_sparse)
          for b in (b_dense, b_sparse)
            if fn in (:tricg, :trimr)
              c_dense = ones(T, 5)
              c_sparse = convert(SparseVector{T,S}, c_dense)
              for c in (c_dense, c_sparse)
                @eval $fn($A, $b, $c)
                @eval $fn($transpose($A), $b, $c)
                @eval $fn($adjoint($A), $b, $c)
              end
            else
              @eval $fn($A, $b)
              @eval $fn($transpose($A), $b)
              @eval $fn($adjoint($A), $b)
            end
          end
        end
      end
    end
    @printf("✔\n")
  end
end

test_variants()
