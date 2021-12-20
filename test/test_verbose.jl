@testset "verbose" begin
  A   = get_div_grad(4, 4, 4)  # Dimension n x n
  n   = size(A, 1)
  m   = div(n, 2)
  Au  = A[1:m,:]  # Dimension m x n
  Ao  = A[:,1:m]  # Dimension n x m
  b   = Ao * ones(m) # Dimension n
  c   = Au * ones(n) # Dimension m
  mem = 10
  shifts = [1.0; 2.0; 3.0; 4.0; 5.0]
  nshifts = 5

  expected = """
  CG: system of 64 equations in 64 variables
      k      ‖r‖       pAp         α         σ
      0  1.3e+01   5.0e+02   3.2e-01   3.2e-01
      1  5.6e+00   1.1e+02   3.0e-01   3.0e-01
      2  2.9e+00   3.5e+01   2.5e-01   2.5e-01
      3  1.5e+00   8.1e+00   2.8e-01   2.8e-01
      4  1.0e+00   4.6e+00   2.4e-01   2.4e-01
      5  3.6e-01   5.4e-01   2.4e-01   2.4e-01
      6  1.2e-01   8.3e-02   1.8e-01   1.8e-01
      7  2.6e-02   4.4e-03   1.6e-01   1.6e-01
      8  1.7e-16
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      cg(A, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  SYMMLQ: system of size 64
      k      ‖r‖        β       cos       sin      ‖A‖     κ(A)    test1
      0  1.3e+01  1.4e+00   1.0e+00   0.0e+00  0.0e+00  0.0e+00
      1  1.3e+01  1.7e+00   9.1e-01   4.1e-01  5.7e+00  1.0e+00  1.6e-01
      2  1.1e+01  2.1e+00  -8.7e-01   4.9e-01  8.0e+00  1.0e+00  5.9e-02
      3  9.5e+00  2.5e+00   8.6e-01   5.1e-01  9.9e+00  1.2e+00  3.5e-02
      4  6.2e+00  1.4e+00  -7.8e-01   6.3e-01  1.2e+01  1.2e+00  1.8e-02
      5  4.8e+00  1.4e+00   9.1e-01   4.1e-01  1.3e+01  1.2e+00  1.2e-02
      6  2.7e+00  1.2e+00  -9.4e-01   3.4e-01  1.4e+01  1.2e+00  5.8e-03
      7  1.4e+00  4.4e-14   9.7e-01   2.3e-01  1.6e+01  1.6e+00  2.8e-03
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      symmlq(A, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  MINRES: system of size 64
      k      ‖r‖    ‖Aᵀr‖        β       cos       sin      ‖A‖     κ(A)    test1    test2
      0  1.3e+01  0.0e+00  1.3e+01  -1.0e+00   0.0e+00  0.0e+00  0.0e+00
      1  5.1e+00  4.3e+01  1.4e+00   9.1e-01   4.1e-01  1.3e+01  1.0e+00  1.1e-01  2.6e-01
      2  2.5e+00  1.8e+01  1.7e+00  -8.7e-01   4.9e-01  1.4e+01  1.0e+00  3.9e-02  2.5e-01
      3  1.3e+00  1.0e+01  2.1e+00   8.6e-01   5.1e-01  1.5e+01  1.2e+00  1.6e-02  2.7e-01
      4  8.1e-01  4.9e+00  2.5e+00  -7.8e-01   6.3e-01  1.6e+01  1.2e+00  9.3e-03  2.4e-01
      5  3.3e-01  2.8e+00  1.4e+00   9.1e-01   4.1e-01  1.7e+01  1.2e+00  3.4e-03  2.0e-01
      6  1.1e-01  1.3e+00  1.4e+00  -9.4e-01   3.4e-01  1.8e+01  1.2e+00  1.1e-03  2.2e-01
      7  2.6e-02  6.2e-01  1.2e+00   9.7e-01   2.3e-01  1.9e+01  1.6e+00  2.4e-04  2.8e-01
      8  1.6e-16  1.6e-01  3.7e-14  -1.0e+00   6.0e-15  2.0e+01  1.8e+00  1.4e-18  3.0e-01
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      minres(A, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')
  
  expected = """
  CG Lanczos: system of 64 equations in 64 variables
      k     ‖rₖ‖
      0  1.3e+01
      1  5.6e+00
      2  2.9e+00
      3  1.5e+00
      4  1.0e+00
      5  3.6e-01
      6  1.2e-01
      7  2.6e-02
      8  2.4e-16
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      cg_lanczos(A, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  CG Lanczos: system of 64 equations in 64 variables with 5 shifts
      0   1.3e+01   1.3e+01   1.3e+01   1.3e+01   1.3e+01
      1   4.2e+00   3.4e+00   2.9e+00   2.4e+00   2.1e+00
      2   1.6e+00   1.1e+00   7.5e-01   5.6e-01   4.3e-01
      3   6.5e-01   3.5e-01   2.1e-01   1.4e-01   9.3e-02
      4   3.4e-01   1.5e-01   7.4e-02   4.2e-02   2.5e-02
      5   8.6e-02   3.0e-02   1.3e-02   6.6e-03   3.6e-03
      6   2.3e-02   6.7e-03   2.5e-03   1.1e-03   5.3e-04
      7   4.1e-03   1.1e-03   3.5e-04   1.4e-04   6.0e-05
      8   3.2e-17   7.1e-18   2.1e-18   7.3e-19   2.9e-19
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      cg_lanczos(A, b, shifts, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  DIOM: system of size 64
      k     ‖rₖ‖
      0  1.3e+01
      1  5.6e+00
      2  2.9e+00
      3  1.5e+00
      4  1.0e+00
      5  3.6e-01
      6  1.2e-01
      7  2.6e-02
      8  2.0e-15
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      diom(A, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  FOM: system of size 64
      k     ‖rₖ‖   hₖ₊₁.ₖ
      0  1.3e+01  ✗ ✗ ✗ ✗
      1  5.6e+00  1.4e+00
      2  2.9e+00  1.7e+00
      3  1.5e+00  2.1e+00
      4  1.0e+00  2.5e+00
      5  3.6e-01  1.4e+00
      6  1.2e-01  1.4e+00
      7  2.6e-02  1.2e+00
      8  2.0e-15  4.7e-13
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      fom(A, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  DQGMRES: system of size 64
      k     ‖rₖ‖
      0  1.3e+01
      1  5.1e+00
      2  2.5e+00
      3  1.3e+00
      4  8.1e-01
      5  3.3e-01
      6  1.1e-01
      7  2.6e-02
      8  2.0e-15
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      dqgmres(A, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  GMRES: system of size 64
      k     ‖rₖ‖   hₖ₊₁.ₖ
      0  1.3e+01  ✗ ✗ ✗ ✗
      1  5.1e+00  1.4e+00
      2  2.5e+00  1.7e+00
      3  1.3e+00  2.1e+00
      4  8.1e-01  2.5e+00
      5  3.3e-01  1.4e+00
      6  1.1e-01  1.4e+00
      7  2.6e-02  1.2e+00
      8  2.0e-15  4.7e-13
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      gmres(A, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  CR: system of 64 equations in 64 variables
      k      ‖x‖      ‖r‖     quad
      0   0.0e+00  1.3e+01  0.0e+00
      1   3.4e+00  5.1e+00 -2.5e+01
      2   4.7e+00  2.5e+00 -3.0e+01
      3   5.2e+00  1.3e+00 -3.1e+01
      4   5.4e+00  8.1e-01 -3.2e+01
      5   5.6e+00  3.3e-01 -3.2e+01
      6   5.6e+00  1.1e-01 -3.2e+01
      7   5.7e+00  2.6e-02 -3.2e+01
      8   5.7e+00  1.8e-07 -3.2e+01
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      cr(A, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  CGS: system of size 64
      k     ‖rₖ‖
      0  1.3e+01
      1  4.3e+00
      2  3.2e+00
      3  1.7e+00
      4  1.0e+00
      5  3.5e-02
      6  1.7e-02
      7  7.5e-04
      8  3.6e-15
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      cgs(A, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  BICGSTAB: system of size 64
      k     ‖rₖ‖        αₖ        ωₖ
      0  1.3e+01   1.0e+00   1.0e+00
      1  2.7e+00   3.2e-01   1.9e-01
      2  1.0e+00   3.0e-01   1.5e-01
      3  4.4e-01   2.5e-01   2.4e-01
      4  1.5e-01   2.8e-01   1.3e-01
      5  1.6e-02   2.4e-01   3.9e-01
      6  1.1e-03   2.4e-01   1.2e-01
      7  4.1e-05   1.8e-01   2.7e-01
      8  1.1e-18   1.6e-01   9.2e-02
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      bicgstab(A, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  CRMR: system of 32 equations in 64 variables
      k     ‖Aᵀr‖       ‖r‖
      0  3.22e+01  9.80e+00
      1  1.83e+01  5.97e+00
      2  1.18e+01  4.57e+00
      3  1.74e+01  3.15e+00
      4  2.14e+00  4.55e-01
      5  8.65e-01  1.93e-01
      6  1.84e-13  1.87e-14
  """


  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      crmr(Au, c, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  CRAIGMR: system of 32 equations in 64 variables
      k      ‖r‖    ‖Aᵀr‖        β        α       cos       sin     ‖A‖²
      0  9.8e+00  3.3e+00  9.8e+00  3.3e+00   0.0e+00   1.0e+00  1.1e+01
      1  6.0e+00  1.8e+01  2.5e+00  3.9e+00   7.9e-01   6.1e-01  3.2e+01
      2  4.6e+00  1.2e+01  3.7e+00  4.0e+00  -6.4e-01   7.7e-01  6.2e+01
      3  3.1e+00  1.7e+01  2.5e+00  7.6e+00   7.2e-01   6.9e-01  1.3e+02
      4  4.5e-01  2.1e+00  8.1e-01  4.8e+00  -9.9e-01   1.4e-01  1.5e+02
      5  1.9e-01  8.7e-01  2.2e+00  4.9e+00   9.1e-01   4.2e-01  1.8e+02
      6  3.6e-14  3.6e-13  8.4e-13  1.0e+01  -1.0e+00   1.9e-13  2.8e+02
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      craigmr(Au, c, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  CGNE: system of 32 equations in 64 variables
      k       ‖r‖
      0  9.80e+00
      1  7.52e+00
      2  7.10e+00
      3  4.34e+00
      4  4.60e-01
      5  2.13e-01
      6  1.56e-14
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      cgne(Au, c, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  LNLQ: system of 32 equations in 64 variables
      k     ‖rₖ‖
      0  9.8e+00
      1  9.8e+00
      2  1.5e+01
      3  1.8e+01
      4  5.1e+01
      5  4.7e+00
      6  2.0e+00
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      lnlq(Au, c, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

 expected = """
  CRAIG: system of 32 equations in 64 variables
      k       ‖r‖       ‖x‖       ‖A‖      κ(A)         α        β
      0  9.80e+00  0.00e+00  0.00e+00  0.00e+00
      1  7.52e+00  2.98e+00  4.15e+00  4.15e+00   3.3e+00  2.5e+00
      2  7.10e+00  3.56e+00  6.75e+00  1.01e+01   3.9e+00  3.7e+00
      3  4.34e+00  3.97e+00  8.24e+00  1.61e+01   4.0e+00  2.5e+00
      4  4.60e-01  4.01e+00  1.13e+01  2.57e+01   7.6e+00  8.1e-01
      5  2.13e-01  4.01e+00  1.24e+01  3.10e+01   4.8e+00  2.2e+00
      6  3.62e-14  4.01e+00  1.34e+01  3.62e+01   4.9e+00  8.4e-13
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      craig(Au, c, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  LSLQ: system of 64 equations in 32 variables
      k      ‖r‖    ‖Aᵀr‖        β        α       cos       sin     ‖A‖²     κ(A)     ‖xL‖
      0  1.3e+01  3.7e+01  1.3e+01  2.9e+00  -1.0e+00   0.0e+00  8.7e+00  0.0e+00  0.0e+00
      1  1.3e+01  3.7e+01  2.6e+00  3.8e+00   8.4e-01   5.4e-01  5.5e+00  1.2e+00  2.0e+00
      2  1.0e+01  4.8e+01  3.3e+00  4.3e+00  -7.5e-01   6.7e-01  7.7e+00  1.3e+00  2.9e+00
      3  6.7e+00  3.1e+01  7.9e-01  8.1e+00   7.1e-01   7.1e-01  1.1e+01  2.3e+00  4.4e+00
      4  7.5e+00  1.6e+02  8.4e-01  4.7e+00  -1.0e+00   8.9e-02  1.2e+01  2.6e+00  5.6e+00
      5  1.3e+01  9.9e+00  2.2e+00  5.0e+00   9.2e-01   3.9e-01  1.3e+01  2.6e+00  5.7e+00
      6  1.2e+01  2.9e+00  4.6e-12  9.9e+00  -1.0e+00   2.4e-12  1.7e+01  2.6e+00  5.7e+00
      7  1.3e+01  3.8e-11  9.3e-01  9.5e+00   1.0e+00   8.8e-02  1.9e+01  4.6e+00  5.7e+00
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      lslq(Ao, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  LSQR: system of 64 equations in 32 variables
      k        α        β      ‖r‖    ‖Aᵀr‖   compat  backwrd      ‖A‖     κ(A)
      0  1.3e+01  2.9e+00  1.3e+01  2.9e+00  0.0e+00  1.0e+00  2.9e+00  0.0e+00
      1  3.8e+00  2.6e+00  8.3e+00  2.4e+01  6.6e-01  5.8e-01  4.9e+00  1.3e+00
      2  4.3e+00  3.3e+00  6.3e+00  1.8e+01  5.0e-01  4.0e-01  7.0e+00  2.6e+00
      3  8.1e+00  7.9e-01  1.7e+00  1.3e+01  1.4e-01  9.4e-01  8.3e+00  4.9e+00
      4  4.7e+00  8.4e-01  1.8e-01  8.5e-01  1.4e-02  4.0e-01  1.2e+01  7.2e+00
      5  5.0e+00  2.2e+00  7.8e-02  3.5e-01  6.2e-03  3.6e-01  1.3e+01  8.2e+00
      6  9.9e+00  4.6e-12  7.9e-14  7.8e-13  6.2e-15  7.3e-01  1.4e+01  9.4e+00
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      lsqr(Ao, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  LSMR: system of 64 equations in 32 variables
      k      ‖r‖    ‖Aᵀr‖        β        α       cos       sin     ‖A‖²
      0  1.3e+01  2.9e+00  1.3e+01  2.9e+00   0.0e+00   1.0e+00  8.7e+00
      1  8.8e+00  2.0e+01  2.6e+00  3.8e+00   7.5e-01   6.6e-01  3.0e+01
      2  6.9e+00  1.3e+01  3.3e+00  4.3e+00   6.5e-01   7.6e-01  6.0e+01
      3  3.8e+00  9.5e+00  7.9e-01  8.1e+00   9.6e-01   2.7e-01  1.3e+02
      4  1.9e-01  8.4e-01  8.4e-01  4.7e+00   9.9e-01   1.1e-01  1.5e+02
      5  8.2e-02  3.3e-01  2.2e+00  5.0e+00   9.0e-01   4.3e-01  1.8e+02
      6  7.9e-14  7.8e-13  4.6e-12  9.9e+00   1.0e+00   1.0e-12  2.8e+02
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      lsmr(Ao, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  CGLS: system of 64 equations in 32 variables
      k     ‖Aᵀr‖       ‖r‖
      0  3.73e+01  1.26e+01
      1  2.38e+01  8.33e+00
      2  1.79e+01  6.33e+00
      3  1.34e+01  1.71e+00
      4  8.48e-01  1.83e-01
      5  3.54e-01  7.80e-02
      6  1.54e-13  1.51e-14
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      cgls(Ao, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  CRLS: system of 64 equations in 32 variables
      k     ‖Aᵀr‖       ‖r‖
      0  3.73e+01  1.26e+01
      1  2.01e+01  8.77e+00
      2  1.33e+01  6.88e+00
      3  9.45e+00  3.76e+00
      4  8.45e-01  1.85e-01
      5  3.26e-01  8.20e-02
      6  7.25e-14  7.59e-15
  """


  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      crls(Ao, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  USYMQR: system of 64 equations in 32 variables
      k     ‖rₖ‖  ‖Aᵀrₖ₋₁‖
      0  1.3e+01  ✗ ✗ ✗ ✗
      1  7.5e+00  3.3e+01
      2  7.4e+00  2.4e+01
      3  4.9e+00  2.3e+01
      4  4.8e+00  1.9e+01
      5  1.7e+00  2.0e+01
      6  1.4e-14  1.2e+01
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      usymqr(Ao, b, c, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  TRILQR: primal system of 64 equations in 64 variables
  TRILQR: dual system of 64 equations in 64 variables
      k     ‖rₖ‖     ‖sₖ‖
      0  1.3e+01  1.3e+01
      1  1.3e+01  1.3e+01
      2  1.1e+01  5.1e+00
      3  9.5e+00  2.5e+00
      4  6.2e+00  1.3e+00
      5  4.8e+00  8.1e-01
      6  2.7e+00  3.3e-01
      7  1.4e+00  1.1e-01
      8           2.6e-02
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      trilqr(A, b, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  BILQ: system of size 64
      k     ‖rₖ‖
      0  1.3e+01
      1  1.3e+01
      2  1.1e+01
      3  9.5e+00
      4  6.2e+00
      5  4.8e+00
      6  2.7e+00
      7  1.4e+00
      8  3.8e-01
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      bilq(A, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  BILQR: systems of size 64
      k     ‖rₖ‖     ‖sₖ‖
      0  1.3e+01  1.3e+01
      1  1.3e+01  1.3e+01
      2  1.1e+01  7.3e+00
      3  9.5e+00  4.4e+00
      4  6.2e+00  2.6e+00
      5  4.8e+00  1.8e+00
      6  2.7e+00  8.1e-01
      7  1.4e+00  3.0e-01
      8           7.3e-02
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      bilqr(A, b, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  MINRES-QLP: system of size 64
      k     ‖rₖ‖  ‖Arₖ₋₁‖
      0  1.3e+01  ✗ ✗ ✗ ✗
      1  5.1e+00  3.9e+01
      2  2.5e+00  1.8e+01
      3  1.3e+00  1.0e+01
      4  8.1e-01  4.9e+00
      5  3.3e-01  2.8e+00
      6  1.1e-01  1.3e+00
      7  2.6e-02  6.2e-01
      8  1.2e-16  1.6e-01
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      minres_qlp(A, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  QMR: system of size 64
      k     ‖rₖ‖
      0  1.3e+01
      1  7.3e+00
      2  4.4e+00
      3  2.6e+00
      4  1.8e+00
      5  8.1e-01
      6  3.0e-01
      7  7.3e-02
      8  1.1e-15
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      qmr(A, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  USYMLQ: system of 32 equations in 64 variables
      k     ‖rₖ‖
      0  9.8e+00
      1  9.8e+00
      2  7.5e+00
      3  7.3e+00
      4  7.1e+00
      5  7.3e+00
      6  3.7e+00
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      usymlq(Au, c, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  TriCG: system of 96 equations in 96 variables
      k     ‖rₖ‖        αₖ     βₖ₊₁     γₖ₊₁
      0  1.6e+01   ✗ ✗ ✗ ✗  9.8e+00  1.3e+01
      1  1.1e+01   2.6e+00  1.3e+00  2.0e+00
      2  2.5e+01   1.3e-01  3.8e+00  4.2e+00
      3  6.7e+00   9.3e-03  2.3e+00  2.6e+00
      4  1.9e+01   1.2e+00  5.2e+00  5.3e+00
      5  1.8e+00  -2.3e+00  1.3e+00  1.6e+00
      6  1.2e-14  -7.5e+00  4.9e-14  5.4e-14
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      tricg(Au, c, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  TriMR: system of 96 equations in 96 variables
      k     ‖rₖ‖        αₖ     βₖ₊₁     γₖ₊₁
      0  1.6e+01   ✗ ✗ ✗ ✗  9.8e+00  1.3e+01
      1  8.9e+00   2.6e+00  1.3e+00  2.0e+00
      2  8.4e+00   1.3e-01  3.8e+00  4.2e+00
      3  5.2e+00   9.3e-03  2.3e+00  2.6e+00
      4  5.1e+00   1.2e+00  5.2e+00  5.3e+00
      5  1.7e+00  -2.3e+00  1.3e+00  1.6e+00
      6  1.2e-14  -7.5e+00  4.9e-14  5.4e-14
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      trimr(Au, c, b, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')

  expected = """
  GPMR: system of 96 equations in 96 variables
      k     ‖rₖ‖   hₖ₊₁.ₖ   fₖ₊₁.ₖ
      0  1.6e+01  ✗ ✗ ✗ ✗  ✗ ✗ ✗ ✗
      1  6.9e+00  2.0e+00  1.3e+00
      2  6.8e+00  4.2e+00  3.8e+00
      3  4.3e+00  2.6e+00  2.3e+00
      4  4.1e+00  5.3e+00  5.2e+00
      5  1.6e+00  1.6e+00  1.3e+00
      6  1.1e-14  4.2e-14  4.7e-14
  """

  open("test.txt", "w") do fileio
    redirect_stdout(fileio) do
      gpmr(Ao, Au, b, c, verbose=1)
      Base.Libc.flush_cstdio()
    end
  end
  showed = read("test.txt", String)
  @test strip(showed, '\n') == strip(expected, '\n')
end
