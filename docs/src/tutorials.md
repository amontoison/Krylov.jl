## BiLQ - QMR - BiLQR

```julia
using LinearAlgebra, SparseArrays 
using Printf
using MatrixMarket
using Krylov
using Plots
```


```julia
#BiLQ - combines BiCG with QMR 

#load matrix from https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/lanpro/nos6.html
#Finite difference approximation to Poisson's equation in an L-shaped region, mixed boundary conditions
A = MatrixMarket.mmread("pde2961.mtx")
m = A.m
n = A.n

#explore the matrix A
con = cond(Array(A))
@printf("Condition number: %3d\n", con)
@show(A)

#create initial vector b 
b = A * ones(n)

#optional additional vector. If not specified, c = b
c = A * (ones(n) .* 0.5)

#run BiLQ to solve for x
x, bilq_stats = bilq(A, b, c=c, history=true)

#find residual from solution
r = b - A * x

#show results
bilq_stats.solved && print("System solved.\n")
@printf("\tResidual norm: %8.1e\n", norm(r))
@printf("\tNumber of iterations: %3d\n", length(bilq_stats.residuals) - 1)
```

    Condition number: 642
    A = 
    ⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⣄⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷
    System solved.
      Residual norm:  3.7e-07
      Number of iterations: 291



```julia
#QMR
x, qmr_stats = qmr(A, b, c=c, history=true)

#find residual from solution
r = b - A * x

#show results
qmr_stats.solved && print("System solved.\n")
@printf("\tResidual norm: %8.1e\n", norm(r))
@printf("\tNumber of iterations: %3d\n", length(qmr_stats.residuals) - 1)
```

    System solved.
      Residual norm:  5.7e-08
      Number of iterations: 316



```julia
#BiLQR
x, y, bilqr_stats = bilqr(A, b, c, history=true)

#find residual from solution
r_x = b - A * x
r_y = c - A * y

#show results
stats.solved_primal && print("Primal system solved.\n")
@printf("\tPrimal sytem residual norm: %8.1e\n", norm(r_x))
@printf("\tNumber of iterations: %3d\n", length(bilqr_stats.residuals_primal) - 1)

stats.solved_primal && print("Dual system solved.\n")
@printf("\tAdjiont system residual norm: %8.1e\n", norm(r_y))
@printf("\tNumber of iterations: %3d\n", length(bilqr_stats.residuals_dual) - 1)
```

    Primal system solved.
      Primal sytem residual norm:  3.7e-07
      Number of iterations: 291
    Dual system solved.
      Adjiont system residual norm:  3.0e+00
      Number of iterations: 285



```julia
#explore the norms of the residuals of each solver
plot(bilqr_stats.residuals_primal, marker=:circle, yscale=:log10, label="BiLQR_primal", title="Residual Norms Across Methods")
plot!(bilqr_stats.residuals_dual, marker=:none, yscale=:log10, label="BiLQR_dual")
plot!(bilq_stats.residuals, marker=:none, yscale=:log10, label="BiLQ")
plot!(qmr_stats.residuals, marker=:none, yscale=:log10, label="QMR")
```




![svg](output_5_0.svg)



# A discussion of each method

Each of the three methods explored in this notebook offer ways of solving a (not necessarily symmetric) square linear system $Ax = b$. All of the methods use the Lanczos biorthogonalization process. The discussions below provide some insight into which method to use and when.

## BiLQ

You can think of this method as a blend of the BiCG (biconjugate gradient) and QMR methods. As with each of these three methods, you begin with two initial vectors, $b$ and $c$ such that $b^Tc \neq 0$. Note that the requirement for two initialization vectors stems from the biorthogonalization process. A few unique facets of this method:
- works when A is ill-conditioned or rank deficient
- This approach uses LQ factorization
- a lookahead procedure is NOT implemented
- This is a quasi-minimum error method
- Neither error nor residual norm are monotonic

## QMR

The quasi-minimal residual method (QMR) is sort of a numerically stable version of the Biconjugate Gradient (BiCG) method that features smoother convergence in certain situations.
- BiCG has irregular convergence behavior and relies on LU factorization (which may not exist)
- Similar to GMRES, but the constructed basis is biorthogonal 
- Typically features much smoother convergence than BiCG
- Converges as quickly as GMRES
- Relies on a lookahead step to prevent breakdown of Lanczos process

## BiLQR

This method combines the approach of BiLQ and QMR to solve both the primal $Ax = b$ and the adjoint system  $A^Ty = c$ simultaneously. This method can be used to estimate integral functionals involving the solution of a primal and an adjoint system
- Solving both systems simultaneously is nice because the discretization of the primal and dual systems usually differ
- No extra factorization updates are necessary to solve the dual system (thus the dual system can be solved cheaply) 
