# C and Fortran interfaces

!!! warning "Prototype"
    This interface is still in early development. Not all Krylov.jl features are exposed — in particular, most solver-specific options (restarts, window size, shifts, ...) are not yet accessible. The API may change in future releases.

Krylov.jl ships `libkrylov`, a native shared library that exposes all its solvers to C, Fortran, Python, R, or any other language that can call C code.

Pre-built self-contained binaries for Linux (x86-64, aarch64), macOS (arm64, x86-64) and Windows (x86-64) are available on the [Releases](https://github.com/JuliaSmoothOptimizers/Krylov.jl/releases) page.
For build instructions and the full API reference, see [`interfaces/C/README.md`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/C/README.md).

## API overview

The workflow is always the same regardless of the solver:

1. **Create** a workspace with `krylov_workspace_create`
2. **Solve** with `krylov_solve`, passing matrix-vector product callbacks
3. **Retrieve** the solution with `krylov_get_x` (and `krylov_get_y` for two-solution solvers)
4. **Free** the workspace with `krylov_workspace_free`

```c
// Callback signature: computes y = A*x, y = Aᴴ*x, or y = M\x
typedef void (*KrylovMatvec)(const void *x, void *y, void *userdata);

int krylov_workspace_create(KrylovSolverType solver,  // KRYLOV_CG, KRYLOV_GMRES, ...
                             int m, int n,
                             KrylovDataType dtype,     // KRYLOV_FLOAT64, ...
                             KrylovDeviceType device,  // KRYLOV_CPU
                             void **ws_out);

int krylov_solve(void *ws,
                 KrylovMatvec matvec_A,   // required
                 KrylovMatvec matvec_At,  // NULL if not needed
                 KrylovMatvec matvec_M,   // NULL = no preconditioner
                 const void *b,           // right-hand side (size m)
                 const void *c,           // second RHS, NULL if not needed
                 void *userdata,
                 double atol, double rtol,
                 int itmax,               // 0 = solver default
                 int verbose);            // 0 = silent

int    krylov_get_x(void *ws, void *x, int n);
int    krylov_get_y(void *ws, void *y, int m);
int    krylov_is_solved(void *ws);         // 1 = converged
int    krylov_niter(void *ws);
double krylov_elapsed_time(void *ws);
int    krylov_warm_start(void *ws, const void *x0, int n);
int    krylov_workspace_free(void *ws);
```

## C example

Solve a 5×5 tridiagonal SPD system with CG in double precision:

```c
#include <stdio.h>
#include "krylov.h"

#define N 5

typedef struct { int n; double diag[N]; double off[N-1]; } TriDiag;

static void matvec_A(const void *xv, void *yv, void *userdata)
{
  const double *x  = (const double *)xv;
  double       *y  = (double *)yv;
  const TriDiag *A = (const TriDiag *)userdata;
  for (int i = 0; i < A->n; i++) {
    y[i] = A->diag[i] * x[i];
    if (i > 0)     y[i] += A->off[i-1] * x[i-1];
    if (i < A->n-1) y[i] += A->off[i]   * x[i+1];
  }
}

int main(void)
{
  TriDiag A = { .n = N };
  for (int i = 0; i < N;   i++) A.diag[i] =  2.0;
  for (int i = 0; i < N-1; i++) A.off[i]  = -1.0;

  double b[N] = {1.0, 0.0, 0.0, 0.0, 1.0};
  double x[N];

  void *ws = NULL;
  krylov_workspace_create(KRYLOV_CG, N, N, KRYLOV_FLOAT64, KRYLOV_CPU, &ws);
  krylov_solve(ws, matvec_A, NULL, NULL, b, NULL, &A, 1e-10, 1e-10, 0, 0);
  krylov_get_x(ws, x, N);

  printf("Solved: %s   niter: %d\n",
         krylov_is_solved(ws) ? "yes" : "no", krylov_niter(ws));
  for (int i = 0; i < N; i++) printf(" %.2f", x[i]);
  printf("\n");

  krylov_workspace_free(ws);
  return 0;
}
```

## Fortran example

The same problem using the Fortran module (`krylov.f90` included in the bundle):

```fortran
program basic_cg
  use iso_c_binding
  implicit none
  include 'krylov.f90'

  integer, parameter    :: n = 5
  real(c_double), target :: diag(n), off(n-1), b(n), x(n)
  type(c_ptr)           :: ws
  integer(c_int)        :: ret

  diag = 2.0_c_double ; off = -1.0_c_double
  b = 0.0_c_double ; b(1) = 1.0_c_double ; b(n) = 1.0_c_double

  ret = krylov_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, ws)
  ret = krylov_solve(ws, c_funloc(matvec_A), c_null_funptr, c_null_funptr, &
                     c_loc(b), c_null_ptr, c_loc(diag), 1d-10, 1d-10, 0_c_int, 0_c_int)
  ret = krylov_get_x(ws, c_loc(x), int(n, c_int))

  write(*,*) "Solved:", krylov_is_solved(ws) == 1, "  niter:", krylov_niter(ws)
  write(*,'(5F6.2)') x

  ret = krylov_workspace_free(ws)

contains
  subroutine matvec_A(x_ptr, y_ptr, userdata) bind(c)
    type(c_ptr), value :: x_ptr, y_ptr, userdata
    real(c_double), pointer :: xv(:), yv(:), dg(:)
    integer :: i
    call c_f_pointer(x_ptr, xv, [n]) ; call c_f_pointer(y_ptr, yv, [n])
    call c_f_pointer(userdata, dg, [n])
    do i = 1, n
      yv(i) = dg(i)*xv(i)
      if (i > 1) yv(i) = yv(i) - xv(i-1)
      if (i < n) yv(i) = yv(i) - xv(i+1)
    end do
  end subroutine
end program
```
