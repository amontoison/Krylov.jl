# LibKrylov — C interface for Krylov.jl

Exposes the Krylov.jl solvers as a native shared library (`libkrylov.so`) callable from C (and any language with a C FFI).

## Requirements

| Tool | Version |
|------|---------|
| Julia | ≥ 1.12 |
| [JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl) | latest |
| CMake | ≥ 3.20 (optional) |
| C compiler | gcc / clang |

[JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl) wraps Julia's `juliac` compiler and adds `--bundle`, which produces a **self-contained** library that embeds the Julia runtime — no separate Julia installation required at runtime.

## Build

All commands run from the **root of the Krylov.jl repository**.

### With JuliaC.jl (recommended — self-contained bundle)

```bash
# Install JuliaC.jl once (Julia app — installs juliac into ~/.julia/bin)
julia -e 'import Pkg; Pkg.Apps.add(url="https://github.com/JuliaLang/JuliaC.jl", rev="v0.3.2")'
export PATH="$HOME/.julia/bin:$PATH"   # add to ~/.bashrc to make permanent

# Build the bundle (library + embedded Julia runtime)
juliac \
    --project . \
    --compile-ccallable \
    --trim=safe \
    --bundle interfaces/C/build \
    --output-lib interfaces/C/build/lib/libkrylov.so \
    interfaces/C/src/LibKrylov.jl

# Generate the C header
julia --startup-file=no --project=. interfaces/C/scripts/generate_header.jl
cp interfaces/C/include/krylov.h     interfaces/C/build/include/
cp interfaces/Fortran/src/krylov.f90 interfaces/C/build/include/

# Compile an example
gcc -o basic_cg interfaces/C/examples/basic_cg.c \
    -I interfaces/C/build/include \
    interfaces/C/build/lib/libkrylov.so \
    -Wl,-rpath,'$ORIGIN/../lib/julia'
```

The `--bundle` flag produces a **relocatable** directory:

```
interfaces/C/build/
├── lib/
│   ├── libkrylov.so     ← the library
│   └── julia/           ← embedded Julia runtime (no system Julia needed)
└── include/
    ├── krylov.h
    └── krylov.f90
```

> **Windows:** use `--output-lib interfaces/C/build/bin/libkrylov.dll`; the bundle lands in `build/bin/`.  
> **macOS:** replace `.so` with `.dylib` and use `-Wl,-rpath,@loader_path/../lib/julia`.

### Without CMake, without bundle

For a simpler (non-relocatable) build that requires Julia to be installed at runtime:

```bash
JULIAC=$(julia --startup-file=no -e \
  "print(joinpath(Sys.BINDIR, \"..\", \"share\", \"julia\", \"juliac\", \"juliac.jl\"))")

julia --startup-file=no --project=. "$JULIAC" \
    --compile-ccallable \
    --experimental --trim=safe \
    --output-lib interfaces/C/build/libkrylov.so \
    interfaces/C/src/LibKrylov.jl

julia --startup-file=no --project=. interfaces/C/scripts/generate_header.jl

gcc -o basic_cg interfaces/C/examples/basic_cg.c \
    -I interfaces/C/include \
    interfaces/C/build/libkrylov.so \
    -Wl,-rpath,$(pwd)/interfaces/C/build
```

**Output sizes** (Linux x86-64, all 34 solvers × 4 precisions):

| Build | Size |
|-------|------|
| No trim | ~269 MB |
| `--trim=safe` | ~19 MB |

### With CMake

```bash
cd interfaces/C
mkdir build && cd build
cmake ..
make
```

CMake automatically:
1. Locates `julia` and `juliac.jl`
2. Runs `juliac --compile-ccallable --trim=safe` to produce `libkrylov.so`
3. Generates `include/krylov.h`
4. Compiles the example executables

## Run the examples

```bash
./build/basic_cg
# Solved: yes   niter: 3   time: 3.2e-05 s
# x = [ 1.00 1.00 1.00 1.00 1.00 ]
```

## API overview

```c
#include "krylov.h"

/* Callback type: computes y = A*x  or  y = A'*x  or  y = M\x */
typedef void (*KrylovMatvec)(const void *x, void *y, void *userdata);

/* 1. Create a workspace for a named solver */
int krylov_workspace_create(KrylovSolverType solver, /* KRYLOV_CG, KRYLOV_GMRES, ...  */
                             int m, int n,             /* operator dimensions           */
                             KrylovDataType dtype,     /* KRYLOV_FLOAT64, ...           */
                             KrylovDeviceType device,  /* KRYLOV_CPU                    */
                             void **ws_out);           /* receives the handle           */

/* 2. Solve */
int krylov_solve(void *ws,
                 KrylovMatvec matvec_A,   /* y = A*x  (required)               */
                 KrylovMatvec matvec_At,  /* y = A'*x (NULL if not needed)     */
                 KrylovMatvec matvec_M,   /* y = M\x  (NULL = no precond.)     */
                 const void *b,           /* right-hand side (size m)          */
                 const void *c,           /* second RHS (NULL if not needed)   */
                 void *userdata,          /* forwarded to every callback        */
                 double atol, double rtol,
                 int itmax,              /* 0 = solver default                 */
                 int verbose);           /* 0 = silent                         */

/* 3. Retrieve results */
int    krylov_get_x(void *ws, void *x, int n);   /* primal solution             */
int    krylov_get_y(void *ws, void *y, int m);   /* dual solution (TriCG, ...)  */
int    krylov_is_solved(void *ws);                /* 1=yes, 0=no, -1=error      */
int    krylov_niter(void *ws);
double krylov_elapsed_time(void *ws);             /* seconds                    */

/* 4. Optional: warm start */
int krylov_warm_start(void *ws, const void *x0, int n);

/* 5. Free */
int krylov_workspace_free(void *ws);
```

### Enumerators

```c
typedef enum { KRYLOV_FLOAT32=0, KRYLOV_FLOAT64=1,
               KRYLOV_COMPLEX32=2, KRYLOV_COMPLEX64=3 } KrylovDataType;

typedef enum { KRYLOV_CPU=0 } KrylovDeviceType;
```

### Which solvers need `matvec_At`?

| Pass `NULL` | Pass a callback |
|-------------|----------------|
| CG, CR, MINRES, MINRES-QLP, SYMMLQ, GMRES, FGMRES, FOM, DIOM, DQGMRES, BiCGSTAB, CGS, CAR, MINARES, TriCG, TriMR, GPMR | BiLQ, QMR, BiLQR, TriLQR, USYMLQ, USYMQR, USYMLQR, LSLQ, LSQR, LSMR, CGLS, CRLS, CGNE, CRMR, CRAIG, CRAIGMR, LNLQ |

### Minimal example (CG, double precision)

```c
#include "krylov.h"

static void my_matvec(const void *x, void *y, void *data) {
    /* fill y = A*x */
}

int main(void) {
    void *ws = NULL;
    krylov_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, &ws);
    krylov_solve(ws, my_matvec, NULL, NULL, b, NULL, NULL, 1e-10, 1e-10, 0, 0);
    krylov_get_x(ws, x, n);
    krylov_workspace_free(ws);
}
```

## Directory structure

```
interfaces/C/
├── src/
│   ├── LibKrylov.jl          # @ccallable functions (compiled by juliac)
│   ├── c_enums.jl          # KrylovDataType / KrylovDevice enum comments
│   ├── c_operator.jl       # COperator: C callback → Julia mul! operator
│   └── c_stores.jl         # AUTO-GENERATED — 136 typed workspace stores
├── scripts/
│   ├── generate_header.jl  # generates include/krylov.h
│   └── generate_stores.jl  # regenerates src/c_stores.jl (run when adding solvers)
├── include/
│   └── krylov.h            # generated — do not edit by hand
├── examples/
│   └── basic_cg.c          # CG on tridiag(-1,2,-1)
├── CMakeLists.txt
└── README.md
```
