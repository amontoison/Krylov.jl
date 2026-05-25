# Enumerators for the CKrylov C interface.
# Must match the values declared in krylov.h.
#
# GC roots are now managed by the typed store dicts in c_stores.jl.

# ---------------------------------------------------------------------------
# Data-type enum  (must match krylov.h)
# ---------------------------------------------------------------------------
# KRYLOV_FLOAT32    = 0
# KRYLOV_FLOAT64    = 1
# KRYLOV_COMPLEX32  = 2
# KRYLOV_COMPLEX64  = 3

# ---------------------------------------------------------------------------
# Device enum  (must match krylov.h)
# ---------------------------------------------------------------------------
# KRYLOV_CPU = 0   (only supported device for now)
