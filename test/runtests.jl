using Test, Krylov, LinearOperators, LinearAlgebra, SparseArrays, Printf, Random

include("test_utils.jl")
include("test_aux.jl")

include("test_cg.jl")
include("test_cr.jl")

include("test_symmlq.jl")
include("test_minres.jl")
include("test_minres_qlp.jl")

#include("test_lslq.jl")
include("test_lsqr.jl")
include("test_lsmr.jl")

include("test_lnlq.jl")
include("test_craig.jl")
include("test_craigmr.jl")

include("test_tricg.jl")
include("test_trimr.jl")

include("test_usymlq.jl")
include("test_usymqr.jl")

# include("test_alloc.jl")
# include("test_mp.jl")
include("test_variants.jl")
