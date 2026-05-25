! krylov.f90 — Fortran interface to libkrylov.so
!
! Usage:
!   Add  use iso_c_binding  and  include 'krylov.f90'  AFTER  implicit none
!   in your program or subroutine.
!
! Example:
!
!   program my_prog
!     use iso_c_binding
!     implicit none
!     include 'krylov.f90'    ! ← here, after implicit none
!     ...
!   end program
!
! Callbacks must match the krylov_matvec abstract interface and be passed
! via c_funloc(my_sub).  Pass c_null_funptr for unused callbacks.

  ! -------------------------------------------------------------------------
  ! Enumerators  (must match krylov.h)
  ! -------------------------------------------------------------------------

  integer(c_int), parameter :: KRYLOV_FLOAT32   = 0
  integer(c_int), parameter :: KRYLOV_FLOAT64   = 1
  integer(c_int), parameter :: KRYLOV_COMPLEX32  = 2
  integer(c_int), parameter :: KRYLOV_COMPLEX64  = 3

  integer(c_int), parameter :: KRYLOV_CPU = 0

  ! -------------------------------------------------------------------------
  ! Callback interface
  !
  ! Declare your matvec subroutine with this exact signature, then pass it
  ! as c_funloc(my_matvec).
  !
  ! Example:
  !
  !   subroutine my_matvec(x_ptr, y_ptr, userdata) bind(c)
  !     use iso_c_binding
  !     type(c_ptr), value :: x_ptr, y_ptr, userdata
  !     real(c_double), pointer :: x(:), y(:)
  !     call c_f_pointer(x_ptr, x, [n])
  !     call c_f_pointer(y_ptr, y, [n])
  !     y = matmul(A, x)
  !   end subroutine
  ! -------------------------------------------------------------------------

  abstract interface
    subroutine krylov_matvec(x_ptr, y_ptr, userdata) bind(c)
      use iso_c_binding
      type(c_ptr), value :: x_ptr    ! read-only input vector
      type(c_ptr), value :: y_ptr    ! output vector
      type(c_ptr), value :: userdata ! opaque user context
    end subroutine krylov_matvec
  end interface

  ! -------------------------------------------------------------------------
  ! C function interfaces
  ! -------------------------------------------------------------------------

  interface

    ! -----------------------------------------------------------------------
    ! krylov_workspace_create
    !
    ! Creates a workspace for the named solver.
    !
    !   solver  : null-terminated solver name, e.g. "cg"//c_null_char
    !   m, n    : operator dimensions
    !   dtype   : KRYLOV_FLOAT32 / KRYLOV_FLOAT64 / KRYLOV_COMPLEX32 / KRYLOV_COMPLEX64
    !   device  : KRYLOV_CPU
    !   ws      : receives the opaque workspace handle
    !
    ! Returns 0 on success, nonzero on error.
    ! -----------------------------------------------------------------------
    function krylov_workspace_create(solver, m, n, dtype, device, ws) &
        bind(c, name='krylov_workspace_create') result(ret)
      use iso_c_binding
      character(kind=c_char), dimension(*), intent(in) :: solver
      integer(c_int),         value                    :: m, n, dtype, device
      type(c_ptr),            intent(out)              :: ws
      integer(c_int)                                   :: ret
    end function krylov_workspace_create

    ! -----------------------------------------------------------------------
    ! krylov_solve
    !
    !   ws         : workspace handle
    !   matvec_A   : callback  y = A*x          (required)
    !   matvec_At  : callback  y = A'*x         (c_null_funptr for CG/GMRES/...)
    !   matvec_M   : callback  y = M\x          (c_null_funptr = no preconditioner)
    !   b          : right-hand side pointer (c_loc of your array)
    !   userdata   : forwarded to every callback (c_loc or c_null_ptr)
    !   atol,rtol  : tolerances
    !   itmax      : max iterations (0 = solver default)
    !   verbose    : verbosity level (0 = silent)
    !
    ! Returns 0 on success, nonzero on error.
    ! -----------------------------------------------------------------------
    function krylov_solve(ws, matvec_A, matvec_At, matvec_M, &
                          b, userdata, atol, rtol, itmax, verbose) &
        bind(c, name='krylov_solve') result(ret)
      use iso_c_binding
      type(c_ptr),    value :: ws
      type(c_funptr), value :: matvec_A    ! y = A*x
      type(c_funptr), value :: matvec_At   ! y = A'*x  or  c_null_funptr
      type(c_funptr), value :: matvec_M    ! y = M\x   or  c_null_funptr
      type(c_ptr),    value :: b           ! c_loc(b_array)
      type(c_ptr),    value :: userdata    ! c_loc(data)  or  c_null_ptr
      real(c_double), value :: atol, rtol
      integer(c_int), value :: itmax, verbose
      integer(c_int)        :: ret
    end function krylov_solve

    ! -----------------------------------------------------------------------
    ! krylov_get_x
    !
    ! Copies the primal solution into the buffer pointed to by x.
    ! Pass c_loc(x_array).
    !
    ! Returns 0 on success, nonzero on error.
    ! -----------------------------------------------------------------------
    function krylov_get_x(ws, x, n) &
        bind(c, name='krylov_get_x') result(ret)
      use iso_c_binding
      type(c_ptr),    value :: ws
      type(c_ptr),    value :: x   ! c_loc(x_array)
      integer(c_int), value :: n
      integer(c_int)        :: ret
    end function krylov_get_x

    ! -----------------------------------------------------------------------
    ! krylov_get_y
    !
    ! Copies the dual solution y (for solvers with two outputs: TriCG, GPMR, ...).
    ! Returns -2 if the solver has only one solution.
    ! -----------------------------------------------------------------------
    function krylov_get_y(ws, y, m) &
        bind(c, name='krylov_get_y') result(ret)
      use iso_c_binding
      type(c_ptr),    value :: ws
      type(c_ptr),    value :: y   ! c_loc(y_array)
      integer(c_int), value :: m
      integer(c_int)        :: ret
    end function krylov_get_y

    ! -----------------------------------------------------------------------
    ! krylov_is_solved
    !
    ! Returns 1 if converged, 0 if not, -1 on error.
    ! -----------------------------------------------------------------------
    function krylov_is_solved(ws) &
        bind(c, name='krylov_is_solved') result(ret)
      use iso_c_binding
      type(c_ptr),   value :: ws
      integer(c_int)       :: ret
    end function krylov_is_solved

    ! -----------------------------------------------------------------------
    ! krylov_niter
    !
    ! Returns the number of iterations performed, or -1 on error.
    ! -----------------------------------------------------------------------
    function krylov_niter(ws) &
        bind(c, name='krylov_niter') result(ret)
      use iso_c_binding
      type(c_ptr),   value :: ws
      integer(c_int)       :: ret
    end function krylov_niter

    ! -----------------------------------------------------------------------
    ! krylov_elapsed_time
    !
    ! Returns elapsed solve time in seconds, or -1.0 on error.
    ! -----------------------------------------------------------------------
    function krylov_elapsed_time(ws) &
        bind(c, name='krylov_elapsed_time') result(ret)
      use iso_c_binding
      type(c_ptr),    value :: ws
      real(c_double)        :: ret
    end function krylov_elapsed_time

    ! -----------------------------------------------------------------------
    ! krylov_warm_start
    !
    ! Sets the initial guess for the next krylov_solve call.
    ! Pass c_loc(x0_array).
    !
    ! Returns 0 on success, nonzero on error.
    ! -----------------------------------------------------------------------
    function krylov_warm_start(ws, x0, n) &
        bind(c, name='krylov_warm_start') result(ret)
      use iso_c_binding
      type(c_ptr),    value :: ws
      type(c_ptr),    value :: x0   ! c_loc(x0_array)
      integer(c_int), value :: n
      integer(c_int)        :: ret
    end function krylov_warm_start

    ! -----------------------------------------------------------------------
    ! krylov_workspace_free
    !
    ! Releases the workspace.  The handle must not be used after this call.
    ! Returns 0 on success, 1 if handle was not found.
    ! -----------------------------------------------------------------------
    function krylov_workspace_free(ws) &
        bind(c, name='krylov_workspace_free') result(ret)
      use iso_c_binding
      type(c_ptr),   value :: ws
      integer(c_int)       :: ret
    end function krylov_workspace_free

  end interface
