# install/make_lapack.inc.  Generated from make_lapack.inc.in by configure.
# -*- Makefile -*-
####################################################################
#  LAPACK make include file.                                       #
#  LAPACK, Version 3.2                                             #
#  November 2008                                                   #
####################################################################
#
# See the INSTALL/ directory for more examples.
#
SHELL = /bin/sh
#
#  The machine (platform) identifier to append to the library names
#
PLAT =
#  
#  Modify the FORTRAN and OPTS definitions to refer to the
#  compiler and desired compiler options for your machine.  NOOPT
#  refers to the compiler options desired when NO OPTIMIZATION is
#  selected.  Define LOADER and LOADOPTS to refer to the loader
#  and desired load options for your machine.
#
FORTRAN  = ifort
OPTS     = -O2 -assume byterecl -g -traceback -par-report0 -vec-report0 -openmp
DRVOPTS  = -O2 -assume byterecl -g -traceback -par-report0 -vec-report0 -openmp
NOOPT    = -O0 -assume byterecl -g -traceback
LOADER   = mpif90
LOADOPTS = -static-intel  -openmp
#
# Timer for the SECOND and DSECND routines
#
# Default : SECOND and DSECND will use a call to the EXTERNAL FUNCTION ETIME
# TIMER    = EXT_ETIME
# For RS6K : SECOND and DSECND will use a call to the EXTERNAL FUNCTION ETIME_
# TIMER    = EXT_ETIME_
# For gfortran compiler: SECOND and DSECND will use a call to the INTERNAL FUNCTION ETIME
# TIMER    = INT_ETIME
# If your Fortran compiler does not provide etime (like Nag Fortran Compiler, etc...)
# SECOND and DSECND will use a call to the Fortran standard INTERNAL FUNCTION CPU_TIME 
TIMER    = INT_CPU_TIME
# If neither of this works...you can use the NONE value... In that case, SECOND and DSECND will always return 0
# TIMER     = NONE
#
#  The archiver and the flag(s) to use when building archive (library)
#  If you system has no ranlib, set RANLIB = echo.
#
ARCH     = ar
ARCHFLAGS= ruv
RANLIB   = ranlib
#
#  The location of BLAS library for linking the testing programs.
#  The target's machine-specific, optimized BLAS library should be
#  used whenever possible.
#
BLASLIB      =   -lmkl_intel_lp64  -lmkl_intel_thread -lmkl_core
#
#  Location of the extended-precision BLAS (XBLAS) Fortran library
#  used for building and testing extended-precision routines.  The
#  relevant routines will be compiled and XBLAS will be linked only if
#  USEXBLAS is defined.
#
# USEXBLAS    = Yes
XBLASLIB     =
# XBLASLIB    = -lxblas
#
#  Names of generated libraries.
#
LAPACKLIB    = lapack$(PLAT).a
TMGLIB       = tmglib$(PLAT).a
EIGSRCLIB    = eigsrc$(PLAT).a
LINSRCLIB    = linsrc$(PLAT).a
