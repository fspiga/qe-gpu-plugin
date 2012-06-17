#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of	Colorado, Denver
#      November 2011
#//////////////////////////////////////////////////////////////////////////////

#
# GPU_TARGET specifies for which GPU you want to compile MAGMA
#      0: Tesla family
#      1: Fermi Family
#
GPU_TARGET = 

CC        = gcc
NVCC      = 
FORT      = gfortran

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib

OPTS      = -O3 -DADD_
FOPTS     = -O3 -DADD_ -x f95-cpp-input
NVOPTS    = --compiler-options -fno-strict-aliasing -DUNIX -O3 -DADD_
LDOPTS    = -fPIC -Xlinker -zmuldefs

LIB      = 

CUDADIR   = /usr/local/cuda/

LIBDIR    = 
INC       = 

#LIBMAGMA     = ../lib/libmagma.a
#LIBMAGMABLAS = ../lib/libmagmablas.a
