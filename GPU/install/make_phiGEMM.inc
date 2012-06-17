#
# Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present package,
# or http://www.gnu.org/copyleft/gpl.txt .
#
 
# From QE, required to retrieve $(MANUAL_DFLAGS)
include ../../make.sys

PHIGEMM_TOP = .

AR          = ar
ARFLAGS     = ruv

CC          = 
FC          = 
LD          = 

CFLAGS          = 
LD_FLAGS        = 
LD_SHARED_FLAGS = 

CPP         = cpp
CPPFLAGS    =  -P -traditional

NVCC        = 
NVCC_FLAGS  = -ccbin gcc -O3 --compiler-options '-fPIC -fopenmp'

EXT_INC     =  
CUDA_LIB    = 
LD_LIB      =   -lmkl_intel_lp64  -lmkl_intel_thread -lmkl_core 

CUDA_PATH   = /usr/local/cuda/

GEMM_OPT    =  $(MANUAL_DFLAGS)
