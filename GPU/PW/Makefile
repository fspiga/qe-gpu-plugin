# Copyright (C) 2011-2014 Quantum ESPRESSO Foundation
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Filippo Spiga (filippo.spiga@quantum-espresso.org)

include ../../make.sys

# **EDIT ME** (opt: ADDUSDENS_version=2)
ADDUSDENS_version=1
NEWD_version=1
QVAN2_version=1
VLOCPSI_GAMMA_version=1
VLOCPSI_K_version=1
FFT_PLAN_MNG_version=1

# location of needed modules
MODFLAGS= $(MOD_FLAG)../../iotk/src $(MOD_FLAG)../../PW/src $(MOD_FLAG)../../Modules $(MOD_FLAG)../Modules $(MOD_FLAG). 

PWGPUOBJS = \
addusdens_gpu.o \
newq_compute_gpu.o \
cdiaghg_gpu.o \
vloc_psi_gpu.o \
rdiaghg_gpu.o \
qvan2_gpu.o \
add_vuspsi.o \
cegterg.o \
regterg.o
#data_structure_gpu.o

CUGPUOBJS = \
addusdens_cuda_v$(ADDUSDENS_version).o \
vloc_psi_k_cuda_v$(VLOCPSI_K_version).o \
vloc_psi_gamma_cuda_v$(VLOCPSI_GAMMA_version).o \
newd_cuda_v$(NEWD_version).o \
qvan2_cuda_v$(QVAN2_version).o \
fft_plans_mgn_v$(FFT_PLAN_MNG_version).o

# Optimized, experimental (opt : -D__CUDA_NOALLOC needed)
#                         (opt2: -D__CUDA_PRELOAD -D__CUDA_NOALLOC needed)
# CUGPUOBJS = \
addusdens_cuda.o \
newd_cuda.o \
vloc_psi_k_cuda_opt2.o \
vloc_psi_gamma_cuda_opt2.o \
nls_precompute.o \


QEMODS=../../Modules/libqemod.a ../Modules/libqemodgpu.a

TLDEPS=bindir gpu-mods mods libs libphiGEMM libMAGMA liblapack libblas libenviron

LIBOBJS	= ../../flib/ptools.a ../../flib/flib.a ../../clib/clib.a ../../iotk/src/libiotk.a

all : tldeps pw-gpu.x

pw-gpu.x : ../../PW/src/pwscf.o libpwgpu.a $(LIBOBJS) $(QEMODS)
	if [ ! -f ./pwscf.o ]; then cp ../../PW/src/pwscf.o . ; fi
	$(LD) $(LDFLAGS) -o $@ \
	 pwscf.o $(PWOBJS) ../../PW/src/libpw.a libpwgpu.a $(QEMODS) $(LIBOBJS) $(LIBS)
	- ( cd ../../bin; ln -fs ../GPU/PW/$@ . )

# This does not work unless you are using espresso from SVN. Ignore
# of comment this to have a clean build (NdFilippo)
manypw-gpu.x : ../../PW/src/manypw.o libpwgpu.a $(LIBOBJS) $(QEMODS)
	if [ ! -f ./manypw.o ]; then cp ../../PW/src/manypw.o . ; fi
	$(LD) $(LDFLAGS) -o $@ \
	 manypw.o $(PWOBJS) ../../PW/src/libpw.a libpwgpu.a $(QEMODS) $(LIBOBJS) $(LIBS)
	- ( cd ../../bin; ln -fs ../GPU/PW/$@ . )

libpwgpu.a : $(PWGPUOBJS) $(CUGPUOBJS)
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@

tldeps:
	test -n "$(TLDEPS)" && ( cd ../.. ; $(MAKE) $(MFLAGS) $(TLDEPS) || exit 1) || :

clean :
	- /bin/rm -f *.x *.o *.a *~ *.F90 *.d *.mod *.i *.L *.lst

#include make.depend

# DO NOT DELETE
