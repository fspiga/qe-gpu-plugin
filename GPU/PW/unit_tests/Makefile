# Copyright (C) 2011-2013 Quantum ESPRESSO Foundation
#
# Makefile for PW-GPU (Filippo Spiga)

include ../../../make.sys

# location of needed modules
MODFLAGS= $(MOD_FLAG)../../../iotk/src $(MOD_FLAG)../../../PW/src $(MOD_FLAG)../../../Modules $(MOD_FLAG)../Modules $(MOD_FLAG). 

PWGPUOBJS = \
../addusdens_gpu.o \
../newq_compute_gpu.o \
../cdiaghg_gpu.o \
../vloc_psi_gpu.o \
../rdiaghg_gpu.o \
../qvan2_gpu.o 
#add_vuspsi.o \
#cegterg.o \
#regterg.o \
#data_structure.o

# Original, well tested
CUGPUOBJS = \
../addusdens_cuda.o \
../vloc_psi_k_cuda.o \
../vloc_psi_gamma_cuda.o \
../newd_cuda.o \
../qvan2_cuda.o

# Optimized, experimental (opt : -D__CUDA_NOALLOC needed)
#                         (opt2: -D__CUDA_PRELOAD -D__CUDA_NOALLOC needed)
# CUGPUOBJS = \
addusdens_cuda.o \
newd_cuda.o \
vloc_psi_k_cuda_opt2.o \
vloc_psi_gamma_cuda_opt2.o \
nls_precompute.o \
fft_plans_mgn.o

QEMODS=../../../Modules/libqemod.a ../../Modules/libqemodgpu.a

TLDEPS=bindir gpu-mods mods libs libphiGEMM libMAGMA liblapack libblas libenviron

LIBOBJS	= ../../../flib/ptools.a ../../../flib/flib.a ../../../clib/clib.a ../../../iotk/src/libiotk.a

all : tldeps just_newd just_addusdens just_vloc_psi

tldeps:
	test -n "$(TLDEPS)" && ( cd ../../.. ; $(MAKE) $(MFLAGS) $(TLDEPS) || exit 1) || :

just_newd : just_newd.o ../../../PW/src/qvan2.o ../../../PW/src/libpw.a ../libpwgpu.a $(LIBOBJS) $(QEMODS)
	$(LD) $(LDFLAGS) -o $@ \
	just_newd.o ../../../PW/src/qvan2.o $(PWOBJS) ../../../PW/src/libpw.a ../libpwgpu.a $(QEMODS) $(LIBOBJS) $(LIBS)

just_addusdens : just_addusdens.o ../../../PW/src/qvan2.o ../../../PW/src/libpw.a ../libpwgpu.a $(LIBOBJS) $(QEMODS)
	$(LD) $(LDFLAGS) -o $@ \
	just_addusdens.o ../../../PW/src/qvan2.o $(PWOBJS) ../../../PW/src/libpw.a ../libpwgpu.a $(QEMODS) $(LIBOBJS) $(LIBS)

just_vloc_psi : just_vloc_psi.o ../../../PW/src/qvan2.o ../../../PW/src/libpw.a ../libpwgpu.a $(LIBOBJS) $(QEMODS)
	$(LD) $(LDFLAGS) -o $@ \
	just_vloc_psi.o ../../../PW/src/qvan2.o $(PWOBJS) ../../../PW/src/libpw.a ../libpwgpu.a $(QEMODS) $(LIBOBJS) $(LIBS)

clean :
	- /bin/rm -f *.x *.o *.a *~ *.F90 *.d *.mod *.i *.L *.lst

#include make.depend

# DO NOT DELETE
