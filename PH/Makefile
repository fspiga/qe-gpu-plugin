# Makefile for phonon (PH)

include ../../make.sys

# location of needed modules
MODFLAGS= $(MOD_FLAG)../../iotk/src $(MOD_FLAG)../../Modules $(MOD_FLAG)../../FFTXlib \
          $(MOD_FLAG)../GPU/PW $(MOD_FLAG)../GPU/Modules \
          $(MOD_FLAG)../../PW/src $(MOD_FLAG)../../PHonon/PH \
          $(MOD_FLAG). 

PWMODS = ../../LR_Modules/liblrmod.a ../../PW/src/libpw.a ../PW/libpwgpu.a
QEMODS = ../../Modules/libqemod.a ../../FFTXlib/libqefft.a ../../LAXlib/libqela.a ../Modules/libqemodgpu.a

LIBOBJS        = ../../clib/clib.a ../../iotk/src/libiotk.a

TLDEPS= bindir mods libs pw-gpu


PHGPUOBJS = \
h_psiq_gpu.o

CUGPUOBJS = \
h_psiq_k_cuda.o

all : tldeps ph-gpu.x

ph-gpu.x : phonon.o ../../PHonon/PH/libph.a libphgpu.a $(PWMODS) $(LIBOBJS)
	$(LD) $(LDFLAGS) -o $@ phonon.o ../../PHonon/PH/libph.a libphgpu.a \
		 $(PWMODS) $(QEMODS) $(LIBOBJS) $(LIBS)
	- ( cd ../../bin ; ln -fs ../GPU/PH/$@ . )

libphgpu.a : $(PHGPUOBJS) $(CUGPUOBJS)
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@

tldeps:
	test -n "$(TLDEPS)" && ( cd ../.. ; $(MAKE) $(MFLAGS) $(TLDEPS) || exit 1) || :
clean :
	- /bin/rm -f *.x *.o *.a *~ *.F90 *.d *.mod *.i *.L *.lst
	- /bin/rm -rf ../../bin/ph-gpu.x ../../bin/dynmat-gpu.x ../../bin/matdyn-gpu.x \
	../../bin/q2r-gpu.x ../../bin/lambda-gpu.x ../../bin/fqha-gpu.x

#include make.depend

# DO NOT DELETE
