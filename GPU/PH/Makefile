# Makefile for phonon (PH)

include ../../make.sys

# location of needed modules
MODFLAGS= $(MOD_FLAG)../../iotk/src $(MOD_FLAG)../../Modules \
          $(MOD_FLAG)../GPU/PW $(MOD_FLAG)../GPU/Modules \
          $(MOD_FLAG)../../PW/src $(MOD_FLAG)../../PHonon/PH \
          $(MOD_FLAG). 

PWMODS = ../../PW/src/libpw.a ../PW/libpwgpu.a
QEMODS = ../../Modules/libqemod.a ../Modules/libqemodgpu.a

LIBOBJS        = ../../flib/ptools.a ../../flib/flib.a ../../clib/clib.a ../../iotk/src/libiotk.a

TLDEPS= bindir mods libs pw

all : tldeps ph-gpu.x

ph-gpu.x : ../../PHonon/PH/phonon.o ../../PHonon/PH/libph.a $(PWMODS) $(LIBOBJS)
	if [ ! -f ./phonon.o ]; then cp ../../PHonon/PH/phonon.o . ; fi
	$(LD) $(LDFLAGS) -o $@ ../../PHonon/PH/phonon.o ../../PHonon/PH/libph.a \
		 $(PWMODS) $(QEMODS) $(LIBOBJS) $(LIBS)
	- ( cd ../../bin ; ln -fs ../GPU/PH/$@ . )

tldeps:
	test -n "$(TLDEPS)" && ( cd ../.. ; $(MAKE) $(MFLAGS) $(TLDEPS) || exit 1) || :
clean :
	- /bin/rm -f *.x *.o *.a *~ *.F90 *.d *.mod *.i *.L *.lst
	- /bin/rm -rf ../../bin/ph-gpu.x ../../bin/dynmat-gpu.x ../../bin/matdyn-gpu.x \
	../../bin/q2r-gpu.x ../../bin/lambda-gpu.x ../../bin/fqha-gpu.x

#include make.depend

# DO NOT DELETE
