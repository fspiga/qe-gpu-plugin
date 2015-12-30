# Makefile for NEB

include ../../make.sys

# location of needed modules
MODFLAGS= $(MOD_FLAG)../../iotk/src \
	  $(MOD_FLAG)../../Modules $(MOD_FLAG)../../FFTXlib $(MOD_FLAG)../Modules \
	  $(MOD_FLAG)../../NEB/src $(MOD_FLAG)../../PW/src $(MOD_FLAG).

#location of needed libraries
LIBOBJS= ../../iotk/src/libiotk.a ../../flib/flib.a \
         ../../clib/clib.a ../../flib/ptools.a

# Still empty...
#NEBGPUOBJS =  

QEGPUMODS=../../Modules/libqemod.a ../../FFTXlib/libqefft.a ../Modules/libqemodgpu.a
PWMODS=../../PW/src/libpw.a ../PW/libpwgpu.a

TLDEPS=bindir gpu-mods mods libs libphiGEMM libMAGMA liblapack libblas pw-gpu

all : tldeps neb-gpu.x

neb-gpu.x : ../../NEB/src/neb.o ../../NEB/src/libneb.a $(LIBOBJS) $(PWMODS) $(QEGPUMODS)
	if [ ! -f ./neb.o ]; then cp ../../NEB/src/neb.o . ; fi
	$(LD) $(LDFLAGS) -o $@ \
	neb.o ../../NEB/src/libneb.a  $(PWMODS) $(QEGPUMODS) $(LIBOBJS) $(LIBS)
	- ( cd ../../bin; ln -fs ../GPU/NEB/$@ . )

#libnebgpu.a : $(NEBGPULIBS)
#	$(AR) $(ARFLAGS) $@ $?
#	$(RANLIB) $@

tldeps:
	test -n "$(TLDEPS)" && ( cd ../.. ; $(MAKE) $(MFLAGS) $(TLDEPS) || exit 1) || :

clean :
	- /bin/rm -f *.x *.o *.a *~ *.F90 *.d *.mod *.i *.L
	- /bin/rm -f ../../bin/neb-gpu.x

#include make.depend
