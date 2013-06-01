# Makefile for CP/FPMD

include	../../make.sys

# location of needed modules
MODFLAGS= $(MOD_FLAG)../../iotk/src $(MOD_FLAG)../../Modules \
	$(MOD_FLAG)../../CPV/src $(MOD_FLAG).

QEMODS=../../Modules/libqemod.a ../Modules/libqemodgpu.a

TLDEPS= bindir mods libs libiotk

LIBOBJS        = ../../flib/ptools.a ../../flib/flib.a ../../clib/clib.a ../../iotk/src/libiotk.a

all : cp-gpu

cp-gpu : tldeps cp-gpu.x 

cp-gpu.x : ../../CPV/src/cprstart.o $(QEMODS) $(LIBOBJS)
	if [ ! -f ./cprstart.o ]; then cp ../../CPV/src/cprstart.o . ; fi
	$(LD) $(LDFLAGS) -o $@ cprstart.o \
		../../CPV/src/libcp.a $(QEMODS) $(LIBOBJS) $(LIBS)
	- ( cd ../../bin ; ln -fs ../GPU/CPV/$@ . )

#libcpgpu.a : $(FOGPUBJS) $(LOGPUBJS) 
#	 $(AR) $(ARFLAGS) $@ $?
#	 $(RANLIB) $@

#cp_version.o : cpver.h

#cpver.h :
#	echo "CHARACTER(LEN=70), PARAMETER :: version_date = '"`date`"'" \
#		> cpver.h

#cppp-gpu.x : ../../CPV/src/fpmdpp.o $(QEGPUMODS) $(LIBOBJS)
#	$(LD) $(LDFLAGS) -o $@ fpmdpp.o $(QEGPUMODS) $(LIBOBJS) $(LIBS)
#	- (cd ../../bin ; ln -fs ../GPU/CPV/$@ . )

#wfdd-gpu.x : ../../CPV/src/wfdd.o
#	$(LD) $(LDFLAGS) -o $@ wfdd.o $(QEGPUMODS) $(LIBOBJS) $(LIBS)
#	- ( cd ../../bin ; ln -fs ../GPU/CPV/$@ . )


tldeps:
	test -n "$(TLDEPS)" && ( cd ../.. ; $(MAKE) $(MFLAGS) $(TLDEPS) || exit 1) || :

clean :
	- /bin/rm -f *.x *.o *.mod cpver.h *.i core* *.F90 fort* \
		*.cpp *.d *.L *.a *.s
	-/bin/rm -rf ../../bin/cp-gpu.x 

#include make.depend

# DO NOT DELETE
	