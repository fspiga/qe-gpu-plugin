# Makefile for CP/FPMD

include	../../make.sys

# location of needed modules
MODFLAGS= $(MOD_FLAG)../../iotk/src $(MOD_FLAG)../../Modules \
	$(MOD_FLAG)../../CPV/src $(MOD_FLAG).

FOGPUBJS = \
../../CPV/src/berryion.o \
../../CPV/src/berry_phase.o \
../../CPV/src/bforceion.o \
../../CPV/src/cell_nose.o \
../../CPV/src/cg.o \
../../CPV/src/cg_sub.o \
../../CPV/src/cglib.o \
../../CPV/src/chargedensity.o \
../../CPV/src/cp_autopilot.o \
../../CPV/src/cp_emass.o \
../../CPV/src/cp_interfaces.o \
../../CPV/src/cp_restart.o \
../../CPV/src/cplib.o \
../../CPV/src/cpr_mod.o \
../../CPV/src/cpr.o \
../../CPV/src/dealloc.o \
../../CPV/src/dforceb.o \
../../CPV/src/efermi.o \
../../CPV/src/efield.o \
../../CPV/src/eigs0.o \
../../CPV/src/electrons.o \
../../CPV/src/electrons_nose.o \
../../CPV/src/energies.o \
../../CPV/src/ensemble_dft.o \
../../CPV/src/exch_corr.o \
../../CPV/src/fft.o \
../../CPV/src/forces.o \
../../CPV/src/fromscra.o \
../../CPV/src/gradrho.o \
../../CPV/src/gram.o \
../../CPV/src/gtable.o \
../../CPV/src/gvecw.o\
../../CPV/src/init.o \
../../CPV/src/init_run.o \
../../CPV/src/inner_loop_cold.o \
../../CPV/src/input.o \
../../CPV/src/ions_nose.o \
../../CPV/src/ions_positions.o \
../../CPV/src/ksstates.o \
../../CPV/src/ldaU.o \
../../CPV/src/ldaUpen.o \
../../CPV/src/mainvar.o \
../../CPV/src/cpr_loop.o \
../../CPV/src/cplib_meta.o \
../../CPV/src/metaxc.o \
../../CPV/src/modules.o \
../../CPV/src/move_electrons.o \
../../CPV/src/newd.o \
../../CPV/src/nl_base.o \
../../CPV/src/nlcc.o \
../../CPV/src/ortho_base.o \
../../CPV/src/ortho.o \
../../CPV/src/phasefactor.o \
../../CPV/src/plugin_initialization.o \
../../CPV/src/plugin_forces.o \
../../CPV/src/polarization.o \
../../CPV/src/potentials.o \
../../CPV/src/pres_ai_mod.o \
../../CPV/src/print_out.o \
../../CPV/src/printout_base.o \
../../CPV/src/problem_size.o \
../../CPV/src/pseudo_base.o \
../../CPV/src/pseudopot.o \
../../CPV/src/pseudopot_sub.o \
../../CPV/src/qmatrixd.o \
../../CPV/src/qqberry.o \
../../CPV/src/restart.o \
../../CPV/src/restart_sub.o \
../../CPV/src/runcp.o \
../../CPV/src/smallbox.o \
../../CPV/src/smallbox_grid.o \
../../CPV/src/smallbox_gvec.o \
../../CPV/src/smallbox_subs.o \
../../CPV/src/smallbox_lib.o \
../../CPV/src/spline.o \
../../CPV/src/spinsq.o \
../../CPV/src/stop_run.o \
../../CPV/src/stress.o \
../../CPV/src/vol_clu.o \
../../CPV/src/vofrho.o \
../../CPV/src/wannier_base.o \
../../CPV/src/wannier.o \
../../CPV/src/wave.o \
../../CPV/src/wave_types.o \
../../CPV/src/wf.o \
../../CPV/src/makov_payne.o

LOGPUBJS = \
../../CPV/src/entropy.o 

QEGPUMODS=../Modules/libqemodgpu.a

TLDEPS= bindir mods libs libiotk

LIBOBJS        = ../../flib/ptools.a ../../flib/flib.a ../../clib/clib.a ../../iotk/src/libiotk.a

all : cp-gpu

cp-gpu : tldeps libcpgpu.a cp-gpu.x cppp-gpu.x wfdd-gpu.x

cp-gpu.x : cprstart.o libcpgpu.a $(QEMODS) $(LIBOBJS)
	$(LD) $(LDFLAGS) -o cp.x cprstart.o \
		libcpgpu.a $(QEGPUMODS) $(LIBOBJS) $(LIBS)
	- ( cd ../../bin ; ln -fs ../GPU/CPV/$@ . )

libcpgpu.a : $(FOGPUBJS) $(LOGPUBJS) 
	 $(AR) $(ARFLAGS) $@ $?
	 $(RANLIB) $@

#cp_version.o : cpver.h

#cpver.h :
#	echo "CHARACTER(LEN=70), PARAMETER :: version_date = '"`date`"'" \
#		> cpver.h

cppp-gpu.x : fpmdpp.o $(QEGPUMODS) $(LIBOBJS)
	$(LD) $(LDFLAGS) -o $@ fpmdpp.o $(QEGPUMODS) $(LIBOBJS) $(LIBS)
	- (cd ../../bin ; ln -fs ../GPU/CPV/$@ . )

wfdd-gpu.x : wfdd.o
	$(LD) $(LDFLAGS) -o $@ wfdd.o $(QEGPUMODS) $(LIBOBJS) $(LIBS)
	- ( cd ../../bin ; ln -fs ../GPU/CPV/$@ . )


tldeps:
	test -n "$(TLDEPS)" && ( cd ../.. ; $(MAKE) $(MFLAGS) $(TLDEPS) || exit 1) || :

clean :
	- /bin/rm -f *.x *.o *.mod cpver.h *.i core* *.F90 fort* \
		*.cpp *.d *.L *.a *.s
	-/bin/rm -rf ../../bin/cppp-gpu.x ../../bin/cp-gpu.x ../../bin/wfdd-gpu.x

#include make.depend
