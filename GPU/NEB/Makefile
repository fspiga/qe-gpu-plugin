# Makefile for NEB

include ../../make.sys

# location of needed modules
MODFLAGS= $(MOD_FLAG)../../iotk/src \
	  $(MOD_FLAG)../../Modules $(MOD_FLAG)../Modules \
	  $(MOD_FLAG)../../NEB/src $(MOD_FLAG)../../PW/src $(MOD_FLAG).

#location of needed libraries
LIBOBJS= ../../iotk/src/libiotk.a ../../flib/flib.a \
         ../../clib/clib.a ../../flib/ptools.a

NEBGPUOBJS = \
neb.o 

NEBGPULIBS = \
../../NEB/src/compute_scf.o \
../../NEB/src/engine_to_path_pos.o \
../../NEB/src/engine_to_path_alat.o \
../../NEB/src/engine_to_path_nat.o \
../../NEB/src/engine_to_path_fix_atom_pos.o \
../../NEB/src/input.o \
../../NEB/src/path_base.o \
../../NEB/src/path_formats.o \
../../NEB/src/path_gen_inputs.o \
../../NEB/src/path_input_parameters_module.o \
../../NEB/src/path_io_routines.o \
../../NEB/src/path_io_tools.o \
../../NEB/src/path_io_units_module.o \
../../NEB/src/path_opt_routines.o \
../../NEB/src/path_reparametrisation.o \
../../NEB/src/path_read_cards_module.o \
../../NEB/src/path_read_namelists_module.o \
../../NEB/src/path_to_engine_fix_atom_pos.o \
../../NEB/src/path_variables.o \
../../NEB/src/set_defaults.o \
../../NEB/src/stop_run_path.o 

QEGPUMODS=../Modules/libqemodgpu.a
PWGPUOBJS= ../PW/libpwgpu.a

TLDEPS=bindir gpu-mods mods libs libphiGEMM libMAGMA liblapack libblas pw-gpu

all : tldeps neb-gpu.x

neb-gpu.x : $(NEBGPUOBJS) libnebgpu.a $(LIBOBJS) $(PWGPUOBJS) $(QEGPUMODS)
	$(LD) $(LDFLAGS) -o $@ \
	$(NEBGPUOBJS) libnebgpu.a $(PWGPUOBJS) $(QEGPUMODS) $(LIBOBJS) $(LIBS)
	- ( cd ../../bin; ln -fs ../GPU/NEB/$@ . )


libnebgpu.a : $(NEBGPULIBS)
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@

tldeps:
	test -n "$(TLDEPS)" && ( cd ../.. ; $(MAKE) $(MFLAGS) $(TLDEPS) || exit 1) || :

clean :
	- /bin/rm -f *.x *.o *.a *~ *.F90 *.d *.mod *.i *.L
	- /bin/rm -f ../../bin/neb-gpu.x

#include make.depend
