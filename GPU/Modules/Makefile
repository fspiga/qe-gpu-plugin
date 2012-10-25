# Makefile for Modules

include ../../make.sys

# location of needed modules
MODFLAGS= $(MOD_FLAG)../iotk/src $(MOD_FLAG)../../Modules $(MOD_FLAG)/../../ELPA/ $(MOD_FLAG).

GPUMODULES = \
mp.o

MODULES = \
../../Modules/atom.o \
../../Modules/autopilot.o \
../../Modules/basic_algebra_routines.o \
../../Modules/becmod.o \
../../Modules/bfgs_module.o \
../../Modules/cell_base.o  \
../../Modules/check_stop.o  \
../../Modules/clocks.o \
../../Modules/compute_dipole.o \
../../Modules/constants.o \
../../Modules/constraints_module.o \
../../Modules/control_flags.o \
../../Modules/coulomb_vcut.o \
../../Modules/descriptors.o \
../../Modules/dspev_drv.o \
../../Modules/electrons_base.o \
../../Modules/environment.o \
../../Modules/error_handler.o \
../../Modules/fd_gradient.o \
../../Modules/fft_base.o \
../../Modules/fft_custom.o \
../../Modules/fft_interfaces.o \
../../Modules/fft_parallel.o \
../../Modules/fft_scalar.o \
../../Modules/fft_types.o \
../../Modules/funct.o \
../../Modules/generate_function.o \
../../Modules/griddim.o \
../../Modules/image_io_routines.o \
../../Modules/input_parameters.o \
../../Modules/io_files.o \
../../Modules/io_global.o  \
../../Modules/ions_base.o \
../../Modules/kernel_table.o \
../../Modules/kind.o \
../../Modules/mm_dispersion.o \
../../Modules/mp_base.o \
../../Modules/mp_global.o \
../../Modules/mp_image_global_module.o \
../../Modules/mp_wave.o \
../../Modules/noncol.o \
../../Modules/open_close_input_file.o \
../../Modules/parallel_include.o \
../../Modules/parameters.o \
../../Modules/parser.o \
../../Modules/paw_variables.o \
../../Modules/plugin_flags.o \
../../Modules/plugin_arguments.o \
../../Modules/pseudo_types.o \
../../Modules/ptoolkit.o \
../../Modules/radial_grids.o \
../../Modules/random_numbers.o \
../../Modules/read_input.o \
../../Modules/read_namelists.o \
../../Modules/read_ncpp.o \
../../Modules/read_pseudo.o \
../../Modules/read_upf_v1.o \
../../Modules/read_upf_v2.o \
../../Modules/read_uspp.o \
../../Modules/read_xml.o \
../../Modules/read_xml_cards.o \
../../Modules/read_xml_fields.o \
../../Modules/recvec.o \
../../Modules/recvec_subs.o \
../../Modules/run_info.o \
../../Modules/set_signal.o \
../../Modules/sic.o \
../../Modules/splinelib.o \
../../Modules/stick_base.o \
../../Modules/stick_set.o \
../../Modules/timestep.o\
../../Modules/version.o \
../../Modules/upf.o \
../../Modules/upf_to_internal.o \
../../Modules/uspp.o \
../../Modules/wave_base.o \
../../Modules/wavefunctions.o \
../../Modules/write_upf_v2.o \
../../Modules/xc_vdW_DF.o \
../../Modules/xml_input.o \
../../Modules/xml_io_base.o \
../../Modules/zhpev_drv.o \
../../Modules/wannier_new.o \
../../Modules/wrappers.o\
../../Modules/ws_base.o \
../../Modules/read_cards.o

CUGPUOBJS = \
cuda_mem_alloc.o \
cuda_env_para.o \
cuda_init.o

all : libqemodgpu.a 

#version.f90: version.f90.in
#	- ( if test -x ../install/update_version ; then \
#                       ../install/update_version; fi )

libqemodgpu.a: $(GPUMODULES) $(CUGPUOBJS)
	$(AR) $(ARFLAGS) $@ $? $(MODULES)     
	$(RANLIB) $@    

clean :
	- /bin/rm -f *.o *.a *.d *.i *~ *.F90 *.mod *.L

#include make.depend
