# Makefile for PW
#Modified for v4.0 by obm

include ../../make.sys
IFLAGS         = -I../../include
LIBOBJS        = ../../flib/ptools.a ../../flib/flib.a ../../clib/clib.a ../../iotk/src/libiotk.a 


# location of needed modules
MODFLAGS= $(MOD_FLAG)../../iotk/src $(MOD_FLAG)../../Modules \
          $(MOD_FLAG)../../TDDFPT/src \
          $(MOD_FLAG). $(MOD_FLAG)../../PW/src \
          $(MOD_FLAG)../../PHonon/PH $(MOD_FLAG)../../GPU/PW \
          $(MOD_FLAG)../../GPU/Modules

QEGPUMODS = ../Modules/libqemodgpu.a
PWGPUOBJS = ../PW/libpwgpu.a
PHOBJS=../../PHonon/PH/libph.a


LRGPUOBJS= \
../../TDDFPT/src/lr_variables.o \
../../TDDFPT/src/lr_charg_resp.o \
../../TDDFPT/src/bcast_lr_input.o \
../../TDDFPT/src/lr_readin.o \
../../TDDFPT/src/lr_alloc_init.o \
../../TDDFPT/src/lr_calc_dens.o \
../../TDDFPT/src/lr_dot.o \
../../TDDFPT/src/lr_dealloc.o \
../../TDDFPT/src/lr_ortho.o \
../../TDDFPT/src/lr_read_wf.o \
../../TDDFPT/src/lr_normalise.o \
../../TDDFPT/src/lr_lanczos.o \
../../TDDFPT/src/lr_apply_liouvillian.o \
../../TDDFPT/src/lr_dv_setup.o \
../../TDDFPT/src/lr_solve_e.o \
../../TDDFPT/src/lr_dvpsi_e.o \
../../TDDFPT/src/lr_sm1_psi.o \
../../TDDFPT/src/stop_lr.o \
../../TDDFPT/src/lr_read_d0psi.o \
../../TDDFPT/src/lr_restart.o \
../../TDDFPT/src/lr_write_restart.o \
../../TDDFPT/src/print_clock_lr.o \
../../TDDFPT/src/sd0psi.o \
../../TDDFPT/src/lr_set_boxes_density.o \
../../TDDFPT/src/lr_init_nfo.o 

TLDEPS=bindir gpu-mods mods libs libphiGEMM libMAGMA pw-gpu ph
 
all :  tldeps turbo_lanczos-gpu.x

libtddfptgpu.a : $(LRGPUOBJS)
	$(AR) $(ARFLAGS) $@ $? $(PWOBJS)
	$(RANLIB) $@
	
turbo_lanczos-gpu.x : lr_main.o libtddfptgpu.a $(PHOBJS) $(PWGPUOBJS) $(QEGPUMODS) $(LIBOBJS)
	$(LD) $(LDFLAGS) -o $@ \
		lr_main.o libtddfptgpu.a $(PHOBJS) $(PWGPUOBJS) $(QEGPUMODS) $(LIBOBJS) $(LIBS)
	- ( cd ../../bin ; ln -fs ../GPU/TDDFPT/$@ . )
	- if [ -d ../bin ] ; then  ( cd ../bin ; ln -fs ../src/$@ . ); fi

dene-gpu.x : obm_tests.o 
	$(MPIF90) $(LDFLAGS) -o dene.x \
        $(PWOBJS) $(PHOBJS) obm_tests.o $(QEMODS) $(LIBOBJS) $(LIBS)

tldeps:
	test -n "$(TLDEPS)" && ( cd ../.. ; $(MAKE) $(MFLAGS) $(TLDEPS) || exit 1) || :


clean :
	- /bin/rm -f ../bin/turbo_lanczos.x *.x *.o *~ *.d *.mod *.i work.pc *.F90 *.a

# DO NOT DELETE
