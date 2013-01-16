# Makefile for phonon (PH)

include ../../make.sys

# location of needed modules
MODFLAGS= $(MOD_FLAG)../../iotk/src $(MOD_FLAG)../../Modules \
          $(MOD_FLAG)../GPU/PW $(MOD_FLAG)../GPU/Modules \
          $(MOD_FLAG)../../PW/src $(MOD_FLAG)../../PHonon/PH \
          $(MOD_FLAG). 

PHOBJS = \
../../PHonon/PH/acfdtest.o \
../../PHonon/PH/add_dkmds.o \
../../PHonon/PH/add_for_charges.o \
../../PHonon/PH/add_zstar_ue.o \
../../PHonon/PH/add_zstar_ue_us.o \
../../PHonon/PH/addcore.o \
../../PHonon/PH/adddvepsi_us.o \
../../PHonon/PH/adddvscf.o \
../../PHonon/PH/addnlcc.o \
../../PHonon/PH/addnlcc_zstar_eu_us.o \
../../PHonon/PH/addusdbec.o \
../../PHonon/PH/addusdbec_nc.o \
../../PHonon/PH/addusddens.o \
../../PHonon/PH/addusddense.o \
../../PHonon/PH/addusdynmat.o \
../../PHonon/PH/addusldos.o \
../../PHonon/PH/allocate_part.o \
../../PHonon/PH/allocate_pert.o \
../../PHonon/PH/allocate_phq.o \
../../PHonon/PH/apply_dpot.o \
../../PHonon/PH/bcast_ph_input.o \
../../PHonon/PH/cch_psi_all.o \
../../PHonon/PH/cg_psi.o \
../../PHonon/PH/ccg_psi.o \
../../PHonon/PH/cgsolve_all.o \
../../PHonon/PH/cgsolve_all_imfreq.o \
../../PHonon/PH/ch_psi_all.o \
../../PHonon/PH/check_initial_status.o \
../../PHonon/PH/check_q_points_sym.o \
../../PHonon/PH/check_restart_recover.o \
../../PHonon/PH/clean_pw_ph.o \
../../PHonon/PH/clinear.o \
../../PHonon/PH/close_phq.o \
../../PHonon/PH/commutator_Hx_psi.o \
../../PHonon/PH/compute_alphasum.o \
../../PHonon/PH/compute_becalp.o \
../../PHonon/PH/compute_becsum_ph.o \
../../PHonon/PH/compute_drhous.o \
../../PHonon/PH/compute_drhous_nc.o \
../../PHonon/PH/compute_dvloc.o \
../../PHonon/PH/compute_nldyn.o \
../../PHonon/PH/compute_vsgga.o \
../../PHonon/PH/compute_weight.o \
../../PHonon/PH/deallocate_part.o \
../../PHonon/PH/deallocate_phq.o \
../../PHonon/PH/d2ionq.o \
../../PHonon/PH/davcio_drho.o \
../../PHonon/PH/dfile_autoname.o \
../../PHonon/PH/dfile_star.o \
../../PHonon/PH/dgradcorr.o \
../../PHonon/PH/dielec.o \
../../PHonon/PH/drho.o \
../../PHonon/PH/drhodv.o \
../../PHonon/PH/drhodvloc.o \
../../PHonon/PH/drhodvnl.o \
../../PHonon/PH/drhodvus.o \
../../PHonon/PH/dv_of_drho.o \
../../PHonon/PH/dvanqq.o \
../../PHonon/PH/dvkb3.o \
../../PHonon/PH/dvpsi_e.o \
../../PHonon/PH/dvqpsi_us.o \
../../PHonon/PH/dvqpsi_us_only.o \
../../PHonon/PH/dyndia.o \
../../PHonon/PH/dynmat0.o \
../../PHonon/PH/dynmat_us.o \
../../PHonon/PH/dynmatcc.o \
../../PHonon/PH/dynmatrix.o \
../../PHonon/PH/ef_shift.o \
../../PHonon/PH/ep_matrix_element_wannier.o \
../../PHonon/PH/elph.o \
../../PHonon/PH/el_ph_collect.o \
../../PHonon/PH/elphon.o \
../../PHonon/PH/find_equiv_sites.o \
../../PHonon/PH/find_mode_sym.o \
../../PHonon/PH/find_irrep.o \
../../PHonon/PH/generate_dynamical_matrix_c.o \
../../PHonon/PH/generate_effective_charges_c.o \
../../PHonon/PH/gmressolve_all.o \
../../PHonon/PH/h_psiq.o \
../../PHonon/PH/incdrhoscf.o \
../../PHonon/PH/incdrhoscf_nc.o \
../../PHonon/PH/incdrhous.o \
../../PHonon/PH/incdrhous_nc.o \
../../PHonon/PH/initialize_ph.o \
../../PHonon/PH/init_representations.o \
../../PHonon/PH/io_dyn_mat.o \
../../PHonon/PH/io_dyn_mat_old.o \
../../PHonon/PH/io_pattern.o \
../../PHonon/PH/localdos.o \
../../PHonon/PH/mix_pot.o \
../../PHonon/PH/mode_group.o \
../../PHonon/PH/newdq.o \
../../PHonon/PH/obsolete.o \
../../PHonon/PH/openfilq.o \
../../PHonon/PH/orthogonalize.o \
../../PHonon/PH/phcom.o \
../../PHonon/PH/ph_restart.o \
../../PHonon/PH/phescf.o \
../../PHonon/PH/phq_init.o \
../../PHonon/PH/phq_readin.o \
../../PHonon/PH/phq_recover.o \
../../PHonon/PH/phq_setup.o \
../../PHonon/PH/phq_summary.o \
../../PHonon/PH/phqscf.o \
../../PHonon/PH/polariz.o \
../../PHonon/PH/print_clock_ph.o \
../../PHonon/PH/prepare_q.o \
../../PHonon/PH/prepare_sym_analysis.o \
../../PHonon/PH/psidspsi.o \
../../PHonon/PH/psymdvscf.o \
../../PHonon/PH/psyme.o \
../../PHonon/PH/psym_dmag.o \
../../PHonon/PH/psym_dmage.o \
../../PHonon/PH/punch_plot_e.o \
../../PHonon/PH/q_points.o \
../../PHonon/PH/q_points_wannier.o \
../../PHonon/PH/q2qstar_ph.o \
../../PHonon/PH/qdipol_cryst.o \
../../PHonon/PH/random_matrix.o \
../../PHonon/PH/read_wfc_rspace_and_fwfft.o \
../../PHonon/PH/rotate_dvscf_star.o \
../../PHonon/PH/rotate_and_add_dyn.o \
../../PHonon/PH/run_pwscf.o \
../../PHonon/PH/save_ph_input.o \
../../PHonon/PH/set_asr_c.o \
../../PHonon/PH/set_defaults_pw.o \
../../PHonon/PH/set_drhoc.o \
../../PHonon/PH/set_int12_nc.o \
../../PHonon/PH/set_irr.o \
../../PHonon/PH/set_irr_nosym.o \
../../PHonon/PH/set_irr_sym.o \
../../PHonon/PH/set_small_group_of_q.o \
../../PHonon/PH/setlocq.o \
../../PHonon/PH/setqmod.o \
../../PHonon/PH/setup_dgc.o \
../../PHonon/PH/sgam_ph.o \
../../PHonon/PH/smallgq.o \
../../PHonon/PH/solve_e.o \
../../PHonon/PH/solve_e_fpol.o \
../../PHonon/PH/solve_linter.o \
../../PHonon/PH/star_q.o \
../../PHonon/PH/stop_ph.o \
../../PHonon/PH/summarize.o \
../../PHonon/PH/sym_and_write_zue.o \
../../PHonon/PH/sym_def.o \
../../PHonon/PH/sym_dmag.o \
../../PHonon/PH/sym_dmage.o \
../../PHonon/PH/symdvscf.o \
../../PHonon/PH/symdyn_munu.o \
../../PHonon/PH/symdynph_gq.o \
../../PHonon/PH/syme.o \
../../PHonon/PH/symm.o \
../../PHonon/PH/transform_int_so.o \
../../PHonon/PH/transform_int_nc.o \
../../PHonon/PH/transform_alphasum_nc.o \
../../PHonon/PH/transform_alphasum_so.o \
../../PHonon/PH/transform_dbecsum_so.o \
../../PHonon/PH/transform_dbecsum_nc.o \
../../PHonon/PH/tra_write_matrix.o \
../../PHonon/PH/trntnsc.o \
../../PHonon/PH/write_epsilon_and_zeu.o \
../../PHonon/PH/write_matrix.o \
../../PHonon/PH/write_modes.o \
../../PHonon/PH/write_rec.o \
../../PHonon/PH/zstar_eu.o \
../../PHonon/PH/zstar_eu_us.o

RAMANOBJS = \
../../PHonon/PH/cft_wave.o \
../../PHonon/PH/psyme2.o \
../../PHonon/PH/chi_test.o \
../../PHonon/PH/dielec_test.o \
../../PHonon/PH/dvpsi_e2.o \
../../PHonon/PH/el_opt.o \
../../PHonon/PH/hdiag.o \
../../PHonon/PH/pcgreen.o \
../../PHonon/PH/ramanm.o \
../../PHonon/PH/syme2.o \
../../PHonon/PH/solve_e2.o \
../../PHonon/PH/solve_e_nscf.o \
../../PHonon/PH/write_ramtns.o \
../../PHonon/PH/d2mxc.o \
../../PHonon/PH/raman_mat.o\
../../PHonon/PH/raman.o \
../../PHonon/PH/dhdrhopsi.o \
../../PHonon/PH/set_dvscf.o

PHGPUAUXOBJS = \
../../PHonon/PH/rigid.o \
../../PHonon/PH/write_eigenvectors.o

PWOBJS = ../PW/libpwgpu.a
QEMODS = ../Modules/libqemodgpu.a


LIBOBJS        = ../../flib/ptools.a ../../flib/flib.a ../../clib/clib.a ../../iotk/src/libiotk.a

TLDEPS= bindir mods libs pw

#all : tldeps libs-ph ph-gpu.x dynmat-gpu.x matdyn-gpu.x q2r-gpu.x lambda-gpu.x 
all : tldeps libs-ph ph-gpu.x

libs-ph : libph.a libphaux.a

ph-gpu.x : phonon.o libph.a $(PWOBJS) $(LIBOBJS)
	$(LD) $(LDFLAGS) -o $@ phonon.o libph.a \
		 $(PWOBJS) $(QEMODS) $(LIBOBJS) $(LIBS)
	- ( cd ../../bin ; ln -fs ../GPU/PH/$@ . )

# Temporary disabled
#dynmat-gpu.x : dynmat.o libphaux.a  libph.a $(PWOBJS)  $(MODULES) $(LIBOBJS)
#	$(LD) $(LDFLAGS) -o $@ dynmat.o libphaux.a libph.a \
#		 $(PWOBJS) $(QEMODS) $(LIBOBJS) $(LIBS)
#	- ( cd ../../bin ; ln -fs ../GPU/PH/$@ . )

#matdyn-gpu.x : matdyn.o libphaux.a libph.a $(PWOBJS)  $(MODULES) $(LIBOBJS)
#	$(LD) $(LDFLAGS) -o $@ matdyn.o libphaux.a libph.a \
#		 $(PWOBJS) $(QEMODS) $(LIBOBJS) $(LIBS)
#	- ( cd ../../bin ; ln -fs ../GPU/PH/$@ . )

#q2r-gpu.x : q2r.o libphaux.a libph.a $(PWOBJS)  $(MODULES) $(LIBOBJS)
#	$(LD) $(LDFLAGS) -o $@ q2r.o libphaux.a libph.a \
#		 $(PWOBJS) $(QEMODS) $(LIBOBJS) $(LIBS)
#	- ( cd ../../bin ; ln -fs ../GPU/PH/$@ . )

#lambda-gpu.x : lambda.o $(PWOBJS) $(QEMODS) $(LIBOBJS)
#	$(LD) $(LDFLAGS) -o $@ lambda.o \
#                $(PWOBJS) $(QEMODS) $(LIBOBJS) $(LIBS)
#	- ( cd ../../bin ; ln -fs ../GPU/PH/$@ . )

tldeps:
	test -n "$(TLDEPS)" && ( cd ../.. ; $(MAKE) $(MFLAGS) $(TLDEPS) || exit 1) || :

libph.a : $(PHOBJS) $(RAMANOBJS)
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@

libphaux.a : $(PHGPUAUXOBJS)
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@

clean :
	- /bin/rm -f *.x *.o *.a *~ *.F90 *.d *.mod *.i *.L
	- /bin/rm -rf ../../bin/ph-gpu.x ../../bin/dynmat-gpu.x ../../bin/matdyn-gpu.x \
	../../bin/q2r-gpu.x ../../bin/lambda-gpu.x ../../bin/fqha-gpu.x

#include make.depend
# DO NOT DELETE
