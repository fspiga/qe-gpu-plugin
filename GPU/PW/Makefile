# Makefile for PW

include ../../make.sys

# location of needed modules
MODFLAGS= $(MOD_FLAG)../../iotk/src $(MOD_FLAG)../../PW/src $(MOD_FLAG)../../Modules $(MOD_FLAG)../Modules $(MOD_FLAG). 

PWGPUOBJS = \
addusdens.o \
newd.o \
cdiaghg.o \
gk_sort.o \
h_psi.o \
rdiaghg.o \
struct_fact.o

PWOBJS = \
../../PW/src/a2fmod.o \
../../PW/src/add_bfield.o \
../../PW/src/add_efield.o \
../../PW/src/add_vuspsi.o \
../../PW/src/add_paw_to_deeq.o \
../../PW/src/addusforce.o \
../../PW/src/addusstress.o \
../../PW/src/allocate_fft.o \
../../PW/src/allocate_fft_custom.o \
../../PW/src/allocate_locpot.o \
../../PW/src/allocate_nlpot.o \
../../PW/src/allocate_wfc.o \
../../PW/src/atomic_rho.o \
../../PW/src/atomic_wfc.o \
../../PW/src/average_pp.o \
../../PW/src/acfdt_in_pw.o \
../../PW/src/bp_mod.o \
../../PW/src/bp_c_phase.o \
../../PW/src/bp_calc_btq.o \
../../PW/src/bp_qvan3.o \
../../PW/src/bp_strings.o \
../../PW/src/buffers.o \
../../PW/src/c_bands.o \
../../PW/src/c_phase_field.o \
../../PW/src/ccgdiagg.o \
../../PW/src/cdiagh.o \
../../PW/src/cegterg.o \
../../PW/src/clean_pw.o \
../../PW/src/close_files.o \
../../PW/src/compute_becsum.o \
../../PW/src/compute_deff.o \
../../PW/src/compute_dip.o \
../../PW/src/compute_rho.o \
../../PW/src/compute_qdipol.o \
../../PW/src/compute_qdipol_so.o \
../../PW/src/compute_ux.o \
../../PW/src/coset.o \
../../PW/src/d_matrix.o \
../../PW/src/data_structure.o \
../../PW/src/data_structure_custom.o \
../../PW/src/deriv_drhoc.o \
../../PW/src/divide_class.o \
../../PW/src/divide_class_so.o \
../../PW/src/realus.o \
../../PW/src/divide.o \
../../PW/src/divide_et_impera.o \
../../PW/src/dqvan2.o \
../../PW/src/drhoc.o \
../../PW/src/dvloc_of_g.o \
../../PW/src/dynamics_module.o \
../../PW/src/efermig.o \
../../PW/src/efermit.o \
../../PW/src/electrons.o \
../../PW/src/eqvect.o \
../../PW/src/esm.o \
../../PW/src/ewald.o \
../../PW/src/ewald_dipole.o \
../../PW/src/exx.o \
../../PW/src/find_group.o \
../../PW/src/forces_bp_efield.o \
../../PW/src/force_cc.o \
../../PW/src/force_corr.o \
../../PW/src/force_ew.o \
../../PW/src/force_hub.o \
../../PW/src/force_lc.o \
../../PW/src/force_us.o \
../../PW/src/forces.o \
../../PW/src/g_psi.o \
../../PW/src/g_psi_mod.o \
../../PW/src/gen_at_dj.o \
../../PW/src/gen_at_dy.o \
../../PW/src/gen_us_dj.o \
../../PW/src/gen_us_dy.o \
../../PW/src/get_locals.o \
../../PW/src/gradcorr.o \
../../PW/src/gweights.o \
../../PW/src/g2_kin.o \
../../PW/src/h_epsi_her_apply.o \
../../PW/src/h_epsi_her_set.o \
../../PW/src/h_1psi.o \
../../PW/src/h_psi_meta.o \
../../PW/src/hinit0.o \
../../PW/src/hinit1.o \
../../PW/src/init_ns.o \
../../PW/src/init_run.o \
../../PW/src/init_us_1.o \
../../PW/src/init_us_2.o \
../../PW/src/init_at_1.o \
../../PW/src/init_vloc.o \
../../PW/src/input.o \
../../PW/src/interpolate.o \
../../PW/src/io_rho_xml.o \
../../PW/src/irrek.o \
../../PW/src/iweights.o \
../../PW/src/start_k.o \
../../PW/src/kpoint_grid.o \
../../PW/src/lchk_tauxk.o \
../../PW/src/make_pointlists.o \
../../PW/src/makov_payne.o \
../../PW/src/martyna_tuckerman.o \
../../PW/src/memory_report.o \
../../PW/src/mix_rho.o \
../../PW/src/move_ions.o \
../../PW/src/ms2.o \
../../PW/src/multable.o \
../../PW/src/n_plane_waves.o \
../../PW/src/new_ns.o \
../../PW/src/new_occ.o \
../../PW/src/ns_adj.o \
../../PW/src/nonloccorr.o \
../../PW/src/non_scf.o \
../../PW/src/offset_atom_wfc.o \
../../PW/src/openfil.o \
../../PW/src/orbm_kubo.o \
../../PW/src/orthoatwfc.o \
../../PW/src/output_tau.o \
../../PW/src/para.o \
../../PW/src/paw_init.o \
../../PW/src/paw_onecenter.o \
../../PW/src/paw_symmetry.o \
../../PW/src/plugin_initialization.o \
../../PW/src/plugin_forces.o \
../../PW/src/plus_u_full.o \
../../PW/src/potinit.o \
../../PW/src/print_clock_pw.o \
../../PW/src/print_ks_energies.o \
../../PW/src/punch.o \
../../PW/src/pw_restart.o \
../../PW/src/pwcom.o \
../../PW/src/pw2blip.o \
../../PW/src/pw2casino.o \
../../PW/src/pw2casino_write.o \
../../PW/src/qvan2.o \
../../PW/src/rcgdiagg.o \
../../PW/src/rdiagh.o \
../../PW/src/read_conf_from_file.o \
../../PW/src/read_file.o \
../../PW/src/regterg.o \
../../PW/src/remove_atomic_rho.o \
../../PW/src/report_mag.o \
../../PW/src/restart_from_file.o \
../../PW/src/restart_in_electrons.o \
../../PW/src/restart_in_ions.o \
../../PW/src/rho2zeta.o \
../../PW/src/rotate_wfc.o \
../../PW/src/rotate_wfc_k.o \
../../PW/src/rotate_wfc_gamma.o \
../../PW/src/ruotaijk.o \
../../PW/src/s_1psi.o \
../../PW/src/s_psi.o \
../../PW/src/save_in_cbands.o \
../../PW/src/save_in_electrons.o \
../../PW/src/save_in_ions.o \
../../PW/src/scale_h.o \
../../PW/src/scf_mod.o \
../../PW/src/set_kplusq.o \
../../PW/src/set_kup_and_kdw.o \
../../PW/src/set_rhoc.o \
../../PW/src/set_vrs.o \
../../PW/src/setlocal.o \
../../PW/src/setqf.o \
../../PW/src/setup.o \
../../PW/src/spinor.o \
../../PW/src/sph_ind.o \
../../PW/src/stop_run.o \
../../PW/src/stres_cc.o \
../../PW/src/stres_ewa.o \
../../PW/src/stres_gradcorr.o \
../../PW/src/stres_har.o \
../../PW/src/stres_hub.o \
../../PW/src/stres_knl.o \
../../PW/src/stres_loc.o \
../../PW/src/stres_us.o \
../../PW/src/stres_nonloc_dft.o \
../../PW/src/stress.o \
../../PW/src/sum_band.o \
../../PW/src/sumkg.o \
../../PW/src/sumkt.o \
../../PW/src/summary.o \
../../PW/src/symme.o \
../../PW/src/symm_base.o \
../../PW/src/symmetrize_at.o \
../../PW/src/tabd.o \
../../PW/src/transform_becsum_so.o \
../../PW/src/transform_becsum_nc.o \
../../PW/src/transform_qq_so.o \
../../PW/src/trnvecc.o \
../../PW/src/tweights.o \
../../PW/src/update_pot.o \
../../PW/src/usnldiag.o \
../../PW/src/v_of_rho.o \
../../PW/src/vcsmd.o \
../../PW/src/vcsubs.o \
../../PW/src/vhpsi.o \
../../PW/src/vloc_of_g.o \
../../PW/src/vloc_psi.o \
../../PW/src/xk_wk_collect.o \
../../PW/src/wfcinit.o \
../../PW/src/write_ns.o \
../../PW/src/wsweight.o \
../../PW/src/weights.o \
../../PW/src/ortho_wfc.o \
../../PW/src/wannier_proj.o \
../../PW/src/wannier_init.o \
../../PW/src/wannier_check.o \
../../PW/src/wannier_clean.o \
../../PW/src/wannier_occ.o \
../../PW/src/wannier_enrg.o

CUGPUOBJS = \
addusdens_cuda.o \
vloc_psi_k_cuda.o \
vloc_psi_gamma_cuda.o \
newd_cuda.o \
struct_fact_cuda.o \
gk_sort_cuda.o

QEMODS=../Modules/libqemodgpu.a

TLDEPS=bindir gpu-mods mods libs libphiGEMM libMAGMA liblapack libblas libenviron

LIBOBJS	= ../../flib/ptools.a ../../flib/flib.a ../../clib/clib.a ../../iotk/src/libiotk.a

all : tldeps pw-gpu.x generate_vdW_kernel_table-gpu.x

pw-gpu.x : pwscf.o libpwgpu.a $(LIBOBJS) $(QEMODS)
	$(LD) $(LDFLAGS) -o $@ \
	   pwscf.o $(PWOBJS) libpwgpu.a $(QEMODS) $(LIBOBJS) $(LIBS)
	- ( cd ../../bin; ln -fs ../GPU/PW/$@ . )

libpwgpu.a : $(PWGPUOBJS) $(CUGPUOBJS)
	$(AR) $(ARFLAGS) $@ $? $(PWOBJS)
	$(RANLIB) $@

tldeps:
	test -n "$(TLDEPS)" && ( cd ../.. ; $(MAKE) $(MFLAGS) $(TLDEPS) || exit 1) || :

clean :
	- /bin/rm -f *.x *.o *.a *~ *.F90 *.d *.mod *.i *.L

generate_vdW_kernel_table-gpu.x : generate_vdW_kernel_table.o libpwgpu.a 
	$(LD) $(LDFLAGS) -o $@ \
		generate_vdW_kernel_table.o $(QEMODS) libpwgpu.a $(LIBOBJS) $(LIBS)
	- ( cd ../../bin; ln -fs ../GPU/PW/$@ . )

#include make.depend

# DO NOT DELETE
