!
! Copyright (C) 2001-2009 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!-----------------------------------------------------------------------
PROGRAM phonon
  !-----------------------------------------------------------------------
  !
  ! ... This is the main driver of the phonon code.
  ! ... It reads all the quantities calculated by pwscf, it
  ! ... checks if some recover file is present and determines
  ! ... which calculation needs to be done. Finally, it calls do_phonon
  ! ... that does the loop over the q points.
  ! ... Presently implemented:
  ! ... dynamical matrix (q/=0)   NC [4], US [4], PAW [4]
  ! ... dynamical matrix (q=0)    NC [5], US [5], PAW [4]
  ! ... dielectric constant       NC [5], US [5], PAW [3]
  ! ... born effective charges    NC [5], US [5], PAW [3]
  ! ... polarizability (iu)       NC [2], US [2]
  ! ... electron-phonon           NC [3], US [3]
  ! ... electro-optic             NC [1]
  ! ... raman tensor              NC [1]
  !
  ! NC = norm conserving pseudopotentials
  ! US = ultrasoft pseudopotentials
  ! PAW = projector augmented-wave
  ! [1] LDA, 
  ! [2] [1] + GGA, 
  ! [3] [2] + LSDA/sGGA, 
  ! [4] [3] + Spin-orbit/nonmagnetic,
  ! [5] [4] + Spin-orbit/magnetic (experimental when available)
  !
  ! Not implemented in ph.x:
  ! [6] [5] + constraints on the magnetization
  ! [7] [6] + Hubbard U
  ! [8] [7] + Hybrid functionals
  ! [9] ? + External Electric field
  ! [10] ? + nonperiodic boundary conditions.

  USE control_ph,      ONLY : bands_computed, qplot
  USE check_stop,      ONLY : check_stop_init
  USE ph_restart,      ONLY : ph_writefile
  USE mp_global,       ONLY : mp_startup
  USE environment,     ONLY : environment_start
  USE fft_base,        ONLY : dffts
  USE omp_lib
  USE io_global, ONLY : stdout
  !
  IMPLICIT NONE
  !
  INTEGER :: iq, ierr
  LOGICAL :: do_band, do_iq, setup_pw
  CHARACTER (LEN=9)   :: code = 'PHONON'
  CHARACTER (LEN=256) :: auxdyn
  external :: h_psiq_cuda_k_cufftplan_init, h_psiq_cuda_k_cufftplan_destroy


#ifdef __CUDA_DEBUG
  double precision :: start_time, end_time, h_psiq_time
  double precision, external ::  get_h_psiq_time_init_1, get_h_psiq_time_init_2, get_h_psiq_time_init_3, get_h_psiq_time_init_4, get_h_psiq_time_init_5, get_h_psiq_time_core, get_h_psiq_time_down
  external :: h_psiq_cuda_k_time_init
  
  
  common h_psiq_time
  
  h_psiq_time = 0.d0


  call h_psiq_cuda_k_time_init()
  
  start_time=omp_get_wtime()
#endif


  !
  ! Initialize MPI, clocks, print initial messages
  !
  CALL mp_startup ( start_images=.true. )
  CALL environment_start ( code )
  !
  ! ... and begin with the initialization part
  !
  CALL phq_readin()
  !
  CALL check_stop_init()
  !
  ! ... Checking the status of the calculation and if necessary initialize
  ! ... the q mesh
  !
  CALL check_initial_status(auxdyn)
  !
  ! ... Do the loop over the q points and irreps.
  !
  call h_psiq_cuda_k_cufftplan_init(dffts%nr1x, dffts%nr2x, dffts%nr3x)
  CALL do_phonon(auxdyn)
  call h_psiq_cuda_k_cufftplan_destroy()
  !
  !  reset the status of the recover files
  !
  CALL ph_writefile('status_ph',1,0,ierr)
  !
  IF (qplot) CALL write_qplot_data(auxdyn)
  !
  IF (bands_computed) CALL print_clock_pw()


#ifdef __CUDA_DEBUG
  end_time=omp_get_wtime()
  
  write(stdout,*)
  write(stdout,*) 'Total H_PSIQ::h_psiq_time_init_1 Running Time : ', get_h_psiq_time_init_1()
  write(stdout,*) 'Total H_PSIQ::h_psiq_time_init_2 Running Time : ', get_h_psiq_time_init_2()
  write(stdout,*) 'Total H_PSIQ::h_psiq_time_init_3 Running Time : ', get_h_psiq_time_init_3()
  write(stdout,*) 'Total H_PSIQ::h_psiq_time_init_4 Running Time : ', get_h_psiq_time_init_4()
  write(stdout,*) 'Total H_PSIQ::h_psiq_time_init_5 Running Time : ', get_h_psiq_time_init_5()
  write(stdout,*) 'Total H_PSIQ::h_psiq_time_core Running Time : ', get_h_psiq_time_core()
  write(stdout,*) 'Total H_PSIQ::h_psiq_time_down Running Time : ', get_h_psiq_time_down()
  write(stdout,*) 'Total H_PSIQ Running Time : ', h_psiq_time
  write(stdout,*) 'Total Phonon Running Time : ', end_time - start_time
  write(stdout,*)
#endif

  !
  CALL stop_smoothly_ph( .TRUE. )
  !
  STOP
  !
END PROGRAM phonon
