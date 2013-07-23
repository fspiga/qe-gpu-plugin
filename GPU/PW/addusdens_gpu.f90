!
! Copyright (C) 2001-2013 Quantum ESPRESSO group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
!----------------------------------------------------------------------
subroutine addusdens_g_gpu(rho)
  !----------------------------------------------------------------------
  !
  !  This routine adds to the charge density the part which is due to
  !  the US augmentation.
  !
  USE lsda_mod,             ONLY : nspin
  USE kinds,                ONLY : DP
  USE ions_base,            ONLY : nat, ntyp => nsp, ityp
  USE fft_base,             ONLY : dfftp
  USE fft_interfaces,       ONLY : invfft
  USE gvect,                ONLY : ngm, nl, nlm, gg, g, &
                                   eigts1, eigts2, eigts3, mill
  USE noncollin_module,     ONLY : noncolin, nspin_mag
  USE uspp,                 ONLY : becsum, okvan, nlx, lpl, lpx, ap, indv, nhtolm
  USE uspp_param,           ONLY : upf, lmaxq, nh, nhm, nbetam
  USE us,                   ONLY : dq, qrad
  USE control_flags,        ONLY : gamma_only
  USE wavefunctions_module, ONLY : psic
  !
  implicit none
  !
  REAL(kind=dp), intent(inout) :: rho(dfftp%nnr,nspin_mag)
  !
  !     here the local variables
  !

  integer :: ig, na, nt, ih, jh, ijh, is, err
  integer :: first_becsum
  ! counters

  real(DP) :: tbecsum(nspin_mag)  
  real(DP), allocatable :: qmod (:), ylmk0 (:,:)
  ! the modulus of G
  ! the spherical harmonics

  complex(DP) :: skk, test
  complex(DP), allocatable ::  aux (:,:), qgm(:)
  ! work space for rho(G,nspin)
  ! Fourier transform of q
  !
  INTEGER, EXTERNAL :: addusdens_cuda

  INTEGER, SAVE :: file_inx = 0
  CHARACTER(LEN=32) :: filename
  LOGICAL :: file_exists
  INTEGER :: junk_nt
  INTEGER :: nh_size

  if (.not.okvan) return

  call start_clock ('addusdens')

  allocate (aux ( ngm, nspin_mag))    
  allocate (qmod( ngm))    
  allocate (qgm( ngm))    
  allocate (ylmk0( ngm, lmaxq * lmaxq))    

  aux (:,:) = (0.d0, 0.d0)
  call ylmr2 (lmaxq * lmaxq, ngm, g, gg, ylmk0)
  do ig = 1, ngm
     qmod (ig) = sqrt (gg (ig) )
  enddo
  do nt = 1, ntyp
     if ( upf(nt)%tvanp ) then
        ijh = 0
        !
		first_becsum = nhm * (nhm + 1)/2
		!
#if defined(__CUDA_QE_TIMING)
        CALL start_clock( 'cu:addusdens' )
#endif
        !
#if defined(__WRITE_UNIT_TEST_DATA)
        WRITE(filename, "(A15,I0.3,A4)"), "addusdens_input", file_inx, ".bin"

        INQUIRE(FILE=filename, EXIST=file_exists)
        IF (.not.file_exists) then 
          OPEN(UNIT= 11, STATUS = 'REPLACE', FILE = filename, FORM='UNFORMATTED') 
          PRINT *, "Writing ", filename
          WRITE(11) dfftp%nr1
          WRITE(11) dfftp%nr2
          WRITE(11) dfftp%nr3
          WRITE(11) first_becsum
          WRITE(11) nat
          WRITE(11) nh
          WRITE(11) nt
          WRITE(11) ngm
          WRITE(11) lmaxq
          WRITE(11) nspin_mag
          WRITE(11) nspin
          WRITE(11) qmod
          WRITE(11) qgm
          WRITE(11) ylmk0
          WRITE(11) eigts1
          WRITE(11) eigts2
          WRITE(11) eigts3
          WRITE(11) mill
          WRITE(11) aux
          WRITE(11) becsum
          WRITE(11) ityp
 
          WRITE(11) SIZE(indv,1)
          WRITE(11) SIZE(indv,2)
          WRITE(11) indv
     
          WRITE(11) SIZE(nhtolm,1)
          WRITE(11) SIZE(nhtolm,2)
          WRITE(11) nhtolm
     
          WRITE(11) SIZE(qrad,1)
          WRITE(11) SIZE(qrad,2)
          WRITE(11) SIZE(qrad,3)
          WRITE(11) SIZE(qrad,4)
          WRITE(11) qrad

          WRITE(11) lpx
          WRITE(11) lpl
          WRITE(11) ap

          WRITE(11) nbetam
  
          CLOSE(11)
        ELSE
          OPEN(UNIT= 11, STATUS = 'OLD', FILE = filename, FORM='UNFORMATTED') 
          READ(11) dfftp%nr1
          READ(11) dfftp%nr2
          READ(11) dfftp%nr3
          READ(11) first_becsum
          READ(11) nat
          READ(11) nh
          READ(11) junk_nt
          READ(11) ngm
          READ(11) lmaxq
          READ(11) qmod
          READ(11) qgm
          READ(11) ylmk0
          READ(11) eigts1
          READ(11) eigts2
          READ(11) eigts3
          READ(11) mill
          READ(11) aux
          READ(11) becsum
          READ(11) ityp
          READ(11) nspin_mag
          READ(11) nspin
 
        ENDIF
#endif
   		err = addusdens_cuda(dfftp%nr1, dfftp%nr2, dfftp%nr3, first_becsum, nat, nh, nt, ngm, qmod, qgm, ylmk0, &
  		    eigts1, eigts2, eigts3, mill (1,:), mill (2,:), mill (3,:), &
  		    aux, becsum, ityp, nspin_mag, nspin, qrad, size(qrad,1), size(qrad,2), size(qrad,3), &
          size(qrad,4), lmaxq, nlx, dq, indv, nhtolm, nbetam, lpx, lpl, ap, size(ap,1), nhm)
  		!
#if defined(__CUDA_QE_TIMING)
        CALL stop_clock( 'cu:addusdens' )
#endif
        !
        IF (err .EQ. 1) THEN
		    !
	        do ih = 1, nh (nt)
	           do jh = ih, nh (nt)
#ifdef DEBUG_ADDUSDENS
	              call start_clock ('addus:qvan2')
#endif
	              call qvan2 (ngm, ih, jh, nt, qmod, qgm, ylmk0)
#ifdef DEBUG_ADDUSDENS
	              call stop_clock ('addus:qvan2')
#endif
	              ijh = ijh + 1
	              do na = 1, nat
	                 if (ityp (na) .eq. nt) then
	                    !
	                    !  Multiply becsum and qg with the correct structure factor
	                    !
	                    do is = 1, nspin_mag
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ig, skk)
	                       do ig = 1, ngm
	                          skk = eigts1 (mill (1,ig), na) * &
	                                eigts2 (mill (2,ig), na) * &
	                                eigts3 (mill (3,ig), na)
	                          aux(ig,is)=aux(ig,is) + qgm(ig)*skk*becsum(ijh,na,is)
	                       enddo
!$OMP END PARALLEL DO
	                    enddo
                        !
	                 endif
	              enddo
	           enddo
	        enddo
            !
        ENDIF
        !
     endif
  enddo
  !
  deallocate (ylmk0)
  deallocate (qgm)
  deallocate (qmod)
  !
  !     convert aux to real space and add to the charge density
  !
  do is = 1, nspin_mag
     psic(:) = (0.d0, 0.d0)
     psic( nl(:) ) = aux(:,is)
     if (gamma_only) psic( nlm(:) ) = CONJG(aux(:,is))
     CALL invfft ('Dense', psic, dfftp)
     rho(:, is) = rho(:, is) +  DBLE (psic (:) )
  enddo
  deallocate (aux)

  call stop_clock ('addusdens')
  return
end subroutine addusdens_g_gpu

