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
  USE uspp,                 ONLY : becsum, okvan
  USE uspp_param,           ONLY : upf, lmaxq, nh, nhm
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
  		err = addusdens_cuda(dfftp%nr1, dfftp%nr2, dfftp%nr3, first_becsum, nat, nh, nt, ngm, qmod, qgm, ylmk0, &
  		    eigts1, eigts2, eigts3, mill (1,:), mill (2,:), mill (3,:), &
  		    aux, becsum, ityp, nspin_mag, nspin)
  		!
#if defined(__CUDA_QE_TIMING)
        CALL stop_clock( 'cu:addusdens' )
#endif
        !
        IF (err .EQ. 1) THEN
		    !
	        do ih = 1, nh (nt)
	           do jh = ih, nh (nt)
	              !
#ifdef DEBUG_ADDUSDENS
                  call start_clock ('addus:qvan2')
#endif
                  call qvan2 (ngm, ih, jh, nt, qmod, qgm, ylmk0)
#ifdef DEBUG_ADDUSDENS
                  call stop_clock ('addus:qvan2')
#endif
                  !
                  ijh = ijh + 1
                  do na = 1, nat
                     if (ityp (na) .eq.nt) then
                        !
                        !  Multiply becsum and qg with the correct structure factor
                        tbecsum(1:nspin_mag) = becsum(ijh,na,1:nspin_mag)
                        !
#ifdef DEBUG_ADDUSDENS
                        call start_clock ('addus:aux')
#endif
                        !
                        do is = 1, nspin_mag
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(skk, ig)
                           do ig = 1, ngm
                              skk = eigts1 (mill (1,ig), na) * &
                                    eigts2 (mill (2,ig), na) * &
                                    eigts3 (mill (3,ig), na)
                              aux(ig,is)=aux(ig,is) + qgm(ig)*skk*tbecsum(is)
                           enddo
!$OMP END PARALLEL DO
                        enddo
                        !
#ifdef DEBUG_ADDUSDENS
                        call stop_clock ('addus:aux')
#endif
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

