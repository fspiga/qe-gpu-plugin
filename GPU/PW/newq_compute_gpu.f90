!
! Copyright (C) 2001-2013 Quantum ESPRESSO group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
SUBROUTINE newq_compute_gpu(vr,deeq,skip_vltot)
  !
  !   This routine computes the integral of the perturbed potential with
  !   the Q function 
  !
  USE kinds,                ONLY : DP
  USE ions_base,            ONLY : nat, ntyp => nsp, ityp
  USE cell_base,            ONLY : omega
  USE fft_base,             ONLY : dfftp
  USE fft_interfaces,       ONLY : fwfft
  USE gvect,                ONLY : g, gg, ngm, gstart, mill, &
                                   eigts1, eigts2, eigts3, nl
  USE lsda_mod,             ONLY : nspin
  USE scf,                  ONLY : vltot
  USE uspp,                 ONLY : okvan, indv, nlx, lpl, lpx, ap, nhtolm !LDB added nhtolm, lpl, lpx and ap for debug/optimization
  USE us,                   ONLY : dq, qrad 
  USE uspp_param,           ONLY : upf, lmaxq, nh, nhm, nbetam ! LDB added nbetam for debuggin
  USE control_flags,        ONLY : gamma_only
  USE wavefunctions_module, ONLY : psic
  USE spin_orb,             ONLY : lspinorb, domag
  USE noncollin_module,     ONLY : nspin_mag
  USE mp_global,            ONLY : intra_bgrp_comm
  USE mp,                   ONLY : mp_sum
  !
  IMPLICIT NONE
  !
  !
  ! Input: potential , output: contribution to integral
  REAL(kind=dp), intent(in)  :: vr(dfftp%nnr,nspin)
  REAL(kind=dp), intent(out) :: deeq( nhm, nhm, nat, nspin )
  LOGICAL, intent(in) :: skip_vltot !If .false. vltot is added to vr when necessary
  ! INTERNAL
  INTEGER :: ig, nt, ih, jh, na, is, nht, nb, mb, ii
  ! counters on g vectors, atom type, beta functions x 2,
  !   atoms, spin, aux, aux, beta func x2 (again)
  
#if defined(__OPENMP)
  INTEGER :: mytid, ntids, omp_get_thread_num, omp_get_num_threads
#endif

  INTEGER :: flag
  COMPLEX(DP), ALLOCATABLE :: aux(:,:), qgm(:), qgm_na(:)

  ! work space
  COMPLEX(DP) :: dtmp, dtmp_sum
  REAL(DP), ALLOCATABLE :: ylmk0(:,:), qmod(:)
    ! spherical harmonics, modulus of G
  REAL(DP) :: fact, ddot

  INTEGER, EXTERNAL :: newd_cuda
  INTEGER :: err

  INTEGER, SAVE :: file_inx = 0
  CHARACTER(LEN=32) :: filename
  LOGICAL :: file_exists
  INTEGER :: junk_nt
  INTEGER :: nh_size

  IF ( gamma_only ) THEN
     !
     fact = 2.D0
     !
  ELSE
     !
     fact = 1.D0
     !
  END IF
  !
  CALL start_clock( 'newd' )
  !
  ALLOCATE( aux( ngm, nspin_mag ),  qmod( ngm ), ylmk0( ngm, lmaxq*lmaxq ) )
  !
  deeq(:,:,:,:) = 0.D0
  !
  CALL ylmr2( lmaxq * lmaxq, ngm, g, gg, ylmk0 )
  !
  qmod(1:ngm) = SQRT( gg(1:ngm) )
  !
  ! ... fourier transform of the total effective potential
  !
  DO is = 1, nspin_mag
     !
     IF ( (nspin_mag == 4 .AND. is /= 1) .or. skip_vltot ) THEN 
        !
        psic(:) = vr(:,is)
        !
     ELSE
        !
        psic(:) = vltot(:) + vr(:,is)
        !
     END IF
     !
     CALL fwfft ('Dense', psic, dfftp)
     !
     aux(1:ngm,is) = psic( nl(1:ngm) )
     !
  END DO
  !
  ! ... here we compute the integral Q*V for each atom,
  ! ...       I = sum_G exp(-iR.G) Q_nm v^*
  !
  DO nt = 1, ntyp
     !
     IF ( upf(nt)%tvanp ) THEN
        !
        flag = 0
        IF ( gamma_only .AND. gstart == 2 ) flag = 1
        !
        err = newd_cuda( dfftp%nr1, dfftp%nr2, dfftp%nr3, na, nh, fact, &
            nt, nat, ngm, nhm, nspin, qmod, ylmk0, &
            eigts1, eigts2, eigts3, mill(1,:), mill(2,:), mill(3,:), &
            deeq, ityp, omega, flag, aux, nspin_mag, qrad, &
            size(qrad,1), size(qrad,2), size(qrad,3), size(qrad,4), lmaxq, &
            nlx, dq, indv, nhtolm, nbetam, lpx, lpl, ap, size(ap,1))
        !
        IF (err .EQ. 1) THEN
            !
	        ALLOCATE( qgm( ngm ))
	        !
	        DO ih = 1, nh(nt)
	           !
	           DO jh = ih, nh(nt)
	              !
	              ! ... The Q(r) for this atomic species without structure factor
	              !
	              CALL qvan2( ngm, ih, jh, nt, qmod, qgm, ylmk0 )
	              !
!$omp parallel default(shared), private(na,qgm_na,is,dtmp,ig,mytid,ntids)
#if defined(__OPENMP)
	              mytid = omp_get_thread_num()  ! take the thread ID
	              ntids = omp_get_num_threads() ! take the number of threads
#endif
	              ALLOCATE(  qgm_na( ngm ) )
	              !
	              DO na = 1, nat
                    !
#if defined(__OPENMP)
	                 ! distribute atoms round robin to threads
	                 !
	                 IF( MOD( na, ntids ) /= mytid ) CYCLE
#endif
	                 !
	                 IF ( ityp(na) == nt ) THEN
	                    !
	                    ! ... The Q(r) for this specific atom
	                    !
	                    qgm_na(1:ngm) = qgm(1:ngm) * eigts1(mill(1,1:ngm),na) &
	                                               * eigts2(mill(2,1:ngm),na) &
	                                               * eigts3(mill(3,1:ngm),na)
	                    !
	                    ! ... and the product with the Q functions
	                    !
	                    DO is = 1, nspin_mag
	                       !
#if defined(__OPENMP)
	                       dtmp = 0.0d0
	                       DO ig = 1, ngm
	                          dtmp = dtmp + aux( ig, is ) * CONJG( qgm_na( ig ) )
	                       END DO
#else
                           dtmp = ddot( 2 * ngm, aux(1,is), 1, qgm_na, 1 )
#endif
	                       deeq(ih,jh,na,is) = fact * omega * DBLE( dtmp )
	                       !
	                       IF ( gamma_only .AND. gstart == 2 ) &
	                           deeq(ih,jh,na,is) = deeq(ih,jh,na,is) - &
	                                           omega * DBLE( aux(1,is) * qgm_na(1) )
	                       !
	                       deeq(jh,ih,na,is) = deeq(ih,jh,na,is)
	                       !
	                    END DO
	                    !
	                 END IF
	                 !
	              END DO
	              !
	              DEALLOCATE( qgm_na )
!$omp end parallel
	              !
	           END DO
	           !
	        END DO
	        !
	        DEALLOCATE( qgm )
	        !
        ENDIF
        !
     END IF
     !
  END DO
  !
  CALL mp_sum( deeq( :, :, :, 1:nspin_mag ), intra_bgrp_comm )
  !
  IF ( ALLOCATED( aux ) )          DEALLOCATE( aux )
  IF ( ALLOCATED( qgm ) )          DEALLOCATE( qgm )
  IF ( ALLOCATED( qmod ) )         DEALLOCATE( qmod )
  IF ( ALLOCATED( ylmk0 ) )        DEALLOCATE( ylmk0 )

END SUBROUTINE newq_compute_gpu
