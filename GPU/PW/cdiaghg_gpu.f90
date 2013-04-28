!
! Copyright (C) 2001-2013 Quantum ESPRESSO group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
#define ZERO ( 0.D0, 0.D0 )
#define ONE  ( 1.D0, 0.D0 )
!
!----------------------------------------------------------------------------
SUBROUTINE cdiaghg_gpu( n, m, h, s, ldh, e, v )
  !----------------------------------------------------------------------------
  !
  ! ... calculates eigenvalues and eigenvectors of the generalized problem
  ! ... Hv=eSv, with H hermitean matrix, S overlap matrix.
  ! ... On output both matrix are unchanged
  !
  ! ... LAPACK version - uses both ZHEGV and ZHEGVX
  !
  USE kinds,            ONLY : DP
  USE mp,               ONLY : mp_bcast, mp_sum, mp_barrier, mp_max
  USE mp_global,        ONLY : me_bgrp, root_bgrp, intra_bgrp_comm
  !
  IMPLICIT NONE
  !
  INTEGER, INTENT(IN) :: n, m, ldh
    ! dimension of the matrix to be diagonalized
    ! number of eigenstates to be calculate
    ! leading dimension of h, as declared in the calling pgm unit
  COMPLEX(DP), INTENT(INOUT) :: h(ldh,n), s(ldh,n)
    ! actually intent(in) but compilers don't know and complain
    ! matrix to be diagonalized
    ! overlap matrix
  REAL(DP), INTENT(OUT) :: e(n)
    ! eigenvalues
  COMPLEX(DP), INTENT(OUT) :: v(ldh,m)
    ! eigenvectors (column-wise)
  !
#if defined(__MAGMA)
  !
  INTEGER                  :: lwork, nb, mm, info, i, j, liwork, lrwork
    ! mm = number of calculated eigenvectors
  REAL(DP)                 :: abstol
  INTEGER,     ALLOCATABLE :: iwork(:), ifail(:)
  REAL(DP),    ALLOCATABLE :: rwork(:), sdiag(:), hdiag(:)
  COMPLEX(DP), ALLOCATABLE :: work(:)
    ! various work space
  LOGICAL                  :: all_eigenvalues
 ! REAL(DP), EXTERNAL       :: DLAMCH
  INTEGER,  EXTERNAL       :: ILAENV
    ! ILAENV returns optimal block size "nb"
#if defined(__ZHEGVD)
  COMPLEX(DP), ALLOCATABLE :: vv(:,:)
#endif
  !
  CALL start_clock( 'cdiaghg' )
  !
#if defined(__CUDA_DEBUG)
  WRITE(*,*) "[CDIAGHG] Compute cdiaghg, n = ", n
#endif
  !
  ! ... only the first processor diagonalizes the matrix
  !
  IF ( me_bgrp == root_bgrp ) THEN
     !
#if defined(__PHIGEMM)
     call phigemmShutdown()
     call deAllocateDeviceMemory()
#endif
     !
     ! ... save the diagonal of input S (it will be overwritten)
     !
     ALLOCATE( sdiag( n ) )
     DO i = 1, n
        sdiag(i) = DBLE( s(i,i) )
     END DO
     !
     all_eigenvalues = ( m == n )
     !
     ! ... check for optimal block size
     !
     ! magic number. See magma_get_zhetrd_nb()
     nb = 32
     lwork = ( nb + 1 )*n
     !
#if defined(__ZHEGVD)
     IF ( all_eigenvalues ) THEN
        lwork = 2 * n * nb + n * n
     ENDIF
#endif
     !
     ALLOCATE( work( lwork ) )
     !
     IF ( all_eigenvalues ) THEN
        !
        ! ... calculate all eigenvalues (overwritten to v)
        !
        v(:,:) = h(:,:)
        !
#if defined( __ZHEGVD)
        !
        ! Added +1 just to be "sure"...
        liwork = 4 + 5*n
        lrwork = 2 + 5*n + 2*n*n
        ALLOCATE( iwork( liwork ) )
        ALLOCATE( rwork( lrwork ) )
        !
        CALL magmaf_zhegvd( 1,  'V', 'U', n, v, ldh, s, ldh, e, &
                      work, lwork, rwork, lrwork, iwork, liwork, info)
        !
#else
        !
        ALLOCATE( rwork( 3*n - 2 ) )
        !
        CALL ZHEGV( 1, 'V', 'U', n, v, ldh, &
                    s, ldh, e, work, lwork, rwork, info )
        !
#endif
        !
     ELSE
        !
        liwork = 5*n
        lrwork = 7*n
        ALLOCATE( rwork( lrwork ) )
        !
        ! ... save the diagonal of input H (it will be overwritten)
        !
        ALLOCATE( hdiag( n ) )
        !
        DO i = 1, n
           hdiag(i) = DBLE( h(i,i) )
        END DO
        !
        ALLOCATE( iwork( liwork ) )
        ALLOCATE( ifail( n ) )
        !
        ! ... calculate only m lowest eigenvalues
        !
        abstol = 0.D0
        ! abstol = 2.D0*DLAMCH( 'S' )
        !
        CALL  magmaf_zhegvx( 1, 'V', 'I', 'U', n, h, ldh, s, ldh, &
                     0.D0, 0.D0, 1, m, abstol, mm, e, v, ldh, &
                     work, lwork, rwork, iwork, ifail, info )
        !
        !
        DEALLOCATE( ifail )
        !
        ! ... restore input H matrix from saved diagonal and lower triangle
        !
        DO i = 1, n
           h(i,i) = CMPLX( hdiag(i), 0.0_DP ,kind=DP)
           DO j = i + 1, n
              h(i,j) = CONJG( h(j,i) )
           END DO
           DO j = n + 1, ldh
              h(j,i) = ( 0.0_DP, 0.0_DP )
           END DO
        END DO
        !
        DEALLOCATE( hdiag )
        !
     !
     END IF
     !
     IF ( ALLOCATED( rwork ) ) DEALLOCATE( rwork )
     IF ( ALLOCATED(  work ) ) DEALLOCATE( work )
     IF ( ALLOCATED( iwork ) ) DEALLOCATE( iwork )
     !
     CALL errore( 'cdiaghg', 'diagonalization (ZHEGV*) failed', ABS( info ) )
     !
     ! ... restore input S matrix from saved diagonal and lower triangle
     !
     DO i = 1, n
        s(i,i) = CMPLX( sdiag(i), 0.0_DP ,kind=DP)
        DO j = i + 1, n
           s(i,j) = CONJG( s(j,i) )
        END DO
        DO j = n + 1, ldh
           s(j,i) = ( 0.0_DP, 0.0_DP )
        END DO
     END DO
     !
     DEALLOCATE( sdiag )
     !
  END IF
  !
  ! ... broadcast eigenvectors and eigenvalues to all other processors
  !
  ! ... if OpenMP is enabled then the GPU memory is re-allocated in
  ! ... parallel during the data broadcasting
  !
!$OMP PARALLEL DEFAULT(SHARED)
  !
!$OMP MASTER
  CALL mp_bcast( e, root_bgrp, intra_bgrp_comm )
  CALL mp_bcast( v, root_bgrp, intra_bgrp_comm )
!$OMP END MASTER
  !
!$OMP SECTIONS
!$OMP SECTION
  IF ( me_bgrp == root_bgrp ) THEN
     ! Reinizialize the GPU memory
     call allocateDeviceMemory()
#if defined(__PHIGEMM)
     call initPhigemm()
#endif
  END IF
!$OMP END SECTIONS
  !
!$OMP END PARALLEL
  !
  CALL stop_clock( 'cdiaghg' )
  !
#endif
  !
  RETURN
  !
END SUBROUTINE cdiaghg_gpu
