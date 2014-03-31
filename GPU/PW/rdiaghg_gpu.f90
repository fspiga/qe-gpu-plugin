! Copyright (C) 2001-2014 Quantum ESPRESSO Foundation
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
!----------------------------------------------------------------------------
SUBROUTINE rdiaghg_gpu( n, m, h, s, ldh, e, v )
  !----------------------------------------------------------------------------
  !
  ! ... calculates eigenvalues and eigenvectors of the generalized problem
  ! ... Hv=eSv, with H symmetric matrix, S overlap matrix.
  ! ... On output both matrix are unchanged
  !
  ! ... LAPACK version - uses both DSYGV and DSYGVX
  !
  USE kinds,            ONLY : DP
  USE mp,               ONLY : mp_bcast
  USE mp_global,        ONLY : me_bgrp, root_bgrp, intra_bgrp_comm
  !
  USE iso_c_binding
  USE cuda_mem_alloc
  !
  IMPLICIT NONE
  !
  INTEGER, INTENT(IN) :: n, m, ldh
    ! dimension of the matrix to be diagonalized
    ! number of eigenstates to be calculated
    ! leading dimension of h, as declared in the calling pgm unit
  REAL(DP), INTENT(INOUT) :: h(ldh,n), s(ldh,n)
    ! matrix to be diagonalized
    ! overlap matrix
  !
  REAL(DP), INTENT(OUT) :: e(n)
    ! eigenvalues
  REAL(DP), INTENT(OUT) :: v(ldh,m)
    ! eigenvectors (column-wise)
  !
#if defined(__MAGMA)
  !
  INTEGER               :: i, j, lwork, nb, mm, info, liwork
    ! mm = number of calculated eigenvectors
  REAL(DP)              :: abstol
  REAL(DP), PARAMETER   :: one = 1_DP
  REAL(DP), PARAMETER   :: zero = 0_DP
  INTEGER,  ALLOCATABLE :: iwork(:), ifail(:)
  REAL(DP), ALLOCATABLE :: work(:), sdiag(:), hdiag(:)
  LOGICAL               :: all_eigenvalues
  INTEGER,  EXTERNAL    :: ILAENV
  !
  INTEGER :: res
  REAL(fp_kind), dimension(:,:), pointer :: vv(:,:)
  TYPE(C_PTR) :: cptr_vv
  INTEGER (C_SIZE_T), PARAMETER :: test_flag = 0
  INTEGER (C_SIZE_T) :: allocation_size
  !
  ! ILAENV returns optimal block size "nb"
  !
  CALL start_clock( 'rdiaghg' )
  !
  ! ... only the first processor diagonalize the matrix
  !
  IF ( me_bgrp == root_bgrp ) THEN
     !
#if defined(__CUDA_DEBUG)
     WRITE(*,*) "[RDIAGHG] Compute rdiaghg, n = ", n
#endif
     !
#if defined(__PHIGEMM)
     ! not sure this is smart
     call phigemmShutdown()
#endif
     call deAllocateDeviceMemory()
     !
     ! ... save the diagonal of input S (it will be overwritten)
     !
     ALLOCATE( sdiag( n ) )
     DO i = 1, n
        sdiag(i) = s(i,i)
     END DO
     !
     all_eigenvalues = ( m == n )
     !
     ! ... check for optimal block size -> MAGMA hard-coded value = 32
     !
     liwork = 6*n
     !
     IF ( all_eigenvalues ) THEN
        lwork = 1 + 6 * n * 32 + 2* n * n;
     ELSE
        lwork = 1 + 6 * n + 2* n * n;
     END IF
     !
     ALLOCATE( work( lwork ) )
     !
     IF ( all_eigenvalues ) THEN
        !
        ! ... calculate all eigenvalues
        !
        v(:,:) = h(:,:)
        !
        ALLOCATE( iwork( liwork ) )
        !
        !CALL DSYGVD( 1, 'V', 'U', n, v, ldh, s, ldh, e, work, lwork, iwork, liwork, info )
        CALL magmaf_dsygvd( 1, 'V', 'U', n, v, ldh, s, ldh, e, work, lwork, iwork, liwork, info )
        !
     ELSE
        !
        ! ... calculate only m lowest eigenvalues
        !
        ALLOCATE( iwork( liwork ) )
        !
!      subroutine magmaf_dsygvdx( itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, &
!                                il, iu, m, w, work, lwork, &
!                                iwork, liwork, info)
        !
!        ALLOCATE (vv(ldh, n) )
        allocation_size = ldh*n*sizeof(fp_kind)*4
        res = cudaHostAlloc ( cptr_vv, allocation_size, test_flag )
        CALL c_f_pointer ( cptr_vv, vv, (/ ldh, n /) )
        !
        vv(:,:) = h(:,:)
        !
        CALL magmaf_dsygvdx( 1, 'V', 'I', 'U', n, vv, ldh, s, ldh, &
                             0.D0, 0.D0, 1, m, mm, e, &
                             work, lwork, iwork, liwork, info )
        !
        v(:,:) = vv(:, 1:m)
        res = cudaFreeHost(cptr_vv)
!        DEALLOCATE( vv)
        !
        !
     END IF
     !
     DEALLOCATE( work )
     if ( allocated(iwork)) DEALLOCATE( iwork )
     !
     IF ( info > n ) THEN
        CALL errore( 'rdiaghg', 'S matrix not positive definite', ABS( info ) )
     ELSE IF ( info > 0 ) THEN
        CALL errore( 'rdiaghg', 'eigenvectors failed to converge', ABS( info ) )
     ELSE IF ( info < 0 ) THEN
        CALL errore( 'rdiaghg', 'incorrect call to DSYGV*', ABS( info ) )
     END IF
     
     ! ... restore input S matrix from saved diagonal and lower triangle
     !
     DO i = 1, n
        s(i,i) = sdiag(i)
        DO j = i + 1, n
           s(i,j) = s(j,i)
        END DO
        DO j = n + 1, ldh
           s(j,i) = 0.0_DP
        END DO
     END DO
     !
     DEALLOCATE( sdiag )
     !
  END IF
  !
  ! ... broadcast eigenvectors and eigenvalues to all other processors
  !
  CALL mp_bcast( e, root_bgrp, intra_bgrp_comm )
  CALL mp_bcast( v, root_bgrp, intra_bgrp_comm )
  !
  IF ( me_bgrp == root_bgrp ) THEN
     ! Reinizialize the GPU memory
     call allocateDeviceMemory()
#if defined(__PHIGEMM)
     call initPhigemm()
#endif
  END IF
  !
  CALL stop_clock( 'rdiaghg' )
  !
#endif
  !
  RETURN
  !
END SUBROUTINE rdiaghg_gpu
