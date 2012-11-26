!
! Copyright (C) 2003-2013 Quantum ESPRESSO group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!----------------------------------------------------------------------------
SUBROUTINE rdiaghg( n, m, h, s, ldh, e, v )
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
#if defined(__CUDA) && defined(__MAGMA)
  USE iso_c_binding
  USE cuda_mem_alloc
#endif
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
#if defined(__CUDA) && defined(__MAGMA)
  INTEGER :: res
  REAL(fp_kind), dimension(:,:), pointer :: vv(:,:)
  TYPE(C_PTR) :: cptr_vv
  INTEGER (C_SIZE_T), PARAMETER :: test_flag = 0
  INTEGER (C_SIZE_T) :: allocation_size
#endif
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
#if defined(__CUDA) && defined(__MAGMA)
#if defined(__PHIGEMM)
     call phigemmShutdown()
#endif
     call deAllocateDeviceMemory()
#endif
     ! ... save the diagonal of input S (it will be overwritten)
     !
     ALLOCATE( sdiag( n ) )
     DO i = 1, n
        sdiag(i) = s(i,i)
     END DO
     !
     all_eigenvalues = ( m == n )
     !
#if defined(__CUDA) && defined(__MAGMA)
     !
     ! ... check for optimal block size -> MAGMA hard-coded value = 32
     nb = 32;
     !
     lwork = 1 + 6 * n * nb + 2* n * n;
     liwork = 5*n + 3
     !
     ALLOCATE ( iwork( liwork) )
     !
#else
     !
     ! ... check for optimal block size
     nb = ILAENV( 1, 'DSYTRD', 'U', n, -1, -1, -1 )
     !
     IF ( nb < 5 .OR. nb >= n ) THEN
        !
        lwork = 8*n
        !
     ELSE
        !
        lwork = ( nb + 3 )*n
        !
     END IF
     !
#endif
     !
     ALLOCATE( work( lwork ) )
     !
     IF ( all_eigenvalues ) THEN
        !
        ! ... calculate all eigenvalues
        !
        v(:,:) = h(:,:)
        !
        !
#if defined(__ESSL)
        !
        ! ... there is a name conflict between essl and lapack ...
        !
        CALL DSYGV( 1, v, ldh, s, ldh, e, v, ldh, n, work, lwork )
        !
        info = 0
#else
        !
#if defined(__CUDA) && defined(__MAGMA)
        !
        !CALL DSYGVD( 1, 'V', 'U', n, v, ldh, s, ldh, e, work, lwork, iwork, liwork, info )
        !
        CALL magmaf_dsygvd( 1, 'V', 'U', n, v, ldh, s, ldh, e, work, lwork, iwork, liwork, info )
        !
#else
        CALL start_clock( 'DSYGV' )
        !
        CALL DSYGV( 1, 'V', 'U', n, v, ldh, s, ldh, e, work, lwork, info )
        !
#endif
        !
#endif
        !
     ELSE
        !
        ! ... calculate only m lowest eigenvalues
        !
#if defined(__CUDA) && defined(__MAGMA)
        !
        !ALLOCATE (vv(ldh, n) )
        allocation_size = ldh*n*sizeof(fp_kind)*4
        res = cudaHostAlloc ( cptr_vv, allocation_size, test_flag )
        CALL c_f_pointer ( cptr_vv, vv, (/ ldh, n /) )
        !
        vv(:,:) = h(:,:)
        !
        CALL magmaf_dsygvd( 1, 'V', 'U', n, vv, ldh, s, ldh, e, work, lwork, iwork, liwork, info )
        !
        !CALL DSYGVD( 1, 'V', 'U', n, vv, ldh, s, ldh, e, work, lwork, iwork, liwork, info )
        !
        v(:,:) = vv(1:m, 1:m)
        ! DEALLOCATE( vv)
        res = cudaFreeHost(cptr_vv)
        !
#else
        !
        ALLOCATE( iwork( 5*n ) )
        ALLOCATE( ifail( n ) )
        !
        ! ... save the diagonal of input H (it will be overwritten)
        !
        ALLOCATE( hdiag( n ) )
        DO i = 1, n
           hdiag(i) = h(i,i)
        END DO
        !
        abstol = 0.D0
       ! abstol = 2.D0*DLAMCH( 'S' )
        !
        CALL DSYGVX( 1, 'V', 'I', 'U', n, h, ldh, s, ldh, &
                     0.D0, 0.D0, 1, m, abstol, mm, e, v, ldh, &
                     work, lwork, iwork, ifail, info )
        !
        DEALLOCATE( ifail )
        DEALLOCATE( iwork )
        !
        ! ... restore input H matrix from saved diagonal and lower triangle
        !
        DO i = 1, n
           h(i,i) = hdiag(i)
           DO j = i + 1, n
              h(i,j) = h(j,i)
           END DO
           DO j = n + 1, ldh
              h(j,i) = 0.0_DP
           END DO
        END DO
        !
        DEALLOCATE( hdiag )
        !
#endif
        !
     END IF
     !
     DEALLOCATE( work )
#if defined(__CUDA) && defined(__MAGMA)
     DEALLOCATE( iwork )
#endif
     !
     CALL errore( 'rdiaghg', 'diagonalization (DSYGV*) failed', ABS( info ) )
     !
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
#if defined(__CUDA) && defined(__MAGMA)
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
#endif
  !
!$OMP END PARALLEL
  !
  CALL stop_clock( 'rdiaghg' )
  !
  RETURN
  !
END SUBROUTINE rdiaghg
!
!----------------------------------------------------------------------------
SUBROUTINE prdiaghg( n, h, s, ldh, e, v, desc )
  !----------------------------------------------------------------------------
  !
  ! ... calculates eigenvalues and eigenvectors of the generalized problem
  ! ... Hv=eSv, with H symmetric matrix, S overlap matrix.
  ! ... On output both matrix are unchanged
  !
  ! ... Parallel version with full data distribution
  !
  USE kinds,            ONLY : DP
  USE mp,               ONLY : mp_bcast
  USE mp_global,        ONLY : root_bgrp, intra_bgrp_comm
  USE descriptors,      ONLY : la_descriptor
#if defined __SCALAPACK
  USE mp_global,        ONLY : ortho_cntx, me_blacs, np_ortho, me_ortho
  USE dspev_module,     ONLY : pdsyevd_drv
#endif
  !
  !
  IMPLICIT NONE
  !
  INTEGER, INTENT(IN) :: n, ldh
    ! dimension of the matrix to be diagonalized and number of eigenstates to be calculated
    ! leading dimension of h, as declared in the calling pgm unit
  REAL(DP), INTENT(INOUT) :: h(ldh,ldh), s(ldh,ldh)
    ! matrix to be diagonalized
    ! overlap matrix
  !
  REAL(DP), INTENT(OUT) :: e(n)
    ! eigenvalues
  REAL(DP), INTENT(OUT) :: v(ldh,ldh)
    ! eigenvectors (column-wise)
  TYPE(la_descriptor), INTENT(IN) :: desc
  !
  INTEGER               :: nx
    ! local block size
  REAL(DP), PARAMETER   :: one = 1_DP
  REAL(DP), PARAMETER   :: zero = 0_DP
  REAL(DP), ALLOCATABLE :: hh(:,:)
  REAL(DP), ALLOCATABLE :: ss(:,:)
#ifdef __SCALAPACK
  INTEGER     :: desch( 16 ), info
#endif
  !
  CALL start_clock( 'rdiaghg' )
  !
  IF( desc%active_node > 0 ) THEN
     !
     nx   = desc%nrcx
     !
     IF( nx /= ldh ) &
        CALL errore(" prdiaghg ", " inconsistent leading dimension ", ldh )
     !
     ALLOCATE( hh( nx, nx ) )
     ALLOCATE( ss( nx, nx ) )
     !
     hh(1:nx,1:nx) = h(1:nx,1:nx)
     ss(1:nx,1:nx) = s(1:nx,1:nx)
     !
  END IF
  !
  CALL start_clock( 'rdiaghg:choldc' )
  !
  ! ... Cholesky decomposition of s ( L is stored in s )
  !
  IF( desc%active_node > 0 ) THEN
     !
#ifdef __SCALAPACK
     CALL descinit( desch, n, n, desc%nrcx, desc%nrcx, 0, 0, ortho_cntx, SIZE( hh, 1 ) , info )
  
     IF( info /= 0 ) CALL errore( ' cdiaghg ', ' descinit ', ABS( info ) )
#endif
     !
#ifdef __SCALAPACK
     CALL PDPOTRF( 'L', n, ss, 1, 1, desch, info )
     IF( info /= 0 ) CALL errore( ' rdiaghg ', ' problems computing cholesky ', ABS( info ) )
#else
     CALL qe_pdpotrf( ss, nx, n, desc )
#endif
     !
  END IF
  !
  CALL stop_clock( 'rdiaghg:choldc' )
  !
  ! ... L is inverted ( s = L^-1 )
  !
  CALL start_clock( 'rdiaghg:inversion' )
  !
  IF( desc%active_node > 0 ) THEN
     !
#ifdef __SCALAPACK
     ! 
     CALL sqr_dsetmat( 'U', n, zero, ss, size(ss,1), desc )

     CALL PDTRTRI( 'L', 'N', n, ss, 1, 1, desch, info )
     !
     IF( info /= 0 ) CALL errore( ' rdiaghg ', ' problems computing inverse ', ABS( info ) )
#else
     CALL qe_pdtrtri ( ss, nx, n, desc )
#endif
     !
  END IF
  !
  CALL stop_clock( 'rdiaghg:inversion' )
  !
  ! ... v = L^-1*H
  !
  CALL start_clock( 'rdiaghg:paragemm' )
  !
  IF( desc%active_node > 0 ) THEN
     !
     CALL sqr_mm_cannon( 'N', 'N', n, ONE, ss, nx, hh, nx, ZERO, v, nx, desc )
     !
  END IF
  !
  ! ... h = ( L^-1*H )*(L^-1)^T
  !
  IF( desc%active_node > 0 ) THEN
     !
     CALL sqr_mm_cannon( 'N', 'T', n, ONE, v, nx, ss, nx, ZERO, hh, nx, desc )
     !
  END IF
  !
  CALL stop_clock( 'rdiaghg:paragemm' )
  !
  IF ( desc%active_node > 0 ) THEN
     ! 
     !  Compute local dimension of the cyclically distributed matrix
     !
#ifdef __SCALAPACK
     CALL pdsyevd_drv( .true., n, desc%nrcx, hh, SIZE(hh,1), e, ortho_cntx )
#else
     CALL qe_pdsyevd( .true., n, desc, hh, SIZE(hh,1), e )
#endif
     !
  END IF
  !
  ! ... v = (L^T)^-1 v
  !
  CALL start_clock( 'rdiaghg:paragemm' )
  !
  IF ( desc%active_node > 0 ) THEN
     !
     CALL sqr_mm_cannon( 'T', 'N', n, ONE, ss, nx, hh, nx, ZERO, v, nx, desc )
     !
     DEALLOCATE( ss )
     DEALLOCATE( hh )
     !
  END IF
  !
  CALL mp_bcast( e, root_bgrp, intra_bgrp_comm )
  !
  CALL stop_clock( 'rdiaghg:paragemm' )
  !
  CALL stop_clock( 'rdiaghg' )
  !
  RETURN
  !
END SUBROUTINE prdiaghg
