! Copyright (C) 2001-2014 Quantum ESPRESSO Foundation
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
#if defined(__CUDA) && defined(__PHIGEMM)
#define dgemm UDGEMM
#define zgemm UZGEMM
#define DGEMM UDGEMM
#define ZGEMM UZGEMM
#if defined(__PHIGEMM_PROFILE)
#define _STRING_LINE_(s) #s
#define _STRING_LINE2_(s) _STRING_LINE_(s)
#define __LINESTR__ _STRING_LINE2_(__LINE__)
#define UDGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC) phidgemm(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC,__FILE__,__LINESTR__)
#define UZGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC) phizgemm(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC,__FILE__,__LINESTR__)
#else
#define UDGEMM phidgemm
#define UZGEMM phizgemm
#endif
#endif
!
#define ZERO ( 0.D0, 0.D0 )
#define ONE  ( 1.D0, 0.D0 )
!
!
!----------------------------------------------------------------------------
SUBROUTINE regterg_gpu( npw, npwx, nvec, nvecx, evc, ethr, &
                    uspp, gstart, e, btype, notcnv, lrot, dav_iter )
  !----------------------------------------------------------------------------
  !
  ! ... iterative solution of the eigenvalue problem:
  !
  ! ... ( H - e S ) * evc = 0
  !
  ! ... where H is an hermitean operator, e is a real scalar,
  ! ... S is an uspp matrix, evc is a complex vector
  ! ... (real wavefunctions with only half plane waves stored)
  !
  USE kinds,         ONLY : DP
  USE io_global,     ONLY : stdout
  USE mp_bands,      ONLY : intra_bgrp_comm
  USE mp,            ONLY : mp_sum 
  !
#if defined(__CUDA) && defined(__CUDA_MEM_PINNED)
  USE iso_c_binding
  USE cuda_mem_alloc
#endif
  !
  IMPLICIT NONE
  !
#if defined(__CUDA) && defined(__CUDA_MEM_PINNED)
  INTEGER :: res
#endif
  !
  INTEGER, INTENT(IN) :: npw, npwx, nvec, nvecx, gstart
    ! dimension of the matrix to be diagonalized
    ! leading dimension of matrix evc, as declared in the calling pgm unit
    ! integer number of searched low-lying roots
    ! maximum dimension of the reduced basis set
    !    (the basis set is refreshed when its dimension would exceed nvecx)
  COMPLEX(DP), INTENT(INOUT) :: evc(npwx,nvec)
    !  evc   contains the  refined estimates of the eigenvectors
  REAL(DP), INTENT(IN) :: ethr
    ! energy threshold for convergence: root improvement is stopped,
    ! when two consecutive estimates of the root differ by less than ethr.
  LOGICAL, INTENT(IN) :: uspp
    ! if .FALSE. : S|psi> not needed
  INTEGER, INTENT(IN) :: btype(nvec)
    ! band type ( 1 = occupied, 0 = empty )
  LOGICAL, INTENT(IN) :: lrot
    ! .TRUE. if the wfc have already been rotated
  REAL(DP), INTENT(OUT) :: e(nvec)
    ! contains the estimated roots.
  INTEGER, INTENT(OUT) :: dav_iter, notcnv
    ! integer  number of iterations performed
    ! number of unconverged roots
  !
  ! ... LOCAL variables
  !
  INTEGER, PARAMETER :: maxter = 20
    ! maximum number of iterations
  !
  INTEGER :: kter, nbase, np, n, m, nb1, ibnd
    ! counter on iterations
    ! dimension of the reduced basis
    ! counter on the reduced basis vectors
    ! do-loop counters
    ! counter on the bands
  INTEGER :: ierr
#if defined(__CUDA) && defined(__CUDA_MEM_PINNED)
  REAL(fp_kind), DIMENSION(:,:), POINTER :: hr(:,:), sr(:,:), vr(:,:)
  REAL(fp_kind), DIMENSION(:), POINTER :: ew(:)
#else
  REAL(DP), ALLOCATABLE :: hr(:,:), sr(:,:), vr(:,:), ew(:)
#endif
    ! Hamiltonian on the reduced basis
    ! S matrix on the reduced basis
    ! eigenvectors of the Hamiltonian
    ! eigenvalues of the reduced hamiltonian
#if defined(__CUDA) && defined(__CUDA_MEM_PINNED)
  COMPLEX(fp_kind), DIMENSION(:,:), POINTER :: hpsi(:,:), spsi(:,:), psi(:,:)
#else
  COMPLEX(DP), ALLOCATABLE :: psi(:,:), hpsi(:,:), spsi(:,:)
#endif    
    ! work space, contains psi
    ! the product of H and psi
    ! the product of S and psi
  LOGICAL, ALLOCATABLE :: conv(:)
    ! true if the root is converged
  REAL(DP) :: empty_ethr 
    ! threshold for empty bands
  INTEGER :: npw2, npwx2
  !
  REAL(DP), EXTERNAL :: ddot
  !
  ! EXTERNAL  h_psi, s_psi, g_psi
    ! h_psi(npwx,npw,nvec,psi,hpsi)
    !     calculates H|psi> 
    ! s_psi(npwx,npw,nvec,psi,spsi)
    !     calculates S|psi> (if needed)
    !     Vectors psi,hpsi,spsi are dimensioned (npwx,nvec)
    ! g_psi(npwx,npw,notcnv,psi,e)
    !    calculates (diag(h)-e)^-1 * psi, diagonal approx. to (h-e)^-1*psi
    !    the first nvec columns contain the trial eigenvectors
  !
#if defined(__CUDA) && defined(__CUDA_MEM_PINNED)
  TYPE(C_PTR) :: cptr_psi, cptr_spsi, cptr_vr, cptr_hpsi, cptr_sr, cptr_ew, cptr_hr
  INTEGER (C_SIZE_T), PARAMETER :: test_flag = 0
  INTEGER (C_SIZE_T) :: allocation_size
#endif
  !
#if defined(__CUDA_DEBUG)
  WRITE(*,*) "[REGTERG] Enter"
#endif
  !
  CALL start_clock( 'regterg' )
  !
  IF ( nvec > nvecx / 2 ) CALL errore( 'regter', 'nvecx is too small', 1 )
  !
  ! ... threshold for empty bands
  !
  empty_ethr = MAX( ( ethr * 5.D0 ), 1.D-5 )
  !
#if defined(__CUDA) && defined(__CUDA_MEM_PINNED)
  allocation_size = npwx*nvecx*sizeof(fp_kind)*4
  res = cudaHostAlloc ( cptr_psi, allocation_size, test_flag )
  CALL c_f_pointer ( cptr_psi, psi, (/ npwx, nvecx /) )
  res = cudaHostAlloc ( cptr_hpsi, allocation_size, test_flag )
  CALL c_f_pointer ( cptr_hpsi, hpsi, (/ npwx, nvecx /) )
#else
  ALLOCATE( psi(  npwx, nvecx ), STAT=ierr )
  IF( ierr /= 0 ) &
     CALL errore( 'regterg ',' cannot allocate psi ', ABS(ierr) )
  ALLOCATE( hpsi( npwx, nvecx ), STAT=ierr )
  IF( ierr /= 0 ) &
     CALL errore( 'regterg ',' cannot allocate hpsi ', ABS(ierr) )
#endif
  !
  IF ( uspp ) THEN
#if defined(__CUDA) && defined(__CUDA_MEM_PINNED)
    res = cudaHostAlloc ( cptr_spsi, allocation_size, test_flag )
    CALL c_f_pointer ( cptr_spsi, spsi, (/ npwx, nvecx /) )
#else
    ALLOCATE( spsi( npwx, nvecx ), STAT=ierr )
    IF( ierr /= 0 ) &
       CALL errore( ' regterg ',' cannot allocate spsi ', ABS(ierr) )
#endif
  END IF
  !
#if defined(__CUDA) && defined(__CUDA_MEM_PINNED)
  allocation_size = nvecx*nvecx*sizeof(fp_kind)*2
  res = cudaHostAlloc ( cptr_sr, allocation_size, test_flag )
  CALL c_f_pointer ( cptr_sr, sr, (/ nvecx, nvecx /) )
  res = cudaHostAlloc ( cptr_hr, allocation_size, test_flag )
  CALL c_f_pointer ( cptr_hr, hr, (/ nvecx, nvecx /) )
  res = cudaHostAlloc ( cptr_vr, allocation_size, test_flag )
  CALL c_f_pointer ( cptr_vr, vr, (/ nvecx, nvecx /) )
  allocation_size = nvecx*sizeof(fp_kind)*2
  res = cudaHostAlloc ( cptr_ew, allocation_size, test_flag )
  CALL c_f_pointer ( cptr_ew, ew, (/ nvecx /) )
#else
  ALLOCATE( sr( nvecx, nvecx ), STAT=ierr )
  IF( ierr /= 0 ) &
     CALL errore( 'regterg ',' cannot allocate sr ', ABS(ierr) )
  ALLOCATE( hr( nvecx, nvecx ), STAT=ierr )
  IF( ierr /= 0 ) &
     CALL errore( 'regterg ',' cannot allocate hr ', ABS(ierr) )
  ALLOCATE( vr( nvecx, nvecx ), STAT=ierr )
  IF( ierr /= 0 ) &
     CALL errore( 'regterg ',' cannot allocate vr ', ABS(ierr) )
  ALLOCATE( ew( nvecx ), STAT=ierr )
  IF( ierr /= 0 ) &
     CALL errore( 'regterg ',' cannot allocate ew ', ABS(ierr) )
#endif
  ALLOCATE( conv( nvec ), STAT=ierr )
  IF( ierr /= 0 ) &
     CALL errore( 'regterg ',' cannot allocate conv ', ABS(ierr) )
  !
  npw2  = 2*npw
  npwx2  = 2*npwx
  notcnv = nvec
  nbase  = nvec
  conv   = .FALSE.
  !
  IF ( uspp ) spsi = ZERO
  !
  hpsi = ZERO
  psi  = ZERO
  psi(:,1:nvec) = evc(:,1:nvec)
  ! ... set Im[ psi(G=0) ] -  needed for numerical stability
  IF ( gstart == 2 ) psi(1,1:nvec) = CMPLX( DBLE( psi(1,1:nvec) ), 0.D0 ,kind=DP)
  !
  ! ... hpsi contains h times the basis vectors
  !
  CALL h_psi( npwx, npw, nvec, psi, hpsi )
  !
  IF ( uspp ) CALL s_psi( npwx, npw, nvec, psi, spsi )
  !
  ! ... hr contains the projection of the hamiltonian onto the reduced
  ! ... space vr contains the eigenvectors of hr
  !
  hr(:,:) = 0.D0
  sr(:,:) = 0.D0
  vr(:,:) = 0.D0
  !
  CALL DGEMM( 'T', 'N', nbase, nbase, npw2, 2.D0 , &
              psi, npwx2, hpsi, npwx2, 0.D0, hr, nvecx )
  !
  IF ( gstart == 2 ) &
     CALL DGER( nbase, nbase, -1.D0, psi, npwx2, hpsi, npwx2, hr, nvecx )
  !
  CALL mp_sum( hr( :, 1:nbase ), intra_bgrp_comm )
  !
  IF ( uspp ) THEN
     !
     CALL DGEMM( 'T', 'N', nbase, nbase, npw2, 2.D0, &
                 psi, npwx2, spsi, npwx2, 0.D0, sr, nvecx )
     !
     IF ( gstart == 2 ) &
        CALL DGER( nbase, nbase, -1.D0, psi, npwx2, spsi, npwx2, sr, nvecx )
     !
  ELSE
     !
     CALL DGEMM( 'T', 'N', nbase, nbase, npw2, 2.D0, &
                 psi, npwx2, psi, npwx2, 0.D0, sr, nvecx )
     !
     IF ( gstart == 2 ) &
        CALL DGER( nbase, nbase, -1.D0, psi, npwx2, psi, npwx2, sr, nvecx )
     !
  END IF
  !
  CALL mp_sum( sr( :, 1:nbase ), intra_bgrp_comm )
  !
  IF ( lrot ) THEN
     !
     DO n = 1, nbase
        !
        e(n) = hr(n,n)
        vr(n,n) = 1.D0
        !
     END DO
     !
  ELSE
     !
     ! ... diagonalize the reduced hamiltonian
     !
     CALL rdiaghg( nbase, nvec, hr, sr, nvecx, ew, vr )
     !
     e(1:nvec) = ew(1:nvec)
     !
  END IF
  !
  ! ... iterate
  !
  iterate: DO kter = 1, maxter
     !
     dav_iter = kter
     !
     CALL start_clock( 'regterg:update' )
     !
     np = 0
     !
     DO n = 1, nvec
        !
        IF ( .NOT. conv(n) ) THEN
           !
           ! ... this root not yet converged ... 
           !
           np = np + 1
           !
           ! ... reorder eigenvectors so that coefficients for unconverged
           ! ... roots come first. This allows to use quick matrix-matrix 
           ! ... multiplications to set a new basis vector (see below)
           !
           IF ( np /= n ) vr(:,np) = vr(:,n)
           !
           ! ... for use in g_psi
           !
           ew(nbase+np) = e(n)
           !   
        END IF
        !
     END DO
     !
     nb1 = nbase + 1
     !
     ! ... expand the basis set with new basis vectors ( H - e*S )|psi> ...
     !
     IF ( uspp ) THEN
        !
#if defined(__IMPROVED_GEMM)
        IF ( notcnv == 1 ) THEN
           !
           CALL DGEMV( 'N', npw2, nbase, 1.0_DP, spsi, npwx2, vr, 1, 0.0_DP, &
                     psi(1,nb1), 1 )
           !
        ELSE
           !
           CALL DGEMM( 'N', 'N', npw2, notcnv, nbase, 1.D0, &
                    spsi, npwx2, vr, nvecx, 0.D0, psi(1,nb1), npwx2 )
           !
        ENDIF
        !
#else
        ! ORIGINAL:
        CALL DGEMM( 'N', 'N', npw2, notcnv, nbase, 1.D0, &
            spsi, npwx2, vr, nvecx, 0.D0, psi(1,nb1), npwx2 )
#endif
        !
     ELSE
        !
#if defined(__IMPROVED_GEMM)
        IF ( notcnv == 1 ) THEN
           !
           CALL DGEMV( 'N', npw2, nbase, 1.0_DP, psi, npwx2, vr, 1, 0.0_DP, &
                     psi(1,nb1), 1 )
           !
        ELSE
           !
        CALL DGEMM( 'N', 'N', npw2, notcnv, nbase, 1.D0, &
                    psi, npwx2, vr, nvecx, 0.D0, psi(1,nb1), npwx2 )
           !
        ENDIF
        !
#else
        ! ORIGINAL:
        CALL DGEMM( 'N', 'N', npw2, notcnv, nbase, 1.D0, &
                    psi, npwx2, vr, nvecx, 0.D0, psi(1,nb1), npwx2 )
#endif
        !
     END IF
     !
     DO np = 1, notcnv
        !
        psi(:,nbase+np) = - ew(nbase+np) * psi(:,nbase+np)
        !
     END DO
     !
#if defined(__IMPROVED_GEMM)
     IF ( notcnv == 1 ) THEN
        !
        CALL DGEMV( 'N', npw2, nbase, 1.0_DP, hpsi, npwx2, vr, 1, 1.0_DP, &
             psi(1,nb1), 1 )
        !
     ELSE
        !
        CALL DGEMM( 'N', 'N', npw2, notcnv, nbase, 1.D0, &
                 hpsi, npwx2, vr, nvecx, 1.D0, psi(1,nb1), npwx2 )
        !
     ENDIF
     !
#else
     ! ORIGINAL:
     CALL DGEMM( 'N', 'N', npw2, notcnv, nbase, 1.D0, &
                 hpsi, npwx2, vr, nvecx, 1.D0, psi(1,nb1), npwx2 )
#endif
     !
     CALL stop_clock( 'regterg:update' )
     !
     ! ... approximate inverse iteration
     !
     CALL g_psi( npwx, npw, notcnv, 1, psi(1,nb1), ew(nb1) )
     !
     ! ... "normalize" correction vectors psi(:,nb1:nbase+notcnv) in 
     ! ... order to improve numerical stability of subspace diagonalization 
     ! ... (rdiaghg) ew is used as work array :
     !
     ! ...         ew = <psi_i|psi_i>,  i = nbase + 1, nbase + notcnv
     !
     DO n = 1, notcnv
        !
        ew(n) = 2.D0 * ddot( npw2, psi(1,nbase+n), 1, psi(1,nbase+n), 1 )
        !
        IF ( gstart == 2 ) ew(n) = ew(n) - psi(1,nbase+n) * psi(1,nbase+n)
        !
     END DO
     !
     CALL mp_sum( ew( 1:notcnv ), intra_bgrp_comm )
     !
     DO n = 1, notcnv
        !
        psi(:,nbase+n) = psi(:,nbase+n) / SQRT( ew(n) )
        ! ... set Im[ psi(G=0) ] -  needed for numerical stability
        IF ( gstart == 2 ) psi(1,nbase+n) = CMPLX( DBLE(psi(1,nbase+n)), 0.D0 ,kind=DP)
        !
     END DO
     !
     ! ... here compute the hpsi and spsi of the new functions
     !
     CALL h_psi( npwx, npw, notcnv, psi(1,nb1), hpsi(1,nb1) )
     !
     IF ( uspp ) CALL s_psi( npwx, npw, notcnv, psi(1,nb1), spsi(1,nb1) )
     !
     ! ... update the reduced hamiltonian
     !
     CALL start_clock( 'regterg:overlap' )
     !
#if defined(__IMPROVED_GEMM)
     IF ( notcnv == 1 ) THEN
        !
        CALL DGEMV( 'T', npw2, nbase+notcnv, 2.0_DP, psi, npwx2, hpsi(1,nb1), 1, 0.0_DP, &
                     hr(1,nb1), 1 )
        !
     ELSE
        !
        CALL DGEMM( 'T', 'N', nbase+notcnv, notcnv, npw2, 2.D0, psi, &
                 npwx2, hpsi(1,nb1), npwx2, 0.D0, hr(1,nb1), nvecx )
        !
     ENDIF
     !
#else
     ! ORIGINAL:
     CALL DGEMM( 'T', 'N', nbase+notcnv, notcnv, npw2, 2.D0, psi, &
           npwx2, hpsi(1,nb1), npwx2, 0.D0, hr(1,nb1), nvecx )
#endif
     !
     IF ( gstart == 2 ) &
        CALL DGER( nbase+notcnv, notcnv, -1.D0, psi, &
                   npwx2, hpsi(1,nb1), npwx2, hr(1,nb1), nvecx )
     !
     CALL mp_sum( hr( :, nb1 : nb1+notcnv-1 ), intra_bgrp_comm )
     !
     IF ( uspp ) THEN
        !
#if defined(__IMPROVED_GEMM)
        IF ( notcnv == 1 ) THEN
           !
           CALL DGEMV( 'T', npw2, nbase+notcnv, 2.0_DP, psi, npwx2, spsi(1,nb1), 1, 0.0_DP, &
                     sr(1,nb1), 1 )
           !
        ELSE
           !
           CALL DGEMM( 'T', 'N', nbase+notcnv, notcnv, npw2, 2.D0, psi, &
                    npwx2, spsi(1,nb1), npwx2, 0.D0, sr(1,nb1), nvecx )
           !
        ENDIF
#else
        ! ORIGINAL:
        CALL DGEMM( 'T', 'N', nbase+notcnv, notcnv, npw2, 2.D0, psi, &
                npwx2, spsi(1,nb1), npwx2, 0.D0, sr(1,nb1), nvecx )
#endif
        !
        IF ( gstart == 2 ) &
           CALL DGER( nbase+notcnv, notcnv, -1.D0, psi, &
                      npwx2, spsi(1,nb1), npwx2, sr(1,nb1), nvecx )
        !
     ELSE
        !
#if defined(__IMPROVED_GEMM)
        IF ( notcnv == 1 ) THEN
           !
           CALL DGEMV( 'T', npw2, nbase+notcnv, 2.0_DP, psi, npwx2, psi(1,nb1), 1, 0.0_DP, &
                     sr(1,nb1), 1 )
           !
        ELSE
           !
           CALL DGEMM( 'T', 'N', nbase+notcnv, notcnv, npw2, 2.D0, psi, &
                    npwx2, psi(1,nb1), npwx2, 0.D0, sr(1,nb1) , nvecx )
           !
        ENDIF
        !
#else
        ! ORIGINAL:
        CALL DGEMM( 'T', 'N', nbase+notcnv, notcnv, npw2, 2.D0, psi, &
                npwx2, psi(1,nb1), npwx2, 0.D0, sr(1,nb1) , nvecx )
#endif
        !
        IF ( gstart == 2 ) &
           CALL DGER( nbase+notcnv, notcnv, -1.D0, psi, &
                      npwx2, psi(1,nb1), npwx2, sr(1,nb1), nvecx )
        !
     END IF
     !
     CALL mp_sum( sr( :, nb1 : nb1+notcnv-1 ), intra_bgrp_comm  )
     !
     CALL stop_clock( 'regterg:overlap' )
     !
     nbase = nbase + notcnv
     !
     DO n = 1, nbase
        !
        DO m = n + 1, nbase
           !
           hr(m,n) = hr(n,m)
           sr(m,n) = sr(n,m)
           !
        END DO
        !
     END DO
     !
     ! ... diagonalize the reduced hamiltonian
     !
     CALL rdiaghg( nbase, nvec, hr, sr, nvecx, ew, vr )
     !
     ! ... test for convergence
     !
     WHERE( btype(1:nvec) == 1 )
        !
        conv(1:nvec) = ( ( ABS( ew(1:nvec) - e(1:nvec) ) < ethr ) )
        !
     ELSEWHERE
        !
        conv(1:nvec) = ( ( ABS( ew(1:nvec) - e(1:nvec) ) < empty_ethr ) )
        !
     END WHERE
     !
     notcnv = COUNT( .NOT. conv(:) )
     !
     e(1:nvec) = ew(1:nvec)
     !
     ! ... if overall convergence has been achieved, or the dimension of
     ! ... the reduced basis set is becoming too large, or in any case if
     ! ... we are at the last iteration refresh the basis set. i.e. replace
     ! ... the first nvec elements with the current estimate of the
     ! ... eigenvectors;  set the basis dimension to nvec.
     !
     IF ( notcnv == 0 .OR. &
          nbase+notcnv > nvecx .OR. dav_iter == maxter ) THEN
        !
        CALL start_clock( 'regterg:last' )
        !
        CALL DGEMM( 'N', 'N', npw2, nvec, nbase, 1.D0, &
                    psi, npwx2, vr, nvecx, 0.D0, evc, npwx2 )
        !
        IF ( notcnv == 0 ) THEN
           !
           ! ... all roots converged: return
           !
           CALL stop_clock( 'regterg:last' )
           !
           EXIT iterate
           !
        ELSE IF ( dav_iter == maxter ) THEN
           !
           ! ... last iteration, some roots not converged: return
           !
           WRITE( stdout, '(5X,"WARNING: ",I5, &
                &   " eigenvalues not converged in regterg")' ) notcnv
           !
           CALL stop_clock( 'regterg:last' )
           !
           EXIT iterate
           !
        END IF
        !
        ! ... refresh psi, H*psi and S*psi
        !
        psi(:,1:nvec) = evc(:,1:nvec)
        !
        IF ( uspp ) THEN
           !
           CALL DGEMM( 'N', 'N', npw2, nvec, nbase, 1.D0, spsi, &
                       npwx2, vr, nvecx, 0.D0, psi(1,nvec+1), npwx2 )
           !
           spsi(:,1:nvec) = psi(:,nvec+1:nvec+nvec)
           !
        END IF
        !
        CALL DGEMM( 'N', 'N', npw2, nvec, nbase, 1.D0, hpsi, &
                    npwx2, vr, nvecx, 0.D0, psi(1,nvec+1), npwx2 )
        !
        hpsi(:,1:nvec) = psi(:,nvec+1:nvec+nvec)
        !
        ! ... refresh the reduced hamiltonian
        !
        nbase = nvec
        !
        hr(:,1:nbase) = 0.D0
        sr(:,1:nbase) = 0.D0
        vr(:,1:nbase) = 0.D0
        !
        DO n = 1, nbase
           !
           hr(n,n) = e(n)
           sr(n,n) = 1.D0
           vr(n,n) = 1.D0
           !
        END DO
        !
        CALL stop_clock( 'regterg:last' )
        !
     END IF
     !
  END DO iterate
  !
  DEALLOCATE( conv )
#if defined(__CUDA) && defined(__CUDA_MEM_PINNED)
  res = cudaFreeHost(cptr_ew)
  res = cudaFreeHost(cptr_vr)
  res = cudaFreeHost(cptr_hr)
  res = cudaFreeHost(cptr_sr)
#else
  DEALLOCATE( ew )
  DEALLOCATE( vr )
  DEALLOCATE( hr )
  DEALLOCATE( sr )
#endif
  !
  IF ( uspp ) THEN
#if defined(__CUDA) && defined(__CUDA_MEM_PINNED)
     res = cudaFreeHost(cptr_spsi)
#else
     DEALLOCATE( spsi )
#endif
  END IF
  !
#if defined(__CUDA) && defined(__CUDA_MEM_PINNED)
  res = cudaFreeHost( cptr_hpsi )
  res = cudaFreeHost( cptr_psi )
#else
  DEALLOCATE( hpsi )
  DEALLOCATE( psi )   
#endif
  !
  CALL stop_clock( 'regterg' )
  !
  RETURN
  !
END SUBROUTINE regterg_gpu
