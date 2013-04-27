!
! Copyright (C) 2003-2009 PWSCF group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
!-----------------------------------------------------------------------
SUBROUTINE vloc_psi_gamma_gpu(lda, n, m, psi, v, hpsi)
  !-----------------------------------------------------------------------
  !
  ! Calculation of Vloc*psi using dual-space technique - Gamma point
  !
  USE kinds,   ONLY : DP
  USE gvecs,   ONLY : nls, nlsm, ngms
  USE gvect,   ONLY : ngm
  USE wvfct,   ONLY : igk, npwx
  USE fft_base,      ONLY : dffts
  USE wavefunctions_module,  ONLY: psic
  !
  IMPLICIT NONE
  !
  INTEGER, INTENT(in) :: lda, n, m
  COMPLEX(DP), INTENT(in)   :: psi (lda, m)
  COMPLEX(DP), INTENT(inout):: hpsi (lda, m)
  REAL(DP), INTENT(in) :: v(dffts%nnr)
  !
#if defined(__CUDA_MULTIPLAN_FFT)
  INTEGER, EXTERNAL :: vloc_psi_multiplan_cuda
#else
  INTEGER, EXTERNAL :: vloc_psi_cuda
#endif
  !
  INTEGER :: ierr
  !
#if defined(__CUDA_MULTIPLAN_FFT)
  ierr =   vloc_psi_multiplan_cuda ( lda, dffts%nnr, dffts%nr1x, dffts%nr2x, dffts%nr3x, &
    n, m, psi, v), hpsi, igk(1:), nls(1:), nlsm(1:), ngms, ngm)
#else
  ierr = vloc_psi_cuda ( lda, dffts%nnr, dffts%nr1x, dffts%nr2x, dffts%nr3x, &
     n, m, psi, v, hpsi, igk(1:), nls(1:), nlsm(1:), ngms, ngm)
#endif
  !
  IF (ierr .EQ. 1) THEN
     ! CPU fall-back
     CALL vloc_psi_gamma ( lda, n, m, psi, v, hpsi )
  ENDIF
  !
  RETURN
END SUBROUTINE vloc_psi_gamma_gpu
!
!-----------------------------------------------------------------------
SUBROUTINE vloc_psi_k_gpu(lda, n, m, psi, v, hpsi)
  !-----------------------------------------------------------------------
  !
  ! Calculation of Vloc*psi using dual-space technique - k-points
  !
  USE kinds,   ONLY : DP
  USE gvecs,   ONLY : nls, ngms
  USE wvfct,   ONLY : igk, npwx
  USE fft_base,      ONLY : dffts
  USE wavefunctions_module,  ONLY: psic
  !
  IMPLICIT NONE
  !
  INTEGER, INTENT(in) :: lda, n, m
  COMPLEX(DP), INTENT(in)   :: psi (lda, m)
  COMPLEX(DP), INTENT(inout):: hpsi (lda, m)
  REAL(DP), INTENT(in) :: v(dffts%nnr)
  !
!#if defined(__CUDA_MULTIPLAN_FFT)
!  INTEGER, EXTERNAL :: vloc_psi_multiplan_cuda_k
!#else
  INTEGER, EXTERNAL :: vloc_psi_cuda_k
!#endif
  !
  INTEGER :: ierr
  !
! __CUDA_MULTIPLAN_FFT broken?
!#if defined(__CUDA_MULTIPLAN_FFT)
!  CALL vloc_psi_multiplan_cuda_k ( lda, dffts%nnr, dffts%nr1x, dffts%nr2x, dffts%nr3x, &
!     n, m, psi, vrs(1,current_spin), hpsi, igk(1:), nls(1:), ngms)
!#else
  ierr = vloc_psi_cuda_k ( lda, dffts%nnr, dffts%nr1x, dffts%nr2x, dffts%nr3x, &
    n, m, psi, v, hpsi, igk(1:), nls(1:), ngms)
!#endif
  !
  IF (ierr .EQ. 1) THEN
     ! CPU fall-back
     CALL vloc_psi_k ( lda, n, m, psi, v, hpsi )
  ENDIF
  !
  RETURN
END SUBROUTINE vloc_psi_k_gpu
