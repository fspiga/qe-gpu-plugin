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
#if defined(__WRITE_UNIT_TEST_DATA)
  INTEGER, SAVE :: file_inx = 0
  CHARACTER(LEN=32) :: filename
  LOGICAL :: file_exists
  WRITE(filename, "(A20,I0.3,A4)"), "vloc_psi_gamma_input", file_inx, ".bin"

  INQUIRE(FILE=filename, EXIST=file_exists)
  IF (.not.file_exists) then 
    OPEN(UNIT= 11, STATUS = 'REPLACE', FILE = filename, FORM='UNFORMATTED') 
    !ALLOCATE( aux( ngm, nspin_mag ),  qmod( ngm ), ylmk0( ngm, lmaxq*lmaxq ) )
    PRINT *, "Writing ", filename
    WRITE(11) lda
    WRITE(11) dffts%nnr
    WRITE(11) dffts%nr1x
    WRITE(11) dffts%nr2x
    WRITE(11) dffts%nr3x
    WRITE(11) n
    WRITE(11) m
    WRITE(11) psi
    WRITE(11) v
    WRITE(11) hpsi
    WRITE(11) npwx
    WRITE(11) ngms
    WRITE(11) ngm
    WRITE(11) igk(1:)
    WRITE(11) nls(1:)
    WRITE(11) nlsm(1:)

    CLOSE(11)
  endif
#endif
#if defined(__CUDA_MULTIPLAN_FFT)
  ierr =   vloc_psi_multiplan_cuda ( lda, dffts%nnr, dffts%nr1x, dffts%nr2x, dffts%nr3x, &
    n, m, psi, v, hpsi, igk(1:), nls(1:), nlsm(1:), ngms, ngm)
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
#if defined(__WRITE_UNIT_TEST_DATA)
  INTEGER, SAVE :: file_inx = 0
  CHARACTER(LEN=32) :: filename
  LOGICAL :: file_exists
  WRITE(filename, "(A16,I0.3,A4)"), "vloc_psi_k_input", file_inx, ".bin"

  INQUIRE(FILE=filename, EXIST=file_exists)
  IF (.not.file_exists) then 
    OPEN(UNIT= 11, STATUS = 'REPLACE', FILE = filename, FORM='UNFORMATTED') 
    !ALLOCATE( aux( ngm, nspin_mag ),  qmod( ngm ), ylmk0( ngm, lmaxq*lmaxq ) )
    PRINT *, "Writing ", filename
    WRITE(11) lda
    WRITE(11) dffts%nnr
    WRITE(11) dffts%nr1x
    WRITE(11) dffts%nr2x
    WRITE(11) dffts%nr3x
    WRITE(11) n
    WRITE(11) m
    WRITE(11) psi
    WRITE(11) v
    WRITE(11) hpsi
    WRITE(11) npwx
    WRITE(11) ngms
    WRITE(11) ngms
    WRITE(11) igk(1:)
    WRITE(11) nls(1:)

    CLOSE(11)
  endif
#endif
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
