! Copyright (C) 2001-2014 Quantum ESPRESSO Foundation
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
PROGRAM main
  USE kinds,     ONLY : DP
  IMPLICIT NONE

  INTEGER :: lda, n, m, nnr, nr1x, nr2x, nr3x, ngms, ngm, npwx, ierr
  COMPLEX(DP), ALLOCATABLE :: psi(:,:)
  COMPLEX(DP), ALLOCATABLE :: hpsi(:,:)
  REAL(DP), ALLOCATABLE :: v(:)

  INTEGER, EXTERNAL :: vloc_psi_cuda_k
#if defined(__CUDA_MULTIPLAN_FFT)
  INTEGER, EXTERNAL :: vloc_psi_multiplan_cuda
#else
  INTEGER, EXTERNAL :: vloc_psi_cuda
#endif

  INTEGER, ALLOCATABLE :: nls(:), nlsm(:), igk(:)
  CHARACTER(LEN=32) :: filename
  LOGICAL :: file_exists

  WRITE(filename, "(A16,I0.3,A4)"), "vloc_psi_k_input", 0, ".bin"
  INQUIRE(FILE=filename, EXIST=file_exists)
  IF (.not.file_exists) then
    PRINT *, "Could not find input file $", filename, "$"
    RETURN
  ENDIF
  
  CALL InitCudaEnv()
  PRINT *," Opening file ", filename
  OPEN(UNIT=11, STATUS = 'OLD', FILE = filename, FORM='UNFORMATTED')
  READ (11) lda
  READ (11) nnr
  READ (11) nr1x
  READ (11) nr2x
  READ (11) nr3x
  READ (11) n
  READ (11) m
  ALLOCATE(psi(lda,m))
  READ (11) psi

  ALLOCATE(v(nnr))
  READ (11) v

  ALLOCATE(hpsi(lda,m))
  READ (11) hpsi

  READ (11) npwx
  READ (11) ngms
  READ (11) ngm
  ALLOCATE(igk(npwx))
  READ (11) igk

  ALLOCATE(nls(ngms))
  READ (11) nls

  CLOSE (11)

! __CUDA_MULTIPLAN_FFT broken?
!#if defined(__CUDA_MULTIPLAN_FFT)
!  CALL vloc_psi_multiplan_cuda_k ( lda, dffts%nnr, dffts%nr1x, dffts%nr2x, dffts%nr3x, &
!     n, m, psi, vrs(1,current_spin), hpsi, igk(1:), nls(1:), ngms)
!#else
  ierr = vloc_psi_cuda_k ( lda, nnr, nr1x, nr2x, nr3x, &
    n, m, psi, v, hpsi, igk(1:), nls(1:), ngms)
!#endif
! Below if for gamma
!#if defined(__CUDA_MULTIPLAN_FFT)
!  ierr =   vloc_psi_multiplan_cuda ( lda, nnr, nr1x, nr2x, nr3x, &
!    n, m, psi, v, hpsi, igk(1:), nls(1:), nlsm(1:), ngms, ngm)
!#else
!  ierr = vloc_psi_cuda ( lda, nnr, nr1x, nr2x, nr3x, &
!     n, m, psi, v, hpsi, igk(1:), nls(1:), nlsm(1:), ngms, ngm)
!#endif

END program
