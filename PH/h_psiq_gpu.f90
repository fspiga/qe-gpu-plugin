!
! \author Kwanmgin Yu <kyu@bnl.gov>
!
!


!---------------------------
SUBROUTINE h_psiq_k_gpu(lda, n, m, psi, hpsi, spsi)
!---------------------------
  USE kinds,   ONLY : DP
  USE gvecs,   ONLY : nls, ngms
  USE wvfct,  ONLY : g2kin, npwx
  USE fft_base,      ONLY : dffts
  USE wavefunctions_module,  ONLY : psic, psic_nc
  USE becmod, ONLY : bec_type, becp, calbec
  USE noncollin_module, ONLY : noncolin, npol
  USE lsda_mod, ONLY : current_spin
  USE fft_interfaces, ONLY: fwfft, invfft
  USE spin_orb, ONLY : domag
  USE scf,    ONLY : vrs
  USE uspp,   ONLY : vkb
  USE qpoint, ONLY: igkq, nksq
  USE control_flags, ONLY : gamma_only ! Needed only for TDDFPT
  USE mp_pools,  ONLY: npool


  IMPLICIT NONE
  !
  INTEGER, INTENT(in) :: lda, n, m
  !INTEGER, POINTER :: igkq(:)
  COMPLEX(DP), INTENT(INOUT)  :: psi (lda*npol, m)
  COMPLEX(DP), INTENT(OUT) :: hpsi (lda*npol, m), spsi (lda*npol, m)
  !
  INTEGER, EXTERNAL :: h_psiq_cuda_k
  !
  INTEGER :: ierr
  INTEGER :: i,j
  
  
  
  CALL calbec ( n, vkb, psi, becp, m)

  ierr = h_psiq_cuda_k( lda, npol, dffts%nnr, dffts%nr1x, dffts%nr2x, dffts%nr3x, n, m, psi, vrs(:,current_spin), hpsi, igkq(1:), nls(1:), ngms, g2kin)
  

  IF (ierr.EQ.1) THEN
    !CPU fall-back
    write (*,*) 'h_psiq_k_gpu()::ierr : ', ierr
    CALL h_psiq()
  ENDIF
  
  CALL add_vuspsi (lda, n, m, hpsi)

  CALL s_psi (lda, n, m, psi, spsi)
  
  !
  RETURN
END SUBROUTINE h_psiq_k_gpu

