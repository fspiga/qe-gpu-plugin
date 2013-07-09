subroutine qvan2_gpu(ngy, ih, jh, np, qmod, qg, ylmk0)
   USE kinds, ONLY: DP
   USE us, ONLY: dq, qrad
   USE uspp_param, ONLY: lmaxq, nbetam
   USE uspp, ONLY: nlx, lpl, lpx, ap, indv, nhtolm

   implicit none
   !
   ! Input variables
   !
   integer,intent(IN) :: ngy, ih, jh, np
   ! ngy   :   number of G vectors to compute
   ! ih, jh:   first and second index of Q
   ! np    :   index of pseudopotentials
   !
   real(DP),intent(IN) :: ylmk0 (ngy, lmaxq * lmaxq), qmod (ngy)
   ! ylmk0 :  spherical harmonics
   ! qmod  :  moduli of the q+g vectors
   !
   ! output: the fourier transform of interest
   !
   real(DP),intent(OUT) :: qg (2,ngy)

   INTEGER, EXTERNAL :: qvan2_cuda
   INTEGER :: error

   error = qvan2_cuda( ngy, ih, jh, np, qmod, qg, ylmk0, size(ylmk0,1), nlx, &
               dq, qrad, size(qrad,1), size(qrad,2), size(qrad,3), indv, &
               size(indv,1), nhtolm, size(nhtolm,1), nbetam, lpx,  &
               size(lpx,1), lpl, size(lpl,1), size(lpl,2), ap, size(ap,1), &
               size(ap,2) )  
   
end subroutine
