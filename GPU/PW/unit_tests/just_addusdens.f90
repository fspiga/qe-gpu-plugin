! Copyright (C) 2001-2014 Quantum ESPRESSO Foundation
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
program main
   USE kinds,                ONLY : DP
   USE uspp,                 ONLY : nlx, lpl, lpx, ap, indv, nhtolm
   USE uspp_param,           ONLY : nbetam, lmaxq, nhm
   USE us,                   ONLY : dq, qrad

   CHARACTER(LEN=32) :: filename  = "addusdens_input001.bin"
   INTEGER :: nr1, nr2, nr3, first_becsum, nat, nt, ngm
   INTEGER, ALLOCATABLE :: nh(:)
   DOUBLE PRECISION, ALLOCATABLE :: qmod(:), ylmk0(:,:)
   COMPLEX(DP), ALLOCATABLE :: qgm(:)
   COMPLEX(DP), ALLOCATABLE :: eigts1(:)
   COMPLEX(DP), ALLOCATABLE :: eigts2(:)
   COMPLEX(DP), ALLOCATABLE :: eigts3(:)
   INTEGER, ALLOCATABLE :: mill(:,:)
   INTEGER :: ityp
   COMPLEX(DP), ALLOCATABLE :: aux(:,:)
   DOUBLE PRECISION, ALLOCATABLE :: becsum(:,:,:)
   INTEGER :: nspin
   INTEGER :: nspin_mag

   INTEGER :: sizeA, sizeB, sizeC, sizeD

   INTEGER, EXTERNAL :: addusdens_cuda
   INTEGER :: err



   CALL InitCudaEnv()
   OPEN(UNIT= 11, STATUS = 'OLD', FILE = filename, FORM='UNFORMATTED') 
    ALLOCATE(nh(10))
    READ(11) nr1
    READ(11) nr2
    READ(11) nr3
    READ(11) first_becsum
    READ(11) nat
    READ(11) nh
    READ(11) nt
    READ(11) ngm
    READ(11) lmaxq
    READ(11) nspin_mag
    READ(11) nspin

    ALLOCATE(aux(ngm,nspin_mag))
    ALLOCATE(qmod(ngm))
    ALLOCATE(qgm(ngm))
    ALLOCATE(ylmk0(ngm,lmaxq*lmaxq))
    ALLOCATE(eigts1((nr1*2+1)*nat))
    ALLOCATE(eigts2((nr2*2+1)*nat))
    ALLOCATE(eigts3((nr3*2+1)*nat))
    ALLOCATE(mill(3,ngm))
    ALLOCATE(becsum(first_becsum, nat, nspin))
 
    READ(11) qmod
    READ(11) qgm
    READ(11) ylmk0
    READ(11) eigts1
    READ(11) eigts2
    READ(11) eigts3
    READ(11) mill
    READ(11) aux
    READ(11) becsum
    READ(11) ityp
 
    READ(11) sizeA
    READ(11) sizeB
    PRINT *, "ALLOCATE(indv(", sizeA, ",", sizeB, "))"
    ALLOCATE(indv(sizeA, sizeB))
    READ(11) indv
   
    READ(11) sizeA
    READ(11) sizeB
    PRINT *, "ALLOCATE(nhtolm(", sizeA, ",", sizeB, "))"
    ALLOCATE(nhtolm(sizeA, sizeB))
    READ(11) nhtolm
   
    READ(11) sizeA
    READ(11) sizeB
    READ(11) sizeC
    READ(11) sizeD
    PRINT *, "ALLOCATE(qrad(", sizeA, ",", sizeB, ",", sizeC, ",", sizeD, "))"
    ALLOCATE(qrad(sizeA, sizeB, sizeC, sizeD))
    READ(11) qrad

    READ(11) lpx
    READ(11) lpl
    READ(11) ap
  
    READ(11) nbetam


    CLOSE(11)




   		err = addusdens_cuda(nr1, nr2, nr3, first_becsum, nat, nh, nt, ngm, qmod, qgm, ylmk0, &
  		    eigts1, eigts2, eigts3, mill (1,:), mill (2,:), mill (3,:), &
  		    aux, becsum, ityp, nspin_mag, nspin, qrad, size(qrad,1), size(qrad,2), size(qrad,3), &
          size(qrad,4), lmaxq, nlx, dq, indv, nhtolm, nbetam, lpx, lpl, ap, size(ap,1), nhm)
  		!
   
end program
