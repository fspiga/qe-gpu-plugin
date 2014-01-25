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
   USE uspp_param,           ONLY : nbetam, lmaxq
   USE us,                   ONLY : dq, qrad
   !USE read_pseudo_mod,      ONLY : readpp
   CHARACTER(LEN=32) :: filename  = "newd_input001.bin"
   CHARACTER(LEN=128) :: pseudopot_file
   INTEGER :: nr1, nr2, nr3, na
   INTEGER, ALLOCATABLE :: nh(:)
   DOUBLE PRECISION :: fact
   INTEGER :: nt, nat, ngm, nhm, nspin
   DOUBLE PRECISION, ALLOCATABLE :: qmod(:), ylmk0(:,:)
   COMPLEX(DP), ALLOCATABLE :: eigts1(:)
   COMPLEX(DP), ALLOCATABLE :: eigts2(:)
   COMPLEX(DP), ALLOCATABLE :: eigts3(:)
   INTEGER, ALLOCATABLE :: mill(:,:)
   DOUBLE PRECISION, ALLOCATABLE :: deeq(:,:,:,:)
   INTEGER :: ityp
   DOUBLE PRECISION :: omega
   INTEGER :: flag
   COMPLEX(DP), ALLOCATABLE :: aux(:,:)
   INTEGER :: nspin_mag
   INTEGER :: sizeA, sizeB, sizeC, sizeD

   INTEGER, EXTERNAL :: newd_cuda
   INTEGER :: err
   
   !CALL allocate_nlpot()
   
   !pseudopot_file = "/home/lbarnes/QuantumEspresso/espresso-svn/espresso/GPU/validation/inputs-pw/Au.pbe-nd-van.UPF"
   !CALL readpp(pseudopot_file)
   !CALL setup()
   !CALL init_run()
   CALL InitCudaEnv()
   PRINT *, "Finished init"
   OPEN(UNIT= 11, STATUS = 'OLD', FILE = filename, FORM='UNFORMATTED') 
   READ(11) sizeA
   ALLOCATE(nh(sizeA))
   READ(11) nr1
   READ(11) nr2
   READ(11) nr3
   READ(11) na
   READ(11) nh
   READ(11) fact
   READ(11) nt
   READ(11) nat
   READ(11) ngm
   READ(11) nhm
   READ(11) nspin
   READ(11) nspin_mag
   READ(11) lmaxq

   PRINT *, "nr1 = ", nr1
   PRINT *, "nr2 = ", nr2
   PRINT *, "nr3 = ", nr3
   PRINT *, "na = ", na
   PRINT *, "nh = ", nh
   PRINT *, "nt = ", nt
   PRINT *, "nat = ", nat
   PRINT *, "ngm = ", ngm
   PRINT *, "nhm = ", nhm
   PRINT *, "nspin = ", nspin
   PRINT *, "nspin_mag = ", nspin_mag
   PRINT *, "lmaxq = ", lmaxq

   ALLOCATE(qmod(ngm))
   ALLOCATE(ylmk0(ngm,lmaxq*lmaxq))
   ALLOCATE(eigts1((nr1*2+1)*nat))
   ALLOCATE(eigts2((nr2*2+1)*nat))
   ALLOCATE(eigts3((nr3*2+1)*nat))
   ALLOCATE(mill(3,ngm))
   ALLOCATE(deeq(nhm,nhm,nat,nspin))
   ALLOCATE(aux(ngm,nspin_mag))
   
   PRINT *, "qmod(",size(qmod),")"
   PRINT *, "ylmk0(",size(ylmk0,1),",",size(ylmk0,2),")"
   PRINT *, "eigts1(",size(eigts1),")"
   PRINT *, "eigts2(",size(eigts2),")"
   PRINT *, "eigts3(",size(eigts3),")"
   PRINT *, "mill(",size(mill,1),",",size(mill,2),")"
   PRINT *, "deeq(",size(deeq,1),",",size(deeq,2),",",size(deeq,3),",",size(deeq,4),")"
   PRINT *, "aux(",size(aux,1),",",size(aux,2),")"
   READ(11) qmod
   READ(11) ylmk0
   READ(11) eigts1
   READ(11) eigts2
   READ(11) eigts3
   READ(11) mill
   READ(11) deeq
   READ(11) ityp
   READ(11) omega
   READ(11) flag
   READ(11) aux
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

   !PRINT *, "Calling newd_cuda"
   err = newd_cuda( nr1, nr2, nr3, na, nh, fact, &
       nt, nat, ngm, nhm, nspin, qmod, ylmk0, &
       eigts1, eigts2, eigts3, mill(1,:), mill(2,:), mill(3,:), &
       deeq, ityp, omega, flag, aux, nspin_mag, qrad, &
       size(qrad,1), size(qrad,2), size(qrad,3), size(qrad,4), lmaxq, &
       nlx, dq, indv, nhtolm, nbetam, lpx, lpl, ap, size(ap,1))

end program

