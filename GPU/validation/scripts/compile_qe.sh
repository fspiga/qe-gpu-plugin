#!/bin/bash

# Copyright (C) 2001-2006 Quantum ESPRESSO group
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Author: Filippo Spiga (spiga _dot_ filippo _at_ gmail _dot_ com)
# Date: August 17, 2013
# Version: 1.0


# This script generates a set of executables with different capabilities 
# (serial, parallel, with/without GPU) using just the configure script

# This script generates executable that _do not take advantage_ of 
# multi-threading (OpenMP is by default OFF) 


# --- EDIT HERE - Un-comment for conditional compilation ---------------------

SELECT_STANDARD=yes
SELECT_PARALLEL=yes
SELECT_GPU=yes

# for GPU DEBUG purposes
#SELECT_ONLY=yes
#SELECT_EXCLUDING=yes

# *** Valid combinations (one at a time):
# _STANDARD + _PARALLEL + _GPU <- *DEFAULT* ( 9 *.x generated)
# _STANDARD + _PARALLEL <- CPU-ONLY ( 4 *.x generated)
# _STANDARD + _GPU
# _STANDARD 
# _STANDARD + _GPU + _ONLY
# _STANDARD + _GPU + _EXCLUDING
# _STANDARD + _GPU + _ONLY + _EXCLUDING
# _STANDARD + _GPU + _ONLY + _PARALLEL
# _STANDARD + _GPU + _EXCLUDING + _PARALLEL
# _STANDARD + _GPU + _ONLY + _EXCLUDING + _PARALLEL <- ALL ( 27 *.x generated)
#  _GPU + _ONLY
#  _GPU + _EXCLUDING
#  _GPU + _ONLY + _EXCLUDING
#  _GPU + _ONLY + _PARALLEL
#  _GPU + _EXCLUDING + _PARALLEL
#  _GPU + _ONLY + _EXCLUDING + _PARALLEL
# ----------------------------------------------------------------------------

# ___DO NOT EDIT BELOW THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING___

if [ "$2" ] ; then
export CUDA_HOME=$2
else
export CUDA_HOME=/usr/local/cuda
fi;

if [ "$1" ] ; then
export FINALDIR=$1
else
export FINALDIR=~/exes_QE_`date +"%Y%m%d-%H%M"`
#/local/fs395/exe_QE-5.0.3-GPU-r216_debug
fi;
mkdir -p ${FINALDIR}


if [ "${SELECT_STANDARD}" ] ; then

# CPU & GPU serial

make distclean
./configure --disable-parallel --disable-openmp
make MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS" pw
cp PW/src/pw.x ${FINALDIR}/pw.x

if [ "${SELECT_GPU}" ] ; then

make -f Makefile.gpu distclean
cd GPU/
./configure --disable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --disable-magma --enable-phigemm  --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-gpu.x

fi;

# CPU & GPU parallel

if [ "${SELECT_PARALLEL}" ] ; then

make distclean
./configure --enable-parallel --disable-openmp --without-scalapack
make MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS" pw
cp PW/src/pw.x ${FINALDIR}/pw-mpi.x

make distclean
./configure --enable-parallel --disable-openmp --with-scalapack
make MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS" pw
cp PW/src/pw.x ${FINALDIR}/pw-mpi-scalapack.x

make distclean
./configure --enable-parallel --disable-openmp --with-scalapack --with-elpa
make MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS" pw
cp PW/src/pw.x ${FINALDIR}/pw-mpi-elpa.x

if [ "${SELECT_GPU}" ] ; then

make -f Makefile.gpu distclean
cd GPU/
./configure --enable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --disable-magma --enable-phigemm  --without-scalapack --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-mpi-gpu.x

make -f Makefile.gpu distclean
cd GPU/
./configure --enable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --disable-magma --enable-phigemm  --with-scalapack --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-mpi-gpu-scalapack.x

make -f Makefile.gpu distclean
cd GPU/
./configure --enable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --disable-magma --enable-phigemm  --with-scalapack --disable-profiling --with-elpa
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-mpi-gpu-elpa.x

make -f Makefile.gpu distclean
cd GPU/
./configure --enable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --enable-magma --enable-phigemm  --without-scalapack --disable-profiling   --with-internal-cblas
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-mpi-gpu-magma.x

fi;
fi;
fi;

# ONLY serial

if [ "${SELECT_GPU}" ] ; then
if [ "${SELECT_ONLY}" ] ; then

make -f Makefile.gpu distclean
cd GPU/
./configure --disable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --disable-magma --enable-phigemm --disable-profiling  
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS -D__DISABLE_CUDA_NEWD -D__DISABLE_CUDA_ADDUSDENS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-gpu_only-PHIGEMM.x

make -f Makefile.gpu distclean
cd GPU/
./configure --disable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --disable-magma --enable-phigemm --enable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS -D__DISABLE_CUDA_NEWD -D__DISABLE_CUDA_ADDUSDENS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-gpu_only-PHIGEMM-profiling.x

make -f Makefile.gpu distclean
cd GPU/
./configure --disable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --enable-magma --disable-phigemm --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS -D__DISABLE_CUDA_NEWD -D__DISABLE_CUDA_ADDUSDENS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-gpu_only-MAGMA.x

make -f Makefile.gpu distclean
cd GPU/
./configure --disable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --disable-magma --disable-phigemm --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS -D__DISABLE_CUDA_ADDUSDENS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-gpu_only-NEWD.x

make -f Makefile.gpu distclean
cd GPU/
./configure --disable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --disable-magma --disable-phigemm --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS -D__DISABLE_CUDA_NEWD" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-gpu_only-ADDUSDENS.x

fi;
fi;

# ONLY parallel

if [ "${SELECT_GPU}" ] ; then
if [ "${SELECT_PARALLEL}" ] ; then
if [ "${SELECT_ONLY}" ] ; then

make -f Makefile.gpu distclean
cd GPU/
./configure --enable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --disable-magma --enable-phigemm  --without-scalapack --disable-profiling  
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS -D__DISABLE_CUDA_NEWD -D__DISABLE_CUDA_ADDUSDENS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-mpi-gpu_only-PHIGEMM.x

make -f Makefile.gpu distclean
cd GPU/
./configure --enable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --disable-magma --enable-phigemm  --without-scalapack --enable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS -D__DISABLE_CUDA_NEWD -D__DISABLE_CUDA_ADDUSDENS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-mpi-gpu_only-PHIGEMM-profiling.x

make -f Makefile.gpu distclean
cd GPU/
./configure --enable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --enable-magma --disable-phigemm  --without-scalapack --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS -D__DISABLE_CUDA_NEWD -D__DISABLE_CUDA_ADDUSDENS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-mpi-gpu_only-MAGMA.x

make -f Makefile.gpu distclean
cd GPU/
./configure --enable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --disable-magma --disable-phigemm  --without-scalapack --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS -D__DISABLE_CUDA_ADDUSDENS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-mpi-gpu_only-NEWD.x

make -f Makefile.gpu distclean
cd GPU/
./configure --enable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --disable-magma --disable-phigemm  --without-scalapack --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS -D__DISABLE_CUDA_NEWD" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-mpi-gpu_only-ADDUSDENS.x

fi;
fi;
fi;

# EXCLUDING serial

if [ "${SELECT_GPU}" ] ; then
if [ "${SELECT_EXCLUDING}" ] ; then

make -f Makefile.gpu distclean
cd GPU/
./configure --disable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --enable-magma --disable-phigemm --disable-profiling  
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-gpu_excluding-PHIGEMM.x

make -f Makefile.gpu distclean
cd GPU/
./configure --disable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --disable-magma --enable-phigemm --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-gpu_excluding-MAGMA.x

make -f Makefile.gpu distclean
cd GPU/
./configure --disable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --enable-magma --enable-phigemm --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS -D__DISABLE_CUDA_NEWD" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-gpu_excluding-NEWD.x

make -f Makefile.gpu distclean
cd GPU/
./configure --disable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --enable-magma --enable-phigemm --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS -D__DISABLE_CUDA_ADDUSDENS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-gpu_excluding-ADDUSDENS.x

fi;
fi;

# EXCLUDING parallel

if [ "${SELECT_GPU}" ] ; then
if [ "${SELECT_PARALLEL}" ] ; then
if [ "${SELECT_EXCLUDING}" ] ; then

make -f Makefile.gpu distclean
cd GPU/
./configure --enable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --enable-magma --disable-phigemm  --without-scalapack --disable-profiling  
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-mpi-gpu_excluding-PHIGEMM.x

make -f Makefile.gpu distclean
cd GPU/
./configure --enable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --disable-magma --enable-phigemm  --without-scalapack --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-mpi-gpu_excluding-MAGMA.x

make -f Makefile.gpu distclean
cd GPU/
./configure --enable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --enable-magma --enable-phigemm  --without-scalapack --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS -D__DISABLE_CUDA_NEWD" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-mpi-gpu_excluding-NEWD.x

make -f Makefile.gpu distclean
cd GPU/
./configure --enable-parallel --disable-openmp --enable-cuda --with-gpu-arch=35 --with-cuda-dir=${CUDA_HOME} --enable-magma --enable-phigemm  --without-scalapack --disable-profiling
cd ../
make -f Makefile.gpu MANUAL_DFLAGS="-D__CLOCK_SECONDS -D__PW_TRACK_ELECTRON_STEPS -D__DISABLE_CUDA_ADDUSDENS" pw-gpu
cp GPU/PW/pw-gpu.x ${FINALDIR}/pw-mpi-gpu_excluding-ADDUSDENS.x

fi;
fi;
fi;
