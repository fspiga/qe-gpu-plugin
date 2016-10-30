# QE-GPU: GPU-Accelerated Quantum ESPRESSO

Quantum ESPRESSO is an integrated suite of Open-Source computer codes for
electronic-structure calculations and materials modeling at the nanoscale.

The aim of QE-GPU is to create a "plugin-like" component for the standard
Quantum ESPRESSO package that allows to exploit the capabilities of NVIDIA
GPU graphics cards in order to allow materials scientists to do better and
fast science. GPU acceleration is currently available for the Plane-Wave
Self-Consistent Field (PWscf) code and the energy barriers and reaction
pathways through the Nudged Elastic Band method (NEB) package.

QE-GPU is provided "as is" and with no warranty. This software is distributed
under the GNU General Public License, please see the files LICENSE and
DISCLAIMER for details. This README presents an introduction to compiling,
installing, and using QE-GPU.

**This version is compatible only with Quantum ESPRESSO 5.4** 

## Supported GPU Architectures

 GPU | gpu-arch |
:---:|:---:|
 M2070 | Fermi |
 M2070Q | Fermi |
 M2090 | Fermi |
 K20 | Kepler |
 K20c | Kepler |
 K20x | Kepler |
 K40 | Kepler |
 K40c | Kepler |
 K80 | Kepler |
 P100 (PCIe) | Pascal |
 P100 (SMX2) | Pascal |


Any other GPU not listed in this table is **not** officially supported. The code may work but due to lack of proper double precision support or ECC the performance will not be ideal.


## How to compile

1. Copy QE-GPU in espresso directory

Move to the espresso root directory, uncompress the archive
```
$ tar zxvf QE-GPU-<TAG-NAME>.tar.gz
```
and create a symbolic link with the name GPU
```
$ ln -s QE-GPU-<TAG-NAME> GPU
```

2. Run QE-GPU configure

The QE-GPU configure is located in the GPU directory. An example of serial
configuration is the following:
```
$ cd GPU
$ ./configure --disable-parallel --enable-openmp \
  --enable-cuda --with-gpu-arch=Kepler \
  --with-cuda-dir=<full-path-where-CUDA-is-installed> \
  --with-magma --with-phigemm
$ cd ..
$ make -f Makefile.gpu pw-gpu
```

An example for parallel execution:
```
$ cd GPU
$ ./configure --enable-parallel --enable-openmp --with-scalapack \
  --enable-cuda --with-gpu-arch=sm_35 \
  --with-cuda-dir=<full-path-where-CUDA-is-installed> \
  --without-magma --with-phigemm
$ cd ..
$ make -f Makefile.gpu pw-gpu
```

For additional options for QE-GPU see ```./configure --help```.
Here a summary of all GPU-related options available:
-  ```--enable-cuda``` : enable CUDA (default: no)
-  ```--with-cuda-dir=<path>``` : specify CUDA installation directory (default is /usr/local/cuda/, _MANDATORY_)
-  ```--with-gpu-arch=<arch>``` : (Fermi|Kepler|Pascal) Specify the GPU target architecture (default: _Kepler_)
-  ```--with-magma``` : (yes|no|\<path\>) Use MAGMA. Self-compile or a \<path\> can be specified (default: _no_)
-  ```--with-phigemm``` : (yes|no|\<path\>) Use PHIGEMM. Self-compile ora \<path\> can be specified (default: _yes_)
