/*
 * Copyright (C) 2001-2012 Quantum ESPRESSO group
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 */

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#ifndef __QE_CUDA_ENVIRONMENT_H
#define __QE_CUDA_ENVIRONMENT_H

#if defined __GPU_NVIDIA_13

#define __CUDA_THREADPERBLOCK__ 256
#define __NUM_FFT_MULTIPLAN__ 4
#define __CUDA_TxB_ADDUSDENS_COMPUTE_AUX__ __CUDA_THREADPERBLOCK__
#define __CUDA_TxB_VLOCPSI_PSIC__ __CUDA_THREADPERBLOCK__
#define __CUDA_TxB_VLOCPSI_PROD__ __CUDA_THREADPERBLOCK__
#define __CUDA_TxB_VLOCPSI_HPSI__ __CUDA_THREADPERBLOCK__
#define __CUDA_TxB_NEWD_QGM__ __CUDA_THREADPERBLOCK__
#define __CUDA_TxB_NEWD_DEEPQ__ __CUDA_THREADPERBLOCK__

#elif defined __GPU_NVIDIA_20

#define __CUDA_THREADPERBLOCK__ 512
#define __NUM_FFT_MULTIPLAN__ 4
// TxB ADDUSDENS_COMPUTE_AUX = 1024 is OK for CUDA 4.0 but NOT for CUDA 4.1
// TxB ADDUSDENS_COMPUTE_AUX = 896 is OK for CUDA 4.1 too but not for CUDA 5.0
// TxB ADDUSDENS_COMPUTE_AUX = 768 is OK for CUDA 5.0 and all the other versions...
#define __CUDA_TxB_ADDUSDENS_COMPUTE_AUX__ 768
#define __CUDA_TxB_VLOCPSI_PSIC__ 64
#define __CUDA_TxB_VLOCPSI_PROD__ 128
#define __CUDA_TxB_VLOCPSI_HPSI__ 448
#define __CUDA_TxB_NEWD_QGM__ 512
#define __CUDA_TxB_NEWD_DEEPQ__ 512


#elif defined __GPU_NVIDIA_30

// SMX multi-processors have much more registers (check the CUDA Occupancy spreadsheet)
// Tested using CUDA 4.2 on GPU K10 (Kepler v1)
#define __CUDA_THREADPERBLOCK__ 512
#define __NUM_FFT_MULTIPLAN__ 4
#define __CUDA_TxB_ADDUSDENS_COMPUTE_AUX__ 128
#define __CUDA_TxB_VLOCPSI_PSIC__ 256
#define __CUDA_TxB_VLOCPSI_PROD__ 512
#define __CUDA_TxB_VLOCPSI_HPSI__ 256
#define __CUDA_TxB_NEWD_QGM__ __CUDA_THREADPERBLOCK__
#define __CUDA_TxB_NEWD_DEEPQ__ __CUDA_THREADPERBLOCK__

#else

#define __CUDA_THREADPERBLOCK__ 256
#define __NUM_FFT_MULTIPLAN__ 1

#endif

/* Sometimes it is not possible to use 'cuMemGetInfo()' to know the amount
 * of memory on the GPU card. For this reason this macro define a "fixed"
 * amount of memory to use in case this behavior happens. Use carefully
 * and edit the amount (in byte) accordingly to the real amount of memory
 * on the card minus ~500MB. [NdFilippo]
 */
#if defined __CUDA_GET_MEM_HACK
#define __GPU_MEM_AMOUNT_HACK__ 2400000000
#endif

#if defined __MAGMA
#define __SCALING_MEM_FACTOR__ 0.75
#else
#define __SCALING_MEM_FACTOR__ 0.95
#endif

#define MAX_QE_GPUS 8

typedef void* qeCudaMemDevPtr[MAX_QE_GPUS];
typedef size_t qeCudaMemSizes[MAX_QE_GPUS];
typedef int qeCudaDevicesBond[MAX_QE_GPUS];

extern qeCudaMemDevPtr dev_scratch_QE;
extern qeCudaMemDevPtr dev_heap_QE;
extern qeCudaMemSizes cuda_memory_allocated;
extern qeCudaMemSizes device_memory_shift;
extern qeCudaMemSizes cuda_memory_unused;
extern qeCudaMemSizes device_memory_left;
extern qeCudaDevicesBond qe_gpu_bonded;

// Pre-loaded data-structure
extern void * preloaded_eigts1_D, * preloaded_eigts2_D, * preloaded_eigts3_D;
extern void * preloaded_ig1_D, * preloaded_ig2_D, * preloaded_ig3_D;
extern void * preloaded_nlsm_D, * preloaded_nls_D, * preloaded_igk_D;;
extern short int preloaded_igk_flag;

extern long ngpus_detected;
extern long ngpus_used;
extern long ngpus_per_process;

extern "C" size_t initCudaEnv();
extern "C" void closeCudaEnv();
extern "C" void preallocateDeviceMemory(int);
extern "C" void initPhigemm(int);

/* These routines are exactly the same in "cutil_inline_runtime.h" but,
 * replicating them here, we remove the annoying dependency to CUTIL & SDK (Filippo)
 *
 * We define these calls here, so the user doesn't need to include __FILE__ and __LINE__
 * The advantage is the developers gets to use the inline function so they can debug
 */

#define qecudaSafeCall(err)  __qecudaSafeCall(err, __FILE__, __LINE__)
#define qecudaGetLastError(msg)  __qecudaGetLastError(msg, __FILE__, __LINE__)
#define qecheck_cufft_call(err) __qecheck_cufft_call(err, __FILE__, __LINE__)

inline void __qecudaSafeCall( cudaError_t err, const char *file, const int line )
{
    if( cudaSuccess != err) {
		printf("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
					file, line, cudaGetErrorString( err) ); fflush(stdout);
		exit(EXIT_FAILURE);
    }
}


inline void __qecudaGetLastError(const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
		printf("%s(%i) : qecudaGetLastError() error : %s : %s.\n",
				file, line, errorMessage, cudaGetErrorString( err) ); fflush(stdout);
		exit(EXIT_FAILURE);
    }
}


inline void __qecheck_cufft_call(  cufftResult cufft_err, const char *file, const int line )
{
	switch ( cufft_err ) {

	case CUFFT_INVALID_PLAN :
		fprintf( stderr, "\n[%s:%d] The plan parameter is not a valid handle! Program exits... \n", file, line );
		break;

	case CUFFT_INVALID_VALUE :
		fprintf( stderr, "\n[%s:%d] The idata, odata, and/or direction parameter is not valid! Program exits... \n", file, line );
		break;

	case CUFFT_EXEC_FAILED :
		fprintf( stderr, "\n[%s:%d] CUFFT failed to execute the transform on GPU! Program exits... \n", file, line );
		break;

	case CUFFT_SETUP_FAILED :
		fprintf( stderr, "\n[%s:%d] CUFFT library failed to initialize! Program exits... \n", file, line );
		break;

	case CUFFT_INVALID_SIZE :
		fprintf( stderr, "\n[%s:%d] The nx parameter is not a supported size! Program exits... \n", file, line );
		break;

	case CUFFT_INVALID_TYPE :
		fprintf( stderr, "\n[%s:%d] The type parameter is not supported! Program exits... \n", file, line );
		break;

	case CUFFT_ALLOC_FAILED :
		fprintf( stderr, "\n[%s:%d] Allocation of GPU resources for the plan failed! Program exits... \n", file, line );
		break;

	case CUFFT_SUCCESS:
		break;

	default:
		fprintf( stderr, "\n[%s:%d] CUFFT returned not recognized value! %d\n", file, line, cufft_err );
		break;
	}

	if (cufft_err != CUFFT_SUCCESS) {
		exit(EXIT_FAILURE);
	}
}

#endif // __QE_CUDA_ENVIRONMENT_H
