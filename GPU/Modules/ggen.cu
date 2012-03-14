/*
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 * Copyright (C) 2001-2011 Quantum ESPRESSO group
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 */

#include <stdio.h>
#include "cuda_env.h"

#if defined(__CUDA) && defined(__PHIGEMM)
#include "phigemm.h"
#endif

extern "C" int ggen_mill_cuda_(int * ptr_ngm, int * ig1, int * ig2, int * ig3)
{
	int ngm = (* ptr_ngm);

#if defined(__CUDA_DEBUG)
	printf("[CUDA_DEBUG - preload ig]  ngm = %d\n", ngm); fflush(stdout);
#endif

    cudaSetDevice(qe_gpu_bonded[0]);

	int  shift = device_memory_shift[0];
	shift += ( (ngm % 2 == 0)? ngm : ngm + 1 )*sizeof(int); // ig1 = mill(1,:)
	preloaded_ig1_D = (char *) dev_heap_QE[0] - shift ;

	shift += ( (ngm % 2 == 0)? ngm : ngm + 1 )*sizeof(int); // ig2 = mill(2,:)
	preloaded_ig2_D = (char *) dev_heap_QE[0] - shift ;

	shift += ( (ngm % 2 == 0)? ngm : ngm + 1 )*sizeof(int); // ig3 = mill(3,:)
	preloaded_ig3_D = (char *) dev_heap_QE[0] - shift ;

    // this is necessary to so other pre-loading in the right position...
 	device_memory_shift[0] = shift;

 	// Recalculated filled space
 	cuda_memory_unused[0] = cuda_memory_unused[0] -
 			3*( (ngm % 2 == 0)? ngm : ngm + 1 )*sizeof(int);

#if defined(__CUDA_PRELOAD_PINNED)
	qecudaSafeCall( cudaMemcpyAsync( (int *) preloaded_ig1_D, ig1,
			sizeof( int ) * ngm, cudaMemcpyHostToDevice, 0 ) );

	qecudaSafeCall( cudaMemcpyAsync( (int *) preloaded_ig2_D, ig2,
			sizeof( int ) * ngm, cudaMemcpyHostToDevice, 0 ) );

	qecudaSafeCall( cudaMemcpyAsync( (int *) preloaded_ig3_D, ig3,
			sizeof( int ) * ngm, cudaMemcpyHostToDevice, 0 ) );
#else
	qecudaSafeCall( cudaMemcpy( (int *) preloaded_ig1_D, ig1,
			sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );

	qecudaSafeCall( cudaMemcpy( (int *) preloaded_ig2_D, ig2,
			sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );

	qecudaSafeCall( cudaMemcpy( (int *) preloaded_ig3_D, ig3,
			sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
#endif

#if defined(__CUDA) && defined(__PHIGEMM)
	// phiGEMM awareness
#if defined(__CUDA_DEBUG)
    printf("[CUDA DEBUG - preload ig] phiGemmSetAvaiableScratchSpace %lu\n",cuda_memory_unused[0]); fflush(stdout);
#endif
	phiGemmSetAvaiableScratchSpace(0, cuda_memory_unused[0]);
#endif

	return 0;
}


extern "C" int ggen_nls_cuda_(int * ptr_ngms, int * nls)
{
	int ngms = (* ptr_ngms);

#if defined(__CUDA_DEBUG)
	printf("[CUDA_DEBUG - preload nls]  ngms = %d\n", ngms); fflush(stdout);
#endif

    cudaSetDevice(qe_gpu_bonded[0]);

	int  shift = device_memory_shift[0];
	shift += ( (ngms % 2 == 0)? ngms : ngms + 1 )*sizeof(int);
	preloaded_nls_D = (char *) dev_heap_QE[0] - shift ;

    // this is necessary to so other pre-loading in the right position...
 	device_memory_shift[0] = shift;

 	// Recalculated filled space
 	cuda_memory_unused[0] = cuda_memory_unused[0] -
 			( (ngms % 2 == 0)? ngms : ngms + 1 )*sizeof(int);

#if defined(__CUDA_PRELOAD_PINNED)
	qecudaSafeCall( cudaMemcpyAsync( (int *) preloaded_nls_D, nls,
			sizeof( int ) * ngms, cudaMemcpyHostToDevice, 0 ) );
#else
	qecudaSafeCall( cudaMemcpy( (int *) preloaded_nls_D, nls,
			sizeof( int ) * ngms, cudaMemcpyHostToDevice ) );
#endif

#if defined(__CUDA) && defined(__PHIGEMM)
	// phiGEMM awareness
#if defined(__CUDA_DEBUG)
    printf("[CUDA DEBUG - preload nls] phiGemmSetAvaiableScratchSpace %lu\n",cuda_memory_unused[0]); fflush(stdout);
#endif
	phiGemmSetAvaiableScratchSpace(0, cuda_memory_unused[0]);
#endif

	return 0;
}


extern "C" int ggen_nlsm_cuda_(int * ptr_ngm, int * nlsm)
{

	int ngm = (* ptr_ngm);

#if defined(__CUDA_DEBUG)
	printf("[CUDA_DEBUG - preload nlsm]  ngm = %d\n", ngm); fflush(stdout);
#endif

    cudaSetDevice(qe_gpu_bonded[0]);

	int  shift = device_memory_shift[0];
	shift += ( (ngm % 2 == 0)? ngm : ngm + 1 )*sizeof(int);
	preloaded_nlsm_D = (char *) dev_heap_QE[0] - shift ;

    // this is necessary to so other pre-loading in the right position...
 	device_memory_shift[0] = shift;

 	// Recalculated filled space
 	cuda_memory_unused[0] = cuda_memory_unused[0] -
 			( (ngm % 2 == 0)? ngm : ngm + 1 )*sizeof(int);

	qecudaSafeCall( cudaMemcpy( (int *) preloaded_nlsm_D, nlsm,
			sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );

#if defined(__CUDA) && defined(__PHIGEMM)
	// phiGEMM awareness
#if defined(__CUDA_DEBUG)
    printf("[CUDA DEBUG - preload nlsm] phiGemmSetAvaiableScratchSpace %lu\n",cuda_memory_unused[0]); fflush(stdout);
#endif
	phiGemmSetAvaiableScratchSpace(0, cuda_memory_unused[0]);
#endif

	return 0;
}
