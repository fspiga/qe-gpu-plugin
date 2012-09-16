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

#include <stdio.h>
#include "cuda_env.h"

#if defined(__CUDA) && defined(__PHIGEMM)
#include "phigemm.h"
#endif

extern "C" int gk_sort_cuda_(int * ptr_ngk, int * ptr_npwx, int * igk)
{
	int ngk = (* ptr_ngk);
	int npwx = (* ptr_npwx);

#if defined(__CUDA_DEBUG)
	printf("[CUDA_DEBUG - preload igk]  npwx = %d, ngk = %d\n", npwx, ngk); fflush(stdout);
#endif

    cudaSetDevice(qe_gpu_bonded[0]);

    if (!preloaded_igk_flag) {

#if defined(__CUDA_DEBUG)
    	printf("[CUDA DEBUG - preload igk] FIRST igk allocation size(npwx)\n"); fflush(stdout);
#endif

		int  shift = device_memory_shift[0];
		shift += ( (npwx % 2 == 0)? npwx : npwx + 1 )*sizeof(int); // max{len(igk)}
		preloaded_igk_D = (char *) dev_heap_QE[0] - shift ;

		// this is necessary to so other pre-loading in the right position...
		device_memory_shift[0] = shift;

		// Recalculated filled space
		cuda_memory_unused[0] = cuda_memory_unused[0] -
				( ( (npwx % 2 == 0)? npwx : npwx + 1 ) )*sizeof(int);

#if defined(__CUDA) && defined(__PHIGEMM)
		// phiGEMM awareness
#if defined(__CUDA_DEBUG)
    	printf("[CUDA DEBUG - preload igk] phiGemmSetAvaiableScratchSpace %lu\n",cuda_memory_unused[0]); fflush(stdout);
#endif
		phiGemmSetAvaiableScratchSpace(0, cuda_memory_unused[0]);
#endif

		preloaded_igk_flag = 1;
    }

#if defined(__CUDA_PRELOAD_PINNED)
	qecudaSafeCall( cudaMemcpyAsync( (int *) preloaded_igk_D, igk,
			sizeof( int ) * ngk, cudaMemcpyHostToDevice, 0 ) );
#else
	qecudaSafeCall( cudaMemcpy( (int *) preloaded_igk_D, igk,
			sizeof( int ) * ngk, cudaMemcpyHostToDevice ) );
#endif

	return 0;
}
