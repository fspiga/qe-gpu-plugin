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

extern "C" int struct_fact_cuda_(int * ptr_nr1, int * ptr_nr2, int * ptr_nr3,
		int * ptr_nat, double * eigts1, double * eigts2, double * eigts3)
{
	int nat = (* ptr_nat);
	int nr1 = (* ptr_nr1);
	int nr2 = (* ptr_nr2);
	int nr3 = (* ptr_nr3);

#if defined(__CUDA_DEBUG)
	printf("[CUDA_DEBUG - preload eigts] preloaded_eigts1 = %d, preloaded_eigts2 = %d, preloaded_eigts3 = %d\n",
			 ( ( ( nr1 * 2 + 1 ) * nat ) * 2 ), ( ( ( nr2 * 2 + 1 ) * nat ) * 2 ), ( ( ( nr3 * 2 + 1 ) * nat ) * 2 ) );
	fflush(stdout);
#endif

    cudaSetDevice(qe_gpu_bonded[0]);

	int  shift = device_memory_shift[0];
	shift += ( ( ( nr1 * 2 + 1 ) * nat ) * 2 )*sizeof(double); // len(eigts1)
	preloaded_eigts1_D = (char *) dev_heap_QE[0] - shift ;

	shift += ( ( ( nr2 * 2 + 1 ) * nat ) * 2 )*sizeof(double); // len(eigts2)
	preloaded_eigts2_D = (char *) dev_heap_QE[0] - shift ;

	shift += ( ( ( nr3 * 2 + 1 ) * nat ) * 2 )*sizeof(double); // len(eigts3)
	preloaded_eigts3_D = (char *) dev_heap_QE[0] - shift ;

    // this is necessary to so other pre-loading in the right position...
 	device_memory_shift[0] = shift;

 	// Recalculated filled space
 	cuda_memory_unused[0] = cuda_memory_unused[0] -
             ( ( ( nr1 * 2 + 1 ) * nat ) * 2 )*sizeof(double) -
             ( ( ( nr2 * 2 + 1 ) * nat ) * 2 )*sizeof(double) -
             ( ( ( nr3 * 2 + 1 ) * nat ) * 2 )*sizeof(double);

#if defined(__CUDA_PRELOAD_PINNED)
	qecudaSafeCall( cudaMemcpyAsync( (double *) preloaded_eigts1_D, eigts1,
			sizeof( double ) * ( ( ( nr1 * 2 + 1 ) * nat ) * 2 ),
			cudaMemcpyHostToDevice, 0 ) );

	qecudaSafeCall( cudaMemcpyAsync( (double *) preloaded_eigts2_D, eigts2,
			sizeof( double ) * ( ( ( nr2 * 2 + 1 ) * nat ) * 2 ),
			cudaMemcpyHostToDevice, 0 ) );

	qecudaSafeCall( cudaMemcpyAsync( (double *) preloaded_eigts3_D, eigts3,
			sizeof( double ) * ( ( ( nr3 * 2 + 1 ) * nat ) * 2 ),
			cudaMemcpyHostToDevice, 0 ) );
#else
	qecudaSafeCall( cudaMemcpy( (double *) preloaded_eigts1_D, eigts1,
			sizeof( double ) * ( ( ( nr1 * 2 + 1 ) * nat ) * 2 ),
			cudaMemcpyHostToDevice ) );

	qecudaSafeCall( cudaMemcpy( (double *) preloaded_eigts2_D, eigts2,
			sizeof( double ) * ( ( ( nr2 * 2 + 1 ) * nat ) * 2 ),
			cudaMemcpyHostToDevice ) );

	qecudaSafeCall( cudaMemcpy( (double *) preloaded_eigts3_D, eigts3,
			sizeof( double ) * ( ( ( nr3 * 2 + 1 ) * nat ) * 2 ),
			cudaMemcpyHostToDevice ) );
#endif

#if defined(__CUDA) && defined(__PHIGEMM)
	// phiGEMM awareness
#if defined(__CUDA_DEBUG)
    printf("[CUDA DEBUG - preload eigts] phiGemmSetAvaiableScratchSpace %lu\n",cuda_memory_unused[0]); fflush(stdout);
#endif
	phiGemmSetAvaiableScratchSpace(0, cuda_memory_unused[0]);
#endif

	return 0;
}
