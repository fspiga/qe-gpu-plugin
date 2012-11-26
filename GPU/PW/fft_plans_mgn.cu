/*
 * Copyright (C) 2001-2013 Quantum ESPRESSO Foundation
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 */

#include <stdio.h>
#include <stdlib.h>

#include "cuda_env.h"

//extern "C" void start_clock_(char * label, unsigned int length_arg );
//extern "C" void stop_clock_(char * label, unsigned int length_arg );

extern "C"  int cuda_dffts_plan_create_(int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s)
{
	size_t free, total;
	int i;
	int nr1s = (* ptr_nr1s);
	int nr2s = (* ptr_nr2s);
	int nr3s = (* ptr_nr3s);

#if defined(__CUDA_DEBUG)
	printf("[DFFTP_PLAN_CREATE] Enter (nr1s [x]=%d, nr2s [y]=%d, nr3s [z]=%d)\n", nr1s, nr2s, nr3s); fflush(stdout);
#endif

	// This works of one GPU only per process...
	for (i = 0; i < 1; i++) {

		if ( cudaSetDevice(qe_gpu_bonded[i]) != cudaSuccess) {
			printf("*** ERROR *** cudaSetDevice(%d) failed!", qe_gpu_bonded[i] ); fflush(stdout);
			exit(EXIT_FAILURE);
		}

#if defined(__CUDA_DEBUG)
		printf("[DFFTP_PLAN_CREATE] pre-plan, qe_gpu_mem_unused[%d]=%lu\n", i, qe_gpu_mem_unused[i]); fflush(stdout);
#endif

		qecheck_cufft_call( cufftPlan3d( &qeCudaFFT_dffts, nr3s, nr2s,  nr1s, CUFFT_Z2Z ) );

		cudaMemGetInfo((size_t*)&free,(size_t*)&total);

		qe_gpu_mem_tot[i] = (size_t) ((free * __SCALING_MEM_FACTOR__ ) / procs_per_gpu);
		qe_gpu_mem_unused[i] = qe_gpu_mem_tot[i];

#if defined(__CUDA_DEBUG)
		printf("[DFFTP_PLAN_CREATE] post-plan, qe_gpu_mem_unused[%d]=%lu\n", i, qe_gpu_mem_unused[i]); fflush(stdout);
#endif
	}

	return 0;
}

extern "C"  int cuda_dffts_plan_destroy_()
{
	int i;

	for (i = 0; i < ngpus_per_process; i++) {
		if ( cudaSetDevice(qe_gpu_bonded[i]) != cudaSuccess) {
			printf("*** ERROR *** cudaSetDevice(%d) failed!", qe_gpu_bonded[i] ); fflush(stdout);
			exit(EXIT_FAILURE);
		}

		qecheck_cufft_call( cufftDestroy(qeCudaFFT_dffts) );
	}

	return 0;
}
