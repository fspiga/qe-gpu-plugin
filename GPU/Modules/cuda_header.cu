/*
 * Copyright (C) 2001-2013 Quantum ESPRESSO Foundation
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 */

#include <stdlib.h>
#include <stdio.h>

#include "cuda_env.h"
#include "gpu-version.h"

extern "C" void print_cuda_header_()
{
	// Print GPU memory allocation information on stdout
#if defined(__CUDA_DEBUG)
	int i;
	size_t free, total;

	for (i = 0; i < ngpus_per_process; i++) {

		if ( cudaSetDevice(qe_gpu_bonded[i]) != cudaSuccess) {
			printf("*** ERROR *** cudaSetDevice(%d) failed!", qe_gpu_bonded[i] ); fflush(stdout);
			exit(EXIT_FAILURE);
		}

		cudaMemGetInfo((size_t*)&free,(size_t*)&total);

#if defined(__PARA)
		printf("[GPU %d - rank: %d] Allocated: %lu, Free: %lu, Total: %lu)\n", qe_gpu_bonded[i], (int) lRank, (unsigned long)qe_gpu_mem_tot[i], (unsigned long)free, (unsigned long)total);
		fflush(stdout);
#else
		printf("[GPU %d] Allocated: %lu, , Free: %lu, Total: %lu\n", qe_gpu_bonded[i], (unsigned long)qe_gpu_mem_tot[i], (unsigned long)free, (unsigned long)total);
		fflush(stdout);
#endif
	}
#if defined(__PARA)
	mybarrier_();
#endif
#endif

	// Print general information on stdout
	#if defined(__PARA)
	if (lRank == 0) {	
#endif
	printf("\n"); fflush(stdout);
	printf("     *******************************************************************\n\n"); fflush(stdout);
#if defined(__PHIGEMM_CPUONLY) && defined(__PHIGEMM_PROFILE)
	printf("       CPU-version with call-by-call GEMM profiling"); fflush(stdout);
#else
	printf("       GPU-accelerated Quantum ESPRESSO (svn rev. %s)\n", CURRENT_QE_VERSION); fflush(stdout);
#if defined(__PARA) || defined(__OPENACC) || defined(__CUDA_PINNED) || !defined(__MAGMA) || defined(__USE_3D_FFT)
	printf("       ");  fflush(stdout);
#if defined(__PARA)
	printf("(parallel: Y "); fflush(stdout);
#else
	printf("(parallel: N "); fflush(stdout);
#endif
#if defined(__OPENACC)
	printf(", OpenACC: Y"); fflush(stdout);
#endif
#if defined(__CUDA_PINNED)
    printf(", Pinned memory: Y "); fflush(stdout);
#endif
#if !defined(__MAGMA)
	printf(", MAGMA : N "); fflush(stdout);
#endif
#if defined(__PARA) && defined(__USE_3D_FFT)
	printf(", USE_3D_FFT : Y"); fflush(stdout);
#endif
	printf(")\n"); fflush(stdout);
#endif

#if defined(__CUDA_DEBUG)
	printf("       # DEBUG MODE #\n");fflush(stdout);
	printf("         GPUs per node     = %d\n", (int) ngpus_detected); fflush(stdout);
	printf("         GPUs per process  = %d\n", (int) ngpus_per_process); fflush(stdout);
#if defined(__DISABLE_CUDA_ADDUSDENS)
	printf("         CUDA addusdens    = disabled\n");fflush(stdout);
#else
	printf("         CUDA addusdens    = enabled\n");fflush(stdout);
#endif
#if defined(__DISABLE_CUDA_VLOCPSI)
	printf("         CUDA vloc_psi     = disabled\n");fflush(stdout);
#else
	printf("         CUDA vloc_psi     = enabled\n");fflush(stdout);
#endif
#if defined(__DISABLE_CUDA_NEWD)
	printf("         CUDA newd         = disabled\n");fflush(stdout);
#else
	printf("         CUDA newd         = enabled\n");fflush(stdout);
#endif
#if defined(__PHIGEMM)
	printf("         phiGEMM           = enabled\n");fflush(stdout);
#if defined(__PHIGEMM_ENABLE_SPECIALK)
	printf("         phiGEMM special-k = enabled\n");fflush(stdout);
#else
	printf("         phiGEMM special-k = disabled\n");fflush(stdout);
#endif
#else
	printf("         phiGEMM           = disabled\n");fflush(stdout);
#endif
#endif

#endif

    printf("\n"); fflush(stdout);
	printf("     *******************************************************************\n"); fflush(stdout);
	printf("\n"); fflush(stdout);

#if defined(__PARA)
	}
#endif
}
