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

#include <stdlib.h>
#include <stdio.h>

#include <driver_types.h>

#if defined(__TIMELOG)
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#endif

#include "cuda_env.h"

#if defined(__PHIGEMM)
#include "phigemm.h"
#endif

qeCudaMemDevPtr dev_scratch_QE;
qeCudaMemDevPtr dev_heap_QE;
qeCudaMemSizes cuda_memory_allocated;
qeCudaMemSizes device_memory_left;
qeCudaMemSizes device_memory_shift;
qeCudaMemSizes cuda_memory_unused;
qeCudaDevicesBond qe_gpu_bonded;

// global useful information
long ngpus_detected;
long ngpus_used;
long ngpus_per_process;
long procs_per_gpu;

// Pre-loaded data-structures
void * preloaded_eigts1_D = 0, * preloaded_eigts2_D = 0, * preloaded_eigts3_D = 0;
void * preloaded_ig1_D = 0, * preloaded_ig2_D = 0, * preloaded_ig3_D = 0;
void * preloaded_nlsm_D = 0, * preloaded_nls_D = 0, * preloaded_igk_D = 0;
short int preloaded_igk_flag;

extern "C" void paralleldetect_(int * lRankThisNode_ptr, int * lSizeThisNode_ptr , int * lRank_ptr);
extern "C" void mybarrier_();

#if defined(__TIMELOG)
double cuda_cclock(void)
{
	struct timeval tv;
	struct timezone tz;
	double t;

	gettimeofday(&tv, &tz);

	t = (double)tv.tv_sec;
	t += ((double)tv.tv_usec)/1000000.0;

	return t;
}
#endif


void gpubinding_(int lRankThisNode, int lSizeThisNode, int lRank){

	int lNumDevicesThisNode = 0;
	int i;

#if defined(__PARA)

	/* Attach all MPI processes on this node to the available GPUs
	 * in round-robin fashion
	 */
	cudaGetDeviceCount(&lNumDevicesThisNode);

	if (lNumDevicesThisNode == 0 && lRankThisNode == 0)
	{
		printf("***ERROR: no CUDA-capable devices were found.\n");
//		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		exit(EXIT_FAILURE);
	}

	ngpus_detected = lNumDevicesThisNode;

	if ( (lSizeThisNode % lNumDevicesThisNode ) != 0  )
	{
		printf("***WARNING: unbalanced configuration (%d MPI per node, %d GPUs per node)\n", lSizeThisNode, lNumDevicesThisNode);
		fflush(stdout);
	}

	if (ngpus_detected <= lSizeThisNode ){
		/* if GPUs are less then (or equal of) the number of  MPI processes on a single node,
		 * then PWscf uses all the GPU and one single GPU is assigned to one or multiple MPI processes with overlapping. */
		ngpus_used = ngpus_detected;
		ngpus_per_process = 1;
	} else {
		/* multi-GPU in parallel calculations is allowed ONLY if CUDA >= 4.0 */

		/* if GPUs are more than the MPI processes on a single node,
		 * then PWscf uses all the GPU and one or more GPUs are assigned
		 * to every single MPI processes without overlapping.
		 * *** NOT IMPLEMENTED YET ***
		 */
		ngpus_used = ngpus_detected;
		ngpus_per_process = 1;
	}

	procs_per_gpu = (lSizeThisNode < lNumDevicesThisNode) ? lSizeThisNode : lSizeThisNode / lNumDevicesThisNode;

	for (i = 0; i < ngpus_per_process; i++) {

		qe_gpu_bonded[i] = lRankThisNode % lNumDevicesThisNode;

#if defined(__CUDA_DEBUG)
		printf("Binding GPU %d on node of rank: %d (internal rank:%d)\n", qe_gpu_bonded[i], lRank, lRankThisNode); fflush(stdout);
#endif

	}

#else

	cudaGetDeviceCount(&lNumDevicesThisNode);

	if (lNumDevicesThisNode == 0)
	{
		fprintf( stderr,"***ERROR*** no CUDA-capable devices were found on the machine.\n");
		exit(EXIT_FAILURE);
	}

	ngpus_detected = lNumDevicesThisNode;

	/* multi-GPU in serial calculations is allowed ONLY if CUDA >= 4.0 */
#if defined(__MULTI_GPU)
	ngpus_used = ngpus_per_process = lNumDevicesThisNode;
#else
	ngpus_used = ngpus_per_process = 1;
#endif

	for (i = 0; i < ngpus_per_process; i++) {
		/* NOTE: qe_gpu_bonded[0] is ALWAYS the main device for non multi-GPU
		 *       kernels.
		 */
		qe_gpu_bonded[i] = i;
	}

	// NULL tag for phiGEMM
	lRank = -1;

#endif
}

#if defined(__PHIGEMM)
extern "C" void initphigemm_(int lRank){
	/* Compatibility with CUDA 4.x (latest phiGEMM): */

#if defined(__PHIGEMM_NOALLOC)
	phiGemmInit(ngpus_per_process , NULL, (qeCudaMemSizes*)&cuda_memory_allocated, (int *)qe_gpu_bonded, lRank);
#else
	phiGemmInit(ngpus_per_process , (qeCudaMemDevPtr*)&dev_scratch_QE, (qeCudaMemSizes*)&cuda_memory_allocated, (int *)qe_gpu_bonded, lRank);
#endif
}
#endif

extern "C" void preallocatedevicememory_(int lRank){

	int ierr = 0;
	int i;

	size_t free, total;

	for (i = 0; i < ngpus_per_process; i++) {

		/* query the real free memory, taking into account the "stack" */
		if ( cudaSetDevice(qe_gpu_bonded[i]) != cudaSuccess) {
			printf("*** ERROR *** cudaSetDevice(%d) failed!", qe_gpu_bonded[i] ); fflush(stdout);
			exit(EXIT_FAILURE);
		}

		cuda_memory_allocated[i] = (size_t) 0;

		ierr = cudaMalloc ( (void**) &(dev_scratch_QE[i]), cuda_memory_allocated[i] );
		if ( ierr != cudaSuccess) {
			fprintf( stderr, "\nError in (first zero) memory allocation , program will be terminated!!! Bye...\n\n");
			exit(EXIT_FAILURE);
		}

#if defined(__PARA)
	}

	// MPI_Barrier(MPI_COMM_WORLD);
	mybarrier_();

	for (i = 0; i < ngpus_per_process; i++) {
#endif

		// see cuda_env.h for a description of the hack
		// this does *NOT* work if everything is not performed at the beginning...
#if defined(__CUDA_GET_MEM_HACK)
		free = (size_t)  __GPU_MEM_AMOUNT_HACK__;
#else
		cudaMemGetInfo((size_t*)&free,(size_t*)&total);
#endif

#if defined(__CUDA_DEBUG)
#if defined(__PARA)
		printf("[GPU %d - rank: %d] before: %lu (total: %lu)\n", qe_gpu_bonded[i], lRank, (unsigned long)free, (unsigned long)total); fflush(stdout);
#else
		printf("[GPU %d] before: %lu (total: %lu)\n", qe_gpu_bonded[i], (unsigned long)free, (unsigned long)total); fflush(stdout);
#endif
#endif

#if defined(__PARA)
		cuda_memory_allocated[i] = (size_t) ((((free * __SCALING_MEM_FACTOR__ ) * 16.0) / 16.0) / procs_per_gpu);
		cuda_memory_unused[i] = cuda_memory_allocated[i];
#else
		cuda_memory_allocated[i] = (size_t) (((free * __SCALING_MEM_FACTOR__ ) * 16.0) / 16.0);
		cuda_memory_unused[i] = cuda_memory_allocated[i];
#endif


		// Temporary hack...
#if !defined(__PHIGEMM_NOALLOC)
		/* Do real allocation */
		ierr = cudaMalloc ( (void**) &(dev_scratch_QE[i]), (size_t) cuda_memory_allocated[i] );
		if ( ierr != cudaSuccess) {
			fprintf( stderr, "\nError in memory allocation, program will be terminated (%d)!!! Bye...\n\n", ierr );
			exit(EXIT_FAILURE);
		}
#endif

#if defined(__PARA)
	}

	// MPI_Barrier(MPI_COMM_WORLD);
	mybarrier_();

	for (i = 0; i < ngpus_per_process; i++) {
#endif

		/* It can be useful to track this information... */
#if defined(__CUDA_GET_MEM_HACK)
		free = __GPU_MEM_AMOUNT_HACK__ - cuda_memory_allocated[i];
#else
		cudaMemGetInfo((size_t*)&free,(size_t*)&total);
#endif

		device_memory_left[i] = free;

		dev_heap_QE[i] = (char * ) dev_scratch_QE[i] + (32*(cuda_memory_allocated[i]/32));
		device_memory_shift[i] = 0;

#if defined(__CUDA_DEBUG)
#if defined(__PARA)
		printf("[GPU %d - rank: %d] after: %lu (total: %lu)\n", qe_gpu_bonded[i], lRank, (unsigned long)free, (unsigned long)total); fflush(stdout);
#else
		printf("[GPU %d] after: %lu (total: %lu)\n", qe_gpu_bonded[i], (unsigned long)free, (unsigned long)total); fflush(stdout);
#endif
#endif
	}
	
	// Print information on screen
#if defined(__PARA)
	if (lRank == 0) {	
#endif
	printf("\n"); fflush(stdout);
	printf("     *******************************************************************\n\n"); fflush(stdout);
#if defined(__PHIGEMM_HACK_CPUONLY)
	printf("       CPU-version plus call-by-call GEMM profiling"); fflush(stdout);
#else

	printf("       GPU-accelerated Quantum ESPRESSO \n\n"); fflush(stdout);

#if defined(__PARA)
	printf("       parallel      : yes (GPUs per node = %d, GPUs per process = %d)  \n", ngpus_detected, ngpus_per_process); fflush(stdout);
#else
	printf("       parallel      : no (GPUs detected = %d, GPUs used = %d)  \n", ngpus_detected, ngpus_used); fflush(stdout);
#endif

#if defined(__OPENACC)
	printf("       OpenACC       : yes \n"); fflush(stdout);
#endif

#if defined(__CUDA_PINNED)
    printf("       pinned memory : yes \n"); fflush(stdout);
#else
	printf("       pinned memory : no \n"); fflush(stdout);
#endif

#if defined(__MAGMA)
	printf("       MAGMA         : yes \n"); fflush(stdout);
#else
	printf("       MAGMA         : no \n"); fflush(stdout);
#endif

#if defined(__PARA) && defined(__USE_3D_FFT)
	printf("       USE_3D_FFT    : yes (check size(pool)=1) \n"); fflush(stdout);
#else
	printf("       USE_3D_FFT    : no \n"); fflush(stdout);
#endif

#if defined(__DISABLE_CUDA_ADDUSDENS) || defined(__DISABLE_CUDA_VLOCPSI) || defined(__DISABLE_CUDA_NEWD) || defined(__PHIGEMM_DISABLE_SPECIALK)
	printf("       # DEBUG MODE #\n");fflush(stdout);
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
#if defined(__PHIGEMM_DISABLE_SPECIALK)
	printf("         phiGEMM special-k = disabled\n");fflush(stdout);
#else
	printf("         phiGEMM special-k = enabled\n");fflush(stdout);
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

extern "C"  void initcudaenv_()
{

	// In case of serial (default)
	int lRankThisNode = 0, lSizeThisNode = 1, lRank = -1;

#if defined(__PARA)
	paralleldetect_(&lRankThisNode, &lSizeThisNode, &lRank);
#endif

	gpubinding_(lRankThisNode, lSizeThisNode, lRank);

	preallocatedevicememory_(lRank);

#if defined(__PHIGEMM)
	initphigemm_(lRank);
#endif
}

void deallocatedevicememory_(){

	int ierr = 0;

#if defined(__CUDA_DEBUG)
	int i;
	size_t free, total;
#endif

	ierr = cudaFree ( dev_scratch_QE[0] );

	if(ierr != cudaSuccess) {
		fprintf( stderr, "\nError in memory release, program will be terminated!!! Bye...\n\n" );
		exit(EXIT_FAILURE);
	}

#if defined(__CUDA_DEBUG)
		cudaMemGetInfo((size_t*)&free,(size_t*)&total);
		for (i = 0; i < ngpus_per_process; i++) {
			printf("[GPU %d] free: %lu (total: %lu)\n", qe_gpu_bonded[i], (unsigned long)free, (unsigned long)total); fflush(stdout);
		}
#endif

}

extern "C" void closecudaenv_()
{
#if !defined(__PHIGEMM_NOALLOC)
	deallocatedevicememory_();
#endif

#if defined(__PHIGEMM)
	phiGemmShutdown();
#endif

}
