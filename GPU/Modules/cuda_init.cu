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

qeCudaMemDevPtr qe_dev_scratch;
qeCudaMemDevPtr qe_dev_zero_scratch;

qeCudaMemSizes qe_gpu_mem_tot;
qeCudaMemSizes qe_gpu_mem_unused;

qeCudaDevicesBond qe_gpu_bonded;

cudaStream_t  qecudaStreams[ MAX_QE_GPUS ];
cublasHandle_t qecudaHandles[ MAX_QE_GPUS ];

// Pre-loaded data-structure
int * preloaded_nlsm_D, * preloaded_nls_D;

// FFT plans (works only with "-D__CUDA_NOALLOC")
cufftHandle qeCudaFFT_dfftp, qeCudaFFT_dffts;

// global useful information
long ngpus_detected;
long ngpus_used;
long ngpus_per_process;
long procs_per_gpu;


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

	procs_per_gpu = 1;

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

#endif
}

#if defined(__PHIGEMM)
extern "C" void initphigemm_(int lRank){
	/* Compatibility with CUDA 4.x (latest phiGEMM): */

#if defined(__CUDA_NOALLOC)
//	phiGemmInit(ngpus_per_process , NULL, (qeCudaMemSizes*)&qe_gpu_mem_tot, (int *)qe_gpu_bonded, lRank);
	phiGemmInit(ngpus_per_process , NULL, NULL, (int *)qe_gpu_bonded, lRank);
#else
	phiGemmInit(ngpus_per_process , (qeCudaMemDevPtr*)&qe_dev_scratch, (qeCudaMemSizes*)&qe_gpu_mem_tot, (int *)qe_gpu_bonded, lRank);
#endif
}
#endif

extern "C" void preallocatedevicememory_(int lRank){

	int ierr = 0;
	int i;

	size_t free, total;

	preloaded_nls_D = NULL;
	preloaded_nlsm_D = NULL;

	for (i = 0; i < ngpus_per_process; i++) {

		/* query the real free memory, taking into account the "stack" */
		if ( cudaSetDevice(qe_gpu_bonded[i]) != cudaSuccess) {
			printf("*** ERROR *** cudaSetDevice(%d) failed!", qe_gpu_bonded[i] ); fflush(stdout);
			exit(EXIT_FAILURE);
		}

		qe_gpu_mem_tot[i] = (size_t) 0;

		ierr = cudaMalloc ( (void**) &(qe_dev_scratch[i]), qe_gpu_mem_tot[i] );
		if ( ierr != cudaSuccess) {
			fprintf( stderr, "\nError in (first zero) memory allocation , program will be terminated!!! Bye...\n\n");
			exit(EXIT_FAILURE);
		}

#if defined(__PARA)
	}

	mybarrier_();

	for (i = 0; i < ngpus_per_process; i++) {

		if ( cudaSetDevice(qe_gpu_bonded[i]) != cudaSuccess) {
			printf("*** ERROR *** cudaSetDevice(%d) failed!", qe_gpu_bonded[i] ); fflush(stdout);
			exit(EXIT_FAILURE);
		}
#endif

		cudaMemGetInfo((size_t*)&free,(size_t*)&total);

#if defined(__CUDA_DEBUG)
#if defined(__PARA)
		printf("[GPU %d - rank: %d] before: %lu (total: %lu)\n", qe_gpu_bonded[i], lRank, (unsigned long)free, (unsigned long)total); fflush(stdout);
#else
		printf("[GPU %d] before: %lu (total: %lu)\n", qe_gpu_bonded[i], (unsigned long)free, (unsigned long)total); fflush(stdout);
#endif
#endif

		qe_gpu_mem_tot[i] = (size_t) ((free * __SCALING_MEM_FACTOR__ ) / procs_per_gpu);
		qe_gpu_mem_unused[i] = qe_gpu_mem_tot[i];


#if !defined(__CUDA_NOALLOC)
		/* Do real allocation */
		ierr = cudaMalloc ( (void**) &(qe_dev_scratch[i]), (size_t) qe_gpu_mem_tot[i] );
		if ( ierr != cudaSuccess) {
			fprintf( stderr, "\nError in memory allocation, program will be terminated (%d)!!! Bye...\n\n", ierr );
			exit(EXIT_FAILURE);
		}

		qe_dev_zero_scratch[i] = qe_dev_scratch[i];
#endif

#if defined(__PARA) && defined(__CUDA_DEBUG)
	}

	mybarrier_();

	for (i = 0; i < ngpus_per_process; i++) {
#endif

#if defined(__CUDA_DEBUG)
#if defined(__PARA)
		printf("[GPU %d - rank: %d] after: %lu (total: %lu)\n", qe_gpu_bonded[i], lRank, (unsigned long)free, (unsigned long)total);
		fflush(stdout);
#else
		printf("[GPU %d] after: %lu (total: %lu)\n", qe_gpu_bonded[i], (unsigned long)free, (unsigned long)total);
		fflush(stdout);
#endif
#endif
	}
	
#if defined(__PARA)
	// Is this necessary?
	mybarrier_();
#endif

	// Print CUDA header
	print_cuda_header_(lRank);
}

extern "C"  void initStreams_()
{
	int ierr, i;

	for (i = 0; i < ngpus_per_process; i++) {
		ierr = cudaStreamCreate( &qecudaStreams[ i ] );
		qecudaGenericErr((cudaError_t) ierr, "INIT_CUDA", "error during stream creation");

		if ( cublasCreate( &qecudaHandles[ i ] ) != CUBLAS_STATUS_SUCCESS ) {
			printf("\n*** CUDA VLOC_PSI_K *** ERROR *** cublasInit() for device %d failed!",qe_gpu_bonded[i]);
			fflush(stdout);
			exit(EXIT_FAILURE);
		}
	}
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

	initStreams_();
}

void deallocatedevicememory_(){

	int ierr = 0;

	// Assumed 1 GPU per process...
	ierr = cudaFree ( qe_dev_scratch[0] );

	if(ierr != cudaSuccess) {
		fprintf( stderr, "\nError in memory release, program will be terminated!!! Bye...\n\n" );
		exit(EXIT_FAILURE);
	}

	return;
}

extern "C"  void destroyStreams_()
{
	int ierr, i;

	for (i = 0; i < ngpus_per_process; i++) {
		ierr = cudaStreamDestroy( qecudaStreams[ i ] );
		qecudaGenericErr((cudaError_t) ierr, "INIT_CUDA", "error during stream creation");

		if ( cublasDestroy( qecudaHandles[ i ] ) != CUBLAS_STATUS_SUCCESS ) {
			printf("\n*** CUDA INIT_CUDA *** ERROR *** cublasDestroy() for device %d failed!",qe_gpu_bonded[i]);
			fflush(stdout);
			exit(EXIT_FAILURE);
		}
	}
}

extern "C" void closecudaenv_()
{
#if !defined(__CUDA_NOALLOC)
	deallocatedevicememory_();
#endif

	destroyStreams_();

#if defined(__PHIGEMM)
	phiGemmShutdown();
#endif

}
