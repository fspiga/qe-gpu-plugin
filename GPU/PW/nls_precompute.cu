/*
 * Copyright (C) 2001-2013 Quantum ESPRESSO Foundation
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 */

/*    ***********************************    *
 *    * IT WORKS ONLY IF __CUDA_NOALLOC *    *
 *    ***********************************    * */

#include <stdlib.h>
#include <stdio.h>

#include "cuda_env.h"

__global__ void build_psic_gamma_index(const  int * __restrict nls, const  int * __restrict nlsm, const  int * __restrict igk, int * psic_index_nls, int * psic_index_nlsm, const int n ){

	register int ix = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	// TODO: Fetch in shared memory igk[ ix ]
	// TODO: In-place index calculation

	if ( ix < n ) {
		psic_index_nls[ix] = ( nls[ igk[ ix ] - 1 ] - 1 ) * 2;
		psic_index_nlsm[ix] = ( nlsm[ igk[ ix ] - 1 ] - 1 ) * 2;
	}

	// TODO: Copy from shared to global memory
}

__global__ void build_psic_k_index(const  int * __restrict nls, const  int * __restrict igk, int * psic_index_nls, const int n ){

	register int ix = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	// TODO: Fetch in shared memory igk[ ix ]
	// TODO: In-place index calculation

	if ( ix < n ) {
		psic_index_nls[ix] = ( nls[ igk[ ix ] - 1 ] - 1 ) * 2;
	}

	// TODO: Copy from shared to global memory

}

extern "C" int nls_precompute_k_( int * ptr_n, int * igk, int * nls, int * ptr_ngms)
{
	void * igk_D, * nls_D; // int*

	int  blocksPerGrid;
	int n = (* ptr_n);
	int ngms = (* ptr_ngms);
#if defined(__CUDA_NOALLOC)
	int ierr;
#endif
	size_t shift;

#if defined(__CUDA_DEBUG)
	printf("[NLS_PRECOMPUTE_K] Enter (n = %d)\n",n); fflush(stdout);
#endif

	blocksPerGrid = ( n + __CUDA_TxB_VLOCPSI_BUILD_PSIC__ - 1) / __CUDA_TxB_VLOCPSI_BUILD_PSIC__ ;
	if ( blocksPerGrid > __CUDA_MAXNUMBLOCKS__) {
		fprintf( stderr, "\n[NLS_PRECOMPUTE_K] build_psic_k_index cannot run, blocks requested ( %d ) > blocks allowed!!!", blocksPerGrid );
		return 1;
	}

	cudaSetDevice(qe_gpu_bonded[0]);

	// Have I already use preloaded_nls_D previously? Yes, then clean
	if (preloaded_nls_D != NULL){
		/* Deallocating... */
#if defined(__CUDA_DEBUG)
	    printf("[NLS_PRECOMPUTE_K] Detected previous index computation, deallocate before recompute  \n"); fflush(stdout);
#endif
		ierr = cudaFree ( preloaded_nls_D );
		qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_K", "error in memory release");

		qe_gpu_mem_unused[0] = qe_gpu_mem_tot[0];
	}

	shift = ( (n % 2 == 0)? n : n + 1 )*sizeof(int);

	if ( shift > qe_gpu_mem_unused[0] ) {
		fprintf( stderr, "\n[NLS_PRECOMPUTE_K] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", shift, qe_gpu_mem_unused[0] );
		return 1;
	}

	ierr = cudaMalloc ( (void**) &preloaded_nls_D, (size_t) n*sizeof(int) );
	qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_K", "error in memory allocation (preload_nls_D)");


#if defined(__CUDA_DEBUG)
	printf("[NLS_PRECOMPUTE_K] preloaded_nls_D allocated (used = %lu byte)\n",(size_t) n*sizeof(int)); fflush(stdout);
#endif

#if defined(__CUDA_KERNEL_MEMSET)
	qecudaSafeCall( cudaMemset( (int *) preloaded_nls_D, 0, (size_t) n*sizeof(int) ) );
#endif

	qe_gpu_mem_unused[0] = qe_gpu_mem_tot[0] - shift;

	/* Do real allocation */
	ierr = cudaMalloc ( (void**) &(qe_dev_scratch[0]), (size_t) qe_gpu_mem_unused[0] );
	qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_K", "error in memory allocation (qe_dev_scratch)");


#if defined(__CUDA_KERNEL_MEMSET)
	qecudaSafeCall( cudaMemset( qe_dev_scratch[0], 0, (size_t) qe_gpu_mem_unused[0] ) );
#endif

	shift = 0;
	nls_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( (ngms % 2 == 0)? ngms : ngms + 1 )*sizeof(int);
	igk_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( (n % 2 == 0)? n : n + 1 )*sizeof(int);

#if defined(__CUDA_DEBUG)
	printf("[NLS_PRECOMPUTE_K] qe_dev_scratch allocated (used = %lu byte)\n", shift); fflush(stdout);
#endif

	if ( shift > qe_gpu_mem_unused[0] ) {

		fprintf( stderr, "\n[NLS_PRECOMPUTE_K] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", shift, qe_gpu_mem_unused[0] );

		/* Deallocating... */
		ierr = cudaFree ( preloaded_nls_D );
		qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_K", "error in memory release (preload_nls_D)");

		ierr = cudaFree ( qe_dev_scratch[0] );
		qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_K", "error in memory release (qe_dev_scratch)");

		return 1;
	}

	qecudaSafeCall( cudaMemcpy( nls_D, nls,  sizeof( int ) * ngms, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( igk_D, igk,  sizeof( int ) * n, cudaMemcpyHostToDevice ) );

	// blocksPerGrid = ( n + __CUDA_TxB_VLOCPSI_BUILD_PSIC__ - 1) / __CUDA_TxB_VLOCPSI_BUILD_PSIC__ ;
	build_psic_k_index<<<blocksPerGrid, __CUDA_TxB_VLOCPSI_BUILD_PSIC__ >>>( (int *) nls_D, (int *) igk_D, (int *) preloaded_nls_D, n );
	qecudaGetLastError("kernel launch failure");

#if defined(__CUDA_DEBUG)
	printf("[NLS_PRECOMPUTE_K] preloaded_nls_D populated\n"); fflush(stdout);
#endif

	/* Deallocating... but NOT preloaded_nls_D */
	ierr = cudaFree ( qe_dev_scratch[0] );
	qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_K", "error in memory release (qe_dev_scratch)");

	// guard
	cudaDeviceSynchronize();

#if defined(__CUDA_DEBUG)
	printf("[NLS_PRECOMPUTE_K] qe_dev_scratch cleaned\n"); fflush(stdout);
#endif

	return 0;
}

extern "C" int nls_precompute_gamma_( int * ptr_n, int * igk, int * nls,  int * nlsm, int * ptr_ngms, int * ptr_ngm)
{
	void * igk_D, * nls_D , * nlsm_D ; // int*

	int  blocksPerGrid;
	int n = (* ptr_n);
	int ngms = (* ptr_ngms);
	int ngm = (* ptr_ngm);
#if defined(__CUDA_NOALLOC)
	int ierr;
#endif
	size_t shift;

#if defined(__CUDA_DEBUG)
	printf("[NLS_PRECOMPUTE_GAMMA] Enter (n = %d)\n",n); fflush(stdout);
#endif

	blocksPerGrid = ( n + __CUDA_TxB_VLOCPSI_BUILD_PSIC__ - 1) / __CUDA_TxB_VLOCPSI_BUILD_PSIC__ ;
	if ( blocksPerGrid > __CUDA_MAXNUMBLOCKS__) {
		fprintf( stderr, "\n[NLS_PRECOMPUTE_GAMMA] build_psic_gamma_index cannot run, blocks requested ( %d ) > blocks allowed!!!", blocksPerGrid );
		return 1;
	}

	cudaSetDevice(qe_gpu_bonded[0]);

	/* ASSUMPTION: preloaded_nls_D and preloaded_nlsm_D works *
	 *             always in pair in gamma calculation        */

	// Have I already use preloaded_nls_D previously? Yes, then clean
	if (preloaded_nls_D != NULL){
#if defined(__CUDA_DEBUG)
	    printf("[NLS_PRECOMPUTE_GAMMA] Detected previous index computation, deallocate before recompute  \n"); fflush(stdout);
#endif
		ierr = cudaFree ( preloaded_nls_D );
		qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_GAMMA", "error in memory release (preload_nls_D)");

		ierr = cudaFree ( preloaded_nlsm_D );
		qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_GAMMA", "error in memory release (preload_nlsm_D)");

		qe_gpu_mem_unused[0] = qe_gpu_mem_tot[0];
	}

	shift = ( (n % 2 == 0)? n : n + 1 )*sizeof(int)*2;

	if ( shift > qe_gpu_mem_unused[0] ) {
		fprintf( stderr, "\n[NLS_PRECOMPUTE_K] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", shift, qe_gpu_mem_unused[0] );
		return 1;
	}

	ierr = cudaMalloc ( (void**) &preloaded_nls_D, (size_t) n*sizeof(int) );
	qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_GAMMA", "error in memory allocation (preload_nls_D)");

#if defined(__CUDA_DEBUG)
	printf("[NLS_PRECOMPUTE_GAMMA] preloaded_nls_D allocated (used = %lu byte)\n",(size_t) n*sizeof(int)); fflush(stdout);
#endif

	ierr = cudaMalloc ( (void**) &preloaded_nlsm_D, (size_t) n*sizeof(int) );
	qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_GAMMA", "error in memory allocation (preload_nlsm_D)");

#if defined(__CUDA_DEBUG)
	printf("[NLS_PRECOMPUTE_GAMMA] preloaded_nlsm_D allocated (used = %lu byte)\n",(size_t) n*sizeof(int)); fflush(stdout);
#endif

#if defined(__CUDA_KERNEL_MEMSET)
	qecudaSafeCall( cudaMemset( (int *) preloaded_nls_D, 0, (size_t) n*sizeof(int) ) );
	qecudaSafeCall( cudaMemset( (int *) preloaded_nlsm_D, 0, (size_t) n*sizeof(int) ) );
#endif

	qe_gpu_mem_unused[0] = qe_gpu_mem_tot[0] - shift;

	/* Do real allocation */
	ierr = cudaMalloc ( (void**) &(qe_dev_scratch[0]), (size_t) qe_gpu_mem_unused[0] );
	qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_GAMMA", "error in memory allocation (qe_dev_scratch)");

#if defined(__CUDA_KERNEL_MEMSET)
	qecudaSafeCall( cudaMemset( qe_dev_scratch[0], 0, (size_t) qe_gpu_mem_unused[0] ) );
#endif

	shift = 0;
	nls_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( (ngms % 2 == 0)? ngms : ngms + 1 )*sizeof(int);
	nlsm_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( (ngm % 2 == 0)? ngm : ngm + 1 )*sizeof(int);
	igk_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( (n % 2 == 0)? n : n + 1 )*sizeof(int);

#if defined(__CUDA_DEBUG)
	printf("[NLS_PRECOMPUTE_GAMMA] qe_dev_scratch allocated (used = %lu byte)\n", shift); fflush(stdout);
#endif

	if ( shift > qe_gpu_mem_unused[0] ) {

		fprintf( stderr, "\n[NLS_PRECOMPUTE_GAMMA] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", shift, qe_gpu_mem_unused[0] );

		ierr = cudaFree ( preloaded_nls_D );
		qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_GAMMA", "error in memory release (preload_nls_D)");

		ierr = cudaFree ( preloaded_nlsm_D );
		qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_GAMMA", "error in memory release (preload_nlsm_D)");

		ierr = cudaFree ( qe_dev_scratch[0] );
		qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_GAMMA", "error in memory allocation (qe_de_scratch)");

		return 1;
	}

	qecudaSafeCall( cudaMemcpy( nls_D, nls,  sizeof( int ) * ngms, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( nlsm_D, nlsm,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( igk_D, igk,  sizeof( int ) * n, cudaMemcpyHostToDevice ) );

	build_psic_gamma_index<<<blocksPerGrid, __CUDA_TxB_VLOCPSI_BUILD_PSIC__ >>>( (int *) nls_D, (int *) nlsm_D, (int *) igk_D, (int *) preloaded_nls_D, (int *) preloaded_nlsm_D, n );
	qecudaGetLastError("kernel launch failure");

#if defined(__CUDA_DEBUG)
	printf("[NLS_PRECOMPUTE_GAMMA] preloaded_nls_D/preloaded_nlsm_D populated\n"); fflush(stdout);
#endif

	/* Deallocating... but NOT preloaded_nls_D/preloaded_nlsm_D */
	ierr = cudaFree ( qe_dev_scratch[0] );
	qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_GAMMA", "error in memory release (qe_dev_scratch)");

	// guard
	cudaDeviceSynchronize();

#if defined(__CUDA_DEBUG)
	printf("[NLS_PRECOMPUTE_GAMMA] qe_dev_scratch cleaned\n"); fflush(stdout);
#endif

	return 0;
}


extern "C" int nls_precompute_k_cleanup_( )
{
	int ierr;

	ierr = cudaFree ( preloaded_nls_D );
	qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_K_CLEANUP", "error in memory release (preloaded_nls_D)");

	preloaded_nls_D = NULL;
	qe_gpu_mem_unused[0] = qe_gpu_mem_tot[0];
	// qe_dev_scratch[0] = qe_dev_zero_scratch[0];

#if defined(__CUDA_DEBUG)
	printf("[NLS_PRECOMPUTE_K_CLEANUP] preloaded_nls_D cleaned \n"); fflush(stdout);
#endif

	return 0;
}

extern "C" int nls_precompute_gamma_cleanup_( )
{
	int ierr;

	ierr = cudaFree ( preloaded_nls_D );
	qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_GAMMA_CLEANUP", "error in memory release (preloaded_nls_D)");

#if defined(__CUDA_DEBUG)
	printf("[NLS_PRECOMPUTE_GAMMA_CLEANUP] preloaded_nls_D cleaned \n"); fflush(stdout);
#endif

	ierr = cudaFree ( preloaded_nlsm_D );
	qecudaGenericErr((cudaError_t) ierr, "NLS_PRECOMPUTE_GAMMA_CLEANUP", "error in memory release (preloaded_nlsm_D)");

#if defined(__CUDA_DEBUG)
	printf("[NLS_PRECOMPUTE_GAMMA_CLEANUP] preloaded_nlsm_D cleaned \n"); fflush(stdout);
#endif

	preloaded_nls_D = NULL;
	preloaded_nlsm_D = NULL;

	qe_gpu_mem_unused[0] = qe_gpu_mem_tot[0];
	// qe_dev_scratch[0] = qe_dev_zero_scratch[0];

	return 0;
}



