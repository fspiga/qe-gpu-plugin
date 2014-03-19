/*****************************************************************************\
 * Copyright (C) 2011-2014 Quantum ESPRESSO Foundation
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 * Filippo Spiga (filippo.spiga@quantum-espresso.org)
\*****************************************************************************/


#include <stdlib.h>
#include <stdio.h>

#include "cuda_env.h"

typedef double fftw_complex[2];

template <unsigned int N>
__global__ void debugMark() {
   //This is only for putting marks into the profile.
}
__global__ void kernel_vec_prod_k( double *a, const  double * __restrict b, int dimx )
{	   
	register int ix = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	register double sup;
	register int ii = ix / 2;

	if ( ix < ( dimx * 2 ) ) {
		sup = a[ix] * b[ii];
		a[ix] = sup;
	}

	return;
}

__global__ void kernel_init_psic_k( const  int * __restrict nls, const  int * __restrict igk, const  double * __restrict psi, double *psic, const int n, const int lda, const int ibnd, const int inx_max )
{	   
	int ix = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	// int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int psic_index_nls, psi_index = ( ix + ( ibnd * lda ) ) * 2;

   //TODO: Make the writes coalesced since caches can't help us there. 
	if ( ix < n ) {
      //TODO precompute nsl_igk s.t. nsl_igk[ ix ] = nsl[ igk[ ix ] ]
      //TODO also, possibly, precompute nsl_igk_inv s.t. ix = nsl_igk_inv[ nsl[ igk[ ix ] ] ]
		psic_index_nls = ( nls[ igk[ ix ] - 1 ] - 1 ) * 2;
      //TODO: Use cuDoubleComplex
                if (psic_index_nls <= inx_max) {
		   psic[ psic_index_nls ] = psi[ psi_index ];
		   psic[ psic_index_nls + 1 ] = psi[ psi_index + 1 ];
      }
	}

	return;
}

__global__ void kernel_save_hpsi_k( const  int * __restrict nls, const  int * __restrict igk, double * hpsi, const  double * __restrict psic, const int n, const int ibnd, const int lda )
{	   
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int pos = ibnd * lda;
	int psic_index_nls, psi_index = (ix + pos) * 2;

	if ( ix < (n) ) {
		psic_index_nls = (nls[ igk[ ix ] - 1 ] - 1) * 2;
		hpsi[ psi_index ] += psic[ psic_index_nls ];
		hpsi[ psi_index + 1 ] += psic[ psic_index_nls + 1 ];
	}

	return;
}


extern "C" int vloc_psi_cuda_k_( int * ptr_lda, int * ptr_nrxxs, int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s, int * ptr_n, int * ptr_m, cufftDoubleComplex * psi, double * v, fftw_complex * hpsi, int * igk, int * nls, int * ptr_ngms)
{
#define MAX_STREAMS 1
	cufftHandle p_global[MAX_STREAMS];
	fftw_complex * psic = NULL;

	void * psic_D, * psi_D, * hpsi_D; // cufftDoubleComplex *
	void * v_D; // double *
	void * igk_D, * nls_D; // int*

	double tscale;

	int n = (* ptr_n);
	int m = (* ptr_m);
	int nr1s = (* ptr_nr1s);
	int nr2s = (* ptr_nr2s);
	int nr3s = (* ptr_nr3s);
	int nrxxs = (* ptr_nrxxs);
	int ngms = (* ptr_ngms);
	int lda = (* ptr_lda);
#if defined(__CUDA_NOALLOC)
	int ierr;
#endif
	int size_psic = nr1s * nr2s * nr3s;
	int ibnd;

	dim3 threads2_psic(qe_gpu_kernel_launch[0].__CUDA_TxB_VLOCPSI_PSIC);
	dim3 grid2_psic( qe_compute_num_blocks(n, threads2_psic.x) );

	dim3 threads2_prod(qe_gpu_kernel_launch[0].__CUDA_TxB_VLOCPSI_PROD);
	dim3 grid2_prod( qe_compute_num_blocks((nrxxs * 2), threads2_prod.x) );

#if defined(__CUDA_DEBUG)
	printf("[CUDA DEBUG] VLOC_PSI_K\n"); fflush(stdout);
#endif

	cudaStream_t vlocpsiStreams[MAX_STREAMS];

	for ( int q = 0; q < MAX_STREAMS; q++ ) 
		cudaStreamCreate( &vlocpsiStreams[q] );

	if ( grid2_psic.x > qe_gpu_kernel_launch[0].__MAXNUMBLOCKS) {
		fprintf( stderr, "\n[VLOC_PSI_K] kernel_init_psic_k cannot run, blocks requested ( %d ) > blocks allowed!!!", grid2_psic.x );
		return 1;
	}

	if ( grid2_prod.x > qe_gpu_kernel_launch[0].__MAXNUMBLOCKS) {
		fprintf( stderr, "\n[VLOC_PSI_K] kernel_vec_prod cannot run, blocks requested ( %d ) > blocks allowed!!!", grid2_prod.x );
		return 1;
	}

	cudaSetDevice(qe_gpu_bonded[0]);

#if defined(__CUDA_NOALLOC)
	/* Do real allocation */
	ierr = cudaMalloc ( (void**) &(qe_dev_scratch[0]), (size_t) qe_gpu_mem_unused[0] );
	qecudaGenericErr((cudaError_t) ierr, "VLOC_PSI_K", "error in memory allocation (qe_dev_scratch)");
#endif

   
	int n_streams = MAX_STREAMS+1;
	size_t shift = 0;
	do {
		shift = 0;
		n_streams--;
		psic_D = (char*) qe_dev_scratch[0] + shift;
		shift += ( size_psic * n_streams)*sizeof( cufftDoubleComplex );
		hpsi_D = (char*) qe_dev_scratch[0] + shift;
		shift += ( lda * m )*sizeof( cufftDoubleComplex );
		psi_D = (char*) qe_dev_scratch[0] + shift;
		shift += ( lda * m )*sizeof( cufftDoubleComplex );
		v_D = (char*) qe_dev_scratch[0] + shift;
		shift += ( nrxxs )*sizeof( double );
		nls_D = (char*) qe_dev_scratch[0] + shift;
		shift += ( (ngms % 2 == 0)? ngms : ngms + 1 )*sizeof(int);
		igk_D = (char*) qe_dev_scratch[0] + shift;
		shift += ( (n % 2 == 0)? n : n + 1 )*sizeof(int);
		// now	shift contains the amount of byte required on the GPU to compute
	} while (n_streams > 0 && shift > qe_gpu_mem_unused[0]);

	if ( n_streams < 1 ) {
		fprintf( stderr, "\n[VLOC_PSI_K] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", shift, qe_gpu_mem_unused[0] );
#if defined(__CUDA_NOALLOC)
		/* Deallocating... */
		ierr = cudaFree ( qe_dev_scratch[0] );
		qecudaGenericErr((cudaError_t) ierr, "VLOC_PSI_K", "error memory release (qe_dev_scratch)");
#endif
		return 1;
	}

	qecudaSafeCall( cudaMemcpy( psi_D, psi,  sizeof( cufftDoubleComplex ) * lda * m, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( v_D, v,  sizeof( double ) * nrxxs, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( nls_D, nls,  sizeof( int ) * ngms, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( igk_D, igk,  sizeof( int ) * n, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( hpsi_D, hpsi,  sizeof( cufftDoubleComplex ) * lda * m, cudaMemcpyHostToDevice ) );

	for (int q=0;q<n_streams;q++) qecheck_cufft_call( cufftPlan3d( &p_global[q], nr3s, nr2s,  nr1s, CUFFT_Z2Z ) );
	//qecheck_cufft_call( cufftSetStream(p_global,qecudaStreams[ 0 ]) );

	qecudaSafeCall( cudaHostAlloc ( (void**) &psic, n_streams * size_psic * sizeof( fftw_complex ), cudaHostAllocPortable ) );

	int this_stream = -1;
	double * this_psic_D;
	for ( ibnd =  0; ibnd < m; ibnd = ibnd + 1) {

		this_stream = (this_stream+1)%n_streams;
		cudaStreamSynchronize(vlocpsiStreams[this_stream]);
		this_psic_D = ((double*)psic_D) + 2 * size_psic * this_stream;

      //TODO : can this be done in kernel_init_psic instead?
		qecudaSafeCall( cudaMemsetAsync( this_psic_D, 0, size_psic * sizeof( cufftDoubleComplex ), vlocpsiStreams[this_stream]) );

		kernel_init_psic_k<<< grid2_psic, threads2_psic, 0, vlocpsiStreams[this_stream] >>>(
				(int *) nls_D, (int *) igk_D, (double *) psi_D, (double *) this_psic_D, n, lda, ibnd, size_psic*2 );
		qecudaGetLastError("kernel launch failure");

		qecheck_cufft_call( cufftSetStream( p_global[this_stream], vlocpsiStreams[this_stream] ) );
		qecheck_cufft_call( cufftExecZ2Z( p_global[this_stream], (cufftDoubleComplex *) this_psic_D, (cufftDoubleComplex *) this_psic_D, CUFFT_INVERSE ) );

		kernel_vec_prod_k<<< grid2_prod, threads2_prod, 0, vlocpsiStreams[this_stream] >>>(
				(double *) this_psic_D, (double *) v_D , nrxxs );
		qecudaGetLastError("kernel launch failure");

		qecheck_cufft_call( cufftExecZ2Z( p_global[this_stream], (cufftDoubleComplex *) this_psic_D,
				(cufftDoubleComplex *)this_psic_D, CUFFT_FORWARD ) );

		tscale = 1.0 / (double) ( size_psic );

		cublasSetStream( qecudaHandles[ 0 ], vlocpsiStreams[this_stream] );
		cublasZdscal(qecudaHandles[ 0 ], size_psic, &tscale, (cuDoubleComplex *) this_psic_D, 1);


		//TODO make sure only one version of this runs at a time
                //TODO combine nls[igk[]] into a single table
		kernel_save_hpsi_k<<<grid2_psic, threads2_psic, 0, vlocpsiStreams[this_stream]>>>( 
                                    (int*)nls_D, (int*)igk_D, (double*)hpsi_D, (double*)this_psic_D, n, ibnd, lda );
#if 0
		//wait until the last stream has finished with psic before trying to read it
		cudaStreamSynchronize(vlocpsiStreams[(this_stream+n_streams-1)%n_streams]);
		for ( j = 0; j <  n && ibnd > 0; j++ ) {
			hpsi[ j + ( ( ibnd  - 1 ) * lda ) ][0] += psic[ size_psic * ((this_stream+n_streams-1)%n_streams) + nls [ igk[ j ] - 1  ] - 1 ][0];
			hpsi[ j + ( ( ibnd  - 1 ) * lda ) ][1] += psic[ size_psic * ((this_stream+n_streams-1)%n_streams) + nls [ igk[ j ] - 1  ] - 1 ][1];
		}

		qecudaGetLastError("stream synchronize failure");
		qecudaSafeCall( cudaMemcpyAsync( (cufftDoubleComplex*)psic + size_psic * this_stream, this_psic_D, sizeof( cufftDoubleComplex ) * size_psic, 
                                       cudaMemcpyDeviceToHost, vlocpsiStreams[this_stream] ) );
#endif
//	    for( j = 0; j <  n; j++ ) {
//	      hpsi[ j + ( ibnd * lda ) ][0] += psic[ nls [ igk[ j ] - 1  ] - 1 ][0];
//	      hpsi[ j + ( ibnd * lda ) ][1] += psic[ nls [ igk[ j ] - 1  ] - 1 ][1];
//	    }

	}
	for (int q = 0; q < n_streams; q++) cudaStreamSynchronize(vlocpsiStreams[q]);
        qecudaSafeCall( cudaMemcpy ( (cufftDoubleComplex*)hpsi, hpsi_D, sizeof(cufftDoubleComplex) * lda * m, cudaMemcpyDeviceToHost) );


#if 0
	for ( j = 0; j <  n; j++ ) {
		hpsi[ j + ( ( m - 1 ) * lda ) ][0] += psic[ size_psic * this_stream + nls [ igk[ j ] - 1  ] - 1 ][0];
		hpsi[ j + ( ( m - 1 ) * lda ) ][1] += psic[ size_psic * this_stream + nls [ igk[ j ] - 1  ] - 1 ][1];
	}   
#endif
#if 0
	for (int q = 0; q < lda * m * 2; q+=697) {
		if (abs(((double*)hpsi)[q]) > 0.00000000001) printf("hpsi[%d].x = %25.8e\n", q, ((double*)hpsi)[q]);
		if (abs(((double*)hpsi)[q+1]) > 0.00000000001) printf("hpsi[%d].y = %25.8e\n", q, ((double*) hpsi)[q+1]);
	}
#endif

	for (int q=0;q<n_streams;q++) qecheck_cufft_call( cufftDestroy(p_global[q]) );
        cudaFreeHost(psic);

#if defined(__CUDA_NOALLOC)
	/* Deallocating... */
	ierr = cudaFree ( qe_dev_scratch[0] );
	qecudaGenericErr((cudaError_t) ierr, "VLOC_PSI_K", "error memory release (qe_dev_scratch)");
#else

#if defined(__CUDA_KERNEL_MEMSET)
	qecudaSafeCall( cudaMemset( qe_dev_scratch[0], 0, (size_t) qe_gpu_mem_unused[0] ) );
#endif

#endif

	return 0;
}

/* This method will be updated in the next build */
#if defined(__CUDA_MULTIPLAN_FFT) && !defined(__CUDA_NOALLOC) && !defined(__CUDA_PRELOAD)
extern "C" void vloc_psi_multiplan_cuda_k_(  int * ptr_lda, int * ptr_nrxxs, int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s, int * ptr_n, int * ptr_m, cufftDoubleComplex * psi, double * v, fftw_complex * hpsi, int * igk, int * nls, int * ptr_ngms)
{
	void * psic_D, * psi_D; // cufftDoubleComplex *
	fftw_complex * psic = NULL;
	void * v_D; // double *
	void * igk_D, * nls_D; // int *

	double tscale;

	int n = (* ptr_n);
	int m = (* ptr_m);
	int nr1s = (* ptr_nr1s);
	int nr2s = (* ptr_nr2s);
	int nr3s = (* ptr_nr3s);
	int nrxxs = (* ptr_nrxxs);
	int ngms = (* ptr_ngms);
	int lda = (* ptr_lda);
    int ierr;
	int dim_multiplepsic, n_singlepsic, n_multiplepsic, size_multiplepsic, i, j, k;
	int array[3];
	int size_psic = nr1s * nr2s * nr3s;
	int last;

	size_t buffer_size = 0L;

	dim3 threads2_psic(qe_gpu_kernel_launch[0].__CUDA_TxB_VLOCPSI_PSIC);
	dim3 grid2_psic( qe_compute_num_blocks(n, threads2_psic.x) );

	dim3 threads2_prod(qe_gpu_kernel_launch[0].__CUDA_TxB_VLOCPSI_PROD);
	dim3 grid2_prod( qe_compute_num_blocks((nrxxs * 2), threads2_prod.x) );

	cudaSetDevice(qe_gpu_bonded[0]);

	dim_multiplepsic = qe_gpu_kernel_launch[0].__NUM_FFT_MULTIPLAN;

	n_multiplepsic = m/dim_multiplepsic;
	n_singlepsic = m%dim_multiplepsic;

	size_multiplepsic = size_psic * dim_multiplepsic;

	buffer_size = size_multiplepsic * sizeof( cufftDoubleComplex ) + sizeof( cufftDoubleComplex ) * n * m + sizeof( int ) * ngms + sizeof( int ) * n + sizeof( double ) * nrxxs;

	if ( buffer_size > qe_gpu_mem_unused[0] ) {
		fprintf( stderr, "\n[VLOC_PSI_K] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", buffer_size, qe_gpu_mem_unused[0] );
		exit(EXIT_FAILURE);
	}

	size_t shift = 0;
	psic_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( size_psic )*sizeof( cufftDoubleComplex );
	psi_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( lda * m )*sizeof( cufftDoubleComplex );
	hpsi_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( lda * m )*sizeof( cufftDoubleComplex );
	v_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( nrxxs )*sizeof( double );
	nls_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( (ngms % 2 == 0)? ngms : ngms + 1 )*sizeof(int);
	igk_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( (n % 2 == 0)? n : n + 1 )*sizeof(int);

	// now	shift contains the amount of byte required on the GPU to compute

	if ( shift > qe_gpu_mem_unused[0] ) {
		fprintf( stderr, "\n[VLOC_PSI_K] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", shift, qe_gpu_mem_unused[0] );
		exit(EXIT_FAILURE);
	}

	qecudaSafeCall( cudaMemcpy( nls_D, nls,  sizeof( int ) * ngms, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( igk_D, igk,  sizeof( int ) * n, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( psi_D, psi,  sizeof( cufftDoubleComplex ) * n * m, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( v_D, v,  sizeof( double ) * nrxxs, cudaMemcpyHostToDevice ) );

	array[0] = nr3s;
	array[1] = nr2s;
	array[2] = nr1s;

	cufftHandle p_global;

	if ( n_multiplepsic > 0 ) {

		qecheck_cufft_call( cufftPlanMany( &p_global, 3, array, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, dim_multiplepsic ) );
        qecheck_cufft_call( cufftSetStream(p_global,qecudaStreams[ 0 ]));

		qecudaSafeCall( cudaHostAlloc ( (void**) &psic, size_multiplepsic * sizeof( fftw_complex ), cudaHostAllocPortable ) );

		for(j = 0; j< (m-n_singlepsic); j = j + dim_multiplepsic ) {

			qecudaSafeCall( cudaMemset( psic_D, 0, size_psic * dim_multiplepsic * sizeof( cufftDoubleComplex ) ) );

			for( i = 0; i < dim_multiplepsic; i++ ) {
				kernel_init_psic_k<<< grid2_psic, threads2_psic >>>( (int*) nls_D, (int*) igk_D, (double*) psi_D, (double*) psic_D, n, lda, (j+i));
				qecudaGetLastError("kernel launch failure");
			}

			qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex*) psic_D, (cufftDoubleComplex*) psic_D, CUFFT_INVERSE ) );

			for( i = 0; i < dim_multiplepsic; i++ ) {
				kernel_vec_prod_k<<< grid2_proc, threads2_proc >>>( (double*) psic_D, (double*) v_D , nrxxs );
				qecudaGetLastError("kernel launch failure");
			}

			for( i = 0; i < dim_multiplepsic; i++ ) {
				if ( (i+j) > 0 ) {
					for ( k = 0; k <  n ; k++ ) {
						hpsi[ k + ( ( (i+j)  - 1 ) * lda ) ][0] += psic[ nls [ igk[ k ] - 1  ] - 1 ][0];
						hpsi[ k + ( ( (i+j)  - 1 ) * lda ) ][1] += psic[ nls [ igk[ k ] - 1  ] - 1 ][1];
					}
				}
			}

			qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex*) psic_D, (cufftDoubleComplex*) psic_D, CUFFT_FORWARD ) );

			tscale = 1.0 / (double) ( size_psic );
			cublasZdscal(qecudaHandles[ 0 ] , size_psic*dim_multiplepsic, &tscale, (cuDoubleComplex *) psic_D, 1);

			qecudaSafeCall( cudaMemcpy( psic, (cufftDoubleComplex*) psic_D, sizeof( cufftDoubleComplex ) * size_psic * dim_multiplepsic, cudaMemcpyDeviceToHost ) );
		}

		last = dim_multiplepsic*n_multiplepsic;
		for ( k = 0; k <  n; k++ ) {
			hpsi[ k + ( ( last - 1 ) * lda ) ][0] += psic[ nls [ igk[ k ] - 1  ] - 1 ][0];
			hpsi[ k + ( ( last - 1 ) * lda ) ][1] += psic[ nls [ igk[ k ] - 1  ] - 1 ][1];
		}

		qecheck_cufft_call( cufftDestroy(p_global) );

		qecudaSafeCall( cudaMemset( psic_D, 0, size_psic * dim_multiplepsic * sizeof( cufftDoubleComplex ) ) );
		qecudaSafeCall( cudaFreeHost(psic));

	}

	if (n_singlepsic > 0 ) {

		printf("n_singlepsic\n");fflush(stdout);

		qecheck_cufft_call( cufftPlanMany( &p_global, 3, array, NULL, 1, 0, NULL, 1, 0,CUFFT_Z2Z, n_singlepsic ) );
        qecheck_cufft_call( cufftSetStream(p_global,qecudaStreams[ 0 ]) );

		qecudaSafeCall( cudaHostAlloc ( (void**) &psic, n_singlepsic * size_psic * sizeof( cufftDoubleComplex ), cudaHostAllocPortable ) );
		qecudaSafeCall( cudaMemset( psic_D, 0, n_singlepsic * size_psic * sizeof( cufftDoubleComplex ) ) );

		for( i = 0; i < n_singlepsic; i++ ) {
			shift = 2*i*size_psic*sizeof(double);
			kernel_init_psic_k<<< grid2_psic, threads2_psic >>>( (int*) nls_D, (int*) igk_D, (double*) psi_D, (double*) ( (char*) psic_D + shift ), n, lda, ((dim_multiplepsic*n_multiplepsic) +i) );
			qecudaGetLastError("kernel launch failure");
		}

		qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex*) psic_D, (cufftDoubleComplex*) psic_D, CUFFT_INVERSE ) );

		for( i = 0; i < n_singlepsic; i++ ) {
			shift = 2*i*size_psic*sizeof(double);
			kernel_vec_prod_k<<< grid2_prod, threads2_prod >>>( (double *) ( (char*) psic_D + shift ), (double *) v_D , nrxxs );
			qecudaGetLastError("kernel launch failure");
		}

		qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex*) psic_D, (cufftDoubleComplex*) psic_D, CUFFT_FORWARD ) );

		tscale = 1.0 / (double) ( size_psic );
		cublasZdscal(qecudaHandles[ 0 ] , n_singlepsic * size_psic, &tscale, (cuDoubleComplex *) psic_D, 1);

		qecudaSafeCall( cudaMemcpy( psic, psic_D, sizeof( cufftDoubleComplex ) * n_singlepsic * size_psic, cudaMemcpyDeviceToHost ) );

		for( i = 0; i < n_singlepsic; i++ ) {
			for( k = 0; k <  n; k++ ) {
				hpsi[ k + ( ((dim_multiplepsic*n_multiplepsic) + i) * lda ) ][0] += psic[ nls [ igk[ k ] - 1  ] - 1 ][0];
				hpsi[ k + ( ((dim_multiplepsic*n_multiplepsic) + i) * lda ) ][1] += psic[ nls [ igk[ k ] - 1  ] - 1 ][1];
			}
		}

		qecheck_cufft_call( cufftDestroy(p_global) );
		qecudaSafeCall( cudaFreeHost(psic));
	}
}
#endif
