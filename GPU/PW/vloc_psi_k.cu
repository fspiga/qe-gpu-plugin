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

#include <stdlib.h>
#include <stdio.h>

#include "cuda_env.h"

typedef double fftw_complex[2];

__global__ void kernel_vec_prod_k( double *a, const double *b, int dimx )
{	   
	register int ix = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	register double sup;
	register int ii = ix / 2;

	if ( ix < ( dimx * 2 ) ) {
		sup = a[ix] * b[ii];
		a[ix] = sup;
	}
}

__global__ void kernel_init_psic_k( const int * nls, const int * igk, const double * psi, double *psic, const int n, const int lda, const int ibnd )
{	   
	int ix = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	// int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int psic_index_nls, psi_index = ( ix + ( ibnd * lda ) ) * 2;

	if ( ix < n ) {
		psic_index_nls = ( nls[ igk[ ix ] - 1 ] - 1 ) * 2;
		psic[ psic_index_nls ] = psi[ psi_index ];
		psic[ psic_index_nls + 1 ] = psi[ psi_index + 1 ];
	}
}

__global__ void kernel_save_hpsi_k( const int * nls, const int * igk, double * hpsi, const double *psic, const int n, const int ibnd )
{	   
	int ix = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	int pos = ibnd * n;
	int psic_index_nls, psi_index = (ix + pos) * 2;

	if ( ix < (n) ) {
		psic_index_nls = (nls[ igk[ ix ] - 1 ] - 1) * 2;
		hpsi[ psi_index ] = hpsi[ psi_index ] + psic[ psic_index_nls ];
		hpsi[ psi_index + 1 ] = hpsi[ psi_index + 1 ] + psic[ psic_index_nls + 1 ];
	}
}


extern "C" void vloc_psi_cuda_k_( int * ptr_lda, int * ptr_nrxxs, int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s, int * ptr_n, int * ptr_m, cufftDoubleComplex * psi, double * v, fftw_complex * hpsi, int * igk, int * nls, int * ptr_ngms)
{
	cufftHandle p_global;
	fftw_complex * psic = NULL;

	void * psic_D, * psi_D; // cufftDoubleComplex *
	void * v_D; // double *
	void * local_igk_D, * local_nls_D; // int*

	int j,  blocksPerGrid, ibnd;
	double tscale;

	int n = (* ptr_n);
	int m = (* ptr_m);
	int nr1s = (* ptr_nr1s);
	int nr2s = (* ptr_nr2s);
	int nr3s = (* ptr_nr3s);
	int nrxxs = (* ptr_nrxxs);
	int ngms = (* ptr_ngms);
	int lda = (* ptr_lda);

	int size_psic = nr1s * nr2s * nr3s;

	cudaStream_t  vlocStreams[ MAX_QE_GPUS ];
	cublasHandle_t vlocHandles[ MAX_QE_GPUS ];

	size_t buffer_size = 0L;

#if defined(__CUDA_DEBUG)
	printf("[CUDA DEBUG] VLOC_PSI_K\n"); fflush(stdout);
#endif

	buffer_size = size_psic * sizeof( cufftDoubleComplex ) + sizeof( cufftDoubleComplex ) * lda * m + sizeof( int ) * ngms + sizeof( int ) * n + sizeof( double ) * nrxxs;

	cudaSetDevice(qe_gpu_bonded[0]);

	if ( cublasCreate( &vlocHandles[ 0 ] ) != CUBLAS_STATUS_SUCCESS ) {
		printf("\n*** CUDA VLOC_PSI_K *** ERROR *** cublasInit() for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	if( cudaStreamCreate( &vlocStreams[ 0 ] ) != cudaSuccess ) {
		printf("\n*** CUDA VLOC_PSI_K *** ERROR *** creating stream for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	if ( buffer_size > cuda_memory_unused[0] ) {
		fprintf( stderr, "\n[VLOC_PSI_K] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", buffer_size, cuda_memory_allocated[0] );
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	size_t shift = 0;
	psic_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( size_psic )*sizeof( cufftDoubleComplex );
	psi_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( lda * m )*sizeof( cufftDoubleComplex );
	v_D = (char*) dev_scratch_QE[0] + shift;
#if defined(__CUDA_PRELOADING_DATA)
	local_nls_D = (void *) preloaded_nls_D;
	local_igk_D = (void *) preloaded_igk_D;
#else
	shift += ( nrxxs )*sizeof( double );
	local_nls_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( (ngms % 2 == 0)? ngms : ngms + 1 )*sizeof(int);
	local_igk_D = (char*) dev_scratch_QE[0] + shift;
#endif

	// Before do anything force sync to terminate async data transfer
#if defined(__CUDA_PRELOADING_DATA) && defined(__CUDA_PRELOAD_PINNED)
	cudaDeviceSynchronize();
#endif

	qecudaSafeCall( cudaMemcpy( psi_D, psi,  sizeof( cufftDoubleComplex ) * lda * m, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( v_D, v,  sizeof( double ) * nrxxs, cudaMemcpyHostToDevice ) );
#if !defined(__CUDA_PRELOADING_DATA)
	qecudaSafeCall( cudaMemcpy( local_nls_D, nls,  sizeof( int ) * ngms, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( local_igk_D, igk,  sizeof( int ) * n, cudaMemcpyHostToDevice ) );
#endif

	qecheck_cufft_call( cufftPlan3d( &p_global, nr3s, nr2s,  nr1s, CUFFT_Z2Z ) );

	if( cufftSetStream(p_global,vlocStreams[ 0 ]) != CUFFT_SUCCESS ) {
		printf("\n*** CUDA VLOC_PSI_K *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
		exit( EXIT_FAILURE );
	}

	qecudaSafeCall( cudaHostAlloc ( (void**) &psic, size_psic * sizeof( fftw_complex ), cudaHostAllocPortable ) );

	for ( ibnd =  0; ibnd < m; ibnd = ibnd + 1) {

		cudaDeviceSynchronize();

		qecudaSafeCall( cudaMemset( psic_D, 0, size_psic * sizeof( cufftDoubleComplex ) ) );

		blocksPerGrid = ( ( n * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
		kernel_init_psic_k<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (int *) local_nls_D, (int *) local_igk_D, (double *) psi_D, (double *) psic_D, n, lda, ibnd );
		qecudaGetLastError("kernel launch failure");

		qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex *) psic_D, (cufftDoubleComplex *) psic_D, CUFFT_INVERSE ) );

		blocksPerGrid = ( (nrxxs * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
		kernel_vec_prod_k<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (double *) psic_D, (double *) v_D , nrxxs );
		qecudaGetLastError("kernel launch failure");

//		for ( j = 0; j <  n && ibnd > 0; j++ ) {
//			hpsi[ j + ( ( ibnd  - 1 ) * lda ) ][0] += psic[ nls [ igk[ j ] - 1  ] - 1 ][0];
//			hpsi[ j + ( ( ibnd  - 1 ) * lda ) ][1] += psic[ nls [ igk[ j ] - 1  ] - 1 ][1];
//		}

		qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex *) psic_D, (cufftDoubleComplex *)psic_D, CUFFT_FORWARD ) );

		tscale = 1.0 / (double) ( size_psic );

		cublasZdscal(vlocHandles[ 0 ] , size_psic, &tscale, (cuDoubleComplex *) psic_D, 1);

		qecudaSafeCall( cudaMemcpy( psic, psic_D, sizeof( cufftDoubleComplex ) * size_psic, cudaMemcpyDeviceToHost ) );

	    for( j = 0; j <  n; j++ ) {
	      hpsi[ j + ( ibnd * lda ) ][0] += psic[ nls [ igk[ j ] - 1  ] - 1 ][0];
	      hpsi[ j + ( ibnd * lda ) ][1] += psic[ nls [ igk[ j ] - 1  ] - 1 ][1];
	    }

	}

//	for ( j = 0; j <  n; j++ ) {
//		hpsi[ j + ( ( m - 1 ) * lda ) ][0] += psic[ nls [ igk[ j ] - 1  ] - 1 ][0];
//		hpsi[ j + ( ( m - 1 ) * lda ) ][1] += psic[ nls [ igk[ j ] - 1  ] - 1 ][1];
//	}

	qecheck_cufft_call( cufftDestroy(p_global) );

#if defined(__CUDA_KERNEL_MEMSET)
	qecudaSafeCall( cudaMemset( dev_scratch_QE[0], 0, (size_t) cuda_memory_unused[0] ) );
#endif

	cudaStreamDestroy( vlocStreams[ 0 ] );
	cublasDestroy( vlocHandles[ 0 ]);
}


extern "C" void vloc_psi_multiplan_cuda_k_(  int * ptr_lda, int * ptr_nrxxs, int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s, int * ptr_n, int * ptr_m, cufftDoubleComplex * psi, double * v, fftw_complex * hpsi, int * igk, int * nls, int * ptr_ngms)
{
	cufftDoubleComplex * psic_D, * psi_D;
	fftw_complex * psic = NULL;

	double * v_D;

	int * igk_D, * nls_D;
	int blocksPerGrid;
	double tscale;

	int n = (* ptr_n);
	int m = (* ptr_m);
	int nr1s = (* ptr_nr1s);
	int nr2s = (* ptr_nr2s);
	int nr3s = (* ptr_nr3s);
	int nrxxs = (* ptr_nrxxs);
	int ngms = (* ptr_ngms);
	int lda = (* ptr_lda);

	int dim_multiplepsic, n_singlepsic, n_multiplepsic, size_multiplepsic, i, j, k;
	int array[3];

	int size_psic = nr1s * nr2s * nr3s;

	int last;

	cudaStream_t  vlocStreams[ MAX_QE_GPUS ];
	cublasHandle_t vlocHandles[ MAX_QE_GPUS ];

	size_t buffer_size = 0L;

	cudaSetDevice(qe_gpu_bonded[0]);

	if ( cublasCreate( &vlocHandles[ 0 ] ) != CUBLAS_STATUS_SUCCESS ) {
		printf("\n*** CUDA VLOC_PSI_K *** ERROR *** cublasInit() for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	if( cudaStreamCreate( &vlocStreams[ 0 ] ) != cudaSuccess ) {
		printf("\n*** CUDA VLOC_PSI_K *** ERROR *** creating stream for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	dim_multiplepsic = __NUM_FFT_MULTIPLAN__;

	n_multiplepsic = m/dim_multiplepsic;
	n_singlepsic = m%dim_multiplepsic;

	size_multiplepsic = size_psic * dim_multiplepsic;

	buffer_size = size_multiplepsic * sizeof( cufftDoubleComplex ) + sizeof( cufftDoubleComplex ) * n * m + sizeof( int ) * ngms + sizeof( int ) * n + sizeof( double ) * nrxxs;

	if ( buffer_size > cuda_memory_allocated[0] ) {
		fprintf( stderr, "\n[VLOC_PSI_K] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", buffer_size, cuda_memory_allocated[0] );
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	v_D = (double *) dev_scratch_QE[0];
	psic_D = (cufftDoubleComplex *) (v_D + nrxxs);
	psi_D = psic_D + size_multiplepsic;
	nls_D = (int *) (psi_D + ( n * m ) );
	igk_D = nls_D + ngms;

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

		if( cufftSetStream(p_global,vlocStreams[ 0 ]) != CUFFT_SUCCESS ) {
			printf("\n*** CUDA VLOC_PSI_K *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		qecudaSafeCall( cudaHostAlloc ( (void**) &psic, size_multiplepsic * sizeof( fftw_complex ), cudaHostAllocPortable ) );

		for(j = 0; j< (m-n_singlepsic); j = j + dim_multiplepsic ) {

			qecudaSafeCall( cudaMemset( psic_D, 0, size_psic * dim_multiplepsic * sizeof( cufftDoubleComplex ) ) );

			blocksPerGrid = ( ( n * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
			for( i = 0; i < dim_multiplepsic; i++ ) {
				kernel_init_psic_k<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( nls_D, igk_D, (double *) psi_D, (double *) psic_D, n, lda, (j+i));
				qecudaGetLastError("kernel launch failure");
			}

			qecheck_cufft_call( cufftExecZ2Z( p_global, psic_D, psic_D, CUFFT_INVERSE ) );

			blocksPerGrid = ( (nrxxs * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
			for( i = 0; i < dim_multiplepsic; i++ ) {
				kernel_vec_prod_k<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (double *) psic_D, (double *) v_D , nrxxs );
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

			qecheck_cufft_call( cufftExecZ2Z( p_global, psic_D, psic_D, CUFFT_FORWARD ) );

			tscale = 1.0 / (double) ( size_psic );
			cublasZdscal(vlocHandles[ 0 ] , size_psic*dim_multiplepsic, &tscale, (cuDoubleComplex *) psic_D, 1);

			qecudaSafeCall( cudaMemcpy( psic, psic_D, sizeof( cufftDoubleComplex ) * size_psic * dim_multiplepsic, cudaMemcpyDeviceToHost ) );
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

		if( cufftSetStream(p_global,vlocStreams[ 0 ]) != CUFFT_SUCCESS ) {
			printf("\n*** CUDA VLOC_PSI_K *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		qecudaSafeCall( cudaHostAlloc ( (void**) &psic, n_singlepsic * size_psic * sizeof( cufftDoubleComplex ), cudaHostAllocPortable ) );

		qecudaSafeCall( cudaMemset( psic_D, 0, n_singlepsic * size_psic * sizeof( cufftDoubleComplex ) ) );

		blocksPerGrid = ( ( n * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
		for( i = 0; i < n_singlepsic; i++ ) {
			kernel_init_psic_k<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( nls_D, igk_D, (double *) psi_D, (double *) ( psic_D + i*size_psic), n, lda, ((dim_multiplepsic*n_multiplepsic) +i) );
			qecudaGetLastError("kernel launch failure");
		}

		qecheck_cufft_call( cufftExecZ2Z( p_global, psic_D, psic_D, CUFFT_INVERSE ) );

		blocksPerGrid = ( (nrxxs * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
		for( i = 0; i < n_singlepsic; i++ ) {
			kernel_vec_prod_k<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (double *) ( psic_D + i*size_psic), (double *) v_D , nrxxs );
			qecudaGetLastError("kernel launch failure");
		}

		qecheck_cufft_call( cufftExecZ2Z( p_global, psic_D, psic_D, CUFFT_FORWARD ) );

		tscale = 1.0 / (double) ( size_psic );
		cublasZdscal(vlocHandles[ 0 ] , n_singlepsic * size_psic, &tscale, (cuDoubleComplex *) psic_D, 1);

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

	cudaStreamDestroy( vlocStreams[ 0 ] );
	cublasDestroy( vlocHandles[ 0 ]);
}


extern "C" void vloc_psi_k_fft_cuda_(int * ptr_m, int * ptr_v_size, int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s, void * psic, double * v)
{
	cufftHandle p_global;

	cufftDoubleComplex * psic_D;
	cufftDoubleComplex * local_p_psic;
	double * v_D, tscale;
	int blocksPerGrid;

	int m = (* ptr_m);
	int nr1s = (* ptr_nr1s);
	int nr2s = (* ptr_nr2s);
	int nr3s = (* ptr_nr3s);
	int v_size = (* ptr_v_size);
	local_p_psic = (cufftDoubleComplex *) psic;
	int size_psic =  v_size; // nr1s * nr2s * nr3s;

	int dim_multiplepsic, n_singlepsic, n_multiplepsic, i, j;

	cudaStream_t  vlocStreams[ MAX_QE_GPUS ];
	cublasHandle_t vlocHandles[ MAX_QE_GPUS ];

	size_t buffer_size = 0L;

	cudaSetDevice(qe_gpu_bonded[0]);

	if ( cublasCreate( &vlocHandles[ 0 ] ) != CUBLAS_STATUS_SUCCESS ) {
		printf("\n*** CUDA VLOC_PSI_K *** ERROR *** cublasInit() for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	if( cudaStreamCreate( &vlocStreams[ 0 ] ) != cudaSuccess ) {
		printf("\n*** CUDA VLOC_PSI_K *** ERROR *** creating stream for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	qecudaSafeCall( cudaMemset( dev_scratch_QE[0], 0, (size_t) cuda_memory_allocated[0] ) );

	dim_multiplepsic = __NUM_FFT_MULTIPLAN__;

	n_multiplepsic = m/dim_multiplepsic;
	n_singlepsic = m%dim_multiplepsic;

	buffer_size = size_psic * dim_multiplepsic * sizeof( cufftDoubleComplex ) + sizeof( double ) * v_size ;

	if ( buffer_size > cuda_memory_allocated[0] ) {
		fprintf( stderr, "\n[VLOC_PSI_K] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", buffer_size, cuda_memory_allocated[0] );
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	psic_D = (cufftDoubleComplex * ) dev_scratch_QE[0];
	v_D = (double * ) (psic_D + ( size_psic * dim_multiplepsic ));

	qecudaSafeCall( cudaMemcpy( v_D, v,  sizeof( double ) * size_psic, cudaMemcpyHostToDevice ) );

	if ( n_multiplepsic > 0 ) {

		qecheck_cufft_call( cufftPlan3d( &p_global, nr3s, nr2s,  nr1s, CUFFT_Z2Z ) );

		if( cufftSetStream(p_global,vlocStreams[ 0 ]) != CUFFT_SUCCESS ) {
			printf("\n*** CUDA VLOC_PSI_K *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		for(j = 0; j< (m-n_singlepsic); j+=dim_multiplepsic ) {

			qecudaSafeCall( cudaMemset( psic_D , 0, dim_multiplepsic * size_psic * sizeof( cufftDoubleComplex ) ) );

			qecudaSafeCall( cudaMemcpy( psic_D, (cufftDoubleComplex * ) (local_p_psic + j*size_psic) ,  sizeof( cufftDoubleComplex ) * ( size_psic * dim_multiplepsic )  , cudaMemcpyHostToDevice ) );

			for( i = 0; i < dim_multiplepsic; i++ ) {
				qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex *) (psic_D + ( i * size_psic )), (cufftDoubleComplex *) (psic_D + ( i * size_psic )), CUFFT_INVERSE ) );
			}

			blocksPerGrid = ( (v_size * 2) + __CUDA_THREADPERBLOCK__  - 1) / __CUDA_THREADPERBLOCK__ ;
			for( i = 0; i < dim_multiplepsic; i++ ) {
				kernel_vec_prod_k<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (double *) (psic_D + ( i * size_psic )), (double *) v_D , v_size );
				qecudaGetLastError("kernel launch failure");
			}

			for( i = 0; i < dim_multiplepsic; i++ ) {
				qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex *) (psic_D + ( i * size_psic )), (cufftDoubleComplex *) (psic_D + ( i * size_psic )), CUFFT_FORWARD ) );
			}

			tscale = 1.0 / (double) ( size_psic );
			cublasZdscal(vlocHandles[ 0 ] , size_psic*dim_multiplepsic, &tscale, (cuDoubleComplex *) psic_D, 1);

			qecudaSafeCall( cudaMemcpy( (cufftDoubleComplex * ) (local_p_psic + j*size_psic) , psic_D, sizeof( cufftDoubleComplex ) * ( size_psic * dim_multiplepsic ), cudaMemcpyDeviceToHost ) );
		}

		qecheck_cufft_call( cufftDestroy(p_global) );
	}

	if (n_singlepsic > 0 ) {

		qecheck_cufft_call( cufftPlan3d( &p_global, nr3s, nr2s,  nr1s, CUFFT_Z2Z ) );

		if( cufftSetStream(p_global,vlocStreams[ 0 ]) != CUFFT_SUCCESS ) {
			printf("\n*** CUDA VLOC_PSI_K *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		qecudaSafeCall( cudaMemset( psic_D , 0, n_singlepsic * size_psic * sizeof( cufftDoubleComplex ) ) );

		qecudaSafeCall( cudaMemcpy( psic_D, (cufftDoubleComplex * ) (local_p_psic + n_multiplepsic*dim_multiplepsic*size_psic) ,  sizeof( cufftDoubleComplex ) * ( size_psic * n_singlepsic)  , cudaMemcpyHostToDevice ) );

		for( i = 0; i < n_singlepsic; i++ ) {
			qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex *) (psic_D + ( i * size_psic )), (cufftDoubleComplex *) (psic_D + ( i * size_psic )), CUFFT_INVERSE ) );
		}

		blocksPerGrid = ( (v_size * 2) + __CUDA_THREADPERBLOCK__  - 1) / __CUDA_THREADPERBLOCK__ ;
		for( i = 0; i < n_singlepsic; i++ ){
			kernel_vec_prod_k<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (double *) (psic_D + (i * size_psic) ), (double *) v_D , v_size );
			qecudaGetLastError("kernel launch failure");
		}

		for( i = 0; i < n_singlepsic; i++ ) {
			qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex *) (psic_D + ( i * size_psic )), (cufftDoubleComplex *) (psic_D + ( i * size_psic )), CUFFT_FORWARD ) );
		}

		tscale = 1.0 / (double) ( size_psic );
		cublasZdscal(vlocHandles[ 0 ] , size_psic * n_singlepsic, &tscale, (cuDoubleComplex *) psic_D, 1);

		qecudaSafeCall( cudaMemcpy( (cufftDoubleComplex * ) (local_p_psic + n_multiplepsic*dim_multiplepsic*size_psic) , psic_D, sizeof( cufftDoubleComplex ) * ( size_psic * n_singlepsic), cudaMemcpyDeviceToHost ) );

		qecheck_cufft_call( cufftDestroy(p_global) );
	}

	qecudaSafeCall( cudaMemset( dev_scratch_QE[0], 0, (size_t) cuda_memory_allocated[0] ) );

	cudaStreamDestroy( vlocStreams[ 0 ] );
	cublasDestroy( vlocHandles[ 0 ]);
}
