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
#include <stdlib.h>
#include "cuda_env.h"

extern "C" void start_clock_(char * label, unsigned int length_arg );
extern "C" void stop_clock_(char * label, unsigned int length_arg );

__global__ void kernel_vec_prod( double *a, const double *b, int dimx )
{	   
	register int ix = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	register double sup;
	register int ii = ix / 2;

	if( ix < ( dimx * 2 ) ){
		sup = a[ix] * b[ii];
		a[ix] = sup;
	}
}

__global__ void kernel_init_psic( const int * nls, const int * nlsm, const int * igk, const double * psi, double * psic, const int n, const int m, const int lda, const int ibnd )
{	   
	register int ix = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	register int pos = ibnd * lda;
	register int pos_plus = (ibnd+1) * lda;

	register int psic_index_nls;
	register int psic_index_nlsm;

	register int psi_index = (ix + pos) * 2;
	register int psi_index_plus = (ix + pos_plus) * 2;

	if ( ix < n ) {

		psic_index_nls = ( nls[ igk[ ix ] - 1 ] - 1 ) * 2;
		psic_index_nlsm = ( nlsm[ igk[ ix ] - 1 ] - 1 ) * 2;

		if ( ibnd  < ( m - 1 ) ) {

			psic[ psic_index_nls ] = psi[ psi_index ] - psi[ psi_index_plus + 1 ];
			psic[ psic_index_nls + 1 ] = psi[ psi_index + 1 ] + psi[ psi_index_plus ];

			psic[ psic_index_nlsm ] = psi[ psi_index ] + psi[ psi_index_plus + 1 ];
			psic[ psic_index_nlsm + 1 ] = -1.0 * ( psi[ psi_index + 1 ] - psi[ psi_index_plus ] );

		} else {

			psic[ psic_index_nls ] = psi[ psi_index ];
			psic[ psic_index_nls + 1 ] = psi[ psi_index + 1 ];

			psic[ psic_index_nlsm ] = psi[ psi_index ];
			psic[ psic_index_nlsm + 1 ] = - 1.0 * psi[ psi_index + 1 ];

		}
	}
}

__global__ void kernel_save_hpsi( const int * nls, const int * nlsm, const int * igk, double * hpsi, const double *psic, const int n, const int m, const int lda, const int ibnd )
{	   
	register int ix = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	register int pos = ibnd * lda;
	register int pos_plus = (ibnd+1) * lda;

	register int psic_index_nls;
	register int psic_index_nlsm;

	register int psi_index = (ix + pos) * 2;
	register int psi_index_plus = (ix + pos_plus) * 2;

	register double real_fp, cmplx_fp, real_fm, cmplx_fm;

	if ( ix < n ) {

		psic_index_nls = (nls[ igk[ ix ] - 1 ] - 1) * 2;
		psic_index_nlsm = (nlsm[ igk[ ix ] - 1 ] - 1) * 2;

		if( ibnd  < ( m - 1 ) ) {

			real_fp = ( psic[ psic_index_nls ] + psic[ psic_index_nlsm ] ) * 0.5;
			cmplx_fp = ( psic[ psic_index_nls + 1 ] + psic[ psic_index_nlsm + 1 ] ) * 0.5;

			real_fm = ( psic[ psic_index_nls ] - psic[ psic_index_nlsm ] ) * 0.5;
			cmplx_fm = ( psic[ psic_index_nls + 1 ] - psic[ psic_index_nlsm + 1 ] ) * 0.5;

			hpsi[ psi_index ] = hpsi[ psi_index ] + real_fp;
			hpsi[ psi_index + 1 ] = hpsi[ psi_index + 1 ] + cmplx_fm;

			hpsi[ psi_index_plus ] = hpsi[ psi_index_plus ] + cmplx_fp;
			hpsi[ psi_index_plus + 1 ] = hpsi[ psi_index_plus + 1 ] - real_fm;

		} else {

			hpsi[ psi_index ] = hpsi[ psi_index ] + psic[ psic_index_nls ];
			hpsi[ psi_index + 1 ] = hpsi[ psi_index + 1 ] + psic[ psic_index_nls + 1 ];

		}
	}
}


extern "C"  void vloc_psi_cuda_(int * ptr_lda, int * ptr_nrxxs, int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s, int * ptr_n, int * ptr_m, void * psi, double * v, void * hpsi, int * igk, int * nls, int * nlsm, int * ptr_ngms, int * ptr_ngm)
{
	cufftHandle p_global;

    void * psic_D, * psi_D, * hpsi_D; // cufftDoubleComplex*
	void * v_D; // double*
	void * local_igk_D, * local_nls_D, * local_nlsm_D; // int*
	int blocksPerGrid, ibnd;

	double tscale;

	int n = (* ptr_n);
	int m = (* ptr_m);
	int m_fake;
	int nr1s = (* ptr_nr1s);
	int nr2s = (* ptr_nr2s);
	int nr3s = (* ptr_nr3s);
	int nrxxs = (* ptr_nrxxs);
	int ngms = (* ptr_ngms);
	int ngm = (* ptr_ngm);
	int lda = (* ptr_lda);

	int size_psic = nr1s * nr2s * nr3s;

	cudaStream_t  vlocStreams[ MAX_QE_GPUS ];
	cublasHandle_t vlocHandles[ MAX_QE_GPUS ];

	size_t buffer_size = 0L;

#if defined(__CUDA_DEBUG)
	printf("\n[CUDA_DEBUG - VLOC_PSI_GAMMA] m = %d, n = %d, nrxxs = %d, size_psic = %d\n", m, n, nrxxs, size_psic);
	fflush(stdout);
#endif

	/* Padding -- really necessary?*/
	if (m%2 == 0)
		m_fake = m ;
	else
		m_fake = m + 1;

	cudaSetDevice(qe_gpu_bonded[0]);

	if ( cublasCreate( &vlocHandles[ 0 ] ) != CUBLAS_STATUS_SUCCESS ) {
		printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cublasInit() for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	if( cudaStreamCreate( &vlocStreams[ 0 ] ) != cudaSuccess ) {
		printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** creating stream for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	// qecudaSafeCall( cudaMemset( dev_scratch_QE[0], 0, (size_t) cuda_memory_allocated[0] ) );

	buffer_size = size_psic * sizeof( cufftDoubleComplex ) + sizeof( cufftDoubleComplex ) * lda * m_fake * 2 + sizeof( int ) * ngms + sizeof( int ) * ngm + sizeof( int ) * lda + sizeof( double ) * nrxxs;

	if ( buffer_size > cuda_memory_unused[0] ) {
		fprintf( stderr, "\n[VLOC_PSI_GAMMA] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", buffer_size, cuda_memory_allocated[0] );
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
	shift += ( lda * m_fake )*sizeof( cufftDoubleComplex );
	hpsi_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( lda * m_fake )*sizeof( cufftDoubleComplex );
	v_D = (char*) dev_scratch_QE[0] + shift;
#if defined(__CUDA_PRELOADING_DATA)
	local_nls_D = (void *) preloaded_nls_D;
	local_nlsm_D = (void *) preloaded_nlsm_D;
	local_igk_D = (void *) preloaded_igk_D;
#else
	shift += ( nrxxs )*sizeof( double );
	local_nls_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( (ngms % 2 == 0)? ngms : ngms + 1 )*sizeof(int);
	local_nlsm_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( (ngm % 2 == 0)? ngm : ngm + 1 )*sizeof(int);
	local_igk_D = (char*) dev_scratch_QE[0] + shift;
#endif

	// Before do anything force sync to terminate async data transfer
#if defined(__CUDA_PRELOADING_DATA) && defined(__CUDA_PRELOAD_PINNED)
	cudaDeviceSynchronize();
#endif

	qecudaSafeCall( cudaMemset( psi_D, 0, sizeof( cufftDoubleComplex ) * lda * m_fake ) );
	qecudaSafeCall( cudaMemcpy( psi_D, psi,  sizeof( cufftDoubleComplex ) * lda * m, cudaMemcpyHostToDevice ) );
	// ???
//	if (m_fake > m) {
//		qecudaSafeCall( cudaMemset( (psi_D + ( lda * m )) , 0, sizeof( cufftDoubleComplex ) * size_psic ) ); // Post-set of (m_fake) zeros
//	}

#if !defined(__CUDA_PRELOADING_DATA)
	qecudaSafeCall( cudaMemcpy( local_nls_D, nls,  sizeof( int ) * ngms, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( local_nlsm_D, nlsm,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( local_igk_D, igk,  sizeof( int ) * n, cudaMemcpyHostToDevice ) );
#endif
	qecudaSafeCall( cudaMemcpy( hpsi_D, hpsi,  sizeof( cufftDoubleComplex ) * lda * m, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( v_D, v,  sizeof( double ) * nrxxs, cudaMemcpyHostToDevice ) );

	qecheck_cufft_call( cufftPlan3d( &p_global, nr3s, nr2s,  nr1s, CUFFT_Z2Z ) );

	if( cufftSetStream(p_global,vlocStreams[ 0 ]) != CUFFT_SUCCESS ) {
		printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
		exit( EXIT_FAILURE );
	}

	for( ibnd =  0; ibnd < m_fake; ibnd = ibnd + 2 )
	{
		qecudaSafeCall( cudaMemset( psic_D , 0, size_psic * sizeof( cufftDoubleComplex ) ) );

		blocksPerGrid = ( ( n * 2) + __CUDA_TxB_VLOCPSI_PSIC__ - 1) / __CUDA_TxB_VLOCPSI_PSIC__ ;
		kernel_init_psic<<<blocksPerGrid, __CUDA_TxB_VLOCPSI_PSIC__ >>>( (int *) local_nls_D, (int *) local_nlsm_D, (int *) local_igk_D, (double *) psi_D, (double *) psic_D, n, m, lda, ibnd );
		qecudaGetLastError("kernel launch failure");

		qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex *) psic_D, (cufftDoubleComplex *) psic_D, CUFFT_INVERSE ) );

		blocksPerGrid = ( (nrxxs * 2) + __CUDA_TxB_VLOCPSI_PROD__  - 1) / __CUDA_TxB_VLOCPSI_PROD__ ;
		kernel_vec_prod<<<blocksPerGrid, __CUDA_TxB_VLOCPSI_PROD__ >>>( (double *) psic_D, (double *) v_D , nrxxs );
		qecudaGetLastError("kernel launch failure");

		qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex *) psic_D, (cufftDoubleComplex *) psic_D, CUFFT_FORWARD ) );

		tscale = 1.0 / (double) ( size_psic );
		cublasZdscal(vlocHandles[ 0 ] , size_psic, &tscale, (cufftDoubleComplex *) psic_D, 1);

		blocksPerGrid = ( ( n * 2) + __CUDA_TxB_VLOCPSI_HPSI__ - 1) / __CUDA_TxB_VLOCPSI_HPSI__ ;
		kernel_save_hpsi<<<blocksPerGrid, __CUDA_TxB_VLOCPSI_HPSI__ >>>( (int *) local_nls_D, (int *) local_nlsm_D, (int *) local_igk_D, (double *) hpsi_D, (double *) psic_D, n, m, lda, ibnd );
		qecudaGetLastError("kernel launch failure");

	}

	qecudaSafeCall( cudaMemcpy( hpsi, (cufftDoubleComplex *) hpsi_D, sizeof( cufftDoubleComplex ) * lda * m, cudaMemcpyDeviceToHost ) );

	qecheck_cufft_call( cufftDestroy(p_global) );

#if defined(__CUDA_KERNEL_MEMSET)
	qecudaSafeCall( cudaMemset( dev_scratch_QE[0], 0, (size_t) cuda_memory_unused[0] ) );
#endif

	cudaStreamDestroy( vlocStreams[ 0 ] );
	cublasDestroy( vlocHandles[ 0 ]);
}


extern "C" void vloc_psi_multiplan_cuda_(int * ptr_lda, int * ptr_nrxxs, int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s, int * ptr_n, int * ptr_m, void * psi, double * v, void * hpsi, int * igk, int * nls, int * nlsm, int * ptr_ngms, int * ptr_ngm)
{
	cufftHandle p_global;

	cufftDoubleComplex * psic_D, * psi_D, * hpsi_D;
	double * v_D;
	int * igk_D, * nls_D, * nlsm_D;
	int blocksPerGrid;

	double tscale;

	int n = (* ptr_n);
	int m = (* ptr_m);
	int m_fake, m_buf;
	int nr1s = (* ptr_nr1s);
	int nr2s = (* ptr_nr2s);
	int nr3s = (* ptr_nr3s);
	int nrxxs = (* ptr_nrxxs);
	int ngms = (* ptr_ngms);
	int ngm = (* ptr_ngm);
	int lda = (* ptr_lda);

	int size_psic = nr1s * nr2s * nr3s;
	int dim_multiplepsic, n_singlepsic, n_multiplepsic, size_multiplepsic, i, j, v_size;
	int array[3];

	cudaStream_t  vlocStreams[ MAX_QE_GPUS ];
	cublasHandle_t vlocHandles[ MAX_QE_GPUS ];

	size_t buffer_size = 0L;

	psic_D = (cufftDoubleComplex * ) dev_scratch_QE[0];

	/* Padding */
	if (m%2 == 0) {
		m_buf = m;
		m_fake = m_buf/2 ;
	} else {
		m_buf = m+1;
		m_fake = m_buf/2;
	}

	dim_multiplepsic = __NUM_FFT_MULTIPLAN__;
	n_multiplepsic = m_fake/dim_multiplepsic;
	n_singlepsic = m_fake%dim_multiplepsic;

	size_multiplepsic = size_psic * (dim_multiplepsic);

	buffer_size = size_multiplepsic * sizeof( cufftDoubleComplex ) + sizeof( cufftDoubleComplex ) * n * m * 2 + sizeof( int ) * ngms + sizeof( int ) * ngm + sizeof( int ) * n + sizeof( double ) * nrxxs;

	cudaSetDevice(qe_gpu_bonded[0]);

	if ( cublasCreate( &vlocHandles[ 0 ] ) != CUBLAS_STATUS_SUCCESS ) {
		printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cublasInit() for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	if( cudaStreamCreate( &vlocStreams[ 0 ] ) != cudaSuccess ) {
		printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** creating stream for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	if ( buffer_size > cuda_memory_allocated[0] ) {
		fprintf( stderr, "\n[VLOC_PSI_GAMMA] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", buffer_size, cuda_memory_allocated[0] );
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	psi_D = psic_D + size_multiplepsic;
	hpsi_D = psi_D + ( n * m );
	v_D = (double * )  (hpsi_D + ( n * m ) );
	nls_D = (int   * ) (v_D + nrxxs);
	nlsm_D = (int *) nls_D + ngms;
	igk_D = nlsm_D + ngm;

	qecudaSafeCall( cudaMemset( dev_scratch_QE[0], 0, (size_t) cuda_memory_allocated[0] ) );

	qecudaSafeCall( cudaMemcpy( psi_D, psi,  sizeof( cufftDoubleComplex ) * n * m_buf, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemset( (psi_D + ( n * m )) , 0, sizeof( cufftDoubleComplex ) * m_buf ) ); // Post-set of (m_fake) zeros

	qecudaSafeCall( cudaMemcpy( hpsi_D, hpsi,  sizeof( cufftDoubleComplex ) * n * m, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( nls_D, nls,  sizeof( int ) * ngms, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( nlsm_D, nlsm,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( igk_D, igk,  sizeof( int ) * n, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( v_D, v,  sizeof( double ) * nrxxs, cudaMemcpyHostToDevice ) );

	array[0] = nr3s;
	array[1] = nr2s;
	array[2] = nr1s;

	v_size = nr1s * nr2s * nr3s;

	if ( n_multiplepsic > 0 ) {

		qecheck_cufft_call( cufftPlanMany( &p_global, 3, array, NULL, 1, 0, NULL,1,0,CUFFT_Z2Z,dim_multiplepsic ) );

		if( cufftSetStream(p_global,vlocStreams[ 0 ]) != CUFFT_SUCCESS ) {
			printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		for(j = 0; j< (m_fake-n_singlepsic); j+=dim_multiplepsic ) {

			qecudaSafeCall( cudaMemset( psic_D , 0, dim_multiplepsic * size_psic * sizeof( cufftDoubleComplex ) ) );

			blocksPerGrid = ( ( n * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
			for (i = 0; i < dim_multiplepsic; i++  )
			{
				kernel_init_psic<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( nls_D, nlsm_D, igk_D, (double *) psi_D, (double *) ( psic_D + i*size_psic), n, m, lda, ((j+i)*2) );
				qecudaGetLastError("kernel launch failure");
			}

			qecheck_cufft_call( cufftExecZ2Z( p_global,  psic_D, psic_D , CUFFT_INVERSE ) );

			blocksPerGrid = ( (v_size * 2) + __CUDA_THREADPERBLOCK__  - 1) / __CUDA_THREADPERBLOCK__ ;
			for( i = 0; i < dim_multiplepsic; i++ ) {
				kernel_vec_prod<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (double *) (psic_D + ( i * size_psic )), (double *) v_D , v_size );
				qecudaGetLastError("kernel launch failure");
			}

			qecheck_cufft_call( cufftExecZ2Z( p_global, psic_D, psic_D, CUFFT_FORWARD ) );

			tscale = 1.0 / (double) ( size_psic );
			cublasZdscal(vlocHandles[ 0 ] , size_psic*dim_multiplepsic, &tscale, (cuDoubleComplex *) psic_D, 1);

			blocksPerGrid = ( ( n * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
			for (i = 0; i < dim_multiplepsic; i++  )
			{
				kernel_save_hpsi<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( nls_D, nlsm_D, igk_D, (double *) hpsi_D, (double *) ( psic_D + i*size_psic), n, m, lda, ((j+i)*2) );
				qecudaGetLastError("kernel launch failure");
			}
		}

		qecheck_cufft_call( cufftDestroy(p_global) );
	}

	if (n_singlepsic > 0 ) {

		qecudaSafeCall( cudaMemset( psic_D , 0, n_singlepsic * size_psic * sizeof( cufftDoubleComplex ) ) );

		qecheck_cufft_call( cufftPlanMany( &p_global, 3, array, NULL, 1, 0, NULL,1,0,CUFFT_Z2Z,n_singlepsic ) );

		if( cufftSetStream(p_global,vlocStreams[ 0 ]) != CUFFT_SUCCESS ) {
			printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		blocksPerGrid = ( ( n * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
		for (i = 0; i < n_singlepsic; i++  )
		{
			kernel_init_psic<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( nls_D, nlsm_D, igk_D, (double *) psi_D, (double *) ( psic_D + i*size_psic), n, m, lda, (dim_multiplepsic*n_multiplepsic + i)*2 );
			qecudaGetLastError("kernel launch failure");
		}

		qecheck_cufft_call( cufftExecZ2Z( p_global,  psic_D, psic_D , CUFFT_INVERSE ) );

		blocksPerGrid = ( (v_size * 2) + __CUDA_THREADPERBLOCK__  - 1) / __CUDA_THREADPERBLOCK__ ;
		for( i = 0; i < n_singlepsic; i++ ){
			kernel_vec_prod<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (double *) (psic_D + (i * size_psic )), (double *) v_D , v_size );
			qecudaGetLastError("kernel launch failure");
		}

		tscale = 1.0 / (double) ( size_psic );
		cublasZdscal(vlocHandles[ 0 ] , size_psic * n_singlepsic, &tscale, (cuDoubleComplex *) psic_D, 1);

		qecheck_cufft_call( cufftExecZ2Z( p_global, psic_D, psic_D, CUFFT_FORWARD ) );

		blocksPerGrid = ( ( n * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
		for (i = 0; i < n_singlepsic; i++  )
		{
			kernel_save_hpsi<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( nls_D, nlsm_D, igk_D, (double *) hpsi_D, (double *) ( psic_D + i*size_psic), n, m, lda, (dim_multiplepsic*n_multiplepsic + i)*2 );
			qecudaGetLastError("kernel launch failure");
		}

		qecheck_cufft_call( cufftDestroy(p_global) );
	}

	qecudaSafeCall( cudaMemcpy( hpsi, hpsi_D, sizeof( cufftDoubleComplex ) * n * m, cudaMemcpyDeviceToHost ) );
	qecudaSafeCall( cudaMemset( dev_scratch_QE[0], 0, (size_t) cuda_memory_allocated[0] ) );

	cudaStreamDestroy( vlocStreams[ 0 ] );
	cublasDestroy( vlocHandles[ 0 ]);
}


extern "C" void vloc_psi_multiplansplit_cuda_(int * ptr_lda, int * ptr_nrxxs, int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s, int * ptr_n, int * ptr_m, void * psi, double * v, void * hpsi, int * igk, int * nls, int * nlsm, int * ptr_ngms, int * ptr_ngm)
{
	cufftHandle p_global;

	cufftDoubleComplex * psic_D, * psi_D, * hpsi_D;
	cufftDoubleComplex * t_psi_D, * t_hpsi;
	double * v_D;
	int * igk_D, * nls_D, * nlsm_D;
	int blocksPerGrid;

	double tscale;

	int n = (* ptr_n);
	int m = (* ptr_m);
	int m_fake, m_buf;
	int nr1s = (* ptr_nr1s);
	int nr2s = (* ptr_nr2s);
	int nr3s = (* ptr_nr3s);
	int nrxxs = (* ptr_nrxxs);
	int ngms = (* ptr_ngms);
	int ngm = (* ptr_ngm);
	int lda = (* ptr_lda);

	int size_psic = nr1s * nr2s * nr3s;
	int dim_multiplepsic, n_singlepsic, n_multiplepsic, size_multiplepsic, i, j, v_size;;
	int array[3];

	cudaStream_t  vlocStreams[ MAX_QE_GPUS ];
	cublasHandle_t vlocHandles[ MAX_QE_GPUS ];

	size_t buffer_size = 0L;

	psic_D = (cufftDoubleComplex * ) dev_scratch_QE[0];

	// Fake padding
	if (m%2 == 0) {
		m_buf = m;
		m_fake = m_buf/2 ;
	} else {
		m_buf = m+1;
		m_fake = m_buf/2;
	}


	dim_multiplepsic = __NUM_FFT_MULTIPLAN__;
	n_multiplepsic = m_fake/dim_multiplepsic;
	n_singlepsic = m_fake%dim_multiplepsic;

	size_multiplepsic = size_psic * (dim_multiplepsic);

	buffer_size = size_multiplepsic * sizeof( cufftDoubleComplex ) + sizeof( cufftDoubleComplex ) * n * dim_multiplepsic * 2 + sizeof( int ) * ngms + sizeof( int ) * ngm + sizeof( int ) * n + sizeof( double ) * nrxxs;

	cudaSetDevice(qe_gpu_bonded[0]);

	if ( cublasCreate( &vlocHandles[ 0 ] ) != CUBLAS_STATUS_SUCCESS ) {
		printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cublasInit() for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	if( cudaStreamCreate( &vlocStreams[ 0 ] ) != cudaSuccess ) {
		printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** creating stream for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	if ( buffer_size > cuda_memory_allocated[0] ) {
		fprintf( stderr, "\nProblem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", buffer_size, cuda_memory_allocated[0] );
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	psi_D = psic_D + size_multiplepsic;
	hpsi_D = psi_D + ( n * 2 * dim_multiplepsic );
	v_D = (double * )  (hpsi_D + ( n * 2 * dim_multiplepsic ) );
	nls_D = (int   * ) (v_D + nrxxs);
	nlsm_D = (int *) nls_D + ngms;
	igk_D = nlsm_D + ngm;

	qecudaSafeCall( cudaMemset( dev_scratch_QE[0], 0, (size_t) cuda_memory_allocated[0] ) );

	qecudaSafeCall( cudaMemcpy( nls_D, nls,  sizeof( int ) * ngms, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( nlsm_D, nlsm,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( igk_D, igk,  sizeof( int ) * n, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( v_D, v,  sizeof( double ) * nrxxs, cudaMemcpyHostToDevice ) );

	array[0] = nr3s;
	array[1] = nr2s;
	array[2] = nr1s;

	v_size = nr1s * nr2s * nr3s;

	if ( n_multiplepsic > 0 ) {

		qecheck_cufft_call( cufftPlanMany( &p_global, 3, array, NULL, 1, 0, NULL,1,0,CUFFT_Z2Z,dim_multiplepsic ) );

		if( cufftSetStream(p_global,vlocStreams[ 0 ]) != CUFFT_SUCCESS ) {
			printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		for(j = 0; j< (m_fake-n_singlepsic); j+=dim_multiplepsic ) {

			qecudaSafeCall( cudaMemset( psi_D , 0, sizeof( cufftDoubleComplex ) * 2 * dim_multiplepsic *n ) );
			t_psi_D = ( cufftDoubleComplex * ) psi + 2*j*n;
			qecudaSafeCall( cudaMemcpy( psi_D, t_psi_D,  sizeof( cufftDoubleComplex ) * n * 2 * dim_multiplepsic, cudaMemcpyHostToDevice ) );

			qecudaSafeCall( cudaMemset( psic_D , 0, dim_multiplepsic * size_psic * sizeof( cufftDoubleComplex ) ) );

			blocksPerGrid = ( ( n * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
			for (i = 0; i < dim_multiplepsic; i++  )
			{
				kernel_init_psic<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( nls_D, nlsm_D, igk_D, (double *) psi_D, (double *) ( psic_D + i*size_psic), n, m, lda, i*2 );
				qecudaGetLastError("kernel launch failure");
			}

			qecheck_cufft_call( cufftExecZ2Z( p_global,  psic_D, psic_D , CUFFT_INVERSE ) );

			blocksPerGrid = ( (v_size * 2) + __CUDA_THREADPERBLOCK__  - 1) / __CUDA_THREADPERBLOCK__ ;
			for( i = 0; i < dim_multiplepsic; i++ ) {
				kernel_vec_prod<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (double *) (psic_D + ( i * size_psic )), (double *) v_D , v_size );
				qecudaGetLastError("kernel launch failure");
			}

			qecheck_cufft_call( cufftExecZ2Z( p_global, psic_D, psic_D, CUFFT_FORWARD ) );

			tscale = 1.0 / (double) ( size_psic );
			cublasZdscal(vlocHandles[ 0 ] , size_psic*dim_multiplepsic, &tscale, (cuDoubleComplex *) psic_D, 1);

			qecudaSafeCall( cudaMemset( hpsi_D, 0, sizeof( cufftDoubleComplex ) * 2* dim_multiplepsic *n ) );
			qecudaSafeCall( cudaMemcpy( hpsi_D, ( cufftDoubleComplex * ) hpsi + 2*j*n,  sizeof( cufftDoubleComplex ) * n * 2 * dim_multiplepsic, cudaMemcpyHostToDevice ) );

			blocksPerGrid = ( ( n * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
			for (i = 0; i < dim_multiplepsic; i++  )
			{
				kernel_save_hpsi<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( nls_D, nlsm_D, igk_D, (double *) hpsi_D, (double *) ( psic_D + i*size_psic), n, m, lda, i*2 );
				qecudaGetLastError("kernel launch failure");
			}

			t_hpsi = ( cufftDoubleComplex * ) hpsi + 2*j*n;
			qecudaSafeCall( cudaMemcpy( t_hpsi , hpsi_D, sizeof( cufftDoubleComplex ) * n * 2 * dim_multiplepsic, cudaMemcpyDeviceToHost ) );

		}
		qecheck_cufft_call( cufftDestroy(p_global) );
	}

	if (n_singlepsic > 0 ) {

		qecudaSafeCall( cudaMemset( psi_D , 0, sizeof( cufftDoubleComplex ) * 2 * dim_multiplepsic *n ) );
		t_psi_D = ( cufftDoubleComplex * ) psi + 2*dim_multiplepsic*n_multiplepsic*n;
		qecudaSafeCall( cudaMemcpy( psi_D, t_psi_D,  sizeof( cufftDoubleComplex ) * n * 2 * n_singlepsic, cudaMemcpyHostToDevice ) );

		qecudaSafeCall( cudaMemset( psic_D , 0, n_singlepsic * size_psic * sizeof( cufftDoubleComplex ) ) );

		qecheck_cufft_call( cufftPlanMany( &p_global, 3, array, NULL, 1, 0, NULL,1,0,CUFFT_Z2Z,n_singlepsic ) );

		if( cufftSetStream(p_global,vlocStreams[ 0 ]) != CUFFT_SUCCESS ) {
			printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		blocksPerGrid = ( ( n * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
		for (i = 0; i < n_singlepsic; i++  )
		{
			kernel_init_psic<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( nls_D, nlsm_D, igk_D, (double *) psi_D, (double *) ( psic_D + i*size_psic), n, m, lda, i*2 );
			qecudaGetLastError("kernel launch failure");
		}

		qecheck_cufft_call( cufftExecZ2Z( p_global,  psic_D, psic_D , CUFFT_INVERSE ) );

		blocksPerGrid = ( (v_size * 2) + __CUDA_THREADPERBLOCK__  - 1) / __CUDA_THREADPERBLOCK__ ;
		for( i = 0; i < n_singlepsic; i++ ){
			kernel_vec_prod<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (double *) (psic_D + (i * size_psic )), (double *) v_D , v_size );
			qecudaGetLastError("kernel launch failure");
		}

		tscale = 1.0 / (double) ( size_psic );
		cublasZdscal(vlocHandles[ 0 ] , size_psic * n_singlepsic, &tscale, (cuDoubleComplex *) psic_D, 1);

		qecheck_cufft_call( cufftExecZ2Z( p_global, psic_D, psic_D, CUFFT_FORWARD ) );

		qecudaSafeCall( cudaMemset( hpsi_D, 0, sizeof( cufftDoubleComplex ) * 2* dim_multiplepsic *n ) );
		qecudaSafeCall( cudaMemcpy( hpsi_D, ( cufftDoubleComplex * ) hpsi + 2*dim_multiplepsic*n_multiplepsic*n,  sizeof( cufftDoubleComplex ) * 2 * n_singlepsic * n, cudaMemcpyHostToDevice ) );

		blocksPerGrid = ( ( n * 2 ) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
		for (i = 0; i < n_singlepsic; i++  )
		{
			kernel_save_hpsi<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( nls_D, nlsm_D, igk_D, (double *) hpsi_D, (double *) ( psic_D + i*size_psic), n, m, lda,  i*2 );
			qecudaGetLastError("kernel launch failure");
		}

		qecudaSafeCall( cudaMemcpy( ( cufftDoubleComplex * ) hpsi + 2*dim_multiplepsic*n_multiplepsic*n, hpsi_D, sizeof( cufftDoubleComplex ) * 2 * n_singlepsic * n, cudaMemcpyDeviceToHost ) );

		qecheck_cufft_call( cufftDestroy(p_global) );
	}

	qecudaSafeCall( cudaMemset( dev_scratch_QE[0], 0, (size_t) cuda_memory_allocated[0] ) );

	cudaStreamDestroy( vlocStreams[ 0 ] );
	cublasDestroy( vlocHandles[ 0 ]);
}


extern "C" void vloc_psi_fft_cuda_(int * ptr_m, int * ptr_v_size, int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s, void * psic, double * v)
{
	cufftHandle p_global;

	cufftDoubleComplex * psic_D;
	cufftDoubleComplex * local_p_psic;
	double * v_D, tscale;
	int blocksPerGrid;

	int array[3];
	int m = (* ptr_m);
	int nr1s = (* ptr_nr1s);
	int nr2s = (* ptr_nr2s);
	int nr3s = (* ptr_nr3s);
	int v_size = (* ptr_v_size);
	local_p_psic = (cufftDoubleComplex *) psic;
	int size_psic =  v_size;

	int dim_multiplepsic, n_singlepsic, n_multiplepsic, i, j;

	cudaStream_t  vlocStreams[ MAX_QE_GPUS ];
	cublasHandle_t vlocHandles[ MAX_QE_GPUS ];

	size_t buffer_size = 0L;

	cudaSetDevice(qe_gpu_bonded[0]);

	if ( cublasCreate( &vlocHandles[ 0 ] ) != CUBLAS_STATUS_SUCCESS ) {
		printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cublasInit() for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	if( cudaStreamCreate( &vlocStreams[ 0 ] ) != cudaSuccess ) {
		printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** creating stream for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	qecudaSafeCall( cudaMemset( dev_scratch_QE[0], 0, (size_t) cuda_memory_allocated[0] ) );

	array[0] = nr3s;
	array[1] = nr2s;
	array[2] = nr1s;

	dim_multiplepsic = __NUM_FFT_MULTIPLAN__;
	n_multiplepsic = m/dim_multiplepsic;
	n_singlepsic = m%dim_multiplepsic;

	buffer_size = size_psic * dim_multiplepsic * sizeof( cufftDoubleComplex ) + sizeof( double ) * v_size ;

	if ( buffer_size > cuda_memory_allocated[0] ) {
		fprintf( stderr, "\nProblem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", buffer_size, cuda_memory_allocated[0] );
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

		qecheck_cufft_call( cufftPlanMany( &p_global, 3, array, NULL, 1, 0, NULL,1,0,CUFFT_Z2Z,dim_multiplepsic ) );

		if( cufftSetStream(p_global,vlocStreams[ 0 ]) != CUFFT_SUCCESS ) {
			printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		for(j = 0; j< (m-n_singlepsic); j+=dim_multiplepsic ) {

			qecudaSafeCall( cudaMemset( psic_D , 0, dim_multiplepsic * size_psic * sizeof( cufftDoubleComplex ) ) );

			qecudaSafeCall( cudaMemcpy( psic_D, (cufftDoubleComplex * ) (local_p_psic + j*size_psic) ,  sizeof( cufftDoubleComplex ) * ( size_psic * dim_multiplepsic )  , cudaMemcpyHostToDevice ) );

			qecheck_cufft_call( cufftExecZ2Z( p_global,  (cufftDoubleComplex *) psic_D , (cufftDoubleComplex *) psic_D, CUFFT_INVERSE ) );

			blocksPerGrid = ( (v_size * 2) + __CUDA_THREADPERBLOCK__  - 1) / __CUDA_THREADPERBLOCK__ ;
			for( i = 0; i < dim_multiplepsic; i++ ) {
				kernel_vec_prod<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (double *) (psic_D + ( i * size_psic )), (double *) v_D , v_size );
				qecudaGetLastError("kernel launch failure");
			}

			qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex *) psic_D , (cufftDoubleComplex *) psic_D, CUFFT_FORWARD ) );

			tscale = 1.0 / (double) ( size_psic );
			cublasZdscal(vlocHandles[ 0 ] , size_psic*dim_multiplepsic, &tscale, (cuDoubleComplex *) psic_D, 1);

			qecudaSafeCall( cudaMemcpy( (cufftDoubleComplex * ) (local_p_psic + j*size_psic) , psic_D, sizeof( cufftDoubleComplex ) * ( size_psic * dim_multiplepsic ), cudaMemcpyDeviceToHost ) );
		}

		qecheck_cufft_call( cufftDestroy(p_global) );
	}

	if (n_singlepsic > 0 ) {

		qecheck_cufft_call( cufftPlanMany( &p_global, 3, array, NULL, 1, 0, NULL,1,0,CUFFT_Z2Z,n_singlepsic ) );

		if( cufftSetStream(p_global,vlocStreams[ 0 ]) != CUFFT_SUCCESS ) {
			printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		qecudaSafeCall( cudaMemset( psic_D , 0, n_singlepsic * size_psic * sizeof( cufftDoubleComplex ) ) );

		qecudaSafeCall( cudaMemcpy( psic_D, (cufftDoubleComplex * ) (local_p_psic + n_multiplepsic*dim_multiplepsic*size_psic) ,  sizeof( cufftDoubleComplex ) * ( size_psic * n_singlepsic)  , cudaMemcpyHostToDevice ) );

		qecheck_cufft_call( cufftExecZ2Z( p_global,  (cufftDoubleComplex *) psic_D, (cufftDoubleComplex *) psic_D , CUFFT_INVERSE ) );

		blocksPerGrid = ( (v_size * 2) + __CUDA_THREADPERBLOCK__  - 1) / __CUDA_THREADPERBLOCK__ ;
		for( i = 0; i < n_singlepsic; i++ ){
			kernel_vec_prod<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (double *) (psic_D + (i * size_psic) ), (double *) v_D , v_size );
			qecudaGetLastError("kernel launch failure");
		}

		tscale = 1.0 / (double) ( size_psic );
		cublasZdscal(vlocHandles[ 0 ] , size_psic * n_singlepsic, &tscale, (cuDoubleComplex *) psic_D, 1);

		qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex *) psic_D, (cufftDoubleComplex *) psic_D, CUFFT_FORWARD ) );

		qecudaSafeCall( cudaMemcpy( (cufftDoubleComplex * ) (local_p_psic + n_multiplepsic*dim_multiplepsic*size_psic) , psic_D, sizeof( cufftDoubleComplex ) * ( size_psic * n_singlepsic), cudaMemcpyDeviceToHost ) );

		qecheck_cufft_call( cufftDestroy(p_global) );
	}

	qecudaSafeCall( cudaMemset( dev_scratch_QE[0], 0, (size_t) cuda_memory_allocated[0] ) );

	cudaStreamDestroy( vlocStreams[ 0 ] );
	cublasDestroy( vlocHandles[ 0 ]);
}

extern "C" void vloc_psi_singlefft_cuda_(int * ptr_m, int * ptr_v_size, int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s, void * psic, double * v)
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

	int i;

	cudaStream_t  vlocStreams[ MAX_QE_GPUS ];
	cublasHandle_t vlocHandles[ MAX_QE_GPUS ];

	size_t buffer_size = 0L;

	/* CUDA 3.x allows "only" one context switch */
	cudaSetDevice(qe_gpu_bonded[0]);

	if ( cublasCreate( &vlocHandles[ 0 ] ) != CUBLAS_STATUS_SUCCESS ) {
		printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cublasInit() for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	if( cudaStreamCreate( &vlocStreams[ 0 ] ) != cudaSuccess ) {
		printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** creating stream for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	qecudaSafeCall( cudaMemset( dev_scratch_QE[0], 0, (size_t) cuda_memory_allocated[0] ) );

	buffer_size = size_psic  * sizeof( cufftDoubleComplex ) + sizeof( double ) * v_size ;

	if ( buffer_size > cuda_memory_allocated[0] ) {
		fprintf( stderr, "\n[VLOC_PSI_GAMMA] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", buffer_size, cuda_memory_allocated[0] );
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}

	psic_D = (cufftDoubleComplex * ) dev_scratch_QE[0];
	v_D = (double * ) (psic_D + ( size_psic));

	qecudaSafeCall( cudaMemcpy( v_D, v,  sizeof( double ) * size_psic, cudaMemcpyHostToDevice ) );

	qecheck_cufft_call( cufftPlan3d( &p_global, nr3s, nr2s,  nr1s, CUFFT_Z2Z ) );

	if( cufftSetStream(p_global,vlocStreams[ 0 ]) != CUFFT_SUCCESS ) {
		printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
		fflush(stdout);
		exit( EXIT_FAILURE );
	}

	for(i = 0; i< m ; i++ ) {

		qecudaSafeCall( cudaMemset( psic_D , 0,  size_psic * sizeof( cufftDoubleComplex ) ) );

		qecudaSafeCall( cudaMemcpy( psic_D, (cufftDoubleComplex *) (local_p_psic + (i*size_psic)) ,  sizeof( cufftDoubleComplex ) * size_psic, cudaMemcpyHostToDevice ) );

		qecheck_cufft_call( cufftExecZ2Z( p_global,  (cufftDoubleComplex *) psic_D , (cufftDoubleComplex *) psic_D, CUFFT_INVERSE ) );

		blocksPerGrid = ( (v_size * 2) + __CUDA_THREADPERBLOCK__  - 1) / __CUDA_THREADPERBLOCK__ ;
		kernel_vec_prod<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (double *) psic_D, (double *) v_D , v_size );
		qecudaGetLastError("kernel launch failure");

		qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex *) psic_D , (cufftDoubleComplex *) psic_D, CUFFT_FORWARD ) );

		tscale = 1.0 / (double) ( size_psic );
		cublasZdscal(vlocHandles[ 0 ] , size_psic, &tscale, (cuDoubleComplex *) psic_D, 1);

		qecudaSafeCall( cudaMemcpy( (cufftDoubleComplex * ) (local_p_psic + i*size_psic) , psic_D, sizeof( cufftDoubleComplex ) * size_psic, cudaMemcpyDeviceToHost ) );
	}

	qecheck_cufft_call( cufftDestroy(p_global) );

	qecudaSafeCall( cudaMemset( dev_scratch_QE[0], 0, (size_t) cuda_memory_allocated[0] ) );

	cudaStreamDestroy( vlocStreams[ 0 ] );
	cublasDestroy( vlocHandles[ 0 ]);
}
